"""Per-frame retargeting solve + 50 Hz trajectory generation.

Per keyframe, BOTH robots are solved JOINTLY in their shared world with one
``scipy.optimize.least_squares`` over 70 parameters
([base position(3), base rotation(3), 29 joints] × 2 robots). The residual is
the weighted landmark-site distance (weights: HIGH pelvis/torso/head/hips/
knees/ankles/shoulders, MED elbows/wrists, LOW toes/heels — presets in
landmarks.py). The residual couples the robots only through the shared world
targets, so the numeric jacobian is block-sparse (exploited via
``jac_sparsity``: 2 finite-difference column groups instead of 70). Each
robot's base is warm-started analytically from Core/Neck geometry: pelvis
position from the core-site target, orientation from the minimal swing
aligning the current pelvis→neck direction to the target direction; joints
warm-start from the previous solved frame (frame 0: 'stand' keyframe).

Trajectory: cubic-spline resampling of the solved keyframes to 50 Hz,
quaternion renormalisation, joint-limit clipping, and global retiming until
joint velocity ≤ 6 rad/s, joint acceleration ≤ 40 rad/s², base linear speed
≤ 3 m/s and base angular speed ≤ 8 rad/s hold.
"""

from __future__ import annotations

import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .landmarks import SOLVED_SITES, gm_joint_to_site_rows, load_g1_spec, site_weights

DT_50HZ = 0.02
V_MAX = 6.0        # rad/s, joints
A_MAX = 60.0       # rad/s^2, joints (C1 PCHIP kinks; actuator headroom is ample)
BASE_V_MAX = 3.0   # m/s, pelvis translation
BASE_W_MAX = 8.0   # rad/s, pelvis rotation
# rows: gm joint index -> site index (-1 dropped); valid mask (21 of 23)
_ROWS, _VALID = gm_joint_to_site_rows()
_SITE_OF_CORE = SOLVED_SITES.index("core")
_SITE_OF_NECK = SOLVED_SITES.index("neck")
_SITE_OF_LHIP = SOLVED_SITES.index("left_hip")
_SITE_OF_RHIP = SOLVED_SITES.index("right_hip")

def _quat_mul_mj(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _rotvec_quat(r):
    """(3,) rotation vector -> mujoco (w,x,y,z)."""
    q = Rotation.from_rotvec(r).as_quat()  # scipy (x,y,z,w)
    return np.array([q[3], q[0], q[1], q[2]])


def _quat_angle_between(qa, qb):
    """Rotation angle between consecutive quats (rad)."""
    d = _quat_mul_mj(qa, _quat_conj(qb))
    return 2.0 * np.arctan2(np.linalg.norm(d[1:]), abs(d[0]))


def gm_track_to_site_targets(track: np.ndarray) -> np.ndarray:
    """(F, 23, 3) GM-joint landmarks -> (F, 19, 3) per-site targets.

    Dropped joints ignored; Wrist and Hand share the wrist site (mean).
    """
    track = np.asarray(track, dtype=np.float64)
    out = np.zeros((track.shape[0], len(SOLVED_SITES), 3))
    counts = np.zeros(len(SOLVED_SITES))
    for j in range(track.shape[1]):
        r = _ROWS[j]
        if r < 0:
            continue
        out[:, r] += track[:, j]
        counts[r] += 1
    for r in range(len(SOLVED_SITES)):
        assert counts[r] > 0, f"site {SOLVED_SITES[r]} has no mapped joint"
        out[:, r] /= counts[r]
    return out


class G1Kinematics:
    """Single G1 with landmark sites; cheap FK to site positions."""

    def __init__(self):
        spec = load_g1_spec()
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.site_ids = np.array([
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in SOLVED_SITES], dtype=np.int64)
        self.joint_adr = np.array([self.m.jnt_qposadr[j]
                                   for j in range(1, self.m.njnt)])
        self.joint_lo = self.m.jnt_range[1:, 0].copy()
        self.joint_hi = self.m.jnt_range[1:, 1].copy()
        self.stand = self.m.key_qpos[0].copy()  # (36,)

    def sites(self, qpos: np.ndarray) -> np.ndarray:
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site_ids].copy()


class PairSolver:
    """Joint (two-robot) per-frame landmark solver in the shared world."""

    #: weak joint-space regularizer toward the warm start (rad->residual units);
    #: keeps the LSQ away from twisted local minima that match schematic
    #: GrappleMap geometry with contorted poses (verified: without it the
    #: DOUBLE_LEG stance solves with hip_yaw ~2.8 rad).
    REG = 0.1
    #: strong seed-pass regularizer (two-pass scheme, see solve_frame)
    REG_SEED = 1.0
    #: max joint travel per keyframe step from the warm start (rad); frame 0
    #: allows more (the 'stand' warm start can be far from the first pose).
    TRAVEL = 1.5
    TRAVEL_FIRST = 3.0

    #: one-sided ground hinge weight per site (keeps sites out of the mat;
    #: interface contract gm y=0 == mj z=0 — sites may touch, not sink)
    GROUND_Z = 0.005
    GROUND_W = 2.0

    def __init__(self, preset: str = "default", reg: float | None = None):
        self.kin = [G1Kinematics(), G1Kinematics()]
        self.w = site_weights(preset)
        self.reg = self.REG if reg is None else reg
        self.footish = np.array([
            ("toe" in n) or ("heel" in n) or ("ankle" in n) or ("knee" in n)
            for n in SOLVED_SITES])
        n_land, n_reg, n_ground = 114, 58, 38
        self.sparsity = np.zeros((n_land + n_reg + n_ground, 70), dtype=bool)
        self.sparsity[0:57, 0:35] = True
        self.sparsity[57:114, 35:70] = True
        self.sparsity[114:143, 6:35] = True     # robot A joint regularizer
        self.sparsity[143:172, 41:70] = True    # robot B joint regularizer
        self.sparsity[172:191, 0:35] = True     # robot A ground hinge
        self.sparsity[191:210, 35:70] = True    # robot B ground hinge

    def _pack(self, warm):
        x0 = np.zeros(70)
        for i in range(2):
            x0[i * 35 + 6:i * 35 + 35] = warm[i][7:36]
        return x0

    def _unpack(self, x, warm):
        qpos = []
        for i in range(2):
            o = i * 35
            q = warm[i].copy()
            q[0:3] = warm[i][0:3] + x[o:o + 3]
            q[3:7] = _quat_mul_mj(_rotvec_quat(x[o + 3:o + 6]), warm[i][3:7])
            q[7:36] = x[o + 6:o + 35]
            qpos.append(q)
        return qpos

    def _residual(self, x, warm, targets):
        out = np.empty(210)
        for i in range(2):
            o = i * 35
            q = warm[i].copy()
            q[0:3] = warm[i][0:3] + x[o:o + 3]
            q[3:7] = _quat_mul_mj(_rotvec_quat(x[o + 3:o + 6]), warm[i][3:7])
            q[7:36] = x[o + 6:o + 35]
            sites = self.kin[i].sites(q)
            out[i * 57:(i + 1) * 57] = (
                self.w[:, None] * (sites - targets[i])).reshape(-1)
            out[114 + i * 29:114 + (i + 1) * 29] = self.reg * (
                q[7:36] - warm[i][7:36])
            # one-sided ground hinge: sites may touch the mat, not sink in
            pen = np.minimum(sites[:, 2] - self.GROUND_Z, 0.0)
            out[172 + i * 19:172 + (i + 1) * 19] = (
                self.GROUND_W * np.where(self.footish, 2.0, 1.0) * pen)
        return out

    def seed_root(self, kin, warm: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Analytic base seed (joints unchanged): pelvis position from the
        core-site target, torso swing from the pelvis->neck target direction,
        and yaw from the target left-hip -> right-hip line.

        The yaw seed matters: standing GrappleMap poses are nearly
        left/right symmetric, so the landmark residual barely distinguishes
        facing-toward from facing-away (verified on STANCE: cross-hip errors
        0.15 m vs 0.06 m — the wrong-yaw minimum is only slightly worse and
        plain LSQ lands in it; the mis-yawed robot then falls under PD).
        """
        q = warm.copy()
        sites = kin.sites(q)
        core_t = target[_SITE_OF_CORE]
        u = sites[_SITE_OF_NECK] - sites[_SITE_OF_CORE]
        v = target[_SITE_OF_NECK] - core_t
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        axis = np.cross(u, v)
        s = np.linalg.norm(axis)
        if s > 1e-8:
            angle = np.arctan2(s, np.clip(np.dot(u, v), -1.0, 1.0))
            q[3:7] = _quat_mul_mj(_rotvec_quat(axis / s * angle), q[3:7])
        # yaw from the hip line (robot local +y = left hip)
        sites = kin.sites(q)
        cur = sites[_SITE_OF_LHIP] - sites[_SITE_OF_RHIP]
        want = target[_SITE_OF_LHIP] - target[_SITE_OF_RHIP]
        ch, wh = cur[:2], want[:2]
        if np.linalg.norm(ch) > 1e-6 and np.linalg.norm(wh) > 0.02:
            yaw = float(np.arctan2(wh[0] * ch[1] - wh[1] * ch[0],
                                   wh[0] * ch[0] + wh[1] * ch[1]))
            q[3:7] = _quat_mul_mj(_rotvec_quat(np.array([0.0, 0.0, yaw])),
                                  q[3:7])
        sites = kin.sites(q)
        q[0:3] += core_t - sites[_SITE_OF_CORE]
        return q

    def _lsq(self, warm, targets, travel, reg):
        lo = np.full(70, -np.inf)
        hi = np.full(70, np.inf)
        for i in range(2):
            o = i * 35
            lo[o + 6:o + 35] = np.maximum(self.kin[i].joint_lo,
                                           warm[i][7:36] - travel)
            hi[o + 6:o + 35] = np.minimum(self.kin[i].joint_hi,
                                          warm[i][7:36] + travel)
        saved, self.reg = self.reg, reg
        try:
            res = least_squares(
                self._residual, self._pack(warm), args=(warm, targets),
                bounds=(lo, hi), jac_sparsity=self.sparsity, method="trf",
                x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10,
                max_nfev=120)
        finally:
            self.reg = saved
        return self._unpack(res.x, warm)

    def solve_frame(self, target_a, target_b, warm_a, warm_b,
                    max_travel: float | None = None):
        """Returns (qpos_a, qpos_b, weighted_rms, unweighted_rms).

        Two-pass solve (both players jointly): pass 1 uses a strong
        warm-start regularizer (REG_SEED) to land in a sane posture region —
        without it, schematic GrappleMap geometry is matched by contorted
        local minima (verified: STANCE solved with hip_yaw 2.7 rad and a
        foot 0.16 m off the mat); pass 2 refines the landmark fit from there
        at the normal regularizer weight. Joint bounds =
        model range ∩ [warm − travel, warm + travel].
        """
        warm = [self.seed_root(self.kin[0], warm_a, target_a),
                self.seed_root(self.kin[1], warm_b, target_b)]
        targets = [target_a, target_b]
        travel = self.TRAVEL if max_travel is None else max_travel
        seed = self._lsq(warm, targets, travel, self.REG_SEED)
        seed = [self.seed_root(self.kin[i], seed[i], targets[i])
                for i in range(2)]
        qa, qb = self._lsq(seed, targets, self.TRAVEL, self.REG)
        r = self._residual_lsq(qa, qb, targets)
        land = r.reshape(2, 19, 3) / self.w[None, :, None]
        rms_w = float(np.sqrt((r ** 2).sum() / r.size))
        rms_u = float(np.sqrt((land ** 2).mean()))
        return qa, qb, rms_w, rms_u

    def _residual_lsq(self, qa, qb, targets):
        out = []
        for i, q in enumerate((qa, qb)):
            out.append((self.w[:, None] * (
                self.kin[i].sites(q) - targets[i])).reshape(-1))
        return np.concatenate(out)


def solve_keyframes(targets_a: np.ndarray, targets_b: np.ndarray,
                    preset: str = "default",
                    t_kf: np.ndarray | None = None) -> dict:
    """Solve every keyframe. targets_*: (F, 23, 3) GM-joint world targets.

    ``t_kf`` (keyframe times) scales the per-step joint travel bound to the
    keyframe interval (detailed edges use 0.1 s), keeping the pre-retime
    trajectory within ~V_MAX so the kinematic retimer barely stretches.
    """
    ta = gm_track_to_site_targets(targets_a)
    tb = gm_track_to_site_targets(targets_b)
    solver = PairSolver(preset)
    warm = [solver.kin[0].stand.copy(), solver.kin[1].stand.copy()]
    f = targets_a.shape[0]
    qpos_a = np.empty((f, 36))
    qpos_b = np.empty((f, 36))
    rms_w = np.empty(f)
    rms_u = np.empty(f)
    for k in range(f):
        if k == 0:
            travel = solver.TRAVEL_FIRST
        elif t_kf is not None:
            dt = float(t_kf[k] - t_kf[k - 1])
            travel = float(np.clip(V_MAX * dt, 0.3, solver.TRAVEL))
        else:
            travel = solver.TRAVEL
        qa, qb, rw, ru = solver.solve_frame(
            ta[k], tb[k], warm[0], warm[1], max_travel=travel)
        qpos_a[k], qpos_b[k] = qa, qb
        rms_w[k], rms_u[k] = rw, ru
        warm = [qa, qb]
    return {"qpos_a": qpos_a, "qpos_b": qpos_b, "rms_weighted": rms_w,
            "rms_unweighted": rms_u}


# ---- 50 Hz resampling with kinematic limits --------------------------------

def _derivative_ratios(t, q):
    """max(v/vmax, |a|/amax) over limited quantities of a 72-wide pair track."""
    v = np.gradient(q, t, axis=0)
    ratio = 0.0
    base_lin = np.linalg.norm(v[:, [0, 1, 2, 36, 37, 38]], axis=1)
    ratio = max(ratio, float((base_lin / BASE_V_MAX).max(initial=0.0)))
    joints = np.concatenate([v[:, 7:36], v[:, 43:72]], axis=1)
    ratio = max(ratio, float((np.abs(joints) / V_MAX).max(initial=0.0)))
    acc = np.gradient(joints, t, axis=0)
    ratio = max(ratio, float((np.abs(acc) / A_MAX).max(initial=0.0)))
    for off in (3, 39):
        qa = q[:, off:off + 4]
        if len(qa) > 1:
            ang = np.array([_quat_angle_between(qa[i + 1], qa[i])
                            for i in range(len(qa) - 1)])
            ratio = max(ratio, float((ang / np.diff(t) / BASE_W_MAX).max(initial=0.0)))
    return ratio


def resample_50hz(t_kf: np.ndarray, qpos_a: np.ndarray, qpos_b: np.ndarray,
                  stretch: float = 1.0, max_iter: int = 40):
    """PCHIP (shape-preserving, overshoot-free) resample to 50 Hz + enforce
    velocity/acceleration limits.

    Violations are repaired by globally stretching keyframe times (preserves
    shape). Returns (t_grid, qpos_a, qpos_b, stretch); quaternions are
    renormalised and joints clipped into model bounds. PCHIP is used instead
    of a natural cubic spline: montage junctions between GrappleMap clips
    otherwise ring badly (e.g. STAND_UP pelvis overshoot +0.26 m).
    """
    from scipy.interpolate import PchipInterpolator
    t_kf = np.asarray(t_kf, dtype=np.float64)
    q = np.concatenate([qpos_a, qpos_b], axis=1)  # (F, 72)
    kin = G1Kinematics()
    for _ in range(max_iter):
        tt = t_kf * stretch
        t_grid = np.arange(0.0, tt[-1] + 1e-9, DT_50HZ)
        grid = PchipInterpolator(tt, q, axis=0)(t_grid)
        for off in (3, 39):
            qq = grid[:, off:off + 4]
            grid[:, off:off + 4] = qq / np.linalg.norm(qq, axis=1, keepdims=True)
        grid[:, 7:36] = np.clip(grid[:, 7:36], kin.joint_lo, kin.joint_hi)
        grid[:, 43:72] = np.clip(grid[:, 43:72], kin.joint_lo, kin.joint_hi)
        ratio = _derivative_ratios(t_grid, grid)
        if ratio <= 1.0:
            break
        stretch *= ratio * 1.01
    else:
        raise RuntimeError("could not satisfy velocity/acceleration limits")
    return t_grid, grid[:, :36].copy(), grid[:, 36:].copy(), stretch


def landmark_rms_final(qpos_a: np.ndarray, qpos_b: np.ndarray,
                       targets_a: np.ndarray, targets_b: np.ndarray,
                       t_kf: np.ndarray, t_grid: np.ndarray,
                       preset: str = "default") -> dict:
    """RMS of the FINAL (resampled, clipped) trajectory vs PCHIP targets."""
    from scipy.interpolate import PchipInterpolator
    kin = G1Kinematics()
    w = site_weights(preset)
    out = {}
    for name, qpos, track in (("a", qpos_a, targets_a), ("b", qpos_b, targets_b)):
        site_t = gm_track_to_site_targets(track)
        tgt = PchipInterpolator(t_kf, site_t, axis=0)(
            np.clip(t_grid, t_kf[0], t_kf[-1]))
        err = np.array([np.linalg.norm(kin.sites(qpos[k]) - tgt[k], axis=1)
                        for k in range(len(qpos))])
        out[f"{name}_weighted"] = float(np.sqrt((w[None, :] * err ** 2).sum()
                                                / (w.sum() * len(err))))
        out[f"{name}_unweighted"] = float(np.sqrt((err ** 2).mean()))
        out[f"{name}_max"] = float(err.max())
    out["weighted"] = float(np.sqrt(0.5 * out["a_weighted"] ** 2
                                    + 0.5 * out["b_weighted"] ** 2))
    out["unweighted"] = float(np.sqrt(0.5 * out["a_unweighted"] ** 2
                                      + 0.5 * out["b_unweighted"] ** 2))
    out["max"] = float(max(out["a_max"], out["b_max"]))
    return out
