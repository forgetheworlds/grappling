"""Single-robot kinematics: footprint geometry, support margins, leg IK.

Everything here works on the bare drill model/data pair (36 qpos, 29 joints,
no name prefixes) and is used both to build the drill posture offline
(:mod:`drill.posture`) and to run the reference IK inside the control loop
(:mod:`drill.balance`).

Foot model: the G1 foot is four 5 mm contact spheres.  ``*_sole0..3`` sites
(mirrored from the collision geoms by :mod:`drill.scene`) are the footprint
corners; "flat on the mat" means all four centres at ``SOLE_REST_Z``.

Conventions: world is Z-up, ``+x`` is the robot's nominal heading (the stance
is built facing ``+x``), radians, SI.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

#: leg joint names in local order (index inside the 6-joint leg block)
LEG_JOINT_NAMES = ("hip_pitch", "hip_roll", "hip_yaw", "knee",
                   "ankle_pitch", "ankle_roll")
#: sides as used by every drill module (0 = left, 1 = right)
SIDES = ("left", "right")
#: sole sphere radius (m) and the centre height of a sole resting on the mat
SOLE_RADIUS = 0.005
SOLE_REST_Z = 0.0031
#: a sole centre below this counts as "on the mat" (1.2 cm = clearly loaded)
TOUCH_Z = 0.015
#: toe / heel sole-row indices in ``*_sole0..3`` (0,1 = heel row, 2,3 = toe row)
TOE_ROWS = (2, 3)
HEEL_ROWS = (0, 1)


def quat_yaw(q: np.ndarray) -> float:
    """Yaw (rad) of a (w, x, y, z) quaternion."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_quat(yaw: float, roll: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    """(w, x, y, z) quaternion for a ZYX base orientation."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


def z_rot(a: float) -> np.ndarray:
    """2-D rotation matrix (active rotation of a vector by ``a``)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def zyx_mat(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Rotation matrix Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


@dataclass(frozen=True)
class RobotIds:
    """Id table for the single-G1 drill model."""

    model: mujoco.MjModel
    pelvis: int
    torso: int
    base_qpos: int
    base_dof: int
    dofs: np.ndarray                      # 29 leg+body dofs (contiguous)
    leg_dofs: dict                        # side -> (6,) dof indices
    leg_qadr: dict                        # side -> (6,) qpos addresses
    leg_limits: dict                      # side -> (6, 2) joint ranges
    site: dict                            # name -> site id
    sole_sites: dict                      # side -> (4,) sole site ids
    foot_body: dict                       # side -> body id of ankle-roll link
    ctrl_lo: np.ndarray
    ctrl_hi: np.ndarray

    @classmethod
    def build(cls, model: mujoco.MjModel) -> "RobotIds":
        n = mujoco.mj_name2id
        obj = mujoco.mjtObj
        leg_dofs, leg_qadr, leg_limits = {}, {}, {}
        for side in SIDES:
            js = [n(model, obj.mjOBJ_JOINT, f"{side}_{j}_joint") for j in LEG_JOINT_NAMES]
            assert all(j > 0 for j in js), f"missing {side} leg joints"
            leg_dofs[side] = np.array([model.jnt_dofadr[j] for j in js])
            leg_qadr[side] = np.array([model.jnt_qposadr[j] for j in js])
            leg_limits[side] = np.array([model.jnt_range[j] for j in js])
        base_q = int(model.jnt_qposadr[n(model, obj.mjOBJ_JOINT, "floating_base_joint")])
        site = {mujoco.mj_id2name(model, obj.mjOBJ_SITE, i): i
                for i in range(model.nsite)}
        return cls(model=model,
                   pelvis=n(model, obj.mjOBJ_BODY, "pelvis"),
                   torso=n(model, obj.mjOBJ_BODY, "torso_link"),
                   base_qpos=base_q, base_dof=int(model.jnt_dofadr[0]),
                   dofs=np.arange(model.nv - 29, model.nv),
                   leg_dofs=leg_dofs, leg_qadr=leg_qadr, leg_limits=leg_limits,
                   site=site,
                   sole_sites={s: np.array([site[f"{s}_sole{k}"] for k in range(4)])
                               for s in SIDES},
                   foot_body={s: n(model, obj.mjOBJ_BODY, f"{s}_ankle_roll_link")
                              for s in SIDES},
                   ctrl_lo=model.actuator_ctrlrange[:, 0].copy(),
                   ctrl_hi=model.actuator_ctrlrange[:, 1].copy())

    # -- derived quantities ------------------------------------------------

    def base_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, float]:
        """(pelvis xyz, yaw) of the live state."""
        b = self.base_qpos
        return data.qpos[b:b + 3].copy(), quat_yaw(data.qpos[b + 3:b + 7])

    def com(self, data: mujoco.MjData) -> np.ndarray:
        return np.array(data.subtree_com[self.pelvis])

    def com_xy(self, data: mujoco.MjData) -> np.ndarray:
        return np.array(data.subtree_com[self.pelvis][:2])

    def torso_up(self, data: mujoco.MjData) -> np.ndarray:
        return data.xmat[self.torso].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])

    def torso_tilt_deg(self, data: mujoco.MjData) -> float:
        up = self.torso_up(data)
        return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))

    def pelvis_z(self, data: mujoco.MjData) -> float:
        return float(data.qpos[self.base_qpos + 2])

    def sole_points(self, data: mujoco.MjData, side: str) -> np.ndarray:
        """World-frame footprint corners (4, 3) of one foot."""
        return np.array(data.site_xpos[self.sole_sites[side]])

    def sole_local(self, side: str) -> np.ndarray:
        """Footprint corners in the foot (ankle-roll) frame (4, 3)."""
        return np.array(self.model.site_pos[self.sole_sites[side]])

    def sole_center(self, data: mujoco.MjData, side: str) -> np.ndarray:
        return self.sole_points(data, side).mean(axis=0)

    def sole_xy(self, data: mujoco.MjData, side: str) -> np.ndarray:
        """Hull of one footprint projected on the floor (n, 2)."""
        return hull2d(self.sole_points(data, side)[:, :2])

    def foot_wrench(self, data: mujoco.MjData, side: str) -> np.ndarray:
        """(6,) external wrench on one foot (world): [torque(3), force(3)].

        MuJoCo's 6-D layout is rotation *first* -- ``cfrc_ext[body, 3:6]`` is the
        force and ``cfrc_ext[body, 2]`` is a torque component, not the vertical
        load (a mix-up that made the unload gate read ~0 N for both feet).
        """
        return np.array(data.cfrc_ext[self.foot_body[side]], float)

    def foot_load(self, data: mujoco.MjData) -> np.ndarray:
        """(2,) vertical contact force on each foot (N, world frame)."""
        return np.array([float(self.foot_wrench(data, s)[5]) for s in SIDES])

    def foot_load_xy(self, data: mujoco.MjData) -> np.ndarray:
        """(2,2) horizontal contact force per foot (N) — slip diagnostics."""
        return np.array([self.foot_wrench(data, s)[3:5] for s in SIDES])

    def foot_clearance(self, data: mujoco.MjData) -> np.ndarray:
        """(2,) lowest footprint-corner height per foot, minus rest height (m)."""
        out = np.empty(2)
        for i, side in enumerate(SIDES):
            out[i] = float(self.sole_points(data, side)[:, 2].min()) - SOLE_REST_Z
        return out

    def foot_contact(self, data: mujoco.MjData) -> np.ndarray:
        """(2,) per-foot "some contact point is on the mat" flags."""
        out = np.empty(2, dtype=bool)
        for i, side in enumerate(SIDES):
            out[i] = bool(self.sole_points(data, side)[:, 2].min() < TOUCH_Z)
        return out

    def flat_contact(self, data: mujoco.MjData, tol: float = 0.012) -> np.ndarray:
        """(2,) per-foot "all four points within ``tol`` of the mat" flags."""
        out = np.empty(2, dtype=bool)
        for i, side in enumerate(SIDES):
            out[i] = bool(self.sole_points(data, side)[:, 2].max() - SOLE_REST_Z < tol)
        return out

    def contact_xy(self, data: mujoco.MjData, side: str) -> np.ndarray:
        """Hull of the footprint corners currently on the mat (n, 2); () if none."""
        p = self.sole_points(data, side)
        low = p[p[:, 2] < TOUCH_Z]
        return hull2d(low[:, :2]) if len(low) >= 3 else np.zeros((0, 2))

    def support_polygon(self, data: mujoco.MjData, loaded: np.ndarray) -> np.ndarray:
        """Hull of the *whole footprint* of every loaded foot (n, 2).

        The full footprint, not just the corners below the touch threshold: a
        foot rolling 1 cm onto its edge still supports the body, and a
        touching-points-only hull collapses to a line in that moment (measured:
        the CoM "margin" jumped to -0.18 m without the robot moving).
        """
        pts = [self.sole_xy(data, s) for i, s in enumerate(SIDES) if loaded[i]]
        pts = [p for p in pts if len(p)]
        return hull2d(np.vstack(pts)) if pts else np.zeros((0, 2))

    def com_margin(self, data: mujoco.MjData, loaded: np.ndarray) -> float:
        """Signed CoM margin (m) vs the support polygon of the loaded feet."""
        return polygon_margin(self.com_xy(data), self.support_polygon(data, loaded))

    def local_xy(self, pos_xy: np.ndarray, data: mujoco.MjData) -> np.ndarray:
        """World xy -> robot heading frame xy (origin = pelvis xy)."""
        base, yaw = self.base_pose(data)
        d = np.asarray(pos_xy, float) - base[:2]
        c, s = np.cos(-yaw), np.sin(-yaw)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def foot_targets(ids: RobotIds, side: str, origin_xy: np.ndarray, yaw: float,
                 sole_z: float = SOLE_REST_Z, pitch: float = 0.0,
                 roll: float = 0.0) -> np.ndarray:
    """World targets (4, 3) for a foot placed with its frame origin at ``origin_xy``.

    ``origin_xy`` is the ankle-roll link origin projected on the floor (the
    handle used by every drill plan; the footprint centre is 3.5 cm further
    forward).  ``sole_z`` is the height of the *lowest* footprint point, so
    ``sole_z = SOLE_REST_Z`` means flat on the mat; ``pitch``>0 raises the toe
    row (heel down), ``roll``>0 tilts the sole towards +y.
    """
    local = ids.sole_local(side)
    R = zyx_mat(yaw, pitch, roll)
    pts = local @ R.T
    base = np.array([origin_xy[0], origin_xy[1], float(sole_z) - pts[:, 2].min()])
    return base + pts


# --------------------------------------------------------------------- 2-D geometry
def hull2d(points: np.ndarray) -> np.ndarray:
    """Convex hull (monotone chain) of (n, 2); counter-clockwise vertices."""
    p = np.unique(np.asarray(points, float).reshape(-1, 2), axis=0)
    if len(p) <= 2:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) <= 1e-12:
                    out.pop()
                else:
                    break
            out.append(q)
        return out

    lower, upper = half(p), half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def polygon_margin(point_xy: np.ndarray, poly: np.ndarray) -> float:
    """Signed distance from ``point_xy`` to a hull boundary (m).

    Positive = inside the (convex, CCW) support polygon, negative = outside.
    A hull with fewer than 3 points has no interior: the margin is the negated
    distance to it, so callers never read a one-point support as stable.
    """
    p = np.asarray(point_xy, float)
    if len(poly) == 0:
        return -np.inf
    if len(poly) < 3:
        return -min(float(np.linalg.norm(p - q)) for q in poly)
    inside = True
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        if e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]) < 0.0:
            inside = False
            break
    d = np.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        t = 0.0 if e @ e < 1e-12 else float(np.clip((p - a) @ e / (e @ e), 0.0, 1.0))
        d = min(d, float(np.linalg.norm(p - (a + t * e))))
    return d if inside else -d


# -------------------------------------------------------------------------- leg IK
def _ik_rows(model, data, ids_sites, dofs, targets, jac) -> tuple:
    """Stacked site-Jacobian rows and residual for the current scratch state."""
    rows, des = [], []
    for j, sid in enumerate(ids_sites):
        err = np.asarray(targets[j], float) - data.site_xpos[int(sid)]
        mujoco.mj_jacSite(model, data, jac, None, int(sid))
        rows.append(jac[:, dofs].copy())
        des.append(err)
    return np.vstack(rows), np.concatenate(des)


def _solve_chain(model: mujoco.MjModel, data: mujoco.MjData, sites, dofs,
                 qadr, targets, q_init, iters: int, damp: float,
                 step_clip: float, tol: float, limits: np.ndarray,
                 seed_knee: int | None = None, knee_min: float = 0.10) -> np.ndarray:
    """Damped Gauss-Newton with a monotone line search.

    Three guards, each for a failure that was measured on this robot:
    * the seed's knee is lifted off full extension (``knee_min``) -- at a
      straight leg the knee Jacobian vanishes and the solver walks into the
      hyperextension limit instead of flexing;
    * the step is clipped;
    * every step must *decrease* the residual, otherwise it is halved (and the
      loop stops when nothing improves, e.g. an unreachable target) -- without
      this the iterate diverged to joint limits on unreachable targets.
    """
    q = np.asarray(q_init, float).copy()
    if seed_knee is not None:
        q[seed_knee] = max(q[seed_knee], knee_min)
    lo = np.asarray(limits, float)[:, 0] + 1e-3
    hi = np.asarray(limits, float)[:, 1] - 1e-3
    jac = np.zeros((3, model.nv))
    n = len(qadr)

    def residual(qv):
        data.qpos[qadr] = qv
        scratch_kinematics(model, data)
        return np.concatenate([np.asarray(t, float) - data.site_xpos[int(sid)]
                               for sid, t in zip(sites, targets)])

    q = np.clip(q, lo, hi)
    b = residual(q)
    for _ in range(int(iters)):
        norm = float(np.linalg.norm(b))
        if norm < tol:
            break
        A, _b2 = _ik_rows(model, data, sites, dofs, targets, jac)
        dq = np.linalg.solve(A.T @ A + damp * np.eye(n), A.T @ b)
        improved = False
        for scale in (1.0, 0.5, 0.25, 0.125):
            q_try = np.clip(q + np.clip(scale * dq, -step_clip, step_clip), lo, hi)
            b_try = residual(q_try)
            if float(np.linalg.norm(b_try)) < norm * (1.0 - 1e-9):
                q, b = q_try, b_try
                improved = True
                break
        if not improved:
            q = np.clip(q, lo, hi)
            break
    data.qpos[qadr] = q
    scratch_kinematics(model, data)
    return q


def leg_ik(model: mujoco.MjModel, data: mujoco.MjData, ids: RobotIds, side: str,
           targets: np.ndarray, q_leg: np.ndarray, iters: int = 4,
           damp: float = 2e-3, step_clip: float = 0.25,
           tol: float = 4e-4) -> np.ndarray:
    """Place a foot's four footprint points at world ``targets`` (4, 3).

    See :func:`_solve_chain` for the numerical guards.  The caller owns the
    scratch qpos: this writes the six leg qpos addresses and calls
    ``scratch_kinematics`` (no dynamics side effects).
    """
    tgt = np.asarray(targets, float)
    return _solve_chain(model, data, ids.sole_sites[side], ids.leg_dofs[side],
                        ids.leg_qadr[side], [tgt[j] for j in range(len(tgt))],
                        q_leg, iters, damp, step_clip, tol,
                        ids.leg_limits[side], seed_knee=3)


def ik_error(model: mujoco.MjModel, data: mujoco.MjData, ids: RobotIds, side: str,
             targets: np.ndarray, q_leg: np.ndarray) -> float:
    """Worst footprint-point error (m) of a leg-IK solution."""
    data.qpos[ids.leg_qadr[side]] = q_leg
    scratch_kinematics(model, data)
    tgt = np.asarray(targets, float)
    return max(float(np.linalg.norm(tgt[j] - data.site_xpos[int(sid)]))
               for j, sid in enumerate(ids.sole_sites[side]))


def scratch_kinematics(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Update positions *and* the compact dofs after a qpos write.

    ``mj_jacSite`` reads ``data.cdof``, which ``mj_kinematics`` alone does NOT
    refresh — a Jacobian taken right after it is silently wrong (verified
    against finite differences: max element error 0.25 here).  ``mj_comPos``
    refreshes it and is much cheaper than a full ``mj_forward`` (the IK runs
    twice per control tick).
    """
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


def chain_ik(model: mujoco.MjModel, data: mujoco.MjData, dofs: np.ndarray,
             qadr: np.ndarray, limits: np.ndarray, site_targets: list,
             q_init: np.ndarray, iters: int = 4, damp: float = 2e-3,
             step_clip: float = 0.25, tol: float = 4e-4) -> np.ndarray:
    """IK for a joint chain driving sites to world targets (see ``_solve_chain``).

    ``site_targets``: list of ``(site_id, target_xyz)``.
    """
    sites = [int(sid) for sid, _ in site_targets]
    targets = [np.asarray(t, float) for _, t in site_targets]
    return _solve_chain(model, data, sites, np.asarray(dofs), np.asarray(qadr),
                        targets, q_init, iters, damp, step_clip, tol,
                        np.asarray(limits))


if __name__ == "__main__":                              # self-check
    from . import scene as scene_mod

    m = scene_mod.load_model()
    ids = RobotIds.build(m)
    d = mujoco.MjData(m)
    d.qpos[:] = scene_mod.keyframe(m, "stand")
    mujoco.mj_forward(m, d)
    print("stand keyframe:")
    print("  clearance (m):", np.round(ids.foot_clearance(d), 4),
          "flat:", ids.flat_contact(d), "contact:", ids.foot_contact(d))
    print("  com:", np.round(ids.com(d), 3), "margin:", round(ids.com_margin(d, np.ones(2, bool)), 4))
    for side in SIDES:
        c = ids.sole_center(d, side)
        info = ids.sole_xy(d, side)
        print(f"  {side}: centre={np.round(c, 3)} footprint_xy={np.round(info, 3).tolist()}")
        tgt = foot_targets(ids, side, c[:2], 0.0)
        q0 = d.qpos[ids.leg_qadr[side]].copy()
        q1 = leg_ik(m, d, ids, side, tgt, q0 + 0.15, iters=10)
        print(f"    IK err={ik_error(m, d, ids, side, tgt, q1):.2e} max|dq|={np.abs(q1-q0).max():.2f} rad")
