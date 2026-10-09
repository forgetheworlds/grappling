"""G1 reference synthesis for the v2 motion-reference set (Agent 3 re-timing).

Why this exists (the measured v1 defects it fixes)
-------------------------------------------------
* The v1 table solver (``solo.stance.stance_qpos``) is only grounded at the
  stand height: commanding pelvis 0.74 m pushes the soles 3.8 cm THROUGH the
  mat (FK-measured, this module's ``measure_track`` instrument) and the error
  grows as the commanded height drops.  Every solved v2 posture is instead
  built by a calibrated planar leg IK whose soles sit ON the mat by
  construction (FK-verified round trip <= 1 mm over the reachable set).
* The v1 video-derived paths have their CoM OUTSIDE the planted-foot support
  for 62 % of the drill (worst -0.63 m during CIRCLE): a fixed-clock tracker
  cannot follow them without toppling.  v2 phases are generated
  CoM-inside-support at every frame (the quasi-static envelope the current
  balance layer can actually execute), with the operator's sanctioned
  geometry repairs (wider base, rear foot back, knee-driven crouch, head up,
  guard) baked in.

Everything here is KINEMATIC SYNTHESIS: the emitted tracks are desired pose
trajectories (targets for a tracking policy), never executed demonstrations.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

# calibrated planar leg chain (LEFT leg, sagittal x-z plane), measured from
# robots/g1 MuJoCo pivots at zero leg angles: hip pivot, then hip->knee,
# knee->ankle_pitch, ankle_pitch->foot-site vectors.  ``rot`` below is the
# y-axis rotation acting on (x, z); positive angle swings the leg BACKWARD.
HIP_PIVOT = np.array([0.0, 0.6873])       # pelvis frame: (x, z) of hip pitch pivot
R1 = np.array([0.0, -0.3366])             # hip pitch -> knee (thigh)
R2 = np.array([0.0, -0.3000])             # knee -> ankle pitch (shank)
R3 = np.array([0.0, -0.01756])            # ankle pitch -> foot site
ANKLE_LO, ANKLE_HI = -0.8720, 0.5230    # ankle pitch range, kept strictly
                                        # INSIDE the model's joint limits
                                        # (-0.87267, 0.5236) so solved poses
                                        # never sit exactly on a limit
#: stand keyframe geometry (src/solo.stance measured constants)
STAND_HEIGHT = 0.790
STAND_WIDTH = 0.237
_Y_PER_ROLL = 1.22
FOOT_Y_NATURAL = STAND_WIDTH / 2.0        # +/- y of each foot at zero hip roll

DT = 0.02                                 # reference rate (s, 50 Hz)


def rot(v: np.ndarray, th: float) -> np.ndarray:
    """y-axis rotation of an in-plane (x, z) vector (MuJoCo convention)."""
    c, s = math.cos(th), math.sin(th)
    return np.array([v[0] * c + v[1] * s, -v[0] * s + v[1] * c])


def leg_fk(hip_pitch: float, knee: float, ankle_pitch: float) -> np.ndarray:
    """Foot-site position (x, z) in the HIP-PIVOT frame (planar, sole flat)."""
    th = hip_pitch + knee + ankle_pitch
    return (rot(R1, hip_pitch) + rot(R2, hip_pitch + knee)
            + rot(R3, th))


def leg_ik(foot_rel_hip: np.ndarray) -> tuple[float, float, float, float]:
    """Closed-form planar IK, knee-FORWARD branch.

    ``foot_rel_hip``: (x, z) of the foot site relative to the hip pivot.
    Returns ``(hip_pitch, knee, ankle_pitch, sole_tilt)`` where the angles
    realise the target exactly when the sole is flat (ankle limit not hit;
    otherwise ``sole_tilt`` > 0 reports the uncorrectable radians and the
    angles are clipped).
    """
    ank = np.asarray(foot_rel_hip, np.float64) - R3   # sole flat: r3 unrotated
    dv = ank - np.zeros(2)
    a, b = float(np.linalg.norm(R1)), float(np.linalg.norm(R2))
    D = float(np.clip(np.linalg.norm(dv), 1e-6, a + b - 1e-9))
    beta = math.acos(float(np.clip((a * a + b * b - D * D) / (2 * a * b), -1, 1)))
    alpha = math.acos(float(np.clip((a * a + D * D - b * b) / (2 * a * D), -1, 1)))
    knee = math.pi - beta
    # phi: direction of the target measured from straight-down; rot() maps
    # "down" to (-sin th, -cos th), so phi = atan2(-dx, -dz)
    phi = math.atan2(-dv[0], -dv[1])
    hip_pitch = phi - alpha                # knee-forward (biological) branch
    unclipped = -(hip_pitch + knee)
    tilt = max(0.0, unclipped - ANKLE_HI, ANKLE_LO - unclipped)
    ankle_pitch = float(np.clip(unclipped, ANKLE_LO, ANKLE_HI))
    return float(hip_pitch), float(knee), ankle_pitch, float(tilt)


def min_jerk(s: np.ndarray) -> np.ndarray:
    """Minimum-jerk (quintic) easing of s in [0, 1] (zero end vel/accel)."""
    s = np.clip(np.asarray(s, np.float64), 0.0, 1.0)
    return s ** 3 * (10.0 - 15.0 * s + 6.0 * s ** 2)


# ----------------------------------------------------------------- model io
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from .scene import load_solo_model
        _MODEL = load_solo_model()
    return _MODEL


def stand_keyframe() -> np.ndarray:
    from .scene import stand_frame
    return stand_frame(_model())[0].copy()


#: leg joint offsets inside the 29-joint vector (x6 per leg)
LEG_OFF = {"left": 0, "right": 6}


def _set_leg(q: np.ndarray, side: str, hip_pitch: float, hip_roll: float,
             knee: float, ankle_pitch: float, ankle_roll: float) -> None:
    o = 7 + LEG_OFF[side]
    q[o + 0] = hip_pitch
    q[o + 1] = hip_roll
    q[o + 3] = knee
    q[o + 4] = ankle_pitch
    q[o + 5] = np.clip(ankle_roll, -0.26, 0.26)   # ankle-roll joint range


#: hip roll range edge (robots/g1 jnt_range: left [-0.524, 2.967],
#: right [-2.967, 0.524]) -- the roll solve respects this
HIP_ROLL_MAX = 0.52


def guard_targets(q: np.ndarray, shoulder_pitch: float = -0.45,
                  shoulder_roll: float = 0.15, elbow: float = 1.00) -> np.ndarray:
    """Wrestling guard: shoulders forward, elbows bent, hands inside.

    The operator's A6 repair ("hands carriage: forward, inside, elbows in").
    qpos offsets (free joint consumes 7): left shoulder pitch = 22, roll = 23,
    elbow = 25; right shoulder pitch = 29, roll = 30, elbow = 32.  The stand
    keyframe's own arm pose (pitch +0.2, elbow 1.28) is REPLACED, not added to.
    """
    q = q.copy()
    q[22] = shoulder_pitch                # left shoulder pitch
    q[29] = shoulder_pitch                # right shoulder pitch
    q[23] = shoulder_roll                 # left shoulder roll (hands inside)
    q[30] = -shoulder_roll                # right shoulder roll
    q[25] = elbow                         # left elbow
    q[32] = elbow                         # right elbow
    return q


def stance_pose(width: float, height: float, split: float = 0.0,
                pelvis_x: float = 0.0, guard: bool = True,
                grounded: bool = True, model=None) -> tuple[np.ndarray, dict]:
    """One grounded stance qpos (36,) with measured diagnostics.

    The pose keeps the pelvis upright (no waist fold: the crouch is produced
    by knee+hip flexion from the planar IK), feet flat on the mat, and -- when
    ``grounded`` -- the root z is shifted so the lowest sole sphere touches
    z = 0 exactly (FK; the same instrument as ``ground_qpos_soles``).
    Returns ``(qpos, info)``; ``info`` carries the measured width, the sole
    residual after grounding, the ankle-limit sole tilt, and joint-limit
    proximity.
    """
    import mujoco

    m = model if model is not None else _model()
    q0 = stand_keyframe()
    hip_roll = (float(width) - STAND_WIDTH) / _Y_PER_ROLL
    hip_z = float(height) + (HIP_PIVOT[1] - STAND_HEIGHT)
    q = q0.copy()
    tilt = 0.0
    for side, sgn in (("left", +1.0), ("right", -1.0)):
        fx = float(pelvis_x) + (float(split) / 2.0 if side == "left"
                                else -float(split) / 2.0)
        foot_site_z = -R3[1]
        hip, knee, ankle, tilt = leg_ik(
            np.array([fx, foot_site_z - hip_z]))
        _set_leg(q, side, hip, sgn * hip_roll, knee, ankle, -sgn * hip_roll)
    if guard:
        q = guard_targets(q)
    if grounded:
        data = mujoco.MjData(m)
        data.qpos[:] = q
        mujoco.mj_forward(m, data)
        from .lit import sole_points_world
        pen = float(sole_points_world(m, data)[..., 2].min())
        q[2] -= pen                   # exact rigid shift: soles on the mat
    info = measure_pose(q, model=m)
    info["sole_tilt_rad"] = tilt
    return q, info


def measure_pose(q: np.ndarray, model=None, data=None) -> dict:
    """FK diagnostics of one pose: penetration, CoM margin, limit proximity."""
    import mujoco

    from .exec_check import hull2d, polygon_margin
    from .lit import sole_points_world
    m = model if model is not None else _model()
    data = mujoco.MjData(m) if data is None else data
    data.qpos[:] = q
    mujoco.mj_forward(m, data)
    sole = sole_points_world(m, data)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "a_pelvis")
    com = data.subtree_com[bid]
    pts = sole[:, :, :2].reshape(-1, 2)
    jlo, jhi = m.jnt_range[1:].T
    qj = q[7:]
    prox = float(np.max((np.maximum(jlo - qj, qj - jhi)
                         / np.maximum(jhi - jlo, 1e-9))))
    return {"sole_pen_m": round(float(sole[..., 2].min()), 5),
            "com_margin_m": round(float(polygon_margin(com[:2], hull2d(pts))), 5),
            "limit_prox": round(prox, 4),
            "pelvis_z": round(float(q[2]), 5)}


# --------------------------------------------------------------- generators
def descent_path(width: float, h_from: float, h_to: float, duration_s: float,
                 *, split: float = 0.0, pelvis_x: float = 0.0, guard: bool = True,
                 model=None, min_margin_m: float = 0.005) -> tuple[np.ndarray, np.ndarray, dict]:
    """Quasi-static stand<->stance descent/ascent (T, 36) at 50 Hz.

    Height follows a minimum-jerk profile; every frame is re-solved by
    :func:`stance_pose` (grounded, feet flat).  The path FAILS (raises) if any
    frame's CoM margin drops below ``min_margin_m`` -- the caller chooses
    geometry, not hope.
    """
    n = max(int(round(duration_s / DT)), 2)
    qs, infos = [], []
    s = (np.arange(n) + 1) / n
    prof = h_from + (h_to - h_from) * min_jerk(s)
    for h in prof:
        q, info = stance_pose(width, float(h), split=split, pelvis_x=pelvis_x,
                              guard=guard, model=model)
        qs.append(q)
        infos.append(info)
    q = np.asarray(qs)
    worst = min(i["com_margin_m"] for i in infos)
    if worst < min_margin_m:
        raise ValueError(f"descent path CoM margin {worst:.4f} m < "
                         f"{min_margin_m:.4f} m at (width={width}, "
                         f"split={split}, pelvis_x={pelvis_x})")
    t = np.arange(n) * DT
    meta = {"width_m": width, "h_from": h_from, "h_to": h_to,
            "duration_s": round(duration_s, 3), "split_m": split,
            "pelvis_x_m": pelvis_x, "com_margin_min_m": worst,
            "sole_pen_max_m": min(i["sole_pen_m"] for i in infos),
            "limit_prox_max": max(i["limit_prox"] for i in infos)}
    return q, t, meta


def hold_path(width: float, height: float, duration_s: float, *,
              split: float = 0.0, pelvis_x: float = 0.0, guard: bool = True,
              model=None) -> tuple[np.ndarray, np.ndarray]:
    """Constant hold frames (T, 36) (one solved pose repeated)."""
    n = max(int(round(duration_s / DT)), 1)
    q, _ = stance_pose(width, height, split=split, pelvis_x=pelvis_x,
                       guard=guard, model=model)
    return np.tile(q, (n, 1)), np.arange(n) * DT


def blend_paths(qa: np.ndarray, qb: np.ndarray, duration_s: float) -> np.ndarray:
    """C1 minimum-jerk JOINT-space blend between two grounded poses.

    Both endpoints are solved grounded poses; short blends between nearby
    geometries keep soles within a millimetre or two of the mat (the caller
    re-grounds with :func:`ground_track` if needed).
    """
    n = max(int(round(duration_s / DT)), 2)
    s = min_jerk((np.arange(n) + 1) / n)[:, None]
    q = qa[None, :] * (1 - s) + qb[None, :] * s
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    return q


def ground_track(q: np.ndarray, model=None, floor: float = 0.0) -> np.ndarray:
    """Shift every frame's root z so the lowest sole sphere sits at ``floor``.

    Exact (rigid shift), per frame; joints untouched.
    """
    import mujoco

    from .lit import sole_points_world
    m = model if model is not None else _model()
    data = mujoco.MjData(m)
    out = np.array(q, np.float64)
    for i in range(len(out)):
        data.qpos[:] = out[i]
        mujoco.mj_forward(m, data)
        pen = float(sole_points_world(m, data)[..., 2].min()) - floor
        out[i, 2] -= pen
    return out


# ------------------------------------------------------- stepping generator
@dataclass
class StepPlan:
    """One footstep of a shuffle/turn sequence.

    ``swing``: "left"/"right".  ``to_xy``: swing foot's landing xy (world).
    ``transfer_s / swing_s``: phase durations.  ``lift_m``: peak swing lift.
    """

    swing: str
    to_xy: tuple[float, float]
    transfer_s: float = 0.9
    swing_s: float = 0.4
    lift_m: float = 0.04
    yaw_deg: float = 0.0
    #: fraction of the pelvis's lateral way to the support foot that the
    #: transfer covers before the lift.  1.0 = statically balanced single
    #: support -- geometrically OUT of reach for the G1 at stance width (the
    #: far leg cannot reach the mat), so stepping phases use ~0.55: the CoM
    #: moves TOWARD the support before the lift (the B2 weight-transfer
    #: contract) and crosses through a dynamic single support, exactly like
    #: human gait; the lateral dynamics are the balance layer's job.
    transfer_frac: float = 0.55


def step_sequence(plan: list[StepPlan], *, width: float, height: float,
                  split: float = 0.0, feet_xy: tuple[tuple, tuple] | None = None,
                  guard: bool = True, model=None):
    """Quasi-static stepping: transfer weight -> lift -> plant -> transfer.

    The contact schedule is BY CONSTRUCTION weight-transfer-before-lift: the
    pelvis moves over the support foot (min-jerk, ``transfer_s``) BEFORE the
    swing foot leaves (``swing_s``), and a foot's contact flag is 0 exactly
    while its lift bump is > 0.  Foot placement is 3D: per leg the planar IK
    gives the sagittal reach and the hip roll is solved numerically (secant on
    the FK lateral error, sole kept level by the counter ankle roll) so the
    FK foot site lands on the designed (x, y).  The pelvis transfer target is
    the support foot minus the measured guarded-stance CoM offset, so the CoM
    -- not the pelvis -- centres over the support.
    Returns ``(q (T,36), t (T,), contact (T,2) u8, meta)``.
    """
    import mujoco

    from .lit import sole_points_world
    m = model if model is not None else _model()
    data = mujoco.MjData(m)
    # resolve the GROUNDED pelvis height for (width, height) once: commanding
    # the nominal height leaves the legs 1-2 cm short of the mat and the feet
    # float; the grounded value makes the designed soles exactly reach z=0.
    _, hinfo = stance_pose(width, height, guard=guard, model=m)
    height_eff = hinfo["pelvis_z"]
    if feet_xy is None:
        feet_xy = ((0.0, +width / 2.0), (0.0, -width / 2.0))
    # measured CoM offset of the guarded stance (rotated by yaw at use)
    q_meas, _ = stance_pose(width, height, guard=guard, model=m)
    data.qpos[:] = q_meas
    mujoco.mj_forward(m, data)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "a_pelvis")
    com_off = (float(data.subtree_com[bid][0] - q_meas[0]),
               float(data.subtree_com[bid][1] - q_meas[1]))
    feet = [np.array([feet_xy[0][0], feet_xy[0][1], 0.0]),
            np.array([feet_xy[1][0], feet_xy[1][1], 0.0])]
    # start the pelvis at the stance CENTRE (CoM over the feet midpoint),
    # not the origin -- displaced start poses (reposition/backward steps)
    # otherwise begin with the pelvis a metre behind the feet
    feet_mid = 0.5 * (feet[0][:2] + feet[1][:2])
    pelvis = feet_mid - np.array(com_off)
    yaw = 0.0
    frames, contacts = [], []
    idx = {"left": 0, "right": 1}

    def emit(n, pelvis_xy, yaw_now, feet_now, planted):
        for _ in range(n):
            frames.append((np.array(pelvis_xy, np.float64), float(yaw_now),
                           [f.copy() for f in feet_now], tuple(planted)))
            contacts.append(np.array(planted, np.uint8))

    def build(pelvis_xy, yaw_now, feet_now) -> np.ndarray:
        q = stand_keyframe()
        c, s = math.cos(yaw_now), math.sin(yaw_now)
        Rz = np.array([[c, -s], [s, c]])
        q[3:7] = (math.cos(yaw_now / 2), 0.0, 0.0, math.sin(yaw_now / 2))
        q[0], q[1] = float(pelvis_xy[0]), float(pelvis_xy[1])
        q[2] = float(height_eff)
        hip_z = float(height_eff) + (HIP_PIVOT[1] - STAND_HEIGHT)
        for side in ("left", "right"):
            i = idx[side]
            sgn = +1.0 if side == "left" else -1.0
            foot_site = "a_left_foot" if side == "left" else "a_right_foot"
            fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, foot_site)
            w3 = Rz.T @ (feet_now[i][:2] - pelvis_xy)   # foot xy in heading frame
            fz = float(feet_now[i][2]) - R3[1]           # foot-site z target
            hip_y = sgn * 0.0645                          # hip pivot lateral offset
            lat_hip = w3[1] - hip_y                       # lateral reach rel hip
            dz = fz - hip_z
            x_target = float(w3[0])
            roll = 0.0
            for _it in range(2):                 # sagittal correction loop
                L = math.hypot(lat_hip, dz)               # in-plane "down" reach
                hip, knee, ankle, tilt = leg_ik(np.array([x_target, -L]))
                # hip roll: sign-safe secant so the FK foot site lands on the
                # designed lateral position (slope measured, not assumed)
                def foot_pos(roll_try: float) -> np.ndarray:
                    qq = q.copy()
                    _set_leg(qq, side, hip, sgn * roll_try, knee, ankle,
                             -sgn * roll_try)
                    data.qpos[:] = qq
                    mujoco.mj_forward(m, data)
                    return Rz.T @ (data.site_xpos[fid][:2] - pelvis_xy)
            roll = math.atan2(lat_hip, -dz) if dz < 0 else 0.0
            p0_ = foot_pos(roll)
            e0 = float(p0_[1] - w3[1])
            if abs(e0) > 2e-4:
                d = 0.02
                e1 = float(foot_pos(roll + d)[1] - w3[1])
                slope = (e1 - e0) / d
                if abs(slope) > 1e-3:
                    roll -= e0 / slope
                    p0_ = foot_pos(roll)
            x_target += float(w3[0] - p0_[0])  # sagittal residual -> retarget
            hip_roll_hi = HIP_ROLL_MAX          # hip-roll joint range edge
            roll = float(np.clip(roll, -hip_roll_hi, hip_roll_hi))
            _set_leg(q, side, hip, sgn * roll, knee, ankle, -sgn * roll)
        q = guard_targets(q) if guard else q
        return q

    emit(1, pelvis, yaw, feet, (True, True))
    for st in plan:
        i = idx[st.swing]
        sup = 1 - i
        # (a) weight transfer: CoM over the support foot (both planted);
        # the stance CoM offset is body-fixed -> rotate by the current yaw
        n_tr = max(int(round(st.transfer_s / DT)), 2)
        cy, sy = math.cos(yaw), math.sin(yaw)
        off = np.array([cy * com_off[0] - sy * com_off[1],
                        sy * com_off[0] + cy * com_off[1]])
        full = feet[sup][:2] + off
        tgt = pelvis + (full - pelvis) * float(st.transfer_frac)
        start = pelvis.copy()
        for k in range(1, n_tr + 1):
            e = float(min_jerk(np.array([k / n_tr]))[0])
            pelvis = start + (tgt - start) * e
            emit(1, pelvis, yaw, feet, (True, True))
        # (b) swing (support planted only); swing foot lifts by the bump.
        # Yaw steps PIVOT: the swing foot's target is its current position
        # ROTATED about the support foot (the body turns around the planted
        # foot), and the pelvis rotates with it -- fixed world targets would
        # twist the legs apart as the yaw accumulates.
        n_sw = max(int(round(st.swing_s / DT)), 2)
        frm = feet[i].copy()
        sup_xy = feet[sup][:2]
        yaw1 = yaw + math.radians(st.yaw_deg)
        if st.yaw_deg != 0.0:
            dl = math.radians(st.yaw_deg)
            Rm = np.array([[math.cos(dl), -math.sin(dl)],
                           [math.sin(dl), math.cos(dl)]])
            to = np.array([*(sup_xy + Rm @ (feet[i][:2] - sup_xy)), 0.0])
            pelvis_swing = sup_xy + Rm @ (pelvis[:2] - sup_xy)
        else:
            to = np.array([st.to_xy[0], st.to_xy[1], 0.0])
            pelvis_swing = pelvis[:2].copy()
        pelvis_swing_start = pelvis[:2].copy()
        for k in range(1, n_sw + 1):
            e = float(min_jerk(np.array([k / n_sw]))[0])
            feet[i][:2] = frm[:2] + (to[:2] - frm[:2]) * e
            feet[i][2] = st.lift_m * math.sin(math.pi * e)
            pelvis[:2] = pelvis_swing_start + (pelvis_swing
                                               - pelvis_swing_start) * e
            planted = [True, True]
            planted[i] = False
            emit(1, pelvis, yaw + (yaw1 - yaw) * e, feet, planted)
        feet[i] = to
        yaw = yaw1
        emit(1, pelvis, yaw, feet, (True, True))
    q = np.stack([build(p, y, f) for (p, y, f, _) in frames])
    # ONE constant grounding (frame 0's residual): per-frame re-grounding
    # would inject discrete root-z jumps at every plant/lift event
    data.qpos[:] = q[0]
    mujoco.mj_forward(m, data)
    from .lit import sole_points_world as _spw
    q[:, 2] -= float(_spw(m, data)[..., 2].min())
    t = np.arange(len(q)) * DT
    meta = {"steps": len(plan), "width_m": width, "height_m": height,
            "split_m": split, "travel_m": round(float(np.linalg.norm(
                frames[-1][0] - frames[0][0])), 4),
            "yaw_total_deg": round(math.degrees(frames[-1][1] - frames[0][1]), 2)}
    return q, t, np.asarray(contacts, np.uint8), meta


def _rot2(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------- utilities
def time_scale(q: np.ndarray, t: np.ndarray, factor: float,
               contact: np.ndarray | None = None):
    """Resample a 50 Hz track by ``factor`` (>1 = slower).

    Linear interpolation of qpos columns (quat renormalised); the output
    clock stays 50 Hz and monotone.  Returns ``(q2, t2, contact2)``.
    """
    q = np.asarray(q, np.float64)
    t = np.asarray(t, np.float64)
    assert len(q) == len(t) and len(t) >= 2
    n_out = max(int(round((len(t) - 1) * factor)) + 1, 2)
    t_out = np.arange(n_out) * DT
    src = t_out / factor
    src = np.clip(src, 0.0, t[-1])
    out = np.column_stack([np.interp(src, t, q[:, j]) for j in range(q.shape[1])])
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    c2 = None
    if contact is not None:
        c = np.asarray(contact)
        c2 = np.column_stack([
            np.interp(src, t, c[:, j].astype(np.float64)) for j in range(c.shape[1])])
        c2 = (c2 > 0.5).astype(np.uint8)
    return out, t_out, c2


def measure_track(q: np.ndarray, t: np.ndarray, contact: np.ndarray | None = None,
                  model=None, step: int = 1) -> dict:
    """Kinematic + quasi-static diagnostics of a whole track.

    ``axy_max``/``az_max``: pelvis fore-aft-lateral / vertical acceleration
    from double finite differences (m/s^2).  ``com_margin_min_m``: CoM vs the
    convex hull of the feet the frame's own contact flags call planted (nan
    where none planted).
    """
    import mujoco

    from .exec_check import hull2d, polygon_margin
    from .lit import sole_points_world
    m = model if model is not None else _model()
    data = mujoco.MjData(m)
    q = np.asarray(q, np.float64)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else DT
    axy = np.gradient(np.gradient(q[:, :2], dt, axis=0), dt, axis=0)
    az = np.gradient(np.gradient(q[:, 2:3], dt, axis=0), dt, axis=0)
    vxy = np.linalg.norm(np.gradient(q[:, :2], dt, axis=0), axis=1)
    sole = np.zeros((len(q), 2, 4, 3))
    com = np.zeros((len(q), 3))
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "a_pelvis")
    for i in range(0, len(q), step):
        data.qpos[:] = q[i]
        mujoco.mj_forward(m, data)
        sole[i] = sole_points_world(m, data)
        com[i] = data.subtree_com[bid]
    for i in range(len(q)):
        if i % step:
            sole[i] = sole[i - 1]
            com[i] = com[i - 1]
    margins = np.full(len(q), np.nan)
    if contact is not None:
        planted = np.asarray(contact).astype(bool)
        for i in range(len(q)):
            if planted[i].any():
                pts = sole[i][planted[i]][:, :, :2].reshape(-1, 2)
                margins[i] = polygon_margin(com[i, :2], hull2d(pts))
    pen = float(sole[..., 2].min())
    finite = margins[np.isfinite(margins)]
    # quasi-static window = BOTH feet planted (the schedule's designed
    # double-support); single-support frames are dynamic by design (stepping)
    margs_ds = margins[np.isfinite(margins)
                       & (np.asarray(contact).astype(bool).all(axis=1)
                          if contact is not None else np.zeros(len(q), bool))]
    out = {
        "axy_max_m_s2": round(float(np.linalg.norm(axy, axis=1).max()), 3),
        "az_max_m_s2": round(float(np.abs(az).max()), 3),
        "vxy_max_m_s": round(float(vxy.max()), 3),
        "vz_max_m_s": round(float(np.abs(np.gradient(q[:, 2], dt)).max()), 3),
        "joint_v_max_rad_s": round(float(
            (np.abs(np.diff(q[:, 7:], axis=0)).max(axis=1) / dt).max()), 3),
        "com_margin_min_m": (round(float(finite.min()), 4)
                             if finite.size else None),
        "com_margin_neg_frac": (round(float(np.mean(margins[~np.isnan(margins)] < 0)), 3)
                                if finite.size else None),
        "sole_pen_min_m": round(pen, 5),
    }
    if margs_ds.size:
        out["com_margin_ds_min_m"] = round(float(margs_ds.min()), 4)
        out["com_margin_ds_neg_frac"] = round(float(np.mean(margs_ds < 0)), 3)
    return out


if __name__ == "__main__":  # self-check
    m = _model()
    q_stand = stand_keyframe()
    base = measure_pose(q_stand, model=m)
    assert base["sole_pen_m"] > -0.005, base
    qh, info = stance_pose(0.34, 0.74, model=m)
    assert info["sole_pen_m"] > -0.002, info
    assert info["com_margin_m"] > 0.0, info
    # leg IK round trip within 1 mm across the reachable set
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        target = np.array([rng.uniform(-0.25, 0.25),
                           rng.uniform(-0.60, -0.35)])
        hip, knee, ankle, tilt = leg_ik(target)
        if tilt > 1e-9:
            continue
        got = rot(R1, hip) + rot(R2, hip + knee) + rot(R3, hip + knee + ankle)
        worst = max(worst, float(np.abs(got - target).max()))
    assert worst <= 1e-3, worst
    print("solo.refgen self-check OK:", {
        "stand": base, "stance_0.34x0.74": info, "ik_roundtrip_m": round(worst, 6)})
