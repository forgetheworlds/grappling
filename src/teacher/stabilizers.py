"""Balance channels for the phase-3 teacher (measured, physically valid).

Why this module exists
----------------------
The robots are position-servo humanoids with a free-floating base and two foot
contacts.  A world-frame task on anything that lives ON the base (CoM, pelvis
site) has **zero authority** through the joint actuators in a pinned-base
Jacobian sense: moving a leg joint moves the *foot*, not the base.  The
channels that do move the base (and therefore the CoM in the world) are:

  * direct base-rotation channels - a symmetric ankle_pitch / knee / hip_pitch
    offset rotates the whole body about the stance feet (ankle strategy).  The
    authority is *currents-based*: measured from ``both_stand`` over a 0.25 s
    burst at two amplitudes, it saturates quickly (ankle +0.10 rad moves the
    base 5.3 cm, +0.20 rad only 5.4 cm) because the centre of pressure hits
    the foot edge;
  * foot-target channel - a differential leg IK on the foot sites (the
    pinned-base foot Jacobian).  Commanding the *world* target of a planted
    foot backwards by ``b`` moves the base forward by ~``b`` (measured: a
    -0.10 m world-x target shift on STANCE moved the base +0.054 m).  This is
    a *static* map (the base settles at the new pose; it does not ramp), which
    makes it the only channel able to hold a *sustained* posture shift (the
    STANCE crouch needs the body shifted ~0.1 m forward of where the
    schematic reference puts it).

Signs and magnitudes below come from the identification probe in
reports/2026-10-08/teacher.md (table "channel identification").  Every
channel is expressed in the robot's own heading frame, so robot A (+x) and
robot B (-x) share the same numeric joints and signs.

Authority is defined as d(base position) / d(joint offset) measured over one
0.25 s control burst from a stable stand.  A channel with authority ``g``
moves the base by ``g*dq``, so to move the CoM by ``-e`` the offset is
``dq = -e/g``.
"""

from __future__ import annotations

import mujoco
import numpy as np

#: measured authority, m per rad, both legs driven with the same offset,
#: robot-local frame (x = forward, y = left), 0.25 s burst from both_stand
#: (probe reports/2026-10-08/teacher.md; values are the small-amplitude
#: secant 0.10 rad / 0.20 rad).
AUTH = {
    "ankle_pitch": np.array([-0.53, 0.00]),
    "knee": np.array([-0.38, 0.00]),
    "hip_pitch": np.array([-0.09, 0.00]),
    "hip_roll": np.array([0.00, -0.46]),
    "ankle_roll": np.array([0.00, -0.02]),
}
#: how much of each axis' demand each channel takes (must be handled jointly:
#: one joint alone is resisted by the other leg, so channels are grouped).
#: ankle_roll is excluded from the roll demand: its measured authority is
#: ~20x weaker than hip_roll and its sign is the same, so its 0.15 share
#: would demand a 7 rad offset.
PITCH_SPLIT = {"ankle_pitch": 0.55, "knee": 0.30, "hip_pitch": 0.15}
#: measured yaw authority of the hip_yaw pair (rad of base yaw per rad, both
#: legs driven): positive hip_yaw turns the base clockwise seen from above.
YAW_AUTH = 0.35
ROLL_SPLIT = {"hip_roll": 1.00}
#: local joint index blocks inside the 29-joint vector
LEG_JOINTS = {"left": {"hip_pitch": 0, "hip_roll": 1, "knee": 3,
                       "ankle_pitch": 4, "ankle_roll": 5},
              "right": {"hip_pitch": 6, "hip_roll": 7, "knee": 9,
                        "ankle_pitch": 10, "ankle_roll": 11}}

G = 9.81


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def channel_offsets(e_loc: np.ndarray, contact: np.ndarray, gains: dict) -> np.ndarray:
    """29-joint direct balance offsets for a robot-local error ``e_loc`` (m).

    ``contact`` (2,) = foot planted flags (left, right). ``gains``: k_pitch,
    k_roll (fraction of the authority-matched demand) and the clamps.
    Offsets use the measured authority signs: every pitch channel moves the
    base in -x_local for a positive offset, hip_roll moves it in -y_local.
    """
    dq = np.zeros(29)
    if not contact.any():
        return dq
    k_pitch = gains.get("k_pitch", 0.0)
    k_roll = gains.get("k_roll", 0.0)
    for side, idx in LEG_JOINTS.items():
        if not contact[0 if side == "left" else 1]:
            continue
        for name, share in PITCH_SPLIT.items():
            g = AUTH[name][0]
            dq[idx[name]] += -share * k_pitch * e_loc[0] / g
        for name, share in ROLL_SPLIT.items():
            g = AUTH[name][1]
            dq[idx[name]] += -share * k_roll * e_loc[1] / g
    return dq


def yaw_offsets(e_yaw: float, contact: np.ndarray, gain: float,
                lim: float = 0.25) -> np.ndarray:
    """29-joint heading offsets: both hip_yaw joints rotate the base in yaw.

    Measured authority (both_stand, both legs, 0.25 s burst): +0.10 rad of
    hip_yaw changes the base yaw by -0.035 rad (both legs; the feet spin on
    the mat, whose torsional friction is 0.005).  The correction is
    ``dq = e_yaw / |g|`` so gain 1.0 cancels the error over one burst.
    """
    dq = np.zeros(29)
    if not contact.any() or gain == 0.0:
        return dq
    v = float(np.clip(gain * e_yaw / YAW_AUTH, -lim, lim))
    dq[2] += v
    dq[8] += v
    return dq


def capture_error(com: np.ndarray, v_com: np.ndarray, support: np.ndarray,
                  kd: float, clamp: float | None = None) -> np.ndarray:
    """World-frame capture-point error vs the support centre (m).

    ``clamp=None`` returns the unclipped error (the posture governor's
    activation input; the clipped value saturates at sqrt(2)*clamp and cannot
    reach emergency thresholds).
    """
    h = max(com[2] - 0.05, 0.30)
    xi = com[:2] + kd * np.sqrt(h / G) * v_com
    err = xi - support
    return err if clamp is None else np.clip(err, -clamp, clamp)


def foot_rows(model, data, ctx, frame: int, dz: float, params: dict,
              jac_buf, out_rows: list, out_des: list) -> None:
    """Foot-target rows: planted-foot anchoring + sole flattening + height.

    For every foot the reference plants (any of its toe/heel sites touching
    in the reference frame) BOTH sole sites get full 3-D rows:

      xy = reference site position (world)
      z  = (1-k_flat) * reference z + k_flat * rest sole height + dz

    Requiring both sites is what flattens the foot onto the mat: the
    retargeted STANCE keeps its heel sites 6-7 cm up, and driving only the
    reference-touching site (the toe) left the foot toe-down with a
    one-point support.  Feet the reference lifts get no rows.
    """
    for foot in range(2):
        if not ctx.ref_touch[frame, foot].any():
            continue
        for s in range(2):
            sid = ctx.site[ctx.foot_sites[foot][s]]
            cur = data.site_xpos[sid]
            tgt = ctx.ref_feet[frame, foot, s].copy()
            rz = ctx.rest_z[foot][s]
            tgt[2] = (1 - params["k_flat"]) * tgt[2] + params["k_flat"] * rz
            tgt[2] += dz
            err = np.clip(tgt - cur, -params["anchor_clamp"], params["anchor_clamp"])
            mujoco.mj_jacSite(model, data, jac_buf, None, sid)
            out_rows.append(jac_buf[:, ctx.dofs].copy())
            out_des.append(params["k_anchor"] * err)


def upright_rows(model, data, ctx, frame: int, params: dict, jac_buf,
                 jacr_buf, out_rows: list, out_des: list) -> None:
    """Torso-upright rows (waist channel): blend the reference up-vector to +z."""
    if params["k_up"] <= 0:
        return
    up = data.xmat[ctx.torso].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
    tgt = ((1 - params["upright_blend"]) * ctx.ref_up[frame]
           + params["upright_blend"] * np.array([0.0, 0.0, 1.0]))
    n = np.linalg.norm(tgt)
    if n < 1e-6:
        return
    tgt = tgt / n
    err = np.cross(up, tgt)
    mujoco.mj_jacBodyCom(model, data, jac_buf, jacr_buf, ctx.torso)
    out_rows.append(jacr_buf[:2, ctx.dofs].copy())
    out_des.append(params["k_up"] * err[:2])


def solve_offsets(rows: list, des: list, reg: float) -> np.ndarray:
    """Regularized weighted least squares for the 29 joint offsets."""
    if not rows:
        return np.zeros(29)
    A = np.vstack(rows)
    b = np.concatenate(des)
    A = np.vstack([A, np.sqrt(reg) * np.eye(29)])
    b = np.append(b, np.zeros(29))
    dq, *_ = np.linalg.lstsq(A, b, rcond=None)
    return dq


def support_center(data, ctx, frame: int) -> np.ndarray:
    """Support centre from the sites currently touching the mat (2,).

    ``ctx.support_foot`` (0/1, set by a stepping scheduler) restricts the
    support to that foot while it touches — the support-mode decision that
    makes a weight transfer to one leg explicit instead of implicit in the
    midpoint of two feet.  Falls back to the reference support when nothing
    is touching.
    """
    if getattr(ctx, "support_foot", None) is not None:
        foot = int(ctx.support_foot)
        pts = [data.site_xpos[ctx.site[s]][:2] for s in ctx.foot_sites[foot]
               if data.site_xpos[ctx.site[s]][2] < ctx.touch_z]
        if pts:
            return np.mean(pts, axis=0)
    pts = [data.site_xpos[ctx.site[s]][:2] for foot in ctx.foot_sites for s in foot
           if data.site_xpos[ctx.site[s]][2] < ctx.touch_z]
    if pts:
        return np.mean(pts, axis=0)
    return ctx.ref_support[frame]


def foot_load(model, data, ctx) -> np.ndarray:
    """(2,) measured normal contact force per foot (N), from the solver.

    Sums the normal components of every contact involving that foot's body
    geoms (floor and everything else).  This is the sensor the load-transfer
    guards use: site height is a kinematic proxy, this is the actual load.
    """
    out = np.zeros(2)
    if not hasattr(ctx, "foot_geoms"):
        ctx.foot_geoms = []
        for foot in ("left", "right"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    f"{ctx.prefix}{foot}_ankle_roll_link")
            ctx.foot_geoms.append(
                {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid})
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = int(con.geom1), int(con.geom2)
        for f in range(2):
            if g1 in ctx.foot_geoms[f] or g2 in ctx.foot_geoms[f]:
                f6 = np.zeros(6)
                mujoco.mj_contactForce(model, data, c, f6)
                out[f] += abs(float(f6[0]))
    return out


def foot_site_z(data, ctx, foot: int) -> float:
    """Lowest sole-site height of one foot (m)."""
    return float(min(data.site_xpos[ctx.site[s]][2] for s in ctx.foot_sites[foot]))


def foot_contact(data, ctx) -> tuple[np.ndarray, np.ndarray]:
    """(2,2) site-touching flags and (2,) per-foot contact flags."""
    site = np.zeros((2, 2), dtype=bool)
    for foot in range(2):
        for s in range(2):
            site[foot, s] = data.site_xpos[ctx.site[ctx.foot_sites[foot][s]]][2] < ctx.touch_z
    return site, site.any(axis=1)


def heading_yaw(data, ctx) -> float:
    """Base heading (yaw) of the robot, from the pelvis frame."""
    R = data.xmat[ctx.pelvis].reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))
