"""Stance geometry construction for the solo drill (operator two-axis rule).

Why this module exists
----------------------
The retargeted STANCE crouch is statically infeasible: the feasibility audit
(``data/support_envelope.json``) classifies every frame with a *negative*
CoM margin under the sole-patch model, and the previous agent's joint-space
"trims" could not repair it (rear-leg-back 0.486 / widened 0.458-0.569 stay-up
vs the shipped torso-pitch trim 0.792, 3 seeds) because small joint offsets
cannot move a CoM that is outside the support polygon back inside it.

The operator rule is therefore applied as **geometry**, not as offsets:

  (a) sagittal  — move the rear foot further back (longer base behind the line
      of action);
  (b) frontal   — widen the stance (feet further apart laterally, so falling
      sideways is harder).

Both are realised as *foot placement targets* on the model's own
verified-stable ``stand`` keyframe; the legs are re-solved by damped
least-squares IK to keep the soles flat on the mat at those placements, and the
pelvis may be lowered (squat) with the feet pinned.  Every candidate is
measured — width, depth, pelvis height, CoM margin (sole-patch hull and, live,
the actual contact points) — before any balance claim is made.

The module is scene-agnostic: ``prefix`` is ``""`` in the single-G1 scene and
``a_``/``b_`` in the paired scene.
"""

from __future__ import annotations

import mujoco
import numpy as np

#: sole-patch half extents used for the *pose-level* support polygon (m):
#: the toe/heel sites sit at the foot's contact spheres; the patch is widened
#: laterally by the foot half width and longitudinally by the sphere radius.
SOLE_HALF_WIDTH = 0.035
SOLE_HALF_LIFT = 0.010


def _scratch(model) -> mujoco.MjData:
    key = id(model)
    d = _SCRATCH.get(key)
    if d is None or d.model is not model:
        d = mujoco.MjData(model)
        _SCRATCH[key] = d
    return d


_SCRATCH: dict[int, mujoco.MjData] = {}


def _fk(model, prefix: str, qpos36: np.ndarray) -> mujoco.MjData:
    d = _scratch(model)
    d.qpos[:] = 0.0
    sl = _qpos_slice(model, prefix)
    d.qpos[sl] = np.asarray(qpos36, float)
    mujoco.mj_kinematics(model, d)
    return d


def _qpos_slice(model, prefix: str) -> slice:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                            f"{prefix}floating_base_joint")
    start = int(model.jnt_qposadr[jid])
    return slice(start, start + 36)


def _site_id(model, prefix: str, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                             f"{prefix}{name}")


#: sole sites of each foot: (toe, heel)
FOOT_SITES = (("left_toe", "left_heel"), ("right_toe", "right_heel"))


def sole_positions(model, prefix: str, qpos36: np.ndarray) -> np.ndarray:
    """(2, 2, 3) world sole-site positions of one robot's pose."""
    d = _fk(model, prefix, qpos36)
    return np.array([[d.site_xpos[_site_id(model, prefix, s)].copy()
                      for s in foot] for foot in FOOT_SITES])


def leg_joint_ids(model, prefix: str) -> np.ndarray:
    """qpos indices of the 12 leg hinges (L hip..ankle, R hip..ankle)."""
    out = []
    for side in ("left", "right"):
        for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee",
                     "ankle_pitch", "ankle_roll"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{prefix}{side}_{name}_joint")
            out.append(int(model.jnt_qposadr[jid]))
    return np.array(out, dtype=int)


def leg_dof_ids(model, prefix: str) -> np.ndarray:
    """dof indices of the 12 leg hinges."""
    out = []
    for side in ("left", "right"):
        for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee",
                     "ankle_pitch", "ankle_roll"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{prefix}{side}_{name}_joint")
            out.append(int(model.jnt_dofadr[jid]))
    return np.array(out, dtype=int)


def solve_ik(model, prefix: str, qpos36: np.ndarray,
             targets: np.ndarray, feet: tuple = (0, 1), iters: int = 60,
             reg: float = 1e-3, step_max: float = 0.25,
             tol: float = 1e-4) -> np.ndarray:
    """Damped-LSQ (Levenberg-Marquardt) leg IK for the sole sites.

    ``targets`` is (2, 2, 3); only the rows of the selected feet are used, and
    only their 6 leg joints move.  The base pose is taken from ``qpos36`` (so
    lowering ``qpos36[2]`` solves a squat with the feet pinned).  Joint limits
    are enforced by clipping to the model ranges.  ``iters=1`` is the
    incremental Newton step used to march a reference across rows; the default
    is a full solve.

    Damping is adapted (halved on accepted steps, quadrupled on rejected ones):
    a *fixed* strong damping leaves a ~1 cm sole residual at large base shifts
    (the runtime foot anchor then drags the already-pinned feet — measured), and
    a *fixed* weak damping takes wild steps at singular configurations.
    """
    q = np.asarray(qpos36, float).copy()
    jids = leg_joint_ids(model, prefix)
    dofs = leg_dof_ids(model, prefix)
    sel_j = np.array([i for foot in feet for i in range(6 * foot, 6 * foot + 6)],
                     dtype=int)
    lo = model.jnt_range[jids[sel_j], 0]
    hi = model.jnt_range[jids[sel_j], 1]
    jac = np.zeros((3, model.nv))
    d = _scratch(model)
    sl = _qpos_slice(model, prefix)
    lam = float(reg)
    err_prev = None
    for _ in range(iters):
        d.qpos[:] = 0.0
        d.qpos[sl] = q
        mujoco.mj_kinematics(model, d)
        rows, errs = [], []
        for foot in feet:
            for s in range(2):
                sid = _site_id(model, prefix, FOOT_SITES[foot][s])
                e = targets[foot, s] - d.site_xpos[sid]
                w = np.array([1.0, 1.0, 2.0])      # ground contact weighted
                mujoco.mj_jacSite(model, d, jac, None, sid)
                rows.append(jac[:, dofs[sel_j]] * w[:, None])
                errs.append(e * w)
        A = np.vstack(rows)
        b = np.concatenate(errs)
        n = len(sel_j)
        err = float(b @ b)
        accepted = err_prev is None or err < err_prev
        lam = max(lam * 0.5, 1e-6) if accepted else min(lam * 4.0, 1.0)
        err_prev = err if err_prev is None else min(err, err_prev)
        dq, *_ = np.linalg.lstsq(np.vstack([A, np.sqrt(lam) * np.eye(n)]),
                                 np.append(b, np.zeros(n)), rcond=None)
        dq = np.clip(dq, -step_max, step_max)
        if accepted and float(np.max(np.abs(dq))) < tol:
            break
        q[jids[sel_j]] = np.clip(q[jids[sel_j]] + dq, lo, hi)
    return q


def solve_legs_ik(model, prefix: str, qpos36: np.ndarray,
                  targets: np.ndarray, iters: int = 60, reg: float = 1e-4,
                  step_max: float = 0.25, tol: float = 1e-4) -> np.ndarray:
    """Both-legs IK (see ``solve_ik``); kept for the stance builder."""
    return solve_ik(model, prefix, qpos36, targets, feet=(0, 1), iters=iters,
                    reg=reg, step_max=step_max, tol=tol)


def solve_foot_ik(model, prefix: str, qpos36: np.ndarray, foot: int,
                  targets: np.ndarray, iters: int = 60, reg: float = 1e-4,
                  step_max: float = 0.25, tol: float = 1e-4) -> np.ndarray:
    """One-foot IK: place that foot's sole sites at ``targets`` (2, 3)."""
    tg = np.zeros((2, 2, 3))
    tg[int(foot)] = targets
    return solve_ik(model, prefix, qpos36, tg, feet=(int(foot),), iters=iters,
                    reg=reg, step_max=step_max, tol=tol)


def foot_centre(model, prefix: str, qpos36: np.ndarray) -> np.ndarray:
    """(2, 3) mean sole position of each foot (world)."""
    return sole_positions(model, prefix, qpos36).mean(axis=1)


def foot_local(model, prefix: str, qpos36: np.ndarray) -> np.ndarray:
    """(2, 3) mean sole position of each foot in the *pelvis* frame."""
    d = _fk(model, prefix, qpos36)
    mujoco.mj_forward(model, d)
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}pelvis")
    R = d.xmat[pid].reshape(3, 3)
    p = d.xpos[pid]
    return (R.T @ (foot_centre(model, prefix, qpos36) - p).T).T


def reachable_shift(model, prefix: str, qpos36: np.ndarray, feet_targets: np.ndarray,
                    direction: np.ndarray, max_shift: float,
                    tol: float = 0.003, iters: int = 5) -> tuple[float, float]:
    """Largest base shift along ``direction`` the legs can hold with the feet
    pinned flat (bisection on the both-legs IK sole residual).

    The G1's legs cannot translate the pelvis over a *wide* stance while both
    feet stay flat (measured: a 0.10 m lateral shift leaves a 1.5 cm residual at
    a 0.26 m stance, 7 cm at 0.14 m — the hip/knee range runs out).  A stepping
    primitive must therefore ask for what is reachable; this returns
    ``(shift, residual)`` measured by the same IK the reference marches with.
    """
    d = np.asarray(direction, float)
    n = float(np.linalg.norm(d))
    if n < 1e-9 or max_shift <= 0:
        return 0.0, 0.0
    d = d / n
    lo, hi = 0.0, float(max_shift)
    res_lo = 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        q = np.asarray(qpos36, float).copy()
        q[0] += d[0] * mid
        q[1] += d[1] * mid
        q2 = solve_ik(model, prefix, q, feet_targets, feet=(0, 1), iters=30,
                      step_max=0.25)
        res = float(np.abs(sole_positions(model, prefix, q2)
                           - feet_targets).max())
        if res <= tol:
            lo, res_lo = mid, res
        else:
            hi = mid
    return lo, res_lo


def stance_metrics(model, prefix: str, qpos36: np.ndarray) -> dict:
    """Pose-level stance metrics: width, depth, pelvis height, CoM margin.

    ``margin`` is the signed distance of the CoM (xy) inside the convex hull
    of the four sole-patch corners (positive = inside), using the
    ``SOLE_HALF_*`` patch approximation; ``com_margin_live`` (see
    ``live_support_margin``) is the contact-based counterpart.
    """
    d = _fk(model, prefix, qpos36)
    mujoco.mj_forward(model, d)
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}pelvis")
    com = d.subtree_com[pid].copy()
    feet = sole_positions(model, prefix, qpos36)
    f = foot_centre(model, prefix, qpos36)
    # pelvis-frame lateral / sagittal separation
    R = d.xmat[pid].reshape(3, 3)
    p = d.xpos[pid]
    loc = (R.T @ (f - p).T).T
    width = float(loc[0, 1] - loc[1, 1])          # left minus right lateral
    depth = float(loc[1, 0] - loc[0, 0]) if loc[1, 0] > loc[0, 0] else \
        float(loc[0, 0] - loc[1, 0])
    poly = []
    for foot in range(2):
        for s in range(2):
            q_ = feet[foot, s]
            poly.append([q_[0] - SOLE_HALF_LIFT, q_[1] - SOLE_HALF_WIDTH])
            poly.append([q_[0] + SOLE_HALF_LIFT, q_[1] + SOLE_HALF_WIDTH])
    poly = np.array(poly)
    return {
        "width": width, "depth": depth,
        "pelvis_z": float(qpos36[2]),
        "com_xy": com[:2].copy(), "com_z": float(com[2]),
        "margin": point_in_hull_margin(poly, com[:2]),
    }


def point_in_hull_margin(poly: np.ndarray, point: np.ndarray) -> float:
    """Signed distance of ``point`` inside the convex hull of ``poly`` (m).

    Positive inside; ``inf``-guarded for degenerate hulls.
    """
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(poly)
    except Exception:
        return float("-inf")
    A, b = hull.equations[:, :2], hull.equations[:, 2]
    return float(-(A @ np.asarray(point) + b).max())


def settle_stance(model, prefix: str, qpos36: np.ndarray, seconds: float = 1.0,
                  substeps: int = 10) -> tuple[np.ndarray, dict]:
    """Settle a stance under the position servos and return the settled qpos.

    An IK-built stance is geometrically consistent but *not* the equilibrium of
    the position servos: at the new joint angles the gravitational torques
    differ, so the servo sag moves the pelvis (measured: pure tracking of an
    IK-built stance topples in ~1.5 s, while the model's own ``stand`` keyframe
    holds).  Running the servos open-loop for a short settle and adopting the
    *settled* pose as the reference makes the reference the equilibrium the
    actuators actually hold (the sag is then ≈ 0 and the CoM sits where the
    simulation puts it).

    Returns ``(qpos_settled, info)`` with the settle drop, drift and tilt.
    """
    q = np.asarray(qpos36, float)
    jid0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                             f"{prefix}left_hip_pitch_joint")
    acts = []
    for j in range(jid0, jid0 + 29):
        ids = np.where(model.actuator_trnid[:, 0] == j)[0]
        acts.append(int(ids[0]))
    acts = np.array(acts, dtype=int)
    d = mujoco.MjData(model)
    sl = _qpos_slice(model, prefix)
    d.qpos[sl] = q
    mujoco.mj_forward(model, d)
    n = int(round(seconds / model.opt.timestep))
    for k in range(n):
        if k % substeps == 0:
            d.ctrl[acts] = q[7:36]
        mujoco.mj_step(model, d)
    settled = d.qpos[sl].copy()
    info = {
        "drop_z": float(q[2] - settled[2]),
        "drift_xy": float(np.linalg.norm(np.asarray(settled[:2]) - q[:2])),
        "settle_s": seconds,
    }
    return settled, info


def live_support_margin(model, data: mujoco.MjData, ctx) -> dict:
    """Contact-based margin: CoM inside the hull of the ACTUAL contact points.

    Uses the solver's contact positions for that robot's foot geoms (floor
    contacts), which is the real support polygon once contact exists — not a
    site-height proximity proxy.
    """
    if not hasattr(ctx, "foot_geoms"):
        ctx.foot_geoms = []
        for foot in ("left", "right"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    f"{ctx.prefix}{foot}_ankle_roll_link")
            ctx.foot_geoms.append(
                {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid})
    pts = []
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 in ctx.foot_geoms[0] or g1 in ctx.foot_geoms[1] or \
                g2 in ctx.foot_geoms[0] or g2 in ctx.foot_geoms[1]:
            if float(con.dist) < 0.01:
                pts.append(np.asarray(data.contact[c].pos)[:2].copy())
    com = data.subtree_com[ctx.pelvis][:2].copy()
    out = {"com_xy": com, "n_contact": len(pts), "margin": float("nan")}
    if len(pts) >= 3:
        out["margin"] = point_in_hull_margin(np.array(pts), com)
    return out


def build_stance(model, prefix: str, stand36: np.ndarray, width: float,
                 depth: float, lead: int = 0, lead_dx: float = 0.0,
                 drop: float = 0.0, flat: bool = True,
                 iters: int = 80) -> tuple[np.ndarray, dict]:
    """Staggered, wide, grounded stance from the model's ``stand`` keyframe.

    Parameters (all metres, measured from the stand keyframe's own foot
    placement): ``width`` = lateral separation of the two foot centres;
    ``depth`` = how far the rear foot sits behind the lead foot; ``lead`` =
    which foot is in front (0 = left); ``lead_dx`` = extra forward offset of
    the lead foot; ``drop`` = pelvis lowering (squat with the feet pinned).

    Returns ``(qpos36, metrics)`` with the metrics of ``stance_metrics`` plus
    ``drift`` = the IK's residual on the sole targets.
    """
    q = np.asarray(stand36, float).copy()
    d = _fk(model, prefix, q)
    mujoco.mj_forward(model, d)
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}pelvis")
    R = d.xmat[pid].reshape(3, 3)
    p = d.xpos[pid]
    feet = sole_positions(model, prefix, q)
    centres = feet.mean(axis=1)
    loc = (R.T @ (centres - p).T).T                       # (2,3) pelvis frame
    rot = R.copy()
    rear = 1 - int(lead)

    # desired pelvis-frame foot centres
    want = loc.copy()
    want[lead, 1] = +0.5 * width
    want[rear, 1] = -0.5 * width
    # sagittal: lead at +lead_dx/1 ... rear at -(depth + lead_dx)... keep the
    # pair centred on the body: lead +0.5*(depth+lead_dx), rear -0.5*(depth+lead_dx)
    span = depth + lead_dx
    want[lead, 0] = +0.5 * span
    want[rear, 0] = -0.5 * span
    # (a foot may be swapped: keep the requested widths attached to the sides
    #  actually used; lead=0 means the LEFT foot is the lead)
    want[0, 1] = +0.5 * width
    want[1, 1] = -0.5 * width

    delta = (rot @ (want - loc).T).T                      # world-frame shift
    targets = feet.copy()
    for foot in range(2):
        for s in range(2):
            targets[foot, s, :2] += delta[foot, :2]
    if flat:
        rest_z = feet[:, :, 2].min()
        targets[:, :, 2] = rest_z
    q[2] = float(stand36[2]) - float(drop)
    q = solve_legs_ik(model, prefix, q, targets, iters=iters)
    m = stance_metrics(model, prefix, q)
    m["drift"] = float(np.max(np.abs(
        sole_positions(model, prefix, q)[:, :, :2]
        - targets[:, :, :2])))
    return q, m
