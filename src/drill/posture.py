"""Drill posture: the stance, its variations, and the measured geometry report.

The stance is built *kinematically* from a small parameter set (foot
placements, hips height, torso and arm carriage) so the operator's two
sanctioned repairs are explicit, testable numbers rather than inherited joint
values:

* **sagittal** -- the rear foot sits further back (longer base behind the line
  of action): ``rear_back``;
* **frontal**  -- the stance is wide (feet further apart laterally, rear foot
  angled away): ``half_width``, ``rear_yaw``.  A minimum width is *enforced*
  (``min_width``): the operator's rule is wide for lateral stability.

Two coupled requirements decide where the hips end up, and both are solved
here rather than guessed:

1. **reach** -- the legs must actually touch the mat at the operator's foot
   placement.  Asking for hips that are too high leaves a sole in the air
   (measured: 1.7 cm), so the height falls until both legs land;
2. **centring** -- the CoM must sit over the footprint centre, otherwise the
   stance is held by constant ankle torque (the retargeted STANCE fails
   exactly this way: its CoM sits behind its support and it topples).  With the
   hands carried forward (a wrestling read, ~17% of body mass) the CoM sits
   ~5 cm ahead of the hips, so the hips slide *back* until CoM and support
   agree -- which is also the operator's "hips back" stance.

``build_stance`` returns the 36-dim qpos in a frame whose origin **is** the
pelvis (``+x`` forward, ``+y`` left at the built heading) plus a report of the
measured rubric-A quantities, so every geometry claim in the report is a
measurement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import mujoco
import numpy as np

from . import kin as K
from . import scene as scene_mod

#: joint names of the arm chain used for the carriage IK (per side)
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
WAIST_JOINTS = ("waist_yaw", "waist_roll", "waist_pitch")
#: deepest pelvis drop used by LEVEL_CHANGE (m below the built stance height).
#: Measured: the built stance already sits in a crouch, and 10 cm more is past
#: what the ankle actuators hold from the stance pose (the pose collapses under
#: pure PD and the balance law cannot save it), so the drill's crouch range is
#: 6 cm and the deepest holdable height is verified per run.
MAX_CROUCH = 0.045
#: IK seed: a generic flexed leg.  Seeding from straight legs is singular (the
#: knee Jacobian vanishes at full extension and the solver walks into the
#: hyperextension limit instead of flexing) -- measured, not theoretical.
LEG_SEED = np.array([-0.20, 0.05, 0.0, 0.35, -0.15, 0.0])


@dataclass(frozen=True)
class StanceSpec:
    """Wrestling stance parameters (metres / radians, foot-placement frame)."""

    #: lead (left) foot frame origin, forward of the feet-frame centre
    lead_fwd: float = 0.150
    #: rear (right) foot frame origin, behind the feet-frame centre
    rear_back: float = -0.200
    #: lateral half-separation of the two foot origins
    half_width: float = 0.150
    #: foot yaw in the stance frame; positive = toe towards +y (left/out)
    lead_yaw: float = 0.12
    rear_yaw: float = -0.30
    #: torso carriage (rad): forward lean and bladed rotation
    waist_pitch: float = 0.16
    waist_yaw: float = 0.12
    waist_roll: float = 0.0
    #: hand targets in the pelvis frame (m): forward / lateral / height above pelvis
    hand_fwd: float = 0.30
    hand_lat: float = 0.105
    hand_up: float = 0.32
    #: minimum lateral foot separation the stance may shrink to (m) — enforced
    min_width: float = 0.24
    #: filled by the solve (informational; not an input)
    pelvis_z: float | None = None

    def targets(self) -> dict:
        """Foot frame origins (feet-placement frame) for the lead and rear foot."""
        return {"lead": np.array([self.lead_fwd, self.half_width]),
                "rear": np.array([self.rear_back, -self.half_width])}

    def foot_yaw(self, side: str) -> float:
        return self.lead_yaw if side == "left" else self.rear_yaw

    def to_dict(self) -> dict:
        return asdict(self)


class Stance:
    """A built stance: qpos + the ids and targets the runtime plan uses."""

    def __init__(self, model: mujoco.MjModel, ids: K.RobotIds, spec: StanceSpec,
                 qpos: np.ndarray, report: dict):
        self.model = model
        self.ids = ids
        self.spec = spec
        self.qpos = np.asarray(qpos, float).copy()
        self.report = report
        self._data = None
        #: arm joint values of the built carriage (per side, ARM_JOINTS order)
        self.arm_q = {s: np.array([self.qpos[model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_{j}_joint")]] for j in ARM_JOINTS])
            for s in K.SIDES}

    def scratch(self) -> mujoco.MjData:
        if self._data is None:
            self._data = mujoco.MjData(self.model)
        return self._data

    def foot_origins(self) -> dict:
        """Foot *frame* origins (ankle-roll body origin, stance frame, m).

        This is the handle every plan uses (``FootTarget.origin_xy``): the
        footprint centre is 3.5 cm further forward (mean of the sole points), and
        mixing the two produced a constant 3.5 cm drag -- and, in the entry, a
        robot that kept stepping forward to chase the mismatch.
        """
        d = self.scratch()
        d.qpos[:] = self.qpos
        mujoco.mj_forward(self.model, d)
        return {s: np.array(d.xpos[self.ids.foot_body[s]][:2], float) for s in K.SIDES}

    def leg_q(self, data: mujoco.MjData | None = None) -> dict:
        """Leg joint values of the built stance (per side)."""
        return {s: np.array(self.qpos[self.ids.leg_qadr[s]]) for s in K.SIDES}

    def pose_at_height(self, height_frac: float) -> np.ndarray:
        """Stance qpos with the pelvis lowered by ``height_frac`` of the crouch.

        ``height_frac`` in [0, 1]: 0 = the built stance, 1 = ``MAX_CROUCH``
        below it.  Feet stay at their stance-frame positions and the legs
        absorb the drop (knees/ankles -- never the waist).
        """
        h = float(np.clip(height_frac, 0.0, 1.0))
        q = self.qpos.copy()
        origins = self.foot_origins()
        d = self.scratch()
        q[2] -= MAX_CROUCH * h
        d.qpos[:] = q                     # the IK needs the lowered base pose
        mujoco.mj_forward(self.model, d)
        for side in K.SIDES:
            tg = K.foot_targets(self.ids, side, origins[side], self.spec.foot_yaw(side))
            qadr = self.ids.leg_qadr[side]
            q[qadr] = K.leg_ik(self.model, d, self.ids, side, tg, q[qadr], iters=20)
        return q


def _joint_qadr(model: mujoco.MjModel, names: tuple) -> np.ndarray:
    return np.array([model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])


def _solve_legs(model: mujoco.MjModel, ids: K.RobotIds, d: mujoco.MjData,
                spec: StanceSpec, pelvis_xyz: np.ndarray, q_seed: np.ndarray,
                iters: int = 30) -> tuple[np.ndarray, dict]:
    """Leg IK for both feet at a candidate pelvis pose; feet stay on the spec.

    Foot targets are world-fixed (``spec.targets()``), so this answers "given
    the hips here, do the feet still land flat where the operator wanted them".
    """
    q = np.asarray(q_seed, float).copy()
    q[:3] = np.asarray(pelvis_xyz, float)
    q[3:7] = K.yaw_quat(0.0)
    d.qpos[:] = q
    mujoco.mj_forward(model, d)
    info = {}
    for side, key in (("left", "lead"), ("right", "rear")):
        tg = K.foot_targets(ids, side, spec.targets()[key], spec.foot_yaw(side))
        qadr = ids.leg_qadr[side]
        q[qadr] = K.leg_ik(model, d, ids, side, tg, q[qadr], iters=iters)
        d.qpos[qadr] = q[qadr]
        info[side] = {"err": K.ik_error(model, d, ids, side, tg, q[qadr]),
                      "knee": float(q[qadr[3]]), "hip": float(q[qadr[0]]),
                      "ankle": float(q[qadr[4]]), "ankle_roll": float(q[qadr[5]])}
    return q, info


def solve_stance_pose(model: mujoco.MjModel, ids: K.RobotIds, spec: StanceSpec,
                      q_seed: np.ndarray | None = None, z_hi: float = 0.80,
                      z_lo: float = 0.58, tol: float = 1e-3, gap_tol: float = 3e-3,
                      knee_target: float = 0.55, iters: int = 14) -> tuple[np.ndarray, dict]:
    """Fixed-point solve for hips that reach the feet, crouch, and centre the CoM.

    Each iteration: solve the legs; lower the hips when a foot misses the mat
    (reach) or when a knee is straighter than ``knee_target`` (the crouch must
    be real -- a straight lead leg reads as a lunge, not a stance); otherwise
    slide the pelvis by 80% of the CoM/support gap.  The caller carries the
    upper body (arms change the CoM -- ~6 cm for a hands-forward carriage,
    which is exactly the offset this loop removes).
    """
    d = mujoco.MjData(model)
    if q_seed is None:
        q_seed = scene_mod.keyframe(model, "stand").copy()
        for side in K.SIDES:
            q_seed[ids.leg_qadr[side]] = LEG_SEED
    z = float(z_hi)
    xy = np.zeros(2)
    hist = []
    for _ in range(int(iters)):
        q, info = _solve_legs(model, ids, d, spec, np.array([xy[0], xy[1], z]), q_seed)
        err = max(v["err"] for v in info.values())
        knee_min = float(min(abs(v["knee"]) for v in info.values()))
        knee = float(np.mean([abs(v["knee"]) for v in info.values()]))
        if err > tol and z > z_lo:
            z = max(z_lo, z - 0.012)
            hist.append({"z": round(z, 4), "err": round(err, 5), "reason": "reach"})
            continue
        if knee_min < knee_target and z > z_lo:
            z = max(z_lo, z - 0.008)
            hist.append({"z": round(z, 4), "knee_min": round(knee_min, 3),
                         "reason": "crouch"})
            continue
        d.qpos[:] = q
        mujoco.mj_forward(model, d)
        pts = np.vstack([ids.sole_points(d, s) for s in K.SIDES])
        gap = pts.mean(axis=0)[:2] - ids.com_xy(d)
        hist.append({"z": round(z, 4), "err": round(err, 5), "knee": round(knee, 3),
                     "gap": gap.round(4).tolist(), "xy": xy.round(4).tolist()})
        if np.linalg.norm(gap) < gap_tol:
            break
        xy = xy + 0.8 * gap
    q, info = _solve_legs(model, ids, d, spec, np.array([xy[0], xy[1], z]), q_seed,
                          iters=40)
    info["history"] = hist
    info["pelvis"] = [round(float(xy[0]), 4), round(float(xy[1]), 4), round(float(z), 4)]
    return q, info


def solve_upper(model: mujoco.MjModel, ids: K.RobotIds, spec: StanceSpec,
                q: np.ndarray) -> np.ndarray:
    """Waist + arm carriage of the stance at the pelvis pose inside ``q``."""
    q = q.copy()
    d = mujoco.MjData(model)
    wadr = _joint_qadr(model, tuple(f"waist_{j}_joint" for j in ("yaw", "roll", "pitch")))
    q[wadr] = [spec.waist_yaw, spec.waist_roll, spec.waist_pitch]
    d.qpos[:] = q
    mujoco.mj_forward(model, d)
    for side in K.SIDES:
        sgn = 1.0 if side == "left" else -1.0
        tgt = np.array([q[0] + spec.hand_fwd, q[1] + sgn * spec.hand_lat,
                        q[2] + spec.hand_up])
        jid = lambda j: mujoco.mj_name2id(  # noqa: E731
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}_joint")
        dofs = np.array([model.jnt_dofadr[jid(j)] for j in ARM_JOINTS])
        qadr = np.array([model.jnt_qposadr[jid(j)] for j in ARM_JOINTS])
        lim = np.array([model.jnt_range[jid(j)] for j in ARM_JOINTS])
        q[qadr] = K.chain_ik(model, d, dofs, qadr, lim,
                             [(ids.site[f"{side}_wrist"], tgt)], q[qadr], iters=60)
        d.qpos[:] = q
        mujoco.mj_forward(model, d)
    return q


def build_stance(model: mujoco.MjModel, spec: StanceSpec | None = None,
                 ids: K.RobotIds | None = None, rounds: int = 4) -> Stance:
    """Solve the stance (hips + legs + arms) from ``spec`` and measure it."""
    spec = spec or StanceSpec()
    ids = ids or K.RobotIds.build(model)
    assert 2.0 * spec.half_width >= spec.min_width - 1e-9, (
        f"stance width {2*spec.half_width:.3f} m below the enforced minimum "
        f"{spec.min_width:.3f} m (operator rule: wide for lateral stability)")
    q = scene_mod.keyframe(model, "stand").copy()
    for side in K.SIDES:
        q[ids.leg_qadr[side]] = LEG_SEED
    legs = {}
    for _ in range(int(rounds)):
        q, legs = solve_stance_pose(model, ids, spec, q_seed=q)
        q = solve_upper(model, ids, spec, q)
    pelvis = legs["pelvis"]
    spec = replace(spec, pelvis_z=pelvis[2])
    hips_back = float(q[0])
    hips_side = float(q[1])
    q[:2] = 0.0                     # the stance frame origin IS the pelvis
    report = measure_stance(model, ids, spec, q)
    report["solve"] = {
        "history": legs["history"], "pelvis_final": pelvis,
        "hips_back_m": round(hips_back, 4), "hips_side_m": round(hips_side, 4),
        "leg_ik": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                       for kk, vv in v.items()} for k, v in legs.items()
                   if k not in ("history", "pelvis")},
    }
    return Stance(model, ids, spec, q, report)


def measure_stance(model: mujoco.MjModel, ids: K.RobotIds, spec: StanceSpec,
                   q: np.ndarray) -> dict:
    """Quantitative stance report (rubric A) for a candidate qpos."""
    d = mujoco.MjData(model)
    d.qpos[:] = q
    mujoco.mj_forward(model, d)
    centers = {s: ids.sole_points(d, s).mean(axis=0) for s in K.SIDES}
    hull = K.hull2d(np.vstack([ids.sole_xy(d, s) for s in K.SIDES]))
    com = ids.com(d)
    margin = K.polygon_margin(com[:2], hull)
    qof = lambda name: float(q[model.jnt_qposadr[mujoco.mj_name2id(  # noqa: E731
        model, mujoco.mjtObj.mjOBJ_JOINT, name)]])
    hands = {}
    for s in K.SIDES:
        p = d.site_xpos[ids.site[f"{s}_wrist"]]
        hands[s] = np.array([*ids.local_xy(p[:2], d), float(p[2])])
    return {
        "pelvis_z": ids.pelvis_z(d),
        "width_m": float(abs(centers["left"][1] - centers["right"][1])),
        "base_depth_m": float(centers["left"][0] - centers["right"][0]),
        "foot_yaw_deg": {s: round(float(np.degrees(_foot_yaw(ids, d, s))), 1)
                         for s in K.SIDES},
        "knee_flex_rad": {s: float(q[ids.leg_qadr[s][3]]) for s in K.SIDES},
        "ankle_pitch_rad": {s: float(q[ids.leg_qadr[s][4]]) for s in K.SIDES},
        "waist_pitch_rad": qof("waist_pitch_joint"),
        "hip_pitch_rad": {s: float(q[ids.leg_qadr[s][0]]) for s in K.SIDES},
        "torso_tilt_deg": ids.torso_tilt_deg(d),
        "head_z": float(d.site_xpos[ids.site["head"]][2]),
        "com": com.tolist(),
        "com_margin_m": float(margin),
        "support_centroid": np.vstack([ids.sole_points(d, s) for s in K.SIDES]
                                      ).mean(axis=0)[:2].round(5).tolist(),
        "sole_flat": {s: bool(ids.flat_contact(d)[i]) for i, s in enumerate(K.SIDES)},
        "sole_min_z": {s: float(ids.sole_points(d, s)[:, 2].min()) for s in K.SIDES},
        "hands_local": {s: hands[s].round(4).tolist() for s in K.SIDES},
        "base_yaw_deg": float(np.degrees(ids.base_pose(d)[1])),
        "spec": spec.to_dict(),
    }


def _foot_yaw(ids: K.RobotIds, d: mujoco.MjData, side: str) -> float:
    """Foot heading (rad) of the sole's long axis in the world."""
    pts = ids.sole_points(d, side)
    heel = pts[list(K.HEEL_ROWS)].mean(axis=0)[:2]
    toe = pts[list(K.TOE_ROWS)].mean(axis=0)[:2]
    v = toe - heel
    return float(np.arctan2(v[1], v[0]))


if __name__ == "__main__":                              # self-check
    model = scene_mod.load_model()
    ids = K.RobotIds.build(model)
    st = build_stance(model, StanceSpec(), ids)
    r = st.report
    print("stance geometry (stance frame origin = pelvis):")
    print(f"  pelvis_z      {r['pelvis_z']:.3f} m  (hips back {r['solve']['hips_back_m']:+.3f} m)")
    print(f"  width         {r['width_m']:.3f} m (spec min 0.24)")
    print(f"  base depth    {r['base_depth_m']:.3f} m  (lead - rear)")
    print(f"  foot yaw      {r['foot_yaw_deg']}")
    print(f"  knee flex     {r['knee_flex_rad']}")
    print(f"  torso tilt    {r['torso_tilt_deg']:.1f} deg, head z {r['head_z']:.3f} m")
    print(f"  com           {np.round(r['com'], 3)}  support {r['support_centroid']}")
    print(f"  com margin    {r['com_margin_m']:.4f} m")
    print(f"  sole flat     {r['sole_flat']}  min z {r['sole_min_z']}")
    print("  hands local   " + "  ".join(
        f"{s}: {np.round(v, 3).tolist()}" for s, v in r["hands_local"].items())
        + "  (x, y, z)")
    print("  foot origins  " + str({k: np.round(v, 4).tolist()
                                    for k, v in st.foot_origins().items()}))
    cr = st.pose_at_height(1.0)
    d = mujoco.MjData(model)
    d.qpos[:] = cr
    mujoco.mj_forward(model, d)
    print(f"deep crouch: pelvis z {d.qpos[2]:.3f} knee L {d.qpos[ids.leg_qadr['left'][3]]:.3f} "
          f"margin {K.polygon_margin(ids.com_xy(d), K.hull2d(np.vstack([ids.sole_xy(d, s) for s in K.SIDES]))):.4f} m "
          f"flat {ids.flat_contact(d)}")
