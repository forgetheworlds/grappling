"""Runtime control core: reference IK + the measured balance law.

Two pieces, both single-robot and 29-target:

``ReferenceSolver``
    Turns a cartesian :class:`DrillPlan` (pelvis pose, per-foot sole targets,
    upper-body joints) into the 29 joint targets the position servos track.
    Legs are solved with the same damped Gauss-Newton IK used offline, warm
    started from the previous tick so the reference is continuous.

``BalanceLaw``
    The feedback layer: capture-point error against the *measured* support
    centre, the measured pitch/roll channel split (ankle/knee/hip, hip roll),
    planted-foot world anchoring + sole flattening, torso upright rows and a
    pelvis-height PI.  Gains are the measured ones from
    ``src/teacher/gains.py`` (STAND row); :func:`measured_gains` documents the
    provenance and ``tests/test_drill.py`` fails if the table drifts.

Deliberate departures from the paired teacher, each because the milestone is
one robot and one continuous state:

* **no phase table** -- the caller supplies the plan each tick (no 900-row
  procedural horizon, no clock-clamped reference);
* **no reference-frame foot flags** -- which feet are planted and where they
  must stay is part of the plan, so a swing foot is never anchored;
* **governor trigger from the unclipped capture error** (the paired teacher
  triggers on an already-clamped error, which makes the emergency blend
  unreachable -- review F3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import kin as K

G = 9.81
DT_CTRL = 0.02                     # 50 Hz control
COM_VEL_LP = 0.35                  # CoM velocity low-pass weight (per tick)


@dataclass(frozen=True)
class BalanceParams:
    """Balance-layer gains (provenance: teacher STAND row, see ``measured_gains``)."""

    k_pitch: float = 1.0           # fraction of the measured-authority correction
    k_roll: float = 0.5
    kd: float = 0.3                # capture-point lookahead (s)
    com_clamp: float = 0.08        # clamp on the CoM-balance error (m)
    task_clamp: float = 0.25       # clamp on the error fed to the channels (m)
    ki: float = 0.15               # task integrator (1/s).  The paired teacher
    #                                ships 0.6/0.12 against a *static* reference;
    #                                with a moving reference the same gain winds
    #                                up a 3 cm bias and slowly walks the CoM out
    #                                of the support (measured, this host)
    int_max: float = 0.03
    k_anchor: float = 0.6          # planted-foot world anchoring gain
    anchor_clamp: float = 0.08     # per-site anchor error clamp (m)
    k_flat: float = 1.0            # 1 = drive the sole flat to the mat
    k_z: float = 0.4               # pelvis-height P -> foot z offset
    ki_z: float = 0.4
    z_clamp: float = 0.05
    k_swing: float = 0.5           # swing-foot world tracking gain (the
    #                                swing leg's own joints only)
    swing_clamp: float = 0.05      # per-site swing tracking clamp (m)
    k_up: float = 0.2              # torso upright (waist) gain
    upright_blend: float = 0.4
    max_leg: float = 0.40          # per-tick clamp on leg offsets (rad)
    max_upp: float = 0.15
    max_rate: float = 0.25         # per-tick offset change (rad)
    reg: float = 0.05              # regularized-LS weight
    govern_e0: float = 0.05        # safety-blend activation (unclipped |e|, m)
    govern_e1: float = 0.14
    govern_up: float = 0.08        # safety blend slew per tick
    govern_down: float = 0.01

    def as_dict(self) -> dict:
        return {k: float(v) for k, v in self.__dict__.items()}


#: the measured channel split (teacher/stabilizers.py) — kept here as data so the
#: single-robot law is self-contained; tests assert equality with the teacher.
PITCH_SPLIT = {"ankle_pitch": 0.55, "knee": 0.30, "hip_pitch": 0.15}
ROLL_SPLIT = {"hip_roll": 1.00}
AUTH_PITCH = {"ankle_pitch": -0.53, "knee": -0.38, "hip_pitch": -0.09}
AUTH_ROLL = {"hip_roll": -0.46}
LEG_IDX = {"left": {"hip_pitch": 0, "hip_roll": 1, "knee": 3,
                    "ankle_pitch": 4, "ankle_roll": 5},
           "right": {"hip_pitch": 6, "hip_roll": 7, "knee": 9,
                     "ankle_pitch": 10, "ankle_roll": 11}}


def measured_gains() -> dict:
    """The teacher's measured STAND gains (lazy import; {} when unavailable).

    Used by the drift test and by the report: the single-robot law adopts these
    numbers, so if the teacher's table changes the drill must be re-validated.
    """
    try:
        from teacher.gains import GAIN_TABLE           # type: ignore
    except Exception:                                   # pragma: no cover
        return {}
    g = GAIN_TABLE.get("STAND")
    return {k: float(v) for k, v in g.__dict__.items()} if g is not None else {}


def channel_offsets(e_loc: np.ndarray, contact: np.ndarray, k_pitch: float,
                    k_roll: float, roll_sides: tuple | None = None,
                    k_roll_scale: float = 1.0) -> np.ndarray:
    """Direct pitch/roll balance offsets for a robot-local error (m).

    Same construction as the teacher's measured channels: the offset whose
    measured authority would cancel the error, split across ankle/knee/hip
    (pitch) or taken entirely by hip roll, and applied only to feet that are
    on the mat.  ``roll_sides`` restricts the *roll* channel to the feet that
    may carry roll authority (the step primitive's support foot while a foot
    is stepping -- the swing leg is compliant, see ``stepping.roll_authority``);
    pitch stays on every loaded foot.
    """
    dq = np.zeros(29)
    if not contact.any():
        return dq
    for side, idx in LEG_IDX.items():
        if not contact[0 if side == "left" else 1]:
            continue
        for name, share in PITCH_SPLIT.items():
            dq[idx[name]] += -share * k_pitch * e_loc[0] / AUTH_PITCH[name]
        if roll_sides is not None and side not in roll_sides:
            continue
        for name, share in ROLL_SPLIT.items():
            dq[idx[name]] += (-share * k_roll * k_roll_scale * e_loc[1]
                              / AUTH_ROLL[name])
    return dq


def capture_error(com: np.ndarray, v_com: np.ndarray, support: np.ndarray,
                  kd: float, clamp: float) -> np.ndarray:
    """World-frame capture-point error vs the support centre (m), clamped."""
    h = max(com[2] - 0.05, 0.30)
    xi = com[:2] + kd * np.sqrt(h / G) * v_com
    return np.clip(xi - support, -clamp, clamp)


@dataclass
class FootTarget:
    """Where one foot must be, and whether it is load bearing."""

    origin_xy: np.ndarray          # foot frame origin on the floor (world)
    yaw: float                     # foot heading (rad, world)
    sole_z: float = K.SOLE_REST_Z  # lowest footprint point (m)
    planted: bool = True           # True = anchored to this world target
    pitch: float = 0.0             # toe up (rad) — swing clearance shaping
    roll: float = 0.0              # sole roll (rad, + raises the +y edge) —
    #                                swing-foot compliance through a weight shift

    def points(self, ids: K.RobotIds, side: str) -> np.ndarray:
        return K.foot_targets(ids, side, self.origin_xy, self.yaw,
                              sole_z=self.sole_z, pitch=self.pitch, roll=self.roll)


@dataclass
class DrillPlan:
    """Cartesian plan for one control tick (world frame)."""

    base_xyz: np.ndarray                       # pelvis position
    base_yaw: float
    feet: dict                                 # side -> FootTarget
    upper: np.ndarray                          # 17 joints: waist 3 + arms 2x7
    label: str = ""

    def copy(self) -> "DrillPlan":
        return DrillPlan(self.base_xyz.copy(), float(self.base_yaw),
                         {s: FootTarget(f.origin_xy.copy(), f.yaw, f.sole_z,
                                        f.planted, f.pitch, f.roll)
                          for s, f in self.feet.items()},
                         self.upper.copy(), self.label)


class ReferenceSolver:
    """Plan -> 29 joint targets (legs by IK, upper body straight from the plan)."""

    def __init__(self, model: mujoco.MjModel, ids: K.RobotIds, iters: int = 3,
                 max_step: float = 0.04):
        self.model = model
        self.ids = ids
        self.iters = int(iters)
        #: per-tick slew limit on the *reference* (rad).  Targets that step
        #: instantly to a far pose are what launches a position-servo humanoid
        #: (measured: a 0.8 rad knee step lifted both feet 5 cm off the mat).
        self.max_step = float(max_step)
        self._data = mujoco.MjData(model)
        #: when set (36-pose tracking), ``solve`` returns these 29 joints
        #: directly instead of solving the cartesian plan -- a retargeted
        #: reference *is* a joint trajectory, and re-solving it through IK only
        #: adds the solver's transient (measured: 3 damped-GN iterations from a
        #: distant warm start left 3.5-6 cm of leg error for ~20 ticks and the
        #: robot collapsed under a plan that its own pose contradicted)
        self.direct: np.ndarray | None = None
        self.q_leg = {s: np.array([-0.20, 0.05, 0.0, 0.35, -0.15, 0.0])
                      for s in K.SIDES}
        self.q_out = {s: self.q_leg[s].copy() for s in K.SIDES}
        self.ik_err = {s: 0.0 for s in K.SIDES}
        self.upper_names = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                            *[f"{s}_{j}_joint" for s in K.SIDES for j in
                              ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                               "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")])

    def reset(self, q_ref: np.ndarray | None = None) -> None:
        """Seed the IK warm start (``q_ref`` is the 29-joint vector, not qpos)."""
        if q_ref is not None:
            for s in K.SIDES:
                # leg_qadr holds *qpos* addresses; inside the 29-vector the leg
                # blocks start at 0 (left) and 6 (right)
                self.q_leg[s] = np.array(q_ref[self.ids.leg_qadr[s] - 7], float)
        else:
            self.q_leg = {s: np.array([-0.20, 0.05, 0.0, 0.35, -0.15, 0.0])
                          for s in K.SIDES}
        self.q_out = {s: self.q_leg[s].copy() for s in K.SIDES}

    def solve(self, plan: DrillPlan) -> np.ndarray:
        """29 joint values for the plan (legs IK; upper body from ``plan.upper``)."""
        model, ids, d = self.model, self.ids, self._data
        if self.direct is not None:
            q = np.asarray(self.direct, float).copy()
            for s in K.SIDES:
                self.q_leg[s] = q[ids.leg_qadr[s] - 7].copy()
                self.q_out[s] = self.q_leg[s].copy()
                self.ik_err[s] = 0.0
            return q
        # scratch base pose for the leg IK: the planned pelvis pose
        d.qpos[:3] = plan.base_xyz
        d.qpos[3:7] = K.yaw_quat(plan.base_yaw)
        for s in K.SIDES:
            d.qpos[ids.leg_qadr[s]] = self.q_leg[s]
        q = np.zeros(29)
        q[12:29] = plan.upper                      # waist + arms (joint order 12..28)
        for side in K.SIDES:
            tg = plan.feet[side].points(ids, side)
            qleg = K.leg_ik(model, d, ids, side, tg, self.q_leg[side],
                            iters=self.iters)
            self.q_leg[side] = qleg
            self.ik_err[side] = K.ik_error(model, d, ids, side, tg, qleg)
            self.q_out[side] = self.q_out[side] + np.clip(
                qleg - self.q_out[side], -self.max_step, self.max_step)
            q[ids.leg_qadr[side] - 7] = self.q_out[side]
        return q


class BalanceLaw:
    """Feedback offsets on top of the reference (single robot, 29 targets)."""

    def __init__(self, model: mujoco.MjModel, ids: K.RobotIds,
                 params: BalanceParams | None = None):
        self.model = model
        self.ids = ids
        self.p = params or BalanceParams()
        self._com = None
        self._v = np.zeros(2)
        self._int = np.zeros(2)
        self._int_z = 0.0
        self._dq = np.zeros(29)
        self._alpha = 0.0
        self._jac = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self.info: dict = {}

    def reset(self) -> None:
        self._com = None
        self._v[:] = 0.0
        self._int[:] = 0.0
        self._int_z = 0.0
        self._dq[:] = 0.0
        self._alpha = 0.0

    # -- measurement -------------------------------------------------------
    def support_center(self, data: mujoco.MjData, touch_z: float = K.TOUCH_Z) -> np.ndarray:
        pts = []
        for side in K.SIDES:
            p = self.ids.sole_points(data, side)
            pts += [q[:2] for q in p if q[2] < touch_z]
        if pts:
            return np.mean(pts, axis=0)
        return self.ids.com_xy(data)

    def support_polygon(self, data: mujoco.MjData,
                        touch_z: float = K.TOUCH_Z) -> np.ndarray:
        """Hull of the full footprint of every foot currently on the mat."""
        contact = self.ids.foot_contact(data)
        return self.ids.support_polygon(data, contact)

    # -- one control tick --------------------------------------------------
    def offsets(self, data: mujoco.MjData, plan: DrillPlan,
                com_ref: np.ndarray, roll_sides: tuple | None = None,
                k_roll_scale: float = 1.0) -> tuple[np.ndarray, dict]:
        """(29,) balance offsets for this tick, and the measurement read-out.

        The task error tracks the *planned* CoM (``com_ref``), not the live
        support centre: deliberate weight transfer is part of the drill, so the
        law must follow the plan rather than fight any CoM that leaves the
        middle of the footprint.  Physical safety is enforced separately, by
        the measured margin and the emergency blend below.

        ``roll_sides`` (see :func:`channel_offsets`) restricts the roll channel
        to the feet allowed to push the body sideways this tick.
        """
        p, ids = self.p, self.ids
        com = ids.com(data)
        if self._com is None:
            v = np.zeros(2)
        else:
            raw = (com[:2] - self._com) / DT_CTRL
            v = (1 - COM_VEL_LP) * self._v + COM_VEL_LP * raw
        self._com = com[:2].copy()
        self._v = v

        contact = ids.foot_contact(data)
        support = self.support_center(data)
        h = max(com[2] - 0.05, 0.30)
        e_track_raw = com[:2] + p.kd * np.sqrt(h / G) * v - np.asarray(com_ref, float)
        e = np.clip(e_track_raw, -p.com_clamp, p.com_clamp)
        self._int = np.clip(self._int + p.ki * DT_CTRL * e, -p.int_max, p.int_max)
        e_dir = np.clip(e + self._int, -p.task_clamp, p.task_clamp)

        base, yaw = ids.base_pose(data)
        dq = channel_offsets(K.z_rot(-yaw) @ e_dir, contact, p.k_pitch, p.k_roll,
                             roll_sides=roll_sides, k_roll_scale=k_roll_scale)

        # planted-foot anchoring + sole flattening + pelvis-height PI
        z_err = float(base[2] - plan.base_xyz[2])
        self._int_z = float(np.clip(self._int_z + z_err * DT_CTRL, -0.03, 0.03))
        dz = float(np.clip(p.k_z * z_err + p.ki_z * self._int_z, -p.z_clamp, p.z_clamp))
        dz = float(np.clip(dz, -0.012, 0.012))     # small authority: the plan
        #                                            owns the height, the law trims
        rows, des = [], []
        for side in K.SIDES:
            ft = plan.feet[side]
            if not ft.planted:
                continue
            tg = ft.points(ids, side).copy()
            for j, sid in enumerate(ids.sole_sites[side]):
                cur = data.site_xpos[int(sid)]
                tgt = tg[j].copy()
                tgt[2] = (1 - p.k_flat) * tgt[2] + p.k_flat * (K.SOLE_REST_Z + dz)
                tgt[2] += dz
                err = np.clip(tgt - cur, -p.anchor_clamp, p.anchor_clamp)
                mujoco.mj_jacSite(self.model, data, self._jac, None, int(sid))
                rows.append(self._jac[:, ids.dofs].copy())
                des.append(p.k_anchor * err)

        # swing feet: track the planned world trajectory with the *swing leg's*
        # own joints (columns restricted), so a body drift cannot fling the
        # foot (measured: 0.15 m of unintended clearance) and the commanded
        # clearance/placement is what actually happens
        if p.k_swing > 0:
            for side in K.SIDES:
                ft = plan.feet[side]
                if ft.planted:
                    continue
                tg = ft.points(ids, side)
                cols = ids.leg_qadr[side] - 7          # 0..5 or 6..11
                for j, sid in enumerate(ids.sole_sites[side]):
                    err = np.clip(tg[j] - data.site_xpos[int(sid)],
                                  -p.swing_clamp, p.swing_clamp)
                    mujoco.mj_jacSite(self.model, data, self._jac, None, int(sid))
                    row = np.zeros((3, 29))
                    row[:, cols] = self._jac[:, ids.leg_dofs[side]]
                    rows.append(row)
                    des.append(p.k_swing * err)

        # torso upright (waist channel)
        if p.k_up > 0:
            up = ids.torso_up(data)
            err = np.cross(up, np.array([0.0, 0.0, 1.0]))
            mujoco.mj_jacBodyCom(self.model, data, self._jac, self._jacr, ids.torso)
            rows.append(self._jacr[:2, ids.dofs].copy())
            des.append(p.k_up * err[:2])
        dq_ls = _solve_rows(rows, des, p.reg)

        # safety: unclipped tracking error + a measured CoM outside support
        poly = self.support_polygon(data)
        margin = float(K.polygon_margin(com[:2], poly))
        gov = float(np.linalg.norm(e_track_raw)) + 2.0 * max(0.0, -margin)
        alpha_t = float(np.clip((gov - p.govern_e0) / (p.govern_e1 - p.govern_e0),
                                0.0, 1.0))
        step = p.govern_up if alpha_t > self._alpha else p.govern_down
        self._alpha += float(np.clip(alpha_t - self._alpha, -step, step))
        self._alpha = float(np.clip(self._alpha, 0.0, 1.0))

        dq = dq + dq_ls
        lim = np.full(29, p.max_upp)
        lim[:12] = p.max_leg
        dq = np.clip(dq, -lim, lim)
        dq = self._dq + np.clip(dq - self._dq, -p.max_rate, p.max_rate)
        self._dq = dq
        self.info = {"e": e.copy(), "e_track": e_track_raw.copy(),
                     "e_support": capture_error(com, v, support, p.kd, np.inf),
                     "e_int": self._int.copy(), "dz": dz, "z_err": z_err,
                     "alpha": self._alpha, "gov": gov,
                     "contact": contact.copy(), "support": support.copy(),
                     "com": com.copy(), "com_ref": np.asarray(com_ref, float).copy(),
                     "v_com": v.copy(), "margin": margin,
                     "poly": poly}
        return dq, self.info


def _solve_rows(rows: list, des: list, reg: float) -> np.ndarray:
    """Regularized least squares for the 29-joint offset."""
    if not rows:
        return np.zeros(29)
    A = np.vstack(rows)
    b = np.concatenate(des)
    A = np.vstack([A, np.sqrt(reg) * np.eye(29)])
    b = np.append(b, np.zeros(29))
    dq, *_ = np.linalg.lstsq(A, b, rcond=None)
    return dq


if __name__ == "__main__":                              # self-check
    from . import scene as scene_mod
    from . import posture as posture_mod

    m = scene_mod.load_model()
    ids = K.RobotIds.build(m)
    st = posture_mod.build_stance(m, posture_mod.StanceSpec(), ids)
    d = mujoco.MjData(m)
    solver = ReferenceSolver(m, ids)
    law = BalanceLaw(m, ids)
    plan = DrillPlan(base_xyz=np.array([0.0, 0.0, st.report["pelvis_z"]], float),
                     base_yaw=0.0,
                     feet={s: FootTarget(st.foot_origins()[s], st.spec.foot_yaw(s))
                           for s in K.SIDES},
                     upper=st.qpos[7 + 12:36], label="stance")
    q = solver.solve(plan)
    print("reference IK err (m):", {s: round(v, 5) for s, v in solver.ik_err.items()})
    print("reference max |q - stance|:",
          round(float(np.abs(q - st.qpos[7:36]).max()), 4), "rad")
    d.qpos[:] = st.qpos
    mujoco.mj_forward(m, d)
    dq, info = law.offsets(d, plan)
    print("balance offsets: max |dq|", round(float(np.abs(dq).max()), 4),
          "| e", np.round(info["e"], 4), "margin", round(info["margin"], 4))
    print("gains match teacher:", measured_gains() == {}, "(no teacher gains loaded)"
          if not measured_gains() else
          all(abs(measured_gains().get(k, getattr(BalanceParams(), k))
                  - getattr(BalanceParams(), k)) < 1e-9
              for k in ("k_pitch", "k_roll", "k_anchor", "k_flat", "k_up", "kd")))
