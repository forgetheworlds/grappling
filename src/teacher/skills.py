"""Driveable drill interface for the phase-3 teacher (operator priority).

A drill scheduler sets, at any control step:

    ctrl.set_skill("LEVEL_CHANGE")      # see SKILLS
    ctrl.set_command(vx=0.05, vy=0.0, wz=0.0)   # body-frame, m/s and rad/s
    ctrl.set_stance_height(-0.5)        # 0 = the reference stance, -1 = the
                                        # fully lowered stance (knees, not waist)
    joint_targets = ctrl.control(data, t)       # (29,) or (58,) every 50 Hz tick

and the *balance layer stays active the whole time*: every command builds a
procedural pose reference, and the stabilised teacher (capture-point ankle/hip
feedback, sole flattening, pelvis-height PI, task integrator) tracks it exactly
as it tracks a phase-2 reference.  This is what makes the elements executable
back-to-back without resetting the simulation.

Contract notes (each one is a measured defect fix):

* **50 Hz cadence.**  ``RobotTeacher`` is a 50 Hz controller — its
  integrators, rate limiter and filters advance per call with
  ``DT_CTRL = 0.02``.  Call ``control()`` once per 50 Hz tick and hold the
  returned target for the ten 2 ms physics substeps (``drill_step()``), as
  ``run_episode`` does for the reference replay.
* **Live reference.**  Every tick rewrites a window of reference rows around
  the drill clock and pushes it into the executor with
  ``RobotTeacher.update_reference``, which refreshes *every* reference-derived
  target (joint table, foot targets, pelvis height, heading, support centre,
  touching sites, phase labels) while preserving feedback state.  Without that
  the executor keeps tracking the pose it was constructed with.
* **Rolling timeline.**  The reference grows on demand (no fixed 18 s cap), so
  a 60-90 s continuous drill needs no rebuild and no clamp.
* **Per-element blending.**  Every element carries its own entry time; skill,
  stance-height and lead-step changes cross-fade over ``BLEND_S`` from the
  previous element evaluated at the same row time, and RECOVER_STAND blends
  over the time since *it* started — not over absolute drill time.
* **Feet are pinned where they are planted.**  The drill tracks a per-foot
  world pin (established at reset / at landing); the leg IK holds planted feet
  while the reference body moves.  A step is the sensor-guarded transfer of
  that pin: load one leg (measured contact forces), lift, transport, land,
  confirm loaded contact, then settle — one physical primitive, from which the
  shuffle/circle elements are built.

Motion model (all measured, see reports/2026-10-08/teacher_exec_fix.md):
  * every skill is a *pose* on a procedural timeline (stance / squat / lead
    step / recovery blend) **plus** foot placement through the pins;
  * movement commands advance the reference base xy/yaw at the measured
    authority and are realised by *stepping*, not by dragging planted feet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

from . import stabilizers as st
from .controller import FOOT_Z_TOUCH, RobotTeacher, TeacherFlags
from .trims import (MIN_STANCE_WIDTH, enforce_stance_width,
                    measure_stance_width, trim_for, trim_vector)

SKILLS = ("STANCE_HOLD", "SHUFFLE_FORWARD", "SHUFFLE_BACK", "SHUFFLE_LATERAL",
          "CIRCLE_L", "CIRCLE_R", "RETREAT", "APPROACH", "LEVEL_CHANGE",
          "SHOT_DOUBLE_LEG", "KNEE_LOWER", "RECOVER_STAND")

#: movement skills run an automatic step cycle from the command
MOVE_SKILLS = ("SHUFFLE_FORWARD", "SHUFFLE_BACK", "SHUFFLE_LATERAL",
               "CIRCLE_L", "CIRCLE_R", "RETREAT", "APPROACH")

DT = 0.02                          # 50 Hz control period
BLEND_S = 0.6                      # element cross-fade (s)
TRIM_RAMP_S = 1.0                  # reference-trim ramp-in from reset (s)
POSE_LOOKAHEAD = 6                 # reference rows written ahead of the clock
SUBSTEPS = 10                      # physics substeps per control tick
ROW_CHUNK = 256                    # table growth chunk (rolling timeline)
#: floor for the drill's own stance width (trims.MIN_STANCE_WIDTH is the same
#: operator rule; the drill enforces it on the posture table)
MIN_WIDTH = MIN_STANCE_WIDTH

# --- step primitive defaults (measured in the report's step table) ----------
STEP_LIFT = 0.07                   # swing-foot clearance (m)
STEP_LOAD_S = 0.45                 # weight-transfer phase (s)
STEP_MOVE_S = 0.34                 # lift + transport (s)
STEP_LAND_S = 0.30                 # landing phase (s)
STEP_SETTLE_S = 0.35               # both feet loaded, shift unwound (s)
STEP_TIMEOUT_S = 4.0               # per-phase guard timeout (s)
STEP_LOAD_FRAC = 0.30              # swing-foot load fraction = unloaded
STEP_LAND_FORCE = 12.0             # N on the landed foot = contact confirmed
STEP_LEN_MAX = 0.16                # max swing-foot displacement per step (m)
STEP_CADENCE = 1.30                # s between automatic steps while moving


def _yaw_of(q: np.ndarray) -> float:
    """Base yaw (rad) of a single-robot qpos."""
    return float(np.arctan2(2.0 * (q[3] * q[6] + q[4] * q[5]),
                            1.0 - 2.0 * (q[5] ** 2 + q[6] ** 2)))


def _yaw_quat(a: float) -> np.ndarray:
    """(w, x, y, z) quaternion of a rotation ``a`` about +z."""
    return np.array([np.cos(0.5 * a), 0.0, 0.0, np.sin(0.5 * a)])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MuJoCo (w, x, y, z) quaternion product."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _point_in_patch_margin(patch: np.ndarray, point: np.ndarray) -> float:
    """Signed distance of ``point`` inside the sole patch hull (m)."""
    from .posture import point_in_hull_margin
    return point_in_hull_margin(np.asarray(patch, float), np.asarray(point, float))


@dataclass
class _Element:
    """One drill element: its posture parameters and its entry time."""
    skill: str = "STANCE_HOLD"
    height: float = 0.0
    lead: float = 0.0
    t_entry: float = 0.0


@dataclass
class StepPlan:
    """One physical step: which foot moves where, and the phase budget."""
    swing: int                  # 0 = left, 1 = right
    dx: float = 0.0             # forward displacement of the swing foot (m)
    dy: float = 0.0             # lateral displacement (+ = left, m)
    lift: float = STEP_LIFT
    base_dx: float | None = None    # reference base advance (default dx/2)
    base_dy: float | None = None    # (default dy/2)
    load_s: float = STEP_LOAD_S
    move_s: float = STEP_MOVE_S
    land_s: float = STEP_LAND_S
    settle_s: float = STEP_SETTLE_S
    timeout_s: float = STEP_TIMEOUT_S
    load_frac: float = STEP_LOAD_FRAC
    load_abs: float = 25.0            # N: swing foot counts as unloaded
    load_margin: float = 0.01         # m: CoM inside the support foot patch
    land_force: float = STEP_LAND_FORCE
    shift_frac: float = 1.0           # fraction of the support-foot offset
    why: str = ""


class StepPrimitive:
    """Sensor-guarded support transfer + foot placement (one step).

    Phases and guards (measured quantities, not timers alone):

    ``LOAD``      shift the *reference body* toward the support foot (the
                  planted legs are re-solved so their feet stay pinned) until
                  the measured load on the swing foot drops below
                  ``load_frac`` of its pre-step value (solver contact forces);
    ``LIFT``      raise the swing foot to ``lift`` (joint reference from a
                  leg-IK solution) until its lowest sole site clears the mat;
    ``TRANSPORT`` carry it to the landing position; the reference body
                  advances by half the step so the CoM ends between the feet;
    ``LAND``      lower it to the mat and wait for measured contact
                  (> ``land_force`` N);
    ``SETTLE``    unwind the lateral shift with both feet loaded.

    Every phase has a timeout; a guard that never fires aborts the step and is
    reported (``result == 'failed'``), never advanced silently.
    """

    PHASES = ("LOAD", "LIFT", "TRANSPORT", "LAND", "SETTLE")

    def __init__(self, drill: "SkillController", plan: StepPlan, t0: float):
        self.drill = drill
        self.model = drill.model
        self.plan = plan
        self.t0 = float(t0)
        self.phase = "LOAD"
        self.phase_t0 = float(t0)
        self.result = "running"
        self.notes: list[str] = []
        self.support = 1 - plan.swing
        self.load0 = None
        self._plan_swing(t0)

    # -- planning (offline IK for the swing-leg reference) -----------------

    def _plan_swing(self, t0: float) -> None:
        from .posture import solve_foot_ik, sole_positions
        d = self.drill
        q0 = d._posture_q()
        self.q0 = q0
        swing = self.plan.swing
        feet = sole_positions(self.model, d.prefix, q0)
        self.feet0 = {0: d._pin[0].copy(), 1: d._pin[1].copy()}
        p = self.plan
        p.base_dx = 0.5 * p.dx if p.base_dx is None else p.base_dx
        p.base_dy = 0.5 * p.dy if p.base_dy is None else p.base_dy
        base_xy, yaw = d._base_at(t0)
        c, s = np.cos(yaw), np.sin(yaw)
        delta = np.array([c * p.dx - s * p.dy, s * p.dx + c * p.dy, 0.0])
        z0 = float(self.feet0[swing][:, 2].mean())
        t_lift = self.feet0[swing].copy()
        t_lift[:, 2] = z0 + p.lift
        t_over = self.feet0[swing] + delta
        t_over[:, 2] = z0 + p.lift
        t_land = self.feet0[swing] + delta
        t_land[:, 2] = z0
        self.swing_targets = {"LIFT": t_lift, "TRANSPORT": t_over, "LAND": t_land}
        self.q_lift = solve_foot_ik(self.model, d.prefix, q0, swing, t_lift,
                                    iters=40)
        self.q_over = solve_foot_ik(self.model, d.prefix, q0, swing, t_over,
                                    iters=40)
        self.q_land = solve_foot_ik(self.model, d.prefix, q0, swing, t_land,
                                    iters=40)
        q = self.feet0[self.support].mean(axis=0)
        b = np.asarray(base_xy, float)
        desired = p.shift_frac * (q[:2] - b)          # world, over the support
        # the legs cannot hold a full-stance-width lateral shift with flat feet
        # (measured kinematic limit): ask only for what the IK can keep pinned
        from .posture import reachable_shift, sole_positions
        feet_pins = np.stack([self.feet0[0], self.feet0[1]])
        reach, res = reachable_shift(self.model, d.prefix, d._posture_q(),
                                     feet_pins, desired, float(np.linalg.norm(desired)))
        self.shift_limit = {"desired": float(np.linalg.norm(desired)),
                            "reachable": float(reach), "residual": float(res)}
        if np.linalg.norm(desired) > 1e-9:
            desired = desired * (min(1.0, reach / float(np.linalg.norm(desired))))
        self.shift_vec = desired

    # -- reference ---------------------------------------------------------

    def swing_joints(self, t_row: float) -> np.ndarray | None:
        """The swing leg's 6 reference joints at ``t_row`` (None while planted).

        During ``LOAD`` both feet are planted — the body shift is realised by
        re-solving both legs against their pins, so the swing leg is not driven
        from the plan yet.
        """
        p = self.plan
        if self.phase == "LOAD":
            return None
        if self.phase == "LIFT":
            u = _smoothstep((t_row - self.phase_t0) / max(0.5 * p.move_s, 1e-3))
            q = (1 - u) * self.q0 + u * self.q_lift
        elif self.phase == "TRANSPORT":
            u = _smoothstep((t_row - self.phase_t0) / max(0.5 * p.move_s, 1e-3))
            q = (1 - u) * self.q_lift + u * self.q_over
        elif self.phase == "LAND":
            u = _smoothstep((t_row - self.phase_t0) / max(p.land_s, 1e-3))
            q = (1 - u) * self.q_over + u * self.q_land
        else:
            q = self.q_land
        return q[7 + 6 * p.swing: 7 + 6 * p.swing + 6]

    def swing_foot(self, t_row: float) -> np.ndarray:
        """The swing foot's world sole targets (2, 3) at row time ``t_row``."""
        p = self.plan
        sw = p.swing
        if self.phase == "LOAD":
            return self.feet0[sw].copy()
        if self.phase == "LIFT":
            u = _smoothstep((t_row - self.phase_t0) / max(0.5 * p.move_s, 1e-3))
            return (1 - u) * self.feet0[sw] + u * self.swing_targets["LIFT"]
        if self.phase == "TRANSPORT":
            u = _smoothstep((t_row - self.phase_t0) / max(0.5 * p.move_s, 1e-3))
            return ((1 - u) * self.swing_targets["LIFT"]
                    + u * self.swing_targets["TRANSPORT"])
        if self.phase == "LAND":
            u = _smoothstep((t_row - self.phase_t0) / max(p.land_s, 1e-3))
            return ((1 - u) * self.swing_targets["TRANSPORT"]
                    + u * self.swing_targets["LAND"])
        return self.swing_targets["LAND"].copy()

    def base_offset(self, t_row: float) -> tuple[np.ndarray, np.ndarray]:
        """(lateral shift, forward advance) of the reference body (world xy)."""
        p = self.plan
        if self.phase == "LOAD":
            u = _smoothstep((t_row - self.phase_t0) / max(p.load_s, 1e-3))
        elif self.phase == "SETTLE":
            u = 1.0 - _smoothstep((t_row - self.phase_t0) / max(p.settle_s, 1e-3))
        else:
            u = 1.0
        lat = u * self.shift_vec
        if self.phase == "LOAD":
            adv_u = 0.0
        elif self.phase in ("LIFT", "TRANSPORT"):
            adv_u = 0.5 * _smoothstep((t_row - self.phase_t0)
                                      / max(p.move_s, 1e-3))
        else:
            adv_u = 1.0
        c, s = np.cos(self.drill._yaw), np.sin(self.drill._yaw)
        adv = adv_u * np.array([c * p.base_dx - s * p.base_dy,
                                s * p.base_dx + c * p.base_dy])
        return lat, adv

    def support_patch(self) -> np.ndarray:
        """(4, 2) corners of the support foot's sole patch (world, from the pin)."""
        from .posture import SOLE_HALF_LIFT, SOLE_HALF_WIDTH
        f = self.feet0[self.support]
        pts = []
        for s in range(2):
            pts.append([f[s, 0] - SOLE_HALF_LIFT, f[s, 1] - SOLE_HALF_WIDTH])
            pts.append([f[s, 0] + SOLE_HALF_LIFT, f[s, 1] + SOLE_HALF_WIDTH])
        return np.array(pts)

    def commit_xy(self) -> np.ndarray:
        """The reference-base advance this step hands to the drill (world)."""
        p = self.plan
        c, s = np.cos(self.drill._yaw), np.sin(self.drill._yaw)
        return np.array([c * p.base_dx - s * p.base_dy,
                         s * p.base_dx + c * p.base_dy])

    # -- runtime -----------------------------------------------------------

    def tick(self, data: mujoco.MjData, t: float) -> str:
        """Advance the state machine with measured sensor guards."""
        p = self.plan
        d = self.drill
        ctx = d.robot.ctx
        loads = st.foot_load(self.model, data, ctx)
        swing = p.swing
        if self.load0 is None:
            self.load0 = float(max(loads.sum(), 1.0))
        elapsed = t - self.phase_t0
        if elapsed > p.timeout_s and self.phase not in ("SETTLE",):
            self.result = "failed"
            self.notes.append(f"{self.phase} timeout at {elapsed:.2f}s")
            return self.phase
        if self.phase == "LOAD":
            frac = float(loads[swing] / max(loads.sum(), 1e-6))
            com = data.subtree_com[ctx.pelvis]
            # the transfer is complete when the swing foot's *measured* load has
            # collapsed to the target fraction and the reach-limited shift ramp
            # is done.  The capture point is reported (the G1's 7 cm foot patch
            # and the leg reach do not allow a full single-support transfer from
            # a wide stance; see the report's step table)
            xi = com[:2] + 0.3 * np.sqrt(max(com[2] - 0.05, 0.30) / 9.81) \
                * self.drill.robot._v
            self.last_margin = _point_in_patch_margin(self.support_patch(), xi)
            if (frac < p.load_frac and elapsed >= 0.9 * p.load_s):
                self._enter("LIFT", t, data)
        elif self.phase == "LIFT":
            if st.foot_site_z(data, ctx, swing) > FOOT_Z_TOUCH and elapsed >= 0.15:
                self._enter("TRANSPORT", t, data)
        elif self.phase == "TRANSPORT":
            want = self.swing_targets["TRANSPORT"][:, :2].mean(axis=0)
            cur = np.array([data.site_xpos[ctx.site[s]][:2]
                            for s in ctx.foot_sites[swing]]).mean(axis=0)
            if float(np.linalg.norm(want - cur)) < 0.05 and elapsed >= 0.15:
                self._enter("LAND", t, data)
        elif self.phase == "LAND":
            if loads[swing] > p.land_force and elapsed >= 0.10:
                self._enter("SETTLE", t, data)
        elif self.phase == "SETTLE":
            if elapsed >= p.settle_s:
                self._enter("DONE", t, data)
        return self.phase

    def _enter(self, phase: str, t: float, data: mujoco.MjData | None = None) -> None:
        self.phase = phase
        self.phase_t0 = float(t)
        if phase == "LIFT" and data is not None:
            # re-resolve the swing-leg trajectory now that the body has shifted
            # over the support foot: seeding from the live leg joints (the
            # swing foot is still planted at its pin, unloaded) makes the
            # reference continuous with what the robot is actually doing
            self._resolve_swing(data)
        if phase == "DONE":
            self.result = "done"

    def _resolve_swing(self, data: mujoco.MjData) -> None:
        from .posture import solve_foot_ik
        d = self.drill
        ctx = d.robot.ctx
        self.q0 = np.asarray(data.qpos[ctx.qpos_slice], float).copy()
        sw = self.plan.swing
        self.q_lift = solve_foot_ik(self.model, d.prefix, self.q0, sw,
                                    self.swing_targets["LIFT"], iters=60)
        self.q_over = solve_foot_ik(self.model, d.prefix, self.q_lift, sw,
                                    self.swing_targets["TRANSPORT"], iters=60)
        self.q_land = solve_foot_ik(self.model, d.prefix, self.q_over, sw,
                                    self.swing_targets["LAND"], iters=60)

    def status(self) -> dict:
        return {"swing": self.plan.swing, "phase": self.phase,
                "result": self.result, "notes": list(self.notes),
                "t0": self.t0, "dx": self.plan.dx, "dy": self.plan.dy,
                "why": self.plan.why}


class SkillController:
    """Procedural-pose + foot-placement driver on top of the robot teacher."""

    def __init__(self, model: mujoco.MjModel, stance: np.ndarray,
                 stand: np.ndarray | None = None, technique: str = "STANCE",
                 flags: TeacherFlags | None = None,
                 stance_b: np.ndarray | None = None, prefix: str | None = None,
                 min_width: float | None = None):
        self.model = model
        self.flags = flags or TeacherFlags()
        self.prefix = prefix if prefix is not None else ("a_" if stance_b is not None else "")
        self.stance = np.asarray(stance, float)[:36].copy()
        self.stance_b = (np.asarray(stance_b, float)[:36].copy()
                         if stance_b is not None else None)
        if stand is None:
            kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
            if kid < 0:
                kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "both_stand")
            stand = np.array(model.key_qpos[kid][:36], float)
        self.stand = np.asarray(stand, float)[:36].copy()
        self.technique = technique
        self.trim = trim_vector(trim_for(technique, self.prefix))
        # operator rule (frontal axis): the posture table enforces a stance
        # WIDTH MINIMUM; measured by FK on the trimmed stance and applied as a
        # hip_roll splay inside the trim (so it ramps in with it).  ``min_width``
        # overrides it (the F1/F2 ablation rows reproduce the pre-fix setup,
        # which had no width enforcement)
        width_min = MIN_WIDTH if min_width is None else float(min_width)
        self.width_min = width_min
        self.width0 = {}
        self.width = {}
        q = self.stance.copy()
        q[7:36] = q[7:36] + self.trim
        self.width0[self.prefix] = measure_stance_width(model, q, self.prefix)
        _, splay, w = enforce_stance_width(model, q, width_min, prefix=self.prefix)
        self.trim = self.trim + splay
        self.width[self.prefix] = w
        # procedural state
        self._yaw0 = _yaw_of(self.stance)
        self._yaw = self._yaw0
        self._xy = self.stance[:2].copy()
        self._cmd = np.zeros(3)
        self._cmd_cur = np.zeros(3)
        self._elem = _Element()
        self._prev: _Element | None = None
        self._t_switch = 0.0
        self._t_pose = 0.0
        self._t0 = 0.0
        self._rows = None
        self.robot: RobotTeacher | None = None
        self.partner: RobotTeacher | None = None
        self._pin = {0: None, 1: None}        # world sole targets (2,3) per foot
        self._step: StepPrimitive | None = None
        self._step_history: list[dict] = []
        self._next_step_t = 0.0
        self._support_foot: int | None = None
        self._ik_seed: np.ndarray | None = None

    # -- scheduler API ----------------------------------------------------

    def reset(self, qpos: np.ndarray, t0: float = 0.0,
              qpos_partner: np.ndarray | None = None,
              prefix: str | None = None):
        """Start/restart the drill from a *live* state (no sim reset).

        ``qpos`` is the live 36-dof qpos of the drill robot; the reference
        base xy/heading is anchored there and the feet are pinned where the
        robot is actually standing, so the executor starts coincident.
        ``prefix`` is accepted for backwards compatibility (the robot context
        is fixed by the constructor).
        """
        from .posture import sole_positions
        self._t0 = float(t0)
        self._t_pose = 0.0
        q = np.asarray(qpos, float)
        self._yaw = _yaw_of(q)
        self._xy = q[:2].copy()
        self._cmd[:] = 0.0
        self._cmd_cur[:] = 0.0
        self._elem = _Element()
        self._prev = None
        self._t_switch = 0.0
        self._step = None
        self._support_foot = None
        self._next_step_t = 0.0
        # pin the feet at their live positions (world)
        feet = sole_positions(self.model, self.prefix, q)
        self._pin = {0: feet[0].copy(), 1: feet[1].copy()}
        self._ik_seed = q[7:19].copy()
        t_ref = np.arange(ROW_CHUNK) * DT
        self._build(t_ref, qpos_partner)

    def _build(self, t_ref: np.ndarray, qpos_partner: np.ndarray | None = None):
        rows = np.tile(self._pose_at(0.0), (len(t_ref), 1))
        self._rows = rows.copy()
        self._ik_seed = None
        if self.robot is None:
            self.robot = RobotTeacher(self.model, self.prefix, rows, t_ref,
                                      technique="", flags=self.flags)
        else:                    # re-arm: drop the grown tables, keep feedback
            self.robot.ctx.reset_ref(self.model, rows, t_ref)
        if self.stance_b is not None or qpos_partner is not None:
            prows = np.tile(self._partner_pose_at(0.0), (len(t_ref), 1))
            if self.partner is None:
                self.partner = RobotTeacher(self.model, "b_", prows, t_ref,
                                            technique="", flags=self.flags)
            else:
                self.partner.ctx.reset_ref(self.model, prows, t_ref)

    def set_skill(self, skill: str):
        if skill not in SKILLS:
            raise ValueError(f"unknown skill {skill!r}; expected one of {SKILLS}")
        if skill != self._elem.skill:
            self._switch(_Element(skill, self._elem.height, self._elem.lead,
                                  self._t_pose))

    def set_command(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0):
        self._cmd[:] = (float(vx), float(vy), float(wz))

    def set_stance_height(self, frac: float):
        """0 = reference stance, -1 = fully lowered (knees/ankles, no waist)."""
        v = float(np.clip(frac, -1.0, 1.0))
        if v != self._elem.height:
            self._switch(_Element(self._elem.skill, v, self._elem.lead,
                                  self._t_pose))

    def set_lead_step(self, frac: float):
        """Lead-leg penetration step depth (0..1) — SHOT_DOUBLE_LEG / KNEE_LOWER."""
        v = float(np.clip(frac, 0.0, 1.0))
        if v != self._elem.lead:
            self._switch(_Element(self._elem.skill, self._elem.height, v,
                                  self._t_pose))

    def _switch(self, new: _Element) -> None:
        """Install a new element and start its cross-fade from the old one."""
        self._prev = self._elem
        self._elem = new
        self._t_switch = self._t_pose

    # -- step primitive API ------------------------------------------------

    def start_step(self, swing: int, dx: float = 0.0, dy: float = 0.0,
                   why: str = "", **kw) -> dict:
        """Begin one sensor-guarded step with the given foot."""
        if self._step is not None and self._step.result == "running":
            return self.step_status()
        plan = StepPlan(swing=int(swing), dx=float(dx), dy=float(dy),
                        why=why or "manual", **kw)
        self._step = StepPrimitive(self, plan, self._t_pose)
        # explicit support mode: the balance layer's support becomes the
        # *loaded* foot (measured-contact decision, not a guess from height)
        self.robot.ctx.support_foot = 1 - plan.swing
        self.robot.ctx.step_mode = True
        self._support_foot = 1 - plan.swing
        return self.step_status()

    def step_status(self) -> dict:
        return {} if self._step is None else self._step.status()

    def _finish_step(self) -> None:
        """Hand a finished/failed step's outcome back to the drill state."""
        stp = self._step
        self._step_history.append(stp.status())
        if stp.result == "done":
            self._pin[stp.plan.swing] = stp.swing_targets["LAND"].copy()
            self._xy = self._xy + stp.commit_xy()   # body advanced over the feet
            self._support_foot = stp.plan.swing    # the landed foot is loaded
        self.robot.ctx.support_foot = None
        self.robot.ctx.step_mode = False
        self._step = None

    def _maybe_auto_step(self, t: float) -> None:
        """Movement skills own the stepping loop: issue the next step when the
        previous one finishes and the command asks for motion."""
        if self._elem.skill not in MOVE_SKILLS:
            return
        if self._step is not None and self._step.result == "running":
            return
        vx, vy, wz = self._cmd_cur
        if abs(vx) < 0.005 and abs(vy) < 0.005 and abs(wz) < 0.05:
            return
        if t < self._next_step_t:
            return
        # alternate: the foot that is NOT currently loaded swings next
        support = self._support_foot if self._support_foot is not None else 0
        swing = 1 - support
        dx = float(np.clip(vx * STEP_CADENCE, -STEP_LEN_MAX, STEP_LEN_MAX))
        dy = float(np.clip(vy * STEP_CADENCE, -STEP_LEN_MAX, STEP_LEN_MAX))
        if abs(wz) >= 0.05:                      # pivot step: turn the feet
            theta = float(np.clip(wz * STEP_CADENCE, -0.5, 0.5))
            loc = self._foot_local(swing)
            c, s = np.cos(theta), np.sin(theta)
            tgt = np.array([c * loc[0] - s * loc[1], s * loc[0] + c * loc[1]])
            dx += float(tgt[0] - loc[0])
            dy += float(tgt[1] - loc[1])
        self.start_step(swing, dx, dy, why=f"auto:{self._elem.skill}")
        self._next_step_t = t + STEP_CADENCE

    def _foot_local(self, foot: int) -> np.ndarray:
        """Planted foot's position in the reference base frame (2,)."""
        pin = self._pin[foot]
        xy, yaw = self._base_at(self._t_pose)
        c, s = np.cos(yaw), np.sin(yaw)
        d = pin[:, :2].mean(axis=0) - xy
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    # -- control tick -----------------------------------------------------

    def control(self, data: mujoco.MjData, t: float) -> np.ndarray:
        """Joint targets for this tick; advances the pose clock.

        Call once per 50 Hz control tick and hold the target for the ten
        physics substeps (``drill_step``).
        """
        if self.robot is None:
            raise RuntimeError("call reset() before control()")
        dt = float(np.clip(t - self._t0 - self._t_pose, 0.0, 0.1))
        a = min(1.0, dt / BLEND_S)
        self._cmd_cur += a * (self._cmd - self._cmd_cur)
        vx, vy, wz = self._cmd_cur
        c, s = np.cos(self._yaw), np.sin(self._yaw)
        self._xy += dt * np.array([c * vx - s * vy, s * vx + c * vy])
        if self._step is None or self._step.result != "running":
            self._yaw += dt * wz      # while stepping, the pivot is the step's
        self._t_pose += dt
        # --- step state machine (guards from measured contact forces) -----
        if self._step is not None:
            if self._step.result == "running":
                self._step.tick(data, self._t_pose)
            if self._step.result in ("done", "failed"):
                self._finish_step()
        self._maybe_auto_step(self._t_pose)
        # --- write the rolling reference window ---------------------------
        k = int(self._t_pose / DT)
        hi = k + POSE_LOOKAHEAD
        self._ensure_rows(hi + 1)
        rows = np.arange(k, hi + 1)
        t_rows = rows * DT
        qa, feet, touch, seed = [], [], [], self._ik_seed
        for tr in t_rows:
            q, f, tc, seed = self._robot_row(tr, seed)
            qa.append(q)
            feet.append(f)
            touch.append(tc)
        self._ik_seed = seed
        qa = np.stack(qa)
        feet = np.stack(feet)
        touch = np.stack(touch)
        self._rows[rows] = qa
        self.robot.update_reference(rows, qa, feet=feet, touch=touch)
        ctrl = self.robot.control(data, self._t_pose)
        if self.partner is not None:
            tb = np.stack([self._partner_pose_at(tr) for tr in t_rows])
            self.partner.update_reference(rows, tb)
            out = np.empty(self.model.nu)
            out[self.robot.act_ids] = ctrl
            out[self.partner.act_ids] = self.partner.control(data, self._t_pose)
            return out
        return ctrl

    def _ensure_rows(self, n: int) -> None:
        """Grow the reference tables on demand (rolling timeline, no cap)."""
        if self._rows is not None and len(self._rows) >= n:
            return
        add = max(ROW_CHUNK, n - (0 if self._rows is None else len(self._rows)))
        if self._rows is None:
            self._rows = np.tile(self._pose_at(self._t_pose), (add, 1))
        else:
            self._rows = np.vstack([self._rows,
                                    np.tile(self._rows[-1], (add, 1))])
        for r in (self.robot, self.partner):
            if r is not None:
                r.ctx.grow_ref(len(self._rows))

    # -- procedural reference --------------------------------------------

    def _base_at(self, t_row: float) -> tuple[np.ndarray, float]:
        """Reference base (xy, yaw) of the drill robot at row time ``t_row``."""
        d = t_row - self._t_pose
        vx, vy, wz = self._cmd_cur
        c, s = np.cos(self._yaw), np.sin(self._yaw)
        xy = self._xy + d * np.array([c * vx - s * vy, s * vx + c * vy])
        yaw = self._yaw + d * wz
        if self._step is not None:
            lat, adv = self._step.base_offset(t_row)
            xy = xy + lat + adv
        return xy, yaw

    def _posture_joints(self, elem: _Element, t_row: float) -> tuple[np.ndarray, float]:
        """(joints29, pelvis-z offset) of one element's posture at ``t_row``.

        The joints are the IK *seeds* (nominal posture); the planted-foot IK in
        ``_robot_row`` re-solves them so the feet stay on their pins.  Elements
        therefore act through the body (height, lean, recovery blend) and
        through foot placement — not through open-loop joint offsets that the
        contact geometry would ignore.
        """
        j = self.stance[7:36] + self.trim * min(1.0, max(t_row, 0.0) / TRIM_RAMP_S)
        dz = 0.0
        h = elem.height
        if h < 0.0:                       # LEVEL_CHANGE: knees, not waist
            dz -= 0.18 * (-h)
        if elem.skill in ("SHOT_DOUBLE_LEG", "KNEE_LOWER"):
            dz -= 0.22 * elem.lead
        if elem.skill == "RECOVER_STAND":     # relative to ITS entry, not t=0
            w = min(1.0, max(t_row - elem.t_entry, 0.0) / 2.0)
            j = (1 - w) * j + w * self.stand[7:36]
            dz = (1 - w) * dz
        return j, dz

    def _blended_posture(self, t_row: float) -> tuple[np.ndarray, float]:
        """Posture at ``t_row`` with the element cross-fade applied."""
        j, dz = self._posture_joints(self._elem, t_row)
        if self._prev is not None:
            w = _smoothstep((t_row - self._t_switch) / BLEND_S)
            if w >= 1.0:
                self._prev = None
            else:
                j0, dz0 = self._posture_joints(self._prev, t_row)
                j = (1 - w) * j0 + w * j
                dz = (1 - w) * dz0 + w * dz
        return j, dz

    def _robot_row(self, t_row: float, seed: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(qpos36, feet(2,2,3), touch(2,2), leg-seed) of the drill robot.

        The planted legs are IK-solved so their sole sites sit exactly on the
        world pins given the reference base pose — the reference is
        *kinematically consistent with the feet it claims to have*, so the
        feed-forward joints and the runtime foot IK agree instead of fighting.
        The solve is marched across the window rows with a warm start (one
        damped-Newton step per row), which keeps it at ~0.1 ms/row.
        """
        from .posture import solve_ik
        j, dz = self._blended_posture(t_row)
        p = self.stance.copy()
        p[7:36] = j
        xy, yaw = self._base_at(t_row)
        p[:2] = xy
        p[2] = self.stance[2] + dz
        p[3:7] = _quat_mul(_yaw_quat(yaw - self._yaw0), self.stance[3:7])
        p[3:7] /= max(float(np.linalg.norm(p[3:7])), 1e-9)
        feet = np.stack([self._pin[0], self._pin[1]])
        planted = [0, 1]
        if self._step is not None and self._step.result == "running":
            swing = self._step.plan.swing
            feet[swing] = self._step.swing_foot(t_row)
            sj = self._step.swing_joints(t_row)
            if sj is not None:                 # airborne: use the plan's joints
                p[7 + 6 * swing: 7 + 6 * swing + 6] = sj
                planted = [1 - swing]
            else:                              # LOAD: still planted, IK it
                planted = [0, 1]
        if seed is not None:
            for f in planted:
                p[7 + 6 * f: 7 + 6 * f + 6] = seed[6 * f: 6 * f + 6]
        p = solve_ik(self.model, self.prefix, p, feet, feet=tuple(planted),
                     iters=6, reg=1e-4, step_max=0.35)
        touch = feet[:, :, 2] < FOOT_Z_TOUCH
        return p, feet, touch, p[7:19].copy()

    def _posture_q(self) -> np.ndarray:
        """Current reference posture (36,) of the drill robot (with foot IK)."""
        p, _, _, seed = self._robot_row(self._t_pose, self._ik_seed)
        self._ik_seed = seed
        return p

    def _pose_at(self, t_row: float) -> np.ndarray:
        """The reference pose (36,) at ``t_row`` (no seed marching)."""
        return self._robot_row(t_row)[0]

    def _partner_pose_at(self, t_row: float) -> np.ndarray:
        """Partner robot's pose: its own stance (a bystander, no commands)."""
        p = self.stance_b.copy()
        p[3:7] = p[3:7] / max(float(np.linalg.norm(p[3:7])), 1e-9)
        return p

    # -- self-check ------------------------------------------------------

    def run_self_check(self, script=None, settle: float = 0.0):
        """Drive the drill robot through a scripted sequence at 50 Hz.

        Returns one row per element with the proximity metrics (min/end pelvis
        height, up fraction, xy drift, tilt, topple time, steps taken).
        """
        model = self.model
        data = mujoco.MjData(model)
        self.reset(self.stance)                 # builds the executor + pins
        sa = self.robot.ctx.qpos_slice
        data.qpos[sa] = self.stance
        if self.partner is not None:
            data.qpos[self.partner.ctx.qpos_slice] = self.stance_b
        mujoco.mj_forward(model, data)
        if script is None:
            script = DEFAULT_SCRIPT
        substeps = int(round(DT / model.opt.timestep))
        out, t = [], 0.0
        tilt_bid = model.body(f"{self.prefix}torso_link").id
        if settle:
            for _ in range(int(settle / DT)):
                t = drill_step(model, data, self, t)
        for elem in script:
            skill, cmd, height, lead, dur = elem
            self.set_skill(skill)
            self.set_command(**cmd)
            self.set_stance_height(height)
            self.set_lead_step(lead)
            t_start, samples, t_fall = t, [], None
            n_steps0 = len(self._step_history)
            nxt = t
            while t < t_start + dur - 1e-9:
                data.ctrl[:] = self.control(data, t)
                for _ in range(substeps):
                    mujoco.mj_step(model, data)
                    t += model.opt.timestep
                if t >= nxt:
                    nxt = t + 0.1
                    z = float(data.qpos[sa][2])
                    up = data.xmat[tilt_bid].reshape(3, 3)[:, 2]
                    samples.append((t, z, float(data.qpos[sa][0]),
                                    float(data.qpos[sa][1]),
                                    float(np.degrees(np.arccos(
                                        np.clip(up[2], -1.0, 1.0))))))
                    if t_fall is None and z < 0.45:
                        t_fall = t
            arr = np.array(samples) if samples else np.zeros((1, 5))
            z = arr[:, 1]
            drift = float(np.hypot(arr[-1, 2] - arr[0, 2], arr[-1, 3] - arr[0, 3]))
            done = [s for s in self._step_history[n_steps0:]]
            out.append({
                "skill": skill, "t_start": float(t_start),
                "min_pelvis_z": float(z.min()), "end_pelvis_z": float(z[-1]),
                "up_frac": float((z >= 0.45).mean()),
                "xy_drift": drift,
                "max_tilt_deg": float(arr[:, 4].max()),
                "t_fall": (None if t_fall is None else float(t_fall - t_start)),
                "steps": len(done),
                "steps_failed": sum(1 for s in done if s["result"] == "failed"),
            })
        return out

    def run_all_skills(self, dur: float = 2.5, transition_dur: float = 1.0):
        """Per-skill sweep: each of the 12 skills from a live STANCE_HOLD."""
        script = []
        for skill in SKILLS:
            script.append((skill, SKILL_CMDS[skill], SKILL_HEIGHT[skill],
                           SKILL_LEAD[skill], dur))
            script.append(("STANCE_HOLD", {}, 0.0, 0.0, transition_dur))
        return self.run_self_check(script=script)


#: leg joint indices in the 29-vector (per foot), for the swing-leg override
_LEG_IDX = {0: np.arange(0, 6), 1: np.arange(6, 12)}


#: scripted self-check: the drill elements of the operator brief, back to back
DEFAULT_SCRIPT = [
    ("STANCE_HOLD", dict(vx=0, vy=0, wz=0), 0.0, 0.0, 2.0),
    ("SHUFFLE_FORWARD", dict(vx=0.08), 0.0, 0.0, 2.0),
    ("LEVEL_CHANGE", dict(), -1.0, 0.0, 1.5),
    ("KNEE_LOWER", dict(), -1.0, 0.7, 1.0),
    ("RECOVER_STAND", dict(), 0.0, 0.0, 2.0),
]

#: per-skill command / posture for the 12-skill sweep (see SKILLS)
SKILL_CMDS = {
    "STANCE_HOLD": dict(),
    "SHUFFLE_FORWARD": dict(vx=0.08),
    "SHUFFLE_BACK": dict(vx=-0.08),
    "SHUFFLE_LATERAL": dict(vy=0.08),
    "CIRCLE_L": dict(wz=0.25),
    "CIRCLE_R": dict(wz=-0.25),
    "RETREAT": dict(vx=-0.06),
    "APPROACH": dict(vx=0.08),
    "LEVEL_CHANGE": dict(),
    "SHOT_DOUBLE_LEG": dict(),
    "KNEE_LOWER": dict(),
    "RECOVER_STAND": dict(),
}
SKILL_HEIGHT = {s: 0.0 for s in SKILLS} | {
    "LEVEL_CHANGE": -1.0, "SHOT_DOUBLE_LEG": -0.5, "KNEE_LOWER": -1.0}
SKILL_LEAD = {s: 0.0 for s in SKILLS} | {
    "SHOT_DOUBLE_LEG": 0.7, "KNEE_LOWER": 1.0}


def drill_step(model: mujoco.MjModel, data: mujoco.MjData,
               ctrl: "SkillController", t: float,
               substeps: int | None = None) -> float:
    """One 50 Hz control tick: call once, then hold the target for the
    physics substeps between control ticks (the ``run_episode`` cadence)."""
    if substeps is None:
        substeps = int(round(DT / model.opt.timestep))
    data.ctrl[:] = ctrl.control(data, t)
    for _ in range(substeps):
        mujoco.mj_step(model, data)
    return t + substeps * model.opt.timestep


if __name__ == "__main__":                       # self-check
    import sys
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO / "src"))
    from teacher.solo_scene import load_single_model, stand_qpos
    from teacher.posture import build_stance

    model = load_single_model()
    stand = stand_qpos(model)
    stance, m = build_stance(model, "", stand, width=0.34, depth=0.24, drop=0.0)
    ctrl = SkillController(model, stance, stand=stand)
    print(f"stance: width={m['width']:.3f} depth={m['depth']:.3f} "
          f"pelvis_z={m['pelvis_z']:.3f} margin={m['margin']:+.3f}")
    if "--all" in sys.argv:
        rows = ctrl.run_all_skills()
    elif "--step" in sys.argv:
        rows = ctrl.run_self_check(script=[
            ("STANCE_HOLD", dict(), 0.0, 0.0, 2.0),
        ])
    else:
        rows = ctrl.run_self_check()
    for row in rows:
        print(f"{row['skill']:16s} up_frac={row['up_frac']:.2f} "
              f"min_z={row['min_pelvis_z']:.3f} end_z={row['end_pelvis_z']:.3f} "
              f"xy_drift={row['xy_drift']:.3f} tilt={row['max_tilt_deg']:.1f} "
              f"t_fall={row['t_fall']} steps={row['steps']}")
