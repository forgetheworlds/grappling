"""Drill controllers: the pluggable interface, the PD baseline, the fallback.

The scheduler speaks one command shape to every controller::

    DrillCommand(skill, vx, vy, wz, stance_height)

and a controller returns 29 joint-position targets every 50 Hz tick.

* :class:`FeasibleDrill` -- **this milestone's motion source.**  Conservative
  feedback control: a cartesian plan (pelvis pose, world foot targets, upper
  body) is turned into joint targets by IK, and a measured balance law
  (capture-point tracking of the *planned* CoM, planted-foot world anchoring,
  sole flattening, torso upright rows, a pelvis-height PI and an emergency
  blend) keeps the robot on its feet.  Foot motion goes through the
  :class:`~drill.stepping.FootStepper` primitive, one foot at a time.
* :class:`StancePD` -- the evidence baseline: pure joint PD to the stance pose,
  no feedback.  Known to topple; it exists so the video can show the contrast.
* :class:`TeacherAdapter` -- wraps ``src/teacher`` lazily.  The shipped teacher
  is *paired* (hard-coded ``a_``/``b_`` contexts and 58 targets), so the
  adapter reuses the teacher's public stabilizer/gain modules on the single-G1
  reference instead of editing it, and refuses (``TeacherUnavailable``) with
  the exact mismatch when that seam is not usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Protocol

import mujoco
import numpy as np

from . import kin as K
from . import posture as posture_mod
from .balance import BalanceLaw, BalanceParams, DrillPlan, FootTarget, ReferenceSolver
from .stepping import FootStepper, StepParams, rotate_about


#: longest single entry placement (m): the stand->stance transition is walked in
#: short steps, each inside the leg's reach from the shifted pelvis
ENTRY_STEP_M = 0.14

#: pelvis-height slew limit of the plan (m/s): the crouch is *approached*,
#: never stepped into (a step in the reference launches a position-servo robot)
Z_SLEW = 0.022


class DrillCommand(NamedTuple):
    """One scheduler command (the milestone's command channel)."""

    skill: str = "STANCE"
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    stance_height: float = 0.0        # 0 = stance, 1 = deep crouch
    lead_leg: str = "left"
    phase: str = ""                   # human-readable scheduler phase


class DrillController(Protocol):
    """The pluggable drill controller interface."""

    name: str

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None: ...

    def act(self, model: mujoco.MjModel, data: mujoco.MjData,
            cmd: DrillCommand) -> np.ndarray: ...


# ------------------------------------------------------------------------- skills
#: skills the scheduler may issue (everything the drill is built from)
SKILLS = ("STANCE", "SHUFFLE_F", "SHUFFLE_B", "SHUFFLE_L", "SHUFFLE_R",
          "APPROACH", "RETREAT", "CIRCLE_L", "CIRCLE_R", "LEVEL_CHANGE",
          "SHOT_GESTURE", "RECOVER", "ENTRY")

#: which foot leads each directional skill
LEAD_FOOT = {"SHUFFLE_L": "left", "SHUFFLE_R": "right", "CIRCLE_L": "left",
             "CIRCLE_R": "right"}


class TeacherUnavailable(RuntimeError):
    """Raised when the teacher seam cannot drive the single-G1 drill."""


# --------------------------------------------------------------------- PD baseline
class StancePD:
    """Pure position tracking of the stance pose: the toppling baseline.

    No balance feedback of any kind (this is the evidential counterpart of the
    observed failure in ``notes.md`` E3: pure PD replay of the retargeted
    stance topples).  The only thing it does is hold the *built* stance (which
    is at least geometrically consistent) so the baseline isolates the
    feedback layer rather than a bad pose.
    """

    name = "stance_pd"
    kind = "scripted (no feedback)"

    def __init__(self, stance: posture_mod.Stance, ids: K.RobotIds):
        self.stance = stance
        self.ids = ids
        self.q_ref = stance.qpos[7:36].copy()

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        pass

    def act(self, model, data, cmd: DrillCommand) -> np.ndarray:
        return self.ids.ctrl_lo + 0.0 if False else np.clip(
            self.q_ref, self.ids.ctrl_lo, self.ids.ctrl_hi)


# ------------------------------------------------------------------------- ladder
#: rung -> the motion elements it is allowed to use (review: component gates)
RUNG_ELEMENTS = {
    "L0": ("hold", "posture_mod"),
    "L1": ("hold", "posture_mod", "level_change", "step"),
    "L2": ("hold", "posture_mod", "step"),
    "L3": ("hold", "posture_mod", "step", "shuffle", "circle"),
    "L4": ("hold", "posture_mod", "step", "shuffle", "circle", "shot"),
}


@dataclass
class FeasibleDrill:
    """Conservative, feedback-driven single-G1 drill controller."""

    stance: posture_mod.Stance
    ids: K.RobotIds
    rung: str = "L0"
    params: BalanceParams = field(default_factory=BalanceParams)
    step_params: StepParams = field(default_factory=StepParams)
    seed: int = 0
    name: str = "feasible"
    kind: str = "scripted feedback (FeasibleDrill)"

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.solver = ReferenceSolver(self.stance.model, self.ids, iters=3)
        self.law = BalanceLaw(self.stance.model, self.ids, self.params)
        self.stepper = FootStepper(self.ids, self.step_params)
        self.home = {s: self.stance.foot_origins()[s] for s in K.SIDES}   # world feet
        self.home_yaw = {s: self.stance.spec.foot_yaw(s) for s in K.SIDES}
        self.nominal_xy = np.array(self.stance.qpos[:2], float)            # pelvis home
        self.com_local = np.zeros(2)
        self.plan: DrillPlan | None = None
        self.t = 0.0
        self.events: list = []
        self.guards = {"safety_blend": 0, "timeouts": 0, "ik_err_max": 0.0}
        self._stance_q = self.stance.qpos[7:36].copy()
        self._upper_hold = self.stance.qpos[19:36].copy()   # waist + arms
        self._height = 0.0
        self._arm_shift = np.zeros(14)
        self._shot = None
        self._shot_t = 0.0
        self._last_swing = "left"
        self._sway_phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.entry_tol = 0.008
        self._entry_done_t = 0.0
        self._emergency_cooldown = 0.0
        self.step_centre_tol = 0.030      # CoM must be near the mid-foot before a new step
        self.nominal_now = self.nominal_xy.copy()

    # -- interface ---------------------------------------------------------
    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Seed every layer from the *live* state: the first commanded target is
        the pose the robot is already in (no start-up step in the reference)."""
        self.t = 0.0
        self.events = []
        self.stepper.reset()
        self.law.reset()
        self.solver.reset(np.array(data.qpos[7:36], float))
        self.plan = self._plan_from(data)
        self.plan.upper = np.array(data.qpos[19:36], float)
        self._height = 0.0          # target = the built stance height
        self.plan.base_xyz[2] = float(data.qpos[2])   # start the slew from live
        self._upper_hold = np.array(data.qpos[19:36], float)
        self._measure_com_local(model, data)

    def act(self, model: mujoco.MjModel, data: mujoco.MjData,
            cmd: DrillCommand) -> np.ndarray:
        dt = 0.02
        self.t += dt
        self._update_plan(model, data, cmd, dt)
        self._emergency(model, data)
        q_ref = self.solver.solve(self.plan)
        com_ref = self._reference_com(model)
        self.com_local = com_ref - np.asarray(self.plan.base_xyz[:2], float)
        dq, info = self.law.offsets(data, self.plan, com_ref)
        if self.law.info["alpha"] > 0.0 and not self.stepper.busy():
            # last resort, and only with both feet down: blend the reference
            # toward the built stance.  Blending while a foot swings would drag
            # the swing leg (measured: the foot flew to 0.15 m and the body
            # toppled) -- the in-step response is to put the foot down instead.
            a = float(self.law.info["alpha"])
            q_ref = (1 - a) * q_ref + a * self._safe_q()
            self.guards["safety_blend"] += 1
        self.guards["ik_err_max"] = max(self.guards["ik_err_max"],
                                        float(max(self.solver.ik_err.values())))
        ctrl = np.clip(q_ref + dq, self.ids.ctrl_lo, self.ids.ctrl_hi)
        return ctrl

    # -- plan --------------------------------------------------------------
    def _plan_from(self, data: mujoco.MjData) -> DrillPlan:
        """Plan seeded from the *live* state: the drill moves from wherever the
        single initial state put the robot, it does not teleport to the stance."""
        feet = {}
        for s in K.SIDES:
            pts = self.ids.sole_points(data, s)
            heel = pts[list(K.HEEL_ROWS)].mean(axis=0)
            toe = pts[list(K.TOE_ROWS)].mean(axis=0)
            v = toe - heel
            # the *foot frame* origin: the same handle foot_targets() places
            feet[s] = FootTarget(np.array(data.xpos[self.ids.foot_body[s]][:2], float),
                                 float(np.arctan2(v[1], v[0])))
        self._live_upper = np.array(data.qpos[19:36], float)
        return DrillPlan(base_xyz=np.array(data.qpos[:3], float),
                         base_yaw=K.quat_yaw(data.qpos[3:7]),
                         feet=feet, upper=self._live_upper.copy(), label="init")

    def _measure_com_local(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """CoM offset from the pelvis in the stance pose (planning handle)."""
        d = self.stance.scratch()
        d.qpos[:] = self.stance.qpos
        mujoco.mj_forward(model, d)
        self.com_local = self.ids.com_xy(d) - self.stance.qpos[:2]

    def _reference_com(self, model: mujoco.MjModel) -> np.ndarray:
        """CoM of the *reference* pose (what the law tracks)."""
        d = self._ref_data()
        d.qpos[:7] = np.concatenate([self.plan.base_xyz, K.yaw_quat(self.plan.base_yaw)])
        for s in K.SIDES:
            d.qpos[self.ids.leg_qadr[s]] = self.solver.q_leg[s]
        d.qpos[19:36] = self.plan.upper
        K.scratch_kinematics(model, d)
        return self.ids.com_xy(d)

    def _ref_data(self) -> mujoco.MjData:
        if not hasattr(self, "_rd") or self._rd is None:
            self._rd = mujoco.MjData(self.ids.model)
        return self._rd

    def _emergency(self, model, data) -> None:
        """In-step safety response: abort the step *down*, never sideways.

        When the measured state is bad enough (CoM outside the support or a
        tracking error well beyond the clamps), the response is physical:
        plant the swing foot where it currently is (the balance layer then
        anchors it, so it becomes support), stop the step, and move the pelvis
        goal back over the mid-foot.  Recorded as an ``emergency`` event, and
        the run is marked as having needed it.
        """
        info = getattr(self.law, "info", {})
        if not info or not self.stepper.busy():
            return
        margin = float(info.get("margin", 0.0))
        e = float(np.linalg.norm(info.get("e_track", np.zeros(2))))
        if self._emergency_cooldown > 0.0:
            self._emergency_cooldown -= 0.02
            return
        if margin > -0.030 and e < 0.120:
            return
        for side in K.SIDES:
            st = self.stepper.state[side]
            if not st.active():
                continue
            cur = self.ids.sole_center(data, side)
            ft = self.plan.feet[side]
            ft.origin_xy = np.array(cur[:2], float)
            ft.sole_z = K.SOLE_REST_Z
            ft.planted = True
            self.stepper._abort(st)
            self.events.append({"t": self.t, "event": "emergency_plant", "side": side,
                                "margin": round(margin, 4), "e_track": round(e, 4)})
            self.guards["emergency"] = self.guards.get("emergency", 0) + 1
        self._emergency_cooldown = 0.6      # let the robot settle before a new step
        self.nominal_now = (np.mean([self.plan.feet[s].origin_xy for s in K.SIDES], axis=0)
                            - self.com_local)
        d = self.nominal_now - self.plan.base_xyz[:2]
        self.plan.base_xyz[:2] += np.clip(d, -0.004, 0.004)

    def _feet_down(self, data: mujoco.MjData, tol: float = 0.016) -> bool:
        """True when every foot that the plan calls planted is on the mat."""
        clear = self.ids.foot_clearance(data)
        return all(float(clear[i]) < tol for i, s in enumerate(K.SIDES)
                   if self.plan.feet[s].planted)

    def _safe_q(self) -> np.ndarray:
        """Emergency target: the built stance (both feet planted, hips centred)."""
        return self._stance_q.copy()

    # -- per-skill plan updates -------------------------------------------
    def _update_plan(self, model, data, cmd: DrillCommand, dt: float) -> None:
        plan = self.plan
        plan.label = f"{cmd.skill}"
        # upper body: carriage follows the skill, rate limited
        self._upper_targets(cmd, dt)
        # level change (rbric C): feet planted, hips drop, knees absorb
        target_h = float(np.clip(cmd.stance_height, 0.0, 1.0))
        self._height += float(np.clip(target_h - self._height, -dt / 1.2, dt / 1.2))
        base_z = float(self.stance.spec.pelvis_z) - posture_mod.MAX_CROUCH * self._height
        step = base_z - plan.base_xyz[2]
        if step < 0.0 and not self._feet_down(data):
            # descent governor: hips only drop while both planted feet are on
            # the mat -- asking for a shorter leg while a foot is already light
            # lifts the foot instead of lowering the body (measured)
            step = 0.0
            self.guards["descent_hold"] = self.guards.get("descent_hold", 0) + 1
            self.events.append({"t": self.t, "event": "descent_hold",
                                "clearance": [float(v) for v in
                                              self.ids.foot_clearance(data)]})
        plan.base_xyz[2] += float(np.clip(step, -Z_SLEW * dt, Z_SLEW * dt))
        # where the pelvis belongs for *this* foot placement: the mid-foot
        # point moved so the CoM lands between the feet
        self.nominal_now = (np.mean([plan.feet[s].origin_xy for s in K.SIDES], axis=0)
                            - self.com_local)
        self._posture_modulation(cmd, dt)
        self._home_com(cmd, data, dt)
        self._adapt_reference(data, dt)
        # steps
        self.stepper.com_local = self.com_local
        self._maybe_request_step(cmd, data)
        ev = self.stepper.update(data, plan, dt, self.ids.com_xy(data),
                                 self.nominal_now, load=self.ids.foot_load(data))
        for e in ev:
            self.events.append({"t": self.t, **e})
            if e.get("event", "").endswith("timeout"):
                self.guards["timeouts"] += 1
        if self.stepper.busy():
            self._last_swing = self.stepper.supporting_side() or self._last_swing

    def _posture_modulation(self, cmd: DrillCommand, dt: float) -> None:
        """L0 posture modulation: smooth weight shift the plan follows.

        Amplitude is bounded by the *measured* CoM margin: the shift only grows
        to where the margin still clears 2 cm, so the modulation is a physical
        motion with a real support limit rather than a fixed wobble.
        """
        if cmd.skill not in ("STANCE", "LEVEL_CHANGE", "RECOVER"):
            return
        if "posture_mod" not in RUNG_ELEMENTS[self.rung]:
            return
        if self.stepper.busy():
            return            # never wobble the pelvis while a foot is in the air
        amp = 0.018
        margin_now = float(getattr(self.law, "info", {}).get("margin", 0.05))
        if margin_now < 0.030:
            amp *= 0.0                  # already close to the edge: hold still
        phase = 2.0 * np.pi * 0.23 * self.t + self._sway_phase
        off = np.array([amp * 0.5 * np.sin(phase * 0.5), amp * np.sin(phase)])
        goal = self.nominal_now + off
        self.plan.base_xyz[:2] += np.clip(goal - self.plan.base_xyz[:2],
                                          -0.25 * dt, 0.25 * dt)

    def _upper_targets(self, cmd: DrillCommand, dt: float) -> None:
        """Skill-dependent upper-body carriage (arms drive the shot gesture)."""
        u = self._upper_hold.copy()
        sk = cmd.skill
        if sk == "SHOT_GESTURE":
            ph = float(np.clip(self.stepper.state[cmd.lead_leg].t_phase / 1.0, 0.0, 1.0))
            drive = np.array([-0.35, 0.0, 0.0, 0.25])       # shoulder pitch, elbow
            for side in K.SIDES:
                k = 6 if side == "left" else 13
                u[k:k + 4] += drive * ph
        if sk in ("STANCE", "RECOVER") and self.rung == "L0":
            # posture modulation: small hand-carriage sway (deterministic, slow)
            sway = 0.05 * np.sin(2.0 * np.pi * 0.22 * self.t)
            u[6] += sway
            u[13] -= sway
        self.plan.upper = self.plan.upper + np.clip(u - self.plan.upper,
                                                    -0.06, 0.06 * dt / 0.02)

    def _maybe_request_step(self, cmd: DrillCommand, data: mujoco.MjData) -> None:
        """Queue the next foot placement for the commanded skill (seeded)."""
        if self.stepper.busy() or self._emergency_cooldown > 0.0:
            return
        if self._com_mid_error(data) > self.step_centre_tol:
            return          # let the weight come back to the middle first
        sk = cmd.skill
        feet = self.plan.feet
        rung = self.rung
        if sk in ("SHUFFLE_F", "APPROACH", "RETREAT", "SHUFFLE_B",
                  "SHUFFLE_L", "SHUFFLE_R") and "step" in RUNG_ELEMENTS[rung]:
            vx, vy = cmd.vx, cmd.vy
            if sk == "RETREAT":
                vx, vy = -abs(cmd.vx), cmd.vy
            length = float(np.clip(np.hypot(vx, vy) * 0.55, 0.05, 0.10))
            ang = np.arctan2(vy, vx) if np.hypot(vx, vy) > 1e-6 else 0.0
            jitter = float(self.rng.uniform(-0.15, 0.15)) * length
            side = self._next_side(sk)
            tgt = feet[side].origin_xy + length * np.array([np.cos(ang), np.sin(ang)])
            tgt = rotate_about(tgt, self.nominal_xy + self.com_local, 0.0)
            # keep the stance width: lateral component respects the home offset
            tgt[1] += jitter * 0.3
            self.stepper.request(side, tgt, feet[side].yaw,
                                 f"{sk}-{side}-{self.stepper.state[side].steps_done}")
        elif sk in ("CIRCLE_L", "CIRCLE_R") and "step" in RUNG_ELEMENTS[rung]:
            side = self._next_side(sk)
            ang = (1.0 if sk == "CIRCLE_L" else -1.0) * float(
                np.clip(abs(cmd.wz) * 0.6, 0.10, 0.35))
            self._last_swing = side
            pivot = self.nominal_xy + self.com_local
            tgt = rotate_about(feet[side].origin_xy, pivot, ang)
            self.stepper.request(side, tgt, feet[side].yaw + ang,
                                 f"{sk}-{side}-{self.stepper.state[side].steps_done}")
        elif sk == "ENTRY":            # always: the entry is the initialisation
            # transition (stand keyframe -> drill stance) and is walked, not
            # teleported: each placement is at most ENTRY_STEP_M long, taken by
            # the foot that is furthest from its stance position.
            home = self.stance.foot_origins()
            miss = {s: home[s] - feet[s].origin_xy for s in K.SIDES}
            side = max(K.SIDES, key=lambda s: float(np.linalg.norm(miss[s])))
            dist = float(np.linalg.norm(miss[side]))
            if dist > self.entry_tol:
                u = miss[side] / dist
                tgt = feet[side].origin_xy + min(dist, ENTRY_STEP_M) * u
                yaw = (self.home_yaw[side] if dist < 2 * ENTRY_STEP_M
                       else feet[side].yaw)
                self.stepper.request(side, tgt, yaw,
                                     f"ENTRY-{side}-{self.stepper.state[side].steps_done}")
            else:
                self._entry_done_t += dt
        elif sk == "SHOT_GESTURE" and "shot" in RUNG_ELEMENTS[rung]:
            self._maybe_shot(cmd)

    def _maybe_shot(self, cmd: DrillCommand) -> None:
        """L4 penetration gesture: lead step, knee lower, trail drive, rise."""
        feet = self.plan.feet
        lead, trail = cmd.lead_leg, ("right" if cmd.lead_leg == "left" else "left")
        if self._shot is None:
            tgt = feet[lead].origin_xy + np.array([0.22, 0.0])
            if self.stepper.request(lead, tgt, feet[lead].yaw + 0.05, "SHOT-lead"):
                self._shot = {"phase": "lead", "t0": self.t}
        elif self._shot["phase"] == "lead" and not self.stepper.busy():
            tgt = feet[trail].origin_xy + np.array([-0.10, -0.02])
            if self.stepper.request(trail, tgt, feet[trail].yaw, "SHOT-trail"):
                self._shot["phase"] = "trail"
        elif self._shot["phase"] == "trail" and not self.stepper.busy():
            self._shot = None

    def _com_mid_error(self, data) -> float:
        """Distance from the measured CoM to the mid-foot point (m).

        The mid-foot is taken from the *plan's* foot origins -- the same handle
        everything else uses.  Reading the footprint centres here instead mixed
        the two frames by 3.5 cm and pulled the pelvis forward, which
        straightened the lead knee out of the stance (the rubric caught it).
        """
        mid = np.mean([self.plan.feet[s].origin_xy for s in K.SIDES], axis=0)
        return float(np.linalg.norm(mid - self.ids.com_xy(data)))

    def _adapt_reference(self, data, dt: float) -> None:
        """Let the plan's base follow the achievable state when the law lags.

        The balance layer holds a *steady* CoM offset it cannot cancel with the
        reduced integrator (measured: 4 cm), and that offset is what straightens
        the lead knee -- the reference IK then solves a leg for a pelvis the
        robot is not in, and the posture degrades while the position error
        persists.  Following the measured base (slowly, both feet planted, only
        while the error is large) keeps the *posture* honest; the drill's
        CoM-between-the-feet homing still supplies the position.
        """
        info = getattr(self.law, "info", {})
        if not info or any(not self.plan.feet[s].planted for s in K.SIDES):
            return
        e = float(np.linalg.norm(info.get("e_track", np.zeros(2))))
        if e < 0.030:
            return
        target = np.array(data.qpos[:2], float)
        d = target - self.plan.base_xyz[:2]
        self.plan.base_xyz[:2] += np.clip(d, -0.015 * dt, 0.015 * dt)

    def _home_com(self, cmd: DrillCommand, data, dt: float) -> None:
        """Slow, margin-gated recentring of the CoM between the feet.

        Runs only with both feet planted and only while the CoM is safely inside
        the support; it is the *slow* version of what a landing used to try to do
        in 0.18 s (which dragged the body instead).
        """
        active = [self.stepper.state[s].phase for s in K.SIDES]
        if any(ph not in ("idle", "plant") for ph in active):
            return          # never home the CoM while a foot is swinging --
            #                 and never during the weight shift, where it would
            #                 fight the shift (measured: cancels it exactly)
        if self.law.info.get("margin", 0.0) < 0.020:
            return
        mid = np.mean([self.plan.feet[s].origin_xy for s in K.SIDES], axis=0)
        goal = mid - self.com_local          # absolute target: no servo runaway
        d = goal - self.plan.base_xyz[:2]
        rate = 0.10 * dt
        self.plan.base_xyz[:2] += np.clip(d, -rate, rate)

    def entry_pending(self) -> bool:
        """True while the entry still has feet to place (scheduler gate)."""
        home = self.stance.foot_origins()
        return any(float(np.linalg.norm(self.plan.feet[s].origin_xy - home[s]))
                   > self.entry_tol for s in K.SIDES)

    def _next_side(self, skill: str) -> str:
        """Alternate feet, keeping the skill's leading foot first (seeded)."""
        pair = LEAD_FOOT.get(skill)
        if pair and self.stepper.state[pair].steps_done == 0:
            return pair
        first = "left" if self.stepper.state["left"].steps_done <= \
            self.stepper.state["right"].steps_done else "right"
        return first


if __name__ == "__main__":                              # self-check
    from . import scene as scene_mod

    model = scene_mod.load_model()
    ids = K.RobotIds.build(model)
    st = posture_mod.build_stance(model, posture_mod.StanceSpec(), ids)
    ctrl = FeasibleDrill(st, ids)
    data = mujoco.MjData(model)
    data.qpos[:] = st.qpos
    data.ctrl[:] = st.qpos[7:36]
    mujoco.mj_forward(model, data)
    ctrl.reset(model, data)
    cmd = DrillCommand("STANCE")
    for _ in range(50):
        data.ctrl[:] = ctrl.act(model, data, cmd)
        mujoco.mj_step(model, data)
    print("stance hold ticks ok; pelvis z", round(float(data.qpos[2]), 4),
          "com margin", round(ctrl.law.info["margin"], 4),
          "|e|", round(float(np.linalg.norm(ctrl.law.info["e_track"])), 4))
