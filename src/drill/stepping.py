"""Footstep primitive: weight shift -> lift -> swing -> plant.

This is the component the whole drill is gated on: a *contact-driven*
single-foot reposition.  Every phase transition is decided by a measured
condition -- the **CoM margin inside the support foot's own footprint hull**,
the swing foot's own clearance, its placement error, its sole contact -- so the
phase clock never advances the motion on its own.  Timeouts exist, and they are
recorded as ``timeout`` events: a timed-out step is a failure, not a completed
step (the metrics count only steps that finish in ``plant``).

Why the lift gate is a *margin*, not a load (this is the L2 fix, with numbers):

* the old gate lifted when the swing foot's *vertical load* fell below a
  fraction of its share.  Measured (``reports/2026-10-08/drill.md`` §7.1): the
  gate fired at 51 N with the CoM still **0.03 m outside** the support foot's
  own hull and ~0.05 m short of the travel the geometry needs; the swing leg
  then absorbed the residual with ankle roll until it saturated and the body
  tipped;
* a foot may leave the mat only when the body is over the *other* foot:
  measured as ``polygon_margin(com_xy, hull(sole_xy(support)))`` >=
  ``support_margin`` (0.02 m), held for ``gate_dwell`` ticks.  The vertical
  load is *recorded, never gated on* (:func:`lift_gate`).

Two pieces make the required travel small enough to be reachable on this robot
(the G1's foot is 0.175 x 0.06 m and the ankle roll limit is +-0.2618 rad):

* **support-foot yaw pivot** (``pivot_*``): a slow toe-out turn of the *loaded*
  support foot about its measured footprint centre swings the hull's near
  corner toward the CoM, which *is* CoM margin the shift does not have to
  deliver (measured trade: see the report table);
* **swing-foot roll compliance** (``roll_*``): the swing foot's plan sole
  orientation follows its measured tilt and tilts further when the swing
  ankle roll leaves its comfort band -- measured: +0.15 rad of sole roll moves
  the swing ankle roll by ~+0.09 rad, so the residual is absorbed by the foot
  rolling on its edge rather than by a saturated ankle.

Two more rules make the sequence *survivable* on a position-servo humanoid,
both of them paid for in measured falls:

* **the reference must stay honest about the body** (``_track_body``): the leg
  IK solves for the plan's base, so a plan that disagrees with the body places
  feet wrongly (a completed step's foot floated 1.5-12 cm, the support hull
  collapsed to one foot and the blend that follows toppled the robot).  The
  plan therefore tracks the body in every phase except the deliberate lead of
  the shift itself, and between steps it leans toward the mid-foot by at most
  ``track_lead_max`` -- the only sustained lateral force this balance law
  applies;
* **the geometry is checked before it is attempted**: a step whose gate point
  needs more CoM travel than ``reach_cap`` (the measured weight-shift
  authority) is *refused*, not driven into.  Refusals are events with the
  required travel in them; they are the honest answer to "how wide a base can
  this robot actually step in".

No foot is ever anchored while it moves: the plan marks the swing foot
``planted=False`` for exactly the phases in which it leaves the mat, so the
balance layer never drags it.  The support foot keeps its world target for the
whole step -- *except* the deliberate, measured pivot/restore above -- which is
what makes "no sliding while loaded" measurable rather than aspirational.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import kin as K
from .balance import DrillPlan, FootTarget

#: phase order of one step.  ``recover`` is the fail-closed return of an
#: aborted shift (the plan is displaced; the weight goes back to the middle),
#: ``settle`` is the between-steps recentre that follows a landing -- measured
#: to be necessary: the landing's own residual CoM velocity otherwise carries
#: the body over the far foot.
PHASES = ("shift", "lift", "move", "plant", "recover", "settle")


@dataclass
class StepParams:
    """Step geometry, the margin gate, and the two reach helpers."""

    lift_height: float = 0.045      # swing clearance above the mat (m)
    lift_time: float = 0.09         # commanded lift duration (guard may shorten)
    move_time: float = 0.18
    plant_time: float = 0.20       # soft touchdown: ~0.2 m/s average descent
    loaded_frac: float = 0.25       # landing must carry this share of half body weight
    step_margin: float = 0.010      # CoM margin inside the combined support at landing
    clearance_frac: float = 0.6     # fraction of the lift that starts the swing
    place_tol: float = 0.006        # swing placement error before planting (m)
    flat_tol: float = 0.020         # sole corners within this of the mat
    shift_speed: float = 0.030      # nominal pelvis speed of the smooth shift (m/s).
    #                                 Measured ceiling: a position-servo humanoid
    #                                 cannot brake a lateral CoM moving faster than
    #                                 ~0.1 m/s on a 0.06 m-wide foot -- it topples
    #                                 (the balance law's capture lookahead is 3 mm
    #                                 of CoM here, and the foot edge caps the CoP).

    # -- the lift gate (measured stability, not load) ----------------------
    support_margin: float = 0.020   # required CoM margin (m) inside the SUPPORT
    #                                 foot's own hull before the swing foot may
    #                                 leave the mat
    v_gate: float = 0.0             # optional CoM-speed cap (m/s) of the lift
    #                                 gate: 0 = off (shipped).  Measured: with
    #                                 the margin alone the lift starts while the
    #                                 CoM still coasts at 0.1-0.2 m/s and the
    #                                 body then leaves the support hull on the
    #                                 far side -- the foot cannot brake it.
    gate_dwell: int = 2             # consecutive ticks the margin must hold
    lock_time: float = 0.06         # seconds the gate must hold before the lift
    reach_cap: float = 0.140        # measured lateral weight-shift authority (m):
    #                                 a step whose gate point needs more CoM
    #                                 travel than this is *refused* (fail closed)
    refuse_cd: float = 1.00         # s before the same step may be re-requested
    gate_inset: float = 0.050       # shift goal aims this far past the gate point,
    #                                 so the gate is crossed late in the profile
    #                                 where the approach velocity (and the
    #                                 resulting CoM coast) is small

    # -- shift reach: the pelvis sinks a little when the stretched swing leg
    #    starts pulling its foot off the mat (measured trigger, see _shift)
    drop_gain: float = 0.22         # planned drop per metre of shift (m/m)
    drop_trigger: float = 0.006     # extra drop if the swing foot is still rising (m)
    drop_rate: float = 0.030        # extra drop rate while that persists (m/s)
    drop_max: float = 0.045         # cap on the total drop (m)

    # -- reference honesty: the plan's base tracks the body (see _track_body)
    track_rate: float = 0.25        # m/s the base may move toward the body
    track_deadband: float = 0.025   # error below which the base stays put (m)
    track_max: float = 0.120        # error beyond which the base stops following (m)
    track_lead_max: float = 0.040   # how far the reference may lead the body (m)
    settle_lead_max: float = 0.040  # lead used by the between-steps recentre:
    #                                 the balance law holds a ~4 cm steady-state
    #                                 CoM offset, so a lead of the same size
    #                                 cancels exactly and the recentre stalls
    #                                 (measured: the settle then times out at 9 s
    #                                 on every step -- the whole step cadence is
    #                                 paid for by this one number)

    # -- recentring between steps (bring the weight back to the middle)
    recover_speed: float = 0.10     # m/s, used by the abort recovery
    recover_tol: float = 0.030      # CoM-to-midfoot error that ends recovery (m)
    t_recover: float = 4.00         # safeguard (s)
    settle_tol: float = 0.020       # measured CoM-to-midfoot error that ends it
    settle_v: float = 0.0           # optional: |CoM speed| (m/s) that ends it.
    #                                 0 = off (shipped).  Measured: the recentre
    #                                 cannot reach the geometric mid-foot at all
    #                                 (the balance law holds a ~4 cm steady-state
    #                                 CoM offset that the 4 cm lead exactly
    #                                 cancels), so the settle ran its full 9 s
    #                                 safeguard on every step and the cadence
    #                                 was spent there.  What physically matters is
    #                                 that the body has *stopped* with the CoM
    #                                 inside the support -- that is the exit.
    settle_min_s: float = 0.40      # minimum time in settle before a speed exit
    settle_margin: float = 0.020    # CoM margin required by the speed exit (m)
    #                                 (tighter than the caller's own
    #                                 step_centre_tol, or the next step is
    #                                 never requested)
    t_settle: float = 9.00          # safeguard (s)

    # -- support-foot yaw pivot (shortens the required lateral travel) -----
    pivot_max: float = 0.50         # max toe-out yaw of the support foot (rad)
    pivot_rate: float = 0.35        # rad/s (slow: a loaded pivot must not scrub)
    pivot_restore: float = 0.20     # rad/s back to the planned yaw once idle

    # -- D2 authority mechanisms (0 = off = the shipped primitive; each one is
    #    measured in reports/2026-10-08/drill_motion.md) --------------------
    lean_gain: float = 0.0          # waist roll (rad) per metre of lateral shift:
    #                                 leans the trunk toward the support foot, so
    #                                 part of the required CoM travel is trunk, not
    #                                 pelvis (measured: 0.065 m of CoM per rad)
    lean_max: float = 0.26          # cap on the commanded lean (rad, waist range 0.52)
    support_roll_gain: float = 0.0  # support-foot edge roll (1 = full relief policy):
    #                                 lets the loaded foot roll onto its edge so the
    #                                 support ankle roll stays inside its band
    support_roll_band: float = 0.10 # |ankle roll| comfort band of the support foot
    support_roll_follow: float = 1.0  # sole roll follows the measured sole tilt
    centre_tol: float = 0.030       # CoM-to-midfoot error before a new step is requested
    settle: bool = True             # run the between-steps recentre phase
    recentre_gain: float = 0.0      # lateral placement recentre (0 = off, shipped):
    #                                 when > 0 the step target is pulled laterally so
    #                                 the new mid-foot point tracks the measured CoM
    #                                 (measured: without it the base drifts laterally
    #                                 until the required travel exceeds ``reach_cap``
    #                                 and the drill stalls in refusals)
    recentre_max: float = 0.05      # cap on that per-step lateral correction (m)

    # -- swing-foot roll compliance (the swing leg carries no roll authority)
    swing_follow: float = 1.00      # how far the swing plan target follows the
    #                                 measured contact each tick (1 = fully
    #                                 compliant: the leg stops pushing)
    roll_follow: float = 1.00       # sole roll target follows the measured tilt
    roll_band: float = 0.180        # |ankle_roll| comfort band (rad)
    roll_relieve: float = 1.70      # sole roll per unit ankle-roll excess (rad/rad)
    roll_rate: float = 0.80         # slew on the sole-roll target (rad/s)
    roll_limit: float = 0.60        # hard cap on the planned sole roll (rad)
    roll_flatten: float = 8.0       # 1/s: roll released while the foot is in the air

    # -- guards (fail closed: abort the step, never lift into a bad state)
    ankle_guard: float = 0.95       # abort when |ankle_roll| > guard * limit
    lag_max: float = 0.050          # body-vs-plan tracking error that is "stalled"
    lag_s: float = 0.60             # how long that may persist before aborting
    t_shift: float = 8.00           # timeouts (s) — recorded as failures
    t_lift: float = 0.45
    t_move: float = 0.55
    t_plant: float = 0.45


def lift_gate(margin_support: float, swing_load: float, body_w: float,
              params: StepParams, v_com: float | None = None) -> tuple[bool, dict]:
    """The lift gate: CoM margin inside the *support* foot's own hull.

    Returns ``(ok, record)``.  ``ok`` is True only when the measured margin is
    at least ``params.support_margin``; the swing foot's vertical load is
    carried in the record for the evidence bundle and **never gates**.  This
    is the regression surface for the L2 fix: a gate that reverted to
    "swing foot unloaded" would answer True to ``(margin=-0.01, load=0.0)``,
    which the test suite asserts against.

    ``v_com`` (m/s, horizontal CoM speed) is a second, optional condition: when
    ``params.v_gate > 0`` the gate additionally requires the measured CoM to be
    slower than ``v_gate``.  Measured reason (reports/2026-10-08/drill_motion.md):
    with the margin alone the lift starts while the CoM is still coasting at
    0.1-0.2 m/s toward/over the support foot, and the 0.06 m-wide foot has no
    braking authority left at that point -- the body then leaves the support
    hull on the far side and topples.  Requiring a near-stationary CoM makes
    the lift start from a body that is *there*, not passing through.
    """
    ok = bool(float(margin_support) >= params.support_margin)
    v_ok = True
    if params.v_gate > 0.0 and v_com is not None:
        v_ok = bool(abs(float(v_com)) <= params.v_gate)
    ok = ok and v_ok
    half = 0.5 * float(body_w)
    return ok, {"margin_support": round(float(margin_support), 4),
                "swing_load_n": round(float(swing_load), 1),
                "load_frac_half_weight": round(float(swing_load / half), 3)
                if half > 0 else 0.0,
                "com_v_m_s": round(float(v_com), 4) if v_com is not None else None,
                "v_gate_m_s": float(params.v_gate),
                "gate": ("com_margin_support_hull" if (ok and v_ok) else
                         ("com_velocity" if (not v_ok) else "hold"))}


def stepping_base_spec():
    """The base the L2 stepping rungs are demonstrated in (a *measured* choice).

    The drill stance's own width is not steppable on this robot: lifting a foot
    in a 0.495 m-wide stance needs ~0.21-0.25 m of lateral CoM travel, against a
    measured authority of ~0.13-0.14 m (the CoM runs away from the plan past
    that -- ``reports/2026-10-08/support_envelope.md`` gives the geometric
    reason: the G1 foot is 0.06 m wide and the ankle roll limit is +-0.2618 rad).
    This spec is the widest *square* base whose required lift travel stays
    inside ``StepParams.reach_cap`` with margin: 0.21 m wide, 0.06 m deep, both
    feet toed out 0.10 rad (measured: at 0.25 m wide the per-step requirement
    reaches 0.14-0.165 m, i.e. the cap itself, and one run in two still topples
    on the residual drift after a landing).  It is used only by the stepping
    evidence; the drill stance (``StanceSpec()``) is unchanged and still the L1
    hold base, and its ``min_width`` rule is about that stance, not this one.
    """
    from .posture import StanceSpec

    return StanceSpec(half_width=0.105, min_width=0.20, lead_fwd=0.03,
                      rear_back=-0.03, lead_yaw=0.10, rear_yaw=-0.10)


@dataclass
class FootState:
    """Per-foot step state (one foot moves at a time by construction)."""

    phase: str = "idle"             # idle | shift | lift | move | plant
    target_xy: np.ndarray | None = None
    target_yaw: float = 0.0
    start_xy: np.ndarray | None = None
    start_yaw: float = 0.0
    #: absolute pelvis target of the weight shift (computed once, at the step
    #: start, from the geometry -- a servo on the *measured* CoM runs away when
    #: the CoM lags the pelvis, which is exactly what happens while the legs
    #: stay planted)
    shift_goal: np.ndarray | None = None
    pivot: float = 0.0              # applied support-foot yaw pivot (rad)
    gate_ticks: int = 0             # consecutive ticks the margin gate held
    lag_t: float = 0.0              # seconds the body has lagged the plan
    com0: np.ndarray | None = None  # measured CoM at the step start (travel base)
    shift_dist: float = 0.0         # accumulated pelvis shift this step (m)
    shift_start: np.ndarray | None = None   # plan pelvis xy at the step start
    shift_T: float = 1.0            # smooth-shift duration (s), set at step start
    drop: float = 0.0               # pelvis sink applied this shift (m)
    base_z0: float = 0.0            # plan pelvis height at the step start (m)
    lock_t: float = 0.0             # seconds the lift gate has been held
    set_goal: np.ndarray | None = None   # recentre goal (set by _begin_settle)
    hold_goal: np.ndarray | None = None  # base held while the foot is in the air
    t_phase: float = 0.0
    label: str = ""
    events: list = field(default_factory=list)
    steps_done: int = 0

    def active(self) -> bool:
        return self.phase != "idle"


class FootStepper:
    """Drives one foot at a time through the reposition phases.

    ``update`` mutates the plan (foot targets, pelvis xy, foot roll) and
    returns the events produced this tick.  ``request`` starts a step only when
    that foot is idle; the caller (gait planner) queues targets.
    """

    def __init__(self, ids: K.RobotIds, params: StepParams | None = None):
        self.ids = ids
        self.p = params or StepParams()
        self.state = {s: FootState() for s in K.SIDES}
        self.queue = {s: None for s in K.SIDES}
        self.last_event: dict = {}
        self.com_local = np.zeros(2)      # CoM offset from the pelvis (planning)
        self._refuse_cd = 0.0             # cooldown after a geometric refusal (s)
        self._lean = 0.0                  # commanded trunk lean of the active shift (rad)

    def reset(self) -> None:
        self.state = {s: FootState() for s in K.SIDES}
        self.queue = {s: None for s in K.SIDES}
        self._refuse_cd = 0.0

    def request(self, side: str, origin_xy: np.ndarray, yaw: float, label: str) -> bool:
        """Queue a step for ``side`` (world target).  False when already busy."""
        if self.state[side].active() or self.queue[side] is not None:
            return False
        if self._refuse_cd > 0.0:
            return False        # a refused geometry is not retried every tick
        self.queue[side] = (np.asarray(origin_xy, float).copy(), float(yaw), label)
        return True

    def busy(self) -> bool:
        return any(st.active() for st in self.state.values()) or any(
            self.queue[s] is not None for s in K.SIDES)

    def supporting_side(self) -> str | None:
        """The foot that must carry the body while the other moves."""
        for s in K.SIDES:
            if self.state[s].active():
                return "right" if s == "left" else "left"
        return None

    def roll_authority(self) -> tuple | None:
        """Sides allowed to carry *roll* balance offsets this tick.

        While a foot is stepping, only the support leg may push the body
        sideways: the swing leg is compliant by construction (its plan roll
        follows the contact, see :meth:`_roll_policy`) and must not also be
        driven by the balance law's roll channel -- measured: with both legs
        carrying roll the CoM transfer stalls and the swing ankle does the
        work until it saturates (drill.md §7.1).
        """
        sup = self.supporting_side()
        return None if sup is None else (sup,)

    # -- one control tick --------------------------------------------------
    def update(self, data, plan: DrillPlan, dt: float, com_xy: np.ndarray,
               nominal_xy: np.ndarray, load: np.ndarray | None = None,
               com_v: np.ndarray | None = None) -> list:
        ev = []
        self._refuse_cd = max(0.0, self._refuse_cd - dt)
        for side in K.SIDES:
            st = self.state[side]
            ft: FootTarget = plan.feet[side]
            if not st.active():
                if abs(ft.roll) > 1e-6 and not (
                        self.p.support_roll_gain > 0
                        and self.supporting_side() == side):
                    # an idle sole target must be flat: any compliance roll is
                    # released as soon as the foot is not being shifted (the
                    # one exception: a support foot carrying a deliberate
                    # edge-roll, which *is* the mechanism under test)
                    ft.roll *= max(0.0, 1.0 - self.p.roll_flatten * dt)
                if self.queue[side] is not None:
                    tgt, yaw, label = self.queue[side]
                    self.queue[side] = None
                    st.phase = "shift"
                    st.t_phase = 0.0
                    st.gate_ticks = 0
                    st.lag_t = 0.0
                    st.shift_dist = 0.0
                    st.drop = 0.0
                    st.lock_t = 0.0
                    st.base_z0 = float(plan.base_xyz[2])
                    st.com0 = np.asarray(com_xy, float).copy()
                    st.target_xy, st.target_yaw, st.label = tgt, yaw, label
                    st.start_xy = ft.origin_xy.copy()
                    st.start_yaw = ft.yaw
                    st.shift_goal = self._shift_goal(side, plan)
                    st.shift_start = np.asarray(plan.base_xyz[:2], float).copy()
                    span = float(np.linalg.norm(np.asarray(st.shift_goal, float)
                                                - st.shift_start))
                    # fail closed on the *geometry*: the CoM must travel to the
                    # gate point, and this robot's measured lateral weight-shift
                    # authority is ~0.13 m (the CoM runs away from the plan past
                    # that).  A step whose gate point is further away cannot be
                    # delivered by any shift -- refusing it keeps the robot up
                    # and names the reason instead of driving into the fall.
                    if span > self.p.reach_cap:
                        st.phase = "idle"
                        st.target_xy = None
                        self._refuse_cd = self.p.refuse_cd
                        ev.append({"side": side, "event": "step_refused",
                                   "label": label,
                                   "required_com_travel_m": round(span, 4),
                                   "reach_cap_m": self.p.reach_cap,
                                   "goal": [round(float(v), 4) for v in st.shift_goal]})
                    else:
                        st.shift_T = float(np.clip(span / max(1e-6, self.p.shift_speed),
                                                   0.45, 4.50))
                        ev.append({"side": side, "event": "step_start", "label": label,
                                   "shift_goal": [round(float(v), 4)
                                                  for v in st.shift_goal],
                                   "required_com_travel_m": round(span, 4)})
                continue
            st.t_phase += dt
            if st.phase == "shift":
                ev += self._shift(side, st, plan, data, dt, com_xy, nominal_xy,
                                  load, com_v)
            elif st.phase == "lift":
                ev += self._lift(side, st, plan, data, dt)
            elif st.phase == "move":
                ev += self._move(side, st, plan, data, dt)
            elif st.phase == "plant":
                ev += self._plant(side, st, plan, data, dt)
            elif st.phase == "recover":
                ev += self._recover(side, st, plan, data, dt)
            elif st.phase == "settle":
                ev += self._settle(side, st, plan, data, dt, com_v)

        # (a) the support-foot edge roll is a *whole-step* mechanism: the loaded
        # ankle saturates during the lift/move/plant too (the CoM keeps drifting
        # while the other foot is in the air), so it is refreshed in every phase
        # while a foot is stepping, not only during the shift.
        sup = self.supporting_side()
        if sup is not None and self.p.support_roll_gain > 0.0:
            self._support_roll_policy(sup, plan, data, dt)

        if not any(st.phase == "shift" for st in self.state.values()):
            # the plan must keep agreeing with the body in every phase but the
            # deliberate lead of the shift itself: a reference that disagrees
            # with the body makes the leg IK place the feet wrongly (measured: a
            # completed step's foot floated 1.5-12 cm, the support hull
            # collapsed to one foot and the safety blend then toppled the
            # robot).  A settling step's goal is carried as a bounded lead.
            goals = []
            for s_ in self.state.values():
                if s_.phase in ("lift", "move", "plant") and s_.hold_goal is not None:
                    goals.append(s_.hold_goal)
                elif s_.phase in ("settle", "recover") and s_.set_goal is not None:
                    goals.append(s_.set_goal)
            goal = goals[0] if goals else self._mid_goal(plan)
            lead = self.p.settle_lead_max if any(
                s_.phase == "settle" for s_ in self.state.values()) else None
            # the reference always keeps a bounded lead toward that goal: the
            # lead is the only sustained lateral force this balance law applies,
            # and without it the body's momentum is undamped (measured: a
            # 0.08 m/s post-step drift that ran the CoM over the far foot)
            self._track_body(plan, com_xy, dt, goal=goal, lead_max=lead)
        if all(st.phase in ("idle", "recover", "settle") for st in self.state.values()):
            ev += self._restore_pivots(plan, data, dt)
        self._update_lean(plan)
        self.last_event = ev[-1] if ev else {}
        return ev

    # -- phases ------------------------------------------------------------
    def _timeout(self, side: str, st: FootState, limit: float) -> bool:
        if st.t_phase > limit:
            st.events.append({"event": "timeout", "phase": st.phase, "t": st.t_phase})
            return True
        return False

    def _advance(self, st: FootState, phase: str) -> None:
        st.phase = phase
        st.t_phase = 0.0

    # -- the margin gate + its goal ---------------------------------------
    def _shift_goal(self, side: str, plan: DrillPlan) -> np.ndarray:
        """Absolute pelvis target that satisfies the lift gate (computed once).

        Walks the segment from the support footprint centre toward the swing
        footprint centre and finds the first point whose *planned* hull margin
        drops below ``support_margin``; the goal is that point inset toward the
        support by ``gate_inset``.  Computed once, from the plan: a goal that
        chases the measured CoM makes the pelvis chase a receding point (the
        runaway documented in :mod:`drill.balance`), while the *gate* itself is
        always read from the measured state.
        """
        support = "right" if side == "left" else "left"
        c_sup = self._footprint_centre(plan.feet[support], support)
        c_sw = self._footprint_centre(plan.feet[side], side)
        # the goal is what the *pivot* will deliver: the support hull turned by
        # pivot_max swings its near corner toward the CoM, so the gate point is
        # closer than the unpivoted hull suggests
        soles = self._planned_soles(plan.feet[support], support)
        dpsi = self._toe_out(support) * self.p.pivot_max
        soles = c_sup + (soles - c_sup) @ K.z_rot(dpsi).T
        hull = K.hull2d(soles)
        d = c_sw - c_sup
        span = float(np.linalg.norm(d))
        if span < 1e-9:
            return c_sup - self.com_local
        t_gate = 1.0
        n = 40
        for k in range(1, n + 1):
            if K.polygon_margin(c_sup + (k / n) * d, hull) < self.p.support_margin:
                t_gate = (k - 1) / n
                break
        t_goal = max(0.0, t_gate - self.p.gate_inset / span)
        return c_sup + t_goal * d - self.com_local

    def _planned_soles(self, ft: FootTarget, side: str) -> np.ndarray:
        """(4, 2) plan-frame sole corners of one foot target."""
        return ft.points(self.ids, side)[:, :2]

    def _footprint_centre(self, ft: FootTarget, side: str) -> np.ndarray:
        """Footprint centre of a *plan* target (the frame origin is 3.5 cm back).

        The distinction is a measured trap of this build (drill.md §9.4):
        mixing frame origins with footprint centres drags the plan by a
        constant 3.5 cm.
        """
        return self._planned_soles(ft, side).mean(axis=0)

    # -- (c) the support-foot yaw pivot ------------------------------------
    @staticmethod
    def _toe_out(side: str) -> float:
        """Sign that turns a foot's toe away from the body midline (+yaw = +y)."""
        return 1.0 if side == "left" else -1.0

    def _pivot(self, side: str, plan: DrillPlan, data, dpsi: float) -> None:
        """Turn one *planted* foot by ``dpsi`` about its measured footprint centre.

        A footprint is 0.175 x 0.06 m on this robot, so a toe-out turn swings
        the hull's near corner toward the CoM (measured trade in the report):
        the pivot buys CoM margin *without* moving the body, which is exactly
        the scarce resource in a wide stagger.  Rotating about the measured
        centre means the contact patch spins in place instead of translating.
        """
        if abs(dpsi) < 1e-12:
            return
        ft = plan.feet[side]
        c = np.asarray(data.site_xpos[self.ids.sole_sites[side]], float).mean(axis=0)[:2]
        d = np.asarray(ft.origin_xy, float) - c
        ft.origin_xy = c + K.z_rot(dpsi) @ d
        ft.yaw = float(ft.yaw + dpsi)

    def _restore_pivots(self, plan: DrillPlan, data, dt: float) -> list:
        """Slew a lingering support-foot pivot back once nothing is stepping."""
        ev = []
        for side in K.SIDES:
            st = self.state[side]
            if st.pivot <= 1e-9:
                continue
            d = min(self.p.pivot_restore * dt, st.pivot)
            self._pivot(side, plan, data, -self._toe_out(side) * d)
            st.pivot = float(max(0.0, st.pivot - d))
            if st.pivot <= 1e-9:
                ev.append({"side": side, "event": "pivot_restored"})
        return ev

    # -- (a) the support-foot edge roll ------------------------------------
    def _support_roll_policy(self, side: str, plan: DrillPlan, data, dt: float) -> float:
        """Let the *loaded* foot roll onto its edge when its ankle roll runs out.

        Humans do this in every real weight shift: the sole rolls onto its medial
        or lateral edge and the ankle stays inside its range.  Measured (this
        report): the support ankle roll is one of the two joints that saturate
        when the pelvis travels laterally, so relieving it is what extends the
        usable shift.  Returns the applied sole roll (rad).
        """
        p = self.p
        if p.support_roll_gain <= 0.0:
            return 0.0
        ft = plan.feet[side]
        a = float(data.qpos[self.ids.leg_qadr[side][5]])
        tilt = self._sole_tilt(data, side)
        excess = max(0.0, abs(a) - p.support_roll_band)
        relief = -np.sign(a) * excess * p.roll_relieve
        target = float(np.clip(p.support_roll_follow * tilt
                               + p.support_roll_gain * relief,
                               -p.roll_limit, p.roll_limit))
        ft.roll += float(np.clip(target - ft.roll, -p.roll_rate * dt,
                                 p.roll_rate * dt))
        return float(ft.roll)

    # -- (b) the trunk lean -------------------------------------------------
    def trunk_lean(self) -> float:
        """Commanded waist-roll offset (rad) of the active shift (0 when idle).

        Sign: leans the trunk *toward the support foot*, the direction the body
        is being shifted, so part of the required CoM travel is trunk rather
        than pelvis.  Computed from the pelvis displacement from the shift
        start, so it unwinds by itself as the next settle brings the body back.
        """
        return float(self._lean)

    def _update_lean(self, plan: DrillPlan) -> None:
        p = self.p
        self._lean = 0.0
        if p.lean_gain <= 0.0:
            return
        for st in self.state.values():
            if st.phase != "idle" and st.shift_start is not None:
                dy = float(plan.base_xyz[1] - st.shift_start[1])
                self._lean = float(np.clip(-p.lean_gain * dy, -p.lean_max, p.lean_max))
                return
    def _sole_tilt(self, data, side: str) -> float:
        """Signed roll (rad) of the sole plane about the foot's long axis.

        Positive = the +y side of the sole is higher, the same convention as
        the ``roll`` argument of :func:`drill.kin.foot_targets`.
        """
        pts = np.asarray(data.site_xpos[self.ids.sole_sites[side]], float)
        long_v = pts[list(K.TOE_ROWS)].mean(axis=0)[:2] - pts[list(K.HEEL_ROWS)].mean(axis=0)[:2]
        n = float(np.linalg.norm(long_v))
        if n < 1e-9:
            return 0.0
        long_u = long_v / n
        lat = np.array([-long_u[1], long_u[0]])
        s = (pts[:, :2] - pts[:, :2].mean(axis=0)) @ lat
        z = pts[:, 2] - pts[:, 2].mean()
        denom = float(s @ s)
        return float((s @ z) / denom) if denom > 1e-12 else 0.0

    def _roll_policy(self, side: str, plan: DrillPlan, data, dt: float) -> None:
        """Keep the swing sole's plan orientation compliant with the contact."""
        ft = plan.feet[side]
        a = float(data.qpos[self.ids.leg_qadr[side][5]])
        tilt = self._sole_tilt(data, side)
        excess = max(0.0, abs(a) - self.p.roll_band)
        relief = -np.sign(a) * excess * self.p.roll_relieve
        target = float(np.clip(self.p.roll_follow * tilt + relief,
                               -self.p.roll_limit, self.p.roll_limit))
        ft.roll += float(np.clip(target - ft.roll, -self.p.roll_rate * dt,
                                 self.p.roll_rate * dt))

    def _follow_swing(self, side: str, plan: DrillPlan, data) -> None:
        """Compliance (b): let the planted *swing* foot's plan target follow it.

        A world-anchored target makes the swing leg a stiff strut: as the pelvis
        transfers it must extend to keep holding its old footprint, so it
        presses its foot into the mat and that downward force *drives the body
        over the support* (measured: 80 N on the swing foot with the CoM 0.20 m
        away, then the body ran away from the plan at 0.3 m/s and toppled).
        Following the measured contact leaves the leg with ~zero commanded
        error -- the foot stays where friction puts it and the leg stops
        pushing.  The lift then starts from the real contact (`_shift` re-bases
        anyway).
        """
        ft = plan.feet[side]
        cur = np.asarray(data.xpos[self.ids.foot_body[side]][:2], float)
        aft = float(self.p.swing_follow)
        ft.origin_xy = (1.0 - aft) * np.asarray(ft.origin_xy, float) + aft * cur

    def _track_body(self, plan: DrillPlan, com_xy: np.ndarray, dt: float,
                    goal: np.ndarray | None = None,
                    lead_max: float | None = None) -> None:
        """Keep the plan's base on the *body*, optionally leading it by a goal.

        This is the reference-honesty rule of the whole primitive: the leg IK
        solves for the plan's base, so a plan that disagrees with the body
        places the feet wrongly (measured: a completed step's foot floated
        1.5-12 cm and the support hull collapsed to one foot).  With ``goal``
        the reference is allowed to *lead* the body by at most
        ``track_lead_max`` toward that goal -- a small constant lead is what
        makes the balance law push the body at all, so the body follows and the
        lead closes as it arrives (a quasi-static servo around the goal).
        """
        body = np.asarray(com_xy, float) - self.com_local
        tgt = body
        if goal is not None:
            lead = np.asarray(goal, float) - body
            n_lead = float(np.linalg.norm(lead))
            lim = self.p.track_lead_max if lead_max is None else float(lead_max)
            if n_lead > lim:
                lead *= lim / n_lead
            tgt = body + lead
        d = tgt - np.asarray(plan.base_xyz[:2], float)
        n = float(np.linalg.norm(d))
        if n <= self.p.track_deadband:
            return
        if n > self.p.track_max:
            # a gap this large is a fall in progress, not a tracking error: the
            # reference must not be dragged after it (measured: following a
            # 0.18 m gap made the leg IK solve for feet 0.18 m off-target and
            # the robot toppled)
            return
        plan.base_xyz[:2] += np.clip(d * (1.0 - self.p.track_deadband / n),
                                     -self.p.track_rate * dt, self.p.track_rate * dt)

    # -- the weight shift, gated on the measured support-foot margin -------
    def _shift(self, side: str, st: FootState, plan: DrillPlan, data, dt: float,
               com_xy: np.ndarray, nominal_xy: np.ndarray,
               load: np.ndarray | None, com_v: np.ndarray | None = None) -> list:
        """Transfer the CoM over the support foot; lift only when it is *there*.

        The gate is :func:`lift_gate` -- the measured CoM margin inside the
        support foot's own footprint hull, held for ``gate_dwell`` ticks.  The
        shift also runs the two reach helpers (support-foot yaw pivot, swing
        roll compliance) and fails **closed**: if the margin cannot be reached
        -- the support ankle roll approaches its limit, the CoM stops
        approaching, or the safeguard fires -- the step is aborted with the
        foot still on the mat instead of lifting into a saturated leg.
        """
        ids = self.ids
        i = 0 if side == "left" else 1
        support = "right" if side == "left" else "left"
        sup_st = self.state[support]
        p = self.p
        # (c) support-foot yaw pivot, toe-out (brings the hull's near corner in)
        if sup_st.pivot < p.pivot_max - 1e-9:
            d = float(min(p.pivot_rate * dt, p.pivot_max - sup_st.pivot))
            self._pivot(support, plan, data, self._toe_out(support) * d)
            sup_st.pivot += d
        # (a) support-foot edge roll (relieves the loaded ankle roll; measured)
        self._support_roll_policy(support, plan, data, dt)
        # (b) swing-side roll compliance (the swing leg carries no roll authority)
        self._roll_policy(side, plan, data, dt)
        self._follow_swing(side, plan, data)
        # the shift itself: a smooth (min-jerk) move of the pelvis from where it
        # is to the once-computed goal, with zero velocity at both ends.  A
        # velocity-limited ramp was measured to arrive *with* the body's
        # momentum: the CoM coasted ~0.06 m past the plan and out of the
        # 0.06 m-wide support, and the balance law cannot brake that (its
        # capture-point lookahead is 0.3 s ~ 3 mm of CoM here).
        goal = st.shift_goal if st.shift_goal is not None else plan.base_xyz[:2]
        if st.gate_ticks < p.gate_dwell:
            u = min(1.0, st.t_phase / max(1e-6, st.shift_T))
            moved = st.shift_start + _smooth(u) * (np.asarray(goal, float) - st.shift_start)
            st.shift_dist += float(np.linalg.norm(moved - plan.base_xyz[:2]))
            plan.base_xyz[:2] = moved
        # travel needs leg length: a *planned* drop proportional to the shift
        # keeps the stretched swing leg inside its reach (measured: 0.15 m of
        # lateral shift with flat soles needs ~3 cm of pelvis drop; without it
        # the leg pulls its foot 1-2 cm off the mat and then cannot hold the
        # pelvis at all -- the body runs away from the plan).  The drop also
        # *grows* on measured evidence: if the swing foot is still being pulled
        # up, sink a little more.
        if st.com0 is not None:
            st.drop = float(min(p.drop_max, p.drop_gain * st.shift_dist))
        clr = float(ids.foot_clearance(data)[i])
        if p.drop_rate > 0.0 and clr > p.drop_trigger:
            st.drop = float(min(p.drop_max, st.drop + p.drop_rate * dt))
        plan.base_xyz[2] = min(float(plan.base_xyz[2]), st.base_z0 - st.drop)
        # -- measurements ----------------------------------------------------
        hull_sup = K.hull2d(ids.sole_xy(data, support))
        margin_sup = float(K.polygon_margin(com_xy, hull_sup))
        combined = K.hull2d(np.vstack([ids.sole_xy(data, s) for s in K.SIDES]))
        margin = float(K.polygon_margin(com_xy, combined))
        swing_load = float(load[i]) if load is not None else 0.0
        body_w = float(ids.model.body_mass.sum() * 9.81)
        a_sup = float(data.qpos[ids.leg_qadr[support][5]])
        a_sw = float(data.qpos[ids.leg_qadr[side][5]])
        a_lim = float(ids.leg_limits[support][5, 1])
        ok, gate = lift_gate(margin_sup, swing_load, body_w, p,
                             None if com_v is None else float(np.linalg.norm(com_v)))
        st.gate_ticks = st.gate_ticks + 1 if ok else 0
        if st.gate_ticks >= p.gate_dwell:
            st.lag_t = 0.0             # the plan is deliberately frozen here
        lag = float(np.linalg.norm(np.asarray(com_xy, float)
                                   - (np.asarray(plan.base_xyz[:2], float) + self.com_local)))
        st.lag_t = st.lag_t + dt if lag > p.lag_max else 0.0
        st.events.append({"event": "shift", "margin": round(margin, 4),
                          "margin_support": round(margin_sup, 4),
                          "load": round(swing_load, 1),
                          "ankle_support": round(a_sup, 4),
                          "ankle_swing": round(a_sw, 4),
                          "com_v": round(float(np.linalg.norm(com_v)), 4)
                          if com_v is not None else None,
                          "pivot": round(sup_st.pivot, 4),
                          "lag": round(lag, 4)})
        travel = float(np.linalg.norm(np.asarray(com_xy, float) - st.com0)) \
            if st.com0 is not None else 0.0
        # The gate must *hold* before the foot may leave: the CoM lags the plan
        # by ~0.1 s, so a lift on the first tick that clears the bar happens
        # while the body is still travelling (measured: the foot left the mat
        # and the CoM kept going until it was outside the support).  Once the
        # gate has held for ``gate_dwell`` ticks the *plan stops moving* and the
        # lock time accumulates (with a leak, so an intermittent gate still
        # counts); the lift fires only when the lock is complete.
        st.lock_t = st.lock_t + dt if st.gate_ticks >= p.gate_dwell \
            else max(0.0, st.lock_t - dt)
        if st.lock_t >= p.lock_time:
            # the CoM is measurably over the support foot and has stopped
            # moving: the swing foot may leave the mat.  Re-base the lift to
            # where the foot actually is (the shift may have dragged it a few
            # mm) so the swing starts from the real contact, not from the
            # plan's stale target.
            cur = np.asarray(data.xpos[ids.foot_body[side]][:2], float)
            plan.feet[side].origin_xy = cur
            st.start_xy = cur.copy()
            st.start_yaw = float(plan.feet[side].yaw)
            self._advance(st, "lift")
            return [{"side": side, "event": "shift_done", **gate,
                     "ankle_roll_support": round(a_sup, 4),
                     "ankle_roll_swing": round(a_sw, 4),
                     "pivot_rad": round(sup_st.pivot, 4),
                     "shift_s": round(st.t_phase, 2),
                     "com_travel_m": round(travel, 4),
                     "margin_combined": round(margin, 4),
                     "gate_ticks": st.gate_ticks,
                     "lock_s": round(st.lock_t, 2)}]
        reason = None
        if abs(a_sup) > p.ankle_guard * a_lim:
            reason = "ankle_guard"
        elif st.lag_t > p.lag_s:
            reason = "body_lag"
        elif self._timeout(side, st, p.t_shift):
            reason = "timeout"
        if reason is not None:
            shift_s = st.t_phase
            self._abort(st)
            return [{"side": side, "event": "step_aborted", "phase": "shift",
                     "reason": reason, **gate,
                     "ankle_roll_support": round(a_sup, 4),
                     "ankle_roll_swing": round(a_sw, 4),
                     "pivot_rad": round(sup_st.pivot, 4),
                     "shift_s": round(shift_s, 2),
                     "com_travel_m": round(travel, 4),
                     "margin_combined": round(margin, 4)}]
        return []

    def _recover(self, side: str, st: FootState, plan: DrillPlan, data, dt: float) -> list:
        """Bring the weight back to the middle after an aborted shift.

        An abort leaves the plan *displaced* (the shift moved it) with both feet
        planted where they were: without this the CoM sits outside the support
        and the robot topples a second later (measured: fall at 2.76 s after an
        abort at 1.40 s).  The return is a slow, measured move of the pelvis to
        the mid-foot point; the step is only releasable when the CoM is back
        near the middle (the caller's own ``step_centre_tol`` gate).
        """
        mid = np.mean([plan.feet[s].origin_xy for s in K.SIDES], axis=0)
        goal = mid - self.com_local
        d = goal - plan.base_xyz[:2]
        plan.base_xyz[:2] += np.clip(d, -self.p.recover_speed * dt,
                                     self.p.recover_speed * dt)
        err = float(np.linalg.norm(np.asarray(mid, float)
                                   - np.asarray(self.ids.com_xy(data), float)))
        if err < self.p.recover_tol or self._timeout(side, st, self.p.t_recover):
            self._advance(st, "idle")
            return [{"side": side, "event": "recovered", "com_mid_err": round(err, 4)}]
        return []

    def _begin_settle(self, st: FootState, plan: DrillPlan) -> None:
        """Set up the smooth, margin-aware recentre that follows a landing.

        The controller's own homing is faster (0.10 m/s) and was measured to
        swing the CoM across the base and out over the far foot after a step
        (a fall 1.3 s after a clean landing).  This one is a zero-velocity
        min-jerk move like the shift, so the body arrives *at* the middle
        instead of through it, and it stops on a measured condition.
        """
        st.set_goal = self._mid_goal(plan)

    def _mid_goal(self, plan: DrillPlan) -> np.ndarray:
        mid = np.mean([plan.feet[s].origin_xy for s in K.SIDES], axis=0)
        return np.asarray(mid, float) - self.com_local

    def _settle(self, side: str, st: FootState, plan: DrillPlan, data, dt: float,
                com_v: np.ndarray | None = None) -> list:
        """Smooth recentre of the CoM between the feet after a landing.

        The recentre is a *lead*, not a trajectory: the reference stays on the
        body and leans toward the mid-foot by at most ``track_lead_max``, so
        the body follows it and the lead closes as it arrives.  (An absolute
        plan trajectory was measured to diverge from the body and to float the
        landed foot 12 cm.)
        """
        goal = st.set_goal if st.set_goal is not None else self._mid_goal(plan)
        mid = np.mean([plan.feet[s].origin_xy for s in K.SIDES], axis=0)
        err = float(np.linalg.norm(np.asarray(mid, float)
                                   - np.asarray(self.ids.com_xy(data), float)))
        v = 0.0 if com_v is None else float(np.linalg.norm(com_v))
        combined = K.hull2d(np.vstack([self.ids.sole_xy(data, s) for s in K.SIDES]))
        margin = float(K.polygon_margin(self.ids.com_xy(data), combined))
        if (self.p.settle_v > 0.0 and st.t_phase >= self.p.settle_min_s
                and v <= self.p.settle_v and margin >= self.p.settle_margin):
            # physical exit: the body has stopped with the CoM inside the support
            self._advance(st, "idle")
            return [{"side": side, "event": "settled", "com_mid_err": round(err, 4),
                     "reason": "stopped", "com_v": round(v, 4),
                     "margin": round(margin, 4)}]
        if err < self.p.settle_tol:
            self._advance(st, "idle")
            return [{"side": side, "event": "settled", "com_mid_err": round(err, 4)}]
        if self._timeout(side, st, self.p.t_settle):
            self._advance(st, "idle")
            return [{"side": side, "event": "settle_timeout", "com_mid_err": round(err, 4)}]
        return []

    def _abort(self, st: FootState) -> None:
        """Abort a step: hold the foot, and hand the body back to the middle.

        The shift has *moved the plan*; simply going idle leaves the robot
        leaning with the CoM outside the support (measured: a fall 1.4 s after
        an aborted shift).  The state therefore enters ``recover``, which
        slows the pelvis back to the mid-foot point.
        """
        st.phase = "recover"
        st.t_phase = 0.0
        st.lock_t = 0.0
        st.target_xy = None
        st.events.append({"event": "aborted"})

    def _lift(self, side: str, st: FootState, plan: DrillPlan, data, dt: float) -> list:
        ft = plan.feet[side]
        u = min(1.0, st.t_phase / self.p.lift_time)
        ft.sole_z = K.SOLE_REST_Z + self.p.lift_height * _smooth(u)
        ft.planted = False
        ft.roll *= max(0.0, 1.0 - self.p.roll_flatten * dt)   # flatten in the air
        clear = float(self.ids.foot_clearance(data)[0 if side == "left" else 1])
        if clear >= self.p.clearance_frac * self.p.lift_height:
            self._advance(st, "move")
            return [{"side": side, "event": "lift_done", "clearance": round(clear, 4)}]
        if self._timeout(side, st, self.p.t_lift):
            self._advance(st, "move")
            return [{"side": side, "event": "lift_timeout"}]
        return []

    def _move(self, side: str, st: FootState, plan: DrillPlan, data, dt: float) -> list:
        ft = plan.feet[side]
        u = min(1.0, st.t_phase / self.p.move_time)
        ft.origin_xy = (1 - _smooth(u)) * st.start_xy + _smooth(u) * st.target_xy
        ft.yaw = (1 - _smooth(u)) * st.start_yaw + _smooth(u) * st.target_yaw
        ft.sole_z = K.SOLE_REST_Z + self.p.lift_height
        ft.planted = False
        ft.roll *= max(0.0, 1.0 - self.p.roll_flatten * dt)
        err = float(np.linalg.norm(ft.origin_xy - st.target_xy))
        if err <= self.p.place_tol and u >= 0.8:
            self._advance(st, "plant")
            return [{"side": side, "event": "swing_done", "place_err": round(err, 5)}]
        if self._timeout(side, st, self.p.t_move):
            self._advance(st, "plant")
            return [{"side": side, "event": "move_timeout", "place_err": round(err, 5)}]
        return []

    def _plant(self, side: str, st: FootState, plan: DrillPlan, data, dt: float) -> list:
        ft = plan.feet[side]
        u = min(1.0, st.t_phase / self.p.plant_time)
        ft.sole_z = K.SOLE_REST_Z + self.p.lift_height * (1 - _smooth(u))
        ft.origin_xy = st.target_xy
        ft.yaw = st.target_yaw
        ft.roll = 0.0
        ft.planted = True
        flat = bool(self.ids.flat_contact(data, self.p.flat_tol)[0 if side == "left" else 1])
        load_now = float(self.ids.foot_load(data)[0 if side == "left" else 1])
        body_w = float(self.ids.model.body_mass.sum() * 9.81)
        loaded = load_now >= self.p.loaded_frac * 0.5 * body_w
        combined = K.hull2d(np.vstack([self.ids.sole_xy(data, s) for s in K.SIDES]))
        margin = K.polygon_margin(self.ids.com_xy(data), combined)
        if flat and loaded and margin > self.p.step_margin and u >= 1.0:
            st.steps_done += 1
            if self.p.settle:
                self._begin_settle(st, plan)
                self._advance(st, "settle")
            else:
                self._advance(st, "idle")
            return [{"side": side, "event": "step_done", "label": st.label,
                     "landing_load_n": round(load_now, 1),
                     "margin": round(float(margin), 4), "steps": st.steps_done}]
        if self._timeout(side, st, self.p.t_plant):
            ft.sole_z = K.SOLE_REST_Z
            self._advance(st, "idle")
            return [{"side": side, "event": "plant_timeout", "flat": flat,
                     "load_n": round(load_now, 1), "margin": round(float(margin), 4)}]
        return []


def _smooth(u: float) -> float:
    """Min-jerk ease (zero velocity at both ends)."""
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)


def rotate_about(point: np.ndarray, center: np.ndarray, ang: float) -> np.ndarray:
    """Rotate ``point`` about ``center`` by ``ang`` (rad, about +z)."""
    d = np.asarray(point, float) - np.asarray(center, float)
    c, s = np.cos(ang), np.sin(ang)
    return np.asarray(center, float) + np.array([c * d[0] - s * d[1],
                                                 s * d[0] + c * d[1]])
