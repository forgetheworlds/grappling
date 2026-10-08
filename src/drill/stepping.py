"""Footstep primitive: weight shift -> lift -> swing -> plant -> settle.

This is the component the whole drill is gated on (review §"smallest next
physical addition"): a *contact-driven* single-foot reposition.  Every phase
transition is decided by a measured condition -- the CoM margin over the
support foot, the swing foot's own clearance, its placement error, its sole
contact -- so the phase clock never advances the motion on its own.  Timeouts
exist, and they are recorded as ``timeout`` events: a timed-out step is a
failure, not a completed step (the metrics count only steps that finish in
``settle`` at every phase).

No foot is ever anchored while it moves: the plan marks the swing foot
``planted=False`` for exactly the phases in which it leaves the mat, so the
balance layer never drags it.  The support foot keeps its world target for the
whole step, which is what makes "no sliding while loaded" measurable rather
than aspirational.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import kin as K
from .balance import DrillPlan, FootTarget

#: phase order of one step.  There is no "put the CoM back in the middle"
#: phase: the step ends when the foot is down, flat and loaded, and the slow
#: recentring is a separate, margin-gated homing motion -- a fast settle right
#: after the landing was measured to drag a 0.07 m tracking error into the body.
PHASES = ("shift", "lift", "move", "plant")


@dataclass
class StepParams:
    lift_height: float = 0.045      # swing clearance above the mat (m)
    lift_time: float = 0.09         # commanded lift duration (guard may shorten)
    move_time: float = 0.18
    plant_time: float = 0.20       # soft touchdown: ~0.2 m/s average descent
    loaded_frac: float = 0.25       # landing must carry this share of half body weight
    step_margin: float = 0.010      # CoM margin inside the combined support at landing
    settle_time: float = 0.18
    shift_speed: float = 0.13       # pelvis shift rate during weight transfer (m/s)
    settle_speed: float = 0.22
    settle_tol: float = 0.004       # pelvis back at nominal (m)
    unload_frac: float = 0.30       # swing foot load to allow the lift
    #                                (fraction of its half-body-weight share;
    #                                0.45 is reachable with a small, quick weight
    #                                shift in a staggered stance, 0.2 is not)
    unload_margin: float = 0.020    # CoM margin inside the *combined* support to lift (m)
    clearance_frac: float = 0.6     # fraction of the lift that starts the swing
    place_tol: float = 0.006        # swing placement error before planting (m)
    flat_tol: float = 0.020         # sole corners within this of the mat
    dwell_s: float = 0.08           # loaded-landing dwell before the step counts
    t_shift: float = 1.80           # timeouts (s) — recorded as failures
    t_lift: float = 0.45
    t_move: float = 0.55
    t_plant: float = 0.45
    t_settle: float = 0.90


@dataclass
class FootState:
    """Per-foot step state (one foot moves at a time by construction)."""

    phase: str = "idle"             # idle | shift | lift | move | plant | settle
    target_xy: np.ndarray | None = None
    target_yaw: float = 0.0
    start_xy: np.ndarray | None = None
    start_yaw: float = 0.0
    #: absolute pelvis target of the weight shift (computed once, at the step
    #: start, from the geometry -- a servo on the *measured* CoM runs away when
    #: the CoM lags the pelvis, which is exactly what happens while the legs
    #: stay planted)
    shift_goal: np.ndarray | None = None
    t_phase: float = 0.0
    label: str = ""
    events: list = field(default_factory=list)
    steps_done: int = 0

    def active(self) -> bool:
        return self.phase != "idle"


class FootStepper:
    """Drives one foot at a time through the reposition phases.

    ``update`` mutates the plan (foot targets, pelvis xy) and returns the
    events produced this tick.  ``request`` starts a step only when that foot
    is idle; the caller (gait planner) queues targets.
    """

    def __init__(self, ids: K.RobotIds, params: StepParams | None = None):
        self.ids = ids
        self.p = params or StepParams()
        self.state = {s: FootState() for s in K.SIDES}
        self.queue = {s: None for s in K.SIDES}
        self.last_event: dict = {}

    def reset(self) -> None:
        self.state = {s: FootState() for s in K.SIDES}
        self.queue = {s: None for s in K.SIDES}

    def request(self, side: str, origin_xy: np.ndarray, yaw: float, label: str) -> bool:
        """Queue a step for ``side`` (world target).  False when already busy."""
        if self.state[side].active() or self.queue[side] is not None:
            return False
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

    # -- one control tick --------------------------------------------------
    def update(self, data, plan: DrillPlan, dt: float, com_xy: np.ndarray,
               nominal_xy: np.ndarray, load: np.ndarray | None = None) -> list:
        ev = []
        ids = self.ids
        for side in K.SIDES:
            st = self.state[side]
            ft: FootTarget = plan.feet[side]
            if not st.active():
                if self.queue[side] is not None:
                    tgt, yaw, label = self.queue[side]
                    self.queue[side] = None
                    st.phase = "shift"
                    st.t_phase = 0.0
                    st.target_xy, st.target_yaw, st.label = tgt, yaw, label
                    st.start_xy = ft.origin_xy.copy()
                    st.start_yaw = ft.yaw
                    st.shift_goal = self._shift_goal(side, plan, st.start_xy)
                    ev.append({"side": side, "event": "step_start", "label": label})
                continue
            st.t_phase += dt
            if st.phase == "shift":
                ev += self._shift(side, st, plan, data, dt, com_xy, nominal_xy, load)
            elif st.phase == "lift":
                ev += self._lift(side, st, plan, data, dt)
            elif st.phase == "move":
                ev += self._move(side, st, plan, data, dt)
            elif st.phase == "plant":
                ev += self._plant(side, st, plan, data, dt)

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

    def _shift_goal(self, side: str, plan: DrillPlan,
                    swing_xy: np.ndarray) -> np.ndarray:
        """Absolute pelvis target that unloads the swing foot (two-foot model).

        With feet at ``P_sw`` and ``P_sup`` the load on the swing foot is
        ``|C - P_sup| / |P_sw - P_sup|`` of the *body weight* (two-point model),
        so asking for ``u`` of its half-weight share puts the CoM at
        ``P_sup + u/2 * (P_sw - P_sup)`` -- e.g. u = 0.45 of a 0.24 m stance is
        a 0.054 m shift, while u = 0.2 would demand 0.1 m and, in a staggered
        stance, more than the legs can reach.  The pelvis target then follows
        from the CoM offset of the built posture.
        """
        support = "right" if side == "left" else "left"
        p_sup = plan.feet[support].origin_xy
        u = float(self.p.unload_frac)
        com_target = p_sup + 0.5 * u * (swing_xy - p_sup)
        return com_target - getattr(self, "com_local", np.zeros(2))

    def _shift(self, side: str, st: FootState, plan: DrillPlan, data, dt: float,
               com_xy: np.ndarray, nominal_xy: np.ndarray,
               load: np.ndarray | None) -> list:
        """Transfer weight onto the support foot, then lift -- on a measured gate.

        The gate is the *measured* load on the swing foot (``cfrc_ext`` vertical
        force) plus the CoM margin over the support foot: the foot may leave the
        mat only when it is genuinely unloaded.  If the shift cannot unload it
        within the safeguard time, the step is **aborted** (the foot goes back to
        where it started) rather than lifted while loaded -- a loaded lift is
        exactly the sliding/lurching failure the video must not show.
        """
        ids = self.ids
        i = 0 if side == "left" else 1
        support = "right" if side == "left" else "left"
        # the shift is a *measured* CoM servo: move the planned pelvis by the
        # current CoM-to-support-centre error (clamped).  A CoM-offset
        # formulation runs away -- the offset shrinks as the pelvis leaves the
        # feet, so the goal recedes and the pelvis chases it (measured).
        poly = K.hull2d(ids.sole_xy(data, support))
        goal = st.shift_goal if st.shift_goal is not None else plan.base_xyz[:2]
        d = goal - plan.base_xyz[:2]
        step = np.clip(d, -self.p.shift_speed * dt, self.p.shift_speed * dt)
        plan.base_xyz[:2] += step
        # the lift gate is the *measured* unload plus static stability on the
        # combined patch -- not "CoM strictly inside the support foot", which
        # would demand a full-width transfer for no physical reason
        margin_sup = K.polygon_margin(com_xy, poly)
        combined = K.hull2d(np.vstack([ids.sole_xy(data, s) for s in K.SIDES]))
        margin = K.polygon_margin(com_xy, combined)
        swing_load = float(load[i]) if load is not None else 0.0
        body_w = float(self.ids.model.body_mass.sum() * 9.81)
        unloaded = swing_load <= self.p.unload_frac * 0.5 * body_w
        prev = st.events[-1]["load"] if st.events and "load" in st.events[-1] else None
        falling = prev is None or swing_load <= prev + 2.0
        st.events.append({"event": "shift", "margin": round(float(margin), 4),
                          "margin_support": round(float(margin_sup), 4),
                          "load": round(swing_load, 1)})
        if unloaded and falling and margin >= self.p.unload_margin:
            self._advance(st, "lift")
            return [{"side": side, "event": "shift_done", "margin": round(float(margin), 4),
                     "margin_support": round(float(margin_sup), 4),
                     "swing_load_n": round(swing_load, 1)}]
        if self._timeout(side, st, self.p.t_shift):
            plan.feet[side].origin_xy = st.start_xy.copy()
            plan.feet[side].yaw = st.start_yaw
            plan.feet[side].sole_z = K.SOLE_REST_Z
            plan.feet[side].planted = True
            self._abort(st)
            return [{"side": side, "event": "step_aborted", "phase": "shift",
                     "swing_load_n": round(swing_load, 1),
                     "margin": round(float(margin), 4)}]
        return []

    def _abort(self, st: FootState) -> None:
        st.phase = "idle"
        st.t_phase = 0.0
        st.target_xy = None
        st.events.append({"event": "aborted"})

    def _lift(self, side: str, st: FootState, plan: DrillPlan, data, dt: float) -> list:
        ft = plan.feet[side]
        u = min(1.0, st.t_phase / self.p.lift_time)
        ft.sole_z = K.SOLE_REST_Z + self.p.lift_height * _smooth(u)
        ft.planted = False
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
        ft.planted = True
        flat = bool(self.ids.flat_contact(data, self.p.flat_tol)[0 if side == "left" else 1])
        load_now = float(self.ids.foot_load(data)[0 if side == "left" else 1])
        body_w = float(self.ids.model.body_mass.sum() * 9.81)
        loaded = load_now >= self.p.loaded_frac * 0.5 * body_w
        combined = K.hull2d(np.vstack([self.ids.sole_xy(data, s) for s in K.SIDES]))
        margin = K.polygon_margin(self.ids.com_xy(data), combined)
        if flat and loaded and margin > self.p.step_margin and u >= 1.0:
            st.steps_done += 1
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

    def _abort(self, st: FootState) -> None:
        st.phase = "idle"
        st.t_phase = 0.0
        st.target_xy = None
        st.events.append({"event": "aborted"})

    def _lift(self, side: str, st: FootState, plan: DrillPlan, data, dt: float) -> list:
        ft = plan.feet[side]
        u = min(1.0, st.t_phase / self.p.lift_time)
        ft.sole_z = K.SOLE_REST_Z + self.p.lift_height * _smooth(u)
        ft.planted = False
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
        ft.planted = True
        flat = bool(self.ids.flat_contact(data, self.p.flat_tol)[0 if side == "left" else 1])
        load_now = float(self.ids.foot_load(data)[0 if side == "left" else 1])
        body_w = float(self.ids.model.body_mass.sum() * 9.81)
        loaded = load_now >= self.p.loaded_frac * 0.5 * body_w
        combined = K.hull2d(np.vstack([self.ids.sole_xy(data, s) for s in K.SIDES]))
        margin = K.polygon_margin(self.ids.com_xy(data), combined)
        if flat and loaded and margin > self.p.step_margin and u >= 1.0:
            st.steps_done += 1
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

    def _settle(self, side: str, st: FootState, plan: DrillPlan, data, dt: float,
                nominal_xy: np.ndarray) -> list:
        """Bring the CoM back between the feet (measured), then finish."""
        centres = [K.hull2d(self.ids.sole_xy(data, s)).mean(axis=0) for s in K.SIDES]
        mid = 0.5 * (centres[0] + centres[1])
        com = self.ids.com_xy(data)
        d = np.clip(mid - com, -0.05, 0.05)
        step = np.clip(d, -self.p.settle_speed * dt, self.p.settle_speed * dt)
        plan.base_xyz[:2] += step
        plan.feet[side].sole_z = K.SOLE_REST_Z
        plan.feet[side].planted = True
        if float(np.linalg.norm(d)) < self.p.settle_tol:
            st.steps_done += 1
            self._advance(st, "idle")
            return [{"side": side, "event": "step_done", "label": st.label,
                     "steps": st.steps_done}]
        if self._timeout(side, st, self.p.t_settle):
            self._advance(st, "idle")
            st.steps_done += 1
            return [{"side": side, "event": "settle_timeout"}]
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
