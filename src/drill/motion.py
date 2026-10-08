"""Motion study: the width-vs-steppability decision table and the authority probes.

This module is the measurement machinery behind ``reports/2026-10-08/drill_motion.md``.
It builds the *same* drill controller in stances of different width/depth and runs
a fixed shuffle/circle programme, so every row of the table is one controlled
experiment:

* :func:`motion_spec` -- a stance spec from (width, depth, yaw) with the
  operator's two sanctioned repair axes explicit;
* :class:`ShuffleSched` / :class:`CircleSched` / :class:`DeliverableSched` --
  the test programmes (alternating in-place shuffle, in-place circle, the
  deliverable clip programme);
* :func:`run_case` -- one episode through :func:`drill.runner.run` with the
  motion controller (non-default step parameters), returning the trace, the
  events and the summary;
* :func:`summarize` -- the per-case numbers of the decision table (falls,
  steps, cadence, CoM travel vs required, slip, margins, ankle-roll peak);
* :func:`analytic_row` -- the no-simulation part of the table (built stance
  margin, foot separation, required CoM travel per lift).

Nothing here re-implements the controller: it only parameterises the shipped
:class:`drill.controller.FeasibleDrill` and reads its measured events.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import kin as K
from . import posture as posture_mod
from . import scene as scene_mod
from . import stepping as stepping_mod
from .balance import DrillPlan, FootTarget
from .controller import DrillCommand, FeasibleDrill
from .runner import RunConfig, run

REPO = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ stance specs
@dataclass(frozen=True)
class MotionSpec:
    """A stance width/depth pair plus the step-parameter overrides of a case."""

    width: float = 0.28            # lateral foot separation (m)
    depth: float = 0.14            # fore-aft foot separation (m)
    yaw: float = 0.10              # toe-out of both feet (rad, symmetric)
    reach_cap: float = 0.30        # lift-gate authority cap
    shift_speed: float = 0.030     # nominal speed of the smooth shift (m/s)
    drop_gain: float = 0.22        # pelvis sink per metre of shift (m/m)
    drop_max: float = 0.045        # cap on the sink (m)
    settle_tol: float = 0.020      # CoM-to-midfoot error that ends the settle (m)
    settle: bool = True            # run the between-steps recentre phase
    settle_v: float = 0.0          # |CoM speed| (m/s) that ends the settle; 0=off
    settle_min_s: float = 0.40     # minimum settle dwell before the speed exit
    centre_tol: float = 0.030      # CoM-to-midfoot error before a new step request
    v_gate: float = 0.0            # CoM-speed cap of the lift gate (m/s); 0 = off
    gate_inset: float = 0.050      # shift goal inset past the gate point (m)
    step_vx: float = 0.10          # commanded speed of the step-length model
    step_vy: float = 0.10
    pivot_max: float = 0.50        # support-foot yaw pivot cap (rad)
    settle_lead_max: float = 0.040 # lead of the between-steps recentre (m)
    t_settle: float = 9.0          # settle safeguard (s): a shorter one trades
    #                                 recentre completeness for cadence
    recentre_gain: float = 0.0     # lateral placement recentre (0 = off)
    recentre_max: float = 0.05     # cap on the per-step lateral correction (m)
    lean_gain: float = 0.0         # waist roll (rad) per metre of shift -- D2(b)
    lean_max: float = 0.26         # cap on the commanded lean (rad)
    support_roll: bool = False     # D2(a) support-foot edge roll
    support_band: float = 0.10     # |ankle roll| comfort band for the support (rad)
    label: str = ""

    def step_params(self, base: stepping_mod.StepParams | None = None,
                    **over) -> stepping_mod.StepParams:
        from dataclasses import replace

        p = base or stepping_mod.StepParams()
        p = replace(p, reach_cap=self.reach_cap, shift_speed=self.shift_speed,
                    drop_gain=self.drop_gain, drop_max=self.drop_max,
                    settle_tol=self.settle_tol, settle=self.settle,
                    settle_v=self.settle_v, settle_min_s=self.settle_min_s,
                    centre_tol=self.centre_tol, v_gate=self.v_gate,
                    settle_lead_max=self.settle_lead_max, t_settle=self.t_settle,
                    gate_inset=self.gate_inset, pivot_max=self.pivot_max,
                    recentre_gain=self.recentre_gain,
                    recentre_max=self.recentre_max,
                    lean_gain=self.lean_gain, lean_max=self.lean_max,
                    support_roll_gain=0.6 if self.support_roll else 0.0,
                    support_roll_band=self.support_band)
        return replace(p, **over) if over else p

    def stance_spec(self) -> posture_mod.StanceSpec:
        return posture_mod.StanceSpec(
            half_width=self.width / 2, min_width=max(0.20, self.width - 0.01),
            lead_fwd=self.depth / 2, rear_back=-self.depth / 2,
            lead_yaw=self.yaw, rear_yaw=-self.yaw)


def build_stance(model, ids, spec: MotionSpec):
    """Build the stance at the case's width/depth (the operator's repair axes)."""
    return posture_mod.build_stance(model, spec.stance_spec(), ids)


def build_controller(stance, ids, spec: MotionSpec, rung="L3", **over) -> FeasibleDrill:
    return FeasibleDrill(stance, ids, rung=rung, step_params=spec.step_params(**over))


# ------------------------------------------------------------------- programmes
class ShuffleSched:
    """Alternate shuffle blocks along one axis: in-place, gate on real steps.

    ``axis="y"`` is the lateral shuffle (SHUFFLE_L/R), ``axis="x"`` the
    fore-aft stalk (SHUFFLE_F/B).  ``block`` is the block length (s); set it
    larger than the run for a single direction.
    """

    def __init__(self, vy: float = 0.09, vx: float = 0.0, block: float = 6.0,
                 axis: str = "y"):
        self.vy, self.vx, self.block, self.axis = vy, vx, block, axis
        self.events: list = []
        self.done_steps = 0

    def reset(self, model, data) -> None:
        self.events = []
        self.done_steps = 0

    def tick(self, ids, data, ctrl, t: float):
        k = int(t // self.block)
        sgn = 1.0 if k % 2 == 0 else -1.0
        if self.axis == "x":
            vx = sgn * self.vx
            return DrillCommand(skill="SHUFFLE_F" if vx > 0 else "SHUFFLE_B", vx=vx,
                                phase=f"shuffle_{'F' if vx > 0 else 'B'}")
        vy = sgn * self.vy
        return DrillCommand(skill="SHUFFLE_L" if vy > 0 else "SHUFFLE_R", vy=vy,
                            phase=f"shuffle_{'L' if vy > 0 else 'R'}")

    def drain_events(self):
        ev, self.events = self.events, []
        return ev


class CircleSched:
    """One continuous CIRCLE command (in place: feet rotate about the pelvis)."""

    def __init__(self, wz: float = 0.35):
        self.wz = wz

    def reset(self, model, data) -> None:
        pass

    def tick(self, ids, data, ctrl, t: float):
        return DrillCommand(skill="CIRCLE_L", wz=self.wz, phase="circle_L")

    def drain_events(self):
        return []


class StepHoldSched:
    """One step, then a hold: the L2 programme shape (shuffle + stance hold).

    ``hold_s`` is the stance dwell after each completed step; the next shuffle
    element waits for the step count to advance (state-gated, like the shipped
    scheduler).  This is the cadence-benchmark programme: it isolates "how fast
    can one step + recovery be" from "can steps chain continuously".
    """

    def __init__(self, vx: float = 0.0, vy: float = 0.0, hold_s: float = 1.0,
                 timeout: float = 25.0, axis: str = "x", dir_every: int = 4):
        self.vx, self.vy = vx, vy
        self.hold_s = hold_s
        self.timeout = timeout
        self.axis = axis
        self.dir_every = int(dir_every)
        self.state = "hold"
        self.t_enter = 0.0
        self.steps0 = 0
        self.sign = 1.0
        self.n_steps = 0

    def reset(self, model, data) -> None:
        self.state = "hold"
        self.t_enter = 0.0
        self.steps0 = 0
        self.n_steps = 0

    def _steps(self, ctrl):
        st = getattr(ctrl, "stepper", None)
        return int(sum(fe.steps_done for fe in st.state.values())) if st else 0

    def tick(self, ids, data, ctrl, t: float):
        if self.t_enter == 0.0:
            self.t_enter = t
            self.steps0 = self._steps(ctrl)
        if self.state == "hold":
            if t - self.t_enter >= self.hold_s or \
                    self._steps(ctrl) > self.steps0:
                self.state = "step"
                self.t_enter = t
                self.steps0 = self._steps(ctrl)
            return DrillCommand(skill="STANCE", phase="hold")
        # step state: wait for a completed step (or the safeguard)
        if self._steps(ctrl) > self.steps0 or t - self.t_enter > self.timeout:
            self.state = "hold"
            self.t_enter = t
            self.steps0 = self._steps(ctrl)
            self.n_steps += 1
            # the direction flips once per block of steps, NOT per step: with a
            # per-step flip each foot only ever steps one way and the base
            # stretches until the reach cap refuses it (measured)
            if self.n_steps % self.dir_every == 0:
                self.sign = -self.sign
        if self.axis == "x":
            vx = self.sign * self.vx
            return DrillCommand(skill="SHUFFLE_F" if vx > 0 else "SHUFFLE_B",
                                vx=vx, phase="step_f" if vx > 0 else "step_b")
        vy = self.sign * self.vy
        return DrillCommand(skill="SHUFFLE_L" if vy > 0 else "SHUFFLE_R",
                            vy=vy, phase="step_l" if vy > 0 else "step_r")

    def drain_events(self):
        return []


class TapSched:
    """One-foot footwork taps: the weight stays over the *planted* foot.

    The stepping foot alternates between forward and backward taps of
    ``length`` (via the command speed) while the other foot never moves.  The
    lift gate only ever needs the CoM over the planted foot, so the lateral
    base-crossing cost that dominates a two-foot shuffle in a wide stance
    disappears; this is the drill programme that makes a *wide* stance steppable
    (measured in reports/2026-10-08/drill_motion.md).  ``switch_s`` > 0 moves the
    planted foot to the other side every ``switch_s`` seconds (one deliberate
    reshuffle, the expensive step, taken slowly).
    """

    def __init__(self, side: str = "left", vx: float = 0.10, hold_s: float = 0.4,
                 timeout: float = 20.0, switch_s: float = 0.0):
        self.side = side
        self.vx = vx
        self.hold_s = hold_s
        self.timeout = timeout
        self.switch_s = switch_s
        self.state = "hold"
        self.t_enter = 0.0
        self.steps0 = 0
        self.sign = -1.0                    # -1 = backward tap first
        self.last_switch = 0.0

    def reset(self, model, data) -> None:
        self.state = "hold"
        self.t_enter = 0.0
        self.steps0 = 0
        self.last_switch = 0.0

    @staticmethod
    def _steps(ctrl) -> int:
        st = getattr(ctrl, "stepper", None)
        return int(sum(fe.steps_done for fe in st.state.values())) if st else 0

    def tick(self, ids, data, ctrl, t: float):
        if hasattr(ctrl, "step_side_override"):
            ctrl.step_side_override = self.side
        if self.switch_s > 0 and t - self.last_switch >= self.switch_s:
            self.last_switch = t
            self.side = "right" if self.side == "left" else "left"
            ctrl.step_side_override = self.side
        if self.t_enter == 0.0:
            self.t_enter = t
            self.steps0 = self._steps(ctrl)
        if self.state == "hold":
            if t - self.t_enter >= self.hold_s or self._steps(ctrl) > self.steps0:
                self.state = "step"
                self.t_enter = t
                self.steps0 = self._steps(ctrl)
                self.sign = -self.sign       # alternate tap direction
            return DrillCommand(skill="STANCE", phase="hold")
        if self._steps(ctrl) > self.steps0 or t - self.t_enter > self.timeout:
            self.state = "hold"
            self.t_enter = t
            self.steps0 = self._steps(ctrl)
        vx = self.sign * self.vx
        return DrillCommand(skill="SHUFFLE_F" if vx > 0 else "SHUFFLE_B", vx=vx,
                            phase="tap_f" if vx > 0 else "tap_b")

    def drain_events(self):
        return []


class DrillProgramSched:
    """The deliverable programme: named step blocks with a hold after each step.

    Blocks: stance hold, forward shuffle xN, backward xN, lateral left xN,
    lateral right xN, stance hold.  Every block is executed *step by step* (the
    next command waits for the physical step count to advance, so a refusal or
    an abort never runs the command ahead of the body), and the phase string
    advances per block and per step -- the HUD therefore shows real progress
    (the L1 clip's frozen phase label is a defect we must not repeat).
    """

    #: (name, skill, params, steps)
    BLOCKS = (
        ("hold_in", "STANCE", {}, 0),
        ("stalk_fwd", "SHUFFLE_F", {"vx": 0.16, "vy": 0.0}, 3),
        ("stalk_back", "SHUFFLE_B", {"vx": 0.16, "vy": 0.0}, 3),
        ("shuffle_left", "SHUFFLE_L", {"vx": 0.0, "vy": 0.16}, 3),
        ("shuffle_right", "SHUFFLE_R", {"vx": 0.0, "vy": 0.16}, 3),
        ("regroup", "STANCE", {}, 0),
    )

    def __init__(self, hold_s: float = 0.3, block_hold_s: float = 3.0,
                 timeout: float = 22.0):
        self.hold_s = float(hold_s)
        self.block_hold_s = float(block_hold_s)
        self.timeout = float(timeout)
        self.ix = 0
        self.steps_in_block = 0
        self.state = "hold"
        self.t_enter = 0.0
        self.steps0 = 0
        self.events: list = []
        self._t_prev = 0.0

    def reset(self, model, data) -> None:
        self.events = []
        self.ix = 0
        self.steps_in_block = 0
        self.state = "hold"
        self.t_enter = 0.0

    @staticmethod
    def _steps(ctrl) -> int:
        st = getattr(ctrl, "stepper", None)
        return int(sum(fe.steps_done for fe in st.state.values())) if st else 0

    def _block(self):
        return self.BLOCKS[self.ix % len(self.BLOCKS)]

    def tick(self, ids, data, ctrl, t: float):
        if self.t_enter == 0.0:
            self.t_enter = t
            self.steps0 = self._steps(ctrl)
        name, skill, p, n = self._block()
        self._t_prev = t
        if n == 0:                                  # a plain hold block
            if t - self.t_enter >= self.block_hold_s:
                self.events.append({"event": "element_done", "phase": name,
                                    "how": "hold", "t": round(t, 3),
                                    "elapsed": round(t - self.t_enter, 3)})
                self.ix += 1
                self.t_enter = t
                self.steps0 = self._steps(ctrl)
            return DrillCommand(skill="STANCE", phase=name)
        if self.state == "hold":
            if t - self.t_enter >= self.hold_s or self._steps(ctrl) > self.steps0:
                self.state = "step"
                self.t_enter = t
                self.steps0 = self._steps(ctrl)
            return DrillCommand(skill=skill, phase=f"{name} hold {self.steps_in_block + 1}/{n}",
                                vx=float(p.get("vx", 0.0)),
                                vy=float(p.get("vy", 0.0)))
        # step state: one physical step (or the safeguard), then back to hold
        if self._steps(ctrl) > self.steps0 or t - self.t_enter > self.timeout:
            self.steps_in_block += 1
            self.t_enter = t
            self.steps0 = self._steps(ctrl)
            self.state = "hold"
            if self.steps_in_block >= n:
                self.events.append({"event": "element_done", "phase": name,
                                    "how": "steps", "t": round(t, 3),
                                    "elapsed": round(t - self.t_enter, 3)})
                self.steps_in_block = 0
                self.ix += 1
        return DrillCommand(skill=skill, phase=f"{name} step {self.steps_in_block + 1}/{n}",
                            vx=float(p.get("vx", 0.0)),
                            vy=float(p.get("vy", 0.0)))

    def drain_events(self):
        ev, self.events = self.events, []
        return ev


class DeliverableSched:
    """The clip programme: stance settle, then repeating shuffle/circle blocks.

    Blocks (each ``block_s`` long) alternate so the robot neither walks out of
    frame nor drifts: lateral shuffle both ways, a circling pair, and a pause
    where the stance is re-held.  Every element is a plain command; the
    controller's own gates decide when steps actually happen.
    """

    #: (skill, params) of the repeating programme
    BLOCKS = (
        ("STANCE", {}),
        ("SHUFFLE_L", {"vy": 0.085}),
        ("SHUFFLE_R", {"vy": 0.085}),
        ("CIRCLE_L", {"wz": 0.30}),
        ("CIRCLE_R", {"wz": 0.30}),
    )

    def __init__(self, block_s: float = 9.0, start_hold: float = 4.0):
        self.block_s = float(block_s)
        self.start_hold = float(start_hold)
        self.events: list = []

    def reset(self, model, data) -> None:
        self.events = []

    def tick(self, ids, data, ctrl, t: float):
        if t < self.start_hold:
            return DrillCommand(skill="STANCE", phase="stance_hold")
        k = int((t - self.start_hold) // self.block_s)
        skill, p = self.BLOCKS[k % len(self.BLOCKS)]
        return DrillCommand(skill=skill, phase=f"{skill.lower()}#{k // len(self.BLOCKS)}",
                            vx=float(p.get("vx", 0.0)), vy=float(p.get("vy", 0.0)),
                            wz=float(p.get("wz", 0.0)),
                            stance_height=float(p.get("stance_height", 0.0)))

    def drain_events(self):
        ev, self.events = self.events, []
        return ev


# ------------------------------------------------------------------ case runner
def run_case(model, ids, spec: MotionSpec, seconds: float, sched=None, tag: str = "",
             stance=None, controller=None, save_dir: Path | None = None) -> dict:
    """Run one motion case; returns {trace, events, summary, res}."""
    st = stance if stance is not None else build_stance(model, ids, spec)
    ctrl = controller if controller is not None else build_controller(st, ids, spec)
    sched = sched if sched is not None else ShuffleSched()
    cfg = RunConfig(controller="feasible", rung="L3", seconds=float(seconds),
                    start="stance", tag=tag or f"MOTION_w{spec.width:.2f}")
    res = run(cfg, stance=st, model=model, scheduler=sched, verbose=False,
              controller=ctrl)
    out = {"model": model, "ids": ids, "stance": st, "ctrl": ctrl, "res": res,
           "spec": spec, "summary": summarize(res, ids, spec, st),
           "events": res.events}
    if save_dir is not None:
        paths = res.save(Path(save_dir))
        out["paths"] = paths
    return out


def summarize(res, ids: K.RobotIds, spec: MotionSpec, stance) -> dict:
    """The decision-table numbers of one run (every one read from the trace)."""
    ev = res.events
    m = res.metrics
    tr = res.trace
    shifts = [e for e in ev if e.get("event") == "shift_done"]
    aborts = [e for e in ev if e.get("event") == "step_aborted"]
    refuse = [e for e in ev if e.get("event") == "step_refused"]
    dones = [e for e in ev if e.get("event") == "step_done"]
    starts = [e for e in ev if e.get("event") == "step_start"]
    t = np.asarray(tr["t"], float)
    margin = np.asarray(tr["margin"], float)
    qpos = np.asarray(tr["qpos"], float)
    ankle_peak = float(max(np.abs(qpos[:, ids.leg_qadr[s][5]]).max() for s in K.SIDES))
    # cadence: interval between completed steps
    td = [float(e.get("t", 0.0)) for e in dones]
    gaps = np.diff(sorted(td)) if len(td) > 1 else np.array([])
    # measured CoM travel of completed shifts vs the event's required travel
    travel = [float(e.get("com_travel_m", 0.0)) for e in shifts]
    req = [float(e.get("required_com_travel_m", np.nan)) for e in starts]
    lat = abs(float(stance.foot_origins()["left"][1]
                    - stance.foot_origins()["right"][1]))
    dep = abs(float(stance.foot_origins()["left"][0]
                    - stance.foot_origins()["right"][0]))
    margin_built = float(np.asarray(stance.report["com_margin"]) if np.ndim(
        stance.report.get("com_margin", 0.0)) else stance.report.get("com_margin", 0.0))
    return {
        "label": spec.label or f"w{spec.width:.2f}",
        "width_m": round(lat, 4), "width_cmd_m": spec.width,
        "depth_m": round(dep, 4),
        "pelvis_z": round(float(stance.report.get("pelvis_z", np.nan)), 4),
        "margin_built": round(margin_built, 4),
        "falls": int(m["falls"]), "fall_times": m.get("fall_times", []),
        "longest_s": round(float(m["longest_continuous_s"]), 2),
        "duration_s": round(float(t[-1] - t[0] + 0.02), 2) if len(t) else 0.0,
        "steps": int(m["steps_completed"]),
        "shifts": len(shifts), "aborts": len(aborts), "refusals": len(refuse),
        "required_m": sorted({round(v, 4) for v in req if np.isfinite(v)}),
        "travel_m": [round(v, 4) for v in travel],
        "travel_max_m": round(float(max(travel)), 4) if travel else 0.0,
        "cadence_s_per_step": round(float(np.mean(gaps)), 3) if len(gaps) else None,
        "step_gaps_s": [round(float(g), 2) for g in gaps],
        "margin_min": round(float(np.min(margin)), 4) if len(margin) else None,
        "margin_p05": round(float(np.percentile(margin, 5)), 4) if len(margin) else None,
        "slip_m": round(float(m["slip"]["max_load_drift_m"]), 4),
        "ik_err_max": round(float(m["ik_err_max"]), 4),
        "ankle_roll_peak": round(ankle_peak, 4),
        "ankle_roll_limit": round(float(ids.leg_limits["left"][5, 1]), 4),
        "saturation_frac": round(float(np.max(np.asarray(tr["sat_frac"], float))), 4)
        if len(tr["sat_frac"]) else None,
        "timeouts": int(m.get("timeouts", 0)),
        "safety_blend_frac": round(float(np.mean(np.asarray(
            tr["safety_alpha"], float) > 0.02)), 4),
        "lean_peak": round(float(np.max(np.abs(np.asarray(tr["qpos"], float)
                                               [:, 20]))), 4),
    }


def analytic_row(model, ids, spec: MotionSpec) -> dict:
    """The no-simulation row: built margin, foot geometry, required travel per lift."""
    st = build_stance(model, ids, spec)
    import mujoco

    d = mujoco.MjData(model)
    d.qpos[:] = st.qpos
    mujoco.mj_forward(model, d)
    com = ids.com_xy(d)
    hull = ids.support_polygon(d, ids.foot_contact(d))
    margin = float(K.polygon_margin(com, hull))
    orig = st.foot_origins()
    fs = stepping_mod.FootStepper(ids, spec.step_params())
    fs.com_local = np.zeros(2)
    plan = DrillPlan(base_xyz=np.array(d.qpos[:3], float),
                     base_yaw=K.quat_yaw(d.qpos[3:7]),
                     feet={s: FootTarget(orig[s], spec.stance_spec().foot_yaw(s))
                           for s in K.SIDES},
                     upper=np.array(d.qpos[19:36], float))
    req = {}
    for side in K.SIDES:
        g = fs._shift_goal(side, plan)
        req[side] = round(float(np.linalg.norm(g - plan.base_xyz[:2])), 4)
    # support-foot roll relief: how much of the hull margin it buys
    return {
        "width_cmd_m": spec.width, "depth_cmd_m": spec.depth,
        "lat_m": round(abs(float(orig["left"][1] - orig["right"][1])), 4),
        "depth_meas_m": round(abs(float(orig["left"][0] - orig["right"][0])), 4),
        "pelvis_z": round(float(st.report.get("pelvis_z", np.nan)), 4),
        "margin_built": round(margin, 4),
        "required_travel_m": req,
        "required_min_m": round(min(req.values()), 4),
        "required_max_m": round(max(req.values()), 4),
        "reach_cap_m": spec.reach_cap,
    }


def phase_advance_count(trace: dict) -> int:
    """How many times the commanded skill changed over a trace.

    The acceptance check the L1 visual review demanded: a clip whose HUD shows a
    frozen skill/phase for the whole run is a *failed* clip no matter how stable
    it is (the shipped L1 clip advanced 185 scheduler elements while the pelvis
    moved 3 mm -- per-element bookkeeping is not motion).
    """
    skill = np.asarray(trace["skill_id"], int)
    return int(np.sum(np.diff(skill) != 0)) if len(skill) > 1 else 0


def motion_span(trace: dict) -> dict:
    """Measured motion of a trace: CoM/corner travel and knee-height range (m)."""
    com = np.asarray(trace["com"], float)
    q = np.asarray(trace["qpos"], float)
    knee = np.asarray(trace["knee_z"], float)
    n = len(com)
    half = max(1, n // 2)
    return {
        "com_span_m": round(float(np.linalg.norm(com.max(axis=0)[:2]
                                                 - com.min(axis=0)[:2])), 4),
        "com_travel_m": round(float(np.abs(np.diff(com[:, :2], axis=0)).sum()), 4),
        "com_span_second_half_m": round(float(np.linalg.norm(
            com[half:].max(axis=0)[:2] - com[half:].min(axis=0)[:2])), 4),
        "pelvis_z_range_m": round(float(q[:, 2].max() - q[:, 2].min()), 4),
        "knee_z_range_m": round(float((knee.max(axis=0) - knee.min(axis=0)).max()), 4),
    }


# ---------------------------------------------------------------------- stages
def stage_width_table(model, ids, widths=(0.21, 0.28, 0.35, 0.42, 0.495),
                      depth: float = 0.14, seconds: float = 30.0,
                      reach_cap: float = 0.40, base: MotionSpec | None = None,
                      sched=None, verbose: bool = True) -> list:
    """One row per stance width: geometry, required travel, and a real shuffle run.

    ``reach_cap`` is lifted above the shipped 0.14 m so the run measures the
    *achievable* travel (and where it actually breaks) instead of the refusal.
    """
    rows = []
    for w in widths:
        spec = replace(base, width=w, depth=depth, reach_cap=reach_cap) if base \
            else MotionSpec(width=w, depth=depth, reach_cap=reach_cap)
        out = run_case(model, ids, spec, seconds, sched=sched or ShuffleSched(),
                       tag=f"MOTION_W{int(round(w * 100))}")
        row = {**analytic_row(model, ids, spec), **out["summary"]}
        rows.append(row)
        if verbose:
            print(f"[W {w:.3f}] steps {row['steps']} shifts {row['shifts']} "
                  f"aborts {row['aborts']} falls {row['falls']} "
                  f"req {row['required_min_m']:.3f}-{row['required_max_m']:.3f} "
                  f"travel {row['travel_max_m']:.3f} "
                  f"cadence {row['cadence_s_per_step']} margin_min {row['margin_min']}",
                  flush=True)
    return rows


def stage_variants(model, ids, variants, seconds: float = 30.0, sched=None,
                   verbose: bool = True) -> dict:
    """Run a dict {label: MotionSpec} and return {label: {analytic+summary}}."""
    out = {}
    for label, spec in variants.items():
        res = run_case(model, ids, spec, seconds, sched=sched or ShuffleSched(),
                       tag=f"MOTION_{label}")
        out[label] = {**analytic_row(model, ids, spec), **res["summary"]}
        if verbose:
            s = out[label]
            print(f"[{label:22s}] steps {s['steps']} shifts {s['shifts']} "
                  f"aborts {s['aborts']} falls {s['falls']} "
                  f"travel {s['travel_max_m']:.3f} cadence {s['cadence_s_per_step']} "
                  f"margin_min {s['margin_min']} slip {s['slip_m']} "
                  f"ankle {s['ankle_roll_peak']} lean {s['lean_peak']}", flush=True)
    return out
