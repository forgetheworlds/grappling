"""Per-task reward term sets for the solo drill (SOLO_DRILL §5).

Design rules (verified by ``tests/test_solo.py``, hand-constructed states):

1. **A positive survival term is mandatory and dominates.**  ``alive`` is a
   per-step positive signal (upright x stance-height), and every task set must
   satisfy ``alive_weight > sum(per-step penalty weights)`` -- asserted at
   construction, tested.  Falling must never be preferable to standing.
2. **Tracking is always paired with uprightness.**  ``track_lin``/``track_ang``
   are multiplied by an uprightness gate, so "fall forward at commanded speed"
   cannot farm tracking reward (a measured exploit probe).
3. **Bounded penalties.**  Every penalty term is bounded by its weight per
   step, which is what makes rule 1 checkable.
4. **Potential-based progress for shot/recovery.**  ``shot_progress`` and
   ``recover_gain`` are ``gamma*phi(s') - phi(s)`` on physical progress
   (penetration depth / pelvis height): entering and *parking* earns nothing
   (and decays as ``(gamma-1)*phi``), which defeats the "enter a shot and never
   leave" and "knee-parked" exploits.
5. **Weights are placeholders.**  They start from the published locomotion
   recipe (Main's 2026-10-08 steering: track 1.0/sigma 0.5, yaw 1.0-2.0,
   feet_air_time 0.25-0.75 gated by ||cmd||>0.1, feet_slide -0.1,
   flat_orientation -1.0, termination -100..-200 excluding timeouts) and are
   **not** claimed to be tuned; S2 sets them from measured baselines.
6. **Gamma** is configurable per task; defaults 0.995 (balance/stance/shot) and
   0.997 (locomotion) because at 50 Hz gamma=0.99 is only a ~2 s horizon.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from .commands import DEFAULT_COMMAND, Command
from .scene import N_JOINTS, STEP_DT

#: placeholder sigmas
SIGMA_LIN = 0.5          # m/s
SIGMA_ANG = 0.6          # rad/s
SIGMA_STANCE = 0.08      # m
SIGMA_WIDTH = 0.05       # m
SIGMA_REACH = 0.35       # m
#: feet_air_time gate: only rewarded when the command asks for motion
AIR_TIME_CMD_GATE = 0.1
MAX_AIR_TIME = 0.5       # s
#: default shot time budget before the timeout penalty applies
SHOT_BUDGET_S = 2.5


@dataclass(frozen=True)
class RewardWeights:
    """Term weights.  Placeholders (module docstring rule 5), not tuned."""

    # positive per-step
    alive: float = 1.0
    track_lin: float = 1.0
    track_ang: float = 1.0
    stance_height: float = 0.5
    stance_width: float = 0.25
    feet_air_time: float = 0.5
    reach: float = 0.5
    shot_progress: float = 1.0
    shot_leg_seq: float = 0.3
    shot_knee: float = 0.1
    shot_exit: float = 0.5
    recover_gain: float = 1.0
    recover_done: float = 1.0
    # negative per-step (bounded by the weight)
    flat_orientation: float = 0.3
    feet_slide: float = 0.1
    action_rate: float = 0.05
    torque_sat: float = 0.05
    joint_limit: float = 0.05
    shot_timeout: float = 0.3
    low_posture: float = 0.3
    # one-off termination penalty (fall/dorsal; timeouts are not penalized)
    termination: float = 100.0

    def as_dict(self) -> dict:
        return asdict(self)

    def per_step_penalty_sum(self) -> float:
        return float(self.flat_orientation + self.feet_slide + self.action_rate
                     + self.torque_sat + self.joint_limit + self.shot_timeout
                     + self.low_posture)


#: term sets per task family (SOLO_DRILL §5)
TASK_TERMS: dict[str, tuple[str, ...]] = {
    "balance": ("alive", "flat_orientation", "action_rate", "torque_sat",
                "joint_limit"),
    "locomotion": ("alive", "track_lin", "track_ang", "feet_air_time",
                   "flat_orientation", "feet_slide", "action_rate",
                   "torque_sat", "joint_limit"),
    "stance": ("alive", "stance_height", "stance_width", "feet_slide",
               "flat_orientation", "action_rate", "torque_sat", "joint_limit"),
    "reach": ("alive", "reach", "track_lin", "flat_orientation", "feet_slide",
              "action_rate", "torque_sat", "joint_limit"),
    "shot": ("alive", "shot_progress", "shot_leg_seq", "shot_knee", "shot_exit",
             "shot_timeout", "flat_orientation", "feet_slide", "action_rate",
             "torque_sat", "joint_limit"),
    "recovery": ("alive", "recover_gain", "recover_done", "low_posture",
                 "flat_orientation", "action_rate", "torque_sat", "joint_limit"),
}

#: default gammas per task (rule 6)
TASK_GAMMA: dict[str, float] = {
    "balance": 0.995, "locomotion": 0.997, "stance": 0.995,
    "reach": 0.995, "shot": 0.995, "recovery": 0.995,
}

PENALTY_TERMS = frozenset({
    "flat_orientation", "feet_slide", "action_rate", "torque_sat",
    "joint_limit", "shot_timeout", "low_posture",
})


@dataclass
class RewardInputs:
    """All quantities the terms read (hand-constructible; see tests)."""

    dt: float = STEP_DT
    cmd: Command = field(default_factory=lambda: DEFAULT_COMMAND)
    # kinematics
    vel_local: np.ndarray = field(default_factory=lambda: np.zeros(2))
    yaw_rate: float = 0.0
    torso_up_z: float = 1.0
    pelvis_z: float = 0.79
    pelvis_z_prev: float = 0.79
    stand_height: float = 0.79
    stance_width_meas: float = 0.237
    # feet
    foot_contact: tuple[bool, bool] = (True, True)
    foot_slip: tuple[float, float] = (0.0, 0.0)
    foot_air_time: tuple[float, float] = (0.0, 0.0)
    foot_landed: tuple[bool, bool] = (False, False)
    # actuation
    action: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))
    prev_action: np.ndarray | None = None
    sat_frac: float = 0.0
    limit_prox: float = 0.0
    # reach
    hand_distance: float | None = None
    # shot
    shot_depth: float = 0.0
    shot_depth_prev: float = 0.0
    shot_phase: float = 0.0
    shot_time: float = 0.0
    shot_budget_s: float = SHOT_BUDGET_S
    shot_leg_ahead: bool = False
    shot_knee_control: bool = False
    shot_exited: bool = False
    # recovery / terminal
    recovered: bool = False
    dorsal: bool = False


def _clamp01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _gate(inp: RewardInputs) -> float:
    return _clamp01(inp.torso_up_z)


# ------------------------------------------------------------------ positive
def t_alive(inp: RewardInputs) -> float:
    """Positive survival: upright and at (or above) the commanded stance height."""
    ref = max(0.4, float(inp.cmd.stance_height))
    return _clamp01(inp.torso_up_z) * _clamp01(inp.pelvis_z / ref)


def t_track_lin(inp: RewardInputs, sigma: float = SIGMA_LIN) -> float:
    d2 = float(np.sum((np.asarray(inp.vel_local, np.float64)
                       - np.array([inp.cmd.vx, inp.cmd.vy])) ** 2))
    return math.exp(-d2 / (sigma * sigma)) * _gate(inp)


def t_track_ang(inp: RewardInputs, sigma: float = SIGMA_ANG) -> float:
    d = float(inp.yaw_rate) - float(inp.cmd.wz)
    return math.exp(-(d * d) / (sigma * sigma)) * _gate(inp)


def t_stance_height(inp: RewardInputs, sigma: float = SIGMA_STANCE) -> float:
    d = float(inp.pelvis_z) - float(inp.cmd.stance_height)
    return math.exp(-(d * d) / (sigma * sigma)) * _gate(inp)


def t_stance_width(inp: RewardInputs, sigma: float = SIGMA_WIDTH) -> float:
    d = float(inp.stance_width_meas) - float(inp.cmd.stance_width)
    return math.exp(-(d * d) / (sigma * sigma)) * _gate(inp)


def t_feet_air_time(inp: RewardInputs, max_air: float = MAX_AIR_TIME) -> float:
    """Step-quality bonus: credit a foot that just landed after real airtime.

    Gated by the command (``|v_cmd| > AIR_TIME_CMD_GATE`` or a yaw command), so
    standing still cannot farm it (the "staying still" exploit).
    """
    moving = (math.hypot(inp.cmd.vx, inp.cmd.vy) > AIR_TIME_CMD_GATE
              or abs(inp.cmd.wz) > AIR_TIME_CMD_GATE)
    if not moving:
        return 0.0
    r = 0.0
    for landed, air in zip(inp.foot_landed, inp.foot_air_time):
        if landed:
            r += min(float(air), float(max_air))
    return float(r)


def t_reach(inp: RewardInputs, sigma: float = SIGMA_REACH) -> float:
    if inp.hand_distance is None:
        return 0.0
    d = float(inp.hand_distance)
    return math.exp(-(d * d) / (sigma * sigma)) * _gate(inp)


# ------------------------------------------------------------------- shot
def t_shot_progress(inp: RewardInputs, gamma: float = 0.995) -> float:
    """Potential-based penetration progress (rule 4): no parking credit."""
    phi_prev = _clamp01(inp.shot_depth_prev)
    phi_next = _clamp01(inp.shot_depth)
    return float(gamma * phi_next - phi_prev)


def t_shot_leg_seq(inp: RewardInputs) -> float:
    return 1.0 if (inp.shot_leg_ahead and inp.shot_phase > 0.0) else 0.0


def t_shot_knee(inp: RewardInputs) -> float:
    return 1.0 if inp.shot_knee_control else 0.0


def t_shot_exit(inp: RewardInputs) -> float:
    return 1.0 if inp.shot_exited else 0.0


def t_shot_timeout(inp: RewardInputs) -> float:
    return -1.0 if inp.shot_time > inp.shot_budget_s else 0.0


# --------------------------------------------------------------- recovery
def t_recover_gain(inp: RewardInputs, gamma: float = 0.995) -> float:
    """Potential on pelvis-height recovery (progress, not clock)."""
    ref = max(0.4, float(inp.stand_height))
    phi_prev = _clamp01(inp.pelvis_z_prev / ref)
    phi_next = _clamp01(inp.pelvis_z / ref)
    return float(gamma * phi_next - phi_prev)


def t_recover_done(inp: RewardInputs) -> float:
    return 1.0 if inp.recovered else 0.0


def t_low_posture(inp: RewardInputs) -> float:
    ref = max(0.4, float(inp.stand_height))
    return -_clamp01(1.0 - float(inp.pelvis_z) / ref)


# --------------------------------------------------------------- negative
def t_flat_orientation(inp: RewardInputs) -> float:
    """Normalized tilt cost in [0, 1] (1 = fully collapsed/upside down)."""
    return -(1.0 - float(np.clip(inp.torso_up_z, -1.0, 1.0))) / 2.0


def t_feet_slide(inp: RewardInputs) -> float:
    cost = 0.5 * (_clamp01(abs(inp.foot_slip[0]) / 2.0)
                  + _clamp01(abs(inp.foot_slip[1]) / 2.0))
    return -float(cost)


def t_action_rate(inp: RewardInputs) -> float:
    if inp.prev_action is None:
        return 0.0
    d = float(np.abs(np.asarray(inp.action, np.float64)
                     - np.asarray(inp.prev_action, np.float64)).mean())
    return -_clamp01(d / 0.5)


def t_torque_sat(inp: RewardInputs) -> float:
    return -_clamp01(inp.sat_frac)


def t_joint_limit(inp: RewardInputs) -> float:
    return -_clamp01(inp.limit_prox)


TERM_FUNCS = {
    "alive": t_alive,
    "track_lin": t_track_lin,
    "track_ang": t_track_ang,
    "stance_height": t_stance_height,
    "stance_width": t_stance_width,
    "feet_air_time": t_feet_air_time,
    "reach": t_reach,
    "shot_progress": t_shot_progress,
    "shot_leg_seq": t_shot_leg_seq,
    "shot_knee": t_shot_knee,
    "shot_exit": t_shot_exit,
    "shot_timeout": t_shot_timeout,
    "recover_gain": t_recover_gain,
    "recover_done": t_recover_done,
    "low_posture": t_low_posture,
    "flat_orientation": t_flat_orientation,
    "feet_slide": t_feet_slide,
    "action_rate": t_action_rate,
    "torque_sat": t_torque_sat,
    "joint_limit": t_joint_limit,
}


class TaskReward:
    """Reward assembly for one task family; every term is logged separately."""

    def __init__(self, task: str = "balance", weights: RewardWeights | None = None,
                 gamma: float | None = None,
                 termination_penalty: float | None = None):
        if task not in TASK_TERMS:
            raise ValueError(f"unknown task {task!r}; choose from {sorted(TASK_TERMS)}")
        self.task = task
        self.weights = weights or RewardWeights()
        self.gamma = float(TASK_GAMMA[task] if gamma is None else gamma)
        self.termination_penalty = float(
            self.weights.termination if termination_penalty is None
            else termination_penalty)
        self.terms = TASK_TERMS[task]
        # invariant: the positive survival term dominates per-step penalties
        pen = sum(getattr(self.weights, t) for t in self.terms if t in PENALTY_TERMS)
        if not self.weights.alive > pen:
            raise ValueError(
                f"alive weight {self.weights.alive} must exceed the sum of per-step "
                f"penalty weights ({pen:.3f}) for task {task!r}")

    # ---------------------------------------------------------------- pieces
    def term_value(self, name: str, inp: RewardInputs) -> float:
        f = TERM_FUNCS[name]
        if name in ("shot_progress",):
            return f(inp, self.gamma)
        if name in ("recover_gain",):
            return f(inp, self.gamma)
        return f(inp)

    def step(self, inp: RewardInputs) -> tuple[float, dict[str, float]]:
        """Per-step reward and the per-term log (all task terms present)."""
        terms: dict[str, float] = {}
        total = 0.0
        w = self.weights
        for name in self.terms:
            v = float(self.term_value(name, inp))
            terms[name] = v
            total += float(getattr(w, name)) * v
        return total, terms

    def terminal(self, cause: str | None) -> tuple[float, dict[str, float]]:
        """One-off terminal reward.  Falls and dorsal contact are penalized;
        timeouts are not (they are not failures, and clock-based penalties
        create the 'fall to stop paying' pathology)."""
        if cause in ("fall", "dorsal"):
            return -self.termination_penalty, {"termination": -self.termination_penalty}
        return 0.0, {"termination": 0.0}

    def as_dict(self) -> dict:
        return {
            "task": self.task,
            "gamma": self.gamma,
            "termination_penalty": self.termination_penalty,
            "terms": list(self.terms),
            "weights": self.weights.as_dict(),
        }


def discounted_return(rewards, gamma: float, terminal_penalty: float = 0.0) -> float:
    """v0 = sum gamma^t r_t (+ gamma^T * terminal_penalty for a terminated run).

    Used by the hand-horizon test: a collapsing policy must score lower than an
    upright one over the same fixed horizon.
    """
    v = 0.0
    for t, r in enumerate(rewards):
        v += (gamma ** t) * float(r)
    return float(v + (gamma ** len(rewards)) * float(terminal_penalty))


if __name__ == "__main__":  # self-check
    tr = TaskReward("locomotion")
    good = RewardInputs(cmd=Command(vx=0.3, vy=0.0), vel_local=np.array([0.3, 0.0]),
                        torso_up_z=1.0, pelvis_z=0.79)
    r_good, terms = tr.step(good)
    bad = replace(good, torso_up_z=0.2, pelvis_z=0.2,
                  vel_local=np.array([0.3, 0.0]), foot_slip=(1.5, 1.5))
    r_bad, terms_bad = tr.step(bad)
    print("reward self-check:", {"good": round(r_good, 4), "collapsed": round(r_bad, 4),
                                "terms_good": {k: round(v, 3) for k, v in terms.items()}})
    assert r_good > r_bad
    # hand horizon: 50 steps upright vs 10 steps upright then collapse + fall
    upright = [tr.step(good)[0]] * 50
    falling = [tr.step(good)[0]] * 10 + [tr.step(bad)[0]] * 40
    v_up = discounted_return(upright, tr.gamma)
    v_fall = discounted_return(falling, tr.gamma, tr.terminal(good.cmd and "fall")[0])
    assert v_up > v_fall, (v_up, v_fall)
    print("horizon check:", {"upright": round(v_up, 3), "collapse": round(v_fall, 3)})
