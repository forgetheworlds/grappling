"""Evaluation harness: the single path that gates training (S1 contract).

Every candidate controller (scripted baseline, exploit probe, or trained
policy) is evaluated through :func:`evaluate` against a :class:`TaskGate`.
Gates are **provisional placeholders** until measured baselines exist
(``docs/MOTOR_CURRICULUM.md`` §3/§5); their purpose in S1 is to certify the
*harness*: the trivial controllers and exploit probes must NOT pass.

Outputs (under ``data/solo/metrics/`` by default): per-step JSONL traces
(one row per control step, all episodes), a run summary JSON with the
aggregated metrics and the verdict + reasons, and the exact env ``config()``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .commands import (DEFAULT_COMMAND, T2_TRAIN_RANGES, Command, CommandRanges,
                       CommandSchedule, Skill)
from .env import SoloEnv
from .metrics import (METRICS_DIR, METRIC_FIELDS, com_offset_max,
                      recovery_time, summarize_rows)
from .pushes import TRAIN_MAX_IMPULSE, PushSchedule, PushSpec
from .scene import STEP_DT, load_solo_model
from .stance_valid import STANCE as STANCE_THRESHOLDS
from .stance_valid import stance_state, stance_valid

#: all metrics a gate may reference (aggregate keys produced by :func:`evaluate`)
GATE_METRICS: tuple[str, ...] = (
    "fall_rate", "dorsal_rate", "termination_rate",
    "mean_upright", "min_upright", "mean_tilt_deg", "mean_pelvis_z",
    "vel_err_mean", "vel_err_p95", "yaw_err_mean", "yaw_err_p95",
    "stance_err_mean", "slip_mean", "slip_p95",
    "act_delta_mean", "sat_frac_mean", "limit_prox_max",
    "reward_mean", "reward_sum_mean",
    "max_recoverable_impulse", "max_recoverable_impulse_heldout",
    "fall_rate_heldout",
    "recovery_success_rate", "recovery_time_mean",
    "time_to_stability_mean", "time_to_stability_rate",
    "com_offset_max", "steps_after_push_mean", "steps_total_mean",
    "steps_per_s", "n_episodes", "n_steps",
    "hand_err_mean", "shot_depth_max", "shot_exit_rate",
    # terminal-state stance validity (one shared predicate, all tasks)
    "ends_in_valid_stance", "survivor_valid_stance_rate",
    # T2 (locomotion) aggregate names -- all measured, none reward-derived
    "vx_err_abs_mean", "vy_err_abs_mean", "yaw_err_abs_mean",
    "slip_ratio_mean", "dist_err_mean", "travelled_m_mean",
    "commanded_m_mean", "slip_travel_mean", "loaded_step_frac_mean",
)


@dataclass(frozen=True)
class Criterion:
    """One gate criterion: ``metric <op> threshold``."""

    metric: str
    op: str
    threshold: float
    note: str = ""

    def check(self, value: float | None) -> bool:
        if value is None:
            return False
        v, t = float(value), float(self.threshold)
        if self.op == "<=":
            return v <= t
        if self.op == ">=":
            return v >= t
        if self.op == "<":
            return v < t
        if self.op == ">":
            return v > t
        if self.op == "==":
            return abs(v - t) < 1e-9
        raise ValueError(f"unknown op {self.op!r}")

    def describe(self) -> str:
        return f"{self.metric} {self.op} {self.threshold:g}" + (
            f" ({self.note})" if self.note else "")


@dataclass
class TaskGate:
    """Success criteria for one task family (provisional until set from data)."""

    name: str
    criteria: tuple[Criterion, ...]
    provisional: bool = True
    note: str = ""

    def verdict(self, summary: dict) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        ok = True
        for c in self.criteria:
            value = summary.get(c.metric)
            passed = c.check(value)
            reasons.append(("PASS " if passed else "FAIL ") + c.describe()
                           + f" (measured {value!r})")
            ok = ok and passed
        return ok, reasons

    def as_dict(self) -> dict:
        return {"name": self.name, "provisional": self.provisional, "note": self.note,
                "criteria": [{"metric": c.metric, "op": c.op,
                              "threshold": c.threshold, "note": c.note}
                             for c in self.criteria]}


# ==========================================================================
# T2 (locomotion / M2) gate inputs: held-out commands + derived thresholds
# ==========================================================================
#
# The gate is *behavioural*: a controller is judged on physically meaningful
# outcomes measured on commands held out from training.  Reward is never a
# criterion (see ``tests/solo/test_locomotion_gate.py``).

# T2 training command domain -- owned by ``solo.commands`` (imported above, so
# ``solo.eval.T2_TRAIN_RANGES`` resolves): deliberately NARROWER than the
# feasible ``CommandRanges``.  The locomotion task preset samples inside these
# bounds (``env.TASKS``); the gate evaluates *outside* them, so "held out" is a
# property of the command range, not of a seed (a seed alone holds nothing out
# when the sampler covers the range).  ``tests/solo/test_locomotion_gate.py``
# asserts the disjointness.

#: minimum margin (m/s, rad/s) by which every held-out command must sit
#: outside :data:`T2_TRAIN_RANGES` on at least one axis
T2_HELDOUT_MARGIN = 0.04

#: T2 evaluation episode length (s)
T2_EPISODE_S = 6.0

#: T2 measurements start here (s): a jittered reset drops the robot and the
#: feet skid 0.1-0.2 m in the first ~0.5 s of *every* episode.  Tracking
#: errors, slip and travelled/commanded distance are measured from
#: ``T2_SETTLE_S`` on; uprightness and terminations stay episode-wide.
T2_SETTLE_S = 0.5

#: denominator floor for ``slip_ratio`` (m): stand-still episodes travel a
#: little during the transient and dividing by an ~0 path is meaningless
T2_SLIP_RATIO_MIN_TRAVEL = 0.10

#: held-out commands: feasible (inside ``CommandRanges``) but outside the
#: training domain above by >= ``T2_HELDOUT_MARGIN`` on >=1 axis.  Each is a
#: magnitude/axis combination the trainer is not allowed to sample: top speed
#: (0.40/0.45 m/s), full-rate turns in place (+/-0.45 rad/s), lateral shuffle
#: (+/-0.18 m/s), backward travel (-0.20 m/s) and a three-axis mixed command.
HELDOUT_COMMANDS: tuple[Command, ...] = (
    Command(vx=0.45, vy=0.0, wz=0.0, skill_id=int(Skill.SHUFFLE_F)),
    Command(vx=0.40, vy=0.0, wz=0.40, skill_id=int(Skill.CIRCLE_L)),
    Command(vx=0.0, vy=0.18, wz=0.0, skill_id=int(Skill.SHUFFLE_L)),
    Command(vx=0.0, vy=-0.18, wz=0.0, skill_id=int(Skill.SHUFFLE_R)),
    Command(vx=0.0, vy=0.0, wz=-0.45, skill_id=int(Skill.CIRCLE_R)),
    Command(vx=-0.20, vy=0.0, wz=0.0, skill_id=int(Skill.RETREAT)),
    Command(vx=0.38, vy=-0.17, wz=0.35, skill_id=int(Skill.CIRCLE_L)),
)

#: tiny-command reference set (NOT held out -- inside the training range by
#: design): the "commanded (nearly) nothing, so holding still is correct"
#: case.  A gate that cannot certify *this* would be vacuously impossible.
TINY_COMMANDS: tuple[Command, ...] = (
    Command(vx=0.02, vy=0.0, wz=0.0, skill_id=int(Skill.SHUFFLE_F)),
    Command(vx=0.0, vy=0.0, wz=0.02, skill_id=int(Skill.CIRCLE_L)),
)


def heldout_command_outside(cmd: Command,
                            ranges: CommandRanges | None = None) -> dict:
    """Per-axis margin by which ``cmd`` lies outside ``ranges`` (0 = inside)."""
    r = T2_TRAIN_RANGES if ranges is None else ranges
    out = {}
    for axis, rng in (("vx", r.vx), ("vy", r.vy), ("wz", r.wz)):
        v = float(getattr(cmd, axis))
        lo, hi = float(rng[0]), float(rng[1])
        out[axis] = max(lo - v, v - hi, 0.0)
    return out


def is_heldout_command(cmd: Command, ranges: CommandRanges | None = None,
                       margin: float = T2_HELDOUT_MARGIN) -> bool:
    """True iff ``cmd`` is outside the training domain by >= ``margin``."""
    return max(heldout_command_outside(cmd, ranges).values()) >= margin - 1e-9


def t2_oscillation_schedule(*, period_s: float = 0.75,
                            episode_s: float = T2_EPISODE_S) -> CommandSchedule:
    """Held-out command-reversal schedule (the "command oscillation" exploit).

    Alternates a held-out forward command with a held-out backward command.
    A controller that "tracks" by drifting in one direction cannot track the
    reversals; a controller that oscillates to farm a tracking reward pays the
    same error on both halves.
    """
    fwd, back = HELDOUT_COMMANDS[0], HELDOUT_COMMANDS[5]
    segs, t, i = [], 0.0, 0
    while t < episode_s - 1e-9:
        segs.append((t, fwd if i % 2 == 0 else back))
        t += float(period_s)
        i += 1
    return CommandSchedule(segs, default=fwd)


def heldout_command_plan(*, episode_s: float = T2_EPISODE_S,
                         oscillation: bool = True) -> list[CommandSchedule]:
    """One held-out episode per command (steady), plus the reversal schedule."""
    plan = [CommandSchedule.steady(c) for c in HELDOUT_COMMANDS]
    if oscillation:
        plan.append(t2_oscillation_schedule(episode_s=episode_s))
    return plan


def tiny_command_plan() -> list[CommandSchedule]:
    """Reference episodes: commands small enough that holding still tracks them."""
    return [CommandSchedule.steady(c) for c in TINY_COMMANDS]


#: T2 thresholds.  Every number is *derived* from the measured baseline table
#: (``data/solo/metrics/t2_gate_baselines.json``); the rule for each is in the
#: criterion note and repeated in ``reports/2026-10-08/t2_gate.md``.  Nothing
#: here is invented and nothing is reward-based.
T2_THRESHOLDS: dict[str, float] = {
    # 25% of the smallest held-out |vx| (0.40 m/s)
    "vx_err_abs_mean": 0.10,
    # 25% of the smallest held-out |vy| (0.18 m/s)
    "vy_err_abs_mean": 0.045,
    # 33% of the smallest held-out |wz| (0.45 rad/s)
    "yaw_err_abs_mean": 0.15,
    # between the worst measured non-falling uprightness (StandHold 0.9999)
    # and the best measured falling one (FallForward 0.76)
    "mean_upright": 0.95,
    # measured: every displacing baseline either falls or topples
    "fall_rate": 0.10,
    "dorsal_rate": 0.10,
    # 15% of the smallest held-out speed (0.40 m/s); measured standing slip is
    # <=0.008 m/s, measured sliding slip >=0.098 m/s
    "slip_mean": 0.06,
    # bracket between the load-transfer measurements (<=0.57) and the
    # planted-foot drags (>=1.75)
    "slip_ratio_mean": 0.80,
    # 15% of the smallest held-out commanded path (0.40 m/s x 5.5 s = 2.2 m)
    "dist_err_mean": 0.30,
}


def locomotion_episode_metrics(rows: list[dict],
                               *, settle_s: float = T2_SETTLE_S) -> dict:
    """Per-episode T2 quantities from the recorded rows (settled window only).

    ``slip_ratio`` is the discriminating anti-exploit metric: metres travelled
    by *loaded* feet per metre the pelvis travelled.  It is position-based on
    purpose -- the per-step ``slip`` field is a single velocity sample taken
    after MuJoCo's friction constraint has already zeroed any slow drag, so a
    planted-foot slider reads the same ``slip`` as a standing robot.
    """
    settled = [r for r in rows if float(r.get("t", 0.0)) >= float(settle_s)]
    if not settled:
        settled = list(rows)

    def abs_mean(key: str) -> float | None:
        vals = [abs(float(r[key])) for r in settled if r.get(key) is not None]
        return round(float(np.mean(vals)), 6) if vals else None

    def mean(key: str) -> float | None:
        vals = [float(r[key]) for r in settled if r.get(key) is not None]
        return round(float(np.mean(vals)), 6) if vals else None

    def p95(key: str) -> float | None:
        vals = [float(r[key]) for r in settled if r.get(key) is not None]
        return round(float(np.percentile(vals, 95)), 6) if vals else None

    travelled = sum(float(r["body_step"]) for r in settled
                    if r.get("body_step") is not None)
    commanded = sum(math.hypot(float(r["cmd_vx"]), float(r["cmd_vy"])) * STEP_DT
                    for r in settled
                    if r.get("cmd_vx") is not None and r.get("cmd_vy") is not None)
    slip_travel = sum(float(r["slip_travel"]) for r in settled
                      if r.get("slip_travel") is not None)
    loaded_steps = sum(1 for r in settled
                       if r.get("contact_l") or r.get("contact_r"))
    return {
        "settle_s": float(settle_s),
        "n_steps_settled": len(settled),
        "vx_err_abs_mean": abs_mean("vx_err"),
        "vy_err_abs_mean": abs_mean("vy_err"),
        "yaw_err_abs_mean": abs_mean("yaw_err"),
        "slip_mean_settled": mean("slip"),
        "slip_p95_settled": p95("slip"),
        "travelled_m": round(travelled, 4),
        "commanded_m": round(commanded, 4),
        "dist_err_m": round(abs(travelled - commanded), 4),
        "slip_travel_m": round(slip_travel, 4),
        "slip_ratio": round(slip_travel / max(travelled, T2_SLIP_RATIO_MIN_TRAVEL),
                            4),
        "loaded_step_frac": round(loaded_steps / max(1, len(settled)), 4),
    }


#: provisional gates.  Thresholds are placeholders (no baseline tuning yet);
#: S2 sets them from measured baselines and the held-out battery.
#: the shared terminal-stance criterion (operator directive 2026-10-08): every
#: episode that does NOT terminate must END inside a valid stance -- the check the
#: fall detector cannot supply (a robot lying on a supporting arm does not
#: terminate; see solo/fall.py).  This does NOT change any existing threshold; it
#: only appends.  Threshold 1.0 is the measured reference (StandHold 11/11
#: survivors, in-band battery); PROVISIONAL for tasks whose own reference run has
#: not been measured -- a run of that task's reference over its gate battery sets
#: it.
_ENDS_IN_VALID_STANCE = Criterion(
    "survivor_valid_stance_rate", ">=", 1.0,
    "TERMINAL state: every non-terminated episode must END inside a valid stance "
    "(solo.stance_valid: hull margin >= 0.02 m, tilt <= 5 deg, pelvis within "
    "0.06 m of 0.79, speed <= 0.15 m/s, pose within 0.10 rad, both feet in "
    "contact). Threshold 1.0 = measured StandHold (11/11 survivors); PROVISIONAL "
    "for this task until its own reference run sets it")


GATES: dict[str, TaskGate] = {
    "balance": TaskGate(
        "T1_balance",
        (Criterion("fall_rate", "<=", 0.05,
                   "falls across the IN-BAND battery (magnitudes <= "
                   "TRAIN_MAX_IMPULSE = 12 N*s, all heights); provenance: "
                   "StandHold measured 0.041667"),
         Criterion("mean_upright", ">=", 0.84,
                   "provenance: below StandHold's in-band hold 0.913239 AND the "
                   "stored full-battery hold 0.849118; the previous 0.95 bar "
                   "exceeded the reference and was unattainable by construction"),
         Criterion("time_to_stability_mean", "<=", 1.0,
                   "s; upright+stance+speed<0.15 m/s held 0.2 s; provenance: baseline table"),
         Criterion("com_offset_max", "<=", 0.20,
                   "m; CoM-to-support-centre proxy (NOT a hull margin); "
                   "provenance: baseline table"),
         Criterion("fall_rate_heldout", "<=", 0.10,
                   "true fall rate over the held-out CONDITIONS = off-training "
                   "application heights {0.79,1.10} m inside the in-band battery "
                   "(training pushes are all at 0.95 m, curriculum.py); "
                   "provenance: StandHold measured 0.0625 (16 eps). The "
                   "16/20/25 N*s magnitudes are REPORTED-ONLY and do not gate"),
         Criterion("survivor_valid_stance_rate", ">=", 1.0,
                   "TERMINAL state: every non-terminated episode must END inside a "
                   "valid stance (solo.stance_valid: hull margin >= 0.02 m, tilt <= 5 "
                   "deg, pelvis within 0.06 m of 0.79, speed <= 0.15 m/s, pose within "
                   "0.10 rad, both feet in contact). Provenance: StandHold measured "
                   "1.0 (11/11 survivors) on the in-band battery. This is the check "
                   "the fall detector cannot supply -- a robot lying on a supporting "
                   "arm does not terminate (see solo/fall.py)")),
        note="T1 = NON-STEPPING dynamic balance only, evaluated on battery "
             "magnitudes <= TRAIN_MAX_IMPULSE; held-out = off-training heights "
             "(not above-cap magnitudes, which would be vacuous once capped); the "
             "stepping-recovery bars move to GATES['stance'] "
             "(reports/2026-10-08/t1_gate_calibration.md)"),
    "locomotion": TaskGate(
        "T2_locomotion",
        (Criterion("vx_err_abs_mean", "<=", T2_THRESHOLDS["vx_err_abs_mean"],
                   "m/s; 25% of the smallest held-out |vx| (0.40 m/s); "
                   "provenance: baseline table (StandHold+tiny 0.02 PASS, "
                   "StandHold@held-out 0.30 FAIL)"),
         Criterion("vy_err_abs_mean", "<=", T2_THRESHOLDS["vy_err_abs_mean"],
                   "m/s; 25% of the smallest held-out |vy| (0.18 m/s); "
                   "provenance: baseline table"),
         Criterion("yaw_err_abs_mean", "<=", T2_THRESHOLDS["yaw_err_abs_mean"],
                   "rad/s; 33% of the smallest held-out |wz| (0.45 rad/s); "
                   "provenance: baseline table"),
         Criterion("mean_upright", ">=", T2_THRESHOLDS["mean_upright"],
                   "tracking must not be bought with a topple; provenance: "
                   "StandHold 0.9999 vs FallForward 0.76"),
         Criterion("fall_rate", "<=", T2_THRESHOLDS["fall_rate"],
                   "falls per episode; provenance: every displacing baseline falls"),
         Criterion("dorsal_rate", "<=", T2_THRESHOLDS["dorsal_rate"],
                   "back-to-mat terminations (also a fall here)"),
         Criterion("slip_mean", "<=", T2_THRESHOLDS["slip_mean"],
                   "m/s loaded-foot speed (point sample); provenance: standing "
                   "<=0.008, sliding >=0.098"),
         Criterion("slip_ratio_mean", "<=", T2_THRESHOLDS["slip_ratio_mean"],
                   "m loaded-foot travel per m travelled -- sliding instead of "
                   "stepping; provenance: load-transfer <=0.57, drag >=1.75"),
         Criterion("dist_err_mean", "<=", T2_THRESHOLDS["dist_err_mean"],
                   "m; |travelled - commanded| path length; 15% of the smallest "
                   "held-out commanded path (2.2 m); provenance: baseline table"),
         _ENDS_IN_VALID_STANCE),
        provisional=False,
        note="T2 (S3) gate: held-out commands outside T2_TRAIN_RANGES; measured "
             "over the settled window (t >= 0.5 s); thresholds DERIVED from "
             "data/solo/metrics/t2_gate_baselines.json -- see "
             "reports/2026-10-08/t2_gate.md. No reward term is a criterion."),
    "stance": TaskGate(
        "T3_stance",
        (Criterion("stance_err_mean", "<=", 0.08, "m pelvis-height error"),
         Criterion("mean_upright", ">=", 0.95),
         Criterion("fall_rate", "<=", 0.05),
         # moved out of T1: stepping recovery is T3/T2 capability. Thresholds are
         # PLACEHOLDERS pending calibration against a stepping-capable reference
         # (v5/T3) -- not against stand_hold, whose held-out survival is 1/24.
         Criterion("max_recoverable_impulse", ">=", 16.0,
                   "N*s; REQUIRES re-calibration against a stepping controller"),
         Criterion("max_recoverable_impulse_heldout", ">=", 16.0,
                   "N*s; recovered = stable to stance (not merely non-terminated); "
                   "REQUIRES re-calibration (v5/T3)"),
         Criterion("fall_rate_heldout", "<=", 0.10,
                   "held-out magnitudes only (magnitudes > train_max_impulse)"),
         Criterion("recovery_success_rate", ">=", 0.9,
                   "stabilised to stance after each held-out push"),
         _ENDS_IN_VALID_STANCE),
        note="T3 stance/stepping; owns the recovery bars moved from T1"),
    "reach": TaskGate(
        "T4_reach",
        (Criterion("hand_err_mean", "<=", 0.20, "m hand-to-target"),
         Criterion("mean_upright", ">=", 0.95),
         Criterion("fall_rate", "<=", 0.05),
         _ENDS_IN_VALID_STANCE)),
    "shot": TaskGate(
        "T5_shot",
        (Criterion("shot_depth_max", ">=", 0.8, "penetration depth reached"),
         Criterion("shot_exit_rate", ">=", 0.5, "shots that exit and recover"),
         Criterion("fall_rate", "<=", 0.10))),
    # NOTE: T5_shot deliberately omits the terminal-stance criterion -- its end
    # state is a penetration posture, not a stance; a stance requirement there
    # would contradict the task.  T6_recovery DOES end in a posture (standing up).
    "recovery": TaskGate(
        "T6_recovery",
        (Criterion("mean_pelvis_z", ">=", 0.70, "m after standing up"),
         Criterion("fall_rate", "<=", 0.10),
         _ENDS_IN_VALID_STANCE)),
}


#: held-out training cap, owned by :mod:`solo.pushes` (see there)

#: the single application height used by training
#: (``solo.curriculum.PushCurriculum.height`` = 0.95 m); battery pushes at other
#: heights are held out by *condition*, so a battery capped to the in-band
#: magnitudes still has a non-empty held-out set.
TRAIN_PUSH_HEIGHT = 0.95


def battery_pushes(*, magnitudes: Sequence[float] = (4.0, 8.0, 12.0, 16.0, 20.0, 25.0),
                   directions: int = 8, heights: Sequence[float] = (0.79, 0.95, 1.10),
                   seed: int = 0, t: float = 1.0, jitter: float = 0.25,
                   train_max_impulse: float = TRAIN_MAX_IMPULSE) -> list[PushSpec]:
    """Seeded held-out push battery (one push per episode, deterministic).

    Design (2026-10-08, R2; held-out redefined 2026-10-08 R2b): magnitudes span
    the stiff-stand-survivable range and 20/25 N*s go beyond the derived
    direction-dependent non-stepping ceiling (6.7-20.7 N*s, see
    ``reports/2026-10-08/t1_gate_calibration.md`` section 2.2); application
    heights cycle over pelvis/chest/upper-chest (0.79/0.95/1.10 m) and
    directions cover 8 evenly spaced world yaws.

    An episode is labelled ``_heldout`` iff it is **above the training cap OR at
    an application height training never uses** (training pushes are all at
    ``TRAIN_PUSH_HEIGHT`` = 0.95 m).  Within the in-band magnitudes (<=
    ``train_max_impulse``) this makes the held-out set the off-training heights,
    which stays non-empty when the T1 gate caps the battery to the band.
    Training schedules must stay at or below ``train_max_impulse``.
    """
    rng = np.random.default_rng(int(seed))
    slots = max(1, int(round(float(jitter) / STEP_DT)))
    out: list[PushSpec] = []
    for i, m in enumerate(magnitudes):
        for j in range(int(directions)):
            ang = 2.0 * np.pi * j / int(directions)
            h = float(heights[(i + j) % len(heights)])
            t0 = float(t) + STEP_DT * int(rng.integers(0, slots + 1))
            # held out = above the training cap OR off the single training
            # application height (curriculum.PushCurriculum.height = 0.95 m).
            # Within the in-band battery this makes the held-out set the
            # off-training HEIGHTS, so capping the battery to the band does not
            # silently empty it.
            held = (float(m) > float(train_max_impulse) + 1e-9
                    or abs(float(h) - TRAIN_PUSH_HEIGHT) > 1e-9)
            label = (f"battery_m{round(float(m), 1)}_d{j}_h{h:.2f}"
                     + ("_heldout" if held else ""))
            out.append(PushSpec(t=t0, impulse=float(m), direction=ang, height=h,
                                duration=STEP_DT, label=label))
    return out


def run_episode(env: SoloEnv, controller: Callable, seed: int, *,
                push: PushSchedule | None = None,
                command: CommandSchedule | None = None,
                clip=None, max_steps: int | None = None) -> dict:
    """Run one episode through ``env`` with ``controller``; returns its summary.

    ``controller(env, data) -> (29,)`` ctrl targets (absolute joint positions).
    ``clip`` (a :class:`solo.video.ClipRecorder`) optionally captures frames.
    """
    env.reset(seed=seed, push=push, command=command)
    t0 = time.perf_counter()
    steps = 0
    cause = None
    truncated = False
    while True:
        action = controller(env, env.data)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if clip is not None:
            clip.capture(env, info)
        if terminated or truncated:
            cause = info.get("termination")
            break
        if max_steps is not None and steps >= max_steps:
            break
    wall = time.perf_counter() - t0
    rows = env.recorder.rows
    reward_sum = float(sum(r["reward"] for r in rows if r.get("reward") is not None))
    hand_err_mean = None
    hand_vals = [r["hand_err"] for r in rows if r.get("hand_err") is not None]
    if hand_vals:
        hand_err_mean = round(float(np.mean(hand_vals)), 6)
    pushes = list(push.pushes) if push is not None else []
    if len(pushes) == 1:
        p = pushes[0]
        push_end = p.t + p.n_steps * STEP_DT
        rec = recovery_time(rows, push_end, pelvis_nominal=float(env._q_stand[2]))
        com_max = com_offset_max(rows, push_end)
        after = [r for r in rows if float(r["t"]) >= push_end - 1e-9]
        steps_after = (int(after[-1]["steps_taken"]) - int(after[0]["steps_taken"])
                       if after else 0)
        heldout = p.label.endswith("_heldout")
        impulse = float(p.impulse)
    else:
        rec = com_max = None
        steps_after = 0
        heldout = False
        impulse = None
    # terminal-state stance validity (operator directive 2026-10-08): measured
    # from the env's final state, not the episode average.
    stance_ok, stance_reasons, stance_channels = _terminal_stance(env)
    summary = env.recorder.summary(extra={
        "termination": cause,
        "truncated": bool(truncated),
        "steps": int(steps),
        "duration_s": round(steps * STEP_DT, 4),
        "wall_s": round(wall, 4),
        "steps_per_s": round(steps / wall, 2) if wall > 0 else None,
        "pushes": [p.as_dict() for p in pushes],
        "impulse": impulse,
        "heldout": bool(heldout),
        "recovery_time_s": None if rec is None else round(rec, 4),
        "recovered": bool(rec is not None),
        "time_to_stability_s": None if rec is None else round(rec, 4),
        "stable": bool(rec is not None),
        "com_offset_max_m": None if com_max is None else round(com_max, 4),
        "steps_after_push": int(steps_after),
        "steps_total": int(rows[-1]["steps_taken"]) if rows else 0,
        "reward_sum": round(reward_sum, 6),
        "final_pelvis_z": round(float(env.data.qpos[2]), 4),
        "final_upright": round(float(env.data.xmat[
            env._torso_bid].reshape(3, 3)[2, 2]), 4),
        "shot_depth_max": round(float(getattr(env, "_shot_max_depth", 0.0)), 4),
        "shot_exited": bool(getattr(env, "_shot_exited", False)),
        "hand_err": hand_err_mean,
        "ends_in_valid_stance": bool(stance_ok),
        "stance_invalid_reasons": stance_reasons,
        "final_stance": _rounded_stance(stance_channels),
        **(locomotion_episode_metrics(rows) if env.task == "locomotion" else {}),
    })
    summary["__rows"] = rows
    return summary


def _terminal_stance(env) -> tuple[bool, list[str], dict]:
    """(valid, reasons, channels) for the env's CURRENT (terminal) state."""
    state = stance_state(env)
    ok, reasons = stance_valid(**state)
    return ok, reasons, state


def _rounded_stance(state: dict) -> dict:
    out: dict = {}
    for k, v in state.items():
        out[k] = (bool(v) if isinstance(v, (bool, np.bool_))
                  else None if v is None or (isinstance(v, float) and np.isnan(v))
                  else round(float(v), 5))
    return out


def _recovered_episode(e: dict) -> bool:
    """A push episode counts as *recovered* iff it did not terminate and reached
    a *stable* stance afterwards (upright + stance height + base speed below
    0.15 m/s, held 0.2 s; see ``metrics.recovery_time``).  This pairs survival
    with actually being on the feet again -- without it, "survived" would count a
    knocked-over robot that merely failed to terminate (e.g. lying on an arm)."""
    if e.get("termination") is not None:
        return False
    return bool(e.get("stable"))


def _episode_metric(summary: dict, metric: str) -> float | None:
    """Episode-level value for an aggregate metric name."""
    m = summary.get("metrics", {})
    if metric == "mean_upright":
        return (m.get("upright") or {}).get("mean")
    if metric == "min_upright":
        return (m.get("upright") or {}).get("min")
    if metric == "mean_tilt_deg":
        return (m.get("tilt_deg") or {}).get("mean")
    if metric == "mean_pelvis_z":
        return (m.get("pelvis_z") or {}).get("mean")
    if metric == "vel_err_mean":
        return (m.get("vel_err") or {}).get("mean")
    if metric == "vel_err_p95":
        return (m.get("vel_err") or {}).get("p95")
    if metric == "yaw_err_mean":
        return (m.get("yaw_err") or {}).get("mean")
    if metric == "yaw_err_p95":
        return (m.get("yaw_err") or {}).get("p95")
    if metric == "stance_err_mean":
        return (m.get("stance_err") or {}).get("mean")
    if metric == "slip_mean":
        # locomotion episodes carry the settled-window value (see
        # locomotion_episode_metrics); other tasks keep the episode-wide stat
        if summary.get("slip_mean_settled") is not None:
            return summary["slip_mean_settled"]
        return (m.get("slip") or {}).get("mean")
    if metric == "slip_p95":
        if summary.get("slip_p95_settled") is not None:
            return summary["slip_p95_settled"]
        return (m.get("slip") or {}).get("p95")
    # T2 (locomotion) per-episode quantities, recorded by run_episode
    if metric in ("vx_err_abs_mean", "vy_err_abs_mean", "yaw_err_abs_mean",
                  "travelled_m", "commanded_m", "dist_err_m", "slip_travel",
                  "slip_ratio", "loaded_step_frac"):
        return summary.get(metric)
    if metric == "dist_err_mean":
        return summary.get("dist_err_m")
    if metric == "slip_ratio_mean":
        return summary.get("slip_ratio")
    if metric == "travelled_m_mean":
        return summary.get("travelled_m")
    if metric == "commanded_m_mean":
        return summary.get("commanded_m")
    if metric == "slip_travel_mean":
        return summary.get("slip_travel_m")
    if metric == "loaded_step_frac_mean":
        return summary.get("loaded_step_frac")
    if metric == "act_delta_mean":
        return (m.get("act_delta") or {}).get("mean")
    if metric == "sat_frac_mean":
        return (m.get("sat_frac") or {}).get("mean")
    if metric == "limit_prox_max":
        return (m.get("limit_prox") or {}).get("max")
    if metric == "reward_mean":
        return (m.get("reward") or {}).get("mean")
    if metric == "reward_sum_mean":
        return summary.get("reward_sum")
    if metric == "shot_depth_max":
        return summary.get("shot_depth_max")
    if metric == "hand_err_mean" or metric == "hand_err":
        return summary.get("hand_err")
    raise KeyError(f"unknown aggregate metric {metric!r}")


def aggregate(episodes: list[dict], *, steps_total: int, wall_total: float,
              extra: dict | None = None) -> dict:
    """Aggregate episode summaries into the run summary gates consume."""
    n = len(episodes)
    if n == 0:
        raise ValueError("no episodes")

    def mean_of(key: str) -> float | None:
        vals = [e[key] for e in episodes if e.get(key) is not None]
        return round(float(np.mean(vals)), 6) if vals else None

    def mean_metric(metric: str) -> float | None:
        vals = [v for v in (_episode_metric(e, metric) for e in episodes)
                if v is not None]
        return round(float(np.mean(vals)), 6) if vals else None

    falls = sum(1 for e in episodes if e.get("termination") == "fall")
    dorsals = sum(1 for e in episodes if e.get("termination") == "dorsal")
    terms = sum(1 for e in episodes if e.get("termination") is not None)
    survivors = [e for e in episodes if e.get("termination") is None]
    rec_ok = [e for e in episodes if e.get("recovered")]
    push_eps = [e for e in episodes if e.get("pushes")]
    heldout_eps = [e for e in episodes if e.get("heldout")]
    impulses = [(e.get("pushes") or [{}])[0].get("impulse") if e.get("pushes") else None
                for e in episodes]
    survived = [imp for e, imp in zip(episodes, impulses)
                if imp is not None and _recovered_episode(e)]
    survived_heldout = [float(e["impulse"]) for e in heldout_eps
                        if e.get("impulse") is not None and _recovered_episode(e)]
    com_vals = [float(e["com_offset_max_m"]) for e in push_eps
                if e.get("com_offset_max_m") is not None]
    stab_vals = [float(e["time_to_stability_s"]) for e in episodes
                 if e.get("time_to_stability_s") is not None]
    out = {
        "n_episodes": n,
        "n_steps": int(steps_total),
        "steps_per_s": round(steps_total / wall_total, 2) if wall_total > 0 else None,
        "fall_rate": round(falls / n, 6),
        "dorsal_rate": round(dorsals / n, 6),
        "termination_rate": round(terms / n, 6),
        "mean_upright": mean_metric("mean_upright"),
        "min_upright": min((v for v in (_episode_metric(e, "min_upright")
                                        for e in episodes) if v is not None),
                           default=None),
        "mean_tilt_deg": mean_metric("mean_tilt_deg"),
        "mean_pelvis_z": mean_metric("mean_pelvis_z"),
        "vel_err_mean": mean_metric("vel_err_mean"),
        "vel_err_p95": mean_metric("vel_err_p95"),
        "yaw_err_mean": mean_metric("yaw_err_mean"),
        "yaw_err_p95": mean_metric("yaw_err_p95"),
        "stance_err_mean": mean_metric("stance_err_mean"),
        "slip_mean": mean_metric("slip_mean"),
        "slip_p95": mean_metric("slip_p95"),
        "act_delta_mean": mean_metric("act_delta_mean"),
        "sat_frac_mean": mean_metric("sat_frac_mean"),
        "limit_prox_max": max((v for v in (_episode_metric(e, "limit_prox_max")
                                           for e in episodes) if v is not None),
                              default=None),
        "reward_mean": mean_metric("reward_mean"),
        "reward_sum_mean": mean_of("reward_sum"),
        "max_recoverable_impulse": round(max(survived), 4) if survived else 0.0,
        "max_recoverable_impulse_heldout": (round(max(survived_heldout), 4)
                                            if survived_heldout else 0.0),
        # same predicate as fall_rate (falls, not any termination)
        "fall_rate_heldout": round(sum(1 for e in heldout_eps
                                       if e.get("termination") == "fall")
                                   / max(1, len(heldout_eps)), 6),
        "recovery_success_rate": round(
            sum(1 for e in push_eps if _recovered_episode(e)) / max(1, len(push_eps)), 6),
        "recovery_time_mean": round(float(np.mean(
            [e["recovery_time_s"] for e in rec_ok])), 4) if rec_ok else None,
        "time_to_stability_mean": round(float(np.mean(stab_vals)), 4) if stab_vals else None,
        "time_to_stability_rate": round(len(stab_vals) / max(1, len(push_eps)), 6),
        "com_offset_max": round(max(com_vals), 4) if com_vals else None,
        "steps_after_push_mean": round(float(np.mean(
            [e.get("steps_after_push", 0) for e in push_eps])), 3) if push_eps else None,
        "steps_total_mean": round(float(np.mean(
            [e.get("steps_total", 0) for e in episodes])), 3),
        "shot_depth_max": max((e.get("shot_depth_max") or 0.0) for e in episodes),
        "shot_exit_rate": round(sum(1 for e in episodes if e.get("shot_exited")) / n, 6),
        "hand_err_mean": mean_metric("hand_err"),
        # T2 (locomotion) -- settled-window tracking / slip / distance
        "vx_err_abs_mean": mean_metric("vx_err_abs_mean"),
        "vy_err_abs_mean": mean_metric("vy_err_abs_mean"),
        "yaw_err_abs_mean": mean_metric("yaw_err_abs_mean"),
        "slip_ratio_mean": mean_metric("slip_ratio_mean"),
        "dist_err_mean": mean_metric("dist_err_mean"),
        "travelled_m_mean": mean_metric("travelled_m_mean"),
        "commanded_m_mean": mean_metric("commanded_m_mean"),
        "slip_travel_mean": mean_metric("slip_travel_mean"),
        "loaded_step_frac_mean": mean_metric("loaded_step_frac_mean"),
        # terminal-state stance validity (one shared predicate).  The
        # unconditional rate counts a terminated (fallen) episode as invalid;
        # the survivor rate asks the question the fall detector cannot: of the
        # episodes that did NOT terminate, how many actually END on the feet?
        # (None -- never a silent pass -- when no episode survived.)
        "ends_in_valid_stance": round(
            sum(1 for e in episodes if e.get("ends_in_valid_stance")) / n, 6),
        "survivor_valid_stance_rate": (
            round(sum(1 for e in episodes
                      if e.get("termination") is None and e.get("ends_in_valid_stance"))
                  / len(survivors), 6) if survivors else None),
    }
    if extra:
        out.update(extra)
    return out


def evaluate(controller_factory: Callable[[SoloEnv, int], Callable], *,
             task: str = "balance", episodes: int = 8, seed0: int = 0,
             gate: TaskGate | None = None,
             push_plan: Sequence[PushSpec] | None = None,
             command_plan: Sequence[CommandSchedule | None] | None = None,
             env_kwargs: dict | None = None, out_dir: str | Path | None = None,
             name: str = "controller",
             clip_factory: Callable[[int, int, SoloEnv], object] | None = None,
             max_episode_s: float | None = None,
             verbose: bool = True) -> dict:
    """Run ``episodes`` through the same harness; return the gated report.

    ``push_plan``: one :class:`PushSpec` per episode (battery).  ``clip_factory``
    is called as ``clip_factory(episode_index, seed, env)`` *before* each episode
    and must return a :class:`solo.video.ClipRecorder` (captured live during the
    rollout); the recorders are returned under ``report["__clips"]`` (they are
    excluded from the JSON summary) so the caller can render media *after* the
    run verdict is known.
    """
    model = load_solo_model()
    env = SoloEnv(model=model, task=task, seed=seed0, **(env_kwargs or {}))
    if max_episode_s is not None:
        env.horizon = float(max_episode_s)
    gate = gate or GATES[task]
    n_ep = len(push_plan) if push_plan is not None else int(episodes)
    summaries: list[dict] = []
    clips: list[object] = []
    steps_total = 0
    wall_total = 0.0
    all_rows: list[dict] = []
    for i in range(n_ep):
        seed = int(seed0) + i
        push = None
        if push_plan is not None:
            push = PushSchedule([push_plan[i]])
        command = None
        if command_plan is not None and i < len(command_plan):
            command = command_plan[i]
        controller = controller_factory(env, seed)
        clip = clip_factory(i, seed, env) if clip_factory is not None else None
        s = run_episode(env, controller, seed, push=push, command=command, clip=clip)
        if clip is not None:
            clips.append(clip)
        all_rows.extend(s.get("__rows", []))
        steps_total += int(s["steps"])
        wall_total += float(s["wall_s"])
        summaries.append(s)
        if verbose:
            print(f"  [{name}] ep {i + 1}/{n_ep} seed={seed} steps={s['steps']} "
                  f"term={s['termination']} upright={_episode_metric(s, 'mean_upright')} "
                  f"vel_err={_episode_metric(s, 'vel_err_mean')}")
    agg = aggregate(summaries, steps_total=steps_total, wall_total=wall_total,
                    extra={"reward_sum_mean": round(float(np.mean(
                        [s.get("reward_sum", 0.0) for s in summaries])), 6)
                        if summaries else None})
    ok, reasons = gate.verdict(agg)
    report = {
        "task": task, "controller": name, "seed0": int(seed0),
        "n_episodes": n_ep, "gate": gate.as_dict(),
        "verdict": "certified" if ok else "not_certified",
        "certified": bool(ok), "reasons": reasons,
        "aggregate": agg,
        "config": env.config(),
        "episodes": [{k: v for k, v in s.items()
                      if not k.startswith("__") and k != "metrics"} | {
                         "metric_means": {mk: mv.get("mean")
                                          for mk, mv in s["metrics"].items()}}
                     for s in summaries],
        "__clips": clips,
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        from .metrics import write_json, write_jsonl
        if all_rows:
            write_jsonl(out / f"{task}_{name}_s{seed0}.jsonl", all_rows)
        write_json(out / f"{task}_{name}_s{seed0}.summary.json",
                   {k: v for k, v in report.items() if not k.startswith("__")})
    return report


def take_clips(report: dict) -> list:
    """Pop the ClipRecorder objects captured by :func:`evaluate`."""
    return report.pop("__clips", [])


def write_traces(env: SoloEnv, path: str | Path, rows: list[dict] | None = None) -> Path:
    """Write per-step JSONL rows (from ``env.recorder`` or an explicit list)."""
    from .metrics import write_jsonl

    return write_jsonl(path, env.recorder.rows if rows is None else rows)


if __name__ == "__main__":  # self-check
    from .baselines import StandHoldController

    rep = evaluate(lambda env, seed: StandHoldController(), task="balance",
                   episodes=2, seed0=0, verbose=False)
    print("solo.eval self-check OK:", {
        "verdict": rep["verdict"], "n_episodes": rep["n_episodes"],
        "agg_keys": len(rep["aggregate"]), "reasons0": rep["reasons"][0]})
