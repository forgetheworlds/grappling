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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .commands import CommandSchedule
from .env import SoloEnv
from .metrics import METRICS_DIR, METRIC_FIELDS, recovery_time, summarize_rows
from .pushes import PushSchedule, PushSpec
from .scene import STEP_DT, load_solo_model

#: all metrics a gate may reference (aggregate keys produced by :func:`evaluate`)
GATE_METRICS: tuple[str, ...] = (
    "fall_rate", "dorsal_rate", "termination_rate",
    "mean_upright", "min_upright", "mean_tilt_deg", "mean_pelvis_z",
    "vel_err_mean", "vel_err_p95", "yaw_err_mean", "yaw_err_p95",
    "stance_err_mean", "slip_mean", "slip_p95",
    "act_delta_mean", "sat_frac_mean", "limit_prox_max",
    "reward_mean", "reward_sum_mean",
    "max_recoverable_impulse", "recovery_success_rate", "recovery_time_mean",
    "steps_per_s", "n_episodes", "n_steps",
    "hand_err_mean", "shot_depth_max", "shot_exit_rate",
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


#: provisional gates.  Thresholds are placeholders (no baseline tuning yet);
#: S2 sets them from measured baselines and the held-out battery.
GATES: dict[str, TaskGate] = {
    "balance": TaskGate(
        "T1_balance",
        (Criterion("fall_rate", "<=", 0.05, "falls across the push battery"),
         Criterion("mean_upright", ">=", 0.95, "uprightness while standing"),
         Criterion("max_recoverable_impulse", ">=", 10.0,
                   "highest impulse survived (N*s); 12 N*s = unseen in training"),
         Criterion("recovery_success_rate", ">=", 0.9, "pushes recovered")),
        note="max magnitude in the battery (12 N*s) is unseen in training"),
    "locomotion": TaskGate(
        "T2_locomotion",
        (Criterion("vel_err_mean", "<=", 0.15, "m/s"),
         Criterion("yaw_err_mean", "<=", 0.25, "rad/s"),
         Criterion("mean_upright", ">=", 0.97, "tracking must not be bought with falls"),
         Criterion("fall_rate", "<=", 0.05)),
        note="tracking is always paired with uprightness"),
    "stance": TaskGate(
        "T3_stance",
        (Criterion("stance_err_mean", "<=", 0.08, "m pelvis-height error"),
         Criterion("mean_upright", ">=", 0.95),
         Criterion("fall_rate", "<=", 0.05))),
    "reach": TaskGate(
        "T4_reach",
        (Criterion("hand_err_mean", "<=", 0.20, "m hand-to-target"),
         Criterion("mean_upright", ">=", 0.95),
         Criterion("fall_rate", "<=", 0.05))),
    "shot": TaskGate(
        "T5_shot",
        (Criterion("shot_depth_max", ">=", 0.8, "penetration depth reached"),
         Criterion("shot_exit_rate", ">=", 0.5, "shots that exit and recover"),
         Criterion("fall_rate", "<=", 0.10))),
    "recovery": TaskGate(
        "T6_recovery",
        (Criterion("mean_pelvis_z", ">=", 0.70, "m after standing up"),
         Criterion("fall_rate", "<=", 0.10))),
}


def battery_pushes(*, magnitudes: Sequence[float] = (4.0, 6.0, 8.0, 10.0, 12.0),
                   directions: int = 16, seed: int = 0, t: float = 1.0,
                   jitter: float = 0.25, height: float = 0.95) -> list[PushSpec]:
    """Seeded held-out push battery (one push per episode, deterministic)."""
    return list(PushSchedule.battery(magnitudes=magnitudes, directions=directions,
                                     seed=seed, t=t, jitter=jitter,
                                     height=height).pushes)


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
        rec = recovery_time(rows, p.t + p.n_steps * STEP_DT,
                            pelvis_nominal=float(env._q_stand[2]))
    else:
        rec = None
    summary = env.recorder.summary(extra={
        "termination": cause,
        "truncated": bool(truncated),
        "steps": int(steps),
        "duration_s": round(steps * STEP_DT, 4),
        "wall_s": round(wall, 4),
        "steps_per_s": round(steps / wall, 2) if wall > 0 else None,
        "pushes": [p.as_dict() for p in pushes],
        "recovery_time_s": None if rec is None else round(rec, 4),
        "recovered": bool(rec is not None),
        "reward_sum": round(reward_sum, 6),
        "final_pelvis_z": round(float(env.data.qpos[2]), 4),
        "final_upright": round(float(env.data.xmat[
            env._torso_bid].reshape(3, 3)[2, 2]), 4),
        "shot_depth_max": round(float(getattr(env, "_shot_max_depth", 0.0)), 4),
        "shot_exited": bool(getattr(env, "_shot_exited", False)),
        "hand_err": hand_err_mean,
    })
    summary["__rows"] = rows
    return summary


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
        return (m.get("slip") or {}).get("mean")
    if metric == "slip_p95":
        return (m.get("slip") or {}).get("p95")
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
    rec_ok = [e for e in episodes if e.get("recovered")]
    impulses = [(e.get("pushes") or [{}])[0].get("impulse") if e.get("pushes") else None
                for e in episodes]
    survived = [imp for e, imp in zip(episodes, impulses)
                if imp is not None and e.get("termination") is None]
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
        "recovery_success_rate": round(len(rec_ok) / max(1, sum(
            1 for e in episodes if e.get("pushes"))), 6),
        "recovery_time_mean": round(float(np.mean(
            [e["recovery_time_s"] for e in rec_ok])), 4) if rec_ok else None,
        "shot_depth_max": max((e.get("shot_depth_max") or 0.0) for e in episodes),
        "shot_exit_rate": round(sum(1 for e in episodes if e.get("shot_exited")) / n, 6),
        "hand_err_mean": mean_metric("hand_err"),
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
