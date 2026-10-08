#!/usr/bin/env python3
"""T2 (locomotion) gate: held-out measurement of the baselines + the gate table.

Runs every reference controller through the *same* harness the T2 training run
will be judged by (``solo.eval.evaluate``) on the gate's **held-out** command
set (``solo.eval.heldout_command_plan``), prints the per-controller table with
the deciding metrics, and writes the provenance JSON the thresholds are derived
from:

    data/solo/metrics/t2_gate_baselines.json

Nothing here trains anything.  Episodes are short (``--episode-s``) and the
whole run is a few minutes on one core.  The reference case (StandHold under
the tiny, non-held-out command set) must be *certified*: a gate the reference
cannot pass is vacuous, so the script reports any criterion it cannot meet as
an explicit UNATTAINABLE finding instead of hiding it.

Usage:
    MUJOCO_GL=egl .venv/bin/python scripts/solo_t2_gate.py [--quick] [--no-lock]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solo.baselines import (AirborneTransferController,  # noqa: E402
                            FallForwardController, PlantedFootDragController,
                            RandomInitPolicyController, StandHoldController,
                            ZeroActionController)
from solo.eval import (GATES, T2_EPISODE_S, T2_HELDOUT_MARGIN,  # noqa: E402
                       T2_SETTLE_S, T2_TRAIN_RANGES, T2_THRESHOLDS,
                       evaluate, heldout_command_outside, heldout_command_plan,
                       take_clips, tiny_command_plan)
from solo.lock import SimLock  # noqa: E402
from solo.metrics import METRICS_DIR, write_json  # noqa: E402

#: the four baselines the T2 thresholds are derived from, in table order
BASELINE_CONTROLLERS = {
    "stand_hold": lambda env, seed: StandHoldController(),
    "zero_action": lambda env, seed: ZeroActionController(),
    "fall_forward": lambda env, seed: FallForwardController(),
    "random_init_policy": lambda env, seed: RandomInitPolicyController(seed=seed),
}
#: metric-relevant probes (not policies): the slide/step discrimination pair
PROBE_CONTROLLERS = {
    "planted_foot_drag": lambda env, seed: PlantedFootDragController(),
    "airborne_transfer": lambda env, seed: AirborneTransferController(),
}
#: the metric columns the gate decides on (in criteria order)
TABLE_METRICS = ("vx_err_abs_mean", "vy_err_abs_mean", "yaw_err_abs_mean",
                 "mean_upright", "fall_rate", "dorsal_rate", "slip_mean",
                 "slip_ratio_mean", "dist_err_mean")


def _row(report: dict) -> dict:
    agg = report["aggregate"]
    row = {"verdict": report["verdict"], "n_episodes": report["n_episodes"],
           "reasons": list(report["reasons"])}
    row.update({m: agg.get(m) for m in TABLE_METRICS})
    row.update({m: agg.get(m) for m in ("travelled_m_mean", "commanded_m_mean",
                                        "slip_travel_mean", "steps_total_mean",
                                        "loaded_step_frac_mean", "n_steps",
                                        "termination_rate")})
    return row


def _print_row(name: str, row: dict) -> None:
    print(f"  {name:20s} {row['verdict']:14s} "
          f"vx_err={row['vx_err_abs_mean']} vy_err={row['vy_err_abs_mean']} "
          f"yaw_err={row['yaw_err_abs_mean']} upright={row['mean_upright']} "
          f"fall={row['fall_rate']}/{row['dorsal_rate']} "
          f"slip={row['slip_mean']} slip_ratio={row['slip_ratio_mean']} "
          f"dist_err={row['dist_err_mean']}")


def _run(factory, *, plan, episodes_s: float, name: str) -> dict:
    # ``episodes`` must equal the plan length: evaluate() falls back to the
    # task's own command sampler for episodes beyond the plan, which would
    # silently mix sampled (i.e. non-held-out) commands into the gate run.
    rep = evaluate(factory, task="locomotion", seed0=0, episodes=len(plan),
                   command_plan=plan, gate=GATES["locomotion"],
                   out_dir=METRICS_DIR, max_episode_s=episodes_s,
                   verbose=False, name=name)
    take_clips(rep)
    if len(rep["episodes"]) != len(plan):
        raise RuntimeError(f"{name}: {len(rep['episodes'])} episodes for a "
                           f"{len(plan)}-command plan")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="3 held-out episodes, 3 s each (smoke; not the table)")
    ap.add_argument("--episode-s", type=float, default=T2_EPISODE_S)
    ap.add_argument("--out", default=str(METRICS_DIR / "t2_gate_baselines.json"))
    ap.add_argument("--lock-wait", type=float, default=900.0,
                    help="seconds to wait for the advisory sim lock")
    ap.add_argument("--no-lock", action="store_true",
                    help="skip the advisory sim lock (short, CPU-light runs)")
    args = ap.parse_args()

    if args.no_lock:
        return _run_all(args)
    with SimLock(owner="solo_t2_gate", wait_s=args.lock_wait):
        return _run_all(args)


def _run_all(args) -> int:
    held_plan = heldout_command_plan(episode_s=args.episode_s)
    tiny_plan = tiny_command_plan()
    if args.quick:
        held_plan = held_plan[:3]
        tiny_plan = tiny_plan[:1]

    t0 = time.perf_counter()
    table: dict[str, dict] = {}
    gate = GATES["locomotion"]
    print(f"T2 gate {gate.name}: {len(gate.criteria)} criteria, held-out plan = "
          f"{len(held_plan)} episodes x {args.episode_s:.1f} s")
    print(f"training domain (nothing held out inside it): vx={T2_TRAIN_RANGES.vx} "
          f"vy={T2_TRAIN_RANGES.vy} wz={T2_TRAIN_RANGES.wz}, margin "
          f"{T2_HELDOUT_MARGIN} m/s | measurements from t >= {T2_SETTLE_S} s")
    print("thresholds: " + ", ".join(f"{k}<={v:g}" for k, v in
                                     T2_THRESHOLDS.items()))
    print(f"held-out commands ({len(held_plan)} episodes; last = reversal "
          "schedule):")
    for s in held_plan:
        c = s.at(0.0)
        margins = heldout_command_outside(c)
        print(f"    vx={c.vx:+.2f} vy={c.vy:+.2f} wz={c.wz:+.2f} "
              f"{c.skill.name:10s} margin={max(margins.values()):.2f} "
              f"{ {k: round(v, 2) for k, v in margins.items()} }")
    print("\nbaselines on HELD-OUT commands:")
    for name, factory in BASELINE_CONTROLLERS.items():
        table[name] = _row(_run(factory, plan=held_plan,
                                episodes_s=args.episode_s, name=f"t2gate_{name}"))
        _print_row(name, table[name])
    print("\nmetric probes (not policies) on HELD-OUT commands:")
    for name, factory in PROBE_CONTROLLERS.items():
        table[name] = _row(_run(factory, plan=held_plan[:1],
                                episodes_s=args.episode_s, name=f"t2probe_{name}"))
        _print_row(name, table[name])
    print("\nreference (StandHold, TINY non-held-out commands -- must be certified):")
    table["reference_stand_hold_tiny"] = _row(
        _run(BASELINE_CONTROLLERS["stand_hold"], plan=tiny_plan,
             episodes_s=args.episode_s, name="t2ref_stand_hold_tiny"))
    _print_row("reference_tiny", table["reference_stand_hold_tiny"])

    ref = table["reference_stand_hold_tiny"]
    unattainable = [r for r in ref["reasons"] if r.startswith("FAIL")]
    if unattainable:
        print(f"FINDING: the reference cannot meet {len(unattainable)} "
              f"threshold(s) -- report them as unattainable, not as a pass:")
        for line in unattainable:
            print(f"    UNATTAINABLE {line}")
    else:
        print("VERIFIED: the reference (stand-hold under tiny commands) is "
              "certified by every criterion, so the gate is not vacuously "
              "impossible; it is only the *held-out* commands nothing can track.")

    payload = {
        "gate": gate.as_dict(),
        "thresholds": T2_THRESHOLDS,
        "heldout": {
            "episode_s": args.episode_s,
            "settle_s": T2_SETTLE_S,
            "margin": T2_HELDOUT_MARGIN,
            "train_ranges": T2_TRAIN_RANGES.as_dict(),
            "plan": [s.as_list() or [s.default.as_dict()] for s in held_plan],
        },
        "controllers": table,
        "reference_unattainable": unattainable,
        "wall_s": round(time.perf_counter() - t0, 2),
    }
    write_json(args.out, payload)
    print(f"\nwrote {args.out} ({payload['wall_s']} s wall)")

    bad = [n for n, r in table.items()
           if n in BASELINE_CONTROLLERS and r["verdict"] != "not_certified"]
    if bad:
        print(f"FINDING: baseline(s) unexpectedly certified: {bad}")
    else:
        print("VERIFIED: every T2 baseline (stand-hold, zero-action, "
              "fall-forward, random-init policy) is NOT certified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
