"""T1 mid-run independent read: evaluate the current t1_balance.pt checkpoint.

Uses the trainer's own loading path (SoloTrainer.load -> rl.checkpoint.apply_checkpoint)
and the repo's existing deterministic PolicyController (mean action = tanh(actor mean)),
through the single gating harness solo.eval.evaluate / battery_pushes.

Writes data/solo/metrics/t1_midrun_policy_summary.json (territory-safe, no other files).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from solo.baselines import PolicyController  # noqa: E402
from solo.eval import battery_pushes, evaluate  # noqa: E402
from solo.lock import SimLock  # noqa: E402
from solo.train import SoloTrainer, TrainConfig  # noqa: E402

CKPT = str(REPO / "checkpoints" / "solo" / "t1_balance.pt")
OUT = REPO / "data" / "solo" / "metrics" / "t1_midrun_policy_summary.json"

KEYS = ("fall_rate", "fall_rate_heldout", "mean_upright", "max_recoverable_impulse",
        "max_recoverable_impulse_heldout", "time_to_stability_mean", "time_to_stability_rate",
        "com_offset_max", "recovery_success_rate", "steps_after_push_mean",
        "n_episodes", "n_steps", "steps_per_s")


def main() -> int:
    # exact config of the in-flight run (from its cmdline)
    cfg = TrainConfig(task="balance", steps=2_000_000, rollout_steps=2048,
                      gamma=0.995, out=CKPT)
    t0 = time.perf_counter()
    trainer = SoloTrainer(cfg)
    trainer.load(CKPT)                       # trainer's own loading path
    net = trainer.net
    net.eval()
    load_s = time.perf_counter() - t0

    battery = battery_pushes()               # full 48-push battery, seed 0
    ckpt_stats = {
        "steps_done": trainer.steps_done, "iteration": trainer.iteration,
        "episode_seed": trainer.episode_seed,
        "recent_returns_mean": (round(float(np.mean(trainer.recent_returns)), 4)
                                if trainer.recent_returns else None),
        "n_recent_returns": len(trainer.recent_returns),
        "ckpt_mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(Path(CKPT).stat().st_mtime)),
    }

    t1 = time.perf_counter()
    with SimLock(owner="t1_midrun_read", wait_s=120):
        report = evaluate(lambda env, seed: PolicyController(net, name="t1_policy"),
                          task="balance", push_plan=battery, seed0=0,
                          name="t1_midrun_policy", out_dir=None, verbose=True)
    wall = time.perf_counter() - t1

    agg = report["aggregate"]
    payload = {
        "checkpoint": CKPT,
        "load_s": round(load_s, 3),
        "eval_wall_s": round(wall, 3),
        "determinism": "PolicyController defaults stochastic=False -> "
                       "actor.deterministic_unit = tanh(actor.mean(obs)); net.eval()",
        "verdict": report["verdict"],
        "reasons": report["reasons"],
        "aggregate": {k: agg.get(k) for k in KEYS},
        "aggregate_full": agg,
        "ckpt_stats": ckpt_stats,
        "episodes": [{k: e.get(k) for k in
                      ("termination", "impulse", "heldout", "stable", "recovered",
                       "time_to_stability_s", "com_offset_max_m", "steps_after_push",
                       "mean_upright")} for e in report["episodes"]],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print("\n=== T1 MIDRUN POLICY SUMMARY ===")
    print(json.dumps({k: payload["aggregate"][k] for k in KEYS}, indent=2))
    print("verdict:", report["verdict"])
    for r in report["reasons"]:
        print(" ", r)
    print("ckpt_stats:", json.dumps(ckpt_stats, default=str))
    print("eval_wall_s:", round(wall, 2), "load_s:", round(load_s, 2))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
