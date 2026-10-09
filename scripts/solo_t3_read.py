#!/usr/bin/env python3
"""T3 stance gate read for a solo checkpoint (the monitor is balance-hardwired).

Loads a training checkpoint (action mode / residual scale from its own config),
runs the T3 battery -- in-band pushes (<= TRAIN_MAX_IMPULSE) + held-out magnitudes
at the training height -- judged by ``GATES["stance"]`` (the T3_stance gate:
stance_err_mean, mean_upright, fall_rate, max_recoverable_impulse_heldout,
fall_rate_heldout, recovery_success_rate, survivor_valid_stance_rate), and
writes the report under data/solo/metrics/.

Usage (repo root, venv active)::
    MUJOCO_GL=egl python scripts/solo_t3_read.py \
        --checkpoint checkpoints/solo/t3_stance_v1.pt [--magnitudes 4 8 12 16 20 25]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from rl.checkpoint import apply_checkpoint, load_checkpoint  # noqa: E402
from rl.net import ActorCritic, NetConfig  # noqa: E402
from solo.baselines import PolicyController  # noqa: E402
from solo.eval import (GATES, TRAIN_MAX_IMPULSE, battery_pushes,  # noqa: E402
                       evaluate, take_clips)
from solo.metrics import METRICS_DIR, write_json  # noqa: E402
from solo.obs import ACTOR_DIM, CRITIC_DIM  # noqa: E402
from solo.scene import N_JOINTS  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--magnitudes", type=float, nargs="+",
                    default=(4.0, 8.0, 12.0, 16.0, 20.0, 25.0),
                    help="in-band = <= TRAIN_MAX_IMPULSE (12), held-out = above")
    ap.add_argument("--directions", type=int, default=8)
    ap.add_argument("--height", type=float, default=0.95,
                    help="push application height (m); the T3 held-out axis is "
                         "magnitude, not height")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import mujoco  # noqa: F401  (GL backend must be selected before solo.scene)

    from solo.scene import load_solo_model
    from solo.train import resolve_action_mode

    src = Path(args.checkpoint)
    if not src.exists():
        raise SystemExit(f"checkpoint {src} does not exist")
    tmp = Path("/tmp") / f"t3read_{src.name}"
    shutil.copy2(src, tmp)                      # never read the live file
    ckpt = load_checkpoint(str(tmp))
    steps = int((ckpt.get("state") or {}).get("steps_done", -1))
    tc = (ckpt.get("config") or {}).get("train", {})
    hidden = tuple(tc.get("hidden", (256, 256)))
    mode, residual_scale = resolve_action_mode(ckpt, None)
    net = ActorCritic(ACTOR_DIM, CRITIC_DIM, act_dim=N_JOINTS,
                      cfg=NetConfig(hidden=hidden, action_mode=mode,
                                    residual_scale=residual_scale))
    apply_checkpoint(ckpt, policy=net)
    net.eval()
    print(f"[t3-read] checkpoint steps={steps} action_mode={mode} "
          f"residual_scale={residual_scale} (from checkpoint config)")

    pushes = battery_pushes(magnitudes=tuple(args.magnitudes),
                            directions=args.directions, heights=(args.height,),
                            seed=0)
    in_band = sum(1 for p in pushes
                  if float(getattr(p, "impulse", 0.0)) <= TRAIN_MAX_IMPULSE)
    print(f"[t3-read] battery: {len(pushes)} pushes "
          f"({in_band} in-band <= {TRAIN_MAX_IMPULSE} N*s, "
          f"{len(pushes) - in_band} held-out) at h={args.height} m")

    rep = evaluate(lambda env, seed: PolicyController(net, name=f"{src.stem}_{steps}",
                                                      stochastic=False),
                   task="stance", seed0=0, push_plan=pushes,
                   gate=GATES["stance"], out_dir=METRICS_DIR, verbose=False,
                   max_episode_s=8.0, name=f"{src.stem}_{steps}")
    agg = rep["aggregate"]
    row = {"provenance": {
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     capture_output=True, text=True).stdout.strip(),
        "ckpt": str(src),
        "ckpt_sha256": hashlib.sha256(Path(tmp).read_bytes()).hexdigest()[:16],
        "ckpt_steps": steps,
        "action_mode": mode, "residual_scale": residual_scale,
        "magnitudes": list(args.magnitudes), "directions": args.directions,
        "height": args.height, "episodes": len(rep.get("episodes", [])),
        "gate": GATES["stance"].as_dict(),
        "command": (f"MUJOCO_GL=egl .venv/bin/python scripts/solo_t3_read.py "
                    f"--checkpoint {src}")},
        "verdict": rep["verdict"], "reasons": rep["reasons"], "aggregate": agg}
    out = METRICS_DIR / f"t3_read_{src.stem}_{steps}.json"
    write_json(out, row)
    print(f"[t3-read] wrote {out}")
    print(f"t3read steps={steps} verdict={rep['verdict']} "
          f"fall={agg['fall_rate']:.3f}/{agg['fall_rate_heldout']:.3f}(held) "
          f"upright={agg['mean_upright']} "
          f"stance_err={agg['stance_err_mean']} "
          f"recovery={agg['recovery_success_rate']} "
          f"maxJ_held={agg['max_recoverable_impulse_heldout']} "
          f"survivor={agg['survivor_valid_stance_rate']}")
    for c in rep.get("reasons", []) or []:
        print(f"  FAIL: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
