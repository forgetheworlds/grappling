#!/usr/bin/env python3
"""Deterministic evaluation of a PPO checkpoint (Phase-5 backbone).

Loads a checkpoint written by ``scripts/train_ppo.py``, runs N seeded matches
with the deterministic (mean) policy against the checkpoint's stage opponent,
and prints

* a per-exchange table (index, cause, winner, duration, back triggers, OOB), and
* movement / outcome summary counts (cause mix, W/L/D, standing fraction,
  ground fraction, knee/hand-contact fraction, dorsal-contact fraction,
  mean pelvis distance, OOB events).

Same ``--seed`` -> identical table (the script verifies this by re-running the
first episode and comparing the exchange records; prints ``EVAL OK``).

Usage::

    .venv/bin/python scripts/eval_ppo.py --checkpoint checkpoints/rl/smoke.pt --episodes 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rl.checkpoint import load_checkpoint  # noqa: E402
from rl.curriculum import Curriculum, PISTY_STAGES, stage_from_dict  # noqa: E402
from rl.ppo import PPOConfig  # noqa: E402
from rl.trainer import evaluate, make_policy  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--learner", default="a", choices=("a", "b"))
    p.add_argument("--match-clock", type=float, default=None,
                   help="override the checkpoint's match clock")
    p.add_argument("--exchange-timeout", type=float, default=None)
    p.add_argument("--no-determinism-check", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ckpt = load_checkpoint(args.checkpoint)
    cfg = PPOConfig.from_dict(ckpt.get("config") or {})
    state = ckpt.get("state") or {}
    extra = ckpt.get("extra") or {}
    stage_key = state.get("stage", {}).get("key", "C")
    index = [s.key for s in PISTY_STAGES].index(stage_key)
    curriculum = Curriculum(start_index=index)
    stage = (stage_from_dict(state["stage"]) if state.get("stage")
             else curriculum.stage)  # prefer the exact stage saved in the checkpoint
    env_kwargs = dict(extra.get("env") or {})
    if args.match_clock is not None:
        env_kwargs["match_clock"] = args.match_clock
    if args.exchange_timeout is not None:
        env_kwargs["exchange_timeout"] = args.exchange_timeout
    env_kwargs.setdefault("match_clock", 30.0)
    env_kwargs.setdefault("exchange_timeout", 6.0)

    policy = make_policy(cfg)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    summary = evaluate(policy, stage=stage, env_kwargs=env_kwargs, episodes=args.episodes,
                       seed=args.seed, learner_robot=args.learner,
                       frame_stack=cfg.frame_stack,
                       check_determinism=not args.no_determinism_check)

    print(f"[eval_ppo] checkpoint {args.checkpoint} @ step {state.get('step_count')} "
          f"| stage {stage.key} ({stage.name}) | opponent {stage.opponent.as_dict()} "
          f"| learner {args.learner} | seed {args.seed} | env {json.dumps(env_kwargs, sort_keys=True)}")
    print("\n[eval_ppo] per-exchange table:")
    hdr = f"{'ep_seed':>9} {'idx':>4} {'cause':>9} {'winner':>7} {'loser':>6} {'dur_s':>7} " \
          f"{'back_a':>7} {'back_b':>7} {'oob_a':>5} {'oob_b':>5} amb"
    print("  " + hdr)
    for r in summary["table"]:
        bt = r["back_triggers"] or {}
        oob = r["oob_events"] or {}
        print("  " + f"{r['episode_seed']:>9} {str(r['index']):>4} {str(r['cause']):>9} "
              f"{str(r['winner']):>7} {str(r['loser']):>6} {float(r['duration']):>7.2f} "
              f"{str(bt.get('a')):>7} {str(bt.get('b')):>7} "
              f"{int(oob.get('a', 0)):>5} {int(oob.get('b', 0)):>5} {bool(r['ambiguous'])}")

    mv = summary["movement"]
    print("\n[eval_ppo] outcome summary:")
    print(f"  episodes             : {summary['episodes']} (steps per episode {summary['episode_steps']})")
    print(f"  exchanges            : {summary['exchanges']} | causes {summary['causes']}")
    print(f"  learner W/L/D        : {summary['learner_wins']}/{summary['learner_losses']}/{summary['draws']}")
    print(f"  mean exchange time   : "
          f"{'--' if summary['mean_exchange_duration_s'] is None else format(summary['mean_exchange_duration_s'], '.2f') + ' s'}")
    print(f"  OOB events           : {summary['oob_events']}")
    print("[eval_ppo] movement summary:")
    print(f"  mean learner pelvis z: {mv['mean_pelvis_z_m']:.3f} m")
    print(f"  standing fraction    : {mv['standing_fraction']:.3f} (pelvis z >= 0.5 m)")
    print(f"  ground fraction      : {mv['ground_fraction']:.3f} (pelvis z < 0.35 m)")
    print(f"  knee/hand contact    : {mv['knee_hand_contact_fraction']:.3f}")
    print(f"  dorsal contact       : {mv['dorsal_contact_fraction']:.3f}")
    print(f"  mean pelvis distance : {mv['mean_pelvis_distance_m']:.3f} m")

    if not args.no_determinism_check:
        assert summary["deterministic"] is True, "evaluation was not reproducible for a fixed seed"
        print("[eval_ppo] EVAL OK (deterministic for fixed seed)")
    else:
        print("[eval_ppo] EVAL OK (determinism check skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
