#!/usr/bin/env python3
"""PPO training entry point for the G1 wrestling env (Phase-5 backbone).

Defaults are the acceptance smoke run: PISTY stage C (dynamic drilling, fixed
DOUBLE_LEG command, STANCE-reference opponent), 3 env workers, 30k env steps,
reduced match/exchange clocks, checkpoint under ``checkpoints/rl/``.

Prints measured throughput (env steps/s), wall time, episode/exchange stats and
the action-bounds check; writes a resumable checkpoint (``--resume`` continues
from one, including curriculum stage, stats and RNG streams).

Examples::

    # smoke run (scripted opponent, ~1 min)
    .venv/bin/python scripts/train_ppo.py

    # same, against a static StandHold opponent
    .venv/bin/python scripts/train_ppo.py --opponent stand_hold --out checkpoints/rl/smoke_hold.pt

    # interrupt and resume
    .venv/bin/python scripts/train_ppo.py --steps 40000 --ckpt-every 4000
    .venv/bin/python scripts/train_ppo.py --resume checkpoints/rl/smoke.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rl.checkpoint import load_checkpoint  # noqa: E402
from rl.curriculum import PISTY_STAGES, Curriculum  # noqa: E402
from rl.obs import COMMAND_DIM, ENV_OBS_DIM, actor_obs_dim  # noqa: E402
from rl.ppo import PPOConfig  # noqa: E402
from rl.privileged import PRIV_DIM  # noqa: E402
from rl.scripted import OpponentSpec  # noqa: E402
from rl.trainer import Trainer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=30_000, help="target env steps (learner decisions x envs)")
    p.add_argument("--stage", default="C", help="PISTY stage key A-E (default C = dynamic drilling)")
    p.add_argument("--envs", type=int, default=3, help="parallel env workers")
    p.add_argument("--backend", default="auto", choices=("auto", "sequential", "subproc"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-stack", type=int, default=1, choices=(1, 4))
    p.add_argument("--action-mode", default="absolute", choices=("absolute", "residual"))
    p.add_argument("--rollout", type=int, default=2048, help="rollout steps per env per iteration")
    p.add_argument("--match-clock", type=float, default=30.0, help="sim seconds per match (episode)")
    p.add_argument("--exchange-timeout", type=float, default=6.0)
    p.add_argument("--out", default="checkpoints/rl/smoke.pt", help="checkpoint path")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=None,
                   help="run a deterministic eval every N steps (stored in the checkpoint)")
    p.add_argument("--eval-episodes", type=int, default=1)
    p.add_argument("--opponent", default=None,
                   help="override the stage opponent: stand_hold | stance | sprawl | teacher")
    p.add_argument("--no-scorer", action="store_true", help="disable the optional technique scorer")
    p.add_argument("--quiet", action="store_true")
    return p


def resolve_opponent(name: str | None, stage_key: str) -> OpponentSpec | None:
    if name is None:
        return None
    if name == "stand_hold":
        return OpponentSpec("stand_hold")
    if name == "stance":
        return OpponentSpec("reference_replay", technique="STANCE", loop=True)
    if name == "sprawl":
        return OpponentSpec("reference_replay", technique="SPRAWL", loop=True)
    if name == "teacher":
        return OpponentSpec("teacher", technique="STANCE")
    raise SystemExit(f"unknown --opponent {name!r}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = PPOConfig(rollout_steps=args.rollout, n_envs=args.envs, seed=args.seed,
                    frame_stack=args.frame_stack, action_mode=args.action_mode)
    index = [s.key for s in PISTY_STAGES].index(args.stage.upper())
    curriculum = Curriculum(start_index=index)
    override = resolve_opponent(args.opponent, args.stage)
    if override is not None:
        stages = list(curriculum.stages)
        stages[index] = dataclasses.replace(stages[index], opponent=override)
        curriculum = Curriculum(stages=tuple(stages), start_index=index)
    env_kwargs = {"match_clock": args.match_clock, "exchange_timeout": args.exchange_timeout}
    trainer = Trainer(cfg, out_path=args.out, curriculum=curriculum, backend=args.backend,
                      n_envs=args.envs, env_kwargs=env_kwargs, ckpt_every=args.ckpt_every,
                      scorer_enabled=not args.no_scorer, verbose=0 if args.quiet else 1,
                      eval_every=args.eval_every, eval_episodes=args.eval_episodes)

    if args.resume:
        trainer.load(args.resume)

    def _handler(signum, frame):
        print(f"\n[train_ppo] signal {signum}: stopping after the current rollout, saving checkpoint...",
              flush=True)
        trainer.request_stop()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    stage = curriculum.stage
    print("[train_ppo] config: " + json.dumps({
        "stage": f"{stage.key} ({stage.name})",
        "opponent": stage.opponent.as_dict(),
        "command_on": stage.command_on,
        "techniques": list(stage.techniques),
        "weights": stage.weights.as_dict(),
        "perturbations": stage.perturbations.as_dict(),
        "actor_obs_dim": actor_obs_dim(cfg.frame_stack),
        "critic_obs_dim": actor_obs_dim(cfg.frame_stack) + PRIV_DIM,
        "env_obs_dim": ENV_OBS_DIM, "command_dim": COMMAND_DIM,
        "params": trainer.policy.n_params(),
        "n_envs": args.envs, "backend": trainer.backend, "frame_stack": cfg.frame_stack,
        "rollout_steps_per_env": cfg.rollout_steps, "minibatches": cfg.minibatches,
        "match_clock": args.match_clock, "exchange_timeout": args.exchange_timeout,
        "scorer": trainer.stage_reward.status()["scorer_note"],
    }, sort_keys=True), flush=True)

    t0 = time.perf_counter()
    try:
        summary = trainer.run(args.steps)
    except KeyboardInterrupt:  # fallback: still save
        trainer.request_stop()
        summary = trainer.summary(wall_s=time.perf_counter() - t0)
        trainer.save()
    wall = time.perf_counter() - t0

    # ---- acceptance self-check
    ckpt = load_checkpoint(summary["checkpoint"])
    assert summary["bounds_bad"] == 0, f"policy actions left ctrlrange: {summary['bounds_bad']}"
    assert ckpt["state"]["step_count"] == summary["steps"], "checkpoint/step mismatch"
    print("\n[train_ppo] ==== run report ====")
    sps = summary["steps_per_s"]
    print(f"  wall time            : {wall:.1f} s")
    print(f"  env steps            : {summary['steps']} "
          f"({'--' if sps is None else format(sps, '.1f') + ' steps/s incl. updates'})")
    print(f"  iterations (updates) : {summary['iterations']}")
    print(f"  episodes (matches)   : {summary['matches']}")
    print(f"  exchanges            : {summary['exchanges']} "
          f"(learner W/L/D {summary['learner_wins']}/{summary['learner_losses']}/{summary['draws']}, "
          f"causes {summary['causes']})")
    mean_dur = summary["mean_exchange_duration_s"]
    mean_ret = summary["mean_match_return"]
    print(f"  mean exchange time   : {'--' if mean_dur is None else format(mean_dur, '.2f') + ' s'}")
    print(f"  mean match return    : {'--' if mean_ret is None else format(mean_ret, '.3f')}")
    print(f"  OOB events           : {summary['oob_events']}")
    print(f"  action bounds        : {summary['bounds_bad']} violations "
          f"(policy stream inside ctrlrange)")
    print(f"  scorer               : {summary['scorer']['scorer_note']}")
    if trainer.eval_summary:
        ev = trainer.eval_summary
        det = ("" if ev.get("deterministic") is None
               else f" (deterministic={ev['deterministic']})")
        print(f"  last eval            : exchanges {ev['exchanges']} causes {ev['causes']} "
              f"W/L/D {ev['learner_wins']}/{ev['learner_losses']}/{ev['draws']} "
              f"standing {ev['movement']['standing_fraction']:.2f}{det}")
    print(f"  checkpoint           : {summary['checkpoint']}")
    print(f"  interrupted          : {summary['interrupted']}")
    print("[train_ppo] SMOKE OK (harness works; training quality not claimed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
