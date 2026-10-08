"""Minimal single-env PPO trainer for the solo drill (S1 harness extension).

This is the *entry point* the later stages (S2 T1/T2 training) start, stop and
resume with; it is deliberately thin and reuses ``rl`` wholesale:

* networks/action map  -- ``rl.net.ActorCritic`` + ``env.ctrl_from_unit``
* PPO/GAE/lr schedule  -- ``rl.ppo`` (``RolloutBatch``, ``compute_gae``,
  ``ppo_update``, ``lr_at``)
* checkpoints          -- ``rl.checkpoint`` (atomic save, strict load,
  optimizer + RNG state, resume-exact)

Commands (from the repo root, venv active)::

    # start T1 balance training (long run: hold the sim lock)
    MUJOCO_GL=egl .venv/bin/python -m solo.train --task balance \\
        --steps 2000000 --out checkpoints/solo/t1_balance.pt

    # stop: Ctrl-C (SIGINT) -- the trainer saves and exits 0
    # resume:
    ... --resume checkpoints/solo/t1_balance.pt --steps 4000000

Episode semantics: ``terminated`` (fall/dorsal) ends the episode; ``truncated``
(time limit) also ends it.  GAE masks the bootstrap at every episode end
(conservative convention: the continuation value at a time-limit truncation is
dropped; noted for S2).  Rewards are the env's per-step task reward (terms
logged separately in ``info["reward_terms"]``).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .env import SoloEnv
from .metrics import METRICS_DIR, write_json
from .obs import ACTOR_DIM, CRITIC_DIM
from .scene import N_JOINTS, load_solo_model

REPO = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO / "checkpoints" / "solo"


@dataclass
class TrainConfig:
    """Training knobs (PPOConfig is built from these; defaults are CPU-sized)."""

    task: str = "balance"
    steps: int = 30_000
    rollout_steps: int = 2048
    seed: int = 0
    gamma: float = 0.995
    lam: float = 0.95
    lr: float = 3e-4
    epochs: int = 4
    minibatches: int = 4
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    hidden: tuple[int, ...] = (256, 256)
    action_mode: str = "absolute"
    out: str = ""
    save_every: int = 50_000
    log_every: int = 2048
    torch_threads: int = 1

    def ppo_config(self):
        from rl.ppo import PPOConfig

        return PPOConfig(rollout_steps=self.rollout_steps, n_envs=1, gamma=self.gamma,
                         lam=self.lam, clip=self.clip, value_coef=self.value_coef,
                         entropy_coef=self.entropy_coef, epochs=self.epochs,
                         minibatches=self.minibatches, hidden=tuple(self.hidden),
                         action_mode=self.action_mode, lr=self.lr, seed=self.seed,
                         torch_threads=self.torch_threads)


class SoloTrainer:
    """Single-env PPO loop over :class:`solo.env.SoloEnv`."""

    def __init__(self, cfg: TrainConfig, model=None):
        from rl.net import ActorCritic, NetConfig

        self.cfg = cfg
        torch.set_num_threads(int(cfg.torch_threads))
        self.model = model if model is not None else load_solo_model()
        self.env = SoloEnv(self.model, task=cfg.task, seed=cfg.seed)
        self.net = ActorCritic(ACTOR_DIM, CRITIC_DIM, act_dim=N_JOINTS,
                               cfg=cfg.ppo_config().net_config())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.steps_done = 0
        self.iteration = 0
        self.episode_seed = int(cfg.seed)
        self.recent_returns: list[float] = []
        self._ep_return = 0.0
        self._stop = False
        self.last_stats: dict = {}
        obs = self.env.reset(seed=self.episode_seed)
        self._obs = obs

    # ------------------------------------------------------------------ data
    def collect(self) -> "object":
        from rl.ppo import RolloutBatch

        T = int(self.cfg.rollout_steps)
        cfg = self.cfg.ppo_config()
        actor = np.zeros((T, 1, ACTOR_DIM), np.float32)
        critic = np.zeros((T, 1, CRITIC_DIM), np.float32)
        actions = np.zeros((T, 1, N_JOINTS), np.float32)
        logp = np.zeros((T, 1), np.float32)
        values = np.zeros((T, 1), np.float32)
        rewards = np.zeros((T, 1), np.float32)
        dones = np.zeros((T, 1), np.float32)
        for t in range(T):
            obs = self._obs
            with torch.no_grad():
                a_obs = torch.from_numpy(obs["actor"]).unsqueeze(0)
                c_obs = torch.from_numpy(obs["critic"]).unsqueeze(0)
                sample = self.net.actor.sample(a_obs)
                unit = sample["unit"].numpy()[0]
                lp = float(sample["logp"][0])
                value = float(self.net.value(c_obs)[0])
            ctrl = self.env.ctrl_from_unit(unit)
            next_obs, reward, terminated, truncated, info = self.env.step(ctrl)
            actor[t, 0] = obs["actor"]
            critic[t, 0] = obs["critic"]
            actions[t, 0] = unit
            logp[t, 0] = lp
            values[t, 0] = value
            rewards[t, 0] = reward
            dones[t, 0] = 1.0 if (terminated or truncated) else 0.0
            self._ep_return += float(reward)
            self.steps_done += 1
            if terminated or truncated:
                self.recent_returns.append(self._ep_return)
                self.recent_returns = self.recent_returns[-50:]
                self._ep_return = 0.0
                self.episode_seed += 1
                self._obs = self.env.reset(seed=self.episode_seed)
            else:
                self._obs = next_obs
        with torch.no_grad():
            last_value = float(self.net.value(
                torch.from_numpy(self._obs["critic"]).unsqueeze(0))[0])
        batch = RolloutBatch(actor_obs=actor, critic_obs=critic, actions_unit=actions,
                            logp=logp, values=values, rewards=rewards, dones=dones,
                            meta={"task": self.cfg.task, "steps": self.steps_done})
        batch.finalize(np.array([last_value], dtype=np.float64), cfg)
        return batch

    # ------------------------------------------------------------------ train
    def run(self) -> dict:
        from rl.ppo import lr_at, ppo_update

        cfg = self.cfg.ppo_config()
        start_steps = self.steps_done
        t0 = time.perf_counter()
        while self.steps_done < self.cfg.steps and not self._stop:
            batch = self.collect()
            progress = self.steps_done / max(1, self.cfg.steps)
            stats = ppo_update(self.net, self.optimizer, batch, cfg,
                               lr_at(progress, cfg), self.generator)
            self.iteration += 1
            wall = time.perf_counter() - t0
            done = self.steps_done - start_steps
            self.last_stats = {
                **stats,
                "iteration": self.iteration,
                "steps": self.steps_done,
                "mean_return_50": (round(float(np.mean(self.recent_returns)), 4)
                                   if self.recent_returns else None),
                "wall_s": round(wall, 2),
                "steps_per_s": round(done / wall, 2) if wall > 0 else None,
            }
            if self.cfg.log_every and self.iteration % max(1, self.cfg.log_every //
                                                           self.cfg.rollout_steps) == 0:
                print(json.dumps(self.last_stats))
            if self.cfg.save_every and self.steps_done % self.cfg.save_every < \
                    self.cfg.rollout_steps:
                self.save()
        if self.cfg.out:
            self.save()
        return self.last_stats

    # -------------------------------------------------------------- checkpoint
    def save(self, path: str | None = None) -> str:
        from rl.checkpoint import rng_state, save_checkpoint

        p = Path(path or self.cfg.out or (CHECKPOINT_DIR / f"{self.cfg.task}.pt"))
        return save_checkpoint(
            p, policy=self.net, optimizer=self.optimizer,
            cfg={"train": self.cfg.__dict__, "env": self.env.config()},
            state={"steps_done": self.steps_done, "iteration": self.iteration,
                   "episode_seed": self.episode_seed,
                   "recent_returns": self.recent_returns,
                   "_ep_return": self._ep_return},
            extra={"rng": rng_state(),
                   "torch_generator": self.generator.get_state()})

    def load(self, path: str) -> None:
        from rl.checkpoint import apply_checkpoint, load_checkpoint, restore_rng

        ckpt = load_checkpoint(path)
        state = apply_checkpoint(ckpt, policy=self.net, optimizer=self.optimizer)
        self.steps_done = int(state.get("steps_done", 0))
        self.iteration = int(state.get("iteration", 0))
        self.episode_seed = int(state.get("episode_seed", self.cfg.seed))
        self.recent_returns = list(state.get("recent_returns", []))
        self._ep_return = float(state.get("_ep_return", 0.0))
        extra = ckpt.get("extra") or {}
        if extra.get("rng"):
            restore_rng(extra["rng"])
        if extra.get("torch_generator") is not None:
            self.generator.set_state(extra["torch_generator"])
        self._obs = self.env.reset(seed=self.episode_seed)


def train(cfg: TrainConfig, *, model=None, on_sigint_save: bool = True) -> dict:
    """Run training from scratch; returns the final stats.

    SIGINT stops the loop gracefully and saves (when ``cfg.out`` is set), so a
    long run is interruptible without losing the checkpoint.
    """
    trainer = SoloTrainer(cfg, model=model)

    def _handler(signum, frame):  # pragma: no cover - signal path
        trainer._stop = True

    signal.signal(signal.SIGINT, _handler)
    stats = trainer.run()
    if on_sigint_save and cfg.out:
        trainer.save()
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="balance",
                    help="balance|locomotion|stance|reach|shot|recovery")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--rollout-steps", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="")
    ap.add_argument("--save-every", type=int, default=50_000)
    ap.add_argument("--resume", default="", help="checkpoint to resume from")
    ap.add_argument("--no-sigint-save", action="store_true")
    args = ap.parse_args(argv)
    cfg = TrainConfig(task=args.task, steps=args.steps,
                      rollout_steps=args.rollout_steps, seed=args.seed,
                      gamma=args.gamma, lr=args.lr, out=args.out,
                      save_every=args.save_every)
    if args.resume:
        from rl.checkpoint import apply_checkpoint, load_checkpoint

        trainer = SoloTrainer(cfg)
        trainer.load(args.resume)
        stats = trainer.run()
        if cfg.out:
            trainer.save()
    else:
        stats = train(cfg, on_sigint_save=not args.no_sigint_save)
    print("train done:", json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
