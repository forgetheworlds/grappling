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

    # N parallel env workers (subprocess vec backend; --steps counts total env
    # steps across workers; seeds/indices are recorded in the checkpoint):
    MUJOCO_GL=egl .venv/bin/python -m solo.train --task balance --n-envs 4 \\
        --steps 2000000 --out checkpoints/solo/t1_balance_vec4.pt

On ``--resume`` the checkpoint's own ``config.train`` is authoritative for
``action_mode`` and ``residual_scale`` (an explicit CLI flag still overrides),
so resuming a residual run cannot silently fall back to the absolute mapping.

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
    residual_scale: float = 0.5
    out: str = ""
    save_every: int = 50_000
    log_every: int = 2048
    torch_threads: int = 1
    #: parallel env workers: 1 = the in-process single-env path (unchanged),
    #: >1 = ``rl.vec_solo`` subprocess backend; ``steps`` then counts the total
    #: env transitions across workers.
    n_envs: int = 1
    vec_backend: str = "subproc"
    # push curriculum (T1 v2 fix): training-time pushes ramped with progress,
    # always <= TRAIN_MAX_IMPULSE so the gate's held-out magnitudes stay unseen
    push_curriculum: bool = True
    push_start_steps: int = 20_000
    push_warmup_steps: int = 400_000
    push_seed: int = 0
    # ------------------------------------------- literature recipe (solo.lit)
    # Every field below defaults to today's behaviour: a run without these flags
    # is byte-reproducible with v5's configuration.  See
    # ``reports/2026-10-08/lit_balance.md`` for the sources and the staged plan.
    reward_set: str = "default"        # "default" | "lit" (TaskReward term_set)
    freeze_joints: bool = False        # freeze non-balance joints at the keyframe
    lit_weights: tuple[tuple[str, float], ...] = ()   # RewardWeights overrides
    lit_push: str = "off"              # "off" | "bernoulli" | "interval"
    lit_push_prob: float = 0.01        # source A: 1%/frame
    lit_push_interval: float = 5.0     # source B: pushes every ~5 s
    lit_push_interval_jitter: float = 0.0
    lit_push_min_frac: float = 0.5     # source B: [0.5x, 2x] the ceiling
    lit_push_max_frac: float = 2.0
    lit_push_cap: float | None = None  # N*s absolute clamp (e.g. 12 = T1 band)
    lit_push_directions: int = 8
    lit_push_height: float = 0.95
    lit_push_seed: int = 0
    mirror_loss_coef: float = 0.0      # PPO mirror (symmetry) loss, source A
    #: observation history stacked into the actor/critic input (1 = today's
    #: memoryless contract).  The source SaW controller is an LSTM and the
    #: hand-built balancer provably uses hidden integrator state, so partial
    #: observability is a competing explanation for any v6 result.
    frame_stack: int = 1
    #: weight of the positive survival term (`alive`).  Measured v1/v2 pathology:
    #: the policy dodges the one-off -100 termination by collapsing into a
    #: non-terminating limb-supported pose (mean_upright 0.006 / 0.041 while
    #: fall_rate stays low).  Scaling the per-step survival signal up makes the
    #: standing-vs-collapsed gap dominate that one-off penalty.
    alive_weight: float = 1.0

    def ppo_config(self):
        from rl.ppo import PPOConfig

        return PPOConfig(rollout_steps=self.rollout_steps, n_envs=int(self.n_envs),
                         frame_stack=int(self.frame_stack), gamma=self.gamma,
                         lam=self.lam, clip=self.clip, value_coef=self.value_coef,
                         entropy_coef=self.entropy_coef, epochs=self.epochs,
                         minibatches=self.minibatches, hidden=tuple(self.hidden),
                         action_mode=self.action_mode, lr=self.lr, seed=self.seed,
                         torch_threads=self.torch_threads)


class SoloTrainer:
    """PPO loop over :class:`solo.env.SoloEnv` (single env, or ``n_envs`` vec workers)."""

    def __init__(self, cfg: TrainConfig, model=None):
        from rl.net import ActorCritic, NetConfig

        self.cfg = cfg
        if int(cfg.n_envs) < 1:
            raise ValueError(f"n_envs must be >= 1 (got {cfg.n_envs})")
        torch.set_num_threads(int(cfg.torch_threads))
        self.model = model if model is not None else load_solo_model()
        from .reward import RewardWeights

        weights = (RewardWeights(alive=float(cfg.alive_weight))
                   if abs(float(cfg.alive_weight) - 1.0) > 1e-9 else None)
        weights = self._lit_weights(weights)
        term_set, mask = self._lit_env_options(cfg)
        self.lit_push = self._lit_push_curriculum(cfg)
        self.frame_stack = max(1, int(cfg.frame_stack))
        if self.frame_stack > 1 and int(cfg.n_envs) > 1:
            raise SystemExit(
                "[solo.train] --frame-stack > 1 is single-env only: the vec port "
                "would have to stack per worker (rl.vec_solo.collect_vec feeds raw "
                "actor obs); refusing to train a wider net on unstacked inputs")
        self.vec = None
        self.env = None
        if int(cfg.n_envs) > 1:                     # subprocess vec backend
            from rl.vec_solo import VecSoloEnv

            self._require_vec_support(VecSoloEnv, term_set, mask)
            self.push_curriculum = self.lit_push or self._vec_curriculum(cfg)
            self.vec = VecSoloEnv(int(cfg.n_envs), backend=cfg.vec_backend,
                                  seed=cfg.seed, task=cfg.task, weights=weights,
                                  curriculum=self.push_curriculum, model=self.model,
                                  action_mode=cfg.action_mode,
                                  residual_scale=cfg.residual_scale,
                                  term_set=term_set, joint_mask=mask)
        else:                                       # unchanged single-env path
            self.env = SoloEnv(self.model, task=cfg.task, seed=cfg.seed, weights=weights,
                               action_mode=cfg.action_mode,
                               residual_scale=cfg.residual_scale,
                               term_set=term_set, joint_mask=mask)
        self.net = ActorCritic(ACTOR_DIM * self.frame_stack,
                               CRITIC_DIM + ACTOR_DIM * (self.frame_stack - 1),
                               act_dim=N_JOINTS,
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
        if self.vec is not None:
            self._vec_obs = self.vec.obs
            self._vec_ep_returns = [0.0] * int(cfg.n_envs)
            self.startup_ctrl_diff = self._vec_startup_check()
        else:
            self.push_curriculum = None
            if (self.lit_push is None and cfg.push_curriculum
                    and self.env.push_curriculum is not None):
                from dataclasses import replace

                self.push_curriculum = replace(self.env.push_curriculum,
                                               start_steps=int(cfg.push_start_steps),
                                               warmup_steps=int(cfg.push_warmup_steps),
                                               seed=int(cfg.push_seed))
            self._apply_push_curriculum()
            obs = self.env.reset(seed=self.episode_seed)
            self._reset_hist(obs)                       # frame stack, if any
            self.startup_ctrl_diff = self._startup_check(obs)
            self._obs = obs

    # ------------------------------------------------- frame-stack helpers
    def _reset_hist(self, obs: dict) -> None:
        """Seed the frame stack with ``frame_stack`` copies of the reset obs."""
        from collections import deque

        self._hist = deque([np.asarray(obs["actor"], np.float32)]
                           * self.frame_stack, maxlen=self.frame_stack)

    def _stacked(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """(actor, critic) network inputs for a *new* observation.

        Called exactly once per environment observation (the append is the
        history update); ``frame_stack == 1`` returns today's vectors unchanged,
        so the default path is bit-identical to before.
        """
        if self.frame_stack == 1:
            return obs["actor"], obs["critic"]
        self._hist.append(np.asarray(obs["actor"], np.float32))
        actor = np.concatenate(list(self._hist))         # oldest frame first
        return actor, np.concatenate([actor, obs["privileged"]])

    # ------------------------------------------------- literature helpers
    def _lit_weights(self, weights):
        """Apply ``--lit-weight NAME=VALUE`` overrides (validated, frozen dataclass)."""
        if not self.cfg.lit_weights:
            return weights
        from dataclasses import fields as _fields
        from dataclasses import replace

        from .reward import RewardWeights

        base = weights if weights is not None else RewardWeights()
        known = {f.name for f in _fields(RewardWeights)}
        over = {}
        for name, value in self.cfg.lit_weights:
            if name not in known:
                raise SystemExit(
                    f"[solo.train] --lit-weight {name}: no such RewardWeights field "
                    f"(known: {', '.join(sorted(known))})")
            over[name] = float(value)
        out = replace(base, **over)
        print(f"[solo.train] reward-weight overrides: {over}")
        return out

    def _lit_env_options(self, cfg: TrainConfig):
        """(term_set, joint_mask) for the env(s); both None unless requested."""
        term_set = None
        if cfg.reward_set != "default":
            from .reward import LIT_TERM_SETS

            key = {"lit": "balance_lit"}.get(cfg.reward_set, cfg.reward_set)
            if key not in LIT_TERM_SETS:
                raise SystemExit(f"[solo.train] unknown --reward-set {cfg.reward_set!r}")
            term_set = key
        mask = None
        if cfg.freeze_joints:
            from .lit import joint_mask

            mask = joint_mask(self.model)
        return term_set, mask

    def _lit_push_curriculum(self, cfg: TrainConfig):
        """The literature push distribution as a ``schedule_for`` object, or None."""
        if cfg.lit_push == "off":
            return None
        from .env import TASKS

        from .lit import LitPushConfig, LitPushCurriculum, measure_ceiling

        ceiling = measure_ceiling(self.model)
        cur = LitPushCurriculum(
            cfg=LitPushConfig(mode=cfg.lit_push, interval=float(cfg.lit_push_interval),
                              interval_jitter=float(cfg.lit_push_interval_jitter),
                              prob=float(cfg.lit_push_prob),
                              min_frac=float(cfg.lit_push_min_frac),
                              max_frac=float(cfg.lit_push_max_frac),
                              directions=int(cfg.lit_push_directions),
                              height=float(cfg.lit_push_height),
                              cap=(None if cfg.lit_push_cap is None
                                   else float(cfg.lit_push_cap)),
                              seed=int(cfg.lit_push_seed)),
            ceiling=ceiling, horizon=float(TASKS[cfg.task].horizon))
        c = ceiling.as_dict()
        band = (f"{cur.cfg.min_frac:g}x..{cur.cfg.max_frac:g}x J(direction), "
                f"cap {cur.cfg.cap if cur.cfg.cap is not None else 'none'} N*s, "
                f"J(direction) {c['j_reject_min_ns']:.2f}"
                f"..{c['j_reject_max_ns']:.2f} N*s")
        print(f"[solo.train] lit push mode={cfg.lit_push}: {band}")
        if cfg.push_curriculum:
            print("[solo.train] --lit-push supersedes the ramped v5 curriculum "
                  "(the lit distribution is stationary by design)")
        return cur

    def _require_vec_support(self, vec_cls, term_set, mask) -> None:
        """Refuse (loudly) to drop a lit option the vec backend cannot carry."""
        if term_set is None and mask is None:
            return
        import inspect

        params = inspect.signature(vec_cls.__init__).parameters
        missing = [k for k in ("term_set", "joint_mask") if k not in params]
        if missing:
            raise SystemExit(
                f"[solo.train] --n-envs > 1 with --reward-set {term_set!r} / "
                f"--freeze-joints needs VecSoloEnv to forward {missing} into each "
                f"worker's SoloEnv; refusing to train a different reward/mask on "
                f"the vec path than on the single-env path")

    def _vec_curriculum(self, cfg: TrainConfig):
        """Push curriculum for the vec workers (same overrides as the env-side one).

        Uses the task preset's curriculum (the same object ``SoloEnv`` would
        attach) so the workers' schedules are built exactly like the single-env
        trainer builds its env's.
        """
        if not cfg.push_curriculum:
            return None
        from dataclasses import replace

        from .env import TASKS

        base = TASKS[cfg.task].push_curriculum
        if base is None:
            return None
        return replace(base, start_steps=int(cfg.push_start_steps),
                       warmup_steps=int(cfg.push_warmup_steps),
                       seed=int(cfg.push_seed))

    def _vec_startup_check(self) -> float:
        """Single-env startup mapping check, run on vec worker 0."""
        with torch.no_grad():
            unit = self.net.actor.deterministic_unit(
                torch.from_numpy(self._vec_obs["actor"][0:1])).numpy()[0]
        diff = self.vec.probe(unit)
        self._vec_obs = self.vec.obs
        print(f"[solo.train] startup check (vec n_envs={self.vec.n_envs}): "
              f"mode={self.cfg.action_mode} max|ctrl - base_action| = {diff:.6f} rad")
        return diff

    def close(self) -> None:
        """Release the vec workers (no-op for the single-env path)."""
        if self.vec is not None:
            self.vec.close()


    def _startup_check(self, obs: dict) -> float:
        """Prove the untrained policy starts as the stand controller.

        With initial weights the policy's deterministic unit action is ~0; in
        residual mode ``ctrl_from_policy(0) = 0`` and ``step`` writes exactly
        ``base`` -- so the ctrl after the FIRST environment step must be within
        a small tolerance of ``env._base_action`` (the ``a_stand`` keyframe
        targets).  Returns the measured max |difference| (rad) and prints it.
        """
        with torch.no_grad():
            a_np, _ = self._stacked(obs)
            unit = self.net.actor.deterministic_unit(
                torch.from_numpy(a_np).unsqueeze(0)).numpy()[0]
        self.env.step(self.env.ctrl_from_policy(unit))
        diff = float(np.abs(np.asarray(self.env.data.ctrl, np.float64)
                            - self.env._base_action).max())
        print(f"[solo.train] startup check: mode={self.cfg.action_mode} "
              f"max|ctrl - base_action| = {diff:.6f} rad")
        obs = self.env.reset(seed=self.episode_seed)
        self._reset_hist(obs)
        return diff

    def _apply_push_curriculum(self) -> None:
        """Install the training push schedule for the next episode (before reset).

        The literature mode (``solo.lit.LitPushCurriculum``) is stationary and
        direction-relative to the measured non-stepping ceiling; the v5 mode is
        the progress-ramped curriculum capped at ``TRAIN_MAX_IMPULSE``.
        """
        if self.lit_push is not None:
            if self.env is None:                    # vec: workers install it
                return
            self.env.set_push_schedule(self.lit_push.schedule_for(
                self.steps_done, self.episode_seed))
            return
        if self.push_curriculum is None:
            return
        self.env.set_push_schedule(self.push_curriculum.schedule_for(
            self.steps_done, self.episode_seed))

    # ------------------------------------------------------------------ data
    def collect(self) -> "object":
        from rl.ppo import RolloutBatch

        T = int(self.cfg.rollout_steps)
        cfg = self.cfg.ppo_config()
        actor = np.zeros((T, 1, ACTOR_DIM * self.frame_stack), np.float32)
        critic = np.zeros((T, 1, CRITIC_DIM + ACTOR_DIM * (self.frame_stack - 1)),
                          np.float32)
        actions = np.zeros((T, 1, N_JOINTS), np.float32)
        logp = np.zeros((T, 1), np.float32)
        values = np.zeros((T, 1), np.float32)
        rewards = np.zeros((T, 1), np.float32)
        dones = np.zeros((T, 1), np.float32)
        for t in range(T):
            obs = self._obs
            a_np, c_np = self._stacked(obs)          # the frame stack, if any
            with torch.no_grad():
                a_obs = torch.from_numpy(a_np).unsqueeze(0)
                c_obs = torch.from_numpy(c_np).unsqueeze(0)
                sample = self.net.actor.sample(a_obs)
                unit = sample["unit"].numpy()[0]
                lp = float(sample["logp"][0])
                value = float(self.net.value(c_obs)[0])
            ctrl = self.env.ctrl_from_policy(unit)
            next_obs, reward, terminated, truncated, info = self.env.step(ctrl)
            actor[t, 0] = a_np
            critic[t, 0] = c_np
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
                self._apply_push_curriculum()
                self._obs = self.env.reset(seed=self.episode_seed)
                self._reset_hist(self._obs)          # no stale frames cross the reset
            else:
                self._obs = next_obs
        with torch.no_grad():
            last_value = float(self.net.value(
                torch.from_numpy(self._stacked(self._obs)[1]).unsqueeze(0))[0])
        batch = RolloutBatch(actor_obs=actor, critic_obs=critic, actions_unit=actions,
                            logp=logp, values=values, rewards=rewards, dones=dones,
                            meta={"task": self.cfg.task, "steps": self.steps_done})
        batch.finalize(np.array([last_value], dtype=np.float64), cfg)
        return batch

    # ------------------------------------------------------------- vec collect
    def collect_vec(self):
        """Rollout of ``rollout_steps`` control steps over ``n_envs`` workers.

        Same transition format as :meth:`collect`, ``(T, N, ...)`` instead of
        ``(T, 1, ...)``: per control step the policy samples all workers at once,
        the workers own ``ctrl_from_policy``/``resolve_action`` and auto-reset
        (parent-driven, next seed of each worker's stream) inside ``step``.  A
        ``done`` row stores the *reset* obs for the next step and ``dones[t] = 1``
        masks the bootstrap exactly like the single-env path.
        """
        from rl.ppo import RolloutBatch

        T = int(self.cfg.rollout_steps)
        N = int(self.cfg.n_envs)
        cfg = self.cfg.ppo_config()
        actor = np.zeros((T, N, ACTOR_DIM), np.float32)
        critic = np.zeros((T, N, CRITIC_DIM), np.float32)
        actions = np.zeros((T, N, N_JOINTS), np.float32)
        logp = np.zeros((T, N), np.float32)
        values = np.zeros((T, N), np.float32)
        rewards = np.zeros((T, N), np.float32)
        dones = np.zeros((T, N), np.float32)
        obs = self._vec_obs
        for t in range(T):
            with torch.no_grad():
                a_obs = torch.from_numpy(obs["actor"])
                c_obs = torch.from_numpy(obs["critic"])
                sample = self.net.actor.sample(a_obs)
                unit = sample["unit"].numpy()
                lp = sample["logp"].numpy()
                value = self.net.value(c_obs).numpy()
            batch = self.vec.step(unit)          # resets done envs inside
            actor[t] = obs["actor"]
            critic[t] = obs["critic"]
            actions[t] = unit
            logp[t] = lp
            values[t] = value
            rewards[t] = batch.rewards
            dones[t] = (batch.terminated | batch.truncated).astype(np.float32)
            self.steps_done += N
            for i in range(N):
                self._vec_ep_returns[i] += float(batch.rewards[i])
            for i in np.flatnonzero(batch.terminated | batch.truncated).tolist():
                self.recent_returns.append(self._vec_ep_returns[i])
                self._vec_ep_returns[i] = 0.0
            self.recent_returns = self.recent_returns[-50:]
            obs = batch.obs
        self._vec_obs = obs
        with torch.no_grad():
            last_value = self.net.value(torch.from_numpy(obs["critic"])).numpy()
        batch = RolloutBatch(actor_obs=actor, critic_obs=critic, actions_unit=actions,
                            logp=logp, values=values, rewards=rewards, dones=dones,
                            meta={"task": self.cfg.task, "steps": self.steps_done,
                                  "n_envs": N})
        batch.finalize(np.asarray(last_value, np.float64), cfg)
        return batch

    # ------------------------------------------------------------------ train
    def run(self) -> dict:
        from rl.ppo import lr_at, ppo_update

        cfg = self.cfg.ppo_config()
        start_steps = self.steps_done
        t0 = time.perf_counter()
        while self.steps_done < self.cfg.steps and not self._stop:
            batch = self.collect_vec() if self.vec is not None else self.collect()
            progress = self.steps_done / max(1, self.cfg.steps)
            stats = ppo_update(self.net, self.optimizer, batch, cfg,
                               lr_at(progress, cfg), self.generator)
            if float(self.cfg.mirror_loss_coef) > 0.0:
                from .mirror import mirror_step

                step = mirror_step(self.net, self.optimizer, batch.actor_obs,
                                   coef=float(self.cfg.mirror_loss_coef),
                                   grad_clip=float(cfg.grad_clip),
                                   model=self.model)
                stats = {**stats, "mirror_loss": round(step.loss, 6),
                         "mirror_grad_norm": round(step.grad_norm, 6)}
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
            per_iter = self.cfg.rollout_steps * max(1, int(self.cfg.n_envs))
            if self.cfg.save_every and self.steps_done % self.cfg.save_every < \
                    per_iter:
                self.save()
        if self.cfg.out:
            self.save()
        return self.last_stats

    # -------------------------------------------------------------- checkpoint
    def save(self, path: str | None = None) -> str:
        from rl.checkpoint import rng_state, save_checkpoint

        p = Path(path or self.cfg.out or (CHECKPOINT_DIR / f"{self.cfg.task}.pt"))
        if self.vec is not None:
            cfg_env = self.vec.config()      # worker-0 conditions + seed block
            state = {"steps_done": self.steps_done, "iteration": self.iteration,
                     "recent_returns": self.recent_returns,
                     "_vec_ep_returns": list(self._vec_ep_returns),
                     "vec": self.vec.state()}
        else:
            cfg_env = self.env.config()
            state = {"steps_done": self.steps_done, "iteration": self.iteration,
                     "episode_seed": self.episode_seed,
                     "recent_returns": self.recent_returns,
                     "_ep_return": self._ep_return}
        return save_checkpoint(
            p, policy=self.net, optimizer=self.optimizer,
            cfg={"train": self.cfg.__dict__, "env": cfg_env},
            state=state,
            extra={"rng": rng_state(),
                   "torch_generator": self.generator.get_state()})

    def load(self, path: str) -> None:
        from rl.checkpoint import apply_checkpoint, load_checkpoint, restore_rng

        ckpt = load_checkpoint(path)
        state = apply_checkpoint(ckpt, policy=self.net, optimizer=self.optimizer)
        self.steps_done = int(state.get("steps_done", 0))
        self.iteration = int(state.get("iteration", 0))
        self.recent_returns = list(state.get("recent_returns", []))
        extra = ckpt.get("extra") or {}
        if extra.get("rng"):
            restore_rng(extra["rng"])
        if extra.get("torch_generator") is not None:
            self.generator.set_state(extra["torch_generator"])
        if self.vec is not None:
            vec_state = state.get("vec") or {}
            if not vec_state:
                raise ValueError(
                    f"checkpoint {path} has no 'vec' state: refusing to resume an "
                    f"n_envs={self.vec.n_envs} run from a single-env checkpoint "
                    f"(start a fresh vec run instead)")
            self._vec_ep_returns = [float(x) for x in
                                    state.get("_vec_ep_returns",
                                              [0.0] * self.vec.n_envs)]
            self.vec.load_state(vec_state)                # re-reset at saved seeds
            self._vec_obs = self.vec.obs
            return
        self.episode_seed = int(state.get("episode_seed", self.cfg.seed))
        self._ep_return = float(state.get("_ep_return", 0.0))
        self._apply_push_curriculum()
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
    try:
        stats = trainer.run()
        if on_sigint_save and cfg.out:
            trainer.save()
    finally:
        trainer.close()          # release vec workers (no-op for single env)
    return stats


def resolve_action_mode(ckpt: dict, cli_mode: str | None = None
                        ) -> tuple[str, float]:
    """Resolve (action_mode, residual_scale) for evaluating a checkpoint.

    The mode is read from the checkpoint's own config (``config.train``); the CLI
    value only overrides it explicitly.  Raises when the checkpoint predates the
    field and no CLI value was given -- evaluating a policy through the wrong
    action mapping is not a measurement.
    """
    train_cfg = ((ckpt.get("config") or {}).get("train") or {})
    env_cfg = ((ckpt.get("config") or {}).get("env") or {})
    ckpt_mode = train_cfg.get("action_mode", env_cfg.get("action_mode"))
    scale = train_cfg.get("residual_scale", env_cfg.get("residual_scale", 0.5))
    if cli_mode in (None, "auto"):
        if ckpt_mode is None:
            raise ValueError(
                "checkpoint does not record action_mode and no --action-mode was "
                "given: refusing to evaluate through a guessed mapping")
        return str(ckpt_mode), float(scale)
    return str(cli_mode), float(scale)


def use_lock_for(lock: str, steps: int, rate: float = 200.0,
                 short_run_s: float = 60.0) -> bool:
    """Whether the trainer should hold the advisory sim lock.

    ``off`` never locks (multi-hour runs must not starve other sims),
    ``on`` always locks, ``auto`` locks only for short runs
    (estimated ``steps / rate`` below ``short_run_s``).
    """
    if lock == "on":
        return True
    if lock == "off":
        return False
    if lock != "auto":
        raise ValueError(f"unknown lock mode {lock!r}")
    return (float(steps) / max(1.0, float(rate))) < float(short_run_s)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _parse_lit_weights(pairs: list[str]) -> tuple[tuple[str, float], ...]:
    """``--lit-weight NAME=VALUE`` -> ((name, value), ...); exits on a typo."""
    out: list[tuple[str, float]] = []
    for item in pairs:
        name, sep, value = str(item).partition("=")
        if not sep or not name.strip():
            raise SystemExit(f"[solo.train] --lit-weight {item!r}: expected NAME=VALUE")
        try:
            out.append((name.strip(), float(value)))
        except ValueError:
            raise SystemExit(f"[solo.train] --lit-weight {item!r}: {value!r} is not a "
                             f"number") from None
    return tuple(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="balance",
                    help="balance|locomotion|stance|reach|shot|recovery")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--rollout-steps", type=int, default=2048)
    ap.add_argument("--n-envs", type=_positive_int, default=1,
                    help="parallel env workers: 1 = the in-process single-env "
                         "path (unchanged); >1 = rl.vec_solo subprocess backend "
                         "(total --steps counts env steps across workers)")
    ap.add_argument("--vec-backend", choices=("subproc", "sequential"),
                    default="subproc",
                    help="vec backend used when --n-envs > 1 (default subproc)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy-coef", type=float, default=0.01,
                    help="PPO entropy bonus (0.001 recommended when the balance "
                         "reward is nearly constant: the outward log-std gradient "
                         "has nothing opposing it otherwise)")
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--hidden", default="256,256",
                    help="comma-separated MLP widths, e.g. 256,256")
    ap.add_argument("--action-mode", choices=("absolute", "residual"), default=None,
                    help="absolute|residual; unset = absolute, but on --resume the "
                         "checkpoint's saved mode wins unless this is passed")
    ap.add_argument("--residual-scale", type=float, default=None,
                    help="residual action magnitude (default 0.5); on --resume the "
                         "checkpoint's saved scale wins unless this is passed")
    ap.add_argument("--log-every", type=int, default=2048)
    ap.add_argument("--push-seed", type=int, default=0,
                    help="seed of the ramped training push schedule")
    ap.add_argument("--reward-set", choices=("default", "lit"), default="default",
                    help="reward term set: 'default' = the task family's own terms "
                         "(unchanged); 'lit' = the literature balance set (sources "
                         "A+B, see reports/2026-10-08/lit_balance.md)")
    ap.add_argument("--freeze-joints", choices=("off", "balance"), default="off",
                    help="freeze the non-balance joints (arms, wrists, hip/waist "
                         "yaw) at their a_stand keyframe ctrl, so the residual acts "
                         "on the 12 balance joints only")
    ap.add_argument("--lit-weight", action="append", default=[], metavar="NAME=VALUE",
                    help="override one RewardWeights field for this run (repeatable), "
                         "e.g. --lit-weight grf_even=0")
    ap.add_argument("--lit-push", choices=("off", "bernoulli", "interval"),
                    default="off",
                    help="training push distribution: 'bernoulli' = source A's "
                         "per-frame probability; 'interval' = source B's repeated "
                         "pushes; magnitudes are a fraction of the measured "
                         "DIRECTIONAL non-stepping ceiling")
    ap.add_argument("--lit-push-prob", type=float, default=0.01,
                    help="bernoulli mode: push probability per control step")
    ap.add_argument("--lit-push-interval", type=float, default=5.0,
                    help="interval mode: seconds between pushes")
    ap.add_argument("--lit-push-interval-jitter", type=float, default=0.0,
                    help="interval mode: +/- seconds of uniform jitter")
    ap.add_argument("--lit-push-min-frac", type=float, default=0.5,
                    help="low end of the magnitude band, as a fraction of J(direction)")
    ap.add_argument("--lit-push-max-frac", type=float, default=2.0,
                    help="high end of the magnitude band, as a fraction of J(direction)")
    ap.add_argument("--lit-push-cap", type=float, default=None,
                    help="absolute impulse clamp (N*s); e.g. 12 keeps every training "
                         "push inside the T1 non-stepping band")
    ap.add_argument("--lit-push-directions", type=int, default=8,
                    help="evenly spaced world push yaws (the gate battery uses 8)")
    ap.add_argument("--lit-push-height", type=float, default=0.95,
                    help="push application height (m); 0.95 = the gate/training height")
    ap.add_argument("--lit-push-seed", type=int, default=0,
                    help="seed of the literature push distribution")
    ap.add_argument("--mirror-loss-coef", type=float, default=0.0,
                    help="weight of the PPO mirror (symmetry) loss, source A; 0 = off")
    ap.add_argument("--lit-ceiling", action="store_true",
                    help="print the measured non-stepping ceiling (m, z_c, dCOP, J "
                         "per direction) and the joint mask for this model, then exit")
    ap.add_argument("--frame-stack", type=int, default=1,
                    help="stack the last N actor observations into the policy input "
                         "(1 = today's memoryless contract; the source SaW policy is "
                         "an LSTM and the hand-built balancer uses hidden state, so "
                         "N>1 tests partial observability). Single-env only")
    ap.add_argument("--torch-threads", type=int, default=1,
                    help="torch intra-op thread count")
    ap.add_argument("--alive-weight", type=float, default=1.0,
                    help="weight of the positive survival term; raise it (e.g. 10) "
                         "when the policy dodges the one-off termination penalty by "
                         "collapsing into a non-terminating pose")
    ap.add_argument("--push-curriculum", choices=("on", "off"), default="on",
                    help="ramped training pushes (<= TRAIN_MAX_IMPULSE=12 N*s; the "
                         "gate's held-out magnitudes stay unseen)")
    ap.add_argument("--push-start-steps", type=int, default=20_000)
    ap.add_argument("--push-warmup-steps", type=int, default=400_000)
    ap.add_argument("--out", default="")
    ap.add_argument("--save-every", type=int, default=50_000)
    ap.add_argument("--resume", default="", help="checkpoint to resume from")
    ap.add_argument("--no-sigint-save", action="store_true")
    ap.add_argument("--lock", choices=("off", "on", "auto"), default="auto",
                    help="advisory data/locks/sim.lock: off = never take it "
                         "(use this for multi-hour runs so other sims are not "
                         "starved); on = always; auto = only when the estimated "
                         "runtime is under 60 s (short smoke runs)")
    ap.add_argument("--lock-wait", type=float, default=900.0)
    args = ap.parse_args(argv)
    if args.lit_ceiling:
        from .lit import joint_mask as _joint_mask
        from .lit import measure_ceiling

        print(json.dumps({"ceiling": measure_ceiling().as_dict(),
                          "joint_mask": _joint_mask().as_dict()},
                         indent=2, sort_keys=True))
        return 0
    hidden = tuple(int(x) for x in str(args.hidden).split(",") if x.strip())
    cfg = TrainConfig(task=args.task, steps=args.steps,
                      rollout_steps=args.rollout_steps, seed=args.seed,
                      n_envs=args.n_envs, vec_backend=args.vec_backend,
                      gamma=args.gamma, lam=args.lam, lr=args.lr,
                      entropy_coef=args.entropy_coef, value_coef=args.value_coef,
                      clip=args.clip, epochs=args.epochs,
                      minibatches=args.minibatches, hidden=hidden,
                      action_mode=(args.action_mode or "absolute"),
                      residual_scale=(0.5 if args.residual_scale is None
                                      else args.residual_scale),
                      log_every=args.log_every,
                      out=args.out, save_every=args.save_every,
                      torch_threads=args.torch_threads,
                      push_curriculum=(args.push_curriculum == "on"),
                      push_start_steps=args.push_start_steps,
                      push_warmup_steps=args.push_warmup_steps,
                      push_seed=args.push_seed,
                      alive_weight=args.alive_weight,
                      reward_set=args.reward_set,
                      freeze_joints=(args.freeze_joints == "balance"),
                      lit_weights=_parse_lit_weights(args.lit_weight),
                      lit_push=args.lit_push,
                      lit_push_prob=args.lit_push_prob,
                      lit_push_interval=args.lit_push_interval,
                      lit_push_interval_jitter=args.lit_push_interval_jitter,
                      lit_push_min_frac=args.lit_push_min_frac,
                      lit_push_max_frac=args.lit_push_max_frac,
                      lit_push_cap=args.lit_push_cap,
                      lit_push_directions=args.lit_push_directions,
                      lit_push_height=args.lit_push_height,
                      lit_push_seed=args.lit_push_seed,
                      mirror_loss_coef=args.mirror_loss_coef,
                      frame_stack=int(args.frame_stack))
    use_lock = use_lock_for(args.lock, args.steps)
    if use_lock:
        from .lock import SimLock

        with SimLock(owner=f"solo.train {args.task}", wait_s=args.lock_wait):
            print(f"[solo.train] holding the sim lock (--lock={args.lock})")
            return _run_train(cfg, args)
    if args.lock == "auto":
        print("[solo.train] long run: NOT taking the sim lock (--lock=auto); "
              "other agents' sims share the box")
    return _run_train(cfg, args)


def _run_train(cfg: TrainConfig, args) -> int:
    if args.resume:
        from dataclasses import replace

        from rl.checkpoint import load_checkpoint

        ckpt = load_checkpoint(args.resume)
        try:
            mode, scale = resolve_action_mode(ckpt, args.action_mode)
        except ValueError as exc:
            raise SystemExit(f"[solo.train] --resume {args.resume}: {exc}")
        if args.residual_scale is not None:      # explicit CLI override
            scale = float(args.residual_scale)
        if (mode, scale) != (cfg.action_mode, cfg.residual_scale):
            print(f"[solo.train] resume: checkpoint config wins over CLI defaults "
                  f"(action_mode={mode}, residual_scale={scale})")
        cfg = replace(cfg, action_mode=mode, residual_scale=scale)
        trainer = SoloTrainer(cfg)
        try:
            trainer.load(args.resume)
            stats = trainer.run()
            if cfg.out:
                trainer.save()
        finally:
            trainer.close()
    else:
        stats = train(cfg, on_sigint_save=not args.no_sigint_save)
    print("train done:", json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
