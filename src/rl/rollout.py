"""Rollout collection: policy -> vec env -> (obs, action, logp, value, reward) -> GAE.

:class:`RolloutCollector` owns the per-env frame stacks and the command clock,
so consecutive rollouts continue seamlessly (the vec env is *not* reset between
rollouts; matches reset themselves inside the backend).  Frames are reseeded
from the fresh match's first observation at episode boundaries.

Determinism: action sampling uses torch's global RNG (seeded by the trainer and
saved in checkpoints); action noise uses the numpy ``rng`` created from the run
seed.  The command clock is a pure function of
``(seed, env_index, exchange_index, exchange_time)`` (:mod:`rl.curriculum`), so
a resumed run cannot desynchronize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .curriculum import CommandScheduler
from .net import ActionMapper, ActorCritic, compose_action
from .obs import COMMAND_DIM, ENV_OBS_DIM, ActorObsBuilder
from .ppo import PPOConfig, RolloutBatch
from .reward import StageReward
from .scripted import reference_base_ctrl
from .vec import VecWrestlingEnv

ROBOT_INDEX = {"a": 0, "b": 1}


@dataclass
class OpponentPolicy:
    """Frozen policy used as the opponent in ``mode="pair"`` (stage E)."""

    policy: ActorCritic
    mapper: ActionMapper
    robot: str = "b"

    def sample_unit(self, actor_obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.policy.actor.sample(torch.as_tensor(np.asarray(actor_obs, dtype=np.float32)))
        return out["unit"].numpy()

    def mean_unit(self, actor_obs: np.ndarray) -> np.ndarray:
        """Deterministic (mean) action, for evaluation runs."""
        with torch.no_grad():
            unit = self.policy.actor.deterministic_unit(
                torch.as_tensor(np.asarray(actor_obs, dtype=np.float32)))
        return unit.numpy()


def _learner_qpos(priv_row: np.ndarray, robot: str) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(priv_row)
    return (p[0:36], p[36:72]) if robot == "a" else (p[36:72], p[0:36])


class RolloutCollector:
    """Collects fixed-length rollouts from a vec env with a persistent stack/clock."""

    def __init__(self, *, policy: ActorCritic, mapper: ActionMapper, scheduler: CommandScheduler,
                 stage_reward: StageReward, cfg: PPOConfig, learner_robot: str = "a",
                 opponent: OpponentPolicy | None = None, seed: int | None = None):
        self.policy = policy
        self.mapper = mapper
        self.scheduler = scheduler
        self.stage_reward = stage_reward
        self.cfg = cfg
        self.learner_robot = learner_robot
        self.opponent = opponent
        self.rng = np.random.default_rng(cfg.seed if seed is None else int(seed))
        self.actor_dim = int(cfg.frame_stack) * ENV_OBS_DIM + COMMAND_DIM
        self.lit = ROBOT_INDEX[learner_robot]
        self.oit = 1 - self.lit
        self.vec: VecWrestlingEnv | None = None
        self.builders: list[ActorObsBuilder] = []
        self.pending = None            # StepBatch at the current decision state
        self.decision_obs = None       # (N, actor_dim) learner obs for that state
        self.decision_obs_opp = None   # (N, actor_dim) opponent obs (pair mode)
        self.cmds = None               # (TechniqueCommand,) x N for that state
        self._running_return = None

    # ------------------------------------------------------------------ setup
    def attach(self, vec: VecWrestlingEnv) -> None:
        """Bind to a (possibly rebuilt) vec env; clears pending state."""
        if vec.mode == "pair" and self.opponent is None:
            raise ValueError("mode='pair' requires an OpponentPolicy")
        if vec.mode == "learner_only" and self.opponent is not None:
            raise ValueError("mode='learner_only' must not receive an opponent policy")
        self.vec = vec
        self.builders = [ActorObsBuilder(self.cfg.frame_stack) for _ in range(vec.n_envs)]
        self.pending = None
        self.decision_obs = None
        self.decision_obs_opp = None
        self.cmds = None

    # ---------------------------------------------------------------- rollout
    def collect(self, n_steps: int | None = None, *, noise_std: float = 0.0,
                stop_check: Callable[[], bool] | None = None) -> RolloutBatch:
        """Collect a rollout; returns it finalized with GAE(lambda)."""
        if self.vec is None:
            raise RuntimeError("RolloutCollector.attach(vec) before collect()")
        vec = self.vec
        T = int(n_steps if n_steps is not None else self.cfg.rollout_steps)
        N = vec.n_envs
        if self.pending is None:
            self.pending = vec.reset()
            self.cmds = [self.scheduler.command(i, 0, 0.0) for i in range(N)]
            obs = np.zeros((N, self.actor_dim), dtype=np.float32)
            obs_opp = np.zeros((N, self.actor_dim), dtype=np.float32)
            for i in range(N):
                oa, ob = self.builders[i].reset(self._pair(self.pending, i),
                                                (self.cmds[i], self.cmds[i]))
                obs[i] = oa if self.lit == 0 else ob
                obs_opp[i] = ob if self.lit == 0 else oa
            self.decision_obs = obs
            self.decision_obs_opp = obs_opp
            self._running_return = np.zeros(N, dtype=np.float64)
        if self._running_return is None:
            self._running_return = np.zeros(N, dtype=np.float64)
        p_dim = int(self.pending.priv.shape[1])

        obs_a = np.zeros((T, N, self.actor_dim), dtype=np.float32)
        obs_c = np.zeros((T, N, self.actor_dim + p_dim), dtype=np.float32)
        act_u = np.zeros((T, N, self.mapper.act_dim), dtype=np.float32)
        logp = np.zeros((T, N), dtype=np.float32)
        values = np.zeros((T, N), dtype=np.float32)
        rewards = np.zeros((T, N), dtype=np.float32)
        dones = np.zeros((T, N), dtype=np.float32)
        meta = {"exchange_records": [], "match_returns": [], "matches_completed": 0,
                "bounds_bad": 0, "techniques": [], "steps": 0, "partial": False}

        t = 0
        while t < T:
            if stop_check is not None and stop_check():
                meta["partial"] = True
                break
            batch = self.pending
            learner_obs = self.decision_obs
            opp_obs = self.decision_obs_opp if vec.mode == "pair" else None
            if vec.mode == "pair" and opp_obs is None:
                raise RuntimeError("pair mode without opponent observations")
            critic_obs = np.concatenate([learner_obs, batch.priv], axis=1)
            with torch.no_grad():
                out = self.policy.actor.sample(torch.as_tensor(learner_obs))
                unit = out["unit"].numpy()
                logp_t = out["logp"].numpy()
                value_t = self.policy.value(torch.as_tensor(critic_obs)).numpy()
            learner_ctrl = np.empty((N, self.mapper.act_dim), dtype=np.float64)
            for i in range(N):
                base = (reference_base_ctrl(self.cmds[i].technique, self.learner_robot, self.cmds[i].phase)
                        if self.mapper.mode == "residual" else None)
                learner_ctrl[i] = self.mapper.to_ctrl(unit[i], base)
            if noise_std > 0.0:
                learner_ctrl += self.rng.normal(0.0, noise_std, size=learner_ctrl.shape)
                learner_ctrl = np.clip(learner_ctrl, self.mapper.lo, self.mapper.hi)
            if vec.mode == "learner_only":
                action = learner_ctrl
            else:
                opp_unit = self.opponent.sample_unit(opp_obs)
                opp_ctrl = np.stack([self.opponent.mapper.to_ctrl(opp_unit[i]) for i in range(N)])
                action = np.stack([compose_action(learner_ctrl[i], opp_ctrl[i], self.learner_robot)
                                   for i in range(N)])

            batch = vec.step(action)
            meta["bounds_bad"] += int(batch.bounds_bad.sum())
            next_cmds = [self.scheduler.command(i, int(batch.infos[i].get("exchange_index", 0)),
                                                float(batch.infos[i].get("exchange_time", 0.0)))
                         for i in range(N)]
            r_learner = batch.rewards[:, self.lit].astype(np.float64)
            for i in range(N):
                q_self, q_opp = _learner_qpos(batch.priv[i], self.learner_robot)
                r_learner[i] += self.stage_reward.shaping(
                    technique=self.cmds[i].technique, phase=self.cmds[i].phase,
                    phase_next=next_cmds[i].phase, qpos_self=q_self, qpos_opp=q_opp)

            obs_a[t], obs_c[t], act_u[t] = learner_obs, critic_obs, unit
            logp[t], values[t], rewards[t] = logp_t, value_t, r_learner
            dones[t] = batch.dones.astype(np.float32)
            self._running_return += r_learner
            meta["steps"] += N
            for i in range(N):
                rec = batch.infos[i].get("exchange_ended")
                if rec is not None:
                    meta["exchange_records"].append(dict(rec, env=int(i)))
                if self.cmds[i].technique is not None:
                    meta["techniques"].append(self.cmds[i].technique)
                if batch.dones[i]:
                    meta["matches_completed"] += 1
                    meta["match_returns"].append(float(self._running_return[i]))
                    self._running_return[i] = 0.0
                    self.builders[i].reseed(self._pair(batch, i))
            # advance to the next decision state
            next_obs = np.zeros((N, self.actor_dim), dtype=np.float32)
            next_obs_opp = np.zeros((N, self.actor_dim), dtype=np.float32)
            for i in range(N):
                oa, ob = self.builders[i].step(self._pair(batch, i), (next_cmds[i], next_cmds[i]))
                next_obs[i] = oa if self.lit == 0 else ob
                next_obs_opp[i] = ob if self.lit == 0 else oa
            self.pending, self.cmds = batch, next_cmds
            self.decision_obs, self.decision_obs_opp = next_obs, next_obs_opp
            t += 1

        rb = RolloutBatch(obs_a[:t], obs_c[:t], act_u[:t], logp[:t], values[:t],
                          rewards[:t], dones[:t], meta=meta)
        if t == 0:
            rb.meta["partial"] = True
            return rb
        last_value = self._bootstrap_value()
        rb.finalize(last_value, self.cfg)
        return rb

    def set_stage_reward(self, stage_reward: StageReward) -> None:
        """Swap the shaping for a stage transition (keeps stacks/pending state)."""
        self.stage_reward = stage_reward

    # ----------------------------------------------------------------- helpers
    def _pair(self, batch, i: int):
        return ((batch.obs_a[i], batch.obs_b[i]) if self.lit == 0
                else (batch.obs_b[i], batch.obs_a[i]))

    def _bootstrap_value(self) -> np.ndarray:
        """V(s_T) for the pending state (already stacked in ``decision_obs``)."""
        crit = np.concatenate([self.decision_obs, self.pending.priv], axis=1)
        with torch.no_grad():
            return self.policy.value(torch.as_tensor(crit)).numpy().astype(np.float64)


if __name__ == "__main__":  # self-check
    from .reward import RewardWeights
    from .scripted import OpponentSpec
    from wrestling.env import load_wrestling_model

    torch.manual_seed(0)
    cfg = PPOConfig(rollout_steps=8, frame_stack=1)
    vec = VecWrestlingEnv(2, backend="sequential", seed=0, opponent_spec=OpponentSpec("stand_hold"),
                          env_kwargs={"match_clock": 5.0, "exchange_timeout": 2.5})
    m = load_wrestling_model()
    mapper = ActionMapper(m.actuator_ctrlrange[0:29, 0], m.actuator_ctrlrange[0:29, 1])
    policy = ActorCritic(92, 92 + 162, act_dim=29)
    sched = CommandScheduler(("DOUBLE_LEG",), command_on=True, seed=0)
    sr = StageReward(RewardWeights(technique_similarity=0.4, progress=0.3, outcome=1.0))
    rc = RolloutCollector(policy=policy, mapper=mapper, scheduler=sched, stage_reward=sr, cfg=cfg)
    rc.attach(vec)
    rb = rc.collect(8)
    assert rb.actor_obs.shape == (8, 2, 92) and rb.advantages.shape == (8, 2)
    assert np.isfinite(rb.advantages).all() and rb.meta["steps"] == 16
    rb2 = rc.collect(4)  # continuation: same envs, no reset, stacks preserved
    assert rb2.actor_obs.shape == (4, 2, 92)
    vec.close()
    print("rl.rollout self-check OK:", {"shape": rb.actor_obs.shape,
                                        "meta": {k: rb.meta[k] for k in ("steps", "bounds_bad",
                                                                         "matches_completed")}})
