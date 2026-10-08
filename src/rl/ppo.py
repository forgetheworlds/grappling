"""PPO core: config, GAE(lambda), clipped surrogate update, LR schedule.

Standard single-agent PPO (Schulman et al. 2017) on the learner's 29-dim
half of the 58-dim env action; the critic is a separate net on privileged
state (see :mod:`rl.net`).  Everything is torch on CPU, sized for 4 cores.

Conventions pinned by tests:

* ``dones[t]`` means the episode ended at step ``t``; the bootstrap value
  ``V(s_{t+1})`` is multiplied by ``1 - dones[t]`` (an auto-resetting vec env
  returns the *reset* obs at that step).
* ``compute_gae`` returns ``(advantages, returns)`` with
  ``returns = advantages + values``.
* Advantages are normalized over the whole rollout before minibatching.
* Gradients are clipped by global norm; the LR follows a linear schedule
  between ``cfg.lr`` and ``cfg.lr * cfg.lr_end_frac``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .net import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    """All PPO/training hyperparameters (CPU-sized defaults)."""

    # data size
    rollout_steps: int = 2048        # per env, per iteration
    n_envs: int = 3
    frame_stack: int = 1
    # objective
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    epochs: int = 4
    minibatches: int = 4
    normalize_advantage: bool = True
    adv_eps: float = 1e-8
    target_kl: float | None = 0.03    # epochs stop once approx KL exceeds 1.5x
    # network / action map
    hidden: tuple[int, ...] = (256, 256)
    log_std_init: float = -1.0
    action_mode: str = "absolute"      # "absolute" | "residual"
    residual_scale: float = 0.5
    # optimization
    lr: float = 3e-4
    lr_schedule: str = "linear"        # "linear" | "constant"
    lr_end_frac: float = 0.1
    grad_clip: float = 0.5
    #: Separate gradient-norm clamps for the actor and the critic (``None`` =
    #: use ``grad_clip`` for both, the historical shared behaviour).  The
    #: 2026-10-08 audit measured the value term's gradient at 38x the policy
    #: term's (289.0 vs 7.5) against a shared clip of 0.5, so the actor moved
    #: ~600x less than the unclipped direction while the critic chased noisy
    #: targets: set these to keep the actor's step independent of value error.
    grad_clip_actor: float | None = None
    grad_clip_critic: float | None = None
    # misc
    seed: int = 0
    device: str = "cpu"
    torch_threads: int = 1             # 4-core box: one torch thread

    def as_dict(self) -> dict:
        return asdict(self)

    def net_config(self):
        """:class:`rl.net.NetConfig` with the network/action fields of this config."""
        from .net import NetConfig

        return NetConfig(hidden=tuple(self.hidden), log_std_init=self.log_std_init,
                         action_mode=self.action_mode, residual_scale=self.residual_scale)

    @staticmethod
    def from_dict(d: dict) -> "PPOConfig":
        return _from_dict(PPOConfig, d)


def _from_dict(cls, d: dict):
    known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in d.items() if k in known})  # type: ignore[call-arg]


def lr_at(progress: float, cfg: PPOConfig) -> float:
    """Learning rate at training ``progress`` in [0, 1] (1 = end of training)."""
    if cfg.lr_schedule == "constant":
        return float(cfg.lr)
    if cfg.lr_schedule != "linear":
        raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}")
    p = float(np.clip(progress, 0.0, 1.0))
    return float(cfg.lr * (1.0 - p * (1.0 - cfg.lr_end_frac)))


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                last_value: np.ndarray, gamma: float, lam: float):
    """Generalized advantage estimation (batched over ``(T, N)``).

    Returns ``(advantages, returns)``, both ``(T, N)`` float64.
    """
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    d = np.asarray(dones, dtype=np.float64)
    if r.shape != v.shape or r.shape != d.shape:
        raise ValueError(f"shape mismatch: rewards {r.shape}, values {v.shape}, dones {d.shape}")
    last = np.asarray(last_value, dtype=np.float64).reshape(r.shape[1])
    T = r.shape[0]
    adv = np.zeros_like(r)
    running = np.zeros(r.shape[1], dtype=np.float64)
    for t in range(T - 1, -1, -1):
        next_value = last if t == T - 1 else v[t + 1]
        nonterminal = 1.0 - d[t]
        delta = r[t] + gamma * next_value * nonterminal - v[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
    return adv, adv + v


@dataclass
class RolloutBatch:
    """One rollout of the learner: ``(T, N, ...)`` float32 arrays + meta."""

    actor_obs: np.ndarray
    critic_obs: np.ndarray
    actions_unit: np.ndarray
    logp: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    meta: dict = field(default_factory=dict)
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.rewards.shape  # (T, N)

    def finalize(self, last_value: np.ndarray, cfg: PPOConfig) -> "RolloutBatch":
        adv, ret = compute_gae(self.rewards, self.values, self.dones, last_value, cfg.gamma, cfg.lam)
        self.advantages = adv
        self.returns = ret
        return self

    def flatten_tensors(self, device: str = "cpu") -> dict:
        """Flatten ``(T, N, ...)`` -> ``(T*N, ...)`` tensors for the update."""
        if self.advantages is None:
            raise RuntimeError("RolloutBatch.finalize() before flatten_tensors()")
        T, N = self.shape
        to_t = lambda a, dt: torch.as_tensor(np.asarray(a).reshape((T * N,) + np.shape(a)[2:]), dtype=dt, device=device)
        return {
            "actor_obs": to_t(self.actor_obs, torch.float32),
            "critic_obs": to_t(self.critic_obs, torch.float32),
            "actions_unit": to_t(self.actions_unit, torch.float32),
            "logp": to_t(self.logp, torch.float32),
            "values": to_t(self.values, torch.float32),
            "advantages": to_t(self.advantages, torch.float32),
            "returns": to_t(self.returns, torch.float32),
        }

    def num_steps(self) -> int:
        return int(self.rewards.size)


def ppo_update(policy: ActorCritic, optimizer: torch.optim.Optimizer, batch: RolloutBatch,
               cfg: PPOConfig, lr: float, generator: torch.Generator | None = None) -> dict:
    """One PPO update (``cfg.epochs`` passes over ``cfg.minibatches`` minibatches)."""
    data = batch.flatten_tensors(cfg.device)
    n = data["advantages"].shape[0]
    adv = data["advantages"]
    if cfg.normalize_advantage:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + cfg.adv_eps)
    old_logp = data["logp"]
    mb_size = max(1, n // cfg.minibatches)
    for group in optimizer.param_groups:
        # a group may carry its own ratio (the split actor/critic lr); the
        # scheduled ``lr`` scales every group from its own base
        group["lr"] = float(lr) * float(group.get("lr_ratio", 1.0))
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "approx_kl": 0.0, "clip_frac": 0.0, "epochs_run": 0, "lr": float(lr),
             "n_samples": int(n)}
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
    epochs_run = 0
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n - mb_size + 1, mb_size):
            idx = perm[start:start + mb_size]
            new_logp, entropy = policy.actor.evaluate(data["actor_obs"][idx], data["actions_unit"][idx])
            values = policy.value(data["critic_obs"][idx])
            ratio = torch.exp(new_logp - old_logp[idx])
            surr1 = ratio * adv[idx]
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * ((values - data["returns"][idx]) ** 2).mean()
            ent = entropy.mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * ent
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip_actor is None and cfg.grad_clip_critic is None:
                nn.utils.clip_grad_norm_(
                    list(policy.actor.parameters()) + list(policy.critic.parameters()),
                    cfg.grad_clip)
            else:
                nn.utils.clip_grad_norm_(
                    list(policy.actor.parameters()),
                    cfg.grad_clip if cfg.grad_clip_actor is None else cfg.grad_clip_actor)
                nn.utils.clip_grad_norm_(
                    list(policy.critic.parameters()),
                    cfg.grad_clip if cfg.grad_clip_critic is None else cfg.grad_clip_critic)
            optimizer.step()
            with torch.no_grad():
                stats["policy_loss"] += float(policy_loss)
                stats["value_loss"] += float(value_loss)
                stats["entropy"] += float(ent)
                stats["clip_frac"] += float(((ratio - 1.0).abs() > cfg.clip).float().mean())
        epochs_run += 1
        with torch.no_grad():
            new_logp_all, _ = policy.actor.evaluate(data["actor_obs"], data["actions_unit"])
            approx_kl = float((old_logp - new_logp_all).mean())
        stats["epochs_run"] = epochs_run
        stats["approx_kl"] = approx_kl
        if cfg.target_kl is not None and approx_kl > 1.5 * cfg.target_kl:
            break
    # average the accumulated per-minibatch stats
    n_mb = max(1, (n // mb_size) * epochs_run)
    for k in ("policy_loss", "value_loss", "entropy", "clip_frac"):
        stats[k] /= n_mb
    with torch.no_grad():
        # the behaviour noise (P1-4's acceptance reads this): sigma in z-space
        stats["log_std_mean"] = float(policy.actor.log_std.mean())
        stats["sigma_mean"] = float(torch.exp(policy.actor.log_std.clamp(
            *policy.actor.cfg.log_std_clip)).mean())
    return stats


if __name__ == "__main__":  # self-check
    # hand-checkable GAE toy case (numbers precomputed in tests/test_rl.py)
    r = np.array([[1.0], [2.0], [3.0], [4.0]])
    v = np.array([[0.5], [1.0], [1.5], [2.0]])
    d = np.zeros_like(r)
    adv, ret = compute_gae(r, v, d, [2.5], 0.9, 0.8)
    assert abs(adv[0, 0] - 6.389024) < 1e-9, adv
    assert np.allclose(ret, adv + v)
    cfg = PPOConfig()
    assert lr_at(0.0, cfg) == cfg.lr
    assert abs(lr_at(1.0, cfg) - cfg.lr * cfg.lr_end_frac) < 1e-12
    from .net import NetConfig

    net = ActorCritic(8, 8 + 4, act_dim=3, cfg=NetConfig(hidden=(16, 16)))
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    T, N = 5, 2
    b = RolloutBatch(
        actor_obs=np.zeros((T, N, 8), np.float32), critic_obs=np.zeros((T, N, 12), np.float32),
        actions_unit=np.zeros((T, N, 3), np.float32), logp=np.zeros((T, N), np.float32),
        values=np.zeros((T, N), np.float32), rewards=np.ones((T, N), np.float32),
        dones=np.zeros((T, N), np.float32))
    b.finalize(np.zeros(N), cfg)
    st = ppo_update(net, opt, b, cfg, cfg.lr, torch.Generator().manual_seed(0))
    assert all(np.isfinite(st[k]) for k in ("policy_loss", "value_loss", "entropy", "approx_kl")), st
    print("rl.ppo self-check OK:", {k: round(st[k], 4) for k in ("policy_loss", "value_loss",
                                                                 "entropy", "approx_kl", "lr")})
