"""Actor/critic networks and the action map to the env ctrlrange.

Architecture (MISSION Phase 3/5: small nets, CPU-only):

* **Actor** -- 2x256 tanh MLP, per-robot input, output = mean of a diagonal
  Gaussian over 29 joint-position targets; a learned state-independent
  ``log_std`` is added; samples are squashed by ``tanh`` and then affinely
  mapped into the model's ``ctrlrange`` (or used as residuals on a reference
  base action, clipping to ``ctrlrange``).  The log-probability includes the
  standard tanh change-of-variables correction.
* **Critic** -- a separate 2x256 tanh MLP on ``[actor obs | privileged]``
  (``rl.privileged.CriticObsBuilder``); input dim = actor dim + 162.
  The two nets share no weights and take different inputs: the actor path
  cannot read privileged state.

The policy is per-robot and shared across robots (``obs_r -> 29``), so a
scripted opponent can occupy the other half of the 58-dim env action; the
full-match action is ``concat(a_half, b_half)`` -> ``(2, 29)``/``(58,)`` as
the environment expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from wrestling.env import ACT_SLICE, N_JOINTS

_ATANH_EPS = 1e-6


@dataclass(frozen=True)
class NetConfig:
    """Network + action-map hyperparameters (defaults are the CPU-sized choice)."""

    hidden: tuple[int, ...] = (256, 256)
    activation: str = "tanh"
    log_std_init: float = -1.0
    log_std_clip: tuple[float, float] = (-5.0, 2.0)
    action_mode: str = "absolute"       # "absolute" | "residual"
    residual_scale: float = 0.5         # fraction of half ctrlrange (residual mode)


def _activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unknown activation {name!r}")


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, activation: str = "tanh") -> nn.Sequential:
    """Small MLP with the given hidden widths."""
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), _activation(activation)]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Diagonal-Gaussian policy over ``act_dim`` joint-position targets."""

    def __init__(self, obs_dim: int, act_dim: int = N_JOINTS, cfg: NetConfig = NetConfig()):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cfg = cfg
        self.trunk = mlp(self.obs_dim, cfg.hidden, self.act_dim, cfg.activation)
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(m.bias)
        # small final layer so the initial policy is near zero-residual
        nn.init.orthogonal_(self.trunk[-1].weight, gain=0.01)
        self.log_std = nn.Parameter(torch.full((self.act_dim,), float(cfg.log_std_init)))

    # ------------------------------------------------------------------ dist
    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def _std(self) -> torch.Tensor:
        lo, hi = self.cfg.log_std_clip
        return torch.exp(self.log_std.clamp(lo, hi)).expand(self.act_dim)

    def distribution(self, obs: torch.Tensor) -> Normal:
        return Normal(self.mean(obs), self._std())

    def sample(self, obs: torch.Tensor) -> dict:
        """Sample: unit actions = tanh(z), logp with the tanh correction."""
        dist = self.distribution(obs)
        z = dist.rsample()
        unit = torch.tanh(z)
        logp = _squashed_logp(dist.log_prob(z), z).sum(-1)
        return {"unit": unit, "logp": logp, "mean": dist.mean, "std": dist.stddev, "z": z}

    def evaluate(self, obs: torch.Tensor, unit_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """logp (tanh-corrected) and Normal entropy of stored unit actions."""
        dist = self.distribution(obs)
        z = torch.atanh(unit_actions.clamp(-1.0 + _ATANH_EPS, 1.0 - _ATANH_EPS))
        logp = _squashed_logp(dist.log_prob(z), z).sum(-1)
        return logp, dist.entropy().sum(-1)

    def deterministic_unit(self, obs: torch.Tensor) -> torch.Tensor:
        """Mean action squashed to ``(-1, 1)`` (evaluation)."""
        return torch.tanh(self.mean(obs))


def _squashed_logp(log_prob_z: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """``log p(tanh(z)) = log p(z) - sum log(1 - tanh(z)^2)`` per sample."""
    return log_prob_z - torch.log(1.0 - torch.tanh(z) ** 2 + 1e-6)


class Critic(nn.Module):
    """State-value head on ``[actor obs | privileged]``."""

    def __init__(self, obs_dim: int, cfg: NetConfig = NetConfig()):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.cfg = cfg
        self.trunk = mlp(self.obs_dim, cfg.hidden, 1, cfg.activation)
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.trunk[-1].weight, gain=1.0)

    def forward(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(critic_obs).squeeze(-1)


class ActorCritic(nn.Module):
    """Container: actor (actor obs) + critic (actor+privileged obs)."""

    def __init__(self, actor_dim: int, critic_dim: int, act_dim: int = N_JOINTS, cfg: NetConfig = NetConfig()):
        super().__init__()
        self.cfg = cfg
        self.actor_dim = int(actor_dim)
        self.critic_dim = int(critic_dim)
        self.act_dim = int(act_dim)
        self.actor = Actor(actor_dim, act_dim, cfg)
        self.critic = Critic(critic_dim, cfg)

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs)

    def n_params(self) -> dict:
        return {
            "actor": int(sum(p.numel() for p in self.actor.parameters())),
            "critic": int(sum(p.numel() for p in self.critic.parameters())),
        }


# ------------------------------------------------------------------ actions
class ActionMapper:
    """Maps unit actions ``[-1, 1]`` to env ctrl targets within ``ctrlrange``.

    * ``absolute``: ``ctrl = mid + half * unit`` where ``mid``/``half`` are the
      actuator ctrlrange centre/half-range -- ``unit = ±1`` reaches the range
      limits exactly (tanh-scaled to the env action range).
    * ``residual``: ``ctrl = clip(base + residual_scale * half * unit)`` --
      a fraction of the half-range around a reference base action.
    """

    def __init__(self, lo: np.ndarray, hi: np.ndarray,
                 mode: str = "absolute", residual_scale: float = 0.5):
        self.lo = np.asarray(lo, dtype=np.float64).reshape(-1)
        self.hi = np.asarray(hi, dtype=np.float64).reshape(-1)
        if self.lo.shape != self.hi.shape:
            raise ValueError("ctrlrange lo/hi shape mismatch")
        if np.any(self.hi <= self.lo):
            raise ValueError("ctrlrange hi <= lo")
        if mode not in ("absolute", "residual"):
            raise ValueError(f"unknown action mode {mode!r}")
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")
        self.mode = mode
        self.residual_scale = float(residual_scale)
        self.mid = 0.5 * (self.lo + self.hi)
        self.half = 0.5 * (self.hi - self.lo)
        self.act_dim = int(self.lo.size)

    def to_ctrl(self, unit: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
        u = np.clip(np.asarray(unit, dtype=np.float64), -1.0, 1.0)
        if u.shape[-1] != self.act_dim:
            raise ValueError(f"unit action dim {u.shape[-1]} != {self.act_dim}")
        if self.mode == "absolute":
            # clip only for floating-point exactness: |u| <= 1 lands on the range
            ctrl = np.clip(self.mid + self.half * u, self.lo, self.hi)
        else:
            if base is None:
                raise ValueError("residual mode requires a base action")
            b = np.asarray(base, dtype=np.float64)
            if b.shape[-1] != self.act_dim:
                raise ValueError(f"base action dim {b.shape[-1]} != {self.act_dim}")
            ctrl = np.clip(b + self.residual_scale * self.half * u, self.lo, self.hi)
        return ctrl

    def in_bounds(self, ctrl: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        c = np.asarray(ctrl, dtype=np.float64)
        return (c >= self.lo - tol) & (c <= self.hi + tol)


def robot_action_bounds(model, robot: str) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` ctrlrange of one robot's 29 actuators."""
    sl = ACT_SLICE[robot]
    return (np.asarray(model.actuator_ctrlrange[:, 0][sl], dtype=np.float64),
            np.asarray(model.actuator_ctrlrange[:, 1][sl], dtype=np.float64))


def action_mapper_for(model, robot: str, cfg: NetConfig = NetConfig()) -> ActionMapper:
    lo, hi = robot_action_bounds(model, robot)
    return ActionMapper(lo, hi, cfg.action_mode, cfg.residual_scale)


def compose_action(learner_ctrl: np.ndarray, opponent_ctrl: np.ndarray, learner_robot: str = "a") -> np.ndarray:
    """Assemble the ``(58,)`` env action from the two 29-dim halves."""
    u = np.empty(2 * N_JOINTS, dtype=np.float64)
    if learner_robot == "a":
        u[0:N_JOINTS], u[N_JOINTS:] = learner_ctrl, opponent_ctrl
    else:
        u[0:N_JOINTS], u[N_JOINTS:] = opponent_ctrl, learner_ctrl
    return u


def count_out_of_bounds(action: np.ndarray, model, tol: float = 1e-9,
                        robot: str | None = None) -> int:
    """Number of action entries outside ``model`` ctrlrange (tolerance ``tol``).

    ``robot`` selects that robot's 29-actuator slice (for 29-dim actions);
    ``None`` expects a full 58-dim action.
    """
    if robot is None:
        lo = np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float64)
        hi = np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float64)
    else:
        lo, hi = robot_action_bounds(model, robot)
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.size != lo.size:
        raise ValueError(f"action size {a.size} != ctrlrange size {lo.size}"
                         + ("" if robot is None else f" for robot {robot}"))
    return int(np.sum((a < lo - tol) | (a > hi + tol)))


if __name__ == "__main__":  # self-check
    torch.manual_seed(0)
    cfg = NetConfig()
    net = ActorCritic(actor_dim=92, critic_dim=92 + 162, cfg=cfg)
    obs = torch.zeros(4, 92)
    out = net.actor.sample(obs)
    assert out["unit"].shape == (4, 29) and torch.all(out["unit"].abs() < 1.0)
    logp, ent = net.actor.evaluate(obs, out["unit"])
    assert torch.allclose(logp, out["logp"], atol=1e-4), (logp, out["logp"])
    assert ent.shape == (4,)
    v = net.value(torch.zeros(4, 92 + 162))
    assert v.shape == (4,)
    from wrestling.env import load_wrestling_model

    model = load_wrestling_model()
    mp = action_mapper_for(model, "a", cfg)
    ctrl = mp.to_ctrl(np.array([1.0] * 29), None)
    assert np.all(ctrl <= mp.hi) and np.all(ctrl >= mp.lo)
    res = ActionMapper(mp.lo, mp.hi, "residual", 0.5)
    c2 = res.to_ctrl(np.ones(29), mp.mid)
    assert np.all(res.in_bounds(c2))
    print("rl.net self-check OK:", {"params": net.n_params(), "actor_dim": net.actor_dim,
                                    "critic_dim": net.critic_dim})
