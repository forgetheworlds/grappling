#!/usr/bin/env python
"""Training-health monitor + reward-alignment diagnostics for the solo drill.

Why this exists (2026-10-08): the v5 residual-balance run DEGRADED from its own
warm start (init = the stand keyframe, 0.0026 rad off; by 301k steps mean
upright 0.583 vs the scripted hold's 0.849) and nobody saw it until a manual
monitor read.  This script is the cheap, re-runnable health check plus the
mechanisms that decide *why*: ranked-trajectory reward scoring, the
keyframe-optimality theorem check, and the init-vs-degraded per-term
decomposition.

Subcommands (all sim runs hold ``data/locks/sim.lock``):

  selfcheck  pure-function assertions (no sim): explained variance, per-term
             decomposition arithmetic.
  health     ONE-SCREEN health table for a checkpoint: upright/fall trend across
             available mid-run reads, per-term reward breakdown on visited
             states, explained variance of the critic, action saturation %,
             joint-limit violations, NaN/Inf checks, episode-length
             distribution, and (default) the keyframe-optimality check.
  ranked     the deciding experiment: score the SAME battery episodes for the
             policy, stand_hold, zero_action and random_init_policy; rank by
             physical quality and by reward side by side, and print the per-term
             table that pays for any wrong ordering.
  keyframe   keyframe-optimality: total + per-term reward of the exact stand
             keyframe ctrl vs per-joint perturbations (+-0.02/0.05/0.10 rad over
             all 29 joints) and seeded random directions, in the NOMINAL no-push
             case.  Scored under every configured reward (shipped balance,
             v5's alive=10 training reward, balance_lit).
  trainlike  init-policy vs the checkpoint policy on the training distribution
             (jittered no-push resets, 8 s horizon): per-term decomposition of
             what the degraded policy earns and which term pays for it.

Commands (from the repo root, venv active; MUJOCO_GL=egl for headless):

    MUJOCO_GL=egl .venv/bin/python scripts/solo_train_health.py health \
        --ckpt checkpoints/solo/t1_balance_v5.pt

    MUJOCO_GL=egl .venv/bin/python scripts/solo_train_health.py ranked \
        --ckpt checkpoints/solo/t1_balance_v5.pt

    MUJOCO_GL=egl .venv/bin/python scripts/solo_train_health.py keyframe
    MUJOCO_GL=egl .venv/bin/python scripts/solo_train_health.py trainlike \
        --ckpt checkpoints/solo/t1_balance_v5.pt

Cost (4-core box, load ~1): health ~40-60 s, ranked ~3-5 min for the full
48-push battery, keyframe ~2-3 min, trainlike ~2 min.  Under heavy peer load
(loadavg > 8) multiply by 2-3x.  ``health --no-keyframe --episodes-nopush 4``
is the ~20 s smoke variant.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from solo.baselines import (PolicyController, RandomInitPolicyController,  # noqa: E402
                            StandHoldController, ZeroActionController)
from solo.env import SoloEnv  # noqa: E402
from solo.lock import SimLock  # noqa: E402
from solo.metrics import METRICS_DIR, write_json  # noqa: E402
from solo.obs import ACTOR_DIM, CRITIC_DIM  # noqa: E402
from solo.reward import RewardWeights, TaskReward  # noqa: E402
from solo.scene import N_JOINTS, load_solo_model, stand_frame  # noqa: E402

# NOTE: ``solo.eval`` (battery_pushes) is imported lazily inside the commands that
# need it: this module stays importable (and the diagnostics tests stay runnable)
# even while a peer is mid-edit in eval.py.

HEALTH_DIR = METRICS_DIR / "solo_health"


# --------------------------------------------------------------------------
# pure scoring helpers (hand-testable; see tests/solo/test_reward_diagnostics.py)
# --------------------------------------------------------------------------
def explained_variance(values, targets) -> float:
    """``1 - Var(target - value) / Var(target)`` (a.k.a. EV of the critic).

    1.0 for a perfect predictor, 0.0 for a constant predictor (or a degenerate
    constant target, where the ratio is undefined and no-variance is reported as
    0.0).  Negative when the predictor is worse than predicting the mean.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    t = np.asarray(targets, dtype=np.float64).reshape(-1)
    if v.shape != t.shape:
        raise ValueError(f"shape mismatch: values {v.shape}, targets {t.shape}")
    if v.size == 0:
        raise ValueError("empty arrays")
    var_t = float(np.var(t))
    if var_t <= 0.0:
        return 0.0
    return float(1.0 - np.var(t - v) / var_t)


@dataclass(frozen=True)
class ScoreConfig:
    """One reward configuration a recorded trajectory can be scored under.

    The trajectory is physics; the reward is a linear function of the logged
    term values, so any weight vector / term set can be scored post-hoc from one
    recorded rollout (``solo.reward.TaskReward`` is linear in the terms).
    """

    name: str
    term_set: str | None = None
    overrides: tuple[tuple[str, float], ...] = ()
    gamma: float = 0.995
    termination_penalty: float = 100.0

    def weights(self) -> RewardWeights:
        w = RewardWeights()
        for key, value in self.overrides:
            w = replace(w, **{str(key): float(value)})
        return w

    def reward(self, task: str = "balance") -> TaskReward:
        return TaskReward(task, weights=self.weights(), gamma=self.gamma,
                          termination_penalty=self.termination_penalty,
                          term_set=self.term_set)


#: shipped balance reward, the reward v5 actually trained on, and the
#: literature term set LitImplement is shipping (all scored on every trace).
DEFAULT_CONFIGS: tuple[ScoreConfig, ...] = (
    ScoreConfig("shipped_balance"),
    ScoreConfig("v5_training", overrides=(("alive", 10.0),)),
    ScoreConfig("balance_lit", term_set="balance_lit"),
)


def score_trace(cfg: ScoreConfig, inputs, terminal_cause: str | None = None) -> dict:
    """Total + per-term (raw and weight x term) reward along one recorded trace.

    ``inputs`` are ``RewardInputs`` captured at each control step; the terminal
    penalty (if any) is applied once, after the last step.  ``sum(weighted)``
    equals ``total`` exactly for every config (linearity).
    """
    tr = cfg.reward()
    raw: dict[str, float] = {}
    weighted: dict[str, float] = {}
    total = 0.0
    disc = 0.0
    gamma = float(cfg.gamma)
    for i, inp in enumerate(inputs):
        r, terms = tr.step(inp)
        total += r
        disc += (gamma ** i) * r
        for k, v in terms.items():
            raw[k] = raw.get(k, 0.0) + float(v)
            weighted[k] = weighted.get(k, 0.0) + float(getattr(tr.weights, k)) * float(v)
    if terminal_cause in ("fall", "dorsal"):
        pen, tterms = tr.terminal(terminal_cause)
        total += pen
        disc += (gamma ** len(inputs)) * pen
        for k, v in tterms.items():
            raw[k] = raw.get(k, 0.0) + float(v)
            weighted[k] = weighted.get(k, 0.0) + float(v)
    return {"total": float(total), "discounted": float(disc),
            "raw": raw, "weighted": weighted}


class TraceRecorder:
    """Wrap a ``TaskReward`` so every step's ``RewardInputs`` is kept.

    The env only ever calls ``step``/``terminal`` on ``env.reward`` (and
    ``as_dict`` from ``config()``), so assignment is a drop-in.
    """

    def __init__(self, reward: TaskReward):
        self._reward = reward
        self.inputs: list = []

    def step(self, inp):
        self.inputs.append(copy.deepcopy(inp))
        return self._reward.step(inp)

    def terminal(self, cause):
        return self._reward.terminal(cause)

    def as_dict(self):
        return self._reward.as_dict()

    @property
    def weights(self):
        return self._reward.weights

    @property
    def terms(self):
        return self._reward.terms


# --------------------------------------------------------------------------
# env / episode plumbing
# --------------------------------------------------------------------------
def make_env(*, task: str = "balance", weights: RewardWeights | None = None,
             action_mode: str = "residual", residual_scale: float = 0.5,
             jitter: bool = True, model=None, seed: int = 0) -> tuple[SoloEnv, TraceRecorder]:
    env = SoloEnv(model=model if model is not None else load_solo_model(),
                  task=task, seed=int(seed), weights=weights,
                  action_mode=action_mode, residual_scale=float(residual_scale),
                  jitter=jitter)
    trace = TraceRecorder(env.reward)
    env.reward = trace
    return env, trace


def run_episode(env: SoloEnv, trace: TraceRecorder, controller, seed: int, *,
                push=None, command=None, max_steps: int | None = None) -> dict:
    """Run one episode; returns physics stats + the recorded reward-input trace."""
    env.reset(seed=int(seed), push=push, command=command)
    trace.inputs.clear()
    steps = 0
    cause = None
    truncated = False
    ret = 0.0
    bad = 0
    while True:
        action = controller(env, env.data)
        obs, reward, terminated, truncated, info = env.step(action)
        ret += float(reward)
        steps += 1
        if not (np.isfinite(reward) and np.isfinite(obs["actor"]).all()
                and np.isfinite(obs["critic"]).all()
                and np.isfinite(np.asarray(env.data.ctrl)).all()):
            bad += 1
        if terminated or truncated:
            cause = info.get("termination")
            break
        if max_steps is not None and steps >= max_steps:
            break
    rows = env.recorder.rows
    inputs = list(trace.inputs)
    return {"seed": int(seed), "steps": int(steps), "termination": cause,
            "truncated": bool(truncated), "return_env": float(ret),
            "nonfinite": int(bad), "rows": rows, "inputs": inputs,
            "stats": _row_stats(rows, int(steps))}


def _row_stats(rows: list[dict], steps: int) -> dict:
    def col(key):
        return np.array([float(r[key]) for r in rows if r.get(key) is not None],
                        dtype=np.float64)

    up = col("upright")
    return {
        "steps": int(steps),
        "mean_upright": float(up.mean()) if up.size else None,
        "min_upright": float(up.min()) if up.size else None,
        "final_upright": float(up[-1]) if up.size else None,
        "mean_pelvis_z": float(col("pelvis_z").mean()) if rows else None,
        "com_offset_max": float(col("com_offset").max()) if rows else None,
        "sat_frac_mean": float(col("sat_frac").mean()) if rows else None,
        "sat_frac_max": float(col("sat_frac").max()) if rows else None,
        "limit_prox_max": float(col("limit_prox").max()) if rows else None,
        "act_delta_mean": float(col("act_delta").mean()) if rows else None,
    }


def aggregate(episodes: list[dict]) -> dict:
    """Mean over episodes of the per-episode physics stats (+ fall rate)."""
    keys = ("mean_upright", "min_upright", "mean_pelvis_z", "com_offset_max",
            "sat_frac_mean", "sat_frac_max", "limit_prox_max", "act_delta_mean",
            "steps")
    out = {}
    for k in keys:
        vals = [e["stats"][k] for e in episodes if e["stats"].get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    n = len(episodes)
    out["n_episodes"] = n
    out["n_steps"] = int(sum(e["steps"] for e in episodes))
    out["fall_rate"] = sum(1 for e in episodes if e["termination"] == "fall") / max(1, n)
    out["termination_rate"] = sum(1 for e in episodes
                                  if e["termination"] is not None) / max(1, n)
    out["nonfinite"] = int(sum(e["nonfinite"] for e in episodes))
    lens = np.array([e["steps"] for e in episodes], dtype=np.float64)
    out["ep_len_min"] = float(lens.min())
    out["ep_len_median"] = float(np.median(lens))
    out["ep_len_max"] = float(lens.max())
    return out


def score_episodes(cfg: ScoreConfig, episodes: list[dict]) -> dict:
    """Score every episode under ``cfg``; returns per-term sums + totals."""
    totals = []
    disc = []
    raw: dict[str, float] = {}
    weighted: dict[str, float] = {}
    steps = 0
    for e in episodes:
        s = score_trace(cfg, e["inputs"], e["termination"])
        totals.append(s["total"])
        disc.append(s["discounted"])
        for k, v in s["raw"].items():
            raw[k] = raw.get(k, 0.0) + v
        for k, v in s["weighted"].items():
            weighted[k] = weighted.get(k, 0.0) + v
        steps += e["steps"]
    return {
        "config": cfg.name,
        "total_mean": float(np.mean(totals)) if totals else None,
        "total_std": float(np.std(totals)) if totals else None,
        "discounted_mean": float(np.mean(disc)) if disc else None,
        "raw": raw,
        "weighted": weighted,
        "raw_per_step": {k: v / max(1, steps) for k, v in raw.items()},
        "weighted_per_step": {k: v / max(1, steps) for k, v in weighted.items()},
        "share": ({k: v / max(1e-12, sum(abs(x) for x in weighted.values()))
                   for k, v in weighted.items()}),
    }


# --------------------------------------------------------------------------
# checkpoint loading
# --------------------------------------------------------------------------
def load_checkpoint_policy(path: str | Path, *, cli_mode: str | None = None):
    """Load a checkpoint copy -> (net, train_cfg, steps, mode, scale, ckpt)."""
    import torch

    from rl.checkpoint import apply_checkpoint, load_checkpoint
    from rl.net import ActorCritic, NetConfig
    from solo.train import resolve_action_mode

    src = Path(path)
    if not src.exists():
        raise SystemExit(f"checkpoint {src} does not exist")
    tmp = Path("/tmp") / f"health_{src.name}"
    shutil.copy2(src, tmp)                      # never read a live writer's file
    ckpt = load_checkpoint(str(tmp))
    tc = dict((ckpt.get("config") or {}).get("train") or {})
    steps = int((ckpt.get("state") or {}).get("steps_done", -1))
    mode, scale = resolve_action_mode(ckpt, cli_mode)
    hidden = tuple(tc.get("hidden", (256, 256)))
    net = ActorCritic(ACTOR_DIM, CRITIC_DIM, act_dim=N_JOINTS,
                      cfg=NetConfig(hidden=hidden, action_mode=mode,
                                    residual_scale=scale))
    apply_checkpoint(ckpt, policy=net)
    net.eval()
    torch.set_num_threads(1)
    return net, tc, steps, mode, scale, ckpt


def fresh_init_policy(*, seed: int = 0, hidden=(256, 256), mode: str = "residual",
                      scale: float = 0.5):
    """A representative *untrained* policy (the v5 init was not checkpointed)."""
    import torch

    from rl.net import ActorCritic, NetConfig

    torch.manual_seed(int(seed))
    net = ActorCritic(ACTOR_DIM, CRITIC_DIM, act_dim=N_JOINTS,
                      cfg=NetConfig(hidden=tuple(hidden), action_mode=mode,
                                    residual_scale=scale))
    net.eval()
    return net


def startup_ctrl_diff(env: SoloEnv, net) -> float:
    """max|ctrl - base_action| after the first step of an untrained policy (rad)."""
    import torch

    env.reset(seed=0)
    with torch.no_grad():
        unit = net.actor.deterministic_unit(
            torch.from_numpy(env.actor_vector().astype(np.float32)).unsqueeze(0)).numpy()[0]
    env.step(env.ctrl_from_policy(unit))
    diff = float(np.abs(np.asarray(env.data.ctrl, np.float64)
                        - env._base_action).max())
    env.reset(seed=0)
    return diff


def install_push_curriculum(env: SoloEnv, tc: dict, steps: int, episode_seed: int) -> None:
    """The trainer's own push-curriculum wiring at ``steps`` (no-op if disabled)."""
    if env.push_curriculum is None or not bool(tc.get("push_curriculum", True)):
        return
    cur = replace(env.push_curriculum,
                  start_steps=int(tc.get("push_start_steps", 20_000)),
                  warmup_steps=int(tc.get("push_warmup_steps", 400_000)),
                  seed=int(tc.get("push_seed", 0)))
    env.set_push_schedule(cur.schedule_for(int(steps), int(episode_seed)))


# --------------------------------------------------------------------------
# critic diagnosis: explained variance over a training-style rollout
# --------------------------------------------------------------------------
def rollout_diagnosis(net, env: SoloEnv, tc: dict, steps: int, episode_seed: int, *,
                      push_curriculum: bool = True, grad_probe: bool = True) -> dict:
    """Mirror of ``SoloTrainer.collect`` minus the update: sampled actions,
    critic values, GAE targets from the checkpoint's own gamma/lam.

    Cheap (~2048 env steps) and reports the numbers the PPO update acts on:
    explained variance, value RMSE, advantage mean/std, return mean/std, and
    (``grad_probe``) the unclipped grad norms of the policy vs value loss terms.
    """
    import torch

    from rl.ppo import compute_gae

    torch.set_num_threads(1)
    gamma = float(tc.get("gamma", 0.995))
    lam = float(tc.get("lam", 0.95))
    obs = env.reset(seed=int(episode_seed))
    rewards = np.zeros((int(steps),), np.float64)
    values = np.zeros((int(steps),), np.float64)
    dones = np.zeros((int(steps),), np.float64)
    actor_obs = np.zeros((int(steps), ACTOR_DIM), np.float32)
    critic_obs = np.zeros((int(steps), CRITIC_DIM), np.float32)
    actions = np.zeros((int(steps), N_JOINTS), np.float32)
    logp = np.zeros((int(steps),), np.float32)
    seed = int(episode_seed)
    for t in range(int(steps)):
        with torch.no_grad():
            a = torch.from_numpy(obs["actor"].astype(np.float32)).unsqueeze(0)
            c = torch.from_numpy(obs["critic"].astype(np.float32)).unsqueeze(0)
            sample = net.actor.sample(a)
            unit = sample["unit"].numpy()[0]
            logp[t] = float(sample["logp"][0])
            values[t] = float(net.value(c)[0])
        actor_obs[t] = obs["actor"]
        critic_obs[t] = obs["critic"]
        actions[t] = unit
        next_obs, reward, terminated, truncated, info = env.step(
            env.ctrl_from_policy(unit))
        rewards[t] = float(reward)
        dones[t] = 1.0 if (terminated or truncated) else 0.0
        if terminated or truncated:
            seed += 1
            if push_curriculum and info.get("episode_end"):
                install_push_curriculum(env, tc, int(steps), seed)
            obs = env.reset(seed=seed)
        else:
            obs = next_obs
    with torch.no_grad():
        last_value = float(net.value(
            torch.from_numpy(obs["critic"].astype(np.float32)).unsqueeze(0))[0])
    adv, ret = compute_gae(rewards.reshape(-1, 1), values.reshape(-1, 1),
                           dones.reshape(-1, 1), np.array([last_value]), gamma, lam)
    adv = adv.reshape(-1)
    ret = ret.reshape(-1)
    value_loss = 0.5 * float(np.mean((values - ret) ** 2))
    # Monte-Carlo (unbootstrapped) return-to-go per state, cut at episode ends:
    # the honest target for "did the critic learn the value function?".
    mc = np.zeros_like(rewards)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = rewards[t] + (0.0 if dones[t] else gamma * acc)
        mc[t] = acc
    out = {
        "steps": int(steps),
        "n_episodes": int(dones.sum()),
        "gamma": gamma, "lam": lam,
        "explained_variance": explained_variance(values, ret),
        "ev_monte_carlo": explained_variance(values, mc),
        "value_rmse": float(np.sqrt(np.mean((values - ret) ** 2))),
        "value_rmse_mc": float(np.sqrt(np.mean((values - mc) ** 2))),
        "value_loss_half_mse": value_loss,
        "return_mean": float(ret.mean()), "return_std": float(ret.std()),
        "return_min": float(ret.min()), "return_max": float(ret.max()),
        "value_mean": float(values.mean()), "value_std": float(values.std()),
        "value_min": float(values.min()), "value_max": float(values.max()),
        "adv_mean": float(adv.mean()), "adv_std": float(adv.std()),
        "reward_mean": float(rewards.mean()), "reward_sum": float(rewards.sum()),
    }
    if grad_probe:
        out["grad_probe"] = _grad_norm_probe(net, tc, actor_obs, critic_obs,
                                             actions, logp, adv, ret)
    return out


def _grad_norm_probe(net, tc: dict, actor_obs, critic_obs, actions, logp, adv,
                     ret) -> dict:
    """Unclipped grad norms of the PPO policy term vs the value term.

    Replicates ``rl.ppo.ppo_update``'s first minibatch backward pass but keeps
    the two loss terms separate, so the answer to "is the update critic-bound?"
    is a number: if the value term's gradient norm alone exceeds
    ``TrainConfig.grad_clip`` (0.5), the global-norm clip is decided by the
    critic and the policy gradient is rescaled (or zeroed) with it.
    """
    import torch

    from rl.ppo import PPOConfig

    clip = float(tc.get("clip", 0.2))
    value_coef = float(tc.get("value_coef", 0.5))
    entropy_coef = float(tc.get("entropy_coef", 0.001))
    minibatches = int(tc.get("minibatches", 4))
    n = int(adv.shape[0])
    mb = max(1, n // max(1, minibatches))
    sl = slice(0, mb)
    a_obs = torch.as_tensor(actor_obs[sl], dtype=torch.float32)
    c_obs = torch.as_tensor(critic_obs[sl], dtype=torch.float32)
    act = torch.as_tensor(actions[sl], dtype=torch.float32)
    old_logp = torch.as_tensor(logp[sl], dtype=torch.float32)
    a = torch.as_tensor(adv[sl], dtype=torch.float32)
    a = (a - a.mean()) / (a.std(unbiased=False) + 1e-8)      # as pp.ppo does
    r = torch.as_tensor(ret[sl], dtype=torch.float32)
    new_logp, entropy = net.actor.evaluate(a_obs, act)
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * a
    surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * a
    loss_p = -torch.min(surr1, surr2).mean() - entropy_coef * entropy.mean()
    values = net.value(c_obs)
    loss_v = value_coef * 0.5 * ((values - r) ** 2).mean()
    actors = list(net.actor.parameters())
    critics = list(net.critic.parameters())
    gp = torch.autograd.grad(loss_p, actors, retain_graph=True, allow_unused=True)
    gv = torch.autograd.grad(loss_v, critics, retain_graph=True)
    g_all = torch.autograd.grad(loss_p + loss_v, actors + critics)

    def norm(gs):
        total = 0.0
        for g in gs:
            if g is not None:
                total += float((g ** 2).sum())
        return math.sqrt(total)

    grad_clip = float(tc.get("grad_clip", 0.5))
    for p in net.parameters():
        p.grad = None
    return {"minibatch": int(mb),
            "policy_term_grad_norm": norm(gp),
            "value_term_grad_norm": norm(gv),
            "total_grad_norm": norm(g_all),
            "grad_clip": grad_clip,
            "clip_binds": bool(norm(g_all) > grad_clip),
            "value_alone_binds": bool(norm(gv) > grad_clip)}


# --------------------------------------------------------------------------
# trend reads
# --------------------------------------------------------------------------
def health_trend(ckpt_path: str | Path | None = None) -> list[dict]:
    """Upright/fall readings already on disk (monitor + health snapshots)."""
    out = []
    pats = ["t1_*monitor_*.json", "solo_health_*.json", "health_*.json"]
    seen = set()
    for pat in pats:
        for p in sorted(METRICS_DIR.glob(pat)):
            if p.name in seen:
                continue
            seen.add(p.name)
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            agg = d.get("aggregate") or d.get("table") or {}
            up = agg.get("mean_upright")
            if up is None:
                up = (d.get("policy") or {}).get("mean_upright")
            if up is None:
                continue
            out.append({
                "source": p.name,
                "steps": d.get("steps"),
                "mean_upright": up,
                "fall_rate": agg.get("fall_rate"),
                "com_offset_max": agg.get("com_offset_max"),
                "action_mode": d.get("action_mode"),
                "verdict": d.get("verdict"),
            })
    out.sort(key=lambda r: (r["steps"] if r["steps"] is not None else 10**12,
                            r["source"]))
    return out


# --------------------------------------------------------------------------
# keyframe optimality
# --------------------------------------------------------------------------
def keyframe_conditions(base: np.ndarray, offsets=(0.02, 0.05, 0.10),
                        n_random: int = 24, seed: int = 0) -> list[dict]:
    """Keyframe + per-joint offsets (all 29 joints) + seeded random directions."""
    base = np.asarray(base, np.float64).reshape(-1)
    out = [{"name": "keyframe", "ctrl": base.copy(), "delta_norm": 0.0}]
    for j in range(base.size):
        for d in offsets:
            for sign in (+1.0, -1.0):
                ctrl = base.copy()
                ctrl[j] += sign * float(d)
                out.append({"name": f"joint{j:+d}", "ctrl": ctrl,
                            "delta_norm": float(d), "joint": int(j),
                            "offset": float(sign * d)})
    rng = np.random.default_rng(int(seed))
    for k in range(int(n_random)):
        v = rng.normal(size=base.size)
        v /= max(float(np.linalg.norm(v)), 1e-12)
        for norm in offsets:
            ctrl = base + norm * v
            out.append({"name": f"random{k}@{norm:g}", "ctrl": ctrl,
                        "delta_norm": float(norm), "direction_seed": int(seed)})
    return out


def keyframe_scan(env: SoloEnv, trace: TraceRecorder, base: np.ndarray, *,
                  steps: int = 100, conditions=None, lo=None, hi=None) -> dict:
    """Hold each perturbed ctrl for ``steps`` steps from the identical reset.

    Returns the baseline row plus every perturbation's scored trace (under every
    :data:`DEFAULT_CONFIGS` config) and physics stats.  Jitter is disabled: the
    nominal case the operator asked about.
    """
    lo = env.lo if lo is None else lo
    hi = env.hi if hi is None else hi
    conditions = conditions if conditions is not None else keyframe_conditions(base)
    rows = []
    for cond in conditions:
        ctrl = np.clip(np.asarray(cond["ctrl"], np.float64), lo, hi)
        clipped = float(np.abs(ctrl - np.asarray(cond["ctrl"], np.float64)).max())

        def controller(_env, _data, ctrl=ctrl):
            return ctrl

        ep = run_episode(env, trace, controller, seed=0, max_steps=int(steps))
        row = {"name": cond["name"], "delta_norm": cond.get("delta_norm"),
               "joint": cond.get("joint"), "offset": cond.get("offset"),
               "clipped": clipped, "steps": ep["steps"],
               "termination": ep["termination"], "stats": ep["stats"],
               "scores": {cfg.name: score_trace(cfg, ep["inputs"],
                                                ep["termination"])
                          for cfg in DEFAULT_CONFIGS}}
        rows.append(row)
    return {"steps": int(steps), "rows": rows}


def keyframe_verdict(scan: dict, cfg_name: str, *, tol: float = 1e-6) -> dict:
    base = scan["rows"][0]
    base_score = base["scores"][cfg_name]
    base_total = base_score["total"]
    beats = []
    for row in scan["rows"][1:]:
        score = row["scores"][cfg_name]
        delta = score["total"] - base_total
        if delta > tol:
            terms_delta = {k: float(score["weighted"].get(k, 0.0)
                                    - base_score["weighted"].get(k, 0.0))
                           for k in set(score["weighted"]) | set(base_score["weighted"])}
            beats.append({"name": row["name"], "delta": float(delta),
                          "delta_frac": float(delta / max(abs(base_total), 1e-12)),
                          "joint": row.get("joint"), "offset": row.get("offset"),
                          "terms_delta": terms_delta,
                          "stats": row["stats"]})
    beats.sort(key=lambda r: -r["delta"])
    return {"config": cfg_name, "base_total": float(base_total),
            "n_conditions": len(scan["rows"]), "n_beating": len(beats),
            "max_delta": float(beats[0]["delta"]) if beats else 0.0,
            "max_delta_frac": float(beats[0]["delta_frac"]) if beats else 0.0,
            "top": beats[:5]}


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------
def term_inventory(episodes: list[dict], cfg: ScoreConfig) -> dict:
    """Per-term realised statistics on visited states (raw mean/std/min/max) plus
    the weighted per-step contribution and its share of the total reward."""
    tr = cfg.reward()
    names = set(tr.terms) | {"termination"}
    raw: dict[str, list[float]] = {k: [] for k in names}
    total = 0.0
    steps = 0
    for e in episodes:
        total += score_trace(cfg, e["inputs"], e["termination"])["total"]
        for row in e["rows"]:
            for k, v in (row.get("terms") or {}).items():
                if k in raw:
                    raw[k].append(float(v))
            steps += 1
    per_step_total = total / max(1, steps)
    out = {}
    for k in sorted(names):
        vals = np.asarray(raw[k], dtype=np.float64)
        w = float(getattr(tr.weights, k, 1.0))
        if k == "termination":
            # one-off penalty: a per-step share is meaningless, report the sum
            out[k] = {"n": int(vals.size), "weight": 1.0, "one_off": True,
                      "sum": float(vals.sum()) if vals.size else 0.0}
            continue
        if vals.size == 0:
            out[k] = {"n": 0, "weight": w}
            continue
        weighted_mean = w * float(vals.mean())
        out[k] = {
            "n": int(vals.size), "weight": w,
            "raw_mean": float(vals.mean()), "raw_std": float(vals.std()),
            "raw_min": float(vals.min()), "raw_max": float(vals.max()),
            "weighted_per_step": weighted_mean,
            "share_of_total": weighted_mean / per_step_total if per_step_total else None,
        }
    out["__total__"] = {"weighted_per_step": per_step_total, "steps": steps}
    return out


def _loadavg() -> str:
    try:
        return f"{os.getloadavg()[0]:.2f}"
    except OSError:
        return "n/a"


def cmd_health(args) -> int:
    from solo.eval import battery_pushes

    t0 = time.perf_counter()
    print(f"[health] load {_loadavg()} | ckpt {args.ckpt}")
    net, tc, steps, mode, scale, ckpt = load_checkpoint_policy(
        args.ckpt, cli_mode=args.action_mode)
    weights = RewardWeights(alive=float(tc.get("alive_weight", 1.0)))
    env, trace = make_env(weights=weights, action_mode=mode, residual_scale=scale)
    episode_seed = int((ckpt.get("state") or {}).get("episode_seed", 0))
    install_push_curriculum(env, tc, steps, episode_seed)
    print(f"[health] ckpt steps={steps} mode={mode} alive_weight={weights.alive} "
          f"gamma={tc.get('gamma')} push_curriculum={tc.get('push_curriculum')}")

    nopush = []
    for i in range(int(args.episodes_nopush)):
        env.set_push_schedule(None)              # no-push (training-ish) episodes
        ep = run_episode(env, trace, PolicyController(net, name="ckpt"), seed=i)
        nopush.append(ep)

    battery = []
    if int(args.battery) > 0:
        plan = battery_pushes()[:int(args.battery)]
        for i, spec in enumerate(plan):
            from solo.pushes import PushSchedule
            ep = run_episode(env, trace, PolicyController(net, name="ckpt"), seed=i,
                             push=PushSchedule([spec]))
            battery.append(ep)

    # the rollout mirrors training conditions at the checkpoint's step: the
    # push curriculum is reinstalled (the no-push probes above disabled it).
    install_push_curriculum(env, tc, steps, episode_seed)
    rollout = rollout_diagnosis(net, env, tc, int(args.rollout_steps),
                                episode_seed, push_curriculum=bool(
                                    tc.get("push_curriculum", True)))
    # --- keyframe optimality (reduced grid by default: 1 + 29*4 conditions)
    keyframe = None
    if not args.no_keyframe:
        nokeys_env, nokeys_trace = make_env(weights=weights, action_mode=mode,
                                            residual_scale=scale, jitter=False)
        nokeys_env.set_push_schedule(None)
        base = np.asarray(stand_frame(nokeys_env.model)[1], np.float64)
        conditions = keyframe_conditions(base, offsets=(0.05, 0.10),
                                         n_random=0)
        scan = keyframe_scan(nokeys_env, nokeys_trace, base,
                             steps=int(args.keyframe_steps), conditions=conditions)
        keyframe = {"scan_steps": scan["steps"],
                    "verdicts": {cfg.name: keyframe_verdict(scan, cfg.name)
                                 for cfg in DEFAULT_CONFIGS},
                    "baseline": {cfg.name: scan["rows"][0]["scores"][cfg.name]
                                 for cfg in DEFAULT_CONFIGS}}

    # --- assemble + print -----------------------------------------------------
    agg_nopush = aggregate(nopush)
    terms = score_episodes(DEFAULT_CONFIGS[1], nopush)      # v5's own reward
    terms_shipped = score_episodes(DEFAULT_CONFIGS[0], nopush)
    inventory = term_inventory(nopush, DEFAULT_CONFIGS[1])
    inventory_battery = (term_inventory(battery, DEFAULT_CONFIGS[1])
                         if battery else None)
    trend = health_trend()
    snapshot = {
        "checkpoint": str(args.ckpt), "steps": steps, "action_mode": mode,
        "residual_scale": scale, "alive_weight": weights.alive,
        "train_cfg": tc,
        "nopush": {"aggregate": agg_nopush, "v5_training": terms,
                   "shipped_balance": terms_shipped,
                   "inventory": inventory},
        "battery": {"aggregate": aggregate(battery),
                    "v5_training": score_episodes(DEFAULT_CONFIGS[1], battery),
                    "inventory": inventory_battery}
        if battery else None,
        "rollout": rollout,
        "keyframe": keyframe,
        "trend": trend,
        "load": _loadavg(),
        "wall_s": None,
    }

    _print_health(snapshot)
    snapshot["wall_s"] = round(time.perf_counter() - t0, 1)
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.json) if args.json else HEALTH_DIR / f"solo_health_{steps}.json"
    write_json(out, snapshot)
    print(f"[health] wrote {out} in {snapshot['wall_s']}s (load {_loadavg()})")
    return 0


def _fmt(x, nd=3):
    if x is None:
        return "  n/a"
    return f"{x:.{nd}f}"


def _print_health(s: dict) -> None:
    print("=" * 100)
    print(f"SOLO BALANCE HEALTH  ckpt steps={s['steps']}  mode={s['action_mode']} "
          f"alive_weight={s['alive_weight']}  load={s['load']}")
    print("-" * 100)
    r = s["rollout"]
    print(f"CRITIC   explained_variance={r['explained_variance']:+.3f} (GAE target)  "
          f"{r['ev_monte_carlo']:+.3f} (Monte-Carlo return)  "
          f"value_rmse={r['value_rmse']:.2f}  value_loss(0.5MSE)={r['value_loss_half_mse']:.1f}")
    print(f"         return  mean {r['return_mean']:+.1f} std {r['return_std']:.1f} "
          f"[{r['return_min']:.1f}, {r['return_max']:.1f}]   "
          f"value mean {r['value_mean']:+.1f} std {r['value_std']:.1f} "
          f"[{r['value_min']:.1f}, {r['value_max']:.1f}]")
    print(f"         advantage mean {r['adv_mean']:+.2f} std {r['adv_std']:.2f} "
          f"(normalised to unit std inside the update: "
          f"{'YES - value error sets the advantage scale' if r['explained_variance'] < 0.5 else 'value fits'})")
    gp = r.get("grad_probe")
    if gp:
        print(f"         grad norms (one minibatch n={gp['minibatch']}): value-term "
              f"{gp['value_term_grad_norm']:.2f} vs policy-term {gp['policy_term_grad_norm']:.2f} "
              f"| total {gp['total_grad_norm']:.2f} vs grad_clip {gp['grad_clip']} "
              f"-> clip {'BINDS' if gp['clip_binds'] else 'inactive'}"
              f"{' (value term alone exceeds it)' if gp['value_alone_binds'] else ''}")
    print("-" * 100)
    print("TREND    (stored reads; upright / fall / com_max)")
    for t in s["trend"]:
        print(f"         steps={str(t['steps']):>9s}  upright={_fmt(t['mean_upright'])}  "
              f"fall={_fmt(t['fall_rate'])}  com_max={_fmt(t['com_offset_max'])}  "
              f"mode={t['action_mode']}  {t['source']}")
    print("-" * 100)
    a = s["nopush"]["aggregate"]
    print(f"EPISODES no-push x{a['n_episodes']}: steps/episode min {a['ep_len_min']:.0f} "
          f"med {a['ep_len_median']:.0f} max {a['ep_len_max']:.0f}  "
          f"fall_rate={a['fall_rate']:.3f}  mean_upright={_fmt(a['mean_upright'])}  "
          f"mean_pelvis_z={_fmt(a['mean_pelvis_z'])}")
    print(f"         sat_frac mean {_fmt(a['sat_frac_mean'])} max {_fmt(a['sat_frac_max'])}  "
          f"limit_prox_max {_fmt(a['limit_prox_max'])}  act_delta {_fmt(a['act_delta_mean'])}  "
          f"nonfinite={a['nonfinite']}")
    if s["battery"]:
        b = s["battery"]["aggregate"]
        print(f"         battery  x{b['n_episodes']}: steps min {b['ep_len_min']:.0f} "
              f"med {b['ep_len_median']:.0f} max {b['ep_len_max']:.0f}  "
              f"fall_rate={b['fall_rate']:.3f}  mean_upright={_fmt(b['mean_upright'])}  "
              f"com_max={_fmt(b['com_offset_max'])}")
    print("-" * 100)
    print("TERM INVENTORY (v5 training reward, no-push episodes; raw mean/std/min/max, "
          "weighted per-step, share)")
    inv = dict(s["nopush"]["inventory"])
    tot = inv.pop("__total__")
    for k in sorted(inv, key=lambda k: -abs(inv[k].get("weighted_per_step") or 0.0)):
        d = inv[k]
        if d.get("one_off"):
            n_ep = s["nopush"]["aggregate"]["n_episodes"]
            print(f"         {k:<16s} one-off: {d['n']} firing(s) in {n_ep} episodes, "
                  f"sum {d['sum']:+.1f} ({d['sum'] / max(1, n_ep):+.1f}/episode)")
            continue
        if not d.get("n"):
            print(f"         {k:<16s} never fired on visited states")
            continue
        print(f"         {k:<16s} w={d['weight']:<5g} raw {d['raw_mean']:+.4f}"
              f"+-{d['raw_std']:.4f} [{d['raw_min']:+.3f},{d['raw_max']:+.3f}]  "
              f"weighted/step {d['weighted_per_step']:+.4f}  "
              f"share {100.0*(d['share_of_total'] or 0.0):+.1f}%")
    print(f"         {'TOTAL':<16s} weighted/step {tot['weighted_per_step']:+.4f} "
          f"over {tot['steps']} steps  |  episode return "
          f"{s['nopush']['v5_training']['total_mean']:+.1f} "
          f"+- {s['nopush']['v5_training']['total_std']:.1f} "
          f"(discounted {s['nopush']['v5_training']['discounted_mean']:+.1f})")
    if s["keyframe"]:
        print("-" * 100)
        print(f"KEYFRAME OPTIMALITY (nominal no-push, {s['keyframe']['scan_steps']} steps, "
              f"grid = all 29 joints x +-0.05/+-0.10 rad)")
        for name, v in s["keyframe"]["verdicts"].items():
            verdict = ("KEYFRAME MAXIMAL" if v["n_beating"] == 0 else
                       f"MISALIGNED: {v['n_beating']} perturbations beat it "
                       f"(max +{v['max_delta']:.3f}, {v['max_delta_frac']*100:+.3f}%)")
            print(f"         {name:<16s} base {v['base_total']:+.2f}  "
                  f"{v['n_conditions']-1} perturbations  -> {verdict}")
            for t in v["top"]:
                print(f"              beat by {t['name']:<14s} +{t['delta']:.3f} "
                      f"({t['delta_frac']*100:+.3f}%)  upright {_fmt(t['stats']['mean_upright'])}")
    flags = []
    if s["nopush"]["aggregate"]["nonfinite"]:
        flags.append("NaN/Inf in obs/ctrl/reward")
    if (s["nopush"]["aggregate"]["mean_upright"] or 0) < 0.5:
        flags.append("mean_upright < 0.5 (collapsing)")
    if r["explained_variance"] < 0.3:
        flags.append(f"critic EV {r['explained_variance']:+.2f} < 0.3 (value error dominates advantages)")
    if gp and gp["value_alone_binds"]:
        flags.append("value-term gradient alone exceeds grad_clip (the update is critic-bound)")
    if s["keyframe"] and any(v["n_beating"] for v in s["keyframe"]["verdicts"].values()):
        flags.append("keyframe not maximal in the nominal reward (see above)")
    print("-" * 100)
    print("VERDICT  " + ("OK" if not flags else "ATTENTION: " + "; ".join(flags)))
    print("=" * 100)


def cmd_ranked(args) -> int:
    """The deciding experiment: same episodes, four controllers, per-term scores."""
    from solo.eval import battery_pushes
    from solo.pushes import PushSchedule

    t0 = time.perf_counter()
    print(f"[ranked] load {_loadavg()}")
    net, tc, steps, mode, scale, _ = load_checkpoint_policy(
        args.ckpt, cli_mode=args.action_mode)
    weights = RewardWeights(alive=float(tc.get("alive_weight", 1.0)))
    plan = battery_pushes() if args.full_battery else battery_pushes(
        magnitudes=(4.0, 12.0, 16.0, 25.0), directions=4)
    print(f"[ranked] battery n={len(plan)} (full={bool(args.full_battery)}) "
          f"mode={mode} alive_weight={weights.alive}")

    names = ["policy", "stand_hold", "zero_action", "random_init_policy"]
    if args.with_zero_abs:
        names.append("zero_action_abs")
    out = {}
    for name in names:
        abs_mode = name == "zero_action_abs"
        env, trace = make_env(weights=weights,
                              action_mode="absolute" if abs_mode else mode,
                              residual_scale=scale)
        env.set_push_schedule(None)
        if name == "policy":
            fixed = PolicyController(net, name="policy")
        elif name == "stand_hold":
            fixed = StandHoldController()
        elif name in ("zero_action", "zero_action_abs"):
            fixed = ZeroActionController()
        elif name == "random_init_policy":
            fixed = None                       # seeded per episode
        else:
            raise ValueError(name)
        eps = []
        for i, spec in enumerate(plan):
            controller = (RandomInitPolicyController(seed=i) if fixed is None
                          else fixed)
            ep = run_episode(env, trace, controller, seed=i,
                             push=PushSchedule([spec]))
            eps.append(ep)
        agg = aggregate(eps)
        scores = {cfg.name: score_episodes(cfg, eps) for cfg in DEFAULT_CONFIGS}
        out[name] = {"aggregate": agg, "scores": scores}
        print(f"  [{name:18s}] steps/ep {agg['steps']:.0f} upright {agg['mean_upright']:.3f} "
              f"fall {agg['fall_rate']:.3f} com_max {agg['com_offset_max']:.3f} | "
              + "  ".join(f"{c}: {scores[c]['total_mean']:+.1f}"
                          for c in scores))

    # --- physical vs reward ranking, side by side ---------------------------
    rows = []
    for name, d in out.items():
        a = d["aggregate"]
        physical = (a["mean_upright"], -a["fall_rate"], -a["com_offset_max"])
        rows.append({"name": name, "physical": physical, "aggregate": a,
                     "scores": d["scores"]})
    phys_order = [r["name"] for r in sorted(rows, key=lambda r: r["physical"],
                                            reverse=True)]
    print("-" * 100)
    print("PHYSICAL RANK (best -> worst): " + " > ".join(phys_order))
    for cfg in DEFAULT_CONFIGS:
        rew_order = [r["name"] for r in sorted(
            rows, key=lambda r: r["scores"][cfg.name]["total_mean"], reverse=True)]
        match = "SAME" if rew_order[0] == phys_order[0] else "MISMATCH"
        print(f"REWARD RANK [{cfg.name}] ({match} top): " + " > ".join(rew_order))
    print("-" * 100)
    for cfg in DEFAULT_CONFIGS:
        print(f"PER-TERM [{cfg.name}] (weighted per-step contribution; "
              f"negative = penalty)")
        for r in rows:
            s = r["scores"][cfg.name]
            contrib = "  ".join(f"{k}:{v:+.3f}" for k, v in
                                sorted(s["weighted_per_step"].items(),
                                       key=lambda kv: -abs(kv[1])))
            print(f"    {r['name']:<18s} total/ep {s['total_mean']:+9.2f}  "
                  f"disc {s['discounted_mean']:+9.2f}  {contrib}")
    report = {"checkpoint": str(args.ckpt), "steps": steps, "mode": mode,
              "alive_weight": weights.alive, "n_battery": len(plan),
              "controllers": out, "physical_rank": phys_order,
              "wall_s": round(time.perf_counter() - t0, 1), "load": _loadavg()}
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(args.json) if args.json else HEALTH_DIR / f"ranked_{steps}.json"
    write_json(path, report)
    print(f"[ranked] wrote {path} in {report['wall_s']}s (load {_loadavg()})")
    return 0


def cmd_keyframe(args) -> int:
    """Keyframe optimality under every configured reward, nominal no-push case."""
    t0 = time.perf_counter()
    print(f"[keyframe] load {_loadavg()}")
    env, trace = make_env(weights=RewardWeights(alive=float(args.alive_weight)),
                          action_mode=args.action_mode,
                          residual_scale=args.residual_scale, jitter=False)
    env.set_push_schedule(None)
    base = np.asarray(stand_frame(env.model)[1], np.float64)
    conditions = keyframe_conditions(
        base, offsets=(0.02, 0.05, 0.10),
        n_random=int(args.n_random), seed=int(args.seed))
    print(f"[keyframe] {len(conditions)} conditions x {args.steps} steps "
          f"(nominal, jitter off, no push)")
    scan = keyframe_scan(env, trace, base, steps=int(args.steps),
                         conditions=conditions)
    base_row = scan["rows"][0]
    for cfg in DEFAULT_CONFIGS:
        s = base_row["scores"][cfg.name]
        total_steps = max(1, base_row["steps"])
        print(f"  keyframe [{cfg.name}] total {s['total']:+.3f} over "
              f"{base_row['steps']} steps ({s['total']/total_steps:+.4f}/step): "
              + "  ".join(f"{k}:{v:+.3f}" for k, v in sorted(s["weighted"].items(),
                                                            key=lambda kv: -abs(kv[1]))))
    verdicts = {}
    for cfg in DEFAULT_CONFIGS:
        v = keyframe_verdict(scan, cfg.name)
        verdicts[cfg.name] = v
        print(f"  {cfg.name:<16s} n_beating={v['n_beating']}/{v['n_conditions']-1} "
              f"max_delta={v['max_delta']:+.4f} ({v['max_delta_frac']*100:+.4f}%)")
        for t in v["top"]:
            print(f"       beat by {t['name']:<16s} {t['delta']:+.4f} "
                  f"({t['delta_frac']*100:+.3f}%) upright {_fmt(t['stats']['mean_upright'])}")
    # long-horizon reference: the keyframe itself over 8 s
    long_env, long_trace = make_env(weights=RewardWeights(alive=float(args.alive_weight)),
                                    action_mode=args.action_mode,
                                    residual_scale=args.residual_scale, jitter=False)
    long_env.set_push_schedule(None)
    long_scan = keyframe_scan(long_env, long_trace, base, steps=400,
                              conditions=[{"name": "keyframe", "ctrl": base,
                                           "delta_norm": 0.0}])
    for cfg in DEFAULT_CONFIGS:
        s = long_scan["rows"][0]["scores"][cfg.name]
        print(f"  keyframe 8s [{cfg.name}] total {s['total']:+.1f} "
              f"({s['total']/400:+.4f}/step, return/episode)")
    report = {"steps": args.steps, "n_conditions": len(conditions),
              "alive_weight": args.alive_weight, "action_mode": args.action_mode,
              "verdicts": verdicts, "baseline": {
                  cfg.name: {**base_row["scores"][cfg.name], "stats": base_row["stats"]}
                  for cfg in DEFAULT_CONFIGS},
              "keyframe_8s": {cfg.name: long_scan["rows"][0]["scores"][cfg.name]
                              for cfg in DEFAULT_CONFIGS},
              "top": {cfg.name: verdicts[cfg.name]["top"] for cfg in DEFAULT_CONFIGS},
              "scan_rows": [{k: v for k, v in row.items() if k != "scores"}
                            | {"scores": {c: {"total": row["scores"][c]["total"],
                                              "weighted": row["scores"][c]["weighted"]}
                                          for c in row["scores"]}}
                            for row in scan["rows"]],
              "wall_s": round(time.perf_counter() - t0, 1), "load": _loadavg()}
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(args.json) if args.json else HEALTH_DIR / "keyframe_optimality.json"
    write_json(path, report)
    print(f"[keyframe] wrote {path} in {report['wall_s']}s (load {_loadavg()})")
    return 0


def cmd_trainlike(args) -> int:
    """Init policy vs the checkpoint on the training distribution (no push, jitter)."""
    t0 = time.perf_counter()
    print(f"[trainlike] load {_loadavg()}")
    net, tc, steps, mode, scale, ckpt = load_checkpoint_policy(
        args.ckpt, cli_mode=args.action_mode)
    weights = RewardWeights(alive=float(tc.get("alive_weight", 1.0)))
    env, trace = make_env(weights=weights, action_mode=mode, residual_scale=scale,
                          jitter=True)
    env.set_push_schedule(None)                  # nominal training distribution
    init = fresh_init_policy(seed=0, hidden=tuple(tc.get("hidden", (256, 256))),
                             mode=mode, scale=scale)
    print(f"[trainlike] ckpt steps={steps} alive_weight={weights.alive}; "
          f"fresh-init startup max|ctrl-base| = {startup_ctrl_diff(env, init):.6f} rad")

    conds = {
        "init_policy": lambda env_, i: PolicyController(init, name="init"),
        "init_policy_sampled": lambda env_, i: PolicyController(init, name="init",
                                                                stochastic=True),
        "ckpt_policy": lambda env_, i: PolicyController(net, name="ckpt"),
        "ckpt_policy_sampled": lambda env_, i: PolicyController(net, name="ckpt",
                                                                stochastic=True),
        "stand_hold": lambda env_, i: StandHoldController(),
        "random_init_policy": lambda env_, i: RandomInitPolicyController(seed=i),
    }
    import torch as _torch
    with _torch.no_grad():
        sigma_init = float(init.actor.log_std.exp().mean())
        sigma_ckpt = float(net.actor.log_std.exp().mean())
    print(f"[trainlike] behaviour std (log_std.exp): init {sigma_init:.4f} "
          f"ckpt {sigma_ckpt:.4f}")
    out = {}
    for name, factory in conds.items():
        eps = []
        for i in range(int(args.episodes)):
            ep = run_episode(env, trace, factory(env, i), seed=i)
            eps.append(ep)
        agg = aggregate(eps)
        scores = {cfg.name: score_episodes(cfg, eps) for cfg in DEFAULT_CONFIGS}
        out[name] = {"aggregate": agg, "scores": scores}
        print(f"  [{name:18s}] steps/ep {agg['steps']:.0f} upright {agg['mean_upright']:.3f} "
              f"fall {agg['fall_rate']:.3f} pelvis_z {agg['mean_pelvis_z']:.3f} | "
              + "  ".join(f"{c}: {scores[c]['total_mean']:+.1f}" for c in scores))
    if args.push_curriculum:
        # the second block is the *actual* late-training distribution: the ramped
        # training pushes at this checkpoint's step (2 pushes/episode at 401k).
        print("  -- with the training push curriculum at this step --")
        for name in ("ckpt_policy", "stand_hold"):
            eps = []
            for i in range(int(args.episodes)):
                install_push_curriculum(env, tc, steps, episode_seed=i)
                ep = run_episode(env, trace, conds[name](env, i), seed=i)
                eps.append(ep)
            env.set_push_schedule(None)
            agg = aggregate(eps)
            scores = {cfg.name: score_episodes(cfg, eps) for cfg in DEFAULT_CONFIGS}
            out[f"{name}_pushed"] = {"aggregate": agg, "scores": scores}
            print(f"  [{name + '_pushed':18s}] steps/ep {agg['steps']:.0f} "
                  f"upright {agg['mean_upright']:.3f} fall {agg['fall_rate']:.3f} | "
                  + "  ".join(f"{c}: {scores[c]['total_mean']:+.1f}" for c in scores))
    print("-" * 100)
    cfg = DEFAULT_CONFIGS[1]                       # v5's own training reward
    init_s = out["init_policy"]["scores"][cfg.name]
    ckpt_s = out["ckpt_policy"]["scores"][cfg.name]
    print(f"PER-TERM DELTA [ckpt - init] under {cfg.name} "
          f"(weighted per-step; raw means)")
    for k in sorted(set(init_s["weighted_per_step"]) | set(ckpt_s["weighted_per_step"]),
                    key=lambda k: abs(ckpt_s["weighted_per_step"].get(k, 0.0)
                                      - init_s["weighted_per_step"].get(k, 0.0)),
                    reverse=True):
        dw = ckpt_s["weighted_per_step"].get(k, 0.0) - init_s["weighted_per_step"].get(k, 0.0)
        dr = ckpt_s["raw_per_step"].get(k, 0.0) - init_s["raw_per_step"].get(k, 0.0)
        flag = "  <-- pays the degradation" if dw < -1e-3 else ""
        print(f"    {k:<16s} d_weighted {dw:+.4f}  d_raw {dr:+.4f}{flag}")
    print(f"    TOTAL            ckpt {ckpt_s['total_mean']:+.1f}/ep vs "
          f"init {init_s['total_mean']:+.1f}/ep  "
          f"(shipped-reward: ckpt {out['ckpt_policy']['scores']['shipped_balance']['total_mean']:+.1f} "
          f"vs init {out['init_policy']['scores']['shipped_balance']['total_mean']:+.1f})")
    report = {"checkpoint": str(args.ckpt), "steps": steps, "alive_weight": weights.alive,
              "episodes": int(args.episodes), "condition": {
                  "jitter": True, "push": "none", "horizon_s": env.horizon},
              "behaviour_std": {"init": sigma_init, "ckpt": sigma_ckpt},
              "controllers": out, "wall_s": round(time.perf_counter() - t0, 1),
              "load": _loadavg()}
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(args.json) if args.json else HEALTH_DIR / f"trainlike_{steps}.json"
    write_json(path, report)
    print(f"[trainlike] wrote {path} in {report['wall_s']}s (load {_loadavg()})")
    return 0


def cmd_selfcheck(args) -> int:
    """Pure-function assertions (no sim); mirrors the pytest numbers."""
    # explained variance
    t = np.array([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(t, t) == 1.0
    assert explained_variance(np.full_like(t, t.mean()), t) == 0.0
    assert explained_variance(np.zeros(3), np.zeros(3)) == 0.0
    # per-term decomposition == total (linearity) on a hand-made trace
    from solo.commands import Command
    from solo.reward import RewardInputs
    inp = RewardInputs(cmd=Command(vx=0.0), torso_up_z=1.0, pelvis_z=0.79,
                       action=np.zeros(N_JOINTS),
                       prev_action=np.zeros(N_JOINTS))
    for cfg in DEFAULT_CONFIGS:
        out = score_trace(cfg, [inp, inp, replace(inp, torso_up_z=0.1, pelvis_z=0.3)],
                          terminal_cause="fall")
        total_w = sum(out["weighted"].values())
        assert abs(total_w - out["total"]) < 1e-9, (cfg.name, total_w, out["total"])
    print("solo_train_health selfcheck OK:",
          {"ev_perfect": 1.0, "ev_constant": 0.0, "decomposition": "exact"})
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    ap.add_argument("--wait-lock", type=float, default=900.0,
                    help="seconds to wait for the advisory sim lock")

    p = sub.add_parser("selfcheck")
    p.set_defaults(func=cmd_selfcheck)

    p = sub.add_parser("health")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--action-mode", default="auto")
    p.add_argument("--episodes-nopush", type=int, default=8)
    p.add_argument("--battery", type=int, default=8,
                   help="leading battery pushes to run (0 disables)")
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--keyframe-steps", type=int, default=100)
    p.add_argument("--no-keyframe", action="store_true")
    p.add_argument("--json", default=None)
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("ranked")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--action-mode", default="auto")
    p.add_argument("--full-battery", action="store_true",
                   help="use the full 48-push gate battery (default: 16-push subset)")
    p.add_argument("--with-zero-abs", action="store_true",
                   help="also run zero_action through the absolute-mode env")
    p.add_argument("--json", default=None)
    p.set_defaults(func=cmd_ranked)

    p = sub.add_parser("keyframe")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--n-random", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alive-weight", type=float, default=10.0)
    p.add_argument("--action-mode", default="residual")
    p.add_argument("--residual-scale", type=float, default=0.5)
    p.add_argument("--json", default=None)
    p.set_defaults(func=cmd_keyframe)

    p = sub.add_parser("trainlike")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--action-mode", default="auto")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--push-curriculum", action="store_true",
                   help="also run ckpt/stand_hold under the ramped training "
                        "pushes at this checkpoint's step (the late-training "
                        "distribution)")
    p.add_argument("--json", default=None)
    p.set_defaults(func=cmd_trainlike)

    args = ap.parse_args(argv)
    if args.command == "selfcheck":
        return args.func(args)
    sims = {"health", "ranked", "keyframe", "trainlike"}
    if args.command in sims:
        with SimLock(owner=f"solo_train_health:{args.command}",
                     wait_s=float(args.wait_lock)):
            return args.func(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
