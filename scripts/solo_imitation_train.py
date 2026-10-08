#!/usr/bin/env python
"""Imitation refinement — PPO on a reference-conditioned episode.

Leg 3 of the stepping-reference experiment (operator 2026-10-08): take ONE
video-derived reference, warm-start from the BC pose prior, and refine with PPO
against the DeepMimic-style imitation reward so the *policy alone* reproduces
the movement (no demonstration controller at inference).

Episode contract (one episode = one pass over the reference):

* reset to the reference's own initial state (``qpos[0]`` + its finite-difference
  velocities) plus an optional seeded joint perturbation ("slightly varied
  conditions");
* one control step per reference frame (50 Hz); the action is the env's own
  unit action (residual on the stand keyframe by default);
* reward = ``solo.imitation.step_terms(env, targets, k)["total"]`` (the
  DeepMimic terms vs the reference at frame ``k``);
* the deviation predicate (``imitation.deviation_reason``) ends the episode
  early with ``--deviation-penalty``;
* the reference ending ends the episode (the reference's own last frames are the
  stance return, so the terminal posture is inside the imitation objective).

Everything else is the shipped PPO core (``rl.ppo``) and the shipped net
(``rl.net.ActorCritic``); the BC artifact warm-starts the actor.

Run:  MUJOCO_GL=egl .venv/bin/python scripts/solo_imitation_train.py \
          --reference data/refs_video/shot_entry_full.npz \
          --bc data/solo/bc/bc_policy.pt --updates 4 --episodes-per-update 2 \
          --out checkpoints/solo/imitation_probe.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rl.checkpoint import warm_start_from_bc  # noqa: E402
from rl.net import ActorCritic, NetConfig  # noqa: E402
from rl.ppo import PPOConfig, RolloutBatch, ppo_update  # noqa: E402
from solo.bc import ReferenceTrack, load_reference, set_reference_initial_velocity  # noqa: E402
from solo.env import SoloEnv  # noqa: E402
from solo.imitation import (DEFAULT_WEIGHTS, PRESETS, ImitationTargets,  # noqa: E402
                            deviation_reason, imitation_reward, state_from_env)
from solo.obs import ACTOR_DIM, CRITIC_DIM  # noqa: E402
from solo.scene import N_JOINTS, load_solo_model  # noqa: E402

DEFAULT_REFERENCE = "data/refs_video/shot_entry_full.npz"
DEFAULT_BC = "data/solo/bc/bc_policy.pt"


class ImitationEpisode:
    """One reference-conditioned episode (see the module docstring)."""

    def __init__(self, track: ReferenceTrack, targets: ImitationTargets, weights,
                 *, action_mode: str = "residual", residual_scale: float = 0.5,
                 perturb_sigma: float = 0.0, model=None, seed: int = 0,
                 terminate_on_deviation: bool = True,
                 deviation_penalty: float = 10.0):
        self.tr = track
        self.targets = targets
        self.weights = weights
        self.perturb_sigma = float(perturb_sigma)
        self.terminate_on_deviation = bool(terminate_on_deviation)
        self.deviation_penalty = float(deviation_penalty)
        self.rng = np.random.default_rng(int(seed))
        self.N = len(track) - 1
        self.env = SoloEnv(model, task="balance", action_mode=action_mode,
                           residual_scale=float(residual_scale), seed=int(seed),
                           jitter=False, horizon=max(8.0, len(track) * 0.02 + 1.0))
        self.k = 0

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        q0 = np.asarray(self.tr.qpos[0], np.float64).copy()
        if self.perturb_sigma > 0.0:
            q0[7:36] += self.rng.normal(0.0, self.perturb_sigma, N_JOINTS)
            q0[:2] += self.rng.normal(0.0, 0.01, 2)
        self.env.reset(pose=q0, jitter=False)
        set_reference_initial_velocity(self.env, self.tr)
        self.k = 0
        return self.env.observation()

    def step(self, unit_action) -> tuple[dict, float, bool, dict]:
        ctrl = self.env.ctrl_from_policy(np.asarray(unit_action, np.float64))
        obs, _r, terminated, truncated, info = self.env.step(ctrl)
        self.k += 1
        cur = state_from_env(self.env)
        ref = self.targets.at(min(self.k, self.N))
        reward = float(imitation_reward(cur, ref, self.weights)["total"])
        done = bool(terminated or truncated) or self.k >= self.N
        reason = deviation_reason(cur, ref, self.weights)
        if reason is not None and not (terminated or truncated):
            if self.terminate_on_deviation:
                info = {**info, "deviation": reason}
                done = True
            else:
                # soft mode: pay for the deviation, keep the episode alive (the
                # BC prior diverges within ~4 frames, so a hard gate gives no
                # learning signal over the trajectory)
                reward -= self.deviation_penalty
                info = {**info, "deviation_soft": reason}
        return obs, reward, done, {**info, "k": self.k}


def collect(net: ActorCritic, ep: ImitationEpisode, cfg: PPOConfig, *, seed: int,
            n_episodes: int) -> RolloutBatch:
    """Collect ``n_episodes`` reference episodes into one padded ``RolloutBatch``.

    Variable-length episodes are concatenated along T with ``dones[t] = 1`` at
    every episode end (the GAE bootstrap uses ``1 - dones``), and the tail is
    padded with done=1 so the extra ticks contribute no advantage.
    """
    chunks = []
    for e in range(int(n_episodes)):
        obs = ep.reset(seed=int(seed) + e)
        a_obs, c_obs, act, logp, val, rew, don = [], [], [], [], [], [], []
        for _ in range(ep.N):
            ao = np.asarray(obs["actor"], np.float32)
            co = np.asarray(obs["critic"], np.float32)
            with torch.no_grad():
                sample = net.actor.sample(torch.from_numpy(ao).unsqueeze(0))
                value = float(net.value(torch.from_numpy(co).unsqueeze(0))[0])
            unit = sample["unit"][0].numpy()
            obs, r, done, info = ep.step(unit)
            a_obs.append(ao); c_obs.append(co); act.append(unit.astype(np.float32))
            logp.append(float(sample["logp"][0])); val.append(value)
            rew.append(float(r)); don.append(bool(done))
            if done:
                break
        chunks.append((a_obs, c_obs, act, logp, val, rew, don))
    # Concatenate the episodes along TIME as one stream with N=1: the episode
    # boundaries carry dones[t]=1, which is exactly what compute_gae's
    # ``1 - dones`` bootstrap needs.
    def stream(i, dt, extra=()):
        out = []
        for c in chunks:
            out.extend(c[i])
        return np.asarray(out, dtype=dt).reshape((len(out), 1) + tuple(extra))

    batch = RolloutBatch(
        actor_obs=stream(0, np.float32, (ACTOR_DIM,)),
        critic_obs=stream(1, np.float32, (CRITIC_DIM,)),
        actions_unit=stream(2, np.float32, (N_JOINTS,)),
        logp=stream(3, np.float32),
        values=stream(4, np.float32),
        rewards=stream(5, np.float32),
        dones=stream(6, np.float32))
    last_value = np.zeros(1, np.float32)
    batch.finalize(last_value, cfg)
    return batch


def evaluate(net: ActorCritic, ep: ImitationEpisode, *, seeds: tuple = (0,), verbose=True) -> dict:
    """Deterministic rollouts on the reference (+ seeded perturbations)."""
    rows = []
    for s in seeds:
        obs = ep.reset(seed=int(s))
        errs, site_errs, zs = [], [], []
        fell = False
        deviation = None
        for _ in range(ep.N):
            ao = np.asarray(obs["actor"], np.float32)
            with torch.no_grad():
                unit = net.actor.deterministic_unit(torch.from_numpy(ao).unsqueeze(0))[0].numpy()
            obs, _r, done, info = ep.step(unit)
            k = info["k"]
            q = np.asarray(ep.env.data.qpos[7:36], np.float64)
            errs.append(float(np.mean(np.abs(q - ep.tr.qpos[min(k, ep.N), 7:36]))))
            cur = state_from_env(ep.env)
            site_errs.append(float(np.mean(np.abs(cur.site_pos["core"]
                                                 - ep.targets.at(min(k, ep.N)).site_pos["core"]))))
            zs.append(float(ep.env.data.qpos[2]))
            if info.get("deviation") and deviation is None:
                deviation = info["deviation"]
            if done:
                fell = bool(info.get("termination") is not None)
                break
        rows.append({"seed": int(s), "steps": len(errs),
                     "joint_err_mean": round(float(np.mean(errs)), 4) if errs else None,
                     "joint_err_final": round(float(errs[-1]), 4) if errs else None,
                     "site_core_err_mean": round(float(np.mean(site_errs)), 4) if site_errs else None,
                     "pelvis_z_min": round(float(np.min(zs)), 4) if zs else None,
                     "fell": fell, "deviation": deviation})
        if verbose:
            print("  eval seed=%d %s" % (s, json.dumps(rows[-1])))
    return {"rollouts": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--bc", default=DEFAULT_BC)
    ap.add_argument("--out", default="checkpoints/solo/imitation_probe.pt")
    ap.add_argument("--updates", type=int, default=4)
    ap.add_argument("--episodes-per-update", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--minibatches", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--hidden", default="128,128")
    ap.add_argument("--preset", default="site_primary", choices=sorted(PRESETS))
    ap.add_argument("--deviation-penalty", type=float, default=10.0)
    ap.add_argument("--soft-deviation", action="store_true",
                    help="pay the deviation penalty instead of ending the episode")
    ap.add_argument("--init-ckpt", default=None,
                    help="a training checkpoint whose ActorCritic actor initialises "
                         "the policy (e.g. the T1 balance v6a: the residual-on-a-"
                         "balancer init); applied after --bc if both are given")
    ap.add_argument("--perturb-sigma", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seeds", default="0,1,2")
    ap.add_argument("--no-bc", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(int(args.seed))
    model = load_solo_model()
    tr = load_reference(args.reference)
    targets = ImitationTargets.from_reference(args.reference, model=model)
    weights = PRESETS[str(args.preset)]
    print(f"[imitation] reference={args.reference} frames={len(tr)} "
          f"preset={args.preset} bc={'(none)' if args.no_bc else args.bc}")

    hidden = tuple(int(x) for x in str(args.hidden).split(",") if x.strip())
    ckpt = None
    if args.init_ckpt:
        from rl.checkpoint import load_checkpoint

        ckpt = load_checkpoint(args.init_ckpt)
        tc = (ckpt.get("config") or {}).get("train", {})
        if tc.get("hidden"):
            hidden = tuple(int(h) for h in tc["hidden"])
            print(f"[imitation] hidden from the init checkpoint: {hidden}")
    net = ActorCritic(ACTOR_DIM, CRITIC_DIM, act_dim=N_JOINTS,
                      cfg=NetConfig(hidden=hidden, action_mode="residual",
                                    residual_scale=0.5))
    if not args.no_bc:
        status = warm_start_from_bc(net, Path(args.bc))
        print(f"[imitation] warm start: {json.dumps(status)[:200]}")
    if ckpt is not None:
        from rl.checkpoint import apply_checkpoint

        apply_checkpoint(ckpt, policy=net)
        print(f"[imitation] init from {args.init_ckpt} "
              f"(steps={(ckpt.get('state') or {}).get('steps_done')})")

    cfg = PPOConfig(gamma=float(args.gamma), lam=float(args.lam), lr=float(args.lr),
                    minibatches=int(args.minibatches), epochs=int(args.epochs),
                    hidden=hidden, action_mode="residual", residual_scale=0.5)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    gen = torch.Generator(device="cpu").manual_seed(int(args.seed))
    ep = ImitationEpisode(tr, targets, weights, perturb_sigma=float(args.perturb_sigma),
                          model=model, seed=int(args.seed),
                          terminate_on_deviation=not bool(args.soft_deviation),
                          deviation_penalty=float(args.deviation_penalty))

    t0 = time.perf_counter()
    for it in range(1, int(args.updates) + 1):
        batch = collect(net, ep, cfg, seed=int(args.seed) + 1000 * it,
                        n_episodes=int(args.episodes_per_update))
        # the deviation penalty: applied by the episode as a reward subtraction
        stats = ppo_update(net, opt, batch, cfg, cfg.lr, gen)
        ret = float(np.sum(batch.rewards)) / max(1, batch.rewards.shape[1])
        print(f"[imitation] update {it}/{args.updates} steps={batch.num_steps()} "
              f"return={ret:+.2f} policy_loss={stats['policy_loss']:+.4f} "
              f"value_loss={stats['value_loss']:.1f} kl={stats['approx_kl']:.4f} "
              f"wall={time.perf_counter() - t0:.1f}s")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(),
                "config": {"reference": args.reference, "preset": args.preset,
                           "hidden": list(hidden), "action_mode": "residual",
                           "residual_scale": 0.5},
                "state": {"updates": int(args.updates), "seed": int(args.seed)}}, out)
    print(f"[imitation] wrote {out}")
    res = evaluate(net, ep, seeds=tuple(int(s) for s in str(args.eval_seeds).split(",")))
    print("[imitation] eval:", json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
