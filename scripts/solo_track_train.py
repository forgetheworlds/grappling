#!/usr/bin/env python
"""Stage-curriculum PPO for the reference-conditioned tracking task.

Formulation (solo.track): the actor's residual base action IS the reference's
next-frame joint targets, the actor OBSERVES the reference block (local root
target, reference joints, contacts, phase, skill, lead, connector flag, short
future root), and the reward is tracking + validity with deviation
termination.  No scripted teacher exists in the loop; the reference enters as
a commanded movement target.

Stages (solo.track.STAGE_ORDER): train one stage, gate it (scripts/
solo_track_eval.py), warm-start the next.  Episodes may reset in training;
the final no-reset evaluation is a separate script.

Run (S1, from-scratch):
    MUJOCO_GL=egl .venv/bin/python scripts/solo_track_train.py \
        --stage S1_stand_lower_hold_rise --updates 300 --out checkpoints/solo/track_s1.pt
Warm-start a later stage:
    ... --stage S2_first_step --init-ckpt checkpoints/solo/track_s1.pt ...
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

from rl.net import ActorCritic, NetConfig  # noqa: E402
from rl.ppo import PPOConfig, RolloutBatch, ppo_update  # noqa: E402
from solo.scene import N_JOINTS, load_solo_model  # noqa: E402
from solo.track import (REF_ACTOR_DIM, REF_CRITIC_DIM, STAGE_ORDER,  # noqa: E402
                        TrackingTask, warm_start_actor)

EVAL_SEEDS = (100, 101, 102)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="S1_stand_lower_hold_rise", choices=STAGE_ORDER)
    ap.add_argument("--out", default=None, help="checkpoint path (default by stage)")
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--episodes-per-update", type=int, default=12)
    ap.add_argument("--max-episode-steps", type=int, default=3200)
    ap.add_argument("--init-ckpt", default=None,
                    help="warm-start checkpoint (first-layer surgery if dims differ)")
    ap.add_argument("--hidden", default="256,256")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-critic", type=float, default=5e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=2)
    ap.add_argument("--entropy", type=float, default=0.003)
    ap.add_argument("--grad-clip-actor", type=float, default=0.5)
    ap.add_argument("--grad-clip-critic", type=float, default=0.5)
    ap.add_argument("--log-std-init", type=float, default=-1.0)
    ap.add_argument("--log-std-final", type=float, default=-2.5)
    ap.add_argument("--log-std-anneal-steps", type=int, default=100_000)
    ap.add_argument("--residual-scale", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--target-kl", type=float, default=0.03)
    return ap.parse_args(argv)


def log_std_at(steps_done: int, args) -> float:
    frac = min(1.0, float(steps_done) / max(1, args.log_std_anneal_steps))
    return float(args.log_std_init + frac * (args.log_std_final - args.log_std_init))


def collect(net: ActorCritic, task: TrackingTask, args, seed: int) -> tuple:
    """One PPO batch of episode-streamed rollouts; returns (batch, stats)."""
    chunks = []
    ep_stats = []
    for e in range(args.episodes_per_update):
        obs = task.reset(seed=int(seed) + e)
        seg = task.segment
        a_obs, c_obs, act, logp, val, rew, don = [], [], [], [], [], [], []
        success, cause, steps = False, None, 0
        for _ in range(args.max_episode_steps):
            ao = np.asarray(obs["actor"], np.float32)
            co = np.asarray(obs["critic"], np.float32)
            with torch.no_grad():
                sample = net.actor.sample(torch.from_numpy(ao).unsqueeze(0))
                value = float(net.value(torch.from_numpy(co).unsqueeze(0))[0])
            unit = sample["unit"][0].numpy()
            obs, r, term, trunc, info = task.step(unit)
            a_obs.append(ao)
            c_obs.append(co)
            act.append(unit.astype(np.float32))
            logp.append(float(sample["logp"][0]))
            val.append(value)
            rew.append(float(r))
            steps += 1
            if term or trunc:
                tk = info["track"]
                success, cause = bool(tk["success"]), tk["cause"]
                don.append(True)
                break
            don.append(False)
        if not don:
            don[-1] = True  # safety cut: treat as episode end
            cause = cause or "max_steps"
        ep_stats.append({"segment": seg.name, "label": seg.label, "steps": steps,
                         "success": success, "cause": cause,
                         "return": float(np.sum(rew))})
        chunks.append((a_obs, c_obs, act, logp, val, rew, don))

    def stream(i, dt, extra=()):
        out = []
        for c in chunks:
            out.extend(c[i])
        return np.asarray(out, dtype=dt).reshape((len(out), 1) + tuple(extra))

    cfg = PPOConfig(epochs=args.epochs, minibatches=args.minibatches,
                    gamma=args.gamma, lam=args.lam)
    batch = RolloutBatch(
        actor_obs=stream(0, np.float32, (REF_ACTOR_DIM,)),
        critic_obs=stream(1, np.float32, (REF_CRITIC_DIM,)),
        actions_unit=stream(2, np.float32, (N_JOINTS,)),
        logp=stream(3, np.float32),
        values=stream(4, np.float32),
        rewards=stream(5, np.float32),
        dones=stream(6, np.float32))
    batch.finalize(np.zeros(1, np.float32), cfg)
    return batch, ep_stats


@torch.no_grad()
def quick_eval(net: ActorCritic, task: TrackingTask, seeds=EVAL_SEEDS) -> dict:
    """Deterministic rollouts on a few fixed segments (the stage's own mix)."""
    was = task.ep
    rows = []
    for s in seeds:
        obs = task.reset(seed=int(s))
        seg = task.segment
        R, steps, success, cause = 0.0, 0, False, None
        site, joint = [], []
        for _ in range(3200):
            ao = np.asarray(obs["actor"], np.float32)
            unit = net.actor.deterministic_unit(torch.from_numpy(ao).unsqueeze(0))[0].numpy()
            obs, r, term, trunc, info = task.step(unit)
            R += r
            steps += 1
            site.append(info["track"]["errs"]["site_err"])
            joint.append(info["track"]["errs"]["joint_err"])
            if term or trunc:
                tk = info["track"]
                success, cause = bool(tk["success"]), tk["cause"]
                break
        rows.append({"seed": int(s), "segment": seg.name, "label": seg.label,
                     "steps": steps, "success": success, "cause": cause,
                     "return": round(float(R), 1),
                     "site_err_mean": round(float(np.mean(site)), 4) if site else None,
                     "joint_err_mean": round(float(np.mean(joint)), 4) if joint else None})
    task.ep = was
    return {"rows": rows,
            "success_rate": float(np.mean([r["success"] for r in rows]))}


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(1)
    torch.manual_seed(int(args.seed))
    out = Path(args.out or f"checkpoints/solo/track_{args.stage}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".jsonl")

    model = load_solo_model()
    hidden = tuple(int(x) for x in str(args.hidden).split(",") if x.strip())
    net = ActorCritic(REF_ACTOR_DIM, REF_CRITIC_DIM, act_dim=N_JOINTS,
                      cfg=NetConfig(hidden=hidden, action_mode="residual",
                                    residual_scale=float(args.residual_scale),
                                    log_std_init=args.log_std_init))
    init_note = "scratch"
    if args.init_ckpt:
        from rl.checkpoint import load_checkpoint

        ck = load_checkpoint(args.init_ckpt)
        status = warm_start_actor(net, ck)
        init_note = f"{args.init_ckpt} ({json.dumps(status)[:160]})"
        print(f"[track-train] warm start: {json.dumps(status)}")
    net.actor.set_log_std(args.log_std_init)

    cfg = PPOConfig(epochs=args.epochs, minibatches=args.minibatches,
                    gamma=args.gamma, lam=args.lam, hidden=hidden,
                    action_mode="residual", residual_scale=float(args.residual_scale),
                    entropy_coef=args.entropy, target_kl=args.target_kl,
                    grad_clip_actor=args.grad_clip_actor,
                    grad_clip_critic=args.grad_clip_critic, seed=int(args.seed))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    # split actor/critic learning rates via param-group ratios (ppo_update scales)
    for group in opt.param_groups:
        group["lr_ratio"] = 1.0
    if abs(args.lr_critic - args.lr) > 1e-12:
        # one group per module: actor at --lr, critic at --lr-critic
        opt = torch.optim.Adam([
            {"params": list(net.actor.parameters()), "lr": args.lr, "lr_ratio": 1.0},
            {"params": list(net.critic.parameters()), "lr": args.lr_critic,
             "lr_ratio": args.lr_critic / args.lr},
        ])

    task = TrackingTask(stage=args.stage, model=model, seed=int(args.seed),
                        residual_scale=float(args.residual_scale))
    print(f"[track-train] stage={args.stage} segments={len(task.segs)} "
          f"actor={REF_ACTOR_DIM} critic={REF_CRITIC_DIM} init={init_note}")
    t0 = time.perf_counter()
    steps_done = 0
    best_success = -1.0
    logf = open(log_path, "a")
    for it in range(1, args.updates + 1):
        net.actor.set_log_std(log_std_at(steps_done, args))
        batch, ep_stats = collect(net, task, args, seed=int(args.seed) + 997 * it)
        stats = ppo_update(net, opt, batch, cfg, args.lr,
                           torch.Generator().manual_seed(int(args.seed) + it))
        steps_done += batch.num_steps()
        sr = float(np.mean([e["success"] for e in ep_stats]))
        mean_len = float(np.mean([e["steps"] for e in ep_stats]))
        causes: dict[str, int] = {}
        for e in ep_stats:
            key = (e["cause"] or "success").split(":")[0]
            causes[key] = causes.get(key, 0) + 1
        row = {"update": it, "steps": steps_done, "success_rate": round(sr, 3),
               "mean_len": round(mean_len, 1), "causes": causes,
               "return_mean": round(float(np.mean(batch.rewards)) * batch.num_steps(), 1),
               "policy_loss": round(stats["policy_loss"], 4),
               "value_loss": round(stats["value_loss"], 1),
               "entropy": round(stats["entropy"], 3),
               "kl": round(stats["approx_kl"], 4),
               "sigma": round(stats["sigma_mean"], 3),
               "wall_s": round(time.perf_counter() - t0, 1)}
        print(f"[track-train] {json.dumps(row)}")
        logf.write(json.dumps(row) + "\n")
        logf.flush()
        if args.eval_every and it % args.eval_every == 0:
            ev = quick_eval(net, task)
            print(f"[track-train] eval@{it} success={ev['success_rate']:.2f} "
                  + " ".join(f"{r['label']}:{'OK' if r['success'] else r['cause']}"
                             for r in ev["rows"]))
            logf.write(json.dumps({"update": it, "eval": ev}) + "\n")
            logf.flush()
            if ev["success_rate"] > best_success:
                best_success = ev["success_rate"]
                torch.save({"model": net.state_dict(),
                            "config": {"stage": args.stage, "hidden": list(hidden),
                                       "action_mode": "residual",
                                       "residual_scale": args.residual_scale,
                                       "ref_actor_dim": REF_ACTOR_DIM,
                                       "ref_critic_dim": REF_CRITIC_DIM,
                                       "init": init_note, "weights": "default"},
                            "state": {"updates": it, "steps": steps_done,
                                      "seed": args.seed,
                                      "eval_success": best_success}}, out)
        if args.save_every and it % args.save_every == 0:
            torch.save({"model": net.state_dict(),
                        "config": {"stage": args.stage, "hidden": list(hidden),
                                   "action_mode": "residual",
                                   "residual_scale": args.residual_scale,
                                   "ref_actor_dim": REF_ACTOR_DIM,
                                   "ref_critic_dim": REF_CRITIC_DIM,
                                   "init": init_note, "weights": "default"},
                        "state": {"updates": it, "steps": steps_done,
                                  "seed": args.seed}}, str(out) + f".it{it}")
    # final save + eval
    torch.save({"model": net.state_dict(),
                "config": {"stage": args.stage, "hidden": list(hidden),
                           "action_mode": "residual",
                           "residual_scale": args.residual_scale,
                           "ref_actor_dim": REF_ACTOR_DIM,
                           "ref_critic_dim": REF_CRITIC_DIM,
                           "init": init_note, "weights": "default"},
                "state": {"updates": args.updates, "steps": steps_done,
                          "seed": args.seed}}, out)
    ev = quick_eval(net, task)
    print(f"[track-train] FINAL eval success={ev['success_rate']:.2f}")
    for r in ev["rows"]:
        print("   ", json.dumps(r))
    logf.write(json.dumps({"update": args.updates, "eval": ev, "final": True}) + "\n")
    logf.close()
    print(f"[track-train] wrote {out} (log {log_path}); "
          f"{steps_done} steps in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
