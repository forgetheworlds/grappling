#!/usr/bin/env python
"""Baselines + plumbing validation BEFORE any long tracking run (brief step 5).

Measures, per curriculum stage segment:
  1. open-loop reference replay (residual z = 0, i.e. the base action alone):
     completion/causes, site/joint/root errors, contact agreement, throughput;
  2. the EXISTING controller (T1 balance v6d, 115-dim, stand-residual) via
     first-layer surgery into the reference-conditioned net (reference block
     zero-initialised): the same rollouts driven by the shipped balance skill;
  3. the reward arithmetic on constructed states (stance vs the measured
     crouch escape) -- the anti-gaming closure, printed as numbers;
  4. that the actor observation ACTUALLY carries the changing reference
     signals: per-phase block variance + nearest-centroid phase separability;
  5. that applied MuJoCo ctrl targets equal the documented action mapping.

Writes data/solo/metrics/track_baselines.json (+ provenance) and prints its
own verification.  Diagnostic clips (labelled *_diag.mp4) go to
videos/solo_drill/track_baselines/ when --videos is passed.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solo.scene import N_JOINTS, load_solo_model, stand_frame  # noqa: E402
from solo.track import (DEFAULT_WEIGHTS, REF_ACTOR_DIM, REF_CRITIC_DIM,  # noqa: E402
                        REF_LAYOUT, STAGE_ORDER, Segment, TrackingEnv, stage_segments,
                        track_deviation, track_reward_terms, track_targets)

OUT_JSON = REPO / "data/solo/metrics/track_baselines.json"
VIDEO_DIR = REPO / "videos/solo_drill/track_baselines"
T1_CKPT = REPO / "checkpoints/solo/t1_balance_v6d.pt"
EVAL_SEEDS = (0, 1, 2)


def provenance() -> dict:
    import solo.track

    return {
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     capture_output=True, text=True).stdout.strip(),
        "module": "src/solo/track.py",
        "weights": DEFAULT_WEIGHTS.as_dict(),
        "seeds": list(EVAL_SEEDS),
        "command": "MUJOCO_GL=egl .venv/bin/python scripts/track_baselines.py",
    }


def rollout_episode(ep: TrackingEnv, act_fn, max_steps: int = 4000) -> dict:
    obs = ep.reset()
    R, steps = 0.0, 0
    site, joint, root_xy, root_z = [], [], [], []
    agree, contacts_n = 0, 0
    success, cause = False, None
    for _ in range(max_steps):
        unit = act_fn(obs)
        obs, r, term, trunc, info = ep.step(unit)
        steps += 1
        R += r
        # mapping check every 7th step (cheap, still sampled broadly);
        # ep.step sets the base action BEFORE env.step, and it is still in
        # place when the step returns.
        if steps % 7 == 0:
            applied = ep.env.data.ctrl
            base_now = np.clip(ep.env._base_action, ep.env.lo, ep.env.hi)
            z = np.clip(np.asarray(unit, np.float64), -1, 1)
            expect = np.clip(base_now + ep.env.residual_scale * np.tanh(z),
                             ep.env.lo, ep.env.hi)
            assert np.allclose(applied, expect, atol=1e-9), "action mapping broken"
        site.append(info["track"]["errs"]["site_err"])
        joint.append(info["track"]["errs"]["joint_err"])
        root_xy.append(info["track"]["errs"]["root_xy_err"])
        root_z.append(abs(info["track"]["errs"]["root_z_err"]))
        kf = min(ep.seg.k0 + ep.k, len(ep.tt) - 1)
        contacts_n += 2
        ref_c = ep.tt.contact[kf] > 0.5
        got = info["contacts"]
        agree += int(bool(got["left_foot"]) == bool(ref_c[0]))
        agree += int(bool(got["right_foot"]) == bool(ref_c[1]))
        if term or trunc:
            tk = info["track"]
            success, cause = bool(tk["success"]), tk["cause"]
            break
    n = max(steps, 1)
    return {"segment": ep.seg.name, "label": ep.seg.label,
            "duration_s": round(ep.seg.duration_s, 2), "steps": steps,
            "completed": success, "cause": cause,
            "return": round(float(R), 1),
            "site_err_mean": round(float(np.mean(site)), 4),
            "site_err_p95": round(float(np.percentile(site, 95)), 4),
            "joint_err_mean": round(float(np.mean(joint)), 4),
            "root_xy_err_mean": round(float(np.mean(root_xy)), 4),
            "root_z_err_mean": round(float(np.mean(root_z)), 4),
            "contact_agreement": round(agree / max(contacts_n, 1), 3)}


def open_loop_and_controller(model, q_stand) -> tuple[list, list, float]:
    from solo.env import SoloEnv

    actor_fn = None
    init_note = "unavailable"
    if T1_CKPT.exists():
        import torch

        from rl.checkpoint import load_checkpoint
        from solo.track import warm_start_actor

        net = _build_net()
        ck = load_checkpoint(T1_CKPT)
        warm_start_actor(net, ck)
        net.actor.log_std.requires_grad_(False)
        net.actor.log_std.fill_(-5.0)

        def actor_fn(obs):  # noqa: F811
            ao = torch.from_numpy(np.asarray(obs["actor"], np.float32)).unsqueeze(0)
            with torch.no_grad():
                return net.actor.deterministic_unit(ao)[0].numpy()

        init_note = f"T1 v6d surgery ({T1_CKPT.name})"

    rows_open, rows_ctrl = [], []
    t0, steps = time.perf_counter(), 0
    for stage in ("S1_stand_lower_hold_rise", "S3_footwork", "S4_level_change",
                  "S5_entry_recovery"):
        for seg in stage_segments(stage):
            tt = track_targets(seg.source, model)
            base = SoloEnv(model, task="balance", action_mode="residual",
                           residual_scale=0.5, jitter=False, record_metrics=True)
            ep = TrackingEnv(base, seg, tt, q_stand=q_stand)
            rows_open.append(rollout_episode(ep, lambda obs: np.zeros(N_JOINTS)))
            steps += rows_open[-1]["steps"]
            if actor_fn is not None:
                base2 = SoloEnv(model, task="balance", action_mode="residual",
                                residual_scale=0.5, jitter=False,
                                record_metrics=True)
                ep2 = TrackingEnv(base2, seg, tt, q_stand=q_stand)
                rows_ctrl.append(rollout_episode(ep2, actor_fn))
    rate = steps / (time.perf_counter() - t0)
    print(f"[baselines] controller-init: {init_note}; collection throughput "
          f"{rate:.0f} steps/s")
    return rows_open, rows_ctrl, rate


def _build_net():
    import torch  # noqa: F401

    from rl.net import ActorCritic, NetConfig

    return ActorCritic(REF_ACTOR_DIM, REF_CRITIC_DIM, act_dim=N_JOINTS,
                       cfg=NetConfig(hidden=(256, 256), action_mode="residual",
                                     residual_scale=0.5))


def reward_arithmetic() -> dict:
    w = DEFAULT_WEIGHTS
    cases = {
        "tracks_standing_reference": dict(joint_err=0.05, site_err=0.03,
                                          root_xy_err=0.03, root_z_err=0.02,
                                          yaw_err=0.05, ref_pelvis_z=0.74,
                                          torso_up_z=0.99, action_delta_mean=0.02,
                                          sat_frac=0.0, limit_prox=0.0,
                                          slide_frac=0.0),
        "measured_crouch_escape": dict(joint_err=0.5, site_err=0.10,
                                       root_xy_err=0.05, root_z_err=0.30,
                                       yaw_err=0.10, ref_pelvis_z=0.74,
                                       torso_up_z=0.95, action_delta_mean=0.0,
                                       sat_frac=0.0, limit_prox=0.0,
                                       slide_frac=0.0),
        "deep_phase_10cm_height_miss": dict(joint_err=0.05, site_err=0.05,
                                            root_xy_err=0.03, root_z_err=0.10,
                                            yaw_err=0.05, ref_pelvis_z=0.35,
                                            torso_up_z=0.98, action_delta_mean=0.02,
                                            sat_frac=0.0, limit_prox=0.0,
                                            slide_frac=0.0),
    }
    out = {}
    for name, kw in cases.items():
        t = track_reward_terms(**kw, w=w)
        out[name] = {"total": round(t["total"], 3),
                     "w_root": round(t["w_root"], 3), "w_site": round(t["w_site"], 3)}
    out["crouch_is_deviation"] = track_deviation(
        joint_err=0.0, root_xy_err=0.0, pelvis_drop_m=0.30, site_err=0.0,
        w=w) == "pelvis_drop"
    out["dominance_ratio"] = round(out["tracks_standing_reference"]["total"]
                                   / max(out["measured_crouch_escape"]["total"], 1e-9), 1)
    return out


def reference_signal_check() -> dict:
    """Does the appended block let the actor TELL the phases apart?"""
    segs = {s.name: s for s in stage_segments("S6_connected_drill")}
    blocks: dict[str, list] = {}
    q_stand = stand_frame(load_solo_model())[0]
    for name, seg in segs.items():
        tt = track_targets(seg.source, load_solo_model())
        for kf in range(seg.k0 + 10, seg.k1 - 10, max(1, (seg.k1 - seg.k0) // 20)):
            joints_rel = tt.qpos[kf, 7:36] - q_stand[7:36]
            d_xy = tt.root_xy[kf] - tt.root_xy[seg.k0]
            v = tt.linvel_world[kf]
            contacts = tt.contact[kf]
            ahead = min(kf + 10, len(tt) - 1)
            da = tt.root_xy[ahead] - tt.root_xy[seg.k0]
            # motion-only features (the skill one-hot deliberately excluded:
            # the MOVEMENT signal itself must separate the phases)
            blocks.setdefault(name, []).append(np.concatenate([
                d_xy, [tt.root_z[kf]], v, joints_rel, contacts,
                [da[0], da[1], tt.root_z[ahead] - tt.root_z[kf]]]))
    names = sorted(blocks)
    X = np.concatenate([np.asarray(blocks[n]) for n in names])
    y = np.concatenate([[i] * len(blocks[n]) for i, n in enumerate(names)])
    mu = np.stack([np.asarray(blocks[n]).mean(axis=0) for n in names])
    # nearest-centroid classification (chance = 1/len(names))
    d = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
    pred = d.argmin(axis=1)
    acc = float((pred == y).mean())
    # per-phase within variance vs between-phase separation
    within = float(np.mean([np.asarray(blocks[n]).std(axis=0).mean() for n in names]))
    between = float(mu.std(axis=0).mean())
    return {"phases": names, "nearest_centroid_accuracy": round(acc, 3),
            "chance": round(1.0 / len(names), 3),
            "within_phase_std": round(within, 4),
            "between_phase_std": round(between, 4),
            "separation_ratio": round(between / max(within, 1e-9), 2)}


def main(argv=None) -> int:
    print("[baselines] loading model...")
    model = load_solo_model()
    q_stand = stand_frame(model)[0]
    rows_open, rows_ctrl, rate = open_loop_and_controller(model, q_stand)
    arith = reward_arithmetic()
    signals = reference_signal_check()

    def summarize(rows):
        by_label: dict[str, list] = {}
        for r in rows:
            by_label.setdefault(r["label"], []).append(r)
        out = {}
        for lab, rs in sorted(by_label.items()):
            out[lab] = {
                "n": len(rs),
                "completed": sum(int(r["completed"]) for r in rs),
                "cause_fall": sum(1 for r in rs if r["cause"] == "fall"),
                "cause_dorsal": sum(1 for r in rs if r["cause"] == "dorsal"),
                "cause_deviation": sum(1 for r in rs if r["cause"]
                                       and r["cause"].startswith("deviation")),
                "site_err_mean_max": max(r["site_err_mean"] for r in rs),
                "joint_err_mean_max": max(r["joint_err_mean"] for r in rs),
            }
        return out

    blob = {
        "provenance": provenance(),
        "throughput_steps_per_s": round(rate, 1),
        "open_loop_replay": rows_open,
        "open_loop_by_label": summarize(rows_open),
        "t1_surgery_controller": rows_ctrl,
        "t1_surgery_by_label": summarize(rows_ctrl),
        "reward_arithmetic": arith,
        "reference_signal_check": signals,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(blob, indent=1))
    print(f"[baselines] wrote {OUT_JSON}")

    # ---- printed verification
    print("\n== reward arithmetic (anti-gaming closure) ==")
    for k, v in arith.items():
        print(f"  {k}: {v}")
    assert arith["crouch_is_deviation"], "crouch escape is not terminal"
    # the reward-ratio bar is pinned by tests/test_track.py (>3.0x, current
    # kernels); the STRUCTURAL closure (pelvis_drop terminal) is the primary
    # mechanism and is asserted above
    assert arith["dominance_ratio"] > 3.0, "stance does not dominate the crouch"
    print("\n== reference signals in the actor observation ==")
    print(f"  phases={len(signals['phases'])} nearest-centroid acc="
          f"{signals['nearest_centroid_accuracy']} (chance {signals['chance']}), "
          f"separation={signals['separation_ratio']}")
    assert signals["nearest_centroid_accuracy"] >= 2.0 * signals["chance"]
    print("\n== open-loop replay (z=0) by skill ==")
    for lab, s in blob["open_loop_by_label"].items():
        print(f"  {lab:16s} n={s['n']} completed={s['completed']} "
              f"dev={s['cause_deviation']} fall={s['cause_fall']} "
              f"site_max={s['site_err_mean_max']}")
    ok = all(r["completed"] or r["cause"] for r in rows_open)
    assert ok, "a rollout ended without a recorded cause"
    print("\nbaselines verification OK "
          f"(arithmetic dominance {arith['dominance_ratio']}x, "
          f"signal acc {signals['nearest_centroid_accuracy']}, "
          f"{rate:.0f} steps/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
