#!/usr/bin/env python
"""WHERE does gait tracking fail? Deviation onset vs the reference's contact rhythm.

For one gait segment + checkpoint, rerun the deterministic hard-gated episode and
record, per control step: the site RMS, both feet's contact states (robot), the
reference's planted flags, and the reference contact-switch events. Then report:

  * the deviation-onset step (first step with site RMS > 0.10 m) and whether it
    falls WITHIN a reference stance window or AT a reference contact switch
    (+/-3 frames);
  * the robot-vs-reference contact agreement around the onset;
  * the step events the robot actually took (air time >= 0.06 s landings).

This is the falsifier measurement for the gait stage: if failures concentrate at
reference contact switches, the policy cannot produce the swing; if they occur
mid-stance, it is a balance/observation question. Prints its own verification.

Run: MUJOCO_GL=egl .venv/bin/python scripts/track_gait_probe.py \
        --ckpt checkpoints/solo/track_s2_v2refs.pt --source drill_continuous \
        --k0 312 --k1 512 --seed 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solo.scene import N_JOINTS, load_solo_model, stand_frame  # noqa: E402
from solo.track import (REF_ACTOR_DIM, REF_CRITIC_DIM,  # noqa: E402
                        Segment, TrackingEnv, track_targets)

ONSET_SITE = 0.10  # m — the deviation-onset bar (below the 0.15 termination)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", default="drill_continuous")
    ap.add_argument("--k0", type=int, required=True)
    ap.add_argument("--k1", type=int, required=True)
    ap.add_argument("--label", default="SHUFFLE_F")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--residual-scale", type=float, default=0.5)
    args = ap.parse_args(argv)

    import torch

    from rl.net import ActorCritic, NetConfig
    from solo.env import SoloEnv

    model = load_solo_model()
    q_stand = stand_frame(model)[0]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    net = ActorCritic(REF_ACTOR_DIM, REF_CRITIC_DIM, act_dim=N_JOINTS,
                      cfg=NetConfig(hidden=tuple(cfg.get("hidden", (256, 256))),
                                    action_mode="residual",
                                    residual_scale=float(args.residual_scale)))
    net.load_state_dict(ck.get("model") or ck.get("policy"))
    net.eval()

    tt = track_targets(args.source, model)
    seg = Segment(args.source, args.k0, args.k1, args.label, 1)
    base = SoloEnv(model, task="balance", action_mode="residual",
                   residual_scale=float(args.residual_scale), jitter=False)
    ep = TrackingEnv(base, seg, tt, q_stand=q_stand, deviation_mode="hard", seed=args.seed)
    obs = ep.reset(seed=int(args.seed))

    # reference contact switches inside the segment
    ref_switches = []
    for kf in range(seg.k0 + 1, seg.k1):
        if (tt.contact[kf] != tt.contact[kf - 1]).any():
            ref_switches.append(kf - seg.k0)

    rows = []
    steps = 0
    cause = None
    with torch.no_grad():
        while steps < 3000:
            ao = torch.from_numpy(np.asarray(obs["actor"], np.float32)).unsqueeze(0)
            unit = net.actor.deterministic_unit(ao)[0].numpy()
            obs, _r, term, trunc, info = ep.step(unit)
            steps += 1
            got = info["contacts"]
            rows.append({
                "k": ep.k, "site": info["track"]["errs"]["site_err"],
                "joint": info["track"]["errs"]["joint_err"],
                "root_xy": info["track"]["errs"]["root_xy_err"],
                "robot_L": int(bool(got["left_foot"])), "robot_R": int(bool(got["right_foot"])),
                "ref_L": int(tt.contact[min(seg.k0 + ep.k, len(tt) - 1)][0]),
                "ref_R": int(tt.contact[min(seg.k0 + ep.k, len(tt) - 1)][1]),
                "air_L": float(ep.env._foot_air[0]), "air_R": float(ep.env._foot_air[1]),
            })
            if term or trunc:
                cause = info["track"]["cause"]
                break

    site = np.array([r["site"] for r in rows])
    onset = int(np.argmax(site > ONSET_SITE)) if (site > ONSET_SITE).any() else -1
    steps_to_switch = [min((abs(onset - s) for s in ref_switches), default=-1)]
    at_switch = any(abs(onset - s) <= 3 for s in ref_switches) if onset >= 0 else None
    robot_steps = sum(1 for r in rows if (r["air_L"] >= 0.06 or r["air_R"] >= 0.06))
    agree = np.mean([r["robot_L"] == r["ref_L"] and r["robot_R"] == r["ref_R"]
                     for r in rows])
    pre = rows[max(onset - 10, 0):onset + 1] if onset >= 0 else []
    out = {
        "segment": seg.name, "label": seg.label, "seed": args.seed,
        "steps": steps, "cause": cause,
        "ref_contact_switches": ref_switches,
        "deviation_onset_step": onset, "onset_time_s": round(onset * 0.02, 2),
        "onset_at_reference_switch(+/-3f)": at_switch,
        "onset_distance_to_nearest_switch": steps_to_switch[0],
        "switch_period_mean": round(float(np.mean(np.diff(ref_switches))) if len(ref_switches) > 1 else 0, 1),
        "robot_step_events(proxy)": robot_steps,
        "contact_agreement_until_onset": round(float(
            np.mean([r["robot_L"] == r["ref_L"] and r["robot_R"] == r["ref_R"]
                     for r in rows[:max(onset, 1)]])), 3) if onset > 0 else None,
        "site_rms_at_onset": round(float(site[onset]), 3) if onset >= 0 else None,
        "site_rms_10_steps_before_onset": ([round(r["site"], 3) for r in pre]),
        "root_xy_at_onset": round(float(rows[onset]["root_xy"]), 3) if onset >= 0 else None,
    }
    print(json.dumps(out, indent=1))
    # own verification: the onset must be inside the recorded episode
    assert onset < steps, "onset outside the episode"
    if ref_switches:
        print("verification: onset", onset, "switches", ref_switches[:8],
              "at_switch", at_switch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
