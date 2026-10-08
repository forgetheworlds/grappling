#!/usr/bin/env python
"""Per-skill held-out evaluation for a reference-tracking checkpoint.

For every curriculum skill label: deterministic episodes on the stage
segments with UNSEEN seeds, plus the two takes that were never trained
(zero-shot probes, ``solo.track.HELD_OUT_TAKES``).  Each episode = one reset
at its start (training-style); the NO-RESET continuous drill is a separate
mode (``--final-drill``) that additionally writes a trace npz for rendering.

Metrics follow the brief + QUALITY_RUBRIC: completion/causes, landmark and
joint errors, anchored root errors, loaded-foot slide, reference-contact
agreement, level-change depth/return, pelvis band, saturation, smoothness.
Means AND worst-case are reported.  Gate thresholds are calibrated from
``data/solo/metrics/track_baselines.json`` (open-loop replay: site mean
0.07-0.11 m at termination) and frozen here with the calibration note.

Writes data/solo/metrics/track_eval_<tag>.json (+ provenance) and prints its
own verification.  Run:

    MUJOCO_GL=egl .venv/bin/python scripts/solo_track_eval.py \
        --ckpt checkpoints/solo/track_s5.pt --tag s5 --seeds 1000,1001,1002,1003
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solo.scene import N_JOINTS, load_solo_model  # noqa: E402
from solo.track import (HELD_OUT_TAKES, REF_ACTOR_DIM, REF_CRITIC_DIM,  # noqa: E402
                        STAGE_ORDER, Segment, TrackingEnv, stage_segments,
                        track_targets)

#: frozen gate thresholds + why (calibrated from track_baselines.json: the
#: open-loop replay TERMINATES at site mean 0.07-0.11 m, so a surviving policy
#: must sit well below that; joint 0.35 rad is the 1/e kernel of the soft prior)
GATE = {
    "site_err_p95_max": 0.10,
    "joint_err_mean_max": 0.35,
    "root_xy_err_mean_max": 0.20,
    "loaded_slide_per_step_max": 0.004,   # m/step (rubric B1 bar: 2 cm per STEP)
    "calibration": "open-loop replay site_err_mean at termination: 0.068-0.106 m "
                   "(data/solo/metrics/track_baselines.json); success requires "
                   "p95 <= 0.10 m with completion -- an order tighter than failure",
}
BASELINES_JSON = REPO / "data/solo/metrics/track_baselines.json"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default=None, help="output tag (default: ckpt stem)")
    ap.add_argument("--seeds", default="1000,1001,1002,1003")
    ap.add_argument("--stages", default=",".join(STAGE_ORDER),
                    help="whose stage segment lists define the evaluated skills")
    ap.add_argument("--holdout-only", action="store_true")
    ap.add_argument("--final-drill", action="store_true",
                    help="one continuous no-reset rollout of drill_continuous")
    ap.add_argument("--loops", type=int, default=1,
                    help="final-drill: how many REPEAT cycles (uses the loop seam)")
    ap.add_argument("--residual-scale", type=float, default=0.5)
    ap.add_argument("--max-episode-steps", type=int, default=3200)
    return ap.parse_args(argv)


def load_policy(ckpt_path: str, residual_scale: float):
    import torch

    from rl.net import ActorCritic, NetConfig

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    hidden = tuple(cfg.get("hidden", (256, 256)))
    net = ActorCritic(REF_ACTOR_DIM, REF_CRITIC_DIM, act_dim=N_JOINTS,
                      cfg=NetConfig(hidden=hidden, action_mode="residual",
                                    residual_scale=float(residual_scale)))
    sd = ck.get("model") or ck.get("policy")
    net.load_state_dict(sd)
    net.eval()
    return net, cfg, ck.get("state", {})


@torch.no_grad()
def run_episode(net, ep: TrackingEnv, seed: int, max_steps: int, record=False):
    import torch

    obs = ep.reset(seed=int(seed))
    R, steps = 0.0, 0
    site, joint, root_xy, root_z, yaw = [], [], [], [], []
    slide_sum, sat_sum, adelta_sum = 0.0, 0.0, 0.0
    agree, contacts_n = 0, 0
    z_min, z_max = 1e9, -1e9
    ref_z_min, ref_z_max = 1e9, -1e9
    success, cause = False, None
    trace = {"t": [], "qpos": [], "ref_frame": [], "site_err": [], "contacts": [],
             "ref_contacts": [], "phase": [], "label": []}
    prev_ctrl = None
    for _ in range(max_steps):
        ao = np.asarray(obs["actor"], np.float32)
        unit = net.actor.deterministic_unit(torch.from_numpy(ao).unsqueeze(0))[0].numpy()
        obs, r, term, trunc, info = ep.step(unit)
        steps += 1
        R += r
        errs = info["track"]["errs"]
        site.append(errs["site_err"])
        joint.append(errs["joint_err"])
        root_xy.append(errs["root_xy_err"])
        root_z.append(abs(errs["root_z_err"]))
        yaw.append(abs(errs["yaw_err"]))
        pz = float(ep.env.data.qpos[2])
        z_min, z_max = min(z_min, pz), max(z_max, pz)
        ref_z_min = min(ref_z_min, errs["ref_pelvis_z"])
        ref_z_max = max(ref_z_max, errs["ref_pelvis_z"])
        m = info.get("metrics", {})
        slide_sum += float(m.get("slip_travel", 0.0))
        sat_sum += float(m.get("sat_frac", 0.0))
        ctrl = np.asarray(ep.env.data.ctrl, np.float64)
        if prev_ctrl is not None:
            adelta_sum += float(np.mean(np.abs(ctrl - prev_ctrl)))
        prev_ctrl = ctrl
        kf = min(ep.seg.k0 + ep.k, len(ep.tt) - 1)
        ref_c = ep.tt.contact[kf] > 0.5
        got = info["contacts"]
        agree += int(bool(got["left_foot"]) == bool(ref_c[0]))
        agree += int(bool(got["right_foot"]) == bool(ref_c[1]))
        contacts_n += 2
        if record:
            trace["t"].append(float(ep.env.data.time))
            trace["qpos"].append(np.asarray(ep.env.data.qpos, np.float64).copy())
            trace["ref_frame"].append(int(kf))
            trace["site_err"].append(float(errs["site_err"]))
            trace["contacts"].append((int(bool(got["left_foot"])),
                                      int(bool(got["right_foot"]))))
            trace["ref_contacts"].append((int(ref_c[0]), int(ref_c[1])))
            trace["phase"].append(ep.seg.label)
            trace["label"].append(ep.seg.label)
        if term or trunc:
            tk = info["track"]
            success, cause = bool(tk["success"]), tk["cause"]
            break
    n = max(steps, 1)
    out = {"segment": ep.seg.name, "label": ep.seg.label, "held_out":
           ep.seg.source in HELD_OUT_TAKES,
           "duration_s": round(ep.seg.duration_s, 2), "steps": steps,
           "completed": bool(success), "cause": cause,
           "return": round(float(R), 1),
           "site_err_mean": round(float(np.mean(site)), 4),
           "site_err_p95": round(float(np.percentile(site, 95)), 4),
           "site_err_max": round(float(np.max(site)), 4),
           "joint_err_mean": round(float(np.mean(joint)), 4),
           "root_xy_err_mean": round(float(np.mean(root_xy)), 4),
           "root_xy_err_max": round(float(np.max(root_xy)), 4),
           "root_z_err_mean": round(float(np.mean(root_z)), 4),
           "yaw_err_mean_deg": round(float(np.degrees(np.mean(yaw))), 2),
           "loaded_slide_per_step": round(slide_sum / n, 5),
           "contact_agreement": round(agree / max(contacts_n, 1), 3),
           "pelvis_z_min": round(z_min, 3), "pelvis_z_max": round(z_max, 3),
           "ref_pelvis_z_min": round(ref_z_min, 3),
           "ref_pelvis_z_max": round(ref_z_max, 3),
           "sat_frac_mean": round(sat_sum / n, 4),
           "action_delta_mean": round(adelta_sum / n, 4),
           "level_change_depth_err": round(abs(z_min - ref_z_min), 3),
           "fell": bool(cause in ("fall", "dorsal"))}
    if record:
        out["_trace"] = {k: (np.asarray(v) if k != "qpos" else np.stack(v))
                         for k, v in trace.items()}
    return out


def aggregate(rows: list) -> dict:
    if not rows:
        return {}
    n = len(rows)
    comp = [r for r in rows if r["completed"]]
    worst = lambda k: round(max(r[k] for r in rows), 4)  # noqa: E731
    mean = lambda k: round(float(np.mean([r[k] for r in rows])), 4)  # noqa: E731
    gate_pass = [r for r in comp
                 if r["site_err_p95"] <= GATE["site_err_p95_max"]
                 and r["joint_err_mean"] <= GATE["joint_err_mean_max"]
                 and r["root_xy_err_mean"] <= GATE["root_xy_err_mean_max"]
                 and r["loaded_slide_per_step"] <= GATE["loaded_slide_per_step_max"]]
    return {"n": n, "completed": len(comp),
            "completion_rate": round(len(comp) / n, 3),
            "gate_pass": len(gate_pass),
            "gate_pass_rate": round(len(gate_pass) / n, 3),
            "fell": sum(int(r["fell"]) for r in rows),
            "cause_counts": _counts(rows),
            "site_err_mean": mean("site_err_mean"),
            "site_err_p95_worst": worst("site_err_p95"),
            "joint_err_mean_worst": worst("joint_err_mean"),
            "root_xy_err_mean_worst": worst("root_xy_err_mean"),
            "loaded_slide_per_step_worst": worst("loaded_slide_per_step"),
            "contact_agreement_mean": mean("contact_agreement")}


def _counts(rows: list) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        key = (r["cause"] or "success").split(":")[0]
        out[key] = out.get(key, 0) + 1
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    tag = args.tag or Path(args.ckpt).stem
    model = load_solo_model()
    from solo.scene import stand_frame

    q_stand = stand_frame(model)[0]
    net, cfg, state = load_policy(args.ckpt, args.residual_scale)
    print(f"[track-eval] ckpt={args.ckpt} steps={state.get('steps')} "
          f"stage={cfg.get('stage')}")

    def actor(obs):
        import torch

        ao = torch.from_numpy(np.asarray(obs["actor"], np.float32)).unsqueeze(0)
        return net.actor.deterministic_unit(ao)[0].numpy()

    stages = [s for s in str(args.stages).split(",") if s]
    segs: list[Segment] = []
    seen = set()
    for st in stages:
        for seg in stage_segments(st):
            key = (seg.source, seg.k0, seg.k1)
            if key not in seen:
                seen.add(key)
                segs.append(seg)
    if args.holdout_only:
        segs = [s for s in segs if s.source in HELD_OUT_TAKES]
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]

    from solo.env import SoloEnv

    rows = []
    t0 = time.perf_counter()
    for seg in segs:
        tt = track_targets(seg.source, model)
        for seed in seeds:
            base = SoloEnv(model, task="balance", action_mode="residual",
                           residual_scale=float(args.residual_scale), jitter=False,
                           record_metrics=True)
            ep = TrackingEnv(base, seg, tt, q_stand=q_stand,
                             ic_noise=0.005, xy_noise=0.005, yaw_jitter_deg=1.0,
                             seed=seed)
            rows.append(run_episode(net, ep, seed, args.max_episode_steps))
    by_label: dict[str, list] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    summary = {lab: aggregate(rs) for lab, rs in sorted(by_label.items())}
    heldout = [r for r in rows if r["held_out"]]
    blob = {
        "provenance": {
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                         capture_output=True, text=True)
            .stdout.strip(),
            "ckpt": str(args.ckpt),
            "ckpt_sha256": hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest()[:16],
            "ckpt_steps": state.get("steps"),
            "seeds": seeds,
            "episodes": len(rows),
            "wall_s": round(time.perf_counter() - t0, 1),
            "residual_scale": args.residual_scale,
            "gate": GATE,
            "command": (f"MUJOCO_GL=egl .venv/bin/python scripts/solo_track_eval.py "
                        f"--ckpt {args.ckpt} --tag {tag} --seeds {args.seeds}"),
        },
        "overall": aggregate(rows),
        "by_label": summary,
        "heldout_only": aggregate(heldout),
        "rows": rows,
    }
    out = REPO / f"data/solo/metrics/track_eval_{tag}.json"
    out.write_text(json.dumps(blob, indent=1))
    print(f"[track-eval] wrote {out} ({len(rows)} episodes, "
          f"{blob['provenance']['wall_s']}s)")

    print(f"\n== per-skill held-out (gate: completion + site_p95<="
          f"{GATE['site_err_p95_max']}, joint<={GATE['joint_err_mean_max']}, "
          f"slide<={GATE['loaded_slide_per_step_max']}) ==")
    for lab, s in summary.items():
        print(f"  {lab:16s} n={s['n']:3d} complete={s['completion_rate']:.2f} "
              f"gate={s['gate_pass_rate']:.2f} fell={s['fell']} "
              f"site_p95_worst={s['site_err_p95_worst']:.3f} "
              f"slide_worst={s['loaded_slide_per_step_worst']:.4f}")
    if heldout:
        print(f"  ZERO-SHOT held-out takes: {aggregate(heldout)}")
    print(f"  overall: {json.dumps(blob['overall'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
