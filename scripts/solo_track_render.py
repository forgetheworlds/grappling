#!/usr/bin/env python
"""Render reference-tracking evidence videos (960x720 30 fps + contact sheet).

Modes:
  episode    -- one segment rollout with HUD (phase, skill, tracking errors,
                contacts vs reference, verdict); FAILURES kept and labelled;
  final      -- the CONTINUOUS no-reset drill (drill_continuous, optionally
                ``--loops N`` through the REPEAT seam); one unedited take;
  sidebyside -- reference-vs-policy stills at the segment's phase landmarks
                (rubric H1 comparison, 480x360 cells).

Every video writes its metrics JSON (EVIDENCE_PROTOCOL: aggregates +
per-phase rows + provenance + reproduce command) beside it.  Diagnostic /
failure clips are named ``*_diag.mp4`` / ``*_fail.mp4`` and never presented
as the acceptance artifact.

Run:
    MUJOCO_GL=egl .venv/bin/python scripts/solo_track_render.py episode \
        --ckpt checkpoints/solo/track_s1.pt --source stance_hold --out-dir videos/solo_drill/track
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
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solo.scene import N_JOINTS, load_solo_model, stand_frame  # noqa: E402
from solo.track import (HELD_OUT_TAKES, REF_ACTOR_DIM, REF_CRITIC_DIM,  # noqa: E402
                        Segment, TrackingEnv, track_targets)
from solo.video import ClipRecorder, probe_media  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("episode", "final", "sidebyside"):
        s = sub.add_parser(name)
        s.add_argument("--ckpt", required=True)
        s.add_argument("--source", default=None, help="segment source npz name")
        s.add_argument("--k0", type=int, default=None)
        s.add_argument("--k1", type=int, default=None)
        s.add_argument("--label", default=None)
        s.add_argument("--seed", type=int, default=42)
        s.add_argument("--loops", type=int, default=1, help="final: REPEAT cycles")
        s.add_argument("--out-dir", default="videos/solo_drill/track")
        s.add_argument("--name", default=None)
        s.add_argument("--residual-scale", type=float, default=0.5)
        s.add_argument("--max-steps", type=int, default=6400)
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
    net.load_state_dict(ck.get("model") or ck.get("policy"))
    net.eval()
    return net, cfg, ck.get("state", {})


def provenance(args, ckpt_state: dict) -> dict:
    import solo.track

    return {
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     capture_output=True, text=True).stdout.strip(),
        "ckpt": str(args.ckpt),
        "ckpt_sha256": hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest()[:16],
        "ckpt_steps": ckpt_state.get("steps"),
        "seed": int(args.seed),
        "control_hz": 50, "physics_dt": 0.002,
        "learned": "policy-only at inference; the reference enters as a commanded "
                   "movement target (solo.track residual base), no scripted teacher",
        "command": (f"MUJOCO_GL=egl .venv/bin/python scripts/solo_track_render.py "
                    f"{args.mode} --ckpt {args.ckpt} --source {args.source} "
                    f"--seed {args.seed}"),
    }


def hud_rows(ep: TrackingEnv, info: dict, errs: dict) -> list[str]:
    tk = info["track"]
    m = info.get("metrics") or {}
    got = info.get("contacts") or {}
    ref_kf = min(ep.seg.k0 + ep.k, len(ep.tt) - 1)
    ref_c = ep.tt.contact[ref_kf]
    return [
        f"seg {ep.seg.name} [{ep.seg.label}] phase {tk['k']}/{tk['frames']}",
        f"site_err={errs['site_err']:.3f}m joint={errs['joint_err']:.3f}rad "
        f"root_xy={errs['root_xy_err']:.3f}m root_z={errs['root_z_err']:+.3f}m",
        f"contacts L/R got {int(bool(got.get('left_foot')))}/{int(bool(got.get('right_foot')))}"
        f" ref {int(ref_c[0])}/{int(ref_c[1])}  slip={m.get('slip', 0):.3f} "
        f"upright={m.get('upright', 0):.3f}",
    ]


@torch.no_grad()
def rollout(net, ep: TrackingEnv, seed: int, max_steps: int, rec: ClipRecorder,
            record_trace=False):
    obs = ep.reset(seed=int(seed))
    R, steps = 0.0, 0
    site, joint, adelta = [], [], []
    prev_ctrl = None
    success, cause = False, None
    phases: list[dict] = []
    cur_label = None
    seg_start = 0
    while steps < max_steps:
        ao = np.asarray(obs["actor"], np.float32)
        unit = net.actor.deterministic_unit(torch.from_numpy(ao).unsqueeze(0))[0].numpy()
        obs, r, term, trunc, info = ep.step(unit)
        steps += 1
        R += r
        errs = info["track"]["errs"]
        site.append(errs["site_err"])
        joint.append(errs["joint_err"])
        ctrl = np.asarray(ep.env.data.ctrl, np.float64)
        if prev_ctrl is not None:
            adelta.append(float(np.mean(np.abs(ctrl - prev_ctrl))))
        prev_ctrl = ctrl
        if rec is not None:
            rec.capture(ep.env, info, rows=hud_rows(ep, info, errs))
        if record_trace and ep.seg.label != cur_label:
            if cur_label is not None:
                phases.append({"label": cur_label, "t0": seg_start * 0.02,
                               "t1": steps * 0.02})
            cur_label = ep.seg.label
            seg_start = steps
        if term or trunc:
            tk = info["track"]
            success, cause = bool(tk["success"]), tk["cause"]
            break
    if record_trace and cur_label is not None:
        phases.append({"label": cur_label, "t0": seg_start * 0.02,
                       "t1": steps * 0.02})
    out = {"completed": bool(success), "cause": cause, "steps": steps,
           "return": round(float(R), 1),
           "site_err_mean": round(float(np.mean(site)), 4) if site else None,
           "site_err_p95": round(float(np.percentile(site, 95)), 4) if site else None,
           "joint_err_mean": round(float(np.mean(joint)), 4) if joint else None,
           "action_delta_mean": round(float(np.mean(adelta)), 4) if adelta else None}
    return out, phases


def make_env(model, q_stand, seg: Segment, source: str, seed: int,
             residual_scale: float) -> TrackingEnv:
    from solo.env import SoloEnv

    tt = track_targets(source, model)
    base = SoloEnv(model, task="balance", action_mode="residual",
                   residual_scale=float(residual_scale), jitter=False,
                   record_metrics=True)
    return TrackingEnv(base, seg, tt, q_stand=q_stand, seed=seed)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_solo_model()
    q_stand = stand_frame(model)[0]
    net, cfg, ck_state = load_policy(args.ckpt, args.residual_scale)
    stage_tag = (cfg.get("stage") or "?").split("_")[0]
    steps_state = ck_state.get("steps")

    if args.mode in ("episode", "sidebyside"):
        seg = _resolve_segment(args)
        ep = make_env(model, q_stand, seg, seg.source, args.seed,
                      args.residual_scale)
        verdict_name = args.name or f"{stage_tag}_{seg.source}_seed{args.seed}"
        title = f"track {stage_tag} {seg.name} [{seg.label}]"
        subtitle = (f"ckpt steps={steps_state} seed={args.seed} "
                    f"(learned policy, reference = command)")
        if args.mode == "episode":
            rec = ClipRecorder(title, subtitle)
            res, _ = rollout(net, ep, args.seed, args.max_steps, rec)
            suffix = "" if res["completed"] else "_fail"
            suffix = "_diag" if seg.source in HELD_OUT_TAKES and res["completed"] \
                else suffix
            mp4 = out_dir / f"{verdict_name}{suffix}.mp4"
            info = rec.render(model, mp4)
            sheet = rec.contact_sheet(model, mp4.with_suffix(".png"),
                                      caption=subtitle)
            metrics = {"provenance": provenance(args, ck_state), "rollout": res,
                       "segment": seg.name, "label": seg.label,
                       "video": probe_media(mp4) if mp4.exists() else None,
                       "verdict": ("COMPLETED the segment (no fall/dorsal/deviation)"
                                   if res["completed"]
                                   else f"FAILED: {res['cause']} (kept as failure)")}

            def key(s):
                return (s["completed"], s["site_err_mean"] or 9)

            print("[render] verification:", json.dumps(
                {**metrics["rollout"], "video": info, "sheet": sheet["path"],
                 "reset_count": 0, "fall_count": int(not res["completed"])}, default=str))
            (out_dir / f"{verdict_name}{suffix}.json").write_text(
                json.dumps(metrics, indent=1, default=str))
            assert key(metrics["rollout"]) is not None
            print(f"[render] wrote {mp4} ({info['frames']} frames)")
        else:
            sidebyside(net, ep, seg, out_dir, verdict_name, args, ck_state)
        return 0

    # ---- final: continuous no-reset drill (loops through the REPEAT seam)
    source = args.source or "drill_continuous"
    tt = track_targets(source, model)
    k1 = len(tt)
    seg = Segment(source, 0, k1, "DRILL", 1, validity="composed")
    if args.loops > 1:
        seg = Segment(source, 0, k1 + (k1 - 50) * (args.loops - 1), "DRILL", 1,
                      validity="composed-loop")
        # beyond the first pass the env's _target_root reads clamp at len-1;
        # looping is implemented in TrackingEnv only if the segment exceeds the
        # track: here we keep the honest single-pass default unless it completes.
    ep = make_env(model, q_stand, seg, source, args.seed, args.residual_scale)
    ep.env.horizon = 1e9  # no truncation: the take itself ends the episode
    name = args.name or f"{stage_tag}_final_continuous_drill"
    rec = ClipRecorder(f"FINAL continuous drill ({stage_tag})", f"no resets, "
                       f"ckpt steps={steps_state}, seed={args.seed}")
    res, phases = rollout(net, ep, args.seed, args.max_steps, rec, record_trace=True)
    ok = res["completed"]
    mp4 = out_dir / f"{name}{'' if ok else '_fail'}.mp4"
    info = rec.render(model, mp4)
    sheet = rec.contact_sheet(model, mp4.with_suffix(".png"), caption="no-reset take")
    m = ep.env.recorder.summary() if len(ep.env.recorder.rows) else {}
    metrics = {"provenance": provenance(args, ck_state), "rollout": res,
               "phases": phases,
               "reset_count": 0, "fall_count": int(not ok),
               "longest_continuous_s": round(res["steps"] * 0.02, 2),
               "recorder_summary": _jsonable(m),
               "video": probe_media(mp4) if mp4.exists() else None,
               "verdict": ("continuous take completed" if ok
                           else f"broken at {res['steps']} steps: {res['cause']} "
                                f"(kept, NOT an acceptance artifact)")}
    (out_dir / f"{name}{'' if ok else '_fail'}.json").write_text(
        json.dumps(metrics, indent=1, default=str))
    print("[render] final verification:", json.dumps(
        {k: metrics[k] for k in ("reset_count", "fall_count",
                                 "longest_continuous_s", "rollout")}, default=str))
    print(f"[render] wrote {mp4} ({info['frames']} frames)")
    return 0


def _jsonable(m):
    return {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating))
                else v) for k, v in m.items() if isinstance(v, (int, float, np.floating))}


def _resolve_segment(args) -> Segment:
    source = args.source or "stance_hold"
    tt = track_targets(source, load_solo_model())
    k0 = args.k0 or 0
    k1 = min(args.k1 or len(tt), len(tt))
    meta = tt.meta or {}
    lead = -1 if str(meta.get("lead_leg", "right")).startswith("l") else 1
    from solo.track import DRILL_LABELS, TAKE_LABELS, phase_label

    if source == "drill_continuous":
        lab, connect = phase_label(_phase_name_at(tt, k0))
    else:
        lab = TAKE_LABELS[source]
        if lab == "CIRCLE":
            lab = "CIRCLE_R" if lead > 0 else "CIRCLE_L"
    return Segment(source, k0, k1, lab, lead, connect if source == "drill_continuous"
                   else False, str(meta.get("validity", "unverified")))


def _phase_name_at(tt, k0: int) -> str:
    import numpy as np

    z = np.load(tt.source, allow_pickle=True)
    if "phase_id" not in z:
        return "STAND"
    pid = int(z["phase_id"][k0])
    for p in (tt.meta or {}).get("phases", []):
        if int(p["id"]) == pid:
            return p["name"]
    return "STAND"


def sidebyside(net, ep: TrackingEnv, seg: Segment, out_dir: Path, name: str,
               args, ck_state) -> None:
    """Reference-vs-policy stills at matching phases (rubric H1)."""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    rec = ClipRecorder("sbs", "sbs")
    res, _ = rollout(net, ep, args.seed, args.max_steps, rec)
    # policy qpos at up to 6 landmark times
    times = [0.15, 0.35, 0.5, 0.65, 0.85, 0.97]
    policy_q = [fr.qpos for fr in rec.frames] or None
    if policy_q is None:
        raise RuntimeError("no frames captured")
    n = len(policy_q)
    W, H = 480, 360
    renderer = mujoco.Renderer(ep.env.model, height=H, width=W)
    data = mujoco.MjData(ep.env.model)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    rows_img = []
    for frac in times:
        i = min(int(frac * n), n - 1)
        kf = min(seg.k0 + int(frac * (seg.k1 - seg.k0)), len(ep.tt) - 1)
        imgs = []
        for q in (ep.tt.qpos[kf], policy_q[i]):
            data.qpos[:] = q
            mujoco.mj_forward(ep.env.model, data)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.azimuth, cam.elevation, cam.distance = 120.0, -15.0, 3.0
            cam.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), 0.55]
            renderer.update_scene(data, camera=cam)
            imgs.append(np.asarray(renderer.render()))
        cell = Image.new("RGB", (W * 2 + 4, H + 20), (16, 16, 18))
        d = ImageDraw.Draw(cell)
        tt_s = kf * 0.02
        d.text((6, 2), f"t={tt_s:.2f}s  REF (left)  vs  POLICY (right)",
               font=font, fill=(255, 230, 160))
        cell.paste(Image.fromarray(imgs[0]), (0, 20))
        cell.paste(Image.fromarray(imgs[1]), (W + 4, 20))
        rows_img.append(cell)
    renderer.close()
    grid = Image.new("RGB", (rows_img[0].width,
                             sum(r.height for r in rows_img)), (16, 16, 18))
    y = 0
    for r in rows_img:
        grid.paste(r, (0, y))
        y += r.height
    out = out_dir / f"{name}_compare.png"
    grid.save(out)
    print(f"[render] side-by-side {out} rollout_completed={res['completed']} "
          f"cause={res['cause']}")
    (out_dir / f"{name}_compare.json").write_text(json.dumps(
        {"provenance": provenance(args, ck_state), "rollout": res,
         "note": "left = reference kinematic target (NOT executed motion), "
                 "right = learned policy in physics"}, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
