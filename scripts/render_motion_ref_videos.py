#!/usr/bin/env python
"""Render the motion-reference evidence bundle (docs/EVIDENCE_PROTOCOL.md).

Videos (all 960x720, 30 fps, h264/yuv420p) + 3-frame contact sheets + metrics
JSON with provenance, under videos/motion_refs/:

  drill_phases.mp4          the composed drill reference, kinematically
                            replayed with a phase/HUD overlay (WHAT TO LOOK
                            FOR: continuous transitions, no teleport, feet
                            lift-and-place, root travel accumulates)
  source_vs_g1_<take>.mp4   source frame with the detected skeleton BESIDE the
                            G1 at the matched fraction of the take (2 takes:
                            level_change_full, shot_entry_full)
  key_postures.mp4          side + three-quarter views of the five key
                            postures taken from the composed drill

Honesty notes baked into the HUD: these are KINEMATIC replays of desired
reference targets (no controller, no physics claim); dynamic evidence lives in
dynamic_probe_*.mp4 (scripts/probe_drill_dynamic.py).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "data/references/yt_gBAhX5t-GW4/derived/tools"))

import mujoco  # noqa: E402

from solo.scene import load_solo_model  # noqa: E402
from solo.video import WIDTH, HEIGHT, FPS, draw_hud, probe_media  # noqa: E402

V1 = REPO / "data/references/motion_refs/v1"
OUT = REPO / "videos/motion_refs"
SOURCE_VIDEO = REPO / "data/references/yt_gBAhX5t-GW4/ref720h264.mp4"
POSE_NPZ = REPO / "data/references/yt_gBAhX5t-GW4/pose/landmarks.npz"


def renderer_for(model):
    r = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    return r


def cam(kind: str, lookat):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = {
        "side": (90.0, -12.0, 2.6),
        "three_quarter": (135.0, -14.0, 2.9),
        "front": (180.0, -10.0, 2.8),
    }[kind]
    cam.lookat[:] = lookat
    return cam


def render_frames(model, rend, qpos_list, lines_fn, cam_kind="side"):
    data = mujoco.MjData(model)
    imgs = []
    for i, q in enumerate(qpos_list):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        look = [float(data.qpos[0]), float(data.qpos[1]), 0.5]
        c = cam(cam_kind, look)
        rend.update_scene(data, camera=c)
        imgs.append(draw_hud(rend.render(), lines_fn(i, q, data)))
    return imgs


def write_video(imgs, path):
    import imageio.v2 as imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), imgs, fps=FPS, quality=8, macro_block_size=None)
    return probe_media(path)


def contact_sheet(imgs, path, title, labels=("start", "mid", "end")):
    from PIL import Image, ImageDraw, ImageFont
    idxs = [0, len(imgs) // 2, len(imgs) - 1]
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 17)
    small = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    head, lab = 58, 26
    sheet = Image.new("RGB", (WIDTH * 3, HEIGHT + head + lab), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    d.text((10, 6), title, font=font, fill=(255, 255, 255))
    d.text((10, 30), "kinematic replay of DESIRED reference targets "
            "(no controller, no physics claim)", font=small, fill=(200, 220, 255))
    for k, (i, label) in enumerate(zip(idxs, labels)):
        d.text((10 + k * WIDTH, head + 4), label, font=small, fill=(255, 235, 160))
        sheet.paste(Image.fromarray(imgs[i]), (k * WIDTH, head + lab))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return {"path": str(path), "frames": 3}


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def provenance(extra):
    return {"git_commit": git_commit(), "seed": 0, "control_rate_hz": 50,
            "physics_timestep_s": 0.002, "generated_by": __file__,
            "wall_s": round(extra.pop("wall_s", 0.0), 1),
            "reproduce": "MUJOCO_GL=egl .venv/bin/python "
                         "scripts/render_motion_ref_videos.py", **extra}


def metrics_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1))
    print(f"  metrics -> {path}")


# ---------------------------------------------------------------- drill ----
def render_drill(model, rend):
    d = np.load(V1 / "drill_continuous.npz", allow_pickle=True)
    q = np.asarray(d["qpos_a"], float)
    meta = json.loads(str(d["meta"]))
    phases = meta["phases"]
    contact = np.asarray(d["contact"], bool)
    phase_of = np.asarray(d["phase_id"], int)
    t = np.asarray(d["t"], float)
    step = int(round((1.0 / FPS) / 0.02))
    qs = q[::step]
    idxs = np.arange(len(qs)) * step
    rows = lambda i, qq, data: (
        f"t={t[idxs[i]]:6.2f}s  phase={phases[phase_of[idxs[i]]]['name']}",
        f"source={phases[phase_of[idxs[i]]]['source']}  "
        f"contact L/R={int(contact[idxs[i]][0])}/{int(contact[idxs[i]][1])}",
        f"pelvis_z={float(qq[2]):.3f} m  KINEMATIC REFERENCE (desired targets)")
    t0 = time.time()
    imgs = render_frames(model, rend, qs, rows, "side")
    wall = time.time() - t0
    media = write_video(imgs, OUT / "drill_phases.mp4")
    sheet = contact_sheet(imgs, OUT / "drill_phases_sheet.png",
                          "composed drill reference v1 -- phase by phase")
    travel = float(np.linalg.norm(q[-1, :2] - q[0, :2]))
    jumps = float(np.abs(np.diff(q[:, 7:], axis=0)).max())
    metrics_json(OUT / "drill_phases.json", {
        "artifact": "videos/motion_refs/drill_phases.mp4",
        "media": media, "sheet": sheet,
        "what_to_look_for": "continuous C1 transitions (no teleport/joint snap); "
                            "feet lift and place (contact flags); root travel "
                            "accumulates; the DOUBLE_LEG_ENTRY crouch is the "
                            "known-infeasible segment (kept, labelled)",
        "kinematic": {"duration_s": float(t[-1]), "n_frames_50hz": int(len(q)),
                      "root_displacement_m": round(travel, 3),
                      "max_joint_step_rad": round(jumps, 5),
                      "contact_frac_L": round(float(contact[:, 0].mean()), 3),
                      "contact_frac_R": round(float(contact[:, 1].mean()), 3),
                      "phases": phases},
        "provenance": provenance({"wall_s": wall,
                                   "config": str(V1 / "drill_continuous.npz"),
                                   "config_hash": hashlib.sha256(
                                       (V1 / "drill_continuous.npz").read_bytes()
                                   ).hexdigest()[:16]}),
    })


# ---------------------------------------------------------- source vs g1 ----
def source_vs_g1(model, rend, take, t0, t1):
    import vidframe  # derived/tools helper (frame extraction)
    from draw_pose_overlay import draw as draw_skel
    p = np.load(POSE_NPZ, allow_pickle=True)
    pt, img_lm, vis = p["t"], p["img_lm"], p["visibility"]
    ch = p["chapter"].astype(str)
    chapter = "5" if take.startswith("level") else "7"
    sel = np.flatnonzero((pt >= t0) & (pt <= t1) & (ch == chapter))
    dref = np.load(V1 / "refs" / f"{take}.npz", allow_pickle=True)
    qr = np.asarray(dref["qpos_a"], float)
    k = 40  # side-by-side samples across the take
    src_idx = np.linspace(0, len(sel) - 1, k).astype(int)
    g1_idx = (src_idx / max(src_idx[-1], 1) * (len(qr) - 1)).astype(int)
    data = mujoco.MjData(model)
    want_times = [float(pt[sel[i]]) for i in src_idx]
    got = {}
    for ts, frame in vidframe.frames(SOURCE_VIDEO, start_s=max(0.0, t0 - 0.5)):
        for wt in want_times:
            if wt not in got and abs(ts - wt) < 0.06:
                got[wt] = frame
        if len(got) == len(want_times):
            break
    from PIL import Image
    imgs = []
    for si, gi in zip(src_idx, g1_idx):
        wt = float(pt[sel[si]])
        if wt not in got:
            continue
        overlay = draw_skel(got[wt], img_lm[sel[si]], vis[sel[si]],
                            f"src t={wt:.2f}s")
        data.qpos[:] = qr[gi]
        mujoco.mj_forward(model, data)
        c = cam("side", [float(data.qpos[0]), float(data.qpos[1]), 0.5])
        rend.update_scene(data, camera=c)
        g1 = rend.render()
        # 2-up on a 960x720 canvas: source left, G1 right
        left = np.asarray(Image.fromarray(overlay).resize((480, 720)))
        right = np.asarray(Image.fromarray(g1).resize((480, 720)))
        both = np.concatenate([left, right], axis=1)
        label = (f"source vs G1: {take}  (left=video+skeleton, right=G1 retarget)",
                 f"matched fraction {si/max(src_idx[-1],1):.2f} of the take",
                 "KINEMATIC retarget -- desired targets, no physics claim")
        imgs.append(draw_hud(both, label))
    media = write_video(imgs, OUT / f"source_vs_g1_{take}.mp4")
    sheet = contact_sheet(imgs, OUT / f"source_vs_g1_{take}_sheet.png",
                          f"source vs G1: {take}")
    rms = float(dref["landmark_rms"]) if "landmark_rms" in dref else None
    metrics_json(OUT / f"source_vs_g1_{take}.json", {
        "artifact": f"videos/motion_refs/source_vs_g1_{take}.mp4",
        "media": media, "sheet": sheet,
        "what_to_look_for": "the G1 keeps the video's phase structure and "
                            "carriage; morphology differences (1.32 m robot, "
                            "no fingers) explain the rest",
        "source_window_s": [t0, t1],
        "landmark_rms_weighted_m": round(rms, 4) if rms else None,
        "provenance": provenance({"source": str(SOURCE_VIDEO.relative_to(REPO)),
                                   "reference": str((V1 / 'refs' / f'{take}.npz').relative_to(REPO))}),
    })


# ------------------------------------------------------------ postures ----
def key_postures(model, rend):
    d = np.load(V1 / "drill_continuous.npz", allow_pickle=True)
    q = np.asarray(d["qpos_a"], float)
    t = np.asarray(d["t"], float)
    meta = json.loads(str(d["meta"]))
    by_name = {p["name"]: p for p in meta["phases"]}

    def at_phase(name, frac=0.5):
        p = by_name[name]
        return int(np.searchsorted(t, p["t_start"] + frac * (p["t_end"] - p["t_start"])))

    picks = {
        "STAND": at_phase("STAND", 0.5),
        "STANCE_HOLD": at_phase("STANCE_HOLD", 0.5),
        "LEVEL_CHANGE": int(np.argmin(q[:, 2])),                 # deepest crouch
        "DOUBLE_LEG_PENETRATION": at_phase("DOUBLE_LEG_PENETRATION", 0.95),  # knee-down
        "RECOVER_TO_STANCE": at_phase("RECOVER_TO_STANCE", 0.95),
    }
    order = list(picks)
    imgs = []
    hold = int(FPS * 0.9)
    for name in order:
        i = picks[name]
        ph = by_name.get(name)
        for kind in ("side", "three_quarter"):
            subset = [q[min(i + j, len(q) - 1)] for j in range(hold)]
            rows = lambda k2, qq, dd, _n=name, _p=ph, _k=kind: (
                f"key posture: {_n}  camera={_k}",
                (f"phase {_p['t_start']:.2f}-{_p['t_end']:.2f}s "
                 f"({_p['source']})") if _p else "posture from the drill track",
                f"pelvis_z={float(qq[2]):.3f} m")
            imgs += render_frames(model, rend, subset, rows, kind)
    media = write_video(imgs, OUT / "key_postures.mp4")
    sheet = contact_sheet(imgs, OUT / "key_postures_sheet.png",
                          "key postures: side + three-quarter",
                          labels=("STAND/STANCE", "LEVEL_CHANGE/ENTRY", "RECOVER"))
    metrics_json(OUT / "key_postures.json", {
        "artifact": "videos/motion_refs/key_postures.mp4",
        "media": media, "sheet": sheet,
        "postures": order,
        "provenance": provenance({"config": str(V1 / "drill_continuous.npz")}),
    })


def main() -> int:
    model = load_solo_model()
    rend = renderer_for(model)
    try:
        render_drill(model, rend)
        source_vs_g1(model, rend, "level_change_full", 259.7, 261.2)
        source_vs_g1(model, rend, "shot_entry_full", 422.8, 431.0)
        key_postures(model, rend)
    finally:
        rend.close()
    print("SELF-VERIFY: rendered drill_phases.mp4, source_vs_g1_*.mp4, "
          "key_postures.mp4 + sheets + metrics JSON")
    return 0


if __name__ == "__main__":
    sys.exit(main())
