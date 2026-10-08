"""Render the retargeted video transitions on the G1 (single robot).

Two products per transition (videos/refs_video/, 960x720, 30 fps, h264):
  <name>_g1.mp4      PD-tracked rollout: the G1 driven by the reference joint
                     targets through the model's position servos, rendered at
                     30 fps with a HUD (name, source window, t, pelvis z,
                     torso tilt, joint error).
  <name>_g1_slow.mp4 same rollout at 0.25x playback (rubric H3).

Metrics JSON (data/refs_video/<name>_metrics.json): joint tracking error,
pelvis z min/mean, max torso tilt, ground penetration, fall flag, plus the
reference's own pelvis-height profile (min/max) so "can it hold it" is
answerable numerically. A 3-frame contact sheet goes next to the mp4.

Run (repo root, EGL):
  MUJOCO_GL=egl .venv/bin/python .../render_refs_video.py --names stance,...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from retarget.landmarks import SITE_SPECS  # noqa: E402

FPS = 30
W, H = 960, 720
SCENE_XML = REPO / "robots/g1/scene.xml"
FOOT_SITES = ("left_toe", "right_toe", "left_heel", "right_heel",
              "left_ankle", "right_ankle")


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    for name, body, local in SITE_SPECS:
        spec.body(body).add_site(name=name, pos=list(local))
    return spec.compile()


def ref_at(t: float, t_ref: np.ndarray, q: np.ndarray) -> np.ndarray:
    i = int(np.clip(np.searchsorted(t_ref, t), 1, len(t_ref) - 1))
    w = (t - t_ref[i - 1]) / max(t_ref[i] - t_ref[i - 1], 1e-9)
    return q[i - 1, 7:36] * (1 - w) + q[i, 7:36] * w


def rollout(model: mujoco.MjModel, qpos_ref: np.ndarray, t_ref: np.ndarray,
            slow: bool = False):
    """PD-track the reference; return (frames, hud_values, metrics)."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_ref[0]
    mujoco.mj_forward(model, data)
    foot_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
                         for n in FOOT_SITES])
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 140.0, -12.0, 3.1
    cam.lookat[:] = [0.0, 0.0, 0.62]

    dt = model.opt.timestep
    hold = 0.6                      # s of extra tracking after the reference
    t_end = float(t_ref[-1]) + hold
    steps = int(np.ceil(t_end / dt))
    cap_dt = 1.0 / FPS
    next_cap = 0.0
    frames, hud = [], []
    errs, pelvis_z, tilt, pen = [], [], [], []
    for k in range(steps):
        t = k * dt
        data.ctrl[:29] = ref_at(t, t_ref, qpos_ref)
        mujoco.mj_step(model, data)
        if k % 10 == 0:
            errs.append(float(np.abs(data.qpos[7:36] - ref_at(t, t_ref, qpos_ref)).mean()))
            pelvis_z.append(float(data.qpos[2]))
            up = data.xmat[torso_id].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
            tilt.append(float(np.degrees(np.arccos(np.clip(up[2], -1, 1)))))
            pen.append(float(max(0.0, -data.site_xpos[foot_ids][:, 2].min())))
        if t >= next_cap - 1e-9:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
            hud.append((t, float(data.qpos[2]), tilt[-1], errs[-1]))
            next_cap += cap_dt
    renderer.close()
    metrics = {
        "joint_err_mean_rad": round(float(np.mean(errs)), 4),
        "joint_err_p95_rad": round(float(np.percentile(errs, 95)), 4),
        "pelvis_z_min_m": round(float(np.min(pelvis_z)), 4),
        "pelvis_z_mean_m": round(float(np.mean(pelvis_z)), 4),
        "pelvis_z_end_m": round(float(pelvis_z[-1]), 4),
        "torso_tilt_max_deg": round(float(np.max(tilt)), 2),
        "ground_penetration_max_m": round(float(np.max(pen)), 5),
        "fall_flag": bool(np.min(pelvis_z) < 0.35 or np.max(tilt) > 60.0),
        "n_frames": len(frames),
    }
    return frames, hud, metrics


def hud_draw(img: np.ndarray, lines: list[str]) -> np.ndarray:
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 19)
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([0, 0, 700, 14 + 24 * len(lines)], fill=(0, 0, 0))
    for i, ln in enumerate(lines):
        d.text((8, 8 + 24 * i), ln, fill=(255, 255, 0), font=font)
    return np.asarray(pil)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs-dir", default=str(REPO / "data/refs_video"))
    ap.add_argument("--out-dir", default=str(REPO / "videos/refs_video"))
    ap.add_argument("--names", default="")
    ap.add_argument("--contact", action="store_true", default=True)
    args = ap.parse_args()

    import imageio.v2 as imageio

    refs_dir, out_dir = Path(args.refs_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [x for x in args.names.split(",") if x.strip()]
    if not names:
        names = sorted(p.stem for p in refs_dir.glob("*.npz"))

    model = build_model()
    summary = {}
    for name in names:
        p = refs_dir / f"{name}.npz"
        d = np.load(p, allow_pickle=True)
        q = d["qpos_a"].astype(np.float64)
        t = d["t"].astype(np.float64)
        meta = json.loads(str(d["meta"]))
        frames, hud, metrics = rollout(model, q, t)
        metrics["ref_pelvis_z_min_m"] = round(float(q[:, 2].min()), 4)
        metrics["ref_pelvis_z_max_m"] = round(float(q[:, 2].max()), 4)
        metrics["ref_pelvis_z_end_m"] = round(float(q[-1, 2]), 4)
        metrics["name"] = name
        metrics["source_window_s"] = meta["source_window_s"]
        metrics["scale_g1_over_human"] = meta["scale_g1_over_human"]
        metrics["rms_weighted_m"] = meta["rms"]["weighted"]
        summary[name] = metrics

        out_frames = []
        for img, (tt, pz, tl, er) in zip(frames, hud):
            out_frames.append(hud_draw(img, [
                f"{name}   source {meta['source_window_s'][0]:.1f}-"
                f"{meta['source_window_s'][1]:.1f}s   G1 PD-track",
                f"t={tt:5.2f}s  pelvis_z={pz:.3f}m  tilt={tl:4.1f}deg  "
                f"jerr={er:.3f}rad",
            ]))
        mp4 = out_dir / f"{name}_g1.mp4"
        imageio.mimwrite(mp4, out_frames, fps=FPS, quality=8,
                         macro_block_size=None)
        (out_dir / f"{name}_hud.json").write_text(json.dumps({
            "t": [round(h[0], 4) for h in hud],
            "pelvis_z": [round(h[1], 5) for h in hud],
            "tilt_deg": [round(h[2], 3) for h in hud],
            "joint_err_rad": [round(h[3], 5) for h in hud],
            "source_window_s": meta["source_window_s"],
            "name": name,
        }))
        # 0.25x slow motion (frame repetition)
        slow = [f for f in out_frames for _ in range(4)]
        imageio.mimwrite(out_dir / f"{name}_g1_slow.mp4", slow, fps=FPS,
                         quality=8, macro_block_size=None)
        if args.contact and len(out_frames) >= 3:
            idx = [0, len(out_frames) // 2, len(out_frames) - 1]
            sheet = np.concatenate([out_frames[i] for i in idx], axis=1)
            Image.fromarray(sheet).save(out_dir / f"{name}_contact.png")
        print(f"{name}: {len(out_frames)} frames  jerr_mean="
              f"{metrics['joint_err_mean_rad']:.3f}  pelvis_min="
              f"{metrics['pelvis_z_min_m']:.3f}  tilt_max="
              f"{metrics['torso_tilt_max_deg']:.1f}  fall={metrics['fall_flag']}")
    (refs_dir / "render_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", refs_dir / "render_summary.json")


if __name__ == "__main__":
    main()
