"""Evidence rendering: HUD overlays, streamed mp4, contact sheets, slow-motion.

Rules this module obeys (review §4-5, operator evidence contract):

* frames are **streamed** to the encoder as they are produced -- a 90 s clip at
  960x720x30 fps is ~5.4 GB of raw RGB, and this host has no swap;
* the 960x720 / 30 fps / h264 / yuv420p output is the acceptance format; the
  renderer can also emit a diagnostic-quality pass (``scale`` < 1) for a quick
  eyeball, but a diagnostic never counts as the evidence clip;
* frames come from the cached physics trajectory (qpos at 50 Hz, sampled to
  30 fps), never from a kinematic replay, and the overlay states the controller
  kind, the rung, the seed/config, the live command, the measured state and the
  push windows, so a clip can be audited without the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
VIDEO_DIR = REPO / "videos" / "solo_drill"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

WIDTH, HEIGHT = 960, 720
FPS = 30
#: fixed side-3/4 camera (MuJoCo free camera, degrees) — the same framing for
#: every clip, so two clips can be compared frame for frame
CAM_AZIMUTH = 132.0
CAM_ELEVATION = -13.0
CAM_DISTANCE = 3.35
CAM_LOOKAT_Z = 0.72

SKILLS = ("STANCE", "ENTRY", "SHUFFLE_F", "SHUFFLE_B", "SHUFFLE_L", "SHUFFLE_R",
          "APPROACH", "RETREAT", "CIRCLE_L", "CIRCLE_R", "LEVEL_CHANGE",
          "SHOT_GESTURE", "RECOVER")


def load_trace(npz: Path | str) -> dict:
    with np.load(npz, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def sample_frames(trace: dict, fps: int = FPS, t0: float = 0.0,
                  t1: float | None = None, speed: float = 1.0) -> dict:
    """Resample the control-rate trace to video frames (linear in qpos).

    ``speed`` < 1 slows the clip down (0.25 = quarter speed) by interpolating
    more frames over the same physics window -- the motion is never
    re-timed in the physics, only in the sampling.
    """
    t = np.asarray(trace["t"], float)
    t1 = float(t[-1]) if t1 is None else min(t1, float(t[-1]))
    n = max(2, int(round((t1 - t0) / speed * fps)) + 1)
    tf = t0 + (t1 - t0) * np.arange(n) / (n - 1)
    out = {"t": tf}
    for key in ("qpos", "com", "margin", "tilt_deg", "clearance", "com_ref",
                "e_track", "safety_alpha", "push_force", "knee_z", "foot_load"):
        arr = np.asarray(trace[key], float)
        arr = arr.reshape(len(t), -1)
        out[key] = np.stack([np.interp(tf, t, arr[:, j]) for j in range(arr.shape[1])],
                            axis=1)
        if arr.shape[1] == 1:
            out[key] = out[key][:, 0]
    for key in ("contact", "skill_id", "cmd", "push", "plan_planted"):
        arr = np.asarray(trace[key])
        idx = np.clip(np.searchsorted(t, tf), 0, len(t) - 1)
        out[key] = arr[idx]
    return out


def _label_row(name_ix: int, cmd: np.ndarray, phase: str) -> str:
    skill = SKILLS[int(name_ix)] if 0 <= int(name_ix) < len(SKILLS) else "?"
    return (f"skill {skill:12s} cmd vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} "
            f"wz={cmd[2]:+.2f} h={cmd[3]:.2f}  phase {phase or '-'}")


def render_trace(trace: dict, out_mp4: Path | str, *, meta: dict | None = None,
                 fps: int = FPS, width: int = WIDTH, height: int = HEIGHT,
                 t0: float = 0.0, t1: float | None = None, speed: float = 1.0,
                 phases: dict | None = None, push_times: tuple = (),
                 sheet_times: tuple | None = None, caption: str = "",
                 scale: float = 1.0, progress: bool = True,
                 label: str = "") -> dict:
    """Render a trajectory to mp4 + a 3-frame contact sheet (streamed)."""
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    from . import scene as scene_mod

    frames = sample_frames(trace, fps=fps, t0=t0, t1=t1, speed=speed)
    model = scene_mod.load_model(None)
    if scale != 1.0:
        width, height = int(width * scale), int(height * scale)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    flags = renderer._scene.flags
    font = ImageFont.truetype(str(FONT), int(15 * scale) or 12)
    font_b = ImageFont.truetype(str(FONT_BOLD), int(16 * scale) or 12)
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    keep, t_keep = {}, {}
    sheet_times = sheet_times or ()
    n = len(frames["t"])
    writer = imageio.get_writer(out_mp4, fps=fps * (1.0 / speed), quality=8,
                                macro_block_size=None, mode="I")
    try:
        base_xy = np.asarray(frames["qpos"])[:, :2].mean(axis=0) if n else np.zeros(2)
        for i in range(n):
            t = float(frames["t"][i])
            data.qpos[:] = frames["qpos"][i]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.azimuth, cam.elevation = CAM_AZIMUTH, CAM_ELEVATION
            cam.distance = CAM_DISTANCE
            cam.lookat[:] = [base_xy[0], base_xy[1] + 0.02, CAM_LOOKAT_Z]
            renderer.update_scene(data, camera=cam)
            flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            img = renderer.render()
            lines = _hud_lines(frames, i, meta or {}, label)
            img = _draw_hud(img, lines, font, font_b, scale)
            writer.append_data(img)
            for st in sheet_times:
                if abs(t - st) < 0.5 / fps and st not in keep:
                    keep[st] = img.copy()
                    t_keep[st] = t
            if progress and (i + 1) % 200 == 0:
                print(f"      {out_mp4.name}: {i+1}/{n} frames t={t:.1f}s")
    finally:
        writer.close()
        renderer.close()
    sheet = out_mp4.with_name(out_mp4.stem + "_sheet.png")
    if keep:
        _write_sheet(keep, sheet, caption or out_mp4.stem, font_b)
    return {"mp4": str(out_mp4), "sheet": str(sheet) if keep else None,
            "frames": n, "t0": t0, "t1": t1, "speed": speed}


def _hud_lines(frames: dict, i: int, meta: dict, label: str) -> list:
    t = float(frames["t"][i])
    cmd = np.asarray(frames["cmd"])[i]
    skill = int(np.asarray(frames["skill_id"])[i])
    z = float(np.asarray(frames["qpos"])[i][2])
    tilt = float(np.asarray(frames["tilt_deg"])[i])
    marg = float(np.asarray(frames["margin"])[i])
    clear = np.asarray(frames["clearance"])[i]
    load = np.asarray(frames["foot_load"])[i]
    push = float(np.asarray(frames["push_force"])[i])
    alpha = float(np.asarray(frames["safety_alpha"])[i])
    err = float(np.linalg.norm(np.asarray(frames["e_track"])[i]))
    knee = np.asarray(frames["knee_z"])[i]
    phase = (meta.get("phases") or {}).get(round(t, 1), "")
    out = [
        f"t={t:6.2f}s   {meta.get('title', '')}",
        f"controller {meta.get('controller_kind', '')}   seed {meta.get('seed', 0)}"
        f"   rung {meta.get('rung', '')}   {label}",
        _label_row(skill, cmd, phase),
        f"pelvis z {z:.3f} m   torso tilt {tilt:4.1f} deg   CoM margin {marg:+.3f} m",
        f"track err {err:.3f} m   foot load L/R {load[0]:5.0f}/{load[1]:5.0f} N",
        f"sole clearance L/R {clear[0]*100:+.1f}/{clear[1]*100:+.1f} cm   "
        f"knee z {knee[0]:.2f}/{knee[1]:.2f} m",
    ]
    if push > 0.5:
        out.append(f"*** PUSH {push:.0f} N ({meta.get('push_labels', '')}) ***")
    if alpha > 0.05:
        out.append(f"!! safety blend alpha={alpha:.2f} (emergency response active)")
    if meta.get("footer"):
        out.append(meta["footer"])
    return out


def _draw_hud(img: np.ndarray, lines: list, font, font_b, scale: float) -> np.ndarray:
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    pad, lh = int(8 * scale) or 6, int(18 * scale) or 14
    for k, line in enumerate(lines):
        f = font_b if (k < 2 or line.startswith(("***", "!!"))) else font
        w = dr.textlength(line, font=f)
        y = pad // 2 + k * lh
        dr.rectangle([pad // 2, y, pad // 2 + w + 8, y + lh - 2], fill=(0, 0, 0, 150))
        colour = (255, 240, 120) if line.startswith("***") else (
            (255, 140, 140) if line.startswith("!!") else (240, 240, 240))
        dr.text((pad // 2 + 4, y + 1), line, font=f, fill=colour)
    return np.asarray(im)


def _write_sheet(keep: dict, out_png: Path, caption: str, font) -> None:
    from PIL import Image, ImageDraw
    imgs = [img for _, img in sorted(keep.items())]
    h, w = imgs[0].shape[:2]
    sheet = Image.new("RGB", (w * len(imgs), h + 26), (16, 16, 16))
    for k, img in enumerate(imgs):
        sheet.paste(Image.fromarray(img), (k * w, 0))
    dr = ImageDraw.Draw(sheet)
    for k, t in enumerate(sorted(keep)):
        dr.text((k * w + 8, h + 4), f"t={t:.2f}s   {caption[:44]}", font=font,
                fill=(230, 230, 230))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


def side_by_side(pairs: list, out_png: Path | str, caption: str = "",
                 max_width: int = 1440) -> str:
    """Compose reference stills against drill frames (rubric H1)."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(FONT_BOLD), 16)
    rows = []
    for left, right, tag in pairs:
        a, b = Image.open(left).convert("RGB"), Image.open(right).convert("RGB")
        h = 360
        a = a.resize((int(a.width * h / a.height), h))
        b = b.resize((int(b.width * h / b.height), h))
        rows.append((a, b, tag))
    w = max(a.width + b.width for a, b, _ in rows)
    sheet = Image.new("RGB", (w, (360 + 24) * len(rows) + 24), (12, 12, 12))
    dr = ImageDraw.Draw(sheet)
    y = 0
    for a, b, tag in rows:
        sheet.paste(a, (0, y))
        sheet.paste(b, (a.width, y))
        dr.text((4, y + 362), f"{tag}", font=font, fill=(240, 240, 240))
        y += 360 + 24
    dr.text((4, y + 4), caption[:120], font=font, fill=(220, 220, 120))
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return str(out)


if __name__ == "__main__":                              # self-check
    import sys
    npz = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/drill_preview.mp4")
    trace = load_trace(npz)
    meta = {"title": npz.stem, "controller_kind": "feasible",
            "seed": 0, "rung": "L0"}
    r = render_trace(trace, out, meta=meta, t0=0.0, t1=3.0, scale=0.5,
                     sheet_times=(0.5, 1.5, 2.5), caption="self-check preview")
    print("rendered", r)
