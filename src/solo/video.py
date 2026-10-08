"""Evidence rendering for the solo drill: mp4 clips + 3-frame contact sheets.

Conventions (AGENTS.md visual-evidence directive, SOLO_DRILL §6): 960x720,
30 fps, h264/yuv420p, MuJoCo-rendered, with an overlay naming stage/task,
controller/checkpoint, seed, command, key metrics and the verdict; plus a
3-frame PNG contact sheet (start / mid / end).  ``renderer.close()`` is always
called (EGL host rule, notes.md).

Frames are captured *during* the rollout (control steps at 50 Hz, sampled at
30 fps) and rendered afterwards from the stored qpos/mocap snapshots, so
rendering never perturbs the physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .scene import MARKER_NAMES, MODEL_DT, STEP_DT

WIDTH = 960
HEIGHT = 720
FPS = 30
_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
_FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


@dataclass
class Frame:
    """One captured video frame (state + overlay text)."""

    t: float
    qpos: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    lines: tuple[str, ...]
    metrics: dict = field(default_factory=dict)


def hud_lines(title: str, subtitle: str, rows: list[str],
              verdict: str | None = None) -> tuple[str, ...]:
    """Compose overlay lines: title (bold), subtitle, metric rows, verdict."""
    out = [title, subtitle, *rows]
    if verdict is not None:
        out.append(f"VERDICT: {verdict}")
    return tuple(out)


def draw_hud(img: np.ndarray, lines: tuple[str, ...]) -> np.ndarray:
    """Overlay text onto a frame (same style as ``scripts/render_env_videos.py``)."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    font = ImageFont.truetype(str(_FONT_PATH), 15)
    bold = ImageFont.truetype(str(_FONT_BOLD), 15)
    pad, lh = 8, 19
    w = min(int(max(dr.textlength(ln, font=font) for ln in lines)) + 2 * pad, WIDTH - 12)
    h = lh * len(lines) + 2 * pad
    dr.rectangle([6, 6, 6 + w, 6 + h], fill=(0, 0, 0, 165),
                 outline=(230, 230, 230, 120))
    for i, ln in enumerate(lines):
        f = bold if i < 2 else font
        col = (255, 255, 255, 255) if i < 2 else (215, 235, 255, 255)
        dr.text((6 + pad, 6 + pad + i * lh), ln, font=f, fill=col)
    return np.asarray(im)


class ClipRecorder:
    """Samples frames at ``fps`` during a rollout (call :meth:`capture` each step)."""

    def __init__(self, title: str = "", subtitle: str = "", *, fps: int = FPS,
                 max_frames: int | None = None, verdict: str | None = None):
        self.title = title
        self.subtitle = subtitle
        self.fps = int(fps)
        self.max_frames = max_frames
        self.verdict = verdict
        self.frames: list[Frame] = []
        self._next_t = 0.0

    def capture(self, env, info: dict, rows: list[str] | None = None) -> None:
        t = float(env.data.time)
        if t + 1e-9 < self._next_t:
            return
        self._next_t = t + 1.0 / self.fps
        if self.max_frames is not None and len(self.frames) >= self.max_frames:
            return
        row = info.get("metrics") or {}
        lines = hud_lines(self.title, self.subtitle,
                          rows if rows is not None else self.default_rows(env, info),
                          self.verdict)
        self.frames.append(Frame(
            t=t, qpos=np.asarray(env.data.qpos, np.float64).copy(),
            mocap_pos=np.asarray(env.data.mocap_pos, np.float64).copy(),
            mocap_quat=np.asarray(env.data.mocap_quat, np.float64).copy(),
            lines=lines, metrics=dict(row)))

    @staticmethod
    def default_rows(env, info: dict) -> list[str]:
        m = info.get("metrics") or {}
        cmd = info.get("command") or {}
        return [
            f"t={info.get('t', 0.0):5.2f}s  cmd vx={cmd.get('vx', 0):+.2f} "
            f"vy={cmd.get('vy', 0):+.2f} wz={cmd.get('wz', 0):+.2f} "
            f"H={cmd.get('stance_height', 0):.2f} skill={cmd.get('skill', '?')}",
            f"vel_err={m.get('vel_err', float('nan')):.3f} yaw_err="
            f"{m.get('yaw_err', float('nan')):.3f} upright={m.get('upright', 0):.3f} "
            f"pelvis_z={m.get('pelvis_z', 0):.3f}",
            f"slip={m.get('slip', 0):.3f} contacts L/R="
            f"{int(m.get('contact_l') or 0)}/{int(m.get('contact_r') or 0)} "
            f"knee/hand={int(m.get('knee_contact') or 0)}/{int(m.get('hand_contact') or 0)} "
            f"sat={m.get('sat_frac', 0):.2f}",
        ]

    # ------------------------------------------------------------------ render
    def render(self, model, out_path: str | Path, *, lookat=(0.0, 0.0, 0.55),
               distance: float = 3.2, azimuth: float = 120.0,
               elevation: float = -15.0,
               extra_lines: tuple[str, ...] = ()) -> dict:
        """Write the captured frames as an mp4; returns a summary dict.

        ``extra_lines`` are appended to every frame's overlay (used to stamp the
        final verdict, which is only known after the episode finishes).
        """
        import imageio.v2 as imageio
        import mujoco

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        data = mujoco.MjData(model)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = float(azimuth)
        cam.elevation = float(elevation)
        cam.distance = float(distance)
        cam.lookat[:] = np.asarray(lookat, np.float64)
        images = []
        try:
            for fr in self.frames:
                data.qpos[:] = fr.qpos
                if model.nmocap:
                    data.mocap_pos[:] = fr.mocap_pos
                    data.mocap_quat[:] = fr.mocap_quat
                mujoco.mj_forward(model, data)
                pel = data.qpos[0:3]
                cam.lookat[0] = 0.5 * (cam.lookat[0] + float(pel[0]))
                cam.lookat[1] = 0.5 * (cam.lookat[1] + float(pel[1]))
                renderer.update_scene(data, camera=cam)
                lines = fr.lines + tuple(extra_lines)
                images.append(draw_hud(renderer.render(), lines))
        finally:
            renderer.close()  # REQUIRED on this EGL host
        if images:
            imageio.mimwrite(str(out_path), images, fps=self.fps, quality=8,
                             macro_block_size=None)
        return {"path": str(out_path), "frames": len(images), "fps": self.fps,
                "seconds": round(len(images) / self.fps, 3)}

    def contact_sheet(self, model, out_path: str | Path, *,
                      caption: str = "", labels=("start", "mid", "end")) -> dict:
        """3-frame PNG sheet (start / mid / end) at 960x720 each."""
        import mujoco
        from PIL import Image, ImageDraw, ImageFont

        if not self.frames:
            raise ValueError("no frames captured")
        idxs = [0, len(self.frames) // 2, len(self.frames) - 1]
        renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        data = mujoco.MjData(model)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance = 120.0, -15.0, 3.2
        images = []
        try:
            for i in idxs:
                fr = self.frames[i]
                data.qpos[:] = fr.qpos
                if model.nmocap:
                    data.mocap_pos[:] = fr.mocap_pos
                    data.mocap_quat[:] = fr.mocap_quat
                mujoco.mj_forward(model, data)
                cam.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), 0.55]
                renderer.update_scene(data, camera=cam)
                images.append(draw_hud(renderer.render(), fr.lines))
        finally:
            renderer.close()
        head, label_h = 58, 26
        font = ImageFont.truetype(str(_FONT_BOLD), 17)
        small = ImageFont.truetype(str(_FONT_PATH), 14)
        sheet = Image.new("RGB", (WIDTH * 3, HEIGHT + head + label_h), (18, 18, 20))
        dr = ImageDraw.Draw(sheet)
        dr.text((10, 6), self.title, font=font, fill=(255, 255, 255))
        dr.text((10, 30), caption or self.subtitle, font=small, fill=(200, 220, 255))
        for k, (i, label) in enumerate(zip(idxs, labels)):
            fr = self.frames[i]
            dr.text((10 + k * WIDTH, head + 4), f"{label}  t = {fr.t:5.2f} s",
                    font=small, fill=(255, 235, 160))
            sheet.paste(Image.fromarray(images[k]), (k * WIDTH, head + label_h))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_path)
        return {"path": str(out_path), "frames": 3}


def probe_media(path: str | Path) -> dict:
    """ffprobe summary (codec/pix_fmt/size/fps/duration) for the report."""
    import json
    import subprocess

    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames,duration",
           "-of", "json", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def frame_count_seconds(n_frames: int) -> float:
    return round(n_frames / FPS, 3)


if __name__ == "__main__":  # self-check
    import mujoco

    from .env import SoloEnv

    env = SoloEnv(seed=0)
    env.horizon = 1e9
    rec = ClipRecorder("solo.video self-check", "stand hold",
                       max_frames=10, verdict="smoke")
    hold = env._ctrl_stand.copy()
    for k in range(20):
        obs, r, term, trunc, info = env.step(hold)
        rec.capture(env, info)
    assert len(rec.frames) >= 5
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = rec.render(env.model, Path(td) / "clip.mp4")
        assert out["frames"] >= 5
        sheet = rec.contact_sheet(env.model, Path(td) / "sheet.png")
        assert sheet["frames"] == 3
    print("solo.video self-check OK:", {"frames": len(rec.frames)})
