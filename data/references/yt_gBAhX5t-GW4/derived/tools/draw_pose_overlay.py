"""Skeleton overlay renderer for the pose pass.

Draws the MediaPipe pose skeleton (33 keypoints) on the source video frames,
for sanity checks (PNG stills) and verification clips (MP4, local only).

CLI:
  # stills at given source times
  ... draw_pose_overlay.py --mode png --times 34,54,74,150,230,500 --out-dir <dir>
  # clip from a take window (matched to the pose timestamps)
  ... draw_pose_overlay.py --mode mp4 --start 26 --end 83 --fps 30 --out <file.mp4>

Skeleton colour encodes landmark confidence (green >=0.7, amber >=0.4, red else);
the label shows the source timestamp.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
from PIL import Image, ImageDraw  # noqa: E402

from vidframe import frames, probe  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
VIDEO = REPO / "data/references/yt_gBAhX5t-GW4/ref720h264.mp4"
NPZ = REPO / "data/references/yt_gBAhX5t-GW4/pose/landmarks.npz"

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def colour_for(c: float) -> tuple[int, int, int]:
    if c >= 0.7:
        return (0, 255, 0)
    if c >= 0.4:
        return (255, 200, 0)
    return (255, 60, 60)


def draw(img: np.ndarray, lm: np.ndarray, vis: np.ndarray, label: str) -> np.ndarray:
    """img HxWx3 uint8; lm (33,3) normalized image coords."""
    pil = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(pil)
    w, h = pil.size
    pts = [(float(x) * w, float(y) * h) for x, y, _ in lm]
    for a, b in EDGES:
        if a >= len(pts) or b >= len(pts):
            continue
        ca = float(vis[a]) if vis is not None else 1.0
        c = colour_for(min(ca, float(vis[b]) if vis is not None else 1.0))
        d.line([pts[a], pts[b]], fill=c, width=3)
    for i, (x, y) in enumerate(pts):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        c = colour_for(float(vis[i])) if vis is not None else (0, 255, 0)
        r = 4
        d.ellipse([x - r, y - r, x + r, y + r], fill=c, outline=(0, 0, 0))
    d.rectangle([0, 0, 260, 26], fill=(0, 0, 0))
    d.text((8, 7), label, fill=(255, 255, 0))
    return np.asarray(pil)


def nearest_row(npz, ts: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(npz["t"] - t)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["png", "mp4"], required=True)
    ap.add_argument("--npz", default=str(NPZ))
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("--times", default="")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out-dir", default="/tmp/overlay")
    ap.add_argument("--out", default="/tmp/overlay.mp4")
    ap.add_argument("--video-start", type=float, default=0.0,
                    help="source seconds trimmed off the video file (clip start)")
    args = ap.parse_args()

    d = np.load(args.npz)
    ts = d["t"]

    if args.mode == "png":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        times = [float(x) for x in args.times.split(",") if x.strip()]
        assert times, "--times required for png mode"
        for t in times:
            # fast per-time seek (input seeking); npz times are absolute, the
            # video file may be a trim starting at args.video_start
            t_rel = max(0.0, t - args.video_start)
            got = None
            for tt, rgb in frames(args.video, start_s=t_rel, fps_out=30.0):
                got = rgb
                break
            assert got is not None, f"frame at {t} not found"
            i = nearest_row(d, ts, t)
            lm, vis = d["img_lm"][i], d["visibility"][i]
            conf = float(d["conf"][i])
            out = draw(got, lm, vis, f"t={ts[i]:.2f}s conf={conf:.2f} pose_pass")
            pth = out_dir / f"overlay_t{t:07.2f}.png"
            Image.fromarray(out).save(pth)
            print("wrote", pth)
        return

    start, end = args.start, args.end
    assert end > start, "--start/--end required for mp4 mode"
    tmp = Path(tempfile.mkdtemp(prefix="ovl_"))
    n = 0
    info = probe(args.video)
    s0 = max(0.0, start - args.video_start)
    for t_rel, rgb in frames(args.video, start_s=s0, fps_out=args.fps):
        t_abs = t_rel + args.video_start
        if t_abs > end:
            break
        if t_abs < start - 1.0 / args.fps:
            continue
        i = nearest_row(d, ts, t_abs)
        lm, vis = d["img_lm"][i], d["visibility"][i]
        out = draw(rgb, lm, vis, f"t={t_abs:7.2f}s conf={float(d['conf'][i]):.2f}")
        Image.fromarray(out).save(tmp / f"f{n:05d}.png")
        n += 1
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(args.fps),
                    "-i", str(tmp / "f%05d.png"), "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-crf", "20", args.out], check=True)
    print(f"wrote {args.out} ({n} frames {args.fps} fps, source {info.width}x{info.height})")
    assert Path(args.out).stat().st_size > 0


if __name__ == "__main__":
    main()
