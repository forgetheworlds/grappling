"""Side-by-side comparison: source video (skeleton overlay) vs G1 rollout,
with a live pelvis/knee-height trace underneath.

Inputs per transition name:
  videos/refs_video/<name>_overlay.mp4     source frames + skeleton (LOCAL ONLY,
                                           gitignored: embeds third-party footage)
  videos/refs_video/<name>_g1.mp4          G1 PD-tracked rollout
  videos/refs_video/<name>_hud.json        per-frame G1 sim values (renderer)
  pose/landmarks.npz                       source pelvis/knee heights

Output: videos/refs_video/<name>_compare_overlay.mp4 (960x720, 30 fps).

Heights are plotted NORMALISED to each side's own standing pelvis height so
the two skeletons can be compared at different body scales; the HUD prints the
raw values.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from analyze_pose import LANK, LKNE, NOSE, Pose, RANK, RKNE, body_frame  # noqa: E402

W, H = 960, 720
TOPH, BOTH = 270, H - 270
FPS = 30


def font(size: int = 18):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--vid-dir", default=str(REPO / "videos/refs_video"))
    ap.add_argument("--refs-dir", default=str(REPO / "data/refs_video"))
    args = ap.parse_args()

    import imageio.v2 as imageio

    vid_dir = Path(args.vid_dir)
    refs_dir = Path(args.refs_dir)
    p = Pose(Path(args.npz))
    s = p.signals()
    f = font()

    for name in [x for x in args.names.split(",") if x.strip()]:
        ov_path = vid_dir / f"{name}_overlay.mp4"
        g1_path = vid_dir / f"{name}_g1.mp4"
        hud_path = vid_dir / f"{name}_hud.json"
        if not (ov_path.exists() and g1_path.exists() and hud_path.exists()):
            print(f"SKIP {name}: missing inputs")
            continue
        src = [fr for fr in imageio.mimread(ov_path, memtest=False)]
        g1 = [fr for fr in imageio.mimread(g1_path, memtest=False)]
        hud = json.loads(hud_path.read_text())
        meta = json.loads(str(np.load(refs_dir / f"{name}.npz",
                                      allow_pickle=True)["meta"]))
        n = min(len(src), len(g1), len(hud["t"]))
        t0, t1 = meta["source_window_s"]

        # source trace, sampled at the take's own fps grid
        sel = (p.t >= t0 - 0.5) & (p.t <= t1 + 1.5)
        idx = np.where(sel)[0]
        w = s["w"][idx]
        bf = body_frame(w)
        ph = bf["core"][:, 1] - bf["floor"]
        kh = np.minimum(w[:, LKNE, 1], w[:, RKNE, 1]) - bf["floor"]
        src_stand = float(np.max(ph))
        g1_ph = np.asarray(hud["pelvis_z"])
        g1_stand = float(np.max(g1_ph))
        # map video frame index -> source trace index (nearest)
        src_t = p.t[idx]
        frame_t = np.arange(n) / FPS

        def src_val(arr, tt):
            j = int(np.argmin(np.abs(src_t - tt)))
            return float(arr[j])

        out_frames = []
        for i in range(n):
            canvas = Image.new("RGB", (W, H), (12, 12, 16))
            d = ImageDraw.Draw(canvas)
            a = Image.fromarray(src[i]).resize((W // 2, TOPH))
            b = Image.fromarray(g1[i]).resize((W // 2, TOPH))
            canvas.paste(a, (0, 0))
            canvas.paste(b, (W // 2, 0))
            d.text((6, 4), f"{name} | SOURCE t={t0 + i / FPS:6.2f}s "
                           f"(skeleton overlay)", fill=(255, 255, 0), font=f)
            d.text((W // 2 + 6, 4), "G1 retarget, PD-tracked (single robot)",
                   fill=(120, 255, 120), font=f)
            # trace panel
            x0, y0, x1, y1 = 60, TOPH + 40, W - 20, H - 30
            d.rectangle([x0, y0, x1, y1], outline=(90, 90, 90))
            d.text((x0, TOPH + 8), "pelvis height (normalised to standing) — "
                                  "source solid, G1 solid | knee height dashed",
                   fill=(200, 200, 200), font=font(15))
            for frac, lab in ((0.0, "1.0"), (0.5, "0.5")):
                yy = y1 - frac * (y1 - y0) / 1.0
                d.line([x0, yy, x1, yy], fill=(60, 60, 60))
                d.text((8, yy - 8), lab, fill=(150, 150, 150), font=font(13))

            def plot(vals, colour, dash=False):
                v = np.asarray(vals[:n], dtype=float)
                pts = [(x0 + (x1 - x0) * k / max(n - 1, 1),
                        y1 - (y1 - y0) * min(max(vv, 0.0), 1.2) / 1.2)
                       for k, vv in enumerate(v)]
                if dash:
                    for k in range(0, len(pts) - 1, 6):
                        d.line([pts[k], pts[min(k + 3, len(pts) - 1)]],
                               fill=colour, width=2)
                else:
                    d.line(pts, fill=colour, width=2)

            sv = [src_val(ph, tt) for tt in frame_t]
            sk = [src_val(kh, tt) for tt in frame_t]
            plot([v / src_stand for v in sv], (255, 210, 60))
            plot([v / src_stand for v in sk], (255, 210, 60), dash=True)
            plot([v / g1_stand for v in g1_ph], (120, 255, 120))
            d.text((x0 + 8, y1 + 4),
                   f"source pelvis now {sv[i]:.2f} m (stand {src_stand:.2f})   "
                   f"G1 pelvis now {g1_ph[i]:.3f} m (stand {g1_stand:.2f})",
                   fill=(220, 220, 220), font=font(15))
            out_frames.append(np.asarray(canvas))

        out = vid_dir / f"{name}_compare_overlay.mp4"
        imageio.mimwrite(out, out_frames, fps=FPS, quality=8,
                         macro_block_size=None)
        print(f"wrote {out} ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()
