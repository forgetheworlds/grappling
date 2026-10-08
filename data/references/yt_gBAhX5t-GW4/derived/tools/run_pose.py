"""Pose pass over the operator reference video (whole video, fixed stride).

MediaPipe Tasks PoseLandmarker, CPU-only. Reads ref720h264.mp4 via an ffmpeg
raw-video pipe at an effective 15 fps (every 2nd source frame at 30 fps) and
stores per-frame landmarks (image + world) with confidences.

Outputs (data/references/yt_gBAhX5t-GW4/pose/):
  landmarks.npz : t (F,), img_lm (F,33,3) normalized, world_lm (F,33,3) m,
                  visibility (F,33), presence (F,33), detected (F,), conf (F,),
                  chapter (F,) int   (1..10 from the author chapter map)
  index.json    : provenance: model, stride, fps, timestamps, counts, gaps.

Missing detections are stored as NaN and flagged (detected=False); nothing is
interpolated.

Run (venv with mediapipe, from repo root):
  /home/ubuntu/venvs/pose/bin/python \
    data/references/yt_gBAhX5t-GW4/derived/tools/run_pose.py [--fps 15] [--model full]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

from vidframe import frames, probe  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
VIDEO = REPO / "data/references/yt_gBAhX5t-GW4/ref720h264.mp4"
OUT = REPO / "data/references/yt_gBAhX5t-GW4/pose"
MODEL_DIR = Path("/home/ubuntu/venvs/pose/models")

# author chapter map (docs/REFERENCES.md §2)
CHAPTERS = [
    ("intro", 0.0, 26.0), ("stance", 26.0, 83.0), ("stalking", 83.0, 133.0),
    ("circling", 133.0, 202.0), ("level_change", 202.0, 261.0),
    ("fake", 261.0, 400.0), ("shots", 400.0, 549.0), ("downblock", 549.0, 599.0),
    ("pepsi", 599.0, 662.0), ("knee_sprawl", 662.0, 823.5),
]


def chapter_of(t: float) -> int:
    for i, (_, a, b) in enumerate(CHAPTERS, 1):
        if a <= t < b:
            return i
    return len(CHAPTERS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=15.0, help="effective sampling fps")
    ap.add_argument("--model", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--segments", default="",
                    help="comma list of 't0:t1' windows (absolute source seconds); "
                         "empty = whole video")
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe(args.video)
    if args.segments:
        segs = [tuple(float(x) for x in s.split(":"))
                for s in args.segments.split(",") if s.strip()]
    else:
        segs = [(0.0, info.duration_s)]

    opts = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_DIR / f"pose_landmarker_{args.model}.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        output_segmentation_masks=False,
    )

    ts_out, img_lm, world_lm, vis, pres, det, seg_out = [], [], [], [], [], [], []
    t0 = time.perf_counter()
    n = 0
    for si, (s0, s1) in enumerate(segs):
        with vision.PoseLandmarker.create_from_options(opts) as lm:
            for ts, rgb in frames(args.video, start_s=s0, fps_out=args.fps):
                if ts >= s1:
                    break
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = lm.detect_for_video(img, int(round(ts * 1000)))
                ts_out.append(ts)
                seg_out.append(si)
                if res.pose_landmarks:
                    pl = res.pose_landmarks[0]
                    wl = res.pose_world_landmarks[0]
                    img_lm.append([[p.x, p.y, p.z] for p in pl])
                    world_lm.append([[p.x, p.y, p.z] for p in wl])
                    vis.append([p.visibility for p in pl])
                    pres.append([p.presence for p in pl])
                    det.append(True)
                else:
                    img_lm.append(np.full((33, 3), np.nan))
                    world_lm.append(np.full((33, 3), np.nan))
                    vis.append(np.zeros(33))
                    pres.append(np.zeros(33))
                    det.append(False)
                n += 1
                if n % 1000 == 0:
                    el = time.perf_counter() - t0
                    print(f"  {n:6d} frames  {el:7.1f}s  {n/el:5.1f} fps  "
                          f"seg{si} t={ts:7.1f}s", flush=True)
    elapsed = time.perf_counter() - t0

    ts_a = np.asarray(ts_out)
    img_a = np.asarray(img_lm)
    world_a = np.asarray(world_lm)
    vis_a = np.asarray(vis)
    pres_a = np.asarray(pres)
    det_a = np.asarray(det)
    conf_a = np.where(det_a, np.minimum(vis_a, pres_a).mean(axis=1), 0.0)

    np.savez_compressed(
        out_dir / "landmarks.npz",
        t=ts_a, img_lm=img_a.astype(np.float32), world_lm=world_a.astype(np.float32),
        visibility=vis_a.astype(np.float32), presence=pres_a.astype(np.float32),
        detected=det_a, conf=conf_a.astype(np.float32),
        chapter=np.array([chapter_of(float(t)) for t in ts_a], dtype=np.int64),
        segment=np.asarray(seg_out, dtype=np.int64),
    )

    idx = {
        "source": str(args.video),
        "source_size": [info.width, info.height],
        "source_fps": info.fps,
        "source_duration_s": info.duration_s,
        "estimator": f"mediapipe {mp.__version__} tasks PoseLandmarker ({args.model})",
        "running_mode": "VIDEO",
        "effective_fps": args.fps,
        "stride_source_frames": info.fps / args.fps,
        "n_frames": int(n),
        "segments": [{"i": i, "t0": float(a), "t1": float(b)}
                     for i, (a, b) in enumerate(segs)],
        "n_detected": int(det_a.sum()),
        "n_missing": int((~det_a).sum()),
        "mean_conf": float(conf_a[det_a].mean()) if det_a.any() else 0.0,
        "wall_s": round(elapsed, 1),
        "measured_fps": round(n / elapsed, 2),
        "landmark_index": "mediapipe pose 33-keypoint order (see mediapipe docs)",
        "arrays": ["t", "img_lm", "world_lm", "visibility", "presence",
                   "detected", "conf", "chapter"],
        "chapters": [{"id": i, "name": nm, "t0": a, "t1": b}
                     for i, (nm, a, b) in enumerate(CHAPTERS, 1)],
        "missing_policy": "stored as NaN, flagged detected=False; never interpolated",
    }
    (out_dir / "index.json").write_text(json.dumps(idx, indent=2))
    print(f"saved {out_dir/'landmarks.npz'}  frames={n}  detected={det_a.sum()} "
          f"missing={(~det_a).sum()}  {n/elapsed:.1f} fps wall")
    assert n > 0 and len(ts_a) == len(img_a) == len(world_a)
    assert np.all(np.diff(ts_a) > 0), "timestamps must be monotonic"


if __name__ == "__main__":
    main()
