"""Benchmark MediaPipe PoseLandmarker (lite/full/heavy) on the ref clip, CPU-only.

Usage: python bench_pose.py <clip.mp4> [model ...]
Reports ms/frame and fps per model over the whole clip.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import mediapipe as mp  # noqa: E402  (must import before tasks)
from mediapipe.tasks.python import vision  # noqa: E402

from vidframe import frames  # noqa: E402

MODELS = {
    "lite": "/home/ubuntu/venvs/pose/models/pose_landmarker_lite.task",
    "full": "/home/ubuntu/venvs/pose/models/pose_landmarker_full.task",
    "heavy": "/home/ubuntu/venvs/pose/models/pose_landmarker_heavy.task",
}


def bench(path: str, model: str, max_frames: int = 0) -> float:
    opts = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODELS[model]),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        output_segmentation_masks=False,
    )
    lm = vision.PoseLandmarker.create_from_options(opts)
    n = 0
    t0 = time.perf_counter()
    detected = 0
    with lm:
        for ts, rgb in frames(path):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = lm.detect_for_video(img, int(ts * 1000))
            n += 1
            detected += 1 if res.pose_landmarks else 0
            if max_frames and n >= max_frames:
                break
    dt = time.perf_counter() - t0
    print(f"{model:6s} frames={n:5d} detected={detected:5d} "
          f"ms/frame={1000*dt/n:7.2f} fps={n/dt:6.2f}")
    return n / dt


if __name__ == "__main__":
    clip = sys.argv[1]
    for m in sys.argv[2:] or ["lite", "full"]:
        bench(clip, m)
