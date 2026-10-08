"""Shared frame reader: stream RGB frames from a video via ffmpeg (no re-encode).

Used by the pose/benchmark/overlay tools for the operator reference video.
No third-party deps beyond numpy (frames are raw rgb24 from ffmpeg stdout).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    nb_frames: int
    duration_s: float


def probe(path: str) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True).stdout.split()
    w, h = int(out[0]), int(out[1])
    num, den = out[2].split("/")
    fps = float(num) / float(den)
    dur = float(out[3])
    return VideoInfo(w, h, fps, int(round(dur * fps)), dur)


def frames(path: str, start_s: float = 0.0, fps_out: float | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_s, HxWx3 uint8 RGB) frames, optionally resampled to fps_out."""
    info = probe(path)
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s}"]
    cmd += ["-i", path]
    if fps_out is not None:
        cmd += ["-vf", f"fps={fps_out}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    w_size = info.width * info.height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w_size)
    assert proc.stdout is not None
    step = 1.0 / (fps_out if fps_out is not None else info.fps)
    i = 0
    while True:
        buf = proc.stdout.read(w_size)
        if len(buf) < w_size:
            break
        yield start_s + i * step, np.frombuffer(buf, dtype=np.uint8).reshape(info.height, info.width, 3)
        i += 1
    proc.stdout.close()
    proc.wait()


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    print(probe(p))
    t, f = next(frames(p))
    print("first frame", t, f.shape, f.dtype)
