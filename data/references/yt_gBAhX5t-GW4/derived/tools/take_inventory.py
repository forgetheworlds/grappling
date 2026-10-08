"""Per-take content inventory: which take actually contains what.

For every detected take in the requested chapters, prints: duration, mean
confidence, min pelvis height, min knee height, counts of knee-near-mat
frames, crouch frames, and step events (foot lifts). Used to pick the take per
chapter by CONTENT (a clean stance hold, a deep level change, knee-down
entries), not just by landmark confidence.

Usage: python take_inventory.py --npz .../pose/landmarks.npz --chapters 2,5,7,10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from analyze_pose import LANK, Pose, RANK, detect_takes, merge_takes  # noqa: E402
from measure_takes import body_frame  # noqa: E402


def inventory(p: Pose, s: dict, tk: dict) -> dict:
    i0, i1 = tk["i0"], tk["i1"]
    w_loc = s["w"][i0:i1 + 1]
    bf = body_frame(w_loc)
    ph = bf["core"][:, 1] - bf["floor"]
    knee = np.minimum(w_loc[:, 25, 1], w_loc[:, 26, 1]) - bf["floor"]
    stance_ph = float(np.percentile(ph, 75))
    lah = w_loc[:, LANK, 1] - bf["floor"]
    rah = w_loc[:, RANK, 1] - bf["floor"]
    lifts = int(((lah > 0.08).astype(int).sum() + (rah > 0.08).astype(int).sum()))
    dt = float(np.median(np.diff(p.t)))
    return {
        "t0": round(tk["t0"], 2), "t1": round(tk["t1"], 2),
        "dur_s": round(tk["t1"] - tk["t0"], 2),
        "conf": round(tk["mean_conf"], 3),
        "pelvis_min_m": round(float(ph.min()), 3),
        "pelvis_max_m": round(float(ph.max()), 3),
        "knee_min_m": round(float(knee.min()), 3),
        "knee_near_mat_frames": int((knee < 0.15).sum()),
        "knee_near_mat_s": round(float((knee < 0.15).sum() * dt), 2),
        "crouch_frames": int((ph < stance_ph - 0.12).sum()),
        "crouch_s": round(float((ph < stance_ph - 0.12).sum() * dt), 2),
        "foot_high_frames": lifts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--chapters", default="2,5,7,10")
    ap.add_argument("--speed-floor", type=float, default=0.25)
    args = ap.parse_args()

    p = Pose(Path(args.npz))
    s = p.signals()
    for ch in [int(x) for x in args.chapters.split(",") if x.strip()]:
        takes = merge_takes(detect_takes(p, s, ch, speed_floor=args.speed_floor))
        print(f"== chapter {ch} ==")
        rows = sorted((inventory(p, s, tk) for tk in takes),
                      key=lambda r: -r["knee_near_mat_s"])
        for r in rows:
            print("  " + "  ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
