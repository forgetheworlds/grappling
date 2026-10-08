"""Diagnostic: print a compact numeric trace of a window (for threshold design).

Usage: python trace_take.py --npz ... --t0 25 --t1 36 [--step 0.2]
Columns: t, pelvis_h(m), knee_min_h(m), foot heights (L/R above floor),
ankle sep width/depth (m), pelvis_v (m/s), speed_lower (m/s).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from analyze_pose import LANK, LKNE, RANK, RKNE, Pose  # noqa: E402
from measure_takes import body_frame, proj  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--step", type=float, default=0.2)
    args = ap.parse_args()

    p = Pose(Path(args.npz))
    s = p.signals()
    w = s["w"]
    sel = (p.t >= args.t0) & (p.t <= args.t1)
    idx = np.where(sel)[0]
    w_loc = w[idx]
    bf = body_frame(w_loc)
    core = bf["core"]
    ph = core[:, 1] - bf["floor"]
    kh = np.minimum(w_loc[:, LKNE, 1], w_loc[:, RKNE, 1]) - bf["floor"]
    lah = w_loc[:, LANK, 1] - bf["floor"]
    rah = w_loc[:, RANK, 1] - bf["floor"]
    la = proj(bf, w_loc[:, LANK])
    ra = proj(bf, w_loc[:, RANK])
    width = np.abs(la[:, 1] - ra[:, 1])
    depth = np.abs(la[:, 0] - ra[:, 0])
    t = p.t[idx]
    stride = max(1, int(round(args.step / float(np.median(np.diff(t))))))
    print("t      pelv_h knee_h  Lfoot  Rfoot  width  depth   pelv_v  spd_lo  lbl")
    for k in range(0, len(idx), stride):
        gi = idx[k]
        print(f"{t[k]:7.2f} {ph[k]:6.3f} {kh[k]:6.3f} {lah[k]:6.3f} {rah[k]:6.3f} "
              f"{width[k]:6.3f} {depth[k]:6.3f} {s['pelvis_v'][gi]:7.2f} "
              f"{s['speed_lower'][gi]:6.2f}")


if __name__ == "__main__":
    main()
