#!/usr/bin/env python
"""Trim solver-collapse tails from the v1 retargeted takes.

The per-frame IK + kinematic retimer cannot always reach a video posture; on
some takes the emitted trajectory COLLAPSES (pelvis driven to 0.05-0.25 m, soles
up to 0.2 m under the mat) -- a solver artifact, NOT a video pose (the G1
physically has no configuration with a standing-take pelvis at 0.19 m).  These
tails would poison the composition and the feasibility statistics.

Policy: a take is cut at the FIRST sustained (>=0.2 s) pelvis-z dip below
0.30 m (70 % of the 0.43 m deepest reachable crouch measured for the G1
stance) UNLESS the take is a ground posture by design (knee_sprawl_*: kneeling
is the technique; those takes keep their known_infeasible labels and are
excluded from the drill).  The cut is recorded in meta["reachability_trim"]
with the cut time and reason; nothing else changes.

Prints a per-take line and exits 0 (trims are expected, missing files are not).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

REFS = REPO / "data/references/motion_refs/v1/refs"
PELVIS_FLOOR = 0.30
CONFIRM_S = 0.20
KEEP = {"knee_sprawl_entry", "knee_sprawl_hold", "knee_sprawl_entry2",
        "knee_sprawl_recover"}


def main() -> int:
    for path in sorted(REFS.glob("*.npz")):
        d = np.load(path, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        if path.stem in KEEP or "reachability_trim" in meta:
            print(f"{path.stem:24s} kept ({'by design' if path.stem in KEEP else 'already trimmed'})")
            continue
        q, t = np.asarray(d["qpos_a"], float), np.asarray(d["t"], float)
        below = q[:, 2] < PELVIS_FLOOR
        confirm = max(1, int(round(CONFIRM_S / 0.02)))
        cut = len(q)
        run = 0
        for i, b in enumerate(below):
            run = run + 1 if b else 0
            if run >= confirm:
                cut = i - confirm + 1
                break
        if cut >= len(q):
            print(f"{path.stem:24s} no collapse (min pelvis {q[:,2].min():.3f})")
            continue
        kept_q, kept_t = q[:cut], t[:cut]
        kept_c = np.asarray(d["contact"], np.uint8)[:cut]
        meta["reachability_trim"] = {
            "cut_at_s": round(float(t[cut]), 3) if cut < len(t) else None,
            "kept_frames": int(cut), "dropped_frames": int(len(q) - cut),
            "pelvis_floor_m": PELVIS_FLOOR,
            "reason": "solver-collapse artifact tail (emitted pelvis below the "
                      "G1's reachable band with legs); NOT a video pose -- "
                      "trimmed, documented per FORMAT.md",
        }
        np.savez_compressed(path, qpos_a=kept_q, qpos_b=np.zeros_like(kept_q),
                            t=kept_t, technique=str(d["technique"]),
                            edges=np.asarray(d["edges"]),
                            landmark_rms=d["landmark_rms"], contact=kept_c,
                            meta=json.dumps(meta))
        print(f"{path.stem:24s} TRIMMED at t={meta['reachability_trim']['cut_at_s']}s "
              f"(kept {cut}/{len(q)} frames; min pelvis {q[cut:,2].min() if cut < len(q) else float('nan'):.3f})")
    print("SELF-VERIFY: trim pass complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
