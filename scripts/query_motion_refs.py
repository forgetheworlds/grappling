#!/usr/bin/env python
"""Load and query the v1 motion references (the Agent-2 entry point).

Examples:
    .venv/bin/python scripts/query_motion_refs.py                 # all tracks
    .venv/bin/python scripts/query_motion_refs.py drill_continuous --phase
    .venv/bin/python scripts/query_motion_refs.py stance_hold --contact

Everything goes through src.solo.bc.load_reference (the SAME loader the BC
pipeline uses), proving the v1 npz files are first-class references.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solo.bc import REFERENCE_DIRS, load_reference  # noqa: E402

ALL_V1 = ["stance_hold", "stance_widen_step", "stand_to_stance", "stance_to_stand",
          "shuffle_back", "level_change_full", "level_change_fast", "shot_entry_full",
          "shot_recover", "knee_sprawl_entry", "knee_sprawl_hold", "knee_sprawl_entry2",
          "knee_sprawl_recover", "stalk_shuffle", "circle_step",
          "drill_continuous", "stance_rise"]


def show(name: str, phase: bool = False, contact: bool = False) -> bool:
    try:
        tr = load_reference(name)
    except FileNotFoundError as e:
        print(f"{name:22s} MISSING: {e}")
        return False
    meta = tr.meta or {}
    dur = float(tr.t[-1]) if len(tr.t) else 0.0
    validity = meta.get("validity", "?")
    lead = meta.get("lead_leg", "?")
    print(f"{name:22s} T={tr.qpos.shape[0]:5d}  dur={dur:6.2f}s  lead={lead:5s}  "
          f"validity={validity}")
    if phase:
        ph = meta.get("phases", [])
        print(f"  phases ({len(ph)}):")
        for p in ph:
            print(f"   {p['t_start']:7.2f}-{p['t_end']:7.2f}s {p['name']:36s} "
                  f"{p['source']:20s} lead={str(p.get('lead_leg')):4s} "
                  f"{p['validity']}")
    if contact:
        path = tr.source
        d = np.load(path, allow_pickle=True)
        if "contact" in d:
            c = d["contact"].astype(bool)
            print(f"  contact: L {c[:,0].mean():.2f}  R {c[:,1].mean():.2f} "
                  f"(fraction planted)")
    return True


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    phase = "--phase" in sys.argv
    contact = "--contact" in sys.argv
    print(f"reference dirs: {[str(p.relative_to(REPO)) for p in REFERENCE_DIRS]}")
    if args:
        ok = all(show(a, phase, contact) for a in args)
    else:
        names = ["drill_continuous", "stance_rise"] + \
                [n for n in ALL_V1 if n not in ("drill_continuous", "stance_rise",
                                                "stance_to_stand_v1")]
        ok = all(show(n, phase=False, contact=True) for n in names)
        print("\ntry: query_motion_refs.py drill_continuous --phase")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
