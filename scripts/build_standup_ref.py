#!/usr/bin/env python3
"""FIX 3: rebuild the STAND_UP reference so it is grounded and continuous.

The shipped STAND_UP montage ([952, 1110, 1125], three unrelated stand-up
edges joined by *rigid* pair alignment, junction rms 0.50 / 0.72 m) comes out
airborne: from t ≈ 4 s the lowest foot site is 0.13-0.72 m above the mat and
the pelvis is 1.0-1.08 m (the G1 stands at 0.79 m).  A per-frame vertical
re-grounding does not repair it (the montage also contorts the torso).

This script rebuilds the technique from a **shared-node chain** of GrappleMap
edges — junctions that need no alignment (rms = 0 by construction, the graph's
own `from_reo`/`to_reo` composition), i.e. one continuous motion:

    t1125  'bottom gets to knees'   'elbow half w/ deep underhook vs whizzer'
                                     -> 'dogfight w/o leg control'
    t380   'stand up'                'dogfight w/o leg control'
                                     -> 'standing w/ whizzer'
    t540   'stand up further'        'standing w/ whizzer'
                                     -> 'standing over+under vs tricep+under'

The recoverer (robot A) is the *bottom* player of the chain (the one with the
lower core at the first frame); roles follow the chain's own player swaps.

The rebuilt npz is written with the same keys as the other references and the
previous file is kept as ``STAND_UP.airborne.npz``.  Every claim is measured:
per-frame minimum foot-site height (both robots), pelvis height range, torso
tilt, junction residuals, and the landmark rms of the solve.

Run:
  PYTHONPATH=src .venv/bin/python scripts/build_standup_ref.py [--check-only]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from retarget.gmframe import CORE, assemble_chain  # noqa: E402
from retarget.landmarks import gm_weights  # noqa: E402
from retarget.solve import (DT_50HZ, G1Kinematics, landmark_rms_final,  # noqa: E402
                            resample_50hz, solve_keyframes)
from retarget.techniques import get_graph  # noqa: E402
from retarget.world import place_world  # noqa: E402

REF = REPO / "data" / "refs" / "STAND_UP.npz"
BACKUP = REPO / "data" / "refs" / "STAND_UP.airborne.npz"
CHAIN = [1125, 380, 540]          # bottom gets to knees -> stand up -> further
TRANSITION_DT = 0.5
RMS_LIMIT = 0.15


def build() -> dict:
    g = get_graph()
    chain = assemble_chain(g, CHAIN, transition_dt=TRANSITION_DT,
                           w=gm_weights("default"))
    frames = chain.frames                       # (F, 2, 23, 3) gm, Y-up
    # robot A = the recoverer = the *bottom* player at the first frame
    first = frames[0]
    bottom = int(np.argmin([first[p, CORE, 1] for p in range(2)]))
    role_raw = np.full(len(chain.frame_edges), bottom, dtype=np.int64)
    swap = chain.world_swap.astype(np.int64)
    idx_a = role_raw ^ swap
    rows = np.arange(len(frames))
    ta = frames[rows, idx_a]
    tb = frames[rows, (1 - role_raw) ^ swap]
    info = {"junctions": chain.junctions, "chain": CHAIN,
            "role_raw": int(bottom), "world_swap": swap.tolist()}
    placement = place_world(ta, tb)
    solved = solve_keyframes(placement.targets_a, placement.targets_b,
                             "default", t_kf=chain.times)
    t_grid, qa, qb, stretch = resample_50hz(chain.times, solved["qpos_a"],
                                            solved["qpos_b"])
    rms = landmark_rms_final(qa, qb, placement.targets_a, placement.targets_b,
                             chain.times * stretch, t_grid, "default")
    meta = {
        "preset": "default", "rebuild": "shared-node chain (fix 3)",
        "roles": "A=recoverer (bottom player), B=partner",
        "scale": placement.scale, "heading": placement.heading,
        "junctions": chain.junctions, "keyframes": int(len(chain.times)),
        "time_stretch_kinematic": float(stretch),
        "duration_s": float(t_grid[-1]), "dt": DT_50HZ,
        "source": "third_party/GrappleMap (shared-node chain t1125->t380->t540)",
    }
    return {"technique": "STAND_UP", "edges": list(CHAIN), "qpos_a": qa,
            "qpos_b": qb, "t": t_grid, "landmark_rms": rms["weighted"],
            "landmark_rms_detail": rms, "meta": meta, "info": info}


def foot_heights(qpos: np.ndarray, kin: G1Kinematics) -> dict:
    """Min sole-site height / pelvis height / torso tilt per frame."""
    from retarget.landmarks import SOLVED_SITES
    idx = {n: SOLVED_SITES.index(n) for n in
           ("left_toe", "left_heel", "right_toe", "right_heel", "core", "neck")}
    mins, pelvis, tilt = [], [], []
    for q in qpos:
        s = kin.sites(q)
        mins.append(min(float(s[idx[n]][2]) for n in
                        ("left_toe", "left_heel", "right_toe", "right_heel")))
        pelvis.append(float(s[idx["core"]][2]))
        v = s[idx["neck"]] - s[idx["core"]]
        tilt.append(float(np.degrees(np.arccos(
            np.clip(v[2] / max(np.linalg.norm(v), 1e-9), -1, 1)))))
    return {"min_foot_z": np.array(mins), "pelvis_z": np.array(pelvis),
            "tilt_deg": np.array(tilt)}


def ground_pair(qa: np.ndarray, qb: np.ndarray, kin: G1Kinematics,
                floor: float = -0.01, win: int = 15) -> dict:
    """Lift the pair rigidly (both robots together) so no foot site sinks
    below ``floor``; the correction is smoothed (zero-phase, edge-padded) so
    the vertical motion stays continuous.  Inter-player relationships are
    preserved because both robots move by the same amount per frame.

    Returns the corrected arrays + a report of what was applied.
    """
    from retarget.landmarks import SOLVED_SITES
    idx = [SOLVED_SITES.index(n) for n in
           ("left_toe", "left_heel", "right_toe", "right_heel")]
    mins = np.empty(len(qa))
    for i, (a, b) in enumerate(zip(qa, qb)):
        sa, sb = kin.sites(a), kin.sites(b)
        mins[i] = min(min(float(sa[j][2]) for j in idx),
                      min(float(sb[j][2]) for j in idx))
    raw = np.maximum(0.0, -(mins - floor))
    if win > 1:
        n = int(win)
        pad = n // 2
        pad_vals = np.concatenate([np.full(pad, raw[0]), raw,
                                   np.full(n - 1 - pad, raw[-1])])
        shift = np.convolve(pad_vals, np.ones(n) / n, mode="valid")
    else:
        shift = raw
    qa = qa.copy()
    qb = qb.copy()
    qa[:, 2] += shift
    qb[:, 2] += shift
    return {"qa": qa, "qb": qb,
            "min_before": float(mins.min()),
            "min_after_raw": float((mins + raw).min()),
            "shift_max": float(shift.max()),
            "shift_mean": float(shift.mean()),
            "frames_shifted": int((shift > 1e-4).sum())}


def report(name: str, res: dict, kin: G1Kinematics) -> None:
    for pfx, key in (("a", "qpos_a"), ("b", "qpos_b")):
        h = foot_heights(res[key], kin)
        airborne = float((h["min_foot_z"] > 0.05).mean())
        print(f"  {name} robot {pfx}: min foot z [{h['min_foot_z'].min():.3f},"
              f" {h['min_foot_z'].max():.3f}]  airborne frames "
              f"{airborne * 100:.0f}%  pelvis z [{h['pelvis_z'].min():.3f},"
              f" {h['pelvis_z'].max():.3f}]  tilt max {h['tilt_deg'].max():.1f} deg")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    kin = G1Kinematics()
    if args.check_only:
        z = np.load(REF, allow_pickle=True)
        print("current STAND_UP:", z["edges"], str(z["meta"])[:120])
        report("shipped", {k: z[k] for k in ("qpos_a", "qpos_b")}, kin)
        return 0
    res = build()
    assert res["landmark_rms"] < RMS_LIMIT, res["landmark_rms"]
    g = ground_pair(res["qpos_a"], res["qpos_b"], kin)
    g2 = ground_pair(g["qa"], g["qb"], kin)
    res["qpos_a"], res["qpos_b"] = g2["qa"], g2["qb"]
    res["meta"]["ground_repair"] = {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in g.items() if k not in ("qa", "qb")}
    res["meta"]["ground_repair_pass2"] = {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in g2.items() if k not in ("qa", "qb")}
    print("ground repair:", res["meta"]["ground_repair"])
    if REF.exists() and not BACKUP.exists():
        shutil.copy2(REF, BACKUP)
        print("kept the old file as", BACKUP)
    np.savez_compressed(
        REF, qpos_a=res["qpos_a"], qpos_b=res["qpos_b"], t=res["t"],
        technique=np.array(res["technique"]), edges=np.array(res["edges"]),
        landmark_rms=np.float64(res["landmark_rms"]),
        meta=json.dumps(res["meta"], sort_keys=True))
    print("wrote", REF)
    print(f"  duration {res['meta']['duration_s']:.2f} s, "
          f"{len(res['t'])} frames, landmark rms {res['landmark_rms']:.4f}")
    print("  junctions:", [(j["kind"], round(j["rms"], 3))
                           for j in res["meta"]["junctions"]])
    report("rebuilt", res, kin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
