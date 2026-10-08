#!/usr/bin/env python3
"""Build the 7 GrappleMap->G1 reference motions (Phase 2, deliverable 5).

Writes data/refs/<technique>.npz with keys:
    qpos_a (T,36) float64  robot A (attacker/recoverer) full qpos @ 50 Hz
    qpos_b (T,36) float64  robot B (defender/opponent)
    t      (T,)   float64  seconds, uniform dt = 0.02
    technique str, edges (int array), landmark_rms float,
    meta   str (JSON: scales, weights preset, junction/alignment residuals,
                stretches, repairs, ...)
Also writes robots/wrestling_scene.xml (two-G1 scene, MjSpec-composed).

Run from repo root:  .venv/bin/python scripts/build_refs.py [--only A,B]
Every build self-checks: quaternion norms, joint limits, dt, RMS thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retarget.scene import write_scene_xml
from retarget.solve import DT_50HZ, G1Kinematics
from retarget.techniques import TECHNIQUES, build_technique

REF_DIR = Path(__file__).resolve().parents[1] / "data" / "refs"
RMS_LIMIT = 0.15  # weighted landmark RMS (m), per technique


def save_npz(result: dict, path: Path) -> None:
    np.savez_compressed(
        path,
        qpos_a=np.ascontiguousarray(result["qpos_a"], dtype=np.float64),
        qpos_b=np.ascontiguousarray(result["qpos_b"], dtype=np.float64),
        t=np.ascontiguousarray(result["t"], dtype=np.float64),
        technique=np.array(result["technique"]),
        edges=np.array(result["edges"], dtype=np.int64),
        landmark_rms=np.float64(result["landmark_rms"]),
        meta=json.dumps(result["meta"], sort_keys=True),
    )


def self_check(result: dict, kin: G1Kinematics) -> None:
    qa, qb, t = result["qpos_a"], result["qpos_b"], result["t"]
    assert qa.shape == qb.shape == (len(t), 36), qa.shape
    assert np.allclose(np.diff(t), DT_50HZ, atol=1e-9), "uniform 50 Hz dt"
    for q, name in ((qa, "a"), (qb, "b")):
        qn = np.linalg.norm(q[:, 3:7], axis=1)
        assert np.abs(qn - 1).max() < 1e-9, f"quat norm drift ({name})"
        j = q[:, 7:36]
        assert (j >= kin.joint_lo - 1e-9).all() and (j <= kin.joint_hi + 1e-9).all(), \
            f"joint limits violated ({name})"
    assert result["landmark_rms"] < RMS_LIMIT, \
        f"landmark rms {result['landmark_rms']:.3f} >= {RMS_LIMIT}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of techniques")
    args = ap.parse_args()
    techniques = (args.only.split(",") if args.only else list(TECHNIQUES))
    for tech in techniques:
        assert tech in TECHNIQUES, f"unknown technique {tech}"

    REF_DIR.mkdir(parents=True, exist_ok=True)
    scene = write_scene_xml()
    print(f"scene xml: {scene}")

    kin = G1Kinematics()
    rows = []
    for tech in techniques:
        result = build_technique(tech)
        self_check(result, kin)
        path = REF_DIR / f"{tech}.npz"
        save_npz(result, path)
        d = result["landmark_rms_detail"]
        rows.append((tech, len(result["t"]), result["meta"]["duration_s"],
                     result["meta"]["scale"], result["landmark_rms"],
                     d["unweighted"], d["max"], path.stat().st_size))
    print(f"\n{'technique':12s} {'T':>5s} {'dur_s':>6s} {'scale':>6s} "
          f"{'rms_w':>6s} {'rms_u':>6s} {'max_m':>6s} {'bytes':>9s}")
    for r in rows:
        print(f"{r[0]:12s} {r[1]:5d} {r[2]:6.2f} {r[3]:6.3f} "
              f"{r[4]:6.3f} {r[5]:6.3f} {r[6]:6.3f} {r[7]:9d}")
    print(f"\nOK: {len(rows)} reference(s) in {REF_DIR} "
          f"(self-checks passed: 50 Hz dt, quat norms, joint limits, "
          f"weighted landmark rms < {RMS_LIMIT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
