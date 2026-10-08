"""Cross-check: video-derived geometry vs our existing retargeted references.

Loads data/refs/{STANCE,DOUBLE_LEG}.npz (read-only) and computes, via the G1
landmark sites, the same quantities the video spec reports: pelvis height,
lateral ankle separation (stance width), front-back ankle separation (depth),
knee height, torso pitch proxy. Prints a table for the cross-check doc.

Run: python compare_refs.py --video-spec .../stance_spec.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

from retarget.landmarks import SOLVED_SITES, load_g1_spec  # noqa: E402

import mujoco  # noqa: E402


def site_series(qpos: np.ndarray) -> dict:
    spec = load_g1_spec()
    m = spec.compile()
    d = mujoco.MjData(m)
    ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in SOLVED_SITES}
    out = {k: [] for k in ("pelvis_z", "lank", "rank", "lknee", "rknee", "head", "neck")}
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        out["pelvis_z"].append(float(q[2]))
        for k in ("lank", "rank", "lknee", "rknee", "head", "neck"):
            out[k].append(d.site_xpos[ids[{"lank": "left_ankle", "rank": "right_ankle",
                                           "lknee": "left_knee", "rknee": "right_knee",
                                           "head": "head", "neck": "neck"}[k]]].copy())
    return {k: np.asarray(v) for k, v in out.items()}


def summarize(name: str, qpos: np.ndarray, t: np.ndarray) -> dict:
    s = site_series(qpos)
    d = s["lank"] - s["rank"]
    width = np.abs(d[:, 1])
    depth = np.abs(d[:, 0])
    knee_h = np.minimum(s["lknee"][:, 2], s["rknee"][:, 2])
    torso = s["head"] - 0.5 * (s["lank"] + s["rank"])
    pitch = np.degrees(np.arccos(np.clip(
        (s["neck"] - s["head"])[:, 2] / np.maximum(np.linalg.norm(s["neck"] - s["head"], axis=1), 1e-9), -1, 1)))
    return {
        "name": name, "dur_s": round(float(t[-1]), 2),
        "pelvis_z_min": round(float(s["pelvis_z"].min()), 3),
        "pelvis_z_max": round(float(s["pelvis_z"].max()), 3),
        "pelvis_z_end": round(float(s["pelvis_z"][-1]), 3),
        "width_med": round(float(np.median(width)), 3),
        "width_max": round(float(width.max()), 3),
        "depth_med": round(float(np.median(depth)), 3),
        "depth_max": round(float(depth.max()), 3),
        "knee_min": round(float(knee_h.min()), 3),
        "knee_at_end": round(float(knee_h[-1]), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs-dir", default=str(REPO / "data/refs"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows = []
    for name in ("STANCE", "DOUBLE_LEG"):
        p = Path(args.refs_dir) / f"{name}.npz"
        if not p.exists():
            print("missing", p)
            continue
        d = np.load(p, allow_pickle=True)
        rows.append(summarize(name, d["qpos_a"], d["t"]))
        for r in rows[-1:]:
            print(f"{r['name']:12s} dur={r['dur_s']:5.2f}s pelvis_z {r['pelvis_z_min']:.3f}"
                  f"-{r['pelvis_z_max']:.3f} (end {r['pelvis_z_end']:.3f})  width med/max "
                  f"{r['width_med']:.3f}/{r['width_max']:.3f}  depth med/max "
                  f"{r['depth_med']:.3f}/{r['depth_max']:.3f}  knee min {r['knee_min']:.3f} "
                  f"(end {r['knee_at_end']:.3f})")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
