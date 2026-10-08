#!/usr/bin/env .venv/bin/python
"""Score a reference trace (or any trace in the same npz contract) with the
technique-validity scorer (deliverable 7).

Trace contract (notes.md "Interface contracts"): ``.npz`` with ``qpos_a`` and
``qpos_b`` of shape (T, 36) — robot A = attacker/recoverer, robot B =
defender/opponent. ``data/refs/*.npz`` satisfy it; rollouts saved in the same
layout can be scored by path.

Run from repo root::

    .venv/bin/python scripts/score_trace.py                     # all refs
    .venv/bin/python scripts/score_trace.py --technique SPRAWL  # one ref
    .venv/bin/python scripts/score_trace.py --technique DOUBLE_LEG --as SINGLE_LEG
    .venv/bin/python scripts/score_trace.py --npz out/rollout.npz --as DOUBLE_LEG

Prints, per phase: frame count, mean score, minimum frame score, and the
predicates whose mean membership is below 0.9 (what dragged the phase down).
Self-check: every reported score is inside [0, 1] and the command exits non-zero
when a trace fails the 0.8 per-phase conformity bar it was asked to meet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402

from scorer import TechniqueScorer, TECHNIQUE_CONFIG, TECHNIQUES, executor_of  # noqa: E402

SCENE = REPO / "robots" / "wrestling_scene.xml"
REF_DIR = REPO / "data" / "refs"
SELF_MIN = 0.8
WORST_BELOW = 0.9


def load_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    missing = [k for k in ("qpos_a", "qpos_b") if k not in d]
    if missing:
        raise SystemExit(f"{path}: missing keys {missing} (need qpos_a, qpos_b)")
    a, b = np.asarray(d["qpos_a"], dtype=np.float64), np.asarray(d["qpos_b"], dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 36:
        raise SystemExit(f"{path}: expected both qpos arrays (T, 36), got "
                         f"{a.shape} and {b.shape}")
    return a, b


def score_one(scorer: TechniqueScorer, technique: str, qpos_a: np.ndarray,
              qpos_b: np.ndarray) -> dict:
    """Per-phase table for one trace scored as ``technique``."""
    self_t, opp_t = (qpos_a, qpos_b) if executor_of(technique) == "A" else (qpos_b, qpos_a)
    T = len(self_t)
    phases = TECHNIQUE_CONFIG[technique]["phases"]
    rows = []
    for i in range(T):
        ph = scorer.phase_at_frac(technique, i / max(T - 1, 1))
        r = scorer.score(technique, ph, self_t[i], opp_t[i])
        rows.append((ph, r.total, r.terms))
    out = []
    for p, (name, t0, t1) in enumerate(phases):
        sel = [r for r in rows if r[0] == p]
        if not sel:
            out.append({"phase": name, "frames": 0, "mean": float("nan"),
                        "min": float("nan"), "worst": []})
            continue
        totals = np.array([s[1] for s in sel])
        agg: dict[str, list[float]] = {}
        for _p, _t, terms in sel:
            for k, (mu, _w) in terms.items():
                agg.setdefault(k, []).append(mu)
        worst = sorted(((float(np.mean(v)), k) for k, v in agg.items()))[:3]
        out.append({"phase": name, "frames": len(sel), "mean": float(totals.mean()),
                    "min": float(totals.min()),
                    "worst": [(k, v) for v, k in worst if v < WORST_BELOW]})
    return {"technique": technique, "executor": executor_of(technique),
            "frames": T, "duration_s": (T - 1) * 0.02, "phases": out,
            "mean": float(np.mean([r[1] for r in rows])),
            "min_frame": float(np.min([r[1] for r in rows]))}


def print_trace(res: dict, source: str, own: bool) -> bool:
    """Print one trace table; returns False when an own-trace phase fails 0.8.

    ``own`` distinguishes "this trace is the reference for the technique being
    scored" (held to the >= 0.8 calibrated bar) from a deliberate cross-score
    (informational: a low score there is the discrimination working).
    """
    ok = True
    tag = "own reference" if own else "cross-score (informational)"
    print(f"\n{res['technique']}  ({source})  [{tag}]")
    print(f"  executor={res['executor']}  frames={res['frames']}  "
          f"duration={res['duration_s']:.2f}s")
    print(f"  {'phase':10s} {'frames':>6s} {'mean':>7s} {'min':>7s}  "
          f"predicates < {WORST_BELOW}")
    for row in res["phases"]:
        worst = ", ".join(f"{k}={v:.2f}" for k, v in row["worst"]) or "-"
        print(f"  {row['phase']:10s} {row['frames']:6d} {row['mean']:7.3f} "
              f"{row['min']:7.3f}  {worst}")
        if own and row["frames"] and row["mean"] < SELF_MIN:
            ok = False
    print(f"  {'OVERALL':10s} {res['frames']:6d} {res['mean']:7.3f} "
          f"{res['min_frame']:7.3f}")
    for p in res["phases"]:
        if p["frames"] and not (0.0 <= p["mean"] <= 1.0):
            raise AssertionError(f"score out of range: {p}")
    return ok


def _display(path: Path) -> str:
    """Repo-relative path when possible (traces may live outside the repo)."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--technique", "-t", choices=TECHNIQUES, action="append",
                    help="reference technique(s) to score (default: all)")
    ap.add_argument("--as", dest="as_technique", choices=TECHNIQUES,
                    help="score the trace as a different technique family")
    ap.add_argument("--npz", type=Path, help="score an arbitrary trace npz "
                    "(needs --as to pick the family)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    scorer = TechniqueScorer(model)

    jobs: list[tuple[str, Path]] = []
    if args.npz is not None:
        if args.as_technique is None:
            raise SystemExit("--npz requires --as <TECHNIQUE> (the commanded family)")
        jobs.append((args.as_technique, args.npz))
    else:
        techs = args.technique or list(TECHNIQUES)
        jobs = [(args.as_technique or t, REF_DIR / f"{t}.npz") for t in techs]

    all_ok = True
    scored = 0
    for technique, path in jobs:
        if not path.exists():
            raise SystemExit(f"no such trace: {path}")
        a, b = load_trace(path)
        res = score_one(scorer, technique, a, b)
        own = path.stem == technique
        all_ok &= print_trace(res, _display(path), own)
        scored += 1
    print(f"\nRESULT: {'OK' if all_ok else f'BELOW {SELF_MIN} ON SOME OWN-TRACE PHASE'}"
          f" ({scored} trace(s) scored)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
