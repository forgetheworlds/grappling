#!/usr/bin/env .venv/bin/python
"""Calibrate the technique-validity scorer's predicate thresholds (deliverable 7).

Reads every reference trace in ``data/refs/*.npz``, extracts the scorer's
relational features over each (technique, phase) band, and fits the fuzzy
membership parameters with the rule in ``src/scorer/calibration.py``. Writes
``data/scorer_calibration.json`` (``--write``) and prints the calibration table
plus the validation matrices.

Run from repo root::

    .venv/bin/python scripts/calibrate_scorer.py            # verify + report
    .venv/bin/python scripts/calibrate_scorer.py --write    # regenerate params

Verification (always, exit 1 on failure):
  * phase-band provenance: every band boundary of a non-montage technique sits
    on a reconstructed keyframe time (times * requested stretch * kinematic
    stretch from the npz meta) within 0.03 s;
  * self-conformity: every technique scores >= 0.8 mean on ITS OWN trace in
    EVERY phase (target ~1.0);
  * cross-conformity: the DOUBLE_LEG trace scored as SINGLE_LEG averages
    < 0.5, and the cross matrix is printed for the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402

from retarget.techniques import build_technique_targets  # noqa: E402
from scorer import calibration as cal  # noqa: E402
from scorer.config import PARAMS_PATH, TECHNIQUE_CONFIG, TECHNIQUE_SPEC, executor_of  # noqa: E402
from scorer.features import FEAT_ORDER, Featurizer  # noqa: E402
from scorer.scorer import TechniqueScorer  # noqa: E402

REF_DIR = REPO / "data" / "refs"
SCENE = REPO / "robots" / "wrestling_scene.xml"
TECHNIQUES = tuple(TECHNIQUE_SPEC)
SELF_MIN = 0.8           # acceptance: per-phase mean on own trace
CROSS_MAX = {"DOUBLE_LEG": {"SINGLE_LEG": 0.5}}   # acceptance: specific cross pair


def load_trace(tech: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """(qpos_executor, qpos_opponent, meta) for one reference technique."""
    d = np.load(REF_DIR / f"{tech}.npz", allow_pickle=False)
    a, b = d["qpos_a"], d["qpos_b"]
    meta = json.loads(str(d["meta"]))
    return (a, b, meta) if executor_of(tech) == "A" else (b, a, meta)


def phase_index(scorer: TechniqueScorer, tech: str, t: int, T: int) -> int:
    return scorer.phase_at_frac(tech, t / max(T - 1, 1))


def features_by_phase(scorer: TechniqueScorer, tech: str,
                      self_t: np.ndarray, opp_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(F[T, n_feat], phase[T]) for one trace."""
    f = scorer._feat
    T = len(self_t)
    F = np.array([f.features(self_t[i], opp_t[i]) for i in range(T)])
    ph = np.array([phase_index(scorer, tech, i, T) for i in range(T)])
    return F, ph


def fit_all(scorer: TechniqueScorer) -> tuple[dict, dict]:
    """(params, diagnostics) fitted from every reference trace."""
    fidx = {n: i for i, n in enumerate(FEAT_ORDER)}
    params: dict = {}
    diag: dict = {}
    for tech in TECHNIQUES:
        self_t, opp_t, _ = load_trace(tech)
        F, ph = features_by_phase(scorer, tech, self_t, opp_t)
        params[tech] = {}
        diag[tech] = {}
        for p, tmpl in TECHNIQUE_SPEC[tech]["templates"].items():
            sel = F[ph == p]
            if sel.shape[0] == 0:
                raise RuntimeError(f"{tech} phase {p}: empty band")
            table = {}
            stats = {}
            for feat, kind, _w in tmpl:
                x = sel[:, fidx[feat]]
                prm = cal.fit_params(kind, feat, x)
                table[f"{feat}:{kind}"] = cal.params_to_list(prm)
                stats[f"{feat}:{kind}"] = {
                    "p10": round(float(np.percentile(x, cal.CORE_PCT)), 4),
                    "p50": round(float(np.median(x)), 4),
                    "p90": round(float(np.percentile(x, 100 - cal.CORE_PCT)), 4),
                    "n": int(x.size),
                }
            params[tech][str(p)] = table
            diag[tech][str(p)] = stats
    return params, diag


def check_bands(verbose: bool = True) -> list[str]:
    """Phase bands vs reconstructed keyframe times (non-montage techniques)."""
    problems: list[str] = []
    for tech in TECHNIQUES:
        d = np.load(REF_DIR / f"{tech}.npz", allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        stretch = float(meta.get("time_stretch_requested", 1.0)) * \
            float(meta.get("time_stretch_kinematic", 1.0))
        ktimes = np.asarray(build_technique_targets(tech).times) * stretch
        bands = TECHNIQUE_CONFIG[tech]["phases"]
        if tech in cal.MONTAGE_TECHNIQUES:
            if verbose:
                print(f"  {tech:11s} montage: bands on feature trajectories "
                      f"(reconstructed keyframes {np.round(ktimes, 2)})")
            continue
        for name, t0, t1 in bands[1:]:
            dist = float(np.min(np.abs(ktimes - t0)))
            ok = dist <= cal.BAND_TOL_S
            if verbose:
                print(f"  {tech:11s} band {name:9s} @ {t0:5.2f}s "
                      f"nearest keyframe {dist:5.3f}s {'OK' if ok else 'MISMATCH'}")
            if not ok:
                problems.append(f"{tech}/{name}: {t0:.2f}s not on a keyframe "
                                f"(nearest {dist:.3f}s)")
    return problems


def evaluate(scorer: TechniqueScorer) -> tuple[dict, dict, dict]:
    """Per-phase self means, trace means and the cross matrix."""
    traces = {t: load_trace(t)[:2] for t in TECHNIQUES}
    self_phase: dict[str, list[float]] = {}
    self_mean: dict[str, float] = {}
    for tech in TECHNIQUES:
        r = scorer.score_trace(tech, *traces[tech])
        self_mean[tech] = r["mean"]
        self_phase[tech] = [v for _n, v, _c in r["per_phase"]]
    cross: dict[tuple[str, str], float] = {}
    for src in TECHNIQUES:
        self_t, opp_t = traces[src]
        for dst in TECHNIQUES:
            if src == dst:
                continue
            cross[(src, dst)] = scorer.score_trace(dst, self_t, opp_t)["mean"]
    return self_phase, self_mean, cross


def print_calibration_table(diag: dict) -> None:
    print("\n== CALIBRATION TABLE (params fitted from reference bands) ==")
    for tech in TECHNIQUES:
        print(f"\n-- {tech} (executor {executor_of(tech)})")
        for p, tmpl in TECHNIQUE_SPEC[tech]["templates"].items():
            name = TECHNIQUE_CONFIG[tech]["phases"][p][0]
            print(f"  phase {p} {name}:")
            for i, (feat, kind, w) in enumerate(tmpl):
                st = diag[tech][str(p)][f"{feat}:{kind}"]
                prm = TECHNIQUE_CONFIG[tech]["predicates"][p][i][2]
                print(f"    {feat:11s} {kind:8s} w={w:.1f} params="
                      f"{np.round(prm, 4).tolist()}  ref p10/p50/p90 = "
                      f"{st['p10']}/{st['p50']}/{st['p90']} (n={st['n']})")


def print_matrices(self_phase: dict, self_mean: dict, cross: dict) -> None:
    print("\n== SELF-CONFORMITY (per phase mean; reference scored as its own technique) ==")
    for tech in TECHNIQUES:
        names = [n for n, _a, _b in TECHNIQUE_CONFIG[tech]["phases"]]
        vals = " ".join(f"{n}={v:.3f}" for n, v in zip(names, self_phase[tech]))
        print(f"  {tech:11s} mean={self_mean[tech]:.3f}  {vals}")
    print("\n== CROSS MATRIX (row trace scored as column technique, executor role "
          "of the column; diagonal = self-conformity) ==")
    print(f"  {'trace':11s}" + "".join(f"{t[:11]:>12s}" for t in TECHNIQUES))
    for src in TECHNIQUES:
        print(f"  {src:11s}" + "".join(
            f"{(self_mean[src] if src == d else cross[(src, d)]):12.3f}"
            for d in TECHNIQUES))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help=f"write {PARAMS_PATH.relative_to(REPO)}")
    ap.add_argument("--quiet", action="store_true", help="skip the long tables")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    scorer = TechniqueScorer(model)

    print("== PHASE-BAND PROVENANCE (reconstructed keyframe times x stretches) ==")
    problems = check_bands(verbose=not args.quiet)

    params, diag = fit_all(scorer)
    if args.write:
        payload = {
            "meta": {
                "generated_by": "scripts/calibrate_scorer.py",
                "rule": {
                    "core_pct": cal.CORE_PCT,
                    "spread_frac": cal.SPREAD_FRAC,
                    "floor_m": cal.FLOOR_M,
                    "floor_rad": cal.FLOOR_RAD,
                },
                "refs_sha256": {
                    t: hashlib.sha256((REF_DIR / f"{t}.npz").read_bytes()).hexdigest()
                    for t in TECHNIQUES},
            },
            "params": params,
        }
        PARAMS_PATH.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {PARAMS_PATH.relative_to(REPO)}")
        # reload so the in-process config matches the file just written
        import importlib
        from scorer import config as cfg
        importlib.reload(cfg)
        scorer = TechniqueScorer(model)

    self_phase, self_mean, cross = evaluate(scorer)
    if not args.quiet:
        print_calibration_table(diag)
        print_matrices(self_phase, self_mean, cross)

    worst = min((min(v), t) for t, v in self_phase.items())
    ds = cross[("DOUBLE_LEG", "SINGLE_LEG")]
    print("\n== VERIFICATION ==")
    print(f"  min per-phase self-conformity : {worst[0]:.3f} ({worst[1]}) "
          f"[acceptance >= {SELF_MIN}]")
    print(f"  DOUBLE_LEG scored as SINGLE_LEG: {ds:.3f} "
          f"[acceptance < {CROSS_MAX['DOUBLE_LEG']['SINGLE_LEG']}]")
    print(f"  band provenance problems      : {len(problems)}")
    ok = (worst[0] >= SELF_MIN and ds < 0.5 and not problems)
    for p in problems:
        print(f"  PROBLEM: {p}")
    print("RESULT:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
