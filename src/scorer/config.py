"""Scorer configuration: hand-authored spec + generated calibration, joined.

``spec.py`` holds the phase bands and predicate templates (the wrestling
semantics); ``data/scorer_calibration.json`` holds the fuzzy membership
parameters fitted from the reference traces by
``scripts/calibrate_scorer.py`` with the rule in ``src/scorer/calibration.py``.
Regenerate the numbers with::

    .venv/bin/python scripts/calibrate_scorer.py --write

This module is the only place that reads the calibration file, so an
uncalibrated tree fails loudly and immediately instead of scoring with
placeholder thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

from .spec import TECHNIQUE_SPEC, TECHNIQUES

REPO = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO / "data" / "scorer_calibration.json"

__all__ = ["TECHNIQUE_SPEC", "TECHNIQUES", "TECHNIQUE_CONFIG", "PARAMS_PATH",
           "load_params", "build_config", "predicates_for", "executor_of"]


def load_params(path: Path | str = PARAMS_PATH) -> dict:
    """Calibrated predicate parameters keyed ``tech -> phase -> 'feat:kind'``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — regenerate with "
            f"`.venv/bin/python scripts/calibrate_scorer.py --write`")
    with p.open() as fh:
        return json.load(fh)["params"]


def build_config(params: dict) -> dict[str, dict]:
    """Pair predicate templates with calibrated parameters.

    Every template entry must have a calibrated ``feature:kind`` entry; a
    missing one is a hard error (the calibration is stale — regenerate it).
    """
    out: dict[str, dict] = {}
    for tech, spec in TECHNIQUE_SPEC.items():
        preds: dict[int, list[tuple]] = {}
        for phase, tmpl in spec["templates"].items():
            table = params.get(tech, {}).get(str(phase))
            if table is None:
                raise KeyError(f"no calibrated params for {tech} phase {phase}")
            keys = [f"{f}:{k}" for f, k, _ in tmpl]
            missing = [k for k in keys if k not in table]
            if missing:
                raise KeyError(f"{tech} phase {phase}: uncalibrated {missing}")
            preds[phase] = [(f, k, tuple(float(v) for v in table[f"{f}:{k}"]), w)
                            for f, k, w in tmpl]
        out[tech] = {"executor": spec["executor"], "phases": spec["phases"],
                     "predicates": preds}
    return out


TECHNIQUE_CONFIG: dict[str, dict] = build_config(load_params())


def predicates_for(technique: str, phase: int) -> list[tuple]:
    """Resolved (feature, kind, params, weight) list for a technique phase."""
    return TECHNIQUE_CONFIG[technique]["predicates"][phase]


def executor_of(technique: str) -> str:
    """Which reference robot demonstrates ``technique`` ('A' or 'B')."""
    return TECHNIQUE_CONFIG[technique]["executor"]
