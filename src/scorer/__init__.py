"""Technique-validity scorer package (deliverable 7).

Hand-engineered relational-geometry judge over the phase-2 reference
techniques: does the commanded technique, in its current phase, still resemble
its movement family? SEPARATION CONTRACT: this package is independent of any RL
critic / value function — nothing here imports reward or value code, and
value-function code MUST NOT import this package (goal.md: "Technique-quality
judge is a SEPARATE component"). Training loops may consume ``score()`` as a
form/validity signal; the return estimator stays a distinct module.

Modules:
- features.py    — relational feature extraction (one mj_kinematics call)
- membership.py  — smooth fuzzy membership functions
- spec.py        — hand-authored phase bands + predicate templates
- calibration.py — the threshold-fitting rule (documented, testable)
- config.py      — spec joined with data/scorer_calibration.json
- scorer.py      — TechniqueScorer / score() public API

Calibration provenance and the per-technique threshold table:
reports/2026-10-08/scorer.md. Regenerate numbers with
``.venv/bin/python scripts/calibrate_scorer.py --write``.
"""

from .config import (PARAMS_PATH, TECHNIQUE_CONFIG, TECHNIQUE_SPEC,  # noqa: F401
                     TECHNIQUES, executor_of, predicates_for)
from .features import FEAT_ORDER, Featurizer, feature_dict  # noqa: F401
from .scorer import ScoreResult, TechniqueScorer, score  # noqa: F401

__all__ = [
    "TECHNIQUE_CONFIG", "TECHNIQUE_SPEC", "TECHNIQUES", "PARAMS_PATH",
    "predicates_for", "executor_of",
    "FEAT_ORDER", "Featurizer", "feature_dict",
    "ScoreResult", "TechniqueScorer", "score",
]
