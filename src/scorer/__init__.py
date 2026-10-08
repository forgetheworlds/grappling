"""Technique-validity scorer package (deliverable 7).

Hand-engineered relational-geometry judge over the phase-2 reference
techniques. SEPARATION CONTRACT: this package is independent of any RL
critic / value function — nothing here imports reward or value code, and
value-function code MUST NOT import this package (goal.md: "Technique-quality
judge is a SEPARATE component"). Training loops may consume ``score()`` as a
form/validity signal; the return estimator must remain a distinct module.

Modules:
- features.py   — relational feature extraction (one mj_kinematics call)
- membership.py — smooth fuzzy membership functions
- config.py     — calibrated phase bands + predicate thresholds
- scorer.py     — TechniqueScorer public API
"""

from .config import TECHNIQUE_CONFIG, TECHNIQUES, predicates_for
from .features import FEAT_ORDER, Featurizer, feature_dict
from .scorer import ScoreResult, TechniqueScorer, executor_of

__all__ = [
    "TECHNIQUE_CONFIG", "TECHNIQUES", "predicates_for",
    "FEAT_ORDER", "Featurizer", "feature_dict",
    "ScoreResult", "TechniqueScorer", "executor_of",
]
