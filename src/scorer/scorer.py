"""Technique-validity scorer (deliverable 7).

Hand-engineered relational-geometry judge: does the commanded technique,
in its current phase, still resemble its movement family?  This module is
COMPLETELY SEPARATE from any RL value function — it must never be imported
by critic/reward code (see goal.md hard rules; the critic estimates return,
this judges technique form).  It is allowed INSIDE training loops as a
monitor/shaping signal only through its public score API.

Public API:
    TechniqueScorer(model)                     # bound to the 2-G1 scene model
    .score(technique, phase, self_qpos, opp_qpos) -> ScoreResult
    .phase_at(technique, t)                    # phase index for wall time
    .phase_at_frac(technique, frac)            # phase index for normalized progress
    .score_trace(technique, self_traj, opp_traj) -> per-phase means + overall

ScoreResult: ``total`` (weighted mean of fuzzy memberships, [0, 1]) and
``terms`` ({feature: (membership, weight)} breakdown for logging/debug).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .config import TECHNIQUE_CONFIG, TECHNIQUES, predicates_for
from .features import FEAT_ORDER, Featurizer, feature_dict
from .membership import ang_near, at_least, at_most, band

_MEMBERSHIP = {"band": band, "at_most": at_most, "at_least": at_least,
               "ang_near": ang_near}


@dataclass
class ScoreResult:
    """One scoring call: total in [0,1] + per-predicate breakdown."""
    technique: str
    phase: int
    phase_name: str
    total: float
    terms: dict[str, tuple[float, float]] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)


class TechniqueScorer:
    """Scores relational-geometry conformity of a commanded technique phase.

    One instance per process (owns an MjData for kinematics); not
    thread-safe. ~0.15 ms per score() on the 4-core ARM host.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self._feat = Featurizer(model)
        self._fidx = {n: i for i, n in enumerate(FEAT_ORDER)}

    # ------------------------------------------------------------------ API
    def score(self, technique: str, phase: int,
              self_qpos: np.ndarray, opp_qpos: np.ndarray) -> ScoreResult:
        """Conformity of (self_qpos, opp_qpos) to technique phase -> [0,1]."""
        if technique not in TECHNIQUE_CONFIG:
            raise KeyError(f"unknown technique {technique!r}; "
                           f"known: {TECHNIQUES}")
        cfg = TECHNIQUE_CONFIG[technique]
        if not 0 <= phase < len(cfg["phases"]):
            raise IndexError(f"phase {phase} out of range for {technique} "
                             f"({len(cfg['phases'])} phases)")
        v = self._feat.features(self_qpos, opp_qpos)
        preds = predicates_for(technique, phase)
        terms: dict[str, tuple[float, float]] = {}
        wsum = 0.0
        acc = 0.0
        for name, kind, params, w in preds:
            mu = _MEMBERSHIP[kind](v[self._fidx[name]], *params)
            terms[f"{name}:{kind}"] = (float(mu), w)
            acc += w * mu
            wsum += w
        return ScoreResult(technique, phase, cfg["phases"][phase][0],
                           float(acc / wsum), terms, feature_dict(v))

    def phase_at(self, technique: str, t: float) -> int:
        """Phase index at trace time t (seconds, reference timeline)."""
        for i, (_, t0, t1) in enumerate(TECHNIQUE_CONFIG[technique]["phases"]):
            if t0 <= t < t1:
                return i
        return len(TECHNIQUE_CONFIG[technique]["phases"]) - 1

    def phase_at_frac(self, technique: str, frac: float) -> int:
        """Phase index at normalized progress frac in [0,1] (any trace length).

        Maps the reference phase bands onto the progress axis; used to score
        rolled-out traces whose duration differs from the reference.
        """
        bands = TECHNIQUE_CONFIG[technique]["phases"]
        t_end = bands[-1][2]
        return self.phase_at(technique, min(max(frac, 0.0), 1.0) * t_end)

    def score_trace(self, technique: str, self_traj: np.ndarray,
                    opp_traj: np.ndarray) -> dict:
        """Score a full (T, 36)/(T, 36) trajectory against ``technique``.

        Frame i is scored at its own normalized-progress phase.  Returns
        {"per_phase": [(name, mean, n)], "mean": overall mean,
         "executor": role letter}.  For reference traces pass the executor
        robot as self (A for all techniques except SPRAWL, whose executor
        is B — see config.py).
        """
        T = len(self_traj)
        if len(opp_traj) != T:
            raise ValueError("self/opp trajectory length mismatch")
        n_ph = len(TECHNIQUE_CONFIG[technique]["phases"])
        sums = np.zeros(n_ph)
        ns = np.zeros(n_ph, dtype=int)
        for i in range(T):
            ph = self.phase_at_frac(technique, i / max(T - 1, 1))
            r = self.score(technique, ph, self_traj[i], opp_traj[i])
            sums[ph] += r.total
            ns[ph] += 1
        per_phase = [(TECHNIQUE_CONFIG[technique]["phases"][i][0],
                      float(sums[i] / ns[i]) if ns[i] else float("nan"),
                      int(ns[i])) for i in range(n_ph)]
        return {"per_phase": per_phase,
                "mean": float(sums.sum() / ns.sum()),
                "executor": TECHNIQUE_CONFIG[technique]["executor"]}


def executor_of(technique: str) -> str:
    """Which reference robot demonstrates ``technique`` ('A' or 'B')."""
    return TECHNIQUE_CONFIG[technique]["executor"]
