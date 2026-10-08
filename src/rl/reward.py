"""Stage-wise reward: env event weighting + minimal per-step shaping.

Separation contract (MISSION "critic versus technique judge"): the optional
technique-validity scorer (``src/scorer``) is imported lazily and used as a
*form* signal only.  The PPO critic (``rl.net.Critic``) is a separate module on
privileged state and never imports this package's scorer adapter.

Terms (weights come from :class:`RewardWeights`, one set per stage):

* ``outcome``     -- ``exchange_end`` event: ``+/- weights.outcome`` (the env
                     emits +1/-1; draw/ambiguous = 0).
* ``oob``         -- ``oob`` event: ``-0.25 * weights.oob`` to the offender.
* ``technique_similarity`` -- per-second rate ``w * sim * dt`` where ``sim`` is
                     the scorer's total in [0, 1] for the commanded technique;
                     0 when the scorer is absent (graceful degradation).
* ``progress``    -- potential-based shaping ``w * (gamma * phase' - phase)``
                     on the normalized command clock (Ng et al. 1999; adds no
                     bias to the optimal policy).  0 without a command.
* ``engagement``  -- per-second rate ``w * dt`` while the pelvis distance to
                     the opponent is within ``engage_radius`` (anti-stall
                     pressure, "minimal shaping" late).

Stage defaults encode the MISSION reward evolution (early similarity +
progress + outcome -> late outcome + minimal shaping); they are the
curriculum skeleton, not tuning results.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import asdict, dataclass, field

import numpy as np

#: env event magnitudes (mirror of wrestling.env.default_reward)
OOB_EVENT_PENALTY = -0.25
OUTCOME_MAGNITUDE = 1.0
#: pelvis distance (m) below which the engagement term applies
ENGAGE_RADIUS = 1.2


@dataclass(frozen=True)
class RewardWeights:
    """Reward-term weights for one stage."""

    technique_similarity: float = 0.0
    progress: float = 0.0
    outcome: float = 1.0
    oob: float = 1.0
    engagement: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def technique_executor(technique: str) -> str:
    """Reference executor robot of a technique (SPRAWL is demonstrated by b)."""
    return "b" if technique == "SPRAWL" else "a"


class ScorerAdapter:
    """Lazy adapter over the optional ``src/scorer`` technique-validity scorer.

    ``available=False`` (import error, missing scorer, or a scoring failure)
    disables the *reward* similarity term -- training continues, the term is
    zero, and the reason is kept in :attr:`note`.  Never raises.
    """

    def __init__(self, model, enabled: bool = True, score_every: int = 5):
        self.score_every = max(1, int(score_every))
        self.available = False
        self.note = "disabled by config"
        self._scorer = None
        if not enabled:
            return
        try:  # optional third-party-ish component, built in parallel: import lazily
            from scorer import TechniqueScorer  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - depends on repo state
            self.note = f"scorer import failed: {exc!r}"
            return
        try:
            self._scorer = TechniqueScorer(model)
            self.available = True
            self.note = "ready"
        except Exception as exc:  # pragma: no cover
            self.note = f"scorer init failed: {exc!r}"

    def phase_index(self, technique: str, frac: float):
        if not self.available:
            return None
        try:
            return int(self._scorer.phase_at_frac(technique, frac))
        except Exception as exc:
            self._degrade(exc)
            return None

    def similarity(self, technique: str, frac: float, self_qpos: np.ndarray,
                   opp_qpos: np.ndarray) -> float | None:
        """Scorer total in [0, 1] for the technique at normalized ``frac``, or None."""
        if not self.available:
            return None
        try:
            phase = int(self._scorer.phase_at_frac(technique, frac))
            result = self._scorer.score(technique, phase, np.asarray(self_qpos, dtype=np.float64),
                                        np.asarray(opp_qpos, dtype=np.float64))
            return float(np.clip(result.total, 0.0, 1.0))
        except Exception as exc:  # degrade, never crash training
            self._degrade(exc)
            return None

    def _degrade(self, exc: BaseException) -> None:
        self.available = False
        self.note = f"scorer disabled after error: {exc!r}"


class StageReward:
    """Reward assembly for one stage; used as the env's ``reward_fn`` and by the
    rollout loop for per-step shaping of the learner."""

    def __init__(self, weights: RewardWeights, *, scorer: ScorerAdapter | None = None,
                 learner_robot: str = "a", gamma: float = 0.99, dt: float = 0.02,
                 engage_radius: float = ENGAGE_RADIUS):
        self.weights = weights
        self.scorer = scorer
        self.learner_robot = learner_robot
        self.gamma = float(gamma)
        self.dt = float(dt)
        self.engage_radius = float(engage_radius)
        self._score_phase = 0
        self.warnings: list[str] = []
        self.last_similarity: float | None = None

    # -------------------------------------------------------------- env side
    def reward_fn(self, env, event: dict) -> tuple[float, float]:
        """Env ``reward_fn(env, event)`` applying the stage weights."""
        if event["kind"] == "oob":
            return tuple(OOB_EVENT_PENALTY * self.weights.oob if p == event["robot"] else 0.0
                         for p in ("a", "b"))
        if event["kind"] == "exchange_end":
            winner = event.get("winner")
            if winner in ("a", "b"):
                return tuple(OUTCOME_MAGNITUDE * self.weights.outcome if p == winner
                             else -OUTCOME_MAGNITUDE * self.weights.outcome for p in ("a", "b"))
            return (0.0, 0.0)
        return (0.0, 0.0)

    # -------------------------------------------------------------- learner side
    def shaping(self, *, technique: str | None, phase: float, phase_next: float,
                qpos_self: np.ndarray, qpos_opp: np.ndarray) -> float:
        """Per-step shaping reward for the learner transition ``s -> s'``.

        ``phase`` is the normalized command phase of state ``s`` (used to score
        similarity), ``phase_next`` that of ``s'`` (the potential difference
        ``gamma*phi(s') - phi(s)``).  Without a command the progress term is 0.
        """
        w = self.weights
        r = 0.0
        # potential-based progress on the command clock (Ng et al. 1999)
        if technique is not None and w.progress:
            r += w.progress * (self.gamma * float(phase_next) - float(phase))
        # technique similarity (subsampled; rate per second of sim time)
        if technique is not None and w.technique_similarity and self.scorer is not None:
            self._score_phase += 1
            if self._score_phase % self.scorer.score_every == 1:
                sim = self._similarity_for(technique, phase, qpos_self, qpos_opp)
                self.last_similarity = sim
                if sim is not None:
                    r += w.technique_similarity * sim * self.dt * self.scorer.score_every
        # engagement pressure
        if w.engagement:
            dist = float(np.linalg.norm(np.asarray(qpos_opp)[0:3] - np.asarray(qpos_self)[0:3]))
            if dist <= self.engage_radius:
                r += w.engagement * self.dt
        return float(r)

    def _similarity_for(self, technique: str, phase: float, qpos_self: np.ndarray,
                        qpos_opp: np.ndarray) -> float | None:
        executor = technique_executor(technique)
        if executor != self.learner_robot:
            msg = (f"technique {technique} executor is {executor} but the learner is "
                   f"{self.learner_robot}: similarity term skipped")
            if msg not in self.warnings:
                self.warnings.append(msg)
            return None
        return self.scorer.similarity(technique, phase, qpos_self, qpos_opp)

    def status(self) -> dict:
        return {
            "weights": self.weights.as_dict(),
            "scorer_available": bool(self.scorer.available) if self.scorer else False,
            "scorer_note": self.scorer.note if self.scorer else "none",
            "warnings": list(self.warnings),
        }


if __name__ == "__main__":  # self-check
    from wrestling.env import WrestlingEnv

    env = WrestlingEnv(seed=0, match_clock=2.0, exchange_timeout=2.0)
    sr = StageReward(RewardWeights(technique_similarity=0.4, progress=0.3, outcome=1.5, oob=1.0, engagement=0.01))
    ra, rb = sr.reward_fn(env, {"kind": "exchange_end", "winner": "a", "loser": "b"})
    assert ra == 1.5 and rb == -1.5
    ra, rb = sr.reward_fn(env, {"kind": "oob", "robot": "b", "count": 1})
    assert abs(ra) < 1e-12 and abs(rb + 0.25) < 1e-12
    # degraded scorer: similarity term silently 0
    sr_ns = StageReward(RewardWeights(technique_similarity=1.0, progress=0.0, outcome=1.0))
    q = np.zeros(36)
    r = sr_ns.shaping(technique="DOUBLE_LEG", phase=0.5, phase_next=0.6, qpos_self=q, qpos_opp=q)
    assert r == 0.0
    # progress potential: gamma*phase_next - phase
    sr_p = StageReward(RewardWeights(technique_similarity=0.0, progress=1.0, outcome=1.0))
    assert abs(sr_p.shaping(technique="DOUBLE_LEG", phase=0.5, phase_next=0.6,
                            qpos_self=q, qpos_opp=q) - (0.99 * 0.6 - 0.5)) < 1e-12
    print("rl.reward self-check OK:", {"progress_term": round(0.99 * 0.6 - 0.5, 4),
                                       "oob": OOB_EVENT_PENALTY})
