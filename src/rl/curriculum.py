"""PISTY stage ladder (A-E) as a dataclass schema, with reward-weight and
perturbation schedules.

MISSION reward evolution, encoded stage-wise (per ``docs/CURRICULUM.md``):

* early (A-C): technique similarity + progress + outcome;
* middle (D): reduced similarity weight + strong outcome;
* late (E): win/loss of exchange + minimal anti-exploit shaping.

Stage fields: opponent controller spec (scripted; stage E may instead point
at an opponent checkpoint for current/snapshot self-play), technique command
on/off + which techniques are sampled, reward weights
``{technique_similarity, progress, outcome, oob, engagement}``, perturbation
schedule (action noise ramp, command-clock jitter), optional phase-3 teacher
BC checkpoint for warm start, and an advance rule whose *criterion* is a
per-stage choice (see below).

Advance rule (deterministic): the rolling window holds one
:class:`ExchangeSample` per finished exchange (outcome, mean technique-scorer
similarity, stood/fallen).  A stage advances once
``steps_in_stage >= rule.min_steps`` AND the window holds at least
``rule.window_episodes`` exchanges AND the window holds at least
``rule.min_successes`` successes AND ``successes / window_episodes >=
rule.min_rate``, where a *success* is, per :class:`AdvanceRule`:

* ``criterion="outcome"`` (resistance/attack-vs-defense, D-E) -- the learner
  won the exchange.  Draws (timeouts, ambiguous endings) and losses occupy
  window slots as non-success, so an all-draw or all-loss window can never
  advance (``docs/MOTOR_CURRICULUM.md`` §3: "Gates are per-capability metrics,
  never exchange/outcome rate"; "Never promote because the opponent collapses").
* ``criterion="execution"`` (imitation/drilling, A-C) -- the learner **stayed on
  its feet** AND its attempt **scored >= rule.min_similarity on the technique
  scorer**: the stage's own objective, not the exchange outcome.  This is
  required because a stage-A-C partner (``stand_hold``) draws forever -- an
  outcome gate there is unsatisfiable -- and because "wins" against a collapsing
  partner are exactly the E1 self-deception case (§0, §3.2).  Without a
  similarity measurement (scorer unavailable, or the learner is not the
  technique's executor) there is no evidence, hence no success and no promotion.

The pre-fix rule was ``metric = (wins + 0.5 * draws) / n >= 0.5``, under which an
all-draw window scored exactly the 0.5 threshold and promoted a stage with zero
demonstrated success.  ``min_successes >= 1`` and ``min_rate > 0`` are enforced
as invariants of :class:`AdvanceRule`.

The *gate numbers* below are the curriculum skeleton, not tuning results
(``min_similarity``/``min_rate``/``window_episodes``/``min_steps``): they are
**placeholders** to be set experimentally per §5 and are recorded in every
checkpoint so later experiments are traceable (see
``reports/2026-10-08/p0_fixes.md``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .obs import TECHNIQUES, TechniqueCommand, reference_phase
from .reward import RewardWeights
from .scripted import OpponentSpec

#: Success criteria for a stage's advancement gate (see :class:`AdvanceRule`):
#: ``"outcome"`` = learner wins; ``"execution"`` = stayed up + scorer similarity.
CRITERIA = ("outcome", "execution")


@dataclass(frozen=True)
class PerturbationSchedule:
    """Learner-side perturbation: action-noise ramp (rad) + command-clock jitter."""

    action_noise_start: float = 0.0
    action_noise_end: float = 0.0
    ramp_steps: int = 50_000
    phase_jitter_s: float = 0.0

    def noise_std(self, stage_steps: int) -> float:
        """Linear ramp from start to end over ``ramp_steps`` (then constant)."""
        if self.ramp_steps <= 0:
            return float(self.action_noise_end)
        frac = float(np.clip(stage_steps / float(self.ramp_steps), 0.0, 1.0))
        return float(self.action_noise_start + frac * (self.action_noise_end - self.action_noise_start))

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExchangeSample:
    """One finished exchange as the advancement gate sees it.

    ``outcome``
        ``+1`` learner win, ``-1`` learner loss, ``0`` draw (timeout or
        ambiguous ending).
    ``similarity``
        mean technique-scorer similarity in ``[0, 1]`` over the attempt, or
        ``None`` when no measurement exists (scorer unavailable, term disabled,
        or the learner is not the technique's executor).
    ``stood``
        True when the learner held its own ground for the whole exchange: no
        back-to-mat trigger against it and no out-of-bounds forfeit (the no-fall
        criterion of the execution gate).
    """

    outcome: int
    similarity: float | None = None
    stood: bool = True

    def as_dict(self) -> dict:
        return {"outcome": int(self.outcome),
                "similarity": None if self.similarity is None else float(self.similarity),
                "stood": bool(self.stood)}

    @classmethod
    def from_dict(cls, d: dict) -> "ExchangeSample":
        sim = d.get("similarity")
        return cls(outcome=int(d["outcome"]),
                   similarity=None if sim is None else float(sim),
                   stood=bool(d.get("stood", True)))


@dataclass(frozen=True)
class AdvanceRule:
    """When a stage is considered passed (see module docstring).

    All fields are configurable per stage:

    ``min_steps``
        control steps the stage must have trained for.
    ``window_episodes``
        rolling window size in exchanges; the gate is only evaluated on a full
        window.
    ``criterion``
        what counts as a *success* in the window: ``"outcome"`` (the learner
        won the exchange; draws and losses are non-success) or ``"execution"``
        (the learner stayed on its feet AND the attempt scored at least
        ``min_similarity`` on the technique scorer).  A-C use ``"execution"``
        because their partner draws forever and because wins against a
        collapsing partner prove nothing; D-E use ``"outcome"``.
    ``min_successes``
        minimum number of successes in the window.  Must be ``>= 1``: a stage
        may never advance without at least one demonstrated success
        (MOTOR_CURRICULUM.md §3).  Raise it to tighten a stage.  For
        ``criterion="outcome"`` a success is a win, so this is the earlier
        ``min_wins`` floor.
    ``min_rate``
        minimum ``successes / window_episodes``.  Must be in ``(0, 1]``; there
        is no half credit, so a winless/draw-only (or, for A-C, a
        low-similarity or falling) window scores 0.0 and cannot advance.
    ``min_similarity``
        per-exchange scorer gate for ``criterion="execution"`` (ignored by the
        outcome criterion).
    ``success_hook``
        **Extension point (not implemented logic).**  Optional predicate taking
        the :class:`ExchangeSample`; when set, a success must also be accepted
        by it.  This is the seam for MISSION's later "success attributable to
        the attempted technique" requirement: the attribution verdict arrives
        with the attempt/technique attached, and only then can an
        ``outcome``-criterion stage (D-E) demand that the win was caused by the
        attempted technique rather than by the opponent collapsing.  It is a
        runtime-only field: :meth:`as_dict` omits it (a callable cannot be
        checkpointed), so a resumed run re-installs it from its config.  The
        A-C execution criterion is deliberately *declarative* fields rather
        than a hook, so the shipped stages stay checkpoint/resume-exact.
    """

    min_steps: int = 20_000
    window_episodes: int = 20
    criterion: str = "outcome"
    min_successes: int = 1
    min_rate: float = 0.5
    min_similarity: float = 0.5
    success_hook: Callable[[ExchangeSample], bool] | None = None

    def __post_init__(self):
        if self.criterion not in CRITERIA:
            raise ValueError(f"unknown advance criterion {self.criterion!r}; known: {list(CRITERIA)}")
        if int(self.min_successes) < 1:
            raise ValueError(
                "AdvanceRule.min_successes must be >= 1: a stage may never advance "
                "without at least one demonstrated success (MOTOR_CURRICULUM.md §3)")
        if not 0.0 < float(self.min_rate) <= 1.0:
            raise ValueError(
                f"AdvanceRule.min_rate must be in (0, 1], got {self.min_rate!r}: "
                "draws/falls count as non-success, so a non-positive gate could be "
                "cleared by a window without demonstrated success")
        if not 0.0 <= float(self.min_similarity) <= 1.0:
            raise ValueError(
                f"AdvanceRule.min_similarity must be in [0, 1], got {self.min_similarity!r}")

    def as_dict(self) -> dict:
        d = asdict(self)
        # the hook is runtime-only (callables do not survive checkpointing)
        d.pop("success_hook", None)
        return d


@dataclass(frozen=True)
class StageConfig:
    """One PISTY stage."""

    key: str
    name: str
    techniques: tuple[str, ...] = ("DOUBLE_LEG",)
    command_on: bool = True
    opponent: OpponentSpec = field(default_factory=OpponentSpec)
    weights: RewardWeights = field(default_factory=RewardWeights)
    perturbations: PerturbationSchedule = field(default_factory=PerturbationSchedule)
    bc_checkpoint: str | None = None       # optional phase-3 teacher warm start
    use_scorer: bool = True                # optional technique scorer for the similarity term
    advance: AdvanceRule | None = None

    def __post_init__(self):
        for t in self.techniques:
            if t not in TECHNIQUES:
                raise KeyError(f"unknown technique {t!r}; known: {TECHNIQUES}")
        if self.command_on and not self.techniques:
            raise ValueError(f"stage {self.key}: command_on requires at least one technique")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["opponent"] = self.opponent.as_dict()
        d["weights"] = self.weights.as_dict()
        d["perturbations"] = self.perturbations.as_dict()
        d["advance"] = None if self.advance is None else self.advance.as_dict()
        return d


#: PISTY stages A-E (CURRICULUM.md progression table + MISSION reward evolution)
PISTY_STAGES: tuple[StageConfig, ...] = (
    StageConfig(
        key="A", name="imitation",
        techniques=("DOUBLE_LEG",), command_on=True,
        opponent=OpponentSpec("stand_hold"),
        weights=RewardWeights(technique_similarity=0.6, progress=0.4, outcome=1.0, oob=1.0, engagement=0.0),
        perturbations=PerturbationSchedule(0.0, 0.0, ramp_steps=1, phase_jitter_s=0.0),
        # gate = execution competence (scorer similarity + stayed up), not wins:
        # the stand_hold partner draws forever, so a wins-only gate here is
        # unsatisfiable (and wins vs a collapsing partner prove nothing, §3.2)
        advance=AdvanceRule(min_steps=20_000, window_episodes=10, criterion="execution",
                            min_successes=1, min_rate=0.5, min_similarity=0.5),
    ),
    StageConfig(
        key="B", name="randomized drilling",
        techniques=("DOUBLE_LEG",), command_on=True,
        opponent=OpponentSpec("stand_hold"),
        weights=RewardWeights(technique_similarity=0.5, progress=0.4, outcome=1.0, oob=1.0, engagement=0.0),
        perturbations=PerturbationSchedule(0.0, 0.03, ramp_steps=20_000, phase_jitter_s=0.3),
        advance=AdvanceRule(min_steps=30_000, window_episodes=20, criterion="execution",
                            min_successes=1, min_rate=0.5, min_similarity=0.5),
    ),
    StageConfig(
        key="C", name="dynamic drilling",
        techniques=("DOUBLE_LEG",), command_on=True,
        opponent=OpponentSpec("reference_replay", technique="STANCE", loop=True),
        weights=RewardWeights(technique_similarity=0.4, progress=0.3, outcome=1.0, oob=1.0, engagement=0.01),
        perturbations=PerturbationSchedule(0.03, 0.05, ramp_steps=30_000, phase_jitter_s=0.3),
        advance=AdvanceRule(min_steps=40_000, window_episodes=20, criterion="execution",
                            min_successes=1, min_rate=0.5, min_similarity=0.5),
    ),
    StageConfig(
        key="D", name="resistance",
        techniques=("DOUBLE_LEG",), command_on=True,
        # scripted stand-in for a resisting body (a true hips-back/sprawl
        # resistance controller is the phase-3 teacher's job; the schema also
        # accepts OpponentSpec("teacher", technique=...) for it)
        opponent=OpponentSpec("reference_replay", technique="SPRAWL", loop=True),
        weights=RewardWeights(technique_similarity=0.15, progress=0.2, outcome=1.5, oob=1.0, engagement=0.02),
        perturbations=PerturbationSchedule(0.05, 0.05, ramp_steps=1, phase_jitter_s=0.5),
        # gate = exchange outcome: draws/losses never advance; the
        # "success attributable to the attempted technique" hook is a documented
        # placeholder for D-E (see AdvanceRule.success_hook), not implemented yet
        advance=AdvanceRule(min_steps=40_000, window_episodes=20, criterion="outcome",
                            min_successes=1, min_rate=0.5),
    ),
    StageConfig(
        key="E", name="attack vs defense",
        techniques=("DOUBLE_LEG",), command_on=False,
        # learned defender: current/snapshot policy (frozen) or a checkpoint;
        # historical-pool league play is phase 6/7
        opponent=OpponentSpec("policy", technique="SPRAWL"),
        weights=RewardWeights(technique_similarity=0.0, progress=0.0, outcome=1.0, oob=1.0, engagement=0.02),
        perturbations=PerturbationSchedule(0.05, 0.02, ramp_steps=20_000, phase_jitter_s=0.0),
        advance=None,
    ),
)


def stage_by_key(key: str) -> StageConfig:
    for s in PISTY_STAGES:
        if s.key.upper() == str(key).upper():
            return s
    raise KeyError(f"unknown stage {key!r}; known: {[s.key for s in PISTY_STAGES]}")


def stage_from_dict(d: dict) -> StageConfig:
    """Rebuild a :class:`StageConfig` from ``as_dict()`` output (checkpoint/resume)."""
    known = set(StageConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in d.items() if k in known}
    kwargs["techniques"] = tuple(kwargs.get("techniques", ()))
    if "opponent" in kwargs:
        kwargs["opponent"] = OpponentSpec.from_dict(kwargs["opponent"])
    if "weights" in kwargs:
        w = kwargs["weights"]
        kwargs["weights"] = RewardWeights(**{k: v for k, v in w.items()
                                             if k in RewardWeights.__dataclass_fields__})
    if "perturbations" in kwargs:
        p = kwargs["perturbations"]
        kwargs["perturbations"] = PerturbationSchedule(
            **{k: v for k, v in p.items() if k in PerturbationSchedule.__dataclass_fields__})
    if "advance" in kwargs:
        a = kwargs["advance"]
        kwargs["advance"] = None if a is None else AdvanceRule(
            **{k: v for k, v in a.items() if k in AdvanceRule.__dataclass_fields__})
    return StageConfig(**kwargs)


class CommandScheduler:
    """Deterministic per-exchange technique command + command clock.

    The command is a pure function of ``(seed, env_index, exchange_index)`` --
    no hidden state -- so training is reproducible and resume-safe.
    """

    def __init__(self, techniques: tuple[str, ...], *, command_on: bool = True,
                 phase_jitter_s: float = 0.0, seed: int = 0):
        self.techniques = tuple(techniques)
        self.command_on = bool(command_on) and bool(self.techniques)
        self.phase_jitter_s = float(phase_jitter_s)
        self.seed = int(seed)
        self._cache: dict[tuple[int, int], tuple[str | None, float]] = {}

    def _draw(self, env_index: int, exchange_index: int) -> tuple[str | None, float]:
        key = (int(env_index), int(exchange_index))
        hit = self._cache.get(key)
        if hit is None:
            rng = np.random.default_rng([self.seed, key[0], key[1]])
            technique = self.techniques[int(rng.integers(len(self.techniques)))] if self.command_on else None
            offset = float(rng.uniform(0.0, self.phase_jitter_s)) if self.phase_jitter_s > 0 else 0.0
            hit = (technique, offset)
            self._cache[key] = hit
        return hit

    def command(self, env_index: int, exchange_index: int, exchange_time: float) -> TechniqueCommand:
        technique, offset = self._draw(env_index, exchange_index)
        if technique is None:
            return TechniqueCommand.off()
        return TechniqueCommand(technique, reference_phase(technique, exchange_time, offset))

    def vector(self, env_index: int, exchange_index: int, exchange_time: float) -> np.ndarray:
        return self.command(env_index, exchange_index, exchange_time).vector()


class Curriculum:
    """Stage pointer + rolling exchange window + deterministic advance rule."""

    def __init__(self, stages: tuple[StageConfig, ...] = PISTY_STAGES, start_index: int = 0):
        if not 0 <= start_index < len(stages):
            raise ValueError(f"start_index {start_index} out of range for {len(stages)} stages")
        self.stages = tuple(stages)
        self.index = int(start_index)
        self.steps_in_stage = 0
        self.window: deque[ExchangeSample] = deque(
            maxlen=self.stages[self.index].advance.window_episodes
            if self.stages[self.index].advance else 20)

    # ------------------------------------------------------------------ state
    @property
    def stage(self) -> StageConfig:
        return self.stages[self.index]

    @property
    def done(self) -> bool:
        return self.index == len(self.stages) - 1

    def tick(self, steps: int = 1) -> None:
        self.steps_in_stage += int(steps)

    def on_exchange(self, outcome: int, *, similarity: float | None = None,
                    stood: bool = True) -> None:
        """Record one finished exchange in the rolling window.

        ``outcome``: +1 learner win, -1 loss, 0 draw.  ``similarity``: mean
        technique-scorer similarity of the attempt (``None`` when the scorer was
        unavailable or the learner is not the technique's executor) -- the
        signal the A-C execution gate needs.  ``stood``: the learner did not go
        to its own back during the exchange.
        """
        if len(self.window) == self.window.maxlen:
            self.window.popleft()
        self.window.append(ExchangeSample(outcome=int(np.sign(outcome)),
                                          similarity=similarity, stood=bool(stood)))

    # ------------------------------------------------------------------ metric
    def _success(self, sample: ExchangeSample) -> bool:
        """Is one window entry a demonstrated success for the current stage?

        ``criterion="outcome"``: the learner won (draws/losses never count).
        ``criterion="execution"``: the learner stayed up AND the attempt scored
        at least ``min_similarity``; an unmeasured attempt (``similarity is
        None``) is not evidence and never counts.  A stage's ``success_hook``
        (the documented seam for MISSION's "success attributable to the
        attempted technique" gate) can only *reject* a success, never grant one.
        """
        rule = self.stage.advance
        if rule is None:
            return False
        if rule.criterion == "outcome":
            ok = sample.outcome > 0
        else:  # "execution"
            ok = (sample.stood and sample.similarity is not None
                  and sample.similarity >= rule.min_similarity)
        if ok and rule.success_hook is not None:
            ok = bool(rule.success_hook(sample))
        return bool(ok)

    def successes(self) -> int:
        """Demonstrated successes in the rolling window."""
        return sum(1 for s in self.window if self._success(s))

    def gate_status(self) -> dict:
        """Why the current stage is (not) advancing (telemetry/debug)."""
        rule = self.stage.advance
        if rule is None:
            return {"stage": self.stage.key, "rule": None, "window": len(self.window)}
        return {"stage": self.stage.key, "criterion": rule.criterion,
                "steps_in_stage": self.steps_in_stage, "min_steps": rule.min_steps,
                "window": len(self.window), "window_episodes": rule.window_episodes,
                "successes": self.successes(), "min_successes": rule.min_successes,
                "metric": self.metric(), "min_rate": rule.min_rate,
                "unmeasured": sum(1 for s in self.window if s.similarity is None)}

    def metric(self) -> float | None:
        """Rolling success rate over the window (``None`` if it is empty).

        ``successes / window size`` under the stage's criterion.  Every
        non-success (draws and losses for ``"outcome"``; unmeasured,
        low-similarity or falling attempts for ``"execution"``) occupies a
        window slot and contributes 0, so a window without demonstrated success
        scores 0.0 and can never clear ``min_rate > 0`` -- promotion without
        evidence is impossible by construction (``docs/MOTOR_CURRICULUM.md`` §3).
        """
        if not self.window:
            return None
        return self.successes() / len(self.window)

    def maybe_advance(self) -> bool:
        """Advance if the current stage's rule is satisfied (returns True once)."""
        rule = self.stage.advance
        if rule is None or self.done:
            return False
        if self.steps_in_stage < rule.min_steps:
            return False
        if len(self.window) < rule.window_episodes:
            return False
        if self.successes() < rule.min_successes:
            return False
        m = self.metric()
        if m is None or m < rule.min_rate:
            return False
        self.index += 1
        self.steps_in_stage = 0
        next_rule = self.stage.advance
        self.window = deque(maxlen=next_rule.window_episodes if next_rule else 20)
        return True

    # ------------------------------------------------------------------ io
    def state_dict(self) -> dict:
        return {"index": self.index, "steps_in_stage": self.steps_in_stage,
                "window": [s.as_dict() for s in self.window]}

    def load_state_dict(self, d: dict) -> None:
        idx = int(d.get("index", 0))
        if not 0 <= idx < len(self.stages):
            raise ValueError(f"checkpoint stage index {idx} out of range")
        self.index = idx
        self.steps_in_stage = int(d.get("steps_in_stage", 0))
        rule = self.stage.advance
        self.window = deque(maxlen=rule.window_episodes if rule else 20)
        for x in d.get("window", []):
            self.window.append(ExchangeSample.from_dict(x))


if __name__ == "__main__":  # self-check
    assert [s.key for s in PISTY_STAGES] == list("ABCDE")
    A, B, C, D, E = PISTY_STAGES
    assert A.weights.technique_similarity > D.weights.technique_similarity > 0
    assert E.weights.technique_similarity == 0 and E.weights.progress == 0
    assert D.weights.outcome > C.weights.outcome
    assert C.perturbations.noise_std(0) == 0.03 and C.perturbations.noise_std(60_000) == 0.05
    sch = CommandScheduler(("DOUBLE_LEG",), command_on=True, phase_jitter_s=0.3, seed=7)
    c1 = sch.command(0, 1, 0.0)
    c2 = sch.command(0, 1, 0.0)
    assert c1 == c2 and 0.0 <= c1.phase <= 1.0
    assert [s.advance.criterion for s in PISTY_STAGES[:3]] == ["execution"] * 3
    assert D.advance.criterion == "outcome" and E.advance is None
    # stage A (execution gate): competent stand_hold draws advance the stage
    cur = Curriculum(start_index=0)
    for _ in range(10):
        cur.on_exchange(0, similarity=0.8, stood=True)
    cur.tick(A.advance.min_steps)
    assert cur.metric() == 1.0 and cur.successes() == 10
    assert cur.maybe_advance() and cur.stage.key == "B"
    # ... but draws without an execution measurement never do
    blind = Curriculum(start_index=0)
    for _ in range(200):
        blind.on_exchange(0)
    blind.tick(100_000)
    assert blind.metric() == 0.0 and blind.successes() == 0
    assert not blind.maybe_advance() and blind.stage.key == "A"
    # stage D (outcome gate): all-draw windows never advance (MOTOR_CURRICULUM §3)
    dcur = Curriculum(start_index=3)
    for _ in range(200):
        dcur.on_exchange(0, similarity=1.0, stood=True)   # best possible execution
    dcur.tick(100_000)
    assert dcur.metric() == 0.0 and not dcur.maybe_advance() and dcur.stage.key == "D"
    print("rl.curriculum self-check OK:", {"stages": [s.key for s in PISTY_STAGES],
                                           "A_weights": A.weights.as_dict(),
                                           "E_weights": E.weights.as_dict()})
