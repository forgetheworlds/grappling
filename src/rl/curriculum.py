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
BC checkpoint for warm start, and an advance rule (min steps + rolling
outcome-rate threshold).

Advance rule (deterministic): a stage advances once
``steps_in_stage >= rule.min_steps`` AND the rolling window holds at least
``rule.window_episodes`` exchanges AND ``metric >= rule.threshold``, where
``metric = (wins + 0.5 * draws) / n`` over the window (draws = timeouts and
ambiguous endings).

The *weights* below are the curriculum skeleton, not tuning results: they
encode the MISSION ordering (similarity decreases, outcome dominates late)
and are recorded in every checkpoint so later experiments are traceable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from .obs import TECHNIQUES, TechniqueCommand, reference_phase
from .reward import RewardWeights
from .scripted import OpponentSpec


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
class AdvanceRule:
    """When a stage is considered passed (see module docstring)."""

    min_steps: int = 20_000
    metric: str = "outcome_rate"
    threshold: float = 0.5
    window_episodes: int = 20

    def as_dict(self) -> dict:
        return asdict(self)


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
        advance=AdvanceRule(min_steps=20_000, threshold=0.5, window_episodes=10),
    ),
    StageConfig(
        key="B", name="randomized drilling",
        techniques=("DOUBLE_LEG",), command_on=True,
        opponent=OpponentSpec("stand_hold"),
        weights=RewardWeights(technique_similarity=0.5, progress=0.4, outcome=1.0, oob=1.0, engagement=0.0),
        perturbations=PerturbationSchedule(0.0, 0.03, ramp_steps=20_000, phase_jitter_s=0.3),
        advance=AdvanceRule(min_steps=30_000, threshold=0.5, window_episodes=20),
    ),
    StageConfig(
        key="C", name="dynamic drilling",
        techniques=("DOUBLE_LEG",), command_on=True,
        opponent=OpponentSpec("reference_replay", technique="STANCE", loop=True),
        weights=RewardWeights(technique_similarity=0.4, progress=0.3, outcome=1.0, oob=1.0, engagement=0.01),
        perturbations=PerturbationSchedule(0.03, 0.05, ramp_steps=30_000, phase_jitter_s=0.3),
        advance=AdvanceRule(min_steps=40_000, threshold=0.5, window_episodes=20),
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
        advance=AdvanceRule(min_steps=40_000, threshold=0.5, window_episodes=20),
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
    """Stage pointer + rolling outcome window + deterministic advance rule."""

    def __init__(self, stages: tuple[StageConfig, ...] = PISTY_STAGES, start_index: int = 0):
        if not 0 <= start_index < len(stages):
            raise ValueError(f"start_index {start_index} out of range for {len(stages)} stages")
        self.stages = tuple(stages)
        self.index = int(start_index)
        self.steps_in_stage = 0
        self.window: deque[int] = deque(maxlen=self.stages[self.index].advance.window_episodes
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

    def on_exchange(self, outcome: int) -> None:
        """Record one exchange result: +1 learner win, -1 loss, 0 draw."""
        o = int(np.sign(outcome))
        if len(self.window) == self.window.maxlen:
            self.window.popleft()
        self.window.append(o)

    # ------------------------------------------------------------------ metric
    def metric(self, name: str = "outcome_rate") -> float | None:
        if name != "outcome_rate":
            raise KeyError(name)
        if not self.window:
            return None
        w = sum(1 for x in self.window if x > 0)
        d = sum(1 for x in self.window if x == 0)
        return (w + 0.5 * d) / len(self.window)

    def maybe_advance(self) -> bool:
        """Advance if the current stage's rule is satisfied (returns True once)."""
        rule = self.stage.advance
        if rule is None or self.done:
            return False
        m = self.metric(rule.metric)
        if self.steps_in_stage < rule.min_steps:
            return False
        if m is None or len(self.window) < rule.window_episodes:
            return False
        if m < rule.threshold:
            return False
        self.index += 1
        self.steps_in_stage = 0
        next_rule = self.stage.advance
        self.window = deque(maxlen=next_rule.window_episodes if next_rule else 20)
        return True

    # ------------------------------------------------------------------ io
    def state_dict(self) -> dict:
        return {"index": self.index, "steps_in_stage": self.steps_in_stage,
                "window": list(self.window)}

    def load_state_dict(self, d: dict) -> None:
        idx = int(d.get("index", 0))
        if not 0 <= idx < len(self.stages):
            raise ValueError(f"checkpoint stage index {idx} out of range")
        self.index = idx
        self.steps_in_stage = int(d.get("steps_in_stage", 0))
        rule = self.stage.advance
        self.window = deque(maxlen=rule.window_episodes if rule else 20)
        for x in d.get("window", []):
            self.window.append(int(x))


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
    cur = Curriculum(start_index=0)
    for _ in range(10):
        cur.on_exchange(1)
    cur.tick(A.advance.min_steps)
    assert cur.metric() == 1.0
    assert cur.maybe_advance() and cur.stage.key == "B"
    print("rl.curriculum self-check OK:", {"stages": [s.key for s in PISTY_STAGES],
                                           "A_weights": A.weights.as_dict(),
                                           "E_weights": E.weights.as_dict()})
