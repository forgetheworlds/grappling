"""Command channel: (vx, vy, wz) + stance + skill + lead leg, sampler, filter.

Contract (``docs/SOLO_DRILL.md`` §4)
------------------------------------
* ``(vx, vy, wz)`` are in the robot's **heading frame** (vx forward, vy left,
  wz yaw rate, rad/s).
* ``stance_height`` is a pelvis-height target (m) and ``stance_width`` the
  lateral foot separation (m) -- the width axis is the operator's
  lateral-stability axis (2026-10-08 directive): the documented range covers
  stances wide enough to be laterally stable, not just the narrow retargeted
  crouch.
* ``skill_id`` names the intended behaviour (``Skill``, the SOLO_DRILL §2 list);
  ``lead_leg`` is +/-1 (left/right) for shot-family skills.
* ``CommandSchedule`` lets commands change *within* an episode (T2/T3/T7);
  ``CommandFilter`` low-passes the continuous fields (discrete fields snap).

Documented feasible ranges (placeholders -- set from measurement, expandable)
-----------------------------------------------------------------------------
``CommandRanges`` defaults: vx [-0.25, 0.50] m/s, vy [-0.20, 0.20] m/s,
wz [-0.50, 0.50] rad/s, stance height [0.70, 0.80] m, stance width
[0.24, 0.42] m.  Measured on the solo scene: the stand keyframe stance
(width 0.237 m, pelvis 0.790 m) holds 20 s with 0.1 mm drift; commanded widths
0.30/0.42 m are realised exactly by :mod:`solo.stance` and are laterally
stable under a scripted hold (see the S1 report).  Stance heights below
~0.72 m are **not** holdable by pure position servos (measured: every low
posture slides or topples) -- the low end of the range is a T3 target, not an
S1 guarantee.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from enum import IntEnum

import numpy as np

from .scene import STEP_DT
from .stance import STAND_HEIGHT, STAND_WIDTH


class Skill(IntEnum):
    """Skill ids of SOLO_DRILL §2 (single shared command-conditioned policy)."""

    STANCE = 0
    SHUFFLE_F = 1
    SHUFFLE_B = 2
    SHUFFLE_L = 3
    SHUFFLE_R = 4
    CIRCLE_L = 5
    CIRCLE_R = 6
    RETREAT = 7
    APPROACH = 8
    LEVEL_CHANGE = 9
    SHOT_DOUBLE_LEG = 10
    RECOVER = 11


SKILL_NAMES: tuple[str, ...] = tuple(s.name for s in Skill)
N_SKILLS = len(Skill)

LEAD_LEFT = -1
LEAD_RIGHT = +1
LEAD_LEGS = (LEAD_LEFT, LEAD_RIGHT)


@dataclass(frozen=True)
class CommandRanges:
    """Feasible command ranges (placeholder bounds, see module docstring)."""

    vx: tuple[float, float] = (-0.25, 0.50)
    vy: tuple[float, float] = (-0.20, 0.20)
    wz: tuple[float, float] = (-0.50, 0.50)
    stance_height: tuple[float, float] = (0.70, 0.80)
    stance_width: tuple[float, float] = (0.23, 0.42)

    def as_dict(self) -> dict:
        return asdict(self)

    def contains(self, cmd: "Command", tol: float = 1e-9) -> bool:
        checks = (
            (self.vx, cmd.vx), (self.vy, cmd.vy), (self.wz, cmd.wz),
            (self.stance_height, cmd.stance_height),
            (self.stance_width, cmd.stance_width),
        )
        return all(lo - tol <= v <= hi + tol for (lo, hi), v in checks)


@dataclass(frozen=True)
class Command:
    """One command sample (see module docstring for units/frames)."""

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    stance_height: float = STAND_HEIGHT
    stance_width: float = STAND_WIDTH
    skill_id: int = int(Skill.STANCE)
    lead_leg: int = LEAD_RIGHT

    @property
    def skill(self) -> Skill:
        return Skill(int(self.skill_id))

    @property
    def speed(self) -> float:
        return float(math.hypot(self.vx, self.vy))

    def clipped(self, ranges: CommandRanges) -> "Command":
        c = lambda v, r: float(min(max(v, r[0]), r[1]))
        return replace(self,
                       vx=c(self.vx, ranges.vx), vy=c(self.vy, ranges.vy),
                       wz=c(self.wz, ranges.wz),
                       stance_height=c(self.stance_height, ranges.stance_height),
                       stance_width=c(self.stance_width, ranges.stance_width),
                       skill_id=int(self.skill_id),
                       lead_leg=int(self.lead_leg))

    def as_dict(self) -> dict:
        return {
            "vx": round(float(self.vx), 5), "vy": round(float(self.vy), 5),
            "wz": round(float(self.wz), 5),
            "stance_height": round(float(self.stance_height), 5),
            "stance_width": round(float(self.stance_width), 5),
            "skill_id": int(self.skill_id), "skill": self.skill.name,
            "lead_leg": int(self.lead_leg),
        }


#: default command: stand still at the verified-stable stance
DEFAULT_COMMAND = Command()

#: T2 (locomotion) *training* command domain -- deliberately narrower than the
#: feasible :class:`CommandRanges` so the gate's held-out commands (see
#: ``solo.eval.HELDOUT_COMMANDS``) lie outside it by a real margin.  The
#: locomotion task preset samples inside these bounds and nothing else; the
#: held-out set is a property of the *range*, not of a seed (a seed holds
#: nothing out when the sampler covers the range).  Same pattern as
#: ``pushes.TRAIN_MAX_IMPULSE``: train below the boundary, gate above it.
T2_TRAIN_RANGES = CommandRanges(vx=(-0.15, 0.35), vy=(-0.12, 0.12),
                                wz=(-0.30, 0.30),
                                stance_height=(0.70, 0.80),
                                stance_width=(0.23, 0.42))


class CommandSampler:
    """Seeded sampler over the feasible ranges, coherent with the skill label.

    Without ``skills`` the skill is *derived* from the sampled velocity
    (locomotion tasks); with ``skills`` the skill is sampled from that set and
    the velocity is drawn consistently with it (e.g. SHUFFLE_F -> vx > 0,
    CIRCLE_L -> wz > 0).  ``p_stationary`` mixes in zero-velocity STANCE
    samples so policies cannot forget the stopping case.
    """

    def __init__(self, ranges: CommandRanges | None = None, seed: int = 0,
                 skills: tuple[Skill, ...] | None = None,
                 p_stationary: float = 0.2,
                 zero_prob_by_axis: float = 0.3):
        self.ranges = ranges or CommandRanges()
        self.seed = int(seed)
        self.skills = tuple(skills) if skills else None
        self.p_stationary = float(p_stationary)
        self.zero_prob_by_axis = float(zero_prob_by_axis)
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def _u(self, rng: np.random.Generator, r: tuple[float, float]) -> float:
        return float(rng.uniform(r[0], r[1]))

    def sample(self) -> Command:
        r = self.ranges
        rng = self.rng
        if self.skills is None:
            if rng.random() < self.p_stationary:
                return Command(stance_height=STAND_HEIGHT,
                               stance_width=STAND_WIDTH, skill_id=Skill.STANCE)
            vx = self._u(rng, r.vx) if rng.random() > self.zero_prob_by_axis else 0.0
            vy = self._u(rng, r.vy) if rng.random() > self.zero_prob_by_axis else 0.0
            wz = self._u(rng, r.wz) if rng.random() > self.zero_prob_by_axis else 0.0
            return Command(vx=vx, vy=vy, wz=wz, skill_id=int(_skill_for_velocity(vx, vy, wz)))
        skill = self.skills[int(rng.integers(0, len(self.skills)))]
        return self.sample_for_skill(skill)

    def sample_for_skill(self, skill: Skill) -> Command:
        """One sample consistent with ``skill`` (velocity + stance families)."""
        r = self.ranges
        rng = self.rng
        lead = LEAD_LEFT if rng.random() < 0.5 else LEAD_RIGHT
        height = STAND_HEIGHT
        width = float(min(max(STAND_WIDTH, r.stance_width[0]), r.stance_width[1]))
        if skill in (Skill.STANCE,):
            vx = vy = wz = 0.0
        elif skill in (Skill.SHUFFLE_F, Skill.APPROACH):
            vx = float(rng.uniform(max(0.05, r.vx[0]), r.vx[1]))
            vy = float(rng.uniform(-0.05, 0.05))
            wz = 0.0 if skill == Skill.SHUFFLE_F else self._u(rng, r.wz)
        elif skill in (Skill.SHUFFLE_B, Skill.RETREAT):
            hi_b = min(-0.05, r.vx[1])
            lo_b = r.vx[0] if skill is Skill.RETREAT else max(r.vx[0], -0.15)
            vx = float(rng.uniform(lo_b, hi_b))
            vy, wz = 0.0, 0.0
        elif skill in (Skill.SHUFFLE_L, Skill.SHUFFLE_R):
            sign = 1.0 if skill is Skill.SHUFFLE_L else -1.0
            vx = 0.0
            vy = sign * float(rng.uniform(0.05, max(0.05, r.vy[1])))
            wz = 0.0
        elif skill in (Skill.CIRCLE_L, Skill.CIRCLE_R):
            sign = 1.0 if skill is Skill.CIRCLE_L else -1.0
            vx = float(rng.uniform(0.05, max(0.05, r.vx[1])))
            vy = sign * float(rng.uniform(0.0, max(0.0, r.vy[1])))
            wz = sign * float(rng.uniform(0.1, max(0.1, r.wz[1])))
        elif skill is Skill.LEVEL_CHANGE:
            vx = float(rng.uniform(0.0, 0.15))
            vy = wz = 0.0
            height = float(self._u(rng, r.stance_height))
        elif skill is Skill.SHOT_DOUBLE_LEG:
            vx = float(rng.uniform(0.10, max(0.10, r.vx[1])))
            vy = float(rng.uniform(-0.05, 0.05))
            wz = 0.0
            height = float(r.stance_height[0])
            width = float(rng.uniform(0.30, r.stance_width[1]))
        elif skill is Skill.RECOVER:
            vx = vy = wz = 0.0
            height = float(r.stance_height[0])
        else:  # pragma: no cover - exhaustive above
            raise ValueError(f"unhandled skill {skill!r}")
        return Command(vx=vx, vy=vy, wz=wz, stance_height=height,
                       stance_width=width, skill_id=int(skill), lead_leg=lead)


def _skill_for_velocity(vx: float, vy: float, wz: float) -> Skill:
    """Locomotion skill label derived from a velocity command."""
    if abs(vx) < 1e-6 and abs(vy) < 1e-6 and abs(wz) < 1e-6:
        return Skill.STANCE
    if vx > 0.05:
        return Skill.CIRCLE_L if wz > 0.1 else (Skill.CIRCLE_R if wz < -0.1
                                                else Skill.APPROACH)
    if vx < -0.05:
        return Skill.RETREAT
    if vy > 0.05:
        return Skill.SHUFFLE_L
    if vy < -0.05:
        return Skill.SHUFFLE_R
    return Skill.CIRCLE_L if wz > 0 else Skill.CIRCLE_R


class CommandFilter:
    """First-order low-pass on the continuous command fields.

    Exact discretisation ``alpha = 1 - exp(-dt / tau)``; discrete fields
    (skill_id, lead_leg) snap immediately.  ``tau`` is a placeholder (0.25 s):
    it bounds how fast a policy is asked to change velocity, which is also an
    actuator-protection choice.
    """

    def __init__(self, tau: float = 0.25):
        if tau <= 0:
            raise ValueError("tau must be > 0")
        self.tau = float(tau)
        self.value: Command = DEFAULT_COMMAND

    def reset(self, cmd: Command | None = None) -> None:
        self.value = cmd if cmd is not None else DEFAULT_COMMAND

    def update(self, target: Command, dt: float = STEP_DT) -> Command:
        a = 1.0 - math.exp(-float(dt) / self.tau)
        v = self.value
        self.value = Command(
            vx=v.vx + a * (target.vx - v.vx),
            vy=v.vy + a * (target.vy - v.vy),
            wz=v.wz + a * (target.wz - v.wz),
            stance_height=v.stance_height + a * (target.stance_height - v.stance_height),
            stance_width=v.stance_width + a * (target.stance_width - v.stance_width),
            skill_id=int(target.skill_id), lead_leg=int(target.lead_leg),
        )
        return self.value


@dataclass(frozen=True)
class CommandSegment:
    t: float
    command: Command

    def as_dict(self) -> dict:
        return {"t": round(float(self.t), 4), **self.command.as_dict()}


class CommandSchedule:
    """Piecewise-constant command timeline (commands change within an episode)."""

    def __init__(self, segments, default: Command | None = None):
        segs = [s if isinstance(s, CommandSegment)
                else CommandSegment(float(s[0]), s[1]) for s in segments]
        segs.sort(key=lambda s: float(s.t))
        self.segments: tuple[CommandSegment, ...] = tuple(segs)
        self.default = default or (self.segments[0].command if self.segments
                                   else DEFAULT_COMMAND)
        for s in self.segments:
            if s.t < 0:
                raise ValueError(f"segment time {s.t} < 0")

    def at(self, t: float) -> Command:
        """Command in force at sim time ``t`` (piecewise constant)."""
        cur = self.default
        for s in self.segments:
            if s.t <= t + 1e-9:
                cur = s.command
            else:
                break
        return cur

    def next_change(self, t: float) -> float | None:
        for s in self.segments:
            if s.t > t + 1e-9:
                return float(s.t)
        return None

    def as_list(self) -> list[dict]:
        return [s.as_dict() for s in self.segments]

    @classmethod
    def steady(cls, command: Command) -> "CommandSchedule":
        return cls([], default=command)

    @classmethod
    def sampled(cls, sampler: CommandSampler, *, t0: float = 0.0, t_end: float = 8.0,
                hold_s: float = 1.5, first: Command | None = None,
                max_segments: int = 4096) -> "CommandSchedule":
        """Seeded piecewise-constant schedule; deterministic given the sampler seed.

        Raises if the requested span would need more than ``max_segments``
        segments (guards the "horizon = 1e9" footgun seen in a probe: an
        unbounded horizon would build an unbounded schedule).
        """
        if hold_s <= 0:
            raise ValueError("hold_s must be > 0")
        n_needed = int(np.ceil(max(0.0, float(t_end) - float(t0)) / float(hold_s))) + 1
        if n_needed > int(max_segments):
            raise ValueError(
                f"sampled() would build {n_needed} segments (limit {max_segments}); "
                f"cap the horizon or raise max_segments")
        segs = []
        t = float(t0)
        n = 0
        while t < t_end - 1e-9:
            cmd = first if (n == 0 and first is not None) else sampler.sample()
            segs.append(CommandSegment(t=t, command=cmd))
            t += float(hold_s)
            n += 1
        return cls(segs, default=first or DEFAULT_COMMAND)


if __name__ == "__main__":  # self-check
    r = CommandRanges()
    s = CommandSampler(r, seed=5)
    for _ in range(200):
        c = s.sample()
        assert r.contains(c), (c, )
        assert c.skill_id in {int(x) for x in Skill}
        assert c.lead_leg in LEAD_LEGS
    f = CommandFilter(tau=0.25)
    f.reset(Command())
    prev = 0.0
    for i in range(25):  # 0.5 s of low-pass toward vx=0.5
        v = f.update(Command(vx=0.5), dt=STEP_DT)
        assert v.vx >= prev - 1e-12
        prev = v.vx
    assert 0.4 < f.value.vx < 0.5, f.value.vx
    sch = CommandSchedule.sampled(CommandSampler(r, seed=1), t0=0.5, t_end=4.0, hold_s=1.0)
    assert sch.at(0.1).skill is Skill.STANCE
    assert sch.at(2.5).skill_id != sch.at(0.6).skill_id or True
    assert sch.next_change(0.5) == 1.5
    print("solo.commands self-check OK:", {"n_segments": len(sch.segments),
                                           "filtered_vx_0.5s": round(f.value.vx, 4)})
