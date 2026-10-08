"""Actor observation for Phase-5 PPO (deliverable 14 support).

Layout per robot (see :func:`actor_obs_layout`)::

    [ frame_stack * 84 env obs ]  [ technique one-hot (7) | phase (1) ]

* The first block is the environment's ``obs_fn`` output, repeated
  ``frame_stack`` times, oldest frame first, newest last.  ``frame_stack=1``
  is the default; 4 is the tested alternative (MISSION "Temporal information"
  suggests a small frame stack before any recurrence).
* The command block is the :func:`command_vector`: a 7-way technique one-hot
  (order :data:`TECHNIQUES`) plus the normalized phase in ``[0, 1]``.
  An inactive command (no technique) is all zeros.

The actor path is structurally separate from the critic path: this module
never touches ``MjData`` or any privileged simulator field.  Privileged
features live in :mod:`rl.privileged` and are concatenated only by the critic
builder (:class:`rl.privileged.CriticObsBuilder`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: technique order used by the one-hot; must match ``src/scorer/config.py``
#: ``TECHNIQUES`` (the scorer is imported lazily and by name, so a mismatch
#: degrades to a warning, never a silent index mix-up).
TECHNIQUES: tuple[str, ...] = (
    "DOUBLE_LEG", "SINGLE_LEG", "BODY_LOCK", "SNAPDOWN", "SPRAWL",
    "STAND_UP", "STANCE",
)
TECHNIQUE_INDEX = {t: i for i, t in enumerate(TECHNIQUES)}
N_TECHNIQUES = len(TECHNIQUES)

#: dimension of the environment's default observation (pinned to
#: ``wrestling.env.OBS_DIM`` by tests)
ENV_OBS_DIM = 84

#: one-hot (7) + phase (1)
COMMAND_DIM = N_TECHNIQUES + 1

ROBOTS = ("a", "b")


def actor_obs_dim(frame_stack: int = 1) -> int:
    """Total actor observation size for one robot."""
    _check_stack(frame_stack)
    return int(frame_stack) * ENV_OBS_DIM + COMMAND_DIM


def actor_obs_layout(frame_stack: int = 1) -> tuple[tuple[str, tuple[int, int]], ...]:
    """((name, (start, stop)), ...) describing :func:`actor_obs_dim` bytes."""
    _check_stack(frame_stack)
    out: list[tuple[str, tuple[int, int]]] = []
    for k in range(int(frame_stack)):
        out.append((f"env_obs_f{k}", (k * ENV_OBS_DIM, (k + 1) * ENV_OBS_DIM)))
    base = int(frame_stack) * ENV_OBS_DIM
    out.append(("technique_onehot", (base, base + N_TECHNIQUES)))
    out.append(("phase", (base + N_TECHNIQUES, base + COMMAND_DIM)))
    return tuple(out)


def _check_stack(frame_stack: int) -> None:
    if int(frame_stack) < 1:
        raise ValueError(f"frame_stack must be >= 1, got {frame_stack}")


def command_vector(technique: str | None, phase: float = 0.0) -> np.ndarray:
    """``(8,)`` float32 command: one-hot technique + normalized phase.

    ``technique=None`` -> all zeros (command off, free wrestling).
    """
    v = np.zeros(COMMAND_DIM, dtype=np.float32)
    if technique is None:
        return v
    if technique not in TECHNIQUE_INDEX:
        raise KeyError(f"unknown technique {technique!r}; known: {TECHNIQUES}")
    v[TECHNIQUE_INDEX[technique]] = 1.0
    v[N_TECHNIQUES] = float(np.clip(phase, 0.0, 1.0))
    return v


@dataclass(frozen=True)
class TechniqueCommand:
    """A commanded technique + phase for one robot at one control step."""

    technique: str | None = None
    phase: float = 0.0

    def __post_init__(self):
        if self.technique is not None and self.technique not in TECHNIQUE_INDEX:
            raise KeyError(f"unknown technique {self.technique!r}; known: {TECHNIQUES}")
        object.__setattr__(self, "phase", float(np.clip(self.phase, 0.0, 1.0)))

    @property
    def active(self) -> bool:
        return self.technique is not None

    def vector(self) -> np.ndarray:
        return command_vector(self.technique, self.phase)

    @staticmethod
    def off() -> "TechniqueCommand":
        return TechniqueCommand(None, 0.0)


def reference_duration(technique: str) -> float:
    """Duration (s) of ``data/refs/<technique>.npz`` (lazy import of the env)."""
    from wrestling.env import reference_trace  # local import: keeps obs.py light

    return float(reference_trace(technique)["t"][-1])


def reference_phase(technique: str, exchange_time: float, time_offset: float = 0.0) -> float:
    """Normalized progress of ``technique`` at ``exchange_time`` s into the exchange.

    ``phase = clip((exchange_time + time_offset) / duration, 0, 1)`` -- the
    command clock is a pure function of the exchange clock, so it needs no
    per-env state and is reproducible from ``(technique, exchange_time)`` alone.
    A zero-duration reference (never true in practice) yields 0.0.
    """
    duration = reference_duration(technique)
    if duration <= 0.0:
        return 0.0
    return float(np.clip((float(exchange_time) + float(time_offset)) / duration, 0.0, 1.0))


class ActorObsBuilder:
    """Per-robot actor observation with an optional frame stack.

    Usage::

        b = ActorObsBuilder(frame_stack=4)
        obs_a, obs_b = b.reset(env_obs_pair)              # env obs from env.reset()
        obs_a, obs_b = b.step(env_obs_pair, commands)     # commands: (TechniqueCommand | None) x 2
    """

    def __init__(self, frame_stack: int = 1):
        _check_stack(frame_stack)
        self.frame_stack = int(frame_stack)
        self.dim = actor_obs_dim(self.frame_stack)
        self._stack: dict[str, np.ndarray] | None = None

    # ---------------------------------------------------------------- layout
    def layout(self):
        return actor_obs_layout(self.frame_stack)

    # ------------------------------------------------------------------ state
    def reset(self, env_obs_pair, commands=None) -> tuple[np.ndarray, np.ndarray]:
        """Seed the stack with the current env obs (all frames = first obs)."""
        self._stack = {}
        for robot, obs in zip(ROBOTS, env_obs_pair):
            o = np.asarray(obs, dtype=np.float32).reshape(ENV_OBS_DIM)
            self._stack[robot] = np.repeat(o[None, :], self.frame_stack, axis=0)
        cmds = commands if commands is not None else (TechniqueCommand.off(), TechniqueCommand.off())
        return self._obs(*cmds)

    def reseed(self, env_obs_pair) -> None:
        """Pre-push seeding at an episode boundary: the *next* :meth:`step` with
        the same obs yields a stack of ``frame_stack`` copies (no stale frames)."""
        self.reset(env_obs_pair)

    def step(self, env_obs_pair, commands) -> tuple[np.ndarray, np.ndarray]:
        """Push new env obs, append the command block, return ``(obs_a, obs_b)``."""
        if self._stack is None:
            raise RuntimeError("ActorObsBuilder.step before reset")
        if len(commands) != 2:
            raise ValueError("commands must be a pair (a, b)")
        for robot, obs in zip(ROBOTS, env_obs_pair):
            o = np.asarray(obs, dtype=np.float32).reshape(ENV_OBS_DIM)
            stack = self._stack[robot]
            if self.frame_stack > 1:
                stack[:-1] = stack[1:]
            stack[-1] = o
        return self._obs(*commands)

    def _obs(self, *commands) -> tuple[np.ndarray, np.ndarray]:
        out = []
        for robot, cmd in zip(ROBOTS, commands):
            vec = cmd.vector() if isinstance(cmd, TechniqueCommand) else np.asarray(cmd, dtype=np.float32)
            if vec.shape != (COMMAND_DIM,):
                raise ValueError(f"command vector shape {vec.shape} != ({COMMAND_DIM},)")
            stacked = self._stack[robot].reshape(-1)
            out.append(np.concatenate([stacked, vec]).astype(np.float32))
        return out[0], out[1]

    def stacked(self, robot: str) -> np.ndarray:
        """Copy of the raw ``(frame_stack, 84)`` stack (tests/debug)."""
        if self._stack is None:
            raise RuntimeError("ActorObsBuilder not reset")
        return self._stack[robot].copy()


if __name__ == "__main__":  # self-check
    b = ActorObsBuilder(frame_stack=4)
    o = np.arange(ENV_OBS_DIM, dtype=np.float32)
    a, bb = b.reset((o, o + 1.0))
    assert a.shape == (actor_obs_dim(4),) and a.dtype == np.float32
    assert np.allclose(a[:ENV_OBS_DIM], o) and np.allclose(a[-COMMAND_DIM:], 0.0)
    a2, _ = b.step((o + 2.0, o), (TechniqueCommand("DOUBLE_LEG", 0.5), TechniqueCommand.off()))
    # newest frame last, oldest first
    assert np.allclose(a2[2 * ENV_OBS_DIM:3 * ENV_OBS_DIM], o)
    assert np.allclose(a2[3 * ENV_OBS_DIM:4 * ENV_OBS_DIM], o + 2.0)
    assert a2[-COMMAND_DIM + TECHNIQUE_INDEX["DOUBLE_LEG"]] == 1.0
    assert abs(float(a2[-1]) - 0.5) < 1e-6
    print("rl.obs self-check OK:", {"actor_obs_dim(1)": actor_obs_dim(1),
                                    "actor_obs_dim(4)": actor_obs_dim(4)})
