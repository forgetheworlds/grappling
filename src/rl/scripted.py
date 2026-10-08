"""Scripted opponents and reference base actions for Phase-5 training.

The learner's opponent before stage E is a scripted controller from the
environment's toolbox (``wrestling.env.StandHold`` / ``ReferenceReplay``) or,
when available, the phase-3 teacher (``src/teacher`` — imported lazily; a
requested teacher that cannot be built fails loudly, while the *technique
scorer* used for rewards degrades gracefully, see :mod:`rl.reward`).

``reference_base_ctrl`` samples ``data/refs/<technique>.npz`` joint targets at
normalized phase, for residual-on-reference action mode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from wrestling.env import ACT_SLICE, N_JOINTS, reference_trace

#: supported opponent kinds ("policy": the trainer runs a frozen policy net,
#: optionally loaded from ``checkpoint``)
KINDS = ("stand_hold", "reference_replay", "teacher", "policy", "none")


@dataclass(frozen=True)
class OpponentSpec:
    """Serializable opponent description used by :class:`rl.curriculum.StageConfig`."""

    kind: str = "stand_hold"
    technique: str | None = None   # reference_replay/teacher: which trace to track
    feedback_gain: float = 0.0     # ReferenceReplay PD feedback gain
    loop: bool = True              # ReferenceReplay: loop or hold-last
    checkpoint: str | None = None  # kind="policy": weights of the frozen opponent

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown opponent kind {self.kind!r}; known: {KINDS}")

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "OpponentSpec":
        known = set(OpponentSpec.__dataclass_fields__)
        return OpponentSpec(**{k: v for k, v in d.items() if k in known})

    def make(self, robot: str, model=None):
        """Build the controller callable ``(env, data) -> (29,)`` for ``robot``.

        Returns ``None`` for ``kind="none"`` (no controller; the environment
        then holds the exchange's start pose for that robot) and for
        ``kind="policy"`` (the trainer runs a frozen policy network instead
        of a scripted callable).
        """
        if self.kind in ("none", "policy"):
            return None
        if self.kind == "stand_hold":
            return StandHoldC(robot)
        if self.kind == "reference_replay":
            technique = self.technique or "STANCE"
            return ReplayC(technique, robot, self.feedback_gain, self.loop)
        if self.kind == "teacher":
            return _TeacherHalf(robot, self.technique or "STANCE", model)
        raise AssertionError(self.kind)


def StandHoldC(robot: str):
    """Late import wrapper so this module stays importable without mujoco env setup."""
    from wrestling.env import StandHold

    return StandHold(robot)


def ReplayC(technique: str, robot: str, feedback_gain: float = 0.0, loop: bool = True):
    from wrestling.env import ReferenceReplay

    return ReferenceReplay(technique, robot, feedback_gain=feedback_gain, hold_last=True, loop=loop)


class _TeacherHalf:
    """One robot's half of the phase-3 :class:`teacher.TeacherController` output.

    The teacher is optional and being built in parallel; a requested teacher
    that cannot be constructed raises ``RuntimeError`` with the underlying
    import/shape error (the *scorer* is the component that degrades silently).
    """

    def __init__(self, robot: str, technique: str, model):
        if model is None:
            from wrestling.env import load_wrestling_model

            model = load_wrestling_model()
        try:
            from teacher import TeacherController
        except Exception as exc:  # ImportError or a half-built module
            raise RuntimeError(f"opponent kind 'teacher' requested but src/teacher "
                               f"is not usable: {exc!r}") from exc
        trace = reference_trace(technique)
        self.tc = TeacherController(model=model, qpos_a=trace["qpos_a"],
                                    qpos_b=trace["qpos_b"], t_ref=trace["t"],
                                    technique=technique)
        self.slice = ACT_SLICE[robot]

    def __call__(self, env, data) -> np.ndarray:
        u = np.asarray(self.tc.control(data, float(env.exchange_time)), dtype=np.float64)
        if u.shape != (2 * N_JOINTS,):
            raise ValueError(f"teacher returned {u.shape}, expected ({2 * N_JOINTS},)")
        return u[self.slice]


_ref_cache: dict = {}


def reference_base_ctrl(technique: str, robot: str, phase: float) -> np.ndarray:
    """Reference joint targets (29,) at normalized ``phase`` in [0, 1].

    Cached ``data/refs`` lookup with linear interpolation between 50 Hz frames;
    the base action for ``action_mode="residual"``.
    """
    key = (technique, robot)
    if key not in _ref_cache:
        tr = reference_trace(technique)
        t = np.asarray(tr["t"], dtype=np.float64)
        q = np.asarray(tr["qpos_a"] if robot == "a" else tr["qpos_b"], dtype=np.float64)
        _ref_cache[key] = (t, q[:, 7:36].copy())
    t, q = _ref_cache[key]
    p = float(np.clip(phase, 0.0, 1.0))
    x = p * float(t[-1])
    if x <= t[0]:
        return q[0].copy()
    if x >= t[-1]:
        return q[-1].copy()
    i = int(np.searchsorted(t, x))
    w = (x - t[i - 1]) / (t[i] - t[i - 1])
    return ((1.0 - w) * q[i - 1] + w * q[i]).copy()


if __name__ == "__main__":  # self-check
    spec = OpponentSpec("stand_hold")
    c = spec.make("b")
    assert np.asarray(c(None, None)).shape == (29,)
    spec2 = OpponentSpec("reference_replay", technique="STANCE", loop=True)
    assert spec2.make("b").duration > 0
    base = reference_base_ctrl("DOUBLE_LEG", "a", 0.5)
    assert base.shape == (29,) and np.all(np.isfinite(base))
    assert np.allclose(reference_base_ctrl("DOUBLE_LEG", "a", 0.0),
                       reference_base_ctrl("DOUBLE_LEG", "a", -1.0))
    print("rl.scripted self-check OK:", {"kinds": KINDS,
                                         "base_mid_sum": float(np.abs(base).sum())})
