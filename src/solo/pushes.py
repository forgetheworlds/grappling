"""Push / perturbation machinery for the solo drill.

Physics
-------
An impulse is applied as a *constant Cartesian force on the torso body* for a
whole number of control steps, written through :func:`mujoco.mj_applyFT` into
``data.qfrc_applied`` at a chosen world point (``PushSpec.height``).  Measured
on this host (2026-10-08, solo scene):

* ``qfrc_applied`` **persists across** ``mj_step`` -- one write per control step
  is applied to all 10 substeps (verified: ``|qfrc|`` unchanged after 10 steps),
  so the env writes it every control step and clears it every control step.
* ``data.xfrc_applied`` also persists (verified: 50 N still present 21 steps
  later) -- the env clears **both** arrays every control step; nothing is ever
  left "unset".
* Application height is physical: the same J = 12 N*s at z = 0.95 m topples the
  robot (tilt 91.7 deg after 1.5 s), at z = 0.79 m it does not (tilt 12.5 deg).
* The analytic non-stepping ceiling for the G1 is **direction-dependent**, not a
  single number.  Derived from this model (m = 33.3411 kg, CoM height
  z_c = 0.6919 m) and the measured sole hull via the capture-point result
  ``J = m*sqrt(g/z_c)*dCOP`` (Yang et al. 2020, eq. 6), it spans **6.7 .. 20.7
  N*s** across the 8 push yaws: ~18-21 N*s laterally/toward the toes, only
  ~6.7 N*s for a sagittal push that drives the CoM back over the heels (the CoM
  sits heel-ward of the foot midpoint).  The previously stated "~13 N*s" scalar
  was wrong; see ``reports/2026-10-08/t1_gate_calibration.md`` section 2.2.

Force magnitude is ``impulse / duration``; the realized impulse is reported
with ``PushSpec.realized_impulse``.  No force ramps: discrete constant-force
windows only (prior-art failure checklist).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from .scene import STEP_DT, TORSO_BODY, body_id

_EPS = 1e-12


#: impulses above this are **held out of training**.  The cap is set to the top
#: of the training curriculum, NOT a physics bound: the derived non-stepping
#: ceiling is direction-dependent (6.7-20.7 N*s, see the module docstring), so
#: 12 N*s already exceeds it in the weakest (sagittal) directions.
TRAIN_MAX_IMPULSE = 12.0


@dataclass(frozen=True)
class PushSpec:
    """One scheduled push: impulse magnitude (N*s) at a world point.

    ``direction`` is the world-frame yaw (rad) of the force direction unless
    ``heading_relative`` is set, in which case it is relative to the robot's
    heading at application time (0 = push in the direction the robot faces).
    ``height`` is the world z of the application point; the point's xy is the
    torso body's current xy.  ``duration`` is quantized to whole control steps.
    """

    t: float
    impulse: float
    direction: float
    height: float = 0.95
    duration: float = STEP_DT
    heading_relative: bool = False
    body: str = TORSO_BODY
    label: str = ""

    @property
    def n_steps(self) -> int:
        return max(1, int(round(self.duration / STEP_DT)))

    @property
    def force_magnitude(self) -> float:
        return float(self.impulse) / (self.n_steps * STEP_DT)

    @property
    def realized_impulse(self) -> float:
        return self.force_magnitude * self.n_steps * STEP_DT

    def force_vector(self, heading: float = 0.0) -> np.ndarray:
        """World-frame force vector (3,) for the given robot heading (rad)."""
        ang = float(self.direction) + (float(heading) if self.heading_relative else 0.0)
        f = self.force_magnitude
        return np.array([f * np.cos(ang), f * np.sin(ang), 0.0])

    def as_dict(self) -> dict:
        return {
            "t": round(float(self.t), 4), "impulse": round(float(self.impulse), 4),
            "direction_deg": round(float(np.degrees(self.direction)), 2),
            "height": round(float(self.height), 4),
            "duration": round(float(self.n_steps * STEP_DT), 4),
            "heading_relative": bool(self.heading_relative),
            "label": self.label,
        }


def clear_applied(data: mujoco.MjData) -> None:
    """Zero both external-force slots (called every control step)."""
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0


def apply_push(model: mujoco.MjModel, data: mujoco.MjData, spec: PushSpec,
               heading: float = 0.0) -> dict:
    """Add this control step's force to ``data.qfrc_applied``; returns a record.

    Must be preceded by :func:`clear_applied` in the same control step.
    """
    bid = body_id(model, spec.body)
    force = spec.force_vector(heading)
    p = np.asarray(data.xpos[bid], dtype=np.float64)
    point = np.array([p[0], p[1], float(spec.height)])
    mujoco.mj_applyFT(model, data, force, np.zeros(3), point, bid,
                      data.qfrc_applied)
    return {
        "force": [round(float(x), 4) for x in force],
        "point": [round(float(x), 4) for x in point],
        "qfrc_norm": round(float(np.abs(data.qfrc_applied).sum()), 6),
        "spec": spec.as_dict(),
    }


class PushSchedule:
    """Deterministic ordered push list; active windows are control-step exact."""

    def __init__(self, pushes=()):
        self.pushes: tuple[PushSpec, ...] = tuple(
            sorted(pushes, key=lambda p: (float(p.t), p.label)))
        self.reset()

    def reset(self) -> None:
        self._applied = 0  # pushes already fully applied (for next_push queries)

    def __len__(self) -> int:
        return len(self.pushes)

    def active(self, t: float) -> list[PushSpec]:
        """Pushes whose force window contains sim time ``t`` (step start)."""
        out = []
        for p in self.pushes:
            if p.t - _EPS <= t < p.t + p.n_steps * STEP_DT - _EPS:
                out.append(p)
        return out

    def next_after(self, t: float) -> PushSpec | None:
        """Earliest push starting strictly after ``t`` (None if exhausted)."""
        for p in self.pushes:
            if p.t > t + _EPS:
                return p
        return None

    def active_index(self, t: float) -> list[int]:
        return [i for i, p in enumerate(self.pushes)
                if p.t - _EPS <= t < p.t + p.n_steps * STEP_DT - _EPS]

    def as_list(self) -> list[dict]:
        return [p.as_dict() for p in self.pushes]

    # ------------------------------------------------------------------ builders
    @classmethod
    def battery(cls, *, magnitudes=(4.0, 6.0, 8.0, 10.0, 12.0), directions=16,
                seed: int = 0, t: float = 1.0, jitter: float = 0.25,
                height: float = 0.95, duration: float = STEP_DT,
                magnitude_jitter: float = 0.0, label: str = "battery") -> "PushSchedule":
        """Seeded held-out push battery: ``directions`` evenly spaced world yaw
        angles x ``magnitudes`` (N*s), randomized start time in ``[t, t+jitter]``
        and (optional) magnitude jitter.  Deterministic given ``seed``.

        The *highest* magnitude is the "unseen in training" point of the T1 gate:
        training schedules must stay below it (see ``docs/MOTOR_CURRICULUM.md``
        gate discipline).
        """
        rng = np.random.default_rng(int(seed))
        slots = max(1, int(round(float(jitter) / STEP_DT)))
        pushes = []
        for m in magnitudes:
            for k in range(int(directions)):
                ang = 2.0 * np.pi * k / int(directions)
                mag = float(m) * (1.0 + float(magnitude_jitter) * rng.uniform(-1, 1))
                t0 = float(t) + STEP_DT * int(rng.integers(0, slots + 1))
                pushes.append(PushSpec(t=t0, impulse=mag, direction=ang,
                                       height=height, duration=duration,
                                       label=f"{label}_m{round(m, 3)}_d{k}"))
        return cls(pushes)

    @classmethod
    def sequence(cls, specs, *, t0: float = 0.0, gap: float = 1.5) -> "PushSchedule":
        """Pushes from ``(impulse, direction, height)`` tuples, evenly spaced."""
        out = []
        for i, s in enumerate(specs):
            imp, ang = float(s[0]), float(s[1])
            h = float(s[2]) if len(s) > 2 else 0.95
            out.append(PushSpec(t=t0 + i * gap, impulse=imp, direction=ang,
                                height=h, label=f"seq{i}"))
        return cls(out)


if __name__ == "__main__":  # self-check
    import mujoco as mj

    from .scene import load_solo_model, stand_frame

    model = load_solo_model()
    data = mj.MjData(model)
    q, c = stand_frame(model)
    data.qpos[:] = q
    data.ctrl[:] = c

    # gravity-off, in-air scaling check: momentum == J exactly (no clamping)
    model.opt.gravity[:] = 0.0
    data.qpos[2] = 2.0
    mj.mj_forward(model, data)
    M = float(model.body_mass.sum())
    spec = PushSpec(t=0.0, impulse=6.0, direction=0.0, height=2.5)
    clear_applied(data)
    rec = apply_push(model, data, spec)
    mmat = np.zeros((model.nv, model.nv))
    for _ in range(10):
        mj.mj_step(model, data)
    mj.mj_fullM(model, data, mmat)
    mom = mmat @ np.asarray(data.qvel, dtype=np.float64)
    print("push scaling:", {"J": spec.realized_impulse, "M": round(M, 3),
                            "momentum_x": round(float(mom[0]), 4),
                            "expected": spec.realized_impulse,
                            "v_com": round(float(mom[0] / M), 4)})
    assert abs(mom[0] - spec.realized_impulse) < 0.01 * spec.realized_impulse, mom[0]
    assert rec["qfrc_norm"] > 0.0
    model.opt.gravity[:] = [0.0, 0.0, -9.81]

    # clearing zeroes both slots
    data.xfrc_applied[1, 0] = 5.0
    data.qfrc_applied[0] = 1.0
    clear_applied(data)
    assert not data.qfrc_applied.any() and not data.xfrc_applied.any()

    # schedule windows are step exact, battery deterministic
    s1 = PushSchedule.battery(seed=3)
    s2 = PushSchedule.battery(seed=3)
    assert s1.as_list() == s2.as_list()
    seq = PushSchedule.sequence([(6.0, 0.0), (8.0, np.pi / 2)], gap=1.5)
    one = seq.pushes[0]
    assert seq.active(one.t) == [one] and not seq.active(one.t + STEP_DT + 1e-6)
    assert seq.active(one.t + STEP_DT - 1e-6) == [one]
    assert seq.next_after(one.t).label == "seq1"
    print("solo.pushes self-check OK:", {"n_battery": len(s1),
                                         "momentum_x": round(float(mom[0]), 4)})
