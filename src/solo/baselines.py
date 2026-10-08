"""Scripted baselines and exploit probes (S1, SOLO_DRILL §3/§10).

Controllers (all ``callable(env, data) -> (29,)`` ctrl targets, the same
absolute joint-position contract the policy's ``ActionMapper`` produces):

* :class:`StandHoldController` -- hold the verified-stable ``a_stand`` keyframe
  ctrl (the scripted stance baseline; stable with no disturbance, no push
  recovery).
* :class:`ZeroActionController` -- unit-zero action through
  ``env.ctrl_from_unit`` (what a zero-output policy sends: mid ``ctrlrange``).
* :class:`FallForwardController` -- fixed joint-target lean that topples the
  robot forward at the commanded speed ("fall with style" velocity-tracking
  exploit candidate).
* :class:`SquatController` -- repeated knee-flex oscillation (level-change /
  stance-reward farming candidate).
* :class:`RandomInitPolicyController` -- an untrained ``rl.net.Actor``
  (deterministic mean or seeded sampling) mapped through the standard action
  mapper.
* :class:`PolicyController` -- wraps a trained ``rl.net.Actor``/``ActorCritic``
  for later stages (same path, no separate eval plumbing).

Probe expectations (measured, see the S1 report): none of these is certified by
the task gates; the harness only certifies a controller that survives the
held-out push battery (T1) or tracks the commanded velocity while staying
upright (T2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .commands import Command, CommandSchedule, Skill
from .env import SoloEnv
from .obs import ACTOR_DIM
from .pushes import PushSpec
from .scene import N_JOINTS, STEP_DT, load_solo_model, stand_frame


class StandHoldController:
    """Hold the model's verified-stable stand keyframe targets."""

    name = "stand_hold"

    def __init__(self, model=None):
        m = model if model is not None else load_solo_model()
        _, ctrl = stand_frame(m)
        self.target = ctrl.copy()

    def __call__(self, env, data) -> np.ndarray:
        return env.action_from_ctrl(self.target)


class ZeroActionController:
    """Unit-zero action -> mid ``ctrlrange`` pose (no-op policy output)."""

    name = "zero_action"

    def __call__(self, env, data) -> np.ndarray:
        return env.ctrl_from_policy(np.zeros(N_JOINTS))


class FallForwardController:
    """Fixed lean targets: topples forward while producing forward base speed.

    Mimics the documented exploit "lean forward and fall at the requested
    speed".  Measured (solo scene, this host): ``ankle = -0.10 rad`` on both
    legs gives mean base vx +0.58 m/s (command 0.5 m/s) and topples forward
    (torso up-axis x-component reaches +1.0) at ~1.7 s -- i.e. it tracks the
    command while falling, and only the uprightness gate rejects it.
    """

    name = "fall_forward"

    def __init__(self, hip: float = 0.0, ankle: float = -0.10, model=None):
        m = model if model is not None else load_solo_model()
        _, ctrl = stand_frame(m)
        self.target = ctrl.copy()
        for off in (0, 6):
            self.target[off + 0] += hip     # hip_pitch
            self.target[off + 4] += ankle   # ankle_pitch
        lo, hi = (np.asarray(m.actuator_ctrlrange[:, 0], np.float64),
                  np.asarray(m.actuator_ctrlrange[:, 1], np.float64))
        self.target = np.clip(self.target, lo, hi)

    def __call__(self, env, data) -> np.ndarray:
        return env.action_from_ctrl(self.target)


class SquatController:
    """Repeated symmetric squats (knee flex with hip/ankle compensation).

    Measured (solo scene): amplitudes above ~0.15 rad topple the robot within
    ~1.2-1.5 s under pure position servos; the default (0.12 rad at 0.25 Hz)
    survives the episode and oscillates around the stand pose without ever
    reaching a commanded low stance -- the "repeated squatting" farming attempt.
    """

    name = "squat_repeat"

    def __init__(self, amplitude: float = 0.12, frequency: float = 0.25, model=None):
        m = model if model is not None else load_solo_model()
        _, self.base = stand_frame(m)
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)

    def __call__(self, env, data) -> np.ndarray:
        s = float(np.sin(2.0 * np.pi * self.frequency * float(env.data.time)))
        c = self.base.copy()
        a = self.amplitude * s
        for off in (0, 6):
            c[off + 3] += a        # knee
            c[off + 0] += 0.5 * a  # hip_pitch
            c[off + 4] -= 0.5 * a  # ankle_pitch
        return env.action_from_ctrl(np.clip(c, env.lo, env.hi))


class RandomInitPolicyController:
    """Untrained ``rl.net.Actor`` (the "random-init policy" baseline).

    ``stochastic=True`` samples from the initial Gaussian (seeded per episode);
    ``stochastic=False`` uses the deterministic mean.  Actions pass through
    ``env.ctrl_from_unit`` -- the identical mapping ``rl.net.ActionMapper``
    uses -- so the baseline exercises the real interface.
    """

    name = "random_init_policy"

    def __init__(self, seed: int = 0, stochastic: bool = True,
                 obs_dim: int = ACTOR_DIM, hidden=(256, 256)):
        import torch

        from rl.net import Actor, NetConfig

        torch.manual_seed(int(seed))
        self.actor = Actor(obs_dim=obs_dim, act_dim=N_JOINTS,
                           cfg=NetConfig(hidden=tuple(hidden)))
        self.stochastic = bool(stochastic)
        self._torch = torch
        self._gen = None

    def __call__(self, env, data) -> np.ndarray:
        obs = env.actor_vector().astype(np.float32)
        with self._torch.no_grad():
            t = self._torch.from_numpy(obs).unsqueeze(0)
            if self.stochastic:
                unit = self.actor.sample(t)["unit"].numpy()[0]
            else:
                unit = self.actor.deterministic_unit(t).numpy()[0]
        return env.ctrl_from_policy(unit)


class PolicyController:
    """Wrap a trained ``ActorCritic``/``Actor`` (deterministic by default)."""

    def __init__(self, policy, *, stochastic: bool = False, name: str = "policy",
                 device: str = "cpu"):
        import torch

        self.policy = policy
        self.torch = torch
        self.stochastic = bool(stochastic)
        self.name = name
        self.device = device

    def __call__(self, env, data) -> np.ndarray:
        obs = env.actor_vector().astype(np.float32)
        with self.torch.no_grad():
            t = self.torch.from_numpy(obs).unsqueeze(0).to(self.device)
            if self.stochastic:
                unit = self.policy.actor.sample(t)["unit"].cpu().numpy()[0]
            else:
                unit = self.policy.actor.deterministic_unit(t).cpu().numpy()[0]
        return env.ctrl_from_policy(unit)


class PlantedFootDragController:
    """T2 metric-discrimination probe: *sliding instead of stepping*.

    Holds the stand keyframe and translates the root every control step while
    both feet stay loaded, so the loaded feet are dragged along the floor.  It
    stays upright (measured mean uprightness 0.9998) and never terminates, and
    the **point-sampled** ``slip`` reads the same as a standing robot (MuJoCo's
    friction constraint zeroes the relative velocity within a substep while the
    positions still integrate).  It is the case the position-based
    ``slip_ratio`` criterion exists for: measured 1.75-2.0 m of loaded-foot
    travel per metre travelled, vs <=0.57 for a mechanism that moves the body
    while the loaded foot is *not* travelling.
    """

    name = "planted_foot_drag"

    def __init__(self, speed: float = 0.50):
        self.speed = float(speed)
        self.hold = StandHoldController()

    def __call__(self, env, data) -> np.ndarray:
        yaw = env._yaw()
        d = self.speed * STEP_DT
        env.data.qpos[0] += math.cos(yaw) * d
        env.data.qpos[1] += math.sin(yaw) * d
        return self.hold(env, data)


class AirborneTransferController:
    """T2 metric-discrimination probe: a **load-transfer** mechanism.

    Crouches, leaves the floor with both feet, translates the root only while
    airborne, then absorbs the landing -- so no loaded foot travels with the
    body.  Counterpart of :class:`PlantedFootDragController`: equal purpose
    (displacement), opposite slip-while-loaded signature (measured
    ``slip_ratio`` 0.22-0.42 with 3-9 step events, vs >=1.75 for the drag).

    This is a *metric probe*, not a gait and not a policy: it topples within
    ~1.5 s (measured) and fails the tracking/uprightness criteria of the T2
    gate.  It exists to show the slip criterion separates the two mechanisms.
    """

    name = "airborne_transfer"

    def __init__(self, speed: float = 0.20, period: float = 0.35,
                 crouch: float = 0.12, squash: float = 0.60):
        m = load_solo_model()
        _, ctrl = stand_frame(m)
        self.base = ctrl.copy()
        self.speed = float(speed)
        self.period = float(period)
        self.crouch = float(crouch)
        self.squash = float(squash)

    def __call__(self, env, data) -> np.ndarray:
        ph = (float(data.time) % self.period) / self.period
        target = self.base.copy()
        amp = self.crouch if ph < 0.5 else self.squash
        c = math.sin(math.pi * (ph if ph < 0.5 else ph - 0.5) / 0.5) * amp
        for off in (0, 6):
            target[off + 3] += c            # knee flex (crouch / absorb)
            target[off + 0] += -0.5 * c     # hip compensation
        if 0.25 < ph < 0.60:                # airborne: this is the "step"
            yaw = env._yaw()
            d = self.speed * STEP_DT
            env.data.qpos[0] += math.cos(yaw) * d
            env.data.qpos[1] += math.sin(yaw) * d
        return env.action_from_ctrl(np.clip(target, env.lo, env.hi))


#: forward-command schedule for the T2 probes (steady 0.3 m/s shuffle)
FORWARD_COMMAND = CommandSchedule.steady(
    Command(vx=0.30, vy=0.0, wz=0.0, skill_id=int(Skill.SHUFFLE_F)))
#: faster forward command for the fall-forward exploit (its lean delivers ~0.58)
EXPLOIT_FORWARD_COMMAND = CommandSchedule.steady(
    Command(vx=0.50, vy=0.0, wz=0.0, skill_id=int(Skill.SHUFFLE_F)))
#: level-change command for the squat-farming probe
LEVEL_COMMAND = CommandSchedule.steady(
    Command(vx=0.0, vy=0.0, wz=0.0, stance_height=0.70,
            skill_id=int(Skill.LEVEL_CHANGE)))


@dataclass(frozen=True)
class ProbeSpec:
    """A named exploit probe: controller + task + command + expectation."""

    name: str
    description: str
    task: str
    controller_factory: object
    command: CommandSchedule | None = None
    push: PushSchedule | None = None
    expect: str = "not_certified"


def probe_specs(*, include_random_init: bool = True) -> list[ProbeSpec]:
    """The four brief-mandated exploit probes (+ random-init baseline)."""
    from .pushes import PushSchedule

    battery = PushSchedule([PushSpec(t=1.0, impulse=10.0, direction=np.pi / 2,
                                     height=0.95, label="probe_T1")])
    out = [
        ProbeSpec("zero_action", "unit-zero action (mid ctrlrange pose)",
                  "balance", lambda env, seed: ZeroActionController()),
        ProbeSpec("stand_hold", "scripted hold of the stand keyframe",
                  "balance", lambda env, seed: StandHoldController(), push=battery),
        ProbeSpec("fall_forward", "lean forward and topple at commanded speed",
                  "locomotion", lambda env, seed: FallForwardController(),
                  command=EXPLOIT_FORWARD_COMMAND),
        ProbeSpec("squat_repeat", "repeated squatting (level-change farming)",
                  "stance", lambda env, seed: SquatController(),
                  command=LEVEL_COMMAND),
    ]
    if include_random_init:
        out.append(ProbeSpec("random_init_policy",
                             "untrained network, sampled action",
                             "balance",
                             lambda env, seed: RandomInitPolicyController(seed=seed)))
    return out


if __name__ == "__main__":  # self-check
    env = SoloEnv(seed=0)
    for ctrl in (StandHoldController(), ZeroActionController(),
                 FallForwardController(), SquatController(),
                 RandomInitPolicyController(seed=0)):
        a = ctrl(env, env.data)
        assert np.asarray(a).shape == (N_JOINTS,), (ctrl.name, np.shape(a))
    # StandHold must reproduce the keyframe ctrl exactly
    sh = StandHoldController()
    _, ctrl = stand_frame(env.model)
    assert np.allclose(sh.target, ctrl)
    print("solo.baselines self-check OK:", {"controllers": 5,
                                            "n_probes": len(probe_specs())})
