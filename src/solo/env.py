"""``SoloEnv``: single-G1 50 Hz control environment for the drill (S1).

Contract
--------
* Scene: :mod:`solo.scene` (``robots/g1/g1.xml`` composed with the ``a_``
  prefix + floor + virtual-opponent markers).  Physics timestep 0.002 s
  (verified at load); **one control step = 10 physics substeps = 20 ms**
  (50 Hz).  ``env.control_dt == 0.02``.
* Action: 29 joint-position targets in rad.  ``action_mode="absolute"`` clips
  them to ``ctrlrange``; ``action_mode="residual"`` treats them as ``z`` and
  applies ``ctrl = clip(base + residual_scale * tanh(z), ctrlrange)`` with
  ``base`` = the reset joint targets (``env.set_base_action`` overrides).
  ``env.ctrl_from_unit(u)`` is the shared unit-action -> ctrl helper
  (``mid + half * clip(u)``), matching ``rl.net.ActionMapper``.
* Reset: the verified-stable ``a_stand`` keyframe + documented jitter (joint
  noise +/-0.03 rad, base xy +/-0.02 m, yaw +/-10 deg, root tilt +/-2 deg);
  the measurement of zero-action holdability of this distribution is part of
  the S1 evidence (``scripts/solo_env_smoke.py holdability``).
* Command: ``(vx, vy, wz)`` heading frame + stance height/width + skill +
  lead leg; schedules change it within an episode; a first-order low-pass
  (``tau = 0.25 s``) yields the command the metrics/observations use.
* Pushes: ``PushSchedule`` (magnitude/direction/height/time), applied through
  ``mj_applyFT`` into ``qfrc_applied``; **both force arrays are cleared every
  control step** (measured: MuJoCo does not clear them).
* Termination: dorsal back-to-mat (``fall.DorsalDetector``, the failed-attempt
  verdict) or the balance fall rule (``fall.FallDetector``); the horizon is a
  *truncation*, not a termination.  Knees/hands/self-chosen low postures are
  not terminal by themselves.
* Observation: ``env.observation()`` returns ``{"actor": (115,),
  "privileged": (43,), "critic": (158,)}`` float32, fixed by :mod:`solo.obs`;
  the actor never sees contacts/markers (tested).
* Metrics: every control step records the fields of :mod:`solo.metrics`;
  ``env.recorder.rows`` holds them; ``env.recorder.summary()`` aggregates.

This environment is the S1 harness: it is deterministic given a seed
(``reset(seed)`` + fixed schedules), and it exposes ``config()`` (a JSON-able
description) so evaluation runs and checkpoints can echo exact conditions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from . import markers as markers_mod
from .curriculum import DEFAULT_CURRICULUM, PushCurriculum
from .commands import (DEFAULT_COMMAND, N_SKILLS, T2_TRAIN_RANGES, Command,
                       CommandFilter, CommandRanges, CommandSampler,
                       CommandSchedule, Skill)
from .fall import (ContactState, DorsalDetector, FallDetConfig, FallDetector,
                   fall_features)
from .lit import JointMask, joint_names, sole_geom_ids, sole_points_world
from .markers import DEFAULT_PLAN, SHOT_PLAN, MarkerPlan
from .metrics import (MetricsRecorder, action_smoothness, foot_slip,
                      joint_limit_proximity, local_xy, saturation_fraction)
from .obs import (ACTOR_DIM, CRITIC_DIM, PRIV_DIM, ObsContext, actor_obs,
                  critic_obs, privileged_obs)
from .pushes import PushSchedule, apply_push, clear_applied
from .reward import RewardInputs, RewardWeights, TaskReward
from .scene import (FOOT_BODIES, MODEL_DT, N_JOINTS, PELVIS_BODY, STEP_DT,
                    SUBSTEPS, TORSO_BODY, body_id, ctrl_range, joint_dof_slice,
                    joint_qpos_slice, load_solo_model, model_facts, site_id,
                    stand_frame)
from .stance import STAND_HEIGHT, STAND_WIDTH, stance_qpos, stance_targets

#: reset jitter (measured holdable; see report)
RESET_JOINT_NOISE = 0.03      # rad
RESET_XY_NOISE = 0.02         # m
RESET_YAW_DEG = 10.0          # deg
RESET_TILT_DEG = 2.0          # deg (root pitch/roll)

#: recovery bookkeeping
RECOVERY_PELVIS_Z = 0.50      # below this the robot counts as "down"
RECOVERY_HOLD_S = 0.3         # stance kept this long -> recovered
RECOVERY_TOL = 0.06           # m tolerance vs stand height
#: a foot landing after at least this much air time counts as a *step* event
STEP_EVENT_MIN_AIR = 0.06     # s

#: shot bookkeeping
SHOT_PHASE_RATE = 1.0 / 1.2   # phase units per second (1.2 s to full entry)
SHOT_ENTER_DEPTH = 0.25
SHOT_EXIT_DEPTH = 0.30


@dataclass
class TaskSpec:
    """Task family preset (reward set + command/push/termination defaults)."""

    name: str = "balance"
    horizon: float = 8.0
    command: CommandSchedule | None = None
    command_sampler: CommandSampler | None = None
    command_hold_s: float = 1.5
    push: PushSchedule | None = None
    marker_plan: MarkerPlan | None = None
    terminate_on_fall: bool = True
    terminate_on_dorsal: bool = True
    start: str = "stand"          # "stand" | "crouch" | "kneel"
    shot_budget_s: float = 2.5
    push_curriculum: PushCurriculum | None = None   # training-time push ramp

    def as_dict(self) -> dict:
        return {
            "name": self.name, "horizon": self.horizon,
            "push_curriculum": (None if self.push_curriculum is None
                                else self.push_curriculum.as_dict()),
            "command_hold_s": self.command_hold_s,
            "start": self.start, "shot_budget_s": self.shot_budget_s,
            "command_schedule": None if self.command is None else self.command.as_list(),
            "command_sampler": None if self.command_sampler is None else {
                "seed": self.command_sampler.seed,
                "ranges": self.command_sampler.ranges.as_dict(),
                "skills": None if self.command_sampler.skills is None
                else [s.name for s in self.command_sampler.skills],
            },
            "push": None if self.push is None else self.push.as_list(),
            "push_curriculum": (None if self.push_curriculum is None
                                else self.push_curriculum.as_dict()),
            "marker_plan": None if self.marker_plan is None
            else self.marker_plan.as_dict(),
            "terminate_on_fall": self.terminate_on_fall,
            "terminate_on_dorsal": self.terminate_on_dorsal,
        }


_LOCOMOTION_SKILLS = (Skill.STANCE, Skill.SHUFFLE_F, Skill.SHUFFLE_B,
                      Skill.SHUFFLE_L, Skill.SHUFFLE_R, Skill.CIRCLE_L,
                      Skill.CIRCLE_R, Skill.RETREAT, Skill.APPROACH)


def _preset_tasks() -> dict[str, TaskSpec]:
    return {
        "balance": TaskSpec(name="balance", horizon=8.0,
                            command=CommandSchedule.steady(DEFAULT_COMMAND),
                            push_curriculum=DEFAULT_CURRICULUM),
        "locomotion": TaskSpec(
            name="locomotion", horizon=8.0,
            # sample the T2 TRAINING domain only: the gate's held-out commands
            # (solo.eval.HELDOUT_COMMANDS) sit outside these bounds, so the
            # default locomotion task can never train on them.
            command_sampler=CommandSampler(ranges=T2_TRAIN_RANGES,
                                           skills=_LOCOMOTION_SKILLS, seed=0),
            command_hold_s=1.5),
        "stance": TaskSpec(
            name="stance", horizon=8.0,
            command_sampler=CommandSampler(
                seed=0, skills=(Skill.STANCE, Skill.LEVEL_CHANGE, Skill.SHUFFLE_F)),
            command_hold_s=2.0),
        "reach": TaskSpec(
            name="reach", horizon=8.0,
            command_sampler=CommandSampler(
                seed=0, skills=(Skill.APPROACH, Skill.STANCE)),
            command_hold_s=2.0, marker_plan=DEFAULT_PLAN),
        "shot": TaskSpec(
            name="shot", horizon=6.0,
            command_sampler=CommandSampler(seed=0, p_stationary=0.0,
                                           skills=(Skill.SHOT_DOUBLE_LEG,)),
            command_hold_s=6.0, marker_plan=SHOT_PLAN),
        "recovery": TaskSpec(name="recovery", horizon=8.0, start="crouch",
                             command=CommandSchedule.steady(
                                 Command(skill_id=int(Skill.RECOVER)))),
    }


TASKS: dict[str, TaskSpec] = _preset_tasks()


class SoloEnv:
    """Single-G1 solo-drill environment (see module docstring for the contract)."""

    def __init__(self, model: mujoco.MjModel | None = None, *, task: str | TaskSpec = "balance",
                 seed: int = 0, action_mode: str = "absolute",
                 residual_scale: float = 0.5, horizon: float | None = None,
                 push: PushSchedule | None = None,
                 command: CommandSchedule | None = None,
                 command_sampler: CommandSampler | None = None,
                 marker_plan: MarkerPlan | None = None,
                 fall_cfg: FallDetConfig | None = None,
                 weights: RewardWeights | None = None, gamma: float | None = None,
                 term_set: str | None = None,
                 joint_mask: "JointMask | None" = None,
                 terminate_on_fall: bool | None = None,
                 terminate_on_dorsal: bool | None = None,
                 record_metrics: bool = True, jitter: bool = True):
        self.model = model if model is not None else load_solo_model()
        if abs(self.model.opt.timestep - MODEL_DT) > 1e-12:
            raise ValueError(f"model timestep {self.model.opt.timestep} != {MODEL_DT}")
        self.data = mujoco.MjData(self.model)
        spec = task if isinstance(task, TaskSpec) else TASKS[task]
        self.spec = spec
        self.task = spec.name
        self.action_mode = action_mode
        if action_mode not in ("absolute", "residual"):
            raise ValueError(action_mode)
        self.residual_scale = float(residual_scale)
        self.horizon = float(spec.horizon if horizon is None else horizon)
        self.marker_plan = marker_plan or spec.marker_plan or DEFAULT_PLAN
        self.fall_det = FallDetector(fall_cfg)
        self.dorsal_det = DorsalDetector()
        self.reward = TaskReward(self.task, weights, gamma, term_set=term_set)
        #: optional action-space restriction (solo.lit.JointMask): frozen joints
        #: are written to their ``a_stand`` keyframe ctrl by ``resolve_action``,
        #: the single choke point training AND evaluation share, so a masked run
        #: and its evaluation cannot diverge.  ``None`` (default) = full action.
        self.joint_mask = joint_mask
        self.terminate_on_fall = (spec.terminate_on_fall if terminate_on_fall is None
                                  else bool(terminate_on_fall))
        self.terminate_on_dorsal = (spec.terminate_on_dorsal if terminate_on_dorsal is None
                                    else bool(terminate_on_dorsal))
        self.record_metrics = bool(record_metrics)
        self.jitter_reset = bool(jitter)

        self.push_curriculum = spec.push_curriculum
        self.ranges = (command_sampler.ranges if command_sampler is not None
                       else (spec.command_sampler.ranges if spec.command_sampler is not None
                             else CommandRanges()))
        self._push_default = push if push is not None else spec.push
        self._command_schedule = command
        self._command_sampler = command_sampler or spec.command_sampler
        self.command_filter = CommandFilter()

        self._q_stand, self._ctrl_stand = stand_frame(self.model)
        self.lo, self.hi = ctrl_range(self.model)
        self.action_dim = N_JOINTS
        self._joint_q = joint_qpos_slice(self.model)
        self._joint_v = joint_dof_slice(self.model)
        self._torso_bid = body_id(self.model, TORSO_BODY)
        self._pelvis_bid = body_id(self.model, PELVIS_BODY)
        self._foot_sites = (site_id(self.model, "a_left_foot"),
                            site_id(self.model, "a_right_foot"))
        self._wrist_sites = {"left": site_id(self.model, "a_left_wrist"),
                             "right": site_id(self.model, "a_right_wrist")}
        #: sole contact spheres (4 per foot) for the literature support-polygon
        #: terms -- the same geoms ``solo.lit.measure_ceiling`` uses
        self._sole_geoms = sole_geom_ids(self.model)
        #: foot bodies for the per-foot vertical GRF (literature GRF term)
        self._foot_bids = (body_id(self.model, FOOT_BODIES[0]),
                           body_id(self.model, FOOT_BODIES[1]))
        #: body masses, cached for the mass-weighted CoM velocity
        self._body_mass = np.asarray(self.model.body_mass, dtype=np.float64)
        self._mass_total = max(float(self._body_mass.sum()), 1e-9)
        #: arm-joint indices (source A's arm-posture term reads their deviation
        #: from the stand keyframe); arms are the joints no balance term needs
        self._arm_idx = np.array(
            [i for i, n in enumerate(joint_names(self.model))
             if n.startswith(("left_shoulder", "right_shoulder", "left_elbow",
                              "right_elbow", "left_wrist", "right_wrist"))],
            dtype=int)
        self._base_action = self._ctrl_stand.copy()
        self.seed = int(seed)
        self.recorder = MetricsRecorder(task=self.task, seed=self.seed)
        self.episode = 0
        self.reset(seed=self.seed)

    # ------------------------------------------------------------------ config
    def config(self) -> dict:
        """JSON-able description of the exact conditions (eval/checkpoints)."""
        return {
            "task": self.task, "seed": self.seed, "horizon": self.horizon,
            "action_mode": self.action_mode, "residual_scale": self.residual_scale,
            "jitter": self.jitter_reset,
            "ranges": self.ranges.as_dict(),
            "reward": self.reward.as_dict(),
            "joint_mask": (None if self.joint_mask is None
                           else self.joint_mask.as_dict()),
            "fall": self.fall_det.cfg.as_dict(),
            "dorsal": self.dorsal_det.cfg.as_dict(),
            "marker_plan": self.marker_plan.as_dict(),
            "spec": self.spec.as_dict(),
            "model": model_facts(self.model),
            "reset_jitter": {"joint_noise": RESET_JOINT_NOISE,
                             "xy_noise": RESET_XY_NOISE,
                             "yaw_deg": RESET_YAW_DEG,
                             "tilt_deg": RESET_TILT_DEG},
        }

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None, *, pose: np.ndarray | None = None,
              push: PushSchedule | None = None, command: CommandSchedule | None = None,
              jitter: bool | None = None):
        """Reset to the standing distribution; returns the observation dict."""
        if seed is not None:
            self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        jit = self.jitter_reset if jitter is None else bool(jitter)
        mujoco.mj_resetData(self.model, self.data)
        if pose is not None:
            q = np.asarray(pose, dtype=np.float64).reshape(36).copy()
        else:
            q = self._start_qpos(rng, jit)
        self.data.qpos[:] = q
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self._ctrl_stand
        clear_applied(self.data)
        mujoco.mj_forward(self.model, self.data)

        self._base_action = self._ctrl_stand.copy()
        self.push_schedule = push if push is not None else self._push_default
        if push is None and self._push_default is not None:
            self.push_schedule = self._push_default
        if command is not None:
            self._command_schedule = command
        elif self._command_sampler is not None:
            self._command_sampler.reset(self.seed)
            self._command_schedule = CommandSchedule.sampled(
                self._command_sampler, t0=0.5, t_end=self.horizon,
                hold_s=self.spec.command_hold_s)
        elif self._command_schedule is None:
            self._command_schedule = CommandSchedule.steady(DEFAULT_COMMAND)
        self.command_filter.reset(self._command_schedule.at(0.0))
        self.command = self.command_filter.value
        self.command_target = self._command_schedule.at(0.0)
        self._t_reset = float(self.data.time)

        self.fall_det.reset()
        self.dorsal_det.reset()
        self.recorder = MetricsRecorder(episode=self.episode, task=self.task,
                                        seed=self.seed)
        self._prev_ctrl = None
        self._prev_pelvis_z = float(self.data.qpos[2])
        self._prev_pelvis_xy = self._pelvis_xy().copy()
        self._prev_foot_xy = {sid: np.asarray(self.data.site_xpos[sid][:2],
                                              np.float64).copy()
                              for sid in self._foot_sites}
        self._prev_action = np.zeros(N_JOINTS)
        self._prev_contact = [False, False]
        self._foot_air = [0.0, 0.0]
        self._foot_landed = [False, False]
        self._foot_air_at_landing = [0.0, 0.0]
        self._step_events = 0
        self._shot_time = 0.0
        self._shot_phase = 0.0
        self._shot_armed = False
        self._shot_max_depth = 0.0
        self._shot_exited = False
        self._shot_depth_prev = 0.0
        self._shot_was_active = False
        self._marker_anchor = (self._pelvis_xy().copy(), self._yaw())
        self._low_since: float | None = (float(self.data.time)
                                         if float(self.data.qpos[2]) < RECOVERY_PELVIS_Z
                                         else None)
        self._recovered_hold = 0
        self._recovered_flag = False
        self._episode_done = False
        self._termination_cause: str | None = None
        self._last_ctx = None
        self._last_push_record: list[dict] = []
        self._last_contacts = ContactState()
        self.markers = self._update_markers()
        obs = self.observation()
        return obs

    def _start_qpos(self, rng: np.random.Generator, jitter: bool) -> np.ndarray:
        if self.spec.start == "crouch":
            q = stance_qpos(0.30, 0.62, self.model)
        else:
            q = self._q_stand.copy()
        if jitter:
            q[0] += rng.uniform(-RESET_XY_NOISE, RESET_XY_NOISE)
            q[1] += rng.uniform(-RESET_XY_NOISE, RESET_XY_NOISE)
            yaw = np.deg2rad(rng.uniform(-RESET_YAW_DEG, RESET_YAW_DEG))
            tilt = np.deg2rad(rng.uniform(-RESET_TILT_DEG, RESET_TILT_DEG))
            roll = np.deg2rad(rng.uniform(-RESET_TILT_DEG, RESET_TILT_DEG))
            q[3:7] = _quat_mul(_quat_yaw(yaw), _quat_pitch_roll(tilt, roll))
            noise = rng.uniform(-RESET_JOINT_NOISE, RESET_JOINT_NOISE, N_JOINTS)
            q[7:36] = np.clip(q[7:36] + noise, self.lo, self.hi)
        return q

    # ------------------------------------------------------------- action glue
    def ctrl_from_unit(self, unit: np.ndarray) -> np.ndarray:
        """Unit action in [-1, 1] -> ctrl targets (``mid + half * u``, clipped).

        This is the documented mapping used by scripted baselines so they take
        the identical path a policy's ``ActionMapper`` takes.
        """
        u = np.clip(np.asarray(unit, dtype=np.float64).reshape(N_JOINTS), -1.0, 1.0)
        mid = 0.5 * (self.lo + self.hi)
        half = 0.5 * (self.hi - self.lo)
        return np.clip(mid + half * u, self.lo, self.hi)

    def ctrl_from_policy(self, unit: np.ndarray) -> np.ndarray:
        """Unit action in [-1, 1] -> the ACTION to hand to :meth:`step`.

        Mode-aware (the only supported conversion for policies):

        * ``absolute``: ``mid + half * unit`` (clipped to ``ctrlrange``)
        * ``residual``: the unit itself -- ``step`` applies
          ``clip(base + residual_scale * tanh(unit))``, so ``unit = 0`` gives
          exactly ``base`` (the stand keyframe ctrl by default).  The residual
          ``z`` is therefore the policy's *already-tanh'd* unit, keeping the
          residual range at ``+/- residual_scale * tanh(1)``.
        """
        u = np.clip(np.asarray(unit, dtype=np.float64).reshape(N_JOINTS), -1.0, 1.0)
        if self.action_mode == "absolute":
            return self.ctrl_from_unit(u)
        return u

    def action_from_ctrl(self, ctrl: np.ndarray) -> np.ndarray:
        """Absolute ctrl targets -> the ACTION to hand to :meth:`step`.

        Inverse of :meth:`ctrl_from_policy`; scripted controllers that emit
        absolute joint targets use this so they behave identically in both
        action modes (residual mode inverts ``base + scale*tanh(z)``).
        """
        c = np.clip(np.asarray(ctrl, dtype=np.float64).reshape(N_JOINTS), self.lo, self.hi)
        if self.action_mode == "absolute":
            return c
        z = (c - self._base_action) / self.residual_scale
        return np.arctanh(np.clip(z, -1.0 + 1e-6, 1.0 - 1e-6))

    def set_push_schedule(self, schedule: PushSchedule | None) -> None:
        """Install the push schedule used from the *next* reset onwards."""
        self._push_default = schedule

    def set_base_action(self, ctrl: np.ndarray) -> None:
        """Base joint targets for residual mode (default: stand keyframe ctrl)."""
        b = np.asarray(ctrl, dtype=np.float64).reshape(N_JOINTS)
        self._base_action = np.clip(b, self.lo, self.hi)

    def resolve_action(self, action: np.ndarray) -> np.ndarray:
        """Apply the action-mode contract; returns clipped ctrl targets.

        With a ``joint_mask`` the frozen joints are written to their stand
        keyframe ctrl *after* the mode mapping, so the mask holds in both action
        modes and for scripted controllers that go through
        :meth:`action_from_ctrl` as well.
        """
        a = np.asarray(action, dtype=np.float64).reshape(N_JOINTS)
        if self.action_mode == "absolute":
            ctrl = np.clip(a, self.lo, self.hi)
        else:
            ctrl = np.clip(self._base_action + self.residual_scale * np.tanh(a),
                           self.lo, self.hi)
        if self.joint_mask is not None:
            ctrl = self.joint_mask.apply(ctrl)
        return ctrl

    # ------------------------------------------------------------------- step
    def step(self, action):
        """Advance one 50 Hz control step.

        Returns ``(obs, reward, terminated, truncated, info)`` with
        ``obs = {"actor", "privileged", "critic"}`` float32 arrays.
        """
        if self._episode_done:
            raise RuntimeError("step() after a terminal/truncated step; call reset()")
        t = float(self.data.time)
        ctrl = self.resolve_action(action)

        # 1. pushes: clear both force slots, then write the scheduled force
        clear_applied(self.data)
        heading = self._yaw()
        active = self.push_schedule.active(t) if self.push_schedule else []
        push_records = [apply_push(self.model, self.data, p, heading) for p in active]

        # 2. physics
        self.data.ctrl[:] = ctrl
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        # 3. command channel (schedule -> low-pass filter)
        self.command_target = self._command_schedule.at(t)
        self.command = self.command_filter.update(self.command_target, STEP_DT)

        # 4. shot bookkeeping, then markers (world-anchored for shot tasks)
        self._update_shot_state()
        self._update_marker_anchor()
        self.markers = self._update_markers()

        # 5. detectors
        f = fall_features(self.model, self.data)
        t_now = float(self.data.time)
        fall_trigger = self.fall_det.update(f, t_now)
        dorsal_trigger = self.dorsal_det.update(self.model, self.data, t_now)
        dorsal_f = self.dorsal_det.last_features

        # 6. contacts / metrics / reward
        contacts = f.contacts
        self._update_recovery(f)
        inp, ctx = self._reward_inputs(ctrl, f, dorsal_f)
        reward, terms = self.reward.step(inp)
        row = self._record_metrics(f, ctrl, inp, reward, terms) if self.record_metrics else None

        # 7. termination
        #
        # Cause priority: the dorsal (back-to-mat) verdict is the *failed attempt*
        # of the game rules and outranks the plain balance fall.  When the fall
        # rule trips while the dorsal condition is being continuously satisfied,
        # the verdict is deferred (up to the dorsal confirmation window) so the
        # episode is reported as `dorsal`; a lapse in the dorsal condition makes
        # the fall verdict fire immediately.
        cause: str | None = None
        if dorsal_trigger and self.terminate_on_dorsal:
            cause = "dorsal"
        elif fall_trigger and self.terminate_on_fall:
            if self.terminate_on_dorsal and self.dorsal_det.condition_active \
                    and self.dorsal_det.trigger_t is None:
                cause = None  # wait for the dorsal persistence window
            else:
                cause = "fall"
        truncated = False
        if cause is not None:
            pen, tterms = self.reward.terminal(cause)
            reward += pen
            terms.update(tterms)
            if row is not None:
                row["reward"] = round(float(reward), 6)
                row["terms"] = {k: round(float(v), 6) for k, v in terms.items()}
            self._episode_done = True
        elif t_now - self._t_reset >= self.horizon - 1e-9:
            truncated = True
            self._episode_done = True

        self._prev_ctrl = ctrl
        self._prev_pelvis_z = f.pelvis_z
        a = np.asarray(action, dtype=np.float64).reshape(N_JOINTS)
        self._prev_action = (self.unit_from_ctrl(ctrl) if self.action_mode == "absolute"
                             else np.tanh(a))
        self._last_contacts = contacts

        info = {
            "task": self.task,
            "t": t_now,
            "episode": int(self.episode),
            "horizon": self.horizon,
            "seed": int(self.seed),
            "command": self.command.as_dict(),
            "command_target": self.command_target.as_dict(),
            "push_applied": push_records,
            "fall": self.fall_det.status(),
            "dorsal": self.dorsal_det.status(),
            "contacts": contacts.as_dict(),
            "markers": {k: [round(float(x), 4) for x in v]
                        for k, v in self.markers.items()},
            "shot_phase": round(self._shot_phase, 4),
            "termination": cause,
            "reward_terms": {k: round(float(v), 6) for k, v in terms.items()},
            "metrics": row,
            "config_task": self.task,
        }
        if cause is not None or truncated:
            info["episode_end"] = {
                "termination": cause, "truncated": truncated,
                "steps": len(self.recorder.rows),
                "episode": int(self.episode),
            }
            self.episode += 1
        return self.observation(), float(reward), bool(cause is not None), bool(truncated), info

    # ------------------------------------------------------------ observation
    def observation(self) -> dict:
        """{"actor", "privileged", "critic"} float32 arrays from the last state."""
        ctx = self._last_ctx if self._last_ctx is not None else self._make_ctx(None)
        return {"actor": actor_obs(ctx), "privileged": privileged_obs(ctx),
                "critic": critic_obs(ctx)}

    def actor_vector(self) -> np.ndarray:
        return self.observation()["actor"]

    def privileged_vector(self) -> np.ndarray:
        return self.observation()["privileged"]

    def critic_vector(self) -> np.ndarray:
        return self.observation()["critic"]

    # -------------------------------------------------------------- internals
    def _yaw(self) -> float:
        R = self.data.xmat[self._pelvis_bid].reshape(3, 3)
        return float(math.atan2(R[1, 0], R[0, 0]))

    def _pelvis_xy(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self._pelvis_bid][:2], dtype=np.float64)

    def unit_from_ctrl(self, ctrl: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`ctrl_from_unit` (the stored ``prev_action``)."""
        mid = 0.5 * (self.lo + self.hi)
        half = 0.5 * (self.hi - self.lo)
        return np.clip((np.asarray(ctrl, np.float64).reshape(N_JOINTS) - mid) / half,
                       -1.0, 1.0)

    def _update_marker_anchor(self) -> None:
        """Follow-mode markers track the robot; shot-mode markers stay fixed.

        For ``plan.follow`` the anchor is the current pose every step.  For
        ``follow=False`` (shot tasks) the anchor is set when the shot skill
        becomes active, so the entry region is a fixed world target and
        penetration depth is physical progress (never clock progress).
        """
        in_shot = self.command.skill is Skill.SHOT_DOUBLE_LEG
        if self.marker_plan.follow:
            self._marker_anchor = (self._pelvis_xy().copy(), self._yaw())
        elif in_shot and not self._shot_was_active:
            self._marker_anchor = (self._pelvis_xy().copy(), self._yaw())
        self._shot_was_active = in_shot

    def _update_markers(self) -> dict:
        xy, yaw = self._marker_anchor
        return markers_mod.apply_markers(self.model, self.data, xy, yaw,
                                         self.marker_plan)

    def _update_shot_state(self) -> None:
        in_shot = self.command.skill is Skill.SHOT_DOUBLE_LEG
        if in_shot:
            self._shot_time += STEP_DT
            self._shot_phase = min(1.0, self._shot_phase + SHOT_PHASE_RATE * STEP_DT)
        else:
            self._shot_time = 0.0
            self._shot_phase = max(0.0, self._shot_phase - 2.0 * SHOT_PHASE_RATE * STEP_DT)
        depth = self._shot_depth()
        self._shot_armed = self._shot_armed or depth >= SHOT_ENTER_DEPTH
        self._shot_max_depth = max(self._shot_max_depth, depth)
        self._shot_exited = bool(
            self._shot_max_depth >= SHOT_ENTER_DEPTH
            and depth <= SHOT_EXIT_DEPTH
            and float(self.data.qpos[2]) > 0.60)

    def _shot_depth(self) -> float:
        return markers_mod.penetration_depth(
            self._pelvis_xy(),
            np.asarray(self.markers["a_marker_pelvis"][:2], dtype=np.float64),
            self.marker_plan)

    def _update_recovery(self, f) -> None:
        pz = f.pelvis_z
        if pz < RECOVERY_PELVIS_Z:
            self._low_since = self._low_since if self._low_since is not None \
                else float(self.data.time)
            self._recovered_hold = 0
            self._recovered_flag = False
            return
        if self._low_since is None:
            self._recovered_flag = False
            return
        ok = (f.torso_up_z >= 0.97
              and abs(pz - STAND_HEIGHT) <= RECOVERY_TOL)
        self._recovered_hold = self._recovered_hold + 1 if ok else 0
        if self._recovered_hold * STEP_DT >= RECOVERY_HOLD_S:
            self._recovered_flag = True
            self._low_since = None

    def _foot_state(self, contacts: ContactState) -> tuple[list[bool], list[float]]:
        """Update per-foot air time; returns (landed, air_time_at_landing)."""
        cur = [bool(contacts.left_foot), bool(contacts.right_foot)]
        landed = [False, False]
        air_at_landing = [0.0, 0.0]
        for i in range(2):
            if cur[i]:
                if not self._prev_contact[i]:
                    landed[i] = True
                    air_at_landing[i] = self._foot_air[i]
                    if self._foot_air[i] >= STEP_EVENT_MIN_AIR:
                        self._step_events += 1  # a real step, not contact chatter
                self._foot_air[i] = 0.0
            else:
                self._foot_air[i] += STEP_DT
        self._prev_contact = cur
        self._foot_landed = landed
        self._foot_air_at_landing = air_at_landing
        return landed, air_at_landing

    def _per_foot_slip(self, contacts: ContactState) -> tuple[float, float]:
        vel = np.zeros(6)
        out = []
        for sid, ok in zip(self._foot_sites, (contacts.left_foot, contacts.right_foot)):
            if not ok:
                out.append(0.0)
                continue
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                     int(sid), vel, 0)
            out.append(float(math.hypot(vel[3], vel[4])))
        return out[0], out[1]

    def _reward_inputs(self, ctrl, f, dorsal_f):
        ctx = self._make_ctx(f, dorsal_f)
        contacts = f.contacts
        landed, air_at_landing = self._foot_state(contacts)
        v_world = np.asarray(self.data.qvel[0:3], dtype=np.float64)
        yaw = self._yaw()
        vel_local = local_xy(v_world, yaw)
        w_world = np.asarray(self.data.qvel[3:6], dtype=np.float64)
        wz_local = float((-math.sin(yaw) * w_world[0] + math.cos(yaw) * w_world[1]))
        slip_l, slip_r = self._per_foot_slip(contacts)
        sat = saturation_fraction(self.model, self.data)
        prox, _ = joint_limit_proximity(self.model, self.data)
        lead = "left" if int(self.command.lead_leg) == -1 else "right"
        hand_world = np.asarray(self.data.site_xpos[self._wrist_sites[lead]], np.float64)
        target_name = "a_marker_hand_l" if lead == "left" else "a_marker_hand_r"
        hand_dist = float(np.linalg.norm(hand_world - self.markers[target_name]))
        self._last_hand_dist = hand_dist
        depth = self._shot_depth()
        lead_ahead = self._lead_foot_ahead()
        knee_ok = (contacts.knees and f.pelvis_z >= 0.25
                   and not (dorsal_f.dorsal_contact if dorsal_f is not None else False)
                   and self._shot_phase > 0.0)
        # literature (source A/B) inputs: world-frame CoM position and velocity,
        # the sole-sphere points of the support hull, per-foot GRF and the
        # actuator torques.  Computed unconditionally (a few microseconds) so a
        # term set can be selected per run without an env flag.
        com_world = np.asarray(self.data.subtree_com[self._pelvis_bid], np.float64)
        com_vel_world = ((self._body_mass[:, None]
                          * np.asarray(self.data.cvel, np.float64)[:, 3:6])
                         .sum(axis=0) / self._mass_total)
        loads = tuple(float(self.data.cfrc_ext[bid][5]) for bid in self._foot_bids)
        joint_pos = np.asarray(self.data.qpos[self._joint_q], np.float64)
        arm_dev = float(np.abs(joint_pos[self._arm_idx]
                               - self._q_stand[7:36][self._arm_idx]).sum())
        inp = RewardInputs(
            dt=STEP_DT, cmd=self.command,
            vel_local=vel_local, yaw_rate=wz_local,
            torso_up_z=f.torso_up_z, pelvis_z=f.pelvis_z,
            pelvis_z_prev=self._prev_pelvis_z, stand_height=STAND_HEIGHT,
            stance_width_meas=self._stance_width_meas(),
            foot_contact=(contacts.left_foot, contacts.right_foot),
            foot_slip=(slip_l, slip_r),
            foot_air_time=(air_at_landing[0], air_at_landing[1]),
            foot_landed=(landed[0], landed[1]),
            action=np.asarray(ctrl, np.float64), prev_action=self._prev_ctrl,
            sat_frac=sat, limit_prox=prox,
            hand_distance=hand_dist,
            shot_depth=depth, shot_depth_prev=self._shot_depth_prev,
            shot_phase=self._shot_phase, shot_time=self._shot_time,
            shot_budget_s=self.spec.shot_budget_s,
            shot_leg_ahead=lead_ahead, shot_knee_control=knee_ok,
            shot_exited=self._shot_exited,
            recovered=self._recovered_flag,
            dorsal=bool(dorsal_f.dorsal_contact) if dorsal_f is not None else False,
            com_xy=com_world[:2].copy(), com_vel_xy=com_vel_world[:2].copy(),
            com_z=float(com_world[2]),
            sole_points=sole_points_world(self.model, self.data, self._sole_geoms),
            foot_load=(loads[0], loads[1]),
            torque=np.asarray(self.data.actuator_force, np.float64).copy(),
            arm_dev=arm_dev,
            gravity=float(-self.model.opt.gravity[2]),
        )
        self._shot_depth_prev = depth
        self._last_ctx = ctx
        return inp, ctx

    def _stance_width_meas(self) -> float:
        lf = np.asarray(self.data.site_xpos[self._foot_sites[0]], np.float64)
        rf = np.asarray(self.data.site_xpos[self._foot_sites[1]], np.float64)
        return float(abs(lf[1] - rf[1]))

    def _lead_foot_ahead(self) -> bool:
        """Lead foot (heading frame) ahead of the trail foot by > 2 cm."""
        yaw = self._yaw()
        c, s = math.cos(yaw), math.sin(yaw)
        out = []
        for sid in self._foot_sites:
            p = np.asarray(self.data.site_xpos[sid], np.float64)
            out.append(c * p[0] + s * p[1])
        lead_idx = 0 if int(self.command.lead_leg) == -1 else 1
        trail_idx = 1 - lead_idx
        return bool(out[lead_idx] - out[trail_idx] > 0.02)

    def _make_ctx(self, f, dorsal_f=None):
        contacts = f.contacts if f is not None else self._last_contacts
        R = self.data.xmat[self._pelvis_bid].reshape(3, 3)
        v_world = np.asarray(self.data.qvel[0:3], dtype=np.float64)
        gravity_local = R.T @ np.array([0.0, 0.0, -1.0])
        pelvis_xy = self._pelvis_xy()
        # CoM (whole robot) in the pelvis frame
        com_world = np.asarray(self.data.subtree_com[1], dtype=np.float64)
        com_rel = R.T @ (com_world - np.asarray(self.data.xpos[self._pelvis_bid]))
        # support centre: mean of loaded foot sites (explicitly not a hull margin)
        loaded = [sid for sid, ok in
                  zip(self._foot_sites, (contacts.left_foot, contacts.right_foot)) if ok]
        if loaded:
            sc_world = np.mean([np.asarray(self.data.site_xpos[sid], np.float64)
                                for sid in loaded], axis=0)
            support_local = R.T @ (sc_world - np.asarray(self.data.xpos[self._pelvis_bid]))
            com_to_support = np.array([com_rel[0] - support_local[0],
                                       com_rel[1] - support_local[1]])
        else:
            support_local = np.zeros(3)
            com_to_support = np.zeros(2)
        marker_local = {name: R.T @ (np.asarray(pos, np.float64)
                                     - np.asarray(self.data.xpos[self._pelvis_bid]))
                        for name, pos in self.markers.items()}
        push = None
        if self.push_schedule is not None:
            nxt = self.push_schedule.next_after(float(self.data.time))
            if nxt is not None:
                push = (max(0.0, nxt.t - float(self.data.time)), nxt.direction,
                        nxt.impulse)
        tilt = float(np.degrees(np.arccos(np.clip(
            self.data.xmat[self._torso_bid].reshape(3, 3)[2, 2], -1.0, 1.0))))
        slip = foot_slip(self.model, self.data, self._foot_sites,
                         (contacts.left_foot, contacts.right_foot))
        return ObsContext(
            base_linvel_local=R.T @ v_world,
            base_angvel_local=R.T @ np.asarray(self.data.qvel[3:6], dtype=np.float64),
            gravity_local=gravity_local,
            joint_pos_rel=np.asarray(self.data.qpos[self._joint_q],
                                     np.float64) - self._q_stand[7:36],
            joint_vel=np.asarray(self.data.qvel[self._joint_v], np.float64),
            prev_action=self._prev_action,
            cmd=self.command,
            phase=self._shot_phase if self.command.skill is Skill.SHOT_DOUBLE_LEG else 0.0,
            contacts=contacts,
            dorsal_contact=bool(dorsal_f.dorsal_contact) if dorsal_f is not None else False,
            tilt_deg=tilt, pelvis_z=float(self.data.qpos[2]),
            stand_height=STAND_HEIGHT,
            com_rel_local=com_rel,
            com_vel_local=R.T @ np.asarray(self.data.qvel[0:3], dtype=np.float64),
            support_center_local=support_local,
            com_to_support_xy=com_to_support,
            marker_local=marker_local,
            next_push=push,
            foot_slip=slip,
            sat_frac=saturation_fraction(self.model, self.data),
            limit_prox=joint_limit_proximity(self.model, self.data)[0],
            shot_phase=self._shot_phase,
            marker_distance=float(np.linalg.norm(
                np.asarray(self.markers["a_marker_pelvis"][:2], np.float64) - pelvis_xy)),
        )

    def _record_metrics(self, f, ctrl, inp, reward, terms) -> dict:
        contacts = f.contacts
        ctx = self._last_ctx
        com_offset = (float(np.linalg.norm(np.asarray(ctx.com_to_support_xy)))
                      if ctx is not None else None)
        vel_local = np.asarray(inp.vel_local, np.float64)
        # T2 (locomotion) measurement support: per-axis tracking error, the
        # pelvis path step, and the *position* travel of each loaded foot (the
        # point-sampled ``slip`` above cannot see a slow drag: MuJoCo's friction
        # constraint zeroes the relative velocity within a substep while the
        # positions still integrate).
        pelvis_xy = self._pelvis_xy()
        body_step = None
        if self._prev_pelvis_xy is not None:
            body_step = float(np.linalg.norm(pelvis_xy - self._prev_pelvis_xy))
        self._prev_pelvis_xy = pelvis_xy.copy()
        slip_travel = 0.0
        for sid, loaded in zip(self._foot_sites,
                               (contacts.left_foot, contacts.right_foot)):
            xy = np.asarray(self.data.site_xpos[sid][:2], np.float64)
            prev_xy = self._prev_foot_xy.get(sid)
            if loaded and prev_xy is not None:
                slip_travel += float(np.linalg.norm(xy - prev_xy))
            self._prev_foot_xy[sid] = xy.copy()
        row = self.recorder.add(
            t=float(self.data.time),
            vel_err=float(np.linalg.norm(vel_local
                                         - np.array([self.command.vx, self.command.vy]))),
            vx_err=float(vel_local[0] - self.command.vx),
            vy_err=float(vel_local[1] - self.command.vy),
            yaw_err=abs(inp.yaw_rate - self.command.wz),
            upright=f.torso_up_z, tilt_deg=f.tilt_deg, pelvis_z=f.pelvis_z,
            stance_err=f.pelvis_z - self.command.stance_height,
            slip=float(np.max(inp.foot_slip)),
            slip_travel=slip_travel,
            body_step=body_step,
            cmd_vx=self.command.vx, cmd_vy=self.command.vy, cmd_wz=self.command.wz,
            speed=float(np.linalg.norm(np.asarray(inp.vel_local, np.float64))),
            com_offset=com_offset,
            steps_taken=int(self._step_events),
            contact_l=contacts.left_foot, contact_r=contacts.right_foot,
            knee_contact=contacts.knees, hand_contact=contacts.hands,
            torso_contact=contacts.torso or contacts.pelvis,
            dorsal_contact=bool(self.dorsal_det.last_features.dorsal_contact)
            if self.dorsal_det.last_features is not None else False,
            act_delta=action_smoothness(self._prev_ctrl, ctrl),
            sat_frac=inp.sat_frac, limit_prox=inp.limit_prox,
            hand_err=getattr(self, "_last_hand_dist", None),
            reward=reward, terms=terms,
        )
        return row


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_yaw(a: float) -> np.ndarray:
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def _quat_pitch_roll(pitch: float, roll: float) -> np.ndarray:
    return _quat_mul(np.array([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]),
                     np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]))


if __name__ == "__main__":  # self-check
    env = SoloEnv(seed=3)
    obs = env.reset(seed=3)
    assert obs["actor"].shape == (ACTOR_DIM,) and obs["critic"].shape == (CRITIC_DIM,)
    hold = env._ctrl_stand.copy()
    r = 0.0
    for k in range(100):
        obs, rew, term, trunc, info = env.step(hold)
        r += rew
        assert not term and not trunc, info["termination"]
    assert len(env.recorder.rows) == 100
    print("solo.env self-check OK:", {
        "task": env.task, "steps": len(env.recorder.rows),
        "pelvis_z": round(info["metrics"]["pelvis_z"], 4),
        "upright": round(info["metrics"]["upright"], 4),
        "hold_return": round(r, 3),
        "actor_dim": int(obs["actor"].size),
        "critic_dim": int(obs["critic"].size)})
