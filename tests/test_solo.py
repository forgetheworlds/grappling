"""Tests for the S1 solo-drill harness (``src/solo``).

Coverage (SOLO_DRILL §3 S1 + the brief's deliverable 10):

* scene/prefix contract (exact dependency names, nq/nv/nu/timestep)
* observation shape/content invariants + actor/privileged separation
* action clipping (absolute and residual) and the unit<->ctrl round trip
* command sampler ranges, low-pass filter, schedules changing within an episode
* pushes measurably perturb the robot; force slots are exactly zero when idle;
  the applied force is not silently clamped (gravity-off momentum check)
* fall detector: persistence semantics, scripted falls fire, normal stance and
  scripted stepping do not, knees/hands postures are never terminal, dorsal
  back-to-mat fires only after its own persistence window
* reward: every term hand-tested on constructed states, all degenerate cases
  score worse than a genuine attempt, collapsing loses to standing over a
  fixed horizon
* env determinism, reset validity, step/info contract, metric emission
* virtual-opponent markers are physical (world-anchored depth), phase selects
  the active target
* eval harness: criterion/gate logic, battery determinism, one real gated run
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco  # noqa: E402

from solo.baselines import (FallForwardController, RandomInitPolicyController,  # noqa: E402
                            SquatController, StandHoldController,
                            ZeroActionController)
from solo.commands import (Command, CommandFilter, CommandRanges,  # noqa: E402
                           CommandSampler, CommandSchedule, Skill, N_SKILLS)
from solo.env import TASKS, SoloEnv  # noqa: E402
from solo.eval import (GATES, Criterion, TaskGate, battery_pushes,  # noqa: E402
                       evaluate, run_episode, take_clips)
from solo.fall import (ContactState, DorsalDetector, FallDetConfig,  # noqa: E402
                       FallDetector, FallFeatures, contact_state, fall_features)
from solo.markers import (MarkerPlan, SHOT_PLAN, active_target,  # noqa: E402
                          apply_markers, penetration_depth, targets)
from solo.metrics import summarize_rows  # noqa: E402
from solo.obs import (ACTOR_DIM, ACTOR_LAYOUT, CRITIC_DIM, PRIV_DIM,  # noqa: E402
                      ObsContext, actor_obs, critic_obs, privileged_obs)
from solo.pushes import PushSchedule, PushSpec, apply_push, clear_applied  # noqa: E402
from solo.reward import (PENALTY_TERMS, TASK_TERMS, TERM_FUNCS, RewardInputs,  # noqa: E402
                         TaskReward, discounted_return)
from solo.scene import (FLOOR_GEOM, HAND_BODIES, KNEE_BODIES, MARKER_NAMES,  # noqa: E402
                        PELVIS_BODY, STEP_DT, SUBSTEPS, TORSO_BODY, body_id,
                        ctrl_range, load_solo_model, model_facts,
                        resolved_dependencies, site_id, stand_frame)
from solo.stance import stance_qpos, stance_targets  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return load_solo_model()


def _fresh(model, qpos, ctrl):
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    d.ctrl[:] = ctrl
    mujoco.mj_forward(model, d)
    return d


def _settle(model, qpos, ctrl, seconds):
    d = _fresh(model, qpos, ctrl)
    for _ in range(int(round(seconds / 0.002))):
        mujoco.mj_step(model, d)
    return d


def _kneel_qpos(model):
    q, c = stand_frame(model)
    q = q.copy()
    q[2] = 0.45
    for off in (0, 6):
        q[7 + off + 0] = 0.15
        q[7 + off + 3] = 1.65
        q[7 + off + 4] = 0.45
    return q


def _hands_qpos(model):
    """Hands-plant pose measured to settle onto the arms (pelvis 0.118 m)."""
    q, c = stand_frame(model)
    q = q.copy()
    q[2] = 0.42
    q[3:7] = np.array([np.cos(np.deg2rad(35)), 0.0, np.sin(np.deg2rad(35)), 0.0])
    for off in (0, 6):
        q[7 + off + 0] = 1.0
        q[7 + off + 3] = 1.4
        q[7 + off + 4] = 0.0
    for off, sgn in ((15, 1), (22, -1)):
        q[7 + off + 0] = -1.1
        q[7 + off + 1] = sgn * 0.3
        q[7 + off + 3] = 0.3
    return q


def _supine_run(model, seconds=2.0):
    """Drive the robot onto its back with a chest-height push; returns env."""
    env = SoloEnv(seed=5, task="balance")
    env.horizon = 30.0
    push = PushSchedule([PushSpec(t=0.5, impulse=16.0, direction=np.pi, height=0.95)])
    env.reset(seed=5, push=push, jitter=False)
    hold = env._ctrl_stand.copy()
    term = None
    for _ in range(int(seconds / STEP_DT)):
        obs, r, done, trunc, info = env.step(hold)
        if done:
            term = info["termination"]
            break
    return env, term, info


# ------------------------------------------------------------------ scene
def test_model_contract(model):
    facts = model_facts(model)
    assert (facts["nq"], facts["nv"], facts["nu"]) == (36, 35, 29)
    assert facts["timestep"] == pytest.approx(0.002, abs=1e-12)
    assert facts["substeps"] == SUBSTEPS == 10
    assert facts["control_dt"] == pytest.approx(0.02)
    assert facts["keyframes"] == ["a_stand"]
    assert facts["total_mass"] == pytest.approx(33.341, abs=0.002)
    lo, hi = ctrl_range(model)
    assert lo.shape == hi.shape == (29,) and np.all(hi > lo)
    # residual/absolute contract constants
    q, c = stand_frame(model)
    assert q.shape == (36,) and c.shape == (29,)
    assert float(q[2]) == pytest.approx(0.79)


def test_scene_dependency_names(model):
    deps = resolved_dependencies(model)
    for name in (PELVIS_BODY, TORSO_BODY, "a_torso_link", "a_left_ankle_roll_link",
                 "virtual_opponent"):
        assert name in deps["bodies"]
    for name in MARKER_NAMES:
        assert name in deps["sites"]
    for body in KNEE_BODIES + HAND_BODIES:
        assert body in deps["bodies"]
    assert deps["geoms"][FLOOR_GEOM] >= 0
    # the prefixed names the shared helpers rely on resolve on the *solo* model
    from wrestling.backdet import body_maps

    maps = body_maps(model, "a")
    assert maps.torso_bid == deps["bodies"][TORSO_BODY]
    assert maps.pelvis_bid == deps["bodies"][PELVIS_BODY]
    assert len(maps.limb_bids) >= 6


# ------------------------------------------------------------ observations
def test_obs_shapes_and_content(model):
    ctx = ObsContext()
    a, p, c = actor_obs(ctx), privileged_obs(ctx), critic_obs(ctx)
    assert a.shape == (ACTOR_DIM,) == (115,)
    assert p.shape == (PRIV_DIM,) == (43,)
    assert c.shape == (CRITIC_DIM,) == (158,)
    assert np.array_equal(c, np.concatenate([a, p]))
    layout = dict(ACTOR_LAYOUT)
    s0, s1 = layout["skill_onehot"]
    assert a[s0:s1].sum() == 1.0
    assert int(np.argmax(a[s0:s1])) == int(Skill.STANCE)
    s0, s1 = layout["gravity_local"]
    assert np.allclose(a[s0:s1], [0, 0, -1], atol=1e-6)
    ctx2 = ObsContext(cmd=Command(vx=0.25, vy=-0.1, wz=0.3,
                                  stance_height=0.72, stance_width=0.35,
                                  skill_id=int(Skill.CIRCLE_L), lead_leg=-1))
    a2 = actor_obs(ctx2)
    for name in ("cmd_vel", "cmd_stance"):
        s0, s1 = layout[name]
        v = {"cmd_vel": [0.25, -0.1, 0.3], "cmd_stance": [0.72, 0.35]}[name]
        assert np.allclose(a2[s0:s1], v, atol=1e-6)
    s0, s1 = layout["lead_leg"]
    assert a2[s0:s1][0] == pytest.approx(-1.0)
    assert a2.size == ACTOR_DIM and a2.dtype == np.float32
    assert a2.sum() == a2.sum()  # no NaN
    assert not np.isnan(a2).any() and not np.isnan(p).any()


def test_actor_obs_never_sees_privileged():
    base = ObsContext()
    enriched = ObsContext(
        contacts=ContactState(left_foot=True, right_knee=True, left_hand=True,
                              torso=True, n_contacts=3),
        dorsal_contact=True, tilt_deg=80.0, pelvis_z=0.2,
        com_rel_local=np.array([0.1, 0.0, 0.05]),
        marker_local={n: np.full(3, 0.5) for n in MARKER_NAMES},
        next_push=(1.0, 0.5, 8.0), foot_slip=1.2, sat_frac=0.5, limit_prox=0.9,
        shot_phase=0.7, marker_distance=0.4)
    assert np.array_equal(actor_obs(base), actor_obs(enriched))
    pa, pb = privileged_obs(base), privileged_obs(enriched)
    assert not np.array_equal(pa, pb)


# --------------------------------------------------------------- actions
def test_action_clipping(model):
    env = SoloEnv(seed=0)
    lo, hi = env.lo, env.hi
    wild = np.linspace(-100.0, 100.0, 29)
    ctrl = env.resolve_action(wild)
    assert np.all(ctrl >= lo - 1e-12) and np.all(ctrl <= hi + 1e-12)
    assert np.allclose(ctrl, np.clip(wild, lo, hi))
    # residual mode: ctrl = clip(base + scale * tanh(z))
    env_r = SoloEnv(seed=0, action_mode="residual", residual_scale=0.4)
    base = env_r._ctrl_stand.copy()
    z0 = env_r.resolve_action(np.zeros(29))
    assert np.allclose(z0, np.clip(base, lo, hi))
    z_big = env_r.resolve_action(np.full(29, 50.0))
    assert np.all(z_big <= hi + 1e-12) and np.all(z_big >= lo - 1e-12)
    assert np.allclose(z_big, np.clip(base + 0.4, lo, hi), atol=1e-9)


def test_unit_ctrl_roundtrip(model):
    env = SoloEnv(seed=0)
    assert np.allclose(env.ctrl_from_unit(np.zeros(29)), 0.5 * (env.lo + env.hi))
    u = np.linspace(-1.0, 1.0, 29)
    ctrl = env.ctrl_from_unit(u)
    assert np.allclose(env.unit_from_ctrl(ctrl), u, atol=1e-12)


# --------------------------------------------------------------- commands
def test_command_sampler_ranges():
    ranges = CommandRanges()
    for skills in (None,
                   (Skill.STANCE, Skill.SHUFFLE_F, Skill.SHUFFLE_B, Skill.SHUFFLE_L,
                    Skill.SHUFFLE_R, Skill.CIRCLE_L, Skill.CIRCLE_R, Skill.RETREAT,
                    Skill.APPROACH, Skill.LEVEL_CHANGE, Skill.SHOT_DOUBLE_LEG,
                    Skill.RECOVER)):
        s = CommandSampler(ranges, seed=1, skills=skills)
        seen_stationary = False
        for _ in range(300):
            c = s.sample()
            assert ranges.contains(c), (c.as_dict(), ranges.as_dict())
            assert 0 <= c.skill_id < N_SKILLS
            assert c.lead_leg in (-1, 1)
            seen_stationary = seen_stationary or (c.vx == 0.0 and c.vy == 0.0)
        assert seen_stationary
    # determinism of a seeded sampler
    a = [CommandSampler(ranges, seed=7).sample() for _ in range(3)]
    b = [CommandSampler(ranges, seed=7).sample() for _ in range(3)]
    assert [c.as_dict() for c in a] == [c.as_dict() for c in b]


def test_command_filter_and_schedule():
    f = CommandFilter(tau=0.25)
    f.reset(Command())
    prev = -1.0
    for _ in range(25):
        v = f.update(Command(vx=0.5), dt=STEP_DT)
        assert v.vx >= prev - 1e-12
        prev = v.vx
    assert 0.4 < v.vx < 0.5
    assert abs(v.wz - 0.0) < 1e-12
    f.reset(Command(skill_id=int(Skill.STANCE)))
    v = f.update(Command(vx=0.5, skill_id=int(Skill.APPROACH), lead_leg=-1), dt=STEP_DT)
    assert v.skill_id == int(Skill.APPROACH) and v.lead_leg == -1  # discrete snaps
    sched = CommandSchedule.sampled(CommandSampler(seed=3), t0=0.5, t_end=8.0,
                                    hold_s=1.5)
    assert len(sched.segments) >= 5
    assert sched.at(0.0).skill is Skill.STANCE
    cmds = {s.command.as_dict()["skill"] for s in sched.segments}
    assert len(cmds) >= 2, "schedule must change commands within the episode"
    assert sched.next_change(0.5) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        CommandSchedule.sampled(CommandSampler(seed=0), t_end=1e9, hold_s=1.5)


# ------------------------------------------------------------------ pushes
def test_push_measurably_perturbs(model):
    push = PushSpec(t=0.5, impulse=8.0, direction=0.0, height=0.95)
    env_a = SoloEnv(seed=1, task="balance")
    env_b = SoloEnv(seed=1, task="balance")
    env_a.horizon = env_b.horizon = 1e9
    env_a.reset(seed=1, push=PushSchedule([]), jitter=False)
    env_b.reset(seed=1, push=PushSchedule([push]), jitter=False)
    hold = env_a._ctrl_stand.copy()
    window_end = push.t + push.n_steps * STEP_DT
    v_a = v_b = None
    force_seen = 0.0
    for k in range(60):  # 1.2 s
        oa, ra, ta, tra, ia = env_a.step(hold)
        ob, rb, tb, trb, ib = env_b.step(hold)
        for rec in ib["push_applied"]:
            force_seen = max(force_seen, float(rec["qfrc_norm"]))
        assert ib["push_applied"] == [] or ia["t"] >= push.t - 1e-9
        if abs(ia["t"] - window_end) < 1e-9:
            v_a = env_a.data.qvel[0:3].copy()
            v_b = env_b.data.qvel[0:3].copy()
    assert v_a is not None and v_b is not None
    dv = float(np.linalg.norm(v_b - v_a))
    assert dv > 0.10, f"push did not measurably perturb the base (dv={dv})"
    assert force_seen > 0.0, "no generalized force was applied during the window"
    # (b) force slots are EXACTLY zero once the window is over
    assert not env_b.data.qfrc_applied.any()
    assert not env_b.data.xfrc_applied.any()
    assert not env_a.data.qfrc_applied.any()
    # (c) no residual drift attributable to a stale force: base acceleration over
    # the second half of the run is small and comparable in both envs
    acc = []
    for env in (env_a, env_b):
        v1 = env.data.qvel[0:3].copy()
        for _ in range(25):
            env.step(hold)
        acc.append((env.data.qvel[0:3] - v1) / (25 * STEP_DT))
    assert np.linalg.norm(acc[1] - acc[0]) < 2.0, acc


def test_push_force_not_clamped(model):
    """Gravity-off in-air impulse: momentum must equal J exactly (no clamping)."""
    d = mujoco.MjData(model)
    q, c = stand_frame(model)
    d.qpos[:] = q
    d.ctrl[:] = c
    grav = model.opt.gravity.copy()
    try:
        model.opt.gravity[:] = 0.0
        d.qpos[2] = 2.0
        mujoco.mj_forward(model, d)
        spec = PushSpec(t=0.0, impulse=5.0, direction=0.0, height=2.4)
        clear_applied(d)
        rec = apply_push(model, d, spec)
        assert rec["qfrc_norm"] > 0
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, d)
        mmat = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, d, mmat)
        p = mmat @ np.asarray(d.qvel, dtype=np.float64)
        assert p[0] == pytest.approx(spec.realized_impulse, rel=0.01)
    finally:
        model.opt.gravity[:] = grav


def test_clear_applied_zeroes_slots():
    d = mujoco.MjData(load_solo_model())
    d.xfrc_applied[1, 0] = 7.0
    d.qfrc_applied[0] = 3.0
    clear_applied(d)
    assert not d.qfrc_applied.any() and not d.xfrc_applied.any()


def test_push_schedule_battery_deterministic():
    a = battery_pushes(seed=3)
    b = battery_pushes(seed=3)
    assert [p.as_dict() for p in a] == [p.as_dict() for p in b]
    mags = sorted({round(p.impulse, 6) for p in a})
    assert len(mags) >= 5 and max(mags) == 12.0  # the unseen point of the gate
    assert all(0.0 < p.duration <= 0.1 for p in a)


# -------------------------------------------------------------- fall rules
def test_fall_persistence_and_reset():
    det = FallDetector(FallDetConfig(confirm_s=0.25))
    fallen = FallFeatures(t=0.0, pelvis_z=0.15, tilt_deg=80.0, torso_up_z=-0.1,
                          contacts=ContactState())
    for i in range(11):  # 0.22 s of condition -> no trigger yet
        assert not det.update(FallFeatures(t=i * STEP_DT, pelvis_z=0.15,
                                           tilt_deg=80.0, torso_up_z=-0.1,
                                           contacts=ContactState()))
    assert det.trigger_t is None
    tripped = det.update(FallFeatures(t=0.26, pelvis_z=0.15, tilt_deg=80.0,
                                      torso_up_z=-0.1, contacts=ContactState()))
    assert tripped and det.trigger_t == pytest.approx(0.26)
    assert not det.update(FallFeatures(t=0.4, pelvis_z=0.15, tilt_deg=80.0,
                                       torso_up_z=-0.1, contacts=ContactState()))
    det.reset()
    assert det.trigger_t is None


def test_fall_detector_exemptions(model):
    det = FallDetector()
    # knees-only support at a normal kneel height: never terminal
    f = FallFeatures(t=0.0, pelvis_z=0.50, tilt_deg=5.0, torso_up_z=1.0,
                     contacts=ContactState(left_knee=True, right_knee=True))
    assert not det.predicts(f)
    # hands support with a very low pelvis: still never terminal (measured case)
    f2 = FallFeatures(t=0.0, pelvis_z=0.118, tilt_deg=59.0, torso_up_z=0.5,
                      contacts=ContactState(left_hand=True, right_hand=True))
    assert not det.predicts(f2)
    # torso contact always terminates, even with limbs down
    f3 = FallFeatures(t=0.0, pelvis_z=0.2, tilt_deg=30.0, torso_up_z=0.85,
                      contacts=ContactState(torso=True, left_hand=True))
    assert det.predicts(f3)
    # unsupported low+tilted terminates
    f4 = FallFeatures(t=0.0, pelvis_z=0.34, tilt_deg=70.0, torso_up_z=0.34,
                      contacts=ContactState())
    assert det.predicts(f4)
    # the strict variant ignores the limb exemption
    strict = FallDetector(FallDetConfig(limb_support_exempt=False))
    assert strict.predicts(f2)


def test_fall_detector_settled_postures(model):
    q, c = stand_frame(model)
    d = _settle(model, q, c, 0.2)
    f = fall_features(model, d)
    assert not FallDetector().predicts(f), f.as_dict()
    assert f.contacts.feet and not f.contacts.grounded
    q_kneel = _kneel_qpos(model)
    d_kneel = _settle(model, q_kneel, q_kneel[7:36], 0.2)
    fk = fall_features(model, d_kneel)
    assert fk.contacts.knees and not fk.contacts.torso, fk.as_dict()
    assert not FallDetector().predicts(fk)
    q_hands = _hands_qpos(model)
    d_hands = _settle(model, q_hands, q_hands[7:36], 0.8)
    fh = fall_features(model, d_hands)
    assert fh.contacts.hands and not fh.contacts.grounded, fh.as_dict()
    assert not FallDetector().predicts(fh)


def test_scripted_falls_terminate(model):
    env = SoloEnv(seed=0, task="balance")
    env.horizon = 30.0
    env.reset(seed=0, jitter=False)
    fell = None
    info = {}
    for k in range(160):  # 3.2 s
        ctrl = FallForwardController()(env, env.data)
        obs, r, done, trunc, info = env.step(ctrl)
        if done:
            fell = (info["termination"], info["t"], info["fall"])
            break
    assert fell is not None and fell[0] in ("fall", "dorsal"), fell
    assert info["metrics"]["tilt_deg"] > 60.0
    # a backward topple terminates too
    env2 = SoloEnv(seed=0, task="balance")
    env2.horizon = 30.0
    env2.reset(seed=0, jitter=False)
    back = FallForwardController(ankle=0.4)
    term2 = None
    for k in range(160):
        obs, r, done, trunc, info = env2.step(back(env2, env2.data))
        if done:
            term2 = info["termination"]
            break
    assert term2 in ("fall", "dorsal"), term2


def test_dorsal_fires_only_after_persistence(model):
    env, term, info = _supine_run(model, seconds=3.0)
    assert term == "dorsal", (term, info["dorsal"])
    trig = info["dorsal"]["trigger_t"]
    assert trig is not None
    # the trigger must be at least the confirmation window after the dorsal
    # condition first held (recomputed from the trace)
    rows = env.recorder.rows
    onset = None
    for row in rows:
        if row.get("dorsal_contact") and float(row["pelvis_z"]) <= 0.35 \
                and float(row["tilt_deg"]) >= 45.0:
            onset = float(row["t"])
            break
    assert onset is not None
    assert trig - onset >= 0.30 - 2 * STEP_DT, (onset, trig)


def test_no_false_positives_stance_and_stepping(model):
    env = SoloEnv(seed=0, task="balance")
    env.horizon = 30.0
    env.reset(seed=0, jitter=False)
    hold = env._ctrl_stand.copy()
    for k in range(150):  # 3 s stand
        obs, r, done, trunc, info = env.step(hold)
        assert not done, (k, info["termination"])
    # scripted step-in-place: alternate small knee lifts (feet-only contacts)
    d = _fresh(model, *stand_frame(model))
    det = FallDetector()
    for k in range(150):  # 3 s
        c = stand_frame(model)[1].copy()
        if (k // 15) % 2 == 0:
            c[3] += 0.25
            c[0] += 0.12
        else:
            c[9] += 0.25
            c[6] += 0.12
        d.ctrl[:] = c
        mujoco.mj_step(model, d)
        f = fall_features(model, d)
        assert not det.update(f, float(d.time)), (k, f.as_dict())
    assert float(d.qpos[2]) > 0.70


# ---------------------------------------------------------------- rewards
def _upright(**kw) -> RewardInputs:
    base = dict(cmd=Command(vx=0.3, vy=0.0), vel_local=np.array([0.3, 0.0]),
                yaw_rate=0.0, torso_up_z=1.0, pelvis_z=0.79, pelvis_z_prev=0.79,
                foot_contact=(True, True), shot_phase=0.5, shot_depth=0.5,
                shot_depth_prev=0.5, hand_distance=0.05)
    base.update(kw)
    return RewardInputs(**base)


def test_every_reward_term_hand_tested():
    """Each term must rank a genuine state above its degenerate counterpart."""
    good = _upright(
        foot_slip=(0.0, 0.0), foot_air_time=(0.3, 0.3), foot_landed=(True, False),
        hand_distance=0.05, stance_width_meas=0.237, sat_frac=0.0, limit_prox=0.0,
        shot_depth=0.5, shot_depth_prev=0.4, pelvis_z=0.79, pelvis_z_prev=0.70,
        shot_leg_ahead=True, shot_knee_control=True, shot_exited=True,
        shot_time=0.5, recovered=True)
    bad = _upright(
        cmd=Command(vx=0.3), vel_local=np.array([-0.4, 0.3]), yaw_rate=1.5,
        torso_up_z=0.1, pelvis_z=0.25, pelvis_z_prev=0.25, foot_slip=(2.0, 2.0),
        foot_air_time=(0.0, 0.0), foot_landed=(False, False),
        hand_distance=0.9, action=np.full(29, 0.5), prev_action=np.full(29, -0.5),
        stance_width_meas=0.42, sat_frac=1.0, limit_prox=1.0,
        shot_leg_ahead=False, shot_knee_control=False, shot_exited=False,
        shot_time=9.0, recovered=False)
    delta = {
        "track_ang": dict(yaw_rate=0.0),
        "stance_height": dict(pelvis_z=0.79),
        "stance_width": dict(stance_width_meas=0.237),
        "feet_air_time": dict(foot_air_time=(0.3, 0.3), foot_landed=(True, False)),
        "reach": dict(hand_distance=0.05),
        "shot_progress": dict(shot_depth=0.6, shot_depth_prev=0.4),
        "shot_leg_seq": dict(shot_leg_ahead=True),
        "shot_knee": dict(shot_knee_control=True),
        "shot_exit": dict(shot_exited=True),
        "shot_timeout": dict(shot_time=0.5),
        "recover_gain": dict(pelvis_z=0.75, pelvis_z_prev=0.45),
        "recover_done": dict(recovered=True),
        "low_posture": dict(pelvis_z=0.79),
        "alive": dict(torso_up_z=1.0, pelvis_z=0.79),
        "flat_orientation": dict(torso_up_z=1.0),
        "feet_slide": dict(foot_slip=(0.0, 0.0)),
        "action_rate": dict(action=np.zeros(29), prev_action=np.zeros(29)),
        "torque_sat": dict(sat_frac=0.0),
        "joint_limit": dict(limit_prox=0.0),
    }
    for name, fn in TERM_FUNCS.items():
        good_v = fn(good)
        bad_v = fn(bad)
        assert good_v > bad_v, (name, good_v, bad_v)
        # and the term-specific pair is even sharper where one exists
        if name in delta:
            g2 = _upright(**delta[name])
            assert fn(g2) >= fn(bad), (name, fn(g2), fn(bad))


def test_reward_degenerate_cases():
    """Zero action, stand hold, fall-forward, squat, knee-park, shot-park,
    sliding and collapse must each score worse than a genuine attempt."""
    loco = TaskReward("locomotion")
    genuine = _upright(vel_local=np.array([0.3, 0.0]), torso_up_z=1.0, pelvis_z=0.79,
                       foot_air_time=(0.3, 0.3), foot_landed=(True, False),
                       foot_slip=(0.0, 0.0))
    r_genuine, _ = loco.step(genuine)
    # (a) zero action / standing still against a motion command
    still = _upright(cmd=Command(vx=0.3), vel_local=np.zeros(2), yaw_rate=0.0)
    assert loco.step(still)[0] < r_genuine
    # (b) StandHold (stands still, no tracking)
    assert loco.step(still)[0] < r_genuine
    # (c) fall forward at commanded speed: velocity matches, uprightness does not
    falling = _upright(vel_local=np.array([0.5, 0.0]), torso_up_z=0.1,
                       pelvis_z=0.35, foot_slip=(1.0, 1.0))
    assert loco.step(falling)[0] < r_genuine
    # (d) squatting under a stance command: never reaches the commanded height
    stance = TaskReward("stance")
    cmd_low = Command(vx=0.0, stance_height=0.70, skill_id=int(Skill.LEVEL_CHANGE))
    hold_low = _upright(cmd=cmd_low, pelvis_z=0.70, stance_width_meas=0.30)
    squat = _upright(cmd=cmd_low, pelvis_z=0.60, stance_width_meas=0.30)
    assert stance.step(squat)[0] < stance.step(hold_low)[0]
    # (e) sliding instead of stepping
    sliding = _upright(vel_local=np.array([0.3, 0.0]), foot_slip=(1.8, 1.8),
                       foot_air_time=(0.0, 0.0), foot_landed=(False, False))
    assert loco.step(sliding)[0] < r_genuine
    # (f) knee-parked under a recovery command
    rec = TaskReward("recovery")
    rising = _upright(pelvis_z=0.75, pelvis_z_prev=0.45, recovered=False)
    parked = _upright(pelvis_z=0.35, pelvis_z_prev=0.35, recovered=False)
    assert rec.step(parked)[0] < rec.step(rising)[0]
    # (g) shot entered and never exited: parking gains nothing and pays the
    #     timeout, while a genuine shot progresses and exits
    shot = TaskReward("shot")
    park = _upright(shot_depth=1.0, shot_depth_prev=1.0, shot_time=9.0,
                    shot_phase=1.0, shot_exited=False, shot_leg_ahead=False,
                    shot_knee_control=True)
    genuine_shot = _upright(shot_depth=1.0, shot_depth_prev=0.2, shot_time=1.0,
                            shot_phase=1.0, shot_exited=True, shot_leg_ahead=True,
                            shot_knee_control=True)
    assert shot.step(park)[0] < shot.step(genuine_shot)[0]
    # the same park vs genuine over a horizon (parking cannot catch up)
    v_park = discounted_return([shot.step(park)[0]] * 100, shot.gamma)
    v_good = discounted_return([shot.step(genuine_shot)[0]] * 100, shot.gamma)
    assert v_park < v_good
    # (h) collapsed pose in every task
    collapsed = _upright(torso_up_z=0.0, pelvis_z=0.15, vel_local=np.zeros(2),
                         foot_slip=(0.5, 0.5))
    for task in TASK_TERMS:
        tr = TaskReward(task)
        assert tr.step(collapsed)[0] < tr.step(genuine)[0] + 1e-9, task


def test_reward_horizon_collapse_scores_lower():
    """A collapsing policy must score lower than an upright one (Main's rule)."""
    tr = TaskReward("balance")
    up = _upright()
    down = _upright(torso_up_z=0.05, pelvis_z=0.15, foot_slip=(1.0, 1.0))
    r_up, _ = tr.step(up)
    r_down, _ = tr.step(down)
    pen, _ = tr.terminal("fall")
    assert r_up > r_down + 0.5
    v_up = discounted_return([r_up] * 200, tr.gamma)          # 4 s upright
    v_down = discounted_return([r_down] * 30 + [r_down] * 170, tr.gamma, pen)
    assert v_up > v_down
    # timeout is NOT penalized (no "fall to stop paying" incentive)
    assert tr.terminal("timeout")[0] == 0.0
    assert tr.terminal(None)[0] == 0.0


def test_reward_invariant_alive_dominates():
    for task in TASK_TERMS:
        tr = TaskReward(task)
        pen = sum(getattr(tr.weights, t) for t in TASK_TERMS[task]
                  if t in PENALTY_TERMS)
        assert tr.weights.alive > pen, (task, tr.weights.alive, pen)
    with pytest.raises(ValueError):
        from solo.reward import RewardWeights
        TaskReward("balance", RewardWeights(alive=0.1))


def test_reward_terms_logged_separately():
    tr = TaskReward("locomotion")
    total, terms = tr.step(_upright())
    assert set(terms) == set(TASK_TERMS["locomotion"])
    assert sum(getattr(tr.weights, k) * v for k, v in terms.items()) == pytest.approx(total)


# ------------------------------------------------------------------- env
def test_env_determinism():
    env1 = SoloEnv(seed=7, task="balance")
    env2 = SoloEnv(seed=7, task="balance")
    o1 = env1.reset(seed=7)
    o2 = env2.reset(seed=7)
    assert np.array_equal(o1["actor"], o2["actor"])
    hold = env1._ctrl_stand.copy()
    for k in range(60):
        a1 = env1.step(hold)
        a2 = env2.step(hold)
        assert np.array_equal(a1[0]["actor"], a2[0]["actor"])
        assert a1[1] == a2[1]
        assert np.array_equal(env1.data.qpos, env2.data.qpos)
    # different seeds give different resets
    e3 = SoloEnv(seed=8)
    q3 = e3.reset(seed=8)
    assert not np.array_equal(o1["actor"], q3["actor"])


def test_reset_validity(model):
    env = SoloEnv(seed=1)
    for seed in (0, 1, 5):
        obs = env.reset(seed=seed)
        q = env.data.qpos.copy()
        assert np.all(np.isfinite(q))
        assert np.linalg.norm(q[3:7]) == pytest.approx(1.0, abs=1e-6)
        lo = model.jnt_range[1:30, 0]
        hi = model.jnt_range[1:30, 1]
        assert np.all(q[7:36] >= lo - 1e-9) and np.all(q[7:36] <= hi + 1e-9)
        assert 0.75 < q[2] < 0.81
        assert np.allclose(env.data.qvel, 0.0)
        foot_z = [float(env.data.site_xpos[site_id(model, s)][2])
                  for s in ("a_left_foot", "a_right_foot")]
        assert min(foot_z) > -0.01 and max(foot_z) < 0.08
        assert obs["actor"].shape == (ACTOR_DIM,)


def test_step_contract_and_metrics(model):
    env = SoloEnv(seed=2, task="balance")
    env.horizon = 0.6
    obs = env.reset(seed=2)
    for key, dim in (("actor", ACTOR_DIM), ("privileged", PRIV_DIM), ("critic", CRITIC_DIM)):
        assert obs[key].shape == (dim,) and obs[key].dtype == np.float32
    n = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(env._ctrl_stand.copy())
        assert set(("task", "t", "command", "metrics", "contacts", "fall",
                    "dorsal", "markers", "reward_terms")) <= set(info)
        assert isinstance(reward, float)
        n += 1
        if terminated or truncated:
            assert truncated and not terminated and info["termination"] is None
            break
    assert n == int(round(0.6 / STEP_DT)) == len(env.recorder.rows)
    with pytest.raises(RuntimeError):
        env.step(env._ctrl_stand.copy())
    summary = env.recorder.summary()
    assert summary["n_steps"] == n
    fields = [r for r in summary["metrics"]][:3]
    for f in ("upright", "vel_err", "pelvis_z", "slip", "sat_frac", "limit_prox"):
        assert f in summary["metrics"] and summary["metrics"][f]["n"] == n
    # metrics emission to JSONL
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = env.recorder.to_jsonl(Path(td) / "trace.jsonl")
        lines = p.read_text().strip().splitlines()
        assert len(lines) == n
    assert summarize_rows(env.recorder.rows)["metrics"]["upright"]["mean"] > 0.99


def test_shot_markers_physical(model):
    env = SoloEnv(seed=0, task="shot")
    env.horizon = 8.0
    env.reset(seed=0)
    # advance until the shot command is active (its first segment starts at 0.5 s)
    while env.command.skill is not Skill.SHOT_DOUBLE_LEG:
        env.step(env._ctrl_stand.copy())
    env.step(env._ctrl_stand.copy())  # anchor is re-set on shot entry this step
    anchor0 = env._marker_anchor[0].copy()
    yaw0 = float(env._marker_anchor[1])
    m0 = env.markers["a_marker_pelvis"].copy()
    for _ in range(25):
        env.step(env._ctrl_stand.copy())
    assert np.allclose(env._marker_anchor[0], anchor0, atol=1e-9), \
        "shot anchor must be world-fixed once the shot is active"
    assert np.allclose(env.markers["a_marker_pelvis"], m0, atol=1e-9)
    expected = targets(anchor0, yaw0, SHOT_PLAN)["a_marker_pelvis"]
    assert np.allclose(m0, expected, atol=1e-6)
    assert np.linalg.norm(m0[:2] - anchor0) == pytest.approx(SHOT_PLAN.distance, abs=1e-6)
    assert m0[2] == pytest.approx(SHOT_PLAN.height)
    # depth is a function of position, not time
    d0 = penetration_depth(anchor0, m0[:2], SHOT_PLAN)
    d1 = penetration_depth(anchor0 + np.array([0.3, 0.0]), m0[:2], SHOT_PLAN)
    assert d0 == 0.0 and 0.5 < d1 < 0.7
    kind, tgt = active_target(env.markers, 0.9, +1)
    assert kind == "pelvis" and np.allclose(tgt, m0)
    kind, tgt = active_target(env.markers, 0.1, -1)
    assert kind == "hand" and np.allclose(tgt, env.markers["a_marker_hand_l"])


def test_shadowed_command_objects(model):
    """A schedule with changing commands changes the filtered command in-env."""
    env = SoloEnv(seed=0, task="balance")
    env.horizon = 6.0
    cmds = [Command(vx=0.0, skill_id=int(Skill.STANCE)),
            Command(vx=0.4, skill_id=int(Skill.APPROACH)),
            Command(vx=-0.2, skill_id=int(Skill.RETREAT))]
    sched = CommandSchedule([(0.0, cmds[0]), (1.0, cmds[1]), (3.0, cmds[2])])
    env.reset(seed=0, command=sched)
    seen = []
    for _ in range(250):  # 5 s
        obs, r, done, trunc, info = env.step(env._ctrl_stand.copy())
        seen.append((info["t"], env.command.vx, env.command.skill_id))
        assert not done
    early = [s for s in seen if s[0] < 0.5][0]
    mid = [s for s in seen if s[0] > 2.0][0]
    late = [s for s in seen if s[0] > 4.0][0]
    assert early[1] == pytest.approx(0.0, abs=1e-9)
    assert 0.3 < mid[1] < 0.4 and mid[2] == int(Skill.APPROACH)
    assert -0.2 < late[1] < -0.15 and late[2] == int(Skill.RETREAT)


# ------------------------------------------------------------------- eval
def test_criterion_and_gate_logic():
    c = Criterion("fall_rate", "<=", 0.05)
    assert c.check(0.0) and not c.check(0.1) and not c.check(None)
    gate = TaskGate("toy", (Criterion("a", ">=", 1.0), Criterion("b", "<", 2.0)))
    ok, reasons = gate.verdict({"a": 1.5, "b": 1.0})
    assert ok and all(r.startswith("PASS") for r in reasons)
    ok, reasons = gate.verdict({"a": 0.5, "b": 3.0})
    assert not ok and all(r.startswith("FAIL") for r in reasons)
    assert GATES["balance"].provisional is True


def test_eval_run_not_certified_for_stand_hold():
    rep = evaluate(lambda env, seed: StandHoldController(), task="balance",
                   episodes=2, seed0=0, verbose=False)
    take_clips(rep)
    assert rep["verdict"] == "not_certified"
    assert rep["aggregate"]["fall_rate"] == 0.0
    assert rep["aggregate"]["steps_per_s"] is not None
    assert any("max_recoverable_impulse" in r for r in rep["reasons"])
    assert rep["config"]["task"] == "balance"


def test_baseline_controllers_run(model):
    env = SoloEnv(seed=0, task="balance")
    env.horizon = 1.0
    for ctrl in (StandHoldController(model), ZeroActionController(),
                 RandomInitPolicyController(seed=0),
                 RandomInitPolicyController(seed=0, stochastic=False)):
        env.reset(seed=0)
        a = ctrl(env, env.data)
        assert np.asarray(a).shape == (29,)
        obs, r, done, trunc, info = env.step(a)
        assert isinstance(r, float)
    assert not env.data.qfrc_applied.any()


def test_trainer_checkpoint_roundtrip(model):
    """The S2 entry point must start, stop and resume (short burst only)."""
    import tempfile

    from solo.train import SoloTrainer, TrainConfig

    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "t1.pt"
        cfg = TrainConfig(task="balance", steps=256, rollout_steps=128,
                          out=str(ckpt), log_every=0, save_every=0)
        tr = SoloTrainer(cfg, model=model)
        stats = tr.run()
        assert stats["steps"] == 256 and stats["steps_per_s"] > 0
        path = tr.save()
        assert Path(path).exists()
        tr2 = SoloTrainer(cfg, model=model)
        tr2.load(path)
        assert tr2.steps_done == 256 and tr2.iteration == tr.iteration
        tr2.cfg.steps = 384
        stats2 = tr2.run()
        assert stats2["steps"] == 384
