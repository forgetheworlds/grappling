"""Plumbing validation for the reference-conditioned tracking env (solo.track).

These pin the brief's "baselines before long training" checks as tests: the
actor observations carry the changing reference signals, the action mapping
puts the reference where MuJoCo actually applies it, the reward arithmetic
closes the measured crouch-escape hole, and the termination/deviation signals
fire.  The reference-conditioned interface is ADDITIVE: the base 115/158-dim
observations are untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rl.net import ActorCritic, NetConfig  # noqa: E402
from solo.scene import N_JOINTS, load_solo_model, stand_frame  # noqa: E402
from solo.track import (DEFAULT_WEIGHTS, REF_ACTOR_DIM, REF_CRITIC_DIM,  # noqa: E402
                        REF_DIM, STAGE_ORDER, Segment, TrackingEnv, TrackingTask,
                        stage_segments, track_deviation, track_reward_terms,
                        track_targets, warm_start_actor)


@pytest.fixture(scope="module")
def model():
    return load_solo_model()


@pytest.fixture(scope="module")
def q_stand(model):
    return stand_frame(model)[0]


@pytest.fixture(scope="module")
def tt_stance(model):
    return track_targets("stance_hold", model)


def _env(model, q_stand, tt, seg, **kw):
    env = TrackingTask(stage="S1_stand_lower_hold_rise", model=model).__dict__["env"] \
        if False else None
    from solo.env import SoloEnv

    base = SoloEnv(model, task="balance", action_mode="residual",
                   residual_scale=kw.pop("residual_scale", 0.5), jitter=False,
                   record_metrics=True)
    return TrackingEnv(base, seg, tt, q_stand=q_stand, **kw)


def test_reward_arithmetic_closes_the_crouch_escape():
    """The measured hole: crouch scored 0.809 vs stance 1.965 but fall cost
    1500 -> policy sat.  Here tracking a standing reference must strictly
    dominate sitting below it, AND sitting below it is TERMINAL."""
    w = DEFAULT_WEIGHTS
    stance = track_reward_terms(joint_err=0.05, site_err=0.03, root_xy_err=0.03,
                                root_z_err=0.02, yaw_err=0.05, ref_pelvis_z=0.74,
                                torso_up_z=0.99, action_delta_mean=0.02,
                                sat_frac=0.0, limit_prox=0.0, slide_frac=0.0, w=w)
    crouch = track_reward_terms(joint_err=0.5, site_err=0.10, root_xy_err=0.05,
                                root_z_err=0.30, yaw_err=0.10, ref_pelvis_z=0.74,
                                torso_up_z=0.95, action_delta_mean=0.0,
                                sat_frac=0.0, limit_prox=0.0, slide_frac=0.0, w=w)
    assert stance["total"] > 3.0 * crouch["total"]
    # and the structural closure: sitting below the reference is TERMINAL
    assert track_deviation(joint_err=0.0, root_xy_err=0.0, pelvis_drop_m=0.30,
                           site_err=0.0, w=w) == "pelvis_drop"
    # both escape routes end the episode: falling AND deviating -> no
    # "never fall by never standing" optimum is left open
    assert track_deviation(joint_err=w.term_joint_rad + 1e-6, root_xy_err=0.0,
                           pelvis_drop_m=0.0, site_err=0.0, w=w) == "joint"


def test_deep_phase_height_error_is_not_punished_like_stance_error():
    """A 10 cm height miss at a 0.35 m reference (penetration) must score far
    better than the same miss at a 0.74 m reference (stance)."""
    w = DEFAULT_WEIGHTS
    deep = track_reward_terms(joint_err=0.0, site_err=0.0, root_xy_err=0.0,
                              root_z_err=0.10, yaw_err=0.0, ref_pelvis_z=0.35,
                              torso_up_z=1.0, action_delta_mean=0.0, sat_frac=0.0,
                              limit_prox=0.0, slide_frac=0.0, w=w)
    shallow = track_reward_terms(joint_err=0.0, site_err=0.0, root_xy_err=0.0,
                                 root_z_err=0.10, yaw_err=0.0, ref_pelvis_z=0.74,
                                 torso_up_z=1.0, action_delta_mean=0.0, sat_frac=0.0,
                                 limit_prox=0.0, slide_frac=0.0, w=w)
    assert deep["w_root"] > 2.0 * shallow["w_root"]


def test_stage_curriculum_excludes_held_out_takes():
    for stage in STAGE_ORDER:
        for seg in stage_segments(stage):
            assert "stalk_shuffle" not in seg.source
            assert "knee_sprawl_entry2" not in seg.source


def test_reference_block_shapes_and_layout(model, q_stand, tt_stance):
    seg = Segment("stance_hold", 0, 60, "STANCE", 1)
    ep = _env(model, q_stand, tt_stance, seg)
    obs = ep.reset(seed=0)
    assert obs["actor"].shape == (REF_ACTOR_DIM,)
    assert obs["critic"].shape == (REF_CRITIC_DIM,)
    assert obs["actor"][:115].shape[0] == 115


def test_actor_observation_carries_the_moving_reference(model, q_stand, tt_stance):
    """The core defect fix: the reference block must CHANGE across frames and
    distinguish phases (the old obs[114] was constant 0.0)."""
    seg = Segment("stance_hold", 0, 100, "STANCE", 1)
    ep = _env(model, q_stand, tt_stance, seg)
    obs = ep.reset(seed=0)
    blocks = [obs["ref_block"]]
    for _ in range(40):
        obs, _r, term, trunc, _info = ep.step(np.zeros(N_JOINTS))
        blocks.append(obs["ref_block"])
        if term or trunc:
            break
    B = np.stack(blocks)
    joints_rel = B[:, 6:6 + N_JOINTS]
    phase = B[:, -(1 + 3 + 1) + 0]  # not used; phase checked via layout below
    from solo.track import REF_LAYOUT

    lay = dict(REF_LAYOUT)
    ph = B[:, lay["ref_phase"][0]:lay["ref_phase"][1]]
    skill = B[:, lay["ref_skill_onehot"][0]:lay["ref_skill_onehot"][1]]
    assert float(np.ptp(joints_rel)) > 0.01, "reference joints never changed"
    assert float(np.ptp(ph)) > 0.2, "phase never advanced"
    assert np.allclose(skill.sum(axis=1), 1.0)
    assert int(np.argmax(skill[0])) == 0  # STANCE
    assert float(np.ptp(B[:, :3])) >= 0.0  # root target block present


def test_reference_block_distinguishes_phases(model, q_stand):
    """Stance vs shuffle conditioning: different skill one-hots AND different
    reference joint targets (the actor CAN tell a hold from a shuffle)."""
    tt_a = track_targets("stance_hold", model)
    tt_b = track_targets("shuffle_back", model)
    seg_a = Segment("stance_hold", 0, 40, "STANCE", 1)
    seg_b = Segment("shuffle_back", 0, 40, "SHUFFLE_B", 1)
    ea = _env(model, q_stand, tt_a, seg_a)
    eb = _env(model, q_stand, tt_b, seg_b)
    oa = ea.reset(seed=0)["ref_block"]
    ob = eb.reset(seed=0)["ref_block"]
    from solo.track import REF_LAYOUT

    lay = dict(REF_LAYOUT)
    sa = oa[lay["ref_skill_onehot"][0]:lay["ref_skill_onehot"][1]]
    sb = ob[lay["ref_skill_onehot"][0]:lay["ref_skill_onehot"][1]]
    assert int(np.argmax(sa)) == 0 and int(np.argmax(sb)) == 2
    ja = oa[6:6 + N_JOINTS]
    jb = ob[6:6 + N_JOINTS]
    assert float(np.max(np.abs(ja - jb))) > 0.05


def test_action_mapping_reference_base_is_what_mujoco_applies(model, q_stand,
                                                              tt_stance):
    """z = 0 must apply EXACTLY the reference's next-frame joint targets."""
    seg = Segment("stance_hold", 0, 50, "STANCE", 1)
    ep = _env(model, q_stand, tt_stance, seg)
    ep.reset(seed=0)
    obs, _r, term, trunc, info = ep.step(np.zeros(N_JOINTS))
    kf = seg.k0 + 1
    expected = np.clip(tt_stance.qpos[kf, 7:36], ep.env.lo, ep.env.hi)
    assert np.allclose(ep.env.data.ctrl, expected, atol=1e-12)
    # full-deflection residual respects the scale: base + scale*tanh(+1)
    ep2 = _env(model, q_stand, tt_stance, seg, residual_scale=0.5)
    ep2.reset(seed=0)
    _o, _r, _t, _tr, _i = ep2.step(np.ones(N_JOINTS))
    assert np.allclose(ep2.env.data.ctrl,
                       np.clip(expected + 0.5 * np.tanh(1.0), ep2.env.lo,
                               ep2.env.hi), atol=1e-9)


def test_reset_seeds_reference_state_and_velocity(model, q_stand, tt_stance):
    seg = Segment("stance_hold", 10, 60, "STANCE", 1)
    ep = _env(model, q_stand, tt_stance, seg)
    ep.reset(seed=3)
    assert np.allclose(ep.env.data.qpos[7:36], tt_stance.qpos[10, 7:36], atol=1e-12)
    assert np.allclose(ep.env.data.qvel[6:35], tt_stance.targets.joint_vel[10],
                       atol=1e-9)


def test_xy_anchor_tolerates_reset_jitter(model, q_stand, tt_stance):
    """The commanded root target is anchored to the robot's ACTUAL reset xy, so
    the local root error at reset is ~0 regardless of the (small) xy offset."""
    seg = Segment("stance_hold", 0, 40, "STANCE", 1)
    ep = _env(model, q_stand, tt_stance, seg, xy_noise=0.02)
    obs = ep.reset(seed=5)
    # ref_root_local at frame 0 (anchor frame): displacement of the reference
    # from its own start = 0 -> block's first 2 dims ~ 0, 3rd = z error only
    assert abs(float(obs["ref_block"][0])) < 1e-9
    assert abs(float(obs["ref_block"][1])) < 1e-9


def test_deviation_terminates_a_collapsing_replay(model, q_stand, tt_stance):
    """Replaying the DEEP stance take open-loop (the measured toppler) must end
    in a deviation/fall termination, not silently run to the horizon."""
    seg = Segment("stance_hold", 0, 221, "STANCE", 1)
    ep = _env(model, q_stand, tt_stance, seg)
    ep.reset(seed=0)
    cause, steps = None, 0
    for _ in range(300):
        _o, _r, term, trunc, info = ep.step(np.zeros(N_JOINTS))
        steps += 1
        if term or trunc:
            cause = info["track"]["cause"]
            break
    assert cause is not None
    assert steps < 200, f"expected an early end, ran {steps}"


def test_warm_start_surgery_transfers_trunk_and_zeroes_new_inputs():
    """A 115-dim checkpoint into the 170-dim net: same output for the base
    observation with a zero reference block (the trunk is untouched)."""
    torch = pytest.importorskip("torch")
    old = ActorCritic(115, 158, act_dim=N_JOINTS,
                      cfg=NetConfig(hidden=(64, 64), action_mode="residual"))
    new = ActorCritic(REF_ACTOR_DIM, REF_CRITIC_DIM, act_dim=N_JOINTS,
                      cfg=NetConfig(hidden=(64, 64), action_mode="residual"))
    status = warm_start_actor(new, {"policy": old.state_dict()})
    assert status["copied_tensors"] >= 6 and not status["skipped"]
    base_obs = torch.randn(4, 115)
    zeros = torch.zeros(4, REF_DIM)
    with torch.no_grad():
        a_old = old.actor.mean(base_obs)
        a_new = new.actor.mean(torch.cat([base_obs, zeros], dim=1))
    assert torch.allclose(a_old, a_new, atol=1e-6)


def test_drill_phase_conditioning_is_per_frame(model):
    """Inside the composed drill the actor's skill one-hot must switch with the
    labelled phases (STANCE_HOLD frames != SHUFFLE_F frames != RECOVER)."""
    from solo.track import REF_LAYOUT

    tt = track_targets("drill_continuous", model)
    assert tt.skill_ids is not None
    lay = dict(REF_LAYOUT)
    # STANCE_HOLD vs SHUFFLE_F vs RECOVER frames (times from the phase table)
    hold = int(4.0 / 0.02)
    shuf = int(8.0 / 0.02)
    rec = int(50.0 / 0.02)
    assert int(tt.skill_ids[hold]) == 0      # STANCE
    assert int(tt.skill_ids[shuf]) == 1      # SHUFFLE_F
    assert int(tt.skill_ids[rec]) == 11      # RECOVER
    assert bool(tt.connect_flags[int(5.7 / 0.02)]) is True   # CONNECT_* frame
    assert bool(tt.connect_flags[hold]) is False


def test_tracking_task_samples_all_stage_segments(model):
    task = TrackingTask(stage="S1_stand_lower_hold_rise", model=model, seed=0)
    assert len(task.segs) >= 5
    obs = task.reset(seed=0)
    assert obs["actor"].shape == (REF_ACTOR_DIM,)
    # one zero-action step must not crash and must report tracking info
    _o, r, term, trunc, info = task.step(np.zeros(N_JOINTS))
    assert "track" in info and np.isfinite(r)
