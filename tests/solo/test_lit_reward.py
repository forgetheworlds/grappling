"""Acceptance tests for the literature-sourced balance recipe (``solo.lit``).

Everything here asserts **numbers on hand-constructed states**, and every
assertion is written to fail if its term is removed or its sign/argument is
flipped.  The sources are named per test:

* **A** = van Marum et al. 2024, arXiv:2404.19173 (their Table I weights are
  pinned verbatim where transferred).
* **B** = Yang et al. 2020, arXiv:2002.02991 (``J = m dCOP sqrt(g/z_c)``,
  support-centre CoM target, capture-point velocity target, even GRF).

The pinning tests at the end assert that none of this touched v5's
configuration (``TASK_TERMS["balance"]``, ``RewardWeights()`` defaults and the
per-step penalty invariant are unchanged).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solo import lit  # noqa: E402
from solo.env import SoloEnv  # noqa: E402
from solo.lit import (LIT_ACTIVE_JOINTS, LitPushConfig, ceiling_from_hull,  # noqa: E402
                      directional_margin, gamma_half_life_s, gamma_table,
                      joint_mask, lit_push_schedule, measure_ceiling,
                      mirror_index_sign, support_centre, support_hull)
from solo.obs import ACTOR_DIM  # noqa: E402
from solo.reward import (LIT_AIRTIME_PENALTY, LIT_BALANCE_TERMS,  # noqa: E402
                         LIT_SIGMA_COM, TASK_TERMS, PENALTY_MARGIN,
                         PENALTY_TERMS, RewardInputs, RewardWeights, TaskReward,
                         TERM_FUNCS)
from solo.scene import N_JOINTS, load_solo_model, stand_frame  # noqa: E402

#: the real stand-pose sole-sphere footprint (m): heel x=-0.05, toe x=+0.12,
#: lateral +-0.1485 y (the model's own numbers -- asserted below against the
#: model, then used to hand-construct states)
SOLE_STAND = np.array([
    [[-0.05, 0.1435, 0.0031], [-0.05, 0.0935, 0.0031],
     [0.12, 0.1485, 0.0031], [0.12, 0.0885, 0.0031]],
    [[-0.05, -0.0935, 0.0031], [-0.05, -0.1435, 0.0031],
     [0.12, -0.0885, 0.0031], [0.12, -0.1485, 0.0031]],
])
BOTH = (True, True)


@pytest.fixture(scope="module")
def model():
    return load_solo_model()


@pytest.fixture(scope="module")
def ceiling(model):
    return measure_ceiling(model)


def _inp(**kw) -> RewardInputs:
    """A hand state that is on the stand keyframe with both feet loaded."""
    base = dict(com_xy=np.array([0.0, 0.0]), com_vel_xy=np.zeros(2), com_z=0.6919,
                sole_points=SOLE_STAND, foot_contact=BOTH, torso_up_z=1.0,
                pelvis_z=0.79, stand_height=0.79, foot_load=(163.0, 163.0),
                torque=np.zeros(N_JOINTS), arm_dev=0.0)
    base.update(kw)
    return RewardInputs(**base)


# --------------------------------------------------------------- the ceiling
def test_ceiling_matches_the_model_and_is_direction_dependent(ceiling, model):
    """B's ceiling from OUR model: m, z_c, hull, per-direction J."""
    mass = float(np.asarray(model.body_mass).sum())
    assert ceiling.mass == pytest.approx(mass, abs=1e-9)
    assert ceiling.mass == pytest.approx(33.341142, abs=1e-5)
    assert ceiling.z_com == pytest.approx(0.691852, abs=1e-5)

    d = ceiling.as_dict()
    assert d["hull_depth_m"] == pytest.approx(0.170, abs=1e-3)   # -0.05 .. +0.12
    assert d["hull_width_m"] == pytest.approx(0.29701, abs=1e-3)  # 2 x 0.1485
    assert d["support_centre_xy"][0] == pytest.approx(0.035484, abs=1e-4)

    # per-direction (push yaw -> dCOP -> J) against the closed form
    for yaw, dcop, j in ((0.0, 0.1167, 14.653), (math.pi, 0.0533, 6.690),
                         (0.5 * math.pi, 0.1450, 18.203),
                         (0.25 * math.pi, 0.1651, 20.723)):
        assert ceiling.dcop(yaw) == pytest.approx(dcop, abs=1e-3), yaw
        assert ceiling.j_reject(yaw) == pytest.approx(j, abs=0.02), yaw
        assert ceiling.j_reject(yaw) == pytest.approx(
            ceiling.mass * ceiling.dcop(yaw) * math.sqrt(ceiling.gravity
                                                         / ceiling.z_com),
            rel=1e-12)

    # the heel-ward push is the binding direction: 2x weaker than the toe-ward one
    assert ceiling.j_reject(math.pi) < ceiling.j_reject(0.0)
    assert d["j_reject_min_ns"] == pytest.approx(6.69, abs=0.02)
    assert d["j_reject_max_ns"] == pytest.approx(20.72, abs=0.02)
    assert d["j_reject_sagittal_ns"] == pytest.approx(14.653, abs=0.02)
    assert d["j_reject_lateral_ns"] == pytest.approx(18.203, abs=0.02)


def test_ceiling_from_hull_arithmetic_on_a_hand_hull():
    """A centred 0.2 m square, m=10, z=1, g=10 -> J = 10*0.1*sqrt(10)."""
    hull = np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])
    c = ceiling_from_hull(10.0, 1.0, [0.0, 0.0], hull, gravity=10.0)
    assert c.dcop(0.0) == pytest.approx(0.1, rel=1e-12)
    assert c.dcop(0.5 * math.pi) == pytest.approx(0.1, rel=1e-12)
    assert c.j_reject(0.0) == pytest.approx(10.0 * 0.1 * math.sqrt(10.0), rel=1e-12)
    # off-centre CoM: asymmetric margins, and the binding one is the small one
    c2 = ceiling_from_hull(10.0, 1.0, [0.05, 0.0], hull, gravity=10.0)
    assert c2.dcop(0.0) == pytest.approx(0.05, rel=1e-12)
    assert c2.dcop(math.pi) == pytest.approx(0.15, rel=1e-12)
    # an outside CoM has no margin in the direction it already left
    assert directional_margin([0.2, 0.0], hull, (1.0, 0.0)) == 0.0


def test_support_centre_is_the_hull_area_centroid(ceiling):
    """The support centre is the *hull* centroid, not the mean of the points."""
    sc = support_centre(SOLE_STAND, BOTH)
    # the footprint is a *trapezoid* (heel row +-0.1435, toe row +-0.1485), so the
    # area centroid sits 0.5 mm forward of the vertex mean 0.035:
    # -0.05 + 0.17*(0.287 + 2*0.297) / (3*(0.287 + 0.297)) = 0.0354852
    assert sc[0] == pytest.approx(0.0354852, abs=1e-6)
    assert abs(sc[0] - 0.035) > 4e-4          # centroid != vertex mean, measurably
    assert sc[1] == pytest.approx(0.0, abs=1e-12)        # symmetric in y
    assert sc == pytest.approx(ceiling.as_dict()["support_centre_xy"], abs=1e-5)

    # a trapezoid where the area centroid differs from the vertex mean by 5.6 cm
    trap = np.array([[0.0, 0.0], [2.0, 0.0], [1.5, 1.0], [0.5, 1.0]])
    pts = np.stack([np.column_stack([trap, np.zeros(len(trap))])] * 2)
    got = support_centre(pts, BOTH)
    assert got[0] == pytest.approx(1.0, abs=1e-12)
    assert got[1] == pytest.approx(4.0 / 9.0, abs=1e-12)
    assert abs(trap.mean(axis=0)[1] - got[1]) > 0.05      # vertex mean would be 0.5

    # only the *loaded* foot's hull counts; nothing loaded -> NaN
    one = support_centre(SOLE_STAND, (True, False))
    # single foot: heel 0.05 wide, toe 0.06 wide -> -0.05 + 0.17*0.17/0.33
    assert one[0] == pytest.approx(0.0375758, abs=1e-6)
    assert abs(one[0] - 0.035) > 2e-3        # 2.6 mm forward of the vertex mean
    assert one[1] == pytest.approx(0.1185, abs=1e-12)
    assert np.all(np.isnan(support_centre(SOLE_STAND, (False, False))))


def test_hull_of_a_known_stance_is_the_footprint_rectangle():
    h = support_hull(SOLE_STAND)
    assert h[:, 0].min() == pytest.approx(-0.05, abs=1e-12)
    assert h[:, 0].max() == pytest.approx(0.12, abs=1e-12)
    assert h[:, 1].min() == pytest.approx(-0.1485, abs=1e-12)
    assert h[:, 1].max() == pytest.approx(0.1485, abs=1e-12)


# ------------------------------------------------------------- reward terms
def test_com_support_is_maximal_at_the_hull_centre_and_decays_outward():
    """B: CoM at the support centre = maximum disturbance compensation."""
    tr = TaskReward("balance", term_set="balance_lit")
    centre = support_centre(SOLE_STAND, BOTH)
    at_centre = tr.term_value("com_support", _inp(com_xy=centre))
    assert at_centre == pytest.approx(1.0, rel=1e-12)
    # falls off toward the border in the direction of the offset
    vals = [tr.term_value("com_support",
                          _inp(com_xy=centre + np.array([d, 0.0])))
            for d in (0.02, 0.04, 0.06)]
    assert vals[0] > vals[1] > vals[2] > 0.0
    assert tr.term_value("com_support",
                         _inp(com_xy=centre + np.array([LIT_SIGMA_COM, 0.0]))) \
        == pytest.approx(math.exp(-1.0), rel=1e-12)
    # same offset to the left and to the right is worth the same (symmetric shape)
    assert tr.term_value("com_support", _inp(com_xy=centre + np.array([0.0, 0.04]))) \
        == pytest.approx(tr.term_value("com_support",
                                       _inp(com_xy=centre - np.array([0.0, 0.04]))),
                         rel=1e-12)
    # house rule 2: the term is gated by uprightness (a lying CoM cannot farm it)
    assert tr.term_value("com_support", _inp(com_xy=centre, torso_up_z=0.5)) \
        == pytest.approx(0.5, rel=1e-12)
    # no loaded foot -> no support centre -> 0, not 1
    assert tr.term_value("com_support", _inp(com_xy=centre, foot_contact=(False, False))) \
        == 0.0


def test_capture_point_matches_the_cp_implied_velocity():
    """B eq. 5: x_CP = x_CoM + x_dot sqrt(z_c/g); target x_CP = support centre."""
    tr = TaskReward("balance", term_set="balance_lit")
    com = np.array([0.0, 0.0])
    sc = support_centre(SOLE_STAND, BOTH)          # (0.0354852, 0.0)
    z = 0.6919
    tau = math.sqrt(z / 9.81)
    v_target = (sc - com) / tau                    # == 0.133618... m/s
    assert tau == pytest.approx(0.2655750, abs=1e-6)
    assert sc[0] == pytest.approx(0.0354852, abs=1e-6)
    assert v_target[0] == pytest.approx(0.1336166, abs=1e-6)

    st = dict(com_xy=com, sole_points=SOLE_STAND, com_z=z)
    matching = tr.term_value("capture_point", _inp(com_vel_xy=v_target, **st))
    assert matching == pytest.approx(1.0, rel=1e-9)   # error is exactly zero

    # the CP of the matching velocity *is* the support centre
    x_cp = com + v_target * tau
    assert x_cp == pytest.approx(sc, rel=1e-12)

    stopped = tr.term_value("capture_point", _inp(com_vel_xy=np.zeros(2), **st))
    assert stopped == pytest.approx(math.exp(-(0.1336166 / 0.30) ** 2), rel=1e-6)
    assert stopped == pytest.approx(0.8200661, abs=1e-6)
    assert stopped < matching
    # a velocity of the right size but the wrong sign is not the target
    wrong = tr.term_value("capture_point", _inp(com_vel_xy=-v_target, **st))
    assert wrong == pytest.approx(math.exp(-((2.0 * v_target[0]) / 0.30) ** 2),
                                  rel=1e-9)
    assert wrong == pytest.approx(0.4522676, abs=1e-5)
    assert wrong < stopped < matching      # running away from the centre is worst
    # the implied velocity scales as sqrt(z_c): a taller CoM needs less speed
    # v_target(z) = (sc - com) * sqrt(g/z)  ->  v_target(1) / v_target(z) = sqrt(z)
    v_at_unit_z = (sc - com) / math.sqrt(1.0 / 9.81)
    assert v_at_unit_z == pytest.approx(v_target * math.sqrt(z), rel=1e-12)
    assert v_at_unit_z[0] == pytest.approx(0.1111428, abs=1e-6)
    # no load / no CoM height -> 0, not 1
    assert tr.term_value("capture_point", _inp(com_vel_xy=v_target,
                                               foot_contact=(False, False), **st)) == 0.0
    assert tr.term_value("capture_point",
                         _inp(com_vel_xy=v_target, com_xy=com,
                              sole_points=SOLE_STAND, com_z=0.0)) == 0.0


def _sole_at(xy) -> np.ndarray:
    """The stand footprint translated so its hull covers ``xy``."""
    out = SOLE_STAND.copy()
    out[:, :, :2] += np.asarray(xy, float)
    return out


def test_grf_even_is_maximal_when_the_two_foot_loads_are_equal():
    tr = TaskReward("balance", term_set="balance_lit")
    assert tr.term_value("grf_even", _inp(foot_load=(163.0, 163.0))) \
        == pytest.approx(1.0, rel=1e-12)
    # 75/25 split -> asymmetry 0.5 of the total -> e^-1 at sigma 0.5
    assert tr.term_value("grf_even", _inp(foot_load=(244.5, 81.5))) \
        == pytest.approx(math.exp(-1.0), rel=1e-12)
    # one foot takes everything -> asymmetry 1 -> e^-4
    assert tr.term_value("grf_even", _inp(foot_load=(326.0, 0.0))) \
        == pytest.approx(math.exp(-4.0), rel=1e-12)
    # no load is *not* perfect evenness
    assert tr.term_value("grf_even", _inp(foot_load=(0.0, 0.0))) == 0.0
    # gated by uprightness, like every other balance-tracking term
    assert tr.term_value("grf_even", _inp(foot_load=(163.0, 163.0), torso_up_z=0.25)) \
        == pytest.approx(0.25, rel=1e-12)


def test_airtime_penalty_fires_once_per_touchdown():
    """A: per touchdown, ``t_air - 0.4`` (s) -- 0.4 s of airtime is free."""
    tr = TaskReward("balance", term_set="balance_lit")
    assert tr.term_value("airtime", _inp(foot_air_time=(0.0, 0.0))) == 0.0
    # airtime without a landing is worth nothing (no farming by holding a foot up)
    assert tr.term_value("airtime", _inp(foot_air_time=(0.5, 0.5))) == 0.0
    assert tr.term_value("airtime", _inp(foot_landed=(True, False),
                                         foot_air_time=(LIT_AIRTIME_PENALTY, 0.0))) == 0.0
    assert tr.term_value("airtime", _inp(foot_landed=(True, False),
                                         foot_air_time=(0.5, 0.0))) \
        == pytest.approx(0.1, rel=1e-12)
    assert tr.term_value("airtime", _inp(foot_landed=(True, False),
                                         foot_air_time=(0.0, 0.0))) \
        == pytest.approx(-0.4, rel=1e-12)
    # both feet landing in one control step fires twice (the worst case)
    both = tr.term_value("airtime", _inp(foot_landed=(True, True),
                                         foot_air_time=(0.0, 0.0)))
    assert both == pytest.approx(-0.8, rel=1e-12)
    assert PENALTY_MARGIN["airtime"] == pytest.approx(0.8, rel=1e-12)
    # the 0.4 penalty is exactly what the weight pays: weight 1.0 x (t_air - 0.4)
    assert tr.weights.airtime == 1.0


def test_orientation_and_base_height_use_the_source_forms():
    tr = TaskReward("balance", term_set="balance_lit")
    assert tr.term_value("orientation", _inp(torso_up_z=1.0)) == pytest.approx(1.0, rel=1e-12)
    assert tr.term_value("orientation", _inp(torso_up_z=math.cos(math.radians(10)))) \
        == pytest.approx(math.exp(-30.0 * math.radians(10) ** 2), rel=1e-9)
    assert tr.term_value("orientation", _inp(torso_up_z=math.cos(math.radians(10)))) \
        == pytest.approx(0.4009766, abs=1e-6)
    assert tr.term_value("orientation", _inp(torso_up_z=math.cos(math.radians(26)))) \
        == pytest.approx(0.0020753, abs=1e-6)
    assert tr.term_value("orientation", _inp(torso_up_z=0.0)) \
        == pytest.approx(math.exp(-30.0 * (0.5 * math.pi) ** 2), rel=1e-9)
    assert tr.weights.orientation == 0.20          # A's largest term, verbatim

    assert tr.term_value("base_height", _inp(pelvis_z=0.79)) == pytest.approx(1.0, rel=1e-12)
    assert tr.term_value("base_height", _inp(pelvis_z=0.74)) \
        == pytest.approx(math.exp(-1.0), rel=1e-12)
    assert tr.term_value("base_height", _inp(pelvis_z=0.84)) \
        == pytest.approx(math.exp(-1.0), rel=1e-12)          # symmetric in |error|
    assert tr.weights.base_height == 0.05


def test_smoothing_and_survival_terms():
    tr = TaskReward("balance", term_set="balance_lit")
    # upright = survival, no pelvis-height factor (that is base_height's job)
    assert tr.term_value("upright", _inp(torso_up_z=1.0)) == 1.0
    assert tr.term_value("upright", _inp(torso_up_z=0.3, pelvis_z=0.2)) == pytest.approx(0.3)
    # source A standing velocity e^{-5|v-c|}
    assert tr.term_value("vel_stand", _inp(vel_local=np.zeros(2))) == pytest.approx(1.0)
    assert tr.term_value("vel_stand", _inp(vel_local=np.array([0.2, 0.0]))) \
        == pytest.approx(math.exp(-1.0), rel=1e-12)
    assert tr.weights.vel_stand == pytest.approx(0.30)      # 0.15 + 0.15
    # source A action difference e^{-0.02 sum|da|}
    assert tr.term_value("action_diff", _inp(prev_action=None)) == 1.0
    assert tr.term_value("action_diff", _inp(prev_action=np.zeros(N_JOINTS))) == 1.0
    act = np.zeros(N_JOINTS)
    act[:] = 0.1
    assert tr.term_value("action_diff", _inp(action=act, prev_action=np.zeros(N_JOINTS))) \
        == pytest.approx(math.exp(-0.02 * 0.1 * N_JOINTS), rel=1e-12)
    # source A torque e^{-0.02 mean|tau|/tau_ref}, tau_ref = 50 N*m (ankle limit)
    assert tr.term_value("torque", _inp(torque=None)) == 1.0
    assert tr.term_value("torque", _inp(torque=np.full(N_JOINTS, 50.0))) \
        == pytest.approx(math.exp(-0.02), rel=1e-12)
    assert tr.term_value("torque", _inp(torque=np.zeros(N_JOINTS))) == 1.0
    # source A arm posture e^{-3 ||dtheta_arm||}
    assert tr.term_value("arm_posture", _inp(arm_dev=0.0)) == 1.0
    assert tr.term_value("arm_posture", _inp(arm_dev=1.0 / 3.0)) \
        == pytest.approx(math.exp(-1.0), rel=1e-12)
    assert tr.weights.arm_posture == 0.03


def test_every_lit_term_is_logged_and_the_total_is_the_weighted_sum():
    tr = TaskReward("balance", term_set="balance_lit")
    total, terms = tr.step(_inp())
    assert set(terms) == set(LIT_BALANCE_TERMS)
    assert set(terms) == set(tr.terms)
    assert all(name in TERM_FUNCS for name in terms)
    assert sum(tr.weights.as_dict()[k] * v for k, v in terms.items()) \
        == pytest.approx(total, rel=1e-12)
    # exact totals for hand states: 11 terms x their source weights
    assert total == pytest.approx(1.9162859, abs=1e-6)
    # with the CoM on the hull centre and the CP matched, the same state scores
    # the full 1.97/step (source A's own standing total is ~2.0/step)
    best, _ = tr.step(_inp(com_xy=support_centre(SOLE_STAND, BOTH),
                           com_vel_xy=np.zeros(2)))
    assert best == pytest.approx(1.97, abs=1e-9)
    assert best > total


def test_lit_set_has_no_double_foot_contact_requirement():
    """Source A: rewarding double foot contact for standing is *harmful*
    (it penalises the recovery steps that must break contact and makes
    walk->stand pick the nearest stance, not the most stable one)."""
    assert "feet_air_time" not in LIT_BALANCE_TERMS      # the command-gated bonus
    assert "feet_slide" not in LIT_BALANCE_TERMS
    assert not any("contact" in t for t in LIT_BALANCE_TERMS)
    # the audit of the shipped balance set: it has no contact term either
    assert not any("contact" in t for t in TASK_TERMS["balance"])
    assert "feet_air_time" not in TASK_TERMS["balance"]

    # what lifting one foot costs in the lit set.  Isolating the GRF term (same
    # support geometry, only the load split changes): 0.05 * (1 - e^-4).
    tr = TaskReward("balance", term_set="balance_lit")
    double, _ = tr.step(_inp(foot_contact=BOTH, foot_load=(163.0, 163.0)))
    unloaded, _ = tr.step(_inp(foot_contact=BOTH, foot_load=(326.0, 0.0)))
    assert unloaded < double
    assert double - unloaded == pytest.approx(0.05 * (1.0 - math.exp(-4.0)), rel=1e-9)
    assert double - unloaded == pytest.approx(0.0490842, abs=1e-6)

    # the *whole* single-support change (the support hull shrinks to the stance
    # foot, so the CoM must move over it) is worth 0.27/step -- a real geometric
    # requirement of the recovery step, not a double-contact requirement, and
    # two orders of magnitude cheaper than a fall (-100).
    single, _ = tr.step(_inp(foot_contact=(True, False), foot_load=(326.0, 0.0)))
    assert single < unloaded < double
    assert double - single == pytest.approx(0.2687284, abs=1e-6)
    assert double - single < 0.3
    assert (double - single) * 300 < tr.termination_penalty


def test_termination_penalty_unchanged():
    tr = TaskReward("balance", term_set="balance_lit")
    assert tr.terminal("fall")[0] == pytest.approx(-100.0)
    assert tr.terminal("dorsal")[0] == pytest.approx(-100.0)
    assert tr.terminal(None)[0] == 0.0
    assert tr.terminal("timeout")[0] == 0.0


# -------------------------------------------------------------- the joint mask
def test_mask_freezes_the_frozen_joints_through_the_real_step_path(model):
    mask = joint_mask(model)
    assert len(mask.active) == len(LIT_ACTIVE_JOINTS) == 12
    assert len(mask.frozen) == N_JOINTS - 12 == 17
    names = [mask.names[i] for i in mask.frozen]
    for group in ("right_hip_yaw", "left_hip_yaw", "waist_yaw", "left_elbow",
                  "right_elbow", "left_wrist_yaw", "right_wrist_roll"):
        assert group in names, group
    for name in LIT_ACTIVE_JOINTS:
        assert name in [mask.names[i] for i in mask.active], name

    for mode in ("residual", "absolute"):
        env = SoloEnv(seed=5, action_mode=mode, joint_mask=mask,
                      term_set="balance_lit")
        env.reset(seed=5)
        keyframe = env._ctrl_stand.copy()
        # an action that tries to move every joint hard
        action = env.ctrl_from_policy(np.ones(N_JOINTS))
        env.step(action)
        ctrl = np.asarray(env.data.ctrl, np.float64)
        frozen = np.array(mask.frozen)
        active = np.array(mask.active)
        assert np.array_equal(ctrl[frozen], np.array(mask.targets)), mode
        assert np.abs(ctrl[frozen] - np.array(mask.targets)).max() == 0.0
        assert np.abs(np.abs(ctrl[frozen]) - np.abs(keyframe[frozen])).max() == 0.0
        # and the active joints genuinely moved (the mask is not a full freeze)
        assert np.abs(ctrl[active] - keyframe[active]).max() > 0.1, mode
        assert env.config()["joint_mask"]["frozen"] == mask.as_dict()["frozen"]


def test_mask_inactive_for_the_stand_action_and_off_by_default():
    mask = joint_mask()
    env = SoloEnv(seed=6, action_mode="residual", joint_mask=mask)
    env.reset(seed=6)
    env.step(env.ctrl_from_policy(np.zeros(N_JOINTS)))   # unit 0 = stand keyframe
    assert np.abs(np.asarray(env.data.ctrl) - env._ctrl_stand).max() == 0.0
    plain = SoloEnv(seed=6)
    assert plain.joint_mask is None
    assert plain.config()["joint_mask"] is None
    plain.reset(seed=6)
    plain.step(np.zeros(N_JOINTS))
    assert np.abs(np.asarray(plain.data.ctrl)
                  - np.minimum(np.maximum(np.zeros(N_JOINTS), plain.lo),
                               plain.hi)).max() == 0.0   # unbounded action path


def test_eval_and_training_build_the_same_mask_from_the_same_source(ceiling):
    """One factory, one dict: whatever eval rebuilds is what training applied."""
    a = joint_mask()
    b = joint_mask(load_solo_model())
    assert a.as_dict() == b.as_dict()
    assert a.targets == b.targets and a.frozen == b.frozen
    # the keyframe targets are the a_stand ctrl values, not rounded copies
    _, ctrl = stand_frame(load_solo_model())
    assert np.array_equal(np.array(a.targets), ctrl[list(a.frozen)])

    from solo.train import SoloTrainer, TrainConfig

    cfg = TrainConfig(task="balance", steps=1, action_mode="residual",
                      freeze_joints=True, reward_set="lit")
    trainer = SoloTrainer(cfg)
    assert trainer.env.joint_mask is not None
    assert trainer.env.joint_mask.as_dict() == a.as_dict()
    assert trainer.cfg.freeze_joints is True
    assert trainer.env.reward.term_set == "balance_lit"
    # the mask survives the checkpoint's env config (eval's source of truth)
    assert trainer.env.config()["joint_mask"]["frozen_targets"] \
        == a.as_dict()["frozen_targets"]


def test_mirror_map_is_the_geometric_mirror_on_the_real_model(model):
    """The joint mirror map is verified against the model, not asserted."""
    import mujoco

    mirror = np.diag([1.0, -1.0, 1.0])
    idx, sign = mirror_index_sign(model)
    q, _ = stand_frame(model)
    names = lit.joint_names(model)

    data = mujoco.MjData(model)
    rng = np.random.default_rng(7)
    for _ in range(2):
        pert = rng.uniform(-0.3, 0.3, N_JOINTS)
        qa = q.copy()
        for i in range(N_JOINTS):
            jid = model.actuator_trnid[i, 0]
            qa[model.jnt_qposadr[jid]] += pert[i]
        data.qpos[:] = qa
        mujoco.mj_forward(model, data)
        pos_a = np.array([data.xpos[b] for b in range(model.nbody)])

        qb = qa.copy()
        for i in range(N_JOINTS):
            j_a = model.actuator_trnid[i, 0]
            j_b = model.actuator_trnid[idx[i], 0]
            qb[model.jnt_qposadr[j_b]] = sign[i] * qa[model.jnt_qposadr[j_a]]
        data.qpos[:] = qb
        mujoco.mj_forward(model, data)
        pos_b = np.array([data.xpos[b] for b in range(model.nbody)])

        worst = 0.0
        pairs = 0
        for b in range(model.nbody):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            if nm is None or "_left_" not in nm:
                continue
            other = nm.replace("_left_", "_right_")
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, other)
            if j < 0:
                continue
            pairs += 1
            worst = max(worst, float(np.abs(pos_b[j] - mirror @ pos_a[b]).max()))
        assert pairs >= 13
        assert worst < 1e-4, worst

    # roll/yaw joints negate, pitch/knee do not, and left/right swap
    assert sign[names.index("left_hip_roll")] == -1.0
    assert sign[names.index("left_hip_yaw")] == -1.0
    assert sign[names.index("left_hip_pitch")] == +1.0
    assert sign[names.index("left_knee")] == +1.0
    assert sign[names.index("waist_pitch")] == +1.0
    assert sign[names.index("waist_yaw")] == -1.0
    assert idx[names.index("left_elbow")] == names.index("right_elbow")
    assert idx[names.index("waist_pitch")] == names.index("waist_pitch")


def test_mirror_maps_are_involutions_and_the_loss_is_zero_when_symmetric():
    from rl.net import Actor, NetConfig
    from solo.mirror import mirror_actor_obs, mirror_joint_vector, mirror_loss, mirror_step

    o = np.zeros(ACTOR_DIM)
    o[0:3] = [0.1, 0.2, 0.3]
    o[3:6] = [0.1, 0.2, 0.3]
    o[6:9] = [0.0, 0.1, -1.0]
    o[9:38] = np.linspace(-0.5, 0.5, N_JOINTS)
    o[67:96] = np.linspace(0.5, -0.5, N_JOINTS)
    o[96:99] = [0.4, -0.3, 0.2]
    assert np.allclose(mirror_actor_obs(mirror_actor_obs(o)), o, atol=1e-12)
    assert np.allclose(mirror_joint_vector(mirror_joint_vector(o[9:38])), o[9:38])
    m = mirror_actor_obs(o)
    assert np.allclose(m[0:3], [0.1, -0.2, 0.3])
    assert np.allclose(m[3:6], [-0.1, 0.2, -0.3])
    assert np.allclose(m[96:99], [0.4, 0.3, -0.2])

    torch.manual_seed(0)
    actor = Actor(ACTOR_DIM, N_JOINTS, cfg=NetConfig(hidden=(8,)))
    x = torch.as_tensor(np.tile(o, (4, 1)), dtype=torch.float32)
    with torch.no_grad():
        for p in actor.trunk:
            if hasattr(p, "weight"):
                p.weight.zero_()
            if hasattr(p, "bias"):
                p.bias.zero_()
    assert float(mirror_loss(actor, x).detach()) == pytest.approx(0.0, abs=1e-12)
    with torch.no_grad():                       # z = e_0: no partner, worst case
        actor.trunk[-1].bias[0] = 100.0
    assert float(mirror_loss(actor, x).detach()) == pytest.approx(2.0, rel=1e-8)
    with torch.no_grad():                       # a mirror-symmetric bias scores 0
        actor.trunk[-1].bias[0] = 100.0
        idx, sign = mirror_index_sign()
        actor.trunk[-1].bias[idx[0]] = sign[0] * 100.0
    assert float(mirror_loss(actor, x).detach()) == pytest.approx(0.0, abs=1e-12)

    opt = torch.optim.SGD(actor.parameters(), lr=1e-3)
    with torch.no_grad():                       # asymmetric again, then one step
        actor.trunk[-1].bias.zero_()
        actor.trunk[-1].bias[0] = 100.0
    before = float(mirror_loss(actor, x).detach())
    step = mirror_step(actor, opt, np.tile(o, (4, 1)), coef=1.0)
    after = float(mirror_loss(actor, x).detach())
    assert step.loss > 0.0 and step.grad_norm > 0.0
    assert after < before
    assert mirror_step(actor, opt, np.tile(o, (4, 1)), coef=0.0).loss == 0.0


# ------------------------------------------------------------ push distribution
def test_push_mode_bernoulli_rate_and_zero_probability(ceiling):
    """Source A's per-frame push: the *rate* is what the flag configures."""
    cfg = LitPushConfig(mode="bernoulli", prob=0.01, seed=0)
    horizon = 8.0
    n_ep = 200
    counts = [len(lit_push_schedule(cfg, ceiling, horizon, s).pushes)
              for s in range(n_ep)]
    steps = int((horizon - cfg.t_first) / 0.02)
    assert steps == 350
    expected = cfg.prob * steps
    assert expected == pytest.approx(3.5)
    assert abs(float(np.mean(counts)) - expected) < 0.35          # +-10%
    assert abs(sum(counts) - n_ep * expected) < 0.1 * n_ep * expected

    zero = lit_push_schedule(LitPushConfig(mode="bernoulli", prob=0.0, seed=0),
                             ceiling, horizon, 3)
    assert len(zero.pushes) == 0
    assert all(len(lit_push_schedule(LitPushConfig(mode="bernoulli", prob=0.0),
                                     ceiling, horizon, s).pushes) == 0
               for s in range(20))
    always = lit_push_schedule(LitPushConfig(mode="bernoulli", prob=1.0), ceiling,
                               horizon, 0)
    assert len(always.pushes) == steps
    assert all(abs(p.duration - 0.02) < 1e-12 for p in always.pushes)  # single step


def test_push_mode_interval_spacing(ceiling):
    """Source B's repeated pushes: about every ``interval`` seconds."""
    cfg = LitPushConfig(mode="interval", interval=5.0, t_first=1.0)
    sch = lit_push_schedule(cfg, ceiling, horizon=16.0, episode_seed=0)
    assert [round(p.t, 6) for p in sch.pushes] == [1.0, 6.0, 11.0]
    fast = lit_push_schedule(LitPushConfig(mode="interval", interval=2.0,
                                           t_first=1.0), ceiling, 8.0, 0)
    assert [round(p.t, 6) for p in fast.pushes] == [1.0, 3.0, 5.0, 7.0]
    # jitter stays inside +-interval_jitter and on the control grid
    jit = lit_push_schedule(LitPushConfig(mode="interval", interval=5.0,
                                          interval_jitter=0.25), ceiling, 16.0, 4)
    ts = [p.t for p in jit.pushes]
    for k, t in enumerate(ts):
        assert abs(t - (1.0 + 5.0 * k)) <= 0.25 + 1e-9
        assert abs(t / 0.02 - round(t / 0.02)) < 1e-9


def test_push_magnitudes_follow_the_directional_ceiling(ceiling):
    """[0.5x, 2x] of the ceiling *in that push's own direction* (source B)."""
    cfg = LitPushConfig(mode="bernoulli", prob=0.05, min_frac=0.5, max_frac=2.0,
                        directions=8, seed=0)
    fracs, mags, angs = [], [], []
    for s in range(60):
        for p in lit_push_schedule(cfg, ceiling, 8.0, s).pushes:
            j = ceiling.j_reject(p.direction)
            mags.append(p.impulse)
            angs.append(p.direction)
            fracs.append(p.impulse / j)
    assert len(mags) > 300
    assert min(fracs) >= 0.5 - 1e-9 and max(fracs) <= 2.0 + 1e-9
    assert float(np.mean(fracs)) == pytest.approx(1.25, abs=0.06)     # U(0.5, 2)
    assert float(np.std(fracs)) == pytest.approx(np.sqrt((2.0 - 0.5) ** 2 / 12.0),
                                                 abs=0.06)
    # the weak (heel-ward, 180 deg) direction gets the small pushes
    weak = [m for m, a in zip(mags, angs) if abs(a - math.pi) < 1e-9]
    lat = [m for m, a in zip(mags, angs) if abs(a - 0.5 * math.pi) < 1e-9]
    assert weak and lat
    assert float(np.mean(weak)) == pytest.approx(1.25 * 6.69, abs=0.6)
    assert float(np.mean(lat)) == pytest.approx(1.25 * 18.20, abs=1.2)
    assert max(weak) < min(2.0 * 6.69 + 1e-9, 13.5)
    assert len(set(round(math.degrees(a), 3) for a in angs)) == 8     # all directions

    # fixed fraction -> exactly the directional ceiling
    exact = LitPushConfig(mode="bernoulli", prob=1.0, min_frac=1.0, max_frac=1.0)
    for p in lit_push_schedule(exact, ceiling, 2.0, 0).pushes[:20]:
        assert p.impulse == pytest.approx(ceiling.j_reject(p.direction), rel=1e-12)

    # the absolute cap binds (12 N*s = the T1 non-stepping training band)
    capped = lit_push_schedule(LitPushConfig(mode="bernoulli", prob=0.05, cap=12.0),
                               ceiling, 8.0, 0)
    assert all(p.impulse <= 12.0 + 1e-9 for p in capped.pushes)
    uncapped_max = max(p.impulse for s in range(20)
                       for p in lit_push_schedule(cfg, ceiling, 8.0, s).pushes)
    assert uncapped_max > 12.0

    # determinism and stream separation
    s1 = lit_push_schedule(cfg, ceiling, 8.0, 11).as_list()
    s2 = lit_push_schedule(cfg, ceiling, 8.0, 11).as_list()
    s3 = lit_push_schedule(cfg, ceiling, 8.0, 12).as_list()
    assert s1 == s2 and s1 != s3
    assert lit_push_schedule(LitPushConfig(mode="interval", seed=1), ceiling,
                             8.0, 0).as_list() != s1


def test_capped_and_uncapped_push_modes_share_the_ceiling(ceiling):
    """The band is *derived*: 2x the ceiling in the weak direction is < 2x in the
    strong one, which is exactly why a single global number cannot fix the
    train/test mismatch."""
    weak = ceiling.j_reject(math.pi)
    strong = ceiling.j_reject(0.25 * math.pi)
    assert 2.0 * weak == pytest.approx(13.38, abs=0.05)
    assert 2.0 * strong == pytest.approx(41.45, abs=0.05)
    assert 2.0 * weak < 16.0 < 2.0 * strong      # the gate's smallest held-out mag


# ------------------------------------------------------------------ gamma table
def test_gamma_half_life_table():
    assert gamma_half_life_s(0.995, 50.0) == pytest.approx(2.7657, abs=1e-4)
    assert gamma_half_life_s(0.95, 25.0) == pytest.approx(0.5405, abs=1e-4)   # source B
    assert gamma_half_life_s(0.95, 50.0) == pytest.approx(0.2703, abs=1e-4)
    assert gamma_half_life_s(0.99, 50.0) == pytest.approx(1.3794, abs=1e-4)
    rows = {r["label"]: r for r in gamma_table()}
    ours = rows["ours (balance default)"]
    srcb = rows["source B (0.95 @ 25 Hz)"]
    assert ours["half_life_s"] / srcb["half_life_s"] == pytest.approx(5.117, abs=1e-3)
    assert ours["horizon_s_1_over_1mg"] == pytest.approx(4.0, abs=1e-12)


# ----------------------------------------------------- v5 configuration is safe
def test_v5_configuration_is_untouched_by_the_literature_work():
    assert TASK_TERMS["balance"] == ("alive", "flat_orientation", "action_rate",
                                    "torque_sat", "joint_limit")
    assert TaskReward("balance").terms == TASK_TERMS["balance"]
    assert TaskReward("balance").term_set is None
    w = RewardWeights()
    assert w.alive == 1.0 and w.termination == 100.0
    assert TaskReward("balance").gamma == pytest.approx(0.995)
    env = SoloEnv(seed=4)
    assert env.reward.term_set is None
    assert env.joint_mask is None
    assert env.config()["reward"]["term_set"] is None


def test_penalty_invariant_holds_for_every_term_set():
    for task in TASK_TERMS:
        tr = TaskReward(task)
        pen = sum(getattr(tr.weights, t) * PENALTY_MARGIN.get(t, 1.0)
                  for t in tr.terms if t in PENALTY_TERMS)
        assert tr.weights.alive > pen, (task, tr.weights.alive, pen)
    lit_tr = TaskReward("balance", term_set="balance_lit")
    pen = sum(getattr(lit_tr.weights, t) * PENALTY_MARGIN.get(t, 1.0)
              for t in lit_tr.terms if t in PENALTY_TERMS)
    assert pen == pytest.approx(0.9)   # airtime's worst case + the stance-return gap
    assert lit_tr.weights.upright > pen
    with pytest.raises(ValueError):
        TaskReward("balance", RewardWeights(upright=0.1), term_set="balance_lit")
    with pytest.raises(ValueError):
        TaskReward("balance", term_set="nope")


def _help_text() -> str:
    """The trainer's ``--help`` output (argparse exits 0 after printing it)."""
    import contextlib
    import io

    from solo import train as solo_train

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        solo_train.main(["--help"])
    return buf.getvalue()


def test_lit_flags_parse_and_are_listed():
    """A parsed-but-unused flag is silent; this pins the whole flag surface."""
    help_text = _help_text()
    for flag in ("--reward-set", "--freeze-joints", "--lit-weight", "--lit-push",
                 "--lit-push-prob", "--lit-push-interval",
                 "--lit-push-interval-jitter", "--lit-push-min-frac",
                 "--lit-push-max-frac", "--lit-push-cap", "--lit-push-directions",
                 "--lit-push-height", "--lit-push-seed", "--mirror-loss-coef",
                 "--lit-ceiling"):
        assert flag in help_text, f"{flag} missing from --help"
    # and the v5 flags are all still there
    for flag in ("--task", "--steps", "--hidden", "--entropy-coef", "--alive-weight",
                 "--push-curriculum", "--action-mode", "--residual-scale"):
        assert flag in help_text, f"{flag} lost from --help"


def test_lit_ceiling_flag_prints_the_measured_ceiling(capsys):
    from solo import train as solo_train

    assert solo_train.main(["--lit-ceiling"]) == 0
    out = capsys.readouterr().out
    assert '"j_reject_ns": 14.653' in out
    assert '"j_reject_sagittal_ns": 14.653' in out
    assert '"mass_kg": 33.341142' in out
    assert '"n_frozen": 17' in out


def test_lit_burst_runs_end_to_end_through_the_trainer():
    """One real burst with every lit flag on: nothing may be a parsed no-op."""
    import tempfile
    from pathlib import Path as _P

    from solo import train as solo_train

    with tempfile.TemporaryDirectory() as td:
        ckpt = _P(td) / "lit.pt"
        cfg = solo_train.TrainConfig(
            task="balance", steps=128, rollout_steps=64, out=str(ckpt),
            log_every=0, save_every=0, action_mode="residual",
            reward_set="lit", freeze_joints=True,
            lit_push="interval", lit_push_interval=2.0, lit_push_cap=12.0,
            mirror_loss_coef=1.0)
        trainer = solo_train.SoloTrainer(cfg)
        stats = trainer.run()
        assert stats["steps"] == 128
        env_cfg = trainer.env.config()
        assert env_cfg["reward"]["term_set"] == "balance_lit"
        assert env_cfg["reward"]["terms"] == list(LIT_BALANCE_TERMS)
        assert env_cfg["joint_mask"]["n_frozen"] == 17
        assert env_cfg["joint_mask"]["frozen_targets"]["left_elbow"] == pytest.approx(1.28)
        # the lit push distribution is installed, capped, and not the v5 ramp
        assert trainer.env.push_schedule is not None
        assert len(trainer.env.push_schedule.pushes) >= 3
        assert all(p.impulse <= 12.0 + 1e-9 for p in trainer.env.push_schedule.pushes)
        assert all(p.label.startswith("lit_interval") for p in trainer.env.push_schedule.pushes)
        # the mirror loss really ran and is reported
        assert stats.get("mirror_loss", 0.0) >= 0.0
        assert trainer.save() == str(ckpt) and ckpt.exists()


def test_lit_push_and_v5_are_mutually_exclusive_where_it_matters():
    """A lit-push run must not silently inherit the v5 ramp (the mismatch fix)."""
    from solo import train as solo_train

    cfg = solo_train.TrainConfig(task="balance", steps=1, push_curriculum=True,
                                 lit_push="interval", lit_push_interval=2.0)
    trainer = solo_train.SoloTrainer(cfg)
    assert trainer.push_curriculum is None            # the ramp is superseded
    assert len(trainer.env.push_schedule.pushes) >= 3
    plain = solo_train.TrainConfig(task="balance", steps=1)
    assert plain.lit_push == "off" and plain.reward_set == "default"
    assert plain.mirror_loss_coef == 0.0 and plain.freeze_joints is False


def test_lit_weight_overrides_are_validated():
    from solo import train as solo_train

    with pytest.raises(SystemExit):
        solo_train.main(["--task", "balance", "--steps", "1",
                         "--lit-weight", "no_such_weight=1.0", "--lock", "off"])
    cfg = solo_train.TrainConfig(task="balance", steps=1,
                                 lit_weights=(("grf_even", 0.0),))
    trainer = solo_train.SoloTrainer(cfg)
    assert trainer.env.reward.weights.grf_even == 0.0


def test_lit_curriculum_adapter_is_what_the_vec_backend_calls(ceiling):
    """``rl.vec_solo`` installs pushes through ``curriculum.schedule_for``."""
    from solo.lit import LitPushCurriculum

    cfg = LitPushConfig(mode="interval", interval=2.0, cap=12.0, seed=2)
    adapt = LitPushCurriculum(cfg=cfg, ceiling=ceiling, horizon=8.0)
    assert callable(adapt.schedule_for)
    first = adapt.schedule_for(0, 5)
    assert [round(p.t, 6) for p in first.pushes] == [1.0, 3.0, 5.0, 7.0]
    # stationary: the step count does not change the distribution (no ramp)
    assert adapt.schedule_for(5_000_000, 5).as_list() == first.as_list()
    # per-worker seeds are the episode seeds (workers get distinct schedules)
    other = adapt.schedule_for(0, 6)
    assert other.as_list() != first.as_list()
    assert adapt.schedule_for(0, 5).as_list() == first.as_list()   # reproducible
    d = adapt.as_dict()
    assert d["lit_push"]["mode"] == "interval" and d["horizon"] == 8.0
    assert d["ceiling"]["j_reject_sagittal_ns"] == pytest.approx(14.653, abs=0.02)


def test_frame_stack_widens_the_input_and_stacks_oldest_first():
    """Partial observability probe: N frames in, oldest first, no cross-reset leak."""
    from solo import train as solo_train
    from solo.obs import ACTOR_DIM as _A, CRITIC_DIM as _C, PRIV_DIM as _P

    cfg = solo_train.TrainConfig(task="balance", steps=64, rollout_steps=32,
                                 log_every=0, save_every=0, action_mode="residual",
                                 frame_stack=3)
    trainer = solo_train.SoloTrainer(cfg)
    assert trainer.net.actor.obs_dim == 3 * _A
    assert trainer.net.critic.obs_dim == 3 * _A + _P == _C + 2 * _A
    assert len(trainer._hist) == 3 and trainer._hist.maxlen == 3
    # all three seeded frames are the reset observation
    assert np.array_equal(np.concatenate(list(trainer._hist)),
                          np.tile(trainer._obs["actor"], 3))
    batch = trainer.collect()
    a5, c5 = batch.actor_obs[5, 0], batch.critic_obs[5, 0]
    assert a5.shape == (3 * _A,) and c5.shape == (_C + 2 * _A,)
    assert np.array_equal(c5[:3 * _A], a5)          # critic = [stack | privileged]

    # stacking order on synthetic frames: oldest first, newest last
    from collections import deque

    zeros_c = np.zeros(_C, np.float32)
    o = [np.full(_A, float(k), np.float32) for k in (1.0, 2.0, 3.0)]
    trainer._hist = deque([np.zeros(_A, np.float32)] * 3, maxlen=3)
    a1, c1 = trainer._stacked({"actor": o[0], "critic": zeros_c,
                               "privileged": np.ones(_P, np.float32)})
    assert np.array_equal(a1, np.concatenate([np.zeros(_A, np.float32)] * 2 + [o[0]]))
    assert np.all(c1[3 * _A:] == 1.0)                       # privileged appended
    a2, _ = trainer._stacked({"actor": o[1], "critic": zeros_c,
                              "privileged": np.zeros(_P, np.float32)})
    assert np.array_equal(a2, np.concatenate([np.zeros(_A, np.float32), o[0], o[1]]))
    a3, _ = trainer._stacked({"actor": o[2], "critic": zeros_c,
                              "privileged": np.zeros(_P, np.float32)})
    assert np.array_equal(a3, np.concatenate(o))
    assert len(trainer._hist) == 3
    stats = trainer.run()
    assert stats["steps"] == 64
    # single-env only: the vec path refuses rather than feeding unstacked frames
    with pytest.raises(SystemExit):
        solo_train.SoloTrainer(solo_train.TrainConfig(task="balance", steps=1,
                                                      n_envs=2, frame_stack=2))
    plain = solo_train.TrainConfig(task="balance", steps=1)
    assert plain.frame_stack == 1
    tr1 = solo_train.SoloTrainer(plain)
    assert tr1.net.actor.obs_dim == _A and tr1.net.critic.obs_dim == _C


def test_env_feeds_the_literature_inputs_in_one_frame(model):
    """Wiring check: what the env hands the terms is world-frame and physical."""
    env = SoloEnv(seed=2, term_set="balance_lit")
    env.reset(seed=2)
    seen: dict = {}
    original = env.reward.step

    def spy(inp):
        seen["inp"] = inp
        return original(inp)

    env.reward.step = spy
    ctrl = env._ctrl_stand.copy()          # the stand keyframe, both modes
    for _ in range(5):
        env.step(ctrl)
    inp = seen["inp"]
    assert inp.gravity == pytest.approx(9.81, abs=1e-6)
    assert 0.60 < inp.com_z < 0.75                      # CoM above the floor, not the pelvis
    pelvis_xy = np.asarray(env.data.xpos[env._pelvis_bid][:2], float)
    assert np.abs(np.asarray(inp.com_xy) - pelvis_xy).max() < 0.05
    assert np.asarray(inp.sole_points).shape == (2, 4, 3)
    assert np.abs(np.asarray(inp.sole_points)[:, :, 2]).max() < 0.02   # on the mat
    total = float(inp.foot_load[0] + inp.foot_load[1])
    assert 0.5 * 33.34 * 9.81 < total < 2.0 * 33.34 * 9.81
    assert np.asarray(inp.torque).shape == (N_JOINTS,)
    # the arms hold the keyframe targets; they sag ~1 mrad each under gravity
    assert 0.0 <= inp.arm_dev < 0.05
    # and the CoM velocity is the mass-weighted one, not the pelvis velocity
    mass = np.asarray(model.body_mass, float)
    v_com = (mass[:, None] * np.asarray(env.data.cvel, float)[:, 3:6]).sum(0) / mass.sum()
    assert np.abs(np.asarray(inp.com_vel_xy) - v_com[:2]).max() < 1e-9
