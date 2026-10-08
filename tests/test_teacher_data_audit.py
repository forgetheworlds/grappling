"""Number-asserting tests for the teacher/data audit.

These tests exercise the measurement code in ``scripts/audit_teacher_data.py``
(loaded by path, the repo convention for script-side helpers) and pin the
numbers the audit report claims:

* the saturation / rate-limit detectors fire exactly at the limit and not
  below it;
* the support-envelope holdability classifier on a hand-constructed two-frame
  case (held stance -> fraction 1/2) and on real data (STANCE has 0/118
  holdable frames; the 7-reference aggregate is 6.92 %);
* the MediaPipe track (15.0 Hz, 19 all-NaN frames, hands noisier than planted
  ankles, planted-ankle jitter > 2 cm peak-to-peak);
* the drill envelope (20 N: no actuator saturation, no governor, CoM margin
  stays positive; 90 N: governor saturates at 1.0 while *no* actuator hits its
  ctrlrange -- the failure is support geometry);
* GrappleMap (1 mm coordinate grid, median 4 keyframes per edge).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402


def _load_audit():
    """Import scripts/audit_teacher_data.py by path (module-level guarded)."""
    spec = importlib.util.spec_from_file_location(
        "audit_teacher_data", REPO / "scripts" / "audit_teacher_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load_audit()


@pytest.fixture(scope="module")
def model():
    return audit.load_scene_model()


@pytest.fixture(scope="module")
def checker(model):
    return audit.HoldabilityChecker(model)


@pytest.fixture(scope="module")
def holdability(checker):
    return audit.holdability_audit(checker.model)


# ---------------------------------------------------------------- detectors

def test_saturation_detector_fires_at_limit_not_inside():
    lo = np.array([-1.0, 0.0])
    hi = np.array([1.0, 2.0])
    # exactly on the high limit -> True; 1e-6 inside -> False
    flags = audit.saturation_flags(np.array([1.0, 2.0 - 1e-6]), lo, hi)
    assert flags.tolist() == [True, False]
    # below the low limit / on it -> True
    assert audit.saturation_flags(np.array([-1.0, 0.0]), lo, hi).all()
    # comfortably inside -> False
    assert not audit.saturation_flags(
        np.array([0.999999, 1e-3]), lo, hi).any()


def test_rate_clip_detector_fires_at_exactly_max_rate():
    prev = np.zeros(3)
    dq = np.array([0.25, 0.125, -0.25])
    flags = audit.rate_clip_flags(prev, dq, max_rate=0.25)
    assert flags.tolist() == [True, False, True]
    # a 1e-6 overshoot is still the limiter (the applied value equals the cap)
    assert audit.rate_clip_flags(prev, np.array([0.25 + 1e-6, 0, 0]), 0.25)[0]


def test_oppose_fraction_hand_case():
    # rows: assist, oppose, excluded (residual below e_min)
    dq = np.array([[0.1, 0.0], [-0.1, 0.0], [0.1, 0.0]])
    e = np.array([[0.2, 0.0], [0.2, 0.0], [0.005, 0.0]])
    assert audit.oppose_fraction(dq, e, e_min=0.02) == pytest.approx(0.5)


# ------------------------------------------------------- holdability math

def test_point_in_hull_margin_hand_case():
    sq = np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]])
    assert audit.point_in_hull_margin(sq, np.array([0.05, 0.05])) == \
        pytest.approx(0.05, abs=1e-9)
    assert audit.point_in_hull_margin(sq, np.array([0.12, 0.05])) == \
        pytest.approx(-0.02, abs=1e-9)
    assert audit.point_in_hull_margin(sq, np.array([0.0, 0.05])) == \
        pytest.approx(0.0, abs=1e-9)


def test_segment_margin_hand_case():
    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert audit._segment_margin(pts, np.array([0.5, 0.0])) == \
        pytest.approx(0.0)
    assert audit._segment_margin(pts, np.array([0.5, 0.3])) == \
        pytest.approx(-0.3)
    # off the ends: distance to the nearest endpoint
    assert audit._segment_margin(pts, np.array([2.0, 0.0])) == \
        pytest.approx(-1.0)


def test_holdability_fraction_hand_constructed(checker, model):
    """Two-frame hand case: held stance (inside) + lifted base (airborne)."""
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "both_stand")
    q = np.array(model.key_qpos[kid], float)
    qa = np.stack([q[0:36], q[0:36].copy()])
    qa[1, 2] += 0.60                       # base lifted 0.60 m -> no contact
    qb = np.stack([q[36:72], q[36:72]])

    marg0, npts, _ = checker.frame(qa[0], qb[0], "a_")
    assert npts >= 3 and marg0 > audit.FEASIBLE_BAND

    res = checker.ref(qa, qb)["a_"]
    assert res["frames"] == 2
    assert (res["stable"], res["marginal"], res["infeasible"],
            res["airborne"]) == (1, 0, 0, 1)
    assert res["inside_frac"] == pytest.approx(0.5)
    # robot b is untouched by the lift (it is a different qpos slice)
    assert checker.ref(qa, qb)["b_"]["stable"] == 2


# ------------------------------------------------- reference data (real)

def test_stance_reference_has_no_holdable_frame(holdability):
    a = holdability["refs"]["STANCE"]["a_"]
    b = holdability["refs"]["STANCE"]["b_"]
    assert (a["frames"], a["stable"], a["marginal"], a["airborne"]) == \
        (118, 0, 0, 0)
    assert a["infeasible"] == 118
    assert b["inside_support"] == 0 and b["infeasible"] == 118


def test_reference_aggregate_is_six_point_nine_percent(holdability):
    agg = holdability["aggregate"]
    assert agg["robot_frames"] == 2514
    assert agg["inside_support"] == 174
    assert agg["inside_pct_of_all"] == pytest.approx(6.921, abs=0.01)


def test_published_class_counts_are_reproduced(holdability):
    """The audit's classifier reproduces the support-envelope counts exactly."""
    a = holdability["refs"]["DOUBLE_LEG"]["a_"]
    assert (a["stable"], a["marginal"], a["infeasible"], a["airborne"]) == \
        (4, 10, 127, 14)
    b = holdability["refs"]["BODY_LOCK"]["b_"]
    assert (b["stable"], b["marginal"], b["infeasible"], b["airborne"]) == \
        (0, 14, 118, 44)


def test_dynamic_references_are_over_80_percent_infeasible(holdability):
    """The '6/7 references fail' finding, per technique, as numbers."""
    for tech in ("SNAPDOWN", "SINGLE_LEG", "STAND_UP"):
        for r in holdability["refs"][tech].values():
            assert r["inside_frac"] <= 0.25
    # the two techniques the teacher can execute at all
    assert holdability["refs"]["SNAPDOWN"]["a_"]["inside_support"] == 0
    assert holdability["refs"]["SNAPDOWN"]["b_"]["inside_support"] == 0


def test_video_references_are_not_grounded(holdability):
    v = holdability["refs_video"]["stance_hold"]["holdability"]["a_"]
    assert v["frames"] == 221
    assert v["stable"] == 0
    assert v["inside_support"] <= 1        # measured 1 (one degenerate margin~0 frame)
    assert v["airborne"] / v["frames"] > 0.5      # measured 123/221
    assert holdability["refs_video"]["stance_hold"]["b_zero_frac"] == 1.0


# ------------------------------------------------------------- MediaPipe

def test_mediapipe_track_rate_and_quality():
    mp = audit.mediapipe_audit()
    assert mp["frames"] == 6401
    assert mp["fps_median_hz"] == pytest.approx(15.0, abs=0.05)
    assert mp["detected_frac"] > 0.99
    assert mp["nan_all_frames"] == 19
    j = mp["planted_jitter"]
    # hands are noisier than the planted feet; ankles do not sit still
    assert max(j["l_wrist"]["hf_p2p_mm"]) > max(j["l_ankle"]["hf_p2p_mm"])
    assert max(j["l_ankle"]["hf_p2p_mm"]) > 20.0     # > 2 cm pk-pk when planted
    assert mp["hip_mid_std_mm"][0] < 1.0             # hip-centred: no root motion


# ---------------------------------------------------------- drill traces

def test_drill_20n_recovery_never_saturates_or_governs():
    z = np.load(REPO / "data" / "drill" /
                "FINAL_L1_push_feasible_L1_seed0.npz", allow_pickle=True)
    a = audit.analyse_drill_trace(z)
    assert a["fall"] is False
    assert a["sat_frac_max"] == 0.0
    assert a["alpha_max"] == 0.0
    assert a["margin_min"] > 0.02
    assert a["push"]["label_force"] == pytest.approx(20.0)


def test_drill_90n_failure_is_geometry_not_saturation():
    z = np.load(REPO / "data" / "drill" /
                "FAIL_PUSH90_feasible_L1_seed0.npz", allow_pickle=True)
    a = audit.analyse_drill_trace(z)
    assert a["fall"] is True
    assert a["sat_frac_max"] == 0.0              # no position actuator at limit
    assert a["alpha_max"] == 1.0                 # posture governor fully engaged
    assert a["push"]["margin_min_in_window"] < -0.5
    assert a["push"]["ctrl_delta_max_in_window"] < 0.1   # no rate-limit spike


# ----------------------------------------------------------- GrappleMap

def test_grapplemap_keyframe_and_quantisation_facts():
    gm = audit.grapplemap_audit()
    assert gm["nodes"] == 725 and gm["edges"] == 1485
    assert gm["min_coordinate_step_m"] == pytest.approx(1e-3, abs=1e-12)
    assert gm["frac_coords_on_1mm_grid"] > 0.999
    assert gm["frames_per_edge"]["median"] == 4.0
    assert gm["has_timestamps"] is False


# ------------------------------------------------- teacher replay (short)

def test_teacher_targets_in_range_low_saturation(model):
    npz = np.load(REPO / "data" / "refs" / "SNAPDOWN.npz", allow_pickle=True)
    qa, qb, t = npz["qpos_a"], npz["qpos_b"], npz["t"]
    ctrl = audit.TeacherController(model, qa, qb, t, technique="SNAPDOWN")
    inst = audit.instrumented_rollout(model, ctrl, qa, qb, t, "SNAPDOWN", 0)
    assert inst["range_ok"]
    ra = inst["robots"]["a_"]
    assert ra["sat_tick_frac"] <= 0.05           # measured 0.0
    assert ra["rms_rad_mean"] < 0.15             # measured ~0.07 rad (4 deg)
