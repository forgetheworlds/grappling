"""Number-asserting tests for the capture-point stability signal
(``src/solo/stability.py``).

The module is the M1 contract's missing half: it decides *why* the robot is about
to fall (the capture point leaving the support hull) and how far the support must
move to stop it.  These tests pin:

* the closed forms (``cp = x + v*sqrt(z/g)``, the step-length rule);
* the sign convention of the hull margin (positive inside);
* **the exploit-killer**: a crouched state with the CoM still inside the hull but
  the capture point outside is UNSAFE -- the case a CoM-inside test passes;
* that the DCM convergence test and the geometric predicate agree;
* the reward band's endpoints;
* the monitor against the real model at the stand keyframe.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solo.stability import (  # noqa: E402
    CAPTURABILITY_BAND_M, StabilityMonitor, brace_step_length, capture_point,
    capture_point_unsafe, capturability_reward, cp_time_constant, dcm_test,
    signed_margin)

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


@pytest.fixture(scope="module")
def model():
    from solo.scene import load_solo_model

    return load_solo_model()


def test_capture_point_closed_form_and_height_floor():
    tau = math.sqrt(0.9 / 9.81)
    assert cp_time_constant(0.9) == pytest.approx(tau, rel=1e-12)
    cp = capture_point([0.0, 0.0], [1.0, 0.0], 0.9)
    assert cp[0] == pytest.approx(tau, rel=1e-12) and cp[1] == pytest.approx(0.0)
    # zero velocity: the CP is the CoM projection
    assert np.allclose(capture_point([0.2, -0.3], [0.0, 0.0], 0.7), [0.2, -0.3])
    # the pendulum-length floor prevents 1/sqrt(z) blowing up on a fallen robot
    assert cp_time_constant(0.05) == cp_time_constant(0.30)
    assert cp_time_constant(0.05) == pytest.approx(math.sqrt(0.30 / 9.81), rel=1e-12)


def test_signed_margin_sign_convention():
    assert signed_margin([0.5, 0.5], SQUARE) == pytest.approx(0.5)
    assert signed_margin([1.5, 0.5], SQUARE) == pytest.approx(-0.5)
    assert signed_margin([1.0, 0.5], SQUARE) == pytest.approx(0.0, abs=1e-12)
    assert signed_margin([0.5, 0.5], np.zeros((0, 2))) == float("-inf")


def test_brace_step_length_is_the_missing_boundary_travel():
    # CP 1.0 m ahead of the CoM, 0.5 m of hull travel already available -> 0.5 m
    assert brace_step_length([0.5, 0.5], [1.5, 0.5], SQUARE) == pytest.approx(0.5)
    # CP inside the hull: no step required
    assert brace_step_length([0.5, 0.5], [0.6, 0.5], SQUARE) == pytest.approx(0.0)
    # the ray distance grows with the CP excursion, one for one
    assert brace_step_length([0.5, 0.5], [2.0, 0.5], SQUARE) == pytest.approx(1.0)


def test_crouch_with_com_inside_but_cp_outside_is_unsafe():
    """The exploit v1-v5 settled into: still inside the hull, already falling."""
    com = np.array([0.5, 0.5])
    z_c = 0.9
    still = capture_point(com, [0.0, 0.0], z_c)
    moving = capture_point(com, [3.0, 0.0], z_c)          # a fast crouch-drift
    assert signed_margin(still, SQUARE) > 0.0
    assert not capture_point_unsafe(still, SQUARE)
    # CoM still inside (a CoM-inside test would PASS this state) ...
    assert signed_margin(com, SQUARE) == pytest.approx(0.5)
    # ... but the capture point has left the hull -> UNSAFE
    assert signed_margin(moving, SQUARE) < 0.0
    assert capture_point_unsafe(moving, SQUARE)
    assert brace_step_length(com, moving, SQUARE) > 0.0


def test_dcm_convergence_agrees_with_the_geometric_predicate():
    z_c = 0.9
    for vel in (0.0, 0.5, 1.0, 1.65, 2.0, 3.0):
        com = np.array([0.5, 0.5])
        cp = capture_point(com, [vel, 0.0], z_c)
        t = dcm_test(com, [vel, 0.0], z_c, SQUARE)
        geometric = not capture_point_unsafe(cp, SQUARE)
        assert t["converging"] is geometric, (vel, t, cp)


def test_capturability_reward_band_endpoints():
    from solo.stability import StabilityState

    def state(margin):
        return StabilityState(margin_cp=margin)

    # prior_art §5.1: flat 1 while the CP is inside the support ...
    assert capturability_reward(state(0.05)) == 1.0
    assert capturability_reward(state(0.001)) == 1.0
    assert capturability_reward(state(0.0)) == 1.0
    # ... decaying to 0 once it is `band` outside
    assert capturability_reward(state(-0.025)) == pytest.approx(0.5)
    assert capturability_reward(state(-0.05)) == 0.0
    assert capturability_reward(state(-0.10)) == 0.0
    assert capturability_reward(state(float("nan"))) == 0.0
    with pytest.raises(ValueError):
        capturability_reward(state(0.01), band=0.0)


def test_monitor_on_the_stand_keyframe(model):
    import mujoco

    from solo.scene import stand_frame

    m = model
    d = mujoco.MjData(m)
    d.qpos[:] = stand_frame(m)[0]
    mujoco.mj_forward(m, d)
    st = StabilityMonitor(m).measure(d)
    assert bool(st.foot_touch.all())
    assert st.margin_com > 0.0 and st.margin_cp > 0.0
    assert st.unsafe is False and st.converging is True
    assert st.brace_step_length == 0.0
    assert st.reason.startswith("capture point +")
