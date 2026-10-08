"""Solo drill tests: interface contract, geometry, gates, traps, evidence.

Run from the repo root: ``.venv/bin/python -m pytest tests/test_drill.py -q``.

The tests are deliberately about the *contract* and the *measured traps* this
milestone hit (each one a real failure observed on this host):

* a Jacobian taken after ``mj_kinematics`` alone is wrong (``mj_comPos`` is
  required) -- the leg IK silently produced garbage references;
* an IK seeded from a straight leg walks into the hyperextension limit;
* the plan mixes foot *frame* origins with footprint centres (a constant 3.5 cm
  drag, and an entry that walks forward chasing its own mismatch);
* the swing foot must track the world trajectory with its own leg's joints, or
  a body drift flings it;
* ``cfrc_ext`` is torque-first, so the vertical *load* is index 5, not 2;
* a stale ``xfrc_applied`` changes the whole run: it must be zeroed every step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402

from drill import balance, controller, kin, lock, metrics, posture, scene, scheduler, video  # noqa: E402


@pytest.fixture(scope="module")
def built():
    model = scene.load_model()
    ids = kin.RobotIds.build(model)
    stance = posture.build_stance(model, posture.StanceSpec(), ids)
    return model, ids, stance


# ------------------------------------------------------------------ contract
def test_scene_is_single_robot(built):
    model, ids, _ = built
    assert model.nq == 36 and model.nu == 29
    assert model.nsite == 31                       # 23 model sites + 8 sole sites
    assert abs(model.opt.timestep - 0.002) < 1e-12
    for need in ("core", "left_toe", "left_heel", "right_toe", "right_heel",
                 "left_knee", "left_wrist", "head", "left_sole3"):
        assert need in ids.site


def test_controllers_meet_the_protocol(built):
    model, ids, stance = built
    data = mujoco.MjData(model)
    data.qpos[:] = stance.qpos
    data.ctrl[:] = np.clip(data.qpos[7:36], ids.ctrl_lo, ids.ctrl_hi)
    mujoco.mj_forward(model, data)
    for ctrl in (controller.StancePD(stance, ids),
                 controller.FeasibleDrill(stance, ids, rung="L0")):
        ctrl.reset(model, data)
        out = ctrl.act(model, data, controller.DrillCommand("STANCE"))
        assert out.shape == (29,)
        assert np.all(out >= ids.ctrl_lo - 1e-9) and np.all(out <= ids.ctrl_hi + 1e-9)
        assert np.all(np.isfinite(out))


def test_stance_rules_are_measured(built):
    """Operator rules: wide enough, rear leg back, knees bent, CoM inside."""
    _, _, stance = built
    r = stance.report
    assert r["width_m"] >= posture.StanceSpec().min_width - 1e-6
    # staggered, not square -- the bar follows the *measured* reference
    # (stance_spec.json: the operator's own depth is 0.062 m) and the A2
    # recalibration declared in reports/2026-10-08/drill.md §6.3
    assert r["base_depth_m"] > 0.22
    assert min(r["knee_flex_rad"].values()) > 0.15        # the crouch is real
    assert all(r["sole_flat"].values())
    assert r["com_margin_m"] > 0.02
    assert r["torso_tilt_deg"] < 20.0                    # leaning, not folded
    # the harness refuses a stance that breaks the width rule
    with pytest.raises(AssertionError):
        posture.build_stance(stance.model, posture.StanceSpec(half_width=0.08), stance.ids)


def test_foot_targets_put_the_lowest_point_on_the_mat(built):
    model, ids, _ = built
    for side in kin.SIDES:
        tgt = kin.foot_targets(ids, side, np.array([0.2, 0.1]), 0.2,
                               sole_z=kin.SOLE_REST_Z, pitch=0.15)
        assert abs(float(tgt[:, 2].min()) - kin.SOLE_REST_Z) < 1e-9
        assert tgt.shape == (4, 3)


# ------------------------------------------------------------------- the traps
def test_jacobian_needs_compos(built):
    """mj_kinematics alone leaves cdof stale: the site Jacobian is wrong."""
    model, ids, stance = built
    data = mujoco.MjData(model)
    data.qpos[:] = stance.qpos
    mujoco.mj_forward(model, data)
    sid = int(ids.sole_sites["left"][0])
    dofs = ids.leg_dofs["left"]
    qadr = ids.leg_qadr["left"]
    # evaluate at a configuration the last mj_forward did NOT see, so the
    # stale compact dofs would give a wrong Jacobian
    q = data.qpos[qadr].copy() + np.array([0.15, 0.05, 0.0, 0.25, -0.05, 0.03])
    jac = np.zeros((3, model.nv))

    def analytic(compos: bool):
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        if compos:
            mujoco.mj_comPos(model, data)
        mujoco.mj_jacSite(model, data, jac, None, sid)
        return jac[:, dofs].copy()

    numeric = np.zeros((3, 6))
    for k in range(6):
        for sign in (+1, -1):
            qq = q.copy()
            qq[k] += sign * 1e-6
            data.qpos[qadr] = qq
            mujoco.mj_kinematics(model, data)
            numeric[:, k] += sign * data.site_xpos[sid] / 2e-6
    assert np.abs(analytic(False) - numeric).max() > 0.05     # stale: wrong
    assert np.abs(analytic(True) - numeric).max() < 1e-6      # refreshed: right


def test_leg_ik_converges_and_never_diverges(built):
    model, ids, stance = built
    data = mujoco.MjData(model)
    data.qpos[:] = stance.qpos
    mujoco.mj_forward(model, data)
    side = "left"
    cur = ids.sole_points(data, side)
    # reachable: restore the current pose from a perturbed seed
    tgt = cur.copy()
    q0 = data.qpos[ids.leg_qadr[side]].copy()
    q1 = kin.leg_ik(model, data, ids, side, tgt, q0 + 0.25, iters=25)
    assert kin.ik_error(model, data, ids, side, tgt, q1) < 2e-3
    # unreachable: 0.5 m in front of the foot -- must stay bounded and not
    # walk into the joint limits (the monotone line search)
    far = tgt + np.array([0.5, 0.0, 0.0])
    q2 = kin.leg_ik(model, data, ids, side, far, q0, iters=25)
    lim = ids.leg_limits[side]
    assert np.all(q2 >= lim[:, 0] - 1e-3) and np.all(q2 <= lim[:, 1] + 1e-3)
    assert np.abs(q2 - q0).max() < 1.5


def test_foot_load_uses_the_force_half_of_the_wrench(built):
    """cfrc_ext is torque-first: the vertical load lives at index 5."""
    model, ids, stance = built
    data = mujoco.MjData(model)
    data.qpos[:] = stance.qpos
    data.ctrl[:] = np.clip(data.qpos[7:36], ids.ctrl_lo, ids.ctrl_hi)
    mujoco.mj_forward(model, data)
    for _ in range(200):
        mujoco.mj_step(model, data)
    load = ids.foot_load(data)
    weight = float(model.body_mass.sum() * 9.81)
    assert load.sum() == pytest.approx(weight, rel=0.25)
    assert abs(float(data.cfrc_ext[ids.foot_body["left"], 2])) < 0.5 * weight


# --------------------------------------------------------------------- gates
def test_rung_programmes_and_step_gating():
    rng = np.random.default_rng(0)
    for rung in ("L0", "L1", "L2", "L3", "L4"):
        prog = scheduler.program(rung, rng)
        assert prog, rung
        assert all(isinstance(el, scheduler.Element) for el in prog)
        if rung in ("L0", "L1"):
            # planted-feet rungs: no entry walk, no stepping elements
            assert "step" not in controller.RUNG_ELEMENTS[rung]
            assert all(el.skill not in ("ENTRY", "SHUFFLE_F", "CIRCLE_L")
                       for el in prog), rung
        else:
            assert prog[0].skill == "STANCE" and prog[1].skill == "ENTRY"


def test_timeout_does_not_advance_progress(built):
    model, ids, stance = built
    sch = scheduler.SkillScheduler("L0", seed=0)
    data = mujoco.MjData(model)
    data.qpos[:] = stance.qpos
    mujoco.mj_forward(model, data)
    ctrl = controller.FeasibleDrill(stance, ids, rung="L0")
    ctrl.reset(model, data)
    sch.reset(model, data)
    el = sch.elements[0]
    for k in range(int((el.timeout + 0.2) / 0.02)):
        sch.tick(ids, data, ctrl, k * 0.02)
    names = [e["event"] for e in sch.events]
    assert "element_timeout" in names
    done = [e for e in sch.events if e["event"] == "element_done"]
    assert done and done[0]["how"] == "timeout"


def test_step_report_counts_only_settled_landings():
    t = np.arange(0, 3, 0.02)
    n = len(t)
    trace = {"t": t, "qpos": np.tile([0, 0, 0.7, 1, 0, 0, 0], (n, 1)),
             "com": np.tile([0.0, 0.0, 0.65], (n, 1)),
             "sole_pts": np.zeros((n, 2, 4, 3)), "plan_planted": np.ones((n, 2), bool),
             "margin": np.ones(n)}
    trace["sole_pts"][:, :, :, 2] = kin.SOLE_REST_Z
    events = [{"event": "step_start", "side": "left", "t": 0.4},
              {"event": "step_done", "side": "left", "t": 1.0,
               "label": "t0", "landing_load_n": 120.0}]
    rep = metrics.step_report(trace, events)
    assert rep["n_completed"] == 1 and rep["n_started"] == 1
    slip = metrics.slip_report(trace)
    assert slip["max_load_drift_m"] == 0.0


def test_fall_detector_ignores_knees_and_fires_on_a_fall(built):
    model, ids, stance = built
    det = controller_mod_fall = __import__("drill.runner", fromlist=["FallDetector"]).FallDetector()
    data = mujoco.MjData(model)
    q = stance.qpos.copy()
    q[2] = 0.45                       # a deep crouch / knees near the mat is not a fall
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    assert not any(det.update(ids, data) for _ in range(20))
    q = stance.qpos.copy()
    q[2] = 0.30
    q[3:7] = kin.yaw_quat(0.0, roll=np.pi / 2)     # on its side
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    det2 = __import__("drill.runner", fromlist=["FallDetector"]).FallDetector()
    assert any(det2.update(ids, data) for _ in range(20))


# ------------------------------------------------------------ evidence & lock
def test_hud_reports_the_required_fields(built):
    model, ids, stance = built
    trace = {"t": np.array([0.0]), "qpos": np.array([stance.qpos]),
             "com": np.array([[0.0, 0.0, 0.65]]), "margin": np.array([0.04]),
             "tilt_deg": np.array([9.0]), "clearance": np.array([[0.0, 0.0]]),
             "com_ref": np.array([[0.0, 0.0]]), "e_track": np.array([[0.0, 0.0]]),
             "safety_alpha": np.array([0.0]), "push_force": np.array([0.0]),
             "knee_z": np.array([[0.38, 0.38]]), "foot_load": np.array([[160.0, 160.0]]),
             "contact": np.array([[True, True]]), "skill_id": np.array([0]),
             "cmd": np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
             "push": np.array([[0.0, 0.0, 0.0]]),
             "plan_planted": np.array([[True, True]])}
    frames = video.sample_frames(trace, fps=30, t1=0.0)
    lines = " | ".join(video._hud_lines(frames, 0,
                                        {"title": "t", "rung": "L1", "seed": 3,
                                         "controller_kind": "scripted feedback"}, ""))
    for field in ("pelvis z", "torso tilt", "CoM margin", "foot load",
                  "sole clearance", "rung L1", "seed 3", "skill STANCE", "cmd vx"):
        assert field in lines, field


def test_sim_lock_round_trip(tmp_path):
    p = tmp_path / "locks" / "sim.lock"
    with lock.sim_lock("test", path=p):
        assert "test" in lock.holder(p)
    assert lock.holder(p) == ""


def test_measured_constants_match_the_teacher():
    """The single-robot law adopts the teacher's measured numbers: drift fails."""
    got = balance.measured_gains()
    if not got:                                          # teacher absent: skip
        pytest.skip("src/teacher gains unavailable")
    p = balance.BalanceParams()
    for key in ("k_pitch", "k_roll", "k_anchor", "k_flat", "k_up", "kd", "k_z"):
        assert getattr(p, key) == pytest.approx(got[key]), key
    from teacher import stabilizers as st  # noqa
    for name, share in balance.PITCH_SPLIT.items():
        assert st.PITCH_SPLIT[name] == pytest.approx(share)
        assert st.AUTH[name][0] == pytest.approx(balance.AUTH_PITCH[name])
    assert st.AUTH["hip_roll"][1] == pytest.approx(balance.AUTH_ROLL["hip_roll"])
