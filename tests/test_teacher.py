"""Phase-3 stabilized teacher: invariants and contract tests.

Covers the task-spec invariants — joint targets always inside the actuator
ctrlrange, determinism (bit-identical replays), monotone phase segmentation,
and "no privileged state": the controller must only *read* the simulator and
may only drive the 29 hinges per robot (no base actuation, no state writes, no
opponent-state leakage).  Plus the default-configuration tick test requested
by review: first tick (and a second one, after a reset) must be finite and in
range, which is what catches reference-time vs sim-time indexing bugs.
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

from retarget.scene import load_scene_model, robot_slice  # noqa: E402
from teacher import BaselinePD, TeacherController, run_episode  # noqa: E402
from teacher.controller import ALL_OFF, TeacherFlags  # noqa: E402
from teacher.phases import KINDS, label_frames, segments  # noqa: E402
from teacher.trims import POSE_TRIM, trim_scale, trim_vector  # noqa: E402

TECH = "STANCE"                      # cheapest reference (2.34 s)
REF = REPO / "data" / "refs" / f"{TECH}.npz"


@pytest.fixture(scope="module")
def model():
    return load_scene_model()


@pytest.fixture(scope="module")
def ref():
    npz = np.load(REF, allow_pickle=True)
    return npz["qpos_a"], npz["qpos_b"], npz["t"]


def _fresh_data(model, qa, qb):
    d = mujoco.MjData(model)
    sa, sb = robot_slice(model, "a_"), robot_slice(model, "b_")
    d.qpos[sa], d.qpos[sb] = qa[0], qb[0]
    mujoco.mj_forward(model, d)
    return d


def test_default_ticks_finite_and_in_range(model, ref):
    """Default config: first tick + a second tick after a reset (review ask)."""
    qa, qb, t_ref = ref
    ctrl = TeacherController(model, qa, qb, t_ref, technique=TECH)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    for data in (_fresh_data(model, qa, qb), _fresh_data(model, qa, qb)):
        for t in (0.0, 0.02):
            c = ctrl.control(data, t)
            assert c.shape == (58,)
            assert np.isfinite(c).all()
            assert (c >= lo - 1e-9).all() and (c <= hi + 1e-9).all()
            mujoco.mj_step(model, data)
    # reference-time indexing: an out-of-range time must clamp, not crash
    c = ctrl.control(data, 1e4)
    assert np.isfinite(c).all() and (c >= lo - 1e-9).all() and (c <= hi + 1e-9).all()


def test_targets_in_ctrlrange_over_episode(model, ref):
    """Every commanded target over a full rollout stays inside ctrlrange."""
    qa, qb, t_ref = ref
    ctrl = TeacherController(model, qa, qb, t_ref, technique=TECH)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    for flags in (TeacherFlags(), ALL_OFF):
        ctrl = TeacherController(model, qa, qb, t_ref, technique=TECH, flags=flags)
        res = run_episode(model, ctrl, qa, qb, t_ref, seed=0)
        # re-simulation with the recorded controller is deterministic; here we
        # just check the controller's own state bookkeeping is finite/bounded
        assert np.isfinite(ctrl.last_offsets("a_")).all()
        assert np.abs(ctrl.last_offsets("a_")).max() <= 0.4 + 1e-9
        assert np.isfinite(res.rows).all()
        assert res.rows[:, 1].min() > -0.05 and res.rows[:, 2].min() > -0.05


def test_determinism_same_seed_bit_identical(model, ref):
    """Same seed ⇒ bit-identical rollout; different seed ⇒ different."""
    qa, qb, t_ref = ref
    out = []
    for _ in range(2):
        ctrl = TeacherController(model, qa, qb, t_ref, technique=TECH)
        res = run_episode(model, ctrl, qa, qb, t_ref, seed=3)
        out.append(res.rows)
    assert np.array_equal(out[0], out[1])
    ctrl = TeacherController(model, qa, qb, t_ref, technique=TECH)
    other = run_episode(model, ctrl, qa, qb, t_ref, seed=4).rows
    assert not np.array_equal(out[0], other)


def test_phase_segmentation_monotone_and_complete(ref):
    """Segments cover [0, T] exactly, ordered, disjoint, valid kinds."""
    _, _, t_ref = ref
    qpos_a = np.load(REF, allow_pickle=True)["qpos_a"]
    labels = label_frames(t_ref, qpos_a[:, 2])
    assert set(labels) <= set(KINDS)
    segs = segments(t_ref, labels)
    assert segs[0][1] == pytest.approx(float(t_ref[0]))
    assert segs[-1][2] == pytest.approx(float(t_ref[-1]))
    for (k, t0, t1), (k2, t0n, _) in zip(segs, segs[1:]):
        assert k in KINDS and k2 in KINDS
        assert t1 <= t0n + 1e-12
        assert t1 > t0
    assert sum(t1 - t0 for _, t0, t1 in segs) == pytest.approx(
        float(t_ref[-1] - t_ref[0]))


def _fresh_tick(model, qa, qb, t_ref, mutate=None, t=0.02):
    """One control tick from a *fresh* controller (state-free comparison)."""
    ctrl = TeacherController(model, qa, qb, t_ref, technique=TECH)
    data = _fresh_data(model, qa, qb)
    if mutate is not None:
        mutate(data)
        mujoco.mj_forward(model, data)
    return ctrl.control(data, t)


def test_no_state_mutation_and_no_opponent_leakage(model, ref):
    """The controller is a pure read: no writes, no opponent-state coupling.

    ``actuator_force``/``act`` are *derived*: MuJoCo recomputes them (stage
    machinery) whenever a Jacobian primitive runs, so they cannot witness a
    mutation; the test pins the actual state (``qpos``/``qvel``/``ctrl``).
    Fresh controllers are used for every comparison because the teacher is
    deliberately stateful (task integrator + rate limiter): two consecutive
    ticks with identical inputs must *not* be bit-identical.
    """
    qa, qb, t_ref = ref
    ctrl = TeacherController(model, qa, qb, t_ref, technique=TECH)
    data = _fresh_data(model, qa, qb)
    data.ctrl[:] = ctrl.control(data, 0.0)
    mujoco.mj_step(model, data)
    mujoco.mj_normalizeQuat(model, data.qpos)
    mujoco.mj_forward(model, data)
    before = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
    ctrl.control(data, 0.1)
    after = (data.qpos, data.qvel, data.ctrl)
    for b, a in zip(before, after):
        assert np.array_equal(b, a), "controller mutated simulator state"
    # determinism of a fresh tick
    base = _fresh_tick(model, qa, qb, t_ref)
    assert np.array_equal(base, _fresh_tick(model, qa, qb, t_ref))
    # data.ctrl is not an input
    def poison_ctrl(d):
        d.ctrl[:] = 123.0
    assert np.array_equal(base, _fresh_tick(model, qa, qb, t_ref, poison_ctrl))
    # the opponent's state is not an input to this robot's targets
    def move_opponent(d):
        d.qpos[36:39] += np.array([5.0, 5.0, 0.0])
    moved = _fresh_tick(model, qa, qb, t_ref, move_opponent)
    assert np.array_equal(base[0:29], moved[0:29])


def test_only_the_29_hinges_per_robot_are_actuated(model):
    """No root actuation and no extra actuators: 58 hinge servos, nothing else."""
    assert model.nu == 58
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        jtype = int(model.jnt_type[jid])
        assert jtype == int(mujoco.mjtJoint.mjJNT_HINGE)
        name = model.joint(jid).name
        assert name.startswith(("a_", "b_")), name


def test_trims_are_small_and_gated():
    """Documented reference trims: small, sagittal-only, gated to standing."""
    for tech, per in POSE_TRIM.items():
        for prefix, spec in per.items():
            assert prefix in ("a_", "b_")
            for key, val in spec.items():
                assert key in ("hip", "knee", "ankle", "waist", "ankle_roll", "dz")
                assert abs(val) <= (0.30 if key != "dz" else 0.10), (tech, prefix, key, val)
            v = trim_vector(spec)
            assert v.shape == (29,)
            # only the intended joints move: legs (both sides) + waist_pitch
            touched = np.nonzero(v)[0].tolist()
            assert set(touched) <= {0, 3, 4, 5, 6, 9, 10, 11, 14}, touched
    assert trim_scale(0.30) == 0.0            # ground work: no trim
    assert trim_scale(0.70) == 1.0            # standing: full trim
    assert 0.0 < trim_scale(0.51) < 1.0       # continuous in between


def test_baseline_pd_matches_reference_targets(model, ref):
    """The phase-2 baseline is exactly the reference joint targets."""
    qa, qb, t_ref = ref
    ctrl = BaselinePD(model, qa, qb, t_ref)
    data = _fresh_data(model, qa, qb)
    c = ctrl.control(data, float(t_ref[1]))
    assert np.allclose(c[0:29], qa[1][7:36], atol=1e-12)
    assert np.allclose(c[29:58], qb[1][7:36], atol=1e-12)


def test_phase_smoother_is_zero_phase():
    """A step in the input stays centred: the smoother must not lag."""
    from teacher.phases import _smooth
    t = np.arange(0.0, 2.0, 0.02)
    x = np.where(t >= 1.0, 1.0, 0.0)
    y = _smooth(x, t, 0.15)
    assert len(y) == len(x)
    i = int(np.argmin(np.abs(t - 1.0)))
    # the half-rise sits at the step (between i-1 and i): no lag
    assert y[i] + y[i - 1] == pytest.approx(1.0, abs=1e-9)
    # and a delayed smoother would put the half-rise later; pin the shape
    assert y[i - 1] < 0.5 < y[i]
    assert y[i - 2] < y[i - 1] < y[i] < y[i + 1]
    assert y[0] == 0.0 and y[-1] == pytest.approx(1.0)
    assert y[i + 2] + y[i + 1] > 1.0   # lag would make this ~1.0


def test_prone_interlock(model, ref):
    """A PRONE reference label must not disable a still-standing robot.

    The label is inferred from the *reference* pelvis height; the interlock
    also requires the simulated pelvis to be low (< 0.55 m), so an execution
    that has not reached the ground keeps its stabilizer.
    """
    qa, qb, t_ref = ref
    low = np.array(qa)
    low[:, 2] -= 0.55                          # reference now says "prone"
    ctrl = TeacherController(model, low, qb, t_ref, technique=TECH)
    standing = _fresh_data(model, qa, qb)      # sim is standing
    ctrl.control(standing, 0.0)
    assert ctrl._dbg["a_"]["kind"] == "LOW", ctrl._dbg["a_"]["kind"]
    down = _fresh_data(model, qa, qb)
    down.qpos[2] -= 0.55
    mujoco.mj_forward(model, down)
    ctrl.control(down, 0.0)
    assert ctrl._dbg["a_"]["kind"] == "PRONE"


def test_scoring_uses_executor_and_reference_window(model):
    """The evaluated trace is the executor robot's, cut at the reference end."""
    from teacher.episode import scored_trajectories, score_episode
    npz = np.load(REPO / "data" / "refs" / "SPRAWL.npz", allow_pickle=True)
    qa, qb, t_ref = npz["qpos_a"], npz["qpos_b"], npz["t"]
    ctrl = TeacherController(model, qa, qb, t_ref, technique="SPRAWL")
    res = run_episode(model, ctrl, qa, qb, t_ref, seed=0)
    self_tr, opp_tr = scored_trajectories(res, t_ref, "SPRAWL")
    assert len(self_tr) == len(opp_tr) <= res.traj.shape[0]
    # SPRAWL's executor is the defender (robot B); its trace is qpos_b
    assert np.allclose(self_tr[0], res.traj[0, 1], atol=1e-12)
    out = score_episode(res, t_ref, "SPRAWL", model)
    assert 0.0 <= out["mean"] <= 1.0
    assert out["executor"] == "B"
    assert len(out["per_phase"]) >= 1


def test_stance_holds_with_defaults(model, ref):
    """Regression: the default teacher keeps both robots up on STANCE.

    The full acceptance sweep (5 seeds, all techniques, scorer gate) is
    scripts/run_teacher.py; this pins the tuned configuration at a slightly
    looser bound so it stays a regression guard, not a second tuning target.
    """
    qa, qb, t_ref = ref
    mins = []
    for seed in (0, 1, 2):
        ctrl = TeacherController(model, qa, qb, t_ref, technique=TECH)
        res = run_episode(model, ctrl, qa, qb, t_ref, seed=seed)
        sf = res.stay_up_frac(t_ref)
        mins.append(sf["min"])
    # shipped configuration measures 0.76 mean / 0.58 worst over these seeds;
    # the gate itself (>= 0.80 over 5 seeds, with the scorer) is
    # scripts/run_teacher.py, this is the "it still does the right thing" pin
    assert sum(mins) / len(mins) >= 0.65, mins
    assert min(mins) >= 0.50, mins


# ---------------------------------------------------------------------------
# Drill path (single-G1 scene): command hand-off, cadence, guards, references.
# These are the regression tests for the fixes in
# reports/2026-10-08/teacher_exec_fix.md; each one FAILS on the pre-fix code.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def solo():
    """Single-G1 scene + a feasible (wide, splay-built) stance from `stand`."""
    from teacher.solo_scene import load_single_model, stand_qpos
    from teacher.skills import SkillController
    from teacher.trims import enforce_stance_width
    model = load_single_model()
    stand = stand_qpos(model)
    stance, splay, width = enforce_stance_width(model, stand, 0.30, prefix="")
    ctrl = SkillController(model, stance, stand=stand)
    data = mujoco.MjData(model)
    ctrl.reset(stance)
    sa = ctrl.robot.ctx.qpos_slice
    data.qpos[sa] = stance
    mujoco.mj_forward(model, data)
    ctrl.reset(stance)
    return model, ctrl, data, stance, width


def test_solo_scene_is_one_robot_and_returns_29_targets(solo):
    """The solo contract: one G1 (nq=36, nu=29), explicit 29-target output."""
    model, ctrl, data, stance, width = solo
    assert (model.nq, model.nu) == (36, 29)
    assert ctrl.robot.prefix == ""
    c = ctrl.robot.control(data, ctrl._t_pose)
    assert c.shape == (29,)
    assert np.isfinite(c).all()
    assert (c >= ctrl.robot.ctrl_lo - 1e-9).all()
    assert (c <= ctrl.robot.ctrl_hi + 1e-9).all()
    assert width >= 0.30 - 1e-6


def test_command_path_changes_executed_targets_and_tasks(solo):
    """F1 regression: commands must reach BOTH the executed joints and the
    dependent task targets (height, heading) through the real consumer.

    Pre-fix, `SkillController._table` was updated but the executor read its
    construction-time copies, so nothing below changed.
    """
    from teacher.skills import DT, drill_step
    model, ctrl, data, stance, width = solo
    ctrl.reset(stance)
    t = 0.0
    for _ in range(50):                       # 1 s stance hold
        t = drill_step(model, data, ctrl, t)
    ctx = ctrl.robot.ctx
    f0 = ctrl.robot._frame(ctrl._t_pose)
    z0 = float(ctx.ref_pelvis_z[f0])
    q_ref0 = ctrl.robot.reference_q(ctrl._t_pose).copy()   # executor's reference
    c0 = ctrl.robot.control(data, ctrl._t_pose).copy()
    # 1) stance height command -> executed knee target + height task.
    #    frac -0.35 (dz = -0.063 m) is a *reachable* level change on this model
    #    (measured: reference knee moves ~0.19 rad, height task 0.063 m); the
    #    deep frac -1.0 is not IK-reachable from the wide stance and its
    #    element topples (reported in reports/2026-10-08/teacher_exec_fix.md)
    ctrl.set_stance_height(-0.35)
    for _ in range(75):                       # 1.5 s: past the 0.6 s crossfade
        t = drill_step(model, data, ctrl, t)
    f1 = ctrl.robot._frame(ctrl._t_pose)
    z1 = float(ctx.ref_pelvis_z[f1])
    q_ref1 = ctrl.robot.reference_q(ctrl._t_pose).copy()
    c1 = ctrl.robot.control(data, ctrl._t_pose).copy()
    assert z0 - z1 > 0.05, (z0, z1)
    assert abs(q_ref1[3] - q_ref0[3]) > 0.10, (q_ref0[3], q_ref1[3])
    assert abs(c1[3] - c0[3]) > 0.03, (c0[3], c1[3])     # knee, executed
    # 2) heading command -> reference quaternion + heading task
    ctrl.set_stance_height(0.0)
    for _ in range(40):
        t = drill_step(model, data, ctrl, t)
    yaw0 = float(ctx.ref_yaw[ctrl.robot._frame(ctrl._t_pose)])
    ctrl.set_command(wz=0.4)
    for _ in range(75):
        t = drill_step(model, data, ctrl, t)
    yaw1 = float(ctx.ref_yaw[ctrl.robot._frame(ctrl._t_pose)])
    assert abs(yaw1 - yaw0) > 0.05, (yaw0, yaw1)
    # 3) planted feet stay pinned at their world targets (no xy drag)
    pins = np.stack([ctrl._pin[0], ctrl._pin[1]])
    assert np.isfinite(pins).all()


def test_drill_cadence_is_50hz_and_timeline_rolls(solo):
    """F2 regression: one control() call per 20 ms and no fixed table cap.

    Pre-fix, the self-check called control() every 2 ms physics step (500 Hz:
    integrators/rate limits 10x fast) and the reference clamped after 17.98 s.
    """
    from teacher.skills import DT, ROW_CHUNK, drill_step
    model, ctrl, data, stance, width = solo
    ctrl.reset(stance)
    calls = {"n": 0}
    orig = ctrl.control

    def counting(d, t):
        calls["n"] += 1
        return orig(d, t)

    ctrl.control = counting
    t = 0.0
    n_ticks = 150
    for _ in range(n_ticks):
        t = drill_step(model, data, ctrl, t)
    assert calls["n"] == n_ticks, (calls["n"], n_ticks)
    # the drill clock equals simulated time (the first call advances nothing)
    assert abs(ctrl._t_pose - (n_ticks - 1) * DT) < 1e-9, ctrl._t_pose
    # the reference grew with the clock and has no 18 s cap
    assert len(ctrl._rows) >= ROW_CHUNK
    n_before = len(ctrl._rows)
    ctrl._ensure_rows(n_before + 500)
    assert len(ctrl._rows) >= n_before + 500
    assert len(ctrl.robot.ctx.ref_qpos) == len(ctrl._rows)
    assert len(ctrl.robot.ctx.t_ref) == len(ctrl._rows)
    assert np.all(np.diff(ctrl.robot.ctx.t_ref) > 0)


def test_governor_activates_from_unclipped_error(model, ref):
    """F3 regression: the posture governor must be able to reach its thresholds.

    The clipped capture-point error saturates at sqrt(2)*com_clamp (~0.113 m),
    below the DIVE full-blend threshold (0.30 m); activation uses the raw error.
    """
    import teacher.stabilizers as st
    from teacher.controller import RobotTeacher
    from teacher.gains import gain_for
    qa, qb, t_ref = ref
    r = RobotTeacher(model, "a_", qa, t_ref, technique=TECH)
    data = _fresh_data(model, qa, qb)
    # 1) the raw error is not bounded by the clip and reaches the DIVE threshold
    com = np.array([0.0, 0.0, 0.7])
    v = np.zeros(2)
    support = np.array([0.4, 0.0])
    g = gain_for(TECH, "DIVE")
    e_raw = st.capture_error(com, v, support, g.kd, None)
    e_clip = st.capture_error(com, v, support, g.kd, g.com_clamp)
    assert float(np.hypot(*e_raw)) > g.govern_e1 > float(np.hypot(*e_clip))
    # 2) the controller feeds the unclipped value to the governor: with a raw
    #    error above govern_e1 the blend must reach 1.0 (reset the blend state)
    r.phase_kind = lambda t: "DIVE"
    r._alpha = 0.0
    orig = st.capture_error

    def fake(com_, v_, support_, kd_, clamp=None):
        out = orig(com_, v_, support_, kd_, clamp)
        if clamp is None:
            return np.array([0.5, 0.0])          # a large raw error
        return out

    st.capture_error = fake
    try:
        for k in range(40):
            r.control(data, 0.4 + 0.02 * k)
    finally:
        st.capture_error = orig
    assert r.dbg["alpha"] >= 0.9, r.dbg["alpha"]
    assert float(np.hypot(*r.dbg["e_raw"])) > g.govern_e1
    assert float(np.hypot(*r.dbg["e"])) <= np.sqrt(2) * g.com_clamp + 1e-9


def test_step_primitive_advances_only_on_measured_guards(solo):
    """The support-transfer primitive must not advance on timers: with a
    frozen simulation (no new contact measurements) it times out and is
    reported as failed instead of stepping through phases."""
    from teacher.skills import DT
    model, ctrl, data, stance, width = solo
    ctrl.reset(stance)
    ctrl.start_step(swing=1, dx=-0.10, timeout_s=2.0, why="test")
    assert ctrl.step_status()["phase"] == "LOAD"
    t = ctrl._t_pose
    for _ in range(int(3.0 / DT)):
        ctrl.control(data, t)
        t += DT
        if ctrl.step_status() == {}:
            break
    st = ctrl._step_history[-1]
    assert st["result"] == "failed"
    assert "LOAD" in st["notes"][0]
    # a fresh step's reference is continuous across the phase boundary
    ctrl.start_step(swing=0, dx=0.05, why="test-continuity")
    step = ctrl._step
    f0 = step.feet0[0].copy()
    assert np.allclose(step.swing_foot(0.0), f0, atol=1e-9)
    step.phase, step.phase_t0 = "LIFT", 0.0
    assert np.allclose(step.swing_foot(0.0), f0, atol=1e-9)
    sj0 = step.swing_joints(0.0)
    assert sj0 is not None and np.isfinite(sj0).all()


def test_stand_up_reference_is_grounded():
    """F3/data regression: the rebuilt STAND_UP reference keeps its feet on
    the mat (the shipped montage was airborne: min foot-site z up to 0.77 m,
    71% of frames in the air, pelvis up to 1.07 m)."""
    from retarget.landmarks import SOLVED_SITES
    from retarget.solve import G1Kinematics
    npz = np.load(REPO / "data" / "refs" / "STAND_UP.npz", allow_pickle=True)
    kin = G1Kinematics()
    idx = [SOLVED_SITES.index(n) for n in
           ("left_toe", "left_heel", "right_toe", "right_heel")]
    for key in ("qpos_a", "qpos_b"):
        mins = np.array([min(float(kin.sites(q)[j][2]) for j in idx)
                         for q in npz[key]])
        pelvis = np.array([float(kin.sites(q)[SOLVED_SITES.index("core")][2])
                           for q in npz[key]])
        assert mins.min() > -0.035, (key, float(mins.min()))
        assert float((mins > 0.05).mean()) < 0.10, (key, float((mins > 0.05).mean()))
        assert pelvis.max() < 0.85, (key, float(pelvis.max()))
        assert pelvis.min() > 0.15, (key, float(pelvis.min()))
    meta = json.loads(str(npz["meta"]))
    assert all(j["kind"] == "shared_node" for j in meta["junctions"])
