"""Tests for the Phase-4 wrestling environment and back-to-mat detector.

Fast subset (runs in a few seconds): reset determinism/randomization, obs and
action contract, controller plumbing, the exchange loop (timeout draw, back
event, ambiguity -- both same-step and cross-step triggers -- OOB forfeit, match
clock) and detector unit semantics (persistence, knees/hands/sprawl negatives,
latch).

The calibration sweep itself lives in scripts/calibrate_backdet.py; here we
only pin the calibrated operating point to data/backdet_calibration.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco  # noqa: E402

from wrestling.backdet import (FLOOR_GEOM, ROBOTS, BackDetConfig,  # noqa: E402
                               BackFeatures, BackToMatDetector,
                               back_features, body_maps, confirmed_mask,
                               first_confirmed_index)
from wrestling.env import (AMBIGUITY_WINDOW_S, OBS_DIM, QPOS_SLICE,  # noqa: E402
                           START_ANGLE_JITTER_DEG, START_DISTANCE_JITTER,
                           START_JOINT_NOISE, STEP_DT, SUBSTEPS, ReferenceReplay,
                           StandHold, WrestlingEnv, keyframe_pair_qpos,
                           load_wrestling_model, obs_layout,
                           reference_trace, resolve_back_events)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    return load_wrestling_model()


def _run_until(env, seconds: float, tol_steps: int = 1):
    """Step the env for ~seconds of sim time; returns the last info."""
    n = int(round(seconds / STEP_DT)) + tol_steps
    info = None
    for _ in range(n):
        if env.match_over:
            break
        _, _, _, _, info = env.step()
    return info


def _run_exchange(env, max_steps: int = 2000):
    """Step until the current exchange ends or the match ends."""
    for _ in range(max_steps):
        obs, reward, terminated, truncated, info = env.step()
        if info["exchange_ended"] is not None or terminated:
            return info, reward
    raise AssertionError("exchange did not end within the step budget")


# ------------------------------------------------------------------ reset/obs


def test_reset_determinism(model):
    a = WrestlingEnv(model=model, seed=11)
    b = WrestlingEnv(model=model, seed=11)
    assert np.array_equal(a.data.qpos, b.data.qpos)
    assert np.array_equal(a.data.qpos, a.data.qpos.copy())
    a.reset()  # same stored seed
    assert np.array_equal(a.data.qpos, b.data.qpos)
    c = WrestlingEnv(model=model, seed=12)
    assert not np.array_equal(a.data.qpos, c.data.qpos)


def test_reset_randomization_ranges(model):
    stance_a, stance_b = reference_trace("STANCE")["qpos_a"][0], reference_trace("STANCE")["qpos_b"][0]
    d0 = float(np.linalg.norm(stance_a[0:2] - stance_b[0:2]))
    for seed in range(8):
        env = WrestlingEnv(model=model, seed=seed)
        qa = env.data.qpos[QPOS_SLICE["a"]]
        qb = env.data.qpos[QPOS_SLICE["b"]]
        dist = float(np.linalg.norm(qa[0:2] - qb[0:2]))
        assert abs(dist - d0) <= START_DISTANCE_JITTER + 1e-9
        # joints stay within the documented +- 0.03 rad of the stance pose
        for q, base in ((qa, stance_a), (qb, stance_b)):
            assert np.all(np.abs(q[7:36] - base[7:36]) <= START_JOINT_NOISE + 1e-9)


def test_reset_pose_override_and_yaw_jitter_bounds(model):
    tr = reference_trace("SPRAWL")
    env = WrestlingEnv(model=model, seed=0)
    env.reset(pose=(tr["qpos_a"][0], tr["qpos_b"][0]))
    assert np.allclose(env.data.qpos[QPOS_SLICE["a"]], tr["qpos_a"][0])
    assert np.allclose(env.data.qpos[QPOS_SLICE["b"]], tr["qpos_b"][0])
    # pair yaw jitter is bounded by +-10 deg
    base_axis = reference_trace("STANCE")["qpos_b"][0][0:2] - reference_trace("STANCE")["qpos_a"][0][0:2]
    base_angle = np.arctan2(base_axis[1], base_axis[0])
    for seed in range(6):
        env.reset(seed=seed)
        axis = env.data.qpos[QPOS_SLICE["b"]][0:2] - env.data.qpos[QPOS_SLICE["a"]][0:2]
        d = np.degrees(np.arctan2(axis[1], axis[0]) - base_angle)
        assert abs((d + 180) % 360 - 180) <= START_ANGLE_JITTER_DEG + 1e-6


def test_obs_and_action_contract(model):
    env = WrestlingEnv(model=model, seed=5)
    obs = env.reset()
    assert env.action_dim == 58
    for o in obs:
        assert o.shape == (OBS_DIM,) and o.dtype == np.float32
    blocks = obs_layout()
    assert [name for name, _ in blocks][-1] == "opp_rel_heading"
    assert blocks[-1][1][1] == OBS_DIM
    assert np.all(np.isfinite(obs[0])) and np.all(np.isfinite(obs[1]))
    # observations follow obs_fn (swappable)
    env.obs_fn = lambda m, d: (np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
    o1, o2 = env.reset()
    assert o1.shape == (3,) and np.allclose(o2, 1.0)
    # action clipping to ctrlrange
    env = WrestlingEnv(model=model, seed=5)
    env.step(np.full((2, 29), 1e6))
    ctrl = env.data.ctrl.copy()
    lo = model.actuator_ctrlrange[:58, 0]
    hi = model.actuator_ctrlrange[:58, 1]
    assert np.all(ctrl <= hi + 1e-9) and np.all(ctrl >= lo - 1e-9)
    for shape in [(58,), (2, 29)]:
        env.step(np.zeros(shape))
    with pytest.raises(ValueError):
        env.step(np.zeros(7))


def test_reference_replay_sampling(model):
    rep = ReferenceReplay("STANCE", "a")
    env = WrestlingEnv(model=model, seed=1)
    data = env.data
    q0 = rep(env, data)
    assert np.allclose(q0, reference_trace("STANCE")["qpos_a"][0, 7:36])
    env.exchange_start = env.time - 1e9  # far past the end -> hold last frame
    qend = rep(env, data)
    assert np.allclose(qend, reference_trace("STANCE")["qpos_a"][-1, 7:36])
    loop = ReferenceReplay("STANCE", "b", loop=True)
    plain = ReferenceReplay("STANCE", "b")
    env.exchange_start = env.time - (loop.duration + 0.1)
    wrapped = loop(env, data)
    env.exchange_start = env.time - 0.1
    assert np.allclose(wrapped, plain(env, data))


# ---------------------------------------------------------------- exchange loop


def test_exchange_timeout_draw(model):
    """STANCE-hold both -> timeout draw, no false back-detector trigger.

    Measured caveat (see reports/2026-10-08/wrestling_env.md): the retargeted
    STANCE pose is not statically balanced under pure joint-PD replay -- it
    topples backwards after ~1.4 s -- so this test uses a 1.0 s exchange clock
    during which both robots are verified still standing.  The 20 s clock is
    exercised by test_stand_hold_long_timeout_draw.
    """
    env = WrestlingEnv(
        model=model, seed=3, exchange_timeout=1.0, match_clock=3.0,
        controllers=(ReferenceReplay("STANCE", "a"), ReferenceReplay("STANCE", "b")))
    info, _ = _run_exchange(env)
    rec = env.exchange_log[0]
    assert rec.cause == "timeout" and rec.winner is None and not rec.ambiguous
    assert rec.duration == pytest.approx(1.0, abs=STEP_DT)
    status = env.back_snapshot()
    assert status["a"]["trigger_t"] is None and status["b"]["trigger_t"] is None
    for robot in ROBOTS:
        z = env.data.xpos[env._pelvis_bid[robot]][2]
        assert z > 0.55, f"robot {robot} should still be standing at the timeout"


def test_stand_hold_long_timeout_draw(model):
    """Full-length exchange clock (20 s default) with the verified-stable stand
    pose: draw, robots standing, detector silent, no OOB."""
    qa, qb = keyframe_pair_qpos(model, "both_stand")
    env = WrestlingEnv(model=model, seed=4, match_clock=21.0,
                       controllers=(StandHold("a"), StandHold("b")))
    env.reset(seed=4, pose=(qa, qb))
    info, _ = _run_exchange(env)
    rec = env.exchange_log[0]
    assert rec.cause == "timeout" and rec.winner is None
    assert rec.duration == pytest.approx(20.0, abs=STEP_DT)
    assert rec.oob_events == {"a": 0, "b": 0}
    assert rec.back_triggers == {"a": None, "b": None}
    for robot in ROBOTS:
        end = rec.end_back[robot]
        assert end["trigger_t"] is None
        assert end["dorsal"] is False
        assert end["pelvis_z"] > 0.7


def test_sprawl_replay_back_event_scores(model):
    """SPRAWL PD-replay: the defender's prone end IS detected as back contact
    after the confirmation period; the attacker wins the exchange."""
    tr = reference_trace("SPRAWL")
    env = WrestlingEnv(
        model=model, seed=0, exchange_timeout=20.0, match_clock=30.0,
        controllers=(ReferenceReplay("SPRAWL", "a"), ReferenceReplay("SPRAWL", "b")))
    env.reset(seed=0, pose=(tr["qpos_a"][0], tr["qpos_b"][0]))
    info, reward = _run_exchange(env)
    rec = env.exchange_log[0]
    assert rec.cause == "back"
    assert rec.winner == "a" and rec.loser == "b" and not rec.ambiguous
    # the defender (b) triggered; the confirmation period elapsed after the
    # first dorsal contact (calibrated: 0.30 s within a 0.02 s control step)
    t_b = rec.back_triggers["b"]
    assert t_b is not None and rec.back_triggers["a"] is None
    assert t_b >= env.backdet.cfg.confirm_s
    # the exchange is held one ambiguity window past the (only) trigger so a
    # near-simultaneous counterpart trigger could still be observed
    assert rec.duration == pytest.approx(t_b + AMBIGUITY_WINDOW_S, abs=STEP_DT)
    assert rec.end_back["b"]["dorsal"] is True
    assert reward == pytest.approx((1.0, -1.0))
    assert 1.0 < rec.duration < 5.0  # terminates long before the 20 s timeout


_SETTLED: dict = {}


def _sprawl_settled_supine(model):
    """The settled end state of the SPRAWL PD-replay (defender on its back).

    Running the 5.7 s replay once is the deterministic way to obtain a physically
    settled supine pose (the kinematic final reference frame is face-down).
    """
    if "qpos" not in _SETTLED:
        tr = reference_trace("SPRAWL")
        qa, qb, t_ref = tr["qpos_a"], tr["qpos_b"], tr["t"]
        data = mujoco.MjData(model)
        data.qpos[QPOS_SLICE["a"]] = qa[0]
        data.qpos[QPOS_SLICE["b"]] = qb[0]
        mujoco.mj_forward(model, data)

        def ref_at(t, q):
            i = int(np.clip(np.searchsorted(t_ref, t), 1, len(t_ref) - 1))
            w = (t - t_ref[i - 1]) / (t_ref[i] - t_ref[i - 1])
            return q[i - 1, 7:36] * (1 - w) + q[i, 7:36] * w

        for step in range(int((float(t_ref[-1]) + 0.5) / model.opt.timestep)):
            t = min(step * model.opt.timestep, float(t_ref[-1]))
            data.ctrl[:29] = ref_at(t, qa)
            data.ctrl[29:58] = ref_at(t, qb)
            mujoco.mj_step(model, data)
        _SETTLED["qpos"] = data.qpos[QPOS_SLICE["b"]].copy()
    return _SETTLED["qpos"]


def test_simultaneous_falls_ambiguous(model):
    """Both wrestlers already supine on the mat -> both detectors confirm in the
    same control step -> ambiguous, no score, logged (MISSION Phase 3)."""
    q_b = _sprawl_settled_supine(model).copy()
    q_a = q_b.copy()
    q_a[0] += 1.8                          # same pose, translated (identical dynamics)
    env = WrestlingEnv(model=model, seed=0, match_clock=4.0)
    env.reset(seed=0, pose=(q_a, q_b))
    info, reward = _run_exchange(env)
    rec = env.exchange_log[0]
    assert rec.cause == "back" and rec.ambiguous is True
    assert rec.winner is None and rec.loser is None
    assert reward == pytest.approx((0.0, 0.0))
    assert rec.back_triggers["a"] is not None and rec.back_triggers["b"] is not None
    assert abs(rec.back_triggers["a"] - rec.back_triggers["b"]) <= 0.10
    assert rec.end_back["a"]["dorsal"] and rec.end_back["b"]["dorsal"]


def _raised_supine_pair(model, dz: float):
    """(q_a, q_b): the settled supine pose 1.8 m apart, with robot ``a`` raised
    ``dz`` above the mat so its back lands (and triggers) after ``b``'s."""
    q_b = _sprawl_settled_supine(model).copy()
    q_a = q_b.copy()
    q_a[0] += 1.8                    # same pose translated -> identical dynamics
    q_a[2] += dz
    return q_a, q_b


def _detector_trigger_times(model, q_a, q_b, max_steps: int = 80):
    """Detector-only control run (no env): when would each back trigger fire?

    Steps the same physics and control (joint targets = the start pose) at the
    env's 50 Hz control rate and samples the detector, so the returned times are
    directly comparable to the env's recorded ``back_triggers``.
    """
    data = mujoco.MjData(model)
    data.qpos[QPOS_SLICE["a"]] = q_a
    data.qpos[QPOS_SLICE["b"]] = q_b
    data.ctrl[:29] = q_a[7:36]
    data.ctrl[29:58] = q_b[7:36]
    mujoco.mj_forward(model, data)
    det = BackToMatDetector()
    for _ in range(max_steps):
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        det.update(model, data, float(data.time))
    return dict(det.triggers)


def test_cross_step_back_triggers_ambiguous(model):
    """Two authentic back triggers 0.04-0.08 s apart (3 steps) are ambiguous.

    The first trigger is at ``t_b``; robot ``a`` lands ~0.06 s later.  Before
    the ambiguity hold the exchange ended at ``t_b`` and ``a`` was never
    observed (winner "a", no ambiguity); now the exchange stays open for the
    window, so both triggers land in the record and the exchange is ambiguous.
    """
    q_a, q_b = _raised_supine_pair(model, 0.02)
    env = WrestlingEnv(model=model, seed=0, match_clock=10.0)
    env.reset(seed=0, pose=(q_a, q_b))
    info, reward = _run_exchange(env)
    rec = env.exchange_log[0]
    t_a, t_b = rec.back_triggers["a"], rec.back_triggers["b"]
    assert t_a is not None and t_b is not None, "second trigger was not held for"
    gap = t_a - t_b
    assert STEP_DT < gap <= AMBIGUITY_WINDOW_S, f"cross-step gap {gap} outside (step, window]"
    assert gap == pytest.approx(0.06, abs=STEP_DT)
    assert rec.cause == "back" and rec.ambiguous is True
    assert rec.winner is None and rec.loser is None
    assert reward == pytest.approx((0.0, 0.0))
    assert rec.duration == pytest.approx(t_a, abs=1e-6)   # ended at the SECOND trigger
    assert rec.end_back["a"]["dorsal"] and rec.end_back["b"]["dorsal"]
    assert info["exchange_ended"]["ambiguous"] is True and not info["match_over"]
    # the hold is deterministic: the same scenario resolves identically
    env2 = WrestlingEnv(model=model, seed=0, match_clock=10.0)
    env2.reset(seed=0, pose=(q_a, q_b))
    _run_exchange(env2)
    assert env2.exchange_log[0].as_dict() == rec.as_dict()


def test_second_trigger_outside_window_yields_first_trigger_winner(model):
    """The hold is bounded: a back trigger landing outside the window is never
    seen and the first trigger alone decides the exchange."""
    q_a, q_b = _raised_supine_pair(model, 0.20)
    env = WrestlingEnv(model=model, seed=0, match_clock=10.0)
    env.reset(seed=0, pose=(q_a, q_b))
    info, reward = _run_exchange(env)
    rec = env.exchange_log[0]
    t_b = rec.back_triggers["b"]
    assert t_b is not None and rec.back_triggers["a"] is None
    assert rec.cause == "back" and rec.ambiguous is False
    assert rec.winner == "a" and rec.loser == "b"      # only b's back was taken
    assert reward == pytest.approx((1.0, -1.0))
    # bounded hold: resolution one ambiguity window after the first trigger
    assert rec.duration == pytest.approx(t_b + AMBIGUITY_WINDOW_S, abs=1e-6)
    # premise, measured by the detector alone: a's back lands 0.20 s after b's,
    # i.e. after the exchange has already been resolved
    det = _detector_trigger_times(model, q_a, q_b)
    assert det["b"] == pytest.approx(t_b, abs=STEP_DT)
    assert det["a"] - det["b"] == pytest.approx(0.20, abs=STEP_DT)
    assert det["a"] > AMBIGUITY_WINDOW_S and det["a"] > rec.duration


def test_oob_forfeit(model):
    """Three out-of-bounds events in one exchange forfeit it (anti-run-away)."""
    pattern = [True, False, True, False, True]  # out, in, out, in, out
    counter = {"i": 0}

    def teleporting_ctrl(env, data):
        i = counter["i"]
        out = pattern[i] if i < len(pattern) else False
        data.qpos[0] = 2.0 if out else -0.35   # robot a: outside / inside the mat
        data.qpos[2] = 0.79
        counter["i"] = i + 1
        return data.qpos[QPOS_SLICE["a"]][7:36]

    env = WrestlingEnv(model=model, seed=1, match_clock=10.0)
    env.set_controllers(teleporting_ctrl, StandHold("b"))
    info, reward = _run_exchange(env, max_steps=40)
    rec = env.exchange_log[0]
    assert rec.cause == "oob"
    assert rec.winner == "b" and rec.loser == "a"
    assert rec.oob_events == {"a": 3, "b": 0}
    # final step: 3rd OOB penalty (-0.25) plus the exchange loss (-1) for a,
    # +1 for the winner b
    assert reward == pytest.approx((-1.25, 1.0))


def test_match_clock_accounting(model):
    """Short clocks: exchanges repeat until the match clock expires; the final
    record is a match_end with no score and the env reports terminated."""
    env = WrestlingEnv(model=model, seed=2, exchange_timeout=0.5, match_clock=2.2)
    steps = 0
    last = None
    while not env.match_over:
        obs, reward, terminated, truncated, info = env.step()
        steps += 1
        last = info
        if terminated:
            assert truncated is False
    assert env.time == pytest.approx(2.2, abs=STEP_DT)
    causes = [r.cause for r in env.exchange_log]
    assert causes[:-1] == ["timeout"] * (len(causes) - 1)
    assert causes[-1] == "match_end"
    assert env.exchange_log[-1].winner is None
    assert last["match_over"] is True
    assert len(env.exchange_log) == 5  # 0.5 s draws + the final partial exchange
    # deterministic re-run
    env2 = WrestlingEnv(model=model, seed=2, exchange_timeout=0.5, match_clock=2.2)
    while not env2.match_over:
        env2.step()
    assert [r.as_dict() for r in env2.exchange_log] == [r.as_dict() for r in env.exchange_log]


# -------------------------------------------------------------- detector units


def _feat(t=0.0, dorsal=False, front=False, tilt=0.0, pelz=0.8, limb=False, n=0):
    return BackFeatures(t=t, dorsal_contact=dorsal, front_contact=front,
                        contact_count=n, contact_x_min=-0.05 if dorsal else np.nan,
                        contact_x_max=0.05 if front else np.nan, tilt_deg=tilt,
                        pelvis_z=pelz, limb_contact=limb)


def test_backdet_persistence_and_negatives():
    cfg = BackDetConfig()
    det = BackToMatDetector(cfg)
    # knees/hands + front contact + upright: never a trigger
    for i in range(50):
        new = det.update_features({r: _feat(t=i * STEP_DT, front=True, tilt=80.0,
                                            pelz=0.3, limb=True, n=3) for r in ROBOTS},
                                  i * STEP_DT)
        assert not new
    # brief dorsal contact shorter than the confirmation period: no trigger
    det.reset()
    for i in range(5):
        assert not det.update_features(
            {r: _feat(t=i * STEP_DT, dorsal=True, tilt=80.0, pelz=0.3, n=2) for r in ROBOTS},
            i * STEP_DT)
    # sustained dorsal contact: triggers exactly once, after the confirmation
    det.reset()
    t0 = 10.0
    fired = []
    for i in range(40):
        t = t0 + i * STEP_DT
        new = det.update_features(
            {r: _feat(t=t, dorsal=True, tilt=80.0, pelz=0.3, n=2) for r in ROBOTS}, t)
        if new:
            fired.append((t, tuple(sorted(new))))
    assert fired == [(pytest.approx(t0 + cfg.confirm_s), ("a", "b"))], fired
    # latched: no re-trigger while the state continues
    assert det.triggers["a"] == pytest.approx(t0 + cfg.confirm_s)
    # a robot that leaves the state before confirmation does not trigger
    det.reset()
    for i in range(int(cfg.confirm_s / STEP_DT) - 1):
        det.update_features({r: _feat(t=i * STEP_DT, dorsal=True, tilt=80.0, pelz=0.3)
                             for r in ROBOTS}, i * STEP_DT)
    det.update_features({r: _feat(t=1.0, dorsal=False, tilt=10.0, pelz=0.8) for r in ROBOTS}, 1.0)
    assert det.triggers == {"a": None, "b": None}


def test_backdet_limb_contact_side_infixed_bodies(model):
    """limb_contact diagnostic must see the scene's real limb bodies.

    Regression: limb ids were built by prefix concatenation (``a_knee_link``),
    but the scene infixes left/right (``a_left_knee_link`` …), so ``limb_bids``
    was always empty and ``limb_contact`` always False even with limbs on the
    mat. The flag is diagnostic-only: it never gates the back-to-mat rule.
    """
    from wrestling.backdet import LIMB_BODIES
    for robot in ROBOTS:
        limb_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
                      for b in body_maps(model, robot).limb_bids}
        assert limb_names, robot
        assert all(n.startswith(f"{robot}_") for n in limb_names)
        assert all(n.endswith(LIMB_BODIES) for n in limb_names)
        # both sides of every limb resolved (knee + both wrists + elbow, L/R)
        assert len(limb_names) == 2 * len(LIMB_BODIES)

    def floor_bodies(data):
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
        out = set()
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 == floor or g2 == floor:
                other = g2 if g1 == floor else g1
                out.add(mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other])))
        return out

    def frame_data(trace, i):
        data = mujoco.MjData(model)
        data.qpos[QPOS_SLICE["a"]] = trace["qpos_a"][i]
        data.qpos[QPOS_SLICE["b"]] = trace["qpos_b"][i]
        mujoco.mj_forward(model, data)
        return data

    # positive: STAND_UP ground-start frame 6 (measured: ncon=40, 23 floor
    # contacts incl. a_left_wrist_yaw_link and b_left_knee_link)
    data = frame_data(reference_trace("STAND_UP"), 6)
    on_mat = floor_bodies(data)
    assert {"a_left_wrist_yaw_link", "b_left_knee_link"} <= on_mat
    det = BackToMatDetector(BackDetConfig())
    for robot in ROBOTS:
        f = back_features(model, data, robot)
        assert f.limb_contact is True, robot
        # diagnostic-only: limb contact never satisfies the rule
        assert not det.condition(f), robot

    # negative: standing frame — an ankle may touch the mat, no limb does
    data = frame_data(reference_trace("STANCE"), 0)
    on_mat = floor_bodies(data)
    assert on_mat and not any(n.endswith(LIMB_BODIES) for n in on_mat)
    for robot in ROBOTS:
        f = back_features(model, data, robot)
        assert f.limb_contact is False, robot
        assert f.pelvis_z > 0.55, robot  # clearly standing


def test_backdet_batch_matches_streaming():
    """The vectorized calibration rule must reproduce the streaming detector."""
    rng = np.random.default_rng(0)
    n = 400
    t = np.arange(n) * STEP_DT
    feats = [_feat(t=t[i], dorsal=bool(rng.random() < 0.4),
                   tilt=float(rng.uniform(0, 130)), pelz=float(rng.uniform(0.05, 0.9)))
             for i in range(n)]
    for cfg in (BackDetConfig(),
                BackDetConfig(tilt_threshold_deg=70, confirm_s=0.1),
                BackDetConfig(pelvis_z_threshold=0.2, confirm_s=0.5)):
        det = BackToMatDetector(cfg)
        streaming = []
        for f in feats:
            if det.update_features({"a": f, "b": f}, f.t).get("a"):
                streaming.append(f.t)
        cond = np.array([det.condition(f) for f in feats])
        idx = first_confirmed_index(cond, t, cfg.confirm_s)
        batch = [] if idx is None else [t[idx]]
        assert streaming == pytest.approx(batch)
        # the batch mask marks the same state and is monotone after the trigger
        mask = confirmed_mask(cond, t, cfg.confirm_s)
        if idx is not None:
            assert mask[idx:].sum() >= 1


def test_resolve_back_events():
    assert resolve_back_events({"a": 1.0, "b": None}) == ("b", False)
    assert resolve_back_events({"a": None, "b": 2.0}) == ("a", False)
    assert resolve_back_events({"a": 1.0, "b": 1.05}) == (None, True)
    assert resolve_back_events({"a": 1.0, "b": 1.20}) == ("b", False)
    assert resolve_back_events({"a": None, "b": None}) == (None, False)


def test_calibration_json_matches_default_config():
    path = REPO / "data" / "backdet_calibration.json"
    payload = json.loads(path.read_text())
    assert payload["config"] == BackDetConfig().as_dict()
    assert payload["metrics"]["sens"] >= 0.95 and payload["metrics"]["spec"] >= 0.95
    assert payload["margins"]["tilt_margin_pos_deg"] > 0
    assert payload["margins"]["tilt_margin_neg_deg"] > 0
    assert payload["margins"]["pelz_margin_pos_m"] > 0
    assert payload["evidence"]["dorsal_normal_agreement"] >= 0.95


def test_sanity_scene_and_limits(model):
    """The env must use the committed two-robot scene contract."""
    assert model.nq == 72 and model.nu == 58
    assert abs(model.opt.timestep - 0.002) < 1e-12
    env = WrestlingEnv(model=model, seed=0)
    assert env.n_substeps == 10
    lo, hi = env._joint_limits[0]
    assert len(lo) == len(hi) == 29
    assert np.all(hi > lo)
    assert env.mat_radius == pytest.approx(1.5)
