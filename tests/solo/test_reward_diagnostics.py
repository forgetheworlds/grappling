"""Number-asserting tests for the solo reward/critic diagnostics.

These pin the *measurement machinery* used by ``scripts/solo_train_health.py``
and the reward's own arithmetic.  They are deliberately about numbers on
hand-constructed states, not about policy behaviour:

* explained variance of a perfect predictor is 1.0, of a constant predictor 0.0
  (and a constant *offset* does not cost explained variance);
* the per-term decomposition sums exactly to the total reward, including the
  one-off termination penalty (checked on a hand-made trace AND through the real
  ``SoloEnv.step`` path);
* each named balance term fires in the state it is defined for and is inert
  otherwise (``flat_orientation``, ``action_rate``, ``torque_sat``,
  ``joint_limit``, ``alive``);
* the keyframe-optimality grid is the documented shape.

The *misalignment* experiment (``solo_train_health.py ranked`` /
``keyframe``) is REPORT-ONLY and deliberately NOT asserted here: a failing
alignment check must not turn the suite red for peers.  Its numbers live in
``reports/2026-10-08/reward_critic_audit.md``.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solo.commands import Command  # noqa: E402
from solo.metrics import METRIC_FIELDS  # noqa: E402
from solo.reward import (RewardInputs, RewardWeights, TASK_TERMS,  # noqa: E402
                         TaskReward)
from solo.scene import N_JOINTS  # noqa: E402


def _health_module():
    """Load ``scripts/solo_train_health.py`` as a module (scripts/ is not a pkg)."""
    path = ROOT / "scripts" / "solo_train_health.py"
    spec = importlib.util.spec_from_file_location("solo_train_health", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses needs cls.__module__ present
    spec.loader.exec_module(mod)
    return mod


HEALTH = _health_module()


# --------------------------------------------------------------------------- #
# explained variance
# --------------------------------------------------------------------------- #
def test_explained_variance_perfect_predictor_is_one():
    targets = np.array([0.0, 1.0, 2.0, 10.0, -3.0])
    assert HEALTH.explained_variance(targets, targets) == pytest.approx(1.0, abs=1e-12)
    # a constant offset is NOT an error for a state-value baseline
    assert HEALTH.explained_variance(targets + 37.0, targets) == pytest.approx(1.0, abs=1e-12)
    # independent noise with a known variance scales EV analytically
    targets = np.linspace(-100.0, 200.0, 512)
    noise = np.linspace(-1.0, 1.0, 512)
    ev = HEALTH.explained_variance(targets + noise, targets)
    assert ev == pytest.approx(1.0 - np.var(noise) / np.var(targets), abs=1e-12)
    assert 0.0 < ev < 1.0


def test_explained_variance_constant_predictor_is_zero():
    targets = np.array([1.0, 2.0, 3.0, 4.0])
    assert HEALTH.explained_variance(np.full_like(targets, targets.mean()), targets) \
        == pytest.approx(0.0, abs=1e-12)
    # degenerate target: no variance to explain -> reported as 0.0, not NaN
    assert HEALTH.explained_variance(np.zeros(4), np.zeros(4)) == 0.0
    # worse than the mean is negative
    assert HEALTH.explained_variance(-targets, targets) < 0.0


# --------------------------------------------------------------------------- #
# per-term decomposition == total
# --------------------------------------------------------------------------- #
def _trace() -> list[RewardInputs]:
    good = RewardInputs(cmd=Command(vx=0.0), torso_up_z=1.0, pelvis_z=0.79,
                        action=np.zeros(N_JOINTS), prev_action=np.zeros(N_JOINTS))
    bad = replace(good, torso_up_z=0.1, pelvis_z=0.30, sat_frac=0.7, limit_prox=0.4,
                  action=np.full(N_JOINTS, 0.6))
    return [good, good, bad, replace(bad, torso_up_z=-0.5, pelvis_z=0.12)]


def test_score_trace_terms_sum_to_total_for_every_config():
    trace = _trace()
    for cfg in HEALTH.DEFAULT_CONFIGS:
        out = HEALTH.score_trace(cfg, trace)
        weighted_sum = float(sum(out["weighted"].values()))
        assert weighted_sum == pytest.approx(out["total"], abs=1e-9), cfg.name
        assert set(out["raw"]) <= set(TaskReward("balance",
                                                 term_set=cfg.term_set).terms) | {"termination"}
        # ... and the raw term values are the ones the reward itself reports
        tr = cfg.reward()
        manual = sum(float(getattr(tr.weights, k)) * float(v)
                     for k, v in out["raw"].items())
        assert manual == pytest.approx(out["total"], abs=1e-9)


def test_score_trace_applies_the_termination_penalty_once():
    trace = _trace()
    cfg = HEALTH.DEFAULT_CONFIGS[1]                    # v5 training reward
    clean = HEALTH.score_trace(cfg, trace, None)
    fall = HEALTH.score_trace(cfg, trace, "fall")
    assert fall["total"] == pytest.approx(clean["total"] - 100.0, abs=1e-9)
    assert fall["weighted"]["termination"] == -100.0
    assert fall["raw"]["termination"] == -100.0
    # a time-limit truncation is NOT a failure
    assert HEALTH.score_trace(cfg, trace, None)["total"] == pytest.approx(clean["total"])
    # a fall adds nothing more when the cause is not fall/dorsal
    assert HEALTH.score_trace(cfg, trace, "timeout")["total"] == pytest.approx(clean["total"])


def test_step_path_decomposition_matches_the_logged_reward():
    """Run 10 real steps; every row's weighted terms must equal its reward."""
    from solo.baselines import StandHoldController
    from solo.env import SoloEnv

    weights = RewardWeights(alive=10.0)                # v5's actual training reward
    env = SoloEnv(task="balance", seed=0, weights=weights,
                  action_mode="residual", residual_scale=0.5)
    tr = TaskReward("balance", weights=weights)
    ctrl = StandHoldController()
    env.reset(seed=0)
    rewards = []
    for _ in range(10):
        _, reward, _, _, info = env.step(ctrl(env, env.data))
        rewards.append(float(reward))
        terms = info["reward_terms"]
        weighted = sum(float(getattr(tr.weights, k)) * float(v) for k, v in terms.items())
        assert weighted == pytest.approx(float(reward), abs=1e-4), (terms, reward)
    rows = env.recorder.rows
    assert len(rows) == 10
    row_sum = sum(float(r["reward"]) for r in rows)
    assert row_sum == pytest.approx(sum(rewards), abs=1e-6)
    # the alive term dominates the per-step total while the robot merely stands
    alive_weighted = 10.0 * float(rows[0]["terms"]["alive"])
    assert alive_weighted > 0.9 * float(rows[0]["reward"])
    assert set(rows[0]["terms"]) <= set(TASK_TERMS["balance"]) | {"termination"}


# --------------------------------------------------------------------------- #
# named terms fire in their state and not otherwise
# --------------------------------------------------------------------------- #
def _base_inputs(**kw) -> RewardInputs:
    return RewardInputs(cmd=Command(vx=0.0), **kw)


def test_flat_orientation_fires_only_when_not_upright():
    tr = TaskReward("balance")
    up = tr.step(_base_inputs(torso_up_z=1.0))[1]["flat_orientation"]
    side = tr.step(_base_inputs(torso_up_z=0.0))[1]["flat_orientation"]
    down = tr.step(_base_inputs(torso_up_z=-1.0))[1]["flat_orientation"]
    assert up == 0.0                       # perfectly upright: no cost
    assert side == pytest.approx(-0.5)     # horizontal torso: half of the unit range
    assert down == pytest.approx(-1.0)     # inverted: the full cost
    assert up > side > down


def test_alive_fires_up_and_shuts_off_when_collapsed():
    tr = TaskReward("balance", weights=RewardWeights(alive=10.0))

    def step(**kw):
        return tr.step(_base_inputs(**kw))

    alive_up = step(torso_up_z=1.0, pelvis_z=0.79)[1]["alive"]
    alive_half = step(torso_up_z=1.0, pelvis_z=0.395)[1]["alive"]
    alive_down = step(torso_up_z=0.2, pelvis_z=0.10)[1]["alive"]
    assert alive_up == pytest.approx(1.0, abs=1e-12)
    assert alive_half == pytest.approx(0.5, abs=1e-12)     # half stance height
    assert alive_down == pytest.approx(0.2 * 0.10 / 0.79, abs=1e-12)
    assert alive_down < 0.05
    # the weighted step totals: a full upright step is 10.0, collapsed is < 0.2
    assert step(torso_up_z=1.0, pelvis_z=0.79)[0] == pytest.approx(10.0, abs=1e-9)
    assert step(torso_up_z=0.2, pelvis_z=0.10)[0] < 0.2


def test_saturation_and_limit_terms_fire_on_their_input():
    tr = TaskReward("balance")
    clean = tr.step(_base_inputs(sat_frac=0.0, limit_prox=0.0))[1]
    hot = tr.step(_base_inputs(sat_frac=1.0, limit_prox=0.75))[1]
    assert clean["torque_sat"] == 0.0 and clean["joint_limit"] == 0.0
    assert hot["torque_sat"] == -1.0
    assert hot["joint_limit"] == pytest.approx(-0.75)


def test_action_rate_is_silent_on_the_first_step_and_fires_on_change():
    tr = TaskReward("balance")
    same = np.zeros(N_JOINTS)
    first = tr.step(_base_inputs(action=same, prev_action=None))[1]["action_rate"]
    hold = tr.step(_base_inputs(action=same, prev_action=same))[1]["action_rate"]
    jerk = tr.step(_base_inputs(action=np.full(N_JOINTS, 0.5),
                                prev_action=np.zeros(N_JOINTS)))[1]["action_rate"]
    assert first == 0.0                    # a reset is not a penalty
    assert hold == 0.0                     # no change, no cost
    assert jerk == pytest.approx(-1.0)     # 0.5 rad mean change saturates the cost


# --------------------------------------------------------------------------- #
# keyframe-optimality grid shape (the monitor's theorem check)
# --------------------------------------------------------------------------- #
def test_keyframe_grid_is_the_documented_shape():
    base = np.linspace(-0.4, 0.4, N_JOINTS)
    full = HEALTH.keyframe_conditions(base, offsets=(0.02, 0.05, 0.10), n_random=0)
    assert len(full) == 1 + N_JOINTS * 6                 # keyframe + 29 joints x 6
    assert np.array_equal(full[0]["ctrl"], base)         # the keyframe is exact
    for row in full[1:]:
        delta = np.abs(row["ctrl"] - base)
        assert delta.max() == pytest.approx(row["delta_norm"], abs=1e-12)
        assert int((delta > 0).sum()) == 1               # one joint per condition
    reduced = HEALTH.keyframe_conditions(base, offsets=(0.05, 0.10), n_random=0)
    assert len(reduced) == 1 + N_JOINTS * 4              # the health default
    with_random = HEALTH.keyframe_conditions(base, offsets=(0.02, 0.05, 0.10),
                                             n_random=8, seed=3)
    assert len(with_random) == 1 + N_JOINTS * 6 + 8 * 3
    # seeded: the random arms are reproducible
    assert ([r["name"] for r in with_random] ==
            [r["name"] for r in HEALTH.keyframe_conditions(
                base, offsets=(0.02, 0.05, 0.10), n_random=8, seed=3)])
    for row in with_random[1 + N_JOINTS * 6:]:
        assert np.linalg.norm(row["ctrl"] - base) == pytest.approx(row["delta_norm"],
                                                                   abs=1e-9)


def test_metric_fields_include_what_the_health_table_reads():
    for key in ("upright", "pelvis_z", "sat_frac", "limit_prox", "act_delta",
                "com_offset", "reward"):
        assert key in METRIC_FIELDS
