"""Vectorised solo backend (``rl.vec_solo``): equivalence, determinism, seeds, faults.

The equivalence test is the one that makes the port safe: with ``n_envs=1`` the
vec path (both backends) must reproduce the **single-env** trajectory
byte-for-byte.  Both sides are driven by the identical unit-action sequence and
the identical reset policy (episode seed +1; push curriculum recomputed at
``steps``), and a sha256 stream accumulates ``(qpos, obs, reward, flags, seed)``
after every step -- so a divergence in physics, in the action-mode mapping, in
the observation dict or in the seed stream is a hard failure, not a tolerance.

Distinct-seed reproducibility: the four workers of an ``n_envs=4`` run start at
``seed + i * SEED_STRIDE``, those exact seeds (and episode indices) are written
into the checkpoint, and reloading re-resets every worker on the same stream
(asserted against a hands-on replay of the same units).

The fault test kills a worker with SIGKILL and requires the next exchange to
raise instead of returning stale observations.
"""

from __future__ import annotations

import hashlib
import os
import signal
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

# The repo has no installed package / conftest; every test file puts src/ on the
# path itself (same convention as tests/solo/test_config_plumbing.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rl.checkpoint import load_checkpoint  # noqa: E402
from rl.vec_solo import SEED_STRIDE, VecSoloEnv  # noqa: E402
from solo.curriculum import DEFAULT_CURRICULUM  # noqa: E402
from solo.env import SoloEnv  # noqa: E402
from solo.lit import joint_mask  # noqa: E402
from solo.obs import ACTOR_DIM, CRITIC_DIM, PRIV_DIM  # noqa: E402
from solo.reward import RewardWeights  # noqa: E402
from solo.scene import N_JOINTS, load_solo_model  # noqa: E402
from solo.train import SoloTrainer, TrainConfig  # noqa: E402

SEED = 11
SCALE = 0.37
#: pushes unlocked immediately: exercises episode resets (and the curriculum
#: recomputation at reset) inside the compared streams
HOT_CURRICULUM = replace(DEFAULT_CURRICULUM, start_steps=0, warmup_steps=1, seed=5)


@pytest.fixture(scope="module")
def model():
    return load_solo_model()


def _mix(h: hashlib._Hash, *arrs) -> None:
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())


def _action_stream(steps: int, seed: int = 4321) -> np.ndarray:
    """Unit actions ``(steps, 29)`` -- the policy-side input both paths share."""
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(steps, N_JOINTS))


def _drive_single(model, actions, *, seed, weights, curriculum, mode, scale,
                  env_kwargs=None):
    """Trainer-equivalent single-env loop; returns (hexdigest, per-step digests)."""
    env = SoloEnv(model, task="balance", seed=seed, weights=weights,
                  action_mode=mode, residual_scale=scale, **(env_kwargs or {}))
    if curriculum is not None:
        env.set_push_schedule(curriculum.schedule_for(0, seed))
    obs = env.reset(seed=seed)
    h = hashlib.sha256()
    digests = []
    _mix(h, obs["actor"], obs["privileged"], obs["critic"], env.data.qpos)
    digests.append(h.digest())
    ep_seed, steps = int(seed), 0
    for unit in actions:
        obs, reward, terminated, truncated, info = env.step(env.ctrl_from_policy(unit))
        steps += 1
        if terminated or truncated:
            ep_seed += 1
            if curriculum is not None:
                env.set_push_schedule(curriculum.schedule_for(steps, ep_seed))
            obs = env.reset(seed=ep_seed)
        _mix(h, np.float64(reward), np.bool_(terminated), np.bool_(truncated),
             env.data.qpos, obs["actor"], obs["privileged"], obs["critic"])
        if terminated or truncated:
            _mix(h, np.int64(ep_seed))
        digests.append(h.digest())
    return h.hexdigest(), digests


def _drive_vec(actions, *, backend, seed, weights, curriculum, mode, scale, model,
               env_kwargs=None):
    """VecSoloEnv (n_envs=1) driven by the same units; same digest stream."""
    vec = VecSoloEnv(1, backend=backend, seed=seed, weights=weights,
                     curriculum=curriculum, action_mode=mode, residual_scale=scale,
                     include_state=True, model=model, **(env_kwargs or {}))
    h = hashlib.sha256()
    digests = []
    _mix(h, vec.obs["actor"], vec.obs["privileged"], vec.obs["critic"], vec.qpos)
    digests.append(h.digest())
    try:
        for unit in actions:
            b = vec.step(unit.reshape(1, N_JOINTS))
            _mix(h, np.float64(b.rewards[0]), np.bool_(b.terminated[0]),
                 np.bool_(b.truncated[0]), vec.qpos, vec.obs["actor"],
                 vec.obs["privileged"], vec.obs["critic"])
            if b.terminated[0] or b.truncated[0]:
                _mix(h, np.int64(b.seeds[0]))
            digests.append(h.digest())
    finally:
        vec.close()
    return h.hexdigest(), digests


def _assert_same_stream(name, hs, ds, hv, dv, steps):
    if hs != hv:
        i = next(i for i, (x, y) in enumerate(zip(ds, dv)) if x != y)
        raise AssertionError(
            f"{name}: trajectory hash mismatch at step {i} of {steps} "
            f"(single={hs[:12]}... vec={hv[:12]}...)")


# ------------------------------------------------------------------ contract
def test_obs_contract_matches_single_env(model):
    """Keys, dims and dtypes of the batched obs are exactly SingleEnv's."""
    ref = SoloEnv(model, task="balance", seed=0)
    vec = VecSoloEnv(2, backend="sequential", seed=5, model=model)
    try:
        assert set(vec.obs) == set(ref.observation()) == {"actor", "privileged", "critic"}
        for key, dim in (("actor", ACTOR_DIM), ("privileged", PRIV_DIM),
                         ("critic", CRITIC_DIM)):
            assert vec.obs[key].shape == (2, dim)
            assert vec.obs[key].dtype == np.float32
        with pytest.raises(ValueError):
            vec.step(np.zeros((2, N_JOINTS + 1)))       # action shape is checked
        with pytest.raises(ValueError):
            VecSoloEnv(0, backend="sequential", seed=0, model=model)
    finally:
        vec.close()


# --------------------------------------------------------------- equivalence
@pytest.mark.parametrize("backend,mode,pushed,steps", [
    pytest.param("subproc", "residual", True, 600, id="subproc-residual+pushes"),
    pytest.param("subproc", "absolute", False, 400, id="subproc-absolute"),
    pytest.param("sequential", "residual", True, 600, id="sequential-residual"),
])
def test_vec_path_matches_single_env_exactly(model, backend, mode, pushed, steps):
    """n_envs=1 vec trajectory == single-env trajectory, bit for bit."""
    weights = RewardWeights(alive=10.0) if mode == "residual" else None
    curriculum = HOT_CURRICULUM if pushed else None
    actions = _action_stream(steps)
    hs, ds = _drive_single(model, actions, seed=SEED, weights=weights,
                           curriculum=curriculum, mode=mode, scale=SCALE)
    hv, dv = _drive_vec(actions, backend=backend, seed=SEED, weights=weights,
                        curriculum=curriculum, mode=mode, scale=SCALE, model=model)
    _assert_same_stream(f"{backend}/{mode}", hs, ds, hv, dv, steps)


def test_env_construction_kwargs_reach_workers(model):
    """``term_set``/``joint_mask`` are forwarded into every worker's ``SoloEnv``.

    The masked vec stream must equal the masked single-env stream bit for bit,
    and must differ from the unmasked one -- i.e. the levers reached the worker
    env, not just the parent.
    """
    mask = joint_mask(model)
    kwargs = {"term_set": "balance_lit", "joint_mask": mask}
    actions = _action_stream(150)
    unmasked = _drive_single(model, actions, seed=SEED, weights=None,
                             curriculum=None, mode="residual", scale=SCALE)
    masked = _drive_single(model, actions, seed=SEED, weights=None,
                           curriculum=None, mode="residual", scale=SCALE,
                           env_kwargs=kwargs)
    vec_masked = _drive_vec(actions, backend="subproc", seed=SEED, weights=None,
                            curriculum=None, mode="residual", scale=SCALE,
                            model=model, env_kwargs=kwargs)
    _assert_same_stream("masked vec vs masked single", masked[0], masked[1],
                        vec_masked[0], vec_masked[1], len(actions))
    assert masked[0] != unmasked[0], (
        "joint_mask/term_set had no observable effect: the comparison cannot "
        "prove the kwargs reached the worker")


def test_equivalence_stream_crosses_episode_boundaries(model):
    """The compared stream must actually contain resets (guards a vacuous test)."""
    actions = _action_stream(600)
    env = SoloEnv(model, task="balance", seed=SEED, action_mode="residual",
                  residual_scale=SCALE)
    env.set_push_schedule(HOT_CURRICULUM.schedule_for(0, SEED))
    env.reset(seed=SEED)
    ends = 0
    for unit in actions:
        _obs, _r, terminated, truncated, _info = env.step(env.ctrl_from_policy(unit))
        if terminated or truncated:
            ends += 1
            env.set_push_schedule(HOT_CURRICULUM.schedule_for(0, SEED + ends))
            env.reset(seed=SEED + ends)
    assert ends >= 2, f"equivalent stream crossed only {ends} episode ends"


# ------------------------------------------------- trainer-level equivalence
def test_collect_vec_at_n1_equals_single_collect(model):
    """``collect_vec`` at n_envs=1 == the single-env ``collect``, byte for byte.

    Covers the trainer-level half of the equivalence acceptance: identical net
    sampling (batched ``(1, D)`` vs unsqueezed ``(D,)`` must draw the same
    tensor), identical ctrl mapping (inside the env in both cases), identical
    resets.  NOTE: ``SoloTrainer`` does not seed net *initialisation* from
    ``cfg.seed`` (pre-existing; it seeds after construction), so the test pins
    the torch RNG before each construction to give both trainers the same
    weights -- that is the "same seed(s)" the acceptance asks for.
    """
    base = dict(task="balance", steps=64, rollout_steps=32, seed=9,
                action_mode="residual", alive_weight=10.0,
                out="", log_every=0, save_every=0)
    torch.manual_seed(4242)
    tr_single = SoloTrainer(TrainConfig(**base), model=model)
    torch.manual_seed(4242)
    tr_vec = SoloTrainer(TrainConfig(**base), model=model)
    try:
        tr_vec.vec = VecSoloEnv(1, seed=base["seed"], model=model,
                                action_mode="residual", residual_scale=0.5,
                                weights=RewardWeights(alive=10.0),
                                curriculum=tr_vec._vec_curriculum(tr_vec.cfg))
        tr_vec._vec_obs = tr_vec.vec.obs
        tr_vec._vec_ep_returns = [0.0]
        torch.manual_seed(base["seed"])
        b_single = tr_single.collect()
        torch.manual_seed(base["seed"])
        b_vec = tr_vec.collect_vec()
        for field in ("actor_obs", "critic_obs", "actions_unit", "logp",
                      "values", "rewards", "dones"):
            assert np.array_equal(getattr(b_single, field),
                                  getattr(b_vec, field)), field
        assert tr_single.steps_done == tr_vec.steps_done == tr_vec.vec.steps == 32
    finally:
        tr_single.close()
        tr_vec.close()


# --------------------------------------------------------------- determinism
def _vec_hash_run(n_envs: int, steps: int, *, seed: int, model) -> tuple:
    vec = VecSoloEnv(n_envs, backend="subproc", seed=seed, model=model,
                     include_state=True, curriculum=HOT_CURRICULUM)
    units = _action_stream(steps, seed=99)
    h = hashlib.sha256()
    _mix(h, vec.obs["actor"], vec.obs["privileged"], vec.obs["critic"], vec.qpos)
    dones = 0
    try:
        for unit in units:
            b = vec.step(np.repeat(unit[None, :], n_envs, axis=0))
            dones += int(np.count_nonzero(b.terminated | b.truncated))
            _mix(h, b.rewards, b.terminated, b.truncated, b.seeds, vec.qpos,
                 vec.obs["actor"], vec.obs["privileged"], vec.obs["critic"])
    finally:
        vec.close()
    return h.hexdigest(), dones


def test_determinism_same_seed_same_n_envs(model):
    """Same seed + same n_envs => identical rollout hash across two runs."""
    h1, d1 = _vec_hash_run(4, 200, seed=13, model=model)
    h2, d2 = _vec_hash_run(4, 200, seed=13, model=model)
    assert h1 == h2, f"{h1[:16]} != {h2[:16]}"
    assert d1 == d2 and d1 > 0, f"episode ends across the run differ: {d1} vs {d2}"


def test_trainer_collect_vec_deterministic_hash(model):
    """The trainer's own vec collect (policy sampling + resets) is reproducible.

    Same caveat as above: ``torch.manual_seed`` pins the (pre-existing,
    unseeded) net initialisation so the two trainers start from equal weights
    and equal sampling state.
    """

    def run():
        cfg = TrainConfig(task="balance", steps=4096, rollout_steps=160, seed=3,
                          n_envs=2, action_mode="residual", alive_weight=10.0,
                          push_curriculum=True, push_start_steps=0,
                          push_warmup_steps=1, out="", log_every=0, save_every=0)
        torch.manual_seed(2718)
        tr = SoloTrainer(cfg, model=model)
        try:
            batch = tr.collect_vec()
            h = hashlib.sha256()
            _mix(h, batch.actor_obs, batch.critic_obs, batch.actions_unit,
                 batch.logp, batch.values, batch.rewards, batch.dones)
            return h.hexdigest(), tr.steps_done, tr.vec.steps
        finally:
            tr.close()

    h1, steps1, vec_steps1 = run()
    h2, steps2, vec_steps2 = run()
    assert h1 == h2, f"{h1[:16]} != {h2[:16]}"
    assert steps1 == steps2 == 160 * 2 and vec_steps1 == vec_steps2 == steps1


# ------------------------------------------------------- seeds / checkpoint
def test_distinct_seeds_recorded_and_reloadable(model, tmp_path):
    cfg = TrainConfig(task="balance", steps=10**9, rollout_steps=64, seed=21,
                      n_envs=4, action_mode="residual",
                      push_curriculum=True, push_start_steps=0,
                      push_warmup_steps=1, out=str(tmp_path / "vec.pt"),
                      log_every=0, save_every=0)
    tr = SoloTrainer(cfg, model=model)
    try:
        seeds = list(tr.vec.worker_seeds)
        assert len(set(seeds)) == 4
        assert seeds == [21 + i * SEED_STRIDE for i in range(4)]
        for _ in range(6):                       # 6 * 64 * 4 = 1536 env steps
            tr.collect_vec()
        assert max(tr.vec.episode_indices) > 0, "no episode ended; test vacuous"
        assert tr.vec.steps == tr.steps_done == 6 * 64 * 4
        path = tr.save()
        ckpt = load_checkpoint(path)
        env_cfg = ckpt["config"]["env"]
        assert ckpt["config"]["train"]["n_envs"] == 4
        assert env_cfg["n_envs"] == 4 and env_cfg["backend"] == "subproc"
        assert env_cfg["worker_seeds"] == seeds
        assert env_cfg["episode_indices"] == tr.vec.episode_indices
        assert env_cfg["current_seeds"] == tr.vec.current_seeds
        current = list(tr.vec.current_seeds)

        cfg2 = replace(cfg, out="")
        tr2 = SoloTrainer(cfg2, model=model)
        try:
            tr2.load(path)
            assert tr2.vec.worker_seeds == seeds           # the same four
            assert tr2.vec.current_seeds == current
            assert tr2.vec.steps == tr.steps_done
            # reload replays the same streams: a fresh reset of tr.vec at the
            # saved seeds must give tr2's loaded obs back, exactly
            obs_a = tr.vec.reset(seeds=tr.vec.current_seeds, steps=tr.vec.steps)
            for key in ("actor", "privileged", "critic"):
                assert np.array_equal(obs_a[key], tr2.vec.obs[key]), key
            units = np.tile(_action_stream(4, seed=2718)[3], (4, 1))
            assert np.array_equal(tr.vec.step(units).obs["actor"],
                                  tr2.vec.step(units).obs["actor"])
        finally:
            tr2.close()
    finally:
        tr.close()


# ---------------------------------------------------------------------- wiring
def test_trainer_uses_vec_only_above_one_env(model):
    base = dict(task="balance", steps=64, rollout_steps=16, seed=0,
                action_mode="residual", out="", log_every=0, save_every=0)
    tr1 = SoloTrainer(TrainConfig(**base), model=model)
    try:
        assert tr1.vec is None and tr1.env is not None      # single-env path kept
        assert tr1.cfg.ppo_config().n_envs == 1
    finally:
        tr1.close()
    tr3 = SoloTrainer(TrainConfig(**base, n_envs=3), model=model)
    try:
        assert tr3.env is None and tr3.vec is not None
        assert tr3.vec.n_envs == 3 and tr3.cfg.ppo_config().n_envs == 3
        assert tr3.startup_ctrl_diff is not None and tr3.startup_ctrl_diff < 0.05
        batch = tr3.collect_vec()
        assert batch.shape == (16, 3)
        assert tr3.vec.steps == tr3.steps_done == 48
    finally:
        tr3.close()


# ----------------------------------------------------------------------- fault
def test_dead_worker_raises_instead_of_stale_obs(model):
    vec = VecSoloEnv(2, backend="subproc", seed=2, model=model)
    units = np.zeros((2, N_JOINTS))
    try:
        vec.step(units)                                    # healthy exchange
        os.kill(vec.worker_pids[0], signal.SIGKILL)
        with pytest.raises(RuntimeError, match="worker 0"):
            vec.step(units)
        with pytest.raises(RuntimeError, match="worker 0"):
            vec.step(units)                                # stays surfaced
    finally:
        vec.close()
