"""Tests for the Phase-5 PPO / resistance-training infrastructure (``src/rl``).

Fast, no training: observation/privileged layouts and separation, action
scaling into the env ctrlrange, hand-computed GAE, checkpoint roundtrip and
RNG continuity, curriculum schedules and advance rules, the optional technique
scorer's graceful degradation, the knees/hands-not-terminal regression through
the RL wrapper, deterministic seed reproducibility, and a light trainer
resume roundtrip.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrestling.env import (OBS_DIM, QPOS_SLICE, QVEL_SLICE, STEP_DT,  # noqa: E402
                           StandHold, WrestlingEnv, load_wrestling_model,
                           reference_trace)
from wrestling.backdet import back_features  # noqa: E402

from rl.checkpoint import (FORMAT_VERSION, apply_checkpoint, load_checkpoint,  # noqa: E402
                           restore_rng, rng_state, save_checkpoint,
                           warm_start_from_bc)
from rl.curriculum import (AdvanceRule, CommandScheduler, Curriculum,  # noqa: E402
                           PISTY_STAGES, PerturbationSchedule, RewardWeights,
                           StageConfig, stage_by_key, stage_from_dict)
from rl.net import (ActionMapper, ActorCritic, NetConfig, action_mapper_for,  # noqa: E402
                    count_out_of_bounds, robot_action_bounds)
from rl.obs import (COMMAND_DIM, ENV_OBS_DIM, N_TECHNIQUES, TECHNIQUES,  # noqa: E402
                    ActorObsBuilder, TechniqueCommand, actor_obs_dim,
                    actor_obs_layout, command_vector, reference_duration,
                    reference_phase)
from rl.ppo import PPOConfig, RolloutBatch, compute_gae, lr_at, ppo_update  # noqa: E402
from rl.privileged import (CONTACT_KINDS, PRIV_DIM, CriticObsBuilder,  # noqa: E402
                           PrivilegedObsBuilder, back_exposure, priv_layout,
                           unpack)
from rl.reward import ScorerAdapter, StageReward, technique_executor  # noqa: E402
from rl.rollout import RolloutCollector  # noqa: E402
from rl.scripted import OpponentSpec, reference_base_ctrl  # noqa: E402
from rl.trainer import Trainer, evaluate, make_policy, resolve_backend  # noqa: E402
from rl.vec import RESET_SEED_STRIDE, VecWrestlingEnv  # noqa: E402

SMOKE_ENV = {"match_clock": 4.0, "exchange_timeout": 2.0}


@pytest.fixture(scope="module")
def model():
    return load_wrestling_model()


@pytest.fixture()
def env(model):
    e = WrestlingEnv(model=model, controllers=(StandHold("a"), StandHold("b")), seed=0,
                     match_clock=4.0, exchange_timeout=2.0)
    e.reset(seed=0)
    return e


def _ground_start_pose():
    """STAND_UP reference frames measured to put knees/hands on the mat."""
    tr = reference_trace("STAND_UP")
    return tr["qpos_a"][6].copy(), tr["qpos_b"][6].copy()


# --------------------------------------------------------------- observations
def test_actor_obs_dim_layout_and_command():
    assert ENV_OBS_DIM == OBS_DIM == 84            # pinned to the env contract
    assert TECHNIQUES == ("DOUBLE_LEG", "SINGLE_LEG", "BODY_LOCK", "SNAPDOWN",
                          "SPRAWL", "STAND_UP", "STANCE")
    assert N_TECHNIQUES == 7 and COMMAND_DIM == 8
    assert actor_obs_dim(1) == 92 and actor_obs_dim(4) == 344
    for k in (1, 4):
        layout = actor_obs_layout(k)
        assert layout[0][1][0] == 0 and layout[-1][1][1] == actor_obs_dim(k)
        for (_, (s0, e0)), (_, (s1, _)) in zip(layout, layout[1:]):
            assert e0 == s1, "layout must be contiguous"
    v = command_vector("DOUBLE_LEG", 0.5)
    assert v.shape == (8,) and v[0] == 1.0 and abs(v[7] - 0.5) < 1e-6
    assert np.allclose(command_vector(None, 0.9), 0.0)   # inactive command = zeros
    with pytest.raises(KeyError):
        command_vector("NOT_A_TECHNIQUE")


def test_actor_obs_frame_stack_evolution():
    b = ActorObsBuilder(frame_stack=4)
    o0 = np.full(ENV_OBS_DIM, 1.0, dtype=np.float32)
    o1 = np.full(ENV_OBS_DIM, 2.0, dtype=np.float32)
    a0, _ = b.reset((o0, o0))
    assert np.allclose(a0[:ENV_OBS_DIM], 1.0) and np.allclose(a0[2 * ENV_OBS_DIM:3 * ENV_OBS_DIM], 1.0)
    cmd = TechniqueCommand("SPRAWL", 0.25)
    a1, _ = b.step((o1, o1), (cmd, cmd))
    frames = [a1[i * ENV_OBS_DIM:(i + 1) * ENV_OBS_DIM] for i in range(4)]
    assert np.allclose(frames[0], 1.0) and np.allclose(frames[1], 1.0)
    assert np.allclose(frames[2], 1.0) and np.allclose(frames[3], 2.0)   # newest last
    assert a1[-COMMAND_DIM + TECHNIQUES.index("SPRAWL")] == 1.0 and abs(a1[-1] - 0.25) < 1e-6
    assert a1.shape == (actor_obs_dim(4),) and a1.dtype == np.float32
    # episode-boundary reseed: next step repeats the fresh obs
    b.reseed((o1, o1))
    a2, _ = b.step((o1, o1), (cmd, cmd))
    for i in range(4):
        assert np.allclose(a2[i * ENV_OBS_DIM:(i + 1) * ENV_OBS_DIM], 2.0)


def test_actor_obs_never_sees_privileged(env, model):
    """The actor path is a function of env obs + command only."""
    obs_pair = env.obs_fn(model, env.data)
    builder = ActorObsBuilder(frame_stack=1)
    actor_before, _ = builder.reset(obs_pair)
    priv_before = PrivilegedObsBuilder(model).build(model, env.data)

    # mutate ONLY privileged state: opponent joints / joint velocities
    env.data.qpos[QPOS_SLICE["b"]][7:36] += 0.2
    env.data.qvel[QVEL_SLICE["b"]][6:35] += 0.5
    import mujoco

    mujoco.mj_kinematics(model, env.data)
    obs_pair2 = env.obs_fn(model, env.data)
    builder2 = ActorObsBuilder(frame_stack=1)
    actor_after, _ = builder2.reset(obs_pair2)
    priv_after = PrivilegedObsBuilder(model).build(model, env.data)

    assert np.allclose(actor_before, actor_after), "actor obs changed with privileged state"
    assert not np.allclose(priv_before, priv_after), "privileged obs must see full state"
    # structural: the actor builder has no access to MjData / privileged inputs
    import inspect

    params = set(inspect.signature(ActorObsBuilder.step).parameters)
    assert params == {"self", "env_obs_pair", "commands"}
    assert set(inspect.signature(ActorObsBuilder.reset).parameters) == {"self", "env_obs_pair", "commands"}


def test_critic_obs_is_actor_plus_privileged(env, model):
    pb = PrivilegedObsBuilder(model)
    priv = pb.build(model, env.data)
    assert priv.shape == (PRIV_DIM,) and priv.dtype == np.float32
    assert [n for n, _, _ in priv_layout()][0] == "qpos_a"
    assert priv_layout()[-1][2] == PRIV_DIM
    assert all(len(s) > 0 for s in pb.describe()["geom_counts"].values())
    cb = CriticObsBuilder(92, pb)
    actor = np.arange(92, dtype=np.float32)
    critic = cb.build(actor, priv)
    assert critic.shape == (92 + PRIV_DIM,)
    assert np.allclose(critic[:92], actor), "critic obs must start with the actor obs"
    u = unpack(priv)
    assert set(u) == {"qpos_a", "qpos_b", "qvel_a", "qvel_b", "contact_a", "contact_b",
                      "back_a", "back_b"}
    assert np.allclose(u["qpos_a"], env.data.qpos[QPOS_SLICE["a"]])
    assert np.allclose(u["qvel_b"], env.data.qvel[QVEL_SLICE["b"]])


def test_privileged_contacts_and_back_exposure(model):
    """Knee/hand floor contacts are visible; the exposure proxy is threshold-consistent."""
    pb = PrivilegedObsBuilder(model)
    env = WrestlingEnv(model=model, controllers=(StandHold("a"), StandHold("b")), seed=0,
                       match_clock=3.0, exchange_timeout=3.0)
    qa, qb = _ground_start_pose()
    env.reset(seed=0, pose=(qa, qb))
    priv = unpack(pb.build(model, env.data))
    assert priv["contact_a"][CONTACT_KINDS.index("hands")] > 0.5     # a hand on the mat
    assert priv["contact_b"][CONTACT_KINDS.index("knees")] > 0.5     # b knees on the mat
    assert priv["qpos_a"][2] < 0.25                                  # low pelvis (kneeling)
    assert back_exposure(45.0, 0.35) == 0.0                          # at the detector threshold
    assert back_exposure(90.0, 0.0) == 1.0                           # fully exposed
    assert back_exposure(80.0, 0.1) > back_exposure(60.0, 0.1)
    assert back_exposure(80.0, 0.3) < back_exposure(80.0, 0.1)


# -------------------------------------------------------------------- actions
def test_action_scaling_maps_to_ctrlrange(model):
    lo, hi = robot_action_bounds(model, "a")
    mid = 0.5 * (lo + hi)
    for mode in ("absolute", "residual"):
        mp = ActionMapper(lo, hi, mode, residual_scale=0.5)
        unit = np.linspace(-1.0, 1.0, 29)
        ctrl = mp.to_ctrl(unit, mid)
        assert np.all(mp.in_bounds(ctrl)) and count_out_of_bounds(ctrl, model, robot="a") == 0
        sat = mp.to_ctrl(np.full(29, 5.0), mid)          # saturates, never escapes
        assert np.all(mp.in_bounds(sat))
        if mode == "absolute":
            assert np.allclose(mp.to_ctrl(np.zeros(29), None), mid)
            assert np.allclose(mp.to_ctrl(np.ones(29), None), hi)
        else:
            with pytest.raises(ValueError):
                mp.to_ctrl(np.zeros(29), None)           # residual needs a base
    with pytest.raises(ValueError):
        ActionMapper(lo, hi, "nope")


def test_reference_base_ctrl_is_trace_backed():
    base = reference_base_ctrl("DOUBLE_LEG", "a", 0.5)
    assert base.shape == (29,)
    tr = reference_trace("DOUBLE_LEG")
    assert np.allclose(reference_base_ctrl("DOUBLE_LEG", "a", 0.0), tr["qpos_a"][0, 7:36])
    assert np.allclose(reference_base_ctrl("DOUBLE_LEG", "a", 2.0), tr["qpos_a"][-1, 7:36])
    assert np.all(np.isfinite(base))


# ------------------------------------------------------------------------ PPO
def test_gae_hand_computed():
    r = np.array([[1.0], [2.0], [3.0], [4.0]])
    v = np.array([[0.5], [1.0], [1.5], [2.0]])
    d = np.zeros_like(r)
    adv, ret = compute_gae(r, v, d, [2.5], 0.9, 0.8)
    # hand-computed: delta = [1.4, 2.35, 3.3, 4.25]; A3=4.25, A2=3.3+.72*4.25=6.36,
    # A1=2.35+.72*6.36=6.9292, A0=1.4+.72*6.9292=6.389024
    expected = np.array([[6.389024], [6.9292], [6.36], [4.25]])
    assert np.allclose(adv, expected, atol=1e-9)
    assert np.allclose(ret, adv + v)
    # done at t=1: no bootstrap through the terminal, A1 = delta1 = 2 + 0.9*0 - 1
    d2 = d.copy()
    d2[1] = 1.0
    adv2, _ = compute_gae(r, v, d2, [2.5], 0.9, 0.8)
    assert abs(adv2[1, 0] - 1.0) < 1e-12
    assert abs(adv2[0, 0] - (1.4 + 0.72 * 1.0)) < 1e-12
    # lambda = 0 -> plain TD residuals
    adv3, _ = compute_gae(r, v, d, [2.5], 0.9, 0.0)
    assert np.allclose(adv3[:, 0], [1.4, 2.35, 3.3, 4.25])
    with pytest.raises(ValueError):
        compute_gae(r, v[:2], d, [0.0], 0.9, 0.8)


def _toy_batch(T=5, N=2, act_dim=3, actor_dim=8, critic_dim=12):
    cfg = PPOConfig()
    b = RolloutBatch(
        actor_obs=np.zeros((T, N, actor_dim), np.float32),
        critic_obs=np.zeros((T, N, critic_dim), np.float32),
        actions_unit=np.zeros((T, N, act_dim), np.float32),
        logp=np.zeros((T, N), np.float32),
        values=np.zeros((T, N), np.float32),
        rewards=np.ones((T, N), np.float32),
        dones=np.zeros((T, N), np.float32))
    b.finalize(np.zeros(N), cfg)
    return b


def test_ppo_update_and_lr_schedule():
    cfg = PPOConfig()
    assert lr_at(0.0, cfg) == cfg.lr                       # 0 = start of training
    assert abs(lr_at(1.0, cfg) - cfg.lr * cfg.lr_end_frac) < 1e-15
    assert lr_at(0.25, cfg) > lr_at(0.75, cfg)             # anneals down
    net = ActorCritic(8, 12, act_dim=3, cfg=NetConfig(hidden=(16, 16)))
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    before = copy.deepcopy(net.state_dict())
    stats = ppo_update(net, opt, _toy_batch(), cfg, cfg.lr, torch.Generator().manual_seed(0))
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "lr"):
        assert np.isfinite(stats[k]), (k, stats[k])
    assert stats["n_samples"] == 10 and stats["epochs_run"] >= 1
    changed = any(not torch.allclose(net.state_dict()[k], v) for k, v in before.items())
    assert changed, "PPO update must change parameters"
    assert all(g["lr"] == cfg.lr for g in opt.param_groups)


# ---------------------------------------------------------------- checkpoints
def test_checkpoint_roundtrip_and_atomicity(tmp_path):
    net = ActorCritic(92, 92 + PRIV_DIM, act_dim=29, cfg=NetConfig(hidden=(32, 32)))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    state = {"step_count": 42, "update_count": 3, "target_steps": 1000,
             "curriculum": {"index": 2, "steps_in_stage": 7, "window": [1, 1, -1]}}
    path = tmp_path / "ck.pt"
    save_checkpoint(path, policy=net, optimizer=opt, cfg=PPOConfig(), state=state)
    assert path.exists() and not list(tmp_path.glob("*.tmp*"))
    before = copy.deepcopy(net.state_dict())
    with torch.no_grad():
        net.actor.trunk[-1].bias += 1.0
    ckpt = load_checkpoint(path)
    assert ckpt["format_version"] == FORMAT_VERSION
    st = apply_checkpoint(ckpt, policy=net, optimizer=opt)
    assert st["step_count"] == 42 and st["curriculum"]["index"] == 2
    for k, v in before.items():
        assert torch.allclose(net.state_dict()[k], v), k
    assert opt.state_dict()["param_groups"][0]["lr"] == 1e-3
    # overwrite in place is safe too
    save_checkpoint(path, policy=net, optimizer=opt, cfg=PPOConfig(), state=st)
    assert load_checkpoint(path)["state"]["step_count"] == 42
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")


def test_rng_state_roundtrip():
    torch.manual_seed(0)
    np.random.seed(7)
    st = rng_state()
    a = torch.randn(3)
    b = np.random.rand(2)
    # perturb both streams, then restore and replay
    torch.manual_seed(99)
    np.random.seed(99)
    torch.randn(5)
    np.random.rand(5)
    restore_rng(st)
    assert torch.allclose(a, torch.randn(3))
    assert np.allclose(b, np.random.rand(2))


def test_bc_warm_start(tmp_path):
    net = ActorCritic(92, 92 + PRIV_DIM, act_dim=29, cfg=NetConfig(hidden=(32, 32)))
    path = tmp_path / "bc.pt"
    src = {f"model.actor.{k}": v.clone() for k, v in net.actor.state_dict().items()}
    src["model.actor.trunk.0.weight"] = torch.zeros(3, 3)  # shape mismatch: skipped
    torch.save({"actor": src, "obs_dim": 92}, path)
    status = warm_start_from_bc(net, path)
    assert status["ok"] and status["loaded"], status
    assert any("trunk.0.weight" in m for m in status["mismatched"])
    with pytest.raises(ValueError):
        warm_start_from_bc(net, path, strict=True)
    status2 = warm_start_from_bc(net, tmp_path / "nope.pt")
    assert not status2["ok"] and "not found" in status2["note"]


# ---------------------------------------------------------------- curriculum
def test_curriculum_stage_schedule_and_roundtrip():
    assert [s.key for s in PISTY_STAGES] == ["A", "B", "C", "D", "E"]
    A, B, C, D, E = PISTY_STAGES
    assert C.weights.technique_similarity > D.weights.technique_similarity > 0
    assert E.weights.technique_similarity == 0.0 and E.weights.progress == 0.0
    assert D.weights.outcome > C.weights.outcome          # middle: strong outcome
    assert E.weights.outcome >= 1.0                       # late: win/loss + minimal shaping
    assert E.weights.engagement <= 0.02 and E.weights.engagement >= 0.0
    assert E.command_on is False and A.command_on is True
    assert A.weights.as_dict().keys() == {"technique_similarity", "progress", "outcome",
                                         "oob", "engagement"}
    p = C.perturbations
    assert p.noise_std(0) == p.action_noise_start
    assert p.noise_std(p.ramp_steps) == p.action_noise_end
    assert p.noise_std(10 * p.ramp_steps) == p.action_noise_end
    assert stage_by_key("c").key == "C"
    for s in PISTY_STAGES:
        assert stage_from_dict(s.as_dict()).as_dict() == s.as_dict()
    with pytest.raises(KeyError):
        stage_by_key("Z")


def test_curriculum_advance_rule():
    stage = StageConfig(key="T", name="test", techniques=("DOUBLE_LEG",),
                        advance=AdvanceRule(min_steps=5, threshold=0.5, window_episodes=4))
    cur = Curriculum(stages=(stage, stage), start_index=0)
    for _ in range(4):
        cur.on_exchange(1)
    assert cur.metric() == 1.0
    assert cur.maybe_advance() is False           # min_steps not reached
    cur.tick(5)
    assert cur.maybe_advance() is True
    assert cur.index == 1 and cur.steps_in_stage == 0 and len(cur.window) == 0
    # below threshold: never advances
    cur2 = Curriculum(stages=(stage, stage), start_index=0)
    for o in (1, -1, -1, 1):
        cur2.on_exchange(o)
    cur2.tick(100)
    assert cur2.metric() == 0.5 and cur2.maybe_advance() is True  # threshold is inclusive
    cur3 = Curriculum(stages=(stage, stage), start_index=0)
    for o in (-1, -1, 0, -1):
        cur3.on_exchange(o)
    cur3.tick(100)
    assert cur3.metric() == 0.125 and cur3.maybe_advance() is False
    # draws count as half
    cur4 = Curriculum(stages=(stage, stage), start_index=0)
    for o in (0, 0, 0, 0):
        cur4.on_exchange(o)
    assert cur4.metric() == 0.5


def test_command_scheduler_determinism():
    s1 = CommandScheduler(("DOUBLE_LEG", "SINGLE_LEG"), command_on=True, phase_jitter_s=0.3, seed=11)
    s2 = CommandScheduler(("DOUBLE_LEG", "SINGLE_LEG"), command_on=True, phase_jitter_s=0.3, seed=11)
    for env_i, ex_i in ((0, 0), (1, 3), (0, 7)):
        c1 = s1.command(env_i, ex_i, 0.5)
        c2 = s2.command(env_i, ex_i, 0.5)
        assert c1 == c2
        assert c1.technique in ("DOUBLE_LEG", "SINGLE_LEG")
        assert 0.0 <= c1.phase <= 1.0
    off = CommandScheduler(("DOUBLE_LEG",), command_on=False, seed=0)
    assert off.command(0, 0, 1.0).technique is None
    assert reference_phase("DOUBLE_LEG", 0.0) == 0.0
    assert reference_phase("DOUBLE_LEG", reference_duration("DOUBLE_LEG")) == 1.0
    assert reference_phase("DOUBLE_LEG", 10 * reference_duration("DOUBLE_LEG")) == 1.0


# -------------------------------------------------------------------- rewards
def test_reward_event_weights_and_scorer_degradation(env, model):
    heavy = StageReward(RewardWeights(outcome=1.5, oob=2.0))
    ra, rb = heavy.reward_fn(env, {"kind": "exchange_end", "winner": "a", "loser": "b"})
    assert ra == 1.5 and rb == -1.5
    ra, rb = heavy.reward_fn(env, {"kind": "oob", "robot": "b", "count": 1})
    assert abs(ra) < 1e-12 and abs(rb + 0.5) < 1e-12
    assert heavy.reward_fn(env, {"kind": "exchange_end", "winner": None}) == (0.0, 0.0)
    # scorer disabled -> similarity term is exactly zero, training continues
    sr = StageReward(RewardWeights(technique_similarity=1.0, progress=0.0),
                     scorer=ScorerAdapter(model, enabled=False))
    q = np.zeros(36)
    assert sr.shaping(technique="DOUBLE_LEG", phase=0.5, phase_next=0.6,
                      qpos_self=q, qpos_opp=q) == 0.0
    assert sr.status()["scorer_available"] is False
    # failing loader also degrades gracefully
    def boom():
        raise ImportError("scorer missing in this environment")

    ad = ScorerAdapter(model, enabled=True)
    ad._scorer = None
    ad.available = False
    ad.note = "forced unavailable"
    assert ad.similarity("DOUBLE_LEG", 0.5, q, q) is None
    assert ad.phase_index("DOUBLE_LEG", 0.5) is None
    # executor mismatch (learner cannot be scored for SPRAWL) -> skipped with a warning
    sr2 = StageReward(RewardWeights(technique_similarity=1.0), learner_robot="a",
                      scorer=ScorerAdapter(model, enabled=False))
    sr2.shaping(technique="SPRAWL", phase=0.1, phase_next=0.2, qpos_self=q, qpos_opp=q)
    assert technique_executor("SPRAWL") == "b" and sr2.warnings


def test_scorer_adapter_with_real_scorer_if_present(model, env):
    try:
        import scorer  # noqa: F401
    except Exception as exc:  # optional component, built in parallel
        pytest.skip(f"src/scorer unavailable: {exc!r}")
    ad = ScorerAdapter(model, enabled=True)
    if not ad.available:
        pytest.skip(f"scorer adapter could not init: {ad.note}")
    tr = reference_trace("DOUBLE_LEG")
    frac = 0.5
    i = int(frac * (len(tr["t"]) - 1))
    sim = ad.similarity("DOUBLE_LEG", frac, tr["qpos_a"][i], tr["qpos_b"][i])
    assert sim is not None and 0.0 <= sim <= 1.0
    assert ad.phase_index("DOUBLE_LEG", frac) is not None


# ------------------------------------------------- env-wrapper regressions
def test_knee_hand_contact_not_terminal_through_rl_wrapper(model):
    """Knees/hands on the mat never end an exchange through the RL wrapper."""
    vec = VecWrestlingEnv(1, backend="sequential", seed=0, learner_robot="a",
                          opponent_spec=OpponentSpec("stand_hold"), env_kwargs=SMOKE_ENV)
    qa, qb = _ground_start_pose()
    batch = vec.reset(seeds=[0], poses=[(qa, qb)])
    inner = vec._states[0].env
    hold = np.asarray(inner.data.ctrl[0:29], dtype=np.float64).copy()
    limb_seen = False
    horizon = 20  # 0.4 s: the ground-start pose is not statically holdable
    for _ in range(horizon):
        batch = vec.step(hold[None, :])
        p = unpack(batch.priv[0])
        contacts = {"a": p["contact_a"], "b": p["contact_b"]}
        for r, flags in contacts.items():
            if flags[CONTACT_KINDS.index("knees")] > 0.5 or flags[CONTACT_KINDS.index("hands")] > 0.5:
                limb_seen = True
        assert not batch.dones[0], "knee/hand contact must not terminate the exchange"
        assert batch.infos[0].get("exchange_ended") is None, batch.infos[0]
    assert limb_seen, "fixture must actually place knees/hands on the mat"
    assert inner.exchange_log == []
    assert inner.backdet.triggers == {"a": None, "b": None}
    vec.close()


def test_deterministic_seeds_end_to_end(model):
    actions = np.linspace(-0.2, 0.2, 29)[None, :].repeat(1, axis=0)

    def run_vec():
        v = VecWrestlingEnv(1, backend="sequential", seed=7, learner_robot="a",
                            opponent_spec=OpponentSpec("stand_hold"), env_kwargs=SMOKE_ENV)
        out = [v.reset(seeds=[7])]
        for _ in range(4):
            out.append(v.step(actions))
        v.close()
        return out

    r1, r2 = run_vec(), run_vec()
    for a, b in zip(r1, r2):
        assert np.array_equal(a.obs_a, b.obs_a) and np.array_equal(a.priv, b.priv)
        assert np.array_equal(a.rewards, b.rewards)

    # policy sampling reproducibility under a fixed torch seed
    net = ActorCritic(92, 92 + PRIV_DIM, act_dim=29, cfg=NetConfig(hidden=(16, 16)))
    obs = torch.zeros(3, 92)
    torch.manual_seed(5)
    o1 = net.actor.sample(obs)
    torch.manual_seed(5)
    o2 = net.actor.sample(obs)
    assert torch.equal(o1["unit"], o2["unit"]) and torch.equal(o1["logp"], o2["logp"])

    # rollout collector reproducibility (same seed, same env seeds, fixed actions)
    def run_collect():
        torch.manual_seed(0)
        v = VecWrestlingEnv(1, backend="sequential", seed=3, learner_robot="a",
                            opponent_spec=OpponentSpec("stand_hold"), env_kwargs=SMOKE_ENV)
        cfg = PPOConfig(rollout_steps=6, n_envs=1)
        rc = RolloutCollector(policy=net, mapper=action_mapper_for(model, "a"),
                              scheduler=CommandScheduler(("DOUBLE_LEG",), seed=0),
                              stage_reward=StageReward(RewardWeights(technique_similarity=0.4,
                                                                    progress=0.3)),
                              cfg=cfg, seed=1)
        rc.attach(v)
        rb = rc.collect(6, noise_std=0.02)
        v.close()
        return rb

    b1, b2 = run_collect(), run_collect()
    assert np.array_equal(b1.rewards, b2.rewards)
    assert np.array_equal(b1.actor_obs, b2.actor_obs)
    assert np.allclose(b1.advantages, b2.advantages)


def test_vec_backends_agree(model):
    actions = np.zeros((1, 29))
    v1 = VecWrestlingEnv(1, backend="sequential", seed=5, learner_robot="a",
                         opponent_spec=OpponentSpec("stand_hold"), env_kwargs=SMOKE_ENV)
    r1 = v1.reset(seeds=[5])
    b1 = [v1.step(actions) for _ in range(3)]
    v1.close()
    v2 = VecWrestlingEnv(1, backend="subproc", seed=5, learner_robot="a",
                         opponent_spec=OpponentSpec("stand_hold"), env_kwargs=SMOKE_ENV)
    r2 = v2.reset(seeds=[5])
    b2 = [v2.step(actions) for _ in range(3)]
    v2.close()
    assert np.allclose(r1.obs_a, r2.obs_a) and np.allclose(r1.priv, r2.priv)
    for a, b in zip(b1, b2):
        assert np.allclose(a.obs_a, b.obs_a) and np.allclose(a.priv, b.priv)
        assert np.allclose(a.rewards, b.rewards)
    assert resolve_backend(3) == "subproc" and resolve_backend(1) == "sequential"


# ------------------------------------------------------------ trainer/eval
def test_trainer_bc_warm_start_wiring(tmp_path):
    """A stage with ``bc_checkpoint`` loads actor weights at Trainer construction."""
    import dataclasses

    cfg = PPOConfig(n_envs=1, rollout_steps=16, seed=0)
    donor = make_policy(cfg)
    bc_path = tmp_path / "bc.pt"
    torch.save({"actor": {k: v.clone() for k, v in donor.actor.state_dict().items()}, "obs_dim": 92},
               bc_path)
    stage = dataclasses.replace(stage_by_key("C"), bc_checkpoint=str(bc_path))
    cur = Curriculum(stages=(stage,), start_index=0)
    t = Trainer(cfg, out_path=tmp_path / "ck.pt", curriculum=cur, backend="sequential",
                env_kwargs=SMOKE_ENV, verbose=0)
    try:
        assert t.bc_status is not None and t.bc_status["ok"], t.bc_status
        assert not t.bc_status["mismatched"]
        # a missing BC path degrades to a status note, not a crash
        stage2 = dataclasses.replace(stage_by_key("C"), bc_checkpoint=str(tmp_path / "nope.pt"))
        t2 = Trainer(cfg, out_path=tmp_path / "ck2.pt",
                     curriculum=Curriculum(stages=(stage2,), start_index=0),
                     backend="sequential", env_kwargs=SMOKE_ENV, verbose=0)
        try:
            assert t2.bc_status is not None and t2.bc_status["ok"] is False
        finally:
            t2.vec.close()
    finally:
        t.vec.close()


def test_trainer_resume_roundtrip(tmp_path):
    cfg = PPOConfig(rollout_steps=32, n_envs=2, seed=0)
    cur = Curriculum(start_index=2)
    t = Trainer(cfg, out_path=tmp_path / "ck.pt", curriculum=cur, backend="sequential",
                env_kwargs=SMOKE_ENV, ckpt_every=10_000, verbose=0)
    summary = t.run(64)
    assert summary["steps"] >= 64 and summary["bounds_bad"] == 0
    t2 = Trainer(cfg, out_path=tmp_path / "ck2.pt", backend="sequential",
                 env_kwargs=SMOKE_ENV, verbose=0)
    t2.load(summary["checkpoint"])
    assert t2.step_count == summary["steps"]
    assert t2.curriculum.stage.key == summary["stage"]
    assert t2.update_count == summary["iterations"]
    more = t2.run(summary["steps"] + 32)      # resume continues the log
    assert more["steps"] > summary["steps"]


def test_evaluate_is_deterministic(tmp_path):
    cfg = PPOConfig(n_envs=1, seed=0)
    policy = make_policy(cfg)
    stage = stage_by_key("C")
    out = evaluate(policy, stage=stage, env_kwargs=SMOKE_ENV, episodes=1, seed=11,
                   check_determinism=True)
    assert out["deterministic"] is True
    assert out["episodes"] == 1 and out["episode_steps"][0] > 0
    assert set(out["movement"]) >= {"standing_fraction", "ground_fraction",
                                    "knee_hand_contact_fraction", "mean_pelvis_z_m"}
    for row in out["table"]:
        assert row["cause"] in ("back", "oob", "timeout", "match_end")
        assert row["duration"] >= 0.0
