"""Core RL-module tests for the mandate path (moved, not rewritten).

``tests/test_rl.py`` was deleted with the two-robot modules it mostly covered
(``rl.obs``/``privileged``/``reward``/``scripted``/``curriculum``/``vec``/
``rollout``/``trainer``).  The tests below cover modules that *stay*
(``rl.net``, ``rl.ppo``, ``rl.checkpoint``) and need no environment, so they
were moved here verbatim -- same assertions, same hand-computed expectations --
with only the import header adapted (deleted-module imports dropped) plus two
fixture details:

* ``model`` is the solo scene (``solo.scene.load_solo_model``), the model
  ``rl.net`` is used with on the mandate path (29 actuators, slice(0, 29));
* ``PRIV_DIM`` is pinned locally as a plain width for the checkpoint fixtures.
  It was the two-robot critic width (``rl.privileged``, deleted); the value
  only sizes an ``ActorCritic`` whose roundtrip is being checked -- the solo
  critic width is ``solo.obs.PRIV_DIM`` and is covered by
  ``tests/solo/test_config_plumbing.py``.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rl.checkpoint import (FORMAT_VERSION, apply_checkpoint, load_checkpoint,  # noqa: E402
                           restore_rng, rng_state, save_checkpoint,
                           warm_start_from_bc)
from rl.net import (ActionMapper, ActorCritic, NetConfig,  # noqa: E402
                    count_out_of_bounds, robot_action_bounds)
from rl.ppo import PPOConfig, RolloutBatch, compute_gae, lr_at, ppo_update  # noqa: E402
from solo.scene import load_solo_model  # noqa: E402

#: actor/critic width used by the checkpoint fixtures below (see module docstring)
PRIV_DIM = 162


@pytest.fixture(scope="module")
def model():
    return load_solo_model()


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

def test_split_grad_clipping_frees_the_actor_from_value_error():
    """P0-3: a shared clip lets the value term decide the actor's step.

    The 2026-10-08 audit measured the value term's gradient at 38x the policy
    term's (289.0 vs 7.5) against a shared clip of 0.5 -- the actor moved ~600x
    less than the unclipped direction while the critic chased noisy targets.
    With separate clamps the actor's step is set by its own gradient norm, so
    the same batch (a large value error, a small policy signal) must move the
    actor further.  SGD (not Adam: Adam normalises per parameter and hides the
    gradient scale) makes the clipped gradient observable in the step size.
    """

    def run(clip_actor, clip_critic):
        net = ActorCritic(8, 12, act_dim=3, cfg=NetConfig(hidden=(16, 16)))
        opt = torch.optim.SGD(net.parameters(), lr=1e-2)
        cfg = PPOConfig(grad_clip_actor=clip_actor, grad_clip_critic=clip_critic)
        b = _toy_batch(T=5, N=2)
        b.returns = b.returns + 5000.0       # a value error 1000x the policy signal
        b.advantages = b.advantages * 0.01   # a small policy signal
        before = copy.deepcopy(net.actor.state_dict())
        ppo_update(net, opt, b, cfg, cfg.lr, torch.Generator().manual_seed(0))
        delta = sum(float(((net.actor.state_dict()[k] - v) ** 2).sum())
                    for k, v in before.items())
        return float(np.sqrt(delta))

    shared = run(None, None)
    split = run(0.5, 0.5)
    assert shared > 0.0 and split > shared, (shared, split)
    assert split > 3.0 * shared, (shared, split)


def test_split_critic_lr_moves_only_the_critic_faster():
    """A separate critic lr: the scheduled lr scales each group from its own base.

    v6c@100k read EV -3.58 with the next scheduled push IN the critic's obs, so
    the value function's failure is optimisation, not information.  The trainer
    builds two param groups (actor ratio 1.0, critic lr_critic/lr); the update
    must scale both by the schedule, not flatten them to one lr -- a flattened
    group would silently undo the split.
    """
    def run(lr_critic, lr_scheduled):
        torch.manual_seed(0)                     # identical nets across calls
        net = ActorCritic(8, 12, act_dim=3, cfg=NetConfig(hidden=(16, 16)))
        ratio = float(lr_critic) / 0.001
        opt = torch.optim.SGD(
            [{"params": list(net.actor.parameters()), "lr": 0.001, "lr_ratio": 1.0},
             {"params": list(net.critic.parameters()), "lr": lr_critic,
              "lr_ratio": ratio}], lr=0.001)
        cfg = PPOConfig()
        b = _toy_batch(T=5, N=2)
        before_a = copy.deepcopy(net.actor.state_dict())
        before_c = copy.deepcopy(net.critic.state_dict())
        ppo_update(net, opt, b, cfg, lr_scheduled, torch.Generator().manual_seed(0))
        da = float(np.sqrt(sum(float(((net.actor.state_dict()[k] - v) ** 2).sum())
                               for k, v in before_a.items())))
        dc = float(np.sqrt(sum(float(((net.critic.state_dict()[k] - v) ** 2).sum())
                               for k, v in before_c.items())))
        return da, dc, [g["lr"] for g in opt.param_groups]

    da1, dc1, lrs1 = run(0.001, 0.001)                 # shared
    da10, dc10, lrs10 = run(0.010, 0.001)              # critic 10x
    assert lrs1 == [0.001, 0.001], lrs1
    assert abs(lrs10[0] - 0.001) < 1e-12 and abs(lrs10[1] - 0.010) < 1e-12, lrs10
    assert dc10 > 3.0 * dc1, (dc1, dc10)               # the critic really moves more
    # the actor's step must NOT scale with the critic's lr.  A ~2% residual is
    # measured and not explained (no parameters are shared; the value term does
    # not enter the actor's loss) -- the tolerance guards the failure mode that
    # matters, a FLATTENED lr group, which would change it by ~10x.
    assert abs(da10 - da1) < 0.05 * max(da1, 1e-9), (da1, da10)
    # the schedule scales both groups from their own base
    _, _, lrs_half = run(0.010, 0.0005)
    assert abs(lrs_half[0] - 0.0005) < 1e-12, lrs_half
    assert abs(lrs_half[1] - 0.005) < 1e-12, lrs_half


def test_log_std_anneal_ramps_and_pins_the_noise():
    """P1-4: the behaviour noise must be settable (v5's never left its init)."""
    from solo.train import TrainConfig, log_std_at

    assert np.isnan(log_std_at(0, TrainConfig()))          # unset -> leave alone
    cfg = TrainConfig(steps=100_000, log_std_final=-2.5, log_std_anneal_steps=100_000)
    assert log_std_at(0, cfg) == pytest.approx(-1.0)       # the PPO init
    assert log_std_at(50_000, cfg) == pytest.approx(-1.75)
    assert log_std_at(100_000, cfg) == pytest.approx(-2.5)
    assert log_std_at(500_000, cfg) == pytest.approx(-2.5)  # clamped at the end
    net = ActorCritic(8, 12, act_dim=3, cfg=NetConfig(hidden=(16, 16)))
    net.set_log_std(-2.5)
    assert net.actor.log_std.detach().mean().item() == pytest.approx(-2.5)
    assert float(np.exp(net.actor.log_std.detach().mean().item())) == pytest.approx(0.0821, abs=1e-3)
    # and the update reports it (the monitor's acceptance reads sigma)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    stats = ppo_update(net, opt, _toy_batch(), PPOConfig(), 1e-3,
                       torch.Generator().manual_seed(0))
    # the update itself moves log_std (it is a parameter), so pin the REPORTING
    assert stats["log_std_mean"] == pytest.approx(-2.5, abs=0.05)
    assert stats["sigma_mean"] == pytest.approx(
        float(np.exp(stats["log_std_mean"])), rel=1e-6)
