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
