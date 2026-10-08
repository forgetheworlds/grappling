"""Sagittal-mirror maps and the PPO mirror loss (source A, arXiv:2404.19173).

Source A trains its SaW controller with PPO "extended with a mirror loss, to
encourage symmetry in policy behavior".  The paper does not print the term, so
the *form* below is ours (stated exactly, not guessed at):

.. math::

    L = \\frac{\\mathbb{E}_{s,j} (z_j(s) - \\tilde z_j(s))^2}
             {\\mathbb{E}_{s,j} z_j(s)^2 + \\epsilon},
    \\qquad \\tilde z(s) = M\\, \\mu(M s)

where ``M`` is the sagittal mirror map on the 29 joint-space action means
``mu`` (pre-tanh, so the loss is linear in the quantity the Gaussian parameterises
and ``tanh`` commutes with ``M``), ``M s`` is the mirrored actor observation and
the normalisation makes ``L`` dimensionless and O(1) at initialisation.  A policy
that is exactly mirror-symmetric has ``L = 0``; an action mean that has no
mirror partner at all (``z = e_0``, whose partner index is flipped in sign or
swapped) gives ``L = 2``.

The mirror maps themselves are *derived*, not guessed:

* joint vector: swap ``left_*`` <-> ``right_*`` and negate the roll/yaw joints
  (a reflection reverses the handedness of a rotation about ``x`` or ``z``);
  an involution, verified geometrically against the real model in
  ``tests/solo/test_lit_reward.py`` (mirroring a perturbed qpos puts every
  ``left`` body exactly where the mirrored ``right`` body is, to 1e-5 m);
* actor observation (``solo.obs.ACTOR_LAYOUT``): velocities and gravity flip
  their lateral/yaw components, the joint blocks use the joint map, the command
  flips ``vy``/``wz``, and stance/skill/phase are invariant.

Flagged OFF by default: the trainer only calls :func:`mirror_step` when
``--mirror-loss-coef > 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .lit import mirror_index_sign
from .obs import ACTOR_LAYOUT
from .scene import N_JOINTS, load_solo_model

_EPS = 1e-6
_MAPS: dict[int, tuple[np.ndarray, np.ndarray]] = {}

#: per-block mirror rule for the actor observation: sign vector, or None for the
#: joint blocks (which use the joint map) / the invariant blocks (all +1)
_BLOCK_SIGNS: dict[str, tuple[float, ...]] = {
    "base_linvel_local": (1.0, -1.0, 1.0),
    "base_angvel_local": (-1.0, 1.0, -1.0),
    "gravity_local": (1.0, -1.0, 1.0),
    "cmd_vel": (1.0, -1.0, -1.0),
    "cmd_stance": (1.0, 1.0),
}
_JOINT_BLOCKS = ("joint_pos_rel", "joint_vel", "prev_action")
_BLOCKS = dict(ACTOR_LAYOUT)

# every declared sign vector must match its block's width (layout drift guard)
for _name, _sgn in _BLOCK_SIGNS.items():
    _start, _stop = _BLOCKS[_name]
    if len(_sgn) != _stop - _start:  # pragma: no cover - import-time contract
        raise ValueError(f"mirror sign vector for {_name!r} has {len(_sgn)} entries, "
                         f"the layout block has {_stop - _start}")


def mirror_maps(model=None) -> tuple[np.ndarray, np.ndarray]:
    """(index, sign) of the joint mirror map, cached per model instance."""
    m = load_solo_model() if model is None else model
    key = id(m)
    if key not in _MAPS:
        _MAPS[key] = mirror_index_sign(m)
    return _MAPS[key]


def _slices() -> dict[str, slice]:
    return {name: slice(start, stop) for name, (start, stop) in ACTOR_LAYOUT}


def mirror_joint_vector(x, model=None) -> np.ndarray:
    """Mirror a joint-indexed vector ``(..., 29)`` (actions, joint obs blocks)."""
    idx, sign = mirror_maps(model)
    a = np.asarray(x, dtype=np.float64)
    if a.shape[-1] != N_JOINTS:
        raise ValueError(f"expected last dim {N_JOINTS}, got {a.shape}")
    return a[..., idx] * sign


def mirror_actor_obs(obs, model=None) -> np.ndarray:
    """Mirror an actor observation ``(..., ACTOR_DIM)``."""
    a = np.asarray(obs, dtype=np.float64)
    total = ACTOR_LAYOUT[-1][1][1]
    if a.shape[-1] != total:
        raise ValueError(f"expected last dim {total}, got {a.shape}")
    out = a.copy()
    for name, sl in _slices().items():
        if name in _JOINT_BLOCKS:
            out[..., sl] = mirror_joint_vector(a[..., sl], model)
        elif name in _BLOCK_SIGNS:
            out[..., sl] = a[..., sl] * np.asarray(_BLOCK_SIGNS[name])
        elif name == "lead_leg":
            out[..., sl] = -a[..., sl]
        # skill_onehot / phase: mirror-invariant
    return out


def _mirror_torch_joints(x: torch.Tensor, model=None) -> torch.Tensor:
    idx, sign = mirror_maps(model)
    i = torch.as_tensor(idx, dtype=torch.long, device=x.device)
    s = torch.as_tensor(sign, dtype=x.dtype, device=x.device)
    return x[..., i] * s


def mirror_actor_obs_torch(obs: torch.Tensor, model=None) -> torch.Tensor:
    """Torch twin of :func:`mirror_actor_obs` (batch leading dims allowed)."""
    out = obs.clone()
    for name, sl in _slices().items():
        if name in _JOINT_BLOCKS:
            out[..., sl] = _mirror_torch_joints(obs[..., sl], model)
        elif name in _BLOCK_SIGNS:
            s = torch.as_tensor(_BLOCK_SIGNS[name], dtype=obs.dtype, device=obs.device)
            out[..., sl] = obs[..., sl] * s
        elif name == "lead_leg":
            out[..., sl] = -obs[..., sl]
    return out


def mirror_loss(actor, obs: torch.Tensor, model=None) -> torch.Tensor:
    """Source A's symmetry loss (form documented in the module docstring).

    ``actor`` is ``rl.net.Actor`` (``actor.mean`` is the pre-tanh Gaussian mean).
    """
    z = actor.mean(obs)
    z_mirror = _mirror_torch_joints(actor.mean(mirror_actor_obs_torch(obs, model)),
                                    model)
    diff = (z - z_mirror) ** 2
    denom = z.detach() ** 2
    return diff.mean() / (denom.mean() + _EPS)


@dataclass
class MirrorStep:
    """Result of one auxiliary mirror gradient step (for logging/tests)."""

    loss: float
    grad_norm: float
    coef: float


def mirror_step(net, optimizer, obs, coef: float, grad_clip: float = 0.5,
                model=None) -> MirrorStep:
    """One auxiliary optimizer step on ``coef * mirror_loss``.

    Applied by ``SoloTrainer`` after ``ppo_update`` on the same rollout batch
    (same optimizer, same learning-rate schedule) when ``--mirror-loss-coef`` is
    non-zero.  With ``coef == 0`` it is a no-op that returns loss 0.0.
    ``net`` is an ``rl.net.ActorCritic`` (its ``.actor`` is used) or an
    ``Actor`` directly.
    """
    if float(coef) == 0.0:
        return MirrorStep(loss=0.0, grad_norm=0.0, coef=0.0)
    actor = getattr(net, "actor", net)
    x = torch.as_tensor(np.asarray(obs, dtype=np.float32))
    loss = float(coef) * mirror_loss(actor, x, model)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), float(grad_clip)))
    optimizer.step()
    return MirrorStep(loss=float(loss.detach()), grad_norm=norm, coef=float(coef))


if __name__ == "__main__":  # self-check
    from rl.net import Actor, NetConfig

    from .obs import ACTOR_DIM

    o = np.zeros(ACTOR_DIM, np.float64)
    o[3:6] = [0.1, 0.2, 0.3]                 # base angvel
    o[9:38] = np.arange(N_JOINTS) * 1e-3     # joint pos
    o[96:99] = [0.5, -0.5, 0.25]             # cmd vel
    o[101 + 8] = -1.0                        # lead leg
    m1 = mirror_actor_obs(o)
    assert np.allclose(mirror_actor_obs(m1), o), "obs mirror is not an involution"
    a = np.arange(N_JOINTS, dtype=float)
    assert np.allclose(mirror_joint_vector(mirror_joint_vector(a)), a)
    print("obs mirror: cmd_vel ->", np.round(m1[96:99], 3),
          "| angvel ->", np.round(m1[3:6], 3),
          "| joint0 ->", round(float(m1[9]), 4), "joint6 ->", round(float(m1[15]), 4),
          "sign", float(mirror_maps()[1][0]))

    torch.manual_seed(0)
    actor = Actor(ACTOR_DIM, N_JOINTS, cfg=NetConfig(hidden=(8,)))
    x = torch.as_tensor(np.tile(o, (4, 1)), dtype=torch.float32)
    with torch.no_grad():
        for p in actor.trunk:
            if hasattr(p, "weight"):
                p.weight.zero_()
            if hasattr(p, "bias"):
                p.bias.zero_()
    print("loss at perfectly symmetric (constant) policy:", float(mirror_loss(actor, x)))
    with torch.no_grad():
        actor.trunk[-1].bias[0] = 1.0      # no mirror partner at all
    print("loss with z = e_0:", float(mirror_loss(actor, x)))
    print("solo.mirror self-check OK")
