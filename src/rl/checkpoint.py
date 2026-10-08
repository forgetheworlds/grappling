"""Checkpoints: atomic save/load of policy, optimizer, curriculum, RNG + stats.

Checkpoint format (``FORMAT_VERSION`` = 1, a plain ``torch.save`` dict)::

    {
      "format_version": 1,
      "created": "<iso8601>",
      "config":  {...PPOConfig.as_dict()...},
      "policy":  <ActorCritic.state_dict()>,
      "optimizer": <Adam.state_dict()> | None,
      "state":   {"step_count": int, "update_count": int, "target_steps": int,
                  "curriculum": {...}, "stats": {...}, "stage": {...}},
      "extra":   {"env": {...}, "backend": str, "n_envs": int, ...},
      "rng":     {"torch": Tensor, "numpy": tuple, "python": tuple},
    }

Writes are atomic: a temporary file in the target directory is fsynced and then
``os.replace``d over the destination, so a crash never leaves a torn file.

BC warm start: :func:`warm_start_from_bc` loads *actor* weights from a phase-3
teacher artifact.  Accepted shapes: a raw ``state_dict``, or a dict with any of
``"actor"``/``"policy"``/``"state_dict"``/``"model"`` holding one; parameter-name
prefixes ``module.``/``actor.``/``policy.``/``model.`` are stripped.  Shape
mismatches are reported and skipped (``strict=True`` turns them into errors).
"""

from __future__ import annotations

import datetime as _dt
import os
import random
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 1


# --------------------------------------------------------------------- RNG
def rng_state() -> dict:
    """Snapshot of torch / numpy / python RNG states (for resume)."""
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state: dict) -> None:
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("python") is not None:
        random.setstate(state["python"])


# ---------------------------------------------------------------- save/load
def save_checkpoint(path, *, policy=None, optimizer=None, cfg=None, state=None,
                    extra=None) -> str:
    """Atomically write a checkpoint; returns the path written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "config": None if cfg is None else (cfg.as_dict() if hasattr(cfg, "as_dict") else dict(cfg)),
        "policy": None if policy is None else policy.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "state": dict(state or {}),
        "extra": dict(extra or {}),
        "rng": rng_state(),
    }
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    return str(p)


def load_checkpoint(path, map_location: str = "cpu") -> dict:
    """Load a checkpoint dict (validates the format version)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    ckpt = torch.load(p, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "format_version" not in ckpt:
        raise ValueError(f"{p} is not an rl checkpoint (missing format_version)")
    if int(ckpt["format_version"]) != FORMAT_VERSION:
        raise ValueError(f"{p}: format_version {ckpt['format_version']} != {FORMAT_VERSION}")
    return ckpt


def apply_checkpoint(ckpt: dict, *, policy=None, optimizer=None) -> dict:
    """Load weights/optimizer into live objects; returns the trainer ``state``."""
    if policy is not None and ckpt.get("policy") is not None:
        policy.load_state_dict(ckpt["policy"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return dict(ckpt.get("state") or {})


# ------------------------------------------------------------- BC warm start
_PREFIXES = ("module.", "actor.", "policy.", "model.")


def _strip_prefix(key: str) -> str:
    out = key
    changed = True
    while changed:
        changed = False
        for pre in _PREFIXES:
            if out.startswith(pre):
                out = out[len(pre):]
                changed = True
    return out


def _candidate_state_dict(obj) -> dict:
    if not isinstance(obj, dict):
        raise ValueError(f"unsupported BC artifact type {type(obj).__name__}")
    for key in ("actor", "policy", "state_dict", "model", "weights"):
        v = obj.get(key)
        if isinstance(v, dict) and v and all(torch.is_tensor(t) for t in v.values()):
            return v
    if obj and all(torch.is_tensor(t) for t in obj.values()):
        return obj
    raise ValueError("no tensor state_dict found in BC artifact "
                     f"(keys: {sorted(map(str, obj.keys()))[:8]})")


def warm_start_from_bc(policy, path, *, strict: bool = False) -> dict:
    """Load actor weights from a phase-3 teacher/BC artifact into ``policy.actor``.

    Returns a status dict; never partially applies a mismatched tensor.
    """
    p = Path(path)
    status = {"path": str(p), "ok": False, "loaded": [], "missing": [], "mismatched": [],
              "unexpected": [], "note": ""}
    if not p.exists():
        status["note"] = "file not found"
        if strict:
            raise FileNotFoundError(p)
        return status
    try:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        src = _candidate_state_dict(obj)
    except Exception as exc:
        status["note"] = f"load failed: {exc!r}"
        if strict:
            raise
        return status
    dst = policy.actor.state_dict()
    remapped = {_strip_prefix(k): v for k, v in src.items()}
    new_state = dict(dst)
    for key, tensor in remapped.items():
        if key not in dst:
            status["unexpected"].append(key)
            continue
        if tuple(tensor.shape) != tuple(dst[key].shape):
            status["mismatched"].append(f"{key}: {tuple(tensor.shape)} != {tuple(dst[key].shape)}")
            continue
        new_state[key] = tensor
        status["loaded"].append(key)
    status["missing"] = [k for k in dst if k not in status["loaded"]]
    if status["mismatched"] and strict:
        raise ValueError(f"BC warm start shape mismatch: {status['mismatched']}")
    policy.actor.load_state_dict(new_state)
    status["ok"] = len(status["loaded"]) > 0
    status["note"] = (f"loaded {len(status['loaded'])} tensors; "
                      f"missing {len(status['missing'])}, mismatched {len(status['mismatched'])}, "
                      f"unexpected {len(status['unexpected'])}")
    return status


if __name__ == "__main__":  # self-check
    import tempfile

    from .net import ActorCritic, NetConfig

    torch.manual_seed(0)
    policy = ActorCritic(92, 92 + 162, act_dim=29, cfg=NetConfig(hidden=(32, 32)))
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    state = {"step_count": 123, "curriculum": {"index": 2, "steps_in_stage": 7}}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.pt"
        save_checkpoint(path, policy=policy, optimizer=opt, cfg={"lr": 3e-4}, state=state)
        assert not list(Path(td).glob("*.tmp*")), "temp file left behind"
        before = {k: v.clone() for k, v in policy.state_dict().items()}
        with torch.no_grad():
            policy.actor.trunk[-1].bias += 1.0
        ckpt = load_checkpoint(path)
        st = apply_checkpoint(ckpt, policy=policy, optimizer=opt)
        assert st["step_count"] == 123 and st["curriculum"]["index"] == 2
        for k, v in before.items():
            assert torch.allclose(policy.state_dict()[k], v), k
        # BC warm start from a raw (prefixed) state dict
        bc_path = Path(td) / "bc.pt"
        torch.save({"model": {f"actor.{k}": v for k, v in policy.actor.state_dict().items()}}, bc_path)
        status = warm_start_from_bc(policy, bc_path)
        assert status["ok"] and not status["mismatched"], status
    print("rl.checkpoint self-check OK:", {"format": FORMAT_VERSION})
