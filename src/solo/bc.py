"""Behaviour cloning of the retargeted references (S6 precursor: the pose prior).

Dataset
-------
(observation, action) pairs are built from the shipped retargeted tracks
(``data/refs_video/*.npz`` from the operator video, ``data/refs/*.npz`` from
GrappleMap), ``qpos_a`` (T,36) at 50 Hz -- already 2.5 Hz low-passed,
continuous-rotation-vector base-filtered and foot-contact-anchored upstream.

**Action mapping (exact).**  The env action is 29 joint-position targets in rad,
``absolute`` mode: ``ctrl = clip(mid + half * u)`` with ``mid``/``half`` the
actuator-ctrlrange centre/half-range.  The reference action is the PD target a
tracking controller would hold over the control interval ``[t, t+dt]``, i.e. the
reference joint angles ONE FRAME AHEAD:

    ctrl_k = clip(q_ref[k+1, 7:36], lo, hi)          # rad, absolute targets
    u_k    = (ctrl_k - mid) / half                   # the unit action the net emits

Rationale: the action is applied for the full 20 ms step, so the pose the servo
should reach by the end of the interval is ``q_ref(t+dt)``; the same-frame map
would be a near-identity copy of ``joint_pos_rel`` (degenerate), while the
one-step lookahead needs the velocity information and is a genuine tracking law.
Residual mode (``base + residual_scale*tanh(z)``, +/-0.5 rad) cannot express the
references -- max joint excursion from the stand keyframe is 2.79-2.89 rad and
0 entries fall outside ctrlrange -- so absolute targets are the only faithful
mapping.

Observation
-----------
The BC input is the FIRST 96 dims of the actor observation
(``solo.obs.ACTOR_LAYOUT``): ``base_linvel_local``, ``base_angvel_local``,
``gravity_local``, ``joint_pos_rel``, ``joint_vel``, ``prev_action``.  These are
exactly the fields computable from the reference alone (finite differences of
``qpos_a``; base-frame rotation uses the pelvis quaternion).  The remaining 19
dims (``cmd_vel``, ``cmd_stance``, skill one-hot, lead leg, phase) require the
command channel / simulation and are CONSTANT in this corpus, so they are
excluded (and recorded as ``actor_slice`` in the artifacts).  Nothing in the
dataset requires the simulation: contacts, markers, CoM and the privileged block
are never used.

Split
-----
Per reference, TIME-based: pair frames ``[0, n_tr)`` train, ``[n_tr, n)`` val with
``n_tr = int(n * train_frac)`` (0.7).  No frame index appears in both sets; the
ranges are contiguous and disjoint by construction and asserted in the tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .commands import Command
from .imitation import REFERENCE_DIRS, reference_velocities
from .obs import ACTOR_DIM, ObsContext, actor_obs
from .scene import N_JOINTS, load_solo_model, stand_frame, ctrl_range
from rl.net import Actor, NetConfig

REPO = Path(__file__).resolve().parents[2]
BC_DIR = REPO / "data" / "solo" / "bc"

#: reference-computable prefix of the actor observation
ACTOR_SLICE = (0, 96)
OBS_DIM = ACTOR_SLICE[1] - ACTOR_SLICE[0]

#: default corpus: the 12 retargeted tracks of the OPERATOR'S reference video
#: (``data/refs_video/*``, verified against ``retarget_summary.json`` -- primary),
#: plus the two GrappleMap techniques shipped in the same format (comparison).
VIDEO_REFS: tuple[str, ...] = (
    "stance_hold", "stance_widen_step", "level_change_full", "level_change_fast",
    "shot_entry_full",     # SHOTS take, entry window 422.8-431.0 s (knee-to-mat)
    "shot_recover",        # SHOTS take, 432.5-437.0 s
    "knee_sprawl_entry", "knee_sprawl_entry2", "knee_sprawl_hold",
    "knee_sprawl_recover",
    "stalk_shuffle", "circle_step",
)
GRAPPLEMAP_REFS: tuple[str, ...] = ("DOUBLE_LEG", "STANCE")
DEFAULT_REFS: tuple[str, ...] = VIDEO_REFS + GRAPPLEMAP_REFS


# ----------------------------------------------------------------- reference
@dataclass
class ReferenceTrack:
    """One retargeted reference (qpos_a of one .npz)."""

    name: str
    source: str
    qpos: np.ndarray          # (T, 36)
    t: np.ndarray             # (T,)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.t.size)


def reference_path(source) -> Path:
    p = Path(source)
    if p.exists():
        return p
    for d in REFERENCE_DIRS:
        cand = d / f"{source}.npz"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"reference {source!r} not found in {REFERENCE_DIRS}")


def load_reference(source) -> ReferenceTrack:
    path = reference_path(source)
    z = np.load(path, allow_pickle=True)
    meta = {}
    if "meta" in z:
        try:
            meta = json.loads(str(z["meta"]))
        except (TypeError, ValueError):
            meta = {}
    return ReferenceTrack(name=path.stem, source=str(path),
                          qpos=np.asarray(z["qpos_a"], dtype=np.float64),
                          t=np.asarray(z["t"], dtype=np.float64), meta=meta)


# --------------------------------------------------------------------- pairs
@dataclass
class Pairs:
    """(observation, action) pairs of one reference (time-ordered)."""

    name: str
    obs: np.ndarray        # (N, 96) float32
    ctrl: np.ndarray       # (N, 29) float64 -- absolute joint targets (rad)
    unit: np.ndarray       # (N, 29) float64 -- (ctrl - mid)/half, in [-1, 1]
    frame: np.ndarray      # (N,) int64 -- observation frame index in the track
    phase: np.ndarray      # (N,) float32 -- actor obs phase dim (offset 114)
    skill: int = -1        # argmax of the actor obs skill one-hot
    clipped: int = 0       # entries that needed ctrlrange clipping

    def __len__(self) -> int:
        return int(self.obs.shape[0])


def _action_arrays(qpos: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                   mid: np.ndarray, half: np.ndarray, lookahead: int):
    """(ctrl_all, unit_all) per FRAME: clip(q_ref[i]) and its unit action."""
    joints = np.asarray(qpos, dtype=np.float64)[:, 7:36]
    ctrl = np.clip(joints, lo, hi)
    unit = (ctrl - mid) / half
    return ctrl, unit


def pairs_from_qpos(qpos: np.ndarray, t: np.ndarray, *, name: str = "hand",
                    model=None, lookahead: int = 1,
                    cmd: Command | None = None) -> Pairs:
    """Build the dataset pairs of one reference track (see module docstring).

    ``cmd`` is the constant command the corpus is trained under.  The default
    (:data:`solo.commands.DEFAULT_COMMAND`) keeps every command field -- and in
    particular the actor phase (offset 114) -- constant; the values are kept in
    ``Pairs.phase``/``Pairs.skill`` for the phase-ambiguity check.
    """
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1, 36)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    assert qpos.shape[0] == t.size, (qpos.shape, t.size)
    assert 1 <= lookahead < qpos.shape[0], lookahead
    cmd = cmd or Command()
    m = model if model is not None else load_solo_model()
    lo, hi = ctrl_range(m)
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    q_stand, _ = stand_frame(m)

    ctrl_all, unit_all = _action_arrays(qpos, lo, hi, mid, half, lookahead)
    clipped = int(np.sum((qpos[:, 7:36] < lo) | (qpos[:, 7:36] > hi)))

    # pair k: observation at frame k, action whose target is frame k+lookahead
    T = qpos.shape[0]
    N = T - lookahead
    ctrl = ctrl_all[lookahead:]
    unit = unit_all[lookahead:]
    # prev_action k = unit all of the previous pair (target frame k); frame 0 -> zeros
    if T > lookahead + 1:
        prev = np.vstack([np.zeros((1, N_JOINTS)), unit_all[lookahead:-1]])[:N]
    else:
        prev = np.zeros((N, N_JOINTS))

    lin, ang, jvel = reference_velocities(qpos, t)
    obs = np.zeros((N, OBS_DIM), dtype=np.float32)
    phase = np.zeros(N, dtype=np.float32)
    for k in range(N):
        R = _quat_to_R(qpos[k, 3:7])
        ctx = ObsContext(
            base_linvel_local=R.T @ lin[k],
            base_angvel_local=R.T @ ang[k],
            gravity_local=R.T @ np.array([0.0, 0.0, -1.0]),
            joint_pos_rel=qpos[k, 7:36] - q_stand[7:36],
            joint_vel=jvel[k],
            prev_action=prev[k],
            cmd=cmd,
        )
        full = actor_obs(ctx)
        obs[k] = full[ACTOR_SLICE[0]:ACTOR_SLICE[1]]
        phase[k] = full[ACTOR_DIM - 1]
    return Pairs(name=str(name), obs=obs, ctrl=ctrl, unit=unit,
                 frame=np.arange(N, dtype=np.int64), phase=phase,
                 skill=int(cmd.skill_id), clipped=clipped)


def _quat_to_R(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ------------------------------------------------------------------- dataset
@dataclass
class Dataset:
    """Concatenated pairs of several references + the per-reference time split."""

    obs: np.ndarray          # (N, 96) float32
    ctrl: np.ndarray         # (N, 29) float64
    unit: np.ndarray         # (N, 29) float64
    ref_id: np.ndarray       # (N,) int64
    frame: np.ndarray        # (N,) int64
    phase: np.ndarray        # (N,) float32 -- actor phase dim (offset 114)
    names: tuple[str, ...]
    split: dict[str, dict]   # name -> {"n", "train": [a, b), "val": [b, n)}
    train_mask: np.ndarray   # (N,) bool
    val_mask: np.ndarray     # (N,) bool
    mid: np.ndarray
    half: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    stand_ctrl: np.ndarray
    train_frac: float = 0.7
    skill: int = -1

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def as_meta(self) -> dict:
        return {
            "names": list(self.names),
            "split": self.split,
            "train_frac": float(self.train_frac),
            "n": int(len(self)),
            "n_train": int(self.train_mask.sum()),
            "n_val": int(self.val_mask.sum()),
            "actor_slice": list(ACTOR_SLICE),
            "mapping": "ctrl_k = clip(q_ref[k+1, 7:36], lo, hi); "
                       "u_k = (ctrl_k - mid)/half",
            "phase": "actor obs offset 114, constant in this corpus "
                     "(see bc.corpus_phase_report)",
        }


def build_dataset(refs: tuple[str, ...] | list[str] = DEFAULT_REFS,
                  train_frac: float = 0.7, model=None,
                  lookahead: int = 1, cmd: Command | None = None) -> Dataset:
    """Build the corpus with a per-reference TIME split (train first 70%)."""
    if not 0.1 <= train_frac <= 0.9:
        raise ValueError("train_frac must be in [0.1, 0.9]")
    cmd = cmd or Command()
    m = model if model is not None else load_solo_model()
    lo, hi = ctrl_range(m)
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    q_stand, c_stand = stand_frame(m)

    obs, ctrl, unit, phase = [], [], [], []
    ref_id, frame = [], []
    names: list[str] = []
    split: dict[str, dict] = {}
    train_mask, val_mask = [], []
    for rid, src in enumerate(refs):
        track = load_reference(src)
        pr = pairs_from_qpos(track.qpos, track.t, name=track.name, model=m,
                             lookahead=lookahead, cmd=cmd)
        n = len(pr)
        n_tr = int(n * train_frac)
        n_tr = max(1, min(n - 1, n_tr))
        names.append(track.name)
        split[track.name] = {
            "n": int(n), "train": [0, int(n_tr)], "val": [int(n_tr), int(n)],
            "pairs_accounted": int(n),
            "source": track.source,
            "duration_s": round(float(track.t[-1] - track.t[0]), 3),
            "frames_src": int(len(track)),
        }
        obs.append(pr.obs)
        ctrl.append(pr.ctrl)
        unit.append(pr.unit)
        phase.append(pr.phase)
        ref_id.append(np.full(n, rid, dtype=np.int64))
        frame.append(pr.frame)
        tr = np.zeros(n, dtype=bool)
        tr[:n_tr] = True
        train_mask.append(tr)
        val_mask.append(~tr)
    return Dataset(
        obs=np.concatenate(obs), ctrl=np.concatenate(ctrl),
        unit=np.concatenate(unit), phase=np.concatenate(phase),
        ref_id=np.concatenate(ref_id), frame=np.concatenate(frame),
        names=tuple(names), split=split,
        train_mask=np.concatenate(train_mask), val_mask=np.concatenate(val_mask),
        mid=mid, half=half, lo=lo, hi=hi, stand_ctrl=c_stand,
        train_frac=float(train_frac), skill=int(cmd.skill_id))


def split_overlap(ds: Dataset) -> dict:
    """Index-level overlap proof: per-reference ranges + pairwise disjointness.

    Returns ``{"per_ref": {name: {...}}, "overlap_pairs": 0, "overlapping": []}``
    where an overlapping entry is a ``(ref, frame)`` tuple present in both masks.
    """
    key = np.stack([ds.ref_id, ds.frame], axis=1)
    tr = {tuple(k) for k in key[ds.train_mask]}
    va = {tuple(k) for k in key[ds.val_mask]}
    return {"per_ref": ds.split,
            "train_frames": len(tr), "val_frames": len(va),
            "overlapping": sorted(tr & va)}


# ------------------------------------------------------- phase / ambiguity
def corpus_phase_report(ds: Dataset) -> dict:
    """Is the actor phase field (offset 114) populated for these references?

    Measured, per reference: the phase value of every built pair.  The corpus is
    built with the constant :data:`solo.commands.DEFAULT_COMMAND` (skill
    ``STANCE``), and ``solo.env`` only advances its phase clock while the command
    skill is ``SHOT_DOUBLE_LEG`` (``env._make_ctx``: ``phase = self._shot_phase
    if skill is SHOT_DOUBLE_LEG else 0.0``) -- a camera clock, not a reference
    clock.  So a constant phase here means the observation cannot see *where in
    the reference* the robot is; :func:`ambiguity_report` measures the cost.
    """
    out = {}
    for i, name in enumerate(ds.names):
        p = ds.phase[ds.ref_id == i]
        out[name] = {"min": float(p.min()), "max": float(p.max()),
                     "n_unique": int(np.unique(p).size),
                     "constant": bool(np.all(p == p[0])),
                     "skill_onehot_argmax": int(ds.skill)}
    return out


def ambiguity_report(ds: Dataset, exclude: int = 25, material_rad: float = 0.25) -> dict:
    """How well do (joint pose, joint velocity) determine the action?

    For every frame, the nearest *non-local* frame of the SAME reference (all
    frames within +/- ``exclude`` steps are skipped, so the trivially identical
    neighbours are excluded) and the nearest frame of the whole TRAIN set
    (cKDTree) are found in the standardised 58-dim (``joint_pos_rel``,
    ``joint_vel``) space; the action difference to that neighbour is the action
    spread among frames that are indistinguishable in the observation -- i.e.
    the intrinsic ambiguity floor any deterministic obs->action map faces.
    ``material_rad`` counts frames whose nearest non-local neighbour demands a
    >0.25 rad different target.
    """
    from scipy.spatial import cKDTree

    feat = np.concatenate([ds.obs[:, 9:38], ds.obs[:, 38:67]], axis=1).astype(np.float64)
    mu = feat[ds.train_mask].mean(axis=0)
    sd = feat[ds.train_mask].std(axis=0)
    sd[sd < 1e-6] = 1.0
    Z = (feat - mu) / sd

    per_ref: dict[str, dict] = {}
    for i, name in enumerate(ds.names):
        idx = np.where(ds.ref_id == i)[0]
        n = len(idx)
        local = Z[idx]
        errs = np.zeros(n)
        # pairwise distances inside one reference (n <= ~900 -> n^2 is small)
        D = np.linalg.norm(local[:, None, :] - local[None, :, :], axis=2)
        D[np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) <= exclude] = np.inf
        have = np.isfinite(D).any(axis=1)
        b = np.argmin(D, axis=1)
        errs[have] = np.abs(ds.ctrl[idx][have] - ds.ctrl[idx[b][have]]).mean(axis=1)
        per_ref[name] = {
            "n": int(n),
            "act_err_median_rad": float(np.median(errs[have])) if have.any() else None,
            "act_err_mean_rad": float(np.mean(errs[have])) if have.any() else None,
            "act_err_p90_rad": float(np.percentile(errs[have], 90)) if have.any() else None,
            "frac_gt_material": float(np.mean(errs[have] > material_rad)) if have.any() else None,
        }

    # global: each val frame -> nearest TRAIN frame (any reference)
    val = np.where(ds.val_mask)[0]
    tr = np.where(ds.train_mask)[0]
    tree = cKDTree(Z[tr])
    _dist, j = tree.query(Z[val], k=1)
    act = np.abs(ds.ctrl[val] - ds.ctrl[tr[j]]).mean(axis=1)
    same_ref = ds.ref_id[val] == ds.ref_id[tr[j]]
    return {
        "exclude_frames": int(exclude),
        "material_rad": float(material_rad),
        "per_ref": per_ref,
        "val_to_train_nn": {
            "n": int(val.size),
            "act_err_median_rad": float(np.median(act)),
            "act_err_mean_rad": float(np.mean(act)),
            "act_err_p90_rad": float(np.percentile(act, 90)),
            "frac_gt_material": float(np.mean(act > material_rad)),
            "frac_nearest_is_same_ref": float(np.mean(same_ref)),
        },
    }


# --------------------------------------------------------------- baselines
def constant_errors(ds: Dataset, const_ctrl: np.ndarray, mask: np.ndarray):
    """Per-joint mean |const_ctrl - ctrl_target| (rad) on ``mask``."""
    err = np.abs(np.asarray(const_ctrl, np.float64)[None, :]
                 - ds.ctrl[mask])
    return err.mean(axis=0)


def stand_baseline_errors(ds: Dataset, mask: np.ndarray):
    """Constant stand-keyframe-target policy: |c_stand - ctrl_target|."""
    return constant_errors(ds, ds.stand_ctrl, mask)


def mean_baseline_errors(ds: Dataset, mask: np.ndarray):
    """Constant mean-of-train-target policy (a constant pose must not pass)."""
    mean_ctrl = ds.ctrl[ds.train_mask].mean(axis=0)
    return constant_errors(ds, mean_ctrl, mask)


# ------------------------------------------------------------------ policy
@dataclass
class BCPolicy:
    """Actor + the frozen action/observation contract of this dataset."""

    actor: Actor
    mid: np.ndarray
    half: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    obs_mean: np.ndarray
    obs_std: np.ndarray
    meta: dict = field(default_factory=dict)

    @torch.no_grad()
    def unit(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic unit action u = tanh(mean(obs)) (float32 in -> out)."""
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        single = x.ndim == 1
        if single:
            x = x.unsqueeze(0)
        x = (x - torch.as_tensor(self.obs_mean, dtype=torch.float32)) \
            / torch.as_tensor(self.obs_std, dtype=torch.float32)
        u = torch.tanh(self.actor.mean(x)).numpy()
        return u[0] if single else u

    def ctrl(self, obs: np.ndarray) -> np.ndarray:
        """Absolute joint-position targets (rad) the env would receive."""
        return self.mid + self.half * self.unit(obs)

    # ------------------------------------------------------------------ io
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": "solo.bc.v1",
            "state_dict": self.actor.state_dict(),
            "net": {"hidden": list(self.actor.cfg.hidden),
                    "activation": self.actor.cfg.activation,
                    "obs_dim": int(self.actor.obs_dim),
                    "act_dim": int(self.actor.act_dim)},
            "action_map": {"mode": "absolute", "mid": self.mid.tolist(),
                           "half": self.half.tolist(), "lo": self.lo.tolist(),
                           "hi": self.hi.tolist()},
            "obs": {"slice": list(ACTOR_SLICE),
                    "mean": self.obs_mean.tolist(), "std": self.obs_std.tolist()},
            "meta": self.meta,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "BCPolicy":
        ck = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = NetConfig(hidden=tuple(ck["net"]["hidden"]),
                        activation=ck["net"]["activation"])
        actor = Actor(int(ck["net"]["obs_dim"]), int(ck["net"]["act_dim"]), cfg)
        actor.load_state_dict(ck["state_dict"])
        actor.eval()
        am = ck["action_map"]
        return cls(actor=actor, mid=np.array(am["mid"]), half=np.array(am["half"]),
                   lo=np.array(am["lo"]), hi=np.array(am["hi"]),
                   obs_mean=np.array(ck["obs"]["mean"], np.float32),
                   obs_std=np.array(ck["obs"]["std"], np.float32),
                   meta=ck.get("meta", {}))


# ------------------------------------------------------------------ training
@dataclass
class BCTrainConfig:
    hidden: tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    epochs: int = 400
    batch_size: int = 128
    seed: int = 0
    weight_decay: float = 0.0
    eval_every: int = 20
    torch_threads: int = 2

    def as_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def train_bc(ds: Dataset, cfg: BCTrainConfig | None = None,
             verbose: bool = False) -> dict:
    """Supervised fit of ``ctrl`` from the 96-dim reference observation.

    Loss: MSE in RADIANS between the predicted and target joint targets (joints
    with a larger ctrlrange get proportionally larger weight, which is the
    physically meaningful scale).  Best-val epoch (MAE in rad) is returned.
    """
    from dataclasses import replace

    cfg = cfg or BCTrainConfig()
    torch.set_num_threads(int(cfg.torch_threads))
    torch.manual_seed(int(cfg.seed))
    rng = np.random.default_rng(int(cfg.seed))

    obs = torch.as_tensor(ds.obs, dtype=torch.float32)
    tgt = torch.as_tensor(ds.ctrl, dtype=torch.float32)
    mean = obs[torch.as_tensor(ds.train_mask)].mean(0)
    std = obs[torch.as_tensor(ds.train_mask)].std(0).clamp_min(1e-3)

    actor = Actor(OBS_DIM, N_JOINTS, NetConfig(hidden=tuple(cfg.hidden)))
    opt = torch.optim.Adam(actor.parameters(), lr=float(cfg.lr),
                           weight_decay=float(cfg.weight_decay))
    mid = torch.as_tensor(ds.mid, dtype=torch.float32)
    half = torch.as_tensor(ds.half, dtype=torch.float32)

    tr_idx = np.where(ds.train_mask)[0]
    va_idx = np.where(ds.val_mask)[0]
    tr_t = torch.as_tensor(tr_idx)
    va_t = torch.as_tensor(va_idx)

    def _mae(idx_t) -> float:
        actor.eval()
        with torch.no_grad():
            x = (obs[idx_t] - mean) / std
            pred = mid + half * torch.tanh(actor.mean(x))
            return float((pred - tgt[idx_t]).abs().mean().item())

    best = {"epoch": -1, "val": float("inf"), "state": None}
    history = []
    for epoch in range(int(cfg.epochs)):
        actor.train()
        order = rng.permutation(tr_idx)
        tot = 0.0
        nb = 0
        for s in range(0, len(order), int(cfg.batch_size)):
            b = torch.as_tensor(order[s:s + int(cfg.batch_size)])
            x = (obs[b] - mean) / std
            pred = mid + half * torch.tanh(actor.mean(x))
            loss = ((pred - tgt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        if (epoch + 1) % int(cfg.eval_every) == 0 or epoch == cfg.epochs - 1:
            actor.eval()
            with torch.no_grad():
                x = (obs[tr_t] - mean) / std
                tr_mae = float(((mid + half * torch.tanh(actor.mean(x)))
                                - tgt[tr_t]).abs().mean().item())
            va_mae = _mae(va_t)
            history.append({"epoch": epoch + 1, "train_mse": tot / max(nb, 1),
                            "train_mae_rad": tr_mae, "val_mae_rad": va_mae})
            if va_mae < best["val"]:
                best = {"epoch": epoch + 1, "val": va_mae,
                        "state": {k: v.clone() for k, v in actor.state_dict().items()}}
            if verbose:
                print(f"  epoch {epoch + 1:4d}  train_mse {tot / max(nb, 1):.5f}"
                      f"  train_mae {tr_mae:.4f}  val_mae {va_mae:.4f} rad")
    if best["state"] is not None:
        actor.load_state_dict(best["state"])
    actor.eval()
    policy = BCPolicy(actor=actor, mid=ds.mid, half=ds.half, lo=ds.lo, hi=ds.hi,
                      obs_mean=mean.numpy().astype(np.float32),
                      obs_std=std.numpy().astype(np.float32),
                      meta={"trained_on": list(ds.names), "config": cfg.as_dict(),
                            "best": {"epoch": best["epoch"],
                                     "val_mae_rad": best["val"]}})
    return {"policy": policy, "history": history, "best": {"epoch": best["epoch"],
                                                           "val_mae_rad": best["val"]}}


_JOINT_NAMES = None


def joint_names(model=None) -> tuple[str, ...]:
    """The 29 hinge names in actuator order (from the compiled model)."""
    global _JOINT_NAMES
    m = model if model is not None else load_solo_model()
    if _JOINT_NAMES is None:
        import mujoco

        names = []
        for i in range(1, m.njnt):  # joint 0 is the free joint
            names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))
        assert len(names) == N_JOINTS, names
        _JOINT_NAMES = tuple(names)
    return _JOINT_NAMES


# --------------------------------------------------------------- evaluation
def per_joint_errors(pred_ctrl: np.ndarray, target_ctrl: np.ndarray) -> np.ndarray:
    """(29,) mean |predicted - target| in rad over the samples given."""
    return np.abs(np.asarray(pred_ctrl, np.float64)
                  - np.asarray(target_ctrl, np.float64)).mean(axis=0)


def validation_report(policy: BCPolicy, ds: Dataset) -> dict:
    """Held-out (val) joint error vs both constant baselines, per joint."""
    mask = ds.val_mask
    bc = per_joint_errors(policy.ctrl(ds.obs[mask]), ds.ctrl[mask])
    stand = stand_baseline_errors(ds, mask)
    mean = mean_baseline_errors(ds, mask)
    return {
        "bc": {"per_joint_rad": bc.tolist(), "mean_rad": float(bc.mean())},
        "stand_baseline_rad": {"per_joint_rad": stand.tolist(),
                               "mean_rad": float(stand.mean())},
        "mean_baseline_rad": {"per_joint_rad": mean.tolist(),
                              "mean_rad": float(mean.mean())},
        "margins": {
            "vs_stand_factor": float(stand.mean() / bc.mean()),
            "vs_mean_factor": float(mean.mean() / bc.mean()),
            "reduction_vs_stand": float(1.0 - bc.mean() / stand.mean()),
        },
        "n_val": int(mask.sum()),
    }


def train_report(policy: BCPolicy, ds: Dataset) -> dict:
    """Train-mask error, for the overfit check."""
    mask = ds.train_mask
    bc = per_joint_errors(policy.ctrl(ds.obs[mask]), ds.ctrl[mask])
    return {"per_joint_rad": bc.tolist(), "mean_rad": float(bc.mean()),
            "n_train": int(mask.sum())}


# ------------------------------------------------------------------ rollout
def set_reference_initial_velocity(env, track: ReferenceTrack) -> None:
    """Write the finite-difference reference velocities into the env state.

    ``qpos_a`` carries no velocities; the dataset's first-frame values are the
    one-sided finite differences of the shipped track (same function the BC
    observations use), so the rollout starts from the reference's own state as
    far as the file defines it (position) plus its measured velocity.
    """
    lin, ang, jvel = reference_velocities(track.qpos, track.t)
    env.data.qvel[0:3] = lin[0]
    env.data.qvel[3:6] = ang[0]
    env.data.qvel[6:35] = jvel[0]
    import mujoco

    mujoco.mj_forward(env.model, env.data)


def rollout(policy: BCPolicy, track: ReferenceTrack | str, *,
            max_steps: int | None = None, seed: int = 0, model=None,
            track_sites: bool = True) -> dict:
    """Open-loop BC rollout on the reference's own initial state.

    No expert correction: the policy sees only the environment's own 96-dim
    observation.  The episode ends when the env terminates (fall/dorsal) or the
    reference ends.  Returns the tracking-error series and the honest verdict.
    """
    from .env import SoloEnv
    from .imitation import ImitationTargets, site_error_m, state_from_env

    tr = load_reference(track) if isinstance(track, str) else track
    env = SoloEnv(model, task="balance", action_mode="absolute", seed=seed,
                  jitter=False, horizon=max(8.0, len(tr) * 0.02 + 1.0))
    env.reset(pose=tr.qpos[0], jitter=False)
    set_reference_initial_velocity(env, tr)
    targets = ImitationTargets.from_reference(tr.source) if track_sites else None

    N = len(tr) - 1
    n_steps = N if max_steps is None else min(N, int(max_steps))
    rows = []
    obs = env.observation()
    terminated = truncated = False
    info = {}
    deviation_time = deviation_reason_txt = None
    for k in range(n_steps):
        u = policy.unit(obs["actor"][ACTOR_SLICE[0]:ACTOR_SLICE[1]])
        ctrl = env.ctrl_from_policy(u)
        obs, _reward, terminated, truncated, info = env.step(ctrl)
        sim_q = np.asarray(env.data.qpos[7:36], np.float64).copy()
        ref_q = tr.qpos[k + 1, 7:36]
        row = {
            "k": k + 1, "t": float(env.data.time),
            "joint_err": float(np.mean(np.abs(sim_q - ref_q))),
            "joint_err_max": float(np.max(np.abs(sim_q - ref_q))),
            "root_err": float(np.linalg.norm(env.data.qpos[:3] - tr.qpos[k + 1, :3])),
            "pelvis_z": float(env.data.qpos[2]),
            "ref_pelvis_z": float(tr.qpos[k + 1, 2]),
        }
        if targets is not None:
            from .imitation import DEFAULT_WEIGHTS, deviation_reason

            state = state_from_env(env)
            target = targets.at(k + 1)
            row["site_err"] = site_error_m(state, target)
            if deviation_reason_txt is None:
                dev = deviation_reason(state, target, DEFAULT_WEIGHTS)
                if dev is not None:
                    deviation_time, deviation_reason_txt = row["t"], dev
        rows.append(row)
        if terminated or truncated:
            break

    err = np.array([r["joint_err"] for r in rows]) if rows else np.zeros(0)
    root = np.array([r["root_err"] for r in rows]) if rows else np.zeros(0)
    site = np.array([r.get("site_err", np.nan) for r in rows]) if rows else np.zeros(0)
    n = len(rows)
    half = max(1, n // 2)
    from .env import RECOVERY_PELVIS_Z

    pelvis_min = float(min(r["pelvis_z"] for r in rows)) if n else None
    verdict = {
        "name": tr.name,
        "steps": n,
        "reference_steps": N,
        "completed_reference": bool(n >= N and not terminated),
        "fell": bool(terminated),
        "termination": info.get("termination"),
        "fall_time": float(rows[-1]["t"]) if (terminated and rows) else None,
        # a sag to a low pose without tripping the fall/dorsal detectors still
        # counts as a collapse (same threshold the env uses for "down")
        "collapsed": bool(pelvis_min is not None and pelvis_min < RECOVERY_PELVIS_Z),
        "pelvis_z_min": pelvis_min,
        # first time the RL-stage deviation predicate would have terminated the
        # episode (imitation.DEFAULT_WEIGHTS), and why
        "deviation_time": deviation_time,
        "deviation_reason": deviation_reason_txt,
        "joint_err_mean": float(err.mean()) if n else None,
        "joint_err_first_half": float(err[:half].mean()) if n else None,
        "joint_err_last_half": float(err[-half:].mean()) if n else None,
        "joint_err_final": float(err[-1]) if n else None,
        "root_err_mean": float(root.mean()) if n else None,
        "site_err_mean": float(np.nanmean(site)) if n else None,
        "site_err_max": float(np.nanmax(site)) if n else None,
        "rows": rows,
    }
    return verdict


# ------------------------------------------------------------------ artifacts
def save_dataset(ds: Dataset, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, obs=ds.obs, ctrl=ds.ctrl, unit=ds.unit, ref_id=ds.ref_id,
        frame=ds.frame, phase=ds.phase, names=np.array(ds.names),
        train_mask=ds.train_mask, val_mask=ds.val_mask, mid=ds.mid,
        half=ds.half, lo=ds.lo, hi=ds.hi, stand_ctrl=ds.stand_ctrl,
        meta=json.dumps(ds.as_meta()))


def load_dataset(path: Path) -> Dataset:
    z = np.load(Path(path), allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    return Dataset(obs=z["obs"], ctrl=z["ctrl"], unit=z["unit"], ref_id=z["ref_id"],
                   frame=z["frame"], phase=z["phase"],
                   names=tuple(str(n) for n in z["names"]),
                   split=meta["split"], train_mask=z["train_mask"],
                   val_mask=z["val_mask"], mid=z["mid"], half=z["half"],
                   lo=z["lo"], hi=z["hi"], stand_ctrl=z["stand_ctrl"],
                   train_frac=float(meta.get("train_frac", 0.7)),
                   skill=int(meta.get("skill", -1)))


if __name__ == "__main__":  # self-check (no training)
    from .imitation import ImitationState, site_error_m

    # exact dataset mapping on a hand snippet (lookahead: action targets k+1)
    q = np.zeros((3, 36))
    q[:, 2] = 0.7
    q[:, 3] = 1.0       # identity quaternion (w-first)
    q[1:, 7] = [0.4, 0.9]
    q[1:, 8] = -0.5
    t = np.array([0.0, 0.02, 0.04])
    pr = pairs_from_qpos(q, t, name="hand")
    m = load_solo_model()
    lo, hi = ctrl_range(m)
    q_stand, _ = stand_frame(m)
    assert np.array_equal(pr.ctrl[0], np.clip(q[1, 7:], lo, hi))
    assert not np.array_equal(pr.ctrl[0], np.clip(q[0, 7:], lo, hi))
    assert np.allclose(pr.obs[0, 9:38], (q[0, 7:] - q_stand[7:36]).astype(np.float32))
    assert np.allclose(pr.obs[1, 67:96], pr.unit[0], atol=1e-6)   # prev_action
    # reference corpus builds + split is disjoint
    ds = build_dataset(("STANCE", "DOUBLE_LEG"))
    ov = split_overlap(ds)
    assert not ov["overlapping"] and ov["train_frames"] + ov["val_frames"] == len(ds)
    # imitation terms are wired to the same FK
    st = ImitationState(joints=ds.ctrl[0])
    assert site_error_m(st, st) == 0.0
    print("solo.bc self-check OK:", {"pairs": len(ds), "split": ds.split,
                                     "clipped": pr.clipped})
