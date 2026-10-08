"""S5 demo capture: teacher-stabilized execution of the operator's retargeted
penetration step (single G1), for BC/DAgger demonstrations.

The reference
-------------
``data/refs_video/shot_entry_full.npz`` -- the operator's own video
(yt_gBAhX5t-GW4, window 422.8-431.0 s) retargeted to a single G1 at 50 Hz:
"crouched stance -> step -> penetration -> knee on mat"
(``docs/references/yt_gBAhX5t-GW4_index.md`` §3).  It is preferred over the
GrappleMap-schematic ``data/refs/DOUBLE_LEG.npz`` because it is the technique
the operator wants copied, it is a single-robot track, its root translation is
reconstructed from foot-contact anchoring (so the penetration step travels
instead of running on a treadmill; meta ``net_travel_m``), and its posture
(crouch 0.60 m, tilt 36-72 deg) is closer to a physically holdable stance than
the schematic track (which dives to 0.33 m / 119 deg and ends prone).

The repair
----------
The entry track ends at the knee-down moment with both feet floating (a
monocular-depth artefact: the retargeted soles sit 13-18 cm up).  The demo
therefore executes a *repaired* track (``build_repaired_track``):

    [entry: the whole reference]                segment 0 (ref_time = entry t)
    [blend: 0.8 s min-jerk to the rise track]   segment 1 (ref_time = NaN)
    [recover: data/refs_video/shot_recover.npz] segment 2 (ref_time = recover t)
    [hold: 0.5 s of the recovered pose]         segment 3 (ref_time = NaN)

The rise material is the operator's own recovery window (432.5-437.0 s, "knee
off -> rise to crouched stance").  The blend exists because the two tracks come
from different windows (1.5 s apart) with a 0.237 m world offset between their
foot-contact anchors; it interpolates joints, base xy/z and the base
quaternion (slerp) with a min-jerk profile (zero velocity at both ends).

Capture
-------
The capture is **source-agnostic** (``CaptureSpec.source``):

* ``cem`` (primary) -- a sampling-based optimizer (CEM, direct dynamic
  retargeting): searches knot-interpolated residual joint targets that
  minimise the weighted landmark-site error against the video reference under
  the actual dynamics, penalised for falls, actuator-force saturation,
  loaded-foot slip and terminal-stance violation.  The best sequence is
  replayed deterministically as the demo; the budget, search dimension and
  convergence history are recorded in the provenance (a downstream reader can
  tell a retargeted trajectory from a passthrough: the reference's own
  passthrough score is recorded beside the optimised one).
* ``policy`` -- an external learned controller (RL tracking / BC): any
  callable ``policy(actor_obs(115,)) -> action(29,)`` (``--policy-module``).
* ``teacher`` -- the phase-3 stabilised teacher: **fallback / initialisation
  only, never the demo source of record** (measured weak expert: stay-up
  0.448-0.690 over 3 seeds, 1.64-2.60 m landmark error, governor active on
  92.9-99.2 % of ticks; and non-Markovian -- its integrators / governor alpha /
  rate-limiter memory are not in the actor observation, so its actions are not
  a function of the observable state: TeacherDataAudit 2026-10-08).

Both the CEM and the policy drive the same 50 Hz loop on the S1 solo scene
(``solo.scene``) with ten 2 ms physics substeps per tick, and every tick is
recorded: qpos/qvel, the commanded joint targets and their action-space image
(``src/solo/env.py`` action contract), the actor/privileged observations
(``src/solo/obs.py`` layout), contacts, foot loads and sole points, CoM, the
support-polygon margin, pelvis/tilt/pitch, saturation fraction and the
reference progress.  Initial states are perturbed (seeded joint noise + base
xy/yaw wobble) and/or start at a reference-time offset, so the dataset is not
one fixed animation.  Every demo ends in the measured holdable stance
(``solo.stance``) -- the operator rule that applies to every trained
behaviour; the checker's ``terminal_stance`` criterion enforces it.

Every episode is written to ``data/solo/demos/<name>/<episode>/`` as
``trace.npz`` (per-tick arrays), ``spec.npz`` (the reference-derived tracks the
checker aligns against), ``meta.json`` (seed, config, reference hashes,
reproduce command) and, with ``--check``, ``check.json`` (the executability
report).

This module imports mujoco lazily: ``plan_capture`` (the ``--dry-run`` path)
is pure metadata + numpy, so a dry run never touches the simulator.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .exec_check import (REFERENCE_SEGMENTS, SEG_BLEND, SEG_ENTRY, SEG_HOLD,
                         SEG_RECOVER, SEG_STAND, ExecSpec, RefTracks, Tol,
                         check_executability, hull2d, polygon_margin)

REPO = Path(__file__).resolve().parents[2]
DEMO_ROOT = REPO / "data" / "solo" / "demos"
DEFAULT_ENTRY_REF = REPO / "data" / "refs_video" / "shot_entry_full.npz"
DEFAULT_RECOVER_REF = REPO / "data" / "refs_video" / "shot_recover.npz"

#: measured on this host 2026-10-08 (4-core ARM, training service on all cores):
#: 106 control ticks in 1.48 s wall including per-tick observation/metrics.
#: CEM rollouts skip the observation build and run measurably faster; used only
#: for planning estimates, never for control.
DEFAULT_RATE_TICKS_PER_S = 72.0

DT = 0.02                      # 50 Hz control (solo.scene.STEP_DT)
PREFIX = "a_"                  # solo.scene robot prefix
SIDES = ("left", "right")
TOUCH_Z = 0.045                # contact rule (audit support_envelope.json)
KNEE_CONTACT_BAND = 0.02       # knee "arrived at depth" = entry min + this
RESET_JITTER_XY = 0.02         # m, at perturb_sigma = 0.01 (env reset jitter)
RESET_JITTER_YAW = 0.10        # rad, at perturb_sigma = 0.01
DEFAULT_ACTION_MODE = "absolute"
DEFAULT_RESIDUAL_SCALE = 0.5


# ---------------------------------------------------------------------------
# capture spec
# ---------------------------------------------------------------------------
@dataclass
class CaptureSpec:
    """Everything one demo-capture invocation needs (recorded verbatim)."""

    ref: str = str(DEFAULT_ENTRY_REF)
    recover_ref: str = str(DEFAULT_RECOVER_REF)
    role: str = "a"
    upto_s: float | None = None      # entry cut (None = the whole entry track)
    blend_s: float = 0.8
    recover_s: float | None = None   # recover cut (None = the whole track)
    stand_s: float = 1.2             # min-jerk rise onto the terminal stance
    stance_width: float = 0.30       # measured holdable stance (solo.stance)
    stance_height: float = 0.79
    hold_s: float = 0.5
    start_offset_s: float = 0.0      # start the episode at this reference time
    perturb_sigma: float = 0.0       # rad, joint noise (0 = clean start)
    seed: int = 0
    episodes: int = 1
    out_root: str = str(DEMO_ROOT)
    name: str | None = None
    touch_repair: str = "none"       # teacher fallback only: none | lead
    source: str = "cem"              # cem | policy | teacher
    policy_module: str = ""          # path/to/module.py:attr (source=policy)
    cem_samples: int = 450           # >= 10x the window search dimension
    cem_iters: int = 10              # upper bound; early stop on stall
    cem_knot_ticks: int = 25
    cem_window: int = 50             # receding-horizon window (ticks; 0=whole)
    cem_stride: int = 25             # replan stride (ticks)
    cem_sigma: float = 0.12
    cem_theta_max: float = 0.8
    cem_mask: str = "legs_waist"     # all | legs_waist | legs
    cem_early_stop: float = 0.02     # relative improvement per iter to continue
    #: seed of the CEM search itself (independent of the episode seed)
    cem_seed: int = 0
    action_mode: str = DEFAULT_ACTION_MODE
    residual_scale: float = DEFAULT_RESIDUAL_SCALE
    check: bool = True
    rate_ticks_per_s: float | None = None   # measured; used for estimates only

    def as_dict(self) -> dict:
        return asdict(self)

    def dir_name(self, technique: str = "shot_entry_full") -> str:
        if self.name:
            return self.name
        bits = [technique, self.role, self.source]
        if self.start_offset_s:
            bits.append(f"off{self.start_offset_s:g}")
        if self.perturb_sigma:
            bits.append(f"p{self.perturb_sigma:g}")
        bits.append(f"s{self.seed}")
        bits.append(f"n{self.episodes}")
        return "_".join(bits)

    def reproduce(self, technique: str = "shot_entry_full") -> str:
        """The exact command line that regenerates this capture."""
        out = Path(self.out_root)
        try:
            out_rel = out.relative_to(REPO)
        except ValueError:
            out_rel = out
        ref = Path(self.ref)
        try:
            ref = ref.relative_to(REPO)
        except ValueError:
            pass
        rec = Path(self.recover_ref)
        try:
            rec = rec.relative_to(REPO)
        except ValueError:
            pass
        parts = [
            "PYTHONPATH=src .venv/bin/python scripts/solo_demo_capture.py",
            f"--source {self.source}",
            f"--ref {ref}",
            f"--recover-ref {rec}",
            f"--role {self.role}",
        ]
        if self.upto_s is not None:
            parts.append(f"--upto {self.upto_s:g}")
        parts += [f"--blend-s {self.blend_s:g}", f"--recover-s {self.recover_s:g}"
                  if self.recover_s is not None else None,
                  f"--stand-s {self.stand_s:g}",
                  f"--stance-width {self.stance_width:g}",
                  f"--stance-height {self.stance_height:g}",
                  f"--hold-s {self.hold_s:g}"]
        parts = [p for p in parts if p is not None]
        if self.source == "cem":
            parts += [f"--cem-samples {self.cem_samples}",
                      f"--cem-iters {self.cem_iters}",
                      f"--cem-knot-ticks {self.cem_knot_ticks}",
                      f"--cem-window {self.cem_window}",
                      f"--cem-stride {self.cem_stride}",
                      f"--cem-mask {self.cem_mask}"]
        if self.source == "policy" and self.policy_module:
            parts.append(f"--policy-module {self.policy_module}")
        if self.start_offset_s:
            parts.append(f"--start-offset {self.start_offset_s:g}")
        if self.perturb_sigma:
            parts.append(f"--perturb-sigma {self.perturb_sigma:g}")
        parts += [f"--seed {self.seed}", f"--episodes {self.episodes}",
                  f"--out {out_rel}", f"--name {self.dir_name(technique)}"]
        if self.touch_repair != "none":
            parts.append(f"--touch-repair {self.touch_repair}")
        parts.append("--check" if self.check else "--no-check")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# reference loading / repaired-track construction (numpy only)
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def load_track(path: str | Path) -> dict:
    """One retargeted reference track (qpos_a + t + meta)."""
    p = Path(path)
    with np.load(p, allow_pickle=True) as z:
        qpos = np.asarray(z["qpos_a"], float)
        t = np.asarray(z["t"], float)
        technique = str(z["technique"]) if "technique" in z.files else ""
        meta = {}
        if "meta" in z.files:
            try:
                meta = json.loads(str(z["meta"]))
            except Exception:                       # pragma: no cover
                meta = {"raw": str(z["meta"])[:400]}
    assert qpos.ndim == 2 and qpos.shape[1] == 36, qpos.shape
    return {"path": str(p), "qpos": qpos, "t": t, "technique": technique,
            "meta": meta, "sha256": sha256_file(p)}


def min_jerk(u: np.ndarray | float):
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    a, b = np.asarray(q0, float), np.asarray(q1, float)
    d = float(a @ b)
    if d < 0.0:
        b, d = -b, -d
    if d > 0.9995:
        out = a + (b - a) * u
        return out / np.linalg.norm(out)
    th = math.acos(max(-1.0, min(1.0, d)))
    s = math.sin(th)
    return (math.sin((1 - u) * th) / s) * a + (math.sin(u * th) / s) * b


def build_repaired_track(entry: dict, recover: dict, *, blend_s: float = 0.8,
                         hold_s: float = 0.5, recover_s: float | None = None,
                         start_offset_s: float = 0.0, upto_s: float | None = None,
                         stand_qpos: np.ndarray | None = None,
                         stand_s: float = 1.2, dt: float = DT) -> dict:
    """Assemble the executed track: entry [+ blend] + recover [+ stand] + hold.

    ``stand_qpos`` (the measured holdable stance, ``solo.stance.stance_qpos``)
    is required when ``stand_s > 0``: the operator rule is that every demo ends
    in a valid standing stance, and the video's own recovery ends in a crouch
    (pelvis ~0.56 m, rear heel up), so the track gets a final min-jerk rise
    onto the stance.  Returns ``track (M,36)``, ``t (M,)`` (0-based control
    clock), ``segment`` and ``ref_time`` (the reference clock on segments 0/2,
    NaN elsewhere).
    """
    qe, te = entry["qpos"], entry["t"]
    qr, tr = recover["qpos"], recover["t"]
    keep = te >= (start_offset_s - 1e-9)
    if upto_s is not None:
        keep &= te <= (upto_s + 1e-9)
    qe, te = qe[keep], te[keep]
    assert len(qe) >= 2, "entry track empty after the start-offset/upto cut"
    if recover_s is not None:
        rkeep = tr <= (recover_s + 1e-9)
        qr, tr = qr[rkeep], tr[rkeep]
        assert len(qr) >= 2, "recover track empty after the cut"
    n_stand = int(round(stand_s / dt)) if stand_s and stand_s > 0 else 0
    if n_stand:
        assert stand_qpos is not None, "stand_s > 0 requires the stance qpos"

    n_blend = int(round(blend_s / dt))
    n_hold = int(round(hold_s / dt))
    M = len(qe) + n_blend + len(qr) + n_stand + n_hold
    track = np.empty((M, 36))
    seg = np.empty(M, np.int8)
    ref_time = np.full(M, np.nan)
    track[:len(qe)] = qe
    seg[:len(qe)] = SEG_ENTRY
    ref_time[:len(qe)] = te
    k = len(qe)

    def _blend_rows(q_from, q_to, n):
        nonlocal k
        u = min_jerk((np.arange(n) + 1.0) / (n + 1.0))
        for i, ui in enumerate(u):
            track[k + i, :3] = q_from[:3] + (q_to[:3] - q_from[:3]) * ui
            track[k + i, 3:7] = slerp(q_from[3:7], q_to[3:7], float(ui))
            track[k + i, 7:] = q_from[7:] + (q_to[7:] - q_from[7:]) * ui
        k += n

    if n_blend:
        _blend_rows(qe[-1], qr[0], n_blend)
        seg[k - n_blend:k] = SEG_BLEND
    track[k:k + len(qr)] = qr
    seg[k:k + len(qr)] = SEG_RECOVER
    ref_time[k:k + len(qr)] = tr
    k += len(qr)
    if n_stand:
        _blend_rows(qr[-1], np.asarray(stand_qpos, float), n_stand)
        seg[k - n_stand:k] = SEG_STAND
    track[k:k + n_hold] = np.asarray(stand_qpos if n_stand else qr[-1], float)
    seg[k:k + n_hold] = SEG_HOLD
    t = np.arange(M) * dt
    return {"track": track, "t": t, "segment": seg, "ref_time": ref_time,
            "dt": dt, "n_entry": len(qe), "n_blend": n_blend,
            "n_recover": len(qr), "n_stand": n_stand, "n_hold": n_hold}


def stance_qpos_for(spec: "CaptureSpec") -> np.ndarray:
    """The measured holdable stance the demo must end in (lazy model load)."""
    from solo.scene import load_solo_model
    from solo.stance import stance_qpos
    return stance_qpos(width=spec.stance_width, height=spec.stance_height,
                       model=load_solo_model())


# ---------------------------------------------------------------------------
# dry-run plan (no simulator)
# ---------------------------------------------------------------------------
def source_dims(spec: CaptureSpec, n_ticks: int) -> dict:
    """Search dimension + rollout budget of the configured source (planning)."""
    if spec.source == "cem":
        n_knots = CEMSource.n_knots(min(spec.cem_window or n_ticks, n_ticks),
                                    spec.cem_knot_ticks)
        active = len(CEM_ACTIVE_DIMS.get(spec.cem_mask, CEM_ACTIVE_DIMS["all"]))
        dims = n_knots * active
        windows = 1 if spec.cem_window <= 0 else int(math.ceil(n_ticks / spec.cem_stride))
        rollouts = windows * (spec.cem_samples * spec.cem_iters + 1)
        return {"kind": "cem", "knots": n_knots, "active_dims": active,
                "search_dim": dims, "samples_per_plan": spec.cem_samples,
                "iters_max": spec.cem_iters, "window_ticks": spec.cem_window,
                "stride_ticks": spec.cem_stride, "windows": windows,
                "rollouts": rollouts, "rollout_ticks": windows * spec.cem_stride
                * spec.cem_samples * spec.cem_iters,
                "samples_per_dim": round(spec.cem_samples / max(1, dims), 2)}
    if spec.source == "policy":
        return {"kind": "policy", "module": spec.policy_module,
                "action_mode": spec.action_mode}
    return {"kind": "teacher", "role": "fallback/initialisation only"}


def plan_capture(spec: CaptureSpec) -> dict:
    """What a capture would do: reference ids, ticks, files, estimate."""
    entry = load_track(spec.ref)
    recover = load_track(spec.recover_ref)
    stand = stance_qpos_for(spec) if spec.stand_s and spec.stand_s > 0 else None
    repaired = build_repaired_track(entry, recover, blend_s=spec.blend_s,
                                    hold_s=spec.hold_s, recover_s=spec.recover_s,
                                    start_offset_s=spec.start_offset_s,
                                    upto_s=spec.upto_s, stand_qpos=stand,
                                    stand_s=spec.stand_s)
    ticks = len(repaired["t"])
    name = spec.dir_name(entry["technique"] or "shot_entry_full")
    root = Path(spec.out_root) / name
    files = []
    for i in range(spec.episodes):
        d = root / f"ep{i:02d}_seed{spec.seed + i}"
        files += [str(d / "trace.npz"), str(d / "meta.json"), str(d / "spec.npz")]
        if spec.check:
            files.append(str(d / "check.json"))
        if spec.source == "cem":
            files.append(str(d / "theta.npz"))
    files.append(str(root / "index.json"))
    src = source_dims(spec, ticks)
    est = None
    if spec.rate_ticks_per_s:
        sim_ticks = max(ticks, int(src.get("rollout_ticks", 0))) if src["kind"] == "cem" \
            else ticks
        est = round(spec.episodes * sim_ticks / float(spec.rate_ticks_per_s), 1)
    return {
        "reference": {
            "entry": {"path": _rel(spec.ref), "sha256": entry["sha256"],
                      "technique": entry["technique"],
                      "frames": len(entry["t"]),
                      "duration_s": float(entry["t"][-1]),
                      "net_travel_m": entry["meta"].get("net_travel_m")},
            "recover": {"path": _rel(spec.recover_ref), "sha256": recover["sha256"],
                        "frames": len(recover["t"]),
                        "duration_s": float(recover["t"][-1]),
                        "net_travel_m": recover["meta"].get("net_travel_m")},
            "role": spec.role,
        },
        "track": {"ticks": ticks, "duration_s": round(float(repaired["t"][-1]), 2),
                  "entry_ticks": repaired["n_entry"],
                  "blend_ticks": repaired["n_blend"],
                  "recover_ticks": repaired["n_recover"],
                  "stand_ticks": repaired["n_stand"],
                  "hold_ticks": repaired["n_hold"],
                  "start_offset_s": spec.start_offset_s,
                  "upto_s": spec.upto_s,
                  "terminal_stance": {"width_m": spec.stance_width,
                                      "height_m": spec.stance_height}},
        "source": src,
        "episodes": spec.episodes,
        "seed": spec.seed,
        "perturb_sigma": spec.perturb_sigma,
        "total_ticks": ticks * spec.episodes,
        "estimated_wall_s": est,
        "rate_ticks_per_s": spec.rate_ticks_per_s,
        "writes": files,
        "reproduce": spec.reproduce(entry["technique"] or "shot_entry_full"),
    }


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# model-side measurement (mujoco)
# ---------------------------------------------------------------------------
class _Ids:
    """Id table for the S1 solo scene (``a_`` prefix)."""

    def __init__(self, model, prefix: str = PREFIX):
        import mujoco
        n = mujoco.mj_name2id
        obj = mujoco.mjtObj
        self.prefix = prefix
        self.pelvis = n(model, obj.mjOBJ_BODY, prefix + "pelvis")
        self.torso = n(model, obj.mjOBJ_BODY, prefix + "torso_link")
        self.base_qadr = int(model.jnt_qposadr[n(
            model, obj.mjOBJ_JOINT, prefix + "floating_base_joint")])
        self.foot_body = {s: n(model, obj.mjOBJ_BODY, f"{prefix}{s}_ankle_roll_link")
                          for s in SIDES}
        self.sole_geoms = {}
        for s in SIDES:
            geoms = [g for g in range(model.ngeom)
                     if int(model.geom_bodyid[g]) == self.foot_body[s]
                     and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
            assert len(geoms) == 4, (s, len(geoms))
            self.sole_geoms[s] = geoms
        self.knee_site = {s: n(model, obj.mjOBJ_SITE, f"{prefix}{s}_knee")
                          for s in SIDES}
        self.foot_site = {s: n(model, obj.mjOBJ_SITE, f"{prefix}{s}_foot")
                          for s in SIDES}
        self.jnt_range = np.asarray(model.jnt_range[1:30], float).copy()
        self.act_limit = np.maximum(np.abs(model.jnt_actfrcrange[1:30, 0]),
                                    np.abs(model.jnt_actfrcrange[1:30, 1]))
        from retarget.landmarks import SOLVED_SITES, site_weights
        self.solved_sites = [n(model, obj.mjOBJ_SITE, prefix + s)
                             for s in SOLVED_SITES]
        self.site_weights = site_weights("default")     # (19,) retarget weights
        self.mass = np.asarray(model.body_mass, float)
        self.mass_total = float(self.mass.sum())
        #: measured sole-centre height of a foot resting on the mat (the
        #: model's own verified stand keyframe) -- "flat" is judged against it
        from solo.scene import stand_frame
        stand_q, _ = stand_frame(model)
        d0 = mujoco.MjData(model)
        d0.qpos[:] = stand_q
        mujoco.mj_forward(model, d0)
        self.sole_rest_z = float(min(
            min(d0.geom_xpos[g][2] for g in self.sole_geoms[s]) for s in SIDES))
        #: the stand keyframe's pelvis height (measured stance reference)
        self.stand_pelvis_z = float(stand_q[2])

    def com(self, data) -> np.ndarray:
        return np.asarray(data.subtree_com[self.pelvis], float).copy()

    def com_vel(self, data) -> np.ndarray:
        """Mass-weighted CoM velocity (the env's ``_reward_inputs`` formula)."""
        return (self.mass[:, None] * np.asarray(data.cvel, float)[:, 3:6]
                ).sum(axis=0) / self.mass_total

    def sole_pts(self, data) -> np.ndarray:
        return np.array([[np.asarray(data.geom_xpos[g], float)
                          for g in self.sole_geoms[s]] for s in SIDES])

    def foot_load(self, data) -> np.ndarray:
        return np.array([float(data.cfrc_ext[self.foot_body[s]][5]) for s in SIDES])

    def pitch_deg(self, data) -> float:
        q = data.qpos[self.base_qadr:self.base_qadr + 7]
        yaw = float(np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                               1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))
        up = data.xmat[self.torso].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        return float(np.degrees(np.arctan2(float(up[:2] @ fwd), float(up[2]))))


def _margin(com_xy: np.ndarray, pts: np.ndarray) -> float:
    """Signed CoM margin vs the hull of the sole points (NaN if no support)."""
    hull = hull2d(pts)
    if len(hull) == 0:
        return float("nan")
    if len(hull) == 1:
        return -float(np.linalg.norm(com_xy - hull[0]))
    if len(hull) == 2:
        a, b = hull
        e = b - a
        L = float(np.linalg.norm(e))
        if L < 1e-12:
            return -float(np.linalg.norm(com_xy - a))
        t = float(np.clip((com_xy - a) @ e / (L * L), 0.0, 1.0))
        return -float(np.linalg.norm(com_xy - (a + t * e)))
    return polygon_margin(com_xy, hull)


def measure_tracks(model, repaired: dict, ids: _Ids) -> RefTracks:
    """FK-measure the reference tracks on the repaired track's grid."""
    import mujoco
    from teacher.phases import label_frames
    d = mujoco.MjData(model)
    M = len(repaired["t"])
    out = {k: np.empty(M) for k in ("pitch_deg", "margin", "pelvis_z")}
    out["contact"] = np.empty((M, 2), bool)
    out["knee_z"] = np.empty((M, 2))
    out["base_xy"] = np.empty((M, 2))
    out["knee_site_xy"] = np.empty((M, 2, 2))
    out["site_pos"] = np.empty((M, len(ids.solved_sites), 3))
    for i in range(M):
        d.qpos[:] = repaired["track"][i]
        mujoco.mj_forward(model, d)
        sole = ids.sole_pts(d)
        touch = sole[:, :, 2].min(axis=1) < TOUCH_Z
        out["contact"][i] = touch
        pts = sole[touch][:, :, :2].reshape(-1, 2) if touch.any() else np.zeros((0, 2))
        out["margin"][i] = _margin(ids.com(d)[:2], pts)
        out["knee_z"][i] = [d.site_xpos[ids.knee_site[s]][2] for s in SIDES]
        out["knee_site_xy"][i] = [d.site_xpos[ids.knee_site[s]][:2] for s in SIDES]
        out["pelvis_z"][i] = float(d.qpos[ids.base_qadr + 2])
        out["base_xy"][i] = d.qpos[ids.base_qadr:ids.base_qadr + 2]
        out["pitch_deg"][i] = ids.pitch_deg(d)
        out["site_pos"][i] = np.asarray(d.site_xpos[ids.solved_sites], float)
    return RefTracks(grid_t=repaired["t"].copy(),
                     ref_time=repaired["ref_time"].copy(),
                     segment=repaired["segment"].copy(),
                     phase=label_frames(repaired["t"], out["pelvis_z"]),
                     pitch_deg=out["pitch_deg"], margin=out["margin"],
                     contact=out["contact"], knee_z=out["knee_z"],
                     pelvis_z=out["pelvis_z"], base_xy=out["base_xy"],
                     knee_site_xy=out["knee_site_xy"], site_pos=out["site_pos"])


def _rising_edges(contact: np.ndarray, min_air: int) -> list:
    """Indices where a foot plants after >= ``min_air`` ticks airborne."""
    out = []
    n = len(contact)
    k = 0
    while k < n:
        if not contact[k]:
            j = k
            while j < n and not contact[j]:
                j += 1
            if (j - k) >= min_air and j < n and contact[j]:
                out.append(j)
            k = j
        else:
            k += 1
    return out


def measure_spec(model, entry: dict, recover: dict, repaired: dict,
                 ids: _Ids) -> tuple:
    """Reference-derived ExecSpec + tracks (see solo.exec_check)."""
    from teacher.phases import segments
    tr = measure_tracks(model, repaired, ids)
    seg = repaired["segment"]
    ref_mask = np.isin(seg, REFERENCE_SEGMENTS)
    entry_mask = seg == SEG_ENTRY
    rec_mask = seg == SEG_RECOVER

    # penetration knee: the knee that reaches the lowest point in the entry
    entry_knee_min = np.array([tr.knee_z[entry_mask, i].min() for i in (0, 1)])
    knee_ix = int(np.argmin(entry_knee_min))
    knee_min_z = float(entry_knee_min[knee_ix])
    knee_min_t = float(tr.grid_t[entry_mask][
        int(np.argmin(tr.knee_z[entry_mask, knee_ix]))])
    knee_contact_z = knee_min_z + KNEE_CONTACT_BAND
    hit = np.flatnonzero(entry_mask & (tr.knee_z[:, knee_ix] <= knee_contact_z))
    knee_contact_t = float(tr.grid_t[hit[0]]) if len(hit) else None

    # lead foot: the foot that plants (after >= 0.1 s airborne) last before
    # the knee arrives at depth
    min_air = max(1, int(round(0.10 / repaired["dt"])))
    knee_row = (int(np.flatnonzero(entry_mask)[int(np.flatnonzero(
        entry_mask & (tr.knee_z[:, knee_ix] <= knee_contact_z))[0])])
        if knee_contact_t is not None else None)
    plants = {}
    for i, s in enumerate(SIDES):
        cand = _rising_edges(tr.contact[entry_mask, i], min_air)
        if knee_row is not None:
            cand = [p for p in cand if np.flatnonzero(entry_mask)[p] <= knee_row]
        plants[i] = cand
    lead_ix, plant_row = None, None
    for i in (0, 1):
        if plants[i]:
            p = plants[i][-1]
            if plant_row is None or p > plant_row:
                lead_ix, plant_row = i, p
    if lead_ix is None:                      # fall back: foot in contact at the end
        last = tr.contact[entry_mask][-1]
        lead_ix = int(np.argmax(last)) if last.any() else 1
    lead_plant_t = (float(tr.grid_t[np.flatnonzero(entry_mask)[plant_row]])
                    if plant_row is not None else None)

    # entry travel in the entry's initial heading frame
    e_idx = np.flatnonzero(entry_mask)
    k0, k1 = int(e_idx[0]), int(e_idx[-1])
    q0 = repaired["track"][k0]
    yaw0 = float(np.arctan2(2.0 * (q0[3] * q0[6] + q0[4] * q0[5]),
                            1.0 - 2.0 * (q0[5] ** 2 + q0[6] ** 2)))
    disp = tr.base_xy[k1] - tr.base_xy[k0]
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    travel_fwd = float(c * disp[0] - s * disp[1])
    travel_lat = float(s * disp[0] + c * disp[1])

    # rise targets: the recover track's own end state
    r_end = int(np.flatnonzero(rec_mask)[-1])
    rise_pelvis_z = float(tr.pelvis_z[r_end]) - 0.03
    rise_knee_z = knee_contact_z + 0.05

    # terminal-stance thresholds: measured on the stance the demo ends in
    # (the model's verified stand keyframe + the commanded stance solution)
    import mujoco
    from solo.commands import CommandRanges
    ranges = CommandRanges()
    ds = mujoco.MjData(model)
    stance_q = np.asarray(repaired["track"][-1], float)
    ds.qpos[:] = stance_q
    mujoco.mj_forward(model, ds)
    sole = ids.sole_pts(ds)
    pts = sole[:, :, :2].reshape(-1, 2)
    stance_margin = _margin(ids.com(ds)[:2], pts)
    stance_tilt = ids.pitch_deg(ds)          # signed sagittal lean of the stance
    up = ds.xmat[ids.torso].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
    stance_tilt_total = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
    stance_flat = float((sole[:, :, 2].max(axis=1) - ids.sole_rest_z).max())
    stance_pz = float(stance_q[ids.base_qadr + 2])

    spec = ExecSpec(
        technique=entry["technique"] or "shot_entry_full",
        role="a",
        sources=(_rel(entry["path"]), _rel(recover["path"])),
        sha256=(entry["sha256"], recover["sha256"]),
        dt=repaired["dt"], duration_s=float(repaired["t"][-1]),
        n_frames=len(repaired["t"]),
        lead_side=SIDES[lead_ix], lead_plant_t=lead_plant_t,
        knee_side=SIDES[knee_ix], knee_min_z=knee_min_z, knee_min_t=knee_min_t,
        knee_contact_z=float(knee_contact_z), knee_contact_t=knee_contact_t,
        travel_forward_m=travel_fwd, travel_lateral_m=travel_lat,
        travel_net_m=float(np.linalg.norm(disp)),
        entry_end_t=float(tr.grid_t[e_idx[-1]]),
        phase_segments=segments(repaired["t"], tr.phase),
        rise_pelvis_z=rise_pelvis_z, rise_knee_z=rise_knee_z,
        stance_pelvis_z=stance_pz,
        stance_pelvis_lo=float(ranges.stance_height[0]),
        stance_pelvis_hi=float(ranges.stance_height[1]),
        stance_tilt_max_deg=float(max(15.0, 2.0 * stance_tilt_total)),
        stance_margin_min=0.02,          # audit STABLY_FEASIBLE band
        stance_speed_max=0.05,           # drill settle (0.02) / lift (0.04) gates
        stance_flat_tol_m=0.012,         # drill kin flat_contact tolerance
        sole_rest_z=float(ids.sole_rest_z),
        term_window_s=0.4,
        tracks=tr)
    return spec, tr


# ---------------------------------------------------------------------------
# action-space helpers (mirror src/solo/env.py; kept local so the demo does not
# depend on modules other agents are editing)
# ---------------------------------------------------------------------------
def action_from_ctrl(ctrl: np.ndarray, lo: np.ndarray, hi: np.ndarray, *,
                     mode: str = DEFAULT_ACTION_MODE, base: np.ndarray | None = None,
                     scale: float = DEFAULT_RESIDUAL_SCALE) -> np.ndarray:
    """Absolute ctrl targets -> the ACTION handed to ``SoloEnv.step``."""
    c = np.clip(np.asarray(ctrl, float), lo, hi)
    if mode == "absolute":
        return c
    z = (c - np.asarray(base, float)) / scale
    return np.arctanh(np.clip(z, -1.0 + 1e-6, 1.0 - 1e-6))


def unit_from_ctrl(ctrl: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Ctrl targets -> the unit action in [-1, 1] stored as ``prev_action``."""
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return np.clip((np.asarray(ctrl, float) - mid) / np.where(half > 0, half, 1.0),
                   -1.0, 1.0)


# ---------------------------------------------------------------------------
# demonstration sources (source-agnostic capture)
#
# The capture is agnostic about *who* produces the actions.  Three sources ship:
#
# * ``cem``     -- the primary one: a sampling-based optimizer (CEM) that
#                  searches control sequences minimising the weighted
#                  landmark-site error against the video reference under the
#                  actual dynamics ("direct dynamic retargeting"; the standard
#                  remedy for kinematic references whose contacts are
#                  ambiguous).  The best sequence is replayed deterministically
#                  as the demo.
# * ``policy``  -- an external learned controller: any callable that consumes
#                  the 115-dim actor observation and returns the 29-dim action
#                  (the ImitationBC / RL tracking objective's policy).  Loaded
#                  from a Python module (``--policy-module path:attr``).
# * ``teacher`` -- the phase-3 stabilised teacher.  FALLBACK / INITIALISATION
#                  ONLY, never the demo source of record: measured stay-up
#                  0.448-0.690 over 3 seeds with 1.64-2.60 m landmark error,
#                  governor active on 92.9-99.2 % of ticks, and non-Markovian
#                  (integrators / governor alpha / rate-limiter memory are not
#                  in the actor observation), so its actions are not a function
#                  of the observable state (TeacherDataAudit, 2026-10-08).
# ---------------------------------------------------------------------------
@dataclass
class SourceContext:
    """Everything a source needs to produce its 29 joint-target commands."""

    model: object
    ids: _Ids
    spec: CaptureSpec
    exec_spec: ExecSpec
    repaired: dict
    tracks: RefTracks
    stand_q: np.ndarray
    stand_ctrl: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    cmd: object
    markers: dict
    track: np.ndarray
    M: int
    #: the episode's initial qpos (the sampling source's rollouts start from it)
    q0: np.ndarray


class DemoSource:
    """One control call per 50 Hz tick; ``prepare`` may optimise first."""

    name = "source"

    def prepare(self, ctx: SourceContext, seed: int, *, verbose: bool = False) -> dict:
        return {}

    def control(self, data, tick: int, ctx: SourceContext,
                obs_prev: np.ndarray | None) -> np.ndarray:
        raise NotImplementedError

    def provenance(self) -> dict:
        return {"kind": self.name}


class TeacherSource(DemoSource):
    """Fallback: the stabilised phase-3 teacher tracks the repaired track."""

    name = "teacher"

    def __init__(self, ctx: SourceContext, *, touch_repair: str = "none"):
        from teacher.controller import RobotTeacher
        self.teacher = RobotTeacher(ctx.model, PREFIX, ctx.track,
                                    ctx.repaired["t"],
                                    technique=ctx.exec_spec.technique)
        self.touch_repair = touch_repair

    def prepare(self, ctx: SourceContext, seed: int, *, verbose: bool = False) -> dict:
        if self.touch_repair == "lead":
            touch = _touch_override(ctx.repaired, ctx.exec_spec)
            if touch is not None:
                self.teacher.update_reference(np.arange(ctx.M), ctx.track, touch=touch)
        return {}

    def control(self, data, tick, ctx, obs_prev=None):
        return self.teacher.control(data, float(ctx.repaired["t"][tick]))

    def provenance(self) -> dict:
        return {"kind": "teacher", "role": "fallback/initialisation only",
                "touch_repair": self.touch_repair}


#: residual-joint masks for the CEM search (local 29-joint order:
#: 0-5 left leg, 6-11 right leg, 12-14 waist yaw/roll/pitch, 15-28 arms)
CEM_ACTIVE_DIMS = {
    "all": tuple(range(29)),
    "legs_waist": tuple(range(15)),
    "legs": tuple(range(12)),
}


class CEMSource(DemoSource):
    """CEM over knot-interpolated residual joint targets (dynamic retargeting).

    Parameterisation: ``K`` knots spaced ``knot_ticks`` ticks apart, each an
    ``n_active``-dim residual (the mask's joints; the rest follow the reference
    exactly) added to the reference's joint row and clipped to the actuator
    ctrlrange.  Search dimension = ``K * n_active`` (recorded in the
    provenance, with the budget, so a downstream reader can tell a retargeted
    trajectory from a passthrough).

    Objective (metres + penalties): weighted landmark-site RMS error to the
    reference (``retarget.landmarks.site_weights``), plus a fall penalty
    (unsimulated ticks count as ``fall_err`` m), the residual regulariser, and
    the physical constraints the parent requires the trajectory to respect:
    commanded actuator-force saturation, loaded-foot slip, and (when the
    window covers the episode end) the terminal-stance violation.  Receding
    horizon (``window`` ticks, ``stride`` between replans) keeps the
    dimension manageable: each plan searches its window from the *current
    simulated state*.
    """

    name = "cem"

    def __init__(self, *, samples: int = 450, iters: int = 10, knot_ticks: int = 25,
                 window: int = 50, stride: int = 25, sigma0: float = 0.12,
                 elite_frac: float = 0.4, sigma_floor: float = 0.01,
                 theta_max: float = 0.8, reg: float = 0.02,
                 fall_err: float = 0.5, mask: str = "legs_waist",
                 w_sat: float = 0.2, w_slip: float = 1.0, w_term: float = 0.3,
                 early_stop: float = 0.02, seed: int = 0):
        self.samples = int(samples)
        self.iters = int(iters)
        self.knot_ticks = int(knot_ticks)
        self.window = int(window)
        self.stride = int(stride) or max(1, (int(window) or 2) // 2)
        self.sigma0 = float(sigma0)
        self.elite_frac = float(elite_frac)
        self.sigma_floor = float(sigma_floor)
        self.theta_max = float(theta_max)
        self.reg = float(reg)
        self.fall_err = float(fall_err)
        self.mask = np.asarray(CEM_ACTIVE_DIMS[mask], int)
        self.mask_name = mask
        self.w_sat = float(w_sat)
        self.w_slip = float(w_slip)
        self.w_term = float(w_term)
        self.early_stop = float(early_stop)
        self.seed = int(seed)
        self.theta: np.ndarray | None = None      # (M, 29) per-tick residuals
        self.history: list = []
        self.n_rollouts = 0
        self.wall_s = 0.0
        self.final: dict = {}
        self._data = None

    @property
    def search_dim(self) -> int:
        return 0 if self.theta is None else self.theta.shape[0]

    # -- parameterisation -------------------------------------------------
    @staticmethod
    def n_knots(n_ticks: int, knot_ticks: int) -> int:
        return int(math.ceil(n_ticks / max(1, knot_ticks))) + 1

    @staticmethod
    def dq_at(knots: np.ndarray, tick: int, knot_ticks: int) -> np.ndarray:
        """Linear interpolation of the knot residuals at a tick."""
        u = tick / max(1, knot_ticks)
        k0 = min(int(u), len(knots) - 1)
        k1 = min(k0 + 1, len(knots) - 1)
        f = float(np.clip(u - k0, 0.0, 1.0))
        return (1.0 - f) * knots[k0] + f * knots[k1]

    def _expand(self, knots_masked: np.ndarray, n_ticks: int) -> np.ndarray:
        """Masked knots -> full (n_ticks, 29) per-tick residuals."""
        out = np.zeros((n_ticks, 29))
        for j in range(n_ticks):
            out[j, self.mask] = self.dq_at(knots_masked, j, self.knot_ticks)
        return out

    def _stance_violation(self, d, ctx: SourceContext) -> float:
        """Normalised terminal-stance violation (0 = valid stance)."""
        spec = ctx.exec_spec
        pz = float(d.qpos[ctx.ids.base_qadr + 2])
        speed = float(np.linalg.norm(np.asarray(d.qvel[0:2], float)))
        up = d.xmat[ctx.ids.torso].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
        tilt = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
        sole = ctx.ids.sole_pts(d)
        touch = sole[:, :, 2].min(axis=1) < TOUCH_Z
        pts = sole[touch][:, :, :2].reshape(-1, 2) if touch.any() else np.zeros((0, 2))
        margin = _margin(ctx.ids.com(d)[:2], pts)
        flat = (sole[:, :, 2].max(axis=1) - ctx.ids.sole_rest_z) <= spec.stance_flat_tol_m
        v = 0.0
        v += max(0.0, (spec.stance_pelvis_lo - pz) / 0.05)
        v += max(0.0, (pz - spec.stance_pelvis_hi) / 0.05)
        v += max(0.0, (speed - spec.stance_speed_max) / 0.05)
        v += max(0.0, (tilt - spec.stance_tilt_max_deg) / 15.0)
        v += 0.0 if (math.isfinite(margin)
                     and margin >= spec.stance_margin_min) else 1.0
        v += 0.0 if bool((flat & touch).all()) else 1.0
        return float(v)

    # -- rollouts ----------------------------------------------------------
    def _rollout(self, theta: np.ndarray, ctx: SourceContext, q0: np.ndarray,
                 start_tick: int = 0, n_ticks: int | None = None) -> tuple:
        """Simulate ``n_ticks`` from ``q0`` under per-tick residuals ``theta``.

        Returns ``(score, info)``; the score is the landmark-site RMS plus the
        fall / saturation / slip / terminal-stance penalties, all in metres.
        """
        import mujoco
        from solo.fall import FallDetector, fall_features
        if self._data is None:
            self._data = mujoco.MjData(ctx.model)
        d = self._data
        d.qpos[:] = q0
        d.qvel[:] = 0.0
        d.ctrl[:] = ctx.stand_ctrl
        mujoco.mj_forward(ctx.model, d)
        det = FallDetector()
        n = int(n_ticks if n_ticks is not None else ctx.M - start_tick)
        err = 0.0
        wsum = float(ctx.ids.site_weights.sum())
        sat_acc = 0.0
        slip_acc = 0.0
        slip_worst = 0.0
        anchors = [None, None]
        fell, done = False, 0
        lo_touch = 0.015                     # drill.kin.TOUCH_Z (clearly loaded)
        for j in range(n):
            tick = start_tick + j
            d.ctrl[:] = np.clip(ctx.track[tick, 7:36] + theta[j], ctx.lo, ctx.hi)
            for _ in range(10):
                mujoco.mj_step(ctx.model, d)
            k = min(tick + 1, ctx.M - 1)
            p = np.asarray(d.site_xpos[ctx.ids.solved_sites], float)
            e = np.linalg.norm(p - ctx.tracks.site_pos[k], axis=1)
            err += float((ctx.ids.site_weights * e ** 2).sum())
            # -- physical-constraint penalties (per the parent's requirement)
            sat_acc += float(np.mean(
                np.abs(np.asarray(d.actuator_force, float))[:29]
                >= 0.95 * ctx.ids.act_limit))
            sole = ctx.ids.sole_pts(d)
            com_xy = ctx.ids.com(d)[:2]
            for i in range(2):
                pts = sole[i]
                hull = hull2d(pts[:, :2])
                loaded = (pts[:, 2].min() < lo_touch
                          and len(hull) >= 3
                          and polygon_margin(com_xy, hull) > -0.01)
                if not loaded:
                    anchors[i] = None
                    continue
                centre = pts[:, :2].mean(axis=0)
                if anchors[i] is None:
                    anchors[i] = centre
                drift = float(np.linalg.norm(centre - anchors[i]))
                slip_acc += max(0.0, drift - 0.02)
                slip_worst = max(slip_worst, drift)
            done += 1
            if det.update(fall_features(ctx.model, d),
                          float(ctx.repaired["t"][tick])):
                fell = True
                break
        err += (n - done) * (self.fall_err ** 2) * wsum   # unsimulated ticks
        rms = math.sqrt(err / max(wsum * n, 1e-9))
        term = (self._stance_violation(d, ctx)
                if (start_tick + n) >= ctx.M else 0.0)
        score = (rms + self.reg * float(np.mean(theta ** 2))
                 + self.w_sat * sat_acc / max(done, 1)
                 + self.w_slip * slip_acc / max(done, 1)
                 + self.w_term * term)
        self.n_rollouts += 1
        return score, {"rms_m": rms, "fell": fell, "ticks": done,
                       "sat": sat_acc / max(done, 1), "slip_m": slip_worst,
                       "term": term,
                       "end_qpos": np.asarray(d.qpos, float).copy()}

    def _plan(self, ctx: SourceContext, q0: np.ndarray, start_tick: int,
              n_ticks: int, rng) -> tuple:
        """One CEM search over a window; returns (best, history)."""
        K = self.n_knots(n_ticks, self.knot_ticks)
        A = len(self.mask)
        mu = np.zeros((K, A))
        sigma = self.sigma0
        best = {"score": math.inf, "theta": self._expand(mu, n_ticks)}
        hist = []
        n_elite = max(1, int(round(self.elite_frac * self.samples)))
        stall = 0
        for it in range(self.iters):
            cand_k = np.clip(mu[None] + sigma * rng.normal(
                size=(self.samples, K, A)), -self.theta_max, self.theta_max)
            cand = [self._expand(c, n_ticks) for c in cand_k]
            scores = [self._rollout(c, ctx, q0, start_tick, n_ticks)[0]
                      for c in cand]
            order = np.argsort(scores)
            elite = cand_k[order[:n_elite]]
            mu = elite.mean(axis=0)
            sigma = max(float(elite.std()), self.sigma_floor)
            if scores[int(order[0])] < best["score"]:
                best = {"score": float(scores[int(order[0])]),
                        "theta": cand[int(order[0])].copy()}
            hist.append({"iter": it, "knots": int(K), "active_dims": int(A),
                         "search_dim": int(K * A),
                         "best": float(scores[int(order[0])]),
                         "mean": float(np.mean(scores)),
                         "sigma": float(sigma)})
            # observed-convergence stop (not a fixed iteration count)
            if it >= 1:
                prev = hist[-2]["best"]
                rel = (prev - hist[-1]["best"]) / max(abs(prev), 1e-9)
                stall = stall + 1 if rel < self.early_stop else 0
                if stall >= 2:
                    break
        return best, hist

    def prepare(self, ctx: SourceContext, seed: int, *, verbose: bool = False) -> dict:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed + int(seed))
        q0 = np.asarray(ctx.q0, float)
        # the degenerate case: the input reference passed through unchanged
        base_score, base_info = self._rollout(np.zeros((ctx.M, 29)), ctx, q0)
        self.baseline = {"score": float(base_score),
                         "rms_m": float(base_info["rms_m"]),
                         "fell": bool(base_info["fell"]),
                         "ticks": int(base_info["ticks"]),
                         "term": float(base_info["term"])}
        theta = np.zeros((ctx.M, 29))
        if self.window <= 0:
            plan, hist = self._plan(ctx, q0, 0, ctx.M, rng)
            theta = plan["theta"]
            self.history = hist
        else:
            start = 0
            while start < ctx.M:
                n = min(self.window, ctx.M - start)
                plan, hist = self._plan(ctx, q0, start, n, rng)
                take = min(self.stride, n)
                theta[start:start + take] = plan["theta"][:take]
                _, info = self._rollout(plan["theta"], ctx, q0, start, take)
                q0 = info["end_qpos"]
                self.history.extend(hist)
                start += take
        self.theta = theta
        self.wall_s = time.perf_counter() - t0
        score, final = self._rollout(theta, ctx, np.asarray(ctx.q0, float))
        self.final = {"score": float(score), "rms_m": float(final["rms_m"]),
                      "fell": bool(final["fell"]), "ticks": int(final["ticks"]),
                      "term": float(final["term"]),
                      "improvement_m": round(float(base_info["rms_m"])
                                             - float(final["rms_m"]), 5)}
        if verbose:
            print(f"    cem: {self.n_rollouts} rollouts, {self.wall_s:.1f}s, "
                  f"landmark RMS {final['rms_m']:.4f} m vs passthrough "
                  f"{base_info['rms_m']:.4f} m (delta {self.final['improvement_m']:+.4f}), "
                  f"fell={final['fell']}, dims={len(self.mask)}x"
                  f"{self.n_knots(min(self.window or ctx.M, ctx.M), self.knot_ticks)}")
        return self.provenance()

    def control(self, data, tick, ctx, obs_prev=None):
        return np.clip(ctx.track[tick, 7:36] + self.theta[tick], ctx.lo, ctx.hi)

    def provenance(self) -> dict:
        n_knots = (self.n_knots(min(self.window or 0, self.theta.shape[0])
                                if (self.window or 0) > 0 else self.theta.shape[0],
                                self.knot_ticks) if self.theta is not None else 0)
        return {"kind": "cem", "samples": self.samples, "iters": self.iters,
                "knot_ticks": self.knot_ticks, "window": self.window,
                "stride": self.stride, "sigma0": self.sigma0,
                "theta_max": self.theta_max, "reg": self.reg,
                "mask": self.mask_name, "active_dims": int(len(self.mask)),
                "knots_per_window": int(n_knots),
                "window_search_dim": int(n_knots * len(self.mask)),
                "penalty_weights": {"sat": self.w_sat, "slip": self.w_slip,
                                    "terminal": self.w_term,
                                    "fall_err_m": self.fall_err},
                "early_stop": self.early_stop,
                "n_rollouts": self.n_rollouts, "wall_s": round(self.wall_s, 2),
                "baseline": getattr(self, "baseline", None),
                "final": self.final, "history": self.history[-12:],
                "theta_sha256": (hashlib.sha256(
                    np.ascontiguousarray(self.theta, np.float64).tobytes()
                ).hexdigest()[:16] if self.theta is not None else None)}


class PolicySource(DemoSource):
    """An external learned controller (RL tracking / BC) as the demo source.

    Interface (the only thing the capture requires): a callable
    ``policy(actor_obs: (115,) float32) -> action (29,)`` where the action is
    the ``src/solo/env.py`` action for the configured action mode.  Load it
    with ``--policy-module path/to/module.py:attr``.
    """

    name = "policy"

    def __init__(self, policy, *, module: str = "", action_mode: str = "absolute",
                 base: np.ndarray | None = None, scale: float = 0.5):
        self.policy = policy
        self.module = module
        self.action_mode = action_mode
        self.base = base
        self.scale = float(scale)

    def control(self, data, tick, ctx, obs_prev=None):
        if obs_prev is None:
            raise RuntimeError("policy source needs the previous actor observation")
        action = np.asarray(self.policy(obs_prev), float).reshape(29)
        if self.action_mode == "absolute":
            return np.clip(action, ctx.lo, ctx.hi)
        return np.clip(self.base + self.scale * np.tanh(action), ctx.lo, ctx.hi)

    def provenance(self) -> dict:
        return {"kind": "policy", "module": self.module,
                "action_mode": self.action_mode, "residual_scale": self.scale}


def load_policy(module_spec: str):
    """Import ``path/to/module.py:attr`` and return the callable."""
    import importlib.util
    path, _, attr = module_spec.partition(":")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"policy module {path!r} not found -- the capture needs a callable "
            "policy(actor_obs(115,)) -> action(29,) (see PolicySource)")
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, attr or "policy", None)
    if fn is None:
        raise AttributeError(f"{path!r} has no attribute {attr!r}")
    return fn


def _scratch(model):
    import mujoco
    return mujoco.MjData(model)


def make_source(spec: CaptureSpec, ctx: SourceContext, seed: int,
                *, verbose: bool = False) -> DemoSource:
    """Instantiate the configured demo source."""
    kind = spec.source
    if kind == "cem":
        return CEMSource(samples=spec.cem_samples, iters=spec.cem_iters,
                         knot_ticks=spec.cem_knot_ticks, window=spec.cem_window,
                         stride=spec.cem_stride, sigma0=spec.cem_sigma,
                         theta_max=spec.cem_theta_max, seed=spec.cem_seed)
    if kind == "policy":
        if not spec.policy_module:
            raise ValueError(
                "source 'policy' requires --policy-module path:attr exposing "
                "policy(actor_obs(115,)) -> action(29,)")
        return PolicySource(load_policy(spec.policy_module),
                            module=spec.policy_module,
                            action_mode=spec.action_mode,
                            base=ctx.stand_ctrl, scale=spec.residual_scale)
    if kind == "teacher":
        return TeacherSource(ctx, touch_repair=spec.touch_repair)
    raise ValueError(f"unknown source {kind!r}")
def _perturb(qpos0: np.ndarray, sigma: float, seed: int, base_qadr: int) -> tuple:
    """Seeded initial-state perturbation (joint noise + base xy/yaw wobble)."""
    rng = np.random.default_rng(int(seed))
    q = np.asarray(qpos0, float).copy()
    info = {"sigma": float(sigma), "seed": int(seed)}
    if sigma <= 0.0:
        return q, info
    dq = rng.normal(0.0, sigma, size=29)
    q[7:36] += dq
    xy_scale = RESET_JITTER_XY * (sigma / 0.01)
    yaw_scale = RESET_JITTER_YAW * (sigma / 0.01)
    dxy = rng.normal(0.0, xy_scale, size=2)
    dyaw = float(rng.normal(0.0, yaw_scale))
    q[base_qadr:base_qadr + 2] += dxy
    cy, sy = math.cos(dyaw / 2.0), math.sin(dyaw / 2.0)
    qw, qx, qy, qz = q[base_qadr + 3:base_qadr + 7]
    q[base_qadr + 3:base_qadr + 7] = np.array([
        qw * cy - qz * sy, qx * cy + qy * sy,
        qy * cy - qx * sy, qz * cy + qw * sy])
    info.update({"joint_rms": float(np.sqrt(np.mean(dq ** 2))),
                 "dxy": dxy.tolist(), "dyaw": dyaw,
                 "max_joint": float(np.abs(dq).max())})
    return q, info


def _touch_override(repaired: dict, spec: ExecSpec) -> np.ndarray | None:
    """Declare the lead foot planted from its reference plant to the end.

    A documented reference repair for the monocular-depth float: the teacher
    then anchors/flattens that sole onto the mat (``stabilizers.foot_rows``).
    """
    if spec.lead_plant_t is None:
        return None
    seg = repaired["segment"]
    ref_time = repaired["ref_time"]
    lead = 0 if spec.lead_side == "left" else 1
    touch = np.zeros((len(repaired["t"]), 2, 2), bool)
    on = np.zeros(len(repaired["t"]), bool)
    on[(seg == SEG_ENTRY) & (ref_time >= spec.lead_plant_t)] = True
    on[seg == SEG_RECOVER] = True
    on[seg == SEG_HOLD] = True
    touch[on, lead, :] = True
    return touch


def _build_obs(model, data, ids: _Ids, *, stand_q: np.ndarray, cmd, prev_action,
               phase: float, markers, foot_slip: float, sat_frac: float,
               limit_prox: float, dorsal: bool, next_push=None):
    """The actor/privileged observation exactly as the env builds it."""
    import mujoco
    from solo.fall import contact_state
    from solo.obs import ObsContext, actor_obs, critic_obs, privileged_obs
    R = data.xmat[ids.pelvis].reshape(3, 3)
    q = data.qpos
    qv = data.qvel
    com = ids.com(data)
    pelvis_xyz = np.asarray(data.xpos[ids.pelvis], float)
    contacts = contact_state(model, data)
    loaded = [ids.foot_site[s] for s, ok in
              zip(SIDES, (contacts.left_foot, contacts.right_foot)) if ok]
    if loaded:
        sc = np.mean([np.asarray(data.site_xpos[sid], float) for sid in loaded], axis=0)
        support_local = R.T @ (sc - pelvis_xyz)
    else:
        support_local = np.zeros(3)
    com_rel = R.T @ (com - pelvis_xyz)
    ctx = ObsContext(
        base_linvel_local=R.T @ np.asarray(qv[0:3], float),
        base_angvel_local=R.T @ np.asarray(qv[3:6], float),
        gravity_local=R.T @ np.array([0.0, 0.0, -1.0]),
        joint_pos_rel=np.asarray(q[7:36], float) - stand_q[7:36],
        joint_vel=np.asarray(qv[6:35], float),
        prev_action=np.asarray(prev_action, float),
        cmd=cmd,
        phase=float(np.clip(phase, 0.0, 1.0)),
        contacts=contacts,
        dorsal_contact=bool(dorsal),
        tilt_deg=float(np.degrees(np.arccos(np.clip(
            (data.xmat[ids.torso].reshape(3, 3) @ np.array([0.0, 0.0, 1.0]))[2],
            -1.0, 1.0)))),
        pelvis_z=float(q[ids.base_qadr + 2]),
        stand_height=float(stand_q[2]),
        com_rel_local=com_rel,
        com_vel_local=R.T @ ids.com_vel(data),
        support_center_local=support_local,
        com_to_support_xy=np.array([com_rel[0] - support_local[0],
                                    com_rel[1] - support_local[1]]),
        marker_local={k: R.T @ (np.asarray(v, float) - pelvis_xyz)
                      for k, v in markers.items()},
        next_push=next_push,
        foot_slip=float(foot_slip),
        sat_frac=float(sat_frac),
        limit_prox=float(limit_prox),
        shot_phase=float(np.clip(phase, 0.0, 1.0)),
        marker_distance=float(np.linalg.norm(
            np.asarray(data.site_xpos[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, "a_marker_pelvis")], float)
            - pelvis_xyz)),
    )
    return {"actor": actor_obs(ctx), "priv": privileged_obs(ctx),
            "critic": critic_obs(ctx), "ctx": ctx}


def run_episode(model, spec: CaptureSpec, exec_spec: ExecSpec, repaired: dict,
                seed: int, *, ids: _Ids | None = None, tracks: RefTracks | None = None,
                source: DemoSource | None = None, verbose: bool = False) -> dict:
    """One demo rollout driven by the configured source; returns trace + meta."""
    import mujoco
    from solo.commands import LEAD_LEFT, LEAD_RIGHT, Command, Skill
    from solo.fall import FallDetector, fall_features
    from solo.markers import SHOT_PLAN, apply_markers, read_positions
    from solo.scene import stand_frame

    ids = ids or _Ids(model)
    stand_q, stand_ctrl = stand_frame(model)
    lo = np.asarray(model.actuator_ctrlrange[:, 0], float).copy()
    hi = np.asarray(model.actuator_ctrlrange[:, 1], float).copy()
    track = repaired["track"]
    M = len(track)
    data = mujoco.MjData(model)

    # initial state: the repaired track's first row (+ perturbation)
    q0, pinfo = _perturb(track[0], spec.perturb_sigma, seed, ids.base_qadr)
    data.qpos[:] = q0
    data.qvel[:] = 0.0
    data.ctrl[:] = stand_ctrl
    mujoco.mj_forward(model, data)

    # markers: world-anchored shot plan in front of the start pose
    yaw0 = float(np.arctan2(2.0 * (q0[3] * q0[6] + q0[4] * q0[5]),
                            1.0 - 2.0 * (q0[5] ** 2 + q0[6] ** 2)))
    apply_markers(model, data, q0[:2], yaw0, SHOT_PLAN)
    mujoco.mj_forward(model, data)
    markers = read_positions(model, data)

    lead_leg = LEAD_LEFT if exec_spec.lead_side == "left" else LEAD_RIGHT
    cmd = Command(vx=0.0, vy=0.0, wz=0.0,
                  stance_height=float(track[0][2]),
                  stance_width=float(abs(
                      data.site_xpos[ids.foot_site["left"]][1]
                      - data.site_xpos[ids.foot_site["right"]][1])),
                  skill_id=int(Skill.SHOT_DOUBLE_LEG), lead_leg=int(lead_leg))

    ctx = SourceContext(model=model, ids=ids, spec=spec, exec_spec=exec_spec,
                        repaired=repaired, tracks=tracks, stand_q=stand_q,
                        stand_ctrl=stand_ctrl, lo=lo, hi=hi, cmd=cmd,
                        markers=markers, track=track, M=M, q0=q0)
    src = source or make_source(spec, ctx, seed, verbose=verbose)
    src.prepare(ctx, seed, verbose=verbose)

    fall_det = FallDetector()
    rec = {k: [] for k in (
        "t", "ref_time", "segment", "phase", "qpos", "qvel", "ctrl", "action",
        "unit", "act_force", "obs_actor", "obs_priv", "obs_critic",
        "contact_foot", "contact_knee", "contact_hand", "grounded", "dorsal",
        "foot_load", "sole_pts", "knee_z", "com", "pelvis_z", "tilt_deg",
        "pitch_deg", "margin", "sat_frac", "foot_slip", "limit_prox")}
    prev_action = np.zeros(29)
    obs_prev = None

    def _phase_kind(t_abs: float) -> str:
        """The reference's own phase label at absolute track time ``t_abs``."""
        for kind, t0, t1 in (exec_spec.phase_segments or ()):
            if float(t0) <= t_abs < float(t1):
                return str(kind)
        return "?"
    terminated_t = None
    termination = None
    tick = 0
    wall0 = time.perf_counter()
    while tick < M:
        t = float(repaired["t"][tick])
        ctrl = np.asarray(src.control(data, tick, ctx, obs_prev), float)
        data.ctrl[:] = ctrl
        for _ in range(10):                      # 2 ms physics substeps
            mujoco.mj_step(model, data)

        f = fall_features(model, data)
        if fall_det.update(f, t) and terminated_t is None:
            terminated_t = t
            termination = {"t": t, "cause": "fall",
                           "status": fall_det.status()}

        sole = ids.sole_pts(data)
        touch = sole[:, :, 2].min(axis=1) < TOUCH_Z
        pts = sole[touch][:, :, :2].reshape(-1, 2) if touch.any() else np.zeros((0, 2))
        com = ids.com(data)
        sat = float(np.mean(np.abs(np.asarray(data.actuator_force, float))[:29]
                            >= 0.95 * ids.act_limit))
        prox = float(np.clip(1.0 - np.minimum(
            np.asarray(data.qpos[7:36], float) - ids.jnt_range[:, 0],
            ids.jnt_range[:, 1] - np.asarray(data.qpos[7:36], float)) / 0.1,
            0.0, 1.0).max())
        slip = 0.0
        vel = np.zeros(6)
        for s, ok in zip(SIDES, (f.contacts.left_foot, f.contacts.right_foot)):
            if not ok:
                continue
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE,
                                     int(ids.foot_site[s]), vel, 0)
            slip = max(slip, float(math.hypot(vel[3], vel[4])))
        dorsal = bool((f.contacts.torso or f.contacts.pelvis) and f.tilt_deg >= 45.0)
        rt = float(repaired["ref_time"][tick])
        shot_end = exec_spec.entry_end_t or 1.0
        phase = (min(rt / shot_end, 1.0) if math.isfinite(rt) and shot_end > 0
                 else 1.0)
        obs = _build_obs(model, data, ids, stand_q=stand_q, cmd=cmd,
                         prev_action=prev_action, phase=phase, markers=markers,
                         foot_slip=slip, sat_frac=sat, limit_prox=prox,
                         dorsal=dorsal)
        action = action_from_ctrl(ctrl, lo, hi, mode=spec.action_mode,
                                  base=stand_ctrl, scale=spec.residual_scale)
        unit = unit_from_ctrl(ctrl, lo, hi)

        rec["t"].append(t)
        rec["ref_time"].append(rt)
        rec["segment"].append(int(repaired["segment"][tick]))
        rec["phase"].append(_phase_kind(t))
        rec["qpos"].append(np.asarray(data.qpos, float).copy())
        rec["qvel"].append(np.asarray(data.qvel, float).copy())
        rec["ctrl"].append(np.asarray(ctrl, float))
        rec["action"].append(np.asarray(action, float))
        rec["unit"].append(np.asarray(unit, float))
        rec["act_force"].append(np.asarray(data.actuator_force, float)[:29].copy())
        rec["obs_actor"].append(obs["actor"])
        rec["obs_priv"].append(obs["priv"])
        rec["obs_critic"].append(obs["critic"])
        rec["contact_foot"].append(touch)
        rec["contact_knee"].append([f.contacts.left_knee, f.contacts.right_knee])
        rec["contact_hand"].append([f.contacts.left_hand, f.contacts.right_hand])
        rec["grounded"].append(bool(f.contacts.grounded))
        rec["dorsal"].append(dorsal)
        rec["foot_load"].append(ids.foot_load(data))
        rec["sole_pts"].append(sole)
        rec["knee_z"].append([data.site_xpos[ids.knee_site[s]][2] for s in SIDES])
        rec["com"].append(com)
        rec["pelvis_z"].append(float(f.pelvis_z))
        rec["tilt_deg"].append(float(f.tilt_deg))
        rec["pitch_deg"].append(ids.pitch_deg(data))
        rec["margin"].append(_margin(com[:2], pts))
        rec["sat_frac"].append(sat)
        rec["foot_slip"].append(slip)
        rec["limit_prox"].append(prox)

        prev_action = unit
        prev_ctrl = ctrl
        tick += 1
        if terminated_t is not None:
            break
    wall = time.perf_counter() - wall0

    out = {k: np.asarray(v) for k, v in rec.items()}
    out["t"] = np.asarray(rec["t"], float)
    out["ref_time"] = np.asarray(rec["ref_time"], float)
    out["segment"] = np.asarray(rec["segment"], np.int8)
    out["phase"] = np.asarray(rec["phase"])
    out["jnt_range"] = ids.jnt_range
    out["act_limit"] = ids.act_limit
    out["stand_ctrl"] = stand_ctrl
    out["stand_q"] = stand_q
    return {"trace": out, "terminated_t": terminated_t, "termination": termination,
            "perturbation": pinfo, "wall_s": wall, "n_ticks": int(tick),
            "m_planned": int(M), "cmd": cmd.as_dict(), "lead_leg": int(lead_leg),
            "start_qpos": q0, "yaw0": yaw0,
            "source": src.provenance(), "theta": getattr(src, "theta", None)}


# ---------------------------------------------------------------------------
# capture driver + dataset writing
# ---------------------------------------------------------------------------
def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=REPO, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:                                # pragma: no cover
        return None


def run_capture(spec: CaptureSpec, *, verbose: bool = True) -> dict:
    """Run every episode, write the dataset, optionally check each trace."""
    from solo.scene import load_solo_model
    model = load_solo_model()
    ids = _Ids(model)
    entry = load_track(spec.ref)
    recover = load_track(spec.recover_ref)
    stand = stance_qpos_for(spec) if spec.stand_s and spec.stand_s > 0 else None
    repaired = build_repaired_track(entry, recover, blend_s=spec.blend_s,
                                    hold_s=spec.hold_s, recover_s=spec.recover_s,
                                    start_offset_s=spec.start_offset_s,
                                    upto_s=spec.upto_s, stand_qpos=stand,
                                    stand_s=spec.stand_s)
    exec_spec, tracks = measure_spec(model, entry, recover, repaired, ids)

    name = spec.dir_name(entry["technique"] or "shot_entry_full")
    root = Path(spec.out_root) / name
    root.mkdir(parents=True, exist_ok=True)
    commit = _git_commit()
    episodes = []
    for i in range(spec.episodes):
        seed = spec.seed + i
        res = run_episode(model, spec, exec_spec, repaired, seed, ids=ids,
                          tracks=tracks, verbose=verbose)
        ep_dir = root / f"ep{i:02d}_seed{seed}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ep_dir / "trace.npz", **res["trace"])
        tracks.save(ep_dir / "spec.npz")
        if res.get("theta") is not None:
            np.savez_compressed(ep_dir / "theta.npz", theta=res["theta"])
        report = None
        if spec.check:
            report = check_executability(res["trace"], exec_spec)
            (ep_dir / "check.json").write_text(
                json.dumps(report.as_dict(), indent=1))
        meta = {
            "name": name, "episode": i, "seed": seed,
            "config": spec.as_dict(),
            "reference": {"entry": _rel(spec.ref), "entry_sha256": entry["sha256"],
                          "recover": _rel(spec.recover_ref),
                          "recover_sha256": recover["sha256"],
                          "technique": entry["technique"],
                          "net_travel_m": entry["meta"].get("net_travel_m")},
            "spec": exec_spec.scalars(),
            "track": {"ticks": len(repaired["t"]),
                      "duration_s": float(repaired["t"][-1]),
                      "entry_ticks": repaired["n_entry"],
                      "blend_ticks": repaired["n_blend"],
                      "recover_ticks": repaired["n_recover"],
                      "stand_ticks": repaired["n_stand"],
                      "hold_ticks": repaired["n_hold"]},
            "source": res["source"],
            "episode": {"n_ticks": res["n_ticks"], "wall_s": round(res["wall_s"], 3),
                        "terminated_t": res["terminated_t"],
                        "termination": res["termination"],
                        "perturbation": res["perturbation"],
                        "cmd": res["cmd"],
                        "action_mode": spec.action_mode,
                        "residual_scale": spec.residual_scale},
            "provenance": {"git_commit": commit, "control_hz": 50,
                           "physics_dt_s": 0.002, "physics_substeps": 10,
                           "scene": "src/solo/scene.py (single G1, a_ prefix)",
                           "reproduce": spec.reproduce(entry["technique"]
                                                       or "shot_entry_full")},
            "check": report.as_dict() if report else None,
        }
        (ep_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        episodes.append({"episode": i, "seed": seed, "dir": str(ep_dir),
                         "n_ticks": res["n_ticks"],
                         "wall_s": round(res["wall_s"], 3),
                         "terminated_t": res["terminated_t"],
                         "source": res["source"].get("kind"),
                         "landmark_rms_m": (res["source"].get("final") or {}
                                            ).get("rms_m"),
                         "check_ok": bool(report.ok) if report else None,
                         "failed": report.failed if report else None})
        if verbose:
            verdict = ("-" if report is None else
                       ("PASS" if report.ok else "FAIL:" + ",".join(report.failed)))
            print(f"[{i + 1}/{spec.episodes}] {ep_dir.name}: "
                  f"{res['n_ticks']} ticks, {res['wall_s']:.1f}s wall, "
                  f"terminated={res['terminated_t']}, check={verdict}")
    index = {"name": name, "config": spec.as_dict(), "git_commit": commit,
             "reference": {"entry": _rel(spec.ref), "entry_sha256": entry["sha256"],
                           "recover": _rel(spec.recover_ref),
                           "recover_sha256": recover["sha256"]},
             "spec": exec_spec.scalars(), "episodes": episodes}
    (root / "index.json").write_text(json.dumps(index, indent=1))
    return {"dir": str(root), "index": index, "spec": exec_spec,
            "episodes": episodes}


def load_spec(ep_dir: str | Path) -> ExecSpec:
    """Rebuild the ExecSpec of a written episode (meta.json + spec.npz)."""
    d = Path(ep_dir)
    meta = json.loads((d / "meta.json").read_text())
    s = meta["spec"]
    spec = ExecSpec(
        technique=s.get("technique", ""), role=s.get("role", "a"),
        sources=tuple(s.get("sources", ())), sha256=tuple(s.get("sha256", ())),
        dt=float(s.get("dt", DT)), duration_s=float(s.get("duration_s", 0.0)),
        n_frames=int(s.get("n_frames", 0)),
        lead_side=s.get("lead_side", "right"),
        lead_plant_t=s.get("lead_plant_t"),
        knee_side=s.get("knee_side", "left"),
        knee_min_z=float(s.get("knee_min_z", 0.0)),
        knee_min_t=float(s.get("knee_min_t", 0.0)),
        knee_contact_z=float(s.get("knee_contact_z", 0.0)),
        knee_contact_t=s.get("knee_contact_t"),
        travel_forward_m=float(s.get("travel_forward_m", 0.0)),
        travel_lateral_m=float(s.get("travel_lateral_m", 0.0)),
        travel_net_m=float(s.get("travel_net_m", 0.0)),
        entry_end_t=float(s.get("entry_end_t", 0.0)),
        phase_segments=[tuple(x) for x in s.get("phase_segments", [])],
        rise_pelvis_z=float(s.get("rise_pelvis_z", 0.0)),
        rise_knee_z=float(s.get("rise_knee_z", 0.0)),
        tol=Tol(**s.get("tol", {})),
        tracks=RefTracks.load(d / "spec.npz"))
    return spec


if __name__ == "__main__":  # self-check (no sim): plan a default capture
    plan = plan_capture(CaptureSpec(episodes=1))
    print(json.dumps(plan, indent=1)[:1400])
