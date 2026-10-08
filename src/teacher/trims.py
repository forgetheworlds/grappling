"""Per-technique reference pose trims (stabilizer candidate 5).

Why this file exists
--------------------
The retargeted references are *schematic* GrappleMap poses scaled onto the G1
(scale 0.75-0.79).  They preserve the drawn relationships, but a schematic
pose is not necessarily a *static equilibrium* for the G1: measured on
``STANCE`` the ankle actuator must supply 36 Nm (of its +-50 Nm limit) just to
hold the pose, the CoM sits behind the effective support, and the robot topples
in 0.8-1.5 s even when the balance feedback is running (see the report's
"stabilizer-only cannot hold STANCE" experiment).  The same is true to a
lesser degree for the first/last frames of the shot techniques.

Fix (allowed, minimal, measured): a small constant joint-space trim added to
the reference, searched *per technique and per robot* with the physics as the
oracle — ``scripts/tune_teacher.py`` grid-searches four sagittal parameters
(hip_pitch, knee, ankle_pitch applied to both legs, plus waist_pitch) and
keeps the trim with the best stay-up fraction (and a scorer floor).

The trim is a *reference-pose adjustment*, not a stabilizer: it is applied to
``q_ref`` itself and is what makes the posture statically holdable; the
stabilizer (controller.py) still does all the feedback on top.  Trim vectors
are in radians, in the robot's own local joint order (legs are symmetric).
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
#: tuning output of scripts/tune_teacher.py (optional: absent = no trims)
TRIM_PATH = REPO / "data" / "teacher_trims.json"

#: param keys: hip (hip_pitch, both legs), knee (both), ankle (ankle_pitch,
#: both), waist (waist_pitch), dz (settled pelvis-height correction, m).
#: Values are radians (dz: metres).
TrimSpec = dict[str, float]


def load_trims(path: Path | str = TRIM_PATH) -> dict[str, dict[str, TrimSpec]]:
    """Read the measured trim table; {} when no tuning has been run yet."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as fh:
        raw = json.load(fh)
    return {tech: {prefix: {k: float(v) for k, v in spec.items()}
                   for prefix, spec in per.items()}
            for tech, per in raw.get("trims", {}).items()}


#: measured trims (scripts/tune_teacher.py, log in reports/2026-10-08/teacher.md)
POSE_TRIM: dict[str, dict[str, TrimSpec]] = load_trims()


def trim_vector(spec: TrimSpec | None, rear: int | None = None) -> np.ndarray:
    """29-joint trim vector from a TrimSpec (local joint order).

    ``hip/knee/ankle`` apply to both legs; ``hip_rear/knee_rear/ankle_rear``
    apply to the *rear* leg only (``rear`` = 0 for the left leg, 1 for the
    right, as measured from the reference's own foot positions).  The rear-leg
    keys are the operator-preferred stance repair: extending the rear leg moves
    that foot backwards, growing the support polygon behind the line of action
    so the CoM falls inside it, instead of flattening the torso or reducing the
    crouch depth (see the report's "stance geometry trims").
    """
    out = np.zeros(29)
    if not spec:
        return out
    hip = float(spec.get("hip", 0.0))
    knee = float(spec.get("knee", 0.0))
    ankle = float(spec.get("ankle", 0.0))
    waist = float(spec.get("waist", 0.0))
    roll = float(spec.get("ankle_roll", 0.0))
    for side in (0, 6):
        out[side + 0] += hip
        out[side + 3] += knee
        out[side + 4] += ankle
        out[side + 5] += roll
    if rear in (0, 1):
        base = 0 if rear == 0 else 6
        out[base + 0] += float(spec.get("hip_rear", 0.0))
        out[base + 3] += float(spec.get("knee_rear", 0.0))
        out[base + 4] += float(spec.get("ankle_rear", 0.0))
    # frontal axis: widen the stance (splay_rear / splay_lead, hip_roll with
    # opposite signs on the two legs) — operator rule (b) for sideways falls
    sr = float(spec.get("splay_rear", 0.0))
    sl = float(spec.get("splay_lead", 0.0))
    if rear in (0, 1):
        for side in (0, 1):
            val = sr if side == rear else sl
            out[6 * side + 1] += val if side == 1 else -val
    out[7] += float(spec.get("splay", 0.0))
    out[1] -= float(spec.get("splay", 0.0))
    out[14] += waist          # waist_pitch
    return out


def trim_scale(z_ref: float, kind: str | None = None) -> float:
    """How much of the trim applies at a reference pelvis height ``z_ref``.

    Continuous in the reference height (no phase-boundary jumps): zero below
    0.42 m (ground work, dives, prone), full above 0.60 m (standing holds).
    """
    lo, hi = 0.42, 0.60
    return float(np.clip((z_ref - lo) / (hi - lo), 0.0, 1.0))


def trim_for(technique: str, prefix: str) -> TrimSpec:
    """TrimSpec for (technique, robot prefix); empty when none was measured."""
    return POSE_TRIM.get(technique, {}).get(prefix, {})


#: minimum stance width (m): lateral separation of the two feet's mean sole
#: sites, measured in the pelvis frame.  Operator rule (frontal axis): "if
#: it's falling sideways, legs are too close together — a little back and away
#: from the other leg can help", and the stance-width command must default to
#: a laterally stable width.  Reference stances measure 0.17-0.29 m; the
#: model's own verified-stable ``both_stand`` keyframe measures 0.237 m; the
#: floor is set above both.
MIN_STANCE_WIDTH = 0.30


def _foot_local(model, qpos: np.ndarray, prefix: str) -> dict:
    """Mean sole-site positions of both feet in the pelvis frame (single robot)."""
    from .posture import foot_local
    loc = foot_local(model, prefix, np.asarray(qpos, float))
    return {"left": loc[0], "right": loc[1]}


def measure_stance_width(model, qpos: np.ndarray, prefix: str = "a_") -> float:
    """Lateral foot separation (m) of a single-robot qpos, pelvis frame."""
    f = _foot_local(model, qpos, prefix)
    return float(f["left"][1] - f["right"][1])


def enforce_stance_width(model, qpos: np.ndarray, min_width: float | None = None,
                         prefix: str = "a_", max_splay: float = 0.35,
                         iters: int = 3) -> tuple[np.ndarray, np.ndarray, float]:
    """Widen a stance qpos to at least ``min_width`` of lateral foot separation.

    Returns ``(qpos_out, splay, width)``: the widened 36-dof qpos, the 29-joint
    hip_roll offset that realises it (also usable as a trim contribution), and
    the measured width after enforcement.  The widening direction and the
    width-per-radian slope are measured by FK (sign-agnostic), not assumed.
    """
    q = np.asarray(qpos, float).copy()
    lim = MIN_STANCE_WIDTH if min_width is None else float(min_width)
    w0 = measure_stance_width(model, q, prefix)
    if w0 >= lim:
        return q, np.zeros(29), w0

    def width_at(mag: float, direction: float) -> float:
        qq = q.copy()
        qq[7:36] += _splay_vec(direction * mag)
        return measure_stance_width(model, qq, prefix)

    probe = 0.05
    wp, wm = width_at(probe, 1.0), width_at(probe, -1.0)
    if max(wp, wm) <= w0 + 1e-6:
        return q, np.zeros(29), w0                # geometry cannot widen
    direction = 1.0 if wp >= wm else -1.0
    slope = (max(wp, wm) - w0) / probe
    mag = float(np.clip((lim - w0) / slope, 0.0, max_splay))
    for _ in range(iters):
        w = width_at(mag, direction)
        if w >= lim - 1e-4:
            break
        slope = (w - w0) / max(mag, 1e-9)
        if slope <= 1e-6:
            break
        mag = float(np.clip(mag + (lim - w) / slope, 0.0, max_splay))
    splay = _splay_vec(direction * mag)
    q[7:36] += splay
    return q, splay, measure_stance_width(model, q, prefix)


def _splay_vec(delta: float) -> np.ndarray:
    """29-joint hip_roll splay vector (both feet away from the midline)."""
    v = np.zeros(29)
    v[1] -= delta            # left hip_roll
    v[7] += delta            # right hip_roll
    return v
