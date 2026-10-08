"""Retarget-world placement: scale, facing axis, centroid, Z-up.

Takes role-resolved GrappleMap landmark tracks (still Y-up, human-scale,
``(F, 23, 3)`` per robot) and produces solver targets in the MuJoCo world.

Steps (per technique, one shared world for both robots):
1. Per-player uniform scale s_p = LSQ fit of G1 segment lengths vs that
   player's median GrappleMap segment lengths over segments
   {Core→Neck, Neck→Head, Hip→Knee, Knee→Ankle} (per side). The two G1s are
   identical hardware, and inter-player contact relationships (attack depth,
   sprawl chest pressure) must survive; the pair is therefore scaled by the
   MEAN of the two per-player estimates about the pair centroid (per-player
   estimates are reported and cross-checked — they agree within a few %).
2. Rotate about the vertical so that the time-averaged
   (robot-A core − robot-B core) horizontal direction maps to −x:
   robot A (attacker/recoverer) starts on the −x side facing +x (the G1
   faces +x: toe sites at ankle_roll-local +x), robot B on +x facing −x.
3. Translate so the time-averaged pair centroid (mean of the two cores) sits
   at the world origin, scale about it, then convert Y-up → Z-up:
   (x, y, z)_gm → (x, z, y)_mj (notes.md interface contract).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gmframe import CORE, J
from .landmarks import g1_reference_geometry

#: GM joint pairs (per side where applicable) used for scale estimation
_SCALE_SEGMENTS = (("Core", "Neck"), ("Neck", "Head"),
                   ("LeftHip", "LeftKnee"), ("RightHip", "RightKnee"),
                   ("LeftKnee", "LeftAnkle"), ("RightKnee", "RightAnkle"))


@dataclass
class WorldPlacement:
    targets_a: np.ndarray   # (F, 23, 3) Z-up, robot-scale, world frame
    targets_b: np.ndarray
    scale_a: float
    scale_b: float
    scale: float            # applied (mean)
    heading: float          # rotation applied about vertical (rad)
    g1_segments: dict


def estimate_player_scale(track: np.ndarray) -> float:
    """Uniform scale G1/GM for one player's landmark track (F, 23, 3)."""
    geo = g1_reference_geometry()
    g1 = {
        ("Core", "Neck"): geo["segments"]["core_neck"],
        ("Neck", "Head"): geo["segments"]["neck_head"],
        ("LeftHip", "LeftKnee"): geo["segments"]["thigh"],
        ("RightHip", "RightKnee"): geo["segments"]["thigh_r"],
        ("LeftKnee", "LeftAnkle"): geo["segments"]["shank"],
        ("RightKnee", "RightAnkle"): geo["segments"]["shank_r"],
    }
    num = den = 0.0
    for seg in _SCALE_SEGMENTS:
        a, b = J[seg[0]], J[seg[1]]
        gm_len = float(np.median(np.linalg.norm(track[:, a] - track[:, b], axis=-1)))
        if gm_len < 1e-6:
            continue
        num += g1[seg] * gm_len
        den += gm_len * gm_len
    return num / den


def place_world(track_a: np.ndarray, track_b: np.ndarray) -> WorldPlacement:
    """track_{a,b}: (F, 23, 3) Y-up GrappleMap landmarks, role-resolved."""
    track_a = np.asarray(track_a, dtype=np.float64)
    track_b = np.asarray(track_b, dtype=np.float64)
    assert track_a.shape == track_b.shape and track_a.ndim == 3

    s_a = estimate_player_scale(track_a)
    s_b = estimate_player_scale(track_b)
    s = 0.5 * (s_a + s_b)

    # facing: time-averaged horizontal direction from B to A -> −x, i.e.
    # rotate so that mean(a_core − b_core) horizontal points to −x.
    v = (track_a[:, CORE] - track_b[:, CORE])[:, [0, 2]]
    v = v.mean(0)
    target = np.array([-1.0, 0.0])
    heading = float(np.arctan2(v[0] * target[1] - v[1] * target[0],
                               v[0] * target[0] + v[1] * target[1]))
    # rotate GM horizontal coords by `heading` (same rot_y algebra):
    # x' = cos h · x + sin h · z ; z' = −sin h · x + cos h · z
    def _rot(p: np.ndarray) -> np.ndarray:
        out = p.copy()
        c, sn = np.cos(heading), np.sin(heading)
        out[..., 0] = c * p[..., 0] + sn * p[..., 2]
        out[..., 2] = -sn * p[..., 0] + c * p[..., 2]
        return out

    # horizontal centroid only: the ground must stay at z=0 after scaling
    # (notes.md interface contract: gm y=0 == mj z=0)
    c3 = 0.5 * (track_a[:, CORE] + track_b[:, CORE]).mean(0)
    centroid = np.array([c3[0], 0.0, c3[2]])
    ta = s * (_rot(track_a - centroid))
    tb = s * (_rot(track_b - centroid))

    # Y-up -> Z-up, chirality-preserving: rotation about +x by +90 deg maps
    # y_gm -> z_mj and z_gm -> -y_mj, i.e. (x,y,z)_gm -> (x,-z,y)_mj.
    # NOTE: the naive axis swap (x,y,z)->(x,z,y) is an IMPROPER transform
    # (det=-1, mirrors the pair): every player's left/right flips and the
    # solver then matches the mirrored targets by facing AWAY from the
    # opponent (verified on STANCE: cross-hip fit 0.15 m vs 0.06 m; robots
    # fall under PD tracking). Ground plane still maps exactly: gm y=0 ==
    # mj z=0.
    out_a = np.empty_like(ta)
    out_b = np.empty_like(tb)
    for src, out in ((ta, out_a), (tb, out_b)):
        out[:, :, 0] = src[:, :, 0]
        out[:, :, 1] = -src[:, :, 2]
        out[:, :, 2] = src[:, :, 1]
    return WorldPlacement(out_a, out_b, s_a, s_b, s, heading,
                          g1_reference_geometry()["segments"])
