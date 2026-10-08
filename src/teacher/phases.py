"""Reference phase segmentation for gain scheduling (phase-3 teacher).

The npz meta records the keyframe COUNT and montage junctions but not the
keyframe timestamps of the final 50 Hz track (they are consumed inside the
phase-2 retime/resample). We therefore label every reference frame from its
own kinematics — pelvis height, vertical speed, foot planting — which is the
same information the keyframe segments carry (level change, dive, recover,
prone), is deterministic, and needs no rebuild of the retarget pipeline.

Labels (per robot, per reference frame):
  STAND  upright and slow        pelvis z >= 0.60, |vz| <= 0.35 m/s
  RISE   rising recovery         vz > 0.35 m/s, pelvis z >= 0.42
  DIVE   dropping into a shot    vz < -0.35 m/s, 0.60 > pelvis z >= 0.42
  LOW    level change / squat    0.42 <= pelvis z < 0.60 (slow)
  PRONE  intended ground work    pelvis z < 0.42  (takedown finishes,
         sprawled defender, stand-up start — stabilizer fully off)

``label_frames`` produces the per-frame labels; ``segments`` merges runs into
a monotone phase table [(kind, t0, t1), ...] covering [0, T] exactly.
"""

from __future__ import annotations

import numpy as np

KINDS = ("STAND", "RISE", "DIVE", "LOW", "PRONE")

#: numeric thresholds on the reference (SI units)
PRONE_Z = 0.42          # below stay-up threshold 0.45 with margin
LOW_Z = 0.60            # below this a slow pose is a level-change/squat
VZ_FAST = 0.35          # m/s vertical speed separating motion from hold
SMOOTH_S = 0.15         # moving-average window for vz (chatter suppression)


def _smooth(x: np.ndarray, t: np.ndarray, win_s: float) -> np.ndarray:
    """Zero-phase moving average over an (approximately) uniform time grid."""
    n = max(1, int(round(win_s / (t[1] - t[0]))))
    if n <= 1:
        return x
    kernel = np.ones(n) / n
    pad = np.concatenate([np.full(n - 1, x[0]), x, np.full(n - 1, x[-1])])
    return np.convolve(pad, kernel, mode="valid")[: len(x)]


def label_frames(t: np.ndarray, pelvis_z: np.ndarray) -> np.ndarray:
    """Per-frame phase labels (dtype '<U5') from the reference pelvis track."""
    t = np.asarray(t, float)
    z = np.asarray(pelvis_z, float)
    vz = _smooth(np.gradient(z, t), t, SMOOTH_S)
    labels = np.empty(len(t), dtype="<U5")
    for i in range(len(t)):
        if z[i] < PRONE_Z:
            labels[i] = "PRONE"
        elif vz[i] < -VZ_FAST:
            labels[i] = "DIVE"
        elif z[i] < LOW_Z:
            labels[i] = "LOW"
        elif vz[i] > VZ_FAST:
            labels[i] = "RISE"
        else:
            labels[i] = "STAND"
    return labels


def segments(t: np.ndarray, labels: np.ndarray) -> list[tuple[str, float, float]]:
    """Merge label runs into a monotone phase table covering [t0, tN]."""
    t = np.asarray(t, float)
    out: list[tuple[str, float, float]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((str(labels[start]), float(t[start]),
                        float(t[min(i, len(t) - 1)])))
            start = i
    return out


def phase_table(t: np.ndarray, pelvis_z_a: np.ndarray,
                pelvis_z_b: np.ndarray) -> dict[str, list[tuple[str, float, float]]]:
    """Per-robot segment tables for one technique reference."""
    return {"a": segments(t, label_frames(t, pelvis_z_a)),
            "b": segments(t, label_frames(t, pelvis_z_b))}


def labels_at(labels: np.ndarray, t: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Label of the nearest reference frame for each query time."""
    idx = np.clip(np.searchsorted(t, t_query) - 1, 0, len(t) - 1)
    return labels[idx]


if __name__ == "__main__":  # self-check
    t = np.arange(0.0, 3.0, 0.02)
    z = np.where(t < 1.0, 0.75 - 0.1 * t,
                 np.where(t < 1.5, 0.75 - 0.5 * (t - 1.0), 0.4))
    lab = label_frames(t, z)
    segs = segments(t, lab)
    assert segs[0][1] == 0.0 and segs[-1][2] == t[-1]
    for (_, t0, t1), (_, u0, _) in zip(segs, segs[1:]):
        assert t0 < t1 <= u0, (t0, t1, u0)
    print("phases self-check OK:", segs)
