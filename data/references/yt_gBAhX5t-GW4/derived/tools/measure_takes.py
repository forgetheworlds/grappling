"""Take selection + phase/transition extraction + rubric-aligned metrics.

Reads pose/landmarks.npz, detects demonstration takes per chapter, segments
each take into phases (stance / crouch / knee-down / moving), extracts the
transitions between phases, and measures the geometry the QUALITY_RUBRIC
scores (stance width/depth, level-change drop, lead-foot travel, knee height,
trail-foot reposition, recovery time, cadence, torso pitch, head, hands).

All horizontal quantities are expressed in the take's own body frame built
from the time-averaged hip line (lateral) and its facing (forward), with the
feet floored at 0. Values depend on mediapipe's monocular metric scale; every
report therefore also prints dimensionless ratios (relative to standing body
height) and the G1-scaled equivalent (coach height 1.78 m assumed -> G1 1.32 m).

Run:
  python .../measure_takes.py --npz .../pose/landmarks.npz \
      --json .../derived/take_metrics.json [--chapters 2,5,7,10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from analyze_pose import (LANK, LHIP, LKNE, LSHO, LWRI, NOSE, Pose,  # noqa: E402
                          RANK, RHIP, RKNE, RSHO, RWRI, detect_takes,
                          merge_takes, phase_labels)

# assumed coach standing height (mediapipe world landmarks are ~metric);
# used ONLY to print a metric hint — all ratios are scale-free
COACH_H = 1.78
G1_H = 1.32
ALPHA = G1_H / COACH_H

LAT_SIGN = None  # set per take: lateral axis points to the person's left


def body_frame(w: np.ndarray) -> dict:
    """Per-frame anchors in the take's canonical frame.

    w: (F,33,3) world landmarks after y-flip (Y-up). Returns arrays:
    lat/ax_f (F,3) horizontal unit axes, floor (F,), pelvis (F,3),
    and a function to project any landmark track into (forward, left, up).
    """
    f = w.shape[0]
    lat = w[:, LHIP] - w[:, RHIP]
    lat[:, 1] = 0.0
    lat /= np.maximum(np.linalg.norm(lat, axis=1, keepdims=True), 1e-9)
    up = np.tile(np.array([0.0, 1.0, 0.0]), (f, 1))
    fwd = np.cross(up, lat)
    fwd[:, 1] = 0.0
    fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)
    # sign: forward should point from pelvis toward the head
    core = 0.5 * (w[:, LHIP] + w[:, RHIP])
    head = w[:, NOSE] - core
    head[:, 1] = 0.0
    dot = np.einsum("ij,ij->i", fwd, head)
    flip = np.where(dot < 0, -1.0, 1.0)[:, None]
    fwd = fwd * flip
    lat = lat * flip     # keep left-handed consistency (det(up,lat,fwd))
    foot = w[:, [LANK, RANK, 29, 30, 31, 32]]
    floor = foot[:, :, 1].min(axis=1)
    return {"lat": lat, "fwd": fwd, "floor": floor, "core": core}


def image_metric(p: "Pose", idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Metric image-plane coordinates for frames ``idx``: (F,33,2) metres.

    u = rightward in the image, v = upward in the image, origin at the first
    frame's pelvis/ankle centre. The per-frame scale s (metres per normalized
    image unit) is self-calibrated from body height (nose to mid-ankle) in the
    image vs the world-landmark track, so absolute displacements survive the
    monocular scale ambiguity. Reliable in the image plane; motion toward or
    away from the camera is compressed (perspective not corrected).
    """
    img = p.img[idx]
    w = p.world[idx]
    body_px = 0.5 * (img[:, LANK, 1] + img[:, RANK, 1]) - img[:, NOSE, 1]
    mid_ank = 0.5 * (w[:, LANK] + w[:, RANK])
    body_m = np.linalg.norm(w[:, NOSE] - mid_ank, axis=1)
    s = body_m / np.maximum(body_px, 1e-6)
    x0 = float(np.mean(img[0, [LHIP, RHIP], 0]))
    y0 = float(np.mean(img[0, [LANK, RANK], 1]))
    u = (img[..., 0] - x0) * s[:, None]
    v = (y0 - img[..., 1]) * s[:, None]
    return np.stack([u, v], axis=-1), s


def proj(bf: dict, p: np.ndarray) -> np.ndarray:
    """(F,3) -> (F,3) [forward, left, up] coordinates relative to the pelvis."""
    rel = p - bf["core"]
    return np.stack([
        np.einsum("ij,ij->i", rel, bf["fwd"]),
        np.einsum("ij,ij->i", rel, bf["lat"]),
        rel[:, 1],
    ], axis=1)


def ang_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at b, degrees, per frame (a,b,c: (F,3))."""
    u = a - b
    v = c - b
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", u, v), -1, 1)))


def torso_pitch_deg(w: np.ndarray) -> np.ndarray:
    """Angle of the pelvis->head axis from vertical (deg)."""
    core = 0.5 * (w[:, LHIP] + w[:, RHIP])
    v = w[:, NOSE] - core
    vn = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.clip(vn[:, 1], -1, 1)))


def measure_phase(w: np.ndarray, bf: dict, i0: int, i1: int,
                  standing_h: float) -> dict:
    """Geometry over frames [i0, i1] (inclusive)."""
    sl = slice(i0, i1 + 1)
    bfs = {k: bf[k][sl] for k in ("lat", "fwd", "floor", "core")}
    core = bfs["core"]
    ph = core[:, 1] - bfs["floor"]
    lank = proj(bfs, w[sl, LANK])
    rank = proj(bfs, w[sl, RANK])
    # stance width = separation along the hip-line (left) axis; depth = along forward
    width = np.abs(lank[:, 1] - rank[:, 1])
    depth = np.abs(lank[:, 0] - rank[:, 0])
    knee_l = ang_at(w[sl, LHIP], w[sl, LKNE], w[sl, LANK])
    knee_r = ang_at(w[sl, RHIP], w[sl, RKNE], w[sl, RANK])
    pitch = torso_pitch_deg(w[sl])
    head_h = w[sl, NOSE, 1] - core[:, 1]
    lw = proj(bfs, w[sl, LWRI])
    rw = proj(bfs, w[sl, RWRI])
    knee_h = np.minimum(w[sl, LKNE, 1], w[sl, RKNE, 1]) - bfs["floor"]
    return {
        "n": int(i1 - i0 + 1),
        "pelvis_h_m": round(float(np.median(ph)), 4),
        "pelvis_h_min_m": round(float(np.min(ph)), 4),
        "pelvis_h_max_m": round(float(np.max(ph)), 4),
        "pelvis_h_over_standing": round(float(np.median(ph) / standing_h), 3),
        "stance_width_m": round(float(np.median(width)), 4),
        "stance_width_over_h": round(float(np.median(width) / standing_h), 3),
        "stance_width_g1_m": round(float(np.median(width) * ALPHA), 4),
        "stance_depth_m": round(float(np.median(depth)), 4),
        "stance_depth_over_h": round(float(np.median(depth) / standing_h), 3),
        "stance_depth_g1_m": round(float(np.median(depth) * ALPHA), 4),
        "knee_flexion_deg": round(float(np.median(np.concatenate([knee_l, knee_r]))), 1),
        "torso_pitch_deg": round(float(np.median(pitch)), 1),
        "head_above_pelvis_m": round(float(np.median(head_h)), 4),
        "knee_min_height_m": round(float(np.min(knee_h)), 4),
        "lwri_rel_pelvis_flm": [round(float(x), 3) for x in np.median(lw, axis=0)],
        "rwri_rel_pelvis_flm": [round(float(x), 3) for x in np.median(rw, axis=0)],
    }


def segment_phases(labels: list[str], t: np.ndarray, i0: int,
                   min_dur: float = 0.30) -> list[dict]:
    """Group consecutive identical labels (take-local, length n) into phase
    segments with global frame indices; absorb segments shorter than min_dur."""
    segs = []
    start = 0
    n = len(labels)
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[start]:
            segs.append({"label": labels[start], "i0": i0 + start,
                         "i1": i0 + i - 1,
                         "t0": float(t[i0 + start]), "t1": float(t[i0 + i - 1])})
            start = i
    # absorb short segments into the previous one
    merged: list[dict] = []
    for s in segs:
        if merged and (s["t1"] - s["t0"]) < min_dur:
            merged[-1]["i1"] = s["i1"]
            merged[-1]["t1"] = s["t1"]
        else:
            merged.append(s)
    for s in merged:
        s["dur_s"] = round(s["t1"] - s["t0"], 3)
    return merged


def take_report(p: Pose, s: dict, tk: dict) -> dict:
    w = s["w"]
    i0, i1 = tk["i0"], tk["i1"]
    w_loc = w[i0:i1 + 1]
    bf = body_frame(w_loc)
    # standing body height = nose height above floor at the most upright frame
    bh = w_loc[:, NOSE, 1] - bf["floor"]
    standing_h = float(np.max(bh))
    core_ph = bf["core"][:, 1] - bf["floor"]
    stance_ph = float(np.percentile(core_ph, 75))
    lab = phase_labels(p, s, i0, i1, stance_ph)
    phases = segment_phases(lab, p.t, i0)
    # transitions between phases with real posture change
    trans = []
    for a, b in zip(phases[:-1], phases[1:]):
        ma = measure_phase(w_loc, bf, a["i0"] - i0, a["i1"] - i0, standing_h)
        mb = measure_phase(w_loc, bf, b["i0"] - i0, b["i1"] - i0, standing_h)
        dph = mb["pelvis_h_m"] - ma["pelvis_h_m"]
        trans.append({
            "from": a["label"], "to": b["label"],
            "t0": round(a["t1"], 3), "t1": round(b["t0"], 3),
            "dur_s": round(b["t0"] - a["t1"], 3),
            "d_pelvis_h_m": round(dph, 4),
            "d_pelvis_h_g1_m": round(dph * ALPHA, 4),
        })
    out = dict(tk)
    out.update({
        "standing_pelvis_h_m": round(standing_h, 4),
        "phases": phases,
        "transitions": trans,
        "phase_metrics": {f"{ph['label']}_{i}": measure_phase(
            w_loc, bf, ph["i0"] - i0, ph["i1"] - i0, standing_h)
            for i, ph in enumerate(phases)},
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--chapters", default="2,5,7,10")
    ap.add_argument("--speed-floor", type=float, default=0.25)
    args = ap.parse_args()

    p = Pose(Path(args.npz))
    s = p.signals()
    wanted = [int(x) for x in args.chapters.split(",") if x.strip()]
    report = {"npz": str(args.npz), "coach_h_assumed_m": COACH_H,
              "alpha_g1_over_coach": round(ALPHA, 4), "chapters": {}}
    for ch in wanted:
        takes = merge_takes(detect_takes(p, s, ch, speed_floor=args.speed_floor))
        takes = sorted(takes, key=lambda k: -k["mean_conf"])
        reps = [take_report(p, s, tk) for tk in takes]
        report["chapters"][ch] = {"n_takes": len(reps), "takes": reps}
        if not reps:
            print(f"ch{ch}: NO demonstration take detected")
            continue
        best = reps[0]
        print(f"ch{ch}: {len(reps)} take(s); best {best['t0']:.1f}-{best['t1']:.1f}s "
              f"conf={best['mean_conf']:.2f} "
              f"phases={[ph['label'] for ph in best['phases']]}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
