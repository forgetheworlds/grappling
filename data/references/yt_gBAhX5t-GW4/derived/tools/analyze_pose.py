"""Analyse the pose pass: canonical body frame, demonstration takes, phase
transitions, and the compact numeric spec.

Reads pose/landmarks.npz (from run_pose.py). Everything here is derived from
the 2D image landmarks (reliable: x/y in the image plane, normalized by body
height) and the mediapipe world landmarks (approximately metric, hip-centred;
reliable for relative joint geometry, approximate for absolute depth/scale).

Outputs a JSON report to stdout / --json:
  {frame_scale, takes: [...], transitions: [...], spec: {...}}

Run from repo root:
  python data/references/yt_gBAhX5t-GW4/derived/tools/analyze_pose.py \
      --npz .../pose/landmarks.npz --json /tmp/analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
NPZ = REPO / "data/references/yt_gBAhX5t-GW4/pose/landmarks.npz"

# mediapipe pose landmark indices
NOSE, LSHO, RSHO = 0, 11, 12
LELB, RELB, LWRI, RWRI = 13, 14, 15, 16
LHIP, RHIP = 23, 24
LKNE, RKNE = 25, 26
LANK, RANK = 27, 28
LHEE, RHEE = 29, 30
LTOE, RTOE = 31, 32

FULLBODY = (NOSE, LSHO, RSHO, LELB, RELB, LWRI, RWRI,
            LHIP, RHIP, LKNE, RKNE, LANK, RANK)
ACTIVE_JOINTS = (LHIP, RHIP, LKNE, RKNE, LANK, RANK, LWRI, RWRI, NOSE)
LOWER_JOINTS = (LHIP, RHIP, LKNE, RKNE, LANK, RANK)


class Pose:
    def __init__(self, npz_path: Path):
        self.path = str(npz_path)
        d = np.load(npz_path)
        self.d = d
        self.t = d["t"].astype(np.float64)
        self.img = d["img_lm"].astype(np.float64)      # (F,33,3) normalized
        self.world = d["world_lm"].astype(np.float64)  # (F,33,3) metres, hip-origin
        self.vis = d["visibility"].astype(np.float64)
        self.pres = d["presence"].astype(np.float64)
        self.det = d["detected"].astype(bool)
        self.conf = d["conf"].astype(np.float64)
        self.chapter = d["chapter"].astype(np.int64)

    # -- body-frame quantities (per frame, from image landmarks) -------------
    def body_px(self) -> np.ndarray:
        """Body height in pixels per frame (nose to mid-ankle, image y)."""
        y_nose = self.img[:, NOSE, 1]
        y_ank = 0.5 * (self.img[:, LANK, 1] + self.img[:, RANK, 1])
        return (y_ank - y_nose)

    def feet_px(self) -> np.ndarray:
        """(F,2,3) ankle image positions."""
        return self.img[:, [LANK, RANK], :]

    def smooth_world(self, win_s: float = 0.20) -> np.ndarray:
        """Zero-phase smoothing of the world track (and a floor-anchored copy).

        Returns world' (F,33,3) in a Y-UP frame (mediapipe's y-down/z-away axes
        flipped by the proper rotation (x,-y,-z); handedness preserved), with
        NaNs preserved where detection failed.
        """
        from scipy.signal import savgol_filter
        w = self.world.copy()
        f = max(3, int(round(win_s / np.median(np.diff(self.t)))) | 1)
        for j in range(33):
            for ax in range(3):
                x = w[:, j, ax]
                good = np.isfinite(x)
                if good.sum() < f + 2:
                    continue
                xi = np.interp(self.t, self.t[good], x[good])
                xs = savgol_filter(xi, f, 2)
                xs[~good] = np.nan
                w[:, j, ax] = xs
        w[..., 1] *= -1.0     # y-up
        w[..., 2] *= -1.0     # +z toward the camera
        return w

    def anchors(self, world: np.ndarray) -> dict:
        """Per-frame anchor points in the Y-up canonical frame.

        ``world`` must already be the Y-up track (smooth_world output);
        the pelvis is re-anchored above the floor via the lowest foot landmark.
        """
        w = world
        foot_z = np.nanmin(w[:, [LANK, RANK, LHEE, RHEE, LTOE, RTOE], 1], axis=1)
        pelvis = 0.5 * (w[:, LHIP] + w[:, RHIP])
        floor = foot_z
        pelvis_h = pelvis[:, 1] - floor
        lank = w[:, LANK]
        rank = w[:, RANK]
        return {"w": w, "pelvis": pelvis, "pelvis_h": pelvis_h,
                "lank": lank, "rank": rank, "floor": floor}

    # -- signals for take/transition detection -------------------------------
    def signals(self) -> dict:
        w = self.smooth_world()
        a = self.anchors(w)
        dt = np.median(np.diff(self.t))
        def speed_of(joints) -> np.ndarray:
            s = np.full(len(self.t), np.nan)
            for j in joints:
                p = w[:, j, :2].copy()
                v = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
                s[1:] = np.nanmax(np.vstack([s[1:], v]), axis=0)
            return s
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            speed = speed_of(ACTIVE_JOINTS)
            speed_lower = speed_of(LOWER_JOINTS)
            speed_upper = speed_of((LWRI, RWRI, NOSE))
            fullbody = np.ones(len(self.t), dtype=bool)
            for j in FULLBODY:
                fullbody &= (self.vis[:, j] >= 0.5) & self.det
            ph = a["pelvis_h"]
            pv = np.full_like(ph, np.nan)
            pv[1:] = np.diff(ph) / dt
            knees = w[:, [LKNE, RKNE], :]
            knee_h = np.nanmin(knees[:, :, 1] - a["floor"][:, None], axis=1)
        return {"w": w, "a": a, "dt": dt, "speed": speed,
                "speed_lower": speed_lower, "speed_upper": speed_upper,
                "fullbody": fullbody, "pelvis_h": ph, "pelvis_v": pv,
                "knee_h": knee_h}


def detect_takes(p: Pose, s: dict, chapter: int,
                 min_dur: float = 3.5, gap: float = 1.0,
                 speed_floor: float = 0.25, min_active: float = 2.0) -> list[dict]:
    """Clusters of active full-body frames inside one chapter = demo takes.

    Activity uses LOWER-body landmark speed (hips/knees/ankles): the coach
    gestures with his arms while explaining, so arm motion must not select
    takes.
    """
    dt = s["dt"]
    in_ch = p.chapter == chapter
    spd = s["speed_lower"]
    active = in_ch & s["fullbody"] & (spd >= speed_floor) & np.isfinite(spd)
    idx = np.where(active)[0]
    takes = []
    if len(idx) == 0:
        return takes
    groups = np.split(idx, np.where(np.diff(idx) * dt > gap)[0] + 1)
    for g in groups:
        dur = (g[-1] - g[0]) * dt
        if dur < min_dur:
            continue
        # require sustained activity: fraction of frames close to the group's
        # own activity level
        t0, t1 = p.t[g[0]] - 0.5, p.t[g[-1]] + 0.5
        seg = (p.t >= t0) & (p.t <= t1)
        takes.append({
            "chapter": chapter, "t0": float(t0), "t1": float(t1),
            "n_active": int(len(g)), "dur_s": float(t1 - t0),
            "mean_conf": float(np.nanmean(p.conf[seg & p.det])) if (seg & p.det).any() else 0.0,
            "fullbody_frac": float(s["fullbody"][seg].mean()),
            "mean_speed": float(np.nanmean(s["speed"][seg])),
            "mean_pelvis_h": float(np.nanmean(s["pelvis_h"][seg])),
            "i0": int(g[0]), "i1": int(g[-1]),
        })
    return takes


def merge_takes(takes: list[dict], max_gap: float = 1.5) -> list[dict]:
    """Merge adjacent takes of the same chapter separated by < max_gap seconds."""
    out = []
    for tk in takes:
        if out and tk["chapter"] == out[-1]["chapter"] and tk["t0"] - out[-1]["t1"] < max_gap:
            m = out[-1]
            m["t1"] = tk["t1"]; m["i1"] = tk["i1"]
            m["dur_s"] = m["t1"] - m["t0"]
            m["mean_conf"] = round(0.5 * (m["mean_conf"] + tk["mean_conf"]), 4)
            m["fullbody_frac"] = round(0.5 * (m["fullbody_frac"] + tk["fullbody_frac"]), 4)
            m["mean_speed"] = round(0.5 * (m["mean_speed"] + tk["mean_speed"]), 4)
            m["merged"] = m.get("merged", 1) + 1
        else:
            out.append(dict(tk))
    return out


def phase_labels(p: Pose, s: dict, i0: int, i1: int, stance_ph: float) -> list[str]:
    """Coarse per-frame phase label inside a take.

    VSTANCE  pelvis near the take's stance height and low vertical speed
    CROUCH   pelvis below stance band (level change / penetration descent)
    KNEE     a knee within 0.18 m of the floor
    MOVE     everything else (steps, circling, walking)
    """
    ph = s["pelvis_h"]
    kz = s["knee_h"]
    lab = []
    for i in range(i0, i1 + 1):
        if not np.isfinite(ph[i]):
            lab.append("NO_POSE")
        elif kz[i] < 0.18:
            lab.append("KNEE")
        elif ph[i] < stance_ph - 0.10:
            lab.append("CROUCH")
        elif ph[i] >= stance_ph - 0.10 and abs(s["pelvis_v"][i]) < 0.6:
            lab.append("VSTANCE")
        else:
            lab.append("MOVE")
    return lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(NPZ))
    ap.add_argument("--json", default="")
    ap.add_argument("--speed-floor", type=float, default=0.25)
    args = ap.parse_args()

    p = Pose(Path(args.npz))
    s = p.signals()
    report = {"n_frames": int(len(p.t)), "dt": float(s["dt"]),
              "detected_frac": float(p.det.mean()), "chapters": {}}
    for ch in range(1, 11):
        takes = merge_takes(detect_takes(p, s, ch, speed_floor=args.speed_floor))
        report["chapters"][ch] = {
            "n_takes": len(takes),
            "takes": sorted(takes, key=lambda k: -k["mean_conf"]),
        }
        for tk in takes:
            print(f"ch{ch:2d} take {tk['t0']:7.2f}-{tk['t1']:7.2f}s "
                  f"dur={tk['dur_s']:5.2f} conf={tk['mean_conf']:.2f} "
                  f"fb={tk['fullbody_frac']:.2f} spd={tk['mean_speed']:.2f} "
                  f"ph={tk['mean_pelvis_h']:.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
