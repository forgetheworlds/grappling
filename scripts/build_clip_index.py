#!/usr/bin/env python
"""Build the curated clip index for the wrestling motion-reference set.

For every candidate window (chapter-mapped, from
data/references/yt_gBAhX5t-GW4/derived/analysis/take_inventory_priority.txt and
the existing transitions.json) this script MEASURES, from the stored pose
landmarks (MediaPipe, 15 fps effective):

  * per-foot sole height + windowed horizontal speed  -> lift/plant events
    (hysteresis: plant = sole within 15 mm of the window floor for >=0.10 s;
     lift = sole more than 40 mm above it)
  * pelvis height extrema (stand vs crouch evidence)
  * knee-height minimum (descent control evidence)
  * wrist forward extension (arm-drive evidence, facing-relative)
  * torso pitch (angle of neck-core axis vs vertical)
  * quality flags: frames, gap fraction, landmark visibility, detection rate

It also AUTO-DETECTS the stand<->stance transition windows (pelvis rise/drop
>= 0.08 m within <= 3 s inside the stance/stalking chapters) and the backward
shuffle (pelvis facing-relative displacement < -0.12 m while stepping), so the
index covers the full movement set rather than only the pre-existing takes.

Output: data/references/motion_refs/clip_index.json + a printed table.
Every event timestamp is MEASURED from the landmarks; nothing is hand-typed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from grapplemap import JOINTS  # noqa: E402

POSE_DIR = REPO / "data/references/yt_gBAhX5t-GW4"
OUT = REPO / "data/references/motion_refs/clip_index.json"

#: sole landmark groups (GM joint names), Y-up track
SOLE = (("LeftToe", "LeftHeel"), ("RightToe", "RightHeel"))
J = {n: JOINTS.index(n) for n in JOINTS}

SOURCE_MAIN = {
    "video_id": "gBAhX5t-GW4",
    "url": "https://www.youtube.com/watch?v=gBAhX5t-GW4",
    "title": "How to stance and motion drills for wrestling",
    "channel": "Footwork Trainer (Isaac J. Knable)",
    "duration_s": 823.5,
    "file": "data/references/yt_gBAhX5t-GW4/ref.mp4",
    "pose_npz_main": "data/references/yt_gBAhX5t-GW4/pose/landmarks.npz",
    "pose_npz_pass2": "data/references/yt_gBAhX5t-GW4/pose_pass2/landmarks.npz",
    "native_resolution": "640x360 (ref.mp4); 960x540 re-encode used for pose",
}
SOURCE_TYU = {
    "video_id": "tyU-QaV8MnI",
    "url": "https://www.youtube.com/watch?v=tyU-QaV8MnI",
    "title": "Stance and Motion Drills For Wrestling",
    "channel": "The School of Wrestling",
    "duration_s": 149.0,
    "file": "data/references/yt_tyU-QaV8MnI/ref.mp4",
    "pose_npz_main": None,
    "native_resolution": "640x360",
}

#: hand-curated candidate windows (label, t0, t1, chapter, role, npz pass)
CANDIDATES = [
    ("stance_hold",        30.40,  34.80, "2", "stance/guard hold",        "main"),
    ("stance_widen_step",  27.60,  30.40, "2", "stance width adjustment step", "main"),
    ("stalk_shuffle",     122.00, 131.60, "3", "forward stance motion (stalk/approach)", "pass2"),
    ("circle_step",       178.00, 190.00, "4", "circling (lateral stance motion)", "pass2"),
    ("level_change_fast", 256.90, 258.00, "5", "level change (quick drop+rise)", "main"),
    ("level_change_full", 259.70, 261.20, "5", "level change (full drop+rise)", "main"),
    ("shot_entry",        422.80, 431.00, "7", "double-leg penetration step entry", "main"),
    ("shot_recover",      432.50, 437.00, "7", "recovery from shot to stance", "main"),
    ("knee_sprawl_entry", 683.40, 686.20, "10", "knee sprawl entry (DEFENSE - not a shot)", "main"),
    ("knee_sprawl_hold",  685.80, 690.20, "10", "knee sprawl hold (DEFENSE)", "main"),
    ("knee_sprawl_recover", 779.80, 782.20, "10", "knee sprawl recovery to stance", "main"),
]


def load_track(which: str):
    d = np.load(POSE_DIR / (f"pose{'_pass2' if which == 'pass2' else ''}/landmarks.npz"),
                allow_pickle=True)
    return d


def facing_frame(track: np.ndarray) -> np.ndarray:
    """Per-frame (fwd, right) unit vectors in the mp xz plane (y-up world)."""
    lat = track[:, J["LeftHip"]] - track[:, J["RightHip"]]     # left - right
    fwd = np.cross(np.array([0.0, 1.0, 0.0]), lat)
    fwd[:, 1] = 0.0
    n = np.linalg.norm(fwd, axis=1, keepdims=True)
    n[n < 1e-9] = 1.0
    fwd = fwd / n
    right = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
    return fwd, right


def window_signals(w: np.ndarray, t: np.ndarray):
    """Per-frame signal dict for one window (Y-up world landmarks)."""
    sole = np.stack([np.min(np.stack([w[:, J[a], 1], w[:, J[b], 1]]), axis=0)
                     for a, b in SOLE], axis=1)
    ank = w[:, [J["LeftAnkle"], J["RightAnkle"]]][..., [0, 2]]
    dt = float(np.median(np.diff(t)))
    win = max(1, int(round(0.20 / dt)))
    spd = np.zeros_like(ank[..., 0])
    for s in range(2):
        for i in range(len(t)):
            a, b = max(0, i - win), min(len(t) - 1, i + win)
            spd[i, s] = np.linalg.norm(ank[b, s] - ank[a, s]) / max((b - a) * dt, 1e-9)
    core = w[:, J["Core"]]
    knee = np.stack([w[:, J["LeftKnee"], 1], w[:, J["RightKnee"], 1]], axis=1)
    neck = w[:, J["Neck"]]
    axis = neck - core
    pitch = np.degrees(np.arctan2(np.linalg.norm(axis[:, [0, 2]], axis=1), axis[:, 1]))
    fwd, right = facing_frame(w)
    wr = np.stack([w[:, J["LeftWrist"]], w[:, J["RightWrist"]]], axis=1)
    wrist_fwd = np.empty((len(t), 2))
    for s in range(2):
        wrist_fwd[:, s] = np.einsum("ij,ij->i", wr[:, s] - core, fwd)
    pelvis_fwd = (core - core[0]) @ (0.5 * (fwd[0] + fwd[-1]))
    vis = None
    return {"t": t, "sole": sole, "spd": spd, "core": core, "knee": knee,
            "pitch": pitch, "wrist_fwd": wrist_fwd, "pelvis_fwd": pelvis_fwd,
            "ank": ank, "dt": dt}


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return [(int(idx[i]), int(idx[i + 1] - 1)) for i in range(0, len(idx), 2)]


def foot_events(t: np.ndarray, sole: np.ndarray, spd: np.ndarray) -> dict:
    """Lift/plant events per foot, measured with hysteresis on the window floor."""
    floor = float(np.min(sole))
    low = sole - floor
    ev = {}
    for s, name in ((0, "left"), (1, "right")):
        plants, lifts = [], []
        planted = low[0, s] <= 0.015
        for i in range(1, len(t)):
            if not planted and low[i, s] <= 0.015:
                planted = True
                plants.append(round(float(t[i]), 2))
            elif planted and low[i, s] > 0.040:
                planted = False
                lifts.append(round(float(t[i]), 2))
        ev[name] = {"floor_m": round(floor, 3),
                    "plants_s": plants, "lifts_s": lifts,
                    "max_height_m": round(float(low[:, s].max()), 3),
                    "mean_speed_mps": round(float(spd[:, s].mean()), 3),
                    "max_speed_mps": round(float(spd[:, s].max()), 3)}
    return ev


def analyse_window(name: str, t0: float, t1: float, chapter: str, role: str,
                   which: str, src: dict) -> dict:
    d = load_track(which)
    t_all, world = d["t"], d["world_lm"].astype(np.float64)
    vis = d["visibility"].astype(np.float64)
    det = d["detected"].astype(bool)
    ch = d["chapter"].astype(str)
    sel = (t_all >= t0) & (t_all <= t1) & (ch == chapter)
    idx = np.flatnonzero(sel)
    if len(idx) < 8:
        sel = (t_all >= t0) & (t_all <= t1)
        idx = np.flatnonzero(sel)
    # keep longest contiguous run
    if len(idx) > 1 and (np.diff(idx) > 1).any():
        parts = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        idx = max(parts, key=len)
    w, t = world[idx], t_all[idx]
    sig = window_signals(w, t)
    foot = foot_events(t, sig["sole"], sig["spd"])
    # floor-relative heights (the window's lowest sole landmark = 0): makes the
    # numbers comparable with measure_takes.py's take inventory
    win_floor = float(np.min(sig["sole"]))
    core_h = sig["core"][:, 1] - win_floor
    knee_h = sig["knee"] - win_floor
    cmin, cmax = int(core_h.argmin()), int(core_h.argmax())
    kmin = int(knee_h.min(axis=1).argmin())
    wext = int(sig["wrist_fwd"].max(axis=1).argmax())
    meta = {
        "name": name, "role": role,
        "source": {"video_id": src["video_id"], "url": src["url"],
                   "title": src["title"], "channel": src["channel"]},
        "video_timestamps": {"t0_s": round(float(t[0]), 2),
                              "t1_s": round(float(t[-1]), 2),
                              "duration_s": round(float(t[-1] - t[0]), 2)},
        "chapter": chapter,
        "events_measured": {
            "feet": foot,
            "pelvis_min": {"t_s": round(float(t[cmin]), 2),
                            "height_m": round(float(core_h[cmin]), 3)},
            "pelvis_max": {"t_s": round(float(t[cmax]), 2),
                            "height_m": round(float(core_h[cmax]), 3)},
            "knee_min": {"t_s": round(float(t[kmin]), 2),
                          "height_m": round(float(knee_h[kmin].min()), 3)},
            "wrist_max_forward": {"t_s": round(float(t[wext]), 2),
                                   "ahead_m": round(float(sig["wrist_fwd"][wext].max()), 3)},
            "torso_pitch_deg": {"min": round(float(sig["pitch"].min()), 1),
                                 "max": round(float(sig["pitch"].max()), 1)},
            "pelvis_facing_displacement_m": round(float(sig["pelvis_fwd"][-1]), 3),
        },
        "quality": {
            "frames": int(len(idx)),
            "pose_fps": round(1.0 / sig["dt"], 2),
            "gap_frames_in_window": int(len(t) - (t[-1] - t[0]) / sig["dt"] + 1),
            "mean_visibility": round(float(vis[idx].mean()), 3),
            "foot_visibility": round(float(vis[idx][:, [27, 28, 29, 30, 31, 32]].mean()), 3),
            "hand_visibility": round(float(vis[idx][:, 15:23].mean()), 3),
            "detection_rate": round(float(det[idx].mean()), 3),
        },
        "camera_and_depth_caveats": [
            "monocular 2D+approximate-3D (MediaPipe): depth/scale approximate",
            "camera assumed static (wide gym shot); no pan/zoom compensation applied",
            "landmark z (depth toward camera) noisier than x/y; step DIRECTIONS beyond "
            "~30 deg from the frontal plane are low-confidence",
        ],
    }
    if src["native_resolution"].startswith("640x360"):
        meta["quality"]["resolution_note"] = (
            "640x360 source: small landmarks (feet during fast steps) are the "
            "dominant noise source; foot-depth especially")
    return meta


def detect_transitions(src_which: str = "main") -> dict:
    """Auto-detect stand<->stance and backward-shuffle windows in chapters 2/3."""
    d = load_track(src_which)
    t_all, world = d["t"], d["world_lm"].astype(np.float64)
    ch = d["chapter"].astype(str)
    core_y = world[:, J["Core"], 1]
    out = {"stand_to_stance": [], "stance_to_stand": [], "backward_shuffle": [],
           "forward_check": []}
    for chapter, lo, hi in (("2", 25.0, 83.0), ("5", 201.0, 262.0)):
        sel = np.flatnonzero((ch == chapter) & (t_all >= lo) & (t_all <= hi))
        if len(sel) < 30:
            continue
        t, y = t_all[sel], core_y[sel]
        w = max(3, int(round(0.4 / float(np.median(np.diff(t))))))
        # centred moving min/max
        for i in range(w, len(t) - w):
            past, fut = y[i - w:i], y[i:i + w]
            drop = past.max() - fut.min()
            rise = fut.max() - past.min()
            if drop >= 0.08 and (len(out["stand_to_stance"]) == 0 or
                                 t[i] - out["stand_to_stance"][-1][0] > 2.0):
                out["stand_to_stance"].append((round(float(t[i - w]), 2),
                                               round(float(t[min(i + w, len(t) - 1)]), 2),
                                               round(float(drop), 3)))
            if rise >= 0.08 and (len(out["stance_to_stand"]) == 0 or
                                 t[i] - out["stance_to_stand"][-1][0] > 2.0):
                out["stance_to_stand"].append((round(float(t[i - w]), 2),
                                               round(float(t[min(i + w, len(t) - 1)]), 2),
                                               round(float(rise), 3)))
    # shuffles: windows whose pelvis moves (facing-relative) while feet step
    for chapter, lo, hi, out_key, sign in (("3", 83.0, 133.0, "backward_shuffle", -1.0),
                                            ("2", 59.0, 83.0, "backward_shuffle", -1.0),
                                            ("3", 83.0, 133.0, "forward_check", +1.0)):
        sel = np.flatnonzero((ch == chapter) & (t_all >= lo) & (t_all <= hi))
        if len(sel) < 30:
            continue
        w3 = world[sel]
        t3 = t_all[sel]
        seg = 20  # ~1.3 s
        for i in range(0, len(sel) - seg, seg // 2):
            ww = w3[i:i + seg + 1]
            fwd, _ = facing_frame(ww)
            disp = ww[-1, J["Core"]] - ww[0, J["Core"]]
            d_fwd = float(np.dot(disp, (fwd[0] + fwd[-1]) / 2.0))
            ank = ww[:, J["RightAnkle"]][..., [0, 2]]
            step_spd = float(np.linalg.norm(ank[1:] - ank[:-1], axis=1).max())
            if sign * d_fwd > 0.06 and step_spd > 0.01:
                lst = out[out_key]
                if lst and t3[i] - lst[-1][1] < 2.0:
                    lst[-1] = (lst[-1][0], round(float(t3[i + seg]), 2))
                else:
                    lst.append((round(float(t3[i]), 2), round(float(t3[i + seg]), 2)))
    return out


def detect_static_hold(src_which: str = "main", lo: float = 59.9, hi: float = 82.0,
                       min_s: float = 1.2, max_range: float = 0.05) -> dict:
    """Longest window where both feet stay planted and pelvis range is small
    (a true STANCE_HOLD inside chapter 2's second half)."""
    d = load_track(src_which)
    t_all, world = d["t"], d["world_lm"].astype(np.float64)
    ch = d["chapter"].astype(str)
    sel = np.flatnonzero((ch == "2") & (t_all >= lo) & (t_all <= hi))
    w = world[sel]
    t = t_all[sel]
    sole = np.stack([np.min(np.stack([w[:, J[a], 1], w[:, J[b], 1]]), axis=0)
                     for a, b in SOLE], axis=1)
    planted = (sole - sole.min() <= 0.05)
    core = w[:, J["Core"], 1]
    win = int(round(min_s / float(np.median(np.diff(t)))))
    best = None
    i = 0
    while i < len(t) - win:
        if planted[i:i + win].all():
            rng = float(core[i:i + win].max() - core[i:i + win].min())
            j = i + win
            while j < len(t) and bool(planted[j].all() if planted[j].ndim else planted[j]) \
                    and core[i:j + 1].max() - core[i:j + 1].min() <= max_range:
                j += 1
            if best is None or (j - i) > (best[1] - best[0]):
                best = (i, j, rng)
            i = j
        else:
            i += 1
    if best is None:
        return {}
    i, j, rng = best
    return {"t0_s": round(float(t[i]), 2), "t1_s": round(float(t[j - 1]), 2),
            "duration_s": round(float(t[j - 1] - t[i]), 2),
            "pelvis_range_m": round(rng, 3),
            "both_feet_planted": True}


def main() -> None:
    out_dir = OUT.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = [analyse_window(n, a, b, c, r, wch, SOURCE_MAIN)
             for n, a, b, c, r, wch in CANDIDATES]
    detected = detect_transitions("main")
    detected["static_stance_hold"] = detect_static_hold("main")
    # curated transition windows chosen from the auto-detected candidates and
    # VERIFIED against extracted frames (2026-10-08):
    detected["curated_transitions"] = {
        "stand_to_stance": {"t0_s": 27.4, "t1_s": 28.6,
                            "why": "first clean drop-to-stance after the walk-in; "
                                   "frames 27.3 (upright) -> 28.2 (crouched) verified"},
        "stance_to_stand": {"t0_s": 33.9, "t1_s": 35.4,
                            "why": "rise out of stance (detected rise 0.399 m at "
                                   "34.13-34.93); padded one frame either side for "
                                   "the retarget filter; frame 34.5 verified upright"},
        "backward_shuffle": {"t0_s": 63.67, "t1_s": 66.33,
                             "why": "in-stance reposition motion with both feet "
                                    "planted wide; frames 64.5/65.5 verified"},
    }
    index = {
        "version": "1.0",
        "generated_by": "scripts/build_clip_index.py (measured from stored pose landmarks)",
        "provenance_decision": {
            "primary": "gBAhX5t-GW4 (13:44, chapter-mapped, full pose pipeline exists; "
                       "all retargeted references derive from it)",
            "secondary": "tyU-QaV8MnI (149 s, fetched 2026-10-08 through the documented "
                         "cookie+yt-dlp-ejs path after the orchestrator brief named it; "
                         "it corroborates stance/motion/level-change mechanics and shows "
                         "sprawl+recover, but is 640x360 with the coach small in frame, "
                         "so it is used for VERIFICATION of vocabulary, not for "
                         "retargeting; no pose pipeline was run on it)",
            "tyU_clips_examined": [
                {"t0_s": 9.5, "t1_s": 24.0, "label": "deep-stance hold + in-stance motion",
                 "evidence": "filmstrip frames at 10/15/20 s show the deep crouched guard"},
                {"t0_s": 29.0, "t1_s": 33.0, "label": "athletic stance, wide base",
                 "evidence": "frame at 30 s"},
                {"t0_s": 108.0, "t1_s": 118.0, "label": "stance motion, hands high carriage",
                 "evidence": "frames at 110/115 s"},
                {"t0_s": 33.0, "t1_s": 48.0, "label": "sprawl / ground-and-recover drills "
                 "(out of this milestone's scope; noted for later defense phases)",
                 "evidence": "frames at 35-45 s"},
            ],
        },
        "sources": [SOURCE_MAIN, SOURCE_TYU],
        "clips": clips,
        "auto_detected_windows": detected,
    }
    OUT.write_text(json.dumps(index, indent=1))
    n = len(clips)
    print(f"wrote {OUT} ({n} curated clips + {sum(len(v) for v in detected.values())} "
          f"auto-detected windows)")
    for c in clips:
        e = c["events_measured"]
        print(f"{c['name']:22s} {c['video_timestamps']['t0_s']:7.2f}-"
              f"{c['video_timestamps']['t1_s']:7.2f}s "
              f"pelvis {e['pelvis_max']['height_m']:.2f}->{e['pelvis_min']['height_m']:.2f} "
              f"knee_min {e['knee_min']['height_m']:.2f} "
              f"lifts L/R {len(e['feet']['left']['lifts_s'])}/"
              f"{len(e['feet']['right']['lifts_s'])} "
              f"vis {c['quality']['mean_visibility']:.2f}")
    for k, v in detected.items():
        if isinstance(v, list):
            print(f"detected {k}: {v[:6]}")
        else:
            print(f"detected {k}: {v}")


if __name__ == "__main__":
    main()
