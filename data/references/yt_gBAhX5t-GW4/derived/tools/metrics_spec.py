"""Compact numeric spec from the video transitions (byproduct deliverable).

Computes, for each named transition window, the numbers a task definition can
consume, each with an explicit status:
  observed  - read directly from the landmark track (image-plane geometry)
  inferred  - derived with a stated assumption (metric scale from body height,
              camera-static, monocular depth)
  unknown   - not determinable from this footage

Output: data/references/yt_gBAhX5t-GW4/derived/stance_spec.json

Run: python .../metrics_spec.py --out .../derived/stance_spec.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from analyze_pose import (LANK, LHEE, LHIP, LKNE, LSHO, LTOE, LWRI,  # noqa: E402
                          NOSE, Pose, RANK, RHEE, RHIP, RKNE, RSHO, RTOE,
                          RWRI)
from measure_takes import (ALPHA, body_frame, image_metric, proj,  # noqa: E402
                           torso_pitch_deg)
from step_stats import reposition_events  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
POSE = REPO / "data/references/yt_gBAhX5t-GW4/pose/landmarks.npz"
POSE2 = REPO / "data/references/yt_gBAhX5t-GW4/pose_pass2/landmarks.npz"
OUT = REPO / "data/references/yt_gBAhX5t-GW4/derived/stance_spec.json"
TRANSITIONS = REPO / "data/references/yt_gBAhX5t-GW4/derived/transitions.json"

_cache: dict[str, Pose] = {}
_stand_cache: dict[str, float] = {}


def pose_of(path: str) -> Pose:
    if path not in _cache:
        _cache[path] = Pose(Path(path))
    return _cache[path]


def standing_height(path: str) -> float:
    """Max nose-above-floor height over the whole track (the coach's standing
    height in mediapipe metric units) — the scale reference for all ratios."""
    if path not in _stand_cache:
        p = pose_of(path)
        s = p.signals()
        w = s["w"]
        bh = w[:, NOSE, 1] - s["a"]["floor"]
        _stand_cache[path] = float(np.nanmax(bh))
    return _stand_cache[path]


def window(p: Pose, s: dict, t0: float, t1: float):
    sel = (p.t >= t0) & (p.t <= t1)
    idx = np.where(sel)[0]
    assert len(idx) > 3, f"empty window {t0}-{t1}"
    w = s["w"][idx]
    bf = body_frame(w)
    im, scale = image_metric(p, idx)
    return idx, w, bf, im, scale


def geom(p: Pose, s: dict, t0: float, t1: float) -> dict:
    """Geometry summary of a window (world-landmark relative + image metric)."""
    idx, w, bf, im, scale = window(p, s, t0, t1)
    dt = float(np.median(np.diff(p.t)))
    with np.errstate(invalid="ignore"):
        ph = bf["core"][:, 1] - bf["floor"]
        knee_h = np.minimum(w[:, LKNE, 1], w[:, RKNE, 1]) - bf["floor"]
        la, ra = proj(bf, w[:, LANK]), proj(bf, w[:, RANK])
        width = np.abs(la[:, 1] - ra[:, 1])
        depth = np.abs(la[:, 0] - ra[:, 0])
        knee_l = 180.0 - np.degrees(np.arccos(np.clip(np.einsum(
            "ij,ij->i",
            (w[:, LHIP] - w[:, LKNE]) / np.maximum(np.linalg.norm(w[:, LHIP] - w[:, LKNE], axis=1, keepdims=True), 1e-9),
            (w[:, LANK] - w[:, LKNE]) / np.maximum(np.linalg.norm(w[:, LANK] - w[:, LKNE], axis=1, keepdims=True), 1e-9)), -1, 1)))
        knee_r = 180.0 - np.degrees(np.arccos(np.clip(np.einsum(
            "ij,ij->i",
            (w[:, RHIP] - w[:, RKNE]) / np.maximum(np.linalg.norm(w[:, RHIP] - w[:, RKNE], axis=1, keepdims=True), 1e-9),
            (w[:, RANK] - w[:, RKNE]) / np.maximum(np.linalg.norm(w[:, RANK] - w[:, RKNE], axis=1, keepdims=True), 1e-9)), -1, 1)))
        knee_l = np.maximum(knee_l, knee_r)  # flexion: 0 deg = straight
        pitch = torso_pitch_deg(w)
        head_h = w[:, NOSE, 1] - bf["core"][:, 1]
        lw = proj(bf, w[:, LWRI])
        rw = proj(bf, w[:, RWRI])
        body_m = np.linalg.norm(w[:, NOSE] - 0.5 * (w[:, LANK] + w[:, RANK]), axis=1)
    stand = standing_height(p.path) if getattr(p, "path", None) else float(np.nanmax(body_m))
    return {
        "t0": t0, "t1": t1, "dur_s": round(len(idx) * dt, 3),
        "n": len(idx),
        "body_height_m": round(stand, 3),
        "pelvis_h_med": round(float(np.nanmedian(ph)), 3),
        "pelvis_h_min": round(float(np.nanmin(ph)), 3),
        "pelvis_h_max": round(float(np.nanmax(ph)), 3),
        "pelvis_h_over_body": round(float(np.nanmedian(ph) / stand), 3),
        "pelvis_drop_m": round(float(np.nanmax(ph) - np.nanmin(ph)), 3),
        "knee_h_min": round(float(np.nanmin(knee_h)), 3),
        "knee_flex_deg_med": round(float(np.nanmedian(knee_l)), 1),
        "torso_pitch_deg_med": round(float(np.nanmedian(pitch)), 1),
        "head_above_pelvis_m": round(float(np.nanmedian(head_h)), 3),
        "stance_width_m": round(float(np.nanmedian(width)), 3),
        "stance_width_over_body": round(float(np.nanmedian(width) / stand), 3),
        "stance_width_g1_m": round(float(np.nanmedian(width) * ALPHA), 3),
        "stance_depth_m": round(float(np.nanmedian(depth)), 3),
        "stance_depth_g1_m": round(float(np.nanmedian(depth) * ALPHA), 3),
        "hands_lwri_rel_pelvis_fwd_lat_up": [round(float(x), 3) for x in np.nanmedian(lw, axis=0)],
        "hands_rwri_rel_pelvis_fwd_lat_up": [round(float(x), 3) for x in np.nanmedian(rw, axis=0)],
        "lead_foot_fwd_offset_m": round(float(np.nanmedian(np.maximum(la[:, 0], ra[:, 0]))), 3),
        "rear_foot_fwd_offset_m": round(float(np.nanmedian(np.minimum(la[:, 0], ra[:, 0]))), 3),
        "scale_m_per_img_unit": round(float(np.nanmedian(scale)), 4),
        "_im": im, "_idx": idx, "_p": p, "_bf": bf, "_w": w,
    }


def strip_internal(g: dict) -> dict:
    return {k: v for k, v in g.items() if not k.startswith("_")}


def lead_trail(p: Pose, s: dict, t0: float, t1: float) -> dict:
    """Which foot is the lead and how far it travels (image metric)."""
    g = geom(p, s, t0, t1)
    im, idx, w = g["_im"], g["_idx"], g["_w"]
    # body-forward axis in the image plane (median over the window)
    fwd = g["_bf"]["fwd"]
    f2 = np.array([np.nanmedian(fwd[:, 0]), np.nanmedian(fwd[:, 2])])
    f2 = f2 / max(np.linalg.norm(f2), 1e-9)
    # forward advance of each ankle over the window
    adv = {}
    for nm, j in (("left", LANK), ("right", RANK)):
        uv = im[:, j]
        adv[nm] = float(np.dot(uv[-1] - uv[0], f2))
    lead = max(adv, key=adv.get)
    trail = "right" if lead == "left" else "left"
    jl = LANK if lead == "left" else RANK
    jt = RANK if lead == "left" else LANK
    # lead foot travel: max forward displacement from its first position
    uvl = im[:, jl]
    fwd_proj = (uvl - uvl[0]) @ f2
    return {
        "lead_foot": lead, "trail_foot": trail,
        "lead_fwd_advance_m": round(float(fwd_proj[-1]), 3),
        "lead_travel_max_m": round(float(np.nanmax(fwd_proj) - np.nanmin(fwd_proj)), 3),
        "trail_fwd_advance_m": round(float(adv[trail]), 3),
        "pelvis_fwd_advance_m": round(float(np.dot(
            (0.5 * (im[-1, LHIP] + im[-1, RHIP])) - (0.5 * (im[0, LHIP] + im[0, RHIP])), f2)), 3),
        "body_height_m": g["body_height_m"],
    }


def time_of_min(arr: np.ndarray, idx: np.ndarray, p: Pose) -> float:
    j = int(np.nanargmin(arr))
    return float(p.t[idx[j]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    spec: dict = {
        "source": "data/references/yt_gBAhX5t-GW4/ref720h264.mp4 (operator reference video)",
        "estimator": "mediapipe PoseLandmarker full, 15 fps effective",
        "units": "metres (SI); body-relative ratios dimensionless",
        "coach_height_assumption_m": 1.78,
        "g1_height_m": 1.32,
        "g1_scale_factor": round(ALPHA, 4),
        "status_legend": {
            "observed": "read directly from the landmark track (image-plane geometry)",
            "inferred": "derived with a stated assumption (monocular metric scale, static camera, 3D depth approximate)",
            "unknown": "not determinable from this footage",
        },
        "params": {},
        "take_selection": {},
        "tie_breaker": ("operator two-axis rule when the video geometry is not statically "
                        "holdable for the G1: (a) sagittal - rear leg further back; "
                        "(b) frontal - widen the stance. Both preserve the wrestling read."),
    }

    p = pose_of(str(POSE))
    s = p.signals()
    SH = standing_height(str(POSE))       # mediapipe metric standing height
    G1H = 1.32

    def g1(ratio: float) -> float:
        return round(ratio * G1H, 3)

    spec["metric_scale"] = {
        "mediapipe_standing_height_m": round(SH, 3),
        "note": ("mediapipe's monocular metric scale reads the coach at ~"
                 f"{SH:.2f} m. If his true height is 1.78 m, multiply every raw metre "
                 f"value by ~{1.78 / SH:.2f}. Ratios and G1-equivalents are unaffected."),
        "g1_equivalents": "computed as ratio x G1 height (1.32 m), independent of the coach's true height",
    }

    def add(key: str, value, unit: str, status: str, evidence: str, conf: str):
        spec["params"][key] = {"value": value, "unit": unit, "status": status,
                               "evidence": evidence, "confidence": conf}

    # ---- STANCE ------------------------------------------------------------
    g_hold = geom(p, s, 30.4, 34.8)
    add("stance_width_m", g_hold["stance_width_m"], "m", "inferred",
        "frames 30.4-34.8s (ch2 stance hold): median ankle lateral separation in the hip-line frame",
        "high for the ratio; metric value depends on monocular scale (assumed coach height 1.78 m)")
    add("stance_width_over_body_height", g_hold["stance_width_over_body"], "-", "observed",
        "same window; separation / standing body height (scale-free)",
        "high")
    add("stance_width_g1_equivalent_m", g1(g_hold["stance_width_over_body"]), "m", "inferred",
        "same window: ratio x G1 height 1.32 m (relationship-preserving; independent of the coach's true height)",
        "medium: uniform scaling, G1 leg proportions differ from a human's")
    add("stance_depth_m", g_hold["stance_depth_m"], "m", "inferred",
        "same window: median front-to-back ankle separation (feet nearly level in the taught square stance)",
        "medium: monocular depth is the least reliable axis")
    add("stance_depth_g1_equivalent_m", g1(g_hold["stance_depth_m"] / SH), "m", "inferred",
        "same window: ratio x G1 height", "medium-low (depth axis)")
    add("stance_knee_flexion_deg", g_hold["knee_flex_deg_med"], "deg", "observed",
        "same window: hip-knee-ankle angle (world landmarks)", "high")
    add("stance_torso_pitch_deg", g_hold["torso_pitch_deg_med"], "deg", "observed",
        "same window: pelvis-to-head axis vs vertical", "high")
    add("stance_pelvis_height_m", g_hold["pelvis_h_med"], "m", "inferred",
        "same window: pelvis above floor (world landmarks)", "medium (monocular metric scale)")
    add("stance_pelvis_height_over_body", g_hold["pelvis_h_over_body"], "-", "observed",
        "same window", "high")
    add("stance_head_above_pelvis_m", g_hold["head_above_pelvis_m"], "m", "observed",
        "same window: nose above pelvis height", "high")
    add("stance_hand_carriage_rel_pelvis", {
        "left": g_hold["hands_lwri_rel_pelvis_fwd_lat_up"],
        "right": g_hold["hands_rwri_rel_pelvis_fwd_lat_up"],
        "axes": "[forward, left, up] metres relative to the pelvis",
    }, "m", "observed", "same window: wrist medians in the body frame", "medium (hand landmarks noisier)")
    add("stance_widen_step_window", {
        "t": [27.6, 30.4], "width_from_m": 0.26, "width_to_m": 0.67,
        "step_fwd_lat_m": [round(0.20, 2), round(-0.38, 2)],
    }, "-", "observed",
        "ch2 27.6-30.4s: stance widened 0.26 -> 0.67 m; the widening step moves the rear foot back-and-out",
        "medium (single take; step direction from image metric)")
    spec["take_selection"]["stance"] = {
        "chosen": "25.6-35.7s take, hold sub-window 30.4-34.8s",
        "why": "highest mean landmark confidence (0.967), feet planted (0 foot-repositioning events), pelvis travel 0.07 m net",
        "alternates": ["71.2-82.0s take (conf 0.949)", "59.9-66.8s take (conf 0.957)"],
    }

    # take-to-take variability of stance width
    g_a = geom(p, s, 30.4, 34.8)
    g_b = geom(p, s, 72.5, 81.0)
    spec["take_selection"]["stance_variability"] = {
        "stance_width_m_take1": g_a["stance_width_m"],
        "stance_width_m_take2": g_b["stance_width_m"],
        "delta_m": round(abs(g_a["stance_width_m"] - g_b["stance_width_m"]), 3),
    }

    # ---- LEVEL CHANGE ------------------------------------------------------
    g_lc = geom(p, s, 259.7, 261.2)
    add("level_change_pelvis_drop_m", g_lc["pelvis_drop_m"], "m", "inferred",
        "ch5 259.7-261.2s: pelvis max-min (world landmarks)", "medium (monocular metric scale)")
    add("level_change_pelvis_drop_g1_m", g1(g_lc["pelvis_drop_m"] / SH), "m", "inferred",
        "same: ratio x G1 height 1.32 m", "medium")
    add("level_change_pelvis_drop_over_body", round(g_lc["pelvis_drop_m"] / g_lc["body_height_m"], 3),
        "-", "observed", "same window (scale-free)", "high")
    add("level_change_duration_s", 1.5, "s", "observed",
        "ch5 259.7-261.2s full down-and-up cycle; the drop itself occupies ~0.4-0.8 s",
        "medium (15 fps sampling)")
    g_fast = geom(p, s, 256.9, 258.0)
    add("level_change_fast_drop_m", g_fast["pelvis_drop_m"], "m", "inferred",
        "ch5 256.9-258.0s: a quicker drop-and-rise repetition", "medium")
    add("level_change_min_pelvis_h_m", g_lc["pelvis_h_min"], "m", "inferred",
        "same window: lowest pelvis height reached", "medium")
    spec["take_selection"]["level_change"] = {
        "chosen": "256.2-261.4s take (deepest drop: pelvis 0.85 -> 0.48 m); windows 259.7-261.2 (full) and 256.9-258.0 (fast)",
        "why": "largest sustained pelvis drop of the chapter with clean tracking (conf 0.93)",
        "alternates": ["201.5-223.4s take (crouch 2.3 s, drop to 0.69 m - shallower)"],
    }

    # ---- SHOT --------------------------------------------------------------
    g_shot = geom(p, s, 422.8, 431.0)
    g_knee = geom(p, s, 430.5, 433.0)     # knee-on-mat hold: final entry geometry
    add("shot_lead_foot_offset_m", g_knee["lead_foot_fwd_offset_m"], "m", "inferred",
        "ch7 430.5-433.0s (knee-on-mat hold): forward offset of the lead ankle from the pelvis "
        "(world-landmark body frame; depth axis approximate)",
        "medium-low: monocular depth; direction (in front) is reliable, magnitude is not")
    add("shot_lead_foot_offset_over_body",
        round(g_knee["lead_foot_fwd_offset_m"] / SH, 3), "-", "observed",
        "same window", "medium")
    add("shot_lead_foot_offset_g1_m", g1(g_knee["lead_foot_fwd_offset_m"] / SH), "m", "inferred",
        "same: ratio x G1 height", "medium-low (depth axis)")
    add("shot_rear_foot_offset_m", g_knee["rear_foot_fwd_offset_m"], "m", "inferred",
        "same window: rear ankle offset (negative = behind the pelvis)", "medium-low")
    add("shot_entry_pelvis_h_start_end_m",
        [g_shot["pelvis_h_max"], g_knee["pelvis_h_med"]], "m", "inferred",
        "ch7: pelvis height at the start of the entry vs during the knee-on-mat hold", "medium")
    add("shot_knee_min_height_m", g_shot["knee_h_min"], "m", "observed",
        "ch7 422.8-431.0s: lowest knee landmark height above the floor (0.00 m = on the mat)",
        "high for 'knee reaches the mat'; the exact contact instant is +-0.07 s")
    add("shot_knee_down_hold_s", 2.5, "s", "observed",
        "ch7 430.5-433.0s: knee stays within 5 cm of the mat", "medium")
    add("shot_entry_duration_s", round(431.0 - 422.8, 1), "s", "observed",
        "ch7 422.8-431.0s: crouched stance -> step -> penetration -> knee down",
        "medium (window chosen by content review of the trace)")
    lt = lead_trail(p, s, 422.8, 431.0)
    add("shot_lead_foot_travel_imgplane_m", lt["lead_travel_max_m"], "m", "inferred",
        "ch7 422.8-431.0s: lead ankle displacement in the image plane (UNDERESTIMATES the step "
        "when it travels toward the camera; the world-landmark offsets above are the usable entry depth)",
        "low for magnitude; the lead/trail assignment is reliable")
    g_rec = geom(p, s, 432.5, 437.0)
    add("shot_recovery_duration_s", round(437.0 - 432.5, 1), "s", "observed",
        "ch7 432.5-437.0s: knee-off to standing crouched stance", "medium")
    add("shot_recovery_pelvis_h_end_m", g_rec["pelvis_h_max"], "m", "inferred",
        "same window: pelvis height at the end of the rise", "medium")
    spec["take_selection"]["shots"] = {
        "chosen": "430.6-442.8s take; entry window 422.8-431.0s, recovery 432.5-437.0s",
        "why": "contains the knee-to-mat contact (knee min 0.0 m, 4.1 s near the mat) and a complete rise",
        "alternates": ["455.1-488.1s take (long crouch, no knee contact)", "399.5-416.9s take (stance/footwork only)"],
        "note": "the entry starts inside the preceding quiet period (423-430 s) that the activity detector split off; selected by content review",
    }

    # ---- KNEE SPRAWL -------------------------------------------------------
    g_sp = geom(p, s, 683.4, 686.2)
    add("knee_sprawl_drop_s", 0.5, "s", "observed",
        "ch10 683.4-686.2s: standing to knee-on-mat (pelvis 0.68 -> 0.24 m in ~0.4 s)",
        "medium (15 fps)")
    add("knee_sprawl_knee_height_m", g_sp["knee_h_min"], "m", "observed",
        "same window: knee landmark reaches the mat plane (negative = at/below the foot plane; "
        "landmark contact offset ~0.1 m, so treat 0.0 as 'on the mat')", "medium")
    add("knee_sprawl_pelvis_h_m", g_sp["pelvis_h_min"], "m", "inferred",
        "same window: lowest pelvis height at the drop (~0.30 m while sprawled)",
        "medium")
    add("knee_sprawl_drop_g1_m", g1((g_sp["pelvis_h_max"] - g_sp["pelvis_h_min"]) / SH), "m",
        "inferred", "same: ratio x G1 height", "medium")
    g_hold = geom(p, s, 685.8, 690.2)
    add("knee_sprawl_hold_s", 4.4, "s", "observed",
        "ch10 685.8-690.2s: knee stays near the mat while the torso is up",
        "medium")
    g_rec2 = geom(p, s, 779.8, 782.2)
    add("knee_sprawl_recover_s", round(782.2 - 779.8, 1), "s", "observed",
        "ch10 779.8-782.2s: rise from the knee to a crouched stance (pelvis 0.04 -> 0.63 m)",
        "medium")
    g_sp2 = geom(p, s, 769.5, 772.5)
    add("knee_sprawl_second_demo", {
        "pelvis_h_before_m": g_sp2["pelvis_h_max"],
        "pelvis_h_min_m": g_sp2["pelvis_h_min"],
        "drop_m": round(g_sp2["pelvis_h_max"] - g_sp2["pelvis_h_min"], 3),
    }, "m", "inferred",
        "ch10 769.5-772.5s: second repetition (stand -> knee down)", "medium")
    spec["take_selection"]["knee_sprawl"] = {
        "chosen": "683.4-686.2s entry + 685.8-690.2s hold (in the 661.5-685.7s take) and 779.8-782.2s recovery",
        "why": "cleanest knee-to-mat contact (knee min 0.00 m) and a complete rise; the 691-703s stretch is excluded (landmark glitches: speeds 8-12 m/s)",
        "alternates": ["732.5-749.8s take", "766.1-771.0s take (second repetition, entry window 769.5-772.5s)"],
    }

    # ---- STALKING / CIRCLING ----------------------------------------------
    p2 = pose_of(str(POSE2))
    s2 = p2.signals()

    def cadence(path: str, t0: float, t1: float) -> dict:
        pp = pose_of(path)
        ss = pp.signals()
        idx, w, bf, im, scale = window(pp, ss, t0, t1)
        dt = float(np.median(np.diff(pp.t)))
        dur = len(idx) * dt
        steps = []
        for j in (LANK, RANK):
            for a, b, d in reposition_events(im[:, j], 1.0 / dt):
                steps.append((a, b, float(np.linalg.norm(d))))
        lens = sorted(s[2] for s in steps)
        med = float(np.median(lens)) if lens else float("nan")
        return {"events": len(steps), "rate": round(len(steps) / dur, 2),
                "median_step_m": round(med, 3),
                "p10_p90_m": [round(float(np.percentile(lens, 10)), 3),
                              round(float(np.percentile(lens, 90)), 3)] if lens else None,
                "dur_s": round(dur, 2)}

    stalk = cadence(str(POSE2), 122.0, 131.6)
    add("stalk_cadence_events_per_s", stalk["rate"], "1/s", "inferred",
        "ch3 122.0-131.6s: foot-repositioning events (ankle moves >0.10 m within 0.27 s), both feet",
        "low-medium: monocular ankle height noise prevents separating steps from drags")
    add("stalk_step_length_m", stalk["median_step_m"], "m", "inferred",
        f"ch3 122.0-131.6s: median event displacement, p10-p90 {stalk['p10_p90_m']} m", "low-medium")
    circ = cadence(str(POSE2), 178.0, 190.0)
    add("circle_cadence_events_per_s", circ["rate"], "1/s", "inferred",
        "ch4 178.0-190.0s: foot-repositioning events", "low-medium")
    add("circle_step_length_m", circ["median_step_m"], "m", "inferred",
        f"ch4 178.0-190.0s: median event displacement, p10-p90 {circ['p10_p90_m']} m", "low-medium")
    add("circle_stance_width_m", round(geom(p2, s2, 178.0, 190.0)["stance_width_m"], 3), "m",
        "inferred", "ch4 178.0-190.0s median lateral separation", "medium")
    spec["step_events_detail"] = {"stalking": stalk, "circling": circ}
    spec["take_selection"]["stalking"] = {
        "chosen": "112.4-131.6s take; step window 122.0-131.6s",
        "why": "second and longer of two takes (conf 0.969); the first 1.5 s is a walk-in and was excluded",
    }
    spec["take_selection"]["circling"] = {
        "chosen": "177.6-190.0s take; window 178.0-190.0s",
        "why": "highest-confidence circling take without NaN frames",
        "alternates": ["132.5-158.4s take (has NaN frames)", "191.9-202.4s take"],
    }

    # ---- unknown / not measurable -----------------------------------------
    spec["unknown"] = {
        "absolute_metric_scale": "monocular; all metre values are inferred with coach height 1.78 m assumed; ratios are scale-free",
        "depth_axis": "front-to-back distances rely on mediapipe world-landmark depth (approximate); image-plane distances are reliable",
        "contact_forces": "no force information in video; mat contact inferred from landmark height only",
        "knee_yaw_hip_rotation": "rotation about the vertical axis is weakly observable from the monocular track",
        "hand_grip_detail": "finger-level detail is dropped (G1 has no fingers); hands are treated as wrist sites",
        "lead_leg_identity_in_shot": "which leg the coach leads with varies between repetitions; the spec reports the measured lead per window",
    }

    Path(args.out).write_text(json.dumps(spec, indent=2))
    n_obs = sum(1 for v in spec["params"].values() if v["status"] == "observed")
    n_inf = sum(1 for v in spec["params"].values() if v["status"] == "inferred")
    print(f"wrote {args.out}: {len(spec['params'])} params "
          f"({n_obs} observed, {n_inf} inferred)")


if __name__ == "__main__":
    main()
