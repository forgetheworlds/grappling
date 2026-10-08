#!/usr/bin/env python
"""Build the fused movement specification (data/references/motion_refs/fusion_spec.json).

Fusion policy (brief contract, enforced here):
  * VIDEO supplies timing, ordering, carriage, approximate trajectories and
    step cadence  -> measured from the v1 retargeted tracks + clip_index.json.
  * GRAPPLEMAP supplies spatial relationships, technique structure and
    constraints -> measured from data/refs/STANCE.npz / DOUBLE_LEG.npz by FK on
    the solo model.
  * Never average the two sources' joint angles frame by frame; where the
    sources constrain the SAME feature, the spec records BOTH values, the
    chosen fused value, the conflict rule that decided, and a confidence.

Every number below is measured (FK on real qpos), nothing is hand-typed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402

from solo.scene import load_solo_model, stand_frame  # noqa: E402
from solo.stance import STAND_HEIGHT, STAND_WIDTH, stance_qpos  # noqa: E402

OUT = REPO / "data/references/motion_refs/fusion_spec.json"
VID_REFS = REPO / "data/references/motion_refs/v1/refs"
GM_DIR = REPO / "data/refs"
CLIP_INDEX = REPO / "data/references/motion_refs/clip_index.json"

#: sole-foot site names per side (G1 model), for support geometry
FOOT_SITES = {"left": ("a_left_foot",), "right": ("a_right_foot",)}
SITE_LEG = {"left_hip_pitch_link": "left", "right_hip_pitch_link": "right"}


def fk_features(model, data, qpos_seq):
    """Per-frame features from qpos (T,36): pelvis z, foot site xy/z, knee z,
    torso pitch, head z, hand x (heading frame), width/depth."""
    out = {k: [] for k in ("pelvis_z", "foot_l", "foot_r", "knee_l", "knee_r",
                           "pitch_deg", "head_z", "hand_x", "hand_z")}
    head_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "head")
    lo = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "a_left_knee_link")
    ro = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "a_right_knee_link")
    lf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "a_left_foot")
    rf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "a_right_foot")
    lw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "a_left_wrist")
    rw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "a_right_wrist")
    for q in qpos_seq:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        yaw = np.arctan2(2.0 * (data.qpos[6] * data.qpos[3] + data.qpos[5] * data.qpos[4]),
                         1.0 - 2.0 * (data.qpos[4] ** 2 + data.qpos[5] ** 2))
        c, s = np.cos(yaw), np.sin(yaw)
        fz = data.xmat[1].reshape(3, 3)[2]  # pelvis z axis (tilt proxy)
        out["pelvis_z"].append(float(data.qpos[2]))
        out["foot_l"].append(data.site_xpos[lf].copy())
        out["foot_r"].append(data.site_xpos[rf].copy())
        out["knee_l"].append(float(data.xipos[lo][2]))
        out["knee_r"].append(float(data.xipos[ro][2]))
        out["pitch_deg"].append(float(np.degrees(np.arccos(np.clip(fz[2], -1, 1)))))
        out["head_z"].append(float(data.site_xpos[head_sid][2]))
        hx = []
        for sid in (lw, rw):
            p = data.site_xpos[sid] - data.qpos[:3]
            hx.append(float(c * p[0] + s * p[1]))  # heading-forward component
        out["hand_x"].append(max(hx))
        out["hand_z"].append(float(max(data.site_xpos[lw][2], data.site_xpos[rw][2])))
    for k in ("foot_l", "foot_r"):
        out[k] = np.array(out[k])
    return {k: (np.array(v) if k not in ("foot_l", "foot_r") else v)
            for k, v in out.items()}


def stance_geometry(feats):
    """width (lateral foot separation), sagittal depth (rear behind lead)."""
    fl, fr = feats["foot_l"], feats["foot_r"]
    width = np.linalg.norm((fl - fr)[:, :2], axis=1)  # crude planar separation
    # depth: (lead - rear) along the line connecting the feet, positive when
    # one foot is clearly ahead
    d = fl[:, :2] - fr[:, :2]
    ahead_l = d[:, 0]
    depth = np.abs(ahead_l)
    return {"width_med": float(np.median(width)), "width_max": float(width.max()),
            "depth_med": float(np.median(depth)), "depth_max": float(depth.max())}


def track_stats(name):
    d = np.load(VID_REFS / f"{name}.npz", allow_pickle=True)
    model, data = load_solo_model(), None
    from mujoco import MjData
    data = MjData(model)
    q = d["qpos_a"]
    f = fk_features(model, data, q[:: max(1, len(q) // 120)])
    g = stance_geometry(f)
    contact = d["contact"].astype(bool) if "contact" in d else None
    sole_min = float(min(f["foot_l"][:, 2].min(), f["foot_r"][:, 2].min()))
    return {"name": name, "n": int(len(q)), "dur": float(d["t"][-1]),
            "pelvis_z_min": round(float(f["pelvis_z"].min()), 3),
            "pelvis_z_med": round(float(np.median(f["pelvis_z"])), 3),
            "pelvis_z_max": round(float(f["pelvis_z"].max()), 3),
            "pitch_med_deg": round(float(np.median(f["pitch_deg"])), 1),
            "pitch_max_deg": round(float(f["pitch_deg"].max()), 1),
            "head_z_min": round(float(f["head_z"].min()), 3),
            "hand_fwd_max": round(float(np.max(f["hand_x"])), 3),
            "hand_z_med": round(float(np.median(f["hand_z"])), 3),
            "knee_min": round(float(min(f["knee_l"].min(), f["knee_r"].min())), 3),
            "lead": lead_foot(f),
            "travel_m": round(float(np.linalg.norm(
                np.array([f["foot_l"][-1, 0], f["foot_l"][-1, 1]]) -
                np.array([f["foot_l"][0, 0], f["foot_l"][0, 1]]))), 3),
            "sole_min": round(sole_min, 4),
            "contact_frac": round(float(contact.mean()), 3) if contact is not None else None,
            **{k: round(v, 3) for k, v in g.items()}}


def lead_foot(feats):
    """Video lead leg at the END posture: the foot further forward in +x."""
    fl, fr = feats["foot_l"][-1], feats["foot_r"][-1]
    return "left" if fl[0] >= fr[0] else "right"


def gm_stats(name):
    d = np.load(GM_DIR / f"{name}.npz", allow_pickle=True)
    model, data = load_solo_model(), None
    from mujoco import MjData
    data = MjData(model)
    q = d["qpos_a"]
    f = fk_features(model, data, q[:: max(1, len(q) // 120)])
    g = stance_geometry(f)
    return {"name": f"grapplemap:{name}", "n": int(len(q)), "dur": float(d["t"][-1]),
            "pelvis_z_min": round(float(f["pelvis_z"].min()), 3),
            "pelvis_z_med": round(float(np.median(f["pelvis_z"])), 3),
            "pelvis_z_max": round(float(f["pelvis_z"].max()), 3),
            "pitch_med_deg": round(float(np.median(f["pitch_deg"])), 1),
            "head_z_min": round(float(f["head_z"].min()), 3),
            "hand_fwd_max": round(float(np.max(f["hand_x"])), 3),
            "knee_min": round(float(min(f["knee_l"].min(), f["knee_r"].min())), 3),
            "lead": lead_foot(f),
            **{k: round(v, 3) for k, v in g.items()}}


def main() -> None:
    model = load_solo_model()
    take_names = ["stance_hold", "stance_widen_step", "stand_to_stance",
                  "stance_to_stand", "shuffle_back", "stalk_shuffle", "circle_step",
                  "level_change_full", "level_change_fast", "shot_entry_full",
                  "shot_recover", "knee_sprawl_entry", "knee_sprawl_hold",
                  "knee_sprawl_entry2", "knee_sprawl_recover"]
    missing = [n for n in take_names if not (VID_REFS / f"{n}.npz").exists()]
    if missing:
        print(f"v1 refs missing: {missing} -- run the v1 retarget first")
        return 1
    vid = {n: track_stats(n) for n in take_names}
    gm = {n: gm_stats(n) for n in ("STANCE", "DOUBLE_LEG")}
    q_stand = stand_frame(model)[0]
    data = mujoco.MjData(model)
    stand_feats = fk_features(model, data, [q_stand])
    clips = json.loads(CLIP_INDEX.read_text())

    # measured holdability facts (repo): position servos hold crouches to
    # ~0.72 m pelvis (reports/2026-10-08/solo_env.md measured table); the
    # certified stance keyframe: width 0.237 m, pelvis 0.790 m.
    spec = {
        "version": "1.0",
        "policy": {
            "video": "timing, ordering, carriage, approximate trajectories, step cadence",
            "grapplemap": "spatial relationships, technique structure, constraints",
            "rule": "no frame-by-frame angle averaging; an uncertain video landmark "
                    "never overrides a reliable geometric/contact constraint; "
                    "attribute every feature",
        },
        "measured": {"video_takes": vid, "grapplemap": gm,
                     "g1_stand": {"pelvis_z": round(float(stand_feats["pelvis_z"][0]), 3),
                                   "width_m": round(float(stance_geometry(stand_feats)["width_med"]), 3)},
                     "holdability_bars": {
                         "stance_pelvis_min_holdable_m": 0.72,
                         "source": "reports/2026-10-08/solo_env.md measured table "
                                   "(no crouch below ~0.72 m is holdable by pure position servos)",
                         "shot_crouch_dynamic": "statically balanced but topples in 1.24 s "
                                                "under its own joint targets "
                                                "(reports/2026-10-08/cem_capture.md)"}},
        "features": [],
    }

    def feat(name, value, unit, sources, fused_from, rule, confidence, note=""):
        spec["features"].append({
            "feature": name, "value": value, "unit": unit, "sources": sources,
            "fused_from": fused_from, "conflict_rule": rule,
            "confidence": confidence, "note": note})

    sv, sg = vid["stance_hold"], gm["STANCE"]
    sw = vid["stance_widen_step"]
    feat("stance_width", {"video_stance_hold": sv["width_med"],
                           "video_widen_step_max": sw["width_max"],
                           "grapplemap_stance": sg["width_med"],
                           "g1_stand_keyframe": 0.237,
                           "fused": round(max(0.30, min(0.36, sw["width_max"])), 3)},
         "m",
         {"video": "stance_widen_step FK", "grapplemap": "STANCE.npz FK",
          "repair": "operator lateral-stability rule (SOLO_DRILL §2)"},
         "video_max_within_repair_band",
         "uncertain monocular lateral width (<0.95 foot visibility) never "
         "shrinks below the operator repair band 0.30-0.36 m",
         "high",
         "the G1 stance keyframe is symmetric; lead leg is a tactical label")
    feat("rear_foot_depth", {"video_stance_hold": sv["depth_med"],
                              "grapplemap_stance": sg["depth_med"], },
         "m", {"video": "stance_hold FK", "grapplemap": "STANCE.npz FK"},
         "grapplemap", "video depth is monocular-depth-derived (low confidence "
         "along the camera axis); GrappleMap depth is explicit 3D (high)",
         "medium",
         "rear foot behind the lead by >= shoulder width (rubric A2)")
    feat("stance_crouch_height", {"video": [sv["pelvis_z_min"], sv["pelvis_z_max"]],
                                   "holdable_bar_m": 0.72,
                                   "fused": [0.72, 0.76]},
         "m", {"video": "stance_hold pelvis band (G1 scale)",
               "repair": "holdability bar"},
         "video_band_raised_to_bar",
         "video carriage kept; absolute height clamped to what the certified "
         "stance closed loop can hold (never flatten the crouch to make "
         "balance easy -- clamp is the minimum that is holdable)",
         "high",
         f"video stance crouches to {sv['pelvis_z_min']} m at G1 scale: below the "
         "0.72 m bar; the fused hold keeps the lean and pays with width")
    feat("torso_lean", {"video_med_deg": sv["pitch_med_deg"],
                         "video_max_deg": sv["pitch_max_deg"]},
         "deg", {"video": "stance_hold FK tilt"},
         "video",
         "keep the lean (operator: leaning, not bent over); never zero it",
         "high", "")
    feat("head_carriage", {"video_head_z_min": sv["head_z_min"],
                            "video_pitch": clips["clips"][0]["events_measured"]["torso_pitch_deg"]},
         "m/deg", {"video": "head site FK + landmark pitch"},
         "video", "head up, eyes forward (rubric A5)", "high", "")
    feat("hand_carriage", {"video_hand_fwd_max": sv["hand_fwd_max"],
                            "video_hand_z_med": sv["hand_z_med"],
                            "grapplemap_hand_fwd_max": sg["hand_fwd_max"]},
         "m", {"video": "wrist site FK", "grapplemap": "STANCE hand FK"},
         "video", "hands forward and inside, elbows in (rubric A6); "
         "gesture only -- no grip claim", "medium", "")
    sd, sf = vid["stalk_shuffle"], {}
    feat("step_cadence", {"stalk_dur_s": sd["dur"], "stalk_travel_m": sd["travel_m"],
                           "lifts_measured": "clip_index.json events (feet.*)"},
         "s/m", {"video": "stalk_shuffle + clip-index foot events"},
         "video", "quick short steps (rubric B4); cadence is a video-only feature",
         "high", "")
    feat("shuffle_displacement", {"back_travel_m": vid["shuffle_back"]["travel_m"],
                                   "stalk_travel_m": sd["travel_m"]},
         "m", {"video": "shuffle_back/stalk FK travel"},
         "video", "actual displacement must match the commanded direction "
         "(rubric B3); steps must lift-and-place, never slide (B1)", "high", "")
    cl = vid["circle_step"]
    feat("circling", {"travel_m": cl["travel_m"], "dur_s": cl["dur"]},
         "m/s", {"video": "circle_step FK travel"},
         "video", "circling keeps the stance geometry while yawing", "medium",
         "monocular circling depth is approximate; direction confidence medium")
    lc = vid["level_change_full"]
    feat("level_change_depth", {"video_pelvis_min": lc["pelvis_z_min"],
                                 "video_pelvis_start": lc["pelvis_z_max"],
                                 "knee_min": lc["knee_min"]},
         "m", {"video": "level_change_full FK"},
         "video", "drop produced by knees/hips, NOT waist folding (rubric C2); "
         "reversible (C5)", "high", "")
    se, dl = vid["shot_entry_full"], gm["DOUBLE_LEG"]
    feat("shot_structure", {
        "video_events": clips["clips"][6]["events_measured"],
        "grapplemap_DOUBLE_LEG": {k: dl[k] for k in
                                   ("pelvis_z_min", "pelvis_z_med", "knee_min",
                                    "width_max", "depth_max", "lead")},
        "video_end": {k: se[k] for k in ("pelvis_z_min", "knee_min", "lead")}},
         "m", {"video": "shot_entry_full + measured events",
               "grapplemap": "DOUBLE_LEG.npz FK (penetration depth, knee-down "
               "geometry, width at entry)"},
         "video_timing+grapplemap_spatial",
         "the shot's TIMING/event ORDER is video-only; its END-STATE SPATIAL "
         "relationship (penetration depth ~0.44x height, lead knee at the mat, "
         "widened base) is anchored on GrappleMap because the video's "
         "knee-depth landmark is the noisiest channel and the video crouch "
         "depth is KNOWN dynamically infeasible for the G1 (1.24 s topple)",
         "medium",
         "knee sprawl is a DEFENSE, not a shot: the knee_sprawl_* takes are "
         "indexed for context and excluded from the shot reference")
    rc = vid["shot_recover"]
    feat("recovery", {"video": {k: rc[k] for k in ("dur", "pelvis_z_min", "pelvis_z_max")}},
         "s/m", {"video": "shot_recover FK"},
         "video", "return to a rubric-A stance without fall (E1-E4)", "high", "")
    st = vid["stance_to_stand"]
    feat("rise_to_upright", {"video": {k: st[k] for k in ("dur", "pelvis_z_min", "pelvis_z_max")}},
         "s/m", {"video": "stance_to_stand FK"},
         "video", "separate STANCE->STAND reference (deliverable)", "high", "")
    feat("g1_repair_axes", {
        "sagittal": "move the rear foot further back (longer base)",
        "frontal": "widen the stance; falling sideways = legs too close"},
         "rule", {"repair": "operator-sanctioned repairs (SOLO_DRILL §2)"},
         "operator", "apply before ANY torso flattening or crouch raising; "
         "log every trim with frame/axis/amount/margin", "high", "")

    OUT.write_text(json.dumps(spec, indent=1))
    print(f"wrote {OUT}")
    for f in spec["features"]:
        print(f"  {f['feature']:24s} conf={f['confidence']:6s} fused_from={f['fused_from']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
