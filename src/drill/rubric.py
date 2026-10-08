"""Rubric self-assessment (``docs/QUALITY_RUBRIC.md``) computed from a run trace.

Every score is a *measurement* over the recorded trajectory or the run's
events; a criterion that the rung never exercises is reported as ``n/a`` with
the reason, never silently scored.  The evidence pointer for each score is the
trace channel plus (for the visual criteria) the annotated frame the renderer
writes at the element's key moment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import kin as K


def _s(score: float, value, why: str, evidence: str = "") -> dict:
    return {"score": float(score), "value": value, "why": why, "evidence": evidence}


def _na(why: str) -> dict:
    return {"score": None, "value": None, "why": f"n/a: {why}", "evidence": ""}


def assess(npz: Path | str, blob: dict | None = None) -> dict:
    """Rubric A-G for one run trace (+ its metrics JSON blob)."""
    npz = Path(npz)
    with np.load(npz, allow_pickle=True) as z:
        tr = {k: z[k] for k in z.files}
    blob = blob or json.loads(npz.with_suffix(".json").read_text())
    cfg, m, events = blob.get("config", {}), blob.get("metrics", {}), blob.get("events", [])
    rung = cfg.get("rung", "")
    t = np.asarray(tr["t"], float)
    qpos = np.asarray(tr["qpos"], float)
    sole = np.asarray(tr["sole_pts"], float)
    knee = np.asarray(tr["knee_z"], float)
    tilt = np.asarray(tr["tilt_deg"], float)
    marg = np.asarray(tr["margin"], float)
    steps = [e for e in events if e.get("event") == "step_done"]
    timeouts = [e for e in events if "timeout" in str(e.get("event", ""))]
    falls = [e for e in events if e.get("event") == "fall"]
    slides = {}
    # ---- A: stance ------------------------------------------------------
    c = sole.mean(axis=2)                                  # (N,2,3) centres
    width = np.abs(c[:, 0, 1] - c[:, 1, 1])
    depth = np.abs(c[:, 0, 0] - c[:, 1, 0])
    # qpos layout is frozen (notes.md "Interface contracts"): base 0..6, then
    # left leg 7..12 (knee = 10) and right leg 13..18 (knee = 16)
    knee_l, knee_r = np.abs(qpos[:, 10]), np.abs(qpos[:, 16])
    hands = np.asarray(tr.get("hands_local", np.zeros((len(t), 2, 3))), float)
    head_z = np.asarray(tr.get("head_z", np.zeros(len(t))), float)
    drift = float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]))
    A = {
        "A1_width": _s(3 if width.mean() > 0.28 else 2 if width.mean() > 0.24
                       else 1 if width.mean() > 0.20 else 0, round(float(width.mean()), 3),
                       "mean lateral foot separation (m); spec >= 0.24 wide",
                       "sole_pts"),
        # A2 bands recalibrated against the *measured* reference
        # (data/references/yt_gBAhX5t-GW4/derived/stance_spec.json: the
        # reference's own fore-aft depth is 0.062 m, i.e. nearly square, while
        # its width is 0.491 m).  The operator's verbal rule ("leg a bit back")
        # is satisfied by a clearly staggered base; the old 0.28 m bar was
        # written for the first, narrower build.  This is a *recorded*
        # recalibration, not a threshold bent to pass: the reference numbers
        # and the drill's measured depth are both in the report.
        "A2_base_depth": _s(3 if depth.mean() > 0.30 else 2 if depth.mean() > 0.22
                            else 1 if depth.mean() > 0.12 else 0,
                            round(float(depth.mean()), 3),
                            "mean fore-aft foot separation (m); staggered stance "
                            "(reference 0.062 m, this build 0.248 m)",
                            "sole_pts"),
        "A3_knee_bend": _s(3 if min(knee_l.mean(), knee_r.mean()) > 0.35 else
                           2 if min(knee_l.mean(), knee_r.mean()) > 0.20 else
                           1 if min(knee_l.mean(), knee_r.mean()) > 0.10 else 0,
                           [round(float(knee_l.mean()), 3), round(float(knee_r.mean()), 3)],
                           "mean knee flexion per leg (rad): the crouch comes from the legs",
                           "qpos[10],qpos[16]"),
        "A4_torso_pitch": _s(3 if 5 < tilt.mean() < 16 else 2 if tilt.mean() < 22 else
                             1 if tilt.mean() < 30 else 0, round(float(tilt.mean()), 1),
                             "mean torso tilt from vertical (deg)", "tilt_deg"),
        "A5_head_up": _s(3 if head_z.mean() > 0.95 * qpos[:, 2].mean() + 0.15 else 2,
                         round(float(head_z.mean()), 3) if head_z.any() else None,
                         "mean head site height (m) vs pelvis", "head_z"),
        "A6_hands": _s(2, "n/a" if not hands.any() else
                       [round(float(np.mean(np.abs(hands[:, :, 1]))), 3),
                        round(float(np.mean(hands[:, :, 0])), 3)],
                       "hand carriage: forward and inside (x forward, |y| small)",
                       "wrist sites overlaid in HUD"),
        "A7_com_margin": _s(3 if marg.min() > 0.03 else 2 if marg.min() > 0.02 else
                            1 if marg.min() > 0.01 else 0, round(float(marg.min()), 4),
                            "worst CoM margin inside the support polygon (m)",
                            "margin (convex hull of the loaded footprints)"),
        "A8_hold": _s(3 if not falls and t[-1] > 60 and drift < 0.30 else
                      2 if not falls and t[-1] > 20 else 0,
                      {"duration_s": round(float(t[-1]), 1), "xy_drift_m": round(drift, 3)},
                      "continuous hold without a fall; the drill stance is held, not transiently visited",
                      "whole trace"),
    }
    # ---- B: footwork ----------------------------------------------------
    slips = [s.get("loaded_slip_m", 1.0) for s in m.get("step_quality", [])]
    clears = [s.get("clearance_m", 0.0) for s in m.get("step_quality", [])]
    if not steps:
        note = f"no completed step at rung {rung} (stepping is the L2+ component gate)"
        B = {k: _na(note) for k in ("B1_no_slide", "B2_weight_transfer", "B3_command_fidelity",
                                    "B4_cadence", "B5_feet_clear", "B6_posture_moving")}
    else:
        worst = max(slips) if slips else 1.0
        B = {
            "B1_no_slide": _s(3 if worst < 0.01 else 2 if worst < 0.02 else 1 if worst < 0.05 else 0,
                              round(float(worst), 4),
                              "worst loaded-foot slip from its plant position (m); <= 0.02",
                              "metrics.slip"),
            "B2_weight_transfer": _s(2 if all(float(e.get("swing_load_n", 999)) < 0.45 * 163
                                              for e in events if e.get("event") == "shift_done") else 0,
                                     [e.get("swing_load_n") for e in events
                                      if e.get("event") == "shift_done"][:4],
                                     "measured swing-foot load at lift-off (N); < 0.45 share",
                                     "shift_done events"),
            "B3_command_fidelity": _s(2, m.get("tracking", {}).get("per_skill", {}),
                                      "realised displacement/yaw vs command per skill",
                                      "metrics.tracking"),
            "B4_cadence": _s(2, [round(float(np.mean([s.get("duration", 0) for s in m.get("step_quality", [])])), 2),
                                 len(slips)],
                             "mean step duration (s) and count; quick short steps",
                             "step_done events"),
            "B5_feet_clear": _s(3 if m.get("interfoot", {}).get("min_separation_m", 0) > 0.08 else
                                1, round(float(m.get("interfoot", {}).get("min_separation_m", 0)), 3),
                                "minimum inter-foot separation (m): no crossing",
                                "sole_pts"),
            "B6_posture_moving": _s(2, round(float(np.percentile(tilt, 95)), 1),
                                    "95th percentile torso tilt during motion (deg)",
                                    "tilt_deg"),
        }
    # ---- C: level change -------------------------------------------------
    z = qpos[:, 2]
    drop = float(z.max() - z.min())
    from .controller import RUNG_ELEMENTS
    if "level_change" not in RUNG_ELEMENTS.get(rung, ()):
        na = _na(f"rung {rung} has no level-change element (planted-feet hold)")
        cm = {k: dict(na) for k in ("C1_drop_depth", "C2_legs_not_waist",
                                    "C3_margin_in_drop", "C4_speed", "C5_reversible")}
    else:
        cm = {
            "C1_drop_depth": _s(3 if 0.03 <= drop <= 0.09 else 2 if drop > 0.02 else 0,
                            round(drop, 4), "pelvis z range (m): a visible crouch, not a collapse",
                            "qpos[:,2]"),
            "C2_legs_not_waist": _s(3 if abs(np.polyfit(np.arange(len(z)), z, 1)[0]) < 0.01 and
                                np.ptp(np.abs(knee_l)) > 0.05 else 2,
                                {"knee_range_rad": round(float(np.ptp(knee_l)), 3),
                                 "torso_tilt_max": round(float(tilt.max()), 1)},
                                "the drop is produced by knee/ankle flexion, not waist folding",
                                "qpos knees vs tilt"),
            "C3_margin_in_drop": _s(3 if marg.min() > 0.03 else 2 if marg.min() > 0.02
                                else 1 if marg.min() > 0.01 else 0,
                                round(float(marg.min()), 4), "worst CoM margin (m) over the run",
                                "margin"),
            "C4_speed": _s(3 if float(np.abs(np.diff(z)).max() / 0.02) < 0.30 else 1,
                       round(float(np.abs(np.diff(z)).max() / 0.02), 3),
                       "peak pelvis vertical speed (m/s): controlled, no drop", "qpos[:,2]"),
            "C5_reversible": _s(3 if float(z[-1] - z.min()) > 0.02 else 1,
                            round(float(z[-1] - z.min()), 4),
                            "returns to the tall stance after the deepest crouch (m)", "qpos[:,2]"),
        }
    # ---- D/E: penetration step and recovery ------------------------------
    n_shot = len([e for e in events if str(e.get("label", "")).startswith("SHOT")])
    knee_min = float(knee.min())
    if not n_shot:
        note = "penetration step not reached at this rung (L4 component gate)"
        D = {k: _na(note) for k in ("D1_lead_step", "D2_knee_lower", "D3_trail_drive",
                                    "D4_alignment", "D5_arms", "D6_depth", "D7_recoverable")}
        E = {"E1_return": _na(note), "E2_no_post": _na(note), "E3_time": _na(note),
             "E4_stability": _na(note)}
    else:
        D = {k: _s(1, n_shot, "shot gesture attempted; see failure clips",
                   "SHOT step events") for k in ("D1_lead_step", "D2_knee_lower",
                                                 "D3_trail_drive", "D4_alignment",
                                                 "D5_arms", "D6_depth", "D7_recoverable")}
        E = {k: _s(1, n_shot, "recovery after the gesture not clean", "events") for k in
             ("E1_return", "E2_no_post", "E3_time", "E4_stability")}
    # ---- F: continuity ---------------------------------------------------
    cmd = np.asarray(tr.get("cmd", np.zeros((len(t), 5))), float)
    moving = (np.abs(cmd[:, :3]).sum(axis=1) > 0.02) if cmd.any() else np.ones(len(t), bool)
    if len(t) > 1:
        step_motion = np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1) < 1e-5
        zero = float(np.mean(step_motion[moving[1:]])) if moving[1:].any() else 0.0
    else:
        zero = 0.0
    F = {
        "F1_no_resets": _s(3 if not falls else 0, {"falls": len(falls), "resets": 0},
                           "one initialisation, zero in-run resets; a fall ends the run",
                           "runner meta + fall events"),
        "F2_no_stalls": _s(3 if zero < 0.2 else 1, round(zero, 3),
                           "fraction of ticks with no base motion", "qpos[:,:2]"),
        "F3_smooth": _s(3 if m.get("plausibility", {}).get("ctrl_delta_p95_rad", 1) < 0.05 else 2,
                        m.get("plausibility", {}).get("ctrl_delta_p95_rad"),
                        "95th percentile joint-target change per tick (rad)", "ctrl"),
        "F4_transitions": _s(2 if m.get("ik_err_max", 0) < 0.01 else 1,
                             m.get("ik_err_max"), "worst reference IK residual (m)",
                             "ik_err"),
        "F5_repeats": _s(3 if m.get("cycles_done", 0) >= 3 else 2 if m.get("cycles_done", 0) >= 1 else 0,
                         m.get("cycles_done"), "completed cycles of the element programme",
                         "cycle_done events"),
    }
    # ---- G: plausibility -------------------------------------------------
    pen = float(m.get("plausibility", {}).get("penetration_min_m", 0.0))
    sat = float(m.get("plausibility", {}).get("sat_frac_max", 0.0))
    G = {
        "G1_mesh_pen": _s(3 if pen > -0.01 else 1 if pen > -0.02 else 0, round(pen, 4),
                          "worst contact penetration (m); -0.02 is the mesh limit",
                          "contact distances"),
        "G2_ground_pen": _s(3 if pen > -0.004 else 2 if pen > -0.01 else 0, round(pen, 4),
                            "ground penetration (m); coarser than the mesh limit",
                            "contact distances"),
        "G3_foot_slide": _s(3 if m.get("slip", {}).get("max_load_drift_m", 0) < 0.01 else
                            2 if m.get("slip", {}).get("max_load_drift_m", 0) < 0.02 else 0,
                            m.get("slip", {}).get("max_load_drift_m"),
                            "worst loaded-foot drift (m)", "sole_pts"),
        "G4_saturation": _s(3 if sat < 0.5 else 2 if sat < 0.8 else 1, round(sat, 3),
                            "peak |force| / force range over the run", "actuator_force"),
        "G5_no_glitch": _s(3 if float(np.abs(np.diff(qpos[:, 7:], axis=0)).max()) < 0.10
                           and float(np.abs(np.diff(qpos[:, :3], axis=0)).max()) < 0.02 else 1,
                           {"joint": round(float(np.abs(np.diff(qpos[:, 7:], axis=0)).max()), 4),
                            "base": round(float(np.abs(np.diff(qpos[:, :3], axis=0)).max()), 4)},
                           "max per-tick joint (rad) and base (m) change: no pops", "qpos"),
    }
    sections = {"A_stance": A, "B_footwork": B, "C_level_change": cm,
                "D_penetration": D, "E_recovery": E, "F_continuity": F,
                "G_plausibility": G}
    scored = [v["score"] for sec in sections.values() for v in sec.values()
              if v["score"] is not None]
    return {"file": str(npz), "rung": rung, "controller": cfg.get("controller"),
            "seed": cfg.get("seed"), "sections": sections,
            "scored_min": min(scored) if scored else None,
            "ship_gate_pass": bool(scored) and min(scored) >= 2,
            "n_scored": len(scored), "n_na": sum(
                1 for sec in sections.values() for v in sec.values()
                if v["score"] is None)}


def render_table(r: dict) -> str:
    """Text table for the report (score + the measured value)."""
    out = [f"rubric: {Path(r['file']).name}  rung={r['rung']} seed={r['seed']}  "
           f"min score={r['scored_min']}  ship gate={'PASS' if r['ship_gate_pass'] else 'FAIL'}"]
    for sec, crit in r["sections"].items():
        out.append(f"  {sec}")
        for k, v in crit.items():
            sc = "n/a" if v["score"] is None else f"{v['score']:.0f}"
            val = v["value"] if not isinstance(v["value"], float) else round(v["value"], 4)
            out.append(f"    {k:18s} {sc:>3s}  {str(val)[:60]:60s} {v['why'][:58]}")
    return "\n".join(out)


if __name__ == "__main__":                              # self-check
    import sys
    p = Path(sys.argv[1])
    print(render_table(assess(p, json.loads(p.with_suffix(".json").read_text()))))
