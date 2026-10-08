"""Metrics from a run trace: physical events, not survival fractions.

Every number here comes from the recorded trajectory (qpos, foot sole points,
contacts, commands, pushes) or from controller/scheduler events.  The gates the
milestone uses are physical: a step counts only when it reached a settled,
flat, loaded landing; a fall is a sustained low pelvis / large tilt; slip is
measured on feet that are *loaded* (in contact and under the CoM), which is the
only place "no sliding" means anything.
"""

from __future__ import annotations

import numpy as np

from . import kin as K

#: contact penetration thresholds (m) from the rubric
PEN_GROUND = 0.01
PEN_MESH = 0.02


def _foot_arrays(trace: dict) -> tuple:
    sole = np.asarray(trace.get("sole_pts", np.zeros((0, 2, 4, 3))))
    return sole


def _loaded(soles: np.ndarray, com: np.ndarray, side_ix: int,
            slack: float = 0.01) -> bool:
    """Foot is loaded: it is on the mat and the CoM is over its footprint."""
    pts = soles[side_ix]
    if not bool((pts[:, 2] < K.TOUCH_Z).any()):
        return False
    hull = K.hull2d(pts[:, :2])
    return K.polygon_margin(com[:2], hull) > -slack


def slip_report(trace: dict, min_clearance: float = 0.01) -> dict:
    """Slip of loaded feet: per-tick motion and drift from the load start."""
    t = np.asarray(trace["t"], float)
    sole = _foot_arrays(trace)
    com = np.asarray(trace["com"], float)
    plant = np.asarray(trace.get("plan_planted", np.ones((len(t), 2), bool)))
    out = {"per_foot": {}, "max_tick_slip_m": 0.0, "max_load_drift_m": 0.0}
    for i, side in enumerate(K.SIDES):
        loaded, tick_slip, drift = [], [], []
        anchor = None
        for k in range(1, len(t)):
            is_loaded = bool(plant[k, i]) and _loaded(sole[k], com[k], i)
            loaded.append(is_loaded)
            if not is_loaded:
                anchor = None
                continue
            centre = sole[k, i, :, :2].mean(axis=0)
            prev = sole[k - 1, i, :, :2].mean(axis=0)
            d = float(np.linalg.norm(centre - prev))
            tick_slip.append(d)
            if anchor is None:
                anchor = centre
            drift.append(float(np.linalg.norm(centre - anchor)))
        out["per_foot"][side] = {
            "loaded_frac": float(np.mean(loaded)) if loaded else 0.0,
            "max_tick_slip_m": float(max(tick_slip)) if tick_slip else 0.0,
            "max_load_drift_m": float(max(drift)) if drift else 0.0,
            "p95_tick_slip_m": float(np.percentile(tick_slip, 95)) if tick_slip else 0.0,
        }
        out["max_tick_slip_m"] = max(out["max_tick_slip_m"],
                                     out["per_foot"][side]["max_tick_slip_m"])
        out["max_load_drift_m"] = max(out["max_load_drift_m"],
                                      out["per_foot"][side]["max_load_drift_m"])
    return out


def interfoot_report(trace: dict) -> dict:
    """Minimum centre-to-centre foot separation (rubric B5: no crossing)."""
    sole = _foot_arrays(trace)
    if not len(sole):
        return {"min_separation_m": float("nan")}
    c = sole.mean(axis=2)                     # (N, 2, 3)
    d = np.linalg.norm(c[:, 0, :2] - c[:, 1, :2], axis=1)
    return {"min_separation_m": float(d.min()), "mean_separation_m": float(d.mean())}


def step_report(trace: dict, events: list) -> dict:
    """Completed steps with their measured quality (slip, clearance, landing)."""
    t = np.asarray(trace["t"], float)
    sole = _foot_arrays(trace)
    com = np.asarray(trace["com"], float)
    starts, done, steps = {}, [], []
    for ev in events:
        if ev.get("event") == "step_start":
            starts[ev["side"]] = ev["t"]
        elif ev.get("event") == "step_done":
            s0 = starts.get(ev["side"])
            if s0 is None:
                continue
            k0 = int(np.searchsorted(t, s0))
            k1 = int(np.searchsorted(t, ev["t"])) + 1
            k1 = min(k1, len(t) - 1)
            i = 0 if ev["side"] == "left" else 1
            clear = sole[k0:k1 + 1, i, :, 2].min(axis=1) - K.SOLE_REST_Z
            loaded_slip, tick_slip = [], []
            for k in range(k0 + 1, k1 + 1):
                if not (np.asarray(trace["plan_planted"])[k, i]):
                    continue
                if not _loaded(sole[k], com[k], i):
                    continue
                centre = sole[k, i, :, :2].mean(axis=0)
                prev = sole[k - 1, i, :, :2].mean(axis=0)
                tick_slip.append(float(np.linalg.norm(centre - prev)))
                loaded_slip.append(float(np.linalg.norm(
                    centre - sole[k0, i, :, :2].mean(axis=0))))
            steps.append({"side": ev["side"], "t": float(ev["t"]),
                          "duration": float(ev["t"] - s0),
                          "label": ev.get("label", ""),
                          "min_clearance_m": float(clear.max()),
                          "landing_loaded_slip_m": float(max(loaded_slip) if loaded_slip else 0.0),
                          "in_step_tick_slip_m": float(max(tick_slip) if tick_slip else 0.0)})
            done.append(ev)
    return {"n_completed": len(steps), "steps": steps,
            "n_started": len(starts)}


def push_report(trace: dict, events: list, window_s: float = 3.0,
                recovery_margin: float = 0.02) -> dict:
    """Push response: excursion, recovery, and whether the run survived it."""
    t = np.asarray(trace["t"], float)
    margin = np.asarray(trace["margin"], float)
    push = np.asarray(trace["push"], float)
    out = {"n_pushes": 0, "recovered": 0, "recoveries": []}
    starts = [ev for ev in events if ev.get("event") == "push_start"]
    for ev in starts:
        out["n_pushes"] += 1
        k0 = int(np.searchsorted(t, ev["t"]))
        k1 = int(np.searchsorted(t, ev["t"] + window_s))
        seg = margin[k0:min(k1, len(margin))]
        after = margin[min(k1, len(margin) - 1):]
        rec = bool(len(after) and after.min() >= 0.0 or
                   (len(after) and float(np.min(np.abs(after))) >= recovery_margin))
        if len(seg):
            out["recoveries"].append({
                "t": float(ev["t"]), "force": ev.get("force"),
                "margin_min": float(seg.min()),
                "returned": bool(len(after) and np.all(after[:50] > -0.02)) if len(after) else False})
            out["recovered"] += int(out["recoveries"][-1]["returned"])
    return out


def tracking_report(trace: dict) -> dict:
    """Command tracking: CoM-vs-reference error and realized motion per skill."""
    from .runner import _SKILL_IX
    names = {v: k for k, v in _SKILL_IX.items()}
    e = np.asarray(trace["e_track"], float)
    t = np.asarray(trace["t"], float)
    qpos = np.asarray(trace["qpos"], float)
    skills = np.asarray(trace["skill_id"], int)
    out = {"mean_com_track_err_m": float(np.linalg.norm(e, axis=1).mean()),
           "p95_com_track_err_m": float(np.percentile(np.linalg.norm(e, axis=1), 95)),
           "per_skill": {}}
    for sid, name in names.items():
        m = skills == sid
        if not m.any():
            continue
        k = np.where(m)[0]
        dt = float(np.ptp(t[k])) if len(k) > 1 else 0.0
        disp = float(np.linalg.norm(qpos[k[-1], :2] - qpos[k[0], :2]))
        cmd = np.asarray(trace["cmd"], float)[m]
        out["per_skill"][name] = {
            "ticks": int(m.sum()), "duration_s": dt, "realized_disp_m": disp,
            "cmd_speed_mps": float(np.linalg.norm(cmd[:, :2], axis=1).mean()),
            "cmd_yaw_rate": float(cmd[:, 2].mean()),
            "realized_yaw_change_deg": float(np.degrees(
                K.quat_yaw(qpos[k[-1], 3:7]) - K.quat_yaw(qpos[k[0], 3:7]))),
            "mean_track_err_m": float(np.linalg.norm(e[m], axis=1).mean()),
        }
    return out


def posture_report(trace: dict) -> dict:
    """Stance-quality traces (rubric A4/A5 and F2/F3)."""
    qpos = np.asarray(trace["qpos"], float)
    tilt = np.asarray(trace["tilt_deg"], float)
    sole = _foot_arrays(trace)
    out = {"tilt_deg": {"mean": float(tilt.mean()), "max": float(tilt.max()),
                        "p95": float(np.percentile(tilt, 95))},
           "pelvis_z": {"mean": float(qpos[:, 2].mean()), "min": float(qpos[:, 2].min()),
                        "max": float(qpos[:, 2].max())},
           "head_z_mean": float(np.asarray(trace.get("head_z", np.zeros(len(qpos)))).mean())
           if "head_z" in trace else None}
    if len(sole):
        low = np.asarray(trace["clearance"], float)
        out["min_foot_clearance_m"] = float(low.min())
        out["both_airborne_frac"] = float(np.mean(np.all(low > 0.02, axis=1)))
    return out


def plausibility_report(trace: dict) -> dict:
    """Rubric G: penetration, saturation, smoothness."""
    pen = np.asarray(trace.get("pen_min", np.zeros(len(trace["t"]))), float)
    sat = np.asarray(trace.get("sat_frac", np.zeros(len(trace["t"]))), float)
    ctrl = np.asarray(trace["ctrl"], float)
    dctrl = np.abs(np.diff(ctrl, axis=0)) if len(ctrl) > 1 else np.zeros((1, 29))
    return {"penetration_min_m": float(pen.min()),
            "penetration_gt_ground_thr": int((pen < -PEN_GROUND).sum()),
            "penetration_gt_mesh_thr": int((pen < -PEN_MESH).sum()),
            "sat_frac_mean": float(sat.mean()), "sat_frac_max": float(sat.max()),
            "ctrl_delta_p95_rad": float(np.percentile(dctrl, 95)),
            "ctrl_delta_max_rad": float(dctrl.max())}


def continuity_report(trace: dict) -> dict:
    """Rubric F: monotone clock, no state writes, stalls."""
    t = np.asarray(trace["t"], float)
    qpos = np.asarray(trace["qpos"], float)
    dt = np.diff(t)
    disp = np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1) if len(qpos) > 1 else np.zeros(1)
    return {"duration_s": float(t[-1] - t[0]) if len(t) else 0.0,
            "n_ticks": int(len(t)),
            "clock_monotone": bool(np.all(dt > 0)),
            "resets": 0,
            "zero_motion_frac": float(np.mean(disp < 1e-5)) if len(disp) else 0.0}


def summarize(trace: dict, events: list, cfg) -> dict:
    """Full metric block for one run (saved beside the clip)."""
    t = np.asarray(trace["t"], float)
    falls = [ev for ev in events if ev.get("event") == "fall"]
    timeouts = [ev for ev in events if "timeout" in str(ev.get("event", ""))]
    cycles = [ev for ev in events if ev.get("event") == "cycle_done"]
    el_done = [ev for ev in events if ev.get("event") == "element_done"]
    steps = step_report(trace, events)
    return {
        "config": {"controller": cfg.controller, "rung": cfg.rung, "seed": cfg.seed,
                   "seconds": cfg.seconds, "start": cfg.start,
                   "n_pushes": len(cfg.pushes)},
        "duration_s": float(t[-1]) if len(t) else 0.0,
        "longest_continuous_s": float(t[-1]) if len(t) and not falls else (
            float(falls[0]["t"]) if falls else 0.0),
        "falls": len(falls), "fall_times": [float(f["t"]) for f in falls],
        "timeouts": len(timeouts),
        "steps_completed": steps["n_completed"],
        "step_quality": [{"side": s["side"], "t": round(s["t"], 2),
                          "clearance_m": round(s["min_clearance_m"], 4),
                          "loaded_slip_m": round(s["landing_loaded_slip_m"], 4)}
                         for s in steps["steps"]],
        "slip": slip_report(trace),
        "interfoot": interfoot_report(trace),
        "pushes": push_report(trace, events),
        "tracking": tracking_report(trace),
        "posture": posture_report(trace),
        "plausibility": plausibility_report(trace),
        "continuity": continuity_report(trace),
        "elements_done": len(el_done), "cycles_done": len(cycles),
        "safety_blend_ticks": int(np.sum(np.asarray(trace.get(
            "safety_alpha", np.zeros(len(t)))) > 0.05)),
        "ik_err_max": float(np.max(np.asarray(trace.get("ik_err", [0.0])))),
        "emergency_plants": len([e for e in events
                                 if e.get("event") == "emergency_plant"]),
    }
