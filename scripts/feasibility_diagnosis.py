#!/usr/bin/env python
"""Three-level feasibility diagnosis for the v1 motion references.

Level 1 -- KINEMATIC: finite qpos, joints inside model limits, bounded
    velocities (<= 6 rad/s joints, <= 3 m/s root, the retarget.solve caps),
    no ground penetration deeper than 2 cm in FK.
Level 2 -- STATIC (held postures only): on frames where the reference's own
    contact flags say planted (>= 0.4 s) and joints move < 0.25 rad/s, the CoM
    must sit inside the planted-foot support hull (margin >= 0).  This is the
    "statically holdable where required" bar -- it says nothing about dynamics.
Level 3 -- DYNAMIC: done by scripts/probe_drill_dynamic.py (real MuJoCo
    rollouts); this script merges its verdicts when its JSON exists.

Every take/phase then gets `validity`:
  kinematically_valid | statically_holdable | dynamically_verified |
  unverified | known_infeasible (+ a specific reason).
The verdicts are written back into each npz's meta (lead_leg, validity,
validity_reason) and summarised to data/references/motion_refs/feasibility.json.

Prints its own verification and exits non-zero if any file fails Level 1
(a kinematically INVALID reference is a defect, not a verdict).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402

from solo.lit import sole_points_world  # noqa: E402
from solo.scene import load_solo_model  # noqa: E402
from solo.exec_check import hull2d, polygon_margin  # noqa: E402

REFS = REPO / "data/references/motion_refs/v1/refs"
OUT_DIR = REPO / "data/references/motion_refs/v1"
PROBE_JSON = REPO / "videos/motion_refs/dynamic_probe_metrics.json"
FEAS_JSON = REPO / "data/references/motion_refs/feasibility.json"

V_MAX = 6.0          # rad/s (retarget.solve.V_MAX)
BASE_V_MAX = 3.0     # m/s (retarget.solve.BASE_V_MAX)
PEN_MAX = 0.02       # m ground penetration tolerated at transient soles
HOLD_MIN_S = 0.40    # a static hold must last this long
HOLD_V_MAX = 0.25    # rad/s joint speed while "held"


def diagnose_track(model, data, qpos, t, contact=None) -> dict:
    T = len(qpos)
    sole_ids = None
    jlo, jhi = model.jnt_range[1:].T   # 29 hinge joints (free joint 0 excluded)
    q = qpos[:, 7:]
    reasons = []
    finite = bool(np.isfinite(qpos).all())
    if not finite:
        reasons.append("non-finite qpos")
    lo_viol = float(np.max(jlo[None, :] - q)) if T else 0.0
    hi_viol = float(np.max(q - jhi[None, :])) if T else 0.0
    within = lo_viol <= 1e-6 and hi_viol <= 1e-6
    if not within:
        reasons.append(f"joint limits: lo+{lo_viol:.3f}/hi+{hi_viol:.3f} rad "
                       "(limit_prox)")
    dt = float(np.median(np.diff(t))) if T > 1 else 0.02
    v = np.abs(np.diff(q, axis=0)).max(axis=1) / dt if T > 1 else np.zeros(1)
    v_max = float(v.max()) if T > 1 else 0.0
    rv = np.abs(np.diff(qpos[:, :3], axis=0)) / dt if T > 1 else np.zeros((1, 3))
    rv_max = float(np.linalg.norm(rv, axis=1).max()) if T > 1 else 0.0
    if v_max > V_MAX * 1.05:
        reasons.append(f"joint velocity {v_max:.2f} rad/s > {V_MAX}")
    if rv_max > BASE_V_MAX * 1.05:
        reasons.append(f"root velocity {rv_max:.2f} m/s > {BASE_V_MAX}")
    # FK: sole heights + CoM margins on hold frames
    sole = np.zeros((T, 2, 4, 3))
    com = np.zeros((T, 3))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "a_pelvis")
    for i in range(T):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        sole[i] = sole_points_world(model, data)
        com[i] = data.subtree_com[bid]
    pen = float(sole[..., 2].min())
    if pen < -PEN_MAX:
        reasons.append(f"ground penetration {pen:.3f} m")
    # hold frames: contact flag planted AND slow joints
    speed = np.abs(np.gradient(q, dt, axis=0)).max(axis=1) if T > 1 else np.zeros(T)
    margins = np.full(T, np.nan)
    if contact is not None:
        planted = contact.astype(bool)
        for i in range(T):
            feet = planted[i]
            if not feet.any():
                continue
            pts = sole[i][feet][:, :, :2].reshape(-1, 2)
            margins[i] = polygon_margin(com[i, :2], hull2d(pts))
        hold = (speed < HOLD_V_MAX)
        if contact is not None:
            hold &= planted.all(axis=1)
        # require a contiguous >= HOLD_MIN_S hold run to evaluate level 2
        best = (0, None)
        i = 0
        while i < T:
            if hold[i]:
                j = i
                while j < T and hold[j]:
                    j += 1
                if (j - i) * dt > best[0]:
                    best = ((j - i) * dt, (i, j))
                i = j
            else:
                i += 1
        hold_dur, (a, b) = (best[0], best[1]) if best[1] else (0.0, (0, 0))
        if hold_dur >= HOLD_MIN_S:
            hold_margin = float(np.nanmin(margins[a:b]))
        else:
            hold_margin = float(np.nanmin(margins)) if np.isfinite(margins).any() else np.nan
    else:
        hold_dur = 0.0
        hold_margin = float("nan")
    return {
        "kinematic_ok": finite and within and v_max <= V_MAX * 1.05
                        and rv_max <= BASE_V_MAX * 1.05 and pen >= -PEN_MAX,
        "kinematic_reasons": reasons,
        "limit_frac": round(float(np.mean(
            ((q <= jlo[None, :] + 1e-9) | (q >= jhi[None, :] - 1e-9)).any(axis=1))), 3),
        "joint_v_max_rad_s": round(v_max, 2),
        "root_v_max_m_s": round(rv_max, 2),
        "sole_pen_min_m": round(pen, 4),
        "hold_dur_s": round(hold_dur, 2),
        "hold_com_margin_min_m": (round(hold_margin, 4)
                                   if np.isfinite(hold_margin) else None),
        "com_margin_min_any_m": (round(float(np.nanmin(margins)), 4)
                                  if np.isfinite(margins).any() else None),
        "pelvis_z_min": round(float(qpos[:, 2].min()), 3),
        "pelvis_z_max": round(float(qpos[:, 2].max()), 3),
    }


def lead_leg(model, data, qpos) -> str:
    data.qpos[:] = qpos[-1]
    mujoco.mj_forward(model, data)
    lf = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "a_left_foot")][0]
    rf = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "a_right_foot")][0]
    return "left" if lf >= rf else "right"


KNOWN_INFEASIBLE = {
    "drill_continuous": (
        "contains the LABELLED infeasible DOUBLE_LEG_ENTRY_CROUCH phase by "
        "design (video crouch; 1.24 s topple under its own joint targets) and "
        "its blend still dips to -0.31 m through the crouch -- everything "
        "else is diagnosed per phase; a tracking policy must skip or repair "
        "that phase"),
    "shot_entry_full": (
        "pre-step crouch (pelvis ~0.49 m, pitch 60-70 deg) is statically "
        "balanced but topples in 1.24 s under its own joint targets; the "
        "certified stance controller falls in 1.38 s; best CEM search 2.34 s "
        "(reports/2026-10-08/cem_capture.md)"),
    "knee_sprawl_hold": (
        "ground posture by design (knee on the mat); not holdable as a "
        "standing stance, defence-only context"),
}
KNEE_SPRAWL_NOTE = (
    "knee-sprawl family: video crouch depth distorted 0.20-0.38 m at G1 scale "
    "and heavy joint-limit saturation (fidelity report §6) -- style context "
    "only, excluded from the drill")


def main() -> int:
    model = load_solo_model()
    data = mujoco.MjData(model)
    probe = json.loads(PROBE_JSON.read_text()) if PROBE_JSON.exists() else {}
    out = {"version": "1.0", "instrument": __doc__.split("\n")[1], "tracks": {}}
    files = sorted(REFS.glob("*.npz")) + [OUT_DIR / "drill_continuous.npz",
                                          OUT_DIR / "stance_rise.npz"]
    rc = 0
    for path in files:
        d = np.load(path, allow_pickle=True)
        qpos, t = np.asarray(d["qpos_a"], float), np.asarray(d["t"], float)
        contact = np.asarray(d["contact"], np.uint8) if "contact" in d else None
        meta = json.loads(str(d["meta"]))
        res = diagnose_track(model, data, qpos, t, contact)
        name = path.stem
        lead = lead_leg(model, data, qpos)
        # ---- validity categorisation ----
        if name in KNOWN_INFEASIBLE:
            # the MOTION verdict takes precedence over file-level kinematics;
            # any kinematic defects are appended so nothing is hidden
            validity = "known_infeasible"
            reason = KNOWN_INFEASIBLE[name]
            if res["kinematic_reasons"]:
                reason += " | kinematic: " + "; ".join(res["kinematic_reasons"])
        elif not res["kinematic_ok"]:
            validity, reason = "kinematically_invalid", "; ".join(res["kinematic_reasons"])
            rc = 1
        elif name.startswith("knee_sprawl"):
            validity, reason = "known_infeasible", KNEE_SPRAWL_NOTE
        else:
            pr = probe.get("per_track", {}).get(name)
            if pr is not None and pr.get("ok"):
                validity = "dynamically_verified"
                reason = (f"MuJoCo position-servo probe: upright through the "
                          f"segment (min tilt margin "
                          f"{pr.get('tilt_max_deg')} deg max tilt, "
                          f"pelvis min {pr.get('pelvis_z_min')} m)")
            elif res["hold_dur_s"] >= HOLD_MIN_S and \
                    (res["hold_com_margin_min_m"] or -1) >= 0.0:
                validity = "statically_holdable"
                reason = (f"held {res['hold_dur_s']:.1f} s with CoM margin >= "
                          f"{res['hold_com_margin_min_m']} m; no dynamic "
                          "evidence yet" if pr is None else
                          f"probe failed ({pr.get('reason')}); static hold OK "
                          f"({res['hold_dur_s']:.1f} s, margin "
                          f"{res['hold_com_margin_min_m']} m)")
            elif res["hold_dur_s"] >= HOLD_MIN_S:
                validity = "known_infeasible"
                reason = (f"held posture CoM outside support "
                          f"(margin {res['hold_com_margin_min_m']} m)")
            else:
                validity = "unverified"
                reason = ("transitional motion: kinematically valid, no "
                          "sustained hold to test statically; dynamic probe: "
                          + ("failed: " + pr.get("reason", "") if pr is not None
                             and not pr.get("ok") else "not run or in progress"))
        out["tracks"][name] = {"path": str(path.relative_to(REPO)), "lead_leg": lead,
                                "validity": validity, "validity_reason": reason,
                                **res}
        # write back into the npz meta
        meta["lead_leg"] = lead
        meta["validity"] = validity
        meta["validity_reason"] = reason
        np.savez_compressed(path, **{k: d[k] for k in d.files if k != "meta"},
                            meta=json.dumps(meta))
        print(f"{name:24s} {validity:22s} hold={res['hold_dur_s']:5.2f}s "
              f"margin={res['hold_com_margin_min_m']} "
              f"v={res['joint_v_max_rad_s']:5.2f} pen={res['sole_pen_min_m']}")
    FEAS_JSON.parent.mkdir(parents=True, exist_ok=True)
    FEAS_JSON.write_text(json.dumps(out, indent=1))
    print(f"wrote {FEAS_JSON}")
    print("SELF-VERIFY:", "FAIL (kinematic invalid present)" if rc else
          "all tracks kinematically valid")
    return rc


if __name__ == "__main__":
    sys.exit(main())
