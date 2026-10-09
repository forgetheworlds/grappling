#!/usr/bin/env python
"""Build the v2 motion-reference set (data/references/motion_refs/v2).

Agent 3 (reference re-timing / dynamic feasibility).  Reuses the v1
composition machinery (cubic-Hermite C1 connectors, root lifting, refusal
gates -- imported from scripts/compose_drill.py) but generates every phase
path from the measured G1 envelope (src/solo/refgen):

  * solved GROUNDED stances (the v1 table solver penetrates up to 8.7 cm
    below stand height -- measured, refgen docstring);
  * CoM-inside-support quasi-static descents (LOWER, LEVEL_CHANGE, entry,
    recovery) -- the v1 video paths carry their CoM OUTSIDE the planted-foot
    hull for 62-100 % of their frames, which is the measured 0.36 s
    fixed-clock wall;
  * weight-transfer-before-lift stepping (shuffle / circle / widen /
    reposition) with reachable foot placement and bounded accelerations;
  * per-phase timing from the measured envelope (scripts/probe_v2_dynamic.py).

Phase NAMES, lead-leg semantics and the label schema are PRESERVED from v1;
`source` is honest about what each v2 phase is: g1_native (solved G1
geometry), generated (synthesised stepping/path replacing an unusable video
path), fused (video timing / GrappleMap geometry), synthetic_connector.
The emitted qpos are kinematic TARGETS -- never demonstrations.

Prints its own verification and refuses to write on boundary violations
(same gates as v1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import mujoco  # noqa: E402

import compose_drill as cd  # noqa: E402  (v1 machinery: blend/lift/gates)
import solo.refgen as rg  # noqa: E402

OUT = REPO / "data/references/motion_refs/v2"
V1 = REPO / "data/references/motion_refs/v1"
GM_REFS = REPO / "data/refs"

# ---- v2 design constants (measured, see RETIMING.md) -----------------------
WIDTH = 0.34            # operator stable band (A1); FK-verified grounded
H_STAND = 0.79          # stand keyframe height
H_STANCE = 0.74         # stance hold height (v1 fused clamp kept)
H_STEP = 0.70           # footwork height: legs need knee margin for the
                        # lateral reach of stepping (measured: at 0.74 the
                        # far leg cannot reach during a transfer)
H_LEVEL = 0.58          # deepest FLAT-SOLE symmetric crouch (ankle limit)
H_ENTRY = 0.62          # repaired shot-entry depth (v1 video: 0.35)
W_ENTRY = 0.40          # operator repair: wider base for the entry
SPLIT_ENTRY = 0.35      # operator repair: rear foot back (sagittal split)
T_LOWER = 2.0           # descent duration (s) -- envelope-probed
T_LEVEL = 1.4           # level-change leg duration (s)
T_ENTRY = 1.2
STEP_DXY = 0.05         # shuffle step length (m)
STEP_TRA = 0.9          # transfer duration (s)
STEP_SWI = 0.4          # swing duration (s)
N_SHUFFLE = 6
N_CIRCLE = 10
YAW_STEP = 12.0         # deg per turn step (bounded pivot slip)
PEN_SCALE = 1.5         # DOUBLE_LEG_PENETRATION time scale (envelope-probed)


def phase(name, source, note, validity="unverified", validity_reason=""):
    return {"id": -1, "name": name, "t_start": None, "t_end": None,
            "source": source, "lead_leg": None, "validity": validity,
            "validity_reason": validity_reason, "note": note}


def seg_of(qpos, contact, ph, dt=rg.DT):
    return {"qpos": np.asarray(qpos, np.float64),
            "contact": np.asarray(contact, np.uint8), "dt": dt, "phase": ph}


def ones_contact(n):
    return np.tile(np.array([1, 1], np.uint8), (n, 1))


def build():
    model = rg._model()
    data = mujoco.MjData(model)
    raw: list[dict] = []

    # ---- STAND -------------------------------------------------------------
    q_stand0, _ = rg.stance_pose(rg.STAND_WIDTH, rg.STAND_HEIGHT, guard=False,
                                 model=model)
    n = int(round(1.0 / rg.DT))
    raw.append(seg_of(np.tile(q_stand0, (n, 1)), ones_contact(n),
                      phase("STAND", "g1_native",
                            "G1 stand keyframe posture, grounded by the v2 "
                            "solver (identical to the certified anchor "
                            "within FK tolerance)")))

    # ---- LOWER_TO_STANCE ---------------------------------------------------
    q_lo, t_lo, meta_lo = rg.descent_path(WIDTH, H_STAND, H_STANCE, T_LOWER,
                                          model=model)
    raw.append(seg_of(q_lo, ones_contact(len(q_lo)),
                      phase("LOWER_TO_STANCE", "g1_native",
                            f"solved quasi-static stand->stance descent "
                            f"({H_STAND}->{H_STANCE} m, width {WIDTH} m, "
                            f"{T_LOWER:.1f} s min-jerk): CoM held inside the "
                            f"support at EVERY frame (min margin "
                            f"{meta_lo['com_margin_min_m']:+.3f} m) -- "
                            f"replaces the v1 video drop whose CoM is outside "
                            f"the support for 97 % of frames (min -0.122 m, "
                            f"peak root xy accel 12.6 m/s^2), the measured "
                            f"0.36 s fixed-clock wall",
                            validity_reason="pending probe")))

    # ---- STANCE_HOLD -------------------------------------------------------
    q_hold, _ = rg.stance_pose(WIDTH, H_STANCE, guard=True, model=model)
    n = int(round(2.0 / rg.DT))
    raw.append(seg_of(np.tile(q_hold, (n, 1)), ones_contact(n),
                      phase("STANCE_HOLD", "fused",
                            f"posture = v1's operator repair (width "
                            f"{WIDTH} m, height {H_STANCE} m) RE-SOLVED by the "
                            f"grounded v2 IK (the v1 table pose penetrates "
                            f"3.8 cm) + guard carriage (A6); measured margin "
                            f"+0.070 m (v1 drill phase: -0.082 m)")))

    # ---- SHUFFLE_F ---------------------------------------------------------
    plan = [rg.StepPlan("left" if i % 2 == 0 else "right",
                        (STEP_DXY * (i + 1), WIDTH / 2 if i % 2 == 0
                         else -WIDTH / 2),
                        transfer_s=STEP_TRA, swing_s=STEP_SWI)
            for i in range(N_SHUFFLE)]
    q_sh, t_sh, c_sh, meta_sh = rg.step_sequence(plan, width=WIDTH,
                                                 height=H_STEP, model=model)
    raw.append(seg_of(q_sh, c_sh,
                      phase("SHUFFLE_F", "generated",
                            f"generated weight-transfer stepping: "
                            f"{N_SHUFFLE} x {STEP_DXY:.2f} m steps, transfer "
                            f"{STEP_TRA:.1f} s BEFORE each lift (B2), swing "
                            f"{STEP_SWI:.1f} s / {meta_sh['travel_m']:.2f} m "
                            f"travel at {H_STEP:.2f} m height -- replaces the "
                            f"v1 video stalk shuffle (pelvis 0.56-0.72 m, "
                            f"root z accel 31.8 m/s^2, CoM outside support "
                            f"59 % of frames)")))

    # ---- SHUFFLE_B (from the shuffled-forward feet) ------------------------
    # the world chain: shuffle_f ends at feet x=STEP_DXY*N; steps go back
    x_end = STEP_DXY * N_SHUFFLE
    plan_b = []
    n_pairs = (N_SHUFFLE + 1) // 2
    for i in range(N_SHUFFLE):
        swing = "right" if i % 2 == 0 else "left"
        y = -WIDTH / 2 if i % 2 == 0 else WIDTH / 2
        back = x_end * ((i // 2) + 1) / n_pairs  # uniform, no crossing, -> 0
        plan_b.append(rg.StepPlan(swing, (x_end - back, y),
                                  transfer_s=STEP_TRA, swing_s=STEP_SWI))
    q_sb, t_sb, c_sb, meta_sb = rg.step_sequence(
        plan_b, width=WIDTH, height=H_STEP,
        feet_xy=((x_end, WIDTH / 2), (x_end, -WIDTH / 2)), model=model)
    raw.append(seg_of(q_sb, c_sb,
                      phase("SHUFFLE_B", "generated",
                            f"generated backward stepping, "
                            f"{meta_sb['travel_m']:.2f} m return (same "
                            f"schedule contract as SHUFFLE_F) -- replaces the "
                            f"v1 video take (CoM outside support 6-25 %, "
                            f"ungrounded timing)")))

    # ---- CIRCLE (turn steps) -----------------------------------------------
    plan_c = [rg.StepPlan("left" if i % 2 == 0 else "right",
                          (0.0, WIDTH / 2 if i % 2 == 0 else -WIDTH / 2),
                          transfer_s=STEP_TRA, swing_s=STEP_SWI,
                          yaw_deg=YAW_STEP)
              for i in range(N_CIRCLE)]
    q_ci, t_ci, c_ci, meta_ci = rg.step_sequence(plan_c, width=WIDTH,
                                                 height=H_STEP, model=model)
    raw.append(seg_of(q_ci, c_ci,
                      phase("CIRCLE", "generated",
                            f"generated turn-stepping (circling identity at "
                            f"G1 capability): {N_CIRCLE} x {YAW_STEP:.0f} deg "
                            f"= {meta_ci['yaw_total_deg']:.0f} deg total, "
                            f"weight-transfer-before-lift schedule -- "
                            f"replaces the v1 video circling (CoM outside "
                            f"support 100 % of frames, min -0.63 m)")))

    # ---- LEVEL_CHANGE (down, hold, up) -------------------------------------
    q_d, t_d, m_d = rg.descent_path(WIDTH, H_STANCE, H_LEVEL, T_LEVEL,
                                    model=model)
    n_h = int(round(0.4 / rg.DT))
    q_u, t_u, m_u = rg.descent_path(WIDTH, H_LEVEL, H_STANCE, T_LEVEL,
                                    model=model)
    q_lc = np.vstack([q_d, np.tile(q_d[-1], (n_h, 1)), q_u])
    c_lc = ones_contact(len(q_lc))
    raw.append(seg_of(q_lc, c_lc,
                      phase("LEVEL_CHANGE", "generated",
                            f"knee-driven level change to the deepest "
                            f"flat-sole crouch ({H_STANCE:.2f}->{H_LEVEL:.2f} "
                            f"m pelvis; ankle limit -0.873 rad is the binding "
                            f"constraint, measured), 0.4 s hold, rise -- "
                            f"replaces the v1 video drop to pelvis 0.40 m "
                            f"(CoM outside support 95 % of frames; the depth "
                            f"band below {H_LEVEL:.2f} m needs sole tilt or a "
                            f"split and is labelled, not silently flattened)")))

    # ---- DOUBLE_LEG_ENTRY_CROUCH (operator two-axis repair) ----------------
    q_en, t_en, m_en = rg.descent_path(W_ENTRY, H_STANCE, H_ENTRY, T_ENTRY,
                                       split=SPLIT_ENTRY, model=model)
    n_eh = int(round(0.5 / rg.DT))
    q_en = np.vstack([q_en, np.tile(q_en[-1], (n_eh, 1))])
    raw.append(seg_of(q_en, ones_contact(len(q_en)),
                      phase("DOUBLE_LEG_ENTRY_CROUCH", "generated",
                            f"REPAIRED entry windup (operator stance rules): "
                            f"width {W_ENTRY:.2f} m + rear foot back "
                            f"{SPLIT_ENTRY:.2f} m (two-axis repair), knee/"
                            f"hip-driven descent to pelvis {H_ENTRY:.2f} m "
                            f"(v1 video crouch: 0.49 m with 60-70 deg torso "
                            f"pitch, waist-folded, CoM margin -0.297 m, "
                            f"toppled in 1.24 s), short 0.5 s load instead of "
                            f"the v1 4.7 s unholdable hold -- every frame "
                            f"CoM-inside-support (min "
                            f"{m_en['com_margin_min_m']:+.3f} m)")))

    # ---- DOUBLE_LEG_PENETRATION (GrappleMap-fused, time-scaled) ------------
    q_pen, t_pen, c_pen, pen_note = load_penetration(model, data)
    # world chain: continue from the entry end (rigid shift, feet included)
    entry_end = raw[-1]["qpos"][-1]
    off = entry_end[:2] - q_pen[0, :2]
    q_pen = q_pen.copy()
    q_pen[:, 0] += off[0]
    q_pen[:, 1] += off[1]
    raw.append(seg_of(q_pen, c_pen,
                      phase("DOUBLE_LEG_PENETRATION", "fused",
                            pen_note)))

    # ---- RECOVER_TO_STANCE (solved rise) -----------------------------------
    q_rec, t_rec, m_rec = rg.descent_path(W_ENTRY, H_ENTRY, H_STANCE, T_ENTRY,
                                          split=SPLIT_ENTRY, model=model)
    n_rs = int(round(0.6 / rg.DT))
    # settle: split -> 0 at stance height (re-solved per frame)
    qs = [rg.stance_pose(WIDTH, H_STANCE,
                         split=SPLIT_ENTRY * (1 - (k + 1) / n_rs),
                         guard=True, model=model)[0]
          for k in range(n_rs)]
    q_rec = np.vstack([q_rec, np.asarray(qs)])
    q_rec = q_rec.copy()
    pen_end = raw[-1]["qpos"][-1]
    q_rec[:, :2] += pen_end[:2] - q_rec[0, :2]
    raw.append(seg_of(q_rec, ones_contact(len(q_rec)),
                      phase("RECOVER_TO_STANCE", "generated",
                            f"solved quasi-static rise from the repaired "
                            f"entry (pelvis {H_ENTRY:.2f}->{H_STANCE:.2f} m) "
                            f"then split -> {WIDTH:.2f} m stance settle -- "
                            f"replaces the v1 video recovery (CoM outside "
                            f"support 99 % of frames)")))

    # ---- REPOSITION (backward steps from wherever we are) ------------------
    x_now = float(raw[-1]["qpos"][-1][0])
    n_rep = 4
    plan_r = []
    for i in range(n_rep):
        swing = "right" if i % 2 == 0 else "left"
        y = -WIDTH / 2 if i % 2 == 0 else WIDTH / 2
        retreat = STEP_DXY * ((i // 2) + 1)      # uniform, no foot crossing
        plan_r.append(rg.StepPlan(swing, (x_now - retreat, y),
                                  transfer_s=STEP_TRA, swing_s=STEP_SWI))
    q_rp, t_rp, c_rp, meta_rp = rg.step_sequence(
        plan_r, width=WIDTH, height=H_STEP,
        feet_xy=((x_now, WIDTH / 2), (x_now, -WIDTH / 2)), model=model)
    raw.append(seg_of(q_rp, c_rp,
                      phase("REPOSITION", "generated",
                            f"generated in-place reposition steps "
                            f"({meta_rp['travel_m']:.2f} m back toward the "
                            f"origin; same schedule contract)")))

    # ---- REPEAT_BLEND (close the cycle to STANCE_HOLD start) ---------------
    prev = raw[-1]
    hold_start = raw[2]["qpos"][0]
    n_rb = int(round(1.5 / rg.DT))
    va = (prev["qpos"][-1] - prev["qpos"][-2]) / prev["dt"]
    s = (np.arange(n_rb) + 1) / (n_rb + 1)
    from compose_drill import hermite
    q_rb = hermite(prev["qpos"][-1][None, :], va[None, :],
                   hold_start[None, :], np.zeros(36)[None, :], s[:, None])
    q_rb[:, 3:7] /= np.linalg.norm(q_rb[:, 3:7], axis=1, keepdims=True)
    c_rb = ones_contact(n_rb)
    from compose_drill import lift_roots
    q_rb = lift_roots(model, data, q_rb)
    raw.append(seg_of(q_rb, c_rb,
                      phase("REPEAT_BLEND", "synthetic_connector",
                            "1.5 s C1 blend from the reposition end posture "
                            "to the STANCE_HOLD start posture -- closes the "
                            "cycle (v1 contract kept)")))

    # ---- pass 2: C1 connectors between consecutive phases ------------------
    out_segments, out_phases = [], []
    for i, seg in enumerate(raw):
        out_segments.append(seg)
        out_phases.append(seg["phase"])
        if i + 1 < len(raw):
            nxt = raw[i + 1]
            maxdist = float(np.max(np.abs(
                raw[i]["qpos"][-1][7:] - nxt["qpos"][0][7:])))
            n_min = int(np.ceil(maxdist / (0.25 * 6.0 * 0.02)))
            n = max(int(round(0.40 / 0.02)), n_min, 20)
            from compose_drill import blend, vel_of
            va, vb = vel_of(seg)[-1], vel_of(nxt)[0]
            q = blend(seg["qpos"][-1], va, nxt["qpos"][0], vb, n, 0.02)
            q = lift_roots(model, data, q)
            cseg = np.tile((seg["contact"][-1] & nxt["contact"][0])[None, :],
                           (n, 1)).astype(np.uint8)
            ph = phase(f"CONNECT_{seg['phase']['name']}->{nxt['phase']['name']}",
                       "synthetic_connector",
                       f"cubic Hermite {n * 0.02:.2f} s (adaptive, pose gap "
                       f"{maxdist:.2f} rad), C1 in position+velocity")
            out_segments.append(seg_of(q, cseg, ph))
            out_phases.append(ph)

    # ---- finalise ----------------------------------------------------------
    qpos = np.concatenate([s["qpos"] for s in out_segments])
    contact = np.concatenate([s["contact"] for s in out_segments])
    phase_id = np.concatenate([np.full(len(s["qpos"]), i, np.int64)
                               for i, s in enumerate(out_segments)])
    t = np.arange(len(qpos)) * rg.DT
    t0 = 0.0
    for i, (s, p) in enumerate(zip(out_segments, out_phases)):
        p["id"] = i
        p["t_start"] = round(t0, 3)
        t0 += len(s["qpos"]) * s["dt"]
        p["t_end"] = round(t0, 3)
        if p["source"] != "synthetic_connector":
            lx, rx = cd.foot_x(model, data, s["qpos"][-1])
            p["lead_leg"] = "left" if lx >= rx else "right"
    jumps = np.abs(np.diff(qpos[:, 7:], axis=0)).max(axis=1)
    root_jumps = float(np.abs(np.diff(qpos[:, :3], axis=0)).max())
    if jumps.max() > cd.REFUSE_JOINT_RAD or root_jumps > cd.REFUSE_ROOT_M:
        print(f"REFUSING to write: boundary jump joint={jumps.max():.4f} rad "
              f"root={root_jumps:.4f} m")
        worst_i = int(np.argmax(jumps))
        print(f"  worst at frame {worst_i}: "
              f"{out_phases[int(phase_id[worst_i])]['name']} -> "
              f"{out_phases[int(phase_id[worst_i + 1])]['name']}")
        return 1
    hold_i = next(i for i, p in enumerate(out_phases)
                  if p["name"] == "STANCE_HOLD")
    repeat_gap = float(np.max(np.abs(
        qpos[-1, 7:] - out_segments[hold_i]["qpos"][0, 7:])))
    meta = {
        "version": "motion_refs/v2",
        "title": "wrestling drill continuous v2 (dynamically re-timed)",
        "dt": rg.DT,
        "duration_s": float(t[-1]),
        "phases": out_phases,
        "composition": {
            "blend_s": "0.40 s minimum, adaptive to the pose gap (6 rad/s cap)",
            "blend_type": "cubic Hermite, C1 (position+velocity)",
            "boundary_max_joint_jump_rad": round(float(jumps.max()), 6),
            "boundary_max_root_jump_m": round(root_jumps, 6),
            "root_displacement_m": round(float(np.linalg.norm(
                qpos[-1, :2] - qpos[0, :2])), 4),
            "repeat_gap_rad": round(repeat_gap, 4),
            "generator": "src/solo/refgen.py (solved grounded stances + "
                         "weight-transfer stepping); v1 connector machinery",
        },
        "validity_note": "phase validity is (re)written by "
                         "scripts/probe_v2_dynamic.py from measured probes; "
                         "'unverified' here means not yet probed in this "
                         "build",
        "reference_not_demonstration":
            "qpos frames are DESIRED kinematic targets; no controller action, "
            "no expert_action, no dynamic-stability claim",
        "build_provenance": {
            "v1_source": "data/references/motion_refs/v1 (labels + fused "
                         "penetration geometry preserved)",
            "design_constants": {"width_m": WIDTH, "h_stand": H_STAND,
                                 "h_stance": H_STANCE, "h_step": H_STEP,
                                 "h_level": H_LEVEL, "h_entry": H_ENTRY,
                                 "w_entry": W_ENTRY, "split_entry_m":
                                     SPLIT_ENTRY, "t_lower_s": T_LOWER,
                                 "pen_scale": PEN_SCALE},
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "drill_continuous.npz", qpos_a=qpos,
                        qpos_b=np.zeros_like(qpos), t=t,
                        technique="drill_continuous",
                        edges=np.array([], dtype=np.int64),
                        landmark_rms=np.nan, contact=contact,
                        phase_id=phase_id, meta=json.dumps(meta))
    print(f"wrote {OUT / 'drill_continuous.npz'}  dur={t[-1]:.2f}s  "
          f"phases={len(out_phases)}  worst joint jump={jumps.max():.5f} rad  "
          f"worst root jump={root_jumps:.5f} m  repeat gap={repeat_gap:.4f} rad")
    return 0


def load_penetration(model, data):
    """The GrappleMap-fused penetration geometry, timing-scaled to v2."""
    d = np.load(GM_REFS / "DOUBLE_LEG.npz", allow_pickle=True)
    q = np.asarray(d["qpos_a"], np.float64)
    t = np.asarray(d["t"], np.float64)
    dt = float(np.median(np.diff(t)))
    c = cd.synth_contact(model, data, q)
    q2, t2, c2 = rg.time_scale(q, t, PEN_SCALE, c)
    # world offset: continue from the entry end (patched at assembly time)
    note = (f"GrappleMap DOUBLE_LEG spatial geometry (G1 FK: pelvis to "
            f"{q[:, 2].min():.3f} m, knee-down depth -- the technique's fused "
            f"identity is PRESERVED, not flattened), time-scaled x{PEN_SCALE} "
            f"({len(q) * dt:.2f}->{len(q2) * rg.DT:.2f} s); contact "
            f"synthesised from FK sole heights.  Traversal verdict comes "
            f"from the probe (this phase needs knee-load dynamics the "
            f"current balance layer may not have -- the label will say so)")
    return q2, t2, c2, note


def _lead(model, data, q_last) -> str:
    lx, rx = cd.foot_x(model, data, q_last)
    return "left" if lx >= rx else "right"


def save_ref(path, q, t, c, meta_extra):
    meta = {
        "version": "motion_refs/v2",
        "dt": rg.DT,
        "duration_s": round(float(t[-1]), 3),
        "reference_not_demonstration":
            "qpos frames are DESIRED kinematic targets; no controller action, "
            "no expert_action, no dynamic-stability claim",
        "validity_note": "validity is (re)written by scripts/"
                         "probe_v2_dynamic.py from measured probes",
    }
    meta.update(meta_extra)
    np.savez_compressed(path, qpos_a=q, qpos_b=np.zeros_like(q), t=t,
                        technique=path.stem,
                        edges=np.array([], dtype=np.int64),
                        landmark_rms=np.nan,
                        contact=np.asarray(c, np.uint8),
                        meta=json.dumps(meta))
    print(f"  {path.name:28s} {t[-1]:6.2f}s")


def build_refs():
    model = rg._model()
    data = mujoco.MjData(model)
    refs = OUT / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    v1_refs = V1 / "refs"
    print("v2 refs:")

    # stance_hold: the repaired hold, 2.5 s
    q, _ = rg.stance_pose(WIDTH, H_STANCE, guard=True, model=model)
    n = int(round(2.5 / rg.DT))
    save_ref(refs / "stance_hold.npz", np.tile(q, (n, 1)),
             np.arange(n) * rg.DT, ones_contact(n),
             {"source": "fused",
              "lead_leg": _lead(model, data, q),
              "validity": "unverified",
              "validity": "unverified",
              "retime": {
                  "what": "v1's raw video hold posture replaced by the v1 "
                          "operator repair (width 0.34, height 0.74) "
                          "RE-SOLVED by the grounded v2 IK + guard; the v1 "
                          "drill phase of the same repair penetrated 3.8 cm "
                          "and margined -0.082 m; this pose is grounded "
                          "(0.0 cm) at +0.070 m margin (FK)",
                  "v1_peer": "v1/refs/stance_hold.npz (raw video posture, "
                             "kept as archive)"},
              "note": "kinematic TARGET; holdability is a measured, labelled "
                      "property"})

    # stand_to_stance: the LOWER take (descent + settle)
    q_lo, t_lo, m_lo = rg.descent_path(WIDTH, H_STAND, H_STANCE, T_LOWER,
                                       model=model)
    n_st = int(round(0.5 / rg.DT))
    q_lo = np.vstack([q_lo, np.tile(q_lo[-1], (n_st, 1))])
    t_lo = np.arange(len(q_lo)) * rg.DT
    save_ref(refs / "stand_to_stance.npz", q_lo, t_lo, ones_contact(len(q_lo)),
             {"source": "g1_native",
              "lead_leg": _lead(model, data, q_lo[-1]),
              "validity": "unverified",
              "retime": {
                  "what": f"v1's video drop (1.22 s, CoM outside support 97 % "
                          f"of frames, min -0.122 m, root xy accel 12.6 m/s^2 "
                          f"-- the measured 0.36 s wall) replaced by a solved "
                          f"quasi-static descent ({T_LOWER:.1f} s + 0.5 s "
                          f"settle), CoM inside support every frame (min "
                          f"{m_lo['com_margin_min_m']:+.3f} m)",
                  "v1_peer": "v1/refs/stand_to_stance.npz"}})

    # stance_widen_step: stand width -> stance width, two lateral steps
    plan_w = [rg.StepPlan("left", (0.0, WIDTH / 2), transfer_frac=0.45),
              rg.StepPlan("right", (0.0, -WIDTH / 2), transfer_frac=0.45)]
    q_w, t_w, c_w, m_w = rg.step_sequence(
        plan_w, width=rg.STAND_WIDTH, height=H_STEP,
        feet_xy=((0.0, 0.1185), (0.0, -0.1185)), model=model)
    save_ref(refs / "stance_widen_step.npz", q_w, t_w, c_w,
             {"source": "generated",
              "lead_leg": _lead(model, data, q_w[-1]),
              "validity": "unverified",
              "retime": {
                  "what": "v1's video widen step replaced by generated "
                          "lateral weight-transfer steps (0.237->0.34 m "
                          "width, transfer-before-lift)",
                  "v1_peer": "v1/refs/stance_widen_step.npz"},
              "note": f"steps={m_w['steps']} travel={m_w['travel_m']} m"})

    # shuffle_back / circle_step at footwork height
    plan_b = []
    for i in range(N_SHUFFLE):
        swing = "right" if i % 2 == 0 else "left"
        y = -WIDTH / 2 if i % 2 == 0 else WIDTH / 2
        back = STEP_DXY * ((i // 2) + 1)         # uniform, no foot crossing
        plan_b.append(rg.StepPlan(swing, (-back, y),
                                  transfer_s=STEP_TRA, swing_s=STEP_SWI))
    q_sb, t_sb, c_sb, m_sb = rg.step_sequence(plan_b, width=WIDTH,
                                              height=H_STEP, model=model)
    save_ref(refs / "shuffle_back.npz", q_sb, t_sb, c_sb,
             {"source": "generated",
              "lead_leg": _lead(model, data, q_sb[-1]),
              "validity": "unverified",
              "retime": {"what": "v1's video reposition take replaced by "
                                 "generated backward weight-transfer steps",
                         "v1_peer": "v1/refs/shuffle_back.npz"},
              "note": f"travel={m_sb['travel_m']} m"})

    plan_c = [rg.StepPlan("left" if i % 2 == 0 else "right",
                          (0.0, WIDTH / 2 if i % 2 == 0 else -WIDTH / 2),
                          transfer_s=STEP_TRA, swing_s=STEP_SWI,
                          yaw_deg=YAW_STEP)
              for i in range(N_CIRCLE)]
    q_ci, t_ci, c_ci, m_ci = rg.step_sequence(plan_c, width=WIDTH,
                                              height=H_STEP, model=model)
    save_ref(refs / "circle_step.npz", q_ci, t_ci, c_ci,
             {"source": "generated",
              "lead_leg": _lead(model, data, q_ci[-1]),
              "validity": "unverified",
              "retime": {"what": "v1's video circling (CoM outside support "
                                 "100 % of frames, min -0.63 m) replaced by "
                                 "generated turn-stepping",
                         "v1_peer": "v1/refs/circle_step.npz"},
              "note": f"yaw_total={m_ci['yaw_total_deg']} deg"})

    # level_change_full / _fast
    for name, leg_t in (("level_change_full", T_LEVEL),
                        ("level_change_fast", 1.0)):
        qd, td, md = rg.descent_path(WIDTH, H_STANCE, H_LEVEL, leg_t,
                                     model=model)
        nh = int(round((0.4 if leg_t > 1.0 else 0.2) / rg.DT))
        qu, tu, mu = rg.descent_path(WIDTH, H_LEVEL, H_STANCE, leg_t,
                                     model=model)
        q_lc = np.vstack([qd, np.tile(qd[-1], (nh, 1)), qu])
        save_ref(refs / f"{name}.npz", q_lc, np.arange(len(q_lc)) * rg.DT,
                 ones_contact(len(q_lc)),
                 {"source": "generated",
                  "lead_leg": _lead(model, data, q_lc[-1]),
                  "validity": "unverified",
                  "retime": {
                      "what": f"v1's video level change (drop to pelvis "
                              f"0.40 m, CoM outside support 95 % of frames) "
                              f"replaced by a knee-driven descent to the "
                              f"deepest FLAT-SOLE crouch ({H_LEVEL:.2f} m; "
                              f"ankle limit -0.873 rad binding, measured), "
                              f"{leg_t:.1f} s per leg + hold + rise",
                      "v1_peer": f"v1/refs/{name}.npz"},
                  "note": "depth band below 0.58 m needs sole tilt or a "
                          "split; labelled rather than silently flattened"})

    # shot_entry_full: repaired entry
    q_e, t_e, m_e = rg.descent_path(W_ENTRY, H_STANCE, H_ENTRY, T_ENTRY,
                                    split=SPLIT_ENTRY, model=model)
    neh = int(round(0.5 / rg.DT))
    q_e = np.vstack([q_e, np.tile(q_e[-1], (neh, 1))])
    save_ref(refs / "shot_entry_full.npz", q_e, np.arange(len(q_e)) * rg.DT,
             ones_contact(len(q_e)),
             {"source": "generated",
              "lead_leg": _lead(model, data, q_e[-1]),
              "validity": "unverified",
              "retime": {
                  "what": "v1's video crouch (pelvis 0.49 m, torso pitch "
                          "60-70 deg, CoM margin -0.297 m, topples in 1.24 s "
                          "under its own targets; certified controller 1.38 "
                          "s; best CEM 2.34 s) REPAIRED along the operator's "
                          "two axes (width 0.40 m + rear foot back 0.35 m), "
                          "knee-driven to 0.62 m, 0.5 s load (v1 held the "
                          "crouch 4.7 s)",
                  "v1_peer": "v1/refs/shot_entry_full.npz"}})

    # shot_recover: rise + settle
    q_r, t_r, m_r = rg.descent_path(W_ENTRY, H_ENTRY, H_STANCE, T_ENTRY,
                                    split=SPLIT_ENTRY, model=model)
    n_rs = int(round(0.6 / rg.DT))
    qs = [rg.stance_pose(WIDTH, H_STANCE,
                         split=SPLIT_ENTRY * (1 - (k + 1) / n_rs),
                         guard=True, model=model)[0] for k in range(n_rs)]
    q_r = np.vstack([q_r, np.asarray(qs)])
    save_ref(refs / "shot_recover.npz", q_r, np.arange(len(q_r)) * rg.DT,
             ones_contact(len(q_r)),
             {"source": "generated",
              "lead_leg": _lead(model, data, q_r[-1]),
              "validity": "unverified",
              "retime": {"what": "v1's video recovery replaced by the solved "
                                 "quasi-static rise from the repaired entry",
                         "v1_peer": "v1/refs/shot_recover.npz"}})

    # knee sprawls + held-out stalk: TIMING-ONLY v2 of the v1 takes
    for name in ("knee_sprawl_entry", "knee_sprawl_hold", "knee_sprawl_recover",
                 "stalk_shuffle", "knee_sprawl_entry2"):
        d1 = np.load(v1_refs / f"{name}.npz", allow_pickle=True)
        q1 = np.asarray(d1["qpos_a"], np.float64)
        t1 = np.asarray(d1["t"], np.float64)
        c1 = np.asarray(d1["contact"], np.uint8)
        m1 = json.loads(str(d1["meta"]))
        q2, t2, c2 = rg.time_scale(q1, t1, 1.5, c1)
        m1["version"] = "motion_refs/v2"
        m1["retime"] = {
            "what": "TIMING-ONLY v2 of the v1 take (uniform 1.5x slow-down; "
                    "geometry and labels unchanged from v1 -- ground-posture "
                    "context takes / held-out probe, excluded from the "
                    "drill's traversable claim)",
            "v1_peer": f"v1/refs/{name}.npz"}
        save_ref(refs / f"{name}.npz", q2, t2, c2, m1)

    # stance_rise (v2 root file): STANCE hold -> rise -> settle
    q_h, _ = rg.stance_pose(WIDTH, H_STANCE, guard=True, model=model)
    n0 = int(round(0.5 / rg.DT))
    q_up, t_up, m_up = rg.descent_path(WIDTH, H_STANCE, H_STAND, 1.4,
                                       model=model)
    n1 = int(round(0.5 / rg.DT))
    q_sr = np.vstack([np.tile(q_h, (n0, 1)), q_up,
                      np.tile(q_up[-1], (n1, 1))])
    t_sr = np.arange(len(q_sr)) * rg.DT
    k0 = n0
    phases2 = [
        {"id": 0, "name": "STANCE", "t_start": 0.0,
         "t_end": round(float(t_sr[k0]), 3), "source": "fused",
         "lead_leg": _lead(model, data, q_sr[-1]), "validity": "unverified",
         "validity_reason": "",
         "note": "repaired stance hold (grounded v2 solver)"},
        {"id": 1, "name": "RISE_TO_STAND", "t_start": round(float(t_sr[k0]), 3),
         "t_end": round(float(t_sr[-1]), 3), "source": "generated",
         "lead_leg": _lead(model, data, q_sr[-1]), "validity": "unverified",
         "validity_reason": "",
         "note": f"solved quasi-static rise ({H_STANCE}->{H_STAND} m, 1.4 s) "
                 f"+ 0.5 s settling hold (v1: video rise, CoM margin -0.088)"},
    ]
    meta2 = {
        "version": "motion_refs/v2", "title": "stance to stand v2",
        "dt": rg.DT, "duration_s": round(float(t_sr[-1]), 3),
        "phases": phases2,
        "composition": {"note": "solved hold + generated rise + settle"},
        "reference_not_demonstration":
            "qpos frames are DESIRED kinematic targets; no controller action",
    }
    np.savez_compressed(OUT / "stance_rise.npz", qpos_a=q_sr,
                        qpos_b=np.zeros_like(q_sr), t=t_sr,
                        technique="stance_rise",
                        edges=np.array([], dtype=np.int64),
                        landmark_rms=np.nan, contact=ones_contact(len(q_sr)),
                        phase_id=np.array([0] * n0 + [1] * (len(q_sr) - n0),
                                          np.int64),
                        meta=json.dumps(meta2))
    print(f"  stance_rise.npz               {t_sr[-1]:6.2f}s")

    # per-take FK diagnostics table
    print("\nper-take measured diagnostics (refgen.measure_track):")
    for npz in sorted(refs.glob("*.npz")):
        d = np.load(npz, allow_pickle=True)
        qq = np.asarray(d["qpos_a"], np.float64)
        tt = np.asarray(d["t"], np.float64)
        cc = np.asarray(d["contact"], np.uint8)
        st = rg.measure_track(qq, tt, cc)
        print(f"  {npz.stem:26s} jv={st['joint_v_max_rad_s']:5.2f} "
              f"axy={st['axy_max_m_s2']:5.2f} pen={st['sole_pen_min_m']:+.4f} "
              f"marg={st['com_margin_min_m']} ds_marg="
              f"{st.get('com_margin_ds_min_m')}")
    return 0


if __name__ == "__main__":
    rc = build()
    if rc == 0:
        print()
        rc = build_refs()
    raise SystemExit(rc)
