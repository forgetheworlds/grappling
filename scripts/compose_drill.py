#!/usr/bin/env python
"""Compose the continuous wrestling-drill G1 reference (v1 dataset).

Builds, from the v1 retargeted takes (data/references/motion_refs/v1/refs):

  drill_continuous.npz  STAND -> LOWER_TO_STANCE -> STANCE_HOLD -> SHUFFLE_F
                        -> SHUFFLE_B -> CIRCLE -> LEVEL_CHANGE
                        -> DOUBLE_LEG_ENTRY -> RECOVER_TO_STANCE
                        -> REPOSITION (shuffle) [-> REPEAT affordance]
  stance_to_stand.npz   the separate STANCE -> STAND rise

Composition rules (brief contract):
  * every phase is labelled observed / g1_native / fused / synthetic_connector;
  * boundaries are C1 (cubic Hermite on all 36 qpos channels, matching BOTH
    position and finite-difference velocity) -> no teleport, no joint snap, no
    zero-velocity freeze;
  * the world offset chain shifts each incoming take so root displacement
    ACCUMULATES in the world (no treadmill reset);
  * if the incoming take's lead foot (sagittal) differs from the outgoing
    posture's by > 0.12 m, the incoming take is MIRRORED (left/right swap with
    the correct root reflection) so consecutive stances stay compatible;
  * contact flags are merged; connector frames keep the conservative AND of
    the endpoints;
  * nothing is deleted to hide infeasibility: the DOUBLE_LEG_ENTRY phase keeps
    the video crouch; its known-infeasibility is recorded in the phase table
    (validity filled by scripts/feasibility_diagnosis.py).

Prints its own verification (boundary metrics) and refuses to write the npz if
any boundary is discontinuous.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402

from solo.mirror import mirror_joint_vector, mirror_maps  # noqa: E402
from solo.scene import load_solo_model  # noqa: E402

REFS = REPO / "data/references/motion_refs/v1/refs"
GM_REFS = REPO / "data/refs"
OUT_DIR = REPO / "data/references/motion_refs/v1"
BLEND_S = 0.40          # connector duration (measured smooth at 50 Hz)
LEAD_MIRROR_M = 0.12    # sagittal lead mismatch that triggers a mirror
#: phase-length caps: the v1 retimer stretched the stalking take to 30 s
#: (stretch 3.12) and circling to 17 s; the drill keeps the first 10 s of each
#: (cadence preserved -- the trims are logged in the phase notes)
PHASE_CAP_S = {"SHUFFLE_F": 10.0, "CIRCLE": 10.0}
#: the video knee-down end-state is NOT G1-reachable through this pipeline
#: (solver collapses to pelvis 0.05 m; the video asks 0.22 m), so the fused
#: entry takes the video up to the crouch and switches to the GrappleMap
#: DOUBLE_LEG penetration (G1-verified: pelvis 0.333 m, knee 0.064 m,
#: depth 0.585 m) -- the fusion_spec.json 'shot_structure' policy
ENTRY_CUT_PELVIS = 0.35
#: C0 sanity gate for the WRITTEN track (connectors make joins C1; a bigger
#: step means a defective take or connector).  The root bar is the retarget
#: pipeline's OWN documented cap (retarget.solve.BASE_V_MAX = 3 m/s -> 0.06
#: m/frame at 50 Hz; the level-change drop legitimately runs ~1-3 m/s
#: vertically).  Joint bar = V_MAX*dt with a 5 % margin.
REFUSE_JOINT_RAD = 0.13
REFUSE_ROOT_M = 0.065
ENTRY_PELVIS_NEAR = 0.02   # GrappleMap frame whose pelvis is within this of the cut

#: (phase name, take name or None, source kind)
PLAN = [
    ("STAND",             None,                "g1_native"),
    ("LOWER_TO_STANCE",   "stand_to_stance",   "observed"),
    ("STANCE_HOLD",       "stance_hold",       "fused"),      # observed posture, synthetic hold
    ("SHUFFLE_F",         "stalk_shuffle",     "observed"),
    ("SHUFFLE_B",         "shuffle_back",      "observed"),
    ("CIRCLE",            "circle_step",       "observed"),
    ("LEVEL_CHANGE",      "level_change_full", "observed"),
    ("DOUBLE_LEG_ENTRY",  "shot_entry_full",   "observed"),
    ("RECOVER_TO_STANCE", "shot_recover",      "observed"),
    ("REPOSITION",        "shuffle_back",      "observed"),
]

STAND_HOLD_S = 1.0
STANCE_HOLD_S = 2.0


def load_take(name: str):
    d = np.load(REFS / f"{name}.npz", allow_pickle=True)
    q = np.asarray(d["qpos_a"], np.float64)
    t = np.asarray(d["t"], np.float64)
    contact = np.asarray(d["contact"], np.uint8)
    meta = json.loads(str(d["meta"]))
    return q, t, contact, meta


def synth_contact(model, data, q: np.ndarray, tol=0.015) -> np.ndarray:
    """Contact flags from FK for legacy-format refs (no contact array): a foot
    is planted when its lowest sole sphere is within tol of the mat."""
    import mujoco as mj
    from solo.lit import sole_points_world
    out = np.zeros((len(q), 2), np.uint8)
    for i in range(len(q)):
        data.qpos[:] = q[i]
        mj.mj_forward(model, data)
        sole = sole_points_world(model, data)
        out[i] = (sole[..., 2].min(axis=1) < tol).astype(np.uint8)
    return out


def foot_x(model, data, q):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    lf = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "a_left_foot")]
    rf = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "a_right_foot")]
    return float(lf[0]), float(rf[0])


def mirror_qpos(q: np.ndarray, model) -> np.ndarray:
    """Mirror a (36,) qpos across the sagittal (x,z) plane; keeps facing +x."""
    out = q.copy()
    out[1] = -q[1]                                   # root y
    w, x, y, z = q[3:7]
    out[3:7] = (w, x, -y, -z)                        # reflection conjugation
    out[7:36] = mirror_joint_vector(q[7:36], model)
    return out


def hermite(p0, v0, p1, v1, s):
    """Cubic Hermite blending columns; s in [0,1] broadcastable."""
    s2, s3 = s * s, s ** 3
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1


def lift_roots(model, data, q, floor=0.008, iters=5, max_step=0.01):
    """Raise the root until no sole sphere is below `floor` (big pose-gap
    angle-space blends otherwise fold the legs through the mat).  The bump is
    enveloped by sin(pi*s) (endpoints untouched -> C1 kept) and rate-limited
    to `max_step` per frame (0.5 m/s vertical -- trackable)."""
    import mujoco as mj
    from solo.lit import sole_points_world
    env = np.sin(np.pi * np.linspace(0.0, 1.0, len(q)))
    bump = np.zeros(len(q))
    for _ in range(iters):
        pens = np.zeros(len(q))
        for i in range(len(q)):
            data.qpos[:] = q[i]
            data.qpos[2] += bump[i]
            mj.mj_forward(model, data)
            pens[i] = floor - float(sole_points_world(model, data)[..., 2].min())
        rise = np.maximum(pens, 0.0) * env * 1.2
        if rise.max() <= 0.0:
            break
        # rate-limit the PROFILE in both directions: endpoints stay ~0 (C1)
        # and no per-frame step exceeds max_step (0.5 m/s vertical)
        b = rise.copy()
        for i in range(1, len(b)):
            b[i] = np.clip(b[i], b[i - 1] - max_step, b[i - 1] + max_step)
        for i in range(len(b) - 2, -1, -1):
            b[i] = np.clip(b[i], b[i + 1] - max_step, b[i + 1] + max_step)
        bump = b
    q = q.copy()
    q[:, 2] += bump
    return q


def blend(qa, va, qb, vb, n, dt):
    """(n,36) cubic Hermite from (qa,va) to (qb,vb), quat renormalised."""
    s = (np.arange(n) + 1) / (n + 1)
    q = hermite(qa[None, :], va[None, :], qb[None, :], vb[None, :], s[:, None])
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    return q


def vel_of(seg):
    v = np.zeros_like(seg["qpos"])
    v[1:-1] = (seg["qpos"][2:] - seg["qpos"][:-2]) / (2 * seg["dt"])
    v[0] = (seg["qpos"][1] - seg["qpos"][0]) / seg["dt"]
    v[-1] = (seg["qpos"][-1] - seg["qpos"][-2]) / seg["dt"]
    return v


def main() -> int:
    model = load_solo_model()
    data = mujoco.MjData(model)
    from solo.scene import stand_frame
    q_stand_kf = stand_frame(model)[0]
    qc, tc, cc, _ = load_take("stance_hold")
    # STANCE_HOLD posture: video carriage + operator two-axis repair
    # (SOLO_DRILL §2): width 0.34 m (video 0.47 lateral is outside the G1's
    # stable band after scaling; falling sideways = legs too close, so the
    # repair WIDENS to the stable band), height 0.74 m (video crouch 0.62-0.67
    # is below the 0.72 m holdability bar; the clamp is the minimum that is
    # holdable).  The RAW video posture stays in refs/stance_hold.npz with its
    # own measured verdict.
    from solo.stance import stance_qpos
    q_hold = stance_qpos(width=0.34, height=0.74, model=model)

    # ---- pass 1: raw segments in order (alignment + mirroring) -------------
    raw: list[dict] = []

    def phase(name, source, note, validity="unverified", validity_reason=""):
        return {"id": -1, "name": name, "t_start": None, "t_end": None,
                "source": source, "lead_leg": None, "validity": validity,
                "validity_reason": validity_reason, "note": note}

    def seg_of(qpos, contact, dt, ph):
        return {"qpos": np.asarray(qpos, np.float64),
                "contact": np.asarray(contact, np.uint8), "dt": dt, "phase": ph}

    def align_take(name, ph, cap_s=0.0, cut_pelvis=0.0):
        q, t, contact, meta = load_take(name)
        dt = float(np.median(np.diff(t)))
        note = ph["note"]
        if cap_s > 0 and len(q) > cap_s / dt:
            q, contact = q[:int(cap_s / dt)], contact[:int(cap_s / dt)]
            note += f" [trimmed to first {cap_s:.0f} s of the v1 take; full take in refs/]"
        if cut_pelvis > 0:
            below = np.flatnonzero(q[:, 2] < cut_pelvis)
            cut = int(below[0]) if len(below) else len(q)
            q, contact = q[:cut], contact[:cut]
            note += (f" [video part ends at the crouch cut: first frame with "
                     f"pelvis < {cut_pelvis} m; the video's knee-down end-state "
                     f"(pelvis 0.22 m) is NOT G1-reachable through this "
                     f"pipeline -- solver collapse to 0.05 m -- so the "
                     f"penetration below is GrappleMap-anchored "
                     f"(fusion_spec.json shot_structure)]")
        if raw:
            prev_q = raw[-1]["qpos"][-1]
            lx, rx = foot_x(model, data, prev_q)
            plx, prx = foot_x(model, data, q[0])
            sgn = np.sign(lx - rx + 1e-12)
            flipped = ((plx - prx) * sgn < -LEAD_MIRROR_M
                       and abs(plx - prx) > LEAD_MIRROR_M)
            if flipped:
                q = np.stack([mirror_qpos(qq, model) for qq in q])
                contact = contact[:, ::-1].copy()
                note += " [mirrored: lead-leg compatibility]"
            # prev_q is ABSOLUTE (previous offsets already baked in), so the
            # incoming take shifts by exactly the endpoint mismatch
            off = prev_q[:2] - q[0][:2]
            q = q.copy()
            q[:, 0] += off[0]
            q[:, 1] += off[1]
            seg = seg_of(q, contact, dt, ph)
            seg["off"] = off
        else:
            seg = seg_of(q, contact, dt, ph)
            seg["off"] = np.zeros(2)
        ph["note"] = note
        raw.append(seg)

    def align_gm(name, ph, near_pelvis=None):
        """Fused penetration: a GrappleMap reference (legacy format, contact
        synthesised from FK).  Enters at the frame whose pelvis matches
        ``near_pelvis``; with None, enters at frame 0 (the standing start --
        the connector then supplies the depth reset from the video crouch,
        which is itself labelled synthetic)."""
        d = np.load(GM_REFS / f"{name}.npz", allow_pickle=True)
        q = np.asarray(d["qpos_a"], np.float64)
        t = np.asarray(d["t"], np.float64)
        dt = float(np.median(np.diff(t)))
        if near_pelvis is None:
            start = 0
        else:
            start = int(np.argmin(np.abs(q[:, 2] - near_pelvis)))
        q, t = q[start:], t[start:] - t[start]
        contact = synth_contact(model, data, q)
        ph["note"] += (f" [entered at GrappleMap frame {start} (pelvis "
                       f"{q[0, 2]:.3f} m); contact synthesised from FK sole "
                       f"heights]")
        prev_q = raw[-1]["qpos"][-1]
        off = prev_q[:2] - q[0][:2]
        q = q.copy()
        q[:, 0] += off[0]
        q[:, 1] += off[1]
        seg = seg_of(q, contact, dt, ph)
        seg["off"] = off
        raw.append(seg)

    def align_repeat_blend(hold_q, hold_contact):
        """Close the cycle: C1 blend from the last take's end posture to the
        STANCE_HOLD start posture so REPEAT is continuous by construction."""
        prev = raw[-1]
        n = int(round(1.5 / 0.02))
        va = (prev["qpos"][-1] - prev["qpos"][-2]) / prev["dt"]
        s = (np.arange(n) + 1) / (n + 1)
        q = hermite(prev["qpos"][-1][None, :], va[None, :],
                    hold_q[None, :], np.zeros(36)[None, :], s[:, None])
        q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
        c = np.tile((prev["contact"][-1] & hold_contact)[None, :], (n, 1)).astype(np.uint8)
        ph = phase("REPEAT_BLEND", "synthetic_connector",
                   "1.5 s C1 blend from the reposition end posture to the "
                   "STANCE_HOLD start posture -- closes the cycle so it can "
                   "repeat without a teleport",
                   validity_reason="synthetic blend; diagnosed with the rest")
        raw.append(seg_of(q, c, 0.02, ph))

    n_hold = lambda sec: int(round(sec / 0.02))

    nh = n_hold(STAND_HOLD_S)
    raw.append(seg_of(np.tile(q_stand_kf, (nh, 1)),
                      np.tile(np.array([1, 1], np.uint8), (nh, 1)), 0.02,
                      phase("STAND", "g1_native",
                            "G1 stand keyframe (src/solo/stance.py); certified anchor")))
    align_take("stand_to_stance",
               phase("LOWER_TO_STANCE", "observed",
                     "video drop into stance (27.40-28.60 s), grounded v1 retarget"))
    nh = n_hold(STANCE_HOLD_S)
    from solo.stance import measured_width_y
    raw.append(seg_of(np.tile(q_hold, (nh, 1)),
                      np.tile(np.array([1, 1], np.uint8), (nh, 1)), 0.02,
                      phase("STANCE_HOLD", "fused",
                            f"posture = video carriage + operator two-axis repair "
                            f"(width 0.34 m -> measured "
                            f"{measured_width_y(q_hold[7:]):.3f} m, height 0.74 m); "
                            f"hold duration synthetic (source has no >=1 s static "
                            f"hold; longest planted low-motion stretch 0.93 s); "
                            f"the RAW video posture is refs/stance_hold.npz with "
                            f"its own measured verdict")))
    align_take("stalk_shuffle", phase("SHUFFLE_F", "observed",
               "video stalking shuffle (122.0-131.6 s)"), cap_s=PHASE_CAP_S["SHUFFLE_F"])
    align_take("shuffle_back", phase("SHUFFLE_B", "observed",
               "video in-stance reposition (63.67-66.33 s)"))
    align_take("circle_step", phase("CIRCLE", "observed",
               "video circling (178.0-190.0 s)"), cap_s=PHASE_CAP_S["CIRCLE"])
    align_take("level_change_full", phase("LEVEL_CHANGE", "observed",
               "video level change (259.7-261.2 s)"))
    align_take("shot_entry_full",
               phase("DOUBLE_LEG_ENTRY_CROUCH", "observed",
                     "video penetration-step approach + pre-step crouch "
                     "(422.8-431.0 s); the crouch is KNOWN dynamically "
                     "infeasible -- kept, not hidden",
                     validity="known_infeasible",
                     validity_reason="pre-step crouch (pelvis ~0.49 m, pitch "
                     "60-70 deg) topples in 1.24 s under its own joint "
                     "targets (reports/2026-10-08/cem_capture.md)"),
               cut_pelvis=ENTRY_CUT_PELVIS)
    align_gm("DOUBLE_LEG",
             phase("DOUBLE_LEG_PENETRATION", "fused",
                   "penetration geometry from GrappleMap DOUBLE_LEG (G1 FK: "
                   "pelvis to 0.333 m, lead knee to 0.064 m, depth 0.585 m); "
                   "video supplies the timing of the approach only -- the "
                   "knee-down spatial structure is GrappleMap-anchored"),
             )
    align_take("shot_recover", phase("RECOVER_TO_STANCE", "observed",
               "video recovery to stance (432.5-437.0 s)"))
    align_take("shuffle_back", phase("REPOSITION", "observed",
               "second reposition shuffle"))
    align_repeat_blend(q_hold, np.array([1, 1], np.uint8))

    # ---- pass 2: interleave C1 connectors ----------------------------------
    out_segments, out_phases = [], []
    for i, seg in enumerate(raw):
        out_segments.append(seg)
        out_phases.append(seg["phase"])
        if i + 1 < len(raw):
            nxt = raw[i + 1]
            # connector length adapts to the pose gap so the blend itself
            # respects the 6 rad/s joint-velocity cap.  Measured: the Hermite
            # peak per-frame step is ~4x dist/n with moving endpoints, so the
            # length is sized for 1/4 of the cap (0.25*6*0.02 = 0.03 rad/frame
            # design step).
            maxdist = float(np.max(np.abs(raw[-1]["qpos"][-1][7:] - nxt["qpos"][0][7:])))
            n_min = int(np.ceil(maxdist / (0.25 * 6.0 * 0.02)))
            n = max(int(round(BLEND_S / 0.02)), n_min, 20)
            va, vb = vel_of(seg)[-1], vel_of(nxt)[0]
            q = blend(seg["qpos"][-1], va, nxt["qpos"][0], vb, n, 0.02)
            q = lift_roots(model, data, q)
            cseg = np.tile((seg["contact"][-1] & nxt["contact"][0])[None, :],
                           (n, 1)).astype(np.uint8)
            ph = phase(f"CONNECT_{seg['phase']['name']}->{nxt['phase']['name']}",
                       "synthetic_connector",
                       f"cubic Hermite {n * 0.02:.2f} s (adaptive, pose gap "
                       f"{maxdist:.2f} rad), C1 in position+velocity",
                       validity_reason="synthetic blend; diagnosed with the rest")
            out_segments.append(seg_of(q, cseg, 0.02, ph))
            out_phases.append(ph)

    # ---- finalise ----------------------------------------------------------
    qpos = np.concatenate([s["qpos"] for s in out_segments])
    contact = np.concatenate([s["contact"] for s in out_segments])
    phase_id = np.concatenate([np.full(len(s["qpos"]), i, np.int64)
                               for i, s in enumerate(out_segments)])
    t = np.arange(len(qpos)) * 0.02
    t0 = 0.0
    for i, (s, p) in enumerate(zip(out_segments, out_phases)):
        p["id"] = i
        p["t_start"] = round(t0, 3)
        t0 += len(s["qpos"]) * s["dt"]
        p["t_end"] = round(t0, 3)
        if p["source"] != "synthetic_connector":
            lx, rx = foot_x(model, data, s["qpos"][-1])
            p["lead_leg"] = "left" if lx >= rx else "right"
    jumps = np.abs(np.diff(qpos[:, 7:], axis=0)).max(axis=1)
    root_jumps = float(np.abs(np.diff(qpos[:, :3], axis=0)).max())
    worst_boundary = float(jumps.max())
    if worst_boundary > REFUSE_JOINT_RAD or root_jumps > REFUSE_ROOT_M:
        print(f"REFUSING to write: boundary jump joint={worst_boundary:.4f} rad "
              f"root={root_jumps:.4f} m")
        return 1
    end_reposition = out_segments[-1]["qpos"][-1]
    hold_i = next(i for i, p in enumerate(out_phases) if p["name"] == "STANCE_HOLD")
    stance_hold_start = out_segments[hold_i]["qpos"][0]
    repeat_gap = float(np.max(np.abs(end_reposition[7:] - stance_hold_start[7:])))

    meta = {
        "version": "motion_refs/v1",
        "title": "wrestling drill continuous v1",
        "dt": 0.02,
        "duration_s": float(t[-1]),
        "phases": out_phases,
        "composition": {
            "blend_s": "0.40 s minimum, adaptive to the pose gap (6 rad/s cap)",
            "blend_type": "cubic Hermite, C1 (position+velocity)",
            "mirror_threshold_m": LEAD_MIRROR_M,
            "boundary_max_joint_jump_rad": round(worst_boundary, 6),
            "boundary_max_root_jump_m": round(float(root_jumps), 6),
            "root_displacement_m": round(float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2])), 4),
            "repeat_gap_rad": round(repeat_gap, 4),
            "note": "root displacement accumulates in the world; connectors "
                    "are marked synthetic in phases[].source; REPEAT: the "
                    "final posture matches STANCE_HOLD's start within "
                    f"{repeat_gap:.3f} rad"},
        "validity_note": "validity fields are filled by "
                         "scripts/feasibility_diagnosis.py (three levels)",
        "reference_not_demonstration":
            "qpos frames are DESIRED kinematic targets; no controller action, "
            "no expert_action, no dynamic-stability claim",
        "build_provenance": {
            "takes_dir": str(REFS.relative_to(REPO)),
            "phases_from_takes": [f"{p['name']} <- {p.get('take', 'hold/blend')}"
                                  for p in out_phases
                                  if p["source"] != "synthetic_connector"],
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "drill_continuous.npz", qpos_a=qpos,
                        qpos_b=np.zeros_like(qpos), t=t,
                        technique="drill_continuous",
                        edges=np.array([], dtype=np.int64), landmark_rms=np.nan,
                        contact=contact, phase_id=phase_id,
                        meta=json.dumps(meta))
    print(f"wrote {OUT_DIR / 'drill_continuous.npz'}  dur={t[-1]:.2f}s  "
          f"phases={len(out_phases)}  worst joint jump={worst_boundary:.5f} rad  "
          f"worst root jump={root_jumps:.5f} m")

    # ---- the separate STANCE -> STAND reference ----------------------------
    q2, t2, c2, m2 = load_take("stance_to_stand")
    dt2 = float(np.median(np.diff(t2)))
    q2 = q2.copy()
    q2[:, 0] += 1.0   # own world patch (standalone reference)
    n_tail = n_hold(0.5)
    q2b = np.concatenate([q2, np.tile(q2[-1], (n_tail, 1))])
    c2b = np.concatenate([c2, np.tile(c2[-1], (n_tail, 1))])
    t2b = np.arange(len(q2b)) * dt2
    k = int(len(q2) * 0.3)
    phases2 = [
        {"id": 0, "name": "STANCE", "t_start": 0.0, "t_end": round(float(t2b[k]), 3),
         "source": "observed", "lead_leg": None, "validity": "unverified",
         "validity_reason": "", "note": "video stance posture (start of the rise)"},
        {"id": 1, "name": "RISE_TO_STAND", "t_start": round(float(t2b[k]), 3),
         "t_end": round(float(t2b[-1]), 3), "source": "observed",
         "lead_leg": None, "validity": "unverified", "validity_reason": "",
         "note": "video rise to upright (34.13-35.10 s) + 0.5 s settling hold "
                 "(synthetic)"},
    ]
    lx, rx = foot_x(model, data, q2b[-1])
    for p in phases2:
        p["lead_leg"] = "left" if lx >= rx else "right"
    meta2 = {
        "version": "motion_refs/v1", "title": "stance to stand v1",
        "dt": dt2, "duration_s": float(t2b[-1]), "phases": phases2,
        "composition": {"note": "single observed take + 0.5 s settling hold "
                                "(synthetic tail)"},
        "reference_not_demonstration":
            "qpos frames are DESIRED kinematic targets; no controller action",
    }
    np.savez_compressed(OUT_DIR / "stance_rise.npz", qpos_a=q2b,
                        qpos_b=np.zeros_like(q2b), t=t2b,
                        technique="stance_rise",
                        edges=np.array([], dtype=np.int64), landmark_rms=np.nan,
                        contact=c2b, phase_id=np.zeros(len(q2b), np.int64),
                        meta=json.dumps(meta2))
    print(f"wrote {OUT_DIR / 'stance_rise.npz'}  dur={t2b[-1]:.2f}s")
    print("phase table:")
    for p in out_phases:
        print(f"  {p['t_start']:6.2f}-{p['t_end']:6.2f}s  {p['name']:36s} "
              f"{p['source']:20s} lead={str(p['lead_leg']):4s} "
              f"validity={p['validity']}")
    print(f"REPEAT affordance: final posture vs STANCE_HOLD start = "
          f"{repeat_gap:.4f} rad max joint gap")
    return 0


if __name__ == "__main__":
    sys.exit(main())


