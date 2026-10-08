#!/usr/bin/env python3
"""Audit: (1) the stabilized teacher as a BC expert, (2) the stability
envelope of the single-G1 drill port, (3) per-source data fitness.

Assignments measured (all numbers printed + written to
``reports/2026-10-08/teacher_data_audit.json``):

A. **Teacher-as-BC-expert** — instrumented reference replays
   (``TeacherController``, paired scene, seeds 0-2) recording, per 50 Hz tick
   and per robot: the commanded 29 targets, the effective reference the
   controller tracks (raw reference + measured pose trim), the simulated
   joints, the balance offsets ``dq``, actuator saturation, rate-limit
   clipping, the posture-governor weight ``alpha``, the unclipped
   capture-point error and the live contact-based CoM margin.  Plus the
   ``run_episode`` acceptance metrics (stay-up, landmark distance) and a
   ``BaselinePD`` contrast (pure reference replay, the toppling baseline).

B. **Stability envelope** — re-runs the drill push scenario
   (``src/drill`` FeasibleDrill, the single-G1 port of the same balance law)
   at 20/35/50 N with the full trace, and analyses the on-disk drill traces
   (20 N x3 clip, 90 N failure, L2 motion).  Names the binding constraint
   with numbers.  Cadence (12.1 s/step) is decomposed from the step events.

C. **Data fitness** — reproduces the reference holdability classification
   (support envelope) for ``data/refs`` and ``data/refs_video``; measures the
   MediaPipe landmark track (rate, gaps, NaNs, planted-foot jitter, rootless
   world landmarks), the GrappleMap database (keyframe density, 1 mm
   quantisation, no timestamps), and inventories the logged traces (which
   channels exist for BC and which do not).

Self-verification: the script re-derives the published support-envelope
headline from ``data/support_envelope.json``, re-checks that every commanded
target lies in ``ctrlrange``, checks the perturbed replay is deterministic,
and checks its own class-count bookkeeping.  Any FAIL exits non-zero.

Usage (from the repo root):
    .venv/bin/python scripts/audit_teacher_data.py            # full
    .venv/bin/python scripts/audit_teacher_data.py --quick    # skip re-runs
    .venv/bin/python scripts/audit_teacher_data.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402

from grapplemap import load_graph  # noqa: E402
from retarget.scene import load_scene_model, robot_slice  # noqa: E402
from teacher import BaselinePD, TeacherController, run_episode  # noqa: E402
from teacher.controller import FOOT_Z_TOUCH  # noqa: E402
from teacher.episode import _perturb_initial, final_landmark_dist  # noqa: E402
from teacher.gains import gain_for  # noqa: E402
from teacher.posture import live_support_margin, point_in_hull_margin  # noqa: E402
from teacher.trims import trim_for  # noqa: E402

REF_DIR = REPO / "data" / "refs"
REFV_DIR = REPO / "data" / "refs_video"
DRILL_DIR = REPO / "data" / "drill"
LANDMARKS = REPO / "data" / "references" / "yt_gBAhX5t-GW4" / "pose" / "landmarks.npz"
SUPPORT_ENV = REPO / "data" / "support_envelope.json"
OUT_JSON = REPO / "reports" / "2026-10-08" / "teacher_data_audit.json"

TECHNIQUES = ("STANCE", "DOUBLE_LEG", "SINGLE_LEG", "BODY_LOCK", "SNAPDOWN",
              "SPRAWL", "STAND_UP")
BC_TECHNIQUES = ("STANCE", "SNAPDOWN", "DOUBLE_LEG")   # hold / best / target
BC_SEEDS = (0, 1, 2)

#: feasible band of the support-envelope audit (m): margin >= band = holdable
FEASIBLE_BAND = 0.02
#: saturation/rate-clip detector tolerance (rad)
EPS = 1e-9

# MediaPipe pose landmark indices used below (33-point model)
MP = {"nose": 0, "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24,
      "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
      "l_heel": 29, "r_toe": 31}


# --------------------------------------------------------------------------
# generic detectors (unit-tested; see tests/test_teacher_data_audit.py)
# --------------------------------------------------------------------------

def saturation_flags(target: np.ndarray, ctrl_lo: np.ndarray,
                     ctrl_hi: np.ndarray, tol: float = EPS) -> np.ndarray:
    """True where a commanded target sits on an actuator limit.

    A target exactly at (or numerically indistinguishable from) the limit is
    saturated; a value ``tol`` inside is not.  This is the detector used for
    the BC-expert audit.
    """
    t = np.asarray(target, float)
    return (t <= np.asarray(ctrl_lo, float) + tol) | \
           (t >= np.asarray(ctrl_hi, float) - tol)


def rate_clip_flags(dq_prev: np.ndarray, dq: np.ndarray,
                    max_rate: float, tol: float = EPS) -> np.ndarray:
    """True where the per-tick offset change sits exactly on the rate limit.

    The controller clips ``dq - dq_prev`` to ``+-max_rate``; a clipped joint
    therefore measures exactly ``max_rate`` of change, an unclipped one less.
    """
    delta = np.abs(np.asarray(dq, float) - np.asarray(dq_prev, float))
    return delta >= float(max_rate) - tol


def oppose_fraction(dq: np.ndarray, e: np.ndarray, e_min: float = 0.02) -> float:
    """Share of ticks where the balance offset opposes the tracking error.

    ``e = q_ref - q_sim`` (rad, 29).  ``dot(dq, e) < 0`` means the offset moves
    the commanded target *away* from where the reference still wants the joint
    (the stabiliser trading tracking for balance).  Ticks with ``||e||`` below
    ``e_min`` are excluded as noise (the residual is not a request).
    """
    dq = np.atleast_2d(dq)
    e = np.atleast_2d(e)
    m = np.linalg.norm(e, axis=1) >= e_min
    if not m.any():
        return float("nan")
    return float((np.einsum("ij,ij->i", dq[m], e[m]) < 0).mean())


# --------------------------------------------------------------------------
# A. teacher-as-BC-expert: instrumented replay
# --------------------------------------------------------------------------

def instrumented_rollout(model, ctrl, qa, qb, t_ref, technique: str,
                         seed: int) -> dict:
    """One 50 Hz replay of ``ctrl`` while recording every signal BC would see.

    Returns per-tick arrays and per-(robot, phase) aggregates.  The controller
    must be a ``TeacherController`` (or ``BaselinePD``); the perturbation
    matches ``run_episode`` so the metrics are directly comparable.
    """
    data = mujoco.MjData(model)
    sa, sb = robot_slice(model, "a_"), robot_slice(model, "b_")
    qa0, qb0, v0 = _perturb_initial(qa[0], qb[0], seed)
    data.qpos[sa], data.qpos[sb] = qa0, qb0
    data.qvel[:] = v0
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    spc = int(round(0.02 / dt))
    n_steps = int(math.ceil((float(t_ref[-1]) + 0.3) / dt))
    prev = {p: np.zeros(29) for p in ("a_", "b_")}
    joint_names = {
        p: [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(a))
            for a in r.act_ids] for p, r in ctrl.robots.items()}
    rec = {p: {k: [] for k in ("t", "tick", "kind", "alpha", "e_raw", "e_int",
                               "dz", "tgt", "qref", "qsim", "dq", "sat",
                               "clip", "margin", "ncon", "base", "ref_base")}
           for p in ("a_", "b_")}
    ok_range = True
    for step in range(n_steps):
        t = step * dt
        if step % spc == 0:
            c = ctrl.control(data, t)
            for p, r in ctrl.robots.items():
                kind = str(r.dbg.get("kind", "STAND"))
                g = gain_for(technique, kind)
                tgt = c[r.act_ids].copy()
                if not ((tgt >= r.ctrl_lo - 1e-9).all()
                        and (tgt <= r.ctrl_hi + 1e-9).all()):
                    ok_range = False
                qref = np.asarray(r.dbg["q_track"], float).copy()
                dq = r.last_offsets().copy()
                qsim = data.qpos[sa if p == "a_" else sb][7:36].copy()
                s = saturation_flags(tgt, r.ctrl_lo, r.ctrl_hi)
                cl = rate_clip_flags(prev[p], dq, g.max_rate)
                prev[p] = dq.copy()
                lm = live_support_margin(model, data, r.ctx)
                sl = sa if p == "a_" else sb
                rq = r.ctx.ref_qpos[r._frame(t)]
                d = rec[p]
                d["t"].append(t)
                d["kind"].append(kind)
                d["alpha"].append(float(r.dbg.get("alpha", 0.0)))
                d["e_raw"].append(float(np.hypot(*r.dbg.get("e_raw", (0, 0)))))
                d["e_int"].append(np.asarray(r.dbg.get("e_int", (0, 0)), float).copy())
                d["dz"].append(float(r.dbg.get("dz", 0.0)))
                d["tgt"].append(tgt)
                d["qref"].append(qref)
                d["qsim"].append(qsim)
                d["dq"].append(dq)
                d["sat"].append(s)
                d["clip"].append(cl)
                d["margin"].append(float(lm["margin"]))
                d["ncon"].append(int(lm["n_contact"]))
                d["base"].append(np.asarray(data.qpos[sl][:3], float).copy())
                d["ref_base"].append(np.asarray(rq[:3], float).copy())
            data.ctrl[:] = c
        mujoco.mj_step(model, data)

    out = {"range_ok": bool(ok_range), "robots": {}}
    for p, d in rec.items():
        qsim = np.array(d["qsim"])
        qref = np.array(d["qref"])
        dq = np.array(d["dq"])
        sat = np.array(d["sat"], bool)
        clip = np.array(d["clip"], bool)
        e = qref - qsim
        per_tick_rms = np.sqrt((e ** 2).mean(axis=1))          # rad
        per_phase: dict[str, dict] = {}
        kinds = np.array(d["kind"])
        for kind in sorted(set(kinds.tolist())):
            m = kinds == kind
            n = int(m.sum())
            per_phase[kind] = {
                "ticks": n,
                "sat_joint_frac": float(sat[m].mean()) if n else None,
                "sat_tick_frac": float(sat[m].any(axis=1).mean()) if n else None,
                "clip_joint_frac": float(clip[m].mean()) if n else None,
                "alpha_mean": float(np.array(d["alpha"])[m].mean()) if n else None,
                "alpha_max": float(np.array(d["alpha"])[m].max()) if n else None,
                "e_raw_mean": float(np.array(d["e_raw"])[m].mean()) if n else None,
                "rms_rad_mean": float(per_tick_rms[m].mean()) if n else None,
                "rms_rad_max": float(per_tick_rms[m].max()) if n else None,
                "dq_abs_mean": float(np.abs(dq[m]).mean()) if n else None,
            }
        out["robots"][p] = {
            "ticks": len(d["t"]),
            "sat_joint_frac": float(sat.mean()),
            "sat_tick_frac": float(sat.any(axis=1).mean()),
            "clip_joint_frac": float(clip.mean()),
            "clip_tick_frac": float(clip.any(axis=1).mean()),
            "alpha_gt05_frac": float((np.array(d["alpha"]) > 0.05).mean()),
            "alpha_max": float(np.max(d["alpha"])),
            "e_raw_mean": float(np.mean(d["e_raw"])),
            "e_raw_max": float(np.max(d["e_raw"])),
            "e_int_absmax": float(np.abs(np.array(d["e_int"])).max()),
            "dz_absmax": float(np.abs(np.array(d["dz"])).max()),
            "rms_rad_mean": float(per_tick_rms.mean()),
            "rms_rad_p95": float(np.percentile(per_tick_rms, 95)),
            "dq_abs_mean": float(np.abs(dq).mean()),
            "dq_abs_p95": float(np.percentile(np.abs(dq), 95)),
            "dq_abs_max": float(np.abs(dq).max()),
            "oppose_frac": oppose_fraction(dq, e),
            "sat_joint_counts": {joint_names[p][j]: int(sat[:, j].sum())
                                 for j in range(29) if sat[:, j].any()},
            "alpha_mean": float(np.mean(d["alpha"])),
            "margin_mean": (float(np.asarray(d["margin"], float)[
                np.isfinite(np.asarray(d["margin"], float))].mean())
                if np.isfinite(np.asarray(d["margin"], float)).any() else float("nan")),
            "margin_min": (float(np.asarray(d["margin"], float)[
                np.isfinite(np.asarray(d["margin"], float))].min())
                if np.isfinite(np.asarray(d["margin"], float)).any() else float("nan")),
            "margin_nan_frac": float(np.mean(np.isnan(d["margin"]))),
            "base_err_end_m": float(np.linalg.norm(
                np.array(d["base"][-1])[:2] - np.array(d["ref_base"][-1])[:2])),
            "base_err_max_m": float(np.max(np.linalg.norm(
                np.array(d["base"])[:, :2] - np.array(d["ref_base"])[:, :2], axis=1))),
            "yaw_err_end_rad": float(abs(math.atan2(
                math.sin(np.array(d["base"])[-1, 2] - np.array(d["ref_base"])[-1, 2]),
                math.cos(np.array(d["base"])[-1, 2] - np.array(d["ref_base"])[-1, 2])))),
            "per_phase": per_phase,
        }
    return out


def bc_teacher_audit(model) -> dict:
    """All BC-expert measurements: replays, acceptance metrics, baseline."""
    out = {"techniques": {}, "seeds": list(BC_SEEDS)}
    for tech in BC_TECHNIQUES:
        npz = np.load(REF_DIR / f"{tech}.npz", allow_pickle=True)
        qa, qb, t_ref = npz["qpos_a"], npz["qpos_b"], npz["t"]
        tech_out = {"seeds": {}, "trim": {}}
        for p in ("a_", "b_"):
            spec = trim_for(tech, p)
            tech_out["trim"][p] = {k: float(v) for k, v in spec.items()}
        for seed in BC_SEEDS:
            ctrl = TeacherController(model, qa, qb, t_ref, technique=tech)
            t0 = time.time()
            inst = instrumented_rollout(model, ctrl, qa, qb, t_ref, tech, seed)
            wall = time.time() - t0
            ctrl2 = TeacherController(model, qa, qb, t_ref, technique=tech)
            res = run_episode(model, ctrl2, qa, qb, t_ref, seed=seed)
            su = res.stay_up_frac(t_ref)
            lm = final_landmark_dist(model, res.end_qpos, qa[-1], qb[-1])
            tech_out["seeds"][seed] = {
                "wall_s": wall, "stay_up_min": su["min"], "stay_up_a": su["a"],
                "stay_up_b": su["b"], "landmark": lm,
                "instrumented": inst,
            }
        # BaselinePD contrast (pure reference replay): seed 0 for all 7
        # techniques (the "6/7 fail naive PD execution" measurement), plus the
        # 3-seed loop for the BC techniques already covered above.
        base = {"seeds": {}}
        for seed in BC_SEEDS[:1]:
            bctrl = BaselinePD(model, qa, qb, t_ref)
            bres = run_episode(model, bctrl, qa, qb, t_ref, seed=seed)
            bsu = bres.stay_up_frac(t_ref)
            base["seeds"][seed] = {"stay_up_min": bsu["min"],
                                   "stay_up_a": bsu["a"], "stay_up_b": bsu["b"]}
        tech_out["baseline_pd"] = base
        out["techniques"][tech] = tech_out
    # full-PD scan over every reference the plan could imitate
    out["pd_execution_scan"] = {}
    for tech in TECHNIQUES:
        npz = np.load(REF_DIR / f"{tech}.npz", allow_pickle=True)
        qa, qb, t_ref = npz["qpos_a"], npz["qpos_b"], npz["t"]
        bctrl = BaselinePD(model, qa, qb, t_ref)
        bres = run_episode(model, bctrl, qa, qb, t_ref, seed=0)
        su = bres.stay_up_frac(t_ref)
        out["pd_execution_scan"][tech] = {
            "stay_up_min": su["min"], "stay_up_a": su["a"], "stay_up_b": su["b"],
            "pelvis_min_a": float(bres.rows[:, 1].min()),
            "pelvis_min_b": float(bres.rows[:, 2].min()),
        }
    return out


# --------------------------------------------------------------------------
# C. data fitness: holdability of the references
# --------------------------------------------------------------------------

def _segment_margin(pts: np.ndarray, point: np.ndarray) -> float:
    """Margin convention for a 1-D support region (2 touching spheres).

    ``pts`` is (2, 2) or (K, 2); the region is the segment through two points
    (the extra points of a degenerate patch collapsed).  margin = -distance to
    the segment (<= 0), the support-envelope convention.
    """
    p = np.asarray(point, float)
    a = np.asarray(pts[0], float)
    b = np.asarray(pts[-1], float)
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
    d = float(np.linalg.norm(p - (a + t * ab)))
    return 0.0 if d == 0.0 else -d


class HoldabilityChecker:
    """The support-envelope classifier, re-implemented from the footprint hull.

    Touch rule: sole site ``z < FOOT_Z_TOUCH`` (0.045 m, the teacher's own
    threshold).  Support region: convex hull of the projected sole collision
    spheres of the touching (foot, site) pairs (4 spheres per foot, r=5 mm).
    Margin: positive inside, negative outside; holdable = ``margin >= band``.
    """

    def __init__(self, model):
        self.model = model
        self.pelvis = {p: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                            f"{p}pelvis") for p in ("a_", "b_")}
        self.sites = {p: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                            f"{p}{n}")
                          for n in ("left_toe", "left_heel", "right_toe",
                                    "right_heel")] for p in ("a_", "b_")}
        self.spheres = {p: {} for p in ("a_", "b_")}
        for p in ("a_", "b_"):
            for foot in ("left", "right"):
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                        f"{p}{foot}_ankle_roll_link")
                ids = [g for g in range(model.ngeom)
                       if model.geom_bodyid[g] == bid
                       and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                       and model.geom_contype[g]]
                assert len(ids) == 4, (p, foot, ids)
                self.spheres[p][foot] = ids
        self.data = mujoco.MjData(model)

    def frame(self, qa_row, qb_row, robot: str):
        """(margin | None, n_patch_points, min_sole_site_z) for one frame.

        ``None`` margin = AIRBORNE (no sole site inside the touch band).  One
        touching site gives a 1-D support segment, on which the margin
        convention of the support-envelope audit is ``-distance`` (<= 0).
        """
        m, d = self.model, self.data
        d.qpos[:] = 0.0
        d.qpos[0:36] = qa_row
        d.qpos[36:72] = qb_row
        mujoco.mj_forward(m, d)
        com = d.subtree_com[self.pelvis[robot]][:2].copy()
        zs = d.site_xpos[self.sites[robot]][:, 2]
        pts = []
        for k, (site, foot) in enumerate((("left_toe", "left"),
                                          ("left_heel", "left"),
                                          ("right_toe", "right"),
                                          ("right_heel", "right"))):
            if zs[k] < FOOT_Z_TOUCH:
                ids = self.spheres[robot][foot]
                use = ids[2:4] if "toe" in site else ids[0:2]
                for g in use:
                    pts.append(d.geom_xpos[g][:2].copy())
        if not pts:
            return None, 0, float(zs.min())
        if len(pts) < 3:
            return _segment_margin(np.array(pts), com), len(pts), float(zs.min())
        return point_in_hull_margin(np.array(pts), com), len(pts), float(zs.min())

    def ref(self, qa: np.ndarray, qb: np.ndarray) -> dict:
        """Per-robot holdability of a (T, 36) x (T, 36) reference."""
        res = {}
        for robot, idx in (("a_", 0), ("b_", 1)):
            n = len(qa)
            stable = marginal = infeasible = airborne = 0
            standing = standing_inf = 0
            worst = 0.0
            minz = []
            for i in range(n):
                marg, npts, mz = self.frame(qa[i], qb[i], robot)
                minz.append(mz)
                pz = float((qa if idx == 0 else qb)[i][2])
                if marg is None:
                    airborne += 1
                    continue
                if pz >= 0.45:
                    standing += 1
                    if marg < 0.0:
                        standing_inf += 1
                worst = min(worst, marg)
                if marg >= FEASIBLE_BAND:
                    stable += 1
                elif marg >= 0.0:
                    marginal += 1
                else:
                    infeasible += 1
            assert stable + marginal + infeasible + airborne == n, (robot, n)
            minz = np.array(minz)
            res[robot] = {"frames": n, "stable": stable, "marginal": marginal,
                          "infeasible": infeasible, "airborne": airborne,
                          "inside_support": stable + marginal,
                          "inside_frac": (stable + marginal) / max(n, 1),
                          "standing": standing, "standing_infeasible": standing_inf,
                          "worst_margin": worst,
                          "min_sole_z_median": float(np.median(minz)),
                          "min_sole_z_p10": float(np.percentile(minz, 10)),
                          "min_sole_z_max": float(minz.max())}
        return res


def holdability_audit(model) -> dict:
    chk = HoldabilityChecker(model)
    out = {"refs": {}, "refs_video": {}, "aggregate": {}}
    tot = stable_tot = air_tot = 0
    for tech in TECHNIQUES:
        npz = np.load(REF_DIR / f"{tech}.npz", allow_pickle=True)
        qa, qb = npz["qpos_a"], npz["qpos_b"]
        out["refs"][tech] = chk.ref(qa, qb)
        for r in out["refs"][tech].values():
            tot += r["frames"]
            air_tot += r["airborne"]
            stable_tot += r["inside_support"]
    out["aggregate"] = {
        "robot_frames": tot, "airborne": air_tot,
        "inside_support": stable_tot,
        "inside_pct_of_all": 100.0 * stable_tot / max(tot, 1),
        "inside_pct_of_grounded": 100.0 * stable_tot / max(tot - air_tot, 1),
    }
    for f in sorted(REFV_DIR.glob("*.npz")):
        npz = np.load(f, allow_pickle=True)
        qa = npz["qpos_a"]
        qb = npz["qpos_b"] if "qpos_b" in npz.files else np.zeros_like(qa)
        meta = {}
        if "landmark_rms" in npz.files:
            meta["landmark_rms"] = float(npz["landmark_rms"])
        if "technique" in npz.files:
            meta["technique"] = str(npz["technique"])
        out["refs_video"][f.stem] = {
            "frames": int(len(qa)),
            "duration_s": (float(npz["t"][-1] - npz["t"][0])
                           if "t" in npz.files else None),
            "b_zero_frac": float((np.abs(qb).max(axis=1) < 1e-9).mean()),
            "holdability": chk.ref(qa, qb),
            "meta": meta,
        }
    return out


# --------------------------------------------------------------------------
# C. data fitness: MediaPipe landmarks, GrappleMap, trace inventory
# --------------------------------------------------------------------------

def mediapipe_audit() -> dict:
    z = np.load(LANDMARKS, allow_pickle=True)
    t = np.asarray(z["t"], float)
    lm = np.asarray(z["world_lm"], float)
    dt = np.diff(t)
    med = float(np.median(dt))
    gaps = np.where(dt > 1.5 * med)[0]
    nan_frames = int(np.isnan(lm).all(axis=(1, 2)).sum())
    # rootless: world landmarks are hip-centred, so the hip midpoint cannot
    # move by construction; measure it as the demonstration.
    hips = lm[:, [MP["l_hip"], MP["r_hip"]], :]
    hips = hips[~np.isnan(hips).any(axis=(1, 2))]
    hip_std = np.std(hips.mean(axis=1), axis=0)
    # planted-foot jitter: the stance_hold source window (both feet planted)
    win = (t >= 30.4) & (t <= 34.8)
    win = win & ~np.isnan(lm[:, MP["l_ankle"], 0])

    def hf_p2p(x, w=5):
        k = np.ones(w) / w
        ma = np.convolve(np.pad(x, (w // 2, w // 2), mode="edge"), k, "valid")
        h = x - ma
        return float(np.ptp(h))

    jitter = {}
    for name, idx in (("l_ankle", MP["l_ankle"]), ("r_ankle", MP["r_ankle"]),
                      ("l_knee", MP["l_knee"]), ("l_hip", MP["l_hip"]),
                      ("l_wrist", MP["l_wrist"]), ("r_wrist", MP["r_wrist"]),
                      ("nose", MP["nose"])):
        seg = lm[win, idx, :]
        seg = seg[~np.isnan(seg).any(axis=1)]
        jitter[name] = {
            "hf_std_mm": [round(float(np.std(seg[:, j] -
                                              np.convolve(np.pad(
                                                  seg[:, j], (2, 2), mode="edge"),
                                                  np.ones(5) / 5, "valid"))) * 1000, 2)
                          for j in range(3)],
            "hf_p2p_mm": [round(hf_p2p(seg[:, j]) * 1000, 1) for j in range(3)],
            "samples": int(len(seg)),
        }
    return {
        "frames": int(len(t)), "t_span_s": [float(t[0]), float(t[-1])],
        "fps_median_hz": round(1.0 / med, 3),
        "dt_median_s": med, "dt_max_s": float(dt.max()), "gaps": int(len(gaps)),
        "detected_frac": float(np.mean(z["detected"])),
        "nan_all_frames": nan_frames,
        "hip_mid_std_mm": [round(float(v) * 1000, 3) for v in hip_std],
        "planted_jitter": jitter,
        "channels": sorted(z.files),
        "has_root_translation": False,   # no global translation channel exists
        "has_contact_forces": False,     # not a MediaPipe output; not logged
        "has_joint_angles": False,       # 33 landmarks only
    }


def grapplemap_audit() -> dict:
    g = load_graph()
    frames = np.array([e.frame_count for e in g.edges])
    coords = np.concatenate([e.frames[:, :, :, 0].ravel() for e in g.edges])
    u = np.unique(coords)
    d = np.diff(u)
    d = d[d > 1e-12]
    on_grid = np.abs(coords * 1000 - np.round(coords * 1000))
    return {
        "nodes": int(g.num_nodes), "edges": int(len(g.edges)),
        "landsmarks_per_player": 23, "players": 2,
        "frames_per_edge": {"min": int(frames.min()),
                            "median": float(np.median(frames)),
                            "mean": float(frames.mean()),
                            "max": int(frames.max())},
        "min_coordinate_step_m": float(d.min()),
        "frac_coords_on_1mm_grid": float((on_grid < 1e-6).mean()),
        "detailed_edge_frac": float(np.mean(
            [("detailed" in e.properties) for e in g.edges])),
        "has_timestamps": False,   # Edge has no time field (parser.py)
        "nominal_rate_hz": {"keyframes": 5.0, "detailed_edges": 10.0},
    }


def trace_inventory() -> dict:
    """What the logged traces actually contain (for BC/RL reuse)."""
    out = {}
    ts = json.loads((REPO / "data" / "teacher_stats.json").read_text())
    out["teacher_stats.json"] = {
        "keys": sorted(ts.keys()),
        "techniques": sorted(ts["techniques"].keys()),
        "per_technique_keys": sorted(ts["techniques"][next(iter(ts["techniques"]))].keys()),
        "has_per_tick_actions": False,
        "has_per_tick_states": False,
        "note": "aggregate metrics only (stay_up/scorer/penetration/landmark)",
    }
    drill = DRILL_DIR / "FINAL_L1_push_feasible_L1_seed0.npz"
    z = np.load(drill, allow_pickle=True)
    out["drill_npz"] = {
        "example": drill.name,
        "channels": sorted(z.files),
        "has_actions": "ctrl" in z.files,
        "has_margin": "margin" in z.files,
        "has_governor": "safety_alpha" in z.files,
        "has_saturation": "sat_frac" in z.files,
        "has_per_step_events": False,
        "note": ("50 Hz ctrl + balance signals, but produced by the drill "
                 "controller (src/drill), not by RobotTeacher rollouts"),
    }
    # what run_episode captures for a teacher rollout
    out["teacher_run_episode"] = {
        "captures": ["rows(t,z,tilt,penetration)", "end_qpos", "traj(qpos)",
                     "ref_landmark track", "base_x", "video frames"],
        "has_actions": False,
        "note": ("EpisodeResult.traj stores qpos only; the commanded ctrl "
                 "sequence is not recorded (re-instrumentation required for "
                 "BC datasets)"),
    }
    return out


# --------------------------------------------------------------------------
# B. stability envelope
# --------------------------------------------------------------------------

def _series(res_or_trace):
    tr = getattr(res_or_trace, "trace", res_or_trace)
    return {k: np.asarray(tr[k], float) if k != "meta" else tr[k]
            for k in tr}


def analyse_drill_trace(z, push_window_s: float = 4.0) -> dict:
    """Envelope signals of one drill trace (npz open file)."""
    t = np.asarray(z["t"], float)
    ctrl = np.asarray(z["ctrl"], float)
    margin = np.asarray(z["margin"], float)
    sat = np.asarray(z["sat_frac"], float)
    alpha = np.asarray(z["safety_alpha"], float)
    e = np.asarray(z["e_track"], float)
    push = np.asarray(z["push_force"], float)
    zq = np.asarray(z["qpos"], float)[:, 2]
    pen = np.asarray(z["pen_min"], float)
    out = {"ticks": int(len(t)), "duration_s": float(t[-1]),
           "margin_min": float(margin.min()), "margin_min_t": float(t[margin.argmin()]),
           "pelvis_min": float(zq.min()),
           "sat_frac_max": float(sat.max()), "sat_frac_mean": float(sat.mean()),
           "alpha_max": float(alpha.max()),
           "alpha_gt05_frac": float((alpha > 0.05).mean()),
           "e_track_max": float(np.hypot(e[:, 0], e[:, 1]).max()),
           "ctrl_delta_max_rad": float(np.abs(np.diff(ctrl, axis=0)).max()),
           "pen_min": float(pen.min()),
           "fall": bool(margin.min() < -0.5 or zq.min() < 0.3)}
    if (push > 1.0).any():
        t0, t1 = float(t[push > 1.0].min()), float(t[push > 1.0].max())
        w = (t >= t0 - 0.1) & (t <= t1 + push_window_s)
        out["push"] = {"t0": t0, "t1": t1, "label_force": float(push.max()),
                       "margin_min_in_window": float(margin[w].min()),
                       "sat_max_in_window": float(sat[w].max()),
                       "alpha_max_in_window": float(alpha[w].max()),
                       "e_track_max_in_window": float(np.hypot(e[w, 0], e[w, 1]).max()),
                       "ctrl_delta_max_in_window": float(np.abs(np.diff(ctrl, axis=0))[w[:-1]].max())}
    return out


def drill_envelope(model, quick: bool = False) -> dict:
    """Push battery (re-run short) + on-disk traces + cadence decomposition."""
    out = {"rerun": {}, "on_disk": {}, "cadence": {}}
    if not quick:
        from drill.runner import PushSpec, RunConfig, run as drill_run
        for mag in (-20.0, -35.0, -50.0):
            cfg = RunConfig(controller="feasible", rung="L1", seconds=12.0,
                            seed=0, start="stance",
                            pushes=[PushSpec(t=6.0, dur=0.12, fx=mag,
                                             label=f"push{mag:.0f}N")],
                            tag=f"AUDIT_push{abs(mag):.0f}", max_wall_s=600)
            t0 = time.time()
            res = drill_run(cfg, verbose=False)
            a = analyse_drill_trace(res.trace)
            a["wall_s"] = time.time() - t0
            a["recovered"] = bool(res.metrics.get("pushes", {}).get("recovered", 0))
            out["rerun"][f"{mag:.0f}N"] = a
    for name, path in (("L1_push_20N_x3", "FINAL_L1_push_feasible_L1_seed0.npz"),
                       ("PUSH90_fail", "FAIL_PUSH90_feasible_L1_seed0.npz"),
                       ("PUSH35_clip", "FAIL_PUSH35_feasible_L1_seed0.npz"),
                       ("L2_motion", "L2_MOTION_feasible_L3_seed0.npz")):
        p = DRILL_DIR / path
        if not p.exists():
            continue
        with np.load(p, allow_pickle=True) as z:
            out["on_disk"][name] = analyse_drill_trace(z)
    # cadence decomposition: the summary (5 steps, 0 falls, 70 s) + the step
    # event timeline of a comparable run (M_w28v).
    ms = json.loads((DRILL_DIR / "motion_singles.json").read_text())
    if ms:
        key = next(iter(ms))
        s = ms[key]
        out["cadence"]["summary"] = {
            "run": key, "cadence_s_per_step": s.get("cadence_s_per_step"),
            "step_gaps_s": s.get("step_gaps_s"), "steps": s.get("steps"),
            "falls": s.get("falls"), "duration_s": s.get("duration_s"),
            "margin_min": s.get("margin_min"), "margin_p05": s.get("margin_p05"),
            "saturation_frac": s.get("saturation_frac"),
            "safety_blend_frac": s.get("safety_blend_frac"),
            "required_m": s.get("required_m"),
        }
    ev_path = DRILL_DIR / "M_w28v_feasible_L3_seed0.json"
    if ev_path.exists():
        ev = json.loads(ev_path.read_text())["events"]
        by = {}
        for e in ev:
            by.setdefault(e["event"], []).append(e)
        shifts = [e for e in ev if e["event"] == "shift_done"]
        steps = [e for e in ev if e["event"] == "step_done"]
        starts = [e for e in ev if e["event"] == "step_start"]
        out["cadence"]["events"] = {
            "run": ev_path.stem,
            "step_start_t": [round(e["t"], 2) for e in starts],
            "shift_done_t": [round(e["t"], 2) for e in shifts],
            "step_done_t": [round(e["t"], 2) for e in steps],
            "required_com_travel_m": [e.get("required_com_travel_m") for e in starts],
            "shift_com_travel_m": [e.get("com_travel_m") for e in shifts],
            "shift_duration_s": [round(s["t"] - st["t"], 2)
                                 for st, s in zip(starts, shifts)],
            "v_gate_m_s": [e.get("v_gate_m_s") for e in shifts],
            "margin_support_at_gate": [e.get("margin_support") for e in shifts],
        }
    return out


# --------------------------------------------------------------------------
# self-verification + report printing
# --------------------------------------------------------------------------

def self_check(model, bc: dict, hold: dict, mp: dict, gm: dict,
               drill: dict, verbose: bool = True) -> list:
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" +
                  (f" — {detail}" if detail else ""))

    # 1. every commanded target inside ctrlrange over every instrumented tick
    ok = all(inst["range_ok"]
             for tech in bc["techniques"].values()
             for s in tech["seeds"].values()
             for inst in [s["instrumented"]])
    chk("teacher targets always in ctrlrange", ok)

    # 2. holdability bookkeeping sums to frame counts + matches prior audit
    ok = all(r["stable"] + r["marginal"] + r["infeasible"] + r["airborne"]
             == r["frames"]
             for tech in hold["refs"].values() for r in tech.values())
    chk("holdability class counts sum to frames", ok)
    if SUPPORT_ENV.exists():
        se = json.loads(SUPPORT_ENV.read_text())
        ref_pct = se["headline"]["feasible_pct_aggregate"]
        got = hold["aggregate"]["inside_pct_of_all"]
        chk("reproduces support-envelope aggregate", abs(got - ref_pct) < 0.05,
            f"got {got:.3f}% vs published {ref_pct:.3f}%")
        per_ok = True
        details = []
        for tech, robots in se["techniques"].items():
            for r in ("a", "b"):
                pub = robots["robots"][r]["class_counts"]
                got_c = hold["refs"][tech][r + "_"]
                if (pub["STABLY_FEASIBLE"] != got_c["stable"]
                        or pub["MARGINAL"] != got_c["marginal"]
                        or pub["AIRBORNE"] != got_c["airborne"]):
                    per_ok = False
                    details.append(f"{tech}/{r}")
        chk("per-technique class counts match the published audit", per_ok,
            ",".join(details) if details else "all 14 robot-tracks")

    # 3. landmark rate
    chk("landmark track is 15 fps", abs(mp["fps_median_hz"] - 15.0) < 0.05,
        f"{mp['fps_median_hz']} Hz")

    # 4. PD baseline reproduces the published phase-2 ablation (seed 0)
    published_pd = {"STANCE": 0.208, "SNAPDOWN": 0.778, "DOUBLE_LEG": 0.444}
    ok = True
    det = []
    for tech, want in published_pd.items():
        got = bc["techniques"][tech]["baseline_pd"]["seeds"][0]["stay_up_min"]
        if abs(got - want) > 0.02:
            ok = False
            det.append(f"{tech} {got:.3f}!={want:.3f}")
    chk("PD baseline matches the published ablation (seed 0)", ok,
        ",".join(det) if det else "STANCE/SNAPDOWN/DOUBLE_LEG")

    # 5. GrappleMap 1 mm quantisation
    chk("GrappleMap 1 mm quantisation", abs(gm["min_coordinate_step_m"] - 1e-3) < 1e-9
        and gm["frac_coords_on_1mm_grid"] > 0.999,
        f"step {gm['min_coordinate_step_m']*1000:.3f} mm")

    # 6. drill reruns: recorded margins consistent with recovery flags
    if drill.get("rerun"):
        ok = True
        det = []
        for k, v in drill["rerun"].items():
            consistent = (v["recovered"] and not v["fall"]) or (not v["recovered"] and v["fall"])
            if not consistent and not v["recovered"]:
                # a run that did not recover without falling is suspicious
                ok = False
                det.append(k)
            if v["sat_frac_max"] < -0.0 or v["sat_frac_max"] > 1.0:
                ok = False
                det.append(k + " sat")
        chk("drill push reruns self-consistent", ok, ",".join(det) if det else "3 magnitudes")
    return checks


def print_report(bc: dict, hold: dict, mp: dict, gm: dict, drill: dict,
                 inv: dict) -> None:
    print("\n=== A. TEACHER AS BC EXPERT (paired scene, seeds 0-2) ===")
    for tech, t in bc["techniques"].items():
        for seed, s in t["seeds"].items():
            ia, ib = s["instrumented"]["robots"]["a_"], s["instrumented"]["robots"]["b_"]
            print(f"{tech:11s} seed{seed} stay_up_min {s['stay_up_min']:.3f} "
                  f"(PD {t['baseline_pd']['seeds'].get(seed, {}).get('stay_up_min', float('nan')):.3f}) "
                  f"landmark {s['landmark']['max']:.3f} m")
            for p, r in (("a", ia), ("b", ib)):
                satj = ",".join(f"{k}:{v}" for k, v in
                                sorted(r.get("sat_joint_counts", {}).items(),
                                       key=lambda kv: -kv[1])[:3])
                print(f"   {p}: track_rms {r['rms_rad_mean']*57.2958:5.2f} deg "
                      f"(p95 {r['rms_rad_p95']*57.2958:5.2f})  sat {100*r['sat_tick_frac']:5.2f}% ticks "
                      f"{'[' + satj + ']' if satj else ''} clip {100*r['clip_tick_frac']:5.2f}%  "
                      f"alpha_mean {r['alpha_mean']:.2f} (>0.05 on {100*r['alpha_gt05_frac']:.1f}%) "
                      f"|dq|p95 {r['dq_abs_p95']:.3f} rad  oppose {r['oppose_frac']*100:4.1f}% "
                      f"margin_min {r['margin_min']:+.3f} m")
        tr = t["trim"]
        if any(tr[p] for p in tr):
            worst = max((abs(v), k, p) for p, d in tr.items() for k, v in d.items())
            print(f"   trim {tech}: max |trim| {worst[0]:.3f} rad ({math.degrees(worst[0]):.1f} deg) "
                  f"at {worst[1]} ({worst[2]})")
    if bc.get("pd_execution_scan"):
        print("naive PD replay, stay_up_min (seed 0): " + "  ".join(
            f"{k} {v['stay_up_min']:.3f}"
            for k, v in bc["pd_execution_scan"].items()))

    print("\n=== B. STABILITY ENVELOPE (drill port) ===")
    for k, v in (drill.get("rerun") or {}).items():
        print(f"rerun {k:>6s}: recovered {v['recovered']} fall {v['fall']} "
              f"margin_min {v['margin_min']:+.4f} sat_max {v['sat_frac_max']:.3f} "
              f"alpha_max {v['alpha_max']:.2f} ctrl_delta_max {v['ctrl_delta_max_rad']:.4f} rad")
    for k, v in drill["on_disk"].items():
        print(f"trace {k:14s}: fall {v['fall']} margin_min {v['margin_min']:+.4f} "
              f"sat_max {v['sat_frac_max']:.3f} alpha_max {v['alpha_max']:.2f}")
    cad = drill["cadence"]
    if cad.get("summary"):
        s = cad["summary"]
        print(f"cadence: {s['cadence_s_per_step']} s/step over {s['steps']} steps, "
              f"falls {s['falls']}, gaps {s['step_gaps_s']}, sat {s['saturation_frac']}")
    if cad.get("events"):
        ev = cad["events"]
        print(f"step durations (M_w28v): shift {ev['shift_duration_s']} s "
              f"for com_travel {ev['shift_com_travel_m']} m "
              f"(required {ev['required_com_travel_m']} m)")

    print("\n=== C. DATA FITNESS ===")
    agg = hold["aggregate"]
    print(f"refs: {agg['inside_support']}/{agg['robot_frames']} grounded frames inside support "
          f"= {agg['inside_pct_of_all']:.2f}% of all, {agg['inside_pct_of_grounded']:.2f}% of grounded")
    for tech, robots in hold["refs"].items():
        a = robots["a_"]
        print(f"   {tech:11s} a: stable {a['stable']:3d} marginal {a['marginal']:3d} "
              f"inf {a['infeasible']:3d} air {a['airborne']:3d}  "
              f"b: stable {robots['b_']['stable']:3d} inf {robots['b_']['infeasible']:3d} "
              f"air {robots['b_']['airborne']:3d}")
    for f, v in hold["refs_video"].items():
        a = v["holdability"]["a_"]
        print(f"refs_video {f:22s}: {v['frames']:4d} f, stable {a['stable']:3d} "
              f"inf {a['infeasible']:4d} air {a['airborne']:4d}, "
              f"min-sole-z med {a['min_sole_z_median']*1000:6.1f} mm, "
              f"b-zero {v['b_zero_frac']:.2f}")
    print(f"landmarks: {mp['frames']} frames @ {mp['fps_median_hz']} Hz, "
          f"detected {mp['detected_frac']:.3f}, all-NaN frames {mp['nan_all_frames']}, "
          f"max dt {mp['dt_max_s']:.0f} s")
    print(f"   planted-window hf p2p (mm) " +
          " ".join(f"{k}:{v['hf_p2p_mm']}" for k, v in mp["planted_jitter"].items()))
    print(f"grapplemap: {gm['nodes']} nodes, {gm['edges']} edges, "
          f"frames/edge median {gm['frames_per_edge']['median']:.0f} "
          f"max {gm['frames_per_edge']['max']}, min coord step "
          f"{gm['min_coordinate_step_m']*1000:.3f} mm")
    inv_ts = inv["teacher_stats.json"]
    print(f"trace inventory: teacher_stats has per-tick actions? "
          f"{inv_ts['has_per_tick_actions']}; drill npz actions? "
          f"{inv['drill_npz']['has_actions']}; run_episode actions? "
          f"{inv['teacher_run_episode']['has_actions']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="skip the drill push re-runs (use on-disk traces only)")
    ap.add_argument("--json", type=Path, default=OUT_JSON)
    args = ap.parse_args()

    t0 = time.time()
    model = load_scene_model()
    print(f"[{time.time()-t0:6.1f}s] loaded paired scene "
          f"({model.nq} qpos, {model.nu} actuators)")

    print(f"[{time.time()-t0:6.1f}s] BC-expert instrumented replays "
          f"{BC_TECHNIQUES} seeds {BC_SEEDS} ...")
    bc = bc_teacher_audit(model)
    print(f"[{time.time()-t0:6.1f}s] reference holdability scan ...")
    hold = holdability_audit(model)
    print(f"[{time.time()-t0:6.1f}s] MediaPipe / GrappleMap / trace inventory ...")
    mp = mediapipe_audit()
    gm = grapplemap_audit()
    inv = trace_inventory()
    print(f"[{time.time()-t0:6.1f}s] drill envelope ...")
    drill = drill_envelope(model, quick=args.quick)

    payload = {"generated_by": "scripts/audit_teacher_data.py",
               "date": "2026-10-08", "quick": bool(args.quick),
               "bc_teacher": bc, "holdability": hold, "mediapipe": mp,
               "grapplemap": gm, "drill_envelope": drill,
               "trace_inventory": inv}

    print("\n=== SELF-CHECKS ===")
    checks = self_check(model, bc, hold, mp, gm, drill)
    payload["self_checks"] = checks
    print_report(bc, hold, mp, gm, drill, inv)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=1, default=str))
    n_fail = sum(1 for c in checks if not c["ok"])
    print(f"\nwrote {args.json} in {time.time()-t0:.1f}s wall "
          f"({n_fail} failed self-checks)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
