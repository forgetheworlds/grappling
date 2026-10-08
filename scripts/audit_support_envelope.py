#!/usr/bin/env python
"""Static support-envelope audit of the retargeted GrappleMap references (uncertainty U3).

Question answered: how much of the retargeted GrappleMap geometry is *statically*
infeasible for the G1 (centre of mass outside the foot support region), and how much
ankle torque would be needed to hold it?  This quantifies the already-measured STANCE
finding (CoM behind the support, ~39 Nm of the 50 Nm ankle limit as it tips) across all
seven references and says how much reference adjustment M4/M5 will need.

Method (every choice stated here and again in reports/2026-10-08/support_envelope.md):

1. KINEMATIC replay.  For every technique in ``data/refs/<TECH>.npz`` and every frame we
   write the reference ``qpos_a``/``qpos_b`` into ``robots/wrestling_scene.xml`` and call
   ``mujoco.mj_forward``.  No dynamics, no PD servos, no contact solver: this measures the
   *intended geometry* the reference encodes, not the drift of a particular controller.
   ``mj_forward`` (not ``mj_kinematics``) because we need ``subtree_com``.
2. Per frame, per robot:
   * CoM = ``data.subtree_com[pelvis body]`` (the whole robot; the pelvis roots each
     robot's kinematic subtree).  Horizontal position = xy part.
   * Foot contact uses the *same rule and sites as the teacher*
     (``src/teacher/controller.py:65-66,110``): a sole site counts as touching when
     ``site_z < FOOT_Z_TOUCH = 0.045 m``.  Sites: ``{left,right}_{toe,heel}``.  The
     0.045 m value is the teacher's phase-label threshold, not a physics contact test;
     planted reference sole sites sit in [0.000, 0.020] m, so 0.045 m admits the whole
     planted band (per-technique sole-height percentiles are in the JSON).
   * Support region, three ways:
       (a) ``hull_site``  - convex hull of the touching sole *sites* (the literal
           request).  Degenerate by construction: the toe/heel sites lie on the foot
           midline, so a flat single foot gives a 1-D region.
       (b) ``hull_foot``  - convex hull of the world-projected sole contact *patches* of
           the touching (foot, site) pairs.  A patch = the pair of sole collision spheres
           the scene already defines (``robots/g1/g1.xml:108-115``: r = 5 mm, heel local
           x = -0.05 y = +-0.025, toe x = +0.12 y = +-0.03) projected to xy.  For point
           contacts the CoP region *is* the convex hull of the contact points, so this is
           the physical support region and is the PRIMARY CLASSIFIER.
       (c) ``line_toeheel`` - union over touching feet of the heel->toe segment (the
           requested simpler representation); a 1-D region, reported as the distance from
           the CoM to that union.
     Margin sign convention: distance to the region boundary, positive inside, negative
     outside.  For 1-D regions the margin is always <= 0 and the magnitude is the distance
     to the nearest point of the segment.
3. Class per frame (from the primary region): STABLY_FEASIBLE (margin >= BAND),
   MARGINAL (0 <= margin < BAND), INFEASIBLE (margin < 0), AIRBORNE (no touching sole
   site - a separate class; a foot-only criterion does not apply).  BAND = 0.02 m, the
   same order as the retargeter's accepted ground-penetration tolerance (<= 0.011 m) and
   small against the 0.175 m foot length; sensitivity at 0.01/0.05 m is in the JSON.
   The contact state is reported as one / both / airborne_ground (nothing on the feet but
   the lowest body site is within 0.05 m of the mat - the body is supported by
   knees/hands/torso) / airborne_float (nothing at all is within 0.05 m: the reference
   floats, a placement defect or intended flight).  Frames whose *reference pelvis*
   (qpos[2]) is below 0.45 m (the phase-2 stay-up gate) are flagged ``ground``: the body
   is then supported by knees/hands/torso and the foot-only criterion is the wrong model.
   Class counts are reported for all frames, the standing subset and the ground subset.
4. Static ankle-torque demand, two estimators:
   * ``tau_pendulum_nm`` (the requested rough model): inverted pendulum about the ankle
     axis, ``tau = m*g*d``, ``d`` = horizontal distance from the CoM to the supporting
     foot's ankle axis (single support) or to the midpoint of the two ankle axes (double
     support).  ``m = body_subtreemass[pelvis] = 33.34 kg``, ``g = 9.81``.  Exact for
     single support; for double support it is the rigid-pendulum equivalent and the true
     demand depends on how the load is shared between the feet.
   * ``tau_minmax_nm`` (load-sharing-aware): the smallest possible *worst-ankle* moment
     over every static allocation of the body weight to the touching feet and every CoP
     placement inside that foot's patch.  This is the tiny LP
         min t  s.t.  sum_i N_i = m g,  sum_i N_i CoP_i = m g * CoM_xy,
                      CoP_i in patch_i,  t >= |N_i (CoP_i - ankle_i)|
     solved with scipy.optimize.linprog over the convex-cone representation
     (CoP_i = sum_j lam_ij v_ij / sum_j lam_ij) with the norm constraint discretised over
     12 unit directions (~3.4% under-estimate).  It is infeasible exactly when the CoM is
     outside the contact hull, so the LP doubles as an independent check of the geometric
     class (verified in the self-check).  The 50 Nm ``actuatorfrcrange`` of the ankle
     joints (``robots/g1/g1.xml:100,106,146,152``) is compared against this number.
   Limits: both estimators ignore dynamics (inertia, CoM velocity, capture point),
   external grappling load, torso/arm inertia, compliant mesh contact, foot slip, and
   actuator rate limits.  ``tau_pendulum`` additionally ignores foot-load indeterminacy.
   A third, deliberately naive bound ``tau_naive_one_ankle_nm = m g max_i |CoM-ankle_i|``
   (one ankle carries the whole moment) is stored in the JSON only, to show why the naive
   metric must not be used in a stagger stance.
5. Per-technique per-robot summary: class fractions (all / standing / ground), support
   states, worst lever arm, worst ``tau_pendulum`` and its frame/time, the fraction of
   frames whose *minimum* static worst-ankle demand already exceeds 50 Nm, and the
   INFEASIBLE episodes (start/end frame index and time; runs separated by <= 2 frames are
   merged, larger gaps split an episode).

Outputs: prints the table, writes ``data/support_envelope.json`` and the figure
``reports/2026-10-08/support_envelope.png``.  Runnable from the repo root:

    .venv/bin/python scripts/audit_support_envelope.py

6. Reference-integrity check (same replay, added 2026-10-08 after the motor audit found
   STAND_UP's reference airborne; decides regeneration vs stabiliser):
   * contact availability - per foot and per robot min sole-site height, frames with no
     foot within 0.02 m of the mat, and the longest contact-free episode measured on the
     *lowest body site* (float: nothing within 0.05 m);
   * placement - the deepest sole penetration below the mat;
   * montage seams - the frame-to-frame max joint delta, base translation, base spin and
     sole-site jump, compared against the retimer's own limits (6 rad/s / 3 m/s / 8 rad/s
     -> 0.12 rad / 0.06 m / 0.16 rad per 0.02 s frame) plus an 0.08 m/frame foot-jump
     limit; exceeding a limit inside a segment is impossible, so such frames are seam
     artifacts;
   * contact-transition sanity - frames with one foot planted whose CoM is more than
     0.05 m outside that foot's support;
   * verdict per technique: REF_INVALID/REGENERATE (sustained float, deep penetration or
     a >2x discontinuity), REF_VALID_ADJUSTABLE/ADJUST (a seam artifact, or a standing
     deficit <= 0.05 m), REF_VALID_DYNAMIC/STABILISE (contact-consistent and continuous
     with a large transient CoM excursion; ``control_limit`` flag when the statics and
     the 50 Nm ankle demand are within range).
   The pairwise CoM-CoM distance is also recorded: in clinch frames the opponent is a
   second support path, so the foot-only criterion is a lower bound on support.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import mujoco
from scipy.optimize import linprog

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(REPO, "robots", "wrestling_scene.xml")
REF_DIR = os.path.join(REPO, "data", "refs")
OUT_JSON = os.path.join(REPO, "data", "support_envelope.json")
OUT_PNG = os.path.join(REPO, "reports", "2026-10-08", "support_envelope.png")

TECHS = ["STANCE", "DOUBLE_LEG", "SINGLE_LEG", "BODY_LOCK", "SNAPDOWN", "SPRAWL",
         "STAND_UP"]

# --- method constants ---------------------------------------------------------------
FOOT_Z_TOUCH = 0.045          # src/teacher/controller.py:66 (teacher's contact rule)
BAND = 0.02                   # marginal band (m); sensitivity 0.01/0.05 in the JSON
BAND_SENSITIVITY = [0.01, 0.02, 0.05]
G = 9.81
STAND_PELVIS_Z = 0.45         # phase-2 stay-up gate; below this a frame is "ground"
MERGE_GAP = 2                 # frames (0.04 s) tolerated inside one INFEASIBLE episode
ANKLE_LIMIT_NM = 50.0         # actuatorfrcrange of both ankle joints, g1.xml
LP_DIRS = 12                  # directions for the discretised norm in the min-max LP
AIRBORNE_GROUND_Z = 0.05      # lowest body site below this = body still on the mat
FOOT_Z_NEAR = 0.02            # tight "foot is on the mat" threshold (1-2 cm)
# Retime limits enforced inside each montage segment (retarget.md section 3):
# joint vel <= 6 rad/s, base speed <= 3 m/s, base spin <= 8 rad/s at 50 Hz.
SEAM_LIMITS = {"dq_rad": 0.12, "dp_m": 0.06, "drot_rad": 0.16}
SEAM_JUMP_FACTOR = 2.0        # a jump of 2x the limit is a positional discontinuity
FOOT_TELEPORT_M = 0.08        # sole-site jump per frame that no 6 rad/s joint can make
PENETRATION_REFUSE_M = 0.05   # sole this far below the mat: reference is not placeable
FLOAT_SUSPECT_S = 0.40        # a float episode this long with a raised body is a defect
FLOAT_MEDIAN_Z = 0.05

SITES = ("left_toe", "left_heel", "right_toe", "right_heel")
FOOT_SIDES = ("left", "right")

# Role of robot a per technique (reports/2026-10-07/retarget.md section 4; the npz meta
# string is generic).  Robot b is the counterpart.
ROLES = {
    "DOUBLE_LEG": "shooter/attacker",
    "SINGLE_LEG": "shooter/attacker",
    "BODY_LOCK": "flanker/mover",
    "SNAPDOWN": "snapper",
    "SPRAWL": "attacker (shot chain)",
    "STAND_UP": "recoverer (bottom)",
    "STANCE": "symmetric start pose",
}
ROLE_B = {
    "DOUBLE_LEG": "defender/opponent", "SINGLE_LEG": "defender/opponent",
    "BODY_LOCK": "opponent", "SNAPDOWN": "snapped opponent",
    "SPRAWL": "defender (ends prone)", "STAND_UP": "opponent",
    "STANCE": "symmetric start pose",
}

CLASS_ORDER = ["STABLY_FEASIBLE", "MARGINAL", "INFEASIBLE", "AIRBORNE"]
LP_DIR_VECTORS = np.array([[np.cos(a), np.sin(a)]
                           for a in np.linspace(0.0, 2.0 * np.pi, LP_DIRS,
                                                endpoint=False)])


# --- geometry helpers ---------------------------------------------------------------
def convex_hull(points):
    """CCW convex hull of 2-D points; returns 0, 1, 2 or >=3 unique vertices."""
    pts = np.asarray(points, float).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2))
    pts = np.unique(np.round(pts, 9), axis=0)
    if len(pts) <= 2:
        return pts

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                cr = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
                if cr <= 0.0:
                    out.pop()
                else:
                    break
            out.append(q)
        return out

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    p = pts[order]
    hull = np.array(half(p)[:-1] + half(p[::-1])[:-1])
    if len(hull) < 3:                       # all points collinear
        return np.array([p[0], p[-1]])
    area = 0.0
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        area += a[0] * b[1] - b[0] * a[1]
    if area < 0.0:                          # enforce CCW
        hull = hull[::-1]
    return hull


def point_seg_dist(p, a, b):
    e = b - a
    denom = float(e @ e)
    t = 0.0 if denom < 1e-18 else float(np.clip((p - a) @ e / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * e)))


def region_margin(p, pts):
    """Signed distance from ``p`` to the convex hull of ``pts``; + inside, - outside.

    ``None`` when the region is empty (no contact).  Degenerate hulls (one point, one
    segment) have no area, so they can only be touched: margin ``-distance``.
    """
    hull = convex_hull(pts)
    n = len(hull)
    if n == 0:
        return None
    if n == 1:
        return -float(np.linalg.norm(p - hull[0]))
    if n == 2:
        return -point_seg_dist(p, hull[0], hull[1])
    inside = True
    dmin = np.inf
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        e = b - a
        cr = e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])
        if cr < 0.0:
            inside = False
        dmin = min(dmin, abs(cr) / float(np.linalg.norm(e)))
    if inside:
        return float(dmin)
    return -min(point_seg_dist(p, hull[i], hull[(i + 1) % n]) for i in range(n))


def line_union_margin(p, segments):
    """Distance from ``p`` to the union of segments, returned as a non-positive margin."""
    if not segments:
        return None
    return -float(min(point_seg_dist(p, a, b) for a, b in segments))


def minmax_ankle_torque(weight, com_xy, patches, ankles):
    """Smallest achievable worst-ankle static moment (Nm) for the given contacts.

    ``patches`` is a list of (K,2) hull vertices (one per touching foot), ``ankles`` the
    matching (2,) ankle-axis points, ``weight`` = m*g.  Returns ``None`` when no static
    allocation exists (the CoM is outside the contact hull) - i.e. geometric
    infeasibility.
    """
    if not patches:
        return None
    nlam = int(sum(len(h) for h in patches))
    A_eq = np.zeros((3, nlam + 1))
    b_eq = np.array([weight, weight * com_xy[0], weight * com_xy[1]])
    col = 0
    for h in patches:
        A_eq[0, col:col + len(h)] = 1.0
        A_eq[1, col:col + len(h)] = h[:, 0]
        A_eq[2, col:col + len(h)] = h[:, 1]
        col += len(h)
    A_ub, b_ub = [], []
    col = 0
    for h, ank in zip(patches, ankles):
        coef = np.zeros(nlam + 1)
        for u in LP_DIR_VECTORS:
            coef[:nlam] = 0.0
            for j in range(len(h)):
                coef[col + j] = float(u @ (h[j] - ank))
            coef[-1] = -1.0                     # t >= sum_j lam_j u.(v_j - a_i)
            A_ub.append(coef.copy())
            b_ub.append(0.0)
        col += len(h)
    c = np.zeros(nlam + 1)
    c[-1] = 1.0
    bounds = [(0.0, weight)] * nlam + [(0.0, None)]
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    return float(res.fun) if res.success else None


# --- model index --------------------------------------------------------------------
class Robot:
    def __init__(self, model, prefix):
        n = mujoco.mj_name2id
        obj = mujoco.mjtObj
        self.prefix = prefix
        self.pelvis = n(model, obj.mjOBJ_BODY, f"{prefix}pelvis")
        self.mass = float(model.body_subtreemass[self.pelvis])
        self.site = {s: n(model, obj.mjOBJ_SITE, f"{prefix}{s}") for s in SITES}
        self.site_ids = [i for i in range(model.nsite)
                         if (mujoco.mj_id2name(model, obj.mjOBJ_SITE, i) or
                             "").startswith(prefix)]
        self.ankle = {side: n(model, obj.mjOBJ_SITE, f"{prefix}{side}_ankle")
                      for side in FOOT_SIDES}
        # world-projected contact patches: sole spheres of each foot, split by x sign
        self.patch = {}
        for side in FOOT_SIDES:
            body = n(model, obj.mjOBJ_BODY, f"{prefix}{side}_ankle_roll_link")
            toe, heel = [], []
            for g in range(model.ngeom):
                if model.geom_bodyid[g] != body:
                    continue
                if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE:
                    continue
                (toe if model.geom_pos[g][0] > 0.0 else heel).append(g)
            self.patch[side] = {"toe": np.array(toe, int), "heel": np.array(heel, int)}
        self.foot_body = {side: n(model, obj.mjOBJ_BODY,
                                  f"{prefix}{side}_ankle_roll_link")
                          for side in FOOT_SIDES}


def classify(margin, band=BAND):
    if margin is None:
        return "AIRBORNE"
    if margin < 0.0:
        return "INFEASIBLE"
    if margin < band:
        return "MARGINAL"
    return "STABLY_FEASIBLE"


# --- per-frame analysis -------------------------------------------------------------
def analyse_frame(model, data, robot, ref_pelvis_z):
    """Per-frame numbers for one robot after mj_forward.

    ``ref_pelvis_z`` is the reference base height (qpos[2]) of this robot's frame, the
    quantity the phase-2 stay-up gate (>= 0.45 m) is defined on.
    """
    com = data.subtree_com[robot.pelvis].copy()
    sites_z = {s: float(data.site_xpos[robot.site[s]][2]) for s in SITES}
    touch = {s: sites_z[s] < FOOT_Z_TOUCH for s in SITES}

    site_pts, foot_pts, segments, feet_down = [], [], [], []
    patches, ankles = [], []
    for side in FOOT_SIDES:
        t_toe, t_heel = touch[f"{side}_toe"], touch[f"{side}_heel"]
        foot_pts_side = []
        for s, t in (("toe", t_toe), ("heel", t_heel)):
            if not t:
                continue
            site_pts.append(data.site_xpos[robot.site[f"{side}_{s}"]][:2].copy())
            for g in robot.patch[side][s]:
                pt = data.geom_xpos[g][:2].copy()
                foot_pts.append(pt)
                foot_pts_side.append(pt)
        if t_toe or t_heel:
            feet_down.append(side)
            ankles.append(data.site_xpos[robot.ankle[side]][:2].copy())
            patches.append(convex_hull(foot_pts_side))
            # teacher rule (foot_rows): a foot is "planted" if ANY of its sole sites
            # touches; the toe-heel line is that foot's fore-aft extent
            segments.append((data.site_xpos[robot.site[f"{side}_heel"]][:2].copy(),
                             data.site_xpos[robot.site[f"{side}_toe"]][:2].copy()))

    com_xy = com[:2]
    m_hull_foot = region_margin(com_xy, foot_pts)
    m_hull_site = region_margin(com_xy, site_pts)
    m_line = line_union_margin(com_xy, segments)
    cls = classify(m_hull_foot)
    lowest = min(float(data.site_xpos[i][2]) for i in robot.site_ids)
    if not feet_down:
        state = "airborne_float" if lowest > AIRBORNE_GROUND_Z else "airborne_ground"
    else:
        state = "both" if len(feet_down) == 2 else "one"

    m_g = robot.mass * G
    lever_i = (np.linalg.norm(com_xy[None, :] - np.array(ankles), axis=1)
               if ankles else np.zeros(0))
    lever = (float(np.linalg.norm(com_xy - np.mean(ankles, axis=0)))
             if ankles else None)
    tau_lp = (minmax_ankle_torque(m_g, com_xy, patches, ankles)
              if ankles else None)
    # fore-aft (pitch) / lateral (roll) split in the mean foot frame
    tau_pitch = tau_roll = None
    if ankle_ok := bool(ankles):
        fwd = np.zeros(2)
        for side in feet_down:
            R = data.xmat[robot.foot_body[side]].reshape(3, 3)
            f = R[:2, 0]
            n = np.linalg.norm(f)
            if n > 1e-9:
                fwd += f / n
        n = np.linalg.norm(fwd)
        if n > 1e-9:
            fwd /= n
            lat = np.array([-fwd[1], fwd[0]])
            vec = com_xy - np.mean(ankles, axis=0)
            tau_pitch = float(m_g * abs(vec @ fwd))
            tau_roll = float(m_g * abs(vec @ lat))

    feet_z = {side: min(sites_z[f"{side}_toe"], sites_z[f"{side}_heel"])
              for side in FOOT_SIDES}
    nofoot = all(z > FOOT_Z_NEAR for z in feet_z.values())

    return {
        "com_xy": com_xy, "com_z": float(com[2]), "pelvis_z": float(ref_pelvis_z),
        "margin_foot_m": m_hull_foot, "margin_site_m": m_hull_site,
        "margin_line_m": m_line, "cls": cls, "state": state, "n_feet": len(feet_down),
        "lever_m": lever,
        "tau_pendulum_nm": None if lever is None else float(m_g * lever),
        "tau_naive_one_ankle_nm": (None if lever_i.size == 0
                                   else float(m_g * float(lever_i.max()))),
        "tau_minmax_nm": tau_lp,
        "tau_pitch_nm": tau_pitch, "tau_roll_nm": tau_roll,
        "site_z_min": min(sites_z.values()), "lowest_site_z": lowest,
        "foot_z_left": feet_z["left"], "foot_z_right": feet_z["right"],
        "foot_z_min": min(feet_z.values()), "nofoot_2cm": bool(nofoot),
        "penetration_m": max(0.0, -min(sites_z.values())),
        "sole_pos": np.array([data.site_xpos[robot.site[s]].copy() for s in SITES]),
    }


def quat_angle(q1, q2):
    """Angle (rad) between two (w,x,y,z) quaternions."""
    return float(2.0 * np.arccos(min(1.0, abs(float(np.dot(q1, q2))))))


def episodes(mask, merge_gap=MERGE_GAP):
    """Merged runs of True; inclusive end frame; gaps <= merge_gap frames are bridged."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > merge_gap + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return [{"start_frame": int(a), "end_frame": int(b), "frames": int(b - a + 1)}
            for a, b in runs]


# --- technique analysis -------------------------------------------------------------
def analyse_technique(model, tech):
    npz = np.load(os.path.join(REF_DIR, f"{tech}.npz"), allow_pickle=True)
    qa, qb = npz["qpos_a"], npz["qpos_b"]
    t = np.asarray(npz["t"], float)
    meta = json.loads(str(npz["meta"]))
    T = len(t)
    data = mujoco.MjData(model)
    robots = {"a": Robot(model, "a_"), "b": Robot(model, "b_")}

    rows = {k: [] for k in ("a", "b")}
    prev = {k: None for k in ("a", "b")}
    for k in range(T):
        data.qpos[:] = 0.0
        data.qpos[0:36] = qa[k]
        data.qpos[36:72] = qb[k]
        mujoco.mj_forward(model, data)
        for key, rob in robots.items():
            q = qa[k] if key == "a" else qb[k]
            row = analyse_frame(model, data, rob, q[2])
            if prev[key] is None:
                row.update({"dq_max_rad": None, "dp_base_m": None,
                            "drot_base_rad": None, "dfoot_m": None})
            else:
                pq, psole = prev[key]
                row.update({
                    "dq_max_rad": float(np.max(np.abs(q[7:36] - pq[7:36]))),
                    "dp_base_m": float(np.linalg.norm(q[:3] - pq[:3])),
                    "drot_base_rad": quat_angle(pq[3:7], q[3:7]),
                    "dfoot_m": float(np.max(np.linalg.norm(row["sole_pos"] - psole,
                                                           axis=1))),
                })
            prev[key] = (q, row["sole_pos"])
            rows[key].append(row)

    out = {"frames": T, "duration_s": float(meta["duration_s"]), "dt": float(meta["dt"]),
           "t": t, "npz_roles": meta.get("roles"), "role_a": ROLES[tech],
           "role_b": ROLE_B[tech], "robots": {}}
    npz_path = os.path.join(REF_DIR, f"{tech}.npz")
    with open(npz_path, "rb") as fh:
        blob = fh.read()
    out["npz"] = {"path": os.path.relpath(npz_path, REPO), "bytes": len(blob),
                  "sha256": hashlib.sha256(blob).hexdigest()[:16],
                  "mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
                                         time.localtime(os.path.getmtime(npz_path)))}
    for key, rob in robots.items():
        rr = rows[key]
        cls = np.array([r["cls"] for r in rr])
        ground = np.array([r["pelvis_z"] < STAND_PELVIS_Z for r in rr])
        standing = ~ground

        def counts_of(sel):
            return {c: int(np.sum(sel & (cls == c))) for c in CLASS_ORDER}

        def pct(cnt, total):
            return {c: (100.0 * v / total if total else 0.0) for c, v in cnt.items()}

        counts, counts_std, counts_gnd = counts_of(np.ones(T, bool)), \
            counts_of(standing), counts_of(ground)

        band_sens = {}
        for b in BAND_SENSITIVITY:
            c2 = np.array([classify(r["margin_foot_m"], band=b) for r in rr])
            band_sens[f"{b:.2f}"] = {c: int(np.sum(c2 == c)) for c in CLASS_ORDER}

        site_cls = np.array([classify(r["margin_site_m"]) for r in rr])
        line_cls = np.array(["AIRBORNE" if r["margin_line_m"] is None else
                             ("MARGINAL" if -r["margin_line_m"] <= BAND else "INFEASIBLE")
                             for r in rr])

        def arr(name):
            return np.array([np.nan if r[name] is None else r[name] for r in rr])

        lever, tau_pend = arr("lever_m"), arr("tau_pendulum_nm")
        tau_lp, tau_naive = arr("tau_minmax_nm"), arr("tau_naive_one_ankle_nm")
        margin = arr("margin_foot_m")
        finite_lp = np.isfinite(tau_lp)
        over = finite_lp & (tau_lp > ANKLE_LIMIT_NM)
        inf_mask = cls == "INFEASIBLE"
        eps = episodes(inf_mask)
        for e in eps:
            e["start_s"] = float(t[e["start_frame"]])
            e["end_s"] = float(t[e["end_frame"]])
        feas_mask = np.isin(cls, ["STABLY_FEASIBLE", "MARGINAL"])
        feps = [e for e in episodes(feas_mask)
                if (e["end_frame"] - e["start_frame"] + 1) >= 5]     # >= 0.10 s
        for e in feps:
            e["start_s"] = float(t[e["start_frame"]])
            e["end_s"] = float(t[e["end_frame"]])

        def argmax(a):
            return int(np.nanargmax(a)) if np.any(np.isfinite(a)) else None

        i_lev, i_tau, i_lp = argmax(lever), argmax(tau_pend), argmax(tau_lp)

        def at(i, name):
            return None if i is None else rr[i][name]

        # --- reference integrity: contact availability, placement, montage seams -------
        nofoot = np.array([r["nofoot_2cm"] for r in rr])
        feps_raw = episodes(np.array([r["state"] == "airborne_float" for r in rr]))
        dt = float(meta["dt"])
        longest_float_s = max((e["frames"] for e in feps_raw), default=0) * dt
        pen = np.array([r["penetration_m"] for r in rr])
        i_pen = int(np.argmax(pen))
        dq = arr("dq_max_rad")
        dp = arr("dp_base_m")
        drot = arr("drot_base_rad")
        dfoot = arr("dfoot_m")
        seams = []
        for i in range(1, T):
            bad = []
            if np.isfinite(dq[i]) and dq[i] > SEAM_LIMITS["dq_rad"]:
                bad.append(f"dq {dq[i]:.3f} rad (limit {SEAM_LIMITS['dq_rad']})")
            if np.isfinite(dp[i]) and dp[i] > SEAM_LIMITS["dp_m"]:
                bad.append(f"dBase {dp[i]:.3f} m (limit {SEAM_LIMITS['dp_m']})")
            if np.isfinite(drot[i]) and drot[i] > SEAM_LIMITS["drot_rad"]:
                bad.append(f"dSpin {drot[i]:.3f} rad (limit {SEAM_LIMITS['drot_rad']})")
            if np.isfinite(dfoot[i]) and dfoot[i] > FOOT_TELEPORT_M:
                bad.append(f"foot jump {dfoot[i]:.3f} m")
            if bad:
                seams.append({"frame": i, "t": float(t[i]), "over": bad})
        i_dq = argmax(dq)
        i_dp = argmax(dp)
        # contact-transition sanity: CoM far outside while a foot is planted
        one = np.array([r["state"] == "one" for r in rr])
        viol_one = one & np.isfinite(margin) & (margin < -0.05)
        i_viol = (int(np.argmin(np.where(viol_one, margin, np.inf)))
                  if viol_one.any() else None)

        out["robots"][key] = {
            "prefix": rob.prefix, "mass_kg": rob.mass,
            "class_counts": counts, "class_pct": pct(counts, T),
            "class_counts_standing": counts_std,
            "class_pct_standing": pct(counts_std, int(standing.sum())),
            "class_counts_ground": counts_gnd,
            "class_pct_ground": pct(counts_gnd, int(ground.sum())),
            "standing_frames": int(standing.sum()), "ground_frames": int(ground.sum()),
            "band_sensitivity": band_sens,
            "class_counts_site_hull": {c: int(np.sum(site_cls == c))
                                       for c in CLASS_ORDER},
            "class_counts_line_union": {c: int(np.sum(line_cls == c))
                                        for c in CLASS_ORDER},
            "support_states": {s: int(sum(1 for r in rr if r["state"] == s))
                               for s in ("one", "both", "airborne_ground",
                                         "airborne_float")},
            "airborne_frames": int(np.sum(cls == "AIRBORNE")),
            "airborne_float_pct": 100.0 * float(np.mean(
                [r["state"] == "airborne_float" for r in rr])),
            "lowest_site_z_median": float(np.median([r["lowest_site_z"] for r in rr])),
            "lowest_site_z_max": float(np.max([r["lowest_site_z"] for r in rr])),
            "lowest_site_z_min": float(np.min([r["lowest_site_z"] for r in rr])),
            "worst_lever_m": None if i_lev is None else float(lever[i_lev]),
            "worst_lever_frame": i_lev,
            "worst_lever_s": None if i_lev is None else float(t[i_lev]),
            "worst_lever_state": at(i_lev, "state"),
            "worst_tau_pendulum_nm": None if i_tau is None else float(tau_pend[i_tau]),
            "worst_tau_pendulum_frame": i_tau,
            "worst_tau_pendulum_s": None if i_tau is None else float(t[i_tau]),
            "worst_tau_pendulum_state": at(i_tau, "state"),
            "worst_tau_pendulum_pitch_nm": at(i_tau, "tau_pitch_nm"),
            "worst_tau_pendulum_roll_nm": at(i_tau, "tau_roll_nm"),
            "worst_tau_minmax_nm": None if i_lp is None else float(tau_lp[i_lp]),
            "worst_tau_minmax_frame": i_lp,
            "worst_tau_minmax_s": None if i_lp is None else float(t[i_lp]),
            "median_tau_minmax_nm": (None if not np.any(finite_lp)
                                     else float(np.nanmedian(tau_lp))),
            "p95_tau_minmax_nm": (None if not np.any(finite_lp)
                                  else float(np.nanpercentile(tau_lp, 95))),
            "max_tau_naive_one_ankle_nm": (None if not np.any(np.isfinite(tau_naive))
                                           else float(np.nanmax(tau_naive))),
            "max_margin_m": (None if not np.any(np.isfinite(margin))
                             else float(np.nanmax(margin))),
            "min_margin_m": (None if not np.any(np.isfinite(margin))
                             else float(np.nanmin(margin))),
            "frames_tau_minmax_over_limit": int(np.sum(over)),
            "pct_tau_minmax_over_limit": 100.0 * float(np.sum(over)) / T,
            "lp_infeasible_frames": int(np.sum((~finite_lp) & (cls != "AIRBORNE"))),
            "infeasible_episodes": eps,
            "feasible_episodes": feps,
            "holdable_total_s": float(sum(e["frames"] for e in feps)
                                      * float(np.median(np.diff(t)))),
            "first_feasible_s": (None if not feps else feps[0]["start_s"]),
            "last_feasible_s": (None if not feps else feps[-1]["end_s"]),
            # reference integrity
            "frames_no_foot_2cm": int(nofoot.sum()),
            "pct_no_foot_2cm": 100.0 * float(nofoot.mean()),
            "longest_no_foot_2cm_s": max((e["frames"] for e in
                                          episodes(nofoot)), default=0) * dt,
            "float_episodes": feps_raw,
            "longest_float_s": longest_float_s,
            "median_lowest_site_z": float(np.median([r["lowest_site_z"] for r in rr])),
            "max_sole_penetration_m": float(pen.max()),
            "max_sole_penetration_frame": i_pen,
            "max_sole_penetration_s": float(t[i_pen]),
            "frames_penetration_over_2cm": int(np.sum(pen > 0.02)),
            "frames_penetration_over_5cm": int(np.sum(pen > PENETRATION_REFUSE_M)),
            "pct_frames_penetration_over_2cm": 100.0 * float(np.mean(pen > 0.02)),
            "median_dq_rad": (None if not np.any(np.isfinite(dq))
                              else float(np.nanmedian(dq))),
            "p99_dq_rad": (None if not np.any(np.isfinite(dq))
                           else float(np.nanpercentile(dq, 99))),
            "max_dq_rad": None if i_dq is None else float(dq[i_dq]),
            "max_dq_frame": i_dq,
            "max_dq_s": None if i_dq is None else float(t[i_dq]),
            "max_dp_base_m": None if i_dp is None else float(dp[i_dp]),
            "max_dp_base_frame": i_dp,
            "max_dp_base_s": None if i_dp is None else float(t[i_dp]),
            "max_drot_base_rad": (None if not np.any(np.isfinite(drot))
                                  else float(np.nanmax(drot))),
            "max_dfoot_m": (None if not np.any(np.isfinite(dfoot))
                            else float(np.nanmax(dfoot))),
            "seam_outliers": seams,
            "seam_outlier_frames": [e["frame"] for e in seams],
            "single_support_violation_frames": int(viol_one.sum()),
            "pct_single_support_violation": 100.0 * float(viol_one.mean()),
            "worst_single_support_violation_m": (None if i_viol is None
                                                 else float(margin[i_viol])),
            "worst_single_support_violation_frame": i_viol,
            "series": {"t": t, "margin_m": margin, "class": cls, "tau_nm": tau_lp,
                       "tau_pendulum_nm": tau_pend,
                       "state": np.array([r["state"] for r in rr]),
                       "pelvis_z": np.array([r["pelvis_z"] for r in rr]),
                       "lowest_site_z": np.array([r["lowest_site_z"] for r in rr]),
                       "foot_z_min": np.array([r["foot_z_min"] for r in rr]),
                       "margin_site_m": arr("margin_site_m")},
        }
    out["verdict"] = reference_verdict(out)
    # pair contact range: how often the two robots are close enough for the opponent to
    # be a second support path (the foot-only criterion is a lower bound on support)
    d_pair = np.array([float(np.linalg.norm(rows["a"][k]["com_xy"] -
                                            rows["b"][k]["com_xy"]))
                       for k in range(T)])
    out["pair"] = {
        "com_distance_median_m": float(np.median(d_pair)),
        "pct_within_0.4m": 100.0 * float(np.mean(d_pair < 0.4)),
        "pct_within_0.6m": 100.0 * float(np.mean(d_pair < 0.6)),
    }
    return out


# --- reporting ----------------------------------------------------------------------
def reference_verdict(r):
    """Rule-based reference-integrity verdict for one technique.

    Decides whether the phase-2 failure of a reference is (i) a data defect
    (rebuild/regenerate), (ii) a local defect or small static-geometry error (adjust the
    reference) or (iii) a physically consistent dynamic trajectory (invest in the
    stabiliser).  Thresholds are the constants above; the evidence dict carries them.
    """
    rb = r["robots"]

    def extreme(fn):
        key = max(rb, key=lambda k: fn(rb[k]))
        return key, fn(rb[key])

    float_rob, longest_float = extreme(lambda b: b["longest_float_s"])
    med_low = max(b["median_lowest_site_z"] for b in rb.values())
    pen_rob, pen_val = extreme(lambda b: b["max_sole_penetration_m"])
    pen_frame = rb[pen_rob]["max_sole_penetration_frame"]
    dq_rob, dq_val = extreme(lambda b: b["max_dq_rad"] or 0.0)
    dp_rob, dp_val = extreme(lambda b: b["max_dp_base_m"] or 0.0)
    drot_rob, drot_val = extreme(lambda b: b["max_drot_base_rad"] or 0.0)
    foot_rob, foot_val = extreme(lambda b: b["max_dfoot_m"] or 0.0)
    seams = [(k, e) for k, b in rb.items() for e in b["seam_outliers"]]
    ev = {
        "longest_float_s": longest_float, "float_robot": float_rob,
        "median_lowest_site_z": med_low,
        "max_sole_penetration_m": pen_val, "penetration_robot": pen_rob,
        "penetration_frame": pen_frame,
        "max_dq_rad": dq_val, "max_dp_base_m": dp_val,
        "max_drot_base_rad": drot_val, "max_dfoot_m": foot_val,
        "n_seam_outliers": len(seams),
        "seam_frames": sorted({e["frame"] for _, e in seams}),
        "frames_penetration_over_2cm": max(b["frames_penetration_over_2cm"]
                                           for b in rb.values()),
        "frames_penetration_over_5cm": max(b["frames_penetration_over_5cm"]
                                           for b in rb.values()),
        "frames": r["frames"],
        "holdable_total_s": max(b["holdable_total_s"] for b in rb.values()),
        "max_minmax_tau_nm": max(b["worst_tau_minmax_nm"] or 0.0 for b in rb.values()),
        "limits": {"dq_rad": SEAM_LIMITS["dq_rad"], "dp_m": SEAM_LIMITS["dp_m"],
                   "drot_rad": SEAM_LIMITS["drot_rad"], "foot_m": FOOT_TELEPORT_M,
                   "penetration_m": PENETRATION_REFUSE_M,
                   "float_s": FLOAT_SUSPECT_S,
                   "median_lowest_site_z": FLOAT_MEDIAN_Z},
    }
    deficits, feas_pct, stands = [], [], []
    for b in rb.values():
        s = b["series"]
        sel = (s["class"] != "AIRBORNE") & (s["pelvis_z"] >= STAND_PELVIS_Z)
        stands.append(int(sel.sum()))
        deficits.append(float(max(0.0, -np.nanmin(s["margin_m"][sel])))
                         if sel.any() else 0.0)
        feas_pct.append(b["class_pct_standing"]["STABLY_FEASIBLE"] +
                        b["class_pct_standing"]["MARGINAL"])
    ev["standing_frames"] = max(stands)
    ev["worst_standing_deficit_m"] = max(deficits)
    ev["stand_feasible_pct"] = max(feas_pct)
    ev["control_limit"] = bool(ev["stand_feasible_pct"] >= 40.0 and
                               ev["max_minmax_tau_nm"] <= ANKLE_LIMIT_NM)

    defects = []
    if ev["n_seam_outliers"]:
        defects.append(f"{ev['n_seam_outliers']} frame(s) around {ev['seam_frames'][:4]} "
                       f"violate the retime limits (dq {ev['max_dq_rad']:.3f} rad, dBase "
                       f"{ev['max_dp_base_m']:.3f} m, spin {ev['max_drot_base_rad']:.3f} "
                       f"rad, foot jump {ev['max_dfoot_m']:.3f} m): localise and resample "
                       "the seam")
    if ev["frames_penetration_over_2cm"]:
        defects.append(f"{ev['frames_penetration_over_2cm']} frame(s) drive a sole up to "
                       f"{ev['max_sole_penetration_m']:.3f} m below the mat "
                       f"(worst at frame {pen_frame}): clamp/re-place them")
    small_deficit = ev["worst_standing_deficit_m"] <= 0.05
    float_bad = (ev["longest_float_s"] >= FLOAT_SUSPECT_S
                 and ev["median_lowest_site_z"] > FLOAT_MEDIAN_Z)
    deep_pen = ev["max_sole_penetration_m"] >= PENETRATION_REFUSE_M
    localized = ev["frames_penetration_over_5cm"] <= max(3, int(0.02 * ev["frames"]))
    hard_jump = (ev["max_dq_rad"] > SEAM_JUMP_FACTOR * SEAM_LIMITS["dq_rad"]
                 or ev["max_dp_base_m"] > SEAM_JUMP_FACTOR * SEAM_LIMITS["dp_m"]
                 or ev["max_drot_base_rad"] > SEAM_JUMP_FACTOR * SEAM_LIMITS["drot_rad"]
                 or ev["max_dfoot_m"] > SEAM_JUMP_FACTOR * FOOT_TELEPORT_M)
    ev["control_limit"] = bool(ev["stand_feasible_pct"] >= 40.0 and
                               ev["max_minmax_tau_nm"] <= ANKLE_LIMIT_NM)

    def ret(verdict, action, why, also=None):
        return {"verdict": verdict, "action": action, "why": why,
                "also": defects if also is None else also, "evidence": ev}

    if float_bad:
        return ret("REF_INVALID", "REGENERATE",
                   f"robot {float_rob} has {ev['longest_float_s']:.2f} s of contact-free "
                   f"flight with the body a median {ev['median_lowest_site_z']:.3f} m "
                   "above the mat: the reference is not placeable on the mat")
    if deep_pen and not localized:
        return ret("REF_INVALID", "REGENERATE",
                   f"robot {pen_rob}'s sole is {ev['max_sole_penetration_m']:.3f} m "
                   f"below the mat on {ev['frames_penetration_over_5cm']} frames: the "
                   "vertical placement is not physical")
    if hard_jump:
        return ret("REF_INVALID", "REGENERATE",
                   "positional discontinuity beyond 2x the retime limits "
                   f"(dq {ev['max_dq_rad']:.3f} rad, dBase {ev['max_dp_base_m']:.3f} m, "
                   f"dSpin {ev['max_drot_base_rad']:.3f} rad, foot jump "
                   f"{ev['max_dfoot_m']:.3f} m)")
    if ev["control_limit"]:
        return ret("REF_VALID_DYNAMIC", "STABILISE",
                   f"{ev['stand_feasible_pct']:.0f}% of standing frames are statically "
                   f"feasible and the worst min-max ankle demand "
                   f"{ev['max_minmax_tau_nm']:.1f} Nm is within the "
                   f"{ANKLE_LIMIT_NM:.0f} Nm limit: the reference is a valid trajectory a "
                   "balancer can track")
    if small_deficit:
        why = ("the standing posture is statically infeasible by only "
               f"{1000 * ev['worst_standing_deficit_m']:.1f} mm: shift the CoM or plant "
               "the feet (the rest of the trajectory is contact-consistent)")
        return ret("REF_VALID_ADJUSTABLE", "ADJUST", why)
    if defects:
        why = defects[0]
        if not small_deficit:
            why += (f"; the standing CoM excursion still reaches "
                    f"{ev['worst_standing_deficit_m']:.3f} m with "
                    f"{ev['holdable_total_s']:.2f} s of holdable windows, so imitation "
                    "beyond the repaired frames has to be dynamic")
        return ret("REF_VALID_ADJUSTABLE", "ADJUST", why, also=defects[1:])
    why = (f"contact-consistent and continuous, but the CoM leaves the support region by "
           f"up to {ev['worst_standing_deficit_m']:.3f} m during the technique "
           f"({ev['holdable_total_s']:.2f} s of holdable windows): the excursion is a "
           "dynamic transient, so imitation must be dynamic (or phase-gated), not "
           "quasi-static")
    return ret("REF_VALID_DYNAMIC", "STABILISE", why)


def print_integrity(results):
    print("--- reference integrity (contact availability, placement, montage seams) ---")
    hdr = (f"{'technique':11s} {'r':1s} {'nofoot<2cm%':>11s} {'longest float':>13s} "
           f"{'med low-z':>9s} {'pen max m@ s':>15s} {'max dq rad@ s':>14s} "
           f"{'max dBase m':>11s} {'max dSpin':>9s} {'foot jump m':>11s} "
           f"{'seam frames':>28s}")
    print(hdr)
    print("-" * len(hdr))
    for tech in TECHS:
        for key in ("a", "b"):
            d = results[tech]["robots"][key]
            seam = ("-") if not d["seam_outliers"] else (", ".join(
                str(e["frame"]) + f"@{e['t']:.2f}" for e in d["seam_outliers"][:4]) +
                ("..." if len(d["seam_outliers"]) > 4 else ""))
            mx = (f"{d['max_dq_rad']:.3f}@{d['max_dq_s']:.2f}"
                  if d["max_dq_rad"] is not None else "-")
            pen = (f"{d['max_sole_penetration_m']:.3f}@{d['max_sole_penetration_s']:.2f}"
                   if d["max_sole_penetration_m"] > 0 else "-")
            print(f"{tech:11s} {key:1s} {d['pct_no_foot_2cm']:11.1f} "
                  f"{d['longest_float_s']:13.2f} {d['median_lowest_site_z']:9.3f} "
                  f"{pen:>15s} {mx:>14s} "
                  f"{d['max_dp_base_m'] if d['max_dp_base_m'] is not None else -1:11.3f} "
                  f"{d['max_drot_base_rad'] if d['max_drot_base_rad'] is not None else -1:9.3f} "
                  f"{d['max_dfoot_m'] if d['max_dfoot_m'] is not None else -1:11.3f} "
                  f"{seam:>28s}")
    print()
    print("nofoot<2cm% = frames where neither foot has a sole site within 0.02 m of the "
          "mat; longest float = longest contact-free (nothing within 0.05 m) episode;")
    print("seam frames = frames whose frame-to-frame delta exceeds the retime limits "
          "(joint 6 rad/s, base 3 m/s, spin 8 rad/s, foot 0.08 m/frame at 50 Hz).")
    print()
    print("--- pair contact range (opponent as a second support path?) ---")
    for tech in TECHS:
        p = results[tech]["pair"]
        print(f"  {tech:11s} CoM-CoM median {p['com_distance_median_m']:.3f} m, "
              f"within 0.4 m {p['pct_within_0.4m']:.0f}%, within 0.6 m "
              f"{p['pct_within_0.6m']:.0f}% of frames")
    print()
    print("--- verdicts: is the failure data, geometry or control? ---")
    for tech in TECHS:
        v = results[tech]["verdict"]
        print(f"  {tech:11s} {v['verdict']:22s} action={v['action']:11s} {v['why']}")
        for d in v.get("also", []):
            print(f"  {'':11s} {'':22s} {'':11s}   also: {d}")


def print_table(results):
    hdr = (f"{'technique':11s} {'r':1s} {'frames':>6s} {'stand':>5s} {'stably%':>7s} "
           f"{'marg%':>6s} {'inf%':>6s} {'inf/stand%':>10s} {'air%':>5s} {'float%':>6s} "
           f"{'1ft%':>5s} {'2ft%':>5s} {'worst lever m@ s':>17s} {'ankle moment Nm@ s':>18s} "
           f"{'inf episodes s':>20s}")
    print(hdr)
    print("-" * len(hdr))
    for tech in TECHS:
        for key in ("a", "b"):
            d = results[tech]["robots"][key]
            cp, cs = d["class_pct"], d["class_pct_standing"]
            eps = d["infeasible_episodes"]
            ep_txt = "-" if not eps else (f"{eps[0]['start_s']:.2f}-{eps[-1]['end_s']:.2f}"
                                          f" x{len(eps)}")
            lever = (f"{d['worst_lever_m']:.3f}@{d['worst_lever_s']:.2f}"
                     if d["worst_lever_m"] is not None else "-")
            tau = (f"{d['worst_tau_pendulum_nm']:.1f}@{d['worst_tau_pendulum_s']:.2f}"
                   if d["worst_tau_pendulum_nm"] is not None else "-")
            n = d["standing_frames"] + d["ground_frames"]
            st = d["support_states"]
            print(f"{tech:11s} {key:1s} {n:6d} "
                  f"{d['standing_frames']:5d} {cp['STABLY_FEASIBLE']:7.1f} "
                  f"{cp['MARGINAL']:6.1f} {cp['INFEASIBLE']:6.1f} {cs['INFEASIBLE']:10.1f} "
                  f"{cp['AIRBORNE']:5.1f} {d['airborne_float_pct']:6.1f} "
                  f"{100.0 * st['one'] / n:5.1f} {100.0 * st['both'] / n:5.1f} "
                  f"{lever:>17s} {tau:>18s} {ep_txt:>20s}")
    print()
    print("inf% = share of ALL frames infeasible on the footprint hull; inf/stand% = share "
          "of STANDING frames (ref pelvis >= 0.45 m) infeasible;")
    print("air% = no foot contact; float% = no *body* contact either (lowest body site "
          "> 0.05 m: the reference floats); 1ft%/2ft% = one/both feet planted;")
    print("ankle moment = m*g*lever to the ankle axis (single support) / ankle-axis "
          "midpoint (double support) at its worst frame; inf episodes = first/last")
    print("INFEASIBLE frame time and run count (gaps <= 0.04 s merged).")
    print()
    print("--- holdable windows (>= 0.10 s contiguous CoM-in-support episodes) ---")
    for tech in TECHS:
        for key in ("a", "b"):
            d = results[tech]["robots"][key]
            feps = d["feasible_episodes"]
            if not feps:
                print(f"  {tech:11s} {key}: none (no frame with the CoM in the footprint)")
                continue
            win = ", ".join(f"{e['start_s']:.2f}-{e['end_s']:.2f}" for e in feps)
            print(f"  {tech:11s} {key}: {d['holdable_total_s']:.2f} s total -> {win}")


def headline(results):
    """Global worst standing frames: deepest support violations + highest demand."""
    viol, pend, lp, viol_any = [], [], [], []
    for tech in TECHS:
        r = results[tech]
        for key in ("a", "b"):
            s = r["robots"][key]["series"]
            for i in range(len(s["t"])):
                if s["class"][i] == "AIRBORNE":
                    continue
                e = {"technique": tech, "robot": key, "frame": i,
                     "t": float(s["t"][i]), "margin_m": float(s["margin_m"][i]),
                     "tau_pendulum_nm": float(s["tau_pendulum_nm"][i]),
                     "tau_minmax_nm": (None if not np.isfinite(s["tau_nm"][i])
                                       else float(s["tau_nm"][i])),
                     "pelvis_z": float(s["pelvis_z"][i])}
                viol_any.append(dict(e))
                if s["pelvis_z"][i] < STAND_PELVIS_Z:
                    continue
                viol.append(e)
                pend.append(e)
                if e["tau_minmax_nm"] is not None:
                    lp.append(e)
    viol.sort(key=lambda e: e["margin_m"])
    viol_any.sort(key=lambda e: e["margin_m"])
    pend.sort(key=lambda e: -e["tau_pendulum_nm"])
    lp.sort(key=lambda e: -e["tau_minmax_nm"])
    return {"deepest_violations": viol[:3],
            "deepest_violations_any": viol_any[:3],
            "highest_pendulum_demand": pend[:3], "highest_minmax_demand": lp[:3],
            "standing_feasible_pct_aggregate": 100.0 * (
                sum(r["robots"][k]["class_counts_standing"]["STABLY_FEASIBLE"] +
                    r["robots"][k]["class_counts_standing"]["MARGINAL"]
                    for r in results.values() for k in ("a", "b")) /
                max(1, sum(r["robots"][k]["standing_frames"]
                           for r in results.values() for k in ("a", "b")))),
            "feasible_pct_aggregate": 100.0 * (
                sum(r["robots"][k]["class_counts"]["STABLY_FEASIBLE"] +
                    r["robots"][k]["class_counts"]["MARGINAL"]
                    for r in results.values() for k in ("a", "b")) /
                max(1, sum(r["frames"] * 2 for r in results.values())))}


def write_json(results, head, checks):
    doc = {
        "meta": {
            "generated_by": "scripts/audit_support_envelope.py",
            "date": "2026-10-08",
            "scene": "robots/wrestling_scene.xml",
            "techniques": TECHS,
            "frame_source": "data/refs/<TECH>.npz (qpos_a, qpos_b, t at 50 Hz)",
            "replay": "kinematic: qpos per frame + mujoco.mj_forward, no dynamics",
            "com": "data.subtree_com[pelvis body] (whole robot)",
            "contact_rule": f"sole site z < {FOOT_Z_TOUCH} m "
                            "(src/teacher/controller.py:66)",
            "contact_sites": list(SITES),
            "primary_region": "convex hull of the world-projected sole contact-sphere "
                              "patches of touching (foot, site) pairs (g1.xml:108-115)",
            "secondary_regions": {
                "site_hull": "convex hull of touching toe/heel sites (degenerate: midline)",
                "line_union": "union of the heel->toe segments of planted feet (any sole "
                              "site touching; 1-D)",
            },
            "margin_band_m": BAND,
            "band_sensitivity_m": BAND_SENSITIVITY,
            "classes": CLASS_ORDER,
            "ground_flag": f"reference pelvis_z < {STAND_PELVIS_Z} m",
            "torque_models": {
                "tau_pendulum": "m*g*d, inverted pendulum about the ankle axis "
                                "(single support exact; double support = combined "
                                "ankle-axis midpoint)",
                "tau_minmax": "smallest achievable worst-ankle moment over all static "
                              "load shares and CoP placements inside the touching "
                              "foot patches (LP, 12 directions, ~3.4% under-estimate); "
                              "infeasible iff the CoM is outside the contact hull",
                "tau_naive_one_ankle": "m*g*max_i|CoM-ankle_i| (one ankle takes the "
                                       "whole moment) - deliberately naive upper bound, "
                                       "NOT the demand",
            },
            "mass_kg": next(iter(results.values()))["robots"]["a"]["mass_kg"],
            "gravity": G,
            "ankle_actuator_limit_nm": ANKLE_LIMIT_NM,
            "ankle_limit_source": "actuatorfrcrange on the ankle joints, robots/g1/g1.xml",
            "episode_merge_gap_frames": MERGE_GAP,
            "integrity": {
                "foot_near_threshold_m": FOOT_Z_NEAR,
                "float_threshold_m": AIRBORNE_GROUND_Z,
                "seam_limits": SEAM_LIMITS,
                "seam_jump_factor": SEAM_JUMP_FACTOR,
                "foot_teleport_m": FOOT_TELEPORT_M,
                "penetration_refuse_m": PENETRATION_REFUSE_M,
                "float_suspect_s": FLOAT_SUSPECT_S,
                "verdict_rules": [
                    "REF_INVALID/REGENERATE: contact-free flight >= 0.40 s with median "
                    "lowest body site > 0.05 m; or sole penetration >= 0.05 m; or a "
                    "frame-to-frame jump beyond 2x the retime limits",
                    "REF_VALID_ADJUSTABLE/ADJUST: a frame violates the retime limits "
                    "(seam artifact), or the worst standing support deficit <= 0.05 m "
                    "(small CoM/foot correction)",
                    "REF_VALID_DYNAMIC/STABILISE: contact-consistent and continuous, and "
                    "either >= 40% of standing frames are feasible within the 50 Nm "
                    "ankle limit, or the CoM leaves the support region by more than "
                    "0.05 m during the technique (dynamic transient)",
                ],
            },
            "role_a_per_technique": ROLES, "role_b_per_technique": ROLE_B,
            "self_checks": checks,
            "caveats": [
                "Kinematic replay describes the reference geometry, not a physics "
                "roll-out.",
                "Foot contacts are a teacher-style height threshold, not the collision "
                "solver; reference sole sites below the mat (ground penetration) count "
                "as touching.",
                "Ground-phase frames (ref pelvis < 0.45 m) are supported by knees, hands "
                "or torso; the foot-only criterion does not govern there.",
                "Frames where the lowest body site is > 0.05 m above the mat "
                "(state airborne_float) have nothing in contact: either intended flight "
                "or a vertical-placement defect of the reference; they are counted "
                "separately (airborne_float_pct) and must not be read as support failure.",
                "Both torque models are static: dynamics, CoM velocity, external "
                "grappling load, torso/arm inertia, foot slip and actuator rates are "
                "ignored; the min-max LP assumes ideal load sharing.",
                "The support criterion uses the FEET only and is therefore a lower bound "
                "on support: in clinch frames the opponent is an additional contact path "
                "(see the per-technique pair contact-range numbers), so the foot-only "
                "deficit is an upper bound on what the feet must supply.",
            ],
        },
        "headline": head,
        "techniques": {},
    }
    for tech, r in results.items():
        entry = {k: v for k, v in r.items() if k != "t"}
        entry["robots"] = {k: {kk: vv for kk, vv in d.items() if kk != "series"}
                           for k, d in r["robots"].items()}
        entry["frame_times_s"] = [round(float(x), 4) for x in r["t"]]
        doc["techniques"][tech] = entry
    with open(OUT_JSON, "w") as f:
        json.dump(doc, f, indent=1)
    return doc


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    colors = ["#2e7d32", "#f9a825", "#c62828", "#9e9e9e"]
    cmap = ListedColormap(colors)
    code = {c: i for i, c in enumerate(CLASS_ORDER)}
    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(len(TECHS), 2, width_ratios=[5.5, 1.0], hspace=0.62,
                          wspace=0.04, left=0.06, right=0.885, top=0.93, bottom=0.035)
    for row, tech in enumerate(TECHS):
        r = results[tech]
        ax, axs = fig.add_subplot(gs[row, 0]), fig.add_subplot(gs[row, 1])
        t = r["t"]
        for key, c, ls in (("a", "tab:blue", "-"), ("b", "tab:orange", "--")):
            ax.plot(t, r["robots"][key]["series"]["margin_m"], ls, lw=1.1, color=c)
        ax.axhline(0.0, color="k", lw=0.9)
        ax.axhspan(0.0, BAND, color=colors[1], alpha=0.18)
        ax.set_ylabel("CoM margin (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        infa = r["robots"]["a"]["class_pct"]["INFEASIBLE"]
        infb = r["robots"]["b"]["class_pct"]["INFEASIBLE"]
        ax.set_title(f"{tech}  ({r['frames']} frames, {r['duration_s']:.2f} s) - "
                     f"INFEASIBLE {infa:.0f}% (a) / {infb:.0f}% (b)   "
                     f"[a = {r['role_a']}, b = {r['role_b']}]", fontsize=9, loc="left")
        if row == len(TECHS) - 1:
            ax.set_xlabel("time (s)", fontsize=8)
        else:
            ax.set_xticklabels([])
        arr = np.array([[code[c] for c in r["robots"][k]["series"]["class"]]
                        for k in ("a", "b")])
        axs.imshow(arr, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5,
                   extent=[t[0], t[-1], 0, 2], interpolation="nearest")
        axs.set_yticks([0.5, 1.5])
        axs.set_yticklabels(["a", "b"], fontsize=7)
        axs.set_xticks([])
        ax.set_xlim(t[0], t[-1])
        axs.set_xlim(t[0], t[-1])
        for key, (y0, y1) in (("a", (0.0, 1.0)), ("b", (1.0, 2.0))):
            st = r["robots"][key]["series"]["state"]
            for e in episodes(st == "airborne_float"):
                axs.axvspan(t[e["start_frame"]], t[e["end_frame"]],
                            ymin=y0 / 2.0, ymax=y1 / 2.0, facecolor="none",
                            edgecolor="k", alpha=0.75, hatch="////", lw=0.0)
        if row == 0:
            ax.legend(handles=[plt.Line2D([], [], color="tab:blue", label="robot a"),
                               plt.Line2D([], [], color="tab:orange", ls="--",
                                          label="robot b")],
                      fontsize=7, loc="lower right", ncol=2)
            proxies = [Patch(facecolor=colors[i], label=CLASS_ORDER[i])
                       for i in range(4)]
            proxies.append(Patch(facecolor="none", edgecolor="k", hatch="////",
                                 label="float: nothing touching"))
            axs.legend(handles=proxies, fontsize=6, loc="upper left",
                       bbox_to_anchor=(1.02, 1.0), title="class", title_fontsize=6)
    fig.suptitle("Static support-envelope audit - kinematic reference replay "
                 "(CoM vs foot support region, colour-banded by class)\n"
                 f"footprint hull of the scene's sole contact patches; margin band "
                 f"{BAND} m; grey = airborne; black dots = reference floats (nothing "
                 f"touching the mat);\nmargin < 0 = CoM outside the support region: no "
                 f"ankle torque can hold that frame statically", fontsize=11)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def self_check(model, results, doc, checks):
    assert model.nq == 72 and model.nu == 58, (model.nq, model.nu)
    mass = results["STANCE"]["robots"]["a"]["mass_kg"]
    assert abs(mass - 33.34) < 0.05, mass
    for tech in TECHS:
        r = results[tech]
        for key in ("a", "b"):
            d = r["robots"][key]
            assert sum(d["class_counts"].values()) == r["frames"], tech
            assert d["standing_frames"] + d["ground_frames"] == r["frames"]
            st = d["support_states"]
            assert sum(st.values()) == r["frames"], (tech, key, st)
            assert d["airborne_frames"] == (st["airborne_ground"] +
                                            st["airborne_float"])
            assert d["airborne_frames"] == d["class_counts"]["AIRBORNE"]
            assert all(0.0 <= d["class_pct"][c] <= 100.0 for c in CLASS_ORDER)
            # reference-integrity consistency
            assert 0 <= d["frames_no_foot_2cm"] <= r["frames"]
            assert abs(d["pct_no_foot_2cm"] * r["frames"] / 100.0 -
                       d["frames_no_foot_2cm"]) < 1.0
            assert d["longest_float_s"] <= r["duration_s"] + r["dt"] + 1e-9
            assert d["longest_no_foot_2cm_s"] <= r["duration_s"] + r["dt"] + 1e-9
            assert all(0 < f < r["frames"] for f in d["seam_outlier_frames"])
            assert d["max_sole_penetration_m"] >= 0.0
            if d["max_sole_penetration_m"] > 0:
                assert d["max_sole_penetration_s"] is not None
    verdicts = {}
    for tech in TECHS:
        v = results[tech]["verdict"]
        e = v["evidence"]
        assert v["verdict"] in ("REF_INVALID", "REF_VALID_ADJUSTABLE",
                                "REF_VALID_DYNAMIC"), (tech, v["verdict"])
        assert v["action"] in ("REGENERATE", "ADJUST", "STABILISE")
        assert v["why"] and len(results[tech]["npz"]["sha256"]) == 16
        assert set(e) >= {"longest_float_s", "max_sole_penetration_m",
                          "n_seam_outliers", "stand_feasible_pct", "frames"}
        float_bad = (e["longest_float_s"] >= FLOAT_SUSPECT_S
                     and e["median_lowest_site_z"] > FLOAT_MEDIAN_Z)
        deep_pen = (e["max_sole_penetration_m"] >= PENETRATION_REFUSE_M)
        hard_jump = (e["max_dq_rad"] > SEAM_JUMP_FACTOR * SEAM_LIMITS["dq_rad"]
                     or e["max_dp_base_m"] > SEAM_JUMP_FACTOR * SEAM_LIMITS["dp_m"]
                     or e["max_drot_base_rad"] > SEAM_JUMP_FACTOR * SEAM_LIMITS["drot_rad"]
                     or e["max_dfoot_m"] > SEAM_JUMP_FACTOR * FOOT_TELEPORT_M)
        if v["verdict"] == "REF_INVALID":
            localized = (e["frames_penetration_over_5cm"] <=
                         max(3, int(0.02 * e["frames"])))
            assert float_bad or hard_jump or (deep_pen and not localized), (tech, e)
        else:
            assert not float_bad and not hard_jump, (tech, e)
        verdicts[tech] = f"{v['verdict']}/{v['action']}"
    # region-geometry unit checks
    sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert abs(region_margin(np.array([0.5, 0.5]), sq) - 0.5) < 1e-9
    assert abs(region_margin(np.array([1.5, 0.5]), sq) + 0.5) < 1e-9
    assert abs(region_margin(np.array([0.0, 2.0]), sq) + 1.0) < 1e-9
    assert region_margin(np.array([0.0, 0.0]), [[0.0, 0.0], [1.0, 0.0]]) == 0.0
    assert region_margin(np.array([1.0, 1.0]), []) is None
    assert convex_hull([[0, 0], [1, 0], [2, 0]]).shape == (2, 2)
    assert convex_hull([[0, 0]])[0].tolist() == [0.0, 0.0]
    # min-max LP unit checks: point contact at the CoM -> zero torque; CoM outside -> infeasible
    assert abs(minmax_ankle_torque(327.0, np.array([0.0, 0.0]),
                                   [np.array([[0.0, 0.0]])], [np.array([0.0, 0.0])])) < 1e-6
    assert minmax_ankle_torque(327.0, np.array([1.0, 0.0]),
                               [np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5],
                                          [-0.5, 0.5]])],
                               [np.array([0.0, 0.0])]) is None
    t_off = minmax_ankle_torque(327.0, np.array([0.2, 0.0]),
                                [np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5],
                                           [-0.5, 0.5]])], [np.array([0.0, 0.0])])
    assert abs(t_off - 327.0 * 0.2) < 327.0 * 0.2 * 0.05, t_off
    assert checks["lp_mismatch_frames"] == 0
    assert os.path.getsize(OUT_JSON) > 1000
    assert os.path.getsize(OUT_PNG) > 20000
    print("self-check: OK - class counts sum to frame counts; region geometry exact; "
          f"min-max LP unit cases pass; LP feasibility agrees with the geometric class "
          f"on all {checks['lp_checked_frames']} non-airborne frames "
          f"(mismatches {checks['lp_mismatch_frames']}); integrity counts and verdict "
          "rules consistent; 7 techniques x 2 robots; JSON + PNG written")


def lp_consistency(results):
    """The min-max LP must be infeasible exactly when the CoM is outside the hull."""
    checked = mismatch = 0
    for tech in TECHS:
        for key in ("a", "b"):
            s = results[tech]["robots"][key]["series"]
            for i in range(len(s["t"])):
                if s["class"][i] == "AIRBORNE":
                    continue
                checked += 1
                lp_feasible = np.isfinite(s["tau_nm"][i])
                geom_feasible = s["class"][i] != "INFEASIBLE"
                if lp_feasible != geom_feasible:
                    mismatch += 1
    return {"lp_checked_frames": checked, "lp_mismatch_frames": mismatch}


def main():
    model = mujoco.MjModel.from_xml_path(SCENE)
    results = {t: analyse_technique(model, t) for t in TECHS}
    head = headline(results)
    print_table(results)
    print_integrity(results)
    print("--- 3 deepest standing support violations (most negative margin) ---")
    for e in head["deepest_violations"]:
        print(f"  {e['technique']:11s} robot {e['robot']} frame {e['frame']:4d} "
              f"t={e['t']:6.2f}s margin {e['margin_m']:+.3f} m "
              f"tau_pend {e['tau_pendulum_nm']:.1f} Nm")
    print("--- 3 deepest support violations, any phase (ground phases included) ---")
    for e in head["deepest_violations_any"]:
        print(f"  {e['technique']:11s} robot {e['robot']} frame {e['frame']:4d} "
              f"t={e['t']:6.2f}s margin {e['margin_m']:+.3f} m pelvis_z "
              f"{e['pelvis_z']:.2f} m tau_pend {e['tau_pendulum_nm']:.1f} Nm")
    print(f"aggregate: {head['feasible_pct_aggregate']:.1f}% of all robot-frames and "
          f"{head['standing_feasible_pct_aggregate']:.1f}% of standing robot-frames have "
          "the CoM inside the foot support region")
    print("--- 3 highest static ankle-torque demands (pendulum model) ---")
    for e in head["highest_pendulum_demand"]:
        print(f"  {e['technique']:11s} robot {e['robot']} frame {e['frame']:4d} "
              f"t={e['t']:6.2f}s tau_pend {e['tau_pendulum_nm']:.1f} Nm "
              f"(limit {ANKLE_LIMIT_NM:.0f})")
    print("--- 3 highest min-max ankle demands (best possible load sharing) ---")
    for e in head["highest_minmax_demand"]:
        print(f"  {e['technique']:11s} robot {e['robot']} frame {e['frame']:4d} "
              f"t={e['t']:6.2f}s tau_minmax {e['tau_minmax_nm']:.1f} Nm "
              f"(limit {ANKLE_LIMIT_NM:.0f})")
    checks = lp_consistency(results)
    doc = write_json(results, head, checks)
    plot(results, OUT_PNG)
    self_check(model, results, doc, checks)


if __name__ == "__main__":
    sys.exit(main())
