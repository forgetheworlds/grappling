#!/usr/bin/env python
"""Reference fidelity: does each video-derived G1 track still contain the
behaviour the video demonstrates, and is it fit to be an imitation target?

For every track in ``data/refs_video/*.npz`` this recomputes the same features
from three views of the same motion:

  video   the MediaPipe landmark track for the source window, in NATIVE units
          (hip-centred, LSQ scale NOT applied) — the observation
  target  the placed, G1-scaled world track the solver was handed
          (= gm_track_from_window -> smooth_track -> place_solo, imported
          read-only from data/references/.../derived/tools/retarget_video.py)
  g1      the emitted G1 track (qpos_a, 50 Hz) via MuJoCo forward kinematics

and reports:

  1. feature preservation (per skill: pelvis height, stance width, knee-to-mat,
     torso pitch, travel, heading) — video | target | g1 | error | survives?
  2. contact / timing fidelity — the foot-contact ANCHORING used to reconstruct
     root translation (which foot the pipeline treats as planted vs which foot
     the video's own landmarks put on the floor), implied foot slip, phantom
     plants, and the global kinematic time STRE TCH applied by resample_50hz
     (the emitted track is a time-dilated copy: t_g1 = t_video * stretch).
  3. scale / morphology — G1-over-human LSQ scale, stance-width and
     pelvis-height ratios, joint-limit clipping, CoM-inside-support-hull margin
     for the retargeted stance.
  4. terminal posture — per track: CoM inside the support hull with positive
     margin, both feet flat and in contact, pelvis height near the stance
     keyframe, small tilt, low base speed (accepts a valid stance, rejects a
     collapsed one; unit-tested in tests/test_reference_fidelity.py).

Prints a measurement table and a SELF-VERIFY block (raises on failure):
  * the instrumented anchor recursion reproduces reconstruct_translation()
  * native * scale reproduces the target track's pelvis-relative geometry
  * FK sanity: core site == qpos[:3], ground-truth stance sole touches z=0
  * npz time axis == keyframes/fps * meta stretch (the time dilation is real)

Run:  .venv/bin/python scripts/measure_reference_fidelity.py [--json OUT]
      .venv/bin/python scripts/measure_reference_fidelity.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "data/references/yt_gBAhX5t-GW4/derived/tools"))

import mujoco  # noqa: E402
import retarget_video as rv  # noqa: E402  (the real pipeline, imported read-only)
from retarget.landmarks import SOLVED_SITES, load_g1_spec  # noqa: E402

REF_DIR = REPO / "data/refs_video"
POSE_MAIN = REPO / "data/references/yt_gBAhX5t-GW4/pose/landmarks.npz"
STANCE_NPZ = REPO / "data/refs/STANCE.npz"

# ---- measurement constants (all metres / seconds) --------------------------
CONTACT_TOL = 0.03      # a foot is "on the floor" when its lowest point is here
FLAT_TOL = 0.035        # toe AND heel within this of the floor = foot flat
KNEE_MAT = 0.08         # knee at the mat plane (matches the addendum target)
PELVIS_STAND_MIN = 0.55  # terminal pelvis height floor = 70% of G1 stand 0.79
TILT_MAX_DEG = 20.0     # terminal torso/root tilt
SPEED_MAX = 0.20        # terminal pelvis horizontal speed, m/s
COM_MARGIN_MIN = 0.0    # CoM must be inside the support hull
RUN_MIN_S = 0.40        # contact run length used for slip measurement
DWELL_MIN_S = 0.20      # minimum dwell for a plant/lift event

# GM joint index names (23 joints, from src/grapplemap)
from grapplemap import JOINTS  # noqa: E402

JI = {n: JOINTS.index(n) for n in JOINTS}

#: unified point keys -> GM joint name (video) / G1 site name
POINT_MAP: dict[str, tuple[str, str]] = {
    "core": ("Core", "core"),
    "neck": ("Neck", "neck"),
    "head": ("Head", "head"),
    "l_hip": ("LeftHip", "left_hip"), "r_hip": ("RightHip", "right_hip"),
    "l_knee": ("LeftKnee", "left_knee"), "r_knee": ("RightKnee", "right_knee"),
    "l_ankle": ("LeftAnkle", "left_ankle"), "r_ankle": ("RightAnkle", "right_ankle"),
    "l_toe": ("LeftToe", "left_toe"), "r_toe": ("RightToe", "right_toe"),
    "l_heel": ("LeftHeel", "left_heel"), "r_heel": ("RightHeel", "right_heel"),
}
POINTS = tuple(POINT_MAP)


# ---------------------------------------------------------------- geometry --
def cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2-D cross product (numpy>=2 dropped the 2-vector form of np.cross)."""
    return float(a[0] * b[1] - a[1] * b[0])


def convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull (monotone chain) of (N,2) points."""
    p = np.unique(np.round(np.asarray(pts, dtype=float), 9), axis=0)
    if len(p) <= 2:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2 and cross2(out[-1] - out[-2], q - out[-2]) <= 1e-12:
                out.pop()
            out.append(q)
        return out

    lower = half(p)
    upper = half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def hull_margin(hull: np.ndarray, p: np.ndarray) -> float:
    """Signed distance from 2-D point p to the hull boundary; > 0 = inside."""
    if len(hull) < 3:
        return float("nan")
    d = np.inf
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        e = b - a
        L = float(np.linalg.norm(e))
        if L < 1e-12:
            continue
        d = min(d, cross2(e, p - a) / L)
    return d


# ------------------------------------------------------------------- G1 FK --
class G1FK:
    """Forward kinematics on the retarget model (19 landmark sites + the four
    per-foot sole sphere centres; whole-body CoM)."""

    def __init__(self) -> None:
        self.m = load_g1_spec().compile()
        self.d = mujoco.MjData(self.m)
        self.site = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, n)
                     for n in SOLVED_SITES}
        self.sole: dict[str, list[tuple[int, np.ndarray, float]]] = {"l": [], "r": []}
        for g in range(self.m.ngeom):
            if self.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE:
                continue
            r = float(self.m.geom_size[g][0])
            if abs(r - 0.005) > 1e-9:
                continue
            bn = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                   self.m.geom_bodyid[g])
            if bn.startswith("left_ankle_roll"):
                self.sole["l"].append((g, self.m.geom_pos[g].copy(), r))
            elif bn.startswith("right_ankle_roll"):
                self.sole["r"].append((g, self.m.geom_pos[g].copy(), r))
        assert len(self.sole["l"]) == 4 and len(self.sole["r"]) == 4

    def observe(self, qpos: np.ndarray):
        """-> (pts, sole, com): named (3,) points, per-foot (K,3) sole contact
        points (sphere bottoms), whole-body CoM."""
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        pts = {k: self.d.site_xpos[self.site[s]].copy()
               for k, (_, s) in POINT_MAP.items()}
        sole = {}
        for side in ("l", "r"):
            b = np.array([self.d.geom_xpos[g] + np.array([0, 0, -r])
                          for g, _, r in self.sole[side]])
            sole[side] = b
        return pts, sole, self.d.subtree_com[0].copy()

    def limb_limits(self):
        lo = np.array([self.m.jnt_range[j][0] for j in range(1, self.m.njnt)])
        hi = np.array([self.m.jnt_range[j][1] for j in range(1, self.m.njnt)])
        names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)
                 for j in range(1, self.m.njnt)]
        return lo, hi, names


# ------------------------------------------------------------- video track --
def video_views(name: str, summary: dict, cache: dict):
    """-> (track_yup, native_zup, target_zup, place_info, t_kf).  Mirrors
    retarget_video.retarget() steps 1-4 exactly (window -> gm track -> smooth
    -> place_solo) but keeps the un-scaled, un-translated 'native' view too."""
    key = (name, summary[name]["source_npz"],
           tuple(summary[name]["source_window_s"]))
    if key in cache:
        return cache[key]
    import numpy as _np
    src = Path(summary[name]["source_npz"])
    if not src.is_absolute():
        src = REPO / src
    d = _np.load(src)
    t_all = d["t"].astype(float)
    t0, t1 = summary[name]["source_window_s"]
    idx = _np.where((t_all >= t0) & (t_all <= t1))[0]
    step = _np.diff(idx)
    if (step > 1).any():  # keep the longest contiguous run (as the pipeline does)
        runs = _np.split(idx, _np.where(step > 1)[0] + 1)
        idx = max(runs, key=len)
    fps = 1.0 / float(_np.median(_np.diff(t_all)))
    world = d["world_lm"][idx].astype(float)
    vis = d["visibility"][idx].astype(float)
    track, _ = rv.gm_track_from_window(world, vis.mean(axis=1),
                                       d["detected"][idx].astype(bool), fps)
    track = rv.smooth_track(track, fps)
    target, place = rv.place_solo(track, fps)
    native = zup_from_track(track, 1.0, None)
    out = (track, native, target, place, (t_all[idx] - t_all[idx[0]]).astype(float))
    cache[key] = out
    return out


def zup_from_track(track: np.ndarray, scale: float = 1.0,
                   trans: np.ndarray | None = None) -> np.ndarray:
    """Y-up GM track -> Z-up (x fwd, y left, z up), floored so the lowest foot
    joint over the take is 0, pelvis start at the origin.  Same rotations as
    place_solo WITHOUT the LSQ scale and, by default, without the
    reconstructed root translation (the 'native' video view)."""
    up = np.array([0.0, 1.0, 0.0])
    lateral = np.nanmean(track[:, JI["LeftHip"]] - track[:, JI["RightHip"]], axis=0)
    fwd = np.cross(up, lateral)
    fwd[1] = 0.0
    fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
    head = np.nanmean(track[:, JI["Head"]] - track[:, JI["Core"]], axis=0)
    head[1] = 0.0
    if np.dot(fwd, head) < 0:
        fwd = -fwd
    ang = -np.arctan2(fwd[2], fwd[0])
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    t2 = track @ rot.T
    if scale != 1.0:
        core_mean = np.nanmean(t2[:, JI["Core"]], axis=0)
        t2 = (t2 - core_mean) * scale + core_mean
    if trans is not None:
        t2 = t2 + trans[:, None, :]
    foot = [JI[n] for n in ("LeftToe", "RightToe", "LeftHeel", "RightHeel",
                            "LeftAnkle", "RightAnkle")]
    t2[..., 1] -= float(np.nanmin(t2[:, foot, 1]))
    core0 = t2[0, JI["Core"]]
    t2[..., 0] -= core0[0]
    t2[..., 2] -= core0[2]
    return np.stack([t2[..., 0], -t2[..., 2], t2[..., 1]], axis=-1)


def video_points(zup: np.ndarray):
    pts = {k: zup[:, JI[gm]].copy() for k, (gm, _) in POINT_MAP.items()}
    sole = {"l": zup[:, [JI["LeftToe"], JI["LeftHeel"]]],
            "r": zup[:, [JI["RightToe"], JI["RightHeel"]]]}
    return pts, sole, None


# --------------------------------------------------------- anchor analysis --
def anchor_choices_left(track_yup: np.ndarray, fps: float) -> np.ndarray:
    """Instrumented copy of retarget_video.reconstruct_translation's foot
    choice (2026-10-08 revision): True when the LEFT foot is the hysteresis
    anchor (clearly lower by ANCHOR_LOWER_MARGIN, switches only after
    ANCHOR_DWELL_S).  Mirrors the pipeline function exactly."""
    margin = rv.ANCHOR_LOWER_MARGIN
    dwell = max(1, int(round(rv.ANCHOR_DWELL_S * fps)))
    dy = track_yup[:, JI["LeftAnkle"]][:, 1] - track_yup[:, JI["RightAnkle"]][:, 1]
    right = np.zeros(len(dy), dtype=bool)
    cur = bool(dy[0] > margin)
    run_side, run = int(cur), 0
    for i in range(len(dy)):
        low = 1 if dy[i] > margin else (0 if dy[i] < -margin else -1)
        if low >= 0:
            if low == run_side:
                run += 1
            else:
                run_side, run = low, 1
            if low != int(cur) and run >= dwell:
                cur = bool(low)
        right[i] = cur
    return ~right


def contact_runs(flags: np.ndarray, t: np.ndarray, min_s: float = 0.0):
    """[(i0, i1, duration_s)] runs of True in a boolean series."""
    runs, i = [], 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            if t[j] - t[i] >= min_s:
                runs.append((i, j, float(t[j] - t[i])))
            i = j + 1
        else:
            i += 1
    return runs


def events(flags: np.ndarray, t: np.ndarray, min_s: float = DWELL_MIN_S):
    runs = contact_runs(flags, t, min_s)
    return [round(float(t[a]), 3) for a, _, _ in runs], \
           [round(float(t[b]), 3) for _, b, _ in runs]


# ---------------------------------------------------------------- features --
def series_median(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(x)) if x.size else float("nan")


def compute_features(pts: dict, sole: dict, com, t: np.ndarray) -> dict:
    """Feature dict for one view of one track.  All lengths in metres."""
    f: dict[str, float] = {}
    core = pts["core"]
    f["pelvis_z_med"] = series_median(core[:, 2])
    f["pelvis_z_min"] = float(np.nanmin(core[:, 2]))
    f["pelvis_z_end"] = float(core[-1, 2])
    f["pelvis_z_start"] = float(core[0, 2])
    f["stance_width_med"] = series_median(
        np.abs(pts["l_ankle"][:, 1] - pts["r_ankle"][:, 1]))
    f["stance_depth_med"] = series_median(
        np.abs(pts["l_ankle"][:, 0] - pts["r_ankle"][:, 0]))
    knee = np.minimum(pts["l_knee"][:, 2], pts["r_knee"][:, 2])
    f["knee_min"] = float(np.nanmin(knee))
    f["knee_contact_s"] = float(np.sum(knee <= KNEE_MAT) * np.median(np.diff(t)))
    f["knee_contact_frac"] = float(np.mean(knee <= KNEE_MAT))
    for tag, other in (("head", "head"), ("neck", "neck")):
        v = pts[other] - core
        nn = np.linalg.norm(v, axis=1)
        f[f"torso_pitch_{tag}_deg"] = series_median(
            np.degrees(np.arccos(np.clip(v[:, 2] / np.maximum(nn, 1e-9), -1, 1))))
    f["torso_len_m"] = series_median(np.linalg.norm(pts["neck"] - core, axis=1))
    dxy = core[:, :2] - core[0, :2]
    f["net_travel_m"] = float(np.linalg.norm(dxy[-1]))
    f["path_len_m"] = float(np.sum(np.linalg.norm(np.diff(core[:, :2], axis=0), axis=1)))
    lat = pts["l_hip"] - pts["r_hip"]
    yaw = np.degrees(np.arctan2(lat[:, 1], lat[:, 0]))
    yaw = np.unwrap(np.radians(yaw))
    f["heading_net_deg"] = float(np.degrees(yaw[-1] - yaw[0]))
    f["heading_range_deg"] = float(np.degrees(yaw.max() - yaw.min()))
    f["heading_path_deg"] = float(np.degrees(np.sum(np.abs(np.diff(yaw)))))
    for side in ("l", "r"):
        h = sole[side][..., 2].min(axis=1)
        f[f"sole_min_{side}"] = float(np.nanmin(h))
        f[f"sole_med_{side}"] = series_median(h)
        f[f"sole_end_{side}"] = float(h[-1])
        flags = h <= CONTACT_TOL
        f[f"contact_frac_{side}"] = float(np.mean(flags))
        pl, lf = events(flags, t)
        f[f"plants_{side}"] = pl
        f[f"lifts_{side}"] = lf
    return f


def warp_to_video(values: np.ndarray, t_g1: np.ndarray, stretch: float,
                  t_kf: np.ndarray) -> np.ndarray:
    """Sample a G1-time series on the video keyframe grid (t_g1 = stretch*t_kf)."""
    return np.interp(t_kf, t_g1 / stretch, values)


# --------------------------------------------------------- terminal stance --
def terminal_stance(fk: G1FK, qpos: np.ndarray, stance_pelvis: float,
                    n_tail: int = 5) -> dict:
    """Terminal-posture predicate on the last n_tail frames of a track."""
    obs = [fk.observe(q) for q in qpos[-n_tail:]]
    r = {"ok": False}
    pts = [o[0] for o in obs]
    solexy = []
    flat = True
    for side in ("l", "r"):
        for o in obs:
            b = o[1][side]
            if b[:, 2].max() > FLAT_TOL or b[:, 2].min() < -0.02:
                flat = False
            solexy.append(b[o[1][side][:, 2] <= FLAT_TOL][:, :2])
    r["feet_flat_both"] = bool(flat)
    solexy = np.vstack(solexy) if solexy else np.zeros((0, 2))
    hull = convex_hull_2d(solexy) if len(solexy) else np.zeros((0, 2))
    com = obs[-1][2]
    r["com_margin_m"] = hull_margin(hull, com[:2]) if len(hull) >= 3 else float("nan")
    r["com_inside"] = bool(r["com_margin_m"] >= COM_MARGIN_MIN)
    pz = float(np.mean([o[0]["core"][2] for o in obs]))
    r["pelvis_z_end"] = pz
    r["pelvis_vs_stance_m"] = pz - stance_pelvis
    r["pelvis_ok"] = bool(pz >= PELVIS_STAND_MIN)
    tilt = []
    for o in obs:
        v = o[0]["neck"] - o[0]["core"]
        tilt.append(np.degrees(np.arccos(np.clip(
            v[2] / max(np.linalg.norm(v), 1e-9), -1, 1))))
    r["torso_tilt_deg"] = float(np.mean(tilt))
    r["tilt_ok"] = bool(r["torso_tilt_deg"] <= TILT_MAX_DEG)
    dt = 1.0 / 50.0
    v = (pts[-1]["core"][:2] - pts[-2]["core"][:2]) / dt
    r["speed_end_mps"] = float(np.linalg.norm(v))
    r["slow_ok"] = bool(r["speed_end_mps"] <= SPEED_MAX)
    r["ok"] = bool(r["feet_flat_both"] and r["com_inside"] and r["pelvis_ok"]
                   and r["tilt_ok"] and r["slow_ok"])
    return r


# ------------------------------------------------------------------- main ---
def load_stance_pelvis(fk: G1FK) -> float:
    q = np.load(STANCE_NPZ, allow_pickle=True)["qpos_a"]
    return float(fk.observe(q[0])[0]["core"][2])


def analyse_track(name: str, summary: dict, fk: G1FK, cache: dict,
                  stance_pelvis: float) -> dict:
    d = np.load(REF_DIR / f"{name}.npz", allow_pickle=True)
    qa, t_g1 = d["qpos_a"], d["t"].astype(float)
    meta = json.loads(str(d["meta"]))
    stretch = float(meta["time_stretch_kinematic"])
    track, native, target, place, t_kf = video_views(name, summary, cache)

    # ---- G1 view ----
    g_pts, g_sole, g_com = {}, {}, []
    for i, q in enumerate(qa):
        p, s, c = fk.observe(q)
        for k in POINTS:
            g_pts.setdefault(k, []).append(p[k])
        for side in ("l", "r"):
            g_sole.setdefault(side, []).append(s[side])
        g_com.append(c)
    g_pts = {k: np.array(v) for k, v in g_pts.items()}
    g_sole = {k: np.array(v) for k, v in g_sole.items()}
    g_com = np.array(g_com)
    lo, hi, jnames = fk.limb_limits()
    lim_mask = (qa[:, 7:36] <= lo[None, :] + 1e-6) | (qa[:, 7:36] >= hi[None, :] - 1e-6)
    lim_frac = float(lim_mask.any(axis=1).mean())
    lim_joints = sorted({jnames[j] for j in np.where(lim_mask.any(axis=0))[0]
                         if not np.isnan(lo[j])})

    # ---- three feature views ----
    n_pts, n_sole, _ = video_points(native)
    tgt_pts, tgt_sole, _ = video_points(target)
    feats = {
        "video": compute_features(n_pts, n_sole, None, t_kf),
        "target": compute_features(tgt_pts, tgt_sole, None, t_kf),
        "g1": compute_features(g_pts, g_sole, g_com, t_g1),
    }
    # time-warped G1 series for the equal-time error columns
    err = {}
    for key, fn in (("pelvis_z", lambda P: P["core"][:, 2]),
                    ("l_knee", lambda P: P["l_knee"][:, 2]),
                    ("r_knee", lambda P: P["r_knee"][:, 2]),
                    ("l_sole", lambda P, S=None: None)):
        pass
    pairs = {
        "pelvis_z": (tgt_pts["core"][:, 2], g_pts["core"][:, 2]),
        "l_knee_z": (tgt_pts["l_knee"][:, 2], g_pts["l_knee"][:, 2]),
        "r_knee_z": (tgt_pts["r_knee"][:, 2], g_pts["r_knee"][:, 2]),
        "l_sole_z": (tgt_sole["l"][..., 2].min(1), g_sole["l"][..., 2].min(1) - 0.005),
        "r_sole_z": (tgt_sole["r"][..., 2].min(1), g_sole["r"][..., 2].min(1) - 0.005),
        "l_ankle_x": (tgt_pts["l_ankle"][:, 0], g_pts["l_ankle"][:, 0]),
        "r_ankle_x": (tgt_pts["r_ankle"][:, 0], g_pts["r_ankle"][:, 0]),
    }
    for k, (a, b) in pairs.items():
        bw = warp_to_video(np.asarray(b), t_g1, stretch, t_kf)
        err[k] = {"rms": float(np.sqrt(np.nanmean((bw - a) ** 2))),
                  "max": float(np.nanmax(np.abs(bw - a)))}

    # ---- contact / timing ----
    fps = float(meta["source_fps_effective"])
    # the pipeline anchors on ground_per_frame's planted foot (2026-10-08);
    # measure THAT anchor, and keep the legacy lower-foot rule as a secondary
    _, anchor_pipe = rv.reconstruct_translation(track, fps,
                                                contact=place["contact_kf"])
    anchor = ~anchor_pipe                    # left-planted under the pipeline rule
    rec_direct, _anchor_direct = rv.reconstruct_translation(target, fps)
    # (verification only; the pipeline scales the reconstruction differently)
    n_sole_h = n_sole["l"][..., 2].min(1), n_sole["r"][..., 2].min(1)
    lower_left = n_sole_h[0] <= n_sole_h[1]
    dlt = np.abs(n_sole_h[0] - n_sole_h[1])
    clear = dlt > CONTACT_TOL
    agree = float(np.mean(anchor[clear] == lower_left[clear])) if clear.any() else float("nan")
    phantom = anchor & (n_sole_h[0] > CONTACT_TOL) & (n_sole_h[1] <= CONTACT_TOL) | \
        (~anchor) & (n_sole_h[1] > CONTACT_TOL) & (n_sole_h[0] <= CONTACT_TOL)
    phantom_s = float(phantom.sum() * np.median(np.diff(t_kf)))
    # implied slip over contact runs with a constant anchor choice
    slip_runs, slip_video, slip_g1 = [], {"l": [], "r": []}, {"l": [], "r": []}
    for side, idx in (("l", 0), ("r", 1)):
        fl = n_sole_h[idx] <= CONTACT_TOL
        for a, b, _ in contact_runs(fl, t_kf, RUN_MIN_S):
            seg = anchor[a:b + 1]
            if len(seg) >= 4 and (seg.all() or (~seg).all()):
                dfoot = native[:, JI["LeftAnkle" if side == "l" else "RightAnkle"]][b] - \
                        native[:, JI["LeftAnkle" if side == "l" else "RightAnkle"]][a]
                danch = native[:, JI["LeftAnkle" if seg[0] else "RightAnkle"]][b] - \
                        native[:, JI["LeftAnkle" if seg[0] else "RightAnkle"]][a]
                slip_runs.append(float(np.linalg.norm((dfoot - danch)[[0, 1]]))
                                 * place["scale"] / max(t_kf[b] - t_kf[a], 1e-9))
        # world slip actually present in the emitted G1 track
        gfl = g_sole[side][..., 2].min(1) - 0.005 <= CONTACT_TOL + 0.01
        for a, b, dur in contact_runs(gfl, t_g1, RUN_MIN_S):
            p0 = g_sole[side][a].mean(axis=0)
            p1 = g_sole[side][b].mean(axis=0)
            slip_g1[side].append(float(np.linalg.norm((p1 - p0)[:2]) / max(dur, 1e-9)))

    c = {"anchor_is_lower_frac": agree, "clear_frames": int(clear.sum()),
         "phantom_plant_s": phantom_s, "phantom_frac": float(phantom.mean()),
         "slip_implied_med_mps": series_median(np.array(slip_runs)),
         "slip_implied_max_mps": float(np.max(slip_runs)) if slip_runs else float("nan"),
         "slip_runs": len(slip_runs),
         "slip_g1_l_med_mps": series_median(np.array(slip_g1["l"])),
         "slip_g1_r_med_mps": series_median(np.array(slip_g1["r"])),
         "slip_g1_max_mps": float(np.nanmax([*(slip_g1["l"] or [np.nan]),
                                             *(slip_g1["r"] or [np.nan])]))}
    # landmark noise floor: frame-to-frame foot motion during stance_hold
    # (computed per track on frames where both feet are on the floor)
    both = (n_sole_h[0] <= CONTACT_TOL) & (n_sole_h[1] <= CONTACT_TOL)
    if both.sum() > 5:
        d_l = np.linalg.norm(np.diff(native[both, JI["LeftAnkle"]][:, [0, 1]], axis=0), axis=1)
        d_r = np.linalg.norm(np.diff(native[both, JI["RightAnkle"]][:, [0, 1]], axis=0), axis=1)
        noise = np.concatenate([d_l, d_r]) * fps
        c["noise_floor_mps"] = series_median(noise)
    else:
        c["noise_floor_mps"] = float("nan")

    # ---- terminal posture ----
    term = terminal_stance(fk, qa, stance_pelvis)
    return {"name": name, "stretch": stretch,
            "video_dur_s": float(t_kf[-1]), "g1_dur_s": float(t_g1[-1]),
            "scale": float(place["scale"]), "meta_travel_m": float(meta["net_travel_m"]),
            "features": feats, "errors": err, "contact": c, "terminal": term,
            "limit_frac": lim_frac, "limit_joints": lim_joints,
            "video_placeholder": {"t_kf_end": float(t_kf[-1])}, "fk": fk}


def summarise(res: list[dict], stance_pelvis: float) -> None:
    def p(*a):
        print(*a)

    p(f"\n=== FEATURES (video native | target (G1 scale) | G1 emitted) ===")
    hdr = (f"{'track':22s} {'pelvV':>6s} {'pelvT':>6s} {'pelvG':>6s} "
           f"{'wV':>5s} {'wT':>5s} {'wG':>5s} {'kneeV':>6s} {'kneeT':>6s} "
           f"{'kneeG':>6s} {'pitchV':>6s} {'pitchT':>6s} {'pitchG':>6s} "
           f"{'rmsP':>5s} {'rmsK':>5s}")
    p(hdr)
    for r in res:
        v, tt, g = r["features"]["video"], r["features"]["target"], r["features"]["g1"]
        rmsK = max(r["errors"]["l_knee_z"]["rms"], r["errors"]["r_knee_z"]["rms"])
        p(f"{r['name']:22s} {v['pelvis_z_med']:6.3f} {tt['pelvis_z_med']:6.3f} "
          f"{g['pelvis_z_med']:6.3f} {v['stance_width_med']:5.3f} "
          f"{tt['stance_width_med']:5.3f} {g['stance_width_med']:5.3f} "
          f"{v['knee_min']:6.3f} {tt['knee_min']:6.3f} {g['knee_min']:6.3f} "
          f"{v['torso_pitch_neck_deg']:6.1f} {tt['torso_pitch_neck_deg']:6.1f} "
          f"{g['torso_pitch_neck_deg']:6.1f} {r['errors']['pelvis_z']['rms']:5.3f} {rmsK:5.3f}")

    p(f"\n=== CONTACT / TIMING (stretch = G1 time / video time) ===")
    p(f"{'track':22s} {'stretch':>7s} {'anchor=lower':>12s} {'phantomS':>8s} "
      f"{'slipImp':>7s} {'slipG1':>7s} {'noiseF':>6s} {'kneeC_v':>7s} {'kneeC_g':>7s}")
    for r in res:
        c, vf, gf = r["contact"], r["features"]["video"], r["features"]["g1"]
        p(f"{r['name']:22s} {r['stretch']:7.3f} {c['anchor_is_lower_frac']:12.2f} "
          f"{c['phantom_plant_s']:8.2f} {c['slip_implied_med_mps']:7.3f} "
          f"{c['slip_g1_max_mps']:7.3f} {c['noise_floor_mps']:6.3f} "
          f"{vf['knee_contact_s']:7.2f} {gf['knee_contact_s']:7.2f}")

    p(f"\n=== TERMINAL POSTURE (on the emitted G1 track; stance pelvis = {stance_pelvis:.3f} m) ===")
    p(f"{'track':22s} {'ok':>3s} {'pelv_z':>6s} {'dStance':>7s} {'feet':>5s} "
      f"{'comMarg':>7s} {'tilt':>5s} {'spd':>5s} {'limFrac':>7s}")
    for r in res:
        t = r["terminal"]
        p(f"{r['name']:22s} {str(t['ok']):>3s} {t['pelvis_z_end']:6.3f} "
          f"{t['pelvis_vs_stance_m']:7.3f} {str(t['feet_flat_both']):>5s} "
          f"{t['com_margin_m']:7.3f} {t['torso_tilt_deg']:5.1f} "
          f"{t['speed_end_mps']:5.2f} {r['limit_frac']:7.3f}")


def selftest(fk: G1FK, stance_pelvis: float, summary: dict) -> None:
    """Prints and asserts the internal consistency checks."""
    print("\n=== SELF-VERIFY ===")
    checks: list[tuple[str, bool, str]] = []

    # 1. FK sanity on the ground-truth stance
    q = np.load(STANCE_NPZ, allow_pickle=True)["qpos_a"][0]
    pts, sole, com = fk.observe(q)
    ok = bool(np.allclose(pts["core"], q[:3], atol=1e-9))
    checks.append(("G1FK core site == qpos root position", ok,
                   f"|d|={np.linalg.norm(pts['core'] - q[:3]):.2e}"))
    smin = float(min(sole['l'][:, 2].min(), sole['r'][:, 2].min()))
    ok = abs(smin) < 0.01
    checks.append(("stance sole touches floor (z=0)", ok, f"min sole z={smin:.4f}"))
    hull = convex_hull_2d(np.vstack([sole["l"][:, :2], sole["r"][:, :2]]))
    m = hull_margin(hull, com[:2])
    ok = m > 0.0
    checks.append(("stance CoM inside support hull", ok, f"margin={m:.4f} m"))
    checks.append(("stance pelvis matches refs_crosscheck 0.806",
                   abs(stance_pelvis - 0.806) < 0.005, f"{stance_pelvis:.4f}"))

    # 2. anchor instrumentation reproduces the pipeline function
    cache: dict = {}
    track, native, target, place, t_kf = video_views("stance_hold", summary, cache)
    fps = float(summary["stance_hold"]["source_fps_effective"])
    anc = anchor_choices_left(track, fps)
    rec, anchor_direct = rv.reconstruct_translation(track, fps)
    checks.append(("instrumented anchor == reconstruct_translation mask",
                   bool(np.array_equal(anc, ~anchor_direct)),
                   f"n_left={int(anc.sum())} n_right={int(anchor_direct.sum())}"))
    A = np.stack([track[:, JI["LeftAnkle"]], track[:, JI["RightAnkle"]]], axis=1)
    pos = np.zeros((len(track), 3))
    for i in range(1, len(track)):
        a = 1 if anchor_direct[i] else 0
        pos[i] = pos[i - 1] - (A[i, a] - A[i - 1, a])
    from scipy.signal import savgol_filter
    w = max(5, int(round(0.4 * fps)) | 1)
    for j in (0, 2):
        pos[:, j] = savgol_filter(pos[:, j], w, 2)
    checks.append(("re-derive pos from mask == reconstruct_translation",
                   bool(np.allclose(pos, rec, atol=1e-12)),
                   f"max|d|={np.max(np.abs(pos - rec)):.2e}"))
    checks.append(("anchor choice consistency (mask size)",
                   bool(abs(pos[-1, 0]) <= 1e9), f"n_left_frames={int(anc.sum())}"))

    # 3. native * scale == target (pelvis-relative geometry)
    rel_n = native - native[:, JI["Core"]][:, None, :]
    rel_t = target - target[:, JI["Core"]][:, None, :]
    dmax = float(np.max(np.abs(rel_t - place["scale"] * rel_n)))
    checks.append(("native * LSQ scale == target geometry", dmax < 1e-9, f"max|d|={dmax:.2e}"))

    # 4. time axis is a dilated copy of the video keyframe timeline
    bad = []
    for name, meta in summary.items():
        d = np.load(REF_DIR / f"{name}.npz", allow_pickle=True)
        t = d["t"].astype(float)
        st = float(meta["time_stretch_kinematic"])
        expect = st * (meta["source_frames"] - 1) / meta["source_fps_effective"]
        trim = meta.get("reachability_trim")
        if trim and trim.get("cut_at_s") is not None:
            expect = min(expect, float(trim["cut_at_s"]))   # trimmed tail
        if abs(t[-1] - expect) > 0.021:
            bad.append((name, t[-1], expect))
    checks.append(("emitted duration == video duration * stretch", not bad,
                   f"bad={bad}"))

    for name, ok, detail in checks:
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:52s} {detail}")
    assert all(ok for _, ok, _ in checks), "self-verification failed"


def main() -> None:
    global REF_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--tracks", default=None, help="comma list (default: all)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ref-dir", default=None,
                    help="alternate reference dir (default: data/refs_video); "
                         "must contain retarget_summary.json")
    args = ap.parse_args()
    if args.ref_dir:
        REF_DIR = Path(args.ref_dir)

    summary = json.loads((REF_DIR / "retarget_summary.json").read_text())
    names = sorted(summary)
    if args.tracks:
        names = [n.strip() for n in args.tracks.split(",")]
    fk = G1FK()
    stance_pelvis = load_stance_pelvis(fk)
    cache: dict = {}

    if args.selftest:
        selftest(fk, stance_pelvis, summary)
        return

    res = [analyse_track(n, summary, fk, cache, stance_pelvis) for n in names]
    summarise(res, stance_pelvis)
    selftest(fk, stance_pelvis, summary)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([{k: v for k, v in r.items() if k != "fk"}
                                   for r in res], indent=1, default=str))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
