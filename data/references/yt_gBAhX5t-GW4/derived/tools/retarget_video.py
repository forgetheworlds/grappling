"""Retarget a monocular pose track (run_pose.py output) to the G1 using the
EXISTING retarget machinery from src/retarget (imported read-only).

Pipeline per transition window [t0, t1]:
  mediapipe 33 landmarks -> GM 23-joint track (Y-up, metres, hip-centred)
  -> light zero-phase smoothing (0.20 s Savitzky-Golay)
  -> Y-up to Z-up placement: rotate so the time-averaged facing maps to +x
     (the G1 faces +x), scale by the LSQ G1/human segment-ratio fit
     (retarget.world.estimate_player_scale), floor the feet at z=0,
     centre the time-averaged pelvis on the origin.
  -> solve_keyframes(targets, targets)   [existing solver; the same track is
     passed for both slots of the two-robot LSQ, robot A's solution is kept]
  -> resample_50hz                     [existing kinematic-limit retimer]

Emitted npz (single-robot; qpos_b is NOT meaningful, kept as zeros):
  qpos_a (T,36) float64   free joint + 29 joints, 50 Hz, model qpos order
  t (T,)                  seconds from window start
  technique (str)         transition name
  edges (int array)       empty (video-sourced, not a GrappleMap chain)
  landmark_rms (float)    weighted site RMS of the final trajectory
  meta (dict)             provenance: source npz, window, scale, facing,
                          fill fraction, ambiguities, solver settings

Run from repo root (project venv, needs mujoco+scipy):
  python data/references/yt_gBAhX5t-GW4/derived/tools/retarget_video.py \
      --npz .../pose/landmarks.npz --spec /tmp/transitions.json \
      --out-dir data/refs_video
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

from grapplemap import JOINTS  # noqa: E402
from retarget.solve import (landmark_rms_final, resample_50hz,  # noqa: E402
                            solve_keyframes)
from retarget.world import estimate_player_scale  # noqa: E402

# mediapipe pose landmark indices
MP = {"nose": 0, "l_ear": 7, "r_ear": 8, "l_sho": 11, "r_sho": 12,
      "l_elb": 13, "r_elb": 14, "l_wri": 15, "r_wri": 16,
      "l_hip": 23, "r_hip": 24, "l_kne": 25, "r_kne": 26,
      "l_ank": 27, "r_ank": 28, "l_hee": 29, "r_hee": 30,
      "l_toe": 31, "r_toe": 32,
      "l_fin": 17, "r_fin": 18, "l_idx": 19, "r_idx": 20,
      "l_thumb": 21, "r_thumb": 22}

J_MID = ("Core", "Neck", "Head")

# ---- grounding constants (2026-10-08 revision; all metres / seconds) --------
# Measured defect: the old take-level floor left emitted G1 soles hovering up
# to +0.111 m or penetrating -0.067 m (reports/2026-10-08/reference_fidelity.md
# §4).  These bound the per-frame, contact-aware replacement below.
GROUND_CONTACT_TOL = 0.05   # sole within this of the frame's lowest = planted
GROUND_LIFT_TOL = 0.08      # sole above the frame's lowest = lifted
GROUND_DWELL_S = 0.13       # continuous plant period required to (re)plant
GROUND_FLOOR_PCT = 5.0      # floor-plane percentile of pooled sole heights
#: target-level offset clip.  The offset must be able to express a REAL crouch:
#: mediapipe landmarks are hip-centred, so a 0.30 m hip drop shows up as a
#: +0.30 m apparent sole rise that grounding must undo (a +0.12 clip left the
#: shot-entry crouch hovering; measured 2026-10-08).  0.60 m covers every
#: take's deepest posture while still bounding landmark blowups.
GROUND_OFFSET_MAX = 0.60
GROUND_OFFSET_MIN = -0.02
ANCHOR_LOWER_MARGIN = 0.02  # m: "clearly lower" hysteresis margin (anchor)
ANCHOR_DWELL_S = 0.20       # s: sole-planted advantage must persist to switch


def gm_track_from_window(world_hip: np.ndarray, vis: np.ndarray,
                         detected: np.ndarray, fps: float) -> tuple[np.ndarray, dict]:
    """(F,33,3) mediapipe world landmarks (hip-centred, y down) -> (F,23,3)
    Y-up GM-JOINTS track, with explicit gap filling.

    Returns (track, info). Frames with any missing keypoint are linear-interpolated
    in time per coordinate; the fraction of filled frames is reported, never hidden.
    """
    f = world_hip.shape[0]
    w = world_hip.astype(np.float64).copy()
    w[..., 1] *= -1.0     # y-up
    w[..., 2] *= -1.0     # +z toward the camera (rotation, handedness kept)

    def pt(name: str) -> np.ndarray:
        return w[:, MP[name]]

    def m(*names: str) -> np.ndarray:
        return np.mean([pt(n) for n in names], axis=0)

    track = np.zeros((f, len(JOINTS), 3))
    track[:, JOINTS.index("LeftToe")] = pt("l_toe")
    track[:, JOINTS.index("RightToe")] = pt("r_toe")
    track[:, JOINTS.index("LeftHeel")] = pt("l_hee")
    track[:, JOINTS.index("RightHeel")] = pt("r_hee")
    track[:, JOINTS.index("LeftAnkle")] = pt("l_ank")
    track[:, JOINTS.index("RightAnkle")] = pt("r_ank")
    track[:, JOINTS.index("LeftKnee")] = pt("l_kne")
    track[:, JOINTS.index("RightKnee")] = pt("r_kne")
    track[:, JOINTS.index("LeftHip")] = pt("l_hip")
    track[:, JOINTS.index("RightHip")] = pt("r_hip")
    track[:, JOINTS.index("LeftShoulder")] = pt("l_sho")
    track[:, JOINTS.index("RightShoulder")] = pt("r_sho")
    track[:, JOINTS.index("LeftElbow")] = pt("l_elb")
    track[:, JOINTS.index("RightElbow")] = pt("r_elb")
    track[:, JOINTS.index("LeftWrist")] = pt("l_wri")
    track[:, JOINTS.index("RightWrist")] = pt("r_wri")
    # GM "Hand" ~ mid-hand: mean of index+pinky+thumb MCP-ish landmarks
    track[:, JOINTS.index("LeftHand")] = m("l_idx", "l_fin", "l_thumb")
    track[:, JOINTS.index("RightHand")] = m("r_idx", "r_fin", "r_thumb")
    track[:, JOINTS.index("LeftFingers")] = m("l_idx", "l_fin", "l_thumb")
    track[:, JOINTS.index("RightFingers")] = m("r_idx", "r_fin", "r_thumb")
    track[:, JOINTS.index("Core")] = 0.5 * (pt("l_hip") + pt("r_hip"))
    track[:, JOINTS.index("Neck")] = m("l_sho", "r_sho")   # chest-top proxy
    track[:, JOINTS.index("Head")] = m("nose", "l_ear", "r_ear")

    # gap filling: any landmark key missing in a frame (detector failure)
    bad = ~np.isfinite(w).all(axis=(1, 2)) | (~detected)
    if bad.any():
        t = np.arange(f, dtype=np.float64) / fps
        good = ~bad
        for j in range(track.shape[1]):
            for ax in range(3):
                x = track[:, j, ax]
                ok = np.isfinite(x)
                if ok.sum() >= 2 and (~ok).any():
                    track[~ok, j, ax] = np.interp(t[~ok], t[ok], x[ok])
        track[~np.isfinite(track)] = 0.0
    info = {"n_frames": int(f), "gap_frames": int(bad.sum()),
            "gap_frac": float(bad.mean()),
            "mean_vis": float(np.mean(vis)) if vis.size else 0.0}
    return track, info


def smooth_track(track: np.ndarray, fps: float, win_s: float = 0.30) -> np.ndarray:
    from scipy.signal import savgol_filter
    f = max(3, int(round(win_s * fps)) | 1)
    out = track.copy()
    for j in range(track.shape[1]):
        out[:, j] = savgol_filter(track[:, j], f, 2, axis=0)
    return out


def smooth_qpos(qpos: np.ndarray, t_kf: np.ndarray,
                cutoff_hz: float = 2.5, order: int = 4) -> np.ndarray:
    """Zero-phase low-pass of the solved keyframes.

    Landmark jitter at 15 fps passes through the per-frame IK; without this the
    kinematic-limit retimer stretches the whole trajectory (measured on the
    stance-hold window: raw solved keyframes need a 3.4x stretch; 3.5 Hz ->
    3.26x, 2.5 Hz -> 1.82x; the residual is base-rotation noise, see the index
    doc 'known limitations'). Real motions here are <= 2.5 Hz (the fastest is
    the level-change drop at ~2.5 Hz).
    """
    from scipy.signal import butter, filtfilt
    from scipy.spatial.transform import Rotation
    fs = 1.0 / float(np.median(np.diff(t_kf)))
    b, a = butter(order, cutoff_hz, fs=fs)
    q = qpos.copy()
    for j in list(range(0, 3)) + list(range(7, 36)):
        q[:, j] = filtfilt(b, a, qpos[:, j])
    # base rotation: build a CONTINUOUS rotation-vector series from the
    # cumulative quaternion deltas (raw keyframes flip yaw representation by
    # ~360 deg between frames near the +/-pi boundary, which unwrap() does not
    # fix), filter it, then rebuild the quaternions by integration.
    quat = qpos[:, 3:7].copy()
    for i in range(1, len(quat)):
        if float(np.dot(quat[i], quat[i - 1])) < 0.0:
            quat[i] = -quat[i]

    def _m(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    def _to_rv(q):
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_rotvec()

    def _from_rv(r):
        qq = Rotation.from_rotvec(r).as_quat()
        return np.array([qq[3], qq[0], qq[1], qq[2]])

    rv = np.zeros((len(quat), 3))
    rv[0] = _to_rv(quat[0])
    for i in range(1, len(quat)):
        d = _m(np.array([quat[i - 1][0], -quat[i - 1][1], -quat[i - 1][2],
                         -quat[i - 1][3]]), quat[i])   # conj(q[i-1]) * q[i]
        if d[0] < 0.0:
            d = -d
        rv[i] = rv[i - 1] + _to_rv(d)
    for j in range(3):
        rv[:, j] = filtfilt(b, a, rv[:, j])
    out_quat = np.zeros_like(quat)
    out_quat[0] = _from_rv(rv[0])
    for i in range(1, len(quat)):
        out_quat[i] = _m(out_quat[i - 1], _from_rv(rv[i] - rv[i - 1]))
    q[:, 3:7] = out_quat / np.linalg.norm(out_quat, axis=1, keepdims=True)
    return q


def reconstruct_translation(track: np.ndarray, fps: float,
                            contact: np.ndarray | None = None,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """(F,3) horizontal world translation of the pelvis + per-frame anchor.

    MediaPipe world landmarks are hip-centred, so the person's global motion is
    absent from them. With a static camera we reconstruct it by anchoring the
    planted foot: the pelvis advances by the negative of the planted foot's
    relative displacement.

    Anchor rule (2026-10-08 revision; the previous "smaller per-frame
    displacement" rule agreed with the video's lower foot in only 26-77 % of
    clear frames and produced up to 0.67 s of phantom plants -- measured in
    reports/2026-10-08/reference_fidelity.md §3): with ``contact`` (F,2) from
    ``ground_per_frame`` the anchor is THE planted foot -- exactly one planted
    picks it; both or neither keep the previous anchor (a swinging or sliding
    foot is never the anchor, and a flight phase cannot invent one).  With
    ``contact=None`` the legacy lower-foot hysteresis (margin
    ANCHOR_LOWER_MARGIN, dwell ANCHOR_DWELL_S) is used instead.

    Returns (pos, anchor) with anchor (F,) bool, True = RIGHT foot planted.
    """
    from scipy.signal import savgol_filter
    f = track.shape[0]
    A = np.stack([track[:, JOINTS.index("LeftAnkle")],
                  track[:, JOINTS.index("RightAnkle")]], axis=1)  # (F,2,3)
    if contact is None:
        dy = A[:, 0, 1] - A[:, 1, 1]      # y-up: >0 => RIGHT clearly lower
        dwell = max(1, int(round(ANCHOR_DWELL_S * fps)))
        anchor = np.zeros(f, dtype=bool)
        cur = bool(dy[0] > ANCHOR_LOWER_MARGIN)
        run_side, run = int(cur), 0
        for i in range(f):
            low = 1 if dy[i] > ANCHOR_LOWER_MARGIN else \
                (0 if dy[i] < -ANCHOR_LOWER_MARGIN else -1)
            if low >= 0:
                if low == run_side:
                    run += 1
                else:
                    run_side, run = low, 1
                if low != int(cur) and run >= dwell:
                    cur = bool(low)
            anchor[i] = cur
    else:
        contact = np.asarray(contact, dtype=bool)
        assert contact.ndim == 2 and contact.shape[1] == 2 and len(contact) == f
        anchor = np.zeros(f, dtype=bool)
        cur = bool(contact[0, 1])
        run_new, run_side = 0, -2
        switch_dwell = max(1, int(round(ANCHOR_DWELL_S * fps)))
        for i in range(f):
            l, r = bool(contact[i, 0]), bool(contact[i, 1])
            want = cur
            if l and not r:
                want = False
            elif r and not l:
                want = True
            # both or neither planted: keep the current anchor.  A switch also
            # needs the new foot to be the only-planted one for switch_dwell,
            # so contact flicker cannot churn the anchor.
            if want != cur:
                if int(want) == run_side:
                    run_new += 1
                else:
                    run_side, run_new = int(want), 1
                if run_new >= switch_dwell:
                    cur = want
            else:
                run_side, run_new = -2, 0
            anchor[i] = cur
    pos = np.zeros((f, 3))
    for i in range(1, f):
        a = 1 if anchor[i] else 0
        pos[i] = pos[i - 1] - (A[i, a] - A[i - 1, a])
    w = max(5, int(round(0.4 * fps)) | 1)
    for j in (0, 2):
        pos[:, j] = savgol_filter(pos[:, j], w, 2)
    return pos, anchor


#: sole height of a foot = min over its toe+heel landmarks
_SOLE_JOINTS = (("LeftToe", "LeftHeel"), ("RightToe", "RightHeel"))


def _sole_y(t2: np.ndarray) -> np.ndarray:
    """(F,2) per-frame lowest sole height (y-up, metres) per foot."""
    cols = [np.nanmin(np.stack([t2[:, JOINTS.index(a), 1],
                                t2[:, JOINTS.index(b), 1]]), axis=0)
            for a, b in _SOLE_JOINTS]
    return np.stack(cols, axis=1)


def ground_per_frame(t2: np.ndarray, fps: float,
                     contact_tol: float = GROUND_CONTACT_TOL,
                     lift_tol: float = GROUND_LIFT_TOL,
                     dwell_s: float = GROUND_DWELL_S,
                     ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-frame contact-aware floor for one take (Y-up, metres).

    Replaces the take-level constant floor whose defect is measured in
    reports/2026-10-08/reference_fidelity.md §4 (emitted soles hover up to
    +0.111 m / penetrate -0.067 m, posture-dependently, because the G1's legs
    are not the human's).

    The floor plane is the 5th percentile of all sole heights (the mat, robust
    to noise).  PLANT criterion -- per-frame RELATIVE sole height: mediapipe
    world landmarks are HIP-CENTRED, so a landmark's height above any
    take-level plane confounds "foot lifted" with "hips dropped" (a crouch
    raises every foot landmark relative to the hip line by exactly the hip
    drop; measured: the take-plane criterion lost 92 % of a level-change
    take's contact frames).  A foot is therefore planted when its sole is
    within ``contact_tol`` of the LOWEST sole in the SAME frame (the stance
    foot), with ``dwell_s`` before (re)planting and lift once it exceeds
    ``lift_tol`` above the frame's lowest sole.  A frame with no planted foot
    then means "feet genuinely split in height" (a real step), which is what
    the anchor rule wants.  ``offset`` (F,) shifts each frame so the lowest
    planted sole sits on the mat plane -- in the WORLD frame this exactly
    undoes the hip-centring artifact while keeping genuine step geometry;
    frames with no planted foot interpolate (a flight phase keeps body height
    continuous instead of inventing contact).  If NO frame has a planted foot
    the offset degrades to the take-level constant (min sole - plane) and the
    fallback is recorded in ``info``.

    Returns (offset, contact, info); contact (F,2) bool = (left, right) planted.
    """
    sole = _sole_y(t2)                              # (F,2)
    floor = float(np.nanpercentile(sole, GROUND_FLOOR_PCT))
    low = sole - floor                              # height above the plane
    rel = low - low.min(axis=1, keepdims=True)      # vs the frame's lowest sole
    f = t2.shape[0]
    dwell = max(1, int(round(dwell_s * fps)))
    contact = np.zeros((f, 2), dtype=bool)
    run = np.zeros(2, dtype=int)
    for i in range(f):
        for s in range(2):
            if i and contact[i - 1, s]:
                contact[i, s] = bool(rel[i, s] <= lift_tol)
            elif rel[i, s] <= contact_tol:
                run[s] += 1
                contact[i, s] = run[s] >= dwell
            else:
                run[s] = 0
    off = np.full(f, np.nan)
    for i in range(f):
        if contact[i].any():
            off[i] = max(0.0, float(np.min(low[i][contact[i]])))
    ok = np.flatnonzero(np.isfinite(off))
    fallback = "per_frame_contact"
    if len(ok) == 0:
        off = np.full(f, max(0.0, float(np.nanmin(sole) - floor)))
        fallback = "constant_floor_no_contact_detected"
    else:
        off = np.interp(np.arange(f), ok, off[ok])
    off = np.clip(off, GROUND_OFFSET_MIN, GROUND_OFFSET_MAX)
    info = {"method": fallback,
            "floor_plane_m": round(floor, 4),
            "contact_frac": round(float(contact.mean()), 3),
            "offset_med_m": round(float(np.median(off)), 4),
            "offset_max_m": round(float(off.max()), 4)}
    return off, contact, info


def place_solo(track: np.ndarray, fps: float) -> tuple[np.ndarray, dict]:
    """Y-up GM track -> Z-up G1-scale world targets for one robot.

    Facing = time-averaged horizontal direction perpendicular to the hip line,
    signed toward the (nose+ears) head point; rotated to +x. Scale from the
    existing LSQ segment fit. Global translation is reconstructed from foot
    contact (mediapipe world landmarks are hip-centred), the feet are floored
    at z=0, and the trajectory starts with the pelvis at the origin.
    """
    up = np.array([0.0, 1.0, 0.0])
    lateral = np.nanmean(track[:, JOINTS.index("LeftHip")]
                         - track[:, JOINTS.index("RightHip")], axis=0)
    fwd = np.cross(up, lateral)
    fwd[1] = 0.0
    fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
    head = np.nanmean(track[:, JOINTS.index("Head")]
                      - track[:, JOINTS.index("Core")], axis=0)
    head[1] = 0.0
    if np.dot(fwd, head) < 0:
        fwd = -fwd
    facing_deg = float(np.degrees(np.arctan2(fwd[2], fwd[0])))
    # rotate about y so fwd -> +x
    ang = -np.arctan2(fwd[2], fwd[0])
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    t2 = track @ rot.T

    scale = estimate_player_scale(t2)
    core_mean = np.nanmean(t2[:, JOINTS.index("Core")], axis=0)
    t2 = (t2 - core_mean) * scale + core_mean

    # per-frame contact-aware grounding (REPLACES the take-level constant
    # floor; see ground_per_frame + reference_fidelity.md §4).  Contact is
    # detected first (horizontal translation cannot change it), then the
    # global translation is reconstructed with the planted foot as anchor.
    off, contact, ginfo = ground_per_frame(t2, fps)
    trans, anchor = reconstruct_translation(t2, fps, contact=contact)
    t2 = t2 + (trans * scale)[:, None, :]
    # ground = the floor PLANE (mediapipe tracks are hip-centred, so the plane
    # sits ~0.65 m below the pelvis) + the per-frame contact offset on top
    t2[..., 1] -= (ginfo["floor_plane_m"] + off)[:, None]
    # start with the pelvis at the origin (travel is preserved from there)
    core0 = t2[0, JOINTS.index("Core")]
    t2[..., 0] -= core0[0]
    t2[..., 2] -= core0[2]
    # Y-up -> Z-up: canonical PROPER rotation (x,y,z)_gm -> (x,-z,y)_mj
    # (notes.md 'CORRECTION: chirality-critical'; the naive (x,z,y) swap mirrors)
    zup = np.stack([t2[..., 0], -t2[..., 2], t2[..., 1]], axis=-1)
    travel = float(np.linalg.norm(
        zup[-1, JOINTS.index("Core")][[0, 1]] - zup[0, JOINTS.index("Core")][[0, 1]]))
    info = {"scale": float(scale), "facing_deg_in_mp_frame": facing_deg,
            "floor_offset_m": ginfo["offset_med_m"],
            "grounding": ginfo,
            "anchor_right_frac": round(float(anchor.mean()), 3),
            "net_travel_m": round(travel, 3),
            "contact_kf": contact, "offset_kf": off}
    return zup, info


def ground_qpos_soles(qa: np.ndarray, t_grid: np.ndarray, t_kf: np.ndarray,
                      contact_kf: np.ndarray, model=None) -> tuple[np.ndarray, dict]:
    """Residual per-frame z grounding of the EMITTED qpos (50 Hz grid).

    The target-level grounding (``place_solo.ground_per_frame``) is matched by
    the joint solver only approximately -- the G1's legs are not the human's --
    so the emitted soles can still hover a few cm above or below the mat.  A
    root z-translation moves the whole robot rigidly (every joint angle is
    untouched), so shifting ``qpos[:, 2]`` by minus the lowest PLANTED sole z
    puts that sole exactly on the mat plane.  Frames with no planted foot
    interpolate between their neighbours (flight phases keep body height
    continuous); the correction is clipped to [-0.10, +0.15] m.

    Returns (qa_corrected, info).
    """
    import mujoco

    if model is None:
        from retarget.landmarks import load_g1_spec
        model = load_g1_spec().compile()
    data = mujoco.MjData(model)
    sole_sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
                  for n in ("left_toe", "left_heel", "right_toe", "right_heel")]
    T = len(t_grid)
    c50 = np.stack([np.interp(t_grid, t_kf, contact_kf[:, s].astype(np.float64))
                    for s in range(2)], axis=1) >= 0.5          # (T,2)
    sole = np.zeros((T, 2))
    for i in range(T):
        data.qpos[:] = qa[i]
        mujoco.mj_forward(model, data)
        z = data.site_xpos[sole_sites][:, 2]
        sole[i] = (min(z[0], z[1]), min(z[2], z[3]))            # (L, R) lowest sole
    shift = np.full(T, np.nan)
    for i in range(T):
        if c50[i].any():
            shift[i] = float(np.min(sole[i][c50[i]]))
    ok = np.flatnonzero(np.isfinite(shift))
    shift = np.interp(np.arange(T), ok, shift[ok])
    shift = np.clip(shift, -0.18, 0.15)
    # rate-limit: contact-state changes otherwise step the root by up to
    # 4 m/s (measured on the stalking take), an untrackable target; a median-5
    # filter plus a 1.2 m/s per-step clamp bounds the correction to real body
    # speeds (a level change drops at ~1 m/s; 0.4 m/s lagged fast drops and
    # left soles up to 5 cm under the mat -- measured, feasibility run 1)
    from scipy.signal import medfilt
    shift = medfilt(shift, 5)
    step = 1.2 * (t_grid[1] - t_grid[0]) if T > 1 else 1.0
    for i in range(1, T):
        shift[i] = np.clip(shift[i], shift[i - 1] - step, shift[i - 1] + step)
    for i in range(T - 2, -1, -1):
        shift[i] = np.clip(shift[i], shift[i + 1] - step, shift[i + 1] + step)
    qa = qa.copy()
    qa[:, 2] -= shift
    resid = np.where(c50, sole - shift[:, None], np.nan)
    # the lowest planted sole is 0 by construction; report the OTHER planted
    # foot's hover (both-planted frames) -- the honest residual metric
    other = np.where(c50.sum(1) == 2, np.nanmax(resid, axis=1), np.nan)
    info = {"method": "per_frame_contact_qpos",
            "shift_med_m": round(float(np.median(shift)), 4),
            "shift_max_m": round(float(shift.max()), 4),
            "shift_min_m": round(float(shift.min()), 4),
            "shift_step_max_m": round(float(np.abs(np.diff(shift)).max()), 4),
            "other_planted_sole_max_m": round(float(np.nanmedian(other)), 5),
            "contact_frac_50hz": round(float(c50.mean()), 3)}
    return qa, info


def retarget(name: str, npz_path: Path, t0: float, t1: float, out_dir: Path,
             preset: str = "default") -> dict:
    d = np.load(npz_path)
    t_all = d["t"].astype(np.float64)
    sel = (t_all >= t0) & (t_all <= t1)
    idx = np.where(sel)[0]
    assert len(idx) >= 8, f"{name}: only {len(idx)} frames in [{t0},{t1}]"
    fps = 1.0 / float(np.median(np.diff(t_all)))
    # window may contain a gap between segments; require contiguity
    step = np.diff(idx)
    if (step > 1).any():
        # keep the longest contiguous run
        runs = np.split(idx, np.where(step > 1)[0] + 1)
        idx = max(runs, key=len)
    world = d["world_lm"][idx].astype(np.float64)
    vis = d["visibility"][idx].astype(np.float64)
    det = d["detected"][idx].astype(bool)

    track, gap = gm_track_from_window(world, vis.mean(axis=1), det, fps)
    track = smooth_track(track, fps)
    targets, place = place_solo(track, fps)

    t_kf = (t_all[idx] - t_all[idx[0]]).astype(np.float64)
    solved = solve_keyframes(targets, targets, preset, t_kf=t_kf)
    qa_s = smooth_qpos(solved["qpos_a"], t_kf)
    t_grid, qa, qb, stretch = resample_50hz(t_kf, qa_s, qa_s)
    # residual per-frame grounding of the EMITTED trajectory (the solver only
    # approximately matches the grounded targets; see ground_qpos_soles).
    # Contact flags come from place_solo's per-frame detector (keyframe grid).
    qa, qg = ground_qpos_soles(qa, t_grid, t_kf, place["contact_kf"])
    rms = landmark_rms_final(qa, qb, targets, targets, t_kf * stretch, t_grid,
                             preset)

    meta = {
        "source_npz": str(npz_path),
        "source_window_s": [t0, t1],
        "source_frames": int(len(idx)),
        "source_fps_effective": round(fps, 3),
        "gap_frames": gap["gap_frames"], "gap_frac": gap["gap_frac"],
        "scale_g1_over_human": place["scale"],
        "facing_deg_in_mp_frame": place["facing_deg_in_mp_frame"],
        "floor_offset_m": place["floor_offset_m"],
        "grounding": {**place["grounding"], "qpos": qg},
        "anchor_right_frac": place["anchor_right_frac"],
        "net_travel_m": place["net_travel_m"],
        "preset": preset,
        "opposite_slot": "targets duplicated in the two-robot solver; qpos_b kept as zeros",
        "keyframes": int(len(t_kf)),
        "time_stretch_kinematic": float(stretch),
        "duration_s": float(t_grid[-1]),
        "dt": 0.02,
        "rms": rms,
        "ambiguities": [
            "monocular depth from mediapipe world landmarks: approximate",
            "absolute human scale cancels in the G1/human LSQ scale fit",
            "camera assumed static (gym wide shot, no pan/zoom observed)",
            "GM joints Neck/Head/Hand are proxies (shoulder mid, nose+ears, index+pinky+thumb)",
        ],
        "source": "operator reference video data/references/yt_gBAhX5t-GW4/ref720h264.mp4",
    }
    out = out_dir / f"{name}.npz"
    # contact (T,2) uint8 at 50 Hz: per-frame planted flags (left, right) from
    # the per-frame, contact-aware grounding (v2 format; see FORMAT.md)
    c50 = np.stack([np.interp(t_grid, t_kf, place["contact_kf"][:, s].astype(np.float64))
                    for s in range(2)], axis=1)
    np.savez_compressed(out, qpos_a=qa, qpos_b=np.zeros_like(qb), t=t_grid,
                        technique=name, edges=np.array([], dtype=np.int64),
                        landmark_rms=rms["weighted"],
                        contact=(c50 >= 0.5).astype(np.uint8),
                        meta=json.dumps(meta))
    print(f"{name:28s} frames={len(idx):4d} -> {len(t_grid):5d} 50Hz steps  "
          f"dur={t_grid[-1]:6.2f}s stretch={stretch:.3f}  "
          f"rms_w={rms['weighted']:.3f} rms_max={rms['max']:.3f}  "
          f"contact={float(c50.mean()):.2f} shift={qg['shift_med_m']:+.3f}m "
          f"gaps={gap['gap_frames']}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--spec", required=True,
                    help="JSON list of {name,t0,t1} transition windows")
    ap.add_argument("--out-dir", default=str(REPO / "data/refs_video"))
    ap.add_argument("--preset", default="default")
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for tr in spec:
        npz_path = Path(tr.get("npz", args.npz))
        summary[tr["name"]] = retarget(tr["name"], npz_path,
                                       float(tr["t0"]), float(tr["t1"]),
                                       out_dir, tr.get("preset", args.preset))
    (out_dir / "retarget_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", out_dir / "retarget_summary.json")


if __name__ == "__main__":
    main()
