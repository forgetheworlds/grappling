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


def reconstruct_translation(track: np.ndarray, fps: float) -> np.ndarray:
    """(F,3) horizontal world translation of the pelvis.

    MediaPipe world landmarks are hip-centred, so the person's global motion is
    absent from them. With a static camera we reconstruct it by anchoring the
    planted foot: at each step the foot with the smaller horizontal relative
    displacement is treated as planted, and the pelvis advances by the negative
    of that foot's relative displacement. The integrated trajectory is
    low-passed (0.4 s) to remove per-frame noise; the vertical component is
    kept zero (the floor is handled separately).
    """
    from scipy.signal import savgol_filter
    f = track.shape[0]
    la = track[:, JOINTS.index("LeftAnkle")]
    ra = track[:, JOINTS.index("RightAnkle")]
    pos = np.zeros((f, 3))
    for i in range(1, f):
        dl = float(np.linalg.norm(la[i, [0, 2]] - la[i - 1, [0, 2]]))
        dr = float(np.linalg.norm(ra[i, [0, 2]] - ra[i - 1, [0, 2]]))
        step = -(la[i] - la[i - 1]) if dl <= dr else -(ra[i] - ra[i - 1])
        pos[i] = pos[i - 1] + step
    w = max(5, int(round(0.4 * fps)) | 1)
    for j in (0, 2):
        pos[:, j] = savgol_filter(pos[:, j], w, 2)
    return pos


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

    # reconstruct the global translation (hip-centred world landmarks lack it)
    trans = reconstruct_translation(t2, fps) * scale
    t2 = t2 + trans[:, None, :]

    # floor: min over foot joints across the whole take -> 0
    foot_idx = [JOINTS.index(n) for n in
                ("LeftToe", "RightToe", "LeftHeel", "RightHeel",
                 "LeftAnkle", "RightAnkle")]
    floor = float(np.nanmin(t2[:, foot_idx, 1]))
    t2[..., 1] -= floor
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
            "floor_offset_m": float(floor), "net_travel_m": round(travel, 3)}
    return zup, info


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
    np.savez_compressed(out, qpos_a=qa, qpos_b=np.zeros_like(qb), t=t_grid,
                        technique=name, edges=np.array([], dtype=np.int64),
                        landmark_rms=rms["weighted"], meta=json.dumps(meta))
    print(f"{name:28s} frames={len(idx):4d} -> {len(t_grid):5d} 50Hz steps  "
          f"dur={t_grid[-1]:6.2f}s stretch={stretch:.3f}  "
          f"rms_w={rms['weighted']:.3f} rms_max={rms['max']:.3f}  gaps={gap['gap_frames']}")
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
