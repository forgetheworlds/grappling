#!/usr/bin/env python3
"""Validate the retargeted references by PD-tracking them in the two-G1 scene.

For each data/refs/<technique>.npz: start the scene at the reference's own
initial state, drive both robots with ctrl = reference joint targets at the
model's position-servo gains, step at 2 ms, and measure:

  * joint tracking error   mean / max |q_sim - q_ref| over 29 joints (rad)
  * pelvis profiles        height error vs ref (m), torso tilt (deg)
  * ground penetration     max(0, -z) of toe/heel/ankle sites (m)
  * inter-robot contact    deepest contact penetration between robots (m)
  * self-collision         contacts between non-adjacent bodies of one robot

ACCEPTANCE: mean joint error <= 0.15 rad; ground penetration <= 2 cm; no
self / inter-robot interpenetration deeper than 2 cm; robots stay up wherever
the technique implies standing (checked on frames whose reference pelvis is
high); SPRAWL's defender must END prone (it legitimately is).

Repair loop on violation: retime x1.25 (up to 3 attempts, rebuilding the
reference on a longer time base), then re-solve with the 'strict' weight
preset. Every repair is logged into the npz meta['repairs'] and printed.

Videos: videos/refs/<technique>.mp4 (30 fps, both robots, tracked rollout).

Run from repo root:
    MUJOCO_GL=egl .venv/bin/python scripts/validate_refs.py [--only A,B]
                 [--no-video] [--no-repair]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco  # noqa: E402

from retarget.scene import load_scene_model, robot_slice  # noqa: E402
from retarget.techniques import build_technique  # noqa: E402

REF_DIR = Path(__file__).resolve().parents[1] / "data" / "refs"
VID_DIR = Path(__file__).resolve().parents[1] / "videos" / "refs"

JOINT_ERR_LIMIT = 0.15     # rad, mean
PENETRATION_LIMIT = 0.02   # m (ground AND robot-robot AND self)
STAND_PELVIS_REF = 0.55    # ref pelvis z above this => robot must stay up
STAND_PELVIS_SIM = 0.45    # min allowed sim pelvis z on those frames
HOLD_TAIL = 0.5            # s of extra PD tracking after the reference ends
RETIME_FACTOR = 1.25
MAX_RETIME_ATTEMPTS = 3
FOOT_SITES = ("left_toe", "right_toe", "left_heel", "right_heel",
              "left_ankle", "right_ankle")


def _torso_tilt_deg(model, data, prefix):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                            f"{prefix}torso_link")
    up = data.xmat[bid].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))


def run_tracking(model, qpos_a, qpos_b, t_ref, capture=None):
    """PD-track the reference from its own initial state.

    Returns metric rows, one per 10 ms:
    (t, err_a, err_b, pen_a, pen_b, tilt_a, tilt_b, inter_pen, self_depth,
     self_count, pelvis_z_a, pelvis_z_b).
    If ``capture`` (a list) is given, (t, qpos) snapshots are appended at
    30 fps for video rendering.
    """
    data = mujoco.MjData(model)
    sa, sb = robot_slice(model, "a_"), robot_slice(model, "b_")
    data.qpos[sa] = qpos_a[0]
    data.qpos[sb] = qpos_b[0]
    mujoco.mj_forward(model, data)

    foot_ids = {p: np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                               f"{p}{n}") for n in FOOT_SITES])
                for p in ("a_", "b_")}
    body_of_geom = np.array([int(model.geom_bodyid[g])
                             for g in range(model.ngeom)])
    robot_of_body = np.array([
        0 if model.body(int(b)).name.startswith("a_")
        else 1 if model.body(int(b)).name.startswith("b_") else -1
        for b in range(model.nbody)])
    parent = model.body_parentid

    def ref_at(t, q):
        i = int(np.clip(np.searchsorted(t_ref, t), 1, len(t_ref) - 1))
        w = (t - t_ref[i - 1]) / (t_ref[i] - t_ref[i - 1])
        return q[i - 1, 7:36] * (1 - w) + q[i, 7:36] * w

    t_end = float(t_ref[-1]) + HOLD_TAIL
    dt = model.opt.timestep
    rows = []
    next_capture_t = 0.0
    n_steps = int(np.ceil(t_end / dt))
    for step in range(n_steps):
        t = step * dt
        data.ctrl[:29] = ref_at(t, qpos_a)
        data.ctrl[29:] = ref_at(t, qpos_b)
        mujoco.mj_step(model, data)

        if capture is not None and t >= next_capture_t - 1e-9:
            capture.append((float(t), data.qpos.copy()))
            next_capture_t += 1.0 / 30.0
        if step % 5:  # sample metrics at 1 kHz? 100 Hz is plenty
            continue
        err_a = float(np.abs(data.qpos[sa][7:36] - ref_at(t, qpos_a)).mean())
        err_b = float(np.abs(data.qpos[sb][7:36] - ref_at(t, qpos_b)).mean())
        pen = {p: float(max(0.0, -data.site_xpos[ids][:, 2].min()))
               for p, ids in foot_ids.items()}
        inter = self_depth = 0.0
        self_count = 0
        for c in range(data.ncon):
            con = data.contact[c]
            b1 = body_of_geom[int(con.geom1)]
            b2 = body_of_geom[int(con.geom2)]
            r1, r2 = robot_of_body[b1], robot_of_body[b2]
            depth = float(-con.dist)
            if r1 >= 0 and r2 >= 0 and r1 != r2:
                inter = max(inter, depth)
            elif r1 == r2 and r1 >= 0:
                if b1 != b2 and parent[b1] != b2 and parent[b2] != b1:
                    self_count += 1
                    self_depth = max(self_depth, depth)
        rows.append((t, err_a, err_b, pen["a_"], pen["b_"],
                     _torso_tilt_deg(model, data, "a_"),
                     _torso_tilt_deg(model, data, "b_"),
                     inter, self_depth, self_count,
                     float(data.qpos[sa][2]), float(data.qpos[sb][2])))
    return np.array(rows)


def summarize(rows, qpos_a, qpos_b, t_ref, technique):
    """Acceptance metrics from the tracking rows."""
    t, err_a, err_b = rows[:, 0], rows[:, 1], rows[:, 2]
    ref_z_a = np.interp(t, t_ref, qpos_a[:, 2])
    ref_z_b = np.interp(t, t_ref, qpos_b[:, 2])
    in_ref = t <= t_ref[-1]
    up_a = (ref_z_a > STAND_PELVIS_REF) & in_ref
    up_b = (ref_z_b > STAND_PELVIS_REF) & in_ref
    z_err = max(
        float(np.abs(rows[:, 10] - ref_z_a)[up_a].mean()) if up_a.any() else 0.0,
        float(np.abs(rows[:, 11] - ref_z_b)[up_b].mean()) if up_b.any() else 0.0)
    min_up = min(
        float(rows[up_a, 10].min()) if up_a.any() else 9.0,
        float(rows[up_b, 11].min()) if up_b.any() else 9.0)
    m = {
        "joint_err_mean_a": float(err_a.mean()),
        "joint_err_mean_b": float(err_b.mean()),
        "joint_err_mean": float(max(err_a.mean(), err_b.mean())),
        "joint_err_max": float(max(err_a.max(), err_b.max())),
        "pelvis_z_err_mean_while_up": z_err,
        "min_pelvis_z_while_ref_up": min_up,
        "max_torso_tilt_deg": float(max(rows[:, 5].max(), rows[:, 6].max())),
        "ground_pen": float(max(rows[:, 3].max(), rows[:, 4].max())),
        "inter_pen": float(rows[:, 7].max()),
        "self_depth": float(rows[:, 8].max()),
        "self_frames": int((rows[:, 9] > 0).sum()),
        "end_pelvis_z_b": float(qpos_b[-1, 2]),
    }
    m["sprawl_defender_prone"] = (
        bool(qpos_b[-1, 2] < 0.45) if technique == "SPRAWL" else None)
    m["pass"] = bool(
        m["joint_err_mean"] <= JOINT_ERR_LIMIT
        and m["ground_pen"] <= PENETRATION_LIMIT
        and m["inter_pen"] <= PENETRATION_LIMIT
        and m["self_depth"] <= PENETRATION_LIMIT
        and m["min_pelvis_z_while_ref_up"] >= STAND_PELVIS_SIM
        and (m["sprawl_defender_prone"] is not False))
    return m


def load_ref(technique: str):
    z = np.load(REF_DIR / f"{technique}.npz", allow_pickle=False)
    return (z["qpos_a"], z["qpos_b"], z["t"], str(z["technique"]),
            z["edges"], float(z["landmark_rms"]),
            json.loads(str(z["meta"])))


def save_ref(technique, result):
    np.savez_compressed(
        REF_DIR / f"{technique}.npz",
        qpos_a=np.ascontiguousarray(result["qpos_a"]),
        qpos_b=np.ascontiguousarray(result["qpos_b"]),
        t=np.ascontiguousarray(result["t"]),
        technique=np.array(technique),
        edges=np.array(result["edges"], dtype=np.int64),
        landmark_rms=np.float64(result["landmark_rms"]),
        meta=json.dumps(result["meta"], sort_keys=True))


def render_video(model, capture, out_path):
    """Render captured (t, qpos) snapshots at 30 fps; both robots, side view."""
    import imageio.v2 as imageio
    renderer = mujoco.Renderer(model, height=720, width=960)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.55]
    cam.distance = 4.2
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 90
    data = mujoco.MjData(model)
    sa, sb = robot_slice(model, "a_"), robot_slice(model, "b_")
    frames = []
    for _, q in capture:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        cam.lookat[0] = 0.5 * (q[sa][0] + q[sb][0])
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    imageio.mimwrite(out_path, frames, fps=30, quality=8,
                     macro_block_size=None)
    renderer.close()  # REQUIRED on this EGL host (notes.md)
    return len(frames)


def repair(model, technique, meta):
    """Spec repair loop: retime x1.25 (<=3), then strict re-solve.

    Returns (accepted_result, repair_log).
    """
    stretch = float(meta.get("time_stretch_requested", 1.0))
    preset = meta.get("preset", "default")
    log = []
    result = None
    for attempt in range(MAX_RETIME_ATTEMPTS + 1):
        kind = "retime" if attempt < MAX_RETIME_ATTEMPTS else "strict"
        if kind == "retime":
            stretch *= RETIME_FACTOR
        else:
            preset = "strict"
        cand = build_technique(technique, preset=preset, time_stretch=stretch)
        rows = run_tracking(model, cand["qpos_a"], cand["qpos_b"], cand["t"])
        metrics = summarize(rows, cand["qpos_a"], cand["qpos_b"],
                            cand["t"], technique)
        log.append({"kind": kind, "time_stretch": stretch, "preset": preset,
                    "pass": metrics["pass"],
                    "joint_err_mean": metrics["joint_err_mean"],
                    "joint_err_max": metrics["joint_err_max"],
                    "ground_pen": metrics["ground_pen"],
                    "inter_pen": metrics["inter_pen"],
                    "self_depth": metrics["self_depth"],
                    "min_pelvis_z_while_ref_up":
                        metrics["min_pelvis_z_while_ref_up"]})
        print(f"    repair[{technique}] {kind} stretch={stretch:.2f} "
              f"preset={preset} -> "
              f"{'PASS' if metrics['pass'] else 'fail'} "
              f"(err={metrics['joint_err_mean']:.3f}, "
              f"grnd={metrics['ground_pen']:.3f}, "
              f"inter={metrics['inter_pen']:.3f}, "
              f"self={metrics['self_depth']:.3f})")
        result = cand
        if metrics["pass"]:
            break
    return result, log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of techniques")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-repair", action="store_true")
    args = ap.parse_args()
    techniques = (args.only.split(",") if args.only
                  else sorted(p.stem for p in REF_DIR.glob("*.npz")))
    model = load_scene_model()
    VID_DIR.mkdir(parents=True, exist_ok=True)

    table = []
    for tech in techniques:
        qpos_a, qpos_b, t_ref, _, edges, rms, meta = load_ref(tech)
        rows = run_tracking(model, qpos_a, qpos_b, t_ref)
        metrics = summarize(rows, qpos_a, qpos_b, t_ref, tech)
        print(f"  {tech}: initial {'PASS' if metrics['pass'] else 'FAIL'} "
              f"(err_mean={metrics['joint_err_mean']:.3f} "
              f"err_max={metrics['joint_err_max']:.3f} "
              f"grnd={metrics['ground_pen']:.3f} "
              f"inter={metrics['inter_pen']:.3f} "
              f"self={metrics['self_depth']:.3f} "
              f"zErr={metrics['pelvis_z_err_mean_while_up']:.3f} "
              f"minUpZ={metrics['min_pelvis_z_while_ref_up']:.2f})")
        repairs = []
        if not metrics["pass"] and not args.no_repair:
            result, repairs = repair(model, tech, meta)
            qpos_a, qpos_b, t_ref = result["qpos_a"], result["qpos_b"], result["t"]
            meta = result["meta"]
            meta["repairs"] = meta.get("repairs", []) + repairs
            save_ref(tech, result)
            rows = run_tracking(model, qpos_a, qpos_b, t_ref)
            metrics = summarize(rows, qpos_a, qpos_b, t_ref, tech)

        capture = []
        run_tracking(model, qpos_a, qpos_b, t_ref, capture)
        n_frames = 0
        if not args.no_video:
            n_frames = render_video(model, capture,
                                    VID_DIR / f"{tech}.mp4")
        table.append((tech, metrics, len(repairs), n_frames))

    print(f"\n{'technique':12s} {'err_mean':>8s} {'err_max':>8s} "
          f"{'grnd_pen':>8s} {'inter':>7s} {'self':>7s} "
          f"{'zErr':>6s} {'tilt_d':>6s} {'prone':>5s} {'pass':>5s} "
          f"{'rep':>3s} {'vid':>4s}")
    for tech, m, nrep, nvid in table:
        prone = "-" if m["sprawl_defender_prone"] is None else (
            "Y" if m["sprawl_defender_prone"] else "N")
        print(f"{tech:12s} {m['joint_err_mean']:8.3f} {m['joint_err_max']:8.3f} "
              f"{m['ground_pen']:8.3f} {m['inter_pen']:7.3f} "
              f"{m['self_depth']:7.3f} "
              f"{m['pelvis_z_err_mean_while_up']:6.3f} "
              f"{m['max_torso_tilt_deg']:6.1f} {prone:>5s} "
              f"{'PASS' if m['pass'] else 'FAIL':>5s} {nrep:3d} {nvid:4d}")
    ok = all(m["pass"] for _, m, _, _ in table)
    print(f"\nacceptance: mean joint err <= {JOINT_ERR_LIMIT} rad, "
          f"all penetrations <= {PENETRATION_LIMIT} m -> "
          + ("ALL PASS" if ok else "SOME FAIL (see table / meta.repairs)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
