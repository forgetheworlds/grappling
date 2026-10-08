#!/usr/bin/env python
"""Genuine MuJoCo dynamic-feasibility probes for the v1 motion references.

Protocol (identical to the CEM capture's "hold its own joint targets" test,
reports/2026-10-08/cem_capture.md, so results are comparable):

  * solo scene physics (timestep 0.002 s), control at 50 Hz;
  * qpos <- track frame 0, qvel <- 0, then at every control step
    ctrl <- qpos_a[i, 7:36] (plain position servos on the reference's OWN
    joint targets; the reference's root trajectory is deliberately NOT fed to
    the robot -- position servos cannot steer a free base, so this measures
    whether the POSTURE SCHEDULE is dynamically sustainable, an indicator,
    never a demonstration);
  * fall = pelvis z < 0.35 m AND pelvis tilt > 60 deg, sustained 0.25 s
    (solo.fall.FallDetConfig);

The probe result is reported AS MEASURED: a phase that topples is a FAILURE
and is kept, labelled, and rendered.  Nothing here upgrades a kinematic
reference to an executable demonstration.

Outputs:
  videos/motion_refs/dynamic_probe_drill.mp4      (+ _sheet.png)
  videos/motion_refs/dynamic_probe_shot_entry.mp4 (+ _sheet.png)
  videos/motion_refs/dynamic_probe_metrics.json   (all tracks, provenance)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402

from solo.scene import load_solo_model  # noqa: E402
from solo.video import WIDTH, HEIGHT, FPS, Frame, ClipRecorder, probe_media  # noqa: E402

OUT_DIR = REPO / "videos/motion_refs"
METRICS = OUT_DIR / "dynamic_probe_metrics.json"
V1 = REPO / "data/references/motion_refs/v1"

FALL_PELVIS = 0.35      # m (solo.fall.FallDetConfig)
FALL_TILT_DEG = 60.0
FALL_CONFIRM_S = 0.25
CTRL_HZ = 50
RENDER = {"drill_continuous", "shot_entry_full"}   # which probes get video
SEED = 0


def tilt_deg(data) -> float:
    zaxis = data.xmat[1].reshape(3, 3)[:, 2]   # pelvis frame z in world
    return float(np.degrees(np.arccos(np.clip(zaxis[2], -1.0, 1.0))))


def probe(model, data, qpos_track: np.ndarray) -> dict:
    """One rollout; returns measured metrics. qpos_track: (T,36) at CTRL_HZ."""
    T = len(qpos_track)
    data.qpos[:] = qpos_track[0]
    data.qvel[:] = 0.0
    data.time = 0.0          # probes share MjData: reset the clock too
    mujoco.mj_forward(model, data)
    fall_t = None
    low_since = None
    pelvis_min, tilt_max = 1.0, 0.0
    sat_steps = 0
    lo = model.actuator_ctrlrange[:, 0]
    hi = model.actuator_ctrlrange[:, 1]
    t_wall0 = time.time()
    n_physics = int(round((1.0 / CTRL_HZ) / model.opt.timestep))
    completed = 0
    for i in range(T):
        ctrl = np.clip(qpos_track[i, 7:], lo, hi)
        data.ctrl[:] = ctrl
        near = (np.abs(ctrl - lo) <= 0.05 * (hi - lo)) | (np.abs(ctrl - hi) <= 0.05 * (hi - lo))
        sat_steps += int(near.any())
        for _ in range(n_physics):
            mujoco.mj_step(model, data)
        pz, tilt = float(data.qpos[2]), tilt_deg(data)
        pelvis_min = min(pelvis_min, pz)
        tilt_max = max(tilt_max, tilt)
        if pz < FALL_PELVIS and tilt > FALL_TILT_DEG:
            if low_since is None:
                low_since = data.time
            if data.time - low_since >= FALL_CONFIRM_S and fall_t is None:
                fall_t = float(data.time)
                break
        else:
            low_since = None
        completed = i
    wall = time.time() - t_wall0
    ok = fall_t is None and completed == T - 1
    return {"ok": bool(ok), "fell_at_s": fall_t,
            "completed_frac": round((completed + 1) / T, 3),
            "pelvis_z_min": round(pelvis_min, 3),
            "tilt_max_deg": round(tilt_max, 1),
            "sat_frac": round(sat_steps / max(T, 1), 3),
            "wall_s": round(wall, 1),
            "reason": ("completed upright" if ok else
                       f"toppled at t={fall_t:.2f}s "
                       f"(pelvis {pelvis_min:.2f} m, tilt {tilt_max:.0f} deg)"
                       if fall_t is not None else "truncated")}


def frame_lines(name, i, T, pz, tilt, verdict):
    return (f"dynamic feasibility probe: {name}",
            "position servos on the reference's OWN joint targets",
            f"frame {i}/{T}  pelvis_z={pz:.3f} m  tilt={tilt:.1f} deg",
            f"fall rule: pelvis<{FALL_PELVIS} & tilt>{FALL_TILT_DEG} deg "
            f"for {FALL_CONFIRM_S} s",
            f"VERDICT: {verdict}")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    model = load_solo_model()
    data = mujoco.MjData(model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tracks = {}
    for npz in sorted(V1.glob("refs/*.npz")) + [V1 / "drill_continuous.npz",
                                                V1 / "stance_rise.npz"]:
        d = np.load(npz, allow_pickle=True)
        tracks[npz.stem] = (np.asarray(d["qpos_a"], float),
                            json.loads(str(d["meta"])) if "meta" in d else {})
    results = {"version": "1.0",
               "protocol": {"control_hz": CTRL_HZ,
                             "physics_timestep_s": float(model.opt.timestep),
                             "ctrl_source": "qpos_a[i, 7:36] (reference's own joint targets)",
                             "fall_rule": f"pelvis<{FALL_PELVIS} m & tilt>{FALL_TILT_DEG} deg "
                                          f"sustained {FALL_CONFIRM_S} s",
                             "seed": SEED,
                             "claim": "dynamic INDICATOR only; a pass does not "
                                      "make the reference a demonstration"},
               "provenance": {}, "per_track": {}}
    for name, (q, meta) in tracks.items():
        res = probe(model, data, q)
        results["per_track"][name] = res
        print(f"{name:24s} {'PASS' if res['ok'] else 'FAIL'}  "
              f"{res['reason']}  sat={res['sat_frac']:.2f}")
    # re-run the two rendered probes with per-frame HUD lines (recording needs
    # the lines at capture time; the first pass measured without rendering)
    for name in sorted(RENDER):
        q, meta = tracks[name]
        frames: list[Frame] = []
        model2 = model
        data2 = mujoco.MjData(model2)
        p2 = V1 / "refs" / f"{name}.npz"
        if not p2.exists():
            p2 = V1 / f"{name}.npz"
        d2 = np.load(p2, allow_pickle=True)
        q_track = np.asarray(d2["qpos_a"], float)
        T = len(q_track)
        data2.qpos[:] = q_track[0]
        data2.qvel[:] = 0.0
        mujoco.mj_forward(model2, data2)
        fall_t, low_since = None, None
        pelvis_min, tilt_max = 1.0, 0.0
        n_physics = int(round((1.0 / CTRL_HZ) / model2.opt.timestep))
        lo, hi = model2.actuator_ctrlrange[:, 0], model2.actuator_ctrlrange[:, 1]
        completed = 0
        t0 = time.time()
        next_cap = 0.0
        for i in range(T):
            ctrl = np.clip(q_track[i, 7:], lo, hi)
            data2.ctrl[:] = ctrl
            for _ in range(n_physics):
                mujoco.mj_step(model2, data2)
            pz, tilt = float(data2.qpos[2]), tilt_deg(data2)
            pelvis_min = min(pelvis_min, pz)
            tilt_max = max(tilt_max, tilt)
            if pz < FALL_PELVIS and tilt > FALL_TILT_DEG:
                if low_since is None:
                    low_since = data2.time
                if data2.time - low_since >= FALL_CONFIRM_S and fall_t is None:
                    fall_t = float(data2.time)
                else:
                    pass
                if fall_t is not None and data2.time - fall_t > 0.5:
                    break
            else:
                low_since = None
            completed = i
            if data2.time >= next_cap:
                next_cap = data2.time + 1.0 / FPS
                running = fall_t is None
                frames.append(Frame(
                    t=float(data2.time), qpos=np.asarray(data2.qpos, float).copy(),
                    mocap_pos=np.zeros((model2.nmocap, 3)),
                    mocap_quat=np.zeros((model2.nmocap, 4)),
                    lines=frame_lines(name, i, T, pz, tilt,
                                      "running" if running else
                                      f"FELL at {fall_t:.2f}s (kept, honest failure)"),
                    metrics={}))
        res = results["per_track"][name]
        verdict = ("SUCCESS (dynamic indicator)" if res["ok"] else
                   f"FAILURE: {res['reason']}")
        rec = ClipRecorder(f"dynamic probe: {name}",
                           "position servos on the reference's own joint targets "
                           "(indicator, not a demonstration)",
                           verdict=verdict)
        rec.frames = frames
        vid = rec.render(model2, OUT_DIR / f"dynamic_probe_{name}.mp4")
        sheet = rec.contact_sheet(model2, OUT_DIR / f"dynamic_probe_{name}_sheet.png",
                                  caption=f"fall rule pelvis<{FALL_PELVIS} & "
                                  f"tilt>{FALL_TILT_DEG} deg / {FALL_CONFIRM_S} s")
        print(f"  rendered {vid['path']} ({vid['frames']} frames), sheet {sheet['path']}")
    results["provenance"] = {
        "git_commit": git_commit(),
        "model": "src/solo/scene.load_solo_model (robots/g1/g1.xml solo scene)",
        "seed": SEED,
        "control_rate_hz": CTRL_HZ,
        "physics_timestep_s": float(model.opt.timestep),
        "wall_s": {k: v["wall_s"] for k, v in results["per_track"].items()},
        "config_hash": {k: hashlib.sha256(
            (V1 / "refs" / f"{k}.npz").read_bytes()[:1 << 20]).hexdigest()[:16]
            for k in list(results["per_track"])[:4]},
        "reproduce": "MUJOCO_GL=egl .venv/bin/python scripts/probe_drill_dynamic.py",
        "media": {p.name: probe_media(p) for p in OUT_DIR.glob("dynamic_probe_*.mp4")},
    }
    METRICS.write_text(json.dumps(results, indent=1))
    print(f"wrote {METRICS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
