#!/usr/bin/env python3
"""Solo-drill runner + the F1/F2 re-measurement (drill, sweep, ablation).

Modes
-----
``--mode drill``    one continuous drill on the single-G1 scene (stance hold ->
                    level change -> recovery), metrics JSON + optional video;
``--mode sweep``    per-skill success for all 12 skills from a live stance hold
                    (`SkillController.run_all_skills`), metrics JSON;
``--mode ablation`` the F1/F2 matrix on the *previous* drill stance (the
                    retargeted STANCE crouch, which is where the pre-fix
                    "elements topple within ~1.6 s" was measured):

                      row 1  500 Hz calls, frozen reference   (pre-fix)
                      row 2   50 Hz calls, frozen reference   (F2 only)
                      row 3  500 Hz calls, live reference     (F1 only)
                      row 4   50 Hz calls, live reference     (shipped)

                    Each row runs the same scripted elements on the same
                    stance; the report states which fix moves which number.

Every mode uses `drill_step` (one control() call per 20 ms, target held for the
ten 2 ms physics steps) except the deliberate 500 Hz rows, which call
`control()` before every physics step (the pre-fix cadence).

Run:
  PYTHONPATH=src .venv/bin/python scripts/run_solo_drill.py --mode sweep
  ... --mode ablation
  ... --mode drill --video videos/teacher/solo_stance_drill.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402

from teacher.skills import DT, SKILLS, SkillController, drill_step  # noqa: E402
from teacher.solo_scene import load_single_model, stand_qpos  # noqa: E402
from teacher.trims import enforce_stance_width  # noqa: E402

OUT_DIR = REPO / "data"
VID_DIR = REPO / "videos" / "teacher"

#: the drill robot's shipped stance: the model's verified `stand` keyframe,
#: widened to the operator's minimum (measured by FK, applied as hip-roll splay)
STANCE_WIDTH = 0.30
#: level-change depth that the position servos hold (measured; see the report)
LEVEL_FRAC = -0.45


def build(model, wide: bool = True):
    stand = stand_qpos(model)
    if wide:
        stance, splay, w = enforce_stance_width(model, stand, STANCE_WIDTH, prefix="")
    else:
        stance, splay, w = stand.copy(), np.zeros(29), 0.0
    return stand, stance, float(w)


def crouch_stance(model):
    """The previous drill stance: retargeted STANCE crouch (raw, no trim)."""
    npz = np.load(REPO / "data" / "refs" / "STANCE.npz", allow_pickle=True)
    return npz["qpos_a"][0].copy()


def run_once(model, stance, stand, script, hz: int = 50, live: bool = True,
             video: Path | None = None, fps: int = 12,
             min_width: float | None = None) -> dict:
    """Run one scripted drill; `hz` 50 = contract cadence, 500 = pre-fix."""
    ctrl = SkillController(model, stance, stand=stand, min_width=min_width)
    ctrl.reset(stance)
    data = mujoco.MjData(model)
    sa = ctrl.robot.ctx.qpos_slice
    data.qpos[sa] = stance
    mujoco.mj_forward(model, data)
    ctrl.reset(stance)
    if not live:
        ctrl.robot.update_reference = lambda *a, **k: None   # frozen reference
    steps_per_ctrl = 1 if hz == 500 else int(round(DT / model.opt.timestep))
    ctrl_dt = model.opt.timestep * steps_per_ctrl
    tilt_bid = model.body("torso_link").id
    frames, rows, t = [], [], 0.0
    step = 0
    frame_every = int(round((1.0 / fps) / model.opt.timestep))
    for skill, cmd, height, lead, dur in script:
        ctrl.set_skill(skill)
        ctrl.set_command(**cmd)
        ctrl.set_stance_height(height)
        ctrl.set_lead_step(lead)
        t_start = t
        samples, t_fall = [], None
        n0 = len(ctrl._step_history)
        while t < t_start + dur - 1e-9:
            data.ctrl[:] = ctrl.control(data, t)
            for _ in range(steps_per_ctrl):
                mujoco.mj_step(model, data)
                t += model.opt.timestep
                step += 1
                if video is not None and step % frame_every == 0:
                    frames.append(data.qpos[sa].copy())
            z = float(data.qpos[sa][2])
            up = data.xmat[tilt_bid].reshape(3, 3)[:, 2]
            samples.append((t, z, float(data.qpos[sa][0]), float(data.qpos[sa][1]),
                            float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))))
            if t_fall is None and z < 0.45:
                t_fall = t - t_start
        arr = np.array(samples)
        done = ctrl._step_history[n0:]
        rows.append({
            "skill": skill, "dur": dur,
            "up_frac": round(float((arr[:, 1] >= 0.45).mean()), 3),
            "min_z": round(float(arr[:, 1].min()), 3),
            "end_z": round(float(arr[:, 1][-1]), 3),
            "drift": round(float(np.hypot(arr[-1, 2] - arr[0, 2],
                                          arr[-1, 3] - arr[0, 3])), 3),
            "max_tilt": round(float(arr[:, 4].max()), 1),
            "t_fall": None if t_fall is None else round(float(t_fall), 2),
            "steps": len(done),
            "steps_failed": sum(1 for s in done if s["result"] == "failed"),
        })
        if t_fall is not None:
            break                       # a fall ends the element sequence
    out = {"rows": rows, "final_z": round(float(data.qpos[sa][2]), 3),
           "live_reference": live, "hz": hz, "script_len": len(script)}
    if video is not None and frames:
        import imageio
        VID_DIR.mkdir(parents=True, exist_ok=True)
        render = _render(model, np.array(frames), video, fps)
        out["video"] = str(video)
        out["video_frames"] = render
    return out


def _render(model, qpos_seq, path: Path, fps: int) -> int:
    """Render the solo robot (side view) from captured qpos snapshots."""
    import imageio
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.7]
    cam.distance, cam.azimuth, cam.elevation = 2.6, 90.0, -12.0
    opt = mujoco.MjvOption()
    d = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 320, 240)
    imgs = []
    for q in qpos_seq:
        d.qpos[:] = q
        mujoco.mj_forward(model, d)
        renderer.update_scene(d, camera=cam, scene_option=opt)
        imgs.append(renderer.render())
    renderer.close()
    imageio.mimwrite(path, imgs, fps=fps, quality=8, macro_block_size=None)
    return len(imgs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sweep",
                    choices=("drill", "sweep", "ablation"))
    ap.add_argument("--video", default=None)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    model = load_single_model()
    res: dict = {"mode": args.mode, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t0 = time.time()
    if args.mode == "ablation":
        stance = crouch_stance(model)
        stand = stand_qpos(model)
        script = [("STANCE_HOLD", {}, 0.0, 0.0, 2.0),
                  ("LEVEL_CHANGE", {}, -1.0, 0.0, 1.5),
                  ("KNEE_LOWER", {}, -1.0, 0.7, 1.0),
                  ("RECOVER_STAND", {}, 0.0, 0.0, 2.0)]
        rows = []
        for label, hz, live in (("pre-fix 500Hz frozen", 500, False),
                                ("F2 only 50Hz frozen", 50, False),
                                ("F1 only 500Hz live", 500, True),
                                ("shipped 50Hz live", 50, True)):
            r = run_once(model, stance, stand, script, hz=hz, live=live,
                         min_width=0.0)   # pre-fix setup had no width floor
            r["label"] = label
            rows.append(r)
            print(f"{label:22s} " + " | ".join(
                f"{x['skill'][:12]}: up={x['up_frac']} min_z={x['min_z']} "
                f"t_fall={x['t_fall']}" for x in r["rows"]))
        res["ablation"] = rows
        res["stance"] = "STANCE retargeted crouch (raw)"
        path = OUT_DIR / "solo_drill_ablation.json"
    elif args.mode == "sweep":
        stand, stance, w = build(model)
        ctrl = SkillController(model, stance, stand=stand)
        rows = ctrl.run_all_skills(dur=2.5, transition_dur=1.0)
        res["sweep"] = rows
        res["stance_width"] = w
        res["stance"] = "stand keyframe + width splay (operator minimum)"
        for r in rows:
            print(f"{r['skill']:16s} up={r['up_frac']:.2f} min_z={r['min_pelvis_z']:.3f} "
                  f"drift={r['xy_drift']:.3f} tilt={r['max_tilt_deg']:.0f} "
                  f"t_fall={r['t_fall']} steps={r['steps']}/{r['steps_failed']} failed")
        path = OUT_DIR / "solo_drill_skills.json"
    else:
        stand, stance, w = build(model)
        script = [("STANCE_HOLD", {}, 0.0, 0.0, 6.0),
                  ("LEVEL_CHANGE", {}, LEVEL_FRAC, 0.0, 3.0),
                  ("STANCE_HOLD", {}, 0.0, 0.0, 2.0),
                  ("RECOVER_STAND", {}, 0.0, 0.0, 3.0)]
        video = Path(args.video) if args.video else None
        if video and not video.is_absolute():
            video = REPO / video
        r = run_once(model, stance, stand, script, video=video, fps=args.fps)
        res.update(r)
        res["stance_width"] = w
        for x in r["rows"]:
            print(f"{x['skill']:16s} up={x['up_frac']} min_z={x['min_z']} "
                  f"end_z={x['end_z']} drift={x['drift']} tilt={x['max_tilt']} "
                  f"t_fall={x['t_fall']}")
        path = OUT_DIR / "solo_drill_metrics.json"
    res["wall_s"] = round(time.time() - t0, 1)
    path.write_text(json.dumps(res, indent=1))
    print("wrote", path, f"({res['wall_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
