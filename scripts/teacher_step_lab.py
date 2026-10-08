#!/usr/bin/env python3
"""Step-primitive lab: does the support-transfer / foot-placement primitive
physically execute a step from a built stance?

Runs the *solo* SkillController on the single-G1 scene: build a staggered,
widened stance from the verified ``stand`` keyframe (posture.build_stance),
settle, then execute steps with a given plan and report, per step and per trial:

* phase reached and result (``done`` / ``failed`` with the phase that timed out);
* swing-foot peak clearance (m) and landing displacement (m);
* support-foot slide during the swing (m) — the rubric-B1 no-slide check;
* the measured foot loads (N) at the load transfer and at landing;
* whether the robot stayed up (pelvis z >= 0.45) and the final pelvis xy.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/teacher_step_lab.py --width 0.26 --steps 3
  ... --grid            # small width/load-time grid, one line per trial
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

from teacher import stabilizers as st  # noqa: E402
from teacher.posture import build_stance, sole_positions  # noqa: E402
from teacher.skills import DT, SkillController, drill_step  # noqa: E402
from teacher.solo_scene import load_single_model, stand_qpos  # noqa: E402

OUT = REPO / "data" / "teacher_step_lab.json"


def run_trial(model, width: float, depth: float, drop: float, n_steps: int,
              direction: tuple[float, float], load_s: float, move_s: float,
              settle_s: float, shift_frac: float, settle: float = 1.5,
              verbose: bool = False) -> dict:
    stand = stand_qpos(model)
    stance, sm = build_stance(model, "", stand, width=width, depth=depth,
                              drop=drop)
    ctrl = SkillController(model, stance, stand=stand)
    data = mujoco.MjData(model)
    ctrl.reset(stance)
    sa = ctrl.robot.ctx.qpos_slice
    data.qpos[sa] = stance
    mujoco.mj_forward(model, data)
    ctrl.reset(stance)
    t = 0.0
    for _ in range(int(settle / DT)):
        t = drill_step(model, data, ctrl, t)
    ctx = ctrl.robot.ctx
    twice = int(2 * 20)     # steps alternate feet
    actions = []
    for k in range(n_steps):
        swing = k % 2
        dx, dy = direction
        ctrl.start_step(swing=swing, dx=dx, dy=dy, load_s=load_s,
                        move_s=move_s, settle_s=settle_s,
                        shift_frac=shift_frac, why=f"lab:{k}")
        t_start = t
        peak, land, support_slide = 0.0, None, 0.0
        foot0 = sole_positions(model, "", data.qpos[sa])[1 - swing].copy()
        support0 = foot0
        fell = None
        while ctrl._step is not None and t - t_start < 8.0:
            t = drill_step(model, data, ctrl, t)
            peak = max(peak, st.foot_site_z(data, ctx, swing))
            sup_now = sole_positions(model, "", data.qpos[sa])[1 - swing]
            support_slide = max(support_slide,
                                float(np.abs(sup_now[:, :2] - support0[:, :2]).max()))
            if fell is None and float(data.qpos[sa][2]) < 0.45:
                fell = t
        stp = ctrl._step_history[-1] if ctrl._step_history else {}
        land = float(np.linalg.norm(
            sole_positions(model, "", data.qpos[sa])[swing][:, :2].mean(axis=0)
            - ctrl._pin[swing][:, :2].mean(axis=0)))
        actions.append({
            "step": k, "swing": swing, "result": stp.get("result", "?"),
            "phase": stp.get("phase", "?"), "notes": stp.get("notes", []),
            "peak_clearance": round(float(peak), 3),
            "land_error": round(land, 3),
            "support_slide": round(float(support_slide), 3),
            "t": round(t - t_start, 2), "fell_at": fell,
        })
        if verbose:
            print("   step", actions[-1])
        if fell is not None:
            break
    return {
        "width_asked": width, "depth_asked": depth, "drop": drop,
        "width_meas": round(float(sm["width"]), 3),
        "depth_meas": round(float(sm["depth"]), 3),
        "margin_pose": round(float(sm["margin"]), 4),
        "load_s": load_s, "move_s": move_s, "settle_s": settle_s,
        "shift_frac": shift_frac,
        "steps": actions,
        "n_done": sum(1 for a in actions if a["result"] == "done"),
        "final_pelvis_z": round(float(data.qpos[sa][2]), 3),
        "final_xy": [round(float(data.qpos[sa][0]), 3),
                     round(float(data.qpos[sa][1]), 3)],
        "stayed_up": bool(float(data.qpos[sa][2]) >= 0.45),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=0.26)
    ap.add_argument("--depth", type=float, default=0.16)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--dx", type=float, default=0.12)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--load-s", type=float, default=1.1)
    ap.add_argument("--move-s", type=float, default=0.45)
    ap.add_argument("--settle-s", type=float, default=0.6)
    ap.add_argument("--shift-frac", type=float, default=0.85)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    model = load_single_model()
    trials = []
    if args.grid:
        for width in (0.20, 0.26):
            for shift in (0.6, 0.85):
                for load_s in (0.8, 1.2):
                    trials.append(dict(width=width, shift_frac=shift,
                                       load_s=load_s))
    else:
        trials.append(dict(width=args.width, shift_frac=args.shift_frac,
                           load_s=args.load_s))
    results = []
    for tr in trials:
        t0 = time.time()
        r = run_trial(model, tr["width"], args.depth, args.drop, args.steps,
                      (args.dx, args.dy), tr["load_s"], args.move_s,
                      args.settle_s, tr["shift_frac"])
        if args.tag:
            r["tag"] = args.tag
        r["wall_s"] = round(time.time() - t0, 1)
        results.append(r)
        print(f"w={tr['width']} shift={tr['shift_frac']} load={tr['load_s']}: "
              f"done={r['n_done']}/{len(r['steps'])} stayed_up={r['stayed_up']} "
              f"final_z={r['final_pelvis_z']} xy={r['final_xy']} "
              f"({r['wall_s']}s)")
        for a in r["steps"]:
            print(f"    step{a['step']} swing={a['swing']} {a['result']:6s} "
                  f"phase={a['phase']:9s} peak={a['peak_clearance']:.3f} "
                  f"land_err={a['land_error']:.3f} slide={a['support_slide']:.3f} "
                  f"t={a['t']} fell={a['fell_at']} {a['notes']}")
    OUT.write_text(json.dumps(results, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
