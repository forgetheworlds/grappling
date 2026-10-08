#!/usr/bin/env python3
"""FIX 4: sweep the operator's two balance axes at fine magnitudes, COMBINED.

Axes (the operator's rule, applied where it is *holdable* on this robot):

* **frontal — stance width.**  Feet further apart laterally.  Realised as a
  hip-roll *splay* on the model's verified ``stand`` keyframe (measured by FK,
  applied by ``trims.enforce_stance_width``): an IK re-solve of the legs to a
  foot-placement target is *not* holdable on this model (servo sag; ankle-roll
  range ±0.26 rad caps flat-foot lateral pelvis travel at ~5 cm) — measured and
  reported in ``reports/2026-10-08/teacher_exec_fix.md``.
* **sagittal — body placement over the feet.**  The rear-leg-back axis, as the
  *holdable* equivalent: the reference body sits forward of the foot-centre
  line (a longer base behind the CoM), which the ankle-pitch channel actually
  reaches (measured authority −0.53 m/rad, saturating ~5 cm).

Each setting is evaluated with the same runner as the acceptance runs (50 Hz
control over 2 ms physics, one continuous simulation per setting):

* ``hold_s``  seconds the pelvis stayed above 0.45 m within an 8 s hold;
* ``drift``   pelvis xy excursion at the end of the hold (m);
* ``tilt``    max torso tilt (deg);
* ``margin``  settled CoM margin inside the ACTUAL contact hull (m), and the
  pose-level sole-patch margin of the built stance;
* ``width``   measured lateral foot separation (m).

Usage:
  PYTHONPATH=src .venv/bin/python scripts/teacher_stance_sweep.py --hold 8
  ... --level            # level-change depth sweep instead
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

from teacher.posture import live_support_margin, stance_metrics  # noqa: E402
from teacher.skills import DT, SkillController, drill_step  # noqa: E402
from teacher.solo_scene import load_single_model, stand_qpos  # noqa: E402
from teacher.trims import enforce_stance_width  # noqa: E402

OUT = REPO / "data" / "teacher_stance_sweep.json"


def hold(model, stance, seconds: float, sagittal: float = 0.0,
         stand=None) -> dict:
    """Run a continuous hold; return duration/drift/tilt/margin metrics."""
    ctrl = SkillController(model, stance, stand=stand)
    ctrl.reset(stance)
    data = mujoco.MjData(model)
    sa = ctrl.robot.ctx.qpos_slice
    data.qpos[sa] = stance
    if sagittal:                      # reference body forward of the foot line
        ctrl._xy = ctrl._xy + np.array([sagittal, 0.0])
    mujoco.mj_forward(model, data)
    ctrl.reset(stance)
    ctrl._xy = ctrl._xy + np.array([sagittal, 0.0])
    t, t_fall, log = 0.0, None, []
    n = int(seconds / DT)
    tilt_bid = model.body("torso_link").id
    for i in range(n):
        t = drill_step(model, data, ctrl, t)
        z = float(data.qpos[sa][2])
        if t_fall is None and z < 0.45:
            t_fall = t
            break
        if i % 25 == 0:
            up = data.xmat[tilt_bid].reshape(3, 3)[:, 2]
            log.append((t, float(data.qpos[sa][0]), float(data.qpos[sa][1]),
                        z, float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))))
    arr = np.array(log) if log else np.zeros((1, 5))
    livem = live_support_margin(model, data, ctrl.robot.ctx)
    return {
        "hold_s": round(float(t_fall if t_fall is not None else seconds), 2),
        "drift": round(float(np.hypot(arr[-1, 1] - arr[0, 1],
                                      arr[-1, 2] - arr[0, 2])), 4),
        "tilt_deg": round(float(arr[:, 4].max()), 1),
        "end_z": round(float(arr[-1, 3]), 3),
        "margin_live": round(float(livem["margin"]), 4),
        "n_contact": int(livem["n_contact"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=float, default=8.0)
    ap.add_argument("--level", action="store_true", help="level-change depth sweep")
    args = ap.parse_args()
    model = load_single_model()
    stand = stand_qpos(model)
    results = []
    t0 = time.time()
    if args.level:
        for drop in (0.03, 0.06, 0.09, 0.12, 0.18):
            stance, splay, w = enforce_stance_width(model, stand, 0.30, prefix="")
            ctrl = SkillController(model, stance, stand=stand)
            ctrl.reset(stance)
            data = mujoco.MjData(model)
            sa = ctrl.robot.ctx.qpos_slice
            data.qpos[sa] = stance
            mujoco.mj_forward(model, data)
            ctrl.reset(stance)
            t, t_fall = 0.0, None
            n_settle = int(1.0 / DT)
            tilt_bid = model.body("torso_link").id
            xs = []
            for i in range(n_settle + int(args.hold / DT)):
                if i == n_settle:
                    ctrl.set_stance_height(-drop / 0.18)   # dz = -0.18*frac
                t = drill_step(model, data, ctrl, t)
                z = float(data.qpos[sa][2])
                if i >= n_settle and t_fall is None and z < 0.45:
                    t_fall = t - 1.0
                    break
                if i >= n_settle and i % 25 == 0:
                    xs.append((float(data.qpos[sa][0]), float(data.qpos[sa][1]), z))
            arr = np.array(xs) if xs else np.zeros((1, 3))
            rec = {"axis": "level_change", "drop": drop,
                   "hold_s": round(float(t_fall if t_fall is not None else args.hold), 2),
                   "drift": round(float(np.hypot(arr[-1, 0] - arr[0, 0],
                                                 arr[-1, 1] - arr[0, 1])), 4),
                   "min_z": round(float(arr[:, 2].min()), 3),
                   "end_z": round(float(arr[-1, 2]), 3)}
            results.append(rec)
            print(rec)
    else:
        for splay_extra in (0.0, 0.04, 0.08, 0.12):
            for sagittal in (0.0, 0.03, 0.06):
                q, splay, w = enforce_stance_width(model, stand, 0.30, prefix="")
                # extra widening beyond the enforced minimum
                if splay_extra:
                    q = q.copy()
                    q[7 + 1] -= splay_extra
                    q[7 + 7] += splay_extra
                m = stance_metrics(model, "", q)
                r = hold(model, q, args.hold, sagittal=sagittal, stand=stand)
                rec = {"splay_extra": splay_extra, "sagittal": sagittal,
                       "width": round(float(m["width"]), 3),
                       "margin_pose": round(float(m["margin"]), 4), **r}
                results.append(rec)
                print(f"splay+{splay_extra:.2f} sag+{sagittal:.2f} width={rec['width']:.3f} "
                      f"hold={rec['hold_s']:.2f}s drift={rec['drift']:.3f} "
                      f"tilt={rec['tilt_deg']:.1f} margin_pose={rec['margin_pose']:+.3f} "
                      f"margin_live={rec['margin_live']:+.3f}")
    results.append({"wall_s": round(time.time() - t0, 1)})
    OUT.write_text(json.dumps(results, indent=1))
    print("wrote", OUT, "in", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
