"""pink-only bounded probe (fix: solver must be named, not None).

Usage: timeout 120 <libtest venv>/bin/python probe_pink_only.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path("/home/ubuntu/grappling")
OUT = REPO / "data/library_audit/pink_probe.json"


def timed(fn, n=200):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main() -> int:
    import pinocchio as pin
    import pink

    m = pin.buildSampleModelHumanoidRandom()
    d = m.createData()
    q = pin.neutral(m)
    robot = pink.Configuration(m, d, q)
    posture = pink.PostureTask(cost=1e-2)
    posture.set_target(q)
    frames = [f for f in m.frames if int(f.parentJoint) > 0]
    frame_name = frames[len(frames) // 2].name
    ft = pink.FrameTask(frame=frame_name, position_cost=1.0, orientation_cost=1.0)
    ft.set_target_from_configuration(robot)
    base = ft.transform_target_to_world
    target = pin.SE3(base.rotation.copy(), base.translation + np.array([0.1, 0.0, 0.1]))
    ft.set_target(target)
    tasks = [posture, ft]

    solve_ms = 1000 * timed(
        lambda: pink.solve_ik(robot, tasks, 0.01, solver="daqp", damping=1e-2), 200)
    for _ in range(100):
        vel = pink.solve_ik(robot, tasks, 0.01, solver="daqp", damping=1e-2)
        robot.integrate_inplace(vel, 0.01)
    err = float(np.linalg.norm(
        robot.get_transform_frame_to_world(frame_name).translation - target.translation))
    out = {
        "pink_version": pink.__version__,
        "model": {"nq": m.nq, "nv": m.nv},
        "solve_ik_ms_per_call": solve_ms,
        "final_frame_pos_err_m": err,
        "loadavg": [round(x, 2) for x in os.getloadavg()],
        "note": "pinocchio sample humanoid; no G1 URDF in repo; solver='daqp' (qpsolvers)",
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
