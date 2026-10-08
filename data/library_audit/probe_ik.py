"""IK / whole-body probes on the project's actual G1 solo model + pinocchio/pink/ikpy.

Sections (each independently guarded):
  A. mink whole-body differential IK on the real `load_solo_model()` scene:
     step the left foot up/forward while keeping the right foot planted, the
     torso upright and the CoM over the support; measures solve time + final
     task errors.
  B. pinocchio (pip `pin`) rigid-body dynamics timing on a humanoid sample model.
  C. pin-pink differential IK on a pinocchio sample model (moves a frame, times it).
  D. ikpy on a synthetic serial chain (G1 has no URDF; MJCF-only project).

Usage: <libtest venv>/bin/python probe_ik.py
Writes data/library_audit/ik_probe.json
"""
from __future__ import annotations

import inspect
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path("/home/ubuntu/grappling")
sys.path.insert(0, str(REPO))
OUT = REPO / "data/library_audit/ik_probe.json"

results: dict = {}


def timeit(fn, n: int = 200) -> float:
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def body_matrix(mujoco, data, bid: int) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = data.xmat[bid].reshape(3, 3)
    m[:3, 3] = data.xpos[bid]
    return m


def section_a_mink() -> dict:
    import mujoco
    import mink

    from src.solo.scene import PREFIX, load_solo_model, stand_frame

    model = load_solo_model()
    q, _ = stand_frame(model)
    cfg = mink.Configuration(model)
    cfg.update(q)
    mujoco.mj_forward(model, cfg.data)

    lb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PREFIX + "left_ankle_roll_link")
    rb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PREFIX + "right_ankle_roll_link")
    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PREFIX + "torso_link")
    T_l0 = body_matrix(mujoco, cfg.data, lb)
    T_r0 = body_matrix(mujoco, cfg.data, rb)
    T_t0 = body_matrix(mujoco, cfg.data, tb)

    fw = PREFIX + "left_ankle_roll_link"

    left = mink.FrameTask(fw, "body", position_cost=1.0, orientation_cost=1.0)
    T_l1 = T_l0.copy()
    T_l1[:3, 3] += np.array([0.12, 0.0, 0.08])  # step forward + lift
    left.set_target(mink.SE3.from_matrix(T_l1))

    right = mink.FrameTask(PREFIX + "right_ankle_roll_link", "body",
                           position_cost=1.0, orientation_cost=1.0)
    right.set_target(mink.SE3.from_matrix(T_r0))

    torso = mink.FrameTask(PREFIX + "torso_link", "body",
                           position_cost=0.0, orientation_cost=1.0)
    torso.set_target(mink.SE3.from_matrix(T_t0))

    posture = mink.PostureTask(model, cost=1e-2)
    posture.set_target_from_configuration(cfg)
    com = mink.ComTask(cost=1.0)
    com.set_target_from_configuration(cfg)

    tasks = [left, right, torso, posture, com]
    limits = [mink.ConfigurationLimit(model)]

    dt = 0.01
    solve_ms = timeit(lambda: mink.solve_ik(cfg, tasks, dt, "daqp", limits=limits), 200)

    t0 = time.perf_counter()
    iters = 0
    for iters in range(1, 401):
        vel = mink.solve_ik(cfg, tasks, dt, "daqp", limits=limits)
        cfg.integrate_inplace(vel, dt)
        lp = cfg.data.xpos[lb]
        if np.linalg.norm(lp - T_l1[:3, 3]) < 1e-3:
            break
    wall = time.perf_counter() - t0

    pos_err = float(np.linalg.norm(cfg.data.xpos[lb] - T_l1[:3, 3]))
    R_err = cfg.data.xmat[lb].reshape(3, 3) @ T_l1[:3, :3].T
    ori_err = float(np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))))

    return {
        "model": {"nq": model.nq, "nv": model.nv, "nu": model.nu},
        "solve_ik_ms_per_call": 1000 * solve_ms,
        "iters_to_1mm": iters if pos_err < 1e-3 else None,
        "wall_s_total_iters": wall,
        "final_left_foot_pos_err_m": pos_err,
        "final_left_foot_ori_err_deg": ori_err,
        "solve_ik_signature": str(inspect.signature(mink.solve_ik)),
        "frame_task_signature": str(inspect.signature(mink.FrameTask)),
        "note": "mink uses qpsolvers+daqp; solve_ms includes QP assembly and solve",
    }


def section_b_pinocchio() -> dict:
    import pinocchio as pin

    m = pin.buildSampleModelHumanoidRandom()
    d = m.createData()
    q = pin.randomConfiguration(m)
    v = np.zeros(m.nv)
    a = np.zeros(m.nv)
    rnea_us = 1e6 * timeit(lambda: pin.rnea(m, d, q, v, a), 2000)
    aba_us = 1e6 * timeit(lambda: pin.aba(m, d, q, v, np.zeros(m.nv)), 2000)
    crba_ms = 1e3 * timeit(lambda: pin.crba(m, d, q), 200)
    frames = [f for f in m.frames if int(f.parentJoint) > 0]
    fid = m.getFrameId(frames[len(frames) // 2].name)
    fjac_us = 1e6 * timeit(lambda: pin.computeFrameJacobian(m, d, q, fid, pin.LOCAL), 2000)
    return {
        "version": pin.__version__,
        "sample_model": {"nq": m.nq, "nv": m.nv, "njoints": m.njoints},
        "rnea_us": rnea_us, "aba_us": aba_us, "crba_us": 1000 * crba_ms,
        "computeFrameJacobian_us": fjac_us,
        "note": "sample humanoid, not the G1 (project is MJCF-only; no URDF in repo)",
    }


def section_c_pink() -> dict:
    import pink
    import pinocchio as pin

    m = pin.buildSampleModelHumanoidRandom()
    d = m.createData()
    q = pin.neutral(m)
    robot = pink.Configuration(m, d, q)
    tasks = [pink.PostureTask(cost=1e-2)]
    tasks[0].set_target(q)
    frames = [f for f in m.frames if int(f.parentJoint) > 0]
    frame_name = frames[len(frames) // 2].name
    ft = pink.FrameTask(frame=frame_name, position_cost=1.0, orientation_cost=1.0)
    ft.set_target_from_configuration(robot)
    base = ft.transform_target_to_world
    target = pin.SE3(base.rotation.copy(), base.translation + np.array([0.1, 0.0, 0.1]))
    ft.set_target(target)
    tasks.append(ft)
    solve_ms = 1000 * timeit(lambda: pink.solve_ik(robot, tasks, 0.01, solver=None, damping=1e-2), 200)
    for _ in range(100):
        vel = pink.solve_ik(robot, tasks, 0.01, solver=None, damping=1e-2)
        robot.integrate_inplace(vel, 0.01)
    err = float(np.linalg.norm(
        robot.get_transform_frame_to_world(frame_name).translation - target.translation))
    return {
        "version": pink.__version__,
        "solve_ik_ms_per_call": solve_ms,
        "final_frame_pos_err_m": err,
        "solve_ik_signature": str(inspect.signature(pink.solve_ik)),
        "note": "pinocchio sample humanoid; pink needs URDF+pinocchio model, not MJCF",
    }


def section_d_ikpy() -> dict:
    import ikpy.chain
    import ikpy.link

    sig = {}
    for name in ("URDFLink", "OriginLink"):
        cls = getattr(ikpy.link, name)
        sig[name] = str(inspect.signature(cls))
    links = [ikpy.link.OriginLink()]
    for i in range(6):
        links.append(ikpy.link.URDFLink(
            name=f"j{i}", origin_translation=np.array([0.0, 0.0, 0.25 * (i + 1)]),
            origin_orientation=np.array([0, 0, 0]), rotation=np.array([0, 1, 0])))
    chain = ikpy.chain.Chain(links, active_links_mask=[False] + [True] * 6)
    target = [0.3, 0.2, 0.9]
    per_call_ms = 1000 * timeit(
        lambda: chain.inverse_kinematics(target_position=target), 20)
    sol = chain.inverse_kinematics(target_position=target)
    fk = chain.forward_kinematics(sol)
    err = float(np.linalg.norm(np.asarray(fk)[:3, 3] - np.asarray(target)))
    return {
        "version": ikpy.__version__,
        "synthetic_chain_dof": 6,
        "inverse_kinematics_ms_per_call": per_call_ms,
        "final_target_err_m": err,
        "signatures": sig,
        "note": "synthetic chain: the repo has no URDF (MJCF-only), ikpy requires URDF",
    }


def main() -> int:
    for name, fn in (("mink", section_a_mink), ("pinocchio", section_b_pinocchio),
                     ("pink", section_c_pink), ("ikpy", section_d_ikpy)):
        try:
            results[name] = fn()
            print(f"[{name}] OK")
        except Exception as e:  # noqa: BLE001
            import traceback
            results[name] = {"error": f"{type(e).__name__}: {e}",
                             "trace": traceback.format_exc()[-1200:]}
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
    OUT.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
