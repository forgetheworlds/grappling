#!/usr/bin/env python
"""Smoke test for the vendored Unitree G1 MJCF model (robots/g1/).

From repo root (venv active):
    MUJOCO_GL=egl python scripts/smoke_g1.py

What it does:
  1. Loads robots/g1/scene.xml (g1.xml + floor) and prints the full spec:
     sizes, timestep, total mass, every joint (name/type/range in rad),
     every actuator (name/joint/ctrlrange/gain+bias params), keyframes.
  2. Resets to the model's "stand" keyframe (pelvis z=0.79, straight legs,
     elbows bent 1.28 rad, shoulders slightly rolled/pitched).
  3. Holds that pose for 5 s of sim time with the model's position
     actuators (ctrl = keyframe joint targets).
  4. Reports pelvis height + torso up-axis tilt at t = 0, 1, 2.5, 5 s.
  5. Renders front + side PNGs into reports/<today>/.
Ends with self-checks; exit 0 on success.
"""

from __future__ import annotations

import datetime as _dt
import os
import sys

import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(REPO, "robots", "g1", "scene.xml")
REPORTS = os.path.join(REPO, "reports", _dt.date.today().isoformat())
HOLD_SECONDS = 5.0
CHECK_TIMES = (0.0, 1.0, 2.5, 5.0)

JOINT_TYPE = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
}


def spec_dump(m: mujoco.MjModel) -> None:
    print("=" * 78)
    print(f"model: {MODEL}")
    print(f"  nq={m.nq} nv={m.nv} nu={m.nu} na={m.na} "
          f"nbody={m.nbody} ngeom={m.ngeom} timestep={m.opt.timestep} s "
          f"integrator={mujoco.mjtIntegrator(int(m.opt.integrator)).name}")
    print(f"  total mass = {m.body_mass.sum():.3f} kg "
          f"(gravity {m.opt.gravity[2]:.1f} m/s^2)")
    print(f"  keyframes: {m.nkey} "
          f"{[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, k) for k in range(m.nkey)]}")

    print("\n--- joints (qpos layout, ranges in radians) ---")
    print(f"{'#':>3} {'name':32} {'type':6} {'qpos_adr':>8} {'dof_adr':>7} "
          f"{'limited':>7} {'range [rad]':>22}")
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        t = JOINT_TYPE[int(m.jnt_type[j])]
        adr, dof = m.jnt_qposadr[j], m.jnt_dofadr[j]
        if m.jnt_limited[j]:
            rng = f"[{m.jnt_range[j][0]:+.4f}, {m.jnt_range[j][1]:+.4f}]"
        else:
            rng = "unlimited"
        print(f"{j:>3} {name:32} {t:6} {adr:>8} {dof:>7} "
              f"{int(m.jnt_limited[j]):>7} {rng:>22}")

    print("\n--- actuators ---")
    print(f"{'#':>3} {'name':32} {'joint':32} {'type':10} {'ctrlrange':>21} "
          f"{'kp':>7} {'kv':>6} {'forcerange':>15}")
    for a in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        jid = m.actuator_trnid[a, 0]
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        # position servo: force = kp*(ctrl-q) - kv*qvel  (gainprm[0]=kp, biasprm=-kp,-kv)
        kp = m.actuator_gainprm[a, 0]
        kv = -m.actuator_biasprm[a, 2]
        ctr = (f"[{m.actuator_ctrlrange[a][0]:+.2f}, {m.actuator_ctrlrange[a][1]:+.2f}]"
               if m.actuator_ctrllimited[a] else "unlimited")
        fr = (f"[{m.actuator_forcerange[a][0]:.0f}, {m.actuator_forcerange[a][1]:.0f}]"
              if m.actuator_forcelimited[a] else "unlimited")
        print(f"{a:>3} {name:32} {jname:32} {'position':10} {ctr:>21} "
              f"{kp:>7.1f} {kv:>6.2f} {fr:>15}")

    if m.nkey:
        print("\n--- keyframe 'stand' (nonzero joint targets) ---")
        q = m.key_qpos[0]
        labels = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                  for j in range(m.njnt)]
        for j in range(m.njnt):
            adr = m.jnt_qposadr[j]
            nq_j = 7 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
            vals = q[adr:adr + nq_j]
            if np.any(vals != 0):
                print(f"    {labels[j]:32} {np.array2string(vals, precision=4)}")


def tilt_deg(m: mujoco.MjModel, d: mujoco.MjData, body: str) -> float:
    """Angle (deg) between body local +z axis and world +z."""
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    r = d.xmat[bid].reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip(r[2, 2], -1.0, 1.0))))


def main() -> int:
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    spec_dump(m)

    # --- initial state: model's own "stand" keyframe -----------------------
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
    assert key_id >= 0, "model must ship the 'stand' keyframe"
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    mujoco.mj_forward(m, d)
    q0 = d.qpos.copy()

    # position actuators: hold keyframe joint targets
    assert all(m.actuator_trnid[a, 1] == -1 for a in range(m.nu)), "expect joint transmissions"
    ctrl0 = np.array([q0[m.jnt_qposadr[m.actuator_trnid[a, 0]]] for a in range(m.nu)])
    if m.actuator_ctrllimited.any():
        lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        ctrl0 = np.clip(ctrl0, lo, hi)
    d.ctrl[:] = ctrl0

    pelvis_z0, torso_tilt0 = d.qpos[2], tilt_deg(m, d, "torso_link")
    print("\n" + "=" * 78)
    print(f"holding keyframe 'stand' for {HOLD_SECONDS:.0f} s "
          f"(position servos, ctrl = keyframe qpos)")
    print(f"{'t [s]':>6} {'pelvis z [m]':>13} {'dz [cm]':>9} {'torso tilt [deg]':>17} "
          f"{'pelvis xy [m]':>16} {'ncon':>5}")

    max_dz, max_tilt = 0.0, torso_tilt0
    rows = {}
    nsteps = int(round(HOLD_SECONDS / m.opt.timestep))
    check_steps = {int(round(t / m.opt.timestep)): t for t in CHECK_TIMES}
    assert 0 in check_steps
    for step in range(nsteps + 1):
        if step in check_steps:
            t = check_steps[step]
            dz = 100.0 * (d.qpos[2] - pelvis_z0)
            tilt = tilt_deg(m, d, "torso_link")
            print(f"{t:>6.1f} {d.qpos[2]:>13.4f} {dz:>9.2f} {tilt:>17.3f} "
                  f"{np.array2string(d.qpos[:2], precision=3):>16} {d.ncon:>5}")
            rows[t] = (d.qpos[2], tilt)
        if step == nsteps:
            break
        mujoco.mj_step(m, d)
        max_dz = max(max_dz, abs(d.qpos[2] - pelvis_z0))
        max_tilt = max(max_tilt, tilt_deg(m, d, "torso_link"))

    # --- renders: front (along +x, robot faces +x) and side (along +y) -----
    os.makedirs(REPORTS, exist_ok=True)
    renderer = mujoco.Renderer(m, height=720, width=960)
    for label, azim in (("front", 0.0), ("side", 90.0)):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, 0.85]
        cam.distance, cam.elevation, cam.azimuth = 3.2, -8.0, azim
        renderer.update_scene(d, camera=cam)
        out = os.path.join(REPORTS, f"g1_{label}.png")
        import imageio
        imageio.imwrite(out, renderer.render())
        print(f"render: {out} ({os.path.getsize(out)} bytes)")
    renderer.close()  # explicit close avoids EGL teardown noise

    # --- verdict + self-checks ---------------------------------------------
    z_end, tilt_end = rows[5.0]
    dz_cm = 100.0 * (z_end - pelvis_z0)
    drift_xy = float(np.hypot(*d.qpos[:2]))
    stable = (abs(dz_cm) < 3.0) and (tilt_end < 10.0) and drift_xy < 0.10
    print("\n" + "=" * 78)
    print(f"STABILITY over 5 s passive-servo hold:")
    print(f"  pelvis height: start {pelvis_z0:.4f} m -> end {z_end:.4f} m "
          f"(dz = {dz_cm:+.2f} cm; |max dz| = {max_dz * 100:.2f} cm)")
    print(f"  torso tilt: start {torso_tilt0:.3f} deg -> end {tilt_end:.3f} deg "
          f"(max {max_tilt:.3f} deg)")
    print(f"  pelvis horizontal drift: {drift_xy * 1000:.1f} mm")
    print(f"  verdict: {'STABLE' if stable else 'NOT STABLE'}")

    for t in CHECK_TIMES:
        assert t in rows, f"missing measurement at t={t}"
    assert os.path.getsize(os.path.join(REPORTS, "g1_front.png")) > 10_000
    assert os.path.getsize(os.path.join(REPORTS, "g1_side.png")) > 10_000
    assert stable, "standing hold failed stability criteria"
    print("\nSELF-CHECK PASSED: model loads, spec dumped, 5 s hold measured, "
          "2 renders written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
