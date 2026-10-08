"""Two-G1 wrestling scene, composed programmatically with MjSpec.

The scene = floor mat + light + two independent G1s (each with the 19
retargeting landmark sites from landmarks.py), attached with name prefixes.

Naming contract (verified on the compiled model):
  robot A prefix ``a_`` : qpos[0:36]  (free joint a_floating_base_joint),
                          actuators a_left_hip_pitch_joint .. a_right_wrist_yaw_joint
                          (29, one per hinge, trnid == joint id)
  robot B prefix ``b_`` : qpos[36:72], same 29 actuators.
  Both robots keep the g1.xml keyframe ('a_stand' / 'b_stand'); an extra
  'both_stand' key separates them by ±0.35 m facing each other.
  Landmark sites: a_core, a_neck, a_head, a_{left,right}_{hip,knee,ankle,
  toe,heel,shoulder,elbow,wrist} and the b_ mirror set (46 sites total).

``write_scene_xml`` serializes the composed spec to robots/wrestling_scene.xml
(the only file created next to the vendored robot; g1.xml is never touched).
``load_scene_model`` re-compiles the spec directly (fast, deterministic).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .landmarks import load_g1_spec
REPO = Path(__file__).resolve().parents[2]
SCENE_XML = REPO / "robots" / "wrestling_scene.xml"
#: separation (m) of the two robots in the 'both_stand' keyframe
STAND_SEPARATION = 0.7


def build_scene_spec() -> mujoco.MjSpec:
    scene = mujoco.MjSpec()
    scene.modelname = "wrestling_scene"
    scene.option.timestep = 0.002
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene.visual.global_.offwidth = 960
    scene.visual.global_.offheight = 720
    scene.worldbody.add_light(pos=[2.5, 0, 3.0], dir=[-0.5, 0, -1],
                              type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    scene.add_material(name="mat", rgba=[0.60, 0.45, 0.32, 1.0])
    scene.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                             size=[0, 0, 0.05], material="mat",
                             friction=[0.7, 0.005, 0.0001], condim=3)
    for prefix, x in (("a_", -STAND_SEPARATION / 2), ("b_", STAND_SEPARATION / 2)):
        spec = load_g1_spec()
        frame = scene.worldbody.add_frame(name=f"robot_{prefix[:-1]}",
                                          pos=[x, 0, 0])
        scene.attach(spec, prefix=prefix, suffix="", frame=frame)

    # combined keyframe: both robots standing, facing each other along ±x
    single = load_g1_spec().compile()
    mujoco.mj_id2name  # (API present)
    stand = np.array(single.key_qpos[0], dtype=np.float64)  # (36,)
    key = scene.add_key(name="both_stand")
    kq = np.zeros(2 * 36)
    kq[0:36] = stand
    kq[0] = -STAND_SEPARATION / 2
    kq[36:72] = stand
    kq[36] = STAND_SEPARATION / 2
    # robot B rotated 180 deg about z (G1 faces +x)
    kq[39] = 0.0  # qw
    kq[42] = 1.0  # qz
    key.qpos = list(kq)
    return scene


def load_scene_model() -> mujoco.MjModel:
    """Compiled two-robot scene model (nq=72, nu=58, nsite=46)."""
    return build_scene_spec().compile()


def write_scene_xml(path: str | Path = SCENE_XML) -> Path:
    """Serialize the composed scene to XML (mesh paths made relative to robots/)."""
    scene = build_scene_spec()
    xml = scene.to_xml()
    xml = xml.replace('<compiler angle="radian"/>',
                      '<compiler angle="radian" meshdir="g1/assets"/>', 1)
    path = Path(path)
    path.write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(path))  # verify it loads
    assert model.nq == 72 and model.nu == 58, (model.nq, model.nu)
    return path


def robot_slice(model: mujoco.MjModel, prefix: str) -> slice:
    """qpos slice of one robot ('a_' or 'b_')."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                            f"{prefix}floating_base_joint")
    start = int(model.jnt_qposadr[jid])
    return slice(start, start + 36)


if __name__ == "__main__":  # self-check
    m = load_scene_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_KEY, "both_stand"))
    mujoco.mj_forward(m, d)
    print(f"scene: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} "
          f"nsite={m.nsite} nkey={m.nkey} timestep={m.opt.timestep}")
    print("keys:", [m.key(i).name for i in range(m.nkey)])
    ok = all(int(m.actuator_trnid[i, 0]) == mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator(i).name)
        for i in range(m.nu))
    print("actuator-joint pairing preserved:", ok)
    for prefix in ("a_", "b_"):
        sl = robot_slice(m, prefix)
        print(f"{prefix}: qpos[{sl.start}:{sl.stop}] pelvis={d.qpos[sl][0:7].round(3)}")
        ncon = sum(1 for c in range(d.ncon)
                   if m.geom_bodyid[d.contact.geom1[c]] ==
                   m.geom_bodyid[d.contact.geom2[c]])
    ncon = d.ncon
    print("contacts at both_stand:", ncon)
