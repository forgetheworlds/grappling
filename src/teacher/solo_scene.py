"""Single-G1 scene for the solo drill (one robot, no bystander body).

The solo milestone needs exactly ONE G1 in the physics: a falling bystander
changes contacts and compute cost, and a camera crop is not a single-robot
simulation.  This module composes the same scene ingredients as the
two-robot wrestling scene (``retarget.scene.build_scene_spec``) — mat,
light, the vendored ``robots/g1/g1.xml`` plus the landmark sites the teacher
reads (``retarget.landmarks.load_g1_spec``) — but attaches exactly one robot,
with NO name prefix.

Contract (verified by ``__main__``):

* ``nq == 36``, ``nv == 35``, ``nu == 29`` (one actuator per hinge);
* keyframe ``stand`` = the model's own verified-stable standing pose
  (notes.md 'G1Model': pelvis 0.7900 -> 0.7916 m over 5 s, tilt <= 0.17 deg);
* the teacher's site names resolve unprefixed: ``core``, ``left_toe``,
  ``left_heel``, ``right_toe``, ``right_heel`` (plus the full landmark set).

``load_single_model`` re-compiles the spec directly (fast, deterministic);
``write_single_xml`` serializes it next to the vendored robot for inspection.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from retarget.landmarks import load_g1_spec

REPO = Path(__file__).resolve().parents[2]
SINGLE_XML = REPO / "robots" / "solo_scene.xml"


def build_single_spec() -> mujoco.MjSpec:
    """MjSpec of the solo scene: floor, light, one G1 (no prefix)."""
    scene = mujoco.MjSpec()
    scene.modelname = "solo_scene"
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
    frame = scene.worldbody.add_frame(name="robot")
    scene.attach(load_g1_spec(), prefix="", suffix="", frame=frame)
    return scene


def load_single_model() -> mujoco.MjModel:
    """Compiled single-G1 scene model (nq=36, nu=29, one robot)."""
    model = build_single_spec().compile()
    assert model.nq == 36 and model.nu == 29, (model.nq, model.nu)
    return model


def write_single_xml(path: str | Path = SINGLE_XML) -> Path:
    """Serialize the solo scene (mesh paths relative to robots/)."""
    xml = build_single_spec().to_xml()
    xml = xml.replace('<compiler angle="radian"/>',
                      '<compiler angle="radian" meshdir="g1/assets"/>', 1)
    path = Path(path)
    path.write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(path))   # verify it loads
    assert model.nq == 36 and model.nu == 29, (model.nq, model.nu)
    return path


def stand_qpos(model: mujoco.MjModel) -> np.ndarray:
    """The model's own standing keyframe as a (36,) qpos."""
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if kid < 0:
        kid = 0
    return model.key_qpos[kid].copy()


if __name__ == "__main__":  # self-check
    m = load_single_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s)
             for s in range(m.nsite)]
    print(f"solo scene: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} "
          f"nsite={m.nsite} keys={[m.key(i).name for i in range(m.nkey)]}")
    print("has teacher sites:", all(n in names for n in
                                    ("core", "left_toe", "left_heel",
                                     "right_toe", "right_heel")))
    print("stand pelvis z:", round(float(d.qpos[2]), 4),
          "contacts:", int(d.ncon))
    ok = all(int(m.actuator_trnid[i, 0]) ==
             mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator(i).name)
             for i in range(m.nu))
    print("actuator-joint pairing preserved:", ok)
