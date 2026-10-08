"""Single-G1 scene for the solo drill: ``a_``-prefixed robot + floor + markers.

Why a composed scene rather than ``robots/g1/scene.xml`` directly
-----------------------------------------------------------------
The vendored single-robot scene names the robot's bodies/joints/actuators
*without* a robot prefix (``pelvis``, ``torso_link``, ...), while every shared,
name-based helper in this repo (``solo.detector.body_maps``,
``src/teacher``'s ``a_``-prefixed robot context, the scorer) resolves
``a_<name>``.  Rather than editing the vendored model (out of bounds) or
parameterising other agents' modules, this module composes the same robot with
the ``a_`` prefix via ``MjSpec.attach(prefix="a_")`` -- the mechanism the
two-robot scene already uses (``src/retarget/scene.py``).  Verified on the
compiled model (test_solo.py asserts the exact name set).

Model facts (measured on this host, 2026-10-08)
-----------------------------------------------
* ``nq`` 36 (free base 7 + 29 hinges), ``nv`` 35, ``nu`` 29, ``nbody`` 32,
  ``nsite`` 27, ``ngeom`` 72, ``nkey`` 1 (``a_stand``)
* timestep 0.002 s (500 Hz), integrator ``implicitfast``, total mass 33.341 kg
* actuators: 29 position servos, ``ctrl`` in rad, ``ctrlrange`` = joint range
  (``inheritrange=1``); the ``a_stand`` keyframe is the verified-stable pose
  (5 s hold: +0.16 cm pelvis drift, <=0.172 deg torso tilt; g1_model.md).

Names this module guarantees (asserted by :func:`resolved_dependencies` and
``tests/test_solo.py``): the robot prefix ``a_`` on every body/joint/actuator/
site/sensor/keyframe of g1.xml; ``floor`` (unprefixed, shared convention with
``solo.detector.FLOOR_GEOM``); the 19 retargeting landmark sites
(``a_core``, ``a_left_knee``, ... from ``src/retarget/landmarks.py``); the
mocap body ``virtual_opponent`` with the four spatial markers
``a_marker_pelvis``, ``a_marker_leg_l``, ``a_marker_leg_r``,
``a_marker_hand_l``, ``a_marker_hand_r``.

The markers are *spatial targets only*: they define the intended shot entry
region (opponent pelvis + leg/hand targets).  They carry no grip, contact or
force semantics and must never be presented as a physical opponent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
G1_DIR = REPO / "robots" / "g1"
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:  # support both `import solo.*` and `python -m src.solo.*`
    sys.path.insert(0, str(_SRC))

#: robot name prefix (shared convention with the retarget scene)
PREFIX = "a_"
#: floor geom name (``solo.detector.FLOOR_GEOM``; unprefixed by convention)
FLOOR_GEOM = "floor"

#: control rate / physics rate (verified: model timestep = 0.002 s)
CONTROL_HZ = 50.0
STEP_DT = 1.0 / CONTROL_HZ          # 0.02 s
MODEL_DT = 0.002                    # s, verified per scene at load
SUBSTEPS = int(round(STEP_DT / MODEL_DT))   # 10
N_JOINTS = 29

#: torso body -- the push target and uprightness reference
TORSO_BODY = PREFIX + "torso_link"
PELVIS_BODY = PREFIX + "pelvis"
FREE_JOINT = PREFIX + "floating_base_joint"

#: virtual-opponent mocap body + marker sites (local offsets in that body)
VIRTUAL_OPPONENT_BODY = "virtual_opponent"
#: marker name -> (local pos in the virtual-opponent body frame, rgba, size)
MARKER_SPECS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float], float]] = {
    "a_marker_pelvis": ((0.0, 0.0, 0.0), (0.95, 0.15, 0.15, 0.55), 0.045),
    "a_marker_leg_l": ((0.10, 0.13, -0.30), (0.95, 0.75, 0.15, 0.50), 0.035),
    "a_marker_leg_r": ((0.10, -0.13, -0.30), (0.95, 0.75, 0.15, 0.50), 0.035),
    "a_marker_hand_l": ((0.22, 0.26, 0.15), (0.25, 0.85, 0.45, 0.50), 0.03),
    "a_marker_hand_r": ((0.22, -0.26, 0.15), (0.25, 0.55, 0.95, 0.50), 0.03),
}
MARKER_NAMES: tuple[str, ...] = tuple(MARKER_SPECS)
#: foot contact bodies (both feet share the ankle_roll body in g1.xml)
FOOT_BODIES = (PREFIX + "left_ankle_roll_link", PREFIX + "right_ankle_roll_link")
#: bodies whose floor contact is a *diagnostic* (never terminal by itself)
KNEE_BODIES = (PREFIX + "left_knee_link", PREFIX + "right_knee_link")
#: arm-support bodies: wrist + elbow links ("hands" in the termination rule)
HAND_BODIES = (PREFIX + "left_wrist_roll_link", PREFIX + "left_wrist_yaw_link",
               PREFIX + "left_elbow_link", PREFIX + "right_wrist_roll_link",
               PREFIX + "right_wrist_yaw_link", PREFIX + "right_elbow_link")
#: diagnostic limb set (matches wrestling.backdet.LIMB_BODIES semantics)
LIMB_BODIES = KNEE_BODIES + HAND_BODIES


def build_solo_spec() -> mujoco.MjSpec:
    """Compose the solo scene spec: floor + light + prefixed G1 + markers."""
    from retarget.landmarks import load_g1_spec  # deferred: shares the site table

    scene = mujoco.MjSpec()
    scene.modelname = "solo_drill_scene"
    scene.option.timestep = MODEL_DT
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene.visual.global_.offwidth = 960
    scene.visual.global_.offheight = 720
    scene.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]
    scene.worldbody.add_light(pos=[2.0, 0.0, 3.5], dir=[-0.3, 0.0, -1.0],
                              type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    # checkered ground plane, same look as robots/g1/scene.xml
    scene.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D,
                      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                      width=300, height=300, rgb1=[0.2, 0.3, 0.4],
                      rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8])
    scene.add_material(name="groundplane", textures=["groundplane"],
                       texuniform=True, texrepeat=[5, 5], reflectance=0.2)
    scene.worldbody.add_geom(name=FLOOR_GEOM, type=mujoco.mjtGeom.mjGEOM_PLANE,
                             size=[0, 0, 0.05], material="groundplane")
    # the robot, prefixed "a_" (19 landmark sites come with the spec)
    scene.attach(load_g1_spec(), prefix=PREFIX, suffix="",
                 frame=scene.worldbody.add_frame(name="robot_a", pos=[0, 0, 0]))
    # virtual-opponent markers: one mocap body, four site+geom pairs, no collisions
    vo = scene.worldbody.add_body(name=VIRTUAL_OPPONENT_BODY, mocap=True,
                                  pos=[1.0, 0.0, 0.79])
    for name, (pos, rgba, size) in MARKER_SPECS.items():
        vo.add_site(name=name, pos=list(pos), size=[size],
                    rgba=list(rgba))
        vo.add_geom(name=name + "_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    pos=list(pos), size=[size], rgba=list(rgba),
                    contype=0, conaffinity=0, group=2, mass=0.0)
    return scene


_model_cache: mujoco.MjModel | None = None


def load_solo_model() -> mujoco.MjModel:
    """Compiled single-G1 scene (nq=36, nu=29), cached per process."""
    global _model_cache
    if _model_cache is None:
        model = build_solo_spec().compile()
        assert model.nq == 36 and model.nv == 35 and model.nu == N_JOINTS, \
            (model.nq, model.nv, model.nu)
        assert abs(model.opt.timestep - MODEL_DT) < 1e-12, model.opt.timestep
        _model_cache = model
    return _model_cache


def body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"body {name!r} not in model")
    return bid


def site_id(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise KeyError(f"site {name!r} not in model")
    return sid


def stand_frame(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """(qpos36, ctrl29) of the verified-stable ``a_stand`` keyframe."""
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, PREFIX + "stand")
    if kid < 0:
        raise KeyError(f"keyframe {PREFIX}stand not in model")
    qpos = np.asarray(model.key_qpos[kid], dtype=np.float64).copy()
    ctrl = np.asarray(model.key_ctrl[kid], dtype=np.float64).copy()
    if qpos.size != 36 or ctrl.size != N_JOINTS:
        raise ValueError((qpos.shape, ctrl.shape))
    return qpos, ctrl


def joint_qpos_slice(model: mujoco.MjModel) -> slice:
    """qpos slice of the 29 hinges (pelvis free joint occupies 0:7)."""
    return slice(7, 36)


def joint_dof_slice(model: mujoco.MjModel) -> slice:
    """qvel slice of the 29 hinges (free joint dofs occupy 0:6)."""
    return slice(6, 35)


def ctrl_range(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) ctrlrange of the 29 position actuators (rad)."""
    return (np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float64).copy(),
            np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float64).copy())


def model_facts(model: mujoco.MjModel | None = None) -> dict:
    """Measured structural facts (S1 report evidence)."""
    m = model if model is not None else load_solo_model()
    mass = float(sum(float(m.body_mass[i]) for i in range(m.nbody)))
    return {
        "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
        "nbody": int(m.nbody), "nsite": int(m.nsite), "ngeom": int(m.ngeom),
        "nkey": int(m.nkey), "nmocap": int(m.nmocap),
        "timestep": float(m.opt.timestep), "control_dt": STEP_DT,
        "substeps": SUBSTEPS, "total_mass": round(mass, 3),
        "keyframes": [m.key(i).name for i in range(m.nkey)],
        "floor_friction": [round(float(x), 6) for x in
                           m.geom_friction[mujoco.mj_name2id(
                               m, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)]],
    }


def resolved_dependencies(model: mujoco.MjModel | None = None) -> dict:
    """Every model name this package depends on, resolved to ids (test hook).

    Returns ``{"bodies": {name: id}, "sites": ..., "joints": ..., "actuators":
    ..., "keys": ..., "geoms": ...}``; raises KeyError listing the first
    unresolved dependency.
    """
    m = model if model is not None else load_solo_model()
    out: dict[str, dict[str, int]] = {"bodies": {}, "sites": {}, "joints": {},
                                      "actuators": {}, "keys": {}, "geoms": {}}
    deps = {
        "bodies": (mujoco.mjtObj.mjOBJ_BODY,
                   [PELVIS_BODY, TORSO_BODY, *FOOT_BODIES, *KNEE_BODIES,
                    *HAND_BODIES, VIRTUAL_OPPONENT_BODY]),
        "sites": (mujoco.mjtObj.mjOBJ_SITE,
                  [PREFIX + n for n in
                   ("left_foot", "right_foot", "core", "head", "left_knee",
                    "right_knee", "left_ankle", "right_ankle", "left_wrist",
                    "right_wrist", "left_toe", "right_toe", "left_heel",
                    "right_heel", "left_hip", "right_hip", "neck")] + list(MARKER_NAMES)),
        "joints": (mujoco.mjtObj.mjOBJ_JOINT, [FREE_JOINT]),
        "actuators": (mujoco.mjtObj.mjOBJ_ACTUATOR,
                      [PREFIX + n for n in
                       ("left_hip_pitch_joint", "left_knee_joint",
                        "left_ankle_roll_joint", "waist_pitch_joint",
                        "right_wrist_yaw_joint")]),
        "keys": (mujoco.mjtObj.mjOBJ_KEY, [PREFIX + "stand"]),
        "geoms": (mujoco.mjtObj.mjOBJ_GEOM, [FLOOR_GEOM]),
    }
    for kind, (obj, names) in deps.items():
        for name in names:
            i = mujoco.mj_name2id(m, obj, name)
            if i < 0:
                raise KeyError(f"unresolved {kind[:-1]} dependency: {name!r}")
            out[kind][name] = int(i)
    return out


if __name__ == "__main__":  # self-check
    m = load_solo_model()
    facts = model_facts(m)
    deps = resolved_dependencies(m)
    q, c = stand_frame(m)
    print("solo.scene self-check OK:", facts)
    print("dependencies resolved:", {k: len(v) for k, v in deps.items()},
          "| stand pelvis z:", round(float(q[2]), 4))
