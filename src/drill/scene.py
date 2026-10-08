"""Single-G1 drill scene: floor + one Unitree G1 (MjSpec-composed).

Milestone constraint (``docs/SOLO_DRILL.md``): ONE robot.  The model is the
vendored ``robots/g1/g1.xml`` plus the 19 retargeting landmark sites from
``src/retarget/landmarks.py`` (consumed read-only: toe/heel sites drive foot
contact logic, knee/wrist/head sites drive the HUD and the rubric probes).
No second robot, no wall, no mat ring: the floor is a checkered plane so
displacement is readable in the evidence clips.

Keyframes
---------
``stand``      -- the vendored keyframe (proven to hold: E3 notes).
``both_stand`` -- name kept from the two-robot scene for the teacher's
                  helpers (``_rest_sole_z``/``_safe_posture`` look it up);
                  holds the drill stance built by :mod:`drill.posture`.

Units SI, radians, Z-up world; base quaternion is w-first (MuJoCo).
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

SCENE_XML = REPO / "robots" / "solo_drill_scene.xml"

#: render/capture size fixed by the milestone contract
WIDTH, HEIGHT = 960, 720

#: site height counting as sole contact (m) — G1 sole sites sit ~2 cm up when flat
FOOT_TOUCH_Z = 0.035


def build_spec(stance: np.ndarray | None = None) -> mujoco.MjSpec:
    """Spec = one G1 (with landmark sites) + checkered floor + light + camera."""
    from retarget.landmarks import load_g1_spec

    spec = load_g1_spec()
    # sole contact points: the G1 foot is four 5 mm spheres; expose each centre
    # as a site so the leg IK and the metrics can read them directly.
    for side in ("left", "right"):
        body = spec.body(f"{side}_ankle_roll_link")
        k = 0
        for geom in body.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
                body.add_site(name=f"{side}_sole{k}", pos=list(geom.pos))
                k += 1
        assert k == 4, f"{side} foot: expected 4 sole spheres, found {k}"
    # a checkered floor makes displacement readable in the evidence clips
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     width=300, height=300, rgb1=[0.22, 0.26, 0.31],
                     rgb2=[0.34, 0.39, 0.45])
    spec.add_material(name="grid", textures=["grid"], texuniform=True,
                      texrepeat=[6.0, 6.0])
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05],
        pos=[0.0, 0.0, 0.0], material="grid",
        condim=4, friction=[1.0, 0.005, 0.0001])
    spec.worldbody.add_light(name="key", pos=[2.5, -2.0, 3.0], dir=[-0.6, 0.5, -1.0],
                             diffuse=[0.8, 0.8, 0.8], specular=[0.3, 0.3, 0.3], castshadow=False)
    spec.worldbody.add_light(name="fill", pos=[-2.0, 2.5, 2.0], dir=[0.5, -0.6, -1.0],
                             diffuse=[0.4, 0.4, 0.45], specular=[0.1, 0.1, 0.1], castshadow=False)
    # fixed side-3/4 camera (whole robot in frame); the renderer may also use a
    # free camera — both are the same framing, see video.CAM.
    spec.worldbody.add_camera(name="side34", pos=[2.55, -2.55, 1.55],
                              xyaxes=[0.707, 0.707, 0.0, -0.19, 0.19, 0.96], fovy=38.0)
    spec.worldbody.add_camera(name="front", pos=[3.6, -0.2, 1.35],
                              xyaxes=[0.055, 0.998, 0.0, -0.12, 0.007, 0.99], fovy=38.0)
    try:
        spec.visual.global_.offwidth = WIDTH
        spec.visual.global_.offheight = HEIGHT
    except Exception:                                   # pragma: no cover - api drift
        pass
    if stance is not None:
        q = np.asarray(stance, float)[:36].copy()
        spec.add_key(name="both_stand", qpos=q, ctrl=q[7:36])
    return spec


def load_model(stance: np.ndarray | None = None) -> mujoco.MjModel:
    """Compiled single-G1 drill model (nq=36, nu=29)."""
    return build_spec(stance).compile()


def write_scene_xml(path: Path | str = SCENE_XML, stance: np.ndarray | None = None) -> Path:
    """Serialize the composed scene (mesh paths made relative to ``robots/``)."""
    spec = build_spec(stance)
    text = spec.to_xml()
    text = text.replace(str(REPO / "robots" / "g1" / ""), "g1/")
    p = Path(path)
    p.write_text(text)
    return p


def keyframe(model: mujoco.MjModel, name: str) -> np.ndarray:
    """qpos of a keyframe (36,)."""
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if kid < 0:
        raise KeyError(f"keyframe {name!r} not in model")
    return np.array(model.key_qpos[kid], float)


if __name__ == "__main__":                              # self-check
    m = load_model()
    assert m.nq == 36 and m.nu == 29 and m.nsite >= 19, (m.nq, m.nu, m.nsite)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
    for need in ("core", "left_toe", "left_heel", "right_toe", "right_heel",
                 "left_knee", "right_knee", "left_wrist", "right_wrist", "head"):
        assert need in names, f"missing site {need}"
    print(f"solo drill scene ok: nq={m.nq} nu={m.nu} nsite={m.nsite} "
          f"timestep={m.opt.timestep} mass={m.body_mass.sum():.3f} kg")
    print("sites:", ", ".join(names))
    d = mujoco.MjData(m)
    d.qpos[:] = keyframe(m, "stand")
    mujoco.mj_forward(m, d)
    print(f"stand keyframe pelvis z={d.qpos[2]:.3f} contacts={d.ncon}")
