"""GrappleMap joint -> Unitree G1 landmark-site mapping.

Every GrappleMap landmark (23 per player, see src/grapplemap/parser.py) is
mapped to a MuJoCo <site> that this module attaches to the G1 spec at load
time (``mujoco.MjSpec``; ``robots/g1/g1.xml`` itself is NEVER modified).

Mapping table (verified against the model; G1 site local frames):
  GM joint          G1 site             body                      local pos      weight
  ---------------   ------------------  ------------------------  -------------  -----
  Core              core                pelvis                    (0,0,0)        HIGH
  Neck              neck                torso_link                (0,0,0.285)    HIGH
  Head              head                torso_link                (0,0,0.385)    HIGH
  LeftHip           left_hip            left_hip_pitch_link      (0,0,0)        HIGH
  RightHip          right_hip           right_hip_pitch_link     (0,0,0)        HIGH
  LeftKnee          left_knee           left_knee_link           (0,0,0)        HIGH
  RightKnee         right_knee          right_knee_link          (0,0,0)        HIGH
  LeftAnkle         left_ankle          left_ankle_pitch_link    (0,0,0)        HIGH
  RightAnkle        right_ankle         right_ankle_pitch_link   (0,0,0)        HIGH
  LeftShoulder      left_shoulder       left_shoulder_pitch_link (0,0,0)        HIGH
  RightShoulder     right_shoulder      right_shoulder_pitch_link(0,0,0)       HIGH
  LeftElbow         left_elbow          left_elbow_link          (0,0,0)        MED
  RightElbow        right_elbow         right_elbow_link         (0,0,0)        MED
  LeftWrist         left_wrist          left_wrist_roll_link     (0,0,0)        MED
  RightWrist        right_wrist         right_wrist_roll_link    (0,0,0)        MED
  LeftHand          left_wrist          (same site as Wrist; the G1 rubber hand
                                          is rigid with the wrist chain)
  RightHand         right_wrist         (same)
  LeftToe           left_toe            left_ankle_roll_link     (0.125,0,-0.03) LOW
  RightToe          right_toe           right_ankle_roll_link    (0.125,0,-0.03) LOW
  LeftHeel          left_heel           left_ankle_roll_link     (-0.055,0,-0.03) LOW
  RightHeel         right_heel          right_ankle_roll_link    (-0.055,0,-0.03) LOW
  LeftFingers       -- DROPPED (G1 has no fingers; wrist site covers the hand)
  RightFingers      -- DROPPED

Site placements (all verified on the compiled model at the "stand" keyframe):
- joint-center sites sit exactly at the hinge origins (anatomical centers).
- neck/head: the G1 has no head body; the head is a collision geom on
  torso_link. The head mesh (compiled mesh 'head_link') COG sits at
  torso-local (0.005, 0, 0.385) (mesh AABB half-height 0.10, author pos
  (0.004, 0, -0.044) + compile mesh_pos (0.004, 0, 0.429)); the neck sits at
  the bottom of the head geom, torso-local z = 0.285.
- toe/heel sites sit at the front/rear foot contact spheres
  (ankle_roll-local x = +0.125 front sphere, x = -0.055 rear sphere, both at
  the sole plane z = -0.03).

Weight presets for the per-frame solve residual (weight * site distance):
  HIGH: pelvis(core), torso(neck), head, hips, knees, ankles, shoulders
  MED : elbows, wrists
  LOW : toes, heels
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from grapplemap import JOINTS

G1_XML = Path(__file__).resolve().parents[2] / "robots" / "g1" / "g1.xml"

#: (site name, body name, local position) — one robot, unprefixed.
SITE_SPECS: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
    ("core", "pelvis", (0.0, 0.0, 0.0)),
    ("neck", "torso_link", (0.0, 0.0, 0.285)),
    ("head", "torso_link", (0.0, 0.0, 0.385)),
    ("left_hip", "left_hip_pitch_link", (0.0, 0.0, 0.0)),
    ("right_hip", "right_hip_pitch_link", (0.0, 0.0, 0.0)),
    ("left_knee", "left_knee_link", (0.0, 0.0, 0.0)),
    ("right_knee", "right_knee_link", (0.0, 0.0, 0.0)),
    ("left_ankle", "left_ankle_pitch_link", (0.0, 0.0, 0.0)),
    ("right_ankle", "right_ankle_pitch_link", (0.0, 0.0, 0.0)),
    ("left_toe", "left_ankle_roll_link", (0.125, 0.0, -0.03)),
    ("right_toe", "right_ankle_roll_link", (0.125, 0.0, -0.03)),
    ("left_heel", "left_ankle_roll_link", (-0.055, 0.0, -0.03)),
    ("right_heel", "right_ankle_roll_link", (-0.055, 0.0, -0.03)),
    ("left_shoulder", "left_shoulder_pitch_link", (0.0, 0.0, 0.0)),
    ("right_shoulder", "right_shoulder_pitch_link", (0.0, 0.0, 0.0)),
    ("left_elbow", "left_elbow_link", (0.0, 0.0, 0.0)),
    ("right_elbow", "right_elbow_link", (0.0, 0.0, 0.0)),
    ("left_wrist", "left_wrist_roll_link", (0.0, 0.0, 0.0)),
    ("right_wrist", "right_wrist_roll_link", (0.0, 0.0, 0.0)),
)

SOLVED_SITES: tuple[str, ...] = tuple(s for s, _, _ in SITE_SPECS)  # 19 sites

#: GrappleMap joint -> G1 site name (None = dropped).
GM_JOINT_TO_SITE: dict[str, str | None] = {
    "LeftToe": "left_toe", "RightToe": "right_toe",
    "LeftHeel": "left_heel", "RightHeel": "right_heel",
    "LeftAnkle": "left_ankle", "RightAnkle": "right_ankle",
    "LeftKnee": "left_knee", "RightKnee": "right_knee",
    "LeftHip": "left_hip", "RightHip": "right_hip",
    "LeftShoulder": "left_shoulder", "RightShoulder": "right_shoulder",
    "LeftElbow": "left_elbow", "RightElbow": "right_elbow",
    "LeftWrist": "left_wrist", "RightWrist": "right_wrist",
    "LeftHand": "left_wrist", "RightHand": "right_wrist",
    "LeftFingers": None, "RightFingers": None,
    "Core": "core", "Neck": "neck", "Head": "head",
}

DROPPED_JOINTS: tuple[str, ...] = tuple(
    j for j, s in GM_JOINT_TO_SITE.items() if s is None)

#: For each GM joint used by the solver: (site index into SOLVED_SITES, weight class)
@dataclass(frozen=True)
class _Map:
    joint: str
    site: str
    site_idx: int
    weight_class: str


WEIGHT_CLASSES: dict[str, str] = {}
for _j in JOINTS:
    _s = GM_JOINT_TO_SITE[_j]
    if _s is None:
        continue
    if _j in ("Core", "Neck", "Head", "LeftHip", "RightHip", "LeftKnee",
              "RightKnee", "LeftAnkle", "RightAnkle", "LeftShoulder",
              "RightShoulder"):
        WEIGHT_CLASSES[_j] = "HIGH"
    elif _j in ("LeftElbow", "RightElbow", "LeftWrist", "RightWrist",
                "LeftHand", "RightHand"):
        WEIGHT_CLASSES[_j] = "MED"
    else:
        WEIGHT_CLASSES[_j] = "LOW"

#: Residual weight presets (class -> weight). 'strict' is the repair profile.
WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "default": {"HIGH": 4.0, "MED": 1.5, "LOW": 0.5},
    "strict": {"HIGH": 8.0, "MED": 3.0, "LOW": 1.0},
}


def gm_weights(preset: str = "default") -> np.ndarray:
    """(23,) weight vector in GM JOINTS order; dropped joints get weight 0."""
    pw = WEIGHT_PRESETS[preset]
    return np.array([pw[WEIGHT_CLASSES[j]] if j in WEIGHT_CLASSES else 0.0
                     for j in JOINTS], dtype=np.float64)


def site_weights(preset: str = "default") -> np.ndarray:
    """(19,) weight per SOLVED_SITES entry: max class weight of mapped joints."""
    pw = WEIGHT_PRESETS[preset]
    w = np.zeros(len(SOLVED_SITES))
    for j, cls in WEIGHT_CLASSES.items():
        idx = SOLVED_SITES.index(GM_JOINT_TO_SITE[j])
        w[idx] = max(w[idx], pw[cls])
    return w


def gm_joint_to_site_rows() -> tuple[np.ndarray, np.ndarray]:
    """Row indices mapping GM joints (23,) into the solver residual.

    Returns (rows, valid): for each GM joint, the site row index (0..18) or
    -1 if dropped; valid = boolean mask of mapped joints (21 of 23).
    """
    rows = np.full(len(JOINTS), -1, dtype=np.int64)
    for i, j in enumerate(JOINTS):
        s = GM_JOINT_TO_SITE[j]
        if s is not None:
            rows[i] = SOLVED_SITES.index(s)
    return rows, rows >= 0


def load_g1_spec() -> mujoco.MjSpec:
    """MjSpec of the single G1 with the 19 landmark sites attached.

    robots/g1/g1.xml is read, never written.
    """
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    for name, body, local in SITE_SPECS:
        spec.body(body).add_site(name=name, pos=list(local))
    return spec


# ---- G1 reference geometry (computed once from the compiled model) ---------

def g1_reference_geometry() -> dict:
    """Site positions + segment lengths of the G1 at the 'stand' keyframe."""
    spec = load_g1_spec()
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    sites = {n: data.site_xpos[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, n)].copy() for n in SOLVED_SITES}

    def seg(a: str, b: str) -> float:
        return float(np.linalg.norm(sites[a] - sites[b]))

    geometry = {
        "sites": sites,
        "segments": {
            "core_neck": seg("core", "neck"),
            "neck_head": seg("neck", "head"),
            "thigh": seg("left_hip", "left_knee"),
            "thigh_r": seg("right_hip", "right_knee"),
            "shank": seg("left_knee", "left_ankle"),
            "shank_r": seg("right_knee", "right_ankle"),
            "core_head": seg("core", "head"),
            "upper_arm": seg("left_shoulder", "left_elbow"),
            "forearm": seg("left_elbow", "left_wrist"),
        },
        "pelvis_z": float(data.qpos[2]),
    }
    return geometry


if __name__ == "__main__":  # self-check
    geo = g1_reference_geometry()
    print("G1 landmark sites (world, stand keyframe):")
    for n, p in geo["sites"].items():
        print(f"  {n:16s} {np.round(p, 3)}")
    print("G1 segments (m):")
    for k, v in geo["segments"].items():
        print(f"  {k:12s} {v:.4f}")
    print(f"pelvis z = {geo['pelvis_z']:.3f}")
    print(f"mapped GM joints: {sum(1 for v in GM_JOINT_TO_SITE.values() if v)}, "
          f"dropped: {DROPPED_JOINTS}")
