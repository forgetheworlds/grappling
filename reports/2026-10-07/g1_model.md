# Unitree G1 MJCF Model — Bring-up Report (Phase 1)

Date: 2026-10-07 · Agent: G1Model · MuJoCo 3.15.0 (CPU, aarch64, MUJOCO_GL=egl)

## Summary

- Vendored the Unitree G1 (29-dof rev 1.0) from `mujoco_menagerie` into
  `robots/g1/` (MJCF + 49 STL meshes + LICENSE, BSD-3-Clause).
- `scripts/smoke_g1.py` runs green from repo root (exit 0): dumps the full
  spec, holds the shipped `stand` keyframe for 5 s on the model's own
  position servos, and renders front + side PNGs.
- **Verdict: the model STANDS.** Over 5 s: pelvis height drift +0.16 cm,
  torso up-axis tilt ≤ 0.172°, horizontal drift 0.1 mm, 8 foot contacts
  maintained throughout. Measured, not assumed.

## Variants available and choice

`mujoco_menagerie` (cloned depth-1 to `third_party/menagerie`) ships exactly
three G1 robot variants under `unitree_g1/` — **no 23-dof variant exists in
menagerie** (23-dof lives only in Unitree's own `unitree_ros/g1_description`,
which menagerie does not vendor):

| file | dof (nu) | colliders | keyframes | servos | notes |
|---|---|---|---|---|---|
| `g1.xml` | 29 (nq 36) | original meshes | `stand` | kp=500, dampratio=1 | canonical, derived from Unitree `g1_29dof_rev_1_0` |
| `g1_mjx.xml` (+`scene_mjx.xml`) | 29 (nq 36) | hand-built capsules + explicit contact pairs | `home`, `knees_bent` (in scene_mjx) | kp=75, kv=2 | tuned for MJX/GPU pipeline |
| `g1_with_hands.xml` | 43 (nq 50) | meshes | `stand` | kp=500 | +7 joints per 3-finger hand |

**Choice: `g1.xml` (29 dof).** Rationale:
1. No 23-dof variant exists in menagerie — cannot prefer what isn't there
   (documented above instead).
2. 29 dof keeps all three wrist dofs per arm, which Phase-2 retargeting of
   GrappleMap arm/hand relationships needs; the 23-dof G1 drops wrists.
3. `g1_mjx.xml` exists for the MJX GPU stack; we are CPU-only and its
   lower gains (kp=75) plus disabled-contact default are a worse fit here.
4. Hands variant adds 14 finger dofs we have no use for yet (GrappleMap
   relationships are palm/grip level); fewer dofs = cheaper RL later.
   Hands can be swapped in later without touching the pipeline (same body
   tree up to the wrist).

## Model spec (measured from `robots/g1/scene.xml` via MuJoCo 3.15.0)

| quantity | value |
|---|---|
| model name | `g1_29dof_rev_1_0` (menagerie) |
| nq / nv | 36 / 35 (free base 7 + 29 hinges; 6 + 29 dofs) |
| nu / na | 29 / 0 (position servos, no activation state) |
| nbody / ngeom | 31 / 72 |
| timestep | 0.002 s, integrator `implicitfast` |
| total mass | 33.341 kg |
| keyframes | 1: `stand` |
| actuator transmission | all joint-position; ctrlrange = joint range (`inheritrange`) |
| base | free joint `floating_base_joint` on pelvis (body `pelvis`, z=0.793 in XML) |

### Joint list (order = qpos layout; ranges in radians; XML `angle="radian"`)

| # | qpos adr | joint | type | range [rad] | force limit [N·m] |
|---|---|---|---|---|---|
| 0 | 0 (7) | floating_base_joint | free | unlimited | — |
| 1 | 7 | left_hip_pitch_joint | hinge | [−2.5307, +2.8798] | ±88 |
| 2 | 8 | left_hip_roll_joint | hinge | [−0.5236, +2.9671] | ±139 |
| 3 | 9 | left_hip_yaw_joint | hinge | [−2.7576, +2.7576] | ±88 |
| 4 | 10 | left_knee_joint | hinge | [−0.0873, +2.8798] | ±139 |
| 5 | 11 | left_ankle_pitch_joint | hinge | [−0.8727, +0.5236] | ±50 |
| 6 | 12 | left_ankle_roll_joint | hinge | [−0.2618, +0.2618] | ±50 |
| 7 | 13 | right_hip_pitch_joint | hinge | [−2.5307, +2.8798] | ±88 |
| 8 | 14 | right_hip_roll_joint | hinge | [−2.9671, +0.5236] | ±139 |
| 9 | 15 | right_hip_yaw_joint | hinge | [−2.7576, +2.7576] | ±88 |
| 10 | 16 | right_knee_joint | hinge | [−0.0873, +2.8798] | ±139 |
| 11 | 17 | right_ankle_pitch_joint | hinge | [−0.8727, +0.5236] | ±50 |
| 12 | 18 | right_ankle_roll_joint | hinge | [−0.2618, +0.2618] | ±50 |
| 13 | 19 | waist_yaw_joint | hinge | [−2.6180, +2.6180] | ±88 |
| 14 | 20 | waist_roll_joint | hinge | [−0.5200, +0.5200] | ±50 |
| 15 | 21 | waist_pitch_joint | hinge | [−0.5200, +0.5200] | ±50 |
| 16 | 22 | left_shoulder_pitch_joint | hinge | [−3.0892, +2.6704] | ±25 |
| 17 | 23 | left_shoulder_roll_joint | hinge | [−1.5882, +2.2515] | ±25 |
| 18 | 24 | left_shoulder_yaw_joint | hinge | [−2.6180, +2.6180] | ±25 |
| 19 | 25 | left_elbow_joint | hinge | [−1.0472, +2.0944] | ±25 |
| 20 | 26 | left_wrist_roll_joint | hinge | [−1.9722, +1.9722] | ±5 |
| 21 | 27 | left_wrist_pitch_joint | hinge | [−1.6144, +1.6144] | ±5 |
| 22 | 28 | left_wrist_yaw_joint | hinge | [−1.6144, +1.6144] | ±5 |
| 23 | 29 | right_shoulder_pitch_joint | hinge | [−3.0892, +2.6704] | ±25 |
| 24 | 30 | right_shoulder_roll_joint | hinge | [−2.2515, +1.5882] | ±25 |
| 25 | 31 | right_shoulder_yaw_joint | hinge | [−2.6180, +2.6180] | ±25 |
| 26 | 32 | right_elbow_joint | hinge | [−1.0472, +2.0944] | ±25 |
| 27 | 33 | right_wrist_roll_joint | hinge | [−1.9722, +1.9722] | ±5 |
| 28 | 34 | right_wrist_pitch_joint | hinge | [−1.6144, +1.6144] | ±5 |
| 29 | 35 | right_wrist_yaw_joint | hinge | [−1.6144, +1.6144] | ±5 |

All joints carry `armature=0.01`, `frictionloss=0.3` (menagerie defaults).
Note the mirrored hip-roll and shoulder-roll ranges (right side negated).

### Actuators (29, one per hinge, same order; type = position servo)

Force = `kp·(ctrl − q) − kv·qvel`, ctrl in rad, ctrlrange = joint range.
`kp = 500` for all; `kv` from `dampratio=1` (measured values below);
force range: not limited in the actuator (joint `actuatorfrcrange` in table
above clamps realized torque per joint).

| # | actuator = joint | kp | kv | # | actuator = joint | kp | kv |
|---|---|---|---|---|---|---|---|
| 0 | left_hip_pitch | 500 | 10.3 | 15 | left_shoulder_pitch | 500 | 10.8 |
| 1 | left_hip_roll | 500 | 15.8 | 16 | left_shoulder_roll | 500 | 10.0 |
| 2 | left_hip_yaw | 500 | 5.8 | 17 | left_shoulder_yaw | 500 | 7.4 |
| 3 | left_knee | 500 | 9.1 | 18 | left_elbow | 500 | 7.1 |
| 4 | left_ankle_pitch | 500 | 4.9 | 19 | left_wrist_roll | 500 | 4.5 |
| 5 | left_ankle_roll | 500 | 4.5 | 20 | left_wrist_pitch | 500 | 5.0 |
| 6 | right_hip_pitch | 500 | 10.3 | 21 | left_wrist_yaw | 500 | 4.7 |
| 7 | right_hip_roll | 500 | 15.8 | 22 | right_shoulder_pitch | 500 | 10.8 |
| 8 | right_hip_yaw | 500 | 5.8 | 23 | right_shoulder_roll | 500 | 10.0 |
| 9 | right_knee | 500 | 9.1 | 24 | right_shoulder_yaw | 500 | 7.4 |
| 10 | right_ankle_pitch | 500 | 4.9 | 25 | right_elbow | 500 | 7.1 |
| 11 | right_ankle_roll | 500 | 4.5 | 26 | right_wrist_roll | 500 | 4.5 |
| 12 | waist_yaw | 500 | 10.3 | 27 | right_wrist_pitch | 500 | 5.0 |
| 13 | waist_roll | 500 | 16.2 | 28 | right_wrist_yaw | 500 | 4.7 |
| 14 | waist_pitch | 500 | 9.3 | | | | |

## Standing stability test (measured FACT)

Method (`scripts/smoke_g1.py`):
1. Reset to keyframe `stand`: pelvis (0, 0, 0.79), identity quaternion,
   all joints 0 except `shoulder_pitch=0.2` both arms, `shoulder_roll=±0.2`,
   `elbow=1.28` both arms (arms bent, forearms forward — documented from the
   keyframe itself, not hand-crafted).
2. Robot placed on ground by the keyframe pose; feet land on the scene floor
   with 4 contact spheres per foot (8 contacts).
3. Position actuators hold the keyframe joint targets: `ctrl = q0_joints`
   (clamped to ctrlrange — no clamping was active), 2500 steps × 0.002 s.
4. Sample pelvis height (`qpos[2]`) and torso (`torso_link`) up-axis tilt
   (angle between body +z and world +z).

| t [s] | pelvis z [m] | Δz [cm] | torso tilt [deg] | pelvis xy [m] | foot contacts |
|---|---|---|---|---|---|
| 0.0 | 0.7900 | 0.00 | 0.000 | (0, 0) | 8 |
| 1.0 | 0.7916 | +0.16 | 0.162 | (−1.2e−4, 6e−7) | 8 |
| 2.5 | 0.7916 | +0.16 | 0.153 | (−5.0e−5, 6e−7) | 8 |
| 5.0 | 0.7916 | +0.16 | 0.154 | (−6.4e−5, 6e−7) | 8 |

**Verdict: YES, stable.** Max |Δz| = 0.16 cm (settles upward in the first
second as the servo preload takes the joint friction/gravity sag out, then
static), max torso tilt 0.172°, horizontal drift 0.1 mm, no foot contact
losses, no limit hits. This is a servo hold, not balance — but it establishes
the model is well-posed for Phase-2 teacher controllers: gains, ranges,
contacts, and keyframe are all usable as shipped.

Renders (final sim state, t = 5 s): `g1_front.png` (azimuth 0°, robot faces
+x, verified: 2.4× wider silhouette than side view), `g1_side.png`
(azimuth 90°), both 960×720.

## Model edits vs upstream menagerie

Exactly one functional line, in `robots/g1/scene.xml` (our copy):

- `<global offwidth="960" offheight="720">` added — upstream caps the
  offscreen framebuffer at 640×480, which hard-errors on larger
  `mujoco.Renderer` sizes. Needed for 960×720 report renders.

Everything else byte-identical (meshes, inertials, actuators, keyframe,
sensors: 2× gyro + 2× accelerometer on `imu_in_torso`/`imu_in_pelvis`).
`robots/g1/LICENSE` (BSD-3-Clause) kept for redistribution.

## Files

- `robots/g1/{g1.xml, scene.xml, assets/*.STL, LICENSE}` — vendored model
- `scripts/smoke_g1.py` — spec dump + 5 s stand hold + renders; exit 0,
  self-checks assert stability thresholds (|Δz|<3 cm, tilt<10°, xy<10 cm)
- `reports/2026-10-07/g1_front.png`, `g1_side.png`
- Menagerie facts + G1 qpos layout contract appended to `notes.md`
