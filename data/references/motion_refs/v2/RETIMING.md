# RETIMING.md — what changed from v1 to v2, and what it bought

Dataset: `data/references/motion_refs/v2/` (format = v1's, plus `retime`
blocks in every take's meta and two new measured validity levels,
`balance_verified` / `balance_blocked`, defined below).
Builder: `scripts/build_v2_refs.py` (generator: `src/solo/refgen.py`).
Probes: `scripts/probe_v2_dynamic.py` -> `v2/feasibility.json` (per track AND
per drill phase; probe command + balance layer + provenance recorded there).
Everything numeric here is measured on this host and recomputable from those
artifacts; the reference stays a kinematic TARGET (no `expert_action`).

## The mechanism (why v1 hit the 0.36 s wall)

The v1 tracks were video retargets whose **root path is quasi-statically
inconsistent**: their CoM sits OUTSIDE the planted-foot hull on 62–100 % of
frames (measured: LOWER_TO_STANCE 97 % of frames, min −0.122 m; CIRCLE 100 %,
min −0.63 m; SHUFFLE_F 59 %), with root xy accelerations up to 16 m/s².  A
fixed-clock tracker commanded to follow such a path must choose between
matching joints and staying upright — the measured failure: feet planted,
joints tracked at 0.04 rad, root drifted 0.156 m fore-aft in 0.36 s,
termination at 0.36 s in every training arm.  On top of that, the v1
`solo.stance` table solver is only grounded at the stand height: the drill's
synthetic STANCE_HOLD posture (0.34 m x 0.74 m) had its soles **3.8 cm through
the mat** (FK; 8.7 cm at 0.40 m commanded) and CoM margin −0.082 m.

v2 therefore re-generates the paths, not just the clock: solved GROUNDED
stances (planar leg IK calibrated to the model's joint pivots, round-trip
<= 1 mm, FK-verified), CoM-inside-support quasi-static descents, and
weight-transfer-before-lift stepping with reachable foot placement.  Timing
then follows the measured envelope (see the sweep below).

## The measured achievable envelope (not assumed)

* Static: every solved stance in 0.50–0.79 m pelvis x 0.24–0.46 m width
  (x split 0/0.30) is grounded (0.0 cm penetration) with CoM margin
  +0.030..+0.082 m.  Deepest FLAT-SOLE symmetric crouch: **0.58 m pelvis**
  (the ankle-pitch limit −0.8727 rad is the binding constraint, measured).
  Split stances (rear foot back) hold deeper with margin to spare.
* Dynamic (probe sweep, `feasibility.json:envelope`): the v2 LOWER descent
  completes under RAW position servos AND under the balance layer at
  1.0x (2.48 s), 1.5x (3.72 s) and 2.0x (4.96 s) time scales — the v1 take
  toppled at 1.00 s under the same raw-servo protocol.  The binding
  constraint on speed was the PATH, not the clock; v2 ships 1.0x.

## Per-segment before/after (measured)

| segment | v1 (before) | v2 (after) |
|---|---|---|
| STAND | keyframe, margin +0.053; drill replay died 2.70 s | balance_verified (layer completes; margin +0.053) |
| LOWER_TO_STANCE | CoM outside support 97 % of frames (min −0.122 m), root xy accel 12.6 m/s²; replay toppled at 1.00 s; **training wall at 0.36 s in all 6 arms** | solved 2.0 s min-jerk descent 0.79→0.74 m, CoM margin >= +0.070 m every frame, az <= 1.2 m/s²; **replay PASS + balance layer COMPLETED**; envelope sweep 1.0–2.0x all PASS |
| STANCE_HOLD | fused repair posture SOLE PENETRATION −3.8 cm, margin −0.082 m; replay toppled 2.18 s | re-solved grounded (0.0 cm) + guard (A6): margin **+0.070 m**; replay PASS; balance COMPLETED |
| stance_rise | margin −0.088 m; replay toppled 1.28 s | solved rise: margin **+0.081 m**; replay PASS; balance COMPLETED |
| LEVEL_CHANGE | video drop to pelvis 0.40 m (below the G1's flat-sole reach), CoM outside 95 % (min −0.272 m); replay toppled 1.76 s | knee-driven to 0.58 m (the measured flat-sole floor) + hold + rise; CoM >= +0.047 m; **replay PASS** (both takes); T1 layer still fails it -> balance_blocked (the deep-crouch capability is T3's learning question) |
| SHUFFLE_F / SHUFFLE_B / CIRCLE / REPOSITION / widen | video stalk/circle paths: pelvis 0.56–0.72 m bouncing (root z accel 31.8 m/s²), CoM outside up to 100 % (min −0.63 m); replays toppled 1.04–2.24 s | generated weight-transfer stepping at 0.70 m: transfer (0.9 s) BEFORE each lift (rubric B2), reachable foot placement, joint vel <= 3.8 rad/s, root xy accel <= 1.1 m/s², double-support margins >= 0; raw servos still fall (no gait layer exists) and the T1 standing layer is balance_blocked — footwork is the T2/T3 ladder's question |
| DOUBLE_LEG_ENTRY_CROUCH | video crouch: pelvis 0.49 m, torso pitch 60–70 deg, margin −0.297 m; toppled in 1.18–1.24 s under its OWN targets; certified stance controller 1.38 s; best CEM (41,840 rollouts) 2.34 s | REPAIRED along the operator's two axes (width 0.40 m + rear foot back 0.35 m), knee-driven to 0.62 m, 0.5 s load instead of v1's 4.7 s hold; CoM >= +0.055 m; **replay PASS** (the take `shot_entry_full` is dynamically_verified); T1 layer blocked |
| DOUBLE_LEG_PENETRATION | GrappleMap-fused geometry (pelvis 0.333 m, knee 0.064 m) | SAME geometry (identity preserved, never flattened), time-scaled x1.5; balance_blocked — with the three independent v1 measurements (1.24 s topple, CEM 2.34 s, training wall) standing as the dynamic-infeasibility evidence for the knee-down drive |
| RECOVER_TO_STANCE | video recovery: margin −0.421 m; replay toppled 0.70 s | solved quasi-static rise from the repaired entry: margin +0.055 m; **replay PASS** (`shot_recover` dynamically_verified) |
| knee_sprawl_* , stalk_shuffle | ground-posture context takes | TIMING-ONLY v2 (1.5x slower), geometry and v1 labels preserved verbatim; excluded from every traversable claim |

Validity labels (extended enum, defined here):
* `dynamically_verified` — raw 50 Hz position servos on the reference's own
  targets complete it upright (v1's level 3, unchanged protocol).
* `balance_verified` — the balance layer completes the segment in the
  reference-conditioned tracking env.
* `balance_blocked` — L1/L2 hold and the path is CoM-consistent, but the only
  probed balance layer (T1 v6d) fails it: a LEARNING question (the T2/T3
  ladder has not trained stepping/deep crouches), **not** a reference-timing
  question.  This is the falsifier distinction: re-timing again would be wrong.
* `known_infeasible` — measured failure with cause; kept, not hidden.

## Balance layer used in the probes

`checkpoints/solo/t1_balance_v6d_401408.pt` (the best T1 candidate, fall
0.083), first-layer surgery into the reference-conditioned net (trunk + base
obs transfer; appended reference columns zero-initialised): the net outputs
residual corrections z over the reference's next-frame joint targets
(`ctrl = ref_next + 0.5*tanh(z)`), i.e. a posture/balance corrector that has
never seen the reference block.  Caveat recorded in feasibility.json.

## What the decisive acceptance now reads

With v2 (the loader's resolution DEFAULT — `solo.imitation.REFERENCE_DIRS`
searches `v2` first), the unchanged interface
(`src/solo/track.py` + `scripts/solo_track_train.py --stage
S1_stand_lower_hold_rise`, fixed clock) consumes v2.  Open-loop replay of the
LOWER segments COMPLETES (v1: dead at 0.36 s), and the balance layer
completes STAND -> LOWER -> STANCE_HOLD.  **If fixed-clock training still
failed at 0.36 s on v2, the blocker would be the RL/balance layer, not the
reference timing** — and the probes already show where that layer ends:
stepping and deep crouches are balance_blocked, which names the next lever
(T2 locomotion / T3 stance footwork in `docs/MOTOR_CURRICULUM.md`), not
another re-timing pass.

## Reproduce

```
MUJOCO_GL=egl .venv/bin/python scripts/build_v2_refs.py          # rebuild v2
MUJOCO_GL=egl .venv/bin/python scripts/probe_v2_dynamic.py       # probes + labels + bundles
MUJOCO_GL=egl .venv/bin/python scripts/probe_v2_dynamic.py --render-only
MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_motion_refs_v2.py -q
```
