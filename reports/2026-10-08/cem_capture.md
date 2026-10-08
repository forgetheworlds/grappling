# CEM capture of the shot-entry reference — the first full-budget run

Question this run answers (operator, 2026-10-08): *can one video-derived stepping
reference be turned into a physically executable G1 demonstration, and if not,
which rung fails — retargeting, demonstration generation, obs/action design, or
imitation training?*

**Answer: the pipeline is built and runs end to end; the demonstration is
rejected, and the failure is isolated at the RETARGETING (data) rung — with two
measured root causes, one of them a defect in our own metric.**

## 1. The run

```
scripts/solo_demo_capture.py --source cem --ref data/refs_video/shot_entry_full.npz \
  --recover-ref data/refs_video/shot_recover.npz --cem-samples 200 --cem-iters 6 \
  --cem-window 50 --cem-stride 25 --seed 0 --name shot_entry_full_a_cem \
  --out data/solo/demos --check
```

* CEM search: **41,840 rollouts, 3,643.8 s (1 h)**, 38 windows, 45 search
  parameters (15 active joints x 3 knots).  The script's own dry-run estimate was
  15,833 s for the capture phase.
* Result: `landmark_rms_m 0.4881` vs `passthrough 0.4967` — the search bought
  almost nothing.
* Episode: **fell at t = 2.34 s** (131 ticks of a 935-tick plan).
* Checker: 7 of 9 criteria failed — `no_fall, contact_sequence, torso_pitch,
  com_margin, no_saturation, travel, terminal_stance`; `knee_depth` and
  `no_foot_slide` passed.
* Artifacts: `data/solo/demos/shot_entry_full_a_cem/ep00_seed0/` —
  `trace.npz`, `reference.npz` (the demo in `solo.bc.load_reference` format, so
  the refinement leg can consume it directly), `spec.npz`, `theta.npz`,
  `check.json`, `meta.json`, `index.json`.

## 2. One "failure" was our own metric bug (fixed)

`_Ids.pitch_deg` read `qpos[base_qadr : base_qadr+7]`, which for a free joint is
`(px, py, pz, qw, qx, qy, qz)` — so the yaw formula consumed the **position**
and `qw` as if they were `(w, x, y, z)`.  The heading, and with it the *sign* of
the sagittal lean, therefore depended on **where the robot stood**.

Measured on the reference: a **122.6 deg pitch jump in one 20 ms tick** while
every joint moved <6 deg.  The checker's `torso_pitch` bar (max deviation ≤15 deg)
was unsatisfiable, and its reported "172 deg deviation at t=1.9 s" was this
artifact, not the demonstration.

Fixed in `da57924` (quaternion slice) with a regression test pinning the
violated property — translating and yawing the root must not change
`pitch_deg`.  With the fix the reference's pitch is smooth (max 2.37 deg/tick,
range 33.5–70.2 deg) and the demo **tracks it to 8.5 deg over its first second**
(bar 15), diverging only as it falls.

## 3. The demonstration fails in the pre-step crouch, and that crouch is not executable

The reference spends t = 0 … 3.62 s in a deep forward crouch (pelvis ≈ 0.5 m,
torso pitch 60–70 deg) before its lead-foot plant.  The demo never reaches the
plant: it falls at 2.34 s.  Three independent measurements say the crouch is
beyond this robot:

| controller | from where | outcome |
|---|---|---|
| static check (no sim) | the reference's own qpos | CoM **inside** the sole hull throughout, margin +0.017 … +0.22 m |
| position servos at the reference's **own** joint targets | the crouch pose | **topples in 1.24 s** (tilt 114.6 deg, pelvis 0.166 m) |
| our certified stance controller (`stand_hold`) | the crouch pose | **falls in 1.38 s** |
| the CEM's best 45-parameter search | the crouch pose | falls at 2.34 s |
| `stand_hold` (control) | our certified stance | holds (tilt 0.2 deg) |

So the crouch is **statically balanced but dynamically infeasible**: the posture
saturates the actuators (the demo's `no_saturation` failure) and no controller we
have — scripted or searched — holds it.  A human's deep wrestling crouch leans on
ankle/hip torque capacity that the G1 does not have at that configuration.

The crouch is a **T3 (level-change) capability**, and the curriculum ladder has
not trained T3 yet (T1 stance is still being certified).  Imitating a reference
whose first 3.6 s demand an untrained, apparently actuator-limited capability
inverts the curriculum.

## 4. The other root cause: grounding

Independently (and previously) isolated in
`reports/2026-10-08/reference_fidelity.md`: `place_solo` floors the retargeted
feet with **one constant per take** (min over 6 foot landmarks over the whole
take, `retarget_video.py:253-258`).  The emitted sole points hover up to
**+0.111 m** and penetrate **−0.067 m** versus our own stance keyframe's 8/8 sole
points at −0.0019 m (same FK).  Any support-polygon or CoM-margin statement built
on this reference is therefore off by up to 10 cm — which is the same order as
the crouch's own margins (+0.017 m late in the entry).

## 5. Where that leaves the operator's experiment

* Legs built and exercised: **capture** (this run), **checker** (9 criteria, and
  it discriminates: the good synthetic trace passes, each perturbation fires its
  own criterion), **refinement** (`scripts/solo_imitation_train.py`, PPO on a
  reference-conditioned episode, three init paths, soft-deviation mode).
* Blocked at the **data** rung, in this order:
  1. **Re-ground the reference per frame** (contact-aware, not one constant per
     take), then re-measure the fidelity verdicts.
  2. **Torque-aware retargeting**: the crouch saturates the actuators; a pose
     that is statically balanced can still be dynamically infeasible.  The
     retarget should either respect actuator limits or the reference must be
     chosen/repaired so its crouch is inside the robot's envelope.
  3. Alternatively (and per the ladder): treat the crouch as the **T3 skill** it
     is — train level change first, then return to this reference with a policy
     that already owns the posture.
* Only after (1)–(3) does it make sense to spend another capture budget
  (1 h for this configuration) or to run the refinement stage.

## 6. Defects fixed on the way

| commit | defect |
|---|---|
| `da57924` | `pitch_deg` read the free joint's position as its quaternion (this report §2) |
| `5a0c75f` | the capture wrote no `reference.npz`, so the demo could not be consumed as a refinement target (found by contract-checking before the 1 h run) |
| `0b1bbef` | no step-stamped snapshots; monitor rows keyed by step count alone (collided across runs) |
| `3b0fcfa` | `movement_lit` (the T1-movement set; `balance_lit` cannot reach the T2 gate's yaw criterion) |
