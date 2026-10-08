# Phase 3A — Stabilized teacher controller (7 techniques)

**Agent:** TeacherRetry · **Date:** 2026-10-08
**Deliverables:** `src/teacher/` (controller, stabilizers, gains, phases, trims,
episode), `scripts/run_teacher.py`, `scripts/tune_teacher.py`,
`data/teacher_trims.json`, `data/teacher_stats.json`, `videos/teacher/<TECH>.mp4`,
`tests/test_teacher.py`, this report.

> STATUS: DRAFT — filled in from measurements as they land (final numbers in
> the acceptance table below).

## 0. Scope and contract

The teacher is a *feedback controller* executed by the 29 position actuators per
robot (58 total). It never touches the floating base, never applies forces
directly, never writes `qpos`/`qvel`, and reads only its own robot's state plus
its own reference. Per 50 Hz tick:

```
ctrl = clip( q_ref(t) + trim_scale(z_ref) * trim + dq_direct + dq_ik , ctrlrange )
```

## 1. What was kept from the interrupted run vs rewritten

| file | status |
|---|---|
| `src/teacher/phases.py` | **kept** (labels, segments); fixed `_smooth()` (see §6) |
| `src/teacher/gains.py` | kept structure, gains re-tuned by measurement; removed the dead base-servo fields |
| `src/teacher/episode.py` | **kept** + added `traj` capture, `score_episode`, `scored_trajectories` |
| `src/teacher/stabilizers.py` | **rewritten**: measured AUTH table, `foot_rows` flattening fix, yaw channel; removed the foot-bias path |
| `src/teacher/controller.py` | **rewritten** routing + trim + integrator + PRONE interlock |
| `src/teacher/trims.py` | **new**: data-driven reference-pose trims (candidate 5) |
| `scripts/tune_teacher.py` | **new**: trim search with the physics as oracle |
| `scripts/run_teacher.py` | **new** |

## 2. Channel identification (measured)

Probe: from the `both_stand` keyframe, apply a symmetric joint offset to both
legs, step 0.25 s, measure the base displacement (probe
`/tmp/auth.py` → numbers in §2.1).

| channel | offset | measured base motion | authority (m/rad) |
|---|---|---|---|
| ankle_pitch | +0.10 / +0.20 rad | −0.053 / −0.054 m | −0.53 → saturates |
| knee | +0.10 / +0.20 | −0.038 / −0.041 | −0.38 |
| hip_pitch | +0.10 / +0.20 | −0.009 / −0.016 | −0.09 |
| hip_roll | +0.10 / +0.20 | −0.046 / −0.061 (y) | −0.46 |
| ankle_roll | +0.10 / +0.20 | −0.002/−0.009 (y) | ≈ −0.03 (useless) |
| hip_yaw (both legs) | ±0.10 / ±0.20 | ∓0.035 / ∓0.089 yaw, **sustained** 1.5 s | −0.35 (static map) |

Consequences used in the design:
* the direct channels saturate (ankle reaches 5.4 cm and stops: the CoP hits the
  foot edge), so they stabilise, they cannot re-posture;
* a **foot-target world bias** was implemented as the "hip strategy" (push the
  planted foot backwards to push the body forwards) and measured at ~0.05 m of
  base motion per 0.1 m of bias — *no usable authority* (the servo/contact
  equilibrium absorbs it). It was removed; see §6.
* `hip_yaw` has a real *static* yaw authority, but running it in the closed loop
  measured **worse** (STANCE stay-up 0.63 → 0.43, scorer 0.63 → 0.44 at k=1.0),
  so the shipped gain table leaves `k_yaw = 0` and the channel is ablation-only.

## 3. The STANCE problem, measured

* The retargeted STANCE crouch holds the ankle actuators at ~36 Nm of their
  ±50 Nm limit just to stand still, its CoM sits behind the effective support
  (the reference keeps the *heel sites* 6-7 cm up), and it topples in 0.8-1.5 s
  under any of the stabilizer configurations tried (stay-up 0.64-0.76).
* Flattening both sole sites onto the mat (`foot_rows`) is still kept (it gives
  the stance a real support polygon), but it cannot fix the CoM placement.
* Therefore the measured reference trim (§4) is used.

## 4. Reference-pose trims (candidate 5)

(measured by `scripts/tune_teacher.py`; bound by the stay-up gate first, then the
scorer gate; every value is 4 sagittal parameters, radians)

| technique | robot | hip | knee | ankle | waist | stay_up (3 seeds) | scorer |
|---|---|---|---|---|---|---|---|
| STANCE | a_ | +0.10 | −0.20 | −0.10 | +0.30 | 0.79 (0.75 worst) | 0.569 |
| STANCE | b_ | +0.10 | 0 | +0.20 | −0.10 | (same run) | (same run) |

Only STANCE needed a trim: the search was run for all seven techniques, and for
the dynamic ones it never found a candidate that moved the stay-up gate (the
recorded search log: DOUBLE_LEG stay 0.37-0.70 / scorer 0.33-0.60 across the
coordinate walk), and no per-technique value survived re-measurement at 3 seeds
— so the shipped table has STANCE only, and `dz` (settled-height correction) was
re-measured and dropped (dz=0: stay_up 0.792 / scorer 0.569 vs the stale search
value 0.764 / 0.513).

## 5. Acceptance table

`scripts/run_teacher.py --seeds 5 --videos` in `robots/wrestling_scene.xml`
(50 Hz control, 2 ms physics; seeded initial perturbation per seed: joint noise
sigma 0.01 rad, base xy/z sigma 5/4 mm, joint velocity 0.05 rad/s).
`stay_up` = mean over seeds of the per-seed **min** of the two robots (the gate
value).  Rows marked *(1 seed)* are the seed-0 ablation numbers measured with
the identical metric machinery while the 5-seed video sweep was still running;
they are honest placeholders for the remaining techniques — the sweep was still
rendering (`videos/teacher/*.mp4` at 2.1 s/frame on this 4-core ARM host) when
this report was written, and its incremental output is in
`data/teacher_stats.json`.

| technique | stay_up (gate 0.80) | a | b | worst seed | inter pen | self pen | landmark | scorer (gate 0.85) | min phase | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| DOUBLE_LEG | 0.520 (5 seeds) | 0.59 | 0.56 | 0.45 | 2.7 cm | 1.0 cm | 2.31 m | 0.324 | 0.15 | FAIL |
| SINGLE_LEG | 0.267 (5 seeds) | 0.31 | 0.27 | 0.23 | 4.3 cm | 2.5 cm | 1.50 m | 0.361 | 0.30 | FAIL |
| BODY_LOCK | 0.200 (5 seeds) | 0.20 | 0.27 | 0.17 | 5.9 cm | 4.4 cm | 1.42 m | 0.679 | 0.42 | FAIL (best achievable: baseline 0.229) |
| SNAPDOWN | **0.956 (5 seeds)** | 0.98 | 0.96 | 0.89 | 5.7 cm | 0.1 cm | 0.43 m | 0.808 | 0.72 | **stay-up PASS**; scorer/min-phase miss the gate |
| SPRAWL | 0.316 (5 seeds) | 0.36 | 0.32 | 0.27 | 1.7 cm | 1.0 cm | 2.54 m | 0.559 | 0.28 | defender prone by design; scorer below gate |
| STAND_UP | 0.000 (5 seeds) | 0.00 | 0.00 | 0.00 | 1.7 cm | 0.1 cm | 1.27 m | 0.324 | 0.00 | FAIL — reference is airborne (§11) |
| STANCE | 0.760 (5 seeds) | 0.86 | 0.83 | 0.62 | 0.0 cm | 0.9 cm | 1.21 m | 0.547 | 0.55 | stay-up below gate; scorer fails |

Ground penetration is ≤ 0.8 cm everywhere (clean).

**Acceptance verdict: FAIL** as a whole.  The stay-up gate is met by
**SNAPDOWN (0.956, 5 seeds)** and approached by STANCE (0.79); the similarity
gate is not met by any technique.  The per-technique causes are measured and listed in
§10-§12; the dominant one is that these schematic references require postures
(or an airborne base) that are not statically viable for the G1, and a
joint-position teacher can stabilise ±5 cm but cannot step.

## 6. Defects found and fixed during this run

1. **Heel flattening was inert**: `foot_rows` gave 3-D Jacobian rows only to
   sites that touch in the reference; the retargeted STANCE touches with the toe
   only, so the heel was never driven down and the foot stayed toe-down with a
   one-point support. Both sole sites of a planted foot now get full 3-D rows.
2. **`_smooth()` was not zero-phase** (centred `valid` convolution without
   padding ⇒ ~(n−1)/2 samples of lag, 0.21 s at 50 Hz); replaced by a
   symmetrically edge-padded convolution, with a step test
   (`test_phase_smoother_is_zero_phase`).
3. **PRONE inferred from reference height alone** could switch the stabilizer off
   while the simulated robot is still standing; the controller now requires the
   simulated pelvis < 0.55 m to confirm PRONE, otherwise it treats the frame as
   LOW and keeps stabilising. The label source is the *reference* (intent), not
   the sim: contact-force signals are not needed because the reference is the
   plan the teacher is executing (documented in §7).
4. The **base-position (foot-bias) servo** was removed after measurement (§2).
5. The **yaw channel** is shipped disabled after measurement (§2).

## 7. Design rationale (short)

* **Why not root actuation / privileged state**: the contract is the 29 position
  servos per robot. Verified by `test_no_state_mutation_and_no_opponent_leakage`
  (no writes to `qpos`/`qvel`/`ctrl`/`act`, no dependence on `data.ctrl`, no
  opponent coupling) and `test_only_the_29_hinges_per_robot_are_actuated`.
* **Phase labels are reference-intent**: the label of a frame is computed from
  the *reference* pelvis track (height + vertical speed).  The stabilizer is
  switched off only for PRONE — intended ground work — and the controller
  additionally requires the *simulated* pelvis to be low (< 0.55 m), so a robot
  that has not reached the ground keeps stabilising (PRONE interlock).  Ground
  contact signals are not used to *infer* intent because the reference already
  encodes the plan; they are used as a safety interlock only.  This is judged
  sufficient for this milestone: the failure mode the audit worried about
  (stabilizer off while standing) is exactly what the interlock removes.
* **Gain scheduling** is by phase kind (STAND/RISE/LOW/DIVE/PRONE): full
  balance while standing, reduced while diving, zero on the ground.

## 8. BODY_LOCK

Measured (5 seeds): stay_up 0.200 (a 0.20 / b 0.27), scorer 0.679
(CLINCH 0.85, DROPOVER 0.61, PIN 0.42), **inter-robot penetration 5.9 cm** and
self-penetration 4.4 cm — both larger than the phase-2 reference values
(4.4 / 4.4 cm): the clinch geometry plus the teacher's balance offsets press the
torsos together.

The derived-scene remediation (stiffened robot-geom contacts,
`robots/wrestling_scene_soft.xml`, solref `(0.010, 1.0)` and solimp
`(0.95, 0.99, 0.001, 0.5, 2.0)` on every robot geom, documented in
`scripts/run_teacher.py`) is wired into `run_teacher.py` and triggers
automatically when `pen_inter > 2 cm`; the BODY_LOCK row above was still
in flight when this report was written, so the soft-scene numbers land in
`data/teacher_stats.json` (`techniques.BODY_LOCK_soft`,
`body_lock_remediation`).  **Measured remediation result (5 seeds): the derived scene
did NOT reach the < 3 cm target** — inter-robot penetration stayed at 3.4-5.9 cm
and self-penetration at 3.9-4.4 cm (stiffening `solref` to a 0.010 s time
constant and `solimp` dmax to 0.99 changed neither, because the overlap is in
the *reference geometry*, not in the contact softness), while stay-up improved
slightly (0.200 → 0.226) and the scorer stayed 0.67.  Best achievable for
BODY_LOCK: stay-up 0.229 with the phase-2 baseline (pure PD) vs 0.226 with the
teacher on the derived scene — the stabiliser
does not help this technique, and the blocker is the schematic clinch geometry
(the phase-2 report's known-failing reference), not the controller.

## 9. Iteration log (hypothesis → measurement)

Every entry is a measured prediction, in order; "stay_up" is the min over the
two robots unless noted, scorer = technique-validity scorer mean.

| # | hypothesis | measurement | verdict |
|---|---|---|---|
| 1 | the phase-2 gap is a missing balance controller; CoM-over-support offsets on ankle/knee/hip (measured authority) close it | STANCE baseline stay_up 0.21/0.57 → 0.75/0.61; DOUBLE_LEG 0.45-0.57 | partial: helps, not enough |
| 2 | flattening both sole sites onto the mat gives the stance a real support polygon | found the heel was never actuated (only XY rows for non-touching sites); after the fix the feet level (heel z 0.06 → 0.006) but the robots still topple | fix kept, not sufficient: the CoM placement, not the foot pitch, is binding |
| 3 | a world-frame foot-target bias ("hip strategy": push the planted foot back to push the body forward) shifts the body ~1:1 | measured **0.05 m of base motion per 0.10 m of bias** (0.10-0.20 m range, stand + STANCE) | refuted: no usable authority (servo+contact equilibrium absorbs it); the whole base-servo design was removed |
| 4 | raising the balance clamp/gain holds the untrimmed STANCE | 6-point sweep (k_pitch 1.0-2.0, com_clamp 0.08-0.25, ki 0.6-2.0): best stay_up 0.76, scorer ≤ 0.54 | refuted: the pose needs 36 Nm of ankle torque (of ±50) to stand still, so the feedback saturates |
| 5 | a small constant reference trim (candidate 5) makes STANCE holdable | 4-parameter grid (hip/knee/ankle/waist, 0.1 rad) per robot, physics as oracle: stay_up 0.99 (a) / 0.85 (b), scorer 0.60 | accepted for stay-up, **trades the similarity gate** (see §10) |
| 6 | hip_yaw heading feedback fixes the drift of the scored relations (rel_yaw band ±0.16 rad) | channel authority is real (−0.35 rad of base yaw per rad, sustained 1.5 s) but in the closed loop STANCE stay-up 0.63 → 0.43 and scorer 0.63 → 0.44 | refuted in the loop: shipped `k_yaw = 0`; the yaw drift is a symptom of the balance drift, not a separate axis |
| 7 | the defenders in the shots fall because the shooter's dive knocks them down | moved the shooter 3 m away (no contact possible): DOUBLE_LEG's defender still falls at t ≈ 2.0 s, SINGLE_LEG's at t ≈ 1.0 s | refuted: their *standing references* are marginally stable, same root cause as STANCE |
| 8 | STAND_UP's 0.00 stay-up is a teacher failure | measured the reference itself: min foot-site z = 0.29-0.72 m and pelvis 1.0-1.08 m for t > 5 s — the reference is **airborne**; a per-frame vertical re-grounding does not repair it (junction alignment rms 0.5-0.72 m, torso tilt ~105°) | refuted: infeasible phase-2 reference (see §11) |

## 10. Why STANCE cannot meet both gates at once

The scorer's STANCE bands are tight around the reference relations (rel_yaw
±0.16 rad, sep ±0.10 m, pitch ±0.15 rad).  Two measured facts collide:

* without a trim the robots topple (stay_up 0.64) — the retargeted crouch needs
  36/50 Nm of ankle torque and its CoM sits behind the effective support;
* the trim that holds them (waist +0.3 rad for robot A) changes exactly the
  features the scorer measures (pitch_s, rel_yaw, sep drift out of band within
  0.3 s), so the scorer stays at 0.60 with stay_up 0.99.

The trim search minimises the compromise (gate-aware fitness, §4), and the
measured frontier is: no-trim (0.64 stay / 0.48 scorer) … tuned trim
(0.99 stay / 0.60 scorer) with the shipped table at (0.76 stay / 0.51 scorer)
over 3 seeds.  Reaching ≥0.85 while holding the pose needs a posture that is
both statically stable *and* relation-preserving — for this schematic crouch
that means foot re-placement (a step), which a joint-position teacher cannot
do without a stepping controller (out of this task's scope).

## 11. Reference defects found (phase-2 hand-off)

1. **STAND_UP is airborne.**  In `data/refs/STAND_UP.npz`, from t ≈ 4 s on the
   *lowest foot site* is 0.13-0.72 m above the mat and the pelvis is 1.0-1.08 m
   (the G1 stands at 0.79 m): the montage alignment put the pair in the air.
   The teacher cannot fly, so it keeps both robots on the mat (pelvis
   0.06-0.12 m) and `stay_up = 0.00` on every eligible frame.  A per-frame
   vertical re-grounding (rigid translation, relationships preserved) was tried
   and does **not** repair it — the second half's geometry is also wrong
   (torso tilt ≈ 105°, junction alignment rms 0.5-0.72 m).  STAND_UP needs a
   phase-2 rebuild (or a fresh GrappleMap chain), not a teacher fix.
2. **STANCE crouch** — see §3: holdable only with a measured pose trim, and the
   trim trades the similarity gate (§10).
3. **Schematic penetrations**: the phase-2 report already documents the clinch
   inter-robot penetration; the teacher's rollouts show 2.7 cm (DOUBLE_LEG),
   4.3 cm (SINGLE_LEG) — the same schematic-clinch geometry, not a teacher
   artifact.

## 12. Honest failures / limitations

* **Gated techniques**: only SNAPDOWN (stay_up 0.89 at seed 0; 5-seed number in
  §5) is near the stay-up gate; DOUBLE_LEG, SINGLE_LEG, BODY_LOCK and STAND_UP
  are not.  The stabilizer is a *balancing* controller: it holds postures that
  are statically viable (±5 cm of CoM authority measured) and cannot execute a
  *step*, which is what these references need when the schematic pose is not
  statically viable for the G1.
* **Scorer gate** (mean ≥ 0.85, no phase < 0.75) is met only by SNAPDOWN at
  seed 0 (0.821, still below).  When a robot falls, the relational features
  diverge; when it is held by a trim, the trim itself moves the features.
* **The stabilizer does not help SINGLE_LEG and slightly hurts BODY_LOCK**
  (stay_up 0.20 vs 0.23 baseline at 2 seeds).  The ablation table in §13 shows
  the split: big wins on the holds (STANCE +0.71), small wins on the shots,
  nothing on the airborne STAND_UP.
* **Videos are 320x240** (software EGL on this 4-core ARM host renders at
  2-5 s/frame); they show both robots at 30 fps and are directly comparable
  with `videos/refs/*.mp4` in content, not in resolution.

## 13. Ablation: teacher vs phase-2 baseline (seed 0, same metric machinery)

| technique | baseline stay_up | teacher stay_up | baseline scorer | teacher scorer |
|---|---|---|---|---|
| DOUBLE_LEG | 0.444 | 0.593 | 0.349 | 0.334 |
| SINGLE_LEG | 0.200 | 0.257 | 0.469 | 0.345 |
| BODY_LOCK | 0.229 | 0.171 | 0.661 | 0.673 |
| SNAPDOWN | 0.778 | 0.889 | 0.865 | 0.821 |
| SPRAWL | 0.105 | 0.347 | 0.577 | 0.567 |
| STAND_UP | 0.000 | 0.000 | 0.406 | 0.298 |
| STANCE | 0.208 | 0.917 | 0.398 | 0.594 |

Per-technique gain overrides were tried (no-govern / k_pitch 0.5 / both, 2
seeds) and none survived: BODY_LOCK 0.20 → 0.21 (stay) with the scorer
0.69 → 0.66, SINGLE_LEG 0.26 → 0.21 stay but 0.37 → 0.48 scorer, SPRAWL
0.33 → 0.37 (kp 0.5).  The shipped table therefore has **no**
`TECHNIQUE_OVERRIDES` — the variations are inside seed noise and a per-technique
gain table would be overfitting to two seeds.

## 14. Tests

`tests/test_teacher.py` (12 tests): default-config first/second tick finite and
in ctrlrange (the review-requested indexing guard), targets in ctrlrange over a
full rollout, per-seed determinism (bit-identical replay) and seed sensitivity,
monotone/complete phase segmentation, zero-phase smoother step test, PRONE
interlock, no state mutation / no `data.ctrl` dependence / no opponent coupling,
58 hinge actuators only, trims small and gated, scoring uses the executor and
the reference window, baseline PD equals the reference targets, and a STANCE
hold regression pin (mean ≥ 0.65, worst seed ≥ 0.50 over seeds 0-2; the shipped
configuration measures 0.76 mean / 0.58 worst).

Full suite on this tree: `.venv/bin/python -m pytest tests/ -q` → **115 passed**
(the task's own modules plus the concurrent agents' suites).

## 15. Stance geometry trims (operator rule, applied and measured)

Both operator-sanctioned axes were implemented and measured on STANCE
(reference frames 0-117 are the held crouch, so every trim below applies to the
whole hold; frame index = the reference hold window 0.00-2.34 s, and the very
same values are applied to the whole trajectory through `trim_scale`, which is
1.0 above a reference pelvis of 0.60 m):

| variant | axis | amount (rad) | stay_up (3 seeds) | scorer | verdict |
|---|---|---|---|---|---|
| shipped: waist +0.30 (a) / −0.10 (b), hip +0.10, knee −0.20, ankle −0.10/+0.20 | torso pitch (fallback) | 0.30 | **0.792** (worst 0.750) | **0.569** | shipped |
| rear leg back: hip_rear +0.30, knee_rear −0.15, ankle_rear +0.15 | sagittal (rule a) | 0.30 | 0.486 | 0.402 | rejected on measurement |
| rear-back + splay_rear 0.10 | sagittal + frontal (a+b) | 0.30 / 0.10 | 0.458 | 0.413 | rejected |
| rear-back + waist 0.30 | a + torso | 0.30 / 0.30 | 0.472 | 0.398 | rejected |
| waist + splay_rear 0.10 | torso + frontal (b) | 0.10 | 0.569 | 0.460 | rejected |
| rear-back finer (hip_rear 0.30, no waist) | sagittal (a) | 0.30 | 0.486 | 0.402 | rejected |

Reading: the rear-foot-back and stance-widening repairs (operator rule) were
implemented first (`hip_rear`/`knee_rear`/`ankle_rear`, `splay_rear`/
`splay_lead` in `src/teacher/trims.py`) and both measured *worse* than the
torso-pitch trim that ships, at the tested magnitudes (0.30 rad rear-leg
extension with knee/ankle compensation; 0.10 rad splay).  The likely reason is
that this crouch's CoM error is not (only) a base-length problem: the support is
a one-point toe contact until the soles are flattened, so extending the rear leg
loads a foot that is not yet on the mat.  A finer magnitude sweep and a
rear-back *plus* flatten-plus-flattened-soles combination remain untested and
are the recommended next step before concluding against the operator's rule.
The CoM-margin improvement is logged for the shipped trim as the pose search's
hold result (stay_up 0.99 at 2 seeds in the search itself, 0.79 at 3
seeds/perturbed) — the margin metric itself was not separately instrumented per
variant, which is an honest gap in this iteration log.

Stance width: the teacher does **not** treat the reference width as a target it
enforces, and it does not limit it — the only width change in the shipped table
is zero (the splay variants were rejected).  Treating the operator's widths as
a *minimum* would therefore be a no-op on the current shipped configuration;
the machinery to widen (`splay_rear`/`splay_lead`) is in place if a future
technique needs it.

## 16. Driveable drill interface (operator priority) — status: interface shipped, elements NOT yet holdable

`src/teacher/skills.py` implements the requested scheduler interface:

```python
from teacher.skills import SkillController, SKILLS
ctrl = SkillController(model, stance_a, stance_b=npz_b_start)   # per-robot stances
ctrl.reset(live_qpos_a)                # start/restart the drill from a live state
ctrl.set_skill("LEVEL_CHANGE")         # STANCE_HOLD, SHUFFLE_FORWARD/BACK/LATERAL,
                                       # CIRCLE_L/R, RETREAT, APPROACH, LEVEL_CHANGE,
                                       # SHOT_DOUBLE_LEG, KNEE_LOWER, RECOVER_STAND
ctrl.set_command(vx=0.08, vy=0.0, wz=0.0)   # body frame, m/s, rad/s
ctrl.set_stance_height(-1.0)           # 0 = reference stance, -1 = fully lowered
ctrl.set_lead_step(0.7)                # lead-leg penetration step depth
joints = ctrl.control(data, t)         # (58,) every 50 Hz tick, balance layer always on
```

Design: every skill is a *procedural pose* (stance + measured trim + squat/step
patterns, `LEVEL_CHANGE` uses knees/ankles only — no waist) on an 18 s timeline;
the movement command advances the reference base xy/yaw, which the measured
balance authority chases; every skill change cross-fades over 0.6 s and the trim
ramps in over the first second, so elements concatenate with no simulation
reset.  The bystander robot gets its own trimmed stance so its fall cannot
disturb the drill.

**Honest status**: the self-check (`PYTHONPATH=src .venv/bin/python -m
teacher.skills`, scripted STANCE_HOLD → SHUFFLE_FORWARD → LEVEL_CHANGE →
KNEE_LOWER → RECOVER_STAND, one continuous sim) currently reports
STANCE_HOLD up_frac 0.85 with min pelvis 0.088 m and 0.97 m of xy drift — i.e.
**the element still topples**, and everything after it runs on the ground.  The
same stance under the reference replay holds (stay_up 0.79, §5), so the
remaining defect is in the *drill* path (constant pose table + live-start
transient), not in the balance layer; it was not found within this session.
Downstream: the drill video should NOT be produced from this interface yet —
the scheduler agent should be told the elements are not holdable until this is
fixed.  The two fixes tried here (trim ramp; per-robot bystander stance) each
moved the number by <0.05.

## 17. Files and how to run

```bash
MUJOCO_GL=egl .venv/bin/python scripts/run_teacher.py --seeds 5 --videos   # acceptance + videos
MUJOCO_GL=egl .venv/bin/python scripts/tune_teacher.py STANCE --seeds 3    # trim search
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m teacher.skills            # drill self-check
MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_teacher.py -q          # 12 tests
```

## 18. QUALITY RUBRIC assessment (docs/QUALITY_RUBRIC.md) — no ship candidate

Measured on the shipped configuration (`scripts/run_teacher.py`, seed 0, and the
rubric-A probe in this section).  Scores are deliberately conservative: the
rubric says "0 by default without evidence", and the elements below are
implemented in the drill interface but their live-start executions topple, so
most evidence is negative evidence and is kept as such.

| element | score | measured evidence |
|---|---|---|
| **A. Stance** | **1** | A1 foot lateral separation 0.270 m (a) / 0.288 m (b), A2 front-to-back 0.156 m / 0.117 m (rear foot behind the lead ✔). A3 the crouch is *partly* waist-driven: the shipped trim adds waist_pitch +0.30 rad while knees stay at 0.43/0.42 — the rubric wants knees, so this is a defect, not a pass. A5/A6 head/hand carriage preserved from the reference (scorer HOLD phase ankle_z/knee_z/lock_l predicates 0.94-0.97 at the pose level, *before* the drift). A7 CoM margin +0.100 m in the reference pose but only **+0.027 m** in the live execution (thin, above the 2 cm floor but not robust). A8 **fails**: the hold is 2.34 s (not 20 s), pelvis xy drift 0.28 m (a) in 2 s and robot b fell in seed 0 (stay_up 0.833, min pelvis 0.19 m). |
| **B. Footwork** | **0** | Not implemented: the movement commands only advance the reference base xy, which the measured authority (≤5 cm) cannot follow — no foot lift/place exists, so B1/B3 (no sliding while loaded, command fidelity) cannot pass. |
| **C. Level change** | **0-1** | Implemented as a knee/hip/ankle squat pattern (`LEVEL_CHANGE`, no waist) in `skills.py`, but the drill self-check reaches it only after STANCE_HOLD has toppled, so the measured trace is a fall (min pelvis 0.097 m). No pass. |
| **D. Penetration step** | **0-1** | `SHOT_DOUBLE_LEG` / `KNEE_LOWER` add a lead-leg hip/knee/ankle step pattern + level change; same caveat: no holdable execution, no D1-D7 measurements. |
| **E. Recovery / rise** | **0-1** | `RECOVER_STAND` blends toward the model's proven-stable standing keyframe; executed only on the ground in the self-check. |
| **H3 slow-motion clips** | **not produced** | Only the full-speed 320x240 technique videos exist (`videos/teacher/`); the 0.25x clips of level change/entry/knee contact/rise were not rendered because no element passes A/C/D/E. |

**Conclusion**: no ship candidate.  The blocker is not video quality — it is
that a live-start drill element cannot yet be held: the stabiliser holds the
*reference replay* of the stance (stay_up 0.79, 2.3 s) but not the continuous
drill pose (measured topple within ~1.6 s).  The two candidate causes are (i)
the drill table is built once while the teacher precomputes its IK/upright
references at construction, so the ramp-in of the trim is fought by the
stabiliser's own reference, and (ii) the drill starts from the raw schematic
pose, whose first second is a real transient.  Fixing (i) means rebuilding the
teacher context when the commanded pose changes materially (a scheduler-side
design decision) — this is the recommended next step, and it is the honest
reason the drill video should not be produced yet.

## 19. Hand-off to the fix agent (exact paths, semantics, commands)

**Files / functions**

| what | where |
|---|---|
| stabilised teacher | `src/teacher/controller.py` — `TeacherController.__init__` (builds `self.ctx[prefix]`), `_RobotCtx.__init__` + `_RobotCtx._precompute_ref` (the frozen reference tables: `ref_qpos`, `ref_pelvis_z`, `ref_scale`, `ref_feet`, `ref_up`, `ref_support`, `ref_touch`, `ref_yaw`, `rear_foot`), `TeacherController._control_robot` (the per-tick composition), `_q_ref` (reference interpolation + trim de-scaling) |
| balance channels | `src/teacher/stabilizers.py` — `channel_offsets` (AUTH map), `yaw_offsets`, `foot_rows` (flatten/anchor/height IK rows), `upright_rows`, `solve_offsets`, `support_center`, `foot_contact`, `heading_yaw`, `capture_error` |
| gains | `src/teacher/gains.py` — `GAIN_TABLE` per phase kind, `gain_for(technique, kind)`, `TECHNIQUE_OVERRIDES` (empty: measured noise) |
| phases | `src/teacher/phases.py` — `label_frames`, `segments`, `_smooth` (zero-phase) |
| reference trims | `src/teacher/trims.py` — `POSE_TRIM` (loaded from `data/teacher_trims.json`), `trim_vector(spec, rear=...)`, `trim_scale(z_ref)`, keys: `hip/knee/ankle/waist/ankle_roll` (both legs), `hip_rear/knee_rear/ankle_rear` (rear leg), `splay/splay_rear/splay_lead` (hip_roll widening), `dz` (settled height) |
| drill interface | `src/teacher/skills.py` — `SKILLS`, `SkillController.__init__/reset/set_skill/set_command/set_stance_height/set_lead_step/control`, `_pose()`, `_build()`, `run_self_check()` |
| evaluation | `scripts/run_teacher.py` (`run_technique`, `render_video`, `make_soft_scene`, acceptance table), `scripts/tune_teacher.py` (`evaluate`, `fitness`, `tune`) |

**Composition of one tick** (`TeacherController._control_robot`): measurements
(CoM, low-pass CoM velocity, foot contacts via sole-site height < 0.045 m,
support centre, heading yaw) -> one world-frame task error
`e = capture_error(com, v, support, kd, com_clamp)` -> task integrator
`e_int` -> `dq_direct = channel_offsets(R(-yaw) e_dir, contact, gains) [+ yaw_offsets]`
-> leg-IK rows via `foot_rows` (both sole sites of each reference-planted foot,
full 3-D rows: xy = reference site position, z = flattened rest height + dz) and
`upright_rows` -> `dq_ls = solve_offsets(rows, des, REG_LAMBDA)` -> clamp to
`max_leg/max_upp`, rate-limit vs the previous tick (`max_rate`) -> posture
governor blend -> `ctrl = clip(q_track + dq_direct + dq_ls, ctrlrange)` where
`q_track = _q_ref(ctx, t)` (the *precomputed* table, interpolated).

**Confirmed defect 1 (mutated table never executed)**: `SkillController.control`
writes `self._table[k] = self._pose()` (skills.py, in `control`) but
`TeacherController` copied its reference at construction — `_RobotCtx` stores
`self.ref_qpos = qpos_ref.copy()` and derives every table from that copy at
`_RobotCtx._precompute_ref`.  Changing `self._table` after `_build()` therefore
never reaches the executor: `_q_ref` interpolates the frozen copy, so
LEVEL_CHANGE / SHOT_DOUBLE_LEG / KNEE_LOWER / RECOVER_STAND execute no pose
change.  Fix options: (a) rebuild the `TeacherController` (and its ctx tables)
whenever the commanded pose changes materially, or (b) pass a live callable
reference (interpolation hook) into `TeacherController` so `_q_ref` and the IK
tables are recomputed per tick.  (b) is the scheduler-friendly one.

**Confirmed defect 2 (call cadence)**: `TeacherController` assumes it is called
at 50 Hz — its integrators, rate limiter and `DT_CTRL` are all per-tick at
0.02 s.  `SkillController.run_self_check` calls `control()` **every 2 ms
physics step**, so those ran 10x fast.  I patched `SkillController.control` to
drive its drill clock from the simulation time (`dt = clip(t - _t0 - _t_pose,
0, 0.1)`, so the pose table and blends advance correctly at any call rate), but
the teacher itself must still be invoked once per 50 Hz control tick — either by
calling it from a 50 Hz gate in the harness or by making `TeacherController`
time-aware.  The "topple within ~1.6 s" drill result was measured on the
mis-cadenced harness and must be re-measured after the fix.

**Reproduce**

```bash
MUJOCO_GL=egl .venv/bin/python scripts/run_teacher.py --seeds 5 --videos           # acceptance + videos + stats
MUJOCO_GL=egl .venv/bin/python scripts/run_teacher.py --only STANCE --seeds 5 --videos   # single technique (merges)
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m teacher.skills                    # drill self-check (see cadence note)
MUJOCO_GL=egl .venv/bin/python scripts/tune_teacher.py STANCE --seeds 3            # trim search (merges)
MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_teacher.py -q
```
