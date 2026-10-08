# Reference fidelity — do the 12 video-derived G1 tracks drive the video's behaviour?

**Status: completed from the killed agent's instrument.** Agent `RefFidelity` was stopped by the
provider outage (opencode-go weekly quota) before writing its report; the measurement script it had
built is complete, self-verifies, and this report is written from its measured output. Nothing here
is quoted from the agent's prose — every number is reproduced from the script run below.

**Instrument:** `scripts/measure_reference_fidelity.py` (re-runnable, prints its own SELF-VERIFY).
**Numbers:** `data/solo/metrics/reference_fidelity.json`. **Tests:** `tests/test_reference_fidelity.py` (5 passed).

```
MUJOCO_GL=egl .venv/bin/python scripts/measure_reference_fidelity.py --selftest
MUJOCO_GL=egl .venv/bin/python scripts/measure_reference_fidelity.py --json data/solo/metrics/reference_fidelity.json
MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_reference_fidelity.py -q
```

Three views of every track are compared: **video** (MediaPipe landmarks, native units, take-level
floored), **target** (placed, G1-scaled — what the solver was handed) and **g1** (emitted `qpos_a`
at 50 Hz, via FK on the retarget model). SELF-VERIFY: 8/8 OK (core site == qpos root; stance sole
touches z=0; stance CoM inside the hull; the instrumented anchor recursion reproduces
`reconstruct_translation`; native×LSQ == target geometry to 2.8e-16; emitted duration == video × stretch).

## 1. Verdict up front

* **No track is fit as a contact-exact imitation target.** The blocking defect is **grounding**: the
  emitted G1 tracks are not per-frame grounded (feet hover up to +0.11 m, or penetrate −0.067 m; §4).
* **Four tracks are usable now as style/motion priors**: `stance_hold`, `stance_widen_step`,
  `stalk_shuffle`, `circle_step` — pelvis height within 7–13 %, stance width within ~20 %, torso
  pitch within 8°, and those are exactly the T1/T1-movement behaviours on the todo.
* **Two tracks are usable after repair**: `level_change_fast` / `level_change_full` (pelvis drop and
  width survive; torso pitch is distorted by +8…+11°).
* **Six tracks should be dropped for now**: the `knee_sprawl_*` family (the crouch depth is
  distorted by 0.20–0.38 m and three of them are ground postures that cannot end in a stance by
  design) and the `shot_*` pair (joint-limit saturation in 57–81 % of frames).

## 2. Feature preservation (video | target | G1)

Values: pelvis z (m), stance width (m), `knee_min` = the closest either knee gets to the floored
plane (m), torso pitch (deg), then the time-warped G1-vs-target RMS error for pelvis z / knee z.

| track | pelvV | pelvT | pelvG | wV | wT | wG | kneeV | kneeT | kneeG | pitchV | pitchT | pitchG | rmsP | rmsK |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| circle_step | 0.871 | 0.697 | 0.766 | 0.614 | 0.498 | 0.503 | 0.455 | 0.293 | 0.283 | 51.3 | 51.3 | 59.2 | 0.066 | 0.020 |
| knee_sprawl_entry | 0.807 | 0.395 | 0.431 | 0.073 | 0.057 | 0.049 | 0.394 | 0.036 | 0.019 | 69.8 | 69.8 | 73.9 | 0.038 | 0.028 |
| knee_sprawl_entry2 | 0.750 | 0.501 | 0.546 | 0.439 | 0.384 | 0.393 | 0.398 | 0.205 | 0.200 | 56.5 | 56.5 | 58.5 | 0.044 | 0.041 |
| knee_sprawl_hold | 0.475 | 0.363 | 0.381 | 0.033 | 0.025 | 0.021 | 0.072 | 0.013 | −0.008 | 69.9 | 69.9 | 71.5 | 0.019 | 0.031 |
| knee_sprawl_recover | 0.667 | 0.358 | 0.409 | 0.242 | 0.189 | 0.163 | 0.299 | 0.098 | 0.019 | 53.2 | 53.2 | 52.9 | 0.040 | 0.065 |
| level_change_fast | 0.827 | 0.633 | 0.672 | 0.097 | 0.082 | 0.070 | 0.440 | 0.363 | 0.314 | 56.0 | 56.0 | 67.4 | 0.064 | 0.049 |
| level_change_full | 0.832 | 0.628 | 0.680 | 0.818 | 0.675 | 0.687 | 0.477 | 0.301 | 0.272 | 50.4 | 50.4 | 58.8 | 0.064 | 0.025 |
| shot_entry_full | 0.648 | 0.427 | 0.487 | 0.740 | 0.599 | 0.604 | 0.225 | 0.109 | 0.100 | 38.6 | 38.6 | 44.5 | 0.061 | 0.018 |
| shot_recover | 0.591 | 0.448 | 0.527 | 0.522 | 0.427 | 0.418 | 0.211 | 0.061 | 0.045 | 26.8 | 26.8 | 34.7 | 0.080 | 0.045 |
| stalk_shuffle | 0.857 | 0.739 | 0.800 | 0.204 | 0.166 | 0.162 | 0.442 | 0.320 | 0.307 | 46.6 | 46.6 | 52.7 | 0.060 | 0.024 |
| stance_hold | 0.822 | 0.660 | 0.713 | 0.567 | 0.456 | 0.460 | 0.443 | 0.370 | 0.349 | 35.1 | 35.1 | 37.6 | 0.062 | 0.015 |
| stance_widen_step | 0.839 | 0.676 | 0.756 | 0.565 | 0.449 | 0.450 | 0.422 | 0.355 | 0.329 | 43.0 | 43.0 | 44.0 | 0.080 | 0.023 |

Readings:

* **Torso pitch survives** everywhere (video→G1 within 2.5–11°). The video's posture leans 27–70° and
  the G1 leans the same way — the reference postures are the *operator's* wide, low, leaning stance,
  not our upright keyframe.
* **Stance width survives** at ~80–100 % of the video value after scaling (e.g. 0.567 → 0.460).
* **The video→target pelvis drop is the LSQ body scale, i.e. correct morphology, not a defect**
  (stance_hold: 0.822 × 0.805 = 0.662 ≈ the target's 0.660). The defect is the **target→G1**
  step: the emitted pelvis sits 5.1–6.9 cm above the placed target (stance_hold +5.3 cm,
  circle_step +6.9 cm, knee_sprawl_recover +5.1 cm), i.e. the vertical placement offset of §4 —
  which is also what puts the feet 2.6–11 cm off the floor.
* **`knee_min` does not survive for the knee-sprawl family**, and that is *not* the scale: video
  0.394/0.398/0.299 → G1 0.019/0.200/0.019, far beyond 0.78–0.88× (the entry/recover clips drive
  the knee 0.20–0.38 m lower than the video ever does), while `knee_sprawl_hold` agrees
  (0.072 → −0.008, i.e. the knee reaches the mat plane).

## 3. Contact / timing

`stretch` = emitted time / video time. `anchor=lower` = fraction of clearly-separated frames where
the pipeline's planted-foot choice equals the video's lower foot. `phantomS` = seconds of "phantom
plant" (the pipeline plants a foot the landmarks do not put on the floor). `kneeC_v/g` = seconds the
knee is at the mat plane, video vs G1.

| track | stretch | anchor=lower | phantomS | slipG1max (m/s) | kneeC_v | kneeC_g |
|---|---|---|---|---|---|---|
| circle_step | 1.45 | 0.70 | 0.40 | 0.204 | 0.00 | 0.00 |
| knee_sprawl_entry | 1.20 | 0.26 | 0.07 | 0.126 | 0.00 | 1.02 |
| knee_sprawl_entry2 | 2.30 | 0.33 | 0.07 | 0.197 | 0.00 | 0.00 |
| knee_sprawl_hold | 1.00 | 0.32 | 0.40 | 0.149 | 0.07 | 4.42 |
| knee_sprawl_recover | 2.31 | 0.64 | 0.13 | 0.331 | 0.00 | 2.86 |
| level_change_fast | 1.26 | 0.77 | 0.13 | 0.210 | 0.00 | 0.00 |
| level_change_full | 1.65 | 0.59 | 0.00 | 0.189 | 0.00 | 0.00 |
| shot_entry_full | 1.00 | 0.42 | 0.33 | 0.086 | 0.00 | 0.00 |
| shot_recover | 1.79 | 0.69 | 0.13 | 0.230 | 0.00 | 0.90 |
| stalk_shuffle | 1.75 | 0.68 | 0.00 | 0.183 | 0.00 | 0.00 |
| stance_hold | 1.00 | 0.35 | 0.40 | 0.041 | 0.00 | 0.00 |
| stance_widen_step | 1.29 | 0.30 | 0.67 | 0.256 | 0.00 | 0.00 |

* **The emitted tracks are time-dilated**: stretch 1.00–2.31 (only three tracks run at video speed).
  A learner that imitates them imitates the *slowed* motion.
* **The planted-foot anchor agrees with the video in only 26–77 % of clear frames** (median ≈ 0.60):
  the assumption the root reconstruction rests on is wrong for a large fraction of the motion, and
  0.0–0.67 s of it produces phantom plants.
* **The G1 shows 0.04–0.33 m/s of foot motion during runs its own detector calls contact** — the
  magnitude of the grounding problem below.

## 4. The blocking finding: the tracks are not grounded

`place_solo` floors the feet with **one constant per take** —
`floor = nanmin(t2[:, foot_idx, 1]); t2[..., 1] -= floor` over the six foot landmarks
(`data/references/yt_gBAhX5t-GW4/derived/tools/retarget_video.py:253-258`) — and the root height
comes from the human's scaled hip height. The G1's legs are not the human's, so the emitted sole
points land wherever the solver puts them:

| view | sole height (m) |
|---|---|
| **our own stance keyframe** through the same FK | **−0.0019 on all 8 sole points** (flat) |
| `stance_hold` emitted | min over take +0.0002 · median +0.0334 · **end +0.0262** |
| `circle_step` emitted | min +0.0004 · median +0.0528 · **end +0.1113** |
| `level_change_full` emitted | min +0.0024 · median +0.0348 · **end +0.0336** |
| `knee_sprawl_recover` emitted | **min −0.0674** · median +0.0189 · end −0.0137 |

The control (our keyframe → 8/8 sole points at −2 mm) proves the FK and the sole geometry are right;
the emitted tracks hover 2–11 cm or penetrate up to 6.7 cm, posture-dependently. Every
contact-dependent quantity downstream (support hull, terminal stance, GRF terms, foot slip) is
therefore measured on feet that are **not on the floor**.

## 5. Terminal posture — the predicate is validated, all 12 tracks fail it

`terminal_stance` requires: feet flat and in contact, CoM inside the hull with positive margin,
pelvis ≥ 0.55 m, torso tilt ≤ 20°, end speed ≤ 0.20 m/s. Its controls (in
`tests/test_reference_fidelity.py`):

* **positive** — our own stance keyframe: `ok=True` (flat, margin +0.0533 m, tilt 0.69°);
* **negative** — the same keyframe pitched 80°: `ok=False` (tilt 79.3°, feet not flat, no hull).

| track | ok | pelv_z | Δstance | feet | comMarg | tilt | spd | limFrac |
|---|---|---|---|---|---|---|---|---|
| circle_step | False | 0.825 | +0.019 | False | nan | 43.8 | 0.23 | 0.589 |
| knee_sprawl_entry | False | 0.431 | −0.376 | False | −0.573 | 73.4 | 0.01 | 0.154 |
| knee_sprawl_entry2 | False | 0.601 | −0.205 | False | nan | 55.3 | 0.31 | 0.444 |
| knee_sprawl_hold | False | 0.402 | −0.405 | False | nan | 64.4 | 0.12 | 0.385 |
| knee_sprawl_recover | False | 0.521 | −0.285 | False | −0.366 | 94.0 | 0.27 | 0.669 |
| level_change_fast | False | 0.719 | −0.088 | False | −0.066 | 64.0 | 0.19 | 0.382 |
| level_change_full | False | 0.710 | −0.097 | False | −0.376 | 39.3 | 0.42 | 0.314 |
| shot_entry_full | False | 0.431 | −0.375 | False | nan | 37.4 | 0.12 | 0.810 |
| shot_recover | False | 0.559 | −0.247 | False | −0.081 | 50.1 | 0.20 | 0.571 |
| stalk_shuffle | False | 0.732 | −0.075 | False | nan | 53.0 | 0.00 | 0.533 |
| stance_hold | False | 0.747 | −0.060 | False | −0.070 | 35.6 | 0.18 | 0.371 |
| stance_widen_step | False | 0.718 | −0.089 | False | nan | 53.3 | 0.03 | 0.624 |

Two distinct reasons, and they must not be conflated:

1. **Grounding** — "feet not flat / not in contact" is a consequence of §4, not of the posture.
2. **Definition mismatch** — the video's stance *is* a 35–70° forward-leaning wide crouch (see the
   stills in `docs/references/stills/`), while the predicate's 20° tilt bar encodes *our* upright
   keyframe. The references faithfully reproduce the operator's lean; they fail *our* stance
   definition. The `knee_sprawl_*` tracks end on the mat **by design** (the technique is a sprawl);
   `knee_sprawl_recover` is the one that must end standing, and it ends at 94° tilt.

## 6. Scale / morphology / solver cost

| track | LSQ scale | limFrac | joints at a limit |
|---|---|---|---|
| knee_sprawl_entry2 | 0.875 | 0.444 | 7 |
| knee_sprawl_recover | 0.781 | 0.669 | 6 |
| shot_entry_full | 0.809 | **0.810** | 5 |
| shot_recover | 0.819 | 0.571 | 3 |
| stance_widen_step | 0.796 | 0.624 | 3 |
| … (all 12 in the JSON) | 0.766–0.875 | 0.154–0.810 | 1–7 |

`limFrac` = fraction of frames with ≥1 joint at a position limit. The G1 is driven into its limits
for most of `shot_entry_full` (81 %) and two-thirds of `knee_sprawl_recover` — the references ask
for joint configurations the robot does not have.

## 7. Which of the 12 the plan should train on

| verdict | tracks | measured reason |
|---|---|---|
| **Use now (style/motion prior)** | `stance_hold`, `stance_widen_step`, `stalk_shuffle`, `circle_step` | pelvis within 7–13 %, width within ~20 %, pitch within 8°; these are the T1/T1-movement behaviours |
| **Use after repair** | `level_change_fast`, `level_change_full` | pelvis drop and width survive; pitch distorted +8…+11°; limFrac 0.31–0.38 |
| **Drop for now** | `knee_sprawl_entry`, `knee_sprawl_entry2`, `knee_sprawl_hold`, `knee_sprawl_recover` | crouch depth distorted 0.20–0.38 m; ends on the mat (by design for 3, by failure for `recover`: 94° tilt) |
| **Drop for now** | `shot_entry_full`, `shot_recover` | joint-limit saturation 0.57–0.81; knee depth 0.225→0.100 / 0.211→0.045 |

## 8. The single most likely way the references mis-drive behaviour

**The take-level floor constant plus hip-height-scaled root placement** (§4): the G1 is trained to
stand 3 cm too high and to crouch 5–11 cm too high (or 6.7 cm too low), posture-dependently, and
every contact-dependent term is measured on feet that are not on the floor. A policy imitating these
tracks learns the wrong absolute heights; a gate measuring support-hull margin, foot slip or terminal
stance on them measures the offset, not the skill.

**Required before any imitation run:** per-frame grounding (re-solve the root height from actual
per-frame foot contact, not one take-level constant) and a re-check of the LSQ-scale-about-the-hips
compression for the crouch phases.

**Second candidate, if grounding is repaired and the behaviour still mis-drives:** the **time
dilation** (§3: stretch 1.00–2.31×, only three tracks at video speed) plus the contact-timing
disagreement (the planted-foot anchor matches the video in only 26–77 % of clear frames) — an
imitator would learn the *slowed* motion on the wrong foot schedule. The two are separable
experiments: fix grounding first, then re-measure the tempo.

## 9. UNMEASURED / UNVERIFIED

* **Implied foot slip is unmeasured**: `slip_implied_med/max` are `nan` for all 12 tracks — no
  contact run with a constant anchor choice lasted ≥0.4 s, so the code path never fired.
* **The landmark noise floor was measurable for one track only** (`shot_entry_full`: 0.371 m/s) and
  it is *larger* than most measured G1 slip values — landmark-derived slip is not resolvable at this
  noise level; the `slipG1max` column is the trustworthy one.
* `heading_*` and `net_travel_m` are computed into the JSON but are not analysed in this report.
* The "video" view shares the take-level flooring convention, so video-vs-G1 vertical comparisons are
  relative to the same (per-take) plane; absolute heights above the real mat are not established for
  the video side.
* **The terminal predicate's thresholds encode OUR upright stance, not a general "good posture"**:
  `TILT_MAX_DEG = 20` is an upright-stance bar (the video tracks legitimately end pitched 35–94°),
  "feet flat" is an absolute `z ≤ 0.035 m` test rather than a planarity-plus-own-floor test, and the
  hull is built from the *z-filtered* subset of sole points (so a foot with its heel up contributes
  only its low corners).  The verdicts above are therefore *definition* verdicts (does the track end
  in OUR stance), which is what the operator's "every behaviour returns to a good stance" requires;
  a general posture-quality instrument would need those three changes first.
* Whether the *repaired* grounding restores the knee-sprawl family is untested.
