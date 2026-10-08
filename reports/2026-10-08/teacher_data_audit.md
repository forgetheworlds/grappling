# Teacher-as-BC-expert and data-fitness audit

**Agent:** TeacherDataAudit · **Date:** 2026-10-08 · **Repo:** `/home/ubuntu/grappling`
**Assignment:** the other half of the operator's diagnosis ("bad data, and poor teacher signals"):
is the stabilized teacher fit to be a BC target, and is each data source fit for the objective
S5/S6 use it for?

**Deliverables**

| artifact | what it is |
|---|---|
| `scripts/audit_teacher_data.py` | the measurement script (re-runnable from the repo root; prints tables + its own verification) |
| `tests/test_teacher_data_audit.py` | 16 number-asserting tests over the script's computation |
| `reports/2026-10-08/teacher_data_audit.json` | every number in this report, machine-readable |
| this report | the verdicts, the fitness table, the repair/re-acquire/ discard list |

**Reproduce**

```
.venv/bin/python scripts/audit_teacher_data.py              # ≈4 min (shared box)
.venv/bin/python scripts/audit_teacher_data.py --quick      # no drill re-runs, ≈95 s
.venv/bin/python -m pytest tests/test_teacher_data_audit.py -q
```

The script self-checks and exits non-zero on failure — final run: **8/8 checks PASS**. Its
checks: teacher targets always in ctrlrange; holdability class counts sum to frame counts; the
holdability aggregate **reproduces the published support-envelope number (6.921 % vs 6.921 %)**
and the per-technique class counts on **all 14 robot-tracks**; the landmark track is 15.0 Hz;
GrappleMap is on the 1 mm grid; the PD baseline reproduces the published phase-2 ablation at
seed 0 (all seven references); drill push re-runs are self-consistent.
`pytest tests/test_teacher_data_audit.py -q` → **16 passed** (79 s).
Timings in this report were measured on the shared 4-core ARM host with sibling test suites
running (the script prints its own wall times; the full run took 225 s wall).

---

## 0. Bottom line

1. **Teacher-as-BC-expert verdict: WEAK EXPERT — cloning its action stream reproduces "a
   controller that is not falling", not "a drill being executed".** Evidence, all measured:

   * the primary S5/S6 target (DOUBLE_LEG) executes with `stay_up_min` **0.448–0.690** over
     3 seeds (published 5-seed 0.520), and its executed end pose is **1.64–2.60 m** (weighted
     landmark RMS) away from the reference end pose;
   * the one technique it keeps up (SNAPDOWN, `stay_up_min` 1.000 all seeds) still lands
     **0.29–0.31 m** from the reference pose with scorer 0.808 (gate 0.85, SNAP phase 0.723);
   * the **posture governor is active on 92.9–99.2 % of ticks on the executor robot (a)** in
     every technique (40–100 % on the defender), at mean authority **0.40–0.84 on a** — i.e.
     the commanded pose is continuously blended toward the model's standing keyframe. That
     *is* the "not falling" behaviour, in the action channel;
   * the balance offsets **oppose the residual tracking error on 77–100 % of ticks**: the
     action is dominated by balance corrections that pull the joints away from the reference.

2. **Data sources fully FIT for the objective S5/S6 assigns them: 0 of 5.** All five are
   FIT-ONLY-FOR narrower uses (GrappleMap as a pose skeleton; the video references as the
   stance spec / style prior; the MediaPipe track for coarse style statistics; the drill
   traces for evaluation; the teacher logs for evaluation — after adding the action channel),
   and three are **UNFIT for the objective they are actually wired to**: the GrappleMap
   references as dynamic imitation targets, the video references as executable/imitable
   demonstrations, and the teacher's logged traces as BC datasets (no action channel).

3. **Single most expensive finding: the S5/S6 imitation plan is built from three layers that
   are each measurably unfit for imitation of a dynamic double-leg — the reference (81.9 % of
   DOUBLE_LEG-a frames statically infeasible; 0.28 s of holdable motion), the teacher (fails
   2 of 3 seeds on stay-up; governor-driven actions), and the logs (no action labels).** The
   fix is not better cloning of the current demo; it is re-acquiring the demo under
   dynamics-in-the-loop retargeting (or restricting BC to the phases the teacher actually
   executes — holds and snapdowns).

---

## 1. Teacher as a BC expert

### 1.1 What was measured

`TeacherController` on the acceptance scene (`robots/wrestling_scene.xml`, 50 Hz control /
2 ms physics, seeded initial perturbation identical to `run_episode`), techniques STANCE,
SNAPDOWN, DOUBLE_LEG, seeds 0–2. Per tick and per robot the instrumented replay
(`instrumented_rollout`) records: the 29 commanded targets, the **effective reference**
(raw reference + measured pose trim), the simulated joints, the balance offsets `dq`
(post-clamp, post-rate-limit), actuator saturation (target at `ctrlrange`), rate-limit binding
(`|Δdq| = max_rate = 0.25 rad/tick`), the posture-governor weight `alpha`, the unclipped
capture-point error, the live contact-based CoM margin, and the base error against the
reference. Acceptance metrics (`stay_up_min`, landmark distance) come from the unmodified
`run_episode` on fresh controllers; the BaselinePD contrast uses the same machinery.

### 1.2 The measured action stream (seed 0 shown; seeds 1–2 in the JSON)

| technique | robot | joint track RMS (deg) | sat. ticks | rate-clip ticks | alpha mean | \|dq\| p95 (rad) | opposes resid. | landmark (m) | stay_up_min |
|---|---|---|---|---|---|---|---|---|---|
| STANCE | a | 1.94 | 0.00 % | 0.00 % | **0.55** | 0.092 | 98.5 % | 1.23 | 0.833 |
| STANCE | b | 3.11 | 13.64 % | 3.03 % | **0.55** | 0.134 | 99.2 % | — | — |
| SNAPDOWN | a | 4.00 | 0.00 % | 0.00 % | **0.78** | 0.097 | 85.0 % | 0.31 | 1.000 |
| SNAPDOWN | b | 5.27 | 0.00 % | 3.33 % | **0.90** | 0.146 | 85.0 % | — | — |
| DOUBLE_LEG | a | 3.01 | 0.00 % | 0.00 % | **0.68** | 0.085 | 84.6 % | 1.65 | 0.630 |
| DOUBLE_LEG | b | 2.61 | 19.53 % | 1.18 % | **0.73** | 0.109 | 76.9 % | — | — |

(`alpha mean` = mean posture-governor authority over the run; `opposes resid.` = share of ticks
where `dot(dq, q_ref − q_sim) < 0` with `‖q_ref − q_sim‖ ≥ 0.02 rad`.)

Over the three techniques and seeds: `alpha > 0.05` on **92.9–99.2 % of ticks on the executor
robot (a)** and 40.2–100 % on b — the governor is engaged essentially continuously on the
robot the solo drill would clone; actuator saturation is *not* the dominant defect (0 % on
most robot-runs; STANCE-b 6.1–13.6 %, DOUBLE_LEG-b 1.2–19.5 % of ticks across seeds — the
saturating joints are the **knee targets**, e.g. DOUBLE_LEG-b 33+29 ticks on both knees at
seed 0), and the rate limiter binds on 0–4.7 % of ticks.

### 1.3 Where and how it fails

* **DOUBLE_LEG (the S5/S6 primary)** — seeds fail at `stay_up_min` 0.448 / 0.630 / 0.690; the
  governor means by phase (seed 0) are STAND 0.63 (a) / 0.64 (b), LOW 0.97, DIVE 0.50 (a) /
  **1.00** (b); the executed end pose is 1.64–2.60 m from the reference end pose. The defender
  (b) saturates actuators on 19.5 % of ticks. This is the technique the operator's plan wants
  to imitate.
* **SNAPDOWN (the best case)** — stay-up 1.000 on all seeds, but the governor means on robot a
  (seed 0) are DIVE 0.24 → LOW 0.93 → RISE **1.00** → STAND 0.92: the "successful" execution is
  the safety blend carrying the robot back to the standing keyframe; the SNAP phase scores
  0.723 on the technique scorer (`data/teacher_stats.json`), below the 0.75 gate.
* **STANCE** — holdable only through a **pose trim of up to 0.30 rad (17.2°) at the waist**
  (a: hip +0.10, knee −0.20, ankle −0.10, waist +0.30; b: hip +0.10, ankle +0.20, waist −0.10).
  The trim is applied to the *effective reference* the controller tracks — i.e. the BC target
  is the trim, not the technique — and the report already measured that this trade keeps
  stay-up at 0.76 and the scorer at 0.60 (`reports/2026-10-08/teacher.md` §10).
* **Saturation/rate clipping is not the mechanism of failure**: DOUBLE_LEG falls with the
  actuators inside range; the governor is what saturates (alpha → 1) and even at 1.0 it cannot
  execute a step.

### 1.4 Would cloning these actions reproduce a good drill?

No — for the dynamic phases it would clone a **balancer that keeps the G1 near its standing
keyframe while tracking a trimmed, physically infeasible reference**:

* the action `ctrl = effective_ref + dq` is dominated by (i) the trim (a fixed posture repair)
  and (ii) the governor blend toward `safe_q` (authority reaching 1.00; run means 0.31–0.90),
  with (iii) balance offsets that oppose the residual on 77–100 % of ticks;
* the *executed* behaviour, not the reference, is what BC would learn: landmark 1.6–2.6 m
  (DOUBLE_LEG) and 0.29–0.31 m (SNAPDOWN) from the reference end pose; stay-up below the
  gate for the shot; a scorer below the technique gate for every technique;
* and the "good" parts it would clone are exactly the fall-prevention reflexes — which S6's
  DAgger correction is supposed to *fix up*, not learn from.

For the STANCE hold the picture is different: the teacher is a *competent holder* of a
measurably stable posture (long continuous holds with centimetre-level pose margins; its
measured CoM stabilisation authority is ±5 cm and its direct channels saturate at the foot
edge — `reports/2026-10-08/teacher.md` §2, §12), and BC on holds is meaningful — provided the
imitation target is defined as the *trimmed* stance (i.e. explicitly a posture repair, not the
schematic crouch).

### 1.5 Hidden-state dependence (a BC-feasibility defect, measured)

The teacher's action at tick `t` depends on internal feedback state that a Markovian BC policy
does not observe in the actor contract (`src/solo/obs.py`: base/joint proprio + prev action +
command + skill one-hot): the task integrator `e_int` (clamped 0.12 m), the height PI state,
the CoM-velocity low-pass, the **rate-limiter memory `_dq`**, and the governor state `alpha`
(which reaches 1.0 on the failing phases). Measured magnitudes in the replays: `alpha` run
means 0.31–0.90 (up to 1.00); `|e_int| ≤ 0.12 m` (teacher clamp; the drill law ships 0.03 m
with a documented windup finding: "with a moving reference the same gain winds up a 3 cm bias
and slowly walks the CoM out of the support", `src/drill/balance.py:54-59`); `dz` clamp 0.05 m. The actor obs
carries `prev_action` but none of these states. Cloning the *actions* therefore does not clone
the teacher's feedback law unless the policy is made recurrent/state-augmented or trained with
DAgger from live teacher rollouts (which S6 does — but its ceiling is the weak expert above).

---

## 2. Teacher stability envelope

The teacher path and the drill path share the balance law (the drill's `BalanceParams` is
documented as the teacher STAND row and its test asserts equality with `teacher/stabilizers`).
The `src/drill` FeasibleDrill port is the runtime controller that produces the measured
push/cadence behaviour, so the envelope is measured there and cross-checked against the
teacher replays (which never saturate either).

### 2.1 Push envelope — measured now, with the mechanism

Re-runs this session (`RunConfig(controller="feasible", rung="L1", start="stance",
push(t=6.0, dur=0.12, fx=-F))`, 12 s each):

| push | impulse | outcome | CoM margin min | actuator saturation | governor max | max target step |
|---|---|---|---|---|---|---|
| −20 N | 2.4 N·s | **recovered** | **+0.0295 m** | 0.000 | 0.00 | 0.031 rad/tick |
| −35 N | 4.2 N·s | **falls** (margin min at t=8.68 s, 2.6 s after the push) | **−0.6204 m** | 0.000 | **1.00** | 0.042 rad/tick |
| −50 N | 6.0 N·s | **falls** (margin min at t=7.40 s) | **−0.6304 m** | 0.000 | **1.00** | 0.054 rad/tick |

On-disk traces agree: `FINAL_L1_push` (3 × 20 N, one 90 s episode) recovered with margin
+0.0295, saturation 0, governor 0; `FAIL_PUSH90` falls with margin −0.6277, saturation 0,
governor 1.00. The older battery (`data/drill/push_battery.json`, 06:03) records 20 N 2/2,
35 N 1/3, 50 N 0/2, 70 N 0/2; one on-disk clip named `FAIL_PUSH35` contains a *recovered* run
(margin +0.020) — a naming/content contradiction worth fixing but not a mechanism question.

**Binding constraint, named: the CoM/support geometry with no step recovery — not
saturation, not rate limits, not integrator windup.** At every failed magnitude the position
targets stay 100 % inside `ctrlrange` (sat_frac 0.000), the largest target step is 0.054 rad
against the 0.25 rad/tick rate clamp (21 %), the integrators are hard-clamped (task 0.03 m,
height 0.03 m; the teacher's 0.12 m task integrator was deliberately reduced *because* of a
measured 3 cm windup walking the CoM out of support), the governor saturates at 1.0 and blends
the pose to the safe stance — and the CoM margin still dives to −0.62 m, i.e. the capture
point left the foot support polygon and the controller has no way to step. The failure is
kinematic, not actuator-level. The recovery boundary sits between 20 and 35 N (2.4–4.2 N·s
sagittal impulse) with the *guaranteed* envelope at 20 N (2.4 N·s), matching the battery's
2/2 and the 3/3 in-episode record.

### 2.2 Cadence: where 12.1 s/step comes from

The operator's "≈12.1 s per step" is the measured best reliable cadence
(`data/drill/motion_singles.json`, run `E28f`: 5 steps / 70 s, cadence **12.145 s/step**,
step gaps 13.44 / 16.56 / 4.82 / 13.76 s, 0 falls, `margin_p05` +0.0335 m, `saturation_frac`
0.0, `safety_blend_frac` 0.0051, 3 timeouts). Decomposed on a comparable run's event timeline
(`M_w28v`): step_start → shift_done (the support transfer) took **3.50 s and 6.22 s** to move
the CoM **0.118 / 0.189 m** (required 0.137 / 0.194 m) through the load-transfer gate
(`margin_support` 0.022 / 0.020 m, CoM-velocity gate 0.04 m/s); lift+transport+land ≈ 0.6 s;
settle-to-mid-foot ≈ 1.5 s; inter-step recentre ≈ 1.5–7 s. The step primitive's own phase
budget is LOAD 0.45 + MOVE 0.34 + LAND 0.30 + SETTLE 0.35 = **1.44 s**; the measured cadence is
**8.4× that**, and the dominant term is the weight-transfer/recentre wait, again bounded by the
CoM authority over the planted feet (the transfer is executed at ~0.02–0.04 m/s effective).
This is the same binding constraint as the push envelope: **support-geometry-limited CoM
authority, no stepping recovery**.

---

## 3. Data fitness, per source, per intended use

| source | what it physically contains | measured quality | intended use (plan) | verdict |
|---|---|---|---|---|
| `data/refs/*.npz` (7 GrappleMap techniques) | 2×36 qpos @50 Hz + t (no timestamps in the source); 46–288 frames | only **6.92 %** of all 2 514 robot-frames inside the foot support (8.23 % of the 2 115 grounded; 47/2 115 = 2.2 % strictly stable); ankle torque NOT the limit (≤23.9/50 N·m, cited); all 7 below the 0.80 stay-up gate under naive PD (below) | BC/imitation targets for S5/S6 (shot family), phase vocabulary | **UNFIT as position/dynamic imitation targets.** FIT as skeleton/phase/label source (what the scorer and phase labels use). |
| `data/refs_video/*.npz` (12 video-retargeted tracks) | single-robot qpos_a (T,36) @50 Hz, `qpos_b` zeros, landmark RMS 4.2–6.8 cm | **inside-support frames across all 12 tracks: 1 in total** (stance_hold; 0 stable frames everywhere); grounded frames infeasible 44–100 %; airborne share 0–93 %; median lowest-sole 15.5–259.6 mm (47.9 mm for `stance_hold`: the "stance hold" hovers ~2.8–5.6 cm above the mat, §3.2); drill_l2 measured topple in 0.5–1.5 s even with the clamp | S5 demo / imitation corpus (`reports/2026-10-08/imitation.md` uses their site targets) | **UNFIT for execution/imitation as shipped.** FIT-ONLY-FOR-X: the 41-param stance spec and coarse style statistics. |
| MediaPipe track (`data/references/yt_gBAhX5t-GW4/pose/landmarks.npz`) | 6 401 frames, 33 landmarks, t 26–823 s | **15.0 Hz** (dt 0.0667 s, gap-count 3, one 139 s chapter join), 0.997 detected, 19 all-NaN frames; hip-centred (hip-mid std 0.1–0.5 mm → **no global translation**); planted-window high-pass p2p: ankles **2.3–4.9 cm**, knees 1.5–4.9 cm, **wrists 4.0–19.2 cm**; no contact forces, no joint angles | source for the retarget and the imitation corpus | **FIT-ONLY-FOR-X**: coarse style/phase statistics (>0.5 s timescales) with the documented weights. UNFIT for limb-level positions/timing (hands), global motion, or force/contact inference. |
| GrappleMap DB (`third_party/GrappleMap/GrappleMap.txt`) | 725 nodes, 1 485 edges; keyframes (median **4** frames/edge, max 22), 1 mm quantisation, **no timestamps** (5 Hz / 10 Hz-detailed playback convention) | per-edge max per-frame joint delta median 0.61 m — keyframes, not motion; 10.7 % of edges "detailed" | retarget source for the 7 techniques; topology/skeleton | **FIT-ONLY-FOR-X**: pose vocabulary, topology, node anchors. **UNFIT as a motion source**: no timing and 1 mm-quantised keyframes cannot be repaired into a dynamic trajectory by interpolation (that is exactly the phase-2 attempt that fails physically). |
| teacher's own logged traces (`data/teacher_stats.json`, `data/drill/*.npz`, `run_episode` outputs) | stats JSON: aggregate metrics only. Drill npz: 29-dim `ctrl` + margin/governor/saturation at 50 Hz, **but produced by the drill controller, not by `RobotTeacher`**; `run_episode` stores qpos traj only | measured: `EpisodeResult` has **no action channel**; teacher_stats has no per-tick rows; 1 on-disk clip's name contradicts its content (`FAIL_PUSH35` recovered) | BC datasets / S5 "capture obs-action pairs" | **UNFIT for BC as-is; FIT-ONLY-FOR-X for evaluation** (stay-up/scorer/penetration). Repair is small (log `ctrl` + `dbg`) and this audit's instrumented rollout shows the cost is zero simulation time. |

### 3.1 Per-reference dynamic infeasibility and PD execution

`inside` = frames with the CoM inside the foot support region (margin ≥ 0; stable is margin
≥ 0.02 m), on the support-envelope classifier (touch: sole site z < 0.045 m; region: hull of
the touching sole spheres). Naive-PD column: `stay_up_min` at seed 0 (pure reference replay,
no balance law).

| technique | a: stable/marginal/infeasible/airborne | b: stable/infeasible/airborne | PD stay_up_min | usable as an imitation target? |
|---|---|---|---|---|
| STANCE | 0 / 0 / 118 / 0 | 0 / 118 / 0 | 0.208 | only as a trimmed hold (needs the 17.2° waist repair) |
| DOUBLE_LEG | 4 / 10 / 127 / 14 | 17 / 34 / 90 | 0.444 | **no** — 2.6 % stable frames; 0.28 s/0.62 s holdable windows over 3.08 s |
| SINGLE_LEG | 0 / 49 / 164 / 17 | 0 / 168 / 45 | 0.200 | no — 0.98 s/0.34 s windows over 4.58 s |
| BODY_LOCK | 18 / 10 / 148 / 0 | 0 / 118 / 44 | 0.229 | no — 10.2 % stable (a); 0.56 s/0.26 s windows; schematic clinch penetration |
| SNAPDOWN | 0 / 0 / 46 / 0 | 0 / 46 / 0 | 0.778 | no stable frame; the teacher executes it only via the governor blend |
| SPRAWL | 0 / 0 / 203 / 85 | 8 / 183 / 84 | 0.105 | no — defender prone by design; 0.32 s window (b) |
| STAND_UP | 0 / 0 / 236 / 8 | 0 / 232 / 12 | 0.000 | no — 100 % of standing frames infeasible (reference geometry broken; §11 of `teacher.md`) |

Naive-PD `stay_up_min` at seed 0, measured this session for all seven references
(0.000–0.778): every reference is at or below the 0.80 stay-up gate under pure position
replay, SNAPDOWN (0.778) the only near-miss — the "6/7 references fail execution" claim is, if
anything, optimistic at this seed. All seven values reproduce the published phase-2 ablation
table exactly (`reports/2026-10-08/teacher.md` §13).

The ankle-torque bound is not binding (≤23.9 N·m of the 50 N·m actuator limit, cited from
`support_envelope.md`); what binds is the CoP-inside-the-patch geometry: **6.92 % of grounded
frames inside support, 174/2 514** (my measurement reproduces the published aggregate to
0.000 pp and its per-technique counts on all 14 robot-tracks).

### 3.2 Why the video references float (measured cause, for the retarget fix)

Spot values (min sole-site height of the lowest of the four sole sites): `stance_hold`
frames 0/110/220 → 0.041 / 0.028 / 0.042 m; `level_change_fast` → 0.040 / 0.018 / 0.020 m;
`shot_recover` → −0.007 / 0.016 / 0.006 m. A flat G1 foot rests with sole sites 0–0.02 m; the
hold references therefore hover a few centimetres above the mat, and the per-window
`floor_offset_m` in `retarget_summary.json` varies from −0.23 to −0.78 m across windows. This
is a *floor/scale alignment* defect of the two-stage (video → kinematic IK → joint targets)
pipeline: the geometric bias the imitation report already flags is measurable as a vertical
offset, and it is not repairable by clamping (drill_l2: topples even with the CoM clamp).

---

## 4. Acquisition vs repair (prioritised)

1. **S5/S6 demo (teacher as expert) — re-acquire, do not clone.** The action stream is a
   balance/governor signal, not the technique (governor active on ≥92.9 % of the executor's
   ticks, run means 0.40–0.84; landmark 1.6–2.6 m on DOUBLE_LEG). Repair option with a number:
   restrict BC to the phases the teacher *executes* (STANCE-family holds with the trim,
   SNAPDOWN setup, level changes) — measurable acceptance: landmark < 0.35 m and scorer ≥ 0.85
   for those phases. Otherwise re-acquire with a stepping controller (the step primitive exists
   and is measured: clearance 3.3 cm, slip 0.0005 m, placement 0.9 mm — `M_w28v` events) so the
   expert actually changes support. BC on the current expert reproduces a fall-avoiding
   controller.
2. **`data/refs_video` — re-acquire with dynamics in the loop.** Two-stage kinematic
   retargeting injects a geometric bias (cited: `reports/2026-10-08/imitation.md` §6 rationale);
   the measured bias here is 2.8–5.6 cm of float in the hold plus an infeasible CoM in 44–100 %
   of the grounded frames. Repair attempt with a number: per-window vertical re-grounding +
   contact fix to bring the median lowest sole within 0–2 cm — if that alone does not clear a
   held-out tracking gate (teacher-stabilized execution of the re-grounded track: no fall under
   the 20 N push, scorer ≥ 0.75), discard as control targets and keep only the stance spec.
3. **`data/refs` (GrappleMap) — repair the *use*, not the file.** As motion they are not
   repairable (no timestamps, keyframes, 1 mm grid; interpolation is the failed phase-2 path).
   Keep as the skeleton (23 landmarks, phase vocabulary, scorer relations). Any dynamic
   imitation must use a reference regenerated from the skeleton *with the physics in the loop*
   (teacher/step primitive producing the trajectory), or the skeleton used as an *objective*
   (site/landmark imitation with dynamics), which is what `imitation.md` recommends.
4. **MediaPipe track — repair by filtering only where it is valid.** Windowed medians/2.5 Hz
   low-pass for style/phase statistics (the existing weights); do not repair limb-level timing
   or hands from this track (4–19 cm p2p at 15 Hz); if precise hand/ankle trajectories are
   needed, re-acquire at a higher rate or with a contact-constrained fitter that pins the
   planted foot (ankle p2p 2.3–4.9 cm vs the 17.5 cm foot: 13–28 % of the foot length).
5. **Teacher traces — repair (cheap, do it in S5).** Add the action channel and the hidden
   state to the capture: `ctrl` (29), `dq`, `alpha`, `e_int`, `dz`, phase label, and the
   reference row index, per tick, per robot. This audit's instrumented rollout is a working
   example and costs no extra simulation (rollout wall 0.28–1 s per seed on this host).
6. **Discard candidates:** the `FAIL_PUSH35` clip (name contradicts content: recovers with
   margin +0.020 while the battery and my re-run show 35 N falls) and `STAND_UP.airborne.npz`
   (already superseded). Update `data/drill/push_battery.json` or mark it stale — the current
   tree's limit measured this session is 20 N recovered / 35 N falls.

---

## 5. Verification, limits, UNVERIFIED

* Reproduced (independent measurement in this audit): support-envelope aggregate 6.921 % and
  all 14 per-robot class counts; the phase-2 PD ablation for **all seven** references
  (0.000–0.778 at seed 0); the 20 N push recovery with margin +0.0295 m; the 90 N failure
  mechanism (governor 1.0, saturation 0); the 12.145 s/step cadence summary.
* Instrumented replays reproduce the acceptance metrics within seed noise (e.g. SNAPDOWN
  stay_up_min 1.000 vs published 5-seed 0.956; DOUBLE_LEG 0.448–0.690 vs published 0.520 mean).
* **UNVERIFIED / not measured here:** (a) the *causal* attribution of the landing failure in
  DOUBLE_LEG per phase by force (the balances are measured, not identified); (b) whether a
  recurrent BC policy can absorb the teacher's hidden state — not trained here; (c) the
  video-retarget floor-offset root cause (the 2.8–5.6 cm float is measured; which stage of
  the retarget produced it is not); (d) contact forces in the MediaPipe track (they do not
  exist in the data, not an omission); (e) the 35 N marginality across seeds (one seeded
  re-run per magnitude; the battery's 1/3 is cited, not re-measured 3×).
* All timings on this host were measured while sibling agents' tests ran (shared 4-core ARM
  box); the script labels its per-section wall times.
