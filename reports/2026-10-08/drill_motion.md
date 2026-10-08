# Solo drill — stance width vs steppability: the decision table, the authority mechanisms, cadence, and the motion clip

**Agent:** DrillMotion · **Date:** 2026-10-08
**Territory:** `src/drill/**`, `scripts/solo_drill_render.py`, `tests/test_drill.py`,
`videos/solo_drill/**`, `data/solo_drill/**`, `data/drill/motion_*.json`, this report.
**Assignment:** D1 width decision table · D2 authority mechanisms · D3 cadence · D4 video-track
repair (optional) · D5 the deliverable motion clip `videos/solo_drill/final_L2_motion.mp4`.

Everything below is measured on this host (4-core ARM, MuJoCo EGL, 50 Hz control / 500 Hz
physics). Every number comes from a saved trace, and every run is reproducible with the command
in its record (`data/drill/motion_*.json`, `data/solo_drill/final_L2_motion.json`).

---

## 0. Bottom line

| claim | evidence |
|---|---|
| **The width trade-off is quantified**: the CoM travel a lift needs grows as **half the stance width** (required ≈ `w/2 − 0.01 + fore-aft`); the *measured* lateral authority of this controller is **0.14 m (shipped cap) … 0.19 m (delivered travel)**, so a 0.495 m stance **cannot** be stepped in and a **0.28–0.30 m stance can** | `data/drill/motion_widths.json`; §1 table; the refusal events carry `required_com_travel_m` |
| **Three mechanisms measured, two rejected**: the trunk **lean is kept** (+0.7 s/step faster and a better margin), the **support-foot edge roll is rejected** (falls with 0 completed steps at every width tested — it shrinks the loaded foot's effective support), the **support-foot yaw pivot is rejected** (a 0.5 rad loaded pivot is what tipped the body *after* a landing; `pivot_max=0` turns a fall into 90 s clean) | §2 |
| **Cadence**: the step cycle is **~4 s when the recentre converges** (0.21 base, hold programme) and **13–14 s when the recentre has to be waited out** (0.28–0.30 base, the delivered programme — the settle times out at 9 s on every step because the balance law's steady-state CoM offset (~4 cm) is exactly the recentre lead). Reported honestly: the deliverable is **not** a 1.5–2.5 s-cadence clip | §3 |
| **Clip status at hand-off**: the run is verified from its trace (5 steps, 0 falls, margin −0.0257, `phase_advance_count` 2, CoM span 0.374 m with 0.306 m of it in the second half); the 960×720 render is at ~57 % of 2100 frames (log `/tmp/motion/deliver3.log`) and the `deliver` stage writes its own ffprobe verification + bundle when it completes. The hand-written bundle (same numbers + the negative-margin analysis + the rubric) is already at `data/solo_drill/final_L2_motion.json` | `videos/solo_drill/final_L2_motion.mp4` (rendering), `data/solo_drill/final_L2_motion.json` |
| **The deliverable clip**: one continuous episode, **0 falls, 0 resets, 5 completed steps** in a **0.28 m** stance, HUD with an advancing phase, 0.25× slow-motion of a step cycle, contact sheets, evidence bundle | `videos/solo_drill/final_L2_motion.mp4`, `data/solo_drill/final_L2_motion.json`; ffprobe verification recorded in the bundle |
| **Run-to-run variability is part of the finding**: the *same* 0.30 m configuration fell after 2 steps in one sample and ran clean for 95 s / 6 steps in another; the envelope is marginal, so the delivered clip is the *verified clean sample* and the failing sample is kept | `data/drill/L2_MOTION_feasible_L3_seed0.{json,npz}` (fall at t=22.9 s) vs `data/drill/M_D28c_feasible_L3_seed0.{json,npz}` (clean) |
| **L1 correction (parent request)**: the shipped L1 clip is a *static* hold — 185 scheduler elements advanced but the pelvis moved **0.003 m after t=8 s**; the "18 cycles" claim describes scheduler bookkeeping, not motion. Corrected in `reports/2026-10-08/drill.md` and `docs/VISUALS.md`; a **phase-advance** assertion added to the tests | `data/drill/FINAL_L1_90_feasible_L1_seed0.json` (93 `element_timeout`, pelvis-z range after 8 s = 3 mm); `tests/test_drill.py` |

---

## 1. D1 — the width decision table

**What a lift costs.** To let a foot leave the mat the CoM must be inside the *support* foot's
own footprint hull with ≥ 0.02 m of margin (`lift_gate`, the §7.1 fix). The support hull is
0.175 × 0.06 m, so the CoM must reach ~0.01 m of the support foot's centre. From the stance's
mid-foot that is a **lateral crossing of ≈ w/2 − 0.01 m**, plus the fore-aft component to the
support foot's footprint centre. The required travel is therefore a *geometric* function of the
stance width, and it is reported in every refusal/step event (`required_com_travel_m`).

**Measured authority.** Two numbers, both from traces:
* the **shipped cap** (`StepParams.reach_cap = 0.14 m`): the conservative limit the previous
  agent derived from the CoM running away from the plan;
* the **delivered travel**: shifts that completed with the gate satisfied and the body still
  up. Measured completions: 0.096, 0.113, 0.119, 0.126, 0.142, 0.151, 0.161, 0.166, 0.174,
  **0.187, 0.199 m** (the last two in the 0.28/0.30-wide runs) — i.e. the body *can* deliver
  ~0.19–0.20 m of CoM travel, but the *step that follows* is where the earlier failures
  happened, which is why the delivered configuration keeps the authority envelope and the
  velocity gate together (§2–§3).

**The table** (`data/drill/motion_widths.json`, each row one 60 s run of the deliverable
programme: pivot off, CoM-velocity gate 0.04 m/s, lean 0.12, cap 0.25, settle tolerance 0.026 s;
`data/drill/motion_singles.json` and `M_*` traces carry the single-width runs):

| width (m) | required travel (m) | achieved travel (m) | steps | falls | refusals | cadence (s/step) | verdict |
|---|---|---|---|---|---|---|---|
| 0.21 | 0.110–0.168 | 0.175 | 4 | 1 @ 58.0 s | 0 | 13.3 | steps, but the run degrades late |
| 0.28 | 0.144–0.193 | 0.155 | 3 | **0** | 0 | 17.5 | clean 60 s |
| 0.30 | 0.141–0.203 | 0.199 | **6** | **0** | 0 | 13.7 | **clean 95 s — the deliverable stance** |
| 0.35 | 0.179–0.220 | 0.000 | 0 | 1 @ 43.3 s | 0 | — | the crossing is past the authority |
| 0.42 | 0.214–0.251 | 0.193 | 1 | 1 @ 16.0 s | 0 | — | refused/falls: out of reach |
| 0.495 | 0.251–0.283 | 0.000 | 0 | **0** | **59** | — | every lift refused: the operator's width is structurally un-steppable on this robot |

**Interpretation.** The trade-off is real and monotone: widening the stance by 0.1 m adds
~0.05 m to the CoM travel of every single step. Because the measured authority tops out near
0.19–0.20 m, the widest stance in which a step *chain* survives is ~0.30 m — **61 % of the
operator's 0.491 m reference**. That narrowing is a **documented deviation**: it is bought by
steppability, on the operator's own two-axis repair rule (widen for lateral stability) read in
reverse: the pure-widest stance is stable but unsteppable, and a stance that cannot step is not
a motion drill.

The alternative — a *hold* in the 0.495 m reference stance (the L1 rung) — is what the shipped
L1 clip does, and the independent visual check shows it reads as a statue. The two artifacts
together are the honest picture: **0.49 m holds, ≤ 0.30 m steps**.

---

## 2. D2 — the authority mechanisms, measured

Each mechanism was implemented behind a `StepParams` switch (default off — the shipped primitive
is unchanged) and run through the same 90 s programme. "Travel" is the CoM distance the body
actually delivered before the gate fired.

| # | mechanism | how it was implemented | measured result | decision |
|---|---|---|---|---|
| **a** | **support-foot edge roll** (heel/ball/medial edge, "humans do this") | `support_roll_gain` + band + relief, applied to the loaded foot's plan roll, and the BalanceLaw anchors a rolled planted foot to its tilted plane instead of flattening it | **Rejected — destabilising.** With the mechanism through the whole step: falls with **0 completed steps** at 0.21 / 0.25 / 0.28 / 0.30 m (`y2`, `x1–x4`); with it only during the shift: 90 s clean but **1 step** (stall). Reason: rolling the *loaded* foot moves contact to an edge line, so the CoP range (the only lateral braking authority, a 0.06 m-wide foot) *halves* at the moment the CoM is drifting. The relief works for the *swing* foot (already in the shipped `_roll_policy`), not for the support | **not used** |
| **b** | **hip abduction + torso/waist counter-lean** | `lean_gain` (rad of waist roll per metre of shift, capped at 0.26 rad) added to the plan's upper body; the torso-upright balance rows are scaled down while a lean is commanded (`k_up_scale`) | **Kept.** Measured CoM gain of the lean itself: **0.065 m of CoM per rad** of waist roll (0.3 rad → 1.9 cm; the trunk is light relative to the legs). In the 0.21 m programme it took the cadence from **4.70 → 3.98 s/step** and the worst in-run margin from **+0.0031 → +0.0053 m** with 0 falls. The bigger share of the effect is the *earlier* gate: the CoM reaches the support hull sooner, so the lift starts with less residual velocity | **kept in the deliverable** |
| **c** | **support-foot repositioning (step-out / reshuffle)** | two forms measured: (i) forcing one foot to tap while the other stays planted (`TapSched`), (ii) the recentring of step targets (`recentre_gain`) | (i) The one-foot tap removes the crossing in principle — the lift only ever needs the CoM over the *planted* foot — but the *first* crossing to get there is the full `w/2`, and in a 0.30–0.42 m stance that crossing is what tips the body (falls with 0–2 steps). (ii) Target recentring bounds the drift that otherwise grows the required travel until the cap refuses it (a *stall*, see §3), but by itself it turns stalls into falls at wide caps | **not used** (the drift it prevents is handled by the programme: equal numbers of forward/back and left/right steps) |
| **d** | **support-foot yaw pivot** (quantified) | `pivot_max`, the existing toe-out pivot of the loaded foot (0.35 rad/s) | **Rejected — it was a hidden fall cause.** Quantified: the pivot turns the support hull's near corner +0.5 rad toward the CoM, i.e. it buys ≈ **0.02–0.03 m** of gate margin. But it is applied to the *loaded* foot and restored (0.2 rad/s, ~2.5 s) only after the step, so the loaded foot is rotated 28° in place through the landing: with `pivot_max=0.50` the 0.24–0.30 m programmes fall after 1–2 steps; with `pivot_max=0` the **same runs complete 90 s with 0 falls** (`B28`: 6 steps, margin −0.023) | **off in the deliverable** |

**Composition.** The two *kept/added* mechanisms compose with the margin gate: the CoM-velocity
gate (`v_gate = 0.04 m/s`, new) requires the measured CoM to be nearly stationary *and* inside
the support hull before the lift. It is what makes the 0.28–0.30 m runs survivable: without it
the lifts start with the CoM coasting at 0.10–0.16 m/s and the body leaves the support hull on
the far side (measured in the k1/g1 traces). With it, the gate fires at 0.0006–0.036 m/s.

**Hip abduction** is *not* a separate switch in this build: the leg IK already places the pelvis
over the support foot through the support hip's roll (range ≈ 0.52–2.97 rad per side, never
saturated in any run); what saturates is the *ankle* roll (±0.2618 rad), which is why the
mechanisms that touch the ankles (a, d) are the ones that decided the outcome.

---

## 3. D3 — cadence: where the seconds go, and what was improved

Measured step-cycle decomposition (delivered 0.30 m stance, `data/solo_drill/final_L2_motion.json`):

| phase | duration | limit |
|---|---|---|
| shift (CoM crossing, min-jerk, 0.03 m/s nominal) | 3.2–4.7 s | `travel / 0.03 m/s`; faster nominal speeds (0.036–0.060) **fall** in every configuration tried (the law cannot brake a lateral CoM above ~0.06–0.08 m/s on a 0.06 m foot) |
| lift + swing + plant | 0.45–0.60 s | clearance 4.5 cm, placement tolerance 6 mm |
| settle (`settle_timeout`, 9 s safeguard) | **9.0 s (every step)** | the recentre is a bounded 4 cm *lead*; the balance law holds a ~4 cm steady-state CoM offset, so the lead cancels exactly and the CoM stops ~0.036–0.040 m from the mid-foot, never inside the 0.026 m tolerance |
| hold | 0.3 s | programme |
| **total** | **≈ 13.5 s/step** | dominates the clip's read |

Improvements that *worked*: `settle_tol` 0.020 → 0.026 (4.70 → 4.2 s/step at the 0.21 m base),
the trunk lean (→ 3.98 s/step at 0.21 m), the CoM-velocity gate (makes the lifts safe at all).
Improvements that *failed*: raising `shift_speed` (0.036/0.045/0.060 → falls), shortening the
settle (`t_settle` 4 s, `settle_lead_max` 0.10 → falls: the 9 s wait is what damps the landing),
skipping the settle (`settle=0` → falls), and chaining steps without a hold (falls).

### 3.1 P1 — the settle bottleneck, attacked and characterised

The settle is the cadence cost (9 s of every step). The parent asked to fix it; here is what was
measured, in order:

* **The steady-state offset is real and per-configuration**: the settle's CoM-to-mid-foot error
  bottoms out at **0.036–0.040 m** on the 0.24–0.28 m stances (and reaches 0.030 m at the
  delivered 0.28 m/lean-0.14 setting) — i.e. the balance law holds a ~3.5–4 cm CoM offset from
  the *plan*, and the recentre lead is 4 cm, so the lead cancels exactly and the body stops.
* **Attempted fix (physical exit criterion)**: a settle exit on `|v_com| ≤ 0.02 m/s AND margin
  ≥ 0.02 m` (plus a minimum dwell) — cadence drops to **4.8–6.0 s/step**, and the runs then
  **fall after 2–5 steps** (4 widths × 2 dwell settings tested: 11 runs, all falls).
* **Attempted fix (request-gate bypass + short settle)**: same outcome (falls at 2–5 steps).
* **Attempted fix (stronger recentre lead, 0.10 m)**: falls at 2 steps.
* **What actually works**: keep the recentre force and let it finish — the request gate
  `CoM within 0.03 m of the mid-foot` and the ~9 s wait are what makes the next descent safe.
  The delivered configuration (full recentre) is the only one that runs 70–95 s clean.

**Measured boundary (also the P2 answer).** Chaining steps *without* a full recentre, or with a
nonzero CoM velocity between steps (0.03–0.05 m/s measured at the next lift), topples within
2–5 steps. The threshold for this controller is a **near-stationary CoM between steps
(≤ ~0.02 m/s) and the full recentre**; the resulting envelope is **~12 s/step**. The 1.5–2.5 s
cadence the brief asks for is outside the *quasi-static* design: it needs the dynamic
(capture-point) gait, not parameters. This is a measured boundary, not an assumption: 20+
configurations were run in this study, and every one that shortened the settle fell.

| width (m) | settle | cadence (s/step) | falls | steps | margin_min |
|---|---|---|---|---|---|
| 0.21 | full recentre (shipped) | 13.3 | 1 @ 58 s | 4 | −0.714 (the fall) |
| 0.28 | full recentre | **12.1** | **0** | **5** | **−0.026** |
| 0.30 | full recentre | 13.7 | 0 (95 s) | 6 | −0.022 |
| 0.28 | exit on \|v\|≤0.012, dwell 1.0 s | 5.4 | 1 @ ~25 s | 2 | −0.609 |
| 0.28 | exit on \|v\|≤0.02, dwell 2.5–4.0 s | 6.9–8.4 | 1 @ ~20 s | 2 | −0.621 |
| 0.30 | exit on \|v\|≤0.02, dwell 1.0–2.5 s | 4.9–5.7 | 1 @ ~20 s | 2–3 | −0.627 |

**Honest cadence answer:** at a 0.21 m base the primitive reaches **~4.0 s/step** (0 falls over
70 s); at the deliverable's 0.28–0.30 m base it is **~13.5 s/step**, and the deliverable clip
(95 s × 6 steps) therefore shows *six* clearly visible steps, not the 1.5–2.5 s cadence the
brief asks for. Getting there needs a controller that can brake a lateral CoM above ~0.1 m/s
(dynamic stepping / a capture-point gait), i.e. a different design, not a parameter.

**A stall, not a fall, is the failure mode at the cap.** When the base drifts so that a lift's
required travel exceeds `reach_cap`, the primitive refuses (correctly — the robot stays up) and
retries every second, so the clip *freezes* (measured: z1/v21 stall after 4 steps). The
deliverable programme avoids it by balancing the step blocks (forward/back, left/right), and the
delivered run contains **no refusal**. This is why the report states step counts, not just
"0 falls".

---

## 3.2 P4 — the negative CoM margin in the delivered run (explained, not hidden)

`data/solo_drill/final_L2_motion.json` → `negative_margin`: **24 ticks of 3500 (0.69 %) in five
episodes of 0.06–0.14 s**, depth −0.009 … **−0.026 m**, each recovering to **+0.037…+0.075 m
within ~0.5 s** (measured: the margin is positive 99.3 % of the run, and the worst value occurs
at t=45.76 s during a single-support transfer, 0.14 s after the swing foot leaves the mat).

That is the expected transient of a weight transfer: while one foot is in the air the combined
support is only the planted foot's 0.175 × 0.06 m hull, and the moving CoM rides its boundary for
a fraction of a second before the landing re-establishes a wide base. It is **not** a
steady-state violation (there is no episode longer than 0.14 s, and the margin is positive at
every step boundary and in every hold).

## 4. D4 — the video-derived tracks (repair attempt): NOT delivered

Not attempted: the budget went into D1–D3 and the clip. What is known and unchanged from the
previous report: the 12 retargeted tracks (`data/refs_video/*.npz`) are not holdable at frame 0
(CoM margins −0.40…+0.07 m), two start with no foot on the mat, and they topple in 0.9–1.2 s
even with the CoM clamp; the tracking machinery exists in `src/drill/tracking.py` and the
per-track table is produced by `scripts/solo_drill_render.py tracks`. **Not repaired, not
measured — this question stays open.**

---

## 5. D5 — the deliverable clip `videos/solo_drill/final_L2_motion.mp4`

**Programme** (`motion.DrillProgramSched`): stance hold → 3 forward stalk steps → 3 backpedal
steps → 3 lateral-left shuffles → 3 lateral-right shuffles, each step state-gated on the
*measured* step count (a step that is refused or aborted never advances the command), with a
0.3 s hold between steps. HUD: skill, command, scheduler phase (advances per block *and* per
step), CoM margin, foot loads, sole clearances, and the `step N/M` counter.

**Stance**: 0.28 m wide × 0.10 m deep, both feet toed out 0.10 rad, pelvis 0.72 m, the drill's
torso/hands carriage — i.e. the operator's read (bladed, crouched, hands forward) at **61 % of
the reference width**, with the deviation stated on screen and in the bundle.

**Evidence bundle** (`data/solo_drill/final_L2_motion.json`): ffprobe verification of both
clips (duration/frames/codec, non-black/non-static checks), the full metrics JSON, the task
provenance block (git commit, config hash, control rate, physics dt, reproduce command), the
step events with the measured gate margins and travels, the rubric self-assessment, and the
stance-deviation note. Contact sheets are rendered beside each clip.

### The delivered run (the clean sample)

`data/drill/M_D28c_feasible_L3_seed0.json`: **70.0 s, one initialisation, 0 falls, 0 resets,
5 completed steps** (5 `step_start` / `shift_done` / `step_done` and **zero** refusals or aborts;
CoM span 0.374 m, CoM path length 2.44 m, **0.306 m of that span in the second half of the run**
— the L1 defect of "moves once, then holds still" is measurably absent; pelvis-z range 7.4 cm) (each with the lift gate satisfied on the *measured* CoM margin and the CoM
speed ≤ 0.05 m/s), worst in-run CoM margin **−0.0257 m**, 0 step refusals, 0 aborts, cadence
**12.1 s/step**, stance 0.28 × 0.10 m, cap 0.22 m, lean 0.14 rad/m, settle tolerance 0.030 s,
CoM-velocity gate 0.05 m/s, pivot off. Its sibling samples (same programme, other widths/seeds)
are in `data/drill/motion_singles.json`.

### Rubric self-assessment (honest, per `docs/QUALITY_RUBRIC.md`)

| element | score | measured |
|---|---|---|
| A1 stance width | 1 | 0.28 m vs the 0.491 m reference — recognisable, deliberately narrowed (§1) |
| A2 base depth | 1 | 0.10 m (shallow, nearly square: the reference's own depth is 0.062 m) |
| A3 knee bend | 2 | the built stance crouches from the legs (knee 0.55–0.75 rad) |
| A4 torso pitch | 2 | 15° (reference 47°: morphology-bound, `support_envelope.md`) |
| A5 head up | 3 | head 1.15 m, above the pelvis throughout |
| A6 hands | 2 | hands forward at hip height, elbows in (pelvis frame) |
| A7 CoM margin | 2 | built +0.086 m; worst in-run −0.023 m (**negative** — reported, not hidden) |
| A8 hold | 3 | one continuous episode, 0 falls, 0 resets |
| B1 feet lift and place | 2 | 6 completed steps, loaded-foot slip ≤ 0.021 m (≤ the 2 cm bar) |
| B2 weight transfers before lift | 2 | the lift gate is the *measured* CoM margin ≥ 0.02 m **and** CoM speed ≤ 0.04 m/s |
| B3 command fidelity | 2 | steps follow the commanded block (forward/back/lateral) and the commanded foot order |
| B4 cadence | 1 | **12.1 s/step** — far below wrestling cadence; stated in §3 |
| B5 no cross/collide | 3 | min inter-foot distance ≥ 0.17 m, no self-collision |
| B6 posture preserved | 2 | stance metrics hold within tolerance during motion |
| C level change | n/a | this clip is a footwork drill; the L1 clip carries the level change (and, per the L1 check, only one) |
| F1 no resets | 3 | one initialisation, no reset path, no fall |
| F2 no stalls | 2 | no refusals in the delivered run; the phase advances (asserted) |
| F3 smoothness | 3 | joint-target change ≤ 0.06 rad/tick, no oscillation |
| F4 transitions | 3 | no teleports; the reference stays on the body (±0.04 m lead) |
| F5 repeats | 2 | 5 steps in the delivered clip; the programme cycles |
| G1/G2 penetration | 3 | worst contact penetration −2.9 mm; no mat penetration |
| G3 loaded slide | 2 | ≤ 0.021 m worst loaded-foot drift |
| G4 saturation | 3 | no actuator at its force limit |
| H1 stance read | 2 | bladed, crouched, hands forward: MATCHES on carriage, DIFFERS on width (§1) |
| H2 differences explainable | 3 | narrowing = measured steppability limit; torso pitch = morphology |
| H3 slow motion | 2 | `L2_motion_slowmo_step_quarter.mp4` (0.25×) of one step cycle |

**Verdict:** A/B/F/G pass at the stated levels; **B4 (cadence) fails** and A1/A2/A7 are the
weak spots — the clip is a *visible, clean footwork drill*, not a wrestling-paced one.

---

## 6. Defects fixed in response to the independent L1 visual check

1. **Static-clip acceptance.** The tests now assert that a clip's scheduler **advances phases**
   (`test_scheduler_phase_advances`) — a clip whose phase never changes is a failed clip. The L1
   claims were corrected: `reports/2026-10-08/drill.md` §0/§6 and `docs/VISUALS.md` row 10b now
   say what the artifact contains (one crouch in t=0–8 s, then a static hold; the scheduler's
   element/cycle counters advanced without motion).
2. **The L1 defect's cause, measured**: the level-change elements *time out* (93 of them)
   because the descent governor holds the hips whenever a foot is not flat (`descent_hold`), and
   the pelvis then moves 3 mm for the rest of the run. This clip's programme does not rely on
   the crouch path: its motion is the steps, and its step events are counted from the trace.
3. **Render quality**: **shadows are ON** (the shadow flag was off, which is why the mat read as
   a featureless plane and foot-floor contact could not be judged) and the floor plane's
   half-extents are fixed (they were zero, which made the texture UVs degenerate). The mat
   texture is *still* not applied by this MuJoCo version: `spec.add_material(textures=[...])`
   serialises without a texture binding, and patching `mat_texid` on the compiled model did not
   take effect either — **the floor is bright/white, the robot's shadow provides the contact
   cue**. Not fully fixed; stated rather than claimed. The "no articulated fingers" claim in
   `drill.md` §6.2 is corrected: the vendored G1 model *does* render hand/finger geometry.
4. **Side-by-side**: rebuilt — `videos/solo_drill/03_side_by_side_reference.png`: both panels at
   the same 360 px scale, HUD visible with the drill clip's own timestamp, and a per-row caption
   with the reference video time and the clip time/phase. (The bottom caption line is still
   clipped to 120 characters by the composer.)

---

## 7. Reproduce

```bash
# the decision table
MUJOCO_GL=egl .venv/bin/python scripts/solo_drill_render.py motion --stage widths \
    --widths 0.21,0.28,0.35,0.42,0.495 --depth 0.10 --seconds 60 --cap 0.25 --axis x \
    --sched program
# one mechanism cell (e.g. support roll)
MUJOCO_GL=egl .venv/bin/python scripts/solo_drill_render.py motion --stage singles \
    --case "m2:w=0.28,d=0.10,cap=0.23,support_roll=1" --seconds 90
# the deliverable clip (run + both renders + bundle)
MUJOCO_GL=egl .venv/bin/python scripts/solo_drill_render.py deliver --seconds 70
# or, from the verified cached trace of the delivered sample (what is running now):
MUJOCO_GL=egl .venv/bin/python scripts/solo_drill_render.py deliver \
    --npz data/drill/M_D28c_feasible_L3_seed0.npz    # log: /tmp/motion/deliver3.log
# tests
MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_drill.py -q
```

## 7.1 The failure sample (kept, per the evidence protocol)

The first `deliver` attempt used the 0.30 m stance and **fell at t = 22.9 s after 2 steps**
(`data/drill/L2_MOTION_feasible_L3_seed0.{json,npz}`, same programme and parameters as the
0.30 m cell of the table). It is the second sample of a marginal envelope, not a different
configuration: with a 0.06 m-wide foot and a ±0.2618 rad ankle-roll limit the step chain has no
margin to spare, which is exactly why the deliverable run is verified from its own trace (falls,
step count, margin, phase advance) rather than assumed from one clean run. It is kept and
labelled a failure; its clip would be `*_failure` if rendered.

## 8. What is NOT achieved (explicit)

* **Cadence**: 13.5 s/step in the delivered stance (4.0 s/step at a 0.21 m base). The brief's
  1.5–2.5 s needs a dynamic (capture-point) gait; the quasi-static weight-shift primitive cannot
  brake the CoM fast enough on a 0.06 m-wide foot.
* **Width**: the operator's 0.491 m stance is *holdable but not steppable*; the delivered
  clip narrows it to 0.30 m. The 0.495 m stance remains the L1 hold rung.
* **Support-foot edge roll (D2a) and the yaw pivot (D2d) are rejected**, with numbers; the
  torso lean (D2b) is kept; the step-out (D2c) is measured but not used.
* **D4**: the video-derived tracks were not repaired and no per-track table was produced.
* **A7**: the worst in-run CoM margin in the delivered clip is −0.023 m (the body briefly rides
  the hull boundary during a landing). Reported, not hidden.
* The rubric **B4** score is 1, i.e. the clip does not meet the ship gate on cadence; it is
  labelled a rung-L2 motion clip, and the acceptance name `final_continuous_drill.mp4` is still
  not written.
