# Solo drill L2 — margin-gated stepping + reference-trajectory tracking

**Agent:** DrillStep2 · **Date:** 2026-10-08
**Territory:** `src/drill/stepping.py`, `src/drill/tracking.py` (new),
`tests/test_drill.py`, `scripts/solo_drill_render.py` (two new modes),
`videos/solo_drill/**`, `data/solo_drill/**`, this report.
**Assignment:** D1 — implement the §7.1 diagnosis (CoM-margin lift gate, swing-leg
compliance, support-foot yaw pivot, load-gated landing) and prove it on
(i) the isolated step, (ii) the entry walk, (iii) one step per L1 cycle,
(iv) a 2–3 step shuffle. D2 — a reference-trajectory tracking mode that drives
the plan from `data/refs_video/*.npz` and reports partial tracking.

## 0. Bottom line

| claim | evidence |
|---|---|
| **The §7.1 fix is implemented and measured**: the lift gate is the *measured CoM margin inside the support foot's own hull* (≥ 0.02 m, dwell + lock), the swing leg is compliant (roll follows contact, no roll authority), the support foot yaw-pivots toe-out, and the landing stays load-gated | `lift_gate()` in `src/drill/stepping.py`; regression test `test_lift_gate_is_margin_not_load` fails if the gate reverts to load-only |
| **The isolated step is clean**: gate fires at margin **+0.024…+0.027 m**, CoM travel 0.086–0.098 m, support ankle −0.156…−0.183 rad (limit 0.2618), swing load 35–62 N, lift clearance 2.8–3.0 cm, placement error 0.0018 m, landing load 198–220 N | `data/drill/L2_ENTRY_SPEC_*.json`, clip `videos/solo_drill/L2_isolated_step.mp4` |
| **The entry walk no longer falls**: 14 s, **falls 0**, the first step completes, every step the geometry forbids is *refused with the required travel in the event* (0.189–0.209 m vs the 0.14 m cap) | `L2_ENTRY_SPEC` (falls 0, steps 1, refusals 9) |
| **In a base the primitive can use** (0.21 × 0.06 m): a **4-step shuffle** runs 34 s with **falls 0**, slip 0.0083 m, margin min −0.0017 m; the L1-cycle programme completes **3 steps** before one fall at 22.6 s | `L2_SHUFFLE`, `L2_CYCLE` in `data/drill/l2_suite_summary.json` |
| **The 0.495 m drill stance is not steppable on this robot** (the measured finding): every lift there needs **0.254–0.276 m** of lateral CoM travel against a **measured authority of 0.14 m**; the primitive refuses and the robot stays up (falls 0) | `L2_STANCE_REFUSED` (falls 0, steps 0, 7 refusals, required 0.254–0.276 m) |
| **L1 is unchanged** (regression check): 20 s at the drill stance, falls 0, **margin min +0.0295 m** (identical to the shipped L1 number), slip 0.0000 m, IK residual 0.0005 m | `/tmp/probe_narrow.py l1` log; the L1 clip is untouched |
| **D2 mode implemented, partial results**: `TrackController` loads a track, drives the plan from it (FK sole targets + the reference's own joints), caps its own authority where the reference would topple the robot (margin-driven clock; support-envelope clamp by *leaning*; base kept within 5 cm of the body), and reports fidelity/deviation. The video-retargeted references themselves are **not holdable** (frame-0 CoM margins −0.40…+0.07 m): probed runs topple in 0.5–1.5 s **even with the clamp**. The 8-track table is **not delivered** (budget) — see §4. | `src/drill/tracking.py`, tests `test_track_load_and_interpolation`, `test_track_mode_caps_authority_and_reports_partial_tracking` (19 tests pass) |

## 1. D1 — what changed, and why (each rule with the measurement behind it)

The §7.1 diagnosis is implemented literally. Around the gate, eight further rules
were forced by measurements made on this host during this session — each one is a
fall that was observed, not a preference:

| # | rule | what it fixes (measured) |
|---|---|---|
| 1 | **margin gate** (`lift_gate`, ≥ 0.02 m inside the support hull) | the old load gate fired at 51 N with the CoM 0.03 m *outside* the support hull; the swing ankle absorbed the residual to saturation |
| 2 | **gate dwell + lock** (2 ticks + 0.06 s) | the CoM lags the plan ~0.1 s: lifting at the first clearing tick puts the foot in the air while the body is still travelling (measured: the CoM then coasts out of the 0.06 m-wide support) |
| 3 | **support-foot yaw pivot** (toe-out ≤ 0.5 rad at 0.35 rad/s, about the measured footprint centre) | buys CoM *margin* without moving the body (live: the gate fires at 0.086–0.10 m of CoM travel) |
| 4 | **swing-foot roll compliance** (roll follows the measured sole tilt; ±0.18 rad band; relief 1.7) | the stretched swing leg otherwise drives its ankle roll to ±0.2618; measured relief: +0.15 rad of sole roll = +0.09 rad of ankle roll (live swing ankle ≈ +0.07…+0.21) |
| 5 | **support-side roll authority mask** (`BalanceLaw.offsets(roll_sides=…)`) | while a foot steps only the support leg may push sideways; with both pushing, the transfer stalls and the swing ankle does the work |
| 6 | **reference honesty** (`_track_body`, lead ≤ 0.04 m) | leg IK solves for the *plan's* base: a plan that disagrees with the body places feet wrongly (measured: a landed foot floating 1.5–12 cm, the support hull collapsing to one foot, then the safety blend toppled the robot) |
| 7 | **shift drop** (0.22 m/m + a clearance-triggered extra ≤ 0.045 m) | the stretched swing leg pulls its foot off the mat at a 0.10–0.13 m shift unless the pelvis sinks |
| 8 | **smooth min-jerk shift** (0.03 m/s nominal, zero end velocity) | a velocity-limited ramp arrives *with* momentum: the CoM coasted ~0.06 m past the plan and out of the support (the law's capture lookahead is 3 mm of CoM here) |
| 9 | **geometry refusal** (`reach_cap` = 0.14 m, `step_refused` event with `required_com_travel_m`) | the failure mode becomes a *named refusal*, not a topple |
| 10 | **settle after landing** (lead-based recentre, 0.02 m exit) | the landing's residual CoM velocity otherwise carries the body over the far foot; an absolute-plan recentre was worse (it diverged from the body and floated the landed foot 12 cm) |

## 2. The measured limit of the primitive (the L2 finding)

* **required** CoM travel to satisfy the gate for one lift (computed with the
  pivot): **0.111 m** in the stand's first step (delivered; the gate fires at
  0.086–0.10 m); **0.14–0.165 m** in a 0.25 m-wide base (marginal — one run in
  two still toppled on the residual drift after a landing); **0.254–0.276 m** in
  the 0.495 m drill stance (needs > the cap: refused).
* **authority** (CoM travel before the body runs away from the plan):
  **~0.13–0.14 m** — the cap, with the provenance in `StepParams.reach_cap`.
* physical reason: `reports/2026-10-08/support_envelope.md` — the CoP must stay
  inside four 5 mm spheres per foot; the foot is 6 cm wide; the ankle roll limit
  is ±0.2618 rad. A wide stance is *holdable* (L1: +0.0295 m worst) but not
  *steppable*.
* **the stepping rungs are therefore demonstrated in `stepping.stepping_base_spec()`:
  0.21 × 0.06 m, both feet toed out 0.10 rad.** The drill stance is unchanged and
  still the L1 hold base. This is the quantified deviation the assignment asks for.

## 3. D1 evidence (`data/drill/l2_suite_summary.json`; clips rendered from their own cached traces, `ffprobe`-verified)

| stage | base | falls | longest | steps | refusals (required m) | CoM margin min | slip | IK res |
|---|---|---|---|---|---|---|---|---|
| L2_ENTRY_SPEC (i)+(ii): stand → **drill** stance | 0.495 m | **0** | 14.0 s | 1 | 9 (0.189, 0.209) | **+0.0137** | 0.0058 | 0.0013 |
| L2_ENTRY_BASE (ii): stand → stepping base | 0.21 m | 1 @ 9.6 s | 9.6 s | 1 | 1 (0.153) | −0.675 | 0.0097 | 0.0014 |
| L2_SHUFFLE (iv): 4-step shuffle | 0.21 m | **0** | **34.0 s** | **4** | 0 | −0.0017 | 0.0083 | 0.0003 |
| L2_CYCLE (iii): L1 cycle + 1 step | 0.21 m | 1 @ 22.6 s | 22.6 s | 3 | 0 | −0.644 | 0.0066 | 0.0013 |
| L2_STANCE_REFUSED: the drill stance refuses | 0.495 m | **0** | 10.0 s | 0 | **7 (0.254–0.276)** | +0.0387 | 0.0000 | 0.0003 |
| L1 (regression, drill stance) | 0.495 m | **0** | 20.0 s | — | — | **+0.0295** | 0.0000 | 0.0005 |

Clips (`videos/solo_drill/`, each with a 3-frame sheet and a
`data/solo_drill/<name>.json` evidence bundle carrying provenance, metrics,
rubric table, `verify_clip` block and the reproduce command):

| clip | what to look for |
|---|---|
| `L2_isolated_step.mp4` (960×720, 6.6 s) | one step out of the stand: the shift, the pivot, the margin-gated lift, the world-tracked swing, the load-gated landing |
| `final_L2_entry_walk.mp4` (960×720, 15.6 s) | the entry walk in the base the primitive can use |
| `L2_cycle_step_diag.mp4` (0.5×, 10 s) | level change + one gate-checked step per cycle |
| `L2_shuffle_3steps_diag.mp4` (0.5×, 14 s) | consecutive single-foot repositions with the settle between |
| `L2_stance_step_refused_diag.mp4` (0.5×, 7 s) | the same primitive in the drill stance: refusals instead of a topple |

## 4. D2 — reference-trajectory tracking (`src/drill/tracking.py`)

Implemented and tested; **the per-track table was not completed** (budget — the
8-track run was prepared as `scripts/solo_drill_render.py tracks` and did not
execute). What is in place and measured:

* `Track` (load/interp of `qpos_a (T,36)`, `t`, `technique`, `meta`), and
  `TrackController` (a `FeasibleDrill` whose plan is the reference frame):
  FK sole targets per foot, **the reference's own joints as the servo targets**
  (`ReferenceSolver.direct` — re-solving a retargeted pose through the cartesian
  IK measured 3.5–6 cm of leg error for ~20 ticks and a collapse), the measured
  balance law on top;
* **authority caps, each reported**: (1) the reference clock
  (`ref_rate_min + (1-ref_rate_min)·a`, `a` from the measured CoM margin);
  (2) a **support-envelope clamp** that *leans* the pose (base moves, foot
  targets stay, the reference's legs are re-solved by IK) until the reference's
  CoM is 0.015 m inside the *measured* hull (`clamp_shift_*`); (3) base honesty
  (`base_follow_err_*`, plan kept ≤ 0.05 m from the body);
* `Track.start_qpos` leans a non-holdable frame 0 into support (measured
  frame-0 margins: −0.396…+0.073 m; two tracks start with no foot on the mat);
* `runner` hooks `controller="track"` / `start="track"` and
  `metrics["reference_tracking"] = ctrl.tracking_report()`;
* **the measured finding so far**: with the clamp + the lean, probe runs of
  `stance_widen_step` still topple in 0.9–1.2 s (the commanded posture is the
  operator's forward lean; the G1's 6 cm-wide feet cannot carry it), i.e. the
  retargeted references need *geometry* adjustment, not only a CoM clamp. The
  per-track numbers (falls, longest, weighted joint/site error, margin min,
  ankle-roll peak, rubric element scores, clips) remain to be produced by the
  `tracks` command.

## 5. What I did NOT achieve (explicit)

* **No step in the 0.495 m drill stance** — measured impossibility on this robot
  (0.254–0.276 m required vs 0.14 m authority). The stepping rungs run in a
  0.21 m-wide base, clearly labelled as a deviation; changing that needs the
  stance (or the foot/ankle geometry) to change.
* **Two of the five stage runs fall in the base at the envelope's edge**
  (0.21 m): L2_CYCLE at 22.6 s after three completed steps and L2_ENTRY_BASE at
  9.6 s (its second entry step was refused — required 0.153 m — and the run then
  fell; the *same* entry at a 0.25 m base ran 25 s clean, so this is the margin
  handover after a refusal, not the shift itself). The 4-step shuffle (34 s) and
  both drill-stance runs are clean. Reported, not hidden.
* **The D2 per-track table and its clips are missing** (§4) — only the mode, its
  two tests and the probes exist.
* **Step cadence is slow**: ~5–7 s per step cycle (3–4 s shift at the measured
  0.03 m/s lateral ceiling + 1.5 s lift/swing/plant + 1–2 s settle).
* No learning anywhere; the controller remains scripted feedback.
* `ENTRY_STEP_M` (0.14 m, DrillDirector's constant) is untouched: refusals are
  emitted by the primitive, the caller is not changed.
