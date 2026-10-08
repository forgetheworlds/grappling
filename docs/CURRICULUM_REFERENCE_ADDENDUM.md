# CURRICULUM_REFERENCE_ADDENDUM.md — reference-video geometry for T3/T5/T6

Additive section (2026-10-08). Source: the operator's reference video
(`data/references/yt_gBAhX5t-GW4/ref720h264.mp4`, coach Isaac J. Knable,
"Stance & Motion 101"), pose-estimated with mediapipe PoseLandmarker and
retargeted to the G1 through `src/retarget` (see
`docs/references/yt_gBAhX5t-GW4_index.md` for the full method, take selection
and evidence). Numbers below come from
`data/references/yt_gBAhX5t-GW4/derived/stance_spec.json`; every value there
carries observed/inferred/unknown status and frame evidence.

This document does **not** change `docs/MOTOR_CURRICULUM.md` or
`docs/SOLO_DRILL.md`; it supplies the reference geometry those tasks need, in
the units and interfaces they already use.

## 0. What the video contributes (one paragraph)

The coach teaches a **wide, crouched, square-ish stance** (feet level, lateral
separation ≈ 37 % of body height, pelvis at ≈ 51 % of standing height, knees
bent ≈ 50°), then a **level change** (pelvis drops ≈ 15 % of body height in
≈ 0.5 s), then a **penetration step into a knee-on-mat shot** (lead foot steps
forward, the knee reaches the mat plane, hold ≈ 2.5 s, rise ≈ 4.5 s), and a
**knee-sprawl / knee-down drill** (drop to the knee in ≈ 0.5 s, hold ≈ 4.4 s,
recover in ≈ 2.4 s). Footwork drills (stance widening, stalking shuffle,
circling) supply step cadence and step length.

## 1. T3 — stance footwork: reference-derived targets

| parameter | video value (coach, ratio) | G1-equivalent target | status | evidence |
|---|---|---|---|---|
| stance width (lateral) | 0.372 × body height | **0.49 m** | inferred (ratio observed) | frames 30.4–34.8 s, ch2 stance hold |
| stance depth (front–back) | 0.041 × body height (feet level) | 0.054 m | inferred (depth axis least reliable) | same window |
| pelvis height | 0.508 × standing | **0.67 m** (standing G1 ≈ 0.79 m) | inferred (ratio observed) | same window |
| knee flexion | 51° | 51° | observed | same window |
| torso pitch (pelvis→head vs vertical) | 47° | 47° | observed | same window |
| head above pelvis | 0.31 × body height | 0.41 m | observed | same window |
| hand carriage (wrists rel. pelvis, fwd/left/up) | left (+0.20, −0.14, −0.04) × body height; right (+0.20, +0.15, −0.04) | left (+0.26, −0.18, −0.05) m; right (+0.26, +0.20, −0.05) m | observed | same window |
| widening step (narrow → wide) | width 0.26 → 0.67 m in ≈ 1.6 s; the moving foot goes back-and-out (fwd +0.20, lat −0.38 m) | step ≈ 0.43 m total | observed | frames 27.6–30.4 s |
| stalking cadence / step length | 2.6 events/s, 0.16 m median | same | inferred (low conf) | ch3 122.0–131.6 s |
| circling cadence / step length | 2.2 events/s, 0.17 m median | same | inferred (low conf) | ch4 178.0–190.0 s |

**Contrast with our current STANCE reference** (`data/refs/STANCE.npz`,
computed read-only via the G1 landmark sites): width 0.263 m, depth 0.150 m,
pelvis z 0.806–0.807 m (i.e. barely crouched), knee min 0.360 m. The video
stance is **≈ 1.9× wider and ≈ 0.12 m lower** than our retargeted STANCE. The
operator's frontal rule ("falling sideways → legs too close together") points
the same way.

### Command/observation fields needed for T3

Already in the contract (`docs/SOLO_DRILL.md` §4): `(vx, vy, wz)`,
`stance_height_cmd`, `skill_id`. Extra parameters this spec needs:

- `stance_width_cmd` (m) — lateral foot separation target (default 0.49 m,
  range to sample 0.42–0.60 m; **never below 0.35 m**, the measured
  non-holdable narrow width).
- `stance_depth_cmd` (m) — front-to-back foot separation target
  (default 0.05 m for the square stance, 0.35–0.55 m for the staggered
  shot-entry stance).
- `lead_leg` already exists; the video's lead/trail assignment per repetition
  is recorded in the spec (the coach alternates).

### Measurable success metric (T3)

A stance is scored as reference-matching when, over a held 20 s:
`|width − 0.49| ≤ 0.06 m`, `|depth − 0.05| ≤ 0.05 m`,
`|pelvis z − 0.67| ≤ 0.05 m`, `|torso pitch − 47°| ≤ 10°`, knee flexion
`51° ± 10°`, no fall, and pelvis xy drift < 0.05 m. Footwork: step events with
median step length 0.15–0.25 m and no loaded-foot slide > 2 cm (rubric B1).

## 2. T5 — level change + penetration step: reference-derived targets

| parameter | video value | G1-equivalent target | status | evidence |
|---|---|---|---|---|
| level-change pelvis drop | 0.154 × body height (0.234 m coach-metric) | **0.20 m** | inferred (ratio observed) | ch5 259.7–261.2 s |
| level-change duration | drop ≈ 0.4–0.8 s; full down+up 1.5 s | same | observed | same window |
| fast repetition (second take) | drop 0.34 m coach-metric | 0.29 m | inferred | ch5 256.9–258.0 s |
| shot entry duration | 8.2 s from crouched stance to knee-down (includes a slow step + settle) | same | observed | ch7 422.8–431.0 s |
| lead-foot forward offset at entry | 0.05 × body height at the knee-down hold (≈ 0.07 m coach-metric) | ≈ 0.07 m | inferred (depth axis) | ch7 430.5–433.0 s |
| rear-foot offset at the knee-down hold | −0.30 × body height behind the pelvis | ≈ −0.40 m | inferred (depth axis) | same window |
| penetration stance (both feet planted, deepest split) | width 0.48 × body height, depth 0.46 × body height, pelvis 0.30 × standing | width 0.63 m, depth 0.61 m, pelvis 0.40 m | inferred | ch7 425.3–426.5 s |
| knee-to-mat | knee reaches the mat plane (0.0 m; landmark floor ≈ −0.04 m) | 0.05 m target (contact) | observed | ch7 428.6–433.0 s |
| knee-down hold | ≈ 2.5 s | same | observed | ch7 430.5–433.0 s |
| recovery (knee-off → crouched stance) | 4.5 s window; pelvis rises 0.32 → 0.59 m | same | observed | ch7 432.5–437.0 s |

**Contrast with our current DOUBLE_LEG reference** (`data/refs/DOUBLE_LEG.npz`):
pelvis z 0.333–0.754 m, width med/max 0.312/0.438 m, depth max 0.585 m, knee
min 0.064 m. Our reference already ends near the mat (knee 6 cm) but is
**narrower than the video's penetration stance** (0.44 m max vs 0.63 m
equivalent) — again consistent with the operator's frontal widening rule.

### Command/observation fields needed for T5

Already in the contract: `stance_height_cmd` (drives the drop),
`(vx, vy, wz)`, `skill_id`, `lead_leg`. Extra parameters:

- `level_depth_cmd` (m) — pelvis drop target (default 0.20 m; the drop is
  currently expressible only as an absolute `stance_height_cmd`, which loses
  the "drop relative to current stance" semantics).
- `penetration_depth_cmd` (m) — lead-foot forward offset at the entry
  (default 0.07 m at the knee-down hold; 0.35–0.45 m for the split
  penetration stance with both feet planted).
- `knee_contact_cmd` — target knee height during the contact phase
  (default 0.05 m) plus a minimum contact duration (≥ 1.5 s).
- Phase timing: entry ≤ 8 s, hold 2.5 s ± 0.5 s, recovery ≤ 4.5 s (the video's
  own tempo; our demo may compress the slow settle).

### Measurable success metric (T5)

Per shot attempt: pelvis drop `0.20 ± 0.05 m` reached within 1.0 s of the
command; lead-foot forward offset `0.07 ± 0.10 m` at contact; knee minimum
height `≤ 0.08 m` sustained `≥ 1.5 s`; return to within 10 % of the T3 stance
metrics within 4.5 s; no fall (pelvis never below 0.35 m without a commanded
knee-down phase, dorsal contact = 0).

## 3. T6 — knee sprawl / knee-down recovery: reference-derived targets

| parameter | video value | G1-equivalent target | status | evidence |
|---|---|---|---|---|
| drop to knee | ≈ 0.5 s (pelvis 0.68 → 0.24 m coach-metric) | ≤ 0.6 s | observed | ch10 683.4–686.2 s |
| knee height at contact | mat plane (0.0 m) | ≤ 0.08 m | observed | same window |
| pelvis height while sprawled | ≈ 0.16–0.20 × standing (0.30 m coach-metric) | ≈ 0.22–0.27 m | inferred | ch10 685.8–690.2 s |
| sprawl hold | 4.4 s | 2–4 s | observed | same window |
| recovery to stance | 2.4 s (pelvis 0.04 → 0.63 m coach-metric) | ≤ 3 s | observed | ch10 779.8–782.2 s |
| second repetition (variability) | drop 0.75 → 0.08 m coach-metric | — | inferred | ch10 769.5–772.5 s |

### Command/observation fields needed for T6

- `knee_target_height_cmd` (m) + `contact_hold_cmd` (s) — same field as T5's
  `knee_contact_cmd`, reused for the sprawl.
- `trail_leg_recover_cmd` — boolean/phase flag for the recovery segment, so
  the policy is not rewarded for parking the knee on the mat (SOLO_DRILL §5
  anti-exploit: "parking the knee on the mat" is explicitly listed).
- Observation: knee height and knee contact flag (already in the critic's
  contact set per the contract; make the knee contact explicit in the actor's
  phase-progress field for this skill).

### Measurable success metric (T6)

Knee-down phase: knee height ≤ 0.08 m within 0.6 s of the command, held
2–4 s, pelvis height 0.22–0.27 m, torso pitch ≤ 60°; recovery: rise to the T3
stance metrics within 3 s, no hand support required (hand-contact frames
counted and reported), dorsal contact = 0.

## 4. Tie-breaker rule (operator, unchanged)

When the video geometry is **not statically holdable** for our robot — and the
retargeted video references show exactly the same open-loop toppling our
GrappleMap references do (see the index doc's verification table) — adjust
along two axes **before** flattening the torso or raising the crouch:

1. **sagittal**: falling forward/backward → move the rear leg further back
   (the video's own rear-foot offset, −0.30 × body height, is the starting
   point);
2. **frontal**: falling sideways → widen the stance (the video's width
   already exceeds ours by ≈ 1.9×; the spec's `stance_width_cmd` default is
   the measured width, and its range is clipped at 0.35 m minimum).

Every trim is logged with frame index, axis, amount and the CoM-margin
improvement (SOLO_DRILL §2). The video supplies the *starting* geometry; the
rule supplies the *repair direction*; MuJoCo supplies the *verdict*.

## 5. Data pointers

| artifact | path |
|---|---|
| pose tracks (per-frame landmarks + confidences) | `data/references/yt_gBAhX5t-GW4/pose/landmarks.npz`, `pose_pass2/landmarks.npz` |
| numeric spec (this document's numbers) | `data/references/yt_gBAhX5t-GW4/derived/stance_spec.json` |
| retargeted G1 transitions (50 Hz, single robot) | `data/refs_video/*.npz` |
| G1 verification videos + overlays | `videos/refs_video/` |
| take/transition selection evidence | `data/references/yt_gBAhX5t-GW4/derived/analysis/` |
