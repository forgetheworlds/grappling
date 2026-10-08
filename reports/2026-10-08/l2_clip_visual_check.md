# L2 MOTION clip — independent frame-level visual check

**Checker:** L2VisualCheck · **Date:** 2026-10-08
**Subject:** `videos/solo_drill/final_L2_motion.mp4` (960×720, 30 fps, 2100 frames, 70.000 s, h264/yuv420p, decode-clean)
**Secondary:** `videos/solo_drill/L2_motion_slowmo_step_quarter.mp4`, contact sheets, operator reference stills
**Claims checked:** `reports/2026-10-08/drill_motion.md`, `data/solo_drill/final_L2_motion.json`

## Method (all measurements made from the rendered pixels)

* `ffmpeg` frame extraction at 0.5 s / 2 s / 5 s over the whole clip (contact tiles), plus dense
  0.05–0.1 s sampling inside every claimed step window.
* Shoe/mask tracking: grayscale thresholding of the floor band (y 480–590), connected-component
  labelling of the two shoes, bbox + IoU registration between frames (numpy/scipy).
* In-frame HUD reads (foot load, sole clearance, CoM margin, skill, phase) by cropping the HUD text
  rows (line bands measured at y = 8/26/44/62/80/98/116 px) and reading them as images.
* Frame-to-frame difference statistics over the robot region for pop/teleport detection.
* Cross-checks against `data/drill/M_D28c_feasible_L3_seed0.npz` (`clearance`, `foot_load`,
  `foot_xy`, `sole_pts`, `margin`) **only after** the pixel evidence was obtained.
* `tesseract` was unusable on this host (65–85 s per HUD image), so HUD numbers were read from
  upscaled crops instead of OCR'd.

---

## Q1 — Is there visible, unambiguous motion? Step count.

**YES. The clip is not static, and I count 5 steps — exactly the claimed 5.**

### The 5 steps I can see (my own observation, timestamps from the pixels)

| # | side | airborne window (visible) | plant | in-frame clearance read (HUD) | foot travel (px, ≈210 px/m) |
|---|------|---------------------------|-------|-------------------------------|------------------------------|
| 1 | left (far foot) | 15.40 – 15.76 s | 15.90 s | `+3.3 / +0.1 cm` @15.50, `+2.7 / +0.1` @15.70, `+0.1 / +0.1` @15.95 | 495→511 px x (+16 px ≈ 8 cm) |
| 2 | right (near foot) | 28.80 – 29.20 s | 29.34 s | `+0.1 / +2.6` @28.90, `+0.1 / +2.9` @29.10, `+0.1 / +0.2` @29.50 | 431→453 px x (+22 px ≈ 11 cm) |
| 3 | left | 45.40 – 45.76 s | 45.90 s | `+3.3 / +0.1` @45.50, `+2.5 / +0.1` @45.70, `+0.1 / +0.1` @45.95 | 513→525 px x (+12–14 px) |
| 4 | right | 50.22 – 50.56 s | 50.72 s | `+0.1 / +3.2` @50.30, `+0.1 / +1.8` @50.50, `+0.1 / +0.2` @50.75 | 445→468 px x (+23 px ≈ 11 cm) |
| 5 | left | 63.98 – 64.34 s | 64.48 s | `+3.0 / +0.1` @64.10, `+1.9 / +0.1` @64.30, `+0.1 / +0.1` @64.50 | 523→542 px x (+19 px) |

Evidence per step, in order of strength:

1. **The foot *re-positions and stays re-positioned*.** The stepping shoe's silhouette jumps
   16–23 px (≈ 8–11 cm) inside 0.4–0.5 s and is then static (±1–2 px) until its next step.
   This is unmistakable frame-by-frame and is what distinguishes this clip from L1.
   Mid-swing samples show the shoe *retreating* first (e.g. 15.50 s: x0 480 vs 495 planted) and
   landing forward (16.0 s: x0 509) — an arc, not a slide.
2. **The HUD printed in the frame states the clearance** (`sole clearance L/R`) for every one of the
   5 episodes: 3.3, 3.2, 3.3, 3.2, 3.0 cm peak, each returning to `+0.1 cm` within 0.4 s.
3. **Weight transfer is visible in the HUD foot loads**: e.g. 45.55 s `L/R 0/364 N` (left foot fully
   unloaded, single support), 45.76 s `64/297`, 45.90 s `180/147`; 12.25 s `45/282 N`.
4. **Largest frame-to-frame differences in the whole clip occur exactly at the 5 swings**
   (15.67–15.73, 29.03–29.10, 45.67–45.73, 50.40–50.60, 64.27–64.30 s), everywhere else the robot
   region changes slowly (mean |Δ| 0.41 grey levels, no value above 5).

**Airborne lift height is small:** 3.2–3.6 cm ≈ 6–7 px vertically at this camera. The white gap
between shoe and shadow grows only ~1 px while airborne (5 px → 6 px under the sole), so the *daylight
under the foot* is not unambiguous on its own — the lift is read from the foot's arc plus the HUD
number, not from a big shadow gap. What is unambiguous is the step: the foot changes plant position
by 8–11 cm each time.

**Cadence:** inter-lift intervals are 13.4, 16.6, 4.8, 13.8 s → mean **12.15 s**, matching the claimed
**12.1 s/step**. Total airborne time ≈ 1.8 s of 70 s (2.6 %).

**Not static (the L1 failure mode is absent):** both feet reposition (left ≈ +34 px total, right
≈ +38 px over the clip), pelvis height varies 0.689–0.762 m (HUD: 0.691 m @15 s vs 0.757 m @0 s),
skill advances STANCE → SHUFFLE_F (~10 s) → SHUFFLE_B (~45.9 s). Quiet stretches exist
(17.5–24.5, 33–37.5, 55–59.5, 67–70 s — HUD diff at baseline) and match the bundle's
`zero_motion_frac 0.219`, i.e. ~22 % of the clip is settle/hold; the remaining ~78 % shows body,
weight and foot motion.

---

## Q2 — Does it read as a wrestling stance, and how does it compare to the reference?

**Yes, on carriage; visibly narrower on base — the declared deviation is real and visible.**

* At every sampled time the robot is **staggered** (one foot clearly forward of the other),
  **hips/knees down** (pelvis 0.69–0.76 m, torso tilt 12.9–16.7°), **hands low and forward** with
  elbows bent in front of the hips, **head up**. It reads as a bladed, crouched ready stance.
* Side-by-side with `data/references/yt_gBAhX5t-GW4/frames/02_stance_t34s.png` and
  `03_stalking_t108s.png`: the operator's base is well outside shoulder width; the clip's feet sit
  nearly under the hips. Rough, scale-free estimate: clip ≈ 1.4 shoulder-widths of base vs reference
  ≈ 2.1 → about **60 % of the reference base**, consistent with the declared 0.28–0.30 m vs 0.49 m
  (61 %). It **does** look meaningfully narrower; it does not look *wrong* or off-balance.
* **Discrepancy:** every HUD frame prints `stance 0.30 x 0.10 m wide (reference is 0.49 m wide:
  narrowed for steppability)`, while the report and bundle say the delivered stance is **0.28 m**
  (config case `w=0.28`). The HUD says 0.30; the report says 0.28 and also "0.28–0.30" elsewhere.

---

## Q3 — Artifacts

| item | finding |
|---|---|
| **Floor/mat visible?** | Yes — white floor plane with a horizon, and contact shadows are ON (shoes cast a clear grey shadow). Foot-to-floor contact is judgeable. |
| **Foot sliding while loaded** | **Found, but small.** The support foot creeps while planted and loaded: shoe mask registration gives IoU 0.34 @0 px → 0.81 at a rigid −8 px shift for the left (far) foot between 46.5 s and 60.0 s; both shoe edges move together (+7 / +8 px), i.e. a translation, not a rotation. The trace agrees (`sole_pts` centre drift **22.1 mm** 46.5→51.5 s, **34.7 mm** 46.5→60.0 s, with `zmin` constant ≈ 4.5 mm, so it never leaves the ground). Other windows are static (17→27 s: 1 px; 31→44 s: 0 px; 52→63 s: 1 px, trace 1.4–7.5 mm). **Contradicts** `loaded_slip_m: 0.0` (all 5 steps), `max_load_drift_m: 0.0065`, and rubric B1 "loaded-foot slip ≤ 0.021 m". It is not noticeable at normal viewing speed (≤ 8 px over 13 s). |
| **Foot penetration / floating** | No floating (sole sits on the floor for both feet except during the 5 lifts). No *visible* penetration; trace `min_foot_clearance −3.2 mm` and `penetration_min −5.7 mm` are sub-pixel here. |
| **Mesh interpenetration** | None visible; feet never come close (trace min inter-foot separation 0.2266 m, consistent with the pixels). |
| **Popping / glitching** | None. Max frame-to-frame difference in the robot region is 4.0 grey levels (mean 0.41); **no frame exceeds 5**, i.e. no teleports, no mesh pops, no dropped-body glitches. |
| **Limb impossibility** | None seen in any sampled frame; knees/ankles stay plausible, hands stay in front of the hips. |
| **HUD legibility** | Text is legible at native resolution (I read every field). Two problems: (a) the **last HUD line is clipped at the right frame edge** (text runs to x = 959, ending mid-word `… scripted feedback, one r…`); (b) there is **no `step N/M` counter anywhere** in the 7 HUD lines. |
| **Do the phase/step counters advance?** | **`phase` does not.** It reads `-` at every one of 14 samples spanning 0–65 s. What advances is `skill`/`cmd` (STANCE → SHUFFLE_F → SHUFFLE_B), which matches `phase_advance_count: 2` only if "phase" means the skill block. The report's claim of "scheduler phase (advances per block *and* per step)" and "the `step N/M` counter" is **not supported by the frames**. |

---

## Q4 — The 0.25× slow-motion clip

`videos/solo_drill/L2_motion_slowmo_step_quarter.mp4`: 480×360, **120 fps, 1081 frames, 9.008 s**.
(The file was still being written at 09:27 — first probe failed with "moov atom not found"; it
completed at 09:36 and is valid now.)

* **It is not slow motion.** Reading the baked HUD clock along the clip: playback 0.00 → sim
  14.30 s, 1.50 → 15.80, 3.00 → 17.30, 4.50 → 18.80, 6.00 → 20.30, 7.50 → 21.80, 9.00 → 23.30.
  Sim time advances **9.0 s over 9.0 s of video = 1.0×**, and the frames are all distinct (no
  4× frame duplication — consecutive-frame diffs are small but non-zero throughout).
* **It does contain one complete step cycle**: the source window 14.30–23.30 s contains step 1's
  shift tail (14.3–15.36), lift (15.36–15.52), swing (→15.68) and plant (15.90). But that cycle
  lasts 0.5 s of a 9 s clip — **≈6 % of the file**; the other ~8.5 s is the 9 s settle/hold.
  So the deliverable "0.25× slow-motion of a step cycle" (rubric H3) does not hold as stated:
  it is real-time and 94 % static.

---

## Q5 — Other contradictions with the bundle / report

1. **`phase` never advances and there is no `step N/M` counter** (report §D5/§5, rubric F2).
2. **Slow-motion is 1.0×, not 0.25×** (rubric H3 = 2).
3. **HUD prints `stance 0.30 x 0.10 m`; report/bundle say 0.28 m.**
4. **Rubric B1 says "6 completed steps"**; the run/bundle/my count say **5**.
5. **Worst in-run margin is reported as both −0.0257 (§3.2, metrics) and −0.023 (rubric A7, §8).**
   Visible in-frame: `CoM margin +0.009 @45.55`, **`−0.008 @45.65`, `−0.022 @45.72`**, `+0.057 @45.76`,
   `+0.079 @45.80` — so the negative-margin episode is real and on screen (trace min −0.0257 @45.76 s,
   within the claimed five 0.06–0.14 s episodes), but the exact worst tick/frame I can read is −0.022.
6. **A red HUD warning `!! safety blend alpha=0.07 (emergency response active)` flashes twice**,
   at **15.77–15.83 s and 45.77–45.83 s** (2 frames each; detected by red-pixel scan, 4 of 2100
   frames). It is absent from the report's narrative (the bundle does record
   `safety_blend_ticks: 11`). Both flashes coincide with a landing and with a negative-margin tick.
7. **Support-foot creep 22–35 mm** vs claimed `loaded_slip_m 0.0` / `max_load_drift_m 0.0065`
   / rubric B1 ≤ 0.021 m (see Q3).
8. **Report §3 text**: "the deliverable clip (95 s × 6 steps) therefore shows *six* clearly visible
   steps" describes a different run; the delivered artifact is 70 s / 5 steps.
9. **What holds:** 0 falls (robot upright in every sampled frame, no frame where it is on the mat),
   0 resets (no cut or teleport — max frame diff 4.0), one continuous 70 s episode, 5 steps,
   12.1 s mean cadence, shadowed floor, no inter-foot collision.

---

## Q6 — What I could NOT determine from the frames

* Whether the *sub-pixel* loaded slip is zero: my pixel noise floor is ±1–2 px (≈ 0.5–1 cm), so I can
  only bound visible slip, not confirm the millimetre numbers in either direction beyond the case
  in Q3.
* Foot **penetration** below the floor (−3 to −6 mm) — far below one pixel; the shadow/silhouette
  never shows it.
* Whether the CoM-margin sign is correct — the HUD prints a number, but I cannot verify the
  underlying CoM/hull computation from pixels; I can only confirm the sign flips to negative
  visibly at 45.65–45.72 s.
* Whether the quiet 6–9 s stretches are pure settle or contain undetectable refusals/retries: the
  HUD clock keeps running and the loads/clearances are steady, but a retry with no motion would be
  visually identical to a settle.
* The clipped tail of the last HUD line (it runs off the right edge) — its full wording (and
  therefore whether it mentions a reset/step counter) is unrecoverable from this render.
* Whether the operator's 0.49 m stance *would* be steppable with a different controller (out of
  scope of a render check).

---

## Verdict

**The claim "a 70 s continuous drill with repeated visible steps in a wrestling stance, 0 falls" →
MATCHES**: 70 s continuous, **5 visible steps** at 15.4, 28.8, 45.4, 50.2, 64.0 s (mean 12.15 s
cadence), staggered hips-down hands-forward stance, 0 falls, 0 resets.
Caveats (they contradict *other* claims in the bundle, not this sentence): the lift is only
3.2–3.6 cm so the airborne moment is subtle; `phase`/step counters do not advance and no `step N/M`
counter exists; the "0.25×" slow-mo plays at 1.0×; HUD says 0.30 m not 0.28 m; rubric B1 says 6
steps; and a loaded support foot creeps 22–35 mm against a claimed ≤ 6.5 mm.
