# Env rule videos — frame-level visual verification (2026-10-08)

**Agent:** FrameCheck · **Method:** Read-only inspection of rendered content (not codec
metadata). Read each `videos/env/<name>_sheet.png` contact sheet, extracted 5–9 additional
frames per clip with ffmpeg to `/tmp` (times 0.1–5.9 s + targeted event times), inspected
every image, plus whole-video checks: full decode (`ffmpeg -f null -`, no errors),
`-count_frames` (152/167/182/167/182 — matches report), `blackdetect` (zero black segments in
all five clips), HUD text-pixel presence on 49 sampled frames (all present, no blank/low-text
frames), frame luminance stable ≈190 (no dark/blank frames). All images were inspected via
self-labeled copies (filename burned into the pixels) to guarantee file↔image identity.

## Per-clip verdicts

### 1. `draw_stance` — **MATCHES**
- Sheet start t=0.00: both robots in engaged crouch, exchange #0 clock 0.00/1.00 s,
  score 0:0, `dorsal=0/0`, rA=rB=0.49 — frames t=0.10 (clock 0.10/1.00) confirm both
  standing/engaged, arms mutually held.
- Timeout/draw cycle visible: t=1.00 banner `EXCHANGE 0 END — cause: timeout — no score
  (draw) — reset`, both reset standing; t=2.00 banner `EXCHANGE 1 END`; t=3.50 exchange #3
  clock 0.50/1.00 + banner `EXCHANGE 2 END`; t=4.50 exchange #4 + banner `EXCHANGE 3 END`
  (HUD/pose at 3.50 and 4.50 identical → bit-identical cycles, as report claims).
- Clock visibly runs 0.10→0.50 of each 1.0 s window; score 0:0 in every inspected frame;
  back detector `dorsal=0` in every frame; only timeout banners, never a back banner.
- Artifacts: hands/forearms of the two robots overlap in a tight cluster during engagement
  (crop `draw_1.0`) — possible finger/hand-level clipping, unconfirmable at 960×720 with
  shadows off. No sinking, no detached limbs, no framing cuts, HUD complete.

### 2. `takedown_back_event` — **MATCHES**
- t=0.10: attacker `a` mid-dive (bent forward, left leg airborne, tilt 56.9°), `b` behind;
  both fully in frame. t=0.70: entangled mid-sprawl (`a` horizontal tilt 97.4°).
- t=1.40: **`b`'s back on the mat** — HUD `b dorsal=1, pelvis_z=0.081`, `a` lying across
  (`dorsal=0`, limb=1), score still 0:0 (persistence window running). Sheet mid t=1.60:
  `b dorsal=1, trig=-` (window still open) — matches report's mid-panel intent.
- t=1.80: banner `EXCHANGE 0 END — cause: back — a scores, b loses — reset`, both standing
  reset, score **+1.00 : −1.00**. t=2.20 / 4.50: standing, score held — reset after score ✓.
- Artifacts: (a) both robots end up on the mat (attacker topples too) — visible at 0.70/1.40,
  disclosed in report §2 as phase-2 limitation; (b) heavy body overlap in the grapple region
  at 0.70/1.40 (3× crops: `a` hip / `b` foot and arm/torso silhouettes occupy the same image
  region) — **cannot conclusively distinguish occlusion from mesh interpenetration** from a
  single camera angle; no gross penetration, no detached limbs, no sinking, HUD complete.

### 3. `nonterminal_knees_hands` — **MATCHES**
- Down-robot `a` on the mat with limb contact while `b` stands, **no reset during the
  phase**: exchange #0 clock observed advancing 0.70 → 0.90 → 1.60 → 2.50 → 3.04 (sheet) →
  3.50 → 4.50, same exchange the whole time, score frozen 0:0, `dorsal=0` every frame,
  no banner until t=5.00 (`EXCHANGE 0 END — cause: timeout — no score (draw) — reset`),
  then both standing (t=5.00/5.60, exchange #1).
- Clearest hands/knees frames: **t=2.50** (`a` on right knee + planted hand, `b` standing
  1.05 m away, `limb=[L_knee,R_knee]`), **t=4.50** (`a` on both wrists, torso horizontal,
  `limb=[L_wrist_yaw,R_wrist_yaw]`); t=0.90/3.50 prone on hands/knees with one leg raised.
- Artifacts: `a`'s poses read as a *toppled collapse* (torso horizontal or past it, legs
  flailing in the air at 0.90/1.60/3.50) rather than a poised kneel — report itself
  discloses tilt 105–125° and the missing recovery controller (§3 deviation). Not supine:
  belly/head down, back plate up, consistent with `dorsal=0`. `b` upright (tilt ≤1.4°)
  throughout. No sinking, detachment, framing cuts, missing HUD or black frames.
- Note: the report's assignment wording "…while it recovers to standing" is explicitly
  marked not achievable (§3 deviation); the video shows the draw reset instead — as
  disclosed.

### 4. `ambiguous_simultaneous` — **MATCHES**
- **Both robots supine at the same time**: sheet start t=0.00 and frames t=0.10 / 0.30 show
  both lying on their backs, `dorsal=1/1`, `trig=-/-`, tilt ≈91.8°, rA=rB=0.90, score 0:0 —
  the pre-trigger persistence state.
- t=0.50: banner `EXCHANGE 0 END — cause: back, AMBIGUOUS (both backs inside 0.10 s) — no
  score — reset`, **both standing again, score still 0:0**; t=0.70 banner persists
  (exchange #1 clock 0.38/20.00); t=1.60 / 3.00 standing, clock advancing. No score for
  either wrestler ✓.
- Artifacts: supine robots hold their arms bent stiffly *in the air* (env holds joints with
  no controllers — mannequin-like, minor); the near robot's feet approach the ring posts but
  stay inside (r=0.90 + body length ≈ ring, no visible intersection). No sinking, no
  clipping between the two bodies (1.8 m apart), HUD complete.

### 5. `oob_forfeit` — **MATCHES**
- Offender outside the yellow ring confirmed visually: **t=1.48 / 1.58** — `a` stands with
  both feet clearly beyond the near ring posts (posts pass behind its calves), HUD
  `rA=1.78, OOB A 1/3, score −0.25`; sheet mid t=1.60 same; t=2.10 returning (`rA=1.11`);
  t=2.50 inside hold (`rA=0.35`); t=3.30 `rA=1.48, 1/3` and t=5.30 `rA=1.48, 2/3` —
  each sampled one 0.02 s control step *before* the report's event times 3.32/5.32,
  consistent with them.
- t=5.80: banner `EXCHANGE 0 END — cause: oob (3 events) — b wins by forfeit — reset`,
  score **−1.75 : +1.00**, OOB counter reset to 0/3, both robots standing inside the ring ✓.
- Artifacts: **the offender glides across the mat with feet planted in a fixed stance**
  (identical foot pose at t=1.30/1.48/1.58/2.10 while its position shifts ~0.3–1.4 m — no
  stepping animation, slight lean wobble 1.5°–17.4°): the scripted root motion, disclosed in
  report §5/§6.4, reads as ice-skating. Both robots stay upright (a ≤17.4°, b ≤0.2°), no
  toppling, no sinking, no detachment. At maximum excursion `a`'s feet come within ~45 px of
  the frame bottom but are never cut off.

## Global checks (all five clips)
- No black/blank/frozen-corrupt frames (`blackdetect d=0.01, pix_th=0.15`: zero segments).
- Full decode with no errors; frame counts exactly 152/167/182/167/182 (= report §0).
- HUD overlay (clock, score, back-detector line, mat radius, rA/rB, OOB counters) present in
  all 49 sampled frames; red end-of-exchange banners legible in every banner frame inspected.
- Both robots fully inside the frame in every inspected frame; no robot floating or sunk
  (all floor contacts sit on the plane); gravity/Z-up orientation correct throughout.
- Contact sheets are correctly named (each `*_sheet.png` shows its own scenario caption +
  HUD) and their mid-panel timings match the report (takedown 1.60, ambiguous 0.28, OOB 1.60).

## Could not be determined from frames
1. **Mesh interpenetration vs. occlusion** in the takedown grapple (t=0.70/1.40): silhouettes
   overlap tightly at the `a`-hip/`b`-foot and arm/torso junctions; one camera angle cannot
   prove clipping, and there is no gross penetration.
2. **Finger-level clipping** during draw_stance arm engagement: overlapping hand geometry is
   visible but indeterminate at 960×720 with shadows off.
3. **Exact event-instant frames** (OOB crossings at 1.32/3.32/5.32 s; back trigger 1.70 s):
   adjacent frames bracket them consistently (r=1.44→1.78 across the first crossing; counter
   0/3→1/3→2/3→3/3), but the precise 0.02 s boundary frame was not directly viewed.
4. **Detector ground truth** (dorsal/limb flags): HUD self-reports are visually plausible
   (prone postures with `dorsal=0`, supine with `dorsal=1`) but contact truth cannot be
   measured from pixels.
5. **Report wording discrepancy (not a video artifact):** report §0 says the yellow ring was
   added "for the OOB clip" only, but the ring is visible in **all five** clips (and §6.6
   itself says far ring posts may be cropped "for the other clips"). The visuals are fine;
   the §0 sentence is inaccurate.

## Verdict summary
| clip | verdict |
|---|---|
| `draw_stance` | **MATCHES** |
| `takedown_back_event` | **MATCHES** (with disclosed attacker-topple; interpenetration unconfirmed) |
| `nonterminal_knees_hands` | **MATCHES** (down robot reads as toppled collapse, disclosed in report) |
| `ambiguous_simultaneous` | **MATCHES** |
| `oob_forfeit` | **MATCHES** (scripted glide/skating artifact, disclosed as scripted root motion) |
