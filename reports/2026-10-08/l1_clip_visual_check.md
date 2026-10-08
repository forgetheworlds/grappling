# Frame-level visual check of `videos/solo_drill/final_L1_90s.mp4`

**Checker:** L1VisualCheck · **Date:** 2026-10-08 · **Rubric:** `docs/QUALITY_RUBRIC.md` H (visual match) + A (stance)
**Scope:** rendered content only. Codec/length (h264, 960×720, 30 fps, 2700 frames, 90.000 s) taken as given.

## Method

* Extracted all 2700 frames (low-res) and 90 frames at 1 Hz into `/tmp/l1chk/`; sampled the 5
  repo frames `data/solo_drill/frames/l1_t{3,20,45,70,88}.png`; built a 3×3 pose grid
  (t = 0,10,…,80), a feet zoom (t = 3,12,30,60,89) and a torso/hip/hand zoom (t = 12,45).
* Measured pixel landmarks over time (head top, pelvis blob, foot silhouettes, silhouette bbox,
  hand tip), scanned all 2700 frames for black/flat/pop frames and for cuts.
* Read the HUD by OCR (`tesseract`) at t = 0,1,3,4,5,7,9,10,11,12,15,20,25,40,45,60,80 s.
  Note: the video's HUD clock prints **video time + 0.47 s** (video t=5.0 → HUD `t=5.47s`).
* Tested `03_side_by_side_reference.png` provenance by panel→clip matching (MAE over the robot
  region) and by OCR of the HUD text inside each panel.

---

## Q1 — Does the stance READ as a wrestling stance?

**Partly. It reads as a staggered, moderately crouched *ready* stance; the three cues that make a
stance read as *wrestling* are all shallow.**

| cue | frame evidence | read |
|---|---|---|
| staggered feet | foot silhouettes at x 412–475 (rear, y≈566) and x 534–588 (lead, y≈537); centre separation ≈117 px (≈0.4 m at the robot's on-screen scale) in every sampled frame (t=3,20,45,70,88) | **yes, clearly staggered** |
| wide base | HUD/repo geometry 0.495–0.519 m lateral; camera is a near-side 3/4 view, so the 0.50 m lateral axis is partly foreshortened — visible, but the dominant on-screen separation is the 0.24 m stagger | visible but weak |
| hips down | pelvis z 0.687–0.730 m (HUD), knee z 0.34–0.35 m; hips only 6–11 cm below the stand keyframe; knee flexion 33–43° (report) | **moderate athletic crouch, not a wrestling squat** — the operator's hips sit at ~75 % of standing height |
| torso pitched forward | HUD tilt 14.0° (t=1) → 15.5° (t=60); visually a slight lean; operator ≈35–45° (visual estimate off `02_stance_t54s`, hip→shoulder) | **too upright to read as a shot-ready wrestling stance** |
| hands low and forward | t=12/45 torso zoom: hands carried forward at abdomen height, elbows ≈90°, forearms horizontal, **fingers spread open** | reads as a generic "arms out ready" guard, not the operator's low hanging arms, not a collar-tie height either |
| head up | helmet is the topmost point in all 90 s (head top y = 224–235 full-res); visor axis slightly downward-tilted | up, acceptable; no eyes to read |

Net: **PARTIALLY** — a human viewer would read "athletic staggered stance", and would need to be
told it is a *wrestling* stance.

## Q2 — Element-by-element vs `data/references/yt_gBAhX5t-GW4/frames/02_stance_*.png`

| element | verdict | reason |
|---|---|---|
| width | **MATCHES** | 0.495–0.519 m measured vs 0.491 m G1-scaled reference; visually comparable base width |
| depth (stagger) | **APPROXIMATES** | robot 0.241–0.248 m vs *measured* reference 0.062 m (4× deeper) — yet the reference frames show a clearly staggered operator. Not an error: deeper stagger matches the operator's verbal "leg a bit back" and is safer; the numeric reference and the visual reference disagree with each other |
| crouch | **DIFFERS** | operator ≈75 % standing hip height, thighs near horizontal, hand near the floor in the level-change frame; robot travels only 4.3 cm (0.687→0.730) and holds 0.677–0.730. **Mostly morphology/constraint-driven** (CoP must stay inside four 5 mm spheres per foot, ankle roll ±0.26 rad — `reports/2026-10-08/support_envelope.md`), **but partly a choice**: the report states a deeper crouch (0.644 m, margin +0.030) was reachable and was "kept shallow for margin" |
| torso pitch | **DIFFERS** | 14.0–15.5° vs ≈35–45°. **Not purely morphology:** the balance law explicitly adds "torso-upright rows", i.e. the controller actively holds the torso upright. Presenting this as an unavoidable morphology gap (drill.md §0/§6.2) overstates it |
| hand carriage | **DIFFERS** | operator's arms hang low with hands near thigh/knee height and elbows nearly straight; robot carries hands forward at abdomen height with elbows at 90°. drill.md itself calls this fixable (`StanceSpec.hand_up`) — see the internal contradiction below |
| head | **APPROXIMATES** | both have the head above the pelvis with eyes forward; the G1 has a featureless helmet whose visor axis looks slightly downward |
| fingers | **note** | drill.md §6.2 says "no articulated fingers, rubber pads". The rendered frames (hand zoom, t=20) show **finger geometry: a thumb plus four fingers**. Articulation cannot be told from stills, but the claim as written is contradicted by the render |

**Report inconsistency found:** drill.md §3 + §0 and `docs/VISUALS.md` row 10b say the hands are
"at hip height / lowered to hip height" (0.83 m in the pelvis frame), while drill.md §6.2 says they
are "forward at chest height (0.30 m forward, 1.06 m up)" and lists the fix as outstanding. Both
cannot be true; the frames show an abdomen-height carriage, i.e. between the two.

## Q3 — Is the motion visible across the 90 s?

**Only in the first ~8 s. From t≈9 to t=89 the clip is visually a static hold.**

| t (video, s) | head-top y (480×360 trace) | HUD pelvis z | HUD skill / phase |
|---|---|---|---|
| 0 | 113–116 | 0.714 | LEVEL_CHANGE / l1_crouch_2 |
| 1–6 | 116–118 (lowest) | 0.692 (t=1), **0.687 (t=3)**, 0.689 (t=5) | LEVEL_CHANGE / l1_crouch_2 |
| 7–8 | 112–114 (risen) | 0.722 (t=7) | RECOVER / l1_crouch_2 |
| 9–25 | 112–113 | 0.727, 0.728, 0.727, 0.728, 0.727, 0.729 | RECOVER / **l1_crouch_2 (never advances)** |
| 40 / 60 / 80 / 88 | 112–113 | 0.728 / 0.729 / 0.728 / 0.729 | RECOVER / l1_crouch_2 |

* Total visible vertical excursion = **12 px full-res ≈ 4.6 cm, all of it inside t=0–8**; over the
  remaining **82 s the head moves ≤1 px (≤0.8 cm)** at 30 fps — sampled every frame, not just at 1 Hz.
* Feet never move: rear foot x 412–477 / y 566–567, lead foot x 534–588 / y 536–539, **±3 px over
  the whole 90 s** (compression noise) — no step, and no slide.
* Silhouette bbox changes: x 412→417 and 587→589 (≤5 px ≈ 2 cm lateral), top y 224–235.
* Hand tip x: 575–587 (12 px ≈ 4.6 cm fore-aft).
* Real but nearly invisible weight shift: HUD foot load L/R oscillates 142–179 N per leg
  (154/173 @t=10, 179/148 @t=15, 159/169 @t=25, 176/150 @t=80) — a ~±5 % body-weight transfer with
  ≤4 px of visible sway.
* Per-second robot-region pixel diff is non-zero everywhere (20–68 gray levels) — the robot is never
  literally frozen — but the 3×3 pose grid at 10 s spacing (t=0…80) is visually near-identical.

**Consequence:** the claims "repeated level changes"/"18 completed programme cycles"
(drill.md §0, §6 F5) and "`repeated crouch/rise level changes`" (`docs/VISUALS.md` row 10b) are
**not supported by this clip**: exactly one crouch-and-rise occurs, and the HUD phase label stays
`l1_crouch_2` with `skill RECOVER` for the last ~82 s (consistent with drill.md §6.1's timeout note,
but it reads on screen as a stalled programme).

## Q4 — Artifacts

| item | result |
|---|---|
| black/blank frames | **none** — all 2700 frames: mean gray 185.8–186.0, 0 frames with mean < 30, 0 flat frames (std < 5) |
| glitch/pop frames | **none found** — frame-to-frame mean abs diff mean 0.11, p99 0.27, max 0.93 (max at t=10.0, coincides with HUD digit rollover) |
| cuts / resets / teleports | **none** — background column probe differs from frame 0 by ≤1 gray level over the whole clip; no jump in any landmark |
| foot sliding while loaded (B1) | **not observed** — feet pixel-stationary ±3 px for 90 s. Note L1 never lifts a foot, so this is a hold, not a step test |
| feet sunk / floating | **cannot be determined from the frames** — there is **no floor plane, horizon line or contact shadow anywhere in the image** (column probe at x=100: monotonic 225→253 gradient, largest horizontal edge is the HUD band at row 176). The robot stands in a white void; only the HUD's `sole clearance +0.1/+0.1 cm` speaks to contact |
| mesh clipping / interpenetration | none obvious in zooms at t=3, 12, 45 (hip, knee, wrist, hand); a black hip-cover shell sits flush against the thigh — ambiguous, **≤2 cm interpenetration cannot be excluded or confirmed** from 960×720 stills with no annotated frames |
| impossible poses | none seen — knees bend forward, elbows bend normally, ankles neutral |
| HUD legibility | metric lines readable (≈11 px cap height) but the **bottom-right status block is clipped at the frame edge**: text reaches x=958 at rows 105–134 in every sampled frame and OCR yields `… the rung-L1 clip: one unbroken e` (cut mid-word). HUD occupies 176/720 = 24 % of frame height |
| camera framing | robot bbox x 412–589, y 224–567 for the entire run — **always fully in frame**, never occluded by the HUD (robot top 224 > HUD bottom 176), static camera, robot ≈48 % of frame height, roughly centred |

## Q5 — Is `03_side_by_side_reference.png` fair and correctly aligned?

**Aligned in intent, misleading in fact.**

1. **The panels are not from the delivered clip.** The two right-hand panels are byte-identical
   (MAE 0.00) to `data/solo_drill/frames/frame_t005.00.png` and `frame_t012.00.png`. Their own HUD
   text reads:
   * top: `t= 5.00s … skill RECOVER … pelvis 0.716 m, tilt 16.6°, load 165/162 N`
   * bottom: `t= 12.00s … skill LEVEL_CHANGE … pelvis 0.677 m, tilt 16.8°, load 209/117 N`
   The delivered clip says the opposite at those times: `t=5.47 → LEVEL_CHANGE, 0.689 m, 14.5°,
   158/168 N` and `t=12.47 → RECOVER, 0.727 m, 15.2°`.
2. **Pixel matching confirms it:** in the robot region the "stance t=5.0" panel best matches clip
   **t=8** (MAE 23.8 vs 29.7 at t=5) and the "crouch bottom t=12.0" panel best matches clip
   **t=2** (25.2 vs 34.0 at t=12). In the delivered clip t=5 is the crouch and t=12 is the tall
   stance — the sheet's phase/time labels are inverted relative to the clip it is supposed to evidence.
3. **Not to a common scale / not the same viewpoint:** left = wide oblique camera shot of the
   operator in a gym; right = a 0.5× near-profile frame of the robot in a white void, HUD still drawn
   but illegible at that size. No scale reference; the figures are not normalised to body height,
   so "MATCHES on … width" cannot be checked from this sheet.
4. **Apples-to-oranges phase in row 2:** the operator's level-change frame is a deep squat with the
   hand at the floor; the robot's is a 4.3 cm dip. The caption ("both drop the hips with the torso
   staying tall") reads as if the two are the same movement.
5. **All three green captions are cut off at the right edge** (text runs to x=1119): the verdicts
   literally end mid-sentence (`… DIFFERS on`, `… the`, `… frames from the cached L1 trace`).

---

## Could NOT be determined from the frames

1. Foot–floor contact quality (sinking, floating, penetration) — the render has **no floor line,
   mat, or shadow**; only the HUD clearance number is available.
2. True 3-D stance dimensions from the picture (0.50 m width, 0.24 m depth, CoM margin, 2 cm
   support margin) — one static near-side camera, no scale bar, no second view.
3. Whether the fingers articulate (single view; fingers never visibly move).
4. Whether ≤2 cm mesh interpenetration occurs anywhere (occluded/ambiguous at 960×720; no
   annotated per-element frames shipped).
5. Whether the metrics JSON / rubric scores (18 cycles, pelvis-z range 4.4 cm, C1/C5/F5 = 3)
   describe **this** mp4 — the side-by-side's panels, taken from the same run tag, do not match it,
   so the run identity of the clip is unverified.
6. "Watching" behaviour (jitter, ease, sense of flow) — analysis is frame-based; the numeric
   magnitudes (≤4 px sway) suggest the motion is subtle, but temporal smoothness at normal
   playback speed was not judged.
7. H3 slow-motion review: `02_slowmo_level_change_quarter_speed.mp4` (3.03 s, 91 frames) exists but
   was not inspected — it is a separate artifact from this clip.

## Verdict

**PARTIALLY MATCHES** — a clean, artifact-free, continuous 90 s episode of one G1 in a readable
staggered stance with **no** black frames, pops, cuts, foot slide or framing problems, but the stance
reads shallow/generic rather than wrestling (upright 15° torso, 4 cm crouch, abdomen-height spread
hands), the clip shows **one** level change in the first 8 s and then 82 s of near-static hold
(contradicting "repeated level changes"), the HUD status line is clipped, and the side-by-side
comparison uses frames from a different timeline than the clip it labels.
