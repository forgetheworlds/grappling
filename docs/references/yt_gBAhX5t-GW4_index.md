# yt_gBAhX5t-GW4_index.md — reference-video pose capture, transitions and retarget

Method + evidence index for the operator reference video
(`data/references/yt_gBAhX5t-GW4/ref720h264.mp4`, H.264 1280×720, 823.5 s;
"Stance & Motion 101", Footwork Trainer / Isaac J. Knable). The video itself is
gitignored; everything here is derived from it. Written 2026-10-08.

## 0. Status summary (honest)

| # | item | state |
|---|---|---|
| 1 | pose estimation (CPU) over demonstration takes of all 10 chapters | **DONE** — mediapipe PoseLandmarker, 15 fps effective, 12 360 frames, 19 + 20 misses flagged, never interpolated |
| 2 | take detection + transition segmentation from the signals | **DONE** — take inventory + 12 selected transitions (`derived/transitions.json`), detected boundaries vs author chapter times below |
| 3 | retarget to G1 via `src/retarget` | **DONE** — `data/refs_video/*.npz` (12 transitions, 50 Hz); measured stretches 1.00–2.31 (see §8) |
| 4 | verification videos (source overlays + G1 rollouts + side-by-side) | **PENDING** — renderer written and validated (`render_refs_video.py`; `draw_pose_overlay.py --mode mp4`; `compose_compare.py`); first render run is the next step |
| 5 | numeric spec byproduct | **DONE** — `derived/stance_spec.json` (41 params, observed/inferred, frame evidence) |

The one known quality caveat: the retarget pipeline needed a fix that was found
late (base-rotation representation flips inflated the kinematic time-stretch by
3–13×). The fix (continuous rotation-vector filtering + reconstructed root
translation) is in place and measured: stance-hold stretch 3.39 → **1.00**
(duration 4.40 s = source duration). All npz on disk were regenerated with the
fixed tool.

## 1. Pose estimation

| item | value |
|---|---|
| estimator | `mediapipe` 1.1.0 (`mediapipe.tasks.python.vision.PoseLandmarker`), CPU-only |
| model | `pose_landmarker_full.task` (float16) |
| source | `ref720h264.mp4` (H.264 720p; the AV1 `ref720.mp4` is not used by libraries) |
| sampling | 15 fps effective (every 2nd source frame at 30 fps), single process |
| measured speed | **13.1 fps wall** pass 1 (6 401 frames, 492 s), **14.0 fps wall** pass 2 (5 961 frames, 430 s) on this 4-core ARM box under load |
| coverage | pass 1: ch2 STANCE 26–83 s, ch5 LEVEL CHANGE 202–261 s, ch7 SHOTS 400–549 s, ch10 KNEE SPRAWL 662–823.5 s; pass 2: ch1 INTRO, ch3 STALKING, ch4 CIRCLING, ch6 FAKE, ch8 DOWNBLOCK, ch9 PEPSI |
| detections | pass 1: 6 382/6 401 (19 missing); pass 2: 5 939/5 959 (20 missing) — stored as NaN with `detected=False`, **never interpolated** |
| stored | `t`, `img_lm` (F,33,3 normalized), `world_lm` (F,33,3 m, hip-centred), `visibility`, `presence`, `detected`, `conf`, `chapter`, `segment` |
| index | `pose/index.json`, `pose_pass2/index.json` (provenance, counts, chapter map) |

Alternatives tried: only mediapipe was needed (wheel `mediapipe-1.1.0-py3-none-manylinux_2_28_aarch64` exists; ultralytics/tflite not required). MediaPipe Tasks API note: `import mediapipe` first, then `from mediapipe.tasks.python import vision`.

**Sanity evidence (skeleton on the coach, not drifting):**
`pose/overlays/overlay_t0030.00.png` (STANCE), `t0260.30` (LEVEL CHANGE),
`t0425.60`, `t0430.80` (SHOTS), `t0684.30`, `t0689.50` (KNEE SPRAWL);
`pose_pass2/overlays/overlay_t0122.50.png` (STALKING), `t0178.50` (CIRCLING).
All eight were inspected: skeleton on the coach, legs/torso correct; hands are
the noisiest landmarks (expected).

## 2. Demonstration takes (detected from the signals, not the chapter clock)

Take = ≥3.5 s of full-body visibility (visibility ≥ 0.5 on 13 key landmarks),
lower-body motion above 0.25 m/s, gaps ≤ 1 s merged. Full tables:
`derived/analysis/take_inventory_priority.txt`, `take_inventory_rest.txt`.

| chapter | author window | takes found | selected | why |
|---|---|---|---|---|
| 2 STANCE | 26–83 s | 25.6–35.7, 59.9–66.8, 71.2–82.0 | **25.6–35.7**, hold sub-window 30.4–34.8 | highest conf (0.967), feet planted (0 foot events), pelvis travel 0.07 m net; alternates kept for width variability |
| 5 LEVEL CHANGE | 202–261 s | 201.5–223.4, 256.2–261.4 | **256.2–261.4** | deepest drop (pelvis 0.85→0.48 m); the 201–223 take is shallower (0.69 m) |
| 7 SHOTS | 400–549 s | 399.5–416.9, 430.6–442.8, 455.1–488.1, 516.8–549.4 | **430.6–442.8** + entry window 422.8–431.0 | only take with knee-to-mat contact (knee min 0.0 m, 4.1 s near the mat); the entry starts in the preceding quiet stretch that the activity detector split off |
| 10 KNEE SPRAWL | 662–823.5 s | 7 takes | **683.4–690.2** (entry+hold) and **779.8–782.2** (recovery); 769.5–772.5 second repetition | cleanest knee contact; the 691–703 s stretch is excluded (landmark glitches, speeds 8–12 m/s); 732.5–749.8 and 766.1–771.0 kept as alternates |
| 3 STALKING | 83–133 s | 82.5–94.5, 112.4–131.6 | **112.4–131.6**, step window 122.0–131.6 | longer take, conf 0.969; first 1.5 s is a walk-in |
| 4 CIRCLING | 133–202 s | 132.5–158.4, 177.6–190.0, 191.9–202.4 | **177.6–190.0** | highest conf without NaN frames |
| 1 INTRO | 0–26 s | 0–26 (conf 0.979) | not used | explanation/stand |
| 6 FAKE | 261–400 s | 3 takes (260.5–311.2, 312.7–351.6, 355.3–400.4) | not used this pass | later selection work |
| 8 DOWNBLOCK | 549–599 s | 548.5–590.6 (crouch 12.9 s, knee near mat 0.87 s) | not used this pass | defense, later phase |
| 9 PEPSI | 599–662 s | 3 takes | not used this pass | operator-specific drill |

Take-to-take variability (stance width, ch2): take 25.6–35.7 → **0.566 m**,
take 72.5–81.0 → 0.612 m (Δ 0.046 m ≈ 8 %), i.e. the coach's own repetition
spread is ≈ ±0.05 m — the tolerance band used in the curriculum addendum.

Detected vs author boundaries: detected takes agree with the author chapter
windows except (a) the SHOTS entry starting ~7 s before the detected take
(quiet, controlled step — selected by content review) and (b) the KNEE SPRAWL
recovery crossing the chapter's own take boundary (holds are low-motion and
end takes). Both are recorded, not silently patched.

## 3. Selected transitions (retargeted)

`derived/transitions.json` (12 windows; `data/refs_video/<name>.npz`):

| name | window (s) | what it contains |
|---|---|---|
| stance_hold | 30.4–34.8 | planted wide stance, still (0 step events) |
| stance_widen_step | 27.6–30.4 | narrow → wide stance, rear foot steps back-and-out |
| level_change_full | 259.7–261.2 | stand → deep crouch → stand (pelvis 0.82→0.59→0.82 m) |
| level_change_fast | 256.9–258.0 | quicker drop-and-rise repetition (0.82→0.54→0.77 m) |
| shot_entry_full | 422.8–431.0 | crouched stance → step → penetration → knee on mat |
| shot_recover | 432.5–437.0 | knee-off → rise to crouched stance |
| knee_sprawl_entry | 683.4–686.2 | stand → drop to the knee |
| knee_sprawl_hold | 685.8–690.2 | knee on the mat, torso up |
| knee_sprawl_entry2 | 769.5–772.5 | second repetition of the drop |
| knee_sprawl_recover | 779.8–782.2 | rise from the knee to a crouch |
| stalk_shuffle | 122.0–131.6 | stalking shuffle steps |
| circle_step | 178.0–190.0 | circling steps |

## 4. Numeric spec (byproduct)

`derived/stance_spec.json` — 41 parameters, each with `value`, `unit`,
`status` (observed / inferred / unknown), `evidence` (window + frames) and a
confidence note. Headline values (metres are mediapipe-metric; ratios are
scale-free; G1-equivalents = ratio × 1.32 m):

| quantity | value | status |
|---|---|---|
| stance width | 0.566 m = 0.372 × body height → **0.49 m G1** | inferred (ratio observed) |
| stance depth | 0.062 m (feet level) | inferred |
| stance pelvis height | 0.508 × standing → 0.67 m G1 | inferred (ratio observed) |
| knee flexion / torso pitch | 51° / 47° | observed |
| head above pelvis | 0.31 × body height | observed |
| hands (wrists rel. pelvis) | (+0.20, ∓0.15, −0.04) × body height | observed |
| level-change drop | 0.154 × body height → 0.20 m G1; 1.5 s full cycle | inferred (ratio observed) |
| shot: lead-foot offset at knee-down | 0.05 × body height (rear foot −0.30 ×) | inferred (depth axis) |
| shot: knee min height / hold | 0.0 m (mat) / 2.5 s | observed |
| shot: recovery | 4.5 s window | observed |
| knee sprawl: drop / hold / recovery | 0.5 s / 4.4 s / 2.4 s | observed |
| stalking cadence / step | 2.6 events/s, 0.16 m | inferred (low conf) |
| circling cadence / step | 2.2 events/s, 0.17 m | inferred (low conf) |

Metric-scale note recorded in the spec: mediapipe's monocular scale reads the
coach at ~1.52 m standing; if his true height is 1.78 m, multiply raw metres by
1.17. Ratios and G1-equivalents are unaffected.

## 5. Cross-check against our existing references (read-only)

`derived/refs_crosscheck.json` (computed with `compare_refs.py`, G1 landmark
sites, `data/refs/*.npz` never modified):

| parameter | our STANCE.npz | video (G1-equivalent) | verdict | source to drive T3/T5 |
|---|---|---|---|---|
| stance width | 0.263 m | **0.49 m** | **CONTRADICTS** — ours is ~1.9× narrower | **video** (operator rule agrees: widen) |
| stance depth | 0.150 m | 0.05–0.09 m | CONTRADICTS (ours staggered, video square) | video for the taught square stance; GrappleMap keeps the staggered shot stance |
| pelvis height | 0.806–0.807 m (≈ standing) | **0.67 m** | **CONTRADICTS** — ours is barely crouched | **video** |
| knee flexion | (site knee 0.360 m above floor) | 51° | ADDS — video gives the crouch the reference lacks | video |
| level change | DOUBLE_LEG pelvis z 0.333–0.754 | drop 0.20 m G1, 1.5 s | **ADDS** — GrappleMap DOUBLE_LEG goes deeper but the video supplies duration/tempo and a shallower drill variant | both: GrappleMap for the technique chain, video for the tempo/depth band |
| shot: knee-to-mat | DOUBLE_LEG knee min 0.064 m (ends near mat) | 0.0 m, hold 2.5 s | **CONFIRMS** | GrappleMap (chain) + video (contact duration) |
| shot: penetration stance width/depth | 0.312–0.438 m / 0.585 m | 0.63 m / 0.61 m | CONTRADICTS width, CONFIRMS depth | video width (widen), GrappleMap depth |

Reasoning: GrappleMap remains the *technique* source (chain structure, role
assignment, the two-robot context); the video is the *geometry/tempo* source
(stance width/height, level-change depth and duration, knee-contact duration).
The operator's two-axis rule is the tie-breaker whenever the merged geometry is
not statically holdable — and both our references are known not to be (STANCE
PD-replay topples ~1.4 s; my PD-tracked render of `data/refs/STANCE.npz` in the
single-G1 scene reproduced this: pelvis min 0.151 m, tilt max 93°, fall flag).

## 6. Reliability: 2D-reliable vs 3D-approximate

- **Reliable (observed):** image-plane geometry — ankle separation projected on
  the hip line (stance width), foot-repositioning events, knee height reaching
  the mat, torso pitch, head-above-pelvis, timing/durations, confidence flags.
- **Approximate (inferred):** monocular depth (front-back separations, lead-foot
  forward offset, rear-foot offset); absolute metric scale (ratios are the
  primary quantities); global translation (reconstructed by foot-contact
  anchoring — mediapipe world landmarks are hip-centred and contain no
  translation at all).
- **Not measurable here (unknown):** contact forces, hip yaw / rotation about
  the vertical (weakly observable), finger/grip detail, which leg the coach
  *intends* to lead (it varies between repetitions).
- **Known limitations:** 15 fps effective sampling (contact instants ±0.07 s);
  the ankle landmark height oscillates ±5 cm even when planted, so steps and
  drags cannot be separated (cadence is "foot-repositioning events");
  camera assumed static (no pan/zoom observed); regions with landmark glitches
  (ch10 691–703 s, 748–750 s) were excluded and are flagged in the take tables.

## 7. Tools + regeneration (all under `derived/tools/`, run from repo root)

| tool | purpose |
|---|---|
| `run_pose.py` | pose pass over a video/segments → `pose*/landmarks.npz` + `index.json` |
| `analyze_pose.py` | signals, take detection (lower-body activity), phase labels |
| `take_inventory.py` | per-take content inventory (knee/crouch/step counts) |
| `measure_takes.py` | per-take metrics + phases + transitions |
| `metrics_spec.py` | builds `derived/stance_spec.json` |
| `retarget_video.py` | landmark track → G1 via `src/retarget` (read-only) → `data/refs_video/*.npz` |
| `compare_refs.py` | video spec vs `data/refs/*.npz` cross-check |
| `draw_pose_overlay.py` | skeleton overlay stills / mp4s |
| `render_refs_video.py` | G1 PD-tracked rollout mp4 + contact sheet + hud json |
| `compose_compare.py` | source|G1 side-by-side + live height traces |
| `step_stats.py`, `trace_take.py` | footwork stats; numeric window traces |

Regenerate everything (in order):

```bash
V=/home/ubuntu/venvs/pose/bin/python   # mediapipe venv (outside the repo)
$V data/references/yt_gBAhX5t-GW4/derived/tools/run_pose.py --model full --fps 15 \
   --segments "26:83,202:261,400:549,662:823.5"
$V data/references/yt_gBAhX5t-GW4/derived/tools/run_pose.py --model full --fps 15 \
   --segments "0:26,83:133,133:202,261:400,549:599,599:662" \
   --out data/references/yt_gBAhX5t-GW4/pose_pass2
.venv/bin/python data/references/yt_gBAhX5t-GW4/derived/tools/metrics_spec.py
.venv/bin/python data/references/yt_gBAhX5t-GW4/derived/tools/retarget_video.py \
   --npz data/references/yt_gBAhX5t-GW4/pose/landmarks.npz \
   --spec data/references/yt_gBAhX5t-GW4/derived/transitions.json \
   --out-dir data/refs_video
MUJOCO_GL=egl .venv/bin/python data/references/yt_gBAhX5t-GW4/derived/tools/render_refs_video.py
```

Retarget output contract (single robot, documented because it differs from
`data/refs/*.npz`): `qpos_a (T,36)` f64 at 50 Hz, `t (T,)`, `technique` (str),
`edges` (empty — video-sourced), `landmark_rms` (weighted site RMS), `meta`
(JSON: source window, scale, facing, floor offset, net travel, gap fraction,
keyframe count, kinematic stretch, rms detail, ambiguities). `qpos_b` is zeros
and is NOT meaningful (the existing two-robot solver was reused by passing the
same target track in both slots and keeping robot A).

## 8. Retarget quality (final run, 12/12 transitions)

Measured on the stance-hold window (worst case, a *static hold* where landmark
jitter dominates): raw solved keyframes need a 3.39× kinematic stretch; with
the 2.5 Hz zero-phase low-pass + continuous-rotation-vector base filtering the
stretch is **1.00**. The root translation is reconstructed from foot-contact
anchoring (world landmarks are hip-centred), so a penetration step actually
travels instead of running on a treadmill.

| transition | source s | output s | stretch | rms_w (m) |
|---|---|---|---|---|
| stance_hold | 4.4 | 4.40 | 1.000 | 0.047 |
| stance_widen_step | 2.8 | 3.60 | 1.288 | 0.046 |
| level_change_full | 1.5 | 2.08 | 1.647 | 0.054 |
| level_change_fast | 1.1 | 1.34 | 1.260 | 0.055 |
| shot_entry_full | 8.2 | 8.20 | 1.000 | 0.046 |
| shot_recover | 4.5 | 7.96 | 1.786 | 0.049 |
| knee_sprawl_entry | 2.8 | 3.36 | 1.200 | 0.045 |
| knee_sprawl_hold | 4.4 | 4.40 | 1.000 | 0.042 |
| knee_sprawl_entry2 | 3.0 | 6.74 | 2.304 | 0.061 |
| knee_sprawl_recover | 2.4 | 5.54 | 2.314 | 0.068 |
| stalk_shuffle | 9.6 | 16.82 | 1.752 | 0.049 |
| circle_step | 12.0 | 17.36 | 1.447 | 0.048 |

Stretch > 1 means the existing kinematic-limit retimer slowed the reference to
respect the G1's joint velocity/acceleration limits (V_MAX 6 rad/s, A_MAX
60 rad/s²). The largest values (2.3× on the two explosive rise/recovery windows)
are partly real morphology (a 1.32 m robot with slower joints than a human) and
partly residual landmark noise — report them as such when comparing to the
video (rubric H2). RMS site fit is 4.2–6.8 cm everywhere, i.e. the pose shapes
match the targets; the render step (`render_refs_video.py`) adds the PD-tracked
rollout metrics (joint error, pelvis height, tilt, fall flag) that say whether
the G1 can *hold* each transition at all.

## 9. Stills for review

`docs/references/stills/` — 14 named stills (chapter + timestamp, 640×360 JPEG,
504 KB total): 8 skeleton-overlay frames (tracking sanity) and 6 raw key frames
(28.4, 259.9, 440.5, 442.3, 686.5, 692.5 s). G1-side stills will be added with
the verification videos (render step, §0 item 4).
