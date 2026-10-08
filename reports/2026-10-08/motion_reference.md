# Motion Reference Engineer — report (G1 wrestling drill, dataset v1)

**Date:** 2026-10-08 · **Scope:** `reports/2026-10-08/briefs/agent1_motion_reference.md`
**Dataset:** `data/references/motion_refs/v1/` (format: `data/references/motion_refs/FORMAT.md`)
**Everything numeric below was measured by the named script on this host; nothing is
copied without re-checking against the artifact.**

## 1. What was produced

| artifact | path |
|---|---|
| clip index (deliverable 1) | `data/references/motion_refs/clip_index.json` + `scripts/build_clip_index.py` |
| fused movement spec (2) | `data/references/motion_refs/fusion_spec.json` + `scripts/build_fusion_spec.py` |
| 15 individual retargeted refs (3) | `data/references/motion_refs/v1/refs/*.npz` |
| composed drill + stance→stand (4) | `data/references/motion_refs/v1/drill_continuous.npz` (60.2 s, 23 phases), `v1/stance_rise.npz` |
| three-level feasibility (5) | `data/references/motion_refs/feasibility.json` + `scripts/feasibility_diagnosis.py` |
| dynamic probe videos + metrics (6) | `videos/motion_refs/dynamic_probe_{drill_continuous,shot_entry_full}.mp4` (+ sheets), `dynamic_probe_metrics.json`, `scripts/probe_drill_dynamic.py` |
| comparison videos (6) | `videos/motion_refs/drill_phases.mp4`, `source_vs_g1_{level_change_full,shot_entry_full}.mp4`, `key_postures.mp4` (+ sheets + metrics JSON), `scripts/render_motion_ref_videos.py` |
| format doc + loader demo (3, 10) | `data/references/motion_refs/FORMAT.md`, `scripts/query_motion_refs.py` |
| tests | `tests/test_motion_refs.py` (24 passed) |
| this report (7) | `reports/2026-10-08/motion_reference.md` |

## 2. Source provenance (acceptance 1) — settled explicitly

* **`tyU-QaV8MnI`** ("Stance and Motion Drills For Wrestling", The School of
  Wrestling, 149 s): the orchestrator's brief named it; it is NOWHERE in the old
  docs. It was FETCHED on 2026-10-08 through the documented authorized path
  (`yt-dlp --cookies-from-browser firefox:... --js-runtimes node`, the same
  recipe as `docs/REFERENCES.md`) to `data/references/yt_tyU-QaV8MnI/ref.mp4`
  (640x360, gitignored) and EXAMINED via a 30-frame filmstrip: deep-stance
  holds/motion (9.5-24 s, 29-33 s), stance motion with high hand carriage
  (108-118 s), sprawl/ground-recover drills (33-48 s). Decision: it CORROBORATES
  the stance/motion vocabulary but is 640x360 with the coach small in frame — it
  is recorded in the clip index as verification-only; no retargeting derives
  from it.
* **`gBAhX5t-GW4`** ("How to stance and motion drills for wrestling", Footwork
  Trainer, 13:44) remains the primary: chapter-mapped, full pose pipeline
  (MediaPipe landmarks at 15 fps effective) already stored, and all retargeted
  references derive from it. Source footage WAS examined directly this session:
  extracted frames at 27.3/30.5-34.5/59.5-65.5/124/427 s confirm (a) the
  "stance" chapter oscillates stand↔stance (the old `stance_hold` window is NOT
  a static hold), (b) 59-66 s is genuine in-stance reposition motion, (c) 427 s
  shows the penetration end-state with the lead knee ON the mat.
* Honest limitation: the pose track is monocular (MediaPipe world landmarks are
  HIP-CENTRED); depth/scale are approximate, and hip-centring makes any foot
  height/speed measurement relative to the hips (this bit the detector twice —
  see §3).

## 3. The grounding defect — FIXED, with the failure history

v0 (`data/refs_video/*`) floored the feet with ONE constant per take; the
fidelity report §4 measured emitted soles at +0.111 m hover / −0.067 m
penetration. The v1 fix is two per-frame, contact-aware passes
(`retarget_video.py: ground_per_frame` + `ground_qpos_soles`, documented in
FORMAT.md). Three detector designs were measured before the final one:

| detector version | measured failure |
|---|---|
| take-level plane + per-frame speed gate | contact 0.25 on a STATIC stance take (landmark jitter = 0.15-0.45 m/s of fake speed) → anchor thrash → stretch 3.27 → solver collapse (pelvis 0.22 m) |
| take-plane, no speed gate | lost 92 % of a level-change take's contact frames: hip-centring makes a crouch read as foot lift (hips drop ⇒ foot landmarks rise) |
| **final: per-frame RELATIVE plant** (sole within 5 cm of the frame's lowest sole + dwell; offset clip [−0.02, +0.60] m to express real crouches) | stance takes contact 0.96-0.99, level changes 0.71-0.86 |

Also fixed in the same pass: the translation anchor is now the PLANTED foot
(v0's "smaller displacement" rule agreed with the video's lower foot in only
26-77 % of frames), the emitted qpos gets a residual per-frame z-shift (a root
translation is exact; joint angles untouched), and that shift is rate-limited
to 1.2 m/s (an unclamped version stepped the root at 4 m/s — untrackable).

**Re-measured fidelity** (`scripts/measure_reference_fidelity.py --ref-dir
data/references/motion_refs/v1/refs --json
data/solo/metrics/reference_fidelity_v1.json`; self-verify 8/8 OK):

| metric | v0 | v1 |
|---|---|---|
| anchor = video's lower foot | 0.26-0.77 | **0.57-1.00** (stance 0.86-1.00, shot 0.97) |
| phantom plants | 0-0.67 s | **0-0.20 s** |
| planted-sole residual (median, drill takes) | +2.6..+5.3 cm | **≈0-2 cm** (test-pinned: median ≤2 cm) |
| time stretch | 1.0-2.31 | 1.0-3.12 (stalk; capped to 10 s in the drill) |

## 4. Fusion (acceptance 3, 8) — `fusion_spec.json`

Policy: video = timing/ordering/carriage/cadence; GrappleMap = spatial
relationships/technique structure. 14 features, each with measured values from
BOTH sides, the chosen fused value, the conflict rule, and a confidence. The
load-bearing decisions:

* **stance width**: video widen-step max 0.56 m (G1 FK) vs GrappleMap STANCE
  0.303 m vs G1 keyframe 0.237 m → fused to the operator repair band
  (0.30-0.36 m); the STANCE_HOLD phase holds `stance_qpos(width=0.34,
  height=0.74)` — measured width 0.345 m.
* **stance crouch height**: video crouches to 0.62 m (below the measured 0.72 m
  position-servo holdability bar) → the hold clamps to 0.74 m; the RAW video
  posture stays in `refs/stance_hold.npz` with its own measured verdict
  (margin −0.036 m — not holdable).
* **shot structure**: video timing/events (clip index: measured weight-transfer,
  knee-descent, wrist-extension timestamps) + GrappleMap DOUBLE_LEG spatial
  (penetration depth 0.593 m, lead knee 0.064 m, width 0.619 m, G1 FK) —
  because the video's knee-depth channel is the noisiest AND its knee-down
  end-state (pelvis 0.22 m) is not G1-reachable through this pipeline (solver
  collapses to 0.05 m; measured).
* No frame-by-frame angle averaging anywhere; every trim is logged in the phase
  note (stance repair, 10 s shuffle/circle caps, crouch cut at pelvis 0.35 m,
  reachability tail trims with cut times in `meta.reachability_trim`).

## 5. The composed drill (acceptance 4, 5, 6)

`drill_continuous.npz`: 60.2 s, 50 Hz, 23 phases — STAND(g1-native) →
LOWER_TO_STANCE(observed) → STANCE_HOLD(fused repair) → SHUFFLE_F(observed,
capped 10 s) → SHUFFLE_B(observed) → CIRCLE(observed, capped 10 s) →
LEVEL_CHANGE(observed) → DOUBLE_LEG_ENTRY_CROUCH(observed,
**known_infeasible**, kept not hidden) → DOUBLE_LEG_PENETRATION(**fused**:
GrappleMap geometry) → RECOVER_TO_STANCE(observed) → REPOSITION(observed) →
REPEAT_BLEND(synthetic) — plus 11 CONNECT_* synthetic connectors, ALL marked
`source: synthetic_connector` in `meta.phases` (never passed off as observed
motion).

Measured composition quality: worst joint step 0.119 rad (= the retimer's own
6 rad/s cap), worst root step 0.063 m/frame (= the pipeline's own 3 m/s cap),
REPEAT closure gap **0.0008 rad**, root path length 1.62 m, connectors are
cubic Hermite matched in position AND finite-difference velocity (adaptive
length: the fixed 0.4 s blend hit 13 rad/s across the crouch→standing gap, so
length scales with the pose gap), connector root-lift keeps soles out of the
mat during big pose transitions. Lead leg is measured per phase from FK
(entry/penetration lead LEFT; CIRCLE/RECOVER right — the connectors absorb the
switches, no mirroring was needed).

## 6. Feasibility — three levels, honest verdicts (acceptance 9)

`feasibility.json` (instrument: FK on the solo model; penetration = lowest sole
sphere vs mat; static = CoM vs planted-foot hull on held frames; dynamic =
real MuJoCo rollouts driving the reference's OWN joint targets at 50 Hz, fall
rule = pelvis<0.35 m & tilt>60° sustained 0.25 s — the CEM capture's protocol):

| track | verdict | evidence |
|---|---|---|
| level_change_fast | **dynamically_verified** | probe PASS: completed upright (1.4 s transient) |
| stance_hold (raw video) | unverified | probe FAIL (topples 2.2 s); crouch 0.62 m < 0.72 m servo bar; CoM margin −0.036 m |
| stance takes / shuffles | unverified | kinematically valid; probe FAIL; no ≥0.4 s static hold in the source (longest 0.93 s — measured) |
| shot_entry_full | **known_infeasible** | crouch topples in 1.24 s under its own joint targets (cem_capture.md); pelvis −0.22 m end-state unreachable; pen −0.032 m |
| knee_sprawl_* | known_infeasible | ground postures by design; context only, excluded from the drill |
| drill_continuous | known_infeasible **by design** | contains the labelled crouch phase; everything else diagnosed per phase |
| stance_rise | known_infeasible | held CoM margin −0.088 m |

**Why 14/16 probes topple is expected, not a defect:** raw position servos have
no balance layer (the repo's founding measurement: PD replay topples ~1.4 s;
pure joint-PD sagged the GrappleMap refs 0.29-0.83 m). The probe's job is to
separate what dumb servos CAN traverse (level_change_fast) from what needs the
imitation/RL stack — that is exactly the gap the M1-M5 curriculum exists to
close. A PASS here would have meant the probe was broken.

Known limitation (restated precisely, measured by the orchestrator's FK and
re-verified): sole penetration is NOT confined to fast phases — the v1 takes
carry a steady-state offset in HELD postures too (stance_hold: min −0.0149 /
median −0.0112 / max +0.0009 m; i.e. ~1.1 cm, AT/OVER the QUALITY_RUBRIC G2
"no ground penetration > 1 cm" bar, vs our certified keyframe's −0.0019 m).
Likely cause (hypothesis, unfixed): the retarget pipeline grounds the
retarget model's toe/heel landmark SITES (sole plane), while the quality FK
measures the solo model's sole-sphere bottoms — a site-vs-sphere bookkeeping
offset plus the rate limiter's small residual. Recorded as a KNOWN DEVIATION
AGAINST G2; cheapest fix path: add the measured +0.011 m constant inside
`ground_qpos_soles` (re-verifying per posture) before any tracking stage
consumes the references.

## 7. What is trustworthy vs not

**Trustworthy:** the grounding fix + its numbers (instrument self-verify 8/8,
synthetic-track tests); the clip index timestamps (measured from stored
landmarks, key windows frame-verified); the composition's continuity/closure
numbers; the probe FAIL verdicts (protocol identical to the CEM control);
GrappleMap spatial values (explicit 3D FK).

**Do NOT trust:** absolute monocular depths for steps oblique to the camera
(clip index carries the caveat); the knee_sprawl family's crouch depths
(distorted 0.20-0.38 m at G1 scale — fidelity §7); any claim that the
references are executable as-is (they are TARGETS; the dynamic column says
exactly one segment is servo-traversable); the static margins on takes whose
soles still penetrate (the hull displacement makes margins read ~5-10 cm
pessimistic there).

**UNVERIFIED / open for Agent 2:** whether the repaired STANCE_HOLD
(0.345 m/0.74 m) closes the loop under the certified stance controller (v6d
holds that class, but the phase itself was not probed with the controller);
whether the fused penetration reads as the technique to a human reviewer
(frames provided); the level_change take's 1.74x stretch (the retimer paid for
the fast drop).

## 8. Reproduce

```
.venv/bin/python scripts/build_clip_index.py
MUJOCO_GL=egl .venv/bin/python data/references/yt_gBAhX5t-GW4/derived/tools/retarget_video.py \
    --npz data/references/yt_gBAhX5t-GW4/pose/landmarks.npz \
    --spec data/references/motion_refs/retarget_spec_v1.json \
    --out-dir data/references/motion_refs/v1/refs      # per take (spec notes pass2 takes)
MUJOCO_GL=egl .venv/bin/python scripts/finalize_v1_refs.py
.venv/bin/python scripts/build_fusion_spec.py
MUJOCO_GL=egl .venv/bin/python scripts/compose_drill.py
MUJOCO_GL=egl .venv/bin/python scripts/probe_drill_dynamic.py
MUJOCO_GL=egl .venv/bin/python scripts/feasibility_diagnosis.py
MUJOCO_GL=egl .venv/bin/python scripts/render_motion_ref_videos.py
.venv/bin/python scripts/query_motion_refs.py drill_continuous --phase
.venv/bin/python -m pytest tests/test_motion_refs.py -q
MUJOCO_GL=egl .venv/bin/python scripts/measure_reference_fidelity.py \
    --ref-dir data/references/motion_refs/v1/refs
```
Live training disclosure: the solo-t1-v6d run used one core throughout; all
batch jobs above were run sequentially on the remaining cores.
