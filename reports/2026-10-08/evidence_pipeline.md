# Evidence pipeline: the verifier, the gate, and label integrity

Date: 2026-10-08 · Author: EvidenceFix · Scope: `scripts/solo_drill_render.py`,
new `scripts/evidence_check.py`, new `tests/test_evidence_integrity.py`,
bundle refreshes under `data/solo_drill/`, this report.
Predecessor: `reports/2026-10-08/clip_wiring_audit.md` (EvidenceAudit, read-only).

The audit's verdict was: the clip→trace attach is mechanical (so F1 was a one-off
hand entry), but every **label** is hand-typed and never compared to its trace,
and `verify_clip` recorded `ok:false` on **12 of 13** bundles — so **no clip was
machine-corroborated** while the docs claimed verification. This pass fixes the
verifier (which was already half-fixed in `103dcee`), makes verification
**gating**, re-verifies every clip, fixes the mechanical label drifts, and adds
the durable **label-integrity check** that would have caught the audited defects.

---

## 1. `verify_clip`: root cause, the fix, and what it actually asserts

### 1.1 Root cause (historical) — a verifier bug, not bad media

Pre-`103dcee` the frame-count seed was:

```python
n = int(meta.get("nframes") or expect_frames)
```

`imageio` reports `nframes: inf` for every ffmpeg-written mp4 in this repo
(confirmed on the current artifacts: `L2_isolated_step.mp4` →
`meta["nframes"] == inf`). `int(inf)` raises
`OverflowError: cannot convert float infinity to integer`; the exception was
swallowed by the broad `except Exception` at the end of `verify_clip` and
recorded as `problems: ["decode failed: ..."]`. **The non-black / non-static
half therefore never ran**, and because the outer handler reported it as a
*decode* failure, a bug in the checker looked exactly like a defect in the clip.

Evidence for the historical claim (verified independently):

* `git log -S "np.isfinite" -- scripts/solo_drill_render.py` → the guard landed in
  `103dcee` (2026-10-08 09:08), which replaced the `int(inf)` line;
* every `ok:false` record in `data/solo_drill/*.json` predates that commit
  (e.g. `01_baseline_stancepd.json` at 06:42);
* re-running the *current* verifier on all 14 bundles gives `ok:true` (this
  pass), and §3's table shows the pixel check producing real numbers
  (`mean_luma`, `frame_diff`) for every clip.

So the 12 `ok:false` records are **stale history**. What was *not* stale is the
consequence: the pipeline accepted those records as if they meant something.

### 1.2 The fix (this pass)

`verify_clip(path, expect_s=None, expect_frames=None, fps=30, speed=1.0)`

* **frame-count seed** is now ffprobe's `nb_frames` first, then imageio's
  `nframes` **only when finite**, then the trace-derived `expect_frames`, then
  `duration*fps`. `OverflowError` is in the caught tuple. The `nframes=inf`
  input can no longer break the check.
* **expectations are trace-derived, never artifact-derived.** All three call
  sites (`cmd_suite`, `cmd_l2suite`, `cmd_deliver`, `cmd_tracks`) compute
  `expect_s`/`expect_frames` with the new `window_expectations(trace, t0, t1,
  speed)`, which mirrors the renderer's own resampler:
  `frames = round((min(t1, trace_end) - t0) * fps / speed) + 1`. Verification is
  therefore possible to run *independently of the render* — the checker holds
  the artifact to what the trace and the window imply.
* **the recorded problem is now typed**: a verifier-side exception is stored in
  `out["error"] = {where, type, message}` (kept separate from media
  `problems`), so a toolchain bug can never again be recorded as a clip defect.
* **measured numbers are always reported** (`measured: {duration_s, frames,
  fps, size, mean_luma_min, frame_diff_max}` plus the raw ffprobe fields), and
  every call site prints them next to `ok`.
* the two callers that had a `fps` parameter but never used it (it was a no-op)
  now **assert `r_frame_rate == fps`** — this is the writer-fps bug class that
  made the 0.25x clip play at 1.0x.
* the old expectations were wrong in two ways for clips whose episode ends
  early or that are slowed: `expect_s=(t1 or t_end) - t0` (no episode-end
  clamp, no `speed`). That produced the three bogus `duration ... vs expected`
  problems (`99_failure_entry_L2` 1.87 s vs "11 s", `99_failure_push90N_diag`
  3.17 s vs "7 s", `final_L2_entry_walk` 9.23 s vs "15.6 s") which a later
  reader mistook for **truncated artifacts**. They are not: 56 / 95 / 277
  frames are exactly `round((min(t1, trace_end) - t0)*30) + 1` for those
  windows (episodes fall at 2.82 / 9.14 / 9.60 s). No clip was ever truncated;
  the expectation ignored the physics.

### 1.3 What the check asserts, and whether each assertion is meaningful

| assertion | how | meaningful? |
|---|---|---|
| decodes (h264 / yuv420p) | ffprobe + imageio read of 4 sampled frames | **yes** — delivery-format rule (`VISUALS.md:4`); a wrong codec fails |
| frame rate == 30 | `r_frame_rate` vs `fps` | **yes** — catches the writer-fps class that silently re-times a clip (the 0.25x→1.0x bug); previously not checked at all |
| duration == `(min(t1,t_end)-t0)/speed` ±0.5 s | from trace+window via `window_expectations` | **yes, once the window is recorded** — catches truncated or mis-windowed renders and slow-mo claims that are not slowed; it does *not* prove every frame is present (tolerance ±0.5 s / ±3 frames) |
| frame count == trace+window expectation ±3 | same | **yes** — previously self-referential (the file was compared with what the renderer claimed to have written, `r["frames"]`), so a renderer-side count bug could not be caught; now independent |
| non-black: min mean luma ≥ 8 | 4 sampled frames | **yes but coarse** — a mostly-black clip passes; it catches black/blank renders |
| non-static: max mean-abs frame diff ≥ 0.5 | 4 sampled frames | **weak but real** — proves *something* moves between the 10/40/70/95 % samples; a clip whose subject is frozen while the HUD ticks would pass |
| — not asserted — | | no pixel↔trace correspondence check (no re-render diff), no semantic-content check ("weight shift", "level change"), no dimension gate (13 shipped files are 320x240 or 480x360 while `VISUALS.md:4` says 960x720 — now *recorded* per clip, not gated) |

---

## 2. The gating rule (verification can no longer fail silently)

* `verify_clip` returns `ok:false` **and each `cmd_*` renderer gates on it**:
  every bundle now carries `"evidence": bool` (plus `render` window/speed and
  `claim_check`); `cmd_suite`, `cmd_l2suite` and `cmd_deliver` print the
  failures and **exit non-zero**; `cmd_deliver --safe` aborts before renaming a
  partial over a final name (pre-existing behaviour, kept).
* The **index is generated, not hand-typed**: `scripts/evidence_check.py table`
  rebuilds the status table from `data/solo_drill/*.json` + ffprobe + the trace,
  and marks any clip whose verification fails or whose numeric label claims
  contradict the trace as `verification-failed` / `claims-failed`. A clip in
  either state is **not presented as evidence**; the gate verdict is printed at
  the top of the table and `gate` exits non-zero.
* Clips that have no bundle at all are listed as
  `unbundled (format+pixel only)` — never as evidence.
* A bundle whose recorded text has been corrected carries
  `label_correction` / `label_history` (and the old `clip_verification` records
  move to `clip_verification_history`), so the correction is auditable rather
  than a silent rewrite.

**Regenerate the table with:**

```
python scripts/evidence_check.py table            # markdown to stdout
python scripts/evidence_check.py table --json data/solo_drill/evidence_table.json
python scripts/evidence_check.py gate             # exit 1 unless every bundled clip is evidence
```

---

## 3. Re-verification of every clip (machine-generated)

`python scripts/evidence_check.py table`, run after the bundle refresh below.
Every row is produced by the script; nothing here is hand-typed.

<!-- TABLE -->

---

## 4. Mechanical label drifts fixed

1. **Slow motion actually renders at the claimed rate** (mode C).
   `cmd_suite`'s render loop passed no `speed` to `render_trace`; the job tuple
   had no speed field; the bundle did not record one. Now:
   `CLIP_SPECS` carries `speed`, the loop passes it, the bundle records
   `render: {t0, t1, speed, scale, fps, frames}` and
   `render_command` includes `--t0/--t1/--speed/--scale` (the old recorded
   command omitted the window and would have reproduced a *different*
   artifact). The `02_slowmo_level_change_quarter_speed.mp4` spec is corrected
   to `speed=0.25` over `t=0–8.06 s` — the full level-change element (crouch
   0–6.02 s + rise to 8.06 s, from the run's own `element_done` events), not the
   old `t=3–6 s` which held only the hold — and the clip was re-rendered from
   the cached trace (§6). `tests/test_evidence_integrity.py::test_slowmo_duration_ratio_must_match_the_claimed_speed`
   pins the ratio test.
2. **The two `_diag` index paths** (mode B). The post-hoc rename in the old
   loop (`name.replace(".mp4", "_diag.mp4")` *after* the table had named the
   file) is replaced by `_final_name(name, scale)` as the single naming rule,
   used by the renderer, the bundle and the writer of this table; the published
   index (row 10b) names `L0_hold_30s_diag.mp4` and
   `99_failure_push90N_diag.mp4`, both of which exist. Pinned by
   `test_visuals_index_paths_exist`, which expands the row-10b brace list and
   asserts every path exists.
3. **The hand-typed window that excluded its claimed event** (mode D). The
   `L1_90s_with_pushes.mp4` render window is a deliberate `t=0–60 s` of a 90 s
   run whose pushes are at `t=15/38/62 s`; the third push cannot be in the
   clip. Per the accepted remedy we corrected the **claim**, not the window
   (a window covering the third push would be a 33–37 min render; the operator
   constraint is short renders): label/caption now say "t=15 s and t=38 s
   recovered; t=62 s outside this clip / the run's metrics record 3/3". The
   *burned-in HUD text* of the already-rendered file still shows the old
   wording — that is stated in §6 and in the index row. Pinned by
   `test_push_outside_window_must_be_acknowledged` (an unacknowledged push
   outside the window fails; an acknowledged one passes).
4. **Other drifts the new check caught and the refresh fixed** (all recorded in
   the bundles with `label_correction`):
   * `01_baseline_stancepd.mp4` — label claimed "it topples" for a trace with
     `falls=0` (audit F1); label/caption now say a 12 s hold that does **not**
     topple and point at the real topples clip. The code path can no longer
     regenerate the false caption (the wording lives only in `CLIP_SPECS`), and
     the toppling run is now wired into `cmd_suite` as its own `BASELINE_PD` job.
   * `L2_shuffle_3steps_diag.mp4` — "2-3 steps" vs 4 measured → "4 consecutive
     steps".
   * `final_L2_entry_walk.mp4` — caption claimed a "0.25 m wide" base; the
     stepping base is `half_width=0.105` (0.21 m; measured 0.212 m) → corrected.
   * `L2_cycle_step_diag.mp4` — label was silent on the run's fall at 22.56 s;
     the caption now names it and states it is outside the 8–18 s window.
   * `L0_hold_30s_diag.mp4` — label now states the window ("t=0–30 s of the
     60 s run"), matching the artifact.
   * `final_L2_motion.json` — the bundle carried no label/caption (added, from
     the renderer constants) and a stale `stance_note` "0.30 x 0.10 m" (the
     deliver-stage default printed because the render ran without
     `--stance-w`); corrected to the run's own 0.28 x 0.10 m with the reason
     and the measured 0.289 m recorded. `cmd_deliver` now warns loudly when a
     cached trace is rendered without `--stance-w`.

---

## 5. The label-integrity check (the durable fix) and its coverage

`scripts/evidence_check.py` parses every numeric claim out of a clip's
label/caption/stance_note (`label + caption + stance_note`) and compares it with
the bundle + trace. A contradiction is a **failure** and the clip becomes
`claims-failed` (not evidence).

| claim kind | checked against | tolerance / rule |
|---|---|---|
| duration ("12 s", "90 s", "(90 s)") | clip duration, run duration, window length | ±0.5 s to any; event times (`t=…`, `at …`, `to ~…`) excluded |
| window range ("t=0-8.06 s", "0-30 s") | recorded render window | range must lie inside `[t0, min(t1,trace_end)]` |
| step count / range ("5 steps", "2-3 steps") | `metrics.steps_completed` | exact; ranges inclusive |
| stance width / depth ("0.28 x 0.10 m", "0.21 m wide", "0.06 m deep", "0.495 m stance") | rubric `A1_width` / `A2_base_depth` (measured sole separation) | ±0.030 m width, ±0.060 m depth |
| push count and force ("3 x 20 N") | `metrics.pushes` / config `PushSpec`s | count exact; max force ±1 N |
| bare force ("90 N", "35 N is marginal") | the run's push forces | matches → ok; no such push in this run → **skipped** as a cross-run claim |
| slow-motion factor ("0.25x") | measured duration ratio `clip_s / window_s` vs `1/speed` | ±5 % (or ±0.1); window unknown → skipped |
| fall words ("topples", "falls", "stays up", "does NOT topple", "refuses instead of toppling") | `metrics.falls` | positive words require `falls ≥ 1`, negations `falls == 0`; "0 falls"/"no falls" excluded |
| "both feet planted throughout" | per-frame `foot_load` in the window | both feet > 20 N in ≥ 95 % of frames (L0: 100 %) |
| event coverage | every push time vs the window | inside the window, **or** acknowledged in the text ("62 s") |

**What is not mechanically checkable** (stated honestly, and reported as
`skip`/unverifiable per bundle):

* **semantic motion claims** — `final_L1_90s.mp4`'s "weight shift + level
  change": the trace is nearly static after t=8 s (pelvis z moves 2.8 mm). The
  index carries the correction; the checker cannot judge "a level change
  happened" from a label. (A motion-magnitude rule could be added; it is not
  implemented.)
* **cross-run numbers** — `99_failure_push90N_diag`'s caption cites "20 N
  recovers, 35 N is marginal" from *other* runs; skipped, not failed.
* **cross-clip references** — "steps complete in isolation (see
  L2_isolated_step)".
* **stance spec vs measurement** — the measured sole separation runs 0–22 mm
  above the commanded spec in the runs on disk, so the width tolerance is
  30 mm. Consequence: the 0.30-vs-0.28 m drift on `final_L2_motion` (measured
  0.289 m) is **not** caught by the stance rule alone — it is caught by the
  duration/step numbers that travelled with it, and it is now corrected in both
  the bundle and the index. The 0.25-vs-0.21 m entry-walk drift (38 mm) *is*
  caught.
* **burned-in HUD text** — correcting a bundle's label does not change a
  rendered file's pixels; for the three clips whose labels were corrected but
  not re-rendered (§4.3, §4.4) the bundle carries `label_correction` and the
  index row states the on-screen staleness.
* quoted pelvis-z values, "0.61× the reference width" style ratios, and
  "one unbroken episode (no resets)" (the latter is in fact checked implicitly:
  every suite run records `reset_count 0` in provenance).

---

## 6. Honest status of every clip we ship

### 6.1 Bundled clips — 14 bundles, 15 videos, all `evidence`

After the refresh every bundle's `clip_verification.ok` is `true` and every
numeric claim passes; the gate is **PASS (15/15 videos)**.  What "verified"
means, precisely: decodes as h264/yuv420p at 30 fps, is non-black and
non-static, and its duration/frame count match the trace + recorded window.  It
does **not** mean the clip passes the drill rubric, and it does not certify
semantic content (see §1.3 and §5).

| clip | status | caveat |
|---|---|---|
| `final_L1_90s.mp4` | evidence | motion claim is semantic: the trace is nearly static after t=8 s; the index carries the correction (this is the honest reading of the artifact) |
| `L0_hold_30s_diag.mp4` | evidence | 480x360 diagnostic scale; 30 s window of the 60 s run (label says so) |
| `L1_90s_with_pushes.mp4` | evidence | window 0–60 s excludes push #3 (t=62 s); the label/caption now acknowledge it; the burned-in HUD text of the shipped file still shows the old wording |
| `01_baseline_pd_topples.mp4` | evidence | the F1 contrast, rendered from the run that falls (8.38 s) |
| `01_baseline_stancepd.mp4` | evidence | label/caption corrected (the old "it topples" text is preserved in `label_history`/`caption_correction`); pixels still show the old HUD line |
| `99_failure_entry_L2.mp4` | evidence | complete episode ending in the fall at 2.82 s (56 f) — **not** truncated |
| `99_failure_push90N_diag.mp4` | evidence | 480x360; complete (95 f, fall at 9.14 s); the caption's 20/35 N figures are cross-run and reported unverifiable |
| `02_slowmo_level_change_quarter_speed.mp4` | evidence | **re-rendered 2026-10-08 at 0.25x**: 32.267 s / 968 f / 960x720, 4.003x real time over t=0–8.06 s (descent+hold+rise), non-static; the old 3.033 s 1.0x file is gone |
| `L2_isolated_step.mp4` | evidence | — |
| `final_L2_entry_walk.mp4` | evidence | label corrected (0.25→0.21 m base); pixels still show the old HUD line |
| `L2_cycle_step_diag.mp4` | evidence | 480x360; the run falls at 22.56 s (outside the 8–18 s window; now stated in the caption) |
| `L2_shuffle_3steps_diag.mp4` | evidence | 480x360; label corrected to "4 consecutive steps" (filename still says 3steps — cosmetic) |
| `L2_stance_step_refused_diag.mp4` | evidence | 480x360; refusal evidence (0 steps, 7 refusals) |
| `final_L2_motion.mp4` | evidence | the bundle's rubric records `ship gate=FAIL` (margin_min −0.0257 m at 45.76 s) — an engineering result, not an evidence-integrity problem; the index must not read "verified" as "passes the ship gate" |
| `L2_motion_slowmo_step_quarter.mp4` | evidence | verified as the bundle's secondary artifact (409 f / 13.633 s, 0.25x); claims checked against its own window |

**KNOWN-BAD: none.**  No clip fails verification; no clip's numeric claims
contradict its trace.  The three clips earlier notes called "truncated
mid-write" are complete episodes (§1.2).

### 6.2 Clips with no per-clip bundle — 28 videos (19 indexed families + 9 probes)

These have **no bundle and no trace**, so no claim can be corroborated; the
table lists them as `unbundled (format+pixel only)` and they are **not presented
as evidence**.  Format/pixel checks (codec, pix_fmt, fps, non-black,
non-static) pass for all 28; the check cannot say more than that.

* **19 clips in the indexed families** (`videos/refs/*` 7, `videos/teacher/*` 7,
  `videos/env/*` 5) — what they *do* have: family-level corroboration, not
  per-clip: the refs/teacher pair is documented in
  `reports/2026-10-08/teacher.md` (A/B stay-up), the env rules in
  `reports/2026-10-08/env_videos_frame_check.md` (frame-level check, all 5
  MATCH), and each row carries its own KNOWN/verdict note.  The 7 teacher clips
  are **320x240**, i.e. they violate the `VISUALS.md:4` "960x720" rule; that is
  recorded per clip now, not hidden.
* **9 baseline probes** (`videos/solo_drill/baselines/probe_*` 5,
  `t1_*` 2, `t2_*` 2) — produced by `scripts/solo_env_smoke.py`; no bundle
  anywhere; each has a `_sheet.png`.  They are not indexed in `VISUALS.md`.
* `l2_stills/frame_t002.00.png`, `frame_t030.00.png` — stills, no bundle (not
  clips).

### 6.3 The file that was mid-write

`01_baseline_pd_topples.mp4` appeared to the earlier audit as an undecodable
524,336 B file (`moov atom not found`) because the audit snapshot caught it
**mid-write**; it completed at ~14:37 (1,111,764 B), verifies (252 f / 8.4 s,
non-static), has a bundle, and is indexed as the F1 topple contrast.  It is not
a defect.  (The repo's safe-write protocol — render to `*.partial.mp4`, verify,
rename — is now also available on the manual `render` path via `--safe`, which
is how the 02_slowmo re-render above was produced.)

---

## Appendix: reproducing this pass

```bash
# re-run the verifier + refresh stale labels/render blocks (writes bundles)
python scripts/evidence_check.py refresh --write
# label-integrity check alone (per claim, per bundle)
python scripts/evidence_check.py claims
# the machine-generated status table used in §3
python scripts/evidence_check.py table
# the gate the pipeline exports
python scripts/evidence_check.py gate
# the regression tests pinning the audited defects
python -m pytest tests/test_evidence_integrity.py -q
```

---

## Addendum — orchestrator verification (2026-10-08, post-kill)

Agent `EvidenceFix` was stopped by the provider outage (opencode-go weekly quota) after landing the
work above; the orchestrator re-ran the tools on the shipped tree. Measured now, not quoted:

* `python scripts/evidence_check.py gate` → **exit 0: 15/15 bundled clips are evidence**.
* `python scripts/evidence_check.py table` regenerates the status table (§3's command); it is the
  machine-generated replacement for any hand-typed index numbers.
* Drift-fix confirmations from the regenerated table:
  * `02_slowmo_level_change_quarter_speed.mp4` measures **32.267 s / 968 f** — 8.06 s of run at the
    0.25× its name claims (pre-fix it was 1.0×);
  * `L1_90s_with_pushes.mp4` measures **60.033 s / 1801 f**, matching the corrected index entry, and
    its title now reads "t=62 s outside this clip" (the window-excluded push is acknowledged);
  * `final_L2_motion.mp4` measures **69.98 s / 5 steps**, matching its trace (the index previously
    claimed 95 s / 6 steps / 0.30 m — D2);
  * `99_failure_push90N_diag.mp4` verifies with the honest note that the 90 N force is a cross-run
    claim, not checkable from its bundle.
* 27 clips remain unbundled (baselines probes, `videos/refs/*`, `videos/teacher/*`, `videos/env/*`):
  format+pixel checks pass, no claim corroboration — they must not be cited as evidence.
* `pytest tests/test_evidence_integrity.py -q` passes; the nine audited defects (D1–D9) stay pinned.

*Provenance note:* commit `449eab8` briefly replaced this file with an orchestrator-written summary;
the original report was restored from `ce8dc3d` and this addendum appended, so the agent's own
record stands above.
