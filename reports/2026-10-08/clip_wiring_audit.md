# Clip wiring audit — is every clip label corroborated by the run it was rendered from?

Date: 2026-10-08 · Author: EvidenceAudit · Scope: REPORT-ONLY (no repo edits)
Question: (a) is F1 ("PD baseline topples at 8.4 s") a one-off hand-entry error or a **wiring** bug
that would make every clip label suspect? (b) which other indexed clips cannot be corroborated by
their own bundle?

Method: read the binding code in `scripts/solo_drill_render.py` (file:line cited below), then sweep
every clip in `docs/VISUALS.md` against `data/drill/*.json` (run identity/metrics),
`data/solo_drill/*.json` (bundle: video path, `trace_npz`, `clip_verification`), the `*.npz` traces
themselves, and `ffprobe` on every artifact. No re-render was needed; no re-render was run. The
live trainer and LedgerFix's in-flight F1 work were not touched.

**Revision pin.** All `scripts/solo_drill_render.py` line numbers below were read from the working
tree at `sha1 9076278e5e8764394088d6f7615dee668a3212ff` (HEAD `72956f3`). While this audit ran,
LedgerFix edited the `BASE_PD` job in the same working tree (caption/claim corrected, +7 lines), so
the `BASE_PD` job now spans `:192-204` instead of `:192-196`; everything else cited is unchanged.
Bundle/trace/`ffprobe` numbers are unaffected by that edit.

---

## 1. Mechanism: how a clip NAME is bound to the trace it renders

The binding is **a shared table plus a mechanical tag→npz lookup**, and the *claim* is **hand-typed
free text that nothing ever checks**. Verbatim:

- `scripts/solo_drill_render.py:184-230` — `jobs = [ (run_tag, RunConfig(**cfg_kw), [(name, title,
  t0, t1, caption, scale), …]), … ]`. One table for the whole suite; the L2 suite repeats the shape
  at `:342-369`, the track suite builds its names by f-string at `:497`.
- `:246-248` — the render pass walks the same table: `for run_tag, cfg_kw, renders in jobs: …`
  `paths = results[run_tag]["paths"]`.
- `:252` — `npz = Path(paths["npz"])`; `:255` — `render_trace(load_trace(npz), …)`. The npz comes
  from `res.save(DATA)` of the run keyed by that same `run_tag` (`:236`), so the **trace a clip
  renders is produced by the config in its own tuple**. There is no separate name→npz table and no
  glob that could silently pick the wrong trace (`solo_drill_render.py:17-18` states the intent:
  "the physics is never re-run by `render`: frames come from the saved trace, so a clip and its
  metrics JSON always describe the same episode").
- `:266-282` — the bundle is written from the *same* `paths`/npz (`"trace_npz": paths["npz"]`,
  `"label": title, "caption": caption`), so bundle↔trace identity is also mechanical.
- What is **not** mechanical: `name`, `title`, `caption`, `t0`, `t1`, `scale`. Nothing compares them
  to the run. `verify_clip` (`:118-176`) checks codec / pix_fmt / duration / frame count / one
  non-black and one non-frozen sample — it never reads the claim, and it never reads the trace.
- Classification: **(i) shared table/loop for the tag→npz link + (iii) bundle fields for identity**,
  with the claim being a **hand-written per-clip string**. Not a name→npz table that can drift.

### Verdict on systemic

- **A single wrong attach does NOT propagate to other clips**: the tag→npz link is mechanical, so
  F1 is a **one-off hand-entry error** (the wrong member of a near-duplicate pair of tags was typed
  into the table). In F1's case both runs exist on disk: `BASE_PD_stance_pd_L1_seed0`
  (falls=0, pelvis z 0.7235–0.7279 m, 11.98 s) is wired to `01_baseline_stancepd.mp4`
  (`solo_drill_render.py:192-196`); the toppling `BASELINE_PD_stance_pd_L1_seed0`
  (falls=1 @ 8.38 s, z→0.299 m) is wired to **no** render job at all.
- **But the guard against it does not exist, and that part IS systemic**: every claim in the index
  is hand-typed and machine-unchecked, and the one routine that could corroborate a clip
  (`verify_clip`) returns `ok:false` on **12 of 13** bundles (§3.4). So *labels are never
  corroborated* — systemically — while *traces are never silently mis-attached* — mechanically.

### Failure modes that ARE automatic (each affects several clips at once)

| id | mode | code | clips exposed | how to detect |
|---|---|---|---|---|
| A | hand-typed claim never compared to the trace | `:249-265` (claim text has no relation to `results[run_tag]`) | all 13 bundled clips | assert claim invariants at bundle-write time (e.g. caption contains "topple" ⇒ `metrics.falls >= 1`; "N x … pushes" ⇒ the clip window covers all push times; "0.25x" ⇒ `speed == 0.25` and `duration == (t1-t0)/speed`) |
| B | the loop **renames** `scale<1.0` clips to `*_diag` *after* the table names them | `:250-251` | `L0_hold_30s_diag.mp4`, `99_failure_push90N_diag.mp4`, `L3_track_*_diag.mp4` | the index must be generated **from** `data/solo_drill/*.json` (`bundle["video"]`), not copied from the table |
| C | the loop passes **no `speed`** to `render_trace`; the slow-motion claim lives only in the name/title/caption | job `:221-225` vs call `:255-260` | `02_slowmo_level_change_quarter_speed.mp4` (the only slow-mo job in the table) | put `speed` in the job tuple and in the bundle; verify `duration ≈ (t1-t0)/speed` |
| D | hand-typed `t0`/`t1` can exclude the event the caption claims (windows are **not recorded in the bundle** at all) | `:188-225` windows; `:268-278` bundle omits t0/t1 | `L1_90s_with_pushes` (push #3 at 62 s, window ends 60 s), `L2_cycle_step_diag` (fall at 22.56 s, window ends 18 s), `99_failure_entry_L2` (window 1–12 s, episode ends 2.82 s), `L0_hold_30s` (window 0–30 s of a 60 s run) | record t0/t1/speed in the bundle and assert every event named in the caption has `t0 <= t <= t1` |
| E | `verify_clip`'s imageio path dies before the pixel check | `:154-158` (`nframes` guard) → caught at `:174-176` as `decode failed: cannot convert float infinity to integer` | **12/13** bundles | use ffprobe's `nb_frames` (already in scope as `expect_frames`) as the fallback for `nframes`; then re-verify |
| F | near-duplicate hand-typed run tags for the same nominal case | `BASE_PD` vs `BASELINE_PD` (`:192-196` vs `data/drill/BASELINE_PD_*`) | the F1 pair; same shape exists for `FINAL_L1_90` vs `FINAL_L1_push` | bundle the trace path **and its sha256**; the index cites the npz name so a tag swap is visible as a different hash |

---

## 2. Complete row set — every clip in `docs/VISUALS.md`

Row 10b (the solo-drill index) plus the two prose-named clips and the L2 family that exists on disk.
`trace` = the npz identity the bundle records; `t_end/steps` = the trace's own duration and step
count; `cv` = the bundle's `clip_verification`.

### 2.1 Indexed in VISUALS row 10b

| clip (as indexed) | bundle JSON | run/trace identity in the bundle | label claim (VISUALS/prose) | cv | ffprobe | trace t_end / steps | MATCH? | evidence |
|---|---|---|---|---|---|---|---|---|
| `videos/solo_drill/final_L1_90s.mp4` | `final_L1_90s.json` | `FINAL_L1_90_feasible_L1_seed0` (tag `FINAL_L1_90`, feasible L1 90 s) | "rung L1 - stance hold + weight shift + level change (90 s)"; index CORRECTION: "PHYSICALLY NEARLY STATIC" | **ok:false** (decoder) | 90.000 s / 2700 f / 960x720 | 89.98 s / 0 steps / 93 timeouts of 185 elements | **NO** (contradicted — motion claim) | pelvis z after t=8 s = **2.8 mm** (44.5 mm total, all in t=0–8 s); `final_L1_90s.json` metrics + `FINAL_L1_90_*.npz`; index already carries the correction |
| `L0_hold_30s.mp4` → **no such file**; exists as `L0_hold_30s_diag.mp4` | `L0_hold_30s_diag.json` | `FINAL_L0_60_feasible_L0_seed0` (tag `FINAL_L0_60`, 60 s) | index: "L0 = the same stance held **60 s** with both feet planted"; bundle label "stance hold + posture modulation" | **ok:false** | 30.033 s / 901 f / **480x360** | 59.98 s / 0 steps / falls 0 | **NO** (path) / **PARTIAL** (artifact is 30 s of a 60 s run) | dir listing; bundle `config.seconds=60`, `trace t_end=59.98`; pelvis-z range 28.6 mm (posture modulation is real) |
| `L1_90s_with_pushes.mp4` | `L1_90s_with_pushes.json` | `FINAL_L1_push_feasible_L1_seed0` (3 pushes, 90 s) | "rung L1 with pushes (3 x 20 N, all recovered) … the push windows are overlaid" | **ok:false** + `duration 60.03s vs expected 90.00s` | 60.033 s / 1801 f / 960x720 | 89.98 s / 0 steps / falls 0; pushes **3/3 recovered** at t=15.0/38.0/**62.0 s** | **PARTIAL** | run metrics corroborate 3/3 recovered; the artifact window is 0–60 s (`:207`), so **push #3 (t=62 s) is not in the clip** |
| `01_baseline_stancepd.mp4` | `01_baseline_stancepd.json` | `BASE_PD_stance_pd_L1_seed0` (stance_pd, 12 s) | bundle caption "the same stance pose under pure position control: **it topples**"; VISUALS "topples on the identical pose at 8.4 s" | **ok:false** | 12.000 s / 360 f / 960x720 | 11.98 s / falls **0** / fall_times `[]` | **NO** (contradicted — **F1**) | pelvis z 0.7235–0.7279 m (4.4 mm), no drop; the toppling run is a *different* file: `BASELINE_PD_stance_pd_L1_seed0` (falls=1 @ 8.38 s, z→0.299 m) |
| `99_failure_entry_L2.mp4` | `99_failure_entry_L2.json` | `FAIL_ENTRY_L2_feasible_L2_seed0` | "FAILURE: stand -> stance entry (L2 stepping)"; caption "steps complete in isolation but the sequence is not clean yet" | **ok:false** + `duration 1.87s vs expected 11.00s` | 1.867 s / 56 f / 960x720 | 2.82 s / **falls 1 @ 2.82 s** / steps 0 | **YES** (failure label truthful) | the "steps complete in isolation" half is the *other* clip (`L2_isolated_step`, refusals=9); duration "mismatch" is mode E: window says 1–12 s, the episode ends at the fall |
| `99_failure_push90N.mp4` → **no such file**; exists as `99_failure_push90N_diag.mp4` | `99_failure_push90N_diag.json` | `FAIL_PUSH90_feasible_L1_seed0` | "FAILURE: 90 N push … 90 N topples" | **ok:false** + `duration 3.17s vs expected 7.00s` | 3.167 s / 95 f / **480x360** | 9.14 s / **falls 1 @ 9.14 s** | **YES** (label) / **NO** (path) | push-90N at t=8.0 s; the clip (6.0–9.14 s) contains the topple |
| `02_slowmo_level_change_quarter_speed.mp4` | `02_slowmo_level_change_quarter_speed.json` | `SLOWMO_L1_feasible_L1_seed0` | "slow motion **0.25x**: level change down / hold / rise"; caption "0.25x of the level change: descent, hold, rise (rubric H3)" | **ok:false** | 3.033 s / 91 f / 960x720 | 19.98 s / falls 0 | **NO** (contradicted ×2: speed **and** content) | (a) a 0.25x render of the 3.0–6.0 s window would be ~12 s; 3.033 s = **1.0x** (no `speed` is passed, mode C). (b) pelvis-z profile: 0.7278 (t=0) → 0.6851 (t=2.8, descent) → 0.6869–0.6890 (t=3–6, **hold only**; range 2.2 mm) → 0.7278 (t=8, rise). The clip window 3.0–6.0 s **contains neither the descent nor the rise** |
| `03_side_by_side_reference.png` | none | n/a (image) | row-10b sub-index | n/a | exists (PNG) | n/a | **UNVERIFIABLE** (no bundle, no metadata; not a clip) | — |

### 2.2 Named in the 10b prose (not in the brace list)

| clip | bundle JSON | run/trace identity | claim | cv | ffprobe | trace t_end / steps | MATCH? | evidence |
|---|---|---|---|---|---|---|---|---|
| `final_L2_motion.mp4` | `final_L2_motion.json` | `M_E28f_feasible_L3_seed0` (`config.seconds=70`, stance **0.28 x 0.10 m**) | index: "**95 s**, 0 falls, **6** visible gate-checked steps … in a **0.30 m**-wide stance" | **ok:true** (verified the `*.partial.mp4` path) | 70.000 s / 2100 f / 960x720 | 69.98 s / **steps 5** (15.9 / 29.34 / 45.9 / 50.72 / 64.48 s) / falls 0 | **NO** (contradicted ×3) | the index's numbers are the **superseded `M_P30_feasible_L3_seed0`** run (94.98 s / 6 steps / 0.30 m — `data/drill/M_P30_*.json`); the shipped clip is `M_E28f`; `reports/2026-10-08/drill_motion.md` §"the rendered clip is the 0.28 x 0.10 m run (M_E28f…)" says so explicitly |
| `L2_motion_slowmo_step_quarter.mp4` | `final_L2_motion.json` → `slowmo` + `slowmo_verification` | `M_E28f_feasible_L3_seed0` | "0.25x slow-motion of a step cycle (rubric H3)" | **ok:true** | 13.633 s / 409 f / **480x360** | 3.4 s of trace at 0.25x ⇒ 13.6 s | **YES** | the only genuine quarter-speed clip on disk (mode C fixed here, not in `cmd_suite`) |

### 2.3 Present with a bundle but NOT indexed in VISUALS

| clip | bundle JSON | run/trace identity | claim | cv | ffprobe | trace t_end / steps | MATCH? | evidence |
|---|---|---|---|---|---|---|---|---|
| `L2_isolated_step.mp4` | `L2_isolated_step.json` | `L2_ENTRY_SPEC_feasible_L2_seed0` | "one isolated step … the rest of the entry is refused on geometry" | **ok:false** | 6.633 s / 199 f / 960x720 | 13.98 s / steps 1 / **refusals 9** / falls 0 | **YES** | `l2_evidence_notes`: refusals 9 |
| `final_L2_entry_walk.mp4` | `final_L2_entry_walk.json` | `L2_ENTRY_BASE_feasible_L2_seed0` | "L2: stand -> stepping base, walked" (VISUALS: "the sequence falls") | **ok:false** + `duration 9.23s vs expected 15.60s` | 9.233 s / 277 f / 960x720 | 9.60 s / **falls 1 @ 9.60 s** / steps 1 | **YES** | failure truthful; duration noise = mode E |
| `L2_cycle_step_diag.mp4` | `L2_cycle_step_diag.json` | `L2_CYCLE_feasible_L2_seed0` | "one step per L1 programme cycle" | **ok:false** | 10.033 s / 301 f / **480x360** | 22.56 s / steps 3 / **falls 1 @ 22.56 s** | **PARTIAL** | steps=3 ✓ but the run **falls** and the label says nothing; the clip window is 8–18 s so the fall is off-screen |
| `L2_shuffle_3steps_diag.mp4` | `L2_shuffle_3steps_diag.json` | `L2_SHUFFLE_feasible_L2_seed0` | "short shuffle (**2-3 steps**)" | **ok:false** | 14.033 s / 421 f / **480x360** | 33.98 s / **steps 4** / falls 0 | **NO** (contradicted — minor) | label understates its own run |
| `L2_stance_step_refused_diag.mp4` | `L2_stance_step_refused_diag.json` | `L2_STANCE_REFUSED_feasible_L2_seed0` | "the drill stance refuses the step … it refuses instead of toppling" | **ok:false** | 7.033 s / 211 f / **480x360** | 9.98 s / **refusals 7** / falls 0 / steps 0 | **YES** | `l2_evidence_notes`: refusals 7, falls 0 |

### 2.4 Non-indexed video files present in `videos/solo_drill/` (no bundle, no index entry)

`01_baseline_pd_topples.mp4` — **524 336 B, mtime 2026-10-08 14:35:52, `ffprobe: moov atom not
found → Invalid data`**, no bundle, no contact sheet, referenced by no bundle and by no index row
[INFERENCE: an in-flight/aborted write under a *final* artifact name — it appeared ~1 min before this
check, while LedgerFix is re-rendering F1; the repo's own safe-write protocol renders to
`*.partial.mp4` and renames only after verification, which `notes.md` §453-457 records being
violated once before]. Not touched by this audit.
`baselines/probe_{fall_forward,random_init_policy,squat_repeat,stand_hold,zero_action}.mp4`,
`baselines/t1_{stand_hold,random_init_policy}.mp4`, `baselines/t2_{stand_hold,random_init_policy}.mp4`
— 9 clips, 0.53–4.00 s, 960x720, **no bundles, no index entries** (each has a `_sheet.png`).
`l2_stills/frame_t002.00.png`, `l2_stills/frame_t030.00.png` — 2 stills, no index entry.
`*_sheet.png` — 13 contact sheets; `02_slowmo…_sheet.png` etc. exist, but note the two `_diag`
clips' sheets are named `*_diag_sheet.png`, not the indexed `*_sheet.png`.

### 2.5 The other indexed families (VISUALS rows 2, 3, 5)

| family | files on disk | ffprobe | per-clip bundle | MATCH? |
|---|---|---|---|---|
| row 2 `videos/refs/{DOUBLE_LEG,SINGLE_LEG,BODY_LOCK,SNAPDOWN,SPRAWL,STAND_UP,STANCE}.mp4` | 7/7 exist | 1.40–9.90 s, 42–297 f, 960x720, h264/yuv420p | **none** | **UNVERIFIABLE per clip** — no `clip_verification` bundle exists anywhere for them (the only clip bundles in the whole `data/` tree are the 13 `data/solo_drill/*.json`); corroboration is family-level only (`reports/2026-10-08/teacher.md`, and the row's own "KNOWN: robots sag/topple") |
| row 3 `videos/teacher/<TECH>.mp4` | 7/7 exist | 1.23–6.07 s, 37–188 f, **320x240**, h264/yuv420p | **none** | **UNVERIFIABLE per clip**; also violates the stated format rule (`VISUALS.md:4`: "Videos are 30 fps, 960x720") |
| row 5 `videos/env/{draw_stance,takedown_back_event,nonterminal_knees_hands,ambiguous_simultaneous,oob_forfeit}.mp4` | 5/5 exist | 5.07–6.07 s, 152–182 f, 960x720, h264/yuv420p | **none** | **UNVERIFIABLE per clip** from bundles; the row cites a frame-level report (`reports/2026-10-08/env_videos_frame_check.md`, exists) as its corroboration |
| rows 6-10 (`videos/ppo/*`, `videos/free/*`, `videos/failures/*`, `final_wrestling_match.mp4`) | **0 files** | — | none | consistent with the row status "pending" |

So: **19 clips in the other indexed families have no per-clip corroboration at all** (no bundle
anywhere in `data/`), and 13 files break the row-4 format rule (7 teacher at 320x240 + 6 solo-drill
`_diag` at 480x360).

---

## 3. Cross-check of the substantive claims (cheap, mechanical)

1. **The F1 pair.** `BASE_PD` (wired) vs `BASELINE_PD` (unwired): 0 falls / z 0.7235–0.7279 m / 600
   frames vs 1 fall @ 8.38 s / z→0.299 m / 420 frames. The indexed clip's caption asserts the
   property that only the *unwired* run has. **Label contradicts retained data.**
2. **Duration/frames.** Every `ffprobe` duration matches the bundle's `clip_verification` to the
   frame, and every clip's duration matches `(t1 or t_end) - t0` from the table — including the
   three "mismatch" clips, whose episodes genuinely end early (falls at 2.82 / 9.14 / 9.60 s). No
   clip is a partial/truncated file on disk today (`final_L2_motion.mp4` is 2100 f / 70 s; the
   525-frame partial described in `notes.md` §453-457 is gone). **Verifier's "expected" is wrong,
   not the artifact** (mode E).
3. **Trace duration vs clip duration.** All 13 traces' `t[-1]` agree with the bundle metrics to
   ±0.02 s; no npz is missing (`npz_exists` true for all 13; `final_L2_motion.json` stores its npz
   as a *relative* path while the other 12 store absolute — cosmetic inconsistency only, resolves
   from the repo root).
4. **Verify status (mode E).** `clip_verification.ok == false` in **12 of 13** bundles:
   `01_baseline_stancepd`, `02_slowmo_level_change_quarter_speed`, `99_failure_entry_L2`,
   `99_failure_push90N_diag`, `L0_hold_30s_diag`, `L1_90s_with_pushes`, `L2_cycle_step_diag`,
   `L2_isolated_step`, `L2_shuffle_3steps_diag`, `L2_stance_step_refused_diag`, `final_L1_90s`,
   `final_L2_entry_walk` — all with `"decode failed: cannot convert float infinity to integer"`, so
   the non-black/non-frozen half **never ran**. The only `ok:true` is `final_L2_motion.json`, and it
   verified the temp path `…final_L2_motion.partial.mp4`. Today `ffprobe` decodes all 13 cleanly, so
   the failures are the verifier/toolchain (mode E), not bad media — but the consequence stands:
   **no clip in the index is machine-corroborated by a passing verification**, and 3 of the 12 also
   carry a spurious duration problem. (This confirms and extends claim_audit F6 and `notes.md`
   §E-verifier: same class in 12/13, root cause located at `solo_drill_render.py:154-158`.)
5. **Motion claims.** pelvis-z range and CoM travel computed from each retained npz corroborate
   "near-static" for `final_L1_90s` (2.8 mm after t=8 s) — the index's own correction is right — but
   *refute* the level-change claim for `02_slowmo` (window holds 2.2 mm) and the 0.25x claim for the
   same clip (1.0x). `L0` posture modulation is real (28.6 mm). `final_L2_motion` motion is real
   (0.374 m CoM span, 5 steps, 0 falls).
6. **Step counts.** `L2_shuffle_3steps_diag` label "2-3 steps" vs `steps_completed=4`;
   `final_L2_motion` index "6 steps" vs bundle `step_count=5`; `L2_cycle_step_diag` "one step per
   cycle" vs 3 steps ✓ (and a fall the label omits); `L2_isolated_step` 1 step + 9 refusals ✓;
   `L2_stance_step_refused_diag` 0 steps + 7 refusals ✓.
7. **Push claims.** `L1_90s_with_pushes`: metrics `pushes {n_pushes: 3, recovered: 3}` corroborate
   the *run*; the *artifact* (0–60 s) cannot show push #3 (t=62 s) despite the caption "the push
   windows are overlaid". `99_failure_push90N_diag`: push-90N at t=8.0 s, fall at 9.14 s, clip 6–9.14 s
   ✓.
8. **Run-tag collisions found** (mode F): `BASE_PD`/`BASELINE_PD` (F1), and the docs' `FINAL_L1_90`
   vs `FINAL_L1_push` pair both rendered into differently-named clips from 90 s L1 runs. Also
   `docs/STATUS.md:28` cites `videos/solo_drill/final_L2_motion.mp4 + data/solo_drill/
   final_L2_motion.json` as the evidence for "same-pose PD baseline topples at 8.4 s" — the same
   mis-citation class (STATUS.md is outside this audit's write scope; flagged only).

---

## 4. Clips whose label the retained data contradicts, ranked by how misleading

| rank | clip | claim | what the retained data says | why it misleads |
|---|---|---|---|---|
| 1 | **`final_L2_motion.mp4`** | "95 s, 0 falls, **6** gate-checked steps in a **0.30 m** stance" | 70.000 s / 2100 f; `step_count` 5; run `M_E28f`, stance 0.28 m | The index's designated *deliverable motion drill* — the artifact an operator will watch to judge the milestone, and the L1 correction's other half. All three headline numbers describe a different (superseded) run, so the operator will time the clip wrong, count 5 steps where 6 are claimed, and believe the stance is 0.30 m. No one is currently re-rendering this clip |
| 2 | **`01_baseline_stancepd.mp4`** (F1) | "the same stance pose under pure position control: **it topples**"; index adds "at 8.4 s" | `falls 0`, `fall_times []`, pelvis z 0.7235–0.7279 m for all 11.98 s, 360 f | The claim is the *inverse* of the pixels; it is the contrast that justifies the whole drill programme. Being corrected by LedgerFix as of this audit, but the index still ships the false claim |
| 3 | **`02_slowmo_level_change_quarter_speed.mp4`** | "slow motion **0.25x**: level change down / hold / rise" | 3.033 s for a 3.0 s window ⇒ **1.0x**; the window 3–6 s holds 2.2 mm of pelvis z — the descent (0→2.8 s) and the rise (6→8 s) are both **outside** it | Two independent false claims on the clip cited as the L1 level-change and rubric-H3 slow-motion evidence; also the only clip in the table whose claim is *structurally* unrenderable (mode C) |
| 4 | **`final_L1_90s.mp4`** | "stance hold + **weight shift + level change** (90 s)" | 89.98 s ✓ but 0 steps, 93/185 element timeouts, pelvis z moves **2.8 mm** after t=8 s | The flagship rung clip claims motion it does not contain. Ranked below the others only because `docs/VISUALS.md` already carries the correction in the same row |
| 5 | **`L2_shuffle_3steps_diag.mp4`** | "short shuffle (**2-3 steps**)" | `steps_completed = 4` | Minor and in the safe direction, but it is a bundle label contradicted by its own trace; the clip is not in VISUALS |
| 6 | **`L1_90s_with_pushes.mp4`** | "rung L1 with pushes (**3 x 20 N, all recovered**) … the push windows are overlaid" | run: 3/3 recovered ✓ at t=15/38/**62** s; artifact: 60.033 s of a 89.98 s run, window 0–60 s ⇒ push #3 not visible | The third push — the one the operator would look for — is outside the clip; the "rung L1 90 s" framing also misdescribes a 60 s artifact |
| 7 | **`L0_hold_30s.mp4`** | "the same stance held **60 s**" (and the name has no file) | artifact = 30.033 s of the 59.98 s `FINAL_L0_60` run; the actual file is `L0_hold_30s_diag.mp4` | Index duration and index path both wrong; the run itself is fine (0 falls, 0 steps, both feet planted, 28.6 mm posture modulation) |
| — | `L2_cycle_step_diag.mp4` | "one step per L1 programme cycle" | steps 3 ✓, **falls 1 @ 22.56 s**, window 8–18 s | Label is not false but is silent on the run's failure — a reader would take it for a clean programme sample |
| — | `99_failure_entry_L2.mp4`, `final_L2_entry_walk.mp4`, `99_failure_push90N_diag.mp4`, `L2_isolated_step.mp4`, `L2_stance_step_refused_diag.mp4` | failure / refusal labels | falls 1 @ 2.82 / 9.60 / 9.14 s; refusals 9 / 7 | **All corroborated** by their own bundles |

---

## 5. Conclusions

1. **Mechanism.** The clip→trace attach is a **shared `jobs` table** (`solo_drill_render.py:184-230`
   → `:246-252`) with a **mechanical** `run_tag → results[run_tag]["paths"]["npz"]` lookup and a
   bundle written from that same npz (`:266-282`). A single wrong attach is therefore a **one-off
   hand-entry error** (F1: the table wired `BASE_PD`, the non-toppling twin of `BASELINE_PD`, which
   is wired to no job) — it does **not** silently retarget other clips. **Systemic = YES**, but for
   the adjacent reason: the per-clip **claim text is hand-typed and never compared to the trace**,
   and the only routine that could corroborate a clip (`verify_clip`) **failed `ok:false` on 12/13
   bundles** (mode E, `:154-158`), so no clip in the index is machine-corroborated. Automatic
   sub-modes that do affect several clips at once: the post-hoc `_diag` rename (`:250-251`, breaks
   index paths), the missing `speed` pass-through (`:255-260` vs `:221-225`, kills the 0.25x claim),
   hand-typed windows that exclude the claimed event (`:188-225`), and the verifier bug itself.
2. **Uncorroborated clips.** (a) **12 of 13** solo-drill bundles record their own
   `clip_verification` as failed, so essentially the whole 10b table is uncorroborated; (b) **7**
   clips carry at least one label claim their own retained data or artifact contradicts
   (the 7 ranked rows above); (c) **19** clips in the other indexed families (refs/teacher/env) have
   **no per-clip bundle at all** (family reports only); (d) 2 index entries point at paths that do
   not exist (`L0_hold_30s.mp4`, `99_failure_push90N.mp4`); (e) 10 video files in the solo-drill tree
   (9 baselines probes + `01_baseline_pd_topples.mp4`) are unindexed and unbundled, one of them not
   decodable at all.
3. **F1 is not systemic as a mis-attach, but it is not a lone accident either**: it is one instance
   of a class the pipeline cannot detect (unchecked hand-typed claim + dead verifier), and the class
   already has three further confirmed members on the two headline clips (`final_L2_motion.mp4`,
   `final_L1_90s.mp4`) plus `02_slowmo_level_change_quarter_speed.mp4`.
4. **Single highest-risk clip: `videos/solo_drill/final_L2_motion.mp4`.** It is the only clip the
   index designates as *the* motion drill / deliverable evidence, its three headline numbers in
   `docs/VISUALS.md` (95 s, 6 steps, 0.30 m) belong to a **superseded run** (`M_P30`) while the
   shipped clip is `M_E28f` (70 s, 5 steps, 0.28 m), and — unlike F1 — no agent is currently
   re-rendering or re-labelling it. `01_baseline_stancepd.mp4` (rank 2) is the same severity but is
   being corrected by LedgerFix as of 2026-10-08 14:36.
5. **Cheapest fix that would make all of this machine-detectable**: record `t0`, `t1`, `speed`, the
   trace **sha256** and a machine-readable `claim` block (`{"falls":…, "steps":…, "stance_w":…}`) in
   every bundle, generate the VISUALS row from `data/solo_drill/*.json` instead of hand-copying the
   jobs table, and repair the `nframes` fallback at `solo_drill_render.py:154-158` so the
   non-black/non-frozen check actually runs.

### Artefacts this audit did not touch
`notes.md`, `docs/VISUALS.md`, `src/**`, `scripts/**`, `tests/**`, the running trainer
(`solo-t1-v5`), LedgerFix's in-flight F1 render (`videos/solo_drill/01_baseline_pd_topples.mp4`),
and any pre-existing bundle/report. No render was executed; only `ffprobe` (read-only) and
`numpy.load` on saved traces.
