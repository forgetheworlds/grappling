# Claim audit — ledger & evidence index vs disk (2026-10-08)

Scope: every "done / exists / N passed / numeric" claim in `notes.md` (E1..E28),
`docs/VISUALS.md`, `docs/SOLO_DRILL.md`, `docs/MOTOR_CURRICULUM.md`, and the reports under
`reports/2026-10-08/`. Report-only; nothing edited except this file. The live trainer
(`solo-t1-v5`) and the sim lock were not touched; no training/rendering was run.

Method: ground truth is the filesystem; video format via `ffprobe`; numeric claims re-run from
the stored JSON/checkpoint; test counts re-run via `pytest` (cheap suites only).

## Failures (FALSE / UNVERIFIABLE)

| # | claim | source (file:line) | check run | result | impact if false |
|---|---|---|---|---|---|
| F1 | "the PD baseline (`01_baseline_stancepd.mp4`) topples on the identical pose at 8.4 s — a clean contrast" | `docs/VISUALS.md:21`; `notes.md:656` | read `data/solo_drill/01_baseline_stancepd.json` + its `trace_npz` (`data/drill/BASE_PD_stance_pd_L1_seed0.json`); ffprobe the clip | **FALSE.** The indexed clip is rendered from `BASE_PD_stance_pd_L1_seed0` which does **not** topple: `falls=0`, `pelvis_z` 0.7235–0.7279 m (4 mm range) over the full 11.98 s, tilt 17.8–18.4°. The toppling run is a *different* file, `BASELINE_PD_stance_pd_L1_seed0` (`falls=1`, `fall_times=[8.38]`, pelvis min 0.299 m). The clip's own caption ("...it topples") contradicts its trace. | HIGH — the "clean contrast" that underwrites the drill-stability claim is not in the shipped artifact |
| F2 | `final_L2_motion.mp4` is "**95 s**, 0 falls, **6** visible gate-checked steps ... in a 0.30 m-wide stance" | `docs/VISUALS.md:21` | ffprobe the mp4; read `data/solo_drill/final_L2_motion.json` (`metrics`, `step_count`, `steps`) | **FALSE.** Clip is **70.000 s / 2100 frames** (960x720, h264/yuv420p 30 fps). Bundle `step_count = 5` (steps at t=15.9/29.34/45.9/50.72/64.48); the independent check (E20) also found 5. The ledger itself (E22/E20) says 70 s / 5 steps — VISUALS contradicts the ledger. Bundle rubric `ship gate=FAIL`. | HIGH — misdescribes the best motion artifact and overstates it by 25 s / one step |
| F3 | `videos/solo_drill/L1_90s_with_pushes.mp4` is a rung-L1 90 s clip | `docs/VISUALS.md:21`; `data/solo_drill/L1_90s_with_pushes.json` (`config.seconds=90`, run `FINAL_L1_push`=89.98 s) | ffprobe the mp4 | **FALSE.** File is **60.033 s / 1801 frames**; its own bundle declares the run as 89.98 s and renders that npz. Truncated artifact (same mid-write-truncation class E22 documented). | MED — indexed evidence is 2/3 of the run it claims |
| F4 | "L0 = the same stance held **60 s**" | `docs/VISUALS.md:21` | ffprobe `videos/solo_drill/L0_hold_30s_diag.mp4`; read its bundle (`config.seconds=60`, trace `FINAL_L0_60`=59.98 s) | **FALSE.** The indexed clip is **30.033 s / 901 frames** (480x360). Also the indexed name `L0_hold_30s` omits the actual `_diag` suffix. | MED — duration claim vs artifact |
| F5 | E26 v3 gate "→ NOT CERTIFIED (**2/7**)" | `notes.md:274` | read `data/solo/metrics/t1_v2_monitor_2000896.json` (`reasons`, `steps=2000896`) | **FALSE.** The stored monitor lists **6 FAIL / 1 PASS = 1/7** (`fall_rate` 0.104 FAIL; only `fall_rate_heldout` 0.083 PASS). The per-metric numbers in E26 are otherwise correct. | LOW-MED — misstates the gate result |
| F6 | E14 "**Every** clip is ffprobe-verified ... plus a non-black/non-static check, with a `data/solo_drill/<name>.json` evidence bundle" | `notes.md:639-641` | read `clip_verification` in each bundle | **FALSE.** Four bundles record the verification as **failed**: `01_baseline_stancepd`, `L1_90s_with_pushes`, `L0_hold_30s_diag`, `02_slowmo_level_change_quarter_speed` all have `"ok": false, "problems": ["decode failed: cannot convert float infinity to integer"]` (no luma/frame-diff check). | MED — "verified" is claimed where the bundle records failure |
| F7 | E18 "w=0.30 → **6 steps, 0 falls, 13.7 s/step**" | `notes.md:490` | read `data/drill/motion_widths.json` (the cited file); scan every `data/drill/*.json` for a 0.30 run | **UNVERIFIABLE / effectively FALSE.** `motion_widths.json` has rows for 0.21/0.28/0.35/0.42/0.495 — **no 0.30 row**. The on-disk 0.30 run (`M_S30_feasible_L3_seed0.json`) **fell** (`falls=1`, 2 steps, 41.82 s). (The other three width rows in E18 match the file exactly.) | MED — a width-table entry with no artifact behind it |
| F8 | Indexed path `videos/solo_drill/99_failure_push90N.mp4` | `docs/VISUALS.md:21` | glob `videos/solo_drill/` | **FALSE (path).** No such file; the actual disk file is `99_failure_push90N_diag.mp4`. (The other five names in the row resolve.) | LOW — broken index reference |
| F9 | E19 delivered run "**slip_max 0.019 m**" | `notes.md:471` | read `data/drill/M_D28c_feasible_L3_seed0.json` `.metrics.slip` | **UNVERIFIABLE.** Stored slip: `max_tick_slip_m` 0.00051 m, `max_load_drift_m` 0.0065 m. No 0.019 m appears in the file. (Other E19 numbers — 69.98 s, 0 falls, 5/5 steps, com_travel 2.44 m, span 0.374 m, margin_min −0.0257, 24/3500 ticks — all match.) | LOW — one unsupported metric in an otherwise-correct row |
| F10 | E13 "`stance_report.json` read directly" → built stance **0.315 m × 0.351 m**, pelvis 0.720 m, margin **+0.077 m**, tilt **9.2°**, "**18 programme cycles**" | `notes.md:651-656` | read `data/drill/stance_report.json`; read `data/drill/FINAL_L1_90_feasible_L1_seed0.json` | **UNVERIFIABLE (superseded on disk).** `stance_report.json` now reports width 0.4945 m, depth 0.2413 m, pelvis 0.728 m, margin 0.0861 m, tilt 14.9°; `FINAL_L1_90` has `cycles_done=30`. E13's numbers describe an earlier stance/run that was overwritten. (L1 "90 s, 0 falls" is independently CONFIRMED by `FINAL_L1_90`=89.98 s, falls 0.) "18 cycles" was already corrected in E17. | LOW — stale snapshot, no longer checkable |
| F11 | E11 STAND_UP scorer **0.324** | `notes.md` (E11 scorer line) | read `data/teacher_stats.json` | **MISMATCH.** `teacher_stats.json` (5 seeds) gives STAND_UP `scorer_mean=0.5768`; E13 records the same rebuild moving it 0.324→0.577. E11 appears to predate the rebuild. All other E11 scorer means match the JSON (SNAPDOWN 0.808, BODY_LOCK 0.679, SPRAWL 0.559, STANCE 0.547, SINGLE_LEG 0.361, DOUBLE_LEG 0.324); stay-up numbers match (SNAPDOWN 0.956 / STANCE 0.758 / worst seed 0.889). | LOW — stale pre-rebuild figure |

### Test-count claims (grown, still green — not lies)

Re-run on this host (cheap suites only, no training/rendering):

| claim | source | claim | re-run now | verdict |
|---|---|---|---|---|
| `tests/test_solo.py` | E15 `notes.md:587` | 32 passed | **36 passed** | grown by 4 |
| `tests/test_drill.py` | E13 `notes.md:651` | 14 passed | **21 passed** | grown by 7 |
| `tests/test_teacher.py` | E13 `notes.md:671` | 18 passed | **18 passed** | match |
| full suite | E25 `notes.md:300` | 175 passed, 0 failed | **181 passed, 0 failed** | grown by 6 |
| full suite | "Phase 5 infra" `notes.md:966` | 89 passed | **181 passed, 0 failed** | grown |
| scorer | `reports/2026-10-08/scorer.md` | 19 tests | **19 passed** | match |

No suite is red; every count mismatch is growth, not regression. `E11`'s "12 tests / full suite
115" is likewise an older snapshot.

### UNVERIFIED-ONLY-WATCHABLE

- `docs/VISUALS.md:21` visual verdicts (feet pixel-stationary ±3 px; no cuts; "physically nearly
  static"; L2 "no pops", "floor+shadows visible") — from `reports/2026-10-08/l1_clip_visual_check.md`
  and `l2_clip_visual_check.md`; not re-watched here. Connate stored metrics corroborate the
  no-fall/no-step parts (`final_L1_90s`: falls 0, steps 0, timeouts 93; `final_L2_motion`: falls 0).
- E4/`env_videos_frame_check.md`: "robots topple", "supine arms hold stiffly", "OOB offender glides
  with planted feet" — pixels only; report exists, not re-verified.
- E13 PD "topples at 8.4 s" as *screen* claim — see F1 (contradicted by the shipped clip's trace).

## Ranked findings that matter

1. **F1 — the baseline-contrast clip contradicts its own claim.** The video indexed as the
   "PD baseline topples" proof is rendered from the run that stays up for 12 s; the run that
   actually topples (8.38 s) is a different file. This is the exact failure class the mandate
   flagged: an indexed artifact whose caption/metrics do not agree.
2. **F2 — VISUALS 10b misstates the main motion artifact** (95 s / 6 steps vs the real 70.000 s /
   5 steps) and omits that its bundle scores `ship gate=FAIL`. The doc overstates the best
   evidence on disk.
3. **F6 — four clip bundles record their own verification as failed** (`ok:false`) while E14
   claims every clip was ffprobe+non-static verified.
4. **F3 — `L1_90s_with_pushes.mp4` is a 60 s truncation** of an 89.98 s run (same truncation
   failure E22 documented, still present in an indexed artifact).
5. **F4/F8 — VISUALS 10b duration/path errors** (L0 "held 60 s" vs a 30 s clip; missing
   `99_failure_push90N.mp4`).
6. **F5 — E26 states 2/7 where the stored monitor says 1/7.**
7. **F7 — E18's width-0.30 row has no artifact and the on-disk 0.30 run fell.**
8. **F9/F10/F11 — unsupported or superseded single numbers** (slip_max 0.019; E13's stance
   snapshot + 18 cycles; E11 STAND_UP scorer 0.324).

## Could not check, and why

- **Pixel content of any clip** — only `ffprobe` metadata and the bundles' own
  `clip_verification` JSON were used; no frames were extracted/watched (out of scope, and the
  verification helper itself is what recorded `ok:false` for four clips).
- **Throughput claims** (E15 "594–615 steps/s", "175–215 steps/s"; E3/E4 step-rate numbers) —
  not re-benchmarked to avoid contending with the live trainer. `data/solo/metrics/holdability.json`
  independently records `throughput_steps_per_s=614.7`, consistent with E15.
- **E1's eval numbers** — corroborated by `reports/2026-10-08/rl_infra.md` §6.2 (verbatim eval
  output); the raw `eval_ppo.py` log is not stored on disk, so only the report corroborates it.
- **Code-level claims** in E2/E10/E27 (e.g. residual double-map, `_smooth` zero-phase) — not
  "done/exists" artifact claims; not re-derived.
- **Git-history claims** (E22's restore commit `ef8c1b2`, provenance commits) — repo history, not
  disk artifacts.
- **`videos/solo_drill/final_continuous_drill.mp4`** — confirmed still absent by glob; this is
  already recorded as corrected in E28, not a new finding.

## Disposition (2026-10-08, LedgerFix) — one row per defect

All 11 defects are dispositioned. Inline corrections were placed at the offending sentence in
`notes.md` (kept verbatim, `+[CORRECTED: …]` appended, the E23b/E28/E29 convention) or in the
`docs/VISUALS.md` 10b row; the consolidated entries are `notes.md` **E32** (a second entry after
E31, which the literature check took concurrently). Nothing was deleted from `notes.md`.

| # | claim (short) | source | disposition | evidence / action |
|---|---|---|---|---|
| F1 | PD baseline clip "topples at 8.4 s" | `docs/VISUALS.md:21`, `notes.md` E13 | **RE-RENDERED** | indexed clip is from `BASE_PD_…` (falls=0, pelvis z 0.7235–0.7279 m); the toppling run `BASELINE_PD_…` (falls=1 @ 8.38 s) **is** on disk as a cached npz → re-rendered `videos/solo_drill/01_baseline_pd_topples.mp4`, ffprobe 8.400000 s / 252 frames, `verify_clip` ok, non-static; bundle `data/solo_drill/01_baseline_pd_topples.json`; old clip kept, relabelled as a stable 12 s base-PD hold + `caption_correction` in its bundle. Suite caption + wiring RESOLVED by EvidenceFix (2026-10-08): `CLIP_SPECS` carries the corrected caption/title and the toppling run now has its own `BASELINE_PD` job (`01_baseline_pd_topples.mp4`), so a future suite run reproduces the contrast from code; the durable check is `python scripts/evidence_check.py table` (trace-derived expectations + numeric claim checks, nonzero exit on contradiction). |
| F2 | L2 motion "95 s / 6 steps / 0.30 m" | `docs/VISUALS.md:21` | **CORRECTED-INLINE** | clip is 70.000 s / 2100 frames (ffprobe); bundle `step_count=5`; figures are the superseded `M_P30` run, shipped clip is `M_E28f` (69.98 s / 5 steps / 0.28 m); row now also records `ship gate=FAIL`. |
| F3 | `L1_90s_with_pushes` "90 s" | `docs/VISUALS.md:21` | **CORRECTED-INLINE** | ffprobe 60.033333 s / 1801 frames — a deliberate t=0–60 s window of the 89.98 s `FINAL_L1_push` run (not a mid-write truncation); the t=62 s third push is not in the clip. |
| F4 | "L0 held 60 s" | `docs/VISUALS.md:21` | **CORRECTED-INLINE** | run 59.98 s, clip `L0_hold_30s_diag.mp4` 30.033333 s / 901 frames @ 480x360 (ffprobe); deliberate t=0–30 s diagnostic window; name now carries `_diag`. |
| F5 | E26 gate "2/7" | `notes.md` E26 | **CORRECTED-INLINE** | `t1_v2_monitor_2000896.json` = 6 FAIL / 1 PASS = 1/7, `not_certified`. |
| F6 | E14 "every clip is verified" | `notes.md` E14 | **CORRECTED-INLINE** + clip dispositions | all bundles indexed in VISUALS 10b record `ok:false` (12/13 bundles overall). Root cause is the VERIFIER (imageio `nframes: inf` → OverflowError; `isfinite` guard added in `103dcee` at 09:08, after the bundles). Re-verified with the current `verify_clip`: all 8 indexed clips `ok:true`, non-static. NO artifact is KNOWN-BAD: the 3 "duration mismatch" clips are complete episodes that END IN A FALL, and their frames match `round((min(t1, trace_end) - t0) * 30) + 1` exactly (56 / 95 / 277). |
| F7 | E18 "w=0.30 → 6 steps, 0 falls" | `notes.md` E18 | **UNVERIFIABLE** (inline) | `motion_widths.json` has no 0.30 row; the on-disk `M_S30` run FELL (falls=1 @ 41.82 s, 2 steps). Not rewritten — a different measurement that contradicts the claim. |
| F8 | indexed `99_failure_push90N.mp4` | `docs/VISUALS.md:21` | **CORRECTED-INLINE** | file is `99_failure_push90N_diag.mp4`; the `_diag` suffix is appended by `cmd_suite` (script:250-251) after the jobs table names the clip. |
| F9 | E19 "slip_max 0.019 m" | `notes.md` E19 | **UNVERIFIABLE** (inline) | `M_D28c…json` stores max_tick_slip_m 0.00051 m / max_load_drift_m 0.0065 m; no 0.019 m. |
| F10 | E13 stance snapshot + "18 cycles" | `notes.md` E13 | **UNVERIFIABLE** (inline) | current `stance_report.json` = 0.4945 × 0.2413 m, pelvis 0.728 m, margin 0.0861 m, tilt 14.9°; `FINAL_L1_90` cycles_done=30; superseded-on-disk (18 cycles already corrected in E17). |
| F11 | E11 STAND_UP scorer "0.324" | `notes.md` E11 | **UNVERIFIABLE** (inline) | `teacher_stats.json` STAND_UP scorer_mean = 0.5768; 0.324 is the pre-rebuild value and DOUBLE_LEG's current value. |

### WIRING MECHANISM (Main's question: one-off or systematic?)

The tag→npz attach in `cmd_suite` is mechanical and sound (`results[run_tag]["paths"]["npz"]`,
script:187-232) — no automatic mis-attach retargets other clips. The defect is hand-typed claim text
with no claim-vs-trace assertion (`BASE_PD` vs `BASELINE_PD` near-duplicate tags). Independent
corroboration and three ADDITIVE automatic mechanisms (the `*_diag` rename, the missing `speed`
argument making the "0.25x" claim unrenderable, and unchecked `t0/t1` windows) are in
`reports/2026-10-08/clip_wiring_audit.md` (EvidenceAudit); all are reflected in the 10b row.

### Two statements that must not be conflated (Main)

1. **The verification itself was broken.** `verify_clip` recorded `ok:false` on **12 of 13** bundles
   ("cannot convert float infinity to integer"); the `isfinite` guard landed in commit `103dcee`
   (09:08) after the bundles were written (pre-`103dcee`: `n = int(meta.get("nframes") or
   expect_frames)` → `int(inf)`). Consequence: **no clip in the index was machine-corroborated by a
   passing verification**, and 3 of the 12 also carry a spurious duration problem. Re-running the
   CURRENT `verify_clip` passes all 12, so the records are historical.
2. **F1 is a one-off hand-entry error, not a systemic binding bug.** The name→npz binding is a shared
   `jobs` table with a mechanical tag→npz lookup, so a wrong attach does not propagate to other clips.
   The *systemic* part is that per-clip **claims are hand-typed free text never compared to the
   trace**. F1 must not be presented as evidence of a wiring bug in either direction.

### Contradicted / uncorroborated (EvidenceAudit, folded in)

- **7 clips are contradicted by their own retained data**, ranked: `final_L2_motion` (index 95 s /
  6 steps / 0.30 m vs artifact 70.000 s / 5 steps / 0.28 m — superseded `M_P30` vs shipped `M_E28f`),
  `01_baseline_stancepd` (F1), `02_slowmo_level_change_quarter_speed` (claims 0.25x, is 1.0x, and the
  window holds neither descent nor rise), `final_L1_90s` (claims motion; 2.8 mm pelvis-z after t=8 s),
  `L2_shuffle_3steps_diag` ("2-3 steps" vs 4; bundle-only), `L1_90s_with_pushes` (F3), `L0_hold_30s`
  (F4); plus `L2_cycle_step_diag`, silent on its run's fall at 22.56 s.
- Disposition of those 7: rows 1/3/4/6/7 **CORRECTED-INLINE** to the artifact's real numbers (not
  re-rendered — the clips are honest footage of their runs); row 5 handed to EvidenceFix.
- **19 clips in the other indexed families** (refs 7, teacher 7, env 5) have **no per-clip bundle
  anywhere in `data/`**, and 9 solo-drill baseline probe clips remain unindexed/unbundled.
- **Code fixes are NOT in this change-set**: Main routed `verify_clip`'s root cause, the `speed`
  pass-through, the `_diag` index paths and the label-integrity check to EvidenceFix. LedgerFix
  reverted the single caption edit it had briefly made to `scripts/solo_drill_render.py`, and handed
  EvidenceFix the exact replacement text.
- **Flagged, not fixed:** `docs/STATUS.md:28` cites `final_L2_motion.mp4` + its bundle as evidence for
  the PD-baseline topple claim — the same mis-citation class as F1, outside this change-set's scope.

### Also found while fixing (not in the 11)

- `02_slowmo_level_change_quarter_speed.mp4` is **1.0x, not 0.25x** (3.033 s / 91 frames for a 3.0 s
  window) and its t=3–6 s window contains only the hold — `cmd_suite` never passes `speed`.
- The 10b "0.30 m stance / 0.61× reference" figures belong to the superseded `M_P30` run.
- `final_L2_motion.json`'s `clip_verification` verified the **`.partial.mp4`** temp path, not the
  final filename.

### ffprobe (the one allowed use) — measured durations vs claims

```
01_baseline_stancepd.mp4      duration 12.000000  nb_frames 360     (claim: "topples at 8.4 s" — trace 11.98 s, falls=0)
L1_90s_with_pushes.mp4        duration 60.033333  nb_frames 1801    (claim: 90 s run — clip covers t=0–60 s)
L0_hold_30s_diag.mp4          duration 30.033333  nb_frames 901     (claim: held 60 s — run is 59.98 s)
final_L2_motion.mp4           duration 70.000000  nb_frames 2100    (claim: 95 s / 6 steps — real 70 s / 5 steps)
final_L1_90s.mp4              duration 90.000000  nb_frames 2700    (claim: 90 s — matches the 89.98 s run)
01_baseline_pd_topples.mp4    duration 8.400000   nb_frames 252     (NEW: trace 8.38 s, falls=1 @ 8.38 s)
```
