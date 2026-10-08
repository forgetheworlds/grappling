# repo_trim.md — repo-wide inventory and evidence-based deletion (2026-10-08)

**Task:** trim `/home/ubuntu/grappling` to the drill mandate (`docs/SOLO_DRILL.md` S1–S10 / T1–T7),
using *evidence per item* (reference map), never names.

**References used in this report**

| name | value | meaning |
|---|---|---|
| **PIN** | `61e8bd1a5723f218ce4f7b1b882ece3af9f24e85` | HEAD when the baseline test count was taken (immutable) |
| **DELREF** | `72956f3` | HEAD immediately *before* the deletions → every deleted path is recoverable with `git show 72956f3:<path>` |
| **TRIM** | `46fb84f` | the deletion commit itself |
| HEAD now | `4875e73` (local) | after concurrent peer commits (E30/E31/E33, T2 gate, T1-gate calibration); the after-count worktree was taken at `96478e1` |

Method: a reference map was built from (a) real import statements (`from X import`, `import X.Y`)
across all `.py/.mjs`, (b) `scripts/*.py` and `workflows/*.js` entry points, (c) `tests/**`,
(d) `docs/**` + `notes.md` + `reports/**` mentions, (e) generated catalogs (`docs/VISUALS.md`,
`data/drill/*.json`, `data/solo/metrics/*_summary.json`). Every candidate was then classified into
exactly one bucket. Nothing was deleted on the strength of a name.

---

## 1. The four buckets

| bucket | verdict | tracked size |
|---|---|---|
| **MANDATE-CRITICAL** | keep | `src/solo` 0.28 MB · `src/drill` 0.27 MB · `src/teacher` 0.14 MB · `src/retarget` 0.06 MB · `src/grapplemap` 0.02 MB · `src/rl/{ppo,net,checkpoint}` 0.03 MB · `robots/**` 35 MB (untouched) · `scripts/solo_*` + `scripts/run_solo_drill.py` · `tests/{test_solo,test_drill,test_retarget,test_grapplemap,test_teacher}.py` · `videos/solo_drill/**` 43 MB · `data/drill/{FINAL_*,L2_*,BASE_PD_*,SLOWMO_*,FAIL_*,M_D28c,M_E28f}*` + summaries · `data/solo_drill/**` · `data/refs_video/**` · `data/grapplemap/**` · `data/support_envelope.json` · `data/{backdet,scorer}_calibration.json` · `docs/SOLO_DRILL.md`, `docs/VISUALS.md`, … |
| **DEFERRED-BUT-WORKING → now a DELETION bucket** (operator decision, §3) | **could not be deleted this pass** — it is an import-level prerequisite of the mandate path; see §3 | `src/wrestling` 0.11 MB · `src/scorer` 0.07 MB · `src/rl/{obs,privileged,reward,scripted,curriculum}` 0.06 MB · `tests/{test_wrestling,test_rl,test_scorer}.py` 0.08 MB · `videos/env/**` 7.9 MB · `videos/refs/**` 11.7 MB · `data/refs/**` 0.8 MB · 16 non-solo `scripts/*.py` 0.23 MB |
| **DEAD** | delete | `workflows/phase3_teacher.mjs` (6.7 KB), `workflows/wave1_assets.mjs` (7.8 KB), `videos/solo_drill/l2_stills/*.png` (131 KB) |
| **GENERATED-BLOAT** | delete | `data/drill/M_*_feasible_L3_seed0.npz` × 41 (73.9 MB), `data/solo/metrics/*.jsonl` × 12 (31.5 MB), `videos/solo_drill/l2_stills/*.png` (131 KB), `__pycache__/` + `.pytest_cache/` × 101 (3.0 MB, untracked/ignored) |

### Why `src/wrestling` is not in the DEAD or deletable bucket

The two-robot package is **import-closed into the mandate** (this is the finding, not an assumption):

```
src/solo/fall.py     → from wrestling.backdet import BackDetConfig, BackToMatDetector   (termination detector, SOLO_DRILL §4)
src/solo/train.py    → rl.ppo, rl.net, rl.checkpoint
src/solo/baselines.py→ rl.net
src/rl/net.py        → from wrestling.env import ACT_SLICE, N_JOINTS
src/rl/vec.py        → from wrestling.env import EXCHANGE_TIMEOUT, MATCH_CLOCK, N_JOINTS, WrestlingEnv
src/rl/trainer.py    → from wrestling.env import STEP_DT, load_wrestling_model
src/rl/rollout.py, src/rl/obs.py, src/rl/scripted.py, src/rl/privileged.py, src/rl/reward.py → wrestling.env
```

Deleting `src/wrestling/env.py` breaks `import src.rl.ppo` (the acceptance smoke) through
`src/rl/net.py`. `src/wrestling/backdet.py` is a **direct** mandate dependency of `src/solo/fall.py`.

---

## 2. Deletion manifest (57 files, 105.55 MB tracked + 3.0 MB untracked caches)

All entries are **tracked** (recoverable from `72956f3`) unless marked otherwise.
Commit: `46fb84f`.

| # | path(s) | size | bucket | reference evidence | tracked |
|---|---|---|---|---|---|
| 1 | `data/drill/M_{B24,B24f,B28,B30,D26d,D28a,D28b,E24,E24n,E28,E28n,F24,F28,F30,P28,P30,Q28,Q30,S21,S28,S28b,S30,T28,T30,T35,V28,V28b,V28c,g1,j21,k1,ns,r3,v21,v24,v24r,v26,v26r,w21v,w24v,w28v}_feasible_L3_seed0.npz` (41) | 73.94 MB | GENERATED-BLOAT | raw per-run traces of the D1 width sweep. **No** individual name reference in src/scripts/tests/docs. The measured results survive in the retained `M_*.json` (config/metrics/events/provenance), `data/drill/motion_widths.json`, `data/drill/motion_singles.json` and `reports/2026-10-08/drill_motion.md` §1 table. Only collective mention: `drill_motion.md:50` “`M_*` traces carry the single-width runs” (historical report — see §7). **Kept:** `M_D28c_*.npz` (source trace of the delivered clip `videos/solo_drill/final_L2_motion.mp4`) and `M_E28f_*.npz` (cited by `data/solo_drill/final_L2_motion.json`). | ✅ |
| 2 | `data/solo/metrics/{balance_random_init_policy_s0,balance_stand_hold_s0,balance_t1_v2_monitor_1101824_s0,balance_t1_v2_monitor_2000896_s0,balance_t1gate_random_init_policy_s0,balance_t1gate_stand_hold_s0,balance_t1gate_zero_action_s0,balance_zero_action_s0,locomotion_fall_forward_s0,locomotion_random_init_policy_s0,locomotion_stand_hold_s0,stance_squat_repeat_s0}.jsonl` (12) | 31.47 MB | GENERATED-BLOAT | zero references to any of these 12 basenames anywhere in src/scripts/tests/docs/reports. They are the per-step traces behind the retained aggregate `.summary.json` + `t1_gate_baselines.json` + `{probes,baselines}_summary.json`; `reports/2026-10-08/solo_env.md` §7 states the producing command is “deterministic and re-runnable (`--no-video` for metrics only)”. All were ≥44 min stale (v2 run stopped); the live service does not write these names. | ✅ |
| 3 | `videos/solo_drill/l2_stills/frame_t002.00.png`, `frame_t030.00.png` | 131 KB | DEAD + GENERATED-BLOAT | zero references; the two L2 stills have “no index entry” (`reports/2026-10-08/clip_wiring_audit.md:119`). | ✅ |
| 4 | *(reversed — see below)* `data/references/yt_gBAhX5t-GW4/derived/tools/bench_pose.py` | 1.8 KB | — | zero *static* references and the only reference-pipeline tool not named in `docs/references/yt_gBAhX5t-GW4_index.md`, **but** it is part of the now-critical video-imitation pipeline (`data/references/yt_gBAhX5t-GW4/derived/tools/*.py`, §5b) — hand-run entry points count as referenced. Restored. | ✅ restored |
| 5 | `workflows/phase3_teacher.mjs` | 6.7 KB | DEAD (two-robot teacher orchestration) | one reference: `notes.md:119`, a generic lesson about workflow-script syntax. The phase-3 two-robot teacher composition it launched is off-mandate. | ✅ |
| 6 | `workflows/wave1_assets.mjs` | 7.8 KB | DEAD | **zero** references. One-off Phase-0/1 bring-up campaign (venv + MuJoCo + GrappleMap + G1 model) from 2026-10-07; its outputs (`data/refs/`, `data/grapplemap/`) are tracked and retained. | ✅ |
| 7 | `__pycache__/`, `.pytest_cache/` (101 files) | 3.0 MB | GENERATED-BLOAT | regenerable caches, gitignored/untracked. | ❌ untracked |

**Reversed after measurement (do not treat as deleted):**

1. `data/drill/BASELINE_PD_stance_pd_L1_seed0.{json,npz}` were first staged for deletion as a “superseded
   duplicate” of `BASE_PD_*`. A re-grep after staging showed they are **ledger-cited**: `notes.md:214` names
   the run as the toppling PD baseline (`falls=1 @ 8.38 s`), and `data/solo_drill/01_baseline_pd_topples.json:4-5`
   (a bundle written concurrently by the claim-audit peer) uses them as its `trace_npz`/`run_json` for
   `videos/solo_drill/01_baseline_pd_topples.mp4`. Both files were restored from `72956f3` and are present in HEAD.
   **Lesson: a concurrent peer can cite a file mid-trim; re-grep immediately before the commit, not only before staging.**
2. `data/references/yt_gBAhX5t-GW4/derived/tools/bench_pose.py` was classified DEAD on static evidence (zero
   references; the only reference-pipeline tool not named in `docs/references/yt_gBAhX5t-GW4_index.md`). The
   operator's **video-imitation keep-list** (received after the deletion commit) makes the whole
   `data/references/yt_gBAhX5t-GW4/**` tree — including every `derived/tools/*.py` — the milestone's critical
   path, with hand-run entry points counting as referenced. It was **re-bucketed** from DEAD to
   MANDATE-CRITICAL and restored from `72956f3`.

**Re-bucketing check against the video-imitation keep-list (all other items):** `data/refs_video/*.npz`,
`data/refs/*.npz`, `src/retarget/**` (incl. `landmarks.py`), `data/references/yt_gBAhX5t-GW4/**`
(`derived/tools/*.py`, `derived/analysis/**`, `derived/{transitions,stance_spec,refs_crosscheck}.json`) were
already bucketed MANDATE-CRITICAL / kept and were **not** deleted. `videos/refs_video/**` does not exist on
this checkout (`ls videos/` → `env refs solo_drill teacher`), so there was nothing to protect; the G1-side
reference renders that do exist are `videos/refs/**` (tracked, kept).

---

## 3. The two-robot pipeline: what the operator asked to delete and why it could not go this pass

Operator decision (2026-10-08, relayed by the orchestrator): the DEFERRED-BUT-WORKING bucket is a
**deletion** bucket — two-robot scene, exchange loop, self-play league, opponent pools, two-robot teacher
composition, their tests/scripts/metrics — *unless it is a prerequisite of the solo-drill path*.

Measured outcome: **almost all of it is a prerequisite**, at the import level, of modules the operator
explicitly keeps (`src/rl`) or of the mandate’s own termination logic (`src/solo/fall.py` → `wrestling.backdet`).
See the import block in §1. Additionally, the operator ordered `src/rl/{vec,trainer,rollout}.py` kept as
**PATTERN-SOURCE** (the only parallel-env-worker implementation; `src/rl/vec_solo.py` is being built from it),
and those three transitively require `src/wrestling/env.py`, `src/rl/{obs,privileged,reward,scripted,curriculum}.py`.

Consequently the following were **not deleted** and remain as the operator's decision list (§5):
`src/wrestling/**`, `src/scorer/**`, `src/rl/{obs,privileged,reward,scripted,curriculum}.py`,
`tests/{test_wrestling,test_rl,test_scorer}.py`, `videos/env/**`, `videos/refs/**`, `data/refs/**`,
and the 16 non-solo `scripts/*.py`.

### Ordered excision plan (for a future pass, once the solo vec backend lands)

1. Port the subprocess-vec pattern into `src/rl/vec_solo.py` (in progress by another agent); delete
   `src/rl/vec.py`, `src/rl/trainer.py`, `src/rl/rollout.py`.
2. Inline `ACT_SLICE`/`N_JOINTS` in `src/rl/net.py` (2 constants) → removes the `wrestling.env` edge.
3. Split `src/wrestling/env.py`: keep `backdet` + the reference-trace loader used by `src/solo`; drop
   `WrestlingEnv`/`StandHold`/`ReferenceReplay` (the two-robot exchange loop) with `src/rl/{obs,privileged,reward,scripted,curriculum}`.
4. Then delete `tests/test_wrestling.py` (its `backdet` tests must first move to `tests/test_solo.py`, which
   owns the same detector through `src/solo/fall.py`), `scripts/{train_ppo,eval_ppo}.py`, `videos/env/**`,
   `videos/refs/**`, `data/refs/**`, `data/{backdet,scorer}_calibration.json` (re-home the calibration tests).
5. Every step is gated by the acceptance smoke (`import src.solo.train, src.solo.env, src.solo.eval,
   src.drill.scheduler, src.teacher.solo_scene, src.rl.ppo`) plus the full suite.

**Ledger citations that pin parts of this bucket (do not delete):** `notes.md:1084-1088` cites
`scripts/eval_ppo.py` + `checkpoints/rl/smoke_stage_c.pt` for the measured “best policy stands 13.1%”
result; `notes.md` E4 cites the five `videos/env/*` rule demos as verified evidence, and the
back-to-mat semantics they demonstrate are inherited by SOLO_DRILL §4.

---

## 4. PATTERN-SOURCE-DO-NOT-DELETE-YET

| file | size | reusable machinery | replacement dependency |
|---|---|---|---|
| `src/rl/vec.py` | 17.4 KB | subprocess (multiprocessing) env-worker backend — the only parallel-env implementation in the repo; solo PPO currently hardcodes `n_envs=1` | `src/rl/vec_solo.py` (being written now) |
| `src/rl/trainer.py` | 27.7 KB | resumable training loop: atomic checkpoint + RNG restore + curriculum stage transitions + stats | solo trainer (`src/solo/train.py`) is a reduced copy |
| `src/rl/rollout.py` | 13.4 KB | rollout collection + GAE(λ) finalisation over a vec backend | same |

Also kept by the same rule (only verification of the machinery above): `tests/test_rl.py`.
Also kept as pattern source for the mandate itself: `scripts/render_env_videos.py` (42 KB) — the HUD/mat
overlay renderer whose style `src/solo/video.py:53` and `src/drill/video.py` follow, and the generator of
`videos/env/**` (ledger-cited, E4).

---

## 5. KEPT-OFF-MANDATE decision list (for the operator)

Each row is deferred-but-working work that survived this pass, with its size and what it implements.

| item | size | what it implements | why it survived |
|---|---|---|---|
| `src/wrestling/env.py` | 31.7 KB | two-robot MuJoCo env: scene builder, exchange loop, match clock, OOB/draw rules, scripted opponents (`StandHold`, `ReferenceReplay`), reference-trace loader | import prerequisite of `src/rl/{net,obs,privileged,reward,rollout,scripted,trainer,vec}.py`; **must not be deleted before the plan in §3** |
| `src/wrestling/backdet.py` | 14.0 KB | calibrated back-to-mat detector | **mandate**: `src/solo/fall.py` imports it (SOLO_DRILL §4 termination) |
| `src/scorer/**` (7 files) | 0.07 MB | technique-validity scorer (7 techniques, phase bands, calibration) | `src/teacher/episode.py` imports `scorer.scorer`/`scorer.config`; the teacher is the S5 demonstration source |
| `src/rl/{obs,privileged,reward,scripted,curriculum}.py` | 0.06 MB | actor obs, critic privileged obs, stage rewards, scripted opponents, PISTY curriculum | transitive deps of the kept pattern-source modules |
| `tests/test_wrestling.py` | 25.7 KB | 19 tests: match rules **and** `test_backdet_persistence_and_negatives`, `test_backdet_batch_matches_streaming`, `test_calibration_json_matches_default_config` | the backdet tests are mandate verification (detector used by `src/solo/fall.py`); the module is kept |
| `tests/test_rl.py` | 39.4 KB | 35 tests incl. `test_checkpoint_roundtrip_and_atomicity`, `test_ppo_update_and_lr_schedule`, `test_rng_state_roundtrip`, `test_bc_warm_start` | mandate verification for `rl/{ppo,checkpoint}` (named prerequisites) |
| `tests/test_scorer.py` | 16.3 KB | scorer calibration/monotonicity | verifies `src/scorer` used by the teacher |
| `videos/env/**` | 7.9 MB | five env-rule demos (draw / back-event / non-terminal knees / ambiguous / OOB) | `notes.md` E4 ledger entry cites all five as rendered+verified; the back-to-mat semantics are inherited by the solo env |
| `videos/refs/**` | 11.7 MB | 7 retargeted technique references | S5 uses DOUBLE_LEG/SINGLE_LEG; `data/refs/*.npz` twins are loaded by tests and the teacher |
| `data/refs/**` | 0.8 MB | 7 reference traces + `STAND_UP.airborne` | loaded by `src/teacher/**`, `tests/test_teacher.py`, `tests/test_rl.py` |
| `data/refs_video/**` | 1.0 MB | 12 video-retargeted G1 tracks + `retarget_summary.json` | produced the 41-parameter stance spec the drill stance is matched to (`docs/STATUS.md:30`) |
| `data/references/**` (tracked part) | 0.25 MB (dir 201 MB, mostly untracked media) | operator-reference pipeline: 14 tools, `stance_spec.json`, `transitions.json`, `refs_crosscheck.json` | `stance_spec.json` is cited by `tests/test_drill.py` and `reports/2026-10-08/drill.md` |
| 16 non-solo `scripts/*.py` | 0.23 MB | `audit_support_envelope`, `build_refs`, `build_standup_ref`, `calibrate_backdet`, `calibrate_scorer`, `eval_ppo`, `render_env_videos`, `run_teacher`, `score_trace`, `smoke_env`, `smoke_g1`, `teacher_stance_sweep`, `teacher_step_lab`, `train_ppo`, `tune_teacher`, `validate_refs` | each has ≥1 real reference from src/tests/docs; `eval_ppo.py` is ledger-cited (13.1% result) |
| `data/{backdet,scorer}_calibration.json` | 0.08 MB | calibrated thresholds | loaded by `src/wrestling/backdet.py`, `src/scorer/{config,spec}.py` and their tests |
| `data/support_envelope.json`, `data/teacher_*.json` | 0.14 MB | support-hull envelope + teacher sweep/lab/trim records | `src/drill/{stepping,tracking}.py`, `src/teacher/posture.py` read the envelope; teacher records are the S5 calibration evidence |
| `docs/MISSION.md`, `docs/CURRICULUM.md`, `docs/MOTOR_CURRICULUM.md`, `docs/prior_art_humanoid_control.md`, `docs/reviews/**`, `docs/REFERENCES.md`, `docs/EVIDENCE_PROTOCOL.md`, `docs/QUALITY_RUBRIC.md`, `docs/SETUP.md`, `docs/STATUS.md` | 0.70 MB (dir) | long-term record, curriculum, protocols | context/record files — never deleted; MISSION.md now carries a header note (§9) |
| `reports/**` | 1.26 MB | 30 tracked reports | historical evidence; deleting them would destroy the ledger's provenance |
| `checkpoints/{solo,rl}/*.pt` | 28 MB (untracked/ignored) | cited run checkpoints (13.1% policy, t1_balance_v1–v5) | ledger-cited measured results; untracked → listed, not deleted |

### Uncertain items I chose to KEEP (one-line reason each)

- `data/drill/rubric_FINAL_L1_90.json` (7 KB, 0 refs) — the rubric self-assessment of the headline L1 run, whose verdict `reports/2026-10-08/drill.md` §6 reports; tiny, and a rubric record is exactly the “measured trap” class the brief protects.
- `data/drill/FAIL_PUSH35_feasible_L1_seed0.{json,npz}` (1.3 MB, 0 refs) — the raw trace of the 35 N failure whose result `drill.md` §“push recovery” reports (`push_battery.json` carries the numbers); failed-run evidence for a cited negative result.
- `data/teacher_stance_sweep_wide.json` (1.3 KB, 0 refs) — sibling of the cited `teacher_stance_sweep.json`; the reports cite the family, and it is 1 KB.
- `data/drill/L2_MOTION_feasible_L3_seed0.{json,npz}` (2.4 MB) — cited by name in `drill_motion.md:24,361` as the failing same-configuration sample of a marginal envelope.
- `data/references/…/derived/analysis/measure_priority.json` (105 KB, 0 refs) — the reference-pipeline take-priority analysis; the index cites its untracked `.txt` siblings, so the JSON is the tracked copy of the same analysis.
- `data/rl/log_*.log` (24 KB, untracked/ignored) — training logs of the ledger-cited smoke runs; untracked ⇒ not deleted.
- `MUJOCO_LOG.TXT`, `WATCHDOG.yml`, `.githooks/**` — repo config/context; never deleted.

---

## 5b. Operator keep-list (received after the deletion commit): the video-imitation critical path

The milestone's primary path changed to **“imitate the operator's video reference”**. These paths are
load-bearing machinery and must not be deleted, however orphaned they look; **hand-run entry points count as
referenced** (they are documented as run-by-hand in `docs/references/yt_gBAhX5t-GW4_index.md`).

| protected path | size | state after this trim |
|---|---|---|
| `data/references/yt_gBAhX5t-GW4/**` — `pose/` (+`landmarks.npz`), `pose_pass2/`, `derived/tools/*.py` (14 tools incl. `run_pose.py`, `analyze_pose.py`, `take_inventory.py`, `measure_takes.py`, `metrics_spec.py`, `retarget_video.py`, `compare_refs.py`, `draw_pose_overlay.py`, `render_refs_video.py`, `bench_pose.py`, `compose_compare.py`, `step_stats.py`, `trace_take.py`, `vidframe.py`), `derived/analysis/**`, `derived/transitions.json`, `derived/stance_spec.json`, `derived/refs_crosscheck.json` | 18 tracked files 0.25 MB + untracked media/frames | **intact**; `bench_pose.py` was deleted then **restored** (§2, reversal 2) |
| `data/refs_video/*.npz` (12 retargeted single-G1 tracks @50 Hz: `stance_hold`, `stalk_shuffle`, `circle_step`, `stance_widen_step`, `level_change_{full,fast}`, `shot_entry_full`, `shot_recover`, `knee_sprawl_{entry,entry2,hold,recover}`) + `retarget_summary.json` | 0.99 MB | **intact** — never a deletion candidate |
| `videos/refs_video/**` | — | **does not exist** on this checkout (`ls videos/` → `env refs solo_drill teacher`); the G1-side reference renders that do exist are `videos/refs/**` (tracked, kept) |
| `data/refs/*.npz` (GrappleMap-sourced references, incl. `STAND_UP.airborne.npz`) | 0.82 MB | **intact** |
| `src/retarget/**` (incl. `landmarks.py`, the G1 landmark-site mapping shared by the video retarget and the site-based imitation objective) | 0.06 MB | **intact** |
| reference video itself | ignored media under `data/references/` | untouched (untracked; never deleted) |

No other item in this keep-list was ever bucketed DEAD or GENERATED-BLOAT.

---

| group | entries | size | note |
|---|---|---|---|
| `data/library_audit/` | 10 | 36.1 MB | peer `LibAudit` probe scripts + MJX scene dumps + assets (in flight) |
| `data/solo/metrics/` (new) | 21 | 14.6 MB | live T1/T2 monitor traces + `t1_v2_monitor_*.json` + `t2_gate_baselines.json` written by peers during this trim |
| `data/references/**` (untracked part) | 61 | 13.4 MB + 187 MB ignored media | operator reference video, extracted frames, pose overlays, `.npz` (gitignored) |
| `videos/teacher/**` | 7 | 1.9 MB | teacher A/B clips (VISUALS #3, in flight) |
| `videos/solo_drill/` (new) | 3 | 1.3 MB | `01_baseline_pd_topples.{mp4,_sheet.png}` + peer bundle |
| `docs/references/**`, `docs/CURRICULUM_REFERENCE_ADDENDUM.md` | 17 | 0.5 MB | operator-reference index + stills |
| `reports/2026-10-08/{claim_audit,clip_wiring_audit,config_plumbing,t1_gate_calibration}.md` | 4 | 0.1 MB | peer reports (in flight) |
| `tests/solo/**` | 3 | 0.06 MB | peer tests (do not delete — operator instruction) |
| `src/rl/vec_solo.py`, `src/solo/lit.py` | 2 | 0.06 MB | peer in-flight modules |
| `robots/wrestling_scene_soft.xml` | 1 | 0.06 MB | derived soft-contact scene (`robots/**` is off-limits) |
| `data/solo_drill/01_baseline_pd_topples.json`, `data/teacher_{stats,trims}.json` | 3 | 0.05 MB | peer bundles / teacher records |
| `checkpoints/**/*.pt` (ignored) | 10 | 28.3 MB | cited run checkpoints — **must not be deleted** |
| `data/locks/sim.lock` (ignored) | 1 | 69 B | live sim lock — untouched |
| `.venv/**`, `third_party/**` | — | 1.3 GB / 2.3 GB | environment + vendored — untouched |

No untracked file was deleted. The only untracked things removed are regenerable caches
(`__pycache__/`, `.pytest_cache/`, 101 files / 3.0 MB), which the bucket definition lists explicitly.

---

## 7. Dangling references introduced (fix rather than leave)

| reference | points at | status |
|---|---|---|
| `reports/2026-10-08/clip_wiring_audit.md:119` | `l2_stills/frame_t002.00.png`, `l2_stills/frame_t030.00.png` | deleted (the peer report itself classifies them as “no index entry”); edit the line to say *removed 2026-10-08* |
| `notes.md:119` | the `phase3_teacher` workflow launch | `workflows/phase3_teacher.mjs` deleted; the lesson is about workflow-script syntax and stays valid — add “(script since removed)” |
| `reports/2026-10-08/drill_motion.md:50` | “`M_*` traces carry the single-width runs” | the 41 sweep `.npz` were deleted; the per-run **metrics** (`M_*.json`), `motion_widths.json`, `motion_singles.json` remain — amend to name those |
| `docs/VISUALS.md:21`, `notes.md:214,249,292,296,933`, `data/solo_drill/01_baseline_pd_topples.json:4-5` | `data/drill/BASELINE_PD_stance_pd_L1_seed0.{json,npz}` | **restored — not dangling** |
| `src/solo/video.py:53` | `scripts/render_env_videos.py` | kept — not dangling |
| `scripts/solo_drill_render.py` docstring | `data/drill/L1_90_feasible_L1_seed0.npz` | **pre-existing** dangling (only `FINAL_L1_90_*` ever existed); not caused by this trim |
| `data/drill/M_*_feasible_L3_seed0.json` `"tag"` fields | `M_*` (labels, not paths) | not a path reference — no action |

---

## 8. Before / after measurements

| metric | before | after | delta |
|---|---|---|---|
| `du -sh .` | 4.2 G | 4.2 G | −105.5 MB tracked + 3.0 MB caches (below `du`/`df` rounding: `.git` 214 MB, `.venv` 1.3 GB and `third_party` 2.3 GB dominate and are untouched) |
| `df -h /` | 104 G used, 90 G avail, 54 % | 104 G used, 89 G avail, 54 % | concurrent peers wrote >100 MB during the trim, masking the gain |
| `git ls-files \| wc -l` | **497** (at PIN) | **437** immediately after the trim commit `46fb84f` | −60; now **458** at `d693c39` because peers committed ~21 further files during/after the trim |
| on-disk files (excl. `.git`, `.venv`, `third_party`) | not captured — the session-start `find` was refused by tool policy | **664** right after the trim, **724** at `d693c39` | lower bound of the pre-trim count: 664 + 57 + 101 = 822 (peers added files concurrently) |
| bytes freed (from blob sizes at `72956f3`) | — | — | **105.55 MB** over 57 tracked files + **3.01 MB** caches |
| `data/` | 327 MB | 268 MB | −59 MB (plus peer additions) |
| `videos/` | 64 MB | 65 MB | peer-added teacher clips |

### Test suite (PIN-based, immutable)

- **before** — throwaway worktree at PIN `61e8bd1` (`git worktree add /tmp/trim_baseline 61e8bd1` + a
  `third_party` symlink, since `third_party/` is gitignored and absent from a worktree):
  `MUJOCO_GL=egl .venv/bin/python -m pytest tests/ -q` → **178 passed, 0 failed** (66.7 s).
  (First run without the symlink gave 5 failed/16 errors, all `FileNotFoundError: …/third_party/GrappleMap/GrappleMap.txt` — an artefact of the worktree, not of the code.)
- **after** — throwaway worktree at HEAD `96478e1` (post-deletion, includes the peers' committed T2-gate work):
  `MUJOCO_GL=egl .venv/bin/python -m pytest tests/ -q` → **185 passed, 0 failed** (85.9 s).
  Delta = **+7**, exactly the 7 tests of `tests/solo/test_locomotion_gate.py` (added by the T2-gate commit
  `377009c`, `git log 61e8bd1..HEAD -- tests/solo/test_locomotion_gate.py`). **Zero test files were deleted,
  so zero tests were lost; no failure is attributable to the trim.**
- **Peer WIP excluded:** `tests/solo/test_vec_solo.py` (untracked, SoloVec's vec-port WIP) and
  `tests/solo/test_lit_reward.py` (untracked) are **not** in any committed tree, so they are absent from both
  worktrees and from both counts. `test_vec_solo.py` is currently red by design (~7 failures:
  `TrainConfig(n_envs=…)`, `SoloTrainer.close`) — **peer WIP, red for reasons unrelated to the trim, excluded
  from the comparison.** Running the suite on the live working tree would pick it up; that is exactly why the
  counts were taken in worktrees of committed trees.
- **The tree moved after PIN for unrelated reasons:** `E30`/`E31`/`E33` (T1 ledger: unattainable gate,
  literature check, v5 negative), the T2 gate (`377009c`, `51949d7`) and the T1 gate calibration (`d693c39`)
  all landed after `61e8bd1`. None of them touches a file this trim deleted, and none is counted against the
  trim; they are the reason the tracked file count is *higher* than 437 by the time of writing (peers added
  `tests/solo/test_{config_plumbing,t1_gate_discrimination}.py`, `scripts/solo_t2_gate.py`, `src/rl/vec_solo.py`, …).
- Attribution rule applied: for any failure, `git log --oneline <PIN>..HEAD -- <path>` is checked before
  blaming the trim; a peer's commit touching the failing file is named and the failure is not counted as trim-caused.
- The live working tree is **not** used for the after-count: peers still hold uncommitted edits in
  `src/solo/{baselines,commands,env,eval,metrics,train}.py`, `scripts/solo_{env_smoke,drill_render}.py`,
  `docs/VISUALS.md`, `data/solo_drill/01_baseline_stancepd.json`.

### Import smoke (after deletion) — both import styles pass

```
MUJOCO_GL=egl .venv/bin/python -c "import src.solo.train, src.solo.env, src.solo.eval, src.drill.scheduler, src.teacher.solo_scene, src.rl.ppo"
  → OK src.* form
MUJOCO_GL=egl .venv/bin/python -c "import sys; sys.path.insert(0,'src'); import solo.train, solo.env, solo.eval, drill.scheduler, teacher.solo_scene, rl.ppo, rl.net, rl.checkpoint, wrestling.env, wrestling.backdet, scorer.scorer, retarget.solve, grapplemap.parser"
  → OK sys.path form
```
`scripts/solo_env_smoke.py --help` → prints `usage: solo_env_smoke.py [-h] [--seed SEED] [--seeds SEEDS]
[--steps STEPS] … [--t2-episodes T2_EPISODES] [--no-video] …` (OK).
`scripts/solo_drill_render.py --help` → prints `usage: solo_drill_render.py [-h]
{run,render,report,suite,l2suite,tracks,motion,deliver} …` (OK).

### Deleted-name re-grep

Every deleted top-level module/file name re-grepped across `src/ scripts/ tests/ docs/ reports/ notes.md`:
`bench_pose` 0 hits · `wave1_assets` 0 hits · `M_*` sweep npz 0 path hits (only the retained JSONs' `"tag"` labels) ·
`*.jsonl` basenames 0 hits · `phase3_teacher` 1 hit (`notes.md:119`, historical lesson — see §7) ·
`l2_stills` 1 hit (`reports/2026-10-08/clip_wiring_audit.md:119`, historical report — see §7).

### Live service

`solo-t1-v5` (`python -m src.solo.train --task balance … --out checkpoints/solo/t1_balance_v5.pt`) was
**stopped by its owner before this verification**: commit `96478e1` “E33: v5 NEGATIVE (stopped)” documents
the stop, and the final checkpoint `checkpoints/solo/t1_balance_v5.pt` was written at 14:41. The trim never
touched `checkpoints/`, `data/locks/sim.lock`, or the process (no signals, no writes). **Not confirmable:
“still training afterwards” — it was already stopped for a measured negative result, not by the trim.**

---

## 9. Record fixes (survivability)

1. **`notes.md` ledger entry** — appended (see the `REPO TRIM` entry at the end of `notes.md`): operator
   decision, the buckets with sizes, the recovery ref `72956f3`, and the dependency finding.
2. **`docs/MISSION.md` header note** — added: the two-robot implementation is unchanged in the tree but is
   *scheduled* for removal (it is currently a mandate import dependency); the active milestone is
   `docs/SOLO_DRILL.md`; the excision plan is §3 above.
3. Dangling references listed in §7.

---

## 10. Process notes (honesty)

- **Commits:** `46fb84f` (the trim) and `3769395` (report + notes.md ledger entry + MISSION.md header) are on
  `origin/main` (a peer pushed them). Every later commit in this trim's series (from `4875e73` on) is
  **local-only**; run `git push` to restore the “all committed and pushed” invariant.

- A git race occurred while restoring `BASELINE_PD_*`: the concurrent T2-gate commit (`14c3e98`) landed
  between staging and `git commit --amend`, so my restore was folded into that peer commit, which was
  rewritten to `377009c`. Content is identical plus the two restored files; `origin/main` is in sync with
  local HEAD (`96478e1`) and the deletion commit `46fb84f` is intact as its ancestor.
- Net deletions are therefore visible in history as 60 paths in `46fb84f` with 2 re-added in `377009c`;
  the effective set is the 58 listed in §2.
- The single riskiest deletion: the 41 `M_*_feasible_L3_seed0.npz` width-sweep traces (73.9 MB). They are
  the raw data behind the `reports/2026-10-08/drill_motion.md` width table and are mentioned collectively
  in that report (`M_*` traces). The table, the per-run metrics JSONs and `motion_widths.json` are retained;
  recover with `git show 72956f3:data/drill/<name>.npz > <name>.npz`.
