# Motor capability audit — implemented vs demonstrated (foundational motor curriculum, step 0)

**Agent:** MotorAudit · **Date:** 2026-10-08 · **Repo:** `/home/ubuntu/grappling` @ worktree on
`main` (HEAD `0b60780`, "docs: foundational motor curriculum (M1-M5) + goal addendum").
**Assignment:** `docs/MOTOR_CURRICULUM.md` §6 step 0 — "Audit implemented-vs-demonstrated motor
capability … avoid building what exists; find reusable primitives."
**Written to:** `reports/2026-10-08/motor_audit.md` (only file touched; read-only audit otherwise).

## 0. Method, evidence taxonomy, and snapshot caveats

Read in full or in the cited ranges: `AGENTS.md`, `goal.md` (+ addendum), `docs/MISSION.md`,
`docs/CURRICULUM.md`, `docs/MOTOR_CURRICULUM.md`, `docs/VISUALS.md`, `notes.md` (whole),
`reports/2026-10-07/*.md`, `reports/2026-10-08/*.md`, and the code under `src/` and `scripts/`.

**Evidence classes used in every table row** (per MISSION "Never confuse plausibility with
verification" and `docs/MOTOR_CURRICULUM.md` §3):

- **MEASURED** — at least one number exists, produced by a run on this repo/host, with a source
  path (report, log, JSON, or a test that measures). The number must bear on the row's claim;
  a doc's assertion without the underlying metric is *not* MEASURED.
- **CODE-ONLY** — implementation exists but no measured outcome of the behavior it would produce
  (no report/log/number). Includes code referenced by pending reports.
- **DOC-CLAIMED** — appears as a plan/claim in docs without an underlying metric (e.g. the
  CURRICULUM line "Locomotion … built procedurally in env (phase 4+)" with no code).
- **ABSENT** — no implementation and no measurement anywhere in `src/`, `scripts/`, `data/`.

Rows marked **MEASURED FAILURE** are MEASURED rows whose number records that the attempted
behavior did not hold (e.g. a toppling replay or a collapsing policy); slashes list mixed
evidence (e.g. "DOC-CLAIMED / ABSENT").

**Three snapshot caveats** (all affect accuracy, all flagged inline):

1. `src/teacher/` was being written throughout this audit (files changed at 04:25, 04:28, ~04:36
   and later). It now has a **DRAFT report** (`reports/2026-10-08/teacher.md`, 159 lines; its
   acceptance table is unfilled), a first stats dump (`data/teacher_stats.json`, 5 seeds × 5
   techniques), all 7 technique videos (`videos/teacher/*.mp4` — 2 existed mid-audit, 7 by the
   end) and a test file (`tests/test_teacher.py`). Teacher numbers below are quoted from the
   draft + stats dump and are marked **DRAFT (in-flux)**; the teacher's own `gains.py` still
   points at measurements "in reports/2026-10-08/teacher.md" whose acceptance sections are
   pending.
2. `videos/env/` was rendered during the audit; the accompanying report
   `reports/2026-10-08/env_videos.md` (331 lines) now exists and documents all 5 scenarios with
   measured per-scenario tables (draw 5×1.0 s; back event contact 1.38 s → trigger 1.70 s →
   exchange end 1.80 s incl. the ambiguity hold; knees/hands 121/150 limb-contact frames, 0
   dorsal, full-clock draw; simultaneous trigger 0.32 s both, ambiguous; OOB 3 crossings at
   1.32/3.32/5.32 s, forfeit + score −1.75/+1.00).
3. **Core files were modified by concurrent agents mid-audit**: `src/rl/curriculum.py` (the
   advancement gate was rewritten, then revised once more after an orchestrator correction to a
   per-stage criterion model), `src/wrestling/env.py` (back-trigger ambiguity hold), plus
   supporting plumbing in `src/rl/{reward,rollout,trainer}.py`. The final API is described in
   §3.2/§3.3 and evidenced in `reports/2026-10-08/p0_fixes.md` (now present). Line numbers for
   these files were re-anchored to the ~04:51 revision; teacher line numbers to the ~04:36 teacher
   revision (§6). A teacher tuning artifact (`data/teacher_trims.json`, 04:29) also landed — see
   §1a/§2.7.

**Reading rules applied throughout:** (a) code completion or passing unit tests is not evidence
of learned behavior (MISSION §3; `docs/MOTOR_CURRICULUM.md` §0/§3); (b) a win-rate number is not
evidence of technique success when the opponent collapses (E1 counterexample, `notes.md`
Experiment ledger); (c) scripted controller behavior (keyframe hold, PD replay) is never
reported as "learned".

---

## 1. Capability table

### 1a. Balance, CoM/support control, whole-body coordination

| capability | implemented? (file:line) | demonstrated? (class + metric + source) | video evidence? | gap / verdict |
|---|---|---|---|---|
| **Static balance — standing hold, no disturbance** | `StandHold` controller (src/wrestling/env.py:367-387) holding `a_stand`/`b_stand` keyframe ctrl; model position servos kp=500 (robots/g1/g1.xml via reports/2026-10-07/g1_model.md) | **MEASURED (scripted, not learned):** 5 s hold, pelvis 0.7900→0.7916 m (+0.16 cm), torso tilt ≤ 0.172°, xy drift 0.1 mm, 8 foot contacts — reports/2026-10-07/g1_model.md; 20 s pair hold: draw at exactly 20.0 s, pelvis 0.7916 m both, 0.1 mm drift — reports/2026-10-08/wrestling_env.md §2.3 | PNGs only: reports/2026-10-07/g1_front.png, g1_side.png; no video | Model stands under its own servos; **no policy** holds any pose. Not a balance controller (open-loop keyframe). |
| **Static balance — wrestling stance hold** | STANCE reference (data/refs/STANCE.npz; src/retarget/techniques.py:123 `_stance`); env/evidence starts at STANCE frame 0 (src/wrestling/env.py:176-179, 502-527) | **MEASURED FAILURE:** PD replay topples both robots — pelvis 0.807→0.740 m (1.0 s) →0.558 (1.3 s) →0.157 (1.5 s); back detector fires at 1.78/2.04 s (reports/2026-10-08/wrestling_env.md §1.2, §3.1); ankle torque reaches ≈39 Nm of the ±50 Nm limit as it tips. Snapshot teacher notes claim topple "0.8–1.5 s even when the balance feedback is running" (src/teacher/trims.py:5-12) — **DRAFT (in-flux)**: now corroborated by the draft report (reports/2026-10-08/teacher.md §3: 36 Nm of the ±50 Nm ankle limit just to stand still; stabilizer-only stay-up 0.64–0.76) and a tuned reference trim reaches stay-up 0.99/0.85 for one measurement (§9 #5) but scores only 0.60 on the technique scorer; the shipped trim frontier is (0.76 stay / 0.51 scorer) over 3 seeds (§10); `data/teacher_trims.json` (5-seed tuner output, STANCE) records stay_up a_ = 1.0 / b_ = 0.9167, scorer 0.631 | videos/refs/STANCE.mp4 (topples); videos/env/draw_stance.mp4 uses the env's 1.0 s clock so each draw ends while standing (render_env_videos.py:28-33; measured `--diag` topple trigger 1.78 s — reports/2026-10-08/env_videos.md §1) | **No *verified end-to-end* stance controller yet.** The DRAFT teacher can *hold* the stance only by trimming the pose, which drops it out of the scorer's relation bands; teacher.md §10 concludes a statically stable AND relation-preserving stance needs foot re-placement (a step) — out of scope for the joint-position teacher. |
| **Dynamic balance — recovery from pushes / perturbations (held-out disturbances)** | **No push/perturbation machinery existed at the audit snapshot**: zero matches for `xfrc|qfrc|apply_force|mj_rne` in `src/`+`scripts/` then; only training-time action noise (src/rl/curriculum.py:59-75) and teacher initial-state wobble ±0.01 rad joints / ±5 mm base (src/teacher/episode.py:59-67, in-flux). Teacher balance feedback is CODE-ONLY: capture-point ankle/hip channels (src/teacher/controller.py:269-365; src/teacher/stabilizers.py:77-124). **Post-audit (outside this snapshot):** `src/drill/runner.py` traces applied pushes and `src/drill/metrics.py:123+` (`push_report`) scores excursion/recovery — not audited here | **ABSENT at snapshot.** Nearest data: 30 000-step random-init PPO policy — standing fraction 0.131 (pelvis z≥0.5 m), ground fraction 0.830 (z<0.35 m), mean pelvis z 0.267 m (`notes.md` E1; scripts/eval_ppo.py output in reports/2026-10-08/rl_infra.md §6.2) | none at snapshot | No held-out push battery existed at the snapshot; the post-audit drill push report is the candidate harness — certify it under the evidence standard. |
| **CoM / capture-point computation and control** | YES, only in the in-flux teacher: `com = data.subtree_com[ctx.pelvis]` (src/teacher/controller.py:277); `capture_error()` (src/teacher/stabilizers.py:120-124: `xi = com + kd*sqrt(h/g)*v`, clamped vs support centre); support centre (src/teacher/stabilizers.py:189-195); foot contact (src/teacher/stabilizers.py:198-203) | **CODE-ONLY / DRAFT (in-flux).** No report section validates these channels themselves, but the DRAFT report (reports/2026-10-08/teacher.md), `data/teacher_stats.json` and `tests/test_teacher.py` now exist and record outcomes (§1c, §3.3). Around them, failure *diagnostics* are measured (CoM behind support in STANCE — reports/2026-10-08/wrestling_env.md §3.1; teacher.md §3 measures 36 Nm ankle) | none | Primitives exist; a *learned* balancer does not, and the scripted teacher's stance hold is a draft tradeoff (teacher.md §10). |
| **Support polygon** | **NONE at audit snapshot.** As of the snapshot, `grep ConvexHull|hull|polygon|scipy.spatial` over `src/`+`scripts/` found no support-polygon/hull computation; the teacher only took the **mean** of touching sole sites ("support centre", src/teacher/stabilizers.py:189-195) with a reference fallback. **Post-audit (outside this snapshot):** hull + signed CoM margin landed in `src/drill/kin.py:201/244/267`, used by drill balance/gates and `metrics.push_report` (see §5/§6) | **ABSENT at snapshot**; post-audit implementation exists — certify separately | — | If M1 wants CoM-margin/support-polygon rewards, the drill package now has the code; audit it under the evidence standard before reuse. |
| **Foot contact sensing** | YES: env obs 2 bits (src/wrestling/env.py:255-258, 267); priv-critic 5 categories `[feet,knees,hands,torso,pelvis]` (src/rl/privileged.py:47, 128-144, 179-181); teacher site-height contact (<0.045 m, src/teacher/stabilizers.py:198-203, in-flux); backdet limb-contact diagnostic (src/wrestling/backdet.py:131 `body_maps`) | **MEASURED as diagnostics:** env contact bits pinned by tests (tests/test_wrestling.py); privileged categories used in the wrapper regression (reports/2026-10-08/rl_infra.md §8/§11); backdet limb_contact defect measured and fixed (notes.md "FIXED (BackdetFix)"). No controller consumes contacts as a *working* balance input today | HUD prints limb contacts in videos/env clips (scripts/render_env_videos.py:264-294) | Sensing exists and is reusable; contact-**aware control** exists only as unverified teacher code. |
| **Whole-body coordination — arms/reach while balancing** | Arms are inside the action space (29 joint targets, src/wrestling/env.py:89 `N_JOINTS=29`); references contain arm motion (src/retarget/*; videos/refs/*.mp4). No arm/reach task, target, or metric exists | **ABSENT as behavior.** Adjacent measured fact: PD reference tracking itself is good (mean joint err 0.010–0.065 rad over the 7 refs — reports/2026-10-07/retarget.md §6) but the robots topple, so "arms while balancing" has never been demonstrated | none | M3 needs a task; the joints/refs are reusable. |
| **Whole-body coordination — torso+arm+leg coordination (waist channel etc.)** | Teacher has a torso-uprightness row on the waist channel (src/teacher/stabilizers.py:159-175, in-flux, CODE-ONLY) | **CODE-ONLY.** No measured coordination result | none | Same as above. |

### 1b. Locomotion

| capability | implemented? (file:line) | demonstrated? (class + metric + source) | video evidence? | gap / verdict |
|---|---|---|---|---|
| **Walk forward/backward/lateral** | **ABSENT.** No locomotion code exists: `grep walk|gait|foot_placement|cycle_time` over `src/`+`scripts/` = no matches. Env observation has no command slot (84-dim layout, src/wrestling/env.py:125-139); scripts/render_env_videos.py:46/166 states "the env has no locomotion policy yet". CURRICULUM.md planned "locomotion … built procedurally in env (phase 4+)" — nothing was built | **ABSENT.** No walking number exists anywhere. The only measured whole-policy movement is degenerate: E1 ground-hugging (0.830 ground fraction, `notes.md` E1) | none | M2 is entirely missing. |
| **Velocity tracking (vx, vy)** | **ABSENT** (no command vector, no tracking reward; reward only has `oob`, `exchange_end`, plus shaping `technique_similarity/progress/engagement` — src/rl/reward.py:44-52, 135-214) | **ABSENT** | none | Requirement defined in docs/MOTOR_CURRICULUM.md §2; no implementation. |
| **Turning / angular-velocity tracking** | **ABSENT** | **ABSENT** | none | — |
| **Accel / decel / stop** | **ABSENT** | **ABSENT** | none | — |
| **Direction change (locomotion)** | **ABSENT.** Closest existing primitive is start randomization yaw/distance (src/wrestling/env.py:502-527) — shuffling the start pose is not direction change | **ABSENT** | none | — |
| **Fall detection / termination for a balance task** | **ABSENT as such.** The only terminal condition is the back-to-mat rule (src/wrestling/backdet.py; src/wrestling/env.py:622-690) plus timeout/OOB/match clock; a robot falling to hands/knees can never terminate an exchange except by back contact or timeout (by design — MISSION "knees/hands are NOT losses"). `terminated` ⇔ match clock only (src/wrestling/env.py:549-601) | **MEASURED (rule behavior):** knees/hands non-terminal pinned by test through the RL wrapper (reports/2026-10-08/rl_infra.md §8, tests/test_rl.py); video evidence with numbers: 121/150 limb-contact frames, 0 dorsal, full 5 s clock to a draw (reports/2026-10-08/env_videos.md §3) | videos/env/nonterminal_knees_hands.mp4 | M1 needs its own fall detector; the backdet is the wrong detector for "fell over" (fires only on persistent dorsal contact). |

### 1c. Wrestling locomotion (M4)

| capability | implemented? (file:line) | demonstrated? (class + metric + source) | video evidence? | gap / verdict |
|---|---|---|---|---|
| **Stance maintenance (hold/stay in wrestling stance)** | STANCE reference + env start (see 1a row 2) | **MEASURED FAILURE / DRAFT improvement**: open-loop ≈1.4 s topple; stabilizer-only stay-up 0.64–0.76; tuned trim reaches 0.99/0.85 but drops the scorer to 0.60; shipped trim frontier 0.76 stay / 0.51 scorer (reports/2026-10-08/teacher.md §3/§9/§10 — DRAFT) | videos/refs/STANCE.mp4 | Not met at both gates simultaneously; needs a stepping/re-placement controller (teacher.md §10). |
| **Shuffle / circle** | **ABSENT at audit snapshot.** No shuffle/circle code existed then; the only scripted "movement" was a pelvis teleport-drag for the OOB demo (scripts/render_env_videos.py:163-212 `SlideController`). **Post-audit (outside this snapshot):** a solo drill suite landed with margin-gated stepping and measured L2 results — 4-step shuffle 34 s / 0 falls, L1-cycle 3 steps then one fall, wide stance refusals with required-travel numbers (reports/2026-10-08/drill_l2.md; src/drill/stepping.py; data/drill/l2_suite_summary.json). Not audited here | **ABSENT at snapshot**; post-audit claims exist (drill_l2.md) — re-audit against the evidence standard | — | — |
| **Level change without collapsing** | References contain level change (DOUBLE_LEG keyframes; teacher labels DIVE/LOW from pelvis kinematics, src/teacher/phases.py:9-32, in-flux); no policy implements it | **MEASURED FAILURE under PD replay:** all shot techniques fail the stay-up gate — DOUBLE_LEG min up-pelvis 0.07 m, SINGLE_LEG 0.16 m, BODY_LOCK 0.26 m (reports/2026-10-07/retarget.md §6, verdicts partial/FAIL); SNAPDOWN is the only ref passing stay-up (0.52 m) but fails inter-penetration (0.056 m). **DRAFT teacher improvement (data/teacher_stats.json, 5 seeds):** stay-up min means SNAPDOWN 0.956, DOUBLE_LEG 0.520, SPRAWL 0.316, SINGLE_LEG 0.267, BODY_LOCK 0.200; scorer means 0.32–0.81; inter-pen 0.017–0.059 m — better than open-loop replay but only SNAPDOWN approaches a holdable shot | videos/refs/DOUBLE_LEG.mp4 etc. (VISUALS.md #2: "Motion shape is right; staying up is not"); videos/teacher/{DOUBLE_LEG,SNAPDOWN}.mp4 (2 of 7 mid-audit; all 7 by the end) | — |
| **Angle change vs opponent** | Only an *observation*: `opp_rel_heading` sin/cos (src/wrestling/env.py:282-285). No angle target, no controller | **ABSENT** | — | — |
| **Penetration step** | **DOC-CLAIMED.** MISSION/CURRICULUM vocabulary list ("level change → penetration → drive"); a scorer phase predicate exists (`"penetration step: depth taken, head to chest, knee UP"`, src/scorer/spec.py:62-63) — a measurement of the reference, not a behavior | **DOC-CLAIMED / ABSENT** | — | GrappleMap contains the geometry; no executable skill exists. |
| **Controlled knee drop** | **ABSENT.** Teacher labels PRONE and disables stabilizers below pelvis z 0.42 (src/teacher/phases.py:15-16, 28-29; src/teacher/gains.py PRONE row) — a code path, not a skill with a measured outcome | **ABSENT** | — | — |
| **Recovery from non-standing states (knees/hands/sprawl→stand)** | Reference exists: STAND_UP (data/refs/STAND_UP.npz, 557 frames/11.12 s, videos/refs/STAND_UP.mp4); env keeps those states non-terminal (reports/2026-10-08/rl_infra.md §8); teacher in flight | **MEASURED FAILURE of execution:** "STAND_UP a/b | 3.282 / 3.802 s. recovery never completes in PD replay; bodies end dorsal" (reports/2026-10-08/wrestling_env.md §1.2 trigger table). No shipped controller brings knees/hands→standing (scripts/render_env_videos.py:40-43; measured pair 3.08 s / solo 1.64 s to dorsal — reports/2026-10-08/env_videos.md §3). **DRAFT teacher finding (§9 #8):** the STAND_UP *reference itself* is airborne (min foot-site z 0.29–0.72 m, pelvis 1.0–1.08 m for t > 5 s) and per-frame re-grounding fails, so this reference is judged infeasible — a phase-2 artifact | videos/refs/STAND_UP.mp4 (falling/ending dorsal) | The *rule* supports recovery; the *motor* capability is unproven, and the current STAND_UP reference is a suspect input (needs re-retargeting per the teacher's measurement). |

### 1d. GrappleMap techniques — reference existence vs executed success

**Row: technique references (deliverable 5)** — **MEASURED.** 7/7 references exist as
`data/refs/<TECHNIQUE>.npz` with measured landmark RMS (weighted mean 0.028–0.091 m), durations
(0.84–11.12 s), and 7/7 PD-replay video renders: reports/2026-10-07/retarget.md §5, §6;
`videos/refs/*.mp4`; VISUALS.md #2.

**Row: technique execution in simulation** — **MEASURED FAILURE / PARTIAL.** Under the only
execution controller that exists (open-loop reference replay through the position servos,
`ReferenceReplay`, src/wrestling/env.py:309-349), no technique passes the phase-2 acceptance
end-to-end; acceptance table (reports/2026-10-07/retarget.md §6) and backdet trigger table
(reports/2026-10-08/wrestling_env.md §1.2):

| technique | ref (T @50 Hz / dur) | PD replay: joint err mean/max (rad) | stay-up (min up-pelvis, m; gate ≥0.45) | inter-pen (gate ≤0.02 m) | measured end state / trigger | verdict |
|---|---|---|---|---|---|---|
| DOUBLE_LEG | 178 / 3.54 s | 0.037 / 0.152 | **0.07 ✗** | 0.017 ✓ | collapse, dorsal trigger 2.842 s (opponent) | partial (motion shape only) |
| SINGLE_LEG | 244 / 4.86 s | 0.038 / 0.079 | **0.16 ✗** | **0.026 ✗** | collapse, triggers 3.702 / 1.762 s | partial |
| BODY_LOCK | 196 / 3.90 s | 0.047 / 0.158 | **0.26 ✗** | **0.044 ✗** (self 0.044 ✗) | collapse | **FAIL** (report's own word) |
| SNAPDOWN | 43 / 0.84 s | 0.058 / 0.277 | 0.52 ✓ (only pass) | **0.056 ✗** | no dorsal trigger (attacker), b none | partial |
| SPRAWL | 334 / 6.66 s | 0.040 / 0.277 | prone by design; min 0.06 | 0.013 ✓ | defender's **back hits the mat** at 1.682 s (tilt 76°, pelvis 0.079 m) despite the kinematic final frame being face-down — "the PD-replay physics settles the defender supine" (wrestling_env.md §1.3) | partial; physically loses by the game rule |
| STAND_UP | 557 / 11.12 s | 0.023 / 0.111 | **0.06 ✗** | 0.020 ✗(edge) | "recovery never completes; bodies end dorsal" (3.282/3.802 s) | partial |
| STANCE | 61 / 1.20 s | 0.010 / 0.047 | 0.45 (boundary, 2.34 s window only) | ✓ | topples ≈1.4 s (wrestling_env.md §3.1) | partial; not holdable |

Independent technique-validity metric: the scorer is **MEASURED** but scores *references*
(self-conformity ≥0.971 per phase; trace means 0.987–1.000; cross-technique ≤0.464;
reports/2026-10-08/scorer.md). Those numbers say the metric can recognize the reference family;
they say nothing about an executed attempt. No learned policy has ever executed a technique; the
nearest run (30 k-step stage-C smoke) has a policy that spends 83% of its time on the ground
(`notes.md` E1 / reports/2026-10-08/rl_infra.md §6.2).

**Video summary:** technique *reference* videos exist (videos/refs, 7 mp4, rendered PD replays,
VISUALS.md #2 = "✅ rendered, ❌ standing"); env-rule videos exist for all 5 scenarios with a
complete measured report (videos/env/*.mp4 + sheets; reports/2026-10-08/env_videos.md); teacher
videos were 2 of 7 mid-audit and all 7 by the end (videos/teacher/*.mp4 — DRAFT report); there is **no video of
any learned motor behavior** in the repo (all existing motion is scripted replay/hold/tuning).

---

## 2. Reusable primitives inventory (with file:line)

### 2.1 Environment API (`src/wrestling/env.py`)

| primitive | where | notes for reuse |
|---|---|---|
| `WrestlingEnv` | env.py:423 | 50 Hz control (10×2 ms substeps, env.py:85-88 (pre-edit revision; see §0 caveat 3)), 2-robot MjData |
| Action contract | env.py:542-543 (`action_dim=58`), env.py:549-601 (`step`), env.py:603-620 (`_resolve_action`) | per-robot 29 joint-position targets, ctrlrange-clipped; accepts `(2,29)` or `(58,)` |
| Pluggable observation | env.py:426-462 (`obs_fn` ctor arg), env.py:219-286 (`default_observation`), env.py:141-144 (`obs_layout`, 84-dim table) | **swappable** — M1/M2 can replace the obs without touching the env (final spec is deliverable 14) |
| Pluggable reward | env.py:289-303 (`default_reward`), env.py:426-462 (`reward_fn`) | event-driven: `oob`, `exchange_end`; per-step sum, no shaping in the env itself |
| Reset randomization | env.py:463-497 (`reset`), env.py:502-527 (`_randomized_stance`) | seeded pair yaw U(±10°), distance 0.98±0.15 m, joint noise U(±0.03 rad); deterministic; `pose=` override hook |
| Exchange loop | env.py:622-690 (`_maybe_end_exchange`), env.py:692-705 (`_start_exchange`) | back / OOB-forfeit / timeout / match-end; 0.10 s back-trigger ambiguity hold (env.py:107-115, §3.3) |
| Match rules constants | env.py:103-123 (mat radius 1.5 m, OOB hysteresis, 3-event forfeit, 20 s exchange, 180 s match) | all constructor knobs |
| Scripted controllers | env.py:324-364 (`ReferenceReplay`), env.py:367-387 (`StandHold`); swap via env.py:545-547 | reuse for baselines/curriculum opponents |
| Exchange log | env.py:391-421 (`ExchangeRecord`), env.py:601+ (`info` dict) | per-exchange table for evaluation |
| Detector hooks | env.py:712-714 (`back_snapshot`), backdet integration env.py:634-690 | see 2.2 |

### 2.2 Back-to-mat detector (`src/wrestling/backdet.py`)

- `BackDetConfig` (backdet.py:63-85; calibrated defaults tilt 45°, pelvis-z 0.35 m, confirm 0.30 s
  at backdet.py:73-75), `back_features` (backdet.py:164-216), streaming `BackToMatDetector`
  (backdet.py:218-277), batch rule `confirmed_mask` / `first_confirmed_index` (backdet.py:279-302),
  `FeatureSequence` for offline sweeps (backdet.py:303-338).
- Calibration harness: `scripts/calibrate_backdet.py` (560-config sweep; sens/spec 1.000 at the
  chosen point) and `data/backdet_calibration.json`. Verified numbers in
  reports/2026-10-08/wrestling_env.md §1.
- Reuse: exchange-end semantics must be preserved (MISSION hard rule); it is an **event** rule for
  matches, not a fall detector for M1 (see 1b).

### 2.3 RL infrastructure (`src/rl/`)

| primitive | where | notes |
|---|---|---|
| Actor obs builder + frame stack + command block | obs.py:47-63 (`actor_obs_dim`, 84×stack + 8), obs.py:70-107 (`command_vector`/`TechniqueCommand`), obs.py:130-192 (`ActorObsBuilder`) | 92-dim at stack 1; command = 7-technique one-hot + phase |
| Privileged critic obs | privileged.py:47 (`CONTACT_KINDS`), 96-157 (`ContactCategorizer`), 159-202 (`PrivilegedObsBuilder`, 162-dim), 204-223 (`CriticObsBuilder`) | includes full qpos/qvel both robots, 5 contact categories, back-exposure proxy (privileged.py:54-57) |
| PPO + nets | ppo.py:31 (`PPOConfig`), ppo.py:94 (`compute_gae`), ppo.py:163 (`ppo_update`); net.py:67/120/138/161 (`Actor`/`Critic`/`ActorCritic`/`ActionMapper`) | ~97 k / 131 k params; absolute or residual actions |
| Curriculum schema + stages | curriculum.py:69-87 (`PerturbationSchedule`), 90-120 (`ExchangeSample`), 124-196 (`AdvanceRule`, final post-fix API — §3.2), 199-228 (`StageConfig`), 230-289 (`PISTY_STAGES` A–E; A-C `execution`, D `outcome`, E none), 297-317 (ser/de), 319-353 (`CommandScheduler`), 355-482 (`Curriculum`) | stage config checkpointed incl. the `ExchangeSample` window; the runtime-only hook is excluded from `as_dict` |
| Gate signal plumbing (post-fix) | rollout.py:98-101, 182-200 (per-exchange mean scorer similarity attached to the exchange record); trainer.py:124-132 (`stood` = no own back trigger ∧ no OOB forfeit; `on_exchange(outcome, similarity, stood)`); reward.py:166-191 (`sample` out-param exports the scored value; §3.2 item 7 quirk) | this is the `execution`-criterion signal path for stages A-C |
| Rollout collector | rollout.py:35-52 (`OpponentPolicy`), 60-227 (`RolloutCollector`) | pairs with a frozen opponent |
| Reward assembly | reward.py:44 (`RewardWeights`), 62-114 (`ScorerAdapter`), 117-214 (`StageReward`: outcome/oob/similarity/progress/engagement) | scorer use is lazy/degrading |
| Scripted opponents | scripted.py:23 (`KINDS` = stand_hold / reference_replay / teacher / policy / none), 27-113 (`OpponentSpec`, `_TeacherHalf`) | `teacher` kind is wired but unused (teacher in-flux) |
| Vectorization | vec.py:37 (`RESET_SEED_STRIDE`), 66-83 (`build_env`), 96-155 (`_ChildState`), 190+ (`VecWrestlingEnv`) | sequential + subproc (fork); measured 421–437 steps/s at 3 envs, subproc wins ≥3 (reports/2026-10-08/rl_infra.md §5) |
| Trainer | trainer.py:36-39 (`resolve_backend`), 45-47 (`make_policy`), 50 (`Trainer`), 130-191 (stage configure/advance), 260-263 (save), 292-327 (resume), 384+ (`evaluate`) | SIGINT-safe resume; deterministic eval |
| Checkpoints | checkpoint.py:12 (`FORMAT_VERSION=1`), 60-82 (`save_checkpoint`, atomic), 84-95 (`load_checkpoint`), 97-107 (`apply_checkpoint`), 135+ (`warm_start_from_bc`) | BC/teacher warm start contract (key/shape checked, never partial) |
| Eval script | scripts/eval_ppo.py | movement summary: standing/ground/knee-hand/dorsal fractions, mean pelvis distance, cause mix, W/L/D; determinism check |

### 2.4 Retarget / reference machinery (`src/retarget/`, `scripts/`)

- Parser: `src/grapplemap/parser.py:493-` (`load_graph`; 601 nodes/1485 edges/725 linked),
  `decode_position` (parser.py:109), mirror/swap (parser.py:153/164), `Reorientation`
  (parser.py:183-205).
- Frame/chain assembly: `gmframe.py:41-99` (`PairTransform`), 102-148 (`yaw_align`,
  `best_pair_align`), 163-229 (`assemble_chain`), 231+ (`single_player_align`).
- World placement + scale: `world.py:49-69` (`estimate_player_scale`), 71-119 (`place_world`;
  chirality-correct Y-up→Z-up at world.py:104-105).
- Landmark map/weights: `landmarks.py:108-167` (`_Map`, `gm_weights`, `site_weights`),
  169-190 (`load_g1_spec`, `g1_reference_geometry`).
- Solve + resample: `solve.py:90-109` (`G1Kinematics`), 112-274 (`PairSolver`),
  275-310 (`solve_keyframes`), 331+ (`resample_50hz` PCHIP + vel/acc limits).
- Technique assembly: `techniques.py:91-99` (`build_technique_targets`), 112-254 (`_simple`,
  `_stance`, `_sprawl`, `_decollide_pair`), 256+ (`build_technique`).
- Scene: `scene.py:35-73` (`build_scene_spec`, `load_scene_model`), 76-87 (`write_scene_xml`),
  89+ (`robot_slice`); artifacts `robots/wrestling_scene.xml`, `robots/g1/`.
- Build/validate: `scripts/build_refs.py`, `scripts/validate_refs.py:67-137` (PD tracking),
  140-178 (acceptance), 199-221 (`render_video`, 960×720 30 fps), 223-262 (repair loop);
  `data/refs/*.npz` (7).

### 2.5 Technique scorer (`src/scorer/`)

- `TechniqueScorer` (scorer.py:47-132): `score` (62), `phase_at` (85), `phase_at_frac` (92),
  `score_trace` (102); module-level `score` (135). `spec.py` (phase bands + weight-2 predicates;
  penetration-step predicate at spec.py:62), `features.py` (25 relational features, one
  `mj_kinematics`), `membership.py`, `calibration.py`; data `data/scorer_calibration.json`;
  scripts `scripts/calibrate_scorer.py`, `scripts/score_trace.py`. 200 µs/call measured
  (reports/2026-10-08/scorer.md).

### 2.6 Renderers / video pipeline

- `scripts/validate_refs.py:199-221` (reference videos, 960×720@30 fps, EGL) → `videos/refs/*.mp4`.
- `scripts/render_env_videos.py` (~885 lines at the latest read; the file was being extended during the audit): scenario harness with HUD, `SlideController`
  (:163), `EvidenceEnv` (:238, allows a scripted standing start for every exchange while keeping
  all rules), `Scenario` (:297), `capture` (:409), render-only mat ring (:581); outputs
  `videos/env/*.mp4` + 3-frame PNG sheets. This is the template for M1–M5 evidence videos.
- Scene render size: `robots/g1/scene.xml` offwidth/offheight 960×720 edit (reports/2026-10-07/g1_model.md
  "Model edits"); `robots/wrestling_scene.xml` composed.

### 2.7 Teacher (present, DRAFT/in-flux)

`src/teacher/` (revision ~04:36; **DRAFT report** reports/2026-10-08/teacher.md):
`controller.py:197` (`TeacherController`), :261 (`control`), :269-365 (`_control_robot`: CoM :277,
foot contact/support reads :285-286, capture error call :292, direct channels :314-317 incl. a yaw
channel shipped disabled, governor :346-357); `stabilizers.py` (measured `AUTH` table 46-53,
`YAW_AUTH` 61, `channel_offsets` 77, `capture_error` 120, `foot_rows` 128, `upright_rows` 159,
`solve_offsets` 177, `support_center` 189, `foot_contact` 198, `heading_yaw` 207); `phases.py`
(STAND/RISE/DIVE/LOW/PRONE labelling); `gains.py` (`GainSet` :43, `GAIN_TABLE`, `k_yaw=0`);
`episode.py` (rollout + metrics; :59 `_perturb_initial`, :166 `run_episode`); `trims.py`
(`load_trims` :43 from `data/teacher_trims.json`); `scripts/tune_teacher.py` +
`scripts/run_teacher.py` (new). Supporting artifacts: `data/teacher_trims.json` (STANCE trims),
`data/teacher_stats.json` (5 seeds × 5 techniques: stay-up/scorer/penetration — §1c),
`videos/teacher/{DOUBLE_LEG,SNAPDOWN}.mp4` (2 of 7), `tests/test_teacher.py` (2 failures reported
by reports/2026-10-08/p0_fixes.md §4 at its snapshot).
**Status: DRAFT** — the report's acceptance table is unfilled; treat every number as in-flux
until the report is finalized.

### 2.8 What a balance/locomotion training task could reuse **today** (no new code)

1. **Sim + scene + action interface**: `WrestlingEnv` at 50 Hz with 29 position targets/robot and
   ctrlrange clipping (env.py:423-620) — M1/M2 can run in the same scene, including a two-robot
   variant or a single-learner/scripted-opponent variant (`VecWrestlingEnv` mode `learner_only`).
2. **Reset/seed determinism + randomization** (env.py:463-527) and per-exchange seed streams
   (vec.py:37) — reusable for held-out condition generation (add new fields; yaw/distance/joint
   noise knobs already there).
3. **The whole RL stack**: obs builder (swap-in new obs), privileged critic, PPO, GAE, curriculum
   schema, checkpoint/resume, subproc vec, deterministic `evaluate()` — all proven end-to-end by
   the smoke runs; BC warm-start contract ready for teacher labels.
4. **Scripted controllers** (`StandHold`, `ReferenceReplay`) as baselines/opponents; teacher
   wrapper slot (`OpponentSpec("teacher", ...)`) pre-wired.
5. **Metrics that already run**: `scripts/eval_ppo.py` movement fractions; `validate_refs.py`
   tracking/penetration/stay-up; `teacher/episode.py` stay-up/travel (unverified but written);
   backdet events; scorer for technique form.
6. **Video pipeline** (`render_env_videos.py` scenario+HUD pattern).
7. **Geometry primitives for CoM/support** (teacher's `subtree_com`, `support_center`,
   `foot_contact`) — reusable *after* verification, or trivially re-derived with MuJoCo APIs.

**Not reusable / missing:** push forces, fall detector, velocity-command obs/reward, support
polygon, locomotion gait/reference, any learned balance policy, any multi-task/forgetting
mechanism, any historical checkpoint league.

---

## 3. Structural findings relevant to a motor curriculum

### 3.1 Where catastrophic forgetting would bite

1. **One policy object is carried across the whole stage ladder.** `Trainer._configure_stage`
   (trainer.py:130-181) rebuilds the vec env, scheduler, reward and opponent per stage but keeps
   the same `self.policy` weights; `_on_stage_advance` (trainer.py:183-191) only logs
   "env/opponent rebuilt". There is **no replay, regularization, frozen sub-network, or
   multi-task evaluation** anywhere in `src/rl/` (grep: none). Every later stage is free to
   overwrite earlier behavior. The MOTOR_CURRICULUM §4 plan ("re-test M1/M2 gates after M3–M5
   training") is exactly the right test, and today nothing in the code records per-stage gate
   metrics to re-test.
2. **The observation/action interface is a hard, shared contract.** Actor obs = `frame_stack*84 +
   8` (obs.py:39-50) and action = 29 ctrl targets (net.py:161+). M1–M4 will need additional obs
   slots (velocity command, disturbance signal, support/CoM features) and/or a fall terminal.
   Changing 84 → N makes every existing checkpoint shape-incompatible: `Trainer.load` guards only
   `frame_stack`/`hidden` equality (trainer.py:296-299) and then `apply_checkpoint`
   (checkpoint.py:97-107) does a strict `load_state_dict` — the failure will be loud (good), but
   it means the current 5 smoke checkpoints are the *end* of this interface generation.
3. **Curriculum advancement state is deliberately per-stage and gets discarded.** The rolling
   window is re-created with the new stage's size on advance (curriculum.py:464 in the current
   revision), and checkpoints store only `{index, steps_in_stage, window}` where window entries
   are now `ExchangeSample`s (curriculum.py:469-471). A model that regressed on an earlier
   capability can therefore advance again on the new metric with no cross-stage record.
   (Compat note, p0_fixes §6.5: a pre-fix checkpoint's int window / `threshold` fields now fail
   loudly on load — by design; no such checkpoint exists in the repo.)
4. **The stage config (weights/perturbations/opponent) is checkpointed with the weights**
   (state["stage"], trainer.py:239-258 + stage_from_dict), so the exact training conditions of a
   checkpoint remain reproducible — that half of the contract is solid and should be preserved.
5. **No snapshot league exists.** Stage E uses an in-memory `deepcopy` of the current policy as a
   frozen opponent (trainer.py:149-159); `OpponentSpec("policy", checkpoint=...)` can load a
   specific checkpoint as opponent (scripted.py; trainer.py:152-156), but nothing writes
   historical snapshots to disk automatically. Forgetting becomes invisible relative to *past*
   selves unless a harness writes those snapshots.

### 3.2 Advancement logic: what it was, the mid-audit fix, and what remains insufficient

**Pre-fix behaviour (HEAD revision).** `AdvanceRule(min_steps, metric="outcome_rate",
threshold=0.5, window_episodes)` with `metric = (wins + 0.5·draws)/window`; an all-draw window
scored exactly 0.5 and could promote a stage with zero demonstrated success, and `1 win + 9
draws` scored 0.55 — both reproduced against the old code by the fixer
(reports/2026-10-08/p0_fixes.md §1.1). Stage A 20 k steps / 10 exchanges / 0.5, B 30 k / 20 / 0.5,
C and D 40 k / 20 / 0.5; stage E has no rule (reports/2026-10-08/rl_infra.md §8).

**Final API after two mid-audit revisions** (worktree revision ~04:51;
`reports/2026-10-08/p0_fixes.md` §1 records the orchestrator correction that superseded an
intermediate wins-only revision). The rolling window holds one
`ExchangeSample(outcome, similarity, stood)` per finished exchange (curriculum.py:90-120). A
stage advances iff `steps_in_stage ≥ min_steps` ∧ full window ∧ `successes ≥ min_successes`
(≥ 1, constructor invariant) ∧ `successes / window ≥ min_rate` (∈ (0,1], invariant)
(curriculum.py:124-196, 448-466). The **criterion is per stage** (curriculum.py:230-289):
A–C `"execution"` = the learner stayed up (no own back trigger, no OOB forfeit) **and** the
attempt scored ≥ `min_similarity` on the technique scorer — an unmeasured attempt is *not
evidence*; D `"outcome"` = learner wins (draws/losses are non-success slots); E none.
`success_hook` (`ExchangeSample -> bool`, runtime-only, omitted from `as_dict`) can only *reject*
a success — the documented attribution seam (curriculum.py:154-165). Invariants raise on
degenerate gates (curriculum.py:176-190). The signal path is wired and verified in the tree: the
rollout collector attaches the per-exchange mean scorer similarity to the exchange record
(rollout.py:98-101, 182-200), the trainer computes `stood` and calls
`on_exchange(outcome, similarity, stood)` (trainer.py:124-132), and the reward exports the scored
sample (reward.py:166-191). `gate_status()` reports
criterion/window/successes/metric/unmeasured/min_* for stuck-stage diagnosis (curriculum.py:422-432).

This removes both measured holes of the old rule (draw-only advancement; half-credit for draws)
and makes "no evidence → no promotion" structural. It does **not** yet remove the central one:

1. **D and E still advance on wins, and wins can still be earned by the opponent collapsing.**
   The execution criterion protects A–C from the collapsing-opponent failure, but D's outcome
   gate is explicitly flagged by the fixer as "still satisfiable by a collapsing opponent"
   (p0_fixes §6.3), with the attribution hook left as a placeholder. The audited counterexample
   is D-shaped: stage-C smoke `learner W/L/D 299/0/20` at 30 k steps while the *same checkpoint*
   evaluates to standing fraction 0.131 / ground 0.830 (`data/rl/log_stage_c.log`;
   reports/2026-10-08/rl_infra.md §6.1-6.2; `notes.md` E1).
2. **Attribution is still unimplemented** — `success_hook` is a documented seam only; nothing in
   `src/rl/` sets it (grep), so outcome-criterion wins still count regardless of cause.
3. **The gates are still not capability gates.** The criterion choices are execution/outcome
   only; there is no slot for held-out push recovery, velocity-tracking error, fall rate,
   uprightness, idle energy or CoM margin (MOTOR_CURRICULUM §2 columns). Those harnesses and
   their metrics must still be built and wired into the window sample payload.
4. **No held-out conditions anywhere**: the gate consumes the same scripted opponents and the
   same randomization ranges as training; and `min_similarity = 0.5` (A–C) is a placeholder —
   mid-scale of the scorer's own reference range 0.46–0.74, not a calibrated recognition gate
   (p0_fixes §1.5/§6.1); `min_steps`/`min_rate` are skeleton values.
5. **Window reset discards history on advance** (curriculum.py:464) and nothing records
   per-capability gate values across stages, so a regression cannot reopen a stage (§3.1.3).
6. **Related reward finding (p0_fixes §5, no behaviour change):** the `progress` term is a
   command-clock potential — identical for executing, standing still, or lying down — so it
   carries no physical progress signal; it must default to 0 in M1–M4 tasks and physical progress
   must be measured instead.
7. **Latent quirk flagged, not fixed (p0_fixes §6.4):** `StageReward.shaping` scores only when
   `counter % score_every == 1`, so `score_every=1` silently disables scoring (default 5 is
   unaffected); relevant if an M-task gate wants per-step scoring.

The supersession is already decided in `docs/MOTOR_CURRICULUM.md` §3/§7: per-capability gates
with held-out conditions for M1–M4, plus "success attributable to the attempted technique" for
M5. This audit's contribution: the measured counterexample (item 1) and the identification of
the draw-only advancement hole, both of which the P0 fixes now close structurally; the
per-capability metric harnesses and attribution logic are the remaining work.

### 3.3 Late-landing changes during this audit (re-verify before building on them)

Four change-sets landed in files this audit cites while it ran (worktree revisions ~04:33–~04:51).
All are documented by their authors; re-verify before building on them:

1. **Advancement gate rewrite + revision** (`src/rl/curriculum.py`; final API §3.2). Its report
   `reports/2026-10-08/p0_fixes.md` now exists, with old-code replays of both defects and 12+ new
   tests; at its snapshot `tests/test_rl.py + tests/test_wrestling.py` = 54 passed, full suite
   99 passed / 2 failed in the peer-owned `tests/test_teacher.py` (p0_fixes §3-4). Re-verify:
   `python -m src.rl.curriculum` self-check, `tests/test_rl.py`, and checkpoint compatibility
   (pre-fix windows fail loudly on load; no such checkpoint exists — p0_fixes §6.5).
2. **Back-trigger ambiguity hold** (`src/wrestling/env.py`): the exchange no longer ends on the
   first back trigger; it stays open up to `AMBIGUITY_WINDOW_S` = 0.10 s (5 control steps) after
   the first trigger so the second trigger of a near-simultaneous pair can be observed, after
   which `resolve_back_events` decides winner/ambiguity (env.py:107-115 comment block;
   `_back_hold_start` init/reset env.py:493, 704; hold logic in `_maybe_end_exchange`, env.py:634-690).
   Measured consequence: the `takedown_back_event` clip was re-rendered against the new code and
   now ends at 1.80 s (trigger 1.70 s + hold); the other four scenarios never fire a back trigger;
   all five re-verified with `--check` (reports/2026-10-08/env_videos.md §6.9).
3. **Teacher phase-3A artifacts.** `reports/2026-10-08/teacher.md` now exists as a **DRAFT** (159
   lines; acceptance table unfilled); `data/teacher_stats.json` (5 seeds × 5 techniques),
   `data/teacher_trims.json` (STANCE trims), `videos/teacher/{DOUBLE_LEG,SNAPDOWN}.mp4` (2 of 7)
   and `tests/test_teacher.py` landed. Measured excerpts are folded into §1a/§1c; the report's own
   §9 iteration log is the primary source (e.g. #7: defenders fall without contact — DOUBLE_LEG
   defender t ≈ 2.0 s, SINGLE_LEG t ≈ 1.0 s; #8: STAND_UP reference airborne). Re-verify once the
   draft is finalized.
4. **Env-video report** (`reports/2026-10-08/env_videos.md`): the five rule-evidence clips now
   carry measured tables and honest deviations (STANCE not holdable → 1.0 s-clock draw recipe;
   `EvidenceEnv` start-pose shim; no recovery controller; scripted root drag for OOB). These are
   the reference evidence for the rules layer; nothing under `src/` was changed by it.

---

## 4. Five biggest uncertainties for the motor curriculum (each with its smallest experiment)

**U1 — Can this actuator interface express sustained dynamic balance at all?**
The G1 here is a position-servo model (kp=500, dampratio=1; reports/2026-10-07/g1_model.md) with a
free-floating base; the phase-2 and env reports show that reference poses exist whose CoM sits
outside the effective support (STANCE: CoM behind a one-point toe support; ankle demand 39/50 Nm —
reports/2026-10-08/wrestling_env.md §3.1) and that pure joint targeting topples. It is currently
unknown whether a *learned* policy in this action space can regulate CoM continuously, or whether
the sim needs different actuators/gains (torque, kp schedules, foot-ground friction/geometry). 
*Smallest experiment:* build the M1 harness (stand-keyframe start, no opponent, fall terminal =
pelvis z<0.45 m or tilt>30° persisted 0.1 s) and run three baselines on identical seeds: (a) zero
action, (b) `StandHold` keyframe targets, (c) `TeacherController` with all flags on (DRAFT, in-flux
— use as-is, report honestly) — then 300 k PPO steps. Predict: (b) survives ≥60 s; (c) survives;
(a) falls in <2 s. If a trained policy cannot beat (b) on held-out initial tilts (2°/4°/6°) within
300 k steps, the action/observation interface, not the algorithm, is the finding.

**U2 — Does the existing env/harness express the M1/M2 tasks, or must a motor task env be built?**
The shipped env terminates only on the match clock and scores only exchanges/OOB
(src/wrestling/env.py:549-690); a robot that falls sideways or to knees is never terminated, so a
balance metric has nowhere to live, and there is no command channel. *Smallest experiment:*
implement a minimal `MotorTaskEnv` wrapper (subclass or obs_fn/reward_fn swap) with fall
detection + a 2-value velocity command in obs, then measure detection correctness on scripted
falls: drive the pelvis/pose through 5 fall directions (side-left/right, forward, backward,
knees) and count backdet hits vs the new fall rule (prediction: backdet catches only
backward/supine falls — it fires only on persistent dorsal torso contact,
src/wrestling/backdet.py:235-241). This one experiment decides the harness design and produces
the fall-rate denominator for every later gate.

**U3 — How much of the retargeted wrestling geometry is outside the static support envelope
(and therefore needs trims/re-posing before any balance learning)?**
The schematics preserve relationships but are not equilibria (STANCE documented; trims.py claims
the same for shot entry/exit frames, UNVERIFIED). If a large fraction of every reference frame has
CoM outside the support and/or ankle demand near saturation, plant-tracking policies will be
fighting the references. *Smallest experiment:* a per-frame equilibrium audit of all 7 refs using
MuJoCo kinematics only: for each frame and robot compute `subtree_com` vs the convex hull of
touching sole sites (toe/heel sites already exist; hull code must be written, ~20 lines), and the
static ankle-pitch torque needed to hold the pose; report the fraction of frames with
margin < 0 or torque > 45 Nm. Prediction (from the two documented cases): STANCE fails at most
frames, shot entries exceed 45 Nm, PRONE ground frames are not applicable. This is also the audit
that decides how much `teacher/trims.py` must fix vs how much must be re-retargeted.

**U4 — Can the outcome-gated PISTY ladder be salvaged for wrestling (M5), and what replaces it
for M1–M4?** The measured counterexample (win rate ~1.0 for an 83%-ground policy) proves
outcome-rate gates are unusable as written; the open question is whether a *per-capability gate
suite* (fall rate under held-out pushes, velocity error, uprightness, scorer form) is sufficient
for M5's "success attributable to the technique" requirement, and how much of the existing stage
machinery survives. *Smallest experiment:* extend `evaluate()` with the M-gate metrics (held-out
push battery with 3 unseen magnitudes, fall rate, velocity-tracking error, uprightness) and score
`checkpoints/rl/smoke_stage_c.pt` with it. Prediction: it scores near-zero on all M-gates while
the existing outcome metric scores ~1.0 — a single table proving the gate gap on real weights,
and the baseline table the new curriculum is measured against.

**U5 — What sample/wall-clock budget does a 4-core CPU budget allow per gate (and does the
92→N-dim obs PPO learn balance/locomotion at that budget)?**
Measured throughput is 130–437 env-steps/s depending on backend/load (reports/2026-10-08/rl_infra.md
§5, §11); a gate that needs, say, 10 M env steps is 6–20 h of wall clock at those rates, which is
feasible overnight but not for many stages, and no one has measured a learning *slope* for any
motor task yet. *Smallest experiment:* benchmark the U2 harness at 3 subproc envs, then run a
single 1 M-step PPO training on the simplest M1 task with a fixed seed and log the quiet-stand
survival time every 50 k steps; fit slope vs E1's 30 k-step baseline. Prediction: ≥450 steps/s on
the 1-robot variant (half the contacts/robots), and a measurable monotone improvement by 300 k
steps. If the slope is flat, the finding is to simplify the task (1 robot, no opponent) or the
architecture before spending multi-hour runs.

---

## 5. Reuse check: does any existing code compute CoM, foot contacts, support polygon, or capture point?

**Method:** searched `src/` and `scripts/` for `subtree_com|center.?of.?mass|com_|capture.?point|
support.?polygon|polytope|centroid|ConvexHull|qhull|hull|polygon|scipy.spatial|cop\b|zero.?moment`
and for contact usage (`foot_contact|foot_rows|support|contacts`). Findings, quoted:

1. **CoM — YES, one place, in-flux teacher.** `src/teacher/controller.py:277`:
   `com = data.subtree_com[ctx.pelvis].copy()`. Nothing else in `src/` or `scripts/` reads
   `subtree_com` or computes a center of mass (grep = this single hit). The retarget pipeline uses
   the pair *landmark centroid* for placement only (`src/retarget/world.py:99`
   `c3 = 0.5 * (track_a[:, CORE] + track_b[:, CORE]).mean(0)`), which is a GrappleMap-landmark
   centroid, not the robot's mass CoM.
2. **Capture point — YES, one place, in-flux teacher.** `src/teacher/stabilizers.py:120-124`:
   ```python
   def capture_error(com: np.ndarray, v_com: np.ndarray, support: np.ndarray,
                     kd: float, clamp: float) -> np.ndarray:
       """World-frame capture-point error vs the support centre (m), clamped."""
       h = max(com[2] - 0.05, 0.30)
       xi = com[:2] + kd * np.sqrt(h / G) * v_com
       return np.clip(xi - support, -clamp, clamp)
   ```
   with `G = 9.81` (stabilizers.py:69) and the measured-authority channel routing
   (`channel_offsets`, stabilizers.py:77-99; `AUTH` table, stabilizers.py:46-53). Used in
   `controller.py:292`. No other capture-point / zero-moment computation exists.
3. **Foot contacts — YES, several independent mechanisms:**
   - Env obs: geom-identity check `floor` vs `left/right_ankle_roll_link` geoms, 2 bits
     (`src/wrestling/env.py:207-216` (`_foot_geom_ids`), :255-258, :267).
   - Privileged critic: 5 floor-contact categories `("feet","knees","hands","torso","pelvis")`
     + opponent-contact count (`src/rl/privileged.py:47`, `ContactCategorizer.floor_flags`
     :128-144, wired at :179-181).
   - Teacher (in-flux): site-height contact `data.site_xpos[...][2] < ctx.touch_z` with
     `FOOT_Z_TOUCH = 0.045` (`src/teacher/controller.py:65-67`; `stabilizers.foot_contact`
     :198-203; reference touch table `ref_touch`, controller.py:143-163).
   - Backdet: torso/pelvis dorsal contact via contact-point geometry (not forces) plus a
     limb-contact diagnostic (`src/wrestling/backdet.py:164-216`; `body_maps` :131).
   - No code uses contact **forces/impulses** (`mj_contactForce` etc.); all are geometric.
4. **Support polygon — NONE at audit snapshot.** As of the audit window there was no convex hull,
   polytope, or polygon computation anywhere in `src/` or `scripts/` (grep hits: only
   `scipy.spatial.transform.Rotation` for quaternions in `src/retarget/solve.py:27`). The closest
   was a **support centre** = mean of currently touching sole-site points, with a fallback to the
   reference frame's touching sites: `src/teacher/stabilizers.py:189-195` (`support_center`; used
   at controller.py:286), plus precomputed `ref_support` (controller.py:142-163). That is a point,
   not a polygon. **Post-audit (outside this snapshot):** a hull implementation landed in the new
   drill package — `src/drill/kin.py:244` (`hull2d`), `:201` (`support_polygon` over loaded foot
   footprints), `:267` (`polygon_margin`, signed CoM margin), used by the drill balance law and
   gates (src/drill/balance.py:288/383-384; src/drill/metrics.py:123+ push-recovery report). This
   was not audited here and must be certified separately (reports/2026-10-08/drill.md,
   drill_l2.md, support_envelope.md).
5. **Bottom line for the M-curriculum harness:** at the audit snapshot, CoM, foot contacts and a
   capture-point-style error were written once (teacher, **DRAFT/in-flux**), the support polygon
   did not exist, and CoM/contacts were trivially re-derivable from MuJoCo APIs at ~µs cost; the
   toe/heel sites a hull would need are already attached (`src/retarget/landmarks.py`, sites
   `left_toe/left_heel/right_toe/right_heel`). **Post-audit:** the new `src/drill` package
   implements exactly those primitives (hull, signed CoM margin, push-recovery report) — certify
   it separately (§6 post-audit arrivals).

---

## 6. Appendix — snapshot provenance

- HEAD: `0b60780`. Worktree dirty at end of audit (teacher, env-video and gate-fix work in
  flight). `git status --short`:
  `M src/rl/curriculum.py`, `M src/teacher/{__init__,controller,episode,gains}.py`,
  `M src/wrestling/env.py`; `?? data/{scorer_calibration.json,teacher_trims.json}`,
  `?? reports/2026-10-08/motor_audit.md`, `?? scripts/{render_env_videos,run_teacher,tune_teacher}.py`,
  `?? src/teacher/{stabilizers,trims}.py`, `?? videos/env/`.
- `src/teacher/` MD5 snapshot (**revision used for the teacher citations, ~04:36 UTC**):
  controller `af5b3f5d…`, episode `021c7743…`, gains `3c0f1bdf…`, phases `b6eef5b0…`,
  stabilizers `82d59e00…`, trims `2a0374d9…`; new scripts `tune_teacher.py` `af7100e7…`,
  `run_teacher.py` `e39afb6a…`. **Drift warning:** these files kept changing all audit long;
  by ~05:05 the controller/stabilizers had changed again (`a1e63ab0…` / `4997dab3…`), so the
  teacher line numbers above are snapshot-scoped to the 04:36 set — re-derive before reuse.
- Core-file hashes used for the re-anchored citations (~04:51): `curriculum.py` `3c199a84…`,
  `wrestling/env.py` `5f6200f6…`, `rl/reward.py` `6bb2f660…`, `rl/rollout.py` `950285de…`,
  `rl/trainer.py` `e64f0c1f…`.
- `videos/env/`: all 5 scenario mp4s + PNG sheets present; `reports/2026-10-08/env_videos.md`
  delivered (331 lines).
- Checkpoints present: `checkpoints/rl/{smoke_stage_c,smoke_hold,smoke_int,smoke_fs4,smoke_residual}.pt`
  (harness smoke runs only — no learning-quality claim, reports/2026-10-08/rl_infra.md §11.1).
- Test suite context: last project-wide figure recorded in `notes.md` was 89 passed; the p0_fixes
  snapshot reports 99 passed / 2 failed in a peer-owned `tests/test_teacher.py` at its own run;
  the suite is in flux while multiple agents work. Re-run before quoting a number.

### Post-audit arrivals (~04:55–05:05 UTC, NOT audited in this report)

The worktree advanced substantially after this audit's evidence window. **None of the following
appears in the capability tables above**; each landed with its own report and must be re-audited
under the same evidence classes before it is treated as demonstrated:

- **New motor packages**: `src/drill/` (4740 lines: `balance.py`, `stepping.py`, `tracking.py`,
  `rubric.py`, `runner.py`, `scheduler.py`, `lock.py`, `posture.py`, `metrics.py`, `kin.py`,
  `controller.py`, `scene.py`, `video.py`) and `src/solo/` (`env.py`, `fall.py`, `commands.py`,
  `eval.py`, `baselines.py`, `metrics.py`, `markers.py`, `lock.py`).
- **Measured claims already written down** (verify before reuse): `reports/2026-10-08/drill.md`,
  `drill_l2.md` (margin-gated stepping; 4-step shuffle 34 s / 0 falls; L1-cycle 3 steps then a
  fall; the 0.495 m drill stance is *not* steppable — needs 0.254–0.276 m travel vs 0.14 m
  authority), `support_envelope.md` (only 6.9 % of reference robot-frames are statically
  holdable), `solo_env.md`, `teacher_exec_fix.md`, `env_videos_frame_check.md`.
- **New scripts/data/videos**: `audit_support_envelope.py`, `build_standup_ref.py`,
  `run_solo_drill.py`, `solo_drill_render.py`, `solo_env_smoke.py`, `teacher_stance_sweep.py`,
  `teacher_step_lab.py`; `data/drill/` (L1/L2 suite summaries + feasible seeds), `data/solo_drill/`,
  `data/refs_video/` (video-retargeted references), `data/references/`; `videos/solo_drill/`,
  `videos/teacher/*` (7/7); `robots/wrestling_scene_soft.xml`;
  `docs/CURRICULUM_REFERENCE_ADDENDUM.md` + `docs/references/`.
- **Implication for this audit's ABSENT rows**: the "shuffle/circle" and "support polygon" rows
  are already superseded in the live tree (see the post-audit notes in §1c and §5), and the M1/M2
  task harness may now exist in `src/solo` / `src/drill`; U3 (reference support envelope) is
  partially answered by `support_envelope.md`. This report remains the record of the state the
  audit actually examined and of what was measured *then*.
