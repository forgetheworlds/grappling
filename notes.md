# notes.md — Orchestrator Knowledge

Living ledger. Every session appends. Distinguish FACT (verified) / HYPOTHESIS /
DECISION. Interface contracts and experiment results go here. Subagents MUST read
this before working and MUST append their verified facts.

## Operating conventions (fixed)

- Orchestrator (this assistant) does NOT write project code. Code via `task` subagents
  (workflow runtime broken, see below). The `delegate` driver (external harnesses:
  opencode/cursor/freebuff) is NOT used for project work unless the operator explicitly
  approves per-session ("do not use delegator unless i say ok", 2026-10-07). Operator
  clarified target stack: **opencode go** (`opencode-go/*`) with deepseek flash +
  mimo flash; fireworks-ai/* EXCLUDED. Note: no fireworks auth found on host, so the
  safe default once approved is the free opencode/mimo-v2.6-flash-free. Pending
  HarnessProbe report + explicit operator OK; verification bursts stay on `task`/zai.
- Orchestrator writes: goal.md, notes.md, AGENTS.md, briefs, workflow scripts, ledger.
- Git: repo initialized (main @ 8c85339, phases 0-2). Remote = forgetheworld/grappling
  (private) once access granted (gh token is clawdbotinstaller; cannot create under
  forgetheworld — needs gh auth login as forgetheworld OR collaborator invite).
  Orchestrator commits at every verified milestone + session end; subagents NEVER
  commit/push (single-writer). third_party/ + .venv/ gitignored; regen per SETUP.md.
- Repo: /home/ubuntu/grappling. Reports from agents: reports/YYYY-MM-DD/<topic>.md.
- Units: SI (m, rad, s, N, Nm). Angles radians in code; degrees only at human UIs.
- Frames: MuJoCo world = Z-up. Document every cross-frame transform in notes.md
  section "Interface contracts".
- Python: single venv at .venv (see docs/SETUP.md once written).
- Hardware budget: ~8 ARM cores. Keep sims fork-parallel, nets small (<2M params).

## Session log

### 2026-10-07 (S1) — bootstrap
- DECISION: bootstrap repo skeleton (goal.md, notes.md, AGENTS.md), then workflow
  wave 1 = EnvSetup → {GrappleMapParser, G1Model} (EnvSetup is a prerequisite:
  installs mujoco/numpy/torch-CPU into .venv). Wave 1 now runs via `task` directly.
### Workflow runtime — RESOLVED 2026-10-08 (was misdiagnosed as broken)
- CORRECTION: the engine is fine. It strips ONLY the first export (`export const meta`)
  and compiles the remainder as a classic vm.Script; any further export (esp.
  `export default`) → "Unexpected keyword 'export'". Doc-conformant shape:
  `export const meta = {...}` as first statement + PLAIN BODY (top-level await/return
  legal; NO export default, no other exports, no imports). Root cause in
  pi-dynamic-workflows dist/workflow.js parseWorkflowScript (~l.1384-1413).
- VERIFIED 2026-10-08 (orchestrator probe, run probe-ok-muyv2zb5): agent() returned
  WF_VERIFY_OK. Workflows are the preferred fan-out again; see rewritten skill
  ~/.omp/agent/managed-skills/omp-workflow-esm-fallback/SKILL.md.
- DelegateTester (glm-5.3-flash) also validated the `delegate` tool: new driver at
  /home/ubuntu/bin/delegate; opencode headless free mimo flash OK (stdin=DEVNULL
  critical); freebuff via tmux+log.jsonl OK (~75 Freebucks left; deepseek-v4.1-flash
  card available; its "MiMo 2.6 Flash" card serves mimo-v2.5). Operator approval
  still required before delegate use for project work.
- Environment recon command issued (CPU/mem/pip/bins) — results to be recorded below.

## Environment FACTS

### From EnvSetup (2026-10-07, verified)
- FACT: .venv works: Python 3.12.3, pip 26.2.1. Installed: mujoco 3.15.0,
  numpy 2.5.3, scipy 1.18.1, pytest 9.1.1, imageio 2.38.0,
  imageio-ffmpeg 0.6.0, matplotlib 3.11.2, torch 2.14.1+cpu.
- FACT: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  works first-try on aarch64 (no PyPI fallback needed).
  `torch.cuda.is_available()`=False, `torch.cpu.is_available()`=True.
- FACT: headless MuJoCo render works with `MUJOCO_GL=egl` (software EGL via
  Mesa; no apt installs needed). Expect cosmetic stderr
  `libEGL warning: failed to open /dev/dri/...` on this GPU-less host.
- FACT: calling `renderer.close()` explicitly avoids EGL-free
  "Exception ignored in __del__" noise at interpreter exit. All rendering
  scripts MUST close renderers explicitly.
- FACT: scripts/smoke_env.py green from repo root (exit 0); PNG at
  reports/2026-10-07/smoke_env.png (640x480, 29.9 KiB). Setup doc:
  docs/SETUP.md. Full report: reports/2026-10-07/env_setup.md.

### From initial recon (2026-10-07, orchestrator)
- FACT: 4 cores ARM Neoverse-N1 aarch64 (NOT 8) → ≤3 sim workers + 1 trainer; small nets.
- FACT: 23 GiB RAM (18 GiB available), 98 GB disk free, no swap. Python 3.12.3;
  ffmpeg, git, cmake present; passwordless sudo; outbound network OK.
- DECISION: all Python work in .venv (Ubuntu 24.04 PEP 668 blocks system pip).

## Model FACTS

### From G1Model (2026-10-07, verified)
- FACT: mujoco_menagerie cloned (depth 1) at third_party/menagerie. Its
  unitree_g1 has NO 23-dof variant; variants are g1.xml (29 dof, mesh
  collision, "stand" keyframe, position actuators kp=500 dampratio=1),
  g1_mjx.xml (29 dof, capsule colliders + explicit pairs, kp=75 kv=2,
  "home"/"knees_bent" keyframes live in scene_mjx.xml), g1_with_hands.xml
  (43 dof = 29 + 7/hand). Unitree's own g1_description repo has 23-dof but
  menagerie does not vendor it.
- FACT: G1 (29-dof rev 1.0) vendored at robots/g1/ (g1.xml + scene.xml +
  assets/ + LICENSE). nq=36 nv=35 nu=29 na=0, nbody=31, ngeom=72,
  timestep=0.002 s (implicitfast), total mass 33.341 kg. All 29 actuated
  joints are hinges w/ limited range + actuatorfrcrange; base is a free joint.
  Actuators are position servos, ctrlrange=joint range (inheritrange=1).
- FACT: standing hold (keyframe "stand", pelvis z=0.79, ctrl=keyframe joint
  targets) for 5 s: pelvis 0.7900 -> 0.7916 m (+0.16 cm), torso up-axis tilt
  <= 0.172 deg, pelvis xy drift 0.1 mm, 8 foot contacts throughout -> G1
  STANDS under model servos. scripts/smoke_g1.py exit 0; renders
  reports/2026-10-07/g1_front.png + g1_side.png. Full report:
  reports/2026-10-07/g1_model.md.
- Only edit vs upstream: robots/g1/scene.xml adds offwidth=960/offheight=720
  (upstream framebuffer caps renders at 640x480).

## Interface contracts

### G1 qpos layout (robots/g1/g1.xml, order FIXED, feeds Phase 2 retargeting)
- qpos[0:7] floating base (x y z qw qx qy qz, pelvis); then 29 hinges in
  model order: L leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch,
  ankle_roll), R leg (same 6), waist (yaw, roll, pitch), L arm (shoulder_p/r/y,
  elbow, wrist_roll/pitch/yaw), R arm (same 7). qvel: 6 (base) + 29, same order.
- Actuator i (0..28) drives hinge joint i+1 exactly (trnid order equals
  joint order); actuator ctrl target in rad.
- qpos addresses: legs 7-18, waist 19-21, left arm 22-28, right arm 29-35.
- GrappleMap tNNN ids = edge list index: g.edges[NNN] (verified t1144 =
  'drop level slightly', 2 frames).
- Frame conversion (CRITICAL, canonical): GrappleMap is Y-up, MuJoCo world is Z-up.
  PROPER rotation: (x,y,z)_gm -> (x, -z, y)_mj. Ground: gm y=0 == mj z=0.
  (The naive (x,z,y) swap is an IMPROPER mirror, det=-1 — it made robots face
  away and fall; corrected + regression-tested in Phase 2, see session log
  2026-10-08 entry and reports/2026-10-07/retarget.md §2. Never use the swap.)
- LESSON (orchestrator, 2026-10-08): the first phase3_teacher launch repeated the
  `export default` mistake — engine strips ONLY `export const meta`; scripts must
  be meta + PLAIN BODY (top-level await/return fine). Rule: every workflow script
  passes the emulate-parse check (strip meta, node --check the wrapped body) BEFORE
  launch; probe output = phase3-teacher-muyx5i4x-pw34fs now running.
- LESSON (2026-10-08, cost real quota): workflow `agent()` calls do NOT inherit
  `task.agentModelOverrides` — without an explicit `{ model: ... }` option they run on
  `modelRoles.default` (zai/glm-5.3). A phase-3A run burned the zai 5-hour window to
  error 1308 and paused. RULE: every workflow agent() passes an explicit model
  (`opencode-go/deepseek-v4.1-flash` for heavy work, `opencode-go/mimo-v2.6-flash` for
  mechanical). Same rule applies to any future workflow script.
- DIRECTIVE (operator, 2026-10-08 night, overnight autonomy): current milestone is the SINGLE-G1
  CONTINUOUS SOLO DRILL (docs/SOLO_DRILL.md); no second active robot. Work autonomously overnight.
  ONE codex model (gpt-6.1-sol) may be requested for debugging/architecture help ONLY — no more
  than that, and only for a concrete blocker with a clear question. Do not claim success without
  video + numbers; preserve best verified checkpoint/footage and name precise blockers.
- DIRECTIVE (operator, 2026-10-08): workflows ONLY when structure needs them
  (multi-stage pipelines/DAGs, Phases 5+); independent one-shot agents via `task` batches.
  This SUPERSEDES the earlier line "Workflows are the preferred fan-out again".


## Experiment ledger

Format per entry: id | question | prediction | result | conclusion.
Entries appended as experiments run (Phase 2 onward).

### E2 (2026-10-08) — external engineering audit: orchestrator verdict, claim by claim
- TRUE (verified in tree): (a) references track joints (≤0.06 rad) while physical execution
  fails — already banked; (b) wins≠competence — E1 is the worked example; (c) the progress
  reward is CLOCK-based (`obs.reference_phase` is a pure function of exchange_time), so it
  carries no physical-progress signal (bias-free shaping, weak signal) —
  src/rl/reward.py:16,159-161, src/rl/obs.py:116-120; (d) curriculum can advance on draws:
  metric=(wins+0.5*draws)/n with threshold 0.5 (src/rl/curriculum.py:18-22,113,121,129,140)
  → all-draw window satisfies it. FIX IN FLIGHT (P0Fixes);
  (e) the 0.10 s simultaneous-fall ambiguity window is unreachable across control steps
  (env ends the exchange on the first trigger; env.py:609-614) — only same-step ambiguity
  works. FIX IN FLIGHT (P0Fixes); (f) actor obs lacks opponent limb detail (shot-onset
  cues) — to be measured in the observation ablation, not assumed.
- FALSE in the current tree (do not chase): teacher "undefined G" (G=9.81 at
  controller.py:65); "one-row sqrt regularizer" (stabilizers.py solve_offsets uses
  sqrt(reg)*eye(29) with 29 zero targets — correct diagonal form); "_support_xy uses
  data.time" (support_center uses ctx.ref_support[frame]; no data.time in controller.py).
  TeacherRetry asked to confirm + add a default-tick finite-output regression test.
- ACCEPTED as scope statements, not bugs: the scorer is a GEOMETRY diagnostic (no force or
  grip semantics) — it must not be treated as proof of execution; contact/grip physics
  (no articulated fingers) and foot-contact-model adequacy must be validated before/
  alongside M1-M2; opponents must not be proxies that collapse on their own.
- ADOPTED into the motor curriculum: the six diagnostic video types (balance A/B with
  overlays; ghost-reference vs actual; reward-vs-behaviour; contact/clinch force transfer;
  locomotion/coordination tracking; genuine exchange vs stationary/moving/competent
  opponent) and their "paired baseline vs candidate on identical seeds" discipline.
- TRUE (third confirmed defect, added after re-check): src/teacher/phases.py `_smooth()` is a
  LAGGING centered average ((n-1)/2 samples ≈ 0.07 s at 50 Hz for n=8) despite its
  "zero-phase" docstring → phase-transition timing lag where phases gate the stabilizer.
  Fix requested from TeacherRetry (true zero-phase or corrected docstring + test). Related
  design concern: phases.py infers PRONE from pelvis height/velocity alone, switching the
  stabilizer off for any low pelvis regardless of cause — mitigation/justification requested.

- QUALITY IS THE ACCEPTANCE AXIS (operator, 2026-10-08): "quality is the main outcome", and the
  timeline is relaxed (">12 h available"). Consequence: the fallback ladder is insurance, not the
  plan; the primary path is the highest-quality motion achievable (teacher/learned execution), and
  breadth never outranks quality. docs/QUALITY_RUBRIC.md is the ship gate: per element 0-3
  (stance / footwork / level change / penetration step / recovery / continuity / plausibility /
  visual match), no element below 2, frame + numeric evidence required for any score >= 2, 0.25x
  slow-motion review of the four critical moments, failures kept, nothing labelled "the drill" if it
  needed a reset or a fall. All motion-producing agents (teacher, drill, mocap reference) are bound
  to the rubric.

### E23b (2026-10-08) — CORRECTION to E23's mechanism (the 'constant reward' claim was wrong)
- DIAGNOSIS — CORRECTED (the first version of this entry, taken from the probe's summary, was
  WRONG): the balance reward is NOT constant. `t_alive = clamp01(torso_up_z) · clamp01(pelvis_z/ref)`
  is a DENSE uprightness × stance-height signal (≈0 collapsed, ≈1 upright; src/solo/reward.py:168-171),
  and reward.py asserts at construction that `alive_weight > sum(per-step penalty weights)`
  (reward.py:329-333). The correct mechanism for the collapse is therefore *termination-risk +
  exploration failure*, not a flat reward: the −100 terminal penalty makes any attempt to stand
  risky while the policy is still high-entropy and its value estimate is poor, so a low-variance
  collapsed pose (low fall rate, uprightness ≈0) is the better-returned behaviour in practice. Note
  the invariant is asserted on WEIGHTS, not on realized values — which is why it did not prevent a
  degenerate solution. Corrected lever order: (1) entropy_coef (v2, done, helped), (2) alive_weight
  (v3, launched), (3) TERMINATION-PENALTY MAGNITUDE if v3 still collapses — reducing the −100 is the
  named next lever rather than "adding reward signal", which is already dense.

### E28 (2026-10-08) — CONTRADICTORY ROADMAPS + A FALSE "SHIPPED" CLAIM: single authority re-established
- TRIGGER: operator pointed out the todo list contained conflicting/contradictory items. It did, and
  worse than a stale list: three overlapping acceptance ladders and one claim that was factually false.
- (1) SCOPE CONTRADICTION: the todo's active pointer sat in phase 3 "Teacher+BC", i.e. the two-robot
  pipeline (PD teacher -> BC, resistance stages A-E, self-play league, opponent pool, 3-minute
  match). docs/SOLO_DRILL.md puts exactly that out of scope for the current mandate ("Out of scope:
  resistance, opponent interaction, self-play, body locks, snapdowns, tactics"). The pointer was
  therefore aimed at superseded work while the real work (solo T1 balance) was one phase away.
- (2) DUPLICATE FRAMINGS: M1-M5 (motor curriculum), S1-S10 (solo drill) and L0-L4 (drill ladder) all
  described the same learned-motor goal with SEPARATE acceptance criteria — three definitions of
  "done". SOLO_DRILL.md itself states the milestone IS the motor curriculum with a demo attached
  (T1..T7 <-> M1..M5), so M is folded into S, not kept parallel.
- (3) FALSE CLAIM, now corrected: notes.md recorded "videos/solo_drill/final_continuous_drill.mp4
  exists but was still being written at inspection". A repo-wide glob (`**/*continuous_drill*`)
  returns NOTHING. The file was an in-flight render that never completed, and it is the mandate's
  ACCEPTANCE ARTIFACT — so the milestone is unmet, not nearly met. What exists is SCRIPTED interim
  footage: final_L1_90s.mp4 (clean 90 s hold), final_L2_motion.mp4 + final_L2_entry_walk.mp4 (L2
  isolated step clean, slip 0.000 m, 3.8 cm clearance — but the entry walk FALLS at 5.8 s, per the
  DRILL ledger entry), plus a slowmo level-change clip and labelled failure clips.
- (4) The todo list is consolidated to ONE authority with four phases: A = the solo-drill mandate
  S1-S10 (S2/T1 balance in progress); B = interim scripted evidence, marked fallback-only with the
  ladder rungs contingent on the learned path stalling; C = ledger/status integrity (this entry);
  D = DEFERRED two-robot mission, retained as the long-term goal in MISSION.md/goal.md rather than as
  open work. Everything removed from the TODO list is still recorded in the docs, so nothing is lost.
- GENERALISED RULE: a ledger entry that says an artifact exists must be re-verified on disk before it
  is used as a premise (the same discipline that caught the residual-map bug and the stale-ref
  failures). "In flight" renders are the easiest thing in this repo to mistake for a deliverable.

### E27 (2026-10-08) — BUG: the residual action path double-mapped; v4's premise was void
- CONFIRMED BY READING CODE (not inferred): `env.step(action)` calls `resolve_action` (env.py:367-376,
  358-364) which in residual mode returns `clip(base + residual_scale*tanh(action))`, while the
  trainer pre-mapped the action with `ctrl_from_unit(unit) = mid + half*unit` (train.py:164,
  env.py:338-347). Net effect: `ctrl = base + 0.5*tanh(mid + half*unit)`. At `unit = 0` that is
  `base + 0.5*tanh(mid)` — for a joint with ctrlrange midpoint 1.0 rad that is a +0.38 rad offset, so
  the commanded pose was NOT the `a_stand` keyframe. v4 therefore never initialised as the stand
  controller, and any gate read from it would have been misattributed to the residual idea.
- ACTION: v4 stopped (14m53s in, before it could produce a misleading result); SoloEnv dispatched to
  make the mapping mode-aware in one place (absolute keeps `ctrl_from_unit`; residual passes the RAW
  unit action because `step` applies `base + scale*tanh(·)` itself), with four required tests —
  crucially `unit = 0 ⇒ data.ctrl == env._base_action` asserted THROUGH the real `step` path, the
  absolute-mode behaviour preserved, a round-trip `ctrl == base + scale*tanh(unit)`, and the same
  check applied to the evaluation/monitor path (a shared bug there would invalidate every gate read).
- GENERALISED LESSON (added to the mid-run probe skill): when a run's premise is an INITIALISATION
  property (e.g. "at init the policy is the base controller"), assert it mechanically at startup and
  print the number — here a one-line max-abs-difference between the first step's ctrl and
  `_base_action` would have caught it before 15 minutes of training and before any interpretation.
  Shaping/optimizer levers should not be blamed while an unverified interface premise is in play.

### E26 (2026-10-08) — T1 v3 (alive-weight 10): return rose, behaviour did not — explore-vs-shape diagnosed
- RESULT: v3 finished 2.0M steps (exit 0, checkpoint t1_balance_v3.pt). Gate (48-push battery vs
  baselines): fall_rate 0.104 / held-out 0.083, mean_upright **0.0475**, recovery 0.0, maxJ_held 0.0,
  t_stab none, com_offset_max 0.769 → NOT CERTIFIED (2/7).
- CONTRAST across the three runs (fall | held-out fall | upright | recovery):
  v1 0.125 | 0.125 | 0.006 | 0.0 → v2 0.042 | 0.000 | 0.0415 | 0.0 → v3 0.104 | 0.083 | 0.0475 | 0.0,
  against StandHold 0.292 | 0.958 | 0.849 | 0.25. The alive-weight change RAISED THE RETURN
  (mean_return_50 +90 vs v2's -85) without raising uprightness: that gain was the weighting, not the
  behaviour. Three shaping changes (entropy, push curriculum, alive weight) have now failed to make a
  random-init policy stand.
- DIAGNOSIS (the pattern, not a single number): this is an EXPLORATION failure, not a shaping failure.
  From a random policy the robot falls almost immediately, so nearly every trajectory ends in the -100
  terminal, and the best-behaved option available is a crouch that stays just clear of the fall
  detector's pelvis/tilt thresholds — a detector-threshold exploit. Reward shaping cannot fix a state
  distribution the policy never sees.
- NAMED SINGLE CHANGE (dispatched as v4): `--action-mode residual`. Verified in code that
  `env.set_base_action()` defaults the residual base to the `a_stand` keyframe ctrl (env.py:353-364,
  scene.py:158-165) — i.e. at initialization the policy IS the verified-stable stand controller
  (0.1 mm drift over 20 s), so it starts inside the high-alive region and learns residuals from there
  instead of having to discover standing. This is also exactly MISSION's "residual motor control"
  option. v4 keeps entropy_coef 0.001, alive-weight 10, the push curriculum and `--save-every 100000`
  so mid-run reads have real artefacts.
- FALLBACK LEVERS if v4 also collapses, in order: (a) the -100 termination MAGNITUDE (it may dominate
  the value estimate for a policy that cannot yet stand), (b) a reset curriculum (start from
  recoverable perturbations instead of only the stand keyframe + noise), (c) longer training
  (published CPU-scale balance work used 2 days, not 1.6 h), (d) imitation warm start (BC on a
  scripted stabiliser) which is MISSION's own prescription if residual alone is not enough.

### E25 (2026-10-08) — the 7 test failures: STALE DERIVED ARTIFACT (not a code bug); suite green again
- RESOLVED (TestRegress; verified by the orchestrator: `pytest tests/ -q` -> 175 passed, 0 failed).
- SCORER (6 failures): root cause was a STALE DERIVED FILE, not the scorer or the test premise.
  `data/scorer_calibration.json` was generated at 03:30; `data/refs/STAND_UP.npz` was REPLACED at 06:24
  (the grounded rebuild, commit 3a7ce45). The on-disk STAND_UP parameters reproduce a refit from the
  OLD airborne reference 22/22, and a refit from the current reference changes exactly those 22
  parameters. The rebuilt reference is NOT degenerate (per-phase knee_z_s 0.040 -> 0.343). FIX:
  regenerated the calibration via `scripts/calibrate_scorer.py --write` (the documented remedy) — no
  source change, no threshold relaxed. Evidence: STAND_UP own-trace mean 0.569 -> 0.995; root-tilt
  monotonicity restored (0.569/0.572/0.496 non-monotone -> 0.995/0.484 monotone); phase-shift
  non-monotone -> 0.995/0.848; self-vs-cross 0.569/0.602 -> 0.995/<=0.795.
- BACKDET (1 failure): the test hardcoded `a_left_wrist_yaw_link` at STAND_UP frame 6 (written 03:53);
  the grounded rebuild plants `a_right_wrist_yaw_link` there. Resolution was correct on both sides
  (limb_contact True for both robots). FIX: the frame-6 expectation now requires "a resolved executor
  wrist (`a_*_wrist_yaw_link`) + `b_left_knee_link`", with a provenance comment — a deliberate
  contract update, not a weakening.
- PROCESS LESSON (new rule): any GENERATED calibration/config that depends on a data file must record
  that file's hash and be regenerated when the input changes; otherwise a legitimate upstream data
  change silently invalidates downstream thresholds. (Earlier in the night the feasibility audit had
  already switched to hash-anchoring for the same reason; this makes it a standing rule.)
- KNOWN RESIDUAL (not a failure, documented in reports/2026-10-08/test_regressions.md): the scorer
  calibration now flags 3 STAND_UP band-provenance deviations (<=0.114 s) because the rebuilt reference
  meta dropped `time_stretch_requested` while `build_technique_targets` still models the old
  t952/t1110/t1125 schedule — re-deriving needs src/retarget + the reference meta (outside that agent's
  ownership). Carried as a known gap for the final report.
- V3 healthy so far (context): 157k steps at 322 steps/s, entropy flat ~12.13 (was climbing to 26 in
  v1), mean_return_50 rising to +90 (v2 never exceeded -85) — the alive-weight change is doing what it
  was intended to do; the gate read at ~1.1M is what will decide.

### E24 (2026-10-08) — T1 v2 and v3: one change at a time, measured
- v2 (entropy_coef 0.01 -> 0.001, plus a ramped training push curriculum ≤12 N*s so the gate's
  held-out 16/20/25 N*s stay unseen; gate battery untouched): the entropy fix WORKED — entropy rose
  to a peak 12.37 (~280k steps) then fell (12.22 at 739k, 12.31 at 1.10M) instead of climbing to 26;
  mean_return_50 rose from −102 to −85. GATE at 1,101,824 steps (48-push battery, vs StandHold):
  fall_rate 0.042 (0.292) PASS, fall_rate_heldout 0.000 (0.958) PASS, mean_upright 0.0415 (0.849)
  FAIL, max_recoverable_impulse_heldout 0.0 (16.0) FAIL, time_to_stability none (0.133 s) FAIL,
  com_offset_max 0.796 (0.135) FAIL, recovery_success 0.0 (0.25) FAIL → 2/7, not certified. It still
  dodges the −100 termination by collapsing (low fall rate, uprightness ≈0).
- DECISION: stopped v2 exactly at the pre-declared stop condition (mean_upright ≈ 0 beyond 800k) —
  one change, one measurement, no stacking. Caveat recorded by the agent: with a single-checkpoint
  path the 400k/800k snapshots were overwritten, so those reads come from the log, not artifacts;
  v3 uses `--save-every 100000` to avoid that.
- v3 LAUNCHED (first iteration 1, mean_return_50 +56.5 vs −102 at v2's start — the alive weight
  flips the sign immediately): `--entropy-coef 0.001 --alive-weight 10`, service `solo_t1_v3`,
  2M steps, `--save-every 100000`, `--lock off`. The alive-weight change was hand-checked first
  (upright +10.0/step vs collapsed −0.05/step with the termination still −100, invariant holds).
- ALSO FOUND AND DISPATCHED: `pytest tests/ -q` is 7 failed / 168 passed — 6 scorer monotonicity
  failures on the REBUILT STAND_UP reference and 1 backdet limb-set mismatch, both introduced by other
  agents' changes. Dispatched to a dedicated agent with instructions to resolve on evidence (fix the
  scorer if the property is genuinely violated; adjust the test premise only if the rebuilt reference
  legitimately lacks the structure the property needs) — never by weakening thresholds.

### E23 (2026-10-08) — T1 mid-run read: policy DEGENERATE (below baselines) — run stopped, two causes named
- VERDICT (T1Midrun, independent, reports/2026-10-08/t1_midrun_read.md): the 1.4M-step checkpoint
  fails ALL SEVEN T1 criteria and sits BELOW the baselines. Protocol: full 48-push battery, 48
  episodes (seeds 0-47), 18515 steps, 126 s wall under the sim lock; the trainer was untouched;
  determinism = tanh(actor.mean(obs)) with net.eval().
- GATE TABLE (policy | zero_action | StandHold | random_init): fall_rate 0.125 | 0.354 | 0.292 | 0.812;
  fall_rate_heldout 0.125 | 0.875 | 0.958 | 0.875; mean_upright 0.006 | 0.192 | 0.849 | 0.176;
  max_rec_impulse_heldout 0.0 | 0.0 | 16.0 | 0.0; time_to_stability none | none | 0.133 | none;
  com_offset_max 0.822 | 0.562 | 0.135 | 0.677; recovery_success 0.0 | 0.0 | 0.25 | 0.0;
  steps_after_push 22.9 | 0.38 | 1.31 | 2.88. The learned policy is better than random-init on falls
  but far worse than a stiff scripted stand on uprightness — it dodges the -100 termination by
  collapsing into a non-terminating limb-supported pose rather than standing.
- DIAGNOSIS (mechanistic, with numbers): entropy_coef=0.01 across 29 joints leaves a non-vanishing
  outward log-std gradient (0.29) with nothing opposing it, because the balance reward is nearly
  constant (+1/step alive; per-step penalties ~0.45; NO push/recovery shaping), so advantages are
  uninformative; advantage normalisation rescales the noise to unit variance; entropy climbs
  monotonically 12 -> 26.4 (still +1.03 per 100 iterations) while mean_return_50 stays flat at ~-77
  for 330+ iterations. SECOND, STRUCTURAL: TASKS['balance'] has push=None, so the gate's held-out push
  criteria (>12 N*s) are unattainable by construction — even StandHold tops out at recovery 0.25.
- DECISION: stopped the run (a degenerate policy on its own training distribution; more steps would
  produce another bad checkpoint — measured, not assumed). Dispatched to SoloEnv: (1) expose
  `--entropy-coef` on the CLI (mirroring --gamma) so tuning needs no code edit; (2) add a ramped
  push curriculum to the balance task staying <= TRAIN_MAX_IMPULSE (12 N*s) so the held-out magnitudes
  stay held out, with the gate battery untouched; (3) run v2 as a named background service with
  entropy_coef 0.001 and monitor the same gate at ~400k/800k/1.2M, stopping if entropy is still rising
  or mean_upright is still ~0 at 800k, and naming the NEXT SINGLE change rather than stacking changes.
- METHOD WIN: the mid-run read (a free-standing probe agent) converted "wait an hour and hope" into a
  measured decision plus a mechanism. Worth reusing for every long run.

### E22 (2026-10-08) — PROCESS FAILURE (orchestrator): a mid-write clip was committed; restored
- WHAT HAPPENED: the corrected-HUD re-render ran as a background job INSIDE the drill agent's process;
  when that agent finished, the job died mid-write, leaving videos/solo_drill/final_L2_motion.mp4 at
  525 frames / 17.5 s / 2.33 MB — overwriting the verified 70 s / 2100-frame clip. The orchestrator
  then committed that partial in f41cf40, violating its OWN rule (EVIDENCE_PROTOCOL: verify duration
  and frame count before committing any mp4). Detected by re-checking the artifact instead of trusting
  the "render in flight" note.
- FIX: restored the verified clip from commit ef8c1b8's predecessor (ef8c1b2) — 2100 frames,
  70.000 s, 960x720; the slow-motion clip is the corrected 409-frame / 13.633 s version at 480x360
  (labelled diagnostic). No data lost.
- PROCESS RULES NOW ENFORCED (for all agents and the orchestrator):
  1) RENDERS WRITE TO A TEMP PATH, get ffprobe-verified (duration + frames + codec/pix_fmt/resolution
     + a non-black/non-static sample), and are then ATOMICALLY RENAMED over the final name. A partial
     file must never occupy a final artifact name.
  2) RENDERS RUN AS SUPERVISED JOBS (named bash services) that outlive an agent's turn — never as
     in-process background tasks that die with the agent.
  3) The orchestrator re-checks duration/frames of any mp4 immediately before committing it, and
     treats "render finished" as an unverified claim otherwise.
- LESSON: the same class of error as the earlier .omo sweep and the oversized yield — a checkable
  artifact was committed without running the check. The fix is mechanical, not attitudinal: verify at
  the moment of commit.

### E21 (2026-10-08) — L2 discrepancies resolved: safety flash benign/inert, B1 holds, evidence fixed
- EMERGENCY HUD FLASH — BENIGN AND INERT, and the HUD was misleading: safety_alpha > 0.02 for only
  18 ticks (0.36 s) in exactly 2 episodes (0.16 s at t=15.76, 0.20 s at t=45.74), peaking 0.09/0.11,
  each starting the instant the swing foot leaves the mat (swing load 0 N, clearance 9-16 mm). The
  pose blend is STRUCTURALLY DISABLED while a foot is airborne, so it could not act: 0 emergency
  plants, 0 step aborts, all 5 steps completed on the normal path. Fix: the HUD now prints the numeric
  alpha and flashes the red warning only when the blend can actually act.
- SUPPORT-FOOT CREEP vs SLIP CLAIM — the physics measurement wins: loaded-foot sole-centre
  displacement between the swing foot's lift and landing is 3.1 / 3.5 / 4.4 / 5.6 / 2.6 mm per step
  (mean load 263-285 N), so rubric B1 (20 mm bar) HOLDS and the bundle's max_load_drift 6.5 mm is
  consistent. Reconciliation: the visual checker's 8 px mask shift / IoU 0.81 -> 0.34 tracked the
  SWING foot (which legitimately moves 55-90 mm) or a rolling sole mask, and its "+34/+38 px over the
  clip" was the shuffle's net travel (the stance walks 8-11 cm per block), not slip.
  METHODOLOGY LESSON: a mask-overlap proxy cannot separate slip from a legitimate reposition — the
  discriminator is the per-foot LOAD in the physics trace; visual slip claims must be validated
  against the trace before being reported.
- EVIDENCE FIXES: HUD now carries phase (advancing via the scheduler's element_done events), a
  "step N/M" counter, a "(swing)" marker and the stance width; the "0.25x" slow-motion was a writer
  fps/speed bug and is re-rendered as a true quarter-speed clip (409 frames, 13.633 s, 480x360 —
  labelled diagnostic) centred on one step cycle; the clip's stance value (0.28, not 0.30), step count
  (5, not 6) and margin readings (HUD resamples the trace at frame times; the trace minimum is
  -0.0257 m at 45.76 s) are now consistent, with the trace as the single source of truth.
- STATE: the 960x720 re-render with the corrected HUD is in flight; the L2 motion artifact is now
  internally consistent and independently verified as real motion (5 steps, 12.15 s mean interval,
  0 falls/resets) in a 0.28 m stance — a documented deviation from the operator's 0.495 m reference,
  justified by the measured steppability limit. Cadence remains 12.1 s/step with the measured reason
  (the full recentre is load-bearing).

### E20 (2026-10-08) — L2 motion clip VERIFIED as real motion (5 steps) with 7 discrepancies to fix
- INDEPENDENT VERDICT (L2VisualCheck, reports/2026-10-08/l2_clip_visual_check.md): **MATCHES** the
  claim "a 70 s continuous drill with repeated visible steps in a wrestling stance, 0 falls".
  Method: own ffmpeg extraction (0.5-5 s tiles + 0.05-0.1 s sampling around each step), shoe-mask
  tracking/IoU, HUD text reads, frame-diff statistics; the trace was consulted only after the pixel
  evidence.
- POSITIVE, MEASURED: NOT static (unlike L1). 5 genuine steps at 15.40 / 28.80 / 45.40 / 50.22 /
  63.98 s, mean interval 12.15 s (claimed 12.1); each re-plants the shoe 16-23 px (8-11 cm); feet
  drift +34/+38 px over the clip; pelvis z 0.689-0.762 m; skill advances STANCE -> SHUFFLE_F ->
  SHUFFLE_B; lift is real but small (3.2-3.6 cm clearance, HUD concordant) with single support
  visible (foot load 0/364 N); no pops (max frame diff 4.0); floor + shadows visible; 0 falls/resets.
  Stance visibly ~60% of the reference width (the declared deviation is apparent on screen).
- DISCREPANCIES FOUND (all being fixed; first two are technical, the rest evidence hygiene):
  (1) the HUD flashes red "safety blend / emergency response active" at 15.77-15.83 s and
  45.77-45.83 s — exactly at two steps — which the report never mentions: must be explained
  (what triggered it, whether the step was completed by the emergency path, whether it masks an
  incipient fall); (2) the loaded support foot CREEPS 22-35 mm (shoe mask shifting rigidly 8 px,
  IoU 0.81 -> 0.34) while the bundle claims slip 0.0 / max_load_drift 6.5 mm — must be reconciled
  with per-step load-phase measurements and B1 rescored honestly; (3) the HUD phase field read '-'
  in all samples and there is no step N/M counter; (4) the "0.25x" slow-motion actually plays at
  1.0x and only ~6% of it contains the step; (5) the clip HUD says stance 0.30 vs the report's 0.28;
  (6) rubric B1 says 6 steps vs the 5 in the clip; (7) the HUD margin (-0.008/-0.022 at 45.65/45.72 s)
  disagrees with the report's -0.0257/-0.023.
- DISPATCHED (DrillMotion): investigate (1) and (2) properly; add the phase + step counter to the HUD;
  re-render a genuine 0.25x slow-mo tightly centred on one step; make stance/step-count/margin
  consistent from the trace as the single source of truth; refresh the bundle and rubric. No new
  physics experiments for this pass.

### E19 (2026-10-08) — quasi-static controller hits its measured ceiling; pivoting the remaining time to the LEARNED path
- NEGATIVE RESULT, MEASURED (DrillMotion): the settle bottleneck cannot be optimised away. The balance
  law holds a per-configuration steady-state CoM offset (0.036-0.040 m on 0.24-0.28 m stances;
  0.030 m at the delivered 0.28 m/lean-0.14 setting) and the recentre lead is 4 cm, so the lead
  cancels exactly and the body stops — that IS the 9 s timeout. Physical exit criteria
  (|v_com| <= v AND margin >= 0.02 after a dwell) raise cadence from 12.1 to 4.8-6.0 s/step but ALL 11
  runs across 4 widths and 2 dwell settings then FALL after 2-5 steps; stronger leads (0.10 m) fall
  at 2 steps; settle-off falls. VERDICT: the full recentre is load-bearing for the quasi-static
  design — 12.1 s/step is its ceiling, and a continuous shuffle is not viable above ~0.02 m/s CoM
  velocity. Target cadence 1.5-2.5 s/step (and 0.49 m reference width) requires a DYNAMIC
  capture-point gait, i.e. a learned low-level policy.
- DELIVERED L2 MOTION RUN (data/drill/M_D28c_feasible_L3_seed0.npz, clip rendering):
  69.98 s, 0 falls, 0 resets, 5/5 steps, 0 refusals/aborts, cadence 12.1 s/step, CoM span 0.374 m
  (0.306 m in the second half), com_travel 2.44 m, pelvis-z range 0.074 m, slip_max 0.019 m,
  phase_advance_count 2, margin_min -0.0257 m. NEGATIVE MARGIN EXPLAINED: 24 of 3500 ticks (0.69%)
  in 5 episodes of 0.06-0.14 s, depths -0.009 to -0.0257 m, each recovering to +0.037..+0.075 m
  within ~0.5 s; positive margin 99.3% of the run; none during a hold. Rubric: A1=1 (0.28 vs 0.49 m),
  A3/A4/A6/A7=2, A5/A8=3; B1/B2/B3/B6=2, B4=1 (cadence); F1/F3/F4=3, F2/F5=2; G=3/3/2/3;
  H1=2, H2=3, H3=2.
- PIVOT (plan for the remaining hours): (1) L2 motion clip finishes rendering and gets an independent
  visual check + doc corrections; (2) SoloEnv adds a `--lock=off` trainer flag (so a multi-hour run
  does not starve the box) and extends the T1 push battery beyond the ~13 N*s non-stepping ceiling
  with time-to-stability, CoM-margin and step-count criteria (its own finding was that the current
  battery cannot distinguish a stiff stand); (3) launch T1 balance training (learned balance policy,
  checkpointed/resumable); (4) evaluate against the extended gate + the same baselines; (5) then T2
  (velocity/stepping) for a dynamic shuffle, and only then re-attempt a drill with a learned
  low-level layer. The quasi-static L2 motion clip remains the honest interim artifact.

### E18 (2026-10-08) — width-vs-steppability resolved by measurement; cadence bottleneck = the 9 s settle
- WIDTH TABLE (DrillMotion, data/drill/motion_widths.json): required CoM travel per lift ≈ w/2 - 0.01 m
  plus the fore-aft distance to the support footprint centre; measured authority 0.14 m (shipped cap)
  to 0.20 m (delivered travel). Stepping is possible up to ~0.30 m width: w=0.28 -> 3 steps, 0 falls,
  17.5 s/step; w=0.30 -> 6 steps, 0 falls, 13.7 s/step; w=0.21 -> 4 steps, 1 fall, 13.3 s/step;
  w=0.35/0.42/0.495 -> past authority (falls or refusals; the operator's 0.495 m width is STRUCTURALLY
  un-steppable: 59 refusals, refused rather than toppled). DELIVERED CHOICE: 0.28 m (57% of the
  reference width) as a documented deviation justified by steppability.
- MECHANISM RESULTS: support-foot EDGE ROLL REJECTED (falls with 0 completed steps at 0.21-0.30 m;
  rolling a loaded foot halves the CoP braking range); TORSO LEAN KEPT (0.065 m CoM per rad of waist
  roll; 4.70 -> 3.98 s/step and margin +0.0031 -> +0.0053 m at the 0.21 m base); support-foot
  step-out reshuffle measured but NOT used (destabilises before it helps); YAW PIVOT REJECTED (a
  0.5 rad loaded pivot is a hidden fall cause: pivot_max=0 converts 1-2-step falls into 90 s clean
  runs); NEW KEPT MECHANISM: CoM-VELOCITY lift gate (|v_com| <= 0.04-0.05 m/s) — lifts now start
  from a stationary CoM (0.0006-0.036 m/s vs 0.10-0.16 m/s before).
- CADENCE: 12.1 s/step delivered (4.0 s/step at a 0.21 m base); limit decomposed as
  crossing/0.03 m/s + 0.5 s swing + **9 s settle**, where the settle times out because the balance
  law's ~4 cm steady-state CoM offset equals the recentre lead. Target 1.5-2.5 s/step NOT reached.
- DELIVERED RUN (data/drill/M_D28c_feasible_L3_seed0.json): 69.98 s, 0 falls, 0 resets, 5/5 steps
  with no refusals or aborts, CoM span 0.374 m (0.306 m in the second half), margin_min -0.0257 m
  (transient single-support transfer — must be stated as such, not hidden). A same-spec 0.30 m run
  fell at 22.9 s after 2 steps and is kept as the variance/failure sample.
- CLIP: videos/solo_drill/final_L2_motion.mp4 was still rendering at hand-off (moov atom absent at
  inspection); L2_motion_slowmo_step_quarter.mp4 present. Defects fixed meanwhile: L1 static-clip
  claims corrected in drill.md + VISUALS; phase-advance acceptance added
  (motion.phase_advance_count + motion_span + 2 tests, 21 passing); shadows on + plane extents fixed
  (mat texture not applied by this MuJoCo version — stated, not claimed as done); fingers claim
  corrected in drill.md §6.2; 03_side_by_side_reference.png rebuilt at matched scale with per-panel
  timestamps.
- NEXT (dispatched): P1 fix the settle (characterise the balance law's steady-state offset; use the
  measured steady-state as the recentre set-point or add a bias/integral term; shorten the settle
  criterion to |v_com| <= 0.01 m/s AND support load >= 90% AND margin >= 0.02 m) targeting <= 4 s/step
  at 0.28 m; P2 then attempt a continuous capture-point shuffle (CoM keeps 0.03-0.08 m/s and the next
  step catches it) and report the measured fall threshold; P3 re-deliver the motion clip with the
  improved cadence; P4 explain the transient negative margin; P5 track repair only if budget remains.

### E17 (2026-10-08) — L1 clip visual verdict: CLEAN but NEARLY STATIC; the "18 cycles" claim is NOT in the artifact
- INDEPENDENT FRAME-LEVEL VERDICT (L1VisualCheck, reports/2026-10-08/l1_clip_visual_check.md):
  **PARTIALLY MATCHES**. CLEAN: 0 black/flat/pop frames over 2700, no cuts, feet pixel-stationary
  (±3 px = no slide), whole robot always in frame. STANCE-MOTION: only PARTIAL.
- CRITICAL: the clip contains exactly ONE crouch/rise in t=0-8 s; for the following 82 s the head
  top moves <= 1 px (<= 0.8 cm) and the HUD's skill/phase labels (RECOVER, l1_crouch_2) never
  advance. The claimed "repeated level changes / 18 cycles" (drill.md §0, VISUALS row 10b) is NOT in
  this clip — the artifact does not support the claim. There is a ±5% body-weight foot-load
  oscillation (142-179 N) and <= 4 px sway; that is all.
- STANCE vs the operator's reference: width MATCHES (0.495-0.519 vs 0.491 m); depth APPROXIMATES
  (0.24 vs 0.062 m); crouch DIFFERS (4.3 cm drop; 0.644 m was reachable but "kept shallow for
  margin"); torso pitch DIFFERS (14-15.5 deg vs the operator's ~35-45 deg) though the controller has
  torso-upright rows; hands DIFFERS (drill.md self-contradicts: hip height in §0/§3/VISUALS vs chest
  height in §6.2); head APPROXIMATES. The render shows articulated fingers, contradicting §6.2's
  "no articulated fingers" claim.
- EVIDENCE DEFECTS to fix: (a) the rubric-H side-by-side (03_side_by_side_reference.png) is
  MISLEADING — its panels are byte-identical to frames whose HUDs contradict the clip (phase/time
  labels inverted), different viewpoints/scales, illegible labels, captions cut at the right edge;
  (b) the HUD status line is clipped at x=958; (c) NO floor/mat is rendered, so foot-floor contact
  is unjudgeable from the frames; (d) cannot be determined from one camera: true 3-D width/depth/CoM,
  finger articulation, <=2 cm interpenetration, whether the metrics JSON/rubric scores describe this
  mp4.
- CONSEQUENCES (dispatched): the deliverable clip must contain GENUINELY VISIBLE motion (the whole
  point of the L2 motion work); the frozen skill/phase labels must be investigated (scheduler stuck
  or labels not updated?) since that is what made a static clip look like 18 cycles; the side-by-side
  must be rebuilt from the correct frames at the same scale/viewpoint with legible captions; a floor
  must be visible; HUD clipping fixed; and every claim in drill.md/VISUALS must be corrected to match
  the artifact. Doc-vs-artifact mismatch is exactly the failure mode the Codex review warned about
  ("a scheduler label is never the acceptance observation").

### E16 (2026-10-08) — CoM-gated lift works, but the WIDE stance cannot be stepped in (authority 0.14 m < required 0.25-0.28 m)
- FIXED AND MEASURED (DrillStep2, reports/2026-10-08/drill_l2.md): the lift gate now fires on the
  measured CoM margin inside the SUPPORT foot's own footprint hull (>= +0.02 m) with a 2-tick dwell
  and 0.06 s lock, load recorded but never gated; regression test
  `test_lift_gate_is_margin_not_load` (fails on a load-only gate). At fire: margin_support
  +0.024..+0.027 m, com_travel 0.086-0.098 m, ankle_roll_support -0.156..-0.183 (limit ±0.2618),
  swing load 35-62 N, pivot 0.32-0.50 rad. Supporting helpers added: support-foot toe-out yaw pivot
  about the measured footprint centre, swing-foot roll compliance + roll-authority mask,
  clearance-triggered pelvis drop (<=0.045 m), min-jerk shift profile (0.03 m/s, zero end velocity),
  lead-based settle after landing, and an explicit geometry REFUSAL (reach cap 0.14 m) that reports
  required_com_travel_m.
- STAGE RESULTS: L2_ENTRY_SPEC 0 falls / 1 step / 9 refusals (required 0.189-0.209 m, margin_min
  +0.0137, slip 0.0058); L2_ENTRY_BASE 1 fall at 9.6 s; L2_CYCLE 1 fall at 22.56 s after 3 steps;
  L2_SHUFFLE **0 falls, 34 s, 4 steps**, margin_min -0.0017, slip 0.0083; L2_STANCE_REFUSED
  (the 0.495 m drill stance) **0 steps, 7 refusals**, required 0.254-0.276 m vs the 0.14 m authority.
- THE CONFLICT (central finding): the stance that matches the operator's reference (0.495 m wide) and
  holds 90 s at +0.0295 m margin CANNOT be shuffled in — the required lateral CoM travel (0.25-0.28 m)
  is roughly double the controller's authority (0.14 m), so stepping is only possible in a ~0.21 m
  base (4 steps/34 s, clean) at a 5-7 s cadence (lateral ceiling ~0.03 m/s). Real wrestling resolves
  this with edge loading/roll, hip abduction + torso counter-rotation, and reshuffle step-outs;
  those mechanisms are now under test (dispatched to DrillMotion) along with a width sweep and a
  cadence investigation.
- REFERENCE TRACKING (D2, implemented + tested but the 8-track table was not produced): the retargeted
  video tracks are NOT holdable at frame 0 (CoM margins -0.396..+0.073 m; 2 tracks start with NO foot
  on the mat) and topple in 0.9-1.2 s even with a CoM clamp — i.e. the video-derived references need
  the same GEOMETRIC REPAIR as the GrappleMap references (E9/E12). Tracking machinery exists
  (src/drill/tracking.py: Track/TrackController/TrackParams + margin-driven reference clock +
  support-envelope leaning clamp + honesty reporting; runner hooks `controller='track'`).
- NEXT (dispatched, DrillMotion): width-vs-steppability decision table; authority mechanisms
  (edge roll, hip abduction/torso counter-rotation, support-foot step-out, pivot quantification);
  cadence improvement; optional track repair; and the deliverable clip
  `videos/solo_drill/final_L2_motion.mp4` (>=60 s, repeated visible steps, 0 falls, HUD, slow-mo of
  one step, evidence bundle).

### E15 (2026-10-08) — S1 solo harness COMPLETE (32 tests) + the T1 gate does not discriminate StandHold
- FACT (SoloEnv, verified by orchestrator: tests/test_solo.py 32 passed): src/solo/ (15 modules:
  scene/stance/commands/pushes/fall/markers/obs/reward/metrics/env/eval/baselines/video/lock/train).
  Contract verified in-code: scene composed at load via MjSpec from robots/g1/g1.xml with the `a_`
  prefix (nq36/nv35/nu29, timestep 0.002 s, control 0.02 s = 10 substeps = 50 Hz, mass 33.341 kg);
  29 joint targets (absolute + residual `base + scale*tanh(z)`); actor obs 115 / privileged 43 /
  critic 158 with the actor PROVEN blind to contacts+markers; reset = `a_stand` + joint noise
  ±0.03 rad, xy ±0.02 m, yaw ±10°, tilt ±2°, with 16/16 seeds holding 2 s (min pelvis 0.7894 m,
  max tilt 3.58°, drift ≤0.061 m).
- PUSH MACHINERY verified as the review demanded: mj_applyFT → qfrc_applied at a chosen world
  height; gravity-off momentum matches J within 0.02% (no silent clamping); qfrc_applied and
  xfrc_applied are EXACTLY zero when idle; A/B base-velocity delta > 0.10 m/s measured; J = 12 N·s
  at z=0.95 topples but at z=0.79 does not.
- FALL DETECTOR: pelvis + tilt + 0.25 s persistence; dorsal is a SEPARATE failed-attempt verdict
  (0.30 s persistence, reusing the wrestling back-detector functional API with a single-robot dict);
  settled kneel (0.498 m knee contact) and hands-plant (0.118 m arm contact) both predict False;
  no false positives over 3 s stand + 3 s scripted stepping.
- REWARDS: 19 terms / 6 task sets, per-term logging, weights = literature placeholders; the
  alive > sum(penalties) invariant is asserted; every term hand-tested; all eight degenerate
  behaviours (zero action, StandHold, fall-forward, squat-repeat, sliding, knee-park,
  shot-never-exits, collapse) score worse than a genuine attempt, and collapse loses to upright
  over a fixed horizon; gamma 0.995 (0.997 for locomotion).
- THROUGHPUT: 594-615 bare env steps/s (~12x realtime); PPO trainer 175-215 steps/s end-to-end
  under concurrent agent load.
- TRAINER: src/solo/train.py reusing rl.net/rl.ppo/rl.checkpoint; start = `MUJOCO_GL=egl
  .venv/bin/python -m solo.train --task balance --steps 2000000 --out checkpoints/solo/t1_balance.pt`;
  Ctrl-C saves and exits 0 (verified via a real SIGINT subprocess); `--resume` restored
  steps_done=3072 with RNG/optimizer/LR.
- HONEST FINDING (important for S2): the T1 battery does NOT discriminate StandHold on fall rate —
  a stiff position-servo stand survives most chest-height impulses up to 12 N·s (the analytic
  non-stepping ceiling is ~13 N·s), and the gate rejects it only on mean_upright (0.923 < 0.95).
  S2 must extend magnitudes/directions BEYOND the ceiling and add time-to-stability and CoM-margin
  criteria (baseline-first measurement, not tuning). All nine probe/baseline clips confirm
  `all_not_certified: true`.
- PLAN: S2 (T1 balance training) is deliberately DEFERRED until the drill's stepping burst frees
  the box — the video is the priority and the CPU is the scarce resource. Note for that dispatch:
  the trainer should take data/locks/sim.lock only around short passes, NOT across a multi-hour run,
  or it will starve the drill work (the same trap the render suite had).

### E14 (2026-10-08) — L2 step: FIRST VIOLATED CONSTRAINT named + render lock fixed
- STEP DIAGNOSIS (DrillDirector, reports/2026-10-08/drill.md §7.1): the first violated constraint in
  the L2 entry run is the SWING leg's ankle_roll limit (±0.2618 rad), 2 ticks after the lift command
  begins. Trace: at the unload gate (t=1.18 s) CoM y=-0.058, base y=-0.075, foot load 68/270 N; then
  swing ankle_roll runs -0.275 (1.22 s), -0.325 (1.24), -0.375 (1.26), -0.415 (1.28) while the CoM
  only reaches y=-0.074. CAUSE: the unload gate fires on the swing foot's VERTICAL LOAD (0.21 of body
  weight) while the CoM is still ~0.05-0.06 m short of the 0.10-0.13 m lateral travel this staggered
  base requires, so the swing leg absorbs the residual in ankle roll, saturates, and the geometry
  breaks. FIX (stated, dispatched): gate the lift on the measured CoM margin over the SUPPORT foot's
  own hull (≥ +0.02 m), keep the swing-side hip/knee compliant through the shift, and allow a small
  support-foot yaw pivot to shorten the required lateral travel; land with the existing load-gated
  landing.
- RENDER LOCK FIXED (self-inflicted bottleneck removed): the suite now takes data/locks/sim.lock only
  for the 6 simulation passes (~5 min) and renders LOCK-FREE from cached traces. Render priority:
  final_L1_90s.mp4 → 01_baseline_stancepd.mp4 → 99_failure_entry_L2.mp4 → pushes → L0/slowmo at 0.5
  scale named `_diag`. Every clip is ffprobe-verified (h264/yuv420p/960x720, duration+frames vs the
  trace) plus a non-black/non-static check, with a data/solo_drill/<name>.json evidence bundle.
- STANCE now matches the operator's measured reference on width (0.495 m vs spec 0.491) with hands
  at hip height; torso pitch 15.3° vs 47° recorded as a morphology-bound difference (CoP inside four
  5 mm contact spheres; ankle roll saturates at 0.26 rad). Margin improved: built +0.086 m, worst
  in-run +0.0295 m; rubric A1 3, A3 3, A6 2.
- NOT DONE (dispatched to a new agent, DrillStep2): the CoM-gated lift implementation, the
  reference-trajectory tracking mode (src/drill/tracking.py over data/refs_video/*.npz — the strongest
  lead for L2/L3), and TeacherAdapter wiring to the single-robot RobotTeacher.

### E13 (2026-10-08) — FIRST CLEAN ARTIFACT: drill L1 (90 s, 0 falls) + teacher fixes landed
- DRILL (DrillDirector, verified by orchestrator: tests/test_drill.py 14 passed; stance_report.json
  read directly): L1 CLEAN and shipped — 90.0 s continuous, 0 falls, 0 resets (one initialisation),
  18 programme cycles, worst CoM margin +0.023 m, loaded-foot slip 0.000 m, actuator saturation
  0.00, worst contact penetration -2.7 mm, worst tilt 13.6 deg. Built stance: 0.315 m wide x
  0.351 m deep, pelvis 0.720 m, CoM margin +0.077 m (analytic hull of the 8 sole contact spheres),
  knees 0.56/0.85 rad, torso 9.2 deg, both soles flat. Same-pose PD baseline TOPPLES at 8.4 s —
  a clean contrast for the video. Push limit measured: 20 N recovered 2/2, 35 N 1/3, >=50 N 0.
  Ladder: L0 clean; L1 clean; L2 PARTIAL (isolated step clean: slip 0.000 m, 3.8 cm clearance,
  load-gated lift; but the stand->stance entry walk falls at 5.8 s); L3/L4 not attempted.
  Artifacts: src/drill/* (13 modules incl. rubric.py), scripts/solo_drill_render.py,
  tests/test_drill.py (14), data/drill/* (stance/push/suite JSON + cached traces), report
  reports/2026-10-08/drill.md. Renders were IN FLIGHT (~35 min per 90 s clip, 9 clips);
  videos/solo_drill/final_continuous_drill.mp4 exists but was still being written at inspection.
  [CORRECTED 2026-10-08, see E28: this file does NOT exist. A repo-wide glob (`**/*continuous_drill*`)
  returns nothing; the in-flight render never completed. Only scripted interim clips exist —
  final_L1_90s, final_L2_motion, final_L2_entry_walk — so the mandate's acceptance artifact is UNMET.]
  Traps fixed and pinned by tests: mj_jacSite needs mj_comPos; straight-leg IK seed is singular;
  mixed foot-frame/footprint-centre plans cause drag; cfrc_ext is torque-first vs xfrc_applied
  force-first; CoM-servo weight shift must use an absolute per-step target; teacher ki/int windup
  (reduced 0.6/0.12 -> 0.15/0.03); a 1 cm edge roll collapses a point-based support polygon.
- TEACHER FIXES (TeacherExecFix, verified: tests/test_teacher.py 18 passed): F1 live-reference
  update (set_rows/update_reference, 7-row window per tick), F2 cadence correct (measured 375
  control calls / 7.48 s = 50 Hz, rolling timeline, no 18 s cap), F3 governor now activates on the
  UNCLIPPED capture-point error, F4 heading written into the reference quaternion. Single-robot
  port done: RobotTeacher = one explicit 29-target context (name-resolved ids, no hardcoded slices),
  TeacherController = thin 58-target composition, teacher/solo_scene.py single-G1 scene.
  Headline numbers: stance hold 12 s, drift 0.034 m, tilt 4 deg (rubric A 1->2); level change
  executes knee 0.18-0.23 rad but topples at every depth (rubric C stays 1); STAND_UP reference
  REBUILT (chain t1125->t380->t540, 244 frames, rms 0.041, 0% airborne vs 71% before, junctions
  rms 0.0, min foot z -0.014/-0.021 m; sha256 e7a80311…, old file kept as STAND_UP.airborne.npz),
  stay_up still 0.000 (no rise primitive) but scorer 0.324 -> 0.577; stance sweep: widths
  0.14-0.42 m all hold 8 s (drift <= 0.037 m) while IK foot-placement stances collapse in a 1 s
  settle -> shipped width 0.30 m, sagittal 0.
- STEP PHYSICS (measured, explains the L2 entry failure and constrains all future stepping): a step
  needs ~0.13 m of lateral CoM shift; with both feet pinned flat the legs deliver only ~0.033 m, and
  ankle roll saturates at +-0.26 rad near a 0.10 m shift. Therefore stepping must unweight first
  (load the support foot, soften the swing-side hip/knee, use foot yaw/pivot for lateral reach) —
  which is exactly what the drill's load-gated lift does.
- NEXT (dispatched): integrate ONE clean step per programme cycle into the headline sequence, fix or
  honestly drop the entry walk, then attempt level change -> lead-foot entry (CoM outside the hull
  is legitimate here) -> knee lowering -> trail-leg drive -> rise, each as its own gate; render the
  headline clip first; frame-level self-critique against the operator's STANCE/STALKING/CIRCLING
  reference frames (rubric H).

### E12 (2026-10-08) — feasibility audit: only 6.9% of reference frames are statically holdable; STANCE misses by 1-10 mm
- SOURCE: reports/2026-10-08/support_envelope.md (+ data/support_envelope.json, script, PNG).
  Method: kinematic replay, CoM = subtree_com (mass-verified 33.341 kg), contact = sole patch rule,
  classifier = convex hull of sole contact patches; 2115 non-airborne robot-frames, LP feasibility
  agrees with the geometric class on ALL of them (0 mismatches). Runtime ~30 s.
- NUMBERS: aggregate 6.9% of all robot-frames / 9.2% of standing frames have the CoM inside the
  support region. Per technique (standing-frame infeasible %, holdable window): STANCE 100/100,
  none, deficit 0.002 (a) / 0.010 (b); SNAPDOWN 100/100, none, 0.029/0.051; STAND_UP 100/100, none,
  0.025/0.036; DOUBLE_LEG 81.7/25.4, 0.28/0.62 s, 0.774/0.255; SINGLE_LEG 72.9/90.2, 0.98/0.34 s,
  0.657/0.452; BODY_LOCK 84.1/81.9, 0.56/0.26 s, 0.377/0.328; SPRAWL 65.1/56.8, none/0.32 s.
  Deepest violation overall: DOUBLE_LEG a f152 (t=3.04 s, margin -0.774 m); worst standing episode:
  SINGLE_LEG a f28-32 (one foot down, CoM 0.62 m from the planted ankle — a lunge in free fall).
- KEY IMPLICATIONS: (1) ankle torque is NOT the binding constraint (best achievable worst-ankle
  moment <= 23.9 Nm vs the 50 Nm limit) — the limit is geometric (CoP must lie inside the contact
  patches); (2) STANCE is only 1-10 mm outside the hull, so FOOT PLACEMENT (centimetres of hull
  movement) fixes it while joint-offset trims cannot — this explains the failed trim measurements;
  (3) dynamic frames (dive/penetration/lean) are outside the hull BY DESIGN and need the dynamic
  balance layer, not trimming.
- CONCURRENT CHANGE recorded: data/refs/STAND_UP.npz was replaced mid-audit (06:03, 470 -> 244
  frames) by the STAND_UP-rebuild work; the audit anchored itself to file hashes. Any claim about
  STAND_UP must now cite the new file's hash.
- PROCESS LESSON: that agent's run FAILED at the end because its yield payload exceeded the tool's
  JSON limits (4 KB+ nested). Convention: yields stay small (status + paths + headline numbers);
  detail goes to the report file on disk.

### E11 (2026-10-08 night) — teacher FINAL: honest FAIL, causes handed off
- ACCEPTANCE (final): stay-up gate MET only by SNAPDOWN 0.956 (a 0.98/b 0.96, worst seed 0.89).
  Missed: STANCE 0.758, DOUBLE_LEG 0.520, SPRAWL 0.316, SINGLE_LEG 0.267, BODY_LOCK 0.200
  (baseline best-achievable 0.229), STAND_UP 0.000. SCORER gate (0.85) met by NO technique:
  best SNAPDOWN 0.808 (min phase 0.72), BODY_LOCK 0.679, SPRAWL 0.559, STANCE 0.547,
  SINGLE_LEG 0.361, DOUBLE_LEG 0.324, STAND_UP 0.324.
- BODY_LOCK remediation DISPROVEN as a contact-softness problem: a derived scene
  (robots/wrestling_scene_soft.xml, solref (0.010,1.0) / solimp (0.95,0.99,0.001,0.5,2.0) on 142
  geoms) left inter-robot penetration UNCHANGED at 5.9 cm and self-penetration at 4.4 cm, with
  stay-up 0.200 -> 0.223. Conclusion: the overlap lives in the REFERENCE CLINCH GEOMETRY, not in
  contact parameters — reference-side work is required (consistent with E9's data-bug hypothesis).
- STANCE trims (the only ones that survived measurement, all seeds): a_ hip +0.10 / knee -0.20 /
  ankle -0.10 / waist +0.30; b_ hip +0.10 / ankle +0.20 / waist -0.10. IMPORTANT: the operator's
  two-axis rule measured WORSE at the magnitudes tried (rear-leg-back 0.486, splay 0.458-0.569 vs
  the shipped torso-pitch trim 0.792) — so the rule is not yet validated by measurement and needs
  the finer/combined sweep now assigned (it is a claim to test, not an established fact).
- Verified fixes this round: `_smooth` is now a symmetrically edge-padded convolution (true
  zero-phase) with a step test asserting the half-rise sits exactly at the step index; PRONE now
  requires simulated pelvis < 0.55 m (else LOW, stabiliser stays on); the requested default-config
  first/second-tick finite+n-in-range test exists. The three external defect claims remain
  non-reproducible in this tree.
- Artifacts: data/teacher_stats.json (8 entries incl. BODY_LOCK_soft), data/teacher_trims.json
  (STANCE only), reports/2026-10-08/teacher.md (19 sections incl. the fix-agent hand-off + rubric
  assessment A=1,B=0,C/D/E=0-1), scripts/run_teacher.py + tune_teacher.py, src/teacher/*,
  tests/test_teacher.py (12 tests; full suite 115), videos/teacher/* (7 clips, 320x240 — NOT the
  960x720 evidence contract).
- STATUS: no ship candidate from the teacher; the driveable interface is code-only until
  TeacherExecFix lands the F3 fix, the single-robot 29-target port and the support-transfer
  primitive. The drill's motion source will be whichever of {fixed teacher, DrillDirector's
  FeasibleDrill} first passes a component gate.

- RENDER BUDGET (orchestrator measurement, single-G1 scene, software EGL, under concurrent load):
  960x720 = 0.860 s per control-step frame -> 60 s @30fps (1800 f) ≈ 25.8 min, 90 s (2700 f) ≈
  38.7 min; 480x360 = 0.653 s/frame (only ~25% faster). Cost is dominated by scene update/readback,
  not pixel count — so render cost is NOT the critical path; multiple 960x720 passes are affordable
  (~30-40 min each) and rendering should start as soon as a trajectory passes its component gates.
  The memory hazard remains (retaining all RGB frames ≈ 3.7 GB for 60 s): stream to the encoder.

### E10 (2026-10-08 night) — Codex (gpt-6.1-sol) design review: verdict + must-dos
- VERDICT (docs/reviews/sol61_review.md): the full clean continuous drill is NOT reachable by running
  the present plan unchanged — live commands were dropped, the cited self-check ran at the wrong
  rate, turning is not represented, the runnable controller is PAIRED, and neither stepping nor
  standing recovery has been demonstrated. Completing S1-S8 paperwork or spending time on PPO does
  not remove those physical dependencies.
- REVIEW-STATUS of the four patch findings: F1 (stale procedural reference) and F2 (500 Hz self-check)
  confirmed addressed by the fix agent (static inspection); F4 (yaw not written into the reference)
  addressed, but yaw feedback gains are ZERO so turning is still unproven; F3 (governor activates on
  the CLIPPED capture-point error, max ≈0.113 m, below the 0.12 m DIVE threshold) STILL PRESENT.
- STRUCTURAL HAZARDS (verified in code): (a) the runnable controller is paired (constructs a_ and b_,
  hard-coded 36/72 and 29/58 slices) — the solo milestone needs one explicit robot context returning
  29 targets; (b) pose modulation is not locomotion — no load/unload/swing/landing primitive, no
  trail-leg stage, no foot-landing guard; (c) RECOVER_STAND blends on ABSOLUTE drill time (after ~2 s
  it selects full stand) and the 900-row table clamps after 17.98 s, which cannot serve a 60-90 s
  drill; (d) PRONE/support-mode decisions use pelvis height and site-height "contact", not loaded
  contact — a kinematic sole hull is not a support polygon; (e) sprawl_end_posture identifies a prone
  defender from low pelvis alone, so a supine pose satisfies it; (f) the env's `stood` notion (no
  dorsal trigger + no OOB) does NOT exclude knees/hands/sideways collapse — E1 had 83% ground time
  with zero dorsal contact; (g) the progress term is clock-based and _start_exchange resets pose and
  velocity — a continuous-evidence runner must keep ONE MjData and never use the exchange reset;
  (h) reward.py's `score_every=1` modulo hole produces NO score samples — fix before any experiment
  relies on scorer gating.
- RESOURCE REALITY (recorded): ~2.1 s per 320x240 rendered frame; 960x720 for 60 s unmeasured; the
  existing renderer retains all RGB frames pre-encode (~3.7 GB at 60 s/960x720 on a no-swap host) —
  cache the physics trajectory, inspect low-res diagnostics, stream the final encode.
- PROVENANCE: teacher.md prose (STANCE 0.792, 3 seeds) disagrees with data/teacher_stats.json
  (0.7583, 5 seeds); stay_up_frac's definition is loose (tilt allowance max(30°, ref+15°), low-ref
  frames excluded) and tests accept mean 0.65 / worst 0.50 — a regression guard, not a ship gate.
- DISPATCH: TeacherExecFix extended (F3, single-robot 29-target port, support-transfer primitive,
  per-element blend times, rolling timeline, provenance reconciliation); DrillDirector extended (one
  robot/one reset/one continuous state, streaming render + low-res diagnostics, component gates on
  physical events rather than rungs of pose modulation, never use `stood`/low-pelvis/geometry score
  as a success gate).

### E9 (2026-10-08 night) — HYPOTHESIS: some reference failures are a DATA bug, not a control limit
- Evidence: STAND_UP's reference is AIRBORNE (min foot-site z 0.13-0.72 m, pelvis 1.0-1.08 m) —
  physically impossible for a grounded stand-up, which is why its stay-up is 0.000. The
  retarget/montage pipeline therefore CAN emit impossible trajectories.
- Hypothesis: part of the "6/7 references fail physically" failure set is reference-invalid
  (airborne frames, montage-seam discontinuities, or support that no CoM position can satisfy)
  rather than a stabiliser limit. Suspects, by stay-up: DOUBLE_LEG 0.520, SINGLE_LEG 0.267,
  BODY_LOCK 0.200 (STAND_UP 0.000 is the confirmed case).
- TEST (dispatched to SupportEnvelope, extending the same audit): per-frame min foot-site z with
  airborne flags, montage-seam continuity (joint/velocity jumps vs in-segment norms), and
  contact-vs-support-impossibility checks, for all 7 references; then classify each technique as
  (i) reference-invalid, (ii) valid-but-dynamically-hard, or (iii) valid-and-a-controller-limit.
- DECISION RULE: only technique class (iii) justifies more stabiliser work; class (i) means the
  reference must be REGENERATED (retime/repair/re-chain) before any controller tuning, because
  no controller can hold an impossible reference. Fixing reference generation may lift several
  stay-up numbers at once.

### E8 (2026-10-08 night) — teacher: NO SHIP CANDIDATE + two verified execution defects
- FACT (TeacherRetry, honest report): driveable SkillController shipped (12 skills, set_command,
  set_stance_height, set_lead_step, 0.6 s crossfades, balance layer on, control() -> (58,) @ 50 Hz)
  but drill elements topple within ~1.6 s from a live start; stance hold 2.34 s with 0.28 m drift;
  rubric scores A=1, B=0, C/D/E=0-1 -> explicitly "do not produce the drill video from it yet".
- FACT (same report, rate table): stay-up met by SNAPDOWN 0.956 (5 seeds) and STANCE 0.792 (3 seeds,
  with a torso-pitch trim); DOUBLE_LEG 0.520; SPRAWL 0.316; SINGLE_LEG 0.267; BODY_LOCK 0.200;
  STAND_UP 0.000 because its reference is AIRBORNE (min foot-site z 0.13-0.72 m, pelvis 1.0-1.08 m)
  -> reference rebuild required. Scorer gate 0.85 met by none (best SNAPDOWN 0.808).
- FACT: coarse two-axis stance trims were REJECTED on evidence (rear-leg-back 0.486 / widened
  0.458-0.569 stay vs torso-pitch trim 0.792, 3 seeds) and the controller enforces no stance-width
  minimum; a finer, combined sweep is required.
- DEFECT 1 (review-verified, src/teacher/controller.py:119-158, src/teacher/skills.py:131-135):
  the procedural table mutated by SkillController never reaches the executor — _build() constructs
  the teacher from the initial constant table and _RobotCtx copies it, precomputing joint/foot/
  height/heading/phase targets — so LEVEL_CHANGE / SHOT_DOUBLE_LEG / KNEE_LOWER / RECOVER_STAND and
  movement commands never change what is actually executed. Matches the previous agent's own
  inference that the stabiliser's reference fights the trim ramp-in.
- DEFECT 2 (review-verified, src/teacher/skills.py:209-212): run_self_check() calls control() before
  every 0.002 s mj_step → the controller runs at 500 Hz, so its 18 s reference is exhausted after
  1.8 s and integrators/rate limits/blends run 10x fast. Consequence: the "topple ~1.6 s" result
  that drove the no-ship verdict was measured on a MIS-CADENCED harness — it may be real, but it is
  NOT yet demonstrated.
- DISPATCH: TeacherExecFix (new agent) fixes both defects, re-measures before/after with the same
  harness, rebuilds the STAND_UP reference, enforces the stance-width minimum and re-sweeps the
  two-axis trims at finer magnitudes + combined. DrillDirector told not to depend on the teacher yet
  (its own conservative FeasibleDrill may out-score the current teacher stance).

### E7 (2026-10-08 night) — outcome-first critical path to the drill video
- OUTCOME: one clean continuous MP4 of one G1 performing a stance-and-motion drill. Decomposed:
  (a) hold a wrestling stance with feedback [today: NOT demonstrated], (b) state-responsive footwork,
  level change, penetration-step gesture, knee, trail-leg, rise, (c) rendering/assembly with overlays
  and no resets. The brief allows a scripted SEQUENCE if execution is feedback-driven, so the critical
  path is CONTROLLER quality; RL/BC is deferrable tonight.
- DISPATCH (outcome-first): TeacherRetry = motion source (driveable interface: skill + velocity +
  stance-height -> 29 joint targets); DrillDirector = scheduler + adapter + continuous harness +
  overlay renderer + FALLBACK LADDER (L0 static stance/posture modulation -> L1 pivot/circle on
  planted feet -> L2 single-foot repositioning (no slide) -> L3 alternating shuffle + circle ->
  L4 penetration-step gesture + knee + trail-leg + rise). Highest CLEAN rung is what the video shows,
  labelled on screen. SupportEnvelope informs which postures are statically feasible; RefToTraining
  (bounded to 4 priority chapters) provides the visual target; SoloEnv provides metrics/baselines.
- DECISION POINT: if the teacher's stance/level-change is not clean in ~2 h, the deliverable video is
  the highest clean rung of the fallback ladder, labelled honestly (controller-driven, not learned),
  with the learned-policy work continuing afterwards.
- HONESTY CONTRACT (applies to the video and its report): no reset inside a successful run; a fall
  or reset means the clip is labelled as such; metrics JSON beside every video; failure clips kept.

### E6 (2026-10-08) — P0 acceptance-logic fixes (P0Fixes; verified by orchestrator)
- Curriculum gate rewritten: success is now per-stage — criterion='execution' for A-C (held ground
  AND mean technique-similarity >= min_similarity) and 'outcome' for D (learner won); draws,
  losses and unmeasured attempts are non-success slots. Old code provably advanced a stage on an
  all-draw window (metric 0.5 >= 0.5); new code cannot (min_successes >= 1 enforced).
  Wiring added: exchange record -> StageReward.shaping(sample=) -> RolloutCollector (per-exchange
  similarity) -> Trainer -> Curriculum.on_exchange. success_hook left as the documented
  attribution seam (can only reject, never grant); min_similarity=0.5 is a PLACEHOLDER.
- Back-to-mat ambiguity now real: after the first trigger the exchange stays open up to
  AMBIGUITY_WINDOW_S simulating normally, then resolves over all observed triggers. Measured:
  cross-step gap 0.06 s -> ambiguous (was: winner decided by the first trigger); gap 0.20 s ->
  first-trigger winner after a bounded 0.10 s hold. Knees/hands still non-terminal; ExchangeRecord
  semantics unchanged.
- Tests: 115 passed (was 89) incl. cross-step ambiguity, bounded hold, all-draws-never-advance for
  every shipped stage, stage-A similarity advance/low-similarity block, degenerate-gate rejection,
  checkpoint-safety of the new rule.

### E5 (2026-10-08) — operator stance/technique constraint (from personal reference)
- OPERATOR PREFERENCE (2026-10-08): reference video https://vimeo.com/501599802 shows the technique
  he favours; his own variation keeps the leg "a little bit more back for balance" — priority is
  that the robot CAN balance, so rear-leg-back / CoM-over-support geometry is the sanctioned
  adjustment direction when a reference posture is not holdable.
- OPERATOR ADDENDUM (2026-10-08): "if it's falling sideways, legs are too close together — a
  little back and away from the other leg can help." So the balance-repair rule has TWO axes:
  (a) SAGITTAL — falling forward/backward → move the rear leg further back (longer base behind
      the line of action);
  (b) FRONTAL — falling sideways → widen the stance (feet further apart laterally, rear foot also
      angled away), increasing the lateral support width.
  Both preserve the wrestling read (staggered, hips down, hands forward) and are preferred over
  flattening the torso or reducing crouch depth. Record each trim with frame index, the axis
  changed, and the resulting CoM-margin improvement.
- The clip remains UNAVAILABLE from this host (yt-dlp 401 OAuth, player config non-JSON) — one
  more cheap attempt at most, then treat it as "reference unavailable" and work from these two
  verbal constraints; GrappleMap already supplies the shot geometry.
- Consistent with our measurements: the STANCE crouch is not open-loop holdable (CoM behind the
  support polygon; ankle 39/50 Nm) and the two-robot random-STANCE reset collapses every exchange.
  Rear-leg-back geometry is the cheapest relationship-preserving repair.
- BLOCKER (no action taken): the video could not be fetched from this host — yt-dlp fails with
  "Failed to fetch macos OAuth token: HTTP Error 401" and the player config endpoint returns
  non-JSON. If the file is dropped on the box (any path under data/references/), we will extract
  frames and fold its geometry into the stance + penetration-step references.

### E4 (2026-10-08) — env rule videos (EnvVideo) + prior-art review (PriorArt)
- EnvVideo: all five rule demonstrations rendered and verified (h264 960x720, decode-clean):
  draw-on-timeout, back-contact -> 0.30 s persistence -> trigger -> score -> reset, knees/hands
  with dorsal=0 -> exchange CONTINUES, simultaneous back events -> ambiguous (same control step),
  3 boundary events -> forfeit. Also visible: the P0 ambiguity hold (trigger 1.70 s -> end 1.80 s).
- FINDING (affects solo env design): the two-robot env's default random-STANCE standing reset is
  NOT balanceable by the position-servo G1 — every exchange after the first collapses within ~2 s;
  the video agent had to substitute the verified-stable `stand` keyframe to produce honest rule
  demos. DECISION: solo-env resets come from the `stand` keyframe with verified-holdable noise.
- Frame-level verification (FrameCheck, reports/2026-10-08/env_videos_frame_check.md): all five
  clips MATCH their claims (49 frames + contact sheets inspected, full-decode, blackdetect,
  HUD-pixel check). Recorded artifacts: (a) robots topple in the takedown clip (disclosed phase-2
  limitation); (b) supine robots hold arms stiffly (no controllers); (c) the OOB offender GLIDES
  across the mat with planted feet instead of stepping — scripted root motion, i.e. a rubric-B1
  (no foot sliding) violation to avoid in the drill deliverable.
- PriorArt (docs/prior_art_humanoid_control.md): (1) penalty-only rewards make FALLING optimal
  (episode ends, penalties stop) -> every task needs a positive per-step alive/upright term that
  dominates per-step penalties while upright; (2) no reward detects a statue — gates must be
  held-out push batteries (published ablation: dropping the balance term collapses max
  recoverable push 230 N -> 21 N); (3) at 50 Hz gamma=0.99 = ~2 s horizon, too short — use
  0.995-0.997; (4) portable G1 templates exist (joystick-style vertex tracking + feet air/slide
  terms) — weights are starting points, not tuned values; (5) CPU realism: 222-437 steps/s here
  -> 10M steps = 5-14 h, 100M = 2-4 days; published 147M-400M budgets are 5-60 days, so balance
  is reachable overnight but long-horizon locomotion is not.

### E3 (2026-10-08) — M0 audit results (MotorAudit; reports/2026-10-08/motor_audit.md)
- ABSENT (no code at all): (1) dynamic-balance push recovery — there is NO push/perturbation
  machinery in the repo (no xfrc/qfrc/apply_force); (2) ALL locomotion — walk fwd/back/
  lateral, velocity tracking, turning, accel/decel/stop, direction change; (3) support
  polygon / convex hull (only a support-centre mean point, inside the in-flux teacher);
  (4) fall detection/termination for balance tasks (termination is match-clock only; the
  back detector catches only persistent dorsal contact); (5) wrestling locomotion —
  shuffle/circle, angle change vs opponent, controlled knee drop; (6) arm/reach while
  balancing, torso+arm+leg coordination.
- CODE-ONLY (exists, never validated): CoM/capture-point/support-centre/foot-contact logic
  lives only in the in-flux teacher (controller.py:277, stabilizers.py:120-124,189,198) —
  no test, no report, no video.
- MEASURED FAILURE: stance maintenance, level change without collapsing, recovery from
  non-standing states (STAND_UP PD replay never completes; bodies end dorsal).
- No learned motor behaviour exists anywhere: best measured policy stands 13.1% of the time
  (E1). Static stands are scripted-hold only (5 s keyframe; 20 s StandHold draw).
- ADVANCEMENT: outcome-rate gate measured to be unsafe (stage-C smoke W/L 299/0/20 while the
  same checkpoint stands 13.1%); wins can come from opponent collapse; no attribution; single
  metric. Gate fix in flight, now keyed to each stage's OWN objective (similarity for A-C,
  outcome for D-E) after the wins-only variant was rejected as a stage-A deadlock risk.
- FORGETTING surfaces: single policy across the ladder (trainer.py:130-191 rebuilds env
  only), shared 84-dim/29-action interface, window reset on advance, no replay/regularisation
  /snapshot league.
- FIVE UNCERTAINTIES + smallest tests (audit §5): U1 can a position-servo G1 learn sustained
  balance at all (M1 harness baselines + 300k PPO, held-out tilts); U2 does the env express
  M1/M2 (fall detector + velocity-command wrapper, validated on scripted falls); U3 how much
  retargeted geometry is outside the static support envelope (→ SupportEnvelope task, launched);
  U4 can the PISTY ladder be salvaged for M5 (score the smoke checkpoint with M-gates);
  U5 CPU budget realism (benchmark M1 task, one 1M-step run).


### 2026-10-08 — Phase 5 infra (RLTrainer, verified by orchestrator)
- FACT: PPO/resistance infra in src/rl (obs, privileged, net, ppo, rollout, checkpoint,
  trainer, curriculum, reward, scripted, vec). Actor obs 92-dim (84 env obs + 7 technique
  one-hot + phase); critic 254-dim = actor + 162 privileged; actor cannot read privileged
  (structurally tested). Actions 29/robot, absolute or residual, ctrlrange-clipped.
- FACT: measured 222.7 steps/s (30k-step stage-C smoke vs STANCE replay, 134.7 s wall);
  subprocess vectorization 1.6-2x sequential for >=3 envs (421-437 vs 172-231 steps/s).
- FACT: SIGINT resume verified (partial rollout discarded, state/RNG/stage restored);
  eval deterministic for fixed seed (re-run asserts identical records).
- FACT: BC warm-start hook exists (StageConfig.bc_checkpoint -> warm_start_from_bc, key/shape
  matched, mismatches reported) — Phase 3B plugs the BC policy in here.
- SUITE: 89 tests pass on this host (orchestrator rerun).
- FIXED (BackdetFix, verified): src/wrestling/backdet.py limb_contact now resolves limb
  bodies by per-robot suffix match (a_left_knee_link etc.). Differential replay over all
  7 refs x 2 robots: detector outputs bit-identical, limb_contact flips False->True as
  intended; regression test added; suite 89 passed. Diagnostic-only path.

### E1 (2026-10-08) — random-init PPO baseline (no imitation) | ABLATION ARM 1 of MISSION
- Question: what does PPO alone learn before any GrappleMap imitation? (MISSION wants
  "random-init PPO" vs "imitation -> PPO": samples to first successful takedown.)
- Setup: 30k steps, stage C vs STANCE reference replay, checkpoint smoke_stage_c.pt.
- Result (orchestrator rerun of scripts/eval_ppo.py, deterministic): standing fraction
  0.131 (pelvis z >= 0.5 m); ground fraction 0.830 (z < 0.35 m); mean pelvis z 0.267 m;
  knee/hand contact 0.413; dorsal contact 0.000; learner W/L/D 48/0/3 over 51 exchanges;
  48/51 cases the opponent's back hit the mat (scripted opponent also collapses).
- Conclusion: without imitation the policy degenerates to ground-hugging and wins only
  because the scripted opponent falls; it does not wrestle standing. This is the measured
  motivation for the teacher/BC stage (Phase 3B) and the reference point for the ablation.



## Technique vocabulary (DECIDED 2026-10-07 — see docs/CURRICULUM.md)

Attacks: double leg, single leg, body lock. Defense: sprawl, hips-back, recover
stance. General: stance, approach, retreat, circle, failed-shot recovery (knees→stand).
Target 5–8 reliable movements, not 30 poor ones.

### 2026-10-07 S1 — GrappleMap data format FACTS (GrappleMapParser; details reports/2026-10-07/grapplemap.md)

- FACT: vendored third_party/GrappleMap @ 032c8f91809786b7b784852abd07cf3e10c0ea35.
  Whole DB = GrappleMap.txt: 601 named nodes, 1485 transitions, 8323 positions;
  after graph linking 725 nodes (124 unnamed auto-added).
- FACT: pose = 2 players x 23 joints x 3 (x,y,z) float64, meters. 23 joints =
  10 L/R pairs (Toe..Fingers) + Core/Neck/Head. "23 landmarks/player" confirmed.
- FACT: Y-up, ground plane y=0 (xz). x,z in [-2,1.843], y in [0,1.903]. Base62:
  2 digits/coord, (d0*62+d1)/1000, x/z shifted by -2 (mm quantization).
- FACT: NO timestamps; uniform keyframes, 5 intervals/s (0.2 s), 10/s if
  property 'detailed'. Edges link nodes via rot-Y+translation (+optional
  mirror/player-swap), match tol 4 cm/joint.
- FACT: mirror = negate x + swap L/R limbs within each player (involution);
  swap_players = exchange 23-joint blocks (involution). Both defined in source.
- FACT: mover metadata = edge 'properties: top|bottom' (player 0|1 of the
  edge's own frames; 616/558/311 top/bottom/unmarked). Tags: 161 distinct
  incl. standing, double_leg_takedown, single_leg_takedown, body_lock, sprawl,
  stand_up, technical_standup, collar_tie.
- DECISION (parser contract): src/grapplemap exposes load_graph() -> Graph with
  Node.position (2,23,3), Edge.frames (F,2,23,3); tests 17 green via
  .venv. Vocab inventory: data/grapplemap/vocab_candidates.json (7 groups, 58
  edges). No reshot/penetration-step/hips-back edges exist in the DB.

### 2026-10-07 S1 — VOCABULARY DECISION (orchestrator, from parser proposal)

- DECISION: 7 conditioned techniques for phases 2-5 (indices = g.edges[N]):
  1. DOUBLE_LEG [1144, 1147, 1141, 1140] (attack)
  2. SINGLE_LEG [978, 1001, 1094] (attack; head-outside alt t1003+t1000 later)
  3. BODY_LOCK [1133] (attack; alt t1150)
  4. SNAPDOWN [989] (offense/counter; sets up shots, teaches head control)
  5. SPRAWL [381, 382, 383] (defense vs double leg; defender legitimately
     ends prone — geometry defines it, not a fall)
  6. STAND_UP [952, 1110, 1125] (recovery: stand-up + technical standup)
  7. STANCE (procedural baseline pose from 'symmetric staggered standing'
     node geometry; not an edge chain)
- Locomotion (approach/retreat/circle) built procedurally in env (phase 4+),
  not from GrappleMap. Deferred: t895 sprawl→single reattack chain (phase 5
  chaining candidate), collar-tie/clinch entries t587/t451 (phase 5+).
- Rationale: MISSION attack families (double/single/bodylock) + defense
  (sprawl; hips-back absent from DB, folded into sprawl/stance behavior) +
  recovery (stand-up) + snapdown for setup/re-engagement behavior. 7 < 8 cap.

### 2026-10-07 S1 — Harness availability (HarnessProbe; reports/2026-10-07/harness_probe.md)
- FACT: delegate driver MISSING (~/.bin, PATH, ~/.agents checked; only SKILL.md exists).
- FACT: opencode v1.3.3 binary works (`models` OK): 11 opencode/*-free entries,
  22 fireworks-ai/* (operator-excluded), **opencode-go/* NOT LISTED** (0 entries).
- FACT: free-tier ping opencode/mimo-v2.6-flash-free FAILED: "OpenCode's free tier
  can only be used from within OpenCode" — `opencode auth list` shows 0 credentials
  (only FIREWORKS_API_KEY env). [INFERENCE] console login missing.
- FACT: cursor absent. freebuff/codex/claude/gemini binaries present, untested (cost).
- BLOCKER (operator action needed): opencode delegation requires `opencode auth
  login` (interactive, user credentials); opencode-go/* may only appear post-login.
- DECISION: until operator completes login + says OK, ALL subagent work stays on
  `task` (zai). Project is not blocked — Phase 2 Retargeter in flight.

### 2026-10-07 S1 — Subagent model routing RESOLVED (operator clarifications)
- FACT: "command code" = the **commandcode** provider in omp (87 models, backed by
  the operator's OpenCode Go plan, authorized). "opencode go" refers to the same plan.
  Relevant ids: commandcode/deepseek/deepseek-v4.1-flash(+fast),
  commandcode/xiaomi/mimo-v2.6-flash, commandcode/z-ai/glm-5.3-flash(+flashx).
- FACT: main session model = zai/glm-5.3 (modelRoles.default). Operator wants
  subagents OFF zai to conserve it: deepseek + mimo flash for implementation work.
- FACT: opencode CLI on this box is NOT authenticated (auth.json 0 credentials,
  free tier blocked, opencode-go/* not listed) — CLI-based delegation unavailable
  until `opencode auth login` (operator action). Not needed if omp routing works.
- BANNED by operator: claude, codex, gemini models; fireworks-ai provider.
  freebuff exists (operator says not quota-limited) as backup.
- DECISION (pending operator OK of the cfg write): set
  task.agentModelOverrides = task→deepseek-v4.1-flash, sonic/scout→mimo-v2.6-flash,
  reviewer→deepseek-v4.1-flash. The cfg write needs interactive approval; retry
  only when operator says ok. Until then subagents run on default (zai).

### 2026-10-07 S1 — Routing UPDATE: commandcode out of credits → opencode-go active
- FACT: ModelProbe (sonic) hit commandcode/xiaomi/mimo-v2.6-flash and got
  HTTP 400 "insufficient credits" — commandcode account exhausted.
- DECISION (operator: "ok use opencode-go"): session cfg task.agentModelOverrides =
  task/reviewer→opencode-go/deepseek-v4.1-flash, sonic/scout→opencode-go/mimo-v2.6-flash.
  Session-only; operator can persist or revert. zai stays main-session only.

- FACT (operator, 2026-10-07): commandcode is "not currently usable" (no credits);
  opencode-go is the working subagent provider. PERSISTED to global omp config:
  task.agentModelOverrides = task/reviewer→opencode-go/deepseek-v4.1-flash,
  sonic/scout→opencode-go/mimo-v2.6-flash (probe-verified). New sessions inherit it.

## 2026-10-08 (DelegateTester) — delegate tool fixed + workflow "ESM" myth corrected
Report: reports/2026-10-08/delegate_and_workflow.md (full verdict table + diffs).
### Delegate driver (was MISSING — built today)
- FACT: omp `delegate` tool = thin execFile wrapper over `~/bin/delegate`
  (`~/.omp/agent/extensions/delegate.ts`: run [task|--brief F] [--session N] [--cwd D]
  [--harness H] [--model M] [--timeout S] / ask <answer> --session N / probe|list|last|template;
  stdout = ONE compact JSON {ok, harness, model, session_id, cost_usd, tokens, question, text, report_path, next_step}).
- FIX: driver IMPLEMENTED at /home/ubuntu/bin/delegate (python3, stdlib-only). Engines:
  opencode headless CLI + freebuff TUI-in-tmux (load-buffer/paste-buffer submit; verdict =
  last `End agent … shouldEndTurn:true` from ~/.config/manicode/projects/<proj>/chats/<iso>/log.jsonl
  anchored AFTER our prompt's log line + quiet ≥9s). auto = opencode→cursor failover; freebuff never auto.
- FACT: opencode headless WORKS now: `opencode run -m opencode/mimo-v2.6-flash-free`
  ~12-15s/turn, $0, no auth needed anonymous; multi-turn via `-s <session_id>`
  (driver stores id from `--format json` events). Prior "free tier only inside OpenCode"
  no longer reproducible (binary 1.18.35).
- FACT: opencode hangs forever when stdin is an open pipe WITHOUT EOF (repro'd: node
  execFile → opencode hung 60s empty streams; held pipe → hang). Driver passes
  `stdin=DEVNULL` — REQUIRED for anyone spawning opencode from node/python
  execFile-style (i.e., from omp tools). Fix verified: NODEFIX_OK 12.7s via node execFile.
- FACT: freebuff TUI pick catalog (0.2.22): Solar Pro 4 / Ling 3.1 (stalled) /
  Laguna S 2.1 (stalled) / Glyph Cluster / Solar Mini 4 / "MiMo 2.6 Flash" (10 FB/hr) /
  GLM 5.3 Flash / DeepSeek V4.1 Flash (15 FB/hr). Binary map: `CH=aV.mimoV25`,
  `s6=aV.mimoV26Pro` → the "MiMo 2.6 Flash" card actually runs `mimo/mimo-v2.5`
  (log-confirmed). NO true mimo v2.6 flash in freebuff (exists on opencode only);
  DeepSeek flash IS reachable via the "DeepSeek V4.1 Flash" card.
- Metering: freebuff = hourly ("51m left", ~10 FB/hr per turn); this session
  105 → 85 Freebucks before guard-fix rerun.
- FACT: freebuff multi-turn works: ask in the SAME TUI chat recalled the prior
  turn's literal (`FB_GUARD_OK`) verbatim after the driver added the turn-anchor
  guard; freebuff queues messages behind busy turns; duplicate submits were observed
  when verification misfired — driver now recognizes all post-submit states
  (saved-HUD / queued HUD / busy thread view / placeholder).
### Workflow runtime: NOT broken — one-export rule
- CORRECTION of "Workflow runtime broken (2026-10-07)": the engine accepts
  `export const meta` FIRST statement + plain body (top-level `await`/`return` okay —
  body is wrapped in an async IIFE and run via `new vm.Script` as a CLASSIC script).
  It REJECTS any other `export` (`export default`, second `export const`) with
  `Unexpected keyword 'export'` shown later as "Workflow was aborted". VERIFIED probe
  (plain body) returned agent replies through a real `agent()` (1 agent, 970 tok).
- File refs: @quintinshaw/pi-dynamic-workflows ~v3.13.1 dist/workflow.js
  parseWorkflowScript (~1384-1413) strips only the first ExportNamedDeclaration and
  vm.Script compile at ~1174; workflow-tool.js ~262 maps abort. omp 18.8.3 == npm latest.
- FIXED: managed skill omp-workflow-esm-fallback/SKILL.md rewritten to the true
  contract (meta + plain body; no export default). Engine NOT patched (upstream
  improvement path documented in report).


## 2026-10-08 (Retargeter) — Phase 2 retargeting FACTS + reference-file contract

### CORRECTION: GrappleMap Y-up -> MuJoCo Z-up conversion is CHIRALITY-CRITICAL
- CORRECTION of the earlier interface-contract entry "(x,y,z)_gm -> (x,z,y)_mj":
  that axis swap is an IMPROPER transform (det = -1) — it mirrors the pair and
  flips every player's left/right. Retargeting with it produced robots facing
  AWAY from each other (solver matches mirrored targets; verified STANCE:
  cross-hip fit 0.15 m vs 0.06 m correct-chirality fit) that then fall under
  PD tracking. The correct conversion is the proper rotation about +x by 90 deg:
  (x, y, z)_gm -> (x, -z, y)_mj. Ground plane still maps exactly gm y=0 == mj
  z=0. Downstream phases MUST use (x,-z,y) (implemented in
  src/retarget/world.py, regression-tested in tests/test_retarget.py).
- FACT: yaw is weakly observable from near-symmetric standing GrappleMap poses;
  the retarget seeds base yaw from the left-hip -> right-hip target line.

### Reference-file contract (data/refs/<TECHNIQUE>.npz, 7 files, Phase 2 deliverable 5)
- Keys: qpos_a (T,36) f64, qpos_b (T,36) f64, t (T,) f64 uniform dt=0.02 s
  (50 Hz), technique (str), edges (int64 array), landmark_rms (f64, weighted,
  meters), meta (JSON str: scales, weights preset, junction + alignment
  residuals, time stretches, repairs log, source commit).
- Robot A = attacker/recoverer (starts -x, faces +x), robot B = defender/
  opponent (+x, faces -x). qpos layout per robot = G1 layout in notes.md
  (base 7 + 29 hinges). quats (w,x,y,z), normed; joints within model ranges.
- Technique -> edges: DOUBLE_LEG [1144,1147,1141,1140]; SINGLE_LEG [978,1001,
  1094]; BODY_LOCK [1133]; SNAPDOWN [989]; SPRAWL [1147,1141,381,382,383]
  (defender chain spliced onto shot frames, alignment rms 0.173 m);
  STAND_UP [952,1110,1125]; STANCE [] (node 94 'symmetric staggered standing',
  65/35 blend with G1 stand pose, hold 1.2 s).
- Two-G1 scene: robots/wrestling_scene.xml (MjSpec-composed; prefixes a_/b_,
  qpos slices [0:36]/[36:72], 58 actuators, keyframes both_stand/a_stand/
  b_stand, offwidth 960). Rebuild: src/retarget/scene.py (g1.xml untouched).
- Scale: per-player LSQ over {Core-Neck, Neck-Head, Hip-Knee, Knee-Ankle}
  medians vs G1 lengths; pair-uniform mean applied (0.75-0.79, human ~1.7 m
  -> G1 1.32 m). Roles resolved geometrically (mover metadata unreliable
  across reorientations; sprawl edges' 'top' contradicts geometry).
- Landmark sites attached at load via MjSpec (19/robot): doc table in
  src/retarget/landmarks.py; Fingers dropped, Hand->wrist; weights HIGH 4 /
  MED 1.5 / LOW 0.5 (strict 8/3/1).
- Validation: scripts/validate_refs.py (PD-track at model servo gains from
  the ref's own initial state; repair = retime x1.25 <=3 then strict preset;
  acceptance mean joint err <=0.15 rad, penetrations <=2 cm, stay-up check,
  SPRAWL defender ends prone).

## 2026-10-08 (agent 2) — deliverable 7: TECHNIQUE-VALIDITY SCORER v1 (FACTS)
Report: reports/2026-10-08/scorer.md (design + full 144-row threshold table +
validation matrices). Deliverables: src/scorer/{__init__,scorer,features,
membership,calibration,spec,config}.py, data/scorer_calibration.json,
scripts/calibrate_scorer.py, scripts/score_trace.py, tests/test_scorer.py.

### Interface contract (append to "Interface contracts")
- API: `TechniqueScorer(model).score(technique, phase, self_qpos, opp_qpos)
  -> ScoreResult(total float [0,1], terms {pred: (mu, weight)}, features
  {name: value})`; module-level `scorer.score(tech, phase, self_qpos, opp_qpos,
  model)` (scorer cached per model object); `phase_at(tech, t)`,
  `phase_at_frac(tech, frac)`, `score_trace(tech, self_traj, opp_traj)` ->
  {per_phase, mean, executor}. Import via `sys.path += src` then `import scorer`
  (package also importable as `src.scorer`).
- self_qpos/opp_qpos = G1 qpos (36,) per notes.md; executor role is per
  technique and recorded in src/scorer/spec.py ("A" for all, "B" for SPRAWL).
- Separation contract: src/scorer imports only numpy/mujoco/stdlib. Value-
  function code must not import it; the ONE sanctioned consumer is
  src/rl/reward.py `ScorerAdapter` (lazy in-function import, form signal only,
  degrades to 0 if unavailable). Enforced by
  tests/test_scorer.py::test_scorer_is_separate_from_value_functions.
- Calibration: thresholds are data, not code. Rule in src/scorer/calibration.py
  (core = reference [p10, p90]; soft ramp = 0.25*(p90-p10) + floor, floor
  0.10 m / 0.15 rad; ang_near circular mean). Regenerate:
  `.venv/bin/python scripts/calibrate_scorer.py --write` (byte-identical
  re-run verified, md5 41a7b634c3b7f97d1e275298b94a5af1); verify:
  same script without --write (exits 1 if acceptance is violated).
- Phase bands = keyframe-index bands; boundaries sit on reconstructed keyframe
  times (build_technique_targets(tech).times * time_stretch_requested *
  time_stretch_kinematic) — machine-checked on every calibrate run, worst
  deviation 0.023 s over 13 boundaries. SPRAWL is the exception (montage:
  junction alignment blurs edge boundaries -> bands placed on the defender's
  own hip/knee/leg-back trajectories).

### Verified numbers (this host, 4-core ARM, `.venv`)
- FACT: self-conformity (reference scored as its own technique, per phase):
  worst phase 0.971 (SPRAWL1); all 24 phases >= 0.971; trace means 0.987-1.000.
- FACT: cross-scores: DOUBLE_LEG as SINGLE_LEG = 0.464 (< 0.5 acceptance);
  diagonal dominance holds for all 7 (self beats every cross by > 0.1).
  Optimistic column: "-> SPRAWL" (DOUBLE 0.756, BODY_LOCK 0.706) because
  SPRAWL's executor is the DEFENDER and the opponent in an attack trace is a
  standing defender satisfying the stand/recover bands (per-phase detail in
  the report; the commanded phase is known in real use).
- FACT: perturbation monotonicity: joint noise (sigma 0->0.4 rad, seed 0) and
  root tilt (0->0.8 rad) strictly monotonically decrease the score for all 7
  techniques; phase shift (progress Delta 0->0.5) is monotone for the five
  progressive techniques (SPRAWL excluded — cyclic montage, shift lands in the
  next equivalent sprawl window; STANCE has a single held phase).
- FACT: speed 200 us/call median (worst 223 us) over 5x300 calls — 5x under
  the 1 ms budget (one mj_kinematics = 6.8 us; ~0.14 ms Python feature
  assembly + ~55 us memberships; no dynamics/contacts/render).
- FACT: rigid transform invariance: yaw +0.7 rad and translate (0.3,-1.2) m of
  BOTH robots changes the trace mean by < 1e-6 (features are pair-relative in
  the self yaw frame).
- FACT: tests 19 new (tests/test_scorer.py); full suite `pytest tests/ -q`
  = 65 passed (was 46).

### DEFECTS FIXED in the pre-existing package (commit 676de77)
- SPRAWL predicate sets were mapped to the wrong bands (phase 1 and 5 used the
  standing template; phases 2/4 were the sprawl list) and its thresholds were
  not fitted to the montage -> SPRAWL self-scored 0.46-0.74 per phase. Fixed
  by explicit per-phase templates + per-band calibration (SPRAWL phases 1/3/5
  = sprawl windows, 2/4 = recoveries, 0 = standing entry).
- SINGLE_LEG/FINISH1 lacked the free-leg predicate (`lock_l at_least`), the
  structural difference from a double leg -> DOUBLE_LEG as SINGLE_LEG was
  0.517; adding it (plus `leg_back_s`) gives 0.464.
- All 144 threshold numbers now come from the calibration rule and are
  re-verified against the refs by test_scorer.py (no hand-edited drift).
