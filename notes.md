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

### E4 (2026-10-08) — env rule videos (EnvVideo) + prior-art review (PriorArt)
- EnvVideo: all five rule demonstrations rendered and verified (h264 960x720, decode-clean):
  draw-on-timeout, back-contact -> 0.30 s persistence -> trigger -> score -> reset, knees/hands
  with dorsal=0 -> exchange CONTINUES, simultaneous back events -> ambiguous (same control step),
  3 boundary events -> forfeit. Also visible: the P0 ambiguity hold (trigger 1.70 s -> end 1.80 s).
- FINDING (affects solo env design): the two-robot env's default random-STANCE standing reset is
  NOT balanceable by the position-servo G1 — every exchange after the first collapses within ~2 s;
  the video agent had to substitute the verified-stable `stand` keyframe to produce honest rule
  demos. DECISION: solo-env resets come from the `stand` keyframe with verified-holdable noise.
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
