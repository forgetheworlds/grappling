# Sol 6.1 external design review

## Scope and evidence rules

This is a read-only review of the working tree and recorded evidence on 2026-10-08. Only this report is written. No training, builds, tests, controller probes, or rendering runs were launched. The teacher and solo harness are being edited concurrently; citations describe the inspected snapshot, not an assertion that the other agents have finished. The tracked teacher diff and the new `src/teacher/skills.py` were reviewed. Existing design defects are distinguished from patch findings.

Evidence labels:

- **[VERIFIED-CODE]**: implementation and its consuming dispatch were inspected directly.
- **[VERIFIED-ARTIFACT]**: a saved result or video was inspected; measurements were not independently rerun.
- **[RECORDED]**: a statement in a report or ledger, not a measurement reproduced by this reviewer.
- **[INFERENCE]**: a conclusion or prediction from the preceding evidence.

The target tonight is one physically simulated G1, a continuous stance/shuffle/circle/level-change/penetration/knee/trail-leg/rise/repeat drill, with no resets. A scripted sequence is allowed by the current operator clarification (`notes.md`, E7); state-responsive physical execution is still required. This does not satisfy the eventual learned two-robot match in `docs/MISSION.md`. A shorter or lower-rung clip is an honestly labelled partial result, not completion of the full drill.

## 1. What the implementation actually is

| Module | Actual implementation / interface | Demonstrated versus code-only |
|---|---|---|
| Wrestling environment | `src/wrestling/env.py`: two G1s, 72 qpos / 70 qvel / 58 position targets; 50 Hz control over ten 2 ms physics steps. Event rewards, dorsal-contact detector, OOB and exchange clocks. `_maybe_end_exchange()` dispatches the back/OOB/timeout events; `_start_exchange()` writes fresh poses and zeroes velocities. Self observations are partly world-frame; opponent blocks are self-relative. | **[VERIFIED-CODE]** match rules and resets exist. **[RECORDED]** rule demonstrations and detector calibration are documented in `wrestling_env.md`, `env_videos.md`, and `p0_fixes.md`. This is not a continuous solo drill harness, a locomotion controller, or proof of physical recovery. |
| RL | `src/rl/{trainer,rollout,ppo,net,vec,obs,privileged,checkpoint}.py`: a small actor-critic, 92 actor inputs at stack 1 and 254 critic inputs, 29 learner outputs, absolute or residual action mapping, sequential/subprocess simulation, checkpoint/resume, deterministic evaluation. `CommandScheduler` supplies a technique plus reference-clock phase, not a velocity/height/lead-leg motor command. | **[RECORDED]** throughput and training plumbing exist. E1 is the behavior result: 13.1% standing, 83.0% ground time, 48/51 wins against a collapsing scripted opponent. There is no demonstrated learned shuffle, circle, penetration, or ground-to-stand skill. A BC warm-start hook is not an imitation dataset or trained BC controller. |
| RL curriculum / reward | `src/rl/curriculum.py`: A-C now consume scorer similarity plus a `stood` boolean; D consumes wins; E uses a frozen policy opponent. The producer-to-consumer path is `StageReward.shaping(sample=)` → `RolloutCollector` mean similarity → `Trainer._absorb()` → `Curriculum.on_exchange()`. Rewards remain outcome/OOB, geometry similarity, clock potential, and engagement. | **[VERIFIED-CODE]** the new sample reaches the gate; it is not silently discarded. **[RECORDED]** the all-draw-without-evidence promotion bug and cross-step ambiguity bug were fixed (P0 report; 115 tests reported). These are still wrestling-stage gates, not the held-out M1-M4 motor gates. |
| Teacher | `src/teacher/controller.py`: reference joints plus measured pose trim, capture-point-style ankle/knee/hip offsets, foot-site differential IK, height PI, torso correction, integrators/rate limiting, and a posture governor. It reads simulated state and returns position targets; it is not a learned policy. `phases.py` selects gains from reference height/vertical motion, with a simulated-height PRONE interlock. | **[VERIFIED-CODE]** a feedback balancer exists. **[VERIFIED-ARTIFACT]** saved teacher results below demonstrate failure of the required shot and recovery. The new command driver is code-only with the hand-off defects in §3. The report is still marked DRAFT and some prose numbers differ from the saved artifact. |
| Retargeting | `src/retarget/`: graph alignment/role resolution, common pair scaling, proper `(x,-z,y)` conversion, landmark IK, PCHIP interpolation, joint/base kinematic retiming, and paired references. The solve fits landmarks and a one-sided ground hinge; it does not enforce dynamics, support feasibility, contact-force execution, or a recoverable final posture. | **[VERIFIED-CODE]** geometric compilation exists. **[RECORDED]** Phase 2 had low joint error while physically toppling; no technique passed all execution gates. References are motion specifications, not successful demonstrations. |
| Scorer | `src/scorer/{features,config,spec,membership,calibration,scorer}.py`: kinematic pair-relative distances/angles and weighted fuzzy memberships; known commanded phase; phase/trace means. It owns separate MuJoCo kinematics data, not the PPO value function. | **[RECORDED]** own-reference phase conformity ≥0.971 and about 200 microseconds/call. **[VERIFIED-CODE]** no velocity, force, actual grip, support transfer, or ordered event history enters `score()`. It cannot certify physical execution or meaningful wrestling. |
| Solo drill, as designed tonight | `docs/SOLO_DRILL.md`: single-G1 task environment, motor command, pushes/fall metrics, shared policy, teacher demos/BC, later robustness, and a physical-progress scheduler. E7 adds the acceptable scripted-sequence route. | **[VERIFIED-CODE]** at inspection `src/solo/` contained only `__init__.py`, describing future modules; no runnable SoloEnv or drill scheduler was present in that snapshot. This is pending work, not a completed harness. `SkillController` currently creates a two-robot teacher, returns 58 targets, and has no stance-to-trail-leg event sequence. |

### Recorded behavior is substantially below the video target

**[VERIFIED-ARTIFACT]** `data/teacher_stats.json` was read directly. Its aggregate fields at inspection were:

| Technique | Mean per-seed pair minimum stay-up | Worst seed | Scorer mean | Important detail |
|---|---:|---:|---:|---|
| STANCE | 0.7583 | 0.6250 | 0.5472 | Final landmark maximum 1.207 m; the reference spans 2.34 s, not a 60 s hold. |
| DOUBLE_LEG | 0.5195 | 0.4483 | 0.3240 | PENETRATE score 0.2831, DRIVE 0.1497; final landmark maximum 2.310 m. |
| SINGLE_LEG | 0.2673 | 0.2286 | 0.3615 | Interpenetration 4.31 cm. |
| STAND_UP | 0.0000 | 0.0000 | 0.3237 | No executed standing recovery; saved foot-site penetration maximum 3.03 cm. |
| SNAPDOWN | 0.9556 | 0.8889 | 0.8082 | Interpenetration 5.71 cm; not the solo drill and not an all-gates pass. |

These are reference-relative frame fractions, not probabilities of a clean completed attempt. Their definition allows tilt up to `max(30 degrees, reference tilt + 15 degrees)` and excludes low-reference frames (`teacher/episode.py`, `stay_up_frac`). Do not present them as an uninterrupted balance or recovery success rate.

**[VERIFIED-ARTIFACT]** the preview of `videos/teacher/DOUBLE_LEG.mp4` is 320×240, 30 fps, h264, approximately 3.4 s, with two robots; later frames show a toppled body, not a recovered stance. It is failure evidence, not a single-G1 drill. **[RECORDED]** `teacher.md` §16 reports a falling command-driver self-check (minimum pelvis 0.088 m and 0.97 m drift), but its cadence is wrong (§3, F2), so the report has not isolated the cause it attributes to the live-start transient.

**[VERIFIED-ARTIFACT]** `data/support_envelope.json`, STANCE: all 118 frames of both robots are classified statically infeasible under the audited sole-patch model. Margins range roughly −1.25 to −2.39 mm for A and −8.63 to −9.52 mm for B. These are kinematic, foot-only results with a height-based contact assumption, not a dynamic impossibility theorem. **[INFERENCE]** replaying that start and hoping for balance wastes the clean-video critical path; constructing a verified feasible starting stance is necessary.

## 2. Biggest risk and the smallest change that removes it

**The biggest immediate risk is that the scheduler can appear to issue a full drill while the physical controller never receives its changing reference.** This is a verified hand-off failure, not a need for a larger network or more PPO.

`SkillController._build()` creates a constant reference table and passes it to `TeacherController`. The consuming `_RobotCtx` copies the table and caches joints, foot positions, support/contact flags, height, heading, and phase. Later `SkillController.control()` edits only its original table. The teacher continues executing the original stance. The trim ramp, squat, shot, knee lowering, recovery and translation edits never reach the consuming targets. A scheduler/HUD can still advance through names and timestamps, producing false apparent progress.

**Smallest coherent fix:** give the teacher a per-robot live-reference update interface for the current pose and its derived foot/height/heading/contact/phase targets, preserving integrators and previous offsets. Do not reconstruct a 900-frame controller every tick, and do not update only cached joints while the competing IK tasks remain stale. Exercise that same interface at 50 Hz with ten physics substeps. First acceptance should show that a level command changes both the executed joint target and the height/foot task before any long simulation or render.

**[INFERENCE]** this removes the dead hand-off, not the whole physical blocker. A contact-aware stepping and recovery implementation remains necessary; no evidence says that moving reference xy while both feet remain planted will supply it.

## 3. Patch findings

Each item is grounded in the inspected patch/new file. Confidence is confidence in the defect, not an estimate of video success after fixing it. The first two findings have been independently confirmed by the main agent and assigned for repair; no landed repair was inspected here.

### F1 — P1, confidence 1.00: Propagate procedural reference updates into the teacher

**Location:** `src/teacher/skills.py:131-135`; consuming code `src/teacher/controller.py:119-158` and `:260-268`.

After reset, `_build()` constructs the teacher from the initial constant table. `_RobotCtx` copies that array and precomputes the targets. Subsequent calls mutate only `SkillController._table`; the consuming teacher reads its copies. LEVEL_CHANGE, SHOT_DOUBLE_LEG, KNEE_LOWER, RECOVER_STAND and movement commands therefore never change the executed reference. Add a live-reference update interface that refreshes every dependent task while preserving feedback state. Sharing only a joint array does not fix the stale IK/height/phase tables.

### F2 — P1, confidence 1.00: Run the self-check at the 50 Hz controller cadence

**Location:** `src/teacher/skills.py:209-212`; time advancement `:124-130`; comparable correct loop in `teacher/episode.py`, `run_episode`.

The self-check calls `control()` before each single 0.002 s `mj_step`, but the driver increments its clock and commands by 0.02 s per call and the teacher assumes the same control period for feedback integration/filtering. The test runs control at 500 Hz, exhausts the 18 s table in 1.8 s of physics, and accelerates integrators, blends and rate limits tenfold. Its reported falling drill is not evidence about the documented 50 Hz path. Hold a target for ten substeps before requesting another target.

### F3 — P2, confidence 1.00: Activate the governor from unclipped capture-point error

**Location:** `src/teacher/controller.py:371-375`; producer `stabilizers.py:119-125`; limits `gains.py:49,79-91`.

The governor takes the norm of `e` after each component has been clipped to ±0.08 m. Its maximum is therefore √2×0.08 ≈0.113 m. DIVE starts its emergency blend at 0.12 m, so that governor cannot activate for any balance error. STAND, RISE and LOW also cannot reach their configured full-blend thresholds. Keep the unclipped measurement for activation and clamp only the error sent to joint corrections. This affects the emergency path intended for fast dives; it does not prove that the stable stand target can recover every already-falling posture.

### F4 — P2, confidence 1.00: Write yaw commands into the procedural pose reference

**Location:** `src/teacher/skills.py:128-135`; consuming heading `controller.py:134-138`.

With `vx=vy=0` and nonzero `wz`, the driver updates `_yaw`, but `_pose()` never reads it and the row update writes only xy. The stance quaternion remains unchanged, so the teacher receives no requested heading change, even after F1 is fixed. Write commanded heading into the reference quaternion and consistent foot-placement targets. Simply rotating the coordinates used for future translation does not turn the current robot. Shipped yaw-feedback gains are also zero (`gains.py:76-94`); a command interface is not evidence of a turning controller.

### Existing design hazards, not additional patch-regression claims

1. **Pose modulation is not locomotion. [VERIFIED-CODE]** `skills.py`, `_pose()`, adds joint offsets for squat/lead-knee and blends toward stand; it has no transfer/load/unload/swing/landing sequence, no trail-leg recovery stage, and no foot-landing guard. The shuffle and circle names do not dispatch to distinct gait implementations. `_cmd_cur` is filtered, but height, lead-step and skill changes are not cross-faded; RECOVER_STAND uses elapsed drill time rather than time since recovery entry, so after two seconds its blend immediately selects full stand. **[INFERENCE]** once F1 is fixed, this path exposes abrupt target changes and remains liable to dragging feet or toppling rather than stepping. Require a real support-transfer primitive before treating circle/shuffle names as capabilities.

2. **The solo seam still requires two robots. [VERIFIED-CODE]** `TeacherController.__init__()` constructs both `a_` and `b_`, uses hard-coded 36/72 and 29/58 slices, and `SkillController` returns 58 targets. The nominal single-robot fallback in `_rest_sole_z()` still looks up both robot prefixes. `SkillController._build()` uses `self.stance` for the B table, not `self.stance_b`, although the self-check initializes B from its own stance. A camera crop is not a single-robot simulation: a falling bystander can change contacts and compute cost. Port the feedback module to one explicit robot context and return exactly 29 targets to the single-G1 environment. The current RL wrappers also hard-code paired observations/privileged state; do not assume they plug into the solo observation contract unchanged.

3. **Current recovery data are unusable demonstrations. [RECORDED]** `teacher.md` §11 reports STAND_UP foot heights 0.13–0.72 m above the mat and pelvis about 1.0–1.08 m after montage alignment, plus failed vertical re-grounding; the saved teacher result is zero stay-up. `retarget/gmframe.py`, `yaw_align()`/`assemble_chain()`, allows vertical translation at mismatched montage junctions; a ground hinge prevents penetration, not levitation. **[INFERENCE]** BC/DAgger from these attempts teaches failure, and more samples do not repair the reference. Use a separately executed knee-to-stance recovery or rebuild the montage; do not make this broken STAND_UP a dependency of the video.

4. **Low height is not a support-mode decision. [VERIFIED-CODE]** PRONE is reference pelvis <0.42 m; the interlock only checks whether simulated pelvis is above 0.55 m. Once both heights are low, the PRONE gains switch balance/height/torso tasks off without establishing knee/hand load or a stable knee support. `stabilizers.foot_contact()` uses site height <0.045 m, not solver contact force. **[INFERENCE]** an unplanned low collapse during knee lowering can be treated like intended ground work. Keep intentional nonterminal knee motion, but gate support-mode transitions by measured contact/support and distinguish it from dorsal or uncontrolled collapse. A kinematic sole hull is not an actual loaded-support polygon.

5. **The geometry score can endorse an infeasible source. [VERIFIED-CODE / RECORDED]** `scorer/features.py` performs kinematics only; own-reference conformity ≥0.971 includes the reference families whose execution fails. The score can stay high without force transfer, step timing, or completed recovery. `teacher/episode.py`, `sprawl_end_posture()`, identifies defender_prone solely from low pelvis height, so a low supine pose also satisfies that field; the env report explicitly documents the PD sprawl settling supine. Do not use either geometry score or low pelvis as a physical success gate. For solo shots use marker-relative geometry plus actual lead-foot/knee/trail-foot/rise events; no fake opponent force or grip claims.

6. **The new execution gate does not mean stayed on feet. [VERIFIED-CODE]** `Trainer._absorb()` defines `stood` as no own dorsal trigger and no OOB forfeit. Knees/hands/sideways collapse deliberately do not trigger that rule. E1 already has 83% ground time with dorsal-contact fraction zero. Thus that measured posture profile is not excluded by `stood`; whether its scorer would clear the new similarity threshold has not been measured here. Mean similarity is pooled over an exchange, with no required ordered penetration/recovery events or per-phase minimum. D still credits opponent-collapse wins, and no attribution hook is installed. M1-M4 must use independent held-out motor gates, not these exchange proxies.

7. **Reward and reset semantics are wrong for continuous evidence. [VERIFIED-CODE]** the existing progress term is an exchange-clock potential, not physical progress (P0 report §5); `WrestlingEnv._start_exchange()` resets pose/velocity after an exchange or timeout. Reusing it unchanged can splice recovered-looking standing states into a drill. The final runner must keep one uninterrupted MjData state and never call an internal exchange reset. `reward.py`, `StageReward.shaping()`, also has the documented `score_every=1` modulo hole (P0 report §6.4): no score samples are produced in that mode. Do not silently run a supposedly scorer-gated experiment through this configuration.

## 4. Constraints the current plan underestimates

- **[RECORDED]** this is a four-core ARM CPU host, not the roughly eight cores in older brief/goal prose (`notes.md`, initial recon). Respect the recorded ≤3 simulation workers plus one trainer. Concurrent pose extraction, full-resolution software rendering, controller sweeps and PPO do not create extra compute.
- **[INFERENCE, arithmetic]** at 200–430 control transitions/s, 1M transitions cost about 0.65–1.39 hours, 10M about 6.5–13.9 hours, and 100M about 2.7–5.8 days, before additional load or changed contact complexity. A measured pair-env throughput is not a single-G1 learning-slope estimate. S2 through S7 cannot be budgeted as several quick training tasks when there is no successful locomotion teacher or learning curve.
- **[RECORDED]** `teacher.md` reports roughly 2.1 s per 320×240 rendered frame during the sweep. At that rate even a 60 s / 30 fps clip requires about 63 minutes of rendering; the required 960×720 rate has not been established here. **[INFERENCE]** rendering every failed candidate before physics acceptance burns the time needed to fix the controller. Cache a physics trajectory, inspect low-resolution diagnostic frames, and render the final accepted trajectory once. Do not claim that a 320×240 teacher failure video meets the 960×720 evidence contract.
- **[VERIFIED-CODE]** the available controller and RL environment simulate two robots, contrary to the current single-G1 milestone. Hiding B visually neither removes its collisions nor halves simulation work.
- **[INFERENCE]** full-video monocular pose extraction/retargeting can inform posture, but cannot supply contact forces, reliable depth, or automatic dynamic feasibility. Bound tonight to the required chapters and do not put complete source-video processing on the clean-video critical path.

## 5. Methodology changes and checks for the main agent

These are recommendations, not checks performed by this reviewer.

1. **Separate wire correctness from controller quality.** Once the current repair agent lands, first test skill/height/velocity/yaw commands through their real consumer. Assert changed joint AND task targets; assert 29 outputs in the solo scene; assert one controller call per 20 ms. Include turn-in-place with zero translation. A scheduler label is never the acceptance observation.
2. **Use the same runner for probes and final evidence.** One reset at initialization, then continuous physics. Keep controller integrators across phase changes. Log control/physics tick counts and every reset/state write. Initializing a test pose is allowed; teleporting the base to execute motion or complete recovery is not.
3. **Gate on completed physical sequences, not mean frame fractions.** Before rendering a full drill, require a continuous feasible stance hold; one observable loaded/unloaded foot reposition; then level change → lead penetration → controlled knee contact → trail recovery → rise; finally repeated complete cycles. Record each guard, actual contact/support, displacement, tilt, saturation, slip and timeout. A phase timeout marks failure/abandonment, not successful advancement. Specify tolerances from baselines before tuning rather than backfitting them to the chosen clip.
4. **Keep diagnostic resolution and acceptance resolution separate.** Fast cached rollouts first; requested 960×720, 30 fps, h264/yuv420p final footage and 3-frame sheet after success. Overlay actual controller kind, seed/config, physical stage, command versus realized motion, contact state, push interval, and failure/reset count. A MuJoCo-rendered robot following physics is required, not a kinematic reference render.
5. **Do not promote source quality into executed evidence.** Reference RMS, scorer self-conformity, passing plumbing tests and opponent-collapse wins stay in their own columns. Report worst-seed complete-cycle success, not just seed-averaged pair minimum frame survival. Earlier motor gates must be retested when later behaviors are learned.
6. **Freeze evidence provenance at the run.** Save config, controller/reference revision, seed, sampling rates and metrics beside each video. The current DRAFT teacher prose and live JSON disagree, e.g. STANCE 0.792 over three seeds in prose versus 0.7583 over five seeds in the saved artifact inspected here. Resolve provenance, not the discrepancy by choosing the favorable number. The old motor audit statement that no teacher report/video exists is now stale; no new absence claim should be copied from it without checking the current files.

## 6. Ordered change list

### Must do before the full drill video

1. **Repair the command hand-off and clocks (F1, F2, F4).** A coherent live-reference consumer, correct quaternion/foot heading, 50 Hz control, preserved feedback state; no 18 s fixed-table cap for a 60–90 s drill. Give each element its own entry time and blend target, not an absolute drill-time recovery blend.
2. **Run exactly one robot.** Reuse the existing per-robot feedback implementation through an explicit 29-target adapter in the single-G1 scene. Remove active bystander execution from this milestone; do not feed 58 targets or paired policy observations into SoloEnv.
3. **Establish the actual initial stance.** Start from the verified stand keyframe; transition through feasible poses to a staggered, sufficiently wide stance. Measure actual loaded support and torque headroom. Apply rear-leg-back and lateral widening in task geometry before accepting a torso-flattening compromise. A 2.34 s partial stay-up number is not this gate.
4. **Implement and prove physical support transfer.** Load one leg, unload/lift/reposition/plant the other, then alternate. Build shuffle/circle from it. Move the body through contact mechanics, not through reference xy, sliding planted feet, or state writes. A joint-position actuator can execute steps; this particular fixed-support balancer has not shown it.
5. **Prove knee/trail-leg/rise as one continuous execution.** Use a physically executed procedural recovery, not the airborne STAND_UP montage. State/contact guards must advance each element; timeouts keep a failure clip. If this fails, the full drill milestone is still incomplete.
6. **Make safety and evidence truthful.** Use an unclipped governor trigger (F3) if this governor is relied on, intentional knee-support handling, fall/dorsal distinction, physical-progress stage guards and reset count zero. Require repeated completed cycles, not survival averaged over frames.
7. **Only then render acceptance footage.** One continuous successful trajectory, requested format, readable overlays, metrics JSON, contact sheet, failure evidence and updated visual index by the owner. No cuts, hidden exchange resets, fake pushes or learned-control labels on a scripted teacher.

### Must do tonight if time remains

1. Run a small reproducible held-out push battery on the accepted stance/step/shot phases; compare the same controller with balance feedback disabled and record real force/impulse intervals.
2. Finish the single-G1 observation/action/command and fall/contact metric contract; reuse PPO mathematics/checkpoint utilities without pretending the paired vec/obs wrapper is already compatible.
3. Re-ground/rebuild only the necessary recovery/shot references and calibrate a solo geometric comparison to feasible marker-relative poses. Capture obs/action labels only from genuinely successful feedback executions.
4. Reconcile the teacher report, saved metrics and visual index; keep the failed command-driver check marked cadence-invalid until the corrected runner produces new evidence. Fix the existing score-every-one hole before any experiment uses it.

### Defer

1. The full S2–S7 learned curriculum as a prerequisite for this particular video; pursue learned feedback afterward from validated primitives and successful demonstrations.
2. Single-leg, body-lock, snapdown, active defender, resistance training, two-robot exchanges, tactics, historical self-play and final-match claims.
3. New policy heads, hierarchies, recurrent models, network growth or broad gain sweeps without an observed failure that they address.
4. Full-video pose extraction and exhaustive reference/visual sweeps; preserve the required source chapters and process additional material off the video critical path.

## Plain verdict and alternative path

**No: the full clean continuous drill is not reachable by simply executing the current plan and interface as written.** Its live commands are discarded, its cited self-check runs at the wrong rate, turning is not represented, the runnable controller is paired, and neither stepping nor standing recovery has been demonstrated. Completing the S1–S8 paperwork or spending the remaining time on PPO does not remove those physical dependencies. This is not proof that a full scripted-feedback drill is physically impossible tonight; it is a rejection of the current route as evidence-backed execution.

**Alternative:** make tonight a controller-first, one-G1 path: fix the four hand-off/safety defects; start at the verified stand keyframe; obtain a feasible wide staggered stance; add one contact-driven foot-reposition primitive; use it for shuffle/circle; add controlled lead-knee lowering, trail-leg replacement and a measured rise; concatenate with state-based guards and render only after complete repeatable cycles exist. Keep RL/BC off this critical path. If the controller cannot clear a component gate, save the highest clean verified rung and its failure evidence, explicitly name the missing component, and call the full drill incomplete. A static squat/pivot montage is not a substitute for the requested full stance-and-motion drill.


---

## 7. Late-arriving code changes and final review status

This addendum preserves the original findings rather than overwriting their evidence. During the remaining read-only inspection, a concurrent repair changed `skills.py` and the teacher reference consumer. The original locations in F1/F2/F4 are historical snapshot locations; the following is the later code state. No tests or physics rollouts were run by this reviewer, and no repair is declared behaviorally validated.

| Original finding | Later static inspection | Remaining acceptance requirement |
|---|---|---|
| F1 — P1, confidence 1.00 in original defect | **[VERIFIED-CODE]** `SkillController.control()` now calls `update_reference(rows, qpos_a=ta, qpos_b=tb)`. `controller.py:282-297` dispatches to `_RobotCtx.set_rows()` (`:152-193`), which refreshes joints, height, yaw, trim scale, feet, torso, touch/support and labels without resetting the controller integrators. The original silent drop is addressed in the inspected code. | Main agent must check the actual executed command and dependent tasks together; static repair is not proof of stance/level/shot execution. |
| F2 — P1, confidence 1.00 in original defect | **[VERIFIED-CODE]** the self-check now holds one target for `round(DT / model.opt.timestep)` physics steps; `drill_step()` documents the same cadence. The procedural clock uses elapsed time rather than a fixed increment per call. | The teacher still integrates per call at a fixed 20 ms. Use the corrected 50 Hz runner everywhere; clock synchronization alone does not permit calling the feedback layer at 500 Hz. Old self-check measurements remain invalid for 50 Hz behavior. |
| F3 — P2, confidence 1.00 | **[VERIFIED-CODE]** still present at the later `controller.py:410-414`: governor activation uses the already clipped `e`, with the same gain thresholds. | Preserve unclipped error for activation; main agent should cover large-error DIVE and full-blend reachability. |
| F4 — P2, confidence 1.00 in original defect | **[VERIFIED-CODE]** later `skills.py:254-257` encodes commanded heading with a normalized yaw-adjusted quaternion. FK refresh then receives that row. The original missing heading representation is addressed. | Nonzero commanded heading does not establish turning while stepping: yaw feedback is still disabled and there is no demonstrated support-transfer/turning behavior. |

The bystander reference now uses `stance_b` and its own trim (`skills.py:260-282`), addressing that part of the earlier existing-design critique. The two-robot requirement itself remains. A later file-name check still found only `src/solo/__init__.py` in the solo module and no landed solo/drill scripts in the inspected snapshot. These are in-flight implementation facts, not a declaration that the owner will fail to finish.

### Remaining physical risk after the wire repairs

**[INFERENCE] The dominant risk now is mistaking a functioning pose-command interface for a working support-transfer and recovery controller.** The later `_pose_at()` still changes joint offsets but has no load-transfer/swing/landing state machine or trail-leg guard; RECOVER_STAND still uses absolute drill time (`skills.py:224-253`). The fixed 900-row horizon remains (`T_TABLE=900`, control row clipping); the 60–90 s runner needs a current-target or rolling timeline interface, not a clock that clamps after 17.98 s. No corrected clean-stance or complete-cycle measurement was inspected. The saved STANCE and STAND_UP results still supply no evidence of a long hold or an executed rise.

The smallest next physical addition is **one sensor-guarded support-transfer/foot-placement primitive, tested from a feasible stance**, rather than another training stage or a list of additional skill names. It must move a foot through a swing and confirm loaded landing before advancing. A controlled knee/trail-leg/rise cycle must then be executed from that live state. Position-target actuation is not the fundamental objection; absence of this physical support management is.

### Supplemental methodology evidence

- **[VERIFIED-CODE]** `tests/test_teacher.py:258-276` accepts a mean reference-relative stay-up fraction of 0.65 and a worst seed of 0.50. This is a short regression guard, not a clean-video gate. No SkillController command-consumer regression was present in the tests inspected at that moment. The owner should add a focused regression for the formerly dropped commands rather than interpret the reported 115-test result as coverage of the drill interface.
- **[VERIFIED-CODE]** scorer calibration explicitly fits the same reference feature quantiles (`scorer/calibration.py:64-90`), and `scorer/scorer.py:75-83` averages memberships. Exact-source conformity is calibration sanity, not held-out validation; no technique-defining predicate is a mandatory veto. For the video, independent physical events must remain the gate. Later recognition calibration should include executed successes and physically invalid negative examples, not only noisy copies of schematic references.
- **[VERIFIED-CODE / INFERENCE]** teacher penetration rows are collected at 10 Hz (`teacher/episode.py:211-212`); per-tick or physics-step contact/saturation extrema should be recorded for safety gates rather than assuming that sampled maxima bound every transient.
- **[VERIFIED-CODE / INFERENCE]** `scripts/run_teacher.py`, `render_video()`, retains all rendered RGB images in `imgs` before encoding. For 60 s at 960×720×30 fps that is about 3.73 GB of raw RGB alone (90 s: 5.60 GB), excluding encoding buffers. Stream final frames to the encoder; cache the small physical trajectory, not an entire high-resolution image list. This is avoidable pressure on a no-swap CPU host, not a measured out-of-memory failure.

## Final ordered change list after the in-flight repairs

### Before-video

1. Validate the now-landed reference/cadence/heading repairs through the real consuming controller; fix the remaining clipped-error governor if relied on. Keep F1/F2/F4 as regressions, not open duplicate assignments.
2. Connect an explicit single-robot 29-target feedback interface to the solo scene, with one initial reset and no exchange-reset path; support the entire 60–90 s timeline and per-element entry/blend times.
3. Establish a physically held feasible wide staggered stance from the stand keyframe, then one loaded-support / foot-swing / landing primitive. Extend only after a real step succeeds without sliding or a fall.
4. Execute penetration → controlled knee → trail-leg replacement → rise, then repeated full cycles with measured state/contact guards. Do not use the failed airborne recovery reference or clock advancement as success.
5. Render the accepted continuous single-G1 physical trajectory once in the required format, with honest scripted-feedback labels, actual progress/contact metrics, reset/fall counts and failure evidence.

### Tonight if time remains

1. Paired baseline-versus-feedback held-out push probes and earlier-capability regression checks.
2. Finish the solo observation/command/fall/support contract; collect only successful teacher labels and make the shared-policy migration explicit.
3. Reconcile report/JSON/config/video provenance, add command-seam regressions, record unsampled contact extrema, and stream encoding instead of retaining raw frames.
4. Repair only the necessary shot/recovery references and the scorer sampling hole before an experiment uses them.

### Defer

1. Making the complete learned S2–S7 ladder a dependency of this scripted-feedback video.
2. Two-robot interaction, resistance, alternative attacks, tactics, self-play and final-match claims.
3. Larger networks or architectural splits without evidence; exhaustive reference sweeps and full-source-video processing off the video critical path.

## Final plain verdict

**The full clean drill remains unsupported and is not reachable by running the present pose-only plan unchanged.** The late code repairs remove three specific interface defects; they do not demonstrate locomotion, loaded knee/trail-leg transitions, recovery, a single-robot runner, or uninterrupted repeated cycles. No behavioral success is claimed for those repairs.

**Alternative path:** use the corrected single-G1 feedback execution path, obtain a feasible stance, add one contact-driven stepping primitive, derive shuffle/circle from it, execute a genuine knee/trail-leg/rise sequence, and concatenate only physically passed elements using state guards. Defer learning for this milestone, not feedback or physical acceptance. If a component remains blocked, preserve the highest clean verified rung and failures and state explicitly that the full drill is incomplete; do not rename a posture-modulation or planted-foot pivot clip as the requested full drill.

