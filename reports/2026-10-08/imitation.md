# Imitation: BC pose prior from the retargeted references + the refinement reward module

**Agent:** ImitationBC · **Date:** 2026-10-08 · **Repo:** `/home/ubuntu/grappling`
**Deliverables:** `src/solo/bc.py` (dataset + BC fit + baselines + open-loop rollout),
`src/solo/imitation.py` (DeepMimic-style terms, two families + deviation predicate),
`scripts/solo_bc_train.py`, `tests/solo/test_bc.py`, `data/solo/bc/{dataset.npz,bc_policy.pt,bc_metrics.json}`,
this report. **Status:** `pytest tests/solo/test_bc.py -q` -> **{TESTS} passed**.
No RL training was launched (a `solo-t1-v5` service owns the box).

---

## 1. Corpus (measured from `data/refs_video/retarget_summary.json` + the npz files)

The 12 retargeted single-G1 tracks derived from the **operator's own reference video** are the
primary corpus (Main's correction; not only the GrappleMap techniques). The two GrappleMap
techniques shipped in the same format are included for comparison. Pair counts below are
`len(qpos_a) - 1` (the last frame has no lookahead target):

| reference | source | window | duration | frames | pairs | train | val |
|---|---|---|---|---|---|---|---|
| `stance_hold` | video | 30.4-34.8 s | 4.40 s | 221 | 220 | 154 | 66 |
| `stance_widen_step` | video | 27.6-30.4 s | 3.60 s | 181 | 180 | 125 | 55 |
| `level_change_full` | video | 259.7-261.2 s | 2.08 s | 105 | 104 | 72 | 32 |
| `level_change_fast` | video | 256.9-258.0 s | 1.34 s | 68 | 67 | 46 | 21 |
| `shot_entry_full` | video | 422.8-431.0 s | 8.20 s | 411 | 410 | 287 | 123 |
| `shot_recover` | video | 432.5-437.0 s | 7.96 s | 399 | 398 | 278 | 120 |
| `knee_sprawl_entry` | video | 683.4-686.2 s | 3.36 s | 169 | 168 | 117 | 51 |
| `knee_sprawl_entry2` | video | 769.5-772.5 s | 6.74 s | 338 | 337 | 235 | 102 |
| `knee_sprawl_hold` | video | 685.8-690.2 s | 4.40 s | 221 | 220 | 154 | 66 |
| `knee_sprawl_recover` | video | 779.8-782.2 s | 5.54 s | 278 | 277 | 193 | 84 |
| `stalk_shuffle` | video | 122.0-131.6 s | 16.82 s | 842 | 841 | 588 | 253 |
| `circle_step` | video | 178.0-190.0 s | 17.36 s | 869 | 868 | 607 | 261 |
| `DOUBLE_LEG` | GrappleMap | — | 3.08 s | 155 | 154 | 107 | 47 |
| `STANCE` | GrappleMap | — | 2.34 s | 118 | 117 | 81 | 36 |
| **total** | 12 video + 2 GM | — | 82.0 s (video) | 4375 | 4361 | **3044** | **1317** |

**Calibration correction.** The corpus is **4102 video frames (82.0 s @ 50 Hz) + 273 GrappleMap
frames**, i.e. **4361 pairs** — not "~2.4k frames of 3.6-4.4 s each". Durations span **1.34 s
(`level_change_fast`) to 17.36 s (`circle_step`)**, and the longest tracks (`stalk_shuffle` 16.82 s,
`circle_step` 17.36 s, `shot_entry_full` 8.20 s) are *chains of repeated in-place gestures*, not
single 4 s clips. All tracks are near-in-place at the pelvis (`net_travel_m` 0.086-0.696 m;
11 of 14 below 0.30 m), each an **isolated single skill with no inter-skill transitions**.
Consequence, stated up front: **this BC fit is a narrow pose prior, not a drill policy.** A
continuous drill needs transitions the corpus does not contain; that remains S8's problem.

## 2. Mapping decision (exact) and its rationale

**Action = absolute joint-position targets, one frame ahead.**

```
ctrl_k = clip(q_ref[k+1, 7:36], ctrlrange_lo, ctrlrange_hi)     # rad, the PD target
u_k    = (ctrl_k - mid) / half                                  # the unit action the net emits
mid    = (lo+hi)/2,  half = (hi-lo)/2                           # per-actuator ctrlrange
```

* The env holds the action for the whole 20 ms control step, so the target a tracking PD
  controller should converge to by the end of the interval is `q_ref(t+dt)` — the one-step
  lookahead, not the same frame. The same-frame map would be a near-identity copy of
  `joint_pos_rel` (degenerate; the action would carry no reference dynamics).
* **Absolute, not residual.** The residual contract is `base + residual_scale*tanh(z)` with
  `residual_scale=0.5` rad around the stand ctrl; the references need up to **2.89 rad** of
  excursion from the stand keyframe (max binned excursion across ctrlrange is 0 entries outside
  the range, all 14 tracks: `clipped=0`), so residual mode cannot express them.
* Clipping is part of the contract; the corpus contains **0 out-of-ctrlrange entries**, so no
  dataset target is clipped in practice (the hand-snippet test exercises the clip path).

**Observation = the 96 reference-computable dims of the actor obs** (`ACTOR_LAYOUT[0:96]`):
`base_linvel_local`, `base_angvel_local`, `gravity_local`, `joint_pos_rel`, `joint_vel`,
`prev_action`. Computable from the reference alone:

| field | how (reference only) |
|---|---|
| `joint_pos_rel` | `q_ref[k,7:36] - stand_keyframe_joints` |
| `joint_vel` | `np.gradient(q_ref[:,7:36])` @50 Hz |
| `base_linvel_local` | `R(q_k)^T @ d/dt pos[k]` |
| `base_angvel_local` | `R(q_k)^T @ d/dt rv[k]`, `rv` = continuous rotation vectors (same construction as the retarget base filter) |
| `gravity_local` | `R(q_k)^T @ (0,0,-1)` |
| `prev_action` | unit action of pair `k-1` (zeros at `k=0`, matching `env._prev_action` after reset) |

Require the simulation / command channel and are therefore **excluded**: `cmd_vel`, `cmd_stance`,
`skill_onehot`, `lead_leg`, `phase` (dims 96:115). They are constant in this corpus and this
rollout, so nothing is lost — but see §6 for the phase finding.

## 3. Split (time-based, proven disjoint)

Per reference: pairs `[0, int(0.7*n))` **train**, `[int(0.7*n), n)` **val**; contiguous ranges,
asserted disjoint on `(reference, frame)` indices (`bc.split_overlap`, test
`test_time_split_has_zero_overlap_and_is_contiguous`). Table in §1; totals: **3044 train /
1317 val**.

## 4. BC fit

Model: `rl.net.Actor` (reused), hidden `(128,128)` tanh, input 96 -> output 29, `tanh` squashed.
Inputs standardised with **train-set** mean/std (stored in the checkpoint). Loss: MSE in
**radians** on the joint targets (physically meaningful per-joint scale). Adam lr 1e-3,
weight decay 1e-4 (explicit regularisation — the corpus is small), batch 128, 600 epochs,
best-val checkpoint. Seed 0 (deterministic). CPU-only, 2 torch threads, {TRAIN_TIME}.

**Overfit check (train/val curves, every 25 epochs):**

| epoch | train MSE | train MAE (rad) | val MAE (rad) |
|---|---|---|---|
{CURVE_TABLE}

**Held-out (val) joint error vs both constant baselines** (mean |pred - target| over val pairs,
rad; per-joint table in §7):

| policy | val MAE (rad) | vs BC |
|---|---|---|
| **BC policy** | **{BC_VAL}** | — |
| stand-keyframe constant (`ctrl = a_stand`) | {STAND_VAL} | x{VS_STAND} worse |
| mean-pose constant (mean of train targets) | {MEAN_VAL} | x{VS_MEAN} worse |

Margin asserted in the test: **BC <= 0.50 x stand AND <= 0.50 x mean** (measured
{VS_STAND}x / {VS_MEAN}x better). Train MAE {BC_TRAIN} rad (gap {GAP}).

## 5. Open-loop rollout (no expert correction)

`bc.rollout`: `SoloEnv(task="balance", absolute, jitter=False)` reset to the reference's own
start pose `q_ref[0]`, velocities from the one-sided finite differences of the shipped track
(`qpos_a` has no velocities); the policy sees only the env's own 96-dim observation and its
`ctrl` output is applied directly. Verdict per reference:

| reference | steps | result | fall time (cause) | joint err 1st half | last half | site err mean (m) |
|---|---|---|---|---|---|---|
{ROLLOUT_TABLE}

{FALL_NARRATIVE}

## 6. Phase / observation-ambiguity check (Main's interface question)

* **The actor phase field (offset 114) is NOT populated for this corpus**: with the constant
  `DEFAULT_COMMAND` (skill `STANCE`) it is identically **0.0** for all {NPAIR} pairs
  (`bc.corpus_phase_report`; test `test_phase_field_is_constant_for_the_corpus`). `solo.env`
  only advances a clock while the command skill is `SHOT_DOUBLE_LEG` (`_make_ctx`), and that
  clock is an **env camera clock, not a reference clock**.
* **The field IS live when populated**: `task="shot"` run for 140 steps gives `obs[114]`
  strictly non-decreasing, 61 unique values, 0.0 -> 1.0 over the 1.2 s shot clock
  (test `test_phase_field_is_live_when_the_shot_skill_is_commanded`).
* **Cost of the missing clock, measured** (`bc.ambiguity_report`): in the standardised
  (joint pose, joint velocity) space, each held-out frame's nearest *non-local* train frame
  demands a **median {NN_MED} rad / p90 {NN_P90} rad different target**, and
  **{NN_FRAC}% of held-out frames have a >0.25 rad twin** (nearest-neighbour check excludes
  +/-25 frames = 0.5 s). So pose+velocity *mostly* disambiguate the phase, but a material
  minority does not — the same observation can correspond to different reference phases with
  genuinely different targets. This is an obs-contract gap, not a data-quality problem.
* **Minimal fix (handed to the obs owner, not worked around):** drive the command channel from
  the reference (`cmd.skill = SHOT_DOUBLE_LEG` for the shot-family refs) and add a per-skill
  **reference clock** to the actor obs (or make the existing phase field reference-driven), so
  the policy can index the reference by time. Until then the BC prior's ceiling is partly set by
  this ambiguity, not by capacity.

## 7. Per-joint held-out errors (rad)

| joint | BC | stand const | mean-pose const |
|---|---|---|---|
{PER_JOINT_TABLE}

## 8. Imitation reward module (`src/solo/imitation.py`) — for the RL refinement stage

DeepMimic shape, `term_i = exp(-err_i^2 / scale_i^2)`, `total = sum_i w_i * term_i`. Each
`scale_i` is the RMS error at which the term falls to `1/e` (0.368). Exact match -> every term
1.0 -> `total` = sum of weights = 1.85.

| term | weight | error | scale (1/e) | family |
|---|---|---|---|---|
| `site` | 1.00 | class-weighted MSE over the 19 landmark sites | 0.05 m | **PRIMARY** |
| `joint_pose` | 0.25 | MSE over 29 joints | 0.35 rad | secondary |
| `joint_vel` | 0.10 | MSE over 29 joint velocities | 1.5 rad/s | secondary |
| `root` | 0.30 | 3-D pelvis position error | 0.10 m | stabiliser |
| `com` | 0.20 | 3-D whole-body CoM error | 0.08 m | stabiliser |

**Primary/secondary rationale.** The refinement objective is **site/landmark-driven**: a
two-stage video -> kinematic IK -> imitate-the-kinematic-reference pipeline injects a geometric
bias, so pinning the policy to the joint projection is the wrong target; the reference is a
dynamically infeasible kinematic projection (measured: 6 of 7 references fail naive PD
execution; `reports/2026-10-08/drill_motion.md:185-189` — not holdable at frame 0, CoM margins
-0.40..+0.07 m, topples in 0.9-1.2 s even with the CoM clamp). Optimising **site positions**
with the physics in the loop let the policy realise the same *movement* in a dynamically valid
way. The joint term stays at a smaller weight because it encodes the pose shape (and supplies
a dense, low-noise gradient), but with a 7x looser kernel.

**Site targets and weights.** Targets are FK of the shipped retargeted `qpos_a` at the 19
`SOLVED_SITES` on the *same* solo model (`a_` prefixed; the scene already attaches them) — the
already low-passed (2.5 Hz zero-phase), continuous-rotation-vector filtered and foot-contact
anchored reference, not re-derived landmarks. Per-site weights reuse the retarget class
weights (`retarget.landmarks.WEIGHT_PRESETS["default"]`):

| class | sites | weight |
|---|---|---|
| HIGH | core, neck, head, hips, knees, ankles, shoulders (11) | 4.0 |
| MED | elbows, wrists (4) | 1.5 |
| LOW | toes, heels (4) | 0.5 |

Hands are the noisiest landmarks and the ankle/toe/heel landmarks oscillate (+/-5 cm) even when
planted, so equal weighting would chase detector noise — this is the same weighting the solve
used.

**Early termination on deviation** (strictly `>`): joint mean-abs 0.60 rad, root xy 0.30 m,
pelvis drop 0.35 m, site RMS 0.12 m; `deviation_reason` returns the first violated axis
(`joint` | `root_xy` | `pelvis_drop` | `site`).

**Call shape for the wiring (sent to LitImplement):**

```python
targets = ImitationTargets.from_reference("data/refs_video/shot_entry_full.npz")
cur     = imitation.state_from_env(env)              # env.model / env.data only
terms   = imitation.imitation_reward(cur, targets.at(k), ImitationWeights())
reason  = imitation.deviation_reason(cur, targets.at(k), ImitationWeights())
# terms: {"site","joint_pose","joint_vel","root","com","total"}; each already weighted;
# add terms["total"] to the task reward.  k = targets.index_at(env.data.time).
# one-shot: terms, reason = imitation.step_terms(env, targets, k, weights)
```

Presets for ablations: `PRESETS["site_primary"]` (default), `PRESETS["joint_only"]`
(joint-space DeepMimic classic — keeps the IK bias), `PRESETS["site_only"]`.

**RL refinement command line (NOT launched in this session):**

```bash
# requires the trainer/obs-owner wiring (--task imitation, --init, --reference, --reward):
# today `solo.train` accepts --task balance|locomotion|stance|reach|shot|recovery only.
MUJOCO_GL=egl .venv/bin/python -m solo.train --task imitation \
    --init data/solo/bc/bc_policy.pt \
    --reference data/refs_video/shot_entry_full.npz \
    --reward site_primary --steps 60000000 \
    --out checkpoints/solo/imitation_shot_entry.pt
```

Budget note: DeepMimic-scale reference imitation measured on this repo's prior art is ~60M
samples (~2 days on 8 CPU cores, no GPU); on this shared 4-core box the same stage must be
scheduled as a background run, and the T1 trainer service must be stopped first.

## 9. Honest limits

* **Narrow pose prior, not a drill policy** (§1): single isolated skills, no transitions.
* Open-loop BC inherits the reference's dynamic infeasibility; falls are reported (§5), not
  hidden, and they are expected (naive PD fails similarly).
* The site term compares FK(qpos_a) targets; the residual between those and the original
  landmarks is the solve's `landmark_rms` (weighted 0.042-0.068 m across the 12 tracks).
* {LIMITS_EXTRA}
