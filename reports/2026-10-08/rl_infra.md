# Phase 5 — PPO / resistance-training infrastructure (deliverable 8)

**Agent:** RLTrainer · **Date:** 2026-10-08
**Deliverables:** `src/rl/` (11 modules), `scripts/train_ppo.py`, `scripts/eval_ppo.py`,
`tests/test_rl.py` (23 tests), this report.
**Verification:** `.venv/bin/python -m pytest tests/ -q` → **88 passed** (all pre-existing
files + 23 new; the new file alone runs in ~11 s). Smoke runs and the backend benchmark
below are measured on this host (4-core ARM Neoverse-N1, CPU-only, shared with other
project agents — load average 3–8.7 during measurement).

Infrastructure only: no tuning results are claimed. All stage weights/hyperparameters
below are documented skeletons, not optima.

---

## 1. What was built

```
src/rl/obs.py          actor observation: [frame_stack x 84 env obs][one-hot(7) | phase(1)]
src/rl/privileged.py   critic-only privileged state (162) + back-exposure proxy
src/rl/net.py          actor/critic MLPs (2x256 tanh), unit-action -> ctrlrange mapper
src/rl/ppo.py          PPOConfig, GAE(lambda), clipped surrogate update, LR schedule
src/rl/scripted.py     OpponentSpec (StandHold / ReferenceReplay / teacher / policy), reference base actions
src/rl/reward.py       stage reward weights, optional technique-scorer adapter (lazy, degrading)
src/rl/curriculum.py   PISTY stages A-E, perturbation schedule, advance rules, command clock
src/rl/vec.py          sequential + multiprocessing rollout backends (+ measured benchmark)
src/rl/rollout.py      rollout collector (frame stacks, command clock, GAE finalization)
src/rl/checkpoint.py   atomic checkpoints (weights/optimizer/curriculum/stats/RNG), BC warm start
src/rl/trainer.py      training loop (resume, stage transitions, periodic eval) + evaluate()
scripts/train_ppo.py   smoke/resumable training entry point (--resume)
scripts/eval_ppo.py    deterministic seeded evaluation (per-exchange table + summaries)
tests/test_rl.py       23 tests, no training
```

## 2. Observation contract (deliverable 14 support)

Per robot, `float32`:

| slice | name | content | dim |
|---|---|---|---|
| `0:84` | `env_obs_f0` | environment `obs_fn` output (see `wrestling.env.default_observation`) | 84 |
| `84:168` | `env_obs_f1` | next-older frame (only when `frame_stack >= 2`; oldest first, newest last) | 84 |
| … | … | … | … |
| `-8:-1` | `technique_onehot` | commanded technique, order `("DOUBLE_LEG","SINGLE_LEG","BODY_LOCK","SNAPDOWN","SPRAWL","STAND_UP","STANCE")`; all-zero when the command is off | 7 |
| `-1:` | `phase` | normalized command-clock progress `clip(t_exchange/duration, 0, 1)`; 0.0 without a command | 1 |

* `actor_obs_dim(1) = 92`, `actor_obs_dim(4) = 344` (frame-stack 1 and 4 both tested;
  the trainer takes `--frame-stack {1,4}`).
* Phase source: normalized progress through `data/refs/<TECHNIQUE>.npz` duration
  (the npz meta carries keyframe *counts*, not keyframe times, so the keyframe-band
  variant is realised through the scorer's `phase_at_frac` when the scorer is present —
  it maps the same fraction onto the reference phase bands and is used for scoring).
* Frame stacks are reseeded from the fresh match's first observation at episode
  boundaries (no stale frames across resets).

**Actor/critic separation** (MISSION "critic versus technique judge" and "PPO critic"):
the actor builder's only inputs are the env obs pair and a command vector; the critic
obs is `[actor obs | privileged]` behind an explicit layout. Tests:
`test_actor_obs_never_sees_privileged` (mutating opponent joints/joint-velocities
changes the privileged vector but not the actor obs; the builder signature has no
`MjData`/privileged parameter) and `test_critic_obs_is_actor_plus_privileged`.

**Privileged block** (162 = 142 state + 12 contacts + 8 back-exposure), critic input
`92 + 162 = 254`:

| slice | content |
|---|---|
| 0:36 / 36:72 | full `qpos` of robot a / b (free joint + 29 hinges, model order) |
| 72:107 / 107:142 | full `qvel` of robot a / b |
| 142:148 / 148:154 | per robot: floor-contact flags `[feet, knees, hands, torso, pelvis]` + opponent-contact count |
| 154:158 / 158:162 | per robot: `[tilt/90, pelvis_z, dorsal_contact, exposure]` |

`exposure = clip((tilt-45)/45) * clip((0.35-pelvis_z)/0.35)` — the graded
"distance to back exposure" proxy built from the calibrated `BackDetConfig`
thresholds (0 in clearly safe postures, 1 once both detector thresholds are met).

## 3. Action contract

* Actor output: diagonal Gaussian over **29** joint-position targets per robot
  (`log_std` state-independent, init −1.0); the full-match action is
  `concat(a_half, b_half)` → `(2, 29)`/`(58,)`, the env's frozen action shape.
  A shared per-robot net is used (MISSION: "try a single shared network first"), so a
  scripted opponent can occupy the other half.
* `absolute` mode: `ctrl = mid + half * tanh(z)` — tanh-scaled into `ctrlrange`
  (`mid/half` from `model.actuator_ctrlrange`), numerically clipped for exactness.
* `residual` mode: `ctrl = clip(base + residual_scale * half * tanh(z))`, where `base`
  is the commanded technique's reference joints at the current phase
  (`reference_base_ctrl`); `residual_scale` default 0.5.
* Log-probabilities include the tanh change-of-variables correction; stored unit
  actions are re-evaluated with clamped `atanh`.
* The learner's action stream is bounds-checked before the env clip in every rollout
  (`count_out_of_bounds`): the smoke runs report **0 violations**.

## 4. PPO core

`PPOConfig` defaults (single dataclass, checkpointed verbatim):
rollout 2048 steps/env × 3 envs, 4 minibatches, 4 epochs, `gamma 0.99`, `lam 0.95`,
`clip 0.2`, `value_coef 0.5`, `entropy_coef 0.01`, advantage normalization,
grad-clip 0.5, Adam `lr 3e-4` with linear anneal to 10 % (`lr_at(progress)`), optional
`target_kl 0.03` early epoch stop, `torch_threads 1` (4-core box), seed explicit.

* `compute_gae` — standard GAE(lambda) with `dones[t]` masking the bootstrap value;
  pinned by a hand-computed toy case (values `[6.389024, 6.9292, 6.36, 4.25]`) plus a
  done-masking case and a `lam=0` (TD-residual) case.
* `ppo_update` — clipped surrogate, MSE value loss, entropy bonus, minibatch epochs,
  global-norm grad clip, per-update LR; returns loss/KL/clip-fraction stats.
* Minibatch shuffling uses a per-update seeded `torch.Generator`
  (`seed*1_000_003 + update_count`), so updates are reproducible independent of
  wall-clock.

## 5. Rollout parallelization: measurement on this box

`python -m src.rl.vec` (3 envs, StandHold opponent, uniform actions in a conservative
band around the stance hold; warmup 5 steps, then 200–300 timed steps per env). Two
measurement rounds, minutes apart, on a box shared with other project agents:

| config | backend | round 1 (load ≈ 3) | round 2 (load ≈ 8.7) |
|---|---|---|---|
| 1 env | sequential | ~290 steps/s (3.45 ms/step) | ~112 steps/s (8.9 ms/step) |
| 2 envs | sequential | 309–318 steps/s | – |
| 2 envs | subproc | 263–292 steps/s | – |
| 3 envs | sequential | 172–231 steps/s | 105 steps/s (28.5 ms/batch) |
| 3 envs | **subproc** | **421–437 steps/s** | **169 steps/s (17.8 ms/batch)** |

**Verdict: subproc at 3 workers** — 1.6–2.0× sequential at 3 envs in both rounds; the
absolute numbers track the host load (4 cores, load average 3 → 8.7 during the
measurements). Sequential is marginally faster at ≤2 envs, where IPC overhead is not
amortised. `resolve_backend` therefore selects `subproc` for `n_envs >= 3` and
`sequential` otherwise; both backends are pinned to produce identical transitions for
identical seeds/actions (`test_vec_backends_agree`, `src/rl/vec.py` self-check).
End-to-end smoke runs measured 130–300 steps/s including PPO updates under the same
load (a single-env env step costs 2.8–8.9 ms depending on load).

Backend mechanics: one forked worker per env owns its `WrestlingEnv`, scripted
opponent, privileged-obs builder and the stage's event-reward weights; the parent does
policy/value inference, observation building, shaping and PPO updates. Workers ignore
`SIGINT` (the parent owns the interrupt/save path). Matches auto-reset inside the
worker with deterministic per-match seeds (`seed + match_index * 1_000_003`); the
returned obs on `done` is the reset obs, masked by `(1-done)` in GAE. Fork (not spawn)
is used: workers inherit the parent's already-compiled MuJoCo model, so startup is
~0.04 s for 3 workers (measured); the documented caveat is the usual
fork-after-threads hazard (torch runs single-threaded here).

## 6. Smoke runs (scripted opponents, no teacher)

`scripts/train_ppo.py` defaults to the acceptance smoke: stage C (dynamic drilling),
fixed `DOUBLE_LEG` command, STANCE-`ReferenceReplay` opponent, 3 subproc envs,
match clock 30 s / exchange timeout 6 s, checkpoint under `checkpoints/rl/`.

### 6.1 Stage C vs STANCE replay — 30 000 env steps

```
[train_ppo] ==== run report ====
  wall time            : 134.7 s
  env steps            : 30000 (222.7 steps/s incl. updates)
  iterations (updates) : 5
  episodes (matches)   : 18
  exchanges            : 319 (learner W/L/D 299/0/20, causes {'back': 299, 'timeout': 2, 'match_end': 18})
  mean exchange time   : 1.87 s
  OOB events           : 5
  action bounds        : 0 violations (policy stream inside ctrlrange)
  checkpoint           : checkpoints/rl/smoke_stage_c.pt
  SMOKE OK
```

(log: `data/rl/log_stage_c.log`; the same command with `--opponent stand_hold` is the
resume demo in §7.) An earlier identical run measured 268 steps/s; the spread is the
shared box, not the code. The STANCE-replay opponent topples on its own (documented
phase-2 finding) and its back hits the mat, which is why the learner "wins" exchanges
here — this is harness evidence, not a training result.

Two shorter smoke variants prove the remaining configuration paths end-to-end
(4096 env steps each, 3 subproc envs, rollout 512):

| variant | wall | steps/s | exchanges | bounds violations |
|---|---|---|---|---|
| `--frame-stack 4` (actor obs 344, critic 506) | 31.1 s | 132.0 | 39 | 0 |
| `--action-mode residual` (reference base + clipped residual) | 29.7 s | 137.9 | 42 | 0 |

checkpoints: `checkpoints/rl/smoke_fs4.pt`, `checkpoints/rl/smoke_residual.pt`.

### 6.2 Deterministic evaluation (`scripts/eval_ppo.py`)

```
[eval_ppo] checkpoint checkpoints/rl/smoke_stage_c.pt @ step 30000 | stage C ... | seed 100
  51 exchanges | causes {'back': 48, 'match_end': 3} | learner W/L/D 48/0/3
  mean exchange time 1.76 s | OOB 0
  mean learner pelvis z 0.267 m | standing 0.131 | ground 0.830
  knee/hand contact 0.413 | dorsal contact 0.000 | mean pelvis distance 1.020 m
  EVAL OK (deterministic for fixed seed)
```

The script re-runs the first episode and asserts the exchange records are identical —
same `--seed` reproduces the table exactly (mean actions, no sampling).

## 7. Resume (interrupt → resume → logs continue)

Run 1 (`--steps 30000 --rollout 1024 --envs 3 --opponent stand_hold --ckpt-every 4000`),
SIGINT sent at ~30 s (`data/rl/log_int_run1.log`):

```
[train_ppo] signal 2: stopping after the current rollout, saving checkpoint...
[trainer] summary: steps=3072, iterations=1, ..., interrupted=True, partial_rollout_discarded=True,
                    checkpoint=checkpoints/rl/smoke_int.pt
  action bounds        : 0 violations (policy stream inside ctrlrange)
[train_ppo] SMOKE OK (harness works; training quality not claimed)      # exit code 0
```

Run 2 (`--resume checkpoints/rl/smoke_int.pt --steps 30000`; the saved stage — including
the `stand_hold` opponent override — is restored from the checkpoint, iteration
numbering and step counter continue):

```
[trainer] resumed checkpoints/rl/smoke_int.pt at step 3072 (stage C, 1 updates)
[iter    2] steps    6144 | ... (logs continue)
[trainer] checkpoint @ 6144 steps -> checkpoints/rl/smoke_int.pt
[iter    3] steps    9216 | ...
...
[iter   10] steps   30000 | ...
[trainer] summary: steps=30000, iterations=10, wall_s=189.3, steps_per_s=158.5, stage=C,
                   matches=15, exchanges=249 (W/L/D 231/0/18), bounds_bad=0, interrupted=False
[train_ppo] SMOKE OK (harness works; training quality not claimed)      # exit code 0
```

The resumed run continued the same iteration numbering (2 → 10), step counter, LR
schedule, curriculum window, statistics and RNG streams; its opponent (`stand_hold`) was
restored from the checkpoint without passing `--opponent` again. Both runs exited 0 and
the final checkpoint matches the reported step count (asserted in the script).

A second demo run (`checkpoints/rl/smoke_hold.pt`, 20 000 steps) additionally exercised
the periodic-eval path (`--eval-every 8000 --eval-episodes 1`): eval summaries were
printed during training and stored in the checkpoint (`state["eval"]`), and the
`--resume` invocation restored them along with everything else.

## 8. Curriculum (PISTY stages A–E)

`StageConfig` = `{key, name, techniques, command_on, opponent, weights, perturbations,
bc_checkpoint, use_scorer, advance}`; all fields serialisable and stored in every
checkpoint (`state["stage"]`), so `--resume` and `scripts/eval_ppo.py` reproduce the
exact stage that produced the weights (including opponent overrides).

| stage | command | opponent | similarity | progress | outcome | oob | engagement | perturbations |
|---|---|---|---|---|---|---|---|---|
| A imitation | DOUBLE_LEG | `stand_hold` (static) | 0.6 | 0.4 | 1.0 | 1.0 | 0.0 | none |
| B randomized drilling | DOUBLE_LEG | `stand_hold` | 0.5 | 0.4 | 1.0 | 1.0 | 0.0 | noise 0→0.03 rad /20k steps, phase jitter 0.3 s |
| C dynamic drilling | DOUBLE_LEG | `reference_replay(STANCE, loop)` | 0.4 | 0.3 | 1.0 | 1.0 | 0.01 | noise 0.03→0.05 /30k, jitter 0.3 s |
| D resistance | DOUBLE_LEG | `reference_replay(SPRAWL, loop)` (scripted stand-in; `teacher` kind available) | 0.15 | 0.2 | 1.5 | 1.0 | 0.02 | noise 0.05, jitter 0.5 s |
| E attack vs defense | none (free) | `policy` (frozen snapshot / checkpoint) | 0.0 | 0.0 | 1.0 | 1.0 | 0.02 | noise 0.05→0.02 /20k |

Reward terms (exact semantics in `src/rl/reward.py`): `outcome` scales the env's
±1 `exchange_end` event, `oob` scales the env's −0.25 event, `technique_similarity` is
a per-second rate of the optional scorer's total in [0, 1], `progress` is
potential-based shaping `w·(γΦ' − Φ)` on the normalized command clock (Ng et al. 1999;
no optimal-policy bias), `engagement` is a per-second rate while the pelvis distance is
within 1.2 m (anti-stall, "minimal shaping" late). Stage transitions are deterministic:
`steps_in_stage ≥ min_steps ∧ window ≥ window_episodes ∧ outcome_rate ≥ threshold`
(draws count 0.5); tests cover both advancing and non-advancing streams.

Terminal semantics come from the env (back-to-mat only): knees/hands/sprawl never
terminate. Regression test through the RL wrapper: the STAND_UP ground-start pose
(measured: hand + knee floor contacts, pelvis 0.18 m) is stepped for 0.4 s — knee/hand
contacts are observed, the detector stays silent, no exchange ends, `terminated` never
becomes true (`test_knee_hand_contact_not_terminal_through_rl_wrapper`).

### Optional imports (lazy, degrading)

* **Technique scorer** (`src/scorer`, separate component per MISSION): imported inside
  `ScorerAdapter.__init__` only when a stage has `use_scorer` and
  `technique_similarity > 0`. Any import/init/scoring error disables the similarity
  term (zero), records the reason in `status()["scorer_note"]`, and never crashes
  training. On this host it loads (`"ready"`) and scores the commanded technique at the
  phase-band index of the current progress; scores are taken every 5 steps (10 Hz) and
  scaled by `score_every` so the rate stays per-second. A command whose reference
  executor is the other robot (e.g. SPRAWL for learner `a`) skips the term with a
  recorded warning instead of scoring the wrong role.
* **Teacher-BC warm start**: `StageConfig.bc_checkpoint` is an optional path; at stage
  entry the trainer calls `warm_start_from_bc(policy, path)` and stores the status
  (loaded/missing/mismatched counts) in `state["bc_status"]`. A missing file degrades
  to a status note; `strict=True` (available programmatically) raises instead.

### Interface for the BC warm start (exact contract)

* **Artifact**: `torch.save` of either a raw `state_dict`, or a dict with one of
  `"actor"`/`"policy"`/`"state_dict"`/`"model"`/`"weights"` holding a tensor dict.
  Parameter-name prefixes `module.`/`actor.`/`policy.`/`model.` are stripped
  repeatedly, so `{"model": {"actor.trunk.0.weight": ...}}` and
  `{"actor.trunk.0.weight": ...}` both load.
* **Target**: `policy.actor.state_dict()` of an `ActorCritic` built from the run's
  `PPOConfig` (`hidden=(256,256)` tanh trunk + linear mean head + `log_std`).
  Keys must match by name and shape: `trunk.0.weight/bias`, `trunk.2.weight/bias`,
  `trunk.4.weight/bias`, `log_std`. Mismatched/unknown tensors are reported and
  skipped (never partially applied); `log_std` may be absent (teacher has no policy
  noise) — it stays at `log_std_init`.
* **Input layout**: the actor's input is the §2 actor observation
  (`92` dims at `frame_stack=1`; `344` at 4). A teacher that was trained on a different
  layout must be projected to this layout (or the run must use the matching
  `--frame-stack`); the loader checks shapes and reports mismatches rather than
  guessing. The env obs block (`0:84`) is exactly `wrestling.env.default_observation`.
* **Checkpoint format** (what `--resume` reads, `FORMAT_VERSION = 1`): `config`
  (`PPOConfig`), `policy`, `optimizer`, `state` (`step_count`, `update_count`,
  `target_steps`, `curriculum` index/steps/window, `stats`, `stage`, `bc_status`,
  `eval`, `collector_rng`), `extra` (`env` kwargs, `n_envs`, `backend`,
  `learner_robot`, `seed`, obs dims), `rng` (torch/numpy/python). Writes are atomic
  (temp file + `fsync` + `os.replace`).

## 9. Checkpointing, resume, determinism

* `save_checkpoint` / `load_checkpoint` / `apply_checkpoint`; the trainer restores
  weights, optimizer, curriculum stage + rolling window, episode/exchange stats, eval
  stats, the collector's numpy RNG and the global torch/numpy/python RNG states.
* `scripts/train_ppo.py --resume PATH` continues step counts, iteration logs, LR
  schedule (`progress = step_count/target_steps`), stage and RNG streams; the saved
  stage (including `--opponent` overrides) is restored from the checkpoint.
* Interrupts: `SIGINT`/`SIGTERM` set a stop flag; the current rollout is discarded
  (`partial_rollout_discarded: true`), the checkpoint is written and the run exits 0.
* Determinism tests: two sequential vec envs with identical seeds and actions produce
  bit-identical obs/privileged/rewards; policy sampling under a fixed torch seed is
  identical; two collectors with the same seed produce identical rollouts; the eval
  table is reproduced by a twin episode.

## 10. Test inventory (`tests/test_rl.py`, 23 tests, ~11 s)

Shapes/bounds and layouts of actor and critic obs (incl. frame stacks 1 and 4);
command vector semantics; frame-stack evolution and episode reseeding; actor-never-
sees-privileged (mutation + signature); privileged layout, contacts, exposure proxy;
action scaling into `ctrlrange` (absolute + residual + saturation); reference base
actions; hand-computed GAE (three cases); PPO update + LR schedule; checkpoint
roundtrip/atomicity/overwrite; RNG roundtrip; BC warm start (prefixes, shape mismatch,
missing file, strict mode); curriculum schedule + `stage_from_dict` roundtrip; advance
rules; command-clock determinism; reward event weights and scorer degradation (plus a
real-scorer check that skips if the parallel-built package is unavailable); knee/hand
non-terminal regression through the wrapper; end-to-end seed reproducibility; backend
equivalence (sequential vs subproc); trainer BC wiring; trainer resume roundtrip;
deterministic evaluation.

## 11. Honest limits

1. **No learning-quality claim.** The smoke runs show the harness works (steps/s,
   bounds, checkpoints, resume, deterministic eval); the policies are barely trained.
   The stage weights, perturbation ramps and PPO hyperparameters are documented
   skeletons, not tuning results.
2. **Stage D has no true resistance controller yet.** `OpponentSpec("reference_replay",
   technique="SPRAWL")` is a scripted stand-in; a hips-back/sprawl resistance opponent
   is the phase-3 teacher's job. The schema already accepts
   `OpponentSpec("teacher", technique=...)` (lazily imported, fails loudly if
   requested and unusable) and `OpponentSpec("policy", checkpoint=...)` for a learned
   defender; stage E currently runs a frozen snapshot of the current policy
   (self-play co-training), with the historical-pool league left to phase 6/7.
3. **Throughput is host-dependent** and was measured on a box shared with other
   project agents (load average 3 → 8.7): 172–437 env-steps/s for 3 envs depending on
   backend and load, 130–270 steps/s end-to-end for the smoke runs including PPO
   updates. 4 cores, no GPU; the nets are ~97 k (actor) + 131 k (critic) parameters.
4. **Subproc uses fork** (fast startup because workers inherit the compiled MuJoCo
   model). Python ≥3.12 emits a `DeprecationWarning` for fork in a multi-threaded
   process; torch runs with one thread here, and both backends are interchangeable if
   a future host needs spawn.
5. **`match_clock`/`exchange_timeout` for training are shortened** (30 s / 6 s vs the
   env defaults 180 s / 20 s) to keep iterations small; the full-length match clock is
   a constructor knob and is used by `evaluate()` when the checkpoint says so.
6. **Frame stacking carries no reward/command history**: only the env-obs frames are
   stacked; the command block is always current. Recurrence (GRU) was deliberately not
   added (MISSION: try a frame stack first).
7. **The scorer's similarity term is only as good as its phase alignment**: it is
   scored at the command-clock phase, so a policy that lags the reference clock is
   penalised by the form signal; that is the intended "does it still look like the
   technique" semantics, not a claim of calibrated correlation with human judgement.
8. **Observation**: the env's world-frame self pose (`0:7`) is unnormalised. The
   actor sees absolute world coordinates for itself and self-relative values for the
   opponent (the env contract); if the eventual free-wrestling policy needs
   invariance, that is a deliverable-14 ablation, not part of this infrastructure.
9. **Cross-component finding (reported to the orchestrator, not fixed here —
   `src/wrestling` is outside this ticket):** `wrestling.backdet.body_maps` resolves
   limb bodies as `f"{prefix}{name}"` → `a_knee_link`/`a_wrist_roll_link`, but the
   scene's bodies are `a_left_knee_link`/`a_left_wrist_yaw_link`/… (left/right
   infixed), so `limb_bids` is empty and `BackFeatures.limb_contact` is **always
   False**. Measured: at the STAND_UP reference's frame 6, `mj_forward` reports
   `ncon=40` with 23 floor contacts including `a_left_wrist_yaw_link` and
   `b_left_knee_link`, while `back_features(...).limb_contact` is False for both
   robots; the same holds across all seven references' PD replays. This is a
   diagnostics-only defect — `limb_contact` never gates the back-to-mat rule, so
   "knees/hands are never terminal" still holds structurally — but the flag cannot
   be used as evidence, and the calibration report's limb-contact claim cannot be
   reproduced. `src/rl/privileged.py` computes its own contact categories with the
   correct left/right names (that is what the wrapper regression test asserts), so
   the RL infrastructure is unaffected.

## 12. Files and how to run

```bash
# tests (all)
.venv/bin/python -m pytest tests/ -q                      # 88 passed

# module self-checks (all print "<module> self-check OK")
for m in obs privileged net ppo scripted reward curriculum rollout checkpoint trainer; do
  .venv/bin/python -m src.rl.$m
done

# backend benchmark (prints the table in §5)
.venv/bin/python -m src.rl.vec

# smoke training (stage C, STANCE opponent, 30k steps, ~2 min)
.venv/bin/python scripts/train_ppo.py

# smoke vs a static opponent / frame stack 4 / residual actions
.venv/bin/python scripts/train_ppo.py --opponent stand_hold --out checkpoints/rl/smoke_hold.pt
.venv/bin/python scripts/train_ppo.py --frame-stack 4 --steps 4096 --rollout 512
.venv/bin/python scripts/train_ppo.py --action-mode residual --steps 4096 --rollout 512

# interrupt (Ctrl-C) then resume
.venv/bin/python scripts/train_ppo.py --resume checkpoints/rl/smoke_hold.pt --steps 20000

# deterministic evaluation
.venv/bin/python scripts/eval_ppo.py --checkpoint checkpoints/rl/smoke_stage_c.pt --episodes 3
```
