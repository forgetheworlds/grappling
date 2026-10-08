# Solo vec backend — N subprocess env workers for PPO (equivalence, seeds, scaling)

**Agent:** SoloVec · **Date:** 2026-10-08
**Files touched:** `src/rl/vec_solo.py` (new), `src/solo/train.py` (wiring),
`tests/solo/test_vec_solo.py` (new), this report.
**Commit:** the `train.py` wiring landed inside `56e05ae` (LitImplement's slice:
they committed the working-tree file, which already contained it);
`src/rl/vec_solo.py`, `tests/solo/test_vec_solo.py` and this report are in the
vec-backend commit (`solo vec backend: --n-envs>1 subprocess env workers,
byte-exact at n_envs=1`). `VecSoloEnv` carries the `term_set`/`joint_mask`
pass-through their flags use.
**Tests:** `pytest tests/solo/ -q` → **105 passed, 1 failed, 5 errors** (95.8 s,
at commit time); every failure/error is `tests/solo/test_bc.py` on a missing data
artifact (`data/solo/bc/bc_metrics.json`, another agent's slice — unrelated to
this port). The new file alone → **12 passed** (~30 s idle-ish, ~162 s under
load 19), none of the suite failures is in `test_vec_solo.py`.
**Flag surface:** `python -m src.solo.train --help` verified — all pre-existing
flags still parse (25 at `HEAD`, 28 in the working tree before this slice), and
this slice added `--n-envs` (default **1**) and `--vec-backend` (default
`subproc`).
**Service:** `solo-t1-v5` was **not running** during any measurement (stopped by
Main). The box was nevertheless **CPU-saturated** for every measurement (see
§3) — no idle window existed at any point this session.

## 1. Design

`VecSoloEnv` (`src/rl/vec_solo.py`) owns N `SoloEnv`s behind one interface:

* **backends** — `subproc` (one fork worker per env; the parent does policy
  work while workers simulate) and `sequential` (same protocol in-process, the
  no-IPC reference used by tests).
* **exchange** — per control step the parent sends each worker its 29-dim
  **unit** action; the worker applies `env.ctrl_from_policy(...)` and
  `env.step(...)` itself, so the action-mode contract stays inside the env
  (the parent never maps actions). Workers return `(obs dict, reward,
  terminated, truncated, info)`; the parent stacks the obs dict into
  `{"actor", "privileged", "critic"}` float32 `(N, D)`.
* **resets are parent-driven** — on `terminated|truncated` the parent bumps
  that worker's episode index and sends `reset(seed, steps)`; the worker
  recomputes the push schedule with `curriculum.schedule_for(steps, seed)` and
  resets. One owner for seed bookkeeping, mirroring `SoloTrainer`'s loop.
* **seed contract** — worker `i` starts at `seed + i * SEED_STRIDE`
  (1_000_003) and advances +1 per episode. `n_envs=1` reduces *exactly* to the
  trainer's stream (`seed, seed+1, …`) — the property the equivalence test
  hinges on; the stride keeps the N streams disjoint.
* **trainer** — `--n-envs 1` (default) keeps the single-env path untouched;
  `>1` uses `collect_vec()` → `(T, N, …)` `RolloutBatch` → unchanged
  `ppo_update` over T·N samples. `steps_done`/`--steps` then count **total env
  transitions** (n=1 semantics unchanged).
* **checkpoint** — `config.env` gains `vec / n_envs / backend / worker_seeds /
  episode_indices / current_seeds / steps`; `state.vec` mirrors them; `load()`
  re-resets every worker at its saved *current* seed. Resuming a vec run from a
  single-env checkpoint, or with mismatched worker seeds, raises (loud).
* **pass-through** — `term_set` / `joint_mask` (the literature levers) are
  forwarded into every worker's `SoloEnv`; defaults `None` = today's behaviour.
* **faults** — every exchange is guarded: a dead/killed worker raises
  `RuntimeError("vec_solo worker i died …")`; worker-side command exceptions
  are sent back as errors rather than a silent pipe EOF.

## 2. Equivalence, determinism, seeds, faults

* **Byte-exact trajectory equivalence (the acceptance test):** a sha256 stream
  over `(qpos, actor/privileged/critic obs, reward, terminated, truncated, next
  episode seed)` after **every** step, both paths driven by the same unit-action
  sequence and the same reset policy. Identical for
  `subproc+residual+pushes` (600 steps), `subproc+absolute` (400),
  `sequential+residual+pushes` (600) — 3 parametrisations of
  `test_vec_path_matches_single_env_exactly`; `test_equivalence_stream_crosses_episode_boundaries`
  proves the compared stream really crosses ≥2 episode ends.
* **Trainer level:** `collect_vec()` at n_envs=1 == `collect()` byte-for-byte on
  all 7 rollout arrays (`test_collect_vec_at_n1_equals_single_collect`).
* **CLI self-check:** `python -m src.rl.vec_solo` → identical hash over a 220-step
  stream incl. episode reset (prints `OK (sha256 …)`).
* **Determinism:** n_envs=4 env-level two-run hash identical
  (`test_determinism_same_seed_same_n_envs`, episode-end count included);
  trainer-level n=2 two-run rollout hash identical
  (`test_trainer_collect_vec_deterministic_hash`).
* **Distinct seeds + reload:** `test_distinct_seeds_recorded_and_reloadable` —
  seed=21 → workers `[21, 1000024, 2000027, 3000030]`; after collect, save →
  reload into a fresh trainer → same four base seeds, same episode indices /
  current seeds, and a hands-on reset+step replay matches bit for bit.
* **Fault:** `test_dead_worker_raises_instead_of_stale_obs` — SIGKILL worker 0;
  the next exchanges raise `RuntimeError` (twice), never stale obs.
* **Obs contract:** keys/dims/dtypes equal `SoloEnv`'s
  (`test_obs_contract_matches_single_env`); action shape is validated.

**Caveat found during this work (pre-existing, not introduced by the port):**
`SoloTrainer` calls `torch.manual_seed(cfg.seed)` *after* building the net, so
two fresh runs with the same `--seed` do **not** get identical initial weights.
The determinism tests therefore pin `torch.manual_seed` before trainer
construction to isolate the vec path; the port adds no further
nondeterminism. One-line follow-up (not in this slice): move the
`torch.manual_seed` above net creation.

## 3. Measured scaling (contended — see the box state in every row)

Machine: 4-core aarch64 box, 23 GB RAM, torch 1 thread, MuJoCo CPU (`mj_step`),
balance task, 29-dim unit actions. **State during all measurements:**
`solo-t1-v5` NOT running; box **not idle** — 100% CPU-saturated by other agents'
jobs plus two 75-day spinning `bash` processes (PIDs 3697184, 3803561 at ~90-96%
each, measured instantaneously), e.g. per-run snapshots: `solo_drill_render.py`
53-59%, `data/library_audit/probe_mjx2.py` 36%, concurrent pytest jobs, load
average 12.4-19.8, `/proc/stat` busy = **100%** for every trainer run.

**(a) env-only, 600 control steps per env, non-interleaved** (via
`rl.vec_solo.benchmark(steps=600, warmup=20)`, which records the box state
around every arm):

| arm | env-steps/s | ratio vs inproc n=1 | load during |
|---|---|---|---|
| in-process n=1 (today's path) | 100.2 | 1.00 | ~12.9 |
| subproc n=1 | 90.1 | 0.90 | ~12.6 |
| subproc n=2 | 168.7 | 1.68 | ~12.4 |
| subproc n=4 | 157.2 | 1.57 | ~12.4-13.4 |
| subproc n=4, info disabled | 132.7 | 1.32 | ~14.9 |

**(b) env-only, interleaved n=1 vs n=4 rounds** (same objects, 200 control
steps per round, 4 rounds; ratios 1.48 / 2.06 / 1.99 / 2.47):
**median ratio 2.06×** at load 16.2-17.0.

**(c) trainer end-to-end** (trainer's own `steps_per_s`: includes IPC, policy,
PPO update; load 17.5-18.9, busy 100%):

| arm | steps | steps/s | ratio |
|---|---|---|---|
| single-env (n=1) | 1024 | 42.3 | 1.00 |
| vec n=2 | 2048 | 87.4 | 2.07 |
| vec n=4 | 4096 | 116.6 | 2.76 |

**(d) load-robust CPU accounting** (600 control steps/arm; `getrusage` incl.
reaped children) — separates vec overhead from ambient contention:

| arm | ms CPU per env step | vs in-process n=1 |
|---|---|---|
| in-process n=1 | 2.73 | 1.00 |
| subproc n=1 | 3.24 | 1.19 |
| subproc n=2 | 3.09 | 1.13 |
| subproc n=4 | 3.01 | 1.10 |

The backend adds **~10-19% CPU per transition** (pickle/IPC + parent work),
*decreasing* with N because the parent's per-exchange cost amortizes. The
in-process arm's CPU accounting implies ~366 env-steps/s of pure simulation
capacity (1/2.73 ms), consistent with v5's ~320 steps/s including policy+PPO —
i.e. contention deflated the *absolute* wall rates several-fold while the
ratios above keep their relative meaning.

**Verdict:** the gain is real and ~proportional to N (per-transition CPU is
~flat; no serial bottleneck seen), but **the idle-box ratio was never
measured**: the contended wall ratios (2.06-2.76) are shares of a saturated
box, not speedups. Expected idle-box wall speedup from (d): **≈1.8× (n=2) and
≈3.6× (n=4)** — an inference from CPU accounting, not a measurement. The
honest label for the wall numbers is **UNTESTED-UNDER-LOAD** (raw kept above).

## 4. Flags and the 4-env command (NOT launched)

Added to `python -m src.solo.train`: `--n-envs N` (default 1 = in-process single
env, unchanged) and `--vec-backend {subproc,sequential}` (default `subproc`).
All 25 pre-existing flags were re-checked to parse: `--task --steps
--rollout-steps --seed --gamma --lam --lr --entropy-coef --value-coef --clip
--epochs --minibatches --hidden --action-mode --residual-scale --log-every
--push-seed --torch-threads --alive-weight --push-curriculum --push-start-steps
--push-warmup-steps --out --save-every --resume --no-sigint-save --lock
--lock-wait`.

The v5 config with 4 envs (NOT launched):

```
MUJOCO_GL=egl .venv/bin/python -m src.solo.train --task balance --n-envs 4 \
    --steps 2000000 --rollout-steps 2048 --gamma 0.995 --entropy-coef 0.001 \
    --alive-weight 10 --action-mode residual \
    --out checkpoints/solo/t1_balance_vec4.pt --save-every 100000 --lock off
```

`--steps` counts **total** env transitions in vec mode (2M total ≈ 500k per
env). Resume with the same `--n-envs` and seed: worker seeds are validated and
reloaded from the checkpoint.

## 5. Overnight budget (honest)

* **n=1:** v5's live rate ≈ 320 steps/s (its premise figure; my contended runs
  cannot reproduce it) → 8 h ≈ **9.2M steps** (the "~9M" premise).
* **n=4:** expected 3.6× (from (d)) → **≈33M steps / 8 h**; the *measured*
  contended trainer ratio 2.76× gives ≈ 25M; the interleaved floor 2.06× gives
  ≈ 19M. Use **~25M (measured, contended floor) to ~33M (expected idle)** with
  the idle number unverified.
* **n=2:** expected ≈1.8× → **≈16.5M steps / 8 h**.
* Caveats: (i) idle ratio unmeasured; (ii) at n>1 the push-curriculum ramp is
  driven by total transitions, so each env sees a given ramp stage in 1/N of
  its own steps — v5's schedule was effectively per-env; (iii) PPO update cost
  per iteration grows with T·N (already inside the (c) ratios).

## 6. Biggest risk in the port

**Step/seed semantics change with n>1.** Trajectories are byte-identical to the
single-env path *only at n=1*; at n>1 `steps_done` counts total transitions and
the curriculum ramp advances N× faster per env, so any monitor/threshold that
compares a vec run's step counts against single-env runs (save cadence, mid-run
probes at fixed steps, gate scripts) will read a different meaning. Seed
streams are recorded and cross-n resumes are rejected loudly, but the *step
unit* is the thing to keep in mind when comparing vec curves to v5's.

## 7. Reproduce

```
# equivalence + benchmark self-check (prints OK + JSON with box state)
.venv/bin/python -m src.rl.vec_solo

# vec test file (12 tests)
.venv/bin/python -m pytest tests/solo/test_vec_solo.py -q
```
