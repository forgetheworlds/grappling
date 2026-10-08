# T1 mid-run read — solo balance policy (`checkpoints/solo/t1_balance.pt`)

**Date:** 2026-10-08 · **Agent:** T1Midrun · **Repo:** `/home/ubuntu/grappling`
**Checkpoint evaluated:** iteration 684, `steps_done` 1 400 832 (mtime 10:52:34Z; the trainer kept
saving while this read ran — probe re-loaded it at it. 709 / 1 452 032). Trainer untouched.

## 1. Independent evaluation of the CURRENT checkpoint

**Loading path (the trainer's own):** `SoloTrainer(TrainConfig(task="balance", steps=2_000_000,
rollout_steps=2048, gamma=0.995, out=...))` then `SoloTrainer.load(ckpt)` →
`rl.checkpoint.load_checkpoint` + `apply_checkpoint(policy=self.net)` (+ RNG restore).
**Determinism:** the repo's `solo.baselines.PolicyController` with `stochastic=False` (default) —
i.e. `env.ctrl_from_unit(net.actor.deterministic_unit(obs))` = `tanh(actor.mean(obs))`, with
`net.eval()`; no sampling, no RNG in the action path.

**Protocol:** the full **48-push battery** (`eval.battery_pushes()`, seed 0: 6 magnitudes
4/8/12/16/20/25 N·s × 8 directions × 3 heights), one push per 4 s episode, seeds 0–47, through
`eval.evaluate`. Taken under `data/locks/sim.lock` (`t1_midrun_read`), released after.
**Cost:** 18 515 steps, **eval wall 125.8 s** @ 148 steps/s (load 4.9 s). A reduced 8-push battery
plus a no-push probe (54 s) was run afterwards for diagnosis. Total ≈ 3 min wall — well under budget.

| metric | gate | **policy (this ckpt)** | zero_action | StandHold | random_init |
|---|---|---|---|---|---|
| `fall_rate` | ≤ 0.05 | **0.125** ✗ | 0.354 | 0.292 | 0.812 |
| `fall_rate_heldout` | ≤ 0.10 | **0.125** ✗ | 0.875 | 0.958 | 0.875 |
| `mean_upright` | ≥ 0.95 | **0.006** ✗ | 0.192 | 0.849 | 0.176 |
| `max_recoverable_impulse_heldout` | ≥ 16 N·s | **0.0** ✗ | 0.0 | 16.0 | 0.0 |
| `time_to_stability_mean` | ≤ 1.0 s | **none** ✗ | none | 0.133 | none |
| `com_offset_max` | ≤ 0.20 m | **0.822** ✗ | 0.562 | 0.135 | 0.677 |
| `recovery_success_rate` | ≥ 0.90 | **0.0** ✗ | 0.00 | 0.25 | 0.00 |
| `steps_after_push_mean` (reported) | — | 22.94 | 0.38 | 1.31 | 2.88 |

`max_recoverable_impulse` (all magnitudes) is also **0.0** — the policy never stably recovered a
single 4 N·s push. Baselines are from `data/solo/metrics/t1_gate_baselines.json`.

**Verdict: NOT on track — not_certified; it fails all seven criteria.** Worse, it is *below the
baselines*: `mean_upright` 0.006 < zero_action 0.192, and `com_offset_max` 0.822 > random_init 0.677.

**Diagnostic probe (why).** With **no push at all**, 2 s, deterministic, seeds 0/1/2: `mean_upright`
0.131 / −0.172 / 0.157, `mean_pelvis_z` 0.49 / 0.55 / 0.45 m, `min_upright` ≈ −0.99 (torso
horizontal/inverted), reward_sum −4.2 / −15.4 / −4.2 per episode. Repeating the reduced battery with
`stochastic=True` gives the same collapse (`mean_upright` −0.043) — so this is **not** a
mean-vs-sample artifact. The policy cannot hold a jittered stand for 2 s on its own training
distribution, and the episodes mostly do **not** terminate (`fall_rate` 0.125 while
lying down): it has settled into a low, limb-supported pose that the fall detector does not
terminate (`FallDetConfig(limb_support_exempt=True)` default), so it dodges the −100 termination
penalty — the documented honest limitation in `reports/2026-10-08/solo_env.md` §4.3.

## 2. Diagnosis from the training log (`proc://solo_t1_train`, 332 parsed iterations)

- **Progress:** steps 768 000 → 1 452 032; wall 2 874 → 5 549 s; **≈ 265 steps/s** (last-100 mean
  264.6). Run started ~1 h 33 m before the read; ~1.45 M / 2 M done.
- **`mean_return_50`: plateaued (no improvement).** Segment means −79.1 / −77.1 / −74.9 / −77.1 /
  −75.4 / −76.1 / −76.4 (it 375→706); last-100 slope ≈ **−0.02 / 100 it**; range −86.4 … −68.8.
  At ~400 steps/episode this is ≈ **−0.19 reward/step** — the signature of *not alive* (an upright
  policy would be ≈ +0.9/step, ≈ +360/episode). It has been flat the whole buffer window.
- **Entropy: monotone rise, still climbing.** 21.27 (it 375) → **26.35** (it 706), slope
  **+1.03 / 100 it**, no dip anywhere; the operator's ≈12 early reading means it has ~doubled.
- `value_loss` oscillates 12 ↔ 61 every few iterations (returns are O(−77)); one
  `WARNING: Nan, Inf or huge value in QACC at DOF 0 … Time = 4.4560` (single physics blow-up).

**Hypothesis for the rising entropy.** In `rl/ppo.py` the objective is
`policy_loss + value_coef·value_loss − entropy_coef·entropy`, and the entropy term is the pre-tanh
Diagonal-Normal entropy summed over **29 joints** (`Actor.evaluate`). With the run's
**`entropy_coef = 0.01`** (TrainConfig default; the cmdline only set task/steps/rollout/gamma/out/lock)
the bonus contributes a *constant* outward gradient of `entropy_coef × 29 = 0.29` in log-std that
never vanishes; the counter-gradient is `E[Â·(z²−1)]`, which averages to ~0 when the advantage is
uninformative. And here it is uninformative by construction: the balance reward set is
`("alive", "flat_orientation", "action_rate", "torque_sat", "joint_limit")` with `alive = 1.0`
and total per-step penalties only 0.45 — i.e. a nearly constant +1/step while upright, ~−0.2/step
when collapsed, plus a one-off −100 on termination. There is **no shaping toward the gate**
(no velocity, no push, no recovery term), `normalize_advantage=True` rescales whatever tiny signal
exists to unit variance, and the collapse escapes the −100. So the entropy bonus inflates `log_std`
essentially unopposed (≈ −1.0 → ≈ −0.51 in log-std) while the policy sits in a degenerate basin.
**Single field to change FIRST: `entropy_coef: 0.01 → 0.001` (or 0.0).** Expected effect: the
outward log-std gradient drops ~10×, the sampled distribution contracts, gradient noise falls,
`mean_return_50` lifts off the −77 plateau and the policy can commit to a consistent stand. (Second,
**env-level**, non-optional change: the gate is about pushes and `TASKS["balance"]` has
`push = None` — the trainer never applies a push, so `max_recoverable_impulse_heldout` /
`recovery_success_rate` / `fall_rate_heldout` are unattainable by any balance run; even the scripted
StandHold tops out at recovery 0.25.)

## 3. Recommendation

**(b) stop early and re-run with a single named change** — the checkpoint is already degenerate on
its own training distribution, return has been flat for 330+ iterations with entropy still rising,
so the remaining ~0.55 M steps (≈ 35 min) will only produce another degenerate checkpoint. (I did
not stop the process — it is untouched per instructions; this is a recommendation to the orchestrator.)

Single named change: **`entropy_coef` 0.01 → 0.001**. Exact command delta (needs the one-line CLI
flag `--entropy-coef` mirroring the existing `--gamma`, since only `TrainConfig` carries it today):

```
MUJOCO_GL=egl .venv/bin/python -m src.solo.train --task balance --steps 2000000 \
    --rollout-steps 2048 --gamma 0.995 --entropy-coef 0.001 \
    --out checkpoints/solo/t1_balance_v2.pt --lock off
```

Then, before this task can ever pass T1, add a **push schedule at ≤ `TRAIN_MAX_IMPULSE` (12 N·s)**
to the balance task in training and hold out ≥16 N·s for the gate. Re-run `data/solo/metrics/t1_midrun_eval.py`
(it takes ≈ 2 min) at the new run's 1 M-step mark.

## Artifacts written (territory-safe)

- `data/solo/metrics/t1_midrun_policy_summary.json` — full 48-push aggregate + per-episode rows.
- `data/solo/metrics/t1_midrun_probe_summary.json` — no-push / deterministic-vs-stochastic probe.
- `data/solo/metrics/t1_midrun_eval.py`, `data/solo/metrics/t1_midrun_probe.py` — re-runnable drivers.
