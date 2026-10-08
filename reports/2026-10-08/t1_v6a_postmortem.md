# T1 v6a post-mortem — the late brace, and why a fall was nearly free

Status: **v6a is NOT certified.** This report records the measured facts, the
mechanism they imply, the record-keeping defects they exposed, and the single
lever v6b changes. It is written from artifacts on disk, not from memory:
`data/solo/metrics/t1_v2_monitor_{501760,600064,1501184}.json`,
`data/solo/metrics/balance_t1_monitor_*_s0.jsonl`,
`data/solo/metrics/solo_health/solo_health_1501184.json`, and the checkpoint
itself (`steps_done = 1,501,184`).

## 1. The verdict (the T1 gate, as committed in `eval.py`)

`scripts/solo_env_smoke.py monitor` calls `solo.eval.evaluate(..., gate=GATES["balance"])`
— it *is* the gate harness, so its verdict is authoritative:

| step | fall (bar ≤0.05) | fall held-out (≤0.10) | upright (≥0.84) | t_stab (≤1.0 s) | com_max (≤0.20 m) | recovery | ends valid |
|---|---|---|---|---|---|---|---|
| 100,352 | 0.458 | — | 0.652 | — | 0.161 | — | — |
| 301,056 | 0.708 | — | 0.583 | — | 0.322 | — | — |
| 501,760 | 0.167 | 0.188 | 0.856 | 0.240 | 0.141 | 0.542 | — |
| **600,064** | **0.083** | **0.125** | 0.833 | 0.336 | 0.139 | 0.583 | 0.417 |
| **1,501,184** | **0.375** | **0.438** | 0.874 | **None** | 0.164 | **0.0** | **0.0** |

The final read fails four criteria (fall, held-out fall, time-to-stability,
`survivor_valid_stance_rate` 0.0 vs the bar 1.0) and passes two (upright, com).

**The run did not converge — it peaked at 600k and then degraded.** The 600k
state was the best of the run, and it is gone: the trainer's `save_every` wrote
every snapshot to the same output path, so the only surviving artifact is the
degraded final state. (Fixed in `0b1bbef`: periodic saves now also write a
step-stamped snapshot beside the rolling "latest".)

## 2. Mechanism (from the per-episode traces, not inferred)

`balance_t1_monitor_<steps>_s0.jsonl` — the battery is 24 episodes ordered
4 N·s (ep 0–7), 8 N·s (8–15), 12 N·s (16–23), one push each at t≈1 s:

| | 600,064 | 1,501,184 |
|---|---|---|
| episodes that terminated early (fell) | ep16, ep17 (both 12 N·s) | ep7 (4 N·s!), ep8, ep9, ep15, ep16, ep17, ep20, ep23 |
| episodes ending in a valid stance | 10/24 | 0/24 |
| mean pelvis height | 0.668 m | 0.712 m |
| `limit_prox_max` | 0.849 | 1.000 |
| `recovery_success_rate` | 0.583 | 0.0 |

So the late policy is **not** a policy that fell apart randomly: it stands
taller (upright 0.874, the best of the run) and braces — saturated against
joint limits — and it has **no recovery response at all**, failing even a 4 N·s
push that the 600k policy shrugged off.

### Why the gradient bought posture instead of recovery

The lit set's per-step maximum is 1.97 (measured: the certified keyframe scores
1.9654/step, and 0/246 perturbations beat it). At γ = 0.995 the discounted
value of *standing for the rest of an episode* is

$$\sum_{k\ge 0}\gamma^k \cdot 1.97 \approx \frac{1.97}{1-0.995} \approx 394
\quad\text{(bounded by the episode: } \approx 340 \text{ for the remaining 350
steps at } t \approx 1\,\text{s)}.$$

The fall penalty in v6a was **400** — i.e. almost exactly the value of standing
the rest of the episode. A fall at t ≈ 1 s was therefore close to *free*:
`-400·γ^50 ≈ -311` against `+340` of standing value. The policy was nearly
indifferent to falling, so the shaping terms (upright, base_height, com_support,
capture_point, …) dominated the decision — and a stiff, tall brace maximises
them while a recovery step temporarily gives them away. At 600k the policy had
not yet finished trading recovery for posture; by 1.5M it had.

The push curriculum is **not** the trigger: `PushCurriculum.progress` saturates
at `start_steps + warmup_steps` = 420k, before the 600k peak. What changed
between 600k and 1.5M is the policy, not the task.

### The critic did not cause this, but it is not healthy either

From `solo_health_1501184.json`: explained_variance **+0.893** (GAE target,
bar ≥0.7 → PASS), value_rmse **15.52** against return std **46.4** = 0.334 (the
0.25 aspiration is NOT met), advantage std 15.18 — i.e. the advantage is almost
entirely value error, not reward signal. Grad norms on one minibatch:
value-term **354** vs policy-term **19.95**, with the split clips active — the
critic's raw gradient is still ~18× the actor's.

## 3. What v6b changes (one lever, pre-registered)

v6b = v6a's exact command with `--lit-weight termination=1500` (3.8× v6a's 400,
≈4× the maximum standing value), so that a fall is strictly dominated by any
recovery trajectory. Everything else is byte-identical (lr 1e-4, minibatches 2,
γ 0.995, σ anneal to −2.5 at 100k, push curriculum on, residual action mode,
1.5M steps).

Pre-registered reads and falsifier:

* **400k snapshot**: `recovery_success_rate > 0` and fall < 0.375.
  *Falsifier*: if recovery stays 0, the brace is not a fall-penalty artefact and
  the next lever is elsewhere (the strongest candidate: running return/value
  normalisation, P0-3, deferred — the critic's RMSE is already 0.334×std, and a
  3.75× larger return spread will make that worse).
* **800k / 1.5M**: the full gate. Snapshot selection is now possible (the
  snapshots exist), so a late regression no longer destroys the best state.
* **Keyframe-optimality re-run** after the weight change: the termination term
  never fires in a no-push scan, so the certified-stance acceptance must be
  unchanged (number preservation, standing rule 4).

## 4. Record-keeping defects this exposed (all fixed in `0b1bbef`)

1. **No snapshots.** Every periodic save overwrote one path; the run's best
   state (600k) was unrecoverable and the degradation could not be diagnosed
   against it after the fact.
2. **Monitor rows collided across runs.** Rows were named
   `t1_v2_monitor_<steps>.json`, so v5's and v6a's reads at the same step count
   shadowed each other; the health trend table silently mixed them (its
   1.1M/2.0M rows are v5's, with `mode=None`). Rows now carry the checkpoint's
   stem (`run`/`checkpoint` fields) and `health_trend(ckpt)` filters to the
   checkpoint's own run, reporting how many unattributable legacy rows it
   dropped.

## 5. Open items

* v6b is running (`solo-t1-v6b`); its 400k read decides the falsifier above.
* The captured-demo leg (the operator's stepping-reference experiment) is
  independent of T1: the CEM capture is in flight (its own dry-run estimate is
  15,833 s ≈ 4.4 h), and the refinement can now consume a captured episode
  directly (`reference.npz`, commit `5a0c75f`).
* `movement_lit` (commit `3b0fcfa`) is the T1-movement reward set: the T2 gate's
  `yaw_err_abs_mean` is unreachable with `balance_lit` because `vel_stand` tracks
  only the linear command.
