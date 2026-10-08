# Reward / critic / stability audit — v5 balance (2026-10-08)

**Agent:** RewardCriticAudit · **Scope:** balance (T1) reward, critic, training stability.
**Report-only** for `src/solo/reward.py`, `src/solo/train.py`, `src/rl/**` (LitImplement owns
reward.py, SoloVec owns train.py). New artifacts: `scripts/solo_train_health.py`,
`tests/solo/test_reward_diagnostics.py`, this report.
**Checkpoint under test:** `checkpoints/solo/t1_balance_v5.pt`, `steps_done = 401 408`
(the last save; the 301 056-step monitor read the operator quotes is cited as `v5@301k`).
**Load label:** every timing below is taken with `os.getloadavg()` printed by the script; peer
load was **10-16 on 4 cores** for the whole session (several agents' test suites), i.e.
throughput 30-120 env steps/s instead of the ~300 steps/s the runs logged when idle.

**Commands (all re-runnable, hold `data/locks/sim.lock`):**

```
MUJOCO_GL=egl .venv/bin/python scripts/solo_train_health.py ranked   --ckpt checkpoints/solo/t1_balance_v5.pt --full-battery
MUJOCO_GL=egl .venv/bin/python scripts/solo_train_health.py keyframe
MUJOCO_GL=egl .venv/bin/python scripts/solo_train_health.py trainlike --ckpt checkpoints/solo/t1_balance_v5.pt --push-curriculum
MUJOCO_GL=egl .venv/bin/python scripts/solo_train_health.py health   --ckpt checkpoints/solo/t1_balance_v5.pt
```

---

## 1. Verdict up front

**Which reward version each number is about (read this first).** Three reward configurations are
scored side by side throughout:

| label | what it is | provenance |
|---|---|---|
| `shipped_balance` | `TASK_TERMS["balance"]` + default weights (alive=1, termination=−100) | HEAD == worktree (verified by diff: term set, weights and term functions identical) |
| `v5_training` | the same term set with `alive=10` — **the reward v5 actually trained on** (`config.train.alive_weight = 10.0`) | HEAD == worktree |
| `balance_lit` | the literature term set LitImplement is shipping | **worktree only, UNCOMMITTED** (not in HEAD) |

Everything measured here is against reward.py `8c0e28a8…` / env.py `908b2158…` /
`t1_balance_v5.pt` md5 `f8837a70…`.

1. **Reward misalignment (operator's claim): NOT reproduced — on the battery, under every
   weighting.** Scoring the four physically-ranked trajectories on identical 48-push episodes
   (stored traces *and* a fresh end-to-end re-run), the reward order equals the physical order
   under `shipped_balance`, `v5_training` **and** `balance_lit`. The degraded v5 policy scores
   **+585.6/episode vs the scripted hold's +1716.9** (+505.4 vs +1217.2 on the 4 s stored
   battery); the term that carries the difference is `alive` (2.4x lower per step), and *every*
   term is worse for v5 (§2.1, §2.2). The reward is not paying for the degradation.
2. **Keyframe-optimality (nominal, no-push): see §3.** This is the reward-side theorem check the
   operator asked for, on the reward version that v1-v5 ran on.
3. **The critic is the first-order defect.** Returns are un-normalised, 10x the shipped/
   literature per-step scale, and `value_loss` sits at 331-2283 (RMSE 25.7-67.6) on advantage
   signals of comparable size; the PPO update's shared global grad-clip (0.5) is decided by the
   value term, after which advantages are re-normalised to unit std — so the update direction is
   set by critic error, not by the reward (§4, §6).
4. **The stability defect is behavioural: the policy never committed.** `log_std` is −1.0155 at
   401k (init −1.0): the *behaviour* policy keeps ±0.13 rad/joint of residual noise and falls in
   ~40 steps, while the *deterministic* policy at the same weights is the stand keyframe and
   survives the full 8 s (+3997 return). v5's training returns never exceeded 791 against a
   keyframe-optimal 4000 (§4, §7).
5. **A monitor did fire too late.** The mid-run reads that caught the degradation were manual;
   `solo_train_health.py health` is the cheap, one-screen check designed for the ~100k-step
   cadence (§7). It also flags a monitor defect: the existing `solo_env_smoke.py monitor` scores
   checkpoints under **default** weights, so every stored v5 reward column is the alive=1 reward,
   not the training reward (§8 P2-7).

---

## 2. The deciding experiment: physically-ranked trajectories, one reward, identical episodes

Method (`ranked`): the frozen 48-push T1 battery (`eval.battery_pushes()`, magnitudes
4/8/12/16/20/25 N·s × 8 directions × heights 0.79/0.95/1.10 m, seed 0), 48 episodes of 4 s,
**identical seeds and push plan for every controller**; env built from the checkpoint's own
config (`action_mode=residual`, `residual_scale=0.5`), and every trace scored post-hoc under
three reward configurations by replaying the logged `RewardInputs` through
`TaskReward.step` (the reward is linear in the terms, so one physics run scores all configs):

* `shipped_balance` — `TASK_TERMS["balance"]`, default weights (alive = 1);
* `v5_training` — the same, `alive = 10` (the reward v5 was launched with);
* `balance_lit` — the literature term set LitImplement is shipping.

### 2.1 Stored traces (independent cross-check, exact same battery)

From the stored 48-episode summaries (`data/solo/metrics/*t1*monitor*`/`*t1gate*`), per-term
sums reconstructed as `mean × n` over each episode (rounding ≤ 1e-6/step):

| controller | mean_upright | fall_rate | term_rate (fall+dorsal) | com_offset_max | steps/ep | return (alive=1) | **return (alive=10)** |
|---|---|---|---|---|---|---|---|
| stand_hold | **0.849** | **0.292** | 0.750 | **0.135** | 152 | +50.3 | **+1217.2** |
| policy v5@301k | 0.583 | 0.708 | 0.708 | 0.322 | 123 | −26.2 | **+505.4** |
| zero_action (abs) | 0.192 | 0.354 | 0.896 | 0.561 | 90 | −81.7 | +80.4 |
| random_init_policy | 0.176 | 0.812 | 0.812 | 0.677 | 101 | −88.9 | +48.9 |

Per-term (weighted per episode, alive=10), v5@301k vs the scripted hold:

| term | weight | v5@301k | stand_hold | Δ (v5 − hold) |
|---|---|---|---|---|
| alive | 10.0 | +590.67 | +1296.57 | **−705.9** |
| termination | 1.0 (one-off) | −70.83 (34 falls) | −75.00 (14 falls + 22 dorsal) | +4.2 |
| flat_orientation | 0.3 | −8.78 | −3.02 | −5.8 |
| joint_limit | 0.05 | −2.50 | −1.28 | −1.2 |
| action_rate | 0.05 | −1.90 | 0.00 | −1.9 |
| torque_sat | 0.05 | −1.29 | −0.02 | −1.3 |
| **total** | | **+505.4** | **+1217.2** | **−711.8** |

**Verdict: no misalignment here.** The physical order (by uprightness: stand_hold > v5 > zero >
random) and the reward order are the *same* under both weightings. The physically worse v5
trajectory scores 712 points **lower**, and every term except the one-off termination penalty is
worse for v5 — the term that carries the difference is `alive` (−706/episode), i.e. the reward
does exactly what it says: it pays for standing upright and v5 fails to earn it.

> Note on physical rank: v5's *termination* rate (0.708) is slightly better than stand_hold's
> (0.750) only because the scripted hold is knocked onto its back (22 dorsal terminations) while
> v5 crumples and trips the fall rule later. On the two graded axes (uprightness, CoM offset)
> and on the gate's own fall_rate, stand_hold is strictly better. A binary "did it terminate"
> metric inverts under this detector — do not use it as the physical score.

### 2.2 Fresh runs (the same experiment re-executed end-to-end, `ranked --full-battery`)

Same 48-push battery, same seeds, executed by `scripts/solo_train_health.py ranked` on
2026-10-08 (756.7 s wall, load 15.1-16.9; checkpoint md5 `f8837a70…`, reward.py `8c0e28a8…`,
env.py `908b2158…`). Episode horizon here is the env default **8 s** (the training episode
length; the stored monitor battery above caps at 4 s) — the ordering is the same under both.

| rank | controller | mean_upright | fall_rate | com_max | steps/ep | shipped | **v5_training** | balance_lit |
|---|---|---|---|---|---|---|---|---|
| 1 | stand_hold | **0.849** | **0.292** | **0.113** | 202 | **+100.0** | **+1716.9** | **+241.3** |
| 1= | zero_action (residual mode) | 0.849 | 0.292 | 0.113 | 202 | +100.0 | +1716.9 | +241.3 |
| 3 | policy v5@401k | 0.572 | 0.708 | 0.180 | 185 | −28.3 | +585.6 | +11.6 |
| 4 | random_init_policy | 0.449 | 0.812 | 0.223 | 153 | −60.1 | +352.6 | −25.9 |
| 5 | zero_action (absolute mode) | 0.191 | 0.354 | 0.534 | 111 | −79.9 | +109.5 | −28.9 |

Per-term (weighted **per step**, v5_training), the terms that carry the ordering:

| controller | alive (w=10) | termination (one-off) | flat_orientation (0.3) | joint_limit (0.05) | action_rate (0.05) | torque_sat (0.05) | total/ep |
|---|---|---|---|---|---|---|---|
| policy | **+3.691** | −0.383 | −0.089 | −0.021 | −0.017 | −0.012 | +585.6 |
| stand_hold | **+8.882** | −0.371 | −0.015 | −0.008 | 0.000 | 0.000 | +1716.9 |
| zero_action (abs) | +1.897 | −0.808 | −0.094 | −0.005 | 0.000 | −0.002 | +109.5 |
| random_init | +2.996 | −0.531 | −0.101 | −0.029 | −0.017 | −0.014 | +352.6 |

**Verdict (fresh, machine-checked): NO misalignment.** The physical rank
(stand_hold > policy > random_init > zero_action_abs on mean uprightness) and the reward rank
are **identical under all three reward configurations**; the broken v5 policy scores
+585.6/episode against the scripted hold's +1716.9, i.e. 2.9x lower, with the `alive` term
(+3.69 vs +8.88 per step) carrying the entire difference. The mixed-sign penalty terms separate
nothing (|0.02-0.09|/step). Per the ticket's mechanical criterion — "does the physically worse
trajectory score higher?" — the answer is **no**.

Two side-findings from the same table:

* **`zero_action` in residual mode IS `stand_hold`** (identical to 3 decimals in every column):
  `ctrl_from_policy(0)` returns the unit 0, and `step` applies `base + scale·tanh(0) = base` =
  the `a_stand` keyframe. The stored baseline table's `zero_action` (mean_upright 0.192) is the
  *absolute-mode* row — my `zero_action_abs` row reproduces it exactly (0.191 vs 0.192), which
  cross-validates both harnesses. Any future "zero-action" probe MUST state its action mode; in
  residual mode it is not a distinct behaviour.
* Under the in-flight `balance_lit` term set (worktree only, uncommitted: not in HEAD) the same
  ordering holds and the spread is wider relative to its own scale (policy +11.6 vs hold +241.3;
  the policy's largest term is `upright` +0.411/step with a −0.112/step `airtime` drag) — the
  literature terms also do not pay for the degraded behaviour.

### 2.3 Init vs degraded (401k): which term pays for being physically worse?

Method (`trainlike`): **training distribution** — jittered reset, no push, 8 s horizon, env
weights = v5's (`alive=10`), residual mode, identical seeds 0-7 per condition, 8 episodes each
(this is the distribution the trainer actually collects). `init_policy` is a fresh
`ActorCritic` built the trainer's way (`torch.manual_seed(0)`); the v5 init was not
checkpointed, but the deterministic init reproduces the warm-start claim (startup
`max|ctrl − base| = 0.002117 rad` vs v5's reported 0.002645 rad).

| condition | steps/ep | mean_upright | fall_rate | pelvis_z | v5_training | shipped | balance_lit |
|---|---|---|---|---|---|---|---|
| init_policy (deterministic) | 400 | **1.000** | 0.000 | 0.792 | **+3997.1** | +397.2 | +760.7 |
| stand_hold (scripted) | 400 | 1.000 | 0.000 | 0.792 | +3997.2 | +397.3 | +760.3 |
| **init_policy_sampled** | 114 | 0.545 | 0.875 | 0.498 | **+310.2** | −63.1 | −38.8 |
| ckpt_policy (deterministic) | 293 | 0.442 | 0.375 | 0.370 | **+658.7** | −9.0 | +50.4 |
| ckpt_policy_sampled | 122 | 0.564 | 0.875 | 0.515 | +384.7 | −57.2 | −29.9 |
| random_init_policy | 167 | 0.442 | 0.750 | 0.407 | +363.4 | −55.8 | −19.9 |
| ckpt_policy + training push curriculum @401k | 179 | 0.599 | 0.750 | — | +590.1 | −30.4 | +8.0 |
| stand_hold + push curriculum @401k | 233 | 0.869 | 0.125 | — | **+2009.3** | +128.8 | +286.4 |

Per-term delta (ckpt − init), weighted per step, `v5_training`:

| term | Δ weighted/step | Δ raw mean/step |
|---|---|---|
| `alive` (w=10) | **−7.4689** | −0.7469 |
| `termination` (one-off) | −0.1279 | −0.1279 |
| `flat_orientation` | −0.1055 | −0.3516 |
| `action_rate` | −0.0175 | −0.3495 |
| `joint_limit` | −0.0142 | −0.2846 |
| `torque_sat` | −0.0116 | −0.2320 |
| **total/episode** | **+658.7 vs +3997.1 (init)** | |

(These are the same numbers as the health table's no-push block for the same seeds 0-7 —
two independent code paths agree.)

**Answer to "which term pays more for being physically worse": none.** Every term is strictly
worse for the degraded policy; `alive` alone accounts for −7.47 of the −9.15/step gap. The
reward does not pay for the collapse — the policy went *down* the reward landscape.

**But the decisive mechanism is visible in the sampled rows.** The deterministic init policy
(= the stand keyframe) earns +3997.1; the **sampled** init policy — which is what PPO collects
from, `log_std = −1` → σ = 0.368 in z-space → ±0.13 rad/joint of residual noise — earns
**+310.2** and falls in 87.5% of episodes. Over 401k steps the policy **improved the behaviour
distribution it samples from** (+310.2 → +384.7, and the deterministic policy improved
marginally in the pushed condition) while the **deterministic policy that every gate and
monitor evaluates degraded from +3997 to +658.7** (6x). The optimiser was never looking at the
keyframe trajectory; it was looking at its own noisy one, and the reward it maximised is not
the one the deployment policy is scored on.

---

## 3. Keyframe optimality (the nominal, no-push alignment theorem)

Method (`keyframe`): jitter **off** (the exact stand keyframe), no push, identical reset for every
condition; each condition holds a constant `ctrl = base + δ` (clipped to ctrlrange; no condition
was actually clipped) for 100 steps and is scored **post hoc** under all three reward
configurations. Grid: all 29 joints × {±0.02, ±0.05, ±0.10} rad + 24 seeded random directions at
norms {0.02, 0.05, 0.10} = **246 perturbations + the keyframe baseline** (247 conditions,
240 s wall, load 11.3). The keyframe baseline itself: **+99.48** over 100 steps under
`shipped_balance`, **+999.17** under `v5_training`, **+179.45** under `balance_lit`; at 8 s the
same keyframe is +398.0 / **+3997.1** / +733.6.

| config | beaten by | max Δ | max Δ rel. | the winning perturbation | terms that pay |
|---|---|---|---|---|---|
| `v5_training` (alive=10) | **61/246** | +0.181 | **+0.018%** | `waist_pitch` +0.05/+0.10 rad (upright 1.000) | `alive` +0.213, `joint_limit` −0.036 |
| `shipped_balance` (alive=1) | 61/246 | +0.111 | +0.112% | `right_knee` +0.05/+0.10 rad (upright 0.998) | `joint_limit` +0.260, `alive` −0.130 |
| `balance_lit` (worktree, uncommitted) | 86/246 | +7.373 | **+4.109%** | `right_ankle_pitch` (+0.05/+0.10) | `com_support` +4.00, `capture_point` +2.23, `vel_stand` +1.09 |

**Verdict on the reward v1-v5 actually ran (`v5_training`): the keyframe is a plateau, not a
misaligned optimum.** 61 of 246 single-joint perturbations "beat" it, but the best wins by
**0.018%** (0.18 points out of 999) by nudging `waist_pitch`; the shipped reward is the same
picture (0.112%, via a knee nudge that relaxes `joint_limit`). Every beating condition stays
upright (≥0.998) — no physically-worse action is preferred. A 0.02-0.1% plateau artefact cannot
explain a policy that ends 6x below the keyframe's return (§2.3): there is no reward gradient
pulling it off the stand pose; there is a flat maximum and a policy that never found it.

**Verdict on the in-flight literature set: a real, small bias.** `balance_lit` is beaten by
**+4.1%** through `com_support`/`capture_point`/`vel_stand`: at the exact stand keyframe the CoM
sits ≈4.6 cm from the support-polygon centre (`com_support` raw 0.716 of max 1.0), and 0.05-0.10
rad of *ankle pitch* moves it closer, which the lit terms pay for. This does not prefer a
physically worse pose (upright stays 1.000), but it means the keyframe is **not** a stationary
point of the literature reward as currently parameterised.
**Recommendation for LitImplement:** recentre the support/CoM target on the stand keyframe's own
CoM, then re-run the check. Acceptance: `max Δ < 1%` for every configuration.

> **PENDING RE-RUN on the merged reward.** LitImplement's terms are uncommitted at measurement
> time (`reward.py` worktree hash `8c0e28a8…`, `balance_lit`/`term_set` not present in HEAD).
> The `shipped_balance`/`v5_training` rows above measure the **shipped** balance path (verified
> identical to HEAD by diff of `TASK_TERMS["balance"]`, `RewardWeights` defaults and the term
> functions), i.e. the reward v1-v5 actually used. The `balance_lit` row measures the current
> **worktree** definitions and MUST be re-run — together with §2.3's init-vs-degraded
> decomposition — once the merge lands:
> `MUJOCO_GL=egl .venv/bin/python scripts/solo_train_health.py keyframe` and
> `… trainlike --ckpt checkpoints/solo/t1_balance_v5.pt --push-curriculum`
> (plus one `ranked --full-battery` for the ranked table). Report the old-vs-merged deltas side
> by side.

---

## 4. Term inventory and realised magnitudes (balance task)

The balance term set is `("alive", "flat_orientation", "action_rate", "torque_sat",
"joint_limit")` + the one-off `termination` penalty. Weights are from `RewardWeights`
(`alive` overridden to 10 in the v5 run). Realised magnitudes on **visited states**, from the
fresh battery run (`ranked`, v5 policy, 48 episodes × 8 s, per-step values):

| term | weight | gate/trigger | policy v5 realised | share of total | dead? |
|---|---|---|---|---|---|
| `alive` | 10.0 | `clamp01(up_z)·clamp01(pz/0.79)`, always on | **+3.691/step** (min ≈0 when down, max 10) | **+117%** | no — it *is* the reward |
| `termination` | 1.0 one-off, −100 | fall **or** dorsal verdict | −0.383/step (−70.8/episode: 34/48 falls) | −12% | fires often |
| `flat_orientation` | 0.3 | `−(1−up_z)/2`, range [−0.5, 0] | −0.089/step | −2.8% | weak |
| `joint_limit` | 0.05 | `−limit_prox`, range [−1, 0] | −0.021/step | −0.7% | near-dead |
| `action_rate` | 0.05 | `−clamp01(mean|Δa|/0.5)`, 0 on the first step | −0.017/step | −0.5% | near-dead |
| `torque_sat` | 0.05 | `−sat_frac` | −0.012/step | −0.4% | near-dead |

For the scripted hold on the same episodes: `alive` +8.882/step and the four penalty terms sum to
**−0.023/step** (0.3% of its `alive`) — for a standing policy they are numerically inert; even for
v5's degraded behaviour the four penalties sum to −0.139/step against `alive` +3.691 (3.6%).
**The balance reward is a one-term reward in practice.** Note the shares can exceed 100%: they are
term / *net* total, and the penalties subtract.

**Flagged: a term whose gating makes an undesired behaviour profitable.** `alive` is paid
unconditionally while the fall rule is silent, and the fall rule is silent **whenever a knee or
hand is in contact** (`FallDetConfig.limb_support_exempt = True`, default; the rule returns
False *before* both the hard `pelvis_z ≤ 0.22` and the soft `pelvis_z ≤ 0.35 ∧ tilt ≥ 60°`
checks). Measured consequence: v5's typical episode runs the full 8 s horizon **crumpled**
(no-push probe: mean pelvis_z 0.185 m, mean upright 0.203, `alive` raw 0.16) and never
terminates, collecting ≈ **+1.6/step = +564/episode** under `v5_training`. A posture that is
physically a collapse pays *positive reward forever*, and the only term that penalises it
(`flat_orientation`, −0.12/step measured) is capped far below the `alive` payout.
This is the P0-2 fix (§8): the collapse escape is the one place where the reward genuinely pays
for the wrong behaviour — not by mis-ranking trajectories, but by leaving the crumple
reward-positive and the fall penalty optional.

*(Full mean/std/min/max table on the checkpoint's visited states: §7 health output. Lit terms on
the same traces are in the `ranked` JSON.)*

---

## 5. Scale audit

Measured (repo, this session):

| quantity | value | source |
|---|---|---|
| per-step reward of the scripted hold under `v5_training` | **+9.9921/step** | 8 s keyframe episode, return 3996.85 |
| per-step reward of the scripted hold under `shipped_balance` | +0.9992/step | same trace, alive=1 |
| v5's realised training returns (`recent_returns`, 50 eps) | mean **334.4**, std 136.3, min 96.3, max **790.9** | checkpoint state |
| v5's `mean_return_50` over 198 iterations | median 346.0, range 290.5-389.2 | `proc://solo-t1-v5` |
| maximum possible episode return (8 s upright, alive=10) | +4000 | 10 × 400 steps |
| termination penalty / standing reward | **−100 = 10 steps of standing** (alive=10); = 100 steps under the shipped alive=1 | `TaskReward.terminal` |
| literature per-step weight range (E31: sources A+B) | 0.02-0.2 per term, Σ ≈ 0.6-1.0/step | notes.md E31 |
| literature episode returns | O(1-100) | E31 |

**Ratio: `v5_training` pays +9.99/step for standing — 10x the shipped +1.0/step and ~10-16x the
literature's entire per-term sum (~0.6-1.0/step). Its episode-return ceiling (4000 over 8 s) is
~40x the top of the literature's band. The shipped reward, by contrast, IS at the literature
scale.** The observed v5 returns (~334) sit at the literature scale only *by accident*: they are
what a collapsed policy earns (`alive ≈ 0.16` → +1.6/step, §4) minus a fall penalty — not what
an upright policy earns (+10/step → +4000 max, +3997 measured for the scripted hold over 8 s).

The `--alive-weight 10` override was applied to the *survival* term only; `termination` stayed at
its fixed −100, so the fall cliff shrank relative to survival pay from **100 steps of standing**
(shipped) to **10 steps** (v5) while the collapse-escape stayed available (P0-1/P0-2, §8).

Implications.

* **GAE / target variance:** γ=0.995, λ=0.95; the value target (`returns = advantage + value`)
  spans 0…4000 across states. Measured on a fresh 2048-step rollout of the checkpoint (§6.2):
  return std, value RMSE, advantage std and explained variance — see the table there.
* **No normalisation exists.** `rl/ppo.py` normalises *advantages* over the rollout
  (`normalize_advantage=True`) but returns are fed to `value_loss = 0.5·MSE(value, return)` raw.
  The 0.5·MSE gradient therefore scales with the reward magnitude while the policy term is
  scale-free — so the shared global grad-clip (0.5) is decided by the critic (§6.2 measured).
* **Recommended fix (P0-1, §8):** rescale the balance reward to the literature band — divide all
  weights (including `termination`) by ~10 so that standing pays ~1/step and the fall penalty
  stays ~10 steps of standing — or add running return/value normalisation in `rl/**` (SoloVec).
  Do not drop `alive` to 1 while keeping `termination=-100` without tightening the collapse
  escape: that restores the 100-step cliff but the detector still lets a crumpled robot farm the
  per-step reward (P0-2).

---

## 6. Critic diagnosis

### 6.1 From the training log (`proc://solo-t1-v5`, 198 iterations, 405 504 steps)

| stat | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| `value_loss` (0.5·MSE) | 331.4 | 712.9 | 974.9 | 1279.0 | 2283.1 |
| `policy_loss` | −0.0564 | −0.0501 | −0.0446 | −0.0406 | −0.0179 |
| `approx_kl` | 0.0236 | 0.0500 | 0.0571 | 0.0652 | 0.0866 |
| `clip_frac` | 0.131 | 0.185 | 0.215 | 0.233 | 0.353 |
| `entropy` | 11.70 | 11.84 | 11.94 | 12.03 | 12.18 |
| `mean_return_50` | 290.5 | 334.1 | 346.0 | 360.9 | 389.2 |

Trends: `mean_return_50` +3.5 per 100k steps (296 → 330 over the run: flat);
`entropy` −0.11 per 100k (12.15 → 11.70: **≈ the init value, see §5.3**);
`value_loss` −218 per 100k (1424 → 642: it tracks the *shrinking* returns, not critic skill);
`value_loss` RMSE = sqrt(2·value_loss) = **25.7-67.6**.
`approx_kl` at the last epoch (median 0.057) exceeds the KL stop threshold (1.5 × target 0.03 =
0.045) while `epochs_run` is mostly 3-4 — i.e. most updates run all epochs past the intended
trust region and clip ~21% of the samples.

### 6.2 Measured on a fresh rollout of the checkpoint (training conditions)

2048 control steps of the checkpoint's **own sampled policy** (the update sees this distribution),
push curriculum installed at `steps_done = 401408`, γ=0.995/λ=0.95 from the checkpoint
(load 6.0-15, 76 s total for the whole health command):

| quantity | value |
|---|---|
| **explained variance** vs GAE targets (`1 − Var(ret − v)/Var(ret)`) | **+0.874** |
| **explained variance** vs Monte-Carlo return-to-go (unbootstrapped) | **+0.733** |
| value RMSE vs GAE target | **39.21** (value_loss = 0.5·MSE = 768.8) |
| returns (GAE targets) | mean −1.0, std 106.0, range [−100.2, 285.8] |
| critic values | mean −12.3, std 81.9, range [−105.5, 138.0] |
| advantages | mean +11.31, std 37.55 |
| unclipped grad norms, first minibatch (n=512): value term vs policy term | **289.02 vs 7.54** (ratio 38x) |
| total grad norm vs `grad_clip` | 289.12 vs **0.5** → the clip **binds**; the value term alone exceeds it 578x |

Interpretation (measured, not inferred): the critic *does* fit the smooth part of the value
function (EV 0.73 against the honest Monte-Carlo target), so "the critic is broken" is too
strong. What is broken is the **scale relationship**:

* the critic's per-state error (RMSE 39.2) is the **same size as the entire advantage signal**
  (std 37.6). Advantages are then normalised to unit std inside `ppo_update`, so the update
  direction mixes signal and value error at O(1) ratio `[INFERENCE]`;
* the **value term's gradient is 38x the policy term's and 578x the clip threshold**, so the
  global-norm clip (0.5, shared over actor+critic) rescales the actor's gradient by
  0.5/289.12 = **0.0017** — whatever the advantage says, the actor moves ~600x less than the
  unclipped direction. At the same time the critic is *not* over-driven: its loss is also
  clipped. The practical effect is a near-frozen actor and a critic chasing noisy targets.
* returns are **not** normalised anywhere (`rl/ppo.py: value_loss = 0.5·MSE(value, return)` on
  raw returns; only `normalize_advantage` exists). A fixed critic therefore needs: (a) scaled /
  normalised targets (P0-1, P0-3), (b) separated or per-parameter gradient clipping so the
  value term cannot decide the actor's step, (c) a larger minibatch (512 → 1024) and (d)
  explained variance reported per iteration — which is exactly what the monitor prints.

### 6.3 Verdict

* **Is the critic fit for purpose?** Partially: EV = 0.73-0.87 says it predicts value better than
  a constant; but its absolute error equals the advantage scale and its gradient dominates the
  shared clip 38:1, so the PPO update is **critic-bound**: the actor is starved while the critic
  chases targets whose magnitude (up to ±4000) is set by an alive term 10x too large.
* **Does value error dominate the advantages?** By absolute size, yes: RMSE 39.2 vs advantage
  std 37.6. Since advantages are renormalised, the *scale* is hidden; the error is not.
* **What the numbers imply for a fixed critic:** reward/value scaling to O(1) per step
  (P0-1/P0-3), un-shared gradient clipping, larger minibatch, and the per-run EV gate
  (`health` reports < 0.7 → do not launch the next 100k steps without a fix).

---

## 7. Stability monitor

PENDING (health table output on v5 + trend)

---

## 8. Prioritised fixes for LitImplement (owner of `src/solo/reward.py`)

Each item carries its evidence and an acceptance check that can be run before a v6 launch.

**P0-1 — Rescale the reward so the fall cliff keeps its ratio to survival pay.**
*Evidence:* v5 ran with `--alive-weight 10` (checkpoint `config.train.alive_weight = 10.0`),
which makes standing pay **+10/step** (measured +9.9921) — 10x the shipped value and ~10x the
literature's whole per-step sum (~0.6-1.0/term-set, notes.md E31). But `termination` is a fixed
one-off weight (100.0), so under alive=10 the fall penalty is worth only **10 steps** of standing
(it is 100 steps under the shipped alive=1). Every episode the policy lets run collapsed is paid
`alive ≈ 0.16` → +1.6/step *indefinitely* (§4 inventory), while a fall costs 10 steps of upside.
*Fix (choose one, keep the ratio):* revert `alive_weight` to 1.0 and raise `termination` to
200-400; or keep alive=10 and set `termination = 1000` (100 steps of standing). Do not change one
without the other.
*Acceptance:* `solo_train_health.py health` TERM INVENTORY line shows `termination` worth
>= 50% of a full-horizon `alive` sum (i.e. a fall cannot be shrugged off after ~5 s of standing).

**P0-2 — Close the collapse escape (the fall detector / low posture).**
*Evidence:* v5's episodes average **182.5 steps** at returns of ~334 (checkpoint:
`steps_done/episode_seed = 401408/2199`), i.e. the policy spends most of every episode crumpled
(`alive` raw mean 0.16, pelvis_z 0.185 m in the no-push probe) and is *never terminated* for it;
on the 48-push battery it survives 14/48 episodes while crumpled (min upright −0.374). The
balance term set penalises that pose by only −0.12/step (`flat_orientation` weighted) against
+1.58/step of `alive`.
*Fix:* terminate (or apply a multi-hundred penalty) when `alive ≈ 0` persists, e.g. pelvis_z
below 0.5 m or `alive < 0.1` for 0.5 s; alternatively move the `low_posture` term from the
recovery set into the balance set with a weight that makes the crumple negative per step.
*Acceptance:* a scripted collapse (zero_action through the residual path, or the v5 checkpoint)
must be **negative per step** under the new balance reward: `health` shows a negative TOTAL on
the no-push episodes for a collapsing checkpoint.

**P0-3 — Bring the value targets into a learnable range (critic).**
*Evidence:* `value_loss = 0.5·MSE(value, return)` is fed **raw returns** (only advantages are
normalised, `rl/ppo.py`): median 975, max 2283 in v5's log (RMSE 25.7-67.6). Measured on a fresh
rollout: return std ≈ 112, value RMSE ≈ 39, advantage std ≈ 38.7 — the critic's error is the same
size as the entire advantage signal, and after `normalize_advantage` the policy update direction
is set by that error. The gradient probe (§6.2) shows the value term alone saturates the shared
global grad-clip.
*Fix:* P0-1 (scale ~10x down) is the cheapest in-repo fix; additionally ask SoloVec for running
return/value normalisation in `rl/**`, and raise the minibatch (512 → 1024) so the critic sees
more states per gradient.
*Acceptance:* `health --rollout-steps 4096` reports `explained_variance (GAE) >= 0.7` and
`value_rmse < 0.25 × return_std` at the ~100k-step read.

**P1-4 — Commit the behaviour: anneal the action noise.**
*Evidence:* `log_std` at 401k is **−1.0155** (init −1.0; σ=0.362 in z-space → the sampled
residual at the reset state is ±0.13 rad/joint) — the entropy never contracted, with
`entropy_coef=0.001` providing a small outward gradient and nothing opposing it. The
*deterministic* init policy is the stand keyframe and survives the full 8 s (+3997 return); the
*sampled* policy that PPO actually collects from falls in ~40 steps (return ≈ +300 = 10·40 − 100,
matching iteration-1 `mean_return_50 = 296.5`).
*Fix:* anneal `log_std` (constant −1.0 for the first ~10k steps, then a linear ramp to ≈ −2.5 by
100k, or set `entropy_coef` to 0 after the exploration phase) — a change in `rl/net|ppo` (SoloVec),
so pair it with a `--log-std-init/-final` config field. Reward-side alternative: keep the alive
term shaped so that the *marginal* value of reducing noise is visible (P0-2/P2-6).
*Acceptance:* `health` prints the behaviour σ and it is <= 0.15 by the 100k read; the
`trainlike` sampled-vs-deterministic return gap closes to <2x.

**P1-5 — Trust region.** `approx_kl` median 0.057 vs the 0.03 target (stop at 0.045) and
`clip_frac` 0.215 → the update runs past the trust region on most iterations. Lower the LR
(3e-4 → 1e-4) or reduce epochs 4 → 2 for balance; alternatively minibatches 4 → 8.
*Acceptance:* `health` (or the trainer log) shows `approx_kl < 0.045` and `clip_frac < 0.10`.

**P2-6 — Adopt the `balance_lit` shaping for the push curriculum (already in flight).**
The training distribution includes ramped pushes (2/episode, ≤12 N·s, from 20k steps; 11.4 N·s at
401k), and the scripted hold cannot survive 12 N·s at the 0.95 m training height (22/48 battery
episodes end back-to-mat) — so surviving pushes requires *recovery*, which the balance term set
does not pay for at all (alive is 90%+ of the reward; there is no posture/CoM/tracking term).
The literature terms (E31: CoM→support-polygon centre, capture-point velocity, roll/pitch
orientation, base-height) are the shaping that pays for it. Gate the merge on the keyframe check
(§3): the new term set must keep the keyframe near-maximal in the nominal case.

**P2-7 — Monitor defect (cheap, do it now).** `scripts/solo_env_smoke.py monitor` builds the env
with **default weights** and reports `reward_mean`/`reward_sum_mean` even when the checkpoint
trained under `alive_weight = 10`; every stored v5 read therefore shows the alive=1 reward
(−0.39/step), not the training reward. Read `alive_weight` from `config.train` next to
`resolve_action_mode` (the new health script already does) or drop reward columns from the
monitor.

*(Out of scope here: the T1 gate thresholds themselves — E30/T1GateCal.)*

---

## Appendix: artifacts

* `scripts/solo_train_health.py` — `selfcheck | health | ranked | keyframe | trainlike`.
* `tests/solo/test_reward_diagnostics.py` — 11 number-asserting tests (EV, decomposition,
  per-term firing, grid shape). The misalignment/keyframe experiments are deliberately not in
  pytest (report-only).
* `data/solo/metrics/solo_health/*.json` — machine-readable snapshots written by each run.

## Appendix: method (why one run can be scored under many rewards)

`TaskReward.step` is **linear in the logged terms**: `r = Σ_k w_k · f_k(inputs)` and the inputs
are pure functions of the env state. `scripts/solo_train_health.py` therefore wraps `env.reward`
with `TraceRecorder`, which deep-copies the `RewardInputs` of every control step; any
`ScoreConfig` (weights, term set, gamma, termination penalty) can then be scored *post hoc* on
the same recorded physics by replaying those inputs through a fresh `TaskReward`. Consequences:

* one physics run per condition serves all reward configurations (this is how the ranked table
  is produced with a single 48-episode battery per controller);
* the decomposition is exact by construction — `sum(weighted terms) == total` (asserted in
  `tests/solo/test_reward_diagnostics.py`), including the one-off termination penalty;
* the physics is reward-independent, so a trajectory's *physical* ranking does not depend on
  which reward it is scored with — only the reward ranking does, which is exactly the question.

The keyframe scan uses `env.reset(seed=S)` with **jitter off** for every condition, so all
conditions start from the identical state; each condition holds a constant ctrl vector
(`base + δ`, clipped to ctrlrange, clipping reported) for `steps` control steps.

The critic numbers replay the trainer's own collecting loop (`SoloTrainer.collect` semantics:
sampled actions from the checkpoint, critic values from the same net, GAE with the checkpoint's
γ/λ, push curriculum installed at the checkpoint's `steps_done`), minus the update, so the
explained variance is measured on the distribution the critic is actually fitted to.
