# Literature balance recipe (T1) — transfer, numbers, staged v6 plan

**Agent:** LitImplement (delegated) · **Date:** 2026-10-08 · **Repo:** `/home/ubuntu/grappling`
**Code:** `src/solo/lit.py` (ceiling, joint mask, push distribution, gamma table),
`src/solo/mirror.py` (mirror maps + PPO mirror loss), `src/solo/reward.py`
(11 new terms, `LIT_BALANCE_TERMS`, `TaskReward(term_set=...)`), `src/solo/env.py`
(additive: `joint_mask`, `term_set`, the literature reward inputs),
`src/solo/train.py` (flags, **all default OFF** — v5's configuration is byte-reproducible).
**Tests:** `tests/solo/test_lit_reward.py`.
**Nothing in this file was launched by me.** v5 (`solo-t1-v5`, pid 3196830) was not touched —
the running process imported its modules before any of these edits, and it was never
signalled, restarted or reconfigured.

**Landable now, flags OFF.** Files: `src/solo/lit.py`, `src/solo/mirror.py`,
`src/solo/reward.py`, `src/solo/env.py`, `src/solo/train.py`,
`tests/solo/test_lit_reward.py`, `tests/test_solo.py` (literature inputs added to the shared
reward hand-state helper). `pytest tests/solo/test_lit_reward.py -q` → **33 passed**;
`scripts/solo_env_smoke.py smoke` (unmodified peer-owned script) still verifies
`reward_sum_5s = 248.3`, i.e. v5's reward is byte-unchanged. The frame-stack lever also
touches `src/solo/train.py` only (single-env path) and refuses loudly on `--n-envs > 1`.

---

## 0. Sources read (primary text, not summaries)

| tag | source | what we take |
|---|---|---|
| **A** | van Marum, Shrestha, Duan, Dugar, Dao, Fern, *Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking*, arXiv:2404.19173v1 (Oregon State) | same interface as ours (joint-space PD setpoints, 50 Hz policy, episodic, early termination, PPO **+ mirror loss**); their **Table I** weights; the **double-foot-contact finding**; the per-frame push scheme; the arm-position term |
| **B** | Yang, Yuan, Merkt, Komura, Vijayakumar, Li, *Learning Whole-body Motor Skills for Humanoids*, arXiv:2002.02991 | non-stepping ceiling `J = m dCOP sqrt(g/z_c)`; pushes every ~5 s from `[0.5x, 2x]` the ceiling; CoM target = **centre of the support polygon**; CoM-velocity target from the **capture point**; even L/R GRF; upper body locked |

A's Table I (verbatim weights, the ones we transfer): x,y velocity **0.15, 0.15**
(standing branch `e^{-5|v_xy - c_xy|}`), yaw orient. 0.1, **roll,pitch orient. 0.2**
(`e^{-30 qd}`), feet contact **0.1** (= **constant 1 when standing**), **base height
0.05** (`e^{-20|p_z - c_h|}`), **feet airtime 1.0** (`sum_f (t_air,f - 0.4) 1_td,f`;
`= 1` constant when standing), feet orientation 0.05, feet position 0.05, arm 0.03,
base acceleration 0.1, action difference 0.02, torque 0.02. Sum for a perfect stand
≈ **2.0/step** — that is the reward scale we inherit.

A's push scheme: *"for each frame there is a 1% chance of getting a random push …
uniformly sampled from a range of 200N to 800N, lasting a single timestep (20ms)"*
⇒ **4–16 N·s of impulse per hit** — which brackets our own derived ceiling (6.7–20.7 N·s),
i.e. A trained *across* its ceiling, and its follow-up ("Single Contact++") moved the
disturbance distribution closer to the test distribution. That is the same lesson as our
train/test magnitude mismatch.

A's own words for the finding we act on: *"we might expect standing to involve rewarding
double foot contact. However, this is problematic since it penalizes the recovery steps
needed to reject disturbances, as that requires breaking ground contact of at least one of
the feet. Additionally, when transitioning from walking to standing, requiring double foot
contact will cause a policy to opt for the closest stance position rather than the most
stable one."*

---

## 1. The non-stepping ceiling, measured from **our** model

`src/solo/lit.py:measure_ceiling()` — `a_stand` keyframe, `mj_forward`, whole-body
`subtree_com`, support hull = convex hull of the **8 sole contact spheres** (the four 5 mm
collision spheres per foot, read from the model: geoms 15–18 / 30–33).

```
m      = 33.341142 kg          (sum of body_mass)
z_c    = 0.691852 m            (CoM height in the stand keyframe)
g      = 9.81 m/s^2            (model gravity)
com_xy = (0.00328, 0.00008)    (CoM ground projection, stand pose)
hull   = (-0.05,-0.1435) (0.12,-0.1485) (0.12,0.1485) (-0.05,0.1435)
         depth 0.170 m (sagittal) x width 0.29701 m (lateral)
support centre (hull area centroid, i.e. source B's CoM target) = (0.03548, 0.0)
```

`J = m * dCOP * sqrt(g / z_c)` with `dCOP` the ray-cast distance from the CoM projection to
the hull boundary **in the push direction** (a single scalar is not a ceiling):

| push yaw | dCOP (m) | J (N·s) | | push yaw | dCOP (m) | J (N·s) |
|---|---|---|---|---|---|---|
| 0° (+x, toe-ward) | 0.1167 | **14.65** | | 180° (−x, heel-ward) | 0.0533 | **6.69** |
| 45° | 0.1651 | 20.72 | | 225° | 0.0754 | 9.46 |
| 90° (+y lateral) | 0.1450 | **18.20** | | 270° | 0.1452 | 18.22 |
| 135° | 0.0754 | 9.46 | | 315° | 0.1651 | 20.72 |

**The two directional values asked for: sagittal (0°) 14.65 N·s, lateral (90°) 18.20 N·s;
the binding one is the heel-ward push at 6.69 N·s; range 6.69 … 20.72 N·s.**

Cross-checks: (i) reproduces `reports/2026-10-08/t1_gate_calibration.md` §2.2 to 3 decimals
(independent derivation, same model conventions); (ii) the measured 12 N·s chest push in
`solo/pushes.py` sits just under the sagittal-lateral band; (iii) it explains the gate's
measured recovery curve (stand_hold recovered a 16 N·s push only in the lateral/oblique
directions). The `~13 N·s` scalar in `pushes.py:18-19` and `eval.py:115` is confirmed wrong
*as a scalar* and is now reproduced as a per-direction table.

Consequence for the train/test mismatch: `TRAIN_MAX_IMPULSE = 12` is **above** the ceiling in
the heel-ward/oblique directions and **below** it laterally — so "trained at ≤12 N·s" never
meant "trained inside the envelope in every direction", and the gate's 16–25 N·s row demands
stepping. `2 x J(180°) = 13.4 N·s < 16 N·s < 2 x J(45°) = 41.4 N·s` (pinned by test
`test_capped_and_uncapped_push_modes_share_the_ceiling`).

---

## 2. Reward terms: shape, weight, source, precedence

`TaskReward("balance", term_set="balance_lit")`; term set
`("upright", "vel_stand", "orientation", "base_height", "com_support", "capture_point",
"grf_even", "airtime", "arm_posture", "action_diff", "torque")`.
Weights live in `RewardWeights` (the `literature` block); every one names its source.
Rule-2 (house) applies: the CoM/velocity/GRF terms are multiplied by the uprightness gate
`clamp01(torso_up_z)` so a *lying* robot cannot farm them; the self-vanishing ones
(orientation, base_height) are noted.

| term | shape | weight | source / justification |
|---|---|---|---|
| `upright` | `clamp01(torso_up_z)` | **1.0** | house rule 1 needs a dominant positive term; A has no alive term (their standing total ≈2.0/step is dominated by the **constant** airtime 1.0), so 1.0 matches A's dominant weight |
| `vel_stand` | `e^{-5 \|v_xy - c_xy\|}` | **0.30** | A x,y velocity **0.15 + 0.15**, standing branch, combined into one 2-D term |
| `orientation` | `e^{-30 * tilt^2}` (tilt = `acos(torso_up_z)`) | **0.20** | A "Roll, pitch orient." — **A's largest term**, verbatim weight and constant K=30 (qd ≈ tilt² for small tilt) |
| `base_height` | `e^{-20 \|p_z - c_h\|}` | **0.05** | A base height, verbatim form + weight; `c_h` = stand keyframe pelvis height 0.79 m |
| `com_support` | `e^{-\|com_xy - SC\|^2 / 0.08^2}` | **0.20** | **B**: CoM horizontal target = centre of the support polygon ("maximum disturbance compensation"). Weight ours (= A's largest); σ = 0.08 m = the G1 foot half-length |
| `capture_point` | `e^{-\|v_com - v*\|^2 / 0.30^2}`, `v* = (SC - com_xy)/sqrt(z_c/g)` | **0.10** | **B** eq. 5 capture point; target `x_CP = SC`. Weight ours (half of `com_support`: at convergence the two are the same signal). σ = 0.30 m/s = 2x the gate's 0.15 m/s stability speed |
| `grf_even` | `e^{-((f_l - f_r)/(f_l + f_r) / 0.5)^2}` | **0.05** | **B** eq. 12 even L/R GRF. Weight ours, deliberately small: this is the one term in tension with A's contact finding (see §2.1) |
| `airtime` | per touchdown: `t_air - 0.4` (s), 0 otherwise | **1.0** | **A** feet airtime, verbatim (`0.4` penalty offset by the airtime the same term pays back); weight 1.0 = A's |
| `arm_posture` | `e^{-3 \|theta_arm - c_arm\|}` | **0.03** | **A** arm position (verbatim form + weight); 1.0 when the mask freezes the arms |
| `action_diff` | `e^{-0.02 sum\|a_t - a_{t-1}\|}` | **0.02** | **A** action difference (verbatim) — "both sources" smoothing |
| `torque` | `e^{-0.02 mean\|tau\| / 50}` | **0.02** | **A** torque (verbatim constant); `tau_max` ours = `TORQUE_REF = 50 N·m`, the ankle actuator limit read from `robots/g1/g1.xml` (`actuatorfrcrange`), the binding balance actuator |

Total for a perfect stand: **1.97/step** (pinned in the test; A's own total ≈2.0/step).
At the stand keyframe the CoM sits 3.22 cm *behind* the hull centre, so `com_support` reads
**0.8516** there — the term's gradient pulls the CoM ~3 cm forward (visible as a small
pelvis-x shift; it is inside the 0.17 m-deep hull and far inside the gate's 0.20 m
`com_offset` bar, whose zero is the mean of the two foot *sites* — 3.5 cm behind the hull
centroid; no conflict, but the operator should expect a forward CoM bias in v6a).

Precedence / safety: `TaskReward` asserts `dominant positive weight > worst-case per-step
penalty sum`. For the lit set that is `upright 1.0 > airtime 0.8` (`PENALTY_MARGIN`
accounts for both feet landing at once). Existing term sets are checked with exactly the
old rule (`alive > sum`), so v5's invariant arithmetic is unchanged.

### 2.1 The double-foot-contact audit (item 3's last bullet)

* **No contact term was added.** `LIT_BALANCE_TERMS` contains no term that requires or
  rewards double support. A's own "Feet contact" term is a **constant 1 for a standing
  command** — "giving no preference for foot contact" — i.e. it has no gradient while
  standing, so transferring it would add a constant, not a signal.
* **Audited the shipped balance set** (`alive, flat_orientation, action_rate, torque_sat,
  joint_limit`): no double-contact and no feet-down term either. What it *does* have is the
  **pelvis-height factor inside `alive`** (`clamp01(pelvis_z/0.79)`), which prices the crouch
  a recovery step needs (at `pelvis_z` 0.55 it costs 30 % of the dominant term). That is the
  term the ticket asks us to flag as "feet-down-ish", and it is why the lit set separates the
  two: `upright` (uprightness only, weight 1.0) + `base_height` (`e^{-20|pz - c_h|}`,
  weight 0.05, A's own height term). The `stance` task's `stance_height` term has the same
  character and should not be used for T1.
* **What breaking contact actually costs in the lit set** (pinned by test): isolating the
  GRF term 0.049/step; the *whole* single-support transition 0.269/step -- that is the
  support hull shrinking to the stance foot, i.e. "put the CoM over the foot you kept", which
  is a real geometric requirement of a recovery step, not a contact-avoidance penalty. 300
  steps of it is still cheaper than one fall (−100).
* `grf_even` is the one transferred term that pulls the other way (it prefers even loading).
  It is weighted 0.05 (deliberately below A's smallest stylistic terms) and
  `--lit-weight grf_even=0` disables it; **a T3 stepping stage should zero it.**

---

## 3. Action mask (item 2)

`src/solo/lit.py:joint_mask()` — built from the model + `a_stand` keyframe; applied by
`SoloEnv.resolve_action`, the single choke point **every** training and evaluation action
passes through (`step()` calls it; scripted controllers reach it via `action_from_ctrl`).
Frozen joints are written to their **exact keyframe ctrl** *after* the mode mapping, so the
mask holds in `absolute` and `residual` mode and cannot be bypassed by an eval that only
knows the action mode. `env.config()["joint_mask"]` echoes the mask (active/frozen names,
frozen target values), which is what a checkpoint carries.

**12 active / 17 frozen of 29.** Active set (source B's recovery set, plus the torso the
gate measures): `hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll` × 2 legs +
`waist_pitch, waist_roll`.

| frozen group | joints | keyframe value (a_stand ctrl, rad) | why it is safe to freeze |
|---|---|---|---|
| arms | shoulder pitch/roll/yaw, elbow ×2 (8) | shoulder_pitch 0.20, shoulder_roll +0.20 (L) / −0.20 (R), elbow 1.28 | the keyframe's own posture (arms out, forearms level, the S1 "verified-stable" stand and the reset-distribution mean, so freezing changes nothing at reset); **source B locks the upper body entirely** and still learns ankle/hip/foot-tilt/stepping recovery |
| wrists | wrist roll/pitch/yaw ×2 (6) | 0.0 | keyframe ctrl is exactly 0; the wrist actuators are the weakest in the robot (±5 N·m vs 50 ankle / 139 knee) and cannot contribute to balance |
| hip yaw | hip_yaw ×2 (2) | 0.0 | leg yaw only steers the foot in double support; T1 commands `wz = 0` and the gate has no yaw criterion |
| waist yaw | waist_yaw (1) | 0.0 | torso yaw is not a balance DOF; the gate measures roll/pitch uprightness and CoM offset |

Honest trade-off: **A keeps the arms free on purpose** ("arm position: loosely defines
desired joint angles for the arms … while allowing use of arms for balancing"), so the mask
is B's choice, not A's. It is a separate stage with its own falsifier (§5, v6b) and it is
default-OFF. Its expected benefit is not only cleanliness: it removes the limb-supported
collapse basin v2/v5 fell into (the arms can no longer catch the body).

**The one-line eval wiring the eval owner must add** (I did not touch `eval.py`): wherever
eval builds its env kwargs, add

```python
env_kwargs = {**env_kwargs, "joint_mask": solo.lit.joint_mask(model)}
# and, for a lit run, "term_set": ckpt["config"]["env"]["reward"]["term_set"]
```

or rebuild it from the checkpoint itself (`ckpt["config"]["env"]["joint_mask"]` carries
`frozen`/`frozen_targets`, and `term_set` is in `ckpt["config"]["env"]["reward"]`). The test
`test_eval_and_training_build_the_same_mask_from_the_same_source` asserts the factory, the
env and the checkpoint config all agree, and `test_mask_freezes_the_frozen_joints_through_the_real_step_path`
asserts `data.ctrl[frozen]` equals the keyframe values **exactly** (`== 0.0` difference)
through the real `step()` in both action modes.

---

## 4. Gamma: what our 0.995 means, next to 0.5 s

`src/solo/lit.py:gamma_table()` (half-life `= ln0.5/ln(gamma)` steps at the control rate).

| setting | rate | half-life (steps) | **half-life (s)** | 1/(1-γ) horizon (s) |
|---|---|---|---|---|
| source B (0.95) | 25 Hz | 13.51 | **0.541** | 0.80 |
| source B's γ at our rate (0.95) | 50 Hz | 13.51 | **0.270** | 0.40 |
| 0.98 @ 50 Hz | 50 Hz | 34.31 | 0.686 | 1.00 |
| 0.99 @ 50 Hz | 50 Hz | 68.97 | 1.379 | 2.00 |
| **ours, balance default (0.995)** | 50 Hz | 138.28 | **2.766** | 4.00 |
| ours, locomotion (0.997) | 50 Hz | 230.70 | 4.614 | 6.67 |

Our half-life is **5.12x** source B's. Source B *derived* 0.95 from "the future-reward
half-life should be ~0.5 s" — that is a **short** horizon for a task whose value is
"sustained standing": at 0.5 s, most of the credit for still standing two seconds from now
(the gate's `time_to_stability ≤ 1 s` plus its 8 s episode) is discounted to noise. Our
task is *closer to source A's regime* (16 s episodes, an explicit standing mode) than to
B's recovery reflex, so **our 0.995 (2.77 s) is the better-matched choice and the default is
unchanged**; 0.95 @ 50 Hz (0.27 s) would be actively wrong here. Gamma is therefore a *late*
lever (v6g) gated on an observed flattening, not a fix.

---

## 5. Staged plan (one lever per run; nothing here is launched yet)

Rules applied: every stage is a **single** change on top of the previous stage's best
checkpoint, never a reward+curriculum or reward+termination bundle; every stage has a
pre-declared observable, a step budget and a falsifier. Each command below is **complete**
(paste the block as-is; it re-lists everything the run needs, so a stage that is falsified is
reverted by simply not using its block for the next stage). **Single-env budget** (v5 measured
≈265 steps/s with a (256,256) net; the source-shaped (100,50,25) net is faster — measure the
stage's own rate from its log at 100k and re-project):

* 1.5 M steps ≈ **78 min** at 320 steps/s; the mid-run read at **400k ≈ 21 min** and a
  second at **800k ≈ 42 min**. Six stages ≈ **7.8 h** — one overnight window.
* If SoloVec's 4× vec backend lands: ≈ **20 min per stage**, the whole sequence in ~2 h, and
  the same window then allows 6 M-step stages if the trajectory justifies it.
* No stage needs >9 M steps. **If any stage needs more than the window, that is a finding,
  not an assumption** — the 400k read is the gate that decides scale-up.

Read protocol (unchanged from T1Midrun/T1GateCal, so numbers stay comparable):
`scripts/solo_env_smoke.py monitor` on a *copy* of the checkpoint (frozen 48-push battery,
takes the sim lock, ≈126 s), plus a 3-seed no-push 2 s deterministic probe
(`mean_upright`, `mean_pelvis_z`). Reference values: v5@100k `mean_upright` **0.652** (full
battery) / **0.665** (non-stepping subset), `recovery_success_rate` 0.0,
`max_recoverable_impulse` 0.0 (it never recovered a single 4 N·s push); StandHold 0.849 /
0.913; proposed T1 bar 0.84.

### v6a — the lit reward set (reward shaping) — **expected biggest lever**

```
MUJOCO_GL=egl .venv/bin/python -m src.solo.train --task balance --steps 1500000 \
    --rollout-steps 2048 --gamma 0.995 --entropy-coef 0.001 --action-mode residual \
    --reward-set lit --out checkpoints/solo/t1_balance_v6a.pt \
    --save-every 50000 --lock off
```
*Single change:* the term set (`--reward-set lit`). Push curriculum, entropy, net, gamma,
termination all as v5. `--alive-weight 10` is dropped because the lit set has no `alive`
term; the survival-dominance arithmetic is `upright 1.0/step vs -100 one-off` ⇒ 400 steps of
standing ≈ +680 ≫ 100, so 1.0 already dominates the termination exactly as v5's 10.0 did.
*Hypothesis:* v5's collapse is a missing-signal problem — its balance set has **no** term for
where the CoM is, how fast it is moving, or whether the torso is level beyond a linear tilt
cost, so "lean back and sit on a limb" scores nearly as well as standing once the entropy
bonus inflates the policy. The lit set gives the gate's own binding metrics direct gradients
(uprightness 0.2 weight + 1.0 gate, CoM offset, stability speed).
*First observable (at 400k, ≈21 min):* subset `mean_upright` ≥ **0.75** (v5: 0.665) **and**
the no-push probe ≥ **0.90** **and** `com_offset_max` ≤ 0.20.
*Falsifier:* subset `mean_upright` ≤ 0.665 at 400k, or the no-push probe < 0.5 (still
collapsing on its own training distribution) ⇒ the reward was not the binding lever: revert
to v5 and start from v6b instead. Also falsified if `mean_return_50` *falls* while
`mean_upright` rises (the policy found a term-farming posture) — reward-term logs are in
`info["reward_terms"]`.

### v6b — the joint mask (action space)

```
MUJOCO_GL=egl .venv/bin/python -m src.solo.train --task balance --steps 1500000 \
    --rollout-steps 2048 --gamma 0.995 --entropy-coef 0.001 --action-mode residual \
    --reward-set lit --freeze-joints balance \
    --out checkpoints/solo/t1_balance_v6b.pt --save-every 50000 --lock off
```
*Single change:* 17 of 29 action dims frozen at the keyframe.
*Hypothesis:* ~2.3 M parameters of the policy's output are balance-irrelevant; freezing them
both concentrates exploration and removes the limb-supported collapse.
*First observable (400k):* no-push probe ≥ **0.95**, entropy flat-to-falling, subset
`mean_upright` ≥ v6a's value at the same step count.
*Falsifier:* no better than v6a at 400k ⇒ drop the mask (it is not needed; A keeps the arms
free). Secondary falsifier: `fall_rate` rises (the mask removed a stabilising arm action).

### v6c — the disturbance distribution (train/test magnitude mismatch)

```
MUJOCO_GL=egl .venv/bin/python -m src.solo.train --task balance --steps 1500000 \
    --rollout-steps 2048 --gamma 0.995 --entropy-coef 0.001 --action-mode residual \
    --reward-set lit --freeze-joints balance \
    --push-curriculum off --lit-push interval --lit-push-interval 5.0 \
    --lit-push-min-frac 0.5 --lit-push-max-frac 2.0 --lit-push-cap 12.0 \
    --lit-push-height 0.95 --lit-push-seed 0 \
    --out checkpoints/solo/t1_balance_v6c.pt --save-every 50000 --lock off
```
*Single change:* the push distribution: source B's 5 s repeated pushes, magnitude
`U(0.5,2.0) x J(direction)` clamped to 12 N·s, i.e. the **whole T1 band in every direction**
with the hard (heel-ward) directions trained at their own ceiling instead of at a scalar 12.
The cap keeps the stage inside the measured non-stepping envelope (12 N·s) so it cannot
silently become a stepping run.
*Variant for T3 (not T1):* drop `--lit-push-cap` (band 3.3 … 41.4 N·s, source B verbatim).
*Hypothesis:* the gate's push criteria are unattainable because training never applied a
push in the direction/magnitude the gate tests; magnitudes are direction-dependent, so a
single scalar cap cannot fix that.
*First observable (400k, ≈21 min):* the monitor's recovery rows leave zero —
`max_recoverable_impulse` (subset) ≥ 4 N·s or `recovery_success_rate` ≥ 0.2 (v5: 0.0/0.0 at
100k), with `fall_rate` ≤ 0.25.
*Falsifier:* recovery still exactly 0.0 at **800k** ⇒ the mismatch was not the binding
cause; stop and re-read the gate calibration rather than extending.

### v6d — the PPO mirror loss

```
MUJOCO_GL=egl .venv/bin/python -m src.solo.train --task balance --steps 1500000 \
    --rollout-steps 2048 --gamma 0.995 --entropy-coef 0.001 --action-mode residual \
    --reward-set lit --freeze-joints balance \
    --push-curriculum off --lit-push interval --lit-push-interval 5.0 \
    --lit-push-min-frac 0.5 --lit-push-max-frac 2.0 --lit-push-cap 12.0 \
    --lit-push-height 0.95 --lit-push-seed 0 --mirror-loss-coef 1.0 \
    --out checkpoints/solo/t1_balance_v6d.pt --save-every 50000 --lock off
```
*Single change:* the auxiliary symmetry loss (one extra optimizer step per iteration on
`mean_j (z_j(s) - [M z(M s)]_j)^2 / (mean z^2 + eps)`, applied after `ppo_update`).
*Hypothesis (source A):* a bipedal stand is L/R symmetric; enforcing it halves the effective
policy space and regularises the sample-starved single-env update.
*First observable (400k):* the logged mirror loss falls from its init value **and** the
monitor read beats v6c's at the same step count.
*Falsifier:* the loss is flat (the map or the coefficient is wrong) or the read is
unchanged ⇒ keep the coefficient at 0.0 (it is only ever a co-flag).
*Known gap:* A does not print the term; the form is stated in `src/solo/mirror.py`, not
guessed silently. The coefficient is ours and must be read off the init loss magnitude
(≈2 for a fully asymmetric mean, ≈0 for the symmetric init) before choosing it.

### v6e — entropy coefficient 0.0 (own stage, per Main)

```
MUJOCO_GL=egl .venv/bin/python -m src.solo.train --task balance --steps 1000000 \
    --rollout-steps 2048 --gamma 0.995 --entropy-coef 0.0 --action-mode residual \
    --reward-set lit --freeze-joints balance \
    --push-curriculum off --lit-push interval --lit-push-interval 5.0 \
    --lit-push-min-frac 0.5 --lit-push-max-frac 2.0 --lit-push-cap 12.0 \
    --lit-push-height 0.95 --lit-push-seed 0 \
    --out checkpoints/solo/t1_balance_v6e.pt --save-every 50000 --lock off
```
*Single change:* the entropy bonus, source-B-exact (they solved this task family with
**no** entropy bonus, and the v1 pathology here was an unopposed outward log-std gradient on
a small net: entropy 12.15 and rising at ~400k with `entropy_coef` 0.01).
*Why not silently adopt 0.0:* with opportunistic learning each update can inflate the mean
too, but the failure we already know is the *opposite* (inflation dominating a flat
advantage). Both readings are reported here; v5's evidence (entropy still climbing, return
flat, `mean_upright` 0.652) says 0.001 is not obviously enough, while 0.0 risks an early
deterministic collapse into a bad basin with no exploration left.
*First observable (200k, ≈10 min — deliberately earlier than the usual read):* the logged
entropy/log-std stops rising (v5's rose monotonically, +1.03/100 iterations) **and** the
no-push probe holds at its v6-previous value.
*Falsifier:* the no-push probe collapses below the previous stage's read before 400k, or
`mean_return_50` falls ⇒ 0.001 stays. Do not stack this with another change: it interacts
with the exploration failure already diagnosed.

### v6f — source-shaped net (100,50,25)

```
MUJOCO_GL=egl .venv/bin/python -m src.solo.train --task balance --steps 1500000 \
    --rollout-steps 2048 --gamma 0.995 --entropy-coef 0.001 --action-mode residual \
    --hidden 100,50,25 --reward-set lit --freeze-joints balance \
    --push-curriculum off --lit-push interval --lit-push-interval 5.0 \
    --lit-push-min-frac 0.5 --lit-push-max-frac 2.0 --lit-push-cap 12.0 \
    --lit-push-height 0.95 --lit-push-seed 0 \
    --out checkpoints/solo/t1_balance_v6f.pt --save-every 50000 --lock off
```
*Single change:* the net. Source B's verified architecture is a **(100, tanh, 50, tanh, 25,
tanh)** policy with a learned log-std and a (100,50,25) ReLU critic; ours is (256,256) and
the mandate suggested 128,128 — theirs is smaller still and is the *verified* number.
*First observable (400k):* matched monitor read with a smaller net and a higher steps/s.
*Falsifier:* worse read, or a wall-clock gain that does not appear in the log.
*Separate stage (v6h, only if wanted):* B's optimiser numbers — 4096 rollout, clip 0.2,
**10 epochs**, lr 3e-4, minibatch 256 (i.e. 16 minibatches), γ 0.95, λ 0.95. Ours are 2048 /
4 epochs / 4 minibatches (512 per minibatch). Adopting B's epochs+minibatch changes the
gradient-per-sample ratio substantially and must **not** ride along with a net or gamma
change; γ 0.95 is rejected outright (§4).

### v6g — gamma (only if a flattening is observed and the horizon is the suspect)

```
MUJOCO_GL=egl .venv/bin/python -m src.solo.train --task balance --steps 1500000 \
    --rollout-steps 2048 --gamma 0.99 --entropy-coef 0.001 --action-mode residual \
    --reward-set lit --freeze-joints balance \
    --push-curriculum off --lit-push interval --lit-push-interval 5.0 \
    --lit-push-min-frac 0.5 --lit-push-max-frac 2.0 --lit-push-cap 12.0 \
    --lit-push-height 0.95 --lit-push-seed 0 \
    --out checkpoints/solo/t1_balance_v6g.pt --save-every 50000 --lock off
```
*Falsifier:* recovery or stability degrades at 400k (the 2.77 s horizon is what values
*sustained* standing).

### v6h — partial observability: `--frame-stack 2` — **competing explanation, own stage**

```
MUJOCO_GL=egl .venv/bin/python -m src.solo.train --task balance --steps 1000000 \
    --rollout-steps 2048 --gamma 0.995 --entropy-coef 0.001 --action-mode residual \
    --frame-stack 2 --out checkpoints/solo/t1_balance_v6h.pt \
    --save-every 50000 --lock off
```
*Single change:* the observation history (1 -> 2 frames; the input widens to
2 x 115 = 230 for the actor and 158 + 115 = 273 for the critic).  Nothing else.
*Why this is a real competing explanation, grounded:* the observation our actor gets today
is **memoryless** — `ACTOR_LAYOUT` = base linear velocity (3), base angular velocity (3),
gravity direction (3), joint positions relative to the keyframe (29), joint velocities (29),
**previous action** (29), command velocity (3), command stance (2), skill one-hot (10),
lead leg (1), phase (1).  Present: positions, velocities, previous action, command.
**Absent: contact state (feet/knees/hands/dorsal), the CoM position and velocity, the support
centre, tilt, pelvis height, the next push, foot slip, saturation, joint-limit proximity,
any history beyond one action.**  Those absent fields live in the *privileged* block, which
only the critic sees — so the value function can measure the CoM offset while the policy
cannot.  Source A's controller is a (64,64) **LSTM** on the same observation set, and
T1GateCal/TeacherDataAudit measured that our hand-built balancer's behaviour depends on
integrator/rate-limiter state that is not observable from a single frame.  A memoryless
policy may simply be being asked for a non-Markov function.
*First observable (200k ≈ 10 min — the shortest useful budget, this is a probe not a
training run):* the no-push probe is unchanged but `mean_upright` on the non-stepping subset
is at least v5's at the same step count, and the entropy stays below v5's trajectory;
falsifier: no better than the memoryless run at 400k ⇒ the missing information is *state*
(not history) and the fix is an observation change, not a stack.
*Cost:* one flag, no new parameters beyond a wider first layer
(2 x 115 -> 100 vs 115 -> 100).  **If frame stacking is not enough I would NOT implement a
recurrent policy yet.**  The design I would use: a GRU/LSTM encoder (64,64) on the actor obs
per step with the rollout storing per-step hidden states and PPO's GAE computed on sequences
(burn-in + truncated BPTT over the rollout, minibatch = whole episodes).  Cost: the trainer
becomes sequential per step (no per-step batching), the checkpoint gains hidden state,
`ActorCritic` grows an encoder path, and the vec backend must carry sequence boundaries —
that is a multi-day change to `src/rl`, not a flag.  Frame stacking is the cheap probe first
because it keeps PPO's per-transition structure exactly as it is and needs one parameter.

---

## 6. What could not be transferred, and why

1. **A's "feet contact" reward and its constant-for-standing airtime branch.** Their contact
   term is a constant 1 while standing (no gradient); their airtime term is likewise a
   constant for standing. Transferring constants buys nothing, and their *non*-standing
   contact form is exactly the double-foot-contact design they warn against. We transfer the
   *touchdown* airtime form instead (the ticket's step-frequency regulariser), documented as a
   deliberate deviation with its bounded cost (−0.4/touchdown, +0.1 max).
2. **A's camera-ready mirror-loss formula.** The paper says only "extended with a mirror
   loss"; the exact term is not published in the version I read. I implemented a stated,
   testable form (§5 v6d) rather than guessing at theirs, and kept it behind a coefficient
   that defaults to 0.0.
3. **A's base-acceleration term (weight 0.1) and feet-orientation/feet-position (0.05).**
   Base acceleration needs `qacc`, which is contact-dominated and noisy in our scene;
   feet orientation/position presume a walking gait with a stance-foot target. Neither is
   needed for the T1 metrics; both are cheap to add later if a stage points at them.
4. **B's "penalise losing foot contact".** Deliberately not transferred: it is the
   contact-requiring term A's finding forbids for standing (and B could afford it because
   their policy *steps*).
5. **B's arm/foot-tilt (whole-foot) actuation.** Their robot has actuated feet; our G1's
   ankle_roll is the closest analogue and is already in the active mask. No extra DOF exist.
6. **B's 25 Hz / PD 500 Hz and their push-application point (pelvis).** We keep 50 Hz and the
   measured chest height (0.95 m) because the gate battery uses that height; changing the
   application height would change the test conditions, which is T1GateCal's territory.
7. **B's exact reward weights.** I read their *structure* (exp-quadratic on pose/CoM/CoM
   velocity/GRF, weighted) and their analytic ceiling and derivations, but not a published
   per-term weight table for our joint set; the weights for the terms we took from B are
   therefore ours, marked as such in §2 (they are ≤ A's largest term by construction).

### 6.1 Cost and isolation

The literature inputs added to `_reward_inputs` (world CoM + mass-weighted CoM velocity,
8 sole-sphere points, 2 GRF reads, the arm deviation, the actuator forces) cost **138 µs
against a 14 ms step (≈1 % of an environment step)** on the loaded box. **v5's running
process is untouched by any of this**: it imported the modules at start-up, so the edits
cannot reach it, and `solo-t1-v5` was never signalled, restarted or reconfigured.

## 8. Test evidence

`tests/solo/test_lit_reward.py` — new file, **33 tests, 33 passed** (`pytest tests/solo/test_lit_reward.py -q`,
13.7 s), each asserting numbers on hand-constructed states. Highlights and the numbers they pin:

| test | what it pins (numbers) |
|---|---|
| `test_ceiling_matches_the_model_and_is_direction_dependent` | m 33.341142, z_c 0.691852, hull 0.170 x 0.29701, support centre x 0.035484, dCOP/J at 0/45/90/135/180°, `J = m dCOP sqrt(g/z_c)` to 1e-12 |
| `test_ceiling_from_hull_arithmetic_on_a_hand_hull` | centred 0.2 m square, m=10, z=1, g=10 ⇒ dCOP 0.1, J = 10*0.1*sqrt(10); an off-centre CoM ⇒ 0.05 / 0.15; an outside CoM ⇒ 0 |
| `test_support_centre_is_the_hull_area_centroid` | the real trapezoid footprint ⇒ 0.0354852 (0.5 mm off the vertex mean 0.035); a trapezoid where the vertex mean is 5.6 cm off; loaded-foot-only and NaN-when-airborne cases |
| `test_com_support_...` | 1.0 at the hull centre, `e^-1` at a 0.08 m offset, monotone toward the border, gated by uprightness (x0.5), 0.0 when airborne |
| `test_capture_point_matches_the_cp_implied_velocity` | tau(z_c=0.6919) = 0.2655750 s, v* = 0.1336166 m/s, term 1.0 exactly at the CP-implied velocity, `x_CP == support centre`, 0.8200661 at v=0, 0.4522676 at −v*, sqrt(z) scaling, 0 when airborne/zero-height |
| `test_grf_even_...` | 1.0 at equal loads, e^-1 at a 75/25 split, e^-4 when one foot carries all, 0.0 with no load |
| `test_airtime_penalty_fires_once_per_touchdown` | 0 with no touchdown, 0.0 at t_air = 0.4, −0.4 at 0.0, +0.1 at 0.5, −0.8 for both feet (== `PENALTY_MARGIN`), weight 1.0 |
| `test_orientation_and_base_height_use_the_source_forms` | `e^{-30 tilt²}`: 1.0 / 0.4009766 (10°) / 0.0020753 (26°); `e^{-20|Δpz|}`: 1.0 / 0.36788 (both ±0.05 m); weights 0.20 and 0.05 |
| `test_smoothing_and_survival_terms` | upright 1.0/0.3; `e^{-5|v|}` 1.0/0.36788; `e^{-0.02 sum|Δa|}` = 0.94365 for 29x0.1; `e^{-0.02 mean|tau|/50}`; arm 0.36788 at 1/3 rad; weights |
| `test_every_lit_term_is_logged_and_the_total_is_the_weighted_sum` | exact lit totals 1.9162859 (hand state) and 1.97 (CoM on the centre) |
| `test_lit_set_has_no_double_foot_contact_requirement` | no contact term in the set; the shipped balance set has none either; breaking contact costs 0.0490842 (GRF) and 0.2687284 (whole single-support change) vs a −100 fall |
| `test_mask_freezes_the_frozen_joints_through_the_real_step_path` | 12 active / 17 frozen; `data.ctrl[frozen] == keyframe` **exactly** in residual **and** absolute mode; active joints move > 0.1 rad; the env config echoes the mask |
| `test_eval_and_training_build_the_same_mask_from_the_same_source` | factory/env/checkpoint agree; the trainer's env carries the same dict; `term_set` recorded |
| `test_mirror_map_is_the_geometric_mirror_on_the_real_model` | mirrored perturbed qpos puts every `left` body where the mirrored `right` body is (< 1e-4 m over 13 body pairs, two random perturbations); roll/yaw signs |
| `test_mirror_maps_are_involutions_and_the_loss_is_zero_when_symmetric` | involution; loss 0.0 for a symmetric bias, **exactly 2.0** for `z = e_0`, 0.0 after mirroring the bias; one SGD step reduces it |
| `test_push_mode_bernoulli_rate_and_zero_probability` | 350 control steps; p=0.01 ⇒ mean 3.5 pushes/episode within 10 % over 200 episodes; p=0 within 20 episodes; p=1 ⇒ 350; single-timestep duration |
| `test_push_mode_interval_spacing` | t = 1.0/6.0/11.0 at 5 s; 1/3/5/7 at 2 s; jitter bound; control-step grid |
| `test_push_magnitudes_follow_the_directional_ceiling` | every frac in [0.5, 2.0]; mean 1.25 ± 0.06; the weak direction means 8.36 N·s vs the lateral 22.75; all 8 directions appear; cap binds; determinism |
| `test_capped_and_uncapped_push_modes_share_the_ceiling` | 2 x J(180°) = 13.38 < 16 < 2 x J(45°) = 41.45 |
| `test_gamma_half_life_table` | 2.7657 s (ours) / 0.5405 s (source B) / 0.2703 s (B's gamma at 50 Hz); ratio 5.117 |
| `test_v5_configuration_is_untouched_by_the_literature_work` | `TASK_TERMS["balance"]`, `RewardWeights()` defaults, gamma 0.995, a default env has no mask/term set |
| `test_penalty_invariant_holds_for_every_term_set` | old rule `alive > Σ` for all 6 task families, new rule `upright 1.0 > 0.8` for the lit set, both raise cases |
| `test_lit_flags_parse_and_are_listed` / `test_lit_ceiling_flag_...` | every new flag is in `--help` (and no v5 flag was lost); `--lit-ceiling` prints 14.653 / 33.341142 / n_frozen 17 |
| `test_lit_burst_runs_end_to_end_through_the_trainer` | 128 real trainer steps with all lit flags: term set recorded, 17 frozen, lit pushes installed and capped, mirror loss reported, checkpoint saved |
| `test_lit_push_and_v5_are_mutually_exclusive_where_it_matters` | a lit-push run does **not** inherit the v5 ramp; all new defaults are off |
| `test_lit_weight_overrides_are_validated` | a bad `--lit-weight NAME=VALUE` exits non-zero; `grf_even=0` reaches the weights |
| `test_lit_curriculum_adapter_is_what_the_vec_backend_calls` | the `schedule_for(steps, seed)` shape the vec backend uses; no ramp; per-worker schedules differ, reproducible; `as_dict` carries the ceiling |
| `test_frame_stack_widens_the_input_and_stacks_oldest_first` | N frames: actor in 3x115, critic 158+2x115; oldest-first order pinned on synthetic frames; reset re-seeds (no stale frames cross an episode boundary); `--n-envs 2` + N>1 refuses to run |
| `test_env_feeds_the_literature_inputs_in_one_frame` | the env's lit inputs are physical and world-frame: com_z 0.60–0.75 m, sole spheres on the mat, foot loads ≈ m g (±2x), mass-weighted CoM velocity |

Independent check that v5's configuration is untouched: the **unmodified** S1 harness script
`scripts/solo_env_smoke.py smoke` still prints its own verification
(`reward_sum_5s` = 248.3, i.e. ≈1.0/step for the default balance set, `stand_pelvis_z`
0.79–0.7916, no termination) — the same values as before this work.

`tests/test_solo.py` (the S1 harness suite) gained the literature inputs in its shared
`_upright` hand state plus `good`/`bad`/`delta` entries for the 11 new terms, so its contract
("every term in `TERM_FUNCS` ranks a genuine state above its degenerate counterpart") now
covers the new terms too — the assertion was **strengthened, not relaxed**.

**Attributed failures in the current working tree** (none of them mine; all pre-date or
post-date my change):
* `tests/test_solo.py::test_push_schedule_battery_deterministic` (line 360) and
  `::test_push_curriculum_ramp_and_held_out_boundary` (line 890) assert "held-out magnitudes
  are strictly above `TRAIN_MAX_IMPULSE`". That is stale after the in-flight R2b redefinition
  in `eval.py` ("held-out = off-training *heights* inside the in-band magnitudes"), which makes
  the held-out set contain 4 N·s pushes. Whoever owns the split must update these two.
* `tests/test_solo.py::test_eval_run_not_certified_for_stand_hold` asserts the balance gate's
  reasons contain `max_recoverable_impulse` — stale after the same split removed that
  criterion from `GATES["balance"]`.
* `tests/solo/test_bc.py` — 1 failure + 5 errors: `data/solo/bc/bc_metrics.json` does not
  exist yet (ImitationBC's training artifacts are not generated).
* `tests/solo/test_vec_solo.py::test_env_construction_kwargs_reach_workers` — `NameError:
  joint_mask`, caught mid-edit in SoloVec's in-flight vec work (they are adding exactly the
  pass-through this work needs).

## 9. Which lever I expect to matter most (honest call)

**v6a, the reward term set.** The evidence: with *no push at all* the v5-era policy collapsed
on its own training distribution (`mean_upright` 0.13 / −0.17 / 0.16 on a 2 s deterministic
hold), and its reward set contains no term that measures the two things the gate actually
grades — where the CoM is relative to the support polygon, and how fast it is moving. Two of
the three shaping levers already tried (entropy, push curriculum, alive weight) moved the
return but not the behaviour, which is what a *missing-signal* failure looks like rather than
an exploration-budget failure.

Second most likely: **v6c** (disturbance distribution) — not because recovery is the current
blocker, but because after v6a/v6b produce a policy that can hold a stand, the gate's push
criteria are the only ones left that no run has ever moved off zero.

**Which single stage I would run FIRST, and why.** **v6a (the lit reward set).** Two pieces of
evidence from our own runs, not from a paper: (i) v5 degraded *from a standing initialisation*
— the policy started as `a_stand` (the startup check measured 2.6 mrad) and moved to a
limb-supported crouch on its own training distribution; (ii) the gate's own reference
(`stand_hold`, a static controller) scores `mean_upright` 0.913 on the non-stepping subset
while our policy scores 0.665, so the gap is not push recovery — it is holding the stand. A
policy that abandons a stable equilibrium it was initialised in is missing a *signal* about
where the equilibrium is; that is what the CoM/support/capture-point/orientation terms supply
and nothing else in our current reward set does.

**Competing explanations for any v6 result**, recorded so a positive result cannot be
attributed by default: (1) **reward scale** (the lit set's per-step total is 1.97 vs v5's
10.0 with `--alive-weight 10` — a scale change alone can change PPO's effective step size and
value-loss calibration; the diagnostic is `value_loss` and the return curve, and the control
is the same stage with `--lit-weight upright=5`); (2) **reward alignment** (the lit terms
target the hull centroid while the gate's `com_offset` is measured against the foot-site
mean, 3.5 cm apart — §7); (3) **the disturbance distribution** (v6c; the mismatch is real but
it cannot explain a no-push collapse); (4) **partial observability** (v6h; the actor cannot
see contacts, the CoM or the support centre, and source A's policy is an LSTM); (5) **the
optimiser/entropy pathology** already diagnosed (v6e); (6) **capacity/architecture** (v6f).
Each has its own stage so the attribution is a measurement, not an argument.

**The honest uncertainty I would name first**: whether the gate's non-stepping subset
(4/8/12 N·s) is inside the *policy's* reachable envelope at all given a memoryless 115-dim
observation — the ceiling says 4-12 N·s is resistible *with full state feedback*, and the
hand-built balancer that does it reads hidden state our actor does not have.
