# Prior art: learned humanoid balance and locomotion, and a CPU-feasible recipe for M1/M2

Scope: what established work does for **dynamic balance (M1)** and **velocity-commanded locomotion
(M2)** as defined in `docs/MOTOR_CURRICULUM.md`, and what of it survives on **this** box:
4-core ARM, no GPU, MuJoCo 3.15.0 CPU + PyTorch 2.14 CPU, small nets, hours-not-days training.

Everything here is a **starting point to be validated experimentally**, never an authoritative value
(`MOTOR_CURRICULUM.md` §3: thresholds are placeholders set from a baseline-first measurement).
Every non-obvious claim carries a source. Claims marked `[INFERENCE]` are arithmetic or reasoning
derived from the cited source, not quoted from it.

Primary sources used, and the key datum each contributes:

| # | Source | Contributes |
|---|---|---|
| D | DeepMimic, Peng et al. 2018 — https://arxiv.org/html/1804.02717v3 | imitation reward formula/weights, RSI+ET, **60M samples ≈ 2 days on 8 CPU cores, no GPU** |
| A | AMP, Peng et al. 2021 — https://arxiv.org/html/2104.02180v2 | discriminator style reward, weights, gradient penalty, failure modes |
| I | IsaacLab velocity task (source, main branch) — https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py | canonical velocity reward term set + weights |
| IG | IsaacLab G1 velocity config — https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py and `.../g1/flat_env_cfg.py` | G1-specific weights, `-200` termination, PPO cfg |
| IM | IsaacLab core reward/termination kernels — https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/envs/mdp/rewards.py | exact formulations (exp kernels, L2 penalties) |
| P | MuJoCo Playground G1 joystick — https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/mujoco_playground/_src/locomotion/g1/joystick.py | 29-DoF G1, sim-to-real-verified reward set, push 0.1–2.0 every 5–10 s |
| PG | MuJoCo Playground paper — https://arxiv.org/html/2502.08844v1 | GPU budgets (400M steps G1 joystick), G1 fall termination |
| G | Gymnasium Humanoid-v5 / HumanoidStandup-v5 (docs + source) — https://gymnasium.farama.org/environments/mujoco/humanoid/ , https://gymnasium.farama.org/environments/mujoco/humanoid_standup/ , https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium/envs/mujoco/humanoidstandup_v5.py | benchmark reward formulas and their documented exploits |
| M | dm_control humanoid stand/walk — https://raw.githubusercontent.com/google-deepmind/dm_control/main/dm_control/suite/humanoid.py | tolerance()-shaped upright/height rewards; standing-still is *deliberate* there |
| L2WIM | Rudin et al. 2021 — https://arxiv.org/html/2109.11978v3 | massively-parallel recipe, DR + push schedule, `<20 min` GPU |
| ISAACG | Isaac Gym paper — https://arxiv.org/html/2108.10470v2 | GPU-vs-CPU throughput ratio (2–3 orders), **39M samples ≈ 30 h on 16 CPU cores** |
| RMA | Kumar et al. 2021 — https://arxiv.org/abs/2107.04034 | DR ranges, **"learns to fall because of the penalty terms"**, privileged adaptation |
| PD18 | Peng et al. 2018 — https://arxiv.org/abs/1710.06537 | DR magnitudes, latency model |
| HK19 | Hwangbo et al. 2019 — https://arxiv.org/abs/1901.08652 | mass/CoM randomization %, noise, ≤11 h CPU+1 GPU sessions |
| F | Ferigo et al. — https://ar5iv.labs.arxiv.org/html/2104.14534 | push-recovery battery with a success-vs-magnitude curve |
| Y | Yang et al. — https://arxiv.org/pdf/2002.02991 | **closed-form max rejectable impulse**, strategy-vs-magnitude table, CPU-only training |
| DU | Duburcq et al. — https://arxiv.org/pdf/2203.01148 | push schedule, **force ramp curriculum hurt** |
| VM | van Marum et al. — https://arxiv.org/pdf/2404.19173 | real-robot push grid, non-monotonic success, reward-induced anti-disturbance |
| PS | Push-and-Step — https://openaccess.thecvf.com/content/CVPR2026/papers/Jensen_Push-and-Step_From_RL-Based_Balance_Recovery_to_Physical_Simulation_of_Dense_CVPR_2026_paper.pdf | CoM/CoP balance reward, max-recoverable-push metric |
| HI | HiFAR — https://arxiv.org/html/2502.20061v2 | push magnitudes for fall recovery, robustness sweeps |
| PR | Pratt et al. 2006 — http://www.ambarish.com/paper/Pratt_Goswami_Humanoids2006.pdf | capture point, capture region |
| ST | Stephens 2007 — https://www.cs.cmu.edu/~cga/papers/stephens-hum07.pdf | ankle/hip/step decision surfaces, optimal step rule |
| H12 | H1-2 balance work — https://arxiv.org/html/2603.08619v1 | force→strategy threshold bands, 10k-trial eval battery |
| N99 | Ng et al. 1999 — http://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf | potential-based shaping is policy-invariant |
| MJ | MuJoCo docs — https://mujoco.readthedocs.io/en/stable/modeling.html , https://mujoco.readthedocs.io/en/stable/computation/index.html , https://mujoco.readthedocs.io/en/stable/XMLreference.html | soft contacts, slip, clamp, integrator |
| MZ | MuJoCo Menagerie G1 — https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_g1/g1.xml | our model's exact actuators/limits |
| R | **our repo** (measured, cited by path) | harness facts that constrain every recipe |

---

## 1. Method summary

Read the "CPU-feasible adaptation" column as *what we do here*, not as a claim that the method was
designed for CPU.

| approach | core idea | what it needs (compute / data) | CPU-feasible adaptation for us | evidence it works |
|---|---|---|---|---|
| **Reference-motion imitation** (DeepMimic) | reward = weighted imitation terms (pose/vel/end-effector/CoM) + task term; episode init sampled from the reference (RSI); early termination on undesired contact | 60M samples per policy ≈ 2 days on 8 CPU cores, **no GPU** (D §A4, code README); reference clips (mocap / retargeted GrappleMap) | Directly usable *in principle*: we already have retargeted refs (`data/refs/*.npz`) and `action_mode="residual"`. But 60M samples ≈ 4 days on our 4 cores `[INFERENCE]`, so **not** for M1 — reserve for M4/M5 where style matters | RSI+ET vs fixed init returns: Backflip .791/.730/.379, Sideflip .823/.717/.355 (D Table 5); ET removes the "falls and mimes on the ground" local optimum (D §10.4) |
| **Adversarial motion priors** (AMP / ADD) | a GAN discriminator scores "does this transition look like the clip data"; reward = task + style | modern runs use 750M–850M samples on A100s (ADD paper, https://arxiv.org/html/2505.04961v1); AMP paper itself reports no compute (A) | **Not applicable as-is.** Needs reference data *and* GPU scale; also the loosest tracker (position error 0.163 m vs DeepMimic 0.013 m, ADD Table 1) — bad for wrestling where geometry must be right | AMP works for stylistic tasks (A); but 300× less sample-efficient for tracking than direct imitation |
| **Isaac-style velocity tracking** (legged_gym / IsaacLab / Playground) | task = follow (vx,vy,ωz); dense penalties for everything ugly; terrain curriculum; privileged critic | 4096 envs ≈ 100k–200k steps/s on GPU (ISAACG §5.2, PG); G1 joystick = **400M steps** (<30 min on 2×4090, PG §4.2.2); IsaacLab G1 flat = 147M, rough = 295M steps (IG PPO cfg: 1500/3000 iters × 4096 envs × 24) | **This is the M2 template**, especially MuJoCo Playground's *G1* config (29 DoF, 50 Hz, position servos — structurally identical to ours). We port the reward set and delete everything that needs terrain/height scans. Budget: 147M steps ≈ 102–204 h on our box `[INFERENCE]` → we run 1–10% of it and accept a cruder gait | G1/H1 policies from exactly this set are sim-to-real deployed (P, IM); the set is the de-facto standard |
| **MuJoCo humanoid benchmarks** | single scalar reward, Gymnasium-normalized; used as a baseline, not a recipe | 10M steps with 256×256 MLPs on CPU for Humanoid/HumanoidStandup (SB3-zoo: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml) | Keep as **regression baselines only** (we can reproduce 10M steps). Their reward sets contain documented bugs/exploits (G: v4 alive bonus paid on the terminal step, issue #526; Standup never terminates and pays absolute height) | Humanoid-v5 reaches ~6–7k return; but HumanoidStandup's reward "does not require staying up" `[derived from G source]` |
| **Push-recovery / fall-recovery RL** | train with randomized pushes; evaluate on a force×direction grid | Ferigo: 20M steps/trial × 11 runs (~hours on GPU); Yang: **CPU-only, converges in 2 days** on an i7-6700K with a 100-50-25 MLP (Y §V) | **The most CPU-relevant literature for M1.** Yang's setup (workspace-scale humanoid, tiny MLP, CPU, dense balance shaping) is *our* regime. Use it for the M1 termination/reward/battery design | Yang: TRPO rejects 240 N·s sagittal vs 53 N·s analytic non-stepping limit (Y); Ferigo: all forces ≤200 N countered, out-of-sample up to 300–400 N (F) |
| **Sim-to-real domain randomization + privileged critic** | randomize dynamics; asymmetric actor–critic (critic sees the truth) | DR is free at runtime; RMA needs an adaptation module at 10 Hz | We are **sim-only** (`MISSION.md`: the artifact is a video), so DR is not for hardware transfer — it is for **robustness to model error and for the match-2 opponent**. We already have the critic split (162-dim privileged vector) | Privileged critic "significantly improves performance" (https://arxiv.org/abs/1710.06542); realistic DR ranges in §4 |
| **Classical model-free-of-learning balance** (capture point / ZMP / CMP; Pratt, Stephens) | closed-form: step exactly when the capture point leaves the base of support | nothing — it is algebra | Use it as (a) a **reward feature** (capture-point margin), (b) an **eval metric**, (c) an **analytic push-magnitude scale** so our push grid is physical rather than arbitrary | Pratt 2006: `x_capture = ẋ·sqrt(z0/g)`, step iff capture point outside BoS, and a flywheel (arm/torso authority) turns the point into a *region* (PR) |
| **Cheap sample-efficiency add-ons** | RSI, assisted exploration, symmetry, potential shaping | free | All of them: residual-on-`stand` actions, reference-state init from `stand`, pull-force assist if M2 stalls (HoST), action-rate limiting, mirror loss | HoST ablation: removing the 60%-gravity pull force collapses get-up success 99.5% → 0.0% (https://arxiv.org/html/2502.08378v1 Table III); shaping is policy-invariant (N99) |

**What is explicitly NOT applicable as-is**: massively parallel Isaac/MJX pipelines (4096–32768 envs,
100k–200k steps/s, 147M–400M-step budgets, `<30 min` wall clocks) — those are GPU throughput claims and
their *convergence* depends on 10²–10³× more samples than we can produce (`ISAACG` §5.2, `PG` Table XX,
`L2WIM` §4.2). Also out of reach: rough-terrain/perceptive locomotion, visual policies, and any
"tune 20 reward weights by grid search" workflow — our budget permits a handful of runs, not a sweep.

---

## 2. Reward-term catalogue for balance and locomotion

Formulations are the ones actually shipped, not textbook ideals. Convention warning: **IsaacLab applies
rewards per control step** (no `dt` factor); **MuJoCo Playground multiplies the weighted sum by `dt`**.
Our `rl.reward` mixes conventions (rate-per-second for `engagement`/`technique_similarity`, per-step for
`progress` — `src/rl/reward.py`, `StageReward.shaping`), so pick one and state it in the config.

### 2.1 The terms

| term | what it is FOR | failure mode it prevents | typical formulation shape | known exploit it invites |
|---|---|---|---|---|
| **Upright / orientation** | keep the trunk gravity-aligned; the cheapest proxy for "still a biped" | leaning, spiral drift, "falling with style" | penalty: `flat_orientation_l2 = Σ projected_gravity_b[:2]²` (IM); reward: dm_control `tolerance(torso_upright, bounds=(0.9, ∞), sigmoid="linear", margin=1.9, value_at_margin=0)` (M). G1 weights: **-1.0** rough/flat (IG), **-2.0** Playground (P); H1 **-1.0** | As a *penalty only* it is satisfied while lying down (no term is paid after termination) — see the penalty trap, §2.2 |
| **Base height** | hold the COM/pelvis at the height the actor was designed for | crouching, sinking, "dragging leg", squats that dodge the upright term | `base_height_l2 = (z - z_target)²` (IM); DWL `φ(z - 0.7, 10)·0.5` (https://arxiv.org/html/2408.14472v1); SaW `e^{-20|z-c_h|}·0.05` (VM). Targets: 0.78 m (unitree 12-DoF G1), 0.7 m (DWL), weight **-10.0** (unitree G1) | Encourages a rigid, non-reactive posture; combined with a strong action-rate penalty it produces a frozen "statue" that survives only the pushes it was trained on |
| **CoM-in-support / capturability** | physically correct "am I balanceable" signal | the *real* M1 failure: COM outside the support polygon with no step taken | Push-and-Step: momentum-based target CoP from CoM height, linear- and angular-momentum rate with damping `d_l=4, d_h=6`, weight `w_g=0.2` (PS); Pratt capture point `x_c = ẋ·sqrt(z0/g)` (PR); our own curriculum asks for "CoM-in-support" (`MOTOR_CURRICULUM.md` §2) | Rewarding *distance travelled by the CoM* invites walking; rewarding *low CoM velocity* invites freezing (VM documents a clock-based reward that "directly impedes disturbance rejection") |
| **Velocity tracking — linear** | the M2 task itself | not moving / not following commands | `exp(-‖cmd_xy - v_yaw_xy‖² / std²)`, `std=0.5` (IM, I); G1 **1.0** (IG); Playground **1.0**; DWL uses `exp(-5‖e‖²)` (https://arxiv.org/html/2408.14472v1) | **Falling forward at the commanded speed** — our own §3.4 rule and the measured E1 baseline; also farming the term by diving (the tracker is high for the seconds before impact) |
| **Velocity tracking — angular** | turn without translating | spinning in place, uncontrolled yaw | `exp(-(cmd_ω - ω_z)²/std²)`, `std=0.5` (IM); G1 rough **2.0**, flat **1.0** (IG); H1 **1.0**; Playground **0.75** (P) | Pure yaw spinning to farm yaw reward while standing (`MOTOR_CURRICULUM.md` §2 lists it); the standard counter is `heading_command` + a yaw-*rate* penalty (-0.1 in L2T/DWL `[INFERENCE]` pattern) |
| **Foot contact / air-time** | force actual steps, regularize step frequency | dragging feet, shuffling, standing while commanded to walk | IsaacLab biped: `feet_air_time_positive_biped` — reward single-stance time clamped at `threshold` (0.4 s G1/H1, 0.5–0.6 elsewhere), **gated by `‖cmd_xy‖ > 0.1`** (I). Weights **0.25** rough (IG), **0.75** flat, **1.0** H1 flat, **2.0** Playground | **Hopping / double-support jumping** (SaW §IV-B); too-high step frequency when unregularized (SaW); **marching in place at zero command** if the `‖cmd‖>0.1` gate is dropped |
| **Gait phase / contact schedule** | enforce a legible gait and proper weight transfer | hopping, asymmetric gallop, "wrong leg at the wrong time" | L2T: `r = c1` if `sign(sin(2πt/h)) == sign(GRF>0)` else `-c2`, cycle `h = 0.68 s`, weight 2.0 (https://arxiv.org/html/2402.06783v3); Playground `feet_phase` 1.0 (P); DWL periodic force/velocity (arxiv 2408.14472) | An over-strong phase term fights perturbations: van Marum document that a clock-based reward inhibits the very steps needed for recovery (VM) |
| **Foot slip** | no skating; contacts must be real grips | sliding, low-friction gait disguise | `Σ ‖v_foot_xy‖ · 1[contact]`, contact if `‖F_net‖ > 1 N` over history (I). G1 **-0.1**, H1 **-0.25**, Playground **-0.25** | Penalizing slip without rewarding steps makes shuffling look good; in MuJoCo, *some* slip is unavoidable (soft contacts do not guarantee zero-velocity stick, MJ) so this penalty cannot be driven to 0 |
| **Action rate (smoothness)** | cheap regularizer, protects the servos, closes the sim-to-sim gap | chattering, bang-bang control, jitter | `Σ (a_t - a_{t-1})²`, weight **-0.005** (IsaacLab G1/H1), **-0.01** (base cfg), **-0.015** (L2T), DWL uses the second difference `‖a_t - 2a_{t-1} + a_{t-2}‖` at -0.01 | Large values produce a low-gain, unresponsive policy that cannot step: exactly the anti-disturbance mechanism VM describe |
| **Torque / effort** | keep torque inside the actuators' honest range, reduce heat | saturation, "fighting itself", high-gain buzzing | `Σ τ²` weight **-1.5e-7** (G1, hips+knee+ankle), **-2.0e-6** (G1 flat, hips+knee), **-1.0e-5** (base), **-1e-5** (L2T) | With **position servos** the action is a target, so penalizing `Σctrl²` penalizes *targets*, not torque; the honest term is `Σ actuator_force²` (`MZ`: `F = kp(ctrl-q) - kv·q̇`, clamped by joint `actuatorfrcrange`) |
| **Energy / power** | encourage economy, enable long matches | wasteful, high-KE flailing | DWL `|τ||θ̇|` at **-1e-4** (arxiv 2408.14472) | Energy efficiency fights recovery: VM measure ≥33 J/m extra energy *attributable to being ready to reject disturbances* (VM) — do not tune this term tight while balance is the goal |
| **Joint deviation (default pose)** | keep non-essential joints (arms, hips yaw/roll, torso) in a neutral range | arms flailing to fake airtime, hip-yaw drift | `Σ |q - q_default|`, weights **-0.1** arms/hip/torso, **-0.05** fingers (IG); **-5.0** hip in L2T | A strong version forbids the arm swings that real balance needs (PR: angular-momentum authority *strictly enlarges* the recoverable set) |
| **Stand-still regularizer** | hold the default pose when the command is ~0 | marching in place at zero command, drift when told to stand | `joint_deviation_l1 · 1[‖cmd_xy‖ < 0.06]` (I); Playground `stand_still` **-1.0** (P); IsaacLab keeps `rel_standing_envs=0.02` so 2% of the batch is standing | Directly opposes disturbance rejection if applied while the robot is being pushed (VM) |
| **Joint position limits** | keep joints off hard stops | limit-banging, torque spikes, sim-only postures | `Σ clip(lo - q, 0) + clip(q - hi, 0)`, weight **-1.0** on ankles (IG), **-5.0** in unitree 12-DoF | — |
| **DoF acceleration** | smoothness at the joint level | vibration, near-limit oscillation | `Σ q̈²`, **-1.25e-7** (G1 rough, hips+knee), **-1.0e-7** (G1 flat), **-2.5e-7** (base) | — |
| **Alive bonus** | **the only term that makes surviving better than dying** | dying to escape the penalty stream (§2.2) | `(~terminated) · w`: Gym Humanoid **5.0/step**, dm_control uses a multiplicative `stand_reward ∈ [0,1]` (M), unitree G1 **0.15**, L2T **0.01**, Playground **0.0**, IsaacLab velocity task **absent** (I) | **Standing frozen** is the optimal policy for alive-alone; and in Gym v2–v4 it was even paid on the terminal step (G issue #526) |
| **Termination penalty** | price the fall | policies that happily fall to restart | `is_terminated · w`, **-200** (IsaacLab G1/H1, applied on `terminated` only, **not** on time-outs, IG + `termination_manager`), **-100** (Playground) | Too large → the policy becomes maximally conservative (frozen); it also inverts the sign of exploration, since falling early *stops* the penalty stream that accrues while standing |
| **Undesired contacts** | keep non-foot bodies off the floor/opponent | knee-dragging, belly-crawling, shouldering | count of bodies with `max_t‖F_net‖ > threshold` (1 N), weight **-1.0** (base cfg), removed in IsaacLab G1 (IG) | Counts *our own robot's* self-collisions in a two-robot scene unless filtered by geom pair (our `ContactCategorizer` already exists for that: `src/rl/privileged.py`) |
| **Reference imitation** (D/A) | teach a *style* and a legible technique | unnatural gaits, mimes-on-the-ground | `r^I = .65·r_pose(exp[-2Σ‖Δq‖²]) + .1·r_vel(exp[-0.1Σ‖Δq̇‖²]) + .15·r_ee(exp[-40Σ‖Δp‖²]) + .1·r_com(exp[-10‖Δp_c‖²])`, total `r = ω^I r^I + ω^G r^G`, `ω^I=0.7/ω^G=0.3` (D) | Imitating a *bad* reference teaches the bad motion; the phase clock is linear in time, so the policy cannot time-warp (D §11) |
| **Adversarial style** (A) | match a data *distribution* without exact tracking | distribution-mismatch across many clips | `r_style = max(0, 1 - 0.25(D-1)²)`, `D` a least-squares GAN with gradient penalty `w_gp=10`, reward `0.5·task + 0.5·style` (A) | Mode collapse onto a subset of clips; without velocity features the character "holds a fixed pose on the ground" (A §9) |

### 2.2 Terms that reward falling, and terms that reward standing frozen — flagged

These are the four traps that matter for M1/M2. Two are *bugs in shipped code*, two are *structural*.

1. **Penalty-only rewards reward falling.** Penalties are paid *while the episode runs*. Terminate on
   the fall and the robot stops paying everything: torque, action-rate, joint-velocity, orientation.
   RMA states it plainly: *"If we naively train our agent with the reward function aggregating all the
   penalty terms, it **learns to fall** because of the penalty terms. To prevent this collapse … apply a
   small multiplier `k_t` to the penalty terms"* (RMA Fig. S3, https://arxiv.org/abs/2107.04034).
   **Check**: `Σ|penalty|` per step while upright must be small compared with the positive per-step
   terms (alive + upright + task). With `-200` termination this is subtle: the fall penalty is a *one-off*,
   the penalties are *recurring*.
2. **Alive-bonus-only rewards standing frozen.** Gym Humanoid's `healthy_reward = 5.0/step` against a
   forward term worth `1.25/step` at 1 m/s makes surviving worth ~4× more than walking (G) — the score
   is dominated by not falling, so a static policy is near-optimal *unless pushes force motion*.
   Gym v2–v4 made it worse by paying the bonus on the unhealthy terminal step too (`healthy_reward =
   float(is_healthy or terminate_when_unhealthy) * healthy_reward`), a bug fixed in v5 (G issue #526).
3. **HumanoidStandup-v5's `uph_cost` is an absolute-height reward with no termination**: `reward =
   z/dt_sim - ctrl - impact + 1`, `dt_sim = 0.003` → the coefficient is effectively **333·z** and the
   env "never terminates" (G source). Lying still is worth ≈ +1/step, standing is worth ≈ +470/step, and
   **nothing pays for staying up** — the documented goal ("stand up *and then keep it standing*") is not
   in the reward. Do not copy it.
4. **dm_control `stand` deliberately rewards frozen standing**: `reward = small_control · stand_reward ·
   dont_move`, with `dont_move = tolerance(com_horizontal_velocity, margin=2)` (M). That is correct for
   *standing*, catastrophic for *balance*: it makes the policy refuse the steps needed to recover.
   van Marum measured the industrial version of this: a clock-based reward "incentivizes low foot
   velocities in standing mode, which directly impedes disturbance rejection capabilities" (VM).

Corollary for M1: the *only* thing that distinguishes a balance policy from a statue is the
**perturbation battery** (§3). No reward term can detect a statue; a held-out push can.

---

## 3. Perturbation curricula and evaluation batteries

### 3.1 How pushes are applied (and the units trap)

Report **force, duration, application point, and direction** together; a bare "push of 1.5" is
meaningless. Mass-normalize impulses (`N·s/kg`) because published robots range 25–137 kg.

| protocol | force | duration | point | schedule | source |
|---|---|---|---|---|---|
| IsaacLab velocity DR | velocity offset ±0.5 m/s | one step | base | every 10–15 s | I (`push_by_setting_velocity`) |
| unitree 12-DoF G1 | velocity offset ≤1.5 m/s | one step | base | every 5 s | https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/legged_gym/envs/g1/g1_config.py |
| MuJoCo Playground G1 | 0.1–2.0 (velocity offset) | one step | base | every 5–10 s | P |
| L2WIM (ANYmal) | ±1 m/s | one step | base | every 10 s | L2WIM |
| Ferigo (iCub 33 kg) | **200 N** | **0.2 s** (= 40 N·s = **1.21 N·s/kg**) | base frame, random direction on a sphere | fixed magnitude, sampled frequency | F |
| HiFAR (T1) training | 0–10 (normalized) | 1 s | trunk, local frame | every 5 s + root-velocity "kicks" every 2 s | https://raw.githubusercontent.com/Hi-FAR/HiFAR/main/envs/T1FallRecoveryRandom.yaml |
| HiFAR evaluation | 50–300 N | 0.2 s | torso, forward/back/lateral | fixed grid, 10 repeats | HI |
| Duburcq (Atalante) | peak **800 N** | 0.4 s | pelvis | every 3 s ± 2 s jitter | DU |
| van Marum (Digit) training v2 | **20–200 N** × **200–500 ms** sampled *jointly* | same | rope, fixed height | random | VM |
| van Marum benchmark | 79–214 N × 200–500 ms | same | rope, 122 cm | full grid, 5 trials/cell, early stop at first failure | VM |
| Push-and-Step training | **70–200 N** | **0.7–1.3 s** (49–260 N·s) | shoulder/upper back | random, all directions | PS |
| Push-and-Step battery | 50–300 N × 16 directions | as above | — | 80 trials | PS |
| H1-2 | 50–300 N (training), 0–500 N (eval) | 0.1–0.2 s | torso | 5 poses × 4 directions, 10 000 trials | H12 |
| HWC-Loco | ±200 N **and** ±200 N·m on *every* link | continuous | all links | resampled at 1 Hz | https://arxiv.org/html/2503.00923v1 |

### 3.2 Magnitude schedules: what is standard, and one documented anti-pattern

- **Ramped / curriculum force is NOT universally right.** Duburcq et al. explicitly report that
  gradually increasing the push from 0 to 800 N over the episode caused **premature convergence to a
  suboptimal policy**, and they switched to a constant peak magnitude with jittered timing (DU).
  Conversely, HiFAR's multi-stage curriculum and H12's "anneal torque limits 10× → 1×" are ramp-shaped
  (HI, H12). The distinction is *what* is ramped: ramping the **task difficulty** (fall-recovery from
  harder initial states) works; ramping the **disturbance amplitude inside an episode** does not.
- **Joint sampling beats a single worst-case push.** van Marum: sampling force *and* duration jointly in
  3D eliminated every gap their benchmark had exposed (VM). A max-magnitude-only DR leaves
  duration/direction holes.
- **Timing jitter, not just magnitude.** Duburcq: every 3 s ± 2 s. IsaacLab: interval 10–15 s. Playground:
  5–10 s.
- **Small, frequent pushes have a use.** HuB (G1 extreme balance) deliberately uses high-frequency small
  root-velocity offsets (up to 0.5 m/s every 1 s) instead of large rare forces, arguing that large
  perturbations "fail to capture the subtle instability dynamics" of single-leg balance
  (https://arxiv.org/html/2505.07294v1).
- **Ramp the *evaluation* magnitude, not the training magnitude** is the safest default here: train at a
  fixed band, then sweep beyond it.

### 3.3 What magnitude should we actually expect to matter? (analytic scale)

Yang et al. give the closed form for the largest impulse a **non-stepping** (ankle/hip only) policy can
reject (Y, from capture-point theory):

```
J_reject = m · sqrt(g / z_com) · Δ_COP
```

For our G1 `[INFERENCE]`: `m = 33.34 kg` (`reports/2026-10-07/g1_model.md`), `z_com ≈ 0.6 m`,
`Δ_COP ≈ 0.10 m` (CoM to the forward edge of the foot support):

```
J_reject ≈ 33.34 · sqrt(9.81/0.6) · 0.10 ≈ 13 N·s  ≈ 0.40 N·s/kg
```

Yang report 53 N·s sagittal / 78 N·s lateral for a 137 kg Valkyrie = **0.39 / 0.57 N·s/kg**, i.e. our
estimate lands on their theory. So: **≈13 N·s is the ankle-strategy ceiling for us; anything above that
must be answered with a step.** Yang's learned TRPO policy rejected 240 N·s sagittal (4.5× the analytic
limit) — stepping is where the headroom is. Strategy bands for an H1-2-class robot (≈ our scale):
50–100 N → ankle, 100–200 N → hip, 200–300 N → stepping, ≥300 N → multi-contact bracing (H12).

Our own scene numbers `[INFERENCE from R]`: ankle pitch/roll torque limit **±50 N·m**
(`reports/2026-10-07/g1_model.md`); the measured STANCE crouch already needs **39/50 N·m** ankle torque
(`docs/MOTOR_CURRICULUM.md` §0) — i.e. a crouched wrestling stance has almost **no** ankle authority left
for disturbance rejection. Balance from STANCE will therefore be step-based, not ankle-based. This is a
design conclusion, not a detail.

### 3.4 Evaluation batteries and metrics (what to report)

| metric | definition | reference values / criterion to imitate |
|---|---|---|
| **Recovery success rate vs magnitude** | fraction of pushes with no fall within a fixed window | Ferigo: success = still standing **7 s** after the push, 12 directions × 5 repeats, grid 50–700 N in 25 N steps (F). H12: success = **CoM > 0.85 m and stable ≥ 1 s** (not a transient bounce), 93.4% over 10 000 trials, mean recovery 5 s (H12) |
| **Held-out magnitudes** | train band ⊂ eval band | HiFAR: trained ~200 N, evaluated to 300 N; success 100% ≤200 N, 80% @250 N, 0% @300 N prone (HI). Ferigo trained ≤200 N, succeeded to 300–400 N (F) |
| **Directional asymmetry** | lateral ≠ sagittal | HiFAR lateral is much harder (prone: 10% @250 N vs 80% sagittal) (HI); Yang lateral non-stepping limit *higher* but stepping harder (Y) |
| **Max recoverable push** | averaged magnitude the policy stays upright against | Push-and-Step: **230 N** with the balance reward vs **21 N** without (a 10× gap) — the single best "is the balance term doing anything?" ablation (PS) |
| **Time-to-stability** | time from push to a quiet upright state | H12 mean 5 s, up to 9 s for supine at high force (H12) |
| **CoM margin / excursion** | CoM (or capture point) distance to the support-polygon edge; peak displacement | HECTOR V2: CoM y displacement 0.11 m vs 0.26 m for a slow-gait baseline (https://arxiv.org/pdf/2409.14342); margin is also the analytic `Δ_COP` in §3.3 |
| **Heading deviation / foot slide / kinetic energy** | recovery *quality*, not just survival | Push-and-Step: 5.93° / 22 cm / 933 J full policy; 44.81° / 24 cm / 2381 J without imitation; 13.19° / 49 cm / 1444 J without the quality terms (PS) |
| **Number of steps / attempts to recover** | did it step, and how often | Yang: 24 N·s → ankle+hip; 72 N·s → +foot tilt; 240 N·s → +stepping (Y); Berkeley: 1.16 vs 3.65 get-up attempts (DAgger vs baseline) (https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-87.pdf) |
| **Endurance under repeated pushes** | consecutive disturbances survived | Ferigo: 60 s episodes with a push every ~3 s (≈20/episode), 50 episodes per cell; policy withstood on average **9 consecutive 300 N / 0.2 s** pushes (F) |
| **Robustness sweeps** | load / friction / torque-limit grid | HiFAR: +80% torso mass → 100%; friction 0.2 → 90–100%, **friction 0.1 → 0%/90%**; torque limit 75% → 100%/60% (HI) |
| **Anti-statue check** | recovery under a push the *training* set never contained | required by `MOTOR_CURRICULUM.md` §3 (gates never reward, only measured); Ferigo's out-of-distribution success is the template (F) |

Push-and-Step's ablations are the cleanest evidence that balance-specific terms are load-bearing: remove
the CoM/CoP reward and the maximum recoverable push collapses 230 N → 21 N (PS). The same conclusion
appears as a full failure to learn in H12's ablation (`[INFERENCE]` named "balance-informed structure":
capture point / CoM / centroidal momentum as critic inputs + shaping): **93.4% with it, failure to learn
stand-up without it** (H12).

---

## 4. Initialization strategies on a CPU budget

### 4.1 What our machine actually does (measured, not assumed)

| measurement | value | source |
|---|---|---|
| Env steps/s, 3 subproc workers, our two-robot wrestling env | **421–437** raw; **222.7** end-to-end incl. PPO updates; 172–437 across load | `reports/2026-10-08/rl_infra.md` §5, `notes.md:144-156` |
| Single env, sequential | ~290 steps/s | `reports/2026-10-08/rl_infra.md` §5 |
| MuJoCo control steps/s on **this host**, gym-humanoid model (24 DoF, 10:1 decimation, 256×256 MLP), own benchmark run during this survey | **344** (1 process), **743** across 4 processes — consistent with our env's 222–437 | own measurement (not prior art) |
| Net size | actor ≈97k params, critic ≈131k, 256×256 tanh | `notes.md:352`, `src/rl/net.py` (`NetConfig.hidden`) |
| Physics/control | 500 Hz physics (0.002 s), **50 Hz control** (decimation 10) | `src/wrestling/env.py` (`MODEL_DT`, `SUBSTEPS`, `CONTROL_HZ`) |

**Arithmetic `[INFERENCE]`** (steps ÷ steps/s ÷ 3600):

| env steps | @200 steps/s | @300 | @400 | @600 |
|---|---|---|---|---|
| 1M | 1.4 h | 0.9 h | 0.7 h | 0.5 h |
| 5M | 6.9 h | 4.6 h | 3.5 h | 2.3 h |
| 10M | 13.9 h | 9.3 h | 6.9 h | 4.6 h |
| 50M | 69 h | 46 h | 35 h | 23 h |
| 100M | 139 h | 93 h | 69 h | 46 h |
| 150M (IsaacLab G1 flat) | 208 h | 139 h | 104 h | 69 h |
| 400M (Playground G1 joystick) | 556 h | 370 h | 278 h | 185 h |

Two independent anchors confirm the bounding box:
- DeepMimic: "**about 60 million samples … about 2 days on an 8-core machine. All simulation and network
  updates are performed on the CPU and no GPU acceleration is used**" (D §A4; same in the code README).
  60M/2 days/8 cores ≈ **350 steps/s/core-cluster** — our box is in the same class, so a DeepMimic-scale
  skill is ~**4 days** here `[INFERENCE]`.
- Isaac Gym reports its `39M`-sample AMP run taking ~6 min on GPU versus **~30 h on 16 CPU cores** in the
  PyBullet reference implementation (ISAACG §6.2) — i.e. ~360 steps/s CPU, matching the DeepMimic anchor.
  Isaac Gym's headline 2–3 orders of magnitude is precisely the gap we cannot close.

### 4.2 What initialization buys (the part that actually rescues us)

| technique | evidence | our cost |
|---|---|---|
| **Reference-state initialization (RSI)** — start episodes from a distribution over the target pose/motion, not a fixed state | DeepMimic: RSI+ET vs RSI-only: Backflip .791/.379, Sideflip .823/.355, Spinkick .848/.358; without RSI the backflip policy "never learns a full mid-air flip" (D Table 5, §10.4) | free — sample `stand` ± noise at reset; our env already jitters ±0.03 rad and ±0.15 m (`src/wrestling/env.py`) |
| **Early termination** | same table; ET removes the "falls and mimes the motion on the ground" local optimum (D §10.4) | one predicate in the env (M1 needs it *anyway*) |
| **Residual-on-reference actions** | standard in a decade of follow-up work; our own `action_mode="residual"` with `residual_scale=0.5` already exists and smoke-tested (`notes.md:177`) | zero — flip a config flag |
| **Assisted exploration / keyframe curriculum** | HoST: removing the 60%-gravity vertical pull force → success **99.5% → 0.0%** (arxiv 2502.08378 Table III). Berkeley: two-stage keyframe curriculum + init sampling 0.4 fallen / 0.4 kneeling / 0.2 standing | cheap, but a *new* mechanism per task |
| **Privileged / asymmetric critic** | full-state critic with partial-obs actor "significantly improves performance" (arxiv 1710.06542); standard in every humanoid line | **already built**: `PRIV_DIM=162`, critic input 92+162=254 (`src/rl/privileged.py`) |
| **Potential-based shaping** | `F = γΦ(s') - Φ(s)` leaves the optimal policy unchanged (N99): safe to add progress/assist terms | free — our `shaping()` already uses the potential form for `progress` (`src/rl/reward.py`) |
| **Curriculum over commands** | IsaacLab samples 2% standing envs and resamples every 10 s (I); Playground finetunes 100M flat → +50M wide yaw → +100M rough (PG) | free; and it is the only way to fit M2 into budget |
| **Domain randomization** | see §4.3 | free |
| **Symmetry / mirror loss** | used in legged RL (w=4 in arxiv 1801.08093) with qualitative gait improvements | cheap (reflect obs/actions on left/right) but an added code path |

### 4.3 Randomization ranges that are standard (port the modest end)

| parameter | range used in published work | source |
|---|---|---|
| ground friction | `[0.5, 1.25]` (L2WIM); `[0.1, 1.25]` (unitree G1); `[0.05, 4.5]` train / `[0.04, 6.0]` test (RMA, quadruped) | L2WIM, unitree G1 cfg, RMA Table I |
| base/added mass | `±5 kg` (IsaacLab `add_base_mass`); `[-1, +3] kg` (unitree G1); link mass `[0.25, 4]×` (PD18); `±15%` (HK19) | I, unitree G1 cfg, PD18 Table I, HK19 |
| CoM offset | `x,y ±0.05 m, z ±0.01 m` (IsaacLab); `±2 cm` (HK19); `±0.15 cm` (RMA — quadruped, much smaller) | I, HK19, RMA |
| motor strength / gains | `[0.90, 1.10]` (RMA); `[0.5, 2]×` position gains (PD18) | RMA, PD18 |
| action latency | `dt = dt0 + Exp(λ)`, `dt0=0.04 s`, `λ ∈ [125, 1000]` s⁻¹ (PD18); 1-step delay with p=0.5 (OpenAI in-hand, https://arxiv.org/abs/1808.00177); MuJoCo exposes `nsample` to *observe* the delay (MJ) | PD18, MJ |
| observation noise | joint pos ±0.01 rad, joint vel ±1.5 rad/s, base lin vel ±0.1 m/s, base ang vel ±0.2 rad/s, gravity ±0.05 (IsaacLab); joint vel ±0.5 rad/s, base lin ±0.08, base ang ±0.16 (HK19); Playground G1: joint pos 0.03, joint vel 1.5, gravity 0.05, linvel 0.1, gyro 0.2 | I, HK19, P |
| foot friction floor | do **not** randomize below ≈0.3: HiFAR's friction sweep collapses (0%/90% success at μ=0.1) | HI |

Our scene ships `friction 0.6` on foot spheres (priority 1, condim 3) and `0.7` on the floor
(`robots/wrestling_scene.xml:24,126`).

### 4.4 What is realistically achievable here, and what is not

**Achievable (per §4.1 + §4.2):**
- **M1 balance from the `stand` keyframe, with pushes:** a few hours (2–30M steps). The reward is dense,
  the horizon is short, the initial pose is already statically stable, and Yang et al. did the equivalent
  (standing push recovery, workspace humanoid) **on a single CPU in 2 days** with a 100-50-25 MLP (Y).
- **M1 from the STANCE crouch with stepping recovery:** plausible but the harder half — see §3.3
  (39/50 N·m ankle torque already spent) and §4.2 (assisted exploration / RSI may be required).
- **M2 crude velocity tracking** (forward 0–0.6 m/s, mild turn): 1–3 days (30–100M steps). The reward set
  is dense and standard; what we lose versus IsaacLab is gait *quality*, not existence.
- **Evaluation batteries**: free. A 1000-push battery at 3 s each is 50 min of sim per policy at 300
  steps/s `[INFERENCE]` — cheap enough to run at every gate.

**Not achievable (say so explicitly, per `MOTOR_CURRICULUM.md` §3 anti-self-deception):**
- Anything in the 147M–400M-step class *as published* (IsaacLab G1 147M, G1 rough 295M, Playground G1
  joystick 400M, HoST ≥2.4B cap) — that is 5–60 days here (`ISAACG`, `PG`, `IG`, HoST).
- Rough terrain, stairs, height-scan observation (187 rays), parkour, visual policies.
- Adversarial style training at AMP/ADD scale (750M–850M steps on A100s, ADD paper).
- Reward-weight grid search. With a few runs per day, the budget allows ~5–10 training runs per stage —
  choose weights from the published tables (§2, §6) and spend the runs on *gates*, not on tuning.

---

## 5. Failure-mode checklist for our harness

Ordered by likelihood, each with the mechanism and the check we can run in *this* code.

1. **Penalty trap → the policy learns to fall.** Mechanism: penalties accrue only while upright (§2.2.1).
   Check: print `Σ|penalty|` and `Σ|positive|` per control step from the reward function
   (`src/rl/reward.py`) at episode start; require the positive side to dominate. If the policy improves
   while ground-time rises, this is the cause (`notes.md` E1: 83% ground time).
2. **Frozen statue.** Mechanism: alive bonus + tight action-rate/torque/slip penalties; a crouched stance
   with no ankle authority invites it (§3.3). Check: the held-out push battery — **not** the training
   return. Also track CoM travel over a 60 s idle hold (`MOTOR_CURRICULUM.md` M1 gate) and the
   quotient `recovery_rate(training pushes) / recovery_rate(unseen pushes)`.
3. **Fall-forward velocity tracking.** Mechanism: the tracking term is satisfied during the dive.
   Checks: uprightness gate ≥99% *while moving*; velocity-tracking error must be paired with a
   step-quality metric; and the M2 episode must not be allowed to end early — `MOTOR_CURRICULUM.md` §3.4
   is exactly this rule, and our E1 baseline ("leaning forward and falling with style") is its measured form.
4. **Hopping / double support / marching in place.** Mechanism: air-time without a single-stance or
   phase constraint; air-time without a command gate. Sources: SaW §IV-B (hopping), I (`>0.1` command
   gate), van Marum (step-frequency/clock interactions). Check: per-episode stance-phase histogram,
   flight-time fraction, and air-time reward value while `‖cmd‖≈0` (must be exactly 0).
5. **Contact/penetration exploits.** MuJoCo's soft contacts permit finite penetration and **do not
   guarantee a zero-velocity stick state**: "a persistent tangential load can produce a nonzero steady
   slip velocity", especially with pyramidal cones; the remedies are elliptic cones with `impratio` and
   the NoSlip solver (`noslip_iterations 1–3`, default 0) (MJ). Also `solref` default stays "0.02 1"
   with the constraint `timeconst ≥ 2·dt` (MJ). Checks: log `data.ncon`, minimum contact distance,
   per-foot contact normal force, and `Σ‖v_foot‖·1[contact]` (feet_slide). Do **not** reward low contact
   force (the Gym `contact_cost`/`impact_cost` pattern) — it is satisfiable by hovering.
6. **Silent actuator clamping.** `ctrllimited` (default auto) clamps `ctrl` to `ctrlrange`, and joint-level
   `actuatorfrcrange` clamps the *realized* force; both happen inside `mj_step` with no error (MJ). Our
   actuators inherit the joint range (`inheritrange`), and the joints carry hard limits
   (hip 88, knee/hip-roll 139, ankle 50, shoulder/elbow/wrist-roll 25, wrist 5 N·m —
   `reports/2026-10-07/g1_model.md`). Our harness already counts out-of-`ctrlrange` actions
   (`count_out_of_bounds`, `src/rl/net.py`); **add** a log of `|data.actuator_force| / limit` so
   saturation is visible. A policy at 100% ankle saturation cannot recover from anything.
7. **Position-servo semantics.** `F = kp·(ctrl - q) - kv·q̇` with `kp=500, dampratio=1` (MZ). Therefore
   `ctrl²` is *not* torque, and `qfrc_actuator` is the joined result; torque/energy penalties must use
   `data.actuator_force`/`qfrc_actuator`, not the action (MJ, `src/wrestling/env.py` control contract).
8. **Quaternion sign flips in the observation.** Our `default_observation` feeds the **pelvis quaternion
   in world frame** (`self_base_quat`, `src/wrestling/env.py`, `_OBS_LAYOUT` + `default_observation`).
   MuJoCo's free-joint quaternion can
   jump between `q` and `-q` — the same rotation, but a discontinuous MLP input with no state change.
   Replace it with **projected gravity** (3 numbers), which is what IsaacLab and Playground use (I, P).
   Our obs also carries `torso_up` (the torso body +z axis expressed in *world coordinates*,
   `data.xmat[tid][:,2]`), which is the same information as projected gravity but in the opposite frame;
   pass `R^T·ẑ` (gravity in the body frame) rather than `R·ẑ` so uprightness is readable without knowing
   the world frame, and never feed both.
9. **Termination semantics.** Two traps in one: (a) our env terminates only on the **match clock**
   (`src/wrestling/env.py`, `WrestlingEnv.step`) — M1 has no fall termination at all, and `backdet` is the *wrestling*
   terminal rule (dorsal torso + low pelvis), which would fire late for a balance task; (b) IsaacLab's
   `-200` is applied on `terminated` but **not** on time-outs (`termination_manager`), i.e. truncation is
   never penalized. Checks: implement M1 fall termination (pelvis height / tilt, *persistence-confirmed*
   per `MOTOR_CURRICULUM.md`), and keep the penalty off the truncation path.
10. **Two-robot scene coupling.** The scene has no `<pair>`/`<exclude>` elements and the menagerie
    collision geoms keep default contype/conaffinity, so **robots a and b collide by default**
    (`robots/wrestling_scene.xml`); `both_stand` puts them 0.7 m apart. `MAT_RADIUS = 1.5 m` with
    `OOB_EVENTS_TO_FORFEIT = 3` means a walking robot that leaves the circle forfeits the exchange
    (`src/wrestling/env.py`, `MAT_RADIUS` / `OOB_EVENTS_TO_FORFEIT`). For M1/M2 the opponent must be either a *verified-static* partner
    (`StandHold` is the one pose that stays up open-loop — `src/wrestling/env.py` docstring) or the
    scene reduced to one robot, with OOB/backdet events disabled for the stage.
11. **Reward/obs scale bookkeeping.** Our `RewardWeights` has no balance terms at all
    (`technique_similarity, progress, outcome, oob, engagement` — `src/rl/reward.py`, `RewardWeights`), and
    `shaping()` takes only `(technique, phase, phase_next, qpos_self, qpos_opp)`. M1/M2 need new terms
    and access to `qvel`, `data.actuator_force`, and contact state → extend the signature (or add a
    parallel `BalanceReward`), and keep one dt convention (§2 preamble).
12. **Discount/rate interaction.** At 50 Hz, `γ=0.99` gives an effective horizon of ~100 steps = **2 s**
    (`0.99^100 = 0.366`). IsaacLab/Playground use `γ=0.99` at 50 Hz; DeepMimic used `γ=0.95` at 30 Hz (D).
    Our PPO default is `γ=0.99` (`src/rl/ppo.py`, `PPOConfig.gamma`) — correct, but a fall 3 s in the future is
    almost free: the *termination penalty* is what carries long-horizon falls, not the discount.
13. **Determinism of the battery.** Our resets derive from `seed + i*EXCHANGE_SEED_STRIDE`
    (`src/wrestling/env.py`, `EXCHANGE_SEED_STRIDE`); the push schedule must be seeded the same way, or the "held-out" claim
    is not reproducible (`MOTOR_CURRICULUM.md` §5 requires reproducible configs).
14. **`implicitfast` + high `kp`.** Default integrator is `implicitfast` (MJ recommends it); with 500 Hz
    physics and `kp=500` the position feedback is stable, but `implicitfast` drops Coriolis derivatives
    from `M̂` (it only reinstates them for standalone free bodies), so a floating-base humanoid is outside
    that special case (MJ). Do not switch to `discrete`/`RK4` casually; the model is verified at this
    configuration (`reports/2026-10-07/g1_model.md`).
15. **Initializing M1 from STANCE.** STANCE is *not* open-loop holdable (topples in ~1.4 s,
    `src/wrestling/env.py` docstring; `reports/2026-10-08/wrestling_env.md`). Start M1 at the `stand`
    keyframe (verified 5 s stable, 8 foot contacts) and treat "balance in STANCE" as the M1 *hard* variant.

---

## 6. Recipes

Both are **STARTING POINTS**. Weights are from the cited tables, adapted by unit inspection; the
*relative* magnitudes are the load-bearing part, the absolute scale is conventional. Validate by the
gates, not by the return. In both recipes: per control step, 50 Hz, `dt = 0.02 s`, net 256×256 tanh,
PPO as configured (`rollout 2048`, `n_envs 3`, `γ=0.99`, `λ=0.95`, `clip=0.2`, `lr=3e-4→linear→0.1×`,
target KL 0.03), privileged critic kept.

### 6.1 M1 — dynamic balance (one page)

**Harness changes required (prerequisites, not optional):**
1. **Fall termination**: `terminated = (pelvis_z < 0.45 m) OR (torso_up_z < 0.5)` sustained ≥ 5 control
   steps (0.1 s). Knees/hands are NOT terminal (MISSION rule: knees/hands are legal).
   Rationale: `MOTOR_CURRICULUM.md` M1 termination + persistence-confirmed; Playground G1 uses
   `gravity_z < 0`, gym uses a `z` band (P, G) — ours is the persistence-confirmed mixture.
2. **Push injection**: `data.xfrc_applied[pelvis_body] = F·dir` for `duration` steps, seeded, with a
   schedule (below). This is the standard mechanism (MUJOCO docs: external Cartesian forces live in
   `xfrc_applied`; `apply_rigid_body_force_tensors` in Booster Gym is the analogue).
3. **Scene**: single-robot variant (recommended) or two-robot with opponent = `StandHold` parked
   ≥2 m away and OOB/backdet events disabled (§5.10).
4. **Reward plumbing**: extend `RewardWeights`/`shaping` with the terms below (§5.11).

**Observation (actor) — replace world quaternion with gravity-aligned features:**
`projected_gravity` (3) · `base_lin_vel` (3) · `base_ang_vel` (3) · `joint_pos_rel` (29) ·
`joint_vel` (29) · `last_action` (29) · `foot_contact` (2: left/right) · `foot_force_lr` (2, from
`data.contact`) · **`com_xy - support_center_xy` (2)** · **`capture_point_xy - support_center_xy` (2)** ·
`time_since_last_push` (1). ≈ 106 dims, frame-stacked ×2 ⇒ ~212. Privileged add: the push vector
(F, dir, remaining duration), actuator-force saturation vector, true CoM/CoM-velocity
(extend `src/rl/privileged.py`'s 162-dim layout).

**Actions:** residual on the `stand` keyframe targets, `action_mode="residual"`, `residual_scale=0.5`
(existing). Add an **action-rate limit** (clip per-step change to ≤0.15 rad per joint `[INFERENCE]`,
0.3 rad seems large at 50 Hz) — until the policy learns to move, this is the cheapest anti-chatter device.

**Rewards (per step while alive; the positive side must dominate):**

| term | weight | formulation | anchored by |
|---|---|---|---|
| alive | **+1.0** | `1.0 · 1[not terminated]` | Gym `5.0`, unitree `0.15`, L2T `0.01` — scale so positives dominate |
| upright | **+1.0** | `tolerance(up_z, bounds=(0.9,∞), margin=1.9, value_at_margin=0)` | dm_control `stand` (M); equivalently `-2.0·Σpg[:2]²` (P) |
| height | **+0.5** | `exp(-10·(pelvis_z - 0.79)²)` | DWL `φ(z-0.7,10)·0.5`; `stand` keyframe z = 0.79 |
| capturability | **+0.5** | `clip(0.05 - dist(capture_point, support_polygon), 0, 0.05)/0.05` — 1 when the capture point is ≥5 cm inside, 0 when it exits | PS (CoM/CoP momentum reward, `w_g=0.2`), PR (capture point) |
| lin_vel_z | **-0.2** | `vz²` | IG flat |
| ang_vel_xy | **-0.05** | `Σω_xy²` | I |
| joint_vel (idle) | **-1e-3** | `Σ q̇²` | unitree G1, L2T |
| action_rate | **-0.005** | `Σ(a_t - a_{t-1})²` | IG |
| torque | **-1.5e-7** | `Σ actuator_force²` (hips/knee/ankle) | IG (G1 rough) |
| dof_acc | **-1.25e-7** | `Σ q̈²` (hips/knee) | IG |
| ankle limits | **-1.0** | `Σ clip beyond soft limits` (ankle pitch/roll only) | IG |
| actuator saturation | **-0.5** | `Σ max(0, |τ|/τ_max - 0.9)` | `MOTOR_CURRICULUM.md` M1 row |
| termination | **-100** | once, on fall | Playground (P); IsaacLab uses -200 |
| **penalty budget rule** | — | `Σ|penalties|` at the nominal pose ≤ ~0.05/step, i.e. ≤5% of the alive bonus | RMA Fig. S3 (§2.2.1) |

**Push schedule (3 phases, all seeded; force applied at the pelvis for 0.2 s, direction uniform on the
circle):**
- P1 (0–1M steps): **no pushes.** Gate: `stand` held ≥30 s without fall termination, CoM drift <5 cm.
- P2 (1–10M): magnitude uniform in **5–25 N·s** (0.15–0.75 N·s/kg; 25–125 N for 0.2 s), one push every
  2–4 s, direction uniform; 20% of resets start from a pushed state (RSI on the disturbance).
- P3 (10–30M): magnitude uniform in **10–40 N·s** (up to 1.2 N·s/kg, Ferigo's training point), plus
  torque-level randomization (75–100% of limit, cf. H12's 10×→1× anneal).
- **Do not** ramp magnitude within an episode (Duburcq's anti-pattern, DU).

**Eval battery (held out; seeded; run at every gate):**
- 16 directions × magnitudes {0, 10, 20, 30, 40, **60**, **80**} N·s, 5 repeats (80–560 trials),
  applied at t = 1.0 s after a quiet start, 0.2 s duration, at the pelvis.
- Success = no fall within **5 s** AND `up_z > 0.9` with `|CoM - support| < 2 cm` sustained 0.5 s
  (H12's "stable ≥1 s, CoM > 0.85 m" adapted).
- Report: recovery-rate-vs-magnitude curve, first-failure magnitude, **max recoverable push** (PS),
  time-to-stability (H12), peak CoM excursion, heading deviation, foot slide, number of recovery steps
  (Y's strategy bands), and the CoM-margin at t = 3 s (PR).
- Idle hold: 60 s, zero pushes, zero falls, and record CoM travel (the statue detector).
- Gate (placeholder, per `MOTOR_CURRICULUM.md`): ≥90% recovery at ≤25 N·s, ≥50% at 40 N·s, 0% falls on
  the 60 s idle hold.

### 6.2 M2 — velocity-commanded locomotion (one page)

**Harness changes**: as M1, plus a command generator and (optionally) a single-robot scene.
**Command**: `(vx, vy, ωz)` resampled every **10 s** with 10% standing entries; start narrow:
`vx ∈ (0, 0.6)`, `vy ∈ (-0.2, 0.2)`, `ωz ∈ (-0.5, 0.5)` rad/s; widen to IsaacLab's G1 flat ranges
(`vx (0,1)`, `vy (-0.5,0.5)`, `ωz (-1,1)`) only after the gate. IsaacLab uses 10 s resampling, 2% standing,
`heading_command` at stiffness 0.5 (I) — the heading form is what keeps yaw well-posed.

**Observation**: M1's set **+ command (3)** + `gait_phase (sin, cos)` (1 clock at ~0.68 s per cycle, L2T)
+ `base_lin_vel` in the yaw frame instead of world. Frame-stack ×2.

**Rewards — port the IsaacLab G1 flat set verbatim as the starting point, then add the balance floor:**

| term | weight | formulation | source |
|---|---|---|---|
| track_lin_vel_xy (yaw frame) | **+1.0** | `exp(-‖cmd_xy - v_yaw_xy‖²/0.25)` | IG (std 0.5) |
| track_ang_vel_z (world) | **+1.0** | `exp(-(cmd_ω - ω_z)²/0.25)` | IG flat 1.0, rough 2.0, H1 1.0, P 0.75 |
| termination | **-100 … -200** | once, on fall (not on timeout) | P / IG |
| feet_air_time (biped, single-stance, clamped, cmd-gated) | **+0.5** | `clamp(min over single-stance time, 0.4) · 1[‖cmd_xy‖>0.1]` | IG flat 0.75, rough 0.25 |
| feet_slide | **-0.2** | `Σ‖v_foot_xy‖·1[contact]` | G1 -0.1, H1/P -0.25 |
| flat_orientation (upright gate) | **-1.0** | `Σ projected_gravity_b[:2]²` | IG, P (-2.0) |
| lin_vel_z | **-0.2** | `vz²` | IG flat |
| ang_vel_xy | **-0.05** | `Σω_xy²` | I |
| yaw-rate guard | **-0.1** | `ω_z²` | L2T/DWL pattern |
| action_rate | **-0.005** | `ΣΔa²` | IG |
| dof_acc | **-1.25e-7** | `Σq̈²` (hips/knee) | IG |
| torque | **-2e-6** | `Σ actuator_force²` (hips/knee) | IG flat |
| ankle limits | **-1.0** | soft-limit excess | IG |
| joint_deviation | **-0.1** hips-yaw/roll, **-0.1** arms, **-0.1** torso | `Σ|q-q_default|` | IG |
| stand_still | **-1.0** | `Σ|q-q_default| · 1[‖cmd_xy‖<0.06]` | P; I's `stand_still_joint_deviation_l1` |
| alive | **+0.15** | `1[not terminated]` | unitree G1; keep small — positives come from tracking |
| gait phase (optional) | **+0.5** | L2T contact-matching: `+c1` if the correct leg is loaded | L2T (2.0) |
| **keep from M1**: capturability | **+0.25** | as M1 | the M1/M2 continuity term; also the M3–M4 retention hook |

**Perturbations/DR for M2**: friction `U[0.5, 1.25]` (L2WIM), added base mass `U[-1, +3] kg` (unitree G1),
motor strength `U[0.9, 1.1]` (RMA), 1-step action delay with p = 0.5 (OpenAI in-hand; implement with a
1-step action buffer, or `nsample` to observe it), obs noise as IsaacLab/Playground (joint pos 0.01–0.03,
joint vel 1.5, gravity 0.05) — and **pushes during walking**: velocity-offset ±0.5 m/s every 5–10 s
(I/P), then forces as §6.1 P2 at a 15 s cadence.

**Curriculum**: (0) stand → (1) `vx ∈ (0,0.3)` only, `ωz = 0` → (2) full narrow command set → (3) widen
to IsaacLab flat ranges → (4) add pushes during walking → (5) friction/mass DR widened. This mirrors
Playground's 100M→+50M→+100M staging (PG) at ~5% of the samples, and IsaacLab's 20 s episodes /
50 Hz control (I) with our 20 s exchange timeout (`src/wrestling/env.py`) as the episode bound.

**Eval battery (held out commands, seeded):** 25 commands sampled from a grid *not* used in training
(speeds ×2 the trained max, unseen heading changes, decelerate-to-stop and stand-from-walk transitions)
× 10 s each. Metrics: mean/p95 linear-velocity error, yaw-rate error, **uprightness ≥99%**, foot slip
(cm/s while in contact), fall rate per 100 s (<1 per `MOTOR_CURRICULUM.md` M2 gate), stand-from-walk
success, drift while commanded to stand, and the anti-fall-forward pair (uprightness + step-quality,
never tracking alone — `MOTOR_CURRICULUM.md` §3.4). Forgetting check: re-run the **M1** battery after M2
convergence (the catastrophic-forgetting test, `MOTOR_CURRICULUM.md` §4).

**Ordering note**: M2 without M1 is the documented trap (our E1: 30k steps of PPO → 83% ground time,
13% standing, `docs/MOTOR_CURRICULUM.md` §0). Train M1 first, freeze or preserve it, then add the
tracking terms — this is also how the retention hook (`capturability`) is defined.

---

## Appendix: one-line answers to the questions this doc exists to answer

- *Can we train a walking humanoid on this box?* Only a crude one: 30–100M steps ≈ 1–3 days, a fraction
  of the 147M–400M Isaac/Playground budgets, with gait quality given up.
- *Can we train balance?* Yes, hours, because dense balance shaping + a stable start pose + short
  horizons put us in the regime Yang et al. did on one CPU in two days.
- *What replaces GPU throughput?* Reference-state initialization, residual actions, a privileged critic,
  potential-based shaping, curricula, and *evaluation batteries* instead of weight sweeps.
- *What is the one thing to get right?* The reward must pay for surviving and must not pay for dying
  (§2.2), and survival must be *proved* by held-out pushes, never by the training return.
