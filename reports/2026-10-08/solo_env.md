# S1 — solo-G1 environment + validation harness (docs/SOLO_DRILL.md step S1)

**Agent:** SoloEnv · **Date:** 2026-10-08 · **Repo:** `/home/ubuntu/grappling`
**Deliverables:** `src/solo/` (scene, commands, stance, pushes, fall, markers, obs, reward,
metrics, env, eval, baselines, video, lock, train), `tests/test_solo.py` (32 tests),
`scripts/solo_env_smoke.py`, `data/solo/metrics/`, `videos/solo_drill/baselines/`, this report.
**Status of the suite at hand-off:** `tests/test_solo.py` **32 passed** (~15 s).
Repo-wide there are 7 failures in *other agents' in-flight files*
(`tests/test_scorer.py` ×6 — STAND_UP scoring/calibration; `tests/test_wrestling.py` ×1 —
STAND_UP reference frame no longer puts `a_left_wrist_yaw_link`/`b_left_knee_link` on the mat).
Nothing under `src/solo/` is imported by those tests; they are unrelated to S1 and are reported,
not patched (they belong to the scorer/retarget/wrestling owners).

---

## 1. What exists now

```
src/solo/scene.py     single-G1 scene: a_-prefixed robot + floor + virtual-opponent markers
src/solo/stance.py    (width, height) -> 29 joint targets (measured geometry table)
src/solo/commands.py  (vx,vy,wz) + stance height/width + skill + lead leg; sampler/filter/schedule
src/solo/pushes.py    PushSpec/PushSchedule; qfrc_applied application; clear-every-step contract
src/solo/fall.py      FallDetector (pelvis+tilt+persistence) + DorsalDetector (single robot)
src/solo/markers.py   world-anchored shot entry region; phase selects the active target
src/solo/obs.py       fixed actor (115) / privileged (43) / critic (158) layouts
src/solo/reward.py    19 terms, 6 task sets, positive-survival invariant, per-term logging
src/solo/metrics.py   per-step metrics + aggregation + JSONL/JSON emission
src/solo/env.py       SoloEnv: 50 Hz control / 500 Hz physics, termination, info, metrics
src/solo/eval.py      the gating harness: TaskGate/Criterion, push battery, evaluate()
src/solo/baselines.py scripted baselines + the four exploit probes (+ random-init policy)
src/solo/video.py     960x720@30fps mp4 + 3-frame contact sheet + overlay
src/solo/lock.py      data/locks/sim.lock advisory lock (other agents can reuse)
src/solo/train.py     minimal single-env PPO built on rl.net/rl.ppo/rl.checkpoint
scripts/solo_env_smoke.py  CLI: smoke | holdability | throughput | probes | baselines | battery | all
```

## 2. Measured model + physics contract (verified, not assumed)

`src/solo/scene.py` composes the scene at load with `MjSpec` from
`robots/g1/g1.xml + retarget landmarks`, attaching the robot with the `a_` prefix
(the same mechanism the two-robot scene uses). Measured on the compiled model:

| quantity | value |
|---|---|
| nq / nv / nu | **36 / 35 / 29** (`nbody` 32, `nsite` 28, `ngeom` 77, `nkey` 1) |
| timestep | **0.002 s** (500 Hz, `implicitfast`), asserted at load |
| control step | **0.02 s = 10 substeps = 50 Hz** (`env.control_dt`) |
| total mass | 33.341 kg (markers are mass-0 geoms) |
| floor friction | 1.0 / 0.005 / 0.0001, condim 3 (scene default) |
| keyframe | `a_stand` (pelvis z 0.79, holds 5 s with <0.2 cm drift) |
| names | every robot body/joint/actuator/site/sensor/keyframe is `a_`-prefixed; `floor` stays unprefixed; markers `a_marker_{pelvis,leg_l,leg_r,hand_l,hand_r}`; `resolved_dependencies()` + a test assert the exact set |

The prefixing decision (Main, 2026-10-08) is what lets `wrestling.backdet.body_maps(model, "a")`
resolve without touching the two-robot modules — pinned by
`test_scene_dependency_names`.

## 3. Contract details (frozen for this milestone)

* **Action** — 29 joint-position targets (rad). `absolute`: clipped to `ctrlrange`.
  `residual`: `ctrl = clip(base + residual_scale*tanh(z))`; `base` = stand keyframe ctrl,
  overridable. `env.ctrl_from_unit(u)` = `mid + half*u` is the shared mapping used by every
  baseline so probes exercise the identical path `rl.net.ActionMapper` uses.
* **Observation** — actor 115 (`base lin/ang vel in pelvis frame`, gravity vector, joint pos
  relative to the stand pose, joint vel, prev unit action, command (vx,vy,wz, stance h/w),
  skill one-hot(12), lead leg, phase); privileged 43 (contacts, dorsal, tilt, pelvis-z error,
  CoM local + CoM velocity, support centre (mean of loaded foot sites — explicitly **not** a
  hull margin), CoM-to-support offset, the four marker targets, next push (dt/dir/mag), foot
  slip, saturation, joint-limit proximity, mean |q̇|, shot phase, marker distance).
  Critic = actor ⊕ privileged (158). The actor is proven blind to contacts/markers by test.
* **Command** — `(vx, vy, wz)` in the heading frame + `stance_height` + `stance_width` +
  `skill_id` (12 SOLO_DRILL §2 skills) + `lead_leg`. Documented **placeholder** ranges:
  vx [-0.25, 0.50] m/s, vy [-0.20, 0.20] m/s, wz [-0.50, 0.50] rad/s, stance height
  [0.70, 0.80] m, stance width [0.23, 0.42] m. `CommandSampler` is skill-coherent and seeded;
  `CommandFilter` is first-order (τ = 0.25 s, discrete fields snap); `CommandSchedule`
  changes commands within an episode (T2/T3/T7).
* **Termination** — `dorsal` (back-to-mat, failed attempt) or `fall`; horizon = truncation.
  Dorsal outranks a plain fall: while the dorsal condition is held the fall verdict is
  deferred (≤ dorsal confirmation window) so the episode is reported as `dorsal`.
* **Push** — `PushSpec(t, impulse N·s, direction, height, duration)` applied as a constant
  Cartesian force on the torso through `mj_applyFT` into `qfrc_applied` at a world point.
  **Both force arrays are cleared every control step** (MuJoCo does not clear them).

### 3.1 Stance parameterisation (operator's sagittal/frontal guidance)

`stance_targets(width, height)` maps to joint targets using the measured table
(hip_roll ≈ (width − 0.237)/1.22; knee flex from the measured foot-rise curve). Measured:
commanded 0.30 → 0.300 m, 0.42 → 0.421 m realised width. **Measured holdability**
(2 s scripted hold): width 0.23/0.30/0.42 at height 0.79 → no termination, pelvis stays
0.783–0.791 m, tilt ≤ 0.4°, lateral drift 0.0000 m. Deep crouches (≤ 0.72 m) remain
**unholdable by pure position servos** (measured: amplitudes > 0.15 rad toppling within
1.2–1.5 s) — the low end of the height range is a T3 target, not an S1 guarantee.

## 4. Verification evidence (all measured on this host)

### 4.1 Reset distribution holdability (Main's requirement)

Reset = `a_stand` + joint noise ±0.03 rad + base xy ±0.02 m + yaw ±10° + root tilt ±2°:
**16/16 seeds hold 2 s** under the stand keyframe ctrl (min pelvis z 0.7894 m, max tilt
3.58°, max pelvis xy drift 0.061 m). `data/solo/metrics/holdability.json`.

### 4.2 Pushes measurably perturb, and are exactly zero when idle

* Gravity-off in-air impulse: momentum after one control step **equals J within 0.02 %**
  (`6.0008` vs `6.0` N·s) → the force is applied in full, not silently clamped.
* Chest-height vs pelvis-height: J = 12 N·s at z = 0.95 topples the robot (tilt 91.7° after
  1.5 s); the same impulse at z = 0.79 does not (tilt 12.5°) → the height axis is physical.
* Standing A/B: with J = 8 N·s the base velocity right after the window differs by
  **> 0.10 m/s** from the no-push control run (test threshold; measured ~0.3 m/s).
* After the window `data.qfrc_applied` and `data.xfrc_applied` are **exactly zero** in both
  runs, and the post-window base accelerations of the push/no-push runs agree within
  2 m/s² (no stale-force drift). `test_push_measurably_perturbs`, `test_push_force_not_clamped`.
* Analytic sanity: the non-stepping ceiling for a 33.3 kg G1 is ~13 N·s; the battery's
  highest magnitude (12 N·s) is the unseen point of the T1 gate.

### 4.3 Fall / dorsal semantics

* Persistence honoured exactly (0.22 s of condition → no trigger; trigger at 0.26 s).
* Scripted forward topple (`FallForwardController`, ankle −0.10) terminates with
  `cause ∈ {fall, dorsal}`, tilt > 60°; backward topple likewise.
* Supine run (J = 16 N·s at the chest) → **`cause="dorsal"`**, and the trigger time is
  ≥ 0.30 s after the dorsal condition first held (own persistence).
* No false positives: 3 s stand hold, 3 s scripted step-in-place (feet-only contacts).
* Knees/hands are **never** terminal: settled kneel (pelvis 0.498 m, knee contacts) and
  settled hands-plant (pelvis 0.118 m, arm contacts, no torso contact) both predict `False`.
  Honest limitation (documented in `fall.py`): a fall that ends supported on an arm (e.g.
  side-lying on the elbow) is therefore not auto-terminated by the balance rule — it still
  shows up in the tilt/uprightness metrics and the T1 gate pairs fall-rate with uprightness.
  `FallDetConfig(limb_support_exempt=False)` selects the stricter variant for callers that
  want low-and-tilted to always end.

### 4.4 Reward hand-tests

Every one of the 19 terms ranks a genuine state above its degenerate counterpart
(`test_every_reward_term_hand_tested`), and the mandated degenerate policies score worse:

| degenerate case | test outcome |
|---|---|
| zero action (still, motion commanded) | worse than tracking (locomotion) |
| StandHold (stands still) | worse than tracking (locomotion) |
| fall forward at commanded speed | worse than a genuine step (uprightness gate kills the tracking) |
| squat repeatedly | worse than holding the commanded stance height (stance) |
| sliding instead of stepping | worse (feet_slide + no feet_air_time) |
| knee-parked under a recovery command | worse than rising |
| shot entered and never exited | worse than progress+exit over the same horizon (potential-based depth + timeout) |
| collapsed pose | worse in **every** task set |

Plus: `alive` (positive survival) is asserted to dominate the sum of per-step penalties for
every task, and a collapsing policy loses to an upright one over a fixed discounted horizon
(`test_reward_horizon_collapse_scores_lower`). Timeouts are not penalized (no "fall to stop
paying" incentive). Gammas: 0.995 (balance/stance/shot/reach/recovery), 0.997 (locomotion).

### 4.5 Throughput (measured on this host, single process, under concurrent agent load)

* bare env: **594–615 control steps/s** (≈12× realtime, 1.63–1.68 ms/step)
* PPO trainer (`src/solo/train.py`): **≈175–215 steps/s** end-to-end (2048-step rollouts)
* eval harness battery episodes: ~555 steps/s

### 4.6 Trainer start/stop/resume (Main's requirement)

```
# start (hold the sim lock; long runs):
MUJOCO_GL=egl .venv/bin/python -m solo.train --task balance --steps 2000000 \
    --out checkpoints/solo/t1_balance.pt
# stop: Ctrl-C -> the loop finishes the current iteration, saves, exits 0
# resume exactly (weights + optimizer + RNG + counters + episode seed):
MUJOCO_GL=egl .venv/bin/python -m solo.train --resume checkpoints/solo/t1_balance.pt \
    --steps 4000000 --out checkpoints/solo/t1_balance.pt
```

Verified: SIGINT at 3072 steps → exit 0, 2.5 MB checkpoint; resume loaded
`steps_done=3072`, continued to 3584 with the restored RNG/LR schedule
(`test_trainer_checkpoint_roundtrip` covers the in-process round trip; the SIGINT path was
verified with a real subprocess).

## 5. Exploit probes and baselines (the eval harness that gates training)

The single harness is `solo.eval.evaluate(controller_factory, task, ...)` → per-step JSONL +
summary JSON + a `TaskGate` verdict with reasons. Gates are **provisional placeholders**
(S2 sets them from measured baselines), which is exactly why trivial controllers must fail
them — and they all do. Measured (2×2 clip metadata: h264/yuv420p 960×720 30 fps):

| run | task | verdict | key measured numbers |
|---|---|---|---|
| probe zero action | balance + battery (4/8/12 N·s × 8 dirs) | **not_certified** | fall rate 0.21, mean upright 0.138, max recoverable J 8.0, recovery 0.0 |
| probe StandHold | balance + battery | **not_certified** | fall rate 0.08, upright 0.918, max recoverable J 12.0, recovery 1.00 |
| probe fall forward (ankle −0.10, cmd 0.5 m/s) | locomotion | **not_certified** | fall rate **1.00**, vel_err 0.407, upright 0.760 (it tracks while toppling) |
| probe squat repeatedly (0.12 rad @ 0.25 Hz) | stance (LEVEL_CHANGE) | **not_certified** | fall 0.00, upright 0.889, vel_err 0.232 (oscillates, never holds the commanded stance) |
| probe random-init policy (sampled) | balance | **not_certified** | fall rate 0.79, upright 0.179 |
| T1 baseline StandHold | balance + battery (4,6,8,10,12 N·s × 8 dirs, 40 eps) | **not_certified** | fall rate 0.03, upright 0.923, max recoverable J 12.0, recovery 1.00 |
| T1 baseline random-init | same battery | **not_certified** | fall rate 0.75, upright 0.164, max recoverable J 12.0, recovery 0.00 |
| T2 baseline StandHold | locomotion (8 seeded command episodes) | **not_certified** | vel_err 0.183 m/s, yaw_err 0.089 rad/s, upright 0.9999, fall 0.00 |
| T2 baseline random-init | locomotion | **not_certified** | vel_err 0.837, yaw_err 5.49, upright 0.140, fall 1.00 |

**Honest finding for S2 (important).** The T1 battery does *not* discriminate StandHold on
fall rate: a stiff position-servo stand survives most chest-height impulses up to 12 N·s
(fall rate 0.03, recovery 1.00, max recoverable J 12). The gate rejects it on
`mean_upright` (0.923 < 0.95 provisional) only. So "push recovery is ABSENT" (audit E3)
remains true as *learned capability*, but the *task* at these magnitudes is partly solvable
by a static stand. S2 should (a) extend the battery past the analytic non-stepping ceiling
(~13 N·s) and include lateral/oblique directions that force a step, (b) add
time-to-stability / CoM-margin criteria, and (c) keep the uprightness pairing. This is a
baseline-first measurement, not a tuning result.

## 6. Reusable helpers for the drill pipeline (Main's requirement)

Exact signatures (import from `solo.*`, with `src/` on `sys.path`):

```python
from solo.env import SoloEnv, TASKS, TaskSpec
env = SoloEnv(model=None, task="balance", seed=0, action_mode="absolute",
              residual_scale=0.5, push=None, command=None, marker_plan=None,
              record_metrics=True)          # TASKS: balance|locomotion|stance|reach|shot|recovery
obs = env.reset(seed=0, pose=None, push=None, command=None)   # -> {"actor","privileged","critic"}
obs, reward, terminated, truncated, info = env.step(ctrl29)   # 50 Hz; info["metrics"], info["command"]

from solo.eval import evaluate, run_episode, battery_pushes, GATES, take_clips
report = evaluate(controller_factory, task="balance", episodes=8, seed0=0, gate=None,
                  push_plan=battery_pushes(...), command_plan=None, env_kwargs=None,
                  out_dir="data/solo/metrics", name="mypolicy",
                  clip_factory=lambda i, seed, env: ClipRecorder("title", "sub"),
                  max_episode_s=4.0)
# controller_factory(env, seed) -> callable(env, data) -> (29,) ctrl targets
# report: {"verdict","certified","reasons","aggregate","episodes","config"}
from solo.eval import take_clips
clips = take_clips(report)            # ClipRecorder objects captured live

from solo.video import ClipRecorder
clip = ClipRecorder("stage/task", "checkpoint/seed", max_frames=120, verdict=None)
clip.render(model, "videos/solo_drill/x.mp4", extra_lines=("VERDICT: ...",))
clip.contact_sheet(model, "videos/solo_drill/x_sheet.png", caption="...")

from solo.lock import SimLock
with SimLock(owner="my run", wait_s=900):   # advisory; data/locks/sim.lock
    ...

from solo.train import TrainConfig, SoloTrainer   # start/stop/resume PPO on SoloEnv
```

## 7. Videos / artifacts

* `videos/solo_drill/baselines/probe_zero_action.mp4` + `_sheet.png` (rendered; ffprobe
  h264/yuv420p 960×720 30 fps).
* Remaining probe/baseline clips + `data/solo/metrics/{probes,baselines}_summary.json`
  were still being produced by `scripts/solo_env_smoke.py probes|baselines` at hand-off;
  the command is deterministic and re-runnable (`--no-video` for metrics only).
* `data/solo/metrics/`: per-step JSONL traces + summary JSON per run (gates, reasons,
  exact `env.config()`), `holdability.json`.

## 8. Honest limitations / notes for S2+

1. **Gate thresholds are placeholders** (documented in `eval.GATES`); S2 sets them from the
   measured baselines. Max battery magnitude (12 N·s) is unseen-in-training by design.
2. **No locomotion reference exists yet** — the T2 "no false positive" evidence for the fall
   detector covers stance + scripted stepping; real locomotion is S3's test.
3. **Side-lying falls supported by an arm** are not auto-terminated by the balance rule
   (the "hands are never terminal" contract); uprightness metrics and the dorsal rule still
   catch them.
4. **Reward weights are the literature starting point, not tuned**; the `alive`-dominates
   invariant is the only structural guarantee.
5. **Time-limit truncation drops the bootstrap value** in the trainer (conservative
   convention); noted for S2 (a bias-free variant would use the value of the next state).
6. **Concurrent-load caveat**: throughput numbers were measured while other agents ran
   MuJoCo/render jobs on the same 4 cores; re-measure on a quiet host before sizing runs.
7. Repo-wide test failures (7) are in the scorer/wrestling files owned by other agents and
   are unrelated to `src/solo/` (details at the top of this report).
