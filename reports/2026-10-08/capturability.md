# Capturability — the capture-point signal, the crouch exploit, and the brace experiment

**Status: completed from the killed agent's module.** Agent `Capturability` was stopped by the
provider outage after writing `src/solo/stability.py`; it never wrote its test file, its self-check,
the brace experiment or this report. All numbers below are from runs on this host; the module's
self-check and the experiment script are re-runnable.

**Artifacts**

| file | what |
|---|---|
| `src/solo/stability.py` | the signal: CP, support hull, signed margins, brace-step length, DCM test, reward term, `StabilityMonitor` |
| `tests/solo/test_stability.py` | 7 tests, numbers on hand-constructed states (incl. the crouch case) |
| `scripts/solo_brace_experiment.py` | the scripted brace experiment (4 strategies × 6 impulses) |
| `data/drill/brace_experiment.json` | its output |

```
MUJOCO_GL=egl .venv/bin/python -c "import sys;sys.path.insert(0,'src');import runpy;runpy.run_module('solo.stability',run_name='__main__')"
MUJOCO_GL=egl .venv/bin/python -m pytest tests/solo/test_stability.py -q          # 7 passed
MUJOCO_GL=egl .venv/bin/python scripts/solo_brace_experiment.py --json data/drill/brace_experiment.json
```

## 1. The signal

Per control step, from the model + contacts: CoM position and mass-weighted `cvel` velocity; the
capture point `cp = com_xy + v_xy * sqrt(z_c/g)` (Pratt 2006; `z_c` floored at 0.30 m so a fallen
robot cannot produce an infinite CP); the support hull of the *touching* feet's sole footprints
(plus a contact-only hull, because MuJoCo soft contacts collapse a contact-point hull to a line
when a foot rolls); the signed margins of both the CoM and the CP; the required brace-step length
`max(0, ||CP − CoM|| − dCOP)`; the DCM divergence test (Schuller 2026 eq. 48) and the geometric
predicate `CP outside the hull` (Stephens 2007 eq. 4) which the tests pin as equivalent.

At the stand keyframe: `margin_cp = +0.0533 m`, `tau = 0.2656 s`, hull 4 points, `unsafe=False`,
`converging=True`, `vdot = −0.0064`, `brace_step_length = 0.0`.

## 2. The exploit-killer: the crouch is UNSAFE while the fall detector is silent

`v5@301k` (the degraded run) sat at pelvis **0.370 m** with a CoM-to-support offset up to
**0.322 m** (stored battery). The solo fall rule
(`src/solo/fall.py`: `pelvis_z_ground=0.35`, `tilt_ground_deg=60`, `pelvis_z_hard=0.22`, 0.25 s
persistence, limbs-support exempt) does **not** fire: 0.370 > 0.35 and the crouch is upright, so the
tilt rule is silent too. The stance hull's worst-cardinal margin at the keyframe is +0.053 m, so a
0.322 m offset puts the CoM — let alone the capture point — far outside the support: the CP
predicate calls every such state **UNSAFE** and asks for a brace step of ~0.3 m.

The constructed case is pinned in `tests/solo/test_stability.py::test_crouch_with_com_inside_but_cp_outside_is_unsafe`:
a state with `signed_margin(com, hull) = +0.5 m` (a CoM-inside test PASSES) and
`capture_point_unsafe(cp, hull) = True`. This is the "CoM outside the support polygon with no step
taken" failure `prior_art` §2.1 names as *the* M1 defect, made machine-checkable.

## 3. Obs contract: the CP is missing from the actor observation (design-doc §5.1 wants it)

Measured layout (`src/solo/obs.py`):

* **actor** (`ACTOR_LAYOUT`, dim = `103 + N_SKILLS`): base lin/ang velocity, gravity, joint
  pos/vel, prev action, cmd_vel, cmd_stance, skill one-hot, lead leg, phase. **No CoM, support or
  capture-point field.**
* **privileged** (`PRIV_LAYOUT`, appended after the actor vector): `com_rel_local` (10,13),
  `com_vel_local` (13,16), `support_center_local` (16,19), **`com_to_support_xy` (19,21)** —
  present but privileged; **no capture-point field anywhere**.

So §5.1's `capture_point_xy − support_center_xy` is missing, and `com_xy − support_center_xy` exists
only in the critic. The exact patch (**NOT applied** — it changes `ACTOR_DIM` and invalidates every
existing checkpoint, so it must be a deliberate run):

```python
# src/solo/obs.py -- append to ACTOR_LAYOUT
    ("capture_point_rel_support", (ACTOR_DIM, ACTOR_DIM + 2)),
    ("cp_margin", (ACTOR_DIM + 2, ACTOR_DIM + 3)),
# ObsContext gains capture_point_rel_support (2,) and cp_margin (float), filled by
# the env from StabilityMonitor.measure(): cp_xy - support_center, and margin_cp.
```

The evidence for wanting it: H12's ablation (cited in `prior_art` §5 line 226) is *failure to learn
stand-up without* the capture point among the critic inputs — 93.4 % with it. Our critic already
carries `com_to_support_xy`; the CP is the missing half, and it is the half that is *predictive*
(the CoM alone cannot distinguish "leaning but recoverable" from "already falling").

## 4. The brace experiment (the operator's question, with numbers)

`scripts/solo_brace_experiment.py`: one +x push at t=1.0 s, 0.12 s long, from the drill stand
keyframe; one reset per run; four scripted strategies. `FALL` = the runner's own fall detector
fired; `ok` = survived AND ended inside the stance predicate.

| impulse (N·s) | no-feedback (`stance_pd`) | static (`feasible` L0) | ankle/hip (L1) | brace step (L2) |
|---|---|---|---|---|
| 5 | FALL | ok | ok | **ok — 1 step, 0.169 m foot travel, margin +0.060** |
| 8 | FALL | FALL | FALL | FALL |
| 12 | FALL | FALL | FALL | FALL |
| 16 | FALL | FALL | FALL | FALL |
| 20 | FALL | FALL | FALL | FALL |
| 25 | FALL | FALL | FALL | FALL |

Measured facts:

* The **pure-PD baseline topples before the push arrives** at every impulse (it cannot hold the
  drill stand keyframe for 1 s) — consistent with the E3 finding; it is a baseline, not a stance.
* The scripted stack's **non-stepping ceiling is between 5 and 8 N·s** — well below the T3 gate's
  16 N·s bar, and below the analytic ~20–23 N·s estimate. At 8 N·s the static and ankle/hip arms
  fall with the CP excursion ~0.28 m and the required brace ~0.28 m.
* The **brace step fires exactly once**, at 5 N·s: 0.169 m of foot-centre travel, and the run ends
  with a *better* margin than the static arm (+0.060 vs +0.046). At ≥8 N·s **no step event is
  recorded** (0 steps; the 0.15–0.20 m of foot travel is the body falling) — the current L2 arm
  steps only when the scheduler asks it to, not reactively when the CP leaves the hull. A reactive
  CP-triggered brace is the missing piece; that is a controller change, not a physics limit.
* The required brace step at the fall threshold (0.28–0.31 m) exceeds what one step supplies from
  the current stance; the stance geometry's single-step reach (the drill's `ENTRY_STEP_M = 0.14 m`
  per placement, plus the pelvis shift) is roughly half of it — consistent with "one small step
  recovers small pushes, larger ones need a second step or a lean-into".

## 5. Wiring diffs (produced, NOT applied)

1. **Reward** — `capturability +0.5` per `prior_art` §5.1, evaluated with
   `solo.stability.capturability_reward(state)`. The env already computes the CoM, CoM velocity and
   (with `--stance-return`) the hull; the wiring is one term in `TaskReward` with
   `RewardInputs.capture_point_xy`/`hull` filled from the monitor. **Defect fixed in the module**:
   the killed agent's implementation was *inverted* — it returned 1.0 exactly when the CP was
   outside the hull (it added `+margin_cp` instead of subtracting the outward distance), i.e. it
   paid the crouch. Now `clip(band − outside_distance, 0, band)/band` (1 while the CP is inside,
   decaying to 0 once it is 5 cm outside), pinned by
   `test_capturability_reward_band_endpoints`.
2. **Gate** — replace/augment the pelvis-tilt criterion with the CP margin: T1's terminal-stance and
   stability criteria should read `cp_margin_min >= 0` (a crouch inside the height/tilt thresholds
   but outside the hull can no longer pass). The stored baseline tables must be re-run through the
   monitor before this becomes primary (standing rule 4).
3. **Curriculum** — trigger the brace step on `stale.cp_margin < 0` (reactive), not on the
   scheduler's skill; the experiment above shows the reactive trigger is what the scripted stack
   lacks at ≥8 N·s.

## 6. What this changes

The reward can no longer pay for a posture that has already lost the hull; the gate can see the
difference between "leaning but capturable" and "falling"; and the stepping trigger becomes a
measured quantity (`brace_step_length`) instead of a heuristic. The T3 bar (≥16 N·s stepping
recovery) is **not** reachable by the current scripted stack — measured ceiling 5–8 N·s — so it
remains a learned-policy target, with the scripted stack as the reference for what "stepping
capable" must beat.

## 7. UNVERIFIED / open

* The brace step's behaviour above 5 N·s is limited by the *scheduler*, not measured as a
  controller limit — the reactive trigger is untested.
* The v5 crouch statement uses the stored battery's `com_offset_max` and the audit's `pelvis_z`
  read; the crouch was not replayed tick-by-tick through the monitor (the constructed case in the
  test stands in for it).
* The actor-obs patch changes `ACTOR_DIM`; no checkpoint compatibility was attempted.
* The experiment runs without the shared sim lock (the box was idle); timings are not throughput
  claims.
