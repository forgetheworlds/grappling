# T2 locomotion gate — behavioural, held-out, and provably discriminating

**Date:** 2026-10-08 · **Author:** T2Gate (delegated) · **Status:** gate + baselines + discrimination
tests **committed** (`14c3e98`); no training started; the T2 run itself is a later ticket.

Everything below is measured on this host with the shipped model; no number is copied from prose
and no threshold is invented. The gate is **not** reward-based: every criterion is a physical
outcome (tracking error, uprightness, terminations, foot slip, distance travelled).

Reproduce:

```bash
MUJOCO_GL=egl .venv/bin/python scripts/solo_t2_gate.py --no-lock      # baseline table, ~30 s
.venv/bin/python -m pytest tests/solo/test_locomotion_gate.py -q -s   # discrimination, ~12 s
```

---

## 1. Gate definition

`src/solo/eval.py::GATES["locomotion"]` (name `T2_locomotion`, `provisional=False`), evaluated by
`evaluate()` against the held-out command plan with one command per episode:

| # | metric (measured) | op | threshold | derivation (from the table in §2) |
|---|---|---|---|---|
| 1 | `vx_err_abs_mean` (m/s) | `<=` | 0.10 | 25% of the smallest held-out &#124;vx&#124; (0.40 m/s); reference 0.021 PASS, StandHold@held-out 0.207 FAIL |
| 2 | `vy_err_abs_mean` (m/s) | `<=` | 0.045 | 25% of the smallest held-out &#124;vy&#124; (0.18 m/s); reference 0.003 PASS |
| 3 | `yaw_err_abs_mean` (rad/s) | `<=` | 0.15 | 33% of the smallest held-out &#124;wz&#124; (0.45 rad/s); reference 0.033 PASS, StandHold@held-out 0.161 FAIL |
| 4 | `mean_upright` | `>=` | 0.95 | bracketed by the measurement: worst non-falling controller 0.9999 vs best falling one 0.756 |
| 5 | `fall_rate` | `<=` | 0.10 | every displacing baseline falls (FallForward/RandomInit 1.0) |
| 6 | `dorsal_rate` | `<=` | 0.10 | back-to-mat terminations are falls too (ZeroAction 0.625) |
| 7 | `slip_mean` (m/s) | `<=` | 0.06 | standing ≤0.008 vs sliding ≥0.098: 15% of the smallest held-out speed (0.40 m/s) |
| 8 | `slip_ratio_mean` (m/m) | `<=` | 0.80 | load-transfer measurements ≤0.57, planted-foot drags ≥1.75 (8× gap) |
| 9 | `dist_err_mean` (m) | `<=` | 0.30 | 15% of the smallest held-out commanded path (0.40 m/s × 5.5 s = 2.2 m); reference 0.040 PASS |

Thresholds live in one place (`solo.eval.T2_THRESHOLDS`) so the gate, the script and the tests
cannot drift apart.

**Measurement protocol.** Episodes are 6.0 s (`T2_EPISODE_S`); the *settled window* is `t >= 0.5 s`
(`T2_SETTLE_S`) — a jittered reset drops the robot and it skids 0.1–0.2 m in the first ~0.5 s, and
standing still is not "travel". Tracking errors, slip and travelled/commanded distance are measured
from 0.5 s on; uprightness and terminations are episode-wide. Per-episode values are reported in the
run summary and in `data/solo/metrics/locomotion_t2gate_*_s0.summary.json`.

**Metrics added for this gate** (all recorded per control step by the env, so the numbers are
traceable):

| field | meaning |
|---|---|
| `vx_err`, `vy_err` | signed per-axis tracking error, heading frame (m/s) |
| `slip` | max horizontal *speed* of loaded foot sites (the pre-existing point sample) |
| `slip_travel` | distance travelled **this step** by loaded foot sites (position-based) |
| `body_step` | distance travelled this step by the pelvis |
| `cmd_vx`, `cmd_vy`, `cmd_wz` | the filtered command in force this step |

`slip_ratio = Σ slip_travel / max(Σ body_step, 0.10 m)` — loaded-foot travel per metre travelled.
It exists because `slip` **cannot** see a slow drag: MuJoCo's friction constraint zeroes the
relative foot velocity inside a substep while the positions still integrate, so a planted-foot
slider reads the same `slip` as a standing robot (measured: 0.0026 vs 0.0021). §3.3 proves this on
the live model.

**Distance travelled vs commanded** is path length (`Σ body_step`) vs `Σ |cmd_xy|·dt` over the
settled window; the criterion is the absolute difference, sign-free, per episode.

## 2. Held-out command set

Held-out is a property of the **range**, not of a seed: a sampler that covers the feasible range
holds nothing out. `solo.commands.T2_TRAIN_RANGES` therefore narrows the training domain, and the
locomotion task preset now samples *only* inside it:

```
T2_TRAIN_RANGES: vx ∈ [-0.15, 0.35]  vy ∈ [-0.12, 0.12]  wz ∈ [-0.30, 0.30]
feasible ranges:  vx ∈ [-0.25, 0.50]  vy ∈ [-0.20, 0.20]  wz ∈ [-0.50, 0.50]
```

Same pattern as `pushes.TRAIN_MAX_IMPULSE`: train below the boundary, gate above it. The gate's
held-out commands (all physically feasible, all outside the training range by ≥ `T2_HELDOUT_MARGIN`
= 0.04 on at least one axis) are:

| # | vx | vy | wz | skill | outside-training margin per axis |
|---|---|---|---|---|---|
| 1 | +0.45 | 0.00 | 0.00 | SHUFFLE_F | vx 0.10 |
| 2 | +0.40 | 0.00 | +0.40 | CIRCLE_L | vx 0.05, wz 0.10 |
| 3 | 0.00 | +0.18 | 0.00 | SHUFFLE_L | vy 0.06 |
| 4 | 0.00 | −0.18 | 0.00 | SHUFFLE_R | vy 0.06 |
| 5 | 0.00 | 0.00 | −0.45 | CIRCLE_R | wz 0.15 |
| 6 | −0.20 | 0.00 | 0.00 | RETREAT | vx 0.05 |
| 7 | +0.38 | −0.17 | +0.35 | CIRCLE_L | vx 0.03, vy 0.05, wz 0.05 |
| 8 | +0.45 ↔ −0.20, 0.75 s each (8 segments) | | | reversal schedule | vx 0.10 / 0.05 |

Episode 8 is the **command-oscillation** anti-exploit check from `docs/SOLO_DRILL.md` §5: a
controller that "tracks" by drifting one way, or that oscillates to farm a tracking signal, pays the
error on both halves.

Integrity (asserted by test 5): every held-out command is contained in the feasible
`CommandRanges()`, is **not** contained in `T2_TRAIN_RANGES`, and 500 draws from the live locomotion
preset sampler (`TASKS["locomotion"].command_sampler`) are all inside the training domain and none
is held out. Consequence, stated explicitly: if a future T2 run samples the *full* feasible ranges
(bypassing the preset), commands 1–7 fall inside its sampling support and the set is no longer held
out — the disjointness is guaranteed by the preset, which is why the preset was changed.

## 3. Baseline table (recorded) and the reference

Recorded in `data/solo/metrics/t2_gate_baselines.json` (gate dict + thresholds + plan + controller
table + `reference_unattainable`), written by `scripts/solo_t2_gate.py`; per-episode detail in
`data/solo/metrics/locomotion_t2gate_*_s0.summary.json`. 8 held-out episodes × 6 s per controller.

| controller | verdict | vx_err | vy_err | yaw_err | upright | fall/dorsal | slip | slip_ratio | dist_err | FAILED criteria |
|---|---|---|---|---|---|---|---|---|---|---|
| **stand_hold** | not_certified | 0.207 | 0.068 | 0.161 | 0.9999 | 0.0/0.0 | 0.0011 | 0.101 | **1.356** | vx, vy, yaw, dist_err |
| **zero_action** | not_certified | 0.390 | 0.068 | 0.962 | **0.223** | 0.0/0.625 | 0.062 | 0.263 | 0.542 | upright, dorsal, vx, vy, yaw, slip, dist |
| **fall_forward** | not_certified | 0.633 | 0.076 | 1.246 | **0.756** | **1.0**/0.0 | **0.146** | 0.323 | 0.575 | upright, fall, vx, vy, yaw, slip, dist |
| **random_init_policy** | not_certified | 0.619 | 0.490 | 4.702 | 0.175 | 1.0/0.0 | 0.851 | 1.429 | 0.604 | 8 of 9 |
| planted_foot_drag *(probe)* | not_certified | 0.450 | 0.004 | 0.038 | 0.9998 | 0.0/0.0 | 0.0019 | **2.000** | 0.276 | vx, slip_ratio |
| airborne_transfer *(probe)* | not_certified | 1.291 | 0.067 | 1.455 | 0.748 | 1.0/0.0 | 0.115 | **0.234** | 0.330 | 7 of 9 (passes slip_ratio) |
| **reference: stand_hold, TINY commands** | **certified** | 0.021 | 0.003 | 0.033 | 0.9999 | 0.0/0.0 | 0.0016 | 0.141 | 0.040 | — none |

Reference commands (inside the training range on purpose): `vx=0.02` and `wz=0.02`.

**Which baselines already FAIL, and how.** All four fail. `stand_hold` fails *only* on behaviour
(tracking and distance) while being upright, alive and never terminating — exactly the case the old
placeholder gate let through. `zero_action` and `random_init_policy` are collapsing controllers.
`fall_forward` is the canonical exploit: it fails on uprightness and the fall criterion, and
measured against the held-out forward command it *travels* (comparable to a genuine solution's
commanded distance), which is why displacement alone is not a criterion the gate trusts.

## 4. Discrimination tests — `tests/solo/test_locomotion_gate.py`

7 tests, `7 passed in 11.7 s` (`-q -s` prints every deciding metric below). Each asserts a
**verdict**, not wiring.

**4.1 Stand-still must FAIL (test 1).** `StandHold` on held-out commands (forward + lateral):
`verdict=not_certified`, `mean_upright=0.999832`, `fall_rate=0.0`, `travelled=0.092 m` vs
`commanded=1.109 m`.
`FAIL vx_err_abs_mean <= 0.1 (measured 0.23389)` · `FAIL dist_err_mean <= 0.3 (measured 1.0168)`.
It cannot pass by being upright and alive. The same controller under tiny commands is
**certified** (`vx_err 0.027`, `dist_err 0.057`), so the tracking bar is reachable, not absurd.
`ZeroAction` also fails (`vx_err 0.780`, `upright 0.061`).

**4.2 Fall-forward must FAIL (test 2).** `FallForward` on the held-out forward command:
`travelled=0.953 m` (the exploit moves!), `mean_upright=0.776547`, `fall_rate=1.0`,
`vx_err=0.393`, `slip=0.118`.
`FAIL mean_upright >= 0.95 (measured 0.776547)` · `FAIL fall_rate <= 0.1 (measured 1.0)`.
The test asserts the displacement is real (>0.30 m) *and* that the rejection comes from the
physical-quality criteria.

**4.3 Slide must FAIL, and the slip metric must separate it from a load transfer (test 3).**
Both probes run the same held-out command, 4 s, same seed:

* `planted_foot_drag` (root translated every step with both feet loaded — the 29-joint stand
  targets held): `slip_ratio=2.0037` (`1.761 m` of loaded-foot travel per `0.879 m` travelled),
  `upright=0.9998`, `fall=0.0`, `steps=0`, `loaded_frac=1.0`,
  `FAIL slip_ratio_mean <= 0.8 (measured 2.0037)` — while its point-sampled `slip` is
  **0.0026**, i.e. the same as a standing robot.
* `airborne_transfer` (the body translated only while both feet are off the floor): 
  `slip_ratio=0.2338` (`0.176 m` per `0.753 m`), `steps=9`, `loaded_frac=0.489`,
  `PASS slip_ratio_mean <= 0.8 (measured 0.2338)`, at comparable displacement (0.75 vs 0.88 m).

So the criterion separates the two mechanisms at equal displacement, on the same command, and the
test prints that the **point-sampled slip metric does not** (drag 0.0026 ≤ 0.06 while the transfer
probe reads 0.115) — which is why the gate uses the position-based `slip_ratio`.

*Honest caveat:* `airborne_transfer` is a **metric probe, not a gait** (crouch, leave the floor,
translate, absorb). It topples within ~1.5 s and fails 7 of 9 criteria; the test claims only that it
**passes the slip criterion**, which is the two-sided property the criterion needs (fires on a drag,
silent on load transfer). It does not claim a stepping solution exists.

**4.4 The reference PASSES every criterion it can physically meet (test 4).**
`StandHold` under the tiny, non-held-out command set: `verdict=certified`, all nine lines `PASS` —
`vx_err 0.027297`, `vy_err 0.003951`, `yaw_err 0.045403`, `upright 0.999832`, `fall 0.0`,
`dorsal 0.0`, `slip 0.002147`, `slip_ratio 0.1205`, `dist_err 0.0568` (`travelled 0.092 m` vs
`commanded 0.0352 m`). No threshold was weakened for it: the same thresholds reject all four
baselines. The test also runs the *same* controller on the held-out forward command
(`not_certified`, `vx_err 0.451`, `dist_err 1.461`) and prints it, so the reference's pass is
visibly tied to the command being small, not to a soft gate.

**4.5 Command-set integrity (test 5).** 7 held-out commands, all feasible, all outside
`T2_TRAIN_RANGES` by ≥0.04 on ≥1 axis (max margin 0.15); 500 live-preset samples never leave the
training domain; the reversal schedule's 8 segments are all held-out (`vx ∈ {−0.20, +0.45}`).

**4.6 Gate invariants (tests 6–7).** No criterion references a reward quantity; the criterion set is
exactly the nine above; every metric the gate reads is a recorded per-step field (`vx_err`, `vy_err`,
`yaw_err`, `slip`, `slip_travel`, `body_step`, `cmd_vx/vy`, `upright`); a live locomotion episode
produces the per-episode quantities (`travelled 0.113 m`, `commanded 1.134 m`, `dist_err 1.021 m`,
`slip_ratio 0.1205`, 126 of 150 steps settled).

## 5. What T2 training must beat

A certified T2 policy must, on the 8 held-out episodes (commands outside the training range,
including the reversal schedule), hold **mean |vx| error ≤ 0.10 m/s, |vy| error ≤ 0.045 m/s and
|yaw-rate| error ≤ 0.15 rad/s** while staying **upright ≥ 0.95 mean with ≤0.10 falls and ≤0.10
back-to-mat terminations**, travel within **0.30 m** of the commanded path, keep **loaded-foot slip
≤ 0.06 m/s** and **loaded-foot travel ≤ 0.80 m per metre travelled** — i.e. it must *step* (move the
body over a planted foot) rather than skate, topple, or stand: the only reference that currently
displaces at all is the topple (upright 0.756, falls 1.0) and the only controller that stays upright
and translates is the planted-foot drag (slip_ratio 2.00). Every threshold is fixed by the table in
§2; the T1 balance gate must be re-run after T2 training (M1 forgetting check, `SOLO_DRILL.md` §3).

## 6. Honest limits and findings

1. **No criterion is unattainable** — the reference passes all nine — but every criterion is
   *unproven* at held-out magnitudes: no controller in the repo (or buildable in an afternoon)
   tracks a held-out command. The reference demonstrates attainability only at tiny commands.
2. **`slip_mean` is blind to kinematic drags** (§1); both slip criteria are kept, because the
   point sample still catches the physical topple-slide (FallForward 0.146, the transfer probe
   0.115) while `slip_ratio` catches the drag.
3. **`slip_ratio` is the only criterion the drag probe fails besides tracking**; a slider that also
   tracks the commanded speed would be rejected *only* by slip_ratio + dist_err, which is why the
   gate reports both.
4. **Nothing physical steps yet.** Lifting a foot (`swing`, hop, crouch-then-extend probes) topples
   within ~1.5 s at every parameter setting tried (36 sweeps: crouch 0.12–0.8 rad, periods 0.2–0.9 s,
   speeds 0.1–0.4 m/s), which matches the earlier finding that a position-servo G1 cannot hold any
   crouch open-loop. The two probes are therefore kinematic (root-translation) instruments for the
   *metric*, not policies.
5. **Reset transient** is excluded by the 0.5 s settle window; without it, a standing robot "travels"
   0.12–0.18 m and any distance criterion measures the drop, not the policy.
6. `evaluate()` falls back to the task's own command sampler for episodes beyond `command_plan`
   (default `episodes=8`). The gate script and the tests pass `episodes=len(plan)` and assert the
   episode count, so a short plan can never silently mix sampled commands into a held-out run — the
   first version of the script had exactly that bug and it produced plausible-looking wrong numbers.
7. `scripts/solo_env_smoke.py` was **not** touched (another agent holds it). To wire the T2 gate
   into `cmd_baselines`, replace its T2 loop body with the plan-driven call used by
   `scripts/solo_t2_gate.py` — one shared line is the key one:
   `evaluate(..., episodes=len(plan), command_plan=heldout_command_plan()[:args.t2_episodes],
   gate=GATES["locomotion"], ...)`. `cmd_monitor` can then gain a `--task locomotion` branch that
   calls `scripts/solo_t2_gate.py`'s `_run` on a checkpoint's policy.
8. `data/solo/metrics/locomotion_*` tables from the S1 pass (gate placeholders) are superseded by
   this gate; they are left in place as history, not re-labelled.

## 7. Artefacts

| path | content |
|---|---|
| `src/solo/eval.py` | held-out command set + `T2_THRESHOLDS` + `GATES["locomotion"]` + query of the settled window |
| `src/solo/commands.py` | `T2_TRAIN_RANGES` (training domain) |
| `src/solo/env.py` | locomotion preset samples `T2_TRAIN_RANGES`; records `vx_err/vy_err/slip_travel/body_step/cmd_*` |
| `src/solo/metrics.py` | the new per-step fields (documented in the module table) |
| `src/solo/baselines.py` | `PlantedFootDragController`, `AirborneTransferController` (metric probes) |
| `scripts/solo_t2_gate.py` | baseline table + gate printout + JSON writer (`--quick`, `--no-lock`) |
| `tests/solo/test_locomotion_gate.py` | 7 discrimination tests |
| `data/solo/metrics/t2_gate_baselines.json` | the recorded table the thresholds are derived from |
| `data/solo/metrics/locomotion_t2gate_*_s0.{jsonl,summary.json}` | per-step traces + per-episode detail |
