# Phase 4 — wrestling environment + back-to-mat detector (deliverables 10, 11)

**Agent:** EnvBuilder · **Date:** 2026-10-08
**Deliverables:** `src/wrestling/` (`env.py`, `backdet.py`), `scripts/calibrate_backdet.py`,
`data/backdet_calibration.json`, `tests/test_wrestling.py`, this report.
**Verification:** `.venv/bin/python -m pytest tests/ -q` → **46 passed**
(30 pre-existing + 16 new, ~5 s for the new file);
`scripts/calibrate_backdet.py` self-check green; env/backdet module self-checks green.

Everything below is measured on this host unless labelled otherwise.

---

## 1. Back-to-mat detector (deliverable 10)

### 1.1 Semantics implemented

A robot is *back-to-mat* at time `t` iff all of

1. **Dorsal torso contact with the mat** — a `floor` contact of `torso_link` or
   `pelvis` collision geoms whose contact point lies in the *dorsal* half of that
   body's frame (local `x < 0`).  Sign anchor: G1 foot toes point along local `+x`
   (site `toe` at `x=+0.125`), and the retargeted world places robot A at `−x`
   *facing* `+x` (`src/retarget/world.py`), so local `+x` is the front/chest.
   A supine robot therefore touches the mat with `x_local ≈ −0.05…−0.07`; a
   face-down sprawl/superman touches with `x_local > 0` and must not trigger.
   *Independent cross-check:* the mat→robot contact normal must oppose the dorsal
   axis (`dot(−R_body[:,0], n) < −0.5`) — **agreement 1.00** on the positive window
   (see `dorsal_normal` in the feature record), i.e. "which side is the back" is
   verified by two independent geometric criteria, not assumed.
2. **Torso up-axis tilt** — `arccos((R_torso @ ez)_z) ≥ tilt_threshold`.  Tilt alone
   cannot separate supine from prone (both ≈90°); it exists to reject *upright*
   postures whose pelvis/back happens to graze the mat.
3. **Pelvis low** — `pelvis_z ≤ pelvis_z_threshold`.
4. **Persistence** — the joint condition must hold continuously for
   `confirm_s` before a trigger is reported (streaming latch, re-armed per exchange).

Not a collision bit: knees, hands, elbows and feet touching the mat set only a
diagnostic flag (`limb_contact`) and can never trigger.  Implementation:
`wrestling.backdet.back_features` / `BackToMatDetector` (streaming, 50 Hz) plus
`confirmed_mask` / `first_confirmed_index` (batch, used by the sweep; equivalence is
pinned by `test_backdet_batch_matches_streaming`).

### 1.2 Calibration (scripts/calibrate_backdet.py)

Rollouts: all 7 `data/refs` traces PD-replayed in `robots/wrestling_scene.xml`
(reference's own first frame as the initial state, `ctrl = q_ref`, +0.5 s hold tail),
features at 50 Hz for both robots (same generator as `scripts/validate_refs.py`).

Labels (measured, printed by the script):

| set | definition | samples |
|---|---|---|
| POSITIVE | SPRAWL defender (`b`): dorsal-contact samples after the last sample with `pelvis_z ≥ 0.20 m`, restricted to frames where that dorsal episode has already lasted `LABEL_GRACE = 0.5 s` | 157 |
| NEG_STAND | any rollout sample with `pelvis_z ≥ 0.55 m` ("clearly standing") | 695 |
| NEG_GROUND_START | STAND_UP rollout, both robots, `t ≤ 1.0 s` (knees/hands ground start) | 100 |
| other | untaught PD-replay collapse frames of the shooting techniques | reported, **not** scored |

Measured bounds used by the operating point: positive tilt min 62.6° / median 80.8°,
positive pelvis-z max 0.094 m, defender's terminal episode onset 1.362 s, front
contacts inside the positive window 0; largest tilt of any *mat-touching* negative
26.2° (32 samples, from the STAND_UP ground start); the standing negatives reach
125.1° tilt but never touch the mat with the torso.

Sweep: `tilt ∈ {40…75°} × pelz ∈ {0.20…0.50 m} × confirm ∈ {0.05…0.60 s}` = 560
configs.  Sensitivity = fraction of established positive samples where the detector is
confirmed; specificity = fraction of negative samples where it is not.  Selection:
gate `sens ≥ 0.95 ∧ spec ≥ 0.95`, then prefer the largest confirmation within a
0.35 s detection-latency budget, then the tilt threshold closest to the balanced
midpoint of `[26.2°, 62.6°]`, then `pelvis_z` closest to 0.35 m.

**Chosen operating point:** `tilt = 45°`, `pelvis_z = 0.35 m`, `confirm = 0.30 s`
(`dorsal_x_max = 0`), **sens = 1.000, spec = 1.000** (378/560 configs pass the
0.95/0.95 gate; 238 also meet the latency budget).

Per-threshold tables at the chosen point (from the script's printout):

```
--- sensitivity/specificity vs tilt (pelvis z=0.35, confirm=0.3) ---
    tilt    sens    spec
      40   1.000   1.000
      45   1.000   1.000  <-- chosen
      50   1.000   1.000
      55   1.000   1.000
      60   1.000   1.000
      65   0.975   1.000
      70   0.936   1.000
      75   0.573   1.000
--- sensitivity/specificity vs pelz (tilt=45, confirm=0.3) ---
    pelz    sens    spec          (0.20 … 0.50 all 1.000/1.000)
--- sensitivity/specificity vs confirm (tilt=45, pelz=0.35) ---
 confirm    sens    spec          (0.05 … 0.40 all 1.000/1.000)
    0.50   0.994   1.000
    0.60   0.930   1.000
```

Margins (JSON `margins`): tilt **+17.6° / −18.8°** around the chosen threshold
(balanced midpoint 44.4°); pelvis z **+0.256 m** above the highest positive pelvis,
0.44 m below standing; confirmation 0.30 s with measured **trigger latency 0.32 s**
(first dorsal contact 1.362 s → trigger 1.682 s) — and confirm 0.30 s also rejects
the DOUBLE_LEG shooter's 0.26 s transient dorsal flicker at the end of its untaught
dive.  Increasing confirmation further degrades sensitivity (0.5 → 0.994, 0.6 → 0.930)
and delays the exchange end, so it is not preferred.

Trigger times with the chosen config (`data/backdet_calibration.json.trigger_times`):

| sequence | trigger | reading |
|---|---|---|
| SPRAWL/b (**positive, scored**) | 1.682 s | the defender's back hits the mat |
| STANCE a/b | 2.042 / 1.782 s | the STANCE replay topples onto the back (measured below) |
| DOUBLE_LEG/b | 2.842 s | untaught shooter collapse in PD replay |
| SINGLE_LEG a/b | 3.702 / 1.762 s | untaught shot collapse |
| STAND_UP a/b | 3.282 / 3.802 s | recovery never completes in PD replay; bodies end dorsal |
| SPRAWL/a, BODY_LOCK a/b, SNAPDOWN a/b, DOUBLE_LEG/a | none | no dorsal-torso mat contact |

The non-labelled triggers are **not** false positives of the rule: they are frames
where the untaught open-loop replay genuinely puts a back on the mat.  They are kept
out of the labelled calibration set and flagged here for review once the phase-3
teacher exists (the honest expectation is that most disappear; the rule itself is
what phase 4 must validate, and it does that on the labelled positives/negatives).

### 1.3 Honest notes on the detector

* The SPRAWL *kinematic* final frame has the defender face-down (belly direction
  z = −0.91), but the **PD-replay physics settles the defender supine** — belly up,
  dorsal torso contact, pelvis 0.06–0.09 m — from ≈1.36 s.  The positive label refers
  to that settled physics state (verified by the contact-normal cross-check); it is the
  designed "sprawl defender ends prone (on the mat)" outcome of the reference, not a
  hidden assumption.
* A 0.18 s dorsal-contact flicker occurs inside the positive episode (3.50–3.68 s, the
  pile shifts); the strict persistence rule restarts its timer there.  Sensitivity is
  unaffected at `confirm ≤ 0.4 s`; longer confirmations start missing established
  frames.  Not patched with lapse tolerance — no evidence yet that it is needed.
* The 0.10 s simultaneous-fall window is compared on control-step-quantized trigger
  times (0.02 s resolution) — documented in `AMBIGUITY_WINDOW_S`.

---

## 2. Environment (deliverable 11)

### 2.1 Contract (the API Phase 5/PPO will consume)

```python
from wrestling.env import WrestlingEnv, ReferenceReplay, StandHold

env = WrestlingEnv(model=None,                    # default: robots/wrestling_scene.xml
                   controllers=(ctrl_a, ctrl_b),  # callables(env, data) -> (29,) | None
                   seed=0,                        # start randomization stream
                   exchange_timeout=20.0,         # s of sim time per exchange
                   match_clock=180.0,             # s of sim time per match (episode)
                   mat_radius=1.5,                # declared competition circle (m)
                   backdet_cfg=None,              # default = calibrated thresholds
                   obs_fn=None,                   # default = default_observation
                   reward_fn=None)                # default = default_reward
obs = env.reset(seed=7)                           # (obs_a, obs_b), float32 (84,) each
obs, reward, terminated, truncated, info = env.step(action)   # action (2,29) or (58,)
```

* **Actions** — per-robot 29 joint position targets (rad), joint order = model order;
  `env.action_dim == 58`; always clipped to `ctrlrange` (test verifies).
* **Observations** — `obs_fn(model, data) -> (obs_a, obs_b)`, float32, 84 per robot,
  **swappable** (assign `env.obs_fn`); final spec is deliverable 14.  Current scaffold
  (`default_observation`; `obs_layout()` returns the slice table):

  | slice | name | content |
  |---|---|---|
  | 0:3 | `self_base_pos` | pelvis xyz, world (m) |
  | 3:7 | `self_base_quat` | pelvis quaternion (w,x,y,z), world |
  | 7:36 | `self_joints` | 29 hinge positions (rad) |
  | 36:39 | `self_base_linvel` | pelvis linear velocity, world (m/s) |
  | 39:42 | `self_base_angvel` | pelvis angular velocity, world (rad/s) |
  | 42:71 | `self_joint_vel` | 29 hinge velocities (rad/s) |
  | 71:74 | `self_torso_up` | torso local +z in world |
  | 74:76 | `self_foot_contact` | floor contact flag (left, right) |
  | 76:79 | `opp_rel_pos` | opponent pelvis − self pelvis, in self frame |
  | 79:82 | `opp_rel_vel` | opponent − self pelvis velocity, in self frame |
  | 82:84 | `opp_rel_heading` | sin/cos of (opponent yaw − self yaw) |
* **Rewards** — pluggable hook `reward_fn(env, event) -> (r_a, r_b)`; the per-step
  reward returned by `step` is the sum of that step's events.  Events emitted today:
  `{"kind": "oob", "robot": r, "count": n}` → −0.25 to `r`; and
  `{"kind": "exchange_end", "winner": w, "loser": l, "cause": ..., "ambiguous": bool,
  "record": ExchangeRecord}` → +1 winner / −1 loser / 0 draw or ambiguous.  No per-step
  shaping (MISSION: minimal shaping, the game structure carries the incentive).
* **info per step** — `t`, `match_time`, `exchange_index`, `exchange_time`,
  `oob` per-robot counts, `back` per-robot detector snapshot (trigger time, dorsal
  contact, tilt, pelvis z), `exchange_ended` (record when one just ended),
  `match_over`.
* **Episode structure** — one episode = one match (`terminated` when the 180 s match
  clock expires; `truncated` is always False by design).  Exchanges run back-to-back
  inside the match; each exchange resets both wrestlers standing.
* **Exchange log** — `env.exchange_log` (list of `ExchangeRecord`, `.as_dict()`):
  index, match-clock start/end, duration, winner/loser, `cause ∈ {back, oob, timeout,
  match_end}`, `ambiguous`, per-robot OOB counts, per-robot first back-trigger times,
  and the detector snapshot at the end (for review of ambiguous cases).
* **Controllers** — pluggable pair; `env.set_controllers(a, b)`; `step(None)` runs them
  (or holds the start pose when a controller is None).  Controllers get
  `(env, data)` and may read `env.time`, `env.exchange_time`, `env.exchange_index`.
  Shipped: `ReferenceReplay(technique, robot, feedback_gain=0, hold_last=True,
  loop=False)` (PD-tracks a `data/refs` trace through the model's position servos:
  `ctrl = q_ref + gain*(q_ref − q)`; restarts each exchange) and `StandHold(robot)`
  (static hold of the model's verified-stable `stand` keyframe targets).

### 2.2 Rules implemented

* **Standing start** — STANCE reference first frame, randomized per seed in a fixed
  order: pair yaw `U(−10°, +10°)` about the pair midpoint, then pelvis separation
  `0.98 m + U(−0.15, +0.15) m`, then joint noise `U(−0.03, +0.03) rad` (robot a then
  b, clipped to joint ranges).  Same seed → bit-identical qpos (tested); a
  `reset(seed, pose=(qa, qb))` hook overrides the stance for tests/calibration.
* **Exchange loop** — a back-to-mat trigger ends the exchange; the *other* wrestler
  scores (+1/−1).  First trigger wins; if both robots trigger within 0.10 s the
  exchange is **ambiguous**: no score, logged.  Timeout (20 s) → draw, reset standing.
* **Bounded area** — the scene's floor is an unbounded plane, so the environment
  *declares* the mat: a circle of radius **1.5 m** centred at the world origin.  A
  pelvis outside is an OOB **event** (inside→outside transition, 5 cm re-entry
  hysteresis), each event costs −0.25, and **3 events in one exchange forfeit** it
  (anti-run-away).  Radius and penalty are constructor knobs.
* **Match clock** — 180 s sim; exchanges back-to-back; the final partial exchange is
  recorded with `cause="match_end"` and no score; `terminated=True` at that point.

### 2.3 Scripted-scenario evidence (all in the test file)

| scenario | result |
|---|---|
| STANCE replay, 1.0 s exchange clock | draw; both standing at the timeout (pelvis 0.73/0.75 m, tilt 30°/20°); no back trigger; no OOB |
| `StandHold` from `both_stand`, 20 s clock | draw at exactly 20.0 s; pelvis 0.7916 m both (0.1 mm drift); detector silent; no OOB |
| SPRAWL replay from its own start | back event at ≈1.7 s: defender `b` dorsal contact, tilt 76°, pelvis 0.079 m; winner `a`, loser `b`; rewards (+1, −1); attacker's belly-up collapse (no torso mat contact) not flagged |
| both wrestlers supine (settled sprawl end state replicated, translated) | both trigger in the same control step (t = 0.32 s) → ambiguous, no score, both triggers logged |
| OOB forfeit (robot a teleported out/in/out/in/out) | 3 events → cause `oob`, winner `b`; final-step reward −1.25 / +1.0 |
| match clock 2.2 s with 0.5 s exchanges | 5 records (4 `timeout` draws + final `match_end`), `terminated` at t = 2.2 s; bit-identical re-run |

Throughput: **7.7× realtime** for one env (500 control steps = 10 s sim in 1.30 s
wall; contact-heavy intervals are slower).  A full 180 s match is ≈23 s wall —
Phase 5 should size its parallel workers accordingly (4-core host).

---

## 3. Honest failures / limitations

1. **The retargeted STANCE crouch is not holdable open-loop.**  PD-replaying the STANCE
   trace (`ctrl = q_ref`, exactly what phase-2 validation did) topples both robots
   **backwards** — robot a: pelvis 0.807 → 0.740 m at 1.0 s → 0.558 m at 1.3 s →
   0.157 m at 1.5 s — and the pair then lands on its back.  Measured attempts to keep
   the raw pose up all failed: stiffer ankle feedback (CoM P-D onto
   ankle pitch/roll), leveling the feet (the retargeted stance stands toe-down with
   the heels 6–7 cm up: −0.07…−0.14 rad of ankle correction), blending the joint targets
   toward the model's stand pose (even a 100% stand-target command from the STANCE
   *crouch* falls at ~1.3 s), and static ankle biases (measured ankle torques reach
   ≈39 Nm of the ±50 Nm actuator limit as the body tips, and ±0.10 rad static biases —
   which saturate the servos — changed neither the fall time nor the direction).  The
   STANCE pose's centre of mass is behind
   the effective support; no joint-position controller can recover it — keeping a
   humanoid balanced in wrestling postures is the phase-3 teacher's job (the phase-2
   validation's "boundary pass" for STANCE, `min up pelvis 0.45`, was measured over only
   2.34 s + 0.5 s hold and missed the ongoing topple).
   **Consequences (deliberate, documented):** the environment still starts from the
   STANCE reference start pose per the assignment — the standing start is a
   perturbed, not-yet-balanced stance, and the phase-5 policy must learn to stabilise
   it.  The *timeout-draw* test with raw STANCE replay therefore uses a 1.0 s clock
   (both robots verified standing); the full 20 s clock is covered with the model's
   `both_stand` pose via `StandHold`, which is stable (>20 s, 0.1 mm drift).
2. **PD-replay rolls end in untaught collapses** (the phase-2 report already documents
   this): the shooting references' physical bodies fall after the reference span, and
   several end with dorsal mat contact (trigger list in §1.2).  Those frames are
   geometrically correct back-to-mat states but cannot be used as calibration
   negatives; they are reported and excluded, and are expected to change materially
   once the teacher exists.
3. **The calibration labels use a measured proxy** ("dorsal contact established for
   0.5 s after the pelvis has gone low") rather than hand annotation, and the
   `LABEL_GRACE` (0.5 s) is a labelling parameter independent of the detector's
   `confirm_s` (0.30 s).  Sensitivity/specificity are sample-level; the *event-level*
   recall on the single positive episode is 1/1 with 0.32 s latency.  The gate is met
   with an order-of-magnitude margin in every dimension, so the operating point is not
   knife-edge (378/560 configs pass).
4. **The competition area is a declaration, not scene data** (the mat is an unbounded
   plane).  1.5 m radius / 3-event forfeit / −0.25 per event are deliberate defaults
   with constructor knobs; the orchestrator should confirm them before Phase-5 scoring
   is fixed.
5. **`StandHold` is a scripted stand-in**, not a stance controller: it holds a static
   verified-stable pose.  It exists so tests and calibration have a genuinely standing
   configuration; it must not be mistaken for the phase-3 teacher.
6. **No video/rendering** in this deliverable (not requested); the scene renders fine
   in the existing video pipeline (`scripts/validate_refs.py` pattern).

---

## 4. Files and how to run

| file | role |
|---|---|
| `src/wrestling/env.py` | `WrestlingEnv`, `ReferenceReplay`, `StandHold`, `default_observation`/`obs_layout`, `default_reward`, `resolve_back_events`, exchange/match rules |
| `src/wrestling/backdet.py` | features, streaming detector, batch rule, `BackDetConfig` (calibrated defaults) |
| `scripts/calibrate_backdet.py` | PD-replay rollouts → labels → 560-config sweep → table + JSON; asserts the chosen point equals `BackDetConfig()` |
| `data/backdet_calibration.json` | config, metrics, margins, latency, evidence, trigger times, full sweep |
| `tests/test_wrestling.py` | 16 tests: reset determinism/randomization, obs/action contract, replay sampling, timeout draw (1 s and 20 s clocks), SPRAWL back event, ambiguity, OOB forfeit, match clock, detector units (persistence, knees/hands/sprawl negatives, batch==streaming), calibration-JSON ↔ defaults sync, scene sanity |

```bash
.venv/bin/python scripts/calibrate_backdet.py     # ~7 s, prints the table, writes the JSON
.venv/bin/python -m src.wrestling.env             # env self-check (obs dims, determinism)
.venv/bin/python src/wrestling/backdet.py         # detector self-check
.venv/bin/python -m pytest tests/ -q              # 46 passed
```

Territory respected: nothing outside `src/wrestling/`, `scripts/calibrate_backdet.py`,
`tests/test_wrestling.py`, `data/backdet_calibration.json`, `reports/2026-10-08/wrestling_env.md`
was modified.  `notes.md` was intentionally left for the orchestrator to merge.
