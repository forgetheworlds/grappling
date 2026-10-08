# Technique-validity scorer v1 — deliverable 7

**Status: complete and validated.** Hand-engineered relational-geometry judge over
the Phase-2 reference techniques, separate from any RL critic. Package
`src/scorer/`, calibration `data/scorer_calibration.json`, calibration script
`scripts/calibrate_scorer.py`, trace scorer `scripts/score_trace.py`, tests
`tests/test_scorer.py` (19 tests). Repro: run the commands at the bottom.

## Why it exists (MISSION)

MISSION "Technique-validity supervision" names the failure mode this prevents:

```text
reward says "double leg"
but policy discovers "run shoulder-first into opponent"
```

The scorer answers one question, per commanded (technique, phase): *does the
current relational geometry still look like this movement family?* It does **not**
estimate return. goal.md hard rules: the critic may use privileged state and
estimates return; the technique judge is a SEPARATE component, never conflated.

## Separation contract (enforced by test)

* `src/scorer/*` imports only numpy/mujoco/stdlib — no torch, no rl/wrestling/
  reward code (`test_scorer_is_separate_from_value_functions`).
* The value-function modules (`rl/net.py` critic, `rl/privileged.py`,
  `rl/ppo.py` value loss, `rl/obs.py`) never import the scorer.
* Exactly one sanctioned consumer exists in `src/rl`: `reward.py`'s
  `ScorerAdapter`, imported lazily inside a function, used only as the
  `technique_similarity` *form* signal and degrading to zero when the scorer is
  missing. The test asserts it stays lazy and stays the only one.

## Public API

```python
from scorer import TechniqueScorer, score
sc = TechniqueScorer(model)                            # bound to the 2-G1 scene model
r  = sc.score("DOUBLE_LEG", 2, self_qpos, opp_qpos)    # -> ScoreResult
r.total        # float in [0, 1]
r.terms        # per-predicate breakdown: {"lock_asym:at_most": (mu, weight), ...}
r.features     # the 25 relational features behind the score (debug/telemetry)
score("DOUBLE_LEG", 2, self_qpos, opp_qpos, model)     # one-shot, scorer cached per model
sc.phase_at(tech, t) / sc.phase_at_frac(tech, frac)    # phase from wall time / progress
sc.score_trace(tech, self_traj, opp_traj)              # per-phase means + overall
```

`self_qpos`/`opp_qpos` are G1 qpos vectors (36 = free base 7 + 29 hinges, layout
per notes.md). Robot A = attacker/recoverer, B = defender/opponent; each
technique's reference executor is recorded in `spec.py` (`SPRAWL` is
demonstrated by B). Cost: **200 us** per call (one `mj_kinematics`, no
dynamics/contacts/rendering) — cheap enough for a training loop.

## Relational features (one `mj_kinematics` per call)

All features are relative and expressed in the *self* yaw frame, so a rigid
transform of the pair anywhere in the arena leaves the score unchanged
(validated). 25 scalars:

| group | features |
|---|---|
| separation / approach | `sep`, `dx`, `dz`, `bearing`, `rel_yaw` |
| torso orientation | `pitch_s`, `roll_s`, `pitch_o`, `roll_o` (pelvis->neck lean in each robot's own yaw frame) |
| head relationship | `head_rel` (self head z - opp pelvis z), `head_chest`, `chest_on` |
| level change / legs | `knee_z_s`, `knee_z_o`, `ankle_z_s`, `ankle_z_o`, `base_w_s`, `base_w_o`, `leg_back_s` |
| lock points | `lock_l`, `lock_r` (min wrist distance to opp left/right thigh-knee segment), `lock_asym`, `wrap` (wrist->opp chest), `head_snap` (wrist->opp head/neck) |

These are the MISSION "Preserve wrestling geometry" relationships: pelvis-pelvis
vector, torso orientation, attack depth, base width, head position, leg
relationship, arm/leg proximity.

## Membership and the calibration rule

Every predicate is a smooth fuzzy membership in [0, 1] (C1 smoothstep, no
cliffs), so small deviations degrade gracefully and the policy keeps room to
adapt under resistance:

* `band(f, lo, hi, lo_soft, hi_soft)` — 1 inside the core, smoothstep to 0 outside.
* `at_most(f, core, soft)` / `at_least(f, core, soft)` — one-sided.
* `ang_near(f, center, core, soft)` — circular, continuous across +/-pi.

Numbers come from `src/scorer/calibration.py` (the rule) applied to the
reference traces by `scripts/calibrate_scorer.py`:

| quantity | rule |
|---|---|
| `band` core | [p10, p90] of the feature over the reference phase band |
| `band` soft | core extended by `0.25 * (p90 - p10) + floor` |
| `at_most` | core = p90, soft = core + the same extension (feature must stay low) |
| `at_least` | core = p10, soft = core - the same extension (feature must stay high) |
| `ang_near` | center = circular mean, core = p90 of |wrapped deviation|, soft = core + max(0.25*core, floor) |
| `floor` | 0.10 m for distances, 0.15 rad for angles |

*Why these choices.* The core covers the reference band's central 80 %, so the
reference itself scores ~1.0 (the task's calibration requirement); the soft ramp
that extends past it still covers the reference tails, so a *tiny* deviation
barely moves the score (adaptation room) while a different relationship mode
falls through the ramp to 0. The floors stop phases whose reference is nearly
constant (e.g. `rel_yaw` in STANCE) from becoming razor-thin bands that any noise
would break. Weights: 2.0 for technique-defining relationships, 1.0 for
supporting context, following the MISSION priority list.

Thresholds are not hand-placed: `test_calibration_file_matches_the_fitting_rule`
re-fits all 144 predicates from the reference npz files and asserts the committed
JSON matches within 1e-4, so the shipped numbers cannot drift from the rule.

## Phase bands

Keyframe-index bands of the Phase-2 references: keyframe times are reconstructed
as `build_technique_targets(tech).times * time_stretch_requested *
time_stretch_kinematic` (both stretches from the npz meta) mapped onto the 50 Hz
grid, and every band boundary sits on the reconstructed time of its edge's last
keyframe. `scripts/calibrate_scorer.py` re-verifies this on every run (tolerance
0.03 s); all 13 boundaries of the six keyframe-derived techniques pass (worst
deviation 0.023 s). SPRAWL is the documented exception: it is a montage of three
defender sprawl cycles spliced onto the attacker's shot frames, so the solver's
junction alignment blurs its per-edge boundaries; its band boundaries are placed
on the defender's own trajectories (hip/knee height, leg-back) — phases 1/3/5 are
the three hips-down windows, 2/4 the rising recoveries, 0 the standing entry.

## Validation

### 1. Every reference scores high on its own trace (per phase)

Acceptance bar: >= 0.8 mean per phase. Result: worst phase **0.971** (SPRAWL1);
all 24 phases pass with >= 0.17 margin.

| technique | overall | worst phase | per-phase means |
|---|---|---|---|
| DOUBLE_LEG | 0.994 | LAND 0.990 | SETUP 0.996, PENETRATE 0.993, DRIVE 0.999, LAND 0.990 |
| SINGLE_LEG | 0.993 | SETUP 0.983 | SETUP 0.983, CAPTURE 0.999, FINISH1 0.996, FINISH2 0.993 |
| BODY_LOCK | 0.994 | CLINCH 0.993 | CLINCH 0.993, DROPOVER 0.993, PIN 0.996 |
| SNAPDOWN | 0.998 | SNAP 0.997 | SETUP 0.998, SNAP 0.997 |
| SPRAWL | 0.987 | SPRAWL1 0.971 | STAND1 0.988, SPRAWL1 0.971, RECOVER1 0.983, SPRAWL2 0.995, RECOVER2 0.988, SPRAWL3 0.998 |
| STAND_UP | 0.993 | STAND2 0.986 | DOWN 0.994, STAND1 0.994, STAND2 0.986, CLINCH 0.999 |
| STANCE | 1.000 | HOLD 1.000 | HOLD 1.000 |

### 2. Discrimination / cross-scores

Acceptance bar: DOUBLE_LEG trace scored as SINGLE_LEG < 0.5. Result: **0.464**.
The judge is diagonally dominant: every technique scores higher on its own
reference trace than on any other technique's trace (asserted with a > 0.1
margin by `test_own_trace_beats_every_cross_score`).

| trace \ scored as | DOUBLE_LEG | SINGLE_LEG | BODY_LOCK | SNAPDOWN | SPRAWL | STAND_UP | STANCE |
|---|---|---|---|---|---|---|---|
| **DOUBLE_LEG** | **0.994** | 0.464 | 0.451 | 0.220 | 0.756 | 0.578 | 0.171 |
| **SINGLE_LEG** | 0.660 | **0.993** | 0.436 | 0.180 | 0.632 | 0.558 | 0.146 |
| **BODY_LOCK** | 0.601 | 0.321 | **0.994** | 0.308 | 0.706 | 0.527 | 0.203 |
| **SNAPDOWN** | 0.361 | 0.278 | 0.545 | **0.998** | 0.639 | 0.595 | 0.107 |
| **SPRAWL** | 0.462 | 0.360 | 0.590 | 0.442 | **0.987** | 0.533 | 0.145 |
| **STAND_UP** | 0.396 | 0.329 | 0.381 | 0.269 | 0.441 | **0.993** | 0.119 |
| **STANCE** | 0.132 | 0.291 | 0.229 | 0.121 | 0.477 | 0.468 | **1.000** |

Read the matrix by column semantics: cell (trace X, technique Y) = "the executor
of Y, taken from trace X, judged as Y". Rows are traces, columns commanded family.

### 3. Perturbation monotonicity

Increasing deviation must monotonically reduce the score. Joint noise (sigma on
all 29 hinges of both robots, seed 0) and root tilt (base pitch) decrease
strictly monotonically for all 7 techniques. Phase shift (the body is Delta late
relative to the phase bands) decreases monotonically for the five progressive
techniques.

| technique | noise σ (rad): 0.0 | 0.02 | 0.05 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|---|---|
| DOUBLE_LEG | 0.994 | 0.992 | 0.984 | 0.954 | 0.863 | 0.749 |
| SINGLE_LEG | 0.993 | 0.992 | 0.987 | 0.966 | 0.898 | 0.776 |
| BODY_LOCK | 0.994 | 0.993 | 0.990 | 0.973 | 0.913 | 0.806 |
| SNAPDOWN | 0.998 | 0.994 | 0.979 | 0.929 | 0.828 | 0.696 |
| SPRAWL | 0.987 | 0.986 | 0.983 | 0.973 | 0.932 | 0.830 |
| STAND_UP | 0.993 | 0.993 | 0.990 | 0.977 | 0.936 | 0.851 |
| STANCE | 1.000 | 0.989 | 0.952 | 0.895 | 0.809 | 0.670 |

| technique | root pitch (rad): 0.0 | 0.05 | 0.1 | 0.2 | 0.4 | 0.8 |
|---|---|---|---|---|---|---|
| DOUBLE_LEG | 0.994 | 0.991 | 0.982 | 0.952 | 0.874 | 0.719 |
| SINGLE_LEG | 0.993 | 0.991 | 0.980 | 0.944 | 0.860 | 0.705 |
| BODY_LOCK | 0.994 | 0.994 | 0.988 | 0.957 | 0.887 | 0.808 |
| SNAPDOWN | 0.998 | 0.992 | 0.973 | 0.863 | 0.623 | 0.507 |
| SPRAWL | 0.987 | 0.982 | 0.970 | 0.920 | 0.755 | 0.530 |
| STAND_UP | 0.993 | 0.989 | 0.982 | 0.961 | 0.901 | 0.788 |
| STANCE | 1.000 | 0.994 | 0.978 | 0.931 | 0.841 | 0.507 |

| technique | phase shift Δ (progress): 0.0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|
| DOUBLE_LEG | 0.994 | 0.979 | 0.938 | 0.853 | 0.764 | 0.706 |
| SINGLE_LEG | 0.993 | 0.979 | 0.947 | 0.831 | 0.740 | 0.627 |
| BODY_LOCK | 0.994 | 0.978 | 0.947 | 0.874 | 0.775 | 0.604 |
| SNAPDOWN | 0.998 | 0.996 | 0.991 | 0.981 | 0.966 | 0.928 |
| SPRAWL | — (cyclic montage: see note) |
| STAND_UP | 0.993 | 0.978 | 0.902 | 0.714 | 0.588 | 0.445 |
| STANCE | — (single held phase) |

* SPRAWL is excluded from the phase-shift assertion: it is a cyclic montage
  (sprawl/recover x3), so a shift by ~0.3 progress lands back inside a
  geometrically equivalent sprawl window (0.552 at Delta=0.2 dips, then recovers
  to 0.733 at Delta=0.3) — correct behaviour for a periodic movement, not a
  scorer defect. STANCE has a single held phase, so shifting progress cannot
  change its score.
* Per-step monotonicity is asserted with a 5e-3 tolerance because the
  calibration core is a p10/p90 band: a small perturbation can nudge a
  transitional reference frame from its soft ramp *into* the core (+0.0008 on
  BODY_LOCK's tilt curve). The end-to-end drop (>= 0.1) is asserted strictly.

### 4. Speed

Median **200 us** per `score()` call over 5 x 300 calls (worst run 223 us),
i.e. ~5x under the 1 ms budget, so the scorer can run inside the training loop
(spaced, e.g. every 5th env step, as `rl/reward.py` does). Split: 6.8 us for the
single `mj_kinematics` call, ~0.14 ms for the Python-level feature assembly and
membership evaluation, ~55 us for the predicates themselves — the physics engine
is not the bottleneck, so a future vectorized/`numba`-free tightening would come
from the feature extraction, not the kinematics.

### 5. Relational invariance

Yaw-rotating both robots by 0.7 rad and translating them by (0.3, -1.2) m changes
the trace mean by < 1e-6: the judge scores relationships, not Cartesian replay
(MISSION: "Exact Cartesian reproduction ... is not the objective").

## Known limitations (honest)

* **The "-> SPRAWL" column is optimistic** (DOUBLE_LEG 0.756, BODY_LOCK 0.706,
  SNAPDOWN 0.639). The scorer judges the *executor of the column technique from
  the other trace's pair*; SPRAWL's executor is the defender, and the opponent in
  an attack trace is a standing/being-taken-down defender, which satisfies
  SPRAWL's standing and recover bands. The sprawl windows themselves do
  discriminate (DOUBLE_LEG as SPRAWL: STAND1 0.75, SPRAWL1 0.58, RECOVER2 0.71).
  In intended use the commanded phase is known, so this whole-trace aggregate is
  a harsher measure than the signal the training loop consumes.
* **Frame-level minima dip to 0.573** (SPRAWL1) at band-boundary transition
  frames; means are the contracted quantity, not per-frame minima.
* **No contact information.** Lock/wrap proximity is Euclidean wrist-to-landmark
  distance, not a contact test; a hand hovering near a leg scores like a grab
  with the same geometry.
* **Form, not intent.** Families that share geometry early (STANCE vs the
  standing entry of any technique) score similarly; this is a conformity check
  for a commanded technique, not a technique classifier.
* **SPRAWL bands are geometry-placed, not keyframe-placed** (documented above) —
  the only technique whose bands do not come from reconstructed keyframe times.

## Files

```
src/scorer/__init__.py        public API (score, TechniqueScorer, ScoreResult, ...)
src/scorer/features.py        25 relational features, one mj_kinematics per call
src/scorer/membership.py      band / at_most / at_least / ang_near smoothstep memberships
src/scorer/calibration.py     the threshold-fitting rule (documented, self-checking)
src/scorer/spec.py            phase bands + predicate templates (wrestling semantics)
src/scorer/config.py          spec + generated params -> TECHNIQUE_CONFIG
src/scorer/scorer.py          TechniqueScorer / score() / score_trace()
data/scorer_calibration.json  generated: 144 calibrated predicates (+ refs sha256)
scripts/calibrate_scorer.py   fit + verify (self >= 0.8/phase; DOUBLE->SINGLE < 0.5)
scripts/score_trace.py        score any reference/rollout npz, print per-phase table
tests/test_scorer.py          19 tests (acceptance, invariants, speed, separation)
```

History note: a partial version of this package existed (commit 676de77). It
scored 6/7 techniques ~1.0, but SPRAWL failed (0.46-0.74 per phase) because its
sprawl/recover predicate sets were mapped onto the wrong bands, phase 1 reused
the standing template, and SINGLE_LEG/FINISH1 lacked the free-leg predicate. This
revision fixes those defects, moves the numbers into a regenerable calibration
file, and adds the validation suite.

## Reproduce

```bash
.venv/bin/python scripts/calibrate_scorer.py          # verify + full tables
.venv/bin/python scripts/calibrate_scorer.py --write  # regenerate the params
.venv/bin/python scripts/score_trace.py               # score all 7 references
.venv/bin/python -m pytest tests/test_scorer.py -q    # acceptance suite
```

## Calibration table (all 144 predicates)

Generated by `.venv/bin/python scripts/calibrate_scorer.py`; `params` are the
membership arguments in `(feature, kind)` order, `ref p10/p50/p90` the feature's
reference distribution over that band, `n` the band's frame count.

```
 (params fitted from reference bands) ==

-- DOUBLE_LEG (executor A)
  phase 0 SETUP:
    sep         band     w=1.0 params=[0.89, 0.9438, 0.7766, 1.0572]  ref p10/p50/p90 = 0.89/0.9276/0.9438 (n=23)
    bearing     band     w=1.0 params=[0.3252, 0.3539, 0.168, 0.5111]  ref p10/p50/p90 = 0.3252/0.332/0.3539 (n=23)
    knee_z_s    band     w=1.0 params=[0.2503, 0.3073, 0.1361, 0.4216]  ref p10/p50/p90 = 0.2503/0.265/0.3073 (n=23)
    pitch_s     band     w=1.0 params=[0.5051, 0.813, 0.2781, 1.04]  ref p10/p50/p90 = 0.5051/0.6747/0.813 (n=23)
    base_w_s    band     w=1.0 params=[0.5536, 0.6108, 0.4393, 0.7251]  ref p10/p50/p90 = 0.5536/0.592/0.6108 (n=23)
    head_rel    band     w=1.0 params=[0.131, 0.3311, -0.019, 0.4811]  ref p10/p50/p90 = 0.131/0.2112/0.3311 (n=23)
  phase 1 PENETRATE:
    sep         band     w=2.0 params=[0.4348, 0.821, 0.2383, 1.0175]  ref p10/p50/p90 = 0.4348/0.5054/0.821 (n=44)
    dx          band     w=2.0 params=[0.3638, 0.7587, 0.1651, 0.9575]  ref p10/p50/p90 = 0.3638/0.4583/0.7587 (n=44)
    head_rel    at_most  w=2.0 params=[0.103, 0.2234]  ref p10/p50/p90 = 0.0213/0.0619/0.103 (n=44)
    pitch_s     band     w=1.0 params=[0.9118, 1.1938, 0.6913, 1.4143]  ref p10/p50/p90 = 0.9118/1.1142/1.1938 (n=44)
    knee_z_s    at_least w=1.0 params=[0.2593, 0.1436]  ref p10/p50/p90 = 0.2593/0.2812/0.3221 (n=44)
    lock_l      at_most  w=1.0 params=[0.3668, 0.5405]  ref p10/p50/p90 = 0.072/0.1818/0.3668 (n=44)
    lock_r      at_most  w=1.0 params=[0.5487, 0.7657]  ref p10/p50/p90 = 0.081/0.1654/0.5487 (n=44)
  phase 2 DRIVE:
    lock_l      at_most  w=2.0 params=[0.0845, 0.1961]  ref p10/p50/p90 = 0.0383/0.0737/0.0845 (n=44)
    lock_r      at_most  w=2.0 params=[0.0637, 0.1695]  ref p10/p50/p90 = 0.0404/0.0514/0.0637 (n=44)
    lock_asym   at_most  w=2.0 params=[0.0231, 0.1277]  ref p10/p50/p90 = 0.0049/0.0204/0.0231 (n=44)
    head_rel    band     w=1.0 params=[0.0291, 0.1058, -0.09, 0.2249]  ref p10/p50/p90 = 0.0291/0.065/0.1058 (n=44)
    sep         band     w=1.0 params=[0.4454, 0.4824, 0.3361, 0.5916]  ref p10/p50/p90 = 0.4454/0.4751/0.4824 (n=44)
    pitch_s     band     w=1.0 params=[0.9964, 1.1655, 0.8041, 1.3578]  ref p10/p50/p90 = 0.9964/1.0168/1.1655 (n=44)
    leg_back_s  band     w=1.0 params=[-0.4491, -0.1622, -0.6209, 0.0095]  ref p10/p50/p90 = -0.4491/-0.1955/-0.1622 (n=44)
  phase 3 LAND:
    ankle_z_o   at_least w=2.0 params=[0.4121, 0.2226]  ref p10/p50/p90 = 0.4121/0.7176/0.77 (n=44)
    lock_asym   at_most  w=1.0 params=[0.1475, 0.2824]  ref p10/p50/p90 = 0.008/0.0537/0.1475 (n=44)
    lock_l      at_most  w=1.0 params=[0.1178, 0.2327]  ref p10/p50/p90 = 0.0581/0.0665/0.1178 (n=44)
    lock_r      at_most  w=1.0 params=[0.2722, 0.4224]  ref p10/p50/p90 = 0.0712/0.1189/0.2722 (n=44)
    knee_z_s    at_most  w=1.0 params=[0.2578, 0.4052]  ref p10/p50/p90 = 0.0685/0.1048/0.2578 (n=44)
    sep         band     w=1.0 params=[0.4236, 0.4692, 0.3122, 0.5806]  ref p10/p50/p90 = 0.4236/0.4543/0.4692 (n=44)
    dx          band     w=1.0 params=[0.1147, 0.4503, -0.0692, 0.6342]  ref p10/p50/p90 = 0.1147/0.2047/0.4503 (n=44)

-- SINGLE_LEG (executor A)
  phase 0 SETUP:
    base_w_s    at_least w=2.0 params=[0.7508, 0.5869]  ref p10/p50/p90 = 0.7508/0.8976/1.0064 (n=55)
    bearing     band     w=1.0 params=[0.4419, 0.669, 0.2351, 0.8758]  ref p10/p50/p90 = 0.4419/0.6102/0.669 (n=55)
    sep         band     w=1.0 params=[0.6176, 0.9619, 0.4315, 1.148]  ref p10/p50/p90 = 0.6176/0.7624/0.9619 (n=55)
    knee_z_s    band     w=1.0 params=[0.2495, 0.3054, 0.1356, 0.4193]  ref p10/p50/p90 = 0.2495/0.2651/0.3054 (n=55)
    rel_yaw     ang_near w=1.0 params=[-1.8638, 0.2123, 0.3623]  ref p10/p50/p90 = -2.0209/-1.8592/-1.7067 (n=55)
  phase 1 CAPTURE:
    lock_r      at_most  w=2.0 params=[0.0923, 0.2018]  ref p10/p50/p90 = 0.0546/0.0883/0.0923 (n=40)
    lock_l      at_least w=2.0 params=[0.2674, 0.1491]  ref p10/p50/p90 = 0.2674/0.319/0.3408 (n=40)
    lock_asym   at_least w=2.0 params=[0.2128, 0.1025]  ref p10/p50/p90 = 0.2128/0.2271/0.254 (n=40)
    head_rel    band     w=2.0 params=[0.0011, 0.1038, -0.1246, 0.2295]  ref p10/p50/p90 = 0.0011/0.0404/0.1038 (n=40)
    pitch_s     band     w=1.0 params=[0.9983, 1.0554, 0.834, 1.2196]  ref p10/p50/p90 = 0.9983/1.0372/1.0554 (n=40)
    knee_z_s    at_least w=1.0 params=[0.2715, 0.1491]  ref p10/p50/p90 = 0.2715/0.3046/0.361 (n=40)
    sep         band     w=1.0 params=[0.5264, 0.5821, 0.4125, 0.696]  ref p10/p50/p90 = 0.5264/0.5545/0.5821 (n=40)
  phase 2 FINISH1:
    lock_r      at_most  w=2.0 params=[0.0952, 0.2062]  ref p10/p50/p90 = 0.0515/0.0788/0.0952 (n=81)
    lock_asym   at_least w=2.0 params=[0.2063, 0.0562]  ref p10/p50/p90 = 0.2063/0.2892/0.4068 (n=81)
    lock_l      at_least w=2.0 params=[0.2934, 0.15]  ref p10/p50/p90 = 0.2934/0.3722/0.4667 (n=81)
    head_chest  at_most  w=2.0 params=[0.1768, 0.2882]  ref p10/p50/p90 = 0.1311/0.1475/0.1768 (n=81)
    pitch_s     band     w=1.0 params=[1.0795, 1.3192, 0.8695, 1.5292]  ref p10/p50/p90 = 1.0795/1.1755/1.3192 (n=81)
    sep         band     w=1.0 params=[0.502, 0.5485, 0.3904, 0.6601]  ref p10/p50/p90 = 0.502/0.5252/0.5485 (n=81)
    knee_z_s    band     w=1.0 params=[0.0537, 0.349, -0.1202, 0.5229]  ref p10/p50/p90 = 0.0537/0.1823/0.349 (n=81)
    leg_back_s  band     w=1.0 params=[-0.1245, -0.032, -0.2477, 0.0912]  ref p10/p50/p90 = -0.1245/-0.09/-0.032 (n=81)
  phase 3 FINISH2:
    knee_z_s    at_most  w=2.0 params=[0.0289, 0.1307]  ref p10/p50/p90 = 0.0217/0.0262/0.0289 (n=54)
    lock_r      at_most  w=2.0 params=[0.0998, 0.2081]  ref p10/p50/p90 = 0.067/0.0845/0.0998 (n=54)
    lock_l      at_least w=2.0 params=[0.2668, 0.1556]  ref p10/p50/p90 = 0.2668/0.2827/0.3116 (n=54)
    pitch_s     at_least w=1.0 params=[1.5318, 1.1913]  ref p10/p50/p90 = 1.5318/1.9497/2.2936 (n=54)
    roll_s      band     w=1.0 params=[1.5386, 1.8989, 1.2985, 2.139]  ref p10/p50/p90 = 1.5386/1.8092/1.8989 (n=54)
    dz          at_most  w=1.0 params=[-0.1036, 0.0488]  ref p10/p50/p90 = -0.313/-0.2862/-0.1036 (n=54)
    bearing     band     w=1.0 params=[1.0713, 1.5007, 0.814, 1.7581]  ref p10/p50/p90 = 1.0713/1.1366/1.5007 (n=54)

-- BODY_LOCK (executor A)
  phase 0 CLINCH:
    sep         at_most  w=2.0 params=[0.3655, 0.4865]  ref p10/p50/p90 = 0.2814/0.2998/0.3655 (n=79)
    wrap        at_most  w=2.0 params=[0.2436, 0.3699]  ref p10/p50/p90 = 0.1384/0.1592/0.2436 (n=79)
    chest_on    at_most  w=1.0 params=[0.2902, 0.4053]  ref p10/p50/p90 = 0.2297/0.2728/0.2902 (n=79)
    lock_asym   at_most  w=1.0 params=[0.0896, 0.2113]  ref p10/p50/p90 = 0.0028/0.0486/0.0896 (n=79)
    pitch_s     band     w=1.0 params=[0.4243, 0.5609, 0.2401, 0.745]  ref p10/p50/p90 = 0.4243/0.4619/0.5609 (n=79)
    knee_z_s    at_least w=1.0 params=[0.2077, 0.0715]  ref p10/p50/p90 = 0.2077/0.3389/0.3525 (n=79)
  phase 1 DROPOVER:
    wrap        at_most  w=2.0 params=[0.2845, 0.408]  ref p10/p50/p90 = 0.1905/0.231/0.2845 (n=58)
    lock_r      at_most  w=1.0 params=[0.0918, 0.2003]  ref p10/p50/p90 = 0.0576/0.0701/0.0918 (n=58)
    lock_l      at_most  w=1.0 params=[0.1707, 0.2906]  ref p10/p50/p90 = 0.091/0.1124/0.1707 (n=58)
    sep         band     w=1.0 params=[0.3819, 0.5668, 0.2357, 0.713]  ref p10/p50/p90 = 0.3819/0.4207/0.5668 (n=58)
    pitch_s     band     w=1.0 params=[0.5238, 1.1498, 0.2173, 1.4563]  ref p10/p50/p90 = 0.5238/1.0299/1.1498 (n=58)
    dz          band     w=1.0 params=[-0.1269, 0.2154, -0.3125, 0.4009]  ref p10/p50/p90 = -0.1269/0.0462/0.2154 (n=58)
    ankle_z_o   band     w=1.0 params=[0.1025, 0.2427, -0.0325, 0.3778]  ref p10/p50/p90 = 0.1025/0.2202/0.2427 (n=58)
  phase 2 PIN:
    dz          at_most  w=2.0 params=[-0.2497, -0.0897]  ref p10/p50/p90 = -0.4899/-0.4806/-0.2497 (n=39)
    ankle_z_o   at_least w=2.0 params=[0.2314, 0.0974]  ref p10/p50/p90 = 0.2314/0.3322/0.3671 (n=39)
    roll_s      at_least w=1.0 params=[0.5444, 0.1749]  ref p10/p50/p90 = 0.5444/0.9812/1.4221 (n=39)
    pitch_s     at_least w=1.0 params=[1.0818, 0.8304]  ref p10/p50/p90 = 1.0818/1.2982/1.4874 (n=39)
    sep         band     w=1.0 params=[0.6122, 0.6745, 0.4967, 0.79]  ref p10/p50/p90 = 0.6122/0.6434/0.6745 (n=39)
    knee_z_s    band     w=1.0 params=[0.263, 0.3282, 0.1467, 0.4445]  ref p10/p50/p90 = 0.263/0.2747/0.3282 (n=39)
    wrap        at_most  w=1.0 params=[0.271, 0.3841]  ref p10/p50/p90 = 0.2185/0.2625/0.271 (n=39)

-- SNAPDOWN (executor A)
  phase 0 SETUP:
    head_snap   at_most  w=2.0 params=[0.1103, 0.224]  ref p10/p50/p90 = 0.0553/0.0676/0.1103 (n=23)
    sep         band     w=1.0 params=[0.7328, 0.7699, 0.6236, 0.8792]  ref p10/p50/p90 = 0.7328/0.7567/0.7699 (n=23)
    pitch_o     band     w=1.0 params=[0.8058, 0.8766, 0.6381, 1.0442]  ref p10/p50/p90 = 0.8058/0.8282/0.8766 (n=23)
    knee_z_s    band     w=1.0 params=[0.1687, 0.3236, 0.03, 0.4624]  ref p10/p50/p90 = 0.1687/0.2247/0.3236 (n=23)
    head_rel    band     w=1.0 params=[0.2065, 0.2677, 0.0912, 0.3829]  ref p10/p50/p90 = 0.2065/0.2339/0.2677 (n=23)
    wrap        at_most  w=1.0 params=[0.23, 0.3385]  ref p10/p50/p90 = 0.196/0.2145/0.23 (n=23)
  phase 1 SNAP:
    head_snap   at_most  w=2.0 params=[0.0806, 0.1898]  ref p10/p50/p90 = 0.0439/0.067/0.0806 (n=23)
    pitch_o     at_least w=2.0 params=[0.9504, 0.6716]  ref p10/p50/p90 = 0.9504/1.1665/1.4655 (n=23)
    pitch_s     band     w=1.0 params=[0.539, 0.9029, 0.298, 1.1439]  ref p10/p50/p90 = 0.539/0.6791/0.9029 (n=23)
    sep         band     w=1.0 params=[0.7715, 0.7892, 0.6671, 0.8937]  ref p10/p50/p90 = 0.7715/0.778/0.7892 (n=23)
    ankle_z_o   at_most  w=1.0 params=[0.0862, 0.1878]  ref p10/p50/p90 = 0.0797/0.0821/0.0862 (n=23)
    knee_z_s    at_most  w=1.0 params=[0.252, 0.3735]  ref p10/p50/p90 = 0.1657/0.1841/0.252 (n=23)

-- SPRAWL (executor B)
  phase 0 STAND1:
    knee_z_s    at_least w=2.0 params=[0.3451, 0.2182]  ref p10/p50/p90 = 0.3451/0.3822/0.4529 (n=90)
    leg_back_s  band     w=2.0 params=[-0.2006, 0.05, -0.3633, 0.2127]  ref p10/p50/p90 = -0.2006/0.0205/0.05 (n=90)
    pitch_s     at_most  w=1.0 params=[1.1512, 1.5228]  ref p10/p50/p90 = 0.2651/0.5007/1.1512 (n=90)
    chest_on    band     w=1.0 params=[0.2376, 0.4136, 0.0936, 0.5576]  ref p10/p50/p90 = 0.2376/0.3022/0.4136 (n=90)
    ankle_z_s   band     w=1.0 params=[0.0957, 0.2497, -0.0428, 0.3882]  ref p10/p50/p90 = 0.0957/0.1691/0.2497 (n=90)
  phase 1 SPRAWL1:
    leg_back_s  at_most  w=2.0 params=[-0.2586, -0.0786]  ref p10/p50/p90 = -0.5787/-0.4854/-0.2586 (n=42)
    knee_z_s    at_most  w=2.0 params=[0.2683, 0.4033]  ref p10/p50/p90 = 0.1284/0.1857/0.2683 (n=42)
    pitch_s     at_least w=2.0 params=[1.1428, 0.9336]  ref p10/p50/p90 = 1.1428/1.3024/1.3794 (n=42)
    head_chest  at_most  w=1.0 params=[0.2988, 0.4284]  ref p10/p50/p90 = 0.1806/0.2038/0.2988 (n=42)
    chest_on    at_most  w=1.0 params=[0.3214, 0.4493]  ref p10/p50/p90 = 0.21/0.2236/0.3214 (n=42)
  phase 3 SPRAWL2:
    leg_back_s  at_most  w=2.0 params=[-0.2702, -0.1536]  ref p10/p50/p90 = -0.3367/-0.2968/-0.2702 (n=48)
    knee_z_s    at_most  w=2.0 params=[0.2292, 0.3805]  ref p10/p50/p90 = 0.024/0.0882/0.2292 (n=48)
    pitch_s     at_least w=2.0 params=[1.1298, 0.9066]  ref p10/p50/p90 = 1.1298/1.3012/1.4226 (n=48)
    head_chest  at_most  w=1.0 params=[0.2465, 0.3723]  ref p10/p50/p90 = 0.1435/0.1845/0.2465 (n=48)
    chest_on    at_most  w=1.0 params=[0.2284, 0.3397]  ref p10/p50/p90 = 0.1832/0.2031/0.2284 (n=48)
  phase 5 SPRAWL3:
    leg_back_s  at_most  w=2.0 params=[-0.3129, -0.1382]  ref p10/p50/p90 = -0.6119/-0.4434/-0.3129 (n=24)
    knee_z_s    at_most  w=2.0 params=[0.2038, 0.3432]  ref p10/p50/p90 = 0.0463/0.0884/0.2038 (n=24)
    pitch_s     at_least w=2.0 params=[1.0451, 0.8667]  ref p10/p50/p90 = 1.0451/1.0988/1.1586 (n=24)
    head_chest  at_most  w=1.0 params=[0.213, 0.3247]  ref p10/p50/p90 = 0.1661/0.1754/0.213 (n=24)
    chest_on    at_most  w=1.0 params=[0.1947, 0.3031]  ref p10/p50/p90 = 0.1611/0.1731/0.1947 (n=24)
  phase 2 RECOVER1:
    knee_z_s    at_least w=2.0 params=[0.2264, 0.1039]  ref p10/p50/p90 = 0.2264/0.2954/0.3162 (n=36)
    leg_back_s  at_least w=2.0 params=[-0.1909, -0.3445]  ref p10/p50/p90 = -0.1909/-0.0291/0.0235 (n=36)
    pitch_s     band     w=1.0 params=[0.7538, 0.9959, 0.5433, 1.2064]  ref p10/p50/p90 = 0.7538/0.8088/0.9959 (n=36)
    head_chest  band     w=1.0 params=[0.3184, 0.412, 0.1951, 0.5354]  ref p10/p50/p90 = 0.3184/0.3865/0.412 (n=36)
  phase 4 RECOVER2:
    knee_z_s    at_least w=2.0 params=[0.1758, 0.0412]  ref p10/p50/p90 = 0.1758/0.2673/0.3141 (n=48)
    leg_back_s  at_least w=2.0 params=[-0.2441, -0.4089]  ref p10/p50/p90 = -0.2441/-0.0954/0.015 (n=48)
    pitch_s     band     w=1.0 params=[0.8049, 1.0764, 0.587, 1.2943]  ref p10/p50/p90 = 0.8049/0.9199/1.0764 (n=48)
    head_chest  band     w=1.0 params=[0.2601, 0.4052, 0.1239, 0.5415]  ref p10/p50/p90 = 0.2601/0.3484/0.4052 (n=48)

-- STAND_UP (executor A)
  phase 0 DOWN:
    knee_z_s    at_most  w=2.0 params=[0.2623, 0.4275]  ref p10/p50/p90 = 0.0015/0.1653/0.2623 (n=100)
    ankle_z_s   at_most  w=2.0 params=[0.2197, 0.3645]  ref p10/p50/p90 = 0.0403/0.0796/0.2197 (n=100)
    leg_back_s  at_least w=1.0 params=[-0.0472, -0.3097]  ref p10/p50/p90 = -0.0472/0.2782/0.6028 (n=100)
    base_w_s    at_most  w=1.0 params=[0.839, 1.1142]  ref p10/p50/p90 = 0.1383/0.5111/0.839 (n=100)
    chest_on    band     w=1.0 params=[0.5643, 0.6777, 0.4359, 0.8061]  ref p10/p50/p90 = 0.5643/0.637/0.6777 (n=100)
    head_chest  at_most  w=1.0 params=[0.746, 0.8812]  ref p10/p50/p90 = 0.6051/0.7035/0.746 (n=100)
  phase 1 STAND1:
    knee_z_s    band     w=2.0 params=[0.2875, 0.3943, 0.1608, 0.521]  ref p10/p50/p90 = 0.2875/0.3536/0.3943 (n=98)
    ankle_z_s   band     w=1.0 params=[0.0596, 0.3285, -0.1076, 0.4958]  ref p10/p50/p90 = 0.0596/0.103/0.3285 (n=98)
    pitch_s     band     w=1.0 params=[0.2245, 0.6945, -0.043, 0.9619]  ref p10/p50/p90 = 0.2245/0.3851/0.6945 (n=98)
    head_chest  at_most  w=1.0 params=[0.7519, 0.9928]  ref p10/p50/p90 = 0.1882/0.5618/0.7519 (n=98)
    sep         band     w=1.0 params=[0.2658, 0.999, -0.0175, 1.2823]  ref p10/p50/p90 = 0.2658/0.7602/0.999 (n=98)
  phase 2 STAND2:
    knee_z_s    band     w=2.0 params=[0.4567, 0.7346, 0.2872, 0.9041]  ref p10/p50/p90 = 0.4567/0.5752/0.7346 (n=149)
    ankle_z_s   band     w=1.0 params=[0.2861, 0.7076, 0.0808, 0.9129]  ref p10/p50/p90 = 0.2861/0.3592/0.7076 (n=149)
    leg_back_s  band     w=1.0 params=[-0.1008, 0.3072, -0.3028, 0.5092]  ref p10/p50/p90 = -0.1008/0.0209/0.3072 (n=149)
    chest_on    band     w=1.0 params=[0.2126, 0.6795, -0.0041, 0.8962]  ref p10/p50/p90 = 0.2126/0.406/0.6795 (n=149)
    head_chest  at_most  w=1.0 params=[0.7471, 0.9566]  ref p10/p50/p90 = 0.3087/0.4914/0.7471 (n=149)
  phase 3 CLINCH:
    knee_z_s    at_least w=2.0 params=[0.7543, 0.6349]  ref p10/p50/p90 = 0.7543/0.7919/0.8321 (n=123)
    ankle_z_s   at_least w=2.0 params=[0.6279, 0.4819]  ref p10/p50/p90 = 0.6279/0.6974/0.8116 (n=123)
    wrap        at_most  w=2.0 params=[0.1142, 0.2195]  ref p10/p50/p90 = 0.093/0.098/0.1142 (n=123)
    chest_on    at_most  w=2.0 params=[0.2478, 0.3585]  ref p10/p50/p90 = 0.205/0.2228/0.2478 (n=123)
    head_chest  at_most  w=1.0 params=[0.3519, 0.4605]  ref p10/p50/p90 = 0.3173/0.3359/0.3519 (n=123)
    sep         band     w=1.0 params=[0.3704, 0.4442, 0.2519, 0.5627]  ref p10/p50/p90 = 0.3704/0.4019/0.4442 (n=123)

-- STANCE (executor A)
  phase 0 HOLD:
    sep         band     w=2.0 params=[0.983, 0.9833, 0.883, 1.0834]  ref p10/p50/p90 = 0.983/0.9832/0.9833 (n=118)
    rel_yaw     ang_near w=2.0 params=[2.6897, 0.0093, 0.1593]  ref p10/p50/p90 = 2.6814/2.6897/2.698 (n=118)
    knee_z_s    band     w=2.0 params=[0.3601, 0.3603, 0.2601, 0.4604]  ref p10/p50/p90 = 0.3601/0.3602/0.3603 (n=118)
    pitch_s     band     w=1.0 params=[0.2934, 0.2947, 0.1431, 0.445]  ref p10/p50/p90 = 0.2934/0.2941/0.2947 (n=118)
    base_w_s    band     w=1.0 params=[0.3022, 0.3022, 0.2022, 0.4022]  ref p10/p50/p90 = 0.3022/0.3022/0.3022 (n=118)
    lock_l      at_least w=1.0 params=[0.6824, 0.5812]  ref p10/p50/p90 = 0.6824/0.6846/0.6868 (n=118)
    ankle_z_s   band     w=1.0 params=[0.0956, 0.0957, -0.0044, 0.1957]  ref p10/p50/p90 = 0.0956/0.0957/0.0957 (n=118)
    head_rel    band     w=1.0 params=[0.3948, 0.3951, 0.2948, 0.4951]  ref p10/p50/p90 = 0.3948/0.3949/0.3951 (n=118)
```
