# S5 demo prep — the capture pipeline and the executability checker

**Status: completed from the killed agent's code.** Agent `S5Prep` was stopped by the provider
outage; its module (`src/solo/demo.py`), its checker (`src/solo/exec_check.py`) and its entry point
(`scripts/solo_demo_capture.py`) exist, but its test file and this report did not. This pass fixed
one blocking defect, wrote the discrimination test, and measured what is below.

**Defect found and fixed.** `solo.demo.build_repaired_track` referenced `SEG_STAND` without
importing it — `NameError` on the first real call, i.e. the capture pipeline could not run at all.
`SEG_STAND` is now imported from `solo.exec_check` and the module self-check runs.

```
MUJOCO_GL=egl .venv/bin/python -c "import sys;sys.path.insert(0,'src');import runpy;runpy.run_module('solo.demo',run_name='__main__')"
MUJOCO_GL=egl .venv/bin/python -c "import sys;sys.path.insert(0,'src');import runpy;runpy.run_module('solo.exec_check',run_name='__main__')"
MUJOCO_GL=egl .venv/bin/python -m pytest tests/solo/test_demo_exec.py -q     # 7 passed
```

## 1. The reference and the repaired track (from the dry run)

`--source cem --dry-run --measure`:

| element | reference | frames | duration | net travel |
|---|---|---|---|---|
| entry (the penetration step) | `data/refs_video/shot_entry_full.npz` (`8826fc538a98ffcb`) | 411 | 8.20 s | 0.237 m |
| recover (the rise) | `data/refs_video/shot_recover.npz` (`bbf321ceb04db4c6`) | 399 | 7.96 s | 0.223 m |

Repaired track: **935 ticks / 18.68 s** = entry 411 + blend 40 + recover 399 + stand 60 + hold 25,
terminal stance 0.30 m wide × 0.79 m high, role `a`. The entry is the double-leg penetration step
(attacker) and the recover is the rise; the stand/hold segments are the operator's "every behaviour
returns to a good stance" requirement, built into the demo itself.

## 2. The nine criteria and their sources

Every threshold is derived from the reference's own landmarks or the measured stance — none is
hand-typed (each `CriterionResult` carries its provenance):

| criterion | what it measures | threshold / source |
|---|---|---|
| `no_fall` | first termination (the S1 `FallDetector` semantics) | pelvis/tilt/grounded, `src/solo/fall.py` |
| `contact_sequence` | lead foot plants → knee contacts → sustained rise | plant/knee timing ±0.30 s vs the reference; rise ≤ 12.0 s (the operator's rise is 7.96 s); the plant must be a genuine airborne→plant edge |
| `torso_pitch` | max pitch deviation on the reference segments | ≤ 15° (retarget fit: 0.149 m landmark error over the ~0.5 m torso lever ≈ 16.6°) |
| `com_margin` | CoM margin vs the support envelope | floor 0.0 m where the reference is inside its own envelope; `data/support_envelope.json` + the delivered L2 transients (0.026 m) |
| `knee_depth` | the penetration knee reaches the reference's depth | ≤ reference min knee height + 0.03 m |
| `no_saturation` | joint-limit margin, \|qvel\|, force fraction | `jnt_range`/`jnt_actfrcrange`; `retarget.solve.V_MAX` |
| `no_foot_slide` | loaded sole drift | ≤ 0.02 m (drill rubric B1; measured per-step drift 2.6–5.6 mm) |
| `travel` | entry travel in the heading frame | ≥ 60 % of the reference's own travel, lateral slack 0.15 m |
| `terminal_stance` | ends in a valid stance | pelvis in [0.70, 0.80] m, margin ≥ 0.02 m, tilt ≤ measured stance + margin, speed ≤ the drill settle gates, all four sole centres flat at rest |

## 3. Discrimination evidence (`tests/solo/test_demo_exec.py`, 7 tests, synthetic, no sim)

The positive control is a hand-constructed trace that satisfies **all nine** criteria (with real
foot footprints, so `no_foot_slide` is exercised rather than vacuous). Each rejection perturbs
exactly one thing and asserts the matching criterion fires:

| test | perturbation | criterion that must fire |
|---|---|---|
| `test_known_good_trace_passes_every_criterion` | — | none (`rep.ok`) |
| `test_fall_is_rejected` | grounded on the mat at the end | `no_fall` |
| `test_foot_slide_is_rejected` | 30 mm of loaded sole drift | `no_foot_slide` |
| `test_torso_collapse_is_rejected` | pitch 90° vs the 45° reference | `torso_pitch` |
| `test_knee_depth_miss_is_rejected` | knee never below 0.45 m | `knee_depth` |
| `test_missing_lead_plant_is_rejected` | the lead foot never leaves the ground | `contact_sequence` |
| `test_terminal_stance_is_required` | ends crouched at 0.55 m / 30° tilt | `terminal_stance` |

## 4. The capture: command and cost

Dry run (measured): CEM with 3 knots, 15 active dims, 450 samples/plan, ≤ 10 iters, 50-tick
windows at a 25-tick stride over 38 windows → **171,038 window rollouts ≈ 8.6 M sim ticks**. At the
drill's measured throughput on this box (≈ 490 ticks/s, `scripts/solo_brace_experiment.py`) that is
**≈ 5 h wall** — run it as a background service, not in a session.

```bash
# the real capture (NOT run in this session):
MUJOCO_GL=egl .venv/bin/python scripts/solo_demo_capture.py --source cem \
    --ref data/refs_video/shot_entry_full.npz \
    --recover-ref data/refs_video/shot_recover.npz \
    --out data/solo/demos --name shot_entry_full_a --check
```

`--check` runs `check_executability` on the capture and reports per-criterion pass/fail; a failing
capture is reported, never silently shipped.

## 5. Honest unknowns

* **Whether the source can actually hold the penetration step under the stance repairs** — the CEM
  source is the primary bet, the teacher is the documented fallback (`--source teacher`), and the
  checker's `contact_sequence`/`knee_depth` criteria are exactly what will decide it.
* **No captures exist on disk yet** (`data/solo/demos/` is absent): the pipeline is verified by its
  dry run and self-check, not by a finished demo. The first real capture is the S5 run itself.
* The checker's thresholds are calibrated on the reference's own landmarks and the measured stance;
  a *legitimate* demo that deviates for a good physical reason will still fail them until the
  tolerance is revisited with a captured example (stated rather than assumed away).
