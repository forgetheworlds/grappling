# T1 gate calibration — the balance gate is unattainable as written, and "held-out" was vacuous

**Date:** 2026-10-08 · **Author:** T1GateCal (delegated) · **Status:** measured split + proposal + discrimination test; **the `src/solo/eval.py` and `src/solo/pushes.py` edits are APPLIED** (§7) after T2Gate committed its locomotion gate (`377009c`).

Everything below is re-derived from stored artifacts, the shipped model, and the training code — not from the ticket's prose. Where a quoted premise differs from what is on disk it is called out loudly (§1.3, §2.2).

---

## 1. Re-verification of the quoted numbers

### 1.1 Gate definition

Source: `src/solo/eval.py:307-326` (`GATES["balance"]`, "T1_balance"). File state at check time: **dirty** (` M`), T2Gate's locomotion work in-tree, uncommitted (`git log -1 -- src/solo/eval.py` = `25069de`).

| # | metric | op | threshold | line | note (verbatim, abridged) |
|---|---|---|---|---|---|
| 1 | `fall_rate` | `<=` | 0.05 | 309 | falls across the full battery |
| 2 | `max_recoverable_impulse_heldout` | `>=` | 16.0 | 310-313 | "provenance: StandHold measured **0.0** held-out, and 16 N*s is beyond the analytic non-stepping ceiling (~13 N*s)" |
| 3 | `mean_upright` | `>=` | 0.95 | 314-315 | "provenance: StandHold measured **0.923**" |
| 4 | `time_to_stability_mean` | `<=` | 1.0 | 316-317 | baseline table |
| 5 | `com_offset_max` | `<=` | 0.20 | 318-320 | baseline table |
| 6 | `recovery_success_rate` | `>=` | 0.9 | 321-322 | stabilised to stance after each push |
| 7 | `fall_rate_heldout` | `<=` | 0.10 | 323-324 | held-out magnitudes only |

### 1.2 Stored baselines — reproduced exactly

Source `data/solo/metrics/t1_gate_baselines.json` → `controllers.<name>.*`, reproduced to the last digit by running the **production** `solo.eval.aggregate` over the stored per-episode summaries (`data/solo/metrics/balance_t1gate_<name>_s0.summary.json` → `episodes[]`).

| metric | stand_hold | zero_action | random_init_policy |
|---|---|---|---|
| `mean_upright` | **0.849118** | 0.191578 | 0.176118 |
| `fall_rate` | **0.291667** | 0.354167 | 0.8125 |
| `fall_rate_heldout` | **0.958333** | 0.875 | 0.875 |
| `recovery_success_rate` | **0.25** | 0.0 | 0.0 |
| `max_recoverable_impulse_heldout` | **16.0** | 0.0 | 0.0 |
| `com_offset_max` | **0.1352** | 0.5615 | 0.6769 |
| `time_to_stability_mean` | **0.1326** | null | null |
| `verdict` | `not_certified` | `not_certified` | `not_certified` |

Every number quoted in the ticket is **confirmed**. Battery (`t1_gate_baselines.json` → `battery`): magnitudes `[4,8,12,16,20,25]`, directions 8, heights `[0.79,0.95,1.10]`, `n_pushes` 48, `train_max_impulse` 12.0.

### 1.3 Premise check — four discrepancies / defects

**(A) `mean_upright` provenance is stale *and* self-refuting.** The gate cites "StandHold measured 0.923". That 0.923 exists — but on a *different, smaller* battery: `reports/2026-10-08/solo_env.md:189` records it for "battery (4,6,8,10,12 N·s × 8 dirs, 40 eps)", and `data/solo/metrics/balance_stand_hold_s0.summary.json` → `aggregate.mean_upright = 0.922786` (40 episodes, **in-train magnitudes only, no held-out magnitudes at all**). The threshold `0.95` sits **above** that reference value, i.e. the gate was already calibrated to reject its own reference before the battery was extended. On the shipped 48-episode battery the reference measures **0.849118**. So `0.95 > 0.923 > 0.849118`: the uprightness bar is unattainable by construction, and always was.

**(B) The impulse-provenance note is flatly contradicted by the stored data.** `eval.py:311` claims "provenance: StandHold measured **0.0** held-out". The stored value is `max_recoverable_impulse_heldout = **16.0**`, because the scripted, **non-stepping** reference (a rigid pose hold, `act_delta_mean = 0.0`) genuinely recovered one held-out 16 N·s push (§2). The note understates the reference's held-out recovery by the entire bar.

**(C) The "~13 N·s" ceiling is wrong as a scalar.** Derived from our own model the ceiling is **direction-dependent, 6.7–20.7 N·s** (§2.2). The `~13 N·s` figure in `eval.py:313` and `src/solo/pushes.py:18-19` must be corrected.

**(D) `fall_rate` and `fall_rate_heldout` use different predicates** for the same quantity. `eval.py:584` computes `fall_rate` with `termination == "fall"`, but `eval.py:630-632` computes `fall_rate_heldout` with `termination is not None` (any termination: fall **or** dorsal). On the reference's held-out set these are 0.0625 and 0.5625 respectively. The `<= 0.10` bar is unattainable under the shipped predicate regardless of the split.

---

## 2. The non-stepping envelope, measured

Definition in code — `solo/eval.py:274-283`:

```python
def _recovered_episode(e):          # used by max_recoverable_impulse / recovery_success_rate
    if e.get("termination") is not None: return False
    return bool(e.get("stable"))    # upright + stance height + base speed < 0.15 m/s, held 0.2 s
```

Per-magnitude recovery of `stand_hold` on the stored battery (8 episodes each):

| magnitude (N·s) | held-out? | recovered | rate | `mean_upright` |
|---|---|---|---|---|
| 4.0 | no | 6/8 | 0.750 | 0.9616 |
| 8.0 | no | 3/8 | 0.375 | 0.9045 |
| 12.0 | no | 2/8 | 0.250 | 0.8737 |
| 16.0 | **yes** | 1/8 | 0.125 | 0.8274 |
| 20.0 | **yes** | 0/8 | 0.000 | 0.7860 |
| 25.0 | **yes** | 0/8 | 0.000 | 0.7416 |

**Largest impulse stand_hold actually recovers, held out: 16.0 N·s — on 1 of 8 directions (12.5%).** The highest magnitude with recovery rate ≥ 0.25 is **12 N·s**; the derivation in §2.2 shows why (it is the in-band / training-cap magnitude).

### 2.1 The split the code uses *today*, and why it fails once the battery is capped

Source `src/solo/eval.py:182-184` (now `:406`), `231`, `340-347`, `379-384`:

```python
held = float(m) > float(train_max_impulse) + 1e-9      # 4/8/12 in-train; 16/20/25 held out
```

Today "held out" = {16, 20, 25} N·s, meaning **magnitudes withheld from training**. This is the meaning the gate's note ("held-out magnitudes only") carries. It agrees with the code — but it has two fatal consequences, and the second is what the operator flagged:

1. A criterion tagged "held-out" is really a criterion about **out-of-training-capacity** behaviour (T3/T2 stepping capability), and is **identically vacuous on the non-stepping subset** (0 held-out episodes ⇒ `max_recoverable_impulse_heldout = 0.0`; observed in §4).
2. **If you cap the T1 battery to the band and keep this definition, the held-out set becomes EMPTY and every held-out criterion trivially passes** — an empty set satisfies a threshold and the discrimination test would not notice. So "held-out" must be redefined *within* the envelope, as held-out **conditions** (§2.3).

### 2.2 Derived non-stepping ceiling — computed from OUR model

The claim "the analytic non-stepping ceiling for a 33.3 kg G1 is ~13 N·s" (`pushes.py:18-19`, `eval.py:313`) is **UNVERIFIED and wrong as a scalar**. Derived from the shipped model and measured geometry (capture-point rejection, Yang et al. 2020, arXiv:2002.02991 eq. 6):

```
J_reject ≈ m · dCOP · sqrt(g / z_c)
```

| symbol | value | provenance |
|---|---|---|
| `m` | **33.3411 kg** | `robots/g1/g1.xml` via `solo.scene.load_solo_model()` → `Σ body_mass` over 32 bodies. Matches "33.3 kg G1". |
| `z_c` (CoM height) | **0.6919 m** | whole-body `data.subtree_com[0][2]` in the `a_stand` keyframe pose (`solo.scene.stand_frame`). |
| support hull | **0.170 m deep × 0.297 m wide** | convex hull of the 8 sole contact spheres (`robots/g1/g1.xml` geoms 15-18/30-33: heel x=−0.05, toe x=+0.12, y=±0.025/±0.03, r=5 mm) projected to the mat in the stand pose. Per-foot footprint 0.175×0.06 m (matches `support_envelope.md:126-127`). |
| hull half-widths | lateral **0.1485 m**; sagittal **+0.1167 m** (toe) / **−0.0533 m** (heel) | same. The CoM sits **heel-ward of the foot midpoint** (`drill.md:92`: "pelvis behind the mid-foot"), so the sagittal margins are asymmetric. |
| measured drill stance | **0.315 m** wide | `drill.md:127` / `data/drill/stance_report.json` (half-widths then 0.1890 lateral, +0.1168/−0.0538 sagittal). |

Per push yaw (`solo.eval.battery_pushes`, 8 evenly spaced world yaws):

| yaw | dCOP (m) | J_reject (N·s) | | yaw | dCOP (m) | J_reject (N·s) |
|---|---|---|---|---|---|---|
| 0° (+x) | 0.1167 | **14.65** | | 180° (−x) | 0.0533 | **6.69** |
| 45° | 0.1651 | **20.72** | | 225° | 0.0754 | **9.46** |
| 90° | 0.1450 | **18.20** | | 270° | 0.1452 | **18.22** |
| 135° | 0.0754 | **9.46** | | 315° | 0.1651 | **20.72** |

**Ceiling range 6.69 … 20.72 N·s (median 16.43)** at the solo stance width; **6.75 … 23.15 (median 17.71)** at the measured drill width 0.315 m.

So the operator's expected "~20 N·s" is right **for lateral/oblique directions**; the binding direction — a sagittal push driving the CoM back over the heels — is only **6.7 N·s**. The scalar "~13 N·s" is not the ceiling in any direction.

**This matches the measured recovery curve** (§2): 16 N·s lies inside the lateral/oblique envelope (18.2–20.7) and outside the sagittal one (6.7–14.7) — exactly why the reference recovers 1/8 at 16 N·s, 0/8 at 20 (only 20.72 yaws marginally reach it) and 0/8 at 25 (beyond every direction). The reference's *one* held-out recovery is therefore consistent with the non-stepping model, and **the "requires stepping" reading is true only for 20/25 N·s and the sagittal component of 16**, not for 16 N·s laterally.

**Consequence for the cap.** The ceiling is above 16 N·s in the lateral/oblique directions, so **the magnitude cap is about matching the TRAINING distribution, not a physics limit** — those are two different justifications and only the training-match one survives for 16 N·s (the physics-bound one holds for 20/25). The train/test gap is real: the curriculum caps at 12 N·s; the gate tests 16/20/25. (And even 12 N·s exceeds the weakest-direction ceiling of 6.7 N·s, so "trained on ≤12" does not mean "trained inside the envelope in all directions".)

### 2.3 Training push distribution, and the held-out CONDITIONS

Read from `src/solo/curriculum.py:36-80` (`PushCurriculum`): magnitudes `(2,4,6,8,10,12)` N·s unlocked with progress; up to `max_directions=8` world yaws (all 8 at full progress); **application height FIXED at 0.95 m** (`height: float = 0.95`; docstring: "application height fixed at height (0.95 m chest) -- height variation stays an evaluation axis"); 1 push per episode (2 once progress ≥ 0.5), t ≈ 1.0 s. The trainer only overrides `start_steps`/`warmup_steps`/`seed` (`src/solo/train.py:134-137`) — never the height.

Per axis, is it disjoint inside the band?

| axis | training values | in-band battery values | disjoint? |
|---|---|---|---|
| magnitude | {2,4,6,8,10,12} | {4,8,12} | **no** — battery ⊆ training |
| direction | all 8 yaws at full progress | all 8 yaws | **no** |
| seed | `seed*1_000_003 + episode_seed` | `default_rng(0)` jitter | only RNG |
| **application height** | **{0.95} only** | {0.79, 0.95, 1.10} | **YES — {0.79, 1.10} are never trained** |

**Chosen held-out conditions within the envelope = application heights {0.79, 1.10} m** (16 of the 24 in-band episodes; exactly 8 episodes at each of the three heights). They are non-empty and disjoint from the training conditions (`(magnitude, direction, height)` sets are disjoint), asserted in the test.

---

## 3. Proposal — split T1 by capability, with a real held-out set

Rules applied: **T1 certifies NON-STEPPING dynamic balance** on the in-band battery (magnitudes ≤ 12 N·s); the uprightness bar is **below** the measured reference hold; **held-out = held-out CONDITIONS (off-training heights)**, non-empty and disjoint; the **stepping-recovery bars move to the stance/footwork stage (T3)**; 16/20/25 N·s are **reported, never gated**. No threshold is tuned so v5 passes.

### 3.1 Proposed T1 criteria

| # | metric | op | threshold | provenance |
|---|---|---|---|---|
| 1 | `fall_rate` | `<=` | **0.05** | as-written bar, applied to the in-band battery (all heights). Reference **0.041667**; trivials 0.25 / 0.75. |
| 2 | `mean_upright` | `>=` | **0.84** | **below** both the in-band reference hold **0.913239** and the stored full-battery hold **0.849118** — never above the reference under either reading. Trivials 0.132 / 0.173. |
| 3 | `time_to_stability_mean` | `<=` | **1.0** | unchanged; reference **0.1233**; trivials `null` ⇒ fail. |
| 4 | `com_offset_max` | `<=` | **0.20** | unchanged; reference **0.1236**; trivials 0.5467 / 0.6517. |
| 5 | `fall_rate_heldout` | `<=` | **0.10** | **redefined**: true fall rate (`termination=="fall"`, aligning its predicate with `fall_rate` — §1.3D) over the **held-out conditions** = off-training heights {0.79,1.10} inside the band. Reference **0.0625** (1/16); zero_action 0.1875; random_init_policy 0.75; v5@100k 0.375. The 16/20/25 N·s magnitudes are REPORTED-ONLY and do not gate. |

**Removed from T1** (moved to T3, §3.2): `recovery_success_rate`, `max_recoverable_impulse`, `max_recoverable_impulse_heldout`. Also **not** added: any metric that credits a *static* controller with "recovery" — a rigid stand satisfies `recovery_success_rate` on small pushes, so leaving it in T1 would label stiff survival as active recovery (test §5c). `fall_rate_heldout` is kept but is a *stability* metric on held-out conditions, not a recovery claim.

### 3.2 Moved to T3 stance/footwork (stepping stage) — for the T3 owner to adopt verbatim

```python
# inside GATES["stance"] (T3_stance) criteria tuple:
Criterion("max_recoverable_impulse", ">=", 16.0,
          "N*s; REQUIRES re-calibration against a stepping-capable reference (v5/T3); "
          "moved out of T1 (data/solo/metrics/t1_gate_baselines.json)"),
Criterion("max_recoverable_impulse_heldout", ">=", 16.0,
          "N*s; recovered = stable to stance (not merely non-terminated); moved out "
          "of T1 (stand_hold recovers only 1/24 held-out pushes, fall_rate_heldout "
          "0.958333); REQUIRES re-calibration (v5/T3)"),
Criterion("fall_rate_heldout", "<=", 0.10,
          "held-out magnitudes only (magnitudes > train_max_impulse)"),
Criterion("recovery_success_rate", ">=", 0.9,
          "stabilised to stance after each held-out push"),
```

Provenance: §2 (measured envelope), §2.2 (derived ceiling 6.7–20.7 N·s: 20/25 N·s exceed it in every direction ⇒ stepping genuinely required), §3.3. The 16.0 thresholds are placeholders carried from the defect and **must not** be re-derived from stand_hold.

### 3.3 What each controller scores (production `aggregate` over stored summaries)

In-band battery = 24 eps (magnitudes ≤ 12 N·s, all heights); held-out-condition subset = 16 eps (heights 0.79/1.10).

| controller | criterion | value | as-written (full batt.) | proposed |
|---|---|---|---|---|
| **stand_hold** | `fall_rate <= 0.05` | 0.041667 | ✗ (0.291667) | **✓** |
| | `mean_upright >= 0.95`→`0.84` | 0.913239 | ✗ (0.849118 < 0.95) | **✓** |
| | `time_to_stability_mean <= 1.0` | 0.1233 | ✓ (0.1326) | **✓** |
| | `com_offset_max <= 0.20` | 0.1236 | ✓ (0.1352) | **✓** |
| | `fall_rate_heldout <= 0.10` (condition) | 0.0625 | ✗ (0.958333, magnitude) | **✓** |
| | `recovery_success_rate >= 0.9` | 0.458333 | ✗ (0.25) | moved to T3 |
| | `max_recoverable_impulse_heldout >= 16` | 0.0 (vacuous in band) | ✓ (16.0) | moved to T3 |
| | **verdict** | | **not_certified** | **certified (T1)** |
| **zero_action** | `fall_rate` / `mean_upright` / `time_to_stability_mean` / `com_offset_max` / `fall_rate_heldout` | 0.25 / 0.132093 / null / 0.5467 / 0.1875 | ✗ | **✗** |
| **random_init_policy** | (same set) | 0.75 / 0.172867 / null / 0.6517 / 0.75 | ✗ | **✗** |
| **v5@100k** | (same set) | 0.375 / 0.664781 / null / 0.1612 / 0.375 | ✗ | **✗** |

(`v5@100k` = `data/solo/metrics/balance_t1_monitor_100352_s0.summary.json`, `steps=100352`.)

### 3.4 Beyond-envelope magnitudes 16/20/25 N·s — REPORTED, non-gating

Kept and reported (the T1 verdict is computed on the in-band battery only; the above-envelope aggregate is still produced so behaviour past the envelope stays visible):

| reported, non-gating (24 above-envelope eps) | stand_hold | v5@100k |
|---|---|---|
| `mean_upright` | 0.784997 | 0.639747 |
| `fall_rate` | 0.541667 | 0.541667 |
| `recovery_success_rate` | 0.041667 | 0.0 |
| `max_recoverable_impulse_heldout` | 16.0 | 0.0 |

---

## 4. Why the as-written gate is unsatisfiable on the non-stepping subset

Frozen as-written gate run on the in-band subset (`stand_hold`):

```
FAIL max_recoverable_impulse_heldout >= 16   (measured 0.0)      <- structurally vacuous: no held-out episode in the band
FAIL mean_upright >= 0.95                    (measured 0.913239) <- bar above the reference it cites
FAIL recovery_success_rate >= 0.9            (measured 0.458333) <- per-push bar calibrated to a full-battery cap
```

Three independent blockers, one of them unsatisfiable **by definition**. On the full battery it also fails `fall_rate` (0.291667), `mean_upright` (0.849118) and `fall_rate_heldout` (0.958333).

---

## 5. Discrimination test

`tests/solo/test_t1_gate_discrimination.py` — behavioural: it drives the real `TaskGate.verdict` / `solo.eval.aggregate` over the stored per-episode summaries, computes the held-out metric with the production aggregator (using its `fall_rate`, which carries the correct predicate), derives the ceiling from the live model, and freezes both candidate criteria sets as literals so the defect and the fix stay pinned.

**Result: 17 passed in 2.40 s** (25 passed together with `tests/solo/test_config_plumbing.py`; `.venv/bin/python -m pytest tests/solo/test_t1_gate_discrimination.py -q -s`).

| test | pins |
|---|---|
| `test_training_push_distribution_is_single_height` | training magnitudes/directions ⊆ battery, height fixed 0.95 (reads `solo.curriculum.DEFAULT_CURRICULUM`) |
| `test_heldout_conditions_are_nonempty_and_disjoint_from_training` | held-out set **non-empty (16)** and **disjoint** from the training conditions; 8 eps at each height |
| `test_gate_heldout_metric_is_computed_over_the_heldout_subset` | held-out metric read over the subset; **0.0625** pins the aligned predicate (the pre-fix any-termination value 0.5625 would fail this assert); poisoning every held-out episode ⇒ 1.0 ⇒ criterion **fails** (degenerate set cannot pass) |
| `test_reference_hold_is_measured` | in-band shape {4,8,12}, reference hold **0.913239** |
| `test_as_written_gate_rejects_the_reference_it_was_calibrated_from` | **FAILS AS INTENDED**: as-written rejects stand_hold on full battery *and* in-band; decisive blocker is `mean_upright` (0.95 > 0.913239) |
| `test_proposed_gate_certifies_the_reference_on_the_nonstepping_subset` | **PASSES**; and rejects the reference on the above-envelope row |
| `test_trivial_controllers_fail_the_proposed_gate[zero_action/random_init_policy]` | trivials rejected |
| `test_v5_checkpoint_does_not_pass_the_proposed_gate` | v5@100k rejected |
| `test_t1_has_no_active_recovery_metric_and_the_bars_moved_to_stepping` | T1 metrics `{fall_rate, mean_upright, time_to_stability_mean, com_offset_max, fall_rate_heldout}` disjoint from `ACTIVE_RECOVERY`; as-written **does** intersect it (regression guard); the recovery bars are in the stepping gate |
| `test_static_survivor_is_not_credited_with_recovery` | never-falls/never-moves/perfect-upright controller passes T1 stability but is credited with no recovery and fails the stepping gate |
| `test_uprightness_bar_is_below_the_measured_reference_hold` | `0.84 ≤ 0.913239` **and** `0.84 ≤ 0.849118` |
| `test_collapsed_but_non_terminating_controller_fails_on_uprightness` | frozen collapsed posture (upright 0.30) fails |
| `test_uprightness_bar_is_not_vacuous` | trivials' holds < bar |
| `test_beyond_envelope_magnitudes_are_reported_but_not_gated` | 16/20/25 reported; held-out axis is height (asserted in the note), not magnitude; no `max_recoverable*` in T1 |
| `test_nonstepping_ceiling_is_direction_dependent_not_the_stale_13n` | derived `J_reject` ∈ [6.7, 20.7]; `max ≥ 16` (16 inside laterally); `min < 12`; `min < 13 < max` (the ~13 scalar is not the ceiling) |
| `test_shipped_balance_gate_is_exactly_as_written_or_exactly_proposed` | migration guard |

Deciding numbers printed by the test: reference in-band hold **0.913239**; proposed bar **0.84**; as-written bar **0.95**; held-out set **16 eps**; held-out reference fall rate **0.0625** (vs 0.5625 under the shipped predicate); ceiling **6.69–20.72 N·s**.

---

## 6. PROPOSING to change / leaving alone

**Changing (balance + the held-out definition only):**
- `mean_upright`: `0.95 → 0.84` on the in-band battery. *Why:* 0.95 exceeds the reference's own hold on every battery measured (0.923 old / 0.913239 in-band / 0.849118 full); it is the defect. 0.84 is below the reference under either reading.
- **Scope**: T1 evaluated on magnitudes `≤ TRAIN_MAX_IMPULSE` (4/8/12). *Why:* 20/25 N·s exceed the derived ceiling in every direction and 16 N·s beyond the sagittal directions; they are stepping territory (T3). (Note: 16 N·s is *inside* the lateral envelope — the cap for 16 is justified by the training-distribution match, the cap for 20/25 by physics.)
- **Held-out redefinition**: from above-cap magnitudes to off-training **conditions** (heights {0.79,1.10}). *Why:* capping the battery otherwise leaves the held-out set empty and the criterion vacuous (§2.1).
- **`fall_rate_heldout` predicate**: `termination is not None` → `termination == "fall"`, matching `fall_rate` (§1.3D). Threshold stays `<= 0.10` (now attainable: reference 0.0625).
- **Remove** `max_recoverable_impulse`, `max_recoverable_impulse_heldout`, `recovery_success_rate` from T1 → move to T3.
- **Correct the provenance notes** in `eval.py:311-313`: "StandHold measured 0.0 held-out" is contradicted by the stored 16.0; the "~13 N·s" ceiling here and in `src/solo/pushes.py:18-19` is wrong — replace with the derived direction-dependent range (6.7–20.7 N·s).

**Leaving alone:**
- `fall_rate <= 0.05`, `com_offset_max <= 0.20`, `time_to_stability_mean <= 1.0` — unchanged; reference passes (0.041667 / 0.1236 / 0.1233), trivials fail.
- The **locomotion gate** (`GATES["locomotion"]`, incl. T2Gate's just-added `T2_THRESHOLDS` block) — untouched.
- The **battery magnitudes** and the **training curriculum** — untouched; `train_max_impulse` stays 12.0; solo-t1-v5 keeps training.
- **Not done:** no threshold was tuned to make v5 pass (v5 fails T1 on 4 of 5 criteria); no change to the running job.

---

## 7. Applied diff for `src/solo/eval.py` + `src/solo/pushes.py` — **APPLIED**

T2Gate originally held the file; it reported done and committed its locomotion gate (`377009c` "T2 locomotion gate: held-out commands + behavioural thresholds (S3)"), after which `git status --porcelain src/solo/eval.py` was empty and `git log -1 -- src/solo/eval.py` = `377009c`. The balance-gate diff below was then applied **only** to `GATES["balance"]`, `GATES["stance"]`, `battery_pushes`' `held` line, `aggregate`'s `fall_rate_heldout` predicate, and the new `TRAIN_PUSH_HEIGHT` constant.

**Locomotion gate unchanged — verified by snapshot, not by eye:** `GATES["locomotion"].as_dict()` was JSON-serialised before the edit and compared after; the check printed `locomotion unchanged: True`. The 9 locomotion criteria (thresholds 0.10 / 0.045 / 0.15 / 0.95 / 0.10 / 0.10 / 0.06 / 0.8 / 0.3) and `provisional=False` are byte-identical.

Applied hunks (line numbers are the post-edit file):

```diff
@@ GATES["balance"] (post-edit lines 303-328) @@
         (Criterion("fall_rate", "<=", 0.05,
                    "falls across the IN-BAND battery (magnitudes <= TRAIN_MAX_IMPULSE = 12 N*s, all heights); ..."),
          Criterion("mean_upright", ">=", 0.84,
                    "provenance: below StandHold's in-band hold 0.913239 AND the stored full-battery hold 0.849118; ..."),
          Criterion("time_to_stability_mean", "<=", 1.0, ...),
          Criterion("com_offset_max", "<=", 0.20, ...),
          Criterion("fall_rate_heldout", "<=", 0.10,
                    "true fall rate over the held-out CONDITIONS = off-training application heights "
                    "{0.79,1.10} m inside the in-band battery (training pushes are all at 0.95 m, curriculum.py); ...")),
         note="T1 = NON-STEPPING dynamic balance only, evaluated on battery magnitudes <= "
              "TRAIN_MAX_IMPULSE; held-out = off-training heights ...")
-        # removed from T1: max_recoverable_impulse_heldout, recovery_success_rate
@@ GATES["stance"] (post-edit lines 362-379) @@
         (Criterion("stance_err_mean", "<=", 0.08, ...),
          Criterion("mean_upright", ">=", 0.95),
-         Criterion("fall_rate", "<=", 0.05))),
+         Criterion("fall_rate", "<=", 0.05),
+         # moved out of T1: stepping recovery is T3/T2 capability
+         Criterion("max_recoverable_impulse", ">=", 16.0, "... REQUIRES re-calibration ..."),
+         Criterion("max_recoverable_impulse_heldout", ">=", 16.0, "... REQUIRES re-calibration (v5/T3)"),
+         Criterion("fall_rate_heldout", "<=", 0.10, "held-out magnitudes only (magnitudes > train_max_impulse)"),
+         Criterion("recovery_success_rate", ">=", 0.9, "stabilised to stance after each held-out push")),
+        note="T3 stance/stepping; owns the recovery bars moved from T1"),
@@ battery_pushes (post-edit line 421-427) @@
-            held = float(m) > float(train_max_impulse) + 1e-9
+            # held out = above the training cap OR off the single training height
+            held = (float(m) > float(train_max_impulse) + 1e-9
+                    or abs(float(h) - TRAIN_PUSH_HEIGHT) > 1e-9)
@@ aggregate (post-edit lines 664-667) @@
-        "fall_rate_heldout": round(sum(1 for e in heldout_eps
-                                       if e.get("termination") is not None)
-                                   / max(1, len(heldout_eps)), 6),
+        # same predicate as fall_rate (falls, not any termination)
+        "fall_rate_heldout": round(sum(1 for e in heldout_eps
+                                       if e.get("termination") == "fall")
+                                   / max(1, len(heldout_eps)), 6),
@@ module constant (post-edit lines 399-403) @@
+#: the single application height used by training; battery pushes at other
+#: heights are held out by *condition*, so a capped battery keeps a held-out set.
+TRAIN_PUSH_HEIGHT = 0.95
```

**`src/solo/pushes.py` comment fix (behaviour unchanged — `TRAIN_MAX_IMPULSE` is still 12.0).** The stale "~13 N·s" ceiling appeared in two places in that file; both are now corrected to the derived range (this was a comment/docstring fix only, not a live threshold):

* module docstring: "the analytic non-stepping ceiling for a 33.3 kg G1 is ~13 N·s" → the direction-dependent 6.7–20.7 N·s derivation (m = 33.3411 kg, z_c = 0.6919 m), citing §2.2.
* `TRAIN_MAX_IMPULSE` comment: "12 N*s is just under the ~13 N*s analytic non-stepping ceiling" → "the cap is the top of the training curriculum, NOT a physics bound; 12 N·s already exceeds the ceiling in the weakest (sagittal) directions".

Call-site convention for T1:

```python
from solo.eval import battery_pushes
from solo.pushes import TRAIN_MAX_IMPULSE
t1_plan = [p for p in battery_pushes() if p.impulse <= TRAIN_MAX_IMPULSE]
report = evaluate(factory, task="balance", push_plan=t1_plan, ...)
```

(Existing callers passing the full battery keep working; the full-battery run simply no longer certifies T1.)

**Post-apply evidence.** `GATES["balance"]` = `[(fall_rate,<=,0.05), (mean_upright,>=,0.84), (time_to_stability_mean,<=,1.0), (com_offset_max,<=,0.2), (fall_rate_heldout,<=,0.1)]`; `GATES["stance"]` carries the 4 moved criteria; `battery_pushes()` gives 24 in-band pushes of which **16 are held out, at heights {0.79, 1.10}**, plus 24 above-envelope held-out pushes. `tests/solo/test_t1_gate_discrimination.py` + `tests/solo/test_config_plumbing.py`: **25 passed**.

---

## 8. Unverified / caveats

- **The analytic ceiling derivation** uses the CoM-only capture-point formula; it ignores angular momentum from the push application point (0.79/0.95/1.10 m, i.e. above the CoM at 0.692 m) and transient dynamics. The per-direction **ordering** reproduces the measured recovery curve (§2.2), but the absolute N·s values should be read as ±10–20%. Marked INFERENCE, not measurement.
- **UNVERIFIED:** whether T2Gate's in-flight locomotion edit will further touch `metrics.py`/`env.py` in a way that changes the `metric` allow-list. The proposed diff touches only `GATES`, `battery_pushes` and the `fall_rate_heldout` line, so it should be orthogonal.
- **UNVERIFIED:** for the *specific* v5@100k checkpoint, the training distribution is smaller than the full curriculum (at 100k steps progress ≈ 0.2 ⇒ magnitudes {2,4}, directions {0°,45°}); the T1 held-out split here is defined against the **full** curriculum distribution (all 8 directions, magnitudes 2-12, height 0.95), which is the right basis for a gate applied to a finished policy.
- The v5@100k summary (`balance_t1_monitor_100352_s0.summary.json`) is **untracked** in git; the v5 figures come from that file plus `t1_v2_monitor_100352.json`.
- `tests/solo/test_vec_solo.py` is a **separate, untracked vec-port WIP** (6 failing at import/runtime over `TrainConfig(n_envs=…)` / `SoloTrainer.close` — confirmed unrelated by T2Gate). It is not part of this work; `tests/solo/` with it ignored is 25 passed.
- `run_episode`'s per-episode `recovered` flag (the summary `"recovered"` key) does **not** require `termination is None`, while the aggregate-level `_recovered_episode` does. This dual meaning is pre-existing and unchanged by the proposal, but it is a latent trap.
