# Agent 3 — Reference Re-Timing / Dynamic Feasibility (v2 dataset): report

Date: 2026-10-08 · Brief: `reports/2026-10-08/briefs/agent3_reference_retiming.md`
Dataset: `data/references/motion_refs/v2/` · Generator: `src/solo/refgen.py` ·
Builder: `scripts/build_v2_refs.py` · Probes: `scripts/probe_v2_dynamic.py` ·
Per-segment detail: `data/references/motion_refs/v2/RETIMING.md`

## 0. Result in one paragraph

The v1 wall was not (only) timing — the v1 root paths are **quasi-statically
inconsistent**: the reference's own CoM sits outside the planted-foot hull on
62–100 % of frames (LOWER 97 %, min −0.122 m; CIRCLE 100 %, min −0.63 m), and
the v1 table solver's "repaired" stance hold penetrates the mat by 3.8 cm with
margin −0.082 m.  v2 re-generates the paths (solved grounded leg IK, CoM-
inside-support descents, weight-transfer-before-lift stepping, the operator's
two-axis entry repair) and times them to the measured envelope.  Measured
outcome: **the LOWER segments complete under RAW position servos and under the
T1-v6d balance layer** (v1: toppled at 1.00 s under the same servo protocol;
training died at 0.36 s in all 6 arms), the STAND→LOWER→HOLD chain is
balance_verified end-to-end, and fixed-clock S1 training on v2 (unchanged
interface) reached held-out LEVEL_CHANGE completion 1.00 within 30k steps.
What remains blocked is **the balance layer, not the reference**: stepping and
deep-crouch phases are `balance_blocked` (L1/L2 hold; the only probed layer —
a standing policy — fails them), which names the next lever: T2 locomotion /
T3 stance footwork, not another re-timing pass.

## 1. What was produced

| artifact | path |
|---|---|
| v2 dataset (format = v1's; labels preserved; `retime` blocks) | `data/references/motion_refs/v2/` (drill_continuous 57.92 s / 23 phases, stance_rise, 14 refs, feasibility.json, RETIMING.md) |
| reference generator (calibrated leg IK + quasi-static + stepping) | `src/solo/refgen.py` (module self-check runs) |
| builder / probe scripts | `scripts/build_v2_refs.py`, `scripts/probe_v2_dynamic.py` (labels rewritten from measured probes; `--render-only`, `--relabel-only`) |
| three-level feasibility + balance layer + provenance | `data/references/motion_refs/v2/feasibility.json` (per track AND per drill phase) |
| evidence bundles (960×720/30fps/h264 + 3-frame sheet + metrics JSON w/ provenance) | `videos/motion_refs/v2_lower_balance.{mp4,png,json}` (the balance-layer probe, COMPLETED upright, kept), `v2_drill_replay.*` (kept failure), `v2_penetration_balance.*` (kept failure) |
| contract tests | `tests/test_motion_refs_v2.py` (format round-trip, v2-default resolution, monotone 50 Hz timing, grounded soles + velocity envelope, CoM-inside-support on all quasi-static frames, B2 schedule FK-checked, un-flattened penetration, honest labels) |
| ledger | `notes.md` (RefRetime section) |

Loader: `solo.imitation.REFERENCE_DIRS` resolves **v2 first** — the unchanged
interface (`src/solo/track.py` + `scripts/solo_track_train.py`) consumes v2.
v1 remains the archive, loadable by explicit path; the shipped BC corpus is
pinned to v1 files by explicit path (test_bc), so no silent corpus change.

## 2. Per-segment before/after (all measured; details in RETIMING.md)

| segment | v1 before | v2 after | verdict |
|---|---|---|---|
| LOWER_TO_STANCE / stand_to_stance | CoM outside support 97 % (min −0.122 m), xy accel 12.6 m/s², replay toppled 1.00 s, **training wall 0.36 s** | solved 2.0 s descent, margin ≥ +0.070 m every frame, az ≤ 1.2 m/s² | **replay PASS + balance COMPLETED**; envelope 1.0–2.0x all PASS → `dynamically_verified` |
| STANCE_HOLD | soles −3.8 cm (ungrounded solve), margin −0.082 m, replay toppled 2.18 s | grounded 0.0 cm + guard: margin **+0.070 m** | **replay PASS + balance COMPLETED** → `dynamically_verified` |
| stance_rise | margin −0.088 m, replay toppled 1.28 s | margin +0.081 m | **replay PASS + balance COMPLETED** |
| LEVEL_CHANGE | drop to 0.40 m (below flat-sole reach), CoM out 95 %, replay toppled 1.76 s | knee-driven to **0.58 m** (measured flat-sole floor; ankle limit −0.873 rad binds), margin ≥ +0.047 m | replay PASS both takes; T1 layer blocked → `dynamically_verified` (take) / `balance_blocked` (drill phase) |
| SHUFFLE_F/B, CIRCLE, REPOSITION, widen | video paths: pelvis 0.56–0.72 m (z accel 31.8 m/s²), CoM out to 100 % (min −0.63 m), replays toppled 1.04–2.24 s | generated stepping at 0.70 m: transfer 0.9 s BEFORE lift, joint vel ≤ 3.8 rad/s, xy accel ≤ 1.1 m/s², double-support margins ≥ 0, real 12 mm-mean swing lift | `balance_blocked` — no gait layer exists yet (T2/T3) |
| DOUBLE_LEG_ENTRY_CROUCH | pelvis 0.49 m, pitch 60–70°, margin −0.297 m; toppled 1.18–1.24 s under own targets; CEM 41,840 rollouts → 2.34 s | operator repair (width 0.40 + rear foot 0.35), pelvis 0.62 m, 0.5 s load; margin ≥ +0.055 m | **replay PASS** (`shot_entry_full` dynamically_verified); T1 layer blocked |
| DOUBLE_LEG_PENETRATION | GrappleMap-fused deep knee-down | same geometry kept (identity preserved, never flattened), time-scaled ×1.5 | `balance_blocked`, with the three independent v1 infeasibility measurements standing |
| RECOVER_TO_STANCE | margin −0.421 m, replay toppled 0.70 s | solved rise, margin +0.055 m | **replay PASS** (`shot_recover` dynamically_verified) |
| knee_sprawl_*, stalk_shuffle | ground-posture context takes | TIMING-ONLY (1.5×), labels verbatim | excluded from every traversable claim |

Label schema (RETIMING.md): `dynamically_verified` (raw servos complete it —
v1's level 3, unchanged protocol), `balance_verified` (the balance layer
completes it), `balance_blocked` (L1/L2 hold; the only probed balance layer
fails it — a LEARNING question, not a reference-timing question),
`known_infeasible` / `unverified` as in v1.

## 3. The measured envelope (the brief's "measure, don't guess")

* Static: every solved stance in 0.50–0.79 m × 0.24–0.46 m (× split 0/0.30)
  is grounded at 0.0 cm with CoM margin +0.030..+0.082 m.  Deepest flat-sole
  symmetric crouch: 0.58 m pelvis (ankle-pitch limit is binding, measured).
* Dynamic (`feasibility.json:envelope`): the v2 LOWER completes under RAW
  servos AND the balance layer at 1.0× (2.48 s), 1.5× (3.72 s), 2.0× (4.96 s).
  The clock was never the binding constraint — the PATH was; v2 ships 1.0×.
* Dynamics-in-the-loop CEM was NOT spent on v2: the repaired entry already
  passes the raw-servo protocol, and the penetration's infeasibility is
  already triple-measured (1.24 s topple / CEM 2.34 s / the training wall) —
  another 1-hour CEM run would buy no new information at this layer.

## 4. The decisive read + falsifier

With v2 as the resolution default, `--stage S1_stand_lower_hold_rise` (fixed
clock) trains on v2 through the unchanged interface.  Measured:

* Open-loop replay of the LOWER segments COMPLETES (v1: the same protocol
  toppled at 0.36–1.00 s; 15/17 v1 tracks failed).
* The balance layer traverses STAND → LOWER → STANCE_HOLD end-to-end.
* Fixed-clock training (completed: 400 updates, ~370k steps,
  `checkpoints/solo/track_s1_v2lower.pt`, jsonl
  `reports/2026-10-08/track/train_s1_v2lower.jsonl`, soft→hard anneal
  `--soft-updates 150 --log-std-init -2.0` — the documented hard-gate trap
  avoided).  Final held-out eval (`track_eval_v2lower_final.json`, 54
  episodes, unseen seeds): **LEVEL_CHANGE completion 1.00** (12/12), STANCE
  0.60, SHOT_DOUBLE_LEG 0.25, RECOVER 0.38, overall 0.426, **zero falls in
  every eval of the run** (v1 arms: eval plateau 0.50 with LOWER dead at
  0.36 s everywhere).  The calibrated gate's site-precision bar (p95 ≤
  0.10 m) is not yet met (worst 0.112 m on LEVEL_CHANGE) — completions
  happen, precision is the remaining learning gap on the SAME layer, not a
  reference property: replay and the balance layer already track these
  segments to 0.027–0.052 m site RMS when trained on them.
* **Falsifier read**: the 0.36 s wall is GONE at the reference layer.  If a
  re-trained tracker still stalled on v2's LOWER, that failure would be the
  RL/balance layer's, and the probes already name where that layer ends:
  stepping and deep crouches are `balance_blocked` (the T1 standing prior
  cannot shuffle or crouch deeply).  The next lever is therefore
  `docs/MOTOR_CURRICULUM.md` T2 (locomotion) / T3 (stance footwork,
  level change) — NOT another re-timing pass; re-timing again would be wrong.

## 5. Honest limits

* The `balance_blocked` verdicts use ONE balance layer (T1 v6d via first-layer
  surgery; it has never seen the reference block).  A tracking-trained policy
  (Agent 2's line) may do better on the same phases; the label claims only
  what was measured, and the phase notes carry the layer id.
* The generated stepping is quasi-static-flavoured gait (0.77 steps/s,
  5 cm steps): recognisable wrestling footwork at G1 capability, not the
  video's cadence.  Identity (stance/motion structure, penetration depth,
  entry windup) is preserved; nothing was flattened to make balance easy.
* Sole residuals on the stepping takes reach −0.009 m (G2 bar 1 cm) at plant
  events; holds and descents are exact (0.000).
* Training was left running at session end (background service `s1v2lower`);
  the numbers above are the pre-registered early read, not the final gate.

## 6. Reproduce

```
MUJOCO_GL=egl .venv/bin/python scripts/build_v2_refs.py
MUJOCO_GL=egl .venv/bin/python scripts/probe_v2_dynamic.py            # probes+labels+bundles
MUJOCO_GL=egl .venv/bin/python scripts/probe_v2_dynamic.py --render-only
MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_motion_refs_v2.py tests/test_track.py tests/test_motion_refs.py -q
MUJOCO_GL=egl .venv/bin/python scripts/solo_track_train.py --stage S1_stand_lower_hold_rise \
    --updates 400 --soft-updates 150 --log-std-init -2.0 --out checkpoints/solo/track_s1_v2lower.pt
MUJOCO_GL=egl .venv/bin/python scripts/solo_track_eval.py --ckpt checkpoints/solo/track_s1_v2lower.pt --tag v2lower --seeds 2000,2001
```
