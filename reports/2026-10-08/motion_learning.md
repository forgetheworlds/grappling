# Agent 2 — Physics-Based Motion Learning (G1 wrestling drill): report

Date: 2026-10-08. Author: MotionLearn (Agent 2). Status: **final for this session**;
every number is recomputable from the named artifact; UNVERIFIED items are named.

## 0. Result in one paragraph

I built and validated the missing learning layer — a **reference-conditioned tracking
environment** (`src/solo/track.py`) in which a PPO policy executes Agent 1's v1 motion
references in MuJoCo physics with **no scripted teacher anywhere in the loop** (the
reference enters purely as a commanded movement target: an appended 55-dim observation
block + the residual base action). The measured acceptance blockers were: the dead phase
channel (**fixed and proven**: phase classification 0.854 vs 0.083 chance), the
anti-gaming crouch escape (**closed structurally**: pelvis-drop termination + 4.2× reward
dominance), and absent baselines (**measured**: open-loop replay completes 1/22 dynamic
rollouts — the balance layer must be learned). Training reached **real, visually verified
physical imitation of the hardest static skill** (deep wrestling stance hold, side-by-side
frames near-identical to the reference) and soft-gated survival of 0.71–0.83 across S1,
but the **dynamic lower/rise segments were not cracked at the hard gate within this
session's compute** (~700k steps, 6 formulation arms): the frame-exact failure mechanism
was isolated to reference **timing infeasibility** (feet planted, joints tracked at
0.04 rad, yet the root drifts 0.156 m in 0.36 s — the retargeted trajectory violates
dynamics even where poses are reachable), and the fix — a **gated reference clock**
(progress earned by tracking) — is implemented and unit-tested but its training result is
**UNVERIFIED** (launched at the session's end).

## 1. What was produced (paths)

| artifact | path |
|---|---|
| tracking env + curriculum + gated clock | `src/solo/track.py` (module self-check runs) |
| plumbing tests (14) | `tests/test_track.py` |
| trainer (stages, soft→hard gate, surgery warm start, pushes) | `scripts/solo_track_train.py` |
| per-skill held-out evaluation (calibrated gates) | `scripts/solo_track_eval.py` |
| evidence renderer (episode / final / side-by-side) | `scripts/solo_track_render.py` |
| baselines + plumbing validation | `scripts/track_baselines.py`, `data/solo/metrics/track_baselines.json` |
| interim held-out evals (honest) | `data/solo/metrics/track_eval_s1v2_it50_probe.json`, `track_eval_s1v2_it150.json` |
| checkpoints | `checkpoints/solo/track_s1*.pt` (+ `.itNN` snapshots; jsonl logs beside them) |
| videos: passing clip + kept failures | `videos/solo_drill/track/s1_it50_stance.mp4`, `s1_it50_lower_fail_fail.mp4` (+ sheets + metrics JSON) |
| side-by-side reference-vs-policy | `videos/solo_drill/track/` (compare PNGs; see §6) |
| training logs | `reports/2026-10-08/track/train_s1*.log` + `checkpoints/solo/track_s1*.jsonl` |
| this report | `reports/2026-10-08/motion_learning.md` |

## 2. Formulation (method contract items 1–3, adopted small)

Inspected TWIST + BeyondMimic/Unitree-RL-mjlab; adopted the smallest defensible subset:

- **Action**: residual `ctrl = clip(q_ref[k+1] + 0.5·tanh(z))` — the residual base IS the
  reference's next-frame joint target, so `z = 0` is exact open-loop replay (the measured
  baseline) and the policy learns only the corrections. This is the repo's own residual
  contract; BeyondMimic-style pose-target servoing.
- **Actor observation** (the reference conditioning): base 115-dim (unchanged, contract-
  stable) + appended block: local root target (pelvis frame), reference root velocity,
  reference joint targets, reference foot contacts, phase progress, per-frame skill
  one-hot (read from the drill's own `phase_id` labels), lead leg, connector flag, 0.2 s
  future root target. Critic adds the 3 live tracking errors.
- **Reward** (prioritised per brief): movement accomplishment = landmarks (site RMS, w 1.0)
  + anchored root (w 0.6, σ_xy 0.08/σ_z 0.04) + yaw (0.2) + root-velocity (0.2) +
  joint-velocity (0.1) timing terms, gated by uprightness; wrestling appearance = joint
  prior only (w 0.2, explicitly soft); validity penalties = action rate, torque sat,
  joint limits, loaded-foot slide.
- **Termination**: dorsal/fall (existing detectors, always) + tracking deviation
  (joint 0.60 rad mean, anchored root 0.35 m, **pelvis drop 0.20 m**, site RMS 0.15 m).
- **Gated clock** (v6): the reference advances only while tracking holds the band
  (0.30 rad / 0.12 m); anti-freeze bound 2× duration without completion credit.

**Inference is teacher-free**: the policy consumes the reference track + clock as a
command; no state-feedback computation by any scripted component.

## 3. Acceptance items, point by point

### 3.1 "Reference conditioning demonstrably present in the actor's observations" — DONE
Motion-only nearest-centroid phase classification over 23 drill phases:
**0.854 accuracy vs 0.083 chance**; per-frame skill/lead/connector switching pinned by
`test_drill_phase_conditioning_is_per_frame`. The BC corpus's dead `obs[114]` is
superseded (the base layout is untouched; the block is appended and documented).

### 3.2 "Anti-gaming hole closed" — DONE (structurally + arithmetically)
- `pelvis_drop > 0.20 m below a standing reference` TERMINATES — sitting is the same
  episode-ending event as falling, so the measured "never fall by never standing" optimum
  (crouch 0.809 vs stance 1.965 with fall cost 1500) is gone.
- Reward arithmetic: standing-reference tracking 2.30/step vs the measured crouch 0.54/step
  (4.2×) — pinned in tests. Deep phases are graded with a widened height kernel so the
  G1-unreachable entry crouch is not punished into a fall.
- Freezing/phase-farming: with the gated clock, reference progress must be earned by
  tracking; with the fixed clock, freezing loses tracking reward and cannot complete a
  dynamic segment (open-loop = the freezing limit — measured to die).

### 3.3 "Real movement in rendered physics" — PARTIAL (honest)
Real MuJoCo rollouts render the policy executing the **STANCE skills** (stand, hold the
deep 0.345 m-wide × 0.74 m-high stance take: site RMS 0.027–0.052 m, joint 0.03–0.05 rad,
contacts matching the reference, upright 0.997, loaded feet, no fall) — verified frame-by-
frame against the reference robot in `*_compare.png`. **Dynamic skills (lower/rise,
shuffles) are NOT yet executed end-to-end at the hard gate** — see §4.

### 3.4 "Skill-by-skill held-out numbers with worst-case" — DONE for S1, honest
`scripts/solo_track_eval.py` (gates frozen from baselines: completion + site-p95 ≤ 0.10 m
+ joint ≤ 0.35 rad + slide ≤ 0.004 m/step). Best-session checkpoint (v2.it150, 144k steps,
hard gate, unseen seeds, 6 segments × 2 seeds):

| skill | n | completion | worst site-p95 | worst joint err | worst slide/step | causes |
|---|---|---|---|---|---|---|
| STANCE | 6 | 0.33 | 0.208 | 0.048 | 0.0117 | 4× deviation, 2× success |
| LEVEL_CHANGE | 6 | 0.00 | 0.342 | 0.048 | 0.0117 | 6× deviation |

(v5/v6 snapshots improve the STANCE rows — v5 eval@20: STANCE success with site 0.027 —
but the LEVEL_CHANGE rows above are the honest session-level statement; the held-out
zero-shot takes `stalk_shuffle` / `knee_sprawl_entry2` were never trained and were not
evaluated because no checkpoint passes their trained siblings yet.)

### 3.5 "No kinematic data mislabelled as a policy" — DONE
All videos/JSONs carry a provenance block; every rendered clip is a policy-driven MuJoCo
rollout; the reference appears only as labelled side-by-side frames ("left = reference
kinematic target (NOT executed motion)").

### 3.6 "Every claim backed by a recomputable artifact" — DONE
See §1 paths + §9 commands. The metrics JSONs embed git commit, ckpt sha256, seeds,
control rate, physics dt, and the reproduce command.

## 4. What failed and the next limiting mechanism (the core engineering finding)

**Failure mechanism, isolated frame-exactly** (probe transcript + `train_s1*.jsonl`):
on the dynamic LOWER segment the policy tracks joints at 0.035–0.048 rad with feet
planted and contact-agreement 0.73–1.00, while the **root drifts 0.156 m fore-aft in
0.36 s** and the upper body diverges (head/shoulders ~0.25 m) — termination at 0.36 s,
the SAME step where open-loop replay dies. The reference is kinematically valid but
**dynamically infeasible in its timing** (E12: 6.9 % of reference frames statically
holdable): no frame-locked tracker can both match it and balance. This is a property of
the v1 references (video retargets), not of the observation design (conditioning is
proven) and not of reward visibility (the drift is punished as it starts).

**Fix implemented, interim result verified (no breakthrough yet)**: the gated reference
clock — reference progress is EARNED by staying inside the tracking band (the
TWIST/BeyondMimic adaptive-pacing idea, and the same principle that made the repo's
quasi-static controller work: state-gated progression). It converts "track an impossible
moving target" into "hold, recover, and advance when ready" while keeping every
anti-gaming property (freezing earns nothing and is bounded; falls/deviations terminate).
v6 (warm-started from v5) ran through update ~44 before the session boundary: the clock
mechanism works (episodes now run longer than the reference duration; `clock_budget`
terminations appear as designed) but the LOWER segment **still fails at the same frame**
— with the clock held, the reference's next **pose transition** itself destabilizes the
robot past the site-deviation bar. Conclusion of the diagnosis chain: the v1 lower/rise
references are not quasi-statically executable by the G1 under position servos, frame
offset or not. Next levers in order (named, untried): (a) offline dynamic RE-TIMING +
RE-SHAPING of the reference trajectory (fit a dynamically consistent path once —
effectively baking the operator's sanctioned geometry repairs into the reference), (b)
reward the root PATH (potential on path progress) instead of path+t timing, (c) learn
these segments from the repo's dynamics-in-the-loop CEM solutions as reference targets.

Also tried and rejected (kept in the ledger): T1-v6d surgery warm start alone (its
corrections are learned around a stand base; no transfer to a moving base), hard
deviation gates from update 0 (no gradient — episodes die at 0.6 s), large exploration σ
(0.16 rad × 29 joints deviates a perfect mean within 30 steps), batch 12 episodes (KL
saturated; 24 fixed it), σ_root_xy 0.06 (over-tight: punished the balance-mandated
modifications; 0.08 final).

## 5. Visual evidence (per EVIDENCE_PROTOCOL; orchestrator indexes in VISUALS.md)

| artifact | what to look for | verdict |
|---|---|---|
| `videos/solo_drill/track/s1_it50_stance.mp4` + sheet + JSON | STANCE window executed in physics: HUD shows site/joint/root errors, contacts vs reference (1/1), upright 0.997, phase 25/50; reset_count 0, fall_count 0 | PASS (short segment) |
| `videos/solo_drill/track/s1_it50_lower_fail_fail.mp4` | the dynamic failure kept: divergence at 0.36 s | KEPT FAILURE |
| `*_compare.png` (side-by-side renderer, `videos/solo_drill/track/` + `/tmp` test grid) | left = reference kinematic target (NOT executed motion), right = learned policy in physics: the deep stance holds match closely for ~2 s; divergence at the reference's transitions | REAL IMITATION of the hold; transitions fail |
| `videos/solo_drill/track/` metrics JSONs | provenance + verdict lines incl. reset_count/fall_count | — |

No continuous final drill video is claimed: **the criteria were not met** (dynamic
segments fail the hard gate), so per the brief the acceptance artifact is NOT produced.

## 6. UNVERIFIED / still failing (explicit)

1. **Dynamic skills end-to-end at the hard gate** (lower, rise, shuffles, circle,
   entry/recovery, connected drill): FAILING at session end; mechanism isolated to
   reference dynamic infeasibility; the gated-clock fix works mechanically but v6's
   interim (update ~44) shows the pose transitions themselves remain infeasible — the
   next levers (§4) are named, untried.
2. **Gated-clock full training outcome**: UNVERIFIED beyond update ~44 (mechanism
   verified working; run cut by the session boundary; checkpoint + log preserved).
3. **Final continuous no-reset drill video**: NOT PRODUCED (correctly withheld).
4. **Zero-shot held-out takes**: not evaluated against a passing checkpoint (moot until
   their trained siblings pass).
5. **S2–S7 stages**: curriculum implemented and unit-checked, not trained.

## 7. What a reviewer should re-run

```bash
MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_track.py -q          # 14 passed
MUJOCO_GL=egl .venv/bin/python -m solo.track                             # module self-check
MUJOCO_GL=egl .venv/bin/python scripts/track_baselines.py                # baselines + arithmetic
MUJOCO_GL=egl .venv/bin/python scripts/solo_track_eval.py \
    --ckpt checkpoints/solo/track_s1_v5.pt --tag v5_check --seeds 2000,2001
MUJOCO_GL=egl .venv/bin/python scripts/solo_track_render.py episode \
    --ckpt checkpoints/solo/track_s1_v5.pt --source stance_hold --seed 42
```

## 8. Commit map

`b6007ee` tracking env + tests + baselines → `d6b79a5` training/eval/render tooling →
`3ac4cb1` per-frame conditioning + velocity terms + yaw-jitter fix → `3c8d3fb` tightened
root kernels + mid-starts → (this commit) gated clock + loader fallback + report/notes.
