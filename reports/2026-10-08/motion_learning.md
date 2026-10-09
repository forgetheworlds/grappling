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

---

# FOLLOW-UP (2026-10-08, after Agent 3's v2 references) — the decisive re-read

## F.0 Pre-registered protocol (stated before the runs)

- **One lever**: the references (v1 → v2, Agent 3's quasi-static consistency fix). The
  tracking interface (`src/solo/track.py`), observation, reward, gates, protocol and
  seeds are UNCHANGED from the v1 arms; fixed clock; same trainer config as the best v1
  arm (batch 24, soft→hard gate at update 150, T1-surgery init, log-std −2.0→−3.0/80k).
- **Decisive read**: does training progress PAST 0.36 s on LOWER?
- **Falsifier**: if training still fails at the same place on v2, the blocker is the
  learning layer, not the references.
- Second (controlled) arm added at the same time, also pre-registered: the SAME protocol
  **from scratch** (no T1 init) — because on v2 the z=0 base action alone (exact open-loop
  replay) already completes 9/9 S1+S4 segments, so a fresh net (which outputs ≈0 = replay)
  is the natural-convergence control and the T1-init's stand-trained residuals are a
  candidate confounder.

## F.1 Baselines on v2 (same protocol, `data/solo/metrics/track_baselines.json`)

| probe | v1 | v2 |
|---|---|---|
| open-loop replay, LEVEL_CHANGE | 0/6 completed (LOWER died at 0.36 s) | **6/6 completed** (LOWER replays 1.98 s at site RMS 0.020 m) |
| open-loop replay, STANCE | 1/3 | **3/3** |
| open-loop replay, gait/shot/recover (CIRCLE, SHUFFLE_*, SHOT, RECOVER) | 0/13 | 0/13 — **balance_blocked stands** (gait layer = T2) |
| phase separability (obs carries the signal) | 0.854 | 0.697 (chance 0.083; separation 2.6) |
| anti-gaming closure | pelvis-drop terminal + 4.2× | unchanged (3.6× with the current kernels; structural closure primary) |

## F.2 The decisive read: the 0.36 s wall is GONE

- **Open-loop** (no learning at all): LOWER now replays 99/99 frames (1.98 s) at site
  RMS 0.020 m — on v1 it toppled at frame 18 (0.36 s).
- **Trained policy (T1-init arm, the v1 protocol verbatim)**: LOWER passed the hard gate
  from eval@80 onward and the run finished **final eval success 1.00** (both eval
  segments complete; 653,863 steps, `checkpoints/solo/track_s1_v2refs.pt`).
- **Answer: training progresses far past 0.36 s on LOWER — the reference
  consistency fix moved the wall.** The falsifier does NOT trigger.

## F.3 Per-skill held-out, v2 vs v1 (same protocol, hard gate, unseen seeds)

Best v2 checkpoint (`track_s1_v2refs.pt`, 653,863 steps; 6 segments × 4 unseen seeds
2000–2003, IC noise 0.005/0.005/1°) vs the best v1 checkpoint at the same protocol
(`track_eval_s1v2_it150.json`, seeds 2000–2001):

| skill | v1 completion | **v2 completion** | v2 worst site-p95 | v2 worst slide/step | falls (v1 → v2) |
|---|---|---|---|---|---|
| STANCE | 0.33 | **0.92** (11/12) | 0.141 | 0.0093 | 0 → 0 |
| LEVEL_CHANGE | 0.00 | **0.75** (9/12) | 0.143 | 0.0087 | 0 → 0 |
| overall | 0.167 | **0.833** (20/24) | 0.143 | 0.0093 | 0 → 0 |

**Denominator note (orchestrator query)**: the v1 artifact evaluated 6 episodes per
skill (seeds 2000–2001), the v2 artifact 12 (seeds 2000–2003) — the rates above are over
different denominators, and the underlying segment timelines also differ (v2 re-timed:
e.g. LOWER is 100 frames on v2 vs 61 on v1 — that difference IS the lever under test).
**Restricted to the SHARED seeds (2000, 2001) — same n, same seeds**: STANCE **6/6**
(v1 2/6), LEVEL_CHANGE **5/6** (v1 0/6), overall **11/12** (v1 2/12); the single
shared-seed failure is `stand_to_stance` seed 2000 (deviation at step 95/125). The
conclusion is unchanged under either denominator.

Causes: 20× success, 4× deviation (kept: `videos/solo_drill/track/v2refs_stand2stance_fail_fail.mp4`,
deviation at step 95/125 — a LATE, different failure from v1's frame-18 signature).
Per-segment worst-cases: `drill[72:172]` LOWER 4/4 (worst site-p95 0.122);
`stance_hold[0:125]` 4/4 (0.132); `drill[192:292]` hold 4/4 (0.138); `drill[0:50]` 3/4
(0.141); `stance_rise` 3/4 (0.136); `stand_to_stance` 2/4 (0.143) — the failing rows are
late-sequence deviations, not the old early-collapse. The 0.10 m site-p95 quality bar is
not yet met by the worst cases (gate_pass 0 at 0.833 completion) — that is now a tuning
question, not a structural one.

**Scratch control** (pre-registered): from-scratch PPO on v2 hits hard-gate success
**1.00 at eval@20** (site 0.008–0.013 m — at/below the replay baseline) with dev_frac
0.024 at update 26. Two findings: (a) on feasible references the learning layer works
out of the box; (b) the T1-surgery init was actively harmful on v2 (dev_frac 0.374 at
update 30 vs 0.024 scratch — its stand-trained residuals fight the moving reference
base). For all later stages: init from scratch or from a prior TRACKING checkpoint, not
from the stand-balance policy.

## F.4 Honest frontier after the v2 fix

- **Passing (hard gate, physics, no resets inside an episode)**: stand, lower-to-stance,
  stance hold (2 s take), stand↔stance takes — 0.833 held-out completion, 0 falls.
- **Still failing**: gait phases (SHUFFLE_*, CIRCLE), shot entry/penetration, recovery —
  open-loop replay fails them and they are labelled `balance_blocked`/`known_infeasible`
  in v2; they need the T2 gait layer and are NOT papered over with reward shaping.
- **Quality bar**: site-p95 ≤ 0.10 m gate not yet met at the worst cases (0.12–0.14 m);
  next lever is plain training-time/scale on the SAME fixed protocol (no design change).

## F.5 Commands

```bash
# baselines on v2 (regenerated artifact)
MUJOCO_GL=egl .venv/bin/python scripts/track_baselines.py
# the pre-registered v2 run (T1-init arm; final eval 1.00)
MUJOCO_GL=egl .venv/bin/python scripts/solo_track_train.py \
  --stage S1_stand_lower_hold_rise --updates 300 --episodes-per-update 24 \
  --soft-updates 150 --soft-penalty 0.5 --completion-bonus 10 --terminal-penalty 30 \
  --log-std-init -2.0 --log-std-final -3.0 --log-std-anneal-steps 80000 \
  --clock fixed --init-ckpt checkpoints/solo/t1_balance_v6d.pt \
  --out checkpoints/solo/track_s1_v2refs.pt
# the scratch control (eval@20 = 1.00)
MUJOCO_GL=egl .venv/bin/python scripts/solo_track_train.py \
  --stage S1_stand_lower_hold_rise --updates 300 --episodes-per-update 24 \
  --soft-updates 150 --soft-penalty 0.5 --completion-bonus 10 --terminal-penalty 30 \
  --log-std-init -2.0 --log-std-final -3.0 --log-std-anneal-steps 80000 \
  --clock fixed --out checkpoints/solo/track_s1_v2refs_scratch.pt
# per-skill held-out (the §F.3 numbers)
MUJOCO_GL=egl .venv/bin/python scripts/solo_track_eval.py \
  --ckpt checkpoints/solo/track_s1_v2refs.pt --tag v2refs_s1 --seeds 2000,2001,2002,2003 \
  --stages S1_stand_lower_hold_rise
# evidence
MUJOCO_GL=egl .venv/bin/python scripts/solo_track_render.py episode \
  --ckpt checkpoints/solo/track_s1_v2refs.pt --source drill_continuous --k0 72 --k1 172 \
  --label LEVEL_CHANGE --name v2refs_lower_pass
```

---

# STAGE 2-3 (GAIT) READ — authorised follow-up on v2 (pre-registered in the ledger)

## G.0 Pre-registration (verbatim before the runs)

- LEVER: the curriculum stage (S2_first_step on v2), unchanged interface/observation/
  reward/gates, init FROM SCRATCH (T1-init measured harmful on a moving base).
- Baseline: open-loop replay completes **0/13** gait/shot/recover segments on v2 —
  unlike S1, replay does NOT solve gait; the policy must add single-support balance.
- DECISIVE READ: gait segments past the 0/13 baseline — hard-gated SHUFFLE_F completion
  with REAL steps, 0 falls, site-p95 <= 0.10.
- FALSIFIER: failure at the same early-deviation signature ⇒ the blocker is the
  learning/observation design, to be answered with a measurement, not reward reshaping.

## G.1 Result: the falsifier TRIGGERS — the gait blocker is in the learning/observation layer

300 updates / **613,475 steps** (`checkpoints/solo/track_s2_v2refs.pt`): soft-phase
completions reached 0.94, but the **final hard-gated eval is 0.00** and per-skill held-out
(8 episodes, 4 unseen seeds) is **SHUFFLE_F 0.25 completion** (2/8), 0 falls, worst
site-p95 0.136, worst joint err 0.022 rad. Training DID progress — the deterministic
deviation onset moved from step 44 (it50) to step 108 (final), i.e. the policy learned to
clear the first two reference transfers — but every measurement points at the same
mechanism:

**Measurement (`scripts/track_gait_probe.py`, committed):**
- Both failures concentrate AT reference contact switches: onset step 44 with the first
  reference switch at 46 (+/-3 frames => True); final-checkpoint onset step 108 with the
  switch at 112 — the policy rides each 0.4-0.9 s transfer window and breaks at the next.
- At onset the JOINT tracking is essentially perfect (0.012-0.022 rad): the swing command
  is observed and followed; what diverges is the whole-body transfer (site RMS rises
  linearly 0.044 -> 0.102 over the last 0.2 s; root_xy 0.099-0.101 m).
- Robot-vs-reference contact agreement until onset: 0.93; robot step events present (8).

**The design fact that closes the argument**: in the shipped interface the ACTOR IS
CONTACT-BLIND — `solo/obs.py` keeps every contact (including the robot's own foot
contacts) in the critic-only privileged block (tested). During the 0.6 s transfer windows
the policy must weight and unweight feet, but it cannot sense which foot is loaded NOW;
the REF block tells it where the reference wants its feet and root, yet the policy has no
direct read of its own support state to correct against. Gait is the first skill in the
curriculum where that gap is load-bearing (S1's skills are double-support; their
deviation gates passed without it).

## G.2 Frontier statement (honest)

- **Passing** (hard gate, physics): stand, lower-to-stance, stance hold, stand<->stance
  (S1 on v2: 0.833 held-out completion, 0 falls).
- **Not passing**: gait (SHUFFLE_F held-out 0.25, 0 falls — completions exist but 6/8
  episodes still die at transfers), circle/shot/recover untouched at this stage
  (balance_blocked / known_infeasible as labelled).
- **Named fix, untried (requires authorisation — it is an interface change)**: add the
  robot's own foot-contact/load state (the per-foot GRF is already computed every step in
  `RewardInputs.foot_load`) and the reference's NEXT contact command (`tt.contact[k+AHEAD]`)
  to the ACTOR block for gait stages; one lever, pre-registered, same protocol. The
  SOLO_DRILL contract keeps contacts privileged, but the motor-curriculum rule "no frozen
  observation interface before evidence" is now met: the evidence is in §G.1.
- NOT done, deliberately: no reward reshaping, no reference reshaping, no gate loosening.

## G.3 Evidence and commands

- kept failure: `videos/solo_drill/track/v2refs_s2_shuffle_fail_fail.mp4` (+sheet+JSON;
  deviation at step 118/200, no fall)
- held-out artifact: `data/solo/metrics/track_eval_v2refs_s2.json`
- probe: `scripts/track_gait_probe.py` (its own printed verification)
- run log: `reports/2026-10-08/track/train_s2_v2refs.log` + `checkpoints/solo/track_s2_v2refs.jsonl`

```bash
MUJOCO_GL=egl .venv/bin/python scripts/solo_track_train.py \
  --stage S2_first_step --updates 300 --episodes-per-update 16 \
  --soft-updates 150 --soft-penalty 0.5 --completion-bonus 10 --terminal-penalty 30 \
  --log-std-init -2.0 --log-std-final -3.0 --log-std-anneal-steps 80000 \
  --clock fixed --out checkpoints/solo/track_s2_v2refs.pt
MUJOCO_GL=egl .venv/bin/python scripts/track_gait_probe.py \
  --ckpt checkpoints/solo/track_s2_v2refs.pt --k0 312 --k1 512 --seed 100
MUJOCO_GL=egl .venv/bin/python scripts/solo_track_eval.py \
  --ckpt checkpoints/solo/track_s2_v2refs.pt --tag v2refs_s2 \
  --seeds 2000,2001,2002,2003 --stages S2_first_step
```
