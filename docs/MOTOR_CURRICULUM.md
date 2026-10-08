# MOTOR_CURRICULUM.md — Foundational motor competence before adversarial wrestling

Additive extension of `docs/MISSION.md` (which stays verbatim and authoritative on intent).
Operator directive (2026-10-08): the wrestling objective is unchanged, but its prerequisites
must be established with **evidence** first. This document defines the progression, the
per-behavior task/reward specification, the advancement gates, the architecture options, and
the verification protocol. It does not replace MISSION's methodology chain — it precedes and
feeds it.

## 0. The gap this addresses

MISSION's chain (GrappleMap → references → imitation → resistance → free wrestling) assumes a
controller that can *hold and move a standing humanoid*. Measured evidence so far says it
cannot yet:

- references track in joint space (≤0.06 rad) but the robot **topples** under pure position
  servos — pelvis drops 0.29–0.83 m (Phase 2, `reports/2026-10-07/retarget.md`);
- even the static STANCE crouch is **not open-loop holdable** (CoM behind the support polygon;
  ankle torque 39/50 Nm) — EnvBuilder, `reports/2026-10-08/wrestling_env.md`;
- 30k steps of PPO with no imitation yields **83% ground time, 13% standing** (E1 in
  `notes.md`) — and "won" 48/51 exchanges only because the scripted opponent collapsed.

Conclusion: the missing layer is *motor competence*, not wrestling knowledge. Reward signals
for it must be task-appropriate (velocity tracking, uprightness, disturbance recovery), never
the sparse exchange outcome — which is what the current curriculum uses to advance.

## 1. Progression (dependency-ordered)

| # | layer | question it answers | prerequisite for |
|---|---|---|---|
| **M1** | Dynamic balance | can it stay up and recover from pushes? | everything |
| **M2** | General locomotion | can it walk/turn/stop at commanded velocities? | M3–M5 |
| **M3** | Whole-body coordination | can arms/torso move without losing M1–M2? | M5 |
| **M4** | Wrestling locomotion | stance, shuffle, circle, level change, knee drops, ground recovery | M5 |
| **M5** | Wrestling techniques | GrappleMap techniques vs resistance; selection; self-play | final match |

Not five independent policies — see §4 (architecture). M1–M2 are the smallest set that makes
M5 reachable; M3–M4 may collapse into one command-conditioned policy with M2.

## 2. Per-behavior task specification

Each row is a *task contract*: no task may be trained before this row is filled and its
success metric has a harness that can measure it on held-out conditions.

| field | M1 dynamic balance | M2 locomotion | M3 coordination | M4 wrestling locomotion | M5 techniques |
|---|---|---|---|---|---|
| **Initial state** | standing (keyframe `stand`), plus randomized perturbation states (pushes, root tilt/velocity, foot offsets) | standing, stopped, mid-stride, randomized heading | M1–M2 states + random arm configurations | stance start + shuffled/circled states + level-changed states | M4 states + opponent-relative start distributions per technique (distance/angle/stance randomized) |
| **Command** | none (target: stay up); later: "recover from disturbance D" | (vx, vy, ωz) target vector, sampled from feasible set | M2 command + end-effector target(s) | M2 command in stance + level target + angle target + opponent-relative objective | technique one-hot + phase; later: intent (skill head) only |
| **Observation** | proprioception (qpos/qvel), gravity-aligned root orientation, feet contact, CoM vs support | + command vector; same proprioceptive base | + torso/arm sites, target vectors (self-relative) | + opponent-relative pelvis/torso/support (mirrors existing env obs) | existing 92-dim actor obs (frozen only after §4 decision) |
| **Action** | 29 joint position targets (existing actuator interface) | same | same | same | same (absolute or residual-on-reference) |
| **Termination** | fall (pelvis below / tilt beyond, persistence-confirmed); no time limit for balance | fall; command timeout | fall; target timeout | fall to non-stance ground; explicit nonterminal states continue (knees/hands are NOT terminal, per MISSION) | MISSION back-to-mat rule (exchange end) |
| **Rewards (physically justified)** | upright (gravity-aligned torso), CoM-in-support, low joint velocity when idle (energy), penalty for actuator saturation; recovery reward = return to stance after push | velocity-tracking (task) **minus** uprightness gate, foot-slip penalty, action-rate/torque penalty; track progress toward command | M2 terms + target-distance term weighted to *not* dominate balance terms | M2 terms in stance + torso-facing/angle error + level-change success (knees reach commanded height without fall) | MISSION schedule: technique similarity (scorer) → progress → outcome, similarity weight decaying with resistance |
| **Randomization** | push magnitude/direction/height, ground friction, initial pose noise, latency | friction, mass/com, motor strength, command distributions, push during walk | arm mass payload, target reachability | opponent lateral pressure (scripted), friction, stance width | full MISSION set (distance/angle/stance/timing) |
| **Likely exploits** | standing perfectly still / rigid pose (defeated by held-out push battery) | leaning forward and "falling with style" at the requested speed (defeated by uprightness gate + step-quality metrics) | using arms as crutches / dropping to knees to reach targets | shuffling without moving the body (defeated by displacement + orientation tracking) | shoulder-first charge, ground-hugging (defeated by scorer + video review) |
| **Independent metrics** | fall rate under held-out push battery; CoM margin; recovery time; idle energy | velocity-tracking error (mean/p95), uprightness, foot slip, fall rate, command-response latency | target error + balance retention (no fall) | stance stability under lateral pressure; circle radius error; level-change depth accuracy; ground→stand success rate | MISSION evaluation axes (attacks by type, takedown success, failed-shot recovery, technique diversity) + scorer |
| **Advancement gate** | ≥90% recovery on held-out pushes with magnitude unseen in training; zero falls across a 60 s idle hold | velocity error below threshold across the feasible command set, fall rate <1 per 100 s, upright >99%, on HELD-OUT commands | target error below threshold with <1% fall rate while combined with M2 commands | stance survives scripted lateral pressure profile; continuous circling without fall; N successful ground→stand recoveries | recognizable execution (scorer ≥ gate) AND success attributable to the attempted technique (not opponent collapse) |

**Thresholds are placeholders** — each is set experimentally from a baseline-first measurement
(see §5) and recorded with the config that produced it. No invented authoritative numbers.

## 3. Advancement and anti-self-deception rules

1. Gates are **per-capability metrics**, never exchange/outcome rate.
2. **Never promote because the opponent collapses.** The E1 baseline is the cautionary case:
   48/51 "wins" against a scripted collapsing opponent, 13% standing time.
3. Every promotion requires: held-out condition results, fixed seeds, a video, and a recorded
   config. Missing any one → the stage stays open.
4. Evaluation metrics ≠ learning rewards. Velocity tracking alone must never be the success
   metric (it rewards falling forward at speed); every tracking metric is paired with an
   uprightness/step-quality metric.
5. Promotion is reversible: a later stage that degrades an earlier gate re-opens it.

## 4. Architecture: smallest thing that works

Candidates (MISSION's options 1–4 generalise here):

| option | shape | when justified |
|---|---|---|
| **A. One shared command-conditioned motor policy** | obs(+command) → 29 joint targets; commands include velocity, stance, level, technique | default first attempt — smallest, one interface, no hand-off seams |
| **B. Backbone + specialists** | shared trunk, per-skill heads (locomotion / stance / technique) | only if A demonstrably fails to serve ≥2 task families without interference |
| **C. Hierarchical** | high-level selector issues skill/velocity commands to a frozen low-level motor policy | only if the wrestling policy cannot learn tactics while also holding balance (measure: tactical metrics plateau while balance metrics are fine) |

Decision protocol: build **A**; measure interference (M1/M2 gates re-tested after M3–M5
training — this *is* the catastrophic-forgetting test); escalate to B or C only with that
evidence. Low-level preservation mechanisms (replay of earlier tasks, regularisation, frozen
sub-networks) are added only when the measured forgetting exceeds the gate tolerance.

Frozen interfaces: none yet. The observation/action interface is explicitly **provisional**
until the free-wrestling ablation (MISSION "end-stage policy design") is run with evidence.

## 5. Verification protocol (per milestone)

Required for every milestone, no exceptions:

1. Runnable train + eval commands reproducible from repo root.
2. Unit/integration tests (fast, no training).
3. Quantitative evaluation: fixed seeds, **held-out** conditions (e.g. push magnitudes unseen
   in training), reported as a table with the exact config path.
4. Videos: `videos/motor/<stage>/{success,failure}/*.mp4` — success behaviour AND failures.
5. Recorded configuration + results file under `data/motor/<stage>/`.
6. Explicit PROCEED / REVISE / STOP criteria written *before* the run.
7. Baseline-first: measure the untrained/naive baseline with the same harness before any long
   training run ("avoid long training runs before validating env, reward, controller, eval").

## 6. Implementation sequence (justified order)

| step | action | why this order |
|---|---|---|
| 0 | **Audit** implemented-vs-demonstrated motor capability (`reports/2026-10-08/motor_audit.md`) | avoid building what exists; find reusable primitives |
| 1 | **Harness for M1** (env task, obs/action reuse, push battery, metrics, videos) + baseline measurement | the largest uncertainty is whether our sim/actuator setup can even express a balance task; harness before learning |
| 2 | **M1 dynamic balance** trained + gated | prerequisite for every other layer |
| 3 | **M2 locomotion** + forgetting check on M1 | unlocks range management (MISSION's core wrestling need) |
| 4 | **M3** (arms) and **M4** (wrestling locomotion) — possibly joint with M2 under option A | M5 needs stance/level/angle control |
| 5 | **M5** techniques: integrate GrappleMap references + teacher labels + scorer + resistance stages | MISSION Phase 2–5, now standing on verified motor competence |
| 6 | Re-run MISSION's ablations (imitation value; technique-prior value; action representation) with the motor backbone in place | final architecture decision needs these |

## 7. Relationship to existing work (nothing is discarded)

- The retargeted references, scorer, env/rules, detector, and RL infrastructure are **reused**
  in M1–M5: the RL infra already provides obs/privileged split, PPO, curriculum config,
  checkpoint/resume, vectorisation, and deterministic eval; the env provides reset
  randomisation, exchange loop, and videos.
- The **teacher** (in flight) remains the algorithmic demonstrator for M4/M5 execution and the
  label generator for imitation; it is not a substitute for learned balance, and learned balance
  is not a substitute for the teacher.
- The current outcome-based advancement logic is superseded by §3 for M1–M4; wrestling stages
  (M5) keep MISSION's rules with the added "success attributable to the technique" requirement.

## 8. Diagnostic video suite (adopted from external engineering audit, 2026-10-08)

Motion evidence alone is not proof; these diagnose *why* a stage succeeds or fails. Each uses
recorded seeds/configs, renders a paired baseline-vs-candidate comparison from identical
scenarios, and overlays numbers (no decorative footage).

| # | video | purpose | stage |
|---|---|---|---|
| V01 | balance A/B: PD-only vs stabilized controller under identical pushes | does the controller actually extend standing time / improve recovery? (overlay pelvis height, torso tilt, CoM, foot contacts, saturation) | M1 gate evidence |
| V02 | ghost reference vs physics robot | where exactly does execution diverge from intent? (overlay site/joint error, penetration, contact forces) | M4/M5 |
| V03 | reward vs behaviour: stationary / collapsing / attempting agent, live reward components | find rewards that look good while the behaviour is absent (exploit audit) | M1–M5, every reward change |
| V04 | contact & clinch validity: collision geoms + force vectors visible | does the arm actually transfer force, or just sit near the limb? does contact slip? | M4/M5 prerequisite |
| V05 | locomotion & coordination: changing commands + arm reach while moving (target vs actual, slip, falls) | velocity tracking without falling; arm use without balance loss | M2/M3 gate evidence |
| V06 | genuine exchange: same attack vs stationary / moving / competent defender, failures included | skill proof requires a resisting opponent, not a collapsing one | M5, abandons-policy check |

Prerequisites these videos presuppose (must be validated during M1/M2 harness work):
contact-model adequacy (foot friction/slip, force transfer through the hand/arm chain),
actuator saturation accounting, and penetration/contact reporting that is trustworthy enough
to serve as evidence.

## 9. Documentation duties

- This file is the working curriculum for foundational motor learning; `docs/CURRICULUM.md`
  continues to own the wrestling technique curriculum.
- `goal.md` phase map gains the M-phases as an addendum (orchestrator-owned).
- `docs/MISSION.md` remains the verbatim operator brief; any future change must be additive.
- Every stage's results, configs, and gate decisions are logged in `notes.md` (Experiment
  ledger) with the experiment format already in use.
