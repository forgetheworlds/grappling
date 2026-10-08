# SOLO_DRILL.md — Single-G1 continuous wrestling drill (session milestone)

Scope: ONE G1, solo, learned feedback control in MuJoCo. No second active robot.
Acceptance artifact: `videos/solo_drill/final_continuous_drill.mp4` (continuous, unedited,
60–90 s if reliable). Technique priority: **double-leg penetration step** (secondary: single-leg
entry only if the primary is reliable). Out of scope: resistance, opponent interaction,
self-play, body locks, snapdowns, tactics.

This milestone *is* the motor curriculum (M1–M5) with a concrete demo attached: T1..T7 map to
M1..M5 as shown below. It builds on — does not duplicate — `docs/MOTOR_CURRICULUM.md`.

## 1. Starting point (from the M0 audit, `notes.md` E3 — do not re-derive)

| layer | state |
|---|---|
| Balance: static stand | scripted hold only (5 s keyframe, 0.16 cm drift; 20 s StandHold draw) |
| Balance: push recovery | **ABSENT** — no push/force machinery exists anywhere |
| Locomotion (any) | **ABSENT** — no command channel, no tracking metrics |
| Stance maintenance | **MEASURED FAILURE** (PD replay topples ~1.4 s; ankle 39/50 Nm) |
| Support polygon / fall detection | **ABSENT** (support-centre point only, inside the in-flight teacher) |
| CoM / capture-point logic | CODE-ONLY, in the in-flight teacher, unvalidated |
| Learned motor behaviour | **NONE** (best policy: 13.1% standing) |
| References (7 techniques) | exist; 6/7 fail execution; BODY_LOCK fails outright |
| Reusable | G1 MJCF + single-G1 scene, 29-joint actuator interface, PPO/vec/checkpoint infra, scorer, back detector, renderers, Visuals conventions |

## 2. Architecture (default: smallest that can work)

```
        ┌── drill scheduler (scripted commands for this milestone) ──┐
        │  skill ∈ {STANCE, SHUFFLE_F/B/L, CIRCLE_L/R, RETREAT,      │
        │           APPROACH, LEVEL_CHANGE, SHOT_DOUBLE_LEG, RECOVER}│
        │  + (v_cmd, w_z_cmd), stance_height_cmd, lead_leg           │
        └───────────────────────┬────────────────────────────────────┘
                                ▼
   proprio + gravity-relative orientation + base vel (local) + joint pos/vel
   + foot/hand/knee contacts + prev action + command + skill id
   + optional reference/ghost targets + phase progress
                                ▼
        ONE shared command-conditioned motor policy (2×256 tanh MLP)
                                ▼
        29 joint-position targets (existing actuator interface, 50 Hz ctrl / 500 Hz physics)
```
Escalate to a skill head / specialist only if measured interference blocks T5; a **hierarchical**
split only if tactical learning later stalls. Both require evidence, per MISSION.

**Virtual opponent**: never simulate grips. The shot is trained against *spatial markers* —
a virtual opponent pelvis/leg target region and hand-target sites — whose positions define
"entry depth/direction" reward terms. No contact force is implied or claimed.

## 3. Dependency plan (execution order; each step gated by its own evidence)

| step | work | depends on | evidence to proceed |
|---|---|---|---|
| S1 | **SoloEnv harness**: single-G1 scene, command channel, push machinery (`xfrc`), fall detector + termination, contact/slip metrics, phase labels, reward terms with hand-state unit tests, exploit probes vs trivial controllers, short **baselines** recorded, T1/T2 video skeleton + overlays | — | env validated: pushes measurably move the robot; fall detector fires on scripted falls; reward terms verified on hand-constructed states; trivial controller cannot farm reward |
| S2 | **T1 dynamic balance** (PPO) | S1 | held-out push battery recovery ≥ threshold; `01_balance.mp4` |
| S3 | **T2 locomotion** (velocity/yaw tracking) | S2 | tracking error + uprightness thresholds on held-out commands; `02_variable_locomotion.mp4`; M1 gate re-tested |
| S4 | **T3 stance footwork** + **T4 arm coordination** | S3 | stance-height tracking + circling without falls; reach accuracy without balance loss; `03_*.mp4`, `04_*.mp4` |
| S5 | **Demonstrations** for level change + penetration step: teacher-stabilized single-robot execution of the retargeted double-leg (attacker role), repaired until physically executable; capture obs-action pairs incl. from perturbed states | teacher (in flight) + S1 | demo executes reliably in sim (no fall, landmarks within tolerance, contact events sane) — else repair or use an alternative physically valid method |
| S6 | **T5 imitation** (BC) of the demo + **corrective imitation** (DAgger-style using the teacher as expert) | S5 | BC policy reproduces the shot under varying entry conditions (not fixed animation); `05a–05d.mp4` with ghost overlay |
| S7 | **T6 shot robustness** (RL with pushes at sampled phases) | S6 | held-out phase-push recovery; both legitimate outcomes appear (continue / abandon-and-recover); `06a–06c.mp4` |
| S8 | **T7 continuous drill**: scheduler over learned skills, physical-progress-driven transitions | S3–S7 | repeated shot+recovery cycles with no resets; `final_*.mp4` + metrics |

Forgetting control: every later stage re-runs the earlier gates; a failure re-opens that stage
(replay mixture of earlier skill commands during training where needed).

## 4. Contract (frozen for this milestone; document + test)

- **Physics**: model timestep 0.002 s (verify per scene), control 50 Hz (20 ms), action = 29
  joint-position targets in radians, clipped to `ctrlrange`; residual mode available
  (`ctrl = base + scale·tanh(z)`).
- **Termination**: sustained dorsal back-to-mat (existing detector semantics, persistence
  confirmed) = failed attempt; knees/hands/self-chosen low postures are NOT terminal.
  Balance/locomotion tasks add a fall detector (pelvis height + tilt + persistence).
- **Observation (actor)**: compact, self-relative; NO global simulator state; privileged
  fields (contacts, opponent/virtual-target internals) go to the critic only. Exact list is
  fixed in `src/solo/obs.py` and tested for shape/content.
- **Command**: `(vx, vy, wz)` in the robot's heading frame + `stance_height` + `skill_id` +
  `lead_leg`; sampled from feasible ranges, expanded gradually.
- **Episodes**: training episodes may reset; the FINAL evaluation must not reset within a
  successful run; evaluation is deterministic given a seed and saved config.

## 5. Rewards (candidate terms; weights are placeholders set from baselines, never invented)

Per-task selection from: velocity tracking `exp(-||Δv||²/σ²)`; yaw-rate tracking
`exp(-Δω²/σ²)`; uprightness (gravity-aligned torso) — paired with, never replaced by, tracking;
support/contact quality (no slip while loaded, sane step timing); joint-action smoothness and
torque/energy penalties; imitation terms (joint/root/site errors to the demo) with weight
decaying as physical performance improves; shot terms (penetration depth+direction toward the
marker, lead/trail leg sequence, controlled knee contact, recoverable posture, rise to stance);
recovery terms (restore support, stand, avoid dorsal contact). Explicit anti-exploit checks
required for: staying still; falling forward at commanded speed; sliding instead of stepping;
repeated squatting for level-change reward; parking the knee on the mat; entering a shot and
never leaving; correct arms while falling; command oscillation farming; terminal-reset farming.
Clock advancement must never be rewarded as physical progress (audit E2: the current progress
term is clock-based; motor tasks default it to 0).

## 6. Evidence duties (per AGENTS.md + §8 of the brief)

Each task: MP4 (960×720, 30 fps, h264/yuv420p, MuJoCo-rendered) + a 3-frame PNG contact sheet +
overlay (stage/task, checkpoint id, seed, command, phase, key metrics, success/fail), including
representative FAILURES; a ghost overlay whenever a reference is imitated; explicit push markers
on disturbance tests. Videos are indexed in `docs/VISUALS.md`; a stage is not done because an
MP4 exists — the numbers must agree with the video.

## 7. Deliverable map (brief §11 → artifacts)

1 env `src/solo/` + tests · 2 contract (this doc §4 + `src/solo/obs.py`) · 3 demos
`data/solo/demos/` · 4 BC `src/solo/bc.py` + checkpoints · 5 PPO runs `checkpoints/solo/` ·
6 curriculum `src/solo/curriculum.py` (competency gates) · 7 scheduler `src/solo/drill.py` ·
8 scripts `scripts/solo_*.py` · 9 checkpoints+configs · 10 metrics `data/solo/metrics/` ·
11 videos `videos/solo_drill/` · 12 `docs/VISUALS.md` · 13 `reports/2026-10-08/solo_drill.md`.

## 8. Honest-stopping rule

If compute, missing demonstrations, or model limits block the full drill: keep the best VERIFIED
checkpoint + footage, name the precise blocker, and report it. Claiming the milestone without
the continuous evidence is a failure of the session, not a shortcut through it.
