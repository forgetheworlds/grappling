# AGENT 3 BRIEF — Reference Re-Timing / Dynamic-Feasibility Engineer

Repo `/home/ubuntu/grappling`. Read `AGENTS.md`, `goal.md`, `notes.md` (the two-agent pipeline section
is current), `docs/SOLO_DRILL.md`, `docs/QUALITY_RUBRIC.md`, `docs/EVIDENCE_PROTOCOL.md`,
`reports/2026-10-08/motion_reference.md`, `reports/2026-10-08/motion_learning.md`, and
`data/references/motion_refs/FORMAT.md` + `feasibility.json` before acting.

## Why this task exists (three independent measurements, one conclusion)

The v1 references are kinematically clean and grounded, but their **timing** is not dynamically
achievable by the G1:
1. The motion-learning agent trained ~700k steps in 6 arms; the dynamic LOWER segment fails at
   **0.36 s** in every arm with the feet planted and joints tracking at 0.04 rad while the ROOT
   drifts **0.156 m fore-aft in 0.36 s** (torso diverges 0.25 m). It is a reference property, not
   an observation/reward defect.
2. Direct probes: the retargeted shot crouch topples the robot in **1.18–1.24 s** while holding its
   OWN joint targets; a 41,840-rollout CEM search (1 h) reached only 2.34 s.
3. The source data audit measured only **6.9 %** of teacher poses statically holdable, and the v1
   `feasibility.json` labels the crouch `known_infeasible`.

## Mission

Produce a **v2 reference set** whose timing (and, where sanctioned, geometry) is dynamically
achievable, so fixed-clock reference-conditioned training can progress past the current 0.36 s
wall. Do not silently delete hard segments: re-time or re-shape them, or label them infeasible with
the measurement that proves it.

## Method (evidence-first; inspect the options before choosing)

- **Per-segment time-scaling**: slow the root/contact schedule until the required root
  accelerations and foot-load transitions are inside what the certified stance controller (and the
  T1 policy) can execute. Measure the achievable envelope; do not guess.
- **Root-path / contact-schedule re-shaping**: weight transfer before a lift, plant before a push,
  bounded fore-aft root acceleration; keep the movement's identity and rhythm recognizable.
- **Dynamics-in-the-loop oracle**: the repo already has one (`src/solo/demo.py` CEM capture,
  `src/solo/exec_check.py` checker). It is expensive (41,840 rollouts ≈ 1 h for the shot) — use it
  where it buys a decisive answer, not as a blanket re-solve.
- **Sanctioned geometry repairs** (operator's stance rules + QUALITY_RUBRIC A-section): lateral
  base width, rear-foot depth, knee/hip-driven crouch rather than waist folding, head up, plausible
  guard. Preserve wrestling identity; never flatten a crouch into ordinary walking to make balance
  easy.
- A balance layer in the probe (the certified stance controller or the T1 v6d policy) is required
  to claim a segment is dynamically traversable — raw position servos have no balance layer and
  their failures are expected, not evidence.

## Deliverables

1. `data/references/motion_refs/v2/` — the re-timed/re-shaped set in the v1 format, with per-phase
   labels preserved (source, lead leg, validity), plus a `RETIMING.md` recording, per segment: what
   changed, the measured before/after (root acceleration, contact timing, probe outcome), and why.
2. `feasibility.json` v2 — three levels per segment, re-measured, with the probe command and the
   balance layer used.
3. Evidence per `docs/EVIDENCE_PROTOCOL.md`: MP4s (960×720/30 fps/h264) + 3-frame sheets + metrics
   JSON with provenance, including at least one **dynamic probe with a balance layer**, labelled
   honestly as success or failure.
4. A short report stating what is now traversable, what is not, and the exact next blocker.
5. Tests pinning real contracts (format round-trip, timing monotonicity, the re-timed root's
   acceleration bounds); never weaken a test; keep the full suite green.
6. `notes.md` updated with verified facts; commit with honest messages.

## Acceptance (the orchestrator gates this)

- **The decisive read**: with the v2 reference, fixed-clock reference-conditioned training
  (`src/solo/track.py` + `scripts/solo_track_train.py`, unchanged interface) progresses **past
  0.36 s on the LOWER segment** — or a specific, measured reason is given for why it cannot, with
  the next lever named.
- **Falsifier**: if a re-timed reference still fails at the same place, the blocker is not the
  reference timing (it is the balance layer, observation or action design) — say so explicitly
  rather than re-timing again.
- Every claim recomputable by the orchestrator from the artifacts; no kinematic data labelled as a
  control demonstration; failures kept and labelled; no framework migrations; no edits to
  `goal.md`/`third_party/`; do not disturb unrelated work on the box.
