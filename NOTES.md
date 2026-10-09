# NOTES.md — branch `successful-reimplementation` (pruned)

Replaces the 2,484-line `notes.md` orchestrator ledger, retired in this prune
(git history keeps it). This file is the standing summary of what the branch
contains, how to reproduce it, and what is still open.

## What this branch contains

Two Unitree G1 humanoids learning standing wrestling in MuJoCo
(GrappleMap -> imitation -> resistance -> self-play; primary artifact
`final_wrestling_match.mp4`, not on this branch). After the prune this branch
carries ONLY the certified v2 motion-reference + solo-track slice:

- `src/`, `robots/`, `workflows/` — full training/eval code (untouched).
- `data/references/motion_refs/v2/` — the re-timed reference set: 14 takes
  under `v2/refs/`, `drill_continuous.npz`, `stance_rise.npz`, `feasibility.json`,
  `RETIMING.md`; plus top-level `clip_index.json`, `fusion_spec.json`,
  `feasibility.json`, `FORMAT.md`. `v1/refs/stance_hold.npz` survives only
  because the two kept tests read it by explicit path.
- `data/solo/metrics/` — full metric archive backing the numbers below.
- `tests/` — `test_track.py`, `test_motion_refs_v2.py` (both green).
- `scripts/` — the 8 pipeline entry points (below); everything else removed.
- `docs/` — `SOLO_DRILL.md`, `QUALITY_RUBRIC.md`, `EVIDENCE_PROTOCOL.md`.
- `reports/2026-10-08/` — `motion_reference.md`, `motion_learning.md`,
  `reference_retiming.md`.
- `checkpoints/solo/` — exactly `t1_balance_v6e.pt`, `t1_balance_v6e_401408.pt`,
  `track_s1_v2refs.pt` (gitignored; force-added).
- `videos/` — exactly `motion_refs/v2_lower_balance.mp4` (+ sheet) and
  `solo_drill/track/v2refs_lower_pass.mp4` (+ sheet).

## Pipeline order

    build_v2_refs -> solo_track_train -> solo_track_eval -> solo_track_render

Build the v2 reference dataset, train S1 conditioned on it, evaluate through
the gates, then render the watchable 30 fps 960x720 mp4 + contact sheet.

## Verified numbers

- v2 LOWER open-loop: 99/99 frames complete @ site RMS 0.020 m.
- Trained S1 hard-gate: 1.00.
- Held-out seeds 1000-1003, S2 SHUFFLE_F: 3/8 complete, gate 0/8,
  deviation-limited (the open failure mode).
- T1 v6e @ 800k: upright 0.949, fall 0.208 — NOT certified;
  survivor-stance 6/19.

## Open items

- Gait contact-observation fix (contact signal the gait policy needs).
- v6f `stance_return=1.0` lever (next structural experiment for T1).

## Verify after checkout

    cd /home/ubuntu/grappling-prune
    MUJOCO_GL=egl /home/ubuntu/grappling/.venv/bin/python -m pytest tests/ -q
    MUJOCO_GL=egl /home/ubuntu/grappling/.venv/bin/python \
        scripts/query_motion_refs.py drill_continuous --phase
