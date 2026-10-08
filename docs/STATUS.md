# STATUS.md — verified state of the grappling project

Living document. Only claims with a measured number and a source path belong here; everything else
belongs in `notes.md` as a hypothesis. Last updated: 2026-10-08 ~11:30.

## The goal and the honest gap

**Goal**: two learned G1 humanoids wrestling a full match (`final_wrestling_match.mp4`).
**Session milestone**: one G1 performing a continuous solo stance-and-motion drill.
**Where we are**: the *plumbing* is real and verified end-to-end (sim, parser, retargeting, rules,
scorer, RL infrastructure, a model-based drill controller), but **no learned motor competence has
been demonstrated yet** — three training runs have been diagnosed and corrected rather than shipped
deaf. The current solo artifact is a *machine-controlled* drill (12.1 s/step, 0 falls), not a learned
one. That distinction is stated everywhere the artifact is referenced.

## Verified, with numbers (source in parentheses)

| layer | evidence |
|---|---|
| Environment | mujoco 3.15.0, torch 2.14.1+cpu, EGL headless render, 4-core ARM (docs/SETUP.md; smoke tests green) |
| GrappleMap | 725 nodes / 1485 edges parsed; 23 landmarks/player verified from source; mirror/swap involutions tested (reports/2026-10-07/grapplemap.md; 17 tests) |
| G1 model | 29-dof, nq36/nv35/nu29, 50 Hz control over 0.002 s physics, 33.341 kg, stands 5 s with 0.16 cm drift (reports/2026-10-07/g1_model.md) |
| References | 7 techniques retargeted and rendered; joint tracking ≤0.06 rad but **only 6.9% of frames are statically holdable** — ankle torque is NOT the binding constraint (≤23.9 N·m vs 50 N·m); the CoP-inside-contact-patch geometry is (reports/2026-10-08/support_envelope.md) |
| Rules env | exchange loop, 0.30 s dorsal persistence (sens/spec 1.000), cross-step ambiguity window, OOB forfeit; 5 rule clips frame-verified (reports/2026-10-08/wrestling_env.md, env_videos_frame_check.md) |
| Scorer | relational-geometry judge, own-trace mean 0.995 (rebuilt STAND_UP) / 0.987–1.000 others; strictly separate from any value function (src/scorer; reports/2026-10-08/scorer.md) |
| RL infrastructure | actor 92 (2-robot) / 115 (solo) obs, privileged critic, PPO with GAE, subprocess vectorisation (1.6–2× sequential), atomic checkpoints, SIGINT resume verified; 175 tests pass (src/rl, src/solo; reports/2026-10-08/rl_infra.md) |
| Push machinery | `qfrc_applied` exactly zero when idle; momentum matches the applied impulse within 0.02 % (no silent clamping); 12 N·s topples at z=0.95 but not 0.79 (src/solo/pushes.py; tests) |
| Drill (machine-controlled) | L1 90 s continuous, 0 falls, 0 resets, worst CoM margin +0.0295 m; L2 **70 s, 5 real steps, 0 falls**, cadence 12.1 s/step, stance 0.495 m wide × 0.241 m deep, CoM margin +0.086 m built; same-pose PD baseline topples at 8.4 s (videos/solo_drill/final_L2_motion.mp4 + data/solo_drill/final_L2_motion.json) |
| Drill — independent verification | frame-level checks confirm CLEAN and, for L2, **real motion** (5 steps seen at 15.40/28.80/45.40/50.22/63.98 s, 8–11 cm re-plants, single support visible); L1 was found nearly STATIC and its "18 cycles" claim corrected (reports/2026-10-08/l1_clip_visual_check.md, l2_clip_visual_check.md) |
| Operator reference | YouTube reference acquired (10 chapters: STANCE/STALKING/CIRCLING/LEVEL CHANGE/SHOTS/…) with CPU pose estimation (mediapipe, 13–14 fps) → **12 retargeted G1 tracks** + a 41-parameter stance spec (G1-equivalent width 0.49 m, pelvis 0.67 m, hands low, torso pitch 47°) (data/refs_video/, docs/REFERENCES.md) |

## Learned policies: three runs, all diagnosed (no shipped policy yet)

| run | change | measured outcome |
|---|---|---|
| T1 v1 | — | **degenerate**: 0/7 gate criteria, mean_upright 0.006 (scripted stand 0.849); learned to dodge the −100 termination by collapsing into a limb-supported pose. Stopped. |
| T1 v2 | `entropy_coef` 0.01→0.001 **+ training push curriculum ≤12 N·s** (gate battery untouched) | entropy now peaks and FALLS (was climbing to 26); return −102→−85; gate **2/7** (fall_rate 0.042 ✓, held-out falls 0.000 ✓, upright 0.0415 ✗). Stopped at the pre-declared condition. |
| T1 v3 | `--alive-weight 10` (hand-checked first) | **running** — entropy flat ~12.13, return rising to +90 (v2 never passed −85). Gate read at ~1.1M steps decides. |

If v3 still collapses, the named next single lever is the **termination-penalty magnitude**, not
"adding reward signal" (the alive term is already dense).

## Open gaps (honest)

1. **No learned locomotion/stepping**: the machine-controlled shuffle is 12.1 s/step; the measured
   reason is that the ~9 s full recentre is load-bearing (11 faster variants all fell), so a dynamic
   gait needs the learned layer. Circle footwork not achieved at all.
2. **No level change / penetration / knee / recovery in the drill**: L3/L4 not reached; the
   reference tracks for those motions exist but are not holdable at frame 0 (they need the same
   geometric repair our GrappleMap references needed).
3. **Two-robot wrestling, resistance, self-play, `final_wrestling_match.mp4`**: not started.
4. **Known residual**: 3 STAND_UP band-provenance deviations (≤0.114 s) from the rebuilt reference
   meta vs the old schedule model (reports/2026-10-08/test_regressions.md).

## Reproduce (each verified tonight)

```bash
# environment + tests
source .venv/bin/activate && python -m pytest tests/ -q          # 175 passed
# the drill artifact (machine-controlled; re-simulates then renders all clips)
MUJOCO_GL=egl python scripts/solo_drill_render.py suite
# training (checkpointed, resumable, never holds the sim lock across a long run)
MUJOCO_GL=egl python -m src.solo.train --task balance --steps 2000000 --rollout-steps 2048 \
    --gamma 0.995 --entropy-coef 0.001 --alive-weight 10 --out checkpoints/solo/t1_balance_v3.pt \
    --save-every 100000 --lock off
# gate evaluation of any checkpoint (the mid-run probe protocol)
python scripts/solo_env_smoke.py monitor     # copies the ckpt, runs the 48-push battery vs baselines
```

## Standing process rules (learned the hard way tonight)

- Verify at the moment of commit: duration/frames for any mp4, hash for any generated calibration,
  truthfulness of every claim against the artifact.
- Renders write `.partial` → verify → atomic rename; supervised jobs, never in-process background.
- One change per training run, then measure; never stack changes.
- A scheduler/HUD label is not evidence; physical events and the trace are.
- Frame-level visual checks catch overstatements that metrics miss.
