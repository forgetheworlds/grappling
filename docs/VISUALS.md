# VISUALS.md — Visual evidence index (operator verifies each stage visually)

Rule: every stage ships watchable artifacts. Agents render them; the orchestrator verifies
and indexes them here. Videos are 30 fps, 960x720, h264/yuv420p (in-repo, pushed to GitHub).
Each entry states WHAT TO LOOK FOR and the honest verdict.

| # | stage / deliverable | artifact | what to look for | status |
|---|---|---|---|---|
| 0 | env bring-up (d2) | `reports/2026-10-07/smoke_env.png` | plane + falling sphere, clean render | ✅ verified |
| 1 | G1 model standing (d5 in)| `reports/2026-10-07/g1_front.png`, `g1_side.png` | 29-dof G1 standing on mat; 5 s hold, 0.16 cm drift | ✅ verified |
| 2 | retargeted references (d5) | `videos/refs/{DOUBLE_LEG,SINGLE_LEG,BODY_LOCK,SNAPDOWN,SPRAWL,STAND_UP,STANCE}.mp4` | the 7 techniques executed by BOTH robots from GrappleMap geometry. **KNOWN:** pure joint-PD has no balance layer → robots sag/topple (pelvis zErr 0.29–0.83 m). Motion shape is right; staying up is not. | ✅ rendered, ❌ standing (expected pre-teacher) |
| 3 | stabilized teacher (d6) | `videos/teacher/<TECH>.mp4` (+ `reports/2026-10-08/teacher.md`) | **A/B vs stage 2**: same techniques, robots now STAY UP (target ≥80% stay-up frames); defender prone on SPRAWL by design | ⏳ in flight (TeacherRetry) |
| 4 | technique scorer (d7) | `reports/2026-10-08/scorer.md` tables (numeric, non-visual) | scores ≥0.98 on own traces; cross-technique low | ✅ verified |
| 5 | env rules + back-to-mat (d10/11) | `videos/env/<scenario>.mp4` (+ 3-frame PNG contact sheets) | standing start → draw; SPRAWL takedown → back event → score; ambiguous simultaneous fall; OOB forfeit | ⏳ rendering now |
| 6 | PPO stage A–B (imitation + drilling) | `videos/ppo/<stage>_<tech>.mp4` | commanded technique reproduced by the learned policy | pending |
| 7 | PPO stage C–E (resistance) | `videos/ppo/<stage>_vs_<opponent>.mp4` | technique survives a moving/resisting opponent | pending |
| 8 | free wrestling (no technique command) | `videos/free/<date>_exchange*.mp4` | range management, circling, attack choice, recovery, re-engage | pending |
| 9 | self-play match (d23) | `final_wrestling_match.mp4` | full 3-minute match: stance → movement → attack → defense → recovery → takedown → score → reset; sequence emergent | pending |
| 10 | failure videos (d20) | `videos/failures/*.mp4` | documented failure modes at each stage (falling, stalls, degenerate attacks) | pending |

## How to verify quickly
- Play a video, or read a PNG contact sheet for a 3-frame summary.
- Compare stage 2 vs stage 3 side by side — that pair is the proof the teacher works.
- Videos live in-repo and in the GitHub remote (`videos/`), so they're reviewable anywhere.
