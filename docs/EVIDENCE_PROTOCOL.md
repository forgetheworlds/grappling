# EVIDENCE_PROTOCOL.md — what a deliverable must ship with

Applies to every motion deliverable (teacher clips, drill ladder, acceptance clip, baselines,
failures). Purpose: make any claim independently checkable from the repo without re-running, and
make the operator's visual verification trivial.

## The bundle (per deliverable)

| item | requirement |
|---|---|
| **video** | 960×720, 30 fps, h264/yuv420p, MuJoCo-rendered; the *acceptance* clip is one continuous unedited trajectory. Diagnostic/low-res clips are allowed but must be named as such (`*_diag.mp4`) and never called the acceptance artifact. |
| **contact sheet** | 3 frames (start/mid/end) as PNG beside the video. |
| **metrics JSON** | beside the video, same basename: per-run aggregates AND per-element/per-phase metrics; include `reset_count`, `fall_count`, `longest_continuous_s`, foot-slip, CoM margin min, tilt max, saturation fraction, and command-vs-realised tracking where applicable. |
| **provenance block** | inside the metrics JSON: git commit of the repo at run time, the controller/reference module + version/date, seed(s), config path + hash, control rate, physics timestep, wall time, and the exact reproduce command. |
| **rubric table** | `docs/QUALITY_RUBRIC.md` scores per element (0–3) with, for every score ≥ 2, a numeric trace and at least one annotated frame proving it. |
| **failures** | representative failures kept and labelled, with the same bundle minus the "acceptance" claim. Never replaced by a better clip. |
| **index** | `docs/VISUALS.md` row updated by the owner: artifact paths, what to look for, verdict, and the honesty note (scripted-feedback vs learned). |

## Claim hygiene

- A video proves *what happened*; numbers prove *how much*. Neither alone is a result.
- Worst-seed / worst-case numbers must be reported alongside means, for complete cycles — not
  frame-fraction survival averaged over seeds.
- Do not gate on: geometry-scorer score alone, low pelvis height, `stood` (no dorsal + no OOB), the
  env's `back`/OOB events in a continuous-evidence run, or clock advancement. Physical events
  (loaded support, completed step, completed rise, completed cycle) are the gate.
- Any number copied from a report into a new report must be re-checked against the artifact, and
  provenance conflicts resolved by re-running — never by picking the favourable value.

## Naming

```
videos/<stage>/<name>.mp4          # acceptance or rung clip
videos/<stage>/<name>_sheet.png    # 3-frame contact sheet
data/<stage>/<name>.json           # metrics + provenance (same basename)
reports/<date>/<stage>.md          # narrative: what works, what does not, next blocker
```
