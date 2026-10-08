# AGENTS.md — Grappling Project Agent Instructions

Read `goal.md` (northstar) and `notes.md` (orchestrator ledger) BEFORE working.
Append any verified FACTS you produce to `notes.md` under the right section
(or to your report; orchestrator merges). Never delete ledger entries.

## Project overview

Learned standing wrestling: two Unitree G1 humanoids in MuJoCo, trained
GrappleMap→imitation→resistance→self-play. Primary artifact: `final_wrestling_match.mp4`.

## Build, test, run

- Python venv: `source .venv/bin/activate` (create if missing; see docs/SETUP.md).
- Run tests: `python -m pytest tests/ -x -q` (fallback: `python -m unittest`).
- Every script must be runnable from repo root and print its own verification.

## Conventions

- Directories: `src/` (importable packages), `scripts/` (runnable entry points),
  `tests/`, `data/` (generated+curated artifacts, npz/json), `robots/` (MJCF+meshes),
  `third_party/` (vendored clones — never edit), `workflows/` (workflow scripts),
  `docs/`, `reports/`, `checkpoints/`, `videos/`.
- Units: SI only. Radians in all code. Z-up world. State frame conversions in one
  place, documented in notes.md "Interface contracts".
- VISUAL EVIDENCE IS REQUIRED (operator directive, 2026-10-08): every deliverable that
  produces motion renders at least one 30 fps 960x720 mp4 under `videos/<stage>/` plus
  a 3-frame PNG contact sheet, states WHAT TO LOOK FOR, and gets indexed in
  `docs/VISUALS.md` (orchestrator maintains the index). A stage is not "done" without
  watchable evidence and an honest verdict.
- NumPy float64 for data, float32 only at network boundaries. Random seeds explicit.
- No GPU. ARM aarch64. Prefer stdlib+numpy; new deps require a note in notes.md.
- Keep modules small and boring. Delete dead code. No speculative abstraction.
- Every deliverable script ends with a self-check (asserts or printed verification).

## Out of scope / forbidden

- No continued ground grappling beyond back-to-mat termination.
- No editing `goal.md` (orchestrator only). No edits in `third_party/`.
- No CUDA installs, no large frameworks (no Isaac, no Brax, no MJX unless orchestrator
  approves in notes.md).
- Never weaken a test to make it pass; report the failure instead.
