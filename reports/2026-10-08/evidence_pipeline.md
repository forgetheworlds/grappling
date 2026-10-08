# Evidence pipeline — verification is gating, and every clip's claims are checked

**Status: completed from the killed agent's work.** Agent `EvidenceFix` was stopped by the provider
outage after landing the fix, the gating rule, the label-integrity check and its tests (committed in
`ce8dc3d`); it never wrote this report. Everything below is reproduced from the tools as they stand
on disk — no claim is quoted from the agent's prose.

**Tools**

| file | what |
|---|---|
| `scripts/solo_drill_render.py` | the renderer; `verify_clip` + the `CLIP_SPECS` single source of truth (window/speed/title/caption per clip) |
| `scripts/evidence_check.py` | `table` (machine status table), `claims` (label-integrity), `refresh`, `gate` |
| `tests/test_evidence_integrity.py` | pins audited defects D1–D9 (each test reconstructs the defect and asserts the new check FAILS on it) |
| `data/solo_drill/<clip>.json` | per-clip bundles carrying `clip_verification`, `evidence`, `render.window`, `render_command` |

```
MUJOCO_GL=egl .venv/bin/python scripts/evidence_check.py table     # the status table
MUJOCO_GL=egl .venv/bin/python scripts/evidence_check.py gate      # exit 0 only if every bundled clip is evidence
MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_evidence_integrity.py -q
```

## 1. The root cause of the never-completing verifier (D9)

`verify_clip` died on `int(inf)`: imageio reports `nframes: inf` for ffmpeg-written mp4s, and the
verifier fed that into an integer conversion, so **no clip had ever been machine-corroborated** —
12 of 13 bundles recorded `clip_verification.ok=false` ("decode failed: cannot convert float
infinity to integer") while the index claimed the clips were verified. The fix (commit `103dcee`)
makes `verify_clip` complete and fail on genuinely static/black output. `test_verify_clip_completes_and_fails_static_or_black`
pins it.

## 2. Verification is gating

Each bundle now carries `evidence: bool(ver["ok"])`, and `evidence_check.py gate` **exits nonzero
unless every bundled clip is evidence**. Measured now: **gate exit 0 — 15/15 bundled clips verify**.
The index may only present a clip as evidence when its bundle says so; a silently-failing check can
no longer coexist with a "verified" claim.

## 3. The re-verification table (machine-generated)

`scripts/evidence_check.py table` regenerates it; current summary:

* **15 bundled clips — all `evidence`** (decode + format + window/frame match + non-black/non-static
  + every numeric label claim checkable from the bundle/trace). Highlights:
  * `02_slowmo_level_change_quarter_speed.mp4`: **32.267 s / 968 f at 0.25×** — the slow-motion
    clip is now actually rendered at the speed its name claims (8.06 s of run / 0.25 = 32.2 s);
  * `L1_90s_with_pushes.mp4`: 60.033 s / 1801 f, and its title now states "t=62 s outside this
    clip" (the third push is outside the deliberate 0–60 s window);
  * `final_L2_motion.mp4`: 69.98 s / 5 steps, matching its trace (the old index claimed 95 s / 6
    steps / 0.30 m — D2);
  * `99_failure_push90N_diag.mp4`: verified, with the honest note "no such push force in this run
    (cross-run claim: not checkable from this bundle)".
* **27 unbundled clips** (baselines probes, `videos/refs/*`, `videos/teacher/*`, `videos/env/*`):
  format + pixel checks pass, but there is no bundle/trace to corroborate any claim — their status
  is *unverified*, stated as such.
* **No bundled clip is KNOWN-BAD** at this revision.

## 4. The mechanical label drifts fixed

* **Speed pass-through** (D3): a clip named "quarter speed" was rendered at 1.0×. The renderer now
  takes `--speed` and `CLIP_SPECS` carries it; `window_expectations` computes
  `(min(t1, trace_end) − t0)/speed` frames independently of the renderer.
  `test_slowmo_duration_ratio_must_match_the_claimed_speed` asserts the ratio.
* **The `_diag` index paths**: repaired; the index's measured numbers now match the evidence table
  (e.g. `L1_90s_with_pushes` 60.033 s / 1801 f).
* **The hand-typed window that excluded its own event** (D7): the t=0–60 s push clip cannot show
  the t=62 s push; the claim now says exactly that
  (`test_push_outside_window_must_be_acknowledged`).
* **Window expectations clamp at the trace's end** (D8): an episode that falls at 2.82 s cannot
  produce 11 s of video; the old expectation `(t1 − t0)` recorded three complete clips as
  "truncated mid-write".
* **Caption corrections**: F1's "it topples" was false for the run it labelled (D6); the stepping
  base is 0.21 m wide, not 0.25 m (D4); `L2_shuffle_3steps_diag` completes 4 steps, not "2–3" (D5).

## 5. The label-integrity check (the durable fix)

`evidence_check.py claims` compares every **numeric** claim a clip's label/caption makes (duration,
step count, stance width, push magnitude) against the bundle/trace it was rendered from and fails
on mismatch. Coverage is the fields that appear in `docs/VISUALS.md`; claims that cannot be checked
mechanically (cross-run forces, hand-typed windows) are **flagged as unverifiable rather than
silently passed** — the 90 N push is the worked example.

## 6. Which audited defects the new tests pin

D1 `test_gate_refuses_unverified_clip` · D2 `test_label_step_and_duration_claims_fail` ·
D3 `test_slowmo_duration_ratio_must_match_the_claimed_speed` ·
D4 `test_stance_width_claim_fails` · D5 `test_step_range_claim_fails` ·
D6 `test_topple_claim_vs_trace_falls` · D7 `test_push_outside_window_must_be_acknowledged` ·
D8 `test_window_expectation_clamps_at_trace_end` ·
D9 `test_verify_clip_completes_and_fails_static_or_black`.

## 7. Honest status of what we ship

Every bundled solo-drill clip is now evidence *for its stated window and claims*; the 27 unbundled
clips are format-verified only and must not be cited as evidence for any claim. The known-bad
classes from the audit (the never-running verifier, the 1.0× "slow motion", the drifted step/duration
claims, the window that excluded its own push) are all pinned by tests, so they cannot return
silently.
