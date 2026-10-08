# Phase 4 visual evidence — wrestling env rules + back-to-mat (VISUALS stage 5)

**Agent:** EnvVideo · **Date:** 2026-10-08
**Deliverables:** `scripts/render_env_videos.py`,
`videos/env/{draw_stance,takedown_back_event,nonterminal_knees_hands,ambiguous_simultaneous,oob_forfeit}.mp4`
(+ `_sheet.png` 3-frame contact sheets), this report.
**Verification:** `.venv/bin/python scripts/render_env_videos.py` (from the repo root,
`MUJOCO_GL=egl`) prints a per-scenario evidence block, per-scenario `PASS`/`FAIL` lines and
a final `SELF-CHECK OK`; `--diag` reproduces the scenario-1 deviation; ffprobe on the
outputs reports `h264 (High), yuv420p, 960x720, 30 fps` for all five clips.
**Files touched:** `scripts/render_env_videos.py`, `videos/env/*` (new), this report.
Nothing under `src/`, `robots/`, other `videos/` directories or `notes.md` was modified by this
work. (Note: `src/wrestling/env.py` *was* changed mid-flight by another agent — the ambiguity
hold — which is why §2's numbers carry a trigger+hold end time; see §6.9.)

---

## 0. What was rendered, and how

All five clips are produced by one script, one process, from the **shipped** env API
(`wrestling.env.WrestlingEnv` + `ReferenceReplay`/`StandHold`), deterministic (`seed=0`),
5.0–6.0 s of simulated time each (152–182 frames at 30 fps):

| scenario | clip | frames | camera distance | render wall |
|---|---|---|---|---|
| `draw_stance` | 5.07 s | 152 | 4.00 m | 66.5 s |
| `takedown_back_event` | 5.57 s | 167 | 4.00 m | 72.2 s |
| `nonterminal_knees_hands` | 6.07 s | 182 | 4.00 m | 78.9 s |
| `ambiguous_simultaneous` | 5.57 s | 167 | 4.00 m | 90.8 s |
| `oob_forfeit` | 6.07 s | 182 | 4.36 m | 92.6 s |

Full pass (sim + render + sheets): **413 s wall** in a single process, one renderer at a
time (`renderer.close()` in a `finally`), no other renders launched in parallel.
Re-rendering a single scenario costs 65–160 s (the three clips re-rendered after the first pass
— the two sheet-timing fixes and the mid-flight rules change, §6.9 — took 94 s, 143 s and 157 s
under concurrent load from another agent's simulations).

**Camera.** Free camera, `azimuth=90` (side view — the pair axis is horizontal in frame),
`elevation=-14`, lookat = the pair centroid (fixed at the declared mat centre for the OOB
clip so the circle reads as a fixed arena), constant distance per clip from the worst-frame
fit clamped to a legibility window (4.0–5.5 m). Both robots are inside the frame in every
frame of every clip (checked on the rendered frames and the sheets).

**HUD (every frame).** `t`, match clock, exchange index, exchange clock/limit, running score,
last exchange event, and per robot: `back_det: dorsal / tilt / pelvis_z / limb (with the
contacting limb bodies) / trigger time`, plus the pelvis radii and the OOB counters. When an
exchange ends a red banner names the cause (back / oob / timeout draw / ambiguous + reset).

**Contact sheets.** 3 panels, 960x720 each, labelled `start` / `mid` / `end` with their sim
times and the scenario's "what to look for" as a caption. `start` and `end` are the first and
last frame of the clip; `mid` is the clip midpoint *except* where the scenario's key moment is
elsewhere, in which case it is that moment (takedown: 1.60 s = dorsal contact established;
ambiguous: 0.28 s = both supine with the window running; OOB: 1.60 s = offender outside the
ring). The reason is stated per scenario below.

**Render settings.** This host renders through software EGL (no `/dev/dri`): MuJoCo's default
shadow pass alone costs ~6x the frame time (measured 0.41 fps with defaults vs 2.6 fps
without), so the evidence clips render with shadows/reflections off and `offsamples=1`
(measured 0.43 s/frame idle, up to 0.85 s/frame while another agent's MuJoCo sims were running
concurrently). Lighting, materials and geometry are otherwise the shipped look.

**Mat ring (render-only).** For the OOB clip a dashed ring of 72 capsule posts
(`contype=0 conaffinity=0`, added through `MjSpec` on a *separate* model) marks the declared
1.5 m competition circle, which is otherwise invisible (the scene floor is an unbounded plane).
The environment keeps using `robots/wrestling_scene.xml` untouched, so every rule below ran on
the shipped model.

**`EvidenceEnv` (start-pose shim).** All env rules, detectors, rewards and the exchange loop are
the shipped ones. Only the *start pose of the reset exchanges* is replaced: the delivered env
resets to the randomized STANCE reference frame, which is not balanceable open-loop
(`reports/2026-10-08/wrestling_env.md` §3.1) — with it every exchange after the first collapses
within ~2 s and the clips would show that fall instead of the rule under test. `EvidenceEnv`
therefore resets to the model's verified-stable `both_stand` keyframe (the pose the delivered
20 s-draw test passes through `reset(pose=...)`); exchange 0 is still overridden through the
env's own `reset(pose=...)` hook where a scenario needs a specific state. Scenario 1 keeps the
raw STANCE reference frame as the start pose (that is its subject).

---

## 1. `draw_stance` — timeout rule → draw → standing reset

**WHAT TO LOOK FOR.** Both robots in the retargeted STANCE (grappling crouch, arms engaged),
the exchange clock in the HUD running to its limit, the score staying `0:0`, the back detector
silent, and the red `EXCHANGE n END — cause: timeout — no score (draw) — reset` banner followed
by both robots resetting standing.

**Setup.** `ReferenceReplay("STANCE","a"/"b")`, start pose = the STANCE reference first frame
for *every* exchange, `exchange_timeout=1.0 s`, `match_clock=60 s`, seed 0.

**Observed (measured).** Five exchanges in 5.0 s, all bit-identical:

| # | cause | winner/loser | ambiguous | oob a/b | back triggers a/b | end tilt a/b | end pelvis z a/b |
|---|---|---|---|---|---|---|---|
| 0–4 | `timeout` (1.0 s each) | none | False | 0/0 | None / None | 31.73° / 20.43° | 0.7403 m / 0.7237 m |

Zero frames with dorsal *or* limb mat contact in any exchange; score `0:0`; the contact sheet
(start / mid / end) shows the stance, the clock and the draws. **Verdict: the timeout rule
draws with no score and resets standing — verified.**

**Deviation (stated, reproducible).** The assignment's "runs to the timeout rule" cannot use the
20 s clock with the raw STANCE replay: `.venv/bin/python scripts/render_env_videos.py --diag`
(reproducing the measurement) gives

```
cause=back, winner=a, loser=b, end_time=1.88 s, back_triggers={"a": null, "b": 1.78},
end_back a: dorsal=True tilt=78.8° pelvis_z=0.062; b: dorsal=True tilt=83.3° pelvis_z=0.070
```

i.e. the untaught crouch topples backwards after ~1.4 s and the detector *correctly* scores it.
The clip therefore uses the env's own 1.0 s-clock draw recipe
(`tests/test_wrestling.py::test_exchange_timeout_draw`), so each exchange ends while both robots
are still up (pelvis 0.72–0.74 m, tilt 20–32°). The topple is reported here, not hidden, and it
disappears once the phase-3 teacher exists.

---

## 2. `takedown_back_event` — back-to-mat rule → persistence → score → reset

**WHAT TO LOOK FOR.** The defender's back reaching the mat, the *persistence window* before the
trigger (0.30 s `confirm_s` + one 0.02 s control step), the score flipping `A +1.00 : -1.00 B`,
and the standing reset.

**Setup.** Exchange 0 starts from the SPRAWL reference first frame (`reset(pose=...)`, the
delivered recipe), both robots on `ReferenceReplay("SPRAWL")`; after the score both switch to
`StandHold`.

**Observed (measured).**

| event | time | evidence |
|---|---|---|
| defender `b` first dorsal torso contact with the mat | **1.38 s** | tilt 71–78°, pelvis 0.079 m, `dorsal=1` in 12 frames |
| detector trigger for `b` | **1.70 s** | 0.32 s of sustained condition ≈ `confirm_s` 0.30 s + one control step |
| exchange 0 record | end **1.80 s** | `cause=back, winner=a, loser=b, ambiguous=False`, rewards `(+1, −1)`, oob `{a:0,b:0}`; the end time is the trigger + the env's 0.10 s ambiguity hold (see §6.9), i.e. the exchange now closes one hold-window after the first trigger |
| attacker `a` | never | `dorsal` contact in **0** frames; its mat contacts are limb only — `a_right_knee_link x17, a_right_wrist_yaw_link x4, a_left_wrist_yaw_link x3, a_left_knee_link x3, a_right_elbow_link x2, a_left_elbow_link x1` (end tilt 115.1°, pelvis 0.311 m) |
| standing reset | 1.80 s → | exchange 1 (`both_stand` + `StandHold`) has no contacts; score stays `+1.00 : −1.00` |

The contact sheet's middle panel is taken at **t = 1.60 s** (dorsal established, `trig=-`) so the
persistence window is visible on the sheet; the delivered calibration JSON gives the same event
at 1.682 s (`data/backdet_calibration.json.trigger_times`) — this run lands one control step
later.

**Honest label: this is a detector demonstration.** The pure-PD SPRAWL replay has no balance
layer, so the *attacker* also topples (tilt 113° at the end) — that is exactly the phase-2
limitation the env report documents. The claim this video supports is narrower and exact: the
rule fires on the defender's dorsal torso contact, after the persistence window, and on nothing
else (the attacker's many limb/knee/hand contacts never trigger).

---

## 3. `nonterminal_knees_hands` — knees/hands are never terminal (MISSION proof)

**WHAT TO LOOK FOR.** A robot down on hands/knees with the mat, `limb=1` (the contacting limb
bodies are named in the HUD), `dorsal=0`, the back detector silent, the score frozen at `0:0`,
the exchange clock running down, then the draw and the standing reset.

**Setup.** Exchange 0: robot `a` = `ReferenceReplay("SPRAWL")` from the SPRAWL reference start
(the attacker dives to the mat), robot `b` = `StandHold` standing 1.05 m away (out of reach, so
no back event can occur); after the draw both `StandHold`. `exchange_timeout=5.0 s`.

**Observed (measured).**

| quantity | value |
|---|---|
| `a` first limb (knee/hand/elbow) mat contact | **0.70 s** |
| frames with `limb_contact=True` (exchange 0) | **121 / 150** |
| limb contacts by body (`a`) | `right_knee_link x78, left_wrist_yaw_link x68, right_wrist_yaw_link x42, left_knee_link x37, right_elbow_link x8` |
| dorsal torso contact (either robot) | **0 frames**, no trigger for either robot |
| `a` torso tilt / pelvis z during the phase | 105–125° / 0.27–0.46 m (end 117.2° / 0.307 m) |
| defender `b` | standing throughout: tilt 0.15°, pelvis 0.7916 m |
| exchange 0 record | `cause=timeout`, duration exactly **5.0 s**, winner None, ambiguous False, oob 0/0, back_triggers None/None |
| after the draw | standing reset; exchange 1 has no contacts |

**Verdict: while the robot was on its knees/hands with the mat the exchange did not end — no
score, no reset — and it ran its full clock to a draw.** That is the MISSION "knees/hands are
never terminal" claim, demonstrated with the shipped detector's own diagnostics.

**Deviation (stated).** The assignment's "…while it recovers to standing" half is **not
achievable with the shipped controllers** — that is the phase-3 teacher's job (env report
§3.1/§3.5). Measured: no shipped controller brings a robot from knees/hands back to standing —
the STAND_UP replay from the ground start goes dorsal at 3.08 s in the pair configuration
(1.64 s when the robot has nothing to brace against), and `StandHold` from the ground pose
collapses at ~1.5 s. The closest faithful version is what is rendered: the legal knees/hands
phase with the detector diagnostics and the exchange continuing to a draw, plus the standing
reset that the draw triggers. No trajectory was hand-authored to fake a recovery.

---

## 4. `ambiguous_simultaneous` — both backs down inside 0.10 s → no score

**WHAT TO LOOK FOR.** Both wrestlers supine on the mat, both `dorsal=1` with `trig=-` while the
persistence window runs, then both triggers in the *same control step*, the
`EXCHANGE 0 END — cause: back, AMBIGUOUS (both backs inside 0.10 s) — no score — reset` banner,
the score still `0:0`, and the standing reset.

**Setup (state construction, allowed by the assignment).** Both robots are placed supine from the
*settled* SPRAWL end state (the deterministic construction of
`tests/test_wrestling.py::_sprawl_settled_supine`), positioned at `x = ±0.9 m, y = 0` — inside
the declared circle, 1.8 m apart. Translation is exactly dynamics-neutral on the uniform floor.
Exchange 0 uses controllers `(None, None)` (the env holds the constructed joints, exactly like
the delivered test); afterwards `StandHold`.

**Observed (measured).**

| quantity | value |
|---|---|
| dorsal contact from | t = 0.00 s for both robots (tilt 88.78°, pelvis 0.0696 m) |
| trigger `a` / trigger `b` | **0.32 s / 0.32 s** — same control step, gap **0.00 s** ≤ 0.10 s window |
| exchange 0 record | end 0.32 s, `cause=back`, **`ambiguous=True`**, winner None, loser None |
| rewards | `(0.0, 0.0)` — no score for either wrestler |
| oob | `{a:0, b:0}` (both pelvises at r = 0.90 m) |
| after | standing reset; exchange 1 has no contacts; score stays `0:0` |

The contact sheet's middle panel is at **t = 0.28 s** (both supine, `trig=-`, the window about to
complete) so the sheet shows the pre-trigger state; the end panel shows the reset.

**Deviation (stated).** The delivered test translates robot `a` by `+1.8 m` in x from wherever the
replay settled; that settles at **r = 2.87 m**, i.e. outside the declared 1.5 m circle, which
injects spurious OOB events (my first run of this scenario produced `oob a:1`, −0.25 score,
`cause=back/ambiguous` still correct but contaminated). The clip places both bodies at
r = 0.90 m instead, so the ambiguous exchange is uncontaminated. Same outcome, no rule
interaction.

---

## 5. `oob_forfeit` — 3 boundary crossings → forfeit

**WHAT TO LOOK FOR.** The yellow dashed ring (the declared 1.5 m circle), the offender's pelvis
radius crossing it (HUD `rA` goes 0.35 → 1.51 → …), the OOB counter stepping `1/3 → 2/3 → 3/3`
with the score dropping 0.25 each time, and the exchange ending in a forfeit.

**Setup.** Robot `a`'s pelvis is driven across the boundary by a scripted controller (the env has
no locomotion policy at this stage): a time-ramped out/in plan perpendicular to the pair axis
(with holds), root position *and* orientation pinned to the standing start and the pelvis
velocity written consistently with the plan, joint servos holding the standing pose. Robot `b`
stands (`StandHold`). This is the delivered OOB test's mechanism (direct pelvis writes; the rule
reads only the pelvis radius), time-ramped instead of a single-step teleport.

**Observed (measured).**

| event | time | offender pelvis radius |
|---|---|---|
| OOB #1 (inside→outside) | **1.32 s** | 1.51 m |
| OOB #2 | **3.32 s** | 1.55 m |
| OOB #3 | **5.32 s** | 1.48 m at the event step (crossing quantized by the 0.02 s control step at ≈5.8 m/s) |
| re-entries (respected the 5 cm hysteresis) | ≈2.3 s, ≈4.3 s | inside again (r = 0.35 m during the holds) |
| exchange 0 record | end 5.32 s | `cause=oob, winner=b, loser=a, oob_events={a:3, b:0}`, back_triggers None/None |
| rewards | final step `(−1.25, +1.0)` | 3rd OOB penalty (−0.25) + exchange loss (−1) for `a`; cumulative `−1.75 : +1.00` |
| detector | silent | 0 dorsal frames for both robots; `a` end tilt 16.47° / pelvis 0.7935 m, `b` 0.15° / 0.7916 m |
| after | standing reset at 5.32 s; exchange 1 starts clean | — |

**Deviation (stated, and a recorded dead end).** The first version dragged `a` along +x, which
swept it through the standing `b`: both robots toppled, and the exchange ended by a *back* event
(2 OOB events) instead of the forfeit. The lateral path (`y`, away from `b`) fixed it; the
recorded failure is kept here because it shows how easy it is to get a rule interaction wrong
when the offender is scripted.

The contact sheet's middle panel is at **t = 1.60 s** (the offender outside the ring with the
first event counted, `rA = 1.78 m`, `OOB events A 1/3`), so the sheet shows the crossing itself;
the start panel is the standing start at r = 0.35 m and the end panel the post-forfeit reset.

---

## 6. Honest limitations / deviations (summary)

1. **STANCE replay is not holdable** → scenario 1 uses the env's 1.0 s-clock draw recipe; the
   topple is measured and printed by `--diag` (trigger at 1.78 s, `cause=back`) rather than
   hidden.
2. **Reset pose shim** (`EvidenceEnv`): every reset uses the verified-stable `both_stand`
   keyframe instead of the randomized STANCE start (reason in §0). The delivered env is
   untouched; only the clips' start pose differs, and the report says so.
3. **No recovery controller exists** → scenario 3 shows the legal knees/hands phase and the
   exchange continuing to a draw, not a completed recovery to standing.
4. **Scenario 5's drag is scripted root motion**, not a locomotion policy (none exists yet);
   the OOB rule reads only the pelvis radius, which is what is being evidenced.
5. **No shadows/reflections and `offsamples=1`** in the clips (software EGL: 0.41 → 2.6 fps).
   The HUD, the ring and the geometry are unaffected; the simulation model is unaffected.
6. **Camera framing is clamped** to 4.0–5.5 m (4.2–5.0 m with the ring): the declared circle is
   fully visible in the OOB clip, and for the other clips the far ring posts may be cropped —
   the *robots* are guaranteed inside the frame, which is the requirement.
7. **Numbers vs the delivered calibration:** the SPRAWL back event here is 1.38 s → trigger
   1.70 s, the calibration JSON says 1.682 s (one control step earlier). The detector config,
   the env and the scene are the shipped ones; only the sampling instant differs.
8. The topple/collapse events in scenarios 2 and 3 are untaught-replay artefacts (phase-2
   limitation), labelled as such; they are not evidence against the rules, and they should be
   re-rendered once the phase-3 teacher exists (the teacher videos are the A/B pair for that).
9. **The rules changed mid-flight.** While these clips were being produced another agent landed
   the *ambiguity hold* in `src/wrestling/env.py` (04:33): after the first back trigger the
   exchange now stays open up to `AMBIGUITY_WINDOW_S` (0.10 s) so the second trigger of a
   near-simultaneous pair is observed before resolution. Consequences, checked rather than
   assumed: the affected clip (`takedown_back_event`, single trigger) was re-rendered against
   the new code, and its exchange 0 now ends at **1.80 s** (trigger 1.70 s + the hold) instead of
   1.70 s — the trigger time, winner and score are unchanged, and the clip's banner/HUD reflect
   the new timing. The other four scenarios never fire a back trigger, so the new code path is
   the same branch as before for them; all five were re-verified with
   `scripts/render_env_videos.py --check` against the current `src/wrestling/env.py` (all
   `PASS`, records in §1–§5). If `src/wrestling/env.py` changes again, re-run `--check` before
   trusting these numbers; the videos themselves are honest renderings of the code at their
   timestamps.

## 7. Reproduction

```bash
cd /home/ubuntu/grappling
MUJOCO_GL=egl .venv/bin/python scripts/render_env_videos.py            # all 5 clips + sheets
MUJOCO_GL=egl .venv/bin/python scripts/render_env_videos.py --check    # sim + records only
MUJOCO_GL=egl .venv/bin/python scripts/render_env_videos.py --only oob_forfeit
MUJOCO_GL=egl .venv/bin/python scripts/render_env_videos.py --diag     # scenario-1 deviation
```

The script prints, per scenario, the frame count, wall time, score, per-exchange first-contact
times, limb-contact tallies, OOB event times, the full exchange records, the `PASS`/`FAIL`
evidence lines and the output paths, and ends with `SELF-CHECK OK` (assertions on the record
structure per scenario, not just file existence). Outputs are 30 fps, 960x720, h264/yuv420p.

## 8. Facts for the ledger (orchestrator merges into notes.md)

- `videos/env/` holds 5 clips + 5 contact sheets (stage 5), 30 fps 960x720 h264/yuv420p,
  rendered in one process in 413 s wall.
- Measured rule outcomes (current `src/wrestling/env.py`): timeout draw at 1.0 s with the STANCE
  replay (5×, both up at 0.72–0.74 m); SPRAWL back event dorsal 1.38 s → trigger 1.70 s →
  exchange end 1.80 s (trigger + the 0.10 s ambiguity hold) → score; knees/hands limb contact
  121/150 frames with 0 dorsal frames and a full 5.0 s draw; ambiguous double fall with both
  triggers at 0.32 s (gap 0.00 s) and no score; 3 OOB crossings at 1.32/3.32/5.32 s (pelvis
  1.51/1.55/1.48 m) → forfeit, cumulative score −1.75 : +1.00.
- Software EGL on this host: MuJoCo's default shadow pass costs ~6x the frame time
  (0.41 fps → 2.6 fps at 960x720 with both G1s); every render script should disable shadows for
  bulk rendering.
- No shipped controller recovers a robot from knees/hands to standing (teacher territory).
- The env's default reset pose (randomized STANCE start) makes multi-exchange clips collapse
  within ~2 s unless the start pose is overridden (`reset(pose=...)` or a `_randomized_stance`
  shim).
