# QUALITY_RUBRIC.md — what "looks proper and clean" means, checkably

Operator: **quality is the main outcome.** This rubric is the acceptance gate for every motion
artifact in the solo-drill milestone. It is deliberately mechanical so an agent can self-assess and
the operator can verify from the video. Every score ≥ 2 needs frame evidence and a number.

## Scoring
Per element: **0 = fail, 1 = recognisable but wrong, 2 = good, 3 = clean → matches a competent human.**
Ship gate: **no element below 2**, and the whole sequence must read as wrestling to a frame-level
reviewer. A clip that needs a reset or contains a fall is not a ship candidate, regardless of scores.

## A. Stance (held, both feet planted)
| # | criterion | how to check |
|---|---|---|
| A1 | Foot separation = spec width (operator: wide enough to be laterally stable) | measure lateral foot distance in m; report vs spec range |
| A2 | Base depth: rear foot behind the lead, longer than shoulder width (operator: leg a bit back) | front-to-back foot distance, m |
| A3 | Knee bend carries the crouch (not waist folding) | knee flexion vs hip/spine flexion proxies; report both |
| A4 | Torso pitch within spec (leaning, not bent over) | torso angle vs gravity, deg |
| A5 | Head up, eyes forward | head height + pitch vs pelvis |
| A6 | Hands carriage: forward, inside, elbows in, not dangling | hand position in the self frame vs spec |
| A7 | CoM inside the support polygon with ≥ 2 cm margin | CoM vs convex hull of contact sites; report the margin |
| A8 | Stable: 20 s hold, no fall, drift small | pelvis xy drift, tilt, foot contacts |

## B. Footwork (shuffle / circle / retreat / approach)
| # | criterion | how to check |
|---|---|---|
| B1 | Feet lift and place — never slide while loaded | per-step slide distance of the loaded foot (≤ 2 cm) |
| B2 | Weight transfers before the lift (support foot loaded) | load proxy before lift; no double-float frames |
| B3 | Command fidelity: actual displacement direction ≈ commanded | body displacement vs command, m and deg |
| B4 | Cadence plausible for wrestling (quick, short steps) | steps/s, step length, m |
| B5 | Feet do not cross or collide | min inter-foot distance, self-collision count |
| B6 | Posture preserved while moving (A1-A6 hold within tolerance) | stance metrics sampled during motion |

## C. Level change
| # | criterion | how to check |
|---|---|---|
| C1 | Pelvis drop depth matches the reference band | pelvis z trace, min/max |
| C2 | Drop is produced by legs (knees), not by folding at the waist | joint-space comparison vs A3 |
| C3 | CoM stays inside support through the drop | margin trace, minimum value |
| C4 | Speed is controlled (no dropping, no bouncing) | pelvis z velocity/accel limits |
| C5 | Reversible: return to stance cleanly | time to re-establish A1-A7 |

## D. Penetration step (the shot gesture)
| # | criterion | how to check |
|---|---|---|
| D1 | Lead foot steps forward into the entry (travel ≈ reference) | foot displacement, m |
| D2 | Lead knee lowers toward the mat in a controlled way | knee height trace, min height, descent rate |
| D3 | Trail leg drives back under the body (support regained under CoM) | trail foot reposition + CoM-over-support at the end of the drive |
| D4 | Torso/head alignment preserved (head up, chest over the lead leg) | torso pitch + head vs pelvis at entry |
| D5 | Arms drive forward/inside plausibly (no simulated grip claims) | hand trajectory in the self frame; mark as gesture only |
| D6 | Entry depth sensible relative to body height | forward CoM advance, m/height |
| D7 | Recoverable: rise back to stance without fall | D5 of section E |

## E. Recovery / rise
| # | criterion | how to check |
|---|---|---|
| E1 | Returns to full stance (A1-A7 within tolerance) | stance metric deltas after recovery |
| E2 | No fall, no hand-posted dependence (unless intentional) | dorsal contact = 0, hand-contact frames counted |
| E3 | Recovery time within spec | seconds, reported |
| E4 | Stability restored (margin back to A7 band) | CoM margin after 1 s |

## F. Continuity (the whole run)
| # | criterion | how to check |
|---|---|---|
| F1 | No resets inside a successful run | sim clock monotonic; reset counter = 0 |
| F2 | No stalls (pelvis velocity ~0 for long stretches while a motion is commanded) | zero-velocity fraction |
| F3 | Smooth: bounded action-rate/jerk, no oscillation | action delta + jerk percentiles |
| F4 | Transitions between elements connect (no teleporting joints, no phase snap) | max per-step joint delta at boundaries |
| F5 | Repeats: the drill cycles several times without degradation | per-cycle metric stability |

## G. Physical plausibility
| # | criterion | how to check |
|---|---|---|
| G1 | No mesh interpenetration > 2 cm (self or ground) | penetration metrics per frame |
| G2 | No ground penetration > 1 cm | min site z over the run |
| G3 | No foot sliding while loaded (see B1) | same measurement |
| G4 | Actuator saturation rare | fraction of steps at/near torque or ctrl limits |
| G5 | No visible glitching (mesh pops, jitter) | frame inspection + joint-acceleration outliers |

## H. Visual match to the reference
| # | criterion | how to check |
|---|---|---|
| H1 | Stance, level change, shot and knee-sprawl read as the same movements as the operator's reference | side-by-side stills at matching phases; state per element MATCHES / APPROXIMATES / DIFFERS |
| H2 | Differences are explainable by morphology (G1 is 1.32 m, no fingers) rather than by error | note the reason for each difference |
| H3 | Slow-motion review (0.25×) of level change, entry, knee contact and rise shows no artefacts | render 0.25× clips for those four moments |

## Evidence duties
- Ship candidate: MP4 (960×720, 30 fps, h264/yuv420p) + HUD overlay (skill, command, rung, phase,
  key metrics, push markers) + metrics JSON + 3-frame contact sheet.
- Per element claiming score ≥ 2: at least one annotated frame and the numeric trace for the
  criteria above. Claims without evidence score 0 by default.
- Failures are kept and labelled; a better clip never replaces a failure in the record.
- The report states explicitly which parts are controller/scripted vs learned, and the rubric table
  with scores + evidence pointers.
