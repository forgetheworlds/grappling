# CURRICULUM.md — Selected Wrestling Curriculum (deliverable 4)

Decided 2026-10-07 from `data/grapplemap/vocab_candidates.json` + parser report.
Edge IDs are `g.edges[N]` indices into the vendored GrappleMap (see notes.md).

## Technique vocabulary (7 conditioned techniques)

| technique | GrappleMap edges | role | notes |
|---|---|---|---|
| STANCE | node geometry: 'symmetric staggered standing' | base | procedural baseline pose, not an edge chain |
| DOUBLE_LEG | t1144 → t1147 → t1141 → t1140 (alt finish t437) | attack | level change → penetration → drive → land |
| SINGLE_LEG | t978 → t1001 → t1094 (alt t1003+t1000) | attack | head-on-chest finish; head-outside later |
| BODY_LOCK | t1133 (alt t1150) | attack | flank + takedown from standing body lock |
| SNAPDOWN | t989 (adjacent t996, t385) | offense/counter | head control; sets up shots + re-engagement |
| SPRAWL | t381, t382, t383 (alt t413, t327) | defense | vs double leg; defender legitimately ends prone |
| STAND_UP | t952, t1110, t1125 (+t446 from knees) | recovery | stand-up + technical standup |

Deferred: t895 sprawl→single reattack (phase-5 chaining), collar-tie/clinch
entries t587/t451 (phase 5+). Locomotion (approach/retreat/circle) is built
procedurally in the environment, not from GrappleMap.

## Progression (MISSION phase 4, PISTY-style)

| stage | command | opponent | question |
|---|---|---|---|
| A imitation | e.g. DOUBLE_LEG | cooperative/static | can the movement be reproduced? |
| B randomized drilling | fixed technique | randomized distance/angle/stance/timing | can it adapt? |
| C dynamic drilling | fixed | physically simulated responsive controller | does it work on a moving body? |
| D resistance | fixed | backs up / widens / hips / partial sprawl | does it succeed vs resistance? |
| E attack vs defense | attacker: DOUBLE_LEG | defender: SPRAWL (learned) | competitive co-training |

## Reward evolution

- Early: technique similarity + progress + outcome.
- Middle: reduced similarity weight + strong outcome.
- Late: win/loss of exchange + minimal anti-exploit shaping.
- Always: back-to-mat = exchange loss; knees/hands/sprawl never terminal.

## Game rules (phase 3+, details in MISSION.md)

- Exchange: standing start → free wrestling → back-to-mat (persistent dorsal
  contact + torso orientation) → score → standing reset.
- Anti-stall: bounded area, exchange clock, light engagement pressure.
- Simultaneous falls: first-event wins; ambiguous → no score + reset, logged.

## Evaluation axes

Movement (standing time, stability, recovery success, accidental falls) ·
Attacking (attempts, by type, takedowns, failed-shot recovery rate) ·
Defense (sprawl attempts, successes, recoveries) · Tactics (engagement
distance, retreat frequency, diversity, timing, re-engagement) ·
Competition (exchange win rate, match win rate, vs historical pool).
