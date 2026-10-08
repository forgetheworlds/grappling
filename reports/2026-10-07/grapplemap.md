# GrappleMap vendoring + parser — verified data-format facts

**Agent:** GrappleMapParser (wave 1) · **Date:** 2026-10-07
**Deliverables:** `third_party/GrappleMap` (vendored, untouched), `src/grapplemap/`
(parser), `tests/test_grapplemap.py` (17 tests, green), `data/grapplemap/vocab_candidates.json`,
this report.

Vendored commit: `032c8f91809786b7b784852abd07cf3e10c0ea35` (2020-03-14, `git clone --depth 1`).
All facts below were verified by reading the C++/JS sources of that commit and by
parsing the data; file/line references given. Nothing here is assumed.

## 1. Where the data lives

The entire database is **one plain-text file**: `third_party/GrappleMap/GrappleMap.txt`
(39,098 lines, 2.7 MB). Everything else in the repo is tooling (C++ editor/renderers,
website JS, Blender scripts, drill definitions).

File structure (= `readSeqs()` in `src/persistence.cpp:58-116`):

- Sequence of **blocks**. A block = one or more description lines followed by ≥ 1
  position records.
- A **position record** is 4 text lines, each `4 spaces + 69 base62 chars`
  (`encoded_pos_size = 2*joint_count*3*2 + 4*5 = 296` bytes incl. newlines,
  persistence.cpp:31).
- A line starting with a space is a position line; any other line is description.
- **Block with exactly 1 position = graph node** (named position). **Block with ≥ 2
  positions = transition (edge)** whose keyframes run from its first to its last
  position (persistence.cpp:154-168, graph.cpp:279-302).
- Description line 1 is the entity name (literal `\n` escapes = intended line
  breaks). Later lines: `tags:` (categories), `properties:` (behavior flags),
  `ref:` (source instructional citation), `todo:`.

Parsed totals (verified against an independent line-scan census, test
`test_counts_match_raw_file`):

| quantity | value |
|---|---|
| named nodes (1-position blocks) | **601** |
| transitions (multi-position blocks) | **1,485** |
| total stored positions | **8,323** (601 node poses + 7,722 keyframes) |
| graph nodes after linking | **725** (601 named + 124 unnamed auto-added) |
| distinct tags | 161 |

The 124 extra nodes are created exactly as `Graph::find_or_add` (graph.cpp:235) does:
when a transition endpoint matches no existing node under reorientation, a new
unnamed node is appended.

## 2. Pose representation — the "23 landmarks per player" claim, verified

`src/players.hpp:8-29` defines the joint enum (macro `JOINTS`). Counted directly:

- 20 paired joints: (Toe, Heel, Ankle, Knee, Hip, Shoulder, Elbow, Wrist, Hand,
  Fingers) × left/right
- 3 midline joints: **Core, Neck, Head**

**= 23 joints per player. The claim "23 3D landmarks per player" is CORRECT.**
`joint_count = 23` (players.hpp:29). A pose (`Position = PerPlayerJoint<V3>`,
positions.hpp:9) is **2 players × 23 joints × 3 coords = 138 doubles**, stored in
the file player-major, joint-major (enum order), x/y/z interleaved
(`playerJoints` = players.hpp:112, make_playerJoints positions.cpp:65).

Players: index 0 and 1; `playerCode()` = `'t'`/`'b'` (players.hpp:36); rendering
colors red/blue (positions.cpp:140). There is **no absolute "top/bottom" frame** —
"top"/"bottom" is metadata on transitions (below) and tags on positions.

Stick-figure limb lengths (src/gm.js segments): thigh/shank 0.43 m, torso
Core→Shoulder 0.37 m, etc. — a ~1.7 m humanoid, confirming meters.

## 3. Encoding, coordinate frame, units, ground plane

Base62 codec (persistence.cpp:13-56, 118-144):

- alphabet `abc…zABC…Z012…9` (lowercase first);
- each coordinate = 2 digits, value `(d0·62 + d1)/1000` (millimeter quantization);
- **stored x and z are shifted by −2** (`decodePosition`: `{g()-2, g(), g()-2}`);
  y is not shifted.

Therefore decoded ranges: x,z ∈ [−2, 1.843], y ∈ [0, 3.843].

**Frame (verified, not assumed):**

- **Y is up. Ground plane = y = 0 (the XZ plane).** Evidence:
  - y is the only unshifted coordinate → never negative (min over all 8,323
    poses = 0.020 m ≈ toe joint radius 0.025);
  - `apply_limits` (positions.cpp:314-322) clamps `p.y ≥ joint radius` ("keep
    joints above the mat") and x,z into [−2, 2];
  - camera height uses `.y` (positions.hpp:182-187); `heading()` uses `xz()`
    (positions.hpp:132); rotations that preserve stance are `yrot` (about Y).
- **Units = meters** (SI): limb lengths above; head-to-toe span of standing
  nodes measured 1.2–2.2 m (test `test_human_scale_meters`); max y = 1.903 m.
- Rotation convention: `yrot(a)` gives x' = cos·x + sin·z, z' = −sin·x + cos·z;
  `angle(V2)` = atan2(x, z).

## 4. Transitions: sparse keyframes, timing, metadata

- An edge's stored frames are **sparse, manually-authored keyframes**; the
  animations are interpolated linearly between them (`between()` positions.hpp:42).
- **No timestamps exist anywhere in the format.** Playback (playback.cpp:129-131)
  runs at **5 keyframe-intervals per second** ("12 frames per segment at 60 fps"),
  or **10/s for transitions with the `detailed` property** ⇒ nominal 0.2 s
  (0.1 s detailed) between consecutive stored frames. README: joints limited to
  5 direction changes/sec; timing of techniques "not really captured at all".
- Edge endpoints connect to nodes **up to reorientation**: the endpoint matches
  a node if some rotation-about-Y + translation, optionally followed by mirror
  and/or player swap, maps one onto the other within per-joint squared distance
  ≤ 0.0016 (~4 cm) — `is_reoriented` (positions.cpp:198-259), `basicallySame`
  (positions.hpp:49-54), head-to-head distance pre-filter ≤ 0.05.
- **Tags** (`tags:` lines, both nodes and edges; 161 distinct): domains
  (side_control, deep_half, …), pose details (bottom_supine, top_posture_broken…),
  grips/controls (crossface, kimura, lockdown…). Wrestling-relevant:
  `standing`, `double_leg_takedown`, `single_leg_takedown`, `body_lock`,
  `sprawl`, `stand_up`, `technical_standup`, `collar_tie`, `ankle_pick`.
- **Properties** (`properties:` lines, edges only; metadata.hpp:44-45):
  - `top` / `bottom` — **which player performs the move** (`is_top_move`/
    `is_bottom_move`): top = player 0, bottom = player 1 *of the edge's own
    stored frames* (mover is relative, since edges may attach to nodes under a
    player-swap reorientation). Coverage: 616 top / 558 bottom / 311 unmarked.
  - `detailed` — 2× playback speed (10 intervals/s);
  - `bidirectional` — transition is also traversable in reverse (38 edges).
- `ref:` lines cite instructional sources (Ryan Hall, Eddie Bravo, …).

## 5. Mirror and player-swap — exact semantics (both defined in source)

- **`mirror`** (positions.cpp:130-136): for every joint of both players negate
  **x** (`mirror(V3) = {−x, y, z}`), then swap left↔right limb joints within
  each player (`swapLimbs`: Shoulder, Hip, Hand, Wrist, Elbow, Fingers, Ankle,
  Toe, Heel, Knee). Involution: mirror(mirror(p)) = p. (Tested exact.)
- **`swap_players`** (positions.hpp:68-71): `std::swap(p[player0], p[player1])`
  — exchange the two 23-joint blocks. Involution. Commutes with mirror. (Tested.)
- Composition order in `PositionReorientation::operator()` (positions.hpp:82-88):
  rotation+translation → mirror → swap. The parser's `Reorientation.apply`
  reproduces this; `inverse`/`compose` also exist in source (positions.cpp:97-128).

## 6. Parser (src/grapplemap/)

`load_graph()` (stdlib + numpy only) → `Graph{nodes: list[Node], edges: list[Edge]}`:

- `Node`: index, name, description lines, tags, `position (2,23,3) float64`
  (Y-up meters), line_nr, in/out edge indices.
- `Edge`: name, tags, properties, `frames (F,2,23,3) float64`, from/to node
  indices + the `Reorientation` mapping node pose → edge endpoint frame,
  `mover` ('top'/'bottom'/None), bidirectional, detailed, line_nr.
- Linking replicates `Graph::Graph` + `find_or_add` (first-match node order,
  head-to-head pre-filter, 4 reorientation variants), including creation of the
  124 unnamed nodes. Batched numpy matcher, loads in ~3 s on the ARM box.
- Self-verifies on load: every edge endpoint re-apply matches its node
  (C++ `Edge` invariants, graph.hpp:57-58).
- Module docstring documents every fact above with source citations.

**Usage:** `python -c "from grapplemap import load_graph; print(load_graph().summary())"`
→ `GrappleMap: 725 nodes (601 named, 124 auto-added), 1485 edges`.

Tests (`python -m pytest tests/ -x -q`, 17 passed): raw-file count census,
pose shapes/dtype, landmark count, Y-up/ground bounds, meter scale,
**decode∘encode roundtrip on 400 random records of all 8,323**, ≥2 frames and no
identity edges, no identical consecutive keyframes (frame ordering is the only
"time"), endpoint/reorientation invariants, adjacency consistency, mover
metadata, mirror/swap involutions + commutation + matcher detection, vocab JSON
integrity vs. graph.

**Bug worth recording:** an initial `swap_players` implementation reversed the
joint axis instead of the player axis (numpy `...-2` vs `-3`); involution tests
masked it but graph linking degraded (944 nodes, sprawls landing on anonymous
nodes). Fixed; correct linking = 725 nodes with sprawl/single-leg chains landing
on named sprawl/turtle nodes. Lesson: structure tests must validate *semantic*
targets (named endpoints), not just algebraic roundtrips.

## 7. Proposed technique vocabulary (5–8 techniques)

Full inventory with per-edge details: `data/grapplemap/vocab_candidates.json`
(7 groups, 58 edges). Chain notation: `node --tNNN--> node`.

| # | technique | GrappleMap edges (exact) | role |
|---|---|---|---|
| 1 | **Double-leg attack** (level change → penetration → finish) | symmetric staggered standing --t1144 `drop level slightly`--> level dropped --t1147 `shoot for high double`--> high double leg --t1141 `drive forward`--> low flying double leg --t1140 `land`--> finish; alt finish t437 `takedown` (low double leg → double leg takedown finish) | attack |
| 2 | **Single-leg attack** | south-paw neutral standing --t978 `shoot for high single`--> shooting for single --t1001 `head to chest`--> high single w/ head on chest --t1094 `finish`-->; head-outside variant t1003 + t1000 `drive and dump` | attack |
| 3 | **Body-lock takedown** | t1133 `flank and takedown` (standing body lock vs kimura → standing in half w/ hip pinned, mover=top, 10f); t1150 `throw` (sacrifice) ; entry node p80 `standing w/ body lock`, t68 `flank` | attack |
| 4 | **Sprawl defense** | t381/t382/t383 `sprawl` (from low double leg → sprawled positions; t382 adds head&arm); t413 (vs knees double); t327 | defense |
| 5 | **Sprawl→single reattack** (failed-shot punishment / whizzer→single) | t895 `sprawl to single` (low double leg → sprawl vs turtle w/ single, detailed) | defense→offense |
| 6 | **Stand-up recovery** | t952 `stand-up` (Position 629 → south-paw neutral standing); t1110 `bottom stands up`; t446 `stand up and turn in` (single-leg from knees → high single); t1125 `bottom gets to knees` (technical_standup); t878 `get up` | recovery |
| 7 | **Stance/engagement entries** | t838 `begin` (staredown → symmetric staggered standing); t447 `switch stance`; t587/t335 collar-tie entries; t451 `clinch` (over+under) | adjacent |
| 8 | **Snapdown (front headlock)** | t989/t996 `snapdown` (standing collar+tricep → snapped down); t385 snapdown → sprawl w/ head&arm | adjacent (reshot setup) |

Curation criterion applied: standing→standing or standing→ground transitions
serving takedown offense/defense/recovery. **No explicit "reshot"/"reattack"/
"penetration step" edges exist** — nearest material is the snapdown group and
t895; "hips-back" defense is not modeled as such (sprawl nodes are the closest
defensive geometry).

## 8. Caveats for downstream use (phases 2+)

- **Schematic timing:** keyframes are uniform-interval textbook shapes; README
  states technique timing is "not really captured at all". Never replay
  keyframes literally (goal.md: "an edge means start-relationship → key
  geometry → end-relationship"). Retarget to relationship space, not joint
  angles at fixed times.
- Millimeter quantization + 4 cm matching tolerance: treat geometry as
  approximate; the `spring()` relaxation (positions.cpp:142) exists because
  stored poses can violate limb-length constraints.
- Mover is relative to each edge's own player indexing; when chaining edges
  with different reorientations, track player identity through
  `Reorientation.swap_players`.
- y is ground-projected height only up to joint radius (≥0.02); no ground
  contact/force semantics in the data.
- Frames are few (2–20 per transition; 7,722 total) — augmentation will be
  needed for imitation learning, but that is phase-3 concern.

## 9. Interface contract (for notes.md)

```
GrappleMap pose: np.float64 (2, 23, 3), axes [player, joint, xyz]
  joint order: JOINTS in src/grapplemap/parser.py (= players.hpp enum)
  frame: Y-up, ground y=0, meters; x,z ∈ [-2, 1.843]
Edge frames: (F, 2, 23, 3); no timestamps; 0.2 s/keyframe nominal (0.1 detailed)
node_pos == apply(Reorientation, edge_endpoint) up to 4 cm/joint
mirror: x→−x then L↔R limb swap (involution); swap_players: exchange blocks
mover: 'top'|'bottom'|None == player 0|1|unmarked (edge-relative)
```
