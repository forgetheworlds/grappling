"""The 7-technique reference recipes (docs/CURRICULUM.md) as target tracks.

Role convention: robot A = attacker / initiator / recoverer, robot B =
defender / opponent. Roles are specified as RAW player indices per edge
(GrappleMap edge frames carry their own player indexing; mover metadata is
edge-relative and unreliable across reorientations, so every assignment below
was verified geometrically against the vendored data — see
reports/2026-10-07/retarget.md):

  DOUBLE_LEG t1144→t1147→t1141→t1140 : A = raw p0 (drops level in t1144,
      penetrates in t1147; t1140 'land' mover=top=p0 finishes on top).
  SINGLE_LEG t978→t1001→t1094        : A = raw p0 (drops level in t978).
  BODY_LOCK t1133                    : A = raw p0 (mover=top, flanks+takes down).
  SNAPDOWN t989                      : A = raw p0 (snapper: p1's head drops
      1.35→0.93 m while p0's stays 1.38→1.20 m).
  SPRAWL t1147→t1141 (A = shot player p0) spliced with the sprawl chain
      t381→t382→t383 (B = raw p1, the sprawler: starts standing 1.00 m core,
      ends prone ~0.4 m; the sprawl edges' 'top' mover metadata refers to the
      END node's swapped indexing and contradicts the geometry, hence the
      geometric assignment). Splice: nearest-frame rigid alignment of the
      sprawl chain's shot player onto the attacker chain's shot player.
  STAND_UP t952→t1110→t1125          : A = raw p1 (bottom/recoverer in all
      three clips; movers of t1110/t1125 are bottom=p1), B = raw p0.
  STANCE node 'symmetric staggered standing' (idx 94): A = p0, B = p1.

Chaining: shared-node junctions are exact; the others (sprawl variants,
stand-up montage) use least-squares rigid alignment with an inserted
transition interval (see gmframe.assemble_chain; residuals reported).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gmframe import (CORE, JOINTS, assemble_chain, best_pair_align,
                      single_player_align)
from .landmarks import GM_JOINT_TO_SITE, gm_weights
from .solve import (DT_50HZ, landmark_rms_final, resample_50hz, solve_keyframes)
from .world import place_world

#: per GM joint: the G1 stand site to borrow for the STANCE blend
#(dropped joints -> wrist site)
_GM_STAND_JOINTS = tuple(GM_JOINT_TO_SITE[j] or "left_wrist" for j in JOINTS)
_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        from grapplemap import load_graph
        _GRAPH = load_graph()
    return _GRAPH


STANCE_NODE = "symmetric staggered standing"
STANCE_HOLD = 1.2  # s


@dataclass
class TechniqueTargets:
    """Role-resolved, unscaled, Y-up keyframe targets for one technique."""
    technique: str
    edges: list[int]
    targets_a: np.ndarray   # (F, 23, 3)
    targets_b: np.ndarray
    times: np.ndarray       # (F,)
    info: dict = field(default_factory=dict)


def _floor_pair(tt: TechniqueTargets) -> TechniqueTargets:
    """Keep both players' landmarks at/above the mat (gm y >= 0.02, the
    GrappleMap joint radius / apply_limits semantics).

    Alignment offsets between clips can push a whole keyframe pair below the
    ground; shifting BOTH players of that keyframe up preserves every
    inter-player relationship (pelvis-pelvis vector, attack depth, contact)
    and only corrects the pair-to-ground relation, which must respect
    gm y=0 == mj z=0.
    """
    floor = 0.02
    mins = np.minimum(tt.targets_a[:, :, 1].min(axis=1),
                      tt.targets_b[:, :, 1].min(axis=1))
    shift = np.maximum(floor - mins, 0.0)
    tt.targets_a = tt.targets_a + shift[:, None, None]
    tt.targets_b = tt.targets_b + shift[:, None, None]
    return tt


def build_technique_targets(technique: str) -> TechniqueTargets:
    if technique == "STANCE":
        return _floor_pair(_stance())
    if technique == "SPRAWL":
        return _floor_pair(_sprawl())
    if technique in _SIMPLE_RECIPES:
        edges, role_a, tdt = _SIMPLE_RECIPES[technique]
        return _floor_pair(_simple(technique, edges, role_a, tdt))
    raise KeyError(technique)

def _resolve_roles(chain, role_a_per_edge: list[int]):
    """Per-frame role tracks from a ChainResult + raw roles per edge."""
    f = len(chain.frames)
    role_a_raw = np.array([role_a_per_edge[e] for e in chain.frame_edges])
    swap = chain.world_swap.astype(np.int64)
    idx_a = role_a_raw ^ swap
    idx_b = (1 - role_a_raw) ^ swap
    rows = np.arange(f)
    return chain.frames[rows, idx_a], chain.frames[rows, idx_b]


def _simple(technique: str, edge_ids: list[int], role_a: list[int],
            transition_dt: float = 0.5) -> TechniqueTargets:
    g = get_graph()
    chain = assemble_chain(g, edge_ids, transition_dt=transition_dt,
                           w=gm_weights("default"))
    ta, tb = _resolve_roles(chain, role_a)
    return TechniqueTargets(technique, list(edge_ids), ta, tb, chain.times,
                            {"junctions": chain.junctions,
                             "world_swap": chain.world_swap.tolist()})


def _stance() -> TechniqueTargets:
    """STANCE baseline: the 'symmetric staggered standing' node relationships
    (facing, separation, stagger, arm carriage) blended 65/35 with the G1's
    own verified-stable 'stand' pose.

    The literal scaled node pose leans the torso ~30 deg forward, which puts
    the ankle balance torque of the position-servo G1 at its 50 Nm limit and
    the robot topples forward within 1.5 s (verified hold test). The blend
    keeps every pair/intra-player relationship while restoring static
    holdability; STANCE is explicitly a procedural baseline (CURRICULUM.md),
    not an edge-chain retarget.
    """
    from .landmarks import SOLVED_SITES, g1_reference_geometry
    from .world import estimate_player_scale
    g = get_graph()
    node = g.node_by_name(STANCE_NODE)
    pose = np.asarray(node.position, dtype=np.float64)  # (2, 23, 3) Y-up

    s = 0.5 * (estimate_player_scale(pose[0:1])
               + estimate_player_scale(pose[1:2]))
    geo = g1_reference_geometry()
    sites = geo["sites"]
    # G1 stand landmarks in GM human space (mj (x,y,z) -> gm (x, z, -y)), /s
    stand_gm = {n: np.array([p[0], p[2], -p[1]]) / s for n, p in sites.items()}
    core = stand_gm["core"].copy()
    stand_track = np.stack([  # (23,3) per player, Fingers->wrist
        [stand_gm.get(j, stand_gm["left_wrist"]) for j in _GM_STAND_JOINTS]
        for _ in range(2)])
    stand_track = stand_track - core

    # each player's stand pose faces the other player's core
    blend = np.empty_like(pose)
    for p in range(2):
        d = pose[1 - p, CORE] - pose[p, CORE]
        yaw = np.arctan2(d[0], d[2])  # gm heading: atan2(x, z) (parser: angle(V2))
        ca, sa = np.cos(yaw), np.sin(yaw)
        rot = stand_track[p].copy()
        rot[:, 0] = ca * stand_track[p][:, 0] + sa * stand_track[p][:, 2]
        rot[:, 2] = -sa * stand_track[p][:, 0] + ca * stand_track[p][:, 2]
        blend[p] = 0.65 * pose[p] + 0.35 * (rot + pose[p, CORE])

    frames = np.stack([blend, blend])  # hold
    times = np.array([0.0, STANCE_HOLD])
    return TechniqueTargets("STANCE", [], frames[:, 0], frames[:, 1], times,
                            {"node": node.index, "node_name": node.display_name,
                             "hold_s": STANCE_HOLD,
                             "stand_blend": 0.65})


def _sprawl() -> TechniqueTargets:
    g = get_graph()
    # attacker shot chain: 'shoot for high double' -> 'drive forward'
    attack = assemble_chain(g, [1147, 1141], w=gm_weights("default"))
    # sanity: shot player (raw p0) is the low core at the chain start
    a0 = attack.frames[0]
    assert a0[0, CORE, 1] < a0[1, CORE, 1] - 0.1, "raw p0 expected to be the shooter"

    # defender sprawl chain: three sprawl variants, alignment-chained
    sprawl = assemble_chain(g, [381, 382, 383], transition_dt=0.35,
                            w=gm_weights("default"))
    s0 = sprawl.frames[0]
    assert s0[0, CORE, 1] < s0[1, CORE, 1] - 0.1, "raw p0 expected to be the shot player"

    # nearest-frame geometric alignment of the shot player onto the shot frames
    splice_idx, splice_t, splice_rms = single_player_align(
        s0, 0, attack.frames, 0, w=gm_weights("default"))
    splice_dt = 0.3  # impact splice interval
    t0 = float(attack.times[splice_idx]) + splice_dt
    part1_f, part1_t = attack.frames[:splice_idx + 1], attack.times[:splice_idx + 1]
    part2_f = splice_t.apply(sprawl.frames)
    part2_t = sprawl.times + t0
    frames = np.concatenate([part1_f, part2_f], axis=0)
    times = np.concatenate([part1_t, part2_t])

    # roles: part 1 from the attack chain (A=raw p0, B=raw p1; swap False);
    #        part 2 from the sprawl chain (A=shooter raw p0, B=sprawler raw p1)
    n1 = len(part1_f)
    swap = np.concatenate([attack.world_swap[:n1], sprawl.world_swap]).astype(np.int64)
    role_a_raw = np.concatenate([np.zeros(n1, dtype=np.int64),
                                 np.zeros(len(part2_f), dtype=np.int64)])
    rows = np.arange(len(frames))
    ta = frames[rows, role_a_raw ^ swap]
    tb = frames[rows, (1 - role_a_raw) ^ swap]

    info = {
        "junctions": attack.junctions + sprawl.junctions,
        "splice": {"attack_edge": 1147, "attack_frame_index": int(splice_idx),
                   "attack_frame_time": float(attack.times[splice_idx]),
                   "sprawl_edge": 381, "alignment_rms": float(splice_rms),
                   "splice_dt": splice_dt},
        "world_swap": swap.tolist(),
    }
    return TechniqueTargets("SPRAWL", [1147, 1141, 381, 382, 383],
                            ta, tb, times, info)


#: raw role-A assignment per edge for the simple chains
_SIMPLE_RECIPES = {
    "DOUBLE_LEG": ([1144, 1147, 1141, 1140], [0, 0, 0, 0], 0.5),
    "SINGLE_LEG": ([978, 1001, 1094], [0, 0, 0], 0.5),
    "BODY_LOCK": ([1133], [0], 0.5),
    "SNAPDOWN": ([989], [0], 0.5),
    "STAND_UP": ([952, 1110, 1125], [1, 1, 1], 0.6),
}


TECHNIQUES: tuple[str, ...] = (
    "DOUBLE_LEG", "SINGLE_LEG", "BODY_LOCK", "SNAPDOWN", "SPRAWL",
    "STAND_UP", "STANCE",
)


def _decollide_pair(placement) -> None:
    """Separate the two players' landmark targets when their cores are closer
    than two robot pelvis/torso shells (~0.30 m).

    Tight GrappleMap clinch geometry (body lock, sprawl chest pressure) is
    schematic: the drawn cores sit closer than two G1 torso meshes can
    physically occupy (measured 4-5 cm mesh interpenetration under PD
    tracking). Each player shifts AWAY along the core-core axis by half the
    deficit (both players move, so every relative relationship — direction,
    lean, contact posture — is preserved; only the attack depth is opened to
    the collision-feasible minimum). In-place on WorldPlacement.
    """
    ta, tb = placement.targets_a, placement.targets_b
    d = ta[:, CORE] - tb[:, CORE]
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    deficit = np.maximum(0.30 - dist, 0.0)
    push = 0.5 * deficit * d / np.maximum(dist, 1e-9)
    ta += push[:, None, :]
    tb -= push[:, None, :]


def build_technique(technique: str, preset: str = "default",
                    time_stretch: float = 1.0) -> dict:
    """Full retarget for one technique: targets -> world -> solve -> 50 Hz.

    Returns dict with qpos_a/qpos_b (T, 36), t (T,), landmark rms dict, meta.
    ``time_stretch`` multiplies all keyframe intervals (repair loop retimer;
    the kinematic-limit retimer inside resample_50hz may stretch further).
    """
    tt = build_technique_targets(technique)
    placement = place_world(tt.targets_a, tt.targets_b)
    _decollide_pair(placement)
    solved = solve_keyframes(placement.targets_a, placement.targets_b, preset,
                             t_kf=tt.times * time_stretch)
    t_kf = tt.times * time_stretch
    t_grid, qa, qb, stretch = resample_50hz(
        t_kf, solved["qpos_a"], solved["qpos_b"])
    rms = landmark_rms_final(qa, qb, placement.targets_a, placement.targets_b,
                             t_kf * stretch, t_grid, preset)
    meta = {
        "preset": preset,
        "roles": "A=attacker/recoverer, B=defender/opponent",
        "scale_a": placement.scale_a, "scale_b": placement.scale_b,
        "scale": placement.scale,
        "heading": placement.heading,
        "g1_segments_m": placement.g1_segments,
        "junctions": tt.info.get("junctions", []),
        "splice": tt.info.get("splice"),
        "node": tt.info.get("node"),
        "keyframes": int(len(t_kf)),
        "time_stretch_requested": time_stretch,
        "time_stretch_kinematic": float(stretch),
        "duration_s": float(t_grid[-1]),
        "dt": DT_50HZ,
        "source": "third_party/GrappleMap @ 032c8f9 (see notes.md)",
        "repairs": [],
    }
    return {
        "technique": technique,
        "edges": list(tt.edges),
        "qpos_a": qa, "qpos_b": qb, "t": t_grid,
        "landmark_rms": rms["weighted"],
        "landmark_rms_detail": rms,
        "meta": meta,
    }
