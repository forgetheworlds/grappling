"""GrappleMap database loader (pure stdlib + numpy).

Usage::

    from grapplemap import load_graph
    g = load_graph()               # third_party/GrappleMap/GrappleMap.txt
    print(g.summary())
    node = g.node_by_name("staredown")
    edge = g.edges_with_tag("double_leg_takedown")[0]
    edge.frames            # (F, 2, 23, 3) float64, Y-up, meters
    edge.mover             # 'top' | 'bottom' | None
"""

from .parser import (
    BASE62,
    DEFAULT_DB_PATH,
    JOINTS,
    JOINT_COUNT,
    MIRROR_PERM,
    PLAYER_COUNT,
    SAME_TOL_SQ,
    Edge,
    Graph,
    Node,
    Reorientation,
    basically_same,
    decode_position,
    encode_position,
    is_reoriented,
    load_graph,
    mirror_position,
    rot_y,
    swap_players,
)

__all__ = [
    "BASE62",
    "DEFAULT_DB_PATH",
    "JOINTS",
    "JOINT_COUNT",
    "MIRROR_PERM",
    "PLAYER_COUNT",
    "SAME_TOL_SQ",
    "Edge",
    "Graph",
    "Node",
    "Reorientation",
    "basically_same",
    "decode_position",
    "encode_position",
    "is_reoriented",
    "load_graph",
    "mirror_position",
    "rot_y",
    "swap_players",
]
