"""Parser for the GrappleMap database (``third_party/GrappleMap/GrappleMap.txt``).

Verified data-format FACTS (sources: src/persistence.cpp, src/players.hpp,
src/positions.{hpp,cpp}, src/graph.{hpp,cpp}, src/metadata.{hpp,cpp} of the
vendored repo, commit 032c8f91809786b7b784852abd07cf3e10c0ea35):

- The whole database is one plain-text file: an alternating series of blocks,
  each block = description lines followed by >= 1 position records.
- A block with exactly 1 position is a graph node (named position); a block
  with >= 2 positions is a transition (edge) whose first/last frames connect
  nodes (possibly under reorientation; see below).
- A position record is 4 text lines, each 4 spaces + 69 base62 chars
  (276 payload chars total). Each coordinate is 2 base62 digits decoded as
  ``(d0*62 + d1)/1000``; stored x and z are shifted by -2, y is not.
  => x, z in [-2, 1.843], y in [0, 3.843]. Values are meters.
- Coordinate frame: Y is up, ground plane is y = 0 (the XZ plane).
  Evidence: persistence stores y unshifted (never negative); positions.cpp
  ``apply_limits`` clamps ``p.y >= joint radius`` and x,z into [-2, 2];
  playback/camera treat y as height.
- Pose = 2 players x 23 joints x 3 coords. The 23 joints (players.hpp JOINTS
  macro) are 20 left/right pairs (Toe, Heel, Ankle, Knee, Hip, Shoulder,
  Elbow, Wrist, Hand, Fingers) plus 3 midline joints (Core, Neck, Head).
  Player 0 is coded 't' ("top"), player 1 'b' ("bottom") in playerCode().
- File order inside a record is player-major, joint-major (enum order),
  coordinates x, y, z.
- Description line 1 of a block is the entity name (literal "\\n" escapes
  mark intended line breaks). Other lines: ``tags:`` (categories),
  ``properties:`` (top/bottom = which player performs the move; detailed;
  bidirectional), ``ref:`` (source instructional), ``todo:``.
- NO timestamps exist in the file. Keyframes are uniform-interval; playback
  speed (playback.cpp) is 5 keyframe-intervals ("segments") per second, or
  10/s for sequences with the "detailed" property => 0.2 s (0.1 s detailed)
  between consecutive stored frames.
- Graph construction (graph.cpp Graph::Graph): named single-position blocks
  become nodes in file order; each transition's first/last frame is matched
  to the FIRST existing node it is "reoriented-equal" to (rotation about Y +
  translation, optionally followed by mirror and/or player swap; per-joint
  squared distance <= 0.0016, i.e. ~4 cm). If nothing matches, a new unnamed
  node is appended. Hence the parsed graph can have MORE nodes than the file
  has named position blocks.

Mirror / player-swap semantics (positions.cpp):
- ``mirror``: negate x of every joint, then swap left<->right limb joints
  within each player. Involution: mirror(mirror(p)) == p.
- ``swap_players``: exchange the two 23-joint player blocks. Involution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Constants

#: Joint names in file/enum order (players.hpp JOINTS macro), 23 entries.
JOINTS: tuple[str, ...] = (
    "LeftToe", "RightToe",
    "LeftHeel", "RightHeel",
    "LeftAnkle", "RightAnkle",
    "LeftKnee", "RightKnee",
    "LeftHip", "RightHip",
    "LeftShoulder", "RightShoulder",
    "LeftElbow", "RightElbow",
    "LeftWrist", "RightWrist",
    "LeftHand", "RightHand",
    "LeftFingers", "RightFingers",
    "Core", "Neck", "Head",
)
JOINT_COUNT = len(JOINTS)  # 23
PLAYER_COUNT = 2
_HEAD = JOINTS.index("Head")

#: base62 alphabet used by persistence.cpp (case-sensitive, lowercase first).
BASE62 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_BASE62_INDEX = {c: i for i, c in enumerate(BASE62)}

_POS_CHARS = PLAYER_COUNT * JOINT_COUNT * 3 * 2  # 276 (2 base62 digits per coordinate)
_POS_LINES = 4

#: Per-joint squared-distance tolerance of C++ basicallySame().
SAME_TOL_SQ = 0.0016

#: Default vendored database location (repo root / third_party/...).
DEFAULT_DB_PATH = (
    Path(__file__).resolve().parents[2] / "third_party" / "GrappleMap" / "GrappleMap.txt"
)

# left<->right joint permutation used by mirror (positions.cpp swapLimbs)
_MIRROR_PERM = list(range(JOINT_COUNT))
for _l, _r in (
    ("LeftShoulder", "RightShoulder"), ("LeftHip", "RightHip"),
    ("LeftHand", "RightHand"), ("LeftWrist", "RightWrist"),
    ("LeftElbow", "RightElbow"), ("LeftFingers", "RightFingers"),
    ("LeftAnkle", "RightAnkle"), ("LeftToe", "RightToe"),
    ("LeftHeel", "RightHeel"), ("LeftKnee", "RightKnee"),
):
    _MIRROR_PERM[JOINTS.index(_l)] = JOINTS.index(_r)
    _MIRROR_PERM[JOINTS.index(_r)] = JOINTS.index(_l)
MIRROR_PERM = np.asarray(_MIRROR_PERM)


# --------------------------------------------------------------------------
# Position codec

def decode_position(payload: str) -> np.ndarray:
    """Decode a 276-char base62 payload into a ``(2, 23, 3)`` float64 array.

    Layout: [player, joint, xyz]; player-major, joint in JOINTS order.
    """
    if len(payload) != _POS_CHARS:
        raise ValueError(f"position payload must be {_POS_CHARS} chars, got {len(payload)}")
    try:
        digits = np.frombuffer(
            np.array([_BASE62_INDEX[c] for c in payload], dtype=np.int64), dtype=np.int64
        )
    except KeyError as e:
        raise ValueError(f"non-base62 character {e} in position payload") from None
    vals = (digits[0::2] * 62 + digits[1::2]) / 1000.0
    pos = vals.reshape(PLAYER_COUNT, JOINT_COUNT, 3).copy()
    pos[..., 0] -= 2.0  # x
    pos[..., 2] -= 2.0  # z
    return pos


def encode_position(pos: np.ndarray) -> str:
    """Encode a ``(2, 23, 3)`` array back into the 276-char payload.

    Inverse of :func:`decode_position` (values must round-trip through the
    1 mm quantization, i.e. come from a decoded position).
    """
    pos = np.asarray(pos, dtype=np.float64)
    shifted = pos.copy()
    shifted[..., 0] += 2.0
    shifted[..., 2] += 2.0
    q = np.rint(shifted * 1000).astype(np.int64)
    flat = q.reshape(-1)
    if flat.min() < 0 or flat.max() >= 62 * 62:
        raise ValueError("position out of encodable range [-2, 3.843]")
    chars = []
    for i in flat:
        chars.append(BASE62[i // 62])
        chars.append(BASE62[i % 62])
    return "".join(chars)


# --------------------------------------------------------------------------
# Pose operations (exact semantics of positions.cpp)

def mirror_position(pos: np.ndarray) -> np.ndarray:
    """GrappleMap ``mirror``: negate x, then swap left/right within players."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape[-2:] != (JOINT_COUNT, 3):
        raise ValueError(f"expected (..., {JOINT_COUNT}, 3) pose, got {pos.shape}")
    out = pos.copy()
    out[..., 0] = -out[..., 0]
    out = out[..., MIRROR_PERM, :]
    return out


def swap_players(pos: np.ndarray) -> np.ndarray:
    """GrappleMap ``swap_players``: exchange the two player blocks."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape[-3] != PLAYER_COUNT:
        raise ValueError(f"expected leading player axis of {PLAYER_COUNT}, got {pos.shape}")
    # reverse the PLAYER axis (third from last: (..., players, joints, xyz))
    return np.flip(pos, axis=-3).copy()


def rot_y(v: np.ndarray, angle: float) -> np.ndarray:
    """Rotation about the Y axis, exactly as C++ ``yrot`` * ``v``.

    x' = cos(a) x + sin(a) z ; y' = y ; z' = -sin(a) x + cos(a) z.
    """
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([c * x + s * z, y, -s * x + c * z], axis=-1)


@dataclass(frozen=True)
class Reorientation:
    """Rigid pose transform: rotate about Y by ``angle``, add ``offset``,
    then optionally mirror (x-negation + limb swap) and swap players.

    Semantics of PositionReorientation::operator() (positions.hpp):
    ``apply(reorientation)`` -> ``mirror`` -> ``swap_players``.
    ``offset`` is applied after the rotation.
    """

    angle: float = 0.0
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mirror: bool = False
    swap_players: bool = False

    def apply(self, pos: np.ndarray) -> np.ndarray:
        out = rot_y(np.asarray(pos, dtype=np.float64), self.angle) + np.asarray(self.offset)
        if self.mirror:
            out = mirror_position(out)
        if self.swap_players:
            out = swap_players(out)
        return out


def basically_same(a: np.ndarray, b: np.ndarray) -> bool:
    """C++ basicallySame: per-joint squared distance <= 0.0016 for all joints."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return bool(np.max(np.sum(d * d, axis=-1)) <= SAME_TOL_SQ)


def _match_no_mirror(a: np.ndarray, b: np.ndarray) -> Reorientation | None:
    """is_reoriented_without_mirror_and_swap: align heads, rotate+translate a onto b."""
    a0h, a1h = a[0, _HEAD], a[1, _HEAD]
    b0h, b1h = b[0, _HEAD], b[1, _HEAD]
    angle_off = math.atan2(b1h[0] - b0h[0], b1h[2] - b0h[2]) - math.atan2(
        a1h[0] - a0h[0], a1h[2] - a0h[2]
    )
    offset = b0h - rot_y(a0h, angle_off)
    reo = Reorientation(angle_off, tuple(offset))
    if basically_same(reo.apply(a), b):
        return reo
    return None


def _match_no_swap(a: np.ndarray, b: np.ndarray) -> Reorientation | None:
    r = _match_no_mirror(a, b)
    if r is not None:
        return r
    r = _match_no_mirror(a, mirror_position(b))
    if r is not None:
        return replace(r, mirror=True)
    return None


def is_reoriented(a: np.ndarray, b: np.ndarray) -> Reorientation | None:
    """Return a Reorientation R with R(a) ~= b, or None (positions.cpp).

    Head-to-head distance is invariant under all these transforms and is used
    as a cheap pre-filter (same trick the C++ uses).
    """
    def h2h(p: np.ndarray) -> float:
        d = p[0, _HEAD] - p[1, _HEAD]
        return float(d @ d)

    if abs(h2h(a) - h2h(b)) > 0.05:
        return None
    r = _match_no_swap(a, b)
    if r is not None:
        return r
    r = _match_no_swap(a, swap_players(b))
    if r is not None:
        return replace(r, swap_players=True)
    return None


# --------------------------------------------------------------------------
# File-level parsing

@dataclass
class _RawBlock:
    description: list[str]
    payloads: list[str]
    line_nr: int


def _read_blocks(text: str) -> list[_RawBlock]:
    """Split the file into description-lines + position-payload blocks.

    Mirrors readSeqs() in persistence.cpp: a line starting with a space is a
    position line (positions come in groups of 4 physical lines); anything
    else is a description line.
    """
    lines = text.splitlines()
    blocks: list[_RawBlock] = []
    desc: list[str] = []
    payloads: list[str] = []
    block_start = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(" "):
            if not payloads:
                block_start = i
            group = [l.strip() for l in lines[i : i + _POS_LINES]]
            if len(group) != _POS_LINES:
                raise ValueError(f"truncated position block at line {i + 1}")
            payload = "".join(group)
            if len(payload) != _POS_CHARS:
                raise ValueError(f"bad position payload length at line {i + 1}")
            payloads.append(payload)
            i += _POS_LINES
        else:
            if payloads:
                blocks.append(_RawBlock(desc, payloads, block_start - len(desc) + 1))
                desc, payloads = [], []
            desc.append(line)
            i += 1
    if desc or payloads:
        blocks.append(_RawBlock(desc, payloads, max(0, block_start - len(desc) + 1)))
    return blocks


def _prefixed(desc: list[str], prefix: str) -> frozenset[str]:
    out: set[str] = set()
    for line in desc:
        if line.startswith(prefix):
            out.update(line[len(prefix):].split())
    return frozenset(out)


# --------------------------------------------------------------------------
# Graph model

@dataclass
class Node:
    index: int
    name: str                  # '' for unnamed nodes auto-added during linking
    description: list[str]     # all raw description lines (name first)
    tags: frozenset[str]
    position: np.ndarray       # (2, 23, 3) float64, Y-up meters
    line_nr: int | None        # 1-based line of the block in GrappleMap.txt
    in_edges: list[int] = field(default_factory=list)   # edge indices (edge.to == self)
    out_edges: list[int] = field(default_factory=list)  # edge indices (edge.from == self)

    @property
    def display_name(self) -> str:
        return self.name.replace("\\n", " ") if self.name else f"Position {self.index}"


@dataclass
class Edge:
    index: int
    name: str
    description: list[str]
    tags: frozenset[str]
    properties: frozenset[str]  # subset of {top, bottom, detailed, bidirectional}
    frames: np.ndarray          # (F, 2, 23, 3) float64, F >= 2, no timestamps
    from_node: int
    to_node: int
    from_reo: Reorientation     # from_reo(nodes[from_node].position) == frames[0]
    to_reo: Reorientation       # to_reo(nodes[to_node].position) == frames[-1]
    line_nr: int | None

    @property
    def display_name(self) -> str:
        return self.name.replace("\\n", " ")

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    @property
    def mover(self) -> str | None:
        """Which player performs the move: 'top' (player 0), 'bottom' (player 1),
        or None if unmarked. Interpretation is relative to this edge's own
        stored frames (player 0 = first player axis)."""
        if "top" in self.properties:
            return "top"
        if "bottom" in self.properties:
            return "bottom"
        return None

    @property
    def bidirectional(self) -> bool:
        return "bidirectional" in self.properties

    @property
    def detailed(self) -> bool:
        return "detailed" in self.properties


@dataclass
class Graph:
    nodes: list[Node]
    edges: list[Edge]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def named_node_count(self) -> int:
        return sum(1 for n in self.nodes if n.name)

    def node_by_name(self, name: str) -> Node:
        want = name.replace("\\n", " ")
        for n in self.nodes:
            if n.name and n.name.replace("\\n", " ") == want:
                return n
        raise KeyError(name)

    def edges_with_tag(self, tag: str) -> list[Edge]:
        return [e for e in self.edges if tag in e.tags]

    def all_tags(self) -> set[str]:
        out: set[str] = set()
        for n in self.nodes:
            out |= n.tags
        for e in self.edges:
            out |= e.tags
        return out

    def summary(self) -> str:
        return (
            f"GrappleMap: {self.num_nodes} nodes ({self.named_node_count} named, "
            f"{self.num_nodes - self.named_node_count} auto-added), {self.num_edges} edges"
        )


def _align_mask(cand: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Vectorized is_reoriented_without_mirror_and_swap over candidates.
    ``cand`` is ``(K, 2, 23, 3)`` node positions, ``t`` a single query pose;
    returns a ``(K,)`` bool mask of nodes that reorient-match ``t``.
    Same arithmetic as :func:`_match_no_mirror`, batched over K.
    """
    t0h, t1h = t[0, _HEAD], t[1, _HEAD]
    a0h, a1h = cand[:, 0, _HEAD], cand[:, 1, _HEAD]
    angle = math.atan2(t1h[0] - t0h[0], t1h[2] - t0h[2]) - np.arctan2(
        a1h[:, 0] - a0h[:, 0], a1h[:, 2] - a0h[:, 2]
    )
    c, s = np.cos(angle), np.sin(angle)
    cb, sb = c[:, None, None], s[:, None, None]
    x, y, z = cand[..., 0], cand[..., 1], cand[..., 2]
    xr = cb * x + sb * z
    zr = -sb * x + cb * z
    off = np.stack(
        [t0h[0] - (c * a0h[:, 0] + s * a0h[:, 2]),
         np.full_like(c, t0h[1]) - a0h[:, 1],
         t0h[2] - (-s * a0h[:, 0] + c * a0h[:, 2])],
        axis=-1,
    )
    transformed = np.stack(
        [xr + off[:, 0, None, None], y + off[:, 1, None, None], zr + off[:, 2, None, None]],
        axis=-1,
    )
    max_sq = np.max(np.sum((transformed - t) ** 2, axis=-1), axis=(1, 2))
    return max_sq <= SAME_TOL_SQ


class _NodeIndex:
    """Incremental node store replicating Graph::find_or_add semantics."""

    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.positions = np.zeros((0, PLAYER_COUNT, JOINT_COUNT, 3))
        self._h2h = np.zeros(0)

    def add(self, node: Node, position: np.ndarray) -> None:
        self.nodes.append(node)
        self.positions = np.concatenate([self.positions, position[None]])
        d = position[0, _HEAD] - position[1, _HEAD]
        self._h2h = np.append(self._h2h, d @ d)

    def find_or_add(self, position: np.ndarray, line_nr: int | None) -> tuple[Node, Reorientation]:
        d = position[0, _HEAD] - position[1, _HEAD]
        query_h2h = float(d @ d)
        # Candidates in index order (C++ scans all nodes, first match wins);
        # the head-to-head pre-filter is exactly the C++ cheap rejection.
        cand = np.nonzero(np.abs(self._h2h - query_h2h) <= 0.05)[0]
        if cand.size:
            # Batched equivalent of calling is_reoriented() per candidate:
            # per candidate try plain then mirrored; if no candidate matches,
            # retry with the query's players swapped (plain then mirrored).
            # First passing candidate in index order wins, plain preferred.
            variants = [
                (False, False, position),
                (False, True, mirror_position(position)),
                (True, False, swap_players(position)),
                (True, True, mirror_position(swap_players(position))),
            ]
            for group in (variants[:2], variants[2:]):
                masks = [_align_mask(self.positions[cand], t) for _, _, t in group]
                hit = np.nonzero(masks[0] | masks[1])[0]
                if hit.size:
                    k = cand[hit[0]]
                    swap, mir, t = (group[0] if masks[0][hit[0]] else group[1])
                    reo = _match_no_mirror(self.positions[k], t)
                    assert reo is not None, "batched/scalar matcher disagreement"
                    return self.nodes[k], Reorientation(
                        reo.angle, reo.offset, mirror=mir, swap_players=swap
                    )
        node = Node(len(self.nodes), "", [], frozenset(), position, line_nr)
        self.add(node, position)
        return node, Reorientation()


def load_graph(path: str | Path | None = None) -> Graph:
    """Parse GrappleMap.txt into a full Graph (nodes + linked edges).

    Self-verifies on load: every edge endpoint is checked to reorient-match
    its node position, mirroring the C++ graph invariants.
    """
    path = Path(path) if path is not None else DEFAULT_DB_PATH
    blocks = _read_blocks(path.read_text())

    idx = _NodeIndex()
    transitions: list[_RawBlock] = []

    # 1) named single-position blocks become nodes (file order), like
    #    Graph::Graph which first moves all single-position sequences into pp.
    for b in blocks:
        if len(b.payloads) == 1:
            pos = decode_position(b.payloads[0])
            name = b.description[0] if b.description else ""
            node = Node(
                index=len(idx.nodes),
                name=name,
                description=list(b.description),
                tags=_prefixed(b.description, "tags:"),
                position=pos,
                line_nr=b.line_nr,
            )
            idx.add(node, pos)
        else:
            transitions.append(b)

    # 2) link transitions, appending unnamed nodes when endpoints match nothing
    edges: list[Edge] = []
    for b in transitions:
        frames = np.stack([decode_position(p) for p in b.payloads])
        from_node, from_reo = idx.find_or_add(frames[0], b.line_nr)
        to_node, to_reo = idx.find_or_add(frames[-1], b.line_nr)
        name = b.description[0] if b.description else ""
        edges.append(
            Edge(
                index=len(edges),
                name=name,
                description=list(b.description),
                tags=_prefixed(b.description, "tags:"),
                properties=_prefixed(b.description, "properties:"),
                frames=frames,
                from_node=from_node.index,
                to_node=to_node.index,
                from_reo=from_reo,
                to_reo=to_reo,
                line_nr=b.line_nr,
            )
        )

    # 3) adjacency (bidirectional edges are additionally traversable in
    #    reverse; that is recorded on the edge, not duplicated here)
    for e in edges:
        idx.nodes[e.from_node].out_edges.append(e.index)
        idx.nodes[e.to_node].in_edges.append(e.index)

    graph = Graph(idx.nodes, edges)

    # self-check: C++ Edge invariants g[from]==positions.front()/back()
    for e in edges:
        if not basically_same(e.from_reo.apply(graph.nodes[e.from_node].position), e.frames[0]):
            raise AssertionError(f"edge {e.index} 'from' endpoint mismatch")
        if not basically_same(e.to_reo.apply(graph.nodes[e.to_node].position), e.frames[-1]):
            raise AssertionError(f"edge {e.index} 'to' endpoint mismatch")

    return graph
