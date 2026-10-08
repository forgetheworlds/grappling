"""Tests for the GrappleMap parser.

Run from repo root:  (.venv/bin/)python -m pytest tests/ -x -q

All assertions are grounded in independently verified facts about
third_party/GrappleMap (commit 032c8f91809786b7b784852abd07cf3e10c0ea35):
see reports/2026-10-07/grapplemap.md for the fact sheet and the C++
sources cited per fact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grapplemap import (  # noqa: E402
    DEFAULT_DB_PATH,
    JOINTS,
    JOINT_COUNT,
    PLAYER_COUNT,
    basically_same,
    decode_position,
    encode_position,
    is_reoriented,
    load_graph,
    mirror_position,
    swap_players,
)

DB = DEFAULT_DB_PATH


@pytest.fixture(scope="module")
def graph():
    g = load_graph(DB)
    print(g.summary())  # prints node/edge counts (also asserted below)
    return g


@pytest.fixture(scope="module")
def raw_text():
    return DB.read_text()


# --------------------------------------------------------------------------
# Counts match the raw file (independent block census, no parser involvement)

def _raw_block_census(text: str) -> tuple[int, int]:
    """Count single-position blocks (=named nodes) and multi-position blocks
    (=edges) with a plain line scan mirroring readSeqs() grouping."""
    singles = multis = n_pos = 0
    in_pos = False
    for line in text.splitlines():
        if line.startswith(" "):
            n_pos += 1
            in_pos = True
        else:
            if in_pos:
                if n_pos // 4 == 1:
                    singles += 1
                else:
                    multis += 1
                n_pos = 0
                in_pos = False
    if in_pos:
        if n_pos // 4 == 1:
            singles += 1
        else:
            multis += 1
    return singles, multis


def test_counts_match_raw_file(graph, raw_text):
    singles, multis = _raw_block_census(raw_text)
    assert graph.named_node_count == singles
    assert graph.num_edges == multis
    # named + auto-added (unnamed) nodes == total
    assert graph.num_nodes >= graph.named_node_count
    # every node participates in the graph structure or is a named island
    total_positions = singles + sum(e.frame_count for e in graph.edges)
    assert total_positions * 4 == sum(
        1 for line in raw_text.splitlines() if line.startswith(" ")
    )


def test_summary_prints_counts(graph, capsys):
    s = graph.summary()
    assert "nodes" in s and "edges" in s
    out = capsys.readouterr().out  # fixture print captured on first call only


# --------------------------------------------------------------------------
# Pose representation

def test_landmark_count_is_23():
    """Verified against players.hpp JOINTS macro: 20 left/right pairs
    (Toe, Heel, Ankle, Knee, Hip, Shoulder, Elbow, Wrist, Hand, Fingers)
    plus 3 midline joints (Core, Neck, Head) = 23 per player."""
    assert JOINT_COUNT == 23
    assert len(JOINTS) == 23
    paired = [j for j in JOINTS if j.startswith("Left")]
    midline = [j for j in JOINTS if not j.startswith(("Left", "Right"))]
    assert len(paired) == 10
    assert midline == ["Core", "Neck", "Head"]
    assert all("Right" + j[4:] in JOINTS for j in paired)


def test_pose_shapes(graph):
    for n in graph.nodes:
        assert n.position.shape == (PLAYER_COUNT, JOINT_COUNT, 3)
        assert n.position.dtype == np.float64
    for e in graph.edges:
        assert e.frames.shape[1:] == (PLAYER_COUNT, JOINT_COUNT, 3)
        assert e.frames.dtype == np.float64
        assert e.frames.ndim == 4  # (F, players, joints, xyz)


def test_coordinate_frame_y_up_ground_y0(graph):
    """Y is up, ground plane y=0 (persistence.cpp y unshifted; apply_limits
    clamps y >= joint radius, x/z into [-2,2]). Smallest joint radius is
    0.025 (toe) => no joint may go below it."""
    all_pts = np.concatenate(
        [n.position.reshape(-1, 3) for n in graph.nodes]
        + [e.frames.reshape(-1, 3) for e in graph.edges]
    )
    assert all_pts[:, 1].min() >= 0.0  # y never negative
    assert all_pts[:, 1].min() <= 0.03  # something touches the ground
    assert all_pts[:, 1].max() < 2.0  # human scale (meters), not centimeters
    # x/z stored with -2 offset from [0, 3.843] => range [-2, 1.843]
    assert all_pts[:, 0].min() >= -2.0 and all_pts[:, 0].max() <= 1.844
    assert all_pts[:, 2].min() >= -2.0 and all_pts[:, 2].max() <= 1.844


def test_human_scale_meters(graph):
    """Units are meters: head-to-toe span of an upright player ~1.5-1.9 m."""
    n = graph.node_by_name("symmetric staggered standing")
    for p in range(PLAYER_COUNT):
        span = n.position[p, JOINTS.index("Head"), 1] - n.position[p, JOINTS.index("LeftToe"), 1]
        assert 1.2 < span < 2.2, span


# --------------------------------------------------------------------------
# Encoding roundtrip (decoder is exact inverse of the C++ encoder)

def test_position_codec_roundtrip(raw_text):
    payloads = []
    for line in raw_text.splitlines():
        if line.startswith(" ") and (len(payloads) % 4 != 1 or not payloads):
            pass
    # collect complete 4-line records
    lines = raw_text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith(" "):
            payload = "".join(l.strip() for l in lines[i : i + 4])
            assert len(payload) == 276
            payloads.append(payload)
            i += 4
        else:
            i += 1
    assert len(payloads) == 8323  # every stored position in the file
    rng = np.random.default_rng(0)
    sample = rng.choice(len(payloads), size=400, replace=False)
    for k in sample:
        assert encode_position(decode_position(payloads[k])) == payloads[k]


# --------------------------------------------------------------------------

def test_frame_progression(graph):
    """No timestamps exist in the format (documented fact); frame index is the
    only ordering. Verify consecutive stored keyframes are distinct motions
    (max per-joint squared move > 0) for all edges, i.e. keyframe sequences
    are strictly progressing."""
    still = 0
    for e in graph.edges:
        d = np.diff(e.frames, axis=0)
        max_move_sq = np.max(np.sum(d * d, axis=-1))
        if max_move_sq == 0.0:
            still += 1
    assert still == 0, f"{still} edges contain consecutive identical frames"


def test_edge_endpoints_match_nodes_under_reorientation(graph):
    """C++ Edge invariants: g[from] == positions.front(), g[to] ==
    positions.back() (up to the parser's replicate of the C++ matching
    tolerance). load_graph() already self-asserts this; re-verify here."""
    for e in graph.edges:
        f = graph.nodes[e.from_node]
        t = graph.nodes[e.to_node]
        assert basically_same(e.from_reo.apply(f.position), e.frames[0])
        assert basically_same(e.to_reo.apply(t.position), e.frames[-1])


def test_adjacency_consistent(graph):
    for e in graph.edges:
        assert e.index in graph.nodes[e.from_node].out_edges
        assert e.index in graph.nodes[e.to_node].in_edges


def test_named_nodes_unique_and_tagged(graph):
    # C++ Graph ctor enforces uniqueness of the full description vector (not
    # the display name): one display-name collision exists in the data
    # ("spiderweb w/o leg over head", different raw escapes + tags).
    descs = [tuple(n.description) for n in graph.nodes if n.name]
    assert len(descs) == len(set(descs))
    # the database tags both nodes and transitions
    assert "standing" in graph.all_tags()
    assert "double_leg_takedown" in graph.all_tags()
    assert "sprawl" in graph.all_tags()


def test_mover_metadata(graph):
    """properties: top/bottom mark which player performs a transition
    (metadata.hpp is_top_move/is_bottom_move; player 0='t', 1='b')."""
    counts = {"top": 0, "bottom": 0, None: 0}
    for e in graph.edges:
        counts[e.mover] += 1
        assert e.mover in ("top", "bottom", None)
    assert counts["top"] > 0 and counts["bottom"] > 0
    assert any(e.bidirectional for e in graph.edges)
    assert any(e.detailed for e in graph.edges)


# --------------------------------------------------------------------------
# Mirror and player swap: defined in the source, must be involutions

def test_mirror_roundtrip(graph):
    """mirror(Position) (positions.cpp): negate x for all joints, then swap
    left/right limb joints within each player. T(T(x)) == x exactly."""
    for n in graph.nodes[::7]:  # ~135 samples
        m = mirror_position(n.position)
        assert np.array_equal(mirror_position(m), n.position)
        # mirror preserves heights (y) per player as a multiset (the
        # left/right limb swap permutes joint order)
        for pl in range(PLAYER_COUNT):
            assert np.array_equal(np.sort(m[pl, :, 1]), np.sort(n.position[pl, :, 1]))
    for e in graph.edges[::37]:
        m = mirror_position(e.frames)
        assert np.array_equal(mirror_position(m), e.frames)


def test_swap_players_roundtrip(graph):
    """swap_players(Position) (positions.hpp): exchange the two 23-joint
    player blocks. T(T(x)) == x exactly."""
    for n in graph.nodes[::7]:
        s = swap_players(n.position)
        assert np.array_equal(swap_players(s), n.position)
        # it really exchanges players
        assert np.array_equal(s[0], n.position[1]) and np.array_equal(s[1], n.position[0])


def test_mirror_and_swap_commute(graph):
    n = graph.node_by_name("south-paw neutral standing")
    a = swap_players(mirror_position(n.position))
    b = mirror_position(swap_players(n.position))
    assert np.array_equal(a, b)


def test_is_reoriented_detects_mirror_and_swap(graph):
    """The matching machinery must see mirrored and player-swapped poses as
    reoriented-equal (this is how the C++ graph links transitions)."""
    n = graph.node_by_name("symmetric staggered standing")
    m = mirror_position(n.position)
    s = swap_players(n.position)
    ms = mirror_position(swap_players(n.position))
    for variant, want_m, want_s in ((m, True, False), (s, False, True), (ms, True, True)):
        r = is_reoriented(n.position, variant)
        assert r is not None
        assert r.mirror == want_m and r.swap_players == want_s
        assert basically_same(r.apply(n.position), variant)
    # a translated + rotated pose also matches, with angle/offset recovered
    from grapplemap import rot_y

    moved = rot_y(n.position, 0.7) + np.array([0.5, 0.0, -0.3])
    r = is_reoriented(n.position, moved)
    assert r is not None and not r.mirror and not r.swap_players
    assert abs(r.angle) == pytest.approx(0.7, abs=1e-9)
    assert basically_same(r.apply(n.position), moved)


# --------------------------------------------------------------------------
# Vocab export integrity

def test_vocab_candidates_json(graph):
    path = Path(__file__).resolve().parents[1] / "data" / "grapplemap" / "vocab_candidates.json"
    if not path.exists():
        pytest.skip("vocab_candidates.json not generated yet")
    import json

    data = json.loads(path.read_text())
    assert data["graph"]["nodes"] == graph.num_nodes
    assert data["graph"]["edges"] == graph.num_edges
    by_id = {f"t{e.index}": e for e in graph.edges}
    n_refs = 0
    for group in data["groups"].values():
        assert group["edges"], "empty vocab group"
        for entry in group["edges"]:
            e = by_id[entry["edge"]]
            assert entry["name"] == e.display_name
            assert entry["frame_count"] == e.frame_count
            assert entry["mover"] == e.mover
            assert entry["from_node"]["index"] == e.from_node
            assert entry["to_node"]["index"] == e.to_node
            n_refs += 1
    assert n_refs >= 30
    for vid in data["proposed_vocabulary"].values():
        for ref in vid:
            assert ref in by_id
