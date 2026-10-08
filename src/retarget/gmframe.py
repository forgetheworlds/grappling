"""GrappleMap pair-pose frames: transforms, alignment, edge-chain assembly.

Everything here works on Y-up GrappleMap pair poses ``(2, 23, 3)`` (or
``(F, 2, 23, 3)`` frame stacks) in meters, before retarget-world placement
(see world.py).

``PairTransform`` mirrors the parser's ``Reorientation`` semantics exactly:
``apply(x) = swap?(mirror?(rotY(angle, x) + offset))`` — plus the
compose/inverse algebra needed to chain transitions through shared graph
nodes. Key identities (M = mirror, S = player swap, G(a,o) = rotY + offset):
  M ∘ rotY(a) = rotY(−a) ∘ M ;  S commutes with both M and G.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grapplemap import JOINTS, Reorientation, mirror_position, swap_players

J = {n: i for i, n in enumerate(JOINTS)}
CORE, NECK, HEAD = J["Core"], J["Neck"], J["Head"]


def rot_y(v: np.ndarray, angle: float) -> np.ndarray:
    """Rotation about the Y axis, C++ ``yrot`` convention:
    x' = cos·x + sin·z ; y' = y ; z' = −sin·x + cos·z."""
    v = np.asarray(v, dtype=np.float64)
    ca, sa = np.cos(angle), np.sin(angle)
    out = v.copy()
    out[..., 0] = ca * v[..., 0] + sa * v[..., 2]
    out[..., 2] = -sa * v[..., 0] + ca * v[..., 2]
    return out


_MIRROR3 = np.array([-1.0, 1.0, 1.0])


@dataclass(frozen=True)
class PairTransform:
    """Rigid (+ optional mirror / player-swap) transform of pair poses.

    Same semantics as grapplemap.Reorientation (rotation & offset first,
    then mirror, then player swap). ``world player p = rot(raw[p ^ swap])``.
    """

    angle: float = 0.0
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mirror: bool = False
    swap_players: bool = False

    def apply(self, pos: np.ndarray) -> np.ndarray:
        out = rot_y(np.asarray(pos, dtype=np.float64), self.angle) \
            + np.asarray(self.offset, dtype=np.float64)
        if self.mirror:
            out = mirror_position(out)
        if self.swap_players:
            out = swap_players(out)
        return out

    def compose(self, outer: "PairTransform") -> "PairTransform":
        """``self ∘ outer``: ``(self.compose(o)).apply(x) == self.apply(o.apply(x))``.

        Derived from the commutation identities; angle of the outer rotation
        flips sign when the outer transform mirrors.
        """
        a_self = -self.angle if outer.mirror else self.angle
        o_self = np.asarray(self.offset, dtype=np.float64)
        if outer.mirror:
            o_self = o_self * _MIRROR3
        angle = a_self + outer.angle
        offset = rot_y(np.asarray(outer.offset, dtype=np.float64), a_self) + o_self
        return PairTransform(float(angle), tuple(float(x) for x in offset),
                             self.mirror != outer.mirror,
                             self.swap_players != outer.swap_players)

    def inverse(self) -> "PairTransform":
        """Exact inverse (mirror & swap are involutions).

        self.apply = S^s ∘ M^m ∘ G(a, o).  Undoing in reverse order and
        re-canonicalising:
          m=0: (−a, −rotY(−a)o, False, s)
          m=1: ( a, −M(rotY(−a)o), True,  s)
        (verified numerically in tests/test_retarget.py).
        """
        o = np.asarray(self.offset, dtype=np.float64)
        if self.mirror:
            u = -(rot_y(o, -self.angle) * _MIRROR3)
            return PairTransform(self.angle, tuple(float(x) for x in u),
                                 True, self.swap_players)
        u = -rot_y(o, -self.angle)
        return PairTransform(-self.angle, tuple(float(x) for x in u),
                             False, self.swap_players)


def from_reorientation(r: Reorientation) -> PairTransform:
    return PairTransform(float(r.angle), tuple(float(x) for x in r.offset),
                         bool(r.mirror), bool(r.swap_players))


def yaw_align(src: np.ndarray, dst: np.ndarray, w: np.ndarray | None = None):
    """Least-squares rot-Y + translation aligning ``src`` onto ``dst``.

    Flattened (N,3) point sets. Returns (PairTransform(no mirror/swap),
    weighted rms).
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if w is None:
        w = np.ones(len(src))
    w = np.asarray(w, dtype=np.float64)
    c_s = (src * w[:, None]).sum(0) / w.sum()
    c_d = (dst * w[:, None]).sum(0) / w.sum()
    s0, d0 = src - c_s, dst - c_d
    angle = float(np.arctan2(
        (w * (d0[:, 0] * s0[:, 2] - d0[:, 2] * s0[:, 0])).sum(),
        (w * (s0[:, 0] * d0[:, 0] + s0[:, 2] * d0[:, 2])).sum()))
    ca, sa = np.cos(angle), np.sin(angle)
    R = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    offset = c_d - R @ c_s
    res = src @ R.T + offset - dst
    rms = float(np.sqrt((w[:, None] * res * res).sum() / w.sum()))
    return PairTransform(angle, tuple(float(x) for x in offset)), rms


def best_pair_align(src: np.ndarray, dst: np.ndarray,
                    w: np.ndarray | None = None, allow_swap: bool = True):
    """Best rigid (rot-Y + translation [+ player swap]) alignment src -> dst.

    Mirror is never used (it would flip the chirality of the scene).
    Returns (PairTransform, weighted rms, swapped: bool).
    """
    best = None
    for swapped in ((False, True) if allow_swap else (False,)):
        s = src[::-1] if swapped else src
        ww = None if w is None else np.concatenate([w, w])
        t, rms = yaw_align(s.reshape(-1, 3), dst.reshape(-1, 3), ww)
        if best is None or rms < best[1]:
            best = (PairTransform(t.angle, t.offset, False, swapped),
                    float(rms), bool(swapped))
    return best


# --------------------------------------------------------------------------
# Edge-chain assembly
# --------------------------------------------------------------------------

@dataclass
class ChainResult:
    frames: np.ndarray        # (F, 2, 23, 3) chain world (Y-up, meters)
    times: np.ndarray         # (F,) keyframe times (s), strictly increasing
    world_swap: np.ndarray    # (F,) bool: world player p came from raw p^swap
    edges: list[int]          # edge ids in order
    junctions: list[dict]     # per junction: kind / rms / swap / transition_dt
    frame_edges: np.ndarray   # (F,) index into `edges` for each output frame


def _edge_dt(edge) -> float:
    return 0.1 if edge.detailed else 0.2


def assemble_chain(g, edge_ids: list[int], transition_dt: float = 0.5,
                   w: np.ndarray | None = None) -> ChainResult:
    """Assemble consecutive edges into one world keyframe track.

    Per junction:
    - exact: prev.to_node == next.from_node — the shared node fixes the
      relative transform exactly (residual ~0); the duplicate shared frame is
      dropped, no extra time inserted.
    - aligned: least-squares rot-Y+translation (+ optional player swap) of
      ``next.frames[0]`` onto the previous world pose; both boundary frames
      kept, separated by ``transition_dt`` seconds.
    """
    out_frames: list[np.ndarray] = []
    times: list[float] = []
    swaps: list[bool] = []
    junctions: list[dict] = []
    frame_edges: list[int] = []
    cum = PairTransform()  # raw frames of the CURRENT edge -> world

    for k, eid in enumerate(edge_ids):
        edge = g.edges[eid]
        assert edge.frame_count >= 2, f"edge t{eid} has <2 frames"
        if k == 0:
            world = cum.apply(edge.frames)
            out_frames.append(world)
            swaps.extend([cum.swap_players] * len(world))
            frame_edges.extend([k] * len(world))
            times.extend([_edge_dt(edge) * i for i in range(len(world))])
            continue

        prev_edge = g.edges[edge_ids[k - 1]]
        prev_last_t = times[-1]
        prev_pose = out_frames[-1][-1]  # world pose at the end of the chain
        if prev_edge.to_node == edge.from_node:
            node_to_world = cum.compose(from_reorientation(prev_edge.to_reo))
            cum = node_to_world.compose(from_reorientation(edge.from_reo).inverse())
            world = cum.apply(edge.frames[1:])  # frames[0] == prev last pose
            out_frames.append(world)
            swaps.extend([cum.swap_players] * len(world))
            frame_edges.extend([k] * len(world))
            times.extend([prev_last_t + _edge_dt(edge) * (i + 1)
                          for i in range(len(world))])
            junctions.append({"after": edge_ids[k - 1], "before": eid,
                              "kind": "shared_node", "rms": 0.0,
                              "swap": cum.swap_players, "transition_dt": 0.0})
        else:
            tr, rms, swapped = best_pair_align(edge.frames[0], prev_pose,
                                               w=w, allow_swap=True)
            cum = tr
            world = cum.apply(edge.frames)
            out_frames.append(world)
            swaps.extend([cum.swap_players] * len(world))
            frame_edges.extend([k] * len(world))
            times.extend([prev_last_t + transition_dt
                          + _edge_dt(edge) * i for i in range(len(world))])
            junctions.append({"after": edge_ids[k - 1], "before": eid,
                              "kind": "aligned", "rms": float(rms),
                              "swap": bool(swapped),
                              "transition_dt": float(transition_dt)})

    frames = np.concatenate(out_frames, axis=0)
    times_arr = np.asarray(times, dtype=np.float64)
    assert np.all(np.diff(times_arr) > 0), "keyframe times must strictly increase"
    return ChainResult(frames, times_arr, np.asarray(swaps, dtype=bool),
                       list(edge_ids), junctions,
                       np.asarray(frame_edges, dtype=np.int64))


def single_player_align(src_pose: np.ndarray, src_player: int,
                        dst_frames: np.ndarray, dst_player: int,
                        w: np.ndarray | None = None):
    """Nearest-frame rigid alignment of ONE player's landmarks.

    Finds the dst frame minimizing weighted landmark rms between
    ``src_pose[src_player]`` and ``dst_frames[i, dst_player]`` under the best
    rot-Y + translation. Returns (dst_index, PairTransform, rms).
    """
    best = None
    for i in range(len(dst_frames)):
        t, rms = yaw_align(src_pose[src_player].reshape(-1, 3),
                           dst_frames[i, dst_player].reshape(-1, 3), w)
        if best is None or rms < best[2]:
            best = (i, t, float(rms))
    return best
