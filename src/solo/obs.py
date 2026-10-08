"""Fixed observation layouts for the solo drill (actor vs privileged critic).

Contract (``docs/SOLO_DRILL.md`` §4): the ACTOR observation is compact and
self-relative -- **no global simulator state** (no world position, no absolute
orientation; only proprioception, the gravity direction in the pelvis frame,
the previous action and the command).  Contacts, CoM/support internals and the
virtual-opponent (marker) internals are **privileged**: they exist only in the
critic vector (``critic_obs = [actor | privileged]``).

Actor (115):
    ``0:3``   base linear velocity, pelvis frame (m/s)
    ``3:6``   base angular velocity, pelvis frame (rad/s)
    ``6:9``   gravity direction in the pelvis frame (R^T @ (0,0,-1))
    ``9:38``  joint positions relative to the ``stand`` keyframe (rad)
    ``38:67`` joint velocities (rad/s)
    ``67:96`` previous action (unit action in [-1, 1])
    ``96:99`` command velocity (vx, vy, wz) -- heading frame
    ``99:101`` stance height / stance width commands (m)
    ``101:113`` skill one-hot (12)
    ``113:114`` lead leg (+1 right / -1 left)
    ``114:115`` shot phase in [0, 1]

Privileged extension (43), used by the critic only:
    contacts (feet, knees, hands, grounded), dorsal contact, tilt, pelvis
    height error, CoM position/velocity relative to the pelvis, support centre
    (mean of loaded foot sites -- explicitly **not** a support-polygon margin),
    CoM-to-support offset, the four marker targets relative to the pelvis,
    the next scheduled push (dt, direction, magnitude), foot slip, actuator
    saturation, joint-limit proximity, mean |joint velocity|, shot phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .commands import N_SKILLS, Command
from .fall import ContactState
from .scene import MARKER_NAMES, N_JOINTS

ACTOR_LAYOUT: tuple[tuple[str, tuple[int, int]], ...] = (
    ("base_linvel_local", (0, 3)),
    ("base_angvel_local", (3, 6)),
    ("gravity_local", (6, 9)),
    ("joint_pos_rel", (9, 38)),
    ("joint_vel", (38, 67)),
    ("prev_action", (67, 96)),
    ("cmd_vel", (96, 99)),
    ("cmd_stance", (99, 101)),
    ("skill_onehot", (101, 101 + N_SKILLS)),
    ("lead_leg", (101 + N_SKILLS, 102 + N_SKILLS)),
    ("phase", (102 + N_SKILLS, 103 + N_SKILLS)),
)
ACTOR_DIM = ACTOR_LAYOUT[-1][1][1]

PRIV_LAYOUT: tuple[tuple[str, tuple[int, int]], ...] = (
    ("contact_foot_lr", (0, 2)),
    ("contact_knee_lr", (2, 4)),
    ("contact_hand_lr", (4, 6)),
    ("contact_grounded", (6, 7)),
    ("dorsal_contact", (7, 8)),
    ("tilt_norm", (8, 9)),
    ("pelvis_z_err", (9, 10)),
    ("com_rel_local", (10, 13)),
    ("com_vel_local", (13, 16)),
    ("support_center_local", (16, 19)),
    ("com_to_support_xy", (19, 21)),
    ("marker_pelvis_local", (21, 24)),
    ("marker_leg_local", (24, 27)),
    ("marker_hand_local", (27, 30)),
    ("marker_hand_other_local", (30, 33)),
    ("next_push_dt", (33, 34)),
    ("next_push_dir", (34, 36)),
    ("next_push_mag", (36, 37)),
    ("foot_slip", (37, 38)),
    ("sat_frac", (38, 39)),
    ("limit_prox", (39, 40)),
    ("joint_speed_mean", (40, 41)),
    ("shot_phase", (41, 42)),
    ("marker_distance", (42, 43)),
)
PRIV_DIM = PRIV_LAYOUT[-1][1][1]
CRITIC_DIM = ACTOR_DIM + PRIV_DIM

#: push magnitude normalization used by the privileged next-push field (N*s)
PUSH_NORM = 12.0
#: next-push dt normalization (s)
PUSH_DT_NORM = 2.0

_ZERO3 = np.zeros(3, dtype=np.float64)


@dataclass
class ObsContext:
    """Everything the observation builders read (hand-constructible for tests)."""

    base_linvel_local: np.ndarray = field(default_factory=lambda: _ZERO3.copy())
    base_angvel_local: np.ndarray = field(default_factory=lambda: _ZERO3.copy())
    gravity_local: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    joint_pos_rel: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))
    joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))
    cmd: Command = field(default_factory=Command)
    phase: float = 0.0
    contacts: ContactState = field(default_factory=ContactState)
    dorsal_contact: bool = False
    tilt_deg: float = 0.0
    pelvis_z: float = 0.79
    stand_height: float = 0.79
    com_rel_local: np.ndarray = field(default_factory=lambda: _ZERO3.copy())
    com_vel_local: np.ndarray = field(default_factory=lambda: _ZERO3.copy())
    support_center_local: np.ndarray = field(default_factory=lambda: _ZERO3.copy())
    com_to_support_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    marker_local: dict[str, np.ndarray] = field(default_factory=dict)
    next_push: tuple[float, float, float] | None = None  # (dt, world yaw, N*s)
    foot_slip: float = 0.0
    sat_frac: float = 0.0
    limit_prox: float = 0.0
    shot_phase: float = 0.0
    marker_distance: float = 1.0


def obs_dims() -> dict:
    return {"actor": ACTOR_DIM, "privileged": PRIV_DIM, "critic": CRITIC_DIM}


def _onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float64)
    v[int(idx) % n] = 1.0
    return v


def actor_obs(ctx: ObsContext) -> np.ndarray:
    """(ACTOR_DIM,) float32 actor observation from ``ctx``."""
    parts = [
        np.asarray(ctx.base_linvel_local, np.float64).reshape(3),
        np.asarray(ctx.base_angvel_local, np.float64).reshape(3),
        np.asarray(ctx.gravity_local, np.float64).reshape(3),
        np.asarray(ctx.joint_pos_rel, np.float64).reshape(N_JOINTS),
        np.asarray(ctx.joint_vel, np.float64).reshape(N_JOINTS),
        np.clip(np.asarray(ctx.prev_action, np.float64).reshape(N_JOINTS), -1.0, 1.0),
        np.array([ctx.cmd.vx, ctx.cmd.vy, ctx.cmd.wz]),
        np.array([ctx.cmd.stance_height, ctx.cmd.stance_width]),
        _onehot(int(ctx.cmd.skill_id), N_SKILLS),
        np.array([float(np.sign(ctx.cmd.lead_leg) or 1.0)]),
        np.array([float(np.clip(ctx.phase, 0.0, 1.0))]),
    ]
    out = np.concatenate(parts).astype(np.float32)
    assert out.size == ACTOR_DIM, out.size
    return out


def _marker_local(ctx: ObsContext, name: str) -> np.ndarray:
    v = ctx.marker_local.get(name)
    return np.zeros(3) if v is None else np.asarray(v, np.float64).reshape(3)


def privileged_obs(ctx: ObsContext) -> np.ndarray:
    """(PRIV_DIM,) float32 critic-only extension from ``ctx``."""
    c = ctx.contacts
    lead = "l" if int(ctx.cmd.lead_leg) == -1 else "r"
    other = "r" if lead == "l" else "l"
    if ctx.next_push is None:
        push = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        dt, yaw, mag = ctx.next_push
        push = np.array([float(np.clip(dt, 0.0, PUSH_DT_NORM)) / PUSH_DT_NORM,
                         math.sin(yaw), math.cos(yaw),
                         float(np.clip(mag / PUSH_NORM, 0.0, 1.0))])
    parts = [
        np.array([c.left_foot, c.right_foot], np.float64),
        np.array([c.left_knee, c.right_knee], np.float64),
        np.array([c.left_hand, c.right_hand], np.float64),
        np.array([1.0 if c.grounded else 0.0]),
        np.array([1.0 if ctx.dorsal_contact else 0.0]),
        np.array([float(np.clip(ctx.tilt_deg / 90.0, 0.0, 1.0))]),
        np.array([float(ctx.pelvis_z - ctx.stand_height)]),
        np.asarray(ctx.com_rel_local, np.float64).reshape(3),
        np.asarray(ctx.com_vel_local, np.float64).reshape(3),
        np.asarray(ctx.support_center_local, np.float64).reshape(3),
        np.asarray(ctx.com_to_support_xy, np.float64).reshape(2),
        _marker_local(ctx, "a_marker_pelvis"),
        _marker_local(ctx, f"a_marker_leg_{lead}"),
        _marker_local(ctx, f"a_marker_hand_{lead}"),
        _marker_local(ctx, f"a_marker_hand_{other}"),
        push[0:1], push[1:3], push[3:4],
        np.array([float(ctx.foot_slip)]),
        np.array([float(ctx.sat_frac)]),
        np.array([float(ctx.limit_prox)]),
        np.array([float(np.mean(np.abs(np.asarray(ctx.joint_vel, np.float64))))]),
        np.array([float(np.clip(ctx.shot_phase, 0.0, 1.0))]),
        np.array([float(ctx.marker_distance)]),
    ]
    out = np.concatenate(parts).astype(np.float32)
    assert out.size == PRIV_DIM, out.size
    return out


def critic_obs(ctx: ObsContext) -> np.ndarray:
    """(CRITIC_DIM,) = [actor | privileged] float32."""
    return np.concatenate([actor_obs(ctx), privileged_obs(ctx)])


def layout_table(which: str = "actor") -> tuple:
    """(name, (start, stop)) table for tests/docs."""
    if which == "actor":
        return ACTOR_LAYOUT
    if which == "privileged":
        return PRIV_LAYOUT
    raise ValueError(which)


if __name__ == "__main__":  # self-check
    ctx = ObsContext()
    a = actor_obs(ctx)
    p = privileged_obs(ctx)
    c = critic_obs(ctx)
    assert a.shape == (ACTOR_DIM,) and a.dtype == np.float32
    assert p.shape == (PRIV_DIM,) and c.shape == (CRITIC_DIM,)
    # content: skill one-hot at the documented slice
    s0, s1 = dict(ACTOR_LAYOUT)["skill_onehot"]
    assert a[s0:s1].sum() == 1.0 and int(np.argmax(a[s0:s1])) == int(ctx.cmd.skill_id)
    # a walking command must appear verbatim in the cmd_vel slice
    ctx2 = ObsContext(cmd=Command(vx=0.3, vy=-0.1, wz=0.2, skill_id=8))
    a2 = actor_obs(ctx2)
    s0, s1 = dict(ACTOR_LAYOUT)["cmd_vel"]
    assert np.allclose(a2[s0:s1], [0.3, -0.1, 0.2], atol=1e-6)
    # markers are privileged: changing them must not touch the actor obs
    ctx3 = ObsContext(marker_local={n: np.ones(3) for n in MARKER_NAMES})
    assert np.array_equal(actor_obs(ctx3), a)
    assert not np.array_equal(privileged_obs(ctx3), p)
    print("solo.obs self-check OK:", obs_dims())
