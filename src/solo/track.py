"""Reference-conditioned motion tracking (Agent 2: physics-based motion learning).

The defect this module fixes: the existing actor observation's phase channel
(``obs[114]``) is dead for every non-shot task, so a reference-conditioned
actor could not see what distinguishes a stance hold from a shuffle, a level
change or a recovery.  Here the actor receives an appended REFERENCE BLOCK
(local-frame targets, the reference joint targets, the reference foot-contact
command, the phase progress inside the tracked segment, the skill label, the
lead leg, a connector flag and a short future root target) -- built ONLY from
the reference track + a phase clock, i.e. a *commanded movement target* (the
formulation used by BeyondMimic/TWIST-style trackers; no scripted state-
feedback teacher exists anywhere in the loop).

Episode contract (one episode = one reference segment):

* reset to the segment's start frame (``qpos[k0]`` + its finite-difference
  velocities), with curriculum-controlled initial-condition noise;
* one control step per reference frame (the references are 50 Hz, the env is
  50 Hz); the action is the env's RESIDUAL unit action;
* the residual BASE ACTION is the reference's next-frame joint targets
  (``set_base_action(q_ref[k+1])``), so ``z = 0`` is exact open-loop reference
  replay in joint space and the policy adds the balance/contact corrections;
* reward = root/site/yaw tracking (movement accomplishment) + a SOFT joint
  prior (appearance) x an uprightness gate - validity penalties (action rate,
  torque saturation, joint limits, loaded-foot slide);
* termination: the env's dorsal/fall verdicts, or TRACKING DEVIATION (the
  anti-gaming closure: collapsing below the reference while it stands is
  terminal exactly like a fall, so "never fall by never standing" is not an
  optimum; thresholds are part of the frozen weights).

Anchoring: the commanded root path is the reference's planar displacement from
the segment start, translated to the robot's actual reset position (so reset
jitter is not punished and no world position enters the observation); root
height and yaw targets stay absolute.  Root travel is therefore still
world-anchored: the robot must really cover the reference's displacement
(no treadmill behaviour can score).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .bc import load_reference
from .commands import Command, CommandSchedule, N_SKILLS, Skill
from .env import SoloEnv
from .fall import contact_state
from .imitation import (ImitationState, ImitationTargets, ImitationWeights,
                        joint_pose_error, site_error_m, state_from_env)
from .obs import ACTOR_DIM, PRIV_DIM
from .pushes import PushSchedule, PushSpec
from .scene import N_JOINTS, load_solo_model

_DT = 0.02  # reference / control rate (s) -- both 50 Hz

# ---------------------------------------------------------------- reference
#: take file -> tracking skill label (drill phases have their own table).
#: ``CIRCLE`` is resolved to CIRCLE_L/CIRCLE_R from the lead leg at build time.
TAKE_LABELS: dict[str, str] = {
    "stance_hold": "STANCE",
    "stance_widen_step": "SHUFFLE_F",
    "stand_to_stance": "LEVEL_CHANGE",
    "stance_to_stand": "LEVEL_CHANGE",
    "stance_rise": "LEVEL_CHANGE",
    "shuffle_back": "SHUFFLE_B",
    "stalk_shuffle": "SHUFFLE_F",
    "circle_step": "CIRCLE",
    "level_change_fast": "LEVEL_CHANGE",
    "level_change_full": "LEVEL_CHANGE",
    "shot_entry_full": "SHOT_DOUBLE_LEG",
    "shot_recover": "RECOVER",
    "knee_sprawl_entry": "SHOT_DOUBLE_LEG",
    "knee_sprawl_entry2": "SHOT_DOUBLE_LEG",
    "knee_sprawl_hold": "RECOVER",
    "knee_sprawl_recover": "RECOVER",
}

#: drill phase name -> tracking skill label
DRILL_LABELS: dict[str, str] = {
    "STAND": "STANCE",
    "LOWER_TO_STANCE": "LEVEL_CHANGE",
    "STANCE_HOLD": "STANCE",
    "SHUFFLE_F": "SHUFFLE_F",
    "SHUFFLE_B": "SHUFFLE_B",
    "CIRCLE": "CIRCLE",
    "LEVEL_CHANGE": "LEVEL_CHANGE",
    "DOUBLE_LEG_ENTRY_CROUCH": "SHOT_DOUBLE_LEG",
    "DOUBLE_LEG_PENETRATION": "SHOT_DOUBLE_LEG",
    "RECOVER_TO_STANCE": "RECOVER",
    "REPOSITION": "APPROACH",
    "REPEAT_BLEND": "STANCE",
    # stance_rise.npz (composed 2-phase file)
    "STANCE": "STANCE",
    "RISE_TO_STAND": "LEVEL_CHANGE",
}

#: takes NEVER trained on (zero-shot generalisation probes for the report)
HELD_OUT_TAKES: tuple[str, ...] = ("stalk_shuffle", "knee_sprawl_entry2")

#: the 7-stage capability curriculum (SOLO_DRILL / brief stage list)
STAGE_ORDER: tuple[str, ...] = (
    "S1_stand_lower_hold_rise", "S2_first_step", "S3_footwork",
    "S4_level_change", "S5_entry_recovery", "S6_connected_drill",
    "S7_robustness",
)
#: stage -> (ic joint noise rad, xy noise m, yaw jitter deg, start-frame jitter)
STAGE_CONDITIONS: dict[str, tuple[float, float, float, int]] = {
    "S1_stand_lower_hold_rise": (0.00, 0.0, 0.0, 15),
    "S2_first_step": (0.01, 0.005, 2.0, 15),
    "S3_footwork": (0.01, 0.01, 3.0, 20),
    "S4_level_change": (0.01, 0.01, 3.0, 15),
    "S5_entry_recovery": (0.01, 0.01, 3.0, 20),
    "S6_connected_drill": (0.005, 0.005, 2.0, 0),
    "S7_robustness": (0.02, 0.015, 5.0, 60),
}


def phase_label(name: str) -> tuple[str, bool]:
    """(skill label, connect flag) for a drill phase name (``CONNECT_*`` too)."""
    if name.startswith("CONNECT_"):
        rest = name[len("CONNECT_"):]
        target = rest.split("->")[-1] if "->" in rest else rest
        label = DRILL_LABELS.get(target, "STANCE")
        return label, True
    if name in DRILL_LABELS:
        return DRILL_LABELS[name], False
    raise KeyError(f"unknown drill phase {name!r}")


def _skill_id(label: str, lead: int) -> int:
    if label == "CIRCLE":
        return int(Skill.CIRCLE_R if lead > 0 else Skill.CIRCLE_L)
    try:
        return int(Skill[label])
    except KeyError as exc:  # pragma: no cover - guarded by the tables
        raise KeyError(f"unknown skill label {label!r}") from exc


# ----------------------------------------------------------------- segments
@dataclass(frozen=True)
class Segment:
    """One trainable window of one reference track (frames ``[k0, k1)``)."""

    source: str
    k0: int
    k1: int
    label: str
    lead: int
    connect: bool = False
    validity: str = "unverified"
    weight: float = 1.0

    @property
    def name(self) -> str:
        return f"{self.source}[{self.k0}:{self.k1}]"

    @property
    def duration_s(self) -> float:
        return (self.k1 - self.k0) * _DT


def _lead_int(lead) -> int:
    if lead is None:
        return 1
    if isinstance(lead, str):
        return -1 if lead.lower().startswith("l") else 1
    return int(lead)


def take_segment(source: str, *, label: str | None = None) -> Segment:
    """The whole take as one segment (label from :data:`TAKE_LABELS`)."""
    tr = load_reference(source)
    lead = _lead_int(tr.meta.get("lead_leg", 1))
    lab = label or TAKE_LABELS[source]
    if lab == "CIRCLE":
        lab = "CIRCLE_R" if lead > 0 else "CIRCLE_L"
    return Segment(source=source, k0=0, k1=len(tr), label=lab, lead=lead,
                   validity=str(tr.meta.get("validity", "unverified")))


def drill_segments(source: str = "drill_continuous", *, only: tuple[str, ...] = (),
                   drop_connect: bool = False, min_s: float = 0.0) -> list[Segment]:
    """Segments for every drill phase (frame indices from the ``phase_id`` runs)."""
    tr = load_reference(source)
    z = np.load(tr.source, allow_pickle=True)
    pid = np.asarray(z["phase_id"], dtype=int)
    meta = tr.meta or {}
    phases = meta.get("phases", [])
    out: list[Segment] = []
    for p in phases:
        name = p["name"]
        lab, connect = phase_label(name)
        if drop_connect and connect:
            continue
        if only and name not in only:
            continue
        frames = np.flatnonzero(pid == int(p["id"]))
        if frames.size == 0:
            continue
        k0, k1 = int(frames[0]), int(frames[-1]) + 1
        if (k1 - k0) * _DT < min_s:
            continue
        lead = _lead_int(p.get("lead_leg"))
        if lab == "CIRCLE":
            lab = "CIRCLE_R" if lead > 0 else "CIRCLE_L"
        out.append(Segment(source=source, k0=k0, k1=k1, label=lab, lead=lead,
                           connect=connect, validity=str(p.get("validity", "unverified"))))
    return out


def stage_segments(stage: str) -> list[Segment]:
    """The segment list of one curriculum stage (see :data:`STAGE_ORDER`)."""
    if stage not in STAGE_ORDER:
        raise KeyError(f"unknown stage {stage!r}; expected one of {STAGE_ORDER}")
    drill = "drill_continuous"
    if stage == "S1_stand_lower_hold_rise":
        segs = drill_segments(only=("STAND", "LOWER_TO_STANCE", "STANCE_HOLD"))
        segs += [take_segment("stance_hold"), take_segment("stand_to_stance"),
                 take_segment("stance_rise")]
    elif stage == "S2_first_step":
        shuffle = [s for s in drill_segments(only=("SHUFFLE_F",))
                   if s.duration_s > 3.0]
        segs = ([Segment(s.source, s.k0, s.k0 + 200, s.label, s.lead,
                         s.connect, s.validity) for s in shuffle]
                + [take_segment("stance_widen_step")])
    elif stage == "S3_footwork":
        segs = drill_segments(only=("SHUFFLE_F", "SHUFFLE_B", "CIRCLE"), min_s=2.0)
        segs += [take_segment("shuffle_back"), take_segment("circle_step")]
    elif stage == "S4_level_change":
        segs = drill_segments(only=("LEVEL_CHANGE",))
        segs += [take_segment("level_change_fast"), take_segment("level_change_full")]
    elif stage == "S5_entry_recovery":
        segs = drill_segments(only=("DOUBLE_LEG_ENTRY_CROUCH", "DOUBLE_LEG_PENETRATION",
                                    "RECOVER_TO_STANCE"))
        segs += [take_segment("shot_entry_full"), take_segment("shot_recover"),
                 take_segment("knee_sprawl_entry"), take_segment("knee_sprawl_hold"),
                 take_segment("knee_sprawl_recover")]
    elif stage == "S6_connected_drill":
        segs = drill_segments(drop_connect=True)
        segs = [s for s in segs if not s.connect]
    else:  # S7_robustness: the whole drill including connectors
        segs = [Segment(drill, 0, len(load_reference(drill)), "STANCE", 1,
                        validity="composed")]
    return segs


# ------------------------------------------------------------------ weights
@dataclass(frozen=True)
class TrackWeights:
    """Frozen tracking reward weights + deviation thresholds (see module doc).

    Priorities (brief): physical validity > movement accomplishment >
    wrestling appearance.  The imitation kernels (site RMS 0.05 m, joint
    0.35 rad) are reused from :class:`solo.imitation.ImitationWeights` so the
    tracking numbers stay comparable with the reference-fidelity baselines.
    """

    im: ImitationWeights = ImitationWeights()
    # movement accomplishment
    w_root: float = 0.6
    sigma_root_xy: float = 0.08          # m: tight enough to see drift early
    #: (the LOWER failure drifts 0.156 m in 0.36 s), loose enough to permit the
    #: balance-mandated fore-aft modifications the brief sanctions
    sigma_root_z: float = 0.04           # m (shallow phases)
    sigma_root_z_deep: float = 0.12      # m (reference pelvis below deep_z_below)
    deep_z_below: float = 0.55           # m
    w_site: float = 1.0                  # the PRIMARY tracking term (landmarks)
    w_yaw: float = 0.2
    sigma_yaw: float = 0.25              # rad
    w_root_vel: float = 0.2              # root velocity tracking (timing signal)
    sigma_root_vel: float = 0.15         # m/s
    w_joint_vel: float = 0.1             # joint velocity tracking
    sigma_joint_vel: float = 1.5         # rad/s (the imitation kernel)
    # wrestling appearance (soft prior, never the top objective)
    w_joint: float = 0.2
    # validity penalties (all in [0,1] x weight)
    pen_action: float = 0.5
    action_norm: float = 0.30            # rad/step at which the penalty is 1
    pen_torque: float = 0.1
    pen_limit: float = 0.1
    pen_slide: float = 0.2
    alive_gate: bool = True              # tracking positives x clamp01(torso_up_z)
    # one-off
    completion_bonus: float = 5.0        # survived to the segment end
    terminal_penalty: float = 20.0       # fall / dorsal / deviation
    # deviation termination (the anti-gaming closure) -- pelvis drop is the
    # crouch-escape killer: the robot may not sit below the reference while
    # the reference stands.
    term_joint_rad: float = 0.60         # mean |joint - ref|
    term_root_xy_m: float = 0.35         # anchored horizontal root error
    term_pelvis_drop_m: float = 0.20     # ref z - robot z (robot BELOW ref)
    term_site_m: float = 0.15            # weighted landmark RMS

    def as_dict(self) -> dict:
        return {"im": self.im.as_dict(),
                **{k: getattr(self, k) for k in (
                    "w_root", "sigma_root_xy", "sigma_root_z", "sigma_root_z_deep",
                    "deep_z_below", "w_site", "w_yaw", "sigma_yaw",
                    "w_root_vel", "sigma_root_vel", "w_joint_vel",
                    "sigma_joint_vel", "w_joint",
                    "pen_action", "action_norm", "pen_torque", "pen_limit",
                    "pen_slide", "alive_gate", "completion_bonus",
                    "terminal_penalty", "term_joint_rad", "term_root_xy_m",
                    "term_pelvis_drop_m", "term_site_m")}}


DEFAULT_WEIGHTS = TrackWeights()


def track_reward_terms(*, joint_err: float, site_err: float, root_xy_err: float,
                       root_z_err: float, yaw_err: float, ref_pelvis_z: float,
                       torso_up_z: float, action_delta_mean: float,
                       sat_frac: float, limit_prox: float, slide_frac: float,
                       root_vel_err: float = 0.0, joint_vel_err: float = 0.0,
                       w: TrackWeights = DEFAULT_WEIGHTS) -> dict[str, float]:
    """All tracking reward terms on CONSTRUCTED quantities (unit-testable).

    ``site_err`` is the class-weighted landmark RMS (m); ``joint_err`` the mean
    absolute joint error (rad); ``root_*_err`` the anchored root errors (m);
    ``yaw_err`` the wrapped yaw error (rad); ``slide_frac`` the loaded-foot
    slide proxy in [0, 1] (velocity fraction of the 2 m/s normaliser);
    ``root_vel_err`` / ``joint_vel_err`` the velocity-tracking RMS (m/s, rad/s).
    """
    sigma_z = w.sigma_root_z_deep if ref_pelvis_z < w.deep_z_below else w.sigma_root_z
    root = math.exp(-((root_xy_err / w.sigma_root_xy) ** 2
                      + (root_z_err / sigma_z) ** 2))
    yaw = math.exp(-(yaw_err / w.sigma_yaw) ** 2)
    joint = math.exp(-(joint_err / w.im.joint_pose_scale_rad) ** 2)
    site = math.exp(-(site_err / w.im.site_scale_m) ** 2)
    root_vel = math.exp(-(root_vel_err / w.sigma_root_vel) ** 2)
    joint_vel = math.exp(-(joint_vel_err / w.sigma_joint_vel) ** 2)
    gate = float(np.clip(torso_up_z, 0.0, 1.0)) if w.alive_gate else 1.0
    pos = (w.w_root * root + w.w_site * site + w.w_yaw * yaw + w.w_joint * joint
           + w.w_root_vel * root_vel + w.w_joint_vel * joint_vel) * gate
    pen = (w.pen_action * float(np.clip(action_delta_mean / w.action_norm, 0, 1))
           + w.pen_torque * float(np.clip(sat_frac, 0, 1))
           + w.pen_limit * float(np.clip(limit_prox, 0, 1))
           + w.pen_slide * float(np.clip(slide_frac, 0, 1)))
    return {
        "root": root, "site": site, "yaw": yaw, "joint": joint,
        "root_vel": root_vel, "joint_vel": joint_vel,
        "w_root": w.w_root * root * gate, "w_site": w.w_site * site * gate,
        "w_yaw": w.w_yaw * yaw * gate, "w_joint": w.w_joint * joint * gate,
        "w_root_vel": w.w_root_vel * root_vel * gate,
        "w_joint_vel": w.w_joint_vel * joint_vel * gate,
        "pen_action": w.pen_action * float(np.clip(action_delta_mean / w.action_norm, 0, 1)),
        "pen_torque": w.pen_torque * float(np.clip(sat_frac, 0, 1)),
        "pen_limit": w.pen_limit * float(np.clip(limit_prox, 0, 1)),
        "pen_slide": w.pen_slide * float(np.clip(slide_frac, 0, 1)),
        "gate": gate, "penalty": pen, "total": pos - pen,
    }


def track_deviation(*, joint_err: float, root_xy_err: float, pelvis_drop_m: float,
                    site_err: float, w: TrackWeights = DEFAULT_WEIGHTS) -> str | None:
    """First exceeded deviation threshold (strictly greater), or ``None``.

    Same semantics as :func:`solo.imitation.deviation_reason` but on the
    ANCHORED root error (see module docstring) and with the pelvis-drop
    threshold tightened to the crouch-escape bar.
    """
    if joint_err > w.term_joint_rad:
        return "joint"
    if root_xy_err > w.term_root_xy_m:
        return "root_xy"
    if pelvis_drop_m > w.term_pelvis_drop_m:
        return "pelvis_drop"
    if site_err > w.term_site_m:
        return "site"
    return None


# ------------------------------------------------------------------ targets
@dataclass
class TrackTargets:
    """Per-frame tracking targets of one reference npz (see module docstring)."""

    name: str
    source: str
    targets: ImitationTargets
    qpos: np.ndarray           # (T, 36)
    contact: np.ndarray        # (T, 2) u8 planted flags
    root_xy: np.ndarray        # (T, 2) world root position
    root_z: np.ndarray         # (T,)
    yaw: np.ndarray            # (T,) world root yaw
    linvel_world: np.ndarray   # (T, 3) world root linear velocity
    #: per-frame conditioning (drill: from the labelled phases; takes: constant)
    skill_ids: np.ndarray | None = None   # (T,) Skill ids per frame
    connect_flags: np.ndarray | None = None  # (T,) bool synthetic-connector
    leads: np.ndarray | None = None       # (T,) +1 right / -1 left per frame
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, source, model=None) -> "TrackTargets":
        from .imitation import _quat_to_R

        tr = load_reference(source)
        tg = ImitationTargets.from_reference(tr.source, model=model)
        qpos = np.asarray(tr.qpos, np.float64)
        z = np.load(tr.source, allow_pickle=True)
        if "contact" in z:
            contact = np.asarray(z["contact"], np.float64).reshape(len(qpos), 2)
        else:
            contact = np.ones((len(qpos), 2))
        yaw = np.zeros(len(qpos))
        lin = np.zeros((len(qpos), 3))
        for i in range(len(qpos)):
            R = _quat_to_R(qpos[i, 3:7])
            yaw[i] = math.atan2(R[1, 0], R[0, 0])
            lin[i] = R @ tg.base_linvel_local[i]
        meta = tr.meta or {}
        skill_ids = connect_flags = leads = None
        if "phase_id" in z and meta.get("phases"):
            # per-frame conditioning from the drill's own labelled phases
            pid = np.asarray(z["phase_id"], dtype=int)
            skill_ids = np.zeros(len(qpos), dtype=int)
            connect_flags = np.zeros(len(qpos), dtype=bool)
            leads = np.ones(len(qpos), dtype=int)
            for p in meta["phases"]:
                frames = pid == int(p["id"])
                if not frames.any():
                    continue
                lab, connect = phase_label(p["name"])
                lead = _lead_int(p.get("lead_leg"))
                if lab == "CIRCLE":
                    lab = "CIRCLE_R" if lead > 0 else "CIRCLE_L"
                skill_ids[frames] = _skill_id(lab, lead)
                connect_flags[frames] = connect
                leads[frames] = lead
        else:
            lead = _lead_int(meta.get("lead_leg", 1))
            lab = TAKE_LABELS.get(tr.name)
            if lab:
                if lab == "CIRCLE":
                    lab = "CIRCLE_R" if lead > 0 else "CIRCLE_L"
                skill_ids = np.full(len(qpos), _skill_id(lab, lead), dtype=int)
                connect_flags = np.zeros(len(qpos), dtype=bool)
                leads = np.full(len(qpos), lead, dtype=int)
        return cls(name=tr.name, source=tr.source, targets=tg, qpos=qpos,
                   contact=contact, root_xy=qpos[:, :2].copy(),
                   root_z=qpos[:, 2].copy(), yaw=yaw, linvel_world=lin,
                   skill_ids=skill_ids, connect_flags=connect_flags,
                   leads=leads, meta=meta)

    def __len__(self) -> int:
        return len(self.qpos)


_TARGETS_CACHE: dict[str, TrackTargets] = {}


def track_targets(source, model=None) -> TrackTargets:
    """Cached :meth:`TrackTargets.load` (FK preprocessing runs once per npz)."""
    path = str(source)
    if path not in _TARGETS_CACHE:
        _TARGETS_CACHE[path] = TrackTargets.load(source, model=model)
    return _TARGETS_CACHE[path]


def clear_target_cache() -> None:
    _TARGETS_CACHE.clear()


# ---------------------------------------------------------- reference block
#: appended actor-observation layout (offsets relative to ``ACTOR_DIM``)
REF_LAYOUT: tuple[tuple[str, tuple[int, int]], ...] = (
    ("ref_root_local", (0, 3)),        # commanded root target - pelvis, pelvis frame
    ("ref_root_vel_local", (3, 6)),    # reference root velocity, heading frame
    ("ref_joints_rel", (6, 6 + N_JOINTS)),
    ("ref_contacts", (6 + N_JOINTS, 8 + N_JOINTS)),
    ("ref_phase", (8 + N_JOINTS, 9 + N_JOINTS)),
    ("ref_skill_onehot", (9 + N_JOINTS, 9 + N_JOINTS + N_SKILLS)),
    ("ref_lead", (9 + N_JOINTS + N_SKILLS, 10 + N_JOINTS + N_SKILLS)),
    ("ref_connect", (10 + N_JOINTS + N_SKILLS, 11 + N_JOINTS + N_SKILLS)),
    ("ref_root_ahead_local", (11 + N_JOINTS + N_SKILLS,
                              14 + N_JOINTS + N_SKILLS)),
    # the contact-observation fix (measured contact-blindness at the gait
    # transfer switches): the ACTUAL per-foot contact of the robot, and the
    # reference's contact NEXT_CONTACT_FRAMES ahead (the switch is visible
    # before it happens).  2 + 2 dims.
    ("own_contacts", (14 + N_JOINTS + N_SKILLS, 16 + N_JOINTS + N_SKILLS)),
    ("ref_contacts_next", (16 + N_JOINTS + N_SKILLS, 18 + N_JOINTS + N_SKILLS)),
)
REF_DIM = REF_LAYOUT[-1][1][1]
REF_ACTOR_DIM = ACTOR_DIM + REF_DIM

#: future-root horizon of ``ref_root_ahead_local`` (frames; 10 = 0.2 s)
AHEAD_FRAMES = 10

#: horizon of ``ref_contacts_next`` (frames; 5 = 0.1 s at the 50 Hz clock)
NEXT_CONTACT_FRAMES = 5

#: appended privileged block (tracking errors; critic-only)
REF_PRIV_LAYOUT: tuple[tuple[str, tuple[int, int]], ...] = (
    ("ref_err_joint", (0, 1)),
    ("ref_err_site", (1, 2)),
    ("ref_err_root_xy", (2, 3)),
)
REF_PRIV_DIM = REF_PRIV_LAYOUT[-1][1][1]
REF_CRITIC_DIM = REF_ACTOR_DIM + PRIV_DIM + REF_PRIV_DIM


def ref_block(*, root_local: np.ndarray, root_vel_local: np.ndarray,
              joints_rel: np.ndarray, contacts: np.ndarray, phase: float,
              skill_id: int, lead: int, connect: bool,
              root_ahead_local: np.ndarray, own_contacts: np.ndarray,
              contacts_next: np.ndarray) -> np.ndarray:
    """(REF_DIM,) float32 reference block (pure; unit-testable)."""
    parts = [
        np.asarray(root_local, np.float64).reshape(3),
        np.asarray(root_vel_local, np.float64).reshape(3),
        np.asarray(joints_rel, np.float64).reshape(N_JOINTS),
        np.clip(np.asarray(contacts, np.float64).reshape(2), 0.0, 1.0),
        np.array([float(np.clip(phase, 0.0, 1.0))]),
        _onehot(int(skill_id), N_SKILLS),
        np.array([float(np.sign(lead) or 1.0)]),
        np.array([1.0 if connect else 0.0]),
        np.asarray(root_ahead_local, np.float64).reshape(3),
        np.clip(np.asarray(own_contacts, np.float64).reshape(2), 0.0, 1.0),
        np.clip(np.asarray(contacts_next, np.float64).reshape(2), 0.0, 1.0),
    ]
    out = np.concatenate(parts).astype(np.float32)
    assert out.size == REF_DIM, out.size
    return out


def ref_priv_block(*, joint_err: float, site_err: float,
                   root_xy_err: float) -> np.ndarray:
    """(REF_PRIV_DIM,) float32 critic-only tracking errors."""
    return np.array([float(joint_err), float(site_err), float(root_xy_err)],
                    dtype=np.float32)


def _onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float64)
    v[int(idx) % n] = 1.0
    return v


def wrap_angle(a: float) -> float:
    return float((float(a) + math.pi) % (2.0 * math.pi) - math.pi)


# --------------------------------------------------------------------- env
def _yaw_of(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_rot(delta: float) -> np.ndarray:
    c, s = math.cos(delta), math.sin(delta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class TrackingEnv:
    """One reference-tracking episode around a live :class:`solo.env.SoloEnv`.

    ``deviation_mode="hard"`` ends the episode at the first deviation (the
    final evaluation semantics).  ``"soft"`` pays a per-step penalty instead
    and keeps the episode alive: early in a stage a hard gate ends every
    episode at ~0.6 s, so the policy never experiences the recovery/rise part
    of the motion and cannot learn to correct it (the same trap the BC
    refinement hit; falls and dorsal contact are ALWAYS terminal).
    """

    def __init__(self, env: SoloEnv, seg: Segment, tt: TrackTargets, *,
                 weights: TrackWeights = DEFAULT_WEIGHTS, q_stand: np.ndarray,
                 ic_noise: float = 0.0, xy_noise: float = 0.0,
                 yaw_jitter_deg: float = 0.0, seed: int = 0,
                 ahead_frames: int = AHEAD_FRAMES,
                 deviation_mode: str = "hard", soft_penalty: float = 2.0,
                 clock_mode: str = "fixed",
                 gate_joint_rad: float = 0.30, gate_root_xy_m: float = 0.12):
        self.env = env
        self.seg = seg
        self.tt = tt
        self.w = weights
        self.q_stand = np.asarray(q_stand, np.float64)
        self.ic_noise = float(ic_noise)
        self.xy_noise = float(xy_noise)
        self.yaw_jitter = math.radians(float(yaw_jitter_deg))
        self.rng = np.random.default_rng(int(seed))
        self.ahead_frames = int(ahead_frames)
        if deviation_mode not in ("hard", "soft"):
            raise ValueError(deviation_mode)
        self.deviation_mode = deviation_mode
        self.soft_penalty = float(soft_penalty)
        if clock_mode not in ("fixed", "gated"):
            raise ValueError(clock_mode)
        #: "gated": the reference frame advances only while the robot actually
        #: tracks it (joint err <= gate_joint_rad AND anchored root xy err <=
        #: gate_root_xy_m).  The v1 references are kinematic retargets whose
        #: TIMING is often dynamically infeasible even where the poses are not
        #: (E12: 6.9 % statically holdable) -- a wall-locked clock forces the
        #: policy to chase a target it cannot match and punishes the very
        #: deviation that balance requires.  A gated clock makes reference
        #: PROGRESS a tracked accomplishment: freezing is bounded (an episode
        #: that never reaches the end earns no completion bonus and is cut at
        #: 2x the reference duration), and "advancing phases without real
        #: motion" is impossible by construction.
        self.clock_mode = clock_mode
        self.gate_joint_rad = float(gate_joint_rad)
        self.gate_root_xy_m = float(gate_root_xy_m)
        self.N = seg.k1 - seg.k0
        self.k = 0
        self._steps = 0
        self.anchor_xy = np.zeros(2)
        self.last_terms: dict[str, float] = {}
        self.last_errs: dict[str, float] = {}

    # ------------------------------------------------------------- targets
    def _target_root(self, kf: int) -> np.ndarray:
        """Anchored commanded root position (world) at reference frame ``kf``."""
        d = self.tt.root_xy[kf] - self.tt.root_xy[self.seg.k0]
        return np.array([self.anchor_xy[0] + d[0], self.anchor_xy[1] + d[1],
                         self.tt.root_z[kf]])

    def _root_err(self, kf: int, pelvis_xyz: np.ndarray,
                  yaw_cur: float) -> tuple[float, float]:
        """(anchored horizontal root error in the heading frame [m],
        signed root-z error ref - robot [m]) at reference frame ``kf``."""
        tgt = self._target_root(kf)
        c, s = math.cos(yaw_cur), math.sin(yaw_cur)
        dx, dy = tgt[0] - pelvis_xyz[0], tgt[1] - pelvis_xyz[1]
        return (math.hypot(c * dx + s * dy, -s * dx + c * dy),
                float(tgt[2] - pelvis_xyz[2]))

    def reset(self, seed: int | None = None, *, k0: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        k0 = self.seg.k0 if k0 is None else int(k0)
        q0 = np.asarray(self.tt.qpos[k0], np.float64).copy()
        yaw_delta = float(self.rng.uniform(-self.yaw_jitter, self.yaw_jitter)) \
            if self.yaw_jitter > 0 else 0.0
        if self.ic_noise > 0.0:
            lo, hi = self.env.lo, self.env.hi
            q0[7:36] = np.clip(q0[7:36] + self.rng.normal(0.0, self.ic_noise, N_JOINTS),
                               lo, hi)
        if self.xy_noise > 0.0:
            q0[0] += float(self.rng.uniform(-self.xy_noise, self.xy_noise))
            q0[1] += float(self.rng.uniform(-self.xy_noise, self.xy_noise))
        if yaw_delta != 0.0:
            # compose the jitter about Z ON TOP of the reference orientation:
            # replacing the quat would wipe the reference's pelvis pitch/roll
            # (a deep stance leans; a pure-yaw quat is a different pose)
            w0, x0, y0, z0 = q0[3:7]
            hd, zd = math.cos(yaw_delta / 2), math.sin(yaw_delta / 2)
            q0[3:7] = np.array([
                hd * w0 - zd * z0,
                hd * x0 - zd * y0,
                hd * y0 + zd * x0,
                hd * z0 + zd * w0])
        self.env.reset(pose=q0, jitter=False,
                       command=CommandSchedule.steady(Command(
                           skill_id=_skill_id(self.seg.label, self.seg.lead),
                           lead_leg=int(self.seg.lead))))
        # seed the measured reference velocity (rotated with the pose jitter)
        Rz = _yaw_rot(yaw_delta)
        self.env.data.qvel[0:3] = Rz @ self.tt.linvel_world[k0]
        _, ang_w = self._ref_angvel(k0)
        self.env.data.qvel[3:6] = Rz @ ang_w
        self.env.data.qvel[6:35] = self.tt.targets.joint_vel[k0]
        from mujoco import mj_forward

        mj_forward(self.env.model, self.env.data)
        self.anchor_xy = np.asarray(self.env.data.qpos[0:2], np.float64).copy()
        self.k = 0
        self.env.set_base_action(np.clip(self.tt.qpos[k0 + 1, 7:36],
                                         self.env.lo, self.env.hi))
        return self.observation()

    def _ref_angvel(self, kf: int) -> tuple[np.ndarray, np.ndarray]:
        """Reference world-frame (lin, ang) velocity at frame ``kf``."""
        lin = self.tt.linvel_world[kf]
        kfp, kfn = max(kf - 1, 0), min(kf + 1, len(self.tt) - 1)
        wz = wrap_angle(self.tt.yaw[kfn] - self.tt.yaw[kfp]) / max(
            (kfn - kfp) * _DT, 1e-9)
        return lin, np.array([0.0, 0.0, float(wz)])

    # ---------------------------------------------------------------- obs
    def _feat(self) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """(REF_DIM block, REF_PRIV block, error dict) for the CURRENT state."""
        kf = min(self.seg.k0 + self.k, len(self.tt) - 1)
        pelvis = np.asarray(self.env.data.qpos[0:3], np.float64)
        yaw_cur = _yaw_of(np.asarray(self.env.data.qpos[3:7], np.float64))
        xy_err, z_err_signed = self._root_err(kf, pelvis, yaw_cur)
        cur: ImitationState = state_from_env(self.env)
        ref = self.tt.targets.at(kf)
        joint_err = joint_pose_error(cur, ref)
        site_err = site_error_m(cur, ref)
        yaw_err = wrap_angle(self.tt.yaw[kf] - yaw_cur)
        # local-frame features (pelvis frame for position, heading frame for vel)
        tgt = self._target_root(kf)
        R = np.asarray(self.env.data.xmat[self.env._pelvis_bid],
                       np.float64).reshape(3, 3)
        root_local = R.T @ (tgt - pelvis)
        yawc, syc = math.cos(yaw_cur), math.sin(yaw_cur)
        lin_w, ang_w = self._ref_angvel(kf)
        root_vel_local = np.array([yawc * lin_w[0] + syc * lin_w[1],
                                   -syc * lin_w[0] + yawc * lin_w[1], lin_w[2]])
        kfa = min(kf + self.ahead_frames, len(self.tt) - 1)
        tgta = self._target_root(kfa)
        root_ahead_local = R.T @ (tgta - pelvis)
        joints_rel = np.asarray(self.tt.qpos[kf, 7:36], np.float64) - self.q_stand[7:36]
        if self.tt.skill_ids is not None:
            skill_id = int(self.tt.skill_ids[kf])
            lead = int(self.tt.leads[kf]) if self.tt.leads is not None else self.seg.lead
            connect = bool(self.tt.connect_flags[kf]) \
                if self.tt.connect_flags is not None else self.seg.connect
        else:
            skill_id = _skill_id(self.seg.label, self.seg.lead)
            lead, connect = self.seg.lead, self.seg.connect
        block = ref_block(root_local=root_local, root_vel_local=root_vel_local,
                          joints_rel=joints_rel, contacts=self.tt.contact[kf],
                          phase=self.k / max(self.N - 1, 1),
                          skill_id=skill_id, lead=lead, connect=connect,
                          root_ahead_local=root_ahead_local,
                          own_contacts=self._own_contacts(),
                          contacts_next=self.tt.contact[min(
                              kf + NEXT_CONTACT_FRAMES, len(self.tt) - 1)])
        priv = ref_priv_block(joint_err=joint_err, site_err=site_err, root_xy_err=xy_err)
        # velocity tracking errors (timing signal): root velocity in the
        # heading frame vs the reference's, and joint-velocity RMS
        v_w = np.asarray(self.env.data.qvel[0:3], np.float64)
        v_local = np.array([yawc * v_w[0] + syc * v_w[1],
                            -syc * v_w[0] + yawc * v_w[1], v_w[2]])
        root_vel_err = float(np.linalg.norm(v_local - root_vel_local))
        jv_robot = np.asarray(self.env.data.qvel[6:35], np.float64)
        joint_vel_err = float(np.sqrt(np.mean(
            (jv_robot - self.tt.targets.joint_vel[kf]) ** 2)))
        errs = {"joint_err": joint_err, "site_err": site_err, "root_xy_err": xy_err,
                "root_z_err": z_err_signed, "yaw_err": yaw_err,
                "root_vel_err": root_vel_err, "joint_vel_err": joint_vel_err,
                "pelvis_drop": max(0.0, z_err_signed), "ref_pelvis_z":
                    float(self.tt.root_z[kf]), "frame": kf}
        return block, priv, errs

    def _own_contacts(self) -> tuple[float, float]:
        """The robot's ACTUAL per-foot contact, fresh from the shared
        ContactState source (fall.contact_state) — not the env's per-step
        cache, so episode-start observations are not stale."""
        c = contact_state(self.env.model, self.env.data)
        return (float(bool(c.left_foot)), float(bool(c.right_foot)))

    def observation(self) -> dict:
        base = self.env.observation()
        block, priv, errs = self._feat()
        self.last_errs = errs  # the gated clock reads the latest tracking
        actor = np.concatenate([base["actor"], block]).astype(np.float32)
        critic = np.concatenate([base["actor"], block, base["privileged"],
                                 priv]).astype(np.float32)
        return {"actor": actor, "privileged": base["privileged"],
                "critic": critic, "ref_block": block, "ref_priv": priv,
                "errs": errs}

    # ---------------------------------------------------------------- step
    def step(self, unit_action) -> tuple[dict, float, bool, bool, dict]:
        if self.k >= self.N - 1:
            raise RuntimeError("TrackingEnv.step past the segment end; reset()")
        kf = min(self.seg.k0 + self.k + 1, len(self.tt) - 1)
        self.env.set_base_action(np.clip(self.tt.qpos[kf, 7:36],
                                         self.env.lo, self.env.hi))
        prev_ctrl = (None if self.env._prev_ctrl is None
                     else np.asarray(self.env._prev_ctrl, np.float64))
        obs, _env_r, terminated, truncated, info = self.env.step(unit_action)
        self._steps += 1
        clock_held = False
        # gated clock: reference progress is EARNED by tracking; when held,
        # the base action re-serves the SAME reference frame so the robot can
        # catch up instead of chasing an unreachable moving target
        if self.clock_mode == "gated" and self.last_errs:
            e = self.last_errs
            if (e["joint_err"] > self.gate_joint_rad
                    or e["root_xy_err"] > self.gate_root_xy_m):
                clock_held = True
            else:
                self.k += 1
        else:
            self.k += 1
        block, priv, errs = self._feat()
        ctrl = np.asarray(self.env.data.ctrl, np.float64)
        action_delta = 0.0 if prev_ctrl is None else float(
            np.mean(np.abs(ctrl - prev_ctrl)))
        contacts = info.get("contacts", {})
        loaded = (bool(contacts.get("left_foot")), bool(contacts.get("right_foot")))
        slip = info.get("metrics", {}).get("slip", 0.0)
        slide_frac = float(np.clip(slip / 2.0, 0.0, 1.0)) if any(loaded) else 0.0
        up_z = float(np.clip(np.asarray(self.env.data.xmat[self.env._torso_bid],
                                        np.float64).reshape(3, 3)[2, 2], -1, 1))
        terms = track_reward_terms(
            joint_err=errs["joint_err"], site_err=errs["site_err"],
            root_xy_err=errs["root_xy_err"], root_z_err=abs(errs["root_z_err"]),
            yaw_err=abs(errs["yaw_err"]), ref_pelvis_z=errs["ref_pelvis_z"],
            torso_up_z=up_z, action_delta_mean=action_delta,
            sat_frac=float(info.get("metrics", {}).get("sat_frac", 0.0)),
            limit_prox=float(info.get("metrics", {}).get("limit_prox", 0.0)),
            slide_frac=slide_frac, root_vel_err=errs["root_vel_err"],
            joint_vel_err=errs["joint_vel_err"], w=self.w)
        reward = float(terms["total"])
        cause = info.get("termination")
        truncated = bool(truncated)
        terminated = bool(terminated)
        if cause is None:
            dev = track_deviation(joint_err=errs["joint_err"],
                                  root_xy_err=errs["root_xy_err"],
                                  pelvis_drop_m=errs["pelvis_drop"],
                                  site_err=errs["site_err"], w=self.w)
            if dev is not None and self.deviation_mode == "hard":
                cause = f"deviation:{dev}"
                terminated = True
            elif dev is not None:
                # soft mode: pay per-step, stay alive (falls/dorsal stay hard)
                reward -= self.soft_penalty
                terms["pen_deviation_soft"] = self.soft_penalty
        success = False
        if cause is not None:
            reward -= self.w.terminal_penalty
            terminated = True
        elif self._steps >= 2 * self.N and self.clock_mode == "gated":
            # anti-freeze bound: the reference end was never reached within
            # twice the reference duration -> no completion credit
            cause = "clock_budget"
            truncated = True
        elif self.k >= self.N - 1:
            reward += self.w.completion_bonus
            success = True
            truncated = True
        self.last_terms = terms
        self.last_errs = errs
        actor = np.concatenate([obs["actor"], block]).astype(np.float32)
        critic = np.concatenate([obs["actor"], block, obs["privileged"],
                                 priv]).astype(np.float32)
        out = {"actor": actor, "privileged": obs["privileged"], "critic": critic,
               "ref_block": block, "ref_priv": priv, "errs": errs}
        info = {**info, "track": {"segment": self.seg.name, "label": self.seg.label,
                                  "k": self.k, "frames": self.N,
                                  "cause": cause, "success": success,
                                  "clock_held": bool(clock_held),
                                  "steps": self._steps,
                                  "terms": terms, "errs": errs}}
        return out, float(reward), terminated, truncated, info


class TrackingTask:
    """Stage-sampling tracking task for PPO (auto-samples segments on reset)."""

    def __init__(self, *, stage: str, model=None, weights: TrackWeights | None = None,
                 residual_scale: float = 0.5, seed: int = 0,
                 conditions: tuple[float, float, float, int] | None = None,
                 push_schedule=None, record_metrics: bool = True,
                 stage_override: list[Segment] | None = None,
                 deviation_mode: str = "hard", soft_penalty: float = 2.0,
                 push_impulses: tuple[float, ...] | None = None,
                 clock_mode: str = "fixed"):
        from .scene import stand_frame

        if stage not in STAGE_ORDER:
            raise KeyError(stage)
        self.stage = stage
        self.model = model if model is not None else load_solo_model()
        self.w = weights or DEFAULT_WEIGHTS
        self.segs = stage_override if stage_override is not None \
            else stage_segments(stage)
        _w = np.array([max(s.weight, 1e-6) for s in self.segs])
        self._p = _w / _w.sum()
        cond = conditions or STAGE_CONDITIONS[stage]
        self.ic_noise, self.xy_noise, self.yaw_jitter, self.start_jitter = cond
        self.q_stand, _ = stand_frame(self.model)
        self.env = SoloEnv(self.model, task="balance", action_mode="residual",
                           residual_scale=float(residual_scale), seed=int(seed),
                           jitter=False, record_metrics=record_metrics,
                           horizon=8.0, push=push_schedule)
        self.rng = np.random.default_rng(int(seed))
        self.deviation_mode = deviation_mode
        self.soft_penalty = float(soft_penalty)
        #: scaled-disturbance mode (S7): per-episode random pushes drawn from
        #: these impulse magnitudes (N*s), 1-3 per episode at random times.
        self.push_impulses = tuple(push_impulses) if push_impulses else None
        self._dynamic_pushes = self.push_impulses is not None
        self.clock_mode = clock_mode
        self.ep = TrackingEnv(self.env, self.segs[0], track_targets(
            self.segs[0].source, self.model), weights=self.w, q_stand=self.q_stand,
            deviation_mode=deviation_mode, soft_penalty=self.soft_penalty,
            clock_mode=clock_mode)
        self.episode = 0

    def sample_segment(self) -> Segment:
        return self.segs[int(self.rng.choice(len(self.segs), p=self._p))]

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        seg = self.sample_segment()
        tt = track_targets(seg.source, self.model)
        n = seg.k1 - seg.k0
        if self.start_jitter > 0 and n > 3 * self.start_jitter:
            # mid-segment starts (BeyondMimic's keyframe-proximity resets): the
            # policy first learns to finish the tail from a live start, which
            # bootstraps the harder early frames; window = remainder of the
            # segment (capped at 300 frames).
            j = int(self.rng.integers(0, min(self.start_jitter, n // 3)))
            k0 = seg.k0 + j
            k1 = min(k0 + max(n - j, 1), seg.k1 if n <= 300 else k0 + 300)
            seg = Segment(seg.source, k0, k1, seg.label, seg.lead, seg.connect,
                          seg.validity, seg.weight)
        self.ep = TrackingEnv(self.env, seg, tt, weights=self.w,
                              q_stand=self.q_stand, ic_noise=self.ic_noise,
                              xy_noise=self.xy_noise,
                              yaw_jitter_deg=self.yaw_jitter,
                              seed=int(self.rng.integers(0, 2**31 - 1)),
                              deviation_mode=self.deviation_mode,
                              soft_penalty=self.soft_penalty,
                              clock_mode=self.clock_mode)
        self.env.horizon = (2.0 if self.clock_mode == "gated" else 1.0) \
            * seg.duration_s + 2.0
        if self._dynamic_pushes:
            n = int(self.rng.integers(1, 4))
            specs = []
            for i in range(n):
                mag = float(self.push_impulses[int(
                    self.rng.integers(0, len(self.push_impulses)))])
                specs.append(PushSpec(
                    t=float(self.rng.uniform(0.3, max(0.4, seg.duration_s - 0.3))),
                    impulse=mag, direction=float(self.rng.uniform(0, 2 * np.pi)),
                    heading_relative=True, label=f"rnd{i}"))
            self.env.set_push_schedule(PushSchedule(specs) if specs else None)
        self.episode += 1
        return self.ep.reset()

    def step(self, unit_action):
        return self.ep.step(unit_action)

    def set_deviation_mode(self, mode: str) -> None:
        """Switch hard/soft deviation for FUTURE episodes (annealed gating)."""
        if mode not in ("hard", "soft"):
            raise ValueError(mode)
        self.deviation_mode = mode

    @property
    def segment(self) -> Segment:
        return self.ep.seg


def warm_start_actor(net, ckpt: dict) -> dict:
    """First-layer surgery: a 115/158-dim checkpoint -> the 170/216-dim net.

    Copies every existing input column and ZERO-initialises the appended
    reference columns (the reference block starts as a no-op; the trunk and
    all deeper layers transfer untouched).  Returns a status dict.
    """
    import torch

    from .obs import CRITIC_DIM

    sd = ckpt.get("policy") or ckpt.get("model")
    if sd is None:
        raise ValueError("checkpoint has no policy weights")
    net_sd = net.state_dict()
    copied, zeroed, skipped = 0, 0, []
    with torch.no_grad():
        for name, p in net_sd.items():
            if name not in sd:
                skipped.append(name)
                continue
            old = sd[name]
            if old.shape == p.shape:
                p.copy_(old)
                copied += 1
            elif p.dim() == 2 and old.dim() == 2 and old.shape[0] == p.shape[0] \
                    and old.shape[1] < p.shape[1]:
                p[:, :old.shape[1]] = old
                p[:, old.shape[1]:] = 0.0
                copied += 1
                zeroed += int(p[:, old.shape[1]:].numel())
            else:
                skipped.append(name)
    net.load_state_dict(net_sd)
    return {"copied_tensors": copied, "zeroed_new_inputs": zeroed,
            "skipped": skipped[:6]}


if __name__ == "__main__":  # self-check
    w = DEFAULT_WEIGHTS
    # exact tracking of a standing reference: positive, no penalties
    t = track_reward_terms(joint_err=0.0, site_err=0.0, root_xy_err=0.0,
                           root_z_err=0.0, yaw_err=0.0, ref_pelvis_z=0.74,
                           torso_up_z=1.0, action_delta_mean=0.0, sat_frac=0.0,
                           limit_prox=0.0, slide_frac=0.0, w=w)
    assert t["total"] > 1.9, t
    # the crouch escape: reference stands (0.74), robot sits 0.30 m low ->
    # reward collapse AND deviation (the measured hole, closed in arithmetic)
    t2 = track_reward_terms(joint_err=0.5, site_err=0.10, root_xy_err=0.02,
                            root_z_err=0.30, yaw_err=0.0, ref_pelvis_z=0.74,
                            torso_up_z=1.0, action_delta_mean=0.0, sat_frac=0.0,
                            limit_prox=0.0, slide_frac=0.0, w=w)
    assert t2["total"] < 0.5 * t["total"], (t["total"], t2["total"])
    assert track_deviation(joint_err=0.0, root_xy_err=0.0, pelvis_drop_m=0.30,
                           site_err=0.0, w=w) == "pelvis_drop"
    # deep-phase scaling: the same 0.10 m height error in a penetration pose
    t3 = track_reward_terms(joint_err=0.0, site_err=0.0, root_xy_err=0.0,
                            root_z_err=0.10, yaw_err=0.0, ref_pelvis_z=0.35,
                            torso_up_z=1.0, action_delta_mean=0.0, sat_frac=0.0,
                            limit_prox=0.0, slide_frac=0.0, w=w)
    assert t3["w_root"] > 0.6 * math.exp(-1.0) - 1e-6
    # curriculum + labels
    segs1 = stage_segments("S1_stand_lower_hold_rise")
    assert any(s.label == "STANCE" for s in segs1)
    assert phase_label("CONNECT_STANCE_HOLD->SHUFFLE_F") == ("SHUFFLE_F", True)
    s5 = stage_segments("S5_entry_recovery")
    assert any(s.label == "SHOT_DOUBLE_LEG" for s in s5)
    assert all(s.source != "stalk_shuffle" for st in STAGE_ORDER
               for s in stage_segments(st)), "held-out take leaked into training"
    # reference block shapes
    b = ref_block(root_local=np.zeros(3), root_vel_local=np.zeros(3),
                  joints_rel=np.zeros(N_JOINTS), contacts=np.ones(2), phase=0.5,
                  skill_id=0, lead=1, connect=False, root_ahead_local=np.zeros(3),
                  own_contacts=np.ones(2), contacts_next=np.ones(2))
    assert b.shape == (REF_DIM,) and REF_ACTOR_DIM == ACTOR_DIM + REF_DIM
    print("solo.track self-check OK:", {
        "ref_actor_dim": REF_ACTOR_DIM, "ref_critic_dim": REF_CRITIC_DIM,
        "ref_dim": REF_DIM, "stages": len(STAGE_ORDER),
        "S1_segments": len(segs1), "S5_segments": len(s5),
        "total_exact": round(t["total"], 3), "total_crouch": round(t2["total"], 3)})
