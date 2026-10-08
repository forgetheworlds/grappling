#!/usr/bin/env python3
"""Render watchable visual evidence for the wrestling-env rules stage (deliverable 5).

Outputs, one pair per scenario, into ``videos/env/``:

* ``<scenario>.mp4``           30 fps, 960x720, h264/yuv420p, side view, both robots
* ``<scenario>_sheet.png``     3-frame contact sheet (start / mid / end)

Scenarios (standing start -> rule outcome), all driven by the *shipped* env API
(``WrestlingEnv`` + ``ReferenceReplay``/``StandHold``) with deterministic seeds:

1. ``draw_stance``            STANCE replay, exchange clock 1.0 s -> timeout draw,
                              standing reset, repeated (5 draws in 5 s).
2. ``takedown_back_event``    SPRAWL replay from the reference's own start -> the
                              defender's back reaches the mat -> persistence window
                              -> trigger -> opponent scores -> standing reset.
3. ``nonterminal_knees_hands``SPRAWL attacker on its hands/knees (limb contact,
                              back detector silent, no score) -> the exchange runs
                              its full clock to a draw -> standing reset.
4. ``ambiguous_simultaneous`` both wrestlers supine on the mat (constructed from the
                              settled SPRAWL end state) -> both triggers inside the
                              0.10 s window -> ambiguous, no score -> standing reset.
5. ``oob_forfeit``            one robot's pelvis driven out of / back into the
                              declared 1.5 m competition circle 3x -> forfeit.

Every clip carries a HUD with the exchange clock, the score, the back-detector
diagnostics (dorsal contact / tilt / pelvis z / limb contact / trigger time), the
pelvis radius and the OOB counts, so each claim is readable off the video itself.

Honest deviations (measured on this host, documented in reports/2026-10-08/):

* The raw STANCE reference pose is **not** balanceable open-loop: PD-replaying it
  topples both robots at ~1.4 s and the back detector correctly fires (measured
  1.78 s).  Scenario 1 therefore uses the env's own 1.0 s-clock draw recipe
  (``tests/test_wrestling.py::test_exchange_timeout_draw``) so each exchange ends
  *while both robots are still standing*; the topple is reported, not hidden.
* Every exchange reset in the delivered env returns to the randomized STANCE start,
  which is not balanceable either (with ``StandHold`` the pair collapses at
  ~2.0-2.4 s).  :class:`EvidenceEnv` keeps all env rules and only replaces the
  standing start pose with the model's verified-stable ``both_stand`` keyframe -- the
  same pose the delivered 20 s-draw test passes through ``reset(pose=...)``.
* No shipped controller brings a robot from knees/hands back to standing (that is the
  phase-3 teacher's job, report sec. 3.1/3.5).  Scenario 3 therefore shows the legal,
  non-terminal knees/hands phase and the exchange continuing, not a completed
  recovery; the deviation is stated in reports/2026-10-08/env_videos.md.
* Scenario 5 drives the offender's pelvis across the boundary with a scripted
  controller (the env has no locomotion policy yet) -- the same mechanism as the
  delivered OOB test, but time-ramped instead of a single-step teleport.

Run from the repo root:

    MUJOCO_GL=egl .venv/bin/python scripts/render_env_videos.py            # all
    MUJOCO_GL=egl .venv/bin/python scripts/render_env_videos.py --check    # sim only
    MUJOCO_GL=egl .venv/bin/python scripts/render_env_videos.py --only draw_stance
    MUJOCO_GL=egl .venv/bin/python scripts/render_env_videos.py --diag     # sec. 1 deviation
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from wrestling.backdet import FLOOR_GEOM, ROBOTS, body_maps  # noqa: E402
from wrestling.env import (MAT_RADIUS, MODEL_DT, QPOS_SLICE, QVEL_SLICE,  # noqa: E402
                           STEP_DT, ReferenceReplay, StandHold, WrestlingEnv,
                           keyframe_pair_qpos, load_wrestling_model, reference_trace,
                           stance_start_qpos)

SCENE_XML = REPO / "robots" / "wrestling_scene.xml"
OUT_DIR = REPO / "videos" / "env"
FPS = 30
WIDTH, HEIGHT = 960, 720
RING_POSTS = 72
CAM_AZIMUTH = 90.0        # side view: the pair axis (x) is horizontal in frame
CAM_ELEVATION = -14.0
#: This host renders through software EGL (no /dev/dri).  MuJoCo's default shadow
#: pass alone costs ~6x the frame time here (measured: 0.41 fps -> 2.6 fps), so the
#: evidence clips render without shadows/reflections and without MSAA.  Everything
#: else (lighting, materials, the mat ring) is the shipped look.
FAST_GL = True
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
BANNER_S = 0.9            # how long an exchange-end banner stays on screen

OOB_PLAN = ((0.0, 0.0), (0.90, 0.0), (1.40, -1.75), (1.90, -1.75), (2.40, 0.0),
            (2.90, 0.0), (3.40, -1.80), (3.90, -1.80), (4.40, 0.0), (4.90, 0.0),
            (5.40, -1.80), (7.00, -1.80))


# --------------------------------------------------------------------------- poses
def stand_pair(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """The model's verified-stable standing pair (``both_stand`` keyframe)."""
    return keyframe_pair_qpos(model, "both_stand")


def settled_supine_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Settled supine body pose: the SPRAWL defender after its PD replay.

    Same deterministic construction as
    ``tests/test_wrestling.py::_sprawl_settled_supine`` (the kinematic final reference
    frame is face-down; the *physics* settles it supine).
    """
    tr = reference_trace("SPRAWL")
    qa, qb, t_ref = tr["qpos_a"], tr["qpos_b"], tr["t"]
    data = mujoco.MjData(model)
    data.qpos[QPOS_SLICE["a"]] = qa[0]
    data.qpos[QPOS_SLICE["b"]] = qb[0]
    mujoco.mj_forward(model, data)

    def ref_at(t, q):
        i = int(np.clip(np.searchsorted(t_ref, t), 1, len(t_ref) - 1))
        w = (t - t_ref[i - 1]) / (t_ref[i] - t_ref[i - 1])
        return q[i - 1, 7:36] * (1 - w) + q[i, 7:36] * w

    for step in range(int((float(t_ref[-1]) + 0.5) / MODEL_DT)):
        t = min(step * MODEL_DT, float(t_ref[-1]))
        data.ctrl[:29] = ref_at(t, qa)
        data.ctrl[29:58] = ref_at(t, qb)
        mujoco.mj_step(model, data)
    return data.qpos[QPOS_SLICE["b"]].copy()


def resolve_pose(key: str | None, model: mujoco.MjModel):
    """(qpos_a, qpos_b) for a pose key, or ``None`` for the env's default start."""
    if key is None or key == "stand":
        return None
    if key == "stand_pair":
        return stand_pair(model)
    if key == "stance":
        return stance_start_qpos()
    if key == "sprawl":
        tr = reference_trace("SPRAWL")
        return tr["qpos_a"][0].copy(), tr["qpos_b"][0].copy()
    if key == "sprawl_a_stand_b":
        qb = stand_pair(model)[1].copy()
        qb[0] = 1.05
        return reference_trace("SPRAWL")["qpos_a"][0].copy(), qb
    if key == "supine":
        q = settled_supine_qpos(model)
        qa, qb = q.copy(), q.copy()
        # Translation is exactly dynamics-neutral here (uniform infinite floor), so the
        # construction places the two identical supine bodies inside the declared mat
        # circle, 1.8 m apart, instead of 1.8 m apart wherever the replay happened to
        # settle (that can be outside the 1.5 m circle and would inject OOB events).
        qa[0], qa[1] = +0.9, 0.0
        qb[0], qb[1] = -0.9, 0.0
        return qa, qb
    raise KeyError(key)


# --------------------------------------------------------------------- controllers
class SlideController:
    """Scripted drag of one robot's pelvis out of / back into the competition circle.

    The env has no locomotion policy at this stage, so the offender's root is driven
    directly (the OOB rule reads only the pelvis radius) -- the same mechanism as the
    delivered OOB test, but time-ramped with holds so each inside->outside crossing is a
    separate, legible event instead of a single-step teleport.  The drag runs along
    ``y`` (perpendicular to the pair axis) so the offender never sweeps through its
    opponent; the root position *and* orientation are pinned to the standing start
    (velocity written consistently with the plan) so the offender stays upright on its
    feet across the boundary instead of toppling from foot friction, while the joint
    servos hold the standing pose.
    """

    def __init__(self, robot: str, plan=OOB_PLAN, pin_x: float = -0.35, pin_y0: float = 0.0,
                 pin_z: float = 0.7902):
        self.robot = robot
        self.plan = tuple((float(t), float(y)) for t, y in plan)
        self.pin_x = float(pin_x)
        self.pin_y0 = float(pin_y0)
        self.pin_z = float(pin_z)
        self.pin_quat = None              # filled from the start state on first call

    def _y(self, t: float) -> tuple[float, float]:
        """(y-offset, dy/dt) of the plan at time ``t``."""
        ts = [p[0] for p in self.plan]
        if t <= ts[0]:
            return self.plan[0][1], 0.0
        if t >= ts[-1]:
            return self.plan[-1][1], 0.0
        i = int(np.searchsorted(ts, t))
        t0, y0 = self.plan[i - 1]
        t1, y1 = self.plan[i]
        w = (t - t0) / (t1 - t0)
        return y0 + w * (y1 - y0), (y1 - y0) / (t1 - t0)

    def __call__(self, env: "WrestlingEnv", data: mujoco.MjData) -> None:
        y, dy = self._y(float(env.exchange_time))
        qi = QPOS_SLICE[self.robot].start
        vi = QVEL_SLICE[self.robot].start
        if self.pin_quat is None:
            self.pin_quat = np.asarray(data.qpos[qi + 3:qi + 7]).copy()
        data.qpos[qi] = self.pin_x
        data.qpos[qi + 1] = self.pin_y0 + y
        data.qpos[qi + 2] = self.pin_z
        data.qpos[qi + 3:qi + 7] = self.pin_quat
        data.qvel[vi:vi + 6] = 0.0
        data.qvel[vi + 1] = dy
        return None                       # env holds the standing joint targets


def build_controllers(key: str, model: mujoco.MjModel, env: "WrestlingEnv") -> tuple:
    """Controller pair for ``env``'s *current* exchange, per scenario schedule."""
    first = env.exchange_index == 0
    if key == "stance":
        return ReferenceReplay("STANCE", "a"), ReferenceReplay("STANCE", "b")
    if key == "hold":
        return StandHold("a"), StandHold("b")
    if key == "sprawl_then_hold":
        if first:
            return ReferenceReplay("SPRAWL", "a"), ReferenceReplay("SPRAWL", "b")
        return StandHold("a"), StandHold("b")
    if key == "sprawl_a_then_hold":
        if first:
            return ReferenceReplay("SPRAWL", "a"), StandHold("b")
        return StandHold("a"), StandHold("b")
    if key == "supine_then_hold":
        if first:
            return None, None             # hold the constructed supine joints
        return StandHold("a"), StandHold("b")
    if key == "oob":
        return SlideController("a"), StandHold("b")
    raise KeyError(key)


class EvidenceEnv(WrestlingEnv):
    """``WrestlingEnv`` with a scripted standing start for **every** exchange.

    All rules, detectors, rewards and the exchange loop are the shipped ones.  Only
    the start pose of the reset exchanges is replaced (``_randomized_stance``), because
    the env's default reset pose -- the randomized STANCE reference frame -- is not
    balanceable open-loop (report sec. 3.1): with it, every exchange after the first
    collapses within ~2 s and the clips would show the fall instead of the rule.

    ``start_pose`` is an explicit ``(qpos_a, qpos_b)`` pair, or ``None`` for the
    model's verified-stable ``both_stand`` keyframe.  Exchange 0 can still be
    overridden through the env's own ``reset(pose=...)`` hook.
    """

    def __init__(self, *args, start_pose=None, **kwargs):
        self._start_pose = start_pose
        super().__init__(*args, **kwargs)

    def _randomized_stance(self, seed):
        if self._start_pose is None:
            return stand_pair(self.model)
        qa, qb = self._start_pose
        return qa.copy(), qb.copy()


# ------------------------------------------------------------------------ contacts
def limb_contact_names(model: mujoco.MjModel, data: mujoco.MjData, robot: str) -> tuple:
    """Names of the robot's limb bodies (knee/wrist/elbow) touching the floor now."""
    maps = body_maps(model, robot)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
    hits = set()
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 != floor and g2 != floor:
            continue
        other = g2 if g1 == floor else g1
        b = int(model.geom_bodyid[other])
        if b in maps.limb_bids:
            hits.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b))
    return tuple(sorted(hits))


def back_row(model, data, robot) -> dict:
    """HUD row for one robot: detector features (from the delivered detector)."""
    from wrestling.backdet import back_features
    f = back_features(model, data, robot)
    return {
        "dorsal": bool(f.dorsal_contact),
        "front": bool(f.front_contact),
        "tilt": float(f.tilt_deg),
        "pelvis_z": float(f.pelvis_z),
        "limb": bool(f.limb_contact),
        "limb_names": limb_contact_names(model, data, robot),
    }


# ----------------------------------------------------------------------- scenarios
@dataclass
class Scenario:
    name: str
    headline: str
    look_for: str
    seconds: float
    exchange_timeout: float = 20.0
    match_clock: float = 60.0
    start: str | None = "stand"      # start pose of every exchange
    first_pose: str | None = None    # exchange-0 override
    schedule: str = "hold"
    frame_ring: bool = False
    #: sim time of the contact sheet's middle panel (default: the clip midpoint).
    #: Set for scenarios whose key moment is not the clip's middle -- the sheet is
    #: start / key moment / end, so it supports the claim being made.
    sheet_mid_t: float | None = None
    expect: dict = field(default_factory=dict)


SCENARIOS = (
    Scenario(
        name="draw_stance",
        headline="STANCE replay -> exchange clock expires -> DRAW -> standing reset",
        look_for=("both robots in the retargeted STANCE (grappling crouch), exchange "
                  "clock running in the HUD, score stays 0:0, no back trigger, the "
                  "exchange ends on the timer and both robots reset standing"),
        seconds=5.0, exchange_timeout=1.0,
        start="stance", schedule="stance",
        expect={"records": 5, "causes": {"timeout"}, "ambiguous": False,
                "winner": None, "min_end_pelvis_z": 0.55},
    ),
    Scenario(
        name="takedown_back_event",
        headline="SPRAWL takedown -> back-to-mat detector -> score -> standing reset",
        look_for=("the defender's back reaching the mat, the persistence window "
                  "before the trigger (0.30 s + one control step), the score change "
                  "and the standing reset"),
        seconds=5.5, first_pose="sprawl", schedule="sprawl_then_hold",
        sheet_mid_t=1.60,               # dorsal contact established, trigger still pending
        expect={"records": 1, "causes": {"back"}, "winner": "a", "loser": "b",
                "both_triggers": False},
    ),
    Scenario(
        name="nonterminal_knees_hands",
        headline="attacker on hands/knees -> legal, non-terminal -> full-clock DRAW",
        look_for=("the attacker's knee/hand floor contacts (limb_contact=1) with the "
                  "back detector silent (dorsal=0), no score and no reset while the "
                  "exchange clock runs down to a draw"),
        seconds=6.0, exchange_timeout=5.0,
        first_pose="sprawl_a_stand_b", schedule="sprawl_a_then_hold",
        expect={"records": 1, "causes": {"timeout"}, "winner": None,
                "min_limb_frames": 60, "no_dorsal_ex0": True},
    ),
    Scenario(
        name="ambiguous_simultaneous",
        headline="both wrestlers supine -> triggers inside 0.10 s -> AMBIGUOUS, no score",
        look_for=("both backs already on the mat, both triggers in the same control "
                  "step (<= 0.10 s apart), 0:0 score kept, 'ambiguous' in the HUD/"
                  "exchange record, then the standing reset"),
        seconds=5.5, first_pose="supine", schedule="supine_then_hold",
        sheet_mid_t=0.28,               # both supine, persistence window still running
        expect={"records": 1, "causes": {"back"}, "ambiguous": True, "winner": None,
                "trigger_gap_max": 0.10},
    ),
    Scenario(
        name="oob_forfeit",
        headline="pelvis leaves the 1.5 m circle 3x -> OOB forfeit -> opponent wins",
        look_for=("the yellow boundary ring, the offender's pelvis radius crossing "
                  "1.5 m, the OOB counter going 1/3 -> 2/3 -> 3/3 and the exchange "
                  "ending in a forfeit"),
        seconds=6.0, first_pose="stand_pair", schedule="oob", frame_ring=True,
        sheet_mid_t=1.60,               # offender outside the ring, first event counted
        expect={"records": 1, "causes": {"oob"}, "winner": "b", "loser": "a",
                "oob_a": 3, "oob_b": 0},
    ),
)


# ------------------------------------------------------------------------ capturing
@dataclass
class Frame:
    t: float
    qpos: np.ndarray
    lines: tuple
    banner: str | None


def _limb_tag(names, robot: str) -> str:
    """Short contact tags for the HUD (kept narrow so the line never clips)."""
    short = []
    for n in names:
        s = n.replace(f"{robot}_", "").replace("_link", "")
        s = s.replace("left_", "L_").replace("right_", "R_")
        short.append(s)
    if len(short) > 3:
        short = short[:3] + [f"+{len(short) - 3}"]
    return ",".join(short) if short else "-"


def _hud_lines(t: float, info: dict, rows: dict, score: tuple, last_event: str,
               timeout: float, oob: dict, radius: dict) -> tuple:
    trig = {r: info["back"][r]["trigger_t"] for r in ROBOTS}
    lines = [
        f"t {t:6.2f}s   match {info['match_time']:6.2f}s   exchange #{info['exchange_index']}"
        f"   clock {info['exchange_time']:5.2f}/{timeout:.2f}s",
        f"score  A {score[0]:+.2f} : {score[1]:+.2f} B      {last_event}",
    ]
    for r in ROBOTS:
        row = rows[r]
        trg = "-" if trig[r] is None else f"{trig[r]:.2f}s"
        lines.append(
            f"{r}  back_det: dorsal={int(row['dorsal'])} tilt={row['tilt']:5.1f}deg "
            f"pelvis_z={row['pelvis_z']:.3f} limb={int(row['limb'])}"
            f"[{_limb_tag(row['limb_names'], r)}] trig={trg}")
    lines.append(
        f"mat radius {MAT_RADIUS:.2f} m   rA={radius['a']:.2f} rB={radius['b']:.2f}   "
        f"OOB events A {oob['a']}/3 B {oob['b']}/3")
    return tuple(lines)


def capture(scn: Scenario, model: mujoco.MjModel, verbose: bool = True) -> dict:
    """Run the scenario; returns frames + exchange records + measured evidence."""
    env = EvidenceEnv(model=model, seed=0, exchange_timeout=scn.exchange_timeout,
                      match_clock=scn.match_clock,
                      start_pose=resolve_pose(scn.start, model))
    env.reset(seed=0, pose=resolve_pose(scn.first_pose, model))

    frames: list[Frame] = []
    records: list[dict] = []
    events: list[dict] = []
    score = {r: 0.0 for r in ROBOTS}
    contact_first: dict[int, dict] = {}
    contact_frames: dict[int, dict] = {}
    oob_at: dict[str, list[tuple]] = {r: [] for r in ROBOTS}
    limb_tally: dict[int, dict] = {}
    oob_seen = {r: 0 for r in ROBOTS}
    state = {"key": None, "last_event": "-", "end_t": -1e9, "banner": None,
             "next_k": 1}
    pelvis_bid = {r: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{r}_pelvis")
                  for r in ROBOTS}

    def sync_controllers() -> None:
        key = (scn.schedule, env.exchange_index)
        if key != state["key"]:
            env.set_controllers(*build_controllers(scn.schedule, model, env))
            state["key"] = key

    def features() -> tuple:
        return {r: back_row(model, env.data, r) for r in ROBOTS}

    def radius() -> dict:
        return {r: float(np.hypot(*env.data.xpos[pelvis_bid[r]][:2])) for r in ROBOTS}

    def grab() -> None:
        rows = features()
        rad = radius()
        info = {
            "t": env.time, "match_time": env.match_time,
            "exchange_index": env.exchange_index, "exchange_time": env.exchange_time,
            "oob": dict(env._oob), "back": env.back_snapshot(),
        }
        banner = state["banner"] if env.time - state["end_t"] <= BANNER_S else None
        frames.append(Frame(
            t=env.time, qpos=env.data.qpos.copy(),
            lines=_hud_lines(env.time, info, rows, (score["a"], score["b"]),
                             state["last_event"], scn.exchange_timeout, env._oob,
                             rad),
            banner=banner))
        ex = env.exchange_index
        contact_frames.setdefault(ex, {r: {"limb": 0, "dorsal": 0, "total": 0}
                                       for r in ROBOTS})
        for r in ROBOTS:
            f = contact_frames[ex][r]
            f["total"] += 1
            f["limb"] += int(rows[r]["limb"])
            f["dorsal"] += int(rows[r]["dorsal"])
            tally = limb_tally.setdefault(ex, {r: {} for r in ROBOTS})[r]
            for name in rows[r]["limb_names"]:
                tally[name] = tally.get(name, 0) + 1
            first = contact_first.setdefault(ex, {r: {} for r in ROBOTS})[r]
            for name in ("dorsal", "front", "limb"):
                if rows[r][name] and name not in first:
                    first[name] = env.time

    sync_controllers()
    grab()
    prev_radius = radius()
    t0 = time.time()
    for _ in range(int(round(scn.seconds / STEP_DT)) + 2):
        if env.match_over:
            break
        _, reward, terminated, _, info = env.step()
        for i, r in enumerate(ROBOTS):
            score[r] += float(reward[i])
        ended = info["exchange_ended"] is not None
        # on a step that also ends the exchange the env has already reset the
        # wrestlers, so the event radius is taken from the previous step
        rad = prev_radius if ended else radius()
        for r in ROBOTS:
            n = info["oob"][r]
            if n != oob_seen[r]:
                oob_seen[r] = n
                oob_at[r].append((env.time, rad[r]))
                events.append({"kind": "oob", "robot": r, "count": n, "t": env.time,
                               "radius": rad[r]})
        if not ended:
            prev_radius = rad
        if info["exchange_ended"] is not None:
            rec = info["exchange_ended"]
            records.append(rec)
            events.append({"kind": "exchange_end", "t": env.time, **{
                k: rec[k] for k in ("index", "cause", "winner", "loser", "ambiguous")}})
            if rec["cause"] == "back" and rec["ambiguous"]:
                state["last_event"] = f"last: exchange {rec['index']} BACK (AMBIGUOUS) - no score"
                state["banner"] = (f"EXCHANGE {rec['index']} END - cause: back, AMBIGUOUS "
                                   f"(both backs inside 0.10 s) - no score - reset")
            elif rec["cause"] == "back":
                state["last_event"] = (f"last: exchange {rec['index']} back -> "
                                       f"{rec['winner']} wins")
                state["banner"] = (f"EXCHANGE {rec['index']} END - cause: back - "
                                   f"{rec['winner']} scores, {rec['loser']} loses - reset")
            elif rec["cause"] == "oob":
                state["last_event"] = (f"last: exchange {rec['index']} OOB forfeit -> "
                                       f"{rec['winner']} wins")
                state["banner"] = (f"EXCHANGE {rec['index']} END - cause: oob (3 events) - "
                                   f"{rec['winner']} wins by forfeit - reset")
            else:
                state["last_event"] = f"last: exchange {rec['index']} {rec['cause']} (draw)"
                state["banner"] = (f"EXCHANGE {rec['index']} END - cause: {rec['cause']} "
                                   f"- no score (draw) - reset")
            state["end_t"] = env.time
        sync_controllers()
        if env.time >= state["next_k"] / FPS - 1e-9:
            state["next_k"] += 1
            grab()
        if terminated:
            break

    sim_wall = time.time() - t0
    ev = {
        "scenario": scn,
        "frames": frames,
        "records": records,
        "events": events,
        "contact_first": contact_first,
        "contact_frames": contact_frames,
        "limb_tally": limb_tally,
        "oob_at": oob_at,
        "sim_wall_s": sim_wall,
        "score": (score["a"], score["b"]),
    }
    if verbose:
        print(f"\n=== {scn.name}  ({scn.headline})")
        print(f"    {len(frames)} frames / {scn.seconds:.1f} s sim, "
              f"{sim_wall:.1f} s wall, score {score['a']:+.2f}:{score['b']:+.2f}, "
              f"timeout {scn.exchange_timeout:.2f}s")
        for ex in sorted(contact_frames):
            first = contact_first[ex]
            cf = contact_frames[ex]
            print(f"    exchange {ex}: first dorsal "
                  + ", ".join(f"{r}={_f(first[r].get('dorsal'))}" for r in ROBOTS)
                  + " | first limb "
                  + ", ".join(f"{r}={_f(first[r].get('limb'))}" for r in ROBOTS)
                  + " | limb frames "
                  + ", ".join(f"{r}={cf[r]['limb']}/{cf[r]['total']}" for r in ROBOTS)
                  + " | dorsal frames "
                  + ", ".join(f"{r}={cf[r]['dorsal']}" for r in ROBOTS))
            for r in ROBOTS:
                t = limb_tally.get(ex, {}).get(r, {})
                if t:
                    print(f"      limb contacts (exchange {ex}, {r}): "
                          + ", ".join(f"{k} x{v}" for k, v in sorted(t.items())))
        for r in ROBOTS:
            if oob_at[r]:
                print(f"    OOB events {r}: "
                      + ", ".join(f"#{i+1}@{t:.2f}s (pelvis r={rr:.2f} m)"
                                  for i, (t, rr) in enumerate(oob_at[r])))
        for rec in records:
            print("    exchange record: " + _json(rec))
    return ev


def _f(v) -> str:
    return "None" if v is None else f"{v:.2f}s"


def _json(obj) -> str:
    import json
    return json.dumps(obj, sort_keys=False)


# ------------------------------------------------------------------------- rendering
def build_render_model(ring: bool = True) -> mujoco.MjModel:
    """Compile the scene with a render-only ring marking the declared 1.5 m circle.

    The ring is :data:`RING_POSTS` capsule posts added to a copy of the spec built
    here; the geoms are ``contype=0 conaffinity=0`` on the world body (no mass, no
    qpos), so the render model's physics is exactly the shipped scene -- and the
    simulation model the environment uses (``load_wrestling_model``) is never touched.
    """
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    if ring:
        for i in range(RING_POSTS):
            ang = 2.0 * np.pi * i / RING_POSTS
            g = spec.worldbody.add_geom()
            g.name = f"mat_ring_{i:02d}"
            g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            g.size = [0.010, 0.045, 0.0]
            g.pos = [MAT_RADIUS * float(np.cos(ang)), MAT_RADIUS * float(np.sin(ang)), 0.045]
            g.contype = 0
            g.conaffinity = 0
            g.rgba = [1.0, 0.85, 0.2, 0.85]
    model = spec.compile()
    ref = load_wrestling_model()
    assert model.nq == ref.nq and model.nu == ref.nu, (model.nq, model.nu)
    return model


def framing(ev: dict, model: mujoco.MjModel) -> list[dict]:
    """Per-frame side camera that keeps both robots (and the mat ring) in frame.

    The camera is the free camera at ``azimuth=90`` (the pair axis is horizontal in
    frame) and a slight elevation, so the image axes are ``x`` (horizontal), ``z``
    (vertical) and ``y`` (depth).  Per frame the lookat is the centre of the x/z
    extents of everything that must be visible and the distance must satisfy
    ``d >= offset/tan(half_angle) + depth`` for the horizontal and vertical offsets
    (the ``depth`` term covers a point that sits closer to the eye than the lookat).
    The distance is a single constant for the clip (worst frame + 5%) so nothing pops,
    clamped to a legibility floor; the lookat pans with the pair.
    """
    data = mujoco.MjData(model)
    fovy = np.radians(float(model.vis.global_.fovy))
    tan_v = np.tan(fovy / 2.0)
    tan_h = tan_v * (WIDTH / HEIGHT)
    alpha = np.radians(-CAM_ELEVATION)          # camera above the lookat when negative
    ring = ev["scenario"].frame_ring
    ids = [b for b in range(model.nbody)
           if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(("a_", "b_"))]
    pelvis = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{r}_pelvis")
              for r in ROBOTS]
    ring_pts = np.asarray([(MAT_RADIUS * np.cos(a), MAT_RADIUS * np.sin(a), 0.05)
                           for a in np.linspace(0, 2 * np.pi, 64)]) if ring \
        else np.zeros((0, 3))
    margin = 0.30 if ring else 0.25             # per-point allowance for mesh extents
    out, need = [], 0.0
    for fr in ev["frames"]:
        data.qpos[:] = fr.qpos
        mujoco.mj_forward(model, data)
        pel = np.asarray(data.xpos[pelvis])
        pts = np.asarray(data.xpos[ids])
        if ring:
            pts = np.vstack([pts, ring_pts])
            # the declared mat centre stays fixed, so the circle reads as the fixed
            # arena the offender leaves (panning the lookat would slide the ring)
            lookat = np.array([0.0, 0.0, 0.5 * (pts[:, 2].min() + pts[:, 2].max())])
        else:
            lookat = np.array([0.5 * (pts[:, 0].min() + pts[:, 0].max()),
                               float(pel[:, 1].mean()),
                               0.5 * (pts[:, 2].min() + pts[:, 2].max())])
        r = pts - lookat
        a_h = np.abs(r[:, 0]) + margin
        b_v = np.abs(-r[:, 1] * np.sin(alpha) + r[:, 2] * np.cos(alpha)) + margin
        depth = r[:, 1] * np.cos(alpha) + r[:, 2] * np.sin(alpha)
        frame_need = float(np.max(np.maximum(a_h / tan_h + depth, b_v / tan_v + depth)))
        need = max(need, frame_need)
        out.append({"lookat": tuple(lookat), "distance": 0.0})
    # clamped to a legibility window: wide enough for the declared circle, close enough
    # that the wrestlers stay readable (the exact worst-frame fit is reported)
    lo, hi = (4.2, 5.0) if ring else (4.0, 5.5)
    dist = float(np.clip(need * 1.05, lo, hi))
    for entry in out:
        entry["distance"] = dist
    return out


def draw_hud(img: np.ndarray, lines: tuple, banner: str | None) -> np.ndarray:
    """Overlay the HUD text (and the transient exchange banner) onto a frame."""
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    font = ImageFont.truetype(str(FONT_PATH), 15)
    bold = ImageFont.truetype(str(FONT_BOLD), 15)
    pad, lh = 8, 19
    w = min(int(max(dr.textlength(ln, font=font) for ln in lines)) + 2 * pad, WIDTH - 12)
    h = lh * len(lines) + 2 * pad
    dr.rectangle([6, 6, 6 + w, 6 + h], fill=(0, 0, 0, 165), outline=(230, 230, 230, 120))
    for i, ln in enumerate(lines):
        f = bold if i < 2 else font
        col = (255, 255, 255, 255) if i < 2 else (215, 235, 255, 255)
        dr.text((6 + pad, 6 + pad + i * lh), ln, font=f, fill=col)
    if banner is not None:
        bw = dr.textlength(banner, font=bold) + 2 * pad
        x0 = (WIDTH - bw) / 2
        y0 = 6 + h + 10
        dr.rectangle([x0, y0, x0 + bw, y0 + lh + 2 * pad], fill=(150, 20, 20, 205))
        dr.text((x0 + pad, y0 + pad), banner, font=bold, fill=(255, 255, 255, 255))
    return np.asarray(im)


def write_sheet(scn: Scenario, images: dict, out_png: Path, caption: str) -> None:
    """Compose the 3-frame contact sheet (start / mid / end + labels)."""
    order = [("start", images["start"]), ("mid", images["mid"]), ("end", images["end"])]
    font = ImageFont.truetype(str(FONT_BOLD), 17)
    small = ImageFont.truetype(str(FONT_PATH), 14)
    head, label_h = 58, 26
    sheet = Image.new("RGB", (WIDTH * 3, HEIGHT + head + label_h), (18, 18, 20))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 6), f"{scn.name} - {scn.headline}", font=font, fill=(255, 255, 255))
    dr.text((10, 30), caption, font=small, fill=(200, 220, 255))
    for i, (label, (t, img)) in enumerate(order):
        dr.text((10 + i * WIDTH, head + 4), f"{label}  t = {t:5.2f} s", font=small,
                fill=(255, 235, 160))
        sheet.paste(Image.fromarray(img), (i * WIDTH, head + label_h))
    sheet.save(out_png)


def render(scn: Scenario, ev: dict, preview: bool = False) -> dict:
    """Render the captured frames to mp4 + contact sheet; returns output paths."""
    model = build_render_model(ring=True)
    if FAST_GL:
        model.vis.quality.offsamples = 1
    cams = framing(ev, model)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    flags = renderer._scene.flags          # mjRND_* render flags (see FAST_GL)
    try:
        sheet_ix = {"start": 0, "mid": len(ev["frames"]) // 2,
                    "end": len(ev["frames"]) - 1}
        if scn.sheet_mid_t is not None:
            ts = np.asarray([f.t for f in ev["frames"]])
            sheet_ix["mid"] = int(np.argmin(np.abs(ts - scn.sheet_mid_t)))
        keep = {}
        out_mp4 = OUT_DIR / f"{scn.name}.mp4"
        if preview:
            out_mp4 = Path("/tmp") / f"preview_{scn.name}.mp4"
        writer = imageio.get_writer(out_mp4, fps=FPS, quality=8, macro_block_size=None,
                                    mode="I")
        t0 = time.time()
        for i, fr in enumerate(ev["frames"]):
            data.qpos[:] = fr.qpos
            mujoco.mj_forward(model, data)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.azimuth, cam.elevation = CAM_AZIMUTH, CAM_ELEVATION
            cam.distance = cams[i]["distance"]
            cam.lookat[:] = cams[i]["lookat"]
            renderer.update_scene(data, camera=cam)
            if FAST_GL:
                flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            img = draw_hud(renderer.render(), fr.lines, fr.banner)
            writer.append_data(img)
            for label, ix in sheet_ix.items():
                if i == ix:
                    keep[label] = (fr.t, img.copy())
            if (i + 1) % 50 == 0:
                print(f"      {scn.name}: {i+1}/{len(ev['frames'])} frames "
                      f"({(time.time()-t0)/(i+1):.2f} s/frame)")
        writer.close()
    finally:
        renderer.close()
    render_wall = time.time() - t0
    sheet = OUT_DIR / f"{scn.name}_sheet.png"
    caption = f"what to look for: {scn.look_for}"
    if preview:
        sheet = Path("/tmp") / f"preview_{scn.name}_sheet.png"
    write_sheet(scn, keep, sheet, caption)
    return {"mp4": out_mp4, "sheet": sheet, "distance": cams[0]["distance"],
            "frames": len(ev["frames"]), "render_wall_s": render_wall}


# ------------------------------------------------------------------------------ main
def check(ev: dict) -> tuple[bool, list[str]]:
    """Scenario-specific acceptance check on the simulated evidence."""
    scn: Scenario = ev["scenario"]
    ex = scn.expect
    msgs, ok = [], True

    def req(cond, text):
        nonlocal ok
        ok = ok and bool(cond)
        msgs.append(("PASS  " if cond else "FAIL  ") + text)

    recs = ev["records"]
    req(len(recs) == ex.get("records", len(recs)),
        f"{len(recs)} exchange record(s), expected {ex.get('records')}")
    if "causes" in ex:
        got = {r["cause"] for r in recs}
        req(got == set(ex["causes"]), f"causes {sorted(got)} == {sorted(ex['causes'])}")
    if "winner" in ex:
        req(recs and recs[0]["winner"] == ex["winner"],
            f"record 0 winner {recs[0]['winner'] if recs else None!r} == {ex['winner']!r}")
    if "loser" in ex:
        req(recs and recs[0]["loser"] == ex["loser"],
            f"record 0 loser {recs[0]['loser'] if recs else None!r} == {ex['loser']!r}")
    if "ambiguous" in ex:
        req(recs and bool(recs[0]["ambiguous"]) == ex["ambiguous"],
            f"record 0 ambiguous {recs[0]['ambiguous'] if recs else None} "
            f"== {ex['ambiguous']}")
    if "trigger_gap_max" in ex:
        t = recs[0]["back_triggers"]
        gap = abs(t["a"] - t["b"])
        req(gap <= ex["trigger_gap_max"],
            f"both triggers {t['a']:.2f}s / {t['b']:.2f}s inside the "
            f"{ex['trigger_gap_max']:.2f} s ambiguity window (gap {gap:.2f} s)")
    if ex.get("no_dorsal_ex0"):
        req(ev["contact_frames"][0]["a"]["dorsal"] == 0,
            "attacker had no dorsal contact in exchange 0 (knees/hands only)")
    if "min_limb_frames" in ex:
        n = ev["contact_frames"][0]["a"]["limb"]
        req(n >= ex["min_limb_frames"],
            f"attacker limb_contact True in {n} frames of exchange 0 "
            f"(>= {ex['min_limb_frames']})")
    if "min_end_pelvis_z" in ex:
        zs = [r["end_back"][p]["pelvis_z"] for r in recs for p in ROBOTS]
        req(min(zs) >= ex["min_end_pelvis_z"],
            f"all exchange-end pelvises >= {ex['min_end_pelvis_z']} m "
            f"(min {min(zs):.3f} m)")
    if "oob_a" in ex:
        req(recs[0]["oob_events"]["a"] == ex["oob_a"]
            and recs[0]["oob_events"]["b"] == ex["oob_b"],
            f"OOB events {recs[0]['oob_events']} == a:{ex['oob_a']} b:{ex['oob_b']}")
        req(len(ev["oob_at"]["a"]) == ex["oob_a"],
            f"{len(ev['oob_at']['a'])} offender OOB event times logged")
    # universal blood checks
    for rec in recs:
        if rec["cause"] == "back" and not rec["ambiguous"]:
            l = rec["loser"]
            trg = rec["back_triggers"][l]
            first = ev["contact_first"][rec["index"]][l].get("dorsal")
            req(trg is not None and first is not None and first <= trg,
                f"exchange {rec['index']}: loser {l} dorsal contact {_f(first)} precedes "
                f"trigger {_f(trg)}")
    req(ev["contact_frames"][0]["a"]["limb"] >= 0, "contact diagnostics recorded")
    return ok, msgs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma-separated scenario names")
    ap.add_argument("--check", action="store_true", help="simulate + print, no rendering")
    ap.add_argument("--preview", action="store_true",
                    help="render the clips to /tmp (visual check, keeps videos/env clean)")
    ap.add_argument("--diag", action="store_true",
                    help="run the scenario-1 deviation diagnostic (raw STANCE replay with "
                         "the default 20 s clock -> back event instead of a draw)")
    args = ap.parse_args()

    model = load_wrestling_model()
    if args.diag:
        env = WrestlingEnv(model=model, seed=0, exchange_timeout=20.0, match_clock=40.0,
                           controllers=(ReferenceReplay("STANCE", "a"),
                                        ReferenceReplay("STANCE", "b")))
        env.reset(seed=0, pose=resolve_pose("stance", model))
        for _ in range(2000):
            _, _, terminated, _, info = env.step()
            if info["exchange_ended"] is not None or terminated:
                break
        print("stance-topple diagnostic (STANCE replay, 20 s exchange clock):")
        print("  " + _json(env.exchange_log[0].as_dict()))
        if args.only is None:
            return 0

    names = args.only.split(",") if args.only else [s.name for s in SCENARIOS]
    scenarios = [s for s in SCENARIOS if s.name in names]
    if not scenarios:
        print(f"unknown scenario(s) {names}; known: {[s.name for s in SCENARIOS]}")
        return 2
    if not args.preview:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_wrestling_model()
    print(f"render_env_videos: {len(scenarios)} scenario(s), {FPS} fps "
          f"{WIDTH}x{HEIGHT}, side view (azimuth {CAM_AZIMUTH:g}), "
          f"mat radius {MAT_RADIUS:g} m")
    verdicts, outputs = [], []
    for scn in scenarios:
        ev = capture(scn, model)
        ok, msgs = check(ev)
        paths = None
        if not args.check:
            paths = render(scn, ev, preview=args.preview)
            print(f"    wrote {paths['mp4']}")
            print(f"    wrote {paths['sheet']}")
            print(f"    {paths['frames']} frames, camera distance "
                  f"{paths['distance']:.2f} m, render {paths['render_wall_s']:.1f} s wall")
        for m in msgs:
            print("    " + m)
        verdicts.append((scn.name, ok))
        if paths is not None:
            outputs.append(paths)

    print("\n--- verdicts ---")
    for name, ok in verdicts:
        print(f"  {name:26s} {'OK' if ok else 'MISMATCH'}")
    if outputs:
        missing = [str(p[k]) for p in outputs for k in ("mp4", "sheet")
                   if not Path(p[k]).exists() or Path(p[k]).stat().st_size < 5000]
        if missing:
            print(f"  SELF-CHECK FAILED: missing/tiny outputs {missing}")
            return 1
        print(f"  SELF-CHECK OK: {len(outputs)} clips and {len(outputs)} contact sheets "
              f"written to {Path(outputs[0]['mp4']).parent}")
    all_ok = all(ok for _, ok in verdicts)
    print(f"  outcomes {'match' if all_ok else 'MISMATCH'} the scenario expectations")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
