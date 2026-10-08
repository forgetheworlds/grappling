"""Back-to-mat detector (deliverable 10).

Semantics (MISSION Phase 3, ``docs/MISSION.md`` "Back-to-mat terminal rule"):

    A wrestler loses the exchange when they are *clearly put onto their back*.

The detector is deliberately NOT a single collision bit.  A robot is in the
back-to-mat state at time ``t`` when all three conditions hold at ``t``:

1. **Dorsal torso contact with the mat.**  A contact between the ``floor``
   geom and the robot's ``torso_link`` or ``pelvis`` collision geoms whose
   contact point lies in the *dorsal* (back) half of that body's frame
   (local ``x < cfg.dorsal_x_max``).  Sign convention: the pelvis/torso local
   ``+x`` is the robot's front (the G1 foot toes point along ``+x``; the
   retargeted world places robot A at ``-x`` *facing* ``+x``, see
   ``src/retarget/world.py``).  A supine robot (belly up) therefore touches
   the mat with contacts at ``x_local < 0``; a prone/face-down robot (sprawl,
   superman) touches with ``x_local > 0`` and must NOT trigger.
2. **Torso up-axis tilt.**  ``tilt = arccos((R_torso @ ez)_z)`` in degrees,
   ``tilt >= cfg.tilt_threshold_deg``.  Tilt alone cannot distinguish supine
   from prone (both ~90 deg); it exists to reject upright postures while the
   pelvis/back happens to graze the mat.
3. **Pelvis low.**  ``pelvis_z <= cfg.pelvis_z_threshold`` (standing pelvis
   is ~0.79 m; knees/hands postures keep the pelvis above the threshold).

and the joint condition must **persist** for at least ``cfg.confirm_s``
seconds before a trigger is reported.  Knee, hand, elbow and foot contacts
never trigger by themselves: they touch the mat without a dorsal torso
contact.

``BackToMatDetector.update`` is the streaming implementation used by the
environment (evaluated at the 50 Hz control rate).  ``back_features`` +
``confirmed_mask`` are the batch equivalents used by the calibration script
(the sweep must be fast, so the classifier is re-implemented as a vectorized
run-length rule over recorded feature rows; both are pinned to the same
semantics by tests).

Thresholds are calibrated by ``scripts/calibrate_backdet.py`` against
PD-replay rollouts of ``data/refs/*.npz``; the chosen operating point is
stored in ``data/backdet_calibration.json`` and mirrored in
``DEFAULT_CONFIG`` below (test_wrestling asserts the two agree).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

ROBOTS = ("a", "b")
FLOOR_GEOM = "floor"
#: torso bodies for the rule: ``torso_link`` + ``pelvis`` collision geoms
#: (see :func:`body_maps`); diagnostics only: knees/hands/elbows touching the
#: mat is NOT a loss
LIMB_BODIES = ("knee_link", "wrist_roll_link", "wrist_yaw_link", "elbow_link")

#: floating-point slack for the persistence comparison
_EPS = 1e-9


@dataclass(frozen=True)
class BackDetConfig:
    """Detector thresholds.

    Defaults are the calibrated operating point from
    ``scripts/calibrate_backdet.py`` (2026-10-08, data/backdet_calibration.json):
    tilt 45 deg, pelvis z 0.35 m, confirmation 0.30 s -> sensitivity 1.000 /
    specificity 1.000 on the calibration set, detection latency 0.32 s, tilt
    margins +17.6 / -18.8 deg and pelvis-z margin +0.256 m (see report).
    """

    tilt_threshold_deg: float = 45.0
    pelvis_z_threshold: float = 0.35
    confirm_s: float = 0.30
    #: contact point local x must be < this to count as dorsal (0.0 = back half)
    dorsal_x_max: float = 0.0

    def as_dict(self) -> dict:
        return {
            "tilt_threshold_deg": self.tilt_threshold_deg,
            "pelvis_z_threshold": self.pelvis_z_threshold,
            "confirm_s": self.confirm_s,
            "dorsal_x_max": self.dorsal_x_max,
        }


DEFAULT_CONFIG = BackDetConfig()


@dataclass
class BackFeatures:
    """Per-robot detector features at one sample time."""

    t: float
    dorsal_contact: bool
    front_contact: bool
    contact_count: int
    contact_x_min: float  # most dorsal contact point (NaN when no contact)
    contact_x_max: float
    tilt_deg: float
    pelvis_z: float
    limb_contact: bool  # knee/hand/elbow contact with the mat (diagnostics)
    dorsal_normal: bool = False  # independent cross-check: mat normal opposes dorsal axis

    def as_dict(self) -> dict:
        return {
            "t": float(self.t),
            "dorsal_contact": bool(self.dorsal_contact),
            "front_contact": bool(self.front_contact),
            "contact_count": int(self.contact_count),
            "contact_x_min": float(self.contact_x_min),
            "contact_x_max": float(self.contact_x_max),
            "tilt_deg": float(self.tilt_deg),
            "pelvis_z": float(self.pelvis_z),
            "limb_contact": bool(self.limb_contact),
            "dorsal_normal": bool(self.dorsal_normal),
        }


@dataclass
class _BodyMaps:
    """Cached model id maps for one robot prefix."""

    torso_bid: int
    pelvis_bid: int
    torso_geoms: frozenset
    limb_bids: frozenset


def body_maps(model: mujoco.MjModel, robot: str) -> _BodyMaps:
    """Resolve the torso/pelvis/limb body and geom ids for robot ``"a"``/``"b"``.

    Recomputed per call (a few µs): MjModel is a C struct that cannot hold
    Python-side caches, and id()-keyed caches are unsafe across model
    lifetimes.
    """
    prefix = robot if robot.endswith("_") else f"{robot}_"

    def bid(name: str) -> int:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{name}")
        if i < 0:
            raise KeyError(f"body {prefix}{name} not in model")
        return i

    torso_bid = bid("torso_link")
    pelvis_bid = bid("pelvis")
    torso_geoms = frozenset(
        g for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) in (torso_bid, pelvis_bid)
    )
    # limb bodies are side-infixed (``a_left_knee_link``, ``b_right_elbow_link``
    # …), so prefix concatenation never matches: resolve by suffix against the
    # model's body names, keeping the robot prefix so each robot resolves only
    # its own limbs.
    limb = {
        b for b in range(model.nbody)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)) is not None
        and name.startswith(prefix) and name.endswith(LIMB_BODIES)
    }
    return _BodyMaps(torso_bid, pelvis_bid, torso_geoms, frozenset(limb))


def back_features(model: mujoco.MjModel, data: mujoco.MjData, robot: str,
                  cfg: BackDetConfig = DEFAULT_CONFIG) -> BackFeatures:
    """Measured features for one robot (``"a"``/``"b"``) at the current ``data`` state."""
    maps = body_maps(model, robot)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
    acc = {"dorsal": False, "front": False, "n": 0,
           "xmin": np.inf, "xmax": -np.inf, "limb": False, "normal": False}
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 != floor and g2 != floor:
            continue
        other = g2 if g1 == floor else g1
        b = int(model.geom_bodyid[other])
        if b in maps.limb_bids:
            acc["limb"] = True
            continue
        if other not in maps.torso_geoms:
            continue
        R = data.xmat[b].reshape(3, 3)
        p_local = R.T @ (np.asarray(con.pos) - np.asarray(data.xpos[b]))
        x = float(p_local[0])
        acc["n"] += 1
        acc["xmin"] = min(acc["xmin"], x)
        acc["xmax"] = max(acc["xmax"], x)
        if x < cfg.dorsal_x_max:
            acc["dorsal"] = True
        else:
            acc["front"] = True
        # independent cross-check: contact normal (oriented mat -> robot) must
        # oppose the dorsal axis (local -x) of the contacting body: lying on
        # the back means the dorsal axis points into the mat (dot ~ -1)
        normal = np.asarray(con.frame[:3], dtype=np.float64)
        if g1 != floor:
            normal = -normal
        if float(np.dot(-R[:, 0], normal)) < -0.5:
            acc["normal"] = True

    R_torso = data.xmat[maps.torso_bid].reshape(3, 3)
    tilt = float(np.degrees(np.arccos(np.clip(R_torso[2, 2], -1.0, 1.0))))
    return BackFeatures(
        t=float(data.time),
        dorsal_contact=acc["dorsal"],
        front_contact=acc["front"],
        contact_count=acc["n"],
        contact_x_min=acc["xmin"] if acc["n"] else np.nan,
        contact_x_max=acc["xmax"] if acc["n"] else np.nan,
        tilt_deg=tilt,
        pelvis_z=float(data.xpos[maps.pelvis_bid][2]),
        limb_contact=acc["limb"],
        dorsal_normal=acc["normal"],
    )


class BackToMatDetector:
    """Streaming per-robot detector with a confirmation period.

    ``update`` latches at most one trigger per robot; ``reset`` clears the
    latch (new exchange).  Trigger timestamps are the first sample time at
    which the persistence requirement is met.
    """

    def __init__(self, cfg: BackDetConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._since: dict[str, float | None] = {p: None for p in ROBOTS}
        self.triggers: dict[str, float | None] = {p: None for p in ROBOTS}
        self.last_features: dict[str, BackFeatures | None] = {p: None for p in ROBOTS}

    def condition(self, f: BackFeatures) -> bool:
        c = self.cfg
        return bool(
            f.dorsal_contact
            and f.tilt_deg >= c.tilt_threshold_deg
            and f.pelvis_z <= c.pelvis_z_threshold
        )

    def update(self, model: mujoco.MjModel, data: mujoco.MjData, t: float) -> dict:
        """Measure + evaluate both robots; returns {prefix: True} for new triggers."""
        feats = {p: back_features(model, data, p, self.cfg) for p in ROBOTS}
        return self.update_features(feats, t)

    def update_features(self, feats: dict, t: float) -> dict:
        new: dict[str, bool] = {}
        for p, f in feats.items():
            self.last_features[p] = f
            if self.triggers[p] is not None:
                continue
            if self.condition(f):
                if self._since[p] is None:
                    self._since[p] = t
                if t - self._since[p] >= self.cfg.confirm_s - _EPS:
                    self.triggers[p] = t
                    new[p] = True
            else:
                self._since[p] = None
        return new

    def status(self) -> dict:
        """Lightweight snapshot for env info dicts / logs."""
        out = {}
        for p in ROBOTS:
            f = self.last_features[p]
            out[p] = {
                "trigger_t": self.triggers[p],
                "since": self._since[p],
                "dorsal_contact": None if f is None else bool(f.dorsal_contact),
                "tilt_deg": None if f is None else round(float(f.tilt_deg), 2),
                "pelvis_z": None if f is None else round(float(f.pelvis_z), 4),
            }
        return out


def confirmed_mask(cond: np.ndarray, t: np.ndarray, confirm_s: float) -> np.ndarray:
    """Sample-level batch rule: True where the current run of ``cond`` has
    lasted >= ``confirm_s`` (same rule as :class:`BackToMatDetector`)."""
    cond = np.asarray(cond, dtype=bool)
    t = np.asarray(t, dtype=float)
    assert cond.shape == t.shape
    out = np.zeros_like(cond)
    if not cond.any():
        return out
    starts = np.flatnonzero(cond & ~np.concatenate(([False], cond[:-1])))
    ends = np.flatnonzero(cond & ~np.concatenate((cond[1:], [False])))
    for s, e in zip(starts, ends):
        j = int(np.searchsorted(t[s:e + 1], t[s] + confirm_s - _EPS))
        out[s + j:e + 1] = True
    return out


def first_confirmed_index(cond: np.ndarray, t: np.ndarray, confirm_s: float):
    """First index where the persistence requirement is met, or None."""
    mask = confirmed_mask(cond, t, confirm_s)
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


@dataclass
class FeatureSequence:
    """Recorded per-robot feature rows, ready for offline sweeps."""

    technique: str
    robot: str
    t: np.ndarray
    dorsal: np.ndarray
    tilt: np.ndarray
    pelz: np.ndarray
    front: np.ndarray = None
    limb: np.ndarray = None
    dnormal: np.ndarray = None
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_rows(cls, technique: str, robot: str, rows: list) -> "FeatureSequence":
        return cls(
            technique=technique,
            robot=robot,
            t=np.array([f.t for f in rows]),
            dorsal=np.array([f.dorsal_contact for f in rows], dtype=bool),
            tilt=np.array([f.tilt_deg for f in rows]),
            pelz=np.array([f.pelvis_z for f in rows]),
            front=np.array([f.front_contact for f in rows], dtype=bool),
            limb=np.array([f.limb_contact for f in rows], dtype=bool),
            dnormal=np.array([f.dorsal_normal for f in rows], dtype=bool),
        )

    def cond(self, cfg: BackDetConfig) -> np.ndarray:
        return self.dorsal & (self.tilt >= cfg.tilt_threshold_deg) & (
            self.pelz <= cfg.pelvis_z_threshold
        )

    def confirmed(self, cfg: BackDetConfig) -> np.ndarray:
        return confirmed_mask(self.cond(cfg), self.t, cfg.confirm_s)


if __name__ == "__main__":  # self-check
    dt = 0.02
    t = np.arange(50) * dt
    dorsal = np.zeros(50, dtype=bool)
    dorsal[10:40] = True          # a 0.6 s contact episode
    tilt = np.where(dorsal, 80.0, 10.0)
    pelz = np.where(dorsal, 0.25, 0.75)
    cfg = DEFAULT_CONFIG
    mask = confirmed_mask(dorsal & (tilt >= cfg.tilt_threshold_deg)
                          & (pelz <= cfg.pelvis_z_threshold), t, cfg.confirm_s)
    idx = np.flatnonzero(mask)
    assert idx.size and abs(float(t[idx[0]]) - (t[10] + cfg.confirm_s)) < 1e-9, t[idx[0]]
    # knees/hands (limb contact) and front contacts never trigger
    assert not mask[0] and not mask[9]
    print("backdet self-check OK:", {"confirm_s": cfg.confirm_s,
                                     "trigger_t": float(t[idx[0]])})
