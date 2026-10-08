"""Fall detection and termination semantics for the solo drill.

Two independent rules (S1 contract, ``docs/SOLO_DRILL.md`` §4):

1. :class:`DorsalDetector` -- the **failed attempt** rule.  It reuses the
   measured dorsal semantics of :mod:`solo.detector` (extracted verbatim from
   ``wrestling.backdet``; dorsal torso/pelvis mat contact + torso tilt >= 45 deg
   + pelvis z <= 0.35 m, sustained 0.30 s) on the single ``a_``-prefixed robot.
   The detector class carries ``ROBOTS = ("a", "b")`` in ``reset``/``status``,
   so this module drives its *functional* API (``back_features`` +
   ``update_features({"a": f}, t)``) with a one-robot dict.
2. :class:`FallDetector` -- the **balance/locomotion** fall rule: pelvis height
   + torso tilt + persistence, with contact awareness so knees/hands/self-chosen
   low postures are never terminal by themselves.

Measured basis for the fall thresholds (this host, solo scene, position servos
only -- see ``reports/2026-10-08/solo_env.md`` for the full table): a held
"stand" keeps pelvis 0.79 m / tilt 0.2 deg; prone collapse lands pelvis
0.076 m tilt 89 deg with torso contact; supine lands pelvis 0.32 m tilt 121 deg
with dorsal contact; a settled "hands planted" posture reads pelvis 0.124 m,
tilt 53 deg, hands contact.  A position-servo G1 cannot hold *any* crouch
open-loop (every low posture slides or topples), so the fall rule cannot key on
pelvis height alone: it keys on *the torso/pelvis being on the mat* (grounded),
which knees/hands-only postures never satisfy, and otherwise on a
pelvis-low-and-tilted condition with the limb-support exemption below.

Semantics summary
-----------------
* ``terminated`` fall == the robot is on the ground: torso/pelvis mat contact,
  or (pelvis z <= ``pelvis_z_ground`` and tilt >= ``tilt_ground_deg``), or
  (pelvis z <= ``pelvis_z_hard``), sustained for ``confirm_s``.
* **knees/hands are never terminal**: while a knee or arm (wrist/elbow)
  supports the body and the torso/pelvis is *not* on the mat, the rule stays
  silent regardless of pelvis height (``limb_support_exempt``, measured on a
  settled kneel: pelvis 0.498 m tilt 0.4 deg, knee contacts, no trigger; and
  on a settled hands-plant: pelvis 0.118 m, arm contacts, no trigger).
  Measured limitation, stated honestly: a fall that ends lying on a supporting
  arm (e.g. side-lying on the elbow) is therefore *not* auto-terminated by this
  rule -- it is still visible in the metrics (tilt ~88 deg, ``alive`` ~0) and
  in the T1 gate's uprightness criterion, and the dorsal rule still ends the
  exchange if the back lands.  ``limb_support_exempt=False`` selects the
  stricter balance variant if a caller wants low-and-tilted to always end.
* ``DorsalDetector`` is a separate verdict (``cause="dorsal"``) matching the
  two-robot exchange-end rule; it always implies the fall rule too (dorsal
  contact + low pelvis), but the two are reported distinctly.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .detector import BackDetConfig, BackToMatDetector, back_features  # noqa: E402

from .scene import (FLOOR_GEOM, HAND_BODIES, KNEE_BODIES, PELVIS_BODY,  # noqa: E402
                    TORSO_BODY)

_EPS = 1e-9


@dataclass(frozen=True)
class FallDetConfig:
    """Fall-rule thresholds (measured basis in the module docstring).

    ``pelvis_z_ground`` / ``tilt_ground_deg`` are the joint "on the ground"
    condition; ``pelvis_z_hard`` is unambiguous regardless of tilt (any posture
    this low is supported by something other than the feet).
    ``limb_support_exempt`` carries the contract "knees/hands/self-chosen low
    postures are NEVER terminal": while an arm (wrist/elbow) or knee supports
    the body and the torso/pelvis is *not* on the mat, the fall rule stays
    silent.  Set it False for the stricter balance variant (low + tilted always
    terminates).  ``confirm_s`` is the persistence window (control-rate samples).
    """

    pelvis_z_ground: float = 0.35
    tilt_ground_deg: float = 60.0
    pelvis_z_hard: float = 0.22
    limb_support_exempt: bool = True
    confirm_s: float = 0.25

    def as_dict(self) -> dict:
        return {
            "pelvis_z_ground": self.pelvis_z_ground,
            "tilt_ground_deg": self.tilt_ground_deg,
            "pelvis_z_hard": self.pelvis_z_hard,
            "limb_support_exempt": self.limb_support_exempt,
            "confirm_s": self.confirm_s,
        }


@dataclass(frozen=True)
class ContactState:
    """Floor-contact flags by body category (one control step)."""

    left_foot: bool = False
    right_foot: bool = False
    left_knee: bool = False
    right_knee: bool = False
    left_hand: bool = False
    right_hand: bool = False
    torso: bool = False
    pelvis: bool = False
    n_contacts: int = 0

    @property
    def feet(self) -> bool:
        return self.left_foot or self.right_foot

    @property
    def knees(self) -> bool:
        return self.left_knee or self.right_knee

    @property
    def hands(self) -> bool:
        return self.left_hand or self.right_hand

    @property
    def limbs(self) -> bool:
        return self.feet or self.knees or self.hands

    @property
    def grounded(self) -> bool:
        return self.torso or self.pelvis

    def as_dict(self) -> dict:
        return {
            "left_foot": self.left_foot, "right_foot": self.right_foot,
            "left_knee": self.left_knee, "right_knee": self.right_knee,
            "left_hand": self.left_hand, "right_hand": self.right_hand,
            "torso": self.torso, "pelvis": self.pelvis,
            "n_contacts": self.n_contacts,
        }


class _ContactResolver:
    """Model-id -> category maps, cached per model (module-level weak cache)."""

    _cache: dict[int, "_ContactResolver"] = {}

    def __init__(self, model: mujoco.MjModel):
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
        if self.floor < 0:
            raise KeyError(f"floor geom {FLOOR_GEOM!r} not in model")
        self.cat: dict[int, str] = {}

        def mark(bodies, cat: str):
            for name in bodies:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid < 0:
                    raise KeyError(f"body {name!r} not in model")
                for g in range(model.ngeom):
                    if int(model.geom_bodyid[g]) == bid:
                        self.cat[g] = cat

        mark([PELVIS_BODY], "pelvis")
        mark([TORSO_BODY], "torso")
        mark(["a_left_ankle_roll_link"], "left_foot")
        mark(["a_right_ankle_roll_link"], "right_foot")
        mark([KNEE_BODIES[0]], "left_knee")
        mark([KNEE_BODIES[1]], "right_knee")
        mark([HAND_BODIES[0], HAND_BODIES[1]], "left_hand")
        mark([HAND_BODIES[2], HAND_BODIES[3]], "right_hand")

    @classmethod
    def for_model(cls, model: mujoco.MjModel) -> "_ContactResolver":
        key = id(model)
        r = cls._cache.get(key)
        if r is None or r.floor >= model.ngeom:
            r = cls(model)
            cls._cache[key] = r
        return r


def contact_state(model: mujoco.MjModel, data: mujoco.MjData) -> ContactState:
    """Floor contacts by body category at the current ``data`` state."""
    res = _ContactResolver.for_model(model)
    flags = {"pelvis": False, "torso": False, "left_foot": False,
             "right_foot": False, "left_knee": False, "right_knee": False,
             "left_hand": False, "right_hand": False}
    n = 0
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 == res.floor:
            other = g2
        elif g2 == res.floor:
            other = g1
        else:
            continue
        cat = res.cat.get(other)
        if cat is None:
            continue
        flags[cat] = True
        n += 1
    return ContactState(**flags, n_contacts=n)


@dataclass(frozen=True)
class FallFeatures:
    """Per-control-step fall features."""

    t: float
    pelvis_z: float
    tilt_deg: float
    torso_up_z: float
    contacts: ContactState

    def as_dict(self) -> dict:
        return {"t": self.t, "pelvis_z": self.pelvis_z, "tilt_deg": self.tilt_deg,
                "torso_up_z": self.torso_up_z, "contacts": self.contacts.as_dict()}


def fall_features(model: mujoco.MjModel, data: mujoco.MjData) -> FallFeatures:
    """Measure pelvis height, torso tilt, contacts (the fall inputs)."""
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PELVIS_BODY)
    up_z = float(data.xmat[tid].reshape(3, 3)[2, 2])
    return FallFeatures(
        t=float(data.time),
        pelvis_z=float(data.xpos[pid][2]),
        tilt_deg=float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))),
        torso_up_z=up_z,
        contacts=contact_state(model, data),
    )


class FallDetector:
    """Pelvis-height + tilt + persistence rule with the limb-support exemption."""

    def __init__(self, cfg: FallDetConfig | None = None):
        self.cfg = cfg or FallDetConfig()
        self.reset()

    def reset(self) -> None:
        self._since: float | None = None
        self.trigger_t: float | None = None
        self.last: FallFeatures | None = None

    def predicts(self, f: FallFeatures) -> bool:
        """Instantaneous condition (no persistence).

        Order: (1) torso/pelvis mat contact always terminates (the robot is on
        the mat); (2) knee/arm support with no torso/pelvis contact is *never*
        terminal (config ``limb_support_exempt``); (3) otherwise low pelvis
        (hard) or low+tilted (soft) terminates.
        """
        c = self.cfg
        if f.contacts.torso or f.contacts.pelvis:
            return True
        if c.limb_support_exempt and (f.contacts.knees or f.contacts.hands):
            return False
        if f.pelvis_z <= c.pelvis_z_hard:
            return True
        return (f.pelvis_z <= c.pelvis_z_ground
                and f.tilt_deg >= c.tilt_ground_deg)

    def update(self, f: FallFeatures, t: float | None = None) -> bool:
        """Feed one control-rate sample; True on the trigger sample (latched)."""
        self.last = f
        t = f.t if t is None else float(t)
        if self.trigger_t is not None:
            return False
        if self.predicts(f):
            if self._since is None:
                self._since = t
            if t - self._since >= self.cfg.confirm_s - _EPS:
                self.trigger_t = t
                return True
        else:
            self._since = None
        return False

    def status(self) -> dict:
        f = self.last
        return {
            "trigger_t": self.trigger_t,
            "since": self._since,
            "pelvis_z": None if f is None else round(f.pelvis_z, 4),
            "tilt_deg": None if f is None else round(f.tilt_deg, 2),
            "torso_contact": None if f is None else f.contacts.torso,
            "limb_contact": None if f is None else f.contacts.limbs,
        }


class DorsalDetector:
    """Single-robot dorsal (back-to-mat) rule reusing :mod:`solo.detector`.

    ``update`` latches at most one trigger per robot (per reset), timestamped at
    the first control-rate sample where the persistence requirement is met.
    """

    def __init__(self, cfg: BackDetConfig | None = None):
        self.cfg = cfg or BackDetConfig()
        self._det = BackToMatDetector(self.cfg)
        self._torso_bid = None
        self._condition_active = False

    def reset(self) -> None:
        self._det.reset()
        self._condition_active = False

    @property
    def condition_active(self) -> bool:
        """True when the instantaneous dorsal condition held at the last update
        (used by the env to defer a plain 'fall' verdict to the dorsal window)."""
        return bool(self._condition_active)

    @property
    def trigger_t(self) -> float | None:
        return self._det.triggers["a"]

    @property
    def last_features(self):
        return self._det.last_features["a"]

    def update(self, model: mujoco.MjModel, data: mujoco.MjData, t: float | None = None) -> bool:
        """Measure + evaluate; True on the trigger sample (latched)."""
        f = back_features(model, data, "a", self.cfg)
        self._condition_active = bool(self._det.condition(f))
        new = self._det.update_features({"a": f}, float(data.time) if t is None else float(t))
        return bool(new.get("a"))

    def status(self) -> dict:
        f = self.last_features
        return {
            "trigger_t": self.trigger_t,
            "since": self._det._since["a"],
            "dorsal_contact": None if f is None else bool(f.dorsal_contact),
            "tilt_deg": None if f is None else round(float(f.tilt_deg), 2),
            "pelvis_z": None if f is None else round(float(f.pelvis_z), 4),
        }


if __name__ == "__main__":  # self-check
    from .scene import load_solo_model, stand_frame

    model = load_solo_model()
    data = mujoco.MjData(model)
    q, c = stand_frame(model)
    data.qpos[:] = q
    data.ctrl[:] = c
    mujoco.mj_forward(model, data)
    det = FallDetector()
    dorsal = DorsalDetector()
    f = fall_features(model, data)
    assert not det.predicts(f), f
    for k in range(50):
        mujoco.mj_step(model, data)
        f = fall_features(model, data)
        assert not det.update(f), f
        assert not dorsal.update(model, data), dorsal.status()
    cs = f.contacts
    assert cs.feet and not cs.grounded, cs
    print("solo.fall self-check OK:", {"stand": f.as_dict()["tilt_deg"],
                                       "contacts": cs.as_dict()})
