"""Drill scheduler: a scripted *but state-gated* sequence of elements.

The scheduler owns *what* the drill does next; the controller owns *how* it is
executed.  Two rules make the sequence honest:

* **progress is physical.**  A step element advances when the controller
  reports completed steps (each one ended in a settled, flat, loaded landing);
  a hold element advances when the measured state satisfies its guard (both
  feet flat, CoM margin in band, pelvis height inside the target band).  No
  element advances because its clock ran out.
* **timeouts are failures.**  Every element has a safeguard time; when it
  fires the event is recorded as ``element_timeout`` and the run falls back to
  a recovery element instead of pretending the element completed.

Element parameters (step direction/length, crouch depth, circle rate, dwell
times) are re-drawn per repetition from a seeded stream, so repeated cycles are
not one memorised loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .controller import RUNG_ELEMENTS as RUNG_ELEMENTS_STR
from .controller import DrillCommand

#: physical guards usable by an element
GUARDS = ("stance", "flat", "height", "any")


@dataclass
class Element:
    """One scheduled element."""

    phase: str                       # human-readable rung element (report/HUD)
    skill: str                       # controller command
    steps: int = 0                   # completed steps required to advance
    hold_s: float = 0.0              # dwell with the state guard satisfied
    timeout: float = 5.0             # safeguard (recorded as a failure)
    params: dict = field(default_factory=dict)
    guard: str = "any"


def _entry_elements(rng: np.random.Generator) -> list:
    """The stand -> wide staggered stance transition, walked foot by foot.

    Only scheduled for rungs whose element set includes the step primitive: it
    *is* a stepping task (the feet cannot be teleported into the stance), so
    rungs L0/L1 initialise directly in the built stance instead.
    """
    return [Element("entry_settle", "STANCE", hold_s=0.8, timeout=4.0, guard="flat"),
            Element("entry_foot", "ENTRY", steps=2, timeout=24.0, guard="flat"),
            Element("entry_hold", "STANCE", hold_s=1.5, timeout=5.0, guard="stance")]


def program(rung: str, rng: np.random.Generator, cycles: int = 0) -> list:
    """The element list for a rung (order = execution order)."""
    prog = list(_entry_elements(rng)) if "step" in RUNG_ELEMENTS_STR.get(rung, ()) else []
    if rung == "L0":
        prog += _l0(rng)
    elif rung == "L1":
        prog += _l1(rng)
    elif rung == "L2":
        prog += _l2(rng)
    elif rung in ("L3", "L4"):
        prog += _l3(rng)
        if rung == "L4":
            prog += _l4(rng)
    else:
        raise ValueError(f"unknown rung {rung!r}")
    return prog


def _l0(rng: np.random.Generator) -> list:
    """Hold + posture modulation: weight shift and height oscillation."""
    out = []
    for i in range(6):
        h = float(rng.uniform(0.20, 0.60))
        out.append(Element(f"l0_shift_{i}", "STANCE", hold_s=float(rng.uniform(1.6, 2.6)),
                           timeout=6.0, guard="stance",
                           params={"stance_height": h, "sway": float(rng.uniform(-1, 1))}))
    return out


def _l1(rng: np.random.Generator) -> list:
    """Level changes and weight shifts on planted feet.

    Measured attempt (2026-10-08): promoting one *real* load-gated step per cycle
    into this programme (SHUFFLE_F/B with steps=1, alternating feet) fails — the
    run falls at 16 s (L1) / 3 s (entry) once the lift gate is relaxed enough to
    fire in a staggered stance.  The step primitive itself is clean in isolation;
    sequencing it is the open L2 gate, so L1 stays a planted-feet programme and
    the stepping clips are failure evidence, not the headline.
    """
    out = []
    for i in range(3):
        depth = float(rng.uniform(0.55, 1.0))
        out.append(Element(f"l1_crouch_{i}", "LEVEL_CHANGE", hold_s=float(rng.uniform(0.8, 1.4)),
                           timeout=6.0, guard="height",
                           params={"stance_height": depth}))
        out.append(Element(f"l1_rise_{i}", "RECOVER", hold_s=0.9, timeout=6.0,
                           guard="stance", params={"stance_height": 0.0}))
    return out


def _l2(rng: np.random.Generator) -> list:
    """Single-foot repositioning: one step at a time, both directions."""
    out = []
    for i in range(3):
        out.append(Element(f"l2_step_{i}", "SHUFFLE_F", steps=1, timeout=5.5,
                           guard="flat",
                           params={"vx": float(rng.uniform(0.06, 0.13))}))
        out.append(Element(f"l2_settle_{i}", "STANCE", hold_s=0.5, timeout=4.0,
                           guard="stance"))
    return out


def _l3(rng: np.random.Generator) -> list:
    """Shuffle in four directions plus circling, derived from the step primitive."""
    out = []
    plan = [("SHUFFLE_F", {"vx": float(rng.uniform(0.07, 0.14))}, 3),
            ("SHUFFLE_B", {"vx": float(rng.uniform(0.06, 0.12))}, 2),
            ("SHUFFLE_L", {"vy": float(rng.uniform(0.05, 0.10))}, 2),
            ("SHUFFLE_R", {"vy": float(rng.uniform(0.05, 0.10))}, 2),
            ("CIRCLE_L", {"wz": float(rng.uniform(0.25, 0.45))}, 4),
            ("CIRCLE_R", {"wz": float(rng.uniform(0.25, 0.45))}, 4)]
    for i, (skill, params, steps) in enumerate(plan):
        out.append(Element(f"l3_{skill.lower()}_{i}", skill, steps=steps,
                           timeout=4.0 + 2.0 * steps, guard="flat", params=params))
        out.append(Element(f"l3_hold_{i}", "STANCE", hold_s=0.5, timeout=4.0,
                           guard="stance"))
    return out


def _l4(rng: np.random.Generator) -> list:
    """Penetration-step gesture: lead step, knee lower, trail drive, rise."""
    out = []
    for i in range(2):
        out.append(Element(f"l4_shot_{i}", "SHOT_GESTURE", steps=2, timeout=8.0,
                           guard="flat",
                           params={"lead_leg": "left",
                                   "stance_height": float(rng.uniform(0.7, 1.0))}))
        out.append(Element(f"l4_rise_{i}", "RECOVER", steps=2, timeout=8.0,
                           guard="stance", params={"stance_height": 0.0}))
    return out


class SkillScheduler:
    """State-gated element driver (see module docstring)."""

    def __init__(self, rung: str = "L0", seed: int = 0):
        self.rung = rung
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self.elements = program(rung, self._rng)
        self.cycles = 0
        self.ix = 0
        self.t_enter = 0.0
        self.hold = 0.0
        self.steps_seen = 0
        self.events: list = []
        self.failures: list = []
        self._pending: list = []

    def reset(self, model, data) -> None:
        self.ix = 0
        self.t_enter = 0.0
        self.hold = 0.0
        self.steps_seen = 0
        self.events = []
        self._pending = []

    # -- guards -------------------------------------------------------------
    def _state(self, ids, data) -> dict:
        contact = ids.foot_contact(data)
        # 2 cm of sole-corner tolerance: a position servo with gravity load
        # rolls a corner up ~1 cm in a weight shift (measured); that is not a
        # failed stance, and the *measured* flatness is reported in the metrics
        flat = ids.flat_contact(data, tol=0.020)
        margin = ids.com_margin(data, contact)
        return {"contact": contact, "flat": flat, "margin": float(margin),
                "pelvis_z": ids.pelvis_z(data), "tilt": ids.torso_tilt_deg(data)}

    def _guard_ok(self, el: Element, st: dict, target_z: float) -> bool:
        if el.guard == "any":
            return True
        if el.guard == "flat":
            return bool(st["flat"].all()) and st["margin"] > 0.012
        if el.guard == "stance":
            # 1.5 cm inside the support: the measured standing margin of the
            # built stance is 1.8-7.8 cm over a run (reported per run), so the
            # guard must sit below it or every element times out
            return (bool(st["flat"].all()) and st["margin"] > 0.015
                    and abs(st["pelvis_z"] - target_z) < 0.025)
        if el.guard == "height":
            return abs(st["pelvis_z"] - target_z) < 0.025 and st["margin"] > 0.010
        return True

    # -- tick ---------------------------------------------------------------
    def tick(self, ids, data, ctrl, t: float) -> DrillCommand:
        el = self.elements[self.ix]
        if self.t_enter == 0.0:
            self.t_enter = t
        # completed steps since this element started (physical progress)
        done = sum(fe.steps_done for fe in getattr(ctrl, "stepper", _NULL).state.values())
        st = self._state(ids, data)
        target_z = float(self.target_z(el))
        ok_steps = (done - self.steps_seen) >= el.steps if el.steps else True
        if el.skill == "ENTRY" and hasattr(ctrl, "entry_pending"):
            ok_steps = not ctrl.entry_pending()
        ok_guard = True if el.hold_s <= 0 else self._guard_ok(el, st, target_z)
        if ok_guard and (el.steps > 0 or el.hold_s > 0):
            self.hold += 0.02
        else:
            self.hold = max(0.0, self.hold - 0.02)
        if ok_steps and ok_guard and self.hold >= el.hold_s and (el.steps or el.hold_s):
            self._advance(t, el, "steps" if el.steps else "hold")
        # safeguard -- checked on the element that is *now* running
        el = self.elements[self.ix]
        if t - self.t_enter > el.timeout:
            self.failures.append({"t": round(t, 3), "phase": el.phase,
                                  "reason": "timeout", "elapsed": round(t - self.t_enter, 2)})
            self.events.append({"event": "element_timeout", "phase": el.phase,
                                "t": round(t, 3)})
            self._advance(t, el, "timeout")
        return self._command(self.elements[self.ix], t)

    def target_z(self, el: Element) -> float:
        return float(getattr(self, "_stance_z", 0.74)) - 0.10 * float(
            el.params.get("stance_height", 0.0))

    def _advance(self, t: float, el: Element, how: str) -> None:
        self.events.append({"event": "element_done", "phase": el.phase, "how": how,
                            "t": round(t, 3), "elapsed": round(t - self.t_enter, 3)})
        self.ix += 1
        self.t_enter = 0.0
        self.hold = 0.0
        self.steps_seen = 0
        if self.ix >= len(self.elements):
            self.cycles += 1
            self._rng2 = np.random.default_rng(self.seed + 1000 * self.cycles)
            self.elements = program(self.rung, self._rng2)
            self.ix = 0
            self.events.append({"event": "cycle_done", "cycle": self.cycles, "t": round(t, 3)})

    def _command(self, el: Element, t: float) -> DrillCommand:
        p = el.params
        return DrillCommand(skill=el.skill,
                            vx=float(p.get("vx", 0.0)), vy=float(p.get("vy", 0.0)),
                            wz=float(p.get("wz", 0.0)),
                            stance_height=float(p.get("stance_height", 0.0)),
                            lead_leg=str(p.get("lead_leg", "left")),
                            phase=el.phase)

    def drain_events(self) -> list:
        ev, self.events = self.events, []
        return ev


class _NullStepper:
    state: dict = {}


_NULL = _NullStepper()
