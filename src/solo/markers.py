"""Virtual-opponent markers: the shot entry region as a function of command/phase.

The solo drill never simulates grips or a physical opponent (``docs/SOLO_DRILL.md``
§2): the shot is trained against *spatial markers*.  Five sites live on the mocap
body ``virtual_opponent`` (attached at load by :mod:`solo.scene`):

==================  ======================================================
site                meaning
==================  ======================================================
``a_marker_pelvis`` virtual opponent pelvis at the **entry region**
``a_marker_leg_l``  left-leg/knee entry target of the shot
``a_marker_leg_r``  right-leg/knee entry target
``a_marker_hand_l`` left-hand target region (arm control)
``a_marker_hand_r`` right-hand target region
==================  ======================================================

Geometry and the anti-clock rule
--------------------------------
``MarkerPlan.distance`` is the opponent pelvis distance (m) at the moment the
shot starts; ``entry_depth`` is the separation the robot must close
(``entry_distance = distance - entry_depth``).  The entry region is placed
**world-anchored** when a shot becomes active (``follow=False``) so that
"penetration depth" is *physical* progress toward a fixed world point, never a
function of the clock -- ``rl.reward``'s audit (``notes.md`` E2) records how a
clock-based progress term carries no physical signal.  For non-shot tasks
(``follow=True``) the marker frame simply stays ``entry_distance`` ahead of the
current heading (a moving target; used for reach/visualisation, never for
progress).

Positions as a function of command/phase
----------------------------------------
* command: ``lead_leg`` selects which leg/hand target is the *active* one;
  the entry distance is the command's approach geometry.
* phase: selects the active target region -- phase < 0.4: hands (guard/arm
  control), >= 0.4: lead leg (entry), >= 0.8: opponent pelvis (finish).

The markers are spatial targets only: no contact, grip or force semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .scene import MARKER_NAMES, MARKER_SPECS, VIRTUAL_OPPONENT_BODY, site_id

#: marker-frame local offsets (x forward, y left, z up) from the scene spec
LOCAL_OFFSETS: dict[str, np.ndarray] = {
    name: np.asarray(pos, dtype=np.float64) for name, (pos, _, _) in MARKER_SPECS.items()
}

#: shot-phase thresholds selecting the active target region
PHASE_HANDS = 0.4
PHASE_ENTRY = 0.8


@dataclass(frozen=True)
class MarkerPlan:
    """Marker geometry for a task (see module docstring)."""

    distance: float = 1.05       # opponent pelvis distance at shot start (m)
    entry_depth: float = 0.50    # separation closed at full entry (m)
    height: float = 0.79         # entry-region height (world z)
    follow: bool = True          # True: frame stays ahead of the robot each step

    @property
    def entry_distance(self) -> float:
        return float(self.distance - self.entry_depth)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["entry_distance"] = self.entry_distance
        return d


#: default (moving) plan; the shot task uses ``follow=False``
DEFAULT_PLAN = MarkerPlan()
SHOT_PLAN = MarkerPlan(follow=False)


def frame(pelvis_xy: np.ndarray, yaw: float, plan: MarkerPlan = DEFAULT_PLAN
          ) -> tuple[np.ndarray, np.ndarray]:
    """(virtual-opponent pelvis origin, R_yaw) in world coordinates."""
    d = plan.distance
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pos = np.array([float(pelvis_xy[0]) + c * d, float(pelvis_xy[1]) + s * d,
                    float(plan.height)])
    return pos, R


def targets(pelvis_xy: np.ndarray, yaw: float, plan: MarkerPlan = DEFAULT_PLAN
            ) -> dict[str, np.ndarray]:
    """World positions of the five marker sites for the given frame."""
    pos, R = frame(pelvis_xy, yaw, plan)
    return {name: pos + R @ LOCAL_OFFSETS[name] for name in MARKER_NAMES}


def apply_markers(model: mujoco.MjModel, data: mujoco.MjData, pelvis_xy: np.ndarray,
                  yaw: float, plan: MarkerPlan = DEFAULT_PLAN) -> dict[str, np.ndarray]:
    """Move the mocap body to the plan; returns the world targets."""
    mo = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, VIRTUAL_OPPONENT_BODY)
    if mo < 0:
        raise KeyError(f"body {VIRTUAL_OPPONENT_BODY!r} not in model")
    pos, R = frame(pelvis_xy, yaw, plan)
    mocapid = int(model.body_mocapid[mo])
    data.mocap_pos[mocapid] = pos
    data.mocap_quat[mocapid] = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    return {name: pos + R @ LOCAL_OFFSETS[name] for name in MARKER_NAMES}


def read_positions(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
    """Current world positions of the marker sites (after a kinematics update)."""
    return {name: np.asarray(data.site_xpos[site_id(model, name)],
                             dtype=np.float64).copy() for name in MARKER_NAMES}


def penetration_depth(pelvis_xy: np.ndarray, target_xy: np.ndarray,
                      plan: MarkerPlan = SHOT_PLAN) -> float:
    """Physical entry progress in [0, 1] toward ``target_xy`` (world-anchored).

    0 at the plan's start separation (``plan.distance``), 1 when the pelvis has
    closed ``plan.entry_depth`` of it.  A function of *position*, not of time.
    """
    d = float(np.linalg.norm(np.asarray(pelvis_xy, np.float64)
                             - np.asarray(target_xy, np.float64)))
    return float(np.clip((plan.distance - d) / max(1e-6, plan.entry_depth), 0.0, 1.0))


def active_target(targets_world: dict[str, np.ndarray], phase: float,
                  lead_leg: int) -> tuple[str, np.ndarray]:
    """(kind, world target) selected by the shot phase and lead leg."""
    side = "l" if int(lead_leg) == -1 else "r"
    p = float(np.clip(phase, 0.0, 1.0))
    if p < PHASE_HANDS:
        return "hand", targets_world[f"a_marker_hand_{side}"]
    if p < PHASE_ENTRY:
        return "leg", targets_world[f"a_marker_leg_{side}"]
    return "pelvis", targets_world["a_marker_pelvis"]


def lead_targets(positions: dict[str, np.ndarray], lead_leg: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(leg target, hand target) for the lead leg (+1 right / -1 left)."""
    side = "l" if int(lead_leg) == -1 else "r"
    return positions[f"a_marker_leg_{side}"], positions[f"a_marker_hand_{side}"]


if __name__ == "__main__":  # self-check
    from .scene import load_solo_model, stand_frame

    model = load_solo_model()
    data = mujoco.MjData(model)
    q, c = stand_frame(model)
    data.qpos[:] = q
    data.ctrl[:] = c
    tgt = apply_markers(model, data, np.array([0.0, 0.0]), 0.0, SHOT_PLAN)
    mujoco.mj_forward(model, data)
    got = read_positions(model, data)
    for name in MARKER_NAMES:
        assert np.allclose(tgt[name], got[name], atol=1e-9), (name, tgt[name], got[name])
    # depth is physical: advancing the pelvis reduces the distance to the target
    p0 = penetration_depth(np.array([0.0, 0.0]), got["a_marker_pelvis"][:2], SHOT_PLAN)
    p1 = penetration_depth(np.array([0.30, 0.0]), got["a_marker_pelvis"][:2], SHOT_PLAN)
    p2 = penetration_depth(np.array([0.55, 0.0]), got["a_marker_pelvis"][:2], SHOT_PLAN)
    kind0, _ = active_target(got, 0.1, -1)
    kind1, _ = active_target(got, 0.5, +1)
    kind2, _ = active_target(got, 0.9, +1)
    assert p0 == 0.0 and p1 > p0 and p2 == 1.0, (p0, p1, p2)
    assert (kind0, kind1, kind2) == ("hand", "leg", "pelvis")
    print("solo.markers self-check OK:", {"entry_distance": SHOT_PLAN.entry_distance,
                                          "depth@0.25m": round(p1, 3),
                                          "marker_pelvis": np.round(got["a_marker_pelvis"], 3).tolist()})
