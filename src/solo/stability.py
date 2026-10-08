"""Capture-point stability signal: *why* the robot is about to fall, and how far
its support would have to move to stop it.

This module is the missing half of the M1 balance contract.  ``docs/prior_art_humanoid_control.md``
§5.1 already specifies ``capture_point_xy - support_center_xy`` in the actor
observation and a ``capturability +0.5`` reward, and §2.1 names the "CoM-in-support /
capturability" term as *the* fix for the real M1 failure: "COM outside the support
polygon with no step taken".  Nothing in ``src/solo`` implemented it.  Here it is,
as a small, documented, testable API.

Formulations (primary sources, not summaries)
---------------------------------------------
* **Capture point / DCM** -- ``xi = x + v * sqrt(z_c / g)``.  Pratt et al. 2006
  introduce the capture point as "the point on the ground where the robot must
  step to come to a complete stop and stay" (Pratt, Carff, Drakunov, Goswami,
  *Capture Point: A Step toward Humanoid Push Recovery*, Humanoids 2006).
  Stephens 2007 eq. (4) states the ankle-only (no-step) decision surface in
  exactly this quantity: ``delta- < x_com + v_com/omega < delta+`` with
  ``omega^2 = g / z_com`` (Stephens, *Humanoid Push Recovery*, Humanoids 2007).
  Caron et al. 2018 call this "zero-step capturability" and note (their eq. 35)
  that the boundedness condition reduces to "the CoP is located at the capture
  point" with ``omega_c = sqrt(g / z_i)`` (Caron, Escande, Lanari, Mallein,
  arXiv:1801.07022, §II-F/III).
* **N-step capturability** -- Koolen et al. 2012 define an N-step capturable
  state as one from which the system can come to a stop without falling "while
  taking a maximum of N steps" (Koolen, de Boer, Rebula, Goswami, Pratt,
  *Capturability-based analysis and control of legged locomotion, Part 1*, IJRR
  31(9) 2012; definition restated in Schuller et al. 2026 §1.2.2).  The
  0-step case is "coming to a stop without changing the current contact
  configuration" (Del Prete et al. 2018, cited ibid.) -- i.e. exactly
  ``capture point inside the current support polygon``.
* **Divergence test** -- the DCM error ``delta_xi = xi - xi_d`` converges at an
  instant iff ``Vdot = (1/b) delta_xi . (delta_xi - delta_e) < 0`` for some
  admissible eCMP offset ``delta_e`` (Schuller, Mesesan, Englsberger, Ott, Lee,
  Albu-Schaeffer, *Grounding the three-dimensional divergent component of
  motion*, IJRR 2026, eqs. 45-50; the ankle-sufficiency test is their eq. 67
  ``Vdot_ankle < 0``, and the strategy schedule their eqs. 68/74).  With the CoP
  confined to the support hull and the DCM target inside it, ``Vdot < 0`` is
  reachable **iff the capture point is strictly inside the hull**: for a convex
  hull and an interior point, the support function in the error direction
  exceeds the projection of the point itself, so a CoP beyond the DCM exists --
  otherwise (CP on/outside the boundary) no admissible CoP makes the error
  shrink.  :func:`dcm_test` implements the general form and
  :func:`capture_point_unsafe` the equivalent geometric one; the test file
  asserts they agree.
* **Step-length rule** -- the step that stops the fall puts the capture point at
  the new foot placement (Stephens 2007 eq. 21: ``v_x^+ sqrt(g/z) + x^+ = 0``,
  i.e. zero capture point measured from the new stance origin), which is also
  the capture region of Pratt et al. 2006.  :func:`brace_step_length` measures,
  on the *current* hull, how much of that placement is still missing.

Conventions
-----------
World frame, Z up, SI.  "Support" is the convex hull of the sole footprints of
the feet in contact; the *contact* hull (actual floor contact points) is
exposed alongside because MuJoCo's soft contacts make a strictly
contact-point-only hull collapse to a line whenever a foot rolls onto an edge
(measured in ``drill.kin.support_polygon``: the CoM margin jumped to -0.18 m
without the robot moving).  Signed margins are positive *inside*, negative
outside -- the same convention as ``drill.kin.polygon_margin`` and the
``capturability`` reward's ``clip(0.05 - dist(cp, hull), 0, 0.05)/0.05``.

No file in ``src/solo`` is imported here except ``scene`` (ids/keyframe only),
so this module stays importable while the reward stack is edited.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .scene import FOOT_BODIES, FLOOR_GEOM, body_id

#: standard gravity (m/s^2).  The model's own value is used when a model is
#: available; this is the fallback used by the pure functions.
GRAVITY = 9.81
#: sole-sphere centre height below which a foot counts as touching the mat (m).
#: Same threshold as ``drill.kin.TOUCH_Z`` (1.2 cm = "clearly loaded").
TOUCH_Z = 0.015
#: floor on the DCM pendulum length (m).  ``drill.balance.capture_error`` uses
#: ``max(com_z - 0.05, 0.30)``; the floor matters because ``1/sqrt(z)`` is
#: singular at z=0 and a crouched/lying robot must not produce an infinite CP.
MIN_CP_HEIGHT = 0.30
#: hull kinds accepted by :meth:`StabilityMonitor.support_hull`.
HULL_KINDS = ("footprint", "contact")

_EPS = 1e-12


# ---------------------------------------------------------------------------
# pure geometry / algebra (hand-constructible; no model needed)
# ---------------------------------------------------------------------------
def cp_time_constant(z_c: float, gravity: float = GRAVITY,
                     min_z: float = MIN_CP_HEIGHT) -> float:
    """DCM/capture-point time constant ``b = sqrt(z_c / g)`` (s).

    ``z_c`` is the CoM height above the contact plane.  Non-finite or
    non-positive heights are floored at ``min_z`` so the value is always
    finite and positive (never a division by zero downstream).
    """
    z = float(z_c)
    if not math.isfinite(z) or z < float(min_z):
        z = float(min_z)
    g = float(gravity)
    if not math.isfinite(g) or g <= 0.0:
        g = GRAVITY
    return math.sqrt(z / g)


def capture_point(com_xy, vel_xy, z_c: float, gravity: float = GRAVITY,
                  min_z: float = MIN_CP_HEIGHT) -> np.ndarray:
    """``xi = x + v * sqrt(z_c / g)`` (2,) -- Pratt et al. 2006.

    A zero velocity (or a zero time constant) returns the CoM projection
    exactly, and no input can produce a division by zero.
    """
    x = np.asarray(com_xy, dtype=np.float64).reshape(2)
    v = np.asarray(vel_xy, dtype=np.float64).reshape(2)
    tau = cp_time_constant(z_c, gravity, min_z)
    out = x + v * tau
    return np.nan_to_num(out, nan=float(x[0]) if False else 0.0) if False else out


def hull2d(points) -> np.ndarray:
    """Convex hull (monotone chain) of (n, 2); CCW vertices.

    Local copy of ``drill.kin.hull2d`` (identical algorithm and ordering) so
    this module has no dependency on the drill stack; ``tests/solo/
    test_stability.py`` asserts the two agree point-for-point.
    """
    p = np.unique(np.asarray(points, dtype=np.float64).reshape(-1, 2), axis=0)
    if len(p) <= 2:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) <= 1e-12:
                    out.pop()
                else:
                    break
            out.append(q)
        return out

    return np.array(half(p)[:-1] + half(p[::-1])[:-1])


def signed_margin(point_xy, hull) -> float:
    """Signed distance from ``point_xy`` to a convex hull boundary (m).

    Positive inside, negative outside (``drill.kin.polygon_margin`` semantics).
    An empty hull returns ``-inf`` (no support: not balanceable); a hull with
    fewer than 3 points has no interior, so the margin is the negated distance
    to it.
    """
    p = np.asarray(point_xy, dtype=np.float64).reshape(2)
    poly = np.asarray(hull, dtype=np.float64).reshape(-1, 2)
    if len(poly) == 0:
        return -math.inf
    if len(poly) < 3:
        return -min(float(np.linalg.norm(p - q)) for q in poly)
    inside = True
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        if e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]) < 0.0:
            inside = False
            break
    d = math.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        t = 0.0 if e @ e < _EPS else float(np.clip((p - a) @ e / (e @ e), 0.0, 1.0))
        d = min(d, float(np.linalg.norm(p - (a + t * e))))
    return d if inside else -d


def ray_hit_distance(origin_xy, direction_xy, hull) -> float:
    """Distance from ``origin_xy`` along ``direction_xy`` to the hull boundary (m).

    ``0.0`` when the origin is at/outside the boundary or the direction is
    degenerate; ``inf`` when the ray misses a (degenerate) hull that the origin
    is not inside.  This is the measured ``dCOP`` used by
    ``reports/2026-10-08/t1_gate_calibration.md`` and ``solo.pushes``.
    """
    o = np.asarray(origin_xy, dtype=np.float64).reshape(2)
    u = np.asarray(direction_xy, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(u))
    poly = np.asarray(hull, dtype=np.float64).reshape(-1, 2)
    if n < _EPS or len(poly) < 3:
        return 0.0
    u = u / n
    best = math.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        den = u[0] * e[1] - u[1] * e[0]
        if abs(den) < _EPS:
            continue                      # parallel to this edge
        t = ((a[0] - o[0]) * e[1] - (a[1] - o[1]) * e[0]) / den
        s = ((a[0] - o[0]) * u[1] - (a[1] - o[1]) * u[0]) / den
        if t > _EPS and -_EPS <= s <= 1.0 + _EPS:
            best = min(best, t)
    return 0.0 if math.isinf(best) else float(best)


def brace_direction(com_xy, cp_xy) -> np.ndarray:
    """Unit vector from the CoM projection to the capture point (2,).

    Falls back to ``(+1, 0)`` only when the two points coincide (zero velocity:
    the CP is the CoM, so there is nothing to brace against and the direction
    carries no information -- callers must read :func:`brace_step_length`,
    which is 0 in that case).
    """
    d = np.asarray(cp_xy, dtype=np.float64).reshape(2) - np.asarray(com_xy, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return np.array([1.0, 0.0])
    return d / n


def brace_step_length(com_xy, cp_xy, hull) -> float:
    """Required brace step: how far the support boundary must move (m).

    ``max(0, ||CP - CoM|| - dCOP)`` where ``dCOP`` is the ray distance from the
    CoM projection to the hull boundary in the CP direction.  That is exactly
    the distance from the capture point to the nearest hull edge *in the CP
    direction*: the support must reach the CP to re-capture it (Stephens 2007
    eq. 21).  Zero when the CP is already inside the hull.  Note this is the
    *boundary* travel; the foot's own footprint supplies the leading ~half of
    the foot (see :func:`foot_lead_offset` for the foot-centre travel).
    """
    com = np.asarray(com_xy, dtype=np.float64).reshape(2)
    cp = np.asarray(cp_xy, dtype=np.float64).reshape(2)
    u = brace_direction(com, cp)
    reach = float(np.linalg.norm(cp - com))
    hit = ray_hit_distance(com, u, hull)
    return max(0.0, reach - hit)


def foot_lead_offset(sole_points_xy, direction_xy) -> float:
    """Distance from a foot's centre to its own leading edge along ``u`` (m).

    ``max_i((p_i - c) . u)`` over that foot's footprint points ``p_i`` about
    their centre ``c``: the footprint the step itself contributes to the hull in
    the direction of the step, so a foot placed with its centre at
    ``CP - r_lead * u`` already puts the boundary at the CP.
    """
    p = np.asarray(sole_points_xy, dtype=np.float64).reshape(-1, 2)
    u = np.asarray(direction_xy, dtype=np.float64).reshape(2)
    if len(p) == 0:
        return 0.0
    n = float(np.linalg.norm(u))
    if n < _EPS:
        return 0.0
    u = u / n
    c = p.mean(axis=0)
    return float(max((q - c) @ u for q in p))


def foot_center_travel(com_xy, cp_xy, hull, sole_points_xy) -> float:
    """Brace step as foot-centre travel (m): ``max(0, boundary_travel - r_lead)``."""
    lead = foot_lead_offset(sole_points_xy, brace_direction(com_xy, cp_xy))
    return max(0.0, brace_step_length(com_xy, cp_xy, hull) - lead)


def support_function(hull, direction_xy) -> float:
    """``max_{p in hull} (p . u_hat)`` (m) -- the hull's extent in ``u``.

    For a convex polygon the maximum of a linear functional is attained at a
    vertex, so evaluating the vertices is exact.
    """
    poly = np.asarray(hull, dtype=np.float64).reshape(-1, 2)
    u = np.asarray(direction_xy, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(u))
    if len(poly) == 0 or n < _EPS:
        return -math.inf
    return float(np.max(poly @ (u / n)))


def dcm_test(com_xy, vel_xy, z_c, hull, target_xy=None, gravity: float = GRAVITY,
             min_z: float = MIN_CP_HEIGHT) -> dict:
    """Instantaneous DCM-error convergence test with the CoP in the hull.

    Schuller et al. 2026 eq. (48): ``Vdot = (1/b) dxi . (dxi - de)`` with the
    CoP/hull as the only actuator.  Here ``dxi = cp - target`` and the best
    admissible CoP is the hull vertex maximising ``dxi . p`` (the support
    function), which is what a controller with full CoP authority would pick.
    ``converging`` True means a CoP inside the hull exists that shrinks the DCM
    error *right now*; for a convex hull and a target inside it this is
    equivalent to "the CP is strictly inside the hull"
    (:func:`capture_point_unsafe`).
    """
    com = np.asarray(com_xy, dtype=np.float64).reshape(2)
    vel = np.asarray(vel_xy, dtype=np.float64).reshape(2)
    tau = cp_time_constant(z_c, gravity, min_z)
    cp = capture_point(com, vel, z_c, gravity, min_z)
    poly = np.asarray(hull, dtype=np.float64).reshape(-1, 2)
    if target_xy is None:
        target = poly.mean(axis=0) if len(poly) else com
    else:
        target = np.asarray(target_xy, dtype=np.float64).reshape(2)
    dxi = cp - target
    norm = float(np.linalg.norm(dxi))
    if len(poly) == 0:
        return {"converging": False, "vdot": math.inf, "cp": cp, "target": target,
                "cop": None, "note": "no support: no admissible CoP"}
    if norm < 1e-9:
        return {"converging": True, "vdot": 0.0, "cp": cp, "target": target,
                "cop": target.copy(), "note": "zero DCM error"}
    proj = poly @ dxi
    cop = poly[int(np.argmax(proj))]
    vdot = float(dxi @ (cp - cop)) / tau
    return {"converging": bool(vdot < 0.0), "vdot": vdot, "cp": cp,
            "target": target, "cop": cop.copy(), "note": ""}


def capture_point_unsafe(cp_xy, hull, tol: float = 0.0) -> bool:
    """The fall predicate: is the capture point outside the support hull?

    ``signed_margin(cp, hull) < -tol`` -- Stephens 2007 eq. (4) decision
    surface, i.e. "no ankle strategy can restore balance; a step is required".
    ``tol > 0`` requires the CP to be *beyond* the boundary by that much
    (hysteresis against contact chatter).  No support at all is unsafe.
    """
    return signed_margin(cp_xy, hull) < -float(tol)


# ---------------------------------------------------------------------------
# measured state
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StabilityConfig:
    """Thresholds.  Defaults are the documented ones, not hand-tuned."""

    #: sole-sphere centre height counting as contact (m); ``drill.kin.TOUCH_Z``
    touch_z: float = TOUCH_Z
    #: floor on the DCM pendulum length (m); ``drill.balance.capture_error``
    min_cp_height: float = MIN_CP_HEIGHT
    #: CoM-height offset used for the pendulum length (m): 0 keeps the height
    #: ground-referenced (Pratt/Stephens); the drill law subtracts the ankle
    #: pivot height (~0.05 m).  Exposed so the choice is visible, never hidden.
    pivot_offset: float = 0.0
    #: hysteresis on the CP predicate (m): how far outside the hull the CP must
    #: be before :attr:`StabilityState.unsafe` is True.
    safe_margin: float = 0.0
    #: hull used for the stability verdict ("footprint" of touching feet, or
    #: the raw "contact" points -- see the module docstring).
    hull_kind: str = "footprint"

    def as_dict(self) -> dict:
        return {"touch_z": self.touch_z, "min_cp_height": self.min_cp_height,
                "pivot_offset": self.pivot_offset, "safe_margin": self.safe_margin,
                "hull_kind": self.hull_kind}


@dataclass
class StabilityState:
    """One control step of the capture-point signal (all world frame, SI)."""

    t: float = 0.0
    com_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    com_z: float = 0.0
    com_vel_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    cp_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    tau: float = 0.0                       # DCM time constant used (s)
    z_c: float = 0.0                       # CoM height used for tau (m)
    hull: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    hull_kind: str = "footprint"
    support_center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    contact_hull: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    foot_touch: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=bool))
    #: signed distance from the capture point to the hull boundary (m, + inside)
    margin_cp: float = -math.inf
    #: signed distance from the CoM projection to the hull boundary (m, + inside)
    margin_com: float = -math.inf
    #: boundary travel the support must make in the CP direction (m)
    brace_step_length: float = 0.0
    #: brace step expressed as foot-centre travel (m) using ``brace_foot``
    brace_foot_travel: float = 0.0
    #: unit vector of the required brace (CoM projection -> capture point)
    brace_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    #: the brace would have to be taken by this foot ("left"/"right"/"either")
    brace_foot: str = "either"
    #: CP outside the hull (the divergence-based fall predicate)
    unsafe: bool = True
    #: instantaneous DCM-error convergence (Schuller et al. eq. 48) + detail
    converging: bool = False
    vdot: float = 0.0
    cop_best: np.ndarray | None = None
    reason: str = ""

    def as_dict(self) -> dict:
        """JSON/roundtrip-friendly view (arrays -> lists, rounded)."""
        return {
            "t": round(float(self.t), 4),
            "com_xy": [round(float(v), 5) for v in self.com_xy],
            "com_z": round(float(self.com_z), 5),
            "com_vel_xy": [round(float(v), 5) for v in self.com_vel_xy],
            "cp_xy": [round(float(v), 5) for v in self.cp_xy],
            "tau": round(float(self.tau), 5),
            "z_c": round(float(self.z_c), 5),
            "hull_kind": self.hull_kind,
            "n_hull": int(len(self.hull)),
            "support_center": [round(float(v), 5) for v in self.support_center],
            "foot_touch": [bool(v) for v in self.foot_touch],
            "margin_cp": None if not math.isfinite(self.margin_cp) else round(float(self.margin_cp), 5),
            "margin_com": None if not math.isfinite(self.margin_com) else round(float(self.margin_com), 5),
            "brace_step_length": round(float(self.brace_step_length), 5),
            "brace_foot_travel": round(float(self.brace_foot_travel), 5),
            "brace_direction": [round(float(v), 5) for v in self.brace_direction],
            "brace_foot": self.brace_foot,
            "unsafe": bool(self.unsafe),
            "converging": bool(self.converging),
            "vdot": round(float(self.vdot), 5),
            "reason": self.reason,
        }


class StabilityMonitor:
    """Measures the capture-point signal from the model + contacts each step.

    ``model`` must be a single-robot scene with foot bodies named
    ``FOOT_BODIES`` (``a_left/right_ankle_roll_link``) and a floor geom.
    The monitor caches all ids on construction; :meth:`measure` is a few
    microseconds plus one pass over the contact list.
    """

    def __init__(self, model: mujoco.MjModel, config: StabilityConfig = StabilityConfig(),
                 foot_bodies: tuple = FOOT_BODIES, floor_geom: str = FLOOR_GEOM):
        self.model = model
        self.config = config
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom)
        if self.floor < 0:
            raise KeyError(f"floor geom {floor_geom!r} not in model")
        self.foot_bodies = tuple(foot_bodies)
        self._bids = tuple(body_id(model, b) for b in self.foot_bodies)
        if any(b < 0 for b in self._bids):
            raise KeyError(f"foot bodies {foot_bodies!r} not all in model")
        # pelvis-subtree CoM (the robot's CoM; the floor is not in the subtree)
        self.pelvis = body_id(model, "a_" + "pelvis") if body_id(model, "a_pelvis") >= 0 \
            else body_id(model, "pelvis")
        if self.pelvis < 0:
            raise KeyError("no pelvis body (a_pelvis/pelvis) in model")
        self._sole_geoms: dict[int, np.ndarray] = {}
        for bid in self._bids:
            gs = [g for g in range(model.ngeom)
                  if int(model.geom_bodyid[g]) == bid
                  and int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)]
            self._sole_geoms[bid] = np.asarray(sorted(gs), dtype=int)
        self._mass = np.asarray(model.body_mass, dtype=np.float64)
        self._mass_total = max(float(self._mass.sum()), _EPS)
        self._gravity = float(abs(model.opt.gravity[2])) or GRAVITY

    # -- measurement -------------------------------------------------------
    def sole_points(self, data: mujoco.MjData) -> np.ndarray:
        """(2, 4, 3) world centres of the sole spheres of each foot."""
        out = np.zeros((2, 4, 3), dtype=np.float64)
        for i, bid in enumerate(self._bids):
            gs = self._sole_geoms[bid]
            out[i, :len(gs)] = np.asarray(data.geom_xpos[gs], dtype=np.float64)
        return out

    def foot_touch(self, data: mujoco.MjData,
                   touch_z: float | None = None) -> np.ndarray:
        """(2,) "some sole sphere is at/below the touch height" flags."""
        z = self.config.touch_z if touch_z is None else float(touch_z)
        pts = self.sole_points(data)
        return np.array([bool(pts[i, :, 2].min() < z) for i in range(2)])

    def contact_points(self, data: mujoco.MjData) -> np.ndarray:
        """(k, 2) world xy of the actual floor <-> foot contacts this step."""
        pts = []
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            other = g2 if g1 == self.floor else (g1 if g2 == self.floor else -1)
            if other < 0:
                continue
            if int(self.model.geom_bodyid[other]) in self._bids:
                pts.append(np.asarray(con.pos, dtype=np.float64)[:2])
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)

    def support_hull(self, data: mujoco.MjData, kind: str | None = None) -> np.ndarray:
        """Convex hull of the supporting footprint (n, 2).

        ``"footprint"`` (default): the full 4-sphere footprint of every foot
        that is touching the mat -- the honest support area of a soft-contact
        foot (see the module docstring).  ``"contact"``: only the actual contact
        points reported by MuJoCo.
        """
        kind = self.config.hull_kind if kind is None else str(kind)
        if kind not in HULL_KINDS:
            raise ValueError(f"hull kind {kind!r} not in {HULL_KINDS}")
        if kind == "contact":
            return hull2d(self.contact_points(data))
        touch = self.foot_touch(data)
        pts = [self.sole_points(data)[i][:, :2] for i in range(2) if touch[i]]
        if not pts:
            return np.zeros((0, 2))
        return hull2d(np.vstack(pts))

    def com_state(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """``(com_xyz, com_vel_xyz)`` world; velocity is mass-weighted ``cvel``.

        MuJoCo's ``cvel`` is the com-based velocity in the *world* frame
        (rotational part about each body's CoM, translational part = that
        body's CoM velocity), so the mass-weighted mean of the translational
        components is the whole-robot CoM velocity.  This is the same formula
        ``src/solo/env.py`` uses for its world-frame CoM velocity (and
        ``solo.demo._Ids.com_vel``); the CoM position is the pelvis-subtree CoM
        (``data.subtree_com[pelvis]``).  Requires a ``mj_forward``/``mj_step``
        to have run -- both ``subtree_com`` and ``cvel`` are computed there.
        """
        com = np.asarray(data.subtree_com[self.pelvis], dtype=np.float64).reshape(3)
        vel = (self._mass[:, None] * np.asarray(data.cvel, dtype=np.float64)[:, 3:6]
               ).sum(axis=0) / self._mass_total
        return com, np.asarray(vel, dtype=np.float64).reshape(3)

    def measure(self, data: mujoco.MjData, hull_kind: str | None = None) -> StabilityState:
        """The full signal for the current ``data`` state."""
        cfg = self.config
        kind = cfg.hull_kind if hull_kind is None else str(hull_kind)
        com, vel = self.com_state(data)
        z_c = float(com[2]) - float(cfg.pivot_offset)
        tau = cp_time_constant(z_c, self._gravity, cfg.min_cp_height)
        cp = com[:2] + vel[:2] * tau
        hull = self.support_hull(data, kind)
        contact_hull = self.contact_points(data) if kind == "contact" else \
            self.support_hull(data, "contact")
        touch = self.foot_touch(data)
        st = StabilityState(
            t=float(data.time), com_xy=com[:2].copy(), com_z=float(com[2]),
            com_vel_xy=vel[:2].copy(), cp_xy=cp.copy(), tau=tau,
            z_c=min(max(z_c, cfg.min_cp_height), max(z_c, cfg.min_cp_height)),
            hull=hull, hull_kind=kind, contact_hull=contact_hull, foot_touch=touch)
        st.support_center = hull.mean(axis=0) if len(hull) else com[:2].copy()
        st.margin_cp = signed_margin(cp, hull)
        st.margin_com = signed_margin(com[:2], hull)
        st.brace_step_length = brace_step_length(com[:2], cp, hull)
        st.brace_direction = brace_direction(com[:2], cp)
        lead, side = self._brace_foot(data, hull, cp, touch)
        st.brace_foot = side
        st.brace_foot_travel = max(0.0, st.brace_step_length - lead)
        test = dcm_test(com[:2], vel[:2], z_c, hull, gravity=self._gravity,
                        min_z=cfg.min_cp_height)
        st.converging = bool(test["converging"])
        st.vdot = float(test["vdot"])
        st.cop_best = None if test["cop"] is None else np.asarray(test["cop"]).copy()
        st.unsafe = capture_point_unsafe(cp, hull, cfg.safe_margin)
        if len(hull) == 0:
            st.reason = "no foot contact: no support polygon"
        elif st.unsafe:
            st.reason = (f"capture point {st.margin_cp:+.3f} m outside the support: "
                         f"needs a {st.brace_step_length:.3f} m brace step")
        elif not st.converging:
            st.reason = "capture point on the boundary: DCM error cannot converge"
        else:
            st.reason = f"capture point {st.margin_cp:+.3f} m inside the support"
        return st

    def _brace_foot(self, data: mujoco.MjData, hull, cp, touch) -> tuple[float, str]:
        """Which foot can supply the brace, and the lead offset it provides (m).

        The foot whose footprint already reaches furthest along the CoM->CP
        direction is the one whose *existing* support already helps; the brace
        step is taken with the other one.  Returns ``(lead_offset, side)``.
        """
        pts = self.sole_points(data)
        u = brace_direction(self.com_state(data)[0][:2], cp)
        if not touch[0] and not touch[1]:
            return 0.0, "either"
        leads = [foot_lead_offset(pts[i][:, :2], u) if touch[i] else -math.inf
                 for i in range(2)]
        if leads[0] == leads[1]:
            return float(leads[0]), "either"
        # the foot already furthest along u is the one *supporting*; brace with
        # the other, whose own footprint supplies r_lead on landing
        brace_ix = 1 if leads[0] > leads[1] else 0
        return float(leads[brace_ix]), ("left" if brace_ix == 0 else "right")


#: the design document's capturability reward band (m): ``prior_art`` §5.1 --
#: 1 when the capture point is >= 5 cm inside the support, 0 when it exits.
CAPTURABILITY_BAND_M = 0.05


def capturability_reward(state: StabilityState,
                         band: float = CAPTURABILITY_BAND_M) -> float:
    """The doc's reward term, evaluated on a measured state: ``[0, 1]``.

    ``clip(band - d, 0, band) / band`` with ``d`` the (unsigned) distance from
    the capture point to the support hull -- 1 while the CP is inside the
    support, decaying to 0 once it is ``band`` outside (``prior_art`` §5.1:
    "1 when the capture point is >= 5 cm inside, 0 when it exits"; the formula
    is flat-1 inside, which is what is implemented here).

    The previous implementation added ``+margin_cp`` instead of subtracting the
    outward distance, i.e. it paid **1.0 exactly when the capture point was
    outside the hull** -- the crouch exploit it exists to kill.  Pinned by
    ``tests/solo/test_stability.py::test_capturability_reward_band_endpoints``.
    """
    b = float(band)
    if b <= 0.0:
        raise ValueError("band must be positive")
    if not math.isfinite(state.margin_cp):
        return 0.0
    outside = max(0.0, -float(state.margin_cp))
    return float(np.clip(b - outside, 0.0, b) / b)


__all__ = [
    "GRAVITY", "TOUCH_Z", "MIN_CP_HEIGHT", "HULL_KINDS", "CAPTURABILITY_BAND_M",
    "StabilityConfig", "StabilityState", "StabilityMonitor",
    "cp_time_constant", "capture_point", "hull2d", "signed_margin",
    "ray_hit_distance", "brace_direction", "brace_step_length",
    "foot_lead_offset", "foot_center_travel", "support_function", "dcm_test",
    "capture_point_unsafe", "capturability_reward",
]


if __name__ == "__main__":                              # self-check
    import mujoco

    from .scene import load_solo_model, stand_frame

    m = load_solo_model()
    d = mujoco.MjData(m)
    d.qpos[:] = stand_frame(m)[0]
    mujoco.mj_forward(m, d)
    st = StabilityMonitor(m).measure(d)
    print("stand keyframe:", st.as_dict())
    assert not st.unsafe and st.converging
    assert st.margin_cp > 0.0 and st.brace_step_length == 0.0
    # the crouch case: CoM inside, capture point outside -> UNSAFE
    crouch = capture_point([0.5, 0.5], [3.0, 0.0], 0.9)
    assert signed_margin([0.5, 0.5], np.array([[0.0, 0.0], [1.0, 0.0],
                                               [1.0, 1.0], [0.0, 1.0]])) > 0.0
    assert capture_point_unsafe(crouch, np.array([[0.0, 0.0], [1.0, 0.0],
                                                  [1.0, 1.0], [0.0, 1.0]]))
    print("self-check OK: stand safe; CoM-inside/CP-outside state unsafe")
