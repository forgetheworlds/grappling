"""Per-task reward term sets for the solo drill (SOLO_DRILL §5).

Design rules (verified by ``tests/test_solo.py``, hand-constructed states):

1. **A positive survival term is mandatory and dominates.**  ``alive`` is a
   per-step positive signal (upright x stance-height), and every task set must
   satisfy ``alive_weight > sum(per-step penalty weights)`` -- asserted at
   construction, tested.  Falling must never be preferable to standing.
2. **Tracking is always paired with uprightness.**  ``track_lin``/``track_ang``
   are multiplied by an uprightness gate, so "fall forward at commanded speed"
   cannot farm tracking reward (a measured exploit probe).
3. **Bounded penalties.**  Every penalty term is bounded by its weight per
   step, which is what makes rule 1 checkable.
4. **Potential-based progress for shot/recovery.**  ``shot_progress`` and
   ``recover_gain`` are ``gamma*phi(s') - phi(s)`` on physical progress
   (penetration depth / pelvis height): entering and *parking* earns nothing
   (and decays as ``(gamma-1)*phi``), which defeats the "enter a shot and never
   leave" and "knee-parked" exploits.
5. **Weights are placeholders.**  They start from the published locomotion
   recipe (Main's 2026-10-08 steering: track 1.0/sigma 0.5, yaw 1.0-2.0,
   feet_air_time 0.25-0.75 gated by ||cmd||>0.1, feet_slide -0.1,
   flat_orientation -1.0, termination -100..-200 excluding timeouts) and are
   **not** claimed to be tuned; S2 sets them from measured baselines.  The
   literature weights (``LIT_*`` constants and the ``balance_lit`` block of
   :class:`RewardWeights`) are the *sources'* weights, not placeholders: each
   one names its source, and every deviation from the source's number is
   marked in ``reports/2026-10-08/lit_balance.md``.
6. **Gamma** is configurable per task; defaults 0.995 (balance/stance/shot) and
   0.997 (locomotion) because at 50 Hz gamma=0.99 is only a ~2 s horizon.
   (Half-life table: ``solo.lit.gamma_table``.)
7. **Literature term sets** (``LIT_TERM_SETS``) are selected with
   ``TaskReward(term_set=...)`` and are not task families: they reuse a task's
   command/gamma defaults.  Source A's finding that *rewarding double foot
   contact harms standing* is why no contact term exists in
   ``balance_lit`` -- see the ``t_airtime``/``t_grf_even`` docstrings.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from .commands import DEFAULT_COMMAND, Command
from .lit import com_margin, support_centre
from .scene import N_JOINTS, STEP_DT

#: placeholder sigmas
SIGMA_LIN = 0.5          # m/s
SIGMA_ANG = 0.6          # rad/s
SIGMA_STANCE = 0.08      # m
SIGMA_WIDTH = 0.05       # m
SIGMA_REACH = 0.35       # m
#: feet_air_time gate: only rewarded when the command asks for motion
AIR_TIME_CMD_GATE = 0.1
MAX_AIR_TIME = 0.5       # s
#: default shot time budget before the timeout penalty applies
SHOT_BUDGET_S = 2.5

# ---------------------------------------------------------------- literature
# Parameters of the ``balance_lit`` term set.  Where a source supplies the
# constant it is used verbatim (name + citation in the docstring); where it does
# not, the parameter is ours and is marked as such in
# ``reports/2026-10-08/lit_balance.md``.
#: source A (arXiv:2404.19173) standing-velocity decay, e^{-5|v - c|}
LIT_VEL_K = 5.0
#: source B (arXiv:2002.02991) CoM-position tracking width (m).  Ours: the G1
#: foot half-length is 0.085 m, so 0.08 m makes the term fall to e^-1 at the
#: footprint edge.
LIT_SIGMA_COM = 0.08
#: source B CoM-velocity (capture-point) tracking width (m/s).  Ours: 2x the
#: gate's stability speed (0.15 m/s), so "stable" is e^-0.25 = 0.78 of the term.
LIT_SIGMA_CP = 0.30
#: source A roll+pitch orientation decay, e^{-30 qd(q_rp, c_rp)}; we use the same
#: constant on the squared tilt (rad^2).
LIT_ORIENT_K = 30.0
#: source A base-height decay, e^{-20 |pz - c_h|}
LIT_BASE_HEIGHT_K = 20.0
#: source A feet-airtime per-touchdown penalty (s); the same term pays the
#: airtime back, so a step near 0.4 s of airtime is free.
LIT_AIRTIME_PENALTY = 0.4
#: source A arm-posture decay, e^{-3 ||theta_arm - c_arm||}
LIT_ARM_K = 3.0
#: source A action-difference decay, e^{-0.02 sum|a_t - a_{t-1}|}
LIT_ACTION_DIFF_K = 0.02
#: source A torque decay, e^{-0.02 (1/N) sum |tau| / tau_max}; tau_max is ours
#: (``TORQUE_REF`` below, the ankle actuator limit, 50 N*m).
LIT_TORQUE_K = 0.02
#: source B even-GRF sigmoid width, on the dimensionless load asymmetry
#: (f_l - f_r)/(f_l + f_r).  Ours: 0.5 makes a 75/25 load split score e^-1.
LIT_GRF_SIGMA = 0.5
#: torque reference (N*m) for the source-A torque term, read from
#: ``robots/g1/g1.xml`` ``actuatorfrcrange``: the ankles and waist roll/pitch
#: actuators are the weakest balance actuators at 50 N*m (hip roll/knee 139,
#: hip pitch/yaw 88, legs are what matter here).
TORQUE_REF = 50.0
#: dense margin-normalisation for the "return to a valid stance" term (m): the
#: worst-cardinal-direction CoM margin at which the term saturates.  0.05 m is the
#: binding heel-ward margin of the stand pose, so the term is 0 exactly when the
#: CoM is a heel-margin inside the hull and 1 when it has left it.
LIT_RETURN_K = 0.05

#: The certified stance's own CoM projection minus its support-polygon centre
#: (measured on the stand keyframe with the term's own helpers:
#: ``sole_points_world`` + ``support_centre`` -> centre (0.0354828, 0.0),
#: CoM (0.0032830, 0.0000823)).  Source B's CoM-support term targets "the centre
#: of the support polygon", but the certified stance's CoM sits this far
#: heel-ward of the footprint centroid, so the raw term's optimum is ~3.2 cm
#: ahead of the certified stance: measured, moving the CoM to the centroid
#: raises the lit total by +2.34%, and the best ankle-pitch perturbation by
#: +4.1% (``reports/2026-10-08/reward_critic_audit.md`` §3).  The target is
#: therefore the centre PLUS this offset, rotated into the world by the heading,
#: which makes the certified stance the term's maximum.
STANCE_COM_OFFSET_XY: tuple[float, float] = (-0.0321998, 0.0000823)


def stance_com_target(centre_xy, heading_rad: float = 0.0) -> np.ndarray:
    """The lit CoM/CP target: support centre + the certified stance's offset.

    The offset is body-fixed (x forward), so it is rotated by the robot's world
    heading before it is applied.  At the certified stance (heading 0) the
    target is the stance's own CoM projection.
    """
    c, s = math.cos(float(heading_rad)), math.sin(float(heading_rad))
    ox, oy = STANCE_COM_OFFSET_XY
    return (np.asarray(centre_xy, np.float64).reshape(2)
            + np.array([c * ox - s * oy, s * ox + c * oy]))


@dataclass(frozen=True)
class RewardWeights:
    """Term weights.  Placeholders (module docstring rule 5), not tuned."""

    # positive per-step
    alive: float = 1.0
    track_lin: float = 1.0
    track_ang: float = 1.0
    stance_height: float = 0.5
    stance_width: float = 0.25
    feet_air_time: float = 0.5
    reach: float = 0.5
    shot_progress: float = 1.0
    shot_leg_seq: float = 0.3
    shot_knee: float = 0.1
    shot_exit: float = 0.5
    recover_gain: float = 1.0
    recover_done: float = 1.0
    # negative per-step (bounded by the weight)
    flat_orientation: float = 0.3
    feet_slide: float = 0.1
    action_rate: float = 0.05
    torque_sat: float = 0.05
    joint_limit: float = 0.05
    shot_timeout: float = 0.3
    low_posture: float = 0.3
    # one-off termination penalty (fall/dorsal; timeouts are not penalized)
    termination: float = 100.0
    # ------------------------------------------------ literature (balance_lit)
    # Source weights are used verbatim where the source gives one; the mapping
    # for each entry is in ``reports/2026-10-08/lit_balance.md``.  These are read
    # only by ``LIT_BALANCE_TERMS`` -- no existing task family sees them.
    upright: float = 1.0           # house rule 1 (source A's largest term is 1.0)
    vel_stand: float = 0.30        # source A x,y velocity 0.15 + 0.15 (standing branch)
    orientation: float = 0.20      # source A roll,pitch orient. (its largest term)
    base_height: float = 0.05      # source A base height
    com_support: float = 0.20      # source B CoM -> support-polygon centre (ours: = A's max)
    capture_point: float = 0.10    # source B CoM velocity -> capture point (ours: half of above)
    grf_even: float = 0.05         # source B even left/right GRF (ours: deliberately small)
    airtime: float = 1.0           # source A feet airtime (term is in seconds)
    arm_posture: float = 0.03      # source A arm position
    action_diff: float = 0.02      # source A action difference
    torque: float = 0.02           # source A torque
    #: "every behaviour must RETURN TO a good stance" (operator priority).  The
    #: dense term is worth more than any pose term (it is the goal), and the
    #: one-off bonus is paid only on a *valid* final stance.
    stance_return: float = 0.10
    return_bonus: float = 5.0

    def as_dict(self) -> dict:
        return asdict(self)

    def per_step_penalty_sum(self) -> float:
        return float(self.flat_orientation + self.feet_slide + self.action_rate
                     + self.torque_sat + self.joint_limit + self.shot_timeout
                     + self.low_posture)


#: term sets per task family (SOLO_DRILL §5)
TASK_TERMS: dict[str, tuple[str, ...]] = {
    "balance": ("alive", "flat_orientation", "action_rate", "torque_sat",
                "joint_limit"),
    "locomotion": ("alive", "track_lin", "track_ang", "feet_air_time",
                   "flat_orientation", "feet_slide", "action_rate",
                   "torque_sat", "joint_limit"),
    "stance": ("alive", "stance_height", "stance_width", "low_posture",
               "feet_air_time", "feet_slide", "flat_orientation", "action_rate",
               "torque_sat", "joint_limit"),
    "reach": ("alive", "reach", "track_lin", "flat_orientation", "feet_slide",
              "action_rate", "torque_sat", "joint_limit"),
    "shot": ("alive", "shot_progress", "shot_leg_seq", "shot_knee", "shot_exit",
             "shot_timeout", "flat_orientation", "feet_slide", "action_rate",
             "torque_sat", "joint_limit"),
    "recovery": ("alive", "recover_gain", "recover_done", "low_posture",
                 "flat_orientation", "action_rate", "torque_sat", "joint_limit"),
}

#: default gammas per task (rule 6)
TASK_GAMMA: dict[str, float] = {
    "balance": 0.995, "locomotion": 0.997, "stance": 0.995,
    "reach": 0.995, "shot": 0.995, "recovery": 0.995,
}

PENALTY_TERMS = frozenset({
    "flat_orientation", "feet_slide", "action_rate", "torque_sat",
    "joint_limit", "shot_timeout", "low_posture", "airtime", "stance_return",
})

#: worst-case per-step magnitude of a mixed-sign penalty term relative to its
#: weight.  ``airtime`` (source A) is in *seconds*: its floor is both feet
#: landing in the same control step with zero airtime, i.e. 2 x 0.4 s.  Every
#: other penalty term is bounded by its weight (margin 1.0), so the existing
#: term sets' arithmetic is unchanged.
PENALTY_MARGIN: dict[str, float] = {
    "airtime": 2.0 * LIT_AIRTIME_PENALTY,
}

#: literature balance term set (sources A + B).  Deliberately NOT in
#: ``TASK_TERMS``: it is a *term set*, selected by ``TaskReward(term_set=...)``,
#: not a task family (no command/push/horizon preset).
LIT_BALANCE_TERMS: tuple[str, ...] = (
    "upright", "vel_stand", "orientation", "base_height", "com_support",
    "capture_point", "grf_even", "airtime", "arm_posture", "action_diff",
    "torque", "stance_return",
)

LIT_TERM_SETS: dict[str, tuple[str, ...]] = {
    "balance_lit": LIT_BALANCE_TERMS,
    #: the movement stage (T1-movement / the T2 gate): the same balance set plus
    #: yaw-rate tracking, because ``vel_stand`` tracks only the linear command and
    #: a circle/step command's ``wz`` would otherwise carry no gradient at all --
    #: the T2 gate's ``yaw_err_abs_mean`` would then be unreachable by
    #: construction.  Additive: ``balance_lit`` is untouched.
    "movement_lit": LIT_BALANCE_TERMS + ("track_ang",),
}

#: every selectable term set (task families + literature sets)
ALL_TERM_SETS: dict[str, tuple[str, ...]] = {**TASK_TERMS, **LIT_TERM_SETS}


@dataclass
class RewardInputs:
    """All quantities the terms read (hand-constructible; see tests)."""

    dt: float = STEP_DT
    cmd: Command = field(default_factory=lambda: DEFAULT_COMMAND)
    # kinematics
    vel_local: np.ndarray = field(default_factory=lambda: np.zeros(2))
    yaw_rate: float = 0.0
    torso_up_z: float = 1.0
    pelvis_z: float = 0.79
    pelvis_z_prev: float = 0.79
    stand_height: float = 0.79
    stance_width_meas: float = 0.237
    # feet
    foot_contact: tuple[bool, bool] = (True, True)
    foot_slip: tuple[float, float] = (0.0, 0.0)
    foot_air_time: tuple[float, float] = (0.0, 0.0)
    foot_landed: tuple[bool, bool] = (False, False)
    # actuation
    action: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))
    prev_action: np.ndarray | None = None
    sat_frac: float = 0.0
    limit_prox: float = 0.0
    # reach
    hand_distance: float | None = None
    # shot
    shot_depth: float = 0.0
    shot_depth_prev: float = 0.0
    shot_phase: float = 0.0
    shot_time: float = 0.0
    shot_budget_s: float = SHOT_BUDGET_S
    shot_leg_ahead: bool = False
    shot_knee_control: bool = False
    shot_exited: bool = False
    # recovery / terminal
    recovered: bool = False
    dorsal: bool = False
    # literature terms (sources A/B).  World frame for the CoM/hull/GRF group
    # (all three in one frame, so the capture-point term is frame-invariant);
    # only the ``balance_lit`` term set reads them.
    com_xy: np.ndarray | None = None           # (2,) world CoM ground position
    com_vel_xy: np.ndarray | None = None       # (2,) world CoM horizontal velocity
    com_z: float = 0.0                         # world CoM height (CP scale)
    #: world yaw of the robot (rad): rotates the body-fixed stance CoM offset
    #: into the world for the com_support / capture_point targets
    heading_rad: float = 0.0
    sole_points: np.ndarray | None = None      # (2, 4, 3) world sole spheres
    foot_load: tuple[float, float] = (0.0, 0.0)   # per-foot vertical GRF (N)
    torque: np.ndarray | None = None           # (29,) actuator force (N*m)
    arm_dev: float = 0.0                       # sum |theta_arm - c_arm| (rad)
    gravity: float = 9.81
    #: support hull (n, 2) of the loaded feet, in the SAME frame as ``com_xy``
    #: (the dense return term needs the margin, not just the centre); and the
    #: shared stance predicate's verdict for the current step
    hull: np.ndarray | None = None
    stance_valid: bool = False


def _clamp01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _gate(inp: RewardInputs) -> float:
    return _clamp01(inp.torso_up_z)


# ------------------------------------------------------------------ positive
def t_alive(inp: RewardInputs) -> float:
    """Positive survival: upright and at (or above) the commanded stance height."""
    ref = max(0.4, float(inp.cmd.stance_height))
    return _clamp01(inp.torso_up_z) * _clamp01(inp.pelvis_z / ref)


def t_track_lin(inp: RewardInputs, sigma: float = SIGMA_LIN) -> float:
    d2 = float(np.sum((np.asarray(inp.vel_local, np.float64)
                       - np.array([inp.cmd.vx, inp.cmd.vy])) ** 2))
    return math.exp(-d2 / (sigma * sigma)) * _gate(inp)


def t_track_ang(inp: RewardInputs, sigma: float = SIGMA_ANG) -> float:
    d = float(inp.yaw_rate) - float(inp.cmd.wz)
    return math.exp(-(d * d) / (sigma * sigma)) * _gate(inp)


def t_stance_height(inp: RewardInputs, sigma: float = SIGMA_STANCE) -> float:
    d = float(inp.pelvis_z) - float(inp.cmd.stance_height)
    return math.exp(-(d * d) / (sigma * sigma)) * _gate(inp)


def t_stance_width(inp: RewardInputs, sigma: float = SIGMA_WIDTH) -> float:
    d = float(inp.stance_width_meas) - float(inp.cmd.stance_width)
    return math.exp(-(d * d) / (sigma * sigma)) * _gate(inp)


def t_feet_air_time(inp: RewardInputs, max_air: float = MAX_AIR_TIME) -> float:
    """Step-quality bonus: credit a foot that just landed after real airtime.

    Gated by the command (``|v_cmd| > AIR_TIME_CMD_GATE`` or a yaw command), so
    standing still cannot farm it (the "staying still" exploit).
    """
    moving = (math.hypot(inp.cmd.vx, inp.cmd.vy) > AIR_TIME_CMD_GATE
              or abs(inp.cmd.wz) > AIR_TIME_CMD_GATE)
    if not moving:
        return 0.0
    r = 0.0
    for landed, air in zip(inp.foot_landed, inp.foot_air_time):
        if landed:
            r += min(float(air), float(max_air))
    return float(r)


def t_reach(inp: RewardInputs, sigma: float = SIGMA_REACH) -> float:
    if inp.hand_distance is None:
        return 0.0
    d = float(inp.hand_distance)
    return math.exp(-(d * d) / (sigma * sigma)) * _gate(inp)


# ------------------------------------------------------------------- shot
def t_shot_progress(inp: RewardInputs, gamma: float = 0.995) -> float:
    """Potential-based penetration progress (rule 4): no parking credit."""
    phi_prev = _clamp01(inp.shot_depth_prev)
    phi_next = _clamp01(inp.shot_depth)
    return float(gamma * phi_next - phi_prev)


def t_shot_leg_seq(inp: RewardInputs) -> float:
    return 1.0 if (inp.shot_leg_ahead and inp.shot_phase > 0.0) else 0.0


def t_shot_knee(inp: RewardInputs) -> float:
    return 1.0 if inp.shot_knee_control else 0.0


def t_shot_exit(inp: RewardInputs) -> float:
    return 1.0 if inp.shot_exited else 0.0


def t_shot_timeout(inp: RewardInputs) -> float:
    return -1.0 if inp.shot_time > inp.shot_budget_s else 0.0


# --------------------------------------------------------------- recovery
def t_recover_gain(inp: RewardInputs, gamma: float = 0.995) -> float:
    """Potential on pelvis-height recovery (progress, not clock)."""
    ref = max(0.4, float(inp.stand_height))
    phi_prev = _clamp01(inp.pelvis_z_prev / ref)
    phi_next = _clamp01(inp.pelvis_z / ref)
    return float(gamma * phi_next - phi_prev)


def t_recover_done(inp: RewardInputs) -> float:
    return 1.0 if inp.recovered else 0.0


def t_low_posture(inp: RewardInputs) -> float:
    ref = max(0.4, float(inp.stand_height))
    return -_clamp01(1.0 - float(inp.pelvis_z) / ref)


# --------------------------------------------------------------- negative
def t_flat_orientation(inp: RewardInputs) -> float:
    """Normalized tilt cost in [0, 1] (1 = fully collapsed/upside down)."""
    return -(1.0 - float(np.clip(inp.torso_up_z, -1.0, 1.0))) / 2.0


def t_feet_slide(inp: RewardInputs) -> float:
    cost = 0.5 * (_clamp01(abs(inp.foot_slip[0]) / 2.0)
                  + _clamp01(abs(inp.foot_slip[1]) / 2.0))
    return -float(cost)


def t_action_rate(inp: RewardInputs) -> float:
    if inp.prev_action is None:
        return 0.0
    d = float(np.abs(np.asarray(inp.action, np.float64)
                     - np.asarray(inp.prev_action, np.float64)).mean())
    return -_clamp01(d / 0.5)


def t_torque_sat(inp: RewardInputs) -> float:
    return -_clamp01(inp.sat_frac)


def t_joint_limit(inp: RewardInputs) -> float:
    return -_clamp01(inp.limit_prox)


# ------------------------------------------------------------- literature
# Sources: A = van Marum et al. 2024 (arXiv:2404.19173), B = Yang et al. 2020
# (arXiv:2002.02991).  Every positive term below is gated by ``_gate``
# (house rule 2) except the ones whose own shape already vanishes when the robot
# is not upright (noted per term); without the gate a *lying* robot could farm
# the CoM/support terms.
def t_upright(inp: RewardInputs) -> float:
    """Survival: torso uprightness alone.

    Source A has no height-gated survival term (their reward is a sum of exp
    errors, with base height carrying the height signal separately).  Splitting
    the pelvis-height factor out of ``alive`` is what keeps this term from
    penalising the crouch a recovery step needs.
    """
    return _clamp01(inp.torso_up_z)


def t_vel_stand(inp: RewardInputs, k: float = LIT_VEL_K) -> float:
    """Source A standing branch: ``e^{-5 |v_xy - c_xy|}`` (weight 0.15 + 0.15)."""
    d = (np.asarray(inp.vel_local, np.float64).reshape(2)
         - np.array([float(inp.cmd.vx), float(inp.cmd.vy)]))
    return math.exp(-float(k) * float(np.linalg.norm(d))) * _gate(inp)


def t_orientation(inp: RewardInputs, k: float = LIT_ORIENT_K) -> float:
    """Source A roll,pitch orientation ``e^{-30 qd}`` on the squared tilt (rad^2).

    Self-gating (the exp vanishes as the torso leaves vertical), which is why
    source A could give it the largest weight in their table (0.2) without a
    separate uprightness gate.
    """
    tilt = math.acos(float(np.clip(inp.torso_up_z, -1.0, 1.0)))
    return math.exp(-float(k) * tilt * tilt)


def t_base_height(inp: RewardInputs, k: float = LIT_BASE_HEIGHT_K) -> float:
    """Source A base height, verbatim: ``e^{-20 |pz - c_h|}`` (weight 0.05)."""
    return math.exp(-float(k) * abs(float(inp.pelvis_z) - float(inp.stand_height)))


def _support_centre(inp: RewardInputs) -> np.ndarray:
    """Area centroid of the loaded feet's sole hull; NaN when nothing loaded."""
    if inp.sole_points is None:
        return np.full(2, np.nan)
    return support_centre(np.asarray(inp.sole_points, np.float64),
                          (bool(inp.foot_contact[0]), bool(inp.foot_contact[1])))


def t_com_support(inp: RewardInputs, sigma: float = LIT_SIGMA_COM) -> float:
    """Source B: CoM horizontal position -> the support centre (+ the certified
    stance's own offset; see :data:`STANCE_COM_OFFSET_XY`).

    "to provide maximum disturbance compensation": with the CoM over the hull
    centre the minimum distance to any support edge is maximal.  The target is
    the *certified stance's* CoM projection, not the raw centroid, so the
    certified stance is the term's maximum (the raw centroid would pull the
    policy ~3.2 cm forward).  Zero when no foot is loaded (an airborne CoM has
    no support centre to sit over).
    """
    if inp.com_xy is None:
        return 0.0
    sc = _support_centre(inp)
    if not np.all(np.isfinite(sc)):
        return 0.0
    target = stance_com_target(sc, inp.heading_rad)
    d = np.asarray(inp.com_xy, np.float64).reshape(2) - target
    return math.exp(-float(d @ d) / (float(sigma) ** 2)) * _gate(inp)


def t_capture_point(inp: RewardInputs, sigma: float = LIT_SIGMA_CP) -> float:
    """Source B: CoM velocity -> the capture point implied by the support centre
    (+ the certified stance's offset; see :data:`STANCE_COM_OFFSET_XY`).

    ``x_CP = x_CoM + x_dot_CoM sqrt(z_c / g)`` (their eq. 5); the term is maximal
    when the CoM velocity already equals ``(x_target - x_CoM) / sqrt(z_c/g)``,
    i.e. when the CoM's capture point sits exactly on the target.  Vertical
    target is 0 (this term reads only the horizontal components).
    """
    if inp.com_xy is None or inp.com_vel_xy is None:
        return 0.0
    sc = _support_centre(inp)
    z = float(inp.com_z)
    if not np.all(np.isfinite(sc)) or z <= 1e-9:
        return 0.0
    tau = math.sqrt(z / float(inp.gravity))
    target = stance_com_target(sc, inp.heading_rad)
    v_target = (target - np.asarray(inp.com_xy, np.float64).reshape(2)) / tau
    dv = np.asarray(inp.com_vel_xy, np.float64).reshape(2) - v_target
    return math.exp(-float(dv @ dv) / (float(sigma) ** 2)) * _gate(inp)


def t_grf_even(inp: RewardInputs, sigma: float = LIT_GRF_SIGMA) -> float:
    """Source B eq. 12: even left/right ground-reaction-force distribution.

    ``exp(-((f_l - f_r)/(f_l + f_r) / sigma)^2)``; exactly 1.0 when the two foot
    loads are equal and 0 when one foot carries nothing.  Zero when no load is
    measured (airborne) instead of claiming perfect evenness.
    """
    fl = float(inp.foot_load[0])
    fr = float(inp.foot_load[1])
    total = fl + fr
    if total <= 1e-9:
        return 0.0
    asym = (fl - fr) / total
    return math.exp(-(asym / float(sigma)) ** 2) * _gate(inp)


def t_airtime(inp: RewardInputs) -> float:
    """Source A feet airtime, verbatim: per touchdown, ``t_air - 0.4`` (s).

    Fires once per foot per landing (``foot_landed``), 0 on every other step:
    a 0.4 s step is free, a contact-chatter landing is penalised -0.4, and
    (their stated purpose) step frequency is regularised without a clock.
    """
    return float(sum(float(air) - LIT_AIRTIME_PENALTY
                     for landed, air in zip(inp.foot_landed, inp.foot_air_time)
                     if landed))


def t_arm_posture(inp: RewardInputs, k: float = LIT_ARM_K) -> float:
    """Source A arm position ``e^{-3 ||theta_arm - c_arm||}`` (weight 0.03).

    ``arm_dev`` is the summed absolute arm-joint deviation from the stand
    keyframe, in rad; 1.0 when the arms are exactly on the keyframe (which is
    the case by construction when the joint mask freezes them).
    """
    return math.exp(-float(k) * float(inp.arm_dev))


def t_action_diff(inp: RewardInputs, k: float = LIT_ACTION_DIFF_K) -> float:
    """Source A action difference ``e^{-0.02 sum|a_t - a_{t-1}|}`` (weight 0.02).

    ``prev_action`` is the ctrl vector (as in ``t_action_rate``); the exp form
    starts at its maximum 1.0 on the first step of an episode, so a reset cannot
    be read as a penalty.
    """
    if inp.prev_action is None:
        return 1.0
    d = float(np.abs(np.asarray(inp.action, np.float64)
                     - np.asarray(inp.prev_action, np.float64)).sum())
    return math.exp(-float(k) * d)


def t_torque(inp: RewardInputs, k: float = LIT_TORQUE_K,
             tau_ref: float = TORQUE_REF) -> float:
    """Source A torque ``e^{-0.02 (1/N) sum|tau| / tau_max}`` (weight 0.02).

    ``tau_ref`` is :data:`TORQUE_REF` (the ankle actuator limit, the binding
    balance actuator); 1.0 when no torque is measured.
    """
    if inp.torque is None:
        return 1.0
    t = np.abs(np.asarray(inp.torque, np.float64).reshape(-1))
    if t.size == 0:
        return 1.0
    return math.exp(-float(k) * float(t.mean()) / float(tau_ref))


def t_stance_return(inp: RewardInputs, k: float = LIT_RETURN_K) -> float:
    """Dense "come back to a valid stance" shaping (potential, rule 4).

    Signed term, matching this module's convention (positive = good, negative =
    cost): **0.0 while the CoM is inside the support hull, falling to -1.0 once
    it has left it**, with the full depth at a ``k``-sized negative margin.  The
    dense companion of the one-off terminal stance bonus -- a bonus alone is one
    reward every 400 steps and cannot pull a policy back (the same reason
    ``recover_gain`` is a potential and not a clock).

    No hull (the ``--stance-return`` wiring is off) or a degenerate hull (< 3
    loaded sole points) returns 0.0: the term never fires on geometry it cannot
    measure; the fall detector and the one-off bonus own those cases.
    """
    if inp.hull is None or inp.com_xy is None:
        return 0.0
    m = com_margin(inp.com_xy, inp.hull)
    if not np.isfinite(m):
        return 0.0
    return -float(np.clip(-m / float(k), 0.0, 1.0))


TERM_FUNCS = {
    "alive": t_alive,
    "track_lin": t_track_lin,
    "track_ang": t_track_ang,
    "stance_height": t_stance_height,
    "stance_width": t_stance_width,
    "feet_air_time": t_feet_air_time,
    "reach": t_reach,
    "shot_progress": t_shot_progress,
    "shot_leg_seq": t_shot_leg_seq,
    "shot_knee": t_shot_knee,
    "shot_exit": t_shot_exit,
    "shot_timeout": t_shot_timeout,
    "recover_gain": t_recover_gain,
    "recover_done": t_recover_done,
    "low_posture": t_low_posture,
    "flat_orientation": t_flat_orientation,
    "feet_slide": t_feet_slide,
    "action_rate": t_action_rate,
    "torque_sat": t_torque_sat,
    "joint_limit": t_joint_limit,
    "upright": t_upright,
    "vel_stand": t_vel_stand,
    "orientation": t_orientation,
    "base_height": t_base_height,
    "com_support": t_com_support,
    "capture_point": t_capture_point,
    "grf_even": t_grf_even,
    "airtime": t_airtime,
    "arm_posture": t_arm_posture,
    "action_diff": t_action_diff,
    "torque": t_torque,
    "stance_return": t_stance_return,
}


class TaskReward:
    """Reward assembly for one task family (or a literature term set).

    ``term_set`` selects one of :data:`LIT_TERM_SETS` instead of the task
    family's terms; the task still supplies the command/gamma defaults, so
    ``TaskReward("balance", term_set="balance_lit")`` is the literature
    balance reward on the balance task.
    """

    def __init__(self, task: str = "balance", weights: RewardWeights | None = None,
                 gamma: float | None = None,
                 termination_penalty: float | None = None,
                 term_set: str | None = None):
        if task not in TASK_TERMS:
            raise ValueError(f"unknown task {task!r}; choose from {sorted(TASK_TERMS)}")
        if term_set is not None and term_set not in LIT_TERM_SETS:
            raise ValueError(f"unknown term set {term_set!r}; choose from "
                             f"{sorted(LIT_TERM_SETS)} (or None for the task family)")
        self.task = task
        self.term_set = term_set
        self.weights = weights or RewardWeights()
        self.gamma = float(TASK_GAMMA[task] if gamma is None else gamma)
        self.termination_penalty = float(
            self.weights.termination if termination_penalty is None
            else termination_penalty)
        self.terms = TASK_TERMS[task] if term_set is None else LIT_TERM_SETS[term_set]
        # invariant: the dominant positive term dominates the per-step penalties.
        # For every task family that term is ``alive`` (unchanged behaviour); for
        # a literature set it is the largest positive weight in the set (the
        # `upright` term), and the *worst case* of a mixed-sign penalty is used
        # rather than its weight (PENALTY_MARGIN).
        positive = [float(getattr(self.weights, t)) for t in self.terms
                    if t not in PENALTY_TERMS]
        dominant = (float(self.weights.alive) if term_set is None
                    else (max(positive) if positive else 0.0))
        pen = float(sum(getattr(self.weights, t) * PENALTY_MARGIN.get(t, 1.0)
                        for t in self.terms if t in PENALTY_TERMS))
        if not dominant > pen:
            raise ValueError(
                f"dominant positive weight {dominant:.3f} must exceed the "
                f"worst-case sum of per-step penalty weights ({pen:.3f}) for "
                f"{term_set or task!r}")

    # ---------------------------------------------------------------- pieces
    def term_value(self, name: str, inp: RewardInputs) -> float:
        f = TERM_FUNCS[name]
        if name in ("shot_progress",):
            return f(inp, self.gamma)
        if name in ("recover_gain",):
            return f(inp, self.gamma)
        return f(inp)

    def step(self, inp: RewardInputs) -> tuple[float, dict[str, float]]:
        """Per-step reward and the per-term log (all task terms present)."""
        terms: dict[str, float] = {}
        total = 0.0
        w = self.weights
        for name in self.terms:
            v = float(self.term_value(name, inp))
            terms[name] = v
            total += float(getattr(w, name)) * v
        return total, terms

    def terminal(self, cause: str | None) -> tuple[float, dict[str, float]]:
        """One-off terminal reward.  Falls and dorsal contact are penalized;
        timeouts are not (they are not failures, and clock-based penalties
        create the 'fall to stop paying' pathology).

        In the ``balance_lit`` return-to-stance mode the env additionally reports
        ``no_recovery`` as the cause when the episode's time limit is reached
        while the robot is *not* in a valid stance and not recovering, which IS a
        failure ("a non-stance ending must be a FAILURE, not a neutral
        truncation") and shares the fall penalty.
        """
        if cause in ("fall", "dorsal", "no_recovery", "stance"):
            return -self.termination_penalty, {"termination": -self.termination_penalty}
        return 0.0, {"termination": 0.0}

    def final(self, inp: RewardInputs | None, cause: str | None
              ) -> tuple[float, dict[str, float]]:
        """One-off end-of-episode reward in the return-to-stance mode.

        Paid only when the episode's FINAL state is a valid stance (the operator's
        "every behaviour must return to a good stance"); 0.0 for every other term
        set, so the default tasks are bit-unchanged.
        """
        if self.term_set is None or "stance_return" not in self.terms:
            return 0.0, {}
        if inp is None or not bool(inp.stance_valid):
            return 0.0, {"return_bonus": 0.0}
        return self.weights.return_bonus, {"return_bonus": self.weights.return_bonus}

    def as_dict(self) -> dict:
        return {
            "task": self.task,
            "term_set": self.term_set,
            "gamma": self.gamma,
            "termination_penalty": self.termination_penalty,
            "terms": list(self.terms),
            "weights": self.weights.as_dict(),
        }


def discounted_return(rewards, gamma: float, terminal_penalty: float = 0.0) -> float:
    """v0 = sum gamma^t r_t (+ gamma^T * terminal_penalty for a terminated run).

    Used by the hand-horizon test: a collapsing policy must score lower than an
    upright one over the same fixed horizon.
    """
    v = 0.0
    for t, r in enumerate(rewards):
        v += (gamma ** t) * float(r)
    return float(v + (gamma ** len(rewards)) * float(terminal_penalty))


if __name__ == "__main__":  # self-check
    tr = TaskReward("locomotion")
    good = RewardInputs(cmd=Command(vx=0.3, vy=0.0), vel_local=np.array([0.3, 0.0]),
                        torso_up_z=1.0, pelvis_z=0.79)
    r_good, terms = tr.step(good)
    bad = replace(good, torso_up_z=0.2, pelvis_z=0.2,
                  vel_local=np.array([0.3, 0.0]), foot_slip=(1.5, 1.5))
    r_bad, terms_bad = tr.step(bad)
    print("reward self-check:", {"good": round(r_good, 4), "collapsed": round(r_bad, 4),
                                "terms_good": {k: round(v, 3) for k, v in terms.items()}})
    assert r_good > r_bad
    # hand horizon: 50 steps upright vs 10 steps upright then collapse + fall
    upright = [tr.step(good)[0]] * 50
    falling = [tr.step(good)[0]] * 10 + [tr.step(bad)[0]] * 40
    v_up = discounted_return(upright, tr.gamma)
    v_fall = discounted_return(falling, tr.gamma, tr.terminal(good.cmd and "fall")[0])
    assert v_up > v_fall, (v_up, v_fall)
    print("horizon check:", {"upright": round(v_up, 3), "collapse": round(v_fall, 3)})
    # movement set: the balance set + yaw tracking, and the yaw term must actually
    # respond to the command's wz (a turn error is the only thing it can see)
    assert LIT_TERM_SETS["movement_lit"] == LIT_BALANCE_TERMS + ("track_ang",)
    mv = TaskReward("locomotion", term_set="movement_lit")
    turning = RewardInputs(cmd=Command(vx=0.3, vy=0.0, wz=0.4), yaw_rate=0.4,
                           vel_local=np.array([0.3, 0.0]), torso_up_z=1.0, pelvis_z=0.79)
    r_turn, terms_turn = mv.step(turning)
    r_still, terms_still = mv.step(replace(turning, yaw_rate=0.0))
    assert terms_turn["track_ang"] > terms_still["track_ang"], (terms_turn, terms_still)
    print("movement check:", {"tracking_wz": round(terms_turn["track_ang"], 4),
                              "ignoring_wz": round(terms_still["track_ang"], 4),
                              "terms": len(terms_turn)})
