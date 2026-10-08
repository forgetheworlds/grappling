"""Executability checker for S5 penetration-step demo traces.

What this checks, and why these criteria
----------------------------------------
A demo trace is *executable* when the single G1 actually performed the
retargeted penetration step rather than merely tracking a pose track: it stayed
up, its lead foot genuinely stepped (planted after an airborne phase, no slide
while loaded), its penetrating knee reached the reference's lowering depth, its
torso followed the reference's pitch instead of collapsing, its CoM stayed
inside the measured support envelope except in the reference's own infeasible
phases, it retained the reference's *travel* (a penetration step travels; a
treadmill step in place does not), and no actuator/joint limit was exceeded.

Every threshold below carries its source; nothing is invented:

* fall rule        -- ``src/solo/fall.py`` (S1 contract; thresholds measured in
                      its docstring: prone 0.076 m / 89 deg, supine 0.32 m /
                      121 deg, kneel 0.498 m / knee contacts, hands-plant
                      0.124 m; knees/hands never terminal).
* contact rule     -- sole-sphere centre < 0.045 m, the same rule the support
                      audit used (``data/support_envelope.json`` meta
                      ``contact_rule``) and the teacher uses (FOOT_Z_TOUCH).
* loaded foot      -- sole centre < 0.015 m (``drill.kin.TOUCH_Z``: "clearly
                      loaded") and the CoM over that foot's footprint
                      (``drill.metrics._loaded`` semantics).
* CoM envelope     -- ``data/support_envelope.json`` (margin band 0.02 m,
                      per-frame reference margins recomputed by FK) plus the
                      delivered L2 run's single-support transient depth
                      (-0.026 m, ``data/solo_drill/final_L2_motion.json``).
* pitch tolerance  -- ``teacher.episode.TILT_SLACK_DEG = 15`` ("allowance over
                      the reference's own tilt").
* slide threshold  -- drill rubric B1 / ``data/drill/M_D28c_...json``
                      (per-step loaded displacement 2.6-5.6 mm, bar 20 mm).
* saturation       -- ``src/solo/metrics.py`` (jnt_actfrcrange at 0.95) and
                      ``src/retarget/solve.py`` V_MAX = 6.0 rad/s (joints have
                      no velocity limit in the model; the retargeter's bound is
                      the model-side constraint in use).
* travel           -- ``data/refs_video/retarget_summary.json`` meta
                      (``net_travel_m``; "the root translation is
                      reconstructed from foot-contact anchoring ... so a
                      penetration step actually travels instead of running on
                      a treadmill", docs/references/yt_gBAhX5t-GW4_index.md §8).

The module is pure numpy: the capture (``solo.demo``) measures the per-tick
fields, and the checker can be run on any stored trace + spec pair, including
hand-built synthetic ones (the discrimination tests do exactly that).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Mapping

import numpy as np

# ---------------------------------------------------------------------------
# thresholds + sources (see the module docstring for the full citation list)
# ---------------------------------------------------------------------------
TOUCH_Z = 0.045                 # contact rule (audit support_envelope.json)
LOAD_TOUCH_Z = 0.015            # drill.kin.TOUCH_Z -- "clearly loaded"
FALL_PELVIS_GROUND = 0.35       # solo.fall.FallDetConfig
FALL_TILT_GROUND_DEG = 60.0     # solo.fall.FallDetConfig
FALL_PELVIS_HARD = 0.22         # solo.fall.FallDetConfig
FALL_CONFIRM_S = 0.25           # solo.fall.FallDetConfig
KNEE_CONTACT_MARGIN = 0.05      # knee "in contact range" = ref min + this
V_MAX_JOINT = 6.0               # retarget.solve.V_MAX (rad/s)
SAT_FRACTION = 0.95             # solo.metrics._SAT_LIMIT_FRACTION

#: segments of a repaired demo track (see solo.demo.build_repaired_track)
SEG_ENTRY, SEG_BLEND, SEG_RECOVER, SEG_STAND, SEG_HOLD = 0, 1, 2, 3, 4
SEG_NAMES = {SEG_ENTRY: "entry", SEG_BLEND: "blend", SEG_RECOVER: "recover",
             SEG_STAND: "stand", SEG_HOLD: "hold"}
REFERENCE_SEGMENTS = (SEG_ENTRY, SEG_RECOVER)

CRITERIA_NAMES = (
    "no_fall", "contact_sequence", "torso_pitch", "com_margin",
    "knee_depth", "no_saturation", "no_foot_slide", "travel",
    "terminal_stance",
)


@dataclass(frozen=True)
class Tol:
    """Criterion tolerances; every value has a source (module docstring).

    ``pitch_deg`` is derived from the retarget's own fit quality: the video
    retarget's worst landmark error is 0.149 m (``rms.a_max`` in
    ``data/refs_video/retarget_summary.json``), which over the torso lever
    (pelvis -> head ~0.5 m) is ~16.6 deg of angular error; 15 deg is the band
    around the reference's own pitch track.  The collapse cases this must
    reject deviate by >= 40 deg, far outside it.
    """

    plant_s: float = 0.30           # event timing vs the reference
    knee_s: float = 0.30
    knee_depth_m: float = 0.03      # reach ref min knee height + this
    pitch_deg: float = 15.0         # retarget fit: a_max 0.149 m / 0.5 m lever
    margin_m: float = 0.03          # delivered L2 transient depth 0.026
    slide_m: float = 0.02           # drill rubric B1 bar
    rise_window_s: float = 12.0     # the operator's rise is slow (7.96 s track)
    rise_hold_s: float = 0.30
    vel_rad_s: float = V_MAX_JOINT
    sat_frac: float = 1.0           # fail only above the model limit
    travel_frac: float = 0.60       # retain >= 60 % of the reference travel
    travel_slack_m: float = 0.15    # lateral mismatch allowance
    travel_max_extra_m: float = 0.25

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# reference tracks (measured by solo.demo.measure_spec)
# ---------------------------------------------------------------------------
@dataclass
class RefTracks:
    """Reference-derived arrays on the repaired track's time grid.

    All arrays share the same length ``M`` and are indexed by
    ``round(ref_time / dt)``; ``ref_time`` is finite only on the reference
    segments (entry / recover).
    """

    grid_t: np.ndarray              # (M,) uniform track time grid, no NaN
    ref_time: np.ndarray            # (M,) s, NaN on blend/hold
    segment: np.ndarray             # (M,) int8
    phase: np.ndarray               # (M,) '<U5' teacher phase labels
    pitch_deg: np.ndarray           # (M,) signed sagittal torso lean
    margin: np.ndarray              # (M,) signed CoM margin, NaN if airborne
    contact: np.ndarray             # (M, 2) bool per foot
    knee_z: np.ndarray              # (M, 2) knee site heights
    pelvis_z: np.ndarray            # (M,)
    base_xy: np.ndarray             # (M, 2)
    knee_site_xy: np.ndarray        # (M, 2, 2) knee site xy (travel diagnosis)
    #: landmark site positions (M, 19, 3) -- the imitation objective's targets
    site_pos: np.ndarray = None     # type: ignore[assignment]

    def save(self, path) -> None:
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path) -> "RefTracks":
        with np.load(path, allow_pickle=True) as z:
            return cls(**{k: z[k] for k in cls.__dataclass_fields__})


@dataclass
class ExecSpec:
    """What the trace claims to be, plus the reference-derived thresholds."""

    technique: str = ""
    role: str = "a"
    sources: tuple = ()             # raw reference npz paths
    sha256: tuple = ()
    dt: float = 0.02
    duration_s: float = 0.0         # repaired track duration
    n_frames: int = 0
    #: the foot that steps/plants (the support foot at knee-down)
    lead_side: str = "right"
    lead_plant_t: float | None = None      # reference rising edge, track time
    #: the knee that reaches the reference's lowest point (the penetration knee)
    knee_side: str = "left"
    knee_min_z: float = 0.0                # reference minimum knee-site height
    knee_min_t: float = 0.0
    knee_contact_z: float = 0.0            # knee_min_z + KNEE_CONTACT_MARGIN
    knee_contact_t: float | None = None    # first crossing in the reference
    #: reference entry travel in the entry's initial heading frame
    travel_forward_m: float = 0.0
    travel_lateral_m: float = 0.0
    travel_net_m: float = 0.0
    #: last reference time of the entry segment (shot progress normaliser)
    entry_end_t: float = 0.0
    #: reference phase table [(kind, t0, t1)] (reporting only)
    phase_segments: list = field(default_factory=list)
    #: rise target (the recover track's own end state, minus a small band)
    rise_pelvis_z: float = 0.0
    rise_knee_z: float = 0.0
    #: terminal stance thresholds, measured from the verified stance (see
    #: solo.demo.measure_spec; all cited in the report)
    stance_pelvis_z: float = 0.0        # the stance keyframe's pelvis height
    stance_pelvis_lo: float = 0.0       # CommandRanges.stance_height bounds
    stance_pelvis_hi: float = 0.0
    stance_tilt_max_deg: float = 0.0    # measured stance tilt + margin
    stance_margin_min: float = 0.0      # audit STABLY_FEASIBLE band (0.02 m)
    stance_speed_max: float = 0.0       # drill settle/lift velocity gates
    stance_flat_tol_m: float = 0.0      # "flat": all four sole centres near rest
    sole_rest_z: float = 0.0            # measured sole-centre height at rest
    term_window_s: float = 0.4          # the terminal window the criteria read
    tol: Tol = field(default_factory=Tol)
    tracks: RefTracks | None = None

    @property
    def lead_ix(self) -> int:
        return 0 if self.lead_side == "left" else 1

    @property
    def knee_ix(self) -> int:
        return 0 if self.knee_side == "left" else 1

    def scalars(self) -> dict:
        """JSON-able summary (no arrays)."""
        d = {k: getattr(self, k) for k in self.__dataclass_fields__
             if k not in ("tol", "tracks")}
        d["sources"] = list(self.sources)
        d["sha256"] = list(self.sha256)
        d["tol"] = self.tol.as_dict()
        return d


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass
class CriterionResult:
    name: str
    ok: bool
    measured: object
    threshold: object
    unit: str
    source: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": bool(self.ok),
                "measured": self.measured, "threshold": self.threshold,
                "unit": self.unit, "source": self.source, "detail": self.detail}

    def line(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return (f"{mark} {self.name:16s} measured={self.measured} "
                f"{self.unit} threshold={self.threshold} [{self.source}]"
                + (f" -- {self.detail}" if self.detail else ""))


@dataclass
class ExecReport:
    ok: bool
    criteria: list

    @property
    def failed(self) -> list:
        return [c.name for c in self.criteria if not c.ok]

    def as_dict(self) -> dict:
        return {"ok": bool(self.ok), "failed": self.failed,
                "criteria": [c.as_dict() for c in self.criteria]}

    def summary(self) -> str:
        verdict = "EXECUTABLE" if self.ok else "NOT EXECUTABLE"
        bad = ", ".join(self.failed) or "none"
        return f"{verdict} ({len(self.criteria)} criteria, failed: {bad})"

    def lines(self) -> list:
        return [c.line() for c in self.criteria]


# ---------------------------------------------------------------------------
# geometry helpers (2-D; same monotone chain as drill.kin, kept local so this
# module stays mujoco-free)
# ---------------------------------------------------------------------------
def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2-D cross product (numpy 2.x dropped the 2-D ``np.cross``)."""
    return float(a[0] * b[1] - a[1] * b[0])


def hull2d(points: np.ndarray) -> np.ndarray:
    p = np.unique(np.asarray(points, float).reshape(-1, 2), axis=0)
    if len(p) <= 2:
        return p
    order = np.lexsort((p[:, 1], p[:, 0]))
    p = p[order]

    def half(pts):
        out = []
        for q in pts:
            while len(out) >= 2 and _cross2(out[-1] - out[-2], q - out[-1]) <= 0:
                out.pop()
            out.append(q)
        return out

    lower, upper = half(p), half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def polygon_margin(point_xy: np.ndarray, poly: np.ndarray) -> float:
    if len(poly) == 0:
        return float("nan")
    if len(poly) == 1:
        return -float(np.linalg.norm(np.asarray(point_xy, float) - poly[0]))
    d = math.inf
    n = len(poly)
    inside = True
    p = np.asarray(point_xy, float)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        denom = float(e @ e)
        t = 0.0 if denom < 1e-18 else float(np.clip((p - a) @ e / denom, 0.0, 1.0))
        d = min(d, float(np.linalg.norm(p - (a + t * e))))
        if _cross2(e, p - a) < 0:
            inside = False
    return d if inside else -d


def _runs(mask: np.ndarray):
    """(start, end) inclusive runs of True."""
    idx = np.flatnonzero(np.asarray(mask, bool))
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def _sustained_from(mask: np.ndarray, start: int, need: int) -> int | None:
    """First index >= start where ``mask`` holds for ``need`` consecutive samples."""
    m = np.asarray(mask, bool)
    for k in range(int(start), len(m) - need + 1):
        if m[k:k + need].all():
            return k
    return None


def _yaw(qpos: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in qpos[3:7])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _heading_xy(displacement: np.ndarray, yaw: float) -> tuple:
    """World xy displacement -> (forward, left) in the heading frame."""
    c, s = math.cos(-yaw), math.sin(-yaw)
    d = np.asarray(displacement, float)
    return (float(c * d[0] - s * d[1]), float(s * d[0] + c * d[1]))


# ---------------------------------------------------------------------------
# trace accessor
# ---------------------------------------------------------------------------
_REQUIRED = ("t", "ref_time", "segment", "qpos", "qvel", "act_force",
             "contact_foot", "contact_knee", "contact_hand", "grounded",
             "sole_pts", "knee_z", "com", "pelvis_z", "tilt_deg", "pitch_deg",
             "margin", "jnt_range", "act_limit")


class _Trace:
    """Thin validated accessor over a trace mapping (dict or npz)."""

    def __init__(self, trace: Mapping):
        self._d = trace
        missing = [k for k in _REQUIRED if k not in trace]
        if missing:
            raise ValueError(f"trace is missing fields: {missing}")
        self.t = np.asarray(trace["t"], float)
        self.n = len(self.t)
        self.dt = float(np.median(np.diff(self.t))) if self.n > 1 else 0.02

    def __getitem__(self, key):
        return np.asarray(self._d[key])

    def get(self, key, default=None):
        return np.asarray(self._d[key]) if key in self._d else default

    def need(self, key):
        if key not in self._d:
            raise ValueError(f"trace is missing field {key!r} (required here)")
        return np.asarray(self._d[key])


def _spec_idx(tracks: "RefTracks", query: np.ndarray) -> np.ndarray:
    """Spec-track index of each query time (uniform grid, clipped)."""
    t0 = float(tracks.grid_t[0])
    dt = float(np.median(np.diff(tracks.grid_t))) if len(tracks.grid_t) > 1 else 0.02
    idx = np.rint((np.asarray(query, float) - t0) / dt).astype(np.int64)
    return np.clip(idx, 0, len(tracks.grid_t) - 1)


def _ref_mask(tr: _Trace, spec: ExecSpec) -> np.ndarray:
    seg = tr["segment"]
    rt = tr["ref_time"]
    return np.isin(seg, REFERENCE_SEGMENTS) & np.isfinite(rt)


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------
def _c_no_fall(tr: _Trace, spec: ExecSpec, tol: Tol) -> CriterionResult:
    pz = tr["pelvis_z"]
    tilt = tr["tilt_deg"]
    grounded = tr["grounded"].astype(bool)
    limb = (tr["contact_knee"].astype(bool)
            | tr["contact_hand"].astype(bool)).any(axis=1)
    pred = grounded | (~limb & ((pz <= FALL_PELVIS_HARD)
                                | ((pz <= FALL_PELVIS_GROUND)
                                   & (tilt >= FALL_TILT_GROUND_DEG))))
    need = max(1, int(round(FALL_CONFIRM_S / tr.dt)))
    trig = _sustained_from(pred, 0, need)
    t_trig = float(tr.t[trig]) if trig is not None else None
    ok = trig is None
    detail = (f"min pelvis z {float(pz.min()):.3f} m at t={float(tr.t[int(pz.argmin())]):.2f}s, "
              f"max tilt {float(tilt.max()):.1f} deg, grounded ticks "
              f"{int(grounded.sum())}")
    if trig is not None:
        detail = (f"fall confirmed at t={t_trig:.2f}s "
                  f"(pelvis {float(pz[trig]):.3f} m, tilt {float(tilt[trig]):.1f} deg, "
                  f"grounded={bool(grounded[trig])}); " + detail)
    return CriterionResult(
        "no_fall", ok, t_trig, None, "s (first termination)",
        "src/solo/fall.py FallDetector semantics (S1 contract)",
        detail)


def _c_contact_sequence(tr: _Trace, spec: ExecSpec, tol: Tol) -> CriterionResult:
    seg = tr["segment"]
    rt = tr["ref_time"]
    finite = np.isfinite(rt)
    ref = np.isin(seg, REFERENCE_SEGMENTS) & finite
    lead = tr["contact_foot"][:, spec.lead_ix].astype(bool)
    kz = tr["knee_z"][:, spec.knee_ix]
    pz = tr["pelvis_z"]

    # -- knee contact: first tick the penetration knee enters contact range
    # (entry-scoped: the penetration happens in the entry; the rise track is
    # the recovery, not a second shot)
    in_range = (seg == SEG_ENTRY) & (kz <= spec.knee_contact_z)
    k_idx = int(np.flatnonzero(in_range)[0]) if in_range.any() else None
    t_knee = float(rt[k_idx]) if k_idx is not None else None
    knee_ok = (t_knee is not None
               and (spec.knee_contact_t is None
                    or abs(t_knee - float(spec.knee_contact_t)) <= tol.knee_s))

    # -- lead-foot plant: rising edge after a genuine airborne phase
    need_air = max(1, int(round(0.10 / tr.dt)))
    plants = []
    for a, b in _runs(~lead):
        if (b - a + 1) >= need_air and b + 1 < tr.n and lead[b + 1]:
            plants.append(b + 1)
    plants = [p for p in plants if seg[p] in REFERENCE_SEGMENTS]
    if k_idx is not None:
        before = [p for p in plants if p <= k_idx]
        plants = before or plants
    p_idx = plants[-1] if plants else None
    t_plant = float(rt[p_idx]) if p_idx is not None and finite[p_idx] else None
    plant_ok = (t_plant is not None and spec.lead_plant_t is not None
                and abs(t_plant - float(spec.lead_plant_t)) <= tol.plant_s)
    if spec.lead_plant_t is None:
        plant_ok = p_idx is not None

    # -- rise: pelvis back up with the knee off the mat, held
    need_hold = max(1, int(round(tol.rise_hold_s / tr.dt)))
    after = (k_idx + 1) if k_idx is not None else 0
    up = (pz >= spec.rise_pelvis_z) & (kz >= spec.rise_knee_z)
    r_idx = _sustained_from(up, after, need_hold)
    t_rise = float(tr.t[r_idx]) if r_idx is not None else None
    rise_ok = False
    rise_dt = None
    if r_idx is not None and k_idx is not None:
        rise_dt = float(tr.t[r_idx] - tr.t[k_idx])
        rise_ok = 0.0 < rise_dt <= tol.rise_window_s
    order_ok = (p_idx is not None and k_idx is not None and p_idx <= k_idx)

    ok = bool(knee_ok and plant_ok and rise_ok and order_ok)
    measured = {"lead_plant_t": t_plant, "knee_contact_t": t_knee,
                "rise_dt_s": rise_dt}
    detail = (f"reference: lead {spec.lead_side} plant t={spec.lead_plant_t}s, "
              f"knee {spec.knee_side} contact t={spec.knee_contact_t}s "
              f"(z<={spec.knee_contact_z:.3f} m), rise within "
              f"{tol.rise_window_s}s")
    if not ok:
        why = []
        if not plant_ok:
            why.append("lead foot did not plant in window"
                       if t_plant is not None else
                       "lead foot never stepped (no airborne->plant edge)")
        if not knee_ok:
            why.append("knee contact missing or off-timing")
        if not rise_ok:
            why.append("no rise (pelvis/knee did not recover) in window")
        if not order_ok:
            why.append("plant/knee order wrong")
        detail += "; " + "; ".join(why)
    return CriterionResult(
        "contact_sequence", ok, measured, {"plant_s": tol.plant_s,
                                           "knee_s": tol.knee_s,
                                           "rise_window_s": tol.rise_window_s},
        "s (ref time) / s", "reference landmarks (measured) + S5 repair protocol",
        detail)


def _c_torso_pitch(tr: _Trace, spec: ExecSpec, tol: Tol) -> CriterionResult:
    tracks = spec.tracks
    ref = _ref_mask(tr, spec)
    k = np.flatnonzero(ref)
    j = _spec_idx(tracks, tr["ref_time"][k])
    assert np.isfinite(tracks.ref_time[j]).all(), "ref_time outside reference segments"
    keep = tracks.phase[j] != "PRONE"     # the deliberate finish is repaired away
    dev = np.abs(tr["pitch_deg"][k][keep] - tracks.pitch_deg[j][keep])
    worst = float(dev.max()) if dev.size else 0.0
    ok = worst <= tol.pitch_deg
    pitch_ref = tracks.pitch_deg[j][keep]
    detail = (f"compared on reference segments (PRONE excluded), {int(keep.sum())} ticks; "
              f"reference pitch range {float(pitch_ref.min()):.1f}..{float(pitch_ref.max()):.1f} deg")
    if dev.size and not ok:
        w = int(np.flatnonzero(keep)[int(dev.argmax())])
        detail += (f"; worst at t={float(tr.t[k[w]]):.2f}s (exec "
                   f"{float(tr['pitch_deg'][k[w]]):.1f} vs ref {float(pitch_ref[w]):.1f} deg)")
    return CriterionResult(
        "torso_pitch", ok, round(worst, 2), tol.pitch_deg, "deg (max deviation)",
        "retarget fit (a_max 0.149 m over the ~0.5 m torso lever ~= 16.6 deg)",
        detail)


def _c_com_margin(tr: _Trace, spec: ExecSpec, tol: Tol) -> CriterionResult:
    tracks = spec.tracks
    seg = tr["segment"]
    margin = tr["margin"]
    ref = _ref_mask(tr, spec)
    k = np.flatnonzero(ref)
    j = _spec_idx(tracks, tr["ref_time"][k])
    ref_margin = tracks.margin[j]
    exec_margin = margin[k]

    ref_air = ~np.isfinite(ref_margin)
    exec_air = ~np.isfinite(exec_margin)
    below = np.zeros(tr.n, bool)
    air = np.zeros(tr.n, bool)
    below[k] = (~exec_air) & (exec_margin < np.where(ref_margin >= 0.0, 0.0,
                                                    ref_margin - tol.margin_m))
    air[k] = exec_air & ~ref_air
    floor = np.full(tr.n, np.nan)
    floor[k] = np.where(ref_margin >= 0.0, 0.0, ref_margin - tol.margin_m)
    # the blends must stay near the envelope; the final hold must be inside it
    for s in (SEG_BLEND, SEG_STAND):
        m = seg == s
        below |= m & np.isfinite(margin) & (margin < -tol.margin_m)
        air |= m & ~np.isfinite(margin)
        floor[m] = -tol.margin_m
    mh = seg == SEG_HOLD
    below |= mh & np.isfinite(margin) & (margin < 0.0)
    air |= mh & ~np.isfinite(margin)
    floor[mh] = 0.0
    depth = float(np.max(floor[below] - margin[below])) if below.any() else 0.0
    n_viol = int(below.sum()) + int(air.sum())
    ok = n_viol == 0
    allowed = sorted({str(p) for p, m in zip(tracks.phase[j], ref_margin) if m < 0.0})
    finite_margin = margin[np.isfinite(margin)]
    detail = (f"floors: 0.0 m where the reference is inside its own envelope, "
              f"ref_margin-{tol.margin_m} m where the reference itself is out "
              f"(phases: {', '.join(allowed) or 'none'}); blend floor "
              f"-{tol.margin_m} m, hold floor 0.0 m; {int(below.sum())} below-floor "
              f"ticks, {int(air.sum())} airborne-while-reference-grounded")
    if below.any():
        w = int(np.flatnonzero(below)[int(np.argmin(np.where(np.isfinite(margin),
                                                             margin, np.inf)[below]))])
        detail += f"; worst at t={float(tr.t[w]):.2f}s (margin {float(margin[w]):.3f} m)"
    return CriterionResult(
        "com_margin", ok,
        round(float(finite_margin.min()), 4) if finite_margin.size else None,
        {"n_violations": n_viol, "worst_depth_m": round(depth, 4)},
        "m (min margin)",
        "data/support_envelope.json envelope + delivered L2 transients (0.026 m)",
        detail)


def _c_knee_depth(tr: _Trace, spec: ExecSpec, tol: Tol) -> CriterionResult:
    ref = _ref_mask(tr, spec) & (tr["segment"] == SEG_ENTRY)
    kz = tr["knee_z"][ref, spec.knee_ix]
    lowest = float(kz.min()) if kz.size else float("inf")
    ok = lowest <= spec.knee_min_z + tol.knee_depth_m
    detail = (f"reference min knee height {spec.knee_min_z:.3f} m at "
              f"t={spec.knee_min_t:.2f}s (track time)")
    return CriterionResult(
        "knee_depth", ok, round(lowest, 4), round(spec.knee_min_z + tol.knee_depth_m, 4),
        "m (min knee-site height)",
        "reference landmark (measured) + 0.03 m tolerance", detail)


def _c_no_saturation(tr: _Trace, spec: ExecSpec, tol: Tol) -> CriterionResult:
    q = tr["qpos"][:, 7:36]
    lo = tr["jnt_range"][:, 0]
    hi = tr["jnt_range"][:, 1]
    pos_margin = float(np.minimum(q - lo, hi - q).min())
    vel = float(np.abs(tr["qvel"][:, 6:35]).max())
    force = np.abs(tr["act_force"][:, :29])
    lim = np.abs(tr["act_limit"])
    frac = float((force / np.maximum(lim, 1e-9)).max())
    n_at = int((force >= SAT_FRACTION * lim).sum())
    ok = (pos_margin >= 0.0) and (vel <= tol.vel_rad_s) and (frac <= tol.sat_frac)
    detail = (f"min joint-limit margin {pos_margin:.4f} rad; "
              f"max |qvel| {vel:.2f} rad/s (limit {tol.vel_rad_s}, "
              f"retarget.solve.V_MAX); max force fraction {frac:.3f} "
              f"({n_at} actuator-ticks >= {SAT_FRACTION:.2f})")
    return CriterionResult(
        "no_saturation", ok,
        {"pos_margin_rad": round(pos_margin, 4), "max_qvel": round(vel, 3),
         "max_force_frac": round(frac, 4)},
        {"pos_margin_rad": 0.0, "max_qvel": tol.vel_rad_s,
         "max_force_frac": tol.sat_frac},
        "rad / rad/s / fraction",
        "jnt_range + jnt_actfrcrange (model); retarget.solve.V_MAX for velocity",
        detail)


def _c_no_foot_slide(tr: _Trace, spec: ExecSpec, tol: Tol) -> CriterionResult:
    sole = tr["sole_pts"]                     # (N, 2, 4, 3)
    com = tr["com"]
    worst = 0.0
    worst_side, worst_t = None, None
    for i, side in enumerate(("left", "right")):
        anchor = None
        for k in range(tr.n):
            pts = sole[k, i]
            on_mat = bool(pts[:, 2].min() < LOAD_TOUCH_Z)
            hull = hull2d(pts[:, :2])
            over = polygon_margin(com[k, :2], hull) > -0.01 if len(hull) >= 3 else False
            loaded = on_mat and over
            if not loaded:
                anchor = None
                continue
            centre = pts[:, :2].mean(axis=0)
            if anchor is None:
                anchor = centre
            drift = float(np.linalg.norm(centre - anchor))
            if drift > worst:
                worst, worst_side, worst_t = drift, side, float(tr.t[k])
    ok = worst <= tol.slide_m
    detail = (f"loaded = sole centre < {LOAD_TOUCH_Z} m and CoM over the "
              f"footprint (drill.metrics semantics)")
    if worst_side is not None:
        detail += f"; worst drift on {worst_side} foot at t={worst_t:.2f}s"
    return CriterionResult(
        "no_foot_slide", ok, round(worst, 4), tol.slide_m, "m (loaded drift)",
        "drill rubric B1 (bar 20 mm; measured per-step drift 2.6-5.6 mm)", detail)


def _c_travel(tr: _Trace, spec: ExecSpec, tol: Tol) -> CriterionResult:
    seg = tr["segment"]
    ref = _ref_mask(tr, spec) & (seg == SEG_ENTRY)
    idx = np.flatnonzero(ref)
    if idx.size < 2:
        return CriterionResult(
            "travel", False, None, None, "m",
            "data/refs_video/retarget_summary.json net_travel_m",
            "trace has no entry segment to measure travel on")
    k0, k1 = int(idx[0]), int(idx[-1])
    disp = tr["qpos"][k1, :2] - tr["qpos"][k0, :2]
    fwd, lat = _heading_xy(disp, _yaw(tr["qpos"][k0]))
    ok = (fwd >= tol.travel_frac * spec.travel_forward_m
          and abs(lat - spec.travel_lateral_m) <= tol.travel_slack_m
          and fwd <= spec.travel_forward_m + tol.travel_max_extra_m)
    detail = (f"reference entry travel (its own heading frame): forward "
              f"{spec.travel_forward_m:.3f} m, lateral {spec.travel_lateral_m:.3f} m "
              f"(net {spec.travel_net_m:.3f} m; foot-contact anchored)")
    return CriterionResult(
        "travel", ok,
        {"forward_m": round(fwd, 4), "lateral_m": round(lat, 4)},
        {"forward_m": round(tol.travel_frac * spec.travel_forward_m, 4),
         "lateral_slack_m": tol.travel_slack_m},
        "m", "retarget meta net_travel_m (foot-contact anchoring)", detail)


def _c_terminal_stance(tr: _Trace, spec: ExecSpec, tol: Tol) -> CriterionResult:
    """The demo must end in a recoverable standing stance (operator rule).

    Thresholds (all measured, see ``solo.demo.measure_spec``):
    * pelvis height inside the documented stance-height command range
      (``solo.commands.CommandRanges.stance_height`` = 0.70-0.80 m) and within
      ``stance_pelvis_tol`` of the stance keyframe's own height;
    * both feet flat and in contact: all four sole-sphere centres within
      ``stance_flat_tol_m`` of the measured rest height (``sole_rest_z``,
      ``drill.kin`` semantics) and the lowest below the contact rule 0.045 m;
    * CoM margin >= ``stance_margin_min`` (the audit's STABLY_FEASIBLE band,
      0.02 m) -- i.e. inside the support hull;
    * torso tilt <= ``stance_tilt_max_deg`` (measured stance tilt + margin);
    * base speed <= ``stance_speed_max`` (the drill's settle/lift velocity
      gates, 0.02-0.04 m/s).
    """
    n_win = max(1, int(round(spec.term_window_s / tr.dt)))
    sl = slice(max(0, tr.n - n_win), tr.n)
    margin = tr["margin"][sl]
    pelvis = tr["pelvis_z"][sl]
    tilt = tr["tilt_deg"][sl]
    qv = tr["qvel"][sl]
    speed = np.linalg.norm(np.asarray(qv[:, 0:2], float), axis=1)
    sole = tr["sole_pts"][sl]                       # (w, 2, 4, 3)
    contact = sole[:, :, :, 2].min(axis=2) < TOUCH_Z          # (w, 2)
    flat = (sole[:, :, :, 2].max(axis=2) - spec.sole_rest_z) <= spec.stance_flat_tol_m

    finite_margin = margin[np.isfinite(margin)]
    m_min = float(finite_margin.min()) if finite_margin.size else float("-inf")
    ok_margin = bool(finite_margin.size == margin.size and
                     (finite_margin >= spec.stance_margin_min).all())
    ok_feet = bool((contact & flat).all())
    ok_pelvis = bool(((pelvis >= spec.stance_pelvis_lo)
                      & (pelvis <= spec.stance_pelvis_hi)).all())
    ok_tilt = bool((tilt <= spec.stance_tilt_max_deg).all())
    ok_speed = bool((speed <= spec.stance_speed_max).all())
    ok = bool(ok_margin and ok_feet and ok_pelvis and ok_tilt and ok_speed)
    measured = {"margin_min_m": None if m_min == float("-inf") else round(m_min, 4),
                "pelvis_z": round(float(pelvis[-1]), 4),
                "tilt_deg": round(float(tilt[-1]), 2),
                "base_speed": round(float(speed[-1]), 4),
                "feet_flat_both": bool((contact & flat).all())}
    detail = (f"last {spec.term_window_s:.1f}s ({n_win} ticks); thresholds: "
              f"pelvis in [{spec.stance_pelvis_lo:.2f}, {spec.stance_pelvis_hi:.2f}] m "
              f"(keyframe {spec.stance_pelvis_z:.3f} m), margin >= "
              f"{spec.stance_margin_min:.2f} m, tilt <= {spec.stance_tilt_max_deg:.1f} deg, "
              f"base speed <= {spec.stance_speed_max:.2f} m/s, soles flat within "
              f"{spec.stance_flat_tol_m * 1000:.0f} mm of rest "
              f"({spec.sole_rest_z * 1000:.1f} mm)")
    bad = [n for n, o in (("margin", ok_margin), ("feet", ok_feet),
                          ("pelvis", ok_pelvis), ("tilt", ok_tilt),
                          ("speed", ok_speed)) if not o]
    if bad:
        detail += "; failing: " + ", ".join(bad)
    return CriterionResult(
        "terminal_stance", ok, measured,
        {"pelvis_m": [spec.stance_pelvis_lo, spec.stance_pelvis_hi],
         "margin_m": spec.stance_margin_min,
         "tilt_deg": spec.stance_tilt_max_deg,
         "speed_mps": spec.stance_speed_max},
        "mixed", "measured stance (L1 stance spec / stand_hold) -- see detail",
        detail)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def check_executability(trace: Mapping, spec: ExecSpec,
                        tol: Tol | None = None) -> ExecReport:
    """Run every criterion on one trace; returns an :class:`ExecReport`.

    ``trace`` is the per-tick record written by ``solo.demo`` (or a synthetic
    mapping with the same fields); ``spec`` carries the reference-derived
    thresholds and tracks (``solo.demo.measure_spec``).
    """
    if spec.tracks is None:
        raise ValueError("spec has no reference tracks; measure them first "
                         "(solo.demo.measure_spec)")
    tr = _Trace(trace)
    t = tol or spec.tol
    criteria = [
        _c_no_fall(tr, spec, t),
        _c_contact_sequence(tr, spec, t),
        _c_torso_pitch(tr, spec, t),
        _c_com_margin(tr, spec, t),
        _c_knee_depth(tr, spec, t),
        _c_no_saturation(tr, spec, t),
        _c_no_foot_slide(tr, spec, t),
        _c_travel(tr, spec, t),
        _c_terminal_stance(tr, spec, t),
    ]
    return ExecReport(ok=all(c.ok for c in criteria), criteria=criteria)


if __name__ == "__main__":  # self-check (no sim, no model)
    n = 400
    t = np.arange(n) * 0.02
    ref_time = t.copy()
    ref_time[300:] = np.nan
    tracks = RefTracks(
        grid_t=t.copy(), ref_time=ref_time, segment=np.zeros(n, np.int8),
        phase=np.full(n, "LOW"), pitch_deg=np.full(n, 45.0),
        margin=np.full(n, 0.02), contact=np.ones((n, 2), bool),
        knee_z=np.tile(np.array([0.30, 0.40]), (n, 1)),
        pelvis_z=np.full(n, 0.60), base_xy=np.zeros((n, 2)),
        knee_site_xy=np.zeros((n, 2, 2)))
    spec = ExecSpec(knee_side="left", lead_side="right",
                    lead_plant_t=None, knee_min_z=0.30, knee_min_t=0.0,
                    knee_contact_z=0.35, knee_contact_t=None,
                    travel_forward_m=0.0, rise_pelvis_z=0.55, rise_knee_z=0.20,
                    tracks=tracks)
    trace = {
        "t": t, "ref_time": ref_time, "segment": np.zeros(n, np.int8),
        "qpos": np.zeros((n, 36)), "qvel": np.zeros((n, 35)),
        "act_force": np.zeros((n, 29)), "jnt_range": np.tile([-1.0, 1.0], (29, 1)),
        "act_limit": np.ones(29), "contact_foot": np.ones((n, 2), bool),
        "contact_knee": np.zeros((n, 2), bool), "contact_hand": np.zeros((n, 2), bool),
        "grounded": np.zeros(n, bool), "sole_pts": np.zeros((n, 2, 4, 3)),
        "knee_z": np.tile(np.array([0.30, 0.40]), (n, 1)), "com": np.zeros((n, 3)),
        "pelvis_z": np.full(n, 0.60), "tilt_deg": np.full(n, 45.0),
        "pitch_deg": np.full(n, 45.0), "margin": np.full(n, 0.02),
    }
    rep = check_executability(trace, spec)
    print("exec_check self-check:", rep.summary())
    for line in rep.lines():
        print("   ", line)
