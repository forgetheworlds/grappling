"""Technique-validity scorer configuration: phase bands + fuzzy predicates.

CALIBRATION PROVENANCE (reports/2026-10-08/scorer.md has the full table):
- Phase bands are keyframe-index bands of the phase-2 references
  (data/refs/<TECH>.npz). Keyframe times were reconstructed as
  ``build_technique_targets(tech).times * time_stretch_requested *
  time_stretch_kinematic`` (both stretches from the npz meta) and mapped to
  the 50 Hz frame grid; phase boundaries sit on the reconstructed keyframe
  times / observed geometry transitions (SPRAWL, STAND_UP are montages whose
  per-edge boundaries are blurred by alignment transitions, so their bands
  were placed on the feature trajectories themselves: defender hip height,
  leg-back, executor knee height).
- Thresholds: ``core`` covers ~the [p10, p90] range of each feature on its
  own reference phase band; ``soft`` extends by ~max(1.5x core width,
  floor) with floor = 0.10 m (distances) / 0.15 rad (angles), widened where
  the reference itself is transitional. Reference frames therefore score
  ~1.0 on their own technique while genuinely different relationships
  (double vs single leg lock asymmetry, sprawl vs stance leg-back, ...) fall
  outside the soft region.
- Weights: 2.0 = technique-defining relationship (MISSION "Preserve
  wrestling geometry"), 1.0 = supporting context.

The executor field records which reference robot demonstrates the technique
(A = attacker/recoverer, B = defender); data/refs roles per notes.md.
"""

from __future__ import annotations

# predicate tuple: (feature, kind, params, weight)
#   kind "band"     -> params (lo, hi, lo_soft, hi_soft)
#   kind "at_most"  -> params (core, soft)
#   kind "at_least" -> params (core, soft)
#   kind "ang_near" -> params (center, core, soft)  [radians, circular]

TECHNIQUE_CONFIG: dict[str, dict] = {
    "DOUBLE_LEG": {
        "executor": "A",
        # keyframes 8; edges t1144/1147/1141/1140 = setup, penetrate, drive, land
        "phases": [("SETUP", 0.0, 0.46), ("PENETRATE", 0.46, 1.34),
                   ("DRIVE", 1.34, 2.22), ("LAND", 2.22, 3.10)],
        "predicates": {
            0: [  # standing square, level starts to drop
                ("sep", "band", (0.86, 0.96, 0.60, 1.30), 1.0),
                ("bearing", "band", (0.28, 0.40, -0.45, 1.10), 1.0),
                ("knee_z_s", "band", (0.24, 0.32, 0.12, 0.45), 1.0),
                ("pitch_s", "band", (0.45, 0.85, 0.10, 1.30), 1.0),
                ("base_w_s", "band", (0.52, 0.63, 0.35, 0.90), 1.0),
                ("head_rel", "band", (0.10, 0.36, -0.10, 0.60), 1.0),
            ],
            1: [  # penetration step: depth taken, head to chest, knee UP
                ("sep", "band", (0.40, 0.85, 0.25, 1.10), 2.0),
                ("dx", "band", (0.33, 0.80, 0.15, 1.05), 2.0),
                ("head_rel", "at_most", (0.12, 0.35), 2.0),
                ("pitch_s", "band", (0.88, 1.22, 0.55, 1.55), 1.0),
                ("knee_z_s", "at_least", (0.24, 0.10), 1.0),
                ("lock_l", "at_most", (0.40, 0.75), 1.0),
                ("lock_r", "at_most", (0.60, 0.90), 1.0),
            ],
            2: [  # BOTH legs locked behind the thigh/knee, head up-chest
                ("lock_l", "at_most", (0.10, 0.35), 2.0),
                ("lock_r", "at_most", (0.08, 0.30), 2.0),
                ("lock_asym", "at_most", (0.03, 0.28), 2.0),
                ("head_rel", "band", (0.02, 0.12, -0.08, 0.32), 1.0),
                ("sep", "band", (0.44, 0.52, 0.30, 0.80), 1.0),
                ("pitch_s", "band", (0.98, 1.20, 0.65, 1.55), 1.0),
                ("leg_back_s", "band", (-0.47, -0.15, -0.70, 0.10), 1.0),
            ],
            3: [  # finish: opponent's feet come up, drive through
                ("ankle_z_o", "at_least", (0.36, 0.05), 2.0),
                ("lock_asym", "at_most", (0.20, 0.45), 1.0),
                ("lock_l", "at_most", (0.14, 0.40), 1.0),
                ("lock_r", "at_most", (0.30, 0.55), 1.0),
                ("knee_z_s", "at_most", (0.28, 0.45), 1.0),
                ("sep", "band", (0.40, 0.49, 0.25, 0.75), 1.0),
                ("dx", "band", (0.08, 0.47, -0.15, 0.70), 1.0),
            ],
        },
    },

    "SINGLE_LEG": {
        "executor": "A",
        # keyframes 13; edges t978/1001/1094 = setup, capture, finish (2 bands)
        "phases": [("SETUP", 0.0, 1.10), ("CAPTURE", 1.10, 1.90),
                   ("FINISH1", 1.90, 3.52), ("FINISH2", 3.52, 4.60)],
        "predicates": {
            0: [  # wide south-paw stagger, angled entry
                ("base_w_s", "at_least", (0.70, 0.35), 2.0),
                ("bearing", "band", (0.42, 0.70, -0.25, 1.35), 1.0),
                ("sep", "band", (0.60, 1.00, 0.40, 1.30), 1.0),
                ("knee_z_s", "band", (0.24, 0.33, 0.12, 0.45), 1.0),
                ("rel_yaw", "ang_near", (-1.85, 0.35, 1.00), 1.0),
            ],
            1: [  # ONE leg captured, head on chest, other leg free
                ("lock_r", "at_most", (0.10, 0.32), 2.0),
                ("lock_l", "at_least", (0.24, 0.06), 2.0),
                ("lock_asym", "at_least", (0.17, 0.02), 2.0),
                ("head_rel", "band", (0.00, 0.11, -0.15, 0.30), 2.0),
                ("pitch_s", "band", (0.98, 1.08, 0.70, 1.35), 1.0),
                ("knee_z_s", "at_least", (0.25, 0.10), 1.0),
                ("sep", "band", (0.52, 0.59, 0.35, 0.85), 1.0),
            ],
            2: [  # standing finish, leg still on one side
                ("lock_r", "at_most", (0.12, 0.35), 2.0),
                ("lock_asym", "at_least", (0.18, 0.03), 2.0),
                ("head_chest", "at_most", (0.19, 0.45), 2.0),
                ("pitch_s", "band", (1.05, 1.35, 0.70, 1.70), 1.0),
                ("sep", "band", (0.49, 0.56, 0.32, 0.82), 1.0),
                ("knee_z_s", "band", (0.03, 0.38, -0.05, 0.55), 1.0),
            ],
            3: [  # down on the knee, angle off to the side
                ("knee_z_s", "at_most", (0.05, 0.22), 2.0),
                ("lock_r", "at_most", (0.11, 0.32), 2.0),
                ("lock_l", "at_least", (0.24, 0.06), 2.0),
                ("pitch_s", "at_least", (1.45, 0.90), 1.0),
                ("roll_s", "band", (1.50, 1.92, 0.90, 2.40), 1.0),
                ("dz", "at_most", (-0.08, 0.15), 1.0),
                ("bearing", "band", (1.05, 1.55, 0.40, 2.10), 1.0),
            ],
        },
    },

    "BODY_LOCK": {
        "executor": "A",
        # single edge t1133 (10 kf): clinch, drop-over, pin
        "phases": [("CLINCH", 0.0, 1.58), ("DROPOVER", 1.58, 2.74),
                   ("PIN", 2.74, 3.52)],
        "predicates": {
            0: [  # chest-to-chest lock, arms around, both standing
                ("sep", "at_most", (0.37, 0.62), 2.0),
                ("wrap", "at_most", (0.25, 0.50), 2.0),
                ("chest_on", "at_most", (0.30, 0.55), 1.0),
                ("lock_asym", "at_most", (0.10, 0.32), 1.0),
                ("pitch_s", "band", (0.40, 0.60, 0.10, 0.95), 1.0),
                ("knee_z_s", "at_least", (0.20, 0.05), 1.0),
            ],
            1: [  # sweeping drop: lock kept, opponent's balance broken
                ("wrap", "at_most", (0.30, 0.55), 2.0),
                ("lock_r", "at_most", (0.09, 0.35), 1.0),
                ("lock_l", "at_most", (0.18, 0.45), 1.0),
                ("sep", "band", (0.37, 0.58, 0.20, 0.85), 1.0),
                ("pitch_s", "band", (0.50, 1.17, 0.20, 1.50), 1.0),
                ("dz", "band", (-0.16, 0.24, -0.40, 0.50), 1.0),
                ("ankle_z_o", "band", (0.08, 0.26, -0.05, 0.45), 1.0),
            ],
            2: [  # opponent driven down at the hip, feet off the mat
                ("dz", "at_most", (-0.20, 0.10), 2.0),
                ("ankle_z_o", "at_least", (0.21, 0.02), 2.0),
                ("roll_s", "at_least", (0.50, 0.05), 1.0),
                ("pitch_s", "at_least", (1.05, 0.60), 1.0),
                ("sep", "band", (0.59, 0.69, 0.40, 0.95), 1.0),
                ("knee_z_s", "band", (0.25, 0.34, 0.10, 0.50), 1.0),
                ("wrap", "at_most", (0.29, 0.55), 1.0),
            ],
        },
    },

    "SNAPDOWN": {
        "executor": "A",
        # single edge t989 (3 kf): collar tie, snap
        "phases": [("SETUP", 0.0, 0.46), ("SNAP", 0.46, 0.92)],
        "predicates": {
            0: [  # collar+tricep tie, opponent upright
                ("head_snap", "at_most", (0.11, 0.35), 2.0),
                ("sep", "band", (0.72, 0.78, 0.50, 1.05), 1.0),
                ("pitch_o", "band", (0.79, 0.90, 0.45, 1.25), 1.0),
                ("knee_z_s", "band", (0.16, 0.33, 0.05, 0.45), 1.0),
                ("head_rel", "band", (0.20, 0.28, 0.00, 0.50), 1.0),
                ("wrap", "at_most", (0.24, 0.50), 1.0),
            ],
            1: [  # head yanked down: opponent pitches forward, feet planted
                ("head_snap", "at_most", (0.09, 0.30), 2.0),
                ("pitch_o", "at_least", (1.00, 0.65), 2.0),
                ("pitch_s", "band", (0.52, 0.92, 0.20, 1.25), 1.0),
                ("sep", "band", (0.75, 0.80, 0.55, 1.05), 1.0),
                ("ankle_z_o", "at_most", (0.10, 0.28), 1.0),
                ("knee_z_s", "at_most", (0.27, 0.42), 1.0),
            ],
        },
    },

    "SPRAWL": {
        "executor": "B",
        # montage of 3 defender sprawl cycles (t381/382/383 spliced onto the
        # attacker's shot); bands placed on the defender hip/leg trajectories
        "phases": [("STAND1", 0.0, 1.80), ("SPRAWL1", 1.80, 2.64),
                   ("RECOVER1", 2.64, 3.36), ("SPRAWL2", 3.36, 4.32),
                   ("RECOVER2", 4.32, 5.28), ("SPRAWL3", 5.28, 5.76)],
        "predicates": {
            0: [  # still standing, feet under hips, chest closing
                ("knee_z_s", "at_least", (0.32, 0.12), 2.0),
                ("leg_back_s", "band", (-0.25, 0.10, -0.50, 0.35), 2.0),
                ("pitch_s", "at_most", (1.20, 1.65), 1.0),
                ("chest_on", "band", (0.22, 0.45, 0.05, 0.70), 1.0),
                ("ankle_z_s", "band", (0.08, 0.27, -0.02, 0.40), 1.0),
            ],
            2: [  # sprawl geometry: legs shot back, hips down, chest through
                ("leg_back_s", "at_most", (-0.24, 0.05), 2.0),
                ("knee_z_s", "at_most", (0.24, 0.42), 2.0),
                ("pitch_s", "at_least", (1.10, 0.65), 2.0),
                ("head_chest", "at_most", (0.31, 0.58), 1.0),
                ("chest_on", "at_most", (0.35, 0.60), 1.0),
            ],
        },
    },
    "STAND_UP": {
        "executor": "A",
        # montage t952/t1110/t1125; bands placed on executor knee trajectory
        "phases": [("DOWN", 0.0, 2.00), ("STAND1", 2.00, 3.96),
                   ("STAND2", 3.96, 6.94), ("CLINCH", 6.94, 9.38)],
        "predicates": {
            0: [  # from knees/hands: knees low, feet pulled forward
                ("knee_z_s", "at_most", (0.30, 0.48), 2.0),
                ("ankle_z_s", "at_most", (0.24, 0.42), 2.0),
                ("leg_back_s", "at_least", (-0.10, -0.45), 1.0),
                ("base_w_s", "at_most", (0.70, 1.00), 1.0),
                ("chest_on", "band", (0.36, 0.72, 0.15, 0.95), 1.0),
                ("head_chest", "at_most", (0.78, 1.05), 1.0),
            ],
            1: [  # first stand-up: rising through knee-high
                ("knee_z_s", "band", (0.27, 0.43, 0.10, 0.60), 2.0),
                ("ankle_z_s", "band", (0.04, 0.35, -0.05, 0.55), 1.0),
                ("pitch_s", "band", (0.20, 0.80, -0.20, 1.20), 1.0),
                ("head_chest", "at_most", (0.80, 1.10), 1.0),
                ("sep", "band", (0.35, 1.00, 0.15, 1.30), 1.0),
            ],
            2: [  # technical stand-up cycle: mid knees, feet narrow->rise
                ("knee_z_s", "band", (0.43, 0.75, 0.25, 0.95), 2.0),
                ("ankle_z_s", "band", (0.22, 0.82, 0.05, 1.05), 1.0),
                ("leg_back_s", "band", (-0.30, 0.35, -0.55, 0.60), 1.0),
                ("chest_on", "band", (0.20, 0.72, 0.05, 0.95), 1.0),
                ("head_chest", "at_most", (0.80, 1.05), 1.0),
            ],
            3: [  # dogfight clinch: fully up, deep underhook, glued
                ("knee_z_s", "at_least", (0.72, 0.45), 2.0),
                ("ankle_z_s", "at_least", (0.60, 0.35), 2.0),
                ("wrap", "at_most", (0.13, 0.38), 2.0),
                ("chest_on", "at_most", (0.27, 0.55), 2.0),
                ("head_chest", "at_most", (0.36, 0.62), 1.0),
                ("sep", "band", (0.36, 0.45, 0.20, 0.70), 1.0),
            ],
        },
    },

    "STANCE": {
        "executor": "A",
        # procedural hold (node 94 blend), single phase
        "phases": [("HOLD", 0.0, 2.34)],
        "predicates": {
            0: [
                ("sep", "band", (0.97, 0.99, 0.70, 1.30), 2.0),
                ("rel_yaw", "ang_near", (3.14159, 0.35, 1.00), 2.0),
                ("knee_z_s", "band", (0.35, 0.37, 0.22, 0.50), 2.0),
                ("pitch_s", "band", (0.27, 0.31, 0.05, 0.60), 1.0),
                ("base_w_s", "band", (0.28, 0.32, 0.15, 0.55), 1.0),
                ("lock_l", "at_least", (0.55, 0.25), 1.0),
                ("ankle_z_s", "band", (0.09, 0.11, 0.00, 0.25), 1.0),
                ("head_rel", "band", (0.37, 0.41, 0.15, 0.65), 1.0),
            ],
        },
    },
}

#: SPRAWL cycle phases share predicates: sprawl cycles {2, 4} reuse phase 2's
#: predicate list; recover phases {3, 5} reuse a common recover list.
SPRAWL_RECOVER_PREDICATES = [
    ("knee_z_s", "at_least", (0.21, 0.02), 2.0),
    ("leg_back_s", "at_least", (-0.22, -0.50), 2.0),
    ("pitch_s", "band", (0.72, 1.12, 0.40, 1.50), 1.0),
    ("head_chest", "band", (0.30, 0.42, 0.10, 0.65), 1.0),
]


def predicates_for(technique: str, phase: int) -> list[tuple]:
    """Resolved predicate list for a technique phase (handles SPRAWL shares)."""
    cfg = TECHNIQUE_CONFIG[technique]
    if technique == "SPRAWL":
        if phase in (2, 4):
            return cfg["predicates"][2]
        if phase in (3, 5):
            return SPRAWL_RECOVER_PREDICATES
        return cfg["predicates"][0]
    return cfg["predicates"][phase]


TECHNIQUES: tuple[str, ...] = tuple(TECHNIQUE_CONFIG)
