"""Phase-3 stabilized teacher: reference tracking + balance feedback.

Public surface:

* ``TeacherController`` — the stabilized teacher (reference + trim + balance
  feedback + leg IK), ``BaselinePD`` — the phase-2 pure-tracking baseline;
* ``run_episode`` / ``EpisodeResult`` — rollout + the task metrics
  (stay-up fraction, penetrations, landmark distance);
* ``score_episode`` — technique-validity score of a rollout (lazy import of
  the separate scorer package);
* ``trims`` — the measured per-technique reference-pose adjustments;
* ``phases`` — reference phase segmentation for gain scheduling.
"""

from .controller import ALL_OFF, BaselinePD, TeacherController, TeacherFlags
from .episode import (EpisodeResult, final_landmark_dist, reference_track,
                      run_episode, score_episode, scored_trajectories,
                      sprawl_end_posture)
from .gains import GAIN_TABLE, TECHNIQUE_OVERRIDES, GainSet, gain_for
from .phases import KINDS, label_frames, phase_table, segments
from .trims import POSE_TRIM, load_trims, trim_for, trim_scale, trim_vector

__all__ = [
    "ALL_OFF", "BaselinePD", "TeacherController", "TeacherFlags",
    "EpisodeResult", "final_landmark_dist", "reference_track", "run_episode",
    "score_episode", "scored_trajectories", "sprawl_end_posture",
    "GAIN_TABLE", "TECHNIQUE_OVERRIDES", "GainSet", "gain_for",
    "KINDS", "label_frames", "phase_table", "segments",
    "POSE_TRIM", "load_trims", "trim_for", "trim_scale", "trim_vector",
]
