"""Phase-3 stabilized teacher: reference tracking + balance feedback."""

from .controller import ALL_OFF, BaselinePD, TeacherController, TeacherFlags
from .episode import (EpisodeResult, final_landmark_dist, run_episode,
                      sprawl_end_posture)
from .gains import GAIN_TABLE, GainSet, gain_for
from .phases import KINDS, label_frames, phase_table, segments

__all__ = [
    "ALL_OFF", "BaselinePD", "TeacherController", "TeacherFlags",
    "EpisodeResult", "final_landmark_dist", "run_episode",
    "sprawl_end_posture", "GAIN_TABLE", "GainSet", "gain_for",
    "KINDS", "label_frames", "phase_table", "segments",
]
