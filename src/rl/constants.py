"""G1 action-layout constants shared by the RL modules.

Extracted (2026-10-08, values unchanged) from ``wrestling.env`` when the
two-robot pipeline was removed: :mod:`rl.net` is on the solo-drill mandate path
and needed exactly these two names.  Nothing else of the environment came with
them.

``ACT_SLICE`` describes the *model-order* action layout: one robot owns the
G1's 29 position actuators, and the two-robot layout concatenated robot ``a``
then robot ``b`` into one 58-dim action.  The mandate uses a single robot
(``"a"``); the ``"b"`` entry is kept because :func:`rl.net.robot_action_bounds`
and :func:`rl.net.compose_action` are generic over the robot name.
"""

from __future__ import annotations

#: actuated joints of the G1 (= one robot's action dimension)
N_JOINTS = 29
#: model-order action slices (``"b"`` = the second half of a 58-dim pair action)
ACT_SLICE = {"a": slice(0, 29), "b": slice(29, 58)}

__all__ = ["ACT_SLICE", "N_JOINTS"]
