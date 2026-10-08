"""Solo single-G1 drill environment (S1 of ``docs/SOLO_DRILL.md``).

Modules
-------
* ``scene``    -- the single-G1 scene (``a_``-prefixed) + virtual-opponent markers
* ``commands`` -- (vx, vy, wz, stance_height, stance_width, skill_id, lead_leg) + sampler/filter
* ``pushes``   -- scheduled impulses/forces on the torso (``qfrc_applied``)
* ``fall``     -- fall detector + single-robot dorsal (back-to-mat) semantics
* ``markers``  -- virtual-opponent marker targets as a function of command/phase
* ``obs``      -- fixed actor/privileged observation layouts
* ``reward``   -- per-task reward term sets (weights are placeholders)
* ``metrics``  -- per-step + aggregated metrics, JSONL emission
* ``env``      -- ``SoloEnv``: 50 Hz control / 500 Hz physics, termination, info
* ``eval``     -- the evaluation harness that gates training (task gates + verdicts)
* ``baselines``-- scripted baselines and exploit probes
* ``video``    -- 960x720/30fps evidence rendering + 3-frame contact sheets
* ``lock``     -- advisory cross-process sim lock (``data/locks/sim.lock``)
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:  # support `import solo.*` and `python -m src.solo.*`
    sys.path.insert(0, str(_SRC))

__all__ = [
    "scene",
    "commands",
    "pushes",
    "fall",
    "markers",
    "obs",
    "reward",
    "metrics",
    "env",
    "eval",
    "baselines",
    "video",
    "lock",
]
