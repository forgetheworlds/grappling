"""PPO training infrastructure shared by the solo drill (mandate path).

Layering (each module is independently importable):

* :mod:`rl.constants`   G1 action-layout constants (``N_JOINTS``, ``ACT_SLICE``)
* :mod:`rl.net`         actor/critic MLPs + unit-action -> ctrlrange action map
* :mod:`rl.ppo`         PPO config, GAE(lambda), clipped update, LR schedule
* :mod:`rl.checkpoint`  atomic checkpoints (weights, optimizer, RNG, curriculum
                        state), BC warm-start loader
* :mod:`rl.vec_solo`    subprocess env-worker backend for ``solo.env.SoloEnv``
                        (``python -m src.rl.vec_solo`` runs the self-check)

The two-robot modules (``obs``, ``privileged``, ``reward``, ``scripted``,
``curriculum``, ``vec``, ``rollout``, ``trainer``) and their entry points went
with ``src/wrestling`` on 2026-10-08; the solo trainer is
:mod:`solo.train`, which reuses ``rl.net``/``rl.ppo``/``rl.checkpoint``.  The
removal and the extracted constants are documented in
``reports/2026-10-08/repo_trim.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The repo's packages live under ``src/`` (see scripts/*.py).  Make the sibling
# packages (``solo``, ``drill``, ``scorer``, ...) importable both when this
# package is imported as ``rl`` (src already on sys.path) and as ``src.rl``
# (repo root on sys.path).
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

__all__ = ["constants", "net", "ppo", "checkpoint", "vec_solo"]
