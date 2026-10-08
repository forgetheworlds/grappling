"""Phase-5 resistance training infrastructure (PPO) for the G1 wrestling env.

Layering (each module is independently importable):

* :mod:`rl.obs`         actor observation (technique one-hot + phase + frame stack)
* :mod:`rl.privileged`  critic-only privileged state (exact contacts, qpos/qvel,
                        back-exposure proxy) -- strictly separate from the actor
* :mod:`rl.net`         actor/critic MLPs + unit-action -> ctrlrange action map
* :mod:`rl.ppo`         PPO config, GAE(lambda), clipped update, LR schedule
* :mod:`rl.scripted`    scripted opponent specs (StandHold / ReferenceReplay /
                        optional phase-3 teacher), reference base actions
* :mod:`rl.reward`      stage-wise reward weights + optional technique scorer
                        (``src/scorer``, imported lazily; degrades gracefully)
* :mod:`rl.curriculum`  PISTY stages A-E (opponent, command, weights, perturbations)
* :mod:`rl.vec`         rollout vectorization: in-process (sequential) and
                        multiprocessing (subproc) backends; ``python -m src.rl.vec``
                        prints the measured backend comparison
* :mod:`rl.rollout`     rollout collection + GAE finalization
* :mod:`rl.checkpoint`  atomic checkpoints (weights, optimizer, RNG, curriculum),
                        BC warm-start loader
* :mod:`rl.trainer`     the training loop (resumable, stage transitions, stats)

Entry points: ``scripts/train_ppo.py`` (smoke/resumable training) and
``scripts/eval_ppo.py`` (deterministic seeded evaluation).
"""

from __future__ import annotations

import sys
from pathlib import Path

# The repo's packages live under ``src/`` (see scripts/*.py).  Make the
# sibling ``wrestling`` package importable both when this package is imported
# as ``rl`` (src already on sys.path) and as ``src.rl`` (repo root on sys.path).
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

__all__ = [
    "obs", "privileged", "net", "ppo", "scripted", "reward", "curriculum",
    "vec", "rollout", "checkpoint", "trainer",
]
