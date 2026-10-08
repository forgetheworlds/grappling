"""Training-time push curriculum for the balance task (T1 v2 fix).

Why: ``TASKS['balance']`` shipped with no pushes, so the T1 gate's push criteria
were unattainable by construction (a policy trained without pushes cannot learn
recovery).  This module ramps push difficulty with training progress while
**staying strictly inside the held-out boundary**: impulses never exceed
``TRAIN_MAX_IMPULSE`` (12 N*s); the gate's held-out magnitudes (>12 N*s) stay
unseen.  The gate battery itself is untouched -- it is the evaluation, never the
training signal.

Schedule (defaults, deterministic given the episode seed):

* progress ``p = clip((steps - start_steps) / warmup_steps, 0, 1)``
* unlocked magnitudes: the first ``1 + round(p*(len(magnitudes)-1))`` of
  ``(2, 4, 6, 8, 10, 12)`` N*s
* unlocked directions: the first ``1 + round(p*(max_directions-1))`` world yaws
  (evenly spaced, same seeding style as the eval battery but a *different*
  seed stream so training conditions are not the eval conditions)
* 0 pushes before ``start_steps``; then 1 push per episode, 2 pushes once
  ``p >= 0.5`` (spaced ``gap`` seconds apart), application height fixed at
  ``height`` (0.95 m chest) -- height variation stays an evaluation axis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .pushes import TRAIN_MAX_IMPULSE, PushSchedule, PushSpec
from .scene import STEP_DT


@dataclass(frozen=True)
class PushCurriculum:
    """Progress-ramped training pushes (impulses <= ``TRAIN_MAX_IMPULSE``)."""

    start_steps: int = 20_000
    warmup_steps: int = 400_000
    magnitudes: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
    max_directions: int = 8
    height: float = 0.95
    t_first: float = 1.0
    gap: float = 1.5
    max_impulse: float = TRAIN_MAX_IMPULSE
    seed: int = 0

    def __post_init__(self) -> None:
        if max(self.magnitudes) > self.max_impulse + 1e-9:
            raise ValueError(
                f"curriculum magnitude {max(self.magnitudes)} exceeds the training "
                f"cap {self.max_impulse} N*s (held-out boundary)")

    def progress(self, steps: int) -> float:
        return float(np.clip((float(steps) - self.start_steps)
                             / max(1.0, float(self.warmup_steps)), 0.0, 1.0))

    def unlocked(self, steps: int) -> tuple[tuple[float, ...], int, int]:
        """(magnitudes, n_directions, n_pushes) unlocked at ``steps``."""
        p = self.progress(steps)
        if float(steps) < self.start_steps:
            return tuple(), 0, 0
        n_mag = 1 + int(round(p * (len(self.magnitudes) - 1)))
        n_dir = 1 + int(round(p * (max(1, self.max_directions) - 1)))
        n_pushes = 2 if p >= 0.5 else 1
        return tuple(self.magnitudes[:n_mag]), n_dir, n_pushes

    def schedule_for(self, steps: int, episode_seed: int | None = None
                     ) -> PushSchedule:
        """Deterministic push schedule for one training episode."""
        mags, n_dir, n_push = self.unlocked(steps)
        if n_push == 0:
            return PushSchedule([])
        rng = np.random.default_rng(int(self.seed) * 1_000_003
                                    + (0 if episode_seed is None
                                       else int(episode_seed)))
        specs = []
        for k in range(n_push):
            m = float(mags[int(rng.integers(0, len(mags)))])
            d = int(rng.integers(0, n_dir))
            ang = 2.0 * np.pi * d / max(1, self.max_directions)
            t0 = float(self.t_first) + k * float(self.gap) * float(rng.uniform(0.9, 1.1))
            t0 = round(t0 / STEP_DT) * STEP_DT
            specs.append(PushSpec(t=t0, impulse=m, direction=ang,
                                  height=self.height, duration=STEP_DT,
                                  label=f"train_m{m:g}_d{d}"))
        return PushSchedule(specs)

    def as_dict(self) -> dict:
        return asdict(self)


#: default curriculum attached to the balance task preset
DEFAULT_CURRICULUM = PushCurriculum()


if __name__ == "__main__":  # self-check
    c = DEFAULT_CURRICULUM
    assert c.unlocked(0)[2] == 0
    assert c.unlocked(c.start_steps - 1)[2] == 0
    start = c.unlocked(c.start_steps)
    mid = c.unlocked(c.start_steps + c.warmup_steps // 2)
    end = c.unlocked(c.start_steps + c.warmup_steps)
    assert start[1] <= mid[1] <= end[1] == c.max_directions
    assert len(start[0]) <= len(mid[0]) <= len(end[0]) == len(c.magnitudes)
    assert start[2] == 1 and mid[2] == 2 and end[2] == 2
    s1 = c.schedule_for(int(c.start_steps + c.warmup_steps), episode_seed=7)
    s2 = c.schedule_for(int(c.start_steps + c.warmup_steps), episode_seed=7)
    assert s1.as_list() == s2.as_list()
    assert all(p.impulse <= c.max_impulse for p in s1.pushes)
    assert all(p.impulse <= c.max_impulse for p in
               c.schedule_for(c.start_steps, episode_seed=1).pushes)
    print("solo.curriculum self-check OK:", {
        "start": start, "mid": mid, "end": end,
        "sample": s1.as_list()[0]["impulse"]})
