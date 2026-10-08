"""Advisory single-heavy-process lock for this host (``data/locks/sim.lock``).

Several agents run simulations and software-EGL renders on four cores; racing
them makes every measurement (and every video) unrepresentative.  Any run
longer than about a minute takes this lock first::

    from drill.lock import sim_lock
    with sim_lock("L1 90 s run"):
        ...                      # the heavy part

The lock is an ``flock`` on ``data/locks/sim.lock``; the holder's purpose, pid
and start time are written into the file so a blocked caller can report who is
running.  It waits (default 20 min) and then raises :class:`LockBusy` -- waiting
is better than racing, and reporting is better than both silently degrading.
Renderers and simulations take the *same* lock: both are CPU hogs here.
"""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO / "data" / "locks" / "sim.lock"


class LockBusy(RuntimeError):
    """Raised when the heavy-process lock could not be taken in time."""


def holder(path: Path | str = LOCK_PATH) -> str:
    """Current holder description ('' when free)."""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        return p.read_text().strip()
    except OSError:
        return ""


@contextmanager
def sim_lock(purpose: str = "sim", path: Path | str = LOCK_PATH,
             timeout: float = 1200.0, poll: float = 2.0, verbose: bool = True):
    """Exclusive advisory lock for one heavy process; waits, then raises."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(p, "a+")
    t0 = time.time()
    waited = False
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if not waited and verbose:
                print(f"[lock] waiting for {holder(p) or 'another holder'}", flush=True)
                waited = True
            if time.time() - t0 > timeout:
                fh.close()
                raise LockBusy(f"{p} held by {holder(p)!r} for >{timeout:.0f}s; "
                               f"wanted it for {purpose!r}")
            time.sleep(poll)
    try:
        fh.seek(0)
        fh.truncate()
        fh.write(f"{purpose} pid={os.getpid()} since={time.strftime('%H:%M:%S')}")
        fh.flush()
        yield fh
    finally:
        try:
            fh.seek(0)
            fh.truncate()
            fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()


if __name__ == "__main__":                              # self-check
    with sim_lock("self-check"):
        print("lock acquired; holder:", holder())
    print("lock released; holder:", repr(holder()))
    assert holder() == ""
