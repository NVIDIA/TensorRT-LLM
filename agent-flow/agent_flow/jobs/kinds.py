"""Handle-kind adapters: cancel a detached job by its durable handle.

Each kind knows how to cancel (``scancel`` / ``SIGTERM``) a job the framework
recorded, so an invalidated node's still-running job is killed rather than
orphaned. Status polling on resume is the AGENT's job — it reads its own
``jobs.json`` and runs ``squeue``/``sacct`` itself — so the framework keeps only
``cancel``.
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Callable, Protocol


class JobKind(Protocol):
    def cancel(self, handle: dict) -> None: ...


def _cli(argv: list[str]) -> tuple[int, str]:
    p = subprocess.run(argv, capture_output=True, text=True)
    return p.returncode, p.stdout


class SlurmKind:
    """Slurm handle-kind: cancel via ``scancel``."""

    def __init__(self, run: Callable[[list[str]], tuple[int, str]] = _cli) -> None:
        self._run = run

    def cancel(self, handle: dict) -> None:
        self._run(["scancel", str(handle["job_id"])])


class LocalKind:
    """Local handle-kind: cancel via ``SIGTERM`` to the recorded pid."""

    def cancel(self, handle: dict) -> None:
        try:
            os.kill(int(handle["pid"]), signal.SIGTERM)
        except OSError:
            pass


KINDS: dict[str, JobKind] = {"slurm": SlurmKind(), "local": LocalKind()}
