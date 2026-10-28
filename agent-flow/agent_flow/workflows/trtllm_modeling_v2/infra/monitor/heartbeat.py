# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The orchestrator's liveness files in the run's ``dispatch/`` directory.

The operator loop (:mod:`.status`) tells a busy orchestrator from one that has exited by
``orchestrator.heartbeat``, rewritten every :data:`ORCHESTRATOR_HEARTBEAT_S`, and
``orchestrator.exit``, left when the process ends.
"""

from __future__ import annotations

import atexit
import errno
import json
import os
import socket
import threading
import time
from pathlib import Path

from ..files import write_text_atomic

#: Subdirectory of the workspace the orchestrator and the operator loop share.
DISPATCH_DIRNAME = "dispatch"

#: The orchestrator's liveness file, and the file it leaves when it exits.
ORCHESTRATOR_HEARTBEAT = "orchestrator.heartbeat"
ORCHESTRATOR_EXIT = "orchestrator.exit"

#: How often the orchestrator says it is alive.
ORCHESTRATOR_HEARTBEAT_S = 30.0


class OrchestratorHeartbeat:
    """This orchestrator's liveness files; start one per process, from the workspace lock holder.

    Each holds ``{"pid", "host", "started", "at"}``, ``started`` being when this process's
    orchestrator side began.
    """

    def __init__(
        self,
        workspace: Path | str,
        interval_s: float,
        *,
        started: float | None = None,
    ) -> None:
        self.root = Path(workspace) / DISPATCH_DIRNAME
        self.root.mkdir(parents=True, exist_ok=True)
        self.record = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started": time.time() if started is None else started,
        }
        self._stop = threading.Event()
        (self.root / ORCHESTRATOR_EXIT).unlink(missing_ok=True)
        threading.Thread(
            target=self._beat, args=(interval_s,), name="dispatch-heartbeat", daemon=True
        ).start()
        atexit.register(self.mark_exit)

    def _beat(self, interval_s: float) -> None:
        while True:
            try:
                write_text_atomic(
                    self.root / ORCHESTRATOR_HEARTBEAT,
                    json.dumps({**self.record, "at": time.time()}) + "\n",
                )
            except OSError as exc:
                # A full filesystem is transient (a shared quota); anything else means the
                # dispatch directory is gone.
                if exc.errno not in (errno.ENOSPC, errno.EDQUOT):
                    return
            if self._stop.wait(interval_s):
                return

    def mark_exit(self) -> None:
        """Stop beating and write ``orchestrator.exit``; what ``atexit`` calls."""
        self._stop.set()
        try:
            write_text_atomic(
                self.root / ORCHESTRATOR_EXIT,
                json.dumps({**self.record, "at": time.time()}) + "\n",
            )
        except OSError:
            pass


def start_orchestrator_heartbeat(
    workspace: Path | str,
    interval_s: float,
    *,
    started: float | None = None,
) -> OrchestratorHeartbeat:
    """Start this process's :class:`OrchestratorHeartbeat` in ``workspace``'s ``dispatch/``."""
    return OrchestratorHeartbeat(workspace, interval_s, started=started)


__all__ = [
    "DISPATCH_DIRNAME",
    "ORCHESTRATOR_EXIT",
    "ORCHESTRATOR_HEARTBEAT",
    "ORCHESTRATOR_HEARTBEAT_S",
    "OrchestratorHeartbeat",
    "start_orchestrator_heartbeat",
]
