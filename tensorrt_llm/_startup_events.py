# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish successful startup phase completions for single-host observers."""

import os
import tempfile
import time
from pathlib import Path


def publish_phase_completion(phase: str, rank: int) -> None:
    """Write the first completion of a named phase on a worker rank.

    Args:
        phase: Startup phase name used in the event filename.
        rank: Rank within the worker's local model-parallel world.

    With `TRTLLM_STARTUP_EVENT_DIR` set, `<phase>.rank-<rank>.done`
    contains the host monotonic completion timestamp in seconds. The worker
    and observer must use the same host clock. A worker that relaunches in the
    same directory preserves the first event.
    """
    directory = os.environ.get("TRTLLM_STARTUP_EVENT_DIR")
    if not directory:
        return
    destination = Path(directory) / f"{phase}.rank-{rank}.done"
    if destination.exists():
        return
    completed = time.monotonic()
    with tempfile.TemporaryDirectory(dir=directory) as staging:
        event = Path(staging) / destination.name
        event.write_text(str(completed), encoding="utf-8")
        try:
            os.link(event, destination)
        except FileExistsError:
            pass  # Another publisher already recorded this rank and phase.
