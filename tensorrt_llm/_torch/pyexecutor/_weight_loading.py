# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit rank-local completion event for opt-in disaggregated QA tests."""

import os
import time
from pathlib import Path


def notify_weight_loading_complete(rank: int) -> None:
    """Publish after ModelLoader.load returns, including weight finalization.

    No environment variable means no filesystem access. No extra distributed
    barrier is introduced: the test supervisor waits for each rank's event.

    Args:
        rank: Rank within the worker's model-parallel world.
    """
    directory = os.environ.get("TRTLLM_TEST_WEIGHT_LOADING_DIR")
    if not directory:
        return
    destination = Path(directory) / f"rank-{rank}.done"
    # A worker relaunched within this attempt republishes its ranks. Keep
    # the first completion at the phase boundary.
    if destination.exists():
        return
    temporary = destination.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(str(time.monotonic()), encoding="utf-8")
    temporary.replace(destination)
