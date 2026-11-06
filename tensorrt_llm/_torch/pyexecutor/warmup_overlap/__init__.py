# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build, while the weights load, kernels the decoder warmup would build after them.

``warmup_overlap`` wraps the model loader's weight loading. On entry it starts the builds
of the parts below on background threads; on exit it waits for them. The warmup then runs
as before, and finds the kernels built.

- ``steps``: the steps ``DecoderWarmup`` starts with that need no model forward.
- Module-level kernels that the warmup's forwards would build, one file per kind of
  module: ``linear`` (the GEMM kernels of Linear layers).

``TLLM_OVERLAP_PREWARM=0`` leaves all of it to the warmup.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from typing import Any, Iterator, List, Optional

import torch

from tensorrt_llm.logger import logger

from . import linear, steps
from .group import BuildGroup

__all__ = ["warmup_overlap"]

_ENV = "TLLM_OVERLAP_PREWARM"
# Each part's build_groups(model, model_config, llm_args, mapping) returns its builds.
_PARTS = (steps, linear)
_PROGRESS_EVERY_S = 60.0
# How long a failed weight load waits for the builds to stop.
_STOP_TIMEOUT_S = 30.0


def _join(groups: List[BuildGroup], timeout: Optional[float] = None) -> List[threading.Thread]:
    """Join the groups' threads, logging their progress every minute; give up after
    ``timeout`` seconds. Returns the threads still running."""
    t0 = last = time.perf_counter()
    while alive := [thread for group in groups for thread in group.alive()]:
        now = time.perf_counter()
        if timeout is not None and now - t0 >= timeout:
            return alive
        if now - last >= _PROGRESS_EVERY_S:
            busy = ", ".join(group.progress() for group in groups if group.alive())
            logger.info(f"[warmup] overlap: waiting {now - t0:.0f}s for the builds: {busy}")
            last = now
        alive[0].join(timeout=1.0)
    return []


@contextlib.contextmanager
def warmup_overlap(
    model: torch.nn.Module, model_config: Any, llm_args: Any, mapping: Any
) -> Iterator[List[BuildGroup]]:
    """While the body of the ``with`` loads ``model``'s allocated weights, build on
    background threads kernels that the decoder warmup would build; leave the ``with`` once
    they are built. Yields the build groups."""
    groups: List[BuildGroup] = []
    if os.environ.get(_ENV, "1") != "0" and torch.cuda.is_available():
        for part in _PARTS:
            try:
                groups += part.build_groups(model, model_config, llm_args, mapping)
            except Exception as e:  # noqa: BLE001 - the warmup builds them instead
                logger.warning(
                    f"Overlapped {part.__name__.rsplit('.', 1)[-1]} builds skipped. "
                    f"{type(e).__name__}: {e}"
                )
        device = torch.cuda.current_device()
        for group in groups:
            group.start(device)
    try:
        yield groups
    except BaseException as e:
        # Leave the rest to the warmup: the builds stop after their current kernel.
        for group in groups:
            group.stop.set()
        if groups:
            running = _join(groups, timeout=_STOP_TIMEOUT_S)
            torch.cuda.empty_cache()
            if running:
                names = ", ".join(thread.name for thread in running)
                logger.warning(f"Overlapped builds still running after the loading failed: {names}")
            if isinstance(e, torch.cuda.OutOfMemoryError):
                logger.error(
                    "The weight loading ran out of device memory next to the overlapped kernel "
                    "builds; TLLM_OVERLAP_PREWARM_LINEAR=0 leaves the Linear GEMMs to the warmup."
                )
        raise
    t0 = time.perf_counter()
    _join(groups)
    if groups:
        # Return the dummy operands' memory before anything measures free memory.
        torch.cuda.empty_cache()
        logger.info(f"[warmup] overlap: waited {time.perf_counter() - t0:.1f}s for the builds")
