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
import time
from typing import Any, Iterator, List

import torch

from tensorrt_llm.logger import logger

from . import linear, steps
from .group import BuildGroup

__all__ = ["warmup_overlap"]

_ENV = "TLLM_OVERLAP_PREWARM"
# Each part's build_groups(model, model_config, llm_args, mapping) returns its builds.
_PARTS = (steps, linear)


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
    except BaseException:
        # Leave the rest to the warmup rather than hold up the error.
        for group in groups:
            group.stop.set()
        raise
    t0 = time.perf_counter()
    for group in groups:
        group.join()
    if groups:
        # Return the dummy operands' memory before anything measures free memory.
        torch.cuda.empty_cache()
        logger.info(f"[warmup] overlap: waited {time.perf_counter() - t0:.1f}s for the builds")
