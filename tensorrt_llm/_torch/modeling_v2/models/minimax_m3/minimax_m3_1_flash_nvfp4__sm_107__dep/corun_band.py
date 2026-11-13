# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_CORUN_BAND: at 8-way expert parallelism, the MegaMoE launches of the buckets above 32 up to 64 rows per rank keep
the shared expert beside the launch on the aux stream; every other bucket keeps the backend's serial scope.

The band is the MegaMoE backend's co-run band (``local_plan.shared_corun_band``), which the backend reads from
the environment when it is built and which defaults to none. ``install`` gives each 8-way MegaMoE backend
``DEFAULT_BAND`` where that environment sets no band, before any launch is captured; other expert-parallel sizes keep
the backend default; where DEP_SE_SIDE engages (``se_side``) the model passes its wider ``se_side.CORUN_BAND``
instead. An explicit ``TRTLLM_M31_MEGAMOE_SHARED_CORUN_BAND=LO:HI`` stays as set, at every size, and
``TRTLLM_M31_DISABLE_MEGAMOE_SHARED_CORUN_BAND=1`` keeps every bucket serial; the backend reads both.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch

from tensorrt_llm._torch.moe.fused_moe.mega_moe import TrtllmCutedslMegaMoeNvfp4Impl

#: ``(LO, HI)`` rows per rank of the band the target defaults to.
DEFAULT_BAND = (32, 64)
#: The expert-parallel size whose MegaMoE backends take the default.
EP_SIZE = 8


def install(layers: Iterable[torch.nn.Module], band: Tuple[int, int] = DEFAULT_BAND) -> int:
    """Give the MegaMoE backend of each MoE layer at ``EP_SIZE``-way expert parallelism ``band`` (``DEFAULT_BAND``,
    or DEP_SE_SIDE's ``se_side.CORUN_BAND`` where that engages) where the environment it was built with sets no band and
    does not disable it; returns how many of those backends run ``band`` afterwards."""
    count = 0
    for layer in layers:
        experts = getattr(getattr(layer, "block_sparse_moe", None), "experts", None)
        backend = getattr(experts, "backend", experts)
        if not isinstance(backend, TrtllmCutedslMegaMoeNvfp4Impl) or backend.ep_size != EP_SIZE:
            continue
        if not hasattr(backend, "default_shared_corun_band"):
            # The backend has no co-run band on this runtime: every bucket keeps its serial scope.
            continue
        backend.default_shared_corun_band(band)
        if backend.shared_corun_band() == band:
            count += 1
    return count
