# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Vendored from vLLM (Apache-2.0):
# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/vllm/models/minimax_m3/nvidia/ops/index_decode_score.py
"""MiniMax-M3 index decode block scorer (CuTe DSL, Blackwell) with its pre-wait schedule.

The kernel of ``tensorrt_llm/_torch/attention/kernels/blackwell/minimax_m3_index_decode_score.py`` (the same scores, bit
for bit) with two schedule options this target compiles in:

* ``prefetch_old``: before the grid-dependency wait, the TMA warp L2-prefetches this CTA's index-K pages that hold only
  tokens older than this step (the producer ahead writes only this step's tokens), up to ``prefetch_depth`` blocks per
  CTA (0 = all of them).
* ``early_issue``: the TMA warp reads a ``seq_lens`` hint and its first page id before the wait and issues the Q tile
  and the first K tile right after it. The tiles are read after the wait, so their contents are final; the block table
  is only written between graph replays, so the early page id is the right one whenever the block is valid. A CTA
  whose early tiles lie past the request's true length drains them before it exits.

Every kernel is compiled before the executor's warmup (``layers.prebuild_index_scorer``) through ``IndexScoreRunner``
and called through this target's ``m3_index_score`` entry.
"""

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.attention.index_score_runtime import (
    IndexScoreRunner as SharedIndexScoreRunner,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.attention.m3_index_score import (
    IndexDecodeScoreKernel as IndexDecodeScoreKernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.attention.m3_index_score import (
    _fp8_to_f16_mma_fragments as _fp8_to_f16_mma_fragments,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.attention.m3_index_score import (
    _l2_prefetch_bulk as _l2_prefetch_bulk,
)

__all__ = ["IndexDecodeScoreKernel", "IndexScoreRunner"]


class IndexScoreRunner(SharedIndexScoreRunner):
    """TP4 schedule and private compilation cache."""

    kernel_cache: dict = {}
    PRESTAGE_OLD = False
    EARLY_TRIGGER = False
