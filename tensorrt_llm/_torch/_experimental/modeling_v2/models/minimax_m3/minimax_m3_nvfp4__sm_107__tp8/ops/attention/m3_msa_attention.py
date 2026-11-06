# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MiniMax-M3 MSA attention backend's step: KV write, sparse top-k block selection, FMHA."""

from __future__ import annotations

from typing import Optional as Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import OpWrapper as OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention.msa import (
    GroupedMsaAttentionBase,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention.msa import (
    _selected_blocks as _selected_blocks,
)
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs as AttentionForwardArgs,
)
from tensorrt_llm._torch.attention.backends.interface import (
    SparseBackendForwardArgs as SparseBackendForwardArgs,
)

from .m3_sparse_decode_lc8 import dequant_nvfp4_pages as dequant_nvfp4_pages


class M3MsaAttention(GroupedMsaAttentionBase):
    """One layer's attention core on the MSA backend (``MiniMaxM3MsaSparseAttention``), written into ``output``
    [T, heads * 128] (bf16, the step's live rows).

    * ``k`` / ``v`` given (the dense layers): written into the layer's paged main cache first (NVFP4 with
      ``kv_scale_orig_quant``, else FP8 / BF16 as the pool is). None: a producer already wrote the caches.
    * ``sparse``: the indexer scores the layer's index-K cache (already written) with ``idx_q`` and publishes the
      top-k blocks; the FMHA attends those blocks. Otherwise the FMHA attends the whole page table.

    The FMHA dispatches on the step's metadata (context rows: the MSA prefill FMHA; generation rows: the MSA decode
    FMHA, the dense layers' trtllm-gen decode at their capped KV split). ``backend`` and ``metadata`` are the runtime's
    objects for this layer and step.
    """

    @staticmethod
    def _dequant_pages(*args, **kwargs) -> torch.Tensor:
        return dequant_nvfp4_pages(*args, **kwargs)

    @staticmethod
    def _select_blocks(*args, **kwargs) -> torch.Tensor:
        return _selected_blocks(*args, **kwargs)

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="dense layer, a prefill chunk, FP8 pool",
            spec=dict(layer="dense", phase="context", tokens=1024),
        ),
        Cell(
            why="dense layer, a decode step of 2 requests",
            spec=dict(layer="dense", phase="generation", reqs=2),
        ),
        Cell(
            why="sparse layer, a prefill chunk beyond 16 blocks, NVFP4 pool",
            spec=dict(layer="sparse", phase="context", tokens=4096),
        ),
        Cell(
            why="sparse layer, a mixed step (context + generation rows)",
            spec=dict(layer="sparse", phase="mixed", tokens=512, reqs=2),
        ),
    )

    note = """
    `reference` is exact (float64) softmax attention over the cache rows the step attends, read back from the paged
    cache the call wrote (or a producer wrote before it) and dequantized as the FMHA dequantizes them: FP8 rows as
    E4M3, NVFP4 rows as E2M1 x E4M3 block scale x the per-tensor ``kv_scale_quant_orig``. A sparse layer's top-k blocks
    are re-selected from ``idx_q`` and the index-K cache with the backend's rule (per-block max over the causal
    keys, the query's own block forced in, ties to the lower block). On an FP8 pool the FMHA takes Q as a plain E4M3
    cast, which the reference applies too. What remains between the two is the FMHA's own arithmetic (its P
    rounding and accumulation order), which the test bounds.
    """


m3_msa_attention = M3MsaAttention()
