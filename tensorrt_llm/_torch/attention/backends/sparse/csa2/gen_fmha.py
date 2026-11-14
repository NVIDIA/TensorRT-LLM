# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 generation attention on the sm_103 sparse MLA CuTe DSL kernel (``kernels/blackwell/sparse_mla``).

Generation tiles of 64 query heads per rank (attention DP) with E4M3 staging on sm_103 that the backend
routes here (``CSA2TrtllmAttention.gen_fmha_serves``) run this core instead of the fused decode or the native
generation cubin: the same FP8 query, BMM scales, sink and staged E4M3 rows, attended with
the core's softmax (lazy rescale, E4M3 P). Both staging pools of such a tile are one allocation and the staging
publishes the extra pool's indices as rows of that bank, so the core reads every row through one base address. The
core writes the BF16 output the native kernel writes, or the grouped MXFP8 O-A input that output quantizes to.
"""

from __future__ import annotations

import functools

import torch

HEADS = 64
HEAD_DIM = 512
# The core's per-token list buffer: 128 SWA rows + 512 selected rows.
LIST_COLUMNS = 640
# The core copies each list in whole tiles of this many keys.
LIST_TILE = 64


@functools.cache
def gen_fmha_device_supported(device_index: int) -> bool:
    """The core reads its row maxima with ``tcgen05.ld.red``, which needs sm_103."""
    return torch.cuda.get_device_capability(device_index) == (10, 3)


def gen_fmha_fits(heads: int, width: int) -> bool:
    """Whether the core serves ``heads`` query heads over ``width``-column staging lists."""
    return heads == HEADS and width <= LIST_COLUMNS and width % LIST_TILE == 0


def staging_bank(metadata) -> tuple[int, int] | None:
    """``(base address, extra pool's first row)`` of a tile whose E4M3 pools ``CSA2TrtllmMetadata``
    allocated as one bank (``gen_fmha_bank``), in 512-byte rows from the primary pool's base; else None."""
    bank = metadata.gen_fmha_bank
    if bank is None or bank[0].dtype != torch.float8_e4m3fn:
        return None
    backing, extra_row = bank
    return backing.data_ptr(), extra_row


def gen_sparse_attention(
    q8: torch.Tensor,
    metadata,
    sinks: torch.Tensor,
    output: torch.Tensor | None,
    mxfp8: tuple[torch.Tensor, ...] | None = None,
) -> None:
    """Attend ``q8`` (E4M3 ``[T, 64, 512]``) over each token's published staging rows into ``output``
    (BF16 ``[T, 64, 512]``), reading the tile's ``prepared_indices`` / ``prepared_lens`` (bank rows)
    and BMM scales. With ``mxfp8`` = (E4M3 ``[8, T, 4096]``, UE8M0 scales, int32 positions, inverse-RoPE
    cos / sin), the core writes the grouped MXFP8 O-A input there instead (``epi="rope_mxfp8"``)."""
    from tensorrt_llm._torch.attention.kernels.blackwell.sparse_mla.launch_gb300 import (
        sparse_mla_attention_gb300,
    )
    from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL

    tokens = q8.shape[0]
    base, _ = staging_bank(metadata)
    # BMM1's second entry is Q dequant x KV dequant x softmax scale x log2(e); BMM2 is the KV dequant.
    # With PDL the core's setup overlaps the staging, which releases its dependent at entry.
    sparse_mla_attention_gb300(
        q8,
        base,
        metadata.prepared_indices[:tokens],
        metadata.prepared_lens[:tokens],
        sinks,
        metadata.mla_bmm1_scale[1:],
        metadata.mla_bmm2_scale,
        output,
        pdl=TRTLLM_ENABLE_PDL,
        **(
            {}
            if mxfp8 is None
            else dict(
                epi="rope_mxfp8",
                out_fp8=mxfp8[0],
                out_scale=mxfp8[1],
                positions=mxfp8[2],
                rope_table=mxfp8[3],
            )
        ),
    )
