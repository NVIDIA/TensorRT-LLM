# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 DSA prefill scoring over FP8 queries and gathered FP8 keys."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.ops.attention.fp8_mqa_logits import (  # noqa: E501
    fp8_mqa_logits,
)


@triton.jit
def _score(
    Q,
    K,
    SCALE,
    WEIGHT,
    START,
    END,
    OUTPUT,
    M,
    N,
    OUTPUT_ROW_STRIDE,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    S0: tl.constexpr,
    START0: tl.constexpr,
    END0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    qh = tl.arange(0, BLOCK_M * 32)
    rows = tl.program_id(0) * BLOCK_M + qh // 32
    heads = qh % 32
    keys = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dim = tl.arange(0, 128)
    query = tl.load(
        Q + rows[:, None] * Q0 + heads[:, None] * Q1 + dim[None, :] * Q2,
        rows[:, None] < M,
        other=0.0,
    )
    key = tl.load(K + dim[:, None] * K1 + keys[None, :] * K0, keys[None, :] < N, other=0.0)
    dots = tl.dot(query, key, out_dtype=tl.float32, max_num_imprecise_acc=0)
    weight = tl.load(WEIGHT + rows * W0 + heads * W1, rows < M, other=0)
    weighted = tl.maximum(dots, 0) * weight[:, None]
    score = tl.sum(weighted.reshape(BLOCK_M, 32, BLOCK_N), axis=1)
    scale = tl.load(SCALE + keys * S0, keys < N, other=0)
    score *= scale[None, :]
    output_rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    start = tl.load(START + output_rows * START0, output_rows < M, other=0)
    end = tl.load(END + output_rows * END0, output_rows < M, other=0)
    valid = (keys[None, :] < N) & (keys[None, :] >= start[:, None]) & (keys[None, :] < end[:, None])
    score = tl.where(valid, score, -float("inf"))
    tl.store(
        OUTPUT + output_rows[:, None].to(tl.int64) * OUTPUT_ROW_STRIDE + keys[None, :],
        score,
        (output_rows[:, None] < M) & (keys[None, :] < OUTPUT_ROW_STRIDE),
    )


class TritonFp8MqaLogits(OpWrapper):
    """FP32 sum of weighted, rectified FP8 dot products, masked by each query's window.

    The key's FP32 scale is applied after the head reduction. Inputs retain
    their original FP8 codes; no BF16 dequantization precedes the dot product.
    Rows are padded to 16-byte alignment for the prefill top-k consumer.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS = tuple(
        Cell(f"SM107 indexer rows={m}, keys={n}", dict(m=m, n=n))
        for m in (1, 17, 129, 2048)
        for n in (33, 2051, 4096)
    ) + (
        Cell("SM107 strided scales and windows", dict(m=17, n=2051, strided=True)),
        Cell("SM107 32K context tile", dict(m=512, n=32768)),
        Cell("SM107 1M context tile", dict(m=64, n=1048576)),
    )

    def __call__(
        self,
        q: torch.Tensor,
        kv: tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seq_len_k_start: torch.Tensor,
        cu_seq_len_k_end: torch.Tensor,
    ) -> torch.Tensor:
        k, scale = kv
        if q.shape[1:] != (32, 128) or k.ndim != 2 or k.shape[1] != 128:
            raise ValueError("SM107 MQA scorer requires 32 query heads and head dimension 128")
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn:
            raise ValueError("SM107 MQA scorer requires FP8 E4M3 queries and keys")
        padded_n = triton.cdiv(k.shape[0], 4) * 4
        output = torch.empty((q.shape[0], padded_n), dtype=torch.float32, device=q.device)[
            :, : k.shape[0]
        ]
        if not output.numel():
            return output
        _score[(triton.cdiv(q.shape[0], 4), triton.cdiv(k.shape[0], 128))](
            q,
            k,
            scale,
            weights,
            cu_seq_len_k_start,
            cu_seq_len_k_end,
            output,
            q.shape[0],
            k.shape[0],
            output.stride(0),
            *q.stride(),
            *k.stride(),
            *weights.stride(),
            scale.stride(0),
            cu_seq_len_k_start.stride(0),
            cu_seq_len_k_end.stride(0),
            4,
            128,
            num_warps=8,
        )
        return output

    def reference(
        self,
        q: torch.Tensor,
        kv: tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seq_len_k_start: torch.Tensor,
        cu_seq_len_k_end: torch.Tensor,
    ) -> torch.Tensor:
        return fp8_mqa_logits.reference(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end)


triton_fp8_mqa_logits = TritonFp8MqaLogits()
