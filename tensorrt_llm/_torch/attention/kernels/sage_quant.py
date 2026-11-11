# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SageAttention quantization with TensorRT-LLM's Sage quantization kernel.

``sage_quantize`` quantizes Q, K, and V, or, given the statistics of an earlier call, only K and
V so that they dequantize like that call's K and V, as the K/V block means of proxy routes must.
The attention core runs it ahead of the FMHA libraries that take quantized inputs.
"""

from typing import NamedTuple

import torch

__all__ = [
    "SageQuantized",
    "sage_quantize",
]


class SageQuantized(NamedTuple):
    """Quantized Q/K/V and their FP32 dequantization scales.

    ``q_scale`` and ``k_scale`` are ``[heads, ceil(tokens / block) + batch - 1]``: sequence
    ``b`` starts at slot ``cu_seqlens[b] // block + b``. ``v_scale``, ``k_mean``, and
    ``v_mean`` are ``[batch, heads_kv, head_dim]``; the means are subtracted from K and V
    before quantization and are ``None`` without K or V smoothing. ``q`` and ``q_scale`` are
    ``None`` when the statistics of an earlier call were reused. The summary fields hold the
    quantized K/V block means of proxy routes and their per-mean key scales, for calls that
    have them.
    """

    q: torch.Tensor | None
    k: torch.Tensor
    v: torch.Tensor
    q_scale: torch.Tensor | None
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    k_mean: torch.Tensor | None = None
    v_mean: torch.Tensor | None = None
    k_summary: torch.Tensor | None = None
    v_summary: torch.Tensor | None = None
    k_summary_scale: torch.Tensor | None = None


def _flat_scale_numel(tokens: int, batch: int, block_size: int) -> int:
    return -(-tokens // block_size) + batch - 1


def _check_qk_quant_dtype(qk_quant_dtype: torch.dtype) -> None:
    if qk_quant_dtype not in (torch.int8, torch.float8_e4m3fn):
        raise ValueError("qk_quant_dtype must be torch.int8 or torch.float8_e4m3fn")


def sage_quantize(
    query: torch.Tensor | None,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_kv: torch.Tensor,
    *,
    q_block_size: int = 0,
    k_block_size: int,
    qk_quant_dtype: torch.dtype,
    smooth_k: bool = False,
    smooth_v: bool = False,
    stats: SageQuantized | None = None,
) -> SageQuantized:
    """Quantize Q/K/V for SageAttention, with V scales and the means per sequence.

    Q and K get one scale per ``q_block_size`` / ``k_block_size`` tokens of a sequence (1, 4,
    16, or 32); V gets one FP8 scale per channel of each sequence. With
    ``smooth_k``, every sequence's per-channel K mean is subtracted before K is quantized,
    which leaves the attention output unchanged. With ``smooth_v``, the per-channel V mean
    is subtracted before V is quantized; the attention kernel must add it back.

    With ``stats``, the result of an earlier call, only K and V are quantized, with the V
    scales and the means of ``stats``, so that they dequantize like its K and V; K still takes
    its own scales. The smoothing follows ``stats``, and the result carries its statistics and
    no Q.

    Args:
        query: Contiguous BF16/FP16 CUDA ``[tokens_q, heads_q, head_dim]``; ``None`` with
            ``stats``.
        key: Contiguous ``[tokens_kv, heads_kv, head_dim]`` with the dtype of ``value``.
        value: Same shape and dtype as ``key``.
        cu_seqlens_q: INT32 ``[batch + 1]`` cumulative query lengths; ``None`` with ``stats``.
        cu_seqlens_kv: INT32 ``[batch + 1]`` cumulative key/value lengths.
        q_block_size: Query tokens per scale.
        k_block_size: Key tokens per scale.
        qk_quant_dtype: ``torch.int8`` or ``torch.float8_e4m3fn`` for Q and K.
        smooth_k: Subtract the per-sequence K mean before quantizing K.
        smooth_v: Subtract the per-sequence V mean before quantizing V.
        stats: ``None``, or an earlier result with the batch, heads, and head dimension of
            ``key``, whose V scales and means to reuse.
    """
    _check_qk_quant_dtype(qk_quant_dtype)
    if stats is not None and (
        query is not None or cu_seqlens_q is not None or smooth_k or smooth_v
    ):
        raise ValueError("with stats, only K and V are quantized, with the smoothing of stats")
    batch = cu_seqlens_kv.numel() - 1
    heads_kv, head_dim = key.shape[1], key.shape[2]
    k = torch.empty_like(key, dtype=qk_quant_dtype)
    v = torch.empty_like(value, dtype=torch.float8_e4m3fn)
    k_scale = torch.empty(
        (heads_kv, _flat_scale_numel(key.shape[0], batch, k_block_size)),
        dtype=torch.float32,
        device=key.device,
    )
    if stats is not None:
        torch.ops.trtllm.sage_quantize(
            None,
            key,
            value,
            None,
            cu_seqlens_kv,
            None,
            k,
            v,
            None,
            k_scale,
            stats.v_scale,
            stats.k_mean,
            stats.v_mean,
            0,
            k_block_size,
            reuse_stats=True,
        )
        return SageQuantized(
            q=None,
            k=k,
            v=v,
            q_scale=None,
            k_scale=k_scale,
            v_scale=stats.v_scale,
            k_mean=stats.k_mean,
            v_mean=stats.v_mean,
        )

    q = torch.empty_like(query, dtype=qk_quant_dtype)
    q_scale = torch.empty(
        (query.shape[1], _flat_scale_numel(query.shape[0], batch, q_block_size)),
        dtype=torch.float32,
        device=query.device,
    )
    v_scale = torch.empty((batch, heads_kv, head_dim), dtype=torch.float32, device=value.device)
    k_mean = torch.empty_like(v_scale) if smooth_k else None
    v_mean = torch.empty_like(v_scale) if smooth_v else None
    torch.ops.trtllm.sage_quantize(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        k_mean,
        v_mean,
        q_block_size,
        k_block_size,
    )
    return SageQuantized(
        q=q,
        k=k,
        v=v,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        k_mean=k_mean,
        v_mean=v_mean,
    )
