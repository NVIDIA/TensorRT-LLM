# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture-stable K=2048 streaming TopK, sharing the SMs with MLA q_b.

The device implementation and its per-row causal lengths are the common GVR kernels.
Only the small-batch streaming launch policy belongs to this model. Workspace is
caller-owned and must not be shared by concurrent launches.
"""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch.kernels.blackwell.top_k import gvr_topk_decode_self_sampling as device
from tensorrt_llm._torch.kernels.blackwell.top_k.gvr_topk_decode_self_sampling_host import (
    kernel_view,
    validate_run_ws,
    workspace_bytes,
)

TOPK = 2048
MAX_ROWS = 24


def streaming_plan(rows: int, envelope: int, num_sms: int, sm_reserve: int) -> tuple:
    """Return the current streaming-main ABI with a cap applied before deriving its fields."""
    if (
        not 1 <= rows <= min(MAX_ROWS, num_sms - sm_reserve)
        or not 0 <= sm_reserve < num_sms
        or envelope < 1
    ):
        raise ValueError("unsupported small-batch TopK launch geometry")
    n_route = max(envelope, TOPK + 1)
    splits = min(
        max(148 // rows, 1),
        max(((n_route >> 2) + 1023) // 1024, 1),
        max((num_sms - sm_reserve) // rows, 1),
    )
    # rows*splits <=148: this is always the large-CTA streaming branch.
    scap = 16384 if splits == 1 else 8192
    cmp = 4096
    q = ((n_route >> 2) + splits - 1) // splits
    per = q >> 10
    unroll = 8 if per >= 8 else 4 if per >= 4 else 2 if per >= 2 else 1
    tpl = (1024, unroll, 1, 256, 2, splits > 1, False)
    aim = 4 * TOPK if splits == 1 else 2 * TOPK
    sfac = 64 if splits == 1 else 32 if splits == 2 else 48
    amin = 3 * TOPK if splits == 2 else 7 * TOPK // 2
    return tpl, splits, scap, cmp, (aim, sfac, amin, 0, 0)


@functools.cache
def _device_profile(index: int) -> tuple[int, int]:
    props = torch.cuda.get_device_properties(index)
    return props.multi_processor_count, 10 * props.major + props.minor


_LAUNCHERS = {}


def _launcher(rows, stride, envelope, next_n, num_sms, sm_version, sm_reserve, deterministic):
    key = (rows, stride, envelope, next_n, num_sms, sm_version, sm_reserve, deterministic)
    hit = _LAUNCHERS.get(key)
    if hit is not None:
        return hit
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm up this full-context TopK shape before graph capture")
    tpl, splits, scap, cmp, tail = streaming_plan(rows, envelope, num_sms, sm_reserve)
    fn = device.get_compiled(
        tpl + (next_n, 0, splits),
        hint_free=True,
        det=deterministic,
        fused_k=TOPK,
        warp_rank=sm_version in (100, 103),
        staged_k=TOPK if splits == 1 and sm_version in (100, 103) else 0,
    )
    # The varlen kernel reads the dynamic causal lengths from kv_lens. The first
    # scalar n and the five sampling-ladder scalars are dead ABI slots in this mode.
    pre = (0, stride, TOPK, scap, cmp, splits, 0, 0, 0, 0, 0)
    result = fn, pre, tail
    _LAUNCHERS[key] = result
    return result


def full_context_topk(
    logits: torch.Tensor,
    kv_lens: torch.Tensor,
    indices: torch.Tensor,
    workspace: torch.Tensor,
    *,
    next_n: int,
    max_seq_len: int,
    sm_reserve: int,
    deterministic: bool,
) -> torch.Tensor:
    """Write int32 indices[rows,2048] from FP32 logits and device request lengths.

    Each request has ``next_n`` rows; row i sees ``kv_len-next_n+i+1`` keys.
    Short rows produce identity indices followed by -1. The physical row pitch
    and context envelope are capture-stable; request lengths remain device data.
    """
    if logits.ndim != 2 or logits.dtype != torch.float32 or not logits.is_cuda:
        raise ValueError("logits must be a CUDA FP32 matrix")
    rows = logits.shape[0]
    if not 1 <= rows <= MAX_ROWS or next_n < 1 or rows % next_n:
        raise ValueError("unsupported number of TopK rows or query length")
    if (
        indices.shape != (rows, TOPK)
        or indices.dtype != torch.int32
        or not indices.is_contiguous()
        or indices.device != logits.device
    ):
        raise ValueError("indices must be contiguous int32[rows,2048] on the logits device")
    if (
        kv_lens.shape != (rows // next_n,)
        or kv_lens.dtype != torch.int32
        or not kv_lens.is_contiguous()
        or kv_lens.device != logits.device
    ):
        raise ValueError("kv_lens must contain one contiguous CUDA int32 length per request")
    stride = logits.stride(0)
    if (
        logits.stride(1) != 1
        or stride < logits.shape[1]
        or stride % 4
        or logits.data_ptr() % 16
        or not 1 <= max_seq_len <= logits.shape[1]
    ):
        raise ValueError("logits must have an aligned row-major physical envelope")
    if logits.untyped_storage().nbytes() // 4 < logits.storage_offset() + rows * stride:
        raise ValueError("logits storage does not cover its physical row pitch")
    validate_run_ws(workspace, logits)
    if not workspace.is_contiguous():
        raise ValueError("TopK workspace must be contiguous")
    lg = logits.as_strided((rows, stride), (stride, 1))
    num_sms, sm_version = _device_profile(logits.device.index)
    fn, pre, tail = _launcher(
        rows,
        stride,
        min(max_seq_len, stride),
        next_n,
        num_sms,
        sm_version,
        sm_reserve,
        bool(deterministic),
    )
    # hint_free=True makes the pre_idx ABI slot unread; no separate hint buffer is needed.
    fn(lg, indices, indices, kernel_view(workspace), *pre, kv_lens, *tail)
    return indices


__all__ = ["MAX_ROWS", "TOPK", "full_context_topk", "streaming_plan", "workspace_bytes"]
