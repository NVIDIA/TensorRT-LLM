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
"""Torch op of the block drafter's context K/V (``trtllm::k3_ctx_kv``).

For the N = B (K + 1) <= 64 context tokens of a decode step (B <= 8 requests, K + 1 <= 8): the stacked K/V projection
of every drafter layer, k_norm, NeoX RoPE, the write mask and the paged store into the drafter's context pool, then
``ctx_len += num_accepted`` (clamped) and the context length each request may advertise, in one launch (see
``k3_ctx_kv_kernel``). Compiled on the first call for its shape (B, K + 1), which must happen outside CUDA-graph
capture.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, List

import torch

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}
_counters: Dict[torch.device, torch.Tensor] = {}
_sm_counts: Dict[torch.device, int] = {}


def _kernel_module():
    from . import k3_ctx_kv_kernel

    return k3_ctx_kv_kernel


def _arg(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    # detach(): DLPack refuses tensors that require grad (weights are parameters).
    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=t.dim() - 1)


def _index_arg(t: torch.Tensor):
    """An int index / length / table tensor the kernel reads element by element, declared at its element's alignment:
    the drafter passes per-request slices such as ``num_accepted_tokens[num_contexts:]``, which start on any element
    boundary."""
    return _arg(t, t.element_size())


def _sm_count(device: torch.device) -> int:
    n = _sm_counts.get(device)
    if n is None:
        n = _sm_counts[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return n


def pick_split(
    n_rows: int, k_in: int, nkv: int, k1: int, n_tokens: int, device: torch.device
) -> int:
    """The widest cluster (8, 4, 2) whose grid fits the SMs at once and whose shared memory holds the step's tokens
    (0: the shape does not run)."""
    kern = _kernel_module()
    for split in (8, 4, 2):
        if (n_rows // kern.CTA_M) * split <= _sm_count(device) and kern.supports(
            n_rows, k_in, split, nkv, k1, n_tokens
        ):
            return split
    return 0


def pool_view(layers: List[torch.Tensor]):
    """``(base, layer_off, page_stride, kv_stride, head_stride)`` for per-layer HND views [pages, 2, heads, slot, 64]
    with one set of strides and dense rows, or None. The views may be separate tensors (a V2 manager wraps each layer's
    base address in its own tensor) of one pool allocation: ``base`` is a one-element view at the lowest layer base and
    ``layer_off`` the element offsets of the layers from it; the kernel addresses the pool from there."""
    first = layers[0]
    if (
        first.dtype != torch.bfloat16
        or first.dim() != 5
        or first.stride(4) != 1
        or first.stride(3) != first.size(4)
    ):
        return None
    ptrs = [t.data_ptr() for t in layers]
    lowest = min(ptrs)
    for t, p in zip(layers, ptrs):
        if (
            t.stride() != first.stride()
            or t.dtype != first.dtype
            or t.device != first.device
            or (p - lowest) % 2
        ):
            return None
    anchor = layers[ptrs.index(lowest)]
    base = anchor.as_strided((1,), (1,), anchor.storage_offset())
    layer_off = torch.tensor(
        [(p - lowest) // 2 for p in ptrs], dtype=torch.int64, device=first.device
    )
    return base, layer_off, first.stride(0), first.stride(1), first.stride(2)


@torch.library.custom_op("trtllm::k3_ctx_kv", mutates_args=("ctx_len", "pool"))
def k3_ctx_kv(
    x: torch.Tensor,
    weight: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    cpos: torch.Tensor,
    num_acc: torch.Tensor,
    ctx_len: torch.Tensor,
    slots: torch.Tensor,
    rows: torch.Tensor,
    table: torch.Tensor,
    counts: torch.Tensor,
    pool: torch.Tensor,
    layer_off: torch.Tensor,
    page_stride: int,
    kv_stride: int,
    head_stride: int,
    eps: float,
    max_ctx: int,
    page: int,
    block_size: int,
    nkv: int,
) -> torch.Tensor:
    """Writes the context K/V of every drafter layer for ``x`` [N = B (K + 1), H] bf16 into the pool addressed from
    ``pool`` (bf16; :func:`pool_view`'s base) at ``layer_off`` [L] int64 element offsets (page / K-V / head strides
    in elements) and updates ``ctx_len`` [slots] int64 in place; returns ``num_ctx`` [B] int32. ``weight``:
    ``_fused_kv_weight`` [L 2 nkv 64, H] bf16; ``k_norm`` [L, 64] bf16; ``cos_sin`` [max_pos, 64] fp32 (cos | sin);
    ``cpos`` [B, K + 1] int64 RoPE positions; ``num_acc`` [B] int32; ``slots`` / ``rows`` [B] int64; ``table``
    [rows, width] int32 and ``counts`` [rows] int64 (the draft pool's block table)."""
    import cuda.bindings.driver as cuda_driver

    kern = _kernel_module()
    batch, k1 = cpos.shape
    n_tokens, k_in = x.shape
    n_rows = weight.shape[0]
    split = pick_split(n_rows, k_in, nkv, k1, n_tokens, x.device)
    if (
        split == 0
        or n_tokens != batch * k1
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or weight.shape[1] != k_in
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or k_norm.dtype != torch.bfloat16
        or k_norm.numel() * 2 * nkv != n_rows
        or cos_sin.dtype != torch.float32
        or cos_sin.shape[-1] != kern.HEAD
        or cpos.dtype != torch.int64
        or num_acc.dtype != torch.int32
        or ctx_len.dtype != torch.int64
        or slots.dtype != torch.int64
        or rows.dtype != torch.int64
        or table.dtype != torch.int32
        or counts.dtype != torch.int64
        or pool.dtype != torch.bfloat16
        or layer_off.dtype != torch.int64
        or layer_off.numel() * 2 * nkv * kern.HEAD != n_rows
    ):
        raise ValueError(
            f"k3_ctx_kv: unsupported call: x {tuple(x.shape)} {x.dtype}, weight {tuple(weight.shape)} {weight.dtype}, "
            f"nkv {nkv}, cpos {tuple(cpos.shape)} (N = B (K + 1) <= {kern.MAX_TOKENS} tokens, B <= {kern.MAX_BATCH}, "
            f"K + 1 <= {kern.CHUNK}, H % {kern.CTA_K} == 0, whole heads of 64, bf16 weights, int64 positions / "
            f"lengths / slots / rows, int32 table)"
        )  # fmt: skip
    device = x.device
    counter = _counters.get(device)
    if counter is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("trtllm::k3_ctx_kv must run once outside CUDA-graph capture first")
        counter = _counters[device] = torch.zeros(1, dtype=torch.int32, device=device)
    num_ctx = torch.empty(batch, dtype=torch.int32, device=device)
    ring = kern.pick_ring(k_in, split, n_tokens)
    args = (
        _arg(weight), _arg(x), _arg(k_norm.reshape(-1)), _arg(cos_sin.reshape(-1)), _index_arg(cpos.reshape(-1)),
        _index_arg(num_acc), _index_arg(ctx_len), _index_arg(slots), _index_arg(rows), _index_arg(table.reshape(-1)),
        _index_arg(counts), _arg(pool), _index_arg(layer_off), _arg(num_ctx), _arg(counter),
    )  # fmt: skip
    scalars = (float(eps), int(max_ctx), int(page), int(block_size), int(table.stride(0)), int(page_stride),
               int(kv_stride), int(head_stride))  # fmt: skip
    use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    consts = (n_rows, k_in, split, ring, nkv, k1, n_tokens, True)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    key = consts + (use_pdl,)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_ctx_kv must run once per shape outside CUDA-graph capture first"
            )
        import cutlass.cute as cute

        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kern.k3_ctx_kv, *args, *scalars, *consts, use_pdl, stream
                )
    fn(*args, *scalars, stream)
    return num_ctx


@k3_ctx_kv.register_fake
def _(x, weight, k_norm, cos_sin, cpos, num_acc, ctx_len, slots, rows, table, counts, pool, layer_off, page_stride,
      kv_stride, head_stride, eps, max_ctx, page, block_size, nkv):  # fmt: skip
    return ctx_len.new_empty((cpos.shape[0],), dtype=torch.int32)
