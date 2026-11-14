# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TAIL_FORK: the DSpark drafter's acceptance-independent input work, forked off the verify's
critical path. This target's copy of the sm_107 TP4 target's ``dspark_tailfork``.

At one decode step the drafter's context rows are projected from the captured target layers (DFlash
``prepare_1st_drafter_inputs``): the layer captures (on the 16-row step the MC finalize boundary residuals that
TAIL_DEFER_CAPTURE keeps by reference, one concatenation into the capture buffer), the replicated fc (a 1056-CTA
split-K GEMM at 16 rows, 30.5 us on SM107) + the hidden norm, then the context K / V GEMM and the k_norm of
DRAFTER_CTX_KV_STORE_FUSED's store (``ops/dspark/dspark_ctx_store``). None of it reads the step's sampled tokens or
accepted counts: only the store kernel at its end does (the accept mask, the RoPE positions, the paged columns). In
stock order it all runs after the verify sampling, on the critical path (TP2 b2: about 57 us of chain B at 16 rows).

Here (``M3DSparkWorker``, dspark.py), right after the verify head GEMM, an aux stream writes the kept captures into
the capture buffer (the same bytes, the same columns), runs the same fc + hidden norm on the same rows (this target's
replicated fc; FC_SHARD16's column shard + multicast gather when opted in) and, under ``TAIL_FORK_KV``, the store's
K / V GEMM, contiguous K, rms_norm and weight multiply (``ctx_store_prep``); the store wrapper then launches only the
store kernel (``ctx_store_launch``) on the joined result. ``TAIL_FORK_REFRESH`` also runs
DRAFTER_BLOCK_TABLE_REFRESH_FUSED's block-table refresh there. The main stream joins the aux stream where the stock path
would first read the projected rows (``project_target_hidden``).

Every op is the stock op on the stock operands (same shapes, dtypes and layouts), only on another stream and earlier,
so the outputs are bit for bit the stock ones as long as cuBLAS picks the same algorithm for the fc and the K / V GEMM
on the aux stream (its reproducibility guarantee is stated per single active stream; a captured graph freezes the
pick): the in-model output identity is the evidence, and a fork arm that alone loses it calls for pinning those
algorithms. ``ctx_store_prep`` / ``ctx_store_launch`` are the two halves of ``fused_ctx_store``, line for line (the TP4
target's functions, verbatim).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_ctx_store as _store,
)


class CtxStorePrep:
    """The acceptance-independent half of ``fused_ctx_store`` for one ``projected`` tensor."""

    __slots__ = ("projected", "kv_flat", "k", "w", "fold")

    def __init__(self, projected, kv_flat, k, w, fold) -> None:
        self.projected = projected
        self.kv_flat = kv_flat
        self.k = k
        self.w = w
        self.fold = fold


def ctx_store_prep(draft_model, projected: torch.Tensor) -> Optional[CtxStorePrep]:
    """``fused_ctx_store``'s GEMM, contiguous K, rms_norm and (unless folded) weight multiply on ``projected`` (the
    tensor the store will be handed), or None when the drafter's fused K / V weight is not built yet (the store then
    runs whole). The lines are fused_ctx_store's, in its order."""
    weight = getattr(draft_model, "_fused_kv_weight", None)
    if weight is None or getattr(draft_model, "_k_norm_stacked", None) is None:
        return None
    src = projected
    # precompute_context_kv's GEMM, unchanged.
    if getattr(draft_model, "_input_ln_eps", None) is not None:
        ph = projected.float()
        ph = ph * torch.rsqrt(ph.pow(2).mean(-1, keepdim=True) + draft_model._input_ln_eps)
        projected = ph.to(weight.dtype)
    elif projected.dtype != weight.dtype:
        projected = projected.to(weight.dtype)
    kv_flat = F.linear(projected, weight, draft_model._fused_kv_bias)
    n = kv_flat.shape[0]
    if n == 0:
        return None
    num_layers, nkv, hd = (
        draft_model._num_attn_layers,
        draft_model._num_kv_heads,
        draft_model._head_dim,
    )
    # precompute_context_kv's k_norm, unchanged: contiguous K, F.rms_norm, the stacked weight multiply.
    k = kv_flat.view(n, num_layers, 2, nkv, hd)[:, :, 0].contiguous()
    k = F.rms_norm(k, (hd,), eps=draft_model._k_norm_eps)
    w = draft_model._k_norm_stacked
    fold = (
        _store._fold_w[0]
        and k.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and w.is_contiguous()
        and w.numel() == num_layers * hd
        and w.device == k.device
    )
    if not fold:
        k = k * w.view(1, num_layers, 1, hd)
    return CtxStorePrep(src, kv_flat, k, w, fold)


def ctx_store_launch(
    prep: CtxStorePrep,
    draft_model,
    projected: torch.Tensor,
    positions: torch.Tensor,
    num_accepted: torch.Tensor,
    request_rows: torch.Tensor,
    cols: Optional[torch.Tensor],
    block_table: torch.Tensor,
    kv_buf: Union[torch.Tensor, Sequence[torch.Tensor]],
    tokens_per_request: int,
    *,
    capacity_counts: Optional[torch.Tensor] = None,
    max_col: Optional[int] = None,
) -> Optional[bool]:
    """The rest of ``fused_ctx_store`` on ``prep``: its declines (False, nothing written: the stock path runs), then the
    store kernel (True). None when ``prep`` is not for ``projected`` or the fused K / V weight changed since (the caller
    then runs ``fused_ctx_store`` whole)."""
    if prep is None or prep.projected is not projected:
        return None
    if not _store.ctx_store_supported(draft_model, kv_buf):
        return False
    if cols is None and (
        (capacity_counts is None) == (max_col is None)
        or (capacity_counts is not None and capacity_counts.dtype != torch.int64)
    ):
        return False
    # fused_ctx_store's declines for the C++ op's index operands (as at TP4), so the op never rejects what this
    # path launches.
    if (
        positions.dtype not in _store._INT
        or num_accepted.dtype not in _store._INT
        or request_rows.dtype not in _store._INT
        or (cols is not None and cols.dtype not in _store._INT)
        or block_table.dtype not in _store._INT
        or block_table.dim() != 2
        or (block_table.stride(1) != 1 and block_table.shape[1] != 1)
    ):
        return False
    cache = _store._cache_arg(kv_buf)
    if cache is None:
        return False
    cache_arg, ptr_list, cache_stride_layer = cache
    kv_flat, k, w, fold = prep.kv_flat, prep.k, prep.w, prep.fold
    if (
        draft_model._fused_kv_weight is None
        or w is not draft_model._k_norm_stacked
        or kv_flat.shape[-1] != draft_model._fused_kv_weight.shape[0]
    ):
        return None
    num_layers, nkv, hd = (
        draft_model._num_attn_layers,
        draft_model._num_kv_heads,
        draft_model._head_dim,
    )
    first = kv_buf if isinstance(kv_buf, torch.Tensor) else kv_buf[0]
    cos_sin = draft_model._get_cos_sin_cache()
    # The store is DRAFTER_CTX_KV_STORE_FUSED's C++ op (no Triton _ctx_store_kernel here): the same operands
    # as fused_ctx_store's launch; it takes the row / table strides and the rotary dim from the tensors. It reads cols
    # and counts through raw pointers, and both fall back to positions, so positions is made contiguous first.
    positions = positions.contiguous()
    family_ops().minimax_m31_dspark_ctx_store(
        kv_flat,
        k,
        w,
        cos_sin,
        positions,
        num_accepted.contiguous(),
        request_rows.contiguous(),
        positions if cols is None else cols.contiguous(),
        capacity_counts if capacity_counts is not None else positions,
        int(max_col) if max_col is not None else 0,
        block_table,
        cache_arg,
        cache_stride_layer,
        tokens_per_request,
        num_layers,
        nkv,
        hd,
        first.shape[-2],
        ptr_list,
        cols is None,
        capacity_counts is not None,
        fold,
    )
    return True


def view_key(t: torch.Tensor) -> Tuple[int, Tuple[int, ...], Tuple[int, ...], torch.dtype]:
    """Identity of a tensor view: (data pointer, shape, strides, dtype)."""
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)
