# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TAIL_FORK: the DSpark drafter's acceptance-independent input work, forked off the verify's critical path.

At one decode step the drafter's context rows are projected from the captured target layers (DFlash
``prepare_1st_drafter_inputs``): the five layer captures (one graph memcpy node each, in the target forward, where each
breaks the PDL chain into the next layer's QKV GEMV), the fc (FC_SHARD: this rank's column shard + the all-gather),
the hidden norm, then the context K / V GEMM and the k_norm of DRAFTER_CTX_KV_STORE_FUSED's store
(``ops/dspark/dspark_ctx_store``).
None of it reads the step's sampled tokens or accepted counts: only the store kernel at its end does (the accept mask,
the RoPE positions, the paged columns). In stock order it all runs after the verify sampling, on the critical path.

Here (``M3DSparkWorker``, dspark.py):

- ``TAIL_DEFER_CAPTURE``: a capture layer's boundary residual is kept by reference instead of being copied into the
  capture buffer (the boundary all-reduce allocates a fresh residual and nothing writes it later), so no memcpy node
  sits between the boundary and the next layer;
- ``TAIL_FORK``: right after the verify head GEMM, an aux stream copies the kept residuals into the capture buffer
  (the same bytes, the same columns) and runs the same fc + hidden norm on the same rows; ``TAIL_FORK_KV`` also runs the
  store's K / V GEMM, contiguous K, rms_norm and weight multiply there (``ctx_store_prep``), and the store wrapper
  then launches only the store kernel (``ctx_store_launch``) on the joined result. The main stream joins the aux
  stream where the stock path would first read the projected rows (``project_target_hidden``).

Every op is the stock op on the stock operands (same shapes, dtypes, layouts and pinned algos), only on another
stream and earlier, so the outputs are bit for bit the stock ones. ``ctx_store_prep`` / ``ctx_store_launch`` are the two
halves of ``fused_ctx_store``, line for line.
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


class StoreOperands:
    """One ``ctx_store_launch`` call's store kernel launch: its grid, its operands (named in the kernel's argument
    order) and its compile-time switches. TAILGLUE_A (``dspark_tailfold``) launches the same operands inside its
    one-launch post-join glue."""

    __slots__ = (
        "grid",
        "kv_flat",
        "k",
        "w",
        "cos_sin",
        "positions",
        "num_accepted",
        "request_rows",
        "cols",
        "counts",
        "max_col",
        "block_table",
        "cache",
        "kv_stride_row",
        "table_stride",
        "cache_stride_layer",
        "meta",
    )

    def args(self) -> tuple:
        return (
            self.kv_flat,
            self.k,
            self.w,
            self.cos_sin,
            self.positions,
            self.num_accepted,
            self.request_rows,
            self.cols,
            self.counts,
            self.max_col,
            self.block_table,
            self.cache,
            self.kv_stride_row,
            self.table_stride,
            self.cache_stride_layer,
        )

    def launch(self) -> None:
        """The catalog store on these operands (``fused_ctx_store``'s launch): the C++ op, which takes the row and
        table strides and the rotary dim from the tensors themselves."""
        m = self.meta
        family_ops().minimax_m31_dspark_ctx_store(
            self.kv_flat,
            self.k,
            self.w,
            self.cos_sin,
            self.positions,
            self.num_accepted,
            self.request_rows,
            self.cols,
            self.counts,
            self.max_col,
            self.block_table,
            self.cache,
            self.cache_stride_layer,
            m["KP1"],
            m["L"],
            m["NKV"],
            m["HD"],
            m["PS"],
            m["PTR_LIST"],
            m["COL_FROM_POS"],
            m["BOUND"],
            m["FOLD_W"],
        )


def ctx_store_operands(
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
) -> Union[StoreOperands, bool, None]:
    """``ctx_store_launch``'s decision without the launch: its declines as it returns them (False, None), else the
    ``StoreOperands`` its kernel launch takes."""
    if prep is None or prep.projected is not projected:
        return None
    if not _store.ctx_store_supported(draft_model, kv_buf):
        return False
    if cols is None and (
        (capacity_counts is None) == (max_col is None)
        or (capacity_counts is not None and capacity_counts.dtype != torch.int64)
    ):
        return False
    # fused_ctx_store's declines for the C++ op's index operands, so the op never rejects what this path launches.
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
    n = kv_flat.shape[0]
    num_layers, nkv, hd = (
        draft_model._num_attn_layers,
        draft_model._num_kv_heads,
        draft_model._head_dim,
    )
    first = kv_buf if isinstance(kv_buf, torch.Tensor) else kv_buf[0]
    cos_sin = draft_model._get_cos_sin_cache()
    ops = StoreOperands()
    ops.grid = (n, num_layers, nkv)
    ops.kv_flat = kv_flat
    ops.k = k
    ops.w = w
    ops.cos_sin = cos_sin
    # The C++ store reads cols and counts through raw pointers, and both fall back to positions.
    positions = positions.contiguous()
    ops.positions = positions
    ops.num_accepted = num_accepted.contiguous()
    ops.request_rows = request_rows.contiguous()
    ops.cols = positions if cols is None else cols.contiguous()
    ops.counts = capacity_counts if capacity_counts is not None else positions
    ops.max_col = int(max_col) if max_col is not None else 0
    ops.block_table = block_table
    ops.cache = cache_arg
    ops.kv_stride_row = kv_flat.stride(0)
    ops.table_stride = block_table.stride(0)
    ops.cache_stride_layer = cache_stride_layer
    ops.meta = dict(
        KP1=tokens_per_request,
        L=num_layers,
        NKV=nkv,
        HD=hd,
        ROT=cos_sin.shape[1],
        PS=first.shape[-2],
        PTR_LIST=ptr_list,
        COL_FROM_POS=cols is None,
        BOUND=capacity_counts is not None,
        FOLD_W=fold,
    )
    return ops


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
    ops = ctx_store_operands(
        prep,
        draft_model,
        projected,
        positions,
        num_accepted,
        request_rows,
        cols,
        block_table,
        kv_buf,
        tokens_per_request,
        capacity_counts=capacity_counts,
        max_col=max_col,
    )
    if not isinstance(ops, StoreOperands):
        return ops
    ops.launch()
    return True


def view_key(t: torch.Tensor) -> Tuple[int, Tuple[int, ...], Tuple[int, ...], torch.dtype]:
    """Identity of a tensor view: (data pointer, shape, strides, dtype)."""
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)


def copy_captures(buf: torch.Tensor, rows: int, hidden: int, layers: int, kept) -> bool:
    """Write the kept captures ``kept`` = [(capture index, [rows, hidden] tensor), ...] into ``buf``'s columns, the
    bytes DFlash's maybe_capture_hidden_states copies. When every layer is kept and ``buf[:rows]`` is the whole
    contiguous [rows, layers x hidden] block, one concatenation writes all of it; otherwise one slice copy per layer
    (the stock op). Returns whether the single concatenation ran."""
    if (
        len(kept) == layers
        and sorted(i for i, _ in kept) == list(range(layers))
        and buf.shape[1] == layers * hidden
        and buf[:rows].is_contiguous()
        and all(t.shape == (rows, hidden) and t.dtype == buf.dtype for _, t in kept)
    ):
        parts = [t for _, t in sorted(kept, key=lambda e: e[0])]
        torch.cat(parts, dim=1, out=buf[:rows])
        return True
    from tensorrt_llm._torch.custom_ops import inplace_slice_copy

    for i, t in kept:
        inplace_slice_copy(buf, t, i * hidden, (i + 1) * hidden)
    return False
