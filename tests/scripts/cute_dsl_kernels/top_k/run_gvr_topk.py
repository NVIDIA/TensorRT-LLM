# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone driver + pytest sweep for the cuTe DSL GVR Top-K kernel.

Compares the kernel output against ``torch.topk`` using tie-aware set
equality. This file exposes every knob
(T, V, ``min_blocks_per_mp``, warp-parallel reduce, both unroll switches,
``use_constant_hint``, ``compress_ratio``, ``max_seq_len`` hint) so bench
scripts can override the heuristic.

**Not in CI** — this file imports the DSL kernel module directly
(no ``trtllm`` runtime dep), to enable knob-A/B development outside the
production op.

Two usage modes:

* `python -m pytest run_gvr_topk.py`` — exhaustive parameterized correctness sweep
  (dtype × K × N × seed × next_n × T × V × warp-parallel-reduce).
* ``python run_gvr_topk.py --dtype bf16 --top_k 1024 --N 8192`` —
  single-case correctness verification on user-specified shape; knob
  overrides via ``--num_threads`` / ``--use_256bit_load`` / etc.
"""

import argparse
import functools
import sys
from pathlib import Path
from typing import Optional

import cutlass
import cutlass.cute as cute
import pytest
import torch

try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.gvr_topk_decode import GvrTopKKernel
except (ModuleNotFoundError, ImportError):
    sys.path.insert(0, str(Path(__file__).parents[4] / "tensorrt_llm/_torch/cute_dsl_kernels"))
    from blackwell.top_k.gvr_topk_decode import GvrTopKKernel  # type: ignore[no-redef]


_DTYPE_TORCH_TO_CUTE = {
    torch.float32: cutlass.Float32,
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
}


@functools.cache
def _compile(
    cute_dtype,
    top_k: int,
    next_n: int,
    enable_unroll_4: bool,
    enable_phase3_unroll: bool,
    use_constant_hint: bool,
    min_blocks_per_mp: int,
    use_256bit_load: bool,
    num_threads_per_block: int,
    enable_warp_parallel_reduce: bool,
    compress_ratio: int,
    return_output_values: bool,
    cluster_size: int = 1,
    seqlen_sorted: bool = False,
):
    """JIT-compile the GVR kernel for a specific knob combination.

    ``functools.cache`` keys on all args so repeated calls in the same
    process reuse the compiled kernel without an explicit module-level dict.
    """
    n_rows = cute.sym_int()
    n_cols = cute.sym_int()
    n_batch = cute.sym_int()
    in_align = 32 if use_256bit_load else 16
    input_fake = cute.runtime.make_fake_compact_tensor(
        cute_dtype,
        (n_rows, n_cols),
        stride_order=(1, 0),
        assumed_align=in_align,
    )
    pre_idx_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n_batch, top_k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    seq_lens_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n_batch,),
        stride_order=(0,),
    )
    # When return_output_values=False the kernel skips all STG.value
    # writes; pass None so cute.compile doesn't materialize the value
    # output placeholder.
    out_values_fake = (
        cute.runtime.make_fake_compact_tensor(
            cute_dtype,
            (n_rows, top_k),
            stride_order=(1, 0),
            assumed_align=16,
        )
        if return_output_values
        else None
    )
    out_indices_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n_rows, top_k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    # When seqlen_sorted=False the kernel never reads order_row (the
    # const_expr branch elides the indirection); pass None so cute.compile
    # doesn't materialize a placeholder buffer.
    order_row_fake = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (n_batch,),
            stride_order=(0,),
        )
        if seqlen_sorted
        else None
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = GvrTopKKernel(
        dtype=cute_dtype,
        top_k=top_k,
        next_n=next_n,
        num_threads=num_threads_per_block,
        enable_unroll_4=enable_unroll_4,
        enable_phase3_unroll=enable_phase3_unroll,
        use_constant_hint=use_constant_hint,
        min_blocks_per_mp=min_blocks_per_mp,
        use_256bit_load=use_256bit_load,
        enable_warp_parallel_reduce=enable_warp_parallel_reduce,
        compress_ratio=compress_ratio,
        return_output_values=return_output_values,
        cluster_size=cluster_size,
        seqlen_sorted=seqlen_sorted,
    )
    return cute.compile(
        kernel,
        input_fake,
        pre_idx_fake,
        seq_lens_fake,
        out_values_fake,
        out_indices_fake,
        order_row_fake,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


def gvr_topk_decode(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int = 1,
    out_values: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
    num_sms: int = 148,  # default number of sms in a B200
    enable_unroll_4: Optional[bool] = None,
    enable_phase3_unroll: Optional[bool] = None,
    use_constant_hint: bool = False,
    min_blocks_per_mp: Optional[int] = None,
    use_256bit_load: Optional[bool] = None,
    num_threads_per_block: Optional[int] = None,
    enable_warp_parallel_reduce: Optional[bool] = None,
    compress_ratio: int = 1,
    max_seq_len: Optional[int] = None,
    return_output_values: bool = False,
    cluster_size: int = 1,
    seqlen_sorted: bool = False,
    order_row: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CuTe DSL GVR Top-K wrapper with every tuning knob exposed.

    ``None``-valued knobs are resolved via the production auto-heuristic
    (same rules as ``CuteDSLGvrTopKDecodeRunner.forward``); concrete values
    override the heuristic for A/B testing.

    Args:
        logits:    ``[num_rows, max_S]`` float32 / bfloat16 / float16.
        pre_idx:   ``[num_rows // next_n, pre_idx_count]`` int32.
                   ``pre_idx[..., 0]`` must be the argmax index — indexer invariant.
        seq_lens:  ``[num_rows // next_n]`` int32 (uncompressed-token space).
        top_k:     K ∈ {512, 1024, 2048} — compile-time specialized.
        next_n:    Temporal stride for V3.2 ``preIdxOffset = (row % next_n) + 1``.
        compress_ratio: KV-indexer compression factor (1 = DSv3.2, 4 = DSv4).
                   When != 1, logits/preIdx live in compressed-token-index space:
                   ``N`` is divided by ``compress_ratio`` and ``preIdxOffset``
                   is forced to 0. Mirrors heuristicTopKDecode.cu PR #14219.
        max_seq_len: Graph-safe hint for peak ``logits.shape[1]`` at replay
                   (same compressed-token-index space as ``logits``).
        seqlen_sorted: When True, the kernel uses ``order_row`` (an LJF
                   request-level dispatch order) to resolve which row a
                   given CTA processes, so longer rows land in earlier
                   waves. Use together with :func:`gvr_topk_sort_prepare`.
                   Compatible with ``cluster_size > 1``.
        order_row: Required iff ``seqlen_sorted=True``. Request-level —
                   ``int32[batch_size = num_rows // next_n]`` on the same
                   device as ``logits``; ``order_row[i]`` is the original
                   request_id of the i-th-priority request. The kernel
                   expands to row level via
                   ``order_row[req] * next_n + nn``.

    Returns:
        ``(out_values, out_indices)`` both shaped ``[num_rows, top_k]``.
    """
    assert logits.is_cuda, "logits must be on CUDA"
    assert logits.dim() == 2, f"logits must be 2D, got {logits.shape}"
    assert pre_idx.dim() == 2 and pre_idx.dtype == torch.int32
    assert seq_lens.dim() == 1 and seq_lens.dtype == torch.int32
    if seqlen_sorted:
        # order_row is request-level (length = seq_lens.shape[0] =
        # num_rows // next_n), NOT row-level.
        assert (
            order_row is not None
            and order_row.dtype == torch.int32
            and order_row.is_cuda
            and order_row.shape == seq_lens.shape
        ), (
            "seqlen_sorted=True requires order_row: int32[batch_size] on CUDA"
            f" (expected shape {tuple(seq_lens.shape)}, got "
            f"{tuple(order_row.shape) if order_row is not None else None})"
        )

    if logits.dtype not in _DTYPE_TORCH_TO_CUTE:
        raise ValueError(f"Unsupported logits dtype: {logits.dtype}")
    cute_dtype = _DTYPE_TORCH_TO_CUTE[logits.dtype]

    num_rows = logits.shape[0]
    if return_output_values:
        if out_values is None:
            out_values = torch.empty((num_rows, top_k), dtype=logits.dtype, device=logits.device)
    if out_indices is None:
        out_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=logits.device)

    # Resolve None defaults via the same heuristic as the production Runner.
    if enable_unroll_4 is None:
        enable_unroll_4 = True
    if enable_phase3_unroll is None:
        enable_phase3_unroll = True

    N_cols = logits.shape[1]
    N_dec = max_seq_len if max_seq_len is not None else N_cols
    if num_threads_per_block is None:
        if max_seq_len is not None and logits.dtype != torch.float32:
            n_thresh_t = 131072
        else:
            n_thresh_t = 65536
        num_threads_per_block = 1024 if (num_rows <= num_sms and N_dec >= n_thresh_t) else 512
    if use_256bit_load is None:
        use_256bit_load = logits.dtype == torch.float32 and N_dec >= 16384
    if enable_warp_parallel_reduce is None:
        enable_warp_parallel_reduce = num_threads_per_block == 1024

    if min_blocks_per_mp is None:
        vec_bits_host = 256 if use_256bit_load else 128
        vec_w_host = vec_bits_host // (32 if logits.dtype == torch.float32 else 16)
        n_vec_iters = max(1, N_dec // (num_threads_per_block * vec_w_host))
        is_fp32 = logits.dtype == torch.float32
        if is_fp32:
            if n_vec_iters < 4:
                min_blocks_per_mp = 0
            elif num_rows <= num_sms:
                min_blocks_per_mp = 1
            elif num_sms * 2 < num_rows <= num_sms * 3 and N_dec <= 32768:
                # Wave-fit + latency-bound; at N>=65K mb=2 wins (bandwidth-bound).
                min_blocks_per_mp = 3
            else:
                min_blocks_per_mp = 2
        else:
            if num_rows > num_sms:
                min_blocks_per_mp = 3
            elif n_vec_iters < 4:
                min_blocks_per_mp = 0
            else:
                min_blocks_per_mp = 1

    compiled = _compile(
        cute_dtype,
        top_k,
        next_n,
        enable_unroll_4,
        enable_phase3_unroll,
        use_constant_hint,
        min_blocks_per_mp,
        use_256bit_load,
        num_threads_per_block,
        enable_warp_parallel_reduce,
        compress_ratio,
        return_output_values,
        cluster_size,
        seqlen_sorted,
    )
    # When return_output_values=False the kernel was compiled to skip
    # STG.value and accepts None for the value-output slot.
    # When seqlen_sorted=False the const_expr branch elides the order_row
    # read so the kernel accepts None for that slot as well.
    compiled(
        logits,
        pre_idx,
        seq_lens,
        out_values if return_output_values else None,
        out_indices,
        order_row if seqlen_sorted else None,
    )
    if return_output_values:
        return out_values, out_indices
    else:
        return None, out_indices


def gvr_topk_sort_prepare(seq_lens: torch.Tensor) -> torch.Tensor:
    """Build the LJF dispatch order for :func:`gvr_topk_decode`.

    Returns ``int32[num_rows]`` whose i-th entry is the original-batch
    index of the i-th longest row. Run once per decode step; the same
    ``order_row`` is reused across all per-layer ``gvr_topk_decode``
    calls with ``seqlen_sorted=True`` (seq_lens is layer-invariant
    within a decode step). For an LB-style two-bucket partition, use
    :func:`gvr_topk_lb_prepare` instead.
    """
    assert seq_lens.is_cuda and seq_lens.dim() == 1 and seq_lens.dtype == torch.int32
    return torch.argsort(seq_lens, descending=True, stable=False).to(torch.int32)


# ---- Load-Balance (Idea C) wrappers ----------------------------------------
try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.gvr_topk_decode_load_balance import (
        GvrTopKLBKernel,
        GvrTopKLBPrepareKernel,
    )
except (ModuleNotFoundError, ImportError):
    from blackwell.top_k.gvr_topk_decode_load_balance import (  # type: ignore[no-redef]
        GvrTopKLBKernel,
        GvrTopKLBPrepareKernel,
    )


@functools.cache
def _compile_lb_prepare(num_threads: int, batch_size: int, long_threshold: int):
    """JIT-compile the LB prepare kernel for a specific triple.

    Specialized over ``(num_threads, batch_size, threshold)``.

    ``num_threads`` = kernel block size + ``order_row`` length.
    ``batch_size``  = compile-time seq_lens shape (must equal runtime
                      ``seq_lens.shape[0]`` for TVM-FFI marshalling).
    """
    prep = GvrTopKLBPrepareKernel(long_threshold=long_threshold, num_threads=num_threads)
    fake_seq = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (batch_size,), stride_order=(0,)
    )
    fake_order = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_threads,), stride_order=(0,)
    )
    fake_ctr = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (2,), stride_order=(0,))
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        prep,
        fake_seq,
        fake_order,
        fake_ctr,
        cutlass.Int32(0),
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


def gvr_topk_lb_prepare(
    seq_lens: torch.Tensor,
    max_batch_size: int = 1024,
    long_threshold: int = 64 * 1024,
    order_row: Optional[torch.Tensor] = None,
    counters: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the LB prepare kernel.

    ``seq_lens`` keeps its actual shape ``(batch_size,)`` — the kernel
    is compiled to match that exact shape; ``max_batch_size`` only
    determines the prepare kernel's block size and the ``order_row``
    buffer length. Returns ``(order_row, counters)`` which the caller
    feeds into :func:`gvr_topk_lb_decode` for every per-layer Top-K
    call within the same decode step.
    """
    assert seq_lens.is_cuda and seq_lens.dtype == torch.int32
    batch_size = seq_lens.shape[0]
    if batch_size > max_batch_size:
        raise ValueError(
            f"batch_size ({batch_size}) must be <= max_batch_size "
            f"({max_batch_size}): the LB prepare kernel hard-wires a block "
            f"of max_batch_size threads and order_row[max_batch_size]; "
            f"tail requests beyond that would never be classified into the "
            f"long/short partition and order_row[batch_size:] would be -1, "
            f"so the decode path's order_row lookup would return invalid "
            f"row indices."
        )
    if order_row is None:
        order_row = torch.full((max_batch_size,), -1, dtype=torch.int32, device=seq_lens.device)
    if counters is None:
        counters = torch.zeros(2, dtype=torch.int32, device=seq_lens.device)

    compiled = _compile_lb_prepare(max_batch_size, batch_size, long_threshold)
    compiled(seq_lens, order_row, counters, cutlass.Int32(batch_size))
    return order_row, counters


@functools.cache
def _compile_lb(
    cute_dtype,
    top_k: int,
    next_n: int,
    num_rows: int,
    N: int,
    compress_ratio: int,
    max_batch_size: int,
    long_threshold: int,
    num_threads: int,
    cluster_size: int,
    return_output_values: bool,
):
    """JIT-compile the LB main kernel.

    ``num_rows`` baked in via fake tensors; ``seq_lens`` fake shape
    uses ``n_groups = num_rows // next_n`` so the caller can feed the
    actual seq_lens without padding. ``max_batch_size`` drives the
    grid (``* next_n * cluster_size`` CTAs) for CUDA Graph
    compatibility.
    """
    kernel = GvrTopKLBKernel(
        dtype=cute_dtype,
        top_k=top_k,
        next_n=next_n,
        num_threads=num_threads,
        compress_ratio=compress_ratio,
        return_output_values=return_output_values,
        cluster_size=cluster_size,
        long_threshold=long_threshold,
        max_batch_size=max_batch_size,
    )
    n_groups = num_rows // next_n
    fake_logits = cute.runtime.make_fake_compact_tensor(
        cute_dtype, (num_rows, N), stride_order=(1, 0), assumed_align=16
    )
    fake_pre_idx = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (n_groups, top_k), stride_order=(1, 0), assumed_align=16
    )
    fake_seq = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (n_groups,), stride_order=(0,))
    fake_out_v = (
        cute.runtime.make_fake_compact_tensor(
            cute_dtype, (num_rows, top_k), stride_order=(1, 0), assumed_align=16
        )
        if return_output_values
        else None
    )
    fake_out_i = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_rows, top_k), stride_order=(1, 0), assumed_align=16
    )
    fake_order = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (max_batch_size,), stride_order=(0,)
    )
    fake_ctr = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (2,), stride_order=(0,))
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        fake_logits,
        fake_pre_idx,
        fake_seq,
        fake_out_v,
        fake_out_i,
        fake_order,
        fake_ctr,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


def gvr_topk_lb_decode(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    order_row: torch.Tensor,
    counters: torch.Tensor,
    top_k: int,
    next_n: int = 1,
    compress_ratio: int = 1,
    cluster_size: int = 4,
    long_threshold: int = 64 * 1024,
    max_batch_size: int = 1024,
    num_threads: int = 512,
    return_output_values: bool = False,
    out_values: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """Run the LB (Idea C) main kernel.

    ``order_row`` and ``counters`` MUST already be populated by a prior
    call to :func:`gvr_topk_lb_prepare` for the current ``seq_lens``
    (the metadata is invariant across per-layer Top-K calls within one
    decode step, so callers run prepare once and reuse).
    """
    assert logits.is_cuda and logits.dim() == 2
    assert pre_idx.dim() == 2 and pre_idx.dtype == torch.int32
    assert seq_lens.dim() == 1 and seq_lens.dtype == torch.int32

    if logits.dtype not in _DTYPE_TORCH_TO_CUTE:
        raise ValueError(f"Unsupported logits dtype: {logits.dtype}")
    cute_dtype = _DTYPE_TORCH_TO_CUTE[logits.dtype]

    num_rows = logits.shape[0]
    N = logits.shape[1]
    if out_indices is None:
        out_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=logits.device)
    if return_output_values and out_values is None:
        out_values = torch.empty((num_rows, top_k), dtype=logits.dtype, device=logits.device)
    if not return_output_values:
        out_values = None  # passed to kernel as ``None``

    compiled = _compile_lb(
        cute_dtype,
        top_k,
        next_n,
        num_rows,
        N,
        compress_ratio,
        max_batch_size,
        long_threshold,
        num_threads,
        cluster_size,
        return_output_values,
    )
    compiled(
        logits,
        pre_idx,
        seq_lens,
        out_values,
        out_indices,
        order_row,
        counters,
    )
    return out_values, out_indices


# ---- Correctness helpers ----------------------------------------------------
def _make_inputs(
    num_rows: int,
    N: int,
    top_k: int,
    dtype: torch.dtype,
    seed: int,
    next_n: int = 1,
    compress_ratio: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (logits, pre_idx, seq_lens) for a multi-row test.

    Shapes:
      logits  : [num_rows, N]                 — compressed-token-index space
      pre_idx : [num_rows // next_n, top_k]   — argmax in slot 0 (indexer invariant)
      seq_lens: [num_rows // next_n]          — UNCOMPRESSED-token space

    Kernel divides ``seq_lens`` by ``compress_ratio`` internally. Setting
    ``seq_lens = N * cr`` makes the kernel's
    ``N_kernel = (seq_lens - next_n + ofs + 1) // cr`` match the reference
    ``N_eff = N - next_n + ofs + 1`` for next_n in {1, 2} (covers the
    current sweep). For cr=1 this reduces to ``seq_lens = N``.

    ``pre_idx.shape[1] == top_k`` per CUDA invariant (heuristic_topk.cuh:810:
    ``preIdxCount == topK`` is a dispatch precondition).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    logits_f32 = torch.randn(num_rows, N, dtype=torch.float32, device="cuda") * 2.0
    logits = logits_f32.to(dtype)
    num_groups = num_rows // next_n
    # argmax must come from the effective scan range, not full N — for
    # next_n>1 the kernel's row-0 N_eff is only (N - next_n + 1) cols.
    effective_len = N - next_n + 1
    argmax_idx = logits[::next_n, :effective_len].argmax(dim=-1).int()
    pre_idx = torch.zeros(num_groups, top_k, dtype=torch.int32, device="cuda")
    pre_idx[:, 0] = argmax_idx
    for j in range(1, top_k):
        pre_idx[:, j] = j
    # seq_lens is uncompressed; ``N * cr`` makes the kernel's
    # ``N_kernel = (seq_lens - next_n + ofs + 1) // cr`` match ref's
    # ``N_eff = N - next_n + ofs + 1`` for next_n in {1, 2}. (For cr=1
    # this reduces to seq_lens = N.)
    seq_lens_val = N * compress_ratio
    seq_lens = torch.full((num_groups,), seq_lens_val, dtype=torch.int32, device="cuda")
    return logits, pre_idx, seq_lens


def _tie_aware_correct(
    kernel_idxs: torch.Tensor,
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int,
    compress_ratio: int = 1,
) -> tuple[bool, str]:
    """Multi-row tie-aware correctness check with strict sort+allclose.

    Per row r: scan range mirrors the kernel formula (see
    ``GvrTopKKernel.gvr_topk_kernel``):

        actual_kv_len = seq_lens[r // next_n] - next_n + (r % next_n) + 1
        N_eff = actual_kv_len // compress_ratio   # cr=1 is identity

    Reference ``torch.topk`` is masked to this range so reference and
    kernel scan exactly the same columns under any (next_n, cr) combo.

    Returns ``(False, message)`` on the first failing row; ``(True, "ok")``
    when all rows pass. Sort+allclose catches the "drop-strictly-above +
    add-tied-at-kth" bug that count-below-kth alone misses on ties.
    """
    num_rows = kernel_idxs.shape[0]
    logits_f32 = logits.to(torch.float32)
    seq_lens_host = seq_lens.cpu().tolist()
    for row in range(num_rows):
        ofs = row % next_n
        actual_kv_len = int(seq_lens_host[row // next_n]) - next_n + ofs + 1
        N_eff = actual_kv_len // compress_ratio
        if N_eff < top_k:
            # Degenerate path — skip; caller's main() guards against this.
            continue
        row_logits = logits_f32[row, :N_eff]
        topk_vals, _ = torch.topk(row_logits, k=top_k, largest=True, sorted=True)
        kth_value = topk_vals[-1].item()
        sel = [int(i) for i in kernel_idxs[row].cpu().tolist() if i >= 0]
        if any(i >= N_eff for i in sel):
            return False, f"row={row}: out-of-range index"
        if len(set(sel)) != len(sel):
            return False, f"row={row}: duplicate indices"
        if len(sel) != top_k:
            return False, f"row={row}: returned {len(sel)} indices, expected {top_k}"
        sel_vals = row_logits[torch.tensor(sel, device=logits.device, dtype=torch.long)]
        n_below = int((sel_vals < kth_value).sum().item())
        if n_below > 0:
            return False, (f"row={row}: {n_below} selected values < Kth-rank ({kth_value:.6f})")
        # Strict: sorted-value multiset must match torch.topk.
        sel_sorted, _ = sel_vals.sort(descending=True)
        if not torch.allclose(sel_sorted, topk_vals, rtol=1e-5, atol=1e-5):
            max_diff = (sel_sorted - topk_vals).abs().max().item()
            return False, f"row={row}: sorted-value mismatch (max diff {max_diff:.4e})"
    return True, "ok"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize(
    "dtype,top_k",
    [
        # Production cells: (bf16, K=512/1024) and (fp32, K=2048) match
        # the deployed K -> dtype mapping. (fp16, K=1024) is added to keep
        # the fp16 convert-to-fp32 tail path under test even though it is
        # not a current production cell.
        (torch.bfloat16, 512),
        (torch.bfloat16, 1024),
        (torch.float16, 1024),
        (torch.float32, 2048),
    ],
)
@pytest.mark.parametrize("N", [4096, 65536])
@pytest.mark.parametrize("next_n", [1])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("use_256bit_load", [False, True])
@pytest.mark.parametrize("num_threads_per_block", [512, 1024])
@pytest.mark.parametrize("enable_warp_parallel_reduce", [False, True])
@pytest.mark.parametrize("cluster_size", [1, 4])
def test_gvr_topk_decode(
    dtype: torch.dtype,
    top_k: int,
    N: int,
    next_n: int,
    batch_size: int,
    use_256bit_load: bool,
    num_threads_per_block: int,
    enable_warp_parallel_reduce: bool,
    cluster_size: int,
) -> None:
    # Kernel scans `N_eff = seq_lens[0] - next_n + (row_idx % next_n) + 1`
    # columns. Smallest row's N_eff = N - next_n + 1. Degenerate path
    # (N_eff <= top_k) is a separate code branch — skip here.
    if N - next_n + 1 < top_k:
        pytest.skip("N_eff < top_k is degenerate; the kernel requires N_eff >= top_k")
    seed = 42
    num_rows = batch_size * next_n
    logits, pre_idx, seq_lens = _make_inputs(
        num_rows,
        N,
        top_k,
        dtype,
        seed,
        next_n=next_n,
        compress_ratio=1,
    )
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    _, out_idxs = gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        top_k,
        next_n=next_n,
        num_sms=num_sms,
        use_256bit_load=use_256bit_load,
        num_threads_per_block=num_threads_per_block,
        enable_warp_parallel_reduce=enable_warp_parallel_reduce,
        return_output_values=False,
        cluster_size=cluster_size,
    )
    torch.cuda.synchronize()
    ok, msg = _tie_aware_correct(out_idxs, logits, seq_lens, top_k, next_n)
    assert ok, (
        f"dtype={dtype} K={top_k} N={N} seed={seed} next_n={next_n} "
        f"batch_size={batch_size} use_256bit_load={use_256bit_load} "
        f"num_threads_per_block={num_threads_per_block} "
        f"enable_warp_parallel_reduce={enable_warp_parallel_reduce} "
        f"cluster_size={cluster_size}: {msg}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--top_k", type=int, default=1024, choices=[512, 1024, 2048])
    p.add_argument("--N", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--next_n", type=int, default=1)
    p.add_argument("--num_sms", type=int, default=148)
    p.add_argument("--compress_ratio", type=int, default=1, choices=[1, 4])
    p.add_argument("--num_threads", type=int, default=512)
    p.add_argument("--use_256bit_load", action="store_true")
    p.add_argument("--min_blocks_per_mp", type=int, default=None)
    p.add_argument("--enable_warp_parallel_reduce", action="store_true")
    p.add_argument("--disable_unroll_4", action="store_true")
    p.add_argument("--disable_phase3_unroll", action="store_true")
    p.add_argument("--use_constant_hint", action="store_true")
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument(
        "--cluster_size",
        type=int,
        default=1,
        help="CTAs per row (1=V5 single-CTA, 2/4=DSMEM cluster).",
    )
    args = p.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    effective_len = args.N - args.next_n + 1
    if effective_len < args.top_k:
        print(f"FAIL: N_eff={effective_len} < top_k={args.top_k} (degenerate path)")
        sys.exit(1)

    seed = 42
    num_rows = args.batch_size * args.next_n
    logits, pre_idx, seq_lens = _make_inputs(
        num_rows,
        args.N,
        args.top_k,
        dtype,
        seed,
        next_n=args.next_n,
        compress_ratio=args.compress_ratio,
    )
    knobs = dict(
        num_threads_per_block=args.num_threads,
        use_256bit_load=args.use_256bit_load,
        min_blocks_per_mp=args.min_blocks_per_mp,
        enable_warp_parallel_reduce=args.enable_warp_parallel_reduce,
        enable_unroll_4=not args.disable_unroll_4,
        enable_phase3_unroll=not args.disable_phase3_unroll,
        use_constant_hint=args.use_constant_hint,
        compress_ratio=args.compress_ratio,
        max_seq_len=args.max_seq_len,
        return_output_values=False,
        cluster_size=args.cluster_size,
    )
    print(
        f"config: dtype={args.dtype} top_k={args.top_k} N={args.N} "
        f"batch_size={args.batch_size} next_n={args.next_n}, num_sms={args.num_sms}"
    )
    print(f"knobs: {knobs}")

    _, out_idxs = gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        args.top_k,
        next_n=args.next_n,
        num_sms=args.num_sms,
        **knobs,
    )
    torch.cuda.synchronize()

    ok, msg = _tie_aware_correct(
        out_idxs,
        logits,
        seq_lens,
        args.top_k,
        args.next_n,
        compress_ratio=args.compress_ratio,
    )
    print(f"correctness: {'PASS' if ok else f'FAIL ({msg})'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
