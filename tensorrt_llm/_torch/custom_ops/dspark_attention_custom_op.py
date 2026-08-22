# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch custom op for fused DSpark rolling-window attention.

The op runs the tcgen05 MMA kernel
(:class:`DSparkAttention`) on the supported DSpark
geometry: 128 heads, head_dim 512, draft block 5 or 6, and a 128-row rolling
window. Other shapes fall back to the pure-PyTorch reference path in
``models/dspark/attention.py``.
"""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
import torch.nn.functional as F

try:
    from cuda.bindings import driver as cuda_driver
except ImportError:
    from cuda import cuda as cuda_driver

from ..._utils import get_sm_version
from ...logger import logger
from ..cute_dsl_kernels.blackwell.dspark.attention import DSparkAttention
from .dspark_rmsnorm_rope_custom_op import (
    _get_dspark_arch_str,
    precompile_dspark_attention_preparation,
)

_DSPARK_NUM_HEADS = 128
_DSPARK_HEAD_DIM = 512
_DSPARK_WINDOW_SIZE = 128
_DSPARK_DRAFT_BLOCK_STORAGE_SIZE = 8
_DSPARK_BLOCK_SIZES = (5, 6)
_DSPARK_ROPE_DIM = 64

# Worker initialization is process-local and single-threaded. These transient
# values let the memoized builder consume the real cache only during precompile.
_dspark_compile_arch: str | None = None
_dspark_compile_window_specimen: torch.Tensor | None = None


def _log_unsupported(reason: str, key: str) -> bool:
    logger.debug_once(
        f"Falling back from fused DSpark attention: {reason}",
        key=("fused_dspark_attention_unsupported", key),
    )
    return False


def _as_dynamic_data(t: torch.Tensor, stride_order, compact: bool = True):
    ct = cute.runtime.from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    if compact:
        ct = ct.mark_compact_shape_dynamic(mode=1, stride_order=stride_order, divisibility=8)
    return ct


def _as_dynamic_lead0(t: torch.Tensor):
    return cute.runtime.from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=0)


def _as_dynamic_1d(t: torch.Tensor):
    return cute.runtime.from_dlpack(t, assumed_align=16).mark_layout_dynamic()


def is_cute_dsl_dspark_attention_supported(
    q: torch.Tensor,
    main_kv: torch.Tensor,
    block_kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
    valid_len: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
) -> bool:
    """Return whether the supported DSpark shape can use the CuteDSL op."""
    if _get_dspark_arch_str() is None:
        return _log_unsupported(f"SM {get_sm_version()} is not SM100 or SM103", "sm_version")
    if not all(
        t.is_cuda
        for t in (
            q,
            main_kv,
            block_kv,
            kv_cache,
            slots,
            start_pos,
            valid_len,
            attn_sink,
            inverse_rope_freqs,
        )
    ):
        return _log_unsupported("all inputs must be CUDA tensors", "device")
    if q.dtype != torch.bfloat16:
        return _log_unsupported(f"q dtype must be BF16, got {q.dtype}", "q_dtype")
    if main_kv.dtype != q.dtype or block_kv.dtype != q.dtype or kv_cache.dtype != q.dtype:
        return _log_unsupported(
            "main_kv, block_kv, and kv_cache dtypes must match q; "
            f"got {main_kv.dtype}, {block_kv.dtype}, and {kv_cache.dtype}",
            "kv_dtype",
        )
    if attn_sink.dtype != torch.float32 or inverse_rope_freqs.dtype != torch.float32:
        return _log_unsupported(
            "attn_sink and inverse_rope_freqs must be FP32; "
            f"got {attn_sink.dtype} and {inverse_rope_freqs.dtype}",
            "aux_dtype",
        )
    if slots.dtype not in (torch.int32, torch.int64) or start_pos.dtype not in (
        torch.int32,
        torch.int64,
    ):
        return _log_unsupported(
            f"slots and start_pos must use INT32 or INT64, got {slots.dtype} and {start_pos.dtype}",
            "index_dtype",
        )
    if valid_len.dtype != torch.int64:
        return _log_unsupported(
            f"valid_len must use INT64, got {valid_len.dtype}", "valid_len_dtype"
        )
    if q.ndim != 4 or main_kv.ndim != 2 or block_kv.ndim != 3 or kv_cache.ndim != 3:
        return _log_unsupported(
            "expected q/main_kv/block_kv/kv_cache ranks 4/2/3/3, "
            f"got {q.ndim}/{main_kv.ndim}/{block_kv.ndim}/{kv_cache.ndim}",
            "tensor_ranks",
        )
    if q.shape[1] not in _DSPARK_BLOCK_SIZES:
        return _log_unsupported(f"draft block size must be 5 or 6, got {q.shape[1]}", "block_size")
    if q.shape[2:] != (_DSPARK_NUM_HEADS, _DSPARK_HEAD_DIM):
        return _log_unsupported(
            "q must have 128 heads and head_dim 512; "
            f"got {q.shape[2]} heads and head_dim {q.shape[3]}",
            "q_shape",
        )
    expected_main_kv_shape = (q.shape[0], _DSPARK_HEAD_DIM)
    if main_kv.shape != expected_main_kv_shape:
        return _log_unsupported(
            f"main_kv shape must be {expected_main_kv_shape}, got {tuple(main_kv.shape)}",
            "main_kv_shape",
        )
    expected_block_kv_shape = (q.shape[0], q.shape[1], _DSPARK_HEAD_DIM)
    if block_kv.shape != expected_block_kv_shape:
        return _log_unsupported(
            f"block_kv shape must be {expected_block_kv_shape}, got {tuple(block_kv.shape)}",
            "block_kv_shape",
        )
    if kv_cache.shape[1:] != (_DSPARK_WINDOW_SIZE, _DSPARK_HEAD_DIM):
        return _log_unsupported(
            f"kv_cache must have trailing shape (128, 512); got {tuple(kv_cache.shape[1:])}",
            "kv_cache_shape",
        )
    if kv_cache.stride(1) != _DSPARK_HEAD_DIM or kv_cache.stride(2) != 1:
        return _log_unsupported(
            f"kv_cache trailing strides must be (512, 1), got {kv_cache.stride()[1:]}",
            "kv_cache_layout",
        )
    expected_sink_shape = (_DSPARK_NUM_HEADS,)
    if attn_sink.shape != expected_sink_shape:
        return _log_unsupported(
            f"attn_sink shape must be {expected_sink_shape}, got {tuple(attn_sink.shape)}",
            "attn_sink_shape",
        )
    expected_batch_shape = (q.shape[0],)
    if (
        slots.shape != expected_batch_shape
        or start_pos.shape != expected_batch_shape
        or valid_len.shape != expected_batch_shape
    ):
        return _log_unsupported(
            "slots, start_pos, and valid_len shapes must match the q batch size; "
            f"expected {expected_batch_shape}, got {tuple(slots.shape)}, "
            f"{tuple(start_pos.shape)}, and {tuple(valid_len.shape)}",
            "index_shape",
        )
    expected_freqs_shape = (q.shape[0], q.shape[1], _DSPARK_ROPE_DIM // 2, 2)
    if inverse_rope_freqs.shape != expected_freqs_shape:
        return _log_unsupported(
            "inverse_rope_freqs shape must be "
            f"{expected_freqs_shape}, got {tuple(inverse_rope_freqs.shape)}",
            "inverse_rope_freqs_shape",
        )
    if not q.is_contiguous():
        return _log_unsupported("q must be contiguous", "q_layout")
    if not main_kv.is_contiguous():
        return _log_unsupported("main_kv must be contiguous", "main_kv_layout")
    if not block_kv.is_contiguous():
        return _log_unsupported("block_kv must be contiguous", "block_kv_layout")
    if not slots.is_contiguous():
        return _log_unsupported("slots must be contiguous", "slots_layout")
    if not start_pos.is_contiguous():
        return _log_unsupported("start_pos must be contiguous", "start_pos_layout")
    if not valid_len.is_contiguous():
        return _log_unsupported("valid_len must be contiguous", "valid_len_layout")
    if not attn_sink.is_contiguous():
        return _log_unsupported("attn_sink must be contiguous", "attn_sink_layout")
    if not inverse_rope_freqs.is_contiguous():
        return _log_unsupported(
            "inverse_rope_freqs must be contiguous", "inverse_rope_freqs_layout"
        )
    return True


def is_cute_dsl_dspark_attention_prepared_supported(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    valid_len: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
) -> bool:
    """Return whether prepared tensors can use the supported attention kernel."""
    if _get_dspark_arch_str() is None:
        return False
    if not all(t.is_cuda for t in (q, kv_cache, valid_len, attn_sink, inverse_rope_freqs)):
        return False
    if q.dtype != torch.bfloat16 or kv_cache.dtype != q.dtype:
        return False
    if valid_len.dtype != torch.int64:
        return False
    if attn_sink.dtype != torch.float32 or inverse_rope_freqs.dtype != torch.float32:
        return False
    if q.ndim != 4 or kv_cache.ndim != 3:
        return False
    return (
        q.shape[1] in _DSPARK_BLOCK_SIZES
        and q.shape[2:] == (_DSPARK_NUM_HEADS, _DSPARK_HEAD_DIM)
        and kv_cache.shape[1:] == (_DSPARK_WINDOW_SIZE, _DSPARK_HEAD_DIM)
        and kv_cache.stride(1) == _DSPARK_HEAD_DIM
        and kv_cache.stride(2) == 1
        and valid_len.shape == (q.shape[0],)
        and attn_sink.shape == (_DSPARK_NUM_HEADS,)
        and inverse_rope_freqs.shape == (q.shape[0], q.shape[1], _DSPARK_ROPE_DIM // 2, 2)
        and q.is_contiguous()
        and valid_len.is_contiguous()
        and attn_sink.is_contiguous()
        and inverse_rope_freqs.is_contiguous()
    )


def precompile_dspark_attention(
    block_size: int,
    num_heads: int,
    kv_cache: torch.Tensor,
    softmax_scale: float,
    eps: float = 1e-6,
) -> None:
    """Compile one dynamic-batch kernel up front (one-time, host-side).

    The specimen tensor marks below carry batch extents as runtime MLIR values,
    so one compiled object covers every supported runtime batch and cache-pool layout.
    Runtime calls reuse the compiled object without padding. This helper owns
    the full support policy and silently no-ops when runtime dispatch would not
    route to this kernel anyway.
    """
    global _dspark_compile_arch, _dspark_compile_window_specimen
    arch_str = _get_dspark_arch_str()
    if not (
        arch_str is not None
        and block_size in _DSPARK_BLOCK_SIZES
        and num_heads == _DSPARK_NUM_HEADS
        and kv_cache.is_cuda
        and kv_cache.dtype == torch.bfloat16
        and kv_cache.ndim == 3
        and kv_cache.shape[1:] == (_DSPARK_WINDOW_SIZE, _DSPARK_HEAD_DIM)
        and kv_cache.stride(1) == _DSPARK_HEAD_DIM
        and kv_cache.stride(2) == 1
    ):
        logger.debug(
            "DSpark attention precompile skipped: geometry or architecture "
            "outside the CuteDSL kernel contract"
        )
        return
    if _dspark_compile_arch is not None:
        raise RuntimeError("DSpark attention precompile is already in progress")

    _dspark_compile_arch = arch_str
    _dspark_compile_window_specimen = kv_cache
    try:
        precompile_dspark_attention_preparation(block_size, eps)
        _compile_dspark_attention(block_size, softmax_scale)
    finally:
        _dspark_compile_arch = None
        _dspark_compile_window_specimen = None


@functools.cache
def _compile_dspark_attention(
    block_size: int,
    softmax_scale: float,
):
    arch_str = _dspark_compile_arch
    window_specimen = _dspark_compile_window_specimen
    if arch_str is None or window_specimen is None:
        raise RuntimeError(
            "DSpark attention kernel was not precompiled during worker initialization: "
            f"block={block_size}"
        )

    # Use inexpensive B=1 query/output specimens plus the worker-owned window.
    # Dynamic marks carry extents and strides as runtime MLIR values, so no
    # second window allocation is needed. The scheduler round-trips
    # problem_shape_b and its fast-divmod divisor as dynamic fields, so the
    # resulting object serves every runtime batch. Real specimens preserve the
    # scheduler's dynamic fields without requiring an all-symbolic fake tensor.
    num_heads, head_dim = _DSPARK_NUM_HEADS, _DSPARK_HEAD_DIM
    specimen_batch = 1
    device = torch.device("cuda")

    # The kernel's native layouts are [H, D, Sq, B] for Q/O and [page, D,
    # physical_page] for both KV streams. The real tensors are inexpensive
    # permuted views of TRT-LLM's row-major buffers; the specimen tensors
    # below replicate those layouts so the compiled strides match runtime.
    q_spec = _as_dynamic_data(
        torch.empty(
            (specimen_batch, block_size, num_heads, head_dim), dtype=torch.bfloat16, device=device
        ).permute(2, 3, 1, 0),
        (3, 2, 0, 1),
    )
    # The worker window pool is a strided stage view (gaps between pages),
    # so it cannot be marked compact; leave its strides fully dynamic.
    window_spec = _as_dynamic_data(
        window_specimen.permute(1, 2, 0),
        None,
        compact=False,
    )
    block_spec = _as_dynamic_data(
        torch.empty(
            (specimen_batch, _DSPARK_DRAFT_BLOCK_STORAGE_SIZE, head_dim),
            dtype=torch.bfloat16,
            device=device,
        ).permute(1, 2, 0),
        (2, 0, 1),
    )
    page_table_spec = _as_dynamic_lead0(
        torch.empty((specimen_batch, 1), dtype=torch.int32, device=device).permute(1, 0)
    )
    output_spec = _as_dynamic_data(
        torch.empty(
            (specimen_batch, block_size, num_heads, head_dim), dtype=torch.bfloat16, device=device
        ).permute(2, 3, 1, 0),
        (3, 2, 0, 1),
    )
    cache_seqs_spec = _as_dynamic_1d(
        torch.empty((specimen_batch,), dtype=torch.int32, device=device)
    )
    valid_len_spec = _as_dynamic_1d(
        torch.empty((specimen_batch,), dtype=torch.int64, device=device)
    )
    sink_spec = _as_dynamic_1d(torch.empty((num_heads,), dtype=torch.float32, device=device))
    freqs_spec = _as_dynamic_1d(
        torch.empty(
            (specimen_batch * block_size * _DSPARK_ROPE_DIM,), dtype=torch.float32, device=device
        )
    )
    stream_arg = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    hardware_info = cutlass_utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(2)
    kernel = DSparkAttention(
        cutlass.Float32,
        (128, 128),
        (128, 256),
        max_active_clusters,
        _DSPARK_DRAFT_BLOCK_STORAGE_SIZE,
        _DSPARK_WINDOW_SIZE,
        0.0,
        seq_len_q=block_size,
        mma_qk_tiler_k=128,
        inverse_rope_dim=_DSPARK_ROPE_DIM,
        arch_str=arch_str,
    )
    compiled = cute.compile(
        kernel,
        q_spec,
        window_spec,
        block_spec,
        page_table_spec,
        output_spec,
        cache_seqs_spec,
        valid_len_spec,
        softmax_scale,
        1.0,
        sink_spec,
        freqs_spec,
        stream_arg,
        options="--opt-level 2",
    )
    logger.info(
        "DSpark Attention enabled: implementation=dspark_attn, "
        f"block={block_size}, heads={num_heads}, head_dim={head_dim}"
    )
    return compiled


def _run_dspark_attention(
    q: torch.Tensor,
    draft_block: torch.Tensor,
    kv_cache: torch.Tensor,
    slots_i32: torch.Tensor,
    cache_seqs: torch.Tensor,
    valid_len: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    block_size = q.shape[1]
    compiled = _compile_dspark_attention(block_size, softmax_scale)
    page_table_win = _as_dynamic_lead0(slots_i32.unsqueeze(1).permute(1, 0))
    output = torch.empty_like(q)

    stream_arg = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        _as_dynamic_data(q.permute(2, 3, 1, 0), (3, 2, 0, 1)),
        _as_dynamic_data(kv_cache.permute(1, 2, 0), None, compact=False),
        _as_dynamic_data(draft_block.permute(1, 2, 0), (2, 0, 1)),
        page_table_win,
        _as_dynamic_data(output.permute(2, 3, 1, 0), (3, 2, 0, 1)),
        _as_dynamic_1d(cache_seqs),
        _as_dynamic_1d(valid_len),
        softmax_scale,
        1.0,
        _as_dynamic_1d(attn_sink),
        _as_dynamic_1d(inverse_rope_freqs.reshape(-1)),
        stream_arg,
    )
    return output


@torch.library.custom_op(
    "trtllm::cute_dsl_dspark_attention",
    mutates_args=("kv_cache",),
    device_types="cuda",
)
def cute_dsl_dspark_attention(
    q: torch.Tensor,
    main_kv: torch.Tensor,
    block_kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
    valid_len: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run the direct/test compatibility path from already-normalized K/V.

    ``slots`` values must be in ``[0, kv_cache.shape[0])``, and ``start_pos``
    values must be nonnegative. The support check does not inspect tensor values,
    so callers are responsible for enforcing these preconditions.
    """
    if not is_cute_dsl_dspark_attention_supported(
        q,
        main_kv,
        block_kv,
        kv_cache,
        slots,
        start_pos,
        valid_len,
        attn_sink,
        inverse_rope_freqs,
    ):
        raise ValueError(
            "cute_dsl_dspark_attention requires contiguous BF16 supported DSpark tensors "
            "([B, 5|6, 128, 512] with a 128-row window) on a supported "
            "SM100 or SM103 GPU; "
            f"got SM {get_sm_version()}"
        )
    block_size = q.shape[1]

    # Compatibility path for direct callers with already-normalized K/V. The
    # model path uses the prepared op below and does not execute these auxiliary
    # kernels.
    _compile_dspark_attention(block_size, softmax_scale)
    kv_cache[slots, start_pos % _DSPARK_WINDOW_SIZE] = main_kv
    draft_block = F.pad(block_kv, (0, 0, 0, _DSPARK_DRAFT_BLOCK_STORAGE_SIZE - block_size))
    slots_i32 = slots.to(torch.int32)
    cache_seqs = start_pos.to(torch.int32)
    return _run_dspark_attention(
        q,
        draft_block,
        kv_cache,
        slots_i32,
        cache_seqs,
        valid_len,
        attn_sink,
        inverse_rope_freqs,
        softmax_scale,
    )


@torch.library.register_fake("trtllm::cute_dsl_dspark_attention")
def _(
    q: torch.Tensor,
    main_kv: torch.Tensor,
    block_kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
    valid_len: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "trtllm::cute_dsl_dspark_attention_prepared",
    mutates_args=(),
    device_types="cuda",
)
def cute_dsl_dspark_attention_prepared(
    q: torch.Tensor,
    draft_block: torch.Tensor,
    kv_cache: torch.Tensor,
    slots_i32: torch.Tensor,
    cache_seqs: torch.Tensor,
    valid_len: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run prepared attention after the caller validates the fused contract."""

    return _run_dspark_attention(
        q,
        draft_block,
        kv_cache,
        slots_i32,
        cache_seqs,
        valid_len,
        attn_sink,
        inverse_rope_freqs,
        softmax_scale,
    )


@torch.library.register_fake("trtllm::cute_dsl_dspark_attention_prepared")
def _(
    q: torch.Tensor,
    draft_block: torch.Tensor,
    kv_cache: torch.Tensor,
    slots_i32: torch.Tensor,
    cache_seqs: torch.Tensor,
    valid_len: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    del draft_block, kv_cache, slots_i32, cache_seqs, valid_len, attn_sink, inverse_rope_freqs
    del softmax_scale
    return torch.empty_like(q)
