# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch custom op for fused DSpark rolling-window attention.

The op runs the specialized tcgen05 MMA kernel
(:class:`DSparkAttention`) on the production DSpark
geometry: 128 heads, head_dim 512, draft block 5 or 6, and a 128-row rolling
window. Other shapes fall back to the pure-PyTorch reference path in
``models/dspark/attention.py``.
"""

import functools
from typing import List, Optional, Tuple

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
from ...math_utils import ceil_div, pad_up
from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DistributedTuningStrategy,
    DynamicTensorSpec,
    TunableRunner,
    TuningConfig,
)
from ..cute_dsl_kernels.blackwell.dspark.attention import DSparkAttention
from ..cute_dsl_utils import IS_CUTLASS_DSL_INTERNAL_AVAILABLE
from ..utils import fp8_scale_infer_shape, mxfp8_scale_infer_shape, next_positive_power_of_2
from .dspark_rmsnorm_rope_custom_op import (
    _get_dspark_arch_str,
    precompile_dspark_attention_preparation,
)

_DSPARK_NUM_HEADS = 128
_DSPARK_HEAD_DIM = 512
_DSPARK_WINDOW_SIZE = 128
_DSPARK_PAGE_SIZE = 8
_DSPARK_BLOCK_SIZES = (5, 6)
_DSPARK_ROPE_DIM = 64

_DSPARK_CONTEXT_WKV_MIN_M_BUCKET = 128
_DSPARK_GENERATION_M_SPEC = DynamicTensorSpec(0, 0)
# These are request-capacity buckets for tactic selection. The Rubin GEMM
# artifact has runtime-dynamic M/N/K, so this does not specialize a kernel for
# each batch value.
_DSPARK_GENERATION_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@functools.cache
def _get_dspark_generation_m_spec(
    rows_per_request: int,
    max_batch: int,
    input_idx: int = 0,
    dim_idx: int = 0,
) -> DynamicTensorSpec:
    """Build the DSpark operator-owned generation-M bucket contract."""
    if rows_per_request < 1 or max_batch < 1:
        raise ValueError(
            "DSpark generation tuning requires positive rows_per_request and max_batch, "
            f"got {rows_per_request=} and {max_batch=}."
        )

    batch_buckets = tuple(
        batch for batch in _DSPARK_GENERATION_BATCH_BUCKETS if batch < max_batch
    ) + (max_batch,)
    m_buckets = tuple(batch * rows_per_request for batch in batch_buckets)

    def map_to_bucket(m: int) -> int:
        batch = ceil_div(m, rows_per_request)
        for bucket in batch_buckets:
            if batch <= bucket:
                return bucket * rows_per_request
        # A shape beyond the configured worker capacity is not expected, but
        # keeping it exact gives the autotuner its normal cache-miss fallback.
        return m

    return DynamicTensorSpec(
        input_idx,
        dim_idx,
        m_buckets,
        map_to_bucket,
    )


@functools.cache
def _get_dspark_generation_mxfp8_tuning_config(
    rows_per_request: int,
    max_batch: int,
) -> TuningConfig:
    """Return generation buckets for a DSpark MXFP8 projection."""
    return TuningConfig(
        dynamic_tensor_specs=(_get_dspark_generation_m_spec(rows_per_request, max_batch),),
        constraint_specs=(ConstraintSpec(2, 0, mxfp8_scale_infer_shape),),
        use_cold_l2_cache=True,
        distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
        use_cuda_graph=True,
    )


def _map_dspark_context_wkv_m_profile(m: int) -> int:
    """Bucket CTX seed and batched backfill rows without changing kernel shapes."""
    return max(_DSPARK_CONTEXT_WKV_MIN_M_BUCKET, next_positive_power_of_2(m))


def _gen_dspark_context_wkv_m_profiles(max_m: int) -> Tuple[int, ...]:
    max_bucket = _map_dspark_context_wkv_m_profile(max_m)
    buckets = []
    bucket = _DSPARK_CONTEXT_WKV_MIN_M_BUCKET
    while bucket <= max_bucket:
        buckets.append(bucket)
        bucket *= 2
    return tuple(buckets)


# Worker initialization is process-local and single-threaded. These transient
# values let the memoized builder consume the real cache only during precompile.
_dspark_compile_arch: str | None = None
_dspark_compile_window_specimen: torch.Tensor | None = None


def _as_dynamic_data(t: torch.Tensor, stride_order, compact: bool = True):
    ct = cute.runtime.from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    if compact:
        ct = ct.mark_compact_shape_dynamic(mode=1, stride_order=stride_order, divisibility=8)
    return ct


def _as_dynamic_lead0(t: torch.Tensor):
    return cute.runtime.from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=0)


def _as_dynamic_lead2(t: torch.Tensor):
    return cute.runtime.from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=2)


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
    """Return whether the production DSpark shape can use the CuteDSL op."""
    if _get_dspark_arch_str() is None:
        return False
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
        return False
    if q.dtype != torch.bfloat16:
        return False
    if main_kv.dtype != q.dtype or block_kv.dtype != q.dtype or kv_cache.dtype != q.dtype:
        return False
    if attn_sink.dtype != torch.float32 or inverse_rope_freqs.dtype != torch.float32:
        return False
    if slots.dtype not in (torch.int32, torch.int64) or start_pos.dtype not in (
        torch.int32,
        torch.int64,
    ):
        return False
    if valid_len.dtype != torch.int64:
        return False
    if q.ndim != 4 or main_kv.ndim != 2 or block_kv.ndim != 3 or kv_cache.ndim != 3:
        return False
    return (
        q.shape[1] in _DSPARK_BLOCK_SIZES
        and q.shape[2] == _DSPARK_NUM_HEADS
        and q.shape[3] == _DSPARK_HEAD_DIM
        and main_kv.shape == (q.shape[0], _DSPARK_HEAD_DIM)
        and block_kv.shape == (q.shape[0], q.shape[1], _DSPARK_HEAD_DIM)
        and kv_cache.shape[1:] == (_DSPARK_WINDOW_SIZE, _DSPARK_HEAD_DIM)
        and kv_cache.stride(1) == _DSPARK_HEAD_DIM
        and kv_cache.stride(2) == 1
        and attn_sink.shape == (_DSPARK_NUM_HEADS,)
        and slots.shape == (q.shape[0],)
        and start_pos.shape == (q.shape[0],)
        and valid_len.shape == (q.shape[0],)
        and inverse_rope_freqs.shape == (q.shape[0], q.shape[1], _DSPARK_ROPE_DIM // 2, 2)
        and inverse_rope_freqs.is_contiguous()
        and q.is_contiguous()
        and main_kv.is_contiguous()
        and block_kv.is_contiguous()
        and slots.is_contiguous()
        and start_pos.is_contiguous()
        and valid_len.is_contiguous()
        and attn_sink.is_contiguous()
    )


def is_cute_dsl_dspark_attention_prepared_supported(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    valid_len: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
) -> bool:
    """Return whether prepared tensors can use the production attention kernel."""
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
    output_groups: int | None = None,
) -> None:
    """Compile one dynamic-batch kernel up front (one-time, host-side).

    The specimen tensor marks below carry batch extents as runtime MLIR values,
    so one compiled object covers every supported runtime batch and cache-pool layout.
    Runtime never compiles or pads: every call is a hot cache hit populated
    here. This helper owns the full support policy and silently no-ops when
    runtime dispatch would not route to this kernel anyway.
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
        and (
            output_groups is None
            or (arch_str == "sm_107" and output_groups > 0 and num_heads % output_groups == 0)
        )
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
        # Keep the BF16 compatibility variant available when a runtime input
        # misses the preparation predicate; serving must never JIT that fallback.
        _compile_dspark_attention(block_size, softmax_scale, None)
        if output_groups is not None:
            _compile_dspark_attention(block_size, softmax_scale, output_groups)
    finally:
        _dspark_compile_arch = None
        _dspark_compile_window_specimen = None


@functools.cache
def _compile_dspark_attention(
    block_size: int,
    softmax_scale: float,
    output_groups: int | None = None,
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
    # resulting object serves every runtime batch. Keeping real specimens also
    # preserves the optimized scheduler lowering that regressed with the older
    # all-symbolic fake-tensor path.
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
    # The production window pool is a strided stage view (gaps between pages),
    # so it cannot be marked compact; leave its strides fully dynamic.
    window_spec = _as_dynamic_data(
        window_specimen.permute(1, 2, 0),
        None,
        compact=False,
    )
    block_spec = _as_dynamic_data(
        torch.empty(
            (specimen_batch, _DSPARK_PAGE_SIZE, head_dim), dtype=torch.bfloat16, device=device
        ).permute(1, 2, 0),
        (2, 0, 1),
    )
    page_table_spec = _as_dynamic_lead0(
        torch.empty((specimen_batch, 1), dtype=torch.int32, device=device).permute(1, 0)
    )
    if output_groups is None:
        output_spec = _as_dynamic_data(
            torch.empty(
                (specimen_batch, block_size, num_heads, head_dim),
                dtype=torch.bfloat16,
                device=device,
            ).permute(2, 3, 1, 0),
            (3, 2, 0, 1),
        )
        output_scale_spec = None
    else:
        rows = specimen_batch * block_size
        group_dim = num_heads * head_dim // output_groups
        output_spec = _as_dynamic_lead2(
            torch.empty(
                (output_groups, rows, group_dim),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
        )
        output_scale_spec = _as_dynamic_lead2(
            torch.empty(
                (output_groups, group_dim // 128, (rows + 3) // 4 * 4),
                dtype=torch.float32,
                device=device,
            )
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
        _DSPARK_PAGE_SIZE,
        _DSPARK_WINDOW_SIZE,
        0.0,
        seq_len_q=block_size,
        mma_qk_tiler_k=128,
        inverse_rope_dim=_DSPARK_ROPE_DIM,
        output_groups=output_groups,
        arch_str=arch_str,
    )
    compiled = cute.compile(
        kernel,
        q_spec,
        window_spec,
        block_spec,
        page_table_spec,
        output_spec,
        output_scale_spec,
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
        f"block={block_size}, heads={num_heads}, head_dim={head_dim}, "
        f"output_groups={output_groups}"
    )
    return compiled


def _run_dspark_attention(
    q: torch.Tensor,
    block_page: torch.Tensor,
    kv_cache: torch.Tensor,
    slots_i32: torch.Tensor,
    cache_seqs: torch.Tensor,
    valid_len: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
    output_groups: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    block_size = q.shape[1]
    compiled = _compile_dspark_attention(block_size, softmax_scale, output_groups)
    page_table_win = _as_dynamic_lead0(slots_i32.unsqueeze(1).permute(1, 0))
    if output_groups is None:
        output = torch.empty_like(q)
        output_native = _as_dynamic_data(output.permute(2, 3, 1, 0), (3, 2, 0, 1))
        output_scale = None
        output_scale_native = None
    else:
        rows = q.shape[0] * block_size
        group_dim = q.shape[2] * q.shape[3] // output_groups
        output = torch.empty(
            (output_groups, rows, group_dim),
            dtype=torch.float8_e4m3fn,
            device=q.device,
        )
        output_scale = torch.empty(
            (output_groups, group_dim // 128, (rows + 3) // 4 * 4),
            dtype=torch.float32,
            device=q.device,
        )
        output_native = _as_dynamic_lead2(output)
        output_scale_native = _as_dynamic_lead2(output_scale)

    stream_arg = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        _as_dynamic_data(q.permute(2, 3, 1, 0), (3, 2, 0, 1)),
        _as_dynamic_data(kv_cache.permute(1, 2, 0), None, compact=False),
        _as_dynamic_data(block_page.permute(1, 2, 0), (2, 0, 1)),
        page_table_win,
        output_native,
        output_scale_native,
        _as_dynamic_1d(cache_seqs),
        _as_dynamic_1d(valid_len),
        softmax_scale,
        1.0,
        _as_dynamic_1d(attn_sink),
        _as_dynamic_1d(inverse_rope_freqs.reshape(-1)),
        stream_arg,
    )
    if output_scale is not None:
        return output, output_scale
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
    """Run the direct/test compatibility path from already-normalized K/V."""
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
            "cute_dsl_dspark_attention requires contiguous BF16 production DSpark tensors "
            "([B, 5|6, 128, 512] with a 128-row window) on a supported "
            "SM100, SM103, or SM107 GPU; "
            f"got SM {get_sm_version()}"
        )
    block_size = q.shape[1]

    # Compatibility path for direct callers with already-normalized K/V. The
    # production model uses the prepared op below and does not execute these
    # auxiliary kernels.
    _compile_dspark_attention(block_size, softmax_scale, None)
    kv_cache[slots, start_pos % _DSPARK_WINDOW_SIZE] = main_kv
    block_page = F.pad(block_kv, (0, 0, 0, _DSPARK_PAGE_SIZE - block_size))
    slots_i32 = slots.to(torch.int32)
    cache_seqs = start_pos.to(torch.int32)
    return _run_dspark_attention(
        q,
        block_page,
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
    block_page: torch.Tensor,
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
        block_page,
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
    block_page: torch.Tensor,
    kv_cache: torch.Tensor,
    slots_i32: torch.Tensor,
    cache_seqs: torch.Tensor,
    valid_len: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    del block_page, kv_cache, slots_i32, cache_seqs, valid_len, attn_sink, inverse_rope_freqs
    del softmax_scale
    return torch.empty_like(q)


@torch.library.custom_op(
    "trtllm::cute_dsl_dspark_attention_prepared_fp8",
    mutates_args=(),
    device_types="cuda",
)
def cute_dsl_dspark_attention_prepared_fp8(
    q: torch.Tensor,
    block_page: torch.Tensor,
    kv_cache: torch.Tensor,
    slots_i32: torch.Tensor,
    cache_seqs: torch.Tensor,
    valid_len: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
    output_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run prepared attention and emit grouped FP8 OA inputs and scales."""

    output = _run_dspark_attention(
        q,
        block_page,
        kv_cache,
        slots_i32,
        cache_seqs,
        valid_len,
        attn_sink,
        inverse_rope_freqs,
        softmax_scale,
        output_groups,
    )
    assert isinstance(output, tuple)
    return output


@torch.library.register_fake("trtllm::cute_dsl_dspark_attention_prepared_fp8")
def _(
    q: torch.Tensor,
    block_page: torch.Tensor,
    kv_cache: torch.Tensor,
    slots_i32: torch.Tensor,
    cache_seqs: torch.Tensor,
    valid_len: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
    output_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del block_page, kv_cache, slots_i32, cache_seqs, valid_len, attn_sink, inverse_rope_freqs
    del softmax_scale
    rows = q.shape[0] * q.shape[1]
    group_dim = q.shape[2] * q.shape[3] // output_groups
    output = torch.empty(
        (output_groups, rows, group_dim),
        dtype=torch.float8_e4m3fn,
        device=q.device,
    )
    output_scale = torch.empty(
        (output_groups, group_dim // 128, (rows + 3) // 4 * 4),
        dtype=torch.float32,
        device=q.device,
    )
    return output, output_scale


if IS_CUTLASS_DSL_INTERNAL_AVAILABLE:
    from .cute_dsl_custom_ops import CuteDSLFp8RubinBmmQuantRunner, CuteDSLMXFP8RubinLinear

    @functools.cache
    def _get_dspark_generation_fp8_bmm_quant_tuning_config(
        rows_per_request: int,
        max_batch: int,
    ) -> TuningConfig:
        # One-engine DSpark runs target verification (K+1 rows/request)
        # before the K-row draft. Tune the shared OA BMM for both real
        # generation families when the draft first publishes its capacity.
        draft_spec = _get_dspark_generation_m_spec(
            rows_per_request, max_batch, input_idx=0, dim_idx=1
        )
        target_spec = _get_dspark_generation_m_spec(
            rows_per_request + 1, max_batch, input_idx=0, dim_idx=1
        )
        m_buckets = tuple(
            sorted(set(draft_spec.gen_tuning_buckets + target_spec.gen_tuning_buckets))
        )

        def map_to_bucket(m: int) -> int:
            for bucket in m_buckets:
                if m <= bucket:
                    return bucket
            return m

        return TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    input_idx=0,
                    dim_idx=1,
                    gen_tuning_buckets=m_buckets,
                    map_to_tuning_buckets=map_to_bucket,
                ),
            ),
            constraint_specs=(
                ConstraintSpec(2, 2, fp8_scale_infer_shape),
                ConstraintSpec(4, 1, lambda input_shapes: input_shapes[0][1]),
                ConstraintSpec(
                    5,
                    0,
                    lambda input_shapes: pad_up(input_shapes[0][1], 128)
                    * pad_up(
                        ceil_div(input_shapes[0][0] * input_shapes[1][1], 32),
                        4,
                    ),
                ),
            ),
            distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
        )

    class CuteDSLDSparkFp8RubinBmmQuantRunner(CuteDSLFp8RubinBmmQuantRunner):
        """DSpark OA runner with operator-owned M buckets and launch plans."""

        tuning_config = TuningConfig(
            dynamic_tensor_specs=(DynamicTensorSpec(0, 1),),
            constraint_specs=(
                ConstraintSpec(2, 2, fp8_scale_infer_shape),
                ConstraintSpec(4, 1, lambda input_shapes: input_shapes[0][1]),
                ConstraintSpec(
                    5,
                    0,
                    lambda input_shapes: pad_up(input_shapes[0][1], 128)
                    * pad_up(
                        ceil_div(input_shapes[0][0] * input_shapes[1][1], 32),
                        4,
                    ),
                ),
            ),
            distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
        )

        def _forward_tvm_ffi_prepared(
            self,
            inputs: List[torch.Tensor],
            compiled_gemm,
        ) -> None:
            if self.fp8_quantize_1x128:
                (
                    a_tensor,
                    b_tensor,
                    a_sf_tensor,
                    b_sf_tensor,
                    c_tensor,
                    scale_tensor,
                ) = inputs
            else:
                a_tensor, b_tensor, a_sf_tensor, b_sf_tensor, c_tensor = inputs
                scale_tensor = None

            batch_size, m, k = a_tensor.shape
            n = b_tensor.shape[1]
            runtime_args = (
                m,
                n,
                k,
                pad_up(m, 4),
                ceil_div(n, 128),
                ceil_div(k, 128),
                batch_size,
                a_tensor.data_ptr(),
                b_tensor.data_ptr(),
                a_sf_tensor.data_ptr(),
                b_sf_tensor.data_ptr(),
                c_tensor.permute(1, 2, 0),
            )
            if self.fp8_quantize_1x128:
                compiled_gemm(*runtime_args, scale_tensor)
            else:
                compiled_gemm(*runtime_args)

        def __init__(self, use_tvm_ffi: bool = True):
            super().__init__(use_tvm_ffi=use_tvm_ffi)
            self._tvm_ffi_launch_plans = {}

        def unique_id(self):
            return (self.output_dtype, self.use_tvm_ffi)

    @functools.cache
    def _get_cute_dsl_dspark_fp8_rubin_bmm_quant_runner(
        use_tvm_ffi: bool,
    ) -> CuteDSLDSparkFp8RubinBmmQuantRunner:
        return CuteDSLDSparkFp8RubinBmmQuantRunner(use_tvm_ffi=use_tvm_ffi)

    class CuteDSLDSparkMXFP8RubinLinear(CuteDSLMXFP8RubinLinear):
        """DSpark MXFP8 runner with process-local tuning and launch plans."""

        tuning_config = TuningConfig(
            # DSpark warmup runs the real generation shapes. Keep those M
            # values exact so runtime lookup uses the tactic measured for
            # the captured graph instead of remapping it to a power of two.
            dynamic_tensor_specs=(_DSPARK_GENERATION_M_SPEC,),
            constraint_specs=(ConstraintSpec(2, 0, mxfp8_scale_infer_shape),),
            use_cold_l2_cache=True,
            distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
            use_cuda_graph=True,
        )

        def _forward_tvm_ffi_prepared(
            self,
            inputs: List[torch.Tensor],
            launch_plan,
            bias: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> torch.Tensor:
            compiled_gemm, swap_ab = launch_plan
            if len(inputs) == 5:
                a_tensor, b_tensor, a_sf_tensor, b_sf_tensor, alpha_tensor = inputs
                output_tensor = None
            else:
                (
                    a_tensor,
                    b_tensor,
                    a_sf_tensor,
                    b_sf_tensor,
                    alpha_tensor,
                    output_tensor,
                ) = inputs

            m, k = a_tensor.shape
            n = b_tensor.shape[0]
            partition_id = kwargs.get("partition_id", -1)
            if output_tensor is not None:
                c_tensor = output_tensor[:, partition_id * n : (partition_id + 1) * n]
            elif self.to_userbuffers:
                c_tensor = torch.ops.trtllm.create_userbuffers_tensor([m, n], self.output_dtype)
            else:
                c_tensor = torch.empty(
                    (m, n),
                    dtype=self.output_dtype,
                    device=a_tensor.device,
                )

            if swap_ab:
                c_tensor = c_tensor.permute(1, 0)

            real_k = k * self.packed_k_multiplier
            sf_m = pad_up(m, 128)
            sf_k = pad_up(real_k // self.sf_vec_size, 4)
            sf_n = pad_up(n, 128)
            if swap_ab:
                kernel_m = n
                kernel_n = m
                kernel_sf_m = sf_n
                kernel_sf_n = sf_m
                kernel_a_tensor = b_tensor
                kernel_a_sf_tensor = b_sf_tensor
                kernel_b_tensor = a_tensor
                kernel_b_sf_tensor = a_sf_tensor
            else:
                kernel_m = m
                kernel_n = n
                kernel_sf_m = sf_m
                kernel_sf_n = sf_n
                kernel_a_tensor = a_tensor
                kernel_a_sf_tensor = a_sf_tensor
                kernel_b_tensor = b_tensor
                kernel_b_sf_tensor = b_sf_tensor

            c_ld = output_tensor.shape[1] if output_tensor is not None else 0
            compiled_gemm(
                kernel_m,
                kernel_n,
                real_k,
                kernel_sf_m // 128,
                kernel_sf_n // 128,
                sf_k // 4,
                kernel_a_tensor.data_ptr(),
                kernel_b_tensor.data_ptr(),
                kernel_a_sf_tensor.data_ptr(),
                kernel_b_sf_tensor.data_ptr(),
                c_tensor.data_ptr(),
                alpha_tensor,
                c_ld,
            )

            if swap_ab:
                c_tensor = c_tensor.permute(1, 0)
            if output_tensor is None:
                if bias is not None:
                    if bias.ndim != 1 or bias.shape[0] != c_tensor.shape[-1]:
                        raise ValueError(
                            f"bias must be a 1-D tensor of shape [N]={c_tensor.shape[-1]}, "
                            f"got shape {tuple(bias.shape)}"
                        )
                    c_tensor = c_tensor + bias
                return c_tensor
            return

        def __init__(
            self,
            output_dtype: torch.dtype,
            to_userbuffers: bool = False,
            use_tvm_ffi: bool = True,
        ):
            super().__init__(
                output_dtype=output_dtype,
                to_userbuffers=to_userbuffers,
                use_tvm_ffi=use_tvm_ffi,
            )
            self._tvm_ffi_launch_plans = {}

    @functools.cache
    def _get_cute_dsl_mxfp8_rubin_alpha(device: torch.device) -> torch.Tensor:
        """Keep the scalar-one GEMM input alive for each CUDA device."""
        return torch.ones((), dtype=torch.float32, device=device)

    @functools.cache
    def _get_cute_dsl_dspark_mxfp8_rubin_runner(
        output_dtype: torch.dtype,
        use_tvm_ffi: bool,
    ) -> CuteDSLDSparkMXFP8RubinLinear:
        return CuteDSLDSparkMXFP8RubinLinear(
            output_dtype=output_dtype,
            use_tvm_ffi=use_tvm_ffi,
        )

    _mxfp8_prevalidated_launch_plans = {}

    def _materialize_dspark_tuning_profile_winners(
        custom_op: str,
        runner: TunableRunner,
        tuning_config: TuningConfig,
        inputs: List[torch.Tensor],
    ) -> None:
        """Compile every cached DSpark bucket winner in this process."""
        tuner = AutoTuner.get()
        if not tuner.is_tuning_mode:
            return

        profiles = tuner._optimization_profiles(tuning_config, inputs)
        key = (
            custom_op,
            inputs[0].device,
            runner.unique_id(),
            tuple(profile.get_opt_shapes() for profile in profiles),
        )
        materialized = getattr(runner, "_dspark_materialized_profiles", None)
        if materialized is None:
            materialized = set()
            runner._dspark_materialized_profiles = materialized
        if key in materialized:
            return

        complete = True
        for profile in profiles:
            found, _, tactic, _ = tuner.profiling_cache.search_cache(
                custom_op,
                [runner],
                profile.get_opt_shapes(),
                tuning_config,
                apply_map_to_tuning_buckets=False,
            )
            if not found:
                complete = False
                continue
            profile_inputs = tuner._prepare_input_tensors(profile, inputs)
            runner(profile_inputs, tactic=tactic)
        if complete:
            materialized.add(key)

    def _run_cute_dsl_dspark_mxfp8_gemm_rubin_prevalidated(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        use_tvm_ffi: bool = True,
        materialize_launch: bool = False,
        tuning_config: Optional[TuningConfig] = None,
    ) -> torch.Tensor:
        """Run DSpark MXFP8 after bind-time contract validation."""
        alpha = _get_cute_dsl_mxfp8_rubin_alpha(input.device)
        runner = _get_cute_dsl_dspark_mxfp8_rubin_runner(output_dtype, use_tvm_ffi)
        inputs = [input, weight, input_scale, weight_scale, alpha]
        tuner = AutoTuner.get()
        materialized_key = None
        if materialize_launch and use_tvm_ffi and not tuner.is_tuning_mode:
            m, k = input.shape
            n = weight.shape[0]
            materialized_key = (
                input.device,
                output_dtype,
                m,
                n,
                k,
                input_scale.numel(),
                weight_scale.numel(),
            )
            launch_plan = _mxfp8_prevalidated_launch_plans.get(materialized_key)
            if launch_plan is not None:
                return runner._forward_tvm_ffi_prepared(inputs, launch_plan)

        active_tuning_config = tuning_config or runner.__class__.tuning_config
        custom_op = "trtllm::dspark_cute_dsl_mxfp8_gemm_rubin"
        _, best_tactic = tuner.choose_one(
            custom_op,
            [runner],
            active_tuning_config,
            inputs,
        )
        output = runner(inputs, tactic=best_tactic)
        if tuning_config is not None and tuner.is_tuning_mode:
            _materialize_dspark_tuning_profile_winners(
                custom_op, runner, active_tuning_config, inputs
            )
        if materialized_key is not None:
            launch_plan_key = (
                best_tactic,
                m,
                n,
                k,
                input_scale.numel(),
                weight_scale.numel(),
                0,
                -1,
            )
            launch_plan = runner._tvm_ffi_launch_plans.get(launch_plan_key)
            if launch_plan is not None:
                _mxfp8_prevalidated_launch_plans[materialized_key] = launch_plan
        return output

    class CuteDSLDSparkContextWkvRubinLinear(CuteDSLDSparkMXFP8RubinLinear):
        """MXFP8 runner with operator-owned buckets for DSpark CTX M."""

        tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    0,
                    0,
                    _gen_dspark_context_wkv_m_profiles,
                    _map_dspark_context_wkv_m_profile,
                ),
            ),
            constraint_specs=(ConstraintSpec(2, 0, mxfp8_scale_infer_shape),),
            use_cold_l2_cache=True,
            distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
            use_cuda_graph=True,
        )

    @functools.cache
    def _get_cute_dsl_dspark_context_wkv_rubin_runner(
        output_dtype: torch.dtype,
        use_tvm_ffi: bool,
    ) -> CuteDSLDSparkContextWkvRubinLinear:
        return CuteDSLDSparkContextWkvRubinLinear(
            output_dtype=output_dtype,
            use_tvm_ffi=use_tvm_ffi,
        )

    @torch.library.custom_op(
        "trtllm::cute_dsl_dspark_fp8_bmm_quantize_rubin_out",
        mutates_args=("output", "packed_scale"),
        device_types="cuda",
    )
    def cute_dsl_dspark_fp8_bmm_quantize_rubin_out(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor,
        packed_scale: torch.Tensor,
        use_tvm_ffi: bool = True,
        dspark_max_batch: int = 0,
        dspark_rows_per_request: int = 0,
    ) -> None:
        if get_sm_version() != 107:
            raise ValueError("CuTe DSL FP8 BMM quantization requires SM107.")
        batch_size, m, _ = input.shape
        n = weight.shape[1]
        if n % 128 != 0:
            raise ValueError(f"Fused FP8 BMM quantization requires N divisible by 128, got {n}.")
        expected_scale_numel = pad_up(m, 128) * pad_up(ceil_div(batch_size * n, 32), 4)
        if output.shape != (batch_size, m, n):
            raise ValueError(
                "Fused FP8 BMM output must have shape "
                f"{(batch_size, m, n)}, got {tuple(output.shape)}."
            )
        if (
            output.dtype != torch.float8_e4m3fn
            or packed_scale.dtype != torch.uint8
            or not packed_scale.is_contiguous()
            or packed_scale.shape != (expected_scale_numel,)
        ):
            raise ValueError(
                "Fused FP8 BMM requires E4M3 output and a flat contiguous "
                f"uint8 R128c4 scale tensor with {expected_scale_numel} elements."
            )

        runner = _get_cute_dsl_dspark_fp8_rubin_bmm_quant_runner(use_tvm_ffi)
        inputs = [
            input,
            weight,
            input_scale,
            weight_scale,
            output,
            packed_scale,
        ]
        if dspark_max_batch > 0 and dspark_rows_per_request > 0:
            tuning_config = _get_dspark_generation_fp8_bmm_quant_tuning_config(
                dspark_rows_per_request,
                dspark_max_batch,
            )
            runner._dspark_generation_tuning_config = tuning_config
        else:
            tuning_config = getattr(
                runner, "_dspark_generation_tuning_config", runner.__class__.tuning_config
            )
        custom_op = "trtllm::cute_dsl_dspark_fp8_bmm_quantize_rubin::gemm"
        tuner = AutoTuner.get()
        _, best_tactic = tuner.choose_one(
            custom_op,
            [runner],
            tuning_config,
            inputs,
        )
        runner(inputs, tactic=best_tactic)
        if tuning_config is not runner.__class__.tuning_config and tuner.is_tuning_mode:
            _materialize_dspark_tuning_profile_winners(
                custom_op,
                runner,
                tuning_config,
                inputs,
            )

    @torch.library.register_fake("trtllm::cute_dsl_dspark_fp8_bmm_quantize_rubin_out")
    def _(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor,
        packed_scale: torch.Tensor,
        use_tvm_ffi: bool = True,
        dspark_max_batch: int = 0,
        dspark_rows_per_request: int = 0,
    ) -> None:
        del input_scale, weight_scale, use_tvm_ffi
        del dspark_max_batch, dspark_rows_per_request
        batch_size, m, _ = input.shape
        n = weight.shape[1]
        assert output.dtype == torch.float8_e4m3fn
        assert output.shape == (batch_size, m, n)
        expected_scale_numel = pad_up(m, 128) * pad_up(ceil_div(batch_size * n, 32), 4)
        assert packed_scale.dtype == torch.uint8
        assert packed_scale.shape == (expected_scale_numel,)
        assert packed_scale.is_contiguous()

    @torch.library.custom_op(
        "trtllm::cute_dsl_dspark_context_wkv_gemm_rubin",
        mutates_args=(),
        device_types="cuda",
    )
    def cute_dsl_dspark_context_wkv_gemm_rubin(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        use_tvm_ffi: bool = True,
    ) -> torch.Tensor:
        """Run stacked DSpark CTX WKV with operator-owned M buckets."""
        if output_dtype != torch.bfloat16:
            raise ValueError(
                f"DSpark context WKV GEMM only supports bfloat16 output, got {output_dtype}"
            )
        if (sm_version := get_sm_version()) != 107:
            raise ValueError(f"DSpark context WKV GEMM requires SM107, got SM{sm_version}.")
        alpha = _get_cute_dsl_mxfp8_rubin_alpha(input.device)
        runner = _get_cute_dsl_dspark_context_wkv_rubin_runner(output_dtype, use_tvm_ffi)
        inputs = [input, weight, input_scale, weight_scale, alpha]
        _, best_tactic = AutoTuner.get().choose_one(
            "trtllm::cute_dsl_dspark_context_wkv_gemm_rubin",
            [runner],
            runner.__class__.tuning_config,
            inputs,
        )
        return runner(inputs, tactic=best_tactic)

    @torch.library.register_fake("trtllm::cute_dsl_dspark_context_wkv_gemm_rubin")
    def _(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        use_tvm_ffi: bool = True,
    ) -> torch.Tensor:
        shape = list(mat_a.shape)
        shape[-1] = mat_b.shape[-2]
        return mat_a.new_empty(shape, dtype=torch.bfloat16)
