# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paged DSpark attention and Rubin FP8 projection operators."""

from collections.abc import Callable
from dataclasses import replace
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass.cute.typing import Numeric, Pointer, Type

from ..._utils import get_sm_version
from ...logger import logger
from ..attention.kernels.blackwell.dspark.attention import DSparkAttention
from ..autotuner import AutoTuner, DynamicTensorSpec
from ..cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from ..kernels.blackwell.utils import make_ptr
from ..utils import next_positive_power_of_2
from .dspark_rmsnorm_rope_custom_op import (
    _DSV4_DSPARK_BLOCK_SIZES,
    _DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE,
    _DSV4_DSPARK_HEAD_DIM,
    _DSV4_DSPARK_ROPE_DIM,
    _DSV4_DSPARK_WINDOW_SIZE,
    _get_dspark_arch_str,
    cute_dsl_dspark_rmsnorm_rope_cache_write,
    cute_dsl_dspark_rmsnorm_rope_draft_block,
)

_DSPARK_CONTEXT_WKV_MIN_M_BUCKET = 128


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


# Attention-only limits; the shared DSpark shape contract (head and RoPE dims, window,
# draft block sizes and storage) comes from dspark_rmsnorm_rope_custom_op.
_DSV4_DSPARK_SUPPORTED_NUM_HEADS = (64, 128)
_DSV4_DSPARK_PAGE_SIZES = (16, 32, 64, 128, 256)


_dspark_attention_kernel_cache: dict[tuple[int, int, int, str], Callable[..., None]] = {}


def _make_compile_gmem_pointer(dtype: Type[Numeric], assumed_align: int) -> Pointer:
    """Create a compile-only pointer specimen without allocating device memory."""
    return make_ptr(
        dtype,
        0,
        cute.AddressSpace.gmem,
        assumed_align=assumed_align,
    )


def _compile_dspark_attention(
    block_size: int,
    arch_str: str,
    page_size: int,
    num_heads: int,
) -> Callable[..., None]:
    """Compile the pointer host wrapper without runtime tensor specimens."""
    head_dim = _DSV4_DSPARK_HEAD_DIM
    batch = cute.sym_int()
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (batch, block_size, num_heads, head_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Real tensor specimens used to make PyTorch's primary CUDA context current
    # implicitly. Fake-only compilation must do so before querying occupancy;
    # the returned stream is not part of the compiled TVM-FFI launch ABI.
    torch.cuda.current_stream()
    hardware_info = cutlass_utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(2)
    kernel = DSparkAttention(
        cutlass.Float32,
        (128, 128),
        (128, 256),
        max_active_clusters,
        _DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE,
        _DSV4_DSPARK_WINDOW_SIZE,
        0.0,
        seq_len_q=block_size,
        mma_qk_tiler_k=128,
        inverse_rope_dim=_DSV4_DSPARK_ROPE_DIM,
        arch_str=arch_str,
        history_page_size=page_size,
        num_heads=num_heads,
    )
    # Scalars and typed, aligned compile-only pointers define the runtime ABI.
    # ``output_fake`` keeps one tensor argument for TVM-FFI environment-stream
    # detection without tying compilation to a real worker buffer.
    compiled = cute.compile(
        kernel.wrapper,
        1,
        1,
        page_size * head_dim,
        1,
        1,
        _make_compile_gmem_pointer(cutlass.BFloat16, 16),
        _make_compile_gmem_pointer(cutlass.BFloat16, 16),
        _make_compile_gmem_pointer(cutlass.BFloat16, 16),
        _make_compile_gmem_pointer(cutlass.Int32, 4),
        _make_compile_gmem_pointer(cutlass.Int64, 8),
        _make_compile_gmem_pointer(cutlass.Int64, 8),
        _make_compile_gmem_pointer(cutlass.Int64, 8),
        _make_compile_gmem_pointer(cutlass.Float32, 4),
        _make_compile_gmem_pointer(cutlass.Float32, 4),
        output_fake,
        cutlass.Float32(1.0),
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )
    logger.info(
        "DSpark Attention enabled: implementation=dspark_attn, "
        f"block={block_size}, heads={num_heads}, head_dim={head_dim}"
    )
    return compiled


@torch.library.custom_op(
    "trtllm::fused_dsv4_dspark_attention", mutates_args=(), device_types="cuda"
)
def fused_dsv4_dspark_attention(
    q: torch.Tensor,
    draft_block: torch.Tensor,
    kv_pages: torch.Tensor,
    block_tables: torch.Tensor,
    positions: torch.Tensor,
    valid_lengths: torch.Tensor,
    capacities: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Attend to a 128-token context window and fuse the inverse-RoPE epilogue."""
    arch_str = _get_dspark_arch_str()
    if arch_str is None:
        raise RuntimeError("Embedded DSpark attention requires SM100, SM103, or SM107")
    batch, block, heads, dim = q.shape
    if block not in _DSV4_DSPARK_BLOCK_SIZES:
        raise ValueError(
            f"DSpark block size must be one of {_DSV4_DSPARK_BLOCK_SIZES}, got {block}"
        )
    if heads not in _DSV4_DSPARK_SUPPORTED_NUM_HEADS:
        raise ValueError(
            f"DSpark num_heads must be one of {_DSV4_DSPARK_SUPPORTED_NUM_HEADS}, got {heads}"
        )
    if dim != _DSV4_DSPARK_HEAD_DIM:
        raise ValueError(f"DSpark head_dim must be {_DSV4_DSPARK_HEAD_DIM}, got {dim}")
    if kv_pages.ndim != 3 or kv_pages.shape[2] != dim:
        raise ValueError(
            f"DSpark pages must have shape [pages, page_size, {dim}], got {tuple(kv_pages.shape)}"
        )
    page_size = kv_pages.shape[1]
    if page_size not in _DSV4_DSPARK_PAGE_SIZES:
        raise ValueError(
            f"DSpark page size must be one of {_DSV4_DSPARK_PAGE_SIZES}, got {page_size}"
        )
    for name, value in (("Q", q), ("draft KV", draft_block), ("pages", kv_pages)):
        if value.dtype != torch.bfloat16:
            raise ValueError(f"DSpark {name} must use BF16, got {value.dtype}")
    expected_draft_shape = (batch, _DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE, dim)
    if draft_block.shape != expected_draft_shape:
        raise ValueError(
            f"DSpark draft KV must have shape {expected_draft_shape}, got {tuple(draft_block.shape)}"
        )
    if block_tables.dtype != torch.int32 or block_tables.stride(1) != 1:
        raise ValueError(
            "DSpark page tables must have contiguous INT32 rows, got "
            f"{block_tables.dtype} with strides {block_tables.stride()}"
        )
    if block_tables.shape[0] != batch:
        raise ValueError(f"DSpark page tables must have {batch} rows, got {block_tables.shape[0]}")
    for name, value in (
        ("positions", positions),
        ("valid lengths", valid_lengths),
        ("capacities", capacities),
    ):
        if value.dtype != torch.int64 or value.shape != (batch,) or not value.is_contiguous():
            raise ValueError(
                f"DSpark {name} must be a contiguous INT64 [{batch}] vector, got "
                f"{value.dtype} {tuple(value.shape)} with strides {value.stride()}"
            )
    if attn_sink.dtype != torch.float32 or attn_sink.shape != (heads,):
        raise ValueError(
            f"DSpark attention sink must be FP32 [{heads}], got "
            f"{attn_sink.dtype} {tuple(attn_sink.shape)}"
        )
    expected_freqs_shape = (batch, block, _DSV4_DSPARK_ROPE_DIM // 2, 2)
    if (
        inverse_rope_freqs.dtype != torch.float32
        or inverse_rope_freqs.shape != expected_freqs_shape
    ):
        raise ValueError(
            f"DSpark inverse RoPE frequencies must be FP32 {list(expected_freqs_shape)}, got "
            f"{inverse_rope_freqs.dtype} {tuple(inverse_rope_freqs.shape)}"
        )
    if (
        kv_pages.stride(0) <= 0
        or kv_pages.stride(0) % 8
        or kv_pages.stride(1) != dim
        or kv_pages.stride(2) != 1
    ):
        raise ValueError(
            "DSpark pages require contiguous tokens and a positive 16-byte-aligned page "
            f"stride, got strides {kv_pages.stride()}"
        )
    for value in (q, draft_block, attn_sink, inverse_rope_freqs):
        if not value.is_contiguous():
            raise ValueError("DSpark query, draft, sink and frequencies must be contiguous")
    if not all(
        value.is_cuda and value.device == q.device
        for value in (
            q,
            draft_block,
            kv_pages,
            block_tables,
            positions,
            valid_lengths,
            capacities,
            attn_sink,
            inverse_rope_freqs,
        )
    ):
        raise ValueError("All DSpark attention inputs must be on the same CUDA device")
    cache_key = (block, heads, page_size, arch_str)
    compiled = _dspark_attention_kernel_cache.get(cache_key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DSpark paged attention must run eagerly before graph capture")
        compiled = _compile_dspark_attention(block, arch_str, page_size, heads)
        _dspark_attention_kernel_cache[cache_key] = compiled
    output = torch.empty_like(q)
    compiled(
        batch,
        kv_pages.shape[0],
        kv_pages.stride(0),
        block_tables.shape[1],
        block_tables.stride(0),
        q.data_ptr(),
        kv_pages.data_ptr(),
        draft_block.data_ptr(),
        block_tables.data_ptr(),
        positions.data_ptr(),
        valid_lengths.data_ptr(),
        capacities.data_ptr(),
        attn_sink.data_ptr(),
        inverse_rope_freqs.data_ptr(),
        output,
        softmax_scale,
    )
    return output


@torch.library.register_fake("trtllm::fused_dsv4_dspark_attention")
def _(
    q: torch.Tensor,
    draft_block: torch.Tensor,
    kv_pages: torch.Tensor,
    block_tables: torch.Tensor,
    positions: torch.Tensor,
    valid_lengths: torch.Tensor,
    capacities: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    return torch.empty_like(q)


if IS_CUTLASS_DSL_RUBIN_AVAILABLE:
    from .cute_dsl_custom_ops import CuteDSLMXFP8RubinLinear, _get_mxfp8_gemm_alpha

    _DSPARK_CONTEXT_WKV_TUNING_CONFIG = replace(
        CuteDSLMXFP8RubinLinear.tuning_config,
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                0,
                0,
                _gen_dspark_context_wkv_m_profiles,
                _map_dspark_context_wkv_m_profile,
            ),
        ),
    )

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
        alpha = _get_mxfp8_gemm_alpha(input.device)
        runner = CuteDSLMXFP8RubinLinear(output_dtype=output_dtype, use_tvm_ffi=use_tvm_ffi)
        inputs = [input, weight, input_scale, weight_scale, alpha]
        _, best_tactic = AutoTuner.get().choose_one(
            "trtllm::cute_dsl_dspark_context_wkv_gemm_rubin",
            [runner],
            _DSPARK_CONTEXT_WKV_TUNING_CONFIG,
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


def _log_unsupported(reason: str, key: str) -> bool:
    logger.debug_once(
        f"Falling back from fused DSV4 DSpark attention: {reason}",
        key=("fused_dsv4_dspark_attention_unsupported", key),
    )
    return False


def is_dsv4_dspark_attention_config_supported(
    block_size: int,
    num_heads: int,
    head_dim: int,
    window_size: int,
) -> bool:
    """Return whether static model geometry matches the DSV4 specialization."""
    return (
        block_size in _DSV4_DSPARK_BLOCK_SIZES
        and num_heads in _DSV4_DSPARK_SUPPORTED_NUM_HEADS
        and head_dim == _DSV4_DSPARK_HEAD_DIM
        and window_size == _DSV4_DSPARK_WINDOW_SIZE
    )


def is_fused_dsv4_dspark_attention_supported(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    valid_len: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
) -> bool:
    """Return whether tensors can use the fused DSV4 DSpark attention kernel."""
    if _get_dspark_arch_str() is None:
        return _log_unsupported(
            f"SM {get_sm_version()} is not SM100, SM103, or SM107", "sm_version"
        )
    if not all(t.is_cuda for t in (q, kv_cache, valid_len, attn_sink, inverse_rope_freqs)):
        return _log_unsupported("all inputs must be CUDA tensors", "device")
    if q.dtype != torch.bfloat16:
        return _log_unsupported(f"q dtype must be BF16, got {q.dtype}", "q_dtype")
    if kv_cache.dtype != q.dtype:
        return _log_unsupported(
            f"kv_cache dtype must match q, got {kv_cache.dtype}", "kv_cache_dtype"
        )
    if valid_len.dtype != torch.int64:
        return _log_unsupported(
            f"valid_len must use INT64, got {valid_len.dtype}", "valid_len_dtype"
        )
    if attn_sink.dtype != torch.float32 or inverse_rope_freqs.dtype != torch.float32:
        return _log_unsupported(
            "attn_sink and inverse_rope_freqs must be FP32; "
            f"got {attn_sink.dtype} and {inverse_rope_freqs.dtype}",
            "aux_dtype",
        )
    if q.ndim != 4 or kv_cache.ndim != 3:
        return _log_unsupported(
            f"expected q/kv_cache ranks 4/3, got {q.ndim}/{kv_cache.ndim}", "tensor_ranks"
        )
    if q.shape[1] not in _DSV4_DSPARK_BLOCK_SIZES:
        return _log_unsupported(f"draft block size must be 5 or 6, got {q.shape[1]}", "block_size")
    if q.shape[2] not in _DSV4_DSPARK_SUPPORTED_NUM_HEADS or q.shape[3] != _DSV4_DSPARK_HEAD_DIM:
        return _log_unsupported(
            f"q must have {_DSV4_DSPARK_SUPPORTED_NUM_HEADS} heads and head_dim 512; "
            f"got {q.shape[2]} heads and head_dim {q.shape[3]}",
            "q_shape",
        )
    if kv_cache.shape[1:] != (_DSV4_DSPARK_WINDOW_SIZE, _DSV4_DSPARK_HEAD_DIM):
        return _log_unsupported(
            f"kv_cache must have trailing shape (128, 512), got {tuple(kv_cache.shape[1:])}",
            "kv_cache_shape",
        )
    if (
        kv_cache.stride(0) <= 0
        or kv_cache.stride(0) % 8 != 0
        or kv_cache.stride(1) != _DSV4_DSPARK_HEAD_DIM
        or kv_cache.stride(2) != 1
    ):
        return _log_unsupported(
            "kv_cache strides must have a positive 16-byte-aligned page stride "
            f"and trailing strides (512, 1), got {kv_cache.stride()}",
            "kv_cache_layout",
        )
    if valid_len.shape != (q.shape[0],):
        return _log_unsupported(
            f"valid_len shape must be {(q.shape[0],)}, got {tuple(valid_len.shape)}",
            "valid_len_shape",
        )
    if attn_sink.shape != (q.shape[2],):
        return _log_unsupported(
            f"attn_sink shape must be {(q.shape[2],)}, got {tuple(attn_sink.shape)}",
            "attn_sink_shape",
        )
    expected_freqs_shape = (q.shape[0], q.shape[1], _DSV4_DSPARK_ROPE_DIM // 2, 2)
    if inverse_rope_freqs.shape != expected_freqs_shape:
        return _log_unsupported(
            f"inverse_rope_freqs shape must be {expected_freqs_shape}, "
            f"got {tuple(inverse_rope_freqs.shape)}",
            "inverse_rope_freqs_shape",
        )
    for name, tensor in (
        ("q", q),
        ("valid_len", valid_len),
        ("attn_sink", attn_sink),
        ("inverse_rope_freqs", inverse_rope_freqs),
    ):
        if not tensor.is_contiguous():
            return _log_unsupported(f"{name} must be contiguous", f"{name}_layout")
    return True


def warmup_fused_dsv4_dspark_attention(block_size: int, eps: float, num_heads: int) -> None:
    """Best-effort prewarm of the production DSV4 DSpark fused-op path."""
    if _get_dspark_arch_str() is None or block_size not in _DSV4_DSPARK_BLOCK_SIZES:
        return

    batch = 1
    device = torch.device("cuda")
    try:
        with torch.inference_mode():
            q = torch.zeros(
                (batch, block_size, num_heads, _DSV4_DSPARK_HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            )
            main_x = torch.zeros(
                (batch, 1, _DSV4_DSPARK_HEAD_DIM), dtype=torch.bfloat16, device=device
            )
            block_x = torch.zeros(
                (batch, block_size, _DSV4_DSPARK_HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            )
            weight = torch.ones((_DSV4_DSPARK_HEAD_DIM,), dtype=torch.bfloat16, device=device)
            main_freqs = torch.zeros(
                (batch, _DSV4_DSPARK_ROPE_DIM // 2, 2),
                dtype=torch.float32,
                device=device,
            )
            block_freqs = torch.zeros(
                (batch * block_size, _DSV4_DSPARK_ROPE_DIM // 2, 2),
                dtype=torch.float32,
                device=device,
            )
            kv_cache = torch.zeros(
                (batch, _DSV4_DSPARK_WINDOW_SIZE, _DSV4_DSPARK_HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            )
            slots = torch.zeros((batch,), dtype=torch.int64, device=device)
            start_pos = torch.zeros((batch,), dtype=torch.int64, device=device)
            slots_i32, cache_seqs = cute_dsl_dspark_rmsnorm_rope_cache_write(
                main_x,
                weight,
                main_freqs,
                kv_cache,
                slots,
                start_pos,
                eps,
            )
            draft_block = cute_dsl_dspark_rmsnorm_rope_draft_block(
                block_x,
                weight,
                block_freqs,
                eps,
            )
            fused_dsv4_dspark_attention(
                q,
                draft_block,
                kv_cache,
                slots_i32[:, None],
                cache_seqs.to(torch.int64),
                torch.zeros((batch,), dtype=torch.int64, device=device),
                torch.full((batch,), _DSV4_DSPARK_WINDOW_SIZE, dtype=torch.int64, device=device),
                torch.zeros((num_heads,), dtype=torch.float32, device=device),
                block_freqs.view(
                    batch,
                    block_size,
                    _DSV4_DSPARK_ROPE_DIM // 2,
                    2,
                ),
                1.0,
            )
        torch.cuda.synchronize()
    except RuntimeError as e:
        logger.warning(
            "DSV4 DSpark CuTe DSL attention prewarm failed; the op will "
            f"self-JIT on first use. {type(e).__name__}: {e}"
        )
