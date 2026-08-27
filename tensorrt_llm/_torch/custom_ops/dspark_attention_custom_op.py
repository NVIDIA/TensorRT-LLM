# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch custom op for fused DSV4 DSpark rolling-window attention.

The op runs the tcgen05 MMA kernel
(:class:`DSparkAttention`) on the supported DSV4 DSpark
geometry: 128 heads, head_dim 512, draft block 5 or 6, and a 128-row rolling
window. Other shapes fall back to the pure-PyTorch reference path in
``models/dspark/attention.py``.
"""

from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass.cute.typing import Numeric, Pointer, Type

from ..._utils import get_sm_version
from ...logger import logger
from ..cute_dsl_kernels.blackwell.dspark.attention import DSparkAttention
from ..cute_dsl_kernels.blackwell.utils import make_ptr
from .dspark_rmsnorm_rope_custom_op import (
    _get_dspark_arch_str,
    cute_dsl_dspark_rmsnorm_rope_cache_write,
    cute_dsl_dspark_rmsnorm_rope_draft_block,
)

_DSV4_DSPARK_NUM_HEADS = 128
_DSV4_DSPARK_HEAD_DIM = 512
_DSV4_DSPARK_WINDOW_SIZE = 128
_DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE = 8
_DSV4_DSPARK_BLOCK_SIZES = (5, 6)
_DSV4_DSPARK_ROPE_DIM = 64


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
        and num_heads == _DSV4_DSPARK_NUM_HEADS
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
        return _log_unsupported(f"SM {get_sm_version()} is not SM100 or SM103", "sm_version")
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
    if q.shape[2:] != (_DSV4_DSPARK_NUM_HEADS, _DSV4_DSPARK_HEAD_DIM):
        return _log_unsupported(
            "q must have 128 heads and head_dim 512; "
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
    if attn_sink.shape != (_DSV4_DSPARK_NUM_HEADS,):
        return _log_unsupported(
            f"attn_sink shape must be {(_DSV4_DSPARK_NUM_HEADS,)}, got {tuple(attn_sink.shape)}",
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


def _is_fused_dsv4_dspark_attention_input_supported(
    q: torch.Tensor,
    draft_block: torch.Tensor,
    kv_cache: torch.Tensor,
    slots_i32: torch.Tensor,
    cache_seqs: torch.Tensor,
    valid_len: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
) -> bool:
    if not is_fused_dsv4_dspark_attention_supported(
        q, kv_cache, valid_len, attn_sink, inverse_rope_freqs
    ):
        return False
    if not all(t.is_cuda for t in (draft_block, slots_i32, cache_seqs)):
        return _log_unsupported("all fused-op inputs must be CUDA tensors", "input_device")
    if draft_block.dtype != q.dtype:
        return _log_unsupported(
            f"draft_block dtype must match q, got {draft_block.dtype}", "draft_block_dtype"
        )
    if slots_i32.dtype != torch.int32 or cache_seqs.dtype != torch.int32:
        return _log_unsupported(
            "slots_i32 and cache_seqs must use INT32; "
            f"got {slots_i32.dtype} and {cache_seqs.dtype}",
            "index_dtype",
        )
    expected_draft_block_shape = (
        q.shape[0],
        _DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE,
        _DSV4_DSPARK_HEAD_DIM,
    )
    if draft_block.shape != expected_draft_block_shape:
        return _log_unsupported(
            f"draft_block shape must be {expected_draft_block_shape}, "
            f"got {tuple(draft_block.shape)}",
            "draft_block_shape",
        )
    expected_batch_shape = (q.shape[0],)
    if slots_i32.shape != expected_batch_shape or cache_seqs.shape != expected_batch_shape:
        return _log_unsupported(
            f"slots_i32 and cache_seqs shapes must be {expected_batch_shape}; "
            f"got {tuple(slots_i32.shape)} and {tuple(cache_seqs.shape)}",
            "index_shape",
        )
    for name, tensor in (
        ("draft_block", draft_block),
        ("slots_i32", slots_i32),
        ("cache_seqs", cache_seqs),
    ):
        if not tensor.is_contiguous():
            return _log_unsupported(f"{name} must be contiguous", f"{name}_layout")
    return True


_dspark_attention_kernel_cache: dict[tuple[int, str], Callable[..., None]] = {}


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
) -> Callable[..., None]:
    """Compile the pointer host wrapper without runtime tensor specimens."""
    num_heads, head_dim = _DSV4_DSPARK_NUM_HEADS, _DSV4_DSPARK_HEAD_DIM
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
    )
    # Scalars and typed, aligned compile-only pointers define the runtime ABI.
    # ``output_fake`` keeps one tensor argument for TVM-FFI environment-stream
    # detection without tying compilation to a real worker buffer.
    compiled = cute.compile(
        kernel.wrapper,
        1,
        1,
        _DSV4_DSPARK_WINDOW_SIZE * head_dim,
        _make_compile_gmem_pointer(cutlass.BFloat16, 16),
        _make_compile_gmem_pointer(cutlass.BFloat16, 16),
        _make_compile_gmem_pointer(cutlass.BFloat16, 16),
        _make_compile_gmem_pointer(cutlass.Int32, 4),
        _make_compile_gmem_pointer(cutlass.Int32, 4),
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
    arch_str = _get_dspark_arch_str()
    if arch_str is None:
        raise RuntimeError("fused DSV4 DSpark attention requires SM100 or SM103")
    cache_key = (block_size, arch_str)
    compiled = _dspark_attention_kernel_cache.get(cache_key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "fused DSV4 DSpark attention must be warmed up before CUDA graph capture"
            )
        compiled = _compile_dspark_attention(block_size, arch_str)
        _dspark_attention_kernel_cache[cache_key] = compiled

    output = torch.empty_like(q)
    compiled(
        q.shape[0],
        kv_cache.shape[0],
        kv_cache.stride(0),
        q.data_ptr(),
        kv_cache.data_ptr(),
        draft_block.data_ptr(),
        slots_i32.data_ptr(),
        cache_seqs.data_ptr(),
        valid_len.data_ptr(),
        attn_sink.data_ptr(),
        inverse_rope_freqs.data_ptr(),
        output,
        softmax_scale,
    )
    return output


@torch.library.custom_op(
    "trtllm::fused_dsv4_dspark_attention",
    mutates_args=(),
    device_types="cuda",
)
def fused_dsv4_dspark_attention(
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
    """Run fused DSV4 DSpark attention after validating its tensor contract."""

    if not _is_fused_dsv4_dspark_attention_input_supported(
        q,
        draft_block,
        kv_cache,
        slots_i32,
        cache_seqs,
        valid_len,
        attn_sink,
        inverse_rope_freqs,
    ):
        raise ValueError(
            "fused_dsv4_dspark_attention requires contiguous supported DSV4 DSpark tensors "
            "([B, 5|6, 128, 512] queries, [B, 8, 512] draft blocks, INT32 indices, "
            "and a 128-row BF16 window) on SM100 or SM103"
        )

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


@torch.library.register_fake("trtllm::fused_dsv4_dspark_attention")
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


def warmup_fused_dsv4_dspark_attention(block_size: int, eps: float) -> None:
    """Best-effort prewarm of the production DSV4 DSpark fused-op path."""
    if _get_dspark_arch_str() is None or block_size not in _DSV4_DSPARK_BLOCK_SIZES:
        return

    batch = 1
    device = torch.device("cuda")
    try:
        with torch.inference_mode():
            q = torch.zeros(
                (batch, block_size, _DSV4_DSPARK_NUM_HEADS, _DSV4_DSPARK_HEAD_DIM),
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
                slots_i32,
                cache_seqs,
                torch.zeros((batch,), dtype=torch.int64, device=device),
                torch.zeros((_DSV4_DSPARK_NUM_HEADS,), dtype=torch.float32, device=device),
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
