# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared MLA FP4 KV-cache helpers.

The high-precision (HP) BF16 KV pool is a standalone circular buffer used
alongside the paged FP4 KV pool when MLA models run with NVFP4 KV cache.
The pool retains one 16-token quantization tile plus speculative-rewind slack
per sequence.

Used by the TRTLLM attention backend FP4 MLA FMHA path.
"""

import importlib.util
import os
from typing import Any, Literal, Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import get_sm_version

from .fp4_mla_kernels import (
    _fp4_mla_context_cache_update_kernel,
    _fp4_mla_generation_fused_qk_rope_cache_update_kernel,
)

HP_BLOCK_SIZE: int = 16
FP4_BLOCK_SIZE: int = 16
FP4_MLA_TOKENS_PER_BLOCK: int = 128
FP4_MLA_SCALE_ROW_GROUP: int = 128
FP4_MLA_SCALE_COL_GROUP: int = 4
FP4_MLA_P_GLOBAL_SCALE: float = 448.0 * 6.0
FP4_MLA_Q_STATIC_AMAX: float = 400.0
FP4_MLA_KV_STATIC_AMAX: float = 30.0
FP4_MLA_Q_GLOBAL_SCALE: float = FP4_MLA_P_GLOBAL_SCALE / FP4_MLA_Q_STATIC_AMAX
FP4_MLA_KV_GLOBAL_SCALE: float = FP4_MLA_P_GLOBAL_SCALE / FP4_MLA_KV_STATIC_AMAX
# Max finite e4m3 magnitude for FP4 MLA block-scale clamping.
FP4_MLA_E4M3_MAX: float = 448.0
FP4_MLA_Q_RESIDUAL_DIM: int = 64
FP4_MLA_K_RESIDUAL_DIM: int = FP4_MLA_Q_RESIDUAL_DIM
FP4_MLA_Q_PREFIX_DIM: int = 512
FP4_MLA_Q_PREFIX_BLOCK_DIM: int = 256
FP4_MLA_Q1_PREFIX_BLOCK_DIM: int = 512
FP4_MLA_Q_LOGICAL_DIM: int = FP4_MLA_Q_PREFIX_DIM + 2 * FP4_MLA_Q_RESIDUAL_DIM
FP4_MLA_Q_PACKED_DIM: int = FP4_MLA_Q_LOGICAL_DIM // 2
FP4_MLA_Q_SF_GROUPS: int = FP4_MLA_Q_LOGICAL_DIM // FP4_BLOCK_SIZE
FP4_MLA_ATTENTION_BACKEND_ENV = "TRTLLM_FP4_MLA_ATTENTION_BACKEND"
FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV = "TRTLLM_FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE"
_FP4_MLA_CUTEDSL_BACKEND = "cutedsl"
_FP4_MLA_K_RESIDUAL_BACKENDS = ("triton", _FP4_MLA_CUTEDSL_BACKEND)
_FP4_MLA_Q1_KV_BLOCKS_SMALL_BATCH = 4
_FP4_MLA_Q1_KV_BLOCKS_MEDIUM_BATCH = 16
_FP4_MLA_Q1_KV_BLOCKS_LARGE_BATCH = 32
_FP4_MLA_Q1_KV_MEDIUM_BATCH_THRESHOLD = 256
_FP4_MLA_Q1_KV_LARGE_BATCH_THRESHOLD = 512
_FP4_MLA_Q1_PREFIX_PAIR_BATCH_THRESHOLD = 640
_FP4_MLA_Q1_PREFIX_GROUP4_BATCH_THRESHOLD = 768
_HPUpdatePhase = Literal["context", "generation"]
_FP4_MLA_TRITON_PRELOAD_KEYS = "_fp4_mla_triton_preload_keys"
_FP4_MLA_PAGE_TABLE_TILE_SIZE = 128
_FP4_MLA_MAX_GRID_Z = 65_535


# Environment helpers


def _env_enabled_default(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _fp4_mla_cutedsl_fused_v_transpose_enabled() -> bool:
    return _env_enabled_default(FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, False)


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _fp4_mla_attention_backend() -> str:
    backend = os.getenv(FP4_MLA_ATTENTION_BACKEND_ENV)
    if backend:
        return backend.lower()
    return _FP4_MLA_CUTEDSL_BACKEND if get_sm_version() == 107 else "triton"


def _cutedsl_backend_available() -> bool:
    try:
        return all(
            importlib.util.find_spec(module) is not None
            for module in ("ctm", "cutlass", "cuda.bindings.driver")
        )
    except ModuleNotFoundError:
        return False


def _fp4_mla_cutedsl_kernel_module() -> Any:
    if _fp4_mla_cutedsl_fused_v_transpose_enabled():
        from . import fp4_mla_cutedsl_mufu16_fused_v_transpose

        return fp4_mla_cutedsl_mufu16_fused_v_transpose

    from . import fp4_mla_cutedsl_mufu16

    return fp4_mla_cutedsl_mufu16


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _fp4_mla_q1_kv_blocks_per_program(num_gen: int, v_head_dim: int) -> int:
    """Select the Q1 KV work per program without changing the launch boundary."""
    large_batch_block_dim = _FP4_MLA_Q1_KV_BLOCKS_LARGE_BATCH * FP4_BLOCK_SIZE
    if num_gen >= _FP4_MLA_Q1_KV_LARGE_BATCH_THRESHOLD and v_head_dim % large_batch_block_dim == 0:
        return _FP4_MLA_Q1_KV_BLOCKS_LARGE_BATCH
    medium_batch_block_dim = _FP4_MLA_Q1_KV_BLOCKS_MEDIUM_BATCH * FP4_BLOCK_SIZE
    if (
        num_gen >= _FP4_MLA_Q1_KV_MEDIUM_BATCH_THRESHOLD
        and v_head_dim % medium_batch_block_dim == 0
    ):
        return _FP4_MLA_Q1_KV_BLOCKS_MEDIUM_BATCH
    small_batch_block_dim = _FP4_MLA_Q1_KV_BLOCKS_SMALL_BATCH * FP4_BLOCK_SIZE
    if v_head_dim % small_batch_block_dim == 0:
        return _FP4_MLA_Q1_KV_BLOCKS_SMALL_BATCH
    return 1


def _fp4_mla_q1_prefix_blocks_per_program(
    num_gen: int,
    q1_kv_blocks_per_program: int,
) -> int:
    """Select rolled Q-prefix work only when the batch keeps it efficient."""
    max_prefix_blocks = FP4_MLA_Q_PREFIX_DIM // FP4_MLA_Q1_PREFIX_BLOCK_DIM
    if (
        num_gen >= _FP4_MLA_Q1_PREFIX_PAIR_BATCH_THRESHOLD
        and q1_kv_blocks_per_program == _FP4_MLA_Q1_KV_BLOCKS_LARGE_BATCH
    ):
        if num_gen >= _FP4_MLA_Q1_PREFIX_GROUP4_BATCH_THRESHOLD:
            return min(4, max_prefix_blocks)
        return min(2, max_prefix_blocks)
    return 1


def _fp4_mla_q1_preload_variants(
    max_num_sequences: int,
    v_head_dim: int,
) -> tuple[tuple[int, int], ...]:
    """Return every Q1 tuning variant reachable by the configured batch limit."""
    batch_sizes = [1]
    for threshold in (
        _FP4_MLA_Q1_KV_MEDIUM_BATCH_THRESHOLD,
        _FP4_MLA_Q1_KV_LARGE_BATCH_THRESHOLD,
        _FP4_MLA_Q1_PREFIX_PAIR_BATCH_THRESHOLD,
        _FP4_MLA_Q1_PREFIX_GROUP4_BATCH_THRESHOLD,
    ):
        if threshold <= max_num_sequences:
            batch_sizes.append(threshold)

    variants = []
    for batch_size in batch_sizes:
        kv_blocks = _fp4_mla_q1_kv_blocks_per_program(batch_size, v_head_dim)
        prefix_blocks = _fp4_mla_q1_prefix_blocks_per_program(
            batch_size,
            kv_blocks,
        )
        variant = (kv_blocks, prefix_blocks)
        if variant not in variants:
            variants.append(variant)
    return tuple(variants)


def _fp4_mla_triton_preload_key_set(metadata: Any) -> set[tuple[object, ...]]:
    """Return the engine-scoped set of Triton variants loaded during warmup."""
    owner = getattr(metadata, "kv_cache_manager", None)
    if owner is None:
        owner = metadata
    preload_keys = getattr(owner, _FP4_MLA_TRITON_PRELOAD_KEYS, None)
    if preload_keys is None:
        preload_keys = set()
        setattr(owner, _FP4_MLA_TRITON_PRELOAD_KEYS, preload_keys)
    return preload_keys


@triton.jit
def _fp4_mla_store_sequence_append_metadata(
    append_lens_ptr,
    kv_lens_ptr,
    batch_indices_ptr,
    positions_ptr,
    sequence_idx,
    num_tokens,
    PREFIX_BLOCK: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    append_len = tl.load(append_lens_ptr + sequence_idx)
    prefix_offsets = tl.arange(0, PREFIX_BLOCK)
    token_start = append_len - append_len
    for prefix_start in tl.range(0, sequence_idx, PREFIX_BLOCK):
        preceding_sequences = prefix_start + prefix_offsets
        preceding_lens = tl.load(
            append_lens_ptr + preceding_sequences,
            mask=preceding_sequences < sequence_idx,
            other=0,
        )
        token_start += tl.sum(preceding_lens)

    cached_len = tl.load(kv_lens_ptr + sequence_idx) - append_len

    token_offsets = tl.arange(0, TOKEN_BLOCK)
    for block_start in tl.range(0, append_len, TOKEN_BLOCK):
        local_offsets = block_start + token_offsets
        token_mask = (local_offsets < append_len) & (token_start + local_offsets < num_tokens)
        output_offsets = token_start + local_offsets
        tl.store(
            batch_indices_ptr + output_offsets,
            sequence_idx,
            mask=token_mask,
        )
        tl.store(
            positions_ptr + output_offsets,
            cached_len + local_offsets,
            mask=token_mask,
        )


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_contexts",
        "num_generation_sequences",
    ],
    do_not_specialize_on_alignment=[
        "num_tokens",
        "num_contexts",
        "num_generation_sequences",
    ],
)
def _fp4_mla_append_metadata_kernel(
    append_lens_ptr,
    kv_lens_ptr,
    batch_indices_ptr,
    positions_ptr,
    num_tokens,
    num_contexts,
    num_generation_sequences,
    ONE_TOKEN_GENERATION: tl.constexpr,
    PREFIX_BLOCK: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    GENERATION_BLOCK: tl.constexpr,
):
    program_idx = tl.program_id(0)
    if ONE_TOKEN_GENERATION:
        if program_idx < num_contexts:
            _fp4_mla_store_sequence_append_metadata(
                append_lens_ptr,
                kv_lens_ptr,
                batch_indices_ptr,
                positions_ptr,
                program_idx,
                num_tokens,
                PREFIX_BLOCK,
                TOKEN_BLOCK,
            )
        else:
            generation_offsets = (program_idx - num_contexts) * GENERATION_BLOCK + tl.arange(
                0, GENERATION_BLOCK
            )
            generation_mask = generation_offsets < num_generation_sequences
            sequence_indices = num_contexts + generation_offsets
            output_offsets = num_tokens - num_generation_sequences + generation_offsets
            generation_mask = generation_mask & (output_offsets < num_tokens)
            generation_positions = (
                tl.load(
                    kv_lens_ptr + sequence_indices,
                    mask=generation_mask,
                    other=1,
                )
                - 1
            )
            tl.store(
                batch_indices_ptr + output_offsets,
                sequence_indices,
                mask=generation_mask,
            )
            tl.store(
                positions_ptr + output_offsets,
                generation_positions,
                mask=generation_mask,
            )
    else:
        _fp4_mla_store_sequence_append_metadata(
            append_lens_ptr,
            kv_lens_ptr,
            batch_indices_ptr,
            positions_ptr,
            program_idx,
            num_tokens,
            PREFIX_BLOCK,
            TOKEN_BLOCK,
        )


def populate_fp4_mla_append_metadata(
    append_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_tokens: int,
    num_sequences: int,
    num_contexts: int,
    num_context_tokens: int,
) -> None:
    """Populate FP4 MLA token-to-sequence metadata in one Triton launch.

    Mixed batches vectorize their one-token generation rows. Multi-token MTP
    and fallback shapes use the generic per-sequence path in the same kernel.
    """
    if num_sequences <= 0 or num_tokens <= 0:
        return
    if not 0 <= num_contexts <= num_sequences:
        raise ValueError(
            f"FP4 MLA num_contexts must be in [0, {num_sequences}], got {num_contexts}."
        )
    if not 0 <= num_context_tokens <= num_tokens:
        raise ValueError(
            f"FP4 MLA num_context_tokens must be in [0, {num_tokens}], got {num_context_tokens}."
        )

    tensors = (
        append_lens,
        kv_lens,
        batch_indices,
        positions,
    )
    if any(tensor.ndim != 1 or tensor.stride(0) != 1 for tensor in tensors):
        raise ValueError("FP4 MLA append metadata tensors must be contiguous and one-dimensional.")
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise TypeError("FP4 MLA append metadata tensors must use int32.")
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("FP4 MLA append metadata tensors must be CUDA tensors.")
    if any(tensor.device != append_lens.device for tensor in tensors[1:]):
        raise ValueError("FP4 MLA append metadata tensors must be on the same device.")
    sequence_tensors = (append_lens, kv_lens)
    if any(tensor.numel() < num_sequences for tensor in sequence_tensors):
        raise ValueError(
            f"FP4 MLA sequence metadata tensors need at least {num_sequences} entries."
        )
    token_tensors = (batch_indices, positions)
    if any(tensor.numel() < num_tokens for tensor in token_tensors):
        raise ValueError(f"FP4 MLA token metadata tensors need at least {num_tokens} entries.")

    num_generation_sequences = num_sequences - num_contexts
    # Each scheduled generation sequence appends at least one token. Equality
    # therefore identifies the common mixed batch with one token per decode
    # row without reading the device append lengths back on the host.
    one_token_generation = num_tokens == num_context_tokens + num_generation_sequences
    generation_block = 128
    grid = num_sequences
    if one_token_generation:
        grid = num_contexts + triton.cdiv(num_generation_sequences, generation_block)

    _fp4_mla_append_metadata_kernel[(grid,)](
        append_lens,
        kv_lens,
        batch_indices,
        positions,
        num_tokens,
        num_contexts,
        num_generation_sequences,
        ONE_TOKEN_GENERATION=one_token_generation,
        PREFIX_BLOCK=128,
        TOKEN_BLOCK=256,
        GENERATION_BLOCK=generation_block,
        num_warps=4,
    )


@triton.jit(
    do_not_specialize=["num_gen", "generation_len"],
    do_not_specialize_on_alignment=["num_gen", "generation_len"],
)
def _fp4_mla_generation_lengths_kernel(
    kv_lens_ptr,
    prompt_lens_ptr,
    corrected_kv_lens_ptr,
    generation_lens_ptr,
    num_gen,
    generation_len,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_gen
    kv_lens = tl.load(kv_lens_ptr + offsets, mask=mask, other=0)
    prompt_lens = tl.load(prompt_lens_ptr + offsets, mask=mask, other=0)
    tl.store(
        corrected_kv_lens_ptr + offsets,
        kv_lens - prompt_lens + generation_len,
        mask=mask,
    )
    tl.store(generation_lens_ptr + offsets, generation_len, mask=mask)


def populate_fp4_mla_generation_lengths(
    kv_lens: torch.Tensor,
    prompt_lens: torch.Tensor,
    corrected_kv_lens: torch.Tensor,
    generation_lens: torch.Tensor,
    *,
    num_gen_tokens: int,
    num_gen: int,
) -> None:
    """Populate reusable FP4 MLA generation lengths in one Triton launch."""
    if num_gen <= 0 or num_gen_tokens % num_gen != 0:
        raise ValueError(
            "FP4 MLA generation lengths require a positive sequence count and "
            f"uniform token count, got {num_gen_tokens} tokens for {num_gen} sequences."
        )
    tensors = (kv_lens, prompt_lens, corrected_kv_lens, generation_lens)
    if any(tensor.ndim != 1 or tensor.stride(0) != 1 for tensor in tensors):
        raise ValueError(
            "FP4 MLA generation length tensors must be contiguous and one-dimensional."
        )
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise TypeError("FP4 MLA generation length tensors must use int32.")
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("FP4 MLA generation length tensors must be CUDA tensors.")
    if any(tensor.device != kv_lens.device for tensor in tensors[1:]):
        raise ValueError("FP4 MLA generation length tensors must be on the same device.")
    if any(tensor.numel() < num_gen for tensor in tensors):
        raise ValueError(f"FP4 MLA generation length tensors need at least {num_gen} entries.")

    block = 128
    _fp4_mla_generation_lengths_kernel[(triton.cdiv(num_gen, block),)](
        kv_lens,
        prompt_lens,
        corrected_kv_lens,
        generation_lens,
        num_gen,
        num_gen_tokens // num_gen,
        BLOCK=block,
        num_warps=4,
    )


def _fp4_mla_page_table_spec(kv_cache_manager: Any) -> Any:
    get_spec = getattr(kv_cache_manager, "get_fp4_mla_page_table_spec", None)
    if not callable(get_spec):
        raise RuntimeError("FP4 MLA requires Fp4MlaKVCacheManagerV2 page metadata.")
    spec = get_spec()
    for field_name in (
        "cache_pool_id",
        "cache_page_index_scale",
        "hp_pool_id",
        "hp_page_index_scale",
    ):
        value = getattr(spec, field_name, None)
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"FP4 MLA page-table spec requires non-negative {field_name}, got {value}."
            )
    if spec.cache_page_index_scale <= 0 or spec.hp_page_index_scale <= 0:
        raise ValueError("FP4 MLA page-index scales must be positive.")
    return spec


# Mixed batches frequently vary by one sequence. Keep per-forward dimensions
# out of Triton's specialization key and tile page rows at one fixed width so
# those shape changes cannot trigger JIT compilation on the critical path.
@triton.jit(
    do_not_specialize=["num_sequences", "num_contexts", "max_pages"],
    do_not_specialize_on_alignment=["num_sequences", "num_contexts", "max_pages"],
)
def _fp4_mla_materialize_page_table_kernel(
    page_ids_ptr,
    paged_kv_indptr_ptr,
    paged_kv_indptr_decode_ptr,
    block_offsets_ptr,
    kv_lens_ptr,
    generation_kv_lens_ptr,
    block_offsets_stride,
    num_sequences,
    num_contexts,
    max_pages,
    PAGE_SIZE: tl.constexpr,
    PAGE_INDEX_SCALE: tl.constexpr,
    PAGE_TILE_SIZE: tl.constexpr,
):
    sequence_idx = tl.program_id(0)
    page_tile_idx = tl.program_id(1)
    page_offsets = page_tile_idx * PAGE_TILE_SIZE + tl.arange(0, PAGE_TILE_SIZE)
    generation_idx = sequence_idx - num_contexts
    is_generation = generation_idx >= 0
    context_kv_len = tl.load(kv_lens_ptr + sequence_idx)
    generation_kv_len = tl.load(
        generation_kv_lens_ptr + generation_idx,
        mask=is_generation,
        other=0,
    )
    kv_len = tl.maximum(tl.where(is_generation, generation_kv_len, context_kv_len), 0)
    num_active_pages = tl.minimum(
        (kv_len + PAGE_SIZE - 1) // PAGE_SIZE,
        max_pages,
    )
    active_page_mask = page_offsets < num_active_pages
    encoded_page_offsets = tl.load(
        block_offsets_ptr + sequence_idx * block_offsets_stride + page_offsets,
        mask=active_page_mask,
        other=-1,
    )
    decoded_page_ids = tl.where(
        encoded_page_offsets >= 0,
        encoded_page_offsets // PAGE_INDEX_SCALE,
        encoded_page_offsets,
    )
    page_ids = tl.where(active_page_mask, decoded_page_ids, 0)
    table_offset = sequence_idx * max_pages + page_offsets
    # Fixed-stride indptrs expose the whole row. Initialize inactive slots so
    # masked or prefetched page-table reads cannot observe stale page IDs.
    tl.store(
        page_ids_ptr + table_offset,
        page_ids,
        mask=page_offsets < max_pages,
    )

    first_lane = page_offsets == 0
    sequence_start = sequence_idx * max_pages
    tl.store(
        paged_kv_indptr_ptr + sequence_idx + page_offsets,
        sequence_start,
        mask=first_lane,
    )
    tl.store(
        paged_kv_indptr_decode_ptr + generation_idx + page_offsets,
        generation_idx * max_pages,
        mask=first_lane & is_generation,
    )
    final_sequence = sequence_idx == num_sequences - 1
    table_end = num_sequences * max_pages
    num_generation_sequences = num_sequences - num_contexts
    tl.store(
        paged_kv_indptr_ptr + num_sequences + page_offsets,
        table_end,
        mask=first_lane & final_sequence,
    )
    tl.store(
        paged_kv_indptr_decode_ptr + num_generation_sequences + page_offsets,
        num_generation_sequences * max_pages,
        mask=first_lane & final_sequence,
    )


def configure_fp4_mla_device_page_table(
    metadata: Any,
    kv_lens: Optional[torch.Tensor] = None,
) -> bool:
    """Configure the fixed-stride, device-materialized page table.

    Context, generation, and fresh mixed batches receive the full block-offset
    table on the GPU. The materialization kernel decodes V2 page indices and
    refreshes rows from the final device KV lengths before cache update.
    """
    metadata._fp4_mla_device_page_table = False
    metadata._fp4_mla_device_page_table_valid = False
    metadata.fp4_mla_page_table_stride = 0
    metadata.fp4_mla_context_repack_max_touched_pages = 1

    kv_cache_manager = getattr(metadata, "kv_cache_manager", None)
    num_contexts = int(getattr(metadata, "num_contexts", 0))
    num_sequences = int(getattr(metadata, "num_seqs", 0))
    num_generation_sequences = num_sequences - num_contexts
    num_tokens = int(getattr(metadata, "num_tokens", 0))
    num_context_tokens = int(getattr(metadata, "num_ctx_tokens", 0))
    num_generation_tokens = num_tokens - num_context_tokens
    block_offsets = getattr(metadata, "kv_cache_block_offsets", None)
    page_ids = getattr(metadata, "_paged_kv_indices", None)
    paged_kv_indptr = getattr(metadata, "_paged_kv_indptr", None)
    paged_kv_indptr_decode = getattr(metadata, "paged_kv_indptr_decode", None)
    max_page_capacity = int(getattr(kv_cache_manager, "max_blocks_per_seq", 0) or 0)
    page_spec = _fp4_mla_page_table_spec(kv_cache_manager)
    page_index_scale = int(page_spec.cache_page_index_scale)

    tensors = (block_offsets, page_ids, paged_kv_indptr, paged_kv_indptr_decode)
    is_cuda_graph = bool(getattr(metadata, "is_cuda_graph", False))
    generation_only = num_contexts == 0
    fresh_mixed = (
        not is_cuda_graph
        and num_contexts > 0
        and num_generation_sequences > 0
        and int(getattr(metadata, "num_ctx_cached_tokens", 0) or 0) == 0
    )
    fresh_context_only = (
        not is_cuda_graph
        and num_contexts > 0
        and num_generation_sequences == 0
        and int(getattr(metadata, "num_ctx_cached_tokens", 0) or 0) == 0
    )
    has_valid_generation = num_generation_sequences == 0 or (
        num_generation_tokens >= num_generation_sequences
        and num_generation_tokens % num_generation_sequences == 0
    )
    # NVFP4 exposes one data pool plus its paired block-scale pool. The
    # materializer reads encoded data offsets from pool 0.
    supported = (
        (generation_only or fresh_mixed or fresh_context_only)
        and kv_cache_manager is not None
        and has_valid_generation
        and int(getattr(metadata, "beam_width", 1)) == 1
        and not bool(getattr(metadata, "is_spec_dec_tree", False))
        and not bool(getattr(metadata, "locality_domain_enabled", False))
        and not bool(getattr(metadata, "enable_helix", False))
        and int(getattr(kv_cache_manager, "tokens_per_block", 0) or 0) == FP4_MLA_TOKENS_PER_BLOCK
        and max_page_capacity > 0
        and page_index_scale > 0
        and all(isinstance(tensor, torch.Tensor) for tensor in tensors)
        and all(tensor.dtype == torch.int32 for tensor in tensors)
        and all(tensor.is_cuda for tensor in tensors)
    )
    if not supported:
        return False

    hp_page_ids = getattr(metadata, "_fp4_mla_hp_page_indices", None)
    max_pool_id = max(page_spec.cache_pool_id, page_spec.hp_pool_id)
    if (
        not isinstance(hp_page_ids, torch.Tensor)
        or hp_page_ids.dtype != torch.int32
        or not hp_page_ids.is_cuda
        or block_offsets.shape[0] <= max_pool_id
    ):
        return False
    metadata._fp4_mla_cache_pool_id = int(page_spec.cache_pool_id)
    metadata._fp4_mla_cache_page_index_scale = int(page_spec.cache_page_index_scale)
    metadata._fp4_mla_hp_pool_id = int(page_spec.hp_pool_id)
    metadata._fp4_mla_hp_page_index_scale = int(page_spec.hp_page_index_scale)

    max_pages = max_page_capacity
    host_kv_lens_available = (
        isinstance(kv_lens, torch.Tensor)
        and kv_lens.device.type == "cpu"
        and kv_lens.ndim == 1
        and kv_lens.numel() >= num_sequences
    )
    if fresh_mixed and not host_kv_lens_available:
        return False
    if not is_cuda_graph and host_kv_lens_available:
        generation_tokens_per_sequence = (
            num_generation_tokens // num_generation_sequences if num_generation_sequences > 0 else 0
        )
        # Eager execution can narrow the fixed row stride to the current
        # batch. CUDA Graph metadata retains full configured capacity so a
        # replay never changes tensor addresses or launch dimensions.
        max_kv_len = int(kv_lens[:num_sequences].max().item()) + max(
            0,
            generation_tokens_per_sequence - 1,
        )
        max_pages = min(
            max_page_capacity,
            max(1, _ceil_div(max_kv_len, FP4_MLA_TOKENS_PER_BLOCK)),
        )
        if num_contexts > 0:
            max_context_len = int(kv_lens[:num_contexts].max().item())
            max_context_pages = _ceil_div(
                max_context_len,
                FP4_MLA_TOKENS_PER_BLOCK,
            )
            metadata.fp4_mla_context_repack_max_touched_pages = min(
                max_pages,
                triton.next_power_of_2(max(1, max_context_pages)),
            )

    assert isinstance(block_offsets, torch.Tensor)
    assert isinstance(page_ids, torch.Tensor)
    assert isinstance(paged_kv_indptr, torch.Tensor)
    assert isinstance(paged_kv_indptr_decode, torch.Tensor)
    required_page_ids = num_sequences * max_pages
    buffers_cover_table = (
        block_offsets.ndim == 4
        and block_offsets.shape[0] >= 1
        and block_offsets.shape[1] >= num_sequences
        and block_offsets.shape[2] >= 1
        and block_offsets.shape[3] >= max_pages
        and page_ids.ndim == 1
        and page_ids.numel() >= required_page_ids
        and paged_kv_indptr.ndim == 1
        and paged_kv_indptr.numel() >= num_sequences + 1
        and paged_kv_indptr_decode.ndim == 1
        and paged_kv_indptr_decode.numel() >= num_generation_sequences + 1
    )
    if not buffers_cover_table:
        return False
    if metadata._fp4_mla_hp_page_indices.numel() < required_page_ids:
        return False

    metadata._fp4_mla_device_page_table = True
    metadata.fp4_mla_page_table_stride = max_pages
    metadata.num_blocks = None
    metadata.num_context_blocks = num_contexts * max_pages
    metadata.num_generation_blocks = num_generation_sequences * max_pages
    return True


def materialize_fp4_mla_device_page_table(
    metadata: Any,
    kv_lens: torch.Tensor,
    generation_kv_lens: Optional[torch.Tensor] = None,
) -> None:
    """Refresh the fixed-stride context and generation page table once per forward."""
    if not bool(getattr(metadata, "_fp4_mla_device_page_table", False)):
        raise RuntimeError("FP4 MLA requires fixed-stride device page metadata.")
    if bool(getattr(metadata, "_fp4_mla_device_page_table_valid", False)):
        return

    num_contexts = int(metadata.num_contexts)
    num_sequences = int(metadata.num_seqs)
    num_generation_sequences = num_sequences - num_contexts
    max_pages = int(metadata.fp4_mla_page_table_stride)
    if num_sequences <= 0 or max_pages <= 0:
        raise RuntimeError(
            "FP4 MLA device page metadata requires positive sequence and page capacities."
        )
    if (
        kv_lens.dtype != torch.int32
        or not kv_lens.is_cuda
        or kv_lens.ndim != 1
        or kv_lens.stride(0) != 1
        or kv_lens.numel() < num_sequences
    ):
        raise ValueError(
            "FP4 MLA device page metadata requires a contiguous CUDA int32 "
            f"KV-length tensor with at least {num_sequences} entries."
        )

    if generation_kv_lens is None:
        generation_kv_lens = kv_lens[num_contexts:num_sequences]
    if (
        generation_kv_lens.dtype != torch.int32
        or not generation_kv_lens.is_cuda
        or generation_kv_lens.ndim != 1
        or generation_kv_lens.stride(0) != 1
        or generation_kv_lens.numel() < num_generation_sequences
    ):
        raise ValueError(
            "FP4 MLA device page metadata requires a contiguous CUDA int32 "
            "generation KV-length tensor with at least "
            f"{num_generation_sequences} entries."
        )

    cache_pool_id = int(getattr(metadata, "_fp4_mla_cache_pool_id", 0))
    block_offsets = metadata.kv_cache_block_offsets[
        cache_pool_id,
        :num_sequences,
        0,
        :max_pages,
    ]
    page_ids = metadata._paged_kv_indices[: num_sequences * max_pages]
    page_index_scale = int(metadata._fp4_mla_cache_page_index_scale)
    if page_index_scale <= 0:
        raise RuntimeError("FP4 MLA device page metadata requires a positive page-index scale.")
    grid = (
        num_sequences,
        triton.cdiv(max_pages, _FP4_MLA_PAGE_TABLE_TILE_SIZE),
    )
    _fp4_mla_materialize_page_table_kernel[grid](
        page_ids,
        metadata._paged_kv_indptr,
        metadata.paged_kv_indptr_decode,
        block_offsets,
        kv_lens,
        generation_kv_lens,
        block_offsets.stride(0),
        num_sequences,
        num_contexts,
        max_pages,
        PAGE_SIZE=metadata.page_size,
        PAGE_INDEX_SCALE=page_index_scale,
        PAGE_TILE_SIZE=_FP4_MLA_PAGE_TABLE_TILE_SIZE,
        num_warps=4,
    )
    hp_page_ids = getattr(metadata, "_fp4_mla_hp_page_indices", None)
    if not isinstance(hp_page_ids, torch.Tensor):
        raise RuntimeError("FP4 MLA requires an HP page-table output tensor.")
    hp_pool_id = int(metadata._fp4_mla_hp_pool_id)
    hp_page_index_scale = int(metadata._fp4_mla_hp_page_index_scale)
    hp_block_offsets = metadata.kv_cache_block_offsets[
        hp_pool_id,
        :num_sequences,
        0,
        :max_pages,
    ]
    _fp4_mla_materialize_page_table_kernel[grid](
        hp_page_ids,
        metadata._paged_kv_indptr,
        metadata.paged_kv_indptr_decode,
        hp_block_offsets,
        kv_lens,
        generation_kv_lens,
        hp_block_offsets.stride(0),
        num_sequences,
        num_contexts,
        max_pages,
        PAGE_SIZE=metadata.page_size,
        PAGE_INDEX_SCALE=hp_page_index_scale,
        PAGE_TILE_SIZE=_FP4_MLA_PAGE_TABLE_TILE_SIZE,
        num_warps=4,
    )
    metadata._fp4_mla_device_page_table_valid = True


@triton.jit
def _cutedsl_swizzled_sf_offset(row_idx, col_idx, sf_cols: tl.constexpr):
    padded_cols = ((sf_cols + 3) // 4) * 4
    return (
        col_idx % 4
        + (col_idx // 4) * (4 * 128)
        + (row_idx % 32) * 16
        + ((row_idx % 128) // 32) * 4
        + (row_idx // 128) * (128 * padded_cols)
    )


@triton.jit
def _cutedsl_pad_q_and_sf_kernel(
    q_padded_ptr,
    q_ptr,
    q_sf_padded_ptr,
    q_sf_ptr,
    num_heads,
    output_heads: tl.constexpr,
    packed_dim: tl.constexpr,
    block_bytes: tl.constexpr,
    sf_cols: tl.constexpr,
    sf_cols_per_byte_block: tl.constexpr,
):
    query_idx = tl.program_id(0)
    byte_block = tl.program_id(1)
    head_offsets = tl.arange(0, output_heads)
    byte_offsets = byte_block * block_bytes + tl.arange(0, block_bytes)
    head_mask = head_offsets < num_heads
    byte_mask = byte_offsets < packed_dim
    source_rows = query_idx * num_heads + head_offsets
    destination_rows = query_idx * output_heads + head_offsets
    values = tl.load(
        q_ptr + source_rows[:, None] * packed_dim + byte_offsets[None, :],
        mask=head_mask[:, None] & byte_mask[None, :],
        other=0,
    )
    tl.store(
        q_padded_ptr + destination_rows[:, None] * packed_dim + byte_offsets[None, :],
        values,
        mask=byte_mask[None, :],
    )
    sf_col_offsets = byte_block * sf_cols_per_byte_block + tl.arange(0, sf_cols_per_byte_block)
    sf_col_mask = sf_col_offsets < sf_cols
    source_offsets = _cutedsl_swizzled_sf_offset(
        source_rows[:, None], sf_col_offsets[None, :], sf_cols
    )
    destination_offsets = _cutedsl_swizzled_sf_offset(
        destination_rows[:, None], sf_col_offsets[None, :], sf_cols
    )
    sf_values = tl.load(
        q_sf_ptr + source_offsets,
        mask=head_mask[:, None] & sf_col_mask[None, :],
        other=1.0,
    )
    tl.store(
        q_sf_padded_ptr + destination_offsets,
        sf_values,
        mask=sf_col_mask[None, :],
    )


_SM_COUNT_CACHE: dict[int, int] = {}


def _get_sm_count(device: torch.device) -> int:
    """Return the SM (multiprocessor) count for ``device``, cached per index."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    count = _SM_COUNT_CACHE.get(index)
    if count is None:
        count = torch.cuda.get_device_properties(index).multi_processor_count
        _SM_COUNT_CACHE[index] = count
    return count


def _validate_fp4_mla_context_rope(
    latent_cache: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    v_head_dim: int,
) -> int:
    head_dim = latent_cache.shape[-1]
    rope_dim = head_dim - v_head_dim
    if rope_dim <= 0 or rope_dim % 2 != 0:
        raise ValueError(
            "FP4 MLA fused context K-RoPE requires a positive even RoPE dimension, "
            f"got head_dim={head_dim}, v_head_dim={v_head_dim}."
        )
    if rotary_cos_sin.device != latent_cache.device:
        raise ValueError("FP4 MLA context latent cache and RoPE table must use the same device.")
    if rotary_cos_sin.dtype != torch.float32:
        raise TypeError(
            f"FP4 MLA fused context K-RoPE requires a float32 table, got {rotary_cos_sin.dtype}."
        )
    if not rotary_cos_sin.is_contiguous():
        raise ValueError("FP4 MLA fused context K-RoPE requires a contiguous RoPE table.")
    table_row_size = rope_dim * 2
    if rotary_cos_sin.numel() < table_row_size or rotary_cos_sin.numel() % table_row_size != 0:
        raise ValueError(
            "FP4 MLA context RoPE table size must be a positive multiple of "
            f"{table_row_size}, got {rotary_cos_sin.numel()}."
        )
    return rope_dim


def _host_int_list_during_forward(value: Any, start: int, end: int) -> Optional[list[int]]:
    if torch.cuda.is_current_stream_capturing():
        return None
    return _host_int_list(value, start, end)


# FP4 MLA scale-layout helpers


def get_fp4_mla_v_scale_pool_size(v_head_dim: int, page_size: int) -> int:
    """Return elements per page for the swizzled FP4 MLA V-scale pool.

    The PV matmul treats V as a RHS matrix shaped ``[v_head_dim, kv_tokens]``.
    NVFP4 block scales therefore group along the token/K axis, not along the
    latent dimension as the K-view cache does.  The physical layout matches the
    Triton block-scaled matmul scale layout:
    ``[ceil(v_head_dim / 128), ceil(page_size / 16 / 4), 32, 16]``.
    """
    return _get_fp4_mla_swizzled_scale_size(v_head_dim, page_size)


def _get_fp4_mla_swizzled_scale_size(rows: int, cols: int) -> int:
    scale_cols = _ceil_div(cols, FP4_BLOCK_SIZE)
    row_groups = _ceil_div(rows, FP4_MLA_SCALE_ROW_GROUP)
    col_groups = _ceil_div(scale_cols, FP4_MLA_SCALE_COL_GROUP)
    return row_groups * col_groups * 32 * 16


def can_fuse_fp4_mla_q_quant(
    metadata: Any,
    q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
) -> bool:
    """Return whether generation can quantize Q in the fused cache update."""
    num_gen = metadata.num_seqs - metadata.num_contexts
    return bool(
        _fp4_mla_attention_backend() in _FP4_MLA_K_RESIDUAL_BACKENDS
        and get_sm_version() == 107
        and num_gen > 0
        and getattr(metadata, "kv_cache_manager", None) is not None
        and q.shape[0] > 0
        and q.shape[0] % num_gen == 0
        and q.is_cuda
        and q_pe.is_cuda
        and latent_cache.is_cuda
        and q.device == q_pe.device == latent_cache.device
        and q.dtype == torch.bfloat16
        and q.is_contiguous()
        and q.ndim == 3
        and 0 < q.shape[1] <= 128
        and q.shape[2] == FP4_MLA_Q_PREFIX_DIM + FP4_MLA_Q_RESIDUAL_DIM
        and q_pe.dtype == torch.bfloat16
        and tuple(q_pe.shape) == (q.shape[0], q.shape[1], FP4_MLA_Q_RESIDUAL_DIM)
        and latent_cache.dtype == torch.bfloat16
        and tuple(latent_cache.shape) == (q.shape[0], FP4_MLA_Q_PREFIX_DIM + FP4_MLA_Q_RESIDUAL_DIM)
        and metadata.page_size == FP4_MLA_TOKENS_PER_BLOCK
    )


def _prepare_fp4_mla_q_buffers(
    metadata: Any,
    num_queries: int,
    num_heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return manager-owned, fixed-capacity packed-Q staging buffers."""
    if num_queries <= 0:
        raise ValueError(f"FP4 MLA Q buffer preparation needs queries, got {num_queries}.")
    owner = getattr(metadata, "kv_cache_manager", None)
    if owner is None:
        raise RuntimeError("Fused FP4 MLA Q quantization requires a KV cache manager.")
    if num_heads <= 0 or num_heads > 128:
        raise ValueError(f"FP4 MLA Q buffers require 1-128 local heads, got {num_heads}.")
    buffers = getattr(owner, "_fp4_mla_q_buffers", None)
    if buffers is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Cannot create FP4 MLA Q buffers while capturing a CUDA graph. "
                "Run a warmup forward first."
            )
        buffers = {}
        setattr(owner, "_fp4_mla_q_buffers", buffers)

    max_num_tokens = int(getattr(metadata, "max_num_tokens", num_queries) or num_queries)
    max_num_sequences = int(
        getattr(metadata, "max_num_sequences", None)
        or getattr(metadata, "max_num_requests", num_queries)
        or num_queries
    )
    max_query_width = 1 + int(getattr(metadata, "max_total_draft_tokens", None) or 0)
    capacity = min(
        max_num_tokens,
        max_num_sequences * max_query_width,
        _FP4_MLA_MAX_GRID_Z,
    )
    if num_queries > capacity:
        raise ValueError(
            f"FP4 MLA active queries exceed the configured Q capacity: {num_queries} > {capacity}."
        )

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    canonical_device = torch.device("cuda", device_index)
    expected_q_shape = (capacity * num_heads, FP4_MLA_Q_PACKED_DIM)
    expected_q_sf_shape = (
        _get_fp4_mla_swizzled_scale_size(capacity * num_heads, FP4_MLA_Q_LOGICAL_DIM),
    )
    q_key = f"q_{num_heads}"
    q_sf_key = f"q_sf_{num_heads}"
    q_storage = buffers.get(q_key)
    q_sf_storage = buffers.get(q_sf_key)
    if q_storage is None and q_sf_storage is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Cannot allocate FP4 MLA Q buffers while capturing a CUDA graph. "
                "Run a warmup forward first."
            )
        q_storage = torch.empty(expected_q_shape, dtype=torch.uint8, device=canonical_device)
        q_sf_storage = torch.empty(
            expected_q_sf_shape,
            dtype=torch.float8_e4m3fn,
            device=canonical_device,
        )
        buffers[q_key] = q_storage
        buffers[q_sf_key] = q_sf_storage
    if (
        q_storage is None
        or q_storage.dtype != torch.uint8
        or q_storage.device != canonical_device
        or tuple(q_storage.shape) != expected_q_shape
        or not q_storage.is_contiguous()
        or q_sf_storage is None
        or q_sf_storage.dtype != torch.float8_e4m3fn
        or q_sf_storage.device != canonical_device
        or tuple(q_sf_storage.shape) != expected_q_sf_shape
        or not q_sf_storage.is_contiguous()
    ):
        raise RuntimeError("FP4 MLA packed-Q buffers do not match the configured capacity.")
    return q_storage, q_sf_storage, capacity


def _get_fp4_mla_context_start_positions(metadata: Any, num_contexts: int) -> torch.Tensor:
    kv_cache_params = getattr(metadata, "kv_cache_params", None)
    cached_token_lens = getattr(kv_cache_params, "num_cached_tokens_per_seq", None)
    if cached_token_lens is not None:
        return torch.as_tensor(cached_token_lens[:num_contexts], dtype=torch.int64, device="cpu")

    return (
        (
            metadata.kv_lens_cuda_runtime[:num_contexts]
            - metadata.prompt_lens_cuda_runtime[:num_contexts]
        )
        .detach()
        .cpu()
    )


def _validate_fp4_mla_context_start_alignment(
    metadata: Any,
    num_contexts: int,
    *,
    alignment: int = HP_BLOCK_SIZE,
) -> None:
    context_start_positions = _get_fp4_mla_context_start_positions(metadata, num_contexts)
    bad_start = (context_start_positions < 0) | ((context_start_positions % alignment) != 0)
    if bool(torch.any(bad_start).item()):
        starts = context_start_positions.detach().cpu().tolist()
        raise ValueError(
            "FP4 MLA shared-tile context update requires every context "
            f"start position to be {alignment}-token aligned, got "
            f"start positions {starts}."
        )


def get_fp4_mla_v_scale_pool_shape(
    num_layers: int,
    num_pages: int,
    v_head_dim: int,
    page_size: int,
) -> tuple[int, int, int, int, int, int]:
    """Return the logical swizzled V-scale view shape.

    The leading dimensions are ``[layer, physical_page]``.  The remaining
    dimensions are the preshuffled ``[N // 128, K // 16 // 4, 32, 16]`` shape
    consumed by Triton block-scaled matmul for the V/PV RHS operand.
    """
    token_scale_cols = _ceil_div(page_size, FP4_BLOCK_SIZE)
    return (
        num_layers,
        num_pages,
        _ceil_div(v_head_dim, FP4_MLA_SCALE_ROW_GROUP),
        _ceil_div(token_scale_cols, FP4_MLA_SCALE_COL_GROUP),
        32,
        16,
    )


def get_fp4_mla_v_scale_pool_view(
    metadata: Any,
    *,
    v_head_dim: int,
) -> torch.Tensor:
    """View the auxiliary MLA V-scale pool in Triton's block-scaled layout."""
    pool = getattr(metadata, "fp4_mla_v_scale_pool", None)
    if pool is None:
        raise RuntimeError("FP4 MLA V scale pool is not allocated.")

    elems_per_page = get_fp4_mla_v_scale_pool_size(v_head_dim, metadata.page_size)
    if pool.shape[-1] < elems_per_page:
        raise RuntimeError(
            f"FP4 MLA V scale pool page stride is too small: got "
            f"{pool.shape[-1]}, need {elems_per_page}."
        )

    token_scale_cols = _ceil_div(metadata.page_size, FP4_BLOCK_SIZE)
    col_groups = _ceil_div(token_scale_cols, FP4_MLA_SCALE_COL_GROUP)
    shape = get_fp4_mla_v_scale_pool_shape(
        pool.shape[0], pool.shape[1], v_head_dim, metadata.page_size
    )
    strides = (
        pool.stride(0),
        pool.stride(1),
        col_groups * 32 * 16,
        32 * 16,
        16,
        1,
    )
    return torch.as_strided(pool, size=shape, stride=strides)


# Python launch helpers


def _get_fp4_mla_global_scale(metadata: Any, device: torch.device) -> torch.Tensor:
    global_scale = getattr(metadata, "_fp4_mla_kv_global_scale", None)
    if (
        not isinstance(global_scale, torch.Tensor)
        or global_scale.device != device
        or global_scale.dtype != torch.float32
        or global_scale.numel() != 1
    ):
        raise RuntimeError("FP4 MLA requires a preallocated FP32 KV global-scale tensor.")
    return global_scale


def _get_fp4_mla_q_global_scale(metadata: Any, device: torch.device) -> torch.Tensor:
    global_scale = getattr(metadata, "_fp4_mla_q_global_scale", None)
    if (
        not isinstance(global_scale, torch.Tensor)
        or global_scale.device != device
        or global_scale.dtype != torch.float32
        or global_scale.numel() != 1
    ):
        raise RuntimeError("FP4 MLA requires a preallocated FP32 Q global-scale tensor.")
    return global_scale


def _get_fp4_mla_kv_cache_tensors(
    metadata: Any, layer_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return metadata.kv_cache_manager.get_fp4_mla_cache_buffers(layer_idx)


def _get_fp4_mla_hp_pool_layout(
    metadata: Any,
    pool: torch.Tensor,
) -> tuple[int, int]:
    """Return the manager-owned HP ring size and per-token head dimension."""
    manager = getattr(metadata, "kv_cache_manager", None)
    if manager is None or not hasattr(manager, "_fp4_mla_hp_pool_size"):
        raise ValueError("FP4 MLA requires a V2 manager-owned HP ring.")
    hp_pool_size = int(manager._fp4_mla_hp_pool_size)
    if (
        hp_pool_size < HP_BLOCK_SIZE
        or pool.ndim != 4
        or pool.shape[2] < 1
        or pool.shape[-1] % hp_pool_size != 0
    ):
        raise ValueError(
            "FP4 MLA high-precision KV pool does not match its configured "
            f"ring: shape={tuple(pool.shape)}, ring_size={hp_pool_size}."
        )
    return hp_pool_size, pool.shape[-1] // hp_pool_size


def _validate_fp4_mla_hp_generation_width(
    hp_pool_size: int,
    generation_len: int,
) -> None:
    """Ensure one target plus rewindable drafts fit without clobbering the live tail."""
    max_rewind_len = hp_pool_size - HP_BLOCK_SIZE
    if generation_len <= 0 or generation_len - 1 > max_rewind_len:
        raise RuntimeError(
            "FP4 MLA generation exceeds the HP ring's rewind slack: "
            f"generation={generation_len}, max_rewind={max_rewind_len}."
        )


def _validate_fp4_mla_kv_storage_shape(
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    *,
    head_dim: int,
    backend: str,
) -> int:
    """Validate the backend-specific physical KV and scale strides."""
    residual_dim = FP4_MLA_K_RESIDUAL_DIM if backend in _FP4_MLA_K_RESIDUAL_BACKENDS else 0
    expected_storage_head_dim = head_dim + residual_dim
    storage_head_dim = kv_cache.shape[-1] * 2
    if storage_head_dim != expected_storage_head_dim:
        raise RuntimeError(
            "FP4 MLA KV cache storage head dimension does not match the selected backend: "
            f"got {storage_head_dim}, expected {expected_storage_head_dim}. Recreate the engine "
            f"after setting {FP4_MLA_ATTENTION_BACKEND_ENV}."
        )

    expected_scale_columns = expected_storage_head_dim // FP4_BLOCK_SIZE
    if sf_cache.shape[-1] != expected_scale_columns:
        raise RuntimeError(
            "FP4 MLA KV cache scale storage does not match the contiguous data layout: "
            f"got {sf_cache.shape[-1]} columns, expected {expected_scale_columns}."
        )
    return storage_head_dim


def _scatter_fp4_mla_kv_cache_2d_context(
    metadata: Any,
    latent_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_sf: torch.Tensor,
    global_scale: torch.Tensor,
    rotary_cos_sin: Optional[torch.Tensor],
    *,
    token_offset: int,
    local_layer: int,
    v_head_dim: int,
    head_dim: int,
    num_tokens: int,
    num_dim_blocks: int,
    sf_per_token: int,
    sf_per_page: int,
    v_packed_base: Optional[torch.Tensor] = None,
    v_page_offset: int = 0,
) -> bool:
    num_contexts = metadata.num_contexts
    if num_contexts > 0:
        prompt_lens_cpu = metadata.prompt_lens_cpu_runtime[:num_contexts]
        ctx_token_count = int(prompt_lens_cpu.sum().item())
        if num_tokens != ctx_token_count:
            raise RuntimeError(
                f"FP4 MLA 2D context scatter needs {ctx_token_count} context tokens, got "
                f"{num_tokens}."
            )
        _validate_fp4_mla_context_start_alignment(metadata, num_contexts, alignment=FP4_BLOCK_SIZE)

    apply_k_rope = rotary_cos_sin is not None
    rope_dim = (
        _validate_fp4_mla_context_rope(latent_cache, rotary_cos_sin, v_head_dim)
        if rotary_cos_sin is not None
        else 0
    )
    rotary_cos_sin_ptr = rotary_cos_sin if rotary_cos_sin is not None else latent_cache

    hp_pool = getattr(metadata, "high_precision_kv_pool", None)
    if not isinstance(hp_pool, torch.Tensor):
        raise TypeError("FP4 MLA high-precision KV pool must be a tensor.")
    if hp_pool.device != latent_cache.device:
        raise ValueError("FP4 MLA latent cache and high-precision pool must share a device.")
    if hp_pool.dtype != torch.bfloat16:
        raise TypeError(f"FP4 MLA high-precision KV pool must use BF16, got {hp_pool.dtype}.")
    hp_pool_size, pool_head_dim = _get_fp4_mla_hp_pool_layout(metadata, hp_pool)
    if pool_head_dim < head_dim:
        raise RuntimeError(
            f"FP4 MLA HP pool head dimension is too small: got "
            f"{pool_head_dim}, need at least {head_dim}."
        )
    if local_layer < 0 or local_layer >= hp_pool.shape[1]:
        raise ValueError(
            f"FP4 MLA local layer {local_layer} is outside the HP pool's {hp_pool.shape[1]} layers."
        )
    if hp_pool.stride(-1) != 1:
        raise ValueError("FP4 MLA high-precision KV pool must be contiguous in head_dim.")
    hp_page_ids = metadata._fp4_mla_hp_page_indices
    if not isinstance(hp_page_ids, torch.Tensor):
        raise RuntimeError("FP4 MLA context cache update requires HP page metadata.")
    store_hp_tail = num_contexts > 0
    num_hp_pages = hp_pool.shape[0]
    pool_s0 = hp_pool.stride(0)
    pool_s1 = hp_pool.stride(1)

    write_v_packed = v_packed_base is not None
    v_packed_output = v_packed_base if write_v_packed else kv_cache
    v_packed_s0 = v_packed_output.stride(0) if write_v_packed else 0
    v_packed_s1 = v_packed_output.stride(1) if write_v_packed else 0

    _fp4_mla_context_cache_update_kernel[
        (
            num_tokens,
            num_dim_blocks,
        )
    ](
        kv_cache,
        sf_cache,
        v_sf,
        v_packed_output,
        latent_cache,
        global_scale,
        rotary_cos_sin_ptr,
        hp_pool,
        hp_page_ids,
        metadata.batch_indices,
        metadata.positions,
        metadata.paged_kv_indices,
        metadata.paged_kv_indptr,
        metadata.paged_kv_indices.shape[0],
        metadata.paged_kv_indptr.shape[0],
        metadata.batch_indices.shape[0],
        v_sf.shape[1],
        v_sf.shape[0],
        num_contexts,
        num_hp_pages,
        token_offset,
        num_tokens,
        local_layer,
        v_page_offset if write_v_packed else 0,
        metadata.page_size,
        kv_cache.stride(0),
        kv_cache.stride(2),
        kv_cache.stride(4),
        sf_cache.stride(0),
        latent_cache.stride(0),
        latent_cache.stride(1),
        v_sf.stride(0),
        v_sf.stride(1),
        v_packed_s0,
        v_packed_s1,
        pool_s0,
        pool_s1,
        HEAD_D=head_dim,
        V_HEAD_D=v_head_dim,
        HP_BLOCK=FP4_BLOCK_SIZE,
        HP_POOL_SIZE=hp_pool_size,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        SF_PER_TOKEN=sf_per_token,
        SF_PER_PAGE=sf_per_page,
        K_RESIDUAL_D=FP4_MLA_K_RESIDUAL_DIM,
        STORE_K_RESIDUAL=(_fp4_mla_attention_backend() in _FP4_MLA_K_RESIDUAL_BACKENDS),
        ROPE_DIM=rope_dim,
        APPLY_K_ROPE=apply_k_rope,
        POOL_HEAD_D=pool_head_dim,
        STORE_HP_TAIL=store_hp_tail,
        WRITE_V_PACKED=write_v_packed,
    )
    return store_hp_tail


def _fp4_mla_generation_num_blocks_device(metadata: Any) -> torch.Tensor:
    """Device-side scalar view holding the generation page-table capacity.

    ``paged_kv_indptr_decode[num_gen]`` is the fixed-stride generation-table
    endpoint. Device kernels combine it with the live KV lengths, so inactive
    slots are never consumed.
    """
    num_gen = metadata.num_seqs - metadata.num_contexts
    return metadata.paged_kv_indptr_decode[num_gen : num_gen + 1]


def _fp4_mla_uniform_generation_lengths(
    metadata: Any, num_gen_tokens: int, num_gen: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return preallocated CUDA generation KV and append lengths."""
    if num_gen <= 0 or num_gen_tokens % num_gen != 0:
        raise RuntimeError("FP4 MLA generation requires a non-empty uniform request batch.")

    num_contexts = metadata.num_contexts
    num_seqs = metadata.num_seqs
    kv_lens_gen = metadata.kv_lens_cuda_runtime[num_contexts:num_seqs]
    prompt_lens_gen = metadata.prompt_lens_cuda_runtime[num_contexts:num_seqs]
    corrected_kv_lens = getattr(metadata, "fp4_mla_generation_kv_lens", None)
    generation_lens = getattr(metadata, "fp4_mla_generation_append_lens", None)
    tensors = (
        kv_lens_gen,
        prompt_lens_gen,
        corrected_kv_lens,
        generation_lens,
    )
    if (
        not all(isinstance(tensor, torch.Tensor) for tensor in tensors)
        or corrected_kv_lens.numel() < num_gen
        or generation_lens.numel() < num_gen
        or not all(tensor.is_cuda for tensor in tensors)
    ):
        raise RuntimeError("FP4 MLA generation lengths require preallocated CUDA buffers.")

    record_for_capture = bool(
        getattr(metadata, "is_cuda_graph", False)
        and torch.cuda.is_current_stream_capturing()
        and not getattr(
            metadata,
            "_fp4_mla_generation_lengths_capture_recorded",
            False,
        )
    )
    precomputed = (
        not record_for_capture
        and metadata.fp4_mla_generation_lengths_num_tokens == num_gen_tokens
        and metadata.fp4_mla_generation_lengths_num_seqs == num_gen
        and metadata.fp4_mla_generation_lengths_num_contexts == num_contexts
    )
    if not precomputed:
        populate_fp4_mla_generation_lengths(
            kv_lens_gen,
            prompt_lens_gen,
            corrected_kv_lens[:num_gen],
            generation_lens[:num_gen],
            num_gen_tokens=num_gen_tokens,
            num_gen=num_gen,
        )
        metadata.fp4_mla_generation_lengths_num_tokens = num_gen_tokens
        metadata.fp4_mla_generation_lengths_num_seqs = num_gen
        metadata.fp4_mla_generation_lengths_num_contexts = num_contexts
        if record_for_capture:
            metadata._fp4_mla_generation_lengths_capture_recorded = True
    return corrected_kv_lens[:num_gen], generation_lens[:num_gen]


def _materialize_fp4_mla_device_page_table_for_forward(
    metadata: Any,
    generation_kv_lens: Optional[torch.Tensor] = None,
) -> None:
    """Materialize all fixed-stride rows from final per-forward device lengths."""
    if not bool(getattr(metadata, "_fp4_mla_device_page_table", False)):
        raise RuntimeError("FP4 MLA cache update requires fixed-stride device page metadata.")
    if bool(getattr(metadata, "_fp4_mla_device_page_table_valid", False)):
        return

    num_contexts = int(metadata.num_contexts)
    num_sequences = int(metadata.num_seqs)
    num_generation_sequences = num_sequences - num_contexts
    if generation_kv_lens is None:
        if num_generation_sequences > 0:
            num_generation_tokens = int(metadata.num_tokens) - int(metadata.num_ctx_tokens)
            generation_kv_lens, _ = _fp4_mla_uniform_generation_lengths(
                metadata,
                num_generation_tokens,
                num_generation_sequences,
            )
        else:
            generation_kv_lens = metadata.kv_lens_cuda_runtime[num_contexts:num_sequences]
    materialize_fp4_mla_device_page_table(
        metadata,
        metadata.kv_lens_cuda_runtime[:num_sequences],
        generation_kv_lens,
    )


def _scatter_fp4_mla_kv_cache_2d_generation(
    metadata: Any,
    latent_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_sf: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    token_offset: int,
    local_layer: int,
    v_head_dim: int,
    head_dim: int,
    num_tokens: int,
    num_dim_blocks: int,
    sf_per_token: int,
    sf_per_page: int,
    rotary_cos_sin: torch.Tensor,
    q_pe: torch.Tensor,
    q_rope_out: torch.Tensor,
    q_quant_input: torch.Tensor,
    q_fp4_out: torch.Tensor,
    q_sf_out: torch.Tensor,
    v_packed_base: Optional[torch.Tensor],
    v_page_offset: int,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    num_contexts = metadata.num_contexts
    num_seqs = metadata.num_seqs
    num_gen = num_seqs - num_contexts
    if num_gen <= 0:
        return
    if num_tokens < num_gen:
        raise RuntimeError(
            f"FP4 MLA 2D generation scatter needs at least {num_gen} generation "
            f"tokens, got {num_tokens}."
        )
    if num_tokens % num_gen != 0:
        raise NotImplementedError(
            "FP4 MLA no-dequant generation scatter requires a uniform linear MTP "
            f"generation length, got {num_tokens} tokens for {num_gen} sequences."
        )
    # The prompt_lens/kv_lens runtime aliases can lag at the decode anchor
    # (seq_lens == 1) under CUDA graph / one-engine MTP while each generation
    # sequence really appends num_tokens // num_gen tokens this step. Recover the
    # true per-sequence lengths for the no-dequant kernel below (a no-op when the
    # aliases already match).
    kv_lens_gen, gen_lens_gen = _fp4_mla_uniform_generation_lengths(metadata, num_tokens, num_gen)
    _materialize_fp4_mla_device_page_table_for_forward(metadata, kv_lens_gen)

    pool = getattr(metadata, "high_precision_kv_pool", None)
    if pool is None:
        raise RuntimeError("FP4 MLA 2D generation scatter requires the HP KV pool.")
    try:
        hp_pool_size, hp_head_dim = _get_fp4_mla_hp_pool_layout(metadata, pool)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if hp_head_dim < head_dim:
        raise RuntimeError(
            f"FP4 MLA 2D generation scatter needs at least {head_dim} HP channels, got "
            f"{hp_head_dim}."
        )
    hp_page_ids = _fp4_mla_generation_hp_page_ids(metadata, num_gen)
    if not isinstance(hp_page_ids, torch.Tensor):
        raise RuntimeError("FP4 MLA generation requires HP page metadata.")
    num_hp_pages = pool.shape[0]

    max_gen_len = num_tokens // num_gen
    _validate_fp4_mla_hp_generation_width(hp_pool_size, max_gen_len)
    max_rewind_len = hp_pool_size - HP_BLOCK_SIZE
    page_ids = _fp4_mla_generation_page_ids(metadata, num_gen)
    rope_dim = head_dim - v_head_dim
    block_q_heads = 32
    q1_kv_blocks_per_program = 1
    # Grouped Q1 kernels reuse K's packed codes for V. Keep one dimension
    # block per program until their warp-specialized V quantizer is split out.
    if max_gen_len == 1:
        q1_kv_blocks_per_program = _fp4_mla_q1_kv_blocks_per_program(num_gen, v_head_dim)
    q_prefix_block_dim = (
        FP4_MLA_Q1_PREFIX_BLOCK_DIM if max_gen_len == 1 else FP4_MLA_Q_PREFIX_BLOCK_DIM
    )
    q_prefix_blocks = FP4_MLA_Q_PREFIX_DIM // q_prefix_block_dim
    q_prefix_blocks_per_program = _fp4_mla_q1_prefix_blocks_per_program(
        num_gen,
        q1_kv_blocks_per_program,
    )
    q_work_blocks = q_prefix_blocks // q_prefix_blocks_per_program + 1
    if latent_cache.dtype != torch.bfloat16:
        raise TypeError(
            "Fused FP4 MLA Q/K RoPE and cache storage requires BF16 latent KV, "
            f"got {latent_cache.dtype}."
        )
    if rope_dim <= 0 or rope_dim % 2 != 0:
        raise ValueError(
            "Fused FP4 MLA K RoPE requires a positive even K tail, "
            f"got head_dim={head_dim} v_head_dim={v_head_dim}."
        )
    if (
        rotary_cos_sin is None
        or rotary_cos_sin.device != latent_cache.device
        or rotary_cos_sin.dtype != torch.float32
        or rotary_cos_sin.numel() % (rope_dim * 2) != 0
    ):
        raise ValueError(
            "Fused FP4 MLA K RoPE requires a same-device FP32 rotary "
            f"table with rows of {rope_dim * 2} values."
        )
    if (
        q_pe is None
        or q_rope_out is None
        or q_pe.dtype != torch.bfloat16
        or q_rope_out.dtype != torch.bfloat16
        or q_pe.device != latent_cache.device
        or q_rope_out.device != latent_cache.device
        or q_pe.ndim != 3
        or q_rope_out.shape != q_pe.shape
        or q_pe.shape[0] != num_tokens
        or q_pe.shape[1] <= 0
        or q_pe.shape[2] != rope_dim
    ):
        raise ValueError(
            "Fused FP4 MLA Q RoPE requires same-device BF16 q_pe and "
            f"q_rope_out tensors shaped [tokens, heads, {rope_dim}]."
        )
    num_q_heads = q_pe.shape[1]
    q_head_blocks = _ceil_div(num_q_heads, block_q_heads)
    max_gen_tiles = _ceil_div(max_gen_len + FP4_BLOCK_SIZE - 1, FP4_BLOCK_SIZE)
    rotary_table = rotary_cos_sin
    q_global_scale = _get_fp4_mla_q_global_scale(metadata, latent_cache.device)
    q_pe_input = q_pe
    q_rope_output = q_rope_out
    q_full_input = q_quant_input
    q_fp4_output = q_fp4_out
    q_sf_output = q_sf_out
    write_v_packed = v_packed_base is not None
    v_packed_output = v_packed_base if write_v_packed else kv_cache
    v_packed_s0 = v_packed_output.stride(0) if write_v_packed else 0
    v_packed_s1 = v_packed_output.stride(1) if write_v_packed else 0
    kv_work_blocks = (
        v_head_dim // FP4_BLOCK_SIZE // q1_kv_blocks_per_program + 1
        if max_gen_len == 1
        else num_dim_blocks
    )
    launch_grid = (
        num_gen,
        max(
            kv_work_blocks,
            max_gen_len * q_head_blocks * q_work_blocks,
        ),
    )
    store_k_residual = _fp4_mla_attention_backend() in _FP4_MLA_K_RESIDUAL_BACKENDS

    def launch_generation_update(
        grid: tuple[int, ...],
        *,
        page_ids_len: int,
        indptr_len: int,
        max_gen_tiles_variant: int,
        q_prefix_block_dim_variant: int,
        q_prefix_blocks_variant: int,
        q_prefix_blocks_per_program_variant: int,
        q1_kv_blocks_per_program_variant: int,
    ) -> None:
        q_work_blocks_variant = q_prefix_blocks_variant // q_prefix_blocks_per_program_variant + 1
        _fp4_mla_generation_fused_qk_rope_cache_update_kernel[grid](
            kv_cache,
            sf_cache,
            v_sf,
            v_packed_output,
            pool,
            latent_cache,
            global_scale,
            q_global_scale,
            rotary_table,
            q_pe_input,
            q_rope_output,
            q_full_input,
            q_fp4_output,
            q_sf_output,
            kv_lens_gen,
            gen_lens_gen,
            page_ids,
            hp_page_ids,
            metadata.paged_kv_indptr_decode,
            page_ids_len,
            hp_page_ids.numel(),
            indptr_len,
            v_sf.shape[1],
            num_hp_pages,
            v_sf.shape[0],
            local_layer,
            v_page_offset if write_v_packed else 0,
            metadata.page_size,
            kv_cache.stride(0),
            kv_cache.stride(2),
            kv_cache.stride(4),
            sf_cache.stride(0),
            pool.stride(0),
            pool.stride(1),
            v_sf.stride(0),
            v_sf.stride(1),
            v_packed_s0,
            v_packed_s1,
            q_pe_input.stride(0),
            q_pe_input.stride(1) if q_pe_input.ndim > 1 else 0,
            q_pe_input.stride(2) if q_pe_input.ndim > 2 else 0,
            q_rope_output.stride(0),
            q_rope_output.stride(1) if q_rope_output.ndim > 1 else 0,
            q_rope_output.stride(2) if q_rope_output.ndim > 2 else 0,
            HEAD_D=hp_head_dim,
            V_HEAD_D=v_head_dim,
            HP_BLOCK=FP4_BLOCK_SIZE,
            HP_POOL_SIZE=hp_pool_size,
            FP4_BLOCK=FP4_BLOCK_SIZE,
            SF_PER_TOKEN=sf_per_token,
            SF_PER_PAGE=sf_per_page,
            K_RESIDUAL_D=FP4_MLA_K_RESIDUAL_DIM,
            STORE_K_RESIDUAL=store_k_residual,
            FUSE_ROPE_CACHE_STORE=True,
            WRITE_V_PACKED=write_v_packed,
            MAX_GEN_TILES=max_gen_tiles_variant,
            ROPE_DIM=rope_dim,
            ROPE_PAIR_BLOCK=triton.next_power_of_2(rope_dim // 2),
            NUM_DIM_BLOCKS=num_dim_blocks,
            NUM_Q_HEADS=num_q_heads,
            Q_HEAD_BLOCKS=max(q_head_blocks, 1),
            BLOCK_Q_HEADS=block_q_heads,
            Q_PREFIX_D=FP4_MLA_Q_PREFIX_DIM,
            Q_PREFIX_BLOCK_D=q_prefix_block_dim_variant,
            Q_PREFIX_BLOCKS=q_prefix_blocks_variant,
            Q_PREFIX_BLOCKS_PER_PROGRAM=q_prefix_blocks_per_program_variant,
            Q_WORK_BLOCKS=q_work_blocks_variant,
            Q_SF_COLS=FP4_MLA_Q_SF_GROUPS,
            WRITE_Q=True,
            Q1_KV_BLOCKS_PER_PROGRAM=q1_kv_blocks_per_program_variant,
            maxnreg=56,
        )

    # Triton compiles and loads a CUDA module on first launch. Use runtime-zero
    # work here so every reachable static tuning variant is resident before
    # warmup hands the engine to serving.
    if getattr(metadata, "is_warmup", False) and not torch.cuda.is_current_stream_capturing():
        configured_generation_len = int(getattr(metadata, "max_total_draft_tokens", 0) or 0) + 1
        max_preload_generation_len = max(max_gen_len, configured_generation_len)
        if max_preload_generation_len - 1 > max_rewind_len:
            raise NotImplementedError(
                "FP4 MLA finite Triton preload exceeds the HP ring's rewind slack: "
                f"max_rewind={max_rewind_len}, generation="
                f"{max_preload_generation_len}."
            )
        max_num_sequences = int(getattr(metadata, "max_num_sequences", num_seqs) or num_seqs)
        q1_variants = _fp4_mla_q1_preload_variants(
            max_num_sequences,
            v_head_dim,
        )
        multi_token_tiles = tuple(
            sorted(
                {
                    _ceil_div(gen_len + FP4_BLOCK_SIZE - 1, FP4_BLOCK_SIZE)
                    for gen_len in range(2, max_preload_generation_len + 1)
                }
            )
        )
        preload_key = (
            "generation-cache-update",
            str(latent_cache.device),
            hp_pool_size,
            hp_head_dim,
            v_head_dim,
            num_q_heads,
            metadata.page_size,
            write_v_packed,
            store_k_residual,
            tuple(q1_variants),
            multi_token_tiles,
            tuple(kv_cache.stride()),
            tuple(sf_cache.stride()),
            tuple(pool.stride()),
            tuple(v_sf.stride()),
            tuple(v_packed_output.stride()),
            tuple(q_pe_input.stride()),
            tuple(q_rope_output.stride()),
            str(kv_cache.dtype),
            str(latent_cache.dtype),
            str(q_pe_input.dtype),
            str(q_fp4_output.dtype),
            str(q_sf_output.dtype),
        )
        preload_keys = _fp4_mla_triton_preload_key_set(metadata)
        if preload_key not in preload_keys:
            context_page_ids = getattr(metadata, "_paged_kv_indices", None)
            if not isinstance(context_page_ids, torch.Tensor):
                context_page_ids = page_ids
            context_indptr = getattr(metadata, "_paged_kv_indptr", None)
            if not isinstance(context_indptr, torch.Tensor):
                context_indptr = metadata.paged_kv_indptr_decode
            context_batch_indices = getattr(metadata, "batch_indices", None)
            if not isinstance(context_batch_indices, torch.Tensor):
                context_batch_indices = hp_page_ids
            context_positions = getattr(metadata, "positions", None)
            if not isinstance(context_positions, torch.Tensor):
                context_positions = hp_page_ids
            _fp4_mla_context_cache_update_kernel[(1, num_dim_blocks)](
                kv_cache,
                sf_cache,
                v_sf,
                v_packed_output,
                latent_cache,
                global_scale,
                rotary_table,
                pool,
                hp_page_ids,
                context_batch_indices,
                context_positions,
                context_page_ids,
                context_indptr,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                metadata.page_size,
                kv_cache.stride(0),
                kv_cache.stride(2),
                kv_cache.stride(4),
                sf_cache.stride(0),
                latent_cache.stride(0),
                latent_cache.stride(1),
                v_sf.stride(0),
                v_sf.stride(1),
                v_packed_s0,
                v_packed_s1,
                pool.stride(0),
                pool.stride(1),
                HEAD_D=head_dim,
                V_HEAD_D=v_head_dim,
                HP_BLOCK=FP4_BLOCK_SIZE,
                HP_POOL_SIZE=hp_pool_size,
                FP4_BLOCK=FP4_BLOCK_SIZE,
                SF_PER_TOKEN=sf_per_token,
                SF_PER_PAGE=sf_per_page,
                K_RESIDUAL_D=FP4_MLA_K_RESIDUAL_DIM,
                STORE_K_RESIDUAL=store_k_residual,
                ROPE_DIM=rope_dim,
                APPLY_K_ROPE=True,
                POOL_HEAD_D=hp_head_dim,
                STORE_HP_TAIL=True,
                WRITE_V_PACKED=write_v_packed,
            )
            q1_prefix_blocks = FP4_MLA_Q_PREFIX_DIM // FP4_MLA_Q1_PREFIX_BLOCK_DIM
            for q1_kv_blocks, q1_prefix_blocks_per_program in q1_variants:
                launch_generation_update(
                    (1, 1),
                    page_ids_len=0,
                    indptr_len=0,
                    max_gen_tiles_variant=1,
                    q_prefix_block_dim_variant=FP4_MLA_Q1_PREFIX_BLOCK_DIM,
                    q_prefix_blocks_variant=q1_prefix_blocks,
                    q_prefix_blocks_per_program_variant=q1_prefix_blocks_per_program,
                    q1_kv_blocks_per_program_variant=q1_kv_blocks,
                )
            multi_prefix_blocks = FP4_MLA_Q_PREFIX_DIM // FP4_MLA_Q_PREFIX_BLOCK_DIM
            for multi_token_tile in multi_token_tiles:
                launch_generation_update(
                    (1, 1),
                    page_ids_len=0,
                    indptr_len=0,
                    max_gen_tiles_variant=multi_token_tile,
                    q_prefix_block_dim_variant=FP4_MLA_Q_PREFIX_BLOCK_DIM,
                    q_prefix_blocks_variant=multi_prefix_blocks,
                    q_prefix_blocks_per_program_variant=1,
                    q1_kv_blocks_per_program_variant=1,
                )
            torch.cuda.synchronize(latent_cache.device)
            preload_keys.add(preload_key)

    launch_generation_update(
        launch_grid,
        page_ids_len=page_ids.shape[0],
        indptr_len=metadata.paged_kv_indptr_decode.shape[0],
        max_gen_tiles_variant=max(max_gen_tiles, 1),
        q_prefix_block_dim_variant=q_prefix_block_dim,
        q_prefix_blocks_variant=q_prefix_blocks,
        q_prefix_blocks_per_program_variant=q_prefix_blocks_per_program,
        q1_kv_blocks_per_program_variant=q1_kv_blocks_per_program,
    )
    return kv_lens_gen, gen_lens_gen, page_ids


# Public cache update and decode entry points


def scatter_fp4_mla_kv_cache(
    metadata: Any,
    latent_cache: torch.Tensor,
    layer_idx: int,
    *,
    token_offset: int,
    phase: _HPUpdatePhase,
    local_layer: int,
    v_head_dim: int,
    rotary_cos_sin: Optional[torch.Tensor] = None,
    q_pe: Optional[torch.Tensor] = None,
    q_rope_out: Optional[torch.Tensor] = None,
    q_quant_input: Optional[torch.Tensor] = None,
) -> bool:
    """Quantize MLA latent tokens and scatter them into the paged FP4 cache.

    Contract: this helper scatters exactly ``latent_cache.shape[0]`` tokens,
    reading index metadata at ``batch_indices[token_offset : token_offset + N]``
    and ``positions[token_offset : token_offset + N]``. Callers must pass a
    latent_cache pre-sliced to the current phase (context or generation) so
    that ``shape[0]`` matches the number of index entries they intend to
    consume. ``MLA.forward_impl`` (tensorrt_llm/_torch/modules/attention.py)
    slices ``latent_cache[:num_ctx_tokens]`` for context and
    ``latent_cache[num_ctx_tokens:]`` for generation before dispatching.

    Callers must pass ``phase``, ``local_layer``, and ``v_head_dim``. Context
    scatter writes the final FP4 tile representation directly. Dimensions
    below ``v_head_dim`` share one 16-token by 16-dim FP4 tile between K
    and V, with the scale written into K's token-major and V's dim-major
    layouts. Tail K-only dimensions use K's per-token 1D scales. For
    exclusively owned CuTeDSL pages, context scatter also writes the
    persistent packed-V sidecar.
    Context scatter can rotate the K tail directly from the unassembled latent
    tensor. Generation scatter rewrites each touched 16-token tile by reading
    old tokens from the HP pool and new tokens from ``latent_cache``. The
    static-scale generation
    specialization can also rotate Q and new K tails while updating the HP pool.
    The context kernel also stores the final incomplete tile in the BF16 HP
    pool. When ``q_quant_input`` is supplied, the generation kernel also emits
    backend-ready residual FP4 Q. The return value reports whether the current
    phase updated the HP pool.
    """
    if phase == "generation":
        metadata._fp4_mla_prequantized_q = None
        metadata._fp4_mla_prequantized_q_sf = None
        metadata._fp4_mla_q_batch_capacity = None
    if latent_cache.numel() == 0:
        raise ValueError("FP4 MLA cache scatter requires at least one latent token.")

    latent_cache = latent_cache.reshape(latent_cache.shape[0], -1).contiguous()
    num_tokens = latent_cache.shape[0]
    head_dim = latent_cache.shape[-1]
    if head_dim % FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP4 MLA KV head_dim must be divisible by {FP4_BLOCK_SIZE}, got {head_dim}."
        )
    indices_len = metadata.batch_indices.shape[0]
    positions_len = metadata.positions.shape[0]
    if token_offset + num_tokens > indices_len or token_offset + num_tokens > positions_len:
        raise RuntimeError(
            f"FP4 MLA scatter would read batch_indices[{token_offset}:"
            f"{token_offset + num_tokens}] / positions[{token_offset}:"
            f"{token_offset + num_tokens}], but only {indices_len} / "
            f"{positions_len} entries are available. This indicates "
            "latent_cache was not pre-sliced to the current phase's token "
            "range (see MLA.forward_impl)."
        )

    _validate_fp4_mla_cache_shape(metadata.page_size, head_dim)

    backend = _fp4_mla_attention_backend()
    global_scale = _get_fp4_mla_global_scale(metadata, latent_cache.device)
    kv_cache, sf_cache = _get_fp4_mla_kv_cache_tensors(metadata, layer_idx)
    storage_head_dim = _validate_fp4_mla_kv_storage_shape(
        kv_cache,
        sf_cache,
        head_dim=head_dim,
        backend=backend,
    )
    sf_per_token = storage_head_dim // FP4_BLOCK_SIZE

    if phase not in ("context", "generation"):
        raise ValueError("FP4 MLA scatter requires phase='context' or 'generation'.")
    if getattr(metadata, "fp4_mla_v_scale_pool", None) is None:
        raise RuntimeError("FP4 MLA scatter requires the auxiliary V scale pool.")
    if metadata.page_size % FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP4 MLA scatter requires page_size divisible by "
            f"{FP4_BLOCK_SIZE}, got {metadata.page_size}."
        )
    if v_head_dim > head_dim:
        raise ValueError(f"FP4 MLA v_head_dim={v_head_dim} cannot exceed head_dim={head_dim}.")
    if head_dim - v_head_dim != FP4_MLA_K_RESIDUAL_DIM:
        raise ValueError(
            "FP4 MLA K residual quantization requires the K-only tail to match "
            f"the {FP4_MLA_K_RESIDUAL_DIM}-channel residual, got "
            f"head_dim={head_dim} v_head_dim={v_head_dim}."
        )
    if v_head_dim % FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP4 MLA v_head_dim must be divisible by {FP4_BLOCK_SIZE}, got {v_head_dim}."
        )

    sf_cache = sf_cache.view(torch.float8_e4m3fn)
    v_sf = get_fp4_mla_v_scale_pool_view(metadata, v_head_dim=v_head_dim)
    num_dim_blocks = triton.cdiv(head_dim, FP4_BLOCK_SIZE)
    sf_per_page = metadata.page_size // FP4_BLOCK_SIZE

    generation_state = None
    generation_inputs = (rotary_cos_sin, q_pe, q_rope_out, q_quant_input)
    q_fp4_out = None
    q_sf_out = None
    if phase == "context":
        if any(arg is not None for arg in (q_pe, q_rope_out, q_quant_input)):
            raise ValueError("FP4 MLA context cache update does not accept generation Q tensors.")
        hp_pool_updated = False
    else:
        if not all(arg is not None for arg in generation_inputs):
            raise ValueError(
                "FP4 MLA generation requires rotary_cos_sin, q_pe, q_rope_out, "
                "and q_quant_input for fused RoPE, cache update, and Q quantization."
            )
        if not can_fuse_fp4_mla_q_quant(metadata, q_quant_input, q_pe, latent_cache):
            raise ValueError(
                "Fused FP4 MLA Q quantization received an unsupported shape, dtype, "
                "scale mode, or backend."
            )
        q_fp4_out, q_sf_out, q_batch_capacity = _prepare_fp4_mla_q_buffers(
            metadata,
            num_tokens,
            q_quant_input.shape[1],
            q_quant_input.device,
        )
        metadata._fp4_mla_prequantized_q = q_fp4_out
        metadata._fp4_mla_prequantized_q_sf = q_sf_out
        metadata._fp4_mla_q_batch_capacity = q_batch_capacity
        hp_pool_updated = True
    cutedsl_backend = _fp4_mla_attention_backend() == _FP4_MLA_CUTEDSL_BACKEND
    fused_v_transpose = cutedsl_backend and _fp4_mla_cutedsl_fused_v_transpose_enabled()
    kv_cache_manager = getattr(metadata, "kv_cache_manager", None)
    persistent_v_packed = None
    v_packed_base = None
    v_page_offset = 0
    direct_v_packed_write = False
    if cutedsl_backend and not fused_v_transpose:
        persistent_v_packed = _get_cutedsl_persistent_v_packed_cache(
            metadata,
            local_layer,
            kv_cache,
            v_head_dim=v_head_dim,
            page_size=metadata.page_size,
            block_v=FP4_MLA_SCALE_ROW_GROUP,
        )
        # Reused/imported pages may not carry this process-local sidecar. Write
        # packed V directly only when the cache pages are exclusively owned.
        # The fused generation kernel handles uniform linear-MTP batches and
        # updates every 16-token tile touched by the verification window.
        num_gen = metadata.num_seqs - metadata.num_contexts
        block_reuse = getattr(kv_cache_manager, "enable_block_reuse", True)
        direct_context_v_packed_write = (
            phase == "context"
            and metadata.num_contexts > 0
            and num_tokens > 0
            and block_reuse is False
        )
        direct_generation_v_packed_write = (
            phase == "generation"
            and num_gen > 0
            and num_tokens >= num_gen
            and num_tokens % num_gen == 0
            and block_reuse is False
        )
        direct_v_packed_write = direct_context_v_packed_write or direct_generation_v_packed_write
        if direct_v_packed_write:
            v_packed_base = _get_fp4_mla_v_packed_pool_base(metadata)
            get_v_page_offset = getattr(kv_cache_manager, "get_mla_v_packed_page_offset", None)
            v_page_offset = (
                int(get_v_page_offset(local_layer))
                if callable(get_v_page_offset)
                else local_layer * kv_cache.shape[0]
            )
            expected_row_width = metadata.page_size // 2
            required_base_rows = (v_page_offset + kv_cache.shape[0]) * v_head_dim
            if (
                not isinstance(v_packed_base, torch.Tensor)
                or v_packed_base.dtype != torch.uint8
                or v_packed_base.device != kv_cache.device
                or v_packed_base.ndim != 2
                or v_packed_base.shape[0] < required_base_rows
                or v_packed_base.shape[1] != expected_row_width
                or not v_packed_base.is_contiguous()
            ):
                raise RuntimeError(
                    "FP4 MLA direct V-packed cache update requires the "
                    "stable full-pool base to be a contiguous uint8 tensor "
                    f"with at least {required_base_rows} rows and "
                    f"{expected_row_width} columns on {kv_cache.device}."
                )
            expected_layer_ptr = v_packed_base.data_ptr() + (
                v_page_offset * v_head_dim * expected_row_width
            )
            if expected_layer_ptr != persistent_v_packed.data_ptr():
                raise RuntimeError(
                    "FP4 MLA V-packed layer view does not match its stable "
                    "full-pool base and page offset."
                )
    v_pack_num_valid = None
    if phase == "context":
        _materialize_fp4_mla_device_page_table_for_forward(metadata)
        hp_pool_updated = _scatter_fp4_mla_kv_cache_2d_context(
            metadata,
            latent_cache,
            kv_cache,
            sf_cache,
            v_sf,
            global_scale,
            rotary_cos_sin,
            token_offset=token_offset,
            local_layer=local_layer,
            v_head_dim=v_head_dim,
            head_dim=head_dim,
            num_tokens=num_tokens,
            num_dim_blocks=num_dim_blocks,
            sf_per_token=sf_per_token,
            sf_per_page=sf_per_page,
            v_packed_base=v_packed_base,
            v_page_offset=v_page_offset,
        )
        v_pack_page_ids = metadata.paged_kv_indices
    else:
        generation_state = _scatter_fp4_mla_kv_cache_2d_generation(
            metadata,
            latent_cache,
            kv_cache,
            sf_cache,
            v_sf,
            global_scale,
            token_offset=token_offset,
            local_layer=local_layer,
            v_head_dim=v_head_dim,
            head_dim=head_dim,
            num_tokens=num_tokens,
            num_dim_blocks=num_dim_blocks,
            sf_per_token=sf_per_token,
            sf_per_page=sf_per_page,
            rotary_cos_sin=rotary_cos_sin,
            q_pe=q_pe,
            q_rope_out=q_rope_out,
            q_quant_input=q_quant_input,
            q_fp4_out=q_fp4_out,
            q_sf_out=q_sf_out,
            v_packed_base=v_packed_base,
            v_page_offset=v_page_offset,
        )
        v_pack_page_ids = _fp4_mla_generation_page_ids(
            metadata, metadata.num_seqs - metadata.num_contexts
        )
        if getattr(metadata, "is_cuda_graph", False):
            # Frozen launch grids cannot follow the per-replay page count;
            # the repack kernels stride over this device-side count instead.
            v_pack_num_valid = _fp4_mla_generation_num_blocks_device(metadata)
    cutedsl_repack_page_indptr = None
    cutedsl_repack_kv_lens = None
    cutedsl_repack_generation_lens = None
    cutedsl_repack_max_touched_pages = 1
    if cutedsl_backend and not fused_v_transpose and not direct_v_packed_write:
        if phase == "context":
            num_contexts = metadata.num_contexts
            cutedsl_v_pack_page_ids = metadata.paged_kv_indices[: metadata.num_context_blocks]
            cutedsl_repack_page_indptr = metadata.paged_kv_indptr[: num_contexts + 1]
            cutedsl_repack_kv_lens = metadata.kv_lens_cuda_runtime[:num_contexts]
            cutedsl_repack_generation_lens = metadata.prompt_lens_cuda_runtime[:num_contexts]
            cutedsl_repack_max_touched_pages = int(
                metadata.fp4_mla_context_repack_max_touched_pages
            )
        elif generation_state is not None:
            kv_lens_gen, gen_lens_gen, generation_page_ids = generation_state
            num_gen = kv_lens_gen.numel()
            cutedsl_v_pack_page_ids = generation_page_ids
            cutedsl_repack_page_indptr = metadata.paged_kv_indptr_decode
            cutedsl_repack_kv_lens = kv_lens_gen
            cutedsl_repack_generation_lens = gen_lens_gen
            cutedsl_repack_max_touched_pages = _ceil_div(
                num_tokens // num_gen + metadata.page_size - 1,
                metadata.page_size,
            )
        else:
            cutedsl_v_pack_page_ids = v_pack_page_ids
        _repack_cutedsl_v_packed_cache(
            persistent_v_packed,
            kv_cache,
            cutedsl_v_pack_page_ids,
            v_head_dim=v_head_dim,
            page_size=metadata.page_size,
            block_v=FP4_MLA_SCALE_ROW_GROUP,
            page_indptr=cutedsl_repack_page_indptr,
            kv_lens=cutedsl_repack_kv_lens,
            generation_lens=cutedsl_repack_generation_lens,
            max_touched_pages=cutedsl_repack_max_touched_pages,
        )
    _maybe_update_triton_v_packed_cache(
        metadata,
        layer_idx,
        kv_cache,
        v_pack_page_ids,
        num_queries=num_tokens,
        v_head_dim=v_head_dim,
        page_size=metadata.page_size,
        local_layer=local_layer,
        v_sf=v_sf[local_layer],
        num_valid_pages=v_pack_num_valid,
    )
    return hp_pool_updated


def _validate_fp4_mla_cache_shape(page_size: int, head_dim: int) -> None:
    if page_size != FP4_MLA_TOKENS_PER_BLOCK:
        raise ValueError(
            f"FP4 MLA KV cache requires tokens_per_block={FP4_MLA_TOKENS_PER_BLOCK} "
            f"for swizzled block scales, got {page_size}."
        )

    sf_per_token = head_dim // FP4_BLOCK_SIZE
    if head_dim % FP4_BLOCK_SIZE != 0 or sf_per_token % 4 != 0:
        raise ValueError(
            f"FP4 MLA KV head_dim must produce a scale column count divisible by 4; "
            f"got head_dim={head_dim}, scale_columns={sf_per_token}."
        )


def _validate_fp4_mla_attention_q_shape(head_dim: int, q_residual_dim: int) -> None:
    if q_residual_dim % FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP4 MLA Q residual_dim must be divisible by {FP4_BLOCK_SIZE}, got {q_residual_dim}."
        )
    if q_residual_dim <= 0 or q_residual_dim > head_dim:
        raise ValueError(
            f"FP4 MLA Q residual_dim must be in (0, head_dim], got "
            f"residual_dim={q_residual_dim}, head_dim={head_dim}."
        )

    q_head_dim = head_dim + q_residual_dim
    q_sf_per_token = q_head_dim // FP4_BLOCK_SIZE
    if q_head_dim % FP4_BLOCK_SIZE != 0 or q_sf_per_token % FP4_MLA_SCALE_COL_GROUP != 0:
        raise ValueError(
            f"FP4 MLA residual Q must produce a scale column count divisible "
            f"by {FP4_MLA_SCALE_COL_GROUP}; got q_head_dim={q_head_dim}, "
            f"scale_columns={q_sf_per_token}."
        )


def _ensure_workspace_tensor(
    metadata: Any,
    attr_name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = getattr(metadata, attr_name, None)
    needs_alloc = (
        tensor is None
        or tensor.dtype != dtype
        or tensor.device != device
        or len(tensor.shape) != len(shape)
        or any(tensor.shape[idx] < dim for idx, dim in enumerate(shape))
    )
    if needs_alloc:
        if torch.cuda.is_current_stream_capturing():
            raise ValueError(
                f"Cannot allocate {attr_name} while capturing a CUDA graph. "
                "Run a warmup prepare/forward first."
            )
        tensor = torch.empty(shape, dtype=dtype, device=device)
        setattr(metadata, attr_name, tensor)

    slices = tuple(slice(0, dim) for dim in shape)
    return tensor[slices]


def _shared_v_pack_storage_enabled() -> bool:
    return os.getenv("TRTLLM_FP4_MLA_SHARE_V_PACK_STORAGE", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _select_triton_block_v(num_queries: int, *, prefer_prepacked_v: bool = False) -> int:
    env_block_v = _env_int("TRTLLM_FP4_MLA_BLOCK_V")
    if env_block_v is not None:
        return env_block_v
    if prefer_prepacked_v:
        return 128
    return 32 if num_queries <= 32 else 128


def _v_packed_shape(
    kv_cache: torch.Tensor,
    v_head_dim: int,
    page_size: int,
    block_v: int,
) -> tuple[int, int]:
    return (kv_cache.shape[0] * _ceil_div(v_head_dim, block_v) * block_v, page_size // 2)


def _get_fp4_mla_v_packed_pool(metadata: Any, local_layer: int) -> Optional[torch.Tensor]:
    return metadata.kv_cache_manager.get_mla_v_packed_pool(local_layer)


def _get_fp4_mla_v_packed_pool_base(metadata: Any) -> Optional[torch.Tensor]:
    return metadata.kv_cache_manager.get_mla_v_packed_pool_base()


def _get_fp4_mla_v_scale_pool_base(metadata: Any) -> Optional[torch.Tensor]:
    return metadata.kv_cache_manager.get_mla_v_scale_pool_base()


def _get_cutedsl_persistent_v_packed_cache(
    metadata: Any,
    local_layer: int,
    kv_cache: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
    block_v: int,
) -> torch.Tensor:
    v_packed = _get_fp4_mla_v_packed_pool(metadata, local_layer)
    if v_packed is None:
        raise RuntimeError(
            "CuTeDSL FP4 MLA requires the manager-owned persistent V-packed "
            "pool; the scratch full-repack fallback has been removed."
        )
    expected_shape = _v_packed_shape(kv_cache, v_head_dim, page_size, block_v)
    if (
        not isinstance(v_packed, torch.Tensor)
        or v_packed.dtype != torch.uint8
        or v_packed.device != kv_cache.device
        or tuple(v_packed.shape) != expected_shape
        or not v_packed.is_contiguous()
    ):
        raise RuntimeError(
            "FP4 MLA persistent V-packed pool must be a contiguous uint8 tensor "
            f"with shape {expected_shape} on {kv_cache.device}; got "
            f"{type(v_packed).__name__}, "
            f"shape={getattr(v_packed, 'shape', None)}, "
            f"dtype={getattr(v_packed, 'dtype', None)}, "
            f"device={getattr(v_packed, 'device', None)}."
        )
    return v_packed


def _repack_cutedsl_v_packed_cache(
    v_packed: torch.Tensor,
    kv_cache: torch.Tensor,
    page_ids: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
    block_v: int,
    page_indptr: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
    generation_lens: Optional[torch.Tensor] = None,
    max_touched_pages: int = 1,
) -> None:
    if page_ids.numel() == 0:
        return
    from .fp4_mla_cutedsl_v_repack import fp4_mla_repack_v_cache

    fp4_mla_repack_v_cache(
        v_packed,
        kv_cache,
        page_ids,
        v_head_dim=v_head_dim,
        page_size=page_size,
        block_v=block_v,
        page_indptr=page_indptr,
        kv_lens=kv_lens,
        generation_lens=generation_lens,
        max_touched_pages=max_touched_pages,
    )


def _v_packed_cache_tag(
    layer_idx: int,
    kv_cache: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
    local_layer: Optional[int] = None,
    v_sf: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    block_v: int = 128,
) -> tuple[Any, ...]:
    v_sf_tag = (
        None
        if v_sf is None
        else (
            int(v_sf.data_ptr()),
            str(v_sf.device),
            str(v_sf.dtype),
            tuple(int(dim) for dim in v_sf.shape),
            tuple(int(stride) for stride in v_sf.stride()),
        )
    )
    page_ids_tag = (
        None
        if page_ids is None
        else (
            int(page_ids.data_ptr()),
            str(page_ids.device),
            str(page_ids.dtype),
            tuple(int(dim) for dim in page_ids.shape),
            tuple(int(stride) for stride in page_ids.stride()),
        )
    )
    return (
        int(layer_idx),
        None if local_layer is None else int(local_layer),
        int(kv_cache.data_ptr()),
        str(kv_cache.device),
        str(kv_cache.dtype),
        tuple(int(dim) for dim in kv_cache.shape),
        tuple(int(stride) for stride in kv_cache.stride()),
        int(v_head_dim),
        int(page_size),
        int(block_v),
        v_sf_tag,
        page_ids_tag,
    )


def _triton_prepack_v_enabled() -> bool:
    if _fp4_mla_attention_backend() != "triton":
        return False
    default = _env_enabled_default("TRTLLM_FP4_MLA_PREPACK_V", True)
    return _env_enabled_default("TRTLLM_FP4_MLA_TRITON_PREPACK_V", default)


def _triton_can_prepack_v(v_head_dim: int, page_size: int, block_v: int) -> bool:
    return (
        _triton_prepack_v_enabled()
        and hasattr(tl, "make_tensor_descriptor")
        and block_v in (32, 128)
        and v_head_dim % block_v == 0
        and page_size == FP4_MLA_TOKENS_PER_BLOCK
    )


def _triton_v_packed_attr(layer_idx: int) -> str:
    if _shared_v_pack_storage_enabled():
        return "_fp4_mla_triton_attention_v_packed_buf"
    return f"_fp4_mla_triton_attention_v_packed_buf_l{layer_idx}"


def _triton_v_packed_valid_attr(layer_idx: int) -> str:
    return f"_fp4_mla_triton_attention_v_packed_valid_l{layer_idx}"


def _triton_shared_v_packed_valid_attr() -> str:
    return "_fp4_mla_triton_attention_v_packed_valid_tag"


def _triton_v_packed_cache_tag(
    layer_idx: int,
    kv_cache: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
    local_layer: Optional[int] = None,
    v_sf: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    block_v: int = 128,
) -> tuple[Any, ...]:
    return (
        "triton",
        _v_packed_cache_tag(
            layer_idx,
            kv_cache,
            v_head_dim=v_head_dim,
            page_size=page_size,
            block_v=block_v,
            local_layer=local_layer,
            v_sf=v_sf,
            page_ids=page_ids,
        ),
    )


def _set_triton_v_packed_cache_valid(
    metadata: Any,
    layer_idx: int,
    kv_cache: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
    local_layer: Optional[int] = None,
    v_sf: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    block_v: int = 128,
) -> None:
    valid_attr = (
        _triton_shared_v_packed_valid_attr()
        if _shared_v_pack_storage_enabled()
        else _triton_v_packed_valid_attr(layer_idx)
    )
    setattr(
        metadata,
        valid_attr,
        _triton_v_packed_cache_tag(
            layer_idx,
            kv_cache,
            v_head_dim=v_head_dim,
            page_size=page_size,
            block_v=block_v,
            local_layer=local_layer,
            v_sf=v_sf,
            page_ids=page_ids,
        ),
    )


def _is_triton_v_packed_cache_valid(
    metadata: Any,
    layer_idx: int,
    kv_cache: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
    local_layer: Optional[int] = None,
    v_sf: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    block_v: int = 128,
) -> bool:
    valid_attr = (
        _triton_shared_v_packed_valid_attr()
        if _shared_v_pack_storage_enabled()
        else _triton_v_packed_valid_attr(layer_idx)
    )
    return getattr(metadata, valid_attr, None) == _triton_v_packed_cache_tag(
        layer_idx,
        kv_cache,
        v_head_dim=v_head_dim,
        page_size=page_size,
        block_v=block_v,
        local_layer=local_layer,
        v_sf=v_sf,
        page_ids=page_ids,
    )


def _get_triton_v_packed_cache(
    metadata: Any,
    layer_idx: int,
    kv_cache: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
    local_layer: Optional[int] = None,
    v_sf: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    block_v: int = 128,
) -> Optional[torch.Tensor]:
    if not _triton_can_prepack_v(v_head_dim, page_size, block_v):
        return None
    if not _is_triton_v_packed_cache_valid(
        metadata,
        layer_idx,
        kv_cache,
        v_head_dim=v_head_dim,
        page_size=page_size,
        block_v=block_v,
        local_layer=local_layer,
        v_sf=v_sf,
        page_ids=page_ids,
    ):
        return None
    v_packed = getattr(metadata, _triton_v_packed_attr(layer_idx), None)
    expected_shape = _v_packed_shape(kv_cache, v_head_dim, page_size, block_v)
    if (
        v_packed is None
        or v_packed.dtype != torch.uint8
        or v_packed.device != kv_cache.device
        or len(v_packed.shape) != 2
        or v_packed.shape[0] < expected_shape[0]
        or v_packed.shape[1] < expected_shape[1]
    ):
        return None
    return v_packed[: expected_shape[0], : expected_shape[1]]


def _update_triton_v_packed_cache(
    metadata: Any,
    layer_idx: int,
    kv_cache: torch.Tensor,
    page_ids: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
    block_v: int,
    local_layer: Optional[int] = None,
    v_sf: Optional[torch.Tensor] = None,
    num_valid_pages: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if not _triton_can_prepack_v(v_head_dim, page_size, block_v):
        return None
    if page_ids.numel() == 0:
        return None
    from .fp4_mla_triton import fp4_mla_repack_v_cache

    def _tma_alloc(size: int, alignment: int, stream):
        return torch.empty(size, device=kv_cache.device, dtype=torch.int8)

    triton.set_allocator(_tma_alloc)
    attr_name = _triton_v_packed_attr(layer_idx)
    v_packed = _ensure_workspace_tensor(
        metadata,
        attr_name,
        _v_packed_shape(kv_cache, v_head_dim, page_size, block_v),
        dtype=torch.uint8,
        device=kv_cache.device,
    )
    fp4_mla_repack_v_cache(
        v_packed,
        kv_cache,
        page_ids,
        v_head_dim=v_head_dim,
        page_size=page_size,
        block_v=block_v,
        num_valid_pages=num_valid_pages,
    )
    _set_triton_v_packed_cache_valid(
        metadata,
        layer_idx,
        kv_cache,
        v_head_dim=v_head_dim,
        page_size=page_size,
        block_v=block_v,
        local_layer=local_layer,
        v_sf=v_sf,
        page_ids=page_ids,
    )
    return v_packed


def _maybe_update_triton_v_packed_cache(
    metadata: Any,
    layer_idx: int,
    kv_cache: torch.Tensor,
    page_ids: torch.Tensor,
    *,
    num_queries: int,
    v_head_dim: int,
    page_size: int,
    local_layer: Optional[int] = None,
    v_sf: Optional[torch.Tensor] = None,
    num_valid_pages: Optional[torch.Tensor] = None,
) -> None:
    block_v = _select_triton_block_v(num_queries, prefer_prepacked_v=_triton_prepack_v_enabled())
    _update_triton_v_packed_cache(
        metadata,
        layer_idx,
        kv_cache,
        page_ids,
        v_head_dim=v_head_dim,
        page_size=page_size,
        block_v=block_v,
        local_layer=local_layer,
        v_sf=v_sf,
        num_valid_pages=num_valid_pages,
    )


def _max_generation_pages(metadata: Any) -> int:
    num_gen = metadata.num_seqs - metadata.num_contexts
    if num_gen <= 0:
        return 0
    if not getattr(metadata, "_fp4_mla_device_page_table", False):
        raise RuntimeError("FP4 MLA generation requires fixed-stride device page metadata.")
    max_pages = int(metadata.fp4_mla_page_table_stride)
    if max_pages <= 0:
        raise RuntimeError("FP4 MLA device page-table stride must be positive.")
    return max_pages


def _fp4_mla_generation_page_ids(metadata: Any, num_gen_seqs: int) -> torch.Tensor:
    """Return the fixed-stride generation page-table view."""
    expected_num_gen = metadata.num_seqs - metadata.num_contexts
    if num_gen_seqs != expected_num_gen:
        raise RuntimeError(
            "FP4 MLA generation sequence count does not match metadata: "
            f"{num_gen_seqs} != {expected_num_gen}."
        )
    max_pages = _max_generation_pages(metadata)
    page_ids = getattr(metadata, "_paged_kv_indices", None)
    start = metadata.num_contexts * max_pages
    end = start + num_gen_seqs * max_pages
    if (
        not isinstance(page_ids, torch.Tensor)
        or page_ids.ndim != 1
        or page_ids.dtype != torch.int32
        or not page_ids.is_contiguous()
        or page_ids.numel() < end
    ):
        raise RuntimeError("FP4 MLA fixed-stride generation page-table backing is invalid.")
    return page_ids[start:end]


def _fp4_mla_generation_hp_page_ids(metadata: Any, num_gen_seqs: int) -> torch.Tensor:
    """Return generation rows from the fixed-stride V2 HP page table."""
    expected_num_gen = metadata.num_seqs - metadata.num_contexts
    if num_gen_seqs != expected_num_gen:
        raise RuntimeError(
            "FP4 MLA generation sequence count does not match metadata: "
            f"{num_gen_seqs} != {expected_num_gen}."
        )
    max_pages = _max_generation_pages(metadata)
    page_ids = getattr(metadata, "_fp4_mla_hp_page_indices", None)
    start = metadata.num_contexts * max_pages
    end = start + num_gen_seqs * max_pages
    if (
        not isinstance(page_ids, torch.Tensor)
        or page_ids.ndim != 1
        or page_ids.dtype != torch.int32
        or not page_ids.is_contiguous()
        or page_ids.numel() < end
    ):
        raise RuntimeError("FP4 MLA fixed-stride generation HP page-table backing is invalid.")
    return page_ids[start:end]


def _host_int_list(value: Any, start: int, end: int) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            return None
        return [int(item) for item in value[start:end].tolist()]
    try:
        return [int(item) for item in value[start:end]]
    except (TypeError, ValueError):
        return None


def _infer_assume_full_pages(metadata: Any, max_pages: int, page_size: int) -> bool:
    if getattr(metadata, "is_cuda_graph", False):
        return False

    start = metadata.num_contexts
    end = metadata.num_seqs
    block_counts = _host_int_list(getattr(metadata, "num_blocks", None), start, end)
    if block_counts is not None and (
        not block_counts or min(block_counts) != max_pages or max(block_counts) != max_pages
    ):
        return False

    kv_lens_cuda = getattr(metadata, "kv_lens_cuda_runtime", None)
    if isinstance(kv_lens_cuda, torch.Tensor):
        cache_key = (
            start,
            end,
            max_pages,
            page_size,
            tuple(block_counts) if block_counts is not None else None,
            kv_lens_cuda.data_ptr(),
        )
        cache = getattr(metadata, "_fp4_mla_full_pages_cache", None)
        if cache is not None and cache[0] == cache_key:
            return bool(cache[1])
        kv_lens = [int(item) for item in kv_lens_cuda[start:end].detach().cpu().tolist()]
        result = bool(kv_lens) and min(kv_lens) == max(kv_lens) == max_pages * page_size
        setattr(metadata, "_fp4_mla_full_pages_cache", (cache_key, result))
        return result

    kv_cache_params = getattr(metadata, "kv_cache_params", None)
    cached_token_lens = _host_int_list(
        getattr(kv_cache_params, "num_cached_tokens_per_seq", None),
        start,
        end,
    )
    seq_lens_kv = _host_int_list(getattr(metadata, "seq_lens_kv", None), start, end)
    if cached_token_lens is not None and seq_lens_kv is not None:
        if len(cached_token_lens) != len(seq_lens_kv):
            return False
        kv_lens = [
            cached_len + seq_len for cached_len, seq_len in zip(cached_token_lens, seq_lens_kv)
        ]
    elif kv_cache_params is None:
        kv_lens = _host_int_list(getattr(metadata, "prompt_lens_cpu_runtime", None), start, end)
    else:
        return False

    return bool(kv_lens) and min(kv_lens) == max(kv_lens) == max_pages * page_size


def _get_linear_mtp_query_len_per_seq(
    metadata: Any,
    *,
    num_queries: int,
    num_gen_seqs: int,
) -> int:
    """Return the uniform generation query length required by linear MTP.

    Derives the length from the real query-token count (``num_queries``, taken
    from the q shape) and the generation sequence count, which are reliable in
    every representation. The host ``prompt_lens``/``seq_lens`` mirror can lag at
    the decode anchor (== 1) under CUDA graph / one-engine MTP, so it is only
    consulted to produce a precise diagnostic when the counts do not divide
    evenly (a genuinely non-uniform batch, which the no-dequant path does not
    support).
    """
    if num_gen_seqs <= 0:
        return 1

    if num_queries % num_gen_seqs == 0:
        return num_queries // num_gen_seqs

    start = metadata.num_contexts
    end = metadata.num_seqs
    query_lens = _host_int_list_during_forward(
        getattr(metadata, "prompt_lens_cpu_runtime", None), start, end
    )
    if query_lens is None:
        query_lens = _host_int_list_during_forward(getattr(metadata, "seq_lens", None), start, end)
    raise NotImplementedError(
        "FP4 MLA no-dequant attention requires a uniform linear MTP generation "
        f"query length; got {num_queries} query tokens for {num_gen_seqs} "
        f"sequences (per-sequence lengths {query_lens})."
    )


def _run_triton_attention_decode(
    *,
    metadata: Any,
    layer_idx: int,
    local_layer: int,
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_sf: torch.Tensor,
    global_scale: torch.Tensor,
    src_page_ids: torch.Tensor,
    kv_lens: torch.Tensor,
    p_fp4: torch.Tensor,
    p_sf: torch.Tensor,
    max_scores: torch.Tensor,
    denom: torch.Tensor,
    output: torch.Tensor,
    num_queries: int,
    num_heads: int,
    head_dim: int,
    kv_lora_rank: int,
    q_residual_dim: int,
    query_len_per_seq: int,
    max_pages: int,
    sm_scale: float,
    q_global_scale: torch.Tensor,
) -> None:
    """Dispatch the ``triton`` FP4 MLA decode pipeline.

    Mirrors the four-stage layout used by ``fp4_mla_cutile.py``
    (page-stats with packed P -> reduce-stats -> prob-scale -> PV) but
    routes through the self-contained kernels in
    ``fp4_mla_triton.py``. Threads through the constexpr assume flags,
    TMA descriptors, occupancy/num-warps launch meta, and pipelined PV loop.
    """
    from .fp4_mla_triton import (
        _fp4_mla_attention_group_reduce_stats_kernel as _attn_group_reduce_stats_kernel,
    )
    from .fp4_mla_triton import _fp4_mla_attention_page_stats_kernel as _attn_page_stats_kernel
    from .fp4_mla_triton import _fp4_mla_attention_prob_scale_kernel as _attn_prob_scale_kernel
    from .fp4_mla_triton import _fp4_mla_attention_pv_kernel as _attn_pv_kernel
    from .fp4_mla_triton import (
        _fp4_mla_attention_pv_prepacked_v_kernel as _attn_pv_prepacked_v_kernel,
    )
    from .fp4_mla_triton import _fp4_mla_attention_pv_reduce_kernel as _attn_pv_reduce_kernel
    from .fp4_mla_triton import _fp4_mla_attention_reduce_stats_kernel as _attn_reduce_stats_kernel

    block_h = 128
    block_t = metadata.page_size
    # Adaptive BLOCK_V: the fallback PV path uses a finer V split at small batch
    # on B200 (~148 SMs). PV grid = num_queries * num_head_blocks(1) *
    # (kv_lora_rank / BLOCK_V). We want >= ~2*num_SMs programs so that >1 CTA
    # lands per SM and hides the L1TEX scoreboard stalls. Empirically (sweep):
    #   bs<=32 -> BLOCK_V=32; bs>=64 -> BLOCK_V=128.
    # (BLOCK_V=16 is rejected by the V TMA descriptor min-stride requirement.)
    # With prepacked V, BLOCK_V=128 avoids reloading the same P tile four times
    # and matches the cutile prepacked-V tile shape.
    block_v = _select_triton_block_v(num_queries, prefer_prepacked_v=_triton_prepack_v_enabled())
    q_storage_head_dim = head_dim + q_residual_dim
    # The virtual GEMM tail evaluates QK + Q_r K + Q K_r in one reduction.
    # Q and Q_r still occupy the 640-channel interleaved physical Q buffer;
    # the final Q term reuses Q's main tail groups while K_r comes from the
    # contiguous 64-channel tail of the primary paged KV cache.
    q_head_dim = head_dim + q_residual_dim + FP4_MLA_K_RESIDUAL_DIM
    # BLOCK_K = 512 aligns the K-window with the 512-channel non-residual prefix.
    block_k = 512
    full_block_end = (q_head_dim // block_k) * block_k
    tail_k = q_head_dim - full_block_end
    tail_block_k = 1 << (tail_k - 1).bit_length() if tail_k > 0 else block_k
    q_sf_per_token = q_storage_head_dim // FP4_BLOCK_SIZE
    k_sf_per_token = (head_dim + FP4_MLA_K_RESIDUAL_DIM) // FP4_BLOCK_SIZE
    sf_per_page = metadata.page_size // FP4_BLOCK_SIZE
    num_head_blocks = triton.cdiv(num_heads, block_h)

    assume_full_heads = num_heads % block_h == 0
    assume_full_v = kv_lora_rank % block_v == 0
    # Match the cutile path: only mark pages "full" when we can prove every
    # generation sequence has the same number of cached tokens AND
    # query_len_per_seq == 1 (so the kv_len adjustment is a no-op).
    assume_full_pages = (
        _infer_assume_full_pages(metadata, max_pages, metadata.page_size) and query_len_per_seq == 1
    )
    # Leave validity checks on. Matches cutile's default and is correctness-
    # safe. The perfect-shape PV fast path (tl.ext.make_view + load_view_tko)
    # remains gated off — when measured on the TileIR backend (ENABLE_TILE=1)
    # it was net-slower on the bench, so the cost of enabling it isn't worth
    # the win on the FP4 MLA shapes we care about.
    assume_valid_pages = False
    num_gen_seqs = num_queries // query_len_per_seq
    if (
        not assume_valid_pages
        and assume_full_pages
        and src_page_ids.numel() == num_gen_seqs * max_pages
    ):
        assume_valid_pages = True
    # cutile checks only `make_tensor_descriptor`; on the nvt backend the
    # presence of TMA descriptors implies `tl.ext.make_view` is available too.
    use_tma_data_load = hasattr(triton.language, "make_tensor_descriptor")

    # Install the device-side scratch allocator on every call. Triton stores
    # the allocator in a ContextVar (triton.runtime._allocation), so a single
    # process-wide install is not visible from worker threads / asyncio tasks
    # that run with a different Context — the kernel launch would then hit the
    # default NullAllocator and raise. Matches the cutile path.
    if use_tma_data_load:

        def _tma_alloc(size: int, alignment: int, stream):
            return torch.empty(size, device=q_fp4.device, dtype=torch.int8)

        triton.set_allocator(_tma_alloc)

    # cutile-equivalent launch meta. occupancy=2 lets two CTAs land per SM
    # which improves wave-tail efficiency at the bs=32 hot point.
    # NOTE: num_stages=2 (instead of the Triton 3.6 default of 3) sidesteps
    # the TritonGPUAutomaticWarpSpecialization + NVWSInsertTmemAref pass that
    # ICEs on the page_stats kernel under Triton 3.6.0 / sm_100.
    launch_meta = {"occupancy": 2}
    # The matmul kernels (page-stats QK and PV) are register-limited: at the
    # Triton default of num_warps=4 the [BLOCK_H, BLOCK_T] epilogue spills the
    # register file down to ~2 CTAs/SM (12.5% occupancy), so there are too few
    # warps to hide the QK/PV load latency (ncu: ~0.3 eligible warps/scheduler).
    # Spreading the tile epilogue over num_warps=8 halves the per-thread
    # register need and roughly doubles resident warps. Matches the cutile
    # ("nvt") backend, which launches page-stats at num_warps=8. Both are
    # overridable for tuning.
    sm_count = _get_sm_count(q_fp4.device)
    # page-stats num_warps: the full-pages fast path (uniform q_len==1 decode)
    # benefits from num_warps=8 (more warps hide the QK load latency); the
    # masked path (q_len>1 / ragged lengths) carries extra per-thread state and
    # measured markedly faster at num_warps=4 (e.g. bs256 q_len4: 131->95ms).
    page_stats_num_warps = _env_int("TRTLLM_FP4_MLA_PAGE_STATS_NUM_WARPS")
    if page_stats_num_warps is None:
        page_stats_num_warps = 8 if assume_full_pages else 4
    page_stats_launch_meta = {"occupancy": 2, "num_warps": page_stats_num_warps}
    # PV benefits from num_warps=8 across shapes measured.
    pv_num_warps = _env_int("TRTLLM_FP4_MLA_PV_NUM_WARPS") or 8
    pv_launch_meta = {"occupancy": 2, "num_warps": pv_num_warps}
    # PV loop pipelining. With TMA loads, num_stages>=2 lets the next page's
    # loads overlap with the current MMA via mbarrier. The PV report shows
    # long_scoreboard=4.5 cycles avg on V loads at PV_LOOP_STAGES=2; bumping the
    # depth pays off when the grid is small enough that occupancy can absorb
    # the extra in-flight tile state — i.e. medium batch / large max_pages.
    # Larger pipelines hurt at small batch (more live state, fewer dim blocks).
    if num_queries <= 16 or max_pages <= 4:
        pv_loop_stages = 2
    else:
        pv_loop_stages = 3

    # Page-stats kernel: per (query, head_block, page) program, does QK,
    # softmax stats, and packs probs into FP4 with the per-page local-max
    # scaling trick. The page-max correction is applied later by
    # prob_scale_kernel via p_sf in-place rescaling.
    page_stats_shape = (num_queries, max_pages, num_heads)
    page_max = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_page_max_buf",
        page_stats_shape,
        dtype=torch.float32,
        device=q_fp4.device,
    )
    page_sum = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_page_sum_buf",
        page_stats_shape,
        dtype=torch.float32,
        device=q_fp4.device,
    )

    pack_prob_in_page_stats = True
    _attn_page_stats_kernel[(num_queries, num_head_blocks, max_pages)](
        page_max,
        page_sum,
        p_fp4,
        p_sf,
        q_fp4,
        q_sf,
        kv_cache,
        sf_cache,
        global_scale,
        q_global_scale,
        src_page_ids,
        metadata.paged_kv_indptr_decode,
        kv_lens,
        src_page_ids.shape[0],
        kv_cache.shape[0],
        q_fp4.stride(0),
        q_fp4.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(2),
        kv_cache.stride(4),
        sf_cache.stride(0),
        page_max.stride(0),
        page_max.stride(1),
        p_fp4.stride(0),
        p_fp4.stride(1),
        p_fp4.shape[0],
        q_fp4.shape[0],
        sm_scale,
        NUM_HEADS=num_heads,
        Q_HEAD_D=q_head_dim,
        Q_STORAGE_HEAD_D=q_storage_head_dim,
        K_HEAD_D=head_dim,
        Q_RESIDUAL_D=q_residual_dim,
        K_RESIDUAL_D=FP4_MLA_K_RESIDUAL_DIM,
        PAGE_SIZE=metadata.page_size,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        Q_SF_PER_TOKEN=q_sf_per_token,
        K_SF_PER_TOKEN=k_sf_per_token,
        SF_PER_PAGE=sf_per_page,
        P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
        QUERY_LEN_PER_SEQ=query_len_per_seq,
        MAX_PAGES=max_pages,
        BLOCK_H=block_h,
        BLOCK_T=block_t,
        BLOCK_K=block_k,
        FULL_BLOCK_END=full_block_end,
        TAIL_BLOCK_K=tail_block_k,
        USE_TMA_DATA_LOAD=use_tma_data_load,
        PACK_PROBS=pack_prob_in_page_stats,
        ASSUME_FULL_HEADS=assume_full_heads,
        ASSUME_FULL_PAGES=assume_full_pages,
        ASSUME_VALID_PAGES=assume_valid_pages,
        **page_stats_launch_meta,
    )
    # Two-level softmax-stats reduction. The single-level reduce launched only
    # (num_queries * num_head_blocks) CTAs, each serially walking all max_pages
    # twice -- at small batch that handful of CTAs left the GPU almost idle and
    # the reduce cost more than the QK matmul. Level 1 parallelizes the page
    # reduction across a page-group axis (online-softmax partials, pipelined);
    # level 2 reuses the existing reduce kernel to fold the few groups into the
    # global (max, denom). When the (query, head) grid already fills the GPU the
    # group count collapses to 1 and this degenerates to the original reduce.
    seqhead_ctas = num_queries * num_head_blocks
    # Aim for ~3 waves of level-1 CTAs so page loads have enough memory-level
    # parallelism to hide latency, while keeping the group count small enough
    # that the level-2 combine loop stays short.
    target_l1_ctas = 3 * sm_count
    num_reduce_groups = _ceil_div(target_l1_ctas, max(seqhead_ctas, 1))
    num_reduce_groups = max(1, min(num_reduce_groups, max_pages, 64))
    # The grouped (two-level) reduce needs an auxiliary workspace, and
    # _ensure_workspace_tensor can only (re)allocate it outside CUDA graph
    # capture. If a warmup forward did not already size that workspace (e.g. the
    # warmup batch took the single-level path), fall back to the single-level
    # reduce during capture so we never allocate mid-capture. The single-level
    # reduce is numerically identical (it just launches fewer CTAs).
    if num_reduce_groups > 1 and torch.cuda.is_current_stream_capturing():
        gmax = getattr(metadata, "_fp4_mla_attention_group_max_buf", None)
        gsum = getattr(metadata, "_fp4_mla_attention_group_sum_buf", None)
        groups_ready = (
            gmax is not None
            and gsum is not None
            and gmax.shape[0] >= num_queries
            and gmax.shape[1] >= num_reduce_groups
            and gmax.shape[2] >= num_heads
            and gsum.shape[0] >= num_queries
            and gsum.shape[1] >= num_reduce_groups
            and gsum.shape[2] >= num_heads
        )
        if not groups_ready:
            num_reduce_groups = 1
    if num_reduce_groups <= 1:
        _attn_reduce_stats_kernel[(num_queries, num_head_blocks)](
            max_scores,
            denom,
            page_max,
            page_sum,
            max_pages,
            max_scores.stride(0),
            page_max.stride(0),
            page_max.stride(1),
            NUM_HEADS=num_heads,
            MAX_PAGES=max_pages,
            BLOCK_H=block_h,
            **launch_meta,
        )
    else:
        group_pages = _ceil_div(max_pages, num_reduce_groups)
        num_reduce_groups = _ceil_div(max_pages, group_pages)
        group_max = _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_attention_group_max_buf",
            (num_queries, num_reduce_groups, num_heads),
            dtype=torch.float32,
            device=q_fp4.device,
        )
        group_sum = _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_attention_group_sum_buf",
            (num_queries, num_reduce_groups, num_heads),
            dtype=torch.float32,
            device=q_fp4.device,
        )
        _attn_group_reduce_stats_kernel[(num_queries, num_head_blocks, num_reduce_groups)](
            group_max,
            group_sum,
            page_max,
            page_sum,
            max_pages,
            group_max.stride(0),
            group_max.stride(1),
            page_max.stride(0),
            page_max.stride(1),
            NUM_HEADS=num_heads,
            GROUP_PAGES=group_pages,
            BLOCK_H=block_h,
            PIPELINE_STAGES=min(group_pages, 4),
            **launch_meta,
        )
        _attn_reduce_stats_kernel[(num_queries, num_head_blocks)](
            max_scores,
            denom,
            group_max,
            group_sum,
            num_reduce_groups,
            max_scores.stride(0),
            group_max.stride(0),
            group_max.stride(1),
            NUM_HEADS=num_heads,
            MAX_PAGES=num_reduce_groups,
            BLOCK_H=block_h,
            **launch_meta,
        )
    _attn_prob_scale_kernel[(num_queries, num_head_blocks, max_pages)](
        p_sf,
        max_scores,
        denom,
        page_max,
        metadata.paged_kv_indptr_decode,
        kv_lens,
        src_page_ids.shape[0],
        max_scores.stride(0),
        page_max.stride(0),
        page_max.stride(1),
        NUM_HEADS=num_heads,
        PAGE_SIZE=metadata.page_size,
        SF_PER_PAGE=sf_per_page,
        QUERY_LEN_PER_SEQ=query_len_per_seq,
        MAX_PAGES=max_pages,
        BLOCK_H=block_h,
        ASSUME_FULL_HEADS=assume_full_heads,
        ASSUME_FULL_PAGES=assume_full_pages,
        ASSUME_VALID_PAGES=assume_valid_pages,
        **launch_meta,
    )
    num_dim_blocks = triton.cdiv(kv_lora_rank, block_v)
    v_packed = _get_triton_v_packed_cache(
        metadata,
        layer_idx,
        kv_cache,
        v_head_dim=kv_lora_rank,
        page_size=metadata.page_size,
        block_v=block_v,
        local_layer=local_layer,
        v_sf=v_sf,
        page_ids=src_page_ids,
    )
    if (
        v_packed is None
        and _triton_can_prepack_v(kv_lora_rank, metadata.page_size, block_v)
        and not torch.cuda.is_current_stream_capturing()
    ):
        v_packed = _update_triton_v_packed_cache(
            metadata,
            layer_idx,
            kv_cache,
            src_page_ids,
            v_head_dim=kv_lora_rank,
            page_size=metadata.page_size,
            block_v=block_v,
            local_layer=local_layer,
            v_sf=v_sf,
        )
    use_triton_v_packed_cache = v_packed is not None

    # PV page split: partition the page range across additional programs and
    # reduce in a follow-up kernel. ncu showed PV at waves/SM=0.49 for bs=32 —
    # PV is L1-bandwidth bound, so raising in-flight CTAs is the lever.
    # BLOCK_V is bounded below by the 16-byte TMA descriptor min-stride.
    # PV page split: ncu shows that with the current shape (bs=32, max_pages=256)
    # the PV kernel is L1-cache-throughput bound (long_scoreboard=4.5 cycles
    # avg, L1 global LD hit-rate <40%). Increasing the program count via page
    # splitting reduced waves/SM idle time but did NOT improve wall-time at
    # current shapes — the per-CTA L1 thrash is the limit. Gate the split off
    # by default; re-enable only for very small grids where occupancy is the
    # bottleneck rather than per-CTA L1 pressure.
    page_split = 1
    base_grid = num_queries * num_head_blocks * num_dim_blocks
    if max_pages >= 16 and base_grid < 148:
        for p in (8, 4, 2):
            if max_pages % p == 0 and max_pages // p >= 16 and base_grid * p <= 148 * 4:
                page_split = p
                break
    # The page-split PV path needs a partial-output workspace, which
    # _ensure_workspace_tensor can only (re)allocate outside CUDA graph capture.
    # Fall back to the unsplit PV (numerically identical) during capture unless a
    # warmup forward already sized that workspace, so capture never allocates.
    if page_split > 1 and torch.cuda.is_current_stream_capturing():
        pbuf = getattr(metadata, "_fp4_mla_attention_pv_partial_buf", None)
        partial_ready = (
            pbuf is not None
            and pbuf.shape[0] >= num_queries
            and pbuf.shape[1] >= page_split
            and pbuf.shape[2] >= num_heads
            and pbuf.shape[3] >= kv_lora_rank
        )
        if not partial_ready:
            page_split = 1
    if page_split > 1:
        pages_per_split = max_pages // page_split
        partial_out = _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_attention_pv_partial_buf",
            (num_queries, page_split, num_heads, kv_lora_rank),
            dtype=torch.float32,
            device=q_fp4.device,
        )
        if use_triton_v_packed_cache:
            _attn_pv_prepacked_v_kernel[
                (num_queries, num_head_blocks, num_dim_blocks * page_split)
            ](
                output,
                p_fp4,
                p_sf,
                v_packed,
                v_sf,
                global_scale,
                src_page_ids,
                metadata.paged_kv_indptr_decode,
                kv_lens,
                src_page_ids.shape[0],
                kv_cache.shape[0],
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.shape[0] * output.shape[1],
                p_fp4.stride(0),
                p_fp4.stride(1),
                p_fp4.shape[0],
                v_sf.stride(0),
                NUM_HEADS=num_heads,
                V_HEAD_D=kv_lora_rank,
                PAGE_SIZE=metadata.page_size,
                FP4_BLOCK=FP4_BLOCK_SIZE,
                SF_PER_PAGE=sf_per_page,
                QUERY_LEN_PER_SEQ=query_len_per_seq,
                MAX_PAGES=max_pages,
                P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
                BLOCK_H=block_h,
                BLOCK_V=block_v,
                USE_TMA_P_LOAD=use_tma_data_load and assume_full_heads and assume_valid_pages,
                USE_TMA_OUT_STORE=use_tma_data_load and assume_full_heads and assume_full_v,
                PV_LOOP_STAGES=pv_loop_stages,
                ASSUME_FULL_HEADS=assume_full_heads,
                ASSUME_FULL_PAGES=assume_full_pages,
                ASSUME_FULL_V=assume_full_v,
                ASSUME_VALID_PAGES=assume_valid_pages,
                PAGE_SPLIT=page_split,
                PAGES_PER_SPLIT=pages_per_split,
                PARTIAL_OUT=True,
                partial_out_ptr=partial_out,
                partial_s0=partial_out.stride(0),
                partial_s1=partial_out.stride(1),
                partial_s2=partial_out.stride(2),
                partial_s3=partial_out.stride(3),
                **pv_launch_meta,
            )
        else:
            _attn_pv_kernel[(num_queries, num_head_blocks, num_dim_blocks * page_split)](
                output,
                p_fp4,
                p_sf,
                kv_cache,
                kv_cache,
                v_sf,
                global_scale,
                src_page_ids,
                metadata.paged_kv_indptr_decode,
                kv_lens,
                src_page_ids.shape[0],
                kv_cache.shape[0],
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.shape[0] * output.shape[1],
                p_fp4.stride(0),
                p_fp4.stride(1),
                p_fp4.shape[0],
                kv_cache.stride(0),
                kv_cache.stride(2),
                kv_cache.stride(4),
                v_sf.stride(0),
                NUM_HEADS=num_heads,
                V_HEAD_D=kv_lora_rank,
                PAGE_SIZE=metadata.page_size,
                FP4_BLOCK=FP4_BLOCK_SIZE,
                SF_PER_PAGE=sf_per_page,
                QUERY_LEN_PER_SEQ=query_len_per_seq,
                MAX_PAGES=max_pages,
                P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
                BLOCK_H=block_h,
                BLOCK_V=block_v,
                USE_TMA_P_LOAD=use_tma_data_load and assume_full_heads and assume_valid_pages,
                USE_TMA_V_LOAD=use_tma_data_load and kv_lora_rank % block_v == 0,
                USE_PREPACKED_V=False,
                PV_LOOP_STAGES=pv_loop_stages,
                ASSUME_FULL_HEADS=assume_full_heads,
                ASSUME_FULL_PAGES=assume_full_pages,
                ASSUME_FULL_V=assume_full_v,
                ASSUME_VALID_PAGES=assume_valid_pages,
                PAGE_SPLIT=page_split,
                PAGES_PER_SPLIT=pages_per_split,
                PARTIAL_OUT=True,
                partial_out_ptr=partial_out,
                partial_s0=partial_out.stride(0),
                partial_s1=partial_out.stride(1),
                partial_s2=partial_out.stride(2),
                partial_s3=partial_out.stride(3),
                **pv_launch_meta,
            )
        _attn_pv_reduce_kernel[(num_queries, num_head_blocks, num_dim_blocks)](
            output,
            partial_out,
            global_scale,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            partial_out.stride(0),
            partial_out.stride(1),
            partial_out.stride(2),
            partial_out.stride(3),
            NUM_HEADS=num_heads,
            V_HEAD_D=kv_lora_rank,
            PAGE_SPLIT=page_split,
            P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
            BLOCK_H=block_h,
            BLOCK_V=block_v,
            ASSUME_FULL_HEADS=assume_full_heads,
            ASSUME_FULL_V=assume_full_v,
            **launch_meta,
        )
    else:
        if use_triton_v_packed_cache:
            _attn_pv_prepacked_v_kernel[(num_queries, num_head_blocks, num_dim_blocks)](
                output,
                p_fp4,
                p_sf,
                v_packed,
                v_sf,
                global_scale,
                src_page_ids,
                metadata.paged_kv_indptr_decode,
                kv_lens,
                src_page_ids.shape[0],
                kv_cache.shape[0],
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.shape[0] * output.shape[1],
                p_fp4.stride(0),
                p_fp4.stride(1),
                p_fp4.shape[0],
                v_sf.stride(0),
                NUM_HEADS=num_heads,
                V_HEAD_D=kv_lora_rank,
                PAGE_SIZE=metadata.page_size,
                FP4_BLOCK=FP4_BLOCK_SIZE,
                SF_PER_PAGE=sf_per_page,
                QUERY_LEN_PER_SEQ=query_len_per_seq,
                MAX_PAGES=max_pages,
                P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
                BLOCK_H=block_h,
                BLOCK_V=block_v,
                USE_TMA_P_LOAD=use_tma_data_load and assume_full_heads and assume_valid_pages,
                USE_TMA_OUT_STORE=use_tma_data_load and assume_full_heads and assume_full_v,
                PV_LOOP_STAGES=pv_loop_stages,
                ASSUME_FULL_HEADS=assume_full_heads,
                ASSUME_FULL_PAGES=assume_full_pages,
                ASSUME_FULL_V=assume_full_v,
                ASSUME_VALID_PAGES=assume_valid_pages,
                **pv_launch_meta,
            )
        else:
            _attn_pv_kernel[(num_queries, num_head_blocks, num_dim_blocks)](
                output,
                p_fp4,
                p_sf,
                kv_cache,
                kv_cache,
                v_sf,
                global_scale,
                src_page_ids,
                metadata.paged_kv_indptr_decode,
                kv_lens,
                src_page_ids.shape[0],
                kv_cache.shape[0],
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.shape[0] * output.shape[1],
                p_fp4.stride(0),
                p_fp4.stride(1),
                p_fp4.shape[0],
                kv_cache.stride(0),
                kv_cache.stride(2),
                kv_cache.stride(4),
                v_sf.stride(0),
                NUM_HEADS=num_heads,
                V_HEAD_D=kv_lora_rank,
                PAGE_SIZE=metadata.page_size,
                FP4_BLOCK=FP4_BLOCK_SIZE,
                SF_PER_PAGE=sf_per_page,
                QUERY_LEN_PER_SEQ=query_len_per_seq,
                MAX_PAGES=max_pages,
                P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
                BLOCK_H=block_h,
                BLOCK_V=block_v,
                USE_TMA_P_LOAD=use_tma_data_load and assume_full_heads and assume_valid_pages,
                USE_TMA_V_LOAD=use_tma_data_load and kv_lora_rank % block_v == 0,
                USE_PREPACKED_V=False,
                PV_LOOP_STAGES=pv_loop_stages,
                ASSUME_FULL_HEADS=assume_full_heads,
                ASSUME_FULL_PAGES=assume_full_pages,
                ASSUME_FULL_V=assume_full_v,
                ASSUME_VALID_PAGES=assume_valid_pages,
                **pv_launch_meta,
            )


def run_fp4_mla_attention_decode(
    metadata: Any,
    layer_idx: int,
    local_layer: int,
    q: torch.Tensor,
    output: torch.Tensor,
    *,
    sm_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    prequantized_q: torch.Tensor,
    prequantized_q_sf: torch.Tensor,
    q_batch_capacity: int,
) -> None:
    """Run MLA decode with FP4 QK and FP4 PV tensor-core matmuls.

    Q is supplied in its assembled ``[latent, RoPE]`` layout and quantized to
    FP4 directly. QK reads ``[KV-nope, K-RoPE, K-RoPE-residual]`` contiguously
    from the primary cache with swizzled block scales. Softmax probabilities
    are quantized to FP4 per page, and PV repacks V nibbles from the shared KV
    cache while reading the auxiliary V-view scale pool. No BF16 dequantized
    KV workspace is materialized on this path. Callers must supply the packed Q
    and scales produced by the fused generation cache update.
    """
    head_dim = kv_lora_rank + qk_rope_head_dim
    if qk_rope_head_dim != FP4_MLA_K_RESIDUAL_DIM:
        raise ValueError(
            "FP4 MLA K residual attention requires "
            f"qk_rope_head_dim={FP4_MLA_K_RESIDUAL_DIM}, got {qk_rope_head_dim}."
        )
    _validate_fp4_mla_cache_shape(metadata.page_size, head_dim)
    if metadata.page_size != FP4_MLA_TOKENS_PER_BLOCK:
        raise ValueError(
            f"FP4 MLA attention decode requires page_size={FP4_MLA_TOKENS_PER_BLOCK}, "
            f"got {metadata.page_size}."
        )

    if q.ndim != 3 or q.shape[-1] != head_dim:
        raise ValueError(
            "FP4 MLA attention Q must have shape "
            f"[tokens, heads, {head_dim}], got {tuple(q.shape)}."
        )
    if not q.is_contiguous():
        raise ValueError("FP4 MLA attention Q must be contiguous.")

    num_queries = q.shape[0]
    if num_queries == 0:
        raise ValueError("FP4 MLA attention decode requires at least one query token.")
    num_gen_seqs = metadata.num_seqs - metadata.num_contexts
    query_len_per_seq = _get_linear_mtp_query_len_per_seq(
        metadata,
        num_queries=num_queries,
        num_gen_seqs=num_gen_seqs,
    )

    num_heads = q.shape[1]
    if output.shape[:2] != (num_queries, num_heads):
        raise ValueError("FP4 MLA attention output batch dimensions do not match.")

    backend = _fp4_mla_attention_backend()
    if getattr(metadata, "fp4_mla_v_scale_pool", None) is None:
        raise RuntimeError(
            "FP4 MLA attention decode requires the auxiliary V scale pool to be allocated."
        )

    global_scale = _get_fp4_mla_global_scale(metadata, q.device)
    q_residual_dim = FP4_MLA_Q_RESIDUAL_DIM
    _validate_fp4_mla_attention_q_shape(head_dim, q_residual_dim)

    if prequantized_q is None or prequantized_q_sf is None or q_batch_capacity is None:
        raise RuntimeError(
            "FP4 MLA decode requires Q prequantized by the fused generation cache update."
        )

    capacity = int(q_batch_capacity)
    expected_q_shape = (capacity * num_heads, FP4_MLA_Q_PACKED_DIM)
    expected_q_sf_shape = (
        _get_fp4_mla_swizzled_scale_size(
            capacity * num_heads,
            FP4_MLA_Q_LOGICAL_DIM,
        ),
    )
    if (
        q.dtype != torch.bfloat16
        or capacity <= 0
        or num_queries > capacity
        or tuple(prequantized_q.shape) != expected_q_shape
        or prequantized_q.dtype != torch.uint8
        or not prequantized_q.is_contiguous()
        or tuple(prequantized_q_sf.shape) != expected_q_sf_shape
        or prequantized_q_sf.dtype != torch.float8_e4m3fn
        or not prequantized_q_sf.is_contiguous()
    ):
        raise ValueError("FP4 MLA prequantized Q does not satisfy the fused-Q contract.")
    active_q_rows = num_queries * num_heads
    active_q_sf_bytes = _get_fp4_mla_swizzled_scale_size(
        active_q_rows,
        FP4_MLA_Q_LOGICAL_DIM,
    )
    q_fp4 = prequantized_q[:active_q_rows]
    q_sf = prequantized_q_sf[:active_q_sf_bytes]
    q_global_scale = _get_fp4_mla_q_global_scale(metadata, q.device)

    kv_cache, sf_cache = _get_fp4_mla_kv_cache_tensors(metadata, layer_idx)
    _validate_fp4_mla_kv_storage_shape(
        kv_cache,
        sf_cache,
        head_dim=head_dim,
        backend=backend,
    )
    sf_cache = sf_cache.view(torch.float8_e4m3fn)

    v_sf_pool = metadata.fp4_mla_v_scale_pool
    v_sf = get_fp4_mla_v_scale_pool_view(metadata, v_head_dim=kv_lora_rank)[local_layer].view(
        torch.float8_e4m3fn
    )
    # The kv_lens runtime alias can lag at the decode anchor (seq_lens == 1) under
    # CUDA graph / one-engine MTP; recover the true total per sequence so the
    # per-query causal masking sees the full 1 + draft_len window (no-op when the
    # alias already matches).
    kv_lens, _ = _fp4_mla_uniform_generation_lengths(metadata, num_queries, num_gen_seqs)
    _materialize_fp4_mla_device_page_table_for_forward(metadata, kv_lens)
    src_page_ids = _fp4_mla_generation_page_ids(metadata, num_gen_seqs)
    max_pages = _max_generation_pages(metadata)
    if max_pages == 0:
        raise RuntimeError("FP4 MLA attention decode requires generation cache pages.")
    if backend == _FP4_MLA_CUTEDSL_BACKEND:
        if get_sm_version() != 107:
            raise RuntimeError(
                "FP4 MLA cutedsl attention backend requires Rubin SM107; "
                f"current architecture is SM{get_sm_version()}."
            )
        if not _cutedsl_backend_available():
            raise RuntimeError(
                "FP4 MLA cutedsl attention backend requires the Rubin CTM and "
                "CuTeDSL runtime packages."
            )
        if not 0 < num_heads <= 128 or kv_lora_rank != 512:
            raise ValueError(
                "FP4 MLA cutedsl attention requires 1-128 local heads and "
                f"kv_lora_rank=512, got num_heads={num_heads}, "
                f"kv_lora_rank={kv_lora_rank}."
            )

        cutedsl_kernel = _fp4_mla_cutedsl_kernel_module()
        QK_LOGICAL_DIM = cutedsl_kernel.QK_LOGICAL_DIM
        QK_SF_GROUPS = cutedsl_kernel.QK_SF_GROUPS
        SMEM_P4_V_N_PER_CTA = cutedsl_kernel.SMEM_P4_V_N_PER_CTA
        run_trtllm_fp4_mla_decode_page_native_from_raw = (
            cutedsl_kernel.run_trtllm_fp4_mla_decode_page_native_from_raw
        )

        physical_heads = 128
        kernel_q = q_fp4
        kernel_q_sf = q_sf
        if num_heads < physical_heads:
            kernel_q_storage, kernel_q_sf_storage, q_batch_capacity = _prepare_fp4_mla_q_buffers(
                metadata,
                num_queries,
                physical_heads,
                q.device,
            )
            active_q_rows = num_queries * physical_heads
            active_q_sf_bytes = _get_fp4_mla_swizzled_scale_size(
                active_q_rows,
                QK_LOGICAL_DIM,
            )
            kernel_q = kernel_q_storage[:active_q_rows]
            kernel_q_sf = kernel_q_sf_storage[:active_q_sf_bytes]
            _cutedsl_pad_q_and_sf_kernel[(num_queries, _ceil_div(QK_LOGICAL_DIM // 2, 64))](
                kernel_q,
                q_fp4,
                kernel_q_sf,
                q_sf,
                num_heads,
                output_heads=physical_heads,
                packed_dim=QK_LOGICAL_DIM // 2,
                block_bytes=64,
                sf_cols=QK_SF_GROUPS,
                sf_cols_per_byte_block=8,
            )

        fused_v_transpose = _fp4_mla_cutedsl_fused_v_transpose_enabled()
        if fused_v_transpose:
            # The fusion kernel reads V from the canonical KV cache and uses
            # the current layer V scales directly. Keep a None placeholder so
            # the mufu16 and fused-V launchers share one Python call site.
            core_v_packed = None
            core_v_sf = v_sf
            v_page_offset = 0
        else:
            v_packed = _get_cutedsl_persistent_v_packed_cache(
                metadata,
                local_layer,
                kv_cache,
                v_head_dim=kv_lora_rank,
                page_size=metadata.page_size,
                block_v=SMEM_P4_V_N_PER_CTA,
            )
            core_v_packed = _get_fp4_mla_v_packed_pool_base(metadata)
            if core_v_packed is None:
                raise RuntimeError("Persistent FP4 MLA V packing requires a stable full-pool base.")
            get_v_page_offset = getattr(
                metadata.kv_cache_manager, "get_mla_v_packed_page_offset", None
            )
            v_page_offset = (
                int(get_v_page_offset(local_layer))
                if callable(get_v_page_offset)
                else local_layer * kv_cache.shape[0]
            )
            page_bytes = kv_lora_rank * (metadata.page_size // 2)
            expected_layer_ptr = core_v_packed.data_ptr() + v_page_offset * page_bytes
            if expected_layer_ptr != v_packed.data_ptr():
                raise RuntimeError(
                    "Persistent FP4 MLA V-packed layer view does not match its "
                    "full-pool base and page offset."
                )

            v_sf_pool_base = _get_fp4_mla_v_scale_pool_base(metadata)
            if v_sf_pool_base is None:
                v_sf_pool_base = v_sf_pool.flatten(0, 1).view(torch.uint8)
                if v_sf_pool_base.data_ptr() != v_sf_pool.data_ptr():
                    raise RuntimeError(
                        "Persistent FP4 MLA V-scale pool must flatten without a copy."
                    )
            if (
                not isinstance(v_sf_pool_base, torch.Tensor)
                or v_sf_pool_base.dtype != torch.uint8
                or v_sf_pool_base.device != v_sf.device
                or v_sf_pool_base.ndim != 2
                or v_sf_pool_base.shape[1] != v_sf_pool.shape[-1]
                or not v_sf_pool_base.is_contiguous()
            ):
                raise RuntimeError(
                    "Persistent FP4 MLA V-scale pool base must be a contiguous "
                    "two-dimensional uint8 tensor with the configured page stride."
                )
            core_v_sf = v_sf_pool_base.view(torch.float8_e4m3fn)
            get_v_sf_page_offset = getattr(
                metadata.kv_cache_manager, "get_mla_v_scale_page_offset", None
            )
            v_sf_page_offset = (
                int(get_v_sf_page_offset(local_layer))
                if callable(get_v_sf_page_offset)
                else local_layer * kv_cache.shape[0]
            )
            if v_sf_page_offset != v_page_offset:
                raise RuntimeError(
                    "Persistent FP4 MLA V-packed and V-scale pools require "
                    "matching encoded layer offsets."
                )
            expected_v_sf_ptr = (
                core_v_sf.data_ptr()
                + v_sf_page_offset * v_sf_pool.stride(1) * v_sf_pool.element_size()
            )
            if expected_v_sf_ptr != v_sf.data_ptr():
                raise RuntimeError(
                    "Persistent FP4 MLA V-scale layer view does not match its "
                    "full-pool base and page offset."
                )

        kernel_output = output
        if num_heads < physical_heads:
            kernel_output = _ensure_workspace_tensor(
                metadata,
                "_fp4_mla_cutedsl_output_buf",
                (num_queries, physical_heads, kv_lora_rank),
                dtype=output.dtype,
                device=output.device,
            )

        run_trtllm_fp4_mla_decode_page_native_from_raw(
            kernel_q,
            kernel_q_sf,
            kv_cache,
            sf_cache,
            core_v_packed,
            core_v_sf,
            global_scale,
            src_page_ids,
            metadata.paged_kv_indptr_decode[: num_gen_seqs + 1],
            kv_lens,
            kernel_output,
            max_kv_len=max_pages * metadata.page_size,
            sm_scale=float(sm_scale),
            num_heads=physical_heads,
            q_global_scale=q_global_scale,
            page_size=metadata.page_size,
            query_len_per_seq=query_len_per_seq,
            v_page_offset=v_page_offset,
            q_batch_capacity=q_batch_capacity,
            partition_runtime_valid_k=bool(getattr(metadata, "is_cuda_graph", False)),
        )
        if kernel_output is not output:
            output.copy_(kernel_output[:, :num_heads])
        return

    total_p_rows = num_queries * max_pages * num_heads
    p_fp4 = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_p_buf",
        (max(total_p_rows, 1), metadata.page_size // 2),
        dtype=torch.uint8,
        device=q.device,
    )[:total_p_rows]
    p_sf = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_p_sf_buf",
        (max(_get_fp4_mla_swizzled_scale_size(total_p_rows, metadata.page_size), 1),),
        dtype=torch.float8_e4m3fn,
        device=q.device,
    )
    stats_shape = (num_queries, num_heads)
    max_scores = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_max_buf",
        stats_shape,
        dtype=torch.float32,
        device=q.device,
    )
    denom = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_denom_buf",
        stats_shape,
        dtype=torch.float32,
        device=q.device,
    )

    if backend != "triton":
        raise ValueError(
            f"Unsupported FP4 MLA attention backend '{backend}'. "
            f"Set {FP4_MLA_ATTENTION_BACKEND_ENV} to 'triton' or "
            f"'{_FP4_MLA_CUTEDSL_BACKEND}'."
        )

    # Self-contained public-Triton path: TMA-loaded QK + fused page-stats pack,
    # reduce-stats, prob-scale, and PV with an optional prepacked V cache.
    _run_triton_attention_decode(
        metadata=metadata,
        layer_idx=layer_idx,
        local_layer=local_layer,
        q_fp4=q_fp4,
        q_sf=q_sf.contiguous().view(-1),
        kv_cache=kv_cache,
        sf_cache=sf_cache,
        v_sf=v_sf,
        global_scale=global_scale,
        src_page_ids=src_page_ids,
        kv_lens=kv_lens,
        p_fp4=p_fp4,
        p_sf=p_sf,
        max_scores=max_scores,
        denom=denom,
        output=output,
        num_queries=num_queries,
        num_heads=num_heads,
        head_dim=head_dim,
        kv_lora_rank=kv_lora_rank,
        q_residual_dim=q_residual_dim,
        query_len_per_seq=query_len_per_seq,
        max_pages=max_pages,
        sm_scale=float(sm_scale),
        q_global_scale=q_global_scale,
    )
