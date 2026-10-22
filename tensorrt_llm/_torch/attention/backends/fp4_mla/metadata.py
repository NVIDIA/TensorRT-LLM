# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA page tables, append metadata, and generation-length preparation."""

from typing import Any, Optional

import torch
import triton
import triton.language as tl

from .config import _FP4_MLA_PAGE_TABLE_TILE_SIZE, FP4_MLA_TOKENS_PER_BLOCK, _ceil_div


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
        raise RuntimeError("FP4 MLA requires V2 cache-layout page metadata.")
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

    Eager context and generation batches receive the full block-offset
    table on the GPU. The materialization kernel decodes V2 page indices and
    refreshes rows from the final device KV lengths before cache update.
    """
    metadata.fp4_mla_state.device_page_table = False
    metadata.fp4_mla_state.device_page_table_valid = False
    metadata.fp4_mla_state.page_table_stride = 0
    metadata.fp4_mla_state.context_repack_max_touched_pages = 1

    kv_cache_manager = getattr(metadata, "kv_cache_manager", None)
    num_contexts = int(getattr(metadata, "num_contexts", 0))
    num_sequences = int(getattr(metadata, "num_seqs", 0))
    num_generation_sequences = num_sequences - num_contexts
    num_tokens = int(getattr(metadata, "num_tokens", 0))
    num_context_tokens = int(getattr(metadata, "num_ctx_tokens", 0))
    num_generation_tokens = num_tokens - num_context_tokens
    block_offsets = getattr(metadata, "kv_cache_block_offsets", None)
    page_ids = getattr(metadata.fp4_mla_state, "_paged_kv_indices", None)
    paged_kv_indptr = getattr(metadata.fp4_mla_state, "_paged_kv_indptr", None)
    paged_kv_indptr_decode = getattr(metadata.fp4_mla_state, "paged_kv_indptr_decode", None)
    max_page_capacity = int(getattr(kv_cache_manager, "max_blocks_per_seq", 0) or 0)
    page_spec = _fp4_mla_page_table_spec(kv_cache_manager)
    page_index_scale = int(page_spec.cache_page_index_scale)

    tensors = (block_offsets, page_ids, paged_kv_indptr, paged_kv_indptr_decode)
    is_cuda_graph = bool(getattr(metadata, "is_cuda_graph", False))
    generation_only = num_contexts == 0
    eager_context = not is_cuda_graph and num_contexts > 0
    has_valid_generation = num_generation_sequences == 0 or (
        num_generation_tokens >= num_generation_sequences
        and num_generation_tokens % num_generation_sequences == 0
    )
    # NVFP4 exposes one data pool plus its paired block-scale pool. The
    # materializer reads encoded data offsets from pool 0.
    supported = (
        (generation_only or eager_context)
        and kv_cache_manager is not None
        and has_valid_generation
        and int(getattr(metadata, "beam_width", 1)) == 1
        and not bool(getattr(metadata, "is_spec_dec_tree", False))
        and not bool(getattr(metadata, "locality_domain_enabled", False))
        and int(getattr(kv_cache_manager, "tokens_per_block", 0) or 0) == FP4_MLA_TOKENS_PER_BLOCK
        and max_page_capacity > 0
        and page_index_scale > 0
        and all(isinstance(tensor, torch.Tensor) for tensor in tensors)
        and all(tensor.dtype == torch.int32 for tensor in tensors)
        and all(tensor.is_cuda for tensor in tensors)
    )
    if not supported:
        return False

    hp_page_ids = getattr(metadata.fp4_mla_state, "hp_page_indices", None)
    max_pool_id = max(page_spec.cache_pool_id, page_spec.hp_pool_id)
    if (
        not isinstance(hp_page_ids, torch.Tensor)
        or hp_page_ids.dtype != torch.int32
        or not hp_page_ids.is_cuda
        or block_offsets.shape[0] <= max_pool_id
    ):
        return False
    metadata.fp4_mla_state.cache_pool_id = int(page_spec.cache_pool_id)
    metadata.fp4_mla_state.cache_page_index_scale = int(page_spec.cache_page_index_scale)
    metadata.fp4_mla_state.hp_pool_id = int(page_spec.hp_pool_id)
    metadata.fp4_mla_state.hp_page_index_scale = int(page_spec.hp_page_index_scale)

    max_pages = max_page_capacity
    host_kv_lens_available = (
        isinstance(kv_lens, torch.Tensor)
        and kv_lens.device.type == "cpu"
        and kv_lens.ndim == 1
        and kv_lens.numel() >= num_sequences
    )
    if eager_context and num_generation_sequences > 0 and not host_kv_lens_available:
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
            metadata.fp4_mla_state.context_repack_max_touched_pages = min(
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
    if metadata.fp4_mla_state.hp_page_indices.numel() < required_page_ids:
        return False

    metadata.fp4_mla_state.device_page_table = True
    metadata.fp4_mla_state.num_sequences = num_sequences
    metadata.fp4_mla_state.page_table_stride = max_pages
    metadata.fp4_mla_state.num_blocks = None
    metadata.fp4_mla_state.num_context_blocks = num_contexts * max_pages
    metadata.fp4_mla_state.num_generation_blocks = num_generation_sequences * max_pages
    return True


def materialize_fp4_mla_device_page_table(
    metadata: Any,
    kv_lens: torch.Tensor,
    generation_kv_lens: Optional[torch.Tensor] = None,
) -> None:
    """Refresh the fixed-stride context and generation page table once per forward."""
    if not bool(getattr(metadata.fp4_mla_state, "device_page_table", False)):
        raise RuntimeError("FP4 MLA requires fixed-stride device page metadata.")
    if bool(getattr(metadata.fp4_mla_state, "device_page_table_valid", False)):
        return

    num_contexts = int(metadata.num_contexts)
    num_sequences = int(metadata.num_seqs)
    num_generation_sequences = num_sequences - num_contexts
    max_pages = int(metadata.fp4_mla_state.page_table_stride)
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

    cache_pool_id = int(getattr(metadata.fp4_mla_state, "cache_pool_id", 0))
    block_offsets = metadata.kv_cache_block_offsets[
        cache_pool_id,
        :num_sequences,
        0,
        :max_pages,
    ]
    page_ids = metadata.fp4_mla_state._paged_kv_indices[: num_sequences * max_pages]
    page_index_scale = int(metadata.fp4_mla_state.cache_page_index_scale)
    if page_index_scale <= 0:
        raise RuntimeError("FP4 MLA device page metadata requires a positive page-index scale.")
    grid = (
        num_sequences,
        triton.cdiv(max_pages, _FP4_MLA_PAGE_TABLE_TILE_SIZE),
    )
    _fp4_mla_materialize_page_table_kernel[grid](
        page_ids,
        metadata.fp4_mla_state._paged_kv_indptr,
        metadata.fp4_mla_state.paged_kv_indptr_decode,
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
    hp_page_ids = getattr(metadata.fp4_mla_state, "hp_page_indices", None)
    if not isinstance(hp_page_ids, torch.Tensor):
        raise RuntimeError("FP4 MLA requires an HP page-table output tensor.")
    hp_pool_id = int(metadata.fp4_mla_state.hp_pool_id)
    hp_page_index_scale = int(metadata.fp4_mla_state.hp_page_index_scale)
    hp_block_offsets = metadata.kv_cache_block_offsets[
        hp_pool_id,
        :num_sequences,
        0,
        :max_pages,
    ]
    _fp4_mla_materialize_page_table_kernel[grid](
        hp_page_ids,
        metadata.fp4_mla_state._paged_kv_indptr,
        metadata.fp4_mla_state.paged_kv_indptr_decode,
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
    metadata.fp4_mla_state.device_page_table_valid = True


def _host_int_list_during_forward(value: Any, start: int, end: int) -> Optional[list[int]]:
    if torch.cuda.is_current_stream_capturing():
        return None
    return _host_int_list(value, start, end)


def _fp4_mla_generation_num_blocks_device(metadata: Any) -> torch.Tensor:
    """Device-side scalar view holding the generation page-table capacity.

    ``paged_kv_indptr_decode[num_gen]`` is the fixed-stride generation-table
    endpoint. Device kernels combine it with the live KV lengths, so inactive
    slots are never consumed.
    """
    num_gen = metadata.num_seqs - metadata.num_contexts
    return metadata.fp4_mla_state.paged_kv_indptr_decode[num_gen : num_gen + 1]


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
    corrected_kv_lens = getattr(metadata.fp4_mla_state, "generation_kv_lens", None)
    generation_lens = getattr(metadata.fp4_mla_state, "generation_append_lens", None)
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
        and not getattr(metadata.fp4_mla_state, "generation_lengths_capture_recorded", False)
    )
    precomputed = (
        not record_for_capture
        and metadata.fp4_mla_state.generation_lengths_num_tokens == num_gen_tokens
        and metadata.fp4_mla_state.generation_lengths_num_seqs == num_gen
        and metadata.fp4_mla_state.generation_lengths_num_contexts == num_contexts
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
        metadata.fp4_mla_state.generation_lengths_num_tokens = num_gen_tokens
        metadata.fp4_mla_state.generation_lengths_num_seqs = num_gen
        metadata.fp4_mla_state.generation_lengths_num_contexts = num_contexts
        if record_for_capture:
            metadata.fp4_mla_state.generation_lengths_capture_recorded = True
    return corrected_kv_lens[:num_gen], generation_lens[:num_gen]


def _materialize_fp4_mla_device_page_table_for_forward(
    metadata: Any,
    generation_kv_lens: Optional[torch.Tensor] = None,
) -> None:
    """Materialize all fixed-stride rows from final per-forward device lengths."""
    if not bool(getattr(metadata.fp4_mla_state, "device_page_table", False)):
        raise RuntimeError("FP4 MLA cache update requires fixed-stride device page metadata.")
    if bool(getattr(metadata.fp4_mla_state, "device_page_table_valid", False)):
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


def _max_generation_pages(metadata: Any) -> int:
    num_gen = metadata.num_seqs - metadata.num_contexts
    if num_gen <= 0:
        return 0
    if not getattr(metadata.fp4_mla_state, "device_page_table", False):
        raise RuntimeError("FP4 MLA generation requires fixed-stride device page metadata.")
    max_pages = int(metadata.fp4_mla_state.page_table_stride)
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
    page_ids = getattr(metadata.fp4_mla_state, "_paged_kv_indices", None)
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
    page_ids = getattr(metadata.fp4_mla_state, "hp_page_indices", None)
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
    block_counts = _host_int_list(getattr(metadata.fp4_mla_state, "num_blocks", None), start, end)
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
        cache = getattr(metadata.fp4_mla_state, "full_pages_cache", None)
        if cache is not None and cache[0] == cache_key:
            return bool(cache[1])
        kv_lens = [int(item) for item in kv_lens_cuda[start:end].detach().cpu().tolist()]
        result = bool(kv_lens) and min(kv_lens) == max(kv_lens) == max_pages * page_size
        setattr(metadata.fp4_mla_state, "full_pages_cache", (cache_key, result))
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
