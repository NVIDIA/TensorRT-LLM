# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch-shared FP4 MLA state with metadata/CUDA-graph buffer lifetimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from .config import FP4_MLA_KV_GLOBAL_SCALE, FP4_MLA_Q_GLOBAL_SCALE, HP_BLOCK_SIZE
from .metadata import configure_fp4_mla_device_page_table, populate_fp4_mla_append_metadata

if TYPE_CHECKING:
    from ....memory_buffer_utils import Buffers
    from ..trtllm import TrtllmAttentionMetadata
    from .fp4_mla_context import _Fp8MlaContextScratch


@dataclass
class Fp4MlaState:
    """State shared by RoPE/cache append and all FP4 FMHAs in one batch.

    Allocate through the metadata buffer allocator so eager forwards and each
    CUDA graph retain their existing storage/aliasing rules. This state is not
    layer-local: MTP and metadata preparation update it before any FMHA runs.
    """

    hp_pool: torch.Tensor | None = None
    hp_page_indices: torch.Tensor | None = None
    v_scale_pool: torch.Tensor | None = None
    q_global_scale: torch.Tensor | None = None
    kv_global_scale: torch.Tensor | None = None
    batch_indices: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    generation_kv_lens: torch.Tensor | None = None
    generation_append_lens: torch.Tensor | None = None
    generation_lengths_num_tokens: int = -1
    generation_lengths_num_seqs: int = -1
    generation_lengths_num_contexts: int = -1
    generation_lengths_capture_recorded: bool = False
    generation_cache_scattered: bool = False
    device_page_table: bool = False
    device_page_table_valid: bool = False
    page_table_stride: int = 0
    context_repack_max_touched_pages: int = 1
    cache_pool_id: int = 0
    cache_page_index_scale: int = 0
    hp_pool_id: int = 0
    hp_page_index_scale: int = 0
    prequantized_q: torch.Tensor | None = None
    prequantized_q_sf: torch.Tensor | None = None
    q_batch_capacity: int | None = None
    fp8_context_state: tuple[_Fp8MlaContextScratch, TrtllmAttentionMetadata] | None = None
    _paged_kv_indptr: torch.Tensor | None = None
    paged_kv_indptr_decode: torch.Tensor | None = None
    _paged_kv_indices: torch.Tensor | None = None
    num_sequences: int = 0
    num_blocks: list[int] | None = None
    num_context_blocks: int = 0
    num_generation_blocks: int = 0
    full_pages_cache: tuple[tuple, bool] | None = None
    workspaces: dict[str, torch.Tensor] = field(default_factory=dict)
    v_packed_cache_tags: dict[str, tuple] = field(default_factory=dict)

    @property
    def paged_kv_indices(self) -> torch.Tensor:
        if self._paged_kv_indices is None:
            raise RuntimeError("FP4 MLA paged_kv_indices is not allocated.")
        return self._paged_kv_indices[: self.num_context_blocks + self.num_generation_blocks]

    @property
    def paged_kv_indptr(self) -> torch.Tensor:
        if self._paged_kv_indptr is None:
            raise RuntimeError("FP4 MLA paged_kv_indptr is not allocated.")
        return self._paged_kv_indptr[: self.num_sequences + 1]

    @classmethod
    def create(cls, metadata: TrtllmAttentionMetadata, buffers: Buffers | None) -> Fp4MlaState:
        manager = metadata.kv_cache_manager
        hp_ring_size = manager.fp4_mla_hp_pool_size
        if hp_ring_size < HP_BLOCK_SIZE:
            raise RuntimeError(
                "FP4 MLA high-precision ring must retain at least one "
                f"{HP_BLOCK_SIZE}-token quantization tile, got {hp_ring_size} slots."
            )
        hp_pool = manager.get_fp4_mla_hp_pool()
        fp4_layers = manager._fp4_mla_compact_to_local
        if (
            not isinstance(hp_pool, torch.Tensor)
            or hp_pool.dtype != torch.bfloat16
            or hp_pool.device.type != "cuda"
            or hp_pool.ndim != 4
            or hp_pool.shape[1] != len(fp4_layers)
            or hp_pool.shape[2] != manager.kv_factor
            or hp_pool.shape[3] != hp_ring_size * manager.head_dim_per_layer[fp4_layers[0]]
        ):
            raise RuntimeError(
                "FP4 MLA V2 HP pool must be a CUDA BF16 tensor shaped "
                "[pages, fp4_layers, kv_factor, ring * head_dim], got "
                f"{getattr(hp_pool, 'shape', None)}."
            )

        def allocate(
            name: str, shape: tuple[int, ...], dtype: torch.dtype = torch.int32
        ) -> torch.Tensor:
            return metadata.get_empty(
                buffers,
                shape,
                cache_name=f"fp4_mla_{name}",
                dtype=dtype,
                capture_graph=metadata.is_cuda_graph,
            )

        state = cls(hp_pool=hp_pool, v_scale_pool=manager.get_mla_v_scale_pool())
        tokens = (metadata.max_num_tokens,)
        sequences = (metadata.max_num_sequences,)
        indptr = (metadata.max_num_sequences + 1,)
        pages = (metadata.max_num_sequences * int(manager.max_blocks_per_seq),)
        state.batch_indices = allocate("batch_indices", tokens)
        state.positions = allocate("positions", tokens)
        state.generation_kv_lens = allocate("generation_kv_lens", sequences)
        state.generation_append_lens = allocate("generation_append_lens", sequences)
        state._paged_kv_indices = allocate("paged_kv_indices", pages)
        state.hp_page_indices = allocate("hp_page_indices", pages)
        state._paged_kv_indptr = allocate("paged_kv_indptr", indptr)
        state.paged_kv_indptr_decode = allocate("paged_kv_indptr_decode", indptr)
        state.q_global_scale = allocate("q_global_scale", (1,), torch.float32)
        state.q_global_scale.fill_(FP4_MLA_Q_GLOBAL_SCALE)
        state.kv_global_scale = allocate("kv_global_scale", (1,), torch.float32)
        state.kv_global_scale.fill_(FP4_MLA_KV_GLOBAL_SCALE)
        return state

    def invalidate_generation_lengths(self) -> None:
        self.generation_lengths_num_tokens = -1
        self.generation_lengths_num_seqs = -1
        self.generation_lengths_num_contexts = -1
        self.generation_lengths_capture_recorded = False

    def prepare(self, metadata: TrtllmAttentionMetadata, kv_lens: torch.Tensor) -> None:
        # All local layers reuse this FP8 view only within the current batch.
        self.fp8_context_state = None
        self.invalidate_generation_lengths()
        if metadata.kv_cache_manager is None or metadata.request_ids is None:
            raise RuntimeError(
                "FP4 MLA device page metadata requires a KV cache manager and request IDs."
            )
        if not configure_fp4_mla_device_page_table(metadata, kv_lens):
            raise RuntimeError(
                "FP4 MLA requires fixed-stride device page metadata; the "
                "current KV-cache manager or batch layout is unsupported."
            )
        # Graph capture's on_update_kv_lens owns the append kernel. Uniform
        # generation derives token positions in the fused update instead.
        if not metadata.is_cuda_graph and metadata.num_tokens > 0 and metadata.num_contexts > 0:
            self.populate_append_metadata(metadata)

    def on_update_kv_lens(self, metadata: TrtllmAttentionMetadata) -> None:
        self.device_page_table_valid = False
        if metadata.num_tokens > 0:
            self.invalidate_generation_lengths()
            if metadata.num_contexts > 0:
                self.populate_append_metadata(metadata)

    def update_for_spec_dec(self, metadata: TrtllmAttentionMetadata) -> None:
        num_seqs = metadata.num_seqs
        metadata.prompt_lens_cuda_runtime = metadata.seq_lens_kv_cuda[:num_seqs]
        if not torch.cuda.is_current_stream_capturing():
            metadata.prompt_lens_cpu_runtime = metadata.seq_lens_kv[:num_seqs]
        self.on_update_kv_lens(metadata)

    def restore_from_spec_dec(self, metadata: TrtllmAttentionMetadata) -> None:
        # Rebind aliases to the restored stable tensors, never to the temporary
        # spec-dec clone whose storage may be released after graph capture.
        num_seqs = metadata.num_seqs
        metadata.kv_lens_cuda_runtime = metadata.kv_lens_cuda[:num_seqs]
        metadata.prompt_lens_cuda_runtime = metadata.seq_lens_kv_cuda[:num_seqs]
        if not torch.cuda.is_current_stream_capturing():
            metadata.prompt_lens_cpu_runtime = metadata.seq_lens_kv[:num_seqs]

    def populate_append_metadata(self, metadata: TrtllmAttentionMetadata) -> None:
        num_seqs = metadata.num_contexts + metadata.num_generations
        if num_seqs == 0 or metadata.num_tokens == 0:
            return
        assert self.batch_indices is not None
        assert self.positions is not None
        assert metadata.kv_lens_cuda_runtime is not None
        assert metadata.prompt_lens_cuda_runtime is not None
        # Canonical tensors can change between MTP sub-steps. Do not capture
        # stale *_runtime views when writing token positions and batch indices.
        append_lens = metadata.seq_lens_kv_cuda[:num_seqs]
        if not append_lens.is_cuda:
            raise RuntimeError("FP4 MLA append metadata requires CUDA sequence lengths.")
        populate_fp4_mla_append_metadata(
            append_lens,
            metadata.kv_lens_cuda[:num_seqs],
            self.batch_indices,
            self.positions,
            num_tokens=metadata.num_tokens,
            num_sequences=num_seqs,
            num_contexts=metadata.num_contexts,
            num_context_tokens=metadata.num_ctx_tokens,
        )
