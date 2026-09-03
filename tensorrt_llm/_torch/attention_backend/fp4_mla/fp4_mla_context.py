# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch
import triton

from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.quantization.mode import QuantMode

from .fp4_mla_kernels import _fp8_mla_context_block_table_kernel

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


_FP8_CONTEXT_SUPPORTED_SMS = {90, 100, 103, 107, 120}
_FP8_CONTEXT_SCRATCH_ATTR = "_fp4_mla_fp8_context_scratch"
_FP8_CONTEXT_ATTN_ATTR = "_fp4_mla_fp8_context_attn"


def require_fp4_mla_fp8_context_support() -> None:
    """Validate support for TRT-LLM's FP8 MLA context FMHA."""
    sm = get_sm_version()
    if sm not in _FP8_CONTEXT_SUPPORTED_SMS:
        supported_sms = ", ".join(
            str(supported_sm) for supported_sm in sorted(_FP8_CONTEXT_SUPPORTED_SMS)
        )
        raise RuntimeError(
            f"FP4 MLA context requires TRT-LLM FP8 FMHA, which is not "
            f"supported on SM{sm}. Supported SM versions: {supported_sms}."
        )


def _execute_fp8_context_with_cache_update(
    attention_fn: Callable[[], None],
    cache_update_fn: Callable[[], None],
    aux_stream: torch.cuda.Stream,
    start_event: torch.cuda.Event,
    done_event: torch.cuda.Event,
) -> None:
    """Overlap FP8 context attention with its independent FP4 cache update."""
    current_stream = torch.cuda.current_stream()
    if aux_stream == current_stream:
        raise RuntimeError(
            "FP4 MLA context cache update requires a distinct auxiliary CUDA stream."
        )

    start_event.record(current_stream)
    # Launch the auxiliary work first so the main stream cannot drain the
    # attention queue before the cache update has been submitted.
    with torch.cuda.stream(aux_stream):
        aux_stream.wait_event(start_event)
        cache_update_fn()
        done_event.record(aux_stream)
    attention_fn()
    current_stream.wait_event(done_event)


@dataclass(frozen=True, slots=True)
class _Fp8MlaContextCacheManagerView:
    """Minimal cache-manager view consumed by TRT-LLM FP8 context FMHA."""

    tokens_per_block: int
    max_seq_len: int
    kv_cache_pool_pointers: torch.Tensor
    kv_cache_pool_mapping: torch.Tensor
    layer_offsets: Tuple[int, ...]


@dataclass
class _Fp8MlaContextScratch:
    pool: torch.Tensor
    block_offsets: torch.Tensor
    block_ids_per_seq: torch.Tensor
    host_pool_pointers: torch.Tensor
    host_pool_mapping: torch.Tensor
    host_total_kv_lens: torch.Tensor
    max_num_sequences: int
    max_blocks_per_seq: int
    capacity_blocks: int
    page_size: int
    head_dim: int
    cache_stream: torch.cuda.Stream
    cache_start_event: torch.cuda.Event
    cache_done_event: torch.cuda.Event
    cache_manager_view: _Fp8MlaContextCacheManagerView
    mapping_signature: Optional[Tuple[int, ...]] = None

    @classmethod
    def create(
        cls,
        meta: "TrtllmAttentionMetadata",
        *,
        device: torch.device,
        head_dim: int,
    ) -> "_Fp8MlaContextScratch":
        kv_cache_manager = meta.kv_cache_manager
        if kv_cache_manager is None:
            raise RuntimeError("FP8 MLA context scratch requires a KV cache manager.")

        page_size = int(meta.tokens_per_block)
        max_num_sequences = int(meta.max_num_sequences or meta.max_num_requests)
        max_blocks_per_seq = int(kv_cache_manager.max_blocks_per_seq)
        max_num_tokens = int(meta.max_num_tokens)
        max_nonempty_sequences = min(max_num_sequences, max_num_tokens)
        capacity_blocks = max(
            1,
            (max_num_tokens + page_size - 1) // page_size + max(0, max_nonempty_sequences - 1),
        )
        pool = torch.empty(
            capacity_blocks * page_size * head_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        block_offsets = torch.zeros(
            1,
            max_num_sequences,
            2,
            max_blocks_per_seq,
            dtype=torch.int32,
            device=device,
        )
        block_ids_per_seq = torch.zeros(
            max_num_sequences,
            max_blocks_per_seq,
            dtype=torch.int32,
            device=device,
        )
        host_pool_pointers = torch.tensor(
            [[pool.data_ptr(), 0]],
            dtype=torch.int64,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        host_pool_mapping = torch.zeros(
            1,
            2,
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        cache_manager_view = _Fp8MlaContextCacheManagerView(
            tokens_per_block=page_size,
            max_seq_len=int(kv_cache_manager.max_seq_len),
            kv_cache_pool_pointers=host_pool_pointers,
            kv_cache_pool_mapping=host_pool_mapping,
            layer_offsets=(0,),
        )
        host_total_kv_lens = torch.zeros(
            2,
            dtype=meta.host_total_kv_lens.dtype,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        return cls(
            pool=pool,
            block_offsets=block_offsets,
            block_ids_per_seq=block_ids_per_seq,
            host_pool_pointers=host_pool_pointers,
            host_pool_mapping=host_pool_mapping,
            host_total_kv_lens=host_total_kv_lens,
            max_num_sequences=max_num_sequences,
            max_blocks_per_seq=max_blocks_per_seq,
            capacity_blocks=capacity_blocks,
            page_size=page_size,
            head_dim=head_dim,
            cache_stream=torch.cuda.Stream(device=device),
            cache_start_event=torch.cuda.Event(),
            cache_done_event=torch.cuda.Event(),
            cache_manager_view=cache_manager_view,
        )

    def matches(
        self,
        meta: "TrtllmAttentionMetadata",
        *,
        device: torch.device,
        head_dim: int,
    ) -> bool:
        kv_cache_manager = meta.kv_cache_manager
        return (
            kv_cache_manager is not None
            and self.pool.device == device
            and self.head_dim == head_dim
            and self.page_size == meta.tokens_per_block
            and self.cache_manager_view.max_seq_len == int(kv_cache_manager.max_seq_len)
            and self.max_num_sequences >= int(meta.max_num_sequences or meta.max_num_requests)
            and self.max_blocks_per_seq >= int(kv_cache_manager.max_blocks_per_seq)
        )

    def prepare(self, meta: "TrtllmAttentionMetadata") -> None:
        context_lengths = tuple(
            int(length) for length in meta.prompt_lens_cpu_runtime[: meta.num_contexts].tolist()
        )
        self.host_total_kv_lens[0] = sum(context_lengths)
        self.host_total_kv_lens[1] = 0
        if context_lengths == self.mapping_signature:
            return

        if len(context_lengths) > self.max_num_sequences:
            raise ValueError(
                f"FP8 MLA context scratch supports at most {self.max_num_sequences} sequences, "
                f"got {len(context_lengths)}."
            )
        required_blocks = 0
        for context_length in context_lengths:
            if context_length < 0:
                raise ValueError(f"Context length must be non-negative, got {context_length}.")
            num_blocks = (context_length + self.page_size - 1) // self.page_size
            if num_blocks > self.max_blocks_per_seq:
                raise ValueError(
                    f"Context sequence requires {num_blocks} FP8 scratch blocks, but the "
                    f"page table holds only {self.max_blocks_per_seq}."
                )
            required_blocks += num_blocks
        if required_blocks > self.capacity_blocks:
            raise RuntimeError(
                f"FP8 MLA context scratch requires {required_blocks} blocks, but only "
                f"{self.capacity_blocks} were allocated."
            )

        context_lengths_cuda = meta.prompt_lens_cuda_runtime
        if (
            context_lengths_cuda.dtype != torch.int32
            or not context_lengths_cuda.is_cuda
            or context_lengths_cuda.ndim != 1
            or context_lengths_cuda.stride(0) != 1
            or context_lengths_cuda.numel() < meta.num_contexts
        ):
            raise ValueError(
                "FP8 MLA context metadata requires a contiguous CUDA int32 context-length tensor."
            )
        # ModelEngine's context warmup reaches this launch before serving.
        # Only engine-lifetime capacities are constexpr; varying request
        # counts and sequence lengths reuse the warmup specialization.
        _fp8_mla_context_block_table_kernel[(self.max_num_sequences,)](
            context_lengths_cuda,
            self.block_offsets,
            self.block_ids_per_seq,
            meta.num_contexts,
            PAGE_SIZE=self.page_size,
            MAX_BLOCKS_PER_SEQUENCE=self.max_blocks_per_seq,
            SEQUENCE_BLOCK=triton.next_power_of_2(self.max_num_sequences),
            PAGE_BLOCK=triton.next_power_of_2(self.max_blocks_per_seq),
            num_warps=8,
        )
        self.mapping_signature = context_lengths


def _build_fp8_mla_context_attn(attn: "TrtllmAttention") -> "TrtllmAttention":
    """Build a direct-attribute FP8 view without per-access Python forwarding."""
    fp8_attn = copy.copy(attn)
    fp8_attn.quant_mode = int(QuantMode(0).set_fp8_kv_cache())
    fp8_attn.has_fp4_kv_cache = False
    fp8_attn.has_fp8_kv_cache = True
    # FMHA instances hold weak references to their owning attention object.
    # Do not reuse instances copied from the FP4 attention; the caller binds
    # this FP8 view explicitly to TRTLLM's regular FMHA implementation.
    fp8_attn.fmha_libs = []
    fp8_attn.local_layer_idx = 0
    # This branch resolves local cache layers through layer_idx. The
    # disposable cache has exactly one layer, so bind the copied view to it.
    fp8_attn.layer_idx = 0
    return fp8_attn


def _build_fp8_mla_context_metadata(
    meta: "TrtllmAttentionMetadata",
    scratch: _Fp8MlaContextScratch,
) -> "TrtllmAttentionMetadata":
    """Route the mandatory FP8 cache write through a direct metadata view."""
    fp8_meta = copy.copy(meta)
    fp8_meta._fp4_mla_fp8_context_state = None
    fp8_meta.kv_cache_manager = scratch.cache_manager_view
    fp8_meta.kv_cache_block_offsets = scratch.block_offsets
    fp8_meta.block_ids_per_seq = scratch.block_ids_per_seq
    fp8_meta.prompt_lens_cuda_runtime = meta.prompt_lens_cuda_runtime[: meta.num_contexts]
    fp8_meta.prompt_lens_cpu_runtime = meta.prompt_lens_cpu_runtime[: meta.num_contexts]
    fp8_meta.host_request_types_runtime = meta.host_request_types_runtime[: meta.num_contexts]
    # The disposable cache represents only this context invocation. Expose
    # exact context-only lengths so cached prefixes, trailing generation
    # metadata, and stale totals cannot extend FP8 K/V quantization.
    fp8_meta.kv_lens_cuda_runtime = meta.prompt_lens_cuda_runtime[: meta.num_contexts]
    fp8_meta.kv_lens_runtime = meta.prompt_lens_cpu_runtime[: meta.num_contexts]
    fp8_meta.host_total_kv_lens = scratch.host_total_kv_lens
    # Scratch lengths intentionally start from zero. Preserve the actual
    # absolute positions for Q/K RoPE through the native kernel's explicit
    # per-token position-offset input.
    fp8_meta.helix_position_offsets = meta.positions[: meta.num_ctx_tokens]
    return fp8_meta


def _get_fp8_mla_context_metadata(
    meta: "TrtllmAttentionMetadata",
    scratch: _Fp8MlaContextScratch,
) -> "TrtllmAttentionMetadata":
    """Prepare and reuse one direct metadata view for all layers in a step."""
    state = meta._fp4_mla_fp8_context_state
    if state is not None and state[0] is scratch:
        return state[1]

    scratch.prepare(meta)
    fp8_meta = _build_fp8_mla_context_metadata(meta, scratch)
    meta._fp4_mla_fp8_context_state = (scratch, fp8_meta)
    return fp8_meta
