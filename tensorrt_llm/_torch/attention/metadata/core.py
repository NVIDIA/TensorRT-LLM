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
"""Core metadata of the TRT-LLM attention backend."""

from __future__ import annotations

import contextlib
import dataclasses
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

import torch

from tensorrt_llm._utils import maybe_pin_memory, prefer_pinned

from .base import Metadata, _Registry, build_metadata

if TYPE_CHECKING:
    from ...metadata import KVCacheParams
    from .batch_inputs import BatchInputs


class TrtllmCoreMetadata(Metadata):
    """Lengths, counts, KV-cache layout, kernel views and the attention workspace.

    This is the object models, workers and attention ops receive as
    ``attn_metadata``. The fields of the other attached metadata read
    through it by name.
    """

    # Features of the TRT-LLM backend that this metadata does not carry hold
    # their disabled values.
    is_cross = False
    cross = None
    has_cross_sub_metadata = False
    enable_flash_mla = False
    enable_helix = False
    fp4_mla_state = None
    multi_item_part_lens = None
    position_ids = None
    cu_q_seqlens = None
    cu_kv_seqlens = None
    max_context_q_len_override = None
    host_uploader = None
    helix_position_offsets = None
    helix_is_inactive_rank = None
    helix_context_local_slots = None
    helix_local_slots = None
    helix_kv_bounds = None
    _helix_spec_tokens_valid = False
    block_ids_per_seq = None
    flash_mla_tile_scheduler_metadata = None
    flash_mla_num_splits = None
    decode_kv_regime = None
    # The draft loop never modifies this metadata, so there is nothing to restore.
    has_spec_dec_saved_state = False
    decode_packed_split = None

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        self.max_num_requests = config.max_num_requests
        self.max_num_tokens = config.max_num_tokens
        self.max_num_sequences = config.max_num_sequences
        self.kv_cache_manager = config.kv_cache_manager
        self.draft_kv_cache_manager = config.draft_kv_cache_manager
        self.mapping = config.mapping
        self.runtime_features = config.runtime_features
        self.kv_layout = config.kv_layout
        self.cache_indirection = config.cache_indirection
        self.max_num_heads_per_kv = config.max_num_heads_per_kv
        self.is_cuda_graph = config.is_cuda_graph
        self.cuda_graph_buffers = config.cuda_graph_buffers
        features = config.runtime_features
        self.use_paged_context_fmha = bool(
            features is not None
            and (
                features.chunked_prefill
                or features.cache_reuse
                or features.has_speculative_draft_tokens
            )
        )

        # The step, as the last prepare() received it.
        self.request_ids: Optional[List[int]] = None
        self.prompt_lens: Optional[List[int]] = None
        self.kv_cache_params: Optional["KVCacheParams"] = None
        self.beam_width = 1
        self.trtllm_gen_jit_warmup = False
        self.num_chunked_ctx_requests = 0
        # Token counts of attention data parallelism; the engine sets them
        # after the count exchange that follows prepare().
        self.padded_num_tokens: Optional[int] = None
        self.all_rank_num_tokens: Optional[List[int]] = None

        self._seq_lens: Optional[torch.Tensor] = None
        self._num_contexts = 0
        self._num_generations = 0
        self._num_ctx_tokens = 0
        self._num_tokens = 0

        self.workspace = torch.empty((0,), device="cuda", dtype=torch.int8)
        self.cuda_graph_workspace = torch.empty((0,), device="cuda", dtype=torch.int8)
        self.workspace_reclaimable = True
        # Plan caches of the FMHA libraries, keyed by library.
        self.fmha_plan_caches: Dict[str, dict] = {}
        # Context-only plan shared by the layers of one forward.
        self.context_plan = None
        self._cute_dsl_mla_split_plan: Optional[torch.Tensor] = None
        self._cute_dsl_mla_split_plan_key: Optional[tuple] = None

        num_rows = self.max_num_sequences
        self.prompt_lens_cuda = self.get_buffer((num_rows,), torch.int, "prompt_lens_cuda")
        self.prompt_lens_cpu = torch.empty_like(
            self.prompt_lens_cuda, device="cpu", pin_memory=prefer_pinned()
        )
        self.kv_lens_cuda = self.get_buffer((num_rows,), torch.int, "kv_lens_cuda")
        # Valid KV length after this step, extra draft tokens included.
        self.kv_lens = torch.empty_like(self.kv_lens_cuda, device="cpu", pin_memory=prefer_pinned())
        self.host_total_kv_lens = torch.empty(2, device="cpu", dtype=torch.int)
        self.host_request_types = torch.empty_like(self.prompt_lens_cpu)

        # Exclusive prefix sums the MLA FMHA kernels read: Q rows and KV
        # lengths of the generation rows, Q tokens of the context rows.
        self.mla_cu_q_rows = self.get_buffer((num_rows + 1,), torch.int, "mla_cu_q_rows")
        self.mla_cu_kv_seqlens = self.get_buffer((num_rows + 1,), torch.int, "mla_cu_kv_seqlens")
        self.mla_ctx_cu_q_seqlens = self.get_buffer(
            (num_rows + 1,), torch.int, "mla_ctx_cu_q_seqlens"
        )
        self.mla_cu_q_rows[:1].zero_()
        self.mla_cu_kv_seqlens[:1].zero_()
        self.mla_ctx_cu_q_seqlens[:1].zero_()
        self._mla_scheduler_buffers_valid = False
        self._mla_ctx_cu_seqlens_valid = False

        manager = self.kv_cache_manager
        num_pools = getattr(manager, "num_attention_op_pools", manager.num_pools)
        self.kv_cache_block_offsets = self.get_buffer(
            (num_pools, num_rows, 2, manager.max_blocks_per_seq),
            torch.int32,
            "kv_cache_block_offsets",
        )
        self.host_kv_cache_block_offsets = manager.host_kv_cache_block_offsets
        # Lengths as the host staged them. Every device derivation recomputes
        # seq_lens_cuda and kv_lens_cuda from these, so nothing accumulates.
        self.base_seq_lens_cuda = self.get_buffer((num_rows,), torch.int, "base_seq_lens_cuda")
        self.base_kv_lens_cuda = self.get_buffer((num_rows,), torch.int, "base_kv_lens_cuda")
        self._seq_lens_cuda = self.get_buffer((num_rows,), torch.int, "seq_lens_cuda")
        # The description of the last prepared step.
        self.inputs: Optional["BatchInputs"] = None

    # Lengths and counts.

    @property
    def seq_lens(self) -> Optional[torch.Tensor]:
        """Query length of each sequence, on the host, [num_seqs]."""
        return self._seq_lens

    @property
    def seq_lens_cuda(self) -> Optional[torch.Tensor]:
        return self._seq_lens_cuda[: self.num_seqs]

    @property
    def seq_lens_kv(self) -> Optional[torch.Tensor]:
        return self._seq_lens

    @property
    def seq_lens_kv_cuda(self) -> Optional[torch.Tensor]:
        return self._seq_lens_cuda

    @property
    def num_contexts(self) -> int:
        return self._num_contexts

    @property
    def num_generations(self) -> int:
        return self._num_generations

    @property
    def num_seqs(self) -> int:
        return self._seq_lens.shape[0]

    @property
    def num_ctx_tokens(self) -> int:
        return self._num_ctx_tokens

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def context_lens(self) -> torch.Tensor:
        return self._seq_lens[: self._num_contexts]

    def _set_rows(self, seq_lens: torch.Tensor, num_contexts: int) -> None:
        self._seq_lens = seq_lens
        self._num_contexts = num_contexts
        self._num_ctx_tokens = int(seq_lens[:num_contexts].sum())
        self._num_generations = seq_lens.shape[0] - num_contexts
        self._num_tokens = int(seq_lens.sum())

    def get_adp_token_counts(self) -> List[int]:
        """Local execution-stage counts carried by the input count exchange."""
        return [self.num_tokens]

    def set_adp_token_counts(self, counts: List[List[int]]) -> List[int]:
        """Save stage counts and return the input-stage counts for padding."""
        return counts[0]

    # KV-cache layout.

    @property
    def max_seq_len(self) -> int:
        return self.kv_cache_manager.max_seq_len

    @property
    def max_context_length(self) -> int:
        return min(self.max_seq_len, self.max_num_tokens)

    @property
    def tokens_per_block(self) -> int:
        return self.kv_cache_manager.tokens_per_block

    @property
    def page_size(self) -> int:
        return self.kv_cache_manager.tokens_per_block

    @property
    def host_kv_cache_pool_pointers(self) -> torch.Tensor:
        return self.kv_cache_manager.kv_cache_pool_pointers

    @property
    def host_kv_cache_pool_mapping(self) -> torch.Tensor:
        return self.kv_cache_manager.kv_cache_pool_mapping

    @property
    def effective_beam_width(self) -> int:
        return self.beam_width

    @property
    def effective_workspace(self) -> torch.Tensor:
        """Attention-kernel workspace, the CUDA-graph copy under capture."""
        return self.cuda_graph_workspace if self.is_cuda_graph else self.workspace

    def is_sm_version_trtllm_gen_kernel(self, sm: int) -> bool:
        from tensorrt_llm._utils import is_sm_100f

        return is_sm_100f(sm)

    # Step preparation.

    def prepare(self, inputs: "BatchInputs") -> None:
        self.inputs = inputs
        self.context_plan = None
        self._mla_scheduler_buffers_valid = False
        self._mla_ctx_cu_seqlens_valid = False
        self._cute_dsl_mla_split_plan_key = None
        self.trtllm_gen_jit_warmup = inputs.trtllm_gen_jit_warmup
        self.num_chunked_ctx_requests = (
            inputs.spec.num_extend_ctx_requests if inputs.spec is not None else 0
        )
        if inputs.padding is not None:
            self.padded_num_tokens = inputs.padding.padded_num_tokens
            self.all_rank_num_tokens = inputs.padding.all_rank_num_tokens
        # Host values that follow from the step's shape.
        self._set_rows(inputs.seq_lens, inputs.num_contexts)
        if inputs.stage_host:
            self._stage_host(inputs)
        self._derive_lengths(inputs)

    def _stage_host(self, inputs: "BatchInputs") -> None:
        """Copy the step's host values into the device buffers."""
        self.request_ids = inputs.request_ids
        self.prompt_lens = inputs.prompt_lens
        self.kv_cache_params = inputs.kv_cache_params
        self.beam_width = inputs.beam_width
        num_seqs = self.num_seqs
        num_contexts = self.num_contexts
        seq_lens = maybe_pin_memory(self._seq_lens)
        self._seq_lens = seq_lens
        self.base_seq_lens_cuda[:num_seqs].copy_(seq_lens, non_blocking=True)

        prompt_lens = torch.as_tensor(self.prompt_lens, dtype=torch.int, device="cpu")
        self.prompt_lens_cpu[:num_seqs].copy_(prompt_lens)
        self.prompt_lens_cuda[:num_seqs].copy_(self.prompt_lens_cpu[:num_seqs], non_blocking=True)

        assert self.kv_cache_params.use_cache, "TRT-LLM core metadata requires a KV cache"
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq, dtype=torch.int, device="cpu"
        )
        kv_lens = cached_token_lens + seq_lens
        self.kv_lens[:num_seqs].copy_(kv_lens + self.kv_cache_params.num_extra_kv_tokens)
        # Copied at once, not through the step-input pack: the device
        # derivation that follows reads these lengths.
        self.base_kv_lens_cuda[:num_seqs].copy_(maybe_pin_memory(kv_lens), non_blocking=True)
        self.host_total_kv_lens[0] = kv_lens[:num_contexts].sum().item()
        self.host_total_kv_lens[1] = kv_lens[num_contexts:num_seqs].sum().item()

        manager = self.kv_cache_manager
        # A manager with per-layer sliding-window tables derives its block
        # offsets from them, so it computes them for this batch first.
        compute_tables = getattr(manager, "compute_sliding_block_tables", None)
        if compute_tables is not None:
            compute_tables(self.request_ids, num_contexts)
        max_kv_len = int(self.kv_lens[:num_seqs].max())
        assert max_kv_len <= manager.max_seq_len, (
            f"The max KV cache length of input sequences ({max_kv_len}) "
            f"exceeds the KV cache manager's maximum supported length "
            f"({manager.max_seq_len})."
        )
        # The full block-table width: device-side length updates (overlap
        # scheduler, draft steps) read columns past the host lengths.
        manager.copy_batch_block_offsets(
            self.kv_cache_block_offsets, self.request_ids, self.beam_width, num_contexts, num_seqs
        )
        # The host KV lengths the FMHA kernels read exclude the extra draft tokens.
        self.kv_lens_runtime = kv_lens[:num_seqs]

    def _derive_lengths(self, inputs: "BatchInputs") -> None:
        """Derive this step's device lengths from the staged ones."""
        num_seqs = self.num_seqs
        num_contexts = self.num_contexts
        draft = inputs.draft
        seq_lens_cuda = self._seq_lens_cuda[:num_seqs]
        if draft is not None and draft.step > 0:
            # Later draft steps run one token per sequence.
            seq_lens_cuda.fill_(1)
        else:
            seq_lens_cuda.copy_(self.base_seq_lens_cuda[:num_seqs])
        kv_lens_cuda = self.kv_lens_cuda[:num_seqs]
        kv_lens_cuda.copy_(self.base_kv_lens_cuda[:num_seqs])
        num_extend = self.num_chunked_ctx_requests
        offset_rows = (
            slice(num_contexts - num_extend, num_contexts)
            if num_extend > 0
            else slice(num_contexts, num_seqs)
        )
        num_offset_rows = offset_rows.stop - offset_rows.start
        if inputs.kv_lens_offsets is not None and num_offset_rows > 0:
            kv_lens_cuda[offset_rows] += inputs.kv_lens_offsets[:num_offset_rows]
        if inputs.kv_lens_offset_delta and num_offset_rows > 0:
            kv_lens_cuda[offset_rows] += inputs.kv_lens_offset_delta
        self.host_request_types[:num_contexts].fill_(0)
        self.host_request_types[num_contexts:num_seqs].fill_(1)
        self.kv_lens_cuda_runtime = kv_lens_cuda
        self.prompt_lens_cuda_runtime = self.prompt_lens_cuda[:num_seqs]
        self.prompt_lens_cpu_runtime = self.prompt_lens_cpu[:num_seqs]
        self.host_request_types_runtime = self.host_request_types[:num_seqs]

    def mla_prepare_scheduler_buffers(self, num_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Cumulative Q rows and KV lengths of the generation rows.

        Built once per step on the device from ``seq_lens_cuda`` and
        ``kv_lens_cuda``, so device-side length updates are picked up after
        the next ``prepare``.
        """
        num_contexts = self.num_contexts
        num_generations = self.num_seqs - num_contexts
        if not self._mla_scheduler_buffers_valid:
            if num_generations > 0:
                torch.cumsum(
                    self._seq_lens_cuda[num_contexts : self.num_seqs],
                    0,
                    dtype=torch.int32,
                    out=self.mla_cu_q_rows[1 : num_generations + 1],
                )
                self.mla_cu_q_rows[1 : num_generations + 1] *= num_heads
                torch.cumsum(
                    self.kv_lens_cuda[num_contexts : self.num_seqs],
                    0,
                    dtype=torch.int32,
                    out=self.mla_cu_kv_seqlens[1 : num_generations + 1],
                )
            self._mla_scheduler_buffers_valid = True
        return (
            self.mla_cu_q_rows[: num_generations + 1],
            self.mla_cu_kv_seqlens[: num_generations + 1],
        )

    def mla_prepare_ctx_cu_seqlens(self) -> Optional[torch.Tensor]:
        """Cumulative Q tokens of the context rows; None without context rows."""
        num_contexts = self.num_contexts
        if num_contexts <= 0:
            return None
        if not self._mla_ctx_cu_seqlens_valid:
            torch.cumsum(
                self._seq_lens_cuda[:num_contexts],
                0,
                dtype=torch.int32,
                out=self.mla_ctx_cu_q_seqlens[1 : num_contexts + 1],
            )
            self._mla_ctx_cu_seqlens_valid = True
        return self.mla_ctx_cu_q_seqlens[: num_contexts + 1]

    def create_cuda_graph_metadata(
        self,
        max_batch_size: int,
        sub_cross_metadata: bool = False,
        max_draft_tokens: int = 0,
        buffers: Any = None,
        encode_only: bool = False,
    ) -> "TrtllmCoreMetadata":
        """Metadata of one generation-only CUDA graph batch size.

        A CUDA graph captures the addresses of the metadata buffers, so each
        captured batch size gets metadata of its own, built from ``buffers``.
        """
        if self.is_cuda_graph:
            return self
        assert not encode_only, "encoder CUDA graphs are not supported"
        graph_metadata = build_metadata(
            self.attached_types(),
            dataclasses.replace(self.config, is_cuda_graph=True, cuda_graph_buffers=buffers),
        )
        assert isinstance(graph_metadata, TrtllmCoreMetadata)
        graph_metadata.workspace = self.workspace
        graph_metadata.cuda_graph_workspace = self.cuda_graph_workspace
        graph_metadata._set_rows(
            torch.ones((max_batch_size,), dtype=torch.int) * (1 + max_draft_tokens), 0
        )
        return graph_metadata

    def set_decode_kv_regime(self, kv_regime: Optional[int]) -> None:
        """The decode path of this backend does not depend on the KV regime."""

    def set_decode_packed_split(self, packed_split: Optional[bool]) -> None:
        """This backend has no split decode."""

    def support_steady_gen_step(self) -> bool:
        return False

    # Ragged verification rows and mixed-precision staging are not carried.

    def token_major_gen_view(self) -> None:
        return None

    @contextlib.contextmanager
    def presented_token_major(self, view: Any) -> Iterator[None]:
        assert view is None
        yield

    @contextlib.contextmanager
    def mixed_precision_kv_staging(self, layer_idx: int) -> Iterator[None]:
        yield
