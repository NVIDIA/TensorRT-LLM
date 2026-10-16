# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute active beam rows while retaining the executor's capacity-stride layout."""

import copy
from typing import Any

import torch
import torch.nn.functional as F

from tensorrt_llm._utils import prefer_pinned

from ..attention.backends.trtllm import TrtllmAttentionMetadata
from .scheduler import ScheduledRequests


class CompactBeamView:
    """Stable decode metadata buffers for one (batch size, execution width) shape.

    Rows are request-major. Selecting the first ``width`` rows of each request
    preserves physical KV page IDs; neither the KV allocator nor the sampler
    changes its capacity-stride layout.
    """

    def __init__(self, source: TrtllmAttentionMetadata, width: int, capacity: int):
        if not 1 < width < capacity:
            raise ValueError("Compact execution width must be at least 2 and smaller than capacity")
        self.source = source
        self.capacity = capacity
        self.width = width
        self.batch_size = source.num_seqs // capacity
        self.row_indices_cpu = (
            torch.arange(self.batch_size, device="cpu")[:, None] * capacity
            + torch.arange(width, device="cpu")[None, :]
        ).flatten()
        self.row_indices_list = self.row_indices_cpu.tolist()
        self.row_indices = self.row_indices_cpu.to(source.kv_cache_block_offsets.device)
        self.metadata = metadata = copy.copy(source)
        count = self.batch_size * width
        metadata.beam_width = width
        metadata.max_num_requests = self.batch_size
        metadata.max_num_sequences = count
        metadata._seq_lens = torch.ones(count, dtype=torch.int32, device="cpu")
        metadata._seq_lens_cuda = metadata._seq_lens.to(self.row_indices.device)
        metadata._seq_lens_kv = None
        metadata._seq_lens_kv_cuda = None
        metadata._num_generations = count
        metadata._num_tokens = count
        metadata._num_ctx_tokens = 0
        metadata.padded_num_tokens = None
        metadata.all_rank_num_tokens = [count]
        metadata._saved_tensors = {}
        metadata.fmha_plan_caches = {}
        metadata.draft_replay_swapped_attrs = {}
        metadata.kv_cache_params = copy.copy(source.kv_cache_params)
        for name in self._row_fields:
            original = getattr(source, name)
            setattr(
                metadata,
                name,
                torch.empty(
                    count,
                    dtype=original.dtype,
                    device=original.device,
                    pin_memory=not original.is_cuda and prefer_pinned(),
                ),
            )
        metadata.host_total_kv_lens = torch.empty_like(source.host_total_kv_lens)
        offsets = source.kv_cache_block_offsets
        metadata.kv_cache_block_offsets = offsets.new_empty(
            offsets.shape[0], count, *offsets.shape[2:]
        )
        metadata.cache_indirection = source.cache_indirection.new_empty(
            self.batch_size, width, source.cache_indirection.shape[-1]
        )
        metadata._bind_runtime_views(
            kv_lens_cuda=metadata.kv_lens_cuda,
            kv_lens=metadata.kv_lens,
            prompt_lens_cuda=metadata.prompt_lens_cuda,
            prompt_lens_cpu=metadata.prompt_lens_cpu,
            host_request_types=metadata.host_request_types,
        )

    _row_fields = (
        "kv_lens",
        "kv_lens_cuda",
        "prompt_lens_cpu",
        "prompt_lens_cuda",
        "host_request_types",
    )

    def update(self) -> None:
        """Refresh graph-visible buffers in place, outside graph capture/replay."""
        source, metadata = self.source, self.metadata
        for name in self._row_fields:
            original = getattr(source, name)
            indices = self.row_indices if original.is_cuda else self.row_indices_cpu
            torch.index_select(original, 0, indices, out=getattr(metadata, name))
        torch.index_select(
            source.kv_cache_block_offsets, 1, self.row_indices, out=metadata.kv_cache_block_offsets
        )
        metadata.cache_indirection.copy_(source.cache_indirection[: self.batch_size, : self.width])
        metadata.host_total_kv_lens[0] = 0
        metadata.host_total_kv_lens[1] = metadata.kv_lens.sum()
        metadata.request_ids = source.request_ids
        metadata.prompt_lens = [source.prompt_lens[i] for i in self.row_indices_list]
        metadata.kv_cache_params.num_cached_tokens_per_seq = [
            source.kv_cache_params.num_cached_tokens_per_seq[i] for i in self.row_indices_list
        ]
        metadata.on_update_kv_lens()

    def pack_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Gather inside the graph so replay can consume capacity-stride inputs."""
        return {
            **inputs,
            "attn_metadata": self.metadata,
            "input_ids": inputs["input_ids"].index_select(0, self.row_indices),
            "position_ids": inputs["position_ids"].index_select(-1, self.row_indices),
        }

    def restore_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Inactive rows remain impossible candidates for the unchanged sampler."""
        return F.pad(
            logits.reshape(self.batch_size, self.width, -1),
            (0, 0, 0, self.capacity - self.width),
            value=-float("inf"),
        ).reshape(self.batch_size * self.capacity, -1)


class BeamCompaction:
    """Select configured widths and own metadata views for eager/CUDA graph use."""

    def __init__(self, capacity: int, widths: tuple[int, ...]):
        self.capacity = capacity
        self.widths = widths
        self._views: dict[tuple[int, int, int], CompactBeamView] = {}

    def clear(self) -> None:
        self._views.clear()

    def execution_width(self, batch: ScheduledRequests, has_previous_tokens: bool) -> int:
        if not self.widths or batch.context_requests:
            return 0
        widths = {
            request.get_beam_width_by_iter(for_next_iteration=has_previous_tokens)
            for request in batch.generation_requests
            if not request.is_dummy
        }
        if len(widths) != 1:
            return 0
        width = widths.pop()
        return width if width in self.widths else 0

    def supports_inputs(self, inputs: dict[str, Any]) -> bool:
        if (
            not self.widths
            or inputs.get("multimodal_params")
            or any(
                inputs.get(key) is not None
                for key in (
                    "inputs_embeds",
                    "spec_metadata",
                    "lora_params",
                    "encoder_hidden_states",
                    "context_requests",
                    "mrope_delta_read_seq_slots",
                    "mrope_delta_write_seq_slots",
                )
            )
        ):
            return False
        metadata = inputs["attn_metadata"]
        return (
            type(metadata) is TrtllmAttentionMetadata
            and not metadata.num_contexts
            and not metadata.has_cross_sub_metadata
            and not metadata.mamba_metadata
            and metadata.beam_width == self.capacity
            and metadata.num_seqs > 0
            and metadata.num_seqs % self.capacity == 0
            and metadata.num_tokens == metadata.num_seqs
            and bool(torch.all(metadata.seq_lens == 1))
            and not metadata.kv_cache_params.num_extra_kv_tokens
            and metadata.kv_cache_block_offsets is not None
            and metadata.cache_indirection is not None
        )

    def prepare_inputs(self, inputs: dict[str, Any], width: int) -> None:
        inputs.pop("beam_compaction", None)
        if width not in self.widths or not self.supports_inputs(inputs):
            return
        source = inputs["attn_metadata"]
        key = (id(source), source.num_seqs, width)
        if key not in self._views:
            if not source.is_cuda_graph:
                # Eager views only need the latest batch shape for each width.
                # CUDA graph views retain stable buffers for replay.
                for cached_key in tuple(self._views):
                    if cached_key[0] == id(source) and cached_key[2] == width:
                        del self._views[cached_key]
            self._views[key] = CompactBeamView(source, width, self.capacity)
        view = self._views[key]
        view.update()
        inputs["beam_compaction"] = view
