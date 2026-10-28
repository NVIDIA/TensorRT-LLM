# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch
from triton import next_power_of_2

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention.backends.sparse.params import SparseAlgorithmExtension
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._utils import prefer_pinned

from .params import RocketKVMetadataParams

ModelConfig = tensorrt_llm.bindings.ModelConfig


class RocketMetadata(SparseAlgorithmExtension):
    """RocketKV state: the prompt budget, the KT cache and the index layouts.

    Serves the TRTLLM and Vanilla backends. On both, a generation request
    attends to at most ``prompt_budget`` prompt tokens, so its cached length is
    reduced before the KV lengths are derived. The TRTLLM backend additionally
    predicts sparse indices from the per-step context and generation layouts
    prepared here; the Vanilla backend predicts per request in Python.
    """

    def __init__(self, metadata) -> None:
        super().__init__(metadata)
        params = metadata.sparse_metadata_params
        if not isinstance(params, RocketKVMetadataParams):
            raise ValueError("RocketKV sparse attention metadata params are not set")
        self.prompt_budget = params.prompt_budget
        self.window_size = params.window_size
        self.page_size = params.page_size
        self.topk = params.topk

    @property
    def _predicts_on_device(self) -> bool:
        return isinstance(self.metadata, TrtllmAttentionMetadata)

    @property
    def kt_tokens_per_block(self) -> Optional[int]:
        """The number of KT tokens per block from the KV cache manager."""
        kv_cache_manager = self.metadata.kv_cache_manager
        return kv_cache_manager.kt_tokens_per_block if kv_cache_manager is not None else None

    def allocate(self, buffers) -> None:
        md = self.metadata
        if not self._predicts_on_device:
            self.kt_cache_block_offsets = torch.empty(
                [md.max_num_sequences, md.kv_cache_manager.max_kt_blocks_per_seq],
                dtype=torch.int32,
                device="cuda",
            )
            self.host_kt_cache_block_offsets = torch.zeros_like(
                self.kt_cache_block_offsets,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            return

        # The TRTLLM kernels read the RocketKV page size as the cache page size.
        md.page_size = self.page_size
        assert self.page_size == next_power_of_2(self.page_size), "Page size must be a power of 2"

        capture_graph = md.is_cuda_graph
        max_num_sequences = md.max_num_sequences

        def device_buffer(shape, name, pool=buffers):
            return md.get_empty(
                pool, shape, dtype=torch.int32, cache_name=name, capture_graph=capture_graph
            )

        def host_mirror(device):
            return torch.zeros_like(device, device="cpu", dtype=torch.int32)

        # Cumulative valid sequence lengths for query and key
        self.q_cu_seqlens_cuda = device_buffer((max_num_sequences + 1,), "q_cu_seqlens_cuda")
        self.q_cu_seqlens = host_mirror(self.q_cu_seqlens_cuda)
        self.k_cu_seqlens_cuda = device_buffer((max_num_sequences + 1,), "k_cu_seqlens_cuda")
        self.k_cu_seqlens = host_mirror(self.k_cu_seqlens_cuda)

        # Context length of RocketKV key for each valid sequence
        self.k_context_lens_cuda = device_buffer((max_num_sequences,), "k_context_lens_cuda")
        self.k_context_lens = host_mirror(self.k_context_lens_cuda)

        # Start index of RocketKV key for each valid sequence
        self.k_context_start_cuda = device_buffer(
            (max_num_sequences,), "k_context_start_cuda", pool=None
        )

        # Cumulative context lengths for each sequence
        self.context_cumsum_cuda = device_buffer((max_num_sequences + 1,), "context_cumsum_cuda")
        self.context_cumsum = host_mirror(self.context_cumsum_cuda)

        # Sparse kv indices offsets for each sequence in context phase
        self.sparse_offsets_ctx_cuda = device_buffer(
            (max_num_sequences + 1,), "sparse_offsets_ctx_cuda"
        )
        self.sparse_offsets_ctx = host_mirror(self.sparse_offsets_ctx_cuda)

        # Valid sequence indices used in sparse kv indices prediction
        self.valid_seq_indices_cuda = device_buffer((max_num_sequences,), "valid_seq_indices_cuda")

        # KT cache block offsets used in KT cache related kernels
        self.kt_cache_block_offsets = device_buffer(
            [max_num_sequences, md.kv_cache_manager.max_kt_blocks_per_seq],
            "kt_cache_block_offsets",
        )
        self.host_kt_cache_block_offsets = torch.zeros_like(
            self.kt_cache_block_offsets,
            device="cpu",
            pin_memory=prefer_pinned(),
        )

        # Number of KT tokens for each sequence
        self.num_kt_tokens = torch.empty(max_num_sequences, device="cpu", dtype=torch.int32)

        # Cumulative KT lengths for each sequence
        self.cum_kt_lens_cuda = device_buffer((max_num_sequences + 1,), "cum_kt_lens_cuda")
        self.cum_kt_lens = host_mirror(self.cum_kt_lens_cuda)

        # Sparse attn indices offsets for each sequence in generation phase
        self.sparse_offsets_gen_cuda = device_buffer(
            (max_num_sequences + 1,), "sparse_offsets_gen_cuda"
        )
        self.sparse_offsets_gen = host_mirror(self.sparse_offsets_gen_cuda)

        # Maximum number of KT tokens
        self.max_kt_tokens = (md.max_seq_len + self.page_size - 1) // self.page_size

    def pre_derive_lengths(self) -> None:
        """Cap each generation request's cached length at the prompt budget."""
        md = self.metadata
        if self._predicts_on_device and md.kv_cache_manager is None:
            return
        num_contexts = md.num_contexts
        num_requests = num_contexts + md.num_generations
        cached = md.kv_cache_params.num_cached_tokens_per_seq
        for i in range(num_requests):
            if i < num_contexts:
                cached[i] = 0
            elif md.prompt_lens[i] > self.prompt_budget:
                cached[i] += self.prompt_budget - md.prompt_lens[i]

    def prepare(self) -> None:
        md = self.metadata
        if md.kv_cache_manager is None:
            return
        md.kv_cache_manager.copy_kt_block_offsets(md.request_ids, self.host_kt_cache_block_offsets)
        self.kt_cache_block_offsets[: md.num_seqs].copy_(
            self.host_kt_cache_block_offsets[: md.num_seqs], non_blocking=True
        )

    def post_derive(self) -> None:
        if not self._predicts_on_device:
            return
        md = self.metadata
        num_contexts = md.num_contexts
        num_generations = md.num_generations

        if md.kv_cache_manager is not None:
            # Generation requests attend to at most prompt_budget prompt tokens.
            prompt_lens = md.prompt_lens.copy()
            for i in range(num_contexts, num_contexts + num_generations):
                prompt_lens[i] = min(prompt_lens[i], self.prompt_budget)
            md.prompt_lens_cpu[: md.num_seqs].copy_(
                torch.tensor(prompt_lens, dtype=torch.int, device="cpu")
            )
            md.prompt_lens_cuda[: md.num_seqs].copy_(
                md.prompt_lens_cpu[: md.num_seqs], non_blocking=True
            )
            md.prompt_lens_cuda_runtime = md.prompt_lens_cuda[: md.num_seqs]
            md.prompt_lens_cpu_runtime = md.prompt_lens_cpu[: md.num_seqs]

        # -------------------------------- Context phase --------------------------------
        self.context_cumsum[1 : num_contexts + 1] = torch.cumsum(
            md.prompt_lens_cpu[:num_contexts], dim=0
        )
        self.context_cumsum_cuda[: num_contexts + 1].copy_(
            self.context_cumsum[: num_contexts + 1], non_blocking=True
        )

        # We need to filter out sequences that are too short to skip sparse kv indices prediction
        valid_mask = md.prompt_lens_cpu[:num_contexts] >= self.prompt_budget
        valid_seq_indices = torch.where(valid_mask)[0]
        invalid_seq_indices = torch.where(~valid_mask)[0]
        valid_batch_size = len(valid_seq_indices)
        self.valid_seq_indices_cuda[:valid_batch_size].copy_(valid_seq_indices, non_blocking=True)

        # Only consider sequences that are long enough for sparse kv indices prediction in context phase
        self.k_context_lens[:valid_batch_size] = (
            md.prompt_lens_cpu[valid_seq_indices] - self.window_size
        )
        self.k_context_lens_cuda[:valid_batch_size].copy_(
            self.k_context_lens[:valid_batch_size], non_blocking=True
        )

        sparse_counts_ctx = torch.zeros(num_contexts, dtype=torch.int32, device="cpu")
        sparse_counts_ctx[valid_seq_indices] = self.prompt_budget
        sparse_counts_ctx[invalid_seq_indices] = md.prompt_lens_cpu[invalid_seq_indices]

        self.sparse_offsets_ctx[1 : num_contexts + 1] = torch.cumsum(sparse_counts_ctx, dim=0)
        self.sparse_offsets_ctx_cuda[: num_contexts + 1].copy_(
            self.sparse_offsets_ctx[: num_contexts + 1], non_blocking=True
        )

        self.q_cu_seqlens[: valid_batch_size + 1] = (
            torch.arange(valid_batch_size + 1, device="cpu", dtype=torch.int32) * self.window_size
        )
        self.q_cu_seqlens_cuda[: valid_batch_size + 1].copy_(
            self.q_cu_seqlens[: valid_batch_size + 1], non_blocking=True
        )

        self.k_cu_seqlens[1 : valid_batch_size + 1] = torch.cumsum(
            self.k_context_lens[:valid_batch_size], dim=0
        )
        self.k_cu_seqlens_cuda[: valid_batch_size + 1].copy_(
            self.k_cu_seqlens[: valid_batch_size + 1], non_blocking=True
        )

        if valid_batch_size > 0:
            # Maximum context length of RocketKV key for valid sequences for padding
            self.max_rocket_k_ctx_len = self.k_context_lens[:valid_batch_size].max().item()
            self.total_rocket_k_ctx_tokens = self.k_cu_seqlens[valid_batch_size].item()
        else:
            self.max_rocket_k_ctx_len = 0
            self.total_rocket_k_ctx_tokens = 0

        self.valid_batch_size = valid_batch_size
        self.total_sparse_ctx_indices = self.sparse_offsets_ctx[num_contexts].item()

        # -------------------------------- Generation phase --------------------------------
        self.num_kt_tokens[:num_generations] = (
            md.kv_lens[num_contexts : md.num_seqs] + self.page_size - 1
        ) // self.page_size

        self.cum_kt_lens[1 : num_generations + 1] = torch.cumsum(
            self.num_kt_tokens[:num_generations], dim=0
        )
        self.cum_kt_lens_cuda[: num_generations + 1].copy_(
            self.cum_kt_lens[: num_generations + 1], non_blocking=True
        )

        self.total_kt_tokens = num_generations * self.max_kt_tokens

        topk_tensor = torch.tensor(self.topk, dtype=torch.int32)

        # Some sequences may have less than topk KT tokens
        # We need to use the minimum of topk and the number of KT tokens
        sparse_counts_gen = torch.minimum(topk_tensor, self.num_kt_tokens[:num_generations])

        self.sparse_offsets_gen[1 : num_generations + 1] = torch.cumsum(
            sparse_counts_gen[:num_generations], dim=0
        )
        self.sparse_offsets_gen_cuda[: num_generations + 1].copy_(
            self.sparse_offsets_gen[: num_generations + 1], non_blocking=True
        )

        self.total_sparse_gen_indices = self.topk * num_generations
