from typing import TYPE_CHECKING, Optional

import torch
from triton import next_power_of_2

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention_backend.vanilla import VanillaAttentionMetadata
from tensorrt_llm._utils import prefer_pinned

ModelConfig = tensorrt_llm.bindings.ModelConfig

if TYPE_CHECKING:
    pass


class RocketTrtllmAttentionMetadata(TrtllmAttentionMetadata):
    def __post_init__(self):
        super().__post_init__()
        if self.sparse_attention_config is None:
            raise ValueError("Sparse attention config is not set")
        self.prompt_budget = self.sparse_attention_config.prompt_budget
        self.window_size = self.sparse_attention_config.window_size
        self.page_size = self.sparse_attention_config.page_size
        self.topk = self.sparse_attention_config.topk

        assert self.page_size == next_power_of_2(self.page_size), "Page size must be a power of 2"

        capture_graph = self.is_cuda_graph

        # Cumulative valid sequence lengths for query and key
        self.q_cu_seqlens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1,),
            dtype=torch.int32,
            cache_name="q_cu_seqlens_cuda",
            capture_graph=capture_graph,
        )

        self.q_cu_seqlens = torch.zeros_like(
            self.q_cu_seqlens_cuda, device="cpu", dtype=torch.int32
        )

        self.k_cu_seqlens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1,),
            dtype=torch.int32,
            cache_name="k_cu_seqlens_cuda",
            capture_graph=capture_graph,
        )
        self.k_cu_seqlens = torch.zeros_like(
            self.k_cu_seqlens_cuda, device="cpu", dtype=torch.int32
        )

        # Context length of RocketKV key for each valid sequence
        self.k_context_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences,),
            dtype=torch.int32,
            cache_name="k_context_lens_cuda",
            capture_graph=capture_graph,
        )
        self.k_context_lens = torch.zeros_like(
            self.k_context_lens_cuda, device="cpu", dtype=torch.int32
        )

        # Start index of RocketKV key for each valid sequence
        self.k_context_start_cuda = self.get_empty(
            None,
            (self.max_num_sequences,),
            dtype=torch.int32,
            cache_name="k_context_start_cuda",
            capture_graph=capture_graph,
        )

        # Cumulative context lengths for each sequence
        self.context_cumsum_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1,),
            dtype=torch.int32,
            cache_name="context_cumsum_cuda",
            capture_graph=capture_graph,
        )
        self.context_cumsum = torch.zeros_like(
            self.context_cumsum_cuda, device="cpu", dtype=torch.int32
        )

        # Sparse kv indices offsets for each sequence in context phase
        self.sparse_offsets_ctx_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1,),
            dtype=torch.int32,
            cache_name="sparse_offsets_ctx_cuda",
            capture_graph=capture_graph,
        )
        self.sparse_offsets_ctx = torch.zeros_like(
            self.sparse_offsets_ctx_cuda, device="cpu", dtype=torch.int32
        )

        # Valid sequence indices used in sparse kv indices prediction
        self.valid_seq_indices_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences,),
            dtype=torch.int32,
            cache_name="valid_seq_indices_cuda",
            capture_graph=capture_graph,
        )

        # KT cache block offsets used in KT cache related kernels
        self.kt_cache_block_offsets = self.get_empty(
            self.cuda_graph_buffers,
            [self.max_num_sequences, self.kv_cache_manager.max_kt_blocks_per_seq],
            dtype=torch.int32,
            cache_name="kt_cache_block_offsets",
            capture_graph=capture_graph,
        )

        self.host_kt_cache_block_offsets = torch.zeros_like(
            self.kt_cache_block_offsets,
            device="cpu",
            pin_memory=prefer_pinned(),
        )

        # Number of KT tokens for each sequence
        self.num_kt_tokens = torch.empty(
            self.max_num_sequences,
            device="cpu",
            dtype=torch.int32,
        )

        # Cumulative KT lengths for each sequence
        self.cum_kt_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1,),
            dtype=torch.int32,
            cache_name="cum_kt_lens_cuda",
            capture_graph=capture_graph,
        )
        self.cum_kt_lens = torch.zeros_like(self.cum_kt_lens_cuda, device="cpu", dtype=torch.int32)

        # Sparse attn indices offsets for each sequence in generation phase
        self.sparse_offsets_gen_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1,),
            dtype=torch.int32,
            cache_name="sparse_offsets_gen_cuda",
            capture_graph=capture_graph,
        )
        self.sparse_offsets_gen = torch.zeros_like(
            self.sparse_offsets_gen_cuda, device="cpu", dtype=torch.int32
        )

        # Maximum number of KT tokens
        self.max_kt_tokens = (self.max_seq_len + self.page_size - 1) // self.page_size

    @property
    def kt_tokens_per_block(self) -> Optional[int]:
        """
        Returns the number of kt tokens per block from the KV cache manager.
        """
        return (
            self.kv_cache_manager.kt_tokens_per_block if self.kv_cache_manager is not None else None
        )

    def prepare(self):
        if self.kv_cache_manager is not None:
            num_contexts = self.num_contexts
            num_generations = self.num_generations
            num_requests = num_contexts + num_generations

            for i in range(num_requests):
                if i < num_contexts:
                    self.kv_cache_params.num_cached_tokens_per_seq[i] = 0
                else:
                    if self.prompt_lens[i] > self.prompt_budget:
                        self.kv_cache_params.num_cached_tokens_per_seq[i] += (
                            self.prompt_budget - self.prompt_lens[i]
                        )

        super().prepare()

        # Update prompt lens for sparse attention
        if self.kv_cache_manager is not None:
            _prompt_lens = self.prompt_lens.copy()
            for i in range(num_requests):
                if i >= num_contexts:
                    _prompt_lens[i] = min(_prompt_lens[i], self.prompt_budget)
            _prompt_lens = torch.tensor(_prompt_lens, dtype=torch.int, device="cpu")
            self.prompt_lens_cpu[: self.num_seqs].copy_(_prompt_lens)
            self.prompt_lens_cuda[: self.num_seqs].copy_(
                self.prompt_lens_cpu[: self.num_seqs], non_blocking=True
            )
            self.prompt_lens_cuda_runtime = self.prompt_lens_cuda[: self.num_seqs]
            self.prompt_lens_cpu_runtime = self.prompt_lens_cpu[: self.num_seqs]

            # for kt cache
            self.kv_cache_manager.copy_kt_block_offsets(
                self.request_ids, self.host_kt_cache_block_offsets
            )
            self.kt_cache_block_offsets[: self.num_seqs].copy_(
                self.host_kt_cache_block_offsets[: self.num_seqs], non_blocking=True
            )

        # -------------------------------- Context phase --------------------------------
        self.context_cumsum[1 : self.num_contexts + 1] = torch.cumsum(
            self.prompt_lens_cpu[: self.num_contexts], dim=0
        )
        self.context_cumsum_cuda[: self.num_contexts + 1].copy_(
            self.context_cumsum[: self.num_contexts + 1], non_blocking=True
        )

        # We need to filter out sequences that are too short to skip sparse kv indices prediction
        valid_mask = self.prompt_lens_cpu[: self.num_contexts] >= self.prompt_budget
        valid_seq_indices = torch.where(valid_mask)[0]
        invalid_seq_indices = torch.where(~valid_mask)[0]
        valid_batch_size = len(valid_seq_indices)
        self.valid_seq_indices_cuda[:valid_batch_size].copy_(valid_seq_indices, non_blocking=True)

        # Only consider sequences that are long enough for sparse kv indices prediction in context phase
        self.k_context_lens[:valid_batch_size] = (
            self.prompt_lens_cpu[valid_seq_indices] - self.window_size
        )
        self.k_context_lens_cuda[:valid_batch_size].copy_(
            self.k_context_lens[:valid_batch_size], non_blocking=True
        )

        sparse_counts_ctx = torch.zeros(self.num_contexts, dtype=torch.int32, device="cpu")
        sparse_counts_ctx[valid_seq_indices] = self.prompt_budget
        sparse_counts_ctx[invalid_seq_indices] = self.prompt_lens_cpu[invalid_seq_indices]

        self.sparse_offsets_ctx[1 : self.num_contexts + 1] = torch.cumsum(sparse_counts_ctx, dim=0)
        self.sparse_offsets_ctx_cuda[: self.num_contexts + 1].copy_(
            self.sparse_offsets_ctx[: self.num_contexts + 1], non_blocking=True
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
        self.total_sparse_ctx_indices = self.sparse_offsets_ctx[self.num_contexts].item()

        # -------------------------------- Generation phase --------------------------------
        self.num_kt_tokens[: self.num_generations] = (
            self.kv_lens[self.num_contexts : self.num_seqs] + self.page_size - 1
        ) // self.page_size

        self.cum_kt_lens[1 : self.num_generations + 1] = torch.cumsum(
            self.num_kt_tokens[: self.num_generations], dim=0
        )
        self.cum_kt_lens_cuda[: self.num_generations + 1].copy_(
            self.cum_kt_lens[: self.num_generations + 1], non_blocking=True
        )

        self.total_kt_tokens = self.num_generations * self.max_kt_tokens

        topk_tensor = torch.tensor(self.topk, dtype=torch.int32)

        # Some sequences may have less than topk KT tokens
        # We need to use the minimum of topk and the number of KT tokens
        sparse_counts_gen = torch.minimum(topk_tensor, self.num_kt_tokens[: self.num_generations])

        self.sparse_offsets_gen[1 : self.num_generations + 1] = torch.cumsum(
            sparse_counts_gen[: self.num_generations], dim=0
        )
        self.sparse_offsets_gen_cuda[: self.num_generations + 1].copy_(
            self.sparse_offsets_gen[: self.num_generations + 1], non_blocking=True
        )

        self.total_sparse_gen_indices = self.topk * self.num_generations


class RocketVanillaAttentionMetadata(VanillaAttentionMetadata):
    def __post_init__(self):
        super().__post_init__()
        if self.sparse_attention_config is None:
            raise ValueError("Sparse attention config is not set")
        self.prompt_budget = self.sparse_attention_config.prompt_budget
        self.kt_cache_block_offsets = torch.empty(
            [self.max_num_sequences, self.kv_cache_manager.max_kt_blocks_per_seq],
            dtype=torch.int32,
            device="cuda",
        )
        self.host_kt_cache_block_offsets = torch.zeros_like(
            self.kt_cache_block_offsets,
            device="cpu",
            pin_memory=prefer_pinned(),
        )

    def prepare(self) -> None:
        super().prepare()
        num_contexts = self.num_contexts
        num_generations = self.num_generations
        num_requests = num_contexts + num_generations

        for i in range(num_requests):
            if i < num_contexts:
                self.kv_cache_params.num_cached_tokens_per_seq[i] = 0
            else:
                if self.prompt_lens[i] > self.prompt_budget:
                    self.kv_cache_params.num_cached_tokens_per_seq[i] += (
                        self.prompt_budget - self.prompt_lens[i]
                    )

        if self.kv_cache_manager is not None:
            # for kt cache
            self.kv_cache_manager.copy_kt_block_offsets(
                self.request_ids, self.host_kt_cache_block_offsets
            )
            self.kt_cache_block_offsets[: self.num_seqs].copy_(
                self.host_kt_cache_block_offsets[: self.num_seqs], non_blocking=True
            )
