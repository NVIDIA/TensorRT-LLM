import math
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from triton import next_power_of_2

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.attention_backend.trtllm import (
    TrtllmAttention, TrtllmAttentionMetadata)
from tensorrt_llm._torch.attention_backend.vanilla import (
    VanillaAttention, VanillaAttentionMetadata, repeat_kv)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import (BlockManager,
                                                             KVCacheManager)
from tensorrt_llm._utils import get_size_in_bytes
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import \
    CacheType as CacheTypeCpp
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from .kernel import (triton_bmm, triton_flatten_to_batch, triton_index_gather,
                     triton_rocket_batch_to_flatten,
                     triton_rocket_paged_kt_cache_bmm, triton_rocket_qk_split,
                     triton_rocket_reduce_scores,
                     triton_rocket_update_kt_cache_ctx,
                     triton_rocket_update_kt_cache_gen, triton_softmax,
                     triton_topk)

ModelConfig = tensorrt_llm.bindings.ModelConfig


class RocketTrtllmAttentionMetadata(TrtllmAttentionMetadata):

    def __post_init__(self):
        super().__post_init__()
        if self.sparse_attention_config is None:
            raise ValueError("Sparse attention config is not set")
        self.prompt_budget = self.sparse_attention_config.prompt_budget
        self.window_size = self.sparse_attention_config.window_size
        self.page_size = self.sparse_attention_config.page_size
        self.topk = self.sparse_attention_config.topk

        assert self.page_size == next_power_of_2(
            self.page_size), "Page size must be a power of 2"

        capture_graph = self.is_cuda_graph

        # Cumulative valid sequence lengths for query and key
        self.q_cu_seqlens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1, ),
            dtype=torch.int32,
            cache_name="q_cu_seqlens_cuda",
            capture_graph=capture_graph,
        )

        self.q_cu_seqlens = torch.zeros_like(self.q_cu_seqlens_cuda,
                                             device='cpu',
                                             dtype=torch.int32)

        self.k_cu_seqlens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1, ),
            dtype=torch.int32,
            cache_name="k_cu_seqlens_cuda",
            capture_graph=capture_graph,
        )
        self.k_cu_seqlens = torch.zeros_like(self.k_cu_seqlens_cuda,
                                             device='cpu',
                                             dtype=torch.int32)

        # Context length of RocketKV key for each valid sequence
        self.k_context_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences, ),
            dtype=torch.int32,
            cache_name="k_context_lens_cuda",
            capture_graph=capture_graph,
        )
        self.k_context_lens = torch.zeros_like(self.k_context_lens_cuda,
                                               device='cpu',
                                               dtype=torch.int32)

        # Start index of RocketKV key for each valid sequence
        self.k_context_start_cuda = self.get_empty(
            None,
            (self.max_num_sequences, ),
            dtype=torch.int32,
            cache_name="k_context_start_cuda",
            capture_graph=capture_graph,
        )

        # Cumulative context lengths for each sequence
        self.context_cumsum_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1, ),
            dtype=torch.int32,
            cache_name="context_cumsum_cuda",
            capture_graph=capture_graph,
        )
        self.context_cumsum = torch.zeros_like(self.context_cumsum_cuda,
                                               device='cpu',
                                               dtype=torch.int32)

        # Sparse kv indices offsets for each sequence in context phase
        self.sparse_offsets_ctx_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1, ),
            dtype=torch.int32,
            cache_name="sparse_offsets_ctx_cuda",
            capture_graph=capture_graph,
        )
        self.sparse_offsets_ctx = torch.zeros_like(self.sparse_offsets_ctx_cuda,
                                                   device='cpu',
                                                   dtype=torch.int32)

        # Valid sequence indices used in sparse kv indices prediction
        self.valid_seq_indices_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences, ),
            dtype=torch.int32,
            cache_name="valid_seq_indices_cuda",
            capture_graph=capture_graph,
        )

        # KT cache block offsets used in KT cache related kernels
        self.kt_cache_block_offsets = self.get_empty(
            self.cuda_graph_buffers,
            [
                self.max_num_sequences,
                self.kv_cache_manager.max_kt_blocks_per_seq
            ],
            dtype=torch.int32,
            cache_name="kt_cache_block_offsets",
            capture_graph=capture_graph,
        )

        self.host_kt_cache_block_offsets = torch.zeros_like(
            self.kt_cache_block_offsets,
            device='cpu',
            pin_memory=True,
        )

        # Number of KT tokens for each sequence
        self.num_kt_tokens = torch.empty(
            self.max_num_sequences,
            device='cpu',
            dtype=torch.int32,
        )

        # Cumulative KT lengths for each sequence
        self.cum_kt_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1, ),
            dtype=torch.int32,
            cache_name="cum_kt_lens_cuda",
            capture_graph=capture_graph,
        )
        self.cum_kt_lens = torch.zeros_like(self.cum_kt_lens_cuda,
                                            device='cpu',
                                            dtype=torch.int32)

        # Sparse attn indices offsets for each sequence in generation phase
        self.sparse_offsets_gen_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1, ),
            dtype=torch.int32,
            cache_name="sparse_offsets_gen_cuda",
            capture_graph=capture_graph,
        )
        self.sparse_offsets_gen = torch.zeros_like(self.sparse_offsets_gen_cuda,
                                                   device='cpu',
                                                   dtype=torch.int32)

        # Maximum number of KT tokens
        self.max_kt_tokens = (self.max_seq_len + self.page_size -
                              1) // self.page_size

    @property
    def kt_tokens_per_block(self) -> Optional[int]:
        """
        Returns the number of kt tokens per block from the KV cache manager.
        """
        return self.kv_cache_manager.kt_tokens_per_block if self.kv_cache_manager is not None else None

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
                        self.kv_cache_params.num_cached_tokens_per_seq[
                            i] += self.prompt_budget - self.prompt_lens[i]

        super().prepare()

        # Update prompt lens for sparse attention
        if self.kv_cache_manager is not None:
            _prompt_lens = self.prompt_lens.copy()
            for i in range(num_requests):
                if i >= num_contexts:
                    _prompt_lens[i] = min(_prompt_lens[i], self.prompt_budget)
            _prompt_lens = torch.tensor(_prompt_lens,
                                        dtype=torch.int,
                                        device='cpu')
            self.prompt_lens_cpu[:self.num_seqs].copy_(_prompt_lens)
            self.prompt_lens_cuda[:self.num_seqs].copy_(
                self.prompt_lens_cpu[:self.num_seqs], non_blocking=True)
            self.prompt_lens_cuda_runtime = self.prompt_lens_cuda[:self.
                                                                  num_seqs]
            self.prompt_lens_cpu_runtime = self.prompt_lens_cpu[:self.num_seqs]

            # for kt cache
            self.kv_cache_manager.copy_kt_block_offsets(
                self.request_ids, self.host_kt_cache_block_offsets)
            self.kt_cache_block_offsets[:self.num_seqs].copy_(
                self.host_kt_cache_block_offsets[:self.num_seqs],
                non_blocking=True)

        # -------------------------------- Context phase --------------------------------
        self.context_cumsum[1:self.num_contexts + 1] = torch.cumsum(
            self.prompt_lens_cpu[:self.num_contexts], dim=0)
        self.context_cumsum_cuda[:self.num_contexts + 1].copy_(
            self.context_cumsum[:self.num_contexts + 1], non_blocking=True)

        # We need to filter out sequences that are too short to skip sparse kv indices prediction
        valid_mask = self.prompt_lens_cpu[:self.
                                          num_contexts] >= self.prompt_budget
        valid_seq_indices = torch.where(valid_mask)[0]
        invalid_seq_indices = torch.where(~valid_mask)[0]
        valid_batch_size = len(valid_seq_indices)
        self.valid_seq_indices_cuda[:valid_batch_size].copy_(valid_seq_indices,
                                                             non_blocking=True)

        # Only consider sequences that are long enough for sparse kv indices prediction in context phase
        self.k_context_lens[:valid_batch_size] = self.prompt_lens_cpu[
            valid_seq_indices] - self.window_size
        self.k_context_lens_cuda[:valid_batch_size].copy_(
            self.k_context_lens[:valid_batch_size], non_blocking=True)

        sparse_counts_ctx = torch.zeros(self.num_contexts,
                                        dtype=torch.int32,
                                        device='cpu')
        sparse_counts_ctx[valid_seq_indices] = self.prompt_budget
        sparse_counts_ctx[invalid_seq_indices] = self.prompt_lens_cpu[
            invalid_seq_indices]

        self.sparse_offsets_ctx[1:self.num_contexts + 1] = torch.cumsum(
            sparse_counts_ctx, dim=0)
        self.sparse_offsets_ctx_cuda[:self.num_contexts + 1].copy_(
            self.sparse_offsets_ctx[:self.num_contexts + 1], non_blocking=True)

        self.q_cu_seqlens[:valid_batch_size + 1] = torch.arange(
            valid_batch_size + 1, device='cpu',
            dtype=torch.int32) * self.window_size
        self.q_cu_seqlens_cuda[:valid_batch_size + 1].copy_(
            self.q_cu_seqlens[:valid_batch_size + 1], non_blocking=True)

        self.k_cu_seqlens[1:valid_batch_size + 1] = torch.cumsum(
            self.k_context_lens[:valid_batch_size], dim=0)
        self.k_cu_seqlens_cuda[:valid_batch_size + 1].copy_(
            self.k_cu_seqlens[:valid_batch_size + 1], non_blocking=True)

        if valid_batch_size > 0:
            # Maximum context length of RocketKV key for valid sequences for padding
            self.max_rocket_k_ctx_len = self.k_context_lens[:
                                                            valid_batch_size].max(
                                                            ).item()
            self.total_rocket_k_ctx_tokens = self.k_cu_seqlens[
                valid_batch_size].item()
        else:
            self.max_rocket_k_ctx_len = 0
            self.total_rocket_k_ctx_tokens = 0

        self.valid_batch_size = valid_batch_size
        self.total_sparse_ctx_indices = self.sparse_offsets_ctx[
            self.num_contexts].item()

        # -------------------------------- Generation phase --------------------------------
        self.num_kt_tokens[:self.num_generations] = (
            self.kv_lens[self.num_contexts:self.num_seqs] + self.page_size -
            1) // self.page_size

        self.cum_kt_lens[1:self.num_generations + 1] = torch.cumsum(
            self.num_kt_tokens[:self.num_generations], dim=0)
        self.cum_kt_lens_cuda[:self.num_generations + 1].copy_(
            self.cum_kt_lens[:self.num_generations + 1], non_blocking=True)

        self.total_kt_tokens = self.num_generations * self.max_kt_tokens

        topk_tensor = torch.tensor(self.topk, dtype=torch.int32)

        # Some sequences may have less than topk KT tokens
        # We need to use the minimum of topk and the number of KT tokens
        sparse_counts_gen = torch.minimum(
            topk_tensor, self.num_kt_tokens[:self.num_generations])

        self.sparse_offsets_gen[1:self.num_generations + 1] = torch.cumsum(
            sparse_counts_gen[:self.num_generations], dim=0)
        self.sparse_offsets_gen_cuda[:self.num_generations + 1].copy_(
            self.sparse_offsets_gen[:self.num_generations + 1],
            non_blocking=True)

        self.total_sparse_gen_indices = self.topk * self.num_generations


class RocketTrtllmAttention(TrtllmAttention):
    Metadata = RocketTrtllmAttentionMetadata

    # Access type for different dtype sizes
    _access_type = {
        1: torch.int8,
        2: torch.int16,
        4: torch.int32,
        8: torch.int64
    }

    def __init__(
            self,
            layer_idx: int,
            num_heads: int,
            head_dim: int,
            num_kv_heads: Optional[int] = None,
            quant_config: Optional[QuantConfig] = None,
            q_scaling: Optional[float] = None,
            sparse_attention_config: Optional["SparseAttentionConfig"] = None,
            **kwargs):
        super().__init__(layer_idx,
                         num_heads,
                         head_dim,
                         sparse_attention_config=sparse_attention_config,
                         num_kv_heads=num_kv_heads,
                         quant_config=quant_config,
                         q_scaling=q_scaling,
                         **kwargs)
        self.topr = sparse_attention_config.topr
        self.topk = sparse_attention_config.topk
        self.prompt_budget = sparse_attention_config.prompt_budget
        self.window_size = sparse_attention_config.window_size
        self.kernel_size = sparse_attention_config.kernel_size
        self.page_size = sparse_attention_config.page_size

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Predict sparse KV indices using optimized SnapKV algorithm.

        Uses a single Triton kernel to compute attention scores between observation window
        and prefix tokens, then selects the most important prefix tokens directly.
        """

        num_ctx_tokens = metadata.num_ctx_tokens
        if num_ctx_tokens == 0:
            return None, None

        if k is None:
            qkv_input = q[:num_ctx_tokens]
        else:
            qkv_input = torch.cat([q, k], dim=1)

        if metadata.valid_batch_size > 0:
            q_window, k_context = triton_rocket_qk_split(
                qkv_input,
                metadata.prompt_lens_cuda,
                metadata.context_cumsum_cuda,
                metadata.valid_seq_indices_cuda,
                metadata.k_cu_seqlens_cuda,
                metadata.total_rocket_k_ctx_tokens,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.window_size,
                metadata.valid_batch_size,
            )

            scores = triton_bmm(q_window,
                                k_context,
                                metadata.q_cu_seqlens_cuda,
                                metadata.k_cu_seqlens_cuda,
                                metadata.valid_batch_size,
                                causal=False)

            scores = triton_softmax(scores, metadata.k_cu_seqlens_cuda,
                                    metadata.valid_batch_size)

            # scores: [num_heads, window_size, total_k_tokens] -> [num_kv_heads, total_k_tokens]
            scores = scores.view(self.num_kv_heads,
                                 self.num_heads // self.num_kv_heads,
                                 self.window_size, -1).sum(dim=(1, 2))

            # Reshape scores to handle variable length sequences with padding using Triton
            # scores: [num_kv_heads, total_k_tokens] -> [valid_batch_size, num_kv_heads, padding_size]
            scores = triton_flatten_to_batch(scores, metadata.k_cu_seqlens_cuda,
                                             metadata.valid_batch_size,
                                             metadata.max_rocket_k_ctx_len)

            scores = torch.nn.functional.max_pool1d(
                scores,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1)

            # Use indexer topk prefill to select topk prefix indices
            total_tasks = metadata.valid_batch_size * self.num_kv_heads

            selected_prefix_indices = torch.empty(
                (total_tasks, self.prompt_budget - self.window_size),
                device=qkv_input.device,
                dtype=torch.int32)

            scores = scores.view(total_tasks, -1)

            row_starts = metadata.k_context_start_cuda[:metadata.
                                                       valid_batch_size].repeat_interleave(
                                                           self.num_kv_heads)
            row_ends = metadata.k_context_lens_cuda[:metadata.
                                                    valid_batch_size].repeat_interleave(
                                                        self.num_kv_heads)
            torch.ops.trtllm.indexer_topk_prefill(
                scores, row_starts, row_ends, selected_prefix_indices,
                self.prompt_budget - self.window_size)

            # Sort selected prefix indices to keep topk indices in ascending order
            selected_prefix_indices = torch.sort(selected_prefix_indices,
                                                 dim=-1).values
        else:
            selected_prefix_indices = torch.empty(
                (0, self.prompt_budget - self.window_size),
                device=qkv_input.device,
                dtype=torch.int32)

        sparse_kv_offsets = metadata.sparse_offsets_ctx_cuda[:metadata.
                                                             num_contexts + 1]

        # Flatten sparse indices
        sparse_kv_indices = triton_rocket_batch_to_flatten(
            selected_prefix_indices, metadata.prompt_lens_cuda,
            metadata.valid_seq_indices_cuda, sparse_kv_offsets,
            metadata.num_contexts, metadata.total_sparse_ctx_indices,
            self.window_size, self.prompt_budget, self.num_kv_heads)

        # Update KT cache
        kt_cache_tensor = metadata.kv_cache_manager.get_kt_buffers(
            self.layer_idx)

        triton_rocket_update_kt_cache_ctx(
            qkv_input.contiguous(),
            kt_cache_tensor,
            metadata.kt_cache_block_offsets[:metadata.num_contexts],
            metadata.context_cumsum_cuda[:metadata.num_contexts + 1],
            sparse_kv_indices,
            sparse_kv_offsets,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            self.prompt_budget,
            metadata.kt_tokens_per_block,
            metadata.kv_cache_manager.max_kt_blocks_per_seq,
        )

        # Reduce overhead of post processing
        if metadata.valid_batch_size == 0:
            return None, None

        return sparse_kv_indices, sparse_kv_offsets

    @torch.compile(dynamic=True)
    def preprocess_for_gen(
        self, q: torch.Tensor, k: Optional[torch.Tensor],
        metadata: RocketTrtllmAttentionMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if k is None:
            qkv_input = q[metadata.num_ctx_tokens:]
            q_hidden_size = self.num_heads * self.head_dim
            k_hidden_size = self.num_kv_heads * self.head_dim
            q = qkv_input[:, :q_hidden_size]
            k = qkv_input[:, q_hidden_size:q_hidden_size + k_hidden_size]
        else:
            q = q[metadata.num_ctx_tokens:]
            k = k[metadata.num_ctx_tokens:]

        q = q.view(-1, self.num_kv_heads, self.num_heads // self.num_kv_heads,
                   self.head_dim)

        return q, k

    @torch.compile(dynamic=True)
    def topr_filter(self, q: torch.Tensor) -> torch.Tensor:
        i1 = torch.topk(q.abs().sum(dim=2, keepdim=True), self.topr,
                        dim=-1).indices
        q_mask = torch.zeros_like(q)
        q_mask.scatter_(-1, i1.expand_as(q[..., :self.topr]), 1)
        return q * q_mask

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if metadata.num_generations == 0:
            return None, None

        q, k = self.preprocess_for_gen(q, k, metadata)

        if self.topr < self.head_dim:
            q = self.topr_filter(q)

        kt_cache_tensor = metadata.kv_cache_manager.get_kt_buffers(
            self.layer_idx)

        # Update KT cache with new key values
        triton_rocket_update_kt_cache_gen(
            k,
            kt_cache_tensor,
            metadata.kt_cache_block_offsets[metadata.num_contexts:],
            metadata.kv_lens_cuda_runtime[metadata.num_contexts:],
            metadata.page_size,
            metadata.kt_tokens_per_block,
            metadata.kv_cache_manager.max_kt_blocks_per_seq,
            self.num_kv_heads,
            self.head_dim,
        )

        # Perform BMM with updated cache
        scores = triton_rocket_paged_kt_cache_bmm(
            q,
            kt_cache_tensor,
            metadata.kt_cache_block_offsets[metadata.num_contexts:],
            metadata.kv_lens_cuda_runtime[metadata.num_contexts:],
            metadata.cum_kt_lens_cuda,
            metadata.page_size,
            metadata.kt_tokens_per_block,
            metadata.kv_cache_manager.max_kt_blocks_per_seq,
            metadata.total_kt_tokens,
        )

        scores = triton_softmax(scores, metadata.cum_kt_lens_cuda,
                                metadata.num_generations)

        # Mean over num_heads_per_kv for each batch separately
        scores = triton_rocket_reduce_scores(
            scores,
            metadata.cum_kt_lens_cuda,
            metadata.num_generations,
            self.num_kv_heads,
            self.num_heads // self.num_kv_heads,
        )

        sparse_attn_offsets = metadata.sparse_offsets_gen_cuda[:metadata.
                                                               num_generations +
                                                               1]

        selected_indices = triton_topk(scores, metadata.cum_kt_lens_cuda,
                                       sparse_attn_offsets,
                                       metadata.total_sparse_gen_indices,
                                       metadata.topk)

        return selected_indices, sparse_attn_offsets


class RocketVanillaAttentionMetadata(VanillaAttentionMetadata):

    def __post_init__(self):
        super().__post_init__()
        if self.sparse_attention_config is None:
            raise ValueError("Sparse attention config is not set")
        self.prompt_budget = self.sparse_attention_config.prompt_budget
        self.kt_cache_block_offsets = torch.empty(
            [
                self.max_num_sequences,
                self.kv_cache_manager.max_kt_blocks_per_seq
            ],
            dtype=torch.int32,
            device='cuda',
        )
        self.host_kt_cache_block_offsets = torch.zeros_like(
            self.kt_cache_block_offsets,
            device='cpu',
            pin_memory=True,
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
                    self.kv_cache_params.num_cached_tokens_per_seq[
                        i] += self.prompt_budget - self.prompt_lens[i]

        if self.kv_cache_manager is not None:
            # for kt cache
            self.kv_cache_manager.copy_kt_block_offsets(
                self.request_ids, self.host_kt_cache_block_offsets)
            self.kt_cache_block_offsets[:self.num_seqs].copy_(
                self.host_kt_cache_block_offsets[:self.num_seqs],
                non_blocking=True)


class RocketVanillaAttention(VanillaAttention):
    """
    RocketKV sparse attention implementation.
        - Context phase: only support sparse kv cache
        - Generation phase: only support sparse attention computation
    """

    Metadata = RocketVanillaAttentionMetadata

    def __init__(
            self,
            layer_idx: int,
            num_heads: int,
            head_dim: int,
            num_kv_heads: Optional[int] = None,
            quant_config: Optional[QuantConfig] = None,
            q_scaling: Optional[float] = None,
            sparse_attention_config: Optional["SparseAttentionConfig"] = None,
            **kwargs):
        super().__init__(layer_idx,
                         num_heads,
                         head_dim,
                         sparse_attention_config=sparse_attention_config,
                         num_kv_heads=num_kv_heads,
                         quant_config=quant_config,
                         q_scaling=q_scaling,
                         **kwargs)
        self.topr = sparse_attention_config.topr
        self.topk = sparse_attention_config.topk
        self.prompt_budget = sparse_attention_config.prompt_budget
        self.window_size = sparse_attention_config.window_size
        self.kernel_size = sparse_attention_config.kernel_size
        self.page_size = sparse_attention_config.page_size
        assert sparse_attention_config.kt_cache_dtype == 'bfloat16', "Only bfloat16 kt cache is supported for Vanilla RocketKV"

    def _single_request_sparse_kv_predict(
            self, q: Optional[Tensor], k: Optional[Tensor], v: Optional[Tensor],
            metadata: RocketVanillaAttentionMetadata, past_seen_token: int,
            sample_idx: int, **kwargs) -> tuple[Optional[torch.Tensor], int]:
        """
        Predict KV indices for writing new key/value pairs.
        For RocketKV:
            - Context phase: return SnapKV sparse kv indices
            - Generation phase: return None
        """
        if k is None or v is None:
            return None, 0

        # Generation phase: do not support sparse kv cache
        if past_seen_token > 0:
            return None, k.size(1)

        # Context phase: predict SnapKV sparse kv indices
        sparse_kv_indices = self._get_snapkv_indices(q, k, sample_idx)

        # Gather the key states using the sparse kv indices
        if sparse_kv_indices is not None:
            k_snap = triton_index_gather(k, sparse_kv_indices)
        else:
            k_snap = k

        # Update KT cache
        kt_cache_tensor = metadata.kv_cache_manager.get_kt_buffers(
            self.layer_idx)
        target_seq_len = past_seen_token + k_snap.size(1)
        kt_cache_position = torch.arange(past_seen_token // self.page_size,
                                         math.ceil(target_seq_len /
                                                   self.page_size),
                                         device=q.device)
        self._single_request_update_kt_cache(
            k_snap,
            kt_cache_tensor,
            metadata.kt_cache_block_offsets[sample_idx],
            target_seq_len,
            kt_cache_position,
            update=False)
        return sparse_kv_indices, k_snap.size(1)

    def _single_request_sparse_attn_predict(
            self, q: Tensor, k: Optional[Tensor], v: Optional[Tensor],
            kv_cache_tensor: Tensor, metadata: RocketVanillaAttentionMetadata,
            past_seen_token: int, sample_idx: int,
            **kwargs) -> tuple[Optional[torch.Tensor], int]:
        """
        Predict KV cache indices for sparse attention computation.
        For RocketKV:
            - Context phase: returns None (use full attention)
            - Generation phase: return RocketKV sparse indices for sparse attention computation
        """
        if k is None or v is None:
            return None, 0

        # Context phase: use full attention
        if past_seen_token == 0:
            return None, k.size(1)

        # Get RocketKV sparse indices
        sparse_indices = self._rocketkv_selection(q, k, metadata,
                                                  past_seen_token, sample_idx)
        return sparse_indices, sparse_indices.size(1)

    def _get_snapkv_indices(self, q: Tensor, k: Tensor,
                            sample_idx: int) -> Tensor:
        """Get SnapKV sparse kv indices from the input sequence for context phase."""
        bsz = 1
        seq_len = k.size(1)

        # If the sequence length is less than the prompt budget, do not enable sparse kv cache
        if seq_len <= self.prompt_budget:
            return None

        # Use last window_size tokens as observation
        # (1, num_kv_heads, window_size, head_dim)
        q_obs = q[:, :, -self.window_size:]
        # (1, num_kv_heads, seq_len, head_dim)
        k_pre = repeat_kv(k.transpose(1, 2), self.num_key_value_groups)

        dist = (torch.arange(0, self.window_size, device=q.device)[:, None] -
                torch.arange(0, seq_len, device=q.device)[None, :] + seq_len -
                self.window_size)
        attention_mask = (dist >= 0)

        score = torch.matmul(q_obs, k_pre.transpose(-1, -2)) / math.sqrt(
            self.head_dim)

        score = torch.masked_fill(
            score,
            attention_mask.view(1, 1, self.window_size, seq_len) == False,
            torch.scalar_tensor(float("-inf"),
                                device=score.device,
                                dtype=score.dtype))

        score = torch.nn.functional.softmax(score, dim=-1)

        score = torch.masked_fill(
            score,
            attention_mask.view(1, 1, self.window_size, seq_len) == False,
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype))

        score = score[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)

        score = score.view(bsz, self.num_kv_heads, self.num_key_value_groups,
                           -1).sum(dim=2)
        score = torch.nn.functional.max_pool1d(score,
                                               kernel_size=self.kernel_size,
                                               padding=self.kernel_size // 2,
                                               stride=1)

        # Select top important tokens from prefix
        prefix_len = seq_len - self.window_size
        selected_prefix_indices = score.topk(self.prompt_budget -
                                             self.window_size,
                                             dim=-1).indices.sort().values

        # Combine selected prefix indices with window indices
        window_indices = torch.arange(
            prefix_len, seq_len,
            device=k.device).unsqueeze(0).unsqueeze(0).expand(
                bsz, self.num_kv_heads, -1)
        selected_indices = torch.cat([selected_prefix_indices, window_indices],
                                     dim=-1)

        return selected_indices.transpose(1, 2)

    def _single_request_update_kt_cache(self,
                                        k,
                                        kt_cache_tensor,
                                        kt_cache_block_offsets,
                                        seq_len,
                                        cache_position,
                                        update=True):
        """Update KT cache for RocketKV algorithm."""
        # (1, num_pages_per_block, num_kv_heads, 2*head_dim)
        k_out = kt_cache_tensor[kt_cache_block_offsets[0], :, :, :].unsqueeze(0)

        # k: (1, seq_len, num_kv_heads, head_dim)
        if k is not None:
            padding_len = self.page_size - (
                (k.size(1) - 1) % self.page_size + 1)
            padding_tensor = torch.full(
                (1, padding_len, self.num_kv_heads, self.head_dim),
                float('inf'),
                device=k.device,
                dtype=k.dtype)
            # (1, seq_len+padding_len, num_kv_heads, head_dim)
            k_min = torch.cat([k, padding_tensor], dim=1)
            k_min = k_min.reshape(1, -1, self.page_size, self.num_kv_heads,
                                  self.head_dim).amin(dim=2)
            k_max = torch.cat([k, -padding_tensor], dim=1)
            k_max = k_max.reshape(1, -1, self.page_size, self.num_kv_heads,
                                  self.head_dim).amax(dim=2)
            if update and (seq_len - 1) % self.page_size > 0:  # gen phase
                k_min = torch.min(k_min,
                                  k_out[:, cache_position, :, :self.head_dim])
                k_max = torch.max(k_max, k_out[:, cache_position, :,
                                               self.head_dim:])
            k_value = torch.cat([k_min, k_max], dim=-1)
            access_type = self._access_type[k_value.dtype.itemsize]
            k_out.view(dtype=access_type).index_copy_(
                1, cache_position, k_value.view(dtype=access_type))

        return k_out[:, :math.ceil(seq_len / self.page_size), :, :]

    def _rocketkv_selection(self, q: Tensor, k: Tensor,
                            metadata: RocketVanillaAttentionMetadata,
                            past_seen_token: int, sample_idx: int) -> Tensor:
        """Implement RocketKV's two-stage selection process for generation phase."""
        bsz = 1
        q_len = q.size(2)

        # Helper functions
        def _gather(t: Tensor, dim: int, i: Tensor) -> Tensor:
            dim += (dim < 0) * t.ndim
            return t.gather(
                dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1:]))

        @torch.compile(disable=not torch.cuda.is_available())
        def _scaled_softmax(x: Tensor, divscale: Tensor | float,
                            dim: int) -> Tensor:
            return torch.softmax(x / divscale, dim=dim)

        # Get KT cache for key-token matching
        kt_cache_tensor = metadata.kv_cache_manager.get_kt_buffers(
            self.layer_idx)
        target_seq_len = past_seen_token + 1  # +1 for current token

        # Update KT cache
        kt_cache_position = torch.arange(past_seen_token // self.page_size,
                                         math.ceil(target_seq_len /
                                                   self.page_size),
                                         device=q.device)
        kt_states = self._single_request_update_kt_cache(
            k, kt_cache_tensor, metadata.kt_cache_block_offsets[sample_idx],
            target_seq_len, kt_cache_position)

        # Reshape query for multi-head processing
        qi = q.view(bsz, self.num_kv_heads, self.num_heads // self.num_kv_heads,
                    q_len, self.head_dim)
        qi_abs = torch.abs(qi)

        # Top-r selection on query features
        i1 = torch.topk(qi_abs.mean(dim=2, keepdim=True), self.topr,
                        dim=-1).indices
        qi_hat = _gather(qi, -1, i1)

        # Generate signed indices for key-token matching
        i1_sign = torch.where(
            qi_hat.sum(dim=2, keepdim=True) > 0, i1 + self.head_dim,
            i1).transpose(-1, -2)

        # Gather key tokens and compute attention scores
        kt_hat = _gather(
            kt_states.permute(0, 2, 3, 1).unsqueeze(2), -2, i1_sign)
        qk_hat = qi_hat @ kt_hat
        qk_hat = qk_hat.repeat_interleave(self.page_size,
                                          dim=-1)[:, :, :, :, :target_seq_len]
        scale = torch.sqrt(self.head_dim *
                           torch.abs(qi_hat).sum(dim=-1, keepdim=True) /
                           qi_abs.sum(dim=-1, keepdim=True))

        # (1, num_kv_heads, num_heads, target_seq_len)
        s_hat = _scaled_softmax(qk_hat, scale, dim=-1)

        topk = min(self.topk, target_seq_len)
        i2 = torch.topk(s_hat.mean(dim=2, keepdim=True), topk, dim=-1).indices

        iKV = i2[:, :, 0, 0, :].transpose(1, 2)  # (1, topk, num_kv_heads)

        return iKV


class RocketKVCacheManager(KVCacheManager):

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        layer_mask: Optional[List[bool]] = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfig] = None,
        max_beam_width: int = 1,
        sparse_attn_config: Optional["SparseAttentionConfig"] = None,
        **kwargs,
    ) -> None:

        assert not kv_cache_config.enable_block_reuse, "RocketKV cache requires block reuse to be disabled in KV cache config"
        self.kt_tokens_per_block = next_power_of_2(
            math.ceil(tokens_per_block / sparse_attn_config.page_size))
        self.kt_cache_dtype = torch.bfloat16 if sparse_attn_config.kt_cache_dtype == 'bfloat16' else torch.float8_e5m2

        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
            spec_config=spec_config,
            layer_mask=layer_mask,
            max_num_tokens=max_num_tokens,
            model_config=model_config,
            max_beam_width=max_beam_width,
            **kwargs,
        )
        self.page_size = sparse_attn_config.page_size
        self.prompt_budget = sparse_attn_config.prompt_budget
        self.max_batch_size = max_batch_size

        # Per layer KT cache pool
        # Use the same number of blocks as the paged kv cache. In this way, the scheduler can use the same number of
        # blocks to schedule requests.
        # Use kt_tokens_per_block to make sure the KT cache is large enough to hold the kt tokens,
        # since kt_tokens_per_block * num_blocks * page_size >= tokens_per_block * num_blocks.
        self.num_blocks = self.blocks_in_primary_pool
        self.kt_cache_pool_per_layer = [
            torch.empty((self.num_blocks, self.kt_tokens_per_block,
                         num_kv_heads, head_dim * 2),
                        device="cuda",
                        dtype=self.kt_cache_dtype)
            for _ in range(self.num_local_layers)
        ]
        self.max_kt_blocks_per_seq = self.num_blocks

        # Block manager to manage the KT cache blocks for each request. Different layers share the
        # same block ids.
        self.kt_cache_manager = BlockManager(self.num_blocks,
                                             self.kt_tokens_per_block)

    def add_dummy_requests(
        self,
        request_ids: List[int],
        token_nums: Optional[List[int]] = None,
        is_gen: bool = False,
        prepare_resource: bool = True,
        max_num_draft_tokens: int = 0,
        use_mrope: bool = False,
        max_beam_width: int = 1,
        num_extra_decoding_steps: int = 0,
    ):
        requests = super().add_dummy_requests(
            request_ids=request_ids,
            token_nums=token_nums,
            is_gen=is_gen,
            prepare_resource=prepare_resource,
            max_num_draft_tokens=max_num_draft_tokens,
            use_mrope=use_mrope,
            max_beam_width=max_beam_width,
            num_extra_decoding_steps=num_extra_decoding_steps,
        )
        if prepare_resource:
            for req in requests:
                request_id = req.py_request_id
                kt_token_num = math.ceil(req.max_beam_num_tokens /
                                         self.page_size)
                self.kt_cache_manager.add_tokens(request_id, kt_token_num)
        return requests

    def get_kt_buffers(self, layer_idx: int):
        return self.kt_cache_pool_per_layer[layer_idx]

    def copy_kt_block_offsets(self, request_ids: List[int],
                              block_offsets: torch.Tensor) -> torch.Tensor:
        self.kt_cache_manager.copy_block_offsets(request_ids, block_offsets)

    def prepare_resources(self, scheduled_batch):
        super().prepare_resources(scheduled_batch)
        for req in scheduled_batch.context_requests:
            request_id = req.py_request_id
            num_tokens = req.prompt_len
            kt_token_num = math.ceil(num_tokens / self.page_size)
            self.kt_cache_manager.add_tokens(request_id, kt_token_num)

        for req in scheduled_batch.generation_requests:
            request_id = req.py_request_id
            num_tokens = req.max_beam_num_tokens + 1
            if num_tokens % self.page_size == 1:
                self.kt_cache_manager.add_tokens(request_id, 1)

    def update_resources(self,
                         scheduled_batch,
                         attn_metadata: AttentionMetadata = None,
                         kv_cache_dtype_byte_size: float = None):
        for request in scheduled_batch.context_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                seq_len = request.get_num_tokens(0)
                rewind_len = max(seq_len - 1 - self.prompt_budget, 0)
                self.rewind_kv_cache(request, rewind_len)
                # get the rewind length for kt cache
                num_tokens = request.max_beam_num_tokens
                updated_kt_token_num = num_tokens - rewind_len
                rewind_len = (math.ceil(num_tokens / self.page_size) -
                              math.ceil(updated_kt_token_num / self.page_size))
                self.kt_cache_manager.rewind_cache(request, rewind_len)

    def free_resources(self, request):
        super().free_resources(request)
        self.kt_cache_manager.free_resources(request)

    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfig, mapping: Mapping,
                                 **kwargs):
        # get kv cache dtype bytes
        mem_per_token = 2
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache(
        ):
            mem_per_token = 1

        # get num key value heads
        config = model_config.pretrained_config
        num_key_value_heads = getattr(config, 'num_key_value_heads',
                                      config.num_attention_heads)
        if isinstance(num_key_value_heads, Iterable):
            num_key_value_heads = sum(num_key_value_heads) / len(
                num_key_value_heads)

        # get head dim
        tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
        head_dim = getattr(config, "head_dim", None)
        if not isinstance(head_dim, int):
            head_dim = config.hidden_size // config.num_attention_heads
        head_dim = head_dim * num_key_value_heads // tp_size

        # provide at least 1 layer to prevent division by zero cache size
        num_attention_layers = max(
            len(mapping.pp_layers(model_config.get_num_attention_layers())), 1)
        mem_per_token *= num_attention_layers * head_dim

        # K and V
        # 2 for K and V, 2 * kt_tokens_per_block / tokens_per_block for KT cache
        tokens_per_block = kwargs['tokens_per_block']
        sparse_attn_config = model_config.sparse_attention_config
        kt_tokens_per_block = next_power_of_2(
            math.ceil(tokens_per_block / sparse_attn_config.page_size))
        kt_factor = 2
        if sparse_attn_config.kt_cache_dtype == "float8_e5m2":
            kt_factor = 1
        kv_factor = 2 + kt_factor * kt_tokens_per_block / tokens_per_block
        mem_per_token *= kv_factor
        return mem_per_token

    def get_cache_bytes_per_token(self):
        # 2 for K and V, 2 * kt_tokens_per_block / tokens_per_block for KT cache
        kt_factor = 2
        if self.kt_cache_dtype == torch.float8_e5m2:
            kt_factor = 1
        kv_factor = self.kv_factor + kt_factor * self.kt_tokens_per_block / self.tokens_per_block
        cache_size_per_token = math.ceil(
            kv_factor * sum(self.num_kv_heads_per_layer) * self.head_dim)

        if self.dtype not in (DataType.FP8, DataType.HALF, DataType.BF16,
                              DataType.FLOAT, DataType.NVFP4):
            raise ValueError(f'Cannot support {self.dtype} KV cache.')

        cache_size_bytes_per_token = get_size_in_bytes(cache_size_per_token,
                                                       self.dtype)
        if self.dtype == DataType.NVFP4:
            cache_size_bytes_per_token += self.calculate_scaling_factor_size_bytes(
                cache_size_per_token,
                quant_vector_size=16,
                scaling_factor_dtype=DataType.FP8)
        return cache_size_bytes_per_token
