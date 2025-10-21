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

from .kernel import triton_index_gather, triton_update_kt_cache

ModelConfig = tensorrt_llm.bindings.ModelConfig


class RocketTrtllmAttentionMetadata(TrtllmAttentionMetadata):

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


@torch.compile(dynamic=True)
def convert_token_to_page_sparse_indices(
        sparse_attn_indices: torch.Tensor, sparse_attn_offsets: torch.Tensor,
        metadata: 'TrtllmAttentionMetadata'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert token-based sparse attention indices to page-based sparse attention indices.

    Args:
        sparse_attn_indices: Token-based indices with shape [num_tokens, num_kv_heads]
        sparse_attn_offsets: Offsets with shape [batch_size+1] indicating token boundaries for each batch
        metadata: Attention metadata containing tokens_per_block (page_size)

    Returns:
        Tuple of (page_indices, page_offsets):
        - page_indices: Page-based indices with shape [num_pages, num_kv_heads]
        - page_offsets: Updated offsets with shape [batch_size+1] indicating page boundaries for each batch

    Example:
        If sparse_attn_indices first dimension is [1, 30, 67] and page_size=32,
        the result will be [0, 2] (token 1 -> page 0, token 30 -> page 0, token 67 -> page 2)
    """
    page_size = metadata.tokens_per_block
    batch_size = sparse_attn_offsets.size(0) - 1
    num_kv_heads = sparse_attn_indices.size(1)

    # Convert token indices to page indices
    page_indices = sparse_attn_indices // page_size

    # Process each batch and each kv_head separately to remove duplicates
    new_page_indices_list = []
    new_offsets = torch.zeros_like(sparse_attn_offsets)

    current_offset = 0
    for batch_idx in range(batch_size):
        start_idx = sparse_attn_offsets[batch_idx]
        end_idx = sparse_attn_offsets[batch_idx + 1]

        if start_idx >= end_idx:
            # Empty batch
            new_offsets[batch_idx + 1] = current_offset
            continue

        batch_page_indices = page_indices[
            start_idx:end_idx]  # [num_tokens_in_batch, num_kv_heads]

        # For each kv_head, remove duplicates while preserving order
        batch_unique_pages = []
        for head_idx in range(num_kv_heads):
            head_pages = batch_page_indices[:, head_idx]
            unique_pages = torch.unique(head_pages, sorted=False)
            batch_unique_pages.append(unique_pages)

        # Find the maximum length among all heads for this batch
        max_len = max(pages.size(0) for pages in batch_unique_pages)

        if max_len > 0:
            batch_result = torch.full((max_len, num_kv_heads),
                                      fill_value=-1,
                                      dtype=page_indices.dtype,
                                      device=page_indices.device)

            for head_idx in range(num_kv_heads):
                unique_pages = batch_unique_pages[head_idx]
                batch_result[:unique_pages.size(0), head_idx] = unique_pages

            new_page_indices_list.append(batch_result)
            current_offset += max_len

        new_offsets[batch_idx + 1] = current_offset

    new_page_indices = torch.cat(new_page_indices_list, dim=0)

    return new_page_indices, new_offsets


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

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Predict sparse attention indices.
        For RocketKV:
        - Generation phase: predict RocketKV sparse attention indices

        Returns:
            - sparse_attn_indices: [total_selected_indices, num_kv_heads]
            - sparse_attn_offsets: [batch_size + 1] with cumulative indices count
        """
        if k is None:
            q, k, _ = q.split([
                self.num_heads * self.head_dim, self.num_kv_heads *
                self.head_dim, self.num_kv_heads * self.head_dim
            ],
                              dim=-1)

        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        seq_lens = metadata.seq_lens
        seq_lens_kv = metadata.seq_lens_kv if metadata.seq_lens_kv is not None else seq_lens
        past_seen_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq

        sparse_attn_indices = []
        sparse_attn_offsets = [0]

        q_offset = 0
        k_offset = 0

        for i in range(num_contexts + num_generations):
            seq_len = seq_lens[i].item()
            seq_len_kv = seq_lens_kv[i].item()

            if seq_len <= 0 or seq_len_kv <= 0:
                assert False, "Invalid sequence length"

            single_q = q[q_offset:q_offset + seq_len]
            single_k = k[k_offset:k_offset + seq_len_kv]

            single_q = single_q.view(1, seq_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
            single_k = single_k.view(1, seq_len_kv, self.num_kv_heads,
                                     self.head_dim)

            past_seen_token = past_seen_tokens[i]
            # Generation phase: RocketKV sparse attention indices
            if i >= num_contexts:
                _sparse_attn_indices = self._rocketkv_selection(
                    single_q, single_k, past_seen_token, metadata, i)
                if _sparse_attn_indices is not None:
                    sparse_attn_indices.append(
                        _sparse_attn_indices.squeeze(0))  # [topk, num_kv_heads]
                    sparse_attn_offsets.append(sparse_attn_offsets[-1] +
                                               _sparse_attn_indices.size(1))
                else:
                    sparse_attn_offsets.append(sparse_attn_offsets[-1])

            q_offset += seq_len
            k_offset += seq_len_kv

        if len(sparse_attn_indices) == 0:
            sparse_attn_indices, sparse_attn_offsets = None, None
        else:
            sparse_attn_indices = torch.cat(sparse_attn_indices,
                                            dim=0).to(torch.int32)
            sparse_attn_offsets = torch.tensor(sparse_attn_offsets,
                                               dtype=torch.int32).to(q.device)
            sparse_attn_indices, sparse_attn_offsets = convert_token_to_page_sparse_indices(
                sparse_attn_indices, sparse_attn_offsets, metadata)
            sparse_attn_indices = sparse_attn_indices.transpose(0,
                                                                1).contiguous()
        return sparse_attn_indices, sparse_attn_offsets

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Predict sparse kv indices.

        For RocketKV:
        - Context phase: predict RocketKV sparse kv indices

        Returns:
            - flattened_indices: [total_selected_indices, num_kv_heads]
            - batch_offsets: [batch_size + 1] with cumulative indices count
        """
        if k is None:
            q, k, _ = q.split([
                self.num_heads * self.head_dim, self.num_kv_heads *
                self.head_dim, self.num_kv_heads * self.head_dim
            ],
                              dim=-1)

        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        seq_lens = metadata.seq_lens
        seq_lens_kv = metadata.seq_lens_kv if metadata.seq_lens_kv is not None else seq_lens
        past_seen_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq

        sparse_kv_indices = []
        sparse_kv_offsets = [0]

        q_offset = 0
        k_offset = 0

        for i in range(num_contexts + num_generations):
            seq_len = seq_lens[i].item()
            seq_len_kv = seq_lens_kv[i].item()

            if seq_len <= 0 or seq_len_kv <= 0:
                assert False, "Invalid sequence length"

            single_q = q[q_offset:q_offset + seq_len]
            single_k = k[k_offset:k_offset + seq_len_kv]

            single_q = single_q.view(1, seq_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
            single_k = single_k.view(1, seq_len_kv, self.num_kv_heads,
                                     self.head_dim)

            past_seen_token = past_seen_tokens[i]
            if i < num_contexts:
                # Context phase: SnapKV sparse kv indices
                _sparse_kv_indices = self._get_snapkv_indices(
                    single_q, single_k, past_seen_token, metadata, i)
                if _sparse_kv_indices is not None:
                    sparse_kv_indices.append(
                        _sparse_kv_indices.squeeze(0))  # [budget, num_kv_heads]
                    sparse_kv_offsets.append(sparse_kv_offsets[-1] +
                                             _sparse_kv_indices.size(1))
                else:
                    sparse_kv_offsets.append(sparse_kv_offsets[-1])

            q_offset += seq_len
            k_offset += seq_len_kv

        if len(sparse_kv_indices) == 0:
            sparse_kv_indices, sparse_kv_offsets = None, None
        else:
            sparse_kv_indices = torch.cat(sparse_kv_indices,
                                          dim=0).to(torch.int32)
            sparse_kv_indices = sparse_kv_indices.transpose(0, 1).contiguous()
            sparse_kv_offsets = torch.tensor(sparse_kv_offsets,
                                             dtype=torch.int32).to(q.device)
        return sparse_kv_indices, sparse_kv_offsets

    def _get_snapkv_indices(self, q: Tensor, k: Tensor, past_seen_token: int,
                            metadata: RocketTrtllmAttentionMetadata,
                            sample_idx: int) -> Optional[Tensor]:
        """
        Get SnapKV sparse kv indices from the input sequence for context phase.
        The shape of output is (1, prompt_budget, num_kv_heads)
        """
        bsz = 1
        seq_len = k.size(1)  # k shape: (1, seq_len, num_kv_heads, head_dim)

        # If the sequence length is less than the prompt budget, do not enable sparse kv cache
        if seq_len <= self.prompt_budget:
            return None

        # Use last window_size tokens as observation
        # (1, num_heads, window_size, head_dim)
        q_obs = q[:, :, -self.window_size:]
        # (1, num_kv_heads, seq_len, head_dim)
        k_pre = repeat_kv(k.transpose(1, 2),
                          self.num_heads // self.num_kv_heads)

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

        score = score.view(bsz, self.num_kv_heads,
                           self.num_heads // self.num_kv_heads, -1).sum(dim=2)
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
                                     dim=-1).transpose(1, 2)

        k = k.reshape(1, -1, self.num_kv_heads, self.head_dim)
        k_snap = triton_index_gather(k, selected_indices)
        # Update KT cache
        kt_cache_tensor = metadata.kv_cache_manager.get_kt_buffers(
            self.layer_idx)
        k_snap_len = torch.clamp(
            metadata.kv_lens_cuda_runtime[sample_idx:sample_idx + 1],
            max=self.prompt_budget).int()
        triton_update_kt_cache(
            k_snap.squeeze(0).contiguous(),
            kt_cache_tensor,
            metadata.kt_cache_block_offsets[sample_idx:sample_idx + 1],
            k_snap_len,
            self.page_size,
            metadata.kt_tokens_per_block,
            metadata.kv_cache_manager.max_kt_blocks_per_seq,
            update=False)

        return selected_indices

    def _rocketkv_selection(self, q: Tensor, k: Tensor, past_seen_token: int,
                            metadata: RocketTrtllmAttentionMetadata,
                            sample_idx: int) -> Tensor:
        """
        Implement RocketKV's two-stage selection process for generation phase.
        The shape of output is (1, topk, num_kv_heads)
        """
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
        kt_states = triton_update_kt_cache(
            k.squeeze(0).contiguous(), kt_cache_tensor,
            metadata.kt_cache_block_offsets[sample_idx:sample_idx + 1],
            metadata.kv_lens_cuda_runtime[sample_idx:sample_idx + 1],
            self.page_size, metadata.kt_tokens_per_block,
            metadata.kv_cache_manager.max_kt_blocks_per_seq)
        kt_states = kt_states.unsqueeze(0).permute(0, 2, 3, 1)

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
        kt_hat = _gather(kt_states.unsqueeze(2), -2, i1_sign)
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
            self.host_kt_cache_block_offsets = self.kv_cache_manager.get_kt_block_offsets(
                self.request_ids)
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
                        dtype=torch.bfloat16)
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
    ):
        requests = super().add_dummy_requests(
            request_ids=request_ids,
            token_nums=token_nums,
            is_gen=is_gen,
            prepare_resource=prepare_resource,
            max_num_draft_tokens=max_num_draft_tokens,
            use_mrope=use_mrope,
            max_beam_width=max_beam_width,
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
        for req in scheduled_batch.all_requests():
            request_id = req.py_request_id
            kt_token_num = math.ceil(req.max_beam_num_tokens / self.page_size)
            self.kt_cache_manager.add_tokens(request_id, kt_token_num)

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
        kv_factor = 2 + 2 * kt_tokens_per_block / tokens_per_block
        mem_per_token *= kv_factor
        return mem_per_token

    def get_cache_bytes_per_token(self):
        # 2 for K and V, 2 * kt_tokens_per_block / tokens_per_block for KT cache
        kv_factor = self.kv_factor + 2 * self.kt_tokens_per_block / self.tokens_per_block
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
