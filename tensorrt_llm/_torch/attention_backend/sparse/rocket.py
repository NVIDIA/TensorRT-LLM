import math
from collections import deque
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.trtllm import (
    TrtllmAttention, TrtllmAttentionMetadata)
from tensorrt_llm._torch.attention_backend.vanilla import (
    VanillaAttention, VanillaAttentionMetadata, repeat_kv)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes, next_power_of_two
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import ExecutorConfig, KvCacheConfig
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
            self.host_kt_cache_block_offsets = self.kv_cache_manager.get_kt_block_offsets(
                self.request_ids)
            self.kt_cache_block_offsets[:self.num_seqs].copy_(
                self.host_kt_cache_block_offsets, non_blocking=True)


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

    def sparse_attention_predict(
        self, q: torch.Tensor, k: torch.Tensor,
        metadata: RocketTrtllmAttentionMetadata
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Predict sparse KV indices and sparse attention indices for the input sequence.

        For RocketKV:
        - Context phase: predict SnapKV sparse kv indices
        - Generation phase: predict RocketKV sparse attention indices

        Returns:
            Tuple of (flattened_indices, batch_offsets)
            - flattened_indices: [total_selected_indices, num_kv_heads]
            - batch_offsets: [batch_size + 1] with cumulative indices count
        """
        q, k, _ = q.split([
            self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim
        ],
                          dim=-1)

        if k is None or metadata is None:
            return None, None

        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        seq_lens = metadata.seq_lens
        seq_lens_kv = metadata.seq_lens_kv if metadata.seq_lens_kv is not None else seq_lens
        past_seen_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq

        sparse_kv_indices = []
        sparse_attn_indices = []
        sparse_kv_offsets = [0]
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
            else:
                # Generation phase: RocketKV sparse attention indices
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

        if len(sparse_kv_indices) == 0:
            sparse_kv_indices, sparse_kv_offsets = None, None
        else:
            sparse_kv_indices = torch.cat(sparse_kv_indices,
                                          dim=0).to(torch.int32)
            sparse_kv_offsets = torch.tensor(sparse_kv_offsets,
                                             dtype=torch.int32).to(q.device)
        if len(sparse_attn_indices) == 0:
            sparse_attn_indices, sparse_attn_offsets = None, None
        else:
            sparse_attn_indices = torch.cat(sparse_attn_indices,
                                            dim=0).to(torch.int32)
            sparse_attn_offsets = torch.tensor(sparse_attn_offsets,
                                               dtype=torch.int32).to(q.device)

        return sparse_kv_indices, sparse_kv_offsets, sparse_attn_indices, sparse_attn_offsets

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
            cache_idx: int, **kwargs) -> Optional[Tensor]:
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
        sparse_kv_indices = self._get_snapkv_indices(q, k)

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
        self._single_request_update_kt_cache(k_snap,
                                             kt_cache_tensor,
                                             target_seq_len,
                                             kt_cache_position,
                                             update=False)
        return sparse_kv_indices, k_snap.size(1)

    def _single_request_sparse_attn_predict(
            self, q: Tensor, k: Optional[Tensor], v: Optional[Tensor],
            kv_cache_tensor: Tensor, metadata: RocketVanillaAttentionMetadata,
            past_seen_token: int, cache_idx: int, **kwargs) -> Optional[Tensor]:
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
                                                  past_seen_token)
        return sparse_indices, sparse_indices.size(1)

    def _get_snapkv_indices(self, q: Tensor, k: Tensor) -> Tensor:
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
                                        seq_len,
                                        cache_position,
                                        update=True):
        """Update KT cache for RocketKV algorithm."""
        # (1, num_pages_per_block, num_kv_heads, 2*head_dim)
        k_out = kt_cache_tensor[0, :, :, :].unsqueeze(0)

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
            if update:
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
                            past_seen_token: int) -> Tensor:
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
            k, kt_cache_tensor, target_seq_len, kt_cache_position)

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
        self.kt_tokens_per_block = next_power_of_two(
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
        self.base_kt_block_offsets = torch.arange(self.num_blocks,
                                                  device="cpu",
                                                  dtype=torch.int32)
        self.max_kt_blocks_per_seq = self.num_blocks

        # Block manager to manage the KT cache blocks for each request. Different layers share the
        # same block ids.
        self.paged_kt_block_ids = dict()
        self.free_blocks = deque(range(self.num_blocks))

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
                self.add_kt_tokens(request_id, kt_token_num)
        return requests

    def get_kt_buffers(self, layer_idx: int):
        return self.kt_cache_pool_per_layer[layer_idx]

    def get_kt_block_offsets(self, request_ids: List[int]) -> torch.Tensor:
        kt_block_offsets = torch.empty(
            [len(request_ids), self.max_kt_blocks_per_seq],
            device="cpu",
            dtype=torch.int32)
        for i in range(len(request_ids)):
            block_ids = self.paged_kt_block_ids[request_ids[i]]
            block_num = len(block_ids)
            kt_block_offsets[i, 0:block_num].copy_(
                self.base_kt_block_offsets[block_ids])
        return kt_block_offsets

    def prepare_resources(self, scheduled_batch):
        super().prepare_resources(scheduled_batch)
        for req in scheduled_batch.all_requests():
            request_id = req.py_request_id
            kt_token_num = math.ceil(req.max_beam_num_tokens / self.page_size)
            self.add_kt_tokens(request_id, kt_token_num)

    def update_resources(self, scheduled_batch):
        for request in scheduled_batch.context_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                seq_len = request.get_num_tokens(0)
                rewind_len = max(seq_len - 1 - self.prompt_budget, 0)
                self.rewind_kv_cache(request, rewind_len)
                self.rewind_kt_cache(request, rewind_len)

    def rewind_kt_cache(self, request, rewind_len):
        request_id = request.py_request_id
        num_tokens = request.max_beam_num_tokens
        updated_kt_token_num = math.ceil(num_tokens -
                                         rewind_len / self.page_size)
        page_count_needed = self.compute_page_count(updated_kt_token_num,
                                                    self.kt_tokens_per_block)
        num_rewind_pages = len(
            self.paged_kt_block_ids[request_id]) - page_count_needed
        if num_rewind_pages > 0:
            self._free_kt_pages(
                self.paged_kt_block_ids[request_id][-num_rewind_pages:])
            self.paged_kt_block_ids[request_id] = self.paged_kt_block_ids[
                request_id][:-num_rewind_pages]

    def free_resources(self, request):
        super().free_resources(request)
        request_id = request.py_request_id
        self._free_kt_pages(self.paged_kt_block_ids[request_id])
        del self.paged_kt_block_ids[request_id]

    def add_kt_tokens(self, request_id: int, kt_token_num: int):
        if kt_token_num > 0:
            page_count_needed = self.compute_page_count(
                kt_token_num, self.kt_tokens_per_block)
            if request_id not in self.paged_kt_block_ids:
                self.paged_kt_block_ids[request_id] = []
            if len(self.paged_kt_block_ids[request_id]) < page_count_needed:
                new_page = self._allocate_kt_pages(
                    page_count_needed -
                    len(self.paged_kt_block_ids[request_id]))
                self.paged_kt_block_ids[request_id].extend(new_page)

    def _allocate_kt_pages(self, page_count: int) -> list:
        assert len(self.free_blocks) >= page_count, "Not enough pages."
        pages = [self.free_blocks.popleft() for _ in range(page_count)]
        return pages

    def _free_kt_pages(self, page_list: list):
        self.free_blocks.extend(page_list)

    def compute_page_count(self, token_count: int, tokens_per_page: int) -> int:
        return (token_count + tokens_per_page) // tokens_per_page

    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfig,
                                 executor_config: ExecutorConfig,
                                 mapping: Mapping):
        sparse_attn_config = executor_config.sparse_attention_config
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
        tokens_per_block = executor_config.tokens_per_block
        kt_tokens_per_block = next_power_of_two(
            math.ceil(tokens_per_block / sparse_attn_config.page_size))
        kv_factor = 2 + 2 * kt_tokens_per_block / tokens_per_block
        mem_per_token *= kv_factor
        return mem_per_token

    def get_cache_bytes_per_token(self):
        # 2 for K and V, 2 * kt_tokens_per_block / tokens_per_block for KT cache
        kv_factor = self.kv_factor + 2 * self.kt_tokens_per_block / self.tokens_per_block
        cache_size_per_token = kv_factor * sum(
            self.num_kv_heads_per_layer) * self.head_dim

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
