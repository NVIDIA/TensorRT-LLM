import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import Tensor

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.attention_backend.vanilla import VanillaAttention, repeat_kv
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..kernels.common import (
    triton_bmm,
    triton_flatten_to_batch,
    triton_index_gather,
    triton_softmax,
    triton_topk,
)
from .kernels import (
    triton_rocket_batch_to_flatten,
    triton_rocket_paged_kt_cache_bmm,
    triton_rocket_qk_split,
    triton_rocket_reduce_scores,
    triton_rocket_update_kt_cache_ctx,
    triton_rocket_update_kt_cache_gen,
)
from .metadata import RocketTrtllmAttentionMetadata, RocketVanillaAttentionMetadata

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


class RocketTrtllmAttention(TrtllmAttention):
    Metadata = RocketTrtllmAttentionMetadata

    # Access type for different dtype sizes
    _access_type = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        q_scaling: Optional[float] = None,
        sparse_attention_config: Optional["SparseAttentionConfig"] = None,
        **kwargs,
    ):
        super().__init__(
            layer_idx,
            num_heads,
            head_dim,
            sparse_attention_config=sparse_attention_config,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            **kwargs,
        )
        self.topr = sparse_attention_config.topr
        self.topk = sparse_attention_config.topk
        self.prompt_budget = sparse_attention_config.prompt_budget
        self.window_size = sparse_attention_config.window_size
        self.kernel_size = sparse_attention_config.kernel_size
        self.page_size = sparse_attention_config.page_size

    def prepare_sparse_params(self, q, k, metadata, **kwargs):
        """Prepare SparseParams for RocketKV."""
        from ..params import SparseParams

        sparse_kv_indices, sparse_kv_offsets = self.sparse_kv_predict(q, k, metadata, **kwargs)
        sparse_attn_indices, sparse_attn_offsets = self.sparse_attn_predict(
            q, k, metadata, **kwargs
        )
        return SparseParams(
            sparse_kv_indices=sparse_kv_indices,
            sparse_kv_offsets=sparse_kv_offsets,
            sparse_attn_indices=sparse_attn_indices,
            sparse_attn_offsets=sparse_attn_offsets,
            sparse_attn_indices_block_size=(self.sparse_attention_config.get_indices_block_size()),
            sparse_mla_topk=(
                metadata.sparse_mla_topk if hasattr(metadata, "sparse_mla_topk") else 0
            ),
        )

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

            scores = triton_bmm(
                q_window,
                k_context,
                metadata.q_cu_seqlens_cuda,
                metadata.k_cu_seqlens_cuda,
                metadata.valid_batch_size,
                causal=False,
            )

            scores = triton_softmax(scores, metadata.k_cu_seqlens_cuda, metadata.valid_batch_size)

            # scores: [num_heads, window_size, total_k_tokens] -> [num_kv_heads, total_k_tokens]
            scores = scores.view(
                self.num_kv_heads, self.num_heads // self.num_kv_heads, self.window_size, -1
            ).sum(dim=(1, 2))

            # Reshape scores to handle variable length sequences with padding using Triton
            # scores: [num_kv_heads, total_k_tokens] -> [valid_batch_size, num_kv_heads, padding_size]
            scores = triton_flatten_to_batch(
                scores,
                metadata.k_cu_seqlens_cuda,
                metadata.valid_batch_size,
                metadata.max_rocket_k_ctx_len,
            )

            scores = torch.nn.functional.max_pool1d(
                scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1
            )

            # Use indexer topk prefill to select topk prefix indices
            total_tasks = metadata.valid_batch_size * self.num_kv_heads

            selected_prefix_indices = torch.empty(
                (total_tasks, self.prompt_budget - self.window_size),
                device=qkv_input.device,
                dtype=torch.int32,
            )

            scores = scores.view(total_tasks, -1)

            row_starts = metadata.k_context_start_cuda[
                : metadata.valid_batch_size
            ].repeat_interleave(self.num_kv_heads)
            row_ends = metadata.k_context_lens_cuda[: metadata.valid_batch_size].repeat_interleave(
                self.num_kv_heads
            )
            torch.ops.trtllm.indexer_topk_prefill(
                scores,
                row_starts,
                row_ends,
                selected_prefix_indices,
                self.prompt_budget - self.window_size,
            )

            # Sort selected prefix indices to keep topk indices in ascending order
            selected_prefix_indices = torch.sort(selected_prefix_indices, dim=-1).values
        else:
            selected_prefix_indices = torch.empty(
                (0, self.prompt_budget - self.window_size),
                device=qkv_input.device,
                dtype=torch.int32,
            )

        sparse_kv_offsets = metadata.sparse_offsets_ctx_cuda[: metadata.num_contexts + 1]

        # Flatten sparse indices
        sparse_kv_indices = triton_rocket_batch_to_flatten(
            selected_prefix_indices,
            metadata.prompt_lens_cuda,
            metadata.valid_seq_indices_cuda,
            sparse_kv_offsets,
            metadata.num_contexts,
            metadata.total_sparse_ctx_indices,
            self.window_size,
            self.prompt_budget,
            self.num_kv_heads,
        )

        # Update KT cache
        kt_cache_tensor = metadata.kv_cache_manager.get_kt_buffers(self.layer_idx)

        triton_rocket_update_kt_cache_ctx(
            qkv_input.contiguous(),
            kt_cache_tensor,
            metadata.kt_cache_block_offsets[: metadata.num_contexts],
            metadata.context_cumsum_cuda[: metadata.num_contexts + 1],
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
        self, q: torch.Tensor, k: Optional[torch.Tensor], metadata: RocketTrtllmAttentionMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if k is None:
            qkv_input = q[metadata.num_ctx_tokens :]
            q_hidden_size = self.num_heads * self.head_dim
            k_hidden_size = self.num_kv_heads * self.head_dim
            q = qkv_input[:, :q_hidden_size]
            k = qkv_input[:, q_hidden_size : q_hidden_size + k_hidden_size]
        else:
            q = q[metadata.num_ctx_tokens :]
            k = k[metadata.num_ctx_tokens :]

        q = q.view(-1, self.num_kv_heads, self.num_heads // self.num_kv_heads, self.head_dim)

        return q, k

    @torch.compile(dynamic=True)
    def topr_filter(self, q: torch.Tensor) -> torch.Tensor:
        i1 = torch.topk(q.abs().sum(dim=2, keepdim=True), self.topr, dim=-1).indices
        q_mask = torch.zeros_like(q)
        q_mask.scatter_(-1, i1.expand_as(q[..., : self.topr]), 1)
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

        kt_cache_tensor = metadata.kv_cache_manager.get_kt_buffers(self.layer_idx)

        # Update KT cache with new key values
        triton_rocket_update_kt_cache_gen(
            k,
            kt_cache_tensor,
            metadata.kt_cache_block_offsets[metadata.num_contexts :],
            metadata.kv_lens_cuda_runtime[metadata.num_contexts :],
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
            metadata.kt_cache_block_offsets[metadata.num_contexts :],
            metadata.kv_lens_cuda_runtime[metadata.num_contexts :],
            metadata.cum_kt_lens_cuda,
            metadata.page_size,
            metadata.kt_tokens_per_block,
            metadata.kv_cache_manager.max_kt_blocks_per_seq,
            metadata.total_kt_tokens,
        )

        scores = triton_softmax(scores, metadata.cum_kt_lens_cuda, metadata.num_generations)

        # Mean over num_heads_per_kv for each batch separately
        scores = triton_rocket_reduce_scores(
            scores,
            metadata.cum_kt_lens_cuda,
            metadata.num_generations,
            self.num_kv_heads,
            self.num_heads // self.num_kv_heads,
        )

        sparse_attn_offsets = metadata.sparse_offsets_gen_cuda[: metadata.num_generations + 1]

        selected_indices = triton_topk(
            scores,
            metadata.cum_kt_lens_cuda,
            sparse_attn_offsets,
            metadata.total_sparse_gen_indices,
            metadata.topk,
        )

        return selected_indices, sparse_attn_offsets


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
        **kwargs,
    ):
        super().__init__(
            layer_idx,
            num_heads,
            head_dim,
            sparse_attention_config=sparse_attention_config,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            **kwargs,
        )
        self.topr = sparse_attention_config.topr
        self.topk = sparse_attention_config.topk
        self.prompt_budget = sparse_attention_config.prompt_budget
        self.window_size = sparse_attention_config.window_size
        self.kernel_size = sparse_attention_config.kernel_size
        self.page_size = sparse_attention_config.page_size
        assert sparse_attention_config.kt_cache_dtype == "bfloat16", (
            "Only bfloat16 kt cache is supported for Vanilla RocketKV"
        )

    def _single_request_sparse_kv_predict(
        self,
        q: Optional[Tensor],
        k: Optional[Tensor],
        v: Optional[Tensor],
        metadata: RocketVanillaAttentionMetadata,
        past_seen_token: int,
        sample_idx: int,
        **kwargs,
    ) -> tuple[Optional[torch.Tensor], int]:
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
        kt_cache_tensor = metadata.kv_cache_manager.get_kt_buffers(self.layer_idx)
        target_seq_len = past_seen_token + k_snap.size(1)
        kt_cache_position = torch.arange(
            past_seen_token // self.page_size,
            math.ceil(target_seq_len / self.page_size),
            device=q.device,
        )
        self._single_request_update_kt_cache(
            k_snap,
            kt_cache_tensor,
            metadata.kt_cache_block_offsets[sample_idx],
            target_seq_len,
            kt_cache_position,
            update=False,
        )
        return sparse_kv_indices, k_snap.size(1)

    def _single_request_sparse_attn_predict(
        self,
        q: Tensor,
        k: Optional[Tensor],
        v: Optional[Tensor],
        kv_cache_tensor: Tensor,
        metadata: RocketVanillaAttentionMetadata,
        past_seen_token: int,
        sample_idx: int,
        **kwargs,
    ) -> tuple[Optional[torch.Tensor], int]:
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
        sparse_indices = self._rocketkv_selection(q, k, metadata, past_seen_token, sample_idx)
        return sparse_indices, sparse_indices.size(1)

    def _get_snapkv_indices(self, q: Tensor, k: Tensor, sample_idx: int) -> Tensor:
        """Get SnapKV sparse kv indices from the input sequence for context phase."""
        bsz = 1
        seq_len = k.size(1)

        # If the sequence length is less than the prompt budget, do not enable sparse kv cache
        if seq_len <= self.prompt_budget:
            return None

        # Use last window_size tokens as observation
        # (1, num_kv_heads, window_size, head_dim)
        q_obs = q[:, :, -self.window_size :]
        # (1, num_kv_heads, seq_len, head_dim)
        k_pre = repeat_kv(k.transpose(1, 2), self.num_key_value_groups)

        dist = (
            torch.arange(0, self.window_size, device=q.device)[:, None]
            - torch.arange(0, seq_len, device=q.device)[None, :]
            + seq_len
            - self.window_size
        )
        attention_mask = dist >= 0

        score = torch.matmul(q_obs, k_pre.transpose(-1, -2)) / math.sqrt(self.head_dim)

        score = torch.masked_fill(
            score,
            attention_mask.view(1, 1, self.window_size, seq_len) == False,  # noqa: E712
            torch.scalar_tensor(float("-inf"), device=score.device, dtype=score.dtype),
        )

        score = torch.nn.functional.softmax(score, dim=-1)

        score = torch.masked_fill(
            score,
            attention_mask.view(1, 1, self.window_size, seq_len) == False,  # noqa: E712
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype),
        )

        score = score[:, :, -self.window_size :, : -self.window_size].sum(dim=-2)

        score = score.view(bsz, self.num_kv_heads, self.num_key_value_groups, -1).sum(dim=2)
        score = torch.nn.functional.max_pool1d(
            score, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1
        )

        # Select top important tokens from prefix
        prefix_len = seq_len - self.window_size
        selected_prefix_indices = (
            score.topk(self.prompt_budget - self.window_size, dim=-1).indices.sort().values
        )

        # Combine selected prefix indices with window indices
        window_indices = (
            torch.arange(prefix_len, seq_len, device=k.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(bsz, self.num_kv_heads, -1)
        )
        selected_indices = torch.cat([selected_prefix_indices, window_indices], dim=-1)

        return selected_indices.transpose(1, 2)

    def _single_request_update_kt_cache(
        self, k, kt_cache_tensor, kt_cache_block_offsets, seq_len, cache_position, update=True
    ):
        """Update KT cache for RocketKV algorithm."""
        # (1, num_pages_per_block, num_kv_heads, 2*head_dim)
        k_out = kt_cache_tensor[kt_cache_block_offsets[0], :, :, :].unsqueeze(0)

        # k: (1, seq_len, num_kv_heads, head_dim)
        if k is not None:
            padding_len = self.page_size - ((k.size(1) - 1) % self.page_size + 1)
            padding_tensor = torch.full(
                (1, padding_len, self.num_kv_heads, self.head_dim),
                float("inf"),
                device=k.device,
                dtype=k.dtype,
            )
            # (1, seq_len+padding_len, num_kv_heads, head_dim)
            k_min = torch.cat([k, padding_tensor], dim=1)
            k_min = k_min.reshape(1, -1, self.page_size, self.num_kv_heads, self.head_dim).amin(
                dim=2
            )
            k_max = torch.cat([k, -padding_tensor], dim=1)
            k_max = k_max.reshape(1, -1, self.page_size, self.num_kv_heads, self.head_dim).amax(
                dim=2
            )
            if update and (seq_len - 1) % self.page_size > 0:  # gen phase
                k_min = torch.min(k_min, k_out[:, cache_position, :, : self.head_dim])
                k_max = torch.max(k_max, k_out[:, cache_position, :, self.head_dim :])
            k_value = torch.cat([k_min, k_max], dim=-1)
            access_type = self._access_type[k_value.dtype.itemsize]
            k_out.view(dtype=access_type).index_copy_(
                1, cache_position, k_value.view(dtype=access_type)
            )

        return k_out[:, : math.ceil(seq_len / self.page_size), :, :]

    def _rocketkv_selection(
        self,
        q: Tensor,
        k: Tensor,
        metadata: RocketVanillaAttentionMetadata,
        past_seen_token: int,
        sample_idx: int,
    ) -> Tensor:
        """Implement RocketKV's two-stage selection process for generation phase."""
        bsz = 1
        q_len = q.size(2)

        # Helper functions
        def _gather(t: Tensor, dim: int, i: Tensor) -> Tensor:
            dim += (dim < 0) * t.ndim
            return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))

        @torch.compile(disable=not torch.cuda.is_available())
        def _scaled_softmax(x: Tensor, divscale: Tensor | float, dim: int) -> Tensor:
            return torch.softmax(x / divscale, dim=dim)

        # Get KT cache for key-token matching
        kt_cache_tensor = metadata.kv_cache_manager.get_kt_buffers(self.layer_idx)
        target_seq_len = past_seen_token + 1  # +1 for current token

        # Update KT cache
        kt_cache_position = torch.arange(
            past_seen_token // self.page_size,
            math.ceil(target_seq_len / self.page_size),
            device=q.device,
        )
        kt_states = self._single_request_update_kt_cache(
            k,
            kt_cache_tensor,
            metadata.kt_cache_block_offsets[sample_idx],
            target_seq_len,
            kt_cache_position,
        )

        # Reshape query for multi-head processing
        qi = q.view(
            bsz, self.num_kv_heads, self.num_heads // self.num_kv_heads, q_len, self.head_dim
        )
        qi_abs = torch.abs(qi)

        # Top-r selection on query features
        i1 = torch.topk(qi_abs.mean(dim=2, keepdim=True), self.topr, dim=-1).indices
        qi_hat = _gather(qi, -1, i1)

        # Generate signed indices for key-token matching
        i1_sign = torch.where(
            qi_hat.sum(dim=2, keepdim=True) > 0, i1 + self.head_dim, i1
        ).transpose(-1, -2)

        # Gather key tokens and compute attention scores
        kt_hat = _gather(kt_states.permute(0, 2, 3, 1).unsqueeze(2), -2, i1_sign)
        qk_hat = qi_hat @ kt_hat
        qk_hat = qk_hat.repeat_interleave(self.page_size, dim=-1)[:, :, :, :, :target_seq_len]
        scale = torch.sqrt(
            self.head_dim
            * torch.abs(qi_hat).sum(dim=-1, keepdim=True)
            / qi_abs.sum(dim=-1, keepdim=True)
        )

        # (1, num_kv_heads, num_heads, target_seq_len)
        s_hat = _scaled_softmax(qk_hat, scale, dim=-1)

        topk = min(self.topk, target_seq_len)
        i2 = torch.topk(s_hat.mean(dim=2, keepdim=True), topk, dim=-1).indices

        iKV = i2[:, :, 0, 0, :].transpose(1, 2)  # (1, topk, num_kv_heads)

        return iKV
