# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
AETHER Attention Backend
========================

Block-sparse attention backend using Event Radar scoring.

This module provides an attention backend that integrates AETHER sparse
attention into TensorRT-LLM's vanilla attention path, enabling dynamic
block selection during inference.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from ..interface import AttentionMetadata
from ..vanilla import VanillaAttention, VanillaAttentionMetadata
from ..sparse.aether_kernels import (
    run_aether_sparse, 
    precompute_metadata,
    aether_sparse_kernel,
)


@dataclass(kw_only=True)
class AetherSparseAttentionConfig:
    """Configuration for AETHER sparse attention."""
    
    # Block size for KV cache partitioning
    block_size: int = 64
    
    # Scoring threshold for block selection
    threshold: float = 0.15
    
    # Target sparsity for adaptive mode (0.8 = skip 80% of blocks)
    target_sparsity: float = 0.8
    
    # Feature flags
    use_variance: bool = True
    use_concentration: bool = True
    is_causal: bool = False
    
    # Causal mode parameters
    local_window: int = 4
    recency_decay: float = 0.95
    
    # Selection mode: "threshold", "adaptive", "topk"
    selection_mode: str = "threshold"
    
    # For top-k mode
    top_k: int = 32


@dataclass(kw_only=True)
class AetherAttentionMetadata(VanillaAttentionMetadata):
    """Metadata for AETHER attention with cached block statistics."""
    
    # Cached block metadata (computed once per KV cache population)
    block_means: Optional[torch.Tensor] = None  # (B, H, N_blocks, D)
    block_radii: Optional[torch.Tensor] = None  # (B, H, N_blocks)
    block_variances: Optional[torch.Tensor] = None  # (B, H, N_blocks)
    block_concentrations: Optional[torch.Tensor] = None  # (B, H, N_blocks)
    
    # Current block mask (computed per query)
    block_mask: Optional[torch.Tensor] = None  # (B, H, N_blocks)
    
    # Statistics for debugging/monitoring
    last_sparsity: float = 0.0
    last_scores: Optional[torch.Tensor] = None
    
    def prepare(self):
        """Prepare metadata before forward pass."""
        super().prepare()
        # Additional AETHER-specific preparation can go here


class AetherAttention(VanillaAttention):
    """
    AETHER sparse attention backend.
    
    Extends VanillaAttention with Event Radar block selection for
    accelerated attention computation on long sequences.
    
    During the forward pass:
    1. If metadata lacks block_means, compute them from the KV cache
    2. Call sparse_attn_predict to generate block mask
    3. Apply block mask to attention computation
    """
    
    Metadata = AetherAttentionMetadata
    
    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        sparse_attention_config: Optional[AetherSparseAttentionConfig] = None,
        **kwargs,
    ):
        super().__init__(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            **kwargs,
        )
        
        self.sparse_config = sparse_attention_config or AetherSparseAttentionConfig()
        self.scale = 1.0 / (head_dim ** 0.5)
    
    def _update_block_metadata(
        self,
        keys: torch.Tensor,
        metadata: AetherAttentionMetadata,
    ) -> None:
        """
        Update block metadata from KV cache.
        
        Called when metadata.block_means is None or stale.
        
        Args:
            keys: Key tensor (B, H, S, D)
            metadata: Attention metadata to update
        """
        config = self.sparse_config
        
        means, radii, variances, concentrations = precompute_metadata(
            keys,
            block_size=config.block_size,
            compute_variance=config.use_variance,
            compute_concentration=config.use_concentration,
        )
        
        metadata.block_means = means
        metadata.block_radii = radii
        metadata.block_variances = variances
        metadata.block_concentrations = concentrations
    
    def _single_request_sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        kv_cache_tensor: torch.Tensor,
        metadata: AetherAttentionMetadata,
        past_seen_token: int,
        sample_idx: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """
        Generate sparse attention mask using AETHER Event Radar.
        
        This method is called during the attention forward pass to determine
        which KV blocks should be attended to.
        
        Args:
            q: Query tensor (1, num_heads, head_dim) for single request
            k: Key tensor (optional, for computing metadata)
            v: Value tensor (optional)
            kv_cache_tensor: Full KV cache tensor
            metadata: Attention metadata with block statistics
            past_seen_token: Number of previously seen tokens
            sample_idx: Index of current sample in batch
        
        Returns:
            Token-level sparse indices or None for full attention
        """
        config = self.sparse_config
        
        # Skip sparse attention for short sequences
        seq_len = past_seen_token + (q.shape[1] if q.ndim > 2 else 1)
        n_blocks = seq_len // config.block_size
        if n_blocks < 4:
            return None
        
        # Ensure metadata is up to date
        if metadata.block_means is None:
            # Need to extract keys from KV cache for metadata computation
            # This is a simplified path - production code would handle paged KV cache
            return None
        
        # Get query for this sample
        if q.ndim == 3:
            query = q[sample_idx]  # (H, D)
        else:
            query = q.view(self.num_heads, self.head_dim)
        
        query = query.unsqueeze(0)  # (1, H, D)
        
        # Get block metadata for this sample
        means = metadata.block_means[sample_idx:sample_idx+1]
        radii = metadata.block_radii[sample_idx:sample_idx+1]
        variances = metadata.block_variances[sample_idx:sample_idx+1] if metadata.block_variances is not None else None
        concentrations = metadata.block_concentrations[sample_idx:sample_idx+1] if metadata.block_concentrations is not None else None
        
        # Run Event Radar scoring
        if config.selection_mode == "threshold":
            mask, scores = run_aether_sparse(
                query, means, radii,
                block_variances=variances,
                block_concentrations=concentrations,
                threshold=config.threshold,
                scale=self.scale,
                use_variance=config.use_variance,
                use_concentration=config.use_concentration,
                is_causal=config.is_causal,
                local_window=config.local_window,
                recency_decay=config.recency_decay,
            )
        elif config.selection_mode == "adaptive":
            from ..sparse.aether_kernels import run_aether_sparse
            
            # Compute scores with very low threshold
            mask, scores = run_aether_sparse(
                query, means, radii,
                block_variances=variances,
                block_concentrations=concentrations,
                threshold=-1e9,
                scale=self.scale,
                use_variance=config.use_variance,
                use_concentration=config.use_concentration,
                is_causal=config.is_causal,
            )
            # Apply adaptive threshold via quantile
            thresholds = torch.quantile(scores, config.target_sparsity, dim=-1, keepdim=True)
            mask = scores > thresholds
        else:  # topk
            mask, scores = run_aether_sparse(
                query, means, radii,
                block_variances=variances,
                block_concentrations=concentrations,
                threshold=-1e9,
                scale=self.scale,
                use_variance=config.use_variance,
                use_concentration=config.use_concentration,
            )
            # Select top-k blocks
            k_actual = min(config.top_k, n_blocks)
            _, top_indices = torch.topk(scores, k=k_actual, dim=-1)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(-1, top_indices, True)
        
        # Store statistics
        metadata.last_sparsity = 1.0 - mask.float().mean().item()
        metadata.last_scores = scores
        metadata.block_mask = mask
        
        # Expand block mask to token indices
        # mask: (1, H, N_blocks) -> indices for gathering
        block_size = config.block_size
        
        # Get active block indices per head
        sparse_indices_list = []
        for h in range(self.num_heads):
            active_blocks = mask[0, h].nonzero(as_tuple=True)[0]
            if len(active_blocks) == 0:
                # Fallback: keep at least one block
                active_blocks = torch.tensor([n_blocks - 1], device=mask.device)
            
            # Expand to token indices
            token_indices = []
            for block_idx in active_blocks:
                start = block_idx * block_size
                end = min(start + block_size, seq_len)
                token_indices.append(torch.arange(start, end, device=mask.device))
            
            sparse_indices_list.append(torch.cat(token_indices))
        
        # For vanilla backend, return the mask tensor
        # The actual sparse gathering is handled by the parent class
        return mask.squeeze(0)  # (H, N_blocks)
    
    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: AetherAttentionMetadata,
        **kwargs,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Batch-level sparse attention prediction.
        
        Called for generation phase to predict which KV tokens to attend.
        
        Args:
            q: Query tensor (total_tokens, num_heads * head_dim)
            k: Key tensor (optional)
            metadata: Attention metadata
        
        Returns:
            Tuple of (sparse_indices, sparse_offsets) or None
        """
        # For context phase, we don't do sparse prediction
        if metadata.num_contexts > 0:
            return None
        
        # Check if we have enough tokens for sparse attention
        if metadata.block_means is None:
            return None
        
        B = metadata.num_seqs
        config = self.sparse_config
        
        # Reshape query
        q_reshaped = q.view(B, -1, self.num_heads, self.head_dim)
        q_last = q_reshaped[:, -1, :, :]  # (B, H, D) - last token query
        
        # Get block metadata
        means = metadata.block_means
        radii = metadata.block_radii
        
        # Run scoring
        mask, scores = run_aether_sparse(
            q_last, means, radii,
            block_variances=metadata.block_variances,
            block_concentrations=metadata.block_concentrations,
            threshold=config.threshold,
            scale=self.scale,
            use_variance=config.use_variance,
            use_concentration=config.use_concentration,
            is_causal=config.is_causal,
            local_window=config.local_window,
            recency_decay=config.recency_decay,
        )
        
        # Store mask
        metadata.block_mask = mask
        metadata.last_sparsity = 1.0 - mask.float().mean().item()
        
        # Convert block mask to token indices for TRT-LLM
        # This would integrate with the paged KV cache system
        # For now, return the block mask for vanilla backend consumption
        
        return mask, scores
