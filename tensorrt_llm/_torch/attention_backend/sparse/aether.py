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
AETHER Sparse Attention Backend for TensorRT-LLM
=================================================

Integrates AETHER block-sparse attention with TRT-LLM's attention system.

AETHER (Adaptive Event-driven Threshold Hybrid Entangled Rendering) uses 
block-level upper-bound scoring to dynamically prune attention blocks,
achieving sub-quadratic complexity for long sequences.

Reference: Sharma, T. (2024). DOI: 10.13141/RG.2.2.14811.27684
"""

from dataclasses import dataclass, field
from typing import Optional, Type

import torch
import torch.nn.functional as F

from tensorrt_llm.logger import logger

from ..vanilla import VanillaAttention, VanillaAttentionMetadata
from ..interface import AttentionMetadata, AttentionBackend

from .aether_kernels import (
    run_aether_sparse,
    run_aether_sparse_attention,
)


@dataclass
class AetherAttentionMetadata(VanillaAttentionMetadata):
    """Extended metadata with AETHER block statistics."""
    
    # Cached block statistics (computed once per KV cache update)
    block_means: Optional[torch.Tensor] = None  # (B, H, N_blocks, D)
    block_radii: Optional[torch.Tensor] = None  # (B, H, N_blocks)
    block_variances: Optional[torch.Tensor] = None  # (B, H, N_blocks)
    block_concentrations: Optional[torch.Tensor] = None  # (B, H, N_blocks)
    
    # Per-forward block mask
    block_mask: Optional[torch.Tensor] = None  # (B, H, N_blocks)
    
    # Sparse indexing data (for true sparse execution)
    active_indices: Optional[torch.Tensor] = None  # (B, H, Max_Active_Blocks)
    active_counts: Optional[torch.Tensor] = None   # (B, H)
    
    # Debug/monitoring stats
    last_sparsity: float = 0.0
    
    def invalidate_block_cache(self):
        """Invalidate cached block statistics when KV cache changes."""
        self.block_means = None
        self.block_radii = None
        self.block_variances = None
        self.block_concentrations = None
        self.active_indices = None
        self.active_counts = None


class AetherIndexManager:
    """
    Manages the conversion of sparse boolean masks to packed index buffers.
    
    Provides optimized routines for extracting 'active' block indices that
    can be used by custom Triton kernels for indirect memory access.
    """
    
    @staticmethod
    def extract_sparse_indices(
        block_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert boolean mask to packed index list.
        
        Args:
            block_mask: [B, H, N_blocks] boolean tensor
            
        Returns:
            packed_indices: [B, H, Max_Active] padded with -1
            active_counts: [B, H] number of active blocks per head
        """
        B, H, N = block_mask.shape
        device = block_mask.device
        
        # Count active blocks per head
        active_counts = block_mask.sum(dim=-1).to(torch.int32)
        max_active = active_counts.max().item()
        
        # Extract indices using nonzero and packing
        # Note: At 128k+ length, a custom prefix-sum kernel is faster,
        # but for an SS-level MVP, we leverage torch.nonzero and scatter.
        packed_indices = torch.full((B, H, max_active), -1, device=device, dtype=torch.int32)
        
        # Get indices of set bits
        b, h, idx = torch.nonzero(block_mask, as_tuple=True)
        
        # Generate target offsets within the packed buffer
        # We need a range [0..count-1] for each (b, h)
        # Using a trick with cumsum to get local indices
        flat_mask = block_mask.reshape(-1, N)
        local_idx = torch.arange(N, device=device).expand_as(flat_mask)
        # Filter for only active ones
        active_local_idx = local_idx[block_mask.reshape(-1, N)]
        
        # Final packing (simplified for now, pending high-throughput optimization)
        for i in range(B):
            for j in range(H):
                count = active_counts[i, j]
                if count > 0:
                    indices = torch.nonzero(block_mask[i, j], as_tuple=True)[0]
                    packed_indices[i, j, :count] = indices.to(torch.int32)
                    
        return packed_indices, active_counts


class AetherVanillaAttention(VanillaAttention):
    """
    AETHER sparse attention for VanillaAttention backend.
    
    Extends VanillaAttention to use block-sparse attention during
    the generation phase for improved throughput on long sequences.
    
    Usage:
        This backend is automatically selected when using:
        ```python
        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi import AetherSparseAttentionConfig
        
        llm = LLM(model="...", sparse_attention_config=AetherSparseAttentionConfig())
        ```
    """
    
    Metadata = AetherAttentionMetadata
    
    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        sparse_attention_config: Optional["SparseAttentionConfig"] = None,
        **kwargs,
    ):
        super().__init__(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            **kwargs,
        )
        
        # Extract AETHER-specific config with defaults
        self.aether_config = sparse_attention_config
        self.block_size = getattr(sparse_attention_config, 'block_size', 64)
        self.threshold = getattr(sparse_attention_config, 'threshold', 0.05)
        self.local_window = getattr(sparse_attention_config, 'local_window', 8)
        self.use_variance = getattr(sparse_attention_config, 'use_variance', True)
        self.use_concentration = getattr(sparse_attention_config, 'use_concentration', True)
        self.min_seq_length = getattr(sparse_attention_config, 'min_seq_length', 128)
        self.target_sparsity = getattr(sparse_attention_config, 'target_sparsity', None)
        self.scale = 1.0 / (head_dim ** 0.5)
        
        logger.info(f"[AETHER] Layer {layer_idx}: block_size={self.block_size}, "
                   f"threshold={self.threshold}, local_window={self.local_window}")
    
    def _compute_block_metadata(
        self,
        k: torch.Tensor,
    ) -> tuple:
        """
        Compute block-level metadata for sparse scoring.
        
        Args:
            k: Key tensor [B, H, S, D]
            
        Returns:
            Tuple of (means, radii, variances, concentrations)
        """
        B, H, S, D = k.shape
        block_size = self.block_size
        
        # Pad sequence to be divisible by block_size
        pad_len = (block_size - S % block_size) % block_size
        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, pad_len), value=0.0)
        
        S_padded = k.shape[2]
        num_blocks = S_padded // block_size
        
        # Reshape to blocks: [B, H, num_blocks, block_size, D]
        k_blocks = k.view(B, H, num_blocks, block_size, D)
        
        # Normalize keys for angular distance computation
        k_norm = F.normalize(k_blocks, dim=-1)
        
        # Block means (normalized)
        means_raw = k_norm.mean(dim=3)
        means = F.normalize(means_raw, dim=-1)
        
        # Compute distances from mean
        distances_sq = ((k_norm - means.unsqueeze(3)) ** 2).sum(dim=-1)
        
        # Radii = max distance
        radii = distances_sq.max(dim=-1).values.sqrt()
        
        # Variances = mean squared distance
        variances = distances_sq.mean(dim=-1) if self.use_variance else None
        
        # Concentrations = average alignment with mean
        concentrations = None
        if self.use_concentration:
            alignment = (k_norm * means.unsqueeze(3)).sum(dim=-1)
            concentrations = alignment.mean(dim=-1).clamp(min=0.1, max=1.0)
        
        return means, radii, variances, concentrations
    
    def _compute_block_scores(
        self,
        q: torch.Tensor,
        means: torch.Tensor,
        radii: torch.Tensor,
        variances: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute upper-bound attention scores per block.
        
        Uses Cauchy-Schwarz inequality to bound maximum possible attention
        score for each block without computing full attention.
        
        Args:
            q: Query tensor [B, H, q_len, D]
            means: Block centroids [B, H, num_blocks, D]
            radii: Block radii [B, H, num_blocks]
            variances: Block variances [B, H, num_blocks] (optional)
            
        Returns:
            block_mask: Boolean mask [B, H, num_blocks] indicating kept blocks
        """
        B, H, q_len, D = q.shape
        num_blocks = means.shape[2]
        
        # Compute centroid scores: dot(q, mean) * scale
        scores = torch.einsum('bhqd,bhnd->bhqn', q, means) * self.scale
        
        # Upper bound: add radius contribution
        q_norm = torch.norm(q, dim=-1, keepdim=True)  # [B, H, q_len, 1]
        radius_bonus = q_norm * radii.unsqueeze(2) * self.scale
        upper_bound = scores + radius_bonus
        
        # Variance penalty (optional): tighter bounds for low-variance blocks
        if variances is not None and self.use_variance:
            variance_factor = 1.0 / (1.0 + variances.unsqueeze(2))
            upper_bound = upper_bound * variance_factor
        
        # Threshold selection
        if self.target_sparsity is not None:
            # Adaptive: select blocks to achieve target sparsity
            thresholds = torch.quantile(
                upper_bound, self.target_sparsity, dim=-1, keepdim=True
            )
            block_mask = upper_bound >= thresholds
        else:
            # Fixed threshold relative to max score
            max_score = upper_bound.max(dim=-1, keepdim=True).values
            adaptive_threshold = self.threshold * max_score
            block_mask = upper_bound >= adaptive_threshold
        
        # For generation, squeeze the q_len dimension
        if q_len == 1:
            block_mask = block_mask.squeeze(2)
        
        return block_mask

    def _apply_aether_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = True,
        metadata: Optional[AetherAttentionMetadata] = None,
    ) -> torch.Tensor:
        """
        Apply SS-Level AETHER block-sparse attention.
        """
        B, H, S, D = k.shape
        q_len = q.shape[2]
        num_blocks = S // self.block_size
        
        # 1. Geometry Phase: Compute Radar Stats
        means, radii, variances, concentrations = self._compute_block_metadata(k)
        
        # 2. Prediction Phase: Score blocks
        block_mask = self._compute_block_scores(q, means, radii, variances)
        
        # 3. Persistence Phase: Local Window
        if self.local_window > 0 and num_blocks > self.local_window:
            local_start = max(0, num_blocks - self.local_window)
            block_mask[..., local_start:] = True
            
        # 4. Compiler Phase: Index Extraction (The Aether Index Manager)
        # For generation (q_len=1), we use the true sparse execution path
        if q_len == 1:
            active_indices, active_counts = AetherIndexManager.extract_sparse_indices(block_mask)
            
            # True Sparse Execution (Sub-quadratic)
            output = run_aether_sparse_attention(
                q.squeeze(2), # (B, H, D)
                k, v,
                active_indices,
                active_counts,
                block_size=self.block_size,
                scale=self.scale
            )
            return output.unsqueeze(2) # Return (B, H, 1, D)
            
        # For prefill (q_len > 1), we currently use the masked path (Future: Sparse Prefill)
        # Expand block mask to token-level mask
        block_size = self.block_size
        token_mask = block_mask.unsqueeze(-1).expand(-1, -1, -1, -1, block_size)
        token_mask = token_mask.reshape(B, H, q_len, -1)[:, :, :, :S]
        
        attn_mask = torch.zeros_like(token_mask, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~token_mask, float('-inf'))
        
        if is_causal:
            seq_positions = torch.arange(S, device=q.device)
            query_positions = torch.arange(S - q_len + 1, S + 1, device=q.device).unsqueeze(1)
            causal_mask = seq_positions.unsqueeze(0) <= query_positions
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_mask = attn_mask.masked_fill(~causal_mask, float('-inf'))
        
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=False,
        )
        
        return output
    
    def _single_request_attn_forward(
        self,
        q,
        key_states,
        value_states,
        is_causal,
        attn_mask,
        sparse_indices=None,
    ):
        """Override to inject AETHER sparse attention."""
        # Get sequence length
        if key_states.dim() == 4:
            seq_len = key_states.shape[2]
        else:
            seq_len = key_states.shape[1]
        
        # AETHER conditions:
        # - Sequence long enough for sparsity benefit
        # - No conflicting sparse indices or attention mask
        use_aether = (
            seq_len >= self.min_seq_length and
            seq_len >= self.block_size * 2 and
            sparse_indices is None and
            attn_mask is None
        )
        
        if use_aether:
            try:
                # Ensure shapes are [B, H, S, D]
                if key_states.dim() == 3:
                    key_states = key_states.unsqueeze(0)
                    value_states = value_states.unsqueeze(0)
                if q.dim() == 3:
                    q = q.unsqueeze(0)
                
                output = self._apply_aether_sparse(
                    q, key_states, value_states, is_causal=is_causal
                )
                return output
            except Exception as e:
                logger.warning(f"[AETHER] Fallback to standard attention: {e}")
        
        # Standard path
        return super()._single_request_attn_forward(
            q, key_states, value_states, is_causal, attn_mask, sparse_indices
        )


# Export for backend registration
__all__ = [
    'AetherVanillaAttention',
    'AetherAttentionMetadata',
]
