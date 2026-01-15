# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
AETHER Sparse Attention Kernel - Pure PyTorch Implementation
============================================================
Block-sparse attention using upper-bound pruning for efficient inference.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AetherConfig:
    """Configuration for AETHER sparse attention."""
    block_size: int = 64
    threshold: float = 0.05  # Lowered for better quality (less pruning)
    use_variance: bool = True
    use_concentration: bool = True
    is_causal: bool = True
    local_window: int = 8  # Increased local window for better quality
    recency_decay: float = 0.95


def precompute_block_metadata(
    k: torch.Tensor,
    block_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precompute block-level metadata for sparse attention scoring.
    
    Args:
        k: Key tensor [B, num_kv_heads, S, D]
        block_size: Size of each KV block
        
    Returns:
        means: Block centroids [B, H, num_blocks, D]
        radii: Block radii [B, H, num_blocks]
        variances: Block variances [B, H, num_blocks]
    """
    B, H, S, D = k.shape
    
    # Pad sequence to be divisible by block_size
    pad_len = (block_size - S % block_size) % block_size
    if pad_len > 0:
        k = F.pad(k, (0, 0, 0, pad_len), value=0.0)
    
    S_padded = k.shape[2]
    num_blocks = S_padded // block_size
    
    # Reshape to blocks: [B, H, num_blocks, block_size, D]
    k_blocks = k.view(B, H, num_blocks, block_size, D)
    
    # Compute block centroids (means)
    means = k_blocks.mean(dim=3)  # [B, H, num_blocks, D]
    
    # Compute block radii (max distance from centroid)
    diff = k_blocks - means.unsqueeze(3)  # [B, H, num_blocks, block_size, D]
    distances = torch.norm(diff, dim=-1)  # [B, H, num_blocks, block_size]
    radii = distances.max(dim=-1).values  # [B, H, num_blocks]
    
    # Compute block variances
    variances = distances.var(dim=-1)  # [B, H, num_blocks]
    
    return means, radii, variances


def compute_block_scores(
    q: torch.Tensor,
    means: torch.Tensor,
    radii: torch.Tensor,
    variances: Optional[torch.Tensor] = None,
    threshold: float = 0.15,
    use_variance: bool = True,
) -> torch.Tensor:
    """
    Compute upper-bound attention scores per block.
    
    Args:
        q: Query tensor [B, H, 1, D] or [B, H, q_len, D]
        means: Block centroids [B, H, num_blocks, D]
        radii: Block radii [B, H, num_blocks]
        variances: Block variances [B, H, num_blocks]
        threshold: Pruning threshold
        use_variance: Whether to use variance-aware scoring
        
    Returns:
        block_mask: Boolean mask [B, H, num_blocks] indicating kept blocks
    """
    B, H, q_len, D = q.shape
    num_blocks = means.shape[2]
    
    # Compute query-centroid similarity
    # q: [B, H, q_len, D], means: [B, H, num_blocks, D]
    # For generation (q_len=1), we use the single query
    
    # Use raw dot product (like attention) instead of normalized similarity
    scale = 1.0 / (D ** 0.5)
    
    # Dot product: [B, H, q_len, num_blocks]
    scores = torch.einsum('bhqd,bhnd->bhqn', q, means) * scale
    
    # Upper bound: score + radius contribution (Cauchy-Schwarz style bound)
    q_norm = torch.norm(q, dim=-1, keepdim=True)  # [B, H, q_len, 1]
    radius_bonus = q_norm * radii.unsqueeze(2) * scale  # [B, H, 1, num_blocks] after broadcast
    upper_bound = scores + radius_bonus
    
    # Variance penalty (lower variance = tighter bound = more confidence)
    if use_variance and variances is not None:
        variance_penalty = 1.0 / (1.0 + variances.unsqueeze(2))  # [B, H, 1, num_blocks]
        upper_bound = upper_bound * variance_penalty
    
    # Compute dynamic threshold based on max score
    max_score = upper_bound.max(dim=-1, keepdim=True).values
    adaptive_threshold = threshold * max_score
    
    # Create block mask
    block_mask = upper_bound >= adaptive_threshold  # [B, H, q_len, num_blocks]
    
    # For generation, squeeze the q_len dimension
    if q_len == 1:
        block_mask = block_mask.squeeze(2)  # [B, H, num_blocks]
    
    return block_mask


def aether_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: Optional[AetherConfig] = None,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    AETHER Sparse Attention - Main Entry Point.
    
    Computes block-sparse attention by:
    1. Computing block-level metadata for keys
    2. Scoring blocks using upper-bound pruning
    3. Gathering only important KV blocks
    4. Computing attention on sparse subset
    
    Args:
        q: Query tensor [B, H, q_len, D]
        k: Key tensor [B, H, S, D]
        v: Value tensor [B, H, S, D]
        config: AETHER configuration
        is_causal: Whether to apply causal masking
        
    Returns:
        output: Attention output [B, H, q_len, D]
    """
    if config is None:
        config = AetherConfig()
    
    B, H, S, D = k.shape
    q_len = q.shape[2]
    block_size = config.block_size
    
    # For very short sequences, fall back to dense attention
    if S <= block_size * 2:
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    
    # Step 1: Compute block metadata
    means, radii, variances = precompute_block_metadata(k, block_size)
    num_blocks = means.shape[2]
    
    # Step 2: Score blocks and get mask
    block_mask = compute_block_scores(
        q, means, radii, variances,
        threshold=config.threshold,
        use_variance=config.use_variance,
    )
    
    # Step 3: Apply causal constraint (only attend to past blocks)
    if is_causal and q_len == 1:
        # For generation: current position determines which blocks are visible
        # Assume we're at position S (end of sequence)
        # All blocks up to current position are visible
        pass  # All blocks are past blocks in generation mode
    
    # Step 4: Apply local window (always keep recent blocks)
    if config.local_window > 0:
        # Always keep the last `local_window` blocks
        local_start = max(0, num_blocks - config.local_window)
        block_mask[..., local_start:] = True
    
    # Step 6: For sequences where all blocks are kept (no sparsity),
    # just use standard SDPA for correctness
    if block_mask.all():
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    
    # Expand block mask to token-level mask
    # block_mask: [B, H, num_blocks]
    # Need: [B, H, q_len, S]
    
    # Expand blocks to tokens
    token_mask = block_mask.unsqueeze(-1).expand(-1, -1, -1, block_size)  # [B, H, num_blocks, block_size]
    token_mask = token_mask.reshape(B, H, -1)  # [B, H, num_blocks * block_size]
    
    # Trim to actual sequence length
    token_mask = token_mask[..., :S]  # [B, H, S]
    
    # Expand for q_len
    token_mask = token_mask.unsqueeze(2).expand(-1, -1, q_len, -1)  # [B, H, q_len, S]
    
    # Convert boolean mask to attention mask
    # True = keep (0.0), False = mask (-inf)
    attn_mask = torch.zeros_like(token_mask, dtype=q.dtype)
    attn_mask = attn_mask.masked_fill(~token_mask, float('-inf'))
    
    # Apply causal mask on top (mask future positions)
    if is_causal and q_len == 1:
        # For generation: we're at position S, can only see positions 0 to S-1
        # All positions are valid (past), no additional masking needed
        pass
    elif is_causal:
        # For prefill: create proper causal mask
        # Position i can attend to positions 0..i
        # Create lower triangular mask
        seq_positions = torch.arange(S, device=q.device)
        query_positions = torch.arange(S - q_len + 1, S + 1, device=q.device).unsqueeze(1)  # [q_len, 1]
        causal_mask = seq_positions.unsqueeze(0) <= query_positions  # [q_len, S]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, S]
        attn_mask = attn_mask.masked_fill(~causal_mask, float('-inf'))
    
    # Compute attention with the combined mask
    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        is_causal=False,  # We already applied causal mask
    )
    
    return output


# Global flag to enable/disable AETHER
AETHER_ENABLED = True


def maybe_aether_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Wrapper that conditionally uses AETHER or falls back to standard SDPA.
    """
    if AETHER_ENABLED and attn_mask is None:
        try:
            return aether_sparse_attention(q, k, v, is_causal=is_causal)
        except Exception as e:
            print(f"[AETHER] Fallback to SDPA due to: {e}")
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
    else:
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, attn_mask=attn_mask, scale=scale
        )
