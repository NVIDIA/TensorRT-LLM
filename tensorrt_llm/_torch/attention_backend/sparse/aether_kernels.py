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
AETHER Sparse Attention Kernels
===============================

Unified Triton kernels for block-sparse attention with deviation-based scoring.

This module provides a consolidated kernel implementation that supports multiple
scoring modes via compile-time flags:
    - USE_VARIANCE: Variance-aware scoring (v2 behavior)
    - USE_CONCENTRATION: Tight bound scoring (v3 behavior)
    - IS_CAUSAL: Causal streaming with recency bias

Reference:
    Sharma, T. (2024). AETHER - Adaptive Event-driven Threshold Hybrid
    Entangled Rendering. DOI: 10.13141/RG.2.2.14811.27684
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional

__all__ = [
    'aether_sparse_kernel',
    'metadata_calculator_kernel',
    'run_aether_sparse',
    'precompute_metadata',
]

# Supported dtypes for kernel operations
SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


# =============================================================================
# Unified Sparse Attention Kernel
# =============================================================================

@triton.jit
def aether_sparse_kernel(
    Q_ptr, Mean_Ptr, Rad_Ptr,
    Var_Ptr,   # Optional: variance data (ignored if USE_VARIANCE=False)
    Conc_Ptr,  # Optional: concentration data (ignored if USE_CONCENTRATION=False)
    Mask_Out_Ptr, Score_Out_Ptr,
    stride_q_b, stride_q_h, stride_q_d,
    stride_m_b, stride_m_h, stride_m_blk, stride_m_d,
    stride_r_b, stride_r_h, stride_r_blk,
    stride_v_b, stride_v_h, stride_v_blk,
    stride_c_b, stride_c_h, stride_c_blk,
    stride_mask_b, stride_mask_h, stride_mask_blk,
    stride_score_b, stride_score_h, stride_score_blk,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    THRESHOLD: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    USE_VARIANCE: tl.constexpr,
    USE_CONCENTRATION: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    RECENCY_DECAY: tl.constexpr,
):
    """
    Unified AETHER sparse attention kernel with configurable scoring modes.
    
    This kernel consolidates event_radar_kernel, event_radar_kernel_v2,
    tight_bound_kernel, and causal_event_radar_kernel into a single
    implementation using compile-time flags.
    
    Scoring formula (combined):
        base_score = dot(q, mean) * scale
        deviation = ||q|| * radius * scale
        
        if USE_VARIANCE:
            deviation *= (1 + sqrt(variance))
        if USE_CONCENTRATION:
            deviation *= concentration
        
        score = base_score + deviation
        
        if IS_CAUSAL:
            recency_bonus = (1 - block_idx/N_BLOCKS) * (1 - RECENCY_DECAY)
            score *= (1 + recency_bonus)
            is_active |= (block_idx >= N_BLOCKS - LOCAL_WINDOW)
    """
    block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    dim_offsets = tl.arange(0, HEAD_DIM)
    
    # Load query vector
    q = tl.load(Q_ptr + bh_idx * stride_q_h + dim_offsets * stride_q_d).to(tl.float32)
    
    # Load block mean
    mean = tl.load(
        Mean_Ptr + bh_idx * stride_m_h + block_idx * stride_m_blk + dim_offsets * stride_m_d
    ).to(tl.float32)
    
    # Load block radius
    radius = tl.load(Rad_Ptr + bh_idx * stride_r_h + block_idx * stride_r_blk).to(tl.float32)
    
    # Compute base scores
    center_score = tl.sum(q * mean) * SCALE
    q_norm = tl.sqrt(tl.sum(q * q))
    deviation = q_norm * radius * SCALE
    
    # Apply variance factor if enabled
    if USE_VARIANCE:
        variance = tl.load(Var_Ptr + bh_idx * stride_v_h + block_idx * stride_v_blk).to(tl.float32)
        variance_factor = 1.0 + tl.sqrt(variance)
        deviation = deviation * variance_factor
    
    # Apply concentration factor if enabled (tighter bound)
    if USE_CONCENTRATION:
        concentration = tl.load(Conc_Ptr + bh_idx * stride_c_h + block_idx * stride_c_blk).to(tl.float32)
        deviation = deviation * concentration
    
    # Compute attention potential
    attention_potential = center_score + deviation
    
    # Apply causal recency bias if enabled
    if IS_CAUSAL:
        recency = block_idx / N_BLOCKS
        recency_bonus = (1.0 - recency) * (1.0 - RECENCY_DECAY)
        attention_potential = attention_potential * (1.0 + recency_bonus)
        
        # Force local window blocks to be active
        is_local = block_idx >= (N_BLOCKS - LOCAL_WINDOW)
        is_active = (attention_potential > THRESHOLD) | is_local
    else:
        is_active = attention_potential > THRESHOLD
    
    tl.store(Mask_Out_Ptr + bh_idx * stride_mask_h + block_idx * stride_mask_blk, is_active)
    tl.store(Score_Out_Ptr + bh_idx * stride_score_h + block_idx * stride_score_blk, attention_potential)


@triton.jit
def metadata_calculator_kernel(
    Keys_Ptr, Mean_Out_Ptr, Rad_Out_Ptr,
    Var_Out_Ptr,   # Optional: written only if COMPUTE_VARIANCE=True
    Conc_Out_Ptr,  # Optional: written only if COMPUTE_CONCENTRATION=True
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_m_b, stride_m_h, stride_m_blk, stride_m_d,
    stride_r_b, stride_r_h, stride_r_blk,
    stride_v_b, stride_v_h, stride_v_blk,
    stride_c_b, stride_c_h, stride_c_blk,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    COMPUTE_VARIANCE: tl.constexpr,
    COMPUTE_CONCENTRATION: tl.constexpr,
):
    """
    Kernel to precompute block metadata from KV cache.
    
    Computes:
        - mean: Normalized average of keys in block
        - radius: Maximum angular deviation from mean
        - variance (optional): Mean squared distance from mean
        - concentration (optional): Average alignment with mean direction
    """
    block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    dim_offsets = tl.arange(0, HEAD_DIM)
    
    # Accumulate for mean computation
    mean_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    
    # First pass: compute mean
    for i in range(BLOCK_SIZE):
        key_offset = bh_idx * stride_k_h + (block_idx * BLOCK_SIZE + i) * stride_k_s
        key = tl.load(Keys_Ptr + key_offset + dim_offsets * stride_k_d).to(tl.float32)
        
        # Normalize key
        key_norm = tl.sqrt(tl.sum(key * key))
        key_normalized = key / (key_norm + 1e-8)
        mean_acc += key_normalized
    
    # Normalize mean
    mean = mean_acc / BLOCK_SIZE
    mean_norm = tl.sqrt(tl.sum(mean * mean))
    mean = mean / (mean_norm + 1e-8)
    
    # Store mean
    mean_offset = bh_idx * stride_m_h + block_idx * stride_m_blk
    tl.store(Mean_Out_Ptr + mean_offset + dim_offsets * stride_m_d, mean)
    
    # Second pass: compute radius, variance, concentration
    max_dist_sq = tl.zeros([1], dtype=tl.float32)
    total_dist_sq = tl.zeros([1], dtype=tl.float32)
    total_alignment = tl.zeros([1], dtype=tl.float32)
    
    for i in range(BLOCK_SIZE):
        key_offset = bh_idx * stride_k_h + (block_idx * BLOCK_SIZE + i) * stride_k_s
        key = tl.load(Keys_Ptr + key_offset + dim_offsets * stride_k_d).to(tl.float32)
        
        # Normalize key
        key_norm_val = tl.sqrt(tl.sum(key * key))
        key_normalized = key / (key_norm_val + 1e-8)
        
        # Distance from mean
        diff = key_normalized - mean
        dist_sq = tl.sum(diff * diff)
        
        max_dist_sq = tl.maximum(max_dist_sq, dist_sq)
        total_dist_sq += dist_sq
        
        # Alignment with mean (for concentration)
        if COMPUTE_CONCENTRATION:
            alignment = tl.sum(key_normalized * mean)
            total_alignment += alignment
    
    # Store radius
    radius = tl.sqrt(max_dist_sq)
    tl.store(Rad_Out_Ptr + bh_idx * stride_r_h + block_idx * stride_r_blk, radius)
    
    # Store variance if requested
    if COMPUTE_VARIANCE:
        variance = total_dist_sq / BLOCK_SIZE
        tl.store(Var_Out_Ptr + bh_idx * stride_v_h + block_idx * stride_v_blk, variance)
    
    # Store concentration if requested
    if COMPUTE_CONCENTRATION:
        concentration = tl.maximum(total_alignment / BLOCK_SIZE, 0.1)
        concentration = tl.minimum(concentration, 1.0)
        tl.store(Conc_Out_Ptr + bh_idx * stride_c_h + block_idx * stride_c_blk, concentration)


# =============================================================================
# Python Runner Functions
# =============================================================================

def run_aether_sparse(
    query: torch.Tensor,
    block_means: torch.Tensor,
    block_radii: torch.Tensor,
    block_variances: Optional[torch.Tensor] = None,
    block_concentrations: Optional[torch.Tensor] = None,
    threshold: float = 0.15,
    scale: Optional[float] = None,
    use_variance: bool = False,
    use_concentration: bool = False,
    is_causal: bool = False,
    local_window: int = 4,
    recency_decay: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run unified AETHER sparse attention kernel.
    
    Args:
        query: Query tensor (B, H, D)
        block_means: Pre-computed block means (B, H, N_blocks, D)
        block_radii: Pre-computed block radii (B, H, N_blocks)
        block_variances: Optional variance tensor (B, H, N_blocks)
        block_concentrations: Optional concentration tensor (B, H, N_blocks)
        threshold: Activation threshold
        scale: Attention scale (default: 1/sqrt(D))
        use_variance: Enable variance-aware scoring
        use_concentration: Enable concentration-based tight bounds
        is_causal: Enable causal masking with recency bias
        local_window: Number of recent blocks to always keep (causal mode)
        recency_decay: Decay factor for recency bonus
    
    Returns:
        Tuple of (mask, scores):
            - mask: Boolean mask of shape (B, H, N_blocks)
            - scores: Attention potential scores of shape (B, H, N_blocks)
    """
    B, H, D = query.shape
    _, _, N_blocks, _ = block_means.shape
    
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    
    # Validate optional tensors
    if use_variance and block_variances is None:
        raise ValueError("block_variances required when use_variance=True")
    if use_concentration and block_concentrations is None:
        raise ValueError("block_concentrations required when use_concentration=True")
    
    # Create dummy tensors if not provided (kernel will ignore them)
    if block_variances is None:
        block_variances = torch.zeros(B, H, N_blocks, device=query.device, dtype=torch.float32)
    if block_concentrations is None:
        block_concentrations = torch.zeros(B, H, N_blocks, device=query.device, dtype=torch.float32)
    
    mask_out = torch.zeros(B, H, N_blocks, device=query.device, dtype=torch.int32)
    score_out = torch.zeros(B, H, N_blocks, device=query.device, dtype=torch.float32)
    
    grid = (N_blocks, B * H)
    
    aether_sparse_kernel[grid](
        query, block_means, block_radii,
        block_variances, block_concentrations,
        mask_out, score_out,
        query.stride(0), query.stride(1), query.stride(2),
        block_means.stride(0), block_means.stride(1), block_means.stride(2), block_means.stride(3),
        block_radii.stride(0), block_radii.stride(1), block_radii.stride(2),
        block_variances.stride(0), block_variances.stride(1), block_variances.stride(2),
        block_concentrations.stride(0), block_concentrations.stride(1), block_concentrations.stride(2),
        mask_out.stride(0), mask_out.stride(1), mask_out.stride(2),
        score_out.stride(0), score_out.stride(1), score_out.stride(2),
        HEAD_DIM=D,
        SCALE=scale,
        THRESHOLD=threshold,
        N_BLOCKS=N_blocks,
        USE_VARIANCE=use_variance,
        USE_CONCENTRATION=use_concentration,
        IS_CAUSAL=is_causal,
        LOCAL_WINDOW=local_window,
        RECENCY_DECAY=recency_decay,
    )
    
    return mask_out.bool(), score_out


def precompute_metadata(
    keys: torch.Tensor,
    block_size: int = 64,
    compute_variance: bool = False,
    compute_concentration: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Precompute block metadata from KV cache using Python (fallback).
    
    For production, use metadata_calculator_kernel for GPU acceleration.
    
    Args:
        keys: Key tensor (B, H, S, D)
        block_size: Block size for partitioning
        compute_variance: Whether to compute variance
        compute_concentration: Whether to compute concentration
    
    Returns:
        Tuple of (means, radii, variances, concentrations)
    """
    B, H, S, D = keys.shape
    assert S % block_size == 0, f"Sequence length {S} not divisible by block_size {block_size}"
    
    N_blocks = S // block_size
    
    # Reshape into blocks
    keys_blocked = keys.reshape(B, H, N_blocks, block_size, D)
    
    # Normalize keys
    keys_norm = torch.nn.functional.normalize(keys_blocked, dim=-1)
    
    # Block means (normalized)
    means_raw = keys_norm.mean(dim=3)
    means = torch.nn.functional.normalize(means_raw, dim=-1)
    
    # Compute alignment of each key with block mean
    alignment = (keys_norm * means.unsqueeze(3)).sum(dim=-1)
    
    # Distances from mean
    distances_sq = ((keys_norm - means.unsqueeze(3)) ** 2).sum(dim=-1)
    
    # Radii = max distance
    radii = distances_sq.max(dim=-1).values.sqrt()
    
    # Variances
    variances = None
    if compute_variance:
        variances = distances_sq.mean(dim=-1)
    
    # Concentrations
    concentrations = None
    if compute_concentration:
        concentrations = alignment.mean(dim=-1).clamp(min=0.1, max=1.0)
    
    return means, radii, variances, concentrations
