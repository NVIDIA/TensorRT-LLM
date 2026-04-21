# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Shared MLA RoPE utilities for auto_deploy custom models.

Contains helper functions for RoPE weight de-interleaving and FP8 dequantization,
used by DeepSeek V3 and GLM4 MoE Lite model implementations.
"""

from typing import Dict

import torch

from tensorrt_llm.quantization.utils.fp8_matrix_weight_dequant import (
    dequant_fp8_nk_weight_auto_scale_layout,
)

from ...utils.quantization_utils import FLOAT8_DTYPES


def _index_select_with_float8_cpu_workaround(
    tensor: torch.Tensor, dim: int, index: torch.Tensor
) -> torch.Tensor:
    """Index-select helper that preserves raw FP8 encodings on CPU.

    PyTorch CPU indexing on float8 tensors currently raises ``index_cpu`` errors.
    This hook only needs to reorder checkpoint values, so for CPU float8 tensors we
    reorder the underlying bytes via a ``uint8`` view and reinterpret them back.
    """
    if tensor.device.type != "cpu" or tensor.dtype not in FLOAT8_DTYPES:
        return tensor.index_select(dim, index.to(device=tensor.device))

    uint8_view = tensor.view(torch.uint8)
    reordered = uint8_view.index_select(dim, index.to(device=tensor.device))
    return reordered.view(tensor.dtype)


def _rope_deinterleave_load_hook(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    *args,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    num_heads: int,
    kv_lora_rank: int,
    num_layers: int,
):
    """Pre-load hook that permutes RoPE weight columns from interleaved to non-interleaved.

    For q_b_proj: output shape is [num_heads * (qk_nope_head_dim + qk_rope_head_dim), q_lora_rank]
      -> permute the last qk_rope_head_dim columns within each head's block

    For kv_a_proj_with_mqa: output shape is [kv_lora_rank + qk_rope_head_dim, hidden_size]
      -> permute the last qk_rope_head_dim rows (the k_pe portion)
    """
    d = qk_rope_head_dim
    perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
    qk_head_dim = qk_nope_head_dim + d

    for layer_idx in range(num_layers):
        layer_prefix = f"{prefix}model.layers.{layer_idx}.self_attn."

        # --- q_b_proj.weight ---
        q_key = layer_prefix + "q_b_proj.weight"
        if q_key in state_dict:
            w = state_dict[q_key]
            orig_dtype = w.dtype
            if not w.is_floating_point() or w.dtype == torch.float8_e4m3fn:
                w = w.to(torch.bfloat16)
            w = w.view(num_heads, qk_head_dim, -1)
            w_nope = w[:, :qk_nope_head_dim, :]
            w_rope = w[:, qk_nope_head_dim:, :]
            w_rope = _index_select_with_float8_cpu_workaround(w_rope, 1, perm)
            w = torch.cat([w_nope, w_rope], dim=1)
            state_dict[q_key] = w.view(-1, w.shape[-1]).to(orig_dtype)

        # --- kv_a_proj_with_mqa.weight ---
        kv_key = layer_prefix + "kv_a_proj_with_mqa.weight"
        if kv_key in state_dict:
            w = state_dict[kv_key]
            orig_dtype = w.dtype
            if not w.is_floating_point() or w.dtype == torch.float8_e4m3fn:
                w = w.to(torch.bfloat16)
            w_kv = w[:kv_lora_rank, :]
            w_pe = w[kv_lora_rank:, :]
            w_pe = _index_select_with_float8_cpu_workaround(w_pe, 0, perm)
            state_dict[kv_key] = torch.cat([w_kv, w_pe], dim=0).to(orig_dtype)

        # --- kv_a_proj_with_mqa.bias (if present) ---
        kv_bias_key = layer_prefix + "kv_a_proj_with_mqa.bias"
        if kv_bias_key in state_dict:
            b = state_dict[kv_bias_key]
            b_kv = b[:kv_lora_rank]
            b_pe = b[kv_lora_rank:]
            b_pe = _index_select_with_float8_cpu_workaround(b_pe, 0, perm)
            state_dict[kv_bias_key] = torch.cat([b_kv, b_pe])


def _kv_b_proj_dequant_load_hook(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    *args,
    num_layers: int,
):
    """Pre-load hook that dequantizes FP8 kv_b_proj weights using per-block scales.

    kv_b_proj.weight is passed directly to the MLA attention kernel (not via a
    quantized linear op), so it is NOT processed by the FineGrainedFP8LinearQuantization
    transform. Without this hook, the FP8 weight is loaded via a raw dtype cast that
    ignores weight_scale_inv, producing values ~1000x too large and NaN/Inf attention.

    This hook performs proper block-wise dequantization before weights are loaded.

    Args:
        num_layers: Must match ``config.num_hidden_layers`` for the decoder (same source
            as :func:`_rope_deinterleave_load_hook`). Do not infer this from checkpoint
            keys: layer count belongs to model config, not the state dict.
    """
    for layer_idx in range(num_layers):
        layer_prefix = f"{prefix}model.layers.{layer_idx}.self_attn."
        w_key = layer_prefix + "kv_b_proj.weight"
        scale_key = layer_prefix + "kv_b_proj.weight_scale_inv"

        if w_key not in state_dict:
            continue

        w = state_dict[w_key]
        if w.dtype != torch.float8_e4m3fn:
            # Already in a floating-point type; no dequantization needed.
            continue

        if scale_key not in state_dict:
            raise KeyError(
                f"Missing {scale_key} for FP8 weight {w_key}; cannot dequantize kv_b_proj."
            )

        scale = state_dict[scale_key]

        # Dequant shared with ``tensorrt_llm.quantization.utils.fp8_matrix_weight_dequant``.
        state_dict[w_key] = dequant_fp8_nk_weight_auto_scale_layout(
            w, scale, dtype=torch.bfloat16, block_k=128
        )

        # Remove scale from state_dict so it is not loaded into a non-existent buffer.
        del state_dict[scale_key]
