# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""MXFP4 weight prep for TRT-LLM-Gen `bf16_mxe2m1_block_scale_moe_runner`.

This produces the kernel-ready stacked tensors that
``auto_deploy::trtllm_mxfp4_w4a16_moe_fused`` expects, starting from the
HuggingFace on-disk MXFP4 layout that the existing AutoDeploy
``quantize_mxfp4_moe`` transform registers.

Layout notes:

* HF on-disk:
    gate_up_proj_blocks  : ``[E, 2I, H/32, 16]`` ``uint8`` (= ``[E, 2I, H/2]`` flattened)
    gate_up_proj_scales  : ``[E, 2I, H/32]``     ``uint8`` (UE8M0)
    gate_up_proj_bias    : ``[E, 2I]``           ``bfloat16``
    down_proj_blocks     : ``[E, H, I/32, 16]``  ``uint8``
    down_proj_scales     : ``[E, H, I/32]``      ``uint8``
    down_proj_bias       : ``[E, H]``            ``bfloat16``

* What the trtllm-gen kernel expects:
    gemm1_weights        : ``[E_local, 2I_pad, H_pad/2]``    ``uint8`` (col-parallel for w1/w3)
    gemm1_weights_scale  : ``[E_local, 2I_pad, H_pad/32]``   ``uint8``
    gemm1_bias           : ``[E_local, 2I_pad]``             ``float32``
    gemm2_weights        : ``[E_local, H_pad, I_pad/2]``     ``uint8`` (row-parallel for w2)
    gemm2_weights_scale  : ``[E_local, H_pad, I_pad/32]``    ``uint8``
    gemm2_bias           : ``[E_local, H_pad]``              ``float32`` (divided by tp_size)

  All weights / scales additionally go through
  ``torch.ops.trtllm.shuffle_matrix`` so the kernel can hit its TMA layout.

This module mirrors PT's ``MXFP4WeightTRTLLMGenFusedMoEMethod``
(`tensorrt_llm/_torch/modules/fused_moe/quantization.py:4135`).
The PT helpers are reused via direct import to keep the algorithm
byte-identical:

* ``maybe_pad_for_mxfp4``                           — alignment padding
* ``trtllmgen_maybe_get_cached_w3_w1_permute_indices`` — gated GEMM shuffle
* ``trtllmgen_maybe_get_cached_w2_permute_indices`` — non-gated GEMM shuffle
* ``_get_weight_alignment``                         — alignment derivation

The first version of this helper (Step 2 of the V4 plan) supports
``tp_size = 1`` only; TP slicing is added in Step 5 alongside a new
``ShardingInfo``.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from tensorrt_llm._torch.modules.fused_moe.quantization import (
    _get_weight_alignment,
    maybe_pad_for_mxfp4,
    trtllmgen_maybe_get_cached_w2_permute_indices,
    trtllmgen_maybe_get_cached_w3_w1_permute_indices,
)

# Cache permute indices to avoid recomputation across calls.
# Keyed by (shape, role, num_elts_per_sf) inside the PT helpers.
_PERMUTE_CACHE: Dict = {}

# MXFP4 block size (UE8M0 scale per 32 elements). Matches HF gpt-oss layout.
_MXFP4_SCALING_VECTOR_SIZE: int = 32

# Kernel layout constants (mirror PT's MXFP4WeightTRTLLMGenFusedMoEMethod).
_INPUT_HIDDEN_ALIGNMENT: int = 512
_WEIGHT_ALIGNMENT: int = 128
_EPILOGUE_TILE_M: int = 128


@dataclass(frozen=True)
class PreparedMXFP4Weights:
    """Output of :func:`prepare_mxfp4_weights_for_trtllm_gen`."""

    fc1_weights_mxfp4: torch.Tensor  # [E, 2I_pad, H_pad/2]  uint8 (shuffled)
    fc1_weights_scale_ue8m0: torch.Tensor  # [E, 2I_pad, H_pad/32] uint8 (shuffled)
    fc1_bias_f32: torch.Tensor  # [E, 2I_pad]         float32
    fc2_weights_mxfp4: torch.Tensor  # [E, H_pad, I_pad/2]   uint8 (shuffled)
    fc2_weights_scale_ue8m0: torch.Tensor  # [E, H_pad, I_pad/32]  uint8 (shuffled)
    fc2_bias_f32: torch.Tensor  # [E, H_pad]          float32 (already /tp_size)
    valid_hidden_size: int  # original H
    valid_intermediate_size: int  # per-rank intermediate size (lean shape)
    intermediate_size_padded: int  # I_pad (per-rank, after pad)
    hidden_size_padded: int  # H_pad


def _flatten_block_dim(blocks_4d: torch.Tensor) -> torch.Tensor:
    """Collapse ``[..., n_blocks, 16]`` -> ``[..., n_blocks * 16]`` (= H/2 or I/2)."""
    if blocks_4d.dim() == 3:
        return blocks_4d
    if blocks_4d.dim() == 4:
        return blocks_4d.contiguous().view(*blocks_4d.shape[:-2], -1)
    raise ValueError(f"Unexpected MXFP4 weight rank {blocks_4d.dim()}; expected 3 or 4.")


def _pad_per_expert_2d(
    weight_3d: torch.Tensor,  # [E, R, C]
    col_alignment: int,
    row_alignment: int,
) -> torch.Tensor:
    """Pad each expert's 2-D matrix to the given row/col alignment."""
    e = weight_3d.size(0)
    out = []
    for i in range(e):
        out.append(maybe_pad_for_mxfp4(weight_3d[i], col_alignment, row_alignment))
    return torch.stack(out, dim=0).contiguous()


def _shuffle_per_expert_w3_w1(
    stacked: torch.Tensor,  # [E, 2I_pad, X]   uint8  (X = H_pad/2 or H_pad/32)
    num_elts_per_sf: int | None = None,
) -> torch.Tensor:
    """Apply the gated-GEMM shuffle (used for both w3/w1 weight and its scale).

    Looping over experts because the PT permute-index helpers compute indices
    from a 2-D shape; applying them slice-by-slice avoids ambiguity at the
    leading expert dim.
    """
    e = stacked.size(0)
    out = []
    for i in range(e):
        slc = stacked[i].contiguous()
        perm = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            slc,
            _PERMUTE_CACHE,
            _EPILOGUE_TILE_M,
            num_elts_per_sf=num_elts_per_sf,
        )
        shuffled = torch.ops.trtllm.shuffle_matrix(slc, perm.to(slc.device))
        out.append(shuffled.view(slc.dtype))
    return torch.stack(out, dim=0).contiguous()


def _shuffle_per_expert_w2(
    stacked: torch.Tensor,  # [E, H_pad, X]   uint8  (X = I_pad/2 or I_pad/32)
    num_elts_per_sf: int | None = None,
) -> torch.Tensor:
    e = stacked.size(0)
    out = []
    for i in range(e):
        slc = stacked[i].contiguous()
        perm = trtllmgen_maybe_get_cached_w2_permute_indices(
            slc,
            _PERMUTE_CACHE,
            _EPILOGUE_TILE_M,
            num_elts_per_sf=num_elts_per_sf,
        )
        shuffled = torch.ops.trtllm.shuffle_matrix(slc, perm.to(slc.device))
        out.append(shuffled.view(slc.dtype))
    return torch.stack(out, dim=0).contiguous()


def prepare_mxfp4_weights_for_trtllm_gen(
    gate_up_blocks: torch.Tensor,  # [E, 2I, H/32, 16]  or  [E, 2I, H/2]   uint8
    gate_up_scales: torch.Tensor,  # [E, 2I, H/32]                         uint8
    gate_up_bias: torch.Tensor,  # [E, 2I]                               bf16
    down_blocks: torch.Tensor,  # [E, H, I/32, 16]   or  [E, H, I/2]    uint8
    down_scales: torch.Tensor,  # [E, H, I/32]                          uint8
    down_bias: torch.Tensor,  # [E, H]                                bf16
    *,
    hidden_size: int,
    intermediate_size: int,
    tp_size: int = 1,
) -> PreparedMXFP4Weights:
    """Convert HF on-disk MXFP4 expert weights into trtllm-gen-ready stacked tensors.

    Mirrors the algorithm in
    ``MXFP4WeightTRTLLMGenFusedMoEMethod.{post_load_weights,
    load_expert_w3_w1_weight, load_expert_w2_weight,
    load_expert_w3_w1_weight_scale_mxfp4, load_expert_w2_weight_scale_mxfp4}``.

    Step-2 scope: ``tp_size = 1`` only. TP slicing arrives in Step 5.
    """
    if tp_size != 1:
        raise NotImplementedError(
            "TP > 1 is added in step 5 (MXFP4TRTLLMGenSharding). "
            "Use single-GPU first to validate steps 1–3."
        )

    e = gate_up_blocks.size(0)
    assert down_blocks.size(0) == e

    # 1. Reshape blocks to 3-D (collapse the inner [..., 16] dim).
    gu_3d = _flatten_block_dim(gate_up_blocks)  # [E, 2I, H/2]
    dn_3d = _flatten_block_dim(down_blocks)  # [E, H,  I/2]

    # 2. Determine per-rank dims (no shard at tp=1).
    valid_hidden = hidden_size
    valid_intermediate = intermediate_size

    # 3. Pad weights.
    #    PT alignment derivation (quantization.py:4221) is per-rank;
    #    at tp=1 it reduces to max(weight_alignment, scaling_vector_size).
    weight_align_w1 = _get_weight_alignment(
        _WEIGHT_ALIGNMENT,
        _MXFP4_SCALING_VECTOR_SIZE,
        tp_size,
        gu_3d.shape[1],  # 2I
    )
    # gate_up: cols = H/2 (need pad to input_hidden_alignment//2),
    #          rows = 2I (need pad to weight_align_w1)
    gu_padded = _pad_per_expert_2d(gu_3d, _INPUT_HIDDEN_ALIGNMENT // 2, weight_align_w1)

    # down: cols = I/2 (need pad to weight_alignment//2),
    #       rows = H   (need pad to weight_alignment)
    dn_padded = _pad_per_expert_2d(dn_3d, _WEIGHT_ALIGNMENT // 2, _WEIGHT_ALIGNMENT)

    # 4. Pad scales (col_alignment uses scaling-vector size).
    gu_scale_padded = _pad_per_expert_2d(
        gate_up_scales,
        _INPUT_HIDDEN_ALIGNMENT // _MXFP4_SCALING_VECTOR_SIZE,
        weight_align_w1,
    )
    dn_scale_padded = _pad_per_expert_2d(
        down_scales,
        _WEIGHT_ALIGNMENT // _MXFP4_SCALING_VECTOR_SIZE,
        _WEIGHT_ALIGNMENT,
    )

    # 5. Shuffle weights + scales for the kernel's TMA layout.
    fc1_weights = _shuffle_per_expert_w3_w1(gu_padded)
    fc1_weights_scale = _shuffle_per_expert_w3_w1(
        gu_scale_padded, num_elts_per_sf=_MXFP4_SCALING_VECTOR_SIZE
    )
    fc2_weights = _shuffle_per_expert_w2(dn_padded)
    fc2_weights_scale = _shuffle_per_expert_w2(
        dn_scale_padded, num_elts_per_sf=_MXFP4_SCALING_VECTOR_SIZE
    )

    # 6. Bias: convert to float32. For w2, divide by tp_size (no-op at tp=1).
    #    Pad to the same row count as the weights.
    fc1_bias_padded = (
        _pad_per_expert_2d(
            gate_up_bias.unsqueeze(-1),  # [E, 2I, 1]
            col_alignment=1,
            row_alignment=weight_align_w1,
        )
        .squeeze(-1)
        .float()
        .contiguous()
    )  # [E, 2I_pad] float32

    fc2_bias_padded = (
        _pad_per_expert_2d(
            down_bias.unsqueeze(-1),
            col_alignment=1,
            row_alignment=_WEIGHT_ALIGNMENT,
        )
        .squeeze(-1)
        .float()
        .contiguous()
    )  # [E, H_pad] float32
    if tp_size > 1:
        fc2_bias_padded = fc2_bias_padded / tp_size

    intermediate_size_padded = fc1_weights.shape[1] // 2  # 2I_pad / 2 = I_pad
    hidden_size_padded = fc1_weights.shape[-1] * 2  # (H_pad/2) * 2 = H_pad

    return PreparedMXFP4Weights(
        fc1_weights_mxfp4=fc1_weights,
        fc1_weights_scale_ue8m0=fc1_weights_scale,
        fc1_bias_f32=fc1_bias_padded,
        fc2_weights_mxfp4=fc2_weights,
        fc2_weights_scale_ue8m0=fc2_weights_scale,
        fc2_bias_f32=fc2_bias_padded,
        valid_hidden_size=valid_hidden,
        valid_intermediate_size=valid_intermediate,
        intermediate_size_padded=intermediate_size_padded,
        hidden_size_padded=hidden_size_padded,
    )


def make_swiglu_param_tensors(
    num_local_experts: int,
    *,
    alpha: float = 1.702,
    beta: float = 1.0,
    limit: float = 7.0,
    device: torch.device | str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the per-expert SwiGLU-bias parameter triple expected by the kernel.

    For gpt-oss-120b: alpha=1.702, beta=1.0, limit=7.0 (constants embedded in the
    HF model config).
    """
    dev = torch.device(device) if device is not None else None
    a = torch.full((num_local_experts,), alpha, dtype=torch.float32, device=dev)
    b = torch.full((num_local_experts,), beta, dtype=torch.float32, device=dev)
    c = torch.full((num_local_experts,), limit, dtype=torch.float32, device=dev)
    return a, b, c
