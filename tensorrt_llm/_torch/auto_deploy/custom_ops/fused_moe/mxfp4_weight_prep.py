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
    is_scale: bool = False,
) -> torch.Tensor:
    """Apply the gated-GEMM shuffle (used for both w3/w1 weight and its scale).

    For scales (``is_scale=True``), additionally apply
    ``torch.ops.trtllm.block_scale_interleave`` after shuffling — PT's
    ``MXFP4WeightTRTLLMGenFusedMoEMethod.load_expert_w3_w1_weight_scale_mxfp4``
    (quantization.py:4382) does both steps; the kernel reads scales in this
    interleaved layout. Without it the dequantization scaling is wrong and
    output logits are garbage.

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
        if is_scale:
            shuffled = torch.ops.trtllm.block_scale_interleave(shuffled).reshape(slc.shape)
        out.append(shuffled.view(slc.dtype))
    return torch.stack(out, dim=0).contiguous()


def _shuffle_per_expert_w2(
    stacked: torch.Tensor,  # [E, H_pad, X]   uint8  (X = I_pad/2 or I_pad/32)
    num_elts_per_sf: int | None = None,
    is_scale: bool = False,
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
        if is_scale:
            shuffled = torch.ops.trtllm.block_scale_interleave(shuffled).reshape(slc.shape)
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
    tp_rank: int = 0,
) -> PreparedMXFP4Weights:
    """Convert HF on-disk MXFP4 expert weights into trtllm-gen-ready stacked tensors.

    Mirrors the algorithm in
    ``MXFP4WeightTRTLLMGenFusedMoEMethod.{post_load_weights,
    load_expert_w3_w1_weight, load_expert_w2_weight,
    load_expert_w3_w1_weight_scale_mxfp4, load_expert_w2_weight_scale_mxfp4}``.

    For ``tp_size > 1`` (TP-MoE / V6, Step 5 of MOE_TRTLLM_GEN_PLAN.md):
    intermediate dim is sharded across ``tp_size`` ranks before the kernel-
    layout pad+shuffle.  PT does this in
    ``load_expert_w3_w1_weight`` / ``load_expert_w2_weight`` via
    ``load_weight_shard(..., COLUMN/ROW)`` after a TP-aware pre-pad
    (``alignment = _get_weight_alignment(weight_alignment, scaling_vector_size,
    tp_size, I)``).  We replicate that here in three steps:
      1. derive ``alignment_tp`` so ``alignment_tp / tp_size`` is
         128-aligned — guarantees per-rank ``I/tp`` is itself 128-aligned
         after the pre-pad, which is what TMA + cubin coverage need.
      2. pre-pad each half (gate / up / scales / biases / down) on the
         intermediate axis to ``alignment_tp``.
      3. slice the intermediate axis to this rank's range
         ``[tp_rank * (alignment_tp / tp_size) : (tp_rank+1) * ...]``.
    The downstream pad+shuffle then operates on per-rank tensors with
    intermediate dim ``alignment_tp / tp_size`` (= 384 for gpt-oss at
    tp=8).  ``valid_intermediate`` is clamped to ``min(intermediate_size,
    slice_stop) - slice_start`` so the kernel hint reflects the unpadded
    portion of this rank's slice (matches PT's
    ``intermediate_size_per_partition_lean``).

    EP (expert dim slicing) is NOT done here — the transform handles EP by
    selecting the expert subset before calling this helper.
    """
    if tp_size > 1 and intermediate_size % tp_size != 0:
        raise ValueError(
            f"intermediate_size ({intermediate_size}) must be divisible by "
            f"tp_size ({tp_size}) for TP-MoE."
        )
    if tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(f"tp_rank {tp_rank} out of range for tp_size {tp_size}")

    e = gate_up_blocks.size(0)
    assert down_blocks.size(0) == e

    # 1. Reshape blocks to 3-D (collapse the inner [..., 16] dim).
    gu_3d = _flatten_block_dim(gate_up_blocks)  # [E, 2I, H/2]
    dn_3d = _flatten_block_dim(down_blocks)  # [E, H,  I/2]

    # 1a. De-interleave w1 (gate) and w3 (up) into SEPARATE per-half tensors.
    #
    # Rationale: HF's gpt-oss-120b dense ``gate_up_proj`` is ``[E, H, 2I]``
    # with gate at even indices and up at odd indices on the last dim
    # (``torch_moe_dense_mlp`` line ~765 splits via
    # ``gate, up = gate_up[..., ::2], gate_up[..., 1::2]``).
    # The MXFP4 quantization moves that 2I axis to dim 1, preserving the
    # *interleaving* across rows: row 0 = gate0, row 1 = up0, row 2 = gate1,
    # row 3 = up1, ...
    #
    # PT's ``MXFP4WeightTRTLLMGenFusedMoEMethod.load_expert_w3_w1_weight``
    # writes a separated layout (see line 4256 in quantization.py):
    #     dst_w3_weight, dst_w1_weight = dst_w3_w1_weight.chunk(2, dim=0)
    #     dst_w3_weight.copy_(w3_weight); dst_w1_weight.copy_(w1_weight)
    # So the trtllm-gen kernel expects rows 0..I_pad-1 = w3 (up),
    # I_pad..2*I_pad-1 = w1 (gate). We keep up and gate separate through the
    # row-padding so the zero-pad rows go INSIDE each half (not at the very
    # end), then stack as [up | gate] before col-padding and shuffling.
    gate_rows_w = gu_3d[:, 0::2, :].contiguous()  # [E, I, H/2]
    up_rows_w = gu_3d[:, 1::2, :].contiguous()  # [E, I, H/2]
    gate_rows_s = gate_up_scales[:, 0::2, :].contiguous()  # [E, I, H/32]
    up_rows_s = gate_up_scales[:, 1::2, :].contiguous()  # [E, I, H/32]
    gate_b = gate_up_bias[:, 0::2].contiguous()  # [E, I]
    up_b = gate_up_bias[:, 1::2].contiguous()  # [E, I]

    # 1b. TP slicing on the intermediate axis (when tp_size > 1).
    #
    # PT (quantization.py:4221-4234) computes a TP-aware alignment first so
    # that ``per_shard = padded_I / tp_size`` is itself a multiple of
    # ``weight_alignment`` (kernel TMA constraint).  For gpt-oss-120b
    # I=2880 at tp=8: ``_get_weight_alignment(128, 32, 8, 2880) = 3072``
    # so each rank holds ``3072/8 = 384`` intermediate elements (= 128*3).
    # The PRE-pad happens before sharding so scaling-factor blocks (32
    # elements each) don't straddle rank boundaries.
    if tp_size > 1:
        alignment_tp = _get_weight_alignment(
            _WEIGHT_ALIGNMENT, _MXFP4_SCALING_VECTOR_SIZE, tp_size, intermediate_size
        )
        # Pad intermediate axis to ``alignment_tp`` BEFORE sharding (PT pads-
        # before-shard semantics; quantization.py:4211-4220 explains why).
        i_padded_tp = ((intermediate_size + alignment_tp - 1) // alignment_tp) * alignment_tp
        per_rank_i = i_padded_tp // tp_size  # = 384 for gpt-oss tp=8
        slice_start = tp_rank * per_rank_i
        slice_stop = (tp_rank + 1) * per_rank_i
        # ``valid_intermediate`` per rank: clamp to original ``intermediate_size``.
        valid_intermediate = max(0, min(intermediate_size, slice_stop) - slice_start)

        def _pad_int_axis(t: torch.Tensor, dim: int, target: int) -> torch.Tensor:
            cur = t.shape[dim]
            if cur >= target:
                return t
            pad_amount = target - cur
            # F.pad pad spec is reversed-axis order; build dynamically.
            pad = [0, 0] * (t.dim() - dim - 1) + [0, pad_amount] + [0, 0] * dim
            return torch.nn.functional.pad(t, pad)

        # Pad I axis (rows) of gate / up to i_padded_tp, then slice this
        # rank's chunk.  Same for the scale tensors (rows) and biases.
        gate_rows_w = _pad_int_axis(gate_rows_w, 1, i_padded_tp)[
            :, slice_start:slice_stop, :
        ].contiguous()
        up_rows_w = _pad_int_axis(up_rows_w, 1, i_padded_tp)[
            :, slice_start:slice_stop, :
        ].contiguous()
        gate_rows_s = _pad_int_axis(gate_rows_s, 1, i_padded_tp)[
            :, slice_start:slice_stop, :
        ].contiguous()
        up_rows_s = _pad_int_axis(up_rows_s, 1, i_padded_tp)[
            :, slice_start:slice_stop, :
        ].contiguous()
        gate_b = _pad_int_axis(gate_b, 1, i_padded_tp)[:, slice_start:slice_stop].contiguous()
        up_b = _pad_int_axis(up_b, 1, i_padded_tp)[:, slice_start:slice_stop].contiguous()

        # down (dn_3d): cols = I/2.  Pad to i_padded_tp/2 then slice.
        dn_3d = _pad_int_axis(dn_3d, 2, i_padded_tp // 2)[
            :, :, slice_start // 2 : slice_stop // 2
        ].contiguous()
        # down scales: cols = I/scaling_vector_size.
        sf_per_rank_start = slice_start // _MXFP4_SCALING_VECTOR_SIZE
        sf_per_rank_stop = slice_stop // _MXFP4_SCALING_VECTOR_SIZE
        sf_padded = i_padded_tp // _MXFP4_SCALING_VECTOR_SIZE
        down_scales = _pad_int_axis(down_scales, 2, sf_padded)[
            :, :, sf_per_rank_start:sf_per_rank_stop
        ].contiguous()

        # The downstream pad+shuffle now treats ``per_rank_i`` as the local
        # intermediate dim.  Reuse the existing variable name so the rest
        # of the function is unchanged.
        intermediate_size_for_local = per_rank_i
    else:
        intermediate_size_for_local = intermediate_size
        valid_intermediate = intermediate_size

    # 2. Determine per-rank dims.
    valid_hidden = hidden_size

    # 3. Pad weights.
    #
    # PT pads on the per-expert *I* (intermediate, line 3712-3713 in
    # quantization.py) and *H* (hidden, line 3715/3717) before constructing
    # the buffer shape — the ``2I`` row dim of w1 is then ``I_pad * 2``,
    # NOT ``round_up(2I, weight_alignment)``.
    #
    # That distinction matters for gpt-oss-120b: I=2880 is not 128-aligned
    # (2880 % 128 = 64), so PT's I_pad = 2944. w1's 2I row dim therefore
    # becomes 5888 (= 2*2944), and w2's I/2 col dim becomes 1472 (= 2944/2).
    # Both reflect the same I_pad — kernel sees a consistent intermediate
    # dim. If we instead pad ``2I = 5760`` directly, weight_alignment=128
    # leaves 5760 unchanged (already 128-aligned), so w1's I_pad stays at
    # 2880 while w2's I_pad jumps to 2944 from the col padding. The kernel
    # then mixes 2880 (w1) and 2944 (w2) for the *same* intermediate dim and
    # the autotune cubin lookup finds no config.
    #
    # Same idea for the hidden axis: PT pads w1.K to 512 (input_hidden_align)
    # and w2.N to 128 (weight_align). H=2880 → w1.K=3072, w2.N=2944.
    # We replicate that exactly so the kernel's args.hidden_size /
    # output_hidden_size match what PT's ``MXFP4WeightTRTLLMGenFusedMoEMethod``
    # exercises.
    intermediate_size_pad = (
        (intermediate_size_for_local + _WEIGHT_ALIGNMENT - 1) // _WEIGHT_ALIGNMENT
    ) * _WEIGHT_ALIGNMENT
    hidden_w1_pad = (
        (hidden_size + _INPUT_HIDDEN_ALIGNMENT - 1) // _INPUT_HIDDEN_ALIGNMENT
    ) * _INPUT_HIDDEN_ALIGNMENT
    hidden_w2_pad = ((hidden_size + _WEIGHT_ALIGNMENT - 1) // _WEIGHT_ALIGNMENT) * _WEIGHT_ALIGNMENT

    # gate_up weights — pad each half [E, I, H/2] to [E, I_pad, H_w1_pad/2]
    # SEPARATELY so the zero-pad rows live inside each half, then stack as
    # [up | gate]. PT's gpt-oss loader (modeling_gpt_oss.py:695-706 +
    # quantization.py:4252-4258) ends up with ``dst_w3 = up`` in the first
    # half and ``dst_w1 = gate`` in the second half via this exact
    # de-interleave + chunk dance.
    up_padded_w = _pad_per_expert_2d(up_rows_w, hidden_w1_pad // 2, intermediate_size_pad)
    gate_padded_w = _pad_per_expert_2d(gate_rows_w, hidden_w1_pad // 2, intermediate_size_pad)
    gu_padded = torch.cat(
        [up_padded_w, gate_padded_w], dim=1
    ).contiguous()  # [E, 2I_pad, H_w1_pad/2]

    # down: rows = H, cols = I/2. Target shape [E, H_w2_pad, I_pad/2 = 1472].
    # PT pads w2's I/2 axis to ``alignment // 2`` where alignment=128,
    # giving 64-multiple (quantization.py:4287). For I/2=1440 → 1472.
    # The kernel then asserts ``gemm2_weights.shape[2] == intermediate_size / 2``,
    # so I_pad_w2 must match I_pad_w1 (both 2944).
    dn_padded = _pad_per_expert_2d(dn_3d, intermediate_size_pad // 2, hidden_w2_pad)

    # 4. Pad scales — same per-half logic for w1; col_alignment uses
    #    scaling-vector size.
    up_padded_s = _pad_per_expert_2d(
        up_rows_s, hidden_w1_pad // _MXFP4_SCALING_VECTOR_SIZE, intermediate_size_pad
    )
    gate_padded_s = _pad_per_expert_2d(
        gate_rows_s, hidden_w1_pad // _MXFP4_SCALING_VECTOR_SIZE, intermediate_size_pad
    )
    gu_scale_padded = torch.cat([up_padded_s, gate_padded_s], dim=1).contiguous()
    dn_scale_padded = _pad_per_expert_2d(
        down_scales,
        intermediate_size_pad // _MXFP4_SCALING_VECTOR_SIZE,
        hidden_w2_pad,
    )

    # 5. Shuffle weights + scales for the kernel's TMA layout.
    fc1_weights = _shuffle_per_expert_w3_w1(gu_padded)
    fc1_weights_scale = _shuffle_per_expert_w3_w1(
        gu_scale_padded, num_elts_per_sf=_MXFP4_SCALING_VECTOR_SIZE, is_scale=True
    )
    fc2_weights = _shuffle_per_expert_w2(dn_padded)
    fc2_weights_scale = _shuffle_per_expert_w2(
        dn_scale_padded, num_elts_per_sf=_MXFP4_SCALING_VECTOR_SIZE, is_scale=True
    )

    # 6. Bias: convert to float32. For w2, divide by tp_size (no-op at tp=1).
    #    Pad each half separately so the [up | gate] split matches the
    #    weights' row layout.
    up_bias_padded = (
        _pad_per_expert_2d(
            up_b.unsqueeze(-1),  # [E, I, 1]
            col_alignment=1,
            row_alignment=intermediate_size_pad,
        )
        .squeeze(-1)
        .float()
        .contiguous()
    )  # [E, I_pad] float32
    gate_bias_padded = (
        _pad_per_expert_2d(
            gate_b.unsqueeze(-1),
            col_alignment=1,
            row_alignment=intermediate_size_pad,
        )
        .squeeze(-1)
        .float()
        .contiguous()
    )
    fc1_bias_padded = torch.cat(
        [up_bias_padded, gate_bias_padded], dim=1
    ).contiguous()  # [E, 2I_pad]

    fc2_bias_padded = (
        _pad_per_expert_2d(
            down_bias.unsqueeze(-1),
            col_alignment=1,
            row_alignment=hidden_w2_pad,
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
