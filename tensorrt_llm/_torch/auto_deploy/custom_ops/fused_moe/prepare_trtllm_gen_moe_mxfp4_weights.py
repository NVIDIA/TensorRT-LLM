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


def _compute_padded_dims(per_rank_i: int, hidden_size: int) -> Tuple[int, int, int]:
    """Returns ``(i_pad, h_w1_pad, h_w2_pad)`` for the trtllm-gen layout.

    ``i_pad`` aligns the per-rank intermediate dim to ``_WEIGHT_ALIGNMENT``
    (128, TMA weight alignment). ``h_w1_pad`` aligns hidden to
    ``_INPUT_HIDDEN_ALIGNMENT`` (512, TMA input constraint) for w1's K-axis.
    ``h_w2_pad`` aligns hidden to ``_WEIGHT_ALIGNMENT`` (128) for w2's
    weight N-axis. Used by :class:`MXFP4PrepScratch.allocate` and the
    main prep helper so all sites share one ceiling formula.
    """
    i_pad = ((per_rank_i + _WEIGHT_ALIGNMENT - 1) // _WEIGHT_ALIGNMENT) * _WEIGHT_ALIGNMENT
    h_w1_pad = (
        (hidden_size + _INPUT_HIDDEN_ALIGNMENT - 1) // _INPUT_HIDDEN_ALIGNMENT
    ) * _INPUT_HIDDEN_ALIGNMENT
    h_w2_pad = ((hidden_size + _WEIGHT_ALIGNMENT - 1) // _WEIGHT_ALIGNMENT) * _WEIGHT_ALIGNMENT
    return i_pad, h_w1_pad, h_w2_pad


@dataclass(frozen=True)
class TRTLLMGenMXFP4MoEWeights:
    """Output of :func:`prepare_trtllm_gen_moe_mxfp4_weights`."""

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


@dataclass
class MXFP4PrepScratch:
    """Reusable GPU scratch buffers for ``prepare_trtllm_gen_moe_mxfp4_weights``.

    Use :meth:`allocate` to pre-allocate once for the per-rank kernel-layout
    shape; pass to :func:`prepare_trtllm_gen_moe_mxfp4_weights` via the
    ``scratch=`` kwarg on every MoE layer in a build/fuse pass. The helper
    writes its pad + shuffle outputs into these buffers in-place, so no
    transient pad/shuffle tensors accumulate or are freed per layer.

    Caller MUST ``.clone()`` (or ``.data.copy_()`` into a fresh nn.Parameter)
    the relevant buffer fields *before the next layer's prep call*; otherwise
    the next call's writes overwrite the previous layer's data. The intended
    usage in :class:`FuseMXFP4Moe` is to pre-allocate the destination
    ``nn.Parameter`` storage for every MoE layer *first* (so all prepared
    blocks are placed contiguously in allocator order, with no transients
    interleaved), then per layer run prep with scratch and ``copy_`` from
    scratch into the pre-allocated parameter storage.

    All buffers are sized for ONE MoE layer's per-rank shape. gpt-oss has
    a cross-layer consistency guarantee (all MoE layers share H/I/E) so
    one scratch is sufficient for every layer in the model.

    Fields ending in ``_pad_buf`` hold pad outputs (post pad, pre shuffle);
    fields without that suffix hold shuffle outputs (kernel-ready layout).
    Two separate buffers per kind because ``trtllm.shuffle_matrix`` reads
    from one tensor and writes a new one — it cannot operate in-place.
    """

    # Shuffle outputs (= kernel-ready layout; what the prepared nn.Parameter
    # will hold).
    fc1_w_buf: torch.Tensor  # [E_local, 2I_pad, H_w1_pad/2]   uint8
    fc1_s_buf: torch.Tensor  # [E_local, 2I_pad, H_w1_pad/32]  uint8
    fc1_b_buf: torch.Tensor  # [E_local, 2I_pad]                fp32
    fc2_w_buf: torch.Tensor  # [E_local, H_w2_pad, I_pad/2]    uint8
    fc2_s_buf: torch.Tensor  # [E_local, H_w2_pad, I_pad/32]   uint8
    fc2_b_buf: torch.Tensor  # [E_local, H_w2_pad]              fp32

    # Pad outputs (post pad, pre shuffle). Same shape as the corresponding
    # shuffle output above.
    fc1_w_pad_buf: torch.Tensor
    fc1_s_pad_buf: torch.Tensor
    fc1_b_pad_buf: torch.Tensor
    fc2_w_pad_buf: torch.Tensor
    fc2_s_pad_buf: torch.Tensor
    fc2_b_pad_buf: torch.Tensor

    # Cached layout dimensions (for shape validation on subsequent calls).
    e_local: int
    hidden_size: int
    per_rank_i: int
    i_pad: int
    h_w1_pad: int
    h_w2_pad: int

    @classmethod
    def allocate(
        cls,
        *,
        e_local: int,
        per_rank_i: int,
        hidden_size: int,
        device: torch.device | str,
    ) -> "MXFP4PrepScratch":
        """Allocate the scratch buffers for one MoE layer's per-rank shape.

        ``e_local`` is the per-rank expert count, ``per_rank_i`` is the
        intermediate dim already TP-sliced (or full ``I`` if no TP), and
        ``hidden_size`` is the model's hidden dim ``H``.
        """
        i_pad, h_w1_pad, h_w2_pad = _compute_padded_dims(per_rank_i, hidden_size)
        u8 = dict(dtype=torch.uint8, device=device)
        f32 = dict(dtype=torch.float32, device=device)
        return cls(
            fc1_w_buf=torch.empty(e_local, 2 * i_pad, h_w1_pad // 2, **u8),
            fc1_s_buf=torch.empty(e_local, 2 * i_pad, h_w1_pad // _MXFP4_SCALING_VECTOR_SIZE, **u8),
            fc1_b_buf=torch.empty(e_local, 2 * i_pad, **f32),
            fc2_w_buf=torch.empty(e_local, h_w2_pad, i_pad // 2, **u8),
            fc2_s_buf=torch.empty(e_local, h_w2_pad, i_pad // _MXFP4_SCALING_VECTOR_SIZE, **u8),
            fc2_b_buf=torch.empty(e_local, h_w2_pad, **f32),
            fc1_w_pad_buf=torch.empty(e_local, 2 * i_pad, h_w1_pad // 2, **u8),
            fc1_s_pad_buf=torch.empty(
                e_local, 2 * i_pad, h_w1_pad // _MXFP4_SCALING_VECTOR_SIZE, **u8
            ),
            fc1_b_pad_buf=torch.empty(e_local, 2 * i_pad, **f32),
            fc2_w_pad_buf=torch.empty(e_local, h_w2_pad, i_pad // 2, **u8),
            fc2_s_pad_buf=torch.empty(e_local, h_w2_pad, i_pad // _MXFP4_SCALING_VECTOR_SIZE, **u8),
            fc2_b_pad_buf=torch.empty(e_local, h_w2_pad, **f32),
            e_local=e_local,
            hidden_size=hidden_size,
            per_rank_i=per_rank_i,
            i_pad=i_pad,
            h_w1_pad=h_w1_pad,
            h_w2_pad=h_w2_pad,
        )


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
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pad each expert's 2-D matrix to the given row/col alignment.

    When ``out`` is provided, write each expert's padded matrix into
    ``out[i]`` in-place (no per-expert allocation accumulated, no final
    ``torch.stack`` allocation). Backward-compatible with the
    ``out=None`` path that builds + stacks a fresh tensor.
    """
    e = weight_3d.size(0)
    if out is None:
        out_list = []
        for i in range(e):
            out_list.append(maybe_pad_for_mxfp4(weight_3d[i], col_alignment, row_alignment))
        return torch.stack(out_list, dim=0).contiguous()

    assert out.shape[0] == e, f"out leading dim {out.shape[0]} != e {e}"
    for i in range(e):
        padded = maybe_pad_for_mxfp4(weight_3d[i], col_alignment, row_alignment)
        out[i].copy_(padded)
    return out


def _shuffle_per_expert_w3_w1(
    stacked: torch.Tensor,  # [E, 2I_pad, X]   uint8  (X = H_pad/2 or H_pad/32)
    num_elts_per_sf: int | None = None,
    is_scale: bool = False,
    *,
    out: torch.Tensor | None = None,
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

    When ``out`` is provided, per-expert shuffle results are copied into
    ``out[i]`` in-place — the per-iter shuffle alloc still happens (the
    ``trtllm.shuffle_matrix`` CUDA op returns its own tensor) but it is
    freed immediately after the copy, so no per-expert tensors accumulate
    in a list and no final ``torch.stack`` allocation is required.
    """
    e = stacked.size(0)
    if out is None:
        out_list = []
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
            out_list.append(shuffled.view(slc.dtype))
        return torch.stack(out_list, dim=0).contiguous()

    assert out.shape[0] == e
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
        out[i].copy_(shuffled.view(slc.dtype))
    return out


def _shuffle_per_expert_w2(
    stacked: torch.Tensor,  # [E, H_pad, X]   uint8  (X = I_pad/2 or I_pad/32)
    num_elts_per_sf: int | None = None,
    is_scale: bool = False,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    e = stacked.size(0)
    if out is None:
        out_list = []
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
            out_list.append(shuffled.view(slc.dtype))
        return torch.stack(out_list, dim=0).contiguous()

    assert out.shape[0] == e
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
        out[i].copy_(shuffled.view(slc.dtype))
    return out


def _shuffle_per_expert_bias_w3_w1(
    stacked: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply gated-GEMM row shuffle to a 1D-per-expert bias tensor.

    Mirrors PT's ``MXFP4WeightTRTLLMGenFusedMoEMethod.load_expert_w3_w1_weight``
    bias path (quantization.py:4237-4271): the same permute (interleave w3/w1
    halves + epilogue-tile block reorder) is applied to bias rows as to weight
    rows, so ``bias[i]`` aligns with ``weight_row[i]`` after the shuffle.
    Skipping this step makes ``gemm1_bias`` index into the wrong post-shuffle
    output rows and produces garbage MoE output.
    """
    e = stacked.size(0)
    if out is None:
        out_list = []
        for i in range(e):
            slc = stacked[i].contiguous()  # [2*I_pad] 1D
            perm = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
                slc,
                _PERMUTE_CACHE,
                _EPILOGUE_TILE_M,
            )
            shuffled = torch.ops.trtllm.shuffle_matrix(slc, perm.to(slc.device))
            out_list.append(shuffled)
        return torch.stack(out_list, dim=0).contiguous()

    assert out.shape[0] == e
    for i in range(e):
        slc = stacked[i].contiguous()
        perm = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            slc,
            _PERMUTE_CACHE,
            _EPILOGUE_TILE_M,
        )
        shuffled = torch.ops.trtllm.shuffle_matrix(slc, perm.to(slc.device))
        out[i].copy_(shuffled)
    return out


def _shuffle_per_expert_bias_w2(
    stacked: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply non-gated TMA row shuffle to a 1D-per-expert bias tensor.

    Mirrors PT's ``MXFP4WeightTRTLLMGenFusedMoEMethod.load_expert_w2_weight``
    bias path (quantization.py:4304-4319): only the epilogue-tile block reorder
    is applied (no gated_act_gemm interleave for the non-gated GEMM2).
    """
    e = stacked.size(0)
    if out is None:
        out_list = []
        for i in range(e):
            slc = stacked[i].contiguous()  # [H_pad] 1D
            perm = trtllmgen_maybe_get_cached_w2_permute_indices(
                slc,
                _PERMUTE_CACHE,
                _EPILOGUE_TILE_M,
            )
            shuffled = torch.ops.trtllm.shuffle_matrix(slc, perm.to(slc.device))
            out_list.append(shuffled)
        return torch.stack(out_list, dim=0).contiguous()

    assert out.shape[0] == e
    for i in range(e):
        slc = stacked[i].contiguous()
        perm = trtllmgen_maybe_get_cached_w2_permute_indices(
            slc,
            _PERMUTE_CACHE,
            _EPILOGUE_TILE_M,
        )
        shuffled = torch.ops.trtllm.shuffle_matrix(slc, perm.to(slc.device))
        out[i].copy_(shuffled)
    return out


def prepare_trtllm_gen_moe_mxfp4_weights(
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
    scratch: MXFP4PrepScratch | None = None,
) -> TRTLLMGenMXFP4MoEWeights:
    """Convert HF on-disk MXFP4 expert weights into trtllm-gen-ready stacked tensors.

    Mirrors the algorithm in
    ``MXFP4WeightTRTLLMGenFusedMoEMethod.{post_load_weights,
    load_expert_w3_w1_weight, load_expert_w2_weight,
    load_expert_w3_w1_weight_scale_mxfp4, load_expert_w2_weight_scale_mxfp4}``.

    For ``tp_size > 1`` (TP-MoE):
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

    Scratch path (``scratch != None``): all kernel-layout outputs (pad +
    shuffle results and the fp32 biases) are written into the pre-allocated
    GPU buffers in :class:`MXFP4PrepScratch`. The returned
    :class:`TRTLLMGenMXFP4MoEWeights` fields are VIEWS of those buffers, so
    the caller MUST consume / copy them out before the next call to this
    function overwrites the scratch. Scratch path only supports
    ``tp_size == 1`` (the intended use case is ``FuseMXFP4Moe`` calling
    this helper after the load hook has already done TP slicing).
    """
    if tp_size > 1 and intermediate_size % tp_size != 0:
        raise ValueError(
            f"intermediate_size ({intermediate_size}) must be divisible by "
            f"tp_size ({tp_size}) for TP-MoE."
        )
    if scratch is not None and tp_size != 1:
        # The scratch path assumes its input is already TP-sliced (caller
        # does that in the load hook). Combining scratch with tp_size > 1
        # would double-slice on the intermediate axis. Loud error rather
        # than silent corruption.
        raise ValueError(
            "prepare_trtllm_gen_moe_mxfp4_weights: scratch is only supported with "
            f"tp_size=1 (got tp_size={tp_size}). The caller is expected to do "
            "TP slicing before this helper when using scratch."
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
    intermediate_size_pad, hidden_w1_pad, hidden_w2_pad = _compute_padded_dims(
        intermediate_size_for_local, hidden_size
    )

    # gate_up weights — pad each half [E, I, H/2] to [E, I_pad, H_w1_pad/2]
    # SEPARATELY so the zero-pad rows live inside each half, then stack as
    # [up | gate]. PT's gpt-oss loader (modeling_gpt_oss.py:695-706 +
    # quantization.py:4252-4258) ends up with ``dst_w3 = up`` in the first
    # half and ``dst_w1 = gate`` in the second half via this exact
    # de-interleave + chunk dance.
    #
    # When scratch is provided, we write the two halves directly into the
    # first/second halves of ``scratch.fc1_w_pad_buf`` (no per-half
    # tensor + no concat alloc).
    if scratch is None:
        up_padded_w = _pad_per_expert_2d(up_rows_w, hidden_w1_pad // 2, intermediate_size_pad)
        gate_padded_w = _pad_per_expert_2d(gate_rows_w, hidden_w1_pad // 2, intermediate_size_pad)
        gu_padded = torch.cat(
            [up_padded_w, gate_padded_w], dim=1
        ).contiguous()  # [E, 2I_pad, H_w1_pad/2]
    else:
        i_pad = intermediate_size_pad
        gu_padded = scratch.fc1_w_pad_buf
        _pad_per_expert_2d(
            up_rows_w,
            hidden_w1_pad // 2,
            intermediate_size_pad,
            out=gu_padded[:, :i_pad, :],
        )
        _pad_per_expert_2d(
            gate_rows_w,
            hidden_w1_pad // 2,
            intermediate_size_pad,
            out=gu_padded[:, i_pad:, :],
        )

    # down: rows = H, cols = I/2. Target shape [E, H_w2_pad, I_pad/2 = 1472].
    # PT pads w2's I/2 axis to ``alignment // 2`` where alignment=128,
    # giving 64-multiple (quantization.py:4287). For I/2=1440 → 1472.
    # The kernel then asserts ``gemm2_weights.shape[2] == intermediate_size / 2``,
    # so I_pad_w2 must match I_pad_w1 (both 2944).
    if scratch is None:
        dn_padded = _pad_per_expert_2d(dn_3d, intermediate_size_pad // 2, hidden_w2_pad)
    else:
        dn_padded = scratch.fc2_w_pad_buf
        _pad_per_expert_2d(dn_3d, intermediate_size_pad // 2, hidden_w2_pad, out=dn_padded)

    # 4. Pad scales — same per-half logic for w1; col_alignment uses
    #    scaling-vector size.
    if scratch is None:
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
    else:
        i_pad = intermediate_size_pad
        gu_scale_padded = scratch.fc1_s_pad_buf
        _pad_per_expert_2d(
            up_rows_s,
            hidden_w1_pad // _MXFP4_SCALING_VECTOR_SIZE,
            intermediate_size_pad,
            out=gu_scale_padded[:, :i_pad, :],
        )
        _pad_per_expert_2d(
            gate_rows_s,
            hidden_w1_pad // _MXFP4_SCALING_VECTOR_SIZE,
            intermediate_size_pad,
            out=gu_scale_padded[:, i_pad:, :],
        )
        dn_scale_padded = scratch.fc2_s_pad_buf
        _pad_per_expert_2d(
            down_scales,
            intermediate_size_pad // _MXFP4_SCALING_VECTOR_SIZE,
            hidden_w2_pad,
            out=dn_scale_padded,
        )

    # 5. Shuffle weights + scales for the kernel's TMA layout.
    if scratch is None:
        fc1_weights = _shuffle_per_expert_w3_w1(gu_padded)
        fc1_weights_scale = _shuffle_per_expert_w3_w1(
            gu_scale_padded, num_elts_per_sf=_MXFP4_SCALING_VECTOR_SIZE, is_scale=True
        )
        fc2_weights = _shuffle_per_expert_w2(dn_padded)
        fc2_weights_scale = _shuffle_per_expert_w2(
            dn_scale_padded, num_elts_per_sf=_MXFP4_SCALING_VECTOR_SIZE, is_scale=True
        )
    else:
        fc1_weights = _shuffle_per_expert_w3_w1(gu_padded, out=scratch.fc1_w_buf)
        fc1_weights_scale = _shuffle_per_expert_w3_w1(
            gu_scale_padded,
            num_elts_per_sf=_MXFP4_SCALING_VECTOR_SIZE,
            is_scale=True,
            out=scratch.fc1_s_buf,
        )
        fc2_weights = _shuffle_per_expert_w2(dn_padded, out=scratch.fc2_w_buf)
        fc2_weights_scale = _shuffle_per_expert_w2(
            dn_scale_padded,
            num_elts_per_sf=_MXFP4_SCALING_VECTOR_SIZE,
            is_scale=True,
            out=scratch.fc2_s_buf,
        )

    # 6. Bias: convert to float32. For w2, divide by tp_size (no-op at tp=1).
    #    Pad each half separately so the [up | gate] split matches the
    #    weights' row layout.
    if scratch is None:
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
    else:
        i_pad = intermediate_size_pad
        # _pad_per_expert_2d writes through ``copy_`` so dtype must match.
        # The scratch fp32 buffer can absorb the bf16-padded values via
        # PyTorch's implicit cast in ``copy_``. We use a tiny per-half view
        # so the layout matches [up | gate] without an explicit concat.
        _pad_per_expert_2d(
            up_b.unsqueeze(-1),
            col_alignment=1,
            row_alignment=intermediate_size_pad,
            out=scratch.fc1_b_pad_buf[:, :i_pad].unsqueeze(-1),
        )
        _pad_per_expert_2d(
            gate_b.unsqueeze(-1),
            col_alignment=1,
            row_alignment=intermediate_size_pad,
            out=scratch.fc1_b_pad_buf[:, i_pad:].unsqueeze(-1),
        )
        fc1_bias_padded = scratch.fc1_b_pad_buf  # [E, 2I_pad] fp32

    # Match PT: bias rows go through the SAME row-permutation as the weight
    # rows so ``bias[i]`` lines up with ``weight_row[i]`` after the kernel's
    # TMA-layout shuffle. Without this the kernel's epilogue adds the wrong
    # bias to each output row and the MoE output is garbage (eval ~2% on
    # gpt-oss-120b GSM8K instead of ~90%).
    if scratch is None:
        fc1_bias_padded = _shuffle_per_expert_bias_w3_w1(fc1_bias_padded)
    else:
        fc1_bias_padded = _shuffle_per_expert_bias_w3_w1(fc1_bias_padded, out=scratch.fc1_b_buf)

    if scratch is None:
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
        # Same TMA-layout shuffle as ``fc2_weights`` (no gated_act interleave for
        # the non-gated GEMM2). PT's ``load_expert_w2_weight`` (quantization.py:
        # 4304-4319) runs this shuffle on the bias too.
        fc2_bias_padded = _shuffle_per_expert_bias_w2(fc2_bias_padded)
    else:
        # Pad bf16 → fp32 into scratch pad buffer.
        _pad_per_expert_2d(
            down_bias.unsqueeze(-1),
            col_alignment=1,
            row_alignment=hidden_w2_pad,
            out=scratch.fc2_b_pad_buf.unsqueeze(-1),
        )
        # tp_size > 1 is rejected for scratch above, so no /tp_size needed.
        fc2_bias_padded = _shuffle_per_expert_bias_w2(scratch.fc2_b_pad_buf, out=scratch.fc2_b_buf)

    intermediate_size_padded = fc1_weights.shape[1] // 2  # 2I_pad / 2 = I_pad
    hidden_size_padded = fc1_weights.shape[-1] * 2  # (H_pad/2) * 2 = H_pad

    return TRTLLMGenMXFP4MoEWeights(
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
