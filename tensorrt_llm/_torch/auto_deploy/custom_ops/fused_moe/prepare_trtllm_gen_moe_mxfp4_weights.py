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

    ``out=None``: build a fresh stacked tensor (one alloc).
    ``out=...``:  write each expert's padded matrix into ``out[i]`` in-place
    so caller-provided storage (e.g. a scratch slice) is filled directly.
    """
    e = weight_3d.size(0)
    padded_per_expert = (
        maybe_pad_for_mxfp4(weight_3d[i], col_alignment, row_alignment) for i in range(e)
    )
    if out is None:
        return torch.stack(list(padded_per_expert), dim=0).contiguous()
    assert out.shape[0] == e, f"out leading dim {out.shape[0]} != e {e}"
    for i, padded in enumerate(padded_per_expert):
        out[i].copy_(padded)
    return out


def _shuffle_one_expert(
    slc: torch.Tensor,
    permute_fn,
    num_elts_per_sf: int | None,
    is_scale: bool,
) -> torch.Tensor:
    """Single-expert TMA-layout shuffle. Looping over experts is required
    because PT's permute-index helpers derive indices from a 2-D shape.

    ``permute_fn`` selects the gated (w3/w1) or non-gated (w2) permutation;
    ``is_scale=True`` chains ``block_scale_interleave`` (kernel scale layout).
    """
    slc = slc.contiguous()
    perm = permute_fn(slc, _PERMUTE_CACHE, _EPILOGUE_TILE_M, num_elts_per_sf=num_elts_per_sf)
    shuffled = torch.ops.trtllm.shuffle_matrix(slc, perm.to(slc.device))
    if is_scale:
        shuffled = torch.ops.trtllm.block_scale_interleave(shuffled).reshape(slc.shape)
    return shuffled.view(slc.dtype)


def _shuffle_per_expert(
    stacked: torch.Tensor,
    permute_fn,
    *,
    num_elts_per_sf: int | None = None,
    is_scale: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-expert TMA-layout shuffle, used for weights, scales, and biases.

    PT mirror points (`tensorrt_llm/_torch/modules/fused_moe/quantization.py`):
      * weights: ``load_expert_w3_w1_weight`` / ``load_expert_w2_weight``
      * scales:  ``..._weight_scale_mxfp4`` (adds ``block_scale_interleave``)
      * biases:  same row permute as weights so ``bias[i]`` aligns with
        ``weight_row[i]`` post-shuffle (gemm1_bias indexes into the wrong
        rows otherwise → MoE output garbage).

    ``out=None`` builds a fresh stacked tensor; otherwise per-expert results
    are ``copy_``-ed into ``out[i]`` so caller-provided storage is filled.
    """
    e = stacked.size(0)
    per_expert = (
        _shuffle_one_expert(stacked[i], permute_fn, num_elts_per_sf, is_scale) for i in range(e)
    )
    if out is None:
        return torch.stack(list(per_expert), dim=0).contiguous()
    assert out.shape[0] == e
    for i, shuffled in enumerate(per_expert):
        out[i].copy_(shuffled)
    return out


def _deinterleave_gate_up(
    gu_3d: torch.Tensor,  # [E, 2I, H/2]
    gate_up_scales: torch.Tensor,  # [E, 2I, H/32]
    gate_up_bias: torch.Tensor,  # [E, 2I]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the interleaved 2I axis into separate gate/up halves.

    HF's MXFP4 gate_up layout interleaves gate at even rows and up at odd rows.
    PT's gpt-oss loader (``modeling_gpt_oss.py:695-706`` +
    ``quantization.py:4252-4258``) ends up with ``dst_w3 = up`` in the first
    half and ``dst_w1 = gate`` in the second half. We keep up/gate separate
    here so downstream row-padding puts the zero-pad rows INSIDE each half,
    not at the very end of the concatenated 2I axis.
    """
    return (
        gu_3d[:, 0::2, :].contiguous(),  # gate_rows_w
        gu_3d[:, 1::2, :].contiguous(),  # up_rows_w
        gate_up_scales[:, 0::2, :].contiguous(),  # gate_rows_s
        gate_up_scales[:, 1::2, :].contiguous(),  # up_rows_s
        gate_up_bias[:, 0::2].contiguous(),  # gate_b
        gate_up_bias[:, 1::2].contiguous(),  # up_b
    )


def _pad_and_slice_axis(t: torch.Tensor, dim: int, target: int, lo: int, hi: int) -> torch.Tensor:
    """Pad ``t`` on ``dim`` to ``target`` then slice ``[lo:hi]`` on that dim."""
    cur = t.shape[dim]
    if cur < target:
        pad_amount = target - cur
        # F.pad spec is reversed-axis order; build dynamically.
        pad_spec = [0, 0] * (t.dim() - dim - 1) + [0, pad_amount] + [0, 0] * dim
        t = torch.nn.functional.pad(t, pad_spec)
    idx = [slice(None)] * t.dim()
    idx[dim] = slice(lo, hi)
    return t[tuple(idx)].contiguous()


def _tp_slice_intermediate_axis(
    gate_rows_w: torch.Tensor,
    up_rows_w: torch.Tensor,
    gate_rows_s: torch.Tensor,
    up_rows_s: torch.Tensor,
    gate_b: torch.Tensor,
    up_b: torch.Tensor,
    dn_3d: torch.Tensor,
    down_scales: torch.Tensor,
    intermediate_size: int,
    tp_size: int,
    tp_rank: int,
):
    """Pre-pad + slice the intermediate axis to this rank's range.

    Mirrors PT's ``MXFP4WeightTRTLLMGenFusedMoEMethod`` shard math
    (``quantization.py:4211-4234``): pad I to ``i_padded_tp`` (a multiple of
    ``alignment_tp`` so ``i_padded_tp / tp_size`` is itself 128-aligned),
    then slice each tensor on its intermediate-encoding axis. PRE-padding
    before sharding guarantees scaling-factor blocks (32 elements each) do
    not straddle rank boundaries.

    For gpt-oss-120b with I=2880 @ tp=8: ``alignment_tp=3072`` →
    ``per_rank_i=384``.

    Returns the sliced tensors plus ``(per_rank_i, valid_intermediate)``.
    """
    alignment_tp = _get_weight_alignment(
        _WEIGHT_ALIGNMENT, _MXFP4_SCALING_VECTOR_SIZE, tp_size, intermediate_size
    )
    i_padded_tp = ((intermediate_size + alignment_tp - 1) // alignment_tp) * alignment_tp
    per_rank_i = i_padded_tp // tp_size
    slice_start = tp_rank * per_rank_i
    slice_stop = (tp_rank + 1) * per_rank_i
    valid_intermediate = max(0, min(intermediate_size, slice_stop) - slice_start)

    # Pad I axis (rows) of gate / up to i_padded_tp, then slice this rank's chunk.
    def shard(t: torch.Tensor, dim: int, target: int, lo: int, hi: int) -> torch.Tensor:
        return _pad_and_slice_axis(t, dim, target, lo, hi)

    sf_padded = i_padded_tp // _MXFP4_SCALING_VECTOR_SIZE
    sf_start = slice_start // _MXFP4_SCALING_VECTOR_SIZE
    sf_stop = slice_stop // _MXFP4_SCALING_VECTOR_SIZE
    return (
        shard(gate_rows_w, 1, i_padded_tp, slice_start, slice_stop),
        shard(up_rows_w, 1, i_padded_tp, slice_start, slice_stop),
        shard(gate_rows_s, 1, i_padded_tp, slice_start, slice_stop),
        shard(up_rows_s, 1, i_padded_tp, slice_start, slice_stop),
        shard(gate_b, 1, i_padded_tp, slice_start, slice_stop),
        shard(up_b, 1, i_padded_tp, slice_start, slice_stop),
        shard(dn_3d, 2, i_padded_tp // 2, slice_start // 2, slice_stop // 2),
        shard(down_scales, 2, sf_padded, sf_start, sf_stop),
        per_rank_i,
        valid_intermediate,
    )


def _pad_concat_gate_up(
    up_rows: torch.Tensor,
    gate_rows: torch.Tensor,
    col_alignment: int,
    intermediate_size_pad: int,
    *,
    scratch_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pad each half [E, I, X] then concat as ``[up | gate]`` on the 2I axis.

    Used for both weights (col_alignment = ``hidden_w1_pad // 2``) and scales
    (col_alignment = ``hidden_w1_pad // _MXFP4_SCALING_VECTOR_SIZE``). When
    ``scratch_buf`` is provided, the two halves are written directly into
    ``scratch_buf[:, :i_pad]`` and ``scratch_buf[:, i_pad:]`` — no per-half
    tensors or final concat alloc.
    """
    if scratch_buf is None:
        up_p = _pad_per_expert_2d(up_rows, col_alignment, intermediate_size_pad)
        gate_p = _pad_per_expert_2d(gate_rows, col_alignment, intermediate_size_pad)
        return torch.cat([up_p, gate_p], dim=1).contiguous()
    i_pad = intermediate_size_pad
    _pad_per_expert_2d(up_rows, col_alignment, intermediate_size_pad, out=scratch_buf[:, :i_pad, :])
    _pad_per_expert_2d(
        gate_rows, col_alignment, intermediate_size_pad, out=scratch_buf[:, i_pad:, :]
    )
    return scratch_buf


def _pad_fc2(
    t: torch.Tensor,
    col_alignment: int,
    row_alignment: int,
    *,
    scratch_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-expert pad on the [E, H, X] down tensor. No concat (single half)."""
    if scratch_buf is None:
        return _pad_per_expert_2d(t, col_alignment, row_alignment)
    _pad_per_expert_2d(t, col_alignment, row_alignment, out=scratch_buf)
    return scratch_buf


def _shuffle_weights_and_scales(
    gu_padded: torch.Tensor,
    dn_padded: torch.Tensor,
    gu_scale_padded: torch.Tensor,
    dn_scale_padded: torch.Tensor,
    *,
    scratch: "MXFP4PrepScratch | None" = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply per-expert TMA-layout shuffle to weights + scales for both GEMMs."""
    w3w1 = trtllmgen_maybe_get_cached_w3_w1_permute_indices
    w2 = trtllmgen_maybe_get_cached_w2_permute_indices
    fc1_w_out = scratch.fc1_w_buf if scratch is not None else None
    fc1_s_out = scratch.fc1_s_buf if scratch is not None else None
    fc2_w_out = scratch.fc2_w_buf if scratch is not None else None
    fc2_s_out = scratch.fc2_s_buf if scratch is not None else None
    fc1_w = _shuffle_per_expert(gu_padded, w3w1, out=fc1_w_out)
    fc1_s = _shuffle_per_expert(
        gu_scale_padded,
        w3w1,
        num_elts_per_sf=_MXFP4_SCALING_VECTOR_SIZE,
        is_scale=True,
        out=fc1_s_out,
    )
    fc2_w = _shuffle_per_expert(dn_padded, w2, out=fc2_w_out)
    fc2_s = _shuffle_per_expert(
        dn_scale_padded,
        w2,
        num_elts_per_sf=_MXFP4_SCALING_VECTOR_SIZE,
        is_scale=True,
        out=fc2_s_out,
    )
    return fc1_w, fc1_s, fc2_w, fc2_s


def _prepare_fc1_bias(
    up_b: torch.Tensor,
    gate_b: torch.Tensor,
    intermediate_size_pad: int,
    *,
    scratch_pad_buf: torch.Tensor | None = None,
    scratch_out_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pad each half [E, I] → [E, I_pad] (fp32), concat ``[up | gate]``, shuffle.

    The TMA-layout row shuffle on the bias is critical: PT applies the SAME
    permute to bias rows as to weight rows so ``bias[i]`` aligns with
    ``weight_row[i]`` after the shuffle. Skipping it → kernel's epilogue adds
    the wrong bias to each output row → MoE output garbage (~2% GSM8K).
    """
    if scratch_pad_buf is None:
        up_p = (
            _pad_per_expert_2d(up_b.unsqueeze(-1), 1, intermediate_size_pad)
            .squeeze(-1)
            .float()
            .contiguous()
        )
        gate_p = (
            _pad_per_expert_2d(gate_b.unsqueeze(-1), 1, intermediate_size_pad)
            .squeeze(-1)
            .float()
            .contiguous()
        )
        fc1_bias_padded = torch.cat([up_p, gate_p], dim=1).contiguous()
    else:
        # ``_pad_per_expert_2d`` writes through ``copy_``; the scratch fp32
        # buffer absorbs bf16-padded values via copy_'s implicit cast.
        i_pad = intermediate_size_pad
        _pad_per_expert_2d(
            up_b.unsqueeze(-1),
            1,
            intermediate_size_pad,
            out=scratch_pad_buf[:, :i_pad].unsqueeze(-1),
        )
        _pad_per_expert_2d(
            gate_b.unsqueeze(-1),
            1,
            intermediate_size_pad,
            out=scratch_pad_buf[:, i_pad:].unsqueeze(-1),
        )
        fc1_bias_padded = scratch_pad_buf
    return _shuffle_per_expert(
        fc1_bias_padded, trtllmgen_maybe_get_cached_w3_w1_permute_indices, out=scratch_out_buf
    )


def _prepare_fc2_bias(
    down_bias: torch.Tensor,
    hidden_w2_pad: int,
    tp_size: int,
    *,
    scratch_pad_buf: torch.Tensor | None = None,
    scratch_out_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pad ``[E, H] → [E, H_pad]`` (fp32), divide by ``tp_size``, shuffle.

    Scratch path enforces ``tp_size == 1`` upstream (TP slicing happens in
    the load hook before this helper is reached), so no division is applied
    when using scratch.
    """
    if scratch_pad_buf is None:
        fc2_b = (
            _pad_per_expert_2d(down_bias.unsqueeze(-1), 1, hidden_w2_pad)
            .squeeze(-1)
            .float()
            .contiguous()
        )
        if tp_size > 1:
            fc2_b = fc2_b / tp_size
    else:
        _pad_per_expert_2d(
            down_bias.unsqueeze(-1),
            1,
            hidden_w2_pad,
            out=scratch_pad_buf.unsqueeze(-1),
        )
        fc2_b = scratch_pad_buf
    return _shuffle_per_expert(
        fc2_b, trtllmgen_maybe_get_cached_w2_permute_indices, out=scratch_out_buf
    )


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
        # Scratch path assumes inputs are already TP-sliced (load hook does
        # that). Combining scratch with tp_size > 1 would double-slice.
        raise ValueError(
            "prepare_trtllm_gen_moe_mxfp4_weights: scratch is only supported with "
            f"tp_size=1 (got tp_size={tp_size}). The caller is expected to do "
            "TP slicing before this helper when using scratch."
        )
    if tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(f"tp_rank {tp_rank} out of range for tp_size {tp_size}")

    assert down_blocks.size(0) == gate_up_blocks.size(0)

    # 1. Flatten blocks ([..., 16] → flattened) and de-interleave gate/up halves.
    gu_3d = _flatten_block_dim(gate_up_blocks)  # [E, 2I, H/2]
    dn_3d = _flatten_block_dim(down_blocks)  # [E, H, I/2]
    gate_rows_w, up_rows_w, gate_rows_s, up_rows_s, gate_b, up_b = _deinterleave_gate_up(
        gu_3d, gate_up_scales, gate_up_bias
    )

    # 2. TP slicing on the intermediate axis (no-op for tp_size == 1).
    if tp_size > 1:
        (
            gate_rows_w,
            up_rows_w,
            gate_rows_s,
            up_rows_s,
            gate_b,
            up_b,
            dn_3d,
            down_scales,
            intermediate_size_for_local,
            valid_intermediate,
        ) = _tp_slice_intermediate_axis(
            gate_rows_w,
            up_rows_w,
            gate_rows_s,
            up_rows_s,
            gate_b,
            up_b,
            dn_3d,
            down_scales,
            intermediate_size,
            tp_size,
            tp_rank,
        )
    else:
        intermediate_size_for_local = intermediate_size
        valid_intermediate = intermediate_size

    # 3. Per-rank padded layout dims (mirrors PT
    #    ``MXFP4WeightTRTLLMGenFusedMoEMethod``: per-expert I + H padded
    #    BEFORE 2I row dim, so w1.2I = 2*I_pad — see ``_compute_padded_dims``).
    intermediate_size_pad, hidden_w1_pad, hidden_w2_pad = _compute_padded_dims(
        intermediate_size_for_local, hidden_size
    )

    # 4. Pad weights + scales (concat [up | gate] for fc1; single half for fc2).
    sv = _MXFP4_SCALING_VECTOR_SIZE
    gu_padded = _pad_concat_gate_up(
        up_rows_w,
        gate_rows_w,
        hidden_w1_pad // 2,
        intermediate_size_pad,
        scratch_buf=scratch.fc1_w_pad_buf if scratch is not None else None,
    )
    dn_padded = _pad_fc2(
        dn_3d,
        intermediate_size_pad // 2,
        hidden_w2_pad,
        scratch_buf=scratch.fc2_w_pad_buf if scratch is not None else None,
    )
    gu_scale_padded = _pad_concat_gate_up(
        up_rows_s,
        gate_rows_s,
        hidden_w1_pad // sv,
        intermediate_size_pad,
        scratch_buf=scratch.fc1_s_pad_buf if scratch is not None else None,
    )
    dn_scale_padded = _pad_fc2(
        down_scales,
        intermediate_size_pad // sv,
        hidden_w2_pad,
        scratch_buf=scratch.fc2_s_pad_buf if scratch is not None else None,
    )

    # 5. Per-expert TMA-layout shuffle (weights + scales).
    fc1_weights, fc1_weights_scale, fc2_weights, fc2_weights_scale = _shuffle_weights_and_scales(
        gu_padded, dn_padded, gu_scale_padded, dn_scale_padded, scratch=scratch
    )

    # 6. Pad + shuffle biases (fp32, w2 bias divided by tp_size at tp>1).
    fc1_bias_padded = _prepare_fc1_bias(
        up_b,
        gate_b,
        intermediate_size_pad,
        scratch_pad_buf=scratch.fc1_b_pad_buf if scratch is not None else None,
        scratch_out_buf=scratch.fc1_b_buf if scratch is not None else None,
    )
    fc2_bias_padded = _prepare_fc2_bias(
        down_bias,
        hidden_w2_pad,
        tp_size,
        scratch_pad_buf=scratch.fc2_b_pad_buf if scratch is not None else None,
        scratch_out_buf=scratch.fc2_b_buf if scratch is not None else None,
    )

    intermediate_size_padded = fc1_weights.shape[1] // 2  # 2I_pad / 2 = I_pad
    hidden_size_padded = fc1_weights.shape[-1] * 2  # (H_pad/2) * 2 = H_pad
    return TRTLLMGenMXFP4MoEWeights(
        fc1_weights_mxfp4=fc1_weights,
        fc1_weights_scale_ue8m0=fc1_weights_scale,
        fc1_bias_f32=fc1_bias_padded,
        fc2_weights_mxfp4=fc2_weights,
        fc2_weights_scale_ue8m0=fc2_weights_scale,
        fc2_bias_f32=fc2_bias_padded,
        valid_hidden_size=hidden_size,
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
