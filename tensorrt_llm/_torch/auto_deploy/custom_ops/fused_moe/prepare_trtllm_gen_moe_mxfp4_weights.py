# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""MXFP4 weight prep for TRT-LLM-Gen ``bf16_mxe2m1_block_scale_moe_runner``.

Transforms HF on-disk MXFP4 expert tensors (``gate_up_proj_*``, ``down_proj_*`` registered by
``quantize_mxfp4_moe``) into the kernel-ready stacked layout that
``auto_deploy::trtllm_quant_mxfp4_trtllm_gen_w4a16_moe_fused`` expects: ``[E_local, 2I_pad, H_pad/2]`` weights,
``[E_local, 2I_pad]`` fp32 biases, etc., all run through ``torch.ops.trtllm.shuffle_matrix`` for
the TMA layout.

Mirrors PT's ``MXFP4WeightTRTLLMGenFusedMoEMethod``
(``tensorrt_llm/_torch/modules/fused_moe/quantization.py:4135``) — PT helpers
(``maybe_pad_for_mxfp4``, ``trtllmgen_maybe_get_cached_*``, ``_get_weight_alignment``) are imported
directly so the algorithm is byte-identical.
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

    ``i_pad`` / ``h_w2_pad`` align to 128 (TMA weight alignment);
    ``h_w1_pad`` aligns to 512 (TMA input-hidden constraint on w1's K-axis).
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

    Allocated once per build pass via :meth:`allocate` (sized for ONE MoE layer's per-rank shape —
    gpt-oss guarantees H/I/E are constant across layers) and reused on every layer to avoid
    per-layer pad/shuffle transients. ``trtllm.shuffle_matrix`` is not in-place, so we keep
    separate ``_pad_buf`` (post-pad, pre-shuffle) and no-suffix (post-shuffle, kernel-ready)
    buffers per tensor kind.

    Caller MUST ``copy_`` the prep result into final storage before the next call — the dataclass
    holds VIEWS of these buffers and the next call overwrites them. The intended use is
    :class:`FuseMXFP4Moe`: pre-allocate all layers' destination ``nn.Parameter`` storage first,
    then per-layer prep + ``copy_`` from scratch.
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
        """Allocate scratch for one MoE layer at the given per-rank shape.

        ``per_rank_i`` is the already-TP-sliced intermediate dim (= full ``I``
        when ``tp_size == 1``).
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
    """Single-expert TMA-layout shuffle (looped per-expert because PT's permute-index helpers
    derive indices from a 2-D shape).

    ``permute_fn``: gated (w3/w1) vs non-gated (w2). ``is_scale=True`` chains
    ``block_scale_interleave`` for the kernel's scale layout.
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
    """Per-expert TMA-layout shuffle (weights, scales, biases all share this).

    Biases use the SAME row permute as their weights so ``bias[i]`` aligns with ``weight_row[i]``
    post-shuffle — mismatch → kernel epilogue adds the wrong bias and MoE output is garbage.
    ``out=None`` returns a fresh stacked tensor; otherwise per-expert results are ``copy_``-ed
    into caller-provided storage.
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
    """Split the HF interleaved 2I axis (gate at even rows, up at odd rows) into separate halves.

    Kept separate so downstream row-padding puts the zero-pad rows INSIDE each half before re-concat
    as ``[up | gate]`` — matches PT's ``dst_w3 = up`` / ``dst_w1 = gate`` chunk layout.
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
    """Pre-pad ``I`` to ``i_padded_tp`` then slice this rank's range (mirrors PT
    ``quantization.py:4211-4234``).

    The alignment guarantees ``i_padded_tp / tp_size`` stays 128-aligned and that scaling-factor
    blocks (32 elements) don't straddle rank boundaries. Example: gpt-oss I=2880 @ tp=8 →
    ``alignment_tp=3072`` → ``per_rank_i=384``.

    No-op when ``tp_size == 1``: returns inputs unchanged with
    ``per_rank_i = valid_intermediate = intermediate_size``.

    Returns the (possibly sliced) tensors plus ``(per_rank_i, valid_intermediate)``.
    """
    if tp_size == 1:
        return (
            gate_rows_w,
            up_rows_w,
            gate_rows_s,
            up_rows_s,
            gate_b,
            up_b,
            dn_3d,
            down_scales,
            intermediate_size,
            intermediate_size,
        )

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


def _pad_concat_fc1(
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
    """Convert HF on-disk MXFP4 expert weights to the trtllm-gen kernel layout.

    Mirrors PT's ``MXFP4WeightTRTLLMGenFusedMoEMethod`` (``post_load_weights`` +
    ``load_expert_w{3_w1,2}_weight{,_scale_mxfp4}``).

    Notes on optional args:
      * ``tp_size > 1``: shard the intermediate dim before the kernel-layout pad+shuffle — see
        :func:`_tp_slice_intermediate_axis` for the TP-aware pre-pad + slice math (mirrors PT
        ``load_weight_shard``).
      * ``scratch != None``: pad/shuffle outputs are written into the pre-allocated GPU buffers and
        the returned dataclass holds VIEWS into that scratch, so caller must ``copy_`` results out
        before the next call. Only supported at ``tp_size == 1`` (load hook does TP slicing first
        — see :class:`MXFP4PrepScratch`).

    EP (expert-axis slicing) is NOT done here — the caller selects the expert subset before
    invoking.
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

    # 3. Per-rank padded layout dims (mirrors PT
    #    ``MXFP4WeightTRTLLMGenFusedMoEMethod``: per-expert I + H padded
    #    BEFORE 2I row dim, so w1.2I = 2*I_pad — see ``_compute_padded_dims``).
    intermediate_size_pad, hidden_w1_pad, hidden_w2_pad = _compute_padded_dims(
        intermediate_size_for_local, hidden_size
    )

    # 4. Pad weights + scales (concat [up | gate] for fc1; single half for fc2).
    sv = _MXFP4_SCALING_VECTOR_SIZE
    gu_padded = _pad_concat_fc1(
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
    gu_scale_padded = _pad_concat_fc1(
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
