# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 decode attention-residual mix + RMSNorm, optionally after the residual add, in one PDL
kernel (CuTe DSL, ``kernels/k3_attn_res.py``)."""

from __future__ import annotations

from typing import Tuple, Union

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm.attn_res_fwd import (
    attn_res_fwd,
)

from ...kernels.k3r_attn_res import k3_attn_res as _kernel

#: Rows of one mid-M scale-atom block (``mx_atom``).
_ATOM_ROWS = 128


def mxfp8_rows_reference(rows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The MXFP8 form ``(xq, xs)`` of BF16 ``rows [M, K]`` that the low-M FP8 GEMM quantizes its
    BF16 operand to: per 1x32 group the UE8M0 exponent of ``2^ceil(log2(amax / 448))`` (amax
    floored at the BF16 value 0x38D2, ~1e-4), and each value times ``2^(127 - e)`` rounded to E4M3
    (round to nearest, saturating). ``xs`` is ``uint8 [M, K / 32]``."""
    m, k = rows.shape
    groups = rows.contiguous().view(torch.int16).to(torch.int32).reshape(m, k // 32, 32) & 0x7FFF
    amax = groups.amax(dim=-1).clamp_min(0x38D2)
    exponent = (amax >> 7) - 8 + ((amax & 0x7F) > 0x60).to(torch.int32)
    scale = torch.exp2((127 - exponent).to(torch.float32)).repeat_interleave(32, dim=1)
    scaled = (rows.float() * scale).clamp(-448.0, 448.0)
    return scaled.to(torch.float8_e4m3fn), exponent.to(torch.uint8)


def mx_atom_offsets(num_rows: int, hidden: int, device: torch.device) -> torch.Tensor:
    """Byte offsets, row-major over ``[num_rows, hidden / 32]``, of the scales ``mx_atom`` stores:
    group ``g`` of row ``t`` at ``(g / 4) * 512 + (t % 32) * 16 + (t / 32) * 4 + g % 4`` (one
    128-row block of the mid-M FP8 GEMM's tcgen05 scale atoms)."""
    t = torch.arange(num_rows, device=device).unsqueeze(1)
    g = torch.arange(hidden // 32, device=device).unsqueeze(0)
    return ((g >> 2) * 512 + (t & 31) * 16 + ((t >> 5) & 3) * 4 + (g & 3)).reshape(-1)


def attn_res_rmsnorm_reference(
    layer: torch.Tensor,
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
) -> torch.Tensor:
    """``bf16(bf16(mixed * rsqrt(mean(mixed^2) + out_eps)) * out_weight)`` of the family
    ``attn_res_fwd`` mixture ``mixed`` of ``block [N - 1, M, H]`` and ``layer [M, H]``."""
    mixed, _, _, _ = attn_res_fwd.reference(layer, block, res_weight, rms_weight, rms_eps)
    mixed = mixed.float()
    normed = mixed * torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + out_eps)
    return out_weight.reshape(-1) * normed.to(torch.bfloat16)


class K3AttnResNorm(OpWrapper):
    """RMSNorm of the attention-residual mix of ``block`` and the layer row, with the layer row

    * ``addends == ()``: ``layer`` itself; returns ``output``;
    * ``(addend,)``: ``updated = bf16(layer + addend)``; returns ``(updated, output)``;
    * ``(routed, shared)``: ``updated = bf16(layer + bf16(routed + shared))`` (the previous MoE
      layer's residual update); returns ``(updated, output)``, with ``mxfp8`` also the MXFP8 form
      ``(xq, xs)`` of ``output`` (``xs`` in the mid-M GEMM's scale layout with ``mx_atom``).

    ``layer`` and the addends are BF16 ``[M, 7168]``, ``block`` BF16 ``[N - 1, M, 7168]`` (the
    snapshots), the three weights BF16 ``[7168]``; outputs are new tensors.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # rows: tokens of the decode step; candidates: N = snapshots + 1; addends: 0 (the mix alone),
    # 1 (the residual add) or 2 (the deferred MoE residual update). tp1 (context) never runs it.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=why,
            spec=dict(
                rows=rows,
                candidates=candidates,
                hidden=7168,
                addends=addends,
                mxfp8=mxfp8,
                entry_trigger=entry_trigger,
                mx_atom=mx_atom,
                early_trigger=False,
                rms_eps=1e-6,
                out_eps=1e-6,
            ),
        )
        for why, rows, candidates, addends, mxfp8, entry_trigger, mx_atom in (
            ("helix16 ULL 1-request step, mix alone: 8 rows, N=2", 8, 2, 0, False, False, False),
            (
                "dep16 per-rank step, mix alone: 64 rows (its cap), N=12",
                64,
                12,
                0,
                False,
                False,
                False,
            ),
            ("helix8 b16 mtp5 verify, mix alone: 96 rows, N=9", 96, 9, 0, False, False, False),
            ("one token, residual add, N=3: 4-CTA cluster", 1, 3, 1, False, False, False),
            ("helix8 A_mid add: 16 rows, N=5, 8-CTA cluster", 16, 5, 1, False, False, False),
            ("helix8 add at its 128-row cap, N=4", 128, 4, 1, False, False, False),
            ("dep16 deferred MoE residual: 32 rows, N=7", 32, 7, 2, False, False, False),
            ("helix8 deferred MoE residual: 96 rows, N=12", 96, 12, 2, False, False, False),
            (
                "helix8 MoE add + MXFP8 for the KDA QKVG low-M GEMM: 16 rows, N=8",
                16,
                8,
                2,
                True,
                False,
                False,
            ),
            ("helix16 ULL MoE add + MXFP8, entry trigger: 8 rows, N=6", 8, 6, 2, True, True, False),
            (
                "helix8 MoE add + MXFP8 in the mid-M GEMM's scale atoms: 96 rows, N=9",
                96,
                9,
                2,
                True,
                False,
                True,
            ),
            (
                "helix16 ULL 4-request step, MXFP8 atoms + entry trigger: 32 rows, N=4",
                32,
                4,
                2,
                True,
                True,
                True,
            ),
        )
    ) + (
        Cell(
            why="helix8 MoE add + MXFP8 atoms, release after the grid wait: 96 rows, N=9",
            spec=dict(
                rows=96,
                candidates=9,
                hidden=7168,
                addends=2,
                mxfp8=True,
                entry_trigger=False,
                mx_atom=True,
                early_trigger=True,
                rms_eps=1e-6,
                out_eps=1e-6,
            ),
        ),
    )

    note = """
    Decode-size fusion: up to the target's attn_res_pdl_max_m rows (helix8: 128,
    others: 64) where `kernels/k3_attn_res.can_use_k3_attn_res` holds (H 7168,
    2 <= N <= 12, M <= 1024, contiguous BF16). Inputs are read-only.
    A cluster of 4 or 8 CTAs per token (`fast_cluster_size`) loads the weights
    and snapshot rows and reduces their statistics before griddepcontrol.wait;
    only the layer row and the addends are read after it. The updated prefix
    has the bits of the unfused adds; the normed output can differ from the
    C++ per-token kernels by one BF16 step in a few elements (other sum order).
    A process-wide cache (`kernels/k3_attn_res._compiled`) holds one compiled
    kernel per (N, add mode, PDL, cluster, mxfp8, entry_trigger, mx_atom,
    early_trigger);
    compile each key eagerly before CUDA-graph capture. TRTLLM_ENABLE_PDL is
    read at import. `entry_trigger` releases the dependent grid at kernel entry
    and prefers the largest shared-memory carve-out; `mx_atom` (M <= 128) stores
    the scales of one 128-row block of the mid-M GEMM's atoms, rows past M are
    not written. `early_trigger` releases the dependent grid right after
    griddepcontrol.wait instead of after the last store.
    """

    def __call__(
        self,
        layer: torch.Tensor,
        addends: Tuple[torch.Tensor, ...],
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        mxfp8: bool = False,
        entry_trigger: bool = False,
        mx_atom: bool = False,
        early_trigger: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if len(addends) == 2:
            routed, shared = addends
            if mxfp8:
                return _kernel.k3_attn_res_moe_add_rmsnorm_mx(
                    layer,
                    routed,
                    shared,
                    block,
                    res_weight,
                    rms_weight,
                    out_weight,
                    rms_eps,
                    out_eps,
                    entry_trigger,
                    mx_atom,
                    early_trigger=early_trigger,
                )
            return _kernel.k3_attn_res_moe_add_rmsnorm(
                layer,
                routed,
                shared,
                block,
                res_weight,
                rms_weight,
                out_weight,
                rms_eps,
                out_eps,
                early_trigger=early_trigger,
            )
        if mxfp8:
            raise ValueError("the MXFP8 output is served with the (routed, shared) addends only")
        if addends:
            return _kernel.k3_attn_res_add_rmsnorm(
                layer,
                addends[0],
                block,
                res_weight,
                rms_weight,
                out_weight,
                rms_eps,
                out_eps,
                early_trigger=early_trigger,
            )
        return _kernel.k3_attn_res_rmsnorm(
            layer,
            block,
            res_weight,
            rms_weight,
            out_weight,
            rms_eps,
            out_eps,
            early_trigger=early_trigger,
        )

    def reference(
        self,
        layer: torch.Tensor,
        addends: Tuple[torch.Tensor, ...],
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        mxfp8: bool = False,
        entry_trigger: bool = False,
        mx_atom: bool = False,
        early_trigger: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """The adds in FP32 rounded to BF16 where the kernel rounds, then the family
        ``attn_res_fwd`` mixture and the twice-rounded trailing RMSNorm. With ``mxfp8`` the
        scales come row-major ``[M, 224]``, or with ``mx_atom`` at :func:`mx_atom_offsets` of a
        zeroed buffer of the op's size. ``entry_trigger`` and ``early_trigger`` only change
        scheduling."""
        if len(addends) == 2:
            routed, shared = addends
            moe = (routed.float() + shared.float()).to(torch.bfloat16)
            row = (layer.float() + moe.float()).to(torch.bfloat16)
        elif addends:
            row = (layer.float() + addends[0].float()).to(torch.bfloat16)
        else:
            row = layer
        output = attn_res_rmsnorm_reference(
            row, block, res_weight, rms_weight, out_weight, rms_eps, out_eps
        )
        if not addends:
            return output
        if not mxfp8:
            return row, output
        xq, xs = mxfp8_rows_reference(output)
        if mx_atom:
            hidden = layer.shape[1]
            atom = torch.zeros((hidden // _ATOM_ROWS) * 512, dtype=torch.uint8, device=layer.device)
            atom[mx_atom_offsets(layer.shape[0], hidden, layer.device)] = xs.reshape(-1)
            xs = atom
        return row, output, xq, xs

    def is_valid(
        self,
        layer: torch.Tensor,
        addends: Tuple[torch.Tensor, ...],
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        mxfp8: bool = False,
        entry_trigger: bool = False,
        mx_atom: bool = False,
        early_trigger: bool = False,
    ) -> None:
        # The launcher does not re-validate: a strided, mis-sized or foreign-device operand is
        # read as dense rows of 7168 and gives a plausible wrong mix.
        assert _kernel.can_use_k3_attn_res(
            layer, tuple(addends), block, res_weight, rms_weight, out_weight
        ), "outside can_use_k3_attn_res (H 7168, 2 <= N <= 12, M <= 1024, contiguous BF16)"
        # Without the MXFP8 output both flags are dropped without an error.
        assert mxfp8 or not (entry_trigger or mx_atom), "entry_trigger / mx_atom need mxfp8"


k3_attn_res_norm = K3AttnResNorm()
