# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the K3.1 Rubin attn_res entries share: the kernel's shapes, the cells, the references
(the mix and RMSNorm, the bf16 adds, the MXFP8 form of an output) and the input guard; for the
entries with a Lamport addend also their reference, finish guard and note. Not an entry."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell

from ..comm._rubin_comm_common import lamport_allreduce_reference

#: The kernel's limits (``kernels/norm/rubin_attn_res.py``): hidden size, rows per launch.
HIDDEN = 7168
MAX_TOKENS = 1024
#: Kimi K3.1's RMSNorm epsilon.
EPS = 1e-5
#: Rows of the K3.1 Rubin generation steps: TEP8 b1 k7; helix8 b16 k5 and DEP16 b12 k7 (per
#: rank); the largest row count the targets send to this kernel.
TEP8_ROWS, VERIFY96_ROWS, PDL_MAX_M = 8, 96, 128
#: Values per UE8M0 scale (one 1x32 group).
SF_VEC = 32
#: The catalog quantizer's ``rcp.approx.ftz.f32(448)``: the FP32 nearest 1 / 448 (the GPU tests'
#: ``448 * 2^k`` maxima check the instruction against it).
_RCP_448 = 1.0 / 448.0


def attn_res_cells(*, include_eps: bool = True) -> tuple[Cell, ...]:
    """The certified (rows, candidates, early_trigger) configurations of the PDL kernels.

    ``N`` is the candidate count (snapshots + the layer row): 2..9 in K3.1 (93 layers, a
    snapshot every 12). The cluster per row is 8 CTAs for up to 16 rows with ``N >= 4``,
    else 4, so 16 / 17 rows cover the switch.
    """
    cells = [
        Cell(
            why=f"TEP8 GEN b1 k7 verify step ({TEP8_ROWS} rows), N={n}",
            spec=dict(rows=TEP8_ROWS, candidates=n, early_trigger=True),
        )
        for n in range(2, 10)
    ]
    cells += [
        Cell(
            why=f"helix8 b16 k5 / DEP16 b12 k7 GEN verify step ({VERIFY96_ROWS} rows), N={n}",
            spec=dict(rows=VERIFY96_ROWS, candidates=n, early_trigger=True),
        )
        for n in range(2, 10)
    ]
    cells += [
        Cell(why="one decode row, N=9", spec=dict(rows=1, candidates=9, early_trigger=True)),
        Cell(
            why="16 rows, N=4: the last row count with eight-CTA clusters",
            spec=dict(rows=16, candidates=4, early_trigger=True),
        ),
        Cell(
            why="17 rows, N=4: four-CTA clusters",
            spec=dict(rows=17, candidates=4, early_trigger=True),
        ),
        Cell(
            why=f"{PDL_MAX_M} rows, N=9: the largest row count the targets send to this kernel",
            spec=dict(rows=PDL_MAX_M, candidates=9, early_trigger=True),
        ),
        Cell(
            why="early trigger off (dependents released after the stores), 96 rows, N=5",
            spec=dict(rows=VERIFY96_ROWS, candidates=5, early_trigger=False),
        ),
        Cell(
            why="early trigger off, 8 rows, N=9",
            spec=dict(rows=TEP8_ROWS, candidates=9, early_trigger=False),
        ),
    ]
    if include_eps:
        cells += [
            Cell(
                why=f"8 rows, N=5, eps=({eps},{out_eps}): distinct norm roots",
                spec=dict(rows=8, candidates=5, early_trigger=True, eps=eps, out_eps=out_eps),
            )
            for eps, out_eps in ((1e-2, 1e-5), (1e-5, 1e-2))
        ]
    return tuple(cells)


def flush_denormal(x: torch.Tensor) -> torch.Tensor:
    """fp32 ``x`` with denormals flushed to a signed zero (the ``.ftz`` PTX modifier)."""
    return torch.where(x.abs() < torch.finfo(torch.float32).tiny, x * 0.0, x)


def bf16_add(a: torch.Tensor, b: torch.Tensor, *, ftz: bool) -> torch.Tensor:
    """``bf16(a + b)``: an fp32 add rounded to nearest, then rounded to bf16 (``ftz``: the
    fast-math add, which flushes denormal operands and results)."""
    if not ftz:
        return (a.float() + b.float()).to(torch.bfloat16)
    s = flush_denormal(flush_denormal(a.float()) + flush_denormal(b.float()))
    return s.to(torch.bfloat16)


def mix_reference(
    layer: torch.Tensor,
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
) -> torch.Tensor:
    """The bf16 mixture of the candidates ``block[0..K-1]``, then ``layer`` ([M, H] each).

    fp32: ``logit_n = rsqrt(mean(v_n^2) + eps) * <v_n, res_w * rms_w>``, a softmax over n,
    ``sum_n p_n v_n``, rounded once to bf16 (where the kernel rounds the mixture).
    """
    values = torch.cat((block, layer.unsqueeze(0)), dim=0).float()
    rsigma = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + rms_eps)
    score_weight = rms_weight.float().reshape(-1) * res_weight.float().reshape(-1)
    logits = (values * rsigma * score_weight).sum(dim=-1)
    probs = torch.softmax(logits, dim=0)
    return (probs.unsqueeze(-1) * values).sum(dim=0).to(torch.bfloat16)


def rmsnorm_reference(
    mixed: torch.Tensor, out_weight: torch.Tensor, out_eps: float
) -> torch.Tensor:
    """``bf16(bf16(x * rsqrt(mean(x^2) + eps)) * w)`` in fp32: the K3.1 RMSNorm's two roundings."""
    x = mixed.float()
    normed = (x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + out_eps)).to(torch.bfloat16)
    return out_weight.reshape(-1) * normed


def mix_rmsnorm_reference(
    layer: torch.Tensor,
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
) -> torch.Tensor:
    """:func:`rmsnorm_reference` of :func:`mix_reference`."""
    return rmsnorm_reference(
        mix_reference(layer, block, res_weight, rms_weight, rms_eps), out_weight, out_eps
    )


def check_attn_res_inputs(
    rows: torch.Tensor, addends: tuple, block: torch.Tensor, *weights: torch.Tensor
) -> None:
    """The kernel's input contract; the launchers do not check it and read raw pointers."""
    assert rows.dim() == 2 and rows.shape[1] == HIDDEN, f"rows must be [M, {HIDDEN}]: {rows.shape}"
    m = rows.shape[0]
    assert 1 <= m <= MAX_TOKENS, f"the kernel takes 1..{MAX_TOKENS} rows, got {m}"
    assert block.dim() == 3 and tuple(block.shape[1:]) == (m, HIDDEN), (
        f"block must be [N - 1, {m}, {HIDDEN}]: {tuple(block.shape)}"
    )
    for name, t in (("rows", rows), ("block", block), *(("addend", a) for a in addends)):
        assert t.dtype == torch.bfloat16 and t.is_contiguous(), f"{name}: bf16 contiguous"
        assert t.is_cuda and t.device == rows.device, f"{name}: on the rows' device"
    for a in addends:
        assert a.shape == rows.shape, f"addend {tuple(a.shape)} != rows {tuple(rows.shape)}"
    for w in weights:
        assert w.dtype == torch.bfloat16 and w.is_contiguous() and w.numel() == HIDDEN, (
            f"weights: bf16 contiguous with {HIDDEN} elements"
        )
        assert w.device == rows.device, "weights: on the rows' device"


def mxfp8_exact_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(xq, xs)`` of bf16 ``x`` [M, K], M <= 128, as the catalog ``mxfp8_quantize`` quantizes it.

    In the library's IEEE arithmetic (it is built without ``--use_fast_math``), per 1x32 group:
    ``amax`` is its ``cuda_max(even, odd)`` = ``even > odd ? even : odd`` of the NaN-skipping
    maxima of the even and of the odd elements (a NaN when every odd element is a NaN, the odd
    maximum when every even one is); ``sf = amax * rcp(448)`` keeps a subnormal; ``e`` =
    ``__nv_cvt_float_to_e8m0(sf, satfinite, +inf)`` (the FP32 exponent field plus one for a
    non-zero mantissa; a subnormal ``sf`` gives 1 above 2^-127, else 0; at most 254; 254 for +inf,
    255 for NaN); ``xq = e4m3_rn_satfinite(x * exp2f_rcp(e))`` with ``exp2f_rcp`` 1 for ``e = 0``
    and ``2^(127 - e)`` otherwise, 0 for a zero ``amax``. ``xs`` holds the exponents in one
    128-row block of tcgen05 scale atoms (the swizzled R128c4 scales of one tile): flat
    ``[K / 128][512]`` with byte ``(r % 32) * 16 + (r % 128 // 32) * 4 + g % 4`` for row r and
    group g; atom bytes past M are left 0 here (the kernel leaves them unwritten). The attn_res
    kernel's MXFP8 bits.
    """
    m, k = x.shape
    g = x.reshape(m, k // SF_VEC, SF_VEC)
    bits = g.view(torch.int16).to(torch.int32) & 0x7FFF
    mag = torch.where(bits > 0x7F80, -1, bits)
    even, odd = mag[..., 0::2].amax(dim=-1), mag[..., 1::2].amax(dim=-1)
    mag = torch.where(odd < 0, -1, torch.maximum(even, odd))
    vmax = torch.where(mag < 0, 0x7FC00000, mag << 16).view(torch.float32)
    sf = vmax * torch.tensor(_RCP_448, dtype=torch.float32, device=x.device)
    e = _ue8m0_round_up(sf)
    tail = torch.where(e == 254, 0x00400000, 0x00200000).to(torch.int32)
    inv_bits = torch.where(e <= 253, (254 - e) << 23, tail)
    inv = torch.where(e == 0, 1.0, inv_bits.view(torch.float32))
    inv = torch.where(vmax == 0, 0.0, inv)
    scaled = g.float() * inv.unsqueeze(-1)
    scaled = torch.where(scaled.isnan(), float("nan"), scaled)  # the GPU's positive NaN
    xq = scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).reshape(m, k)
    xs = torch.zeros((k // 128) * 512, dtype=torch.uint8, device=x.device)
    row = torch.arange(m, device=x.device).view(m, 1)
    grp = torch.arange(k // SF_VEC, device=x.device).view(1, -1)
    off = (grp // 4) * 512 + (row % 32) * 16 + (row % 128 // 32) * 4 + grp % 4
    xs[off.reshape(-1)] = e.to(torch.uint8).reshape(-1)
    return xq, xs


def _ue8m0_round_up(sf: torch.Tensor) -> torch.Tensor:
    """``__nv_cvt_float_to_e8m0(sf, __NV_SATFINITE, cudaRoundPosInf)`` of FP32 ``sf`` (>= +0 or
    NaN) as int32, as the CUDA header defines it."""
    b = sf.view(torch.int32)
    e = (b >> 23) & 0xFF
    m = b & 0x7FFFFF
    up = torch.where(e == 0, m > 0x400000, m != 0) & (e < 255)
    e = (e + up.to(torch.int32)).clamp_max(254)
    return torch.where((b & 0x7FFFFFFF) > 0x7F800000, 255, e)


def lamport_reference(
    prefix: torch.Tensor,
    partials_in_rank_order: Sequence[torch.Tensor],
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
    chunk: int,
    add_rn: bool,
    ag_rows: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    addend = lamport_allreduce_reference(partials_in_rank_order, chunk, ag_rows)
    updated = bf16_add(prefix, addend, ftz=not add_rn)
    output = mix_rmsnorm_reference(
        updated, block, res_weight, rms_weight, out_weight, rms_eps, out_eps
    )
    return updated, output


def check_finish(finish_world: int, finish_chunk: int, rank: int) -> None:
    """Finish arguments the launcher takes and misreads."""
    assert 0 <= rank < finish_world, f"rank {rank} outside a group of {finish_world}"
    assert 1 <= finish_chunk <= finish_world, "finish_chunk ranks per partial sum"


LAMPORT_NOTE = """
    Same kernel and rounding as rubin_attn_res_add_rmsnorm; the addend's bits are the finished
    all-reduce's (rubin_lamport_finish). Contract the launcher does not check: the workspace
    (``uc_local``, ``flags``, ``mc_base``) holds this call's rows; every rank pushed into the
    call and its producer triggers only after its own griddepcontrol.wait; the next call on the
    workspace starts after this kernel completed. ``early_poll`` finishes the call before the
    grid wait and stores after it; ``early_trigger`` releases the dependent after the wait.
    Neither changes results. ``finish_ag`` (a call with all-gather rows, one-shot) is driven by
    the column-parallel MoE tail's own test.
    """
