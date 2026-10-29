# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp4_block_scale_moe_runner catalog entry.

The entry's other cases come first; the Mistral Large 4 NVFP4 tep4 section
(H 6144 / I 6144, top-4 of 128 experts, 32-wide windows, operands at the
checkpoint's magnitudes, 1..32768 tokens) follows them, then the dep4 section
(the same windows over the four ranks' gathered tokens, up to 131072), the tp4
and tp8 section (all 128 experts on every rank, each cut to its 1536- or
768-wide slice of the intermediate, 1..32768 tokens), the dep8 section (16-wide
windows over the eight ranks' gathered tokens, in calls of at most 32768), and
the tep8 section (the same windows, every rank over all of a step's 1..32768
tokens, decode graphs on windows (0, 16) and (112, 16)) closes the file.
"""

import math

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    fp4_block_scale_moe_runner as moe,
)
from tensorrt_llm._torch.autotuner import AutoTuner, autotune

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "fp4_block_scale_moe_runner requires a CUDA device"

DEV = "cuda"
# The reference GEMMs must be true fp32; TF32 would leave the reference with
# 10 mantissa bits, coarser than the bf16 output it is meant to bound.
torch.backends.cuda.matmul.allow_tf32 = False

# Relative distance between neighbouring bf16 values (1 + 7 stored mantissa
# bits -> one ulp is 2^-8 of the binade top).
ULP = 2.0**-8

# The 16 e2m1 code points, in code order: sign bit 3, exponent bits 2:1,
# mantissa bit 0.
E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
    device=DEV,
)
SV = 16  # NVFP4 block size: one e4m3 scale per 16 elements along K
E2M1_MAX = 6.0  # largest e2m1 magnitude
E4M3_MAX = 448.0  # largest finite e4m3 magnitude


# ── NVFP4 arithmetic (pure torch) ─────────────────────────────────────────


def _compare(
    out: torch.Tensor, ref: torch.Tensor, elt_ulp: float = 16.0, rms_ulp: float | None = 2.0
) -> None:
    """16 ulp per element against the row's own scale (`elt_ulp`, which the
    Mistral Large 4 cells scale with the FC2 contraction width, below), 2 ulp
    relative RMS (`rms_ulp`; None gates the element figure alone, which the
    Mistral Large 4 decode graph replays pass -- see the decode paragraphs).

    Kernel and reference consume bit-identical NVFP4 weights and activations
    and model the same FC1-output requantization, so they differ only in
    accumulation order and in whether a marginal FC1 value rounds to the
    same e2m1 code. An e2m1 code is a coarse step -- two mantissa bits -- so
    one flipped intermediate moves the output row by a visible fraction, and
    torch's default band cannot express that: its bf16 `atol` of 1e-5 sits
    three orders of magnitude below one output ulp of a two-GEMM chain, and
    per-element `rtol` is meaningless where cancellation drives `|ref|` to
    zero. That requantization is the only lossy stage between the GEMMs, and
    exactly one: read straight out of the kernel it is bit-exact at H = I =
    256 and again at H 2560 / I 640 over 512 experts, so the band absorbs no
    second, unmodelled quantization.

    Both numbers are measured, not chosen. Worst over every configuration
    the entry's test covers: 9.97 ulp element-wise and 0.88 ulp RMS; the R1
    geometry lands at 9.62 / 0.74 for a 64-wide window and 4.68 / 0.70 for
    the 256-expert stack. Only the RMS gate is scale-free -- the element
    figure is a max over `T*H` elements, so its tail grows with `T` at
    constant accuracy and reaches 16.33 at the top of the certified token
    column. Size a caller's tolerance off the RMS number. The qwen4_exp
    geometry (512 experts, top-10, H 2560 / I 640) is the least strained:
    3.82 / 0.70 over its column, element tails re-drawn over 12 seeds up to
    10.67 at T=4096 with the RMS figure at 0.67-0.69, and 5.39 / 0.67 with
    uniform rather than renormalized weights.

    Both were sized against deliberately wrong computations: a gate/up swap
    (263-278 ulp), a scalar-role swap (177/48), a missing scale swizzle
    (408/357), a missing row shuffle (508/432), and a reference skipping the
    FC1 requantization (43/28 element, 23/23 RMS). The smallest RMS distance
    any of them reaches is 39.8 at the deepseek geometries; at qwen4_exp's the
    same five land 223, 62, 265, 362 and 23 RMS ulp out, the last the closest
    any wrong variant comes.

    Mistral Large 4 tep4 (H = I = 6144, 32 of 128 experts, top-4, operands at the
    checkpoint's magnitudes and per-expert global scales folded into the
    reference's fp32 block scales) adds nothing the band has to absorb: its
    cells land at 2.38 elt / 0.903 rms over the decode graphs and 6.89-11.28
    elt / 0.727-0.781 rms at 32768 tokens. At 32768 tokens the element figure
    is a max over 201M elements: re-drawn over 56 fresh seeds there (layers 1,
    3, 45, 65 and 71, two of them with every slot in one window) it reached
    14.86, near this gate, while the RMS figure stayed at or below 0.792. Read
    the RMS gate. `test_sw_wrong_variants_land_outside_the_gate` measures the
    wrong readings a Mistral Large 4 loader could make: every one lands outside
    the RMS gate, all but one by more than 20 ulp.

    Mistral Large 4 dep4 (the same windows over up to 131072 gathered tokens, so
    an element figure over 805M elements) adds nothing either: its cells land
    at 9.34-12.97 elt / 0.731-0.780 rms at 131072 tokens and 1.50-7.47 elt /
    0.700-0.808 rms over the decode graphs at 4..64. Re-drawn over 132 fresh
    seeds at 131072 tokens (80 uniform over the four windows, 26 with every
    slot in one window, 26 under the padding skew) the element figure reached
    16.31 once -- past this gate on a correct kernel, as R1's 16.33 at its own
    top -- and at most 14.73 on the other 131, while the RMS figure stayed at
    or below 0.790. The cells' seeds are fixed; read the RMS gate.

    Mistral Large 4 tp4 cuts every expert to a 1536-wide slice of its
    intermediate, so FC2 contracts over 1536 terms instead of 6144, and the
    element figure grows while the RMS figure does not. The element tail is a
    marginal FC1-output value rounding to the neighbouring e2m1 code in one
    of kernel and reference: it moves an output element by one W2 term,
    while the row it is divided by is a sum over the contraction, so at a
    quarter of the width the same flip reads up to sqrt(6144 / 1536) = 2
    times larger (2 when one slot carries the token's weight, 1 when four
    slots carry a quarter each). Measured under one draw protocol at 32768
    tokens (layers 1, 3, 45, 65 and 71, fresh seeds): at I 1536 over all 128
    experts, 180 draws, element median 14.15, p90 18.82, max 25.42 -- 52 of
    them past 16, 2 past 24, none past 32; at I 6144 over 32-wide windows,
    90 draws, median 9.33, p90 12.11, max 14.85 -- a ratio of 1.5-1.7. The
    RMS figure stayed at or below 0.800 and 0.798. So the Mistral Large 4 drivers
    gate elements at 16 * sqrt(6144 / I) ulp (`_ml4_elt_ulp`): 16 at the
    whole width, as before for tep4 and dep4, and 32 at tp4's slice; the
    four slices summed are gated against the whole-width layer at 16. The
    RMS gate stays 2 everywhere. `test_sw_tp4_wrong_variants_land_outside_the_gate`
    measures the wrong readings at the slice: all land at 115 element ulp or
    more but one, the layer-45 scalar-role swap (25.1 elt, inside the element
    band, and 3.56 rms, outside the RMS gate), as at tep4.

    Mistral Large 4 tep8 (16-wide windows, every rank over all T tokens, so a
    window carries an eighth of the slots) adds nothing either: its cells
    land at 1.08-1.97 elt / 0.702-0.766 rms over the decode counts and
    7.20-10.50 elt / 0.732-0.783 rms at 32768 tokens, and the eight windows'
    partials summed at 9.93 / 0.875 in fp32 and 10.09 / 0.974 as a bf16
    running sum. Re-drawn over 48 fresh seeds at 32768 tokens (40 uniform
    over the eight windows, 8 with every slot in one window) the element
    figure reached 14.59 (median 9.10) and the RMS figure 0.805. Rank 7's
    decode graphs on window (112, 16) land at 0.98-6.43 elt / 0.708-1.453
    rms.

    At decode counts a 16-wide window has few live rows (1..13), and there
    the RMS figure is a small-sample one. Re-drawn over 5120 fresh steps
    (1..16 tokens, 40 per count at each of the eight windows) it passed
    this gate on 31, at most 4.30 (median 0.72), every one with 1..9 live
    rows, at seven of the eight windows, window 0 among them (window 112's
    640 reached 1.99); the element figure stayed at or below 8.44. The
    kernel is not what moves: against an fp64 twin of `reference` the same
    steps land the same (32 past 2 ulp), the fp32 and fp64 references
    differ by 3.55 rms on one step, and the FC1 output read out of the
    kernel on all 5120 (an identity down projection and
    `do_finalize=False`, as `test_intermediate_is_nvfp4_requantized` reads
    it) departs from the fp64 requantization only where a value sits within
    5.24e-5 relative of a rounding midpoint -- 992 e2m1 codes and 400 e4m3
    block scales, each the neighbour of the fp64 choice, over 161323008
    elements. One such flip on one of a few live rows is what crosses: the
    tep8 decode cell's own step at T=4 on sm_103 (one live row) lands at
    6.99 elt / 4.98 rms, its FC1 output read out of the kernel differing from
    the reference's in one e2m1 code of 6144, 4.94e-7 relative from the
    midpoint, and the reference's FC2 over the kernel's own FC1 output lands
    at 1.40 / 0.740. So the decode graph replays (`_ml4_replay`) gate the
    element figure alone and print the RMS one.

    Mistral Large 4 dep8 (16-wide windows over a call of at most 32768 of the
    eight ranks' gathered tokens, so a window carries an eighth of the
    slots) adds nothing either: its cells land at 1.57-5.76 elt /
    0.694-0.982 rms over the decode graphs at 8..128 on windows (0, 16) and
    (112, 16), 3.80 / 0.789 at 128 tokens and 7.62-9.41 elt / 0.733-0.785
    rms at 32768 tokens (the padding chunk, one live row, at 1.21 / 0.732),
    and the eight windows' partials sum at 9.97 / 0.866 in fp32 and 9.97 /
    0.966 as a bf16 running sum. Re-drawn over 36 fresh seeds at 32768
    tokens (32 uniform over the eight windows, 4 with every slot in one
    window) the element figure reached 12.31 and the RMS figure 0.799. At
    decode counts a 16-wide window has few live rows (1..73 of 8..128
    gathered tokens), where the RMS figure is a small-sample one: re-drawn
    over 2560 fresh steps (20 per count at each of the eight windows) it
    reached 1.98 (median 0.73) on a correct kernel, though none crossed this
    gate, while the element figure stayed at or below 10.27. The decode
    graph replays gate the element figure alone, as at tep8.

    Mistral Large 4 tp8 cuts every expert to a 768-wide slice of its
    intermediate, so FC2 contracts over 768 terms instead of 6144, and the
    element figure grows while the RMS figure does not. The element tail is a
    marginal FC1-output value rounding to the neighbouring e2m1 code in one
    of kernel and reference: it moves an output element by one W2 term,
    while the row it is divided by is a sum over the contraction, so at an
    eighth of the width the same flip reads up to sqrt(6144 / 768) = 2.83
    times larger. Measured under one draw protocol at 32768 tokens (layers
    1, 3, 45, 65 and 71 in turn, fresh seeds, every sixth draw with all four
    slots on experts 124..127): at I 768 over all 128 experts, 180 draws,
    element median 17.12, p90 23.54, p99 31.11, max 35.56 -- 110 of them
    past 16, 14 past 24, 1 past 32, none past 40; at I 6144 over 32-wide
    windows, 90 draws, median 9.33, p90 12.11, max 14.85 -- a ratio of
    1.8-2.4. The RMS figure stayed at or below 0.833 and 0.798. So the
    Mistral Large 4 drivers gate elements at 16 * sqrt(6144 / I) ulp
    (`_ml4_elt_ulp`): 16 at the whole width, as before for tep4 and dep4, and
    45.25 at tp8's slice; the eight slices summed are gated against the
    whole-width layer at 16. The RMS gate stays 2 everywhere.
    `test_sw_tp8_wrong_variants_land_outside_the_gate` measures the wrong
    readings at the slice: all land at 120 element ulp or more but one, the
    layer-45 scalar-role swap (34.5 elt, inside the element band, and 3.95
    rms, outside the RMS gate), as at tep4.

    At tp8 decode every token is live on every rank, so a step of T tokens
    gates T rows of 4 x 768 FC1 outputs each, and the RMS figure is a
    small-sample one: one FC1 output rounded to the neighbouring e2m1 code
    (or one block to the neighbouring e4m3 scale) moves the whole call's
    figure. Measured over fresh decode draws (layers 1 and 45, T 1..16, 160
    seeds each, one generator seed per draw, the cold fallback eager; a
    replay is bitwise the eager call): the RMS figure exceeds 2 on 21 of
    5120 (7 past 3, max 7.05, a single flipped element at T 3) while the
    element figure stays at or below 22.61. Against an fp64 twin of the
    reference the same 21 exceed it, and the fp32 and fp64 references differ
    from each other by at most 1.56 rms. Read out of the kernel through an
    identity down projection, every FC1-output difference from the fp64
    requantization is an adjacent rounding at a midpoint -- 676 e2m1 codes
    within 6.5e-5 relative and 352 e4m3 block scales within 2.0e-5, of
    134M elements, none non-adjacent -- and the 4205 draws with no such
    difference stay at or below 0.767 rms. So the 2 ulp RMS gate is not a
    bound for a correct kernel at decode counts at this width, nor at tep8's
    16-wide windows: the decode graph replays gate the element figure alone.
    """
    assert out.dtype == ref.dtype == torch.bfloat16, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    ulp = 2.0**-8
    row = ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    torch.testing.assert_close(out.float() / row, ref.float() / row, rtol=0.0, atol=elt_ulp * ulp)
    rel_rms = (
        (out.float() - ref.float()).pow(2).mean().sqrt()
        / ref.float().pow(2).mean().sqrt().clamp_min(1e-9)
    ).item() / ulp
    if rms_ulp is not None:
        assert rel_rms <= rms_ulp, f"relative RMS {rel_rms:.2f} ulp > {rms_ulp:g} ulp"


def _e2m1_rne(a: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even onto the e2m1 grid, saturating at +-6.

    The grid step is 0.5 below 2, 1 below 4 and 2 above, i.e. the binade step
    of e2m1's three exponents (the subnormal binade shares the step of the
    first normal one). `torch.round` is banker's rounding, which reproduces
    the format's ties-to-even-code rule: 0.25 -> 0, 0.75 -> 1, 1.25 -> 1,
    1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5.0 -> 4.
    """
    step = torch.where(a.abs() < 2.0, 0.5, torch.where(a.abs() < 4.0, 1.0, 2.0))
    return torch.sign(a) * torch.clamp(torch.round(a.abs() / step) * step, max=E2M1_MAX)


def _q_nvfp4(x: torch.Tensor, g: float) -> torch.Tensor:
    """NVFP4 quantize-dequantize of `x` under global scale `g`.

    Per 16 consecutive columns: `sf = e4m3(g * blockmax / 6)`,
    `data = e2m1(x * g / sf)`. Returns `data * sf`, i.e. the value a
    downstream block-scaled MMA multiplies — `g` times the reconstruction of
    `x`. Pinned bit-exactly against the kernel's FC1 epilogue by
    `test_intermediate_is_nvfp4_requantized`.
    """
    rows, cols = x.shape
    b = x.reshape(rows, cols // SV, SV)
    sf = (
        (g * b.abs().amax(dim=-1, keepdim=True) / E2M1_MAX)
        .clamp(max=E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .float()
    )
    out_scale = torch.where(sf == 0, torch.zeros_like(sf), g / sf)
    return (_e2m1_rne(b * out_scale) * sf).reshape(rows, cols)


def _rand_nvfp4(e: int, n: int, k: int, gen: torch.Generator):
    """Random NVFP4 expert stack.

    Returns `(packed [E, N, K/2] uint8, scales [E, N, K/16] float8_e4m3fn,
    codes [E, N, K] uint8)`. Codes and (power-of-two, hence exact) e4m3 scales
    are drawn directly, so the fp32 value of every weight is exact — the
    reference never has to model a quantizer.
    """
    codes = torch.randint(0, 16, (e, n, k), dtype=torch.uint8, device=DEV, generator=gen)
    exps = torch.randint(
        -2, 1, (e, n, k // SV), device=DEV, generator=gen, dtype=torch.int32
    ).float()
    sf = torch.exp2(exps).to(torch.float8_e4m3fn)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
    return packed, sf, codes


def _dequant(codes_e: torch.Tensor, sf_e: torch.Tensor) -> torch.Tensor:
    """One expert's `[N, K]` fp32 weight from its codes and e4m3 block scales."""
    return E2M1[codes_e.long()] * sf_e.float().repeat_interleave(SV, dim=1)


def _unpack_activation(data: torch.Tensor, sf: torch.Tensor, hidden: int, g: float):
    """Exact fp32 value of an NVFP4 activation pair, undoing the global scale.

    `data` is `[T, hidden/2]` uint8 (element 2i in the low nibble), `sf` a 1-D
    linear `[T * hidden/16]` e4m3 buffer.
    """
    rows = data.shape[0]
    codes = torch.stack([data & 0x0F, data >> 4], dim=-1).reshape(rows, hidden).long()
    scale = sf.view(torch.float8_e4m3fn).reshape(rows, hidden // SV).float()
    return E2M1[codes] * scale.repeat_interleave(SV, dim=1) / g


def _rand_activation(rows: int, hidden: int, g: float, gen: torch.Generator):
    """Random NVFP4 activation in the layout this op consumes.

    Returns `(data [rows, hidden/2] uint8, sf [rows * hidden/16] uint8, x
    [rows, hidden] fp32)`, `x` being the exact dequantization. Codes and
    (power-of-two) e4m3 scales are drawn directly, so no quantizer is modelled.
    """
    codes = torch.randint(0, 16, (rows, hidden), dtype=torch.uint8, device=DEV, generator=gen)
    exps = torch.randint(
        -2, 1, (rows, hidden // SV), device=DEV, generator=gen, dtype=torch.int32
    ).float()
    sf = torch.exp2(exps).to(torch.float8_e4m3fn)
    data = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    sf_flat = sf.reshape(-1).view(torch.uint8).contiguous()
    x = E2M1[codes.long()] * sf.float().repeat_interleave(SV, dim=1) / g
    return data, sf_flat, x


# ── kernel weight layout (pure torch) ─────────────────────────────────────


def _blk32_perm(m: int) -> torch.Tensor:
    """Gather index of the 32-row block shuffle: within each block of 32 rows,
    source row `4u + v` lands at destination row `8v + u`."""
    assert m % 32 == 0
    j = torch.arange(32)
    dst = (j % 4) * 8 + j // 4
    idx = torch.empty(32, dtype=torch.long)
    idx[dst] = j
    return (idx.repeat(m // 32) + torch.arange(m // 32).repeat_interleave(32) * 32).to(DEV)


def _gate_interleave_perm(m: int) -> torch.Tensor:
    """Gather index that interleaves the `[up | gate]` halves of a `2*I` row
    stack into `up0, gate0, up1, gate1, ...`."""
    p = torch.empty(m, dtype=torch.long)
    p[0::2] = torch.arange(0, m // 2)
    p[1::2] = torch.arange(m // 2, m)
    return p.to(DEV)


def _swizzle_scales(s: torch.Tensor) -> torch.Tensor:
    """Per-expert 128x4 block-scale swizzle, result viewed back as `[E, M, C]`.

    Flat destination of scale `(e, m, c)` is
    `e*M*C + (m//128)*512*(C//4) + (c//4)*512 + (m%32)*16 + ((m%128)//32)*4 + (c%4)`.
    """
    e, m, c = s.shape
    assert m % 128 == 0 and c % 4 == 0
    v = s.reshape(e, m // 128, 4, 32, c // 4, 4)
    v = v.permute(0, 1, 4, 3, 2, 5)
    return v.reshape(e, m, c).contiguous()


def _fc1_perm(rows: int) -> torch.Tensor:
    return _gate_interleave_perm(rows)[_blk32_perm(rows)]


def _prep_fc1(up_p, gt_p, up_s, gt_s, swizzle=True, shuffle=True):
    """Concat as `[up | gate]`, interleave + block-shuffle rows, swizzle scales."""
    w = torch.cat([up_p, gt_p], dim=1)
    s = torch.cat([up_s, gt_s], dim=1)
    if shuffle:
        perm = _fc1_perm(w.shape[1])
        w = torch.index_select(w, 1, perm)
        s = torch.index_select(s, 1, perm)
    s = s.view(torch.uint8)
    s = _swizzle_scales(s) if swizzle else s.contiguous()
    return w.contiguous(), s.view(torch.float8_e4m3fn).contiguous()


def _prep_fc2(dn_p, dn_s, swizzle=True, shuffle=True):
    """Block-shuffle the down projection's rows, swizzle its scales."""
    w, s = dn_p, dn_s
    if shuffle:
        perm = _blk32_perm(w.shape[1])
        w = torch.index_select(w, 1, perm)
        s = torch.index_select(s, 1, perm)
    s = s.view(torch.uint8)
    s = _swizzle_scales(s) if swizzle else s.contiguous()
    return w.contiguous(), s.view(torch.float8_e4m3fn).contiguous()


def _build(num_experts: int, hidden: int, inter: int, seed: int):
    """Build one MoE layer: kernel-ready tensors plus the reference operands."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    up_p, up_s, up_c = _rand_nvfp4(num_experts, inter, hidden, gen)
    gt_p, gt_s, gt_c = _rand_nvfp4(num_experts, inter, hidden, gen)
    dn_p, dn_s, dn_c = _rand_nvfp4(num_experts, hidden, inter, gen)
    w1, s1 = _prep_fc1(up_p, gt_p, up_s, gt_s)
    w2, s2 = _prep_fc2(dn_p, dn_s)
    args = dict(
        gemm1_weights=w1,
        gemm1_weights_scale=s1,
        gemm2_weights=w2,
        gemm2_weights_scale=s2,
        intermediate_size=inter,
    )
    ref = dict(
        up=(up_c, up_s),
        gate=(gt_c, gt_s),
        down=(dn_c, dn_s),
        raw=(up_p, gt_p, up_s, gt_s, dn_p, dn_s),
        hidden=hidden,
        inter=inter,
        num_experts=num_experts,
        gen=gen,
    )
    return args, ref


def _scalars(num_local: int, g1: float, g2: float):
    """The three per-expert fp32 scalars, for weight global scales of 1.

    `output1_scale_gate_scalar = alpha1` dequantizes the FC1 accumulator;
    `output1_scale_scalar = g2 * alpha1` additionally applies the FC2-input
    global scale to the linear half; `output2_scale_scalar = alpha2`
    dequantizes the FC2 accumulator.
    """
    a1 = torch.full((num_local,), 1.0 / g1, dtype=torch.float32, device=DEV)
    a2 = torch.full((num_local,), 1.0 / g2, dtype=torch.float32, device=DEV)
    return a1 * g2, a1, a2


def _call(data, sf, args, num_experts, top_k, scale_as_is=False, **kw):
    """Invoke the wrapper with this layer's tensors and size scalars.

    `sf` is handed over as `float8_e4m3fn`, the dtype the op demands, unless
    `scale_as_is` asks for the raw tensor (used by the dtype negative tests).
    """
    sf = sf if scale_as_is else sf.view(torch.float8_e4m3fn)
    kw.setdefault("routing_logits", None)
    kw.setdefault("routing_bias", None)
    kw.setdefault("gemm1_bias", None)
    kw.setdefault("gemm1_alpha", None)
    kw.setdefault("gemm1_beta", None)
    kw.setdefault("gemm1_clamp_limit", None)
    kw.setdefault("gemm2_bias", None)
    kw.setdefault("n_group", None)
    kw.setdefault("topk_group", None)
    kw.setdefault("local_expert_offset", 0)
    kw.setdefault("local_num_experts", num_experts)
    kw.setdefault("routed_scaling_factor", None)
    kw.setdefault("routing_method_type", 1)
    kw.setdefault("do_finalize", True)
    kw.setdefault("act_type", 0)
    kw.setdefault("intermediate_size", args["intermediate_size"])
    for k in (
        "gemm1_weights",
        "gemm1_weights_scale",
        "gemm2_weights",
        "gemm2_weights_scale",
    ):
        kw.setdefault(k, args[k])
    with validating(moe):
        return moe(
            kw.pop("routing_logits"),
            kw.pop("routing_bias"),
            data,
            sf,
            kw.pop("gemm1_weights"),
            kw.pop("gemm1_weights_scale"),
            kw.pop("gemm1_bias"),
            kw.pop("gemm1_alpha"),
            kw.pop("gemm1_beta"),
            kw.pop("gemm1_clamp_limit"),
            kw.pop("gemm2_weights"),
            kw.pop("gemm2_weights_scale"),
            kw.pop("gemm2_bias"),
            kw.pop("output1_scale_scalar"),
            kw.pop("output1_scale_gate_scalar"),
            kw.pop("output2_scale_scalar"),
            num_experts,
            top_k,
            kw.pop("n_group"),
            kw.pop("topk_group"),
            kw.pop("intermediate_size"),
            kw.pop("local_expert_offset"),
            kw.pop("local_num_experts"),
            kw.pop("routed_scaling_factor"),
            kw.pop("routing_method_type"),
            kw.pop("do_finalize"),
            kw.pop("act_type"),
            **kw,
        )


# ── reference ─────────────────────────────────────────────────────────────


def _ref_moe(
    x,
    ids,
    scales,
    ref,
    g2: float,
    offset: int = 0,
    num_local: int | None = None,
    swap_gate_up: bool = False,
    swap_scalars: bool = False,
    quantize_intermediate: bool = True,
):
    """Native-torch MoE over dequantized NVFP4 weights, fp32 throughout.

    `ids` carries global expert ids; this rank answers for
    `[offset, offset + num_local)`. `scales[t, j]` multiplies slot `j`'s expert
    output; nothing is renormalized. The FC1 activation is requantized to
    NVFP4 under global scale `g2` before FC2, which is what the kernel does.
    """
    num_tokens, hidden = x.shape
    up_c, up_s = ref["up"]
    gt_c, gt_s = ref["gate"]
    dn_c, dn_s = ref["down"]
    if swap_gate_up:
        (up_c, up_s), (gt_c, gt_s) = (gt_c, gt_s), (up_c, up_s)
    num_local = up_c.shape[0] if num_local is None else num_local
    out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)
    for local_e in range(num_local):
        tok, slot = (ids == offset + local_e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        xe = x[tok]
        up = xe @ _dequant(up_c[local_e], up_s[local_e]).t()
        gate = xe @ _dequant(gt_c[local_e], gt_s[local_e]).t()
        if swap_scalars:
            up, gate = up / g2, gate * g2
        act = up * gate * torch.sigmoid(gate)
        # the FC1 epilogue emits g2 * act as NVFP4; FC2's alpha divides it back
        act = _q_nvfp4(act, g2) / g2 if quantize_intermediate else act
        y = act @ _dequant(dn_c[local_e], dn_s[local_e]).t()
        out.index_add_(0, tok, y * scales[tok, slot].float().unsqueeze(1))
    return out


def _calibrate_g2(x, ids, ref) -> float:
    """`448*6 / amax` of the true FC1 activation — the FC2 input global scale a
    checkpoint's `down_proj.input_scale` encodes."""
    up_c, up_s = ref["up"]
    gt_c, gt_s = ref["gate"]
    amax = 0.0
    for e in range(up_c.shape[0]):
        tok = (ids == e).nonzero(as_tuple=True)[0].unique()
        if tok.numel() == 0:
            continue
        xe = x[tok]
        up = xe @ _dequant(up_c[e], up_s[e]).t()
        gate = xe @ _dequant(gt_c[e], gt_s[e]).t()
        amax = max(amax, (up * gate * torch.sigmoid(gate)).abs().max().item())
    return E4M3_MAX * E2M1_MAX / amax


def _dev(y: torch.Tensor, ref: torch.Tensor):
    """(worst element deviation, relative RMS deviation), both in bf16 ulp."""
    o, r = y.float(), ref.float()
    row = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    elt = ((o - r).abs() / row).max().item() / ULP
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-9)).item() / ULP
    return elt, rms


def _assert_moe_close(y: torch.Tensor, ref: torch.Tensor) -> None:
    """Two gates: per-element, row-scaled; and aggregate relative RMS.

    Kernel and reference consume bit-identical NVFP4 weights and activations
    and model the same FC1-output NVFP4 requantization, so they differ only in
    accumulation order and in whether a marginal FC1 value rounds to the same
    e2m1 code — and an e2m1 code is a *coarse* step (2 mantissa bits), so one
    flipped intermediate element moves the output row by a visible fraction.
    Default `assert_close` tolerances cannot express that: their bf16
    `atol=1e-5` sits three orders of magnitude below one output ulp of a
    two-GEMM chain, and per-element `rtol` is meaningless where cancellation
    drives `|ref|` to zero. So the element gate is 16 ulp of the row's largest
    magnitude and the aggregate gate is 2 ulp of relative RMS. Worst values
    measured over every configuration covered here: 9.97 ulp element-wise (an
    18-wide expert-parallel window at 8192 tokens, H=2560, I=1536) and 0.88
    ulp RMS (those four windows summed); a full-window call sits at 4.75 /
    0.72. The DeepSeek-R1 routed geometry (H=7168, I=2048, 256 experts,
    top-8) lands just inside both: 9.62 / 0.74 for a 64-wide expert-parallel
    window, 4.68 / 0.70 for the 256-expert stack, 5.02 / 0.85 for the four
    windows summed. Two things inflate the element figure and not the RMS
    one: it is an extreme order statistic over every element compared, and it
    is normalized by the *reference's own* row magnitude, so a window carrying
    a quarter of the routed slots divides a similar absolute error by a
    smaller row.
    Re-drawn over 144 independent (seed, window) comparisons at T=1024/4096 it
    reached 12.0 while the RMS figure stayed at 0.86 — headroom on this
    element gate is ~1.3x for a window, ~2.3x on the RMS gate. Re-drawn over
    16 fresh seeds at T=8192 (16 full-stack + 64 window comparisons) it
    reached 16.33 for a window (p90 9.82) and 10.41 for the full stack, while
    the RMS figure was unchanged from T=4096 (window 0.74, full 0.70, sum
    0.85): the element metric is a max over T*H elements, so its tail grows
    with T at constant accuracy and *crosses* this gate at the top of the
    certified range. Both of those re-draws are the H=2560 / I=1536 geometry;
    re-drawn the same way at H=7168 / I=2048 over 12 fresh weight+activation
    seeds the tail is *lower* — max 12.74 for a 64-wide window at T=8192 (p90
    8.40, mean 6.73), 12.15 at T=4096, and 7.11 / 6.42 for the 256-expert
    stack, with the RMS figures flat at 0.74 (window) / 0.70 (full) / 0.86
    (sum) at both token counts. The seeded draws shipped here land at 9.97 /
    3.20 (H=2560) and 9.62 / 4.68 (H=7168), but only the RMS gate is
    scale-free — read it, not the element one, when sizing a caller's
    tolerance.
    The many-small-experts geometry (E=512, H=2560, I=640, top-10) is the
    least strained of the three: the seeded sweep in
    `test_many_experts_top10_geometry` tops out at 3.82 elt / 0.70 rms over
    the whole token column (3.82 / 0.67 at T=8192), and re-drawn over 12 fresh
    weight+activation+routing seeds the element tail reaches 10.67 at T=4096
    (p90 4.36, mean 3.57) and 5.03 at T=8192 (p90 3.60, mean 3.20), with the
    RMS figure between 0.670 and 0.694 at both counts. Ten renormalised slot
    weights summing to 1 are not what holds it: the same geometry with uniform
    [0, 1) weights lands at 5.39 elt / 0.67 rms over the two counts that test
    repeats.
    Both gates were sized against deliberately wrong computations when they
    were set. At H=256/I=128 and at H=7168/I=2048 respectively: a gate/up
    swap (263 / 278 ulp), a scalar-role swap (177 / 48), a missing scale
    swizzle (408 / 357), a
    missing row shuffle (508 / 432), and a reference that skips the FC1-output
    requantization (43 / 28 ulp element, 23 / 23 ulp RMS); the smallest RMS
    distance any of those reaches is 39.8, at the R1 shape's scalar-role swap.
    `test_many_experts_top10_geometry` measures the same five at E=512 /
    H=2560 / I=640 and asserts each is more than 10 ulp RMS out: 223, 62, 265,
    362 and 23 respectively, the last being the closest any wrong variant comes
    anywhere in this file.
    `test_ep_window_18_of_72` and `test_ep_window_64_of_256` add a mis-set
    expert-parallel window (304-363 ulp RMS) and one window left out of the
    sum (>=124 ulp RMS overall, >=128 at T=8192).
    """
    assert y.dtype == ref.dtype == torch.bfloat16, (y.dtype, ref.dtype)
    assert y.shape == ref.shape, (y.shape, ref.shape)
    row = ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    torch.testing.assert_close(y.float() / row, ref.float() / row, rtol=0.0, atol=16 * ULP)
    _, rms = _dev(y, ref)
    assert rms <= 2.0, f"relative RMS {rms:.2f} ulp > 2 ulp"


def _routing(num_tokens, num_experts, top_k, gen):
    """Random ids/weights for the pre-routed entry point."""
    ids = torch.stack(
        [torch.randperm(num_experts, device=DEV, generator=gen)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    wts = torch.rand(num_tokens, top_k, device=DEV, generator=gen).to(torch.bfloat16)
    return ids, wts


# DeepSeek-V3-Lite routed-expert geometry, built once and shared.
_DSV3 = None


def _dsv3():
    global _DSV3
    if _DSV3 is None:
        _DSV3 = _build(72, 2560, 1536, seed=100)
    return _DSV3


# DeepSeek-R1 routed-expert geometry, built once and shared: 256 experts,
# H = 7168, I = 2048. The kernel-ready stack alone is ~6.3 GiB (FC1
# [256, 4096, 3584] + [256, 4096, 448], FC2 [256, 7168, 1024] +
# [256, 7168, 128]) and the reference operands take it to ~22 GiB.
_DSR1 = None


def _dsr1():
    global _DSR1
    if _DSR1 is None:
        _DSR1 = _build(256, 7168, 2048, seed=256)
    return _DSR1


# The token column certified at the R1 geometry. 8192 is trtllm's default
# `max_num_tokens`; a dep4 target gathers four ranks' tokens before the expert
# call, so its own T reaches 4 * 8192 and it must chunk down into calls this
# column covers.
_R1_TOKENS = (1, 2, 3, 5, 7, 8, 16, 24, 32, 33, 40, 64, 128, 256, 1024, 4096, 8192)


# qwen4_exp's many-small-experts routed geometry, built once and shared: 512
# experts, H = 2560, I = 640, top-10. Three axes at once that neither DeepSeek
# geometry reaches -- 512 experts, a top_k past 8, and an intermediate width of
# 640 -- with every alignment rule still satisfied (H % 256 == 0, 2*I = 1280 %
# 128 == 0, H/16 = 160 % 4 == 0, I/16 = 40 % 4 == 0), so nothing is padded. The
# kernel-ready stack is ~1.4 GiB (FC1 [512, 1280, 1280] + [512, 1280, 160], FC2
# [512, 2560, 320] + [512, 2560, 40]) and the reference operands take it to
# ~5 GiB.
_MANY = None


def _many_experts():
    global _MANY
    if _MANY is None:
        _MANY = _build(512, 2560, 640, seed=512640)
    return _MANY


# The token column swept at that geometry. The pre-routed path picks its
# permutation kernel by row count (`runPostTopKPipeline` in
# `cpp/tensorrt_llm/kernels/moe/trtllmGen/routing/RoutingFromTopKIds.cu`): the
# static-block kernel takes 4 rows and below, the dynamic-block one 5..12 rows
# (it reaches 1024 experts on this path, so 512 is inside it), and the
# single-cluster one everything from 13 rows up to 8192 -- read back from the
# launched kernels at 512 experts / top-10 on sm_103. So the column brackets
# the 4/5 boundary as well as the decode and prefill ends.
_MANY_TOKENS = (1, 2, 3, 4, 5, 8, 16, 64, 256, 1024, 4096, 8192)


def _renorm_routing(num_tokens, num_experts, top_k, gen):
    """Pre-routed ids/weights in the form a renormalising router emits.

    Softmax over every expert, top-k of that, and the k weights rescaled to sum
    to 1 -- the `norm_topk_prob` convention. `torch.topk` returns distinct
    indices, which is what this op requires of a row.
    """
    logits = torch.randn(num_tokens, num_experts, device=DEV, dtype=torch.float32, generator=gen)
    w, idx = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    return idx.to(torch.int32).contiguous(), w.to(torch.bfloat16).contiguous()


# ── tests ─────────────────────────────────────────────────────────────────


def test_layout_helpers_match():
    """The trtllm preprocessing ops reproduce the pure-torch layout exactly.

    The middle two scale shapes are the DeepSeek-R1 routed geometry's, at
    H = 7168 / I = 2048: FC1 `[E, 2*I, H/16] = [E, 4096, 448]` and FC2
    `[E, H, I/16] = [E, 7168, 128]`; the next two are Mistral Large 4's, at
    H = I = 6144: FC1 `[E, 12288, 384]` and FC2 `[E, 6144, 384]`; the next two
    are Mistral Large 4 tp4's 1536-wide slice: FC1 `[E, 3072, 384]` and FC2
    `[E, 6144, 96]`; the last two tp8's 768-wide slice: FC1 `[E, 1536, 384]`
    and FC2 `[E, 6144, 48]`. All satisfy the swizzle's `M % 128 == 0` and `C % 4 == 0`,
    so `block_scale_interleave` returns exactly `E*M*C` bytes and reshapes
    back — no padding enters either operand at these geometries.
    """
    gen = torch.Generator(device=DEV).manual_seed(201)
    shapes = (
        (256, 12),
        (4096, 448),
        (7168, 128),
        (12288, 384),
        (6144, 384),
        (3072, 384),
        (6144, 96),
        (1536, 384),
        (6144, 48),
    )
    for rows, cols in shapes:
        x = torch.randint(0, 256, (4, rows, cols), dtype=torch.uint8, device=DEV, generator=gen)
        perm = _fc1_perm(rows)
        for e in range(x.shape[0]):
            assert torch.equal(
                torch.ops.trtllm.shuffle_matrix(x[e].contiguous(), perm),
                torch.index_select(x[e], 0, perm),
            ), f"shuffle_matrix is not a plain row gather at {rows}x{cols}"
        swz = torch.ops.trtllm.block_scale_interleave(x)
        assert swz.numel() == x.numel(), (
            f"block_scale_interleave padded {rows}x{cols}: {swz.numel()} bytes "
            f"for {x.numel()} scales"
        )
        assert torch.equal(swz.reshape(x.shape), _swizzle_scales(x)), (
            f"block_scale_interleave does not match the documented 128x4 swizzle at {rows}x{cols}"
        )
    print("  test_layout_helpers_match OK")


def test_deepseek_geometry():
    """E=72, H=2560, I=1536, top-6: decode- through prefill-sized batches.

    The top of the sweep is 8192 rows in one call — trtllm's default
    `max_num_tokens`, i.e. the widest prefill a stock engine hands this op
    without chunking. Token counts are appended in ascending order, so every
    count below the top draws exactly the values it drew before.
    """
    args, ref = _dsv3()
    gen = ref["gen"]
    worst = (0.0, 0.0)
    token_counts = (1, 2, 8, 64, 256, 1024, 4096, 8192)
    for num_tokens in token_counts:
        g1 = 137.0
        data, sf, x = _rand_activation(num_tokens, 2560, g1, gen)
        ids, wts = _routing(num_tokens, 72, 6, gen)
        g2 = _calibrate_g2(x, ids, ref)
        s1, sg, s2 = _scalars(72, g1, g2)
        out = _call(
            data,
            sf,
            args,
            72,
            6,
            topk_ids=ids,
            topk_weights=wts,
            output1_scale_scalar=s1,
            output1_scale_gate_scalar=sg,
            output2_scale_scalar=s2,
        )
        assert len(out) == 1, len(out)
        y = out[0]
        assert y.shape == (num_tokens, 2560) and y.dtype == torch.bfloat16, y.shape
        exp = _ref_moe(x, ids, wts, ref, g2).to(torch.bfloat16)
        _assert_moe_close(y, exp)
        top = _dev(y, exp)  # after the loop: the largest token count's numbers
        worst = tuple(max(a, b) for a, b in zip(worst, top))
    print(
        f"  test_deepseek_geometry OK (worst {worst[0]:.2f} elt / {worst[1]:.2f} rms ulp; "
        f"at T={token_counts[-1]} {top[0]:.2f} elt / {top[1]:.2f} rms ulp)"
    )


def test_other_geometries():
    """Smaller expert counts, hidden and intermediate sizes, and top_k values."""
    worst = (0.0, 0.0)
    for num_experts, hidden, inter, top_k, tokens in (
        (8, 256, 128, 2, (1, 3, 16, 128)),
        (4, 512, 256, 1, (1, 7, 64)),
        (16, 256, 64, 4, (2, 33)),
        (2, 256, 512, 1, (5,)),
        (3, 768, 192, 2, (1, 40)),
    ):
        args, ref = _build(num_experts, hidden, inter, seed=7 * num_experts + hidden)
        gen = ref["gen"]
        for num_tokens in tokens:
            g1 = 71.0
            data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
            ids, wts = _routing(num_tokens, num_experts, top_k, gen)
            g2 = _calibrate_g2(x, ids, ref)
            s1, sg, s2 = _scalars(num_experts, g1, g2)
            y = _call(
                data,
                sf,
                args,
                num_experts,
                top_k,
                topk_ids=ids,
                topk_weights=wts,
                output1_scale_scalar=s1,
                output1_scale_gate_scalar=sg,
                output2_scale_scalar=s2,
            )[0]
            assert y.shape == (num_tokens, hidden), y.shape
            exp = _ref_moe(x, ids, wts, ref, g2).to(torch.bfloat16)
            _assert_moe_close(y, exp)
            worst = tuple(max(a, b) for a, b in zip(worst, _dev(y, exp)))
    print(f"  test_other_geometries OK (worst {worst[0]:.2f} elt / {worst[1]:.2f} rms ulp)")


def test_intermediate_is_nvfp4_requantized():
    """Read the FC1 epilogue's output straight out of the kernel.

    An identity down projection with unit block scales and
    `output2_scale_scalar = 1` makes the combined output equal the requantized
    intermediate value element for element, and e2m1 x e4m3 products are exact
    in bf16 — so the comparison is bit-exact, not toleranced.
    """
    num_experts, hidden, top_k, num_tokens = 4, 256, 1, 64
    gen = torch.Generator(device=DEV).manual_seed(77)
    up_p, up_s, up_c = _rand_nvfp4(num_experts, hidden, hidden, gen)
    gt_p, gt_s, gt_c = _rand_nvfp4(num_experts, hidden, hidden, gen)
    w1, s1 = _prep_fc1(up_p, gt_p, up_s, gt_s)
    eye = torch.zeros(num_experts, hidden, hidden, dtype=torch.uint8, device=DEV)
    d = torch.arange(hidden, device=DEV)
    eye[:, d, d] = 2  # e2m1 code 2 == +1.0
    dn_p = (eye[..., 0::2] | (eye[..., 1::2] << 4)).contiguous()
    dn_s = torch.ones(num_experts, hidden, hidden // SV, device=DEV).to(torch.float8_e4m3fn)
    w2, s2 = _prep_fc2(dn_p, dn_s)

    g1 = 137.0
    data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
    ids = torch.randint(
        0, num_experts, (num_tokens, 1), dtype=torch.int32, device=DEV, generator=gen
    )
    wts = torch.ones(num_tokens, 1, dtype=torch.bfloat16, device=DEV)

    act = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=DEV)
    for e in range(num_experts):
        tok = (ids[:, 0] == e).nonzero(as_tuple=True)[0]
        if tok.numel() == 0:
            continue
        xe = x[tok]
        up = xe @ _dequant(up_c[e], up_s[e]).t()
        gate = xe @ _dequant(gt_c[e], gt_s[e]).t()
        act[tok] = up * gate * torch.sigmoid(gate)
    g2 = E4M3_MAX * E2M1_MAX / act.abs().max().item()
    a1 = torch.full((num_experts,), 1.0 / g1, dtype=torch.float32, device=DEV)
    y = _call(
        data,
        sf,
        dict(
            gemm1_weights=w1,
            gemm1_weights_scale=s1,
            gemm2_weights=w2,
            gemm2_weights_scale=s2,
            intermediate_size=hidden,
        ),
        num_experts,
        top_k,
        topk_ids=ids,
        topk_weights=wts,
        output1_scale_scalar=a1 * g2,
        output1_scale_gate_scalar=a1,
        output2_scale_scalar=torch.ones(num_experts, dtype=torch.float32, device=DEV),
    )[0].float()

    exact = _q_nvfp4(act, g2)
    assert torch.equal(y, exact), (
        f"FC1 epilogue is not sf=e4m3(g*amax/6), data=e2m1(g*x/sf): "
        f"{(y != exact).sum().item()} of {y.numel()} elements differ"
    )
    # an unrounded block scale and the unquantized activation both fail here
    b = (act * g2).reshape(num_tokens, hidden // SV, SV)
    sfx = b.abs().amax(-1, keepdim=True) / E2M1_MAX
    unrounded = (_e2m1_rne(b / sfx.clamp_min(1e-30)) * sfx).reshape(num_tokens, hidden)
    fracs = {}
    for name, cand in (
        ("no requantization", act * g2),
        ("exact (unrounded) scale", unrounded),
    ):
        fracs[name] = (y == cand).float().mean().item()
        assert fracs[name] < 0.9, f"{name} also matches {fracs[name]:.2%} — no discrimination"
    print(
        f"  test_intermediate_is_nvfp4_requantized OK (bit-exact on all "
        f"{y.numel()} elements; unrounded scale {fracs['exact (unrounded) scale']:.1%}, "
        f"no requantization {fracs['no requantization']:.1%})"
    )


def test_fp4_quantize_pairing():
    """`fp4_quantize(x, g, 16, False, False)` is the activation pairing.

    The linear scale layout is required; the swizzled one is rejected by size
    except when the byte counts coincide, where it is silently wrong.
    """
    num_experts, hidden, inter, top_k = 8, 256, 128, 2
    args, ref = _build(num_experts, hidden, inter, seed=404)
    gen = ref["gen"]
    for num_tokens in (1, 7, 128):
        xb = torch.randn(num_tokens, hidden, device=DEV, dtype=torch.bfloat16, generator=gen)
        g1 = E4M3_MAX * E2M1_MAX / xb.abs().max().float().item()
        gs = torch.tensor([g1], dtype=torch.float32, device=DEV)
        data, sf = torch.ops.trtllm.fp4_quantize(xb, gs, SV, False, False)
        assert data.shape == (num_tokens, hidden // 2) and data.dtype == torch.uint8
        assert sf.shape == (num_tokens * hidden // SV,) and sf.dtype == torch.uint8
        x = _unpack_activation(data, sf, hidden, g1)
        ids, wts = _routing(num_tokens, num_experts, top_k, gen)
        g2 = _calibrate_g2(x, ids, ref)
        s1, sg, s2 = _scalars(num_experts, g1, g2)
        kw = dict(
            topk_ids=ids,
            topk_weights=wts,
            output1_scale_scalar=s1,
            output1_scale_gate_scalar=sg,
            output2_scale_scalar=s2,
        )
        y = _call(data, sf, args, num_experts, top_k, **kw)[0]
        _assert_moe_close(y, _ref_moe(x, ids, wts, ref, g2).to(torch.bfloat16))

        _, swz = torch.ops.trtllm.fp4_quantize(xb, gs, SV, False, True)
        if swz.numel() == sf.numel():
            assert num_tokens % 128 == 0, num_tokens
            bad = _call(data, swz, args, num_experts, top_k, **kw)[0]
            elt, _ = _dev(bad, y.float())
            assert elt > 25.0, f"swizzled scales only {elt:.1f} ulp off"
            print(f"    T={num_tokens}: swizzled scale buffer accepted, {elt:.0f} ulp wrong")
        else:
            try:
                _call(data, swz, args, num_experts, top_k, **kw)
                raise AssertionError("swizzled scale buffer of a different size accepted")
            except RuntimeError as e:
                assert "hidden_states_scale has incorrect size" in str(e), e
    print("  test_fp4_quantize_pairing OK")


def test_scalars_and_optional_tensors():
    """`None` is accepted for the five schema-mandatory bias/GLU tensors, and is
    exactly their neutral value; the three scale scalars are per local expert."""
    num_experts, hidden, inter, top_k, num_tokens = 8, 256, 128, 2, 16
    args, ref = _build(num_experts, hidden, inter, seed=505)
    gen = ref["gen"]
    g1, g2 = 137.0, 1.0 / 32.0
    data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    s1, sg, s2 = _scalars(num_experts, g1, g2)
    kw = dict(
        topk_ids=ids,
        topk_weights=wts,
        output1_scale_scalar=s1,
        output1_scale_gate_scalar=sg,
        output2_scale_scalar=s2,
    )
    base = _call(data, sf, args, num_experts, top_k, **kw)[0]

    def f32(shape, value):
        return torch.full(shape, value, dtype=torch.float32, device=DEV)

    neutral = (
        ("gemm1_bias", f32((num_experts, 2 * inter), 0.0)),
        ("gemm2_bias", f32((num_experts, hidden), 0.0)),
        ("gemm1_alpha", f32((num_experts,), 1.0)),
        ("gemm1_beta", f32((num_experts,), 0.0)),
        ("gemm1_clamp_limit", f32((num_experts,), 1e9)),
    )
    for name, t in neutral:
        y = _call(data, sf, args, num_experts, top_k, **kw, **{name: t})[0]
        assert torch.equal(y, base), f"{name} at its neutral value differs from None"
    # ...and each slot is genuinely live: a non-neutral value changes the result
    live = (
        ("gemm1_bias", f32((num_experts, 2 * inter), 1e4)),
        ("gemm2_bias", f32((num_experts, hidden), 1e2)),
        ("gemm1_alpha", f32((num_experts,), 8.0)),
        ("gemm1_beta", f32((num_experts,), 50.0)),
        ("gemm1_clamp_limit", f32((num_experts,), 1e-3)),
    )
    for name, t in live:
        y = _call(data, sf, args, num_experts, top_k, **kw, **{name: t})[0]
        assert not torch.equal(y, base), f"{name} is silently ignored"

    for name in (
        "output1_scale_scalar",
        "output1_scale_gate_scalar",
        "output2_scale_scalar",
    ):
        for bad, msg in (
            (kw[name][:-1].contiguous(), "incorrect dim 0"),
            (kw[name].double(), "must be float"),
        ):
            try:
                _call(data, sf, args, num_experts, top_k, **{**kw, name: bad})
                raise AssertionError(f"{name} {msg} accepted")
            except RuntimeError as e:
                assert "scalar" in str(e) and msg in str(e), (name, msg, str(e))
    print("  test_scalars_and_optional_tensors OK")


def test_expert_window_and_ids():
    """local_expert_offset / local_num_experts, out-of-window ids, duplicates."""
    num_experts, hidden, inter, top_k, num_tokens = 8, 256, 128, 2, 24
    args, ref = _build(num_experts, hidden, inter, seed=606)
    gen = ref["gen"]
    g1, g2 = 137.0, 1.0 / 32.0
    data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    s1, sg, s2 = _scalars(num_experts, g1, g2)

    off, loc = 4, 4
    y = _call(
        data,
        sf,
        args,
        num_experts,
        top_k,
        topk_ids=ids,
        topk_weights=wts,
        local_expert_offset=off,
        local_num_experts=loc,
        gemm1_weights=args["gemm1_weights"][off:].contiguous(),
        gemm1_weights_scale=args["gemm1_weights_scale"][off:].contiguous(),
        gemm2_weights=args["gemm2_weights"][off:].contiguous(),
        gemm2_weights_scale=args["gemm2_weights_scale"][off:].contiguous(),
        output1_scale_scalar=s1[off:].contiguous(),
        output1_scale_gate_scalar=sg[off:].contiguous(),
        output2_scale_scalar=s2[off:].contiguous(),
    )[0]
    sub = dict(ref)
    sub["up"] = (ref["up"][0][off:], ref["up"][1][off:])
    sub["gate"] = (ref["gate"][0][off:], ref["gate"][1][off:])
    sub["down"] = (ref["down"][0][off:], ref["down"][1][off:])
    exp = _ref_moe(x, ids, wts, sub, g2, offset=off, num_local=loc).to(torch.bfloat16)
    _assert_moe_close(y, exp)

    kw = dict(
        topk_weights=wts,
        output1_scale_scalar=s1,
        output1_scale_gate_scalar=sg,
        output2_scale_scalar=s2,
    )
    # ids outside [offset, offset+local) are dropped, negative and >= num_experts too
    for bad_id in (-1, num_experts, num_experts + 5):
        mixed = ids.clone()
        mixed[:, 0] = bad_id
        dropped = ids.clone()
        dropped[:, 0] = -1
        a = _call(data, sf, args, num_experts, top_k, topk_ids=mixed, **kw)[0]
        b = _call(data, sf, args, num_experts, top_k, topk_ids=dropped, **kw)[0]
        assert torch.equal(a, b), f"expert id {bad_id} was not dropped"

    # A repeated id inside one row is accepted, but whether it contributes once
    # or twice is not stable: with this geometry it collapsed to the first slot
    # at 16 tokens and contributed both slots at 24. Pin the disjunction, and
    # which branch each token count took.
    seen = set()
    for tokens in (16, 24):
        d16, s16, x16 = _rand_activation(tokens, hidden, g1, gen)
        i16, w16 = _routing(tokens, num_experts, top_k, gen)
        dup = i16.clone()
        dup[:, 1] = dup[:, 0]
        drop = torch.full_like(dup[:, 0], -1)
        kw16 = dict(kw, topk_weights=w16)
        y = _call(d16, s16, args, num_experts, top_k, topk_ids=dup, **kw16)[0]
        slot0 = _call(
            d16,
            s16,
            args,
            num_experts,
            top_k,
            topk_ids=torch.stack([dup[:, 0], drop], 1).contiguous(),
            **kw16,
        )[0]
        slot1 = _call(
            d16,
            s16,
            args,
            num_experts,
            top_k,
            topk_ids=torch.stack([drop, dup[:, 1]], 1).contiguous(),
            **kw16,
        )[0]
        both = (slot0.float() + slot1.float()).to(torch.bfloat16)
        branch = (
            "once"
            if torch.equal(y, slot0)
            else ("twice" if _dev(y, both.float())[0] < 2.0 else None)
        )
        assert branch is not None, (
            f"a duplicated expert id at {tokens} tokens matched neither one nor two "
            f"contributions ({_dev(y, slot0.float())[0]:.1f} / {_dev(y, both.float())[0]:.1f} ulp)"
        )
        seen.add((tokens, branch))
    print(f"  test_expert_window_and_ids OK (duplicate ids: {sorted(seen)})")


def _window_args(args, off: int, num_local: int):
    """The kernel-ready weight operands of one expert-parallel window."""
    return dict(
        gemm1_weights=args["gemm1_weights"][off : off + num_local].contiguous(),
        gemm1_weights_scale=args["gemm1_weights_scale"][off : off + num_local].contiguous(),
        gemm2_weights=args["gemm2_weights"][off : off + num_local].contiguous(),
        gemm2_weights_scale=args["gemm2_weights_scale"][off : off + num_local].contiguous(),
        intermediate_size=args["intermediate_size"],
    )


def _window_ref(ref, off: int, num_local: int):
    """The reference operands of one expert-parallel window."""
    sub = dict(ref)
    for k in ("up", "gate", "down"):
        sub[k] = (ref[k][0][off : off + num_local], ref[k][1][off : off + num_local])
    return sub


def _window_scalars(s1, sg, s2, off: int, num_local: int):
    """The three per-expert fp32 scalars restricted to one window."""
    return dict(
        output1_scale_scalar=s1[off : off + num_local].contiguous(),
        output1_scale_gate_scalar=sg[off : off + num_local].contiguous(),
        output2_scale_scalar=s2[off : off + num_local].contiguous(),
    )


def test_ep_window_18_of_72():
    """The tep4 expert-parallel split: 72 routed experts as four 18-wide windows.

    `num_experts` stays 72 on every call — routing happens outside and every
    rank sees the whole space — while `local_num_experts = 18` and
    `local_expert_offset` runs 0 / 18 / 36 / 54 over the DeepSeek-V3-Lite
    routed geometry (H=2560, I=1536, top-6). Three things are checked: each
    window answers for exactly the slots routed into it, a token routed
    entirely elsewhere comes back bitwise zero, and the four bf16 outputs sum
    back to the 72-expert result (the value the tep4 all-reduce adds up).

    The sweep tops out at 8192 rows per call — trtllm's default
    `max_num_tokens`, the widest prefill a stock engine hands each rank
    unchunked. Counts are appended in ascending order, so every count below
    the top draws exactly the values it drew before.
    """
    args, ref = _dsv3()
    num_experts, hidden, top_k, shard = 72, 2560, 6, 18
    windows = [(r * shard, shard) for r in range(4)]
    win_args = [_window_args(args, off, n) for off, n in windows]
    win_ref = [_window_ref(ref, off, n) for off, n in windows]
    gen = torch.Generator(device=DEV).manual_seed(1818)
    g1 = 137.0
    worst_win = (0.0, 0.0)
    worst_sum = (0.0, 0.0)
    worst_vs_full = (0.0, 0.0)
    worst_drop = 1e9
    token_counts = (1, 8, 256, 1024, 4096, 8192)

    for num_tokens in token_counts:
        data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
        ids, wts = _routing(num_tokens, num_experts, top_k, gen)
        g2 = _calibrate_g2(x, ids, ref)
        s1, sg, s2 = _scalars(num_experts, g1, g2)
        full = _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts,
            **_window_scalars(s1, sg, s2, 0, num_experts),
        )[0]
        exp_full = _ref_moe(x, ids, wts, ref, g2).to(torch.bfloat16)
        _assert_moe_close(full, exp_full)

        parts = []
        top_win = (0.0, 0.0)  # after the loop: the largest token count's numbers
        for (off, n), wa, wr in zip(windows, win_args, win_ref):
            y = _call(
                data,
                sf,
                wa,
                num_experts,
                top_k,
                local_expert_offset=off,
                local_num_experts=n,
                topk_ids=ids,
                topk_weights=wts,
                **_window_scalars(s1, sg, s2, off, n),
            )[0]
            assert y.shape == (num_tokens, hidden), y.shape
            exp = _ref_moe(x, ids, wts, wr, g2, offset=off, num_local=n).to(torch.bfloat16)
            _assert_moe_close(y, exp)
            outside = ~((ids >= off) & (ids < off + n)).any(dim=1)
            assert torch.equal(y[outside], torch.zeros_like(y[outside])), (
                f"window (off={off}, n={n}) wrote into rows whose token routes "
                f"entirely elsewhere ({int(outside.sum())} such rows at T={num_tokens})"
            )
            top_win = tuple(max(a, b) for a, b in zip(top_win, _dev(y, exp)))
            worst_win = tuple(max(a, b) for a, b in zip(worst_win, _dev(y, exp)))
            parts.append(y)

        summed = sum(p.float() for p in parts).to(torch.bfloat16)
        _assert_moe_close(summed, exp_full)
        top_sum = _dev(summed, exp_full)
        worst_sum = tuple(max(a, b) for a, b in zip(worst_sum, _dev(summed, exp_full)))
        worst_vs_full = tuple(max(a, b) for a, b in zip(worst_vs_full, _dev(summed, full.float())))
        # The sum gate bites: leaving any one window out lands far outside it,
        # so passing it is not something four near-arbitrary partials could do.
        # Only meaningful once every window carries a settled share of the
        # slots — at a handful of tokens a window can legitimately hold none.
        if num_tokens >= 256:
            top_drop = 1e9  # after the loop: the largest token count's figure
            for drop in range(4):
                partial = sum(p.float() for i, p in enumerate(parts) if i != drop)
                _, rms = _dev(partial.to(torch.bfloat16), exp_full)
                assert rms > 25.0, (
                    f"dropping window {drop} at T={num_tokens} is only {rms:.1f} rms ulp away"
                )
                worst_drop = min(worst_drop, rms)
                top_drop = min(top_drop, rms)

    # Every routed slot inside one window: that window alone reproduces the
    # 72-expert call bitwise, and the other three return exactly zero.
    num_tokens = 64
    data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
    ids = torch.stack(
        [torch.randperm(shard, device=DEV, generator=gen)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    wts = torch.rand(num_tokens, top_k, device=DEV, generator=gen).to(torch.bfloat16)
    g2 = _calibrate_g2(x, ids, ref)
    s1, sg, s2 = _scalars(num_experts, g1, g2)
    full = _call(
        data,
        sf,
        args,
        num_experts,
        top_k,
        topk_ids=ids,
        topk_weights=wts,
        **_window_scalars(s1, sg, s2, 0, num_experts),
    )[0]
    for (off, n), wa in zip(windows, win_args):
        y = _call(
            data,
            sf,
            wa,
            num_experts,
            top_k,
            local_expert_offset=off,
            local_num_experts=n,
            topk_ids=ids,
            topk_weights=wts,
            **_window_scalars(s1, sg, s2, off, n),
        )[0]
        if off == 0:
            assert torch.equal(y, full), (
                "the window holding every routed slot did not reproduce the 72-expert call bitwise"
            )
        else:
            assert torch.equal(y, torch.zeros_like(y)), (
                f"window off={off} holds no routed slot but returned nonzero rows"
            )

    # Window mistakes a caller can make, against the correct window-1 result.
    num_tokens = 256
    data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    g2 = _calibrate_g2(x, ids, ref)
    s1, sg, s2 = _scalars(num_experts, g1, g2)
    off, n = windows[1]
    good = _call(
        data,
        sf,
        win_args[1],
        num_experts,
        top_k,
        local_expert_offset=off,
        local_num_experts=n,
        topk_ids=ids,
        topk_weights=wts,
        **_window_scalars(s1, sg, s2, off, n),
    )[0]
    good_ref = _ref_moe(x, ids, wts, win_ref[1], g2, offset=off, num_local=n).to(torch.bfloat16)
    _assert_moe_close(good, good_ref)
    wrong_offset = _call(
        data,
        sf,
        win_args[1],
        num_experts,
        top_k,
        local_expert_offset=0,
        local_num_experts=n,
        topk_ids=ids,
        topk_weights=wts,
        **_window_scalars(s1, sg, s2, off, n),
    )[0]
    wrong_weights = _call(
        data,
        sf,
        win_args[0],
        num_experts,
        top_k,
        local_expert_offset=off,
        local_num_experts=n,
        topk_ids=ids,
        topk_weights=wts,
        **_window_scalars(s1, sg, s2, off, n),
    )[0]
    all_at_zero = sum(
        _call(
            data,
            sf,
            wa,
            num_experts,
            top_k,
            local_expert_offset=0,
            local_num_experts=w[1],
            topk_ids=ids,
            topk_weights=wts,
            **_window_scalars(s1, sg, s2, w[0], w[1]),
        )[0].float()
        for w, wa in zip(windows, win_args)
    ).to(torch.bfloat16)
    full_ref = _ref_moe(x, ids, wts, ref, g2).to(torch.bfloat16)
    # The element metric is degenerate for the first two — they write into rows
    # the correct window leaves at exactly zero — so gate on relative RMS,
    # which the correct window holds under 1 ulp.
    for name, bad, against in (
        ("offset left at 0", wrong_offset, good_ref),
        ("neighbouring window's weights", wrong_weights, good_ref),
        ("all four windows at offset 0", all_at_zero, full_ref),
    ):
        elt, rms = _dev(bad, against)
        assert rms > 25.0, f"{name} only {rms:.1f} rms ulp away — not discriminated"
        print(f"    {name:31s} {elt:9.3g} elt / {rms:6.1f} rms ulp")

    # The window is a plain range test, not a validated partition: local slot i
    # answers for global id off+i, and off + local_num_experts > num_experts is
    # accepted — the surplus local slots are simply never addressed.
    over = 60
    y = _call(
        data,
        sf,
        win_args[3],
        num_experts,
        top_k,
        local_expert_offset=over,
        local_num_experts=shard,
        topk_ids=ids,
        topk_weights=wts,
        **_window_scalars(s1, sg, s2, windows[3][0], shard),
    )[0]
    _assert_moe_close(
        y,
        _ref_moe(x, ids, wts, win_ref[3], g2, offset=over, num_local=num_experts - over).to(
            torch.bfloat16
        ),
    )
    past = _call(
        data,
        sf,
        win_args[3],
        num_experts,
        top_k,
        local_expert_offset=num_experts,
        local_num_experts=shard,
        topk_ids=ids,
        topk_weights=wts,
        **_window_scalars(s1, sg, s2, windows[3][0], shard),
    )[0]
    assert torch.equal(past, torch.zeros_like(past)), (
        "a window entirely past the routing space returned nonzero rows"
    )
    print(
        f"  test_ep_window_18_of_72 OK (window {worst_win[0]:.2f} elt / "
        f"{worst_win[1]:.2f} rms, sum {worst_sum[0]:.2f} elt / {worst_sum[1]:.2f} rms, "
        f"sum vs 72-expert call {worst_vs_full[0]:.2f} elt, "
        f"one window dropped >= {worst_drop:.0f} rms ulp; "
        f"at T={token_counts[-1]} window {top_win[0]:.2f} elt / {top_win[1]:.2f} rms, "
        f"sum {top_sum[0]:.2f} elt / {top_sum[1]:.2f} rms, "
        f"one window dropped >= {top_drop:.0f} rms ulp)"
    )


def test_r1_geometry():
    """E=256, H=7168, I=2048, top-8: the DeepSeek-R1 routed expert layer.

    The whole certified token column in one sweep, decode-sized (1 row) through
    the widest unchunked prefill a stock engine issues (8192 = trtllm's default
    `max_num_tokens`). Nothing about this geometry needs padding: FC1 is
    `[256, 4096, 3584]` + `[256, 4096, 448]` and FC2 `[256, 7168, 1024]` +
    `[256, 7168, 128]`, and `4096 % 128`, `448 % 4`, `7168 % 128`, `128 % 4`
    are all zero, so the operands go in at their nominal shapes.

    Token counts are swept in ascending order, so every count below the top
    draws exactly the values it drew before.
    """
    args, ref = _dsr1()
    gen = ref["gen"]
    num_experts, hidden, top_k = 256, 7168, 8
    worst = (0.0, 0.0)
    for num_tokens in _R1_TOKENS:
        g1 = 137.0
        data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
        ids, wts = _routing(num_tokens, num_experts, top_k, gen)
        g2 = _calibrate_g2(x, ids, ref)
        s1, sg, s2 = _scalars(num_experts, g1, g2)
        out = _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts,
            output1_scale_scalar=s1,
            output1_scale_gate_scalar=sg,
            output2_scale_scalar=s2,
        )
        assert len(out) == 1, len(out)
        y = out[0]
        assert y.shape == (num_tokens, hidden) and y.dtype == torch.bfloat16, y.shape
        exp = _ref_moe(x, ids, wts, ref, g2).to(torch.bfloat16)
        _assert_moe_close(y, exp)
        top = _dev(y, exp)  # after the loop: the largest token count's numbers
        worst = tuple(max(a, b) for a, b in zip(worst, top))
    print(
        f"  test_r1_geometry OK (worst {worst[0]:.2f} elt / {worst[1]:.2f} rms ulp; "
        f"at T={_R1_TOKENS[-1]} {top[0]:.2f} elt / {top[1]:.2f} rms ulp)"
    )


def test_ep_window_64_of_256():
    """The dep4 expert-parallel split: 256 routed experts as four 64-wide windows.

    `num_experts` stays 256 on every call — routing happens outside and every
    rank sees the whole space — while `local_num_experts = 64` and
    `local_expert_offset` runs 0 / 64 / 128 / 192 over the DeepSeek-R1 routed
    geometry (H=7168, I=2048, top-8), across the whole certified token column.
    Three things are checked: each window answers for exactly the slots routed
    into it, a token routed entirely elsewhere comes back bitwise zero, and the
    four bf16 outputs sum back to the 256-expert result (the value the dep4
    all-reduce adds up).
    """
    args, ref = _dsr1()
    num_experts, hidden, top_k, shard = 256, 7168, 8, 64
    windows = [(r * shard, shard) for r in range(4)]
    win_args = [_window_args(args, off, n) for off, n in windows]
    win_ref = [_window_ref(ref, off, n) for off, n in windows]
    gen = torch.Generator(device=DEV).manual_seed(6464)
    g1 = 137.0
    worst_win = (0.0, 0.0)
    worst_sum = (0.0, 0.0)
    worst_vs_full = (0.0, 0.0)
    worst_drop = 1e9
    worst_zero_rows = 0

    for num_tokens in _R1_TOKENS:
        data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
        ids, wts = _routing(num_tokens, num_experts, top_k, gen)
        g2 = _calibrate_g2(x, ids, ref)
        s1, sg, s2 = _scalars(num_experts, g1, g2)
        full = _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts,
            **_window_scalars(s1, sg, s2, 0, num_experts),
        )[0]
        exp_full = _ref_moe(x, ids, wts, ref, g2).to(torch.bfloat16)
        _assert_moe_close(full, exp_full)

        parts = []
        top_win = (0.0, 0.0)  # after the loop: the largest token count's numbers
        for (off, n), wa, wr in zip(windows, win_args, win_ref):
            y = _call(
                data,
                sf,
                wa,
                num_experts,
                top_k,
                local_expert_offset=off,
                local_num_experts=n,
                topk_ids=ids,
                topk_weights=wts,
                **_window_scalars(s1, sg, s2, off, n),
            )[0]
            assert y.shape == (num_tokens, hidden), y.shape
            exp = _ref_moe(x, ids, wts, wr, g2, offset=off, num_local=n).to(torch.bfloat16)
            _assert_moe_close(y, exp)
            outside = ~((ids >= off) & (ids < off + n)).any(dim=1)
            assert torch.equal(y[outside], torch.zeros_like(y[outside])), (
                f"window (off={off}, n={n}) wrote into rows whose token routes "
                f"entirely elsewhere ({int(outside.sum())} such rows at T={num_tokens})"
            )
            worst_zero_rows = max(worst_zero_rows, int(outside.sum()))
            top_win = tuple(max(a, b) for a, b in zip(top_win, _dev(y, exp)))
            worst_win = tuple(max(a, b) for a, b in zip(worst_win, _dev(y, exp)))
            parts.append(y)

        summed = sum(p.float() for p in parts).to(torch.bfloat16)
        _assert_moe_close(summed, exp_full)
        top_sum = _dev(summed, exp_full)
        worst_sum = tuple(max(a, b) for a, b in zip(worst_sum, top_sum))
        worst_vs_full = tuple(max(a, b) for a, b in zip(worst_vs_full, _dev(summed, full.float())))
        # The sum gate bites: leaving any one window out lands far outside it,
        # so passing it is not something four near-arbitrary partials could do.
        # Only meaningful once every window carries a settled share of the
        # slots — at a handful of tokens a window can legitimately hold none.
        if num_tokens >= 256:
            top_drop = 1e9  # after the loop: the largest token count's figure
            for drop in range(4):
                partial = sum(p.float() for i, p in enumerate(parts) if i != drop)
                _, rms = _dev(partial.to(torch.bfloat16), exp_full)
                assert rms > 25.0, (
                    f"dropping window {drop} at T={num_tokens} is only {rms:.1f} rms ulp away"
                )
                worst_drop = min(worst_drop, rms)
                top_drop = min(top_drop, rms)

    # Every routed slot inside one window: that window alone reproduces the
    # 256-expert call bitwise, and the other three return exactly zero.
    num_tokens = 64
    data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
    ids = torch.stack(
        [torch.randperm(shard, device=DEV, generator=gen)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    wts = torch.rand(num_tokens, top_k, device=DEV, generator=gen).to(torch.bfloat16)
    g2 = _calibrate_g2(x, ids, ref)
    s1, sg, s2 = _scalars(num_experts, g1, g2)
    full = _call(
        data,
        sf,
        args,
        num_experts,
        top_k,
        topk_ids=ids,
        topk_weights=wts,
        **_window_scalars(s1, sg, s2, 0, num_experts),
    )[0]
    for (off, n), wa in zip(windows, win_args):
        y = _call(
            data,
            sf,
            wa,
            num_experts,
            top_k,
            local_expert_offset=off,
            local_num_experts=n,
            topk_ids=ids,
            topk_weights=wts,
            **_window_scalars(s1, sg, s2, off, n),
        )[0]
        if off == 0:
            assert torch.equal(y, full), (
                "the window holding every routed slot did not reproduce the 256-expert call bitwise"
            )
        else:
            assert torch.equal(y, torch.zeros_like(y)), (
                f"window off={off} holds no routed slot but returned nonzero rows"
            )

    # Window mistakes a caller can make, against the correct window-1 result.
    num_tokens = 256
    data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    g2 = _calibrate_g2(x, ids, ref)
    s1, sg, s2 = _scalars(num_experts, g1, g2)
    off, n = windows[1]
    good = _call(
        data,
        sf,
        win_args[1],
        num_experts,
        top_k,
        local_expert_offset=off,
        local_num_experts=n,
        topk_ids=ids,
        topk_weights=wts,
        **_window_scalars(s1, sg, s2, off, n),
    )[0]
    good_ref = _ref_moe(x, ids, wts, win_ref[1], g2, offset=off, num_local=n).to(torch.bfloat16)
    _assert_moe_close(good, good_ref)
    wrong_offset = _call(
        data,
        sf,
        win_args[1],
        num_experts,
        top_k,
        local_expert_offset=0,
        local_num_experts=n,
        topk_ids=ids,
        topk_weights=wts,
        **_window_scalars(s1, sg, s2, off, n),
    )[0]
    wrong_weights = _call(
        data,
        sf,
        win_args[0],
        num_experts,
        top_k,
        local_expert_offset=off,
        local_num_experts=n,
        topk_ids=ids,
        topk_weights=wts,
        **_window_scalars(s1, sg, s2, off, n),
    )[0]
    all_at_zero = sum(
        _call(
            data,
            sf,
            wa,
            num_experts,
            top_k,
            local_expert_offset=0,
            local_num_experts=w[1],
            topk_ids=ids,
            topk_weights=wts,
            **_window_scalars(s1, sg, s2, w[0], w[1]),
        )[0].float()
        for w, wa in zip(windows, win_args)
    ).to(torch.bfloat16)
    full_ref = _ref_moe(x, ids, wts, ref, g2).to(torch.bfloat16)
    # The element metric is degenerate for the first two — they write into rows
    # the correct window leaves at exactly zero — so gate on relative RMS,
    # which the correct window holds under 1 ulp.
    for name, bad, against in (
        ("offset left at 0", wrong_offset, good_ref),
        ("neighbouring window's weights", wrong_weights, good_ref),
        ("all four windows at offset 0", all_at_zero, full_ref),
    ):
        elt, rms = _dev(bad, against)
        assert rms > 25.0, f"{name} only {rms:.1f} rms ulp away — not discriminated"
        print(f"    {name:31s} {elt:9.3g} elt / {rms:6.1f} rms ulp")
    print(
        f"  test_ep_window_64_of_256 OK (window {worst_win[0]:.2f} elt / "
        f"{worst_win[1]:.2f} rms, sum {worst_sum[0]:.2f} elt / {worst_sum[1]:.2f} rms, "
        f"sum vs 256-expert call {worst_vs_full[0]:.2f} elt, "
        f"one window dropped >= {worst_drop:.0f} rms ulp, "
        f"<= {worst_zero_rows} bitwise-zero rows per window; "
        f"at T={_R1_TOKENS[-1]} window {top_win[0]:.2f} elt / {top_win[1]:.2f} rms, "
        f"sum {top_sum[0]:.2f} elt / {top_sum[1]:.2f} rms, "
        f"one window dropped >= {top_drop:.0f} rms ulp)"
    )


def test_many_experts_top10_geometry():
    """E=512, H=2560, I=640, top-10: qwen4_exp's many-small-experts routed layer.

    Three axes at once that neither DeepSeek geometry above reaches — 512
    experts, a top_k past 8, and an intermediate width of 640 — swept over the
    whole token column, 1 row through 8192 (trtllm's default `max_num_tokens`,
    the widest unchunked prefill a stock engine issues). Token counts are
    appended in ascending order, so every count below the top draws exactly the
    values it drew before.

    Slot weights are renormalised to sum to 1, which is what a
    `norm_topk_prob` router emits, and two counts are repeated with plain
    uniform weights so the gate is not resting on that convex combination.
    Three further things are pinned here: `routing_method_type` is still inert
    at the (512, top-10) pair where the routing tier tables would matter, two
    identical calls are bitwise equal, and the wrong computations land far
    outside the gate at this geometry too.
    """
    args, ref = _many_experts()
    num_experts, hidden, inter, top_k = 512, 2560, 640, 10
    gen = torch.Generator(device=DEV).manual_seed(51264)
    g1 = 137.0
    worst = (0.0, 0.0)
    keep = {}
    for num_tokens in _MANY_TOKENS:
        data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
        ids, wts = _renorm_routing(num_tokens, num_experts, top_k, gen)
        assert (ids.sort(dim=-1).values.diff(dim=-1) != 0).all(), (
            "the routing draw repeated an expert id inside one token's row, which "
            "this op does not reliably deduplicate"
        )
        g2 = _calibrate_g2(x, ids, ref)
        s1, sg, s2 = _scalars(num_experts, g1, g2)
        kw = dict(
            topk_ids=ids,
            topk_weights=wts,
            output1_scale_scalar=s1,
            output1_scale_gate_scalar=sg,
            output2_scale_scalar=s2,
        )
        out = _call(data, sf, args, num_experts, top_k, **kw)
        assert len(out) == 1, len(out)
        y = out[0]
        assert y.shape == (num_tokens, hidden) and y.dtype == torch.bfloat16, y.shape
        exp = moe.reference(x, ids, wts, ref["up"], ref["gate"], ref["down"], g2).to(torch.bfloat16)
        _compare(y, exp)
        top = _dev(y, exp)  # after the loop: the largest token count's numbers
        worst = tuple(max(a, b) for a, b in zip(worst, top))
        if num_tokens in (256, 8192):
            keep[num_tokens] = (data, sf, x, ids, wts, g2, s1, sg, s2, y, exp)

    # Un-renormalised slot weights, so the sum-to-1 combine is not what is
    # holding the gate: ten weights drawn uniformly from [0, 1) instead.
    uniform = (0.0, 0.0)
    for num_tokens in (256, 8192):
        data, sf, x, ids, _, g2, s1, sg, s2, _, _ = keep[num_tokens]
        wts = torch.rand(num_tokens, top_k, device=DEV, generator=gen).to(torch.bfloat16)
        y = _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts,
            output1_scale_scalar=s1,
            output1_scale_gate_scalar=sg,
            output2_scale_scalar=s2,
        )[0]
        exp = moe.reference(x, ids, wts, ref["up"], ref["gate"], ref["down"], g2).to(torch.bfloat16)
        _compare(y, exp)
        uniform = tuple(max(a, b) for a, b in zip(uniform, _dev(y, exp)))

    data, sf, x, ids, wts, g2, s1, sg, s2, y256, exp256 = keep[256]
    kw = dict(
        topk_ids=ids,
        topk_weights=wts,
        output1_scale_scalar=s1,
        output1_scale_gate_scalar=sg,
        output2_scale_scalar=s2,
    )
    # Which (expert count, top_k) pairs the routing kernels are compiled for is
    # a per-policy tier list (`PolicyTraits` / `TierList` in
    # `cpp/tensorrt_llm/kernels/moe/trtllmGen/routing/RoutingCustomPolicy.cuh`),
    # and a pair with no tier is a logged error rather than a raise. That would
    # bite at (512, top-10) if the caller's `routing_method_type` chose the
    # policy — but on the pre-routed path it does not: `runPostTopKPipeline`
    # (`RoutingFromTopKIds.cu`) substitutes its own preprocess/postprocess pair
    # before dispatching. Checked at the pair where it would show.
    for rmt in (0, 2, 4, 5, 6):
        alt = _call(data, sf, args, num_experts, top_k, routing_method_type=rmt, **kw)[0]
        assert torch.equal(alt, y256), (
            f"routing_method_type={rmt} changed the pre-routed result at "
            f"num_experts=512, top_k=10: {_dev(alt, y256.float())} (elt / rms ulp)"
        )
    assert torch.equal(_call(data, sf, args, num_experts, top_k, **kw)[0], y256), (
        "two identical calls at this geometry were not bitwise equal"
    )

    # How far the wrong computations land, at this geometry, from the result
    # the gate accepts. The RMS gate is 2 ulp; the closest of these is an order
    # of magnitude outside it.
    dist = {}
    for name, wrong in (
        ("gate/up swapped", _ref_moe(x, ids, wts, ref, g2, swap_gate_up=True)),
        ("output1_* roles swapped", _ref_moe(x, ids, wts, ref, g2, swap_scalars=True)),
        (
            "no FC1-output requantization",
            _ref_moe(x, ids, wts, ref, g2, quantize_intermediate=False),
        ),
    ):
        dist[name] = _dev(y256, wrong.to(torch.bfloat16))
    up_p, gt_p, up_s, gt_s, dn_p, dn_s = ref["raw"]
    for name, prep in (("no scale swizzle", "swizzle"), ("no row shuffle", "shuffle")):
        w1, sw1 = _prep_fc1(up_p, gt_p, up_s, gt_s, **{prep: False})
        w2, sw2 = _prep_fc2(dn_p, dn_s, **{prep: False})
        bad = _call(
            data,
            sf,
            dict(
                gemm1_weights=w1,
                gemm1_weights_scale=sw1,
                gemm2_weights=w2,
                gemm2_weights_scale=sw2,
                intermediate_size=inter,
            ),
            num_experts,
            top_k,
            **kw,
        )[0]
        dist[name] = _dev(bad, exp256.float())
        del w1, sw1, w2, sw2, bad
        torch.cuda.empty_cache()
    for name, (elt, rms) in dist.items():
        assert rms > 10.0, (
            f"{name} is only {rms:.1f} rms ulp from the correct result, against a "
            f"2 ulp gate — this test would not catch it"
        )
    print(
        f"  test_many_experts_top10_geometry OK (worst {worst[0]:.2f} elt / "
        f"{worst[1]:.2f} rms ulp; at T={_MANY_TOKENS[-1]} {top[0]:.2f} elt / "
        f"{top[1]:.2f} rms ulp; uniform weights {uniform[0]:.2f} / "
        f"{uniform[1]:.2f}; wrong variants "
        + ", ".join(f"{n} {v[1]:.0f}" for n, v in dist.items())
        + " rms ulp)"
    )


def test_narrow_intermediate_is_nvfp4_requantized():
    """The FC1 epilogue quantizes once, and still exactly, at I = 640.

    `test_intermediate_is_nvfp4_requantized` pins the formula with an identity
    down projection, which forces I = H; 640 is not a legal hidden size here
    (`H % 256 != 0`), so this repeats the measurement at H = 2560 / I = 640 and
    512 experts with a down projection that is the identity on its first 640
    output rows and zero on the rest. The kernel's own output then carries the
    requantized intermediate in columns [0, 640) and must be exactly zero past
    it. Bit-exact, not toleranced: e2m1 x e4m3 products are exact in bf16.

    This is what the entry's tolerance rests on. Other runners in this family
    requantize more than once per slot, and a reference modelling a single
    stage cannot match those; here one modelled stage reproduces every element,
    so the tolerance is absorbing accumulation order and nothing else.
    """
    num_experts, hidden, inter, num_tokens = 512, 2560, 640, 64
    gen = torch.Generator(device=DEV).manual_seed(640640)
    up_p, up_s, up_c = _rand_nvfp4(num_experts, inter, hidden, gen)
    gt_p, gt_s, gt_c = _rand_nvfp4(num_experts, inter, hidden, gen)
    w1, sw1 = _prep_fc1(up_p, gt_p, up_s, gt_s)
    sel = torch.zeros(num_experts, hidden, inter, dtype=torch.uint8, device=DEV)
    d = torch.arange(inter, device=DEV)
    sel[:, d, d] = 2  # e2m1 code 2 == +1.0
    dn_p = (sel[..., 0::2] | (sel[..., 1::2] << 4)).contiguous()
    dn_s = torch.ones(num_experts, hidden, inter // SV, device=DEV).to(torch.float8_e4m3fn)
    w2, sw2 = _prep_fc2(dn_p, dn_s)
    del sel, up_p, gt_p, dn_p
    torch.cuda.empty_cache()

    g1 = 137.0
    data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
    ids = torch.randint(
        0, num_experts, (num_tokens, 1), dtype=torch.int32, device=DEV, generator=gen
    )
    wts = torch.ones(num_tokens, 1, dtype=torch.bfloat16, device=DEV)

    act = torch.zeros(num_tokens, inter, dtype=torch.float32, device=DEV)
    for e in ids[:, 0].unique().tolist():
        tok = (ids[:, 0] == e).nonzero(as_tuple=True)[0]
        xe = x[tok]
        up = xe @ _dequant(up_c[e], up_s[e]).t()
        gate = xe @ _dequant(gt_c[e], gt_s[e]).t()
        act[tok] = up * gate * torch.sigmoid(gate)
    g2 = E4M3_MAX * E2M1_MAX / act.abs().max().item()
    a1 = torch.full((num_experts,), 1.0 / g1, dtype=torch.float32, device=DEV)
    y = _call(
        data,
        sf,
        dict(
            gemm1_weights=w1,
            gemm1_weights_scale=sw1,
            gemm2_weights=w2,
            gemm2_weights_scale=sw2,
            intermediate_size=inter,
        ),
        num_experts,
        1,
        topk_ids=ids,
        topk_weights=wts,
        output1_scale_scalar=a1 * g2,
        output1_scale_gate_scalar=a1,
        output2_scale_scalar=torch.ones(num_experts, dtype=torch.float32, device=DEV),
    )[0].float()

    head, tail = y[:, :inter], y[:, inter:]
    exact = _q_nvfp4(act, g2)
    assert torch.equal(head, exact), (
        f"at I={inter} the FC1 epilogue is not sf=e4m3(g*amax/6), data=e2m1(g*x/sf): "
        f"{(head != exact).sum().item()} of {head.numel()} elements differ"
    )
    assert torch.equal(tail, torch.zeros_like(tail)), (
        "the zero rows of the selection down projection did not produce zeros — "
        "the columns past I do not isolate the intermediate"
    )
    # A second quantization anywhere in the chain, an unrounded block scale, or
    # no quantization at all would each break the equality above; these two
    # record how far the near misses are from matching.
    b = (act * g2).reshape(num_tokens, inter // SV, SV)
    sfx = b.abs().amax(-1, keepdim=True) / E2M1_MAX
    unrounded = (_e2m1_rne(b / sfx.clamp_min(1e-30)) * sfx).reshape(num_tokens, inter)
    fracs = {}
    for name, cand in (
        ("no requantization", act * g2),
        ("exact (unrounded) scale", unrounded),
    ):
        fracs[name] = (head == cand).float().mean().item()
        assert fracs[name] < 0.9, f"{name} also matches {fracs[name]:.2%} — no discrimination"
    print(
        f"  test_narrow_intermediate_is_nvfp4_requantized OK (bit-exact on all "
        f"{head.numel()} elements at I={inter}; unrounded scale "
        f"{fracs['exact (unrounded) scale']:.1%}, no requantization "
        f"{fracs['no requantization']:.1%})"
    )


@pytest.fixture
def clear_autotuner_cache():
    """Leave the profiling cache cold even when the test fails part-way, so a
    failure here does not turn into a failure of the test that runs next."""
    yield
    AutoTuner.get().clear_cache()


def test_r1_autotuner_cache_is_inert(clear_autotuner_cache):
    """A warm autotuner profiling cache does not change the R1-geometry result.

    Every other call in this file runs with a **cold** cache, where the tuner
    returns the fallback tactic. A serving engine is not cold — the PyTorch
    runtime profiles this op during warm-up whenever `enable_autotuner` is set.
    So this test drives the other state: it warms the cache through trtllm's
    own `autotune()` context at `local_num_experts` 256 and 64, then re-runs
    the whole token column for all five expert layouts and demands **bitwise**
    equality with the cold results. Bitwise is the right gate here — the two
    runs consume identical operands, so any tactic-dependent difference at all
    is visible.

    It must run last: it leaves the tuner warm until the `clear_cache()` at the
    end, and the entry's receipt is a cold-cache receipt.
    """
    args, ref = _dsr1()
    num_experts, hidden, top_k, shard = 256, 7168, 8, 64
    layouts = [(0, num_experts, args)] + [
        (r * shard, shard, _window_args(args, r * shard, shard)) for r in range(4)
    ]
    gen = torch.Generator(device=DEV).manual_seed(4242)
    g1 = 137.0
    cases = {}
    for num_tokens in _R1_TOKENS:
        data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
        ids, wts = _routing(num_tokens, num_experts, top_k, gen)
        cases[num_tokens] = (
            data,
            sf,
            ids,
            wts,
            _scalars(num_experts, g1, _calibrate_g2(x, ids, ref)),
        )

    def run(off, n, wa, num_tokens):
        data, sf, ids, wts, (s1, sg, s2) = cases[num_tokens]
        return _call(
            data,
            sf,
            wa,
            num_experts,
            top_k,
            local_expert_offset=off,
            local_num_experts=n,
            topk_ids=ids,
            topk_weights=wts,
            **_window_scalars(s1, sg, s2, off, n),
        )[0]

    tuner = AutoTuner.get()
    op = "trtllm::fp4_block_scale_moe_runner"

    def entries():
        return [k for k in tuner.profiling_cache.cache if k[0] == op]

    assert not entries(), (
        "the profiling cache already holds entries for this op — the cold-cache "
        "receipt every other test in this file records would be void"
    )
    cold = {(off, n, t): run(off, n, wa, t) for off, n, wa in layouts for t in _R1_TOKENS}
    assert not entries(), (
        "an ordinary call wrote a profiling-cache entry — calls outside "
        "autotune() are not cold after all"
    )

    profiled = []
    with autotune():
        for off, n, wa in layouts:
            run(off, n, wa, _R1_TOKENS[-1])
            profiled.append(tuner.stats.tuned_op_profiled_configs.get(op, 0))
    # What the tuner's key does and does not separate, counted rather than
    # timed: switching from local_num_experts 256 to 64 forces a fresh sweep,
    # while the three remaining local_expert_offsets profile nothing at all —
    # they hit the entries the offset-0 window just wrote.
    assert profiled[1] > profiled[0], (
        f"local_num_experts 64 reused the 256-expert tuning: {profiled}"
    )
    assert profiled[2:] == [profiled[1]] * 3, (
        f"a 64-wide window at a nonzero local_expert_offset re-profiled: {profiled}"
    )
    keys = entries()
    assert keys, "autotune() recorded nothing for this op — nothing was warmed"
    tactics = {str(tuner.profiling_cache.cache[k][1]) for k in keys}
    assert "-1" not in tactics, (
        f"the tuner recorded the fallback tactic, so warm and cold are the same "
        f"execution and this comparison proves nothing: {sorted(tactics)}"
    )
    unique_ids = sorted({str(k[2]) for k in keys})

    for off, n, wa in layouts:
        for num_tokens in _R1_TOKENS:
            warm = run(off, n, wa, num_tokens)
            ref_cold = cold[(off, n, num_tokens)]
            assert torch.equal(warm, ref_cold), (
                f"a warm autotuner cache changed the result at "
                f"local_expert_offset={off}, local_num_experts={n}, "
                f"T={num_tokens}: {_dev(warm, ref_cold.float())} (elt / rms ulp)"
            )
    print(
        f"  test_r1_autotuner_cache_is_inert OK ({len(keys)} tuned entries over "
        f"unique_ids {unique_ids}, {len(tactics)} distinct tactics, "
        f"profiled-config counts {profiled} across the five layouts; all "
        f"{len(layouts) * len(_R1_TOKENS)} (expert layout, T) results bitwise "
        f"unchanged)"
    )


def test_many_experts_autotuner_cache_is_inert(clear_autotuner_cache):
    """A warm autotuner profiling cache does not change the 512-expert result.

    `local_num_experts` separates the tuner's cache entries — the R1 test
    above measures that by counting profiling sweeps — so a cache warmed at 256
    or 64 experts says nothing about 512, and a serving engine's cache is warm
    while every other call in this file runs cold. Same shape as the R1 test:
    take the cold results, warm through trtllm's own `autotune()` context at
    this layout, re-run, demand **bitwise** equality. It clears the cache at
    the end, as the R1 test does, so the cold-cache receipt the rest of this
    file records survives either ordering of the two.
    """
    args, ref = _many_experts()
    num_experts, hidden, top_k = 512, 2560, 10
    gen = torch.Generator(device=DEV).manual_seed(5125)
    g1 = 137.0
    token_counts = (1, 8, 256, 1024, 4096, 8192)
    cases = {}
    for num_tokens in token_counts:
        data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
        ids, wts = _renorm_routing(num_tokens, num_experts, top_k, gen)
        cases[num_tokens] = (
            data,
            sf,
            ids,
            wts,
            _scalars(num_experts, g1, _calibrate_g2(x, ids, ref)),
        )

    def run(num_tokens):
        data, sf, ids, wts, (s1, sg, s2) = cases[num_tokens]
        return _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts,
            output1_scale_scalar=s1,
            output1_scale_gate_scalar=sg,
            output2_scale_scalar=s2,
        )[0]

    tuner = AutoTuner.get()
    op = "trtllm::fp4_block_scale_moe_runner"

    def entries():
        return [k for k in tuner.profiling_cache.cache if k[0] == op]

    assert not entries(), (
        "the profiling cache already holds entries for this op — the cold-cache "
        "receipt every other test in this file records would be void"
    )
    cold = {t: run(t) for t in token_counts}
    assert not entries(), (
        "an ordinary call wrote a profiling-cache entry — calls outside "
        "autotune() are not cold after all"
    )
    with autotune():
        run(token_counts[-1])
    keys = entries()
    assert keys, "autotune() recorded nothing for this op — nothing was warmed"
    tactics = {str(tuner.profiling_cache.cache[k][1]) for k in keys}
    assert "-1" not in tactics, (
        f"the tuner recorded the fallback tactic, so warm and cold are the same "
        f"execution and this comparison proves nothing: {sorted(tactics)}"
    )
    unique_ids = sorted({str(k[2]) for k in keys})
    for num_tokens in token_counts:
        warm = run(num_tokens)
        assert torch.equal(warm, cold[num_tokens]), (
            f"a warm autotuner cache changed the result at T={num_tokens}: "
            f"{_dev(warm, cold[num_tokens].float())} (elt / rms ulp)"
        )
    print(
        f"  test_many_experts_autotuner_cache_is_inert OK ({len(keys)} tuned "
        f"entries over unique_ids {unique_ids}, {len(tactics)} distinct tactics; "
        f"all {len(token_counts)} token counts bitwise unchanged)"
    )


def test_do_finalize_and_output_buffer():
    """Both output modes, and what `do_finalize=False` actually returns."""
    num_experts, hidden, inter, top_k, num_tokens = 8, 256, 128, 2, 16
    args, ref = _build(num_experts, hidden, inter, seed=707)
    gen = ref["gen"]
    g1, g2 = 137.0, 1.0 / 32.0
    data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    s1, sg, s2 = _scalars(num_experts, g1, g2)
    kw = dict(
        topk_ids=ids,
        topk_weights=wts,
        output1_scale_scalar=s1,
        output1_scale_gate_scalar=sg,
        output2_scale_scalar=s2,
    )
    y = _call(data, sf, args, num_experts, top_k, **kw)[0]

    # caller-provided buffer: written in place, [0] returned, tail untouched
    buf = torch.full((num_tokens + 3, hidden), 7.0, dtype=torch.bfloat16, device=DEV)
    r = _call(data, sf, args, num_experts, top_k, **kw, output=buf[:num_tokens])
    assert len(r) == 1 and r[0].shape == (0,) and r[0].dtype == torch.bfloat16
    assert torch.equal(buf[:num_tokens], y)
    assert bool((buf[num_tokens:] == 7.0).all()), "wrote past the output rows"

    for bad, msg in (
        (torch.empty(num_tokens, hidden, dtype=torch.float32, device=DEV), "bfloat16"),
        (torch.empty(num_tokens + 1, hidden, dtype=torch.bfloat16, device=DEV), "dim0"),
    ):
        try:
            _call(data, sf, args, num_experts, top_k, **kw, output=bad)
            raise AssertionError(f"bad output buffer ({msg}) accepted")
        except RuntimeError as e:
            assert msg in str(e), e

    part, scales, permuted = _call(data, sf, args, num_experts, top_k, **kw, do_finalize=False)
    assert part.dtype == torch.bfloat16 and part.shape[1] == hidden
    assert part.shape[0] >= num_tokens * top_k
    assert scales.shape == (num_tokens, top_k) and scales.dtype == torch.bfloat16
    assert permuted.shape == (num_tokens, top_k) and permuted.dtype == torch.int32
    assert int(permuted.min()) >= 0 and int(permuted.max()) < part.shape[0]
    comb = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=DEV)
    for j in range(top_k):
        comb += part[permuted[:, j].long()].float() * wts[:, j].float().unsqueeze(1)
    assert torch.equal(comb.to(torch.bfloat16), y), (
        "gathering per-slot rows by the third output and weighting them with the "
        "caller's topk_weights did not reproduce the finalized result"
    )
    try:
        _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            **kw,
            do_finalize=False,
            output=buf[:num_tokens],
        )
        raise AssertionError("output= with do_finalize=False accepted")
    except RuntimeError as e:
        assert "only supported when do_finalize=true" in str(e), e
    print("  test_do_finalize_and_output_buffer OK")


def test_inert_and_rejected_arguments():
    """Knobs that change nothing on the pre-routed path, and hard rejections."""
    num_experts, hidden, inter, top_k, num_tokens = 8, 256, 128, 2, 16
    args, ref = _build(num_experts, hidden, inter, seed=808)
    gen = ref["gen"]
    g1, g2 = 137.0, 1.0 / 32.0
    data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    s1, sg, s2 = _scalars(num_experts, g1, g2)
    kw = dict(
        topk_ids=ids,
        topk_weights=wts,
        output1_scale_scalar=s1,
        output1_scale_gate_scalar=sg,
        output2_scale_scalar=s2,
    )
    base = _call(data, sf, args, num_experts, top_k, **kw)[0]
    assert torch.equal(_call(data, sf, args, num_experts, top_k, **kw)[0], base), (
        "two identical calls are not bitwise equal"
    )

    inert = (
        [dict(routing_method_type=r) for r in (0, 1, 2, 4, 5, 6)]
        + [
            dict(routing_method_type=2, n_group=n, topk_group=t)
            for n, t in ((1, 1), (8, 4), (4, 2))
        ]
        + [dict(n_group=None, topk_group=None), dict(n_group=1, topk_group=1)]
        + [dict(routed_scaling_factor=r) for r in (None, 1.0, 2.5)]
        + [dict(tune_max_num_tokens=t) for t in (8192, 128)]
        + [dict(use_dp=u) for u in (False, True)]
        + [dict(routing_bias=torch.zeros(num_experts, dtype=torch.bfloat16, device=DEV))]
    )
    for over in inert:
        y = _call(data, sf, args, num_experts, top_k, **kw, **over)[0]
        assert torch.equal(y, base), f"{over} changed the pre-routed result"

    # n_group > 1 needs routing_method_type 2 even though it is then inert
    try:
        _call(data, sf, args, num_experts, top_k, **kw, n_group=8, topk_group=4)
        raise AssertionError("n_group>1 with routing_method_type=1 accepted")
    except RuntimeError as e:
        assert "DeepSeekV3 routing method" in str(e), e

    rejections = (
        (dict(topk_weights=wts.float()), "topk_weights must be bfloat16"),
        (dict(topk_weights=wts.half()), "topk_weights must be bfloat16"),
        (dict(topk_ids=ids.long()), "topk_ids must be int"),
        (dict(topk_ids=None, topk_weights=None), "must be provided"),
        (dict(intermediate_size=inter // 2), "incorrect dim 2"),
        (dict(num_experts_override=top_k), "num_experts must be greater than top_k"),
    )
    for over, msg in rejections:
        ne = over.pop("num_experts_override", num_experts)
        try:
            _call(data, sf, args, ne, top_k, **{**kw, **over})
            raise AssertionError(f"expected rejection: {msg}")
        except RuntimeError as e:
            assert msg in str(e), (msg, str(e))

    e4m3 = sf.view(torch.float8_e4m3fn)
    for bad, msg in (
        (sf.view(torch.uint8), "must be fp8"),
        (e4m3.reshape(num_tokens, -1), "must be 1D"),
        (e4m3[:-1], "incorrect size"),
    ):
        try:
            _call(data, bad, args, num_experts, top_k, scale_as_is=True, **kw)
            raise AssertionError(f"expected rejection: {msg}")
        except RuntimeError as e:
            assert msg in str(e), (msg, str(e))
    for name in ("gemm1_weights_scale", "gemm2_weights_scale"):
        try:
            _call(
                data,
                sf,
                args,
                num_experts,
                top_k,
                **kw,
                **{name: args[name].view(torch.uint8)},
            )
            raise AssertionError(f"{name} as uint8 accepted")
        except RuntimeError as e:
            assert "must be fp8" in str(e), e
    try:
        _call(data.view(torch.float8_e4m3fn), sf, args, num_experts, top_k, **kw)
        raise AssertionError("hidden_states as float8_e4m3fn accepted")
    except RuntimeError as e:
        assert "must be byte" in str(e), e
    print("  test_inert_and_rejected_arguments OK")


def test_hidden_must_be_a_multiple_of_256():
    """A hidden size that is not a multiple of 256 is silently wrong.

    The fault is in the weight path, not the activation one: it survives an
    activation whose block scales are all exactly 1.0. The wrapper guards it.
    """
    num_experts, inter, top_k, num_tokens = 8, 128, 2, 8
    for hidden in (256, 384, 512, 640, 768, 896):
        args, ref = _build(num_experts, hidden, inter, seed=1000 + hidden)
        gen = ref["gen"]
        g1 = 71.0
        data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
        ids, wts = _routing(num_tokens, num_experts, top_k, gen)
        g2 = _calibrate_g2(x, ids, ref)
        s1, sg, s2 = _scalars(num_experts, g1, g2)
        exp = _ref_moe(x, ids, wts, ref, g2).to(torch.bfloat16)
        if hidden % 256 == 0:
            y = _call(
                data,
                sf,
                args,
                num_experts,
                top_k,
                topk_ids=ids,
                topk_weights=wts,
                output1_scale_scalar=s1,
                output1_scale_gate_scalar=sg,
                output2_scale_scalar=s2,
            )[0]
            _assert_moe_close(y, exp)
            continue
        try:
            _call(
                data,
                sf,
                args,
                num_experts,
                top_k,
                topk_ids=ids,
                topk_weights=wts,
                output1_scale_scalar=s1,
                output1_scale_gate_scalar=sg,
                output2_scale_scalar=s2,
            )
            raise AssertionError(f"wrapper accepted hidden={hidden}")
        except AssertionError as e:
            assert "multiple of 256" in str(e), e
        raw = torch.ops.trtllm.fp4_block_scale_moe_runner(
            None,
            None,
            data,
            sf.view(torch.float8_e4m3fn),
            args["gemm1_weights"],
            args["gemm1_weights_scale"],
            None,
            None,
            None,
            None,
            args["gemm2_weights"],
            args["gemm2_weights_scale"],
            None,
            s1,
            sg,
            s2,
            num_experts,
            top_k,
            None,
            None,
            inter,
            0,
            num_experts,
            None,
            1,
            True,
            0,
            topk_weights=wts,
            topk_ids=ids,
        )[0]
        elt, _ = _dev(raw, exp)
        assert elt > 25.0, f"hidden={hidden} unexpectedly correct ({elt:.1f} ulp)"
        print(f"    hidden={hidden}: accepted by the op, {elt:.0f} ulp wrong")
    print("  test_hidden_must_be_a_multiple_of_256 OK")


def test_strided_inputs_are_silently_wrong():
    """Justifies the wrapper's contiguity asserts: the kernel ignores strides."""
    num_experts, hidden, inter, top_k, num_tokens = 8, 256, 128, 2, 16
    args, ref = _build(num_experts, hidden, inter, seed=909)
    gen = ref["gen"]
    g1, g2 = 137.0, 1.0 / 32.0
    data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    s1, sg, s2 = _scalars(num_experts, g1, g2)
    kw = dict(
        topk_ids=ids,
        topk_weights=wts,
        output1_scale_scalar=s1,
        output1_scale_gate_scalar=sg,
        output2_scale_scalar=s2,
    )
    base = _call(data, sf, args, num_experts, top_k, **kw)[0]
    for name, strided in (
        ("hidden_states", torch.stack([data, data], -1)[..., 0]),
        ("topk_weights", torch.stack([wts, wts], -1)[..., 0]),
        ("gemm1_weights", torch.stack([args["gemm1_weights"]] * 2, -1)[..., 0]),
    ):
        assert not strided.is_contiguous()
        try:
            moe_kw = dict(kw)
            if name == "hidden_states":
                y = _call(strided, sf, args, num_experts, top_k, **moe_kw)[0]
            else:
                moe_kw[name] = strided
                y = _call(data, sf, args, num_experts, top_k, **moe_kw)[0]
        except AssertionError:
            continue  # the wrapper's guard fired, which is the point
        raise AssertionError(
            f"wrapper did not reject a strided {name} (result equal: {torch.equal(y, base)})"
        )

    # and the raw op does accept them, silently
    raw = torch.ops.trtllm.fp4_block_scale_moe_runner(
        None,
        None,
        torch.stack([data, data], -1)[..., 0],
        sf.view(torch.float8_e4m3fn),
        args["gemm1_weights"],
        args["gemm1_weights_scale"],
        None,
        None,
        None,
        None,
        args["gemm2_weights"],
        args["gemm2_weights_scale"],
        None,
        s1,
        sg,
        s2,
        num_experts,
        top_k,
        None,
        None,
        inter,
        0,
        num_experts,
        None,
        1,
        True,
        0,
        topk_weights=wts,
        topk_ids=ids,
    )[0]
    assert not torch.equal(raw, base), (
        "a strided hidden_states no longer changes the result — re-check the guard"
    )
    print("  test_strided_inputs_are_silently_wrong OK")


# ── Mistral Large 4 tep4 routed experts, at the checkpoint's magnitudes ───────

# Read off the Mistral-Large-4-1T-A52B-NVFP4 checkpoint, every routed expert of
# layers 1..71: per layer, `input_scale` is one value for all 128 experts and
# the same for w1 and w3; `weight_scale_2` is per expert, the same for w1 and
# w3. The layers kept are the extremes the cells drive: 1 and 45 hold the
# largest and smallest FC2 global scale (1 / input_scale(w2): 6898.53 and
# 1.36898), 3 and 65 the largest and smallest input global scale
# (1 / input_scale(w1): 9039.45 and 23.0112), and 71 is the last layer.
_ML4_INPUT_SCALES = {  # layer: (input_scale of w1 and w3, input_scale of w2)
    1: (2.2411346e-04, 1.449585e-04),
    3: (1.1062622e-04, 4.4822693e-04),
    45: (1.0864258e-02, 7.3046875e-01),
    65: (4.345703e-02, 1.5136719e-01),
    71: (2.0751953e-02, 2.6953125e-01),
}
# `weight_scale_2` extremes over all 9088 routed experts: w1 (== w3) and w2.
_ML4_WS13 = (1.7620268e-05, 1.6348703e-04)
_ML4_WS2 = (1.7438617e-05, 3.226144e-04)
# Block scales are drawn as e4m3 bytes, uniformly. Weights: 8..448 (the
# checkpoint's expert block scales read 12..448). Activations: the whole
# non-negative e4m3 range -- 0, the subnormals, through 448, where a static
# input global scale saturates a live activation past its calibrated amax.
_ML4_WEIGHT_SF_BYTES = (80, 126)
_ML4_ACT_SF_BYTES = (0, 126)
_ML4_HIDDEN = _ML4_INTER = 6144
_ML4_LOCAL = 32
_ML4_LOCAL16 = 16


def _ml4_elt_ulp(spec) -> float:
    """The element band of a Mistral Large 4 cell: 16 ulp at an expert's whole
    width 6144, scaled by sqrt(6144 / I) at a width slice I -- 32 at tp4's
    1536, 45.25 at tp8's 768. The derivation and the measurements are in
    `_compare`."""
    return 16.0 * math.sqrt(_ML4_INTER / spec["intermediate"])


_ML4_STACK = None
_ML4_STACK16 = None


def _ml4_build_stack(local: int, seed: int):
    """`local` whole experts at the checkpoint's magnitudes: kernel-ready, and
    as reference operands.

    Local experts 0 and 1 carry the checkpoint's smallest and largest w1/w3
    `weight_scale_2`, `local - 1` and `local - 2` its smallest and largest w2
    one, the rest are log-uniform between. The reference operands fold each
    expert's global scale into fp32 block scales.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)
    lo, hi = _ML4_WEIGHT_SF_BYTES

    def stack(n, k):
        codes = torch.randint(0, 16, (local, n, k), dtype=torch.uint8, device=DEV, generator=gen)
        sf = torch.randint(
            lo, hi + 1, (local, n, k // SV), dtype=torch.uint8, device=DEV, generator=gen
        ).view(torch.float8_e4m3fn)
        return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous(), sf, codes

    def global_scales(smallest, largest):
        v = torch.empty(local, dtype=torch.float64, device=DEV)
        v.uniform_(math.log(smallest), math.log(largest), generator=gen)
        v = v.exp().float()
        v[0], v[1] = smallest, largest
        return v

    up_p, up_s, up_c = stack(_ML4_INTER, _ML4_HIDDEN)
    gt_p, gt_s, gt_c = stack(_ML4_INTER, _ML4_HIDDEN)
    dn_p, dn_s, dn_c = stack(_ML4_HIDDEN, _ML4_INTER)
    ws13 = global_scales(*_ML4_WS13)
    ws2 = global_scales(*_ML4_WS2).flip(0).contiguous()
    w1, s1 = _prep_fc1(up_p, gt_p, up_s, gt_s)
    w2, s2 = _prep_fc2(dn_p, dn_s)

    def fold(sf, ws):
        return sf.float() * ws.view(-1, 1, 1)

    return dict(
        gemm1_weights=w1,
        gemm1_weights_scale=s1,
        gemm2_weights=w2,
        gemm2_weights_scale=s2,
        ws13=ws13,
        ws2=ws2,
        up=(up_c, fold(up_s, ws13)),
        gate=(gt_c, fold(gt_s, ws13)),
        down=(dn_c, fold(dn_s, ws2)),
        raw=(up_p, gt_p, up_s, gt_s, dn_p, dn_s),
    )


def _ml4_stack():
    """The 32 experts one tep4 rank holds (dep4's too). Every rank's window
    holds this one stack -- the offset only names which global ids it answers
    for. ~8 GiB, built once."""
    global _ML4_STACK
    if _ML4_STACK is None:
        _ML4_STACK = _ml4_build_stack(_ML4_LOCAL, seed=61446144)
    return _ML4_STACK


def _ml4_stack16():
    """The 16 experts one dep8 rank holds, every rank's window alike. ~4 GiB,
    built once."""
    global _ML4_STACK16
    if _ML4_STACK16 is None:
        _ML4_STACK16 = _ml4_build_stack(_ML4_LOCAL16, seed=61441616)
    return _ML4_STACK16


def _ml4_scalars(layer: int, ws13: torch.Tensor, ws2: torch.Tensor):
    """The three per-local-expert scalars, computed in fp32 the way a loader does.

    `output1_scale_gate_scalar = input_scale(w1) * weight_scale_2(w1)`
    dequantizes the FC1 accumulator; `output1_scale_scalar` divides that by
    `input_scale(w2)`, applying the FC2 input's global scale to the linear
    half; `output2_scale_scalar = input_scale(w2) * weight_scale_2(w2)`
    dequantizes the FC2 accumulator. Returned in the op's argument order.
    """
    isc1, isc2 = (
        torch.tensor(v, dtype=torch.float32, device=DEV) for v in _ML4_INPUT_SCALES[layer]
    )
    gate = isc1 * ws13
    return (gate / isc2).contiguous(), gate.contiguous(), (isc2 * ws2).contiguous()


def _ml4_activation(tokens: int, layer: int, gen: torch.Generator):
    """NVFP4 hidden states in the layout `fp4_quantize(x, 1 / input_scale(w1),
    16, False, False)` returns, and their exact fp32 value."""
    codes = torch.randint(
        0, 16, (tokens, _ML4_HIDDEN), dtype=torch.uint8, device=DEV, generator=gen
    )
    lo, hi = _ML4_ACT_SF_BYTES
    sf = torch.randint(
        lo, hi + 1, (tokens, _ML4_HIDDEN // SV), dtype=torch.uint8, device=DEV, generator=gen
    )
    data = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    scale = sf.view(torch.float8_e4m3fn).float().repeat_interleave(SV, dim=1)
    x = E2M1[codes.long()] * scale * _ML4_INPUT_SCALES[layer][0]
    return data, sf.reshape(-1).view(torch.float8_e4m3fn).contiguous(), x


def _ml4_routing(spec, tokens: int, gen: torch.Generator, far: int = 0):
    """Pre-routed ids and bf16 weights, the way renorm_moe_routing_op emits them.

    Top-k of the logits, softmax over those k. Token 0 always has a slot in
    the window, so no call is vacuously all-zero. With `far`, slot 0 of every
    token is one of the window's last `far` experts, in turn (tep4: local 29,
    30, 31). With `skew="one_window"` every slot of every token lands in the
    window: its last expert on every token (tep4: local 31, tep8 and dep8:
    15), three distinct others drawn from the rest of it. With
    `skew="top_experts"` every token's slots are the window's last `top_k`
    experts, in an order drawn per token.
    """
    num_experts, top_k = spec["num_experts"], spec["top_k"]
    off, local = spec["local_expert_offset"], spec["local_num_experts"]
    if spec.get("skew") == "top_experts":
        order = torch.rand(tokens, top_k, device=DEV, generator=gen).argsort(dim=1)
        weights = torch.softmax(torch.randn(tokens, top_k, device=DEV, generator=gen), dim=-1)
        ids = off + local - top_k + order
        return ids.to(torch.int32).contiguous(), weights.to(torch.bfloat16).contiguous()
    if spec.get("skew") == "one_window":
        rest = torch.rand(tokens, local - 1, device=DEV, generator=gen).argsort(dim=1)
        ids = torch.cat(
            [
                torch.full((tokens, 1), local - 1, device=DEV, dtype=torch.long),
                rest[:, : top_k - 1],
            ],
            dim=1,
        )
        weights = torch.softmax(torch.randn(tokens, top_k, device=DEV, generator=gen), dim=-1)
        return (ids + off).to(torch.int32).contiguous(), weights.to(torch.bfloat16).contiguous()
    logits = torch.randn(tokens, num_experts, device=DEV, generator=gen)
    force = torch.zeros_like(logits)
    rows = torch.arange(tokens, device=DEV)
    if far:
        force[rows, off + local - far + rows % far] = 1e4
    else:
        force[0, off + 7] = 1e4
    ids = torch.topk(logits + force, top_k, dim=-1).indices
    weights = torch.softmax(logits.gather(1, ids), dim=-1)
    return ids.to(torch.int32).contiguous(), weights.to(torch.bfloat16).contiguous()


def _ml4_padding_rows(spec, tokens: int) -> torch.Tensor:
    """Which of the gathered rows are padding, for `spec["rank_tokens"]`: each
    rank's live rows lead its `T_max = max(rank_tokens)` gathered rows, and the
    rest of them are padding. With `chunk`, the call is that `tokens`-row slice
    of the gathered rows (dep8 issues them in chunks); without, all of them
    (dep4's one call)."""
    rank_tokens = spec["rank_tokens"]
    t_max = max(rank_tokens)
    gathered = len(rank_tokens) * t_max
    pad = torch.ones(gathered, dtype=torch.bool, device=DEV)
    for rank, live in enumerate(rank_tokens):
        pad[rank * t_max : rank * t_max + live] = False
    lo = spec.get("chunk", 0) * tokens
    whole = lo + tokens <= gathered if "chunk" in spec else tokens == gathered
    assert whole, (tokens, rank_tokens, spec.get("chunk"))
    return pad[lo : lo + tokens]


def _ml4_pad(args, ref, pad: torch.Tensor):
    """Padding rows as the dep4 target hands them to the op: `fp4_quantize` of
    an all-zero row -- every code and every block scale 0, read back from that
    op -- routed the way renorm_moe_routing_op routes the all-+0.0 logits the
    router makes of it, to experts 0..top_k-1 at 1/top_k each."""
    tokens, top_k = args["topk_ids"].shape
    rows = pad.unsqueeze(1)
    scale = args["hidden_states_scale"].view(torch.uint8).view(tokens, -1).masked_fill(rows, 0)
    ids = torch.where(rows, torch.arange(top_k, dtype=torch.int32, device=DEV), args["topk_ids"])
    weights = torch.where(
        rows,
        torch.tensor(1.0 / top_k, dtype=torch.bfloat16, device=DEV),
        args["topk_weights"],
    )
    args = dict(
        args,
        hidden_states=args["hidden_states"].masked_fill(rows, 0),
        hidden_states_scale=scale.reshape(-1).view(torch.float8_e4m3fn),
        topk_ids=ids.contiguous(),
        topk_weights=weights.contiguous(),
    )
    ref = dict(
        ref,
        hidden_states=ref["hidden_states"].masked_fill(rows, 0.0),
        topk_ids=args["topk_ids"],
        topk_weights=args["topk_weights"],
    )
    return args, ref


def _ml4_stack_for(spec):
    """The expert stack a Mistral Large 4 cell's geometry names: tep4's (and dep4's)
    32 whole experts, tep8's and dep8's 16, or tp4's and tp8's 128 experts at
    their 1536- and 768-wide slices."""
    geometry = (spec["hidden"], spec["intermediate"], spec["local_num_experts"])
    if geometry == (_ML4_HIDDEN, _ML4_INTER, _ML4_LOCAL):
        return _ml4_stack()
    if geometry == (_ML4_HIDDEN, _ML4_INTER, _ML4_LOCAL16):
        return _ml4_stack16()
    assert geometry in (
        (_ML4_HIDDEN, _ML4_TP4_INTER, _ML4_TP4_LOCAL),
        (_ML4_HIDDEN, _ML4_TP8_INTER, _ML4_TP8_LOCAL),
    ), spec
    return _ml4_width_stack(spec["intermediate"])


def _ml4_args(spec, tokens: int, seed: int, far: int = 0):
    """One Mistral Large 4 call (tep4, dep4 over gathered tokens, tep8 on a 16-wide
    window, dep8 on one over a chunk of the gathered tokens, or tp4 or tp8 on
    one width slice): the op's arguments and `reference`'s. With `skew="padded_zero_rows"` the rows `rank_tokens`
    leaves to padding are zero rows, routed as dep4 and dep8 route them."""
    st = _ml4_stack_for(spec)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    data, sf, x = _ml4_activation(tokens, spec["layer"], gen)
    ids, wts = _ml4_routing(spec, tokens, gen, far=far)
    s1, sg, s2 = _ml4_scalars(spec["layer"], st["ws13"], st["ws2"])
    args = dict(
        routing_logits=None,
        routing_bias=None,
        hidden_states=data,
        hidden_states_scale=sf,
        gemm1_weights=st["gemm1_weights"],
        gemm1_weights_scale=st["gemm1_weights_scale"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=st["gemm2_weights"],
        gemm2_weights_scale=st["gemm2_weights_scale"],
        gemm2_bias=None,
        output1_scale_scalar=s1,
        output1_scale_gate_scalar=sg,
        output2_scale_scalar=s2,
        num_experts=spec["num_experts"],
        top_k=spec["top_k"],
        n_group=None,
        topk_group=None,
        intermediate_size=spec["intermediate"],
        local_expert_offset=spec["local_expert_offset"],
        local_num_experts=spec["local_num_experts"],
        routed_scaling_factor=None,
        routing_method_type=1,
        do_finalize=True,
        act_type=0,
        topk_weights=wts,
        topk_ids=ids,
        tune_max_num_tokens=spec["tune_max_num_tokens"],
    )
    ref = dict(
        hidden_states=x,
        topk_ids=ids,
        topk_weights=wts,
        up=st["up"],
        gate=st["gate"],
        down=st["down"],
        fc2_global_scale=1.0 / _ML4_INPUT_SCALES[spec["layer"]][1],
        local_expert_offset=spec["local_expert_offset"],
        local_num_experts=spec["local_num_experts"],
    )
    if spec.get("skew") == "padded_zero_rows":
        args, ref = _ml4_pad(args, ref, _ml4_padding_rows(spec, tokens))
    return args, ref


#: The arguments that change every step; the rest is weights and configuration.
_ML4_PER_STEP = ("hidden_states", "hidden_states_scale", "topk_ids", "topk_weights")


def _ml4_prefix(args, ref, m: int):
    """The same call on its first `m` tokens, every operand contiguous."""
    a = dict(
        args,
        hidden_states=args["hidden_states"][:m],
        hidden_states_scale=args["hidden_states_scale"][: m * _ML4_HIDDEN // SV],
        topk_ids=args["topk_ids"][:m],
        topk_weights=args["topk_weights"][:m],
    )
    r = dict(
        ref,
        hidden_states=ref["hidden_states"][:m],
        topk_ids=ref["topk_ids"][:m],
        topk_weights=ref["topk_weights"][:m],
    )
    return a, r


#: Tokens per slice of the reference. A token's row depends on that token
#: alone; in one piece, the fp32 reference of 131072 gathered tokens with every
#: slot in one window took 30.7 GiB beyond its operands.
_ML4_REF_TOKENS = 65536


def _ml4_reference(ref) -> torch.Tensor:
    """The entry's `reference`, cross-checked bitwise against this file's
    `_ref_moe` on the same inputs, rounded to bf16 for the gate -- over slices
    of `_ML4_REF_TOKENS` tokens, which bounds its memory."""
    tokens = ref["hidden_states"].shape[0]
    rows = []
    for start in range(0, tokens, _ML4_REF_TOKENS):
        part = dict(ref)
        for name in ("hidden_states", "topk_ids", "topk_weights"):
            part[name] = ref[name][start : start + _ML4_REF_TOKENS]
        entry_ref = moe.reference(**part)
        local_ref = _ref_moe(
            part["hidden_states"],
            part["topk_ids"],
            part["topk_weights"],
            part,
            part["fc2_global_scale"],
            offset=part["local_expert_offset"],
            num_local=part["local_num_experts"],
        )
        torch.testing.assert_close(entry_ref, local_ref, rtol=0.0, atol=0.0)
        rows.append(entry_ref.to(torch.bfloat16))
        del entry_ref, local_ref
    return torch.cat(rows)


def _ml4_assert_window(out: torch.Tensor, args) -> None:
    """A token with no slot in the window is bitwise zero; some token has one."""
    local = args["topk_ids"].long() - args["local_expert_offset"]
    live = ((local >= 0) & (local < args["local_num_experts"])).any(dim=-1)
    assert bool(live.any()), "no token has a slot in the window: the call checks nothing"
    assert bool((out[~live] == 0).all()), "a token with no slot in the window must be zero"


def _ml4_replay(spec, tokens: int, seed: int) -> None:
    """Capture one call, replay it on new tokens and new routing: bitwise the
    eager call on that data, which is gated against `reference` on the element
    figure (a decode step has few live rows; `_compare` has why).

    At a non-zero window the new routing holds slots both inside the window
    and outside it, and the same data at offset 0 lands far outside the gate,
    so a graph that kept window 0's offset would fail here."""
    static, _ = _ml4_args(spec, tokens, seed)
    fresh, fresh_ref = _ml4_args(spec, tokens, seed + 7919)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        moe(**static)  # the runner and its kernels' modules, before capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = moe(**static)[0]
    for name in _ML4_PER_STEP:
        static[name].copy_(fresh[name])
    graph.replay()
    eager = moe(**fresh)[0]
    assert torch.equal(captured, eager), f"T={tokens}: graph replay differs from eager"
    ref = _ml4_reference(fresh_ref)
    elt, rms = _dev(eager, ref)
    print(f"  graph T={tokens}: {elt:.2f} elt / {rms:.3f} rms ulp")
    _compare(eager, ref, _ml4_elt_ulp(spec), rms_ulp=None)
    _ml4_assert_window(eager, fresh)
    off = spec["local_expert_offset"]
    if off:
        local = fresh["topk_ids"].long() - off
        inside = int(((local >= 0) & (local < spec["local_num_experts"])).sum())
        assert 0 < inside < local.numel(), f"T={tokens}: {inside} of {local.numel()} slots inside"
        _, far = _dev(moe(**dict(fresh, local_expert_offset=0))[0], ref)
        print(f"    {inside} of {local.numel()} slots inside; at offset 0: {far:.1f} rms ulp")
        assert far > 20.0, f"T={tokens}: offset 0 is only {far:.2f} rms ulp from offset {off}"


def _ml4_counts(spec) -> tuple:
    """The token counts a cell drives: `sweep`, or every count of the inclusive
    `sweep_range` and then `sweep_also`, or `tokens` alone."""
    if "sweep_range" in spec:
        lo, hi = spec["sweep_range"]
        return (*range(lo, hi + 1), *spec["sweep_also"])
    return spec.get("sweep", (spec["tokens"],))


def _ml4_sweep(spec, seed: int) -> None:
    """The largest count gated against `reference`; every smaller count bitwise
    the leading rows of that call, so each is gated too."""
    counts = _ml4_counts(spec)
    assert len(set(counts)) == len(counts), "a count is driven twice"
    top = max(counts)
    assert top == spec["tokens"], (top, spec["tokens"])
    args, ref = _ml4_args(spec, top, seed)
    if spec.get("skew") == "one_window":
        rows = torch.bincount(
            args["topk_ids"].flatten().long() - spec["local_expert_offset"],
            minlength=spec["local_num_experts"],
        )
        assert (
            rows.numel() == spec["local_num_experts"] and int(rows.sum()) == top * spec["top_k"]
        ), rows
        assert int(rows.max()) == top, f"no local expert carries all {top} tokens: {rows.max()}"
    if spec.get("skew") == "top_experts":
        last = spec["local_expert_offset"] + spec["local_num_experts"]
        rows = torch.bincount(args["topk_ids"].flatten().long(), minlength=spec["num_experts"])
        tail = rows[last - spec["top_k"] : last]
        assert bool((tail == top).all()) and int(rows.sum() - tail.sum()) == 0, tail
    pad = None
    if spec.get("skew") == "padded_zero_rows":
        pad = _ml4_padding_rows(spec, top)
        rows = torch.bincount(args["topk_ids"].flatten().long(), minlength=spec["num_experts"])
        assert int(rows[: spec["top_k"]].min()) >= int(pad.sum()) > 0, (int(pad.sum()), rows[:4])
    full = moe(**args)[0]
    assert full.shape == (top, spec["hidden"]) and full.dtype == torch.bfloat16, full.shape
    exp = _ml4_reference(ref)
    elt, rms = _dev(full, exp)
    print(f"  T={top}: {elt:.2f} elt / {rms:.3f} rms ulp; {len(counts)} counts")
    _compare(full, exp, _ml4_elt_ulp(spec))
    _ml4_assert_window(full, args)
    if pad is not None:
        assert bool((full[pad].view(torch.int16) == 0).all()), "a padding row is not +0.0"
    differ = [
        m
        for m in counts
        if m < top and not torch.equal(moe(**_ml4_prefix(args, ref, m)[0])[0], full[:m])
    ]
    assert not differ, f"token counts whose rows differ from the {top}-token call: {differ}"


# The Mistral Large 4 tep4 call the tests below drive, at window (0, 32).
_ML4_SPEC = next(c.spec for c in moe.CELLS if c.spec.get("layer") == 1 and "graph" in c.spec)

_BLOCK_ROUTING, _DYN_ROUTING = "routingIndicesBlockKernel<", "routingIndicesDynBlockKernel<"
_CLUSTER_ROUTING, _COOP_ROUTING = "routingIndicesClusterKernel<", "routingIndicesCoopKernel<"
_SCALAR_FIN, _VECTOR_FIN = "finalizeKernel<", "finalizeKernelVecLoad<"

#: (tokens, routing kernel, token tile of both GEMMs, finalize kernel) the
#: Mistral Large 4 tep4 call launches on the cold path, at each side of every
#: switch the sweep cell's token counts claim to straddle. The tile is
#: `nextPowerOfTwo(int(T * top_k / local_num_experts))` clamped to 8..256.
_ML4_KERNELS = (
    (1, _BLOCK_ROUTING, 8, _SCALAR_FIN),
    (4, _BLOCK_ROUTING, 8, _SCALAR_FIN),
    (5, _DYN_ROUTING, 8, _SCALAR_FIN),
    (12, _DYN_ROUTING, 8, _SCALAR_FIN),
    (13, _CLUSTER_ROUTING, 8, _SCALAR_FIN),
    (49, _CLUSTER_ROUTING, 8, _SCALAR_FIN),
    (50, _CLUSTER_ROUTING, 8, _VECTOR_FIN),
    (71, _CLUSTER_ROUTING, 8, _VECTOR_FIN),
    (72, _CLUSTER_ROUTING, 16, _VECTOR_FIN),
    (135, _CLUSTER_ROUTING, 16, _VECTOR_FIN),
    (136, _CLUSTER_ROUTING, 32, _VECTOR_FIN),
    (263, _CLUSTER_ROUTING, 32, _VECTOR_FIN),
    (264, _CLUSTER_ROUTING, 64, _VECTOR_FIN),
    (519, _CLUSTER_ROUTING, 64, _VECTOR_FIN),
    (520, _CLUSTER_ROUTING, 128, _VECTOR_FIN),
    (1031, _CLUSTER_ROUTING, 128, _VECTOR_FIN),
    (1032, _CLUSTER_ROUTING, 256, _VECTOR_FIN),
    (8192, _CLUSTER_ROUTING, 256, _VECTOR_FIN),
    (8193, _COOP_ROUTING, 256, _VECTOR_FIN),
    (32768, _COOP_ROUTING, 256, _VECTOR_FIN),
)


def test_sw_kernels_by_token_count():
    """The switches the Mistral Large 4 sweep cell straddles are where it says:
    read back from the kernels each call launches, on the cold path."""
    from torch.profiler import ProfilerActivity, profile

    args, ref = _ml4_args(_ML4_SPEC, 32768, seed=5)
    for tokens, routing, tile, finalize in _ML4_KERNELS:
        call = _ml4_prefix(args, ref, tokens)[0]
        moe(**call)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            moe(**call)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if str(e.device_type) == "DeviceType.CUDA"]
        gemms = [n for n in names if n.startswith("bmm_")]
        assert len(gemms) == 2 and all(f"_t128x{tile}x" in n for n in gemms), (tokens, gemms)
        for want in (routing, finalize):
            assert any(want in n for n in names), (tokens, want, names)
    print(f"  test_sw_kernels_by_token_count OK ({len(_ML4_KERNELS)} token counts)")


#: The token counts whose every valid tactic is driven: each decode count the
#: graphs capture, and each autotuner bucket above it (powers of two to 8192,
#: then `tune_max_num_tokens` 32768, which carries 8193..32768).
_ML4_TACTIC_TOKENS = (*range(1, 17), 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 32768)


def test_sw_every_valid_tactic_passes_the_gate():
    """What lets the cold cells certify the tuned path an engine warms up.

    The engine profiles this op at warm-up and keeps one tactic per token
    bucket; every call in the cells runs the cold fallback. So each tactic the
    op's own tunable runner offers at each count is run here: one whose bits
    equal the fallback's is certified by the cells, and one whose bits do not
    is gated against `reference` itself.
    """
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    spec = _ML4_SPEC
    summary = []
    for tokens in _ML4_TACTIC_TOKENS:
        args, ref = _ml4_args(spec, tokens, seed=300 + tokens)
        base = moe(**args)[0]
        runner = FP4BlockScaleMoERunner(
            spec["num_experts"],
            spec["top_k"],
            None,
            None,
            spec["intermediate"],
            spec["local_expert_offset"],
            spec["local_num_experts"],
            None,
            1,
            True,
            0,
            tune_max_num_tokens=spec["tune_max_num_tokens"],
        )
        inputs = [args[k] for k in ("routing_logits", "routing_bias", "hidden_states")] + [
            args[k]
            for k in (
                "hidden_states_scale",
                "gemm1_weights",
                "gemm1_weights_scale",
                "gemm1_bias",
                "gemm1_alpha",
                "gemm1_beta",
                "gemm1_clamp_limit",
                "gemm2_weights",
                "gemm2_weights_scale",
                "gemm2_bias",
                "output1_scale_scalar",
                "output1_scale_gate_scalar",
                "output2_scale_scalar",
                "topk_weights",
                "topk_ids",
            )
        ]
        valid = runner.get_runner().get_valid_configs(
            spec["top_k"], spec["hidden"], spec["intermediate"], spec["local_num_experts"], tokens
        )
        assert len({t[0] for t in valid}) > 1, f"T={tokens}: one tile only, {valid}"
        exp = None
        worst = (0.0, 0.0)
        differ = []
        for tactic in valid:
            y = runner.forward(inputs, list(tactic))[0]
            if torch.equal(y, base):
                continue
            differ.append(tuple(tactic))
            if exp is None:
                exp = _ml4_reference(ref)
                _compare(base, exp)
            _compare(y, exp)
            worst = tuple(max(a, b) for a, b in zip(worst, _dev(y, exp)))
        summary.append((tokens, len(valid), len(differ), worst))
        if differ:
            print(
                f"  T={tokens}: {len(differ)} of {len(valid)} tactics differ from the fallback, "
                f"all inside the gate (worst {worst[0]:.2f} elt / {worst[1]:.3f} rms): {differ}"
            )
    bitwise = [t for t, _, d, _ in summary if d == 0]
    print(
        "  test_sw_every_valid_tactic_passes_the_gate OK ("
        + ", ".join(f"T={t}: {n}" for t, n, _, _ in summary)
        + f" valid tactics; every tactic bitwise the fallback at T in {bitwise})"
    )


def test_sw_experts_past_2_31_elements_give_the_same_bits_below_it():
    """FC1 weights of local experts 29..31 start past 2^31 e2m1 elements
    (75,497,472 per expert); moved to slot 1, each expert's tokens come back
    bitwise the same."""
    spec = _ML4_SPEC
    args, ref = _ml4_args(spec, 64, seed=31, far=3)
    out = moe(**args)[0]
    _compare(out, _ml4_reference(ref))
    per_expert = args["gemm1_weights"][0].numel() * 2
    assert 28 * per_expert < 2**31 < 29 * per_expert, "experts 29..31 must start past 2^31"
    ids, off = args["topk_ids"], spec["local_expert_offset"]
    per_local = (
        "gemm1_weights",
        "gemm1_weights_scale",
        "gemm2_weights",
        "gemm2_weights_scale",
        "output1_scale_scalar",
        "output1_scale_gate_scalar",
        "output2_scale_scalar",
    )
    for far in (29, 30, 31):
        order = list(range(spec["local_num_experts"]))
        order[1], order[far] = far, 1
        index = torch.tensor(order, device=DEV)
        near_id, far_id = off + 1, off + far
        swapped = torch.where(ids == far_id, near_id, torch.where(ids == near_id, far_id, ids))
        moved = dict(args, topk_ids=swapped.to(torch.int32).contiguous())
        for name in per_local:
            moved[name] = args[name].index_select(0, index).contiguous()
        assert bool((ids == far_id).any()), f"no token reaches expert {far}"
        assert torch.equal(moe(**moved)[0], out), f"expert {far} differs from itself at slot 1"
        del moved
        torch.cuda.empty_cache()
    print("  test_sw_experts_past_2_31_elements_give_the_same_bits_below_it OK")


@pytest.mark.parametrize("layer", [1, 45])
def test_sw_wrong_variants_land_outside_the_gate(layer):
    """How far the wrong computations a Mistral Large 4 loader could make land from
    the result the gate accepts, at 1024 tokens over window (0, 32).

    Every one lands outside the 2 ulp RMS gate, and all but one more than ten
    times outside it. The exception is real, not a blind spot: swapping the two
    `output1_*` scalars moves the FC2 input's global scale from the linear half
    onto the gate's SiLU argument, and at layer 45 that scale is 1.36898 -- the
    two readings differ by that factor inside a SiLU only, and land 4.10 rms
    ulp apart. At layer 1 (6898.53) the same swap is 183.86 rms ulp out.
    """
    spec = dict(_ML4_SPEC, layer=layer)
    args, ref = _ml4_args(spec, 1024, seed=4242 + layer)
    y = moe(**args)[0]
    exp = _ml4_reference(ref)
    _compare(y, exp)
    st = _ml4_stack()
    scalars = ("output1_scale_scalar", "output1_scale_gate_scalar", "output2_scale_scalar")
    wrong_calls = {
        "output1 scalar roles swapped": dict(
            output1_scale_scalar=args["output1_scale_gate_scalar"],
            output1_scale_gate_scalar=args["output1_scale_scalar"],
        ),
        "per-expert scalars of the wrong experts": {
            k: args[k].flip(0).contiguous() for k in scalars
        },
        "w1's weight_scale_2 in output2_scale_scalar": dict(
            output2_scale_scalar=_ml4_scalars(layer, st["ws13"], st["ws13"])[2]
        ),
    }
    dist = {name: _dev(moe(**dict(args, **kw))[0], exp) for name, kw in wrong_calls.items()}
    swapped = moe.reference(**dict(ref, up=ref["gate"], gate=ref["up"]))
    dist["gate/up swapped"] = _dev(y, swapped.to(torch.bfloat16))
    unquantized = _ref_moe(
        ref["hidden_states"],
        ref["topk_ids"],
        ref["topk_weights"],
        ref,
        ref["fc2_global_scale"],
        quantize_intermediate=False,
    )
    dist["no FC1-output requantization"] = _dev(y, unquantized.to(torch.bfloat16))
    up_p, gt_p, up_s, gt_s, dn_p, dn_s = st["raw"]
    for name, prep in (("no scale swizzle", "swizzle"), ("no row shuffle", "shuffle")):
        w1, s1 = _prep_fc1(up_p, gt_p, up_s, gt_s, **{prep: False})
        w2, s2 = _prep_fc2(dn_p, dn_s, **{prep: False})
        layout = dict(
            gemm1_weights=w1, gemm1_weights_scale=s1, gemm2_weights=w2, gemm2_weights_scale=s2
        )
        dist[name] = _dev(moe(**dict(args, **layout))[0], exp)
        del w1, s1, w2, s2, layout
        torch.cuda.empty_cache()
    print(f"  layer {layer}: correct {_dev(y, exp)}")
    for name, (elt, rms) in dist.items():
        print(f"    {name:44s} {elt:10.3g} elt / {rms:8.2f} rms ulp")
        assert rms > 2.0, f"{name} is {rms:.2f} rms ulp from the correct result: inside the gate"
        if not (layer == 45 and name == "output1 scalar roles swapped"):
            assert rms > 20.0, f"{name} is only {rms:.2f} rms ulp from the correct result"


# ── Mistral Large 4 dep4: the same windows over the four ranks' gathered tokens ─


def _ml4_dep4_cell(skew: str):
    """The dep4 cell (`tune_max_num_tokens` 131072) with this `skew`."""
    return next(
        c
        for c in moe.CELLS
        if c.spec.get("tune_max_num_tokens") == 131072 and c.spec.get("skew") == skew
    )


_ML4_DEP4_ONE_WINDOW = _ml4_dep4_cell("one_window")
_ML4_DEP4_PADDED = _ml4_dep4_cell("padded_zero_rows")
# Window (0, 32), layer 1, 131072 gathered tokens, routed uniformly.
_ML4_DEP4_UNIFORM = {k: v for k, v in _ML4_DEP4_ONE_WINDOW.spec.items() if k != "skew"}

#: Gathered token counts the dep4 kernel test reads back under uniform routing.
_ML4_DEP4_KERNEL_TOKENS = (8193, 32768, 32769, 65536, 131072)


def test_sw_dep4_kernels_by_token_count():
    """From 8193 to 131072 gathered tokens the call launches the same kernels
    -- the expert-count init and the cooperative routing kernel, both GEMMs at
    token tile 256, the vector-load finalize -- under uniform routing at each
    count, and at 131072 with every slot in one window and under the padding
    skew. So the sweep cell's counts sit on one kernel set."""
    from torch.profiler import ProfilerActivity, profile

    args, ref = _ml4_args(_ML4_DEP4_UNIFORM, 131072, seed=7)
    calls = [(f"uniform T={m}", _ml4_prefix(args, ref, m)[0]) for m in _ML4_DEP4_KERNEL_TOKENS]
    del ref
    for cell in (_ML4_DEP4_ONE_WINDOW, _ML4_DEP4_PADDED):
        calls.append((cell.spec["skew"], _ml4_args(cell.spec, 131072, seed=7)[0]))
    for name, call in calls:
        moe(**call)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            moe(**call)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if str(e.device_type) == "DeviceType.CUDA"]
        gemms = [n for n in names if n.startswith("bmm_")]
        assert len(gemms) == 2 and all("_t128x256x" in n for n in gemms), (name, gemms)
        for want in ("routingInitExpertCounts<", _COOP_ROUTING, _VECTOR_FIN):
            assert any(want in n for n in names), (name, want, names)
    print(f"  test_sw_dep4_kernels_by_token_count OK ({len(calls)} calls)")


def test_sw_dep4_every_valid_tactic_passes_the_gate():
    """The tuned path at dep4. With `tune_max_num_tokens` 131072 the tuner
    profiles 8193..131072 as one bucket at 131072 and keeps one tactic for
    all of it. Every valid tactic is run at 8193, 65537 and 131072 gathered
    tokens under uniform routing, and at 131072 with every slot in one window
    and under the padding skew: one whose bits equal the fallback's is
    certified by the cells, and one whose bits do not is gated against
    `reference` itself."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    uniform, uniform_ref = _ml4_args(_ML4_DEP4_UNIFORM, 131072, seed=9)
    calls = [
        (f"uniform T={m}", *_ml4_prefix(uniform, uniform_ref, m)) for m in (8193, 65537, 131072)
    ]
    for cell in (_ML4_DEP4_ONE_WINDOW, _ML4_DEP4_PADDED):
        calls.append((cell.spec["skew"], *_ml4_args(cell.spec, 131072, seed=9)))
    spec = _ML4_DEP4_UNIFORM
    runner = FP4BlockScaleMoERunner(
        spec["num_experts"],
        spec["top_k"],
        None,
        None,
        spec["intermediate"],
        spec["local_expert_offset"],
        spec["local_num_experts"],
        None,
        1,
        True,
        0,
        tune_max_num_tokens=spec["tune_max_num_tokens"],
    )
    names = ("routing_logits", "routing_bias", "hidden_states", "hidden_states_scale")
    names += ("gemm1_weights", "gemm1_weights_scale", "gemm1_bias", "gemm1_alpha", "gemm1_beta")
    names += ("gemm1_clamp_limit", "gemm2_weights", "gemm2_weights_scale", "gemm2_bias")
    names += ("output1_scale_scalar", "output1_scale_gate_scalar", "output2_scale_scalar")
    names += ("topk_weights", "topk_ids")
    summary = []
    for name, args, ref in calls:
        tokens = args["hidden_states"].shape[0]
        base = moe(**args)[0]
        inputs = [args[k] for k in names]
        valid = runner.get_runner().get_valid_configs(
            spec["top_k"], spec["hidden"], spec["intermediate"], spec["local_num_experts"], tokens
        )
        assert len({t[0] for t in valid}) > 1, f"{name}: one tile only, {valid}"
        exp = None
        differ = []
        for tactic in valid:
            y = runner.forward(inputs, list(tactic))[0]
            if torch.equal(y, base):
                continue
            differ.append(tuple(tactic))
            if exp is None:
                exp = _ml4_reference(ref)
                _compare(base, exp)
            _compare(y, exp)
        summary.append(f"{name}: {len(valid)} valid, {len(differ)} differ {differ}")
    print("  test_sw_dep4_every_valid_tactic_passes_the_gate OK (" + "; ".join(summary) + ")")


def test_sw_dep4_four_windows_sum_to_the_layer():
    """The four ranks' calls over one gathered token set answer every slot
    exactly once: their partials, summed, pass the gate against the summed
    window references -- in fp32, and as a bf16 running sum in rank order.
    Leaving one window out, or running window (0, 32) in place of (32, 32),
    lands far outside it. Gathered layout: ranks at 2048, 700, 1 and 1
    tokens, so 8192 rows, 5442 of them padding."""
    spec = dict(_ML4_DEP4_PADDED.spec, tokens=8192, rank_tokens=(2048, 700, 1, 1))
    args, ref = _ml4_args(spec, 8192, seed=11)
    offsets = (0, 32, 64, 96)
    ids = args["topk_ids"]
    slots = [int(((ids >= o) & (ids < o + spec["local_num_experts"])).sum()) for o in offsets]
    assert sum(slots) == ids.numel() and min(slots) > 0, slots
    outs = [moe(**dict(args, local_expert_offset=o))[0] for o in offsets]
    want = sum(moe.reference(**dict(ref, local_expert_offset=o)) for o in offsets)
    want = want.to(torch.bfloat16)
    summed = sum(y.float() for y in outs).to(torch.bfloat16)
    running = ((outs[0] + outs[1]) + outs[2]) + outs[3]
    for name, y in (("fp32 sum", summed), ("bf16 running sum", running)):
        elt, rms = _dev(y, want)
        print(f"  four windows, {name}: {elt:.2f} elt / {rms:.3f} rms ulp")
        _compare(y, want)
    wrong = {
        "window (96, 32) left out": outs[:3],
        "window (0, 32) in place of (32, 32)": (outs[0], outs[0], outs[2], outs[3]),
    }
    for name, partials in wrong.items():
        y = sum(p.float() for p in partials)
        elt, rms = _dev(y.to(torch.bfloat16), want)
        print(f"    {name}: {elt:.3g} elt / {rms:.2f} rms ulp")
        assert rms > 20.0, f"{name} is only {rms:.2f} rms ulp from the summed layer"


def test_sw_dep4_one_window_rows_cross_2_31():
    """The one-window dep4 cell's own draw puts per-slot FC2 rows past both
    2^31 marks of the op's bf16 FC2 buffer -- row 174763, where a row's byte
    offset passes 2^31, and row 349526, where its element offset does -- read
    back from the op's expanded-slot -> row map (`do_finalize=False`).
    Gathered and weighted in torch they equal the finalized output bitwise, so
    the cell's gate reads rows past both marks."""
    cell = _ML4_DEP4_ONE_WINDOW
    spec = cell.spec
    args, _ = _ml4_args(spec, spec["tokens"], cell.seed_base() % 2**31)
    out = moe(**args)[0]
    rows, _, slot_rows = moe(**dict(args, do_finalize=False))
    byte_mark = 2**31 // (2 * spec["hidden"]) + 1
    elt_mark = 2**31 // spec["hidden"] + 1
    past_bytes = int((slot_rows >= byte_mark).sum())
    past_elts = int((slot_rows >= elt_mark).sum())
    print(
        f"  {tuple(rows.shape)} per-slot rows; slot rows up to {int(slot_rows.max())}: "
        f"{past_bytes} at or past {byte_mark}, {past_elts} at or past {elt_mark}"
    )
    assert past_elts > 0 and past_bytes > past_elts, (past_bytes, past_elts)
    finalized = torch.zeros(out.shape, dtype=torch.float32, device=DEV)
    weights = args["topk_weights"].float()
    for j in range(spec["top_k"]):
        finalized += rows[slot_rows[:, j].long()].float() * weights[:, j].unsqueeze(1)
    assert torch.equal(finalized.to(torch.bfloat16), out), (
        "the per-slot rows, gathered and weighted, differ from the finalized output"
    )


# ── Mistral Large 4 tp4 and tp8: all 128 experts on every rank, each cut along its width ─

# moe_expert_parallel_size 1: every rank holds all 128 experts of a layer, each cut to its
# slice of the intermediate -- 6144 / 4 at tp4 (moe_tensor_parallel_size 4), 6144 / 8 at tp8.
_ML4_WIDTH_LOCAL = 128
_ML4_TP4_INTER = 1536
_ML4_TP4_LOCAL = _ML4_WIDTH_LOCAL
_ML4_TP8_INTER = 768
_ML4_TP8_LOCAL = _ML4_WIDTH_LOCAL


#: Rank 0's width slice of each width drawn so far, with its reference operands.
_ML4_WIDTH_STACKS = {}


def _ml4_width_global_scales(inter: int):
    """The width-sliced layer's per-expert `weight_scale_2`, (w1 == w3, w2),
    fp32 [128].

    Unsplit, so every width slice carries the same values. Experts 0 and 1
    hold the checkpoint's smallest and largest w1/w3 one, 127 and 126 its
    smallest and largest w2 one, the rest are log-uniform between. The seed
    names the two widths, 6144 and the slice's `inter`.
    """
    gen = torch.Generator(device=DEV).manual_seed(int(f"{_ML4_INTER}{inter}"))

    def draw(smallest, largest):
        v = torch.empty(_ML4_WIDTH_LOCAL, dtype=torch.float64, device=DEV)
        v.uniform_(math.log(smallest), math.log(largest), generator=gen)
        v = v.exp().float()
        v[0], v[1] = smallest, largest
        return v

    return draw(*_ML4_WS13), draw(*_ML4_WS2).flip(0).contiguous()


def _unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    """One e2m1 code per byte, from two per byte (the even element in the low nibble)."""
    return torch.stack([packed & 0x0F, packed >> 4], dim=-1).flatten(-2)


def _ml4_fold(sf: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """e4m3 block scales `[E, M, C]` with each expert's global scale folded in, fp32."""
    return sf.float() * ws.view(-1, 1, 1)


def _ml4_width_slice(inter: int, rank: int, reference: bool = True):
    """Rank `rank`'s slice, `inter` wide, of a width-sliced layer's 128 experts.

    The layer is drawn one slice at a time, slice r from its own seed, so a
    whole expert is its slices side by side: w1 and w3 output rows
    [inter*r, inter*(r + 1)) are slice r's FC1 rows, and w2 input columns
    [inter*r, inter*(r + 1)) -- bytes [inter*r/2, ...), block-scale columns
    [inter*r/16, ...) -- its FC2 columns. The layout is applied to the slice,
    as the tp8 loader applies it: at tp8's 768, FC1 [128, 1536, 3072] u8 and
    its scales [128, 1536, 384], FC2 [128, 6144, 384] u8 and its scales
    [128, 6144, 48]. Codes and block scales are drawn as in tep4's stack.
    `raw` keeps the packed codes and e4m3 scales; with `reference`, the
    reference operands (one code per byte, each expert's global scale folded
    into fp32 block scales) come too. ~4.3 GiB with them at 768, ~2 GiB
    without.
    """
    gen = torch.Generator(device=DEV).manual_seed(int(f"{inter}{_ML4_INTER}") + rank)
    lo, hi = _ML4_WEIGHT_SF_BYTES

    def stack(n, k):
        codes = torch.randint(
            0, 16, (_ML4_WIDTH_LOCAL, n, k), dtype=torch.uint8, device=DEV, generator=gen
        )
        sf = torch.randint(
            lo, hi + 1, (_ML4_WIDTH_LOCAL, n, k // SV), dtype=torch.uint8, device=DEV, generator=gen
        ).view(torch.float8_e4m3fn)
        return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous(), sf

    up_p, up_s = stack(inter, _ML4_HIDDEN)
    gt_p, gt_s = stack(inter, _ML4_HIDDEN)
    dn_p, dn_s = stack(_ML4_HIDDEN, inter)
    ws13, ws2 = _ml4_width_global_scales(inter)
    w1, s1 = _prep_fc1(up_p, gt_p, up_s, gt_s)
    w2, s2 = _prep_fc2(dn_p, dn_s)
    st = dict(
        gemm1_weights=w1,
        gemm1_weights_scale=s1,
        gemm2_weights=w2,
        gemm2_weights_scale=s2,
        ws13=ws13,
        ws2=ws2,
        raw=(up_p, gt_p, up_s, gt_s, dn_p, dn_s),
    )
    if reference:
        st.update(
            up=(_unpack_codes(up_p), _ml4_fold(up_s, ws13)),
            gate=(_unpack_codes(gt_p), _ml4_fold(gt_s, ws13)),
            down=(_unpack_codes(dn_p), _ml4_fold(dn_s, ws2)),
        )
    return st


def _ml4_width_stack(inter: int):
    """Rank 0's slice at width `inter`, with its reference operands; built once."""
    if inter not in _ML4_WIDTH_STACKS:
        _ML4_WIDTH_STACKS[inter] = _ml4_width_slice(inter, 0)
    return _ML4_WIDTH_STACKS[inter]


#: Experts per piece of the whole-width reference: at I 6144 one piece of 16
#: experts is 1.8 GiB of codes, one per byte.
_ML4_WIDTH_REF_EXPERTS = 16


def _cat_e4m3(scales, dim: int) -> torch.Tensor:
    return torch.cat([s.view(torch.uint8) for s in scales], dim=dim).view(torch.float8_e4m3fn)


def _ml4_whole_width_reference(ref, raws, ws13, ws2) -> torch.Tensor:
    """The entry's `reference` over whole experts (I 6144), fp32.

    Each expert's slices are put back side by side in rank order -- w1 and w3
    codes and block scales by rows, w2 codes and block scales by columns --
    and the reference runs over windows of `_ML4_WIDTH_REF_EXPERTS` experts,
    its own expert-parallel split, whose outputs are summed.
    """
    out = None
    for e0 in range(0, _ML4_WIDTH_LOCAL, _ML4_WIDTH_REF_EXPERTS):
        e = slice(e0, e0 + _ML4_WIDTH_REF_EXPERTS)
        up = (
            torch.cat([_unpack_codes(r[0][e]) for r in raws], dim=1),
            _ml4_fold(_cat_e4m3([r[2][e] for r in raws], dim=1), ws13[e]),
        )
        gate = (
            torch.cat([_unpack_codes(r[1][e]) for r in raws], dim=1),
            _ml4_fold(_cat_e4m3([r[3][e] for r in raws], dim=1), ws13[e]),
        )
        down = (
            torch.cat([_unpack_codes(r[4][e]) for r in raws], dim=2),
            _ml4_fold(_cat_e4m3([r[5][e] for r in raws], dim=2), ws2[e]),
        )
        assert up[0].shape[1:] == down[0].shape[1:] == (_ML4_HIDDEN, _ML4_INTER), up[0].shape
        part = moe.reference(
            **dict(
                ref,
                up=up,
                gate=gate,
                down=down,
                local_expert_offset=e0,
                local_num_experts=_ML4_WIDTH_REF_EXPERTS,
            )
        )
        out = part if out is None else out.add_(part)
        del up, gate, down, part
    return out


_ML4_WIDTH_LAYOUT = ("gemm1_weights", "gemm1_weights_scale", "gemm2_weights", "gemm2_weights_scale")


def _ml4_width_slices(spec, seed: int) -> None:
    """The ranks' calls over one token set, each on its own width slice of the
    same 128 experts with the same scalars, sum to the whole-width layer:
    their partials, summed in fp32 and as a bf16 running sum in rank order,
    pass the gate against the entry's `reference` over the whole experts
    (I 6144). Leaving the last slice out, or running slice 0 in place of
    slice 1, lands far outside it."""
    tokens, slices, inter = spec["tokens"], spec["width_slices"], spec["intermediate"]
    assert slices * inter == _ML4_INTER, spec
    args, ref = _ml4_args(spec, tokens, seed)
    st = _ml4_stack_for(spec)
    outs, raws = [moe(**args)[0]], [st["raw"]]
    for rank in range(1, slices):
        other = _ml4_width_slice(inter, rank, reference=False)
        outs.append(moe(**dict(args, **{k: other[k] for k in _ML4_WIDTH_LAYOUT}))[0])
        raws.append(other["raw"])
        del other
    want = _ml4_whole_width_reference(ref, raws, st["ws13"], st["ws2"]).to(torch.bfloat16)
    del raws
    summed = sum(y.float() for y in outs).to(torch.bfloat16)
    running = outs[0]
    for y in outs[1:]:
        running = running + y
    for name, y in (("fp32 sum", summed), ("bf16 running sum", running)):
        elt, rms = _dev(y, want)
        print(f"  {slices} width slices at T={tokens}, {name}: {elt:.2f} elt / {rms:.3f} rms ulp")
        _compare(y, want)  # the whole-width layer: the I 6144 band
    wrong = {
        f"slice {slices - 1} left out": outs[:-1],
        "slice 0 run in place of slice 1": (outs[0], outs[0], *outs[2:]),
    }
    for name, partials in wrong.items():
        y = sum(p.float() for p in partials).to(torch.bfloat16)
        elt, rms = _dev(y, want)
        print(f"    {name}: {elt:.3g} elt / {rms:.2f} rms ulp")
        assert rms > 20.0, f"{name} is only {rms:.2f} rms ulp from the whole-width layer"


# The Mistral Large 4 tp4 call the tests below drive: all 128 experts at the slice.
_ML4_TP4_SPEC = next(
    c.spec for c in moe.CELLS if c.spec.get("intermediate") == _ML4_TP4_INTER and "graph" in c.spec
)


#: (tokens, routing kernel, token tile of both GEMMs, finalize kernel) the
#: Mistral Large 4 tp4 call launches on the cold path, at each side of every
#: switch the tp4 sweep cell's token counts claim to straddle. The tile is
#: `nextPowerOfTwo(int(T * top_k / local_num_experts))` clamped to 8..256,
#: which at 128 local experts is T / 32.
_ML4_TP4_KERNELS = (
    (1, _BLOCK_ROUTING, 8, _SCALAR_FIN),
    (4, _BLOCK_ROUTING, 8, _SCALAR_FIN),
    (5, _DYN_ROUTING, 8, _SCALAR_FIN),
    (12, _DYN_ROUTING, 8, _SCALAR_FIN),
    (13, _CLUSTER_ROUTING, 8, _SCALAR_FIN),
    (49, _CLUSTER_ROUTING, 8, _SCALAR_FIN),
    (50, _CLUSTER_ROUTING, 8, _VECTOR_FIN),
    (287, _CLUSTER_ROUTING, 8, _VECTOR_FIN),
    (288, _CLUSTER_ROUTING, 16, _VECTOR_FIN),
    (543, _CLUSTER_ROUTING, 16, _VECTOR_FIN),
    (544, _CLUSTER_ROUTING, 32, _VECTOR_FIN),
    (1055, _CLUSTER_ROUTING, 32, _VECTOR_FIN),
    (1056, _CLUSTER_ROUTING, 64, _VECTOR_FIN),
    (2079, _CLUSTER_ROUTING, 64, _VECTOR_FIN),
    (2080, _CLUSTER_ROUTING, 128, _VECTOR_FIN),
    (4127, _CLUSTER_ROUTING, 128, _VECTOR_FIN),
    (4128, _CLUSTER_ROUTING, 256, _VECTOR_FIN),
    (8192, _CLUSTER_ROUTING, 256, _VECTOR_FIN),
    (8193, _COOP_ROUTING, 256, _VECTOR_FIN),
    (32768, _COOP_ROUTING, 256, _VECTOR_FIN),
)


def test_sw_tp4_kernels_by_token_count():
    """The switches the Mistral Large 4 tp4 sweep cell straddles are where it says:
    read back from the kernels each call launches, on the cold path."""
    from torch.profiler import ProfilerActivity, profile

    args, ref = _ml4_args(_ML4_TP4_SPEC, 32768, seed=6)
    for tokens, routing, tile, finalize in _ML4_TP4_KERNELS:
        call = _ml4_prefix(args, ref, tokens)[0]
        moe(**call)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            moe(**call)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if str(e.device_type) == "DeviceType.CUDA"]
        gemms = [n for n in names if n.startswith("bmm_")]
        assert len(gemms) == 2 and all(f"_t128x{tile}x" in n for n in gemms), (tokens, gemms)
        for want in (routing, finalize):
            assert any(want in n for n in names), (tokens, want, names)
    print(f"  test_sw_tp4_kernels_by_token_count OK ({len(_ML4_TP4_KERNELS)} token counts)")


_ML4_RUNNER_INPUTS = (
    "routing_logits",
    "routing_bias",
    "hidden_states",
    "hidden_states_scale",
    "gemm1_weights",
    "gemm1_weights_scale",
    "gemm1_bias",
    "gemm1_alpha",
    "gemm1_beta",
    "gemm1_clamp_limit",
    "gemm2_weights",
    "gemm2_weights_scale",
    "gemm2_bias",
    "output1_scale_scalar",
    "output1_scale_gate_scalar",
    "output2_scale_scalar",
    "topk_weights",
    "topk_ids",
)


def test_sw_tp4_every_valid_tactic_passes_the_gate():
    """The tuned path at tp4: each tactic the op's own tunable runner offers at
    I 1536 / 128 local experts, run at each decode count the graphs capture and
    each autotuner bucket above it (`_ML4_TACTIC_TOKENS`). One whose bits equal
    the fallback's is certified by the cells; one whose bits do not is gated
    against `reference` itself."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    spec = _ML4_TP4_SPEC
    band = _ml4_elt_ulp(spec)
    runner = FP4BlockScaleMoERunner(
        spec["num_experts"],
        spec["top_k"],
        None,
        None,
        spec["intermediate"],
        spec["local_expert_offset"],
        spec["local_num_experts"],
        None,
        1,
        True,
        0,
        tune_max_num_tokens=spec["tune_max_num_tokens"],
    )
    summary = []
    for tokens in _ML4_TACTIC_TOKENS:
        args, ref = _ml4_args(spec, tokens, seed=1536 + tokens)
        base = moe(**args)[0]
        inputs = [args[k] for k in _ML4_RUNNER_INPUTS]
        valid = runner.get_runner().get_valid_configs(
            spec["top_k"], spec["hidden"], spec["intermediate"], spec["local_num_experts"], tokens
        )
        assert len({t[0] for t in valid}) > 1, f"T={tokens}: one tile only, {valid}"
        exp = None
        worst = (0.0, 0.0)
        differ = []
        for tactic in valid:
            y = runner.forward(inputs, list(tactic))[0]
            if torch.equal(y, base):
                continue
            differ.append(tuple(tactic))
            if exp is None:
                exp = _ml4_reference(ref)
                _compare(base, exp, band)
            _compare(y, exp, band)
            worst = tuple(max(a, b) for a, b in zip(worst, _dev(y, exp)))
        summary.append((tokens, len(valid), len(differ)))
        if differ:
            print(
                f"  T={tokens}: {len(differ)} of {len(valid)} tactics differ from the fallback, "
                f"all inside the gate (worst {worst[0]:.2f} elt / {worst[1]:.3f} rms): {differ}"
            )
    bitwise = [t for t, _, d in summary if d == 0]
    print(
        "  test_sw_tp4_every_valid_tactic_passes_the_gate OK ("
        + ", ".join(f"T={t}: {n}" for t, n, _ in summary)
        + f" valid tactics; every tactic bitwise the fallback at T in {bitwise})"
    )


def test_sw_tp4_experts_past_2_31_elements_give_the_same_bits_below_it():
    """FC1 codes of local expert e start at e2m1 element e * 2I * H, 18,874,368
    per expert at I 1536: expert 113 straddles 2^31, 114..127 start past it.
    Moved to slot 1, each of those 15 experts' tokens come back bitwise the
    same."""
    spec = _ML4_TP4_SPEC
    far = 15
    args, ref = _ml4_args(spec, 64, seed=113, far=far)
    out = moe(**args)[0]
    _compare(out, _ml4_reference(ref), _ml4_elt_ulp(spec))
    per_expert = args["gemm1_weights"][0].numel() * 2
    first = spec["local_num_experts"] - far
    assert first * per_expert < 2**31 < (first + 1) * per_expert, "expert 113 must straddle 2^31"
    ids, off = args["topk_ids"], spec["local_expert_offset"]
    per_local = _ML4_WIDTH_LAYOUT + (
        "output1_scale_scalar",
        "output1_scale_gate_scalar",
        "output2_scale_scalar",
    )
    for far_e in range(first, spec["local_num_experts"]):
        order = list(range(spec["local_num_experts"]))
        order[1], order[far_e] = far_e, 1
        index = torch.tensor(order, device=DEV)
        near_id, far_id = off + 1, off + far_e
        swapped = torch.where(ids == far_id, near_id, torch.where(ids == near_id, far_id, ids))
        moved = dict(args, topk_ids=swapped.to(torch.int32).contiguous())
        for name in per_local:
            moved[name] = args[name].index_select(0, index).contiguous()
        assert bool((ids == far_id).any()), f"no token reaches expert {far_e}"
        assert torch.equal(moe(**moved)[0], out), f"expert {far_e} differs from itself at slot 1"
        del moved
    torch.cuda.empty_cache()
    print(
        f"  test_sw_tp4_experts_past_2_31_elements_give_the_same_bits_below_it OK ({far} experts)"
    )


@pytest.mark.parametrize("layer", [1, 45])
def test_sw_tp4_wrong_variants_land_outside_the_gate(layer):
    """How far the wrong computations a tp4 loader could make land from the
    result the gate accepts, at 1024 tokens over all 128 experts at the
    1536-wide slice: the tep4 test's readings, and the slice's own -- w2 codes,
    w2 block scales or w1/w3 block scales taken from the neighbouring rank's
    slice of the same experts."""
    spec = dict(_ML4_TP4_SPEC, layer=layer)
    args, ref = _ml4_args(spec, 1024, seed=1536 + layer)
    y = moe(**args)[0]
    exp = _ml4_reference(ref)
    _compare(y, exp, _ml4_elt_ulp(spec))
    st = _ml4_width_stack(_ML4_TP4_INTER)
    scalars = ("output1_scale_scalar", "output1_scale_gate_scalar", "output2_scale_scalar")
    neighbour = _ml4_width_slice(_ML4_TP4_INTER, 1, reference=False)
    wrong_calls = {
        "output1 scalar roles swapped": dict(
            output1_scale_scalar=args["output1_scale_gate_scalar"],
            output1_scale_gate_scalar=args["output1_scale_scalar"],
        ),
        "per-expert scalars of the wrong experts": {
            k: args[k].flip(0).contiguous() for k in scalars
        },
        "w1's weight_scale_2 in output2_scale_scalar": dict(
            output2_scale_scalar=_ml4_scalars(layer, st["ws13"], st["ws13"])[2]
        ),
        "w2 codes of the neighbouring slice": dict(gemm2_weights=neighbour["gemm2_weights"]),
        "w2 block scales of the neighbouring slice": dict(
            gemm2_weights_scale=neighbour["gemm2_weights_scale"]
        ),
        "w1/w3 block scales of the neighbouring slice": dict(
            gemm1_weights_scale=neighbour["gemm1_weights_scale"]
        ),
    }
    dist = {name: _dev(moe(**dict(args, **kw))[0], exp) for name, kw in wrong_calls.items()}
    del neighbour, wrong_calls
    swapped = moe.reference(**dict(ref, up=ref["gate"], gate=ref["up"]))
    dist["gate/up swapped"] = _dev(y, swapped.to(torch.bfloat16))
    unquantized = _ref_moe(
        ref["hidden_states"],
        ref["topk_ids"],
        ref["topk_weights"],
        ref,
        ref["fc2_global_scale"],
        quantize_intermediate=False,
    )
    dist["no FC1-output requantization"] = _dev(y, unquantized.to(torch.bfloat16))
    up_p, gt_p, up_s, gt_s, dn_p, dn_s = st["raw"]
    for name, prep in (("no scale swizzle", "swizzle"), ("no row shuffle", "shuffle")):
        w1, s1 = _prep_fc1(up_p, gt_p, up_s, gt_s, **{prep: False})
        w2, s2 = _prep_fc2(dn_p, dn_s, **{prep: False})
        layout = dict(
            gemm1_weights=w1, gemm1_weights_scale=s1, gemm2_weights=w2, gemm2_weights_scale=s2
        )
        dist[name] = _dev(moe(**dict(args, **layout))[0], exp)
        del w1, s1, w2, s2, layout
    torch.cuda.empty_cache()
    print(f"  layer {layer}: correct {_dev(y, exp)}")
    for name, (elt, rms) in dist.items():
        print(f"    {name:44s} {elt:10.3g} elt / {rms:8.2f} rms ulp")
        assert rms > 2.0, f"{name} is {rms:.2f} rms ulp from the correct result: inside the gate"
        if not (layer == 45 and name == "output1 scalar roles swapped"):
            assert rms > 20.0, f"{name} is only {rms:.2f} rms ulp from the correct result"


# The Mistral Large 4 tp8 call the tests below drive: all 128 experts at the slice.
_ML4_TP8_SPEC = next(
    c.spec for c in moe.CELLS if c.spec.get("intermediate") == _ML4_TP8_INTER and "graph" in c.spec
)


#: (tokens, routing kernel, token tile of both GEMMs, finalize kernel) the
#: Mistral Large 4 tp8 call launches on the cold path, at each side of every
#: switch the tp8 sweep cell's token counts claim to straddle. The tile is
#: `nextPowerOfTwo(int(T * top_k / local_num_experts))` clamped to 8..256,
#: which at 128 local experts is T / 32.
_ML4_TP8_KERNELS = (
    (1, _BLOCK_ROUTING, 8, _SCALAR_FIN),
    (4, _BLOCK_ROUTING, 8, _SCALAR_FIN),
    (5, _DYN_ROUTING, 8, _SCALAR_FIN),
    (12, _DYN_ROUTING, 8, _SCALAR_FIN),
    (13, _CLUSTER_ROUTING, 8, _SCALAR_FIN),
    (49, _CLUSTER_ROUTING, 8, _SCALAR_FIN),
    (50, _CLUSTER_ROUTING, 8, _VECTOR_FIN),
    (287, _CLUSTER_ROUTING, 8, _VECTOR_FIN),
    (288, _CLUSTER_ROUTING, 16, _VECTOR_FIN),
    (543, _CLUSTER_ROUTING, 16, _VECTOR_FIN),
    (544, _CLUSTER_ROUTING, 32, _VECTOR_FIN),
    (1055, _CLUSTER_ROUTING, 32, _VECTOR_FIN),
    (1056, _CLUSTER_ROUTING, 64, _VECTOR_FIN),
    (2079, _CLUSTER_ROUTING, 64, _VECTOR_FIN),
    (2080, _CLUSTER_ROUTING, 128, _VECTOR_FIN),
    (4127, _CLUSTER_ROUTING, 128, _VECTOR_FIN),
    (4128, _CLUSTER_ROUTING, 256, _VECTOR_FIN),
    (8192, _CLUSTER_ROUTING, 256, _VECTOR_FIN),
    (8193, _COOP_ROUTING, 256, _VECTOR_FIN),
    (32768, _COOP_ROUTING, 256, _VECTOR_FIN),
)


def test_sw_tp8_kernels_by_token_count():
    """The switches the Mistral Large 4 tp8 sweep cell straddles are where it says:
    read back from the kernels each call launches, on the cold path."""
    from torch.profiler import ProfilerActivity, profile

    args, ref = _ml4_args(_ML4_TP8_SPEC, 32768, seed=8)
    for tokens, routing, tile, finalize in _ML4_TP8_KERNELS:
        call = _ml4_prefix(args, ref, tokens)[0]
        moe(**call)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            moe(**call)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if str(e.device_type) == "DeviceType.CUDA"]
        gemms = [n for n in names if n.startswith("bmm_")]
        assert len(gemms) == 2 and all(f"_t128x{tile}x" in n for n in gemms), (tokens, gemms)
        for want in (routing, finalize):
            assert any(want in n for n in names), (tokens, want, names)
    print(f"  test_sw_tp8_kernels_by_token_count OK ({len(_ML4_TP8_KERNELS)} token counts)")


def test_sw_tp8_every_valid_tactic_passes_the_gate():
    """The tuned path at tp8: each tactic the op's own tunable runner offers at
    I 768 / 128 local experts, run at each decode count the graphs capture and
    each autotuner bucket above it (`_ML4_TACTIC_TOKENS`). One whose bits equal
    the fallback's is certified by the cells; one whose bits do not is gated
    against `reference` itself."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    spec = _ML4_TP8_SPEC
    band = _ml4_elt_ulp(spec)
    runner = FP4BlockScaleMoERunner(
        spec["num_experts"],
        spec["top_k"],
        None,
        None,
        spec["intermediate"],
        spec["local_expert_offset"],
        spec["local_num_experts"],
        None,
        1,
        True,
        0,
        tune_max_num_tokens=spec["tune_max_num_tokens"],
    )
    summary = []
    for tokens in _ML4_TACTIC_TOKENS:
        args, ref = _ml4_args(spec, tokens, seed=768 + tokens)
        base = moe(**args)[0]
        inputs = [args[k] for k in _ML4_RUNNER_INPUTS]
        valid = runner.get_runner().get_valid_configs(
            spec["top_k"], spec["hidden"], spec["intermediate"], spec["local_num_experts"], tokens
        )
        assert len({t[0] for t in valid}) > 1, f"T={tokens}: one tile only, {valid}"
        exp = None
        worst = (0.0, 0.0)
        differ = []
        for tactic in valid:
            y = runner.forward(inputs, list(tactic))[0]
            if torch.equal(y, base):
                continue
            differ.append(tuple(tactic))
            if exp is None:
                exp = _ml4_reference(ref)
                _compare(base, exp, band)
            _compare(y, exp, band)
            worst = tuple(max(a, b) for a, b in zip(worst, _dev(y, exp)))
        summary.append((tokens, len(valid), len(differ)))
        if differ:
            print(
                f"  T={tokens}: {len(differ)} of {len(valid)} tactics differ from the fallback, "
                f"all inside the gate (worst {worst[0]:.2f} elt / {worst[1]:.3f} rms): {differ}"
            )
    bitwise = [t for t, _, d in summary if d == 0]
    print(
        "  test_sw_tp8_every_valid_tactic_passes_the_gate OK ("
        + ", ".join(f"T={t}: {n}" for t, n, _ in summary)
        + f" valid tactics; every tactic bitwise the fallback at T in {bitwise})"
    )


@pytest.mark.parametrize("layer", [1, 45])
def test_sw_tp8_wrong_variants_land_outside_the_gate(layer):
    """How far the wrong computations a tp8 loader could make land from the
    result the gate accepts, at 1024 tokens over all 128 experts at the
    768-wide slice: the tep4 test's readings, and the slice's own -- w2 codes,
    w2 block scales or w1/w3 block scales taken from the neighbouring rank's
    slice of the same experts."""
    spec = dict(_ML4_TP8_SPEC, layer=layer)
    args, ref = _ml4_args(spec, 1024, seed=768 + layer)
    y = moe(**args)[0]
    exp = _ml4_reference(ref)
    _compare(y, exp, _ml4_elt_ulp(spec))
    st = _ml4_stack_for(spec)
    scalars = ("output1_scale_scalar", "output1_scale_gate_scalar", "output2_scale_scalar")
    neighbour = _ml4_width_slice(spec["intermediate"], 1, reference=False)
    wrong_calls = {
        "output1 scalar roles swapped": dict(
            output1_scale_scalar=args["output1_scale_gate_scalar"],
            output1_scale_gate_scalar=args["output1_scale_scalar"],
        ),
        "per-expert scalars of the wrong experts": {
            k: args[k].flip(0).contiguous() for k in scalars
        },
        "w1's weight_scale_2 in output2_scale_scalar": dict(
            output2_scale_scalar=_ml4_scalars(layer, st["ws13"], st["ws13"])[2]
        ),
        "w2 codes of the neighbouring slice": dict(gemm2_weights=neighbour["gemm2_weights"]),
        "w2 block scales of the neighbouring slice": dict(
            gemm2_weights_scale=neighbour["gemm2_weights_scale"]
        ),
        "w1/w3 block scales of the neighbouring slice": dict(
            gemm1_weights_scale=neighbour["gemm1_weights_scale"]
        ),
    }
    dist = {name: _dev(moe(**dict(args, **kw))[0], exp) for name, kw in wrong_calls.items()}
    del neighbour, wrong_calls
    swapped = moe.reference(**dict(ref, up=ref["gate"], gate=ref["up"]))
    dist["gate/up swapped"] = _dev(y, swapped.to(torch.bfloat16))
    unquantized = _ref_moe(
        ref["hidden_states"],
        ref["topk_ids"],
        ref["topk_weights"],
        ref,
        ref["fc2_global_scale"],
        quantize_intermediate=False,
    )
    dist["no FC1-output requantization"] = _dev(y, unquantized.to(torch.bfloat16))
    up_p, gt_p, up_s, gt_s, dn_p, dn_s = st["raw"]
    for name, prep in (("no scale swizzle", "swizzle"), ("no row shuffle", "shuffle")):
        w1, s1 = _prep_fc1(up_p, gt_p, up_s, gt_s, **{prep: False})
        w2, s2 = _prep_fc2(dn_p, dn_s, **{prep: False})
        layout = dict(
            gemm1_weights=w1, gemm1_weights_scale=s1, gemm2_weights=w2, gemm2_weights_scale=s2
        )
        dist[name] = _dev(moe(**dict(args, **layout))[0], exp)
        del w1, s1, w2, s2, layout
    torch.cuda.empty_cache()
    print(f"  layer {layer}: correct {_dev(y, exp)}")
    for name, (elt, rms) in dist.items():
        print(f"    {name:44s} {elt:10.3g} elt / {rms:8.2f} rms ulp")
        assert rms > 2.0, f"{name} is {rms:.2f} rms ulp from the correct result: inside the gate"
        if not (layer == 45 and name == "output1 scalar roles swapped"):
            assert rms > 20.0, f"{name} is only {rms:.2f} rms ulp from the correct result"


# ── Mistral Large 4 dep8: 16-wide windows over chunks of the gathered tokens ────


def _ml4_dep8_cell(**match):
    """The dep8 cell (`local_num_experts` 16) whose spec holds `match`."""
    return next(
        c
        for c in moe.CELLS
        if c.spec.get("local_num_experts") == _ML4_LOCAL16
        and all(c.spec.get(k) == v for k, v in match.items())
    )


# The Mistral Large 4 dep8 call the tests below drive: window (0, 16), layer 1.
_ML4_DEP8_SPEC = next(
    c.spec
    for c in moe.CELLS
    if c.spec.get("local_num_experts") == _ML4_LOCAL16 and "graph" in c.spec
)
_ML4_DEP8_ONE_WINDOW = _ml4_dep8_cell(skew="one_window").spec
_ML4_DEP8_PADDED = _ml4_dep8_cell(skew="padded_zero_rows").spec

#: (tokens, routing kernel, token tile of both GEMMs, finalize kernel) the
#: Mistral Large 4 dep8 call launches on the cold path, at each side of every
#: switch the sweep cell's token counts claim to straddle. The tile is
#: `nextPowerOfTwo(int(T * top_k / local_num_experts))` clamped to 8..256,
#: so at 16 local experts its edges sit at half tep4's token counts; the
#: routing and finalize kernels switch where they do at tep4.
_ML4_DEP8_KERNELS = (
    (1, _BLOCK_ROUTING, 8, _SCALAR_FIN),
    (4, _BLOCK_ROUTING, 8, _SCALAR_FIN),
    (5, _DYN_ROUTING, 8, _SCALAR_FIN),
    (12, _DYN_ROUTING, 8, _SCALAR_FIN),
    (13, _CLUSTER_ROUTING, 8, _SCALAR_FIN),
    (35, _CLUSTER_ROUTING, 8, _SCALAR_FIN),
    (36, _CLUSTER_ROUTING, 16, _SCALAR_FIN),
    (49, _CLUSTER_ROUTING, 16, _SCALAR_FIN),
    (50, _CLUSTER_ROUTING, 16, _VECTOR_FIN),
    (67, _CLUSTER_ROUTING, 16, _VECTOR_FIN),
    (68, _CLUSTER_ROUTING, 32, _VECTOR_FIN),
    (131, _CLUSTER_ROUTING, 32, _VECTOR_FIN),
    (132, _CLUSTER_ROUTING, 64, _VECTOR_FIN),
    (259, _CLUSTER_ROUTING, 64, _VECTOR_FIN),
    (260, _CLUSTER_ROUTING, 128, _VECTOR_FIN),
    (515, _CLUSTER_ROUTING, 128, _VECTOR_FIN),
    (516, _CLUSTER_ROUTING, 256, _VECTOR_FIN),
    (8192, _CLUSTER_ROUTING, 256, _VECTOR_FIN),
    (8193, _COOP_ROUTING, 256, _VECTOR_FIN),
    (32768, _COOP_ROUTING, 256, _VECTOR_FIN),
)

#: Rows of a bf16 `[rows, 6144]` buffer at which a row's byte offset passes 2^31.
_ML4_ROW_2_31_BYTES = 2**31 // (2 * _ML4_HIDDEN) + 1


def _ml4_permuted_rows(tokens: int, top_k: int, local: int, tile: int) -> int:
    """Rows of the op's per-slot FC2 buffer: one tile per local expert while
    slots last, a tile per further `tile` slots, at least 128 KiB of bf16."""
    slots = tokens * top_k
    filled = min(local, slots)
    least = math.ceil(128 * 1024 / (_ML4_HIDDEN * 2))
    return max((filled + (slots - filled) // tile) * tile, least)


def test_sw_dep8_kernels_by_token_count():
    """The switches the Mistral Large 4 dep8 sweep cell straddles are where it
    says, read back from the kernels each call launches on the cold path; a
    whole 32768-token chunk with every slot in window (0, 16), and the padding
    chunk with 131068 slots on experts 0..3, launch the same kernels. The
    per-slot FC2 buffer (`do_finalize=False`) has the rows `_ml4_permuted_rows`
    gives at each count, 134,912 at 32768 tokens, and every slot's row in it
    stays below 174,763, where a bf16 row's byte offset passes 2^31."""
    from torch.profiler import ProfilerActivity, profile

    args, ref = _ml4_args(_ML4_DEP8_SPEC, 32768, seed=5)
    calls = [
        (f"T={tokens}", tokens, routing, tile, finalize, _ml4_prefix(args, ref, tokens)[0])
        for tokens, routing, tile, finalize in _ML4_DEP8_KERNELS
    ]
    del ref
    for spec in (_ML4_DEP8_ONE_WINDOW, _ML4_DEP8_PADDED):
        call = _ml4_args(spec, 32768, seed=5)[0]
        calls.append((spec["skew"], 32768, _COOP_ROUTING, 256, _VECTOR_FIN, call))
    top_rows = {}
    for name, tokens, routing, tile, finalize, call in calls:
        moe(**call)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            moe(**call)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if str(e.device_type) == "DeviceType.CUDA"]
        gemms = [n for n in names if n.startswith("bmm_")]
        assert len(gemms) == 2 and all(f"_t128x{tile}x" in n for n in gemms), (name, gemms)
        for want in (routing, finalize):
            assert any(want in n for n in names), (name, want, names)
        rows, _, slot_rows = moe(**dict(call, do_finalize=False))
        want_rows = _ml4_permuted_rows(tokens, 4, _ML4_LOCAL16, tile)
        assert rows.shape[0] == want_rows, (name, rows.shape[0], want_rows)
        top_rows[name] = int(slot_rows.max())
        assert top_rows[name] < rows.shape[0] < _ML4_ROW_2_31_BYTES, (name, top_rows[name])
    assert _ml4_permuted_rows(32768, 4, _ML4_LOCAL16, 256) == 134912
    print(
        f"  test_sw_dep8_kernels_by_token_count OK ({len(calls)} calls; slot rows up to "
        + ", ".join(
            f"{n}: {r}" for n, r in top_rows.items() if not n.startswith("T=") or "32768" in n
        )
        + ")"
    )


#: The gathered token counts whose every valid tactic is driven at dep8: each
#: decode graph's 8 x 1..16, with every tactic valid at the autotuner bucket the
#: count maps to (a warm engine runs that bucket's tactic there), and each bucket
#: above them (256 to 8192, then `tune_max_num_tokens` 32768, which carries
#: 8193..32768).
_ML4_DEP8_TACTIC_TOKENS = (*range(8, 129, 8), 256, 512, 1024, 2048, 4096, 8192, 32768)


def test_sw_dep8_every_valid_tactic_passes_the_gate():
    """The tuned path at dep8. The engine profiles this op at warm-up and keeps
    one tactic per token bucket, and runs it at every count the bucket takes:
    a decode graph's 8 x b gathered tokens map to the power of two at or below
    them. So every tactic valid at that bucket is run at each decode count,
    at each bucket above, and on the one-window and padding chunks at 32768:
    one whose bits equal the fallback's is certified by the cells, and one
    whose bits do not is gated against `reference` itself."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    spec = _ML4_DEP8_SPEC
    calls = [(f"T={m}", *_ml4_args(spec, m, seed=300 + m)) for m in _ML4_DEP8_TACTIC_TOKENS]
    for skewed in (_ML4_DEP8_ONE_WINDOW, _ML4_DEP8_PADDED):
        calls.append((f"{skewed['skew']} T=32768", *_ml4_args(skewed, 32768, seed=300)))
    runner = FP4BlockScaleMoERunner(
        spec["num_experts"],
        spec["top_k"],
        None,
        None,
        spec["intermediate"],
        spec["local_expert_offset"],
        spec["local_num_experts"],
        None,
        1,
        True,
        0,
        tune_max_num_tokens=spec["tune_max_num_tokens"],
    )
    bucket_of = runner.tuning_config.dynamic_tensor_specs[0].map_to_tuning_buckets
    names = ("routing_logits", "routing_bias", "hidden_states", "hidden_states_scale")
    names += ("gemm1_weights", "gemm1_weights_scale", "gemm1_bias", "gemm1_alpha", "gemm1_beta")
    names += ("gemm1_clamp_limit", "gemm2_weights", "gemm2_weights_scale", "gemm2_bias")
    names += ("output1_scale_scalar", "output1_scale_gate_scalar", "output2_scale_scalar")
    names += ("topk_weights", "topk_ids")
    summary = []
    for name, args, ref in calls:
        tokens = args["hidden_states"].shape[0]
        bucket = bucket_of(tokens)
        base = moe(**args)[0]
        inputs = [args[k] for k in names]
        valid = runner.get_runner().get_valid_configs(
            spec["top_k"], spec["hidden"], spec["intermediate"], spec["local_num_experts"], bucket
        )
        assert len({t[0] for t in valid}) > 1, f"{name}: one tile only, {valid}"
        exp = None
        worst = (0.0, 0.0)
        differ = []
        for tactic in valid:
            y = runner.forward(inputs, list(tactic))[0]
            if torch.equal(y, base):
                continue
            differ.append(tuple(tactic))
            if exp is None:
                exp = _ml4_reference(ref)
                _compare(base, exp)
            _compare(y, exp)
            worst = tuple(max(a, b) for a, b in zip(worst, _dev(y, exp)))
        summary.append(f"{name} (bucket {bucket}): {len(valid)} valid, {len(differ)} differ")
        if differ:
            print(
                f"  {name}: {len(differ)} of {len(valid)} bucket-{bucket} tactics differ from the "
                f"fallback, all inside the gate (worst {worst[0]:.2f} elt / {worst[1]:.3f} rms): "
                f"{differ}"
            )
    print("  test_sw_dep8_every_valid_tactic_passes_the_gate OK (" + "; ".join(summary) + ")")


class _SwTiled:
    """`stack[e % n]` for every global expert `e`: the 128-expert layer whose
    16-wide windows each hold the same `n`-expert stack, indexed the way
    `reference` indexes its `(codes, scales)` operands."""

    def __init__(self, stack: torch.Tensor):
        self.stack = stack

    def __getitem__(self, e: int) -> torch.Tensor:
        return self.stack[e % self.stack.shape[0]]


def _ml4_windows(spec, seed: int) -> None:
    """The eight tep8 or dep8 ranks' calls over one step's tokens (at dep8 a
    chunk of the gathered tokens), each at its own window of the same
    16-expert stack, answer every slot exactly once: their partials, summed in fp32 and as a bf16 running sum in rank
    order, pass the gate against the entry's `reference` over all 128 experts
    at once (expert `e` holding stack expert `e % 16`, as every window does).
    Leaving one window out, or running window (0, 16) in place of (16, 16),
    lands far outside it."""
    tokens, windows, local = spec["tokens"], spec["windows"], spec["local_num_experts"]
    num_experts = spec["num_experts"]
    assert windows * local == num_experts and spec["local_expert_offset"] == 0, spec
    args, ref = _ml4_args(spec, tokens, seed)
    offsets = tuple(range(0, num_experts, local))
    ids = args["topk_ids"]
    slots = [int(((ids >= o) & (ids < o + local)).sum()) for o in offsets]
    assert sum(slots) == ids.numel() and min(slots) > 0, slots
    outs = [moe(**dict(args, local_expert_offset=o))[0] for o in offsets]
    for o, y in zip(offsets, outs):
        _ml4_assert_window(y, dict(args, local_expert_offset=o))
    layer = dict(
        ref,
        up=tuple(_SwTiled(t) for t in ref["up"]),
        gate=tuple(_SwTiled(t) for t in ref["gate"]),
        down=tuple(_SwTiled(t) for t in ref["down"]),
        local_expert_offset=0,
        local_num_experts=num_experts,
    )
    entry_ref = moe.reference(**layer)
    local_ref = _ref_moe(
        layer["hidden_states"],
        layer["topk_ids"],
        layer["topk_weights"],
        layer,
        layer["fc2_global_scale"],
        offset=0,
        num_local=num_experts,
    )
    torch.testing.assert_close(entry_ref, local_ref, rtol=0.0, atol=0.0)
    del local_ref
    want = entry_ref.to(torch.bfloat16)
    del entry_ref
    summed = sum(y.float() for y in outs).to(torch.bfloat16)
    running = outs[0]
    for y in outs[1:]:
        running = running + y
    for name, y in (("fp32 sum", summed), ("bf16 running sum", running)):
        elt, rms = _dev(y, want)
        print(f"  {windows} windows, {name}: {elt:.2f} elt / {rms:.3f} rms ulp")
        _compare(y, want)
    wrong = {
        f"window ({offsets[-1]}, {local}) left out": outs[:-1],
        f"window (0, {local}) in place of ({local}, {local})": (outs[0], outs[0], *outs[2:]),
    }
    for name, partials in wrong.items():
        y = sum(p.float() for p in partials)
        elt, rms = _dev(y.to(torch.bfloat16), want)
        print(f"    {name}: {elt:.3g} elt / {rms:.2f} rms ulp")
        assert rms > 20.0, f"{name} is only {rms:.2f} rms ulp from the layer"


# ── Mistral Large 4 tep8: 16-wide windows, every rank over the step's tokens ───


def _ml4_tep8_cell(**match):
    """The tep8 cell (`local_num_experts` 16) whose spec holds `match`."""
    return next(
        c
        for c in moe.CELLS
        if c.spec["local_num_experts"] == _ML4_LOCAL16
        and all(c.spec.get(k) == v for k, v in match.items())
    )


# The Mistral Large 4 tep8 call the tests below drive, at window (0, 16).
_ML4_TEP8_SPEC = next(
    c.spec
    for c in moe.CELLS
    if c.spec["local_num_experts"] == _ML4_LOCAL16
    and c.spec.get("layer") == 1
    and c.spec.get("graph") == tuple(range(1, 17))
)
_ML4_TEP8_ONE_WINDOW = _ml4_tep8_cell(skew="one_window").spec

#: (tokens, routing kernel, token tile of both GEMMs, finalize kernel) the
#: Mistral Large 4 tep8 call launches on the cold path, at each side of every
#: switch the sweep cell's token counts claim to straddle. The tile is
#: `nextPowerOfTwo(int(T * top_k / local_num_experts))` clamped to 8..256,
#: so at 16 local experts its edges sit at half tep4's token counts; the
#: routing and finalize kernels switch where they do at tep4.
_ML4_TEP8_KERNELS = (
    (1, _BLOCK_ROUTING, 8, _SCALAR_FIN),
    (4, _BLOCK_ROUTING, 8, _SCALAR_FIN),
    (5, _DYN_ROUTING, 8, _SCALAR_FIN),
    (12, _DYN_ROUTING, 8, _SCALAR_FIN),
    (13, _CLUSTER_ROUTING, 8, _SCALAR_FIN),
    (35, _CLUSTER_ROUTING, 8, _SCALAR_FIN),
    (36, _CLUSTER_ROUTING, 16, _SCALAR_FIN),
    (49, _CLUSTER_ROUTING, 16, _SCALAR_FIN),
    (50, _CLUSTER_ROUTING, 16, _VECTOR_FIN),
    (67, _CLUSTER_ROUTING, 16, _VECTOR_FIN),
    (68, _CLUSTER_ROUTING, 32, _VECTOR_FIN),
    (131, _CLUSTER_ROUTING, 32, _VECTOR_FIN),
    (132, _CLUSTER_ROUTING, 64, _VECTOR_FIN),
    (259, _CLUSTER_ROUTING, 64, _VECTOR_FIN),
    (260, _CLUSTER_ROUTING, 128, _VECTOR_FIN),
    (515, _CLUSTER_ROUTING, 128, _VECTOR_FIN),
    (516, _CLUSTER_ROUTING, 256, _VECTOR_FIN),
    (8192, _CLUSTER_ROUTING, 256, _VECTOR_FIN),
    (8193, _COOP_ROUTING, 256, _VECTOR_FIN),
    (32768, _COOP_ROUTING, 256, _VECTOR_FIN),
)


def test_sw_tep8_kernels_by_token_count():
    """The switches the Mistral Large 4 tep8 sweep cell straddles are where it
    says, read back from the kernels each call launches on the cold path; at
    32768 tokens with every slot in window (0, 16) the call launches the same
    kernels. The per-slot FC2 buffer (`do_finalize=False`) has the rows
    `_ml4_permuted_rows` gives at each count: 134,912 at 32768 tokens, below the
    174,763 rows where a bf16 row's byte offset passes 2^31."""
    from torch.profiler import ProfilerActivity, profile

    args, ref = _ml4_args(_ML4_TEP8_SPEC, 32768, seed=5)
    calls = [
        (tokens, routing, tile, finalize, _ml4_prefix(args, ref, tokens)[0])
        for tokens, routing, tile, finalize in _ML4_TEP8_KERNELS
    ]
    del ref
    one_window = _ml4_args(_ML4_TEP8_ONE_WINDOW, 32768, seed=5)[0]
    calls.append((32768, _COOP_ROUTING, 256, _VECTOR_FIN, one_window))
    for tokens, routing, tile, finalize, call in calls:
        moe(**call)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            moe(**call)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if str(e.device_type) == "DeviceType.CUDA"]
        gemms = [n for n in names if n.startswith("bmm_")]
        assert len(gemms) == 2 and all(f"_t128x{tile}x" in n for n in gemms), (tokens, gemms)
        for want in (routing, finalize):
            assert any(want in n for n in names), (tokens, want, names)
        rows = moe(**dict(call, do_finalize=False))[0].shape[0]
        assert rows == _ml4_permuted_rows(tokens, 4, _ML4_LOCAL16, tile), (tokens, rows)
    assert _ml4_permuted_rows(32768, 4, _ML4_LOCAL16, 256) == 134912 < 2**31 // (2 * _ML4_HIDDEN)
    print(f"  test_sw_tep8_kernels_by_token_count OK ({len(calls)} calls)")


#: The token counts whose every valid tactic is driven at tep8: each decode
#: count the graphs capture, and each autotuner bucket above it (powers of two
#: to 8192, then `tune_max_num_tokens` 32768, which carries 8193..32768).
_ML4_TEP8_TACTIC_TOKENS = (*range(1, 17), 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 32768)


def test_sw_tep8_every_valid_tactic_passes_the_gate():
    """The tuned path at tep8. The engine profiles this op at warm-up and keeps
    one tactic per token bucket; every call in the cells runs the cold
    fallback. So each valid tactic the op's tunable runner offers at 16 local
    experts is run at every decode count and every bucket, and at 32768 tokens
    with every slot in one window too: one whose bits equal the fallback's is
    certified by the cells, and one whose bits do not is gated against
    `reference` itself."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    spec = _ML4_TEP8_SPEC
    calls = [(f"T={m}", *_ml4_args(spec, m, seed=300 + m)) for m in _ML4_TEP8_TACTIC_TOKENS]
    calls.append(("one window T=32768", *_ml4_args(_ML4_TEP8_ONE_WINDOW, 32768, seed=300)))
    runner = FP4BlockScaleMoERunner(
        spec["num_experts"],
        spec["top_k"],
        None,
        None,
        spec["intermediate"],
        spec["local_expert_offset"],
        spec["local_num_experts"],
        None,
        1,
        True,
        0,
        tune_max_num_tokens=spec["tune_max_num_tokens"],
    )
    names = ("routing_logits", "routing_bias", "hidden_states", "hidden_states_scale")
    names += ("gemm1_weights", "gemm1_weights_scale", "gemm1_bias", "gemm1_alpha", "gemm1_beta")
    names += ("gemm1_clamp_limit", "gemm2_weights", "gemm2_weights_scale", "gemm2_bias")
    names += ("output1_scale_scalar", "output1_scale_gate_scalar", "output2_scale_scalar")
    names += ("topk_weights", "topk_ids")
    summary = []
    for name, args, ref in calls:
        tokens = args["hidden_states"].shape[0]
        base = moe(**args)[0]
        inputs = [args[k] for k in names]
        valid = runner.get_runner().get_valid_configs(
            spec["top_k"], spec["hidden"], spec["intermediate"], spec["local_num_experts"], tokens
        )
        assert len({t[0] for t in valid}) > 1, f"{name}: one tile only, {valid}"
        exp = None
        worst = (0.0, 0.0)
        differ = []
        for tactic in valid:
            y = runner.forward(inputs, list(tactic))[0]
            if torch.equal(y, base):
                continue
            differ.append(tuple(tactic))
            if exp is None:
                exp = _ml4_reference(ref)
                _compare(base, exp)
            _compare(y, exp)
            worst = tuple(max(a, b) for a, b in zip(worst, _dev(y, exp)))
        summary.append(f"{name}: {len(valid)} valid, {len(differ)} differ")
        if differ:
            print(
                f"  {name}: {len(differ)} of {len(valid)} tactics differ from the fallback, all "
                f"inside the gate (worst {worst[0]:.2f} elt / {worst[1]:.3f} rms): {differ}"
            )
    print("  test_sw_tep8_every_valid_tactic_passes_the_gate OK (" + "; ".join(summary) + ")")


# The shared layer each cell's (num_experts, hidden, intermediate) names.
_GEOMETRIES = {(256, 7168, 2048): _dsr1, (512, 2560, 640): _many_experts}


@pytest.mark.parametrize("cell", moe.CELLS, ids=[c.why[:44] for c in moe.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    """Drive every configuration the entry claims, through the entry's own
    reference and its own gate.

    The cases above test the op; this one tests the *claim*. A cell list that
    nothing executes reads as coverage and is not.

    Each cell cross-checks `moe.reference` against this file's `_ref_moe`
    before either is used to gate the kernel. Two references disagreeing is a
    different failure from the kernel moving, and collapsing them into one
    assert loses which happened.
    """
    certifying(moe)
    spec = cell.spec
    if "layer" in spec:  # Mistral Large 4 tep4, dep4, tp4 and dep8: the checkpoint's magnitudes
        seed = cell.seed_base() % 2**31
        if "graph" in spec:
            for tokens in spec["graph"]:
                _ml4_replay(spec, tokens, seed + tokens)
        elif "width_slices" in spec:
            _ml4_width_slices(spec, seed)
        elif "windows" in spec:
            _ml4_windows(spec, seed)
        else:
            _ml4_sweep(spec, seed)
        return
    args, ref = _GEOMETRIES[(spec["num_experts"], spec["hidden"], spec["intermediate"])]()
    off, n = spec["local_expert_offset"], spec["local_num_experts"]
    num_experts, hidden, top_k = spec["num_experts"], spec["hidden"], spec["top_k"]
    gen = torch.Generator(device=DEV).manual_seed(cell.seed_base() % 2**31)
    g1 = 137.0

    data, sf, x = _rand_activation(spec["tokens"], hidden, g1, gen)
    routing = _renorm_routing if spec.get("routing") == "renormalized" else _routing
    ids, wts = routing(spec["tokens"], num_experts, top_k, gen)
    g2 = _calibrate_g2(x, ids, ref)
    s1, sg, s2 = _scalars(num_experts, g1, g2)

    win_args = args if n == num_experts else _window_args(args, off, n)
    win_ref = ref if n == num_experts else _window_ref(ref, off, n)

    # 1. The entry's reference against this file's, on identical inputs.
    entry_ref = moe.reference(
        x,
        ids,
        wts,
        win_ref["up"],
        win_ref["gate"],
        win_ref["down"],
        g2,
        local_expert_offset=off,
        local_num_experts=n,
    )
    local_ref = _ref_moe(x, ids, wts, win_ref, g2, offset=off, num_local=n)
    torch.testing.assert_close(entry_ref, local_ref, rtol=0.0, atol=0.0)

    # 2. The entry's reference against the kernel, through the entry's gate.
    y = _call(
        data,
        sf,
        win_args,
        num_experts,
        top_k,
        local_expert_offset=off,
        local_num_experts=n,
        topk_ids=ids,
        topk_weights=wts,
        **_window_scalars(s1, sg, s2, off, n),
    )[0]
    assert y.shape == (spec["tokens"], hidden), y.shape
    _compare(y, entry_ref.to(torch.bfloat16))
