# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 MoE test operands and reference arithmetic shared by operation tests.

Importing this module allocates no device tensors and changes no precision
settings. Large expert operands are allocated lazily by a module-scoped fixture.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

import pytest
import torch

__extra_import_path__ = ["."]
from _precision import true_fp32_matmul  # noqa: E402
from _validating import validating  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    fp4_block_scale_moe_runner as moe,
)

DEV = "cuda"
ULP = 2.0**-8
SV = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
_E2M1_VALUES = (
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
)
_Layer = tuple[dict, dict]


def _e2m1_values(device: torch.device) -> torch.Tensor:
    return torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=device)


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """16 ulp per element against the row's own scale, 2 ulp relative RMS.

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
    """
    assert out.dtype == ref.dtype == torch.bfloat16, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    ulp = 2.0**-8
    row = ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    torch.testing.assert_close(out.float() / row, ref.float() / row, rtol=0.0, atol=16 * ulp)
    rel_rms = (
        (out.float() - ref.float()).pow(2).mean().sqrt()
        / ref.float().pow(2).mean().sqrt().clamp_min(1e-9)
    ).item() / ulp
    assert rel_rms <= 2.0, f"relative RMS {rel_rms:.2f} ulp > 2 ulp"


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


def _rand_nvfp4(
    e: int, n: int, k: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return _e2m1_values(codes_e.device)[codes_e.long()] * sf_e.float().repeat_interleave(SV, dim=1)


def _unpack_activation(data: torch.Tensor, sf: torch.Tensor, hidden: int, g: float) -> torch.Tensor:
    """Exact fp32 value of an NVFP4 activation pair, undoing the global scale.

    `data` is `[T, hidden/2]` uint8 (element 2i in the low nibble), `sf` a 1-D
    linear `[T * hidden/16]` e4m3 buffer.
    """
    rows = data.shape[0]
    codes = torch.stack([data & 0x0F, data >> 4], dim=-1).reshape(rows, hidden).long()
    scale = sf.view(torch.float8_e4m3fn).reshape(rows, hidden // SV).float()
    return _e2m1_values(data.device)[codes] * scale.repeat_interleave(SV, dim=1) / g


def _rand_activation(
    rows: int, hidden: int, g: float, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    x = _e2m1_values(codes.device)[codes.long()] * sf.float().repeat_interleave(SV, dim=1) / g
    return data, sf_flat, x


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


def _prep_fc1(
    up_p, gt_p, up_s, gt_s, swizzle=True, shuffle=True
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _prep_fc2(dn_p, dn_s, swizzle=True, shuffle=True) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-shuffle the down projection's rows, swizzle its scales."""
    w, s = dn_p, dn_s
    if shuffle:
        perm = _blk32_perm(w.shape[1])
        w = torch.index_select(w, 1, perm)
        s = torch.index_select(s, 1, perm)
    s = s.view(torch.uint8)
    s = _swizzle_scales(s) if swizzle else s.contiguous()
    return w.contiguous(), s.view(torch.float8_e4m3fn).contiguous()


def _build(num_experts: int, hidden: int, inter: int, seed: int) -> tuple[dict, dict]:
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


def _scalars(
    num_local: int, g1: float, g2: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three per-expert fp32 scalars, for weight global scales of 1.

    `output1_scale_gate_scalar = alpha1` dequantizes the FC1 accumulator;
    `output1_scale_scalar = g2 * alpha1` additionally applies the FC2-input
    global scale to the linear half; `output2_scale_scalar = alpha2`
    dequantizes the FC2 accumulator.
    """
    a1 = torch.full((num_local,), 1.0 / g1, dtype=torch.float32, device=DEV)
    a2 = torch.full((num_local,), 1.0 / g2, dtype=torch.float32, device=DEV)
    return a1 * g2, a1, a2


def _call(data, sf, args, num_experts, top_k, scale_as_is=False, **kw) -> tuple[torch.Tensor, ...]:
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


@true_fp32_matmul()
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
) -> torch.Tensor:
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


@true_fp32_matmul()
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


def _dev(y: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
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
    anywhere in the shared FP4 test suite.
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


def _routing(num_tokens, num_experts, top_k, gen) -> tuple[torch.Tensor, torch.Tensor]:
    """Random ids/weights for the pre-routed entry point."""
    ids = torch.stack(
        [torch.randperm(num_experts, device=DEV, generator=gen)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    wts = torch.rand(num_tokens, top_k, device=DEV, generator=gen).to(torch.bfloat16)
    return ids, wts


def _renorm_routing(num_tokens, num_experts, top_k, gen) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-routed ids/weights in the form a renormalising router emits.

    Softmax over every expert, top-k of that, and the k weights rescaled to sum
    to 1 -- the `norm_topk_prob` convention. `torch.topk` returns distinct
    indices, which is what this op requires of a row.
    """
    logits = torch.randn(num_tokens, num_experts, device=DEV, dtype=torch.float32, generator=gen)
    w, idx = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    return idx.to(torch.int32).contiguous(), w.to(torch.bfloat16).contiguous()


@pytest.fixture(scope="module")
def many_expert_layer() -> Iterator[Callable[[], _Layer]]:
    """Lazily reuse the 512-expert operand stack within one test module.

    Kernel-ready operands occupy about 1.4 GiB; reference operands bring the
    stack to about 5 GiB. The fixture releases its reference at module teardown.
    """
    layer = None

    def get_layer() -> _Layer:
        nonlocal layer
        if layer is None:
            layer = _build(512, 2560, 640, seed=512640)
        return layer

    try:
        yield get_layer
    finally:
        layer = None


def renorm_operands(
    layer: _Layer,
    spec: Mapping,
    seed: int,
    router: Callable,
) -> tuple:
    """Build activation/routing operands for a renormalized NVFP4 MoE call."""
    args, ref = layer
    gen = torch.Generator(device=DEV).manual_seed(seed)
    global_scale = 137.0
    data, sf, x = _rand_activation(spec["tokens"], spec["hidden"], global_scale, gen)
    shape = (spec["tokens"], spec["num_experts"])
    if spec.get("dist") == "coarse":
        logits = torch.randint(-1, 2, shape, generator=gen, device=DEV).float()
    else:
        logits = torch.randn(*shape, generator=gen, device=DEV)
    logits = logits.to(spec["logits_dtype"]).contiguous()
    ids, weights = router(logits, spec["top_k"], torch.bfloat16)
    fc2_scale = _calibrate_g2(x, ids, ref)
    scalars = _scalars(spec["num_experts"], global_scale, fc2_scale)
    return args, ref, data, sf, x, logits, ids, weights, fc2_scale, scalars
