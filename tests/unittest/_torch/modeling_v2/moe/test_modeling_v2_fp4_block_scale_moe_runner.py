# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp4_block_scale_moe_runner catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
    ACT_TYPE_RELU2,
    ACT_TYPE_SWIGLU,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
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

    Relu2 adds no second band. Its FC1 epilogue is likewise one requantization,
    read out bit for bit at every token tile (8 through 256) by
    `test_relu2_intermediate_is_nvfp4_requantized`, and Nemotron-3 Ultra's
    geometry (512 experts, top-22, H 2048 / I 5120, 128-wide windows,
    per-expert weight_scale_2) lands at 5.14 / 0.74 per window over its whole
    column and 2.77 / 0.84 for the four windows summed. Its wrong variants sit
    19 (no FC1-output requantization) to 9561 RMS ulp out.
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


def _proj(x, operand, e: int) -> torch.Tensor:
    """`x @ W_e^T` for expert `e` of a `(codes, scales[, weight_scale_2])`
    operand; a per-expert `weight_scale_2` multiplies the fp32 product."""
    y = x @ _dequant(operand[0][e], operand[1][e]).t()
    return y * operand[2][e].float() if len(operand) > 2 else y


def _fc1_act(xe, up, gate, e: int, act_type: int) -> torch.Tensor:
    """Expert `e`'s true FC1 activation for rows `xe`, fp32, before any
    requantization: `up * silu(gate)` for SwiGlu, `relu(up) ** 2` for Relu2."""
    u = _proj(xe, up, e)
    if act_type == ACT_TYPE_RELU2:
        r = torch.relu(u)
        return r * r
    g = _proj(xe, gate, e)
    return u * g * torch.sigmoid(g)


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
    `ref["act_type"]` (SwiGlu when absent) picks the activation; the two swaps
    are SwiGlu's wrong variants.
    """
    num_tokens, hidden = x.shape
    act_type = ref.get("act_type", ACT_TYPE_SWIGLU)
    up, gate, down = ref["up"], ref["gate"], ref["down"]
    if swap_gate_up:
        up, gate = gate, up
    num_local = up[0].shape[0] if num_local is None else num_local
    out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)
    for local_e in range(num_local):
        tok, slot = (ids == offset + local_e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        xe = x[tok]
        if swap_scalars:
            u = _proj(xe, up, local_e) / g2
            g = _proj(xe, gate, local_e) * g2
            act = u * g * torch.sigmoid(g)
        else:
            act = _fc1_act(xe, up, gate, local_e, act_type)
        # the FC1 epilogue emits g2 * act as NVFP4; FC2's alpha divides it back
        act = _q_nvfp4(act, g2) / g2 if quantize_intermediate else act
        y = _proj(act, down, local_e)
        out.index_add_(0, tok, y * scales[tok, slot].float().unsqueeze(1))
    return out


def _calibrate_g2(x, ids, ref) -> float:
    """`448*6 / amax` of the true FC1 activation — the FC2 input global scale a
    checkpoint's `down_proj.input_scale` encodes."""
    act_type = ref.get("act_type", ACT_TYPE_SWIGLU)
    amax = 0.0
    for e in range(ref["up"][0].shape[0]):
        tok = (ids == e).nonzero(as_tuple=True)[0].unique()
        if tok.numel() == 0:
            continue
        act = _fc1_act(x[tok], ref["up"], ref["gate"], e, act_type)
        amax = max(amax, act.abs().max().item())
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
# static-block kernel takes 4 rows and below, the dynamic-block one 5 to 12
# rows at expert tiers up to 1024 -- 512 here -- and the single-cluster one
# everything from 13 rows up to 8192. Measured by kernel name at this geometry.
# So the column brackets the 4/5 boundary as well as the decode and prefill
# ends.
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


# ── Relu2 (element-wise, non-gated) layout and scalars ────────────────────
#
# Self-contained on purpose: a target copies these four into its own
# weights.py (targets share nothing but the catalog), so they take the
# checkpoint's tensors as stored and depend on nothing else in this file.
# `test_relu2_layout_matches_the_engine_loader` pins the layout against
# trtllm's own MoE weight preparation, `test_relu2_intermediate_is_nvfp4_
# requantized` pins the scalars' roles bit for bit, and the Relu2 cells and
# `test_ultra_relu2_geometry` drive both end to end.


def blk32_row_perm(rows: int, device=None) -> torch.Tensor:
    """Gather index of trtllm-gen's 32-row block shuffle (epilogue tile M 128).

    Within each block of 32 rows, source row `4u + v` lands at destination row
    `8v + u`: `shuffled = w.index_select(-2, blk32_row_perm(w.shape[-2]))`.
    """
    assert rows % 32 == 0, f"{rows} rows is not a whole number of 32-row blocks"
    j = torch.arange(32, device=device)
    src = torch.empty_like(j)
    src[(j % 4) * 8 + j // 4] = j
    block_base = torch.arange(rows // 32, device=device).repeat_interleave(32) * 32
    return src.repeat(rows // 32) + block_base


def swizzle_128x4(sf: torch.Tensor) -> torch.Tensor:
    """trtllm-gen's 128x4 block-scale swizzle per expert, viewed back as `[E, M, C]`.

    Byte `(e, m, c)` lands at flat offset `e*M*C + (m // 128)*512*(C // 4) +
    (c // 4)*512 + (m % 32)*16 + ((m % 128) // 32)*4 + c % 4`, which is what
    `torch.ops.trtllm.block_scale_interleave` writes when `M % 128 == 0` and
    `C % 4 == 0` -- the case where it pads nothing.
    """
    e, m, c = sf.shape
    assert m % 128 == 0 and c % 4 == 0, f"[{m}, {c}] block scales would need padding"
    v = sf.reshape(e, m // 128, 4, 32, c // 4, 4).permute(0, 1, 4, 3, 2, 5)
    return v.reshape(e, m, c).contiguous()


def nvfp4_moe_single_layout(packed: torch.Tensor, block_scales: torch.Tensor):
    """An NVFP4 expert stack the kernel reads as one matrix, in kernel form.

    That is Relu2's FC1 (`up_proj` stacked to `[E, I, H/2]` -- element-wise,
    so there is no gate half to interleave) and every FC2 (`down_proj`,
    `[E, H, I/2]`). Takes the checkpoint's tensors stacked over the window's
    experts: `packed` uint8 `[E, M, K/2]` (element 2i in the low nibble) and
    `block_scales` float8_e4m3fn `[E, M, K/16]`, linear. Returns `(uint8
    [E, M, K/2], float8_e4m3fn [E, M, K/16])`: rows block-shuffled in both,
    the scales then 128x4-swizzled. Requires `M % 128 == 0` and `K % 64 == 0`.
    """
    perm = blk32_row_perm(packed.shape[1], packed.device)
    weights = packed.index_select(1, perm).contiguous()
    scales = block_scales.view(torch.uint8).index_select(1, perm)
    return weights, swizzle_128x4(scales).view(torch.float8_e4m3fn)


def relu2_moe_scalars(up_input_scale, up_weight_scale_2, down_input_scale, down_weight_scale_2):
    """The op's scalars for a Relu2 layer, from a ModelOpt NVFP4 checkpoint.

    `*_input_scale` are the layer's shared activation scales (0-d or `[1]`,
    `amax / (6 * 448)`), `*_weight_scale_2` the experts' fp32 `[E]` global
    weight scales. Returns fp32 `(activation global scale [1] for
    fp4_quantize, output1_scale_scalar [E], output1_scale_gate_scalar [E],
    output2_scale_scalar [E])`; a rank slices the three `[E]` vectors to its
    window. The FC1 epilogue applies `output1_scale_gate_scalar` before the
    relu2 and `output1_scale_scalar` after it, so for Relu2 the latter is the
    FC2-input quantization scale alone -- unlike SwiGlu's, which also carries
    the dequantization.
    """
    up_is = up_input_scale.float().reshape(())
    down_is = down_input_scale.float().reshape(())
    num_experts = up_weight_scale_2.numel()
    act_global_scale = (1.0 / up_is).reshape(1)
    output1_scale_scalar = (1.0 / down_is).expand(num_experts).contiguous()
    output1_scale_gate_scalar = (up_is * up_weight_scale_2.float().reshape(-1)).contiguous()
    output2_scale_scalar = (down_is * down_weight_scale_2.float().reshape(-1)).contiguous()
    return act_global_scale, output1_scale_scalar, output1_scale_gate_scalar, output2_scale_scalar


def _rand_weight_scale_2(num_experts: int, gen: torch.Generator) -> torch.Tensor:
    """Per-expert fp32 global weight scales, log-uniform over a factor of 4 and
    never a power of two, so a scalar applied to the wrong expert shows."""
    return torch.exp2(torch.rand(num_experts, device=DEV, generator=gen) * 2.0 - 1.0) / 64.0


def _build_relu2(num_experts: int, hidden: int, inter: int, seed: int):
    """One Relu2 MoE layer: kernel-ready tensors plus the reference operands.

    FC1 is the up projection alone (`[E, I, H/2]`), every expert has its own
    `weight_scale_2` on both projections, and the layout comes from the
    self-contained helpers above -- the ones a target copies.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)
    up_p, up_s, up_c = _rand_nvfp4(num_experts, inter, hidden, gen)
    dn_p, dn_s, dn_c = _rand_nvfp4(num_experts, hidden, inter, gen)
    up_ws2 = _rand_weight_scale_2(num_experts, gen)
    dn_ws2 = _rand_weight_scale_2(num_experts, gen)
    w1, s1 = nvfp4_moe_single_layout(up_p, up_s)
    w2, s2 = nvfp4_moe_single_layout(dn_p, dn_s)
    args = dict(
        gemm1_weights=w1,
        gemm1_weights_scale=s1,
        gemm2_weights=w2,
        gemm2_weights_scale=s2,
        intermediate_size=inter,
    )
    ref = dict(
        up=(up_c, up_s, up_ws2),
        gate=None,
        down=(dn_c, dn_s, dn_ws2),
        raw=(up_p, up_s, dn_p, dn_s),
        hidden=hidden,
        inter=inter,
        num_experts=num_experts,
        gen=gen,
        act_type=ACT_TYPE_RELU2,
    )
    return args, ref


# Nemotron-3 Ultra's routed experts in latent space, built once and shared:
# 512 experts, H = 2048, I = 5120, Relu2. The kernel-ready stack is ~6 GiB (FC1
# [512, 5120, 1024] + [512, 5120, 128], FC2 [512, 2048, 2560] + [512, 2048,
# 320]) and the reference operands and raw packed copies take it to ~23 GiB.
_ULTRA = None


def _ultra():
    global _ULTRA
    if _ULTRA is None:
        _ULTRA = _build_relu2(512, 2048, 5120, seed=5122048)
    return _ULTRA


# The routing arguments the tep4 target passes with its pre-routed call. Inert
# on a cold cache; under `autotune()` DeepSeekV3's dummy routing needs all
# three as numbers.
_ULTRA_ROUTING_ARGS = dict(
    routing_method_type=2, n_group=1, topk_group=1, routed_scaling_factor=5.0
)

# The token column swept at that geometry, per 128-wide window: the static-block
# permutation kernel's last count (4), the dynamic-block kernel's span (5-12),
# the single-cluster kernel from 13 up, decode (1-32) through the 8192-row
# prefill ceiling; cold-cache FC1 token tiles 8 (to 32 rows), 16, 64 and 256.
_ULTRA_TOKENS = (1, 2, 4, 5, 8, 12, 13, 32, 64, 128, 300, 1024, 4096, 8192)


def _sigmoid_x5_routing(num_tokens, num_experts, top_k, gen):
    """Pre-routed ids/weights in the form Nemotron-3 Ultra's router emits.

    Sigmoid scores; selection on scores plus a large, nearly flat correction
    bias (selection only, as the router's is); the selected *unbiased* scores
    renormalized to sum to 1 and scaled by 5. Distinct ids per row.
    """
    scores = torch.sigmoid(
        torch.randn(num_tokens, num_experts, device=DEV, dtype=torch.float32, generator=gen)
    )
    bias = 25.25 + 0.125 * torch.rand(num_experts, device=DEV, generator=gen)
    idx = torch.topk(scores + bias, top_k, dim=-1).indices
    w = torch.gather(scores, 1, idx)
    w = w / w.sum(-1, keepdim=True) * 5.0
    return idx.to(torch.int32).contiguous(), w.to(torch.bfloat16).contiguous()


def _ultra_scalars(ref, g1: float, g2: float):
    """The three per-expert vectors for the whole 512-expert stack, from the
    checkpoint-form quantities a target holds: input scales `1/g1`, `1/g2`
    shared by the layer, per-expert `weight_scale_2`. Returns `(s1, sg, s2,
    g2_used)` where `g2_used` is the fp32 FC2-input scale the kernel is handed
    -- the value a reference has to requantize with to match it."""
    up_is = torch.tensor(1.0 / g1, dtype=torch.float32, device=DEV)
    down_is = torch.tensor(1.0 / g2, dtype=torch.float32, device=DEV)
    _, s1, sg, s2 = relu2_moe_scalars(up_is, ref["up"][2], down_is, ref["down"][2])
    return s1, sg, s2, s1[0].item()


# ── tests ─────────────────────────────────────────────────────────────────


def test_layout_helpers_match():
    """The trtllm preprocessing ops reproduce the pure-torch layout exactly.

    The last two scale shapes are the DeepSeek-R1 routed geometry's, at
    H = 7168 / I = 2048: FC1 `[E, 2*I, H/16] = [E, 4096, 448]` and FC2
    `[E, H, I/16] = [E, 7168, 128]`. Both satisfy the swizzle's `M % 128 == 0`
    and `C % 4 == 0`, so `block_scale_interleave` returns exactly `E*M*C` bytes
    and reshapes back — no padding enters either operand at this geometry.
    """
    gen = torch.Generator(device=DEV).manual_seed(201)
    for rows, cols in ((256, 12), (4096, 448), (7168, 128)):
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
        if ref[k] is not None:
            sub[k] = tuple(t[off : off + num_local] for t in ref[k])
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


def test_relu2_layout_matches_the_engine_loader():
    """The self-contained Relu2 layout helpers are trtllm's own MoE weight prep.

    Pinned at Nemotron-3 Ultra's shapes -- FC1 `[5120, 1024]` bytes with
    `[5120, 128]` scales, FC2 `[2048, 2560]` with `[2048, 320]` -- against
    the row permutations trtllm's fused-MoE loader builds for a non-gated FC1
    and for FC2 (`trtllmgen_maybe_get_cached_*_permute_indices`, epilogue tile
    128), applied by `shuffle_matrix`, and against `block_scale_interleave` for
    the scales. Bitwise. The loader's gated FC1 permutation (the `[up; gate]`
    interleave) is a different one at the same shape, so a helper that applied
    it would fail here.
    """
    from tensorrt_llm._torch.moe.fused_moe.quantization import (
        trtllmgen_maybe_get_cached_w2_permute_indices,
        trtllmgen_maybe_get_cached_w3_w1_permute_indices,
    )

    gen = torch.Generator(device=DEV).manual_seed(2025)
    num_experts = 3
    for rows, packed_cols, which in ((5120, 1024, "fc1"), (2048, 2560, "fc2")):
        packed = torch.randint(
            0, 256, (num_experts, rows, packed_cols), dtype=torch.uint8, device=DEV, generator=gen
        )
        sf = torch.randint(
            0, 256, (num_experts, rows, packed_cols * 2 // SV), dtype=torch.uint8, device=DEV,
            generator=gen,
        )  # fmt: skip
        weights, scales = nvfp4_moe_single_layout(packed, sf.view(torch.float8_e4m3fn))
        assert weights.shape == packed.shape and scales.shape == sf.shape
        for e in range(num_experts):
            if which == "fc1":
                perm_w = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
                    packed[e], {}, 128, is_gated_act_gemm=False
                )
                perm_s = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
                    sf[e], {}, 128, num_elts_per_sf=SV, is_gated_act_gemm=False
                )
                gated = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
                    packed[e], {}, 128, is_gated_act_gemm=True
                )
                assert not torch.equal(gated.to(DEV), perm_w.to(DEV)), (
                    "the gated permutation no longer differs from the non-gated one"
                )
            else:
                perm_w = trtllmgen_maybe_get_cached_w2_permute_indices(packed[e], {}, 128)
                perm_s = trtllmgen_maybe_get_cached_w2_permute_indices(
                    sf[e], {}, 128, num_elts_per_sf=SV
                )
            ref_w = torch.ops.trtllm.shuffle_matrix(packed[e].contiguous(), perm_w.to(DEV))
            ref_s = torch.ops.trtllm.block_scale_interleave(
                torch.ops.trtllm.shuffle_matrix(sf[e].contiguous(), perm_s.to(DEV))
            )
            assert torch.equal(weights[e], ref_w), f"{which} rows differ from the loader's"
            assert ref_s.numel() == sf[e].numel(), f"{which} scales were padded"
            assert torch.equal(scales[e].view(torch.uint8).reshape(-1), ref_s.reshape(-1)), (
                f"{which} scales differ from the loader's shuffle + 128x4 swizzle"
            )
    print("  test_relu2_layout_matches_the_engine_loader OK")


def test_relu2_intermediate_is_nvfp4_requantized():
    """Read the Relu2 FC1 epilogue straight out of the kernel, at every token tile.

    The SwiGlu tests' device -- a down projection that is the identity on the
    intermediate's columns, a unit FC2 scalar, one slot per row -- at
    Nemotron-3 Ultra's latent width H = 2048 (I = 2048 so the identity fits;
    the FC1 kernel is picked by H and the token tile, not by I), with
    per-expert `weight_scale_2` and the scalars from `relu2_moe_scalars`. A
    16-wide window at offset 384 of 512 experts makes the cold-cache token
    tile walk 8, 16, 32, 64, 128 and 256 within 4096 rows, and each tile is a
    different kernel, so each is pinned.

    Exact on every element against
        C = output1_scale_scalar * relu(output1_scale_gate_scalar * acc) ** 2
    quantized once: `sf = e4m3(amax16(C) / 6)` saturating at 448, `e2m1(C /
    sf)`. `acc` is the FC1 accumulator, computed exactly here (dyadic operands
    whose sums fit fp32's mantissa, so every accumulation order gives it), and
    the products are the fp32 products the kernel forms. A reference that
    instead dequantizes the activation first (`x / g1`, as `reference` does)
    is off in the last fp32 bit, which moves an e2m1 rounding or a block scale
    on a few elements in a million -- counted here, and absorbed by the
    entry's band rather than unexplained by it.

    Also driven: an FC2-input scale 8x too large for the data, so block scales
    saturate at 448 and codes at 6 -- what a static checkpoint scale does when
    an activation outruns its calibration -- still exact; and the two scalar
    mistakes a Relu2 caller can make, the SwiGlu convention
    (`output1_scale_scalar * output1_scale_gate_scalar`) and the two
    swapped, each matching almost none of the nonzero elements.
    """
    num_experts, hidden, inter, nloc, off = 512, 2048, 2048, 16, 384
    gen = torch.Generator(device=DEV).manual_seed(20482048)
    up_p, up_s, up_c = _rand_nvfp4(nloc, inter, hidden, gen)
    up_ws2 = _rand_weight_scale_2(nloc, gen)
    w1, sw1 = nvfp4_moe_single_layout(up_p, up_s)
    sel = torch.zeros(nloc, hidden, inter, dtype=torch.uint8, device=DEV)
    d = torch.arange(inter, device=DEV)
    sel[:, d, d] = 2  # e2m1 code 2 == +1.0
    dn_p = (sel[..., 0::2] | (sel[..., 1::2] << 4)).contiguous()
    dn_s = torch.ones(nloc, hidden, inter // SV, device=DEV).to(torch.float8_e4m3fn)
    w2, sw2 = nvfp4_moe_single_layout(dn_p, dn_s)
    del sel, dn_p, up_p
    weights = dict(
        gemm1_weights=w1,
        gemm1_weights_scale=sw1,
        gemm2_weights=w2,
        gemm2_weights_scale=sw2,
        intermediate_size=inter,
    )
    up_dq = [_dequant(up_c[e], up_s[e]) for e in range(nloc)]  # exact: dyadic values
    g1 = 137.0
    up_is = torch.tensor(1.0 / g1, dtype=torch.float32, device=DEV)
    unit = torch.ones(nloc, dtype=torch.float32, device=DEV)

    plain_flips = 0
    compared = 0
    saturated = 0
    for num_tokens in (16, 128, 256, 512, 1024, 2048, 4096):
        data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
        xs = _unpack_activation(data, sf, hidden, 1.0)  # codes * block scales, exact
        ids = (
            torch.randint(0, nloc, (num_tokens, 1), dtype=torch.int32, device=DEV, generator=gen)
            + off
        )
        wts = torch.ones(num_tokens, 1, dtype=torch.bfloat16, device=DEV)
        # output1_scale_gate_scalar does not depend on the down scale
        sg = relu2_moe_scalars(up_is, up_ws2, torch.ones((), device=DEV), unit)[2]
        act = torch.zeros(num_tokens, inter, dtype=torch.float32, device=DEV)
        plain = torch.zeros_like(act)
        for e in (ids[:, 0] - off).unique().tolist():
            tok = (ids[:, 0] == off + e).nonzero(as_tuple=True)[0]
            acc = (xs[tok].double() @ up_dq[e].double().t()).float()
            r = torch.relu(acc * sg[e])
            act[tok] = r * r
            rp = torch.relu((x[tok] @ up_dq[e].t()) * up_ws2[e])
            plain[tok] = rp * rp
        g2 = E4M3_MAX * E2M1_MAX / act.max().item()
        for over in (1.0, 8.0):
            down_is = torch.tensor(1.0 / (g2 * over), dtype=torch.float32, device=DEV)
            _, s1, sg_, _ = relu2_moe_scalars(up_is, up_ws2, down_is, unit)
            assert torch.equal(sg_, sg)
            y = _call(
                data,
                sf,
                weights,
                num_experts,
                1,
                act_type=ACT_TYPE_RELU2,
                local_expert_offset=off,
                local_num_experts=nloc,
                topk_ids=ids,
                topk_weights=wts,
                output1_scale_scalar=s1,
                output1_scale_gate_scalar=sg,
                output2_scale_scalar=unit,
            )[0].float()
            g = s1[0].item()
            exact = _q_nvfp4(act, g)
            assert torch.equal(y, exact), (
                f"T={num_tokens}, FC2-input scale x{over:g}: the Relu2 FC1 epilogue is not "
                f"o1 * relu(o1g * acc)**2 quantized once: {(y != exact).sum().item()} of "
                f"{y.numel()} elements differ"
            )
            plain_flips += (y != _q_nvfp4(plain, g)).sum().item()
            compared += y.numel()
            if over > 1.0:
                saturated += (exact.abs() == E2M1_MAX * E4M3_MAX).sum().item()
        if num_tokens == 1024:
            keep = (data, sf, ids, wts, s1, sg, exact)
    assert saturated > 0, "the x8 scale saturated nothing, so saturation was not driven"
    assert plain_flips <= 1e-5 * compared, (
        f"the dequantize-first reference differs on {plain_flips} of {compared} elements -- "
        "more than last-bit rounding can explain"
    )

    # The scalar mistakes, against the 1024-row result with the x8 scale.
    data, sf, ids, wts, s1, sg, exact = keep
    nonzero = exact != 0
    fracs = {}
    for name, o1, o1g in (
        ("SwiGlu-convention output1_scale_scalar", (s1 * sg).contiguous(), sg),
        ("output1 scalars swapped", sg, s1),
    ):
        bad = _call(
            data,
            sf,
            weights,
            num_experts,
            1,
            act_type=ACT_TYPE_RELU2,
            local_expert_offset=off,
            local_num_experts=nloc,
            topk_ids=ids,
            topk_weights=wts,
            output1_scale_scalar=o1,
            output1_scale_gate_scalar=o1g,
            output2_scale_scalar=unit,
        )[0].float()
        fracs[name] = (bad[nonzero] == exact[nonzero]).float().mean().item()
        assert fracs[name] < 0.1, f"{name} still matches {fracs[name]:.1%} of the nonzero elements"
    print(
        f"  test_relu2_intermediate_is_nvfp4_requantized OK (bit-exact on all {compared} "
        f"elements over tiles 8-256, {saturated} saturated; dequantize-first reference "
        f"flips {plain_flips}; nonzero matches: "
        + ", ".join(f"{n} {v:.1%}" for n, v in fracs.items())
        + ")"
    )


def test_ultra_relu2_geometry():
    """E=512, H=2048, I=5120, top-22, Relu2: Nemotron-3 Ultra's latent experts.

    As its tep4 target runs them: every rank routes the same tokens over all
    512 experts and runs its own 128-wide window, and the four windows' bf16
    outputs sum (the all-reduce). Over the whole token column
    `_ULTRA_TOKENS`, each window is gated against the reference over its own
    experts, rows routed entirely outside a window must come back bitwise
    zero, and the four-window sum is gated against the 512-expert reference.
    Scalars come from `relu2_moe_scalars` -- input scales shared by the
    layer, `weight_scale_2` per expert -- the routing from
    `_sigmoid_x5_routing`, and the routing arguments are the target's.

    Then, at 300 rows in the last window, the wrong computations a target
    could ship, each required to land more than 10 rms ulp from the correct
    result against the 2-ulp gate: the SwiGlu `output1_scale_scalar`
    convention, the two output1 scalars swapped, one layer-wide alpha for
    every expert, a reference without the FC1-output requantization, scales
    left unswizzled, rows left unshuffled, SwiGlu's `[up; gate]` interleave
    applied to the up rows, and the window offset left at 0.
    """
    args, ref = _ultra()
    num_experts, hidden, inter, top_k, shard = 512, 2048, 5120, 22, 128
    windows = [(r * shard, shard) for r in range(4)]
    win_args = [_window_args(args, off, n) for off, n in windows]
    win_ref = [_window_ref(ref, off, n) for off, n in windows]
    gen = torch.Generator(device=DEV).manual_seed(22512)
    g1 = 137.0
    worst_win = (0.0, 0.0)
    worst_sum = (0.0, 0.0)
    worst_drop = 1e9

    for num_tokens in _ULTRA_TOKENS:
        data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
        ids, wts = _sigmoid_x5_routing(num_tokens, num_experts, top_k, gen)
        s1, sg, s2, g2 = _ultra_scalars(ref, g1, _calibrate_g2(x, ids, ref))
        parts = []
        for (off, n), wa, wr in zip(windows, win_args, win_ref):
            y = _call(
                data,
                sf,
                wa,
                num_experts,
                top_k,
                act_type=ACT_TYPE_RELU2,
                local_expert_offset=off,
                local_num_experts=n,
                topk_ids=ids,
                topk_weights=wts,
                **_window_scalars(s1, sg, s2, off, n),
                **_ULTRA_ROUTING_ARGS,
            )[0]
            assert y.shape == (num_tokens, hidden) and y.dtype == torch.bfloat16, y.shape
            exp = _ref_moe(x, ids, wts, wr, g2, offset=off, num_local=n).to(torch.bfloat16)
            _assert_moe_close(y, exp)
            outside = ~((ids >= off) & (ids < off + n)).any(dim=1)
            assert torch.equal(y[outside], torch.zeros_like(y[outside])), (
                f"window (off={off}) wrote into rows routed entirely elsewhere at T={num_tokens}"
            )
            worst_win = tuple(max(a, b) for a, b in zip(worst_win, _dev(y, exp)))
            parts.append(y)
        exp_full = _ref_moe(x, ids, wts, ref, g2).to(torch.bfloat16)
        summed = sum(p.float() for p in parts).to(torch.bfloat16)
        _assert_moe_close(summed, exp_full)
        worst_sum = tuple(max(a, b) for a, b in zip(worst_sum, _dev(summed, exp_full)))
        if num_tokens >= 256:
            for drop in range(4):
                partial = sum(p.float() for i, p in enumerate(parts) if i != drop)
                _, rms = _dev(partial.to(torch.bfloat16), exp_full)
                assert rms > 25.0, f"dropping window {drop} at T={num_tokens} is only {rms:.1f}"
                worst_drop = min(worst_drop, rms)

    # The wrong computations, at 300 rows in the last window.
    off, n = windows[3]
    data, sf, x = _rand_activation(300, hidden, g1, gen)
    ids, wts = _sigmoid_x5_routing(300, num_experts, top_k, gen)
    s1, sg, s2, g2 = _ultra_scalars(ref, g1, _calibrate_g2(x, ids, ref))
    good_scalars = _window_scalars(s1, sg, s2, off, n)
    kw = dict(
        act_type=ACT_TYPE_RELU2,
        local_num_experts=n,
        topk_ids=ids,
        topk_weights=wts,
        **_ULTRA_ROUTING_ARGS,
    )
    good_ref = _ref_moe(x, ids, wts, win_ref[3], g2, offset=off, num_local=n).to(torch.bfloat16)
    good = _call(
        data, sf, win_args[3], num_experts, top_k, local_expert_offset=off, **good_scalars, **kw
    )[0]
    _assert_moe_close(good, good_ref)

    # The routing arguments do not reach a pre-routed result at (512, top-22)
    # either: the other methods a caller might pass, and the target's own with
    # its group and scaling arguments absent, are all bitwise the target's.
    for routing in (
        dict(routing_method_type=0),
        dict(routing_method_type=1),
        dict(routing_method_type=4),
        dict(routing_method_type=6),
        dict(routing_method_type=2, n_group=None, topk_group=None, routed_scaling_factor=None),
    ):
        alt = _call(
            data,
            sf,
            win_args[3],
            num_experts,
            top_k,
            local_expert_offset=off,
            **good_scalars,
            **{**kw, **routing},
        )[0]
        assert torch.equal(alt, good), f"{routing} changed the pre-routed result at 512 / top-22"

    o1 = good_scalars["output1_scale_scalar"]
    o1g = good_scalars["output1_scale_gate_scalar"]
    o2 = good_scalars["output2_scale_scalar"]
    up_p, up_s, dn_p, dn_s = (t[off : off + n] for t in ref["raw"])
    interleave = _fc1_perm(inter)
    bad_layouts = {
        "scales left unswizzled": (
            _prep_fc2(up_p, up_s, swizzle=False),
            _prep_fc2(dn_p, dn_s, swizzle=False),
        ),
        "rows left unshuffled": (
            _prep_fc2(up_p, up_s, shuffle=False),
            _prep_fc2(dn_p, dn_s, shuffle=False),
        ),
        "SwiGlu's [up; gate] interleave on the up rows": (
            (
                torch.index_select(up_p, 1, interleave).contiguous(),
                _swizzle_scales(torch.index_select(up_s, 1, interleave).view(torch.uint8))
                .view(torch.float8_e4m3fn)
                .contiguous(),
            ),
            (win_args[3]["gemm2_weights"], win_args[3]["gemm2_weights_scale"]),
        ),
    }
    scalar_mistakes = {
        "SwiGlu-convention output1_scale_scalar": (
            dict(good_scalars, output1_scale_scalar=(o1 * o1g).contiguous())
        ),
        "output1 scalars swapped": (
            dict(good_scalars, output1_scale_scalar=o1g, output1_scale_gate_scalar=o1)
        ),
        "one layer-wide alpha for every expert": (
            dict(
                good_scalars,
                output1_scale_gate_scalar=o1g.mean().expand(n).contiguous(),
                output2_scale_scalar=o2.mean().expand(n).contiguous(),
            )
        ),
    }
    dist = {}
    for name, scalars in scalar_mistakes.items():
        bad = _call(
            data, sf, win_args[3], num_experts, top_k, local_expert_offset=off, **scalars, **kw
        )[0]
        dist[name] = _dev(bad, good_ref)
    for name, ((bw1, bs1), (bw2, bs2)) in bad_layouts.items():
        bad_args = dict(
            gemm1_weights=bw1,
            gemm1_weights_scale=bs1,
            gemm2_weights=bw2,
            gemm2_weights_scale=bs2,
            intermediate_size=inter,
        )
        bad = _call(
            data, sf, bad_args, num_experts, top_k, local_expert_offset=off, **good_scalars, **kw
        )[0]
        dist[name] = _dev(bad, good_ref)
    no_requant = _ref_moe(
        x, ids, wts, win_ref[3], g2, offset=off, num_local=n, quantize_intermediate=False
    )
    dist["no FC1-output requantization"] = _dev(good, no_requant.to(torch.bfloat16))
    at_zero = _call(
        data, sf, win_args[3], num_experts, top_k, local_expert_offset=0, **good_scalars, **kw
    )[0]
    dist["window offset left at 0"] = _dev(at_zero, good_ref)
    for name, (elt, rms) in dist.items():
        assert rms > 10.0, f"{name} is only {rms:.1f} rms ulp from the correct result"
    print(
        f"  test_ultra_relu2_geometry OK (window {worst_win[0]:.2f} elt / {worst_win[1]:.2f} "
        f"rms, four-window sum {worst_sum[0]:.2f} elt / {worst_sum[1]:.2f} rms, one window "
        f"dropped >= {worst_drop:.0f} rms ulp; wrong variants "
        + ", ".join(f"{k} {v[1]:.0f}" for k, v in dist.items())
        + " rms ulp)"
    )


def test_relu2_act_type_must_match_fc1_rows():
    """Gated or not is read off FC1's row count; the kernel is picked by act_type.

    The op accepts either row count with either activation. Relu2 handed a
    SwiGlu stack (`[E, 2I, H/2]`) walks it at I-row strides and returns
    plausible rows from the wrong weights -- driven raw here, and far from
    the Relu2 layer the caller meant. SwiGlu handed a Relu2 stack would read
    2I rows per expert from an I-row stack, past its end, so that direction is
    only shown refused by the guard, before the kernel.
    """
    num_experts, hidden, inter, top_k, num_tokens = 8, 256, 128, 2, 16
    gated_args, gated_ref = _build(num_experts, hidden, inter, seed=1212)
    relu2_args, relu2_ref = _build_relu2(num_experts, hidden, inter, seed=1313)
    gen = relu2_ref["gen"]
    g1 = 137.0
    data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)

    meant = dict(gated_ref, gate=None, act_type=ACT_TYPE_RELU2)  # relu2 over the up halves
    g2 = _calibrate_g2(x, ids, meant)
    s1 = torch.full((num_experts,), g2, dtype=torch.float32, device=DEV)
    sg = torch.full((num_experts,), 1.0 / g1, dtype=torch.float32, device=DEV)
    s2 = torch.full((num_experts,), 1.0 / g2, dtype=torch.float32, device=DEV)
    raw = torch.ops.trtllm.fp4_block_scale_moe_runner(
        None,
        None,
        data,
        sf.view(torch.float8_e4m3fn),
        gated_args["gemm1_weights"],
        gated_args["gemm1_weights_scale"],
        None,
        None,
        None,
        None,
        gated_args["gemm2_weights"],
        gated_args["gemm2_weights_scale"],
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
        ACT_TYPE_RELU2,
        topk_weights=wts,
        topk_ids=ids,
    )[0]
    assert torch.isfinite(raw.float()).all() and raw.abs().max() > 0
    elt, rms = _dev(raw, _ref_moe(x, ids, wts, meant, g2).to(torch.bfloat16))
    assert rms > 25.0, f"Relu2 over a SwiGlu stack is only {rms:.1f} rms ulp off -- re-check"

    scalars = dict(output1_scale_scalar=s1, output1_scale_gate_scalar=sg, output2_scale_scalar=s2)
    with pytest.raises(AssertionError, match=f"act_type 1 needs FC1 with {inter} rows"):
        _call(data, sf, gated_args, num_experts, top_k, act_type=ACT_TYPE_RELU2,
              topk_ids=ids, topk_weights=wts, **scalars)  # fmt: skip
    with pytest.raises(AssertionError, match=f"act_type 0 needs FC1 with {2 * inter} rows"):
        _call(data, sf, relu2_args, num_experts, top_k, act_type=ACT_TYPE_SWIGLU,
              topk_ids=ids, topk_weights=wts, **scalars)  # fmt: skip
    print(
        f"  test_relu2_act_type_must_match_fc1_rows OK (Relu2 over a SwiGlu stack: {rms:.0f} rms ulp)"
    )


def test_ultra_window_replays_under_a_cuda_graph():
    """The Relu2 window call captures into a CUDA graph and replays exactly.

    A tep4 decode step runs as a captured graph. Captured here at the decode
    ceiling (32 rows) in the last window with the target's routing arguments,
    then replayed three times with fresh activations and fresh routing copied
    into the captured input buffers; each replay must be bitwise the eager
    call on the same buffers. One static FC2-input scale serves every step,
    as a served layer's does. The cache is cold; a warm one is bitwise inert
    (`test_ultra_autotuner_cache_is_inert`).
    """
    args, ref = _ultra()
    num_experts, hidden, top_k, off, n, num_tokens = 512, 2048, 22, 384, 128, 32
    wa = _window_args(args, off, n)
    gen = torch.Generator(device=DEV).manual_seed(3232)
    g1 = 137.0

    def draw():
        data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
        ids, wts = _sigmoid_x5_routing(num_tokens, num_experts, top_k, gen)
        return [data, sf.view(torch.float8_e4m3fn), ids, wts], x

    bufs, x = draw()
    s1, sg, s2, _ = _ultra_scalars(ref, g1, _calibrate_g2(x, bufs[2], ref))
    scalars = _window_scalars(s1, sg, s2, off, n)

    def step():
        data, sf, ids, wts = bufs
        return moe(
            None,
            None,
            data,
            sf,
            wa["gemm1_weights"],
            wa["gemm1_weights_scale"],
            None,
            None,
            None,
            None,
            wa["gemm2_weights"],
            wa["gemm2_weights_scale"],
            None,
            scalars["output1_scale_scalar"],
            scalars["output1_scale_gate_scalar"],
            scalars["output2_scale_scalar"],
            num_experts,
            top_k,
            _ULTRA_ROUTING_ARGS["n_group"],
            _ULTRA_ROUTING_ARGS["topk_group"],
            wa["intermediate_size"],
            off,
            n,
            _ULTRA_ROUTING_ARGS["routed_scaling_factor"],
            _ULTRA_ROUTING_ARGS["routing_method_type"],
            True,
            ACT_TYPE_RELU2,
            topk_weights=wts,
            topk_ids=ids,
        )[0]

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for replay in range(3):
        fresh, _ = draw()
        for buf, new in zip(bufs, fresh):
            buf.copy_(new)
        graph.replay()
        eager = step()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager), (
            f"replay {replay} differs from the eager call on the same inputs: "
            f"{_dev(captured, eager.float())} (elt / rms ulp)"
        )
    print("  test_ultra_window_replays_under_a_cuda_graph OK (3 replays bitwise eager)")


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


def test_ultra_autotuner_cache_is_inert(clear_autotuner_cache):
    """A warm autotuner cache leaves the Relu2 windows' results unchanged, and
    warming it needs the routing arguments as numbers.

    The tep4 target's engine profiles this op during warm-up, so the Relu2
    windows run warm in service while every other call here runs cold. Same
    shape as the R1 test: cold results first, then warm through trtllm's own
    `autotune()` at the target's call (`_ULTRA_ROUTING_ARGS`, 8192 rows) for
    the first and last window, then **bitwise** equality over the token column.

    Before warming, the one place the pre-routed call reads its routing
    arguments: under `autotune()` the op builds DeepSeekV3's dummy routing
    (`routing_method_type` 2) in Python, and `None` for `n_group` /
    `topk_group` or for `routed_scaling_factor` raises `TypeError` there --
    while the same call on a cold cache is fine.
    """
    args, ref = _ultra()
    num_experts, hidden, top_k = 512, 2048, 22
    windows = ((0, 128), (384, 128))
    win_args = {off: _window_args(args, off, n) for off, n in windows}
    gen = torch.Generator(device=DEV).manual_seed(2222)
    g1 = 137.0
    token_counts = (1, 8, 32, 300, 1024, 8192)
    cases = {}
    for num_tokens in token_counts:
        data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
        ids, wts = _sigmoid_x5_routing(num_tokens, num_experts, top_k, gen)
        cases[num_tokens] = (
            data,
            sf,
            ids,
            wts,
            _ultra_scalars(ref, g1, _calibrate_g2(x, ids, ref)),
        )

    def run(off, n, num_tokens, **over):
        data, sf, ids, wts, (s1, sg, s2, _) = cases[num_tokens]
        return _call(
            data,
            sf,
            win_args[off],
            num_experts,
            top_k,
            act_type=ACT_TYPE_RELU2,
            local_expert_offset=off,
            local_num_experts=n,
            topk_ids=ids,
            topk_weights=wts,
            **_window_scalars(s1, sg, s2, off, n),
            **{**_ULTRA_ROUTING_ARGS, **over},
        )[0]

    tuner = AutoTuner.get()
    op = "trtllm::fp4_block_scale_moe_runner"

    def entries():
        return [k for k in tuner.profiling_cache.cache if k[0] == op]

    assert not entries(), (
        "the profiling cache already holds entries for this op — the cold-cache "
        "receipt every other test in this file records would be void"
    )
    cold = {(off, t): run(off, n, t) for off, n in windows for t in token_counts}
    nones = (dict(n_group=None, topk_group=None), dict(routed_scaling_factor=None))
    for over in nones:
        assert torch.equal(run(0, 128, 32, **over), cold[(0, 32)]), f"{over} changed a cold call"
    assert not entries(), "an ordinary call wrote a profiling-cache entry"

    for over in nones:
        with pytest.raises(TypeError, match="NoneType"):
            with autotune():
                run(0, 128, token_counts[-1], **over)
    tuner.clear_cache()
    with autotune():
        for off, n in windows:
            run(off, n, token_counts[-1])
    keys = entries()
    assert keys, "autotune() recorded nothing for this op — nothing was warmed"
    tactics = {str(tuner.profiling_cache.cache[k][1]) for k in keys}
    assert "-1" not in tactics, (
        f"the tuner recorded the fallback tactic, so warm and cold are the same "
        f"execution and this comparison proves nothing: {sorted(tactics)}"
    )
    unique_ids = sorted({str(k[2]) for k in keys})
    for off, n in windows:
        for num_tokens in token_counts:
            warm = run(off, n, num_tokens)
            assert torch.equal(warm, cold[(off, num_tokens)]), (
                f"a warm autotuner cache changed the result at local_expert_offset={off}, "
                f"T={num_tokens}: {_dev(warm, cold[(off, num_tokens)].float())} (elt / rms ulp)"
            )
    print(
        f"  test_ultra_autotuner_cache_is_inert OK ({len(keys)} tuned entries over unique_ids "
        f"{unique_ids}, {len(tactics)} distinct tactics; all "
        f"{len(windows) * len(token_counts)} (window, T) results bitwise unchanged)"
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


# The shared layer each cell's (num_experts, hidden, intermediate, act_type) names.
_GEOMETRIES = {
    (256, 7168, 2048, ACT_TYPE_SWIGLU): _dsr1,
    (512, 2560, 640, ACT_TYPE_SWIGLU): _many_experts,
    (512, 2048, 5120, ACT_TYPE_RELU2): _ultra,
}

# How each cell's `routing` names its pre-routed ids and weights.
_ROUTINGS = {None: _routing, "renormalized": _renorm_routing, "sigmoid_x5": _sigmoid_x5_routing}


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

    A Relu2 cell is driven the way its target calls the op: scalars from the
    checkpoint-form quantities through `relu2_moe_scalars` (per-expert
    `weight_scale_2`, layer-shared input scales) and the target's routing
    arguments, `_ULTRA_ROUTING_ARGS`.
    """
    certifying(moe)
    spec = cell.spec
    act_type = spec.get("act_type", ACT_TYPE_SWIGLU)
    key = (spec["num_experts"], spec["hidden"], spec["intermediate"], act_type)
    args, ref = _GEOMETRIES[key]()
    off, n = spec["local_expert_offset"], spec["local_num_experts"]
    num_experts, hidden, top_k = spec["num_experts"], spec["hidden"], spec["top_k"]
    gen = torch.Generator(device=DEV).manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    g1 = 137.0

    data, sf, x = _rand_activation(spec["tokens"], hidden, g1, gen)
    ids, wts = _ROUTINGS[spec.get("routing")](spec["tokens"], num_experts, top_k, gen)
    g2 = _calibrate_g2(x, ids, ref)
    if act_type == ACT_TYPE_RELU2:
        s1, sg, s2, g2 = _ultra_scalars(ref, g1, g2)
        routing_args = _ULTRA_ROUTING_ARGS
    else:
        s1, sg, s2 = _scalars(num_experts, g1, g2)
        routing_args = {}

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
        act_type=act_type,
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
        act_type=act_type,
        local_expert_offset=off,
        local_num_experts=n,
        topk_ids=ids,
        topk_weights=wts,
        **_window_scalars(s1, sg, s2, off, n),
        **routing_args,
    )[0]
    assert y.shape == (spec["tokens"], hidden), y.shape
    _compare(y, entry_ref.to(torch.bfloat16))
