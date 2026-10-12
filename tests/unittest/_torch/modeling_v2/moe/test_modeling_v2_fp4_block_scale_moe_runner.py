# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp4_block_scale_moe_runner catalog entry."""

import zlib

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    fp4_block_scale_moe_runner as moe,
)
from tensorrt_llm._torch.autotuner import AutoTuner, autotune

__extra_import_path__ = [".."]
from _validating import validating  # noqa: E402 — needs the path declared above

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
    Both gates were sized against deliberately wrong computations when they
    were set. At H=256/I=128 and at H=7168/I=2048 respectively: a gate/up
    swap (263 / 278 ulp), a scalar-role swap (177 / 48), a missing scale
    swizzle (408 / 357), a
    missing row shuffle (508 / 432), and a reference that skips the FC1-output
    requantization (43 / 28 ulp element, 23 / 23 ulp RMS); the smallest RMS
    distance any of those reaches is 39.8, at the R1 shape's scalar-role swap.
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


# ── tests ─────────────────────────────────────────────────────────────────


def test_layout_helpers_match():
    """The trtllm preprocessing ops reproduce the pure-torch layout exactly.

    Two scale shapes are the DeepSeek-R1 routed geometry's, at
    H = 7168 / I = 2048: FC1 `[E, 2*I, H/16] = [E, 4096, 448]` and FC2
    `[E, H, I/16] = [E, 7168, 128]`; four are DeepSeek-V4.1's, at H = 5120:
    FC1 `[E, 4608, 320]` / FC2 `[E, 5120, 144]` for I = 2304 and FC1
    `[E, 1152, 320]` / FC2 `[E, 5120, 36]` for a MoE-TP4 rank's I = 576. All
    satisfy the swizzle's `M % 128 == 0` and `C % 4 == 0`, so
    `block_scale_interleave` returns exactly `E*M*C` bytes and reshapes back —
    no padding enters any operand at these geometries.
    """
    gen = torch.Generator(device=DEV).manual_seed(201)
    for rows, cols in (
        (256, 12),
        (4096, 448),
        (7168, 128),
        (4608, 320),
        (5120, 144),
        (1152, 320),
        (5120, 36),
    ):
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


# ── DeepSeek-V4.1-Flash routed geometry ───────────────────────────────────
#
# 384 routed experts, H = 5120, I = 2304, top-6, and the clamped SwiGLU
# gate = min(x W_gate^T, 10), up = clamp(x W_up^T, -10, 10). Unlike the geometries
# above, every operand carries a PER-EXPERT NVFP4 global scale, as a modelopt
# checkpoint does: the FC1 and FC2 weight global scales `gw1[e]`, `gw2[e]` and
# the FC2-input global scale `g2[e]` (one per expert, as a checkpoint's
# `w2.input_scale` is), so the three scale scalars and the clamp limit are
# genuinely different numbers for every expert.

V41_E, V41_H, V41_I, V41_K = 384, 5120, 2304, 6
V41_LIMIT = 10.0
V41_EP = 96  # local experts per rank under four-way expert parallelism
V41_TP_I = 576  # intermediate columns per rank under four-way MoE tensor parallelism
_V41_TOKENS = (1, 2, 3, 5, 8, 16, 32, 64, 128, 256, 1024, 4096, 8192)
_V41 = None


def _v41():
    """The V4.1 routed layer, built once: kernel-ready tensors, reference
    operands, per-expert global scales, and a static per-expert g2."""
    global _V41
    if _V41 is None:
        args, ref = _build(V41_E, V41_H, V41_I, seed=4141)
        gen = torch.Generator(device=DEV).manual_seed(4142)
        # FC1 weight global scales log-uniform in [1/8, 1/2]: with g1 = 137 the
        # true-domain gate/up RMS then runs from ~4 to ~16 across experts, so
        # the limit of 10 bites on anywhere from ~1% to ~50% of an expert's
        # elements. FC2's in [1/2, 2].
        ref["gw1"] = torch.exp2(torch.empty(V41_E, device=DEV).uniform_(-3.0, -1.0, generator=gen))
        ref["gw2"] = torch.exp2(torch.empty(V41_E, device=DEV).uniform_(-1.0, 1.0, generator=gen))
        ref["g1"] = 137.0
        # One static FC2-input global scale per expert, fixed before any test
        # batch is drawn (as a checkpoint's is): 448*6 / the amax of that
        # expert's clamped activation over a 4096-token calibration batch,
        # times a per-expert factor in [1/2, 1] so no two experts share it.
        _, _, xc = _rand_activation(4096, V41_H, ref["g1"], gen)
        idc, _ = _v41_routing(4096, gen)
        amax = torch.zeros(V41_E, device=DEV)
        for e in range(V41_E):
            tok = (idc == e).any(dim=1).nonzero(as_tuple=True)[0]
            up, gate = _v41_fc1(xc[tok], ref, e, V41_LIMIT)
            amax[e] = (up * gate * torch.sigmoid(gate)).abs().max()
        factor = torch.empty(V41_E, device=DEV).uniform_(0.5, 1.0, generator=gen)
        ref["g2"] = E4M3_MAX * E2M1_MAX / amax * factor
        _V41 = (args, ref)
    return _V41


def _v41_routing(num_tokens, gen):
    """Distinct top-6 ids of 384 and V4.1-style weights (positive, summing to 1.5)."""
    ids = torch.rand(num_tokens, V41_E, device=DEV, generator=gen).argsort(dim=1)[:, :V41_K]
    w = torch.rand(num_tokens, V41_K, device=DEV, generator=gen) + 0.05
    w = w / w.sum(dim=1, keepdim=True) * 1.5
    return ids.to(torch.int32).contiguous(), w.to(torch.bfloat16).contiguous()


def _v41_fc1(xe, ref, e, limit, swap_gate_up=False):
    """True-domain (up, gate) of expert `e` over rows `xe`, clamped when `limit`."""
    up_c, up_s = ref["up"]
    gt_c, gt_s = ref["gate"]
    wu = _dequant(up_c[e], up_s[e]) / ref["gw1"][e]
    wg = _dequant(gt_c[e], gt_s[e]) / ref["gw1"][e]
    if swap_gate_up:
        wu, wg = wg, wu
    up, gate = xe @ wu.t(), xe @ wg.t()
    if limit is not None:
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
    return up, gate


def _ref_v41(x, ids, wts, ref, limit=V41_LIMIT, swap_gate_up=False):
    """Native-torch V4.1 routed MoE, fp32, over all 384 experts.

    Returns `(full, windows, slices)`: the whole layer, the four 96-expert
    expert-parallel windows' shares, and the four 576-column MoE-TP slices'
    shares. A slice's share is exact to take from the full computation: its
    columns start on a multiple of 16, so the FC1-output NVFP4 blocks — and
    with the expert's own g2 their scales — are the ones the full layer has.
    Also returns the fraction of gate / up elements the limit clamped.
    """
    num_tokens = x.shape[0]
    dn_c, dn_s = ref["down"]
    g2 = ref["g2"]
    full = torch.zeros(num_tokens, V41_H, dtype=torch.float32, device=DEV)
    windows = [torch.zeros_like(full) for _ in range(V41_E // V41_EP)]
    slices = [torch.zeros_like(full) for _ in range(V41_I // V41_TP_I)]
    clamped = torch.zeros(2, dtype=torch.float64, device=DEV)
    count = 0
    for e in range(V41_E):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        up, gate = _v41_fc1(x[tok], ref, e, None, swap_gate_up)
        if limit is not None:
            clamped[0] += (gate > limit).sum()
            clamped[1] += (up.abs() > limit).sum()
            count += gate.numel()
            gate = torch.clamp(gate, max=limit)
            up = torch.clamp(up, min=-limit, max=limit)
        act = up * gate * torch.sigmoid(gate)
        g2e = float(g2[e])
        act = _q_nvfp4(act, g2e) / g2e
        wd = _dequant(dn_c[e], dn_s[e]) / ref["gw2"][e]
        w = wts[tok, slot].float().unsqueeze(1)
        y = (act @ wd.t()) * w
        full.index_add_(0, tok, y)
        windows[e // V41_EP].index_add_(0, tok, y)
        for s in range(len(slices)):
            a, b = s * V41_TP_I, (s + 1) * V41_TP_I
            slices[s].index_add_(0, tok, (act[:, a:b] @ wd[:, a:b].t()) * w)
    frac = (clamped / max(count, 1)).tolist()
    return full, windows, slices, frac


def _v41_scalars(ref, g2=None, gw1=None, limit=V41_LIMIT):
    """The kernel's four per-expert fp32 tensors for the V4.1 layer.

    With `alpha1[e] = 1 / (g1 * gw1[e])` (the FC1 accumulator's dequant):
    `output1_scale_gate_scalar = alpha1`, `output1_scale_scalar = alpha1 * g2`,
    `output2_scale_scalar = 1 / (g2 * gw2)`, and the clamp in the accumulator
    domain, `gemm1_clamp_limit = limit / alpha1`.
    """
    g2 = ref["g2"] if g2 is None else g2
    gw1 = ref["gw1"] if gw1 is None else gw1
    a1 = 1.0 / (ref["g1"] * gw1)
    return dict(
        output1_scale_scalar=(a1 * g2).contiguous(),
        output1_scale_gate_scalar=a1.contiguous(),
        output2_scale_scalar=(1.0 / (g2 * ref["gw2"])).contiguous(),
        gemm1_clamp_limit=(limit / a1).contiguous(),
    )


def _v41_slice(d, off: int, n: int):
    return {k: v[off : off + n].contiguous() for k, v in d.items()}


_V41_TP_ARGS = {}


def _v41_tp_args(width: int = V41_TP_I):
    """Kernel-ready operands of the MoE-TP ranks at `width` intermediate columns
    per rank (576: the four MoE-TP4 ranks; 1152: the two MoE-TP2 ranks): rank r
    holds intermediate columns [width r, width (r + 1)) of every expert — its
    rows of up and gate, its columns of down — prepared exactly like a full
    layer with I = width."""
    if width not in _V41_TP_ARGS:
        _, ref = _v41()
        up_p, gt_p, up_s, gt_s, dn_p, dn_s = ref["raw"]
        ranks = []
        for r in range(V41_I // width):
            a, b = r * width, (r + 1) * width
            w1, s1 = _prep_fc1(up_p[:, a:b], gt_p[:, a:b], up_s[:, a:b], gt_s[:, a:b])
            w2, s2 = _prep_fc2(
                dn_p[:, :, a // 2 : b // 2].contiguous(), dn_s[:, :, a // SV : b // SV].contiguous()
            )
            ranks.append(
                dict(
                    gemm1_weights=w1,
                    gemm1_weights_scale=s1,
                    gemm2_weights=w2,
                    gemm2_weights_scale=s2,
                    intermediate_size=width,
                )
            )
        _V41_TP_ARGS[width] = ranks
    return _V41_TP_ARGS[width]


def test_v41_clamp_is_in_the_accumulator_domain():
    """`gemm1_clamp_limit` clamps the FC1 accumulator, before any scalar.

    Read straight out of the kernel (identity FC2 with unit block scales and
    `output2_scale_scalar = 1`, so the output IS the requantized intermediate,
    bit for bit) with four experts whose `output1_scale_gate_scalar` (alpha1)
    and FC2-input global scales g2 are all different, one g2 deliberately too
    large so some blocks saturate the e4m3 scale at 448. Passing
    `gemm1_clamp_limit[e] = L[e] / alpha1[e]` gives, bit-exactly on every
    element, `gate = min(gate_true, L)`, `up = clamp(up_true, -L, L)` — gate
    clamped from above only, up on both sides, both halves by the same alpha1
    (up's own scalar `output1_scale_scalar` plays no part). A true-domain limit
    handed over as is clamps at `L * alpha1` instead.
    """
    num_experts, hidden, top_k, num_tokens = 4, 256, 1, 256
    gen = torch.Generator(device=DEV).manual_seed(4321)
    up_p, up_s, up_c = _rand_nvfp4(num_experts, hidden, hidden, gen)
    gt_p, gt_s, gt_c = _rand_nvfp4(num_experts, hidden, hidden, gen)
    w1, s1 = _prep_fc1(up_p, gt_p, up_s, gt_s)
    eye = torch.zeros(num_experts, hidden, hidden, dtype=torch.uint8, device=DEV)
    d = torch.arange(hidden, device=DEV)
    eye[:, d, d] = 2  # e2m1 code 2 == +1.0
    dn_p = (eye[..., 0::2] | (eye[..., 1::2] << 4)).contiguous()
    dn_s = torch.ones(num_experts, hidden, hidden // SV, device=DEV).to(torch.float8_e4m3fn)
    w2, s2 = _prep_fc2(dn_p, dn_s)
    args = dict(
        gemm1_weights=w1,
        gemm1_weights_scale=s1,
        gemm2_weights=w2,
        gemm2_weights_scale=s2,
        intermediate_size=hidden,
    )
    g1 = 137.0
    gw1 = torch.tensor([1.0, 0.37, 2.9, 0.0061], device=DEV)  # per-expert FC1 weight global scales
    alpha1 = 1.0 / (g1 * gw1)
    data, sf, x = _rand_activation(num_tokens, hidden, g1, gen)
    ids = torch.arange(num_tokens, device=DEV, dtype=torch.int32).remainder(num_experts)
    ids = ids.unsqueeze(1).contiguous()
    eid = ids[:, 0].long()
    wts = torch.ones(num_tokens, 1, dtype=torch.bfloat16, device=DEV)
    up_t = torch.zeros(num_tokens, hidden, device=DEV)
    gate_t = torch.zeros(num_tokens, hidden, device=DEV)
    for e in range(num_experts):
        tok = (eid == e).nonzero(as_tuple=True)[0]
        up_t[tok] = x[tok] @ _dequant(up_c[e], up_s[e]).t() / gw1[e]
        gate_t[tok] = x[tok] @ _dequant(gt_c[e], gt_s[e]).t() / gw1[e]
    # a true-domain limit per expert that bites on ~40% of its gate elements
    lim = torch.stack([gate_t[eid == e].abs().flatten().quantile(0.6) for e in range(num_experts)])
    act_free = up_t * gate_t * torch.sigmoid(gate_t)
    g2 = torch.stack(
        [E4M3_MAX * E2M1_MAX / act_free[eid == e].abs().max() for e in range(num_experts)]
    ) * torch.tensor([1.0, 0.5, 0.25, 4.0], device=DEV)

    def kernel(clamp):
        return _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts,
            output1_scale_scalar=(alpha1 * g2).contiguous(),
            output1_scale_gate_scalar=alpha1.contiguous(),
            output2_scale_scalar=torch.ones(num_experts, device=DEV),
            gemm1_clamp_limit=clamp,
        )[0].float()

    def expect(gate_lim, up_lim, gate_both=False, up_above_only=False):
        g, u = gate_t.clone(), up_t.clone()
        if gate_lim is not None:
            gl = gate_lim[eid].unsqueeze(1)
            g = torch.clamp(g, min=-gl, max=gl) if gate_both else torch.minimum(g, gl)
        if up_lim is not None:
            ul = up_lim[eid].unsqueeze(1)
            u = torch.minimum(u, ul) if up_above_only else torch.clamp(u, min=-ul, max=ul)
        a = u * g * torch.sigmoid(g)
        out = torch.empty_like(a)
        for e in range(num_experts):
            r = eid == e
            out[r] = _q_nvfp4(a[r], float(g2[e]))
        return out

    assert torch.equal(kernel(None), expect(None, None)), "unclamped intermediate is not bit-exact"
    saturated = sum(
        int(
            (
                (g2[e] * act_free[eid == e]).reshape(-1, SV).abs().amax(-1) / E2M1_MAX > E4M3_MAX
            ).sum()
        )
        for e in range(num_experts)
    )
    assert saturated > 0, "no e4m3 block scale saturated — the g2 x4 expert did not exercise it"

    y = kernel((lim / alpha1).contiguous())
    assert torch.equal(y, expect(lim, lim)), (
        f"accumulator-domain clamp: {(y != expect(lim, lim)).sum().item()} of {y.numel()} "
        "elements differ from gate<=L, |up|<=L"
    )
    # the readings that are NOT what the kernel does, each far from bit-exact
    fracs = {}
    for name, cand in (
        ("clamp in the true domain", expect(lim / alpha1, lim / alpha1)),
        ("gate clamped on both sides", expect(lim, lim, gate_both=True)),
        ("up clamped from above only", expect(lim, lim, up_above_only=True)),
        ("up clamped after output1_scale_scalar", expect(lim, lim / (alpha1 * g2))),
        ("gate only", expect(lim, None)),
        ("up only", expect(None, lim)),
    ):
        fracs[name] = (y == cand).float().mean().item()
        assert fracs[name] < 0.9, f"'{name}' also matches {fracs[name]:.2%} — no discrimination"
    # control: a true-domain limit handed over as is clamps at L * alpha1
    raw = kernel(lim.contiguous())
    assert torch.equal(raw, expect(lim * alpha1, lim * alpha1))
    wrong = (raw == expect(lim, lim)).float().mean().item()
    assert wrong < 0.9, f"true-domain limit passed raw still matches {wrong:.2%}"
    clamped = (gate_t > lim[eid].unsqueeze(1)).float().mean().item()
    print(
        f"  test_v41_clamp_is_in_the_accumulator_domain OK (bit-exact on all {y.numel()} elements, "
        f"gate clamped on {clamped:.1%}, {saturated} saturated block scales; other readings match "
        f"{max(fracs.values()):.1%} at most; the true-domain limit passed raw matches {wrong:.1%})"
    )


def test_v41_routed_geometry():
    """The V4.1 routed expert layer: 384 experts, H=5120, I=2304, top-6, clamp.

    Three layouts, over the whole token column (1 through 8192 rows per call):
    the full 384-expert stack at offset 0; the four EP4 windows (96 local
    experts at offsets 0 / 96 / 192 / 288, `num_experts` 384 on every call);
    and the four MoE-TP4 ranks (`I = 576` — a multiple of 64 but not of 128 —
    with all 384 experts local). Every expert has its own FC1/FC2 weight global
    scale, FC2-input global scale and clamp limit.
    """
    args, ref = _v41()
    tp_args = _v41_tp_args()
    gen = torch.Generator(device=DEV).manual_seed(4143)
    g1 = ref["g1"]
    scal = _v41_scalars(ref)
    worst = dict(
        full=(0.0, 0.0), win=(0.0, 0.0), ep_sum=(0.0, 0.0), tp=(0.0, 0.0), tp_sum=(0.0, 0.0)
    )
    worst_ep_vs_full = worst_tp_vs_full = 0.0
    zero_rows = 0
    clamp_frac = [1.0, 1.0]
    top = {}

    def keep(name, y, exp):
        dv = _dev(y, exp)
        worst[name] = tuple(max(a, b) for a, b in zip(worst[name], dv))
        top[name] = tuple(max(a, b) for a, b in zip(top.get(name, (0.0, 0.0)), dv))

    for num_tokens in _V41_TOKENS:
        top = {}
        data, sf, x = _rand_activation(num_tokens, V41_H, g1, gen)
        ids, wts = _v41_routing(num_tokens, gen)
        ref_full, ref_win, ref_tp, frac = _ref_v41(x, ids, wts, ref)
        if num_tokens >= 256:
            clamp_frac = [min(a, b) for a, b in zip(clamp_frac, frac)]
        exp_full = ref_full.to(torch.bfloat16)
        full = _call(data, sf, args, V41_E, V41_K, topk_ids=ids, topk_weights=wts, **scal)[0]
        assert full.shape == (num_tokens, V41_H) and full.dtype == torch.bfloat16
        _assert_moe_close(full, exp_full)
        keep("full", full, exp_full)

        parts = []
        for r in range(V41_E // V41_EP):
            off = r * V41_EP
            y = _call(
                data,
                sf,
                _window_args(args, off, V41_EP),
                V41_E,
                V41_K,
                local_expert_offset=off,
                local_num_experts=V41_EP,
                topk_ids=ids,
                topk_weights=wts,
                **_v41_slice(scal, off, V41_EP),
            )[0]
            exp = ref_win[r].to(torch.bfloat16)
            _assert_moe_close(y, exp)
            keep("win", y, exp)
            outside = ~((ids >= off) & (ids < off + V41_EP)).any(dim=1)
            assert torch.equal(y[outside], torch.zeros_like(y[outside])), (
                f"EP window {r} wrote rows routed entirely elsewhere at T={num_tokens}"
            )
            zero_rows = max(zero_rows, int(outside.sum()))
            parts.append(y)
        ep_sum = sum(p.float() for p in parts).to(torch.bfloat16)
        _assert_moe_close(ep_sum, exp_full)
        keep("ep_sum", ep_sum, exp_full)
        worst_ep_vs_full = max(worst_ep_vs_full, _dev(ep_sum, full.float())[0])

        parts = []
        for r, ta in enumerate(tp_args):
            y = _call(data, sf, ta, V41_E, V41_K, topk_ids=ids, topk_weights=wts, **scal)[0]
            exp = ref_tp[r].to(torch.bfloat16)
            _assert_moe_close(y, exp)
            keep("tp", y, exp)
            parts.append(y)
        tp_sum = sum(p.float() for p in parts).to(torch.bfloat16)
        _assert_moe_close(tp_sum, exp_full)
        keep("tp_sum", tp_sum, exp_full)
        worst_tp_vs_full = max(worst_tp_vs_full, _dev(tp_sum, full.float())[0])

    # the clamp was genuinely exercised on both halves at every batch of >= 256 rows
    assert clamp_frac[0] > 0.01 and clamp_frac[1] > 0.01, clamp_frac
    fmt = lambda d: f"{d[0]:.2f} elt / {d[1]:.2f} rms"  # noqa: E731
    print(
        f"  test_v41_routed_geometry OK (worst: full {fmt(worst['full'])}, EP4 window "
        f"{fmt(worst['win'])}, EP4 sum {fmt(worst['ep_sum'])}, TP4 rank {fmt(worst['tp'])}, "
        f"TP4 sum {fmt(worst['tp_sum'])} ulp; EP4 sum vs full call {worst_ep_vs_full:.2f} elt, "
        f"TP4 sum vs full call {worst_tp_vs_full:.2f} elt; <= {zero_rows} bitwise-zero rows per "
        f"window; clamp active on >= {clamp_frac[0]:.1%} of gate / {clamp_frac[1]:.1%} of up "
        f"elements; at T={_V41_TOKENS[-1]}: full {fmt(top['full'])}, window {fmt(top['win'])}, "
        f"TP4 rank {fmt(top['tp'])})"
    )


def test_v41_wrong_operands_are_discriminated():
    """The V4.1 layer's plausible mistakes, each run and measured against the gate.

    At T = 1024 on the V4.1 geometry: the clamp limit given in the true domain
    (10, not 10 / alpha1); FC1 stacked `[gate ; up]` (the dense-MLP order)
    instead of `[up ; gate]`; an EP window issued at `local_expert_offset = 0`
    (tokens routed to its experts are silently dropped, others computed with the
    wrong expert's weights); per-expert scalars shifted by one expert; one TP4
    rank left out of the sum. Each must land far outside the 2-ulp RMS gate the
    correct call passes.
    """
    args, ref = _v41()
    tp_args = _v41_tp_args()
    gen = torch.Generator(device=DEV).manual_seed(4144)
    num_tokens = 1024
    data, sf, x = _rand_activation(num_tokens, V41_H, ref["g1"], gen)
    ids, wts = _v41_routing(num_tokens, gen)
    ref_full, ref_win, ref_tp, _ = _ref_v41(x, ids, wts, ref)
    exp_full = ref_full.to(torch.bfloat16)
    scal = _v41_scalars(ref)
    good = _call(data, sf, args, V41_E, V41_K, topk_ids=ids, topk_weights=wts, **scal)[0]
    _assert_moe_close(good, exp_full)
    _, good_rms = _dev(good, exp_full)
    off = V41_EP  # window 1
    win1 = _window_args(args, off, V41_EP)
    up_p, gt_p, up_s, gt_s, _, _ = ref["raw"]
    w1_swapped, s1_swapped = _prep_fc1(gt_p, up_p, gt_s, up_s)
    bad = {
        "clamp limit in the true domain": (
            _call(
                data,
                sf,
                args,
                V41_E,
                V41_K,
                topk_ids=ids,
                topk_weights=wts,
                **{**scal, "gemm1_clamp_limit": torch.full((V41_E,), V41_LIMIT, device=DEV)},
            )[0],
            exp_full,
        ),
        "FC1 stacked [gate ; up]": (
            _call(
                data,
                sf,
                {**args, "gemm1_weights": w1_swapped, "gemm1_weights_scale": s1_swapped},
                V41_E,
                V41_K,
                topk_ids=ids,
                topk_weights=wts,
                **scal,
            )[0],
            exp_full,
        ),
        "EP window 1 at offset 0": (
            _call(
                data,
                sf,
                win1,
                V41_E,
                V41_K,
                local_expert_offset=0,
                local_num_experts=V41_EP,
                topk_ids=ids,
                topk_weights=wts,
                **_v41_slice(scal, off, V41_EP),
            )[0],
            ref_win[1].to(torch.bfloat16),
        ),
        "scalars shifted by one expert": (
            _call(
                data,
                sf,
                args,
                V41_E,
                V41_K,
                topk_ids=ids,
                topk_weights=wts,
                **{k: v.roll(1).contiguous() for k, v in scal.items()},
            )[0],
            exp_full,
        ),
        "one TP4 rank left out of the sum": (
            sum(
                _call(data, sf, ta, V41_E, V41_K, topk_ids=ids, topk_weights=wts, **scal)[0].float()
                for ta in tp_args[:-1]
            ).to(torch.bfloat16),
            exp_full,
        ),
    }
    # the correct window-1 call, for the offset control's scale
    right = _call(
        data,
        sf,
        win1,
        V41_E,
        V41_K,
        local_expert_offset=off,
        local_num_experts=V41_EP,
        topk_ids=ids,
        topk_weights=wts,
        **_v41_slice(scal, off, V41_EP),
    )[0]
    _assert_moe_close(right, ref_win[1].to(torch.bfloat16))
    lines = []
    for name, (y, exp) in bad.items():
        elt, rms = _dev(y, exp)
        assert rms > 25.0, (
            f"{name}: only {rms:.1f} rms ulp from the correct result — not discriminated"
        )
        lines.append(f"{name} {rms:.0f}")
    # the swapped-FC1 mistake is also visible from the reference side
    swapped_ref = _ref_v41(x, ids, wts, ref, swap_gate_up=True)[0].to(torch.bfloat16)
    _, rms_swapped_ref = _dev(good, swapped_ref)
    assert rms_swapped_ref > 25.0, rms_swapped_ref
    dropped = int(((ids >= off) & (ids < off + V41_EP)).any(dim=1).sum())
    print(
        f"  test_v41_wrong_operands_are_discriminated OK (correct call {good_rms:.2f} rms ulp; "
        f"wrong: {', '.join(lines)} rms ulp; kernel vs a [gate ; up] reference "
        f"{rms_swapped_ref:.0f}; the offset-0 window dropped the slots of {dropped} tokens)"
    )


def test_r1_autotuner_cache_is_inert():
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
    tuner.clear_cache()
    print(
        f"  test_r1_autotuner_cache_is_inert OK ({len(keys)} tuned entries over "
        f"unique_ids {unique_ids}, {len(tactics)} distinct tactics, "
        f"profiled-config counts {profiled} across the five layouts; all "
        f"{len(layouts) * len(_R1_TOKENS)} (expert layout, T) results bitwise "
        f"unchanged)"
    )


def test_v41_autotuner_cache_is_inert():
    """A warm autotuner cache does not change the V4.1 results.

    The V4.1 layer presents three tuning keys of its own — the 384-expert stack
    and the EP4 windows at `I = 2304` (96 local experts), and the MoE-TP4 ranks
    at `I = 576`. This warms all three through `autotune()` at 8192 rows, then
    re-runs every layout (full, four windows, four TP ranks) over a spread of
    token counts and demands **bitwise** equality with the cold results.
    Clears the cache at the end, like the R1 test.
    """
    args, ref = _v41()
    tp_args = _v41_tp_args()
    scal = _v41_scalars(ref)
    layouts = (
        [("full", 0, V41_E, args, scal)]
        + [
            (
                f"ep{r}",
                r * V41_EP,
                V41_EP,
                _window_args(args, r * V41_EP, V41_EP),
                _v41_slice(scal, r * V41_EP, V41_EP),
            )
            for r in range(V41_E // V41_EP)
        ]
        + [(f"tp{r}", 0, V41_E, ta, scal) for r, ta in enumerate(tp_args)]
    )
    tokens = (1, 8, 32, 256, 4096, 8192)
    gen = torch.Generator(device=DEV).manual_seed(4145)
    cases = {}
    for num_tokens in tokens:
        data, sf, _ = _rand_activation(num_tokens, V41_H, ref["g1"], gen)
        cases[num_tokens] = (data, sf) + _v41_routing(num_tokens, gen)

    def run(layout, num_tokens):
        _, off, n, la, ls = layout
        data, sf, ids, wts = cases[num_tokens]
        return _call(
            data,
            sf,
            la,
            V41_E,
            V41_K,
            local_expert_offset=off,
            local_num_experts=n,
            topk_ids=ids,
            topk_weights=wts,
            **ls,
        )[0]

    tuner = AutoTuner.get()
    op = "trtllm::fp4_block_scale_moe_runner"

    def entries():
        return [k for k in tuner.profiling_cache.cache if k[0] == op]

    assert not entries(), "the profiling cache already holds entries for this op"
    cold = {(lay[0], t): run(lay, t) for lay in layouts for t in tokens}
    assert not entries(), "an ordinary call wrote a profiling-cache entry"
    with autotune():
        for lay in layouts:
            if lay[0] in ("full", "ep0", "tp0"):
                run(lay, tokens[-1])
    keys = entries()
    unique_ids = sorted({str(k[2]) for k in keys})
    assert len(unique_ids) == 3, f"expected three tuning keys, got {unique_ids}"
    tactics = {str(tuner.profiling_cache.cache[k][1]) for k in keys}
    per_uid = {
        u: {str(tuner.profiling_cache.cache[k][1]) for k in keys if str(k[2]) == u}
        for u in unique_ids
    }
    assert all(v - {"-1"} for v in per_uid.values()), (
        f"a tuning key recorded only the fallback tactic, so warm == cold proves nothing there: {per_uid}"
    )
    for lay in layouts:
        for num_tokens in tokens:
            warm = run(lay, num_tokens)
            assert torch.equal(warm, cold[(lay[0], num_tokens)]), (
                f"a warm autotuner cache changed the {lay[0]} result at T={num_tokens}: "
                f"{_dev(warm, cold[(lay[0], num_tokens)].float())} (elt / rms ulp)"
            )
    tuner.clear_cache()
    print(
        f"  test_v41_autotuner_cache_is_inert OK ({len(keys)} tuned entries over unique_ids "
        f"{unique_ids}, {len(tactics)} distinct tactics; all {len(layouts) * len(tokens)} "
        f"(layout, T) results bitwise unchanged)"
    )


# ── V4.1 decode under CUDA-graph capture ──────────────────────────────────
# The V4.1 target calls this op inside its decode CUDA graphs, as an EP4 window
# (tep4 / dep4: 96 local experts, I = 2304) or a MoE-TP4 rank (tp4: all 384
# experts, I = 576), with the FC2-input global scale LAYER-level — one
# g2 = 1 / max_e w2.input_scale[e], i.e. min_e g2[e], in every slot — and its
# output buffer allocated inside the captured region.
#
# The captured row counts, per rank. `_GSET` is every CUDA-graph batch G an
# engine at max_batch_size <= 256 captures: every G 1..32 plus TRT-LLM's padded
# list at 256 (`_generate_cuda_graph_batch_sizes`: 1, 2, 4, 8, 16..128 in steps
# of 8, 192, 256); the padding-off list at 256 (1..32, 64, 128, 256) is a
# subset. A captured step is generation-only with next_n 1 (decode) or 6
# (DSpark target-verify: 1 + 5 drafts), so a tep4 EP4 window and a tp4 rank
# route T = G and 6G of their own rows (82 counts, 1..1536). dep4 gathers the
# four ranks' rows, each padded to the same G, ahead of the router onto its EP4
# windows: T = 4G and 24G (82 counts, 4..6144). The routed MoE chunks at 8192
# rows (`_moe_chunks`), so no captured call is split. dep4 pads each rank's
# block to the largest rank's row count with zero rows, which the router sends
# to one set of six experts; the "pad" case models that at every dep4 count.

_GSET = tuple(range(1, 33)) + (40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 192, 256)
_V41_RANK_TOKENS = tuple(sorted(set(_GSET) | {6 * g for g in _GSET}))  # tep4 / tp4: G, 6G
_V41_DEP4_TOKENS = tuple(sorted({4 * g for g in _GSET} | {24 * g for g in _GSET}))  # dep4: 4G, 24G
_V41_GATHERED_TOKENS = tuple(
    n for n in _V41_DEP4_TOKENS if n not in _V41_RANK_TOKENS
)  # EP4 windows only
_V41_GRAPH_TOKENS = _V41_RANK_TOKENS
_MOE_OP = "trtllm::fp4_block_scale_moe_runner"


def _v41_counts(base, nloc):
    """The captured row counts of one rank layout: `base`, plus dep4's gathered
    verify counts on an EP4 window (a TP4 rank is never gathered)."""
    return base + (_V41_GATHERED_TOKENS if nloc < V41_E else ())


def _moe_cache_entries():
    return [k for k in AutoTuner.get().profiling_cache.cache if k[0] == _MOE_OP]


def _v41_layer_g2():
    """The V4.1 layer with its FC2-input global scale made layer-level."""
    args, ref = _v41()
    return args, dict(ref, g2=torch.full_like(ref["g2"], float(ref["g2"].min())))


def _v41_graph_layouts(args, ref_l):
    """(name, offset, local experts, operands, scalars) of the eight rank layouts."""
    scal = _v41_scalars(ref_l)
    lays = [
        (
            f"ep{r}",
            r * V41_EP,
            V41_EP,
            _window_args(args, r * V41_EP, V41_EP),
            _v41_slice(scal, r * V41_EP, V41_EP),
        )
        for r in range(V41_E // V41_EP)
    ]
    lays += [(f"tp{r}", 0, V41_E, ta, scal) for r, ta in enumerate(_v41_tp_args())]
    return lays


def _v41_case(kind, n, off, nloc, g1, gen, ranks=4):
    """One replay's inputs. `skew` puts every token on the same six local experts;
    `none` routes every token outside this rank's window; `pad` is a dep member's
    zero-padded gathered batch over `ranks` ranks (dep4 by default: EP4 windows,
    `n` a multiple of 4; dep2 and dep8 on the EP2 and EP8 windows)."""
    data, sf, x = _rand_activation(n, V41_H, g1, gen)
    ids, wts = _v41_routing(n, gen)
    if kind == "skew":
        ids = (
            (off + torch.arange(V41_K, device=DEV, dtype=torch.int32)).expand(n, V41_K).contiguous()
        )
    elif kind == "pad":
        # The gathered batch: `ranks` equal rank blocks, rank 0 full, the others
        # ending in 0..rows-1 zero rows, every zero row routed to the same six
        # experts (spread over this window) with the same weights.
        rows = n // ranks
        pad = torch.zeros(n, dtype=torch.bool, device=DEV)
        for b in range(1, ranks):
            p = int(torch.randint(0, rows, (1,), device=DEV, generator=gen))
            pad[(b + 1) * rows - p : (b + 1) * rows] = True
        six = off + torch.tensor([5, 17, 33, 50, 71, 90], device=DEV, dtype=torch.int32) % nloc
        ids[pad] = six
        wts[pad] = torch.tensor([0.4, 0.3, 0.3, 0.2, 0.2, 0.1], device=DEV, dtype=torch.bfloat16)
        data[pad] = 0
        sf.view(n, V41_H // SV)[pad] = 0
        x[pad] = 0
    elif kind == "none":
        nxt = (off + nloc) % V41_E
        pick = torch.rand(n, nloc, device=DEV, generator=gen).argsort(dim=1)[:, :V41_K]
        ids = (nxt + pick).to(torch.int32).contiguous()
    return kind, data, sf, x, ids, wts


class _V41Graph:
    """One captured call over static input buffers, the way a decode graph holds it.

    With `chunks` (row ranges `[a, b)` tiling `n`, as `_moe_chunks` cuts a gathered
    batch) the graph holds one call per chunk, each on its rows of the static
    buffers and into its own output allocated inside the captured region."""

    def __init__(self, lay, n, first, chunks=None):
        self.lay, self.n = lay, n
        self.chunks = chunks or [(0, n)]
        self.data = torch.empty(n, V41_H // 2, dtype=torch.uint8, device=DEV)
        self.sf = torch.empty(n * V41_H // SV, dtype=torch.uint8, device=DEV)
        self.ids = torch.empty(n, V41_K, dtype=torch.int32, device=DEV)
        self.wts = torch.empty(n, V41_K, dtype=torch.bfloat16, device=DEV)
        self._load(first)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._body()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._body()

    def _body(self):
        _, off, nloc, la, ls = self.lay
        outs = []
        for a, b in self.chunks:
            out = torch.empty(b - a, V41_H, dtype=torch.bfloat16, device=DEV)
            _call(
                self.data[a:b],
                self.sf[a * V41_H // SV : b * V41_H // SV],
                la,
                V41_E,
                V41_K,
                local_expert_offset=off,
                local_num_experts=nloc,
                topk_ids=self.ids[a:b],
                topk_weights=self.wts[a:b],
                output=out,
                **ls,
            )
            outs.append(out)
        return outs[0] if len(outs) == 1 else outs

    def _load(self, case):
        _, data, sf, _, ids, wts = case
        self.data.copy_(data)
        self.sf.copy_(sf)
        self.ids.copy_(ids)
        self.wts.copy_(wts)

    def replay(self, case):
        self._load(case)
        self.graph.replay()
        torch.cuda.synchronize()
        return self.out.clone() if len(self.chunks) == 1 else torch.cat(self.out)

    def kernels(self, case):
        """Names of the CUDA kernels one replay launches."""
        self._load(case)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            self.graph.replay()
            torch.cuda.synchronize()
        return [
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0
        ]


def _v41_eager(lay, case, chunks=None):
    """The same call outside any graph, into a NaN-filled output buffer; with
    `chunks`, one call per chunk, each into its own NaN-filled buffer."""
    _, off, nloc, la, ls = lay
    _, data, sf, _, ids, wts = case
    outs = []
    for a, b in chunks or [(0, data.shape[0])]:
        out = torch.full((b - a, V41_H), float("nan"), dtype=torch.bfloat16, device=DEV)
        _call(
            data[a:b],
            sf[a * V41_H // SV : b * V41_H // SV],
            la,
            V41_E,
            V41_K,
            local_expert_offset=off,
            local_num_experts=nloc,
            topk_ids=ids[a:b],
            topk_weights=wts[a:b],
            output=out,
            **ls,
        )
        outs.append(out)
    return outs[0] if len(outs) == 1 else torch.cat(outs)


def test_v41_cuda_graph_replay_is_bitwise_eager():
    """Captured once per (rank layout, T), replayed on routing the capture never saw.

    Cold autotuner cache (fallback tactic), the target's call shape: pre-routed,
    clamp at 10, layer-level g2, `output` allocated inside the captured region.
    Each graph is captured after one side-stream warm-up call and replayed on two
    uniform routings, a skewed one (every token on the same six local experts)
    and — EP windows — one routing nothing into the window, then on the capture's
    own data again. At dep4's gathered counts an EP window also replays a
    zero-padded batch (`pad`). Every replay must be **bitwise** equal to an
    eager call on the same values; the replayed output is also held to the
    torch reference's gates (the empty-window replay to exact zeros).
    Row counts: G and 6G over `_GSET` (82 counts, 1..1536) on every layout, and
    dep4's gathered 4G and 24G (82 counts, 4..6144) on the EP4 windows. The
    reference is `_ref_v41_fast`, first tied bitwise to `_ref_v41`.
    """
    assert not _moe_cache_entries(), "the cold-cache graph test ran with a warm cache"
    args, ref_l = _v41_layer_g2()
    layouts = _v41_graph_layouts(args, ref_l)
    g1 = ref_l["g1"]
    gen = torch.Generator(device=DEV).manual_seed(4146)
    # The fast reference is the certified one, bit for bit (windows and TP slices).
    for kind, n, off, nloc in (
        ("uniform", 48, 0, V41_E),
        ("pad", 768, 96, V41_EP),
        ("skew", 1536, 0, V41_E),
    ):
        c = _v41_case(kind, n, off, nloc, g1, gen)
        fast, slow = _ref_v41_fast(c[3], c[4], c[5], ref_l), _ref_v41(c[3], c[4], c[5], ref_l)
        for r in range(4):
            assert torch.equal(fast[0][r], slow[1][r]) and torch.equal(fast[1][r], slow[2][r]), (
                kind,
                n,
                r,
            )
    counts = sorted(set(_V41_GRAPH_TOKENS) | set(_V41_GATHERED_TOKENS))
    assert len(_V41_GRAPH_TOKENS) == 82 and len(_V41_DEP4_TOKENS) == 82 and len(counts) == 122
    assert max(_V41_GRAPH_TOKENS) == 1536 and max(counts) == 6144
    worst = (0.0, 0.0)
    worst_at = [None, None]
    worst_big = (0.0, 0.0)
    replays = graphs = 0
    per_layout = {}
    for n in counts:
        lays = [lay for lay in layouts if n in _v41_counts(_V41_GRAPH_TOKENS, lay[2])]
        shared = [_v41_case("uniform", n, 0, V41_E, g1, gen) for _ in range(2)]
        refs = [
            _ref_v41_fast(c[3], c[4], c[5], ref_l, slices=n in _V41_GRAPH_TOKENS) for c in shared
        ]
        for lay in lays:
            name, off, nloc = lay[:3]
            r = int(name[2:])
            own = [_v41_case("skew", n, off, nloc, g1, gen)]
            if nloc < V41_E:
                own.append(_v41_case("none", n, off, nloc, g1, gen))
                if n in _V41_DEP4_TOKENS:
                    own.append(_v41_case("pad", n, off, nloc, g1, gen))
            exps = [rf[0 if nloc < V41_E else 1][r] for rf in refs]
            for c in own:
                rf = (
                    _ref_v41_fast(c[3], c[4], c[5], ref_l, slices=nloc == V41_E)
                    if c[0] != "none"
                    else None
                )
                exps.append(rf[0 if nloc < V41_E else 1][r] if rf else None)
            cases = shared + own + shared[:1]
            exps = exps + exps[:1]
            g = _V41Graph(lay, n, cases[0])
            prev = None
            for case, exp in zip(cases, exps):
                y = g.replay(case)
                eager = _v41_eager(lay, case)
                assert torch.equal(y, eager), (
                    f"{name} T={n} {case[0]}: graph replay differs from eager "
                    f"{_dev(y, eager.float())} (elt / rms ulp)"
                )
                if exp is None:
                    assert torch.equal(y, torch.zeros_like(y)), (
                        f"{name} T={n}: an empty window is not zero"
                    )
                else:
                    exp = exp.to(torch.bfloat16)
                    _assert_moe_close(y, exp)
                    dv = _dev(y, exp)
                    for i in range(2):
                        if dv[i] > worst[i]:
                            worst_at[i] = (name, n, case[0])
                    worst = tuple(max(a, b) for a, b in zip(worst, dv))
                    if n > max(_V41_GRAPH_TOKENS):
                        worst_big = tuple(max(a, b) for a, b in zip(worst_big, dv))
                # consecutive replays compute different results, so a replay that
                # kept serving the previous inputs could not pass (two all-zero
                # results — few tokens, none routed into this window — are exempt;
                # the skew case is never zero and the empty-window case always is)
                assert prev is None or not torch.equal(prev, y) or not y.any(), (
                    f"{name} T={n}: two cases gave one result"
                )
                prev = y
                replays += 1
            graphs += 1
            per_layout[name] = per_layout.get(name, 0) + 1
            del g
        del refs, shared
    assert not _moe_cache_entries()
    assert per_layout == {
        **{f"ep{r}": 122 for r in range(4)},
        **{f"tp{r}": 82 for r in range(4)},
    }, per_layout
    print(
        f"  test_v41_cuda_graph_replay_is_bitwise_eager OK ({graphs} (layout, T) graphs over "
        f"{len(layouts)} layouts {per_layout}, {replays} replays bitwise equal to eager; "
        f"worst vs reference {worst[0]:.2f} elt at {worst_at[0]} / {worst[1]:.2f} rms ulp at {worst_at[1]}, "
        f"at T 1537..6144 (dep4 only) "
        f"{worst_big[0]:.2f} elt / {worst_big[1]:.2f} rms ulp; layer g2 {float(ref_l['g2'][0]):.4g}; "
        f"EP4 counts {counts}; TP4 counts {list(_V41_GRAPH_TOKENS)})"
    )


def test_v41_cuda_graph_swizzled_scales_are_caught_at_gathered_counts():
    """Control: the graph harness sees the one silent layout mistake at dep4's counts.

    dep4's gathered counts 384, 768, 1536 and 6144 (24G at G 16, 32, 64, 256)
    are multiples of 128, where a 128x4-swizzled activation-scale buffer has
    exactly the linear one's byte count and passes the op's size check.
    Replaying a captured EP4 window (offsets 0 and 288) on the same scale bytes
    in swizzled order must fail the reference gate, while the linear bytes of
    the same case pass it.
    """
    args, ref_l = _v41_layer_g2()
    layouts = [lay for lay in _v41_graph_layouts(args, ref_l) if lay[0] in ("ep0", "ep3")]
    g1 = ref_l["g1"]
    gen = torch.Generator(device=DEV).manual_seed(4148)
    seen = []
    for n in (384, 768, 1536, 6144):
        case = _v41_case("uniform", n, 0, V41_E, g1, gen)
        windows = _ref_v41_windows(case[3], case[4], case[5], ref_l)
        kind, data, sf, x, ids, wts = case
        swz = _swizzle_scales(sf.view(1, n, V41_H // SV)).reshape(-1).contiguous()
        assert swz.numel() == sf.numel() and not torch.equal(swz, sf)
        for lay in layouts:
            name, off = lay[:2]
            exp = windows[off // V41_EP].to(torch.bfloat16)
            g = _V41Graph(lay, n, case)
            _assert_moe_close(g.replay(case), exp)
            bad = g.replay((kind, data, swz, x, ids, wts))
            elt, rms = _dev(bad, exp)
            assert rms > 2.0 and elt > 16.0, (
                f"{name} T={n}: swizzled scales passed ({elt:.1f} / {rms:.2f} ulp)"
            )
            seen.append((name, n, round(elt), round(rms)))
            del g
    print(
        "  test_v41_cuda_graph_swizzled_scales_are_caught_at_gathered_counts OK "
        f"(swizzled replay vs reference, elt / rms ulp: {seen}; gates 16 / 2)"
    )


# The first row count of each kernel change the eager profile of
# `test_v41_row_axis_by_bucket` finds besides the tuner buckets and untuned
# tiles: the routing kernel's instantiation (5, 65, 129) and the finalize
# kernel's vector-load variant (60).
_V41_ROUTING_FIRSTS = (5, 60, 65, 129)


def _v41_segment(n: int, nloc: int, warm: bool):
    """(lo, hi): the row counts around `n` that launch `n`'s kernels, per the eager
    profile of `test_v41_row_axis_by_bucket` -- `n`'s tuner bucket (warm) or untuned
    token tile (cold), cut at the routing / finalize kernel changes and, on an untuned
    TP4 rank, at 3 (its FC1 config)."""
    firsts = set(_V41_ROUTING_FIRSTS)
    if warm:
        firsts |= {1 << k for k in range(1, 14)}
    else:
        firsts |= {
            m for m in range(2, 8193) if _v41_cold_tile(m, nloc) != _v41_cold_tile(m - 1, nloc)
        }
        if nloc == V41_E:
            firsts.add(3)
    lo = max([1] + [f for f in firsts if f <= n])
    hi = min([8193] + [f for f in firsts if f > n]) - 1
    return lo, hi


def test_v41_cuda_graph_tactic_is_fixed_at_capture():
    """A graph keeps the tactic it was captured with; no tactic changes the result.

    The engine autotunes during warm-up and captures its decode graphs after, so
    its graphs hold tuned tactics. Here: graphs are captured cold (fallback
    tactic), then the cache is warmed through `autotune()` at 8192 rows for the
    two V4.1 tuning keys; the cold graphs replay unchanged — same kernels, same
    bits — while graphs captured warm launch the tuned kernels (different ones
    for some (layout, T), which is what makes the comparison mean something) and
    are still **bitwise** equal to the cold results, and so to a warm eager call.
    Clearing the cache after capture changes neither. Clears the cache at the
    end. Row counts: G and 6G over `_GSET` on every layout, and dep4's gathered
    4G and 24G on the EP4 windows (the counts of the replay test).

    Per-bucket kernel choice: on EP4 window 0 and TP4 rank 0 at every captured
    count, and on all four EP4 windows at dep4's counts above 1536, the kernels a
    graph replays (captured cold, then warm) are exactly the kernels an eager call
    launches at that count in the same tuner state, and at the lower and upper
    edge of the count's bucket (warm) or untuned tile (cold) (`_v41_segment`) --
    the edges `test_v41_row_axis_by_bucket` holds to the reference.
    """
    assert not _moe_cache_entries(), "the profiling cache already holds entries for this op"
    args, ref_l = _v41_layer_g2()
    layouts = _v41_graph_layouts(args, ref_l)
    g1 = ref_l["g1"]
    gen = torch.Generator(device=DEV).manual_seed(4147)
    kgen = torch.Generator(device=DEV).manual_seed(4157)
    eager_k = {"cold": {}, "warm": {}}

    def eager_kernels(state, lay, m):
        """Kernels of one eager call at row count m on `lay`, output preallocated."""
        key = (lay[0], m)
        if key not in eager_k[state]:
            _, off, nloc, la, ls = lay
            c = _v41_case("uniform", m, 0, V41_E, g1, kgen)
            out = torch.empty(m, V41_H, dtype=torch.bfloat16, device=DEV)

            def run():
                _call(
                    c[1],
                    c[2],
                    la,
                    V41_E,
                    V41_K,
                    local_expert_offset=off,
                    local_num_experts=nloc,
                    topk_ids=c[4],
                    topk_weights=c[5],
                    output=out,
                    **ls,
                )

            run()
            eager_k[state][key] = _kernel_names(run)
        return eager_k[state][key]

    big = max(_V41_GRAPH_TOKENS)
    segs = {"cold": {}, "warm": {}}

    def check_bucket(state, lay, n, k_graph):
        if not (lay[0] in ("ep0", "tp0") or n > big):
            return
        lo, hi = _v41_segment(n, lay[2], state == "warm")
        for m in (n, lo, hi):
            k = eager_kernels(state, lay, m)
            assert tuple(k_graph) == k, (
                f"{state} {lay[0]} T={n}: the graph's kernels differ from an eager call at T={m} "
                f"(segment {lo}..{hi}): {k_graph} vs {k}"
            )
        segs[state][(lay[0], n)] = (lo, hi)

    cold = {}
    for lay in layouts:
        name, off, nloc = lay[:3]
        for n in _v41_counts(_V41_GRAPH_TOKENS, nloc):
            kinds = ("uniform", "skew") + (("none",) if nloc < V41_E else ())
            cases = [_v41_case(k, n, off, nloc, g1, gen) for k in kinds]
            g = _V41Graph(lay, n, cases[0])
            k_cold = g.kernels(cases[0])
            check_bucket("cold", lay, n, k_cold)
            cold[(name, n)] = (g, cases, [_v41_eager(lay, c) for c in cases], k_cold)
    assert not _moe_cache_entries(), "an ordinary or captured call wrote a profiling-cache entry"

    with autotune():
        for lay in layouts:
            if lay[0] in ("ep0", "tp0"):
                _, off, nloc, la, ls = lay
                c = _v41_case("uniform", 8192, off, nloc, g1, gen)
                _call(
                    c[1],
                    c[2],
                    la,
                    V41_E,
                    V41_K,
                    local_expert_offset=off,
                    local_num_experts=nloc,
                    topk_ids=c[4],
                    topk_weights=c[5],
                    **ls,
                )
    keys = _moe_cache_entries()

    def tactic(k):
        t = AutoTuner.get().profiling_cache.cache[k][1]
        return tuple(t) if isinstance(t, (list, tuple)) else (t,)

    tuned = {(str(k[2]), k[3][2][0]): tactic(k) for k in keys}
    uids = sorted({u for u, _ in tuned})
    assert uids == ["(6, 2304, 96, 0)", "(6, 576, 384, 0)"], uids
    # every token bucket a graph batch maps to (the power of two at or below T)
    for u in uids:
        for n in _v41_counts(_V41_GRAPH_TOKENS, V41_EP if u == "(6, 2304, 96, 0)" else V41_E):
            b = 1 << (n.bit_length() - 1)
            assert tuned.get((u, b), (-1,)) != (-1,), (
                f"bucket {b} of {u} is not tuned: {tuned.get((u, b))}"
            )

    changed = []
    for lay in layouts:
        name = lay[0]
        for n in _v41_counts(_V41_GRAPH_TOKENS, lay[2]):
            g, cases, want, k_cold = cold[(name, n)]
            for c, w in zip(cases, want):
                assert torch.equal(g.replay(c), w), (
                    f"{name} T={n} {c[0]}: a cold graph changed after warming"
                )
                assert torch.equal(_v41_eager(lay, c), w), (
                    f"{name} T={n} {c[0]}: warm eager != cold eager"
                )
            assert g.kernels(cases[0]) == k_cold, (
                f"{name} T={n}: a cold graph re-read the tuner at replay"
            )
            del g
            warm = _V41Graph(lay, n, cases[0])
            k_warm = warm.kernels(cases[0])
            check_bucket("warm", lay, n, k_warm)
            if k_warm != k_cold:
                changed.append((name, n))
            for c, w in zip(cases, want):
                assert torch.equal(warm.replay(c), w), (
                    f"{name} T={n} {c[0]}: warm graph != cold result"
                )
            cold[(name, n)] = (warm, cases, want, k_warm)
    assert changed, (
        "no graph captured warm launched different kernels — the comparison proved nothing"
    )
    AutoTuner.get().clear_cache()
    for (name, n), (warm, cases, want, k_warm) in cold.items():
        assert warm.kernels(cases[0]) == k_warm, f"{name} T={n}: a warm graph re-read the tuner"
        for c, w in zip(cases, want):
            assert torch.equal(warm.replay(c), w), (
                f"{name} T={n} {c[0]}: warm graph changed after clear"
            )
    del cold
    assert len(segs["cold"]) == len(segs["warm"]) == 122 + 82 + 3 * 10, (
        len(segs["cold"]),
        len(segs["warm"]),
    )
    above = sorted(n for n in _V41_GATHERED_TOKENS if n > big)
    by_seg = {
        state: {
            f"{lo}..{hi}": [n for n in above if segs[state][("ep0", n)] == (lo, hi)]
            for lo, hi in sorted({segs[state][("ep0", n)] for n in above})
        }
        for state in ("cold", "warm")
    }
    print(
        f"  test_v41_cuda_graph_tactic_is_fixed_at_capture OK ({len(keys)} tuned entries over {uids}, "
        f"{len(set(tuned.values()))} distinct tactics; "
        f"{sum(len(_v41_counts(_V41_GRAPH_TOKENS, lay[2])) for lay in layouts)} "
        f"(layout, T) graphs, cold- and warm-captured, bitwise equal on every replay; warm capture launched "
        f"different kernels at {len(changed)} of them, e.g. {changed[:4]}; at T > {big}: "
        f"{[c for c in changed if c[1] > big]}; per-bucket kernel check at {len(segs['cold'])} (layout, T) "
        f"per state: graph kernels == eager kernels at T and at its segment's edges; dep4 counts {above} by "
        f"segment: cold {by_seg['cold']}, warm {by_seg['warm']})"
    )


# ── V4.1 eager row axis, bucket by bucket ──────────────────────────────────
# The op picks its kernels by row count in two places, so its row axis is
# certified by bucket rather than by a prefix argument:
# - the autotuner's token bucket (`FP4BlockScaleMoERunner.get_dynamic_tensor_specs`,
#   `_torch/custom_ops/trtllm_gen_custom_ops.py`): the largest power of two at or
#   below T, capped at 8192, so the buckets are [2^k, 2^(k+1) - 1] for k < 13 and
#   {8192}. A warm cache holds one tactic per bucket, tuned at its lower edge;
# - with no tuned tactic, the C++ runner (`FP4BlockScaleMoeRunner::run`,
#   `thop/moe/trtllmGen/fp4BlockScaleMoe.cpp`) picks the token tile as the next
#   power of two of the integer part of T * top_k / local_num_experts, clamped to
#   8..256 — so the tile edges fall at T = 16 * (tile + 1) for an EP4 window and
#   64 * (tile + 1) for a TP4 rank — and takes the first valid config of that tile.
# Around the GEMMs, kernels also change with T independently of both: the
# pre-routed routing step runs a single-block kernel up to 4 rows and a
# single-cluster kernel above (`RoutingFromTopKIds.cu`), whose instantiation
# changes again above 64 and 128 rows, and the finalize kernel turns into its
# vector-load variant from 60 rows. Profiled at every T from 1 to 8192 on one
# EP4 window and one TP4 rank, cold and warm, the launched kernels change only
# at the bucket edges, the untuned-tile edges and these (plus 2 / 3, where the
# untuned TP4 rank's FC1 config changes); both sides of each are run below.
# Every row count the target passes is at most 8192 (`_moe` chunks the gathered
# rows by 8192).

_V41_DUMPED_TOKENS = (48, 56, 192, 1018, 1935, 2048, 3552, 3560)  # EP4 (TP4: 48, 1018, 1935, 2048)
# Every row count the batch-256 / long-context engine dumps show on the EP4 windows
# (tep4 graph and eager at max_batch_size 256, L 1M at batch 8, L 128K at batch 32)
# and the TP4 ranks (tp4 graph at 256), captured and eager.
_V41_DUMPED_TOKENS_B256 = (
    1,
    2,
    4,
    5,
    6,
    7,
    8,
    12,
    16,
    18,
    24,
    29,
    32,
    34,
    48,
    60,
    96,
    108,
    128,
    134,
    144,
    168,
    192,
    228,
    240,
    288,
    330,
    336,
    384,
    432,
    462,
    480,
    528,
    546,
    576,
    610,
    624,
    672,
    720,
    768,
    780,
    852,
    1152,
    1172,
    1178,
    1536,
    2252,
    2255,
    2259,
    2778,
    2784,
    3870,
    4208,
    4245,
    4388,
    4394,
    4591,
    4868,
    8116,
    8124,
    8128,
    8130,
    8137,
    8141,
    8148,
    8192,
)
_V41_KERNEL_EDGES = (
    2,
    3,
    4,
    5,
    59,
    60,
    64,
    65,
    128,
    129,
)  # routing / finalize / TP4 cold FC1 config


def _v41_bucket_tokens():
    """Lower edge, one interior point and upper edge of every autotuner bucket."""
    out, b = set(), 1
    while b <= 8192:
        hi = min(2 * b - 1, 8192)
        out |= {b, (b + hi) // 2, hi}
        b *= 2
    return out


def _v41_cold_tile(n: int, nloc: int) -> int:
    """The untuned runner's token tile (the C++ runner's rule, int truncation and all)."""
    v = int(float(n * V41_K) / nloc)
    p = 0 if v <= 0 else 1 << (v - 1).bit_length()
    return min(max(p, 8), 256)


def _v41_tile_edges(nloc: int) -> set:
    """The last and first row count of every untuned-tile change up to 8192."""
    out = set()
    for n in range(2, 8193):
        if _v41_cold_tile(n, nloc) != _v41_cold_tile(n - 1, nloc):
            out |= {n - 1, n}
    return out


def _kernel_names(fn):
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return tuple(
        e.name
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0
    )


def test_v41_row_axis_by_bucket():
    """Every tuner bucket and untuned tile up to 8192, at the V4.1 layouts, cold and warm.

    For each EP4 window (96 local experts, I = 2304) and each TP4 rank (all 384,
    I = 576), with the target's call (pre-routed, clamp 10, layer-level g2): the
    lower edge, an interior point and the upper edge of every autotuner bucket,
    both sides of every untuned-tile edge of that layout and of every other
    observed kernel change, and every row count the runtime-value dump shows. Each call is held to the torch reference;
    rows need not be bitwise equal across row counts. Then the cache is warmed
    through `autotune()` at 8192 rows (one tactic per bucket, tuned at its lower
    edge) and every call is re-run on the same inputs against the reference —
    the warm tactic at a bucket's interior and upper edge is the one engines
    run. Clears the cache at the end.
    """
    assert not _moe_cache_entries(), "the profiling cache already holds entries for this op"
    args, ref_l = _v41_layer_g2()
    layouts = _v41_graph_layouts(args, ref_l)
    g1 = ref_l["g1"]
    gen = torch.Generator(device=DEV).manual_seed(4149)
    common = (
        _v41_bucket_tokens()
        | set(_V41_KERNEL_EDGES)
        | set(_V41_DUMPED_TOKENS)
        | set(_V41_DUMPED_TOKENS_B256)
    )
    counts = {
        V41_EP: sorted(common | _v41_tile_edges(V41_EP)),
        V41_E: sorted(common | _v41_tile_edges(V41_E)),
    }
    assert {143, 144, 2063, 2064} <= set(counts[V41_EP]) and {575, 576, 4159, 4160} <= set(
        counts[V41_E]
    )
    all_counts = sorted(set(counts[V41_EP]) | set(counts[V41_E]))

    def run(lay, case):
        _, off, nloc, la, ls = lay
        _, data, sf, _, ids, wts = case
        return _call(
            data,
            sf,
            la,
            V41_E,
            V41_K,
            local_expert_offset=off,
            local_num_experts=nloc,
            topk_ids=ids,
            topk_weights=wts,
            **ls,
        )[0]

    cases, cold, kern = {}, {}, {}
    worst = {"ep": (0.0, 0.0), "tp": (0.0, 0.0)}
    worst_at = {"ep": [None, None], "tp": [None, None]}
    for n in all_counts:
        case = _v41_case("uniform", n, 0, V41_E, g1, gen)
        _, windows, slices, _ = _ref_v41(case[3], case[4], case[5], ref_l)
        exps = {}
        for lay in layouts:
            name, off, nloc = lay[:3]
            if n not in counts[nloc]:
                continue
            exp = (windows if nloc < V41_E else slices)[int(name[2:])].to(torch.bfloat16)
            y = run(lay, case)
            _assert_moe_close(y, exp)
            dv = _dev(y, exp)
            for i in range(2):
                if dv[i] > worst[name[:2]][i]:
                    worst_at[name[:2]][i] = (name, n)
            worst[name[:2]] = tuple(max(a, b) for a, b in zip(worst[name[:2]], dv))
            cold[(name, n)] = y
            exps[name] = exp
            if name in ("ep0", "tp0"):
                kern[(name, n)] = _kernel_names(lambda: run(lay, case))
        cases[n] = (case, exps)
    assert not _moe_cache_entries(), "an ordinary call wrote a profiling-cache entry"

    def runs(name):
        """Consecutive row counts launching the same kernels, as (first, last) ranges."""
        out = []
        for n in counts[V41_EP if name == "ep0" else V41_E]:
            if out and kern[(name, n)] == out[-1][2]:
                out[-1][1] = n
            else:
                out.append([n, n, kern[(name, n)]])
        return [(a, b) for a, b, _ in out]

    cold_runs = {name: runs(name) for name in ("ep0", "tp0")}

    with autotune():
        for lay in layouts:
            if lay[0] in ("ep0", "tp0"):
                c = _v41_case("uniform", 8192, lay[1], lay[2], g1, gen)
                run(lay, c)
    uids = sorted({str(k[2]) for k in _moe_cache_entries()})
    assert uids == ["(6, 2304, 96, 0)", "(6, 576, 384, 0)"], uids
    same = total = 0
    worst_warm = (0.0, 0.0)
    for n in all_counts:
        case, exps = cases[n]
        for lay in layouts:
            name = lay[0]
            if name not in exps:
                continue
            y = run(lay, case)
            _assert_moe_close(y, exps[name])
            worst_warm = tuple(max(a, b) for a, b in zip(worst_warm, _dev(y, exps[name])))
            same += int(torch.equal(y, cold[(name, n)]))
            total += 1
            if name in ("ep0", "tp0"):
                kern[(name, n)] = _kernel_names(lambda: run(lay, case))
    warm_runs = {name: runs(name) for name in ("ep0", "tp0")}
    AutoTuner.get().clear_cache()
    print(
        f"  test_v41_row_axis_by_bucket OK (EP4 windows at {len(counts[V41_EP])} row counts, TP4 ranks at "
        f"{len(counts[V41_E])}, 1..8192; cold worst EP {worst['ep'][0]:.2f} / {worst['ep'][1]:.2f}, TP "
        f"{worst['tp'][0]:.2f} / {worst['tp'][1]:.2f} elt / rms ulp (at {worst_at}); warm worst {worst_warm[0]:.2f} / "
        f"{worst_warm[1]:.2f}; warm bitwise equal to cold on {same} of {total}; kernel runs cold {cold_runs}; "
        f"warm {warm_runs})"
    )


# ── dep4's zero-padding skew, eager, at every chunk size ───────────────────
# dep4 pads every rank's rows to the step's largest per-rank count with zero
# rows before the all-gather (`_moe` in the unified target), so the gathered
# batch is four equal blocks, rank r's real rows first and its padding after,
# and `_moe_chunks` cuts it into calls of 8192 rows and one remainder: every
# eager row count is a multiple of 4 in 4..8192. A zero row has zero router
# logits, so the router sends every one of them to the six bias-top experts
# with weight 1.5 / 6 = 0.25, and `fp4_quantize` turns it into zero codes and
# zero scale bytes. A chunk can therefore hold up to 8191 padding rows (a rank
# with one real token behind a full rank) or none but padding (the tail of a
# short rank's block), all on six experts, which in one window can be all six.

_DEP4_SIX_SPREAD = (5, 17, 101, 200, 290, 377)  # 2 / 1 / 1 / 2 of the six per EP4 window

# Every row count the dep4 accuracy-gate preview dumped for the EP4 windows
# (spool/logs/acc-preview-dep4.log, lines of exactly one 5-tuple; garbled lines
# dropped). All are multiples of 4.
_DEP4_DUMPED_TOKENS = (
    4,
    8,
    16,
    32,
    64,
    96,
    128,
    236,
    256,
    348,
    356,
    396,
    468,
    512,
    544,
    568,
    580,
    1256,
    1580,
    1968,
    2016,
    2320,
    2344,
    2408,
    2440,
    2500,
    2512,
    2520,
    2528,
    2644,
    2648,
    2652,
    2664,
    2696,
    2700,
    2708,
    2716,
    2720,
    2732,
    2736,
    2740,
    2748,
    2768,
    2772,
    2792,
    2812,
    2820,
    2824,
    2828,
    2860,
    2864,
    2876,
    2884,
    2888,
    2900,
    2908,
    2928,
    2932,
    2940,
    2968,
    2976,
    3000,
    3004,
    3024,
    3028,
    3052,
    3056,
    3072,
    3076,
    3080,
    3088,
    3092,
    3104,
    3112,
    3116,
    3120,
    3132,
    3136,
    3140,
    3144,
    3148,
    3152,
    3156,
    3160,
    3164,
    3168,
    3176,
    3180,
    3196,
    3204,
    3216,
    3224,
    3228,
    3232,
    3236,
    3240,
    3244,
    3248,
    3260,
    3264,
    3268,
    3276,
    3296,
    3304,
    3308,
    3316,
    3320,
    3328,
    3340,
    3348,
    3352,
    3368,
    3372,
    3376,
    3380,
    3384,
    3392,
    3400,
    3404,
    3412,
    3416,
    3420,
    3424,
    3428,
    3432,
    3436,
    3440,
    3444,
    3448,
    3456,
    3460,
    3464,
    3468,
    3476,
    3488,
    3496,
    3500,
    3504,
    3512,
    3516,
    3524,
    3528,
    3532,
    3536,
    3540,
    3544,
    3548,
    3552,
    3556,
    3560,
    3564,
    3568,
    3572,
    3580,
    3592,
    3596,
    3600,
    3604,
    3608,
    3616,
    3620,
    3624,
    3628,
    3632,
    3636,
    3640,
    3644,
    3648,
    3652,
    3656,
    3660,
    3664,
    3668,
    3672,
    3676,
    3680,
    3688,
    3692,
    3696,
    3700,
    3704,
    3708,
    3712,
    3716,
    3724,
    3728,
    3736,
    3740,
    3748,
    3752,
    3756,
    3760,
    3764,
    3768,
    3772,
    3776,
    3780,
    3784,
    3792,
    3796,
    3800,
    3804,
    3808,
    3812,
    3816,
    3820,
    3824,
    3828,
    3832,
    3836,
    3840,
    3844,
    3848,
    3852,
    3856,
    3868,
    3872,
    3880,
    3884,
    3888,
    3892,
    3896,
    3900,
    3904,
    3912,
    3924,
    3928,
    3932,
    3936,
    3940,
    3944,
    3952,
    3956,
    3960,
    3968,
    3976,
    3980,
    3984,
    3988,
    3992,
    3996,
    4004,
    4012,
    4016,
    4020,
    4028,
    4032,
    4044,
    4056,
    4068,
    4072,
    4076,
    4080,
    4084,
    4088,
    4092,
    4100,
    4108,
    4112,
    4116,
    4120,
    4124,
    4132,
    4136,
    4140,
    4144,
    4152,
    4156,
    4168,
    4176,
    4180,
    4184,
    4196,
    4200,
    4204,
    4208,
    4224,
    4228,
    4236,
    4248,
    4252,
    4260,
    4268,
    4276,
    4280,
    4284,
    4296,
    4308,
    4316,
    4324,
    4344,
    4352,
    4356,
    4372,
    4380,
    4384,
    4392,
    4396,
    4404,
    4440,
    4444,
    4448,
    4452,
    4460,
    4464,
    4480,
    4492,
    4496,
    4500,
    4504,
    4520,
    4524,
    4528,
    4540,
    4548,
    4568,
    4584,
    4600,
    4604,
    4612,
    4628,
    4644,
    4664,
    4668,
    4672,
    4680,
    4688,
    4692,
    4696,
    4704,
    4720,
    4724,
    4740,
    4748,
    4752,
    4764,
    4804,
    4808,
    4812,
    4832,
    4836,
    4852,
    4864,
    4916,
    4924,
    4928,
    4940,
    4968,
    4992,
    5028,
    5072,
    5168,
    5216,
    5328,
    5372,
    5496,
    5732,
    5816,
    5876,
    5968,
    5976,
    6008,
    6152,
    6164,
    6168,
    6232,
    6260,
    6272,
    6324,
    6368,
    6384,
    6416,
    6420,
    6456,
    6504,
    6524,
    6540,
    6544,
    6580,
    6624,
    6628,
    6636,
    6668,
    6708,
    6744,
    6764,
    6768,
    6772,
    6792,
    6832,
    6852,
    6860,
    6868,
    6896,
    6924,
    6928,
    6932,
    6940,
    6948,
    6956,
    6960,
    6968,
    6984,
    7000,
    7012,
    7016,
    7056,
    7060,
    7088,
    7092,
    7096,
    7108,
    7120,
    7128,
    7148,
    7152,
    7156,
    7176,
    7184,
    7240,
    7248,
    7284,
    7292,
    7296,
    7316,
    7328,
    7352,
    7360,
    7368,
    7400,
    7420,
    7432,
    7464,
    7468,
    7492,
    7544,
    7576,
    7596,
    7612,
    7648,
    7660,
    7772,
    7776,
    7804,
    7876,
    7908,
    7928,
    7932,
    7936,
    7944,
    7996,
    8008,
    8012,
    8020,
    8056,
    8136,
    8140,
    8152,
    8172,
    8192,
)
# Every row count the batch-256 dep4 engine dumped (graph and eager; 99-logs/cg/cg-b256-dep4).
_DEP4_DUMPED_TOKENS_B256 = (
    4,
    8,
    24,
    28,
    48,
    64,
    96,
    192,
    384,
    576,
    768,
    804,
    844,
    960,
    1152,
    1344,
    1536,
    1728,
    1920,
    2112,
    2304,
    2388,
    2496,
    2688,
    2880,
    3072,
    4608,
    5328,
    6144,
    8168,
    8192,
)


def _dep4_six(w):
    """The six experts every padding row is routed to: all inside EP4 window ``w``,
    or (``w`` None) spread over the four windows."""
    if w is None:
        return _DEP4_SIX_SPREAD
    return tuple(w * V41_EP + e for e in (5, 17, 33, 50, 71, 90))


def _dep4_eager_counts():
    """dep4's eager row counts (multiples of 4) at every tuner bucket's lower edge, one
    interior point and upper edge, and on both sides of every untuned-tile and other
    kernel edge of an EP4 window."""
    out, b = set(), 4
    while b <= 8192:
        vals = list(range(b, min(2 * b - 1, 8192) + 1, 4))
        out |= {vals[0], vals[len(vals) // 2], vals[-1]}
        b *= 2
    for e in _v41_tile_edges(V41_EP) | set(_V41_KERNEL_EDGES):
        out |= {v for v in (4 * (e // 4), 4 * -(-e // 4)) if 4 <= v <= 8192}
    return sorted(out)


def _dep4_pad_mask(n, cpu):
    """Padding rows of one `_moe_chunks` chunk of `n` rows cut from a random dep4 gathered
    batch: rows = (8192 j + n) / 4 per rank (the chunk is the last one; for n = 8192 any
    chunk of 8192 (j + 1) rows), one rank full (the step's largest), the rank whose block
    overlaps the chunk most holding ONE real token, the other two a random count."""
    j = int(torch.randint(0, 4, (1,), generator=cpu))
    if n == 8192:
        rows, start = 2048 * (j + 1), 8192 * int(torch.randint(0, j + 1, (1,), generator=cpu))
    else:
        rows, start = (8192 * j + n) // 4, 8192 * j
    ov = [max(0, min(start + n, (r + 1) * rows) - max(start, r * rows)) for r in range(4)]
    big = max(range(4), key=lambda r: (ov[r], float(torch.rand(1, generator=cpu))))
    rest = [r for r in range(4) if r != big]
    full = min(rest, key=lambda r: (ov[r], float(torch.rand(1, generator=cpu))))
    counts = [int(torch.randint(1, rows + 1, (1,), generator=cpu)) for _ in range(4)]
    counts[full], counts[big] = rows, 1
    pad = torch.zeros(4 * rows, dtype=torch.bool)
    for r, c in enumerate(counts):
        pad[r * rows + c : (r + 1) * rows] = True
    return pad[start : start + n].to(DEV), (rows, counts, start)


def _dep4_case(kind, n, w, g1, seed):
    """One eager call's inputs (deterministic in ``seed``). ``dep4``: a chunk of a padded
    gathered batch (`_dep4_pad_mask`); ``one_real``: one real row then n - 1 padding rows;
    ``all_pad``: n padding rows; ``skew6``: n real rows all routed to the six experts (the
    per-expert count at its maximum, n, with nonzero data). Padding rows are zero codes and
    zero scale bytes, routed to ``_dep4_six(w)`` with weight 0.25 each."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    data, sf, x = _rand_activation(n, V41_H, g1, gen)
    ids, wts = _v41_routing(n, gen)
    six = torch.tensor(_dep4_six(w), device=DEV, dtype=torch.int32)
    layout = None
    if kind == "skew6":
        pad = torch.zeros(n, dtype=torch.bool, device=DEV)
        ids = six.expand(n, V41_K).contiguous()
    elif kind == "dep4":
        pad, layout = _dep4_pad_mask(n, cpu)
    else:
        pad = torch.ones(n, dtype=torch.bool, device=DEV)
        pad[0] = kind == "all_pad"
    ids[pad] = six
    wts[pad] = 0.25
    data[pad] = 0
    sf.view(n, V41_H // SV)[pad] = 0
    x[pad] = 0
    return (kind, data, sf, x, ids, wts), pad, layout


_V41_DQ = None


def _v41_dequant_cache():
    """The V4.1 layer's dequantized weights (code x block scale), cached in bf16 --
    exact: an e2m1 code times a power-of-two e4m3 scale has at most two significant
    bits -- so the eager sweep's reference does not re-dequantize 1152 matrices per call."""
    global _V41_DQ
    if _V41_DQ is None:
        _, ref = _v41()
        _V41_DQ = {
            k: torch.stack([_dequant(c[e], s[e]).to(torch.bfloat16) for e in range(V41_E)])
            for k, (c, s) in ((k, ref[k]) for k in ("up", "gate", "down"))
        }
    return _V41_DQ


def _ref_v41_fast(x, ids, wts, ref, slices=True):
    """`_ref_v41`'s four EP4 window shares and (with `slices`) its four MoE-TP slice
    shares, computed the same way (same per-expert row order, same fp32 ops) from the
    cached dequantized weights; tied bitwise to `_ref_v41` in the tests that use it."""
    dq = _v41_dequant_cache()
    flat = ids.flatten().long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=V41_E).tolist()
    tok_all, slot_all = order // V41_K, order % V41_K
    g2 = ref["g2"].tolist()
    windows = [
        torch.zeros(x.shape[0], V41_H, dtype=torch.float32, device=DEV)
        for _ in range(V41_E // V41_EP)
    ]
    parts = [torch.zeros_like(windows[0]) for _ in range(V41_I // V41_TP_I)] if slices else None
    start = 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        tok, slot = tok_all[start : start + c], slot_all[start : start + c]
        start += c
        xe = x[tok]
        up = xe @ (dq["up"][e].float() / ref["gw1"][e]).t()
        gate = xe @ (dq["gate"][e].float() / ref["gw1"][e]).t()
        gate = torch.clamp(gate, max=V41_LIMIT)
        up = torch.clamp(up, min=-V41_LIMIT, max=V41_LIMIT)
        act = up * gate * torch.sigmoid(gate)
        act = _q_nvfp4(act, g2[e]) / g2[e]
        wd = dq["down"][e].float() / ref["gw2"][e]
        w = wts[tok, slot].float().unsqueeze(1)
        windows[e // V41_EP].index_add_(0, tok, (act @ wd.t()) * w)
        if slices:
            for s in range(len(parts)):
                a, b = s * V41_TP_I, (s + 1) * V41_TP_I
                parts[s].index_add_(0, tok, (act[:, a:b] @ wd[:, a:b].t()) * w)
    return windows, parts


def _ref_v41_windows(x, ids, wts, ref):
    """`_ref_v41`'s four EP4 window shares (`_ref_v41_fast` without the TP slices)."""
    return _ref_v41_fast(x, ids, wts, ref, slices=False)[0]


def test_v41_dep4_padding_skew_eager_by_bucket():
    """dep4's zero-padding skew on the EP4 windows, eager, at every chunk size the target passes.

    The target's call (pre-routed, clamp 10, layer-level g2, `output=`) on each of the four
    96-local windows (I = 2304). Row counts: every autotuner bucket's lower edge, an interior
    point and upper edge in dep4's multiples of 4, and both sides of every untuned-tile and
    other kernel edge (`_dep4_eager_counts`). At each count, each run on all four windows:
    a random dep4 chunk (`_dep4_pad_mask`) with the six padding experts spread over the
    windows and, for every window, with all six inside it; one real row behind n - 1
    padding rows; n padding rows; and, for every window, n real rows all on its six experts
    (up to 8192 rows per expert). Each call is held to the torch reference; every row the
    reference leaves at zero (padding rows, rows with no slot in the window) must be
    exactly zero. Run cold (the untuned tile rule), then with the cache warmed by
    `autotune()` at 8192 rows (one tuned tactic per bucket, the engine's state): every
    case again, plus a dep4 chunk at every row count the dep4 preview and the
    batch-256 dep4 engine dumped. The warm
    calls write no cache entry (the tactic stays the bucket's), and the kernels launched
    at a row count are the same for uniform, skewed and all-padding routing, cold and warm.
    Control: a reference that loses the last quarter of one expert's rows (a kernel whose
    per-expert capacity assumed balanced routing) fails the gate. Clears the cache at the end.
    """
    assert not _moe_cache_entries(), "the profiling cache already holds entries for this op"
    args, ref_l = _v41_layer_g2()
    layouts = [lay for lay in _v41_graph_layouts(args, ref_l) if lay[2] == V41_EP]
    assert [lay[1] for lay in layouts] == [0, 96, 192, 288]
    g1 = ref_l["g1"]
    counts = _dep4_eager_counts()
    assert {4, 8188, 8192, 140, 144, 2060, 2064} <= set(counts) and all(n % 4 == 0 for n in counts)

    # The fast reference is the certified one, bit for bit.
    for kind, n, w, seed in (
        ("dep4", 1024, 1, 4150),
        ("skew6", 256, 2, 4151),
        ("dep4", 8192, None, 4152),
    ):
        case, _, _ = _dep4_case(kind, n, w, g1, seed)
        fast = _ref_v41_windows(case[3], case[4], case[5], ref_l)
        slow = _ref_v41(case[3], case[4], case[5], ref_l)[1]
        for r in range(4):
            assert torch.equal(fast[r], slow[r]), (kind, n, r)

    def plan(n):
        """(kind, window of the six, seed) of every case at row count n."""
        s = 1_000_003 * n
        cases = [("dep4", None, s)] + [("dep4", w, s + 1 + w) for w in range(4)]
        cases += [("one_real", (n // 4) % 4, s + 5), ("all_pad", (n // 4 + 1) % 4, s + 6)]
        return cases + [("skew6", w, s + 7 + w) for w in range(4)]

    def run(lay, case):
        _, off, nloc, la, ls = lay
        _, data, sf, _, ids, wts = case
        out = torch.full((data.shape[0], V41_H), float("nan"), dtype=torch.bfloat16, device=DEV)
        _call(
            data,
            sf,
            la,
            V41_E,
            V41_K,
            local_expert_offset=off,
            local_num_experts=nloc,
            topk_ids=ids,
            topk_weights=wts,
            output=out,
            **ls,
        )
        return out

    stats = {
        "calls": 0,
        "zero_rows": 0,
        "pad_rows": 0,
        "max_expert_rows": 0,
        "worst": (0.0, 0.0),
        "worst_skew": (0.0, 0.0),
        "worst_pad": (0.0, 0.0),
        "worst_dump": (0.0, 0.0),
    }
    kern = {}

    def check(tag, n, kind, w, seed, profile_it):
        case, pad, _ = _dep4_case(kind, n, w, g1, seed)
        refs = _ref_v41_windows(case[3], case[4], case[5], ref_l)
        per_expert = torch.bincount(case[4].flatten().long(), minlength=V41_E)
        stats["max_expert_rows"] = max(stats["max_expert_rows"], int(per_expert.max()))
        stats["pad_rows"] += int(pad.sum())
        for lay, rf in zip(layouts, refs):
            name = lay[0]
            y = run(lay, case)
            exp = rf.to(torch.bfloat16)
            zero = (exp == 0).all(dim=1)
            assert pad.logical_not().logical_or(zero).all(), (
                f"{tag} {name} T={n} {kind}: a padding row is nonzero"
            )
            assert (y[zero] == 0).all(), (
                f"{tag} {name} T={n} {kind} six@{w}: a row the reference leaves zero is not"
            )
            stats["zero_rows"] += int(zero.sum())
            if not zero.all():
                _assert_moe_close(y, exp)
                dv = _dev(y[~zero], exp[~zero])
                keys = ["worst", "worst_skew" if kind == "skew6" else "worst_pad"] + (
                    ["worst_dump"] if tag == "dump" else []
                )
                for key in keys:
                    stats[key] = tuple(max(a, b) for a, b in zip(stats[key], dv))
            stats["calls"] += 1
            if profile_it and name == f"ep{w}":
                kern[(tag, n, kind)] = _kernel_names(lambda: run(lay, case))
        return case

    uniform_kernels = {}
    for tag in ("cold", "warm"):
        if tag == "warm":
            with autotune():
                _, off, nloc, la, ls = layouts[0]
                c = _v41_case(
                    "uniform", 8192, off, nloc, g1, torch.Generator(device=DEV).manual_seed(4153)
                )
                _call(
                    c[1],
                    c[2],
                    la,
                    V41_E,
                    V41_K,
                    local_expert_offset=off,
                    local_num_experts=nloc,
                    topk_ids=c[4],
                    topk_weights=c[5],
                    **ls,
                )
            uids = sorted({str(k[2]) for k in _moe_cache_entries()})
            assert uids == ["(6, 2304, 96, 0)"], uids
            tuned = len(_moe_cache_entries())
        for n in counts:
            u = _v41_case(
                "uniform", n, 0, V41_EP, g1, torch.Generator(device=DEV).manual_seed(4154 + n)
            )
            uniform_kernels[(tag, n)] = _kernel_names(lambda: run(layouts[0], u))
            for kind, w, seed in plan(n):
                check(
                    tag,
                    n,
                    kind,
                    w,
                    seed,
                    profile_it=kind == "all_pad" or (kind == "skew6" and w == 0),
                )
            for kind in ("skew6", "all_pad"):
                if (tag, n, kind) in kern:
                    assert kern[(tag, n, kind)] == uniform_kernels[(tag, n)], (
                        f"{tag} T={n} {kind}: the routing changed the launched kernels"
                    )
        if tag == "warm":
            for i, n in enumerate(sorted(set(_DEP4_DUMPED_TOKENS) | set(_DEP4_DUMPED_TOKENS_B256))):
                check("dump", n, "dep4", None if i % 5 == 4 else i % 4, 2_000_003 * n, False)
            assert len(_moe_cache_entries()) == tuned, "a skewed call re-tuned a bucket"
        assert (tag == "warm") == bool(_moe_cache_entries()), (
            "an ordinary call wrote a profiling-cache entry"
        )

    # Control: a kernel whose per-expert capacity assumed balanced routing would lose rows of
    # the overloaded experts; a reference missing the last quarter of one expert's rows is
    # what it would return, and the gate must reject it.
    ctrl = []
    for n in (256, 2048, 8192):
        case, _, _ = _dep4_case("skew6", n, 0, g1, 4155 + n)
        y = run(layouts[0], case)
        lost = case[5].clone()
        lost[3 * n // 4 :, 5] = 0  # those rows' slot on the sixth expert contributes nothing
        bad = _ref_v41_windows(case[3], case[4], lost, ref_l)[0].to(torch.bfloat16)
        ctrl.append((n, *(round(v, 1) for v in _dev(y, bad))))
        try:
            _assert_moe_close(y, bad)
        except AssertionError:
            continue
        raise AssertionError(
            f"T={n}: the gate accepts a result missing a quarter of one expert's rows"
        )
    AutoTuner.get().clear_cache()
    global _V41_DQ
    _V41_DQ = None
    torch.cuda.empty_cache()
    print(
        f"  test_v41_dep4_padding_skew_eager_by_bucket OK ({len(counts)} dep4 row counts {counts[:6]}..{counts[-3:]} "
        f"x 11 cases x 4 windows, cold and warm ({tuned} tuned entries), plus "
        f"{len(set(_DEP4_DUMPED_TOKENS) | set(_DEP4_DUMPED_TOKENS_B256))} dumped counts "
        f"warm: {stats['calls']} calls, {stats['pad_rows']} padding rows, max rows on one expert "
        f"{stats['max_expert_rows']}, {stats['zero_rows']} rows exactly zero; worst elt / rms ulp all "
        f"{stats['worst'][0]:.2f} / {stats['worst'][1]:.2f}, all-on-six {stats['worst_skew'][0]:.2f} / "
        f"{stats['worst_skew'][1]:.2f}, padded chunks {stats['worst_pad'][0]:.2f} / {stats['worst_pad'][1]:.2f}, "
        f"dumped {stats['worst_dump'][0]:.2f} / {stats['worst_dump'][1]:.2f}; kernels at a row count are "
        f"routing-independent at all {len(counts)} counts cold and warm; lost-rows control (T, elt, rms) {ctrl})"
    )


# ── V4.1 rank forms of the tp 2 / 8 members: EP2, EP8 windows, MoE-TP2 ranks ─
# ("E6" in the names below). Besides tep4 / dep4 / tp4 (above), the unified
# target admits tep8 and dep8
# (tp 8 over two trays) and, with two pipeline stages, pp2 x {tep2, dep2, tp2,
# tep4, dep4, tp4} (`_parallel` in `models/deepseek_v41/routing.py`). A pipeline
# stage runs its layers' MoE exactly as the same member without stages does, so
# the rank forms this op sees that the sections above do not cover are:
# - EP2: 192 local experts at offsets 0 / 192, I = 2304 (tep2, dep2);
# - EP8: 48 local experts at offsets 0, 48, ..., 336, I = 2304 (tep8, dep8);
# - MoE-TP2: all 384 experts, I = 1152, rank r holding intermediate columns
#   [1152r, 1152r + 1152) of every expert (tp2).
# Under capture every rank routes G and 6G of its own rows over `_GSET`; dep2's
# EP2 windows route the gathered 2G and 12G (to 3072) and dep8's EP8 windows the
# gathered 8G and 48G (to 12288). The target cuts a routed batch into calls of
# at most 8192 rows (`_moe_chunks`), so dep8's 48G at G 192 and 256 (9216 and
# 12288 rows) is two calls captured in one graph: 8192 + 1024 and 8192 + 4096.
# Eager, every call is at most 8192 rows, as for the members above; dep2 and
# dep8 pad and gather like dep4, so their eager chunks are multiples of 2 and 8.
#
# Kernel choice, profiled at every T 1..8192 on EP2 window 0, EP8 window 0 and
# TP2 rank 0, cold and after `autotune()` (a probe; the tests below profile their
# sweep points and assert the same edges): the launched kernels change only at
# the routing / finalize edges of the forms above (4/5, 59/60, 64/65, 128/129),
# the untuned token tile's edges (the C++ runner's rule, `_v41_cold_tile`) or,
# warm, the tuner's power-of-two bucket edges, plus 1/2 on an untuned TP2 rank
# (its FC1 config; 2/3 on a TP4 rank).

_E6 = {
    "ep2": (192, V41_I, 2),
    "ep8": (48, V41_I, 8),
    "tp2": (V41_E, 1152, 2),
}  # local experts, I, ranks
_MOE_MAX_T = 8192
_E6_GATHERED = {
    "ep2": tuple(sorted({2 * g for g in _GSET} | {12 * g for g in _GSET})),  # dep2: 2G, 12G
    "ep8": tuple(sorted({8 * g for g in _GSET} | {48 * g for g in _GSET})),  # dep8: 8G, 48G
    "tp2": (),  # a MoE-TP rank is never gathered
}
_E6_UIDS = {"ep2": "(6, 2304, 192, 0)", "ep8": "(6, 2304, 48, 0)", "tp2": "(6, 1152, 384, 0)"}
# Every runner row count the E6 members' engine dumps show (tep8 boot and GSM8K, dep8 boot and
# GSM8K, pp2 x tp2 and pp2 x tp4 GSM8K, per rank, lines of exactly one well-formed
# tuple): the EP8 windows' 1140 counts, the TP2 ranks' 631 and the TP4 ranks' 646,
# 1437 distinct, 1..8192.
_E6_DUMPED_TOKENS = (
    1,
    2,
    6,
    7,
    8,
    12,
    16,
    22,
    23,
    24,
    32,
    38,
    40,
    48,
    54,
    56,
    64,
    80,
    96,
    104,
    120,
    128,
    129,
    130,
    140,
    144,
    152,
    160,
    192,
    232,
    240,
    248,
    304,
    308,
    312,
    314,
    336,
    344,
    384,
    400,
    408,
    416,
    424,
    432,
    436,
    464,
    472,
    496,
    504,
    512,
    546,
    552,
    558,
    600,
    616,
    624,
    640,
    680,
    735,
    767,
    768,
    772,
    776,
    783,
    785,
    795,
    798,
    802,
    805,
    807,
    808,
    809,
    812,
    815,
    816,
    817,
    818,
    820,
    821,
    823,
    826,
    829,
    830,
    832,
    835,
    836,
    838,
    839,
    840,
    842,
    847,
    848,
    849,
    851,
    852,
    853,
    854,
    856,
    860,
    861,
    862,
    865,
    866,
    867,
    869,
    870,
    871,
    872,
    873,
    874,
    875,
    876,
    877,
    878,
    880,
    882,
    883,
    884,
    885,
    886,
    887,
    888,
    889,
    890,
    892,
    893,
    896,
    897,
    898,
    899,
    900,
    904,
    905,
    906,
    907,
    908,
    909,
    910,
    911,
    912,
    913,
    914,
    915,
    916,
    917,
    919,
    920,
    921,
    922,
    923,
    924,
    925,
    926,
    927,
    928,
    929,
    931,
    932,
    933,
    934,
    935,
    937,
    938,
    939,
    941,
    942,
    943,
    944,
    945,
    946,
    947,
    948,
    949,
    950,
    951,
    952,
    953,
    954,
    955,
    956,
    957,
    958,
    959,
    960,
    961,
    962,
    963,
    964,
    965,
    966,
    967,
    968,
    969,
    970,
    971,
    972,
    973,
    974,
    975,
    976,
    977,
    978,
    979,
    980,
    981,
    982,
    983,
    984,
    985,
    986,
    987,
    988,
    989,
    990,
    991,
    992,
    993,
    994,
    995,
    996,
    997,
    998,
    1000,
    1001,
    1002,
    1003,
    1004,
    1005,
    1006,
    1008,
    1009,
    1010,
    1011,
    1012,
    1013,
    1014,
    1015,
    1016,
    1017,
    1018,
    1019,
    1020,
    1021,
    1022,
    1023,
    1024,
    1025,
    1026,
    1027,
    1028,
    1030,
    1031,
    1032,
    1033,
    1034,
    1035,
    1036,
    1037,
    1038,
    1039,
    1040,
    1041,
    1042,
    1043,
    1044,
    1045,
    1046,
    1047,
    1048,
    1049,
    1050,
    1052,
    1053,
    1054,
    1055,
    1056,
    1057,
    1058,
    1059,
    1060,
    1061,
    1062,
    1063,
    1064,
    1065,
    1066,
    1067,
    1068,
    1069,
    1070,
    1071,
    1072,
    1073,
    1074,
    1075,
    1076,
    1077,
    1078,
    1079,
    1080,
    1081,
    1082,
    1083,
    1084,
    1085,
    1086,
    1087,
    1088,
    1089,
    1090,
    1091,
    1092,
    1093,
    1094,
    1095,
    1096,
    1097,
    1098,
    1099,
    1100,
    1101,
    1102,
    1103,
    1104,
    1105,
    1106,
    1107,
    1108,
    1109,
    1110,
    1111,
    1112,
    1113,
    1114,
    1115,
    1116,
    1117,
    1118,
    1119,
    1120,
    1121,
    1122,
    1124,
    1125,
    1126,
    1127,
    1128,
    1129,
    1130,
    1131,
    1132,
    1133,
    1135,
    1136,
    1137,
    1138,
    1139,
    1140,
    1141,
    1142,
    1143,
    1144,
    1145,
    1146,
    1147,
    1148,
    1149,
    1150,
    1151,
    1152,
    1153,
    1154,
    1155,
    1156,
    1158,
    1159,
    1160,
    1161,
    1163,
    1165,
    1166,
    1167,
    1168,
    1169,
    1170,
    1172,
    1173,
    1174,
    1175,
    1176,
    1177,
    1178,
    1179,
    1180,
    1182,
    1183,
    1184,
    1185,
    1186,
    1187,
    1189,
    1190,
    1191,
    1193,
    1194,
    1197,
    1198,
    1199,
    1200,
    1201,
    1202,
    1204,
    1205,
    1206,
    1207,
    1209,
    1210,
    1211,
    1212,
    1214,
    1217,
    1220,
    1222,
    1224,
    1225,
    1226,
    1228,
    1229,
    1230,
    1231,
    1232,
    1234,
    1235,
    1236,
    1241,
    1243,
    1244,
    1248,
    1250,
    1251,
    1252,
    1253,
    1254,
    1256,
    1258,
    1260,
    1262,
    1265,
    1266,
    1267,
    1268,
    1270,
    1271,
    1272,
    1273,
    1275,
    1276,
    1277,
    1278,
    1279,
    1280,
    1281,
    1283,
    1284,
    1285,
    1286,
    1287,
    1288,
    1290,
    1292,
    1295,
    1297,
    1298,
    1301,
    1304,
    1305,
    1306,
    1308,
    1312,
    1314,
    1315,
    1316,
    1322,
    1323,
    1325,
    1326,
    1327,
    1329,
    1331,
    1332,
    1335,
    1336,
    1340,
    1356,
    1357,
    1358,
    1360,
    1363,
    1364,
    1368,
    1384,
    1386,
    1387,
    1388,
    1390,
    1397,
    1400,
    1403,
    1421,
    1422,
    1423,
    1432,
    1447,
    1455,
    1459,
    1465,
    1481,
    1484,
    1493,
    1506,
    1510,
    1516,
    1528,
    1530,
    1536,
    1552,
    1554,
    1556,
    1560,
    1568,
    1577,
    1584,
    1589,
    1592,
    1599,
    1601,
    1603,
    1604,
    1607,
    1609,
    1613,
    1616,
    1619,
    1620,
    1622,
    1627,
    1628,
    1630,
    1632,
    1634,
    1638,
    1639,
    1640,
    1642,
    1644,
    1646,
    1651,
    1652,
    1653,
    1656,
    1657,
    1661,
    1663,
    1665,
    1667,
    1670,
    1671,
    1672,
    1673,
    1675,
    1677,
    1679,
    1680,
    1686,
    1687,
    1688,
    1691,
    1692,
    1693,
    1695,
    1697,
    1703,
    1704,
    1705,
    1706,
    1707,
    1708,
    1710,
    1711,
    1713,
    1715,
    1716,
    1717,
    1718,
    1719,
    1720,
    1722,
    1723,
    1724,
    1726,
    1727,
    1728,
    1729,
    1730,
    1731,
    1740,
    1741,
    1742,
    1744,
    1746,
    1747,
    1748,
    1749,
    1750,
    1751,
    1752,
    1753,
    1754,
    1755,
    1757,
    1759,
    1760,
    1761,
    1762,
    1764,
    1765,
    1766,
    1767,
    1768,
    1769,
    1770,
    1771,
    1772,
    1774,
    1775,
    1776,
    1777,
    1779,
    1781,
    1783,
    1784,
    1786,
    1788,
    1790,
    1793,
    1794,
    1795,
    1796,
    1798,
    1799,
    1800,
    1802,
    1803,
    1804,
    1805,
    1806,
    1807,
    1808,
    1810,
    1811,
    1812,
    1813,
    1814,
    1816,
    1817,
    1818,
    1820,
    1821,
    1822,
    1823,
    1824,
    1825,
    1826,
    1827,
    1828,
    1829,
    1830,
    1831,
    1832,
    1833,
    1835,
    1836,
    1837,
    1839,
    1840,
    1841,
    1842,
    1843,
    1844,
    1845,
    1846,
    1847,
    1848,
    1850,
    1851,
    1852,
    1853,
    1854,
    1855,
    1860,
    1862,
    1863,
    1864,
    1865,
    1866,
    1867,
    1868,
    1869,
    1870,
    1871,
    1872,
    1873,
    1874,
    1876,
    1877,
    1878,
    1879,
    1880,
    1882,
    1883,
    1884,
    1885,
    1886,
    1889,
    1890,
    1891,
    1892,
    1893,
    1894,
    1896,
    1897,
    1899,
    1902,
    1904,
    1905,
    1906,
    1907,
    1908,
    1910,
    1911,
    1914,
    1915,
    1916,
    1917,
    1918,
    1919,
    1920,
    1922,
    1923,
    1924,
    1925,
    1926,
    1927,
    1929,
    1931,
    1932,
    1933,
    1934,
    1936,
    1937,
    1938,
    1939,
    1940,
    1942,
    1943,
    1944,
    1945,
    1946,
    1948,
    1949,
    1950,
    1951,
    1952,
    1953,
    1954,
    1955,
    1956,
    1958,
    1960,
    1961,
    1962,
    1964,
    1965,
    1966,
    1969,
    1970,
    1971,
    1972,
    1974,
    1975,
    1976,
    1977,
    1978,
    1979,
    1980,
    1981,
    1982,
    1983,
    1984,
    1985,
    1986,
    1987,
    1988,
    1989,
    1990,
    1992,
    1993,
    1994,
    1996,
    1997,
    1998,
    1999,
    2000,
    2001,
    2002,
    2004,
    2005,
    2007,
    2008,
    2009,
    2010,
    2012,
    2013,
    2014,
    2017,
    2019,
    2020,
    2021,
    2024,
    2025,
    2026,
    2027,
    2031,
    2032,
    2035,
    2036,
    2037,
    2038,
    2039,
    2040,
    2041,
    2042,
    2043,
    2045,
    2047,
    2048,
    2049,
    2051,
    2052,
    2053,
    2054,
    2055,
    2056,
    2057,
    2059,
    2061,
    2062,
    2063,
    2064,
    2069,
    2070,
    2071,
    2072,
    2075,
    2077,
    2081,
    2082,
    2088,
    2089,
    2090,
    2091,
    2093,
    2095,
    2098,
    2099,
    2100,
    2101,
    2102,
    2104,
    2105,
    2106,
    2107,
    2108,
    2111,
    2114,
    2115,
    2118,
    2119,
    2120,
    2121,
    2123,
    2125,
    2130,
    2131,
    2132,
    2136,
    2137,
    2139,
    2140,
    2146,
    2148,
    2149,
    2152,
    2154,
    2155,
    2156,
    2159,
    2160,
    2162,
    2163,
    2164,
    2166,
    2167,
    2168,
    2169,
    2173,
    2175,
    2177,
    2178,
    2180,
    2184,
    2187,
    2188,
    2189,
    2190,
    2191,
    2193,
    2195,
    2198,
    2203,
    2206,
    2209,
    2213,
    2216,
    2217,
    2219,
    2222,
    2224,
    2230,
    2231,
    2234,
    2239,
    2242,
    2246,
    2247,
    2252,
    2255,
    2257,
    2259,
    2267,
    2272,
    2273,
    2280,
    2282,
    2284,
    2291,
    2292,
    2298,
    2301,
    2312,
    2313,
    2320,
    2322,
    2335,
    2336,
    2344,
    2348,
    2349,
    2351,
    2354,
    2356,
    2359,
    2362,
    2371,
    2373,
    2377,
    2380,
    2382,
    2388,
    2390,
    2395,
    2410,
    2413,
    2416,
    2424,
    2427,
    2432,
    2440,
    2444,
    2458,
    2460,
    2463,
    2471,
    2480,
    2483,
    2487,
    2490,
    2492,
    2495,
    2498,
    2501,
    2506,
    2507,
    2509,
    2513,
    2516,
    2520,
    2527,
    2528,
    2529,
    2538,
    2540,
    2541,
    2543,
    2544,
    2545,
    2549,
    2553,
    2560,
    2562,
    2566,
    2568,
    2576,
    2578,
    2579,
    2581,
    2584,
    2586,
    2589,
    2595,
    2597,
    2598,
    2601,
    2602,
    2603,
    2604,
    2607,
    2611,
    2613,
    2615,
    2616,
    2620,
    2621,
    2623,
    2626,
    2629,
    2630,
    2641,
    2642,
    2644,
    2646,
    2653,
    2654,
    2656,
    2659,
    2660,
    2662,
    2665,
    2670,
    2673,
    2676,
    2679,
    2686,
    2688,
    2689,
    2691,
    2694,
    2695,
    2697,
    2698,
    2699,
    2701,
    2703,
    2704,
    2715,
    2716,
    2720,
    2724,
    2725,
    2726,
    2727,
    2730,
    2734,
    2736,
    2737,
    2738,
    2743,
    2745,
    2749,
    2751,
    2755,
    2758,
    2762,
    2763,
    2769,
    2770,
    2771,
    2773,
    2774,
    2784,
    2785,
    2789,
    2790,
    2792,
    2795,
    2796,
    2797,
    2799,
    2800,
    2803,
    2805,
    2806,
    2808,
    2811,
    2813,
    2814,
    2818,
    2824,
    2826,
    2838,
    2840,
    2845,
    2846,
    2847,
    2851,
    2852,
    2855,
    2858,
    2861,
    2863,
    2865,
    2866,
    2871,
    2872,
    2875,
    2879,
    2880,
    2884,
    2887,
    2888,
    2891,
    2893,
    2894,
    2900,
    2906,
    2908,
    2913,
    2917,
    2918,
    2925,
    2931,
    2937,
    2938,
    2943,
    2948,
    2951,
    2954,
    2958,
    2962,
    2963,
    2966,
    2967,
    2974,
    2976,
    2977,
    2982,
    2987,
    2988,
    2990,
    3015,
    3017,
    3022,
    3024,
    3025,
    3032,
    3036,
    3040,
    3041,
    3043,
    3044,
    3048,
    3055,
    3060,
    3064,
    3067,
    3072,
    3084,
    3086,
    3092,
    3094,
    3100,
    3102,
    3116,
    3117,
    3120,
    3122,
    3130,
    3140,
    3142,
    3143,
    3161,
    3181,
    3182,
    3195,
    3205,
    3214,
    3221,
    3242,
    3257,
    3271,
    3278,
    3280,
    3332,
    3333,
    3342,
    3352,
    3353,
    3358,
    3383,
    3384,
    3408,
    3409,
    3422,
    3428,
    3449,
    3451,
    3458,
    3491,
    3494,
    3499,
    3502,
    3510,
    3528,
    3537,
    3542,
    3543,
    3560,
    3569,
    3581,
    3591,
    3613,
    3616,
    3638,
    3648,
    3656,
    3664,
    3671,
    3672,
    3683,
    3695,
    3704,
    3706,
    3717,
    3719,
    3723,
    3725,
    3736,
    3814,
    3820,
    3826,
    3827,
    3834,
    3840,
    3858,
    3862,
    3893,
    3920,
    3928,
    3947,
    3983,
    3986,
    3990,
    4017,
    4046,
    4057,
    4093,
    4128,
    4169,
    4178,
    4185,
    4222,
    4239,
    4246,
    4261,
    4276,
    4285,
    4288,
    4298,
    4299,
    4378,
    4386,
    4500,
    4526,
    4581,
    4587,
    4589,
    4664,
    4671,
    4752,
    4788,
    4792,
    4825,
    4920,
    4952,
    4968,
    5040,
    5192,
    5232,
    5248,
    5272,
    5312,
    5360,
    5378,
    5384,
    5440,
    5512,
    5600,
    5648,
    5656,
    5696,
    5704,
    5744,
    5771,
    5784,
    5800,
    5912,
    5928,
    6032,
    6048,
    6056,
    6104,
    6112,
    6144,
    6152,
    6216,
    6240,
    6320,
    6368,
    6384,
    6392,
    6432,
    6456,
    6464,
    6472,
    6488,
    6504,
    6520,
    6528,
    6552,
    6560,
    6568,
    6608,
    6640,
    6648,
    6704,
    6744,
    6768,
    6800,
    6808,
    6824,
    6864,
    6880,
    6896,
    6904,
    6928,
    6936,
    6976,
    7000,
    7016,
    7040,
    7048,
    7064,
    7088,
    7120,
    7128,
    7160,
    7176,
    7192,
    7200,
    7248,
    7288,
    7304,
    7328,
    7344,
    7416,
    7440,
    7456,
    7512,
    7568,
    7576,
    7592,
    7632,
    7640,
    7648,
    7664,
    7672,
    7680,
    7696,
    7704,
    7712,
    7760,
    7824,
    7848,
    7872,
    7896,
    7904,
    7912,
    7944,
    7952,
    7968,
    7992,
    8000,
    8016,
    8024,
    8040,
    8048,
    8080,
    8088,
    8098,
    8104,
    8112,
    8120,
    8127,
    8128,
    8136,
    8144,
    8154,
    8158,
    8159,
    8165,
    8168,
    8176,
    8177,
    8179,
    8181,
    8184,
    8192,
)


def _moe_chunks(total: int):
    """Row ranges `[a, b)` of the target's cut of a routed batch into calls of at
    most 8192 rows (`_moe_chunks` in the unified target's `modeling.py`)."""
    out, a = [], 0
    while a < total:
        b = min(total, a + _MOE_MAX_T)
        out.append((a, b))
        a = b
    return out


def _e6_counts(fam):
    """Captured row counts of one E6 rank form: G and 6G, plus the dep member's
    gathered counts on an EP window."""
    return tuple(sorted(set(_V41_RANK_TOKENS) | set(_E6_GATHERED[fam])))


def _e6_layouts(fam, args, ref_l, ranks=None):
    """(name, offset, local experts, operands, scalars) of an E6 family's ranks."""
    nloc, inter, n_ranks = _E6[fam]
    scal = _v41_scalars(ref_l)
    ranks = range(n_ranks) if ranks is None else ranks
    if fam.startswith("ep"):
        return [
            (
                f"{fam}.{r}",
                r * nloc,
                nloc,
                _window_args(args, r * nloc, nloc),
                _v41_slice(scal, r * nloc, nloc),
            )
            for r in ranks
        ]
    tp = _v41_tp_args(inter)
    return [(f"{fam}.{r}", 0, V41_E, tp[r], scal) for r in ranks]


def _e6_part(fam):
    """The `_ref_v41_parts` key of an E6 family's rank shares."""
    nloc, inter, _ = _E6[fam]
    return ("ep", nloc) if fam.startswith("ep") else ("tp", inter)


def _v41_kernel_firsts(nloc: int, inter: int, warm: bool) -> set:
    """First row count of every kernel change of a rank form (`nloc` local experts,
    `inter` intermediate columns) up to 8192, as the eager profile at every T found:
    the routing / finalize edges, then the tuner's bucket edges (warm) or the untuned
    token tile's edges and, on an untuned MoE-TP rank, its FC1 config change (at 3 on a
    TP4 rank, at 2 on a TP2 rank)."""
    firsts = set(_V41_ROUTING_FIRSTS)
    if warm:
        return firsts | {1 << k for k in range(1, 14)}
    firsts |= {m for m in range(2, 8193) if _v41_cold_tile(m, nloc) != _v41_cold_tile(m - 1, nloc)}
    return firsts | {(V41_E, V41_TP_I): {3}, (V41_E, 1152): {2}}.get((nloc, inter), set())


def _e6_segment(n: int, nloc: int, inter: int, warm: bool):
    """(lo, hi): the row counts around `n` that launch `n`'s kernels."""
    firsts = _v41_kernel_firsts(nloc, inter, warm)
    lo = max([1] + [f for f in firsts if f <= n])
    hi = min([8193] + [f for f in firsts if f > n]) - 1
    return lo, hi


def _ref_v41_parts(x, ids, wts, ref, eps=(), tps=()):
    """`_ref_v41_fast` over any rank partitions at once: for each width in `eps` the
    expert-parallel windows of that many local experts (384: the whole layer), for each
    width in `tps` the MoE-TP slices of that many intermediate columns. Same per-expert
    row order and fp32 ops as `_ref_v41`, to which it is tied bitwise at 384 / 96 / 576
    in `test_v41_e6_rank_forms_geometry`. Returns {("ep" | "tp", width): [shares]}."""
    dq = _v41_dequant_cache()
    flat = ids.flatten().long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=V41_E).tolist()
    tok_all, slot_all = order // V41_K, order % V41_K
    g2 = ref["g2"].tolist()

    def zeros(k):
        return [torch.zeros(x.shape[0], V41_H, dtype=torch.float32, device=DEV) for _ in range(k)]

    out = {("ep", w): zeros(V41_E // w) for w in eps}
    out.update({("tp", w): zeros(V41_I // w) for w in tps})
    start = 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        tok, slot = tok_all[start : start + c], slot_all[start : start + c]
        start += c
        xe = x[tok]
        up = xe @ (dq["up"][e].float() / ref["gw1"][e]).t()
        gate = xe @ (dq["gate"][e].float() / ref["gw1"][e]).t()
        gate = torch.clamp(gate, max=V41_LIMIT)
        up = torch.clamp(up, min=-V41_LIMIT, max=V41_LIMIT)
        act = up * gate * torch.sigmoid(gate)
        act = _q_nvfp4(act, g2[e]) / g2[e]
        wd = dq["down"][e].float() / ref["gw2"][e]
        w = wts[tok, slot].float().unsqueeze(1)
        if eps:
            y = (act @ wd.t()) * w
            for ew in eps:
                out[("ep", ew)][e // ew].index_add_(0, tok, y)
        for tw in tps:
            for s in range(V41_I // tw):
                a, b = s * tw, (s + 1) * tw
                out[("tp", tw)][s].index_add_(0, tok, (act[:, a:b] @ wd[:, a:b].t()) * w)
    return out


def _worst(acc, key, dv):
    acc[key] = tuple(max(a, b) for a, b in zip(acc.get(key, (0.0, 0.0)), dv))


def test_v41_e6_rank_forms_geometry():
    """The E6 rank forms at the V4.1 geometry, with per-expert global scales.

    EP2 (two 192-expert windows), EP8 (eight 48-expert windows) and MoE-TP2 (two
    I = 1152 ranks), at every T of `_V41_TOKENS` (1..8192): `num_experts` 384 on
    every call, pre-routed, the clamp at 10 / alpha1, every expert with its own
    gw1, gw2 and g2. Each window matches a reference masking every slot outside it
    and each TP2 rank its share of the reference; tokens with no slot in a window
    come back bitwise zero; the bf16 outputs of a member's ranks summed match the
    full-layer reference. The reference's new partitions are tied to the certified
    one first: at 384 / 96 / 576 `_ref_v41_parts` is `_ref_v41` bit for bit, and its
    EP8 / EP2 windows and TP2 slices sum (fp32) to the EP4 windows and TP4 slices
    within 1e-5 of the row magnitude. Controls: EP8 window 1 issued at offset 0,
    and a TP2 sum without rank 1, fail the gate.
    """
    args, ref = _v41()
    gen = torch.Generator(device=DEV).manual_seed(4160)
    g1 = ref["g1"]
    fams = {fam: _e6_layouts(fam, args, ref) for fam in _E6}
    worst, zero_rows, ctrl, tie_worst = {}, {}, [], 0.0
    for num_tokens in _V41_TOKENS:
        data, sf, x = _rand_activation(num_tokens, V41_H, g1, gen)
        ids, wts = _v41_routing(num_tokens, gen)
        parts = _ref_v41_parts(x, ids, wts, ref, eps=(V41_E, 192, V41_EP, 48), tps=(V41_TP_I, 1152))
        full = parts[("ep", V41_E)][0]
        if num_tokens == 1024:
            slow_full, slow_win, slow_tp, _ = _ref_v41(x, ids, wts, ref)
            assert torch.equal(full, slow_full)
            for r in range(4):
                assert torch.equal(parts[("ep", V41_EP)][r], slow_win[r]), r
                assert torch.equal(parts[("tp", V41_TP_I)][r], slow_tp[r]), r
        row = full.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
        pairs = [(("ep", 48), ("ep", V41_EP), w) for w in range(4)]
        pairs += [(("ep", V41_EP), ("ep", 192), w) for w in range(2)]
        pairs += [(("tp", V41_TP_I), ("tp", 1152), w) for w in range(2)]
        for fine, coarse, w in pairs:
            err = (
                ((parts[fine][2 * w] + parts[fine][2 * w + 1] - parts[coarse][w]).abs() / row)
                .max()
                .item()
            )
            assert err < 1e-5, f"{fine} pair {w} vs {coarse} at T={num_tokens}: {err:.2e}"
            tie_worst = max(tie_worst, err)
        exp_full = full.to(torch.bfloat16)
        for fam, lays in fams.items():
            nloc = _E6[fam][0]
            outs = []
            for r, (name, off, n_loc, la, ls) in enumerate(lays):
                y = _call(
                    data,
                    sf,
                    la,
                    V41_E,
                    V41_K,
                    local_expert_offset=off,
                    local_num_experts=n_loc,
                    topk_ids=ids,
                    topk_weights=wts,
                    **ls,
                )[0]
                exp = parts[_e6_part(fam)][r].to(torch.bfloat16)
                _assert_moe_close(y, exp)
                _worst(worst, fam, _dev(y, exp))
                if fam.startswith("ep"):
                    outside = ~((ids >= off) & (ids < off + nloc)).any(dim=1)
                    assert torch.equal(y[outside], torch.zeros_like(y[outside])), (
                        f"{name} wrote rows routed entirely elsewhere at T={num_tokens}"
                    )
                    zero_rows[fam] = max(zero_rows.get(fam, 0), int(outside.sum()))
                outs.append(y)
            total = sum(o.float() for o in outs).to(torch.bfloat16)
            _assert_moe_close(total, exp_full)
            _worst(worst, fam + " sum", _dev(total, exp_full))
            if num_tokens == 1024 and fam in ("ep8", "tp2"):
                if fam == "ep8":
                    _, _, n_loc, la, ls = lays[1]
                    bad = _call(
                        data,
                        sf,
                        la,
                        V41_E,
                        V41_K,
                        local_expert_offset=0,
                        local_num_experts=n_loc,
                        topk_ids=ids,
                        topk_weights=wts,
                        **ls,
                    )[0]
                    exp, what = parts[("ep", 48)][1].to(torch.bfloat16), "EP8 window 1 at offset 0"
                else:
                    bad, exp, what = outs[0], exp_full, "TP2 sum without rank 1"
                ctrl.append((what, *(round(v, 1) for v in _dev(bad, exp))))
                try:
                    _assert_moe_close(bad, exp)
                except AssertionError:
                    continue
                raise AssertionError(f"the gate accepts {what}")
    fmt = {k: f"{v[0]:.2f} / {v[1]:.2f}" for k, v in sorted(worst.items())}
    print(
        f"  test_v41_e6_rank_forms_geometry OK (worst elt / rms ulp: {fmt}; bitwise-zero rows per window "
        f"<= {zero_rows}; new partitions vs certified ones <= {tie_worst:.1e} of the row; controls (elt, rms): {ctrl})"
    )


def test_v41_e6_row_axis_by_bucket():
    """Every tuner bucket and untuned tile up to 8192 at the E6 rank forms, cold and warm,
    and every dumped row count warm.

    On every EP2 window, EP8 window and TP2 rank, with the target's call (pre-routed,
    clamp 10, layer-level g2): the lower edge, an interior point and the upper edge of
    every autotuner bucket, and both sides of every untuned-tile edge of the form and of
    every other kernel change, each against the torch reference on fresh inputs. The
    kernels launched on EP2 / EP8 window 0 and TP2 rank 0 change between two neighbouring
    counts of the sweep only across a row count `_v41_kernel_firsts` names, and cold at
    every one of them. Then the cache is warmed through `autotune()` at 8192 rows on each
    form and on TP4 rank 0, and every count runs again. Warm, every row count of
    `_E6_DUMPED_TOKENS` also runs on every E6 rank and on the four TP4 ranks, on the first
    n rows of one 8192-row batch against the same rows of its reference. Clears the cache.
    """
    assert not _moe_cache_entries(), "the profiling cache already holds entries for this op"
    args, ref_l = _v41_layer_g2()
    fams = {fam: _e6_layouts(fam, args, ref_l) for fam in _E6}
    tp4 = [lay for lay in _v41_graph_layouts(args, ref_l) if lay[0].startswith("tp")]
    g1 = ref_l["g1"]
    gen = torch.Generator(device=DEV).manual_seed(4162)
    base = _v41_bucket_tokens() | set(_V41_KERNEL_EDGES)
    counts = {fam: sorted(base | _v41_tile_edges(_E6[fam][0])) for fam in _E6}
    assert {287, 288, 4127, 4128} <= set(counts["ep2"]) and {71, 72, 1031, 1032} <= set(
        counts["ep8"]
    )
    assert {1, 2, 575, 576, 4159, 4160} <= set(counts["tp2"])
    all_counts = sorted(set().union(*map(set, counts.values())))

    def run(lay, data, sf, ids, wts):
        _, off, nloc, la, ls = lay
        return _call(
            data,
            sf,
            la,
            V41_E,
            V41_K,
            local_expert_offset=off,
            local_num_experts=nloc,
            topk_ids=ids,
            topk_weights=wts,
            **ls,
        )[0]

    cases, cold, kern, worst = {}, {}, {}, {}
    for n in all_counts:
        _, data, sf, x, ids, wts = _v41_case("uniform", n, 0, V41_E, g1, gen)
        parts = _ref_v41_parts(x, ids, wts, ref_l, eps=(192, 48), tps=(1152,))
        exps = {}
        for fam, lays in fams.items():
            if n not in counts[fam]:
                continue
            for r, lay in enumerate(lays):
                exp = parts[_e6_part(fam)][r].to(torch.bfloat16)
                y = run(lay, data, sf, ids, wts)
                _assert_moe_close(y, exp)
                _worst(worst, ("cold", fam), _dev(y, exp))
                cold[(lay[0], n)] = y
                exps[lay[0]] = exp
                if r == 0:
                    kern[("cold", fam, n)] = _kernel_names(lambda: run(lay, data, sf, ids, wts))
        cases[n] = ((data, sf, ids, wts), exps)
        del parts, x
    assert not _moe_cache_entries(), "an ordinary call wrote a profiling-cache entry"

    with autotune():
        for lay in [lays[0] for lays in fams.values()] + [tp4[0]]:
            c = _v41_case("uniform", 8192, lay[1], lay[2], g1, gen)
            run(lay, c[1], c[2], c[4], c[5])
    uids = sorted({str(k[2]) for k in _moe_cache_entries()})
    assert uids == sorted([*_E6_UIDS.values(), "(6, 576, 384, 0)"]), uids
    same = total = 0
    for n in all_counts:
        (data, sf, ids, wts), exps = cases[n]
        for fam, lays in fams.items():
            if n not in counts[fam]:
                continue
            for r, lay in enumerate(lays):
                y = run(lay, data, sf, ids, wts)
                _assert_moe_close(y, exps[lay[0]])
                _worst(worst, ("warm", fam), _dev(y, exps[lay[0]]))
                same += int(torch.equal(y, cold[(lay[0], n)]))
                total += 1
                if r == 0:
                    kern[("warm", fam, n)] = _kernel_names(lambda: run(lay, data, sf, ids, wts))
    del cases, cold

    runs, changes = {}, {}
    for state in ("cold", "warm"):
        for fam in _E6:
            nloc, inter, _ = _E6[fam]
            firsts = _v41_kernel_firsts(nloc, inter, state == "warm")
            cs = counts[fam]
            seen, out = set(), []
            for a, b in zip(cs, cs[1:]):
                if kern[(state, fam, a)] != kern[(state, fam, b)]:
                    assert b == a + 1 and b in firsts, (
                        f"{state} {fam}: the kernels change between T={a} and T={b}, not at a known edge"
                    )
                    seen.add(b)
            if state == "cold":
                want = {f for f in firsts if f in cs and f - 1 in cs}
                assert seen == want, (
                    f"cold {fam}: expected kernel changes at {sorted(want - seen)} not seen"
                )
            for n in cs:
                if out and kern[(state, fam, n)] == out[-1][2]:
                    out[-1][1] = n
                else:
                    out.append([n, n, kern[(state, fam, n)]])
            runs[(state, fam)] = [(a, b) for a, b, _ in out]
            changes[(state, fam)] = sorted(seen)

    # Every dumped row count, warm, on the first n rows of one 8192-row batch.
    _, data, sf, x, ids, wts = _v41_case("uniform", 8192, 0, V41_E, g1, gen)
    parts = _ref_v41_parts(x, ids, wts, ref_l, eps=(192, 48), tps=(1152, V41_TP_I))
    exp_all = {key: [p.to(torch.bfloat16) for p in v] for key, v in parts.items()}
    del parts, x
    forms = dict(fams, tp4=tp4)
    dumped_calls = 0
    for n in _E6_DUMPED_TOKENS:
        for fam, lays in forms.items():
            key = ("tp", V41_TP_I) if fam == "tp4" else _e6_part(fam)
            for r, lay in enumerate(lays):
                y = run(lay, data[:n], sf[: n * V41_H // SV], ids[:n], wts[:n])
                exp = exp_all[key][r][:n]
                _assert_moe_close(y, exp)
                _worst(worst, ("dump", fam), _dev(y, exp))
                dumped_calls += 1
    entries = len(_moe_cache_entries())
    AutoTuner.get().clear_cache()
    fmt = {f"{k[0]} {k[1]}": f"{v[0]:.2f} / {v[1]:.2f}" for k, v in sorted(worst.items())}
    print(
        f"  test_v41_e6_row_axis_by_bucket OK (row counts per form: "
        f"{ {fam: len(c) for fam, c in counts.items()} }, every rank, 1..8192; worst elt / rms ulp {fmt}; "
        f"warm bitwise equal to cold on {same} of {total}; {entries} tuned entries over {uids}; "
        f"{len(_E6_DUMPED_TOKENS)} dumped counts x {sum(len(v) for v in forms.values())} ranks = {dumped_calls} "
        f"warm calls; kernel changes at {changes}; kernel runs {runs})"
    )


def test_v41_e6_cuda_graph_replay_is_bitwise_eager():
    """Captured once per (E6 rank, T), replayed on data the capture never saw, cold tuner.

    As `test_v41_cuda_graph_replay_is_bitwise_eager`, at the E6 forms: every EP2 window
    at G and 6G and dep2's gathered 2G and 12G (115 counts, 1..3072), every EP8 window
    at G and 6G and dep8's gathered 8G and 48G (130 counts, 1..12288), and both TP2
    ranks at G and 6G (82 counts, 1..1536). At 9216 and 12288 the graph holds the
    target's two calls (`_moe_chunks`: 8192 + 1024 and 8192 + 4096), each on its rows
    of the static buffers and into its own output. Each graph replays two uniform
    routings, a skewed one (every token on the same six local experts), on an EP window
    one routing nothing into it and, at the dep member's gathered counts, its
    zero-padded batch over 2 or 8 rank blocks (`pad`), then the capture's own data.
    Every replay is bitwise equal to the eager calls on the same values (one per
    chunk), and within the torch reference's gates (the empty window exactly zero);
    consecutive replays differ, so a stale replay could not pass.
    """
    assert not _moe_cache_entries(), "the cold-cache graph test ran with a warm cache"
    args, ref_l = _v41_layer_g2()
    fams = {fam: _e6_layouts(fam, args, ref_l) for fam in _E6}
    g1 = ref_l["g1"]
    gen = torch.Generator(device=DEV).manual_seed(4163)
    fam_counts = {fam: _e6_counts(fam) for fam in _E6}
    assert [len(fam_counts[f]) for f in ("ep2", "ep8", "tp2")] == [115, 130, 82]
    assert [max(fam_counts[f]) for f in ("ep2", "ep8", "tp2")] == [3072, 12288, 1536]
    counts = sorted(set().union(*map(set, fam_counts.values())))
    worst, worst_at, worst_split = {}, {}, (0.0, 0.0)
    replays = graphs = 0
    per_layout, split = {}, []
    for n in counts:
        chunks = _moe_chunks(n)
        need = [fam for fam in _E6 if n in fam_counts[fam]]
        eps = tuple(_E6[f][0] for f in need if f.startswith("ep"))
        tps = tuple(_E6[f][1] for f in need if f.startswith("tp"))
        shared = [_v41_case("uniform", n, 0, V41_E, g1, gen) for _ in range(2)]
        refs = [_ref_v41_parts(c[3], c[4], c[5], ref_l, eps, tps) for c in shared]
        for fam in need:
            nloc, inter, ranks = _E6[fam]
            key = _e6_part(fam)
            for r, lay in enumerate(fams[fam]):
                name, off = lay[:2]
                own = [_v41_case("skew", n, off, nloc, g1, gen)]
                if fam.startswith("ep"):
                    own.append(_v41_case("none", n, off, nloc, g1, gen))
                    if n in _E6_GATHERED[fam]:
                        own.append(_v41_case("pad", n, off, nloc, g1, gen, ranks=ranks))
                exps = [rf[key][r] for rf in refs]
                for c in own:
                    if c[0] == "none":
                        exps.append(None)
                    else:
                        sub = ((nloc,), ()) if fam.startswith("ep") else ((), (inter,))
                        exps.append(_ref_v41_parts(c[3], c[4], c[5], ref_l, *sub)[key][r])
                cases = shared + own + shared[:1]
                exps = exps + exps[:1]
                g = _V41Graph(lay, n, cases[0], chunks)
                prev = None
                for case, exp in zip(cases, exps):
                    y = g.replay(case)
                    eager = _v41_eager(lay, case, chunks)
                    assert torch.equal(y, eager), (
                        f"{name} T={n} {case[0]}: graph replay differs from eager "
                        f"{_dev(y, eager.float())} (elt / rms ulp)"
                    )
                    if exp is None:
                        assert torch.equal(y, torch.zeros_like(y)), (
                            f"{name} T={n}: an empty window is not zero"
                        )
                    else:
                        exp = exp.to(torch.bfloat16)
                        _assert_moe_close(y, exp)
                        dv = _dev(y, exp)
                        for i in range(2):
                            if dv[i] > worst.get(fam, (0.0, 0.0))[i]:
                                worst_at[(fam, i)] = (name, n, case[0])
                        _worst(worst, fam, dv)
                        if len(chunks) > 1:
                            worst_split = tuple(max(a, b) for a, b in zip(worst_split, dv))
                    assert prev is None or not torch.equal(prev, y) or not y.any(), (
                        f"{name} T={n}: two cases gave one result"
                    )
                    prev = y
                    replays += 1
                graphs += 1
                per_layout[name] = per_layout.get(name, 0) + 1
                if len(chunks) > 1:
                    split.append((name, n))
                del g
        del refs, shared
    assert not _moe_cache_entries()
    assert per_layout == {
        **{f"ep2.{r}": 115 for r in range(2)},
        **{f"ep8.{r}": 130 for r in range(8)},
        **{f"tp2.{r}": 82 for r in range(2)},
    }, per_layout
    assert len(split) == 16, split  # 9216 and 12288 on each of the eight EP8 windows
    fmt = {k: f"{v[0]:.2f} / {v[1]:.2f}" for k, v in worst.items()}
    print(
        f"  test_v41_e6_cuda_graph_replay_is_bitwise_eager OK ({graphs} (rank, T) graphs {per_layout}, "
        f"{replays} replays bitwise equal to eager; worst vs reference elt / rms ulp {fmt} (at {worst_at}); "
        f"two-call graphs {len(split)} at 9216 / 12288: {worst_split[0]:.2f} / {worst_split[1]:.2f}; "
        f"counts EP2 {list(fam_counts['ep2'])}; EP8 {list(fam_counts['ep8'])}; TP2 {list(fam_counts['tp2'])})"
    )


def test_v41_e6_cuda_graph_swizzled_scales_are_caught():
    """Control: the graph harness sees the swizzled activation-scale mistake at the E6
    gathered counts, the two-call graph included.

    At dep2's 3072 (12G at G 256) on EP2 windows 0 and 1, and at dep8's 6144 and 12288
    (48G at G 128 and 256; 12288 as 8192 + 4096) on EP8 windows 0 and 7, every call's
    row count is a multiple of 128, where a 128x4-swizzled scale buffer has the linear
    one's byte count. Replaying on each call's scale bytes in swizzled order must fail
    the reference gate, while the linear bytes pass.
    """
    args, ref_l = _v41_layer_g2()
    g1 = ref_l["g1"]
    gen = torch.Generator(device=DEV).manual_seed(4167)
    seen = []
    for fam, ranks, n in (("ep2", (0, 1), 3072), ("ep8", (0, 7), 6144), ("ep8", (0, 7), 12288)):
        nloc = _E6[fam][0]
        chunks = _moe_chunks(n)
        case = _v41_case("uniform", n, 0, V41_E, g1, gen)
        kind, data, sf, x, ids, wts = case
        windows = _ref_v41_parts(x, ids, wts, ref_l, eps=(nloc,))[("ep", nloc)]
        swz = torch.cat(
            [
                _swizzle_scales(
                    sf[a * V41_H // SV : b * V41_H // SV].view(1, b - a, V41_H // SV)
                ).reshape(-1)
                for a, b in chunks
            ]
        ).contiguous()
        assert swz.numel() == sf.numel() and not torch.equal(swz, sf)
        for lay in _e6_layouts(fam, args, ref_l, ranks):
            name, off = lay[:2]
            exp = windows[off // nloc].to(torch.bfloat16)
            g = _V41Graph(lay, n, case, chunks)
            _assert_moe_close(g.replay(case), exp)
            bad = g.replay((kind, data, swz, x, ids, wts))
            elt, rms = _dev(bad, exp)
            assert rms > 2.0 and elt > 16.0, (
                f"{name} T={n}: swizzled scales passed ({elt:.1f} / {rms:.2f} ulp)"
            )
            seen.append((name, n, round(elt), round(rms)))
            del g
    print(
        "  test_v41_e6_cuda_graph_swizzled_scales_are_caught OK "
        f"(swizzled replay vs reference, elt / rms ulp: {seen}; gates 16 / 2)"
    )


def test_v41_e6_cuda_graph_tactic_is_fixed_at_capture():
    """At the E6 forms: a graph keeps the tactic it was captured with; no tactic seen
    changes the result.

    As `test_v41_cuda_graph_tactic_is_fixed_at_capture`, one group of ranks at a time
    (EP2 windows 0-1, EP8 windows 0-3, EP8 windows 4-7, TP2 ranks 0-1), each from a
    cold cache. The group's graphs at every captured count (`_e6_counts`, the two-call
    graphs at 9216 and 12288 included) are captured cold; the cache is warmed through
    `autotune()` at 8192 rows on the group's first rank (every token bucket a captured
    call maps to must then hold a non-fallback tactic); the cold graphs replay
    unchanged, kernels and bits; graphs captured warm replay bitwise equal to the cold
    results and to a warm eager call; clearing the cache changes neither.

    Per-bucket kernel choice: a graph launches exactly the kernels of eager calls at
    its calls' row counts in the same tuner state, and each of those equals the
    kernels of eager calls at the lower and upper edge of the count's bucket (warm) or
    untuned tile (cold) (`_e6_segment`), checked on the group's first rank at every
    count and on every rank at the counts above 1536.
    """
    assert not _moe_cache_entries(), "the profiling cache already holds entries for this op"
    args, ref_l = _v41_layer_g2()
    g1 = ref_l["g1"]
    gen = torch.Generator(device=DEV).manual_seed(4164)
    kgen = torch.Generator(device=DEV).manual_seed(4165)

    def tactic(k):
        t = AutoTuner.get().profiling_cache.cache[k][1]
        return tuple(t) if isinstance(t, (list, tuple)) else (t,)

    groups = (("ep2", (0, 1)), ("ep8", (0, 1, 2, 3)), ("ep8", (4, 5, 6, 7)), ("tp2", (0, 1)))
    summary, changed, checked, graphs = [], [], {"cold": 0, "warm": 0}, 0
    for fam, ranks in groups:
        assert not _moe_cache_entries()
        nloc, inter, _ = _E6[fam]
        lays = _e6_layouts(fam, args, ref_l, ranks)
        counts = _e6_counts(fam)
        eager_k = {"cold": {}, "warm": {}}

        def eager_kernels(state, lay, m):
            key = (lay[0], m)
            if key not in eager_k[state]:
                _, off, nl, la, ls = lay
                c = _v41_case("uniform", m, 0, V41_E, g1, kgen)
                out = torch.empty(m, V41_H, dtype=torch.bfloat16, device=DEV)

                def run():
                    _call(
                        c[1],
                        c[2],
                        la,
                        V41_E,
                        V41_K,
                        local_expert_offset=off,
                        local_num_experts=nl,
                        topk_ids=c[4],
                        topk_weights=c[5],
                        output=out,
                        **ls,
                    )

                run()
                eager_k[state][key] = _kernel_names(run)
            return eager_k[state][key]

        segs = {"cold": {}, "warm": {}}

        def check_bucket(state, lay, n, k_graph):
            if not (lay[0] == lays[0][0] or n > 1536):
                return
            sizes = [b - a for a, b in _moe_chunks(n)]
            want = sum((eager_kernels(state, lay, m) for m in sizes), ())
            assert tuple(k_graph) == want, (
                f"{state} {lay[0]} T={n}: the graph's kernels differ from eager calls at {sizes}: {k_graph} vs {want}"
            )
            for m in sizes:
                lo, hi = _e6_segment(m, nloc, inter, state == "warm")
                for e in (lo, hi):
                    assert eager_kernels(state, lay, e) == eager_kernels(state, lay, m), (
                        f"{state} {lay[0]} T={m}: eager kernels differ at the segment edge {e} ({lo}..{hi})"
                    )
                if lay[0] == lays[0][0]:
                    segs[state].setdefault(f"{lo}..{hi}", set()).add(m)
            checked[state] += 1

        held = {}
        for lay in lays:
            name, off = lay[:2]
            for n in counts:
                chunks = _moe_chunks(n)
                kinds = ("uniform", "skew") + (("none",) if fam.startswith("ep") else ())
                cases = [
                    c[:3] + (None,) + c[4:]
                    for c in (_v41_case(k, n, off, nloc, g1, gen) for k in kinds)
                ]
                g = _V41Graph(lay, n, cases[0], chunks)
                k_cold = g.kernels(cases[0])
                check_bucket("cold", lay, n, k_cold)
                held[(name, n)] = (g, cases, [_v41_eager(lay, c, chunks) for c in cases], k_cold)
        assert not _moe_cache_entries(), (
            "an ordinary or captured call wrote a profiling-cache entry"
        )

        with autotune():
            _, off, nl, la, ls = lays[0]
            c = _v41_case("uniform", 8192, off, nl, g1, gen)
            _call(
                c[1],
                c[2],
                la,
                V41_E,
                V41_K,
                local_expert_offset=off,
                local_num_experts=nl,
                topk_ids=c[4],
                topk_weights=c[5],
                **ls,
            )
        keys = _moe_cache_entries()
        tuned = {(str(k[2]), k[3][2][0]): tactic(k) for k in keys}
        assert sorted({u for u, _ in tuned}) == [_E6_UIDS[fam]], tuned
        for n in counts:
            for a, b in _moe_chunks(n):
                bkt = 1 << ((b - a).bit_length() - 1)
                assert tuned.get((_E6_UIDS[fam], bkt), (-1,)) != (-1,), (
                    f"bucket {bkt} of {fam} is not tuned"
                )

        grp_changed = []
        for lay in lays:
            name = lay[0]
            for n in counts:
                chunks = _moe_chunks(n)
                g, cases, want, k_cold = held[(name, n)]
                for c, w in zip(cases, want):
                    assert torch.equal(g.replay(c), w), (
                        f"{name} T={n} {c[0]}: a cold graph changed after warming"
                    )
                    assert torch.equal(_v41_eager(lay, c, chunks), w), (
                        f"{name} T={n} {c[0]}: warm eager != cold"
                    )
                assert g.kernels(cases[0]) == k_cold, (
                    f"{name} T={n}: a cold graph re-read the tuner at replay"
                )
                del g
                warm = _V41Graph(lay, n, cases[0], chunks)
                k_warm = warm.kernels(cases[0])
                check_bucket("warm", lay, n, k_warm)
                if k_warm != k_cold:
                    grp_changed.append((name, n))
                for c, w in zip(cases, want):
                    assert torch.equal(warm.replay(c), w), (
                        f"{name} T={n} {c[0]}: warm graph != cold result"
                    )
                held[(name, n)] = (warm, cases, want, k_warm)
                graphs += 1
        AutoTuner.get().clear_cache()
        for (name, n), (warm, cases, want, k_warm) in held.items():
            assert warm.kernels(cases[0]) == k_warm, f"{name} T={n}: a warm graph re-read the tuner"
            for c, w in zip(cases, want):
                assert torch.equal(warm.replay(c), w), (
                    f"{name} T={n} {c[0]}: warm graph changed after clear"
                )
        del held
        torch.cuda.empty_cache()
        changed += grp_changed
        summary.append(
            (
                fam,
                ranks,
                len(keys),
                len(set(tuned.values())),
                len(grp_changed),
                {
                    s: {
                        k: sorted(v)
                        for k, v in sorted(
                            segs[s].items(), key=lambda kv: int(kv[0].split("..")[0])
                        )
                    }
                    for s in ("cold", "warm")
                },
            )
        )
    assert changed, (
        "no graph captured warm launched different kernels — the comparison proved nothing"
    )
    print(
        f"  test_v41_e6_cuda_graph_tactic_is_fixed_at_capture OK ({graphs} (rank, T) graphs, cold- and "
        f"warm-captured, bitwise equal on every replay; warm capture launched different kernels at "
        f"{len(changed)} of them, e.g. {changed[:4]}; per-bucket kernel check at {checked} graphs; per group "
        f"(family, ranks, tuned entries, distinct tactics, changed, window-0 call sizes by segment): {summary})"
    )


# dep2's and dep8's zero-padding skew. Like dep4, both pad every rank's rows to the
# step's largest per-rank count with zero rows before the all-gather, and
# `_moe_chunks` cuts the gathered batch into calls of 8192 rows and one remainder:
# dep2's eager row counts are multiples of 2 in 2..8192, dep8's multiples of 8 in
# 8..8192. Every padding row goes to the router's six bias-top experts with weight
# 0.25; on an EP8 window of 48 experts all six can be local.

_DEP_SIX_SPREAD = {192: (5, 17, 101, 200, 290, 377), 48: (5, 60, 110, 170, 250, 377)}


def _dep_six(w, nloc):
    """The six experts every padding row is routed to: all inside window ``w`` of
    ``nloc`` experts, or (``w`` None) spread over the windows."""
    if w is None:
        return _DEP_SIX_SPREAD[nloc]
    return tuple(w * nloc + e % nloc for e in (5, 17, 33, 50, 71, 90))


def _dep_eager_counts(ranks, nloc):
    """A dep member's eager row counts (multiples of ``ranks``) at every tuner bucket's
    lower edge, one interior point and upper edge, and on both sides of every
    untuned-tile and other kernel edge of its ``nloc``-expert window."""
    out, b = set(), ranks
    while b <= 8192:
        vals = list(range(b, min(2 * b - 1, 8192) + 1, ranks))
        out |= {vals[0], vals[len(vals) // 2], vals[-1]}
        b *= 2
    for e in _v41_tile_edges(nloc) | set(_V41_KERNEL_EDGES):
        out |= {v for v in (ranks * (e // ranks), ranks * -(-e // ranks)) if ranks <= v <= 8192}
    return sorted(out)


def _dep_pad_mask(n, cpu, ranks):
    """Padding rows of one `_moe_chunks` chunk of `n` rows cut from a random gathered
    batch of ``ranks`` rank blocks of `rows` rows (at most 16384 per rank): the chunk
    index j is random below 2 * ranks, the batch holds 8192 (j + 1) rows when n is 8192
    (the chunk any of them) and 8192 j + n otherwise (the chunk the last); one rank is
    full (the step's largest), the rank whose block overlaps the chunk most holds ONE
    real token, the others a random count."""
    j = int(torch.randint(0, 2 * ranks, (1,), generator=cpu))
    if n == 8192:
        rows, start = (
            8192 * (j + 1) // ranks,
            8192 * int(torch.randint(0, j + 1, (1,), generator=cpu)),
        )
    else:
        rows, start = (8192 * j + n) // ranks, 8192 * j
    ov = [max(0, min(start + n, (r + 1) * rows) - max(start, r * rows)) for r in range(ranks)]
    big = max(range(ranks), key=lambda r: (ov[r], float(torch.rand(1, generator=cpu))))
    rest = [r for r in range(ranks) if r != big]
    full = min(rest, key=lambda r: (ov[r], float(torch.rand(1, generator=cpu))))
    counts = [int(torch.randint(1, rows + 1, (1,), generator=cpu)) for _ in range(ranks)]
    counts[full], counts[big] = rows, 1
    pad = torch.zeros(ranks * rows, dtype=torch.bool)
    for r, c in enumerate(counts):
        pad[r * rows + c : (r + 1) * rows] = True
    return pad[start : start + n].to(DEV)


def _dep_case(kind, n, w, g1, seed, ranks, nloc):
    """As `_dep4_case`, for a dep member of ``ranks`` ranks on ``nloc``-expert windows
    (kinds ``dep``, ``one_real``, ``all_pad``, ``skew6``)."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    data, sf, x = _rand_activation(n, V41_H, g1, gen)
    ids, wts = _v41_routing(n, gen)
    six = torch.tensor(_dep_six(w, nloc), device=DEV, dtype=torch.int32)
    if kind == "skew6":
        pad = torch.zeros(n, dtype=torch.bool, device=DEV)
        ids = six.expand(n, V41_K).contiguous()
    elif kind == "dep":
        pad = _dep_pad_mask(n, cpu, ranks)
    else:
        pad = torch.ones(n, dtype=torch.bool, device=DEV)
        pad[0] = kind == "all_pad"
    ids[pad] = six
    wts[pad] = 0.25
    data[pad] = 0
    sf.view(n, V41_H // SV)[pad] = 0
    x[pad] = 0
    return (kind, data, sf, x, ids, wts), pad


def _dep_padding_skew(fam):
    """dep2's (``ep2``) or dep8's (``ep8``) zero-padding skew on every window of the
    member, eager, cold then warm; see the two tests below."""
    assert not _moe_cache_entries(), "the profiling cache already holds entries for this op"
    nloc, _, ranks = _E6[fam]
    args, ref_l = _v41_layer_g2()
    layouts = _e6_layouts(fam, args, ref_l)
    g1 = ref_l["g1"]
    counts = _dep_eager_counts(ranks, nloc)
    assert all(n % ranks == 0 for n in counts) and counts[0] == ranks and counts[-1] == 8192
    reach = {e for e in _v41_kernel_firsts(nloc, V41_I, False) if e % ranks == 0}
    assert all(e in counts and e - ranks in counts for e in reach), sorted(reach - set(counts))

    def plan(n):
        """(kind, window of the six, seed) of every case at row count n."""
        s = 1_000_003 * n + 7 * ranks
        cases = [("dep", None, s)] + [("dep", w, s + 1 + w) for w in range(ranks)]
        cases += [
            ("one_real", (n // ranks) % ranks, s + ranks + 1),
            ("all_pad", (n // ranks + 1) % ranks, s + ranks + 2),
        ]
        return cases + [("skew6", w, s + ranks + 3 + w) for w in range(ranks)]

    def run(lay, case):
        _, off, nl, la, ls = lay
        _, data, sf, _, ids, wts = case
        out = torch.full((data.shape[0], V41_H), float("nan"), dtype=torch.bfloat16, device=DEV)
        _call(
            data,
            sf,
            la,
            V41_E,
            V41_K,
            local_expert_offset=off,
            local_num_experts=nl,
            topk_ids=ids,
            topk_weights=wts,
            output=out,
            **ls,
        )
        return out

    stats = {"calls": 0, "zero_rows": 0, "pad_rows": 0, "max_expert_rows": 0}
    worst = {}
    kern = {}

    def check(tag, n, kind, w, seed, profile_it):
        case, pad = _dep_case(kind, n, w, g1, seed, ranks, nloc)
        refs = _ref_v41_parts(case[3], case[4], case[5], ref_l, eps=(nloc,))[("ep", nloc)]
        per_expert = torch.bincount(case[4].flatten().long(), minlength=V41_E)
        stats["max_expert_rows"] = max(stats["max_expert_rows"], int(per_expert.max()))
        stats["pad_rows"] += int(pad.sum())
        for lay, rf in zip(layouts, refs):
            name = lay[0]
            y = run(lay, case)
            exp = rf.to(torch.bfloat16)
            zero = (exp == 0).all(dim=1)
            assert pad.logical_not().logical_or(zero).all(), (
                f"{tag} {name} T={n} {kind}: a padding row is nonzero"
            )
            assert (y[zero] == 0).all(), (
                f"{tag} {name} T={n} {kind} six@{w}: a row the reference leaves zero is not"
            )
            stats["zero_rows"] += int(zero.sum())
            if not zero.all():
                _assert_moe_close(y, exp)
                dv = _dev(y[~zero], exp[~zero])
                _worst(worst, "all", dv)
                _worst(worst, "all-on-six" if kind == "skew6" else "padded", dv)
                if tag == "dump":
                    _worst(worst, "dumped", dv)
            stats["calls"] += 1
            if profile_it and name == f"{fam}.{w}":
                kern[(tag, n, kind)] = _kernel_names(lambda: run(lay, case))

    uniform_kernels = {}
    tuned = 0
    dumped = [n for n in _E6_DUMPED_TOKENS if n % ranks == 0] if fam == "ep8" else []
    for tag in ("cold", "warm"):
        if tag == "warm":
            with autotune():
                _, off, nl, la, ls = layouts[0]
                c = _v41_case(
                    "uniform", 8192, off, nl, g1, torch.Generator(device=DEV).manual_seed(4168)
                )
                _call(
                    c[1],
                    c[2],
                    la,
                    V41_E,
                    V41_K,
                    local_expert_offset=off,
                    local_num_experts=nl,
                    topk_ids=c[4],
                    topk_weights=c[5],
                    **ls,
                )
            uids = sorted({str(k[2]) for k in _moe_cache_entries()})
            assert uids == [_E6_UIDS[fam]], uids
            tuned = len(_moe_cache_entries())
        for n in counts:
            u = _v41_case(
                "uniform", n, 0, nloc, g1, torch.Generator(device=DEV).manual_seed(4169 + n)
            )
            uniform_kernels[(tag, n)] = _kernel_names(lambda: run(layouts[0], u))
            for kind, w, seed in plan(n):
                check(
                    tag,
                    n,
                    kind,
                    w,
                    seed,
                    profile_it=kind == "all_pad" or (kind == "skew6" and w == 0),
                )
            for kind in ("skew6", "all_pad"):
                if (tag, n, kind) in kern:
                    assert kern[(tag, n, kind)] == uniform_kernels[(tag, n)], (
                        f"{tag} T={n} {kind}: the routing changed the launched kernels"
                    )
        if tag == "warm":
            for i, n in enumerate(dumped):
                check(
                    "dump",
                    n,
                    "dep",
                    None if i % (ranks + 1) == ranks else i % (ranks + 1),
                    2_000_003 * n,
                    False,
                )
            assert len(_moe_cache_entries()) == tuned, "a skewed call re-tuned a bucket"
        assert (tag == "warm") == bool(_moe_cache_entries()), (
            "an ordinary call wrote a profiling-cache entry"
        )

    # Control: a result missing the last quarter of one expert's rows (a kernel whose
    # per-expert capacity assumed balanced routing) must fail the gate.
    ctrl = []
    for n in (256, 2048, 8192):
        case, _ = _dep_case("skew6", n, 0, g1, 4170 + n, ranks, nloc)
        y = run(layouts[0], case)
        lost = case[5].clone()
        lost[3 * n // 4 :, 5] = 0
        bad = _ref_v41_parts(case[3], case[4], lost, ref_l, eps=(nloc,))[("ep", nloc)][0].to(
            torch.bfloat16
        )
        ctrl.append((n, *(round(v, 1) for v in _dev(y, bad))))
        try:
            _assert_moe_close(y, bad)
        except AssertionError:
            continue
        raise AssertionError(
            f"T={n}: the gate accepts a result missing a quarter of one expert's rows"
        )
    AutoTuner.get().clear_cache()
    fmt = {k: f"{v[0]:.2f} / {v[1]:.2f}" for k, v in worst.items()}
    return (
        f"{len(counts)} row counts {counts[:6]}..{counts[-3:]} x {len(plan(8))} cases x {ranks} windows, cold and "
        f"warm ({tuned} tuned entries), plus {len(dumped)} dumped counts warm: {stats['calls']} calls, "
        f"{stats['pad_rows']} padding rows, max rows on one expert {stats['max_expert_rows']}, "
        f"{stats['zero_rows']} rows exactly zero; worst elt / rms ulp {fmt}; kernels at a row count are "
        f"routing-independent at all {len(counts)} counts cold and warm; lost-rows control (T, elt, rms) {ctrl}"
    )


def test_v41_e6_dep2_padding_skew_eager_by_bucket():
    """dep2's zero-padding skew on both EP2 windows, eager, at every chunk size the target passes.

    As `test_v41_dep4_padding_skew_eager_by_bucket` with two rank blocks: row counts in
    multiples of 2 at every tuner bucket's lower edge, an interior point and upper edge,
    and on both sides of every untuned-tile and other kernel edge of a 192-expert window
    (`_dep_eager_counts`). At each count, each case runs on both windows with `output=`
    NaN-filled: a random chunk of a padded gathered batch (`_dep_pad_mask`) with the six
    padding experts spread over the windows and with all six inside each window in turn;
    one real row behind n - 1 padding rows; n padding rows; and, for each window, n real
    rows all on its six experts. Held to the torch reference; every row the reference
    leaves at zero must be exactly zero. Cold, then warm (`autotune()` at 8192 rows): no
    warm call writes a cache entry, and the kernels at a row count are the same for
    uniform, skewed and all-padding routing. Control: a reference that loses the last
    quarter of one expert's rows fails the gate. Clears the cache at the end.
    """
    print(f"  test_v41_e6_dep2_padding_skew_eager_by_bucket OK ({_dep_padding_skew('ep2')})")


def test_v41_e6_dep8_padding_skew_eager_by_bucket():
    """dep8's zero-padding skew on all eight EP8 windows, eager, at every chunk size the target passes.

    As `test_v41_e6_dep2_padding_skew_eager_by_bucket` with eight rank blocks and 48-expert
    windows (row counts in multiples of 8; every case on all eight windows; all six
    padding experts inside each window in turn, and n real rows on each window's six),
    and the warm pass adds a padded chunk at every multiple of 8 among
    `_E6_DUMPED_TOKENS`.
    """
    print(f"  test_v41_e6_dep8_padding_skew_eager_by_bucket OK ({_dep_padding_skew('ep8')})")


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


def _certify_v41_cell(cell) -> None:
    """One DeepSeek-V4.1 cell: the target's call on one rank, against the entry's
    reference, which is first tied to `_ref_v41_parts`.

    The layer is `_v41()`'s -- per-expert `gw1`, `gw2`, `g2`, the clamp at 10
    passed as `10 / output1_scale_gate_scalar` -- or, for the layer-level cells,
    `_v41_layer_g2()`'s, the form the target passes. With `output` the kernel
    writes into a NaN-filled `output=` buffer, as the target's call does, and
    returns the empty list element; otherwise it returns the result.

    An EP window's entry reference is `_ref_v41_parts`' share of that window bit
    for bit (same per-expert row order, same fp32 ops), and a token with no slot
    in the window must come back bitwise zero. A MoE-TP rank's entry reference
    runs its FC1 GEMM over the rank's own columns while `_ref_v41_parts` slices
    the full-width one; the two GEMMs may round an fp32 element differently and
    flip one e2m1 code of the intermediate, so that tie takes the entry's gate
    rather than bitwise equality.
    """
    spec = cell.spec
    assert (spec["hidden"], spec["num_experts"], spec["top_k"]) == (V41_H, V41_E, V41_K), spec
    assert spec["swiglu_limit"] == V41_LIMIT, spec
    args, ref = _v41() if spec["g2"] == "expert" else _v41_layer_g2()
    off, n, inter = spec["local_expert_offset"], spec["local_num_experts"], spec["intermediate"]
    tokens = spec["tokens"]
    gen = torch.Generator(device=DEV).manual_seed(zlib.crc32(cell.why.encode()))

    data, sf, x = _rand_activation(tokens, V41_H, ref["g1"], gen)
    ids, wts = _v41_routing(tokens, gen)
    scal = _v41_scalars(ref)
    if spec["tp_rank"] is None:
        kargs = args if n == V41_E else _window_args(args, off, n)
        kscal = _v41_slice(scal, off, n)
        ops = _window_ref(ref, off, n)
        share = _ref_v41_parts(x, ids, wts, ref, eps=(n,))[("ep", n)][off // n]
    else:
        r = spec["tp_rank"]
        a, b = r * inter, (r + 1) * inter
        kargs = _v41_tp_args(inter)[r]
        kscal = scal
        ops = dict(
            up=(ref["up"][0][:, a:b], ref["up"][1][:, a:b]),
            gate=(ref["gate"][0][:, a:b], ref["gate"][1][:, a:b]),
            down=(ref["down"][0][:, :, a:b], ref["down"][1][:, :, a // SV : b // SV]),
        )
        share = _ref_v41_parts(x, ids, wts, ref, tps=(inter,))[("tp", inter)][r]
    assert kargs["intermediate_size"] == inter, (kargs["intermediate_size"], inter)
    win = slice(off, off + n)

    # 1. The entry's reference against this file's, on identical inputs.
    entry_ref = moe.reference(
        x,
        ids,
        wts,
        ops["up"],
        ops["gate"],
        ops["down"],
        ref["g2"][win],
        local_expert_offset=off,
        local_num_experts=n,
        fc1_weight_global_scale=ref["gw1"][win],
        fc2_weight_global_scale=ref["gw2"][win],
        swiglu_limit=spec["swiglu_limit"],
    )
    if spec["tp_rank"] is None:
        torch.testing.assert_close(entry_ref, share, rtol=0.0, atol=0.0)
    else:
        moe.compare(entry_ref.to(torch.bfloat16), share.to(torch.bfloat16))

    # 2. The entry's reference against the kernel, through the entry's gate.
    kw = {}
    if spec["output"]:
        kw["output"] = torch.full((tokens, V41_H), float("nan"), dtype=torch.bfloat16, device=DEV)
    res = _call(
        data,
        sf,
        kargs,
        V41_E,
        V41_K,
        local_expert_offset=off,
        local_num_experts=n,
        topk_ids=ids,
        topk_weights=wts,
        **kw,
        **kscal,
    )
    assert len(res) == 1, [tuple(t.shape) for t in res]
    if spec["output"]:
        assert res[0].numel() == 0, tuple(res[0].shape)
    out = kw.get("output", res[0])
    assert out.shape == (tokens, V41_H), out.shape
    if spec["tp_rank"] is None:
        outside = ~((ids >= off) & (ids < off + n)).any(dim=1)
        assert torch.equal(out[outside], torch.zeros_like(out[outside])), (
            "a token with no slot in the window did not come back zero"
        )
    moe.compare(out, entry_ref.to(torch.bfloat16))


@pytest.mark.parametrize("cell", moe.CELLS, ids=[c.why[:44] for c in moe.CELLS])
def test_certified_cells(cell):
    """Drive every configuration the entry claims, through the entry's own
    reference and its own gate.

    The cases above test the op; this one tests the *claim*. A cell list that
    nothing executes reads as coverage and is not.

    Each cell cross-checks `moe.reference` against this file's `_ref_moe`
    before either is used to gate the kernel. Two references disagreeing is a
    different failure from the kernel moving, and collapsing them into one
    assert loses which happened.
    """
    spec = cell.spec
    if spec.get("model") == "deepseek-v4.1":
        _certify_v41_cell(cell)
        return
    args, ref = _dsr1()
    off, n = spec["local_expert_offset"], spec["local_num_experts"]
    num_experts, hidden, top_k = spec["num_experts"], spec["hidden"], spec["top_k"]
    gen = torch.Generator(device=DEV).manual_seed(abs(hash(cell.why)) % 2**31)
    g1 = 137.0

    data, sf, x = _rand_activation(spec["tokens"], hidden, g1, gen)
    ids, wts = _routing(spec["tokens"], num_experts, top_k, gen)
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
    moe.compare(y, entry_ref.to(torch.bfloat16))
