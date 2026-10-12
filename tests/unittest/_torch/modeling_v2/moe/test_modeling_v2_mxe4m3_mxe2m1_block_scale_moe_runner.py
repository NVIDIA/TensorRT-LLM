# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mxe4m3_mxe2m1_block_scale_moe_runner catalog entry."""

import contextlib
import math
import zlib

import pytest
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.mxe4m3_mxe2m1_block_scale_moe_runner import (
    mxe4m3_mxe2m1_block_scale_moe_runner as moe,
)
from tensorrt_llm._torch.autotuner import AutoTuner, autotune, fallback_tactic

__extra_import_path__ = [".."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "mxe4m3_mxe2m1_block_scale_moe_runner requires a CUDA device"


DEV = "cuda"
# The reference GEMMs must be true fp32; TF32 would leave the reference with
# 10 mantissa bits, coarser than the bf16 output it is meant to bound.
torch.backends.cuda.matmul.allow_tf32 = False

# Relative distance between neighbouring bf16 values (1 + 2^-7 stored mantissa
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
SV = 32  # mx block size: one e8m0 scale per 32 elements along K
WEIGHT_ALIGN = 128  # TMA row alignment for both GEMMs
HIDDEN_ALIGN = 512  # extra alignment on FC1's K axis
E4M3_MAX = 448.0  # largest finite e4m3 magnitude


def _pad_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


# ── operand construction ──────────────────────────────────────────────────


def _exp_base(k: int) -> int:
    """Base of the 6-wide e8m0 exponent window for a `[*, k]` weight.

    Chosen so a `k`-long dot product against the activations built by
    `_rand_mxfp8` lands at a standard deviation near 3: large enough that the
    clamp limit of 7 bites on a few percent of the elements, small enough that
    the gated activation is not a constant +-56 everywhere. A saturated
    activation would make every candidate intermediate-quantization recipe fit
    equally well and would leave `test_intermediate_is_mxfp8_quantized` blind.
    """
    return 127 + round(0.5 * math.log2(0.01057 / k))


def _rand_mxfp4(e: int, n: int, k: int, gen: torch.Generator, exp_shift: int = 0):
    """Random mxfp4 expert stack.

    Returns `(packed [E, N, K/2] uint8, scales [E, N, K/32] uint8, codes
    [E, N, K] uint8)`. Codes and e8m0 exponents are drawn directly, so the
    fp32 value of every weight is exact — the reference never has to model a
    quantizer. `exp_shift` scales the whole stack by `2^exp_shift`.
    """
    base = _exp_base(k) + exp_shift
    codes = torch.randint(0, 16, (e, n, k), dtype=torch.uint8, device=DEV, generator=gen)
    exps = torch.randint(
        base, base + 6, (e, n, k // SV), dtype=torch.uint8, device=DEV, generator=gen
    )
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
    return packed, exps, codes


def _dequant(codes_e: torch.Tensor, exps_e: torch.Tensor) -> torch.Tensor:
    """One expert's `[N, K]` fp32 weight from its codes and e8m0 exponents."""
    scale = torch.exp2(exps_e.float() - 127.0).repeat_interleave(SV, dim=1)
    return E2M1[codes_e.long()] * scale


def _rand_mxfp8(rows: int, valid_k: int, padded_k: int, gen: torch.Generator):
    """Random MXFP8 activation in the layout this op consumes.

    Returns `(data [rows, padded_k] float8_e4m3fn, sf [rows * padded_k/32]
    uint8, x [rows, valid_k] fp32)`, where `x` is the exact dequantization of
    the first `valid_k` columns. Columns past `valid_k` carry the zero byte and
    a zero scale byte, exactly as `torch.ops.trtllm.mxfp8_quantize` writes its
    column padding. Drawing the e4m3 elements and the e8m0 exponents directly
    keeps the reference operand exact — no quantizer is modelled here.
    """
    assert valid_k % SV == 0 and padded_k % SV == 0
    data = torch.zeros(rows, padded_k, dtype=torch.float32, device=DEV)
    data[:, :valid_k] = torch.randn(rows, valid_k, device=DEV, generator=gen)
    data = data.to(torch.float8_e4m3fn)
    sf = torch.zeros(rows, padded_k // SV, dtype=torch.uint8, device=DEV)
    sf[:, : valid_k // SV] = torch.randint(
        125,
        128,
        (rows, valid_k // SV),
        dtype=torch.uint8,
        device=DEV,
        generator=gen,
    )
    x = data.float() * torch.exp2(sf.float() - 127.0).repeat_interleave(SV, dim=1)
    return data, sf.reshape(-1).contiguous(), x[:, :valid_k].contiguous()


def _dequant_mxfp8(data: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Exact fp32 value of an `[rows, k]` e4m3 tensor with a linear scale buffer."""
    rows, k = data.shape
    scale = torch.exp2(sf.view(rows, k // SV).float() - 127.0)
    return data.float() * scale.repeat_interleave(SV, dim=1)


# ── kernel weight layout (pure torch) ─────────────────────────────────────


def _pad3(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    out = torch.zeros(t.shape[0], rows, cols, dtype=t.dtype, device=t.device)
    out[:, : t.shape[1], : t.shape[2]] = t
    return out


def _pad2(t: torch.Tensor, cols: int) -> torch.Tensor:
    out = torch.zeros(t.shape[0], cols, dtype=t.dtype, device=t.device)
    out[:, : t.shape[1]] = t
    return out


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
    v = s.reshape(e, m // 128, 4, 32, c // 4, 4)  # (e, m/128, (m%128)/32, m%32, c/4, c%4)
    v = v.permute(0, 1, 4, 3, 2, 5)
    return v.reshape(e, m, c).contiguous()


def _prep_fc1(up_p, gt_p, up_s, gt_s, up_b, gt_b, i_pad, h1_pad):
    """Pad both halves, concat as `[up | gate]`, interleave + block-shuffle rows."""
    w = torch.cat([_pad3(up_p, i_pad, h1_pad // 2), _pad3(gt_p, i_pad, h1_pad // 2)], dim=1)
    s = torch.cat([_pad3(up_s, i_pad, h1_pad // SV), _pad3(gt_s, i_pad, h1_pad // SV)], dim=1)
    b = torch.cat([_pad2(up_b, i_pad), _pad2(gt_b, i_pad)], dim=1).float()
    m = w.shape[1]
    perm = _gate_interleave_perm(m)[_blk32_perm(m)]
    return (
        torch.index_select(w, 1, perm).contiguous(),
        _swizzle_scales(torch.index_select(s, 1, perm)),
        torch.index_select(b, 1, perm).contiguous(),
    )


def _prep_fc2(dn_p, dn_s, dn_b, h2_pad, i_pad):
    """Pad the down projection and block-shuffle its rows."""
    w = _pad3(dn_p, h2_pad, i_pad // 2)
    s = _pad3(dn_s, h2_pad, i_pad // SV)
    b = _pad2(dn_b, h2_pad).float()
    perm = _blk32_perm(h2_pad)
    return (
        torch.index_select(w, 1, perm).contiguous(),
        _swizzle_scales(torch.index_select(s, 1, perm)),
        torch.index_select(b, 1, perm).contiguous(),
    )


def _build(
    num_experts: int,
    hidden: int,
    inter: int,
    seed: int,
    fc1_exp_shift: int = 0,
    with_bias: bool = True,
):
    """Build one MoE layer: kernel-ready tensors plus the reference operands.

    `fc1_exp_shift` scales both FC1 halves by `2^fc1_exp_shift` (so a clamp
    limit bites more often); `with_bias=False` passes both biases as `None`
    and gives the reference zero biases.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)
    i_pad = _pad_up(inter, WEIGHT_ALIGN)
    h1_pad = _pad_up(hidden, HIDDEN_ALIGN)
    h2_pad = _pad_up(hidden, WEIGHT_ALIGN)
    up_p, up_s, up_c = _rand_mxfp4(num_experts, inter, hidden, gen, fc1_exp_shift)
    gt_p, gt_s, gt_c = _rand_mxfp4(num_experts, inter, hidden, gen, fc1_exp_shift)
    dn_p, dn_s, dn_c = _rand_mxfp4(num_experts, hidden, inter, gen)
    if with_bias:
        up_b = torch.randn(num_experts, inter, device=DEV, generator=gen)
        gt_b = torch.randn(num_experts, inter, device=DEV, generator=gen)
        dn_b = torch.randn(num_experts, hidden, device=DEV, generator=gen) * 0.05
    else:
        up_b = torch.zeros(num_experts, inter, device=DEV)
        gt_b = torch.zeros(num_experts, inter, device=DEV)
        dn_b = torch.zeros(num_experts, hidden, device=DEV)
    w1, s1, b1 = _prep_fc1(up_p, gt_p, up_s, gt_s, up_b, gt_b, i_pad, h1_pad)
    w2, s2, b2 = _prep_fc2(dn_p, dn_s, dn_b, h2_pad, i_pad)
    args = dict(
        gemm1_weights=w1,
        gemm1_weights_scale=s1,
        gemm1_bias=b1 if with_bias else None,
        gemm2_weights=w2,
        gemm2_weights_scale=s2,
        gemm2_bias=b2 if with_bias else None,
        intermediate_size=i_pad,
        valid_hidden_size=hidden,
        valid_intermediate_size=inter,
    )
    ref = dict(
        up=(up_c, up_s, up_b),
        gate=(gt_c, gt_s, gt_b),
        down=(dn_c, dn_s, dn_b),
        h1_pad=h1_pad,
        gen=gen,
    )
    return args, ref


def _call(data, sf, args, num_experts, top_k, **kw):
    """Invoke the wrapper with this layer's tensors and size scalars."""
    kw.setdefault("routing_logits", None)
    kw.setdefault("routing_bias", None)
    kw.setdefault("n_group", None)
    kw.setdefault("topk_group", None)
    kw.setdefault("local_expert_offset", 0)
    kw.setdefault("local_num_experts", num_experts)
    kw.setdefault("routed_scaling_factor", None)
    kw.setdefault("routing_method_type", 1)
    kw.setdefault("act_type", 0)
    kw.setdefault("gemm1_alpha", None)
    kw.setdefault("gemm1_beta", None)
    kw.setdefault("gemm1_clamp_limit", None)
    kw.setdefault("valid_hidden_size", args["valid_hidden_size"])
    kw.setdefault("valid_intermediate_size", args["valid_intermediate_size"])
    kw.setdefault("intermediate_size", args["intermediate_size"])
    # _call is the single call site: run the guard over every call it makes.
    # The negative cases reach the raw op directly when they want it unguarded.
    with validating(moe):
        return moe(
            kw.pop("routing_logits"),
            kw.pop("routing_bias"),
            data,
            sf,
            args["gemm1_weights"],
            args["gemm1_weights_scale"],
            args["gemm1_bias"],
            kw.pop("gemm1_alpha"),
            kw.pop("gemm1_beta"),
            kw.pop("gemm1_clamp_limit"),
            args["gemm2_weights"],
            args["gemm2_weights_scale"],
            args["gemm2_bias"],
            num_experts,
            top_k,
            kw.pop("n_group"),
            kw.pop("topk_group"),
            kw.pop("intermediate_size"),
            kw.pop("valid_hidden_size"),
            kw.pop("valid_intermediate_size"),
            kw.pop("local_expert_offset"),
            kw.pop("local_num_experts"),
            kw.pop("routed_scaling_factor"),
            kw.pop("routing_method_type"),
            kw.pop("act_type"),
            **kw,
        )


# ── reference ─────────────────────────────────────────────────────────────


# The FC1 epilogue's block-scale recipe is architecture-specific, and the
# difference is bit-exact rather than a tolerance: trtllm-gen ships one cubin
# per architecture. Measured on each, with an identity down-projection reading
# the intermediate out element by element (test_intermediate_is_mxfp8_quantized):
#
#   sm_100   e8m0 = floor(log2(amax)) - 8      "OCP scale"
#   sm_103   e8m0 = ceil(log2(amax / 448))     "round-up scale"
#
# The round-up form is the one `torch.ops.trtllm.mxfp8_quantize` has always
# used, so sm_103 makes the MoE epilogue and the standalone quantizer agree.
# Both are named here, and each architecture's test refutes the other's recipe,
# so a future cubin that switches back cannot pass silently.
_OCP_SCALE, _ROUND_UP_SCALE = "ocp", "round_up"

_SCALE_RECIPE_BY_SM = {
    (10, 0): _OCP_SCALE,
    (10, 3): _ROUND_UP_SCALE,
}


def _scale_recipe() -> str:
    sm = torch.cuda.get_device_capability()
    recipe = _SCALE_RECIPE_BY_SM.get(sm)
    assert recipe is not None, (
        f"the FC1 epilogue's requantization recipe is not certified on sm_{sm[0]}{sm[1]}; "
        "run the identity-down-projection probe and record it before trusting this entry"
    )
    return recipe


def _block_scale(amax: torch.Tensor, recipe: str) -> torch.Tensor:
    """The per-32-column e8m0 scale, under the named recipe."""
    if recipe == _OCP_SCALE:
        exp = torch.floor(torch.log2(amax)) - 8.0
    else:
        exp = torch.ceil(torch.log2(amax / E4M3_MAX))
    exp = torch.where(amax == 0, torch.full_like(amax, -127.0), exp)
    return torch.exp2(exp.clamp(-127.0, 127.0))


def _q_intermediate(act: torch.Tensor, recipe: str | None = None) -> torch.Tensor:
    """MXFP8 requantization the FC1 epilogue applies to its activation output.

    Per 32 consecutive intermediate columns: the architecture's block scale
    (see above), then round-to-nearest-even into e4m3 with saturation at
    +-448. Returns the dequantized fp32 value FC2 actually consumes. Pinned
    bit-exactly by `test_intermediate_is_mxfp8_quantized`.
    """
    rows, cols = act.shape
    blocks = act.reshape(rows, cols // SV, SV)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = _block_scale(amax, recipe or _scale_recipe())
    q = (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(rows, cols)


def _ref_moe(
    x_valid,
    ids,
    scales,
    ref,
    alpha=None,
    beta=None,
    limit=None,
    offset: int = 0,
    num_local: int | None = None,
    swap_gate_up: bool = False,
    quantize_intermediate: bool = True,
    recipe: str | None = None,
    weights_before_down: bool = False,
    stats: dict | None = None,
):
    """Native-torch MoE over dequantized mxfp4 weights, fp32 throughout.

    `ids` carries global expert ids; this rank answers for
    `[offset, offset + num_local)`. `scales[t, j]` multiplies slot `j`'s
    expert output; nothing is renormalized. The FC1 activation is requantized
    to MXFP8 before FC2, which is what the kernel does.

    `weights_before_down=True` is the DeepSeek-V4.1 checkpoint reference's
    order instead (`Expert.forward` in the checkpoint's `inference/model.py`):
    the routing weight multiplies the activation, which is rounded to bf16 and
    then MXFP8-quantized as FC2's input. `stats`, when given, accumulates how
    many FC1 elements the clamp limit changed.
    """
    num_tokens, hidden = x_valid.shape
    up_c, up_s, up_b = ref["up"]
    gt_c, gt_s, gt_b = ref["gate"]
    dn_c, dn_s, dn_b = ref["down"]
    if swap_gate_up:
        (up_c, up_s, up_b), (gt_c, gt_s, gt_b) = (gt_c, gt_s, gt_b), (up_c, up_s, up_b)
    num_local = up_c.shape[0] if num_local is None else num_local
    xf = x_valid.float()
    out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x_valid.device)
    for local_e in range(num_local):
        mask = ids == offset + local_e
        tok, slot = mask.nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        xe = xf[tok]
        up = xe @ _dequant(up_c[local_e], up_s[local_e]).t() + up_b[local_e]
        gate = xe @ _dequant(gt_c[local_e], gt_s[local_e]).t() + gt_b[local_e]
        if limit is not None:
            lim = float(limit[local_e])
            if stats is not None:
                stats["gate"] = stats.get("gate", 0) + int((gate > lim).sum())
                stats["up"] = stats.get("up", 0) + int((up.abs() > lim).sum())
                stats["n"] = stats.get("n", 0) + gate.numel()
            gate = gate.clamp(max=lim)
            up = up.clamp(-lim, lim)
        a = 1.0 if alpha is None else float(alpha[local_e])
        b = 0.0 if beta is None else float(beta[local_e])
        act = (up + b) * gate * torch.sigmoid(a * gate)
        w = scales[tok, slot].float().unsqueeze(1)
        if weights_before_down:
            act = (act * w).to(torch.bfloat16).float()
        if quantize_intermediate:
            act = _q_intermediate(act, recipe)
        y = act @ _dequant(dn_c[local_e], dn_s[local_e]).t() + dn_b[local_e]
        out.index_add_(0, tok, y if weights_before_down else y * w)
    return out


def _dev(y: torch.Tensor, ref: torch.Tensor):
    """(worst element deviation, relative RMS deviation), both in bf16 ulp."""
    o, r = y.float(), ref.float()
    row = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    elt = ((o - r).abs() / row).max().item() / ULP
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-9)).item() / ULP
    return elt, rms


def _assert_moe_close(y: torch.Tensor, ref: torch.Tensor) -> None:
    """Two gates: per-element, row-scaled; and aggregate relative RMS.

    Kernel and reference consume bit-identical mxfp4 weights and MXFP8
    activations and model the same FC1-output MXFP8 requantization, so they
    differ only in accumulation order and in whether a marginal FC1 value
    rounds to the same e4m3 code. Default `assert_close` tolerances cannot
    express that: their bf16 `atol=1e-5` sits three orders of magnitude below
    one output ulp of a two-GEMM chain, and per-element `rtol` is meaningless
    where cancellation drives `|ref|` to zero. So the element gate is 8 ulp of
    the row's largest magnitude and the aggregate gate is 4 ulp of relative
    RMS. Worst values measured over every configuration covered here: 2.2 ulp
    element-wise (a skewed-routing capture replay at the DSpark drafter
    geometry; 2.0 at 128 experts, 8192 tokens, H=I=2880) and 0.87 ulp RMS. Both
    gates were sized against deliberately wrong computations when they were
    set: a reference that skips the FC1-output requantization lands at
    19.6 / 10.6 ulp, and a gate/up-swapped or unshuffled operand at 100+ ulp.
    """
    assert y.dtype == ref.dtype == torch.bfloat16, (y.dtype, ref.dtype)
    assert y.shape == ref.shape, (y.shape, ref.shape)
    row = ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    torch.testing.assert_close(y.float() / row, ref.float() / row, rtol=0.0, atol=8 * ULP)
    _, rms = _dev(y, ref)
    assert rms <= 4.0, f"relative RMS {rms:.2f} ulp > 4 ulp"


def _routing(num_tokens, num_experts, top_k, gen):
    """Random ids/weights for the pre-routed entry point."""
    ids = torch.stack(
        [torch.randperm(num_experts, device=DEV, generator=gen)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    wts = torch.rand(num_tokens, top_k, device=DEV, generator=gen).to(torch.bfloat16)
    return ids, wts


def _logits(num_tokens, num_experts, gen):
    """Router logits whose per-token values are all distinct in bf16, so the
    kernel's top-k and torch.topk cannot disagree on a tie."""
    ladder = torch.linspace(-4.0, 4.0, num_experts, device=DEV)
    rows = [
        ladder[torch.randperm(num_experts, device=DEV, generator=gen)] for _ in range(num_tokens)
    ]
    lg = torch.stack(rows).to(torch.bfloat16)
    assert all(len(set(r.tolist())) == num_experts for r in lg), (
        "logit ties would make topk ambiguous"
    )
    return lg


# gpt-oss-120b MoE geometry, built once and shared by the tests that use it.
_GPT_OSS = None


def _gpt_oss():
    global _GPT_OSS
    if _GPT_OSS is None:
        _GPT_OSS = _build(128, 2880, 2880, seed=100)
    return _GPT_OSS


def _swiglu_params(num_local):
    """gpt-oss clamped-GLU constants, one entry per local expert."""
    return (
        torch.full((num_local,), 1.702, dtype=torch.float32, device=DEV),
        torch.full((num_local,), 1.0, dtype=torch.float32, device=DEV),
        torch.full((num_local,), 7.0, dtype=torch.float32, device=DEV),
    )


# ── tests ─────────────────────────────────────────────────────────────────


def test_layout_helpers_match():
    """The trtllm preprocessing ops reproduce the pure-torch layout exactly."""
    gen = torch.Generator(device=DEV).manual_seed(201)
    rows, cols = 256, 12
    x = torch.randint(0, 256, (4, rows, cols), dtype=torch.uint8, device=DEV, generator=gen)
    perm = _gate_interleave_perm(rows)[_blk32_perm(rows)]
    for e in range(x.shape[0]):
        assert torch.equal(
            torch.ops.trtllm.shuffle_matrix(x[e].contiguous(), perm),
            torch.index_select(x[e], 0, perm),
        ), "shuffle_matrix is not a plain row gather"
    assert torch.equal(
        torch.ops.trtllm.block_scale_interleave(x).reshape(x.shape),
        _swizzle_scales(x),
    ), "block_scale_interleave does not match the documented 128x4 swizzle"
    print("  test_layout_helpers_match OK")


def test_gpt_oss_pre_routed():
    """E=128, H=I=2880, top-4, clamped GLU: decode- through prefill-sized batches."""
    args, ref = _gpt_oss()
    alpha, beta, limit = _swiglu_params(128)
    gen = torch.Generator(device=DEV).manual_seed(11)
    for num_tokens in (1, 2, 4, 17, 256, 1024, 8192):
        data, sf, xv = _rand_mxfp8(num_tokens, 2880, ref["h1_pad"], gen)
        ids, wts = _routing(num_tokens, 128, 4, gen)
        out = _call(
            data,
            sf,
            args,
            128,
            4,
            topk_ids=ids,
            topk_weights=wts,
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=limit,
        )
        assert out.shape == (num_tokens, 2880), out.shape
        exp = _ref_moe(xv, ids, wts, ref, alpha, beta, limit).to(torch.bfloat16)
        _assert_moe_close(out, exp)
    print("  test_gpt_oss_pre_routed OK")


def test_gpt_oss_router_entry_point():
    """Routing inside the call: Renormalize (top-k then softmax) and Default."""
    args, ref = _gpt_oss()
    alpha, beta, limit = _swiglu_params(128)
    gen = torch.Generator(device=DEV).manual_seed(12)
    for num_tokens in (4, 128):
        data, sf, xv = _rand_mxfp8(num_tokens, 2880, ref["h1_pad"], gen)
        lg = _logits(num_tokens, 128, gen)
        kw = dict(gemm1_alpha=alpha, gemm1_beta=beta, gemm1_clamp_limit=limit)

        out = _call(data, sf, args, 128, 4, routing_logits=lg, **kw)
        top_v, top_i = torch.topk(lg.float(), 4, dim=-1)
        exp = _ref_moe(
            xv,
            top_i.to(torch.int32),
            torch.softmax(top_v, dim=-1),
            ref,
            alpha,
            beta,
            limit,
        ).to(torch.bfloat16)
        _assert_moe_close(out, exp)

        out0 = _call(data, sf, args, 128, 4, routing_logits=lg, routing_method_type=0, **kw)
        sm_v, sm_i = torch.topk(torch.softmax(lg.float(), dim=-1), 4, dim=-1)
        exp0 = _ref_moe(xv, sm_i.to(torch.int32), sm_v, ref, alpha, beta, limit).to(torch.bfloat16)
        _assert_moe_close(out0, exp0)

        # fp32 logits are accepted and give the same answer as their bf16 form
        out32 = _call(data, sf, args, 128, 4, routing_logits=lg.float().contiguous(), **kw)
        assert torch.equal(out32, out)
    print("  test_gpt_oss_router_entry_point OK")


def test_mxfp8_quantize_pairing():
    """`mxfp8_quantize(x, swizzled_layout=False, alignment=512)` is the pairing.

    Shape, dtype, dimensionality and scale order of that op's two outputs are
    exactly what this op consumes; every violation of the pairing is pinned.
    """
    args, ref = _gpt_oss()
    alpha, beta, limit = _swiglu_params(128)
    gen = torch.Generator(device=DEV).manual_seed(13)
    kw = dict(gemm1_alpha=alpha, gemm1_beta=beta, gemm1_clamp_limit=limit)
    for num_tokens in (1, 128):
        xb = torch.randn(num_tokens, 2880, device=DEV, dtype=torch.bfloat16, generator=gen)
        data, sf = torch.ops.trtllm.mxfp8_quantize(xb, False, 512)
        assert data.shape == (num_tokens, 3072) and data.dtype == torch.float8_e4m3fn
        assert sf.shape == (num_tokens * 96,) and sf.dtype == torch.uint8
        assert data.shape[1] == args["gemm1_weights"].shape[-1] * 2
        ids, wts = _routing(num_tokens, 128, 4, gen)
        out = _call(data, sf, args, 128, 4, topk_ids=ids, topk_weights=wts, **kw)
        xv = _dequant_mxfp8(data, sf)[:, :2880].contiguous()
        exp = _ref_moe(xv, ids, wts, ref, alpha, beta, limit).to(torch.bfloat16)
        _assert_moe_close(out, exp)

        # the swizzled scale buffer has the same byte count whenever the token
        # count is a multiple of 128 -- and is then taken without complaint
        sw_data, sw_sf = torch.ops.trtllm.mxfp8_quantize(xb, True, 512)
        assert torch.equal(sw_data, data), "swizzling must not change the data bytes"
        if num_tokens % 128 == 0:
            assert sw_sf.numel() == sf.numel()
            bad = _call(data, sw_sf, args, 128, 4, topk_ids=ids, topk_weights=wts, **kw)
            elt, rms = _dev(bad, exp)
            assert elt > 50.0 and rms > 20.0, (
                f"a swizzled scale buffer was not detectably wrong: {elt:.1f} ulp"
            )
        else:
            assert sw_sf.numel() != sf.numel()

    def rejected(name, fn):
        try:
            fn()
        except (RuntimeError, AssertionError):
            return
        raise AssertionError(f"{name} was accepted")

    xb = torch.randn(8, 2880, device=DEV, dtype=torch.bfloat16, generator=gen)
    data, sf = torch.ops.trtllm.mxfp8_quantize(xb, False, 512)
    ids, wts = _routing(8, 128, 4, gen)
    kw2 = dict(topk_ids=ids, topk_weights=wts, **kw)
    # alignment 32 leaves the hidden un-padded: rejected, never zero-extended
    d32, s32 = torch.ops.trtllm.mxfp8_quantize(xb, False, 32)
    assert d32.shape == (8, 2880) and s32.shape == (8 * 90,)
    rejected("alignment=32 hidden", lambda: _call(d32, s32, args, 128, 4, **kw2))
    rejected("2-D scale buffer", lambda: _call(data, sf.view(8, 96), args, 128, 4, **kw2))
    rejected(
        "fp8-typed scale buffer",
        lambda: _call(data, sf.view(torch.float8_e4m3fn), args, 128, 4, **kw2),
    )
    rejected(
        "short scale buffer",
        lambda: _call(data, sf[:-32].contiguous(), args, 128, 4, **kw2),
    )
    rejected(
        "long scale buffer",
        lambda: _call(data, torch.cat([sf, sf[:32]]), args, 128, 4, **kw2),
    )
    rejected("bf16 hidden_states", lambda: _call(xb, sf, args, 128, 4, **kw2))
    rejected(
        "uint8-typed hidden_states",
        lambda: _call(data.view(torch.uint8), sf, args, 128, 4, **kw2),
    )
    print("  test_mxfp8_quantize_pairing OK")


def test_intermediate_is_mxfp8_quantized():
    """The FC1 activation reaches FC2 as MXFP8, on the OCP scale, bit-exactly.

    A down projection set to the identity turns the returned rows into the
    kernel's own post-activation intermediate, so the requantization can be
    read out element by element instead of inferred from output noise.
    """
    for num_experts, hidden, num_tokens, seed in ((4, 512, 32, 71), (2, 2880, 16, 72)):
        inter = hidden
        gen = torch.Generator(device=DEV).manual_seed(seed)
        i_pad = _pad_up(inter, WEIGHT_ALIGN)
        h1_pad = _pad_up(hidden, HIDDEN_ALIGN)
        h2_pad = _pad_up(hidden, WEIGHT_ALIGN)
        up_p, up_s, up_c = _rand_mxfp4(num_experts, inter, hidden, gen)
        gt_p, gt_s, gt_c = _rand_mxfp4(num_experts, inter, hidden, gen)
        # down = identity: e2m1 code 2 is 1.0, e8m0 byte 127 is 2^0
        dn_c = torch.zeros(num_experts, hidden, inter, dtype=torch.uint8, device=DEV)
        dn_c[:, torch.arange(hidden), torch.arange(inter)] = 2
        dn_p = (dn_c[..., 0::2] | (dn_c[..., 1::2] << 4)).contiguous()
        dn_s = torch.full((num_experts, hidden, inter // SV), 127, dtype=torch.uint8, device=DEV)
        zero1 = torch.zeros(num_experts, inter, device=DEV)
        zero2 = torch.zeros(num_experts, hidden, device=DEV)
        w1, s1, b1 = _prep_fc1(up_p, gt_p, up_s, gt_s, zero1, zero1, i_pad, h1_pad)
        w2, s2, b2 = _prep_fc2(dn_p, dn_s, zero2, h2_pad, i_pad)
        args = dict(
            gemm1_weights=w1,
            gemm1_weights_scale=s1,
            gemm1_bias=b1,
            gemm2_weights=w2,
            gemm2_weights_scale=s2,
            gemm2_bias=b2,
            intermediate_size=i_pad,
            valid_hidden_size=hidden,
            valid_intermediate_size=inter,
        )
        data, sf, xv = _rand_mxfp8(num_tokens, hidden, h1_pad, gen)
        ids = torch.zeros(num_tokens, 1, dtype=torch.int32, device=DEV)
        wts = torch.ones(num_tokens, 1, dtype=torch.bfloat16, device=DEV)
        alpha, beta, limit = _swiglu_params(num_experts)
        out = _call(
            data,
            sf,
            args,
            num_experts,
            1,
            topk_ids=ids,
            topk_weights=wts,
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=limit,
        ).float()

        up = xv @ _dequant(up_c[0], up_s[0]).t()
        gate = xv @ _dequant(gt_c[0], gt_s[0]).t()
        act = (
            (up.clamp(-7.0, 7.0) + 1.0)
            * gate.clamp(max=7.0)
            * torch.sigmoid(1.702 * gate.clamp(max=7.0))
        )
        blk_amax = act.reshape(num_tokens, -1, SV).abs().amax(dim=-1)
        mantissa = blk_amax / torch.exp2(torch.floor(torch.log2(blk_amax)))
        saturating = (mantissa > 1.75).float().mean().item()
        assert saturating > 0.05, f"only {saturating:.3f} of blocks exercise the saturating branch"
        recipe = _scale_recipe()
        assert torch.equal(out, _q_intermediate(act, recipe)), (
            f"the FC1 epilogue's MXFP8 requantization is not the {recipe!r} "
            "block scale with round-to-nearest-even and +-448 saturation. If "
            "this architecture's cubin changed recipe, probe it with an "
            "identity down-projection and record the new one in "
            "_SCALE_RECIPE_BY_SM -- do not widen a tolerance, the two recipes "
            "differ bit-exactly and every other cell's reference depends on "
            "which one is in force"
        )
        # The other architecture's recipe must NOT also fit, or this case does
        # not actually separate them and the bit-exact claim is vacuous.
        other = _OCP_SCALE if recipe == _ROUND_UP_SCALE else _ROUND_UP_SCALE
        assert not torch.equal(out, _q_intermediate(act, other)), (
            f"the {other!r} scale fits too — the case that separates the recipes is not covered"
        )
        assert not torch.equal(out, act.to(torch.bfloat16).float()), (
            "the intermediate was not requantized at all"
        )
    print("  test_intermediate_is_mxfp8_quantized OK")


def test_dspark_intermediate_is_mxfp8_quantized():
    """DeepSeek-V4.1 DSpark drafter geometry, plain clamped SwiGLU: the FC1
    epilogue's MXFP8 intermediate, read out bit-exactly.

    H = 5120, I = 2304 is not square, so FC2 is a *partial* identity
    (`down[i, i] = 1` for `i < I`): output columns `[0, I)` are the kernel's own
    intermediate and `[I, H)` must be zero. alpha/beta are left at `None` and
    the clamp limit is 10 -- the V4.1 `Expert` (gate <= 10, up in [-10, 10],
    `silu(gate) * up`). Two FC1 scales: one where the clamp bites on ~10% of
    gate and ~20% of up elements, one where the saturating-scale branch is hit
    by ~19% of blocks.
    """
    hidden, inter, num_experts, num_tokens = 5120, 2304, 2, 16
    for seed, shift in ((81, 1), (83, 0)):
        gen = torch.Generator(device=DEV).manual_seed(seed)
        up_p, up_s, up_c = _rand_mxfp4(num_experts, inter, hidden, gen, shift)
        gt_p, gt_s, gt_c = _rand_mxfp4(num_experts, inter, hidden, gen, shift)
        dn_c = torch.zeros(num_experts, hidden, inter, dtype=torch.uint8, device=DEV)
        dn_c[:, torch.arange(inter), torch.arange(inter)] = 2  # e2m1 code 2 = 1.0
        dn_p = (dn_c[..., 0::2] | (dn_c[..., 1::2] << 4)).contiguous()
        dn_s = torch.full((num_experts, hidden, inter // SV), 127, dtype=torch.uint8, device=DEV)
        zero1 = torch.zeros(num_experts, inter, device=DEV)
        zero2 = torch.zeros(num_experts, hidden, device=DEV)
        w1, s1, _ = _prep_fc1(up_p, gt_p, up_s, gt_s, zero1, zero1, inter, hidden)
        w2, s2, _ = _prep_fc2(dn_p, dn_s, zero2, hidden, inter)
        args = dict(
            gemm1_weights=w1,
            gemm1_weights_scale=s1,
            gemm1_bias=None,
            gemm2_weights=w2,
            gemm2_weights_scale=s2,
            gemm2_bias=None,
            intermediate_size=inter,
            valid_hidden_size=hidden,
            valid_intermediate_size=inter,
        )
        data, sf, xv = _rand_mxfp8(num_tokens, hidden, hidden, gen)
        ids = torch.zeros(num_tokens, 1, dtype=torch.int32, device=DEV)
        wts = torch.ones(num_tokens, 1, dtype=torch.bfloat16, device=DEV)
        limit = torch.full((num_experts,), 10.0, dtype=torch.float32, device=DEV)
        out = _call(
            data, sf, args, num_experts, 1, topk_ids=ids, topk_weights=wts, gemm1_clamp_limit=limit
        ).float()
        assert torch.count_nonzero(out[:, inter:]) == 0, "columns past I must be zero"
        up = xv @ _dequant(up_c[0], up_s[0]).t()
        gate = xv @ _dequant(gt_c[0], gt_s[0]).t()
        act = up.clamp(-10.0, 10.0) * torch.nn.functional.silu(gate.clamp(max=10.0))
        recipe = _scale_recipe()
        assert torch.equal(out[:, :inter], _q_intermediate(act, recipe)), (
            f"the FC1 epilogue's MXFP8 requantization at H=5120/I=2304 is not the {recipe!r} "
            "block scale; see test_intermediate_is_mxfp8_quantized"
        )
        other = _OCP_SCALE if recipe == _ROUND_UP_SCALE else _ROUND_UP_SCALE
        assert not torch.equal(out[:, :inter], _q_intermediate(act, other)), (
            f"the {other!r} scale fits too -- this case does not separate the recipes"
        )
        assert not torch.equal(out[:, :inter], act.to(torch.bfloat16).float())
        if shift == 1:
            assert (gate > 10.0).float().mean() > 0.05 and (
                up.abs() > 10.0
            ).float().mean() > 0.05, "the clamp limit does not bite in this case"
    print("  test_dspark_intermediate_is_mxfp8_quantized OK")


def _gate_rejects(y: torch.Tensor, ref: torch.Tensor) -> bool:
    try:
        _assert_moe_close(y, ref)
    except AssertionError:
        return True
    return False


def test_dspark_drafter_surface():
    """DeepSeek-V4.1 DSpark drafter MoE: H=5120, I=2304, 128 experts all local
    (offset 0), top-3, clamp limit 10 per expert, no alpha/beta, no biases,
    pre-routed bf16 weights; decode blocks T = 5..160 and prefill rows to 8192.

    Deliberately wrong references (no clamp, the gpt-oss clamp 7, gate/up
    swapped, the gpt-oss alpha/beta, no intermediate requantization) must all
    land outside the gate, and the checkpoint's own order -- routing weight
    multiplied into the activation before FC2's MXFP8 input quantization --
    is measured to be a different computation from this kernel's.
    """
    num_experts, hidden, inter, top_k = 128, 5120, 2304, 3
    args, ref = _build(num_experts, hidden, inter, seed=300, fc1_exp_shift=1, with_bias=False)
    gen = ref["gen"]
    assert ref["h1_pad"] == hidden and args["intermediate_size"] == inter, (
        "5120/2304 need no padding"
    )
    limit = torch.full((num_experts,), 10.0, dtype=torch.float32, device=DEV)
    for num_tokens in (1, 5, 40, 85, 160, 1024, 8192):
        data, sf, xv = _rand_mxfp8(num_tokens, hidden, hidden, gen)
        ids, _ = _routing(num_tokens, num_experts, top_k, gen)
        wts = (torch.rand(num_tokens, top_k, device=DEV, generator=gen) * 1.5).to(torch.bfloat16)
        out = _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts,
            gemm1_clamp_limit=limit,
        )
        assert out.shape == (num_tokens, hidden), out.shape
        stats = {}
        exp = _ref_moe(xv, ids, wts, ref, None, None, limit, stats=stats).to(torch.bfloat16)
        _assert_moe_close(out, exp)
        elt, rms = _dev(out, exp)
        print(
            f"    T={num_tokens}: {elt:.2f} ulp elt, {rms:.2f} ulp RMS; clamp changed "
            f"{stats['gate'] / stats['n']:.3f} of gate, {stats['up'] / stats['n']:.3f} of up"
        )
        if num_tokens >= 40:
            assert stats["gate"] / stats["n"] > 0.02 and stats["up"] / stats["n"] > 0.02, stats
        if num_tokens != 160:
            continue
        wrong = {
            "no clamp": _ref_moe(xv, ids, wts, ref, None, None, None),
            "clamp 7": _ref_moe(xv, ids, wts, ref, None, None, limit * 0.7),
            "gate/up swapped": _ref_moe(xv, ids, wts, ref, None, None, limit, swap_gate_up=True),
            "gpt-oss alpha/beta": _ref_moe(
                xv,
                ids,
                wts,
                ref,
                torch.full((num_experts,), 1.702, device=DEV),
                torch.ones(num_experts, device=DEV),
                limit,
            ),
            "no intermediate requant": _ref_moe(
                xv, ids, wts, ref, None, None, limit, quantize_intermediate=False
            ),
        }
        for name, bad in wrong.items():
            elt, rms = _dev(out, bad.to(torch.bfloat16))
            print(f"    WRONG {name}: {elt:.2f} ulp elt, {rms:.2f} ulp RMS")
            assert _gate_rejects(out, bad.to(torch.bfloat16)), (
                f"the {name} reference passed the gate"
            )
        # the checkpoint's order: weight * act -> bf16 -> MXFP8 -> FC2. The
        # kernel multiplies after FC2 instead; the two differ by a whole
        # requantization's worth of rounding (measured ~10 ulp RMS).
        ckpt = _ref_moe(xv, ids, wts, ref, None, None, limit, weights_before_down=True)
        elt, rms = _dev(out, ckpt.to(torch.bfloat16))
        print(f"    checkpoint order: {elt:.2f} ulp elt, {rms:.2f} ulp RMS")
        assert rms > 6.0, f"checkpoint-order reference unexpectedly close: {rms:.2f} ulp RMS"

    # fed by the certified quantizer: at H = 5120 alignment 512 adds no padding
    for num_tokens in (40, 160):
        xb = torch.randn(num_tokens, hidden, device=DEV, dtype=torch.bfloat16, generator=gen) * 2
        data, sf = torch.ops.trtllm.mxfp8_quantize(xb, False, 512)
        assert data.shape == (num_tokens, hidden) and sf.shape == (num_tokens * hidden // SV,)
        ids, _ = _routing(num_tokens, num_experts, top_k, gen)
        wts = (torch.rand(num_tokens, top_k, device=DEV, generator=gen) * 1.5).to(torch.bfloat16)
        out = _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts,
            gemm1_clamp_limit=limit,
        )
        exp = _ref_moe(_dequant_mxfp8(data, sf), ids, wts, ref, None, None, limit)
        _assert_moe_close(out, exp.to(torch.bfloat16))
        elt, rms = _dev(out, exp.to(torch.bfloat16))
        print(f"    mxfp8_quantize-fed T={num_tokens}: {elt:.2f} ulp elt, {rms:.2f} ulp RMS")

    # CUDA-graph capture at a decode size, replayed over new inputs written
    # into the captured buffers, equals an eager call on those inputs bitwise
    num_tokens = 40
    data, sf, _ = _rand_mxfp8(num_tokens, hidden, hidden, gen)
    ids, _ = _routing(num_tokens, num_experts, top_k, gen)
    wts = (torch.rand(num_tokens, top_k, device=DEV, generator=gen) * 1.5).to(torch.bfloat16)
    kw = dict(topk_ids=ids, topk_weights=wts, gemm1_clamp_limit=limit)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _call(data, sf, args, num_experts, top_k, **kw)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _call(data, sf, args, num_experts, top_k, **kw)
    data2, sf2, _ = _rand_mxfp8(num_tokens, hidden, hidden, gen)
    ids2, _ = _routing(num_tokens, num_experts, top_k, gen)
    wts2 = (torch.rand(num_tokens, top_k, device=DEV, generator=gen) * 1.5).to(torch.bfloat16)
    data.copy_(data2)
    sf.copy_(sf2)
    ids.copy_(ids2)
    wts.copy_(wts2)
    graph.replay()
    torch.cuda.synchronize()
    eager = _call(
        data2,
        sf2,
        args,
        num_experts,
        top_k,
        topk_ids=ids2,
        topk_weights=wts2,
        gemm1_clamp_limit=limit,
    )
    assert torch.equal(captured, eager), "graph replay over new inputs differs from eager"
    print("  test_dspark_drafter_surface OK")


# The DSpark drafter's MoE runs on its 5-token draft block: T = 5G for the
# drafter batch G = 1..32, inside the target's decode CUDA graphs (the capture
# set 5 * {1..8, 16, 24, 32} is a subset) and eagerly in mixed steps.
# Replicated on every rank, so no dep4 gather reaches it.
_DSPARK_BLOCK_TOKENS = tuple(range(5, 161, 5))


def _skewed_routing(num_tokens, num_experts, top_k, gen):
    """Every token routed to the same top_k experts (first, middle and last), in a per-token order:
    the most unbalanced routing a captured launch can meet."""
    hot = torch.tensor(
        [0, num_experts // 2, num_experts - 1][:top_k], dtype=torch.int32, device=DEV
    )
    ids = torch.stack(
        [hot[torch.randperm(top_k, device=DEV, generator=gen)] for _ in range(num_tokens)]
    )
    wts = (torch.rand(num_tokens, top_k, device=DEV, generator=gen) * 1.5).to(torch.bfloat16)
    return ids.contiguous(), wts


def test_dspark_every_block_size_eager_and_capture():
    """DSpark drafter MoE at every T = 5..160 step 5: eager against the reference, then captured.

    Per T: an eager call on uniform random routing matches the kernel-order
    reference. The call is then warmed up on a side stream and captured with
    the op's autotuner cold, so the kernel configuration the graph holds is the
    one chosen for this T at capture. Two replays follow, each after writing
    new activations and new routing into the captured buffers: new uniform
    routing, then every token on the same three experts (first, middle, last:
    the most unbalanced load). Each replay is bitwise equal to an eager call
    on the same inputs, and the skewed one also matches the reference.
    """
    num_experts, hidden, inter, top_k = 128, 5120, 2304, 3
    args, ref = _build(num_experts, hidden, inter, seed=400, fc1_exp_shift=1, with_bias=False)
    gen = ref["gen"]
    limit = torch.full((num_experts,), 10.0, dtype=torch.float32, device=DEV)
    worst = [0.0, 0.0]
    for num_tokens in _DSPARK_BLOCK_TOKENS:
        data, sf, xv = _rand_mxfp8(num_tokens, hidden, hidden, gen)
        ids, _ = _routing(num_tokens, num_experts, top_k, gen)
        wts = (torch.rand(num_tokens, top_k, device=DEV, generator=gen) * 1.5).to(torch.bfloat16)
        kw = dict(topk_ids=ids, topk_weights=wts, gemm1_clamp_limit=limit)
        out = _call(data, sf, args, num_experts, top_k, **kw)
        exp = _ref_moe(xv, ids, wts, ref, None, None, limit).to(torch.bfloat16)
        _assert_moe_close(out, exp)
        elt, rms = _dev(out, exp)
        worst = [max(worst[0], elt), max(worst[1], rms)]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _call(data, sf, args, num_experts, top_k, **kw)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _call(data, sf, args, num_experts, top_k, **kw)
        for skewed in (False, True):
            d2, s2, x2 = _rand_mxfp8(num_tokens, hidden, hidden, gen)
            if skewed:
                i2, w2 = _skewed_routing(num_tokens, num_experts, top_k, gen)
            else:
                i2, _ = _routing(num_tokens, num_experts, top_k, gen)
                w2 = (torch.rand(num_tokens, top_k, device=DEV, generator=gen) * 1.5).to(
                    torch.bfloat16
                )
            data.copy_(d2)
            sf.copy_(s2)
            ids.copy_(i2)
            wts.copy_(w2)
            captured.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            eager = _call(
                d2,
                s2,
                args,
                num_experts,
                top_k,
                topk_ids=i2,
                topk_weights=w2,
                gemm1_clamp_limit=limit,
            )
            assert torch.equal(captured, eager), (num_tokens, "skewed" if skewed else "uniform")
            if skewed:
                exp2 = _ref_moe(x2, i2, w2, ref, None, None, limit).to(torch.bfloat16)
                _assert_moe_close(captured, exp2)
        del graph, captured
    print(
        f"  test_dspark_every_block_size_eager_and_capture OK: worst {worst[0]:.2f} ulp elt, "
        f"{worst[1]:.2f} ulp RMS over T 5..160"
    )


def test_activation_variants():
    """alpha/beta/clamp_limit are independently optional and per-expert."""
    num_experts, hidden, inter, top_k, num_tokens = 8, 1024, 512, 3, 24
    args, ref = _build(num_experts, hidden, inter, seed=13)
    gen = ref["gen"]
    data, sf, xv = _rand_mxfp8(num_tokens, hidden, ref["h1_pad"], gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    one = torch.ones(num_experts, dtype=torch.float32, device=DEV)
    cases = [
        (None, None, None),
        (None, None, torch.full((num_experts,), 7.0, device=DEV)),
        (one * 1.702, one, None),
        (
            torch.linspace(1.0, 2.0, num_experts, device=DEV),
            torch.linspace(0.0, 1.5, num_experts, device=DEV),
            torch.linspace(2.0, 9.0, num_experts, device=DEV),
        ),
    ]
    for alpha, beta, limit in cases:
        out = _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts,
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=limit,
        )
        exp = _ref_moe(xv, ids, wts, ref, alpha, beta, limit).to(torch.bfloat16)
        _assert_moe_close(out, exp)
    print("  test_activation_variants OK")


def test_bias_combinations():
    """Each bias tensor is independently optional."""
    num_experts, hidden, inter, top_k, num_tokens = 8, 512, 256, 2, 16
    args, ref = _build(num_experts, hidden, inter, seed=14)
    gen = ref["gen"]
    data, sf, xv = _rand_mxfp8(num_tokens, hidden, ref["h1_pad"], gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    alpha, beta, limit = _swiglu_params(num_experts)
    for want1, want2 in ((True, True), (True, False), (False, True), (False, False)):
        sub = dict(args)
        sub_ref = dict(ref)
        if not want1:
            sub["gemm1_bias"] = None
            for role in ("up", "gate"):
                c, s, b = ref[role]
                sub_ref[role] = (c, s, torch.zeros_like(b))
        if not want2:
            sub["gemm2_bias"] = None
            c, s, b = ref["down"]
            sub_ref["down"] = (c, s, torch.zeros_like(b))
        out = _call(
            data,
            sf,
            sub,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts,
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=limit,
        )
        exp = _ref_moe(xv, ids, wts, sub_ref, alpha, beta, limit).to(torch.bfloat16)
        _assert_moe_close(out, exp)
    print("  test_bias_combinations OK")


def test_expert_counts_and_top_k():
    """Small expert counts and top_k = 1."""
    hidden, inter, num_tokens = 512, 128, 12
    for num_experts, top_k in ((2, 1), (3, 2), (5, 1), (16, 4)):
        args, ref = _build(num_experts, hidden, inter, seed=200 + num_experts)
        gen = ref["gen"]
        data, sf, xv = _rand_mxfp8(num_tokens, hidden, ref["h1_pad"], gen)
        ids, wts = _routing(num_tokens, num_experts, top_k, gen)
        alpha, beta, limit = _swiglu_params(num_experts)
        out = _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts,
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=limit,
        )
        exp = _ref_moe(xv, ids, wts, ref, alpha, beta, limit).to(torch.bfloat16)
        _assert_moe_close(out, exp)
    print("  test_expert_counts_and_top_k OK")


def test_expert_parallel_window():
    """local_expert_offset/local_num_experts select a global-id window; slots
    routed outside it contribute nothing."""
    num_experts, hidden, inter, top_k, num_tokens = 8, 512, 256, 4, 12
    args, ref = _build(num_experts, hidden, inter, seed=15)
    gen = ref["gen"]
    data, sf, xv = _rand_mxfp8(num_tokens, hidden, ref["h1_pad"], gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    offset, num_local = 4, 4
    sub = dict(args)
    for key in (
        "gemm1_weights",
        "gemm1_weights_scale",
        "gemm1_bias",
        "gemm2_weights",
        "gemm2_weights_scale",
        "gemm2_bias",
    ):
        sub[key] = args[key][offset : offset + num_local].contiguous()
    sub_ref = dict(ref)
    for role in ("up", "gate", "down"):
        c, s, b = ref[role]
        sub_ref[role] = (
            c[offset : offset + num_local],
            s[offset : offset + num_local],
            b[offset : offset + num_local],
        )
    alpha, beta, limit = _swiglu_params(num_local)
    kw = dict(
        local_expert_offset=offset,
        local_num_experts=num_local,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
    )
    out = _call(data, sf, sub, num_experts, top_k, topk_ids=ids, topk_weights=wts, **kw)
    exp = _ref_moe(
        xv, ids, wts, sub_ref, alpha, beta, limit, offset=offset, num_local=num_local
    ).to(torch.bfloat16)
    _assert_moe_close(out, exp)

    # ids outside the window (including negative and >= num_experts) are dropped
    stray = ids.clone()
    stray[0, 0] = 999
    stray[1, 1] = -1
    out = _call(data, sf, sub, num_experts, top_k, topk_ids=stray, topk_weights=wts, **kw)
    exp = _ref_moe(
        xv, stray, wts, sub_ref, alpha, beta, limit, offset=offset, num_local=num_local
    ).to(torch.bfloat16)
    _assert_moe_close(out, exp)
    print("  test_expert_parallel_window OK")


def test_output_buffer():
    """`output=` writes in place, returns an empty tensor, touches no extra row."""
    num_experts, hidden, inter, top_k, num_tokens = 8, 512, 256, 2, 10
    args, ref = _build(num_experts, hidden, inter, seed=16)
    gen = ref["gen"]
    data, sf, _ = _rand_mxfp8(num_tokens, hidden, ref["h1_pad"], gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    alpha, beta, limit = _swiglu_params(num_experts)
    kw = dict(
        topk_ids=ids,
        topk_weights=wts,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
    )
    fresh = _call(data, sf, args, num_experts, top_k, **kw)

    buf = torch.full((num_tokens, hidden), -3.0, device=DEV, dtype=torch.bfloat16)
    ret = _call(data, sf, args, num_experts, top_k, output=buf, **kw)
    assert ret.numel() == 0 and ret.dtype == torch.bfloat16, (ret.shape, ret.dtype)
    assert torch.equal(buf, fresh)

    big = torch.full((num_tokens + 3, hidden), -3.0, device=DEV, dtype=torch.bfloat16)
    snap = big.clone()
    view = big[:num_tokens]
    assert view.is_contiguous()
    _call(data, sf, args, num_experts, top_k, output=view, **kw)
    assert torch.equal(big[:num_tokens], fresh)
    assert torch.equal(big[num_tokens:], snap[num_tokens:]), "wrote past the requested rows"
    print("  test_output_buffer OK")


def test_shapes():
    """Hidden/intermediate sizes on and off the padding boundaries."""
    for hidden, inter in ((512, 128), (640, 128), (2048, 512), (2880, 1024)):
        num_experts, top_k, num_tokens = 4, 2, 8
        args, ref = _build(num_experts, hidden, inter, seed=17)
        gen = ref["gen"]
        data, sf, xv = _rand_mxfp8(num_tokens, hidden, ref["h1_pad"], gen)
        ids, wts = _routing(num_tokens, num_experts, top_k, gen)
        alpha, beta, limit = _swiglu_params(num_experts)
        out = _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts,
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=limit,
        )
        assert out.shape == (num_tokens, hidden), out.shape
        exp = _ref_moe(xv, ids, wts, ref, alpha, beta, limit).to(torch.bfloat16)
        _assert_moe_close(out, exp)

    # token counts, including one past the default autotuner bucket cap
    num_experts, hidden, inter, top_k = 8, 512, 256, 4
    args, ref = _build(num_experts, hidden, inter, seed=18)
    gen = ref["gen"]
    alpha, beta, limit = _swiglu_params(num_experts)
    for num_tokens in (1, 2, 17, 1024, 8192):
        data, sf, xv = _rand_mxfp8(num_tokens, hidden, ref["h1_pad"], gen)
        ids, wts = _routing(num_tokens, num_experts, top_k, gen)
        out = _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts,
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=limit,
        )
        exp = _ref_moe(xv, ids, wts, ref, alpha, beta, limit).to(torch.bfloat16)
        _assert_moe_close(out, exp)
    print("  test_shapes OK")


def test_valid_sizes():
    """valid_hidden_size is the output width — it does not follow the widened
    activation; valid_intermediate_size bounds the intermediate columns read."""
    num_experts, hidden, inter, top_k, num_tokens = 4, 2880, 2880, 2, 8
    args, ref = _build(num_experts, hidden, inter, seed=19)
    gen = ref["gen"]
    data, sf, xv = _rand_mxfp8(num_tokens, hidden, ref["h1_pad"], gen)
    assert data.shape[1] == 3072, data.shape
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    alpha, beta, limit = _swiglu_params(num_experts)
    kw = dict(
        topk_ids=ids,
        topk_weights=wts,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
    )
    exp = _ref_moe(xv, ids, wts, ref, alpha, beta, limit).to(torch.bfloat16)

    out = _call(data, sf, args, num_experts, top_k, **kw)
    assert out.shape == (num_tokens, 2880)
    _assert_moe_close(out, exp)

    # the padded row band [2880, 2944) is all-zero weight, so widening the
    # output to the padded hidden must reproduce the same values plus zeros
    wide = _call(data, sf, args, num_experts, top_k, valid_hidden_size=2944, **kw)
    assert wide.shape == (num_tokens, 2944)
    _assert_moe_close(wide[:, :2880].contiguous(), exp)
    assert torch.count_nonzero(wide[:, 2880:]) == 0

    # ... but it never follows the widened activation: None means "use the
    # hidden_states width" (3072 here), and that width is not an output width
    for bad in (None, 3072):
        try:
            _call(data, sf, args, num_experts, top_k, valid_hidden_size=bad, **kw)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"valid_hidden_size={bad} was accepted")

    # padded intermediate columns carry zero weight, so any value at or above
    # the true intermediate size is equivalent
    for vis in (2880, 2944, None):
        same = _call(data, sf, args, num_experts, top_k, valid_intermediate_size=vis, **kw)
        _assert_moe_close(same, exp)

    # below the true intermediate size the kernel silently drops columns
    short = _call(data, sf, args, num_experts, top_k, valid_intermediate_size=1024, **kw)
    elt, _ = _dev(short, exp)
    assert elt > 50.0, f"truncating the intermediate should change the result, got {elt:.2f} ulp"
    print("  test_valid_sizes OK")


def test_hidden_padding_is_inert():
    """Columns of hidden_states past valid_hidden_size multiply zero weight."""
    num_experts, hidden, inter, top_k, num_tokens = 4, 2880, 1024, 2, 8
    args, ref = _build(num_experts, hidden, inter, seed=20)
    gen = ref["gen"]
    data, sf, _ = _rand_mxfp8(num_tokens, hidden, ref["h1_pad"], gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    alpha, beta, limit = _swiglu_params(num_experts)
    kw = dict(
        topk_ids=ids,
        topk_weights=wts,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
    )
    base = _call(data, sf, args, num_experts, top_k, **kw)

    # arbitrary finite junk in the padded columns, with a live scale byte
    junk = data.clone().float()
    junk[:, hidden:] = 400.0
    junk = junk.to(torch.float8_e4m3fn)
    junk_sf = sf.view(num_tokens, -1).clone()
    junk_sf[:, hidden // SV :] = 133
    assert torch.equal(
        _call(junk, junk_sf.reshape(-1).contiguous(), args, num_experts, top_k, **kw),
        base,
    )

    # ... but a NaN there still poisons the result
    nan = data.clone().float()
    nan[:, hidden:] = float("nan")
    nan = nan.to(torch.float8_e4m3fn)
    assert not torch.isfinite(
        _call(nan, junk_sf.reshape(-1).contiguous(), args, num_experts, top_k, **kw)
    ).all()
    print("  test_hidden_padding_is_inert OK")


def test_duplicate_expert_id_counts_once():
    """A repeated id in one token's row contributes once, at the first slot's weight."""
    num_experts, hidden, inter, top_k, num_tokens = 8, 512, 256, 4, 6
    args, ref = _build(num_experts, hidden, inter, seed=21)
    gen = ref["gen"]
    data, sf, xv = _rand_mxfp8(num_tokens, hidden, ref["h1_pad"], gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    alpha, beta, limit = _swiglu_params(num_experts)
    ids[2, 1] = ids[2, 0]
    out = _call(
        data,
        sf,
        args,
        num_experts,
        top_k,
        topk_ids=ids,
        topk_weights=wts,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
    )
    first_only = wts.clone()
    first_only[2, 1] = 0.0
    _assert_moe_close(
        out, _ref_moe(xv, ids, first_only, ref, alpha, beta, limit).to(torch.bfloat16)
    )
    elt, _ = _dev(out, _ref_moe(xv, ids, wts, ref, alpha, beta, limit).to(torch.bfloat16))
    assert elt > 10.0, f"counted-twice reference should be rejected, got {elt:.2f} ulp"
    print("  test_duplicate_expert_id_counts_once OK")


def test_inert_arguments():
    """Knobs the contract holds inert really are inert on the certified paths."""
    num_experts, hidden, inter, top_k, num_tokens = 8, 512, 256, 4, 12
    args, ref = _build(num_experts, hidden, inter, seed=202)
    gen = ref["gen"]
    data, sf, _ = _rand_mxfp8(num_tokens, hidden, ref["h1_pad"], gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    lg = _logits(num_tokens, num_experts, gen)
    alpha, beta, limit = _swiglu_params(num_experts)
    kw = dict(
        topk_ids=ids,
        topk_weights=wts,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
    )
    base = _call(data, sf, args, num_experts, top_k, **kw)

    # routing_method_type is not read once routing is already done
    for rmt in (0, 1, 2, 4, 5, 6):
        assert torch.equal(
            _call(data, sf, args, num_experts, top_k, routing_method_type=rmt, **kw),
            base,
        ), f"routing_method_type={rmt} changed the pre-routed result"

    # given both entry points, the pre-routed pair wins and the logits are dead
    assert torch.equal(_call(data, sf, args, num_experts, top_k, routing_logits=lg, **kw), base), (
        "routing_logits changed the result although topk_ids/topk_weights were given"
    )

    # routed_scaling_factor does nothing on either certified entry point
    for rsf in (1.0, 2.5):
        assert torch.equal(
            _call(data, sf, args, num_experts, top_k, routed_scaling_factor=rsf, **kw),
            base,
        )
    routed_kw = dict(gemm1_alpha=alpha, gemm1_beta=beta, gemm1_clamp_limit=limit, routing_logits=lg)
    routed = _call(data, sf, args, num_experts, top_k, **routed_kw)
    assert torch.equal(
        _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            routed_scaling_factor=2.5,
            **routed_kw,
        ),
        routed,
    )

    # autotuner bucket knobs never move the numbers
    for tmax in (128, 8192):
        assert torch.equal(
            _call(data, sf, args, num_experts, top_k, tune_max_num_tokens=tmax, **kw),
            base,
        )
    for dp in (False, True):
        assert torch.equal(_call(data, sf, args, num_experts, top_k, use_dp=dp, **kw), base)
    print("  test_inert_arguments OK")


def test_rejects_unsupported():
    """Domains the contract declares unsupported are rejected loudly."""
    num_experts, hidden, inter, top_k, num_tokens = 8, 512, 256, 4, 6
    args, ref = _build(num_experts, hidden, inter, seed=23)
    gen = ref["gen"]
    data, sf, _ = _rand_mxfp8(num_tokens, hidden, ref["h1_pad"], gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    alpha, beta, limit = _swiglu_params(num_experts)
    act_kw = dict(gemm1_alpha=alpha, gemm1_beta=beta, gemm1_clamp_limit=limit)
    kw = dict(topk_ids=ids, topk_weights=wts, **act_kw)

    def rejected(name, fn):
        try:
            fn()
        except (RuntimeError, AssertionError):
            return
        raise AssertionError(f"{name} was accepted")

    # only the gated SwiGlu kernel family exists for MXFP8 x MXFP4
    for act in (1, 2):
        rejected(
            f"act_type={act}",
            lambda a=act: _call(data, sf, args, num_experts, top_k, act_type=a, **kw),
        )
    rejected(
        "int64 topk_ids",
        lambda: _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids.long(),
            topk_weights=wts,
            **act_kw,
        ),
    )
    rejected(
        "fp32 topk_weights",
        lambda: _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts.float(),
            **act_kw,
        ),
    )
    rejected(
        "no routing input",
        lambda: _call(data, sf, args, num_experts, top_k, **act_kw),
    )
    rejected(
        "top_k == num_experts",
        lambda: _call(
            data,
            sf,
            args,
            num_experts,
            num_experts,
            topk_ids=torch.stack(
                [torch.arange(num_experts, device=DEV, dtype=torch.int32)] * num_tokens
            ),
            topk_weights=torch.ones(num_tokens, num_experts, device=DEV, dtype=torch.bfloat16),
            **act_kw,
        ),
    )
    rejected(
        "top_k = 0",
        lambda: _call(
            data,
            sf,
            args,
            num_experts,
            0,
            topk_ids=ids[:, :0].contiguous(),
            topk_weights=wts[:, :0].contiguous(),
            **act_kw,
        ),
    )
    rejected(
        "bf16 gemm1_bias",
        lambda: _call(
            data,
            sf,
            {**args, "gemm1_bias": args["gemm1_bias"].bfloat16()},
            num_experts,
            top_k,
            **kw,
        ),
    )
    rejected(
        "int8-typed weight scale",
        lambda: _call(
            data,
            sf,
            {
                **args,
                "gemm1_weights_scale": args["gemm1_weights_scale"].view(torch.int8),
            },
            num_experts,
            top_k,
            **kw,
        ),
    )
    rejected(
        "alpha sized 1 instead of local_num_experts",
        lambda: _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts,
            gemm1_alpha=alpha[:1].contiguous(),
            gemm1_beta=beta,
            gemm1_clamp_limit=limit,
        ),
    )
    rejected(
        "fp32 output buffer",
        lambda: _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            output=torch.zeros(num_tokens, hidden, device=DEV, dtype=torch.float32),
            **kw,
        ),
    )
    rejected(
        "wrong-shape output buffer",
        lambda: _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            output=torch.zeros(num_tokens, hidden + 8, device=DEV, dtype=torch.bfloat16),
            **kw,
        ),
    )
    rejected(
        "valid_hidden_size not a multiple of 32",
        lambda: _call(data, sf, args, num_experts, top_k, valid_hidden_size=500, **kw),
    )
    rejected(
        "intermediate_size != gemm1_weights.shape[1] // 2",
        lambda: _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            intermediate_size=args["intermediate_size"] // 2,
            **kw,
        ),
    )
    rejected(
        "routing_logits column count != num_experts",
        lambda: _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            routing_logits=torch.zeros(
                num_tokens, num_experts + 4, device=DEV, dtype=torch.bfloat16
            ),
            **act_kw,
        ),
    )
    rejected(
        "3-D hidden_states",
        lambda: _call(data.view(1, num_tokens, -1), sf, args, num_experts, top_k, **kw),
    )
    rejected(
        "zero tokens",
        lambda: _call(
            data[:0].contiguous(),
            sf[:0].contiguous(),
            args,
            num_experts,
            top_k,
            topk_ids=torch.zeros(0, top_k, device=DEV, dtype=torch.int32),
            topk_weights=torch.zeros(0, top_k, device=DEV, dtype=torch.bfloat16),
            **act_kw,
        ),
    )
    print("  test_rejects_unsupported OK")


def test_wrapper_guards():
    """The wrapper's asserts cover cases the op itself takes silently."""
    num_experts, hidden, inter, top_k, num_tokens = 8, 512, 256, 4, 6
    args, ref = _build(num_experts, hidden, inter, seed=24)
    gen = ref["gen"]
    data, sf, _ = _rand_mxfp8(num_tokens, hidden, ref["h1_pad"], gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    lg = _logits(num_tokens, num_experts, gen)
    alpha, beta, limit = _swiglu_params(num_experts)
    kw = dict(
        topk_ids=ids,
        topk_weights=wts,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
    )
    base = _call(data, sf, args, num_experts, top_k, **kw)

    def raw(**over):
        call = dict(
            routing_logits=None,
            routing_bias=None,
            hidden_states=data,
            hidden_states_scale=sf,
            gemm1_weights=args["gemm1_weights"],
            gemm1_weights_scale=args["gemm1_weights_scale"],
            gemm1_bias=args["gemm1_bias"],
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=limit,
            gemm2_weights=args["gemm2_weights"],
            gemm2_weights_scale=args["gemm2_weights_scale"],
            gemm2_bias=args["gemm2_bias"],
            topk_weights=wts,
            topk_ids=ids,
            output=None,
            routing_method_type=1,
        )
        call.update(over)
        return torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner(
            call["routing_logits"],
            call["routing_bias"],
            call["hidden_states"],
            call["hidden_states_scale"],
            call["gemm1_weights"],
            call["gemm1_weights_scale"],
            call["gemm1_bias"],
            call["gemm1_alpha"],
            call["gemm1_beta"],
            call["gemm1_clamp_limit"],
            call["gemm2_weights"],
            call["gemm2_weights_scale"],
            call["gemm2_bias"],
            num_experts,
            top_k,
            None,
            None,
            args["intermediate_size"],
            args["valid_hidden_size"],
            args["valid_intermediate_size"],
            0,
            num_experts,
            None,
            call["routing_method_type"],
            0,
            topk_weights=call["topk_weights"],
            topk_ids=call["topk_ids"],
            output=call["output"],
        )

    def strided(t, extra=4):
        fat = torch.zeros(*t.shape[:-1], t.shape[-1] + extra, dtype=t.dtype, device=DEV)
        fat[..., : t.shape[-1]] = t
        view = fat[..., : t.shape[-1]]
        assert not view.is_contiguous()
        return view

    # a 1-D scale buffer keeps its element count but reads every other byte
    spread = torch.zeros(sf.numel() * 2, dtype=torch.uint8, device=DEV)
    spread[::2] = sf
    sf_strided = torch.as_strided(spread, (sf.numel(),), (2,))
    assert not sf_strided.is_contiguous()

    # every tensor the wrapper guards: strided is silently honoured by the op
    for name in (
        "hidden_states",
        "hidden_states_scale",
        "gemm1_weights",
        "gemm1_weights_scale",
        "gemm1_bias",
        "gemm2_weights",
        "gemm2_weights_scale",
        "gemm2_bias",
        "topk_weights",
        "topk_ids",
    ):
        src = {
            "hidden_states": data,
            "hidden_states_scale": sf,
            "topk_weights": wts,
            "topk_ids": ids,
        }.get(name, args.get(name))
        bad = sf_strided if name == "hidden_states_scale" else strided(src)
        got = raw(**{name: bad})
        assert not torch.equal(got, base), f"{name}: strided view was NOT silently wrong"
        if name == "hidden_states":
            guarded = lambda: _call(bad, sf, args, num_experts, top_k, **kw)  # noqa: E731
        elif name == "hidden_states_scale":
            guarded = lambda: _call(data, bad, args, num_experts, top_k, **kw)  # noqa: E731
        elif name in ("topk_weights", "topk_ids"):
            guarded = lambda: _call(  # noqa: E731
                data, sf, args, num_experts, top_k, **{**kw, name: bad}
            )
        else:
            guarded = lambda: _call(  # noqa: E731
                data, sf, {**args, name: bad}, num_experts, top_k, **kw
            )
        try:
            guarded()
        except AssertionError:
            pass
        else:
            raise AssertionError(f"wrapper did not reject a strided {name}")

    # a strided routing_logits is likewise silently honoured
    got = raw(routing_logits=strided(lg), topk_weights=None, topk_ids=None)
    ok = raw(routing_logits=lg, topk_weights=None, topk_ids=None)
    assert not torch.equal(got, ok)

    # a strided output buffer is written as if dense
    buf = torch.zeros(num_tokens, hidden + 4, device=DEV, dtype=torch.bfloat16)
    raw(output=buf[:, :hidden])
    assert not torch.equal(buf[:, :hidden], base)

    # routing_bias is a no-op for every non-grouped routing method
    bias = torch.zeros(num_experts, dtype=torch.float32, device=DEV)
    bias[0], bias[num_experts - 1] = 1.0e3, -1.0e3
    for rmt in (0, 1, 4, 6):
        with_bias = raw(
            routing_logits=lg,
            routing_bias=bias,
            topk_weights=None,
            topk_ids=None,
            routing_method_type=rmt,
        )
        without = raw(
            routing_logits=lg,
            routing_bias=None,
            topk_weights=None,
            topk_ids=None,
            routing_method_type=rmt,
        )
        assert torch.equal(with_bias, without), f"routing_bias was honoured at rmt={rmt}"
        try:
            _call(
                data,
                sf,
                args,
                num_experts,
                top_k,
                routing_logits=lg,
                routing_bias=bias,
                routing_method_type=rmt,
                gemm1_alpha=alpha,
                gemm1_beta=beta,
                gemm1_clamp_limit=limit,
            )
        except AssertionError:
            pass
        else:
            raise AssertionError(f"wrapper did not reject routing_bias at rmt={rmt}")
    # ... and on the pre-routed entry point, whatever the routing method
    try:
        _call(
            data,
            sf,
            args,
            num_experts,
            top_k,
            routing_bias=bias,
            routing_method_type=2,
            **kw,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("wrapper did not reject routing_bias on the pre-routed path")

    # folding the bias into the logits is the caller's job and does change the result
    shifted = _call(
        data,
        sf,
        args,
        num_experts,
        top_k,
        routing_logits=(lg.float() + bias).to(torch.bfloat16).contiguous(),
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
    )
    assert not torch.equal(shifted, ok)
    print("  test_wrapper_guards OK")


def test_determinism_and_purity():
    """Two identical calls agree bitwise; no input is mutated."""
    num_experts, hidden, inter, top_k, num_tokens = 8, 512, 256, 4, 32
    args, ref = _build(num_experts, hidden, inter, seed=25)
    gen = ref["gen"]
    data, sf, _ = _rand_mxfp8(num_tokens, hidden, ref["h1_pad"], gen)
    ids, wts = _routing(num_tokens, num_experts, top_k, gen)
    alpha, beta, limit = _swiglu_params(num_experts)
    kw = dict(
        topk_ids=ids,
        topk_weights=wts,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
    )
    snaps = {k: v.clone() for k, v in args.items() if torch.is_tensor(v)}
    data_snap, sf_snap = data.clone(), sf.clone()
    ids_snap, wts_snap = ids.clone(), wts.clone()
    a = _call(data, sf, args, num_experts, top_k, **kw)
    b = _call(data, sf, args, num_experts, top_k, **kw)
    assert torch.equal(a, b)
    for k, v in snaps.items():
        assert torch.equal(args[k], v), f"{k} was mutated"
    assert torch.equal(data, data_snap) and torch.equal(sf, sf_snap)
    assert torch.equal(ids, ids_snap) and torch.equal(wts, wts_snap)
    print("  test_determinism_and_purity OK")


# ── the DSpark drafter at max_batch_size 256 ─────────────────────────────
#
# The drafter runs this op once per stage on its N = 5G block rows, G the
# number of drafting requests on the rank (at most max_batch_size = 256), so
# N <= 1280 and N is always a multiple of 5. Under CUDA graphs G is a captured
# batch size: every G in Gset, the integers 1..32 plus TRT-LLM's padded list
# for max_batch_size 256. The target calls the op inside the autotuner's
# `fallback_tactic()` scope, in which `AutoTuner.choose_one`
# returns the untuned fallback without reading the profiling cache, so every
# rank takes the same tactic; the tests below call it exactly that way and
# record the tactic the AutoTuner hands the op on every call.

_OP = "trtllm::mxe4m3_mxe2m1_block_scale_moe_runner"
_DSPARK_GSET = tuple(range(1, 33)) + (40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 192, 256)
_DSPARK_CAPTURE_N = tuple(5 * g for g in _DSPARK_GSET)
_DSPARK_N_MAX = 5 * 256
# Every drafter-MoE row count the runtime-value dumps of the max_batch_size
# 256 engines recorded (padded CUDA-graph, eager, tp4 and dep4 legs) plus the
# long-context legs.
_DSPARK_DUMPED_N = (
    5, 10, 15, 20, 40, 50, 80, 90, 110, 115, 120, 125, 140, 160, 190, 200, 240, 275, 280, 320,
    360, 370, 385, 400, 440, 455, 480, 520, 560, 580, 585, 600, 640, 650, 665, 710, 715, 720,
    960, 1280,
)  # fmt: skip


def _drafter_layer(seed):
    """The drafter's routed experts: 128 local, H 5120, I 2304, clamp 10 per expert, no biases."""
    args, ref = _build(128, 5120, 2304, seed=seed, fc1_exp_shift=1, with_bias=False)
    limit = torch.full((128,), 10.0, dtype=torch.float32, device=DEV)
    return args, ref, limit


def _pinned(data, sf, args, num_experts, top_k, **kw):
    """The call as the target's drafter makes it: inside `fallback_tactic()`."""
    with fallback_tactic():
        return _call(data, sf, args, num_experts, top_k, **kw)


@contextlib.contextmanager
def _tactics():
    """Record `(tactic, tuning mode, capturing)` for every AutoTuner.choose_one call this op makes."""
    tuner = AutoTuner.get()
    real = tuner.choose_one
    rec = []

    def spy(custom_op, runners, tuning_config, inputs, **kw):
        out = real(custom_op, runners, tuning_config, inputs, **kw)
        if custom_op == _OP:
            rec.append((out[1], tuner.is_tuning_mode, torch.cuda.is_current_stream_capturing()))
        return out

    tuner.choose_one = spy
    try:
        yield rec
    finally:
        del tuner.choose_one


def _op_cache_entries():
    return len(AutoTuner.get().profiling_cache.get_specific_custom_op(_OP))


def _kernel_map(run, rows) -> dict:
    """{N: the device activity one call at N launched, in launch order}.

    Each call runs in its own `record_function` range, bracketed by two
    synchronizations, so the device events that start inside the range are
    exactly the ones that call launched (the range's own device-side
    annotation is excluded by name).
    """
    names = {}
    for i in range(0, len(rows), 256):
        chunk = rows[i : i + 256]
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for n in chunk:
                with record_function(f"N={n}"):
                    run(n)
                    torch.cuda.synchronize()
        evs = prof.events()
        ranges = [
            (int(e.name[2:]), e.time_range.start, e.time_range.end)
            for e in evs
            if e.name.startswith("N=") and e.device_type == torch.autograd.DeviceType.CPU
        ]
        device = sorted(
            (e.time_range.start, e.name)
            for e in evs
            if e.device_type == torch.autograd.DeviceType.CUDA and not e.name.startswith("N=")
        )
        for n, lo, hi in ranges:
            names[n] = tuple(nm for t, nm in device if lo <= t <= hi)
        del prof
    return names


def _runs(names, rows):
    """Maximal runs `[first N, last N, activity]` of consecutive N that launch the same device activity."""
    runs = []
    for n in rows:
        assert names.get(n), f"no device activity attributed to the call at N={n}"
        if runs and runs[-1][2] == names[n]:
            runs[-1][1] = n
        else:
            runs.append([n, n, names[n]])
    return runs


def test_dspark_eager_every_row_count_to_1280():
    """Every drafter row count N = 1..1280, called as the target calls it, against the reference.

    One 1280-row activation and uniform random top-3 routing; the call at N
    reads their first N rows and writes the first N rows of a NaN-poisoned
    buffer (the target passes `output=`). Every N is held to the reference's
    first N rows at the entry's gates, and a second call on the same inputs
    must be bitwise equal to the first (every rank of the replicated drafter
    computes the same rows). Every call is pinned as in the target and must
    take the untuned fallback tactic.

    The cold tactic picks its kernels by N (the tokens-per-expert tile and the
    default GEMM configuration, plus the routing kernels), so the test then
    profiles every call 1..1280, derives the kernel map (maximal runs of
    consecutive N launching the same device activity), and checks the first,
    middle and last N of every run again with every token routed to the same
    three experts (the most unbalanced load), against the reference (every
    switch point at its edges).
    """
    num_experts, hidden, top_k = 128, 5120, 3
    args, ref, limit = _drafter_layer(seed=500)
    gen = ref["gen"]
    n_max, sfw = _DSPARK_N_MAX, hidden // SV
    data, sf, xv = _rand_mxfp8(n_max, hidden, hidden, gen)
    ids, _ = _routing(n_max, num_experts, top_k, gen)
    wts = (torch.rand(n_max, top_k, device=DEV, generator=gen) * 1.5).to(torch.bfloat16)
    exp = _ref_moe(xv, ids, wts, ref, None, None, limit).to(torch.bfloat16)
    buf = torch.empty(n_max, hidden, dtype=torch.bfloat16, device=DEV)

    def run(n, out=None, i=None, w=None):
        out = buf[:n] if out is None else out
        _pinned(
            data[:n],
            sf[: n * sfw],
            args,
            num_experts,
            top_k,
            topk_ids=ids[:n] if i is None else i,
            topk_weights=wts[:n] if w is None else w,
            gemm1_clamp_limit=limit,
            output=out,
        )
        return out

    assert _op_cache_entries() == 0, (
        "a cached tuned entry would move the pinned call off the fallback"
    )
    with _tactics() as rec:
        full = run(
            n_max, torch.full((n_max, hidden), float("nan"), dtype=torch.bfloat16, device=DEV)
        )
        _assert_moe_close(full, exp)
        worst, off_prefix = [0.0, 0.0], []
        for n in range(1, n_max + 1):
            buf[:n].fill_(float("nan"))
            out = run(n)
            _assert_moe_close(out, exp[:n])
            elt, rms = _dev(out, exp[:n])
            worst = [max(worst[0], elt), max(worst[1], rms)]
            again = run(n, torch.full((n, hidden), float("nan"), dtype=torch.bfloat16, device=DEV))
            assert torch.equal(again, out), f"two identical calls at N={n} differ"
            if not torch.equal(out, full[:n]):
                off_prefix.append(n)
        calls = len(rec)
        rows = list(range(1, n_max + 1))
        names = _kernel_map(run, rows)
        runs = _runs(names, rows)
        edges = sorted({m for lo, hi, _ in runs for m in (lo, (lo + hi) // 2, hi)})
        edge_worst = [0.0, 0.0]
        for n in edges:
            i2, w2 = _skewed_routing(n, num_experts, top_k, gen)
            out = run(
                n, torch.full((n, hidden), float("nan"), dtype=torch.bfloat16, device=DEV), i2, w2
            )
            exp2 = _ref_moe(xv[:n], i2, w2, ref, None, None, limit).to(torch.bfloat16)
            _assert_moe_close(out, exp2)
            elt, rms = _dev(out, exp2)
            edge_worst = [max(edge_worst[0], elt), max(edge_worst[1], rms)]
    assert calls == 1 + 2 * n_max, calls
    assert {r[0] for r in rec} == {-1}, sorted({str(r[0]) for r in rec})
    assert _op_cache_entries() == 0
    assert set(_DSPARK_DUMPED_N) <= set(rows)
    # The cold path's kernel map as measured on sm_103: at 2 the FC1/FC2
    # configurations change (N = 1 runs FC1 with a static scheduler and FC2
    # split-K 2); the routing kernel changes at 5 (block -> dynamic block),
    # 17, 65 and 129 (cluster kernel sizes 256 / 512 / 1024); the finalize
    # kernel at 60 (vectorized loads); and the tokens-per-expert tile
    # (clamp(next_pow2(floor(3N / 128)), 8, 256)) goes 8 -> 16 at 384 and
    # 16 -> 32 at 726. The contract lists these; a move means both are stale.
    assert [lo for lo, _, _ in runs] == [1, 2, 5, 17, 60, 65, 129, 384, 726], [
        lo for lo, _, _ in runs
    ]
    # Rows do not depend on N: every N >= 2 is bitwise the first N rows of
    # the N = 1280 call; N = 1 differs only through its split-K FC2.
    assert off_prefix == [1], off_prefix[:20]
    kinds = sorted({nm for _, _, nm in runs})
    print(
        f"  test_dspark_eager_every_row_count_to_1280: {n_max} row counts in the gates, worst "
        f"{worst[0]:.2f} ulp elt / {worst[1]:.2f} ulp RMS; repeat calls bitwise equal; {len(rec)} calls "
        f"all on tactic -1; not bitwise equal to the first N rows of the N={n_max} call at "
        f"{len(off_prefix)} N (first {off_prefix[:12]})"
    )
    print(
        f"    kernel map: {len(runs)} runs of {len(kinds)} distinct sequences over N 1..{n_max}; switch "
        f"points (first N of each run) {[lo for lo, _, _ in runs]}; {len(edges)} run edges/middles on "
        f"skewed routing: worst {edge_worst[0]:.2f} ulp elt / {edge_worst[1]:.2f} ulp RMS"
    )
    prev = ()
    for lo, hi, nm in runs:
        changed = [x for x in nm if x not in prev]
        print(
            f"    N {lo}..{hi}: {len(nm)} kernels; new vs previous run: {' || '.join(x[:400] for x in changed)}"
        )
        prev = nm


def test_dspark_capture_every_5G_over_gset():
    """Capture at every drafter block size N = 5G for G in Gset (46 values, 5..1280), as the target runs it.

    Per N, in the engine's order: an eager call at that shape (the warm-up,
    checked against the reference), a side-stream warm-up, then the capture
    of the pinned call writing into an `output` allocated inside the capture
    (as the target's `_moe` does). Every other capture is made while an
    outer `autotune()` is open (the engine's tuning pass also runs the
    drafter). Two replays follow, each after writing new activations and new
    routing into the captured buffers -- uniform random routing, then every
    token on the same three experts -- with the output poisoned first. Each
    replay must be bitwise equal to an eager pinned call on the same inputs,
    inside the gates against the reference, and differ from the capture-time
    output on more than 90% of elements (a replay that reran on the old
    inputs is visible). Every call, captured ones included, must get the
    fallback tactic, and the op's AutoTuner cache must stay empty.
    """
    num_experts, hidden, top_k = 128, 5120, 3
    args, ref, limit = _drafter_layer(seed=600)
    gen = ref["gen"]
    assert len(_DSPARK_CAPTURE_N) == 46 and max(_DSPARK_CAPTURE_N) == _DSPARK_N_MAX
    assert _op_cache_entries() == 0
    worst, stale_min = [0.0, 0.0], 1.0
    with _tactics() as rec:
        for i, n in enumerate(_DSPARK_CAPTURE_N):
            data, sf, xv = _rand_mxfp8(n, hidden, hidden, gen)
            ids, _ = _routing(n, num_experts, top_k, gen)
            wts = (torch.rand(n, top_k, device=DEV, generator=gen) * 1.5).to(torch.bfloat16)
            kw = dict(topk_ids=ids, topk_weights=wts, gemm1_clamp_limit=limit)
            warm = torch.full((n, hidden), float("nan"), dtype=torch.bfloat16, device=DEV)
            _pinned(data, sf, args, num_experts, top_k, output=warm, **kw)
            _assert_moe_close(
                warm, _ref_moe(xv, ids, wts, ref, None, None, limit).to(torch.bfloat16)
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                side = torch.empty(n, hidden, dtype=torch.bfloat16, device=DEV)
                _pinned(data, sf, args, num_experts, top_k, output=side, **kw)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with autotune() if i % 2 else contextlib.nullcontext():
                with torch.cuda.graph(graph):
                    captured = torch.empty(n, hidden, dtype=torch.bfloat16, device=DEV)
                    _pinned(data, sf, args, num_experts, top_k, output=captured, **kw)
            for skewed in (False, True):
                d2, s2, x2 = _rand_mxfp8(n, hidden, hidden, gen)
                if skewed:
                    i2, w2 = _skewed_routing(n, num_experts, top_k, gen)
                else:
                    i2, _ = _routing(n, num_experts, top_k, gen)
                    w2 = (torch.rand(n, top_k, device=DEV, generator=gen) * 1.5).to(torch.bfloat16)
                data.copy_(d2)
                sf.copy_(s2)
                ids.copy_(i2)
                wts.copy_(w2)
                captured.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                eager = torch.full((n, hidden), float("nan"), dtype=torch.bfloat16, device=DEV)
                _pinned(
                    d2, s2, args, num_experts, top_k,
                    topk_ids=i2, topk_weights=w2, gemm1_clamp_limit=limit, output=eager,
                )  # fmt: skip
                assert torch.equal(captured, eager), (n, "skewed" if skewed else "uniform")
                exp2 = _ref_moe(x2, i2, w2, ref, None, None, limit).to(torch.bfloat16)
                _assert_moe_close(captured, exp2)
                elt, rms = _dev(captured, exp2)
                worst = [max(worst[0], elt), max(worst[1], rms)]
                frac = (captured != warm).float().mean().item()
                stale_min = min(stale_min, frac)
                assert frac > 0.9, (n, skewed, frac)
            del graph, captured, side
    captured_calls = [r for r in rec if r[2]]
    assert len(captured_calls) == len(_DSPARK_CAPTURE_N), len(captured_calls)
    assert len(rec) == 5 * len(_DSPARK_CAPTURE_N), len(rec)
    assert {r[0] for r in rec} == {-1}, sorted({str(r[0]) for r in rec})
    assert sum(r[1] for r in rec) == len(_DSPARK_CAPTURE_N) // 2, (
        "the outer autotune() was not open at capture"
    )
    assert _op_cache_entries() == 0
    print(
        f"  test_dspark_capture_every_5G_over_gset OK: {len(_DSPARK_CAPTURE_N)} captures "
        f"({len(_DSPARK_CAPTURE_N) // 2} inside an open autotune()), {2 * len(_DSPARK_CAPTURE_N)} replays bitwise = "
        f"eager and in the gates (worst {worst[0]:.2f} ulp elt / {worst[1]:.2f} ulp RMS), replay vs "
        f"capture-time output differing on >= {stale_min:.3f} of elements; {len(rec)} calls "
        f"({len(captured_calls)} under capture, {sum(r[1] for r in rec)} with the outer tuning mode on) all on "
        f"tactic -1; N {list(_DSPARK_CAPTURE_N)}"
    )


def test_pin_holds_the_cold_tactic_inside_a_tuning_pass():
    """The target's pin inside the engine's tuning pass, with a tuned entry cached, and a control.

    (1) A pinned call made inside an open `autotune()` -- the engine's warm-up
    tuning pass, which runs the drafter -- takes the fallback tactic although
    tuning mode is on at the call, leaves the outer tuning mode on afterwards,
    adds no cache entry, and is bitwise equal to a call made outside any
    tuning context, at every dumped row count, every captured N and every
    cold-tile edge.
    (2) Control: the same call UNPINNED inside `autotune()` is tuned -- the
    recorder sees a tactic other than the fallback and the cache gains an
    entry -- so the recorder in (1) would have seen a tuned call.
    (3) With that entry cached, the pinned call still takes the fallback and
    the cold bits, while `autotune(tune_mode=False)` alone (which only stops
    tuning) takes the cached tactic.
    """
    num_experts, hidden, top_k = 128, 5120, 3
    args, ref, limit = _drafter_layer(seed=700)
    gen = ref["gen"]
    tuner = AutoTuner.get()
    rows = sorted(set(_DSPARK_DUMPED_N) | set(_DSPARK_CAPTURE_N) | {1, 383, 384, 725, 726})
    n_max, sfw = max(rows), hidden // SV
    data, sf, _ = _rand_mxfp8(n_max, hidden, hidden, gen)
    ids, _ = _routing(n_max, num_experts, top_k, gen)
    wts = (torch.rand(n_max, top_k, device=DEV, generator=gen) * 1.5).to(torch.bfloat16)

    def call(n, pin=True):
        out = torch.full((n, hidden), float("nan"), dtype=torch.bfloat16, device=DEV)
        (_pinned if pin else _call)(
            data[:n], sf[: n * sfw], args, num_experts, top_k,
            topk_ids=ids[:n], topk_weights=wts[:n], gemm1_clamp_limit=limit, output=out,
        )  # fmt: skip
        return out

    assert _op_cache_entries() == 0
    cold = {n: call(n) for n in rows}
    cache = tuner.profiling_cache.cache
    try:
        with _tactics() as rec:
            restored, diff = [], []
            with autotune():
                for n in rows:
                    if not torch.equal(call(n), cold[n]):
                        diff.append(n)
                    restored.append(tuner.is_tuning_mode)
            assert diff == [], f"pinned inside autotune() != cold at N {diff}"
            assert all(restored), "the pin did not restore the outer tuning mode"
            assert (
                len(rec) == len(rows) and {r[0] for r in rec} == {-1} and all(r[1] for r in rec)
            ), rec
            assert _op_cache_entries() == 0
            pinned_calls = len(rec)

            rec.clear()
            with autotune(skip_dynamic_tuning_buckets=True):
                call(5, pin=False)
            tuned = rec[-1]
            assert tuned[1], "the unpinned call inside autotune() did not run in tuning mode"
            assert tuned[0] != -1, "the control call was not tuned"
            assert _op_cache_entries() >= 1

            rec.clear()
            assert torch.equal(call(5), cold[5])
            assert [r[0] for r in rec] == [-1], (rec, tuned)
            rec.clear()
            with autotune(tune_mode=False):
                call(5, pin=False)
            untuned_only = rec[-1][0]
            assert untuned_only == tuned[0], (rec, tuned)
        print(
            f"  test_pin_holds_the_cold_tactic_inside_a_tuning_pass OK: {pinned_calls} pinned calls "
            f"inside autotune() at N {rows[0]}..{rows[-1]} ({len(rows)} row counts) on tactic -1, bitwise = "
            f"cold, cache empty; control: unpinned call tuned to {tuned[0]}; with that entry cached the pinned "
            f"call takes -1 and the cold bits, autotune(tune_mode=False) alone takes {untuned_only}"
        )
    finally:
        for key in [k for k in cache if k[0] == _OP]:
            del cache[key]
    assert _op_cache_entries() == 0
    with _tactics() as rec:
        assert torch.equal(call(5), cold[5])
    assert [r[0] for r in rec] == [-1], rec


def test_tactics_change_output_bits():
    """Why the drafter pins the cold tactic: the op's tunable tactics do not all give the same bits.

    At one row count per cold tile band and at the drafter's largest block
    (N 5, 160, 385, 1280), every tactic the AutoTuner may choose from
    (`get_valid_tactics` of the op's TunableRunner) is run on the same inputs
    and compared bitwise with the pinned (cold) call, and against the
    reference at the entry's gates. A rank that tunes on its own can therefore
    land in a different bit class from its neighbour; the pinned call cannot.
    """
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import MxE4m3MxE2m1BlockScaleMoERunner

    num_experts, hidden, inter, top_k = 128, 5120, 2304, 3
    args, ref, limit = _drafter_layer(seed=800)
    gen = ref["gen"]
    runner = MxE4m3MxE2m1BlockScaleMoERunner(
        num_experts, top_k, None, None, inter, hidden, inter, 0, num_experts, None, 1, 0
    )
    summary = []
    for n in (5, 160, 385, 1280):
        data, sf, xv = _rand_mxfp8(n, hidden, hidden, gen)
        ids, _ = _routing(n, num_experts, top_k, gen)
        wts = (torch.rand(n, top_k, device=DEV, generator=gen) * 1.5).to(torch.bfloat16)
        exp = _ref_moe(xv, ids, wts, ref, None, None, limit).to(torch.bfloat16)
        cold = torch.full((n, hidden), float("nan"), dtype=torch.bfloat16, device=DEV)
        _pinned(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts,
            gemm1_clamp_limit=limit,
            output=cold,
        )
        inputs = [
            None, None, data, sf,
            args["gemm1_weights"], args["gemm1_weights_scale"], None, None, None, limit,
            args["gemm2_weights"], args["gemm2_weights_scale"], None, wts, ids,
        ]  # fmt: skip
        tactics = runner.get_valid_tactics(inputs, None)
        differ, outside, worst = [], [], [0.0, 0.0]
        for tac in tactics:
            out = torch.full((n, hidden), float("nan"), dtype=torch.bfloat16, device=DEV)
            runner.forward(inputs, tactic=list(tac), output=out)
            if not torch.equal(out, cold):
                differ.append(tuple(tac))
            if _gate_rejects(out, exp):
                outside.append(tuple(tac))
            elt, rms = _dev(out, exp)
            worst = [max(worst[0], elt), max(worst[1], rms)]
        summary.append((n, len(tactics), len(differ), len(outside), worst, differ[:4], outside[:4]))
    for n, total, nd, no, w, d4, o4 in summary:
        print(
            f"    N={n}: {total} valid tactics, {nd} not bitwise equal to the cold call (first {d4}), "
            f"{no} outside the gates (first {o4}); worst {w[0]:.2f} ulp elt / {w[1]:.2f} ulp RMS"
        )
    assert all(total > 0 for _, total, *_ in summary)
    assert all(no == 0 for _, _, _, no, *_ in summary), "a valid tactic lands outside the gates"
    assert any(nd > 0 for _, _, nd, *_ in summary), (
        "every tactic gave the cold bits: this case separates nothing"
    )


def test_pin_ignores_a_cached_tactic():
    """The target's pin takes the fallback even when the process holds a tuned entry for this op.

    The drafter never tunes this op, but a process can still hold tuned
    entries for it: `TLLM_AUTOTUNER_CACHE_PATH` makes the engine's warm-up
    load an autotuner cache file (`autotune(cache_path=...)`), which another
    engine may have tuned, rank by rank. The test writes such a file the
    engine's way. It tunes the op unpinned at one row count per cold tile band
    and at the drafter maximum (N 5, 160, 385, 1280), sets each entry to a
    valid tactic whose bits differ from the fallback's where one exists (none
    does at 1280, where the first valid tactic is used), saves the cache,
    drops the entries, and loads the file through `autotune(cache_path=...)`.

    Control: the cache is then hit -- `autotune(tune_mode=False)` alone takes
    the cached tactic and, at N 5, 160 and 385, gives different bits. The
    target's pin must take tactic -1 and give the uncached call's bits at every
    N: eagerly, inside an open `autotune()`, and under CUDA-graph capture
    replayed on new inputs.
    """
    import tempfile

    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import MxE4m3MxE2m1BlockScaleMoERunner

    def _tac(t):
        return list(t) if isinstance(t, (list, tuple)) else t

    num_experts, hidden, inter, top_k = 128, 5120, 2304, 3
    args, ref, limit = _drafter_layer(seed=900)
    gen = ref["gen"]
    tuner = AutoTuner.get()
    cache = tuner.profiling_cache.cache
    rows = (5, 160, 385, 1280)

    def case(n):
        data, sf, _ = _rand_mxfp8(n, hidden, hidden, gen)
        ids, _ = _routing(n, num_experts, top_k, gen)
        wts = (torch.rand(n, top_k, device=DEV, generator=gen) * 1.5).to(torch.bfloat16)
        return data, sf, ids, wts

    def call(fn, c, out=None):
        data, sf, ids, wts = c
        out = (
            torch.full((data.shape[0], hidden), float("nan"), dtype=torch.bfloat16, device=DEV)
            if out is None
            else out
        )
        fn(
            data,
            sf,
            args,
            num_experts,
            top_k,
            topk_ids=ids,
            topk_weights=wts,
            gemm1_clamp_limit=limit,
            output=out,
        )
        return out

    def old_pin(*a, **kw):
        with autotune(tune_mode=False):
            return _call(*a, **kw)

    assert _op_cache_entries() == 0
    cases = {n: case(n) for n in rows}
    fresh = {n: case(n) for n in rows}
    cold = {n: call(_pinned, cases[n]) for n in rows}
    cold_fresh = {n: call(_pinned, fresh[n]) for n in rows}
    runner = MxE4m3MxE2m1BlockScaleMoERunner(
        num_experts, top_k, None, None, inter, hidden, inter, 0, num_experts, None, 1, 0
    )
    chosen, differs = {}, {}
    try:
        for n in rows:
            data, sf, ids, wts = cases[n]
            inputs = [
                None, None, data, sf,
                args["gemm1_weights"], args["gemm1_weights_scale"], None, None, None, limit,
                args["gemm2_weights"], args["gemm2_weights_scale"], None, wts, ids,
            ]  # fmt: skip
            tactics = [list(t) for t in runner.get_valid_tactics(inputs, None)]
            pick = None
            for tac in tactics:
                out = torch.full((n, hidden), float("nan"), dtype=torch.bfloat16, device=DEV)
                runner.forward(inputs, tactic=tac, output=out)
                if not torch.equal(out, cold[n]):
                    pick = tac
                    break
            differs[n] = pick is not None
            chosen[n] = pick if pick is not None else tactics[0]
            before = set(k for k in cache if k[0] == _OP)
            with autotune(skip_dynamic_tuning_buckets=True):
                call(_call, cases[n])
            new = set(k for k in cache if k[0] == _OP) - before
            assert new, f"the unpinned call at N={n} inside autotune() wrote no cache entry"
            for key in new:
                runner_id, _, min_time = cache[key]
                cache[key] = (runner_id, chosen[n], min_time)
        assert [differs[n] for n in rows] == [True, True, True, False], differs
        written = {k: _tac(v[1]) for k, v in cache.items() if k[0] == _OP}
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/autotuner_cache.json"
            tuner.profiling_cache.save_cache(path, tuner.mapping.rank)
            for key in [k for k in cache if k[0] == _OP]:
                del cache[key]
            assert _op_cache_entries() == 0
            with autotune(tune_mode=False, cache_path=path):
                pass
        cache = tuner.profiling_cache.cache
        loaded = {k: _tac(v[1]) for k, v in cache.items() if k[0] == _OP}
        assert loaded == written, (loaded, written)

        with _tactics() as rec:
            for n in rows:
                rec.clear()
                out = call(old_pin, cases[n])
                assert [_tac(r[0]) for r in rec] == [chosen[n]], (n, rec)
                assert torch.equal(out, cold[n]) != differs[n], n
        print(
            f"    control: autotune(tune_mode=False) takes the loaded tactics {[chosen[n] for n in rows]}"
        )

        with _tactics() as rec:
            for n in rows:
                assert torch.equal(call(_pinned, cases[n]), cold[n]), (
                    f"pinned call != uncached call at N={n}"
                )
                with autotune():
                    assert torch.equal(call(_pinned, cases[n]), cold[n]), (
                        f"pinned in autotune() != uncached at N={n}"
                    )
            for i, n in enumerate(rows):
                c = case(n)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    call(_pinned, c)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with autotune() if i % 2 else contextlib.nullcontext():
                    with torch.cuda.graph(graph):
                        captured = call(
                            _pinned, c, torch.empty(n, hidden, dtype=torch.bfloat16, device=DEV)
                        )
                for dst, src in zip(c, fresh[n]):
                    dst.copy_(src)
                captured.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(captured, cold_fresh[n]), (
                    f"replay with a cached tactic present != uncached at N={n}"
                )
                del graph, captured
        assert {str(r[0]) for r in rec} == {"-1"}, sorted({str(r[0]) for r in rec})
        assert sum(r[2] for r in rec) == len(rows)
        print(
            f"  test_pin_ignores_a_cached_tactic OK: with tactics {[chosen[n] for n in rows]} loaded for "
            f"N {list(rows)}, {len(rec)} pinned calls ({len(rows)} under capture) took -1 and the uncached bits"
        )
    finally:
        for key in [k for k in tuner.profiling_cache.cache if k[0] == _OP]:
            del tuner.profiling_cache.cache[key]
    assert _op_cache_entries() == 0


def _window(args, ref, offset: int, num_local: int):
    """The `(args, ref)` pair narrowed to one expert-parallel window."""
    sub = dict(args)
    for key in (
        "gemm1_weights",
        "gemm1_weights_scale",
        "gemm1_bias",
        "gemm2_weights",
        "gemm2_weights_scale",
        "gemm2_bias",
    ):
        sub[key] = args[key][offset : offset + num_local].contiguous()
    sub_ref = dict(ref)
    for role in ("up", "gate", "down"):
        c, s, b = ref[role]
        sub_ref[role] = (
            c[offset : offset + num_local],
            s[offset : offset + num_local],
            b[offset : offset + num_local],
        )
    return sub, sub_ref


_DSPARK_LAYER = None


def _dspark_layer():
    """The drafter layer of `test_dspark_drafter_surface`, built once for the cells."""
    global _DSPARK_LAYER
    if _DSPARK_LAYER is None:
        _DSPARK_LAYER = _drafter_layer(seed=300)
    return _DSPARK_LAYER


def _certify_dspark_cell(cell) -> None:
    """One DeepSeek-V4.1 drafter cell, called the way the drafter calls it.

    Pre-routed with bf16 weights in `[0, 1.5)`, the clamp at 10 per expert, no
    alpha / beta, no biases. A pinned cell runs inside `fallback_tactic()` and
    writes a NaN-filled `output=` buffer, getting back the empty tensor; the
    other form returns a fresh tensor. The entry's reference is tied bitwise to
    this file's `_ref_moe` before it gates the kernel.
    """
    spec = cell.spec
    num_experts, hidden, inter, top_k = (
        spec["num_experts"],
        spec["hidden"],
        spec["intermediate"],
        spec["top_k"],
    )
    assert (num_experts, hidden, inter, top_k) == (128, 5120, 2304, 3), spec
    assert spec["local_expert_offset"] == 0 and spec["local_num_experts"] == num_experts, spec
    assert not spec["routed_by_kernel"], spec
    args, ref, limit = _dspark_layer()
    assert ref["h1_pad"] == hidden and args["intermediate_size"] == inter, "no padding at 5120/2304"
    assert torch.equal(limit, torch.full_like(limit, spec["swiglu_limit"])), limit
    tokens = spec["tokens"]
    gen = torch.Generator(device=DEV).manual_seed(zlib.crc32(cell.why.encode()))
    data, sf, xv = _rand_mxfp8(tokens, hidden, hidden, gen)
    ids, _ = _routing(tokens, num_experts, top_k, gen)
    wts = (torch.rand(tokens, top_k, device=DEV, generator=gen) * 1.5).to(torch.bfloat16)

    # 1. The entry's reference against this file's, on identical inputs.
    entry_ref = moe.reference(xv, ids, wts, ref["up"], ref["gate"], ref["down"], clamp_limit=limit)
    local_ref = _ref_moe(xv, ids, wts, ref, None, None, limit)
    torch.testing.assert_close(entry_ref, local_ref, rtol=0.0, atol=0.0)

    # 2. The entry's reference against the kernel, through the entry's gate.
    kw = dict(topk_ids=ids, topk_weights=wts, gemm1_clamp_limit=limit)
    if spec["output"]:
        out = torch.full((tokens, hidden), float("nan"), dtype=torch.bfloat16, device=DEV)
        kw["output"] = out
    run = _pinned if spec["pinned"] else _call
    y = run(data, sf, args, num_experts, top_k, **kw)
    if spec["output"]:
        assert y.numel() == 0, tuple(y.shape)
        y = out
    assert y.shape == (tokens, hidden), y.shape
    moe.compare(y, entry_ref.to(torch.bfloat16))


@pytest.mark.parametrize("cell", moe.CELLS, ids=[c.why[:44] for c in moe.CELLS])
def test_certified_cells(cell):
    """Drive every configuration the entry claims, through the entry's own
    reference and its own gate.

    The cases above test the op; this one tests the *claim*. A cell list that
    nothing executes reads as coverage and is not.

    Each cell cross-checks `moe.reference` against this file's `_ref_moe`
    before either is used to gate the kernel. Two references disagreeing is a
    different failure from the kernel moving, and collapsing them into one
    assert loses which happened. That cross-check earns its keep here: both
    references model the FC1 epilogue's architecture-specific block-scale
    recipe, and a disagreement there is bit-exact rather than a tolerance.

    `routed_by_kernel` is honoured rather than approximated. The gpt-oss target
    hands the call logits and lets it route, so those cells do too; the expected
    routing is recomputed here the way Renormalize defines it (top-k, then
    softmax over the kept logits), which is what makes the reference comparable
    at all.
    """
    spec = cell.spec
    if spec.get("model") == "deepseek-v4.1":
        _certify_dspark_cell(cell)
        return
    args, ref = _gpt_oss()
    off, n = spec["local_expert_offset"], spec["local_num_experts"]
    num_experts, hidden, top_k = spec["num_experts"], spec["hidden"], spec["top_k"]
    gen = torch.Generator(device=DEV).manual_seed(abs(hash(cell.why)) % 2**31)

    data, sf, xv = _rand_mxfp8(spec["tokens"], hidden, ref["h1_pad"], gen)
    alpha, beta, limit = _swiglu_params(n)
    sub_args, sub_ref = (args, ref) if n == num_experts else _window(args, ref, off, n)

    if spec["routed_by_kernel"]:
        logits = _logits(spec["tokens"], num_experts, gen)
        top_v, top_i = torch.topk(logits.float(), top_k, dim=-1)
        ids, wts = top_i.to(torch.int32), torch.softmax(top_v, dim=-1)
        entry = dict(routing_logits=logits, routing_method_type=1)  # Renormalize
    else:
        ids, wts = _routing(spec["tokens"], num_experts, top_k, gen)
        entry = dict(topk_ids=ids, topk_weights=wts)

    # 1. The entry's reference against this file's, on identical inputs.
    entry_ref = moe.reference(
        xv,
        ids,
        wts,
        sub_ref["up"],
        sub_ref["gate"],
        sub_ref["down"],
        alpha=alpha,
        beta=beta,
        clamp_limit=limit,
        local_expert_offset=off,
        local_num_experts=n,
    )
    local_ref = _ref_moe(xv, ids, wts, sub_ref, alpha, beta, limit, offset=off, num_local=n)
    torch.testing.assert_close(entry_ref, local_ref, rtol=0.0, atol=0.0)

    # 2. The entry's reference against the kernel, through the entry's gate.
    y = _call(
        data,
        sf,
        sub_args,
        num_experts,
        top_k,
        local_expert_offset=off,
        local_num_experts=n,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
        **entry,
    )
    assert y.shape == (spec["tokens"], hidden), y.shape
    moe.compare(y, entry_ref.to(torch.bfloat16))
