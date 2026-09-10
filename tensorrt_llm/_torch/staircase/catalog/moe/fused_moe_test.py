# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_moe catalog entry."""

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.autotuner import AutoTuner, autotune

from .fused_moe import fused_moe

assert torch.cuda.is_available(), "fused_moe requires a CUDA device"

DEV = "cuda"
# The reference GEMMs must be true fp32; TF32 would leave the reference with
# 10 mantissa bits, coarser than the bf16 output it is meant to bound.
torch.backends.cuda.matmul.allow_tf32 = False

# Relative distance between neighbouring representable values (1 + 2^-m for m
# stored mantissa bits): bf16 has 7, fp16 has 10.
_ULP = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11}


def _ref_moe(
    x: torch.Tensor,
    ids: torch.Tensor,
    scales: torch.Tensor | None,
    w31: torch.Tensor,
    w2: torch.Tensor,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    act: str = "swiglu",
    alpha: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    limit: torch.Tensor | None = None,
    ep_size: int = 1,
    ep_rank: int = 0,
) -> torch.Tensor:
    """Native-torch MoE: fp32 GEMM accumulation, activation rounded to x's dtype.

    `w31[e]` is `[up | gate]` stacked on dim 0; `ids` carries global expert ids
    and this rank owns `[ep_rank * E_local, (ep_rank + 1) * E_local)`.
    """
    num_tokens, hidden = x.shape
    num_local = w31.shape[0]
    out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)
    xf = x.float()
    for local_e in range(num_local):
        mask = ids == ep_rank * num_local + local_e
        tok, slot = mask.nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        xe = xf[tok]
        w_up, w_gate = w31[local_e].float().chunk(2, dim=0)
        h_up = xe @ w_up.t()
        h_gate = xe @ w_gate.t()
        if b1 is not None:
            b_up, b_gate = b1[local_e].float().chunk(2, dim=0)
            h_up, h_gate = h_up + b_up, h_gate + b_gate
        if limit is not None:
            lim = float(limit[local_e])
            h_gate = h_gate.clamp(max=lim)
            h_up = h_up.clamp(min=-lim, max=lim)
        if act == "swiglu":
            a = float(alpha[local_e]) if alpha is not None else 1.0
            gated = h_gate * torch.sigmoid(a * h_gate)
        elif act == "geglu":
            gated = F.gelu(h_gate, approximate="tanh")
        else:
            raise ValueError(act)
        if beta is not None:
            h_up = h_up + float(beta[local_e])
        # the kernel materializes the FC1 activation in the input dtype
        inter = (gated * h_up).to(x.dtype).float()
        y = inter @ w2[local_e].float().t()
        if b2 is not None:
            y = y + b2[local_e].float()
        weight = (
            torch.ones(tok.numel(), device=x.device)
            if scales is None
            else scales[tok, slot].float()
        )
        out.index_add_(0, tok, y * weight.unsqueeze(1))
    return out.to(x.dtype)


def _assert_moe_close(y: torch.Tensor, ref: torch.Tensor) -> None:
    """Two gates: per-element, row-scaled; and aggregate relative RMS.

    Kernel and reference consume bit-identical low-precision operands and
    differ only in GEMM accumulation order and in which side of a rounding
    boundary each intermediate lands — one flipped intermediate moves an
    output element by about one ulp of that row's scale. Default assert_close
    tolerances cannot express this: their bf16 `atol=1e-5` sits three orders
    of magnitude below one output ulp of a two-GEMM chain, and per-element
    `rtol` is meaningless where cancellation makes `|ref|` near zero. So the
    element gate is 8 ulp of the row's largest magnitude and the aggregate gate
    is 4 ulp of relative RMS. Measured worst case across everything this file
    drives: 2.8 ulp / 1.2 ulp on the cold fallback tactic, 3.98 ulp / 1.43 ulp
    once test_r1_mtp_tactic_space walks the whole tactic space -- 2.0x and 2.8x
    margin. Both gates bite: test_reference_discriminates shows a wrong
    computation landing at 176+ ulp per element and ~190 ulp RMS, and
    test_r1_mtp_reference_discriminates at 284-997 / 197-212 ulp.
    """
    assert y.dtype == ref.dtype, (y.dtype, ref.dtype)
    assert y.shape == ref.shape, (y.shape, ref.shape)
    ulp = _ULP[ref.dtype]
    row_scale = ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    torch.testing.assert_close(
        y.float() / row_scale, ref.float() / row_scale, rtol=0.0, atol=8 * ulp
    )
    rel_rms = (
        (y.float() - ref.float()).pow(2).mean().sqrt()
        / ref.float().pow(2).mean().sqrt().clamp_min(1e-9)
    ).item()
    assert rel_rms <= 4 * ulp, f"relative RMS {rel_rms:.3e} > {4 * ulp:.3e}"


def _make(
    num_tokens: int,
    hidden: int,
    inter: int,
    num_experts: int,
    top_k: int,
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
):
    """(x, ids, scales, w31, w2) with realistic magnitudes and top-k routing."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(num_tokens, hidden, device=DEV, generator=g).to(dtype)
    w31 = (torch.randn(num_experts, 2 * inter, hidden, device=DEV, generator=g) / hidden**0.5).to(
        dtype
    )
    w2 = (torch.randn(num_experts, hidden, inter, device=DEV, generator=g) / inter**0.5).to(dtype)
    logits = torch.randn(num_tokens, num_experts, device=DEV, generator=g)
    vals, ids = torch.topk(logits, top_k, dim=-1)
    return x, ids.to(torch.int32), torch.softmax(vals, dim=-1), w31, w2


def test_qwen3_moe_shape_bf16() -> None:
    # Qwen3-30B-A3B MoE block: hidden 2048, moe_intermediate 768, 128 experts,
    # top-8. Decode-like through prefill-like token counts.
    # 8192 is one token past the op's default tune_max_num_tokens bucket cap.
    for num_tokens in [1, 2, 37, 256, 2048, 8192]:
        x, ids, scales, w31, w2 = _make(num_tokens, 2048, 768, 128, 8, seed=num_tokens)
        y = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
        assert y.shape == (num_tokens, 2048) and y.is_contiguous()
        _assert_moe_close(y, _ref_moe(x, ids, scales, w31, w2))
        del x, ids, scales, w31, w2, y
        torch.cuda.empty_cache()


def test_dtypes() -> None:
    for dtype in [torch.bfloat16, torch.float16]:
        for num_tokens in [1, 64, 512]:
            x, ids, scales, w31, w2 = _make(
                num_tokens, 1024, 512, 32, 4, seed=num_tokens, dtype=dtype
            )
            y = fused_moe(x, ids, scales, w31, None, w2, None, dtype, [])[0]
            _assert_moe_close(y, _ref_moe(x, ids, scales, w31, w2))


def test_shape_sweep() -> None:
    # hidden and inter must be multiples of 8; expert count and top-k are free.
    cases = [
        (8, 8, 8, 2, 1),
        (4, 16, 8, 1, 1),
        (33, 192, 96, 7, 3),
        (64, 128, 64, 8, 8),
        (16, 256, 128, 256, 2),
        (9, 512, 256, 32, 16),
        (64, 2048, 768, 128, 1),
    ]
    for num_tokens, hidden, inter, num_experts, top_k in cases:
        x, ids, scales, w31, w2 = _make(
            num_tokens, hidden, inter, num_experts, top_k, seed=hidden + num_experts
        )
        y = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
        _assert_moe_close(y, _ref_moe(x, ids, scales, w31, w2))


def test_expert_biases() -> None:
    g = torch.Generator(device=DEV).manual_seed(77)
    x, ids, scales, w31, w2 = _make(64, 512, 256, 16, 4, seed=77)
    b1 = (torch.randn(16, 512, device=DEV, generator=g) * 0.3).to(torch.bfloat16)
    b2 = (torch.randn(16, 512, device=DEV, generator=g) * 0.3).to(torch.bfloat16)
    # fc1 bias is split into [up | gate] halves exactly like fc1's rows
    y = fused_moe(x, ids, scales, w31, b1, w2, b2, torch.bfloat16, [])[0]
    _assert_moe_close(y, _ref_moe(x, ids, scales, w31, w2, b1=b1, b2=b2))
    # a zero fc2 bias reproduces the bias-free result
    y = fused_moe(x, ids, scales, w31, b1, w2, torch.zeros_like(b2), torch.bfloat16, [])[0]
    _assert_moe_close(y, _ref_moe(x, ids, scales, w31, w2, b1=b1))


def test_out_tensor_is_written_in_place() -> None:
    x, ids, scales, w31, w2 = _make(37, 512, 256, 16, 4, seed=5)
    ref = _ref_moe(x, ids, scales, w31, w2)
    # a NaN prefill proves every element is written, not accumulated into
    pool = torch.full((41, 512), float("nan"), dtype=torch.bfloat16, device=DEV)
    out = pool[:37]
    ret = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [], out_tensor=out)
    assert ret == [], "out_tensor form must return an empty list"
    _assert_moe_close(out, ref)
    # nothing was written past the requested rows
    assert bool(torch.isnan(pool[37:]).all()), "wrote outside out_tensor's rows"


def test_none_scales_and_arbitrary_scales() -> None:
    x, ids, scales, w31, w2 = _make(16, 256, 128, 16, 4, seed=7)
    # token_final_scales=None combines the selected experts with weight 1.0
    y = fused_moe(x, ids, None, w31, None, w2, None, torch.bfloat16, [])[0]
    _assert_moe_close(y, _ref_moe(x, ids, None, w31, w2))
    # nothing is renormalized inside: any fp32 weights are used as given
    g = torch.Generator(device=DEV).manual_seed(8)
    free = (torch.rand(16, 4, device=DEV, generator=g) * 4 - 2).contiguous()
    y = fused_moe(x, ids, free, w31, None, w2, None, torch.bfloat16, [])[0]
    _assert_moe_close(y, _ref_moe(x, ids, free, w31, w2))
    # all-zero weights zero the output exactly
    zeros = torch.zeros_like(free)
    y = fused_moe(x, ids, zeros, w31, None, w2, None, torch.bfloat16, [])[0]
    assert bool((y == 0).all()), "zero combine weights did not zero the output"


def test_expert_parallel_split() -> None:
    # ep_rank r owns global expert ids [r * E_local, (r + 1) * E_local); tokens
    # routed elsewhere contribute nothing to that rank's output, so the two
    # rank outputs sum to the single-rank result.
    x, ids, scales, w31, w2 = _make(64, 256, 128, 16, 4, seed=13)
    full = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
    _assert_moe_close(full, _ref_moe(x, ids, scales, w31, w2))
    parts = []
    for rank in (0, 1):
        lo, hi = rank * 8, (rank + 1) * 8
        w31_r, w2_r = w31[lo:hi].contiguous(), w2[lo:hi].contiguous()
        y = fused_moe(
            x,
            ids,
            scales,
            w31_r,
            None,
            w2_r,
            None,
            torch.bfloat16,
            [],
            ep_size=2,
            ep_rank=rank,
        )[0]
        _assert_moe_close(y, _ref_moe(x, ids, scales, w31_r, w2_r, ep_size=2, ep_rank=rank))
        parts.append(y.float())
    _assert_moe_close((parts[0] + parts[1]).to(torch.bfloat16), full)


def test_swiglu_alpha_beta_limit() -> None:
    # act = clamp(gate) * sigmoid(alpha * clamp(gate)) * (clamp(up) + beta),
    # with gate clamped above by limit and up clamped to [-limit, limit].
    num_experts = 16
    x, ids, scales, w31, w2 = _make(32, 256, 128, num_experts, 4, seed=17)
    g = torch.Generator(device=DEV).manual_seed(18)
    alpha = 1.0 + torch.rand(num_experts, device=DEV, generator=g)
    beta = torch.rand(num_experts, device=DEV, generator=g)
    limit = 1.0 + 4.0 * torch.rand(num_experts, device=DEV, generator=g)
    for a, b, lim in [
        (alpha, None, None),
        (None, beta, None),
        (None, None, limit),
        (alpha, beta, limit),
    ]:
        y = fused_moe(
            x,
            ids,
            scales,
            w31,
            None,
            w2,
            None,
            torch.bfloat16,
            [],
            swiglu_alpha=a,
            swiglu_beta=b,
            swiglu_limit=lim,
        )[0]
        _assert_moe_close(y, _ref_moe(x, ids, scales, w31, w2, alpha=a, beta=b, limit=lim))
    # activation_type 7 (SwigluBias) computes the same thing on this path
    y5 = fused_moe(
        x,
        ids,
        scales,
        w31,
        None,
        w2,
        None,
        torch.bfloat16,
        [],
        swiglu_alpha=alpha,
        swiglu_beta=beta,
        swiglu_limit=limit,
        activation_type=5,
    )[0]
    y7 = fused_moe(
        x,
        ids,
        scales,
        w31,
        None,
        w2,
        None,
        torch.bfloat16,
        [],
        swiglu_alpha=alpha,
        swiglu_beta=beta,
        swiglu_limit=limit,
        activation_type=7,
    )[0]
    assert torch.equal(y5, y7), "activation_type 7 differed from 5"


def test_geglu_activation() -> None:
    x, ids, scales, w31, w2 = _make(32, 256, 128, 16, 4, seed=19)
    y = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [], activation_type=6)[0]
    _assert_moe_close(y, _ref_moe(x, ids, scales, w31, w2, act="geglu"))


def test_use_fused_finalize_false() -> None:
    x, ids, scales, w31, w2 = _make(64, 512, 256, 16, 4, seed=23)
    y = fused_moe(
        x,
        ids,
        scales,
        w31,
        None,
        w2,
        None,
        torch.bfloat16,
        [],
        use_fused_finalize=False,
    )[0]
    _assert_moe_close(y, _ref_moe(x, ids, scales, w31, w2))


def test_out_of_range_expert_ids_are_dropped() -> None:
    # Ids outside this rank's slot range are silently skipped, not clamped.
    x, ids, scales, w31, w2 = _make(32, 256, 128, 8, 4, seed=29)
    for bad_id in [8, 100, -1]:
        bad = ids.clone()
        bad[:, 1] = bad_id
        y = fused_moe(x, bad, scales, w31, None, w2, None, torch.bfloat16, [])[0]
        kept = torch.zeros_like(bad, dtype=torch.bool)
        kept[:, 0] = True
        kept[:, 2:] = True
        dropped_scales = scales * kept
        _assert_moe_close(y, _ref_moe(x, bad, dropped_scales, w31, w2))


def test_repeated_expert_ids() -> None:
    # A row may name the same expert twice; each slot is combined separately.
    # This holds only for num_tokens <= 256, which is what this case drives --
    # past that the op takes a different expert-map path and reads out of
    # bounds on a repeated id (see the contract's Notes).
    x, ids, scales, w31, w2 = _make(32, 256, 128, 8, 4, seed=30)
    dup = ids.clone()
    dup[:, 1] = dup[:, 0]
    y = fused_moe(x, dup, scales, w31, None, w2, None, torch.bfloat16, [])[0]
    _assert_moe_close(y, _ref_moe(x, dup, scales, w31, w2))


def test_inputs_untouched_and_deterministic() -> None:
    x, ids, scales, w31, w2 = _make(256, 512, 256, 32, 4, seed=31)
    snap = [t.clone() for t in (x, ids, scales, w31, w2)]
    y_a = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
    y_b = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
    for t, s in zip((x, ids, scales, w31, w2), snap):
        assert torch.equal(t, s), "an input tensor was mutated"
    assert y_a.data_ptr() != y_b.data_ptr(), "two calls shared an output buffer"
    assert torch.equal(y_a, y_b), "two identical calls disagreed"


def test_reference_discriminates() -> None:
    # The gates must reject a computation that only differs in which half of
    # fc1 is the gate: without this control the tolerances prove nothing.
    x, ids, scales, w31, w2 = _make(64, 256, 128, 16, 4, seed=37)
    y = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
    swapped = torch.cat([w31[:, 128:], w31[:, :128]], dim=1).contiguous()
    try:
        _assert_moe_close(y, _ref_moe(x, ids, scales, swapped, w2))
    except AssertionError:
        pass
    else:
        raise AssertionError("tolerance accepted a gate/up-swapped reference")
    # ...and a dropped expert must be caught too
    dropped = scales.clone()
    dropped[:, 0] = 0.0
    try:
        _assert_moe_close(y, _ref_moe(x, ids, dropped, w31, w2))
    except AssertionError:
        pass
    else:
        raise AssertionError("tolerance accepted a reference missing one expert")


def test_wrapper_rejects_output_dtype_mismatch() -> None:
    # Observed silent failure: the store happens in the activation dtype while
    # the buffer is allocated as output_dtype, so the bits are reinterpreted.
    x, ids, scales, w31, w2 = _make(8, 128, 64, 8, 2, seed=41)
    for bad in [torch.float16, torch.float32]:
        try:
            fused_moe(x, ids, scales, w31, None, w2, None, bad, [])
        except AssertionError:
            continue
        raise AssertionError(f"wrapper accepted output_dtype={bad} for a bf16 input")
    # positive control: the matching dtype still works
    y = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
    _assert_moe_close(y, _ref_moe(x, ids, scales, w31, w2))


def test_op_rejects_unsupported_domains() -> None:
    x, ids, scales, w31, w2 = _make(8, 128, 64, 8, 2, seed=43)
    args = (x, ids, scales, w31, None, w2, None, torch.bfloat16, [])
    wide_x = torch.randn(8, 256, dtype=torch.bfloat16, device=DEV)
    wide_ids = torch.zeros(8, 4, dtype=torch.int32, device=DEV)
    wide_scales = torch.zeros(8, 4, dtype=torch.float32, device=DEV)
    x_bad, ids_bad, sc_bad, w31_bad, w2_bad = _make(8, 136, 72, 8, 2, seed=44)
    x_0, ids_0, sc_0, w31_0, w2_0 = _make(0, 128, 64, 8, 2, seed=45)
    cases = [
        (
            "zero tokens",
            lambda: fused_moe(x_0, ids_0, sc_0, w31_0, None, w2_0, None, torch.bfloat16, []),
        ),
        ("cpu input", lambda: fused_moe(x.cpu(), *args[1:])),
        (
            "cpu weights",
            lambda: fused_moe(x, ids, scales, w31.cpu(), None, w2, None, torch.bfloat16, []),
        ),
        ("3D input", lambda: fused_moe(x.unsqueeze(0), *args[1:])),
        (
            "2D fc1",
            lambda: fused_moe(x, ids, scales, w31[0], None, w2, None, torch.bfloat16, []),
        ),
        ("non-contiguous input", lambda: fused_moe(wide_x[:, :128], *args[1:])),
        ("non-contiguous ids", lambda: fused_moe(x, wide_ids[:, :2], *args[2:])),
        (
            "non-contiguous scales",
            lambda: fused_moe(x, ids, wide_scales[:, :2], *args[3:]),
        ),
        (
            "non-contiguous fc1",
            lambda: fused_moe(
                x,
                ids,
                scales,
                w31.transpose(1, 2).contiguous().transpose(1, 2),
                None,
                w2,
                None,
                torch.bfloat16,
                [],
            ),
        ),
        ("int64 ids", lambda: fused_moe(x, ids.long(), *args[2:])),
        ("bf16 scales", lambda: fused_moe(x, ids, scales.bfloat16(), *args[3:])),
        (
            "fp32 activations",
            lambda: fused_moe(
                x.float(),
                ids,
                scales,
                w31.float(),
                None,
                w2.float(),
                None,
                torch.float32,
                [],
            ),
        ),
        (
            "mismatched weight dtype",
            lambda: fused_moe(
                x, ids, scales, w31.half(), None, w2.half(), None, torch.bfloat16, []
            ),
        ),
        (
            "fp32 bias",
            lambda: fused_moe(
                x,
                ids,
                scales,
                w31,
                torch.zeros(8, 128, device=DEV),
                w2,
                torch.zeros(8, 128, device=DEV),
                torch.bfloat16,
                [],
            ),
        ),
        (
            "fc1 bias without fc2 bias",
            lambda: fused_moe(
                x,
                ids,
                scales,
                w31,
                torch.zeros(8, 128, dtype=torch.bfloat16, device=DEV),
                w2,
                None,
                torch.bfloat16,
                [],
            ),
        ),
        (
            "fc2 bias without fc1 bias",
            lambda: fused_moe(
                x,
                ids,
                scales,
                w31,
                None,
                w2,
                torch.zeros(8, 128, dtype=torch.bfloat16, device=DEV),
                torch.bfloat16,
                [],
            ),
        ),
        (
            "expert count mismatch",
            lambda: fused_moe(
                x, ids, scales, w31, None, w2[:4].contiguous(), None, torch.bfloat16, []
            ),
        ),
        (
            "token count mismatch",
            lambda: fused_moe(x, ids[:4].contiguous(), scales[:4].contiguous(), *args[3:]),
        ),
        (
            "top-k mismatch",
            lambda: fused_moe(x, ids, scales[:, :1].contiguous(), *args[3:]),
        ),
        (
            "hidden_size not a multiple of 8",
            lambda: fused_moe(
                x_bad[:, :132].contiguous(),
                ids_bad,
                sc_bad,
                w31_bad[:, :, :132].contiguous(),
                None,
                w2_bad[:, :132].contiguous(),
                None,
                torch.bfloat16,
                [],
            ),
        ),
        ("min_latency_mode on bf16", lambda: fused_moe(*args, min_latency_mode=True)),
        (
            "deepseek fp8 block scale on bf16",
            lambda: fused_moe(*args, use_deepseek_fp8_block_scale=True),
        ),
        ("int8 woq on bf16", lambda: fused_moe(*args, use_int8_woq_per_channel=True)),
        (
            "mxfp8 weight scaling on bf16",
            lambda: fused_moe(*args, use_mxfp8_weight_scaling=True),
        ),
        (
            "swiglu_alpha with wrong length",
            lambda: fused_moe(*args, swiglu_alpha=torch.ones(1, device=DEV)),
        ),
        (
            "swiglu_alpha in bf16",
            lambda: fused_moe(*args, swiglu_alpha=torch.ones(8, dtype=torch.bfloat16, device=DEV)),
        ),
        # non-gated activation types want an [E, I, H] fc1 instead
        ("Identity activation", lambda: fused_moe(*args, activation_type=1)),
        ("Gelu activation", lambda: fused_moe(*args, activation_type=2)),
        ("Silu activation", lambda: fused_moe(*args, activation_type=4)),
        ("Relu2 activation", lambda: fused_moe(*args, activation_type=8)),
        ("unknown activation", lambda: fused_moe(*args, activation_type=99)),
        ("ep_rank >= ep_size", lambda: fused_moe(*args, ep_size=2, ep_rank=2)),
        (
            "cluster_size without min-latency",
            lambda: fused_moe(*args, cluster_size=2, cluster_rank=0),
        ),
        (
            "tuner_num_tokens without alltoall",
            lambda: fused_moe(*args, tuner_num_tokens=8),
        ),
        ("alltoall without tuner args", lambda: fused_moe(*args, enable_alltoall=True)),
        (
            "lora without max low rank",
            lambda: fused_moe(*args, fc1_lora_ranks=torch.zeros(1, dtype=torch.int32)),
        ),
        (
            "out_tensor with wrong dtype",
            lambda: fused_moe(
                *args, out_tensor=torch.empty(8, 128, dtype=torch.float32, device=DEV)
            ),
        ),
        (
            "out_tensor with wrong shape",
            lambda: fused_moe(
                *args, out_tensor=torch.empty(4, 128, dtype=torch.bfloat16, device=DEV)
            ),
        ),
        (
            "non-contiguous out_tensor",
            lambda: fused_moe(*args, out_tensor=wide_x[:, :128]),
        ),
        (
            "fp8 input without quant scales",
            lambda: fused_moe(
                x.to(torch.float8_e4m3fn),
                ids,
                scales,
                w31.to(torch.float8_e4m3fn),
                None,
                w2.to(torch.float8_e4m3fn),
                None,
                torch.bfloat16,
                [],
            ),
        ),
    ]
    for tag, call in cases:
        try:
            call()
        except (RuntimeError, ValueError, AssertionError):
            continue
        raise AssertionError(f"op accepted an unsupported domain: {tag}")


def test_quant_flags_ignored_on_the_unquantized_path() -> None:
    # These three neither raise nor change the result for bf16 weights; a
    # caller cannot rely on them being honoured.
    x, ids, scales, w31, w2 = _make(32, 256, 128, 16, 4, seed=47)
    base = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
    _assert_moe_close(base, _ref_moe(x, ids, scales, w31, w2))
    args = (x, ids, scales, w31, None, w2, None, torch.bfloat16, [])
    flagged = [
        ("use_w4_group_scaling", fused_moe(*args, use_w4_group_scaling=True)),
        ("use_mxfp8_act_scaling", fused_moe(*args, use_mxfp8_act_scaling=True)),
        ("use_dynamic_fc2_scale", fused_moe(*args, use_dynamic_fc2_scale=True)),
    ]
    for flag, ret in flagged:
        assert torch.equal(ret[0], base), f"{flag}=True changed the result"
    # a non-empty quant_scales list is likewise ignored here
    y = fused_moe(
        x,
        ids,
        scales,
        w31,
        None,
        w2,
        None,
        torch.bfloat16,
        [torch.ones(1, device=DEV)],
    )[0]
    assert torch.equal(y, base), "quant_scales changed the unquantized result"


# ---------------------------------------------------------------------------
# DeepSeek-R1-0528 MTP layer (model.layers.61) routed-expert geometry.
# hidden 7168, moe_intermediate 2048, 256 routed experts, top-8, bf16 weights
# and activations. Under ep_size 4 each rank holds E = 64 of those experts and
# is handed global ids in [0, 256).
R1_H, R1_I, R1_K = 7168, 2048, 8
R1_E_LOCAL, R1_EP = 64, 4
R1_E_GLOBAL = R1_E_LOCAL * R1_EP


def _make_r1_weights(num_experts: int, seed: int):
    """One rank's [E, 2I, H] / [E, H, I] bf16 weight pair, built in chunks.

    5.6 GB at E = 64. Materializing the fp32 randn for the whole stack at once
    would transiently need another 11 GB; chunking caps the temporary at 64 MB.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    w31 = torch.empty(num_experts, 2 * R1_I, R1_H, dtype=torch.bfloat16, device=DEV)
    w2 = torch.empty(num_experts, R1_H, R1_I, dtype=torch.bfloat16, device=DEV)
    chunk = max(1, (1 << 26) // (2 * R1_I * R1_H))
    for lo in range(0, num_experts, chunk):
        hi = min(lo + chunk, num_experts)
        w31[lo:hi] = (torch.randn(hi - lo, 2 * R1_I, R1_H, device=DEV, generator=g) / R1_H**0.5).to(
            torch.bfloat16
        )
        w2[lo:hi] = (torch.randn(hi - lo, R1_H, R1_I, device=DEV, generator=g) / R1_I**0.5).to(
            torch.bfloat16
        )
    return w31, w2


def _make_r1_routing(num_tokens: int, num_experts: int, seed: int):
    """(x, ids, scales) with top-8 routing over `num_experts` global experts."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(num_tokens, R1_H, device=DEV, generator=g).to(torch.bfloat16)
    logits = torch.randn(num_tokens, num_experts, device=DEV, generator=g)
    vals, ids = torch.topk(logits, R1_K, dim=-1)
    return (
        x,
        ids.to(torch.int32).contiguous(),
        torch.softmax(vals, dim=-1).contiguous(),
    )


def _rejects(y: torch.Tensor, ref: torch.Tensor, tag: str) -> None:
    """Assert the gates reject `ref` as a description of `y`."""
    try:
        _assert_moe_close(y, ref)
    except AssertionError:
        return
    raise AssertionError(f"tolerance accepted a wrong reference: {tag}")


def test_r1_mtp_moe_shape_bf16() -> None:
    # 3.5x the largest hidden and 2.7x the largest intermediate size covered by
    # the shapes above, at the expert count one ep_size-4 rank holds.
    # Measured worst deviation over this function's nine comparisons: 2.54 ulp
    # element-wise, 1.16 ulp relative RMS -- inside the 8/4 ulp gates.
    w31, w2 = _make_r1_weights(R1_E_LOCAL, seed=1234)
    for num_tokens in [1, 2, 37, 256, 2048, 8192]:
        x, ids, scales = _make_r1_routing(num_tokens, R1_E_LOCAL, seed=num_tokens)
        y = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
        assert y.shape == (num_tokens, R1_H) and y.is_contiguous()
        _assert_moe_close(y, _ref_moe(x, ids, scales, w31, w2))
        del x, ids, scales, y
        torch.cuda.empty_cache()

    # the two output-side switches, at this geometry
    x, ids, scales = _make_r1_routing(2048, R1_E_LOCAL, seed=7)
    ref = _ref_moe(x, ids, scales, w31, w2)
    y = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
    _assert_moe_close(y, ref)
    y_unfused = fused_moe(
        x,
        ids,
        scales,
        w31,
        None,
        w2,
        None,
        torch.bfloat16,
        [],
        use_fused_finalize=False,
    )[0]
    _assert_moe_close(y_unfused, ref)
    # the flag changes nothing on this path: the cached C++ runner is keyed on
    # the dtypes only, so whichever value the process's first call passed is
    # the one in force -- and both values give the same bits anyway, measured
    # in two fresh processes with the call order swapped
    assert torch.equal(y, y_unfused), "use_fused_finalize changed the result"
    pool = torch.full((2051, R1_H), float("nan"), dtype=torch.bfloat16, device=DEV)
    out = pool[:2048]
    assert fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [], out_tensor=out) == []
    _assert_moe_close(out, ref)
    assert bool(torch.isnan(pool[2048:]).all()), "wrote outside out_tensor's rows"
    del w31, w2, x, ids, scales, ref, y, y_unfused, pool, out
    torch.cuda.empty_cache()


def test_r1_mtp_expert_parallel_4way() -> None:
    # ep_size 4 over a 256-expert global stack: each rank passes its own 64
    # weights and the *global* ids, and answers only for
    # [64*ep_rank, 64*ep_rank + 64). The four rank outputs must sum to the
    # 256-expert result, which is what the caller's correctness rests on.
    # The four windows are never held at once: each is built, driven, folded
    # into the accumulators and freed, so this needs 5.6 GB rather than 22.5.
    token_counts = [1, 1024, 8192]
    acts = {t: _make_r1_routing(t, R1_E_GLOBAL, seed=1000 + t) for t in token_counts}
    got = {t: torch.zeros(t, R1_H, dtype=torch.float32, device=DEV) for t in token_counts}
    want = {t: torch.zeros(t, R1_H, dtype=torch.float32, device=DEV) for t in token_counts}
    for rank in range(R1_EP):
        w31_r, w2_r = _make_r1_weights(R1_E_LOCAL, seed=900 + rank)
        for t in token_counts:
            x, ids, scales = acts[t]
            y = fused_moe(
                x,
                ids,
                scales,
                w31_r,
                None,
                w2_r,
                None,
                torch.bfloat16,
                [],
                ep_size=R1_EP,
                ep_rank=rank,
            )[0]
            ref = _ref_moe(x, ids, scales, w31_r, w2_r, ep_size=R1_EP, ep_rank=rank)
            _assert_moe_close(y, ref)
            got[t] += y.float()
            want[t] += ref.float()
            del y, ref
        if rank == R1_EP - 1:
            # a window that catches nothing contributes exactly zero, and the
            # same weights read under the wrong window are a different answer
            x, ids, scales = acts[1024]
            inside_rank0 = (ids % R1_E_LOCAL).to(torch.int32).contiguous()
            y = fused_moe(
                x,
                inside_rank0,
                scales,
                w31_r,
                None,
                w2_r,
                None,
                torch.bfloat16,
                [],
                ep_size=R1_EP,
                ep_rank=rank,
            )[0]
            assert bool((y == 0).all()), "an out-of-window rank wrote nonzero output"
            y = fused_moe(
                x,
                ids,
                scales,
                w31_r,
                None,
                w2_r,
                None,
                torch.bfloat16,
                [],
                ep_size=R1_EP,
                ep_rank=0,
            )[0]
            _rejects(
                y,
                _ref_moe(x, ids, scales, w31_r, w2_r, ep_size=R1_EP, ep_rank=rank),
                "rank 3's weights driven at ep_rank=0",
            )
            del inside_rank0, y
        del w31_r, w2_r
        torch.cuda.empty_cache()
    for t in token_counts:
        # each rank rounds its own partial to bf16 before this add, so the sum
        # sits slightly wider than any single rank does (measured 3.53 ulp
        # element-wise at 8192 tokens against 2.69 for the widest single rank)
        _assert_moe_close(got[t].to(torch.bfloat16), want[t].to(torch.bfloat16))
    del acts, got, want
    torch.cuda.empty_cache()


def test_r1_mtp_expert_relabeling_is_bitwise() -> None:
    # Nothing in the permutation/gather path may key on which expert index a
    # weight sits at: permuting the 64-expert stack and relabelling the ids to
    # match must reproduce the same output. Measured bitwise identical.
    w31, w2 = _make_r1_weights(R1_E_LOCAL, seed=2222)
    x, ids, scales = _make_r1_routing(512, R1_E_LOCAL, seed=31337)
    y = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
    _assert_moe_close(y, _ref_moe(x, ids, scales, w31, w2))
    g = torch.Generator(device=DEV).manual_seed(4)
    perm = torch.randperm(R1_E_LOCAL, device=DEV, generator=g)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(R1_E_LOCAL, device=DEV)
    y_perm = fused_moe(
        x,
        inv[ids.long()].to(torch.int32).contiguous(),
        scales,
        w31[perm].contiguous(),
        None,
        w2[perm].contiguous(),
        None,
        torch.bfloat16,
        [],
    )[0]
    assert torch.equal(y, y_perm), "relabelling the experts changed the output"
    del w31, w2, x, ids, scales, y, y_perm, perm, inv
    torch.cuda.empty_cache()


def test_r1_mtp_reference_discriminates() -> None:
    # The 8/4 ulp gates must reject wrong computations at this geometry too,
    # not just at the small shapes. Measured against the correct result's 2.09
    # ulp: 284 / 197 ulp for the swapped halves and 997 / 212 for a dropped
    # expert, i.e. 36x and 125x the element gate, 49x and 53x the RMS gate.
    w31, w2 = _make_r1_weights(R1_E_LOCAL, seed=555)
    x, ids, scales = _make_r1_routing(512, R1_E_LOCAL, seed=556)
    y = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
    _assert_moe_close(y, _ref_moe(x, ids, scales, w31, w2))
    swapped = torch.cat([w31[:, R1_I:], w31[:, :R1_I]], dim=1).contiguous()
    _rejects(y, _ref_moe(x, ids, scales, swapped, w2), "gate/up halves swapped")
    del swapped
    torch.cuda.empty_cache()
    dropped = scales.clone()
    dropped[:, 0] = 0.0
    _rejects(y, _ref_moe(x, ids, dropped, w31, w2), "one expert dropped")
    del w31, w2, x, ids, scales, y, dropped
    torch.cuda.empty_cache()


def test_r1_mtp_autotuned_tactics() -> None:
    # Serving runs on the hot side of the tuner, which a cold receipt never
    # sees. One autotune() pass at this geometry fills 2 tunable GEMMs x 14
    # power-of-2 token buckets, moves the bits, and still lands inside the
    # gates; clearing the cache restores the cold bits exactly.
    tuner = AutoTuner.get()
    tuner.clear_cache()
    w31, w2 = _make_r1_weights(R1_E_LOCAL, seed=1234)
    cases = {}
    for t in (1, 256, 8192):
        x, ids, scales = _make_r1_routing(t, R1_E_LOCAL, seed=t)
        cold = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
        cases[t] = (x, ids, scales, cold, _ref_moe(x, ids, scales, w31, w2))
        _assert_moe_close(cold, cases[t][4])
    try:
        x, ids, scales = cases[8192][:3]
        with autotune():
            fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])
        assert len(tuner.profiling_cache) == 28, len(tuner.profiling_cache)
        for key, (_, tactic, _) in tuner.profiling_cache.cache.items():
            assert tactic >= 0, f"{key} kept the fallback tactic after tuning"
        moved = 0
        for t, (x, ids, scales, cold, ref) in cases.items():
            hot = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
            _assert_moe_close(hot, ref)
            moved += int(not torch.equal(hot, cold))
        # the control this whole test rests on: if tuning had not changed a
        # single bit, a clean pass here would mean nothing
        assert moved > 0, "no warm result differed from its cold counterpart"
    finally:
        tuner.clear_cache()
    for t, (x, ids, scales, cold, _) in cases.items():
        again = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
        assert torch.equal(again, cold), f"clearing the cache did not restore T={t}"
    del w31, w2, cases
    torch.cuda.empty_cache()


def _sweep_tactics(combos, args, ref, count_distinct: bool) -> int:
    """Drive the op once per (runner, tactic) config, gating every output."""
    distinct = set()
    for cfg in combos:
        with AutoTuner.get().replay(cfg):
            y = fused_moe(*args)[0]
        _assert_moe_close(y, ref)
        if count_distinct:
            distinct.add(y.view(torch.int16).cpu().numpy().tobytes())
        del y
    return len(distinct)


def _split_index(combos) -> int:
    """Number of gemm2 tactics: itertools.product varies that context fastest."""
    return next(i for i in range(1, len(combos)) if combos[i][0][1] != combos[0][0][1])


def test_r1_mtp_tactic_space() -> None:
    # get_valid_tactics takes no shape, so the tactic population is a property
    # of the build, not of the geometry -- checked here by capturing it at a
    # certified small cell and at this one and comparing the sizes. Then every
    # tactic in it is driven against the torch reference, which makes the
    # receipt cover the warm path as a whole rather than one tuner outcome.
    tuner = AutoTuner.get()
    x_s, ids_s, scales_s, w31_s, w2_s = _make(64, 512, 256, 32, 4, seed=61)
    with tuner.capture() as cap_small:
        fused_moe(x_s, ids_s, scales_s, w31_s, None, w2_s, None, torch.bfloat16, [])
    small = list(cap_small)
    del x_s, ids_s, scales_s, w31_s, w2_s
    torch.cuda.empty_cache()

    w31, w2 = _make_r1_weights(R1_E_LOCAL, seed=1234)
    x, ids, scales = _make_r1_routing(256, R1_E_LOCAL, seed=99)
    args = (x, ids, scales, w31, None, w2, None, torch.bfloat16, [])
    ref = _ref_moe(x, ids, scales, w31, w2)
    with tuner.capture() as cap:
        fused_moe(*args)
    combos = list(cap)
    assert len(combos) == len(small), (len(combos), len(small))
    n2 = _split_index(combos)
    n1 = len(combos) // n2
    assert n1 > 1 and n2 > 1 and n1 * n2 == len(combos), (n1, n2)

    # gemm2 carries nearly all of the spread: measured 101 distinct outputs
    # from its 309 tactics here against 2 from gemm1's 209, the same counts the
    # small shapes give. Worst deviation over the whole space: 3.71 ulp
    # element-wise here and 3.98 at 8192 tokens below -- inside the 8 ulp gate
    # with 2.0x margin.
    d2 = _sweep_tactics(combos[:n2], args, ref, count_distinct=True)
    d1 = _sweep_tactics(combos[::n2], args, ref, count_distinct=True)
    # the blindness control: a harness that could not tell one tactic from
    # another would report the same clean sweep with nothing measured
    assert d2 > 1, d2
    assert d1 < d2, (d1, d2)
    del x, ids, scales, args, ref, combos
    torch.cuda.empty_cache()

    # and at the token count the caller chunks to
    x, ids, scales = _make_r1_routing(8192, R1_E_LOCAL, seed=8192)
    args = (x, ids, scales, w31, None, w2, None, torch.bfloat16, [])
    ref = _ref_moe(x, ids, scales, w31, w2)
    with tuner.capture() as cap:
        fused_moe(*args)
    combos = list(cap)
    _sweep_tactics(combos[: _split_index(combos)], args, ref, count_distinct=False)
    del w31, w2, x, ids, scales, args, ref, combos
    torch.cuda.empty_cache()
