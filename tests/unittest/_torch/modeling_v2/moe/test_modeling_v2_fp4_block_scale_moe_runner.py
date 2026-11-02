# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp4_block_scale_moe_runner catalog entry."""

import zlib
from collections.abc import Callable, Iterator

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    Fp4BlockScaleMoeRunner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
    fp4_block_scale_moe_runner as moe,
)
from tensorrt_llm._torch.autotuner import AutoTuner, autotune

__extra_import_path__ = [".."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _fp4_moe_test_utils import (  # noqa: E402
    DEV,
    E2M1_MAX,
    E4M3_MAX,
    SV,
    _assert_moe_close,
    _build,
    _calibrate_g2,
    _call,
    _compare,
    _dequant,
    _dev,
    _e2m1_rne,
    _fc1_perm,
    _prep_fc1,
    _prep_fc2,
    _q_nvfp4,
    _rand_activation,
    _rand_nvfp4,
    _ref_moe,
    _renorm_routing,
    _routing,
    _scalars,
    _swizzle_scales,
    _unpack_activation,
    many_expert_layer,  # noqa: E402,F401 — fixture by name
)
from _precision import true_fp32_matmul  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "fp4_block_scale_moe_runner requires a CUDA device"


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


# The last R1 layer built at a moe_tp slice of the expert intermediate. One
# slot rather than one per width: a 256-expert layer is several GB of codes.
_DSR1_SLICE: dict = {}


def _dsr1_at(inter: int):
    """The R1 layer at an expert intermediate of `inter`.

    A moe_tp rank's slice is a layer of its own: the target cuts the up/gate
    rows and down columns before the kernel layout, so the kernel sees an
    ordinary layer of that width (the deepseek default shard test pins the
    cut-then-layout order)."""
    if inter == 2048:
        return _dsr1()
    if inter not in _DSR1_SLICE:
        _DSR1_SLICE.clear()
        _DSR1_SLICE[inter] = _build(256, 7168, inter, seed=256 + inter)
    return _DSR1_SLICE[inter]


@pytest.fixture(scope="module", autouse=True)
def _release_reference_layers() -> Iterator[None]:
    global _DSV3, _DSR1
    try:
        yield
    finally:
        _DSV3 = _DSR1 = None
        _DSR1_SLICE.clear()


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


# The token column swept at that geometry. The pre-routed path picks its
# permutation kernel by row count: `BlockKernelMaxNumTokens` takes 4 rows and
# below, the dynamic-block kernel is out of reach here because
# `DynBlockKernelMaxNumExperts` is 256, and `MaxNumTokensSingleCluster` carries
# everything from 5 rows up to 8192 (all three in
# `cpp/tensorrt_llm/kernels/moe/trtllmGen/routing/RoutingCustomPolicy.cuh`). So
# the column brackets the 4/5 boundary as well as the decode and prefill ends.
_MANY_TOKENS = (1, 2, 3, 4, 5, 8, 16, 64, 256, 1024, 4096, 8192)


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

    with true_fp32_matmul():
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


def test_many_experts_top10_geometry(many_expert_layer):  # noqa: F811
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
    args, ref = many_expert_layer()
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
        with true_fp32_matmul():
            exp = moe.reference(x, ids, wts, ref["up"], ref["gate"], ref["down"], g2).to(
                torch.bfloat16
            )
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
        with true_fp32_matmul():
            exp = moe.reference(x, ids, wts, ref["up"], ref["gate"], ref["down"], g2).to(
                torch.bfloat16
            )
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

    with true_fp32_matmul():
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


def test_many_experts_autotuner_cache_is_inert(clear_autotuner_cache, many_expert_layer):  # noqa: F811
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
    args, ref = many_expert_layer()
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


# The shared layer each cell's (num_experts, hidden, intermediate) names. An
# R1 cell at a moe_tp slice of the intermediate takes `_dsr1_at` instead.
def _cell_layer(spec: dict, layer_factory: Callable[[], tuple[dict, dict]]) -> tuple[dict, dict]:
    """The shared layer and reference a cell's geometry names."""
    key = (spec["num_experts"], spec["hidden"], spec["intermediate"])
    if key == (512, 2560, 640):
        return layer_factory()
    if key == (256, 7168, 2048):
        return _dsr1()
    assert key[:2] == (256, 7168), f"no shared layer for cell geometry {key}"
    return _dsr1_at(key[2])


# qwen4_exp's (Qwen3.8-Flash-Next) routed geometry: many small experts, a
# top_k past 8, an intermediate width of 640, all 512 experts local. Its
# router renormalizes the top-10 softmax weights to sum to 1 before the
# call (`routing="renormalized"`, a test axis rather than an op argument;
# the R1 cells draw uniform [0, 1) weights). Nothing is padded:
# H % 256, 2I % 128, H/16 % 4 and I/16 % 4 are all zero.
_QWEN4_EXP = dict(
    hidden=2560,
    intermediate=640,
    num_experts=512,
    top_k=10,
    local_expert_offset=0,
    local_num_experts=512,
    routing="renormalized",
)

# The shapes each shipped target calls this entry with, by target directory
# under `models/`. The entry's own `CELLS` are the op's whoever calls it.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "deepseek_v3/r1_0528_nvfp4__sm_10x__default": (
        Cell(
            why="deepseek default at tp2: all 256 experts, intermediate split 2 ways (1024)",
            spec=dict(
                tokens=64,
                local_expert_offset=0,
                local_num_experts=256,
                **dict(Fp4BlockScaleMoeRunner.R1, intermediate=1024),
            ),
        ),
        Cell(
            why="deepseek default at tp4: all 256 experts, intermediate split 4 ways (512)",
            spec=dict(
                tokens=64,
                local_expert_offset=0,
                local_num_experts=256,
                **dict(Fp4BlockScaleMoeRunner.R1, intermediate=512),
            ),
        ),
        Cell(
            why="deepseek default at tp8: all 256 experts, intermediate split 8 ways (256)",
            spec=dict(
                tokens=64,
                local_expert_offset=0,
                local_num_experts=256,
                **dict(Fp4BlockScaleMoeRunner.R1, intermediate=256),
            ),
        ),
        Cell(
            why="deepseek default at ep8 (tp8 + ep8 and dep8): the last 32-wide window",
            spec=dict(
                tokens=64,
                local_expert_offset=224,
                local_num_experts=32,
                **Fp4BlockScaleMoeRunner.R1,
            ),
        ),
        Cell(
            why="deepseek default at dep16: a 16-wide window",
            spec=dict(
                tokens=64,
                local_expert_offset=112,
                local_num_experts=16,
                **Fp4BlockScaleMoeRunner.R1,
            ),
        ),
    ),
    "qwen4_exp/qwen3_8_flash_next_nvfp4__sm_103__tp1": (
        Cell(
            why="qwen4_exp decode: one token over 512 experts, top-10 renormalized, H 2560 / I 640",
            spec=dict(tokens=1, **_QWEN4_EXP),
        ),
        Cell(
            why="qwen4_exp at 4 rows: the last count the pre-routed path's static-block "
            "permutation kernel takes",
            spec=dict(tokens=4, **_QWEN4_EXP),
        ),
        Cell(
            why="qwen4_exp at 5 rows: the first count on the single-cluster permutation "
            "kernel (the dynamic-block one stops at 256 experts)",
            spec=dict(tokens=5, **_QWEN4_EXP),
        ),
        Cell(
            why="qwen4_exp prefill: 8192 rows, the single-cluster kernel's cap and the top "
            "of the certified column",
            spec=dict(tokens=8192, **_QWEN4_EXP),
        ),
    ),
}

CELLS = certified_cells(moe, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying, many_expert_layer):  # noqa: F811
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
    args, ref = _cell_layer(spec, many_expert_layer)
    off, n = spec["local_expert_offset"], spec["local_num_experts"]
    num_experts, hidden, top_k = spec["num_experts"], spec["hidden"], spec["top_k"]
    gen = torch.Generator(device=DEV).manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    g1 = 137.0

    data, sf, x = _rand_activation(spec["tokens"], hidden, g1, gen)
    routing = _renorm_routing if spec.get("routing") == "renormalized" else _routing
    ids, wts = routing(spec["tokens"], num_experts, top_k, gen)
    g2 = _calibrate_g2(x, ids, ref)
    s1, sg, s2 = _scalars(num_experts, g1, g2)

    win_args = args if n == num_experts else _window_args(args, off, n)
    win_ref = ref if n == num_experts else _window_ref(ref, off, n)

    # 1. The entry's reference against this file's, on identical inputs.
    with true_fp32_matmul():
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
