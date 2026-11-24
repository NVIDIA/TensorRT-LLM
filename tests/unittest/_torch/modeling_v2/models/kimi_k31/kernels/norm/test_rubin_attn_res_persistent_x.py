# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``rubin_attn_res_persistent_x`` (CuTe DSL): the fused epilogues keep every byte.

Each case compares the kernel with what production runs without it on the same inputs: the engine's
attn_res_add_rmsnorm_persistent_fwd, the MoE residual update bf16(residual + bf16(routed + shared)) as two torch
bf16 adds, output.float(), and mxfp8_quantize(output, swizzled_layout=True, 32). Byte equality, every N in 2..9,
at the K3.1 Rubin DEP16 generation step's 96 rows, just above the per-token range (33) and across two 128-row scale
tiles (160), grid cap off and on. The MXFP8 form is also checked on groups where a fast-math (FTZ) quantizer would
differ from the IEEE one the image runs: group maxima whose scale amax / 448 is subnormal above 2^-127, groups
holding an infinity, zero groups and an all-NaN row. A CUDA graph replays the launch with the same bytes, and the
launch without PDL gives them too.
"""

import pytest
import torch

H = 7168
EPS = 1e-5

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM 10.x GPU",
)


@pytest.fixture(scope="module", autouse=True)
def _ops():
    import tensorrt_llm._torch.custom_ops  # noqa: F401  (registers torch.ops.trtllm.*)
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not IS_CUTLASS_DSL_AVAILABLE:
        pytest.skip("needs the CuTe DSL")


def _kx():
    """The kernel module (imported on use: the module fixture has checked for the CuTe DSL)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.norm import (
        rubin_attn_res_persistent_x,
    )

    return rubin_attn_res_persistent_x


def _inputs(n, t, seed):
    g = torch.Generator(device="cuda").manual_seed(1000 * n + 7 * t + seed)

    def rn(shape, scale=1.0, offset=0.0):
        return (torch.randn(shape, generator=g, device="cuda") * scale + offset).to(torch.bfloat16)

    bank = rn((n - 1, t, H))
    prefix = rn((t, H))
    addend = rn((t, H), 0.5)
    routed = rn((t, H), 0.5)
    shared = rn((t, H), 0.5)
    w = (rn((H,), 0.05), rn((H,), 0.1, 1.0), rn((H,), 0.1, 1.0))  # res_proj, res_norm, out_norm
    return bank, prefix, addend, routed, shared, w


def _bits(t):
    t = t.contiguous()
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def _eq(a, b, what):
    assert a.shape == b.shape, (what, a.shape, b.shape)
    n = int((_bits(a) != _bits(b)).sum())
    assert n == 0, f"{what}: {n} of {a.numel()} elements differ"


def _bytes_or_nan(a, b, what):
    """Byte equality except where both are NaN (torch's bf16 add and the kernel's write different NaN bits)."""
    assert a.shape == b.shape, (what, a.shape, b.shape)
    n = int(((_bits(a) != _bits(b)) & ~(torch.isnan(a) & torch.isnan(b))).sum())
    assert n == 0, f"{what}: {n} of {a.numel()} elements differ"


def _today(prefix, addend, bank, w):
    t, k = prefix.shape[0], bank.shape[0]
    upd, out = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd(
        prefix.view(t, 1, H),
        None if addend is None else addend.view(t, 1, H),
        bank.view(k, t, 1, H),
        w[0],
        w[1],
        w[2],
        EPS,
        EPS,
    )
    return upd.view(t, H), out.view(t, H)


def _x(prefix, addend, addend2, bank, w, f32=False, mx=False, cap=False):
    return _kx().attn_res_add_rmsnorm_persistent_x(
        prefix, addend, addend2, bank, w[0], w[1], w[2], EPS, EPS, f32, mx, cap
    )


def _moe_update(prefix, routed, shared):
    moe = (routed.float() + shared.float()).to(torch.bfloat16)
    return (prefix.float() + moe.float()).to(torch.bfloat16)


ROWS = (96, 33, 160)


@pytest.mark.parametrize("cap", [False, True])
@pytest.mark.parametrize("t", ROWS)
@pytest.mark.parametrize("n", range(2, 10))
def test_plain_and_addend(n, t, cap):
    for seed in (0, 1):
        bank, prefix, addend, _, _, w = _inputs(n, t, seed)
        _, out = _today(prefix, None, bank, w)
        u, o, f, c, s = _x(prefix, None, None, bank, w, cap=cap)
        _eq(o, out, "output, no addend")
        assert u is prefix and f is None and c is None and s is None
        upd, out = _today(prefix, addend, bank, w)
        u, o, _, _, _ = _x(prefix, addend, None, bank, w, cap=cap)
        _eq(u, upd, "updated, addend")
        _eq(o, out, "output, addend")


@pytest.mark.parametrize("cap", [False, True])
@pytest.mark.parametrize("t", ROWS)
@pytest.mark.parametrize("n", range(2, 10))
def test_moe_residual_fold(n, t, cap):
    for seed in (0, 1):
        bank, prefix, _, routed, shared, w = _inputs(n, t, seed)
        updated = _moe_update(prefix, routed, shared)
        _, out = _today(updated, None, bank, w)
        u, o, _, _, _ = _x(prefix, routed, shared, bank, w, cap=cap)
        _eq(u, updated, "updated = prefix + bf16(routed + shared)")
        _eq(o, out, "output after the MoE residual fold")


@pytest.mark.parametrize("cap", [False, True])
@pytest.mark.parametrize("t", ROWS)
@pytest.mark.parametrize("n", range(2, 10))
def test_fp32_copy(n, t, cap):
    for seed in (0, 1):
        bank, prefix, addend, _, _, w = _inputs(n, t, seed)
        for add in (None, addend):
            upd, out = _today(prefix, add, bank, w)
            u, o, f, _, _ = _x(prefix, add, None, bank, w, f32=True, cap=cap)
            _eq(o, out, "output")
            _eq(f, out.float(), "FP32 copy = output.float()")
            if add is not None:
                _eq(u, upd, "updated")


@pytest.mark.parametrize("cap", [False, True])
@pytest.mark.parametrize("t", ROWS)
@pytest.mark.parametrize("n", range(2, 10))
def test_mxfp8_form_with_moe_fold(n, t, cap):
    for seed in (0, 1):
        bank, prefix, _, routed, shared, w = _inputs(n, t, seed)
        updated = _moe_update(prefix, routed, shared)
        _, out = _today(updated, None, bank, w)
        codes, scales = torch.ops.trtllm.mxfp8_quantize(out, True, 32)
        u, o, _, c, s = _x(prefix, routed, shared, bank, w, mx=True, cap=cap)
        _eq(u, updated, "updated")
        _eq(o, out, "output")
        _eq(c.view(torch.uint8), codes.view(torch.uint8).view(t, H), "MXFP8 codes")
        _eq(
            s, scales.view(torch.uint8).reshape(-1), "MXFP8 swizzled scales (padding rows included)"
        )


# Per 32-column group of the output norm's weight: normal, the band where amax / 448 is subnormal above 2^-127
# (amax ~2.63e-36 .. 5.27e-36, scale byte 1 under IEEE, 0 under FTZ), overflow to infinity (scale byte 254, whose
# multiplier 2^-127 is subnormal), zero, tiny normal maxima, just below the band.
_EDGE_SCALES = (
    1.0,
    1.1e-36,
    1.4e-36,
    1.7e-36,
    2.0e-36,
    2.4e-36,
    3.0e38,
    0.0,
    1e-30,
    6e-37,
    1e-35,
    1.25e-36,
)


@pytest.mark.parametrize("cap", [False, True])
@pytest.mark.parametrize("t", (96, 160))
@pytest.mark.parametrize("n", (2, 6, 9))
def test_mxfp8_form_on_edge_groups(n, t, cap):
    bank, prefix, _, routed, shared, w = _inputs(n, t, 2)
    prefix[5, 0] = float("nan")  # an all-NaN output row
    scales = torch.tensor(_EDGE_SCALES, device="cuda")
    group = scales[(torch.arange(H // 32, device="cuda") + n) % scales.numel()].repeat_interleave(
        32
    )
    g = torch.Generator(device="cuda").manual_seed(n + t)
    w = (
        w[0],
        w[1],
        (group * (1 + 0.1 * torch.randn(H, generator=g, device="cuda").abs())).to(torch.bfloat16),
    )
    updated = _moe_update(prefix, routed, shared)
    _, out = _today(updated, None, bank, w)
    groups = out.view(t, H // 32, 32).float()
    amax = groups.abs().nan_to_num(nan=0.0).amax(-1)
    assert int(((amax > 2.63e-36) & (amax < 5.27e-36)).sum()) > 0, (
        "no group in the subnormal-scale band"
    )
    assert int(torch.isinf(groups).any(-1).sum()) > 0, "no group holding an infinity"
    codes, scales_ref = torch.ops.trtllm.mxfp8_quantize(out, True, 32)
    u, o, _, c, s = _x(prefix, routed, shared, bank, w, mx=True, cap=cap)
    _bytes_or_nan(u, updated, "updated")
    _bytes_or_nan(o, out, "output")
    _eq(c.view(torch.uint8), codes.view(torch.uint8).view(t, H), "MXFP8 codes on edge groups")
    _eq(s, scales_ref.view(torch.uint8).reshape(-1), "MXFP8 scales on edge groups")


@pytest.mark.parametrize("n", (2, 9))
def test_graph_replay_and_no_pdl_keep_the_bytes(n):
    t = 96
    bank, prefix, _, routed, shared, w = _inputs(n, t, 3)
    eager = _x(prefix, routed, shared, bank, w, mx=True, cap=True)
    outs = [torch.empty_like(prefix), torch.empty_like(prefix)]
    codes = torch.empty((t, H), dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.empty(128 * 224, dtype=torch.uint8, device="cuda")

    def launch(use_pdl=None):
        _kx().run_persistent_x(
            prefix,
            routed,
            shared,
            bank,
            w[0],
            w[1],
            w[2],
            EPS,
            EPS,
            output=outs[1],
            updated=outs[0],
            mx_codes=codes,
            mx_scales=scales,
            grid_cap=True,
            use_pdl=use_pdl,
        )

    want = (eager[0], eager[1], eager[3], eager[4])
    names = ("updated", "output", "codes", "scales")
    for use_pdl in (False, True):
        launch(use_pdl)
        torch.cuda.synchronize()
        for got, ref, what in zip((outs[0], outs[1], codes, scales), want, names):
            _eq(got, ref, f"{what}, use_pdl={use_pdl}")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for out in outs:
        out.zero_()
    codes.view(torch.uint8).zero_()
    scales.zero_()
    graph.replay()
    torch.cuda.synchronize()
    for got, ref, what in zip((outs[0], outs[1], codes, scales), want, names):
        _eq(got, ref, f"{what}, graph replay")


def test_refuses_unbuilt_flag_sets():
    bank, prefix, addend, routed, shared, w = _inputs(4, 96, 0)
    with pytest.raises(ValueError):
        _x(prefix, None, shared, bank, w)  # a second addend without the first
    with pytest.raises(ValueError):
        _x(prefix, addend, None, bank, w, mx=True)  # the MXFP8 form only with the MoE fold
    with pytest.raises(ValueError):
        _x(prefix, routed, shared, bank, w, f32=True)  # the FP32 copy has no MoE-fold variant
    with pytest.raises(ValueError):
        _x(prefix, None, None, bank[:0], w)  # N = 1
    with pytest.raises(ValueError):
        _x(prefix, None, None, torch.cat([bank, bank, bank]), w)  # N = 10
