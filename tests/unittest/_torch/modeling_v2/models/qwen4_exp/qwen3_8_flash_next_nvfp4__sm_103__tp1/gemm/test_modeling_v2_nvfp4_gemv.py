# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: `nvfp4_gemv` against an fp32 reference over the dequantized NVFP4 weight, per
cell; the quantizer against its definition; CUDA-graph replay; one compiled kernel per row
octet; the f16-range guard. The split entries (NVFP4 units on `fp8_gemv`'s kernel) per cell:
the projection against the fp32 reference, the conv entries also against the two-launch
paths they replace (the split projection + `causal_conv1d_update` /
`causal_conv1d_triton_update`)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.causal_conv1d_triton_update import (  # noqa: E501
    causal_conv1d_triton_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.causal_conv1d_update import (  # noqa: E501
    causal_conv1d_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.gemm import (  # noqa: E501
    nvfp4_gemv as nv,
)

assert torch.cuda.is_available(), "nvfp4_gemv requires a CUDA device"

DEV = "cuda"
nvfp4_gemv_op = nv.nvfp4_gemv_op


def _weight(n, k, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=DEV) * k**-0.5
    w[:, 7] *= 30.0  # outlier column: its blocks' scales run up to e4m3's top
    w[: min(n, 64), 128:256] *= 1e-4  # near-zero blocks: small (subnormal) e4m3 scales
    w = w.to(torch.bfloat16)
    codes, scales, gscale = nv.quantize_nvfp4(w)
    return w, codes, scales, gscale, nv.pack_weight(codes, scales, gscale)


def _x(m, k, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    # A column slice of a wider row, as the target's split activations hand over.
    return torch.randn(m, k + 256, generator=g, device=DEV).to(torch.bfloat16)[:, :k]


def _close(y, ref, x, deq) -> None:
    """Within one bf16 rounding of the fp32 reference, plus fp32 summation noise."""
    with true_fp32_matmul():
        absref = x.float().abs() @ deq.abs().t()
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    excess = ((y.float() - ref).abs() - 0.5 * ulp).clamp_min(0) / absref.clamp_min(1e-30)
    assert excess.max().item() < 1e-5, f"error beyond one rounding: {excess.max().item():.2e}"


@pytest.mark.parametrize("cell", nvfp4_gemv_op.CELLS, ids=[c.why[:40] for c in nvfp4_gemv_op.CELLS])
def test_gemv_cells(cell) -> None:
    n, k, m = cell.spec["n"], cell.spec["k"], cell.spec["m"]
    _, codes, scales, gscale, packed = _weight(n, k, n + m)
    x = _x(m, k, m)
    nvfp4_gemv_op.is_valid(x, packed)
    y = nvfp4_gemv_op(x, packed)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    ref = nvfp4_gemv_op.reference(x, codes, scales, gscale)
    _close(y, ref, x, nv.dequantize_nvfp4(codes, scales, gscale))


def test_quantizer_matches_its_definition() -> None:
    """Codes are the nearest e2m1 value of w / (block scale x global scale); the largest block
    scale is e4m3's 448; the relative error is NVFP4's (well under 2^-2 a row)."""
    w, codes, scales, gscale, _ = _weight(512, 1024, 3)
    assert codes.max().item() <= 15 and torch.isfinite(scales.float()).all()
    assert scales.float().max().item() == 448.0
    step = scales.float().repeat_interleave(nv.SF_VEC, 1) * gscale
    v = torch.where(step > 0, w.float() / step.clamp_min(1e-38), 0.0)
    grid = torch.tensor(nv.E2M1_VALUES, device=DEV)
    best = (v.abs()[..., None] - grid).abs().min(-1).values
    got = nv.dequantize_nvfp4(codes, scales, gscale) / step.clamp_min(1e-38)
    err = (v - torch.where(step > 0, got, 0.0)).abs()
    assert (err <= best + 1e-6).all(), "a code is not the nearest e2m1 value"
    deq = nv.dequantize_nvfp4(codes, scales, gscale)
    rel = (deq - w.float()).norm(dim=1) / w.float().norm(dim=1)
    assert rel.max().item() < 0.2, rel.max().item()


def test_chunked_pack_matches_whole() -> None:
    """`pack_from_bf16` (row chunks, a partial last tile) is `pack_weight` of the whole."""
    w = torch.randn(1000, 256, device=DEV).to(torch.bfloat16)
    a = nv.pack_from_bf16(w, rows_per_chunk=192)
    b = nv.pack_weight(*nv.quantize_nvfp4(w))
    assert a.global_scale == b.global_scale and torch.equal(a.data, b.data)


def test_one_kernel_per_row_octet() -> None:
    """The row count is a launch argument: rows 1..16 build two kernels per shape."""
    _, codes, scales, gscale, packed = _weight(2048, 512, 5)
    before = nv._compiled.cache_info().currsize
    for m in range(1, nv.MAX_ROWS + 1):
        x = _x(m, 512, m)
        _close(
            nvfp4_gemv_op(x, packed),
            nvfp4_gemv_op.reference(x, codes, scales, gscale),
            x,
            nv.dequantize_nvfp4(codes, scales, gscale),
        )
    assert nv._compiled.cache_info().currsize - before <= 2


def test_graph_replay_matches_eager() -> None:
    _, _, _, _, packed = _weight(4096, 2560, 9)
    x = torch.randn(4, 2560, device=DEV).to(torch.bfloat16)
    nvfp4_gemv_op(x, packed)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = nvfp4_gemv_op(x, packed)
    x.copy_(torch.randn(4, 2560, device=DEV).to(torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, nvfp4_gemv_op(x, packed))


def test_is_valid_rejects_activation_beyond_f16() -> None:
    _, _, _, _, packed = _weight(256, 256, 1)
    x = torch.randn(2, 256, device=DEV).to(torch.bfloat16)
    nvfp4_gemv_op.is_valid(x, packed)
    x[1, 3] = 1e5
    with pytest.raises(AssertionError):
        nvfp4_gemv_op.is_valid(x, packed)


# ------------------------------------------------------------------------------------------
# split entries: NVFP4 units on fp8_gemv's kernel
# ------------------------------------------------------------------------------------------

split_op = nv.nvfp4_split_gemv_op
split_conv_op = nv.nvfp4_split_gemv_conv1d_op
split_verify_op = nv.nvfp4_split_gemv_conv1d_verify_op
GDN_N, GDN_K, GDN_CONV = 16480, 2560, 10240


@pytest.mark.parametrize("cell", split_op.CELLS, ids=[c.why[:40] for c in split_op.CELLS])
def test_split_gemv_cells(cell) -> None:
    n, k, m = cell.spec["n"], cell.spec["k"], cell.spec["m"]
    _, codes, scales, gscale, _ = _weight(n, k, n + m)
    packed = nv.pack_split(codes, scales, gscale)
    x = _x(m, k, m)
    split_op.is_valid(x, packed)
    y = split_op(x, packed)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    _close(
        y,
        split_op.reference(x, codes, scales, gscale),
        x,
        nv.dequantize_nvfp4(codes, scales, gscale),
    )


def test_split_layouts_agree() -> None:
    """Tile-major and k-block-major units give the same projection (same plan)."""
    _, codes, scales, gscale, _ = _weight(1001, 384, 4)
    x = _x(5, 384, 5)
    a = split_op(x, nv.pack_split(codes, scales, gscale, tile_major=True))
    b = split_op(x, nv.pack_split(codes, scales, gscale, tile_major=False))
    assert torch.equal(a, b)


@pytest.mark.parametrize("cell", split_conv_op.CELLS, ids=[c.why[:40] for c in split_conv_op.CELLS])
def test_split_conv1d_cells(cell) -> None:
    pool, pad = 64, -1
    _, codes, scales, gscale, _ = _weight(GDN_N, GDN_K, 3)
    packed = nv.pack_split(codes, scales, gscale)
    deq = nv.dequantize_nvfp4(codes, scales, gscale)
    slots = list(cell.spec["slots"])
    idx = torch.tensor(slots, dtype=torch.int32, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(11)
    backing = torch.randn(pool, GDN_CONV, 4, generator=g, device=DEV).to(torch.bfloat16)
    conv_state = backing[:, :, 1:]  # a strided pool view
    conv_weight = torch.randn(GDN_CONV, 4, generator=g, device=DEV).to(torch.bfloat16)
    x = torch.randn(len(slots), GDN_K, generator=g, device=DEV).to(torch.bfloat16)
    before = conv_state.clone()
    split_conv_op.is_valid(x, packed)
    out = split_conv_op(x, packed, conv_state, conv_weight, idx, GDN_CONV, pad)

    # Against the two-launch path over the same projection.
    proj = split_op(x, packed)
    state_b = before.clone()
    two = proj.clone()
    causal_conv1d_update(
        x=two[:, :GDN_CONV],
        conv_state=state_b,
        weight=conv_weight,
        bias=None,
        silu_activation=True,
        cache_seqlens=None,
        conv_state_indices=idx,
        pad_slot_id=pad,
    )
    assert torch.equal(out[:, GDN_CONV:], proj[:, GDN_CONV:])
    assert (out == two).float().mean().item() > 0.999
    assert ((out.float() - two.float()).abs() / two.float().abs().clamp_min(1e-2)).max() < 0.02
    assert torch.equal(conv_state, state_b), "the pool as the conv kernel leaves it"

    # Against the fp32 reference: the projection may round one ulp differently (fp32 summation
    # order, the activation rounded to f16), and a flip carries through the conv (one tap times
    # one ulp of the input: within 2^-5 of the row's peak).
    with true_fp32_matmul():
        ref = (x.float() @ deq.t()).to(torch.bfloat16)
    want = ref.clone()
    for r, s in enumerate(slots):
        if s == pad:
            continue
        window = torch.cat([before[s].float(), ref[r, :GDN_CONV].float()[:, None]], 1)
        c = (window * conv_weight.float()).sum(1)
        want[r, :GDN_CONV] = (c * torch.sigmoid(c)).to(torch.bfloat16)
    assert (out == want).float().mean().item() > 0.999
    peak = want.float().abs().amax(1, keepdim=True)
    assert ((out.float() - want.float()).abs() <= 2.0**-5 * peak).all()
    others = [s for s in range(pool) if s not in slots]
    assert torch.equal(conv_state[others], before[others]), "other slots untouched"


def _verify_args(spec, seed):
    """As the fp8 verify test: a column slice of a wider row, a strided pool (slot 0 zero), a
    pre-filled snapshot buffer."""
    k, conv_dim, p = spec["k"], spec["conv_dim"], spec["p"]
    slots = list(spec["slots"])
    m = len(slots) * p
    g = torch.Generator(device=DEV).manual_seed(seed)
    wide = torch.randn(m, k + 128, generator=g, device=DEV).to(torch.bfloat16)
    backing = torch.randn(spec["pool_slots"], conv_dim, 4, generator=g, device=DEV)
    conv_state = backing.to(torch.bfloat16)[:, :, 1:]
    conv_state[0] = 0
    conv_weight = torch.randn(conv_dim, 4, generator=g, device=DEV).to(torch.bfloat16)
    window = torch.randn(
        spec["snapshot_rows"], spec["snapshot_steps"], conv_dim, 3, generator=g, device=DEV
    ).to(torch.bfloat16)
    idx = torch.tensor(slots, dtype=torch.int32, device=DEV)
    rows = torch.arange(len(slots), dtype=torch.int32, device=DEV)
    return [wide[:, :k], conv_state, conv_weight, idx, window, rows, conv_dim, p, -1]


@pytest.mark.parametrize(
    "cell", split_verify_op.CELLS, ids=[c.why[:40] for c in split_verify_op.CELLS]
)
def test_split_conv1d_verify_cells(cell) -> None:
    spec = cell.spec
    _, codes, scales, gscale, _ = _weight(spec["n"], spec["k"], 3)
    packed = nv.pack_split(codes, scales, gscale)
    x, conv_state, conv_weight, idx, window, rows, conv_dim, p, pad = _verify_args(spec, 7)
    live = [r for r in range(x.shape[0]) if spec["slots"][r // p] != pad]
    split_verify_op.is_valid(
        x, packed, conv_state, conv_weight, idx, window, rows, conv_dim, p, pad
    )
    state_a, window_a = conv_state.clone(), window.clone()
    conv, rest = split_verify_op(
        x, packed, state_a, conv_weight, idx, window_a, rows, conv_dim, p, pad
    )
    # Bitwise the two-launch path over the same projection.
    proj = split_op(x, packed)
    n = x.shape[0] // p
    state_b, window_b = conv_state.clone(), window.clone()
    x3 = torch.transpose(torch.reshape(proj[:, :conv_dim], [n, p, conv_dim]), 1, 2)
    two = causal_conv1d_triton_update(
        x3, state_b, conv_weight, None, "silu", idx, pad, None, window_b, rows, False
    )
    two = torch.reshape(torch.transpose(two, 1, 2), [n * p, conv_dim])
    assert torch.equal(rest, proj[:, conv_dim:])
    assert torch.equal(conv[live], two[live])
    assert torch.equal(state_a, state_b) and torch.equal(window_a, window_b)
    # Against the fp32 reference.
    want, want_rest, want_state, want_window = split_verify_op.reference(
        x, codes, scales, gscale, conv_state, conv_weight, idx, window, rows, conv_dim, p, pad
    )
    deq = nv.dequantize_nvfp4(codes, scales, gscale)
    with true_fp32_matmul():
        ref = x.float() @ deq.t()
    _close(rest, ref[:, conv_dim:], x, deq[conv_dim:])
    assert (conv == want).float().mean().item() > 0.999
    peak = want.float().abs().amax(1, keepdim=True)
    assert ((conv.float() - want.float()).abs() <= 2.0**-5 * peak).all()
    for got, ref_ in ((state_a, want_state), (window_a, want_window)):
        assert (got == ref_).float().mean().item() > 0.9999
        assert (got.float() - ref_.float()).abs().max() <= 2.0**-7 * ref_.float().abs().max()


def test_split_graph_replay_matches_eager() -> None:
    _, codes, scales, gscale, _ = _weight(GDN_N, GDN_K, 9)
    packed = nv.pack_split(codes, scales, gscale)
    x = torch.randn(4, GDN_K, device=DEV).to(torch.bfloat16)
    split_op(x, packed)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = split_op(x, packed)
    x.copy_(torch.randn(4, GDN_K, device=DEV).to(torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, split_op(x, packed))


def test_split_precompile_covers_the_decode_variants() -> None:
    """After `precompile_split`, no projection / conv / verify call within its rows compiles."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.gemm import (  # noqa: E501
        fp8_gemv as f8,
    )

    _, codes, scales, gscale, _ = _weight(GDN_N, GDN_K, 2)
    packed = nv.pack_split(codes, scales, gscale)
    nv.precompile_split(packed, 4, GDN_CONV, -1, (2, 4))
    before = f8._compiled.cache_info().misses
    conv_state = torch.zeros(8, GDN_CONV, 3, device=DEV).to(torch.bfloat16)
    conv_weight = torch.randn(GDN_CONV, 4, device=DEV).to(torch.bfloat16)
    window = torch.zeros(4, 4, GDN_CONV, 3, device=DEV).to(torch.bfloat16)
    for m in range(1, 5):
        x = torch.randn(m, GDN_K, device=DEV).to(torch.bfloat16)
        idx = torch.arange(m, dtype=torch.int32, device=DEV)
        split_op(x, packed)
        split_conv_op(x, packed, conv_state, conv_weight, idx, GDN_CONV, -1)
        for p in (2, 4):
            if m % p == 0:
                n = m // p
                ids = torch.arange(n, dtype=torch.int32, device=DEV)
                split_verify_op(
                    x, packed, conv_state, conv_weight, ids, window, ids, GDN_CONV, p, -1
                )
    assert f8._compiled.cache_info().misses == before
