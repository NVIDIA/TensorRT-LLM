# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: `fp8_gemv` / `fp8_gemv_conv1d` / `fp8_gemv_conv1d_verify` against an fp32
reference over the dequantized weight (128x128-block e4m3), per cell; the verify entry also
bitwise against the two-launch path it replaces (`fp8_gemv` + `causal_conv1d_triton_update`);
strided activations; CUDA-graph replay."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.gemm.fp8_gemv import (  # noqa: E501
    dequantize_fp8_block,
    fp8_gemv_conv1d_op,
    fp8_gemv_conv1d_verify_op,
    fp8_gemv_op,
    pack_weight,
    quantize_fp8_block,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.causal_conv1d_triton_update import (  # noqa: E501
    causal_conv1d_triton_update,
)

assert torch.cuda.is_available(), "fp8_gemv requires a CUDA device"

DEV = "cuda"


def _weight(n, k, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=DEV) * k**-0.5
    w[:, 7] *= 30.0  # outlier column
    w[: min(n, 128), 128:256] *= 1e-4  # a tiny block: subnormal codes
    codes, scale = quantize_fp8_block(w.to(torch.bfloat16))
    return codes, scale, pack_weight(codes, scale)


def _close(y, ref, x, deq) -> None:
    """Within one bf16 rounding of the fp32 reference (plus fp32 summation noise)."""
    with true_fp32_matmul():
        absref = x.float().abs() @ deq.abs().t()
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    excess = ((y.float() - ref).abs() - 0.5 * ulp).clamp_min(0) / absref.clamp_min(1e-30)
    assert excess.max().item() < 1e-5, f"error beyond one rounding: {excess.max().item():.2e}"


@pytest.mark.parametrize("cell", fp8_gemv_op.CELLS, ids=[c.why[:40] for c in fp8_gemv_op.CELLS])
def test_gemv_cells(cell) -> None:
    n, k, m = cell.spec["n"], cell.spec["k"], cell.spec["m"]
    codes, scale, packed = _weight(n, k, n + m)
    g = torch.Generator(device=DEV).manual_seed(m)
    # A column slice of a wider row, as the target's split projections hand over.
    wide = torch.randn(m, k + 256, generator=g, device=DEV).to(torch.bfloat16)
    x = wide[:, :k]
    y = fp8_gemv_op(x, packed)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    _close(y, fp8_gemv_op.reference(x, codes, scale), x, dequantize_fp8_block(codes, scale))


@pytest.mark.parametrize(
    "cell", fp8_gemv_conv1d_op.CELLS, ids=[c.why[:40] for c in fp8_gemv_conv1d_op.CELLS]
)
def test_conv1d_cells(cell) -> None:
    n, k, conv_dim, pool, pad = 16480, 2560, 10240, 64, -1
    codes, scale, packed = _weight(n, k, 3)
    deq = dequantize_fp8_block(codes, scale)
    slots = list(cell.spec["slots"])
    idx = torch.tensor(slots, dtype=torch.int32, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(11)
    backing = torch.randn(pool, conv_dim, 4, generator=g, device=DEV).to(torch.bfloat16)
    conv_state = backing[:, :, 1:]  # a strided pool view
    conv_weight = torch.randn(conv_dim, 4, generator=g, device=DEV).to(torch.bfloat16)
    x = torch.randn(len(slots), k, generator=g, device=DEV).to(torch.bfloat16)
    before = conv_state.clone()
    with true_fp32_matmul():
        proj = (x.float() @ deq.t()).to(torch.bfloat16)
    want, want_state = proj.clone(), conv_state.clone()
    for r, s in enumerate(slots):
        if s == pad:
            continue
        window = torch.cat([conv_state[s].float(), proj[r, :conv_dim].float()[:, None]], 1)
        c = (window * conv_weight.float()).sum(1)
        want[r, :conv_dim] = (c * torch.sigmoid(c)).to(torch.bfloat16)
        want_state[s] = window[:, 1:].to(torch.bfloat16)
    out = fp8_gemv_conv1d_op(x, packed, conv_state, conv_weight, idx, conv_dim, pad)
    # The projection may round one ulp differently than the reference's fp32 matmul; a flip
    # carries through the conv. Nearly every output is bitwise the reference's.
    assert (out == want).float().mean().item() > 0.999
    assert ((out.float() - want.float()).abs() / want.float().abs().clamp_min(1e-2)).max() < 0.02
    assert ((conv_state.float() - want_state.float()).abs()).max().item() < 0.02
    others = [s for s in range(pool) if s not in slots]
    assert torch.equal(conv_state[others], before[others]), "other slots untouched"


def test_graph_replay_matches_eager() -> None:
    _, _, packed = _weight(2560, 6144, 5)
    x = torch.randn(4, 6144, device=DEV).to(torch.bfloat16)
    fp8_gemv_op(x, packed)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fp8_gemv_op(x, packed)
    x.copy_(torch.randn(4, 6144, device=DEV).to(torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, fp8_gemv_op(x, packed))


def _verify_args(spec, seed):
    """`fp8_gemv_conv1d_verify`'s arguments for a cell: the activation a column slice of a wider
    row, the pool a strided view (every 4th column skipped), slot 0's window zero (a fresh
    sequence), the snapshot buffer pre-filled so untouched steps and rows are visible."""
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


def _two_launch(x, packed, conv_state, conv_weight, idx, window, rows, conv_dim, p, pad):
    """The target's verify path before the fusion: the projection, then the Triton conv over
    the transposed q|k|v view; returns `(conv, rest)` in the fused entry's layout."""
    proj = fp8_gemv_op(x, packed)
    n = x.shape[0] // p
    x3 = torch.transpose(torch.reshape(proj[:, :conv_dim], [n, p, conv_dim]), 1, 2)
    out = causal_conv1d_triton_update(
        x3, conv_state, conv_weight, None, "silu", idx, pad, None, window, rows, False
    )
    return torch.reshape(torch.transpose(out, 1, 2), [n * p, conv_dim]), proj[:, conv_dim:]


@pytest.mark.parametrize(
    "cell",
    fp8_gemv_conv1d_verify_op.CELLS,
    ids=[c.why[:40] for c in fp8_gemv_conv1d_verify_op.CELLS],
)
def test_conv1d_verify_cells(cell) -> None:
    spec = cell.spec
    codes, scale, packed = _weight(spec["n"], spec["k"], 3)
    x, conv_state, conv_weight, idx, window, rows, conv_dim, p, pad = _verify_args(spec, 7)
    live = [r for r in range(x.shape[0]) if spec["slots"][r // p] != pad]

    state_a, window_a = conv_state.clone(), window.clone()
    conv, rest = fp8_gemv_conv1d_verify_op(
        x, packed, state_a, conv_weight, idx, window_a, rows, conv_dim, p, pad
    )
    assert conv.shape == (x.shape[0], conv_dim) and conv.is_contiguous()
    assert rest.shape == (x.shape[0], spec["n"] - conv_dim)

    # Bitwise the two-launch path (same projection plan, the Triton conv's arithmetic).
    state_b, window_b = conv_state.clone(), window.clone()
    conv_b, rest_b = _two_launch(
        x, packed, state_b, conv_weight, idx, window_b, rows, conv_dim, p, pad
    )
    assert torch.equal(rest.view(torch.int16), rest_b.view(torch.int16))
    assert torch.equal(conv[live].view(torch.int16), conv_b[live].view(torch.int16))
    assert torch.equal(state_a.view(torch.int16), state_b.view(torch.int16)), "whole pool"
    assert torch.equal(window_a.view(torch.int16), window_b.view(torch.int16)), "whole buffer"

    # Against the fp32 reference: the projection may round one ulp differently (fp32 summation
    # order), and a flip carries through the conv (one tap times one ulp of the input: within
    # 2^-5 of the row's peak), the snapshots and the pool (one ulp: within 2^-7 of the peak).
    want, want_rest, want_state, want_window = fp8_gemv_conv1d_verify_op.reference(
        x, codes, scale, conv_state, conv_weight, idx, window, rows, conv_dim, p, pad
    )
    deq = dequantize_fp8_block(codes, scale)
    with true_fp32_matmul():
        ref = x.float() @ deq.t()
    _close(rest, ref[:, conv_dim:], x, deq[conv_dim:])
    assert (conv == want).float().mean().item() > 0.999
    peak = want.float().abs().amax(1, keepdim=True)
    assert ((conv.float() - want.float()).abs() <= 2.0**-5 * peak).all()
    for got, ref_ in ((state_a, want_state), (window_a, want_window)):
        assert (got == ref_).float().mean().item() > 0.9999
        assert (got.float() - ref_.float()).abs().max() <= 2.0**-7 * ref_.float().abs().max()
    # Non-vacuity: every live slot and snapshot row was rewritten.
    for r, slot in enumerate(spec["slots"]):
        if slot != pad:
            assert not torch.equal(state_a[slot], conv_state[slot]), slot
            assert not torch.equal(window_a[r, :p], window[r, :p]), r
    if spec["snapshot_steps"] > p:
        assert torch.equal(window_a[:, p:], window[:, p:]), "steps past p untouched"


def test_conv1d_verify_graph_replay_matches_eager() -> None:
    spec = fp8_gemv_conv1d_verify_op.CELLS[2].spec
    _, _, packed = _weight(spec["n"], spec["k"], 5)
    x, conv_state, conv_weight, idx, window, rows, conv_dim, p, pad = _verify_args(spec, 9)
    state0, window0 = conv_state.clone(), window.clone()
    fp8_gemv_conv1d_verify_op(
        x, packed, conv_state, conv_weight, idx, window, rows, conv_dim, p, pad
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        conv, rest = fp8_gemv_conv1d_verify_op(
            x, packed, conv_state, conv_weight, idx, window, rows, conv_dim, p, pad
        )
    x.copy_(torch.randn(x.shape, device=DEV).to(torch.bfloat16))
    conv_state.copy_(state0)
    window.copy_(window0)
    graph.replay()
    torch.cuda.synchronize()
    got = (conv.clone(), rest.clone(), conv_state.clone(), window.clone())
    conv_state.copy_(state0)
    window.copy_(window0)
    want = fp8_gemv_conv1d_verify_op(
        x, packed, conv_state, conv_weight, idx, window, rows, conv_dim, p, pad
    )
    for a, b in zip(got, (*want, conv_state, window)):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
