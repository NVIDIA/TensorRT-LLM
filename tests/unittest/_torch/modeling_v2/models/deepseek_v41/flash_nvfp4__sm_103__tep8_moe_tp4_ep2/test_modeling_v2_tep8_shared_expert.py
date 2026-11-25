# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's shared expert on stock's bytes (`weights.load_shared_expert` + `_SharedExpert`).

A synthetic shared expert in the checkpoint's own names (`<prefix>{w1,w2,w3}.{weight,scale}`: E4M3 bytes, 32x32 UE8M0
block scales, bytes as in the checkpoint file (`float8_e8m0fnu`) and varying within each 128x128 tile; the real width
2304 at tp 8, so each rank's 288-wide slice is padded to 384) is loaded for every rank.
Proved: every rank's weights and per-row scales are bit for bit what stock's loader and decode path build
(`_requantize_dense_stem`, the 128x128 requantization of `_V41_REQUANT_STEM_TAILS`, then the TP2 shard and the piece of
it this rank serves, `full_tp_shared_expert_weights`); the padding is zero weight with unit scales; and the sum of the
eight ranks' partial outputs is the whole (requantized) shared expert: exact-weight fp32 products up to bf16 rounding of
the intermediate up to 64 rows (stock's piece path, eager and graph alike: the low-M kernel), each rank's piece bit for
bit stock's `full_tp_shared_expert_forward`; above 64 rows each rank holds stock's `GatedMLP` shard (the TP2 shard with
its 128x128 FP32 tile multipliers, bit for bit) and runs stock's DeepGEMM path on it (`shared_expert_fp8_deep_gemm`)
scaled by stock's `shared_output_scale` (2 / 8), within the band of two FP8 1x128 activation roundings (x before w13,
the intermediate before w2: 5%); also under a CUDA graph with the side stream.
"""

import contextlib
import importlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # _validating
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.models.modeling_deepseekv4 import full_tp_shared_expert_weights
from tensorrt_llm._torch.models.modeling_deepseekv41 import _requantize_dense_stem

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)

_TARGET = (
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__tep8_moe_tp4_ep2"
)
modeling = importlib.import_module(f"{_TARGET}.modeling")
wts = importlib.import_module(f"{_TARGET}.weights")
DEV = "cuda"
H, INTER, TP, LIMIT = 5120, 2304, 8, 10.0
PREFIX = "layers.3.ffn.shared_experts."


def _checkpoint(gen):
    def fp8(r, c):
        return (torch.randn(r, c, generator=gen, device=DEV) * 2).to(torch.float8_e4m3fn)

    def scales(r, c):  # UE8M0 bytes around 2**-6
        return (
            (torch.randint(118, 124, (r // 32, c // 32), generator=gen, device=DEV))
            .to(torch.uint8)
            .view(torch.float8_e8m0fnu)
        )

    return {
        PREFIX + "w1.weight": fp8(INTER, H), PREFIX + "w1.scale": scales(INTER, H),
        PREFIX + "w3.weight": fp8(INTER, H), PREFIX + "w3.scale": scales(INTER, H),
        PREFIX + "w2.weight": fp8(H, INTER), PREFIX + "w2.scale": scales(H, INTER),
    }  # fmt: skip


def _stock(ck, w):
    """Stock's load of one projection: the 128x128 E4M3 weight and FP32 tile multipliers."""
    q, s = _requantize_dense_stem(ck[PREFIX + f"{w}.weight"], ck[PREFIX + f"{w}.scale"])
    return q.to(DEV), s.to(DEV)


def _dequant(ck, w):
    q, s = _stock(ck, w)
    return (
        q.float()
        * s.repeat_interleave(128, 0).repeat_interleave(128, 1)[: q.shape[0], : q.shape[1]]
    )


def _reference(ck, x):
    a = x.float()
    g, u = a @ _dequant(ck, "w1").t(), a @ _dequant(ck, "w3").t()
    h = (
        (torch.nn.functional.silu(g.clamp(max=LIMIT)) * u.clamp(-LIMIT, LIMIT))
        .to(torch.bfloat16)
        .float()
    )
    return h @ _dequant(ck, "w2").t()


@pytest.fixture(scope="module")
def experts():
    gen = torch.Generator(device=DEV).manual_seed(7)
    ck = _checkpoint(gen)
    out = []
    for rank in range(TP):
        se = modeling._SharedExpert(H, INTER, LIMIT, TP).to(DEV)
        wts.load_shared_expert(se, ck, PREFIX, DEV, TP, rank)
        se.post_load()
        out.append(se)
    return ck, out


def test_bytes_match_stock(experts):
    """Each rank holds the bytes stock's decode path builds for it: the stock GatedMLP's TP-`gcd(18, 8)` = TP2 shard of
    the requantized projections (`tp_rank % 2`), split into the `tp_rank // 2`-th of 4 pieces."""
    ck, ses = experts
    (q1, s1), (q3, s3), (q2, s2) = (_stock(ck, w) for w in ("w1", "w3", "w2"))
    shards = math.gcd(INTER // 128, TP)
    width = INTER // shards
    for rank, se in enumerate(ses):
        shard, piece = rank % shards, rank // shards
        rows, tiles = (
            slice(shard * width, (shard + 1) * width),
            slice(shard * width // 128, (shard + 1) * width // 128),
        )
        stock_mlp = SimpleNamespace(
            gate_up_proj=SimpleNamespace(weight=torch.cat([q1[rows], q3[rows]]),
                                         weight_scale_block128=torch.cat([s1[tiles], s3[tiles]])),
            down_proj=SimpleNamespace(weight=q2[:, rows].contiguous(), weight_scale_block128=s2[:, tiles].contiguous()),
        )  # fmt: skip
        w13, sc13, w2, sc2 = full_tp_shared_expert_weights(stock_mlp, piece, TP // shards)
        part, pp, sp = (
            se.part,
            se.part_p,
            w13.shape[0] // 2,
        )  # stock pads each half to 320, this target to 384
        for half in range(2):
            ours, theirs = slice(half * pp, half * pp + part), slice(half * sp, half * sp + part)
            assert torch.equal(se.w13.view(torch.uint8)[ours], w13.view(torch.uint8)[theirs]), (
                rank,
                half,
            )
            assert torch.equal(se.w13_s[ours], sc13[theirs]), (rank, half)
        assert torch.equal(se.w2.view(torch.uint8)[:, :part], w2.view(torch.uint8)[:, :part]), rank
        assert torch.equal(se.w2_s[:, : part // 32], sc2[:, : part // 32]), rank


def test_gated_mlp_shard_matches_stock(experts):
    """Past the piece path each rank runs stock's `GatedMLP` shard: TP `gcd(18, 8)` = 2, shard `tp_rank % 2` of the
    requantized projections (gate rows then up rows; w2 by columns) with their FP32 128x128 tile multipliers."""
    ck, ses = experts
    (q1, s1), (q3, s3), (q2, s2) = (_stock(ck, w) for w in ("w1", "w3", "w2"))
    shards = math.gcd(INTER // 128, TP)
    width = INTER // shards
    for rank, se in enumerate(ses):
        shard = rank % shards
        rows, tiles = (
            slice(shard * width, (shard + 1) * width),
            slice(shard * width // 128, (shard + 1) * width // 128),
        )
        assert (se.shards, se.shard_scale) == (shards, shards / TP)
        assert torch.equal(
            se.g13.view(torch.uint8), torch.cat([q1[rows], q3[rows]]).view(torch.uint8)
        ), rank
        assert torch.equal(se.g13_s, torch.cat([s1[tiles], s3[tiles]])), rank
        assert torch.equal(se.g2.view(torch.uint8), q2[:, rows].contiguous().view(torch.uint8)), (
            rank
        )
        assert torch.equal(se.g2_s, s2[:, tiles].contiguous()), rank


@pytest.mark.parametrize("m", [65, 300])
def test_wide_steps_run_stocks_gated_mlp(experts, m):
    """Past 64 rows a rank's output is stock's `GatedMLP` forward on its shard (the family's
    `shared_expert_fp8_deep_gemm`, stock's SM100 DeepGEMM path) times stock's `shared_output_scale`, bit for bit."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.shared_expert_fp8 import (
        deep_gemm_weight_scale,
        shared_expert_fp8_deep_gemm,
    )

    _, ses = experts
    x = torch.randn(m, H, generator=torch.Generator(device=DEV).manual_seed(m), device=DEV).to(
        torch.bfloat16
    )
    for rank, se in enumerate(ses[:2]):
        sf13, sf2 = (
            deep_gemm_weight_scale(se.g13_s, *se.g13.shape),
            deep_gemm_weight_scale(se.g2_s, *se.g2.shape),
        )
        want = (
            shared_expert_fp8_deep_gemm(x, se.g13, sf13, se.g13_s, se.g2, sf2, se.g2_s, LIMIT)
            * se.shard_scale
        )
        assert torch.equal(se(x), want), rank


def test_boundary_quantized_steps_run_stocks_gated_mlp(experts):
    """From 1024 rows stock's `GatedMLP` reads the FFN boundary's FP8 1x128 copy of its input (`nvfp4_fp8_quantize`);
    with that copy a rank's output is `shared_expert_fp8_deep_gemm_prequant` on its shard times the shard scale, bit for
    bit, and the ranks still sum to the shared expert within the band."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.nvfp4_fp8_quantize import (
        nvfp4_fp8_quantize,
    )
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.shared_expert_fp8 import (
        deep_gemm_weight_scale,
        shared_expert_fp8_deep_gemm_prequant,
    )

    ck, ses = experts
    m = 1100
    x = torch.randn(m, H, generator=torch.Generator(device=DEV).manual_seed(m), device=DEV).to(
        torch.bfloat16
    )
    fp8 = nvfp4_fp8_quantize(x, torch.ones(1, device=DEV))[2:]
    got = []
    for rank, se in enumerate(ses):
        out = se(x, fp8)
        if rank < 2:
            sf13, sf2 = (
                deep_gemm_weight_scale(se.g13_s, *se.g13.shape),
                deep_gemm_weight_scale(se.g2_s, *se.g2.shape),
            )
            want = (
                shared_expert_fp8_deep_gemm_prequant(*fp8, se.g13, sf13, se.g2, sf2, LIMIT)
                * se.shard_scale
            )
            assert torch.equal(out, want), rank
        got.append(out.float())
    ref = _reference(ck, x)
    assert float((sum(got) - ref).norm() / ref.norm()) < 5e-2


def test_padding(experts):
    _, ses = experts
    se = ses[3]
    assert (se.part, se.part_p) == (288, 384)
    pad = torch.cat([se.w13[288:384], se.w13[384 + 288 :]]).view(torch.uint8)
    assert pad.numel() and not pad.any()
    assert not se.w2[:, 288:].view(torch.uint8).any() and bool((se.w2_s[:, 9:] == 127).all())


@pytest.mark.parametrize("graph", [True, False])
@pytest.mark.parametrize("m", [1, 6, 16, 24, 64, 300])
def test_ranks_sum_to_the_shared_expert(experts, m, graph):
    """`graph`: a decode-graph step (under `flashinfer_mxfp8_decode_graph_capture`); up to 64 rows take stock's piece
    path (the low-M kernel) either way, larger steps stock's GatedMLP shard."""
    from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_decode_graph_capture

    ck, ses = experts
    gen = torch.Generator(device=DEV).manual_seed(m)
    x = torch.randn(m, H, generator=gen, device=DEV).to(torch.bfloat16)
    with flashinfer_mxfp8_decode_graph_capture() if graph else contextlib.nullcontext():
        got = sum(se(x).float() for se in ses)
    ref = _reference(ck, x)
    err = (got - ref).norm() / ref.norm()
    # low-M (<= 64 rows): exact FP8 weights x bf16 activations (bf16 intermediate and output roundings); otherwise
    # stock's GatedMLP: two FP8 1x128 activation roundings
    assert err < (1e-2 if m <= 64 else 5e-2), (m, graph, float(err))


@pytest.mark.parametrize("m", [6, 14, 64])
def test_piece_is_stocks_forward(experts, m):
    """Up to 64 rows each rank's partial output is bit for bit stock's `full_tp_shared_expert_forward` on the piece
    stock builds for that rank (its 320-wide padding vs this target's 384 adds only zero products)."""
    from tensorrt_llm._torch.models.modeling_deepseekv4 import full_tp_shared_expert_forward

    ck, ses = experts
    (q1, s1), (q3, s3), (q2, s2) = (_stock(ck, w) for w in ("w1", "w3", "w2"))
    shards = math.gcd(INTER // 128, TP)
    width = INTER // shards
    x = torch.randn(m, H, generator=torch.Generator(device=DEV).manual_seed(m), device=DEV).to(
        torch.bfloat16
    )
    for rank, se in enumerate(ses):
        shard, piece = rank % shards, rank // shards
        rows, tiles = (
            slice(shard * width, (shard + 1) * width),
            slice(shard * width // 128, (shard + 1) * width // 128),
        )
        stock_mlp = SimpleNamespace(
            gate_up_proj=SimpleNamespace(weight=torch.cat([q1[rows], q3[rows]]),
                                         weight_scale_block128=torch.cat([s1[tiles], s3[tiles]])),
            down_proj=SimpleNamespace(weight=q2[:, rows].contiguous(), weight_scale_block128=s2[:, tiles].contiguous()),
        )  # fmt: skip
        theirs = full_tp_shared_expert_forward(
            x, full_tp_shared_expert_weights(stock_mlp, piece, TP // shards), LIMIT
        )
        assert torch.equal(se(x), theirs), rank


def test_side_stream_under_cuda_graph(experts):
    """`_Layer._shared_overlapped`'s pattern: fork to the side stream, wait on its event, inside a graph."""
    _, ses = experts
    se = ses[0]
    gen = torch.Generator(device=DEV).manual_seed(11)
    x = torch.randn(6, H, generator=gen, device=DEV).to(torch.bfloat16)
    eager = se(x).clone()
    side = torch.cuda.Stream()
    fork, done = torch.cuda.Event(), torch.cuda.Event()
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(g, stream=stream):
            fork.record()
            with torch.cuda.stream(side):
                fork.wait()
                out = se(x)
                done.record()
            torch.cuda.current_stream().wait_event(done)
            res = out * 1
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(res, eager, rtol=0, atol=0)


_lowm = importlib.import_module(f"{_TARGET}.ops.moe.shared_expert_lowm")


@pytest.mark.parametrize(
    "cell", _lowm.shared_expert_lowm.CELLS, ids=[c.why[:44] for c in _lowm.shared_expert_lowm.CELLS]
)
def test_shared_expert_lowm_cells(cell, certifying):  # noqa: F811
    certifying(_lowm.shared_expert_lowm)
    m, p = cell.spec["m"], cell.spec["piece"]
    gen = torch.Generator(device=DEV).manual_seed(m + p)
    w13 = (torch.randn(2 * p, H, generator=gen, device=DEV) * 0.5).to(torch.float8_e4m3fn)
    s13 = torch.randint(
        118, 122, (2 * p, H // 32), generator=gen, device=DEV, dtype=torch.int32
    ).to(torch.uint8)
    w2 = (torch.randn(H, p, generator=gen, device=DEV) * 0.5).to(torch.float8_e4m3fn)
    s2 = torch.randint(118, 122, (H, p // 32), generator=gen, device=DEV, dtype=torch.int32).to(
        torch.uint8
    )
    x = torch.randn(m, H, generator=gen, device=DEV).to(torch.bfloat16)
    args = (x, w13, s13, w2, s2, LIMIT)
    with validating(_lowm.shared_expert_lowm):
        got = _lowm.shared_expert_lowm(*args)
    ref = _lowm.shared_expert_lowm.reference(*args)
    err = (got.float() - ref.float()).norm() / ref.float().norm()
    assert got.shape == (m, H) and got.dtype == torch.bfloat16 and err < 1e-2, float(err)


@pytest.mark.parametrize("m", [1, 6, 16, 64])
def test_shared_expert_lowm_fused_is_three_launches(experts, m):
    """On the checkpoint's piece (a bf16 limit, so gate | up stores the clamped SwiGLU itself): bit for bit the three
    launches (gate_up GEMM, the engine's SwiGLU, down GEMM), and the target's module takes it up to 64 rows."""
    from tensorrt_llm._torch.modules.low_m_gemm import mxfp8_lowm_linear, swiglu_out_supported
    from tensorrt_llm._torch.modules.swiglu import swiglu

    _, ses = experts
    se = ses[5]
    gen = torch.Generator(device=DEV).manual_seed(m)
    x = torch.randn(m, H, generator=gen, device=DEV).to(torch.bfloat16)
    assert swiglu_out_supported(m, H, se.w13, LIMIT)
    args = (x, se.w13, se.w13_s, se.w2, se.w2_s, LIMIT)
    got = _lowm.shared_expert_lowm(*args)
    three = mxfp8_lowm_linear(
        swiglu(mxfp8_lowm_linear(x, se.w13, se.w13_s), swiglu_limit=LIMIT), se.w2, se.w2_s
    )
    assert torch.equal(got, three)
    assert torch.equal(se(x), got)
