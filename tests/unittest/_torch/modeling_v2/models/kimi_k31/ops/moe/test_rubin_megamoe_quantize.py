# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rubin_megamoe_quantize entry: every cell writes fp4_quantize's bytes into a MegaMoE bucket
and empties the bucket's routing past the rows, on SM107, a programmatic launch (``pdl``) included;
the guard on the CPU."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_megamoe_quantize import (  # noqa: E501
    rubin_megamoe_quantize,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 -- path above; fixture by name

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = pytest.mark.skipif(
    torch.cuda.is_available() and not _SM107, reason="certified on SM107 only"
)
gpu = pytest.mark.skipif(not _SM107, reason="needs an SM107 GPU")

HIDDEN, TOP_K = 3584, 16
# What the bucket holds before a call: bytes no quantization writes, and routing that still routes.
STALE_ACT, STALE_SF, STALE_ID, STALE_WEIGHT = 0xAB, 0xCD, 7, 3.0


def _x(rows, seed, mode="normal", device="cpu"):
    """Latent-like BF16 rows; ``edges``: all-zero blocks, tiny blocks (their scale underflows to
    0) and large ones (their scale saturates at 448) in every row."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=g) * 0.15
    if mode == "edges":
        blocks = x.view(rows, HIDDEN // 16, 16)
        blocks[:, 0::7] = 0.0
        blocks[:, 1::7] *= 1e-30
        blocks[:, 2::7] *= 1e4
    return x.to(torch.bfloat16).to(device)


def _bucket(t, device):
    return (
        torch.full((t, HIDDEN // 2), STALE_ACT, dtype=torch.uint8, device=device),
        torch.full((t, HIDDEN // 16), STALE_SF, dtype=torch.uint8, device=device),
        torch.full((t, TOP_K), STALE_ID, dtype=torch.int32, device=device),
        torch.full((t, TOP_K), STALE_WEIGHT, device=device),
    )


def _unpack(packed):
    """[M, K/2] packed bytes -> [M, K] codes (element 2i in the low nibble)."""
    m, half = packed.shape
    codes = torch.empty(m, half * 2, dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    return codes


def _today(x, gs):
    """fp4_quantize as dep16's staging path calls it: values and linear scales [n, hidden / 16]."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
        fp4_quantize,
    )

    x4, sf = fp4_quantize(x, gs, 16, False, False)
    return x4.view(torch.uint8), sf.view(torch.uint8).view(x.shape[0], HIDDEN // 16)


def _check_bucket(n, x, gs, act, sf, ids, weights):
    want_x4, want_sf = _today(x, gs)
    assert torch.equal(act[:n], want_x4), "values are not fp4_quantize's bits"
    assert torch.equal(sf[:n], want_sf), "block scales are not fp4_quantize's bits"
    assert bool((act[n:] == STALE_ACT).all()) and bool((sf[n:] == STALE_SF).all()), (
        "rows past the input were written"
    )
    assert bool((ids[:n] == STALE_ID).all()) and bool((weights[:n] == STALE_WEIGHT).all()), (
        "the routing of the rows was written"
    )
    assert bool((ids[n:] == -1).all()) and not weights[n:].any(), "the routing tail is not empty"


@gpu
@pytest.mark.parametrize(
    "cell", rubin_megamoe_quantize.CELLS, ids=[c.why[:48] for c in rubin_megamoe_quantize.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811 -- fixture resolved by name
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
        fp4_quantize,
    )

    certifying(rubin_megamoe_quantize)
    spec = cell.spec
    n, t = spec["rows"], spec["bucket"]
    x = _x(n, zlib.crc32(cell.why.encode()), spec.get("mode", "normal"), "cuda")
    gs = torch.tensor([spec.get("scale", 1.0)], device="cuda")
    act, sf, ids, weights = _bucket(t, "cuda")
    pdl = spec.get("pdl", False)
    rubin_megamoe_quantize(x, gs, act, sf, ids, weights, pdl=pdl)
    torch.cuda.synchronize()
    _check_bucket(n, x, gs, act, sf, ids, weights)
    # The catalog entry's own rule against its reference: scales exact, codes exact away from the
    # rounding ties its approximate reciprocals decide.
    ref_x4, ref_sf = rubin_megamoe_quantize.reference(x, gs, act, sf, ids, weights, pdl=pdl)
    assert torch.equal(sf[:n], ref_sf), "block scales differ from the reference"
    tie = fp4_quantize.near_tie(fp4_quantize.scaled(x, gs, 16))
    codes, ref_codes = _unpack(act[:n]), _unpack(ref_x4)
    assert torch.equal(codes[~tie], ref_codes[~tie]), "codes differ away from a rounding tie"


@gpu
def test_cuda_graph_replay_after_a_longer_call():
    """Captured at 96 rows into bucket 128, replayed after a 120-row call (and its routing) left
    rows of routing in the tail: the replay empties them again."""
    gs = torch.ones(1, device="cuda")
    x96, x120 = _x(96, 1, device="cuda"), _x(120, 2, device="cuda")
    act, sf, ids, weights = _bucket(128, "cuda")
    rubin_megamoe_quantize(x96, gs, act, sf, ids, weights)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rubin_megamoe_quantize(x96, gs, act, sf, ids, weights)
    rubin_megamoe_quantize(x120, gs, act, sf, ids, weights)
    ids[:120] = 5
    weights[:120] = 0.25
    graph.replay()
    torch.cuda.synchronize()
    want_x4, want_sf = _today(x96, gs)
    assert torch.equal(act[:96], want_x4) and torch.equal(sf[:96], want_sf)
    assert bool((ids[96:] == -1).all()) and not weights[96:].any()
    assert bool((ids[:96] == 5).all()), "the replay wrote the routing of its rows"


@gpu
@pytest.mark.parametrize("rows", [96, 48])
def test_a_programmatic_launch_behind_a_writer_reads_its_rows(rows):
    """Behind a kernel that writes the rows it quantizes (one stream, captured), a programmatic
    launch waits for that grid: on replays with new inputs the bucket holds what a launch in stream
    order leaves after the same writer."""
    gs = torch.full((1,), 96.0, device="cuda")
    src = _x(rows, 10 + rows, device="cuda")
    x = torch.empty_like(src)
    buckets = {pdl: _bucket(128, "cuda") for pdl in (False, True)}
    graphs = {}
    for pdl, (act, sf, ids, weights) in buckets.items():
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x.copy_(src)
            rubin_megamoe_quantize(x, gs, act, sf, ids, weights, pdl=pdl)
        torch.cuda.synchronize()
        graphs[pdl] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[pdl], stream=stream):
            torch.mul(src, 2.0, out=x)  # the writer: its grid ends right before the launch
            rubin_megamoe_quantize(x, gs, act, sf, ids, weights, pdl=pdl)
    for seed in (11, 12):
        src.copy_(_x(rows, seed, mode="edges" if seed % 2 else "normal", device="cuda"))
        for pdl in (False, True):
            graphs[pdl].replay()
        torch.cuda.synchronize()
        _check_bucket(rows, src * 2.0, gs, *buckets[True])
        for a, b in zip(buckets[True], buckets[False]):
            assert torch.equal(a, b), "the programmatic launch left other bytes"


@pytest.mark.cpu_only
def test_guard_admits_the_dep16_call():
    act, sf, ids, weights = _bucket(128, "cpu")
    rubin_megamoe_quantize.is_valid(_x(96, 3), torch.ones(1), act, sf, ids, weights)
    rubin_megamoe_quantize.is_valid(_x(128, 3), torch.ones(1), act, sf, ids, weights)
    rubin_megamoe_quantize.is_valid(_x(96, 3), torch.ones(1), act, sf, ids, weights, pdl=True)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "case",
    [
        "no_rows",
        "too_many_rows",
        "fp16_x",
        "strided_x",
        "sf_width",
        "ids_int64",
        "weights_t",
        "two_scales",
        "hidden_448",
        "pdl_int",
    ],  # fmt: skip
)
def test_guard_refuses_what_the_kernel_writes_wrong(case):
    x, gs = _x(96, 4), torch.ones(1)
    act, sf, ids, weights = _bucket(128, "cpu")
    if case == "no_rows":
        x = x[:0]
    elif case == "too_many_rows":
        x = _x(129, 4)
    elif case == "fp16_x":
        x = x.half()
    elif case == "strided_x":
        x = _x(96, 4).repeat(1, 2)[:, :HIDDEN]
    elif case == "sf_width":
        sf = torch.zeros(128, HIDDEN // 16 + 4, dtype=torch.uint8)
    elif case == "ids_int64":
        ids = ids.long()
    elif case == "weights_t":
        weights = torch.zeros(TOP_K, 128).t()
    elif case == "two_scales":
        gs = torch.ones(2)
    elif case == "hidden_448":
        x = torch.zeros(96, 448, dtype=torch.bfloat16)
        act = torch.zeros(128, 224, dtype=torch.uint8)
        sf = torch.zeros(128, 28, dtype=torch.uint8)
    pdl = 1 if case == "pdl_int" else False
    with pytest.raises(AssertionError):
        rubin_megamoe_quantize.is_valid(x, gs, act, sf, ids, weights, pdl=pdl)
