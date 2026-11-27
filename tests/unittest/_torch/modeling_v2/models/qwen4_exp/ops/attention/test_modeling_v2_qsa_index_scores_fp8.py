# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: `qsa_index_scores_fp8` (e4m3 index anchors) over its cells against the float64
reference on the e4m3 operands; `index_anchors_to_fp8` against torch's e4m3 rounding; the
ctx-tile kernel bitwise the request-group kernel; graph replay against eager."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention import (
    qsa_index_scores_fp8 as _module,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_index_scores_fp8 import (
    index_anchors_to_fp8,
    qsa_index_scores_fp8,
)

assert torch.cuda.is_available(), "qsa_index_scores_fp8 requires a CUDA device"

DEV = "cuda"
TPB, RATIO, DIM = 32, 4, 128


def build(spec, seed):
    """A bf16 index plane (each row's slots of a scratch-sliced `[pages, 32, 1, 128]` view), its
    anchors converted to e4m3 by the op under test, and the call's arguments."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    requests = spec["requests"]
    table = spec["table"]
    pages_each = [(s + n) // TPB + 1 for s, n in requests]
    pool_pages = sum(pages_each) + 3
    raw = torch.randn(pool_pages, TPB, 2, DIM, generator=gen, device=DEV, dtype=torch.bfloat16)
    plane = raw[:, :, :1]  # a strided view, as the pool's planes are
    block_table = torch.full((len(requests) + 2, table), -1, dtype=torch.int32, device=DEV)
    ids = torch.randperm(pool_pages, generator=gen, device=DEV).to(torch.int32)
    cursor = 0
    for r, n in enumerate(pages_each):
        block_table[r, :n] = ids[cursor : cursor + n]
        cursor += n
    positions = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in requests])
    req = torch.cat([torch.full((n,), r, device=DEV) for r, (_, n) in enumerate(requests)])
    req = req.to(torch.int32)
    # every anchor of every request, as the writers' compressions would have named them
    every = torch.cat([torch.arange(RATIO - 1, s + n, RATIO, device=DEV) for s, n in requests])
    owners = torch.cat(
        [torch.full(((s + n) // RATIO,), r, device=DEV) for r, (s, n) in enumerate(requests)]
    ).to(torch.int32)
    bf16_keys = plane.clone()
    index_anchors_to_fp8(plane, block_table, owners, every.long(), TPB, RATIO)
    scratch = torch.randn(positions.numel(), 6, DIM, generator=gen, device=DEV).to(torch.bfloat16)
    args = dict(
        q=scratch[:, : spec["heads"]],
        index_cache=plane,
        block_table=block_table,
        query_positions=positions.long(),
        request_indices=req,
        tokens_per_block=TPB,
        compress_ratio=RATIO,
        only_visible_blocks=spec["only_visible_blocks"],
        context_rows=spec["context_rows"],
        rows_per_request=spec["rows_per_request"],
    )
    if spec.get("early"):
        args["early"] = True
    return args, bf16_keys, (every, owners)


def _visible_close(out, ref, positions):
    for r in range(out.shape[0]):
        v = (int(positions[r]) + 1) // RATIO
        torch.testing.assert_close(out[r, :v].double(), ref[r, :v], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "cell", qsa_index_scores_fp8.CELLS, ids=[c.why[:40] for c in qsa_index_scores_fp8.CELLS]
)
def test_certified_cells(cell):
    args, _, _ = build(cell.spec, seed=len(cell.why))
    qsa_index_scores_fp8.is_valid(**args)
    out = qsa_index_scores_fp8(**args)
    ref = qsa_index_scores_fp8.reference(**args)
    _visible_close(out, ref, args["query_positions"])
    if not args["only_visible_blocks"]:
        assert torch.equal(torch.isinf(out), torch.isinf(ref))


def test_anchor_convert_is_torch_e4m3_rounding():
    spec = _module._scores_spec(requests=((0, 300), (1000, 9)))
    args, bf16_keys, (every, owners) = build(spec, seed=3)
    plane8 = args["index_cache"].view(torch.float8_e4m3fn)
    table = args["block_table"]
    pages = table[owners.long(), every // TPB].long()
    slots = every % TPB
    want = bf16_keys[pages, slots, 0].to(torch.float8_e4m3fn)
    assert torch.equal(plane8[pages, slots, 0, :DIM].view(torch.uint8), want.view(torch.uint8))
    # raw (non-anchor) slots stay bf16
    raw_slot = (every[:1] - 1) % TPB
    assert torch.equal(
        args["index_cache"][pages[:1], raw_slot, 0], bf16_keys[pages[:1], raw_slot, 0]
    )


def test_ctx_tile_is_bitwise_the_group_kernel():
    spec = _module._scores_spec(requests=((5000, 300), (17, 200)), context_rows=True)
    args, _, _ = build(spec, seed=11)
    tiled = qsa_index_scores_fp8(**args)
    assert _module._takes_ctx_tile(args["q"], True, True, 1)
    group = _module._qsa_index_scores_fp8(
        args["q"], args["index_cache"], args["block_table"], args["query_positions"],
        args["request_indices"], TPB, RATIO, True, True, 1,
    )  # fmt: skip
    for r in range(tiled.shape[0]):
        v = (int(args["query_positions"][r]) + 1) // RATIO
        assert torch.equal(tiled[r, :v], group[r, :v])


def test_replay_matches_eager():
    spec = _module._scores_spec(
        requests=tuple((900 + 311 * r, 4) for r in range(8)), rows_per_request=4
    )
    args, _, _ = build(spec, seed=5)
    qsa_index_scores_fp8(**args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qsa_index_scores_fp8(**args)
    fresh = torch.randn_like(args["q"].float()).to(torch.bfloat16)
    args["q"].copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    eager = qsa_index_scores_fp8(**args)
    _visible_close(captured, eager.double(), args["query_positions"])


@pytest.mark.parametrize(
    "requests, rpr",
    [
        (tuple((900 + 7919 * r, 4) for r in range(12)), 4),  # dep verify, 12 runs of 4
        (tuple((2000 + 4800 * r, 1) for r in range(52)), 1),  # draft rows, 52 runs of 1
        (tuple((300 + 997 * r, 4) for r in range(64)), 4),  # 256 rows, the most it serves
    ],
    ids=["verify12", "draft52", "verify64"],
)
def test_flat_is_bitwise_the_group_kernel(requests, rpr):
    spec = _module._scores_spec(requests=requests, rows_per_request=rpr, table=8196)
    args, _, _ = build(spec, seed=13)
    q = args["q"]
    assert (
        _module._flat_refusal(
            q,
            args["index_cache"],
            args["block_table"],
            args["query_positions"],
            args["request_indices"],
            TPB,
            RATIO,
            rpr,
        )  # fmt: skip
        is None
    )
    flat = qsa_index_scores_fp8(**args)
    group = _module._qsa_index_scores_fp8(
        q, args["index_cache"], args["block_table"], args["query_positions"],
        args["request_indices"], TPB, RATIO, True, False, rpr,
    )  # fmt: skip
    for r in range(flat.shape[0]):
        v = (int(args["query_positions"][r]) + 1) // RATIO
        assert torch.equal(flat[r, :v], group[r, :v])


@pytest.mark.parametrize(
    "requests, rpr",
    [
        (tuple((900 + 7919 * r, 4) for r in range(12)), 4),  # dep verify, 12 runs of 4
        (tuple((300 + 997 * r, 4) for r in range(64)), 4),  # 256 rows, the most it serves
        (tuple((5, 4) for _ in range(3)), 4),  # fewer tiles than programs: early exits wait too
    ],
    ids=["verify12", "verify64", "short"],
)
def test_early_is_bitwise_the_default_flat_launch(requests, rpr):
    """The early launch behind a PDL anchor convert (the trunk verify's chain): the convert
    writes the anchors the scores then read, so a read before the wait would show as other
    bits; eager and under graph replay with fresh anchors."""
    spec = _module._scores_spec(requests=requests, rows_per_request=rpr, table=8196)
    args, bf16_keys, (every, owners) = build(spec, seed=17)
    want = qsa_index_scores_fp8(**args)
    plane = args["index_cache"]

    def chain():
        plane.copy_(bf16_keys)
        index_anchors_to_fp8(plane, args["block_table"], owners, every.long(), TPB, RATIO, True)
        return qsa_index_scores_fp8(**args, early=True)

    got = chain()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = chain()
    graph.replay()
    torch.cuda.synchronize()
    for out in (got, captured):
        for r in range(out.shape[0]):
            v = (int(args["query_positions"][r]) + 1) // RATIO
            assert torch.equal(out[r, :v], want[r, :v])


@pytest.mark.parametrize(
    "requests, rpr",
    [
        (tuple((900 + 7919 * r, 4) for r in range(12)), 4),  # dep verify, 12 runs of 4
        (tuple((4800 + 4100 * r, 4) for r in range(40)), 4),  # dep4 40 a rank: 160 rows
        (tuple((5, 4) for _ in range(3)), 4),  # fewer tiles than programs
    ],
    ids=["verify12", "verify40", "short"],
)
def test_early_without_prefetch_is_bitwise(monkeypatch, requests, rpr):
    """`TRTLLM_QWEN4_EXP_INDEX_EARLY_PF_ROWS`: the early launch without its L2 prefetch gives the
    default launch's scores bit for bit (the prefetch is a hint), behind a PDL anchor convert,
    eager and under graph replay."""
    spec = _module._scores_spec(requests=requests, rows_per_request=rpr, table=8196)
    args, bf16_keys, (every, owners) = build(spec, seed=31)
    want = qsa_index_scores_fp8(**args)
    plane = args["index_cache"]
    monkeypatch.setattr(_module, "_EARLY_PF_ROWS", 1)

    def chain():
        plane.copy_(bf16_keys)
        index_anchors_to_fp8(plane, args["block_table"], owners, every.long(), TPB, RATIO, True)
        return qsa_index_scores_fp8(**args, early=True)

    got = chain()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = chain()
    graph.replay()
    torch.cuda.synchronize()
    for out in (got, captured):
        for r in range(out.shape[0]):
            v = (int(args["query_positions"][r]) + 1) // RATIO
            assert torch.equal(out[r, :v], want[r, :v]), f"row {r}"


def test_early_pf_rows_env(monkeypatch):
    """The switch's parse: unset or 0 is off, a negative or non-integer value raises."""
    monkeypatch.delenv(_module.EARLY_PF_ROWS_ENV, raising=False)
    assert _module._early_pf_rows() == 0
    monkeypatch.setenv(_module.EARLY_PF_ROWS_ENV, "128")
    assert _module._early_pf_rows() == 128
    for bad in ("-1", "x", "1.5"):
        monkeypatch.setenv(_module.EARLY_PF_ROWS_ENV, bad)
        with pytest.raises(ValueError):
            _module._early_pf_rows()


@pytest.mark.parametrize(
    "cutoff, early, want",
    [
        (0, True, None),  # unset: the default launch
        (128, True, 0),
        (159, True, 0),
        (160, True, None),  # not more than the cutoff
        (128, False, 0),  # not early: never a prefetch
    ],
    ids=["off", "past128", "past159", "at160", "not_early"],
)
def test_early_pf_rows_selects_the_launch(monkeypatch, cutoff, early, want):
    """`TRTLLM_QWEN4_EXP_INDEX_EARLY_PF_ROWS`: the flat launch of a 160-row call prefetches
    `_EARLY_PF_TILES` tiles unless the call is early and has more rows than the cutoff (the
    bitwise test above cannot see which launch ran: the prefetch is only a hint)."""
    want = _module._EARLY_PF_TILES if want is None else want
    spec = _module._scores_spec(
        requests=tuple((4800 + 4100 * r, 4) for r in range(40)), rows_per_request=4, table=8196
    )
    args, _, _ = build(spec, seed=37)
    real = _module._qsa_index_scores_flat_kernel
    seen = []

    class Spy:
        def __getitem__(self, grid):
            def launch(*a, **kw):
                seen.append(kw["PF_TILES"])
                return real[grid](*a, **kw)

            return launch

    monkeypatch.setattr(_module, "_qsa_index_scores_flat_kernel", Spy())
    monkeypatch.setattr(_module, "_EARLY_PF_ROWS", cutoff)
    qsa_index_scores_fp8(**args, early=early)
    torch.cuda.synchronize()
    assert seen == [want]
