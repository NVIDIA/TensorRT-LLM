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
