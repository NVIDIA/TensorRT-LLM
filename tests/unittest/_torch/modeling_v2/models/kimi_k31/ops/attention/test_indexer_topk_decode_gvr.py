# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 indexer_topk_decode_gvr entry (deterministic self-sampling GVR).

Each row's selected set must equal the reference's exactly (ties at the K-th score to the smaller
column) and the CUDA radix's on tie-free rows; a second call must repeat the output, order included.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_decode import (  # noqa: E501
    indexer_topk_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_decode_gvr import (  # noqa: E501
    indexer_topk_decode_gvr as op,
)

__extra_import_path__ = ["../../../..", "."]
import _k31_topk_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = inputs.on_sm103

_SPLIT_KV = 256


def _scores(n: int, data: str, gen: torch.Generator) -> torch.Tensor:
    if data == "ties":
        return torch.randint(0, 64, (n,), generator=gen, device="cuda").float()
    # Indexer-like scores: a broad negative bulk with a heavy positive tail.
    bulk = torch.randn(n, generator=gen, device="cuda") * 1.5 - 2.0
    tail = torch.rand(n, generator=gen, device="cuda") < 0.02
    spikes = torch.rand(n, generator=gen, device="cuda") * 6.0
    return torch.where(tail, bulk.abs() + spikes, bulk)


def _build(spec, seed: int) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    batch, next_n, k = spec["batch"], spec["next_n"], spec["topk"]
    lo, hi = spec["context_lens"]
    lens = torch.randint(lo, hi + 1, (batch,), generator=gen, device="cuda", dtype=torch.int32)
    width = spec["max_seq_len"]
    stride = -(-width // _SPLIT_KV) * _SPLIT_KV + spec["row_pad"]
    rows = batch * next_n
    base = torch.full((rows, stride), float("nan"), dtype=torch.float32, device="cuda")
    for r in range(rows):
        n = int(lens[r // next_n]) - next_n + r % next_n + 1
        base[r, :n] = _scores(n, spec["data"], gen)
    return dict(
        logits=base[:, :width],
        kv_lens=lens,
        indices=torch.full((rows, k), -5, dtype=torch.int32, device="cuda"),
        next_n=next_n,
        max_seq_len=width,
    )


def _sorted(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t < 0, 2**31 - 1, t).sort(dim=1).values


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    op(**args)
    out = args["indices"].clone()
    ref = torch.empty_like(out)
    op.reference(**dict(args, indices=ref))
    torch.cuda.synchronize()
    assert torch.equal(_sorted(out), _sorted(ref))
    # Deterministic: the same output, order included, on a second call.
    args["indices"].fill_(-5)
    op(**args)
    assert torch.equal(args["indices"], out)
    if cell.spec["data"] != "ties":
        # The CUDA radix it replaces selects the same sets on tie-free rows.
        rows = out.shape[0]
        radix = torch.empty_like(out)
        aux_i = torch.empty(rows, 10, out.shape[1], dtype=torch.int32, device="cuda")
        aux_l = torch.empty(rows, 10, out.shape[1], dtype=torch.float32, device="cuda")
        indexer_topk_decode(
            args["logits"],
            args["kv_lens"],
            radix,
            args["next_n"],
            out.shape[1],
            1,
            aux_i,
            aux_l,
            None,
        )
        assert torch.equal(_sorted(out), _sorted(radix))


def test_graph_replay_matches_eager():
    args = _build(op.CELLS[1].spec, seed=7)
    op(**args)
    eager = args["indices"].clone()
    args["indices"].fill_(-5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**args)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(args["indices"], eager)


def test_guard_refuses_ragged_rows():
    args = _build(op.CELLS[0].spec, seed=3)
    with pytest.raises(AssertionError, match="uniform next_n"):
        with validating(op):
            op(**dict(args, next_n=3))


def test_guard_refuses_nan_in_a_valid_column():
    args = _build(op.CELLS[0].spec, seed=4)
    logits = args["logits"].clone()
    logits[0, 10] = float("nan")
    with pytest.raises(AssertionError, match="NaN"):
        with validating(op):
            op(**dict(args, logits=logits))


# ---------------------------------------------------------------------------------------------
# indexer_topk_decode_gvr_paged: the decode rows as ordered KV-pool rows
# ---------------------------------------------------------------------------------------------

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.convert_req_index_to_global import (  # noqa: E402,E501
    convert_req_index_to_global,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_decode_gvr import (  # noqa: E402,E501
    _split_launch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_decode_gvr import (  # noqa: E402,E501
    indexer_topk_decode_gvr_paged as paged_op,
)

_BLOCK = 64
_SCALE = 24
_LAYER = 7


def _build_paged(spec, seed: int) -> dict:
    args = _build(spec, seed)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    batch = spec["batch"]
    width = -(-spec["context_lens"][1] // _BLOCK) + 1
    if spec.get("holes"):
        width = width // 2
    full = torch.randint(
        0, 50_000, (batch + 3, width), generator=gen, dtype=torch.int32, device="cuda"
    )
    if spec.get("holes"):
        full.masked_fill_(torch.rand(full.shape, generator=gen, device="cuda") < 0.05, -1)
    return dict(
        args,
        block_table=full[3:],
        block_size=_BLOCK,
        stride_factor=_SCALE * _BLOCK,
        layer_id=_LAYER,
    )


@pytest.mark.parametrize("cell", paged_op.CELLS, ids=[c.why[:48] for c in paged_op.CELLS])
def test_paged_certified_cells(cell, certifying):  # noqa: F811
    certifying(paged_op)
    args = _build_paged(cell.spec, inputs.seed_of(cell))
    paged_op(**args)
    out = args["indices"].clone()
    ref = torch.empty_like(out)
    paged_op.reference(**dict(args, indices=ref))
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    if cell.spec["data"] != "ties":
        # The current path: CUDA radix, torch sort, convert_req_index_to_global.
        rows, k = out.shape
        radix = torch.empty_like(out)
        aux_i = torch.empty(rows, 10, k, dtype=torch.int32, device="cuda")
        aux_l = torch.empty(rows, 10, k, dtype=torch.float32, device="cuda")
        indexer_topk_decode(
            args["logits"], args["kv_lens"], radix, args["next_n"], k, 1, aux_i, aux_l, None
        )
        ordered = _sorted(radix)
        ordered = torch.where(ordered == 2**31 - 1, -1, ordered).contiguous()
        req = (torch.arange(rows, device="cuda") // args["next_n"]).int()
        current = convert_req_index_to_global(
            req, args["block_table"], ordered, _BLOCK, k, _SCALE * _BLOCK, _LAYER
        )
        assert torch.equal(out, current)


@pytest.mark.parametrize(
    "cell_index,split",
    [(0, True), (2, True), (3, True), (5, False), (6, True), (7, True), (8, False), (9, False)],
)
def test_paged_takes_the_gvr_epilogue_on_split_launches(cell_index, split):
    """Launches of up to 49 rows are SPLIT launches (pool rows from the GVR epilogue); 72+ rows are not."""
    spec = paged_op.CELLS[cell_index].spec
    args = _build_paged(spec, seed=3)
    assert _split_launch(args["logits"], args["indices"].shape[1], args["max_seq_len"]) == split


def test_paged_graph_replay_matches_eager():
    args = _build_paged(paged_op.CELLS[0].spec, seed=9)
    paged_op(**args)
    eager = args["indices"].clone()
    args["indices"].fill_(-5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_op(**args)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(args["indices"], eager)


# ---------------------------------------------------------------------------------------------
# The SPLIT routing (CTAs per row up to 49 rows, tile width) is a launch choice: the rows are the
# same bits as the previous routing (one CTA per row past 32 rows + the order / remap launch, wider
# tiles), pool rows exactly and local columns as sets.
# ---------------------------------------------------------------------------------------------

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention import (  # noqa: E402,E501
    gvr_topk_decode_host as _host,
)

_ROUTING_SPECS = [
    dict(batch=1, next_n=5, context_lens=(60_000, 64_000)),
    dict(batch=1, next_n=8, context_lens=(118_000, 122_000)),
    dict(batch=2, next_n=5, context_lens=(160_000, 170_000)),
    dict(batch=2, next_n=8, context_lens=(100_000, 140_000)),
    dict(batch=3, next_n=5, context_lens=(250_000, 300_000)),
    dict(batch=3, next_n=8, context_lens=(60_000, 170_000)),
    dict(batch=4, next_n=5, context_lens=(1_000, 120_000)),
    dict(batch=6, next_n=5, context_lens=(110_000, 130_000)),
    dict(batch=5, next_n=8, context_lens=(118_000, 122_000)),
    dict(batch=6, next_n=8, context_lens=(1_500, 170_000)),
    dict(batch=7, next_n=5, context_lens=(100_000, 140_000)),
    dict(batch=8, next_n=8, context_lens=(60_000, 130_000)),
    dict(batch=9, next_n=8, context_lens=(118_000, 122_000)),
    dict(batch=2, next_n=8, context_lens=(60_000, 80_000), data="ties"),
    dict(batch=6, next_n=8, context_lens=(30_000, 40_000), data="ties"),
]


def _routing_spec(d):
    return dict(
        topk=2048,
        max_seq_len=1 << 20,
        row_pad=0,
        data=d.get("data", "real"),
        **{k: v for k, v in d.items() if k != "data"},
    )


def _run_with_routing(fn, previous: bool):
    saved = (_host._VARLEN_SPLIT_MAX_ROWS, _host._VARLEN_TILE_CAPS)
    try:
        if previous:
            _host._VARLEN_SPLIT_MAX_ROWS = 32
            _host._VARLEN_TILE_CAPS = ((24, 2), (12, 4))
        _host._VARLEN_CACHE.clear()
        _host._SPLIT_LAUNCH_CACHE.clear()
        fn()
        torch.cuda.synchronize()
    finally:
        _host._VARLEN_SPLIT_MAX_ROWS, _host._VARLEN_TILE_CAPS = saved
        _host._VARLEN_CACHE.clear()
        _host._SPLIT_LAUNCH_CACHE.clear()


@pytest.mark.parametrize(
    "spec",
    _ROUTING_SPECS,
    ids=[f"{d['batch']}x{d['next_n']}-{d.get('data', 'real')}" for d in _ROUTING_SPECS],
)
def test_split_routing_matches_previous_routing(spec):
    full = _routing_spec(spec)
    args = _build_paged(full, seed=11 + spec["batch"] * 16 + spec["next_n"])
    outs = {}
    for previous in (False, True):
        args["indices"].fill_(-5)
        _run_with_routing(lambda: paged_op(**args), previous)
        outs[previous] = args["indices"].clone()
    assert torch.equal(outs[False], outs[True])
    plain = {
        k: v
        for k, v in args.items()
        if k not in ("block_table", "block_size", "stride_factor", "layer_id")
    }
    sets = {}
    for previous in (False, True):
        plain["indices"].fill_(-5)
        _run_with_routing(lambda: op(**plain), previous)
        sets[previous] = _sorted(plain["indices"])
    assert torch.equal(sets[False], sets[True])
