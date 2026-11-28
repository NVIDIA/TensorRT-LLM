# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: `qsa_index_scores_fp8` (e4m3 index anchors) over its cells against the float64
reference on the e4m3 operands; `index_anchors_to_fp8` against torch's e4m3 rounding; the
ctx-tile kernel bitwise the request-group kernel; graph replay against eager."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp import archs
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention import (
    qsa_index_scores_fp8 as _module,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_index_scores_fp8 import (
    index_anchors_to_fp8,
    qsa_index_scores_fp8,
)

#: The GPU cases; the operand-check cases run without a GPU (`cpu_only`).
_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or not archs.served(torch.cuda.get_device_capability()),
    reason=f"qsa_index_scores_fp8 runs on {archs.served_names()}",
)

DEV = "cuda"
TPB, RATIO, DIM = 32, 4, 128


def build(spec, seed, convert_every_position=False):
    """A bf16 index plane (each row's slots of a scratch-sliced `[pages, 32, 1, 128]` view), its
    anchors converted to e4m3 by the op under test, and the call's arguments.
    `convert_every_position`: the converter is handed every position of every request, anchors
    and not, as a writer's call names them; it must convert the anchors only."""
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
    if convert_every_position:
        named = torch.cat([torch.arange(0, s + n, device=DEV) for s, n in requests])
        named_owners = torch.cat(
            [torch.full((s + n,), r, device=DEV) for r, (s, n) in enumerate(requests)]
        ).to(torch.int32)
        index_anchors_to_fp8(plane, block_table, named_owners, named.long(), TPB, RATIO)
    else:
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


@_GPU
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


@_GPU
def test_anchor_convert_is_torch_e4m3_rounding():
    """The converter is handed every position (anchors and the raw positions between them): the
    anchors are torch's e4m3 rounding, every other slot it was handed keeps its bf16 key."""
    requests = ((0, 300), (1000, 9))
    spec = _module._scores_spec(requests=requests)
    args, bf16_keys, (every, owners) = build(spec, seed=3, convert_every_position=True)
    plane8 = args["index_cache"].view(torch.float8_e4m3fn)
    table = args["block_table"]
    pages = table[owners.long(), every // TPB].long()
    slots = every % TPB
    want = bf16_keys[pages, slots, 0].to(torch.float8_e4m3fn)
    assert torch.equal(plane8[pages, slots, 0, :DIM].view(torch.uint8), want.view(torch.uint8))
    # every raw (non-anchor) position handed to the converter keeps its bf16 bytes
    raw = torch.cat([torch.arange(0, s + n, device=DEV) for s, n in requests])
    raw_owners = torch.cat(
        [torch.full((s + n,), r, device=DEV) for r, (s, n) in enumerate(requests)]
    )
    keep = (raw + 1) % RATIO != 0
    raw, raw_owners = raw[keep], raw_owners[keep].long()
    raw_pages = table[raw_owners, raw // TPB].long()
    assert raw.numel() > 0
    assert torch.equal(
        args["index_cache"][raw_pages, raw % TPB, 0], bf16_keys[raw_pages, raw % TPB, 0]
    )


@pytest.fixture
def _launches(monkeypatch):
    """The kernel each public call reached (`flat`, `ctx_tile` or `group`)."""
    taken = []
    for name, tag in (
        ("_qsa_index_scores_flat", "flat"),
        ("_qsa_index_scores_fp8_ctx_tile", "ctx_tile"),
        ("_qsa_index_scores_fp8", "group"),
    ):
        real = getattr(_module, name)

        def record(*a, _real=real, _tag=tag, **k):
            taken.append(_tag)
            return _real(*a, **k)

        monkeypatch.setattr(_module, name, record)
    return taken


@_GPU
def test_ctx_tile_is_bitwise_the_group_kernel(_launches):
    spec = _module._scores_spec(requests=((5000, 300), (17, 200)), context_rows=True)
    args, _, _ = build(spec, seed=11)
    tiled = qsa_index_scores_fp8(**args)
    assert _launches == ["ctx_tile"]
    group = _module._qsa_index_scores_fp8(
        args["q"], args["index_cache"], args["block_table"], args["query_positions"],
        args["request_indices"], TPB, RATIO, True, True, 1,
    )  # fmt: skip
    for r in range(tiled.shape[0]):
        v = (int(args["query_positions"][r]) + 1) // RATIO
        assert torch.equal(tiled[r, :v], group[r, :v])


@_GPU
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
@_GPU
def test_flat_is_bitwise_the_group_kernel(requests, rpr, _launches):
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
    assert _launches == ["flat"]
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


#: A page index far past any pool: a load through it faults.
_FAR_PAGE = 1 << 28


@_GPU
@pytest.mark.parametrize(
    "rpr, context_rows, kernel",
    [(2, False, "flat"), (2, False, "group"), (1, True, "group")],
    ids=["flat_rows_by_row", "group_rows_by_row", "group_context_run"],
)
def test_a_short_request_in_a_mixed_run_reads_only_its_own_pages(rpr, context_rows, kernel):
    """Two requests in one run at positions 3 and 35 (ratio 4, 32-token pages): the first sees
    one block, the second nine. The first request's table holds one page; its next entry names
    no page of the pool. Every kernel reads a row's keys only up to that row's own visible
    blocks, so the run's longer row does not take the short row through that entry; both rows'
    visible scores match the reference."""
    spec = _module._scores_spec(
        requests=((3, 1), (35, 1)), rows_per_request=rpr, context_rows=context_rows
    )
    args, _, _ = build(spec, seed=17)
    args["block_table"][0, 1:] = _FAR_PAGE
    if kernel == "group":
        out = _module._qsa_index_scores_fp8(**{**args, "only_visible_blocks": True})
    else:
        assert (
            _module._flat_refusal(
                args["q"],
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
        out = qsa_index_scores_fp8(**args)
    torch.cuda.synchronize()
    ref = qsa_index_scores_fp8.reference(**args)
    _visible_close(out, ref, args["query_positions"])


@_GPU
def test_strided_row_metadata_is_read_as_its_values():
    """Positions and request indices given as strided views score as the same values in
    contiguous storage, on the request-group kernel (decode rows of two a request) and on the
    ctx-tile kernel (a 300-row context call)."""
    for spec in (
        _module._scores_spec(
            requests=tuple((900 + 311 * r, 2) for r in range(70)), rows_per_request=2, table=8196
        ),
        _module._scores_spec(requests=((5000, 300),), context_rows=True),
    ):
        args, _, _ = build(spec, seed=19)
        want = qsa_index_scores_fp8(**args)
        rows = args["query_positions"].numel()
        pos = torch.zeros(2 * rows, dtype=torch.int64, device=DEV)
        req = torch.zeros(2 * rows, dtype=torch.int32, device=DEV)
        pos[::2], req[::2] = args["query_positions"], args["request_indices"]
        got = qsa_index_scores_fp8(
            **{**args, "query_positions": pos[::2], "request_indices": req[::2]}
        )
        for r in range(rows):
            v = (int(args["query_positions"][r]) + 1) // RATIO
            assert torch.equal(got[r, :v], want[r, :v]), r


def _cpu_args(rows: int, heads: int = 4, dim: int = DIM, table=None):
    q = torch.zeros(rows, heads, dim, dtype=torch.bfloat16)
    plane = torch.zeros(8, TPB, 1, dim, dtype=torch.bfloat16)
    table = torch.zeros(1, 16, dtype=torch.int32) if table is None else table
    return dict(
        q=q,
        index_cache=plane,
        block_table=table,
        query_positions=torch.arange(rows, dtype=torch.int64),
        request_indices=torch.zeros(rows, dtype=torch.int32),
        tokens_per_block=TPB,
        compress_ratio=RATIO,
        only_visible_blocks=True,
        context_rows=True,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows", [256, 257])
def test_a_strided_block_table_is_refused_on_both_sides_of_the_ctx_tile(rows):
    """A block table sliced `[:, ::2]` is refused before any kernel is picked: at 256 context
    rows (the request-group kernel) and at 257 (the ctx-tile kernel) alike."""
    wide = torch.zeros(1, 32, dtype=torch.int32)
    args = _cpu_args(rows, table=wide[:, ::2])
    assert _module._takes_ctx_tile(args["q"], True, True, 1) is (rows > 256)
    with pytest.raises(ValueError, match="block table"):
        _module.qsa_index_scores_fp8_launch(**args)


@pytest.mark.cpu_only
@pytest.mark.parametrize("dim", [16, 24])
def test_index_head_dimensions_below_the_fp8_dot_are_refused(dim):
    """Triton's 8-bit dot contracts at least 32: the flat guard declines dimension 16, and the
    entry refuses it (and any non-power of two) before a kernel would fail to compile."""
    args = _cpu_args(4, dim=dim)
    args["context_rows"] = False
    if dim == 16:
        why = _module._flat_refusal(
            args["q"], args["index_cache"], args["block_table"], args["query_positions"],
            args["request_indices"], TPB, RATIO, 1,
        )  # fmt: skip
        assert why is not None and "dimension" in why
    with pytest.raises(ValueError, match="dimension"):
        _module.qsa_index_scores_fp8_launch(**args)
