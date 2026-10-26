# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the csa_index_scores entry.

The paged state is real: a pool of pages larger than the requests map, shuffled
page ids, an int32 block table whose rows are request *slots* (more slots than
requests, assigned by a seeded permutation) with -1 past each sequence. The
"anchor" layout stores each region's summary in its last-token slot of a bf16
`[pages, 32, 512]` k|z plane whose other slots hold large garbage, and hands the
op the strided view `plane[:, 7::8, :256]`; the "dense" layout is a
`[pages, 4, 256]` e4m3 region plane, the one the target passes. The pool's
highest page is always mapped. q is a head view of a wider projection row.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified.ops.attention.csa_index_scores import (  # noqa: E501
    csa_index_scores as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "csa_index_scores requires a CUDA device"

DEV = torch.device("cuda")
#: Positions every ragged decode cell carries: region edges, the all-inclusive
#: selection edge (4104) and the last position of `max_seq_len` 9216.
_EDGE_POSITIONS = (0, 7, 8, 15, 16, 4103, 4104, 9215)
#: Unit roundoff of fp32.
_U32 = 2.0**-24


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _segments(spec, gen):
    """(first position, rows) per request, in row order."""
    if "segments" in spec:
        return list(spec["segments"])
    rows = spec["rows"]
    extra = rows - len(_EDGE_POSITIONS)
    drawn = torch.randint(0, 9216, (max(extra, 0),), generator=gen, device=DEV).tolist()
    return [(p, 1) for p in (list(_EDGE_POSITIONS) + drawn)[:rows]]


def _build(spec, seed):
    """Real paged state for one spec, keyed by `__call__`'s parameter names."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    groups, per_group, dim = spec["groups"], spec["heads_per_group"], spec["proxy_dim"]
    heads, region, tpb = groups * per_group, spec["region_size"], spec["tokens_per_block"]
    per_page = tpb // region
    segments = _segments(spec, gen)
    rows = sum(n for _, n in segments)

    # Pages each request maps: every token stored so far, through its last row.
    mapped = [math.ceil((first + n) / tpb) for first, n in segments]
    assert max(mapped) <= spec["table_pages"]
    pool_pages = sum(mapped) + spec["pool_pages_spare"]
    page_ids = torch.randperm(pool_pages, generator=gen, device=DEV).to(spec["table_dtype"])
    # The pool's highest page is always one a request maps, so the top page
    # index is driven.
    top = int((page_ids == pool_pages - 1).nonzero())
    if top >= sum(mapped):
        swap = torch.tensor([top, sum(mapped) - 1], device=DEV)
        page_ids[swap] = page_ids[swap.flip(0)]

    if spec["table"] == "shared":
        assert len(segments) == 1, "a shared table serves one request"
        slot_of = [0]
        slots = 1
    else:
        slots = spec["slots"]
        assert slots > len(segments), "pool slots must exceed the batch"
        cpu_gen = torch.Generator().manual_seed(seed)
        slot_of = torch.randperm(slots, generator=cpu_gen)[: len(segments)].tolist()
    block_table = torch.full(
        (slots, spec["table_pages"]), -1, dtype=spec["table_dtype"], device=DEV
    )

    if spec["layout"] == "anchor":
        plane = (torch.randn(pool_pages, tpb, 2 * dim, generator=gen, device=DEV) * 64.0).to(
            torch.bfloat16
        )
        summary_cache = plane[:, region - 1 :: region, :dim]
        assert summary_cache.shape == (pool_pages, per_page, dim)
        assert not summary_cache.is_contiguous()
    elif spec["layout"] == "dense":
        summary_cache = (
            torch.randn(pool_pages, per_page, dim, generator=gen, device=DEV) * 64.0
        ).to(torch.float8_e4m3fn)
    else:
        raise AssertionError(f"unknown layout {spec['layout']!r}")

    cursor = 0
    for (first, n), slot, count in zip(segments, slot_of, mapped):
        pages = page_ids[cursor : cursor + count]
        block_table[slot, :count] = pages
        cursor += count
        complete = (first + n) // region  # regions whose 8 tokens are all stored
        if complete:
            r = torch.arange(complete, device=DEV)
            values = torch.randn(complete, dim, generator=gen, device=DEV) * 0.5
            summary_cache[pages.long()[r // per_page], r % per_page] = values.to(
                torch.float8_e4m3fn
            ).to(summary_cache.dtype)

    assert int(block_table.max()) == pool_pages - 1, "the pool's last page is not mapped"

    positions = torch.cat([torch.arange(first, first + n, device=DEV) for first, n in segments]).to(
        spec["positions_dtype"]
    )
    request_indices = torch.cat(
        [torch.full((n,), s, device=DEV) for (_, n), s in zip(segments, slot_of)]
    ).to(torch.int32)

    # q is a head view of the indexer's q|k|z|w projection row.
    width = heads * dim + 2 * dim + heads
    proj = torch.randn(rows, width, generator=gen, device=DEV).to(spec["q_dtype"])
    q = proj[:, : heads * dim].view(rows, heads, dim)
    if rows > 1:
        assert not q.is_contiguous()
    weights = (torch.randn(rows, heads, generator=gen, device=DEV) * 0.5).to(torch.bfloat16)
    weights = weights.to(spec["weights_dtype"]) * 0.5

    return dict(
        q=q,
        weights=weights,
        summary_cache=summary_cache,
        block_table=block_table,
        request_indices=request_indices,
        positions=positions,
        num_groups=groups,
        region_size=region,
        num_columns=spec["num_columns"],
    )


def _hidden(args):
    """[G*T, C] True where a column is past its row's history."""
    columns = torch.arange(args["num_columns"], device=DEV)
    bound = args["positions"].long() // args["region_size"]
    hidden = columns[None, :] >= bound[:, None]
    return hidden.repeat(args["num_groups"], 1)


def _abs(t):
    """|t|, by clearing the sign bit for e4m3 (torch has no fp8 abs kernel)."""
    if t.dtype == torch.float8_e4m3fn:
        return (t.view(torch.uint8) & 0x7F).view(torch.float8_e4m3fn)
    return t.abs()


def _band(args):
    """Per-element bound on |op - reference|, from the operands' magnitudes.

    Every product of two e4m3 values is exact in fp32, so the op's error is
    summation alone: D-1 additions in the dot (any order, any blocking), the
    weight multiply, the head sum's Hg additions, and the reference's own final
    rounding to fp32. The worst-case bound for n roundings is n * u * A with A
    the same sum over absolute values, so the band is (D + Hg + 2) * 2^-24 * A,
    A = sum_h |w_h| sum_d |q_hd s_rd|. relu is 1-Lipschitz, so a dot that
    straddles zero stays inside. A is the reference fed absolute values (relu
    is then the identity; e4m3 rounding commutes with abs).
    """
    dim = args["q"].shape[-1]
    per_group = args["q"].shape[1] // args["num_groups"]
    magnitude = op.reference(
        **{
            **args,
            "q": args["q"].abs(),
            "weights": args["weights"].abs(),
            "summary_cache": _abs(args["summary_cache"]),
        }
    )
    return (dim + per_group + 2) * _U32 * magnitude


def _compare(out, ref, args):
    """`-inf` exactly where the row's history ends; the derived band elsewhere.

    Returns the worst |out - ref| / band so callers can log it.
    """
    rows = args["q"].shape[0]
    assert out.dtype == ref.dtype == torch.float32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape == (args["num_groups"] * rows, args["num_columns"])
    hidden = _hidden(args)
    assert torch.equal(torch.isneginf(ref), hidden), "reference mask disagrees with bound"
    assert torch.equal(torch.isneginf(out), hidden), "op is -inf at the wrong columns"
    if not bool((~hidden).any()):
        return 0.0
    band = _band(args)[~hidden]
    diff = (out[~hidden] - ref[~hidden]).abs()
    assert bool(torch.isfinite(out[~hidden]).all()), "non-finite score in a visible column"
    worst = float((diff / band.clamp_min(torch.finfo(torch.float32).tiny)).max())
    assert bool((diff <= band).all()), f"outside the derived band: worst {worst:.3g}x"
    return worst


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=_seed(cell.why))
    out = op(**args)
    worst = _compare(out, op.reference(**args), args)
    print(f"{cell.why[:44]}: worst |op - ref| = {worst:.3g} of the band")


def _variant(args, *, round_q=True, relu_after_sum=False, interleave=False, slot_shift=0):
    """A deliberately wrong reading of the contract, in float64.

    `slot_shift` reads each summary from a different token slot of the region
    (anchor layout only: the view is re-taken at the shifted offset).
    """
    q, w = args["q"], args["weights"]
    rows, heads, dim = q.shape
    groups = args["num_groups"]
    per_group = heads // groups
    cache = args["summary_cache"]
    if slot_shift:
        token_stride = cache.stride(1) // args["region_size"]
        cache = cache.as_strided(
            cache.shape, cache.stride(), cache.storage_offset() - slot_shift * token_stride
        )
    per_page = cache.shape[1]
    pages = -(-args["num_columns"] // per_page)
    qd = (q.to(torch.float8_e4m3fn) if round_q else q).double()
    if interleave:
        order = torch.arange(heads, device=DEV).view(per_group, groups).t().reshape(-1)
        qd, w = qd[:, order], w[:, order]
    qd = qd.reshape(rows, groups, per_group, dim)
    wd = w.double().reshape(rows, groups, per_group)
    out = torch.empty(rows, groups, args["num_columns"], dtype=torch.float64, device=DEV)
    table = args["block_table"].long()
    for t in range(rows):
        row = 0 if table.shape[0] == 1 else int(args["request_indices"][t])
        logical = cache[table[row, :pages].clamp_min(0)].reshape(-1, dim)
        keys = logical[: args["num_columns"]].to(torch.float8_e4m3fn).double()
        dots = torch.einsum("ghd,cd->ghc", qd[t], keys)
        if relu_after_sum:
            out[t] = torch.relu((dots * wd[t][..., None]).sum(1))
        else:
            out[t] = (torch.relu(dots) * wd[t][..., None]).sum(1)
    out = out.float().masked_fill(_hidden(args).view(groups, rows, -1).transpose(0, 1), -math.inf)
    return out.transpose(0, 1).reshape(groups * rows, -1)


def test_wrong_variants_land_outside_the_band() -> None:
    """The band is derived, not tuned: each misreading of the contract must fall
    outside it. Query-major rows are checked against the group-major output."""
    spec = dict(op.CELLS[0].spec, rows=24, slots=32, pool_pages_spare=16)
    args = _build(spec, seed=_seed("wrong variants"))
    out = op(**args)
    ref = op.reference(**args)
    _compare(out, ref, args)
    hidden = _hidden(args)
    band = _band(args)[~hidden]

    def excess(candidate):
        return float(((candidate[~hidden] - ref[~hidden]).abs() / band).max())

    measured = {
        "q not rounded to e4m3": excess(_variant(args, round_q=False)),
        "relu after the head sum": excess(_variant(args, relu_after_sum=True)),
        "heads interleaved across groups": excess(_variant(args, interleave=True)),
        "summary read from the region's first token": excess(_variant(args, slot_shift=7)),
    }
    groups, rows = args["num_groups"], args["q"].shape[0]

    def query_major(t):
        return t.view(groups, rows, -1).transpose(0, 1).reshape(groups * rows, -1)

    both = ~(hidden | query_major(hidden))
    measured["query-major row order"] = float(
        ((query_major(ref)[both] - ref[both]).abs() / _band(args)[both]).max()
    )
    print("wrong variants, worst |variant - ref| in bands:", measured)
    for name, value in measured.items():
        assert value > 1.0, f"{name} lands inside the band ({value:.3g})"


def _capture_and_replay(spec, tag):
    """Capture the op on `spec`, rewrite q, weights, positions and the summaries
    in place, replay, and match an eager call bitwise and the reference in band."""
    args = _build(spec, seed=_seed(tag))
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        op(**args)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = op(**args)
    for trial in range(3):
        fresh = _build(spec, seed=_seed(f"{tag} replay {trial}"))
        args["q"].copy_(fresh["q"])
        args["weights"].copy_(fresh["weights"])
        # Earlier positions stay inside the pages each request maps.
        args["positions"].copy_((args["positions"] - 9 * (trial + 1)).clamp_min(0))
        cache = args["summary_cache"]
        cache.copy_(
            (torch.randn(cache.shape, device=DEV) * 0.5).to(torch.float8_e4m3fn).to(cache.dtype)
        )
        graph.replay()
        torch.cuda.synchronize()
        eager = op(**args)
        assert torch.equal(graph_out, eager), f"{tag}: replay {trial} differs from eager"
        _compare(graph_out, op.reference(**args), args)


def test_cuda_graph_capture_and_replay() -> None:
    """The decode site is captured: 8 rows at 1152 columns on the anchor view,
    4 groups."""
    _capture_and_replay(dict(op.CELLS[0].spec, rows=8, slots=16, pool_pages_spare=16), "graph")


#: The tp4/tep4 decode site as the target runs it: the dense e4m3 plane, 1 group.
_DENSE_TP4_DECODE = next(
    c.spec
    for c in op.CELLS
    if c.spec["layout"] == "dense" and c.spec["groups"] == 1 and c.spec.get("rows") == 128
)


@pytest.mark.parametrize("rows", (8, 128))
def test_cuda_graph_capture_and_replay_dense_one_group(rows) -> None:
    """The tp4/tep4 decode site is captured on the dense e4m3 plane at 1 group,
    at the smallest and largest captured batch the call site sees here."""
    _capture_and_replay(dict(_DENSE_TP4_DECODE, rows=rows), f"graph dense g1 {rows}")


def test_history_wider_than_the_output_is_truncated() -> None:
    """A caller precondition, pinned: a row whose history (position // 8) exceeds
    `num_columns` loses the regions past the output silently -- every column is
    finite and nothing raises."""
    spec = dict(op.CELLS[2].spec, num_columns=1000)
    args = _build(spec, seed=_seed("truncated"))
    assert int(args["positions"][0]) // args["region_size"] == 1151
    out = op(**args)
    assert bool(torch.isfinite(out).all())
    _compare(out, op.reference(**args), args)


def test_e4m3_rounding_saturates() -> None:
    """The note's cast semantics: torch's e4m3 cast saturates at +-448 on this
    build, so an out-of-range q element scores as 448, not NaN."""
    probe = torch.tensor([455.0, 500.0, 1.0e4, -500.0, math.inf], device=DEV)
    assert probe.to(torch.float8_e4m3fn).float().tolist() == [448.0, 448.0, 448.0, -448.0, 448.0]
    spec = dict(op.CELLS[2].spec)
    args = _build(spec, seed=_seed("saturate"))
    big = args["q"].clone()
    big[0, 0, 0] = 1.0e4
    clipped = big.clone()
    clipped[0, 0, 0] = 448.0
    assert torch.equal(op(**{**args, "q": big}), op(**{**args, "q": clipped}))


def test_rejects_shapes_it_cannot_score() -> None:
    """The op's own loud refusals -- why the entry has no `is_valid`."""
    args = _build(dict(op.CELLS[0].spec, rows=8, slots=16, pool_pages_spare=16), seed=7)
    with pytest.raises(ValueError, match="do not split"):
        op(**{**args, "num_groups": 3})
    with pytest.raises(ValueError, match="weights must be"):
        op(**{**args, "weights": args["weights"][:, :4]})
    with pytest.raises(ValueError, match="one entry per row"):
        op(**{**args, "positions": args["positions"][:4]})
    with pytest.raises(ValueError, match="summary width"):
        op(**{**args, "summary_cache": args["summary_cache"][..., :128]})
    with pytest.raises(ValueError, match="block_table must be"):
        op(**{**args, "block_table": args["block_table"][:, :100]})
    with validating(op):
        op(**args)
