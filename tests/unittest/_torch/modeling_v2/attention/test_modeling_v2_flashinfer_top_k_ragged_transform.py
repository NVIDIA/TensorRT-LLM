# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_top_k_ragged_transform entry.

The op selects; it computes nothing. `_compare` therefore checks the output
exactly: the selected id set (and, under `deterministic=True`, the order),
the `-1` padding and the prefix layout. The scores are shaped like Step-5's
indexer output -- a sum over 4 heads of `relu(dot) * w` with signed `w`,
accumulated onto `+0.0`, so about 1 in 16 visible columns is an exact zero --
and the rows are group-major with `lengths = position // 8`.
"""

import zlib

import pytest
import torch
from flashinfer.topk import TopKTieBreak

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.flashinfer_top_k_ragged_transform import (  # noqa: E501
    flashinfer_top_k_ragged_transform as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified.ops.attention.csa_index_scores import (  # noqa: E501
    csa_index_scores,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "flashinfer_top_k_ragged_transform requires a CUDA device"

DEV = torch.device("cuda")
K = 512
#: Decode positions every ragged decode cell carries: region edges, the visible
#: counts on either side of `k` (511, 512, 513) and the last position of
#: `max_seq_len` 9216 (1151 visible regions).
EDGE_POSITIONS = (0, 7, 8, 4095, 4096, 4104, 4111, 9215)
#: Indexer heads per provider group and the proxy width (csa_index_scores' geometry).
HEADS_PER_GROUP = 4
PROXY_DIM = 256


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _positions(spec: dict, gen: torch.Generator) -> torch.Tensor:
    kind = spec["positions"][0]
    if kind == "ragged":
        rows = spec["positions"][1]
        drawn = torch.randint(0, 9216, (rows - len(EDGE_POSITIONS),), generator=gen, device=DEV)
        return torch.cat([torch.tensor(EDGE_POSITIONS, device=DEV), drawn]).to(torch.int32)
    assert kind == "chunk", kind
    _, first, rows = spec["positions"]
    return torch.arange(first, first + rows, dtype=torch.int32, device=DEV)


def _relu_heads(rows: int, columns: int, gen: torch.Generator, kind: str) -> torch.Tensor:
    """sum_h relu(x_h) * w_h accumulated onto +0.0, as the indexer does."""
    scores = torch.zeros(rows, columns, device=DEV)
    for _ in range(HEADS_PER_GROUP):
        x = torch.randn(rows, columns, generator=gen, device=DEV)
        w = torch.randn(rows, 1, generator=gen, device=DEV)
        if kind == "zero_cut":
            # Mostly-negative dots and positive weights: ~44% of the visible
            # columns are exactly 0 and fewer than k are positive on many rows,
            # so the cut lands inside the zero group.
            x, w = x - 0.9, w.abs()
        scores += torch.relu(x) * w
    return scores


def _producer(groups: int, positions: torch.Tensor, columns: int, gen: torch.Generator):
    """csa_index_scores' own output on a shared one-row table (the prefill path).

    Group 0's weights are all negative, so every all-relu-zero region there is
    a sum of `-0.0` products: the case that would hand this op a `-0.0`.
    """
    rows = positions.numel()
    heads = groups * HEADS_PER_GROUP
    per_page = 4
    table_pages = -(-columns // per_page)
    pool_pages = table_pages + 48
    q = torch.randn(rows, heads, PROXY_DIM, generator=gen, device=DEV).to(torch.bfloat16)
    weights = torch.randn(rows, heads, generator=gen, device=DEV) * 0.5
    weights[:, :HEADS_PER_GROUP] = -weights[:, :HEADS_PER_GROUP].abs()
    summaries = torch.randn(pool_pages, per_page, PROXY_DIM, generator=gen, device=DEV)
    table = torch.randperm(pool_pages, generator=gen, device=DEV)[:table_pages]
    table = table.to(torch.int32)[None, :]
    scores = csa_index_scores(
        q,
        weights,
        summaries.to(torch.bfloat16),
        table,
        torch.zeros(rows, dtype=torch.int32, device=DEV),
        positions,
        groups,
        8,
        columns,
    )
    visible = (
        torch.arange(columns, device=DEV)[None, :] < (positions.long() // 8).repeat(groups)[:, None]
    )
    zeros = (scores == 0) & visible
    assert bool(zeros.any()), "the producer cell must carry exact zeros"
    assert not bool((zeros & torch.signbit(scores)).any()), "csa_index_scores emitted -0.0"
    return scores


def _build(spec: dict, seed: int) -> dict:
    """The op's arguments for one cell, keyed by `__call__`'s parameter names."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    groups, columns = spec["groups"], spec["columns"]
    positions = _positions(spec, gen)
    lengths = (positions // spec["region_size"]).repeat(groups).to(torch.int32)
    assert int(lengths.max()) <= columns, "a row's history must fit the score width"
    rows = lengths.numel()
    if spec["scores"] == "producer":
        scores = _producer(groups, positions, columns, gen)
    else:
        scores = _relu_heads(rows, columns, gen, spec["scores"])
    if spec["tail"] is not None:
        hidden = torch.arange(columns, device=DEV)[None, :] >= lengths.long()[:, None]
        scores.masked_fill_(hidden, float(spec["tail"]))
    return dict(
        input=scores,
        offsets=torch.zeros(rows, dtype=torch.int32, device=DEV),
        lengths=lengths,
        k=spec["k"],
        deterministic=spec["deterministic"],
        tie_break=TopKTieBreak.SMALL,
        dsa_graph_safe=spec["dsa_graph_safe"],
    )


def _ascending(ids: torch.Tensor) -> torch.Tensor:
    """Each row's ids ascending with every -1 moved to the end."""
    big = torch.iinfo(torch.int32).max
    ordered = torch.sort(ids.masked_fill(ids == -1, big), dim=1).values
    return ordered.masked_fill(ordered == big, -1)


def _compare(out: torch.Tensor, ref: torch.Tensor, deterministic: bool) -> None:
    """Exact, nothing tolerated: a selection does no arithmetic.

    `ref` is ascending and `-1`-padded. The set (and the count, the padding and
    the absence of duplicates) is compared through `_ascending`; the order
    only under `deterministic=True`, which promises it.
    """
    assert out.dtype == torch.int32, out.dtype
    assert out.shape == ref.shape, (out.shape, ref.shape)
    valid = out != -1
    assert not bool((valid[:, 1:] & ~valid[:, :-1]).any()), "a valid id follows a -1 pad"
    if deterministic:
        assert torch.equal(out, ref), "deterministic ids differ from the ascending selection"
    else:
        wrong = (_ascending(out) != ref).any(1)
        assert not bool(wrong.any()), f"{int(wrong.sum())} rows select the wrong set"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, _seed(cell.why))
    if cell.spec["scores"] == "zero_cut":
        positive = ((args["input"] > 0) & ~torch.isnan(args["input"])).sum(1)
        cut_in_zeros = (positive < K) & (args["lengths"] > K)
        assert int(cut_in_zeros.sum()) > 0, "no row's cut falls inside the zero group"
    out = op(**args)
    _compare(out, op.reference(**args), cell.spec["deterministic"])


@pytest.mark.parametrize(
    "dsa_graph_safe, deterministic",
    [(False, False), (True, False), (False, True)],
    ids=["default", "graph_safe", "deterministic"],
)
def test_cuda_graph_capture_and_replay(dsa_graph_safe: bool, deterministic: bool) -> None:
    """The decode site: 512 rows (4 groups x 128) at 1152, captured once, then
    replayed against new scores and new lengths written in place."""
    gen = torch.Generator(device=DEV).manual_seed(_seed(f"graph{dsa_graph_safe}{deterministic}"))
    rows, columns = 4 * 128, 1152
    scores = _relu_heads(rows, columns, gen, "relu_heads")
    lengths = torch.randint(0, columns, (rows,), generator=gen, device=DEV).to(torch.int32)
    offsets = torch.zeros(rows, dtype=torch.int32, device=DEV)
    kwargs = dict(
        deterministic=deterministic, tie_break=TopKTieBreak.SMALL, dsa_graph_safe=dsa_graph_safe
    )

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        op(scores, offsets, lengths, K, **kwargs)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(scores, offsets, lengths, K, **kwargs)

    for _ in range(3):
        lengths.copy_(torch.randint(0, columns, (rows,), generator=gen, device=DEV))
        fresh = _relu_heads(rows, columns, gen, "relu_heads")
        hidden = torch.arange(columns, device=DEV)[None, :] >= lengths.long()[:, None]
        scores.copy_(fresh.masked_fill(hidden, float("nan")))
        out.fill_(12345)
        graph.replay()
        torch.cuda.synchronize()
        _compare(out, op.reference(scores, offsets, lengths, K, **kwargs), deterministic)
        if deterministic:
            assert torch.equal(out, op(scores, offsets, lengths, K, **kwargs))


@pytest.mark.parametrize("batch", [8, 128], ids=["batch8", "batch128"])
@pytest.mark.parametrize("groups", [1, 4], ids=["1group", "4groups"])
def test_cuda_graph_deterministic_decode(groups: int, batch: int) -> None:
    """The decode call Step-5's modeling captures: `deterministic=True`,
    `dsa_graph_safe=False`, `groups * batch` group-major rows at 1152 with
    `lengths = position // 8`. Captured once, then replayed after rewriting
    the scores (relu_heads, then zero_cut, alternating) and the lengths in
    place; every replay must equal an eager call on the same inputs, ids and
    order, and the reference's ascending selection exactly."""
    gen = torch.Generator(device=DEV).manual_seed(_seed(f"det graph {groups} {batch}"))
    spec = dict(
        groups=groups,
        positions=("ragged", batch),
        columns=1152,
        k=K,
        region_size=8,
        deterministic=True,
        dsa_graph_safe=False,
        scores="relu_heads",
        tail="nan",
    )
    args = _build(spec, _seed(f"det graph build {groups} {batch}"))
    scores, offsets, lengths = args["input"], args["offsets"], args["lengths"]
    rows, columns = scores.shape
    assert rows == groups * batch and columns == 1152, (rows, columns)
    kwargs = dict(deterministic=True, tie_break=TopKTieBreak.SMALL, dsa_graph_safe=False)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        op(scores, offsets, lengths, K, **kwargs)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(scores, offsets, lengths, K, **kwargs)

    for replay, kind in enumerate(("relu_heads", "zero_cut", "relu_heads", "zero_cut")):
        positions = torch.randint(0, 9216, (batch,), generator=gen, device=DEV)
        if replay == 0:
            positions[: min(batch, len(EDGE_POSITIONS))] = torch.tensor(
                EDGE_POSITIONS[:batch], device=DEV
            )
        lengths.copy_((positions // 8).repeat(groups))
        fresh = _relu_heads(rows, columns, gen, kind)
        hidden = torch.arange(columns, device=DEV)[None, :] >= lengths.long()[:, None]
        scores.copy_(fresh.masked_fill(hidden, float("nan")))
        out.fill_(12345)
        graph.replay()
        torch.cuda.synchronize()
        eager = op(scores, offsets, lengths, K, **kwargs)
        assert torch.equal(out, eager), f"replay {replay} ({kind}) differs from the eager call"
        _compare(out, op.reference(scores, offsets, lengths, K, **kwargs), deterministic=True)


def test_tie_break_is_what_makes_the_set_exact() -> None:
    """The API's default tie-break, NONE, takes an arbitrary part of the tied
    group at the cut; only SMALL gives the Step-5 selection."""
    args = _build(op.CELLS[0].spec | dict(scores="zero_cut"), _seed("tie_break"))
    ref = op.reference(**args)
    small = op(**args)
    _compare(small, ref, deterministic=False)
    loose = op(**(args | dict(tie_break=TopKTieBreak.NONE)))
    assert bool((_ascending(loose) != ref).any(1).any()), "NONE matched SMALL on every row"


def test_negative_zero_ranks_below_positive_zero() -> None:
    """The kernel orders fp32 by bit pattern: `-0.0` is strictly below `+0.0`.

    300 ones, then 400 `-0.0`, then `+0.0`: numerically ids 300..511 complete
    the top 512 (ties to the lower id), but the op takes 212 `+0.0` ids from
    700 on. Pinned so that a FlashInfer change in either direction shows up
    here and in the entry's note.
    """
    rows, columns = 4, 1152
    scores = torch.zeros(rows, columns, device=DEV)
    scores[:, :300] = 1.0
    scores[:, 300:700] = -0.0
    lengths = torch.full((rows,), columns, dtype=torch.int32, device=DEV)
    offsets = torch.zeros(rows, dtype=torch.int32, device=DEV)
    out = _ascending(op(scores, offsets, lengths, K, tie_break=TopKTieBreak.SMALL))
    expected_by_bits = torch.cat(
        [torch.arange(300, device=DEV), torch.arange(700, 700 + K - 300, device=DEV)]
    ).to(torch.int32)
    assert torch.equal(out, expected_by_bits.expand(rows, K))
    numeric = op.reference(scores, offsets, lengths, K, tie_break=TopKTieBreak.SMALL)
    assert torch.equal(numeric[0, 300:303].cpu(), torch.tensor([300, 301, 302], dtype=torch.int32))


def _guard_case(seed: str):
    args = _build(op.CELLS[1].spec, _seed(seed))
    return args, op.reference(**args)


def test_guard_refuses_int64_lengths() -> None:
    args, ref = _guard_case("int64 lengths")
    bad = args | dict(lengths=args["lengths"].long())
    assert bool((_ascending(op(**bad)) != ref).any()), "int64 lengths happened to match"
    with pytest.raises(AssertionError, match="lengths must be int32"):
        with validating(op):
            op(**bad)


def test_guard_refuses_int64_offsets() -> None:
    args, _ = _guard_case("int64 offsets")
    rows = args["lengths"].numel()
    offsets = torch.arange(rows, dtype=torch.int32, device=DEV) * 1152
    ref = op.reference(**(args | dict(offsets=offsets)))
    bad = args | dict(offsets=offsets.long())
    assert bool((_ascending(op(**bad)) != ref).any()), "int64 offsets happened to match"
    with pytest.raises(AssertionError, match="offsets must be int32"):
        with validating(op):
            op(**bad)


@pytest.mark.parametrize("name", ["lengths", "offsets"])
def test_guard_refuses_a_short_per_row_tensor(name: str) -> None:
    """One entry short: the op reads the next word of the caller's buffer as
    the last row's value. The buffer belongs to the test, so the read stays in
    owned memory; its extra word is chosen to change the last row."""
    args, ref = _guard_case(f"short {name}")
    rows = args["lengths"].numel()
    assert int(args["lengths"][-1]) > 0, "the last row must select something"
    buffer = torch.empty(rows, dtype=torch.int32, device=DEV)
    buffer.copy_(args[name])
    buffer[-1] = 0 if name == "lengths" else 7
    bad = args | {name: buffer[: rows - 1]}
    out = _ascending(op(**bad))
    assert torch.equal(out[:-1], ref[:-1]) and not torch.equal(out[-1], ref[-1])
    with pytest.raises(AssertionError, match=f"{name} must be \\[{rows}\\]"):
        with validating(op):
            op(**bad)
