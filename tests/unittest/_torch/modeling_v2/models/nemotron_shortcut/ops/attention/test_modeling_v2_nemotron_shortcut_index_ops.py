# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the nemotron_shortcut index entries: index_key_store, index_scores,
index_topk_rows and index_pool_rows.

A cell's `rows[b]` is the number of rows request `b` brings to the step and `past[b]` the
tokens it holds before the step: its rows sit at positions `past[b] + i` and each sees
`position + 1` keys. The paged state is real: int32 block offsets `[requests, 2, pages]`
over shuffled slots, `-1` past each request, and an index-key cache that is a view of a
wider page.

Keys and queries are small integers, exact in bf16, so a score is the same fp32 integer in
every summation order. Scores are compared bitwise, and the many equal scores at the cutoff
make the selection depend on the tie rule.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.ops.attention.index_key_store import (
    index_key_store,
)
from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.ops.attention.index_pool_rows import (
    index_pool_rows,
)
from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.ops.attention.index_scores import (
    index_scores,
)
from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.ops.attention.index_topk_rows import (
    index_topk_rows,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires the Blackwell sparse indexer",
)

DEV = "cuda"
#: Width of an index key.
DIM = 128
#: Keys a row keeps. The cells' histories sit below, at and above it.
TOPK = 512
#: A page's K offset is its slot times this; its V offset is one more.
SLOT_SCALE = 2
#: The layer's place in its pool at the cells that do not name one.
LAYER_IN_POOL = 3
#: Pre-fill for scores, above every reachable score. Negated, the pre-fill for index and
#: pool-row outputs, where it is no valid value.
SENTINEL = 12345


def _why(cell) -> str:
    return cell.why


def _generator(cell) -> torch.Generator:
    return torch.Generator(device=DEV).manual_seed(zlib.crc32(cell.why.encode()))


def _integers(shape, gen) -> torch.Tensor:
    """Small integers, exact in bf16."""
    return torch.randint(-3, 4, shape, generator=gen, device=DEV).to(torch.bfloat16)


def _block_offsets(tokens, tpb, gen) -> tuple[torch.Tensor, int]:
    """Block offsets of requests holding `tokens[b]` tokens, and the slots they index."""
    pages = [-(-n // tpb) for n in tokens]
    slots = sum(pages) + 3
    slot_of = torch.randperm(slots, generator=gen, device=DEV).to(torch.int32)
    offsets = torch.full((len(tokens), 2, max(pages)), -1, dtype=torch.int32, device=DEV)
    first = 0
    for b, count in enumerate(pages):
        offsets[b, 0, :count] = slot_of[first : first + count] * SLOT_SCALE
        offsets[b, 1, :count] = offsets[b, 0, :count] + 1
        first += count
    return offsets, slots


def _paged_keys(spec, gen):
    """One cell's step: each row's request and position, the block offsets and the key cache."""
    requests = torch.cat(
        [torch.full((n,), b, dtype=torch.int32, device=DEV) for b, n in enumerate(spec["rows"])]
    )
    positions = torch.cat(
        [
            torch.arange(past, past + n, dtype=torch.int32, device=DEV)
            for past, n in zip(spec["past"], spec["rows"])
        ]
    )
    tpb = spec["tpb"]
    tokens = [past + n for past, n in zip(spec["past"], spec["rows"])]
    offsets, slots = _block_offsets(tokens, tpb, gen)
    # The layer's index-key view of a wider page.
    page = _integers((slots, tpb * DIM + 24), gen)
    cache = page[:, : tpb * DIM].view(slots, tpb, 1, DIM)
    assert cache.stride() == (page.stride(0), DIM, DIM, 1)
    return requests, positions, offsets, cache


def _row_tiles(rows, block_m) -> tuple[torch.Tensor, torch.Tensor]:
    """First row and row count of every run of at most `block_m` rows of one request."""
    tiles = []
    first = 0
    for n in rows:
        tiles += [(first + i, min(block_m, n - i)) for i in range(0, n, block_m)]
        first += n
    grid = torch.tensor(tiles, dtype=torch.int32, device=DEV).T.contiguous()
    return grid[0], grid[1]


def _score_buffer(rows, width) -> torch.Tensor:
    """Pre-filled fp32 scores `[rows, width]` at a row stride that is a multiple of four."""
    return torch.full((rows, -(-width // 4) * 4), float(SENTINEL), device=DEV)[:, :width]


@pytest.mark.parametrize("cell", index_key_store.CELLS, ids=_why)
def test_index_key_store_cells(cell, certifying) -> None:  # noqa: F811
    certifying(index_key_store)
    gen = _generator(cell)
    requests, positions, offsets, cache = _paged_keys(cell.spec, gen)
    # The keys are a column slice of a wider projection.
    projection = torch.randn(requests.numel(), DIM + 16, generator=gen, device=DEV)
    keys = projection.to(torch.bfloat16)[:, 16:]
    args = (keys, cache, offsets[:, 0], SLOT_SCALE, requests, positions)
    expected = index_key_store.reference(*args)
    assert index_key_store(*args) is None
    # The whole plane: a stray write shows as well as a missing one.
    assert torch.equal(cache, expected)


@pytest.mark.parametrize("cell", index_scores.CELLS, ids=_why)
def test_index_scores_cells(cell, certifying) -> None:  # noqa: F811
    certifying(index_scores)
    spec = cell.spec
    gen = _generator(cell)
    requests, positions, offsets, cache = _paged_keys(spec, gen)
    lengths = positions + 1
    rows = requests.numel()
    logits = _score_buffer(rows, int(lengths.max()))
    args = (
        _integers((rows, DIM), gen),
        cache,
        offsets[:, 0],
        SLOT_SCALE,
        requests,
        lengths,
        *_row_tiles(spec["rows"], 16),
        logits,
        spec["splits"],
    )
    # The reference keeps the pre-fill at and past each row's length, as the op must.
    expected = index_scores.reference(*args)
    assert index_scores(*args) is logits
    assert torch.equal(logits, expected)


@pytest.mark.parametrize("cell", index_topk_rows.CELLS, ids=_why)
def test_index_topk_rows_cells(cell, certifying) -> None:  # noqa: F811
    certifying(index_topk_rows)
    spec = cell.spec
    gen = _generator(cell)
    requests, positions, offsets, cache = _paged_keys(spec, gen)
    lengths = positions + 1
    rows = requests.numel()
    block_m = 64 if spec.get("prefill") else 16
    args = (
        _integers((rows, DIM), gen),
        cache,
        offsets,
        SLOT_SCALE,
        LAYER_IN_POOL,
        requests,
        lengths,
        *_row_tiles(spec["rows"], block_m),
        _score_buffer(rows, int(lengths.max())),
        torch.full((rows, TOPK), -SENTINEL, dtype=torch.int32, device=DEV),
        torch.full((1, rows, 2, TOPK), -SENTINEL, dtype=torch.int32, device=DEV),
        1,
        block_m,
    )
    expected = index_topk_rows.reference(*args)
    out = index_topk_rows(*args)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("cell", index_pool_rows.CELLS, ids=_why)
def test_index_pool_rows_cells(cell, certifying) -> None:  # noqa: F811
    certifying(index_pool_rows)
    spec = cell.spec
    gen = _generator(cell)
    lens = spec["lens"]
    # Rows of consecutive lengths are one request's chunk; any other row starts a request.
    requests = [0]
    for previous, length in zip(lens, lens[1:]):
        requests.append(requests[-1] + (length != previous + 1))
    tokens = [0] * (requests[-1] + 1)
    for request, length in zip(requests, lens):
        tokens[request] = max(tokens[request], length)
    offsets, _ = _block_offsets(tokens, spec["tpb"], gen)
    # A short row keeps its whole prefix; a long one a sorted subset of it.
    indices = torch.full((len(lens), TOPK), -1, dtype=torch.int32, device=DEV)
    for row, length in enumerate(lens):
        if length <= TOPK:
            indices[row, :length] = torch.arange(length, dtype=torch.int32, device=DEV)
        else:
            kept = torch.randperm(length, generator=gen, device=DEV)[:TOPK]
            indices[row] = kept.sort().values.to(torch.int32)
    args = (
        indices,
        torch.tensor(requests, dtype=torch.int32, device=DEV),
        offsets,
        spec["layer"],
        spec["tpb"],
    )
    out = index_pool_rows(*args)
    assert torch.equal(out, index_pool_rows.reference(*args))


def test_cutoff_ties_keep_logical_order_under_graph_replay() -> None:
    """One key wins and every other key ties: the selection keeps the earliest tied
    positions in logical order, a row shorter than the top-k keeps its whole prefix, and
    a captured call replays to the same indices and pool rows."""
    width, topk, tpb = 2813, 2048, 64
    pages = -(-width // tpb)
    with torch.device(DEV):
        keys = torch.zeros((pages, tpb, 1, DIM), dtype=torch.bfloat16)
        keys.flatten(0, 1)[width - 1].fill_(1)
        query = torch.ones((2, DIM), dtype=keys.dtype)
        page = torch.arange(pages, dtype=torch.int32) * SLOT_SCALE
        offsets = torch.stack((page, page + 1)).unsqueeze(0)
        lengths = torch.tensor([width, 5], dtype=torch.int32)
        scores = torch.empty((2, width + 3), dtype=torch.float32)[:, :width]
        indices = torch.empty((2, topk), dtype=torch.int32)
        out = torch.empty((1, 2, 2, topk), dtype=torch.int32)
        row = torch.arange(2, dtype=torch.int32)
        requests = torch.zeros_like(row)
        arguments = (
            query,
            keys,
            offsets,
            SLOT_SCALE,
            0,
            requests,
            lengths,
            row,
            torch.ones_like(row),
            scores,
            indices,
            out,
        )
        index_topk_rows(*arguments)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            index_topk_rows(*arguments)
        indices.fill_(-2)
        out.fill_(-2)
        graph.replay()
        expected = torch.full_like(indices, -1)
        expected[0, :-1] = torch.arange(topk - 1, dtype=torch.int32)
        expected[0, -1] = width - 1
        expected[1, :5] = torch.arange(5, dtype=torch.int32)
        torch.testing.assert_close(indices, expected, atol=0, rtol=0)
        pool_rows = index_pool_rows.reference(expected, requests, offsets, 0, tpb)
        torch.testing.assert_close(out, pool_rows, atol=0, rtol=0)
