# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the unified target's qsa_expand_block_indices variant.

Selections are built the way `qsa_topk_select` emits them: a random subset of
the complete visible groups, prefix-packed, then `-1`. Every gate is bitwise.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_expand_block_indices import (  # noqa: E501
    qsa_expand_block_indices as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_expand_block_indices requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bitwise: these are KV-cache token indices, and one wrong entry reads
    a different key. There is no rounding for a tolerance to absorb."""
    assert out.dtype == ref.dtype == torch.int32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


DEV = "cuda"
CR, TOPK = 4, 2048
# The tp1 engine's max_seq_len: where `_serving_rows` places its continuation chunks.
ENGINE_MAX_SEQ_LEN = 8192
# The unified target's longest sequence (the tune's perf configs): every cell's bound.
ENVELOPE_MAX_SEQ_LEN = 262144


def _serving_rows(requests: int, context_lens: tuple) -> tuple[list[int], list[int]]:
    """Per-row positions and lengths of one packed step.

    Chunk 0 starts at 0, so its rows span every visible-group count up to the
    budget; later chunks are chunked-prefill continuations. Each remaining
    request is an MTP generation request with two rows at consecutive positions.
    """
    positions: list[int] = []
    lengths: list[int] = []
    for chunk, length in enumerate(context_lens):
        start = 0 if chunk == 0 else (907 * chunk) % (ENGINE_MAX_SEQ_LEN - length + 1)
        positions.extend(range(start, start + length))
        lengths.extend([start + length] * length)
    for gen in range(requests - len(context_lens)):
        position = 100 + (499 * gen) % 8000
        positions.extend((position, position + 1))
        lengths.extend((position + 2, position + 2))
    assert max(lengths, default=0) <= ENGINE_MAX_SEQ_LEN
    return positions, lengths


def _decode_positions(rows: int, top: int) -> list[int]:
    """One decode row per request, spread over [1, top): the first row sees one group,
    the last is near `top`, and `i % 4` cycles the tail width through 0..3."""
    step = (top - 4) // (rows - 1) // CR * CR
    positions = [1 + i * step + i % CR for i in range(rows)]
    assert positions[-1] < top
    return positions


def _rows(batch) -> tuple[list[int], list[int]]:
    kind = batch[0]
    if kind == "sweep":
        positions = [4096 + 13 * i for i in range(batch[1])]
    elif kind == "decode":
        positions = _decode_positions(batch[1], batch[2])
        assert max(positions) + 1 <= ENVELOPE_MAX_SEQ_LEN
    elif kind == "serving":
        return _serving_rows(batch[1], batch[2])
    elif kind == "chunk":
        start, rows = batch[1], batch[2]
        assert start + rows <= ENVELOPE_MAX_SEQ_LEN
        return list(range(start, start + rows)), [start + rows] * rows
    else:
        assert kind == "verify", kind
        starts, per_request = batch[1], batch[2]
        positions = [s + i for s in starts for i in range(per_request)]
        lengths = [s + per_request for s in starts for _ in range(per_request)]
        assert max(lengths) <= ENVELOPE_MAX_SEQ_LEN
        return positions, lengths
    return positions, [p + 1 for p in positions]


def _packed_selection(positions: torch.Tensor, compress_ratio: int, token_topk: int, seed: int):
    """A legal Top-K result: distinct complete visible groups, then -1 padding."""
    block_topk = token_topk // compress_ratio
    rows = positions.numel()
    if rows == 0:
        return torch.empty((0, block_topk), dtype=torch.int32, device=DEV)
    available = (positions.long() + 1) // compress_ratio
    width = max(block_topk, int(available.max()))
    gen = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.rand(rows, width, generator=gen, device=DEV)
    groups = torch.arange(width, device=DEV)
    scores = torch.where(groups[None, :] < available[:, None], scores, torch.inf)
    chosen = scores.argsort(dim=1)[:, :block_topk]
    taken = available.clamp(max=block_topk)
    chosen = torch.where(torch.arange(block_topk, device=DEV)[None, :] < taken[:, None], chosen, -1)
    return chosen.to(torch.int32)


def _build(spec, seed: int):
    positions, lengths = _rows(spec["batch"])
    cr, topk = spec["compress_ratio"], spec["token_topk"]
    query_positions = torch.tensor(positions, dtype=spec["position_dtype"], device=DEV)
    sequence_lengths = torch.tensor(lengths, dtype=spec["length_dtype"], device=DEV)
    blocks = _packed_selection(query_positions, cr, topk, seed)
    return (blocks, query_positions, sequence_lengths), dict(compress_ratio=cr, token_topk=topk)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args, geometry = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    out = op(*args, **geometry)
    rows = args[0].shape[0]
    final = geometry["token_topk"] + geometry["compress_ratio"] - 1
    assert out.dtype == torch.int32 and out.is_cuda and out.is_contiguous()
    assert tuple(out.shape) == (rows, final)
    _compare(out, op.reference(*args, **geometry))
    if cell.spec["batch"][0] in ("serving", "chunk"):
        positions, lengths = args[1], args[2]
        assert bool((lengths.long() > positions.long() + 1).any()), (
            "a context cell must drive sequence_lengths past position + 1"
        )


@pytest.mark.parametrize("rows", [1, 128, 256, 1024])
def test_cuda_graph_capture_and_replay(rows: int) -> None:
    """Capturable, and a replay recomputes from whatever the buffers now hold, at both
    launch shapes: one decode row (columns split across programs), the 128-request decode
    batch (the first whole-row launch), and the verify steps of 32 and 128 requests at
    max_draft_len 7 (256 and 1024 rows)."""
    positions = torch.tensor([4096 + 7 * i for i in range(rows)], device=DEV)
    lengths = (positions + 1).to(torch.int32)
    blocks = _packed_selection(positions, CR, TOPK, seed=31)

    def call():
        return op(blocks, positions, lengths, compress_ratio=CR, token_topk=TOPK)

    eager = call().clone()
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)

    blocks.copy_(_packed_selection(positions, CR, TOPK, seed=32))
    graph.replay()
    torch.cuda.synchronize()
    _compare(captured, op.reference(blocks, positions, lengths, compress_ratio=CR, token_topk=TOPK))


def test_op_rejects_a_width_that_does_not_match_the_geometry() -> None:
    """A selection buffer whose width is not `token_topk // compress_ratio` is the
    callable's own loud error, not a silent mis-expansion: why is_valid is empty."""
    positions = torch.tensor([4096, 4097], device=DEV)
    lengths = torch.tensor([4097, 4098], dtype=torch.int32, device=DEV)
    blocks = _packed_selection(positions, CR, TOPK, seed=17)
    with pytest.raises(ValueError, match="width does not match"):
        op(blocks, positions, lengths, compress_ratio=CR, token_topk=1024)
