# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_expand_block_indices catalog entry.

Selections are built the way `qsa_topk_select` emits them: a random subset of
the complete visible groups, prefix-packed, then `-1`. Every gate is bitwise.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_expand_block_indices import (  # noqa: E501
    qsa_expand_block_indices as op,
)

__extra_import_path__ = ["../../../.."]
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
BLOCK_TOPK = TOPK // CR
FINAL_TOPK = TOPK + CR - 1
ENGINE_MAX_SEQ_LEN = 8192


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


def _rows(batch) -> tuple[list[int], list[int]]:
    kind = batch[0]
    if kind == "sweep":
        positions = [4096 + 13 * i for i in range(batch[1])]
    elif kind == "serving":
        return _serving_rows(batch[1], batch[2])
    else:
        positions = list(batch[1])
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
    block_topk = topk // cr
    query_positions = torch.tensor(positions, dtype=spec["position_dtype"], device=DEV)
    sequence_lengths = torch.tensor(lengths, dtype=spec["length_dtype"], device=DEV)
    blocks = _packed_selection(query_positions, cr, topk, seed)
    rows = blocks.shape[0]
    junk = torch.randint(0, 64, (rows + 37, 2 * block_topk), dtype=torch.int32, device=DEV)
    if spec["layout"] == "row_slice":
        # The engine's buffer is max_num_tokens tall; rows past this step hold
        # whatever the last step left there.
        tall = junk[:, :block_topk].clone()
        tall[:rows] = blocks
        blocks = tall[:rows]
    elif spec["layout"] == "column_slice":
        wide = junk[:rows].clone()
        wide[:, :block_topk] = blocks
        blocks = wide[:, :block_topk]
        assert not blocks.is_contiguous() and blocks.stride(1) == 1
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
    if cell.spec["batch"][0] == "serving":
        positions, lengths = args[1], args[2]
        assert bool((lengths.long() > positions.long() + 1).any()), (
            "a serving cell must drive sequence_lengths past position + 1"
        )


def _expand(blocks, positions, lengths, *, compress_ratio=CR, token_topk=TOPK):
    """One call, gated on the entry's own reference, returned on the host."""
    args = (
        blocks.to(DEV),
        torch.as_tensor(positions, dtype=torch.int64).to(DEV),
        torch.as_tensor(lengths, dtype=torch.int32).to(DEV),
    )
    geometry = dict(compress_ratio=compress_ratio, token_topk=token_topk)
    out = op(*args, **geometry)
    _compare(out, op.reference(*args, **geometry))
    return out.cpu()


def test_worked_example_small_geometry() -> None:
    """Two rows written out by hand, end to end: the reference agrees too."""
    # Row 0: groups 1 and 0 -> tokens 4..7 then 0..3; position 9 leaves {8, 9}.
    # Row 1: group 0 -> 0..3; position 5 leaves {4, 5}; the rest is padding.
    blocks = torch.tensor([[1, 0], [0, -1]], dtype=torch.int32)
    out = _expand(blocks, [9, 5], [10, 6], compress_ratio=4, token_topk=8)
    assert out.tolist() == [
        [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, -1],
        [0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1],
    ]


def test_launch_shapes_agree_on_shared_rows() -> None:
    """The same rows expand identically either side of the 128-row switch."""
    positions = torch.tensor([4096 + 13 * i for i in range(128)], device=DEV)
    blocks = _packed_selection(positions, CR, TOPK, seed=7)
    whole_row = _expand(blocks, positions, positions + 1)
    column_split = _expand(blocks[:127], positions[:127], positions[:127] + 1)
    assert torch.equal(whole_row[:127], column_split)


@pytest.mark.parametrize(
    "position, expected_tail",
    [(4095, 0), (4096, 1), (4097, 2), (4098, 3), (4099, 0), (7, 0), (3, 0), (0, 1)],
)
def test_tail_width_tracks_the_query_position(position: int, expected_tail: int) -> None:
    """`(position + 1) % cr` tokens are appended right after the selection, ascending."""
    blocks = _packed_selection(torch.tensor([position], device=DEV), CR, TOPK, seed=position + 1)
    out = _expand(blocks, [position], [position + 1])
    selected = int((blocks[0] >= 0).sum()) * CR
    assert int((out[0] >= 0).sum()) == selected + expected_tail
    tail_start = (position + 1) // CR * CR
    assert out[0, selected : selected + expected_tail].tolist() == list(
        range(tail_start, tail_start + expected_tail)
    )
    assert bool((out[0, selected + expected_tail :] == -1).all())


def test_selected_groups_are_bounded_by_sequence_length_not_query_position() -> None:
    """The causal bound on the selection is the producer's, not this call's.

    Both inputs are illegal for a QSA producer; they pin what the call does.
    """
    # Group 8 is past position 7 but inside length 64: expanded verbatim.
    out = _expand(torch.tensor([[8, 0]], dtype=torch.int32), [7], [64], token_topk=8)
    assert out.tolist() == [[32, 33, 34, 35, 0, 1, 2, 3, -1, -1, -1]]
    # Group 2 is incomplete at length 10: -1 holes in place, then its real
    # tokens repeated in the tail.
    out = _expand(torch.tensor([[0, 1, 2, -1]], dtype=torch.int32), [9], [10], token_topk=16)
    assert out.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -1, 8, 9, -1, -1, -1, -1, -1]]


def test_unpacked_prefix_is_silently_dropped() -> None:
    """A -1 before a valid group loses that group, with no error.

    Prefix packing is a value property, so no metadata guard can check it.
    """
    blocks = torch.tensor([[-1, 3]], dtype=torch.int32, device=DEV)
    positions = torch.tensor([15], device=DEV)
    lengths = torch.tensor([16], dtype=torch.int32, device=DEV)
    out = op(blocks, positions, lengths, compress_ratio=4, token_topk=8)
    # Group 3 is tokens 12..15; every one of them is gone.
    assert out.cpu().tolist() == [[-1] * 11]


def test_cuda_graph_capture_and_replay() -> None:
    """Capturable, and a replay recomputes from whatever the buffers now hold."""
    positions = torch.tensor([4096 + 7 * i for i in range(4)], device=DEV)
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


def test_output_is_fresh_and_inputs_are_untouched() -> None:
    """Every column is written, two calls do not alias, and nothing is in place."""
    positions = torch.tensor([4096, 97, 4095], device=DEV)
    lengths = (positions + 1).to(torch.int32)
    blocks = _packed_selection(positions, CR, TOPK, seed=41)
    before = (blocks.clone(), positions.clone(), lengths.clone())
    outputs = [op(blocks, positions, lengths, compress_ratio=CR, token_topk=TOPK) for _ in range(5)]
    assert len({out.data_ptr() for out in outputs}) == len(outputs)
    # The output comes from torch.empty: a skipped column would differ between
    # calls or land outside the legal range.
    for out in outputs[1:]:
        assert torch.equal(outputs[0], out)
    first = outputs[0]
    assert bool(((first == -1) | ((first >= 0) & (first < int(lengths.max())))).all())
    for tensor, saved in zip((blocks, positions, lengths), before):
        assert torch.equal(tensor, saved)


def test_op_rejects_malformed_arguments() -> None:
    """Every metadata violation is the callable's own loud error: why is_valid is empty."""
    positions = torch.tensor([4096, 4097], device=DEV)
    lengths = torch.tensor([4097, 4098], dtype=torch.int32, device=DEV)
    blocks = _packed_selection(positions, CR, TOPK, seed=17)
    geometry = dict(compress_ratio=CR, token_topk=TOPK)
    cases = [
        (r"must have shape \[rows, block_topk\]", (blocks.reshape(-1), positions, lengths), {}),
        (r"must have shape \[rows, block_topk\]", (blocks.unsqueeze(0), positions, lengths), {}),
        ("divisible by compress_ratio >= 2", (blocks, positions, lengths), dict(compress_ratio=1)),
        ("divisible by compress_ratio >= 2", (blocks, positions, lengths), dict(token_topk=0)),
        ("divisible by compress_ratio >= 2", (blocks, positions, lengths), dict(token_topk=2049)),
        ("width does not match", (blocks, positions, lengths), dict(token_topk=1024)),
        ("one entry per row", (blocks, torch.zeros(3, dtype=torch.int64, device=DEV), lengths), {}),
        ("must use int32 storage", (blocks.long(), positions, lengths), {}),
        ("must use integer storage", (blocks, positions.to(torch.int16), lengths), {}),
        ("must use integer storage", (blocks, positions, lengths.float()), {}),
        (
            "contiguous along their last dimension",
            (
                torch.zeros((2, 2 * BLOCK_TOPK), dtype=torch.int32, device=DEV)[:, ::2],
                positions,
                lengths,
            ),
            {},
        ),
        (
            "metadata must be contiguous",
            (blocks, torch.zeros(4, dtype=torch.int64, device=DEV)[::2], lengths),
            {},
        ),
        ("requires CUDA tensors", (blocks.cpu(), positions, lengths), {}),
        ("requires CUDA tensors", (blocks, positions.cpu(), lengths), {}),
    ]
    for message, args, override in cases:
        with pytest.raises(ValueError, match=message):
            op(*args, **{**geometry, **override})


def test_discriminates_plausible_wrong_variants() -> None:
    """The bitwise gate separates this layout from the ones a rewrite would produce."""
    positions = [4096, 97, 4095]
    blocks = _packed_selection(torch.tensor(positions, device=DEV), CR, TOPK, seed=23).cpu()
    out = _expand(blocks, positions, [p + 1 for p in positions])
    first_free = (blocks >= 0).sum(dim=1) * CR

    dropped_tail = out.clone()
    tail_at_fixed_column = out.clone()
    for row, position in enumerate(positions):
        dropped_tail[row, first_free[row] :] = -1
        tail_at_fixed_column[row, first_free[row] : TOPK] = -1
        tail_start = (position + 1) // CR * CR
        for offset in range(min(position + 1 - tail_start, CR - 1)):
            tail_at_fixed_column[row, TOPK + offset] = tail_start + offset
    assert not torch.equal(out, dropped_tail)
    assert not torch.equal(out, tail_at_fixed_column)
    assert not torch.equal(out, torch.sort(out, dim=1).values)
