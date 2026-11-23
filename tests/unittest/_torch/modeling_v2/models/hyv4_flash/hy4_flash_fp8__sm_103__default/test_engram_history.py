# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Engram n-gram lookback across forward boundaries.

HF's `build_ngram_tokens` reads the `max_ngram - 1` tokens before each position
out of the whole sequence, so a decode step's n-gram is `(tok, prev, prev2)` with
the prompt and generation history behind it. A step's `input_ids` carry only the
step's tokens, so without `EngramHistory` a decode row's n-gram degenerates to
`(tok, pad, pad)` -- a memory the checkpoint never chose -- and a chunked
prefill's first two rows do too.

These tests drive the entry the forward drives (`lookback` merged over
`_ngram_tokens`) with the metadata it reads: the encoded block table, the
runtime KV lengths and the per-token request rows. They are the handoff the
boot's generated text depends on: context -> decode, chunk -> chunk, a recycled
KV block, a padding row and a key collision, each of which must give the
checkpoint's n-gram or `pad`, never another request's token.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.modeling import (  # noqa: E501
    EngramHistory,
    _ngram_tokens,
)

assert torch.cuda.is_available(), "the Engram history is a device table"

#: The checkpoint's n-gram order (bigram + trigram) and the pad id it hashes
#: when a position has no predecessor. Both are config values; the entry reads
#: them from `dims`, so a fixed pair here is the contract, not the model's.
MAX_NGRAM = 3
PAD = 120_000


class _Core:
    """The two `dims` fields the history reads."""

    def __init__(self, max_num_tokens: int) -> None:
        self.dims = SimpleNamespace(engram_max_ngram=MAX_NGRAM, max_num_tokens=max_num_tokens)


class _Md:
    """The attention metadata fields the history reads.

    ``blocks`` is each sequence's encoded first block (`kv_cache_block_offsets`
    is the C++ encoded table, so a real request's first entry is unique and
    stable) and ``kv_lens`` its runtime KV length; a padding row of a captured
    decode graph has a placeholder block and a zero length.
    """

    def __init__(self, blocks, kv_lens) -> None:
        table = torch.zeros((1, len(blocks), 2, 1), dtype=torch.int32, device="cuda")
        table[0, :, 0, 0] = torch.tensor(blocks, dtype=torch.int32, device="cuda")
        self.kv_cache_block_offsets = table
        self.kv_lens_cuda_runtime = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")


def _step(history, core, md, tokens, positions, rows):
    """One forward's columns: `_ngram_tokens` with the history filling its pads."""
    tokens = torch.tensor(tokens, dtype=torch.int32, device="cuda")
    positions = torch.tensor(positions, dtype=torch.int32, device="cuda")
    rows = torch.tensor(rows, dtype=torch.int64, device="cuda")
    columns = _ngram_tokens(tokens, positions, rows, MAX_NGRAM, PAD)
    lookback = history.lookback(md, tokens, positions, rows, PAD)
    if lookback is not None:
        columns[:, 1:] = torch.where(columns[:, 1:] == PAD, lookback, columns[:, 1:])
    return columns.tolist()


def test_decode_reads_the_prompt_tail():
    """The first generation step hashes the prompt's last two tokens, not pads."""
    core = _Core(max_num_tokens=64)
    history = EngramHistory(core)
    # The context step: five prompt tokens of request 0, whose cache starts at
    # block 0, written at positions 0..4.
    _step(history, core, _Md([0], [5]), [11, 12, 13, 14, 15], [0, 1, 2, 3, 4], [0] * 5)
    # First generation step: one token at position 5.
    assert _step(history, core, _Md([0], [6]), [16], [5], [0]) == [[16, 15, 14]]
    # Second: the token before it is the first generated one.
    assert _step(history, core, _Md([0], [7]), [17], [6], [0]) == [[17, 16, 15]]


def test_chunk_boundary_reads_the_previous_chunk():
    """A chunk's first rows take their predecessors from the cache, the rest in-step."""
    core = _Core(max_num_tokens=64)
    history = EngramHistory(core)
    _step(history, core, _Md([0], [5]), [11, 12, 13, 14, 15], [0, 1, 2, 3, 4], [0] * 5)
    # Chunk two of the same request: positions 5..9. Row 5 has no predecessor in
    # this forward; rows 6 and 7 have theirs in-step and must keep them.
    assert _step(history, core, _Md([0], [10]), [16, 17, 18, 19, 20], [5, 6, 7, 8, 9], [0] * 5) == [
        [16, 15, 14],
        [17, 16, 15],
        [18, 17, 16],
        [19, 18, 17],
        [20, 19, 18],
    ]


def test_recycled_block_starts_empty():
    """A KV block handed to a new request carries none of the old request's tokens."""
    core = _Core(max_num_tokens=64)
    history = EngramHistory(core)
    _step(history, core, _Md([0], [5]), [11, 12, 13, 14, 15], [0, 1, 2, 3, 4], [0] * 5)
    # The same block now holds a new request's three-token prompt at 0..2: its
    # first rows have no predecessor and must not see request A's tail.
    assert _step(history, core, _Md([0], [3]), [31, 32, 33], [0, 1, 2], [0] * 3) == [
        [31, PAD, PAD],
        [32, 31, PAD],
        [33, 32, 31],
    ]
    # And its first generation step reads its own prompt.
    assert _step(history, core, _Md([0], [4]), [34], [3], [0]) == [[34, 33, 32]]


def test_padding_row_writes_nothing():
    """A captured graph's padding row reads pads and leaves a live request alone."""
    core = _Core(max_num_tokens=64)
    history = EngramHistory(core)
    _step(history, core, _Md([0], [3]), [11, 12, 13], [0, 1, 2], [0] * 3)
    # One live generation row and one padding row (placeholder block, zero KV).
    md = _Md([0, -1], [4, 0])
    assert _step(history, core, md, [14, 999], [3, 0], [0, 1]) == [
        [14, 13, 12],
        [999, PAD, PAD],
    ]
    # The padding row's token did not land in block 0's row.
    assert _step(history, core, _Md([0], [5]), [15], [4], [0]) == [[15, 14, 13]]


def test_colliding_keys_read_pad():
    """Two live requests in one ring row read pads, never each other's tokens."""
    core = _Core(max_num_tokens=64)
    history = EngramHistory(core)
    # Encoded first blocks congruent modulo the ring's row count.
    first, second = 42, 42 + EngramHistory.ROWS
    _step(history, core, _Md([first], [3]), [11, 12, 13], [0, 1, 2], [0] * 3)
    md = _Md([first, second], [4, 2])
    columns = _step(history, core, md, [14, 21], [3, 1], [0, 1])
    # Whichever request the row was last written for, neither reads the other's
    # token: the row's stored key rejects the one that does not own it.
    assert columns[1][1:] == [PAD, PAD]
    assert columns[0][1:] in ([13, 12], [PAD, PAD])


@pytest.mark.parametrize("width", [4, 8, 8192])
def test_ring_wraps_by_position(width):
    """A position one ring-width back is rejected, so an old cell is never read."""
    core = _Core(max_num_tokens=width)
    history = EngramHistory(core)
    _step(history, core, _Md([0], [1]), [11], [0], [0])
    # Position `width` shares a column with position 0. Neither predecessor
    # position was ever written, and the stale token 11 in position 0's column
    # is not read back as one of them.
    assert _step(history, core, _Md([0], [width + 1]), [50], [width], [0]) == [[50, PAD, PAD]]
    # The next step reads the token just written, one back; two back is a
    # position whose column now holds a different position's cell: pad.
    assert _step(history, core, _Md([0], [width + 2]), [51], [width + 1], [0]) == [[51, 50, PAD]]
