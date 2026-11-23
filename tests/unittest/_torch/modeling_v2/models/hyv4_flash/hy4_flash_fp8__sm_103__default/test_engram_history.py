# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Engram n-gram lookback across steps: the checkpoint's n-gram or pad, never another
request's token."""

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.modeling import (  # noqa: E501
    EngramHistory,
    _ngram_tokens,
)

assert torch.cuda.is_available(), "the Engram history is a device table"

MAX_NGRAM = 3
PAD = 120_000


class _Core:
    def __init__(self, max_num_tokens: int) -> None:
        self.dims = SimpleNamespace(engram_max_ngram=MAX_NGRAM, max_num_tokens=max_num_tokens)


class _Md:
    """Each sequence's encoded first block and runtime KV length (0 for a graph padding row)."""

    def __init__(self, blocks, kv_lens) -> None:
        table = torch.zeros((1, len(blocks), 2, 1), dtype=torch.int32, device="cuda")
        table[0, :, 0, 0] = torch.tensor(blocks, dtype=torch.int32, device="cuda")
        self.kv_cache_block_offsets = table
        self.kv_lens_cuda_runtime = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")


def _step(history, md, tokens, positions, rows):
    """One forward's columns, merged as `Hyv4FlashCore.forward` merges them."""
    tokens = torch.tensor(tokens, dtype=torch.int32, device="cuda")
    positions = torch.tensor(positions, dtype=torch.int32, device="cuda")
    rows = torch.tensor(rows, dtype=torch.int64, device="cuda")
    columns = _ngram_tokens(tokens, positions, rows, MAX_NGRAM, PAD)
    lookback = history.lookback(md, tokens, positions, rows, PAD)
    if lookback is not None:
        columns[:, 1:] = torch.where(columns[:, 1:] == PAD, lookback, columns[:, 1:])
    return columns.tolist()


def test_decode_reads_the_prompt_tail():
    history = EngramHistory(_Core(max_num_tokens=64))
    _step(history, _Md([0], [5]), [11, 12, 13, 14, 15], [0, 1, 2, 3, 4], [0] * 5)
    assert _step(history, _Md([0], [6]), [16], [5], [0]) == [[16, 15, 14]]
    assert _step(history, _Md([0], [7]), [17], [6], [0]) == [[17, 16, 15]]


def test_chunk_boundary_reads_the_previous_chunk():
    history = EngramHistory(_Core(max_num_tokens=64))
    _step(history, _Md([0], [5]), [11, 12, 13, 14, 15], [0, 1, 2, 3, 4], [0] * 5)
    assert _step(history, _Md([0], [10]), [16, 17, 18, 19, 20], [5, 6, 7, 8, 9], [0] * 5) == [
        [16, 15, 14],
        [17, 16, 15],
        [18, 17, 16],
        [19, 18, 17],
        [20, 19, 18],
    ]


def test_recycled_block_starts_empty():
    history = EngramHistory(_Core(max_num_tokens=64))
    _step(history, _Md([0], [5]), [11, 12, 13, 14, 15], [0, 1, 2, 3, 4], [0] * 5)
    # The same block now holds a new request's prompt.
    assert _step(history, _Md([0], [3]), [31, 32, 33], [0, 1, 2], [0] * 3) == [
        [31, PAD, PAD],
        [32, 31, PAD],
        [33, 32, 31],
    ]
    assert _step(history, _Md([0], [4]), [34], [3], [0]) == [[34, 33, 32]]


def test_padding_row_writes_nothing():
    history = EngramHistory(_Core(max_num_tokens=64))
    _step(history, _Md([0], [3]), [11, 12, 13], [0, 1, 2], [0] * 3)
    md = _Md([0, -1], [4, 0])
    assert _step(history, md, [14, 999], [3, 0], [0, 1]) == [
        [14, 13, 12],
        [999, PAD, PAD],
    ]
    assert _step(history, _Md([0], [5]), [15], [4], [0]) == [[15, 14, 13]]


def test_colliding_keys_read_pad():
    history = EngramHistory(_Core(max_num_tokens=64))
    first, second = 42, 42 + EngramHistory.ROWS
    _step(history, _Md([first], [3]), [11, 12, 13], [0, 1, 2], [0] * 3)
    columns = _step(history, _Md([first, second], [4, 2]), [14, 21], [3, 1], [0, 1])
    assert columns[1][1:] == [PAD, PAD]
    assert columns[0][1:] in ([13, 12], [PAD, PAD])


def test_ring_wraps_by_position():
    width = 4
    history = EngramHistory(_Core(max_num_tokens=width))
    _step(history, _Md([0], [1]), [11], [0], [0])
    # Position `width` shares position 0's column; the stale token is not read back.
    assert _step(history, _Md([0], [width + 1]), [50], [width], [0]) == [[50, PAD, PAD]]
    assert _step(history, _Md([0], [width + 2]), [51], [width + 1], [0]) == [[51, 50, PAD]]
