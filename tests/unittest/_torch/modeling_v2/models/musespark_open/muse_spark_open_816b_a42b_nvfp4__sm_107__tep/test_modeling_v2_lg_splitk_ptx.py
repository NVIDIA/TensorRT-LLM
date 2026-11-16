# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of ``ops/gemm/lg_splitk_ptx`` (the S3 / S3b port's address map and PTX text).

The module is pure (no torch, CUDA or CuTe DSL), loaded here by file path:

* every chunk of a k-tile is polled exactly once across the polling warps' lanes, for each served
  token-tile width (8 / 16 / 32 rows) and polling-warp count;
* each chunk's shared offset is the CuTe K-major 128-byte-swizzle image of its logical offset
  (8-row x 64-column atoms tiled over the rows first, then over K: bits 4..6 XOR bits 7..9), and
  the offsets of a k-tile are a permutation of its 16-byte slots; each global offset is its row and
  column;
* the polling group (k-tiles per round) is the largest power of two within ``MAX_POLL_CHUNKS`` and
  the ring;
* the PTX statements declare as many operands as their constraint strings, and the poll's
  arrival test (per 16-bit half) takes exactly the chunks without a sentinel half.
"""

import importlib.util
import re
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parents[7]
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    / "muse_spark_open_816b_a42b_nvfp4__sm_107__tep/ops/gemm/lg_splitk_ptx.py"
)


def _ptx():
    spec = importlib.util.spec_from_file_location("_lg_splitk_ptx_under_test", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CASES = [(8, 1), (8, 2), (8, 4), (16, 1), (16, 2), (16, 4), (32, 1), (32, 2), (32, 4)]


@pytest.mark.parametrize("cta_n,poll_warps", CASES)
def test_chunks_cover_the_tile_once(cta_n, poll_warps):
    ptx = _ptx()
    per_lane = ptx.chunks_per_lane(cta_n, poll_warps)
    seen = [
        ptx.chunk_index(lane, p, poll_warps, j)
        for p in range(poll_warps)
        for lane in range(ptx.LANES)
        for j in range(per_lane)
    ]
    assert sorted(seen) == list(range(cta_n * ptx.CHUNKS_PER_ROW))


@pytest.mark.parametrize("cta_n", [8, 16, 32])
def test_shared_offsets_are_the_swizzle_image(cta_n):
    ptx = _ptx()
    row_bytes = 8192 * 2
    shared = []
    for chunk in range(cta_n * ptx.CHUNKS_PER_ROW):
        g, s = ptx.b_chunk_offsets(chunk, row_bytes, cta_n)
        row, column_chunk = chunk >> 4, chunk & 15
        assert g == row * row_bytes + column_chunk * 16
        logical = (column_chunk >> 3) * cta_n * 128 + row * 128 + (column_chunk & 7) * 16
        assert s == logical ^ ((logical >> 3) & 0x70)
        shared.append(s)
    assert sorted(shared) == list(range(0, cta_n * 256, 16))


def test_poll_groups():
    ptx = _ptx()
    assert ptx.poll_group(8, 12, 1) == 4
    assert ptx.poll_group(16, 10, 1) == 2
    assert ptx.poll_group(32, 9, 1) == 1
    assert ptx.poll_group(32, 9, 2) == 2
    assert ptx.poll_group(32, 9, 4) == 4
    assert ptx.poll_group(16, 11, 4) == 8
    assert ptx.poll_group(8, 3, 4) == 2
    with pytest.raises(ValueError):
        ptx.chunks_per_lane(8, 8)
    with pytest.raises(ValueError):
        ptx.chunks_per_lane(12, 1)


def _operands(text: str) -> int:
    return max(int(m) for m in re.findall(r"\$(\d+)", text)) + 1


@pytest.mark.parametrize("chunks", [1, 4, 8, 16])
def test_statement_operands(chunks):
    ptx = _ptx()
    assert _operands(ptx.poll_tile_ptx(chunks)) == len(ptx.poll_tile_constraints(chunks).split(","))
    assert _operands(ptx.copy_chunks_ptx(chunks)) == len(
        ptx.copy_chunks_constraints(chunks).split(",")
    )
    with pytest.raises(ValueError):
        ptx.poll_tile_ptx(ptx.MAX_POLL_CHUNKS + 1)


def _stale(words) -> bool:
    """The poll's arrival test on 32-bit words, evaluated as its PTX does."""
    m = 0xFFFFFFFF
    for w in words:
        x = w ^ ptx_sentinel()
        m = min(m >> 16, x >> 16) << 16 | min(m & 0xFFFF, x & 0xFFFF)
    t = ((m - 0x00010001) & 0xFFFFFFFF) & (~m & 0xFFFFFFFF) & 0x80008000
    return t != 0


def ptx_sentinel() -> int:
    return _ptx().SENTINEL_WORD


@pytest.mark.parametrize(
    "words,stale",
    [
        ([0x3F803F80, 0x00000000], False),
        ([0xFFFEFFFE], True),
        ([0x1234FFFE], True),
        ([0xFFFE1234], True),
        ([0xFFFFFFFF, 0x7FC07FC0], False),
        ([0xFFFDFFFF, 0x0001FFFE], True),
    ],
)
def test_arrival_test_is_per_half(words, stale):
    assert _stale(words) == stale
