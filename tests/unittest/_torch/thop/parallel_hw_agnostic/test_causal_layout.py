# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""trtllm::causal_layout_ on a geometry small enough to lay out by hand. The cache's
own tests cover it on real geometries against the host model."""

import pytest
import torch

import tensorrt_llm  # noqa: F401  # loads the custom ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")

TPB = 4
TABLE = [10, 11, 12, 13, 14, 15]  # view page of each logical page
REGION = [20, 21, 22, 23]  # one block's private pages
RPP = 2 * 1 * TPB  # pool rows per view page with one K/V head
KV_FACTOR, KV_OFFSET = 2, 1


def i64(values):
    return torch.tensor(values, dtype=torch.int64, device=DEV)


def i32(values):
    return torch.tensor(values, dtype=torch.int32, device=DEV)


def layout(block_size=4, drop_pages=0, past=6, fixed=2, window=3, shared=True):
    """One block of ``block_size`` staged at ``past``; returns the written tensors."""
    n, row_len = 1, len(TABLE)
    out = dict(
        rows=torch.full((n, row_len), -7, dtype=torch.int32, device=DEV),
        block_offsets=torch.full((1, n, 2, row_len), -7, dtype=torch.int32, device=DEV),
        seq_len_kv=torch.zeros(n, dtype=torch.int32, device=DEV),
        own_slots=torch.zeros(n * block_size, dtype=torch.int64, device=DEV),
        extra_src=torch.zeros(n * 2 * (TPB - 1), dtype=torch.int64, device=DEV),
        extra_dst=torch.zeros(n * 2 * (TPB - 1), dtype=torch.int64, device=DEV),
        piece_src=torch.zeros(n * 3 * (TPB - 1), dtype=torch.int64, device=DEV),
        piece_dst=torch.zeros(n * 3 * (TPB - 1), dtype=torch.int64, device=DEV),
    )
    extra = dict(
        staged_slots=torch.zeros(block_size, dtype=torch.int64, device=DEV),
        refill_src=torch.zeros(TPB, dtype=torch.int64, device=DEV),
        refill_dst=torch.zeros(TPB, dtype=torch.int64, device=DEV),
    )
    torch.ops.trtllm.causal_layout_(
        i32(TABLE),
        i32([REGION]),
        *out.values(),
        *(extra.values() if shared else (None, None, None)),
        block_size,
        TPB,
        past,
        fixed,
        window,
        len(TABLE),
        drop_pages,
        KV_FACTOR,
        KV_OFFSET,
        RPP,
    )
    out.update(extra)
    return {k: v.tolist() for k, v in out.items()}


def test_hand_layout():
    # past 6, fixed 2, window 3: block 0 starts at 6, sees [0, 2) and [3, 6). No whole
    # page; partial pages 0 (positions 0, 1, 3) and 1 (positions 4, 5); all resident.
    got = layout()
    assert got["rows"] == [[20, 21, 22, 23, 0, 0]]
    assert got["block_offsets"] == [[[[40, 42, 44, 46, 0, 0], [41, 43, 45, 47, 1, 1]]]]
    assert got["seq_len_kv"] == [0 * TPB + 5 + 4]
    # Own tokens take region slots 5..8: page 21 slots 1-3, then page 22 slot 0.
    assert got["own_slots"] == [21 * TPB + 1, 21 * TPB + 2, 21 * TPB + 3, 22 * TPB + 0]
    # Pieces: positions 0, 1, 3 (page 10) and 4, 5 (page 11) into region slots 0..4.
    assert got["piece_src"] == [10 * RPP + 0, 10 * RPP + 1, 10 * RPP + 3, 11 * RPP + 0, 11 * RPP + 1] + [-1] * 4
    assert got["piece_dst"] == [20 * RPP + 0, 20 * RPP + 1, 20 * RPP + 2, 20 * RPP + 3, 21 * RPP + 0] + [-1] * 4
    # No staged token on a partial page: every extra is the padding (own first token).
    assert got["extra_src"] == [0] * 6
    assert got["extra_dst"] == [21 * TPB + 1] * 6
    # Staged tokens 6..9 sit on logical pages 1 and 2.
    assert got["staged_slots"] == [11 * TPB + 2, 11 * TPB + 3, 12 * TPB + 0, 12 * TPB + 1]
    assert got["refill_src"] == [-1] * TPB and got["refill_dst"] == [-1] * TPB


def test_rotation_reads_the_fixed_tail_from_the_moved_page():
    # One page dropped: the fixed tail (positions 0, 1) still sits on the old head page,
    # now at logical page 5 (view 15), and is refilled into the new head page (view 10).
    got = layout(drop_pages=1)
    assert got["refill_src"] == [15 * RPP + 0, 15 * RPP + 1, -1, -1]
    assert got["refill_dst"] == [10 * RPP + 0, 10 * RPP + 1, -1, -1]
    assert got["piece_src"][:2] == [15 * RPP + 0, 15 * RPP + 1]  # the fixed tail, from the old page
    assert got["piece_src"][2:5] == [10 * RPP + 3, 11 * RPP + 0, 11 * RPP + 1]  # history, as mapped now


def test_staged_tokens_on_a_partial_page_become_extras():
    # Two blocks of 2: block 1 starts at 8 and sees [0, 2) and [5, 8); positions 6, 7 are
    # block 0's staged tokens on block 1's start page (page 1 holds 4..7).
    n, block_size, row_len = 2, 2, len(TABLE)
    regions = i32([REGION, [30, 31, 32, 33]])
    out = dict(
        rows=torch.zeros(n, row_len, dtype=torch.int32, device=DEV),
        block_offsets=torch.zeros(1, n, 2, row_len, dtype=torch.int32, device=DEV),
        seq_len_kv=torch.zeros(n, dtype=torch.int32, device=DEV),
        own_slots=torch.zeros(n * block_size, dtype=torch.int64, device=DEV),
        extra_src=torch.zeros(n * 2 * (TPB - 1), dtype=torch.int64, device=DEV),
        extra_dst=torch.zeros(n * 2 * (TPB - 1), dtype=torch.int64, device=DEV),
        piece_src=torch.zeros(n * 3 * (TPB - 1), dtype=torch.int64, device=DEV),
        piece_dst=torch.zeros(n * 3 * (TPB - 1), dtype=torch.int64, device=DEV),
    )
    torch.ops.trtllm.causal_layout_(
        i32(TABLE), regions, *out.values(), None, None, None, block_size, TPB, 6, 2, 3,
        len(TABLE), 0, KV_FACTOR, KV_OFFSET, RPP,
    )
    got = {k: v.tolist() for k, v in out.items()}
    # Block 1: partial positions 0, 1 (page 0), 5 (page 1, resident), 6, 7 (page 1, staged).
    assert got["seq_len_kv"][1] == 5 + block_size
    assert got["piece_src"][9:12] == [10 * RPP + 0, 10 * RPP + 1, 11 * RPP + 1]
    assert got["piece_src"][12:18] == [-1] * 6
    extra_src, extra_dst = got["extra_src"][6:], got["extra_dst"][6:]
    assert extra_src[:2] == [0, 1]  # staged tokens 6 - past, 7 - past
    assert extra_dst[:2] == [30 * TPB + 3, 31 * TPB + 0]  # region slots 3 and 4
    assert extra_src[2:] == [block_size * 1] * 4 and extra_dst[2:] == [31 * TPB + 1] * 4


def test_bad_arguments():
    with pytest.raises(RuntimeError, match="fixed_tokens <= past"):
        layout(fixed=7)
    with pytest.raises(RuntimeError, match="drop_pages < num_pages"):
        layout(drop_pages=6)
    with pytest.raises(RuntimeError, match="end past page"):
        layout(past=22)
    with pytest.raises(RuntimeError, match="cannot hold"):
        layout(block_size=8)
    with pytest.raises(RuntimeError, match="go together"):
        n, row_len = 1, len(TABLE)
        z = lambda *shape, dtype=torch.int64: torch.zeros(*shape, dtype=dtype, device=DEV)  # noqa: E731
        torch.ops.trtllm.causal_layout_(
            i32(TABLE), i32([REGION]), z(n, row_len, dtype=torch.int32),
            z(1, n, 2, row_len, dtype=torch.int32), z(n, dtype=torch.int32), z(4), z(6), z(6),
            z(9), z(9), None, z(TPB), None, 4, TPB, 6, 2, 3, len(TABLE), 0, KV_FACTOR, KV_OFFSET, RPP,
        )
