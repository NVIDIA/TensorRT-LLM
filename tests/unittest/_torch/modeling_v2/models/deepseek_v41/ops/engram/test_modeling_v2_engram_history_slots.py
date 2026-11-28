# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`EngramHistorySlots`: stock's history-row ownership for the targets' look-back ring. A request keeps its row until
it is released; the ring is sized to every request the rank holds, so more than its initial 512 rows can be
scheduled in one step; a full ring takes back an absent request's row, and grows (keeping the rows) when every row
is live."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_history_slots import (
    MIN_ROWS,
    EngramHistorySlots,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="device buffers")
DEV = "cuda"


def test_rows_are_owned_until_released():
    s = EngramHistorySlots(16, -1, 64, DEV)
    assert (
        s.dummy == MIN_ROWS
        and s.hist.shape == (MIN_ROWS + 1, 16)
        and int(s.req_slot[0]) == MIN_ROWS
    )
    a, new_a = s.acquire(7, {7})
    assert new_a and s.acquire(7, {7}) == (a, False)
    b, _ = s.acquire(8, {8})  # 7 is not scheduled: it keeps its row
    assert b != a and s.row_of(7) == a
    s.release(7)
    c, new_c = s.acquire(9, {9})
    assert new_c and c == a and s.row_of(7) is None


@pytest.mark.parametrize("capacity, live", [(640, 600), (4096, 4096)])
def test_one_step_of_more_requests_than_512(capacity, live):
    s = EngramHistorySlots(16, -1, 8192, DEV)
    s.reserve(capacity)
    ids = list(range(10_000, 10_000 + live))
    rows = [s.acquire(i, set(ids))[0] for i in ids]
    assert len(set(rows)) == live and max(rows) < s.dummy and s.dummy >= capacity


def test_a_full_ring_takes_back_an_absent_request_then_grows_keeping_the_rows():
    s = EngramHistorySlots(4, -1, 16, DEV, rows=2)
    s.acquire(1, {1, 2})
    s.acquire(2, {1, 2})
    s.hist[s.row_of(2)] = 22
    r3, _ = s.acquire(3, {2, 3})  # 1 is absent and was least recently scheduled
    assert s.row_of(1) is None and r3 != s.row_of(2)
    s.req_slot[:3] = torch.tensor([s.row_of(2), r3, s.dummy], dtype=torch.int32, device=DEV)
    r4, _ = s.acquire(4, {2, 3, 4})  # every row live: the ring doubles
    assert s.dummy == 4 and s.hist.shape[0] == 5 and r4 not in (s.row_of(2), r3)
    assert (
        s.hist[s.row_of(2)].tolist() == [22] * 4 and int(s.req_slot[2]) == s.dummy
    )  # rows kept, padding re-aimed
