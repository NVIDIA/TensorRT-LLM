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
"""Dkv: page ownership, the group minimum and the per-step page-table exchange, with threads
standing in for the ranks of the group."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.dkv import Dkv

pytestmark = pytest.mark.cpu_only

TPB = 64


class _Group:
    """An all-gather over ``size`` threads: every rank deposits its object and waits for all."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._barrier = threading.Barrier(size)
        self._slots = [None] * size

    def all_gather_of(self, rank: int):
        def all_gather(obj):
            self._slots[rank] = obj
            self._barrier.wait()
            out = list(self._slots)
            self._barrier.wait()
            return out

        return all_gather


def _run(size: int, body, check: bool = False):
    """Run ``body(DKV)`` on ``size`` ranks at once; return their results in rank order."""
    group = _Group(size)
    dkvs = [Dkv(size, r, TPB, group.all_gather_of(r), check=check) for r in range(size)]
    with ThreadPoolExecutor(size) as pool:
        return list(pool.map(body, dkvs))


def test_ownership_and_local_rows():
    dkv = Dkv(8, 3, TPB, lambda obj: [obj] * 8)
    assert dkv.dkv_tokens_per_block == 512
    for position in range(0, 20_000, 7):
        owner = dkv.owner(position)
        assert owner == (position // TPB) % 8
        local = dkv.local_len(position, owner)
        # The token's row in its owner's pages: the owner's slot of its DKV block.
        assert local // TPB == position // 512 and local % TPB == position % TPB
    assert dkv.round_down(1535) == 1024 and dkv.round_down(1536) == 1536
    assert sum(dkv.local_len(5000, r) for r in range(8)) == 5000


def test_agree_takes_the_group_minimum():
    assert _run(4, lambda dkv: dkv.agree(100 + 10 * dkv.rank)) == [100] * 4


def test_agree_all_takes_each_group_minimum_in_one_all_gather():
    calls = []

    def body(dkv):
        calls.append(dkv.rank)
        # Rank 1 has no part in the second value; nobody in the third.
        second = None if dkv.rank == 1 else 50 + dkv.rank
        return dkv.agree_all([100 - dkv.rank, second, None])

    assert _run(3, body) == [[98, 50, None]] * 3
    assert sorted(calls) == [0, 1, 2]


def test_a_copy_on_any_rank_fences_every_rank_once():
    """The copy flags travel with the page tables: every rank learns that some rank copied pages
    this step, and the flag lasts until the reader takes it."""

    def body(dkv):
        dkv.exchange_pages([((7, 1), 0, [dkv.rank])], copied=dkv.rank == 2)
        dkv.exchange_pages([((7, 1), 1, [dkv.rank])])  # a later exchange keeps the flag
        first, second = dkv.take_fence(), dkv.take_fence()
        dkv.exchange_pages([((7, 1), 2, [dkv.rank])])
        return first, second, dkv.take_fence()

    assert _run(4, body) == [(True, False, False)] * 4


def test_every_rank_learns_every_rank_s_slots():
    """Ranks allocate different slot ids for the same blocks; after an exchange each rank resolves
    every token to its owner's slot, also after an update of the tail and a rewrite."""

    def body(dkv):
        r = dkv.rank
        mine = [100 * r + b for b in range(6)]  # this rank's slot of DKV blocks 0..5
        dkv.exchange_pages([((7, 1), 0, mine[:4])])
        first = [list(t) for t in dkv.peer_pages((7, 1))]
        # Blocks 4 and 5 come one step later; block 2 moved (a rewrite from block 2 on).
        mine[2] += 50
        dkv.exchange_pages([((7, 1), 2, mine[2:])])
        position = 2 * dkv.dkv_tokens_per_block + 3 * TPB + 5  # block 2, owner 3
        return first, dkv.peer_pages((7, 1)), dkv.page_of((7, 1), position)

    results = _run(4, body)
    for first, tables, page in results:
        assert first == [[100 * r + b for b in range(4)] for r in range(4)]
        assert tables == [[100 * r + b + (50 if b == 2 else 0) for b in range(6)] for r in range(4)]
        assert page == (3, 352, 5)


def test_forget_drops_every_life_cycle_of_a_request():
    def body(dkv):
        dkv.exchange_pages([((7, 1), 0, [dkv.rank]), ((7, 2), 0, [9]), ((8, 1), 0, [1])])
        dkv.forget(7)
        return sorted(dkv._peer_pages)

    assert _run(2, body) == [[(8, 1)]] * 2


def test_check_mode_refuses_ranks_that_allocated_different_blocks():
    def body(dkv):
        blocks = 3 if dkv.rank == 0 else 2
        with pytest.raises(RuntimeError, match="different blocks"):
            dkv.exchange_pages([((7, 1), 0, list(range(blocks)))])
        return True

    assert all(_run(2, body, check=True))


def test_a_group_needs_two_ranks():
    with pytest.raises(ValueError):
        Dkv(1, 0, TPB, lambda obj: [obj])
