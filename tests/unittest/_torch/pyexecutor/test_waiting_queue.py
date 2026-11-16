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

"""Tests for WaitingQueue implementations.

This module tests the waiting queue functionality including:
- FCFSWaitingQueue operations
- PriorityWaitingQueue operations
- AgingPriorityWaitingQueue operations
- WaitingQueue abstract interface
- create_waiting_queue factory function
"""

from unittest.mock import Mock, patch

import pytest

import tensorrt_llm.llmapi.llm_args as llm_args_mod
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.scheduler import (
    FCFSWaitingQueue,
    WaitingQueue,
    create_waiting_queue,
)
from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import (
    AgingPriorityWaitingQueue,
    PriorityWaitingQueue,
)
from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.llmapi.llm_args import (
    CacheTransceiverConfig,
    SchedulerConfig,
    TorchLlmArgs,
    WaitingQueuePolicy,
)

pytestmark = pytest.mark.cpu_only


def create_mock_request_item(request_id: int) -> RequestQueueItem:
    """Create a mock RequestQueueItem for testing."""
    mock_request = Mock()
    return RequestQueueItem(request_id, mock_request)


def create_priority_request_item(
    request_id: int, priority: float = DEFAULT_REQUEST_PRIORITY
) -> RequestQueueItem:
    """Create a mock RequestQueueItem with an explicit float priority."""
    mock_request = Mock()
    mock_request.priority = priority
    return RequestQueueItem(request_id, mock_request)


class TestFCFSWaitingQueue:
    """Tests for FCFSWaitingQueue."""

    def test_add_request(self):
        """Test adding a single request."""
        queue = FCFSWaitingQueue()
        item = create_mock_request_item(1)

        queue.add_request(item)

        assert len(queue) == 1
        assert queue.peek_request() == item

    def test_add_requests(self):
        """Test adding multiple requests."""
        queue = FCFSWaitingQueue()
        items = [create_mock_request_item(i) for i in range(3)]

        queue.add_requests(items)

        assert len(queue) == 3

    def test_pop_request_fcfs_order(self):
        """Test that pop_request returns requests in FCFS order."""
        queue = FCFSWaitingQueue()
        items = [create_mock_request_item(i) for i in range(3)]
        queue.add_requests(items)

        # Should pop in order: 0, 1, 2
        assert queue.pop_request().id == 0
        assert queue.pop_request().id == 1
        assert queue.pop_request().id == 2

    def test_pop_from_empty_queue(self):
        """Test that pop_request raises IndexError on empty queue."""
        queue = FCFSWaitingQueue()

        with pytest.raises(IndexError):
            queue.pop_request()

    def test_peek_request(self):
        """Test peeking at the front of the queue."""
        queue = FCFSWaitingQueue()
        items = [create_mock_request_item(i) for i in range(3)]
        queue.add_requests(items)

        # Peek should return first item without removing it
        assert queue.peek_request().id == 0
        assert len(queue) == 3  # Size unchanged

    def test_peek_from_empty_queue(self):
        """Test that peek_request raises IndexError on empty queue."""
        queue = FCFSWaitingQueue()

        with pytest.raises(IndexError):
            queue.peek_request()

    def test_peek_requests(self):
        """peek_requests(k) returns the first k requests in pop order without
        removing them; they form one admission group."""
        queue = FCFSWaitingQueue()
        queue.add_requests([create_mock_request_item(i) for i in range(2, 6)])
        queue.prepend_requests([create_mock_request_item(i) for i in [0, 1]])

        assert [item.id for item in queue.peek_requests(3)] == [0, 1, 2]
        assert queue.admission_groups(queue.peek_requests(3)) == [0, 0, 0]
        assert [item.id for item in queue.peek_requests(10)] == [0, 1, 2, 3, 4, 5]
        assert queue.peek_requests(0) == []
        assert len(queue) == 6
        assert FCFSWaitingQueue().peek_requests(4) == []

    def test_prepend_request(self):
        """Test prepending a request to the front."""
        queue = FCFSWaitingQueue()
        queue.add_request(create_mock_request_item(1))
        queue.add_request(create_mock_request_item(2))

        # Prepend item 0 to front
        queue.prepend_request(create_mock_request_item(0))

        # Should pop in order: 0, 1, 2
        assert queue.pop_request().id == 0
        assert queue.pop_request().id == 1
        assert queue.pop_request().id == 2

    def test_prepend_requests(self):
        """Test prepending multiple requests."""
        queue = FCFSWaitingQueue()
        queue.add_request(create_mock_request_item(3))

        # Prepend items [1, 2] — first item in the list ends up at the front
        queue.prepend_requests([create_mock_request_item(i) for i in [1, 2]])

        # Expected order: 1, 2, 3
        assert queue.pop_request().id == 1
        assert queue.pop_request().id == 2
        assert queue.pop_request().id == 3

    def test_remove_by_ids(self):
        """Test removing requests by their IDs."""
        queue = FCFSWaitingQueue()
        items = [create_mock_request_item(i) for i in range(5)]
        queue.add_requests(items)

        # Remove items 1 and 3
        queue.remove_by_ids({1, 3})

        assert len(queue) == 3
        remaining_ids = [item.id for item in queue]
        assert remaining_ids == [0, 2, 4]

    def test_remove_nonexistent_ids(self):
        """Test removing IDs that don't exist (should not raise)."""
        queue = FCFSWaitingQueue()
        items = [create_mock_request_item(i) for i in range(3)]
        queue.add_requests(items)

        # Remove IDs that don't exist
        queue.remove_by_ids({10, 20})

        assert len(queue) == 3

    def test_bool_empty_queue(self):
        """Test bool conversion for empty queue."""
        queue = FCFSWaitingQueue()
        assert not queue
        assert bool(queue) is False

    def test_bool_nonempty_queue(self):
        """Test bool conversion for non-empty queue."""
        queue = FCFSWaitingQueue()
        queue.add_request(create_mock_request_item(1))
        assert queue
        assert bool(queue) is True

    def test_len(self):
        """Test length of queue."""
        queue = FCFSWaitingQueue()
        assert len(queue) == 0

        queue.add_request(create_mock_request_item(1))
        assert len(queue) == 1

        queue.add_requests([create_mock_request_item(i) for i in range(2, 5)])
        assert len(queue) == 4

    def test_iter(self):
        """Test iteration over queue."""
        queue = FCFSWaitingQueue()
        items = [create_mock_request_item(i) for i in range(3)]
        queue.add_requests(items)

        iterated_ids = [item.id for item in queue]
        assert iterated_ids == [0, 1, 2]

        # Iteration should not consume items
        assert len(queue) == 3

    def test_is_waiting_queue_subclass(self):
        """Test that FCFSWaitingQueue is a WaitingQueue."""
        queue = FCFSWaitingQueue()
        assert isinstance(queue, WaitingQueue)


class TestCreateWaitingQueue:
    """Tests for create_waiting_queue factory function."""

    def test_create_fcfs_queue(self):
        """Test creating FCFS queue."""
        queue = create_waiting_queue(WaitingQueuePolicy.FCFS)
        assert isinstance(queue, FCFSWaitingQueue)

    def test_create_default_queue(self):
        """Test creating queue with default policy."""
        queue = create_waiting_queue()
        assert isinstance(queue, FCFSWaitingQueue)

    def test_create_priority_queue(self):
        """Test creating a PriorityWaitingQueue via PRIORITY policy."""
        queue = create_waiting_queue(WaitingQueuePolicy.PRIORITY)
        assert isinstance(queue, PriorityWaitingQueue)


class TestPriorityWaitingQueue:
    """Tests for PriorityWaitingQueue.

    Covers priority ordering, FCFS tiebreak for equal priorities, and
    the full WaitingQueue interface (add, pop, peek, prepend, remove,
    bool, len, iter).
    """

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    def test_high_priority_served_before_low(self):
        """Higher priority requests are popped before lower priority ones."""
        q = PriorityWaitingQueue()
        q.add_request(create_priority_request_item(1, priority=0.1))
        q.add_request(create_priority_request_item(2, priority=0.9))
        q.add_request(create_priority_request_item(3, priority=0.5))
        assert [q.pop_request().id for _ in range(3)] == [2, 3, 1]

    def test_equal_priority_falls_back_to_fcfs(self):
        """Requests with equal priority are served in arrival (FCFS) order."""
        q = PriorityWaitingQueue()
        for req_id in [10, 20, 30]:
            q.add_request(create_priority_request_item(req_id, priority=0.7))
        assert [q.pop_request().id for _ in range(3)] == [10, 20, 30]

    def test_default_priority_is_fcfs(self):
        """Requests using the default priority are served in arrival order."""
        q = PriorityWaitingQueue()
        for req_id in range(5):
            q.add_request(create_priority_request_item(req_id))
        assert [q.pop_request().id for _ in range(5)] == [0, 1, 2, 3, 4]

    def test_mixed_priorities_correct_order(self):
        """Mixed priorities are served strictly in descending priority order."""
        q = PriorityWaitingQueue()
        items = [
            (1, 0.3),
            (2, 1.0),
            (3, 0.0),
            (4, 0.8),
            (5, 0.5),
        ]
        for req_id, priority in items:
            q.add_request(create_priority_request_item(req_id, priority))
        # Expected order: 2 (1.0), 4 (0.8), 5 (0.5), 1 (0.3), 3 (0.0)
        assert [q.pop_request().id for _ in range(5)] == [2, 4, 5, 1, 3]

    def test_interleaved_add_and_pop(self):
        """Priority order is maintained when requests are added between pops."""
        q = PriorityWaitingQueue()
        q.add_request(create_priority_request_item(1, priority=0.5))
        q.add_request(create_priority_request_item(2, priority=0.2))
        assert q.pop_request().id == 1  # highest so far

        # Add a higher-priority request after the first pop
        q.add_request(create_priority_request_item(3, priority=0.9))
        assert q.pop_request().id == 3
        assert q.pop_request().id == 2

    # ------------------------------------------------------------------
    # add_requests (batch insert)
    # ------------------------------------------------------------------

    def test_add_requests_batch(self):
        """add_requests inserts all items and respects priority ordering."""
        q = PriorityWaitingQueue()
        q.add_requests(
            [
                create_priority_request_item(1, priority=0.2),
                create_priority_request_item(2, priority=0.8),
                create_priority_request_item(3, priority=0.5),
            ]
        )
        assert len(q) == 3
        assert [q.pop_request().id for _ in range(3)] == [2, 3, 1]

    # ------------------------------------------------------------------
    # peek_request
    # ------------------------------------------------------------------

    def test_peek_returns_highest_priority_without_removing(self):
        """peek_request returns the highest-priority item and does not remove it."""
        q = PriorityWaitingQueue()
        q.add_request(create_priority_request_item(1, priority=0.3))
        q.add_request(create_priority_request_item(2, priority=0.9))

        peeked = q.peek_request()
        assert peeked.id == 2
        assert len(q) == 2  # not consumed

    def test_peek_from_empty_raises(self):
        """peek_request raises IndexError on an empty queue."""
        with pytest.raises(IndexError):
            PriorityWaitingQueue().peek_request()

    def test_pop_from_empty_raises(self):
        """pop_request raises IndexError on an empty queue."""
        with pytest.raises(IndexError):
            PriorityWaitingQueue().pop_request()

    # ------------------------------------------------------------------
    # prepend_request / prepend_requests
    # ------------------------------------------------------------------

    def test_prepend_request_respects_priority(self):
        """prepend_request re-inserts by priority, not unconditionally to front."""
        q = PriorityWaitingQueue()
        q.add_request(create_priority_request_item(1, priority=0.8))
        q.add_request(create_priority_request_item(2, priority=0.6))

        # Prepend a low-priority item — it should not jump to the front
        q.prepend_request(create_priority_request_item(3, priority=0.1))

        assert [q.pop_request().id for _ in range(3)] == [1, 2, 3]

    def test_prepend_requests_respects_priority(self):
        """prepend_requests re-inserts all items by priority."""
        q = PriorityWaitingQueue()
        q.add_request(create_priority_request_item(1, priority=0.5))
        q.prepend_requests(
            [
                create_priority_request_item(2, priority=0.9),
                create_priority_request_item(3, priority=0.1),
            ]
        )
        assert [q.pop_request().id for _ in range(3)] == [2, 1, 3]

    def test_prepend_request_beats_later_arrivals_of_same_priority(self):
        """A prepended request comes out before same-priority requests added after it."""
        q = PriorityWaitingQueue()
        # Add two requests, then pop the first one to simulate the DP-constraint path.
        req_a = create_priority_request_item(1, priority=0.5)
        req_b = create_priority_request_item(2, priority=0.5)
        q.add_request(req_a)
        q.add_request(req_b)
        popped = q.pop_request()  # removes req_a (earliest arrival)
        assert popped.id == 1

        # A new request arrives while req_a was being processed.
        req_c = create_priority_request_item(3, priority=0.5)
        q.add_request(req_c)

        # req_a is returned to the queue (couldn't be scheduled).
        q.prepend_request(popped)

        # req_a should still come out before req_c despite req_c being in the
        # queue before the prepend call.
        assert q.pop_request().id == 1  # req_a
        assert q.pop_request().id == 2  # req_b
        assert q.pop_request().id == 3  # req_c

    def test_prepend_requests_restores_original_queue_order(self):
        """Simulates the request_utils.py pattern: pop several requests that
        cannot be scheduled, then prepend them back.  The resulting pop
        sequence must match the original queue order."""
        q = PriorityWaitingQueue()
        # Original queue: A(0.9) > B(0.5) > C(0.5) — B and C tied on priority,
        # so B comes before C by FCFS.
        req_a = create_priority_request_item(1, priority=0.9)
        req_b = create_priority_request_item(2, priority=0.5)
        req_c = create_priority_request_item(3, priority=0.5)
        q.add_request(req_a)
        q.add_request(req_b)
        q.add_request(req_c)

        # Simulate the scheduler dequeuing all three but failing to schedule them.
        pending = [q.pop_request(), q.pop_request(), q.pop_request()]
        assert [r.id for r in pending] == [1, 2, 3]

        # Put them back — prepend_requests preserves input order internally.
        q.prepend_requests(pending)

        # Order must be fully restored.
        assert [q.pop_request().id for _ in range(3)] == [1, 2, 3]

    def test_prepend_requests_restores_order_with_existing_queue_entries(self):
        """Simulates the request_utils.py pattern when new requests have
        arrived while the scheduler was trying to place the popped ones.

        Returned requests must come out ahead of any same-priority request
        that arrived after they were originally enqueued, and the relative
        order among the returned requests must be preserved.
        """
        q = PriorityWaitingQueue()
        # Original queue: A(0.9) > B(0.5) > C(0.5)
        req_a = create_priority_request_item(1, priority=0.9)
        req_b = create_priority_request_item(2, priority=0.5)
        req_c = create_priority_request_item(3, priority=0.5)
        q.add_request(req_a)
        q.add_request(req_b)
        q.add_request(req_c)

        # Scheduler dequeues all three but cannot schedule them.
        pending = [q.pop_request(), q.pop_request(), q.pop_request()]
        assert [r.id for r in pending] == [1, 2, 3]

        # While the scheduler was working, new requests arrive at various
        # priorities: one higher (0.9), one equal (0.5), one lower (0.1).
        q.add_request(create_priority_request_item(4, priority=0.9))
        q.add_request(create_priority_request_item(5, priority=0.5))
        q.add_request(create_priority_request_item(6, priority=0.1))

        # Put the pending requests back.
        q.prepend_requests(pending)

        # Expected order:
        #   priority 0.9: A(1) before D(4) — A arrived first
        #   priority 0.5: B(2) before C(3) before E(5) — B and C arrived first, in FCFS order
        #   priority 0.1: F(6)
        assert [q.pop_request().id for _ in range(6)] == [1, 4, 2, 3, 5, 6]

    def test_prepend_requests_does_not_starve_across_multiple_rounds(self):
        """A request returned to the queue multiple times still comes out
        ahead of newly-arrived requests of the same priority."""
        q = PriorityWaitingQueue()
        req_a = create_priority_request_item(1, priority=0.5)
        q.add_request(req_a)

        # Simulate req_a being returned to the queue twice (two scheduling rounds
        # where it couldn't run due to resource constraints).
        for _ in range(2):
            popped = q.pop_request()
            # New request arrives while req_a couldn't be scheduled.
            q.add_request(create_priority_request_item(99, priority=0.5))
            q.prepend_request(popped)

        # req_a must still be served before the newly-arrived requests.
        assert q.pop_request().id == 1

    # ------------------------------------------------------------------
    # remove_by_ids
    # ------------------------------------------------------------------

    def test_remove_by_ids_removes_correct_items(self):
        """remove_by_ids removes exactly the specified requests."""
        q = PriorityWaitingQueue()
        for i, p in enumerate([0.9, 0.7, 0.5, 0.3, 0.1]):
            q.add_request(create_priority_request_item(i, priority=p))

        q.remove_by_ids({1, 3})

        assert len(q) == 3
        remaining = [q.pop_request().id for _ in range(3)]
        assert remaining == [0, 2, 4]

    def test_remove_by_ids_nonexistent_is_noop(self):
        """remove_by_ids with unknown IDs does not change the queue."""
        q = PriorityWaitingQueue()
        q.add_request(create_priority_request_item(1, priority=0.8))
        q.add_request(create_priority_request_item(2, priority=0.4))

        q.remove_by_ids({99, 100})

        assert len(q) == 2

    def test_remove_all_ids_leaves_empty_queue(self):
        """remove_by_ids can empty the queue entirely."""
        q = PriorityWaitingQueue()
        q.add_request(create_priority_request_item(1, priority=0.8))
        q.add_request(create_priority_request_item(2, priority=0.4))

        q.remove_by_ids({1, 2})

        assert len(q) == 0
        assert not q

    def test_priority_order_preserved_after_remove(self):
        """Priority ordering is intact after remove_by_ids."""
        q = PriorityWaitingQueue()
        for i, p in enumerate([0.9, 0.7, 0.5, 0.3]):
            q.add_request(create_priority_request_item(i, priority=p))

        q.remove_by_ids({1})  # remove priority=0.7

        assert [q.pop_request().id for _ in range(3)] == [0, 2, 3]

    # ------------------------------------------------------------------
    # bool / len / iter
    # ------------------------------------------------------------------

    def test_bool_empty(self):
        """Empty queue is falsy."""
        assert not PriorityWaitingQueue()

    def test_bool_nonempty(self):
        """Non-empty queue is truthy."""
        q = PriorityWaitingQueue()
        q.add_request(create_priority_request_item(1, priority=0.5))
        assert q

    def test_len_tracks_adds_and_pops(self):
        """__len__ accurately reflects the number of items."""
        q = PriorityWaitingQueue()
        assert len(q) == 0
        q.add_request(create_priority_request_item(1, priority=0.5))
        assert len(q) == 1
        q.add_requests([create_priority_request_item(i, priority=0.5) for i in range(2, 5)])
        assert len(q) == 4
        q.pop_request()
        assert len(q) == 3

    def test_iter_yields_in_priority_order(self):
        """__iter__ yields items in descending priority order."""
        q = PriorityWaitingQueue()
        for i, p in enumerate([0.1, 0.9, 0.5]):
            q.add_request(create_priority_request_item(i, priority=p))

        iterated_ids = [item.id for item in q]
        assert iterated_ids == [1, 2, 0]  # 0.9, 0.5, 0.1

    def test_peek_requests_follows_pop_order(self):
        """peek_requests(k) matches the pop order and removes nothing; each
        priority is one admission group."""
        q = PriorityWaitingQueue()
        for i, p in enumerate([0.1, 0.9, 0.5, 0.5, 0.9]):
            q.add_request(create_priority_request_item(i, priority=p))
        q.prepend_requests([create_priority_request_item(5, priority=0.5)])

        assert [item.id for item in q.peek_requests(4)] == [1, 4, 5, 2]
        assert q.admission_groups(q.peek_requests(4)) == [0, 0, 1, 1]
        assert [item.id for item in q.peek_requests(10)] == [item.id for item in q]
        assert q.peek_requests(0) == []
        assert len(q) == 6
        assert [q.pop_request().id for _ in range(6)] == [1, 4, 5, 2, 3, 0]

    def test_peek_requests_does_not_sort_the_heap(self, monkeypatch):
        """peek_requests does not go through __iter__, which sorts a copy of
        the whole heap."""
        q = PriorityWaitingQueue()
        for i in range(1000):
            q.add_request(create_priority_request_item(i, priority=(i % 7) / 10))

        def _no_full_sort(self):
            raise AssertionError("peek_requests must not sort the whole heap")

        monkeypatch.setattr(PriorityWaitingQueue, "__iter__", _no_full_sort)
        peeked = [item.id for item in q.peek_requests(4)]
        assert peeked == [6, 13, 20, 27]

    def test_peek_requests_reads_only_the_entries_it_returns_and_their_children(self):
        """A home-wait round re-peeks a growing window: each peek walks O(k) heap entries, not the whole heap, and
        matches the pop order for any k."""
        import random

        rng = random.Random(3)
        q = PriorityWaitingQueue()
        for i in range(5000):
            q.add_request(
                create_priority_request_item(i, priority=rng.choice([0.1, 0.3, 0.5, 0.7]))
            )
        q.prepend_requests([create_priority_request_item(9000 + i, priority=0.5) for i in range(3)])
        order = [e[2].id for e in sorted(q._heap)]
        for k in (0, 1, 7, 256, 5003, 6000):
            assert [item.id for item in q.peek_requests(k)] == order[:k]

        class _Counting(list):
            reads = 0

            def __getitem__(self, index):
                _Counting.reads += 1
                return super().__getitem__(index)

            def __iter__(self):
                _Counting.reads += len(self)
                return super().__iter__()

        q._heap = _Counting(q._heap)
        assert [item.id for item in q.peek_requests(16)] == order[:16]
        assert _Counting.reads <= 1 + 2 * 16

    def test_iter_does_not_consume_items(self):
        """Iterating over the queue does not remove items."""
        q = PriorityWaitingQueue()
        q.add_request(create_priority_request_item(1, priority=0.8))
        q.add_request(create_priority_request_item(2, priority=0.4))

        _ = list(q)
        assert len(q) == 2

    def test_is_waiting_queue_subclass(self):
        """PriorityWaitingQueue is a WaitingQueue."""
        assert isinstance(PriorityWaitingQueue(), WaitingQueue)


def create_aging_request_item(
    request_id: int, priority: float, arrival=None, orig_arrival=None
) -> RequestQueueItem:
    """Create a mock RequestQueueItem with a priority and rank 0's arrival stamps."""
    mock_request = Mock()
    mock_request.priority = priority
    item = RequestQueueItem(request_id, mock_request)
    item.admit_arrival = arrival
    item.admit_orig_arrival = orig_arrival
    return item


def _started(queue: AgingPriorityWaitingQueue, now: float = 0.0) -> AgingPriorityWaitingQueue:
    queue.set_now(now)
    return queue


class TestAgingPriorityWaitingQueue:
    """Tests for AgingPriorityWaitingQueue (waiting_queue_policy=priority with a wait cap)."""

    def test_highest_priority_first_ties_in_arrival_order(self):
        queue = _started(AgingPriorityWaitingQueue(max_wait_s=60.0))
        queue.add_requests(
            [
                create_aging_request_item(1, 0.3, 0.0),
                create_aging_request_item(2, 0.9, 0.0),
                create_aging_request_item(3, 0.1, 0.0),
                create_aging_request_item(4, 0.9, 0.0),
            ]
        )
        assert [item.id for item in queue] == [2, 4, 1, 3]
        assert [item.id for item in queue.peek_requests(2)] == [2, 4]
        assert queue.peek_request().id == 2
        assert [queue.pop_request().id for _ in range(4)] == [2, 4, 1, 3]

    def test_same_order_as_priority_queue_until_a_request_ages(self):
        """Below the cap the order is PriorityWaitingQueue's, including its admission groups."""
        priorities = [0.5, 0.2, 0.9, 0.2, 0.7, 0.9, 0.5]
        aging = _started(AgingPriorityWaitingQueue(max_wait_s=60.0))
        stock = PriorityWaitingQueue()
        for i, p in enumerate(priorities):
            aging.add_request(create_aging_request_item(i, p, 0.0))
            stock.add_request(create_aging_request_item(i, p, 0.0))
        assert [item.id for item in aging] == [item.id for item in stock]
        window = aging.peek_requests(len(priorities))
        assert aging.admission_groups(window) == stock.admission_groups(stock.peek_requests(7))
        assert [aging.pop_request().id for _ in priorities] == [
            stock.pop_request().id for _ in priorities
        ]

    def test_put_back_keeps_arrival_order(self):
        queue = _started(AgingPriorityWaitingQueue(max_wait_s=60.0))
        queue.add_requests(
            [create_aging_request_item(1, 0.5, 0.0), create_aging_request_item(2, 0.5, 0.0)]
        )
        first = queue.pop_request()
        queue.add_request(create_aging_request_item(3, 0.5, 0.0))
        queue.prepend_request(first)
        assert [item.id for item in queue] == [1, 2, 3]

    @pytest.mark.parametrize("now, first", [(59.0, 2), (60.0, 1)])
    def test_wait_cap_goes_ahead_oldest_first(self, now, first):
        queue = _started(AgingPriorityWaitingQueue(max_wait_s=60.0))
        queue.add_requests(
            [create_aging_request_item(1, 0.1, 0.0), create_aging_request_item(2, 0.9, 30.0)]
        )
        queue.set_now(now)
        assert queue.pop_request().id == first

    def test_wait_cap_ages_from_the_original_arrival(self):
        """In disaggregated serving the cap counts from the orchestrator's arrival, which the
        context server's KV transfer timeout follows."""
        queue = AgingPriorityWaitingQueue(max_wait_s=60.0)
        queue.set_now(50.0)
        queue.add_requests(
            [
                create_aging_request_item(1, 0.2, arrival=50.0, orig_arrival=0.0),
                create_aging_request_item(2, 0.1, arrival=40.0, orig_arrival=-5.0),
                create_aging_request_item(3, 0.9, arrival=50.0, orig_arrival=45.0),
            ]
        )
        assert queue.peek_request().id == 3
        queue.set_now(60.0)
        # Both aged ones go first, oldest original arrival first.
        assert [item.id for item in queue] == [2, 1, 3]
        assert [queue.pop_request().id for _ in range(3)] == [2, 1, 3]

    def test_request_aged_when_added_or_put_back(self):
        queue = _started(AgingPriorityWaitingQueue(max_wait_s=60.0), 100.0)
        queue.add_requests(
            [create_aging_request_item(1, 0.9, 90.0), create_aging_request_item(2, 0.1, 0.0)]
        )
        aged = queue.pop_request()
        assert aged.id == 2
        queue.prepend_request(aged)
        assert [item.id for item in queue] == [2, 1]

    def test_cap_needs_clock(self):
        queue = AgingPriorityWaitingQueue(max_wait_s=60.0)
        queue.add_request(create_aging_request_item(1, 0.5, 0.0))
        with pytest.raises(RuntimeError):
            queue.pop_request()

    @pytest.mark.parametrize("bad", [None, 0.0, -1.0, float("inf"), float("nan")])
    def test_cap_must_be_positive_and_finite(self, bad):
        with pytest.raises((ValueError, TypeError)):
            AgingPriorityWaitingQueue(max_wait_s=bad)

    def test_admission_groups_keep_aged_requests_in_order(self):
        """Out-of-order admission may reorder only requests that have not aged and share a priority."""
        queue = _started(AgingPriorityWaitingQueue(max_wait_s=60.0))
        queue.add_requests(
            [
                create_aging_request_item(1, 0.1, 0.0),
                create_aging_request_item(2, 0.2, 1.0),
                create_aging_request_item(3, 0.9, 50.0),
                create_aging_request_item(4, 0.9, 50.0),
                create_aging_request_item(5, 0.5, 50.0),
            ]
        )
        queue.set_now(61.0)
        window = queue.peek_requests(5)
        assert [item.id for item in window] == [1, 2, 3, 4, 5]
        assert queue.admission_groups(window) == [0, 1, 2, 2, 3]
        assert queue.admission_groups(window[2:]) == [0, 0, 1]

    def test_unadmitted_request_keeps_its_arrival_order(self):
        """A popped request put back after a later admission step keeps its place."""
        queue = _started(AgingPriorityWaitingQueue(max_wait_s=60.0))
        queue.add_requests([create_aging_request_item(i, 0.5, 0.0) for i in range(1, 4)])
        popped = [queue.pop_request() for _ in range(2)]
        queue.mark_admitted(popped[:1])
        queue.add_request(create_aging_request_item(4, 0.5, 0.0))
        queue.prepend_requests(popped[1:])
        assert [item.id for item in queue] == [2, 3, 4]

    def test_put_back_every_iteration_adds_no_entries(self, monkeypatch):
        """Requests that admission rejects and puts back every iteration (strict-rank attention
        DP) reuse their heap entries, so the heaps stay bounded without compaction and the
        requests still age once, oldest first."""
        queue = _started(AgingPriorityWaitingQueue(max_wait_s=60.0))
        queue.add_requests(
            [create_aging_request_item(i, 0.9 - 0.1 * i, float(3 - i)) for i in range(4)]
        )
        compactions = []
        compact = AgingPriorityWaitingQueue._compact
        monkeypatch.setattr(
            AgingPriorityWaitingQueue,
            "_compact",
            lambda self: compactions.append(len(self)) or compact(self),
        )
        for step in range(1000):
            queue.set_now(3.0 + step * 0.01)
            popped = [queue.pop_request() for _ in range(4)]
            assert [item.id for item in popped] == [0, 1, 2, 3]
            queue.prepend_requests(popped)
        assert len(queue._priority_heap) == 4
        assert len(queue._arrival_heap) == 4
        assert compactions == []
        queue.set_now(63.0)
        assert [item.id for item in queue] == [3, 2, 1, 0]
        assert [queue.pop_request().id for _ in range(4)] == [3, 2, 1, 0]

    def test_peek_requests_does_not_sort_the_queue(self, monkeypatch):
        queue = _started(AgingPriorityWaitingQueue(max_wait_s=60.0))
        queue.add_requests(
            [create_aging_request_item(i, 1.0 - i / 1000.0, float(i)) for i in range(1000)]
        )
        queue.set_now(62.0)

        def _no_full_sort(self):
            raise AssertionError("peek_requests must not sort the whole queue")

        monkeypatch.setattr(AgingPriorityWaitingQueue, "__iter__", _no_full_sort)
        assert [item.id for item in queue.peek_requests(5)] == [0, 1, 2, 3, 4]

    def test_never_reads_max_tokens(self):
        """The order comes from the priority alone (the orchestrator's predicted output length)."""

        class NoMaxTokens:
            priority = 0.5

            @property
            def max_tokens(self):
                raise AssertionError("max_tokens must not be read")

        queue = _started(AgingPriorityWaitingQueue(max_wait_s=60.0))
        for i in range(3):
            item = RequestQueueItem(i, NoMaxTokens())
            item.admit_arrival = 0.0
            queue.add_request(item)
        queue.set_now(100.0)
        assert [queue.pop_request().id for _ in range(3)] == [0, 1, 2]

    def test_remove_by_ids(self):
        queue = _started(AgingPriorityWaitingQueue(max_wait_s=60.0))
        queue.add_requests([create_aging_request_item(i, 1.0 / i, 0.0) for i in range(1, 5)])
        queue.set_now(60.0)
        queue.remove_by_ids({2, 3})
        assert len(queue) == 2 and [item.id for item in queue] == [1, 4]
        assert [queue.pop_request().id for _ in range(2)] == [1, 4]
        assert not queue

    def test_factory(self):
        queue = create_waiting_queue(WaitingQueuePolicy.PRIORITY, max_wait_s=120.0)
        assert isinstance(queue, AgingPriorityWaitingQueue)
        assert queue.max_wait_s == 120.0
        # Without a cap the policy keeps the stock queue.
        assert type(create_waiting_queue(WaitingQueuePolicy.PRIORITY)) is PriorityWaitingQueue
        with pytest.raises(ValueError, match="priority"):
            create_waiting_queue(WaitingQueuePolicy.FCFS, max_wait_s=60.0)

    def test_factory_refuses_cap_with_gen_admission(self):
        gen_admission = Mock(enabled=True)
        gen_admission.describe.return_value = "p90"
        with pytest.raises(ValueError, match="GEN admission"):
            create_waiting_queue(
                WaitingQueuePolicy.PRIORITY, gen_admission=gen_admission, max_wait_s=60.0
            )

    def test_scheduler_config_max_wait(self):
        priority = WaitingQueuePolicy.PRIORITY
        assert SchedulerConfig().waiting_queue_max_wait_s is None
        config = SchedulerConfig(waiting_queue_policy=priority, waiting_queue_max_wait_s=120.0)
        assert config.waiting_queue_max_wait_s == 120.0
        SchedulerConfig(**config.model_dump())
        SchedulerConfig(**SchedulerConfig().model_dump())
        with pytest.raises(ValueError, match="priority"):
            SchedulerConfig(
                waiting_queue_policy=WaitingQueuePolicy.FCFS, waiting_queue_max_wait_s=60.0
            )
        with pytest.raises(ValueError):
            SchedulerConfig(waiting_queue_policy="shortest_decode_first")
        for bad in (float("inf"), float("nan"), 0.0):
            with pytest.raises(ValueError):
                SchedulerConfig(waiting_queue_policy=priority, waiting_queue_max_wait_s=bad)

    @pytest.mark.parametrize(
        "max_wait_s, timeout_ms, warns",
        [(60.0, 60000, False), (30.0, 60000, False), (120.0, 60000, True), (60.0, 30000, True)],
        ids=["defaults", "shorter_cap", "longer_cap", "shorter_timeout"],
    )
    def test_max_wait_above_transfer_timeout_warns(self, tmp_path, max_wait_s, timeout_ms, warns):
        """Only a cap longer than the KV transfer timeout warns: the defaults are equal and stay quiet."""
        with patch.object(llm_args_mod.logger, "warning") as warning:
            TorchLlmArgs(
                model=tmp_path,
                gpus_per_node=1,
                scheduler_config=SchedulerConfig(
                    waiting_queue_policy=WaitingQueuePolicy.PRIORITY,
                    waiting_queue_max_wait_s=max_wait_s,
                ),
                cache_transceiver_config=CacheTransceiverConfig(
                    backend="NIXL", kv_transfer_timeout_ms=timeout_ms
                ),
            )
        warned = any(
            "waiting_queue_max_wait_s" in str(call.args[0]) for call in warning.call_args_list
        )
        assert warned == warns
