"""Tests for WaitingQueue implementations.

This module tests the waiting queue functionality including:
- FCFSWaitingQueue operations
- PriorityWaitingQueue operations
- WaitingQueue abstract interface
- create_waiting_queue factory function
"""

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.scheduler import (
    FCFSWaitingQueue,
    WaitingQueue,
    create_waiting_queue,
)
from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import PriorityWaitingQueue
from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.llmapi.llm_args import WaitingQueuePolicy


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
