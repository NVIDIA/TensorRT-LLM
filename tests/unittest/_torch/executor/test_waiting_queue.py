"""Tests for WaitingQueue implementations.

This module tests the waiting queue functionality including:
- FCFSWaitingQueue operations
- PriorityWaitingQueue operations
- SJFWaitingQueue operations
- WaitingQueue abstract interface
- create_waiting_queue factory function
"""

from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.scheduler import (
    FCFSWaitingQueue,
    SJFWaitingQueue,
    WaitingQueue,
    create_waiting_queue,
)
from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import PriorityWaitingQueue
from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.llmapi.llm_args import SJFConfig, WaitingQueueConfig, WaitingQueuePolicy


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
        config = WaitingQueueConfig(policy=WaitingQueuePolicy.FCFS)
        queue = create_waiting_queue(config)
        assert isinstance(queue, FCFSWaitingQueue)

    def test_create_default_queue(self):
        """Test creating queue with default config (None → FCFS)."""
        queue = create_waiting_queue()
        assert isinstance(queue, FCFSWaitingQueue)

    def test_create_priority_queue(self):
        """Test creating a PriorityWaitingQueue via PRIORITY policy."""
        config = WaitingQueueConfig(policy=WaitingQueuePolicy.PRIORITY)
        queue = create_waiting_queue(config)
        assert isinstance(queue, PriorityWaitingQueue)

    def test_create_sjf_queue(self):
        """Test creating an SJFWaitingQueue via SJF policy."""
        config = WaitingQueueConfig(policy=WaitingQueuePolicy.SJF)
        queue = create_waiting_queue(config)
        assert isinstance(queue, SJFWaitingQueue)

    def test_create_sjf_queue_with_params(self):
        """Test creating SJF queue with custom parameters."""
        mock_kv = Mock()
        config = WaitingQueueConfig(
            policy=WaitingQueuePolicy.SJF,
            sjf=SJFConfig(cache_aware=True, aging_factor=0.01),
        )
        queue = create_waiting_queue(config, kv_cache_manager=mock_kv)
        assert isinstance(queue, SJFWaitingQueue)
        assert queue._cache_aware is True
        assert queue._aging_factor == 0.01


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


# ------------------------------------------------------------------
# SJF helpers
# ------------------------------------------------------------------


def create_sjf_request_item(request_id: int, prompt_len: int) -> RequestQueueItem:
    """Create a mock RequestQueueItem with input_token_ids of given length."""
    mock_request = Mock()
    mock_request.input_token_ids = list(range(prompt_len))
    mock_request.lora_config = None
    return RequestQueueItem(request_id, mock_request)


def create_sjf_request_with_lora(
    request_id: int, prompt_len: int, lora_task_id: int
) -> RequestQueueItem:
    """Create a mock RequestQueueItem with LoRA config."""
    mock_request = Mock()
    mock_request.input_token_ids = list(range(prompt_len))
    lora_config = Mock()
    lora_config.task_id = lora_task_id
    mock_request.lora_config = lora_config
    return RequestQueueItem(request_id, mock_request)


class TestSJFWaitingQueue:
    """Tests for SJFWaitingQueue.

    Covers SJF ordering, cache-aware mode, aging, FCFS tiebreak, and
    the full WaitingQueue interface.
    """

    # ------------------------------------------------------------------
    # Basic SJF ordering (no cache, no aging)
    # ------------------------------------------------------------------

    def test_shorter_job_served_first(self):
        """Requests with fewer tokens are popped first."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        q.add_request(create_sjf_request_item(1, prompt_len=100))
        q.add_request(create_sjf_request_item(2, prompt_len=50))
        q.add_request(create_sjf_request_item(3, prompt_len=75))
        assert [q.pop_request().id for _ in range(3)] == [2, 3, 1]

    def test_equal_length_falls_back_to_fcfs(self):
        """Requests with equal prompt_len are served in arrival order."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        for req_id in [10, 20, 30]:
            q.add_request(create_sjf_request_item(req_id, prompt_len=100))
        assert [q.pop_request().id for _ in range(3)] == [10, 20, 30]

    def test_interleaved_add_and_pop(self):
        """SJF order is maintained when requests are added between pops."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        q.add_request(create_sjf_request_item(1, prompt_len=200))
        q.add_request(create_sjf_request_item(2, prompt_len=100))
        assert q.pop_request().id == 2  # shorter

        q.add_request(create_sjf_request_item(3, prompt_len=50))
        assert q.pop_request().id == 3  # even shorter
        assert q.pop_request().id == 1  # remaining

    def test_add_requests_batch(self):
        """add_requests inserts all items and respects SJF ordering."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        q.add_requests(
            [
                create_sjf_request_item(1, prompt_len=300),
                create_sjf_request_item(2, prompt_len=100),
                create_sjf_request_item(3, prompt_len=200),
            ]
        )
        assert len(q) == 3
        assert [q.pop_request().id for _ in range(3)] == [2, 3, 1]

    def test_request_with_none_request(self):
        """A RequestQueueItem with request=None gets compute_tokens=0."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        q.add_request(create_sjf_request_item(1, prompt_len=100))
        q.add_request(RequestQueueItem(2, None))  # request=None
        # None request has compute_tokens=0, so it comes first
        assert q.pop_request().id == 2
        assert q.pop_request().id == 1

    # ------------------------------------------------------------------
    # Cache-aware mode
    # ------------------------------------------------------------------

    def test_cache_aware_reduces_compute_tokens(self):
        """Cache-aware mode uses (prompt_len - cached_len) for ordering."""
        mock_kv = Mock()
        # Request 1: 200 tokens, 150 cached → compute=50
        # Request 2: 100 tokens, 0 cached → compute=100
        mock_kv.probe_prefix_match_length = Mock(side_effect=[150, 0])

        q = SJFWaitingQueue(kv_cache_manager=mock_kv, cache_aware=True, aging_factor=0)
        q.add_request(create_sjf_request_item(1, prompt_len=200))
        q.add_request(create_sjf_request_item(2, prompt_len=100))

        # Request 1 has fewer compute tokens (50 < 100), so it comes first
        assert q.pop_request().id == 1
        assert q.pop_request().id == 2

    def test_cache_aware_with_lora(self):
        """Cache-aware mode passes lora_task_id to probe."""
        mock_kv = Mock()
        mock_kv.probe_prefix_match_length = Mock(return_value=50)

        q = SJFWaitingQueue(kv_cache_manager=mock_kv, cache_aware=True, aging_factor=0)
        req = create_sjf_request_with_lora(1, prompt_len=100, lora_task_id=42)
        q.add_request(req)

        mock_kv.probe_prefix_match_length.assert_called_once()
        call_args = mock_kv.probe_prefix_match_length.call_args
        assert call_args[0][1] == 42  # lora_task_id

    def test_cache_aware_false_ignores_cache(self):
        """cache_aware=False uses raw prompt_len even if kv_cache_manager is set."""
        mock_kv = Mock()
        q = SJFWaitingQueue(kv_cache_manager=mock_kv, cache_aware=False, aging_factor=0)
        q.add_request(create_sjf_request_item(1, prompt_len=200))
        q.add_request(create_sjf_request_item(2, prompt_len=100))

        # No probe calls should be made
        mock_kv.probe_prefix_match_length.assert_not_called()
        # Pure prompt_len ordering
        assert q.pop_request().id == 2
        assert q.pop_request().id == 1

    def test_cache_aware_degrades_without_kv_manager(self):
        """cache_aware=True with kv_cache_manager=None degrades to prompt_len."""
        q = SJFWaitingQueue(kv_cache_manager=None, cache_aware=True, aging_factor=0)
        q.add_request(create_sjf_request_item(1, prompt_len=200))
        q.add_request(create_sjf_request_item(2, prompt_len=100))
        # Should work without error, using prompt_len
        assert q.pop_request().id == 2
        assert q.pop_request().id == 1

    # ------------------------------------------------------------------
    # Aging
    # ------------------------------------------------------------------

    def test_aging_promotes_old_requests(self):
        """Aging reduces priority score, eventually promoting old requests."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0.005)

        # time=0: add long request (8000 tokens)
        with patch("tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue.time") as mock_time:
            mock_time.monotonic.return_value = 0.0
            q.add_request(create_sjf_request_item(1, prompt_len=8000))

        # time=0: add short request (2000 tokens)
        with patch("tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue.time") as mock_time:
            mock_time.monotonic.return_value = 0.0
            q.add_request(create_sjf_request_item(2, prompt_len=2000))

        # At time=0, request 2 (2000) should come first
        with patch("tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue.time") as mock_time:
            mock_time.monotonic.return_value = 0.0
            assert q.peek_request().id == 2

        # At time=180, request 1 score: 8000*(1-0.005*180) = 8000*0.1 = 800
        #               request 2 score: 2000*(1-0.005*180) = 2000*0.1 = 200
        # Request 2 still first (lower score)
        # But at time=195, request 1 score: 8000*(1-0.005*195) = 8000*0.025 = 200
        #                  request 2 score: 2000*(1-0.005*195) = 2000*0.025 = 50
        # Request 2 still first. Aging is proportional — both shrink equally.
        # The design ensures no starvation via the priority=0 floor.

        # At time=200, both hit 0 — first one (request 1) by counter tiebreak
        # Actually request 1 has counter=0 (earlier), request 2 has counter=1
        with patch("tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue.time") as mock_time:
            mock_time.monotonic.return_value = 200.0
            # Both scores are 0, so counter tiebreak applies: req 1 (counter=0) first
            assert q.pop_request().id == 1
            assert q.pop_request().id == 2

    def test_aging_factor_zero_is_pure_sjf(self):
        """aging_factor=0 means scores never change (pure SJF)."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)

        with patch("tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue.time") as mock_time:
            mock_time.monotonic.return_value = 0.0
            q.add_request(create_sjf_request_item(1, prompt_len=8000))

        with patch("tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue.time") as mock_time:
            mock_time.monotonic.return_value = 0.0
            q.add_request(create_sjf_request_item(2, prompt_len=2000))

        # Even far in the future, short request still comes first
        with patch("tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue.time") as mock_time:
            mock_time.monotonic.return_value = 10000.0
            assert q.pop_request().id == 2
            assert q.pop_request().id == 1

    # ------------------------------------------------------------------
    # prepend_request / prepend_requests
    # ------------------------------------------------------------------

    def test_prepend_request_sorts_ahead_of_same_score(self):
        """Prepended request beats same-score later arrivals."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        req_a = create_sjf_request_item(1, prompt_len=100)
        req_b = create_sjf_request_item(2, prompt_len=100)
        q.add_request(req_a)
        q.add_request(req_b)
        popped = q.pop_request()  # removes req_a (earlier by counter)
        assert popped.id == 1

        # New request arrives with same prompt_len
        req_c = create_sjf_request_item(3, prompt_len=100)
        q.add_request(req_c)

        # Prepend req_a back — it should come before req_c
        q.prepend_request(popped)

        assert q.pop_request().id == 1  # req_a (prepended)
        assert q.pop_request().id == 2  # req_b (earlier than req_c)
        assert q.pop_request().id == 3  # req_c (latest)

    def test_prepend_requests_preserves_order(self):
        """Simulates pop-several-then-prepend pattern from request_utils.py."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        # All same length — pure FCFS within SJF
        for req_id in [1, 2, 3]:
            q.add_request(create_sjf_request_item(req_id, prompt_len=100))

        pending = [q.pop_request(), q.pop_request(), q.pop_request()]
        assert [r.id for r in pending] == [1, 2, 3]

        q.prepend_requests(pending)
        assert [q.pop_request().id for _ in range(3)] == [1, 2, 3]

    def test_prepend_requests_restores_order_with_new_arrivals(self):
        """Prepended requests come out before same-score new arrivals."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        for req_id in [1, 2, 3]:
            q.add_request(create_sjf_request_item(req_id, prompt_len=100))

        pending = [q.pop_request(), q.pop_request(), q.pop_request()]
        assert [r.id for r in pending] == [1, 2, 3]

        # New arrivals while pending
        q.add_request(create_sjf_request_item(4, prompt_len=100))
        q.add_request(create_sjf_request_item(5, prompt_len=100))

        q.prepend_requests(pending)

        # Prepended [1,2,3] should come before new [4,5]
        assert [q.pop_request().id for _ in range(5)] == [1, 2, 3, 4, 5]

    def test_prepend_does_not_starve_across_rounds(self):
        """A request returned to the queue multiple times still comes first."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        req_a = create_sjf_request_item(1, prompt_len=100)
        q.add_request(req_a)

        for _ in range(3):
            popped = q.pop_request()
            q.add_request(create_sjf_request_item(99, prompt_len=100))
            q.prepend_request(popped)

        assert q.pop_request().id == 1

    def test_prepend_preserves_aging_benefit(self):
        """Prepended request retains its original enqueue_time for aging."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0.005)

        # time=0: add long request (8000 tokens)
        with patch("tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue.time") as mock_time:
            mock_time.monotonic.return_value = 0.0
            q.add_request(create_sjf_request_item(1, prompt_len=8000))

        # time=0: add short request (2000 tokens)
        with patch("tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue.time") as mock_time:
            mock_time.monotonic.return_value = 0.0
            q.add_request(create_sjf_request_item(2, prompt_len=2000))

        # time=190: pop request 2 (shorter, comes first)
        with patch("tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue.time") as mock_time:
            mock_time.monotonic.return_value = 190.0
            popped = q.pop_request()
            assert popped.id == 2

        # time=190: prepend request 2 back (couldn't schedule)
        # It should keep its original enqueue_time of 0.0
        with patch("tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue.time") as mock_time:
            mock_time.monotonic.return_value = 190.0
            q.prepend_request(popped)

        # time=200: both requests should have score=0 (both enqueued at t=0,
        # 200s elapsed). If prepend reset the clock, request 2 would have
        # score = 2000 * (1 - 0.005*10) = 2000*0.95 = 1900 instead of 0.
        with patch("tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue.time") as mock_time:
            mock_time.monotonic.return_value = 200.0
            # Both at score 0 — prepended request 2 has negative counter,
            # so it sorts first
            result = q.pop_request()
            assert result.id == 2

    def test_prepend_respects_sjf_ordering(self):
        """Prepend does not unconditionally put request at front if shorter
        jobs exist."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        q.add_request(create_sjf_request_item(1, prompt_len=50))
        # Prepend a longer request
        q.prepend_request(create_sjf_request_item(2, prompt_len=200))
        # Shorter job still comes first
        assert q.pop_request().id == 1
        assert q.pop_request().id == 2

    # ------------------------------------------------------------------
    # remove_by_ids
    # ------------------------------------------------------------------

    def test_remove_by_ids_removes_correct_items(self):
        """remove_by_ids removes exactly the specified requests."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        for i, plen in enumerate([300, 200, 100, 50, 25]):
            q.add_request(create_sjf_request_item(i, prompt_len=plen))

        q.remove_by_ids({1, 3})  # remove prompt_len 200 and 50
        assert len(q) == 3
        remaining = [q.pop_request().id for _ in range(3)]
        assert remaining == [4, 2, 0]  # 25, 100, 300

    def test_remove_by_ids_nonexistent_is_noop(self):
        """remove_by_ids with unknown IDs does not change the queue."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        q.add_request(create_sjf_request_item(1, prompt_len=100))
        q.add_request(create_sjf_request_item(2, prompt_len=200))

        q.remove_by_ids({99, 100})
        assert len(q) == 2

    def test_remove_all_ids_leaves_empty_queue(self):
        """remove_by_ids can empty the queue entirely."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        q.add_request(create_sjf_request_item(1, prompt_len=100))
        q.add_request(create_sjf_request_item(2, prompt_len=200))

        q.remove_by_ids({1, 2})
        assert len(q) == 0
        assert not q

    def test_sjf_order_preserved_after_remove(self):
        """SJF ordering is intact after remove_by_ids."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        for i, plen in enumerate([300, 200, 100, 50]):
            q.add_request(create_sjf_request_item(i, prompt_len=plen))

        q.remove_by_ids({1})  # remove prompt_len=200
        assert [q.pop_request().id for _ in range(3)] == [3, 2, 0]  # 50, 100, 300

    # ------------------------------------------------------------------
    # pop / peek edge cases
    # ------------------------------------------------------------------

    def test_pop_from_empty_raises(self):
        """pop_request raises IndexError on an empty queue."""
        with pytest.raises(IndexError):
            SJFWaitingQueue(cache_aware=False, aging_factor=0).pop_request()

    def test_peek_from_empty_raises(self):
        """peek_request raises IndexError on an empty queue."""
        with pytest.raises(IndexError):
            SJFWaitingQueue(cache_aware=False, aging_factor=0).peek_request()

    def test_peek_returns_shortest_without_removing(self):
        """peek_request returns the shortest job without removing it."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        q.add_request(create_sjf_request_item(1, prompt_len=200))
        q.add_request(create_sjf_request_item(2, prompt_len=50))

        peeked = q.peek_request()
        assert peeked.id == 2
        assert len(q) == 2  # not consumed

    # ------------------------------------------------------------------
    # bool / len / iter
    # ------------------------------------------------------------------

    def test_bool_empty(self):
        """Empty queue is falsy."""
        assert not SJFWaitingQueue(cache_aware=False, aging_factor=0)

    def test_bool_nonempty(self):
        """Non-empty queue is truthy."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        q.add_request(create_sjf_request_item(1, prompt_len=100))
        assert q

    def test_len_tracks_adds_and_pops(self):
        """__len__ accurately reflects the number of items."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        assert len(q) == 0
        q.add_request(create_sjf_request_item(1, prompt_len=100))
        assert len(q) == 1
        q.add_requests([create_sjf_request_item(i, prompt_len=100) for i in range(2, 5)])
        assert len(q) == 4
        q.pop_request()
        assert len(q) == 3

    def test_iter_yields_in_sjf_order(self):
        """__iter__ yields items in SJF order (shortest first)."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        for i, plen in enumerate([200, 50, 100]):
            q.add_request(create_sjf_request_item(i, prompt_len=plen))

        iterated_ids = [item.id for item in q]
        assert iterated_ids == [1, 2, 0]  # 50, 100, 200

    def test_iter_does_not_consume_items(self):
        """Iterating over the queue does not remove items."""
        q = SJFWaitingQueue(cache_aware=False, aging_factor=0)
        q.add_request(create_sjf_request_item(1, prompt_len=100))
        q.add_request(create_sjf_request_item(2, prompt_len=200))

        _ = list(q)
        assert len(q) == 2

    def test_is_waiting_queue_subclass(self):
        """SJFWaitingQueue is a WaitingQueue."""
        assert isinstance(SJFWaitingQueue(cache_aware=False, aging_factor=0), WaitingQueue)
