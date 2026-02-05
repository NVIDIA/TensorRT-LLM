"""Tests for WaitingQueue implementations.

This module tests the waiting queue functionality including:
- FCFSWaitingQueue operations
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
from tensorrt_llm.llmapi.llm_args import WaitingQueuePolicy


def create_mock_request_item(request_id: int) -> RequestQueueItem:
    """Create a mock RequestQueueItem for testing."""
    mock_request = Mock()
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

        # Prepend items [1, 2] - note: extendleft reverses order
        queue.prepend_requests([create_mock_request_item(i) for i in [1, 2]])

        # After extendleft([1, 2]), order is: 2, 1, 3
        assert queue.pop_request().id == 2
        assert queue.pop_request().id == 1
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
