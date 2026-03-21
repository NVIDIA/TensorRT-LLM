# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
- SJFWaitingQueue operations
- WaitingQueue abstract interface
- create_waiting_queue factory function
"""

import time
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.scheduler import (
    FCFSWaitingQueue,
    SJFWaitingQueue,
    WaitingQueue,
    create_waiting_queue,
)
from tensorrt_llm.llmapi.llm_args import SjfConfig, WaitingQueuePolicy


def create_mock_request_item(request_id: int) -> RequestQueueItem:
    """Create a mock RequestQueueItem for testing."""
    mock_request = Mock()
    return RequestQueueItem(request_id, mock_request)


def create_sjf_request_item(request_id: int,
                             prompt_len: int,
                             arrival_time: float = None) -> RequestQueueItem:
    """Create a RequestQueueItem with SJF-relevant fields set."""
    mock_request = Mock()
    mock_request.input_token_ids = list(range(prompt_len))
    mock_request.py_arrival_time = arrival_time
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


class TestSJFWaitingQueue:
    """Tests for SJFWaitingQueue."""

    def test_short_requests_pop_first(self):
        """Short requests should be popped before long requests."""
        now = time.time()
        queue = SJFWaitingQueue()
        # Long request (10000 tokens) added first
        queue.add_request(create_sjf_request_item(1, 10000, now))
        # Short request (100 tokens) added second
        queue.add_request(create_sjf_request_item(2, 100, now))

        # Short request should pop first despite arriving second
        assert queue.pop_request().id == 2
        assert queue.pop_request().id == 1

    def test_equal_length_pops_by_wait_time(self):
        """Equal-length requests should pop in order of wait time (older first)."""
        now = time.time()
        queue = SJFWaitingQueue()
        queue.add_request(create_sjf_request_item(1, 1000, now - 10))
        queue.add_request(create_sjf_request_item(2, 1000, now - 1))

        # Older request (waited 10s) should pop first
        assert queue.pop_request().id == 1
        assert queue.pop_request().id == 2

    def test_aging_prevents_starvation(self):
        """Long requests with enough wait time should eventually get priority."""
        now = time.time()
        # Use config where time aging is strong
        config = SjfConfig(length_weight=0.5, time_weight=0.5,
                           time_median=1.0)
        queue = SJFWaitingQueue(config)

        # Long request that has been waiting 100 seconds
        queue.add_request(create_sjf_request_item(1, 50000, now - 100))
        # Short request that just arrived
        queue.add_request(create_sjf_request_item(2, 100, now))

        # Long request should pop first due to massive wait time
        # score_long = 0.5 * 1/(1+50000/32768) + 0.5 * 100/1.0 = ~0.20 + 50 = ~50.2
        # score_short = 0.5 * 1/(1+100/32768) + 0.5 * 0/1.0 = ~0.498 + 0 = ~0.498
        assert queue.pop_request().id == 1

    def test_peek_pop_consistency(self):
        """peek_request and pop_request should return the same item."""
        now = time.time()
        queue = SJFWaitingQueue()
        queue.add_request(create_sjf_request_item(1, 5000, now))
        queue.add_request(create_sjf_request_item(2, 100, now))
        queue.add_request(create_sjf_request_item(3, 20000, now))

        peeked = queue.peek_request()
        popped = queue.pop_request()
        assert peeked.id == popped.id

    def test_add_request(self):
        """Test adding a single request."""
        queue = SJFWaitingQueue()
        item = create_sjf_request_item(1, 100, time.time())
        queue.add_request(item)
        assert len(queue) == 1

    def test_add_requests(self):
        """Test adding multiple requests."""
        now = time.time()
        queue = SJFWaitingQueue()
        items = [create_sjf_request_item(i, 100 * (i + 1), now) for i in range(3)]
        queue.add_requests(items)
        assert len(queue) == 3

    def test_pop_from_empty_queue(self):
        """Test that pop_request raises IndexError on empty queue."""
        queue = SJFWaitingQueue()
        with pytest.raises(IndexError):
            queue.pop_request()

    def test_peek_from_empty_queue(self):
        """Test that peek_request raises IndexError on empty queue."""
        queue = SJFWaitingQueue()
        with pytest.raises(IndexError):
            queue.peek_request()

    def test_prepend_request(self):
        """Prepended request should be popped first."""
        now = time.time()
        queue = SJFWaitingQueue()
        # Add a short request
        queue.add_request(create_sjf_request_item(1, 100, now))
        # Prepend a long request — should still pop first
        queue.prepend_request(create_sjf_request_item(2, 99999, now))

        assert queue.pop_request().id == 2
        assert queue.pop_request().id == 1

    def test_prepend_requests_order(self):
        """Prepended requests maintain extendleft semantics (reversed)."""
        now = time.time()
        queue = SJFWaitingQueue()
        queue.add_request(create_sjf_request_item(3, 100, now))

        # Simulate caller pattern: reversed([item1, item2])
        items = [create_sjf_request_item(i, 100, now) for i in [1, 2]]
        queue.prepend_requests(items)

        # After extendleft-style reversal of [1, 2], order is: 2, 1, 3
        assert queue.pop_request().id == 2
        assert queue.pop_request().id == 1
        assert queue.pop_request().id == 3

    def test_remove_by_ids(self):
        """Test removing requests by their IDs."""
        now = time.time()
        queue = SJFWaitingQueue()
        for i in range(5):
            queue.add_request(create_sjf_request_item(i, 100, now))

        queue.remove_by_ids({1, 3})
        assert len(queue) == 3
        remaining_ids = [item.id for item in queue]
        assert set(remaining_ids) == {0, 2, 4}

    def test_remove_from_prepended(self):
        """Test removing IDs that are in the prepended list."""
        now = time.time()
        queue = SJFWaitingQueue()
        queue.add_request(create_sjf_request_item(1, 100, now))
        queue.prepend_request(create_sjf_request_item(2, 100, now))

        queue.remove_by_ids({2})
        assert len(queue) == 1
        assert queue.pop_request().id == 1

    def test_bool_empty_queue(self):
        """Test bool conversion for empty queue."""
        queue = SJFWaitingQueue()
        assert not queue

    def test_bool_nonempty_queue(self):
        """Test bool conversion for non-empty queue."""
        queue = SJFWaitingQueue()
        queue.add_request(create_sjf_request_item(1, 100, time.time()))
        assert queue

    def test_bool_only_prepended(self):
        """Queue with only prepended items should be truthy."""
        queue = SJFWaitingQueue()
        queue.prepend_request(create_sjf_request_item(1, 100, time.time()))
        assert queue
        assert len(queue) == 1

    def test_len(self):
        """Test length includes both prepended and regular requests."""
        now = time.time()
        queue = SJFWaitingQueue()
        queue.add_request(create_sjf_request_item(1, 100, now))
        queue.add_request(create_sjf_request_item(2, 100, now))
        queue.prepend_request(create_sjf_request_item(3, 100, now))
        assert len(queue) == 3

    def test_iter(self):
        """Test iteration returns prepended first, then sorted requests."""
        now = time.time()
        queue = SJFWaitingQueue()
        # Add long then short
        queue.add_request(create_sjf_request_item(1, 10000, now))
        queue.add_request(create_sjf_request_item(2, 100, now))
        # Prepend one
        queue.prepend_request(create_sjf_request_item(3, 50000, now))

        iterated_ids = [item.id for item in queue]
        # Prepended first (3), then sorted by score: short (2), long (1)
        assert iterated_ids == [3, 2, 1]
        # Iteration should not consume items
        assert len(queue) == 3

    def test_is_waiting_queue_subclass(self):
        """Test that SJFWaitingQueue is a WaitingQueue."""
        queue = SJFWaitingQueue()
        assert isinstance(queue, WaitingQueue)

    def test_custom_config(self):
        """Test that custom SjfConfig parameters are respected."""
        now = time.time()
        # Heavy length weight, no time weight
        config = SjfConfig(length_weight=1.0, time_weight=0.0)
        queue = SJFWaitingQueue(config)

        # Even though request 1 has waited much longer, length_weight=1, time_weight=0
        queue.add_request(create_sjf_request_item(1, 10000, now - 1000))
        queue.add_request(create_sjf_request_item(2, 100, now))

        # Short request should still win since time_weight=0
        assert queue.pop_request().id == 2

    def test_no_arrival_time_fallback(self):
        """Requests without py_arrival_time should still work."""
        queue = SJFWaitingQueue()
        # Create request without arrival time
        item = create_sjf_request_item(1, 100, arrival_time=None)
        queue.add_request(item)
        assert len(queue) == 1
        assert queue.pop_request().id == 1

    def test_dirty_flag_reset_on_add(self):
        """Adding a request after sort should trigger re-sort on next peek."""
        now = time.time()
        queue = SJFWaitingQueue()
        queue.add_request(create_sjf_request_item(1, 10000, now))
        queue.add_request(create_sjf_request_item(2, 5000, now))

        # Trigger sort
        assert queue.peek_request().id == 2

        # Add a shorter request — should become new best
        queue.add_request(create_sjf_request_item(3, 100, now))
        assert queue.peek_request().id == 3


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

    def test_create_sjf_queue(self):
        """Test creating SJF queue."""
        queue = create_waiting_queue(WaitingQueuePolicy.SJF)
        assert isinstance(queue, SJFWaitingQueue)

    def test_create_sjf_queue_with_config(self):
        """Test creating SJF queue with custom config."""
        config = SjfConfig(length_median=1024, time_median=10.0)
        queue = create_waiting_queue(WaitingQueuePolicy.SJF,
                                     sjf_config=config)
        assert isinstance(queue, SJFWaitingQueue)
        assert queue._config.length_median == 1024
        assert queue._config.time_median == 10.0
