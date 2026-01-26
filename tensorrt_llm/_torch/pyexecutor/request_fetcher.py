# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Request Fetcher - Encapsulates queue fetch operations.

This class owns a RequestQueue and provides fetch operations with
timeout and batch wait support.

Broadcasting and merging are handled by request_utils.py.
ADP distribution is handled by distribution_strategy.py.
"""

import datetime
import queue
import time
from dataclasses import dataclass
from typing import List, Optional

from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm._torch.pyexecutor.request_queue import RequestQueue, RequestQueueItem


@dataclass
class FetcherConfig:
    """Configuration for RequestFetcher."""
    max_batch_size: int = 8
    batch_wait_timeout_ms: float = 0.0


class RequestFetcher:
    """
    Encapsulates queue fetch operations.
    
    Owns a RequestQueue and provides:
    - Fetching with timeout
    - Batch wait (accumulate requests up to max_batch_size)
    - Queue state queries (is_empty, can_enqueue)
    """
    
    def __init__(self, request_queue: RequestQueue, config: FetcherConfig):
        """
        Initialize the fetcher.
        
        Args:
            request_queue: The queue to fetch from
            config: Fetcher configuration
        """
        self.queue = request_queue
        self.config = config
        self._disable_mpi = mpi_disabled()
    
    def fetch(self, idle: bool) -> List[RequestQueueItem]:
        """
        Fetch requests from the queue.
        
        Args:
            idle: Whether the system is idle (no active requests, no waiting requests).
                  If idle, will wait longer for requests; otherwise non-blocking.
        
        Returns:
            List of fetched RequestQueueItems
        """
        # Calculate timeout based on idle state
        if idle:
            # In Ray path (TLLM_DISABLE_MPI=1), use a periodic heartbeat timeout so rank 0
            # reaches the broadcast path regularly to prevent trtllm-serve timeout when idle.
            timeout = datetime.timedelta(seconds=1200) if self._disable_mpi else None
        else:
            timeout = datetime.timedelta(0)
        
        items = []
        timeout_secs = timeout.total_seconds() if timeout is not None else None
        
        try:
            if self.queue.is_empty() and (timeout_secs is None or timeout_secs > 0):
                # If queue is empty and want to wait, wait
                items.append(self.queue.get(timeout=timeout_secs))
            else:
                # If not empty or don't want to wait, just return all items in queue
                while True:
                    items.append(self.queue.get_nowait())
        except queue.Empty:
            pass
        
        # Batch wait logic
        if self.config.batch_wait_timeout_ms == 0:
            return items
        
        if len(items) >= self.config.max_batch_size:
            return items
        
        deadline = time.monotonic() + self.config.batch_wait_timeout_ms / 1000.0
        while len(items) < self.config.max_batch_size:
            remaining_timeout = deadline - time.monotonic()
            
            if remaining_timeout <= 0:
                break
            
            try:
                items.append(self.queue.get(timeout=remaining_timeout))
            except queue.Empty:
                break
        
        return items
    
    def can_enqueue(self) -> bool:
        """Check if requests can be enqueued (delegates to queue)."""
        return self.queue.can_enqueue_request()