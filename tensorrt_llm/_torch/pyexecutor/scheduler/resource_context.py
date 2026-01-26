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
Scheduling Resource Context for Two-Phase Scheduling.

This module provides the SchedulingResourceContext class that holds mutable
state across multiple schedule calls within a single iteration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_request import LlmRequest


@dataclass
class SchedulingResourceContext:
    """
    Mutable scheduling state shared across multiple schedule calls.
    
    Stores:
    - BlocksManagers for KV cache tracking (created on first use)
    - PEFT/LoRA state (seen task IDs, used pages)
    - Scheduled requests (context + generation, accumulated across phases)
    - Token budget tracking
    """
    
    # BlocksManager instances - created on first use, reused across phases
    _blocks_manager: Optional['NoEvictScheduledBlocksManager'] = None
    _cross_blocks_manager: Optional['NoEvictScheduledBlocksManager'] = None
    _max_util_blocks_manager: Optional['MaxUtilizationScheduledBlocksManager'] = None
    
    # PEFT state
    seen_lora_task_ids: Set[int] = field(default_factory=set)
    used_peft_pages: int = 0
    
    # Block reuse optimization state (for GuaranteedNoEvict and MaxUtilization policies)
    newly_contributed_context_blocks: Set = field(default_factory=set)
    newly_contributed_cross_context_blocks: Set = field(default_factory=set)
    
    # Scheduled requests - accumulated across phases (updated by MicroBatchScheduler)
    context_requests: List['LlmRequest'] = field(default_factory=list)
    generation_requests: List['LlmRequest'] = field(default_factory=list)
    
    # MicroBatch token budget tracking
    scheduled_tokens: int = 0
    
    # --- BlocksManager accessors ---
    
    def get_no_evict_blocks_manager(self, kv_cache_manager) -> 'NoEvictScheduledBlocksManager':
        """Get or create NoEvictScheduledBlocksManager for main KV cache."""
        if self._blocks_manager is None:
            self._blocks_manager = NoEvictScheduledBlocksManager(kv_cache_manager)
        return self._blocks_manager
    
    def get_no_evict_cross_blocks_manager(self, cross_kv_cache_manager) -> 'NoEvictScheduledBlocksManager':
        """Get or create NoEvictScheduledBlocksManager for cross KV cache."""
        if self._cross_blocks_manager is None:
            self._cross_blocks_manager = NoEvictScheduledBlocksManager(cross_kv_cache_manager)
        return self._cross_blocks_manager
    
    def get_max_util_blocks_manager(self, kv_cache_manager, two_steps_look_ahead: bool) -> 'MaxUtilizationScheduledBlocksManager':
        """Get or create MaxUtilizationScheduledBlocksManager."""
        if self._max_util_blocks_manager is None:
            self._max_util_blocks_manager = MaxUtilizationScheduledBlocksManager(
                kv_cache_manager, two_steps_look_ahead)
        return self._max_util_blocks_manager
    
    # --- PEFT helpers ---
    
    def add_lora_task(self, lora_task_id: int, pages: int) -> None:
        """Record a new LoRA task being scheduled."""
        self.seen_lora_task_ids.add(lora_task_id)
        self.used_peft_pages += pages
    
    def is_lora_task_seen(self, lora_task_id: int) -> bool:
        """Check if a LoRA task has already been scheduled."""
        return lora_task_id in self.seen_lora_task_ids
    
    # --- Scheduled requests helpers ---
    
    def get_scheduled_count(self) -> int:
        """Get total number of scheduled requests (context + generation)."""
        return len(self.context_requests) + len(self.generation_requests)
    
    def get_all_scheduled_requests(self) -> List['LlmRequest']:
        """Get all scheduled requests (context + generation)."""
        return self.context_requests + self.generation_requests
    
    def reset(self) -> None:
        """Reset context for a new iteration."""
        self._blocks_manager = None
        self._cross_blocks_manager = None
        self._max_util_blocks_manager = None
        self.seen_lora_task_ids.clear()
        self.used_peft_pages = 0
        self.newly_contributed_context_blocks.clear()
        self.newly_contributed_cross_context_blocks.clear()
        self.context_requests.clear()
        self.generation_requests.clear()
        self.scheduled_tokens = 0
    
    def __repr__(self) -> str:
        return (
            f"SchedulingResourceContext("
            f"ctx={len(self.context_requests)}, "
            f"gen={len(self.generation_requests)}, "
            f"tokens={self.scheduled_tokens}, "
            f"peft_pages={self.used_peft_pages}, "
            f"lora_tasks={len(self.seen_lora_task_ids)})"
        )


class NoEvictScheduledBlocksManager:
    """
    Tracks available blocks per window size for GUARANTEED_NO_EVICT scheduling.
    C++ reference: cpp/tensorrt_llm/batch_manager/scheduledBlocksManager.h:29-62
    """

    def __init__(self, kv_cache_manager):
        self.kv_cache_manager = kv_cache_manager
        stats = kv_cache_manager.get_kv_cache_stats()
        self.available_blocks: dict[int, int] = dict(stats.num_free_blocks_per_window_size)

    def decrement_reserved_blocks(self, req: 'LlmRequest') -> None:
        """Decrement available blocks by blocks needed to complete this request."""
        for window_size in self.available_blocks:
            needed = self.kv_cache_manager.get_remaining_blocks_to_completion(req, window_size)
            self.available_blocks[window_size] -= needed

    def enough_available_blocks(self, req: 'LlmRequest') -> bool:
        """Check if there are enough available blocks for this request."""
        return all(
            self.kv_cache_manager.get_remaining_blocks_to_completion(req, ws) <= avail
            for ws, avail in self.available_blocks.items()
        )


class MaxUtilizationScheduledBlocksManager:
    """
    Tracks scheduled blocks per window size for MAX_UTILIZATION scheduling.
    C++ reference: cpp/tensorrt_llm/batch_manager/scheduledBlocksManager.h:64-117
    """

    def __init__(self, kv_cache_manager, two_steps_look_ahead: bool):
        self.kv_cache_manager = kv_cache_manager
        self.two_steps_look_ahead = two_steps_look_ahead
        window_sizes = set(kv_cache_manager.max_attention_window_vec)
        self.num_scheduled_blocks: dict[int, int] = {ws: 0 for ws in window_sizes}

    def prepare_blocks_if_schedulable(self, req: 'LlmRequest') -> Optional[dict[int, int]]:
        """
        Check if request can be scheduled and return new block counts if so.
        Returns None if request cannot fit.
        """
        from tensorrt_llm.logger import logger
        
        blocks_if_scheduled = {}
        for window_size, num_scheduled in self.num_scheduled_blocks.items():
            required = self.kv_cache_manager.get_needed_blocks_one_step(
                req, self.two_steps_look_ahead, window_size)
            logger.debug(
                f"MaxUtilizationScheduler: request ID {req.request_id} "
                f"required blocks {required} for {window_size} window size")
            scheduled_total = num_scheduled + required
            has_free = self.kv_cache_manager.scheduling_has_free_blocks(scheduled_total, window_size)
            if not has_free:
                return None
            blocks_if_scheduled[window_size] = scheduled_total
        return blocks_if_scheduled

    def update_scheduled_blocks(self, blocks: dict[int, int]) -> None:
        """Update the scheduled blocks after successfully scheduling a request."""
        from tensorrt_llm.logger import logger
        
        assert len(blocks) == len(self.num_scheduled_blocks), \
            f"Block count mismatch: {len(blocks)} vs {len(self.num_scheduled_blocks)}"
        for window_size, blocks_if_scheduled in blocks.items():
            logger.debug(
                f"MaxUtilizationScheduler: scheduled blocks {blocks_if_scheduled} "
                f"for window size {window_size}")
            self.num_scheduled_blocks[window_size] = blocks_if_scheduled
