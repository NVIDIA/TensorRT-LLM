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
Data types for the Unified SPMD Scheduler.

Public types:
- SchedulerConfig: Scheduler configuration
- ScheduledRequests: Output container for scheduled requests
- BatchScheduleResult: Complete result from batch scheduling

For future types (RankState, RequestScore, GlobalScoreMatrix, etc.),
see types_future.py.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm_request import LlmRequest


@dataclass 
class SchedulerConfig:
    """
    Configuration for the SPMD scheduler.
    
    Attributes:
        max_batch_size: Maximum requests per batch
        max_num_tokens: Maximum tokens per iteration
        max_num_active_requests: Maximum active requests per rank
        enable_attention_dp: Enable Attention Data Parallelism
        enable_adp_balance: Enable ADP context batch balancing
        adp_timeout_iters: Iterations to wait before ADP balance timeout
        adp_batching_wait_iters: Iterations to wait for batch accumulation in ADP
        enable_chunked_prefill: Enable chunked context processing
        chunk_unit_size: Unit size for context chunking
        attn_dp_min_batch: Minimum batch size for ADP (typically = attn_dp_size)
        batch_wait_timeout_ms: Time to wait for batch accumulation
        batch_wait_timeout_iters: Iterations to wait for batch accumulation
        batch_wait_max_tokens_ratio: Token threshold ratio for batch waiting
        enable_disagg: Enable disaggregated serving mode
        is_disagg_context: True if this is a disagg context server
        is_disagg_generation: True if this is a disagg generation server
    """
    max_batch_size: int = 8
    max_num_tokens: int = 8192
    max_num_active_requests: int = 128
    enable_attention_dp: bool = False
    enable_adp_balance: bool = False
    adp_timeout_iters: int = 50
    adp_batching_wait_iters: int = 10
    enable_chunked_prefill: bool = False
    chunk_unit_size: int = 512
    attn_dp_min_batch: int = 1
    batch_wait_timeout_ms: float = 0.0
    batch_wait_timeout_iters: int = 0
    batch_wait_max_tokens_ratio: float = 0.0
    enable_disagg: bool = False
    is_disagg_context: bool = False
    is_disagg_generation: bool = False


@dataclass
class ScheduledRequests:
    """
    Container for requests scheduled in the current iteration.
    
    Aligned with ScheduledRequests in cpp/tensorrt_llm/batch_manager/common.h
    """
    context_requests: List['LlmRequest'] = field(default_factory=list)
    generation_requests: List['LlmRequest'] = field(default_factory=list)
    paused_requests: List['LlmRequest'] = field(default_factory=list)
    
    def all_requests(self) -> List['LlmRequest']:
        """Get all scheduled requests (context + generation)."""
        return self.context_requests + self.generation_requests
    
    def total_count(self) -> int:
        """Get total number of scheduled requests."""
        return len(self.context_requests) + len(self.generation_requests)
    
    @property
    def batch_size(self) -> int:
        """Get batch size (alias for total_count for compatibility)."""
        return self.total_count()
    
    def is_empty(self) -> bool:
        """Check if no requests are scheduled."""
        return len(self.context_requests) == 0 and len(self.generation_requests) == 0


@dataclass
class BatchScheduleResult:
    """Complete result from batch scheduling."""
    scheduled_batch: ScheduledRequests = field(default_factory=ScheduledRequests)
    iter_stats: Optional[dict] = None
    fitting_disagg_gen_init: List['LlmRequest'] = field(default_factory=list)
    num_fitting_requests: int = 0
    # Drafter-related fields for executor to handle model_engine and _prepare_draft_requests
    use_spec_decode: bool = False
    max_draft_tokens: int = 0


# Legacy alias
SchedulerOutput = BatchScheduleResult


@dataclass
class LocalScheduleReport:
    """
    Report from Phase 1 local scheduling.
    
    Contains both:
    - Information for legacy ADP distribution (num_active_requests, num_active_tokens)
    - Information for new capacity-aware distribution (remaining_batch_slots, remaining_token_budget)
    
    This allows the same report to work with both old and new assignment strategies.
    """
    rank: int = 0
    
    # === For legacy ADP distribution (MinHeapADPBalancer) ===
    # These are the total active counts BEFORE scheduling this iteration
    num_active_requests: int = 0  # Total active requests on this rank
    num_active_tokens: int = 0    # Total active tokens on this rank
    
    # === For new capacity-aware distribution ===
    # These are what was scheduled THIS iteration and remaining capacity
    scheduled_request_ids: List[int] = field(default_factory=list)
    remaining_batch_slots: int = 0      # How many more requests can fit
    remaining_token_budget: int = 0     # How many more tokens can fit
    num_context_scheduled: int = 0      # Context requests scheduled this iter
    num_generation_scheduled: int = 0   # Generation requests scheduled this iter
    num_context_tokens: int = 0         # Context tokens scheduled this iter
    num_generation_tokens: int = 0      # Generation tokens scheduled this iter
    
    def to_sync_data(self) -> List[int]:
        """Convert to list of ints for AllGather synchronization."""
        return [
            self.rank,
            self.num_active_requests,
            self.num_active_tokens,
            self.remaining_batch_slots,
            self.remaining_token_budget,
            self.num_context_scheduled,
            self.num_generation_scheduled,
            self.num_context_tokens,
            self.num_generation_tokens,
        ]
    
    @classmethod
    def from_sync_data(cls, data: List[int]) -> 'LocalScheduleReport':
        """Create from synchronized data."""
        return cls(
            rank=data[0],
            num_active_requests=data[1],
            num_active_tokens=data[2],
            remaining_batch_slots=data[3],
            remaining_token_budget=data[4],
            num_context_scheduled=data[5],
            num_generation_scheduled=data[6],
            num_context_tokens=data[7],
            num_generation_tokens=data[8],
        )


