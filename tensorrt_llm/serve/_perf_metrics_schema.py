# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union

from typing_extensions import NotRequired, TypedDict


class StepMetrics(TypedDict):
    forward_start_time: float
    forward_end_time: float
    sample_start_time: float
    sample_end_time: float
    gpu_forward_time: float
    gpu_sample_time: float
    token_time: NotRequired[float]
    scheduled_time: NotRequired[float]
    prev_batch_token_time: NotRequired[float]
    iter: NotRequired[int]


class TimeBreakdownMetrics(TypedDict):
    step_metrics: NotRequired[List[StepMetrics]]
    ctx_chunk_metrics: NotRequired[List[StepMetrics]]
    ctx_gpu_forward_time: NotRequired[float]
    ctx_gpu_sample_time: NotRequired[float]


class TimingMetrics(TypedDict):
    arrival_time: Optional[float]
    first_scheduled_time: NotRequired[Optional[float]]
    first_token_time: NotRequired[Optional[float]]
    last_token_time: Optional[float]
    server_arrival_time: NotRequired[Optional[float]]
    server_first_token_time: NotRequired[Optional[float]]
    kv_cache_size: NotRequired[int]
    kv_cache_transfer_start: NotRequired[Optional[float]]
    kv_cache_transfer_end: NotRequired[Optional[float]]


class KvCacheMetrics(TypedDict):
    num_total_allocated_blocks: int
    num_new_allocated_blocks: int
    num_reused_blocks: int
    num_missed_blocks: int


class SpeculativeDecodingMetrics(TypedDict):
    acceptance_rate: float
    total_accepted_draft_tokens: int
    total_draft_tokens: int


class PerfMetrics(TypedDict):
    timing_metrics: TimingMetrics
    first_iter: NotRequired[int]
    last_iter: NotRequired[int]
    kv_cache_metrics: NotRequired[KvCacheMetrics]
    speculative_decoding: NotRequired[SpeculativeDecodingMetrics]


class WorkerPerfMetrics(TypedDict):
    request_id: Union[int, str]
    perf_metrics: PerfMetrics
    ctx_request_id: NotRequired[int]
    time_breakdown_metrics: NotRequired[TimeBreakdownMetrics]


class WorkerPerfMetricsRecord(WorkerPerfMetrics):
    status: str
    disagg_request_id: NotRequired[int]


class DisaggPerfMetricsRecord(TypedDict):
    ctx_server: str
    gen_server: str
    disagg_server_arrival_time: float
    disagg_ctx_dispatch_time: float
    disagg_server_first_token_time: float
    status: str
    disagg_request_id: NotRequired[int]
    ctx_perf_metrics: NotRequired[WorkerPerfMetrics]
    gen_perf_metrics: NotRequired[WorkerPerfMetrics]


PerfMetricsRecord = Union[WorkerPerfMetricsRecord, DisaggPerfMetricsRecord]
