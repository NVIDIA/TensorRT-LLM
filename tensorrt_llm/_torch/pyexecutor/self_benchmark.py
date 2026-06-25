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

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from tensorrt_llm.bindings.executor import OutputConfig, Request, SamplingConfig
from tensorrt_llm.logger import logger

from .executor_request_queue import RequestQueueItem
from .resource_manager import ResourceManagerType
from .scheduler import ScheduledRequests, WaitingQueue

if TYPE_CHECKING:
    from .llm_request import LlmRequest
    from .py_executor import PyExecutor

_BENCHMARK_REQUEST_ID_BASE = 900_000_000


@dataclass
class BenchmarkPoint:
    point_type: Literal["warmup", "prefill_seed", "prefill", "decode"]
    index: int
    isl: int = 0
    kv_read_tokens: int = 0
    context_length: int = 0
    batch_size: int = 1
    cache_salt_id: Optional[int] = None


@dataclass
class BenchmarkPointResult:
    point: BenchmarkPoint
    stats: list[dict] = field(default_factory=list)
    skipped_reason: Optional[str] = None
    observed_kv_read_tokens: Optional[int] = None
    cache_hit_validated: Optional[bool] = None


class SelfBenchmark:
    """Runs synthetic startup benchmark points inside the PyExecutor loop."""

    def __init__(self, executor: "PyExecutor") -> None:
        self._executor = executor
        self.config = executor.llm_args.self_benchmark_config
        self._grid = self._build_grid()
        self._grid_index = 0
        self._current: Optional[BenchmarkPointResult] = None
        self._results: list[BenchmarkPointResult] = []
        self._done = self.config is None
        self._timed_out = False
        self._started_at = time.monotonic()
        if not self._done:
            logger.info("Self-benchmark enabled: %s", self.config)
            logger.info("Self-benchmark grid: %d point(s)", len(self._grid))

    @property
    def active(self) -> bool:
        return not self._done

    def make_prefill_queue_items(
            self,
            active_requests: list["LlmRequest"],
            waiting_queue: WaitingQueue) -> list[RequestQueueItem]:
        if not self._can_start_next_point(active_requests, waiting_queue):
            return []
        point = self._peek_next_point()
        if point is None:
            self._finish()
            return []
        if point.point_type not in ("warmup", "prefill_seed", "prefill"):
            return []

        self._start_point(point)
        request = self._make_prefill_request(point)
        return [
            RequestQueueItem(
                id=self._request_id(point.index, 0),
                request=request,
                child_req_ids=None)
        ]

    def make_decode_requests(self, active_requests: list["LlmRequest"],
                             waiting_queue: WaitingQueue) -> list["LlmRequest"]:
        if not self._can_start_next_point(active_requests, waiting_queue):
            return []
        point = self._peek_next_point()
        if point is None:
            self._finish()
            return []
        if point.point_type != "decode":
            return []

        self._start_point(point)
        kv_cache_manager = self._executor.resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER)
        if kv_cache_manager is None:
            self._skip_current_point("KV cache manager is not available")
            return []
        draft_kv_cache_manager = self._executor.resource_manager.get_resource_manager(
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER)
        token_num = max(1, point.context_length + 1)
        request_ids = [
            self._request_id(point.index, offset)
            for offset in range(point.batch_size)
        ]
        requests = kv_cache_manager.add_dummy_requests(
            request_ids=request_ids,
            token_nums=[token_num] * point.batch_size,
            is_gen=True,
            max_num_draft_tokens=self._executor.max_total_draft_tokens,
            kv_reserve_draft_tokens=getattr(self._executor.model_engine,
                                            "max_draft_loop_tokens",
                                            self._executor.max_total_draft_tokens),
            use_mrope=getattr(self._executor.model_engine, "use_mrope", False),
            max_beam_width=self._executor.max_beam_width,
            draft_kv_cache_manager=draft_kv_cache_manager)
        if requests is None:
            self._skip_current_point("insufficient KV cache for synthetic decode")
            return []
        for request in requests:
            self._mark_benchmark_request(request, point)
        return requests

    def observe_iteration(self, scheduled_batch: ScheduledRequests,
                          stats: dict) -> bool:
        """Record benchmark stats for benchmark-only batches.

        Returns:
            True when the iteration belonged to self-benchmarking and should
            not be appended to the normal public iteration-stats buffer.
        """
        if self._current is None:
            return False

        requests = scheduled_batch.all_requests()
        if not requests:
            return False
        if not all(self._is_benchmark_request(req)
                   or bool(getattr(req, "is_dummy", False))
                   for req in requests):
            logger.warning(
                "Self-benchmark saw a mixed batch; ignoring it for benchmark output."
            )
            return False

        self._sanitize_queue_counters(stats)
        self._record_cache_hit_validation(requests, stats)
        if self._should_record_current_point():
            self._current.stats.append(stats)

        if self._point_is_complete(stats):
            self._finish_current_point()

        return True

    def write_output(self) -> None:
        if self.config is None:
            return
        output_path = self._rank_output_path(self.config.output_path)
        output = {
            "config": self.config.model_dump(),
            "limits": self._limits(),
            "timed_out": self._timed_out,
            "results": [
                {
                    "point": asdict(result.point),
                    "iteration_stats": result.stats,
                    "skipped_reason": result.skipped_reason,
                    "observed_kv_read_tokens": result.observed_kv_read_tokens,
                    "cache_hit_validated": result.cache_hit_validated,
                } for result in self._results
            ],
        }
        tmp_path = f"{output_path}.tmp"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(output, f, indent=2)
        os.replace(tmp_path, output_path)
        logger.info("Self-benchmark results written to %s (%d point(s))",
                    output_path, len(self._results))

    def _build_grid(self) -> list[BenchmarkPoint]:
        if self.config is None:
            return []
        grid: list[BenchmarkPoint] = []
        next_index = 0
        warmup_isl = min(8, self._max_prefill_isl())
        for _ in range(self.config.warmup_iterations):
            grid.append(
                BenchmarkPoint(point_type="warmup",
                               index=next_index,
                               isl=warmup_isl))
            next_index += 1

        if self.config.mode in ("prefill", "agg"):
            for isl in self._sample_values(self._max_prefill_isl(),
                                           self.config.prefill_isl_granularity):
                for kv_read_tokens in self._kv_read_values_for_isl(isl):
                    cache_salt_id = self._cache_salt_id(next_index)
                    if kv_read_tokens > 0:
                        grid.append(
                            BenchmarkPoint(point_type="prefill_seed",
                                           index=next_index,
                                           isl=kv_read_tokens,
                                           kv_read_tokens=kv_read_tokens,
                                           cache_salt_id=cache_salt_id))
                        next_index += 1
                    grid.append(
                        BenchmarkPoint(point_type="prefill",
                                       index=next_index,
                                       isl=isl,
                                       kv_read_tokens=kv_read_tokens,
                                       cache_salt_id=cache_salt_id))
                    next_index += 1

        if self.config.mode in ("decode", "agg"):
            context_values = self._sample_values(
                self._max_decode_context_length(),
                self.config.decode_context_granularity)
            batch_values = self._sample_values(self._max_decode_batch_size(),
                                               self.config.decode_batch_granularity)
            for context_length in context_values:
                for batch_size in batch_values:
                    grid.append(
                        BenchmarkPoint(point_type="decode",
                                       index=next_index,
                                       context_length=context_length,
                                       batch_size=batch_size))
                    next_index += 1
        return grid

    def _make_prefill_request(self, point: BenchmarkPoint) -> Request:
        output_config = OutputConfig()
        output_config.return_perf_metrics = True
        cache_salt_id = point.cache_salt_id
        if cache_salt_id is None:
            cache_salt_id = self._cache_salt_id(point.index)
        request = Request(
            input_token_ids=[1] * max(1, point.isl),
            max_tokens=1,
            streaming=False,
            sampling_config=SamplingConfig(),
            end_id=-1,
            pad_id=0,
            output_config=output_config,
            return_all_generated_tokens=False,
            cache_salt=str(cache_salt_id))
        request.py_is_self_benchmark_request = True
        request.py_self_benchmark_point_id = point.index
        return request

    def _mark_benchmark_request(self, request: "LlmRequest",
                                point: BenchmarkPoint) -> None:
        request.is_self_benchmark_request = True
        request.py_self_benchmark_point_id = point.index

    def _can_start_next_point(self, active_requests: list["LlmRequest"],
                              waiting_queue: WaitingQueue) -> bool:
        if self._done or self._current is not None:
            return False
        if getattr(self._executor, "is_shutdown", False):
            logger.info(
                "Self-benchmark stopping early because shutdown was requested."
            )
            self._finish()
            return False
        if self.config is not None and (
                time.monotonic() - self._started_at) >= self.config.timeout_s:
            self._timed_out = True
            logger.warning(
                "Self-benchmark timed out after %ds; writing partial results.",
                self.config.timeout_s)
            self._finish()
            return False
        return not active_requests and len(waiting_queue) == 0

    def _peek_next_point(self) -> Optional[BenchmarkPoint]:
        if self._grid_index >= len(self._grid):
            return None
        return self._grid[self._grid_index]

    def _start_point(self, point: BenchmarkPoint) -> None:
        self._grid_index += 1
        self._current = BenchmarkPointResult(point=point)
        logger.debug("Starting self-benchmark point: %s", point)

    def _finish_current_point(self) -> None:
        if self._current is None:
            return
        if self._should_record_current_point():
            self._results.append(self._current)
        self._current = None

    def _skip_current_point(self, reason: str) -> None:
        if self._current is None:
            return
        logger.warning("Skipping self-benchmark point %s: %s",
                       self._current.point, reason)
        self._current.skipped_reason = reason
        if self._should_record_current_point():
            self._results.append(self._current)
        self._current = None

    def _finish(self) -> None:
        if self._current is not None:
            self._finish_current_point()
        self._done = True
        self.write_output()
        logger.info("Self-benchmark completed in %.3fs",
                    time.monotonic() - self._started_at)

    def _point_is_complete(self, stats: dict) -> bool:
        if self._current is None:
            return False
        if self._current.point.point_type in ("warmup", "prefill_seed",
                                              "prefill"):
            return self._num_context_requests(stats) > 0
        if self._current.point.point_type == "decode":
            return self._num_decode_requests(stats) > 0
        return False

    def _record_cache_hit_validation(self, requests: list["LlmRequest"],
                                     stats: dict) -> None:
        if self._current is None:
            return
        point = self._current.point
        if point.point_type != "prefill" or point.kv_read_tokens <= 0:
            return
        observed = self._observed_cached_tokens(requests, point)
        self._current.observed_kv_read_tokens = observed
        validated = observed is not None and observed >= point.kv_read_tokens
        self._current.cache_hit_validated = validated
        stats["selfBenchmark"] = {
            "expectedKvReadTokens": point.kv_read_tokens,
            "observedCachedTokens": observed,
            "cacheHitValidated": validated,
        }
        if not validated:
            logger.warning("Self-benchmark cache-hit validation failed for "
                           "point %s: observed cached_tokens=%s",
                           point, observed)

    @staticmethod
    def _observed_cached_tokens(requests: list["LlmRequest"],
                                point: BenchmarkPoint) -> Optional[int]:
        observed = [
            int(getattr(req, "cached_tokens"))
            for req in requests
            if (getattr(req, "py_self_benchmark_point_id", None) == point.index
                and hasattr(req, "cached_tokens"))
        ]
        return max(observed) if observed else None

    def _should_record_current_point(self) -> bool:
        return (self._current is not None
                and self._current.point.point_type
                not in ("warmup", "prefill_seed"))

    @staticmethod
    def _sanitize_queue_counters(stats: dict) -> None:
        stats["numQueuedRequests"] = 0
        inflight_stats = stats.get("inflightBatchingStats", {})
        inflight_stats["numQueuedContextRequests"] = 0
        inflight_stats["numQueuedCtxTokens"] = 0
        inflight_stats["numQueuedGenRequests"] = 0
        inflight_stats["numQueuedGenKvTokens"] = 0

    @staticmethod
    def _num_context_requests(stats: dict) -> int:
        return int(stats.get("inflightBatchingStats", {}).get(
            "numContextRequests", 0))

    @staticmethod
    def _num_decode_requests(stats: dict) -> int:
        return int(stats.get("inflightBatchingStats", {}).get(
            "numGenRequests", 0))

    @staticmethod
    def _sample_values(max_value: int, granularity: int) -> list[int]:
        max_value = max(1, int(max_value))
        if granularity <= 1:
            return [max_value]
        values = {
            round(1 + i * (max_value - 1) / (granularity - 1))
            for i in range(granularity)
        }
        return sorted(max(1, min(max_value, int(value))) for value in values)

    def _kv_read_values_for_isl(self, isl: int) -> list[int]:
        block_size = self._tokens_per_block()
        if block_size <= 0 or not self._enable_block_reuse():
            return [0]
        max_read_tokens = ((max(0, isl - 1)) // block_size) * block_size
        if max_read_tokens == 0:
            return [0]
        return self._sample_block_aligned_values(
            max_read_tokens, self.config.prefill_kv_read_granularity,
            block_size)

    @staticmethod
    def _sample_block_aligned_values(max_value: int, granularity: int,
                                     block_size: int) -> list[int]:
        block_values = list(range(0, max_value + 1, block_size))
        if granularity <= 1:
            return [0]
        if len(block_values) <= granularity:
            return block_values
        sampled_indices = {
            round(i * (len(block_values) - 1) / (granularity - 1))
            for i in range(granularity)
        }
        return [block_values[i] for i in sorted(sampled_indices)]

    def _max_prefill_isl(self) -> int:
        return self._bounded_length(default=1)

    def _max_decode_context_length(self) -> int:
        return max(1, self._bounded_length(default=2) - 1)

    def _bounded_length(self, default: int) -> int:
        candidates = []
        for value in (
                getattr(self._executor, "max_input_len", None),
                getattr(self._executor, "max_seq_len", None),
                getattr(self._executor, "max_num_tokens", None),
        ):
            if isinstance(value, int) and 0 < value < 2**30:
                candidates.append(value)
        return min(candidates) if candidates else default

    def _max_decode_batch_size(self) -> int:
        candidates = []
        for value in (
                getattr(self._executor, "max_num_active_requests", None),
                getattr(self._executor, "max_batch_size", None),
        ):
            if isinstance(value, int) and value > 0:
                candidates.append(value)
        return min(candidates) if candidates else 1

    def _tokens_per_block(self) -> int:
        kv_cache_manager = self._executor.resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER)
        if kv_cache_manager is None:
            return 0
        kv_stats = kv_cache_manager.get_kv_cache_stats()
        return int(getattr(kv_stats, "tokens_per_block", 0) or 0)

    def _enable_block_reuse(self) -> bool:
        kv_cache_manager = self._executor.resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER)
        return bool(getattr(kv_cache_manager, "enable_block_reuse", False))

    def _limits(self) -> dict:
        kv_cache_manager = self._executor.resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER)
        kv_stats = kv_cache_manager.get_kv_cache_stats(
        ) if kv_cache_manager is not None else None
        return {
            "max_num_scheduled_tokens": self._executor.max_num_tokens,
            "max_num_running_reqs": self._executor.max_num_active_requests,
            "max_model_len": self._executor.max_seq_len,
            "max_input_len": self._executor.max_input_len,
            "tokens_per_block": getattr(kv_stats, "tokens_per_block", None),
            "num_gpu_blocks": getattr(kv_stats, "max_num_blocks", None),
        }

    def _rank_output_path(self, base_path: str) -> str:
        rank = getattr(self._executor.dist, "rank", 0)
        if rank == 0:
            return base_path
        stem, ext = os.path.splitext(base_path)
        return f"{stem}_rank{rank}{ext}"

    @staticmethod
    def _request_id(point_index: int, offset: int) -> int:
        return _BENCHMARK_REQUEST_ID_BASE + point_index * 1024 + offset

    @staticmethod
    def _cache_salt_id(point_index: int) -> int:
        return _BENCHMARK_REQUEST_ID_BASE + 500_000 + point_index

    @staticmethod
    def _is_benchmark_request(request: "LlmRequest") -> bool:
        return bool(getattr(request, "is_self_benchmark_request", False))
