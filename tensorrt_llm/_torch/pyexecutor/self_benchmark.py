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
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Callable, Literal, Optional

from tensorrt_llm.bindings.executor import OutputConfig, Request, SamplingConfig
from tensorrt_llm.logger import logger

from .llm_request import executor_request_to_llm_request
from .resource_manager import ResourceManagerType
from .scheduler import ScheduledRequests, WaitingQueue

if TYPE_CHECKING:
    from .llm_request import LlmRequest
    from .py_executor import PyExecutor

_BENCHMARK_REQUEST_ID_BASE = 900_000_000
_PLANNING_REQUEST_ID_BASE = 1_800_000_000
_SHUTDOWN_INTERRUPT_REASON = "shutdown_requested"


@dataclass(frozen=True)
class BenchmarkCase:
    case_type: Literal["warmup", "prefill_seed", "prefill", "decode"]
    case_id: int
    isl: int = 0
    kv_read_tokens: int = 0
    context_length: int = 0
    batch_size: int = 1
    cache_salt_id: Optional[int] = None


@dataclass(frozen=True)
class BenchmarkCasePlan:
    case_id: int
    planner_backend: str
    requested_batch_size: int
    proposed_batch_size: int
    request_capacity: int
    microbatch_capacity: int
    target_kv_capacity: int
    draft_kv_capacity: Optional[int]
    limiting_constraint: str
    token_components: dict[str, int]
    kv_snapshot: dict
    planner_origin_rank: int = 0


@dataclass(frozen=True)
class _AxisPlan:
    planner_backend: str
    requested_capacity: int
    request_capacity: int
    microbatch_capacity: int
    target_kv_capacity: int
    draft_kv_capacity: Optional[int]
    proposed_capacity: int
    limiting_constraint: str
    token_components: dict[str, int]
    kv_snapshot: dict


@dataclass(frozen=True)
class _AxisCoordinate:
    case_type: Literal["warmup", "prefill_seed", "prefill", "decode"]
    isl: int = 0
    kv_read_tokens: int = 0
    context_length: int = 0


class BenchmarkOutcome(IntEnum):
    COMPLETE = 0
    CONTINUE = 1
    SKIP = 2
    ABORT = 3


class TrialState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DRAINING = "draining"
    COMPLETE = "complete"
    SKIPPED = "skipped"
    FAILED = "failed"


class RunState(str, Enum):
    RUNNING = "running"
    DRAINING = "draining"
    COMPLETE = "complete"
    ABORTED = "aborted"
    INTERRUPTED = "interrupted"


class DrainTarget(str, Enum):
    SKIP = "skip"
    ABORT = "abort"
    INTERRUPT = "interrupt"


@dataclass(frozen=True)
class RequestScheduleSignature:
    phase: Literal["prefill", "decode"]
    total_tokens: int
    expected_cached_tokens: int
    expected_context_chunk_size: int


@dataclass
class BenchmarkTrial:
    trial_id: int
    case: BenchmarkCase
    expected_request_ids: set[int]
    expected_schedule: dict[int, RequestScheduleSignature]
    constructed_request_ids: set[int] = field(default_factory=set)
    scheduled_request_ids: set[int] = field(default_factory=set)
    executed_request_ids: set[int] = field(default_factory=set)
    completed_request_ids: set[int] = field(default_factory=set)
    terminated_request_ids: set[int] = field(default_factory=set)
    failed_request_ids: set[int] = field(default_factory=set)
    stats: list[dict] = field(default_factory=list)
    state: TrialState = TrialState.PENDING
    pending_outcome: BenchmarkOutcome = BenchmarkOutcome.CONTINUE
    terminal_reason: Optional[str] = None
    origin_rank: Optional[int] = None
    observed_kv_read_tokens: Optional[int] = None
    cache_hit_validated: Optional[bool] = None
    scheduled_batch_size: Optional[int] = None
    executed_batch_size: Optional[int] = None
    admission_reason: Optional[str] = None
    source_admission_reason: Optional[str] = None
    rank_consensus_applied: bool = False
    admission_warning_emitted: bool = False


@dataclass(frozen=True)
class BenchmarkTrialResult:
    trial_id: int
    case: BenchmarkCase
    iteration_stats: tuple[dict, ...]
    observed_kv_read_tokens: Optional[int] = None
    cache_hit_validated: Optional[bool] = None
    admission: dict = field(default_factory=dict)


class SelfBenchmark:
    """Runs synthetic startup benchmark cases inside the PyExecutor loop."""

    def __init__(self, executor: "PyExecutor") -> None:
        self._executor = executor
        self.config = executor.llm_args.self_benchmark_config
        self._case_plans: dict[int, BenchmarkCasePlan] = {}
        self._local_axis_plans: dict[_AxisCoordinate, _AxisPlan] = {}
        self._axis_origin_ranks: dict[_AxisCoordinate, int] = {}
        self._planner_events: list[dict] = []
        self._planner_error: Optional[dict[str, str]] = None
        self._artifact_write_error: Optional[dict[str, str]] = None
        self._planning_request_id_cursor = 0
        self._cases: list[BenchmarkCase] = []
        self._next_case_index = 0
        self._next_trial_id = 0
        # Per-trial request-id stride must exceed the largest synthetic batch so
        # trial N's request ids never overflow into trial N+1's id range.
        self._id_stride = max(1024, self._max_decode_batch_size() + 1)
        self._current_trial: Optional[BenchmarkTrial] = None
        self._results: list[BenchmarkTrialResult] = []
        self._done = self.config is None
        self._run_state = RunState.COMPLETE if self._done else RunState.RUNNING
        self._drain_target: Optional[DrainTarget] = None
        self._interrupt_requested = False
        self._skipped_cases: list[dict] = []
        self._abort: Optional[dict] = None
        self._admission_records: dict[int, dict] = {}
        self._started_at = time.monotonic()
        self._run_id = uuid.uuid4().hex[:12]
        self._output_path = (
            None if self.config is None else self._rank_output_path(self.config.output_path)
        )
        self._identity = self._build_identity()
        if not self._done:
            logger.info("Self-benchmark enabled: %s", self.config)
            initialization_stage = "output_invalidation"
            try:
                # Invalidate any stale results from a previous run up front,
                # so a crash mid-sweep cannot leave a prior run looking valid.
                self._invalidate_output()
                initialization_stage = "case_planning"
                self._cases = self._build_cases()
            except Exception as exc:
                if initialization_stage == "output_invalidation":
                    # The synchronized abort cannot publish on a path that
                    # already failed. Keep this rank alive through consensus;
                    # writable peers still publish their terminal artifacts.
                    self._output_path = None
                self._planner_error = {
                    "type": type(exc).__name__,
                    "reason": str(exc),
                }
                self._cases = []
                self._case_plans = {}
                self._planner_events = [
                    {
                        "event": "planner_error",
                        "stage": initialization_stage,
                        "error_type": type(exc).__name__,
                        "reason": str(exc),
                    }
                ]
                logger.error("Self-benchmark planning failed on this rank: %s", exc)
            else:
                logger.info("Self-benchmark case set: %d case(s)", len(self._cases))

    @property
    def active(self) -> bool:
        return self._run_state in (RunState.RUNNING, RunState.DRAINING)

    def local_planner_record(self, rank: int) -> dict:
        grid_config = self._grid_config_signature()
        if self._planner_error is not None:
            return {
                "rank": rank,
                "ok": False,
                "error": dict(self._planner_error),
                "grid_config": grid_config,
                "signature": [],
                "axes": [],
            }

        axes = []
        signature = []
        for coordinate in sorted(self._local_axis_plans, key=self._axis_sort_key):
            plan = self._local_axis_plans[coordinate]
            axes.append(
                {
                    "coordinate": asdict(coordinate),
                    "plan": asdict(plan),
                }
            )
            signature.append(
                {
                    **asdict(coordinate),
                    "planner_backend": plan.planner_backend,
                    "requested_capacity": plan.requested_capacity,
                    "token_components": dict(plan.token_components),
                    "draft_kv_configured": plan.draft_kv_capacity is not None,
                }
            )
        return {
            "rank": rank,
            "ok": True,
            "error": None,
            "grid_config": grid_config,
            "signature": signature,
            "axes": axes,
        }

    def apply_planner_consensus(self, records: list[dict]) -> None:
        if self.config is None or not self.active:
            return
        if not records:
            self._abort_planning("planner_consensus_empty", origin_rank=0)
            return

        ordered_records = sorted(records, key=lambda record: int(record["rank"]))
        failed_records = [record for record in ordered_records if not record.get("ok", False)]
        if failed_records:
            canonical = failed_records[0]
            error = canonical.get("error") or {}
            error_type = error.get("type", "RuntimeError")
            detail = error.get("reason", "unknown planner failure")
            self._abort_planning(
                f"planner_error: {error_type}: {detail}",
                origin_rank=int(canonical["rank"]),
            )
            return

        reference_signature = ordered_records[0].get("signature")
        reference_grid_config = ordered_records[0].get("grid_config")
        if any(
            record.get("signature") != reference_signature
            or record.get("grid_config") != reference_grid_config
            for record in ordered_records[1:]
        ):
            self._abort_planning("planner_signature_mismatch", origin_rank=0)
            return

        try:
            global_plans, origin_ranks = self._merge_planner_records(ordered_records)
            self._cases = self._build_cases(global_plans, origin_ranks)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            self._abort_planning(
                f"planner_consensus_error: {type(exc).__name__}: {exc}",
                origin_rank=0,
            )
            return

        self._planner_events.append(
            {
                "event": "tp_plan_consensus",
                "ranks": [int(record["rank"]) for record in ordered_records],
                "axis_count": len(global_plans),
            }
        )

    def _abort_planning(self, reason: str, origin_rank: int) -> None:
        self._planner_events.append(
            {
                "event": "planner_consensus_abort",
                "reason": reason,
                "origin_rank": origin_rank,
            }
        )
        logger.error("Self-benchmark planning aborted: %s", reason)
        self.apply_global_outcome(
            BenchmarkOutcome.ABORT,
            reason=reason,
            origin_rank=origin_rank,
            drain_target=DrainTarget.ABORT,
        )

    def _merge_planner_records(
        self, records: list[dict]
    ) -> tuple[dict[_AxisCoordinate, _AxisPlan], dict[_AxisCoordinate, int]]:
        reference_axes = records[0]["axes"]
        if any(len(record["axes"]) != len(reference_axes) for record in records[1:]):
            raise ValueError("planner axis count differs across TP ranks")

        global_plans = {}
        origin_ranks = {}
        for axis_index, reference_axis in enumerate(reference_axes):
            coordinate = _AxisCoordinate(**reference_axis["coordinate"])
            ranked_plans = []
            for record in records:
                axis = record["axes"][axis_index]
                if axis["coordinate"] != reference_axis["coordinate"]:
                    raise ValueError("planner axis ordering differs across TP ranks")
                ranked_plans.append((int(record["rank"]), axis["plan"]))
            global_plan, origin_rank = self._merge_axis_plans(ranked_plans)
            global_plans[coordinate] = global_plan
            origin_ranks[coordinate] = origin_rank
        return global_plans, origin_ranks

    def _merge_axis_plans(self, ranked_plans: list[tuple[int, dict]]) -> tuple[_AxisPlan, int]:
        reference = ranked_plans[0][1]
        draft_capacities = [plan["draft_kv_capacity"] for _, plan in ranked_plans]
        if any(capacity is None for capacity in draft_capacities) and not all(
            capacity is None for capacity in draft_capacities
        ):
            raise ValueError("draft KV planner presence differs across TP ranks")

        request_capacity = min(int(plan["request_capacity"]) for _, plan in ranked_plans)
        microbatch_capacity = min(int(plan["microbatch_capacity"]) for _, plan in ranked_plans)
        target_kv_capacity = min(int(plan["target_kv_capacity"]) for _, plan in ranked_plans)
        draft_kv_capacity = (
            None
            if draft_capacities[0] is None
            else min(int(capacity) for capacity in draft_capacities)
        )
        merged = self._make_axis_plan(
            backend=str(reference["planner_backend"]),
            requested_capacity=int(reference["requested_capacity"]),
            request_capacity=request_capacity,
            microbatch_capacity=microbatch_capacity,
            target_kv_capacity=target_kv_capacity,
            draft_kv_capacity=draft_kv_capacity,
            token_components=dict(reference["token_components"]),
            kv_snapshot={},
        )
        capacity_field = {
            "request_capacity": "request_capacity",
            "microbatch_token_capacity": "microbatch_capacity",
            "target_kv_capacity": "target_kv_capacity",
            "draft_kv_capacity": "draft_kv_capacity",
        }[merged.limiting_constraint]
        origin_rank, origin_plan = min(
            (
                (rank, plan)
                for rank, plan in ranked_plans
                if int(plan[capacity_field]) == merged.proposed_capacity
            ),
            key=lambda item: item[0],
        )
        merged = replace(merged, kv_snapshot=dict(origin_plan["kv_snapshot"]))
        return merged, origin_rank

    @staticmethod
    def _axis_sort_key(coordinate: _AxisCoordinate) -> tuple:
        return (
            coordinate.case_type,
            coordinate.isl,
            coordinate.kv_read_tokens,
            coordinate.context_length,
        )

    def _grid_config_signature(self) -> dict:
        if self.config is None:
            return {}
        return self.config.model_dump(exclude={"output_path"})

    def start_trial(self, case: BenchmarkCase, requests: list["LlmRequest"]) -> None:
        """Start one case attempt and register requests constructed on this rank."""
        trial_id = self._next_trial_id
        expected_ids = {self._request_id(trial_id, offset) for offset in range(case.batch_size)}
        phase: Literal["prefill", "decode"] = "decode" if case.case_type == "decode" else "prefill"
        total_tokens = case.context_length + 1 if phase == "decode" else max(1, case.isl)
        if phase == "decode":
            expected_cached_tokens = case.context_length
        elif case.case_type == "prefill":
            expected_cached_tokens = case.kv_read_tokens
        else:
            expected_cached_tokens = 0
        expected_context_chunk_size = (
            total_tokens - expected_cached_tokens if phase == "prefill" else 0
        )
        expected_schedule = {
            request_id: RequestScheduleSignature(
                phase=phase,
                total_tokens=total_tokens,
                expected_cached_tokens=expected_cached_tokens,
                expected_context_chunk_size=expected_context_chunk_size,
            )
            for request_id in expected_ids
        }
        self._next_case_index += 1
        self._next_trial_id += 1
        self._current_trial = BenchmarkTrial(
            trial_id=trial_id,
            case=case,
            expected_request_ids=expected_ids,
            expected_schedule=expected_schedule,
            constructed_request_ids={self._request_id_of(request) for request in requests},
            state=TrialState.RUNNING,
        )
        logger.debug("Starting self-benchmark trial %d for case: %s", trial_id, case)

    def observe_scheduled_batch(self, scheduled_batch: ScheduledRequests) -> None:
        trial = self._current_trial
        if trial is None or trial.state != TrialState.RUNNING:
            return

        all_requests = scheduled_batch.all_requests()
        if not all_requests and trial.scheduled_request_ids == trial.expected_request_ids:
            # The overlap scheduler may expose an empty next batch while the
            # exactly admitted trial is still owned by previous_batch.
            return
        benchmark_requests = [
            request for request in all_requests if self._is_benchmark_request(request)
        ]
        actual_ids = {self._request_id_of(request) for request in benchmark_requests}
        trial.scheduled_batch_size = len(actual_ids)
        mixed = any(
            not self._is_benchmark_request(request)
            and not bool(getattr(request, "is_dummy", False))
            for request in all_requests
        )
        exact_ids = actual_ids == trial.expected_request_ids
        if trial.case.case_type in ("warmup", "prefill_seed", "prefill"):
            exact_phase = (
                not scheduled_batch.context_requests_chunking
                and {
                    self._request_id_of(request)
                    for request in scheduled_batch.context_requests_last_chunk
                    if self._is_benchmark_request(request)
                }
                == trial.expected_request_ids
                and not any(
                    self._is_benchmark_request(request)
                    for request in scheduled_batch.generation_requests
                )
            )
        else:
            exact_phase = {
                self._request_id_of(request)
                for request in scheduled_batch.generation_requests
                if self._is_benchmark_request(request)
            } == trial.expected_request_ids and not any(
                self._is_benchmark_request(request) for request in scheduled_batch.context_requests
            )

        chunked = any(
            self._is_benchmark_request(request)
            for request in scheduled_batch.context_requests_chunking
        )
        mismatch_reason = None
        if mixed:
            mismatch_reason = "mixed_non_benchmark_request"
        elif trial.case.case_type != "decode" and chunked:
            mismatch_reason = "context_chunked"
        elif len(actual_ids) != trial.case.batch_size:
            mismatch_reason = "scheduled_batch_size_mismatch"
        elif not exact_ids:
            mismatch_reason = "scheduled_request_identity_mismatch"
        elif not exact_phase or not self._schedule_signatures_match(benchmark_requests, trial):
            mismatch_reason = "scheduled_batch_shape_mismatch"
        if mismatch_reason is not None:
            self._request_local_transition(BenchmarkOutcome.SKIP, mismatch_reason)
            return
        trial.scheduled_request_ids.update(actual_ids)

    def observe_executed_batch(self, scheduled_batch: ScheduledRequests, stats: dict) -> bool:
        benchmark_requests = [
            request
            for request in scheduled_batch.all_requests()
            if self._is_benchmark_request(request)
        ]
        if not benchmark_requests:
            return False

        trial = self._current_trial
        if trial is None or trial.state != TrialState.RUNNING:
            return True
        executed_ids = {self._request_id_of(request) for request in benchmark_requests}
        trial.executed_batch_size = len(executed_ids)
        mismatch_reason = None
        if len(trial.scheduled_request_ids) != trial.case.batch_size:
            mismatch_reason = "scheduled_batch_size_mismatch"
        elif trial.scheduled_request_ids != trial.expected_request_ids:
            mismatch_reason = "scheduled_batch_shape_mismatch"
        elif len(executed_ids) != trial.case.batch_size:
            mismatch_reason = "executed_batch_size_mismatch"
        elif executed_ids != trial.expected_request_ids:
            mismatch_reason = "executed_batch_shape_mismatch"
        if mismatch_reason is not None:
            self._request_local_transition(BenchmarkOutcome.SKIP, mismatch_reason)
            return True

        trial.executed_request_ids.update(executed_ids)
        self._sanitize_queue_counters(stats)
        self._record_cache_hit_validation(benchmark_requests, stats)
        trial.stats.append(stats)
        return True

    def observe_finished_requests(self, requests: list["LlmRequest"]) -> None:
        trial = self._current_trial
        if trial is None:
            return
        trial.completed_request_ids.update(
            self._request_id_of(request)
            for request in requests
            if self._belongs_to_current_trial(request, trial)
        )

    def observe_failed_requests(self, requests: list["LlmRequest"], reason: str) -> None:
        trial = self._current_trial
        if trial is None:
            return
        trial.failed_request_ids.update(
            self._request_id_of(request)
            for request in requests
            if self._belongs_to_current_trial(request, trial)
        )
        self._request_local_transition(BenchmarkOutcome.ABORT, reason)

    def observe_terminated_requests(self, requests: list["LlmRequest"]) -> None:
        trial = self._current_trial
        if trial is None:
            return
        trial.terminated_request_ids.update(
            self._request_id_of(request)
            for request in requests
            if self._belongs_to_current_trial(request, trial)
        )

    def should_hold_from_scheduler(self, request: "LlmRequest") -> bool:
        trial = self._current_trial
        if trial is None or not self._is_benchmark_request(request):
            return False
        request_id = self._request_id_of(request)
        return trial.state == TrialState.DRAINING or request_id in trial.scheduled_request_ids

    def should_terminate(self, request: "LlmRequest") -> bool:
        trial = self._current_trial
        if trial is None or not self._is_benchmark_request(request):
            return False
        request_id = self._request_id_of(request)
        if trial.state == TrialState.DRAINING:
            return request_id in trial.constructed_request_ids
        return request_id in trial.completed_request_ids

    def request_interrupt(self) -> None:
        if not self.active:
            return
        if not self._interrupt_requested:
            logger.info("Self-benchmark stopping early because shutdown was requested.")
        self._interrupt_requested = True

    def local_outcome(self) -> BenchmarkOutcome:
        trial = self._current_trial
        if trial is None:
            return (
                BenchmarkOutcome.ABORT if self._interrupt_requested else BenchmarkOutcome.CONTINUE
            )
        if trial.state == TrialState.DRAINING:
            released = trial.constructed_request_ids <= trial.terminated_request_ids
            return BenchmarkOutcome.COMPLETE if released else BenchmarkOutcome.CONTINUE
        if trial.pending_outcome == BenchmarkOutcome.ABORT:
            return trial.pending_outcome
        if self._interrupt_requested:
            return BenchmarkOutcome.ABORT
        if trial.pending_outcome == BenchmarkOutcome.SKIP:
            return trial.pending_outcome
        return (
            BenchmarkOutcome.COMPLETE
            if self._trial_is_exactly_complete(trial)
            else BenchmarkOutcome.CONTINUE
        )

    def local_diagnostic(self, outcome: BenchmarkOutcome, rank: int) -> Optional[dict]:
        trial = self._current_trial
        if (
            outcome == BenchmarkOutcome.ABORT
            and self._interrupt_requested
            and (trial is None or trial.pending_outcome != BenchmarkOutcome.ABORT)
        ):
            return {
                "rank": rank,
                "outcome": int(outcome),
                "reason": _SHUTDOWN_INTERRUPT_REASON,
                "drain_target": DrainTarget.INTERRUPT.value,
                "trial_id": None if trial is None else trial.trial_id,
                "case": None if trial is None else asdict(trial.case),
            }
        if trial is None or trial.pending_outcome != outcome:
            return None
        drain_target = DrainTarget.SKIP if outcome == BenchmarkOutcome.SKIP else DrainTarget.ABORT
        return {
            "rank": rank,
            "outcome": int(outcome),
            "reason": trial.terminal_reason,
            "drain_target": drain_target.value,
            "trial_id": trial.trial_id,
            "case": asdict(trial.case),
        }

    def apply_global_outcome(
        self,
        outcome: BenchmarkOutcome,
        reason: Optional[str] = None,
        origin_rank: Optional[int] = None,
        drain_target: Optional[DrainTarget] = None,
    ) -> None:
        if outcome == BenchmarkOutcome.CONTINUE:
            return
        if outcome == BenchmarkOutcome.SKIP:
            self._begin_drain(DrainTarget.SKIP, reason, origin_rank)
            return
        if outcome == BenchmarkOutcome.ABORT:
            target = drain_target or DrainTarget.ABORT
            if target not in (DrainTarget.ABORT, DrainTarget.INTERRUPT):
                raise ValueError(f"Invalid ABORT drain target: {target}")
            if self._current_trial is None:
                self._abort = {
                    "trial_id": None,
                    "case": None,
                    "reason": reason,
                    "origin_rank": origin_rank,
                }
                if target == DrainTarget.INTERRUPT:
                    self._finish_interrupted()
                else:
                    self._finish_aborted()
                return
            self._begin_drain(target, reason, origin_rank)
            return
        self._finalize_global_complete()

    def make_prefill_requests(
        self, active_requests: list["LlmRequest"], waiting_queue: WaitingQueue
    ) -> list["LlmRequest"]:
        """Construct a deterministic prefill trial independently on every TP rank."""
        if not self._can_start_next_trial(active_requests, waiting_queue):
            return []
        case = self._peek_next_case()
        if case is None:
            self._finish_complete()
            return []
        if case.case_type not in ("warmup", "prefill_seed", "prefill"):
            return []

        trial_id = self._next_trial_id
        requests: list["LlmRequest"] = []
        failure_reason = None
        for offset in range(case.batch_size):
            try:
                request = self._make_prefill_request(case, trial_id, offset)
                llm_request = executor_request_to_llm_request(
                    req_id=self._request_id(trial_id, offset),
                    executor_request=request,
                    child_req_ids=None,
                    exclude_last_generation_logits=self._exclude_last_generation_logits(),
                )
                self._mark_benchmark_request(llm_request, trial_id)
                requests.append(llm_request)
            except Exception as exc:
                failure_reason = f"synthetic_request_construction_failed: {exc}"
                logger.error(
                    "Self-benchmark prefill construction failed for trial %d: %s",
                    trial_id,
                    exc,
                )
                break

        self.start_trial(case, requests)
        if failure_reason is not None:
            self.observe_failed_requests([], failure_reason)
        return requests

    def make_decode_requests(
        self, active_requests: list["LlmRequest"], waiting_queue: WaitingQueue
    ) -> list["LlmRequest"]:
        if not self._can_start_next_trial(active_requests, waiting_queue):
            return []
        case = self._peek_next_case()
        if case is None:
            self._finish_complete()
            return []
        if case.case_type != "decode":
            return []

        trial_id = self._next_trial_id
        kv_cache_manager = self._executor.resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        if kv_cache_manager is None:
            self.start_trial(case, [])
            self.observe_failed_requests([], "KV cache manager is not available")
            return []
        draft_kv_cache_manager = self._executor.resource_manager.get_resource_manager(
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER
        )
        token_num = max(1, case.context_length + 1)
        request_ids = [self._request_id(trial_id, offset) for offset in range(case.batch_size)]
        try:
            requests = kv_cache_manager.add_dummy_requests(
                request_ids=request_ids,
                token_nums=[token_num] * case.batch_size,
                is_gen=True,
                max_num_draft_tokens=self._executor.max_total_draft_tokens,
                kv_reserve_draft_tokens=getattr(
                    self._executor.model_engine,
                    "max_draft_loop_tokens",
                    self._executor.max_total_draft_tokens,
                ),
                use_mrope=getattr(self._executor.model_engine, "use_mrope", False),
                max_beam_width=self._executor.max_beam_width,
                draft_kv_cache_manager=draft_kv_cache_manager,
            )
        except Exception as exc:
            self.start_trial(case, [])
            self.observe_failed_requests([], f"synthetic_decode_kv_allocation_failed: {exc}")
            logger.error(
                "Self-benchmark decode construction failed for trial %d: %s",
                trial_id,
                exc,
            )
            return []
        if requests is None:
            self.start_trial(case, [])
            self._request_local_transition(
                BenchmarkOutcome.SKIP, "insufficient_kv_cache_for_synthetic_decode"
            )
            return []
        for request in requests:
            self._mark_benchmark_request(request, trial_id)
        self.start_trial(case, requests)
        return requests

    def _invalidate_output(self) -> None:
        """Replace any prior-run output with an invalid 'running' sentinel.

        Readiness consumers must not treat the mere existence of the output
        file as a completed result: a crash mid-sweep would otherwise leave a
        previous run's file looking valid. Writing the sentinel at init makes
        existence insufficient and the run_id/identity authoritative.
        """
        if self._output_path is None:
            return
        output = {
            "schema_version": 2,
            "status": "running",
            "valid": False,
            "run_id": self._run_id,
            "started_at_unix": time.time(),
            "identity": self._identity,
            "output_path": self._output_path,
            "message": "Self-benchmark is running; previous results are invalid.",
        }
        try:
            os.unlink(self._output_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning(
                "Failed to remove stale self-benchmark output %s: %s", self._output_path, exc
            )
        self._atomic_write_json(self._output_path, output)

    def _coverage(self) -> dict[str, int]:
        expected_cases = sum(case.case_type in ("prefill", "decode") for case in self._cases)
        return {
            "expected_cases": expected_cases,
            "completed_trials": len(self._results),
            "skipped_cases": len(self._skipped_cases),
        }

    def _build_admission_record(
        self,
        trial: BenchmarkTrial,
        *,
        case: Optional[BenchmarkCase] = None,
        trial_id: Optional[int] = None,
        normalized_reason: Optional[str] = None,
        scheduled_batch_size: Optional[int] = None,
        executed_batch_size: Optional[int] = None,
    ) -> dict:
        case = case or trial.case
        trial_id = trial.trial_id if trial_id is None else trial_id
        plan = self._case_plans.get(case.case_id)
        if plan is not None:
            token_components = dict(plan.token_components)
            kv_snapshot = dict(plan.kv_snapshot)
            request_capacity = plan.request_capacity
            planner_backend = plan.planner_backend
            limiting_constraint = plan.limiting_constraint
            capacities = {
                "request": plan.request_capacity,
                "microbatch_tokens": plan.microbatch_capacity,
                "target_kv": plan.target_kv_capacity,
                "draft_kv": plan.draft_kv_capacity,
            }
        else:
            reusable_tokens = (
                case.context_length if case.case_type == "decode" else case.kv_read_tokens
            )
            base_tokens = case.context_length + 1 if case.case_type == "decode" else case.isl
            uncached_tokens = max(0, base_tokens - reusable_tokens)
            draft_tokens = int(getattr(self._executor, "max_total_draft_tokens", 0) or 0)
            beam_width = int(getattr(self._executor, "max_beam_width", 1) or 1)
            compute_tokens = (
                beam_width + draft_tokens
                if case.case_type == "decode"
                else max(1, uncached_tokens + draft_tokens)
            )
            token_components = {
                "base_tokens": base_tokens,
                "reusable_tokens": reusable_tokens,
                "uncached_tokens": uncached_tokens,
                "beam_width": beam_width,
                "draft_tokens": draft_tokens,
                "compute_tokens_per_request": compute_tokens,
            }
            kv_snapshot = {}
            request_capacity = self._max_decode_batch_size()
            planner_backend = None
            limiting_constraint = None
            capacities = {
                "request": request_capacity,
                "microbatch_tokens": None,
                "target_kv": None,
                "draft_kv": None,
            }

        if normalized_reason is None:
            normalized_reason = (
                "exact"
                if trial.state == TrialState.COMPLETE
                else trial.admission_reason
                or self._normalize_admission_reason(trial.terminal_reason)
            )
        source_normalized_reason = trial.source_admission_reason or normalized_reason
        microbatch_tokens_per_request = int(
            token_components.get("compute_tokens_per_request", 0) or 0
        )
        proposed_batch_size = case.batch_size
        return {
            "trial_id": trial_id,
            "case_id": case.case_id,
            "phase": "decode" if case.case_type == "decode" else "prefill",
            "case_type": case.case_type,
            "requested_batch_size": case.batch_size,
            "proposed_batch_size": proposed_batch_size,
            "scheduled_batch_size": (
                trial.scheduled_batch_size if scheduled_batch_size is None else scheduled_batch_size
            )
            or 0,
            "executed_batch_size": (
                trial.executed_batch_size if executed_batch_size is None else executed_batch_size
            )
            or 0,
            "isl": case.isl,
            "context_length": case.context_length,
            "expected_kv_read_tokens": case.kv_read_tokens,
            "token_components": token_components,
            "microbatch_tokens_per_request": microbatch_tokens_per_request,
            "microbatch_tokens_proposed": (microbatch_tokens_per_request * proposed_batch_size),
            "limits": {
                "max_num_tokens": self._executor.max_num_tokens,
                "max_num_active_requests": self._executor.max_num_active_requests,
                "max_batch_size": self._executor.max_batch_size,
                "request_capacity": request_capacity,
            },
            "capacities": capacities,
            "kv_snapshot": kv_snapshot,
            "planner_backend": planner_backend,
            "limiting_constraint": limiting_constraint,
            "normalized_reason": normalized_reason,
            "source_normalized_reason": source_normalized_reason,
            "source_reason": trial.terminal_reason,
            "origin_rank": trial.origin_rank,
            "rank_consensus_applied": trial.rank_consensus_applied,
            "expected_reuse_observed": trial.observed_kv_read_tokens,
            "expected_reuse_validated": trial.cache_hit_validated,
        }

    @staticmethod
    def _point_from_case_dict(case: dict) -> dict:
        # Project the internal `BenchmarkCase` onto Dynamo's `point` shape
        # (point_type + dims). Dynamo's TRT-LLM self-benchmark consumer keys the
        # emitted results by `point`/`point_type` (matching the vLLM instrumented
        # scheduler payload), so we emit it alongside `case` to keep the JSON
        # consumable by that normalizer without changing the engine's internals.
        return {
            "point_type": case.get("case_type"),
            "isl": case.get("isl", 0),
            "kv_read_tokens": case.get("kv_read_tokens", 0),
            "context_length": case.get("context_length", 0),
            "batch_size": case.get("batch_size", 1),
        }

    @staticmethod
    def _serialize_result(result: BenchmarkTrialResult) -> dict:
        case = asdict(result.case)
        return {
            "trial_id": result.trial_id,
            "case": case,
            "point": SelfBenchmark._point_from_case_dict(case),
            "iteration_stats": list(result.iteration_stats),
            "observed_kv_read_tokens": result.observed_kv_read_tokens,
            "cache_hit_validated": result.cache_hit_validated,
            "admission": result.admission,
        }

    def _serialize_skipped_case(self, skipped_case: dict) -> dict:
        serialized = dict(skipped_case)
        admission = self._admission_records.get(int(skipped_case["trial_id"]), {})
        serialized["normalized_reason"] = admission.get(
            "normalized_reason",
            self._normalize_admission_reason(skipped_case.get("reason")),
        )
        serialized["admission"] = admission
        case = serialized.get("case")
        if isinstance(case, dict):
            serialized["point"] = self._point_from_case_dict(case)
        # `skipped_reason` mirrors `reason` for the Dynamo consumer's vocabulary.
        serialized.setdefault("skipped_reason", serialized.get("reason"))
        return serialized

    def _serialize_abort(self) -> Optional[dict]:
        if self._abort is None:
            return None
        serialized = dict(self._abort)
        trial_id = self._abort.get("trial_id")
        admission = {} if trial_id is None else self._admission_records.get(int(trial_id), {})
        serialized["normalized_reason"] = admission.get(
            "normalized_reason",
            self._normalize_admission_reason(self._abort.get("reason")),
        )
        serialized["admission"] = admission
        return serialized

    def _write_terminal(self, status: Literal["complete", "aborted", "interrupted"]) -> None:
        if self.config is None or self._output_path is None:
            return
        try:
            coverage = self._coverage()
            valid = (
                status == "complete"
                and coverage["completed_trials"] == coverage["expected_cases"]
                and coverage["skipped_cases"] == 0
            )
            output = {
                "schema_version": 2,
                "status": status,
                "valid": valid,
                "run_id": self._run_id,
                "completed_at_unix": time.time(),
                "elapsed_time_ms": (time.monotonic() - self._started_at) * 1000.0,
                "identity": self._identity,
                "output_path": self._output_path,
                "config": self.config.model_dump(),
                "limits": self._limits(),
                "coverage": coverage,
                "cases": [asdict(case) for case in self._cases],
                "case_plans": [
                    asdict(self._case_plans[case_id]) for case_id in sorted(self._case_plans)
                ],
                "planner_events": list(self._planner_events),
                "trial_admissions": [
                    self._admission_records[trial_id]
                    for trial_id in sorted(self._admission_records)
                ],
                "skipped_cases": [
                    self._serialize_skipped_case(skipped_case)
                    for skipped_case in self._skipped_cases
                ],
                "abort": self._serialize_abort(),
                "results": [self._serialize_result(result) for result in self._results],
            }
            self._atomic_write_json(self._output_path, output)
        except Exception as exc:
            self._artifact_write_error = {
                "stage": "terminal",
                "type": type(exc).__name__,
                "reason": str(exc),
            }
            logger.error(
                "Failed to write self-benchmark terminal artifact %s: %s",
                self._output_path,
                exc,
            )
            return
        logger.info(
            "Self-benchmark terminal artifact written to %s (%s, valid=%s)",
            self._output_path,
            status,
            valid,
        )

    def _finish_complete(self) -> None:
        self._done = True
        self._run_state = RunState.COMPLETE
        self._write_terminal("complete")

    def _finish_aborted(self) -> None:
        self._done = True
        self._run_state = RunState.ABORTED
        self._write_terminal("aborted")

    def _finish_interrupted(self) -> None:
        self._done = True
        self._run_state = RunState.INTERRUPTED
        self._write_terminal("interrupted")

    def _atomic_write_json(self, output_path: str, output: dict) -> None:
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(output_path)
        # Process-unique temp file so co-located writers never clobber a shared
        # ".tmp"; os.replace then publishes atomically.
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{basename}.{os.getpid()}.", suffix=".tmp", dir=output_dir, text=True
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(output, f, indent=2)
                f.write("\n")
            os.replace(tmp_path, output_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _build_identity(self) -> dict:
        executor = self._executor
        llm_args = getattr(executor, "llm_args", None)
        dist = getattr(executor, "dist", None)
        parallel = getattr(llm_args, "parallel_config", None)
        model = getattr(llm_args, "model", None)
        if model is not None and not isinstance(model, (str, int, float, bool)):
            model = str(model)
        return {
            "model": model,
            "benchmark_mode": getattr(self.config, "mode", None),
            "rank": getattr(dist, "rank", 0),
            "world_size": getattr(parallel, "world_size", None),
            "tp_size": getattr(parallel, "tp_size", None),
            "pp_size": getattr(parallel, "pp_size", None),
            "pid": os.getpid(),
        }

    def _build_cases(
        self,
        axis_plans: Optional[dict[_AxisCoordinate, _AxisPlan]] = None,
        axis_origin_ranks: Optional[dict[_AxisCoordinate, int]] = None,
    ) -> list[BenchmarkCase]:
        if self.config is None:
            return []
        self._case_plans = {}
        self._planner_events = []
        if axis_plans is None:
            self._local_axis_plans = {}
            self._axis_origin_ranks = {}
        else:
            axis_plans = dict(axis_plans)
            self._axis_origin_ranks = dict(axis_origin_ranks or {})

        cases: list[BenchmarkCase] = []
        next_case_id = 0
        if self.config.warmup_iterations > 0:
            warmup_isl = min(8, self._max_prefill_isl())
            warmup_coordinate = _AxisCoordinate(case_type="warmup", isl=warmup_isl)
            warmup_plan = self._resolve_axis_plan(
                warmup_coordinate,
                axis_plans,
                lambda: self._plan_prefill_axis("warmup", warmup_isl, 0),
            )
            self._require_positive_capacity("warmup", warmup_plan)
            warmup_origin = self._axis_origin_rank(warmup_coordinate)
            for _ in range(self.config.warmup_iterations):
                case = BenchmarkCase(case_type="warmup", case_id=next_case_id, isl=warmup_isl)
                self._append_case(cases, case, warmup_plan, warmup_origin)
                next_case_id += 1

        if self.config.mode in ("prefill", "agg"):
            for isl in self._sample_values(
                self._max_prefill_isl(), self.config.prefill_isl_granularity
            ):
                for kv_read_tokens in self._kv_read_values_for_isl(isl):
                    measured_coordinate = _AxisCoordinate(
                        case_type="prefill",
                        isl=isl,
                        kv_read_tokens=kv_read_tokens,
                    )
                    measured_plan = self._resolve_axis_plan(
                        measured_coordinate,
                        axis_plans,
                        lambda: self._plan_prefill_axis("prefill", isl, kv_read_tokens),
                    )
                    measured_origin = self._axis_origin_rank(measured_coordinate)
                    seed_plan = None
                    seed_coordinate = None
                    seed_origin = measured_origin
                    if kv_read_tokens > 0:
                        seed_isl = kv_read_tokens + 1
                        seed_coordinate = _AxisCoordinate(
                            case_type="prefill_seed",
                            isl=seed_isl,
                            kv_read_tokens=kv_read_tokens,
                        )
                        seed_plan = self._resolve_axis_plan(
                            seed_coordinate,
                            axis_plans,
                            lambda: self._plan_prefill_axis("prefill_seed", seed_isl, 0),
                        )
                        seed_origin = self._axis_origin_rank(seed_coordinate)
                        seed_capacity = seed_plan.proposed_capacity
                        measured_capacity = measured_plan.proposed_capacity
                        shared_capacity = min(seed_capacity, measured_capacity)
                        if measured_capacity <= seed_capacity:
                            seed_constraint = (
                                seed_plan.limiting_constraint
                                if measured_capacity == seed_capacity
                                else "measured_capacity"
                            )
                            seed_plan = replace(
                                seed_plan,
                                proposed_capacity=shared_capacity,
                                limiting_constraint=seed_constraint,
                            )
                            if measured_capacity < seed_capacity:
                                seed_origin = measured_origin
                        else:
                            measured_plan = replace(
                                measured_plan,
                                proposed_capacity=shared_capacity,
                                limiting_constraint="seed_capacity",
                            )
                            measured_origin = seed_origin
                        measured_plan = replace(measured_plan, proposed_capacity=shared_capacity)
                        seed_plan = replace(seed_plan, proposed_capacity=shared_capacity)
                        if axis_plans is None:
                            self._local_axis_plans[measured_coordinate] = measured_plan
                            self._local_axis_plans[seed_coordinate] = seed_plan
                            self._axis_origin_ranks[measured_coordinate] = measured_origin
                            self._axis_origin_ranks[seed_coordinate] = seed_origin
                    self._require_positive_capacity(
                        f"prefill(isl={isl},kv_read_tokens={kv_read_tokens})", measured_plan
                    )
                    self._record_planner_reduction("prefill", isl, kv_read_tokens, 0, measured_plan)
                    batch_values = self._sample_values(
                        measured_plan.proposed_capacity,
                        self.config.prefill_batch_granularity,
                    )
                    for batch_size in batch_values:
                        cache_salt_id = self._cache_salt_id(next_case_id)
                        if kv_read_tokens > 0:
                            seed_case = BenchmarkCase(
                                case_type="prefill_seed",
                                case_id=next_case_id,
                                isl=seed_isl,
                                kv_read_tokens=kv_read_tokens,
                                batch_size=batch_size,
                                cache_salt_id=cache_salt_id,
                            )
                            assert seed_plan is not None
                            self._append_case(cases, seed_case, seed_plan, seed_origin)
                            next_case_id += 1
                        measured_case = BenchmarkCase(
                            case_type="prefill",
                            case_id=next_case_id,
                            isl=isl,
                            kv_read_tokens=kv_read_tokens,
                            batch_size=batch_size,
                            cache_salt_id=cache_salt_id,
                        )
                        self._append_case(cases, measured_case, measured_plan, measured_origin)
                        next_case_id += 1

        if self.config.mode in ("decode", "agg"):
            context_values = self._sample_values(
                self._max_decode_context_length(), self.config.decode_context_granularity
            )
            for context_length in context_values:
                coordinate = _AxisCoordinate(case_type="decode", context_length=context_length)
                plan = self._resolve_axis_plan(
                    coordinate,
                    axis_plans,
                    lambda: self._plan_decode_axis(context_length),
                )
                self._require_positive_capacity(f"decode(context_length={context_length})", plan)
                self._record_planner_reduction("decode", 0, 0, context_length, plan)
                batch_values = self._sample_values(
                    plan.proposed_capacity, self.config.decode_batch_granularity
                )
                for batch_size in batch_values:
                    case = BenchmarkCase(
                        case_type="decode",
                        case_id=next_case_id,
                        context_length=context_length,
                        batch_size=batch_size,
                    )
                    self._append_case(cases, case, plan, self._axis_origin_rank(coordinate))
                    next_case_id += 1
        return cases

    def _resolve_axis_plan(
        self,
        coordinate: _AxisCoordinate,
        axis_plans: Optional[dict[_AxisCoordinate, _AxisPlan]],
        planner: Callable[[], _AxisPlan],
    ) -> _AxisPlan:
        if axis_plans is not None:
            try:
                return axis_plans[coordinate]
            except KeyError as exc:
                raise RuntimeError(
                    f"Missing synchronized self-benchmark axis: {coordinate}"
                ) from exc

        plan = planner()
        self._local_axis_plans[coordinate] = plan
        self._axis_origin_ranks[coordinate] = self._local_planner_rank()
        return plan

    def _axis_origin_rank(self, coordinate: _AxisCoordinate) -> int:
        return self._axis_origin_ranks.get(coordinate, self._local_planner_rank())

    def _local_planner_rank(self) -> int:
        dist = getattr(self._executor, "dist", None)
        return int(getattr(dist, "tp_rank", getattr(dist, "rank", 0)))

    def _append_case(
        self,
        cases: list[BenchmarkCase],
        case: BenchmarkCase,
        axis_plan: _AxisPlan,
        planner_origin_rank: int,
    ) -> None:
        cases.append(case)
        self._case_plans[case.case_id] = BenchmarkCasePlan(
            case_id=case.case_id,
            planner_backend=axis_plan.planner_backend,
            requested_batch_size=case.batch_size,
            proposed_batch_size=case.batch_size,
            request_capacity=axis_plan.request_capacity,
            microbatch_capacity=axis_plan.microbatch_capacity,
            target_kv_capacity=axis_plan.target_kv_capacity,
            draft_kv_capacity=axis_plan.draft_kv_capacity,
            limiting_constraint=axis_plan.limiting_constraint,
            token_components=dict(axis_plan.token_components),
            kv_snapshot=dict(axis_plan.kv_snapshot),
            planner_origin_rank=planner_origin_rank,
        )

    def _plan_prefill_axis(
        self,
        case_type: Literal["warmup", "prefill_seed", "prefill"],
        isl: int,
        kv_read_tokens: int,
    ) -> _AxisPlan:
        request_capacity = self._max_decode_batch_size()
        target_manager = self._kv_cache_manager()
        if target_manager is None:
            raise RuntimeError("KV cache manager is not available")
        draft_manager = self._draft_kv_cache_manager()
        draft_tokens = 0
        compute_tokens = max(1, isl - kv_read_tokens + draft_tokens)
        token_components = {
            "base_tokens": isl,
            "reusable_tokens": kv_read_tokens,
            "uncached_tokens": max(0, isl - kv_read_tokens),
            "beam_width": int(self._executor.max_beam_width),
            "draft_tokens": draft_tokens,
            "compute_tokens_per_request": compute_tokens,
        }

        if self._is_v2_manager(target_manager):
            backend = "v2_solver"
            microbatch_capacity = self._formula_microbatch_capacity(compute_tokens)
            target_kv_capacity = self._v2_solver_capacity(
                target_manager, isl, request_capacity, max_num_draft_tokens=draft_tokens
            )
            draft_kv_capacity = (
                self._v2_solver_capacity(
                    draft_manager, isl, request_capacity, max_num_draft_tokens=draft_tokens
                )
                if draft_manager is not None
                else None
            )
        else:
            backend = self._v1_planner_backend()
            capacity_requests = self._make_planning_prefill_requests(
                isl, request_capacity, max_tokens=1
            )
            target_kv_capacity = self._probe_primary_capacity(capacity_requests)
            draft_kv_capacity = self._probe_draft_capacity(capacity_requests, draft_manager)
            compute_requests = self._make_planning_prefill_requests(
                isl, request_capacity, max_tokens=1
            )
            for request in compute_requests:
                request.estimated_reusable_tokens = kv_read_tokens
            microbatch_capacity = self._probe_microbatch_capacity(compute_requests, phase="prefill")

        return self._make_axis_plan(
            backend=backend,
            requested_capacity=request_capacity,
            request_capacity=request_capacity,
            microbatch_capacity=microbatch_capacity,
            target_kv_capacity=target_kv_capacity,
            draft_kv_capacity=draft_kv_capacity,
            token_components=token_components,
            kv_snapshot=self._kv_snapshot(target_manager, draft_manager),
        )

    def _plan_decode_axis(self, context_length: int) -> _AxisPlan:
        request_capacity = self._max_decode_batch_size()
        target_manager = self._kv_cache_manager()
        if target_manager is None:
            raise RuntimeError("KV cache manager is not available")
        draft_manager = self._draft_kv_cache_manager()
        compute_requests = target_manager.add_dummy_requests(
            request_ids=self._planning_request_ids(request_capacity),
            token_nums=[max(1, context_length + 1)] * request_capacity,
            is_gen=True,
            prepare_resource=False,
            max_num_draft_tokens=self._executor.max_total_draft_tokens,
            kv_reserve_draft_tokens=self._construction_kv_reserve(),
            use_mrope=getattr(self._executor.model_engine, "use_mrope", False),
            max_beam_width=self._executor.max_beam_width,
            draft_kv_cache_manager=draft_manager,
        )
        if not compute_requests:
            raise RuntimeError("Unable to construct no-allocation decode planning requests")
        first_request = compute_requests[0]
        beam_width = int(first_request.get_beam_width_by_iter(False))
        draft_tokens = int(getattr(first_request, "num_draft_tokens", 0))
        compute_tokens = beam_width + draft_tokens
        token_components = {
            "base_tokens": 1,
            "reusable_tokens": context_length,
            "uncached_tokens": 1,
            "beam_width": beam_width,
            "draft_tokens": draft_tokens,
            "compute_tokens_per_request": compute_tokens,
        }

        if self._is_v2_manager(target_manager):
            backend = "v2_solver"
            microbatch_capacity = self._formula_microbatch_capacity(compute_tokens)
            target_reserve = self._construction_kv_reserve() + draft_tokens + 2
            target_kv_capacity = self._v2_solver_capacity(
                target_manager,
                max(1, context_length + 1),
                request_capacity,
                max_num_draft_tokens=target_reserve,
            )
            draft_kv_capacity = None
            if draft_manager is not None:
                target_extra = int(getattr(target_manager, "num_extra_kv_tokens", 0))
                draft_extra = int(getattr(draft_manager, "num_extra_kv_tokens", 0))
                draft_reserve = (
                    self._construction_kv_reserve()
                    + max(draft_tokens, self._manager_kv_reserve(draft_manager))
                    + 2
                    + max(0, target_extra - draft_extra)
                )
                draft_kv_capacity = self._v2_solver_capacity(
                    draft_manager,
                    max(1, context_length + 1),
                    request_capacity,
                    max_num_draft_tokens=draft_reserve,
                )
        else:
            backend = self._v1_planner_backend()
            microbatch_capacity = self._probe_microbatch_capacity(compute_requests, phase="decode")
            target_growth = (
                2
                + int(getattr(target_manager, "num_extra_kv_tokens", 0))
                + self._construction_kv_reserve()
                + max(draft_tokens, self._manager_kv_reserve(target_manager))
            )
            target_requests = self._make_planning_prefill_requests(
                context_length, request_capacity, max_tokens=target_growth
            )
            target_kv_capacity = self._probe_primary_capacity(target_requests)
            draft_kv_capacity = None
            if draft_manager is not None:
                draft_growth = (
                    2
                    + int(getattr(target_manager, "num_extra_kv_tokens", 0))
                    + self._construction_kv_reserve()
                    + max(draft_tokens, self._manager_kv_reserve(draft_manager))
                )
                draft_requests = self._make_planning_prefill_requests(
                    context_length, request_capacity, max_tokens=draft_growth
                )
                draft_kv_capacity = self._probe_draft_capacity(draft_requests, draft_manager)

        return self._make_axis_plan(
            backend=backend,
            requested_capacity=request_capacity,
            request_capacity=request_capacity,
            microbatch_capacity=microbatch_capacity,
            target_kv_capacity=target_kv_capacity,
            draft_kv_capacity=draft_kv_capacity,
            token_components=token_components,
            kv_snapshot=self._kv_snapshot(target_manager, draft_manager),
        )

    def _make_axis_plan(
        self,
        *,
        backend: str,
        requested_capacity: int,
        request_capacity: int,
        microbatch_capacity: int,
        target_kv_capacity: int,
        draft_kv_capacity: Optional[int],
        token_components: dict[str, int],
        kv_snapshot: dict,
    ) -> _AxisPlan:
        capacities = [
            ("request_capacity", request_capacity),
            ("microbatch_token_capacity", microbatch_capacity),
            ("target_kv_capacity", target_kv_capacity),
        ]
        if draft_kv_capacity is not None:
            capacities.append(("draft_kv_capacity", draft_kv_capacity))
        limiting_constraint, proposed_capacity = min(capacities, key=lambda item: item[1])
        return _AxisPlan(
            planner_backend=backend,
            requested_capacity=requested_capacity,
            request_capacity=request_capacity,
            microbatch_capacity=microbatch_capacity,
            target_kv_capacity=target_kv_capacity,
            draft_kv_capacity=draft_kv_capacity,
            proposed_capacity=proposed_capacity,
            limiting_constraint=limiting_constraint,
            token_components=token_components,
            kv_snapshot=kv_snapshot,
        )

    def _make_planning_prefill_requests(
        self, isl: int, batch_size: int, *, max_tokens: int
    ) -> list["LlmRequest"]:
        requests: list["LlmRequest"] = []
        output_config = OutputConfig()
        for request_id in self._planning_request_ids(batch_size):
            executor_request = Request(
                input_token_ids=[1] * max(1, isl),
                max_tokens=max(1, max_tokens),
                streaming=False,
                sampling_config=SamplingConfig(beam_width=self._executor.max_beam_width),
                end_id=-1,
                pad_id=0,
                output_config=output_config,
                return_all_generated_tokens=False,
                cache_salt=f"planner:{request_id}",
            )
            requests.append(
                executor_request_to_llm_request(
                    req_id=request_id,
                    executor_request=executor_request,
                    child_req_ids=None,
                    exclude_last_generation_logits=self._exclude_last_generation_logits(),
                )
            )
        return requests

    def _probe_primary_capacity(self, requests: list["LlmRequest"]) -> int:
        capacity_scheduler = getattr(self._executor.scheduler, "capacity_scheduler", None)
        if capacity_scheduler is None:
            raise RuntimeError("Active V1 scheduler does not expose a capacity scheduler")
        fitting, _, _ = capacity_scheduler.schedule_request(requests)
        return len(fitting)

    def _probe_draft_capacity(self, requests: list["LlmRequest"], draft_manager) -> Optional[int]:
        if draft_manager is None:
            return None
        capacity_scheduler = getattr(self._executor.scheduler, "capacity_scheduler", None)
        capacity_impl = getattr(capacity_scheduler, "impl", None)
        draft_impl = getattr(draft_manager, "impl", None)
        if capacity_impl is None or draft_impl is None:
            raise RuntimeError(
                "Separate draft KV planning requires the bound default capacity scheduler"
            )
        fitting, _, _ = capacity_impl(requests, draft_impl, None, None)
        return len(fitting)

    def _probe_microbatch_capacity(
        self, requests: list["LlmRequest"], *, phase: Literal["prefill", "decode"]
    ) -> int:
        microbatch_scheduler = getattr(self._executor.scheduler, "micro_batch_scheduler", None)
        if microbatch_scheduler is None:
            raise RuntimeError("Active V1 scheduler does not expose a microbatch scheduler")
        _, context_requests, generation_requests = microbatch_scheduler.schedule(
            requests, self._executor.inflight_req_ids
        )
        return len(context_requests if phase == "prefill" else generation_requests)

    def _v1_planner_backend(self) -> str:
        scheduler = self._executor.scheduler
        capacity_scheduler = getattr(scheduler, "capacity_scheduler", None)
        microbatch_scheduler = getattr(scheduler, "micro_batch_scheduler", None)
        if capacity_scheduler is None or microbatch_scheduler is None:
            raise RuntimeError("V1 self-benchmark requires split capacity/microbatch scheduling")
        return (
            "v1_bound_default"
            if getattr(capacity_scheduler, "impl", None) is not None
            and getattr(microbatch_scheduler, "impl", None) is not None
            else "v1_python"
        )

    def _formula_microbatch_capacity(self, per_request_tokens: int) -> int:
        max_num_tokens = int(getattr(self._executor, "max_num_tokens", 0) or 0)
        if max_num_tokens <= 0:
            return 1
        return min(self._max_decode_batch_size(), max_num_tokens // max(1, per_request_tokens))

    @staticmethod
    def _v2_solver_capacity(
        manager, base_tokens: int, request_capacity: int, *, max_num_draft_tokens: int
    ) -> int:
        capacity = 0
        for batch_size in range(1, request_capacity + 1):
            available = manager.get_num_available_tokens(
                token_num_upper_bound=base_tokens,
                batch_size=batch_size,
                max_num_draft_tokens=max_num_draft_tokens,
            )
            if available < base_tokens:
                break
            capacity = batch_size
        return capacity

    @staticmethod
    def _is_v2_manager(manager) -> bool:
        from .kv_cache_manager_v2 import KVCacheManagerV2

        return isinstance(manager, KVCacheManagerV2)

    def _planning_request_ids(self, count: int) -> list[int]:
        start = _PLANNING_REQUEST_ID_BASE + self._planning_request_id_cursor
        self._planning_request_id_cursor += count
        return list(range(start, start + count))

    def _kv_cache_manager(self):
        return self._executor.resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )

    def _draft_kv_cache_manager(self):
        return self._executor.resource_manager.get_resource_manager(
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER
        )

    def _construction_kv_reserve(self) -> int:
        return int(
            getattr(
                self._executor.model_engine,
                "max_draft_loop_tokens",
                self._executor.max_total_draft_tokens,
            )
        )

    @staticmethod
    def _manager_kv_reserve(manager) -> int:
        return int(getattr(manager, "_kv_reserve_draft_tokens", 0) or 0)

    @staticmethod
    def _kv_snapshot(target_manager, draft_manager) -> dict:
        def snapshot(manager) -> Optional[dict]:
            if manager is None:
                return None
            stats = manager.get_kv_cache_stats()
            result = {
                name: getattr(stats, name, None)
                for name in (
                    "max_num_blocks",
                    "free_num_blocks",
                    "used_num_blocks",
                    "tokens_per_block",
                )
            }
            per_window = getattr(stats, "num_free_blocks_per_window_size", None)
            if per_window is not None:
                result["num_free_blocks_per_window_size"] = dict(per_window)
            result["num_extra_kv_tokens"] = int(getattr(manager, "num_extra_kv_tokens", 0) or 0)
            result["kv_reserve_draft_tokens"] = int(
                getattr(manager, "_kv_reserve_draft_tokens", 0) or 0
            )
            return result

        return {"target": snapshot(target_manager), "draft": snapshot(draft_manager)}

    def _record_planner_reduction(
        self,
        case_type: str,
        isl: int,
        kv_read_tokens: int,
        context_length: int,
        plan: _AxisPlan,
    ) -> None:
        if plan.proposed_capacity >= plan.requested_capacity:
            return
        granularity = (
            self.config.decode_batch_granularity
            if case_type == "decode"
            else self.config.prefill_batch_granularity
        )
        requested_batch_sizes = self._sample_values(plan.requested_capacity, granularity)
        proposed_batch_sizes = self._sample_values(plan.proposed_capacity, granularity)
        omitted_batch_sizes = [
            batch_size
            for batch_size in requested_batch_sizes
            if batch_size not in proposed_batch_sizes
        ]
        event = {
            "event": "axis_reduced",
            "case_type": case_type,
            "isl": isl,
            "kv_read_tokens": kv_read_tokens,
            "context_length": context_length,
            "requested_max_batch_size": plan.requested_capacity,
            "proposed_max_batch_size": plan.proposed_capacity,
            "requested_batch_sizes": requested_batch_sizes,
            "proposed_batch_sizes": proposed_batch_sizes,
            "omitted_batch_sizes": omitted_batch_sizes,
            "limiting_constraint": plan.limiting_constraint,
        }
        self._planner_events.append(event)
        logger.info(
            "Self-benchmark planner reduced %s axis isl=%d kv_read=%d "
            "context=%d max_batch=%d->%d limiting=%s omitted=%s",
            case_type,
            isl,
            kv_read_tokens,
            context_length,
            plan.requested_capacity,
            plan.proposed_capacity,
            plan.limiting_constraint,
            omitted_batch_sizes,
        )

    @staticmethod
    def _require_positive_capacity(coordinate: str, plan: _AxisPlan) -> None:
        if plan.proposed_capacity <= 0:
            raise RuntimeError(f"No schedulable self-benchmark batch for {coordinate}")

    def _make_prefill_request(self, case: BenchmarkCase, trial_id: int, offset: int) -> Request:
        output_config = OutputConfig()
        output_config.return_perf_metrics = True
        cache_salt_id = case.cache_salt_id
        if cache_salt_id is None:
            cache_salt_id = self._cache_salt_id(case.case_id)
        request = Request(
            input_token_ids=[1] * max(1, case.isl),
            max_tokens=1,
            streaming=False,
            # Match the engine's beam width so the synthetic prefill batch is
            # shaped like real requests (the decode path already passes
            # max_beam_width to add_dummy_requests). Using the default beam=1
            # under a beam>1 engine would mis-shape the forward batch.
            sampling_config=SamplingConfig(beam_width=self._executor.max_beam_width),
            end_id=-1,
            pad_id=0,
            output_config=output_config,
            return_all_generated_tokens=False,
            cache_salt=f"{cache_salt_id}:{offset}",
        )
        return request

    def _exclude_last_generation_logits(self) -> bool:
        # Engine-wide flag derived from disable_overlap_scheduler and pp_size;
        # identical across TP ranks, so it does not perturb per-rank lockstep.
        return bool(getattr(self._executor, "should_exclude_last_generation_logits", False))

    def _mark_benchmark_request(self, request: "LlmRequest", trial_id: int) -> None:
        request.is_self_benchmark_request = True
        request.py_self_benchmark_trial_id = trial_id

    def _can_start_next_trial(
        self, active_requests: list["LlmRequest"], waiting_queue: WaitingQueue
    ) -> bool:
        if not self.active or self._current_trial is not None:
            return False
        if getattr(self._executor, "is_shutdown", False):
            self.request_interrupt()
            return False
        return not active_requests and len(waiting_queue) == 0

    def _peek_next_case(self) -> Optional[BenchmarkCase]:
        if self._next_case_index >= len(self._cases):
            return None
        return self._cases[self._next_case_index]

    @staticmethod
    def _request_id_of(request: "LlmRequest") -> int:
        request_id = getattr(request, "py_request_id", None)
        if request_id is None:
            request_id = getattr(request, "request_id")
        return int(request_id)

    def _belongs_to_current_trial(self, request: "LlmRequest", trial: BenchmarkTrial) -> bool:
        return (
            self._is_benchmark_request(request)
            and getattr(request, "py_self_benchmark_trial_id", None) == trial.trial_id
            and self._request_id_of(request) in trial.constructed_request_ids
        )

    @staticmethod
    def _schedule_signatures_match(requests: list["LlmRequest"], trial: BenchmarkTrial) -> bool:
        if len(requests) != len(trial.expected_schedule):
            return False
        for request in requests:
            request_id = SelfBenchmark._request_id_of(request)
            signature = trial.expected_schedule.get(request_id)
            if signature is None:
                return False
            try:
                total_tokens = int(request.get_num_tokens(0))
            except (AttributeError, TypeError, ValueError):
                return False
            if total_tokens != signature.total_tokens:
                return False
            if signature.phase == "prefill":
                estimated_reuse = int(getattr(request, "estimated_reusable_tokens", 0))
                prepared_reuse = max(
                    int(getattr(request, "context_current_position", 0)),
                    int(getattr(request, "prepopulated_prompt_len", 0)),
                )
                observed_reuse = max(estimated_reuse, prepared_reuse)
                context_chunk_size = int(getattr(request, "context_chunk_size", 0))
                if observed_reuse != signature.expected_cached_tokens or context_chunk_size not in (
                    signature.expected_context_chunk_size,
                    signature.total_tokens,
                ):
                    return False
            elif (
                int(getattr(request, "prompt_len", signature.expected_cached_tokens))
                != signature.expected_cached_tokens
            ):
                return False
        return True

    def _request_local_transition(self, outcome: BenchmarkOutcome, reason: str) -> None:
        trial = self._current_trial
        if trial is None or trial.state != TrialState.RUNNING:
            return
        if outcome <= trial.pending_outcome:
            return
        trial.pending_outcome = outcome
        trial.terminal_reason = reason
        trial.admission_reason = self._normalize_admission_reason(reason)
        trial.source_admission_reason = trial.admission_reason
        if outcome == BenchmarkOutcome.SKIP:
            self._warn_admission_mismatch(trial)

    @staticmethod
    def _normalize_admission_reason(reason: Optional[str]) -> str:
        prefix = (reason or "").split(":", 1)[0]
        normalized = {
            "cache_hit_validation_failed": "seed_cache_validation_failed",
            "insufficient_kv_cache_for_synthetic_decode": "kv_capacity",
            "synthetic_decode_kv_allocation_failed": "kv_capacity",
            "KV cache manager is not available": "kv_capacity",
            "synthetic_request_construction_failed": "request_construction_failed",
        }.get(prefix, prefix)
        slug = "".join(
            character.lower() if character.isalnum() else "_" for character in normalized
        ).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug or "admission_mismatch"

    def _warn_admission_mismatch(self, trial: BenchmarkTrial) -> None:
        if trial.admission_warning_emitted:
            return
        self._emit_admission_warning(self._build_admission_record(trial))
        trial.admission_warning_emitted = True

    def _emit_admission_warning(self, admission: dict) -> None:
        logger.warning(
            "Self-benchmark admission mismatch rank=%d trial=%d case=%d "
            "proposed=%d scheduled=%s executed=%s limiting=%s reason=%s source_reason=%s",
            self._local_planner_rank(),
            admission["trial_id"],
            admission["case_id"],
            admission["proposed_batch_size"],
            admission["scheduled_batch_size"],
            admission["executed_batch_size"],
            admission["limiting_constraint"],
            admission["normalized_reason"],
            admission["source_normalized_reason"],
        )

    @staticmethod
    def _trial_is_exactly_complete(trial: BenchmarkTrial) -> bool:
        expected = trial.expected_request_ids
        return (
            trial.constructed_request_ids == expected
            and trial.scheduled_request_ids == expected
            and trial.executed_request_ids == expected
            and trial.completed_request_ids == expected
            and trial.constructed_request_ids <= trial.terminated_request_ids
            and bool(trial.stats)
            and (
                trial.case.case_type != "prefill"
                or trial.case.kv_read_tokens == 0
                or trial.cache_hit_validated is True
            )
        )

    def _begin_drain(
        self,
        target: DrainTarget,
        reason: Optional[str],
        origin_rank: Optional[int],
    ) -> None:
        trial = self._current_trial
        if trial is None:
            return
        source_reason = self._normalize_admission_reason(reason or trial.terminal_reason)
        local_skip = trial.pending_outcome == BenchmarkOutcome.SKIP
        trial.source_admission_reason = source_reason
        if target == DrainTarget.SKIP and not local_skip:
            trial.rank_consensus_applied = True
            trial.admission_reason = "rank_consensus_skip"
        elif trial.admission_reason is None:
            trial.admission_reason = source_reason
        trial.state = TrialState.DRAINING
        trial.pending_outcome = (
            BenchmarkOutcome.SKIP if target == DrainTarget.SKIP else BenchmarkOutcome.ABORT
        )
        trial.terminal_reason = reason or trial.terminal_reason
        trial.origin_rank = origin_rank
        self._drain_target = target
        self._run_state = RunState.DRAINING
        if target == DrainTarget.SKIP and not local_skip:
            self._warn_admission_mismatch(trial)

    def _finalize_global_complete(self) -> None:
        trial = self._current_trial
        if trial is None:
            return

        if trial.state == TrialState.DRAINING:
            if not trial.constructed_request_ids <= trial.terminated_request_ids:
                return
            target = self._drain_target
            if target == DrainTarget.SKIP:
                trial.state = TrialState.SKIPPED
                self._admission_records[trial.trial_id] = self._build_admission_record(trial)
                skipped_case = trial.case
                skipped_trial_id = trial.trial_id
                dependent_case = self._dependent_measure_case(trial.case)
                if dependent_case is not None:
                    skipped_case = dependent_case
                    skipped_trial_id = self._next_trial_id
                    self._next_trial_id += 1
                    self._next_case_index += 1
                    dependent_admission = self._build_admission_record(
                        trial,
                        case=skipped_case,
                        trial_id=skipped_trial_id,
                        normalized_reason="dependent_seed_skipped",
                        scheduled_batch_size=0,
                        executed_batch_size=0,
                    )
                    self._admission_records[skipped_trial_id] = dependent_admission
                    self._emit_admission_warning(dependent_admission)
                self._skipped_cases.append(
                    {
                        "trial_id": skipped_trial_id,
                        "case": asdict(skipped_case),
                        "reason": trial.terminal_reason,
                        "origin_rank": trial.origin_rank,
                    }
                )
                self._current_trial = None
                self._drain_target = None
                self._run_state = RunState.RUNNING
                if self._next_case_index >= len(self._cases):
                    self._finish_complete()
                return
            trial.state = TrialState.FAILED
            self._admission_records[trial.trial_id] = self._build_admission_record(trial)
            self._abort = {
                "trial_id": trial.trial_id,
                "case": asdict(trial.case),
                "reason": trial.terminal_reason,
                "origin_rank": trial.origin_rank,
            }
            self._current_trial = None
            self._drain_target = None
            if target == DrainTarget.INTERRUPT:
                self._finish_interrupted()
            else:
                self._finish_aborted()
            return

        if not self._trial_is_exactly_complete(trial):
            return
        trial.state = TrialState.COMPLETE
        admission = self._build_admission_record(trial)
        self._admission_records[trial.trial_id] = admission
        if self._should_record_current_trial():
            self._results.append(
                BenchmarkTrialResult(
                    trial_id=trial.trial_id,
                    case=trial.case,
                    iteration_stats=tuple(trial.stats),
                    observed_kv_read_tokens=trial.observed_kv_read_tokens,
                    cache_hit_validated=trial.cache_hit_validated,
                    admission=admission,
                )
            )
        self._current_trial = None
        if self._next_case_index >= len(self._cases):
            self._finish_complete()

    def _dependent_measure_case(self, completed_case: BenchmarkCase) -> Optional[BenchmarkCase]:
        if completed_case.case_type != "prefill_seed":
            return None
        candidate = self._peek_next_case()
        if candidate is None or candidate.case_type != "prefill":
            return None
        if (
            candidate.cache_salt_id != completed_case.cache_salt_id
            or candidate.kv_read_tokens != completed_case.kv_read_tokens
            or candidate.batch_size != completed_case.batch_size
        ):
            return None
        return candidate

    def _record_cache_hit_validation(self, requests: list["LlmRequest"], stats: dict) -> None:
        trial = self._current_trial
        if trial is None:
            return
        case = trial.case
        if case.case_type != "prefill" or case.kv_read_tokens <= 0:
            return
        observed = self._observed_cached_tokens(requests, trial)
        trial.observed_kv_read_tokens = observed
        validated = observed is not None and observed >= case.kv_read_tokens
        trial.cache_hit_validated = validated
        stats["selfBenchmark"] = {
            "expectedKvReadTokens": case.kv_read_tokens,
            "observedCachedTokens": observed,
            "cacheHitValidated": validated,
        }
        if not validated:
            self._request_local_transition(BenchmarkOutcome.SKIP, "cache_hit_validation_failed")

    @staticmethod
    def _observed_cached_tokens(
        requests: list["LlmRequest"], trial: BenchmarkTrial
    ) -> Optional[int]:
        observed = [
            int(getattr(req, "cached_tokens"))
            for req in requests
            if (
                getattr(req, "py_self_benchmark_trial_id", None) == trial.trial_id
                and hasattr(req, "cached_tokens")
            )
        ]
        if len(observed) != trial.case.batch_size:
            return None
        return min(observed)

    def _should_record_current_trial(self) -> bool:
        return self._current_trial is not None and self._current_trial.case.case_type not in (
            "warmup",
            "prefill_seed",
        )

    @staticmethod
    def _sanitize_queue_counters(stats: dict) -> None:
        stats["numQueuedRequests"] = 0
        inflight_stats = stats.get("inflightBatchingStats", {})
        inflight_stats["numQueuedContextRequests"] = 0
        inflight_stats["numQueuedCtxTokens"] = 0
        inflight_stats["numQueuedGenRequests"] = 0
        inflight_stats["numQueuedGenKvTokens"] = 0

    @staticmethod
    def _sample_values(max_value: int, granularity: int) -> list[int]:
        max_value = max(1, int(max_value))
        if granularity <= 1:
            return [max_value]
        values = {round(1 + i * (max_value - 1) / (granularity - 1)) for i in range(granularity)}
        return sorted(max(1, min(max_value, int(value))) for value in values)

    def _kv_read_values_for_isl(self, isl: int) -> list[int]:
        block_size = self._tokens_per_block()
        if block_size <= 0 or not self._enable_block_reuse():
            return [0]
        max_read_tokens = ((max(0, isl - 1)) // block_size) * block_size
        if max_read_tokens == 0:
            return [0]
        return self._sample_block_aligned_values(
            max_read_tokens, self.config.prefill_kv_read_granularity, block_size
        )

    @staticmethod
    def _sample_block_aligned_values(
        max_value: int, granularity: int, block_size: int
    ) -> list[int]:
        block_values = list(range(0, max_value + 1, block_size))
        if granularity <= 1:
            return [0]
        if len(block_values) <= granularity:
            return block_values
        sampled_indices = {
            round(i * (len(block_values) - 1) / (granularity - 1)) for i in range(granularity)
        }
        return [block_values[i] for i in sorted(sampled_indices)]

    def _max_prefill_isl(self) -> int:
        max_isl = self._bounded_length(default=1)
        max_seq_len = getattr(self._executor, "max_seq_len", None)
        if isinstance(max_seq_len, int) and 0 < max_seq_len < 2**30:
            max_isl = min(max_isl, max_seq_len - 1)
        if max_isl < 1:
            raise RuntimeError("Self-benchmark requires max_seq_len >= 2")
        return max_isl

    def _max_decode_context_length(self) -> int:
        max_context_length = self._bounded_length(default=3) - 2
        if max_context_length < 1:
            raise RuntimeError("Self-benchmark decode requires max_seq_len >= 3")
        return max_context_length

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
            ResourceManagerType.KV_CACHE_MANAGER
        )
        if kv_cache_manager is None:
            return 0
        kv_stats = kv_cache_manager.get_kv_cache_stats()
        return int(getattr(kv_stats, "tokens_per_block", 0) or 0)

    def _enable_block_reuse(self) -> bool:
        kv_cache_manager = self._executor.resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        return bool(getattr(kv_cache_manager, "enable_block_reuse", False))

    def _limits(self) -> dict:
        kv_cache_manager = self._executor.resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        kv_stats = kv_cache_manager.get_kv_cache_stats() if kv_cache_manager is not None else None
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

    def _request_id(self, trial_id: int, offset: int) -> int:
        return _BENCHMARK_REQUEST_ID_BASE + trial_id * self._id_stride + offset

    @staticmethod
    def _cache_salt_id(case_id: int) -> int:
        return _BENCHMARK_REQUEST_ID_BASE + 500_000 + case_id

    @staticmethod
    def _is_benchmark_request(request: "LlmRequest") -> bool:
        return bool(getattr(request, "is_self_benchmark_request", False))
