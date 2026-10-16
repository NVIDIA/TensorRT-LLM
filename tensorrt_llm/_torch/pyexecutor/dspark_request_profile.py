# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit offline request ownership over the unchanged executor lifecycle."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm._torch.pyexecutor.dspark_profile_engine import (
    DSparkCostProfileEngine,
    _valid_previous_state,
)
from tensorrt_llm._torch.pyexecutor.llm_request import ExecutorRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.py_executor_creator import (
    _RuntimeConstructors,
    create_py_executor,
)

if TYPE_CHECKING:
    from ray.actor import ActorHandle

__all__ = ["OwnedDSparkProfileExecutor", "create_profile_executor"]


def _token_snapshot(request: Any) -> dict[str, Any]:
    tokens = list(request.get_tokens(0))
    prompt = tokens[: request.py_orig_prompt_len]
    return {
        "request_id": request.py_request_id,
        "process_local_request_object": id(request),
        "seq_slot": request.py_seq_slot,
        "batch_idx": request.py_batch_idx,
        "decoding_iter": request.py_decoding_iter,
        "tokens": tokens,
        "original_input_sha256": hashlib.sha256(
            json.dumps(prompt, separators=(",", ":")).encode()
        ).hexdigest(),
        "total_drafted": request.py_total_draft_tokens,
        "total_accepted": request.py_total_accepted_draft_tokens,
    }


@dataclass
class _ForwardJoin:
    generation_id: int
    executor_iteration: int
    route: str
    requests: tuple[Any, ...]
    before: list[dict[str, Any]]
    sample_index: int | None = None
    update_before: list[dict[str, Any]] | None = None
    sampler_update_applied: list[bool] | None = None
    after: list[dict[str, Any]] | None = None
    host_step_time_ms: float | None = None


class OwnedDSparkProfileEngine(DSparkCostProfileEngine):
    """Only a source-owned executor boundary may enter a real request forward."""

    def forward(
        self, batch, resource_manager, new_tensors_device=None, cache_indirection_buffer=None
    ):
        if self.is_warmup:
            return PyTorchModelEngine.forward(
                self,
                batch,
                resource_manager,
                new_tensors_device,
                cache_indirection_buffer=cache_indirection_buffer,
            )
        boundary = getattr(self, "_owned_forward_boundary", None)
        if boundary is None or boundary[2] is not batch or boundary[3] is not new_tensors_device:
            raise RuntimeError("No original-request owner boundary; ordinary serving denied")
        route, cell, _, _ = boundary
        if route == "cell":
            return self.forward_profile_cell(
                cell,
                batch,
                resource_manager,
                new_tensors_device=new_tensors_device,
                cache_indirection_buffer=cache_indirection_buffer,
            )
        # Genuine prefill/first-draft/short tail/peer-not-ready transitions use
        # the unchanged native path and produce no profile event record.
        for request in batch.generation_requests:
            request.py_verify_len = None
        self.cuda_graph_runner.agreed_ragged_bucket = None
        return PyTorchModelEngine.forward(
            self,
            batch,
            resource_manager,
            new_tensors_device,
            cache_indirection_buffer=cache_indirection_buffer,
        )


class OwnedDSparkProfileExecutor(PyExecutor):
    """Original executor loops plus fixed-cell provenance; not a serving factory."""

    def __init__(self, *args, **kwargs):
        self._owned_generation = None
        self._owned_generation_ids = set()
        self._owned_joins = []
        self._owned_pending = {}
        self._owned_last_join = None
        self._owned_request_objects = {}
        self._owned_final_requests = {}
        self._owned_failures = []
        self._owned_kv_capacity_enqueue = None
        super().__init__(*args, **kwargs)

    @contextmanager
    def _profiler(self):
        """Join the original complete-loop timer, without new events or waits.

        The parent step reads the just-finished loop at the next loop boundary.
        An overlap sampler may consume another batch there; its forward-event
        duration and iterLatencyMS are not substitutes for this loop's cost.
        """
        with super()._profiler() as step:
            if self.is_warmup or self.model_engine._cost_profile.measurement_scope != "whole-step":
                yield step
                return
            if not self.enable_iter_perf_stats:
                raise ValueError("Whole-step costs require existing enable_iter_perf_stats")
            previous_iteration = None

            def advance():
                nonlocal previous_iteration
                step()
                join = self._owned_joins[-1] if self._owned_joins else None
                if (
                    join is not None
                    and join.executor_iteration == previous_iteration
                    and join.route == "cell"
                ):
                    duration = self._latest_host_step_time_ms
                    if (
                        isinstance(duration, bool)
                        or not isinstance(duration, (int, float))
                        or not math.isfinite(duration)
                        or duration <= 0
                    ):
                        raise ValueError("Original cell loop has no finite positive hostStepTimeMS")
                    if join.host_step_time_ms is not None:
                        raise RuntimeError("Original cell loop timing was joined twice")
                    join.host_step_time_ms = duration
                previous_iteration = self.iter_counter

            yield advance

    def begin_profile_generation(self, generation_id: int, cell: Any) -> None:
        if type(generation_id) is not int or generation_id < 0:
            raise ValueError("Original protocol generation ID must be a nonnegative integer")
        self.model_engine._cost_profile.require_cell(cell)
        with self.control_action(drain=True):
            if self._owned_pending or generation_id in self._owned_generation_ids:
                raise RuntimeError("Undrained or repeated original generation")
            if self._owned_generation is not None:
                raise RuntimeError("End the previous original generation before another")
            self._owned_generation_ids.add(generation_id)
            self._owned_generation = (generation_id, cell)

    def enqueue_requests(self, requests, result_wait_queue=None):
        if self._owned_generation is None and not self.is_warmup:
            authorization = self._owned_kv_capacity_enqueue
            if (
                authorization is None
                or requests is not authorization[0]
                or authorization[1]
                or result_wait_queue is not None
            ):
                raise RuntimeError("Ordinary requests require an explicit offline generation")
            # The original KV estimator enqueues this exact internal list before
            # setting warmup. Consume before delegation, even if enqueue fails.
            self._owned_kv_capacity_enqueue = (requests, True)
        return super().enqueue_requests(requests, result_wait_queue)

    def enqueue_request(
        self, request: ExecutorRequest, result_wait_queue: "Optional[ActorHandle]" = None
    ) -> int:
        if self._owned_generation is None and not self.is_warmup:
            raise RuntimeError("Ordinary requests require an explicit offline generation")
        return super().enqueue_request(request, result_wait_queue)

    def _handle_errors(
        self,
        error_msg=None,
        *,
        requests=None,
        charge_budget=True,
        fatal_is_collective_aligned=False,
    ):
        if self._owned_generation is not None:
            self._owned_failures.append(str(error_msg or "Original executor error"))
        return super()._handle_errors(
            error_msg,
            requests=requests,
            charge_budget=charge_budget,
            fatal_is_collective_aligned=fatal_is_collective_aligned,
        )

    def _forward_step(self, batch, new_tensors_device=None):
        if self.is_warmup:
            return super()._forward_step(batch, new_tensors_device)
        generation = self._owned_generation
        engine = self.model_engine
        requests = tuple(request for request in batch.all_requests() if not request.is_dummy)
        ready = generation is not None
        generation_id, cell = generation if ready else (-1, None)
        eligible = (
            ready
            and batch.num_context_requests == 0
            and batch.num_encoder_requests == 0
            and len(batch.generation_requests) == cell.graph_batch_size
            and all(
                not r.is_dummy
                and not r.py_is_first_draft
                and r.py_beam_width == 1
                and r.py_seq_slot is not None
                for r in batch.generation_requests
            )
            and len({r.py_request_id for r in batch.generation_requests})
            == len(batch.generation_requests)
            and _valid_previous_state(
                new_tensors_device, batch.generation_requests, engine._cost_profile.physical_k
            )
        )
        row = [
            int(ready),
            generation_id,
            self.iter_counter,
            cell.graph_batch_size if ready else -1,
            cell.verifier_budget if ready else -1,
            int(eligible),
        ]
        rows = self.dist.tp_allgather_int64(row).tolist() if self.dist.tp_size > 1 else [row]
        if (
            len(rows) != self.dist.tp_size
            or any(len(r) != 6 or r[0] != 1 for r in rows)
            or any(r[:5] != rows[0][:5] for r in rows)
        ):
            raise RuntimeError("All ranks must agree on the original generation/iteration/cell")
        route = "cell" if all(r[5] == 1 for r in rows) else "native-transition"
        for request in requests:
            key = (generation_id, request.py_request_id)
            previous = self._owned_request_objects.setdefault(key, request)
            if previous is not request:
                raise RuntimeError("Request ID reused with a different original incarnation")
        join = _ForwardJoin(
            generation_id,
            self.iter_counter,
            route,
            requests,
            [_token_snapshot(request) for request in requests],
        )
        before_records = len(engine.cuda_graph_runner._profile_records)
        if getattr(engine, "_owned_forward_boundary", None) is not None:
            raise RuntimeError("Overlapping owner forward boundaries")
        engine._owned_forward_boundary = (route, cell, batch, new_tensors_device)
        try:
            output = super()._forward_step(batch, new_tensors_device)
            if output is None:
                raise RuntimeError("Original forward failed; no observation promotion")
            count = len(engine.cuda_graph_runner._profile_records)
            if count != before_records + int(route == "cell"):
                raise RuntimeError("Native transition or fixed-cell replay count differs")
            if route == "cell":
                record = engine.cuda_graph_runner._profile_records[-1]
                if tuple(record["request_ids"]) != tuple(
                    r.py_request_id for r in batch.generation_requests
                ):
                    raise RuntimeError("Captured replay and original request identities differ")
                join.sample_index = record["sample_index"]
            self._owned_joins.append(join)
            self._owned_last_join = join
            return output
        except BaseException as error:
            self._owned_failures.append(type(error).__name__ + ": " + str(error))
            raise
        finally:
            engine._owned_forward_boundary = None

    def _sample_async(self, batch, output):
        state = super()._sample_async(batch, output)
        if self.is_warmup:
            return state
        join = self._owned_last_join
        if state is None or join is None or id(state) in self._owned_pending:
            raise RuntimeError("Missing or reused original sampler state")
        if any(
            not any(request is original for original in join.requests)
            for request in state.requests
            if not request.is_dummy
        ):
            raise RuntimeError("Sampler state belongs to another original forward")
        if join.sample_index is not None and tuple(state.requests) != join.requests:
            raise RuntimeError("Measured sampler requests differ from original decode forward")
        self._owned_pending[id(state)] = (state, join)
        self._owned_last_join = None
        return state

    def _update_requests(self, state, resource_manager=None):
        if self.is_warmup:
            return super()._update_requests(state, resource_manager)
        pending = self._owned_pending.get(id(state))
        if pending is None or pending[0] is not state:
            raise RuntimeError("Unknown or already consumed original sampler state")
        join = pending[1]
        join.update_before = [_token_snapshot(request) for request in join.requests]
        completed_before_update = [
            request.state == LlmRequestState.GENERATION_COMPLETE for request in join.requests
        ]
        super()._update_requests(state, resource_manager)
        if join.sample_index is not None:
            self.model_engine.cuda_graph_runner.complete_sample(join.sample_index)
        # Read only after the original sampler sync/token/KV/stop update and
        # original accumulation. Under overlap this is the previous iteration.
        join.after = [_token_snapshot(request) for request in join.requests]
        join.sampler_update_applied = [
            after["decoding_iter"] > before["decoding_iter"]
            for before, after in zip(join.update_before, join.after)
        ]
        if join.sample_index is not None and any(
            not applied and not completed
            for applied, completed in zip(join.sampler_update_applied, completed_before_update)
        ):
            self._owned_failures.append("Original measured request update did not advance")
            raise RuntimeError("Original measured request update did not advance")
        del self._owned_pending[id(state)]

    def _free_request_resources(self, request):
        generation = self._owned_generation
        final = None
        if generation is not None and not request.is_dummy:
            key = (generation[0], request.py_request_id)
            if (
                self._owned_request_objects.get(key) is request
                and request.state == LlmRequestState.GENERATION_COMPLETE
            ):
                final = (key, _token_snapshot(request))
        result = super()._free_request_resources(request)
        if final is not None:
            self._owned_final_requests[final[0]] = final[1]
        return result

    def _profile_drain_predicates(self):
        previous = self.previous_batch
        state = previous.sample_state if previous is not None else None
        pending = self._owned_pending.get(id(state)) if state is not None else None
        join = pending[1] if pending is not None else None
        return {
            "generation_id": self._owned_generation[0] if self._owned_generation else None,
            "failures": len(self._owned_failures),
            "active_requests": len(self.active_requests),
            "waiting_requests": len(self.waiting_queue),
            "pending_samples": len(self._owned_pending),
            "last_join_present": self._owned_last_join is not None,
            "forward_boundary_present": (
                getattr(self.model_engine, "_owned_forward_boundary", None) is not None
            ),
            "previous_batch_present": previous is not None,
            "pending_matches_previous": (
                state is not None
                and len(self._owned_pending) == 1
                and pending is not None
                and pending[0] is state
                and any(join is original for original in self._owned_joins)
                and self._owned_generation is not None
                and join.generation_id == self._owned_generation[0]
                and join.update_before is None
                and join.sampler_update_applied is None
                and join.after is None
            ),
        }

    def end_profile_generation(self, *, control=None) -> dict[str, Any]:
        if self._owned_generation is None:
            raise RuntimeError("No original generation to drain")
        with self.control_action(drain=True):
            before = self._profile_drain_predicates()
            if control is None:
                raise RuntimeError(
                    "Offline profile drain needs the original gang-control communicator"
                )
            peers = control.allgather(before)
            quiescent = all(
                row["failures"] == 0
                and row["active_requests"] == 0
                and row["waiting_requests"] == 0
                and not row["last_join_present"]
                and not row["forward_boundary_present"]
                and row["generation_id"] == before["generation_id"]
                for row in peers
            )
            no_tail = all(row["pending_samples"] == 0 for row in peers)
            exact_tail = all(
                row["pending_samples"] == 1 and row["pending_matches_previous"] for row in peers
            )
            if not quiescent or not (no_tail or exact_tail):
                raise RuntimeError(
                    f"Original generation has not cleanly drained: gang_predicates={peers!r}"
                )
            consumed = False
            if exact_tail:
                # All ranks are paused in the original exclusive control action.
                # Delegate every sampler/update/resource/timing operation to the
                # existing executor helper; do not clear pending joins directly.
                self._consume_previous_batch_for_rebalance()
                consumed = True
            after = self._profile_drain_predicates()
            if (
                self._owned_pending
                or self._owned_last_join is not None
                or self._owned_failures
                or self.active_requests
                or len(self.waiting_queue)
            ):
                raise RuntimeError(
                    "Original generation has not cleanly drained: "
                    f"before={before!r}, after={after!r}"
                )
            generation_id = self._owned_generation[0]
            keys = [key for key in self._owned_request_objects if key[0] == generation_id]
            if not keys or any(key not in self._owned_final_requests for key in keys):
                raise RuntimeError("Original request final resource-release joins are incomplete")
            observation = self.model_engine.profile_observations()
            by_sample = {
                join.sample_index: join
                for join in self._owned_joins
                if join.sample_index is not None
            }
            if set(by_sample) != {row["sample_index"] for row in observation["samples"]}:
                raise RuntimeError("Completed replay events and lifecycle joins differ")
            joins = [
                {
                    "generation_id": join.generation_id,
                    "executor_iteration": join.executor_iteration,
                    "route": join.route,
                    "sample_index": join.sample_index,
                    "before": join.before,
                    "update_before": join.update_before,
                    "sampler_update_applied": join.sampler_update_applied,
                    "after": join.after,
                    "host_step_time_ms": join.host_step_time_ms,
                }
                for join in self._owned_joins
                if join.generation_id == generation_id
            ]
            if any(join["after"] is None for join in joins):
                raise RuntimeError("Original sampling/update join missing")
            if self.model_engine._cost_profile.measurement_scope == "whole-step" and any(
                join["sample_index"] is not None and join["host_step_time_ms"] is None
                for join in joins
            ):
                raise RuntimeError("Measured cell lacks its original complete-loop timing")
            generation_samples = {
                join["sample_index"] for join in joins if join["sample_index"] is not None
            }
            if not generation_samples:
                raise RuntimeError("Prescribed generation did not execute a measured graph cell")
            observation = {
                **observation,
                "samples": [
                    row
                    for row in observation["samples"]
                    if row["sample_index"] in generation_samples
                ],
            }
            self._owned_generation = None
            return {
                "schema": "dspark-original-request-profile-generation-v1",
                "generation_id": generation_id,
                "drain": {
                    "before": before,
                    "after": after,
                    "original_previous_batch_consumed": consumed,
                },
                "observation": observation,
                "lifecycle_joins": joins,
                "original_final_requests": [self._owned_final_requests[key] for key in keys],
                "original_shutdown_still_required": True,
                "task_accuracy_qualified": False,
                "native_or_gpu_qualified": False,
                "full_policy_h": None,
                "production_admission_qualified": False,
            }


class _ProfileKvCacheCreator(KvCacheCreator):
    def configure_kv_cache_capacity(self, py_executor=None) -> None:
        if py_executor is None or self._skip_est:
            return super().configure_kv_cache_capacity(py_executor)
        if (
            not isinstance(py_executor, OwnedDSparkProfileExecutor)
            or py_executor._owned_generation is not None
            or py_executor._owned_kv_capacity_enqueue is not None
        ):
            raise RuntimeError("Invalid or nested offline KV startup phase")
        py_executor._owned_kv_capacity_enqueue = (self._dummy_reqs, False)
        try:
            return super().configure_kv_cache_capacity(py_executor)
        finally:
            py_executor._owned_kv_capacity_enqueue = None


def create_profile_executor(*, llm_args, profile, checkpoint_dir=None, tokenizer=None):
    """Construct only the prescribed offline owner; never a serving executor.

    No source pins, private manifests, historical receipts or global cloning are
    required. Runtime and loaded-byte identity is measured by the public driver.
    Normal factory construction, resource/KV estimation and loops are retained.
    """
    if llm_args.enable_resource_governor:
        raise ValueError("The offline profile owner does not support resource-governor control")
    constructors = _RuntimeConstructors(
        model_engine=partial(OwnedDSparkProfileEngine, profile=profile),
        executor=OwnedDSparkProfileExecutor,
        kv_cache_creator=_ProfileKvCacheCreator,
    )
    return create_py_executor(
        llm_args=llm_args,
        checkpoint_dir=checkpoint_dir,
        tokenizer=tokenizer,
        runtime_constructors=constructors,
    )
