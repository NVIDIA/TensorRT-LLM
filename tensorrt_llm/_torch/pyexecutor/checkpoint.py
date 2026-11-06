# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resident checkpoint protection for a fixed, single-node MPI cohort."""

import os
import socket
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import fields
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._utils import mpi_comm, set_thread_local_mpi_comm
from tensorrt_llm.executor.checkpoint import (
    CheckpointContext,
    CheckpointProfile,
    CheckpointStep,
    RankCheckpointEvidence,
    checkpoint_digest,
)

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    from .py_executor import PyExecutor

CHECKPOINT_ENABLE_ENV = "TRTLLM_ENABLE_CHECKPOINT_LIFECYCLE"
_ACTION_TAG = 0
_ACK_TAG = 1


def _checkpoint_configuration(args: "TorchLlmArgs") -> tuple[dict, dict]:
    """Separate shared settings from resolved rank-local capacity.

    Args:
        args: Final engine arguments after native memory budgeting.

    Returns:
        Detached shared configuration and local capacity. Both participate in
        profile identity, but only shared settings must match across ranks.
    """
    shared = args.model_dump(mode="json", exclude={"_mpi_session", "env_overrides"})
    local = {"max_gpu_total_bytes": shared["kv_cache_config"].pop("max_gpu_total_bytes")}
    return shared, local


class NativeCheckpoint:
    """Own a persistent scheduler park and evidence on each MPI rank.

    Construct collectively on the final executor's startup thread, before any
    event loop starts. The dedicated communicator is not an MPI failure domain.
    The supervisor must remain outside the worker job's effective abort scope.
    """

    def __init__(self, engine: "PyExecutor") -> None:
        """Resolve the cohort and reject unsupported resource owners collectively.

        Args:
            engine: Warmed final PyExecutor, with worker threads not yet started.

        Raises:
            RuntimeError: If any rank has an unsupported owner or configuration.
        """
        self.engine = engine
        self.session_id: str | None = None
        self._source_session_id: str | None = None
        self._parked: AbstractContextManager | None = None
        self._baseline: dict | None = None
        self._command_lock = threading.Lock()
        self._stop = threading.Event()
        self._listener: threading.Thread | None = None
        self._failed = False
        self._phase: CheckpointStep | None = None
        self._validation_replays = 0
        self._completed: dict[tuple[str, str], tuple[str, dict]] = {}
        self._pending_sends: list = []
        if engine._disable_mpi:
            raise RuntimeError("Checkpoint lifecycle currently requires MPI.")
        self._comm = mpi_comm().Dup() if engine.dist.world_size > 1 else None
        local = {"rank": engine.dist.rank, "host": socket.gethostname(), "error": None}
        try:
            self._check_profile()
            local["device"] = str(torch.cuda.get_device_properties(engine.device_id).uuid)
            local["model"] = type(engine.model_engine.model).__name__
            args, local["capacity"] = _checkpoint_configuration(engine.llm_args)
            local["profile"] = checkpoint_digest(args)
            runner = engine.model_engine.cuda_graph_runner
            if runner is None or not runner.graphs:
                raise RuntimeError("Checkpoint requires final warmed CUDA graphs.")
        except Exception as exc:
            # Every rank must reach the startup agreement even on a local probe failure.
            local["error"] = str(exc)
        cohort = self._comm.allgather(local) if self._comm is not None else [local]
        if (
            any(item["error"] is not None for item in cohort)
            or len({item["host"] for item in cohort}) != 1
            or sorted(item["rank"] for item in cohort) != list(range(engine.dist.world_size))
        ):
            raise RuntimeError(f"Unsupported checkpoint cohort: {cohort}")
        if len({item["profile"] for item in cohort}) != 1:
            raise RuntimeError("MPI ranks resolved different checkpoint configurations.")
        self.profile = CheckpointProfile(
            profile_digest=checkpoint_digest(
                {
                    "shared": local["profile"],
                    "rank_capacities": [item["capacity"] for item in cohort],
                }
            ),
            cohort_digest=checkpoint_digest(cohort),
            ranks=list(range(engine.dist.world_size)),
        )
        self._host = local["host"]
        self._device_uuid = local["device"]
        runner.register_pre_replay_hook(self._observe_replay)

    def _observe_replay(
        self, key: object, num_tokens: int, static_tensors: dict, current_inputs: dict
    ) -> None:
        """Count actual graph submissions during the private validation window.

        Args:
            key: Graph selected by the existing runner.
            num_tokens: Token count for this replay.
            static_tensors: Runner-owned captured inputs, not modified here.
            current_inputs: Current generation inputs, not modified here.
        """
        if self._phase == CheckpointStep.VALIDATION_START:
            self._validation_replays += 1

    def _check_profile(self) -> None:
        """Prove absence of unqualified owners, rather than infer it from idleness."""
        from .kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
        from .resource_manager import ResourceManagerType

        engine = self.engine
        args = engine.llm_args
        with engine._profile_state_lock:
            if (
                engine._profile_enabled
                or engine._runtime_profile_pending_start_iter is not None
                or engine.profile_start_iters
            ):
                raise RuntimeError("Checkpoint has an unqualified profiling window.")
        if (
            engine._disable_mpi
            or engine.dist.world_size not in (1, 2)
            or engine.dist.pp_size != 1
            or engine.dist.cp_size != 1
            or engine.enable_attention_dp
            or args.num_serve_frontends != 1
            or args.num_postprocess_workers != 0
            or args.enable_resource_governor
            or args.allreduce_strategy != "NCCL"
            or args.torch_compile_config is not None
            or args.guided_decoding_backend is not None
            or type(engine.model_engine.model).__name__ != "Qwen3ForCausalLM"
        ):
            raise RuntimeError(
                "Checkpoint currently requires dense Qwen3, MPI TP, PP1/CP1, "
                "ordinary NCCL and one text frontend without auxiliary owners."
            )
        if (
            engine.kv_cache_transceiver is not None
            or engine.kv_connector_manager is not None
            or engine.draft_model_engine is not None
            or engine.drafter is not None
            or engine.dwdp_manager is not None
            or bool(engine._openengine_discovery)
        ):
            raise RuntimeError(
                "Checkpoint cannot prove absence of external KV or auxiliary owners."
            )
        manager = engine.kv_cache_manager
        events = args.kv_cache_config.kv_events_config
        if events is not None and events.enable_kv_cache_events:
            raise RuntimeError("Checkpoint has an unqualified external KV event publisher.")
        if (
            type(manager) is not KVCacheManagerV2
            or manager._sharing is not None
            or len(manager.impl.cache_tier_list) != 1
            or engine.enable_kv_cache_reuse
        ):
            raise RuntimeError(
                "Resident checkpoint requires dense GPU-only V2 without reuse or lender."
            )
        allowed = {ResourceManagerType.KV_CACHE_MANAGER, ResourceManagerType.SEQ_SLOT_MANAGER}
        if any(
            manager is not None and resource_type not in allowed
            for resource_type, manager in engine.resource_manager.resource_managers.items()
        ):
            raise RuntimeError("Checkpoint has an unqualified resource manager.")
        if (
            os.environ.get("TLLM_NCCL_SYMMETRIC_ZERO_COPY") != "0"
            or os.environ.get("TRTLLM_PROXY_BATCH_REQUESTS") != "0"
            or os.environ.get("TRTLLM_GPU_KEEPALIVE", "0") != "0"
        ):
            raise RuntimeError(
                "Checkpoint requires explicit exclusion of optional collective "
                "windows, proxy batching and GPU keepalive."
            )

    def start(self) -> None:
        """Start the peer command listener after collective startup is complete."""
        if self._comm is not None and self.engine.dist.rank != 0:
            self._listener = threading.Thread(
                target=self._listen, daemon=True, name="checkpoint-control"
            )
            self._listener.start()

    def _listen(self) -> None:
        """Process bounded checkpoint commands independently of the parked scheduler."""
        torch.cuda.set_device(self.engine.device_id)
        set_thread_local_mpi_comm(self._comm)
        try:
            while not self._stop.is_set():
                if not self._comm.Iprobe(source=0, tag=_ACTION_TAG):
                    self._stop.wait(0.01)
                    continue
                message = self._comm.recv(source=0, tag=_ACTION_TAG)
                with self._command_lock:
                    if self._stop.is_set():
                        break
                    reply = self._run_message(message)
                    self._pending_sends.append(self._comm.isend(reply, dest=0, tag=_ACK_TAG))
                    self._retire_sends()
        finally:
            set_thread_local_mpi_comm(None)

    def _retire_sends(self) -> None:
        """Retain pending MPI sends until the transport proves completion."""
        self._pending_sends = [request for request in self._pending_sends if not request.test()[0]]

    def _remaining(self, deadline: float) -> float:
        """Reject work outside the original local command budget.

        Args:
            deadline: Monotonic local deadline for this command.

        Returns:
            Positive remaining seconds.

        Raises:
            RuntimeError: If shutdown started.
            TimeoutError: If the command budget expired.
        """
        if self._stop.is_set():
            raise RuntimeError("Checkpoint control is shutting down.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Checkpoint rank operation timed out.")
        return remaining

    def group_step(
        self,
        step: str,
        context: dict,
        remaining_seconds: float,
        validation_token: str | None = None,
        source_session_id: str | None = None,
    ) -> list[dict]:
        """Coordinate one step with exact acknowledgements from every participant.

        Args:
            step: Internal CheckpointStep value.
            context: Validated logical operation envelope.
            remaining_seconds: Controller's remaining budget, never a template timestamp.
            validation_token: Private one-use generation token when starting validation.
            source_session_id: Captured source identity on the first restore command.

        Returns:
            Per-rank evidence, only after all required ranks have acknowledged.

        Raises:
            RuntimeError: If a rank fails or returns inconsistent evidence.
            TimeoutError: If any required acknowledgement is missing.
        """
        if self.engine.dist.rank != 0:
            raise RuntimeError("Only rank zero coordinates checkpoint commands.")
        if remaining_seconds <= 0:
            raise TimeoutError("Checkpoint operation budget expired before dispatch.")
        deadline = time.monotonic() + remaining_seconds
        if not self._command_lock.acquire(timeout=remaining_seconds):
            raise TimeoutError("Another checkpoint command is still running.")
        try:
            parsed = CheckpointContext.model_validate(context)
            action = CheckpointStep(step)
            message = {
                "step": action.value,
                "context": parsed.model_dump(mode="json"),
                "remaining_seconds": self._remaining(deadline),
                "validation_token": validation_token,
                "source_session_id": source_session_id,
            }
            # Commit the leader last: only rank zero accepts public requests.
            for rank in self.profile.ranks[1:]:
                self._pending_sends.append(self._comm.isend(message, dest=rank, tag=_ACTION_TAG))
            replies = []
            if action != CheckpointStep.ACTIVATION_COMMIT:
                replies.append(self._run_message(message))
            for rank in self.profile.ranks[1:]:
                while True:
                    self._remaining(deadline)
                    if self._comm.Iprobe(source=rank, tag=_ACK_TAG):
                        reply = self._comm.recv(source=rank, tag=_ACK_TAG)
                        if (
                            reply.get("operation_id") == parsed.operation_id
                            and reply.get("step") == action.value
                        ):
                            replies.append(reply)
                            break
                    time.sleep(0.01)
            expected = (
                self.profile.ranks[1:]
                if action == CheckpointStep.ACTIVATION_COMMIT
                else self.profile.ranks
            )
            self._validate_replies(replies, parsed, expected)
            if action == CheckpointStep.ACTIVATION_COMMIT:
                self._remaining(deadline)
                replies.append(self._run_message(message))
            self._validate_replies(replies, parsed, self.profile.ranks)
            self._retire_sends()
            return [reply["evidence"] for reply in replies]
        except Exception:
            self._failed = True
            self.engine.executor_request_queue.fail_sleep_wakeup_transition()
            raise
        finally:
            self._command_lock.release()

    def _validate_replies(
        self, replies: list[dict], context: CheckpointContext, ranks: list[int]
    ) -> None:
        """Validate participant identities before the leader commits admission.

        Args:
            replies: Native acknowledgements received for this step.
            context: Current operation identity.
            ranks: Exact required participants at this point.

        Raises:
            RuntimeError: If an acknowledgement is missing, stale or failed.
        """
        if any(reply.get("error") is not None for reply in replies):
            raise RuntimeError(f"Checkpoint participant failed: {replies}")
        evidence = [RankCheckpointEvidence.model_validate(reply["evidence"]) for reply in replies]
        if sorted(item.rank for item in evidence) != sorted(ranks) or any(
            item.session_id != context.session_id
            or item.profile_digest != context.profile_digest
            or item.cohort_digest != context.cohort_digest
            or not item.checks
            or not item.state_digest
            for item in evidence
        ):
            raise RuntimeError("Checkpoint received incomplete or stale participant evidence.")

    def _run_message(self, message: dict) -> dict:
        """Execute an identity-bound step once; cache its terminal acknowledgement.

        Args:
            message: Internal command from the authenticated rank-zero dispatcher.

        Returns:
            Identity-bound evidence or an error acknowledgement. An exact retry
            returns the recorded result without repeating native mutations.
        """
        step = CheckpointStep(message["step"])
        context = CheckpointContext.model_validate(message["context"])
        key = (context.operation_id, step.value)
        fingerprint = checkpoint_digest(
            {
                "step": step.value,
                "context": message["context"],
                "token": message.get("validation_token"),
                "source": message.get("source_session_id"),
            }
        )
        previous = self._completed.get(key)
        if previous is not None:
            if previous[0] != fingerprint:
                return {
                    "operation_id": context.operation_id,
                    "step": step.value,
                    "error": "Conflicting checkpoint rank command.",
                }
            return previous[1]
        reply = {"operation_id": context.operation_id, "step": step.value, "error": None}
        try:
            torch.cuda.set_device(self.engine.device_id)
            if len(self._completed) >= 256:
                raise RuntimeError("Checkpoint rank history is full.")
            deadline = time.monotonic() + min(
                message["remaining_seconds"], context.time_budget_seconds
            )
            self._remaining(deadline)
            if (
                context.profile_digest != self.profile.profile_digest
                or context.cohort_digest != self.profile.cohort_digest
            ):
                raise ValueError("Checkpoint command targets another profile or cohort.")
            if self._failed and step != CheckpointStep.ABORT:
                raise RuntimeError("Checkpoint rank failed; host cleanup is required.")
            reply["evidence"] = self._apply_step(
                step,
                context,
                deadline,
                message.get("validation_token"),
                message.get("source_session_id"),
            ).model_dump()
            self._remaining(deadline)
        except Exception as error:
            self._failed = True
            self.engine.executor_request_queue.fail_sleep_wakeup_transition()
            reply["error"] = str(error)
        self._completed[key] = (fingerprint, reply)
        return reply

    def _park(self, context: CheckpointContext, deadline: float) -> None:
        """Hold the existing drain context until validation or activation releases it.

        Args:
            context: Session that owns the native admission gate.
            deadline: Local deadline for the scheduler acknowledgement.

        Raises:
            TimeoutError: If scheduler ownership or drain cannot complete in time.
        """
        if self._parked is not None:
            return
        control_id = f"checkpoint:{context.operation_id}"
        parked = self.engine.control_action(
            drain=True,
            control_id=control_id,
            checkpoint_owner=self,
            timeout=self._remaining(deadline),
        )
        parked.__enter__()
        self._parked = parked
        self.engine.model_engine.wait_for_input_copy()
        torch.cuda.synchronize()

    def _unpark(self) -> None:
        """Release scheduler protection, without changing checkpoint admission."""
        if self._parked is not None:
            parked, self._parked = self._parked, None
            parked.__exit__(None, None, None)

    def _inventory(self) -> dict:
        """Inspect retained native owners and graph-visible resident storage.

        Returns:
            JSON-compatible owner identities and tensor addresses, layouts and
            warmup metrics. Tensor contents are not copied or hashed.

        Raises:
            RuntimeError: If the cohort, profile or protected boundary is invalid.
        """
        from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

        engine = self.engine
        self._check_profile()
        if (
            socket.gethostname() != self._host
            or str(torch.cuda.get_device_properties(engine.device_id).uuid) != self._device_uuid
        ):
            raise RuntimeError("Checkpoint destination changed its fixed host/device cohort.")
        if (
            self._parked is None
            or engine.active_requests
            or engine.waiting_queue
            or engine.previous_batch is not None
            or engine._has_pending_connector_transfers()
            or engine.executor_request_queue.get_request_queue_size() != 0
        ):
            raise RuntimeError("Checkpoint lacks a fully drained, protected scheduler boundary.")
        manager = engine.kv_cache_manager
        owners = []
        for request_id, cache in sorted(manager.kv_cache_map.items()):
            if not manager.impl.is_stats_excluded(request_id):
                raise RuntimeError("Checkpoint still has a user KV owner.")
            owners.append((request_id, cache.num_committed_tokens))
        runner = engine.model_engine.cuda_graph_runner
        if runner is None or not runner.graphs:
            raise RuntimeError("Checkpoint requires the final warmed CUDA graphs.")
        buffers = engine.model_engine._runner._buffers
        runner_storage = {}
        for field in fields(buffers):
            tensor = getattr(buffers, field.name)
            if tensor is not None:
                runner_storage[field.name] = (
                    tensor.data_ptr(),
                    tensor.untyped_storage().data_ptr(),
                    list(tensor.shape),
                    list(tensor.stride()),
                    str(tensor.dtype),
                )
        weights = {
            name: (
                tensor.data_ptr(),
                tensor.untyped_storage().data_ptr(),
                list(tensor.shape),
                list(tensor.stride()),
                str(tensor.dtype),
            )
            for name, tensor in list(engine.model_engine.model.named_parameters())
            + list(engine.model_engine.model.named_buffers())
        }
        pools = []
        for layer in sorted(manager.layer_offsets):
            tensor = manager.get_buffers(layer)
            if tensor is None:
                raise RuntimeError("Checkpoint cannot inspect a dense KV pool.")
            pools.append((layer, tensor.data_ptr(), list(tensor.shape), list(tensor.stride())))
        return {
            "owners": owners,
            "pools": pools,
            "weights": weights,
            "runner_storage": runner_storage,
            "graphs": sorted(repr(key) for key in runner.graphs),
            "gpu_bytes": manager.impl.get_num_bytes(0, CacheTier.GPU_MEM),
            "capacity_budget": engine.llm_args.kv_cache_config.max_gpu_total_bytes,
            "warmup_metrics": dict(engine.model_engine._metrics),
        }

    def _apply_step(
        self,
        step: CheckpointStep,
        context: CheckpointContext,
        deadline: float,
        token: str | None,
        source_session_id: str | None,
    ) -> RankCheckpointEvidence:
        """Apply one internal state transition under the rank's ownership checks.

        Args:
            step: Next native transition, or terminal abort.
            context: Operation identity bound to the fixed cohort.
            deadline: Local command deadline.
            token: Single-use private generation permit, only for validation start.
            source_session_id: Captured source identity, only for restore binding.

        Returns:
            Checks completed by this rank and its retained inventory hash.

        Raises:
            RuntimeError: If ordering, ownership or resident validity checks fail.
            TimeoutError: If native drain exceeds the local budget.
        """
        queue = self.engine.executor_request_queue
        predecessors = {
            CheckpointStep.PREPARE: None,
            CheckpointStep.RESTORE: CheckpointStep.PREPARE,
            CheckpointStep.VALIDATION_START: CheckpointStep.RESTORE,
            CheckpointStep.VALIDATION_END: CheckpointStep.VALIDATION_START,
            CheckpointStep.ACTIVATION_PREPARE: CheckpointStep.VALIDATION_END,
            CheckpointStep.ACTIVATION_COMMIT: CheckpointStep.ACTIVATION_PREPARE,
        }
        if step != CheckpointStep.ABORT and self._phase != predecessors[step]:
            raise RuntimeError("Checkpoint rank transition is out of order.")
        checks = []
        if step == CheckpointStep.PREPARE:
            if self.session_id is not None:
                raise RuntimeError("Checkpoint source is already claimed.")
            self._check_profile()
            self.session_id = context.session_id
            self._source_session_id = context.session_id
            queue.close_checkpoint_admission(context.session_id)
            self._park(context, deadline)
            self._baseline = self._inventory()
            checks = [
                "admission_closed",
                "scheduler_drained",
                "no_external_exposure",
                "resident_owners_recorded",
            ]
        elif step == CheckpointStep.RESTORE:
            if source_session_id != self._source_session_id or self._baseline is None:
                raise RuntimeError("Restore does not identify the protected source.")
            queue.bind_checkpoint_restore(source_session_id, context.session_id)
            self.session_id = context.session_id
            if self._inventory() != self._baseline:
                raise RuntimeError("Restored resident storage or internal owners changed.")
            checks = ["fresh_restore_session", "memory_present", "resident_state_matches"]
        else:
            if self.session_id != context.session_id:
                raise RuntimeError("Stale checkpoint rank session.")
            if step == CheckpointStep.VALIDATION_START:
                if self._baseline is None or self._parked is None or not token:
                    raise RuntimeError("Private validation requires restored protected state.")
                if self.engine.dist.rank == 0:
                    queue.set_checkpoint_validation(context.session_id, token)
                self._validation_replays = 0
                self._unpark()
                checks = ["private_validation_only", "public_admission_closed"]
            elif step == CheckpointStep.VALIDATION_END:
                queue.set_checkpoint_validation(context.session_id, None)
                self._park(context, deadline)
                if self._validation_replays == 0:
                    raise RuntimeError("Private validation did not replay a warmed CUDA graph.")
                if self._inventory() != self._baseline:
                    raise RuntimeError("Validation left stale owners or changed graph storage.")
                checks = [
                    "validation_retired",
                    "scheduler_drained",
                    "resident_state_matches",
                    "warmed_graph_replayed",
                ]
            elif step == CheckpointStep.ACTIVATION_PREPARE:
                if self._parked is None or self._inventory() != self._baseline:
                    raise RuntimeError("Activation lacks current protected runtime evidence.")
                checks = ["activation_prepared", "runtime_valid", "public_admission_closed"]
            elif step == CheckpointStep.ACTIVATION_COMMIT:
                if self._parked is None:
                    raise RuntimeError("Activation lacks a prepared rank.")
                self._unpark()
                queue.open_checkpoint_admission(context.session_id)
                checks = ["activation_committed"]
            elif step == CheckpointStep.ABORT:
                queue.set_checkpoint_validation(context.session_id, None)
                queue.fail_sleep_wakeup_transition()
                self._failed = True
                self._unpark()
                checks = ["progression_stopped", "public_admission_closed", "host_cleanup_required"]
            else:
                raise ValueError("Unsupported checkpoint rank action.")
        self._phase = step
        return RankCheckpointEvidence(
            rank=self.engine.dist.rank,
            session_id=context.session_id,
            profile_digest=self.profile.profile_digest,
            cohort_digest=self.profile.cohort_digest,
            checks=checks,
            state_digest=checkpoint_digest(self._baseline),
        )

    def fail_closed(self) -> None:
        """Revoke validation and public admission without releasing resources."""
        self._failed = True
        queue = self.engine.executor_request_queue
        queue.fail_sleep_wakeup_transition()
        if self.session_id is not None and queue.checkpoint_admission_closed():
            queue.set_checkpoint_validation(self.session_id, None)

    def close(self, timeout: float = 5.0) -> None:
        """Retire commands before engine shutdown can release their resources.

        Args:
            timeout: Maximum local wait for command and listener retirement, in seconds.

        Raises:
            TimeoutError: If control is still active. The caller must retain engine
                resources and arrange process cleanup outside the worker.
        """
        deadline = time.monotonic() + timeout
        self._stop.set()
        self.fail_closed()
        if not self._command_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise TimeoutError("Checkpoint command is still active; host cleanup is required.")
        try:
            self._unpark()
        finally:
            self._command_lock.release()
        if self._listener is not None:
            self._listener.join(timeout=max(0.0, deadline - time.monotonic()))
            if self._listener.is_alive():
                raise TimeoutError("Checkpoint listener is still active; host cleanup is required.")
        # MPI owns the duplicated communicator through process teardown. Do not
        # collectively free it while another rank can still be in failure cleanup.
