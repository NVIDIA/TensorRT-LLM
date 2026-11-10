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
"""Model-free GPU/NIXL lifecycle qualification in two owned CTX2/GEN2 MPI jobs.

The completion gate first observes genuine NIXL DONE, then withholds that evidence
from the production sender. These are backend/ownership and software-containment
tests, NOT outstanding-DMA, platform-fence, replacement, or initialized-executor tests.
The cancellation/cleanup seam uses real PyExecutor methods on a model-free shell.
The context-first profile uses BF16 K/V storage and two active TP ranks per job.
Generic V2 NVFP4 key/scale storage isolates lifecycle coverage from the unfinished
dense FP4 MLA manager/serving integration and its extra high-precision-tail roles.
FP4 transport settings stay unchanged; context-first explicitly selects its required
UCX backend. No replacement starts after fail-stop.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

if TYPE_CHECKING:
    import torch

    from tensorrt_llm._torch.disaggregation.native.retirement import QuiescenceFatalEvent
    from tensorrt_llm._torch.disaggregation.native.transfer import RxSession, TxSession
    from tensorrt_llm._torch.disaggregation.nixl._agent_cpp import BindingsNixlTransferStatus
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

_CASES = (
    "baseline",
    "cancel_before_publication",
    "cancel_late_aux",
    "cancel_late_kv",
    "timeout_late_aux",
    "source_fatal",
    "destination_fatal",
)
_CONTEXT_FIRST_CASES = ("baseline", "cancel_late_kv", "source_fatal", "destination_fatal")
_PHASE_TIMEOUT_S = 30.0
_TIMEOUT_S = 4 * _PHASE_TIMEOUT_S
_WORKER_TIMEOUT_S = 600.0


def _wait(
    predicate: Callable[[], bool],
    description: str,
    timeout: float = 30.0,
    *,
    deadline: float | None = None,
) -> None:
    """Wait for a bounded test precondition, never treating timeout as success.

    Args:
        predicate: Observation that must become true.
        description: Diagnostic identifying the missing precondition.
        timeout: Maximum wall-clock wait in seconds.
        deadline: Shared monotonic deadline, overriding the relative timeout.

    Raises:
        AssertionError: The precondition did not become true in time.
    """
    if deadline is None:
        deadline = time.monotonic() + timeout
    while True:
        ready = predicate()
        # Evaluating the predicate can consume the remaining deadline budget.
        assert time.monotonic() < deadline, f"Timed out: {description}"
        if ready:
            return
        time.sleep(0.01)


def _record(directory: Path, name: str, **fields: object) -> None:
    """Atomically publish one assertion-backed event to the test supervisor.

    Args:
        directory: Shared test-only rendezvous/artifact directory.
        name: Unique event name.
        fields: JSON-serializable evidence attached to the event.
    """
    target = directory / f"{name}.json"
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"pid": os.getpid(), "time": time.monotonic(), **fields}))
    temporary.replace(target)


def _report_startup_failure(
    jobs: dict[str, subprocess.Popen], directory: Path, context_first: bool = False
) -> None:
    """Include bounded worker diagnostics in captured CI output before cleanup.

    Args:
        jobs: Owned MPI launchers whose startup did not complete.
        directory: Per-rank markers and CTX/GEN logs for this test.
        context_first: Whether both TP ranks own active transfer sessions.
    """
    for role, job in jobs.items():
        expected = [
            *(f"{role}.{rank}.{phase}" for rank in range(2) for phase in ("spawned", "ready")),
            *(
                (f"{rank}/{role}.allocated" for rank in range(2))
                if context_first
                else (f"{role}.allocated", f"{role}.1.blocked")
            ),
        ]
        missing = [name for name in expected if not (directory / f"{name}.json").exists()]
        print(f"Startup failure: {role} launcher_exit={job.poll()}, missing={missing}")
        try:
            with (directory / f"{role}.log").open("rb") as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 8192))
                tail = log.read(8192).decode(errors="replace")
        except OSError as error:
            tail = f"Log unavailable: {error}"
        print(f"{role} log tail (at most 8192 bytes):\n{tail}")


def _stop_owned_jobs(jobs: dict[str, subprocess.Popen], directory: Path) -> None:
    """Stop owned launchers and ranks, including ranks with independent process groups.

    Args:
        jobs: Launchers created by this test, never unrelated cluster jobs.
        directory: Rank identities recorded before importing the GPU runtime.

    Raises:
        AssertionError: An owned rank survives bounded TERM/KILL cleanup.
    """
    import psutil

    owned = set()
    for job in jobs.values():
        if job.poll() is None:
            try:
                launcher = psutil.Process(job.pid)
                owned.update([launcher, *launcher.children(recursive=True)])
                os.killpg(job.pid, signal.SIGTERM)
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
    for identity in directory.glob("*.spawned.json"):
        recorded = json.loads(identity.read_text())
        try:
            process = psutil.Process(recorded["pid"])
            if process.create_time() == recorded["created"]:
                owned.add(process)
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(owned), timeout=3)
    for process in alive:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=3)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=3)
    for job in jobs.values():
        job.wait(timeout=15)
    survivors = []
    for process in alive:
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                survivors.append(process.pid)
        except psutil.NoSuchProcess:
            pass
    assert not survivors, f"owned MPI ranks survived cleanup: {survivors}"


def _wait_for_rank_exit(directory: Path, role: str, timeout: float = 15.0) -> None:
    """Observe both original ranks exiting, without sending cleanup signals.

    A zombie has exited but has not been reaped. Neither that state nor PID reuse
    means the original rank is still executing; neither proves GPU/RDMA fencing.

    Args:
        directory: Existing per-rank PID/create-time records and output directory.
        role: Endpoint world whose production containment must stop both ranks.
        timeout: Maximum passive observation time, independent of transfer grace.

    Raises:
        AssertionError: Any original rank remains live or cannot be observed.
    """
    import psutil

    identities = [
        json.loads((directory / f"{role}.{rank}.spawned.json").read_text()) for rank in range(2)
    ]
    started = time.monotonic()
    history = []
    previous = None
    while True:
        observations = []
        for rank, identity in enumerate(identities):
            observation = {"rank": rank, "pid": identity["pid"], "created": identity["created"]}
            try:
                process = psutil.Process(identity["pid"])
                observed_created = process.create_time()
                observation["observed_created"] = observed_created
                if observed_created != identity["created"]:
                    state = "reused"
                else:
                    status = process.status()
                    observation["status"] = status
                    state = "zombie" if status == psutil.STATUS_ZOMBIE else "live"
            except psutil.NoSuchProcess:
                state = "absent"
            except psutil.AccessDenied:
                state = "unobservable"
            observation["state"] = state
            observations.append(observation)
        elapsed = time.monotonic() - started
        if observations != previous:
            history.append({"elapsed": elapsed, "ranks": observations})
            previous = observations
        exited = all(item["state"] in ("absent", "zombie", "reused") for item in observations)
        if exited or elapsed >= timeout:
            _record(
                directory, f"{role}.exit_check", passed=exited, elapsed=elapsed, history=history
            )
            assert exited, f"Original {role} ranks did not exit before cleanup: {observations}"
            return
        time.sleep(0.01)


class _MaskedCompletion:
    """Retain a real native status while masking its already-observed DONE."""

    def __init__(self, status: BindingsNixlTransferStatus, directory: Path) -> None:
        """Initialize the software-only evidence gate.

        Args:
            status: Exact status returned by the real C++ NIXL binding.
            directory: Supervisor rendezvous directory.
        """
        self.status = status
        self.directory = directory
        self.observed_done = threading.Event()

    def wait(self, timeout_ms: int | None = None) -> bool:
        """Observe real DONE, then inject an ambiguous initial wait result.

        Args:
            timeout_ms: Unused production wait argument; native wait is bounded here.

        Returns:
            False after the supervisor allows the software ambiguity injection.
        """
        assert self.status.wait(30000), self.status.last_status_str()
        assert self.status.is_completed() is True
        self.observed_done.set()
        _record(self.directory, "native_done_masked", evidence="software_completion_mask")
        _wait(
            lambda: (self.directory / "report_ambiguous.json").exists(),
            "inject wait result",
            _TIMEOUT_S + 2 * _PHASE_TIMEOUT_S,
        )
        return False

    def is_completed(self) -> bool:
        """Expose fresh exact-handle DONE only after the late-settlement gate opens.

        Returns:
            True only when both the test gate and native status permit settlement.
        """
        return (self.directory / "allow_done.json").exists() and self.status.is_completed() is True

    def last_status_str(self) -> str:
        """Identify this test-only ambiguity, not a native NIXL failure.

        Returns:
            Diagnostic describing completion evidence masking.
        """
        return "test-only mask of already observed native DONE"


def _pool_state(manager: KVCacheManagerV2) -> dict[str, int]:
    """Read the real, locked slot accounting for this fixture's one GPU pool.

    Args:
        manager: Native-backed V2 manager, including its granularity-rounded capacity.

    Returns:
        Physical slot counts, not the nominal max_tokens budget.
    """
    from tensorrt_llm.runtime.kv_cache_manager_v2 import GPU_LEVEL

    statistics = manager.impl.get_storage_statistics(GPU_LEVEL)
    assert len(statistics) == 1
    pool = statistics[0]
    return {
        "total": pool.total,
        "free": pool.free,
        "available": pool.available,
        "evictable": pool.evictable,
    }


def _new_request(
    role: str, request_id: int, unique_id: int, endpoint: str, context_first: bool = False
) -> LlmRequest:
    """Construct a real model-free request with an immutable selected CTX instance.

    Args:
        role: CTX or GEN endpoint role.
        request_id: Local manager request identity.
        unique_id: Global no-retry disaggregated identity.
        endpoint: Selected CTX instance discovery endpoint.
        context_first: Select the ordinary BF16 context-first protocol.

    Returns:
        Request with one immutable selected CTX cohort.
    """
    import tensorrt_llm.bindings as bindings
    from tensorrt_llm import DisaggregatedParams, SamplingParams
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
    from tensorrt_llm.disaggregated_params import DisaggScheduleStyle

    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=list(range(127)),
        sampling_config=bindings.SamplingConfig(SamplingParams()._get_sampling_config()),
        is_streaming=False,
        llm_request_type=(
            LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY
            if role == "ctx"
            else LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
        ),
    )
    request.py_disaggregated_params = DisaggregatedParams(
        disagg_request_id=unique_id,
        schedule_style=(
            DisaggScheduleStyle.CONTEXT_FIRST
            if context_first
            else DisaggScheduleStyle.GENERATION_FIRST
        ),
        ctx_request_id=(unique_id if context_first else 1) if role == "gen" else None,
        ctx_dp_rank=0 if role == "gen" else None,
        ctx_info_endpoint=endpoint if role == "gen" else None,
    )
    return request


def _regions(transceiver: KvCacheTransceiverV2, request: LlmRequest) -> list[torch.Tensor]:
    """View actual allocated GPU regions using the production page extractor.

    Args:
        transceiver: Real transceiver owning the page table and extractor.
        request: Allocated request whose complete transfer extent is inspected.

    Returns:
        One-dimensional uint8 GPU views of both storage roles, excluding padding.
    """
    import torch

    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
    from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor

    chunk = transceiver._create_chunk(request)
    extractor = transceiver._transfer_worker._peer_registrar.self_extractor
    result = []
    seen = set()
    roles = set()
    for group, blocks in enumerate(chunk.block_ids_per_layer_groups):
        for pool in range(len(extractor.page_table.layer_groups[group].pool_views)):
            memory = extractor.extract(blocks, group, pool).memory
            view = extractor.page_table.layer_groups[group].pool_views[pool]
            assert len(memory.ptrs) > 0 and len(view.buffer_entries) > 0
            roles.update(view.pool_role)
            for pointer in memory.ptrs:
                for entry in view.buffer_entries:
                    key = (int(pointer) + int(entry["offset"]), int(entry["size"]))
                    assert key[0] > 0 and key[1] > 0, "allocated role extent must be nonempty"
                    if key not in seen:
                        seen.add(key)
                        result.append(
                            convert_to_torch_tensor(TensorWrapper(key[0], torch.uint8, [key[1]]))
                        )
    expected_roles = (
        {str(Role.KEY), str(Role.VALUE)}
        if transceiver._context_first_ownership
        else {str(Role.KEY), str(Role.KEY_BLOCK_SCALE)}
    )
    assert roles == expected_roles
    assert len(result) == 2, "one allocated block must expose both expected storage roles"
    return result


def _worker(args: argparse.Namespace) -> None:
    """Run one rank of an owned real MPI endpoint without a model or executor.

    Args:
        args: Test supervisor's role, case, GPU offset and artifact directory.
    """
    import psutil

    _record(
        Path(args.directory),
        f"{args.role}.{os.environ['OMPI_COMM_WORLD_RANK']}.spawned",
        created=psutil.Process().create_time(),
    )

    import torch
    from mpi4py import MPI

    from tensorrt_llm import Mapping
    from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
    from tensorrt_llm._torch.disaggregation.nixl._agent_cpp import BindingsNixlTransferAgent
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
    from tensorrt_llm._torch.distributed.communicator import MPIDist
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
    from tensorrt_llm._torch.pyexecutor.resource_manager import (
        CacheTypeCpp,
        ResourceManager,
        ResourceManagerType,
    )
    from tensorrt_llm.bindings import DataType, LlmRequestState
    from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig, KvCacheConfig

    world = MPI.COMM_WORLD
    assert MPI.Is_initialized() and MPI.Query_thread() == MPI.THREAD_MULTIPLE
    assert world.size == 2
    rank, role, directory, case = world.rank, args.role, Path(args.directory), args.case
    root_directory = directory
    context_first = args.context_first
    torch.cuda.set_device(args.gpu_offset + rank)
    mapping = Mapping(
        world_size=2, rank=rank, tp_size=2, pp_size=1, enable_attention_dp=not context_first
    )
    manager = KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=128,
            dtype="auto" if context_first else "nvfp4",
            enable_block_reuse=False,
            host_cache_size=0,
            max_util_for_resume=1.0,
        ),
        CacheTypeCpp.SELF if context_first else CacheTypeCpp.SELFKONLY,
        num_layers=1,
        num_kv_heads=4 if context_first else 1,
        head_dim=128 if context_first else 576,
        tokens_per_block=128,
        max_seq_len=256,
        max_batch_size=2,
        mapping=mapping,
        dtype=DataType.BF16 if context_first else DataType.NVFP4,
        max_num_tokens=128,
        is_disagg=True,
    )
    fatal_role = "ctx" if case == "source_fatal" else "gen"
    fatal_case = case.endswith("fatal")
    # Asymmetric finite deadlines are fault isolation, not matching-config qualification.
    timeout = _TIMEOUT_S if not fatal_case or role == fatal_role else 3 * _TIMEOUT_S
    transceiver = KvCacheTransceiverV2(
        mapping,
        MPIDist(mapping),
        manager,
        CacheTransceiverConfig(
            backend="NIXL",
            transceiver_runtime="PYTHON",
            kv_cache_bounce_size_mb=0,
            enable_pipelined_transfer=False,
            kv_transfer_timeout_ms=int(timeout * 1000),
        ),
    )
    worker = transceiver._transfer_worker
    assert transceiver._context_first_ownership == context_first
    assert transceiver._fp4_mla_bridge_enabled == (not context_first)
    assert isinstance(worker._agent, BindingsNixlTransferAgent)
    assert not worker._agent.bounce_enabled
    assert {registration.type for registration in worker._registered_mem} == {"VRAM", "DRAM"}
    worker._agent.deregister_memory = Mock(wraps=worker._agent.deregister_memory)
    manager.free_resources = Mock(wraps=manager.free_resources)
    worker._aux_buffer.free_slot = Mock(wraps=worker._aux_buffer.free_slot)
    fatal_check: list[Callable[[QuiescenceFatalEvent], None]] = []
    real_fatal = worker._config.quiescence_fatal_callback

    def observe_fatal(event: QuiescenceFatalEvent) -> None:
        """Inspect pre-abort ownership without replacing real containment.

        Args:
            event: Sticky production watchdog expiry.
        """
        try:
            assert fatal_check, "fatal expiry occurred before transfer preconditions"
            fatal_check[0](event)
        finally:
            real_fatal(event)

    worker._config.quiescence_fatal_callback = observe_fatal
    _record(
        directory, f"{role}.{rank}.ready", gpu=torch.cuda.get_device_name(), thread_multiple=True
    )
    if role == "ctx" and rank == 0:
        _record(directory, "endpoint", endpoint=transceiver._context_info_endpoint)
    _wait(
        lambda: (directory / "endpoint.json").exists(),
        "CTX endpoint",
        deadline=args.startup_deadline,
    )
    endpoint = json.loads((directory / "endpoint.json").read_text())["endpoint"]
    world.Barrier()
    if rank == 1 and not context_first:
        _record(directory, f"{role}.1.blocked", state="MPI_Barrier")
        world.Barrier()
        transceiver.shutdown()
        manager.shutdown()
        _record(directory, f"{role}.1.clean")
        return

    if context_first:
        # Both ranks perform real transfers and consensus. Keep per-rank evidence
        # separate; a successful rank must not overwrite a failing rank's record.
        directory = root_directory / str(rank)
    request = _new_request(role, 1 if role == "ctx" else 2, args.unique_id, endpoint, context_first)
    if role == "ctx":
        assert manager.prepare_context(request)
        assert manager.resize_context(request, request.prompt_len)
        cache = manager.kv_cache_map[request.py_request_id]
        assert cache.resize(cache.capacity, request.prompt_len)
    else:
        assert manager.prepare_disagg_gen_init(request)
    regions = _regions(transceiver, request)
    addresses = tuple((region.data_ptr(), region.numel()) for region in regions)
    source_byte = 123 + rank if context_first else 123
    for region in regions:
        region.fill_(source_byte if role == "ctx" else 17)
    torch.cuda.synchronize()
    # Native quota is rounded to GPU granularity, not one physical block. Pin its
    # spare slots before either request clock starts, so only the tested block can
    # become reusable later. A native cache avoids the wrapper's two-request index limit.
    initial_pool = _pool_state(manager)
    assert initial_pool["evictable"] == 0 and initial_pool["total"] - initial_pool["free"] == 1
    filler = manager.impl.create_kv_cache()
    filler.stop_committing()
    assert filler.resume(manager._stream.cuda_stream)
    assert filler.resize(initial_pool["free"] * manager.tokens_per_block)
    full_pool = _pool_state(manager)
    assert full_pool["total"] == initial_pool["total"]
    assert full_pool["free"] == full_pool["available"] == full_pool["evictable"] == 0
    # No model, scheduler loop, response transport, or async-send manager is initialized.
    executor = object.__new__(PyExecutor)
    executor.sampler = Mock(spec=["release_request"])
    executor.model_engine = None
    executor.draft_model_engine = None
    executor.kv_cache_transceiver = transceiver
    executor.kv_cache_manager = manager
    executor.resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: manager})
    executor.dist = transceiver._dist
    executor.enable_attention_dp = not context_first
    executor._is_kv_manager_v2 = True
    executor._prefetched_request_ids = {request.py_request_id}
    executor.gather_all_responses = False
    executor.result_wait_queues = {request.py_request_id: None} if rank == 0 else {}
    assert executor.disagg is not None
    _record(
        directory,
        f"{role}.allocated",
        addresses=addresses,
        pool=full_pool,
        filler_blocks=initial_pool["free"],
    )
    _wait(
        lambda: (directory / "start.json").exists(),
        "both endpoint worlds initialized",
        deadline=args.startup_deadline,
    )

    submissions: list[object] = []
    masked: list[_MaskedCompletion] = []
    submit = worker._agent.submit_transfer_requests

    def submit_observed(transfer_request: object) -> object:
        """Submit through the real binding and mask only the selected native handle.

        Args:
            transfer_request: Unmodified native transfer descriptors.

        Returns:
            Real status, or a test-only completion-evidence proxy retaining it.
        """
        status = submit(transfer_request)
        submissions.append(status)
        if case not in ("baseline", "cancel_before_publication") and len(submissions) == (
            1 if context_first or case == "cancel_late_kv" else 2
        ):
            gate = _MaskedCompletion(status, directory)
            masked.append(gate)
            return gate
        return status

    worker._agent.submit_transfer_requests = submit_observed
    if role == "gen":
        create_rx = worker.create_rx_session

        def observed_rx(incoming: LlmRequest) -> RxSession:
            """Capture actual reports before any publication can race their arrival.

            Args:
                incoming: Request passed unchanged to the production factory.

            Returns:
                Real receive session with forwarding evidence spies.
            """
            receive_session = create_rx(incoming)
            receive_session.process_aux_agent_result = Mock(
                wraps=receive_session.process_aux_agent_result
            )
            receive_session.process_kv_agent_result = Mock(
                wraps=receive_session.process_kv_agent_result
            )
            return receive_session

        worker.create_rx_session = observed_rx
    if case == "cancel_before_publication":
        if role == "gen":
            session = worker.create_rx_session(request)
            session.cancel()
            session.receive(transceiver._create_chunk(request))
            assert session.status == SessionStatus.CANCELLED
            assert not session._publication_may_have_escaped and not session._kv_tasks
            assert session.resources_drained() and session.close()
            assert session.close()
            assert worker._aux_buffer.free_slot.call_count == 1
            _record(directory, "cancelled_before_publication")
        _wait(lambda: (directory / "cancelled_before_publication.json").exists(), "pre-cancel")
        assert not submissions
        assert all(bool(torch.all(region == (123 if role == "ctx" else 17))) for region in regions)
        manager.free_resources(request)
    else:
        if role == "gen":
            if context_first:
                _wait(
                    lambda: (directory / "ctx.published.json").exists(),
                    "context-first publication precedes destination fetch",
                )
            transceiver.request_and_receive_async(request)
            session = transceiver._recv_sessions[args.unique_id]
        elif not context_first:
            transceiver.prepare_context_requests([request])

            def receiver_ready() -> bool:
                """Progress the real gen-first admission handshake.

                Returns:
                    Whether the immutable selected receiver cohort is ready.
                """
                transceiver.prepare_context_requests([])
                return request.state == LlmRequestState.CONTEXT_INIT

            _wait(receiver_ready, "generation-first peer request")
            transceiver.respond_and_send_async(request)
            session = transceiver._send_sessions[args.unique_id]
        else:
            transceiver.respond_and_send_async(request)
            session = transceiver._send_sessions[args.unique_id]
            assert request.context_phase_params.req_id == args.unique_id
            _record(directory, "ctx.published")
        assert session._enforce_physical_ownership and session._retirement is not None
        assert session._need_aux == (not context_first)
        _exercise_live_session(
            args,
            transceiver,
            request,
            session,
            regions,
            masked,
            submissions,
            fatal_check,
            executor,
            directory,
        )

    _record(directory, f"{role}.finished", submissions=len(submissions))
    _wait(
        lambda: (directory / f"{'gen' if role == 'ctx' else 'ctx'}.finished.json").exists(),
        "peer finished",
    )
    world.Barrier()
    transceiver.shutdown()
    filler.close()
    manager.shutdown()
    _record(root_directory, f"{role}.{rank}.clean")


def _exercise_live_session(
    args: argparse.Namespace,
    transceiver: KvCacheTransceiverV2,
    request: LlmRequest,
    session: RxSession | TxSession,
    regions: list[torch.Tensor],
    masked: list[_MaskedCompletion],
    submissions: list[object],
    fatal_check: list[Callable[[QuiescenceFatalEvent], None]],
    executor: PyExecutor,
    directory: Path,
) -> None:
    """Assert real allocation retention, stable outcome, and exact late settlement.

    Args:
        args: Supervisor configuration.
        transceiver: Real native-backed transceiver.
        request: Allocated request with an admitted transfer.
        session: Exact production session retained across status polling.
        regions: Allocated GPU byte views, not clones.
        masked: Sender's test-only completion gate, if applicable.
        submissions: Native status roots proving real backend submissions.
        fatal_check: Receives the pre-abort observer before any test wait.
        executor: Model-free shell prepared before starting either request clock.
        directory: Rendezvous directory unique to this active TP rank.
    """
    import torch

    from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus

    role, case = args.role, args.case
    context_first = args.context_first
    peer_rank = transceiver._mapping.rank if context_first else 0
    kv_masked = context_first or case == "cancel_late_kv"
    expected_byte = 123 + peer_rank
    worker, manager = transceiver._transfer_worker, transceiver._kv_cache_manager
    rid, slot = args.unique_id, session.aux_slot
    addresses = tuple((region.data_ptr(), region.numel()) for region in regions)
    cache = manager.kv_cache_map[request.py_request_id]
    block_ids = tuple(
        tuple(blocks.tolist())
        for blocks in transceiver._create_chunk(request).block_ids_per_layer_groups
    )
    poll = (
        transceiver.check_context_transfer_status
        if role == "ctx"
        else transceiver.check_gen_transfer_status
    )
    sessions = transceiver._send_sessions if role == "ctx" else transceiver._recv_sessions
    requests = transceiver._send_reqs if role == "ctx" else transceiver._recv_reqs

    def retained() -> None:
        """Assert real KV/AUX/registration roots remain with no ordinary release."""
        assert sessions[rid] is session and requests[rid] is request
        assert manager.kv_cache_map[request.py_request_id] is cache
        assert block_ids == tuple(
            tuple(blocks.tolist())
            for blocks in transceiver._create_chunk(request).block_ids_per_layer_groups
        )
        assert slot in worker._aux_buffer._occupied_slots
        assert worker._aux_buffer.free_slot.call_count == 0
        assert not any(call.args[0] is request for call in manager.free_resources.call_args_list)
        assert worker._agent.deregister_memory.call_count == 0
        assert worker._registered_mem and not session.resources_drained()
        if role == "ctx" and masked:
            task = session.kv_tasks[0] if kv_masked else session.aux_task
            operation = task._physical_operations[peer_rank]
            assert operation.status is masked[0] and operation.request is not None
            assert masked[0].status is submissions[-1]

    def check_fatal(event: QuiescenceFatalEvent) -> None:
        """Require the target's sticky fail-closed decision before the real abort.

        Args:
            event: Watchdog decision already committed independently of polling.
        """
        retained()
        assert case.endswith("fatal")
        assert role == ("ctx" if case == "source_fatal" else "gen")
        assert session.status == SessionStatus.ERROR
        assert worker._retirement_watchdog.fatal is event
        assert event.direction == ("send" if role == "ctx" else "receive")
        assert event.reason != "transfer timeout", event
        assert event.deadline == event.started_at + _TIMEOUT_S
        assert _TIMEOUT_S <= event.expired_at - event.started_at <= _TIMEOUT_S + 2.0
        with pytest.raises(RuntimeError, match="admission is closed"):
            worker.create_tx_session(_new_request("ctx", 5, rid + 2, "", context_first))
        with pytest.raises(RuntimeError, match="admission is closed"):
            worker.create_rx_session(_new_request("gen", 6, rid + 4, "", context_first))
        assert not session.close()
        _record(
            directory,
            f"{role}.fatal",
            direction=event.direction,
            elapsed=event.expired_at - event.started_at,
        )

    fatal_check.append(check_fatal)

    if case != "baseline":
        _wait(
            lambda: (directory / "native_done_masked.json").exists(),
            "real native submission and DONE",
        )
        if role == "ctx":
            assert len(submissions) == (1 if kv_masked else 2)
            assert masked[0].observed_done.is_set()
        retained()
        assert all(bool(torch.all(region == expected_byte)) for region in regions)
        probe = _new_request(role, 3, rid + 1, "")
        assert manager.prepare_context_cache(probe) is not None
        assert not manager.kv_cache_map[3].resize(128, 127), "live pages reused under pressure"
        retained()
        manager.free_resources(probe)
        assert _pool_state(manager)["available"] == 0
        retirement = session._retirement
        assert retirement._request_deadline is not None
        assert retirement._drain_started is None, retirement._reason
        _record(
            directory,
            f"{role}.prepared",
            request_deadline=retirement._request_deadline,
            timeout=retirement.timeout_s,
        )
        _wait(
            lambda: (directory / "transition.json").exists(), "both endpoints ready for transition"
        )
        if case.startswith("cancel_"):
            assert retirement._request_deadline - time.monotonic() > _PHASE_TIMEOUT_S
            # The other endpoint may already have sent cancellation after the gate.
            # Local cancellation must preserve that first drain, not restart grace.
            previous_drain = retirement._drain_started
            assert not executor._try_cancel_request(request)
            assert session.status == SessionStatus.CANCELLED, (session.status, retirement._reason)
            if previous_drain is not None:
                assert retirement._drain_started == previous_drain
        elif case == "timeout_late_aux":
            # Do not poll the session before the independent request clock commits timeout.
            _wait(
                lambda: session._retirement._drain_started is not None,
                "independent request timeout",
                _TIMEOUT_S + _PHASE_TIMEOUT_S,
            )
            assert session.status == SessionStatus.ERROR
            assert retirement._reason == "transfer timeout"
            assert retirement._drain_started == retirement._request_deadline
            _record(directory, f"{role}.timed_out")
            _wait(
                lambda: all(
                    (directory / f"{peer}.timed_out.json").exists() for peer in ("ctx", "gen")
                ),
                "both logical timeouts precede cancellation notification",
                2 * _PHASE_TIMEOUT_S,
            )
        retained()
        drain_started = session._retirement._drain_started
        for _ in range(3):
            if case.startswith("cancel_"):
                assert not executor._try_cancel_request(request)
                assert session._retirement._drain_started == drain_started
            status = poll(0)
            assert not any(status)
            retained()
        _record(
            directory,
            f"{role}.retained",
            logical=session.status.name,
            addresses=addresses,
            native_submissions=len(submissions),
            pool=_pool_state(manager),
        )
        _wait(
            lambda: (directory / "report_ambiguous.json").exists(),
            "supervisor commits software ambiguity injection",
        )
        if role == "ctx":
            task = session.kv_tasks[0] if kv_masked else session.aux_task
            _wait(
                lambda: task._physical_operations[peer_rank].state.name == "IN_DOUBT",
                "retained native handle enters IN_DOUBT",
            )
        else:
            owner = (
                session._kv_tasks[0]._physical_owner if kv_masked else session._aux_physical_owner
            )
            _wait(
                lambda: peer_rank in owner._in_doubt_writers,
                "actual IN_DOUBT report reaches receiver",
            )
        retained()
        if case.endswith("fatal"):
            assert retirement._drain_started is not None
            assert retirement._reason != "transfer timeout", retirement._reason
        _record(directory, f"{role}.in_doubt", logical=session.status.name)

        if case.endswith("fatal"):
            fatal_role = "ctx" if case == "source_fatal" else "gen"
            if role != fatal_role:
                _wait(
                    lambda: any(
                        (path / f"{fatal_role}.fatal.json").exists()
                        for path in _evidence_directories(Path(args.directory), context_first)
                    ),
                    "opposite endpoint fatal",
                    _TIMEOUT_S + _PHASE_TIMEOUT_S,
                )
                retained()
                _record(directory, f"{role}.survived", resources_retained=True)
            _wait(
                lambda: False,
                "supervisor terminates surviving owned job",
                _TIMEOUT_S + 2 * _PHASE_TIMEOUT_S,
            )
            return

        _wait(
            lambda: (directory / "allow_done.json").exists(), "release exact retained native status"
        )
        assert retirement._drain_started is not None
        assert time.monotonic() < retirement._drain_started + retirement.timeout_s
        assert worker._retirement_watchdog.fatal is None

    terminal_before = session.status if case != "baseline" else SessionStatus.TRANSFERRED
    results = []

    def retired() -> bool:
        """Poll the real transceiver until the allocator release boundary opens.

        Returns:
            Whether this exact session has retired.
        """
        results.append(poll(0))
        return rid not in sessions

    _wait(retired, "native completion and session retirement")
    actual_status = session.status
    drained = session.resources_drained()
    _record(
        directory,
        f"{role}.retired",
        expected=terminal_before.name,
        actual=actual_status.name,
        drained=drained,
        drain_reason=session._retirement._reason,
        poll_results=[
            {
                "completed": result[0],
                "failed": result[1],
                "cancelled_count": len(result[2]) if len(result) == 3 else 0,
            }
            for result in results
        ],
    )
    assert drained, "retired session still owns physical accesses"
    assert actual_status == terminal_before, (
        actual_status,
        terminal_before,
        session._retirement._reason,
    )
    assert (
        worker._aux_buffer.free_slot.call_count == 1
        and slot not in worker._aux_buffer._occupied_slots
    )
    if case == "baseline":
        assert any(result[0] == [rid] for result in results)
        if role == "ctx":
            assert len(submissions) == (1 if context_first else 2)
            assert all(status.is_completed() for status in submissions)
        elif not context_first:
            assert request.context_phase_params.first_gen_tokens
    else:
        assert not any(result[0] for result in results), "late DONE revived logical success"
    if context_first or case != "cancel_late_kv":
        assert all(bool(torch.all(region == expected_byte)) for region in regions)

    assert executor._try_cancel_request(request)
    executor._do_terminate_request(request)
    assert request.py_request_id not in executor._prefetched_request_ids
    assert request.py_request_id not in executor.result_wait_queues
    retired_pool = _pool_state(manager)
    assert retired_pool["free"] == retired_pool["available"] == 1, retired_pool
    replacement = _new_request(role, 4, rid + 3, "")
    assert manager.prepare_context_cache(replacement) is not None
    assert manager.kv_cache_map[4].resize(128, 127)
    assert _pool_state(manager)["available"] == 0
    replacement_regions = _regions(transceiver, replacement)
    assert set(addresses) == {(region.data_ptr(), region.numel()) for region in replacement_regions}
    for region in replacement_regions:
        region.fill_(77)
    torch.cuda.synchronize()
    aux = worker._aux_buffer
    replacement_slots = [aux.alloc_slot().id for _ in range(len(aux._free_slots))]
    assert slot in replacement_slots
    aux._first_tokens_buffer[slot].fill_(77)
    report = (
        (session.process_kv_agent_result if context_first else session.process_aux_agent_result)
        if role == "gen"
        else None
    )
    duplicate_result = report.call_args if report is not None else None
    for _ in range(3):
        assert session.close()
        assert not any(poll(0))
        if role == "gen":
            assert duplicate_result is not None
            report(*duplicate_result.args, **duplicate_result.kwargs)
        elif masked:
            assert masked[0].is_completed()
            task = session.kv_tasks[0] if kv_masked else session.aux_task
            assert not task.poll_in_doubt_physical_operation(peer_rank)
    assert worker._aux_buffer.free_slot.call_count == 1
    assert slot in aux._occupied_slots and bool(torch.all(aux._first_tokens_buffer[slot] == 77))
    assert sum(call.args[0] is request for call in manager.free_resources.call_args_list) == 1
    assert all(bool(torch.all(region == 77)) for region in replacement_regions)
    for replacement_slot in replacement_slots:
        aux.free_slot(replacement_slot)
    manager.free_resources(replacement)
    _record(directory, f"{role}.reused_after_done", stable_outcome=session.status.name, sentinel=77)


def _evidence_directories(directory: Path, context_first: bool) -> list[Path]:
    """Locate separate evidence for every active rank without merging observations.

    Args:
        directory: Supervisor rendezvous directory.
        context_first: Both ranks are active in the non-ADP context-first profile.

    Returns:
        One directory per active rank in each endpoint job.
    """
    return [directory / str(rank) for rank in range(2)] if context_first else [directory]


def _run_transfer_lifecycle_gpu(case: str, tmp_path: Path, context_first: bool) -> None:
    """Run the shared four-GPU supervisor with real, separate CTX/GEN MPI jobs.

    Args:
        case: Distinct baseline, cancellation, late-settlement or containment contract.
        tmp_path: Persistent-per-test logs and rendezvous artifacts.
        context_first: BF16 TP2 profile rather than the original FP4 ADP profile.
    """
    import torch

    assert sys.platform == "linux" and shutil.which("mpirun"), "requires Linux and mpirun"
    assert torch.cuda.device_count() >= 4, (
        "requires four visible GPUs; missing hardware is not a pass"
    )
    prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_", "I_MPI_", "HYDRA_", "MPI_")
    environment = {key: value for key, value in os.environ.items() if not key.startswith(prefixes)}
    environment.update(
        TLLM_DISABLE_MPI="0",
        MPI4PY_RC_THREAD_LEVEL="multiple",
        TRTLLM_ENABLE_FP4_MLA_KV_OWNERSHIP_BRIDGE="0" if context_first else "1",
        TRTLLM_DISAGG_NO_RETRY="1",
        TRTLLM_USE_PY_NIXL_KVCACHE="0",
        TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP="0",
        TRTLLM_DISAGG_LAYERWISE="0",
    )
    if context_first:
        environment["TRTLLM_ENABLE_KV_TRANSFER_LIFECYCLE_ENFORCEMENT"] = "1"
        environment["TRTLLM_NIXL_KVCACHE_BACKEND"] = "UCX"
    else:
        environment.pop("TRTLLM_ENABLE_KV_TRANSFER_LIFECYCLE_ENFORCEMENT", None)
    evidence_dirs = _evidence_directories(tmp_path, context_first)
    for directory in evidence_dirs:
        directory.mkdir(exist_ok=True)

    def all_recorded(phase: str) -> bool:
        """Require evidence from every active rank on both endpoint jobs."""
        return all(
            (directory / f"{role}.{phase}.json").exists()
            for directory in evidence_dirs
            for role in ("ctx", "gen")
        )

    def release_gate(name: str) -> None:
        """Release the same test gate for every active rank."""
        for directory in evidence_dirs:
            _record(directory, name)

    unique_id = uuid.uuid4().int & ((1 << 62) - 1)
    jobs = {}
    # All ranks are local, so their monotonic clocks share this startup budget.
    startup_deadline = time.monotonic() + _WORKER_TIMEOUT_S
    startup_complete = False
    with ExitStack() as stack:
        try:
            for role, offset in (("ctx", 0), ("gen", 2)):
                log = stack.enter_context((tmp_path / f"{role}.log").open("w"))
                jobs[role] = subprocess.Popen(
                    [
                        "mpirun",
                        "--allow-run-as-root",
                        "--oversubscribe",
                        "-n",
                        "2",
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        *(["--context-first"] if context_first else []),
                        "--case",
                        case,
                        "--role",
                        role,
                        "--gpu-offset",
                        str(offset),
                        "--directory",
                        str(tmp_path),
                        "--unique-id",
                        str(unique_id),
                        "--startup-deadline",
                        str(startup_deadline),
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            _wait(
                lambda: all(
                    (tmp_path / f"{role}.{rank}.ready.json").exists()
                    for role in jobs
                    for rank in range(2)
                ),
                "four native MPI ranks",
                deadline=startup_deadline,
            )
            _wait(
                lambda: (
                    all_recorded("allocated")
                    and (
                        context_first
                        or all((tmp_path / f"{role}.1.blocked.json").exists() for role in jobs)
                    )
                ),
                "real V2 allocations on every active rank",
                deadline=startup_deadline,
            )
            release_gate("start")
            startup_complete = True
            if case not in ("baseline", "cancel_before_publication"):
                _wait(
                    lambda: all_recorded("prepared"),
                    "native DONE and measured allocator pressure on both endpoints",
                )
                clocks = [
                    json.loads((directory / f"{role}.prepared.json").read_text())
                    for directory in evidence_dirs
                    for role in jobs
                ]
                # Preserve enough grace for both independent timeouts plus bounded rendezvous.
                assert (
                    min(clock["request_deadline"] for clock in clocks) - time.monotonic()
                    > 2 * _PHASE_TIMEOUT_S
                ), clocks
                if case == "timeout_late_aux":
                    assert max(
                        clock["request_deadline"] for clock in clocks
                    ) + 2 * _PHASE_TIMEOUT_S < min(
                        clock["request_deadline"] + clock["timeout"] for clock in clocks
                    ), clocks
                release_gate("transition")
                _wait(
                    lambda: all_recorded("retained"),
                    "both sides retain real resources",
                    _TIMEOUT_S + 2 * _PHASE_TIMEOUT_S,
                )
                release_gate("report_ambiguous")
                _wait(
                    lambda: all_recorded("in_doubt"),
                    "both endpoints retain explicit IN_DOUBT ownership",
                )
                if not case.endswith("fatal"):
                    release_gate("allow_done")
            if case.endswith("fatal"):
                fatal_role = "ctx" if case == "source_fatal" else "gen"
                survivor = "gen" if fatal_role == "ctx" else "ctx"
                _wait(
                    lambda: any(
                        (directory / f"{fatal_role}.fatal.json").exists()
                        for directory in evidence_dirs
                    ),
                    "asserted fatal decision",
                    _TIMEOUT_S + _PHASE_TIMEOUT_S,
                )
                jobs[fatal_role].wait(timeout=15)
                assert jobs[fatal_role].returncode != 0
                _wait(
                    lambda: all(
                        (directory / f"{survivor}.survived.json").exists()
                        for directory in evidence_dirs
                    ),
                    "independent peer survives",
                )
                assert jobs[survivor].poll() is None
                _wait_for_rank_exit(tmp_path, fatal_role)
            else:
                for role, job in jobs.items():
                    job.wait(timeout=60)
                    assert job.returncode == 0, (tmp_path / f"{role}.log").read_text()
                    assert all(
                        (tmp_path / f"{role}.{rank}.clean.json").exists() for rank in range(2)
                    )
        finally:
            try:
                if not startup_complete:
                    _report_startup_failure(jobs, tmp_path, context_first)
            finally:
                _stop_owned_jobs(jobs, tmp_path)


@pytest.mark.timeout(_WORKER_TIMEOUT_S + 4 * _TIMEOUT_S, method="signal")
@pytest.mark.parametrize("case", _CASES)
def test_transfer_lifecycle_gpu(case: str, tmp_path: Path) -> None:
    """Qualify the original generation-first FP4 ADP software lifecycle.

    Args:
        case: Baseline, cancellation, late-settlement or containment contract.
        tmp_path: Persistent-per-test logs and rendezvous artifacts.
    """
    _run_transfer_lifecycle_gpu(case, tmp_path, context_first=False)


@pytest.mark.gpu4
@pytest.mark.timeout(_WORKER_TIMEOUT_S + 4 * _TIMEOUT_S, method="signal")
@pytest.mark.parametrize("case", _CONTEXT_FIRST_CASES)
def test_context_first_bf16_lifecycle_gpu(case: str, tmp_path: Path) -> None:
    """Qualify K/V retention and late/fatal settlement with both TP2 ranks active.

    Native DONE is masked only after the real write completes. Fatal cases prove
    software fail-stop of both ranks, not outstanding-DMA fencing or replacement.

    Args:
        case: Baseline, late KV completion, or source/destination fatal contract.
        tmp_path: Per-rank native identities, pool evidence, and pre-cleanup exit checks.
    """
    _run_transfer_lifecycle_gpu(case, tmp_path, context_first=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--context-first", action="store_true")
    parser.add_argument("--case", choices=_CASES, required=True)
    parser.add_argument("--role", choices=("ctx", "gen"), required=True)
    parser.add_argument("--gpu-offset", type=int, required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--unique-id", type=int, required=True)
    parser.add_argument("--startup-deadline", type=float, required=True)
    _worker(parser.parse_args())
