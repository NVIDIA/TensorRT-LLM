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
"""Test-owned observers and bounded gates around the real serving implementation.

The sender gate delays observation after a real NIXL submission, not device DMA.
It proves retained ownership and progress, not fencing or safe replacement.
"""

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch

import tensorrt_llm
from tensorrt_llm._torch.disaggregation.native.transfer import Sender
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.worker import GenerationExecutorWorker

PROBE_DIR_ENV = "TLLM_CONTEXT_CANCELLATION_PROBE_DIR"
PROBE_ROLE_ENV = "TLLM_CONTEXT_CANCELLATION_PROBE_ROLE"
GATE_TIMEOUT = 45.0
_EVENT_LOCK = threading.Lock()


def _directory() -> Path:
    """Return this test's process-shared artifact directory."""
    return Path(os.environ[PROBE_DIR_ENV])


def emit(event: str, pair_id: int | None = None, **fields: Any) -> None:
    """Persist an observation from the real owner without editing its state.

    Args:
        event: Observed production boundary.
        pair_id: Original disaggregated pairing identity, when available.
        **fields: Snapshot fields obtained at that boundary.
    """
    role = os.environ[PROBE_ROLE_ENV]
    rank = mpi_rank() if role != "coordinator" else 0
    record = dict(
        event=event,
        pair_id=pair_id,
        role=role,
        rank=rank,
        pid=os.getpid(),
        time=time.monotonic(),
        **fields,
    )
    with _EVENT_LOCK:
        with (_directory() / f"{role}-{rank}-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")


def _pair(request: Any) -> int | None:
    """Read the request's pairing identity without changing its lifetime."""
    params = request.py_disaggregated_params
    return None if params is None else params.disagg_request_id


def _wait_gate(name: str, pair_id: int) -> None:
    """Delay one sender thread, leaving executor and transport progress intact.

    Args:
        name: Gate phase selected by the parent test.
        pair_id: Exact request whose sender is delayed.

    Raises:
        TimeoutError: The parent did not release the bounded gate.
    """
    gate = _directory() / f"{name}-{pair_id}"
    if not gate.exists():
        return
    emit(name, pair_id)
    expires = time.monotonic() + GATE_TIMEOUT
    while gate.exists():
        if time.monotonic() >= expires:
            emit("gate_expired", pair_id, gate=name)
            raise TimeoutError(f"Qualification gate {name} did not release")
        time.sleep(0.01)
    emit("gate_released", pair_id, gate=name)


class CancellationProbeWorker(GenerationExecutorWorker):
    """Observe each real TP executor after its ordinary construction and warmup."""

    def setup_engine(self) -> None:
        """Construct the real engine, then observe its ordinary owner methods."""
        _install_sender_gate()
        super().setup_engine()
        engine = self.engine
        assert isinstance(engine, PyExecutor)
        manager = engine.kv_cache_manager
        assert isinstance(manager, KVCacheManagerV2)
        assert engine.kv_connector_manager is None
        transceiver = engine.kv_cache_transceiver
        assert transceiver._context_first_ownership
        observed: dict[int, int] = {}
        cancelled: set[int] = set()

        for method_name, event in (
            ("respond_and_send_async", "source_retained"),
            ("request_and_receive_async", "receive_published"),
        ):
            original = getattr(transceiver, method_name)

            def observe(request: Any, _original: Callable = original, _event: str = event) -> None:
                """Delegate transfer admission and record its real KV owner."""
                pair_id = _pair(request)
                observed[request.py_request_id] = pair_id
                _original(request)
                emit(
                    _event,
                    pair_id,
                    request_id=request.py_request_id,
                    kv_owned=request.py_request_id in manager.kv_cache_map,
                )

            setattr(transceiver, method_name, observe)

        cancel = engine.disagg.cancel_gen_logically

        def cancel_gen(request: Any) -> bool:
            """Record the coordinator's unchanged cancellation disposition."""
            accepted = cancel(request)
            if accepted:
                cancelled.add(request.py_request_id)
                emit("gen_cancel_accepted", _pair(request), request_id=request.py_request_id)
            return accepted

        engine.disagg.cancel_gen_logically = cancel_gen
        terminate = engine._terminate_request

        def terminate_request(request: Any) -> None:
            """Observe both logical deferral and later physical termination."""
            if request.py_request_id in cancelled:
                emit(
                    "termination_attempt",
                    _pair(request),
                    state=request.state.name,
                    cancelled=request.is_finished_due_to_cancellation,
                    kv_owned=request.py_request_id in manager.kv_cache_map,
                    retained=request.py_request_id in engine.disagg._retiring_gen_receives,
                )
            terminate(request)

        engine._terminate_request = terminate_request
        free = engine._free_request_resources

        def free_resources(request: Any) -> None:
            """Observe actual resource release, without supplying safety evidence."""
            pair_id = _pair(request)
            owned = request.py_request_id in manager.kv_cache_map
            free(request)
            if pair_id is not None:
                emit(
                    "resources_released",
                    pair_id,
                    request_id=request.py_request_id,
                    was_owned=owned,
                    kv_owned=request.py_request_id in manager.kv_cache_map,
                    state=request.state.name,
                    transfer_owned=request.py_request_id
                    in engine.async_transfer_manager.requests_in_transfer(),
                )

        engine._free_request_resources = free_resources
        enqueue = engine._enqueue_responses

        def enqueue_responses(responses: Any) -> None:
            """Observe every original result; do not deduplicate terminal output."""
            responses = list(responses)
            for request_id, response in responses:
                if response is not None and request_id in observed:
                    result = None if response.result is None else response.result.get_result()
                    emit(
                        "response",
                        observed[request_id],
                        error=response.error_msg,
                        terminal=response.result is None or response.result.is_final,
                        finish_reasons=[]
                        if result is None
                        else [reason.name for reason in result.finish_reasons],
                    )
            enqueue(responses)

        engine._enqueue_responses = enqueue_responses
        forward = engine.model_engine.forward

        def model_forward(batch: Any, *args: Any, **kwargs: Any) -> Any:
            """Record real scheduled requests and preserve model execution."""
            for request in batch.all_requests():
                pair_id = _pair(request)
                if pair_id is not None:
                    emit("model_batch", pair_id, request_id=request.py_request_id)
            return forward(batch, *args, **kwargs)

        engine.model_engine.forward = model_forward
        emit(
            "worker_ready",
            world_size=engine.dist.world_size,
            manager=type(manager).__name__,
            transceiver=type(transceiver).__name__,
            agent=type(transceiver._transfer_worker._agent).__name__,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            trtllm_version=tensorrt_llm.__version__,
            binding_path=tensorrt_llm.bindings.__file__,
        )


def _create_executor(
    worker_kwargs: dict,
    model_world_size: int,
    mpi_session: Any,
    postproc_worker_config: Any,
    is_llm_executor: bool,
    use_worker: bool = False,
) -> GenerationExecutorProxy:
    """Use the normal IPC/MPI worker path with observation installed per rank.

    Args:
        worker_kwargs: Original worker constructor arguments.
        model_world_size: Number of actual model ranks.
        mpi_session: Existing MPI session, or None for normal construction.
        postproc_worker_config: Original postprocessing configuration.
        is_llm_executor: Original executor-role selection.
        use_worker: Must be false for this IPC qualification path.

    Returns:
        The real proxy, using the observing worker subclass on each rank.
    """
    assert not use_worker
    return GenerationExecutorProxy(
        worker_kwargs,
        model_world_size=model_world_size,
        mpi_session=mpi_session,
        worker_cls=CancellationProbeWorker,
        postproc_worker_config=postproc_worker_config,
        is_llm_executor=is_llm_executor,
    )


def _install_sender_gate() -> None:
    """Install a bounded observer after genuine native submission."""
    submit = Sender._submit_transfer

    def submit_transfer(
        sender: Sender, task: Any, peer_rank: int, request: Any, on_submitted: Any = None
    ) -> Any:
        """Preserve submission and its result while delaying status observation."""

        def submitted() -> None:
            """Preserve the original callback before entering the test gate."""
            if on_submitted is not None:
                on_submitted()
            _wait_gate("native_submitted", task._unique_rid)

        return submit(sender, task, peer_rank, request, on_submitted=submitted)

    Sender._submit_transfer = submit_transfer


def _install_handoff_gate() -> None:
    """Observe coordinator dispatch and optionally delay the CTX-to-GEN handoff."""
    from tensorrt_llm.serve.openai_client import OpenAIClient
    from tensorrt_llm.serve.openai_disagg_service import OpenAIDisaggregatedService

    verify = OpenAIDisaggregatedService._verify_ctx_response
    send = OpenAIClient.send_request

    async def verify_response(service: Any, response: Any) -> Any:
        """Keep real validation and its result; delay only the selected handoff."""
        result = await verify(service, response)
        gate = _directory() / "hold_handoff"
        if not gate.exists():
            return result
        pair_id = response.choices[0].disaggregated_params.disagg_request_id
        emit("handoff_held", pair_id)
        expires = time.monotonic() + GATE_TIMEOUT
        try:
            while gate.exists():
                if time.monotonic() >= expires:
                    emit("gate_expired", pair_id, gate="handoff")
                    raise TimeoutError("Handoff qualification gate did not release")
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            emit("handoff_cancelled", pair_id)
            raise
        return result

    async def send_request(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        """Observe every real GEN dispatch without changing HTTP handling."""
        params = request.disaggregated_params
        if params is not None and params.request_type == "generation_only":
            emit("gen_dispatched", params.disagg_request_id)
        return await send(client, request, *args, **kwargs)

    OpenAIDisaggregatedService._verify_ctx_response = verify_response
    OpenAIClient.send_request = send_request


def main() -> None:
    """Install test-only instrumentation, then invoke the official serving CLI."""
    if sys.argv[1] == "disaggregated":
        _install_handoff_gate()
    else:
        GenerationExecutor._create_ipc_executor = staticmethod(_create_executor)
    from tensorrt_llm.commands.serve import main as serve

    serve()


if __name__ == "__main__":
    main()
