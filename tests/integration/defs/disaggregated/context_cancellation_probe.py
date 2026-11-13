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
import atexit
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil
import torch

import tensorrt_llm
from tensorrt_llm._torch.disaggregation import diagnostics
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.worker import GenerationExecutorWorker

PROBE_DIR_ENV = "TLLM_CONTEXT_CANCELLATION_PROBE_DIR"
PROBE_ROLE_ENV = "TLLM_CONTEXT_CANCELLATION_PROBE_ROLE"
GATE_TIMEOUT = 45.0
_CAPTURE_STOP = threading.Event()
_CAPTURE_THREAD: threading.Thread | None = None


def _directory() -> Path:
    """Return this test's process-shared artifact directory."""
    return Path(os.environ[PROBE_DIR_ENV])


def emit(event: str, pair_id: int | None = None, **fields: Any) -> None:
    """Use the production diagnostic stream for test-owned gates and checkpoints.

    Args:
        event: Observed production boundary.
        pair_id: Original disaggregated pairing identity, when available.
        **fields: Snapshot fields obtained at that boundary.
    """
    role = os.environ[PROBE_ROLE_ENV]
    rank = mpi_rank() if role != "coordinator" else 0
    diagnostics.emit_event(event, side=role, request_id=pair_id, rank=rank, **fields)


def _start_capture() -> None:
    """Capture one loss-accounted diagnostic stream per owned serving process."""
    global _CAPTURE_THREAD
    role = os.environ[PROBE_ROLE_ENV]
    rank = mpi_rank() if role != "coordinator" else 0
    path = _directory() / f"{role}-{rank}-{os.getpid()}.jsonl"

    def capture(record: diagnostics.DiagnosticRecord) -> None:
        # The diagnostic capture lock serializes complete writes and sequence
        # assignment. Failed writes or record preparation are latched there.
        with path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def submitted(pair_id: int, local_rank: int, peer_rank: int, piece_id: int | None) -> None:
        _wait_gate("native_submitted", pair_id)

    diagnostics.install_qualification_observers(capture, submitted)
    _CAPTURE_STOP.clear()
    emit("qualification_capture_ready")

    def checkpoints() -> None:
        previous = None
        with diagnostics.suppress_diagnostic_errors():
            while not _CAPTURE_STOP.wait(0.02):
                marker = _directory() / "capture-checkpoint"
                if not marker.exists():
                    continue
                token = marker.read_text()
                if token and token != previous:
                    emit("qualification_checkpoint", checkpoint=token)
                    previous = token

    _CAPTURE_THREAD = threading.Thread(target=checkpoints, daemon=True)
    _CAPTURE_THREAD.start()
    atexit.register(_stop_capture)


def _stop_capture() -> None:
    """Remove test observers only after the owned server has stopped work."""
    _CAPTURE_STOP.set()
    if _CAPTURE_THREAD is not None:
        _CAPTURE_THREAD.join(timeout=1)
    diagnostics.install_qualification_observers(None)


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
        """Construct the real engine with reliable semantic-event capture."""
        process = psutil.Process()
        identity = _directory() / f"rank-{process.pid}.identity"
        temporary = identity.with_suffix(".tmp")
        temporary.write_text(json.dumps([process.pid, process.create_time()]))
        temporary.replace(identity)
        _start_capture()
        super().setup_engine()
        engine = self.engine
        assert isinstance(engine, PyExecutor)
        manager = engine.kv_cache_manager
        assert isinstance(manager, KVCacheManagerV2)
        assert engine.kv_connector_manager is None
        transceiver = engine.kv_cache_transceiver
        assert transceiver.supports_logical_context_cancellation()
        assert transceiver.supports_group_receive_cancellation()
        emit(
            "worker_ready",
            world_size=engine.dist.world_size,
            manager=type(manager).__name__,
            transceiver=type(transceiver).__name__,
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
        _start_capture()
        _install_handoff_gate()
    else:
        GenerationExecutor._create_ipc_executor = staticmethod(_create_executor)
    from tensorrt_llm.commands.serve import main as serve

    try:
        serve()
    finally:
        _stop_capture()


if __name__ == "__main__":
    main()
