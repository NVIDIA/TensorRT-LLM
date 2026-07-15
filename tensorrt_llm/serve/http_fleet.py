# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Generic HTTP process fleets and generator-backed FastAPI workers."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import socket
import subprocess  # nosec B404
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import uvloop

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.inputs.multimodal import MultimodalServerConfig
    from tensorrt_llm.llmapi.disagg_utils import (
        DisaggClusterConfig,
        MetadataServerConfig,
        ServerRole,
    )


class HttpFleet:
    """Supervise a homogeneous collection of HTTP worker processes."""

    def __init__(self, processes: Sequence[Any]) -> None:
        self.processes = list(processes)

    @staticmethod
    def _return_code(process: Any) -> Optional[int]:
        poll = getattr(process, "poll", None)
        if poll is not None:
            return poll()
        return process.exitcode

    def exited(self) -> list[tuple[int, Any, int]]:
        return [
            (index, process, return_code)
            for index, process in enumerate(self.processes)
            if (return_code := self._return_code(process)) is not None
        ]

    def wait(self, poll_interval: float = 1.0) -> tuple[int, Any, int]:
        """Block until a worker exits and return its index, handle, and code."""
        while True:
            exited = self.exited()
            if exited:
                return exited[0]
            time.sleep(poll_interval)

    async def wait_async(self, poll_interval: float = 1.0) -> tuple[int, Any, int]:
        """Asynchronously wait until a worker exits."""
        while True:
            exited = self.exited()
            if exited:
                return exited[0]
            await asyncio.sleep(poll_interval)

    def cleanup(self, timeout: float = 10.0) -> None:
        """Terminate all workers, escalating to kill after the deadline."""
        for process in self.processes:
            if self._return_code(process) is None:
                process.terminate()
        deadline = time.monotonic() + timeout
        for process in self.processes:
            remaining = max(0.0, deadline - time.monotonic())
            wait = getattr(process, "wait", None)
            if wait is not None:
                try:
                    wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    process.kill()
                    wait()
                continue
            process.join(timeout=remaining)
            if process.is_alive():
                process.kill()
                process.join()


def launch_subprocess_http_fleet(
    command: Sequence[str],
    worker_environments: Sequence[Mapping[str, str]],
    *,
    name: str,
) -> HttpFleet:
    """Launch fresh HTTP workers with per-worker environments."""
    if not worker_environments:
        raise ValueError("HTTP fleet requires at least one worker")
    processes = []
    try:
        for worker_id, environment in enumerate(worker_environments):
            process = subprocess.Popen(  # nosec B603
                command,
                env=dict(environment),
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            logger.info("%s worker %d launched (pid=%d)", name, worker_id, process.pid)
            processes.append(process)
    except (OSError, ValueError):
        HttpFleet(processes).cleanup()
        raise
    return HttpFleet(processes)


@dataclass
class GeneratorProcessConfig:
    """Serializable description of the concrete generator process."""

    kind: str
    kwargs: dict[str, Any]
    service_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrontendProcessConfig:
    """Serializable OpenAIServer construction arguments."""

    host: str
    port: int
    model: str
    tool_parser: Optional[str]
    server_role: Optional[ServerRole]
    metadata_server_cfg: Optional[MetadataServerConfig]
    disagg_cluster_config: Optional[DisaggClusterConfig] = None
    multimodal_server_config: Optional[MultimodalServerConfig] = None
    chat_template: Optional[str] = None
    allow_request_chat_template: bool = False
    embedding_max_queue_delay: float = 0.005
    embedding_max_queue_size: int = 2048
    input_processor_workers: int = 8
    media_load_workers: int = 8
    middleware: Sequence[str] = field(default_factory=tuple)


def _create_generator(config: GeneratorProcessConfig) -> Any:
    kwargs = dict(config.kwargs)
    kwargs.pop("build_config", None)
    if config.kind in ("pytorch", "embedding"):
        from tensorrt_llm import LLM

        if config.kind == "embedding":
            kwargs.pop("encode_only", None)
            kwargs["encode_only"] = True
        return LLM(**kwargs)
    if config.kind == "autodeploy":
        from tensorrt_llm._torch.auto_deploy import LLM

        return LLM(**kwargs)
    if config.kind == "mm_encoder":
        from tensorrt_llm import MultimodalEncoder

        return MultimodalEncoder(**kwargs)
    if config.kind == "visual_gen":
        from tensorrt_llm.visual_gen import VisualGen
        from tensorrt_llm.visual_gen.args import VisualGenArgs

        args = kwargs.get("args")
        if args is not None and not isinstance(args, VisualGenArgs):
            raise TypeError("VisualGen args must be a VisualGenArgs instance")
        return VisualGen(model=kwargs["model"], args=args)
    raise ValueError(f"Unknown generator process kind: {config.kind}")


def _run_generator_process(
    config: GeneratorProcessConfig,
    endpoint: str,
    hmac_key: bytes,
    ready: multiprocessing.synchronize.Event,
    stop: multiprocessing.synchronize.Event,
) -> None:
    from .generator_ipc import GeneratorIpcClient, GeneratorIpcServer
    from .generator_proxy import GeneratorService

    generator = None
    server = None
    service = None

    def request_stop(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        generator = _create_generator(config)
        service = GeneratorService(generator, **config.service_kwargs)
        server = GeneratorIpcServer(service, endpoint, hmac_key=hmac_key)
        server.start()
        with GeneratorIpcClient(server.address):
            pass
        ready.set()
        while not stop.wait(1):
            fatal_error = service.fatal_error()
            if fatal_error is not None:
                raise RuntimeError(f"Generator failed: {fatal_error}")
    finally:
        if server is not None:
            server.close()
        if generator is not None:
            generator.shutdown()


def bind_reuseport_http_socket(host: str, port: int) -> socket.socket:
    addr_info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    address_family = (
        socket.AF_INET6 if all(info[0] == socket.AF_INET6 for info in addr_info) else socket.AF_INET
    )
    sock = socket.socket(address_family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, port))
    return sock


def _run_frontend_process(
    config: FrontendProcessConfig, address: tuple[str, bytes], worker_id: int, sock: socket.socket
) -> None:
    from .generator_proxy import GeneratorProxy
    from .openai_server import OpenAIServer

    os.environ.pop("WEB_CONCURRENCY", None)
    proxy = GeneratorProxy(address, owns_lifecycle=worker_id == 0)
    server = OpenAIServer(
        generator=proxy,
        model=config.model,
        tool_parser=config.tool_parser,
        server_role=config.server_role,
        metadata_server_cfg=config.metadata_server_cfg,
        disagg_cluster_config=config.disagg_cluster_config,
        multimodal_server_config=config.multimodal_server_config,
        chat_template=config.chat_template,
        allow_request_chat_template=config.allow_request_chat_template,
        embedding_max_queue_delay=config.embedding_max_queue_delay,
        embedding_max_queue_size=config.embedding_max_queue_size,
        input_processor_workers=config.input_processor_workers,
        media_load_workers=config.media_load_workers,
    )
    if config.middleware:
        from tensorrt_llm.commands.serve import _apply_fastapi_middlewares

        _apply_fastapi_middlewares(server.app, config.middleware)
    with sock:
        uvloop.run(server(config.host, config.port, sockets=[sock]))


def run_generator_http_fleet(
    generator_config: GeneratorProcessConfig,
    frontend_config: FrontendProcessConfig,
    num_http_workers: int,
    *,
    startup_timeout: float = 7200,
) -> None:
    """Run one generator process and a supervised FastAPI process fleet."""
    if num_http_workers <= 1:
        raise ValueError("HTTP fleet requires at least two workers")

    from tensorrt_llm._utils import set_prometheus_multiproc_dir
    from tensorrt_llm.llmapi.mpi_session import find_free_ipc_addr

    context = multiprocessing.get_context("spawn")
    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        set_prometheus_multiproc_dir()
    endpoint = find_free_ipc_addr()
    hmac_key = os.urandom(32)
    ready = context.Event()
    stop = context.Event()
    reservation = bind_reuseport_http_socket(frontend_config.host, frontend_config.port)
    if frontend_config.port == 0:
        frontend_config.port = reservation.getsockname()[1]
    generator_process = context.Process(
        target=_run_generator_process,
        args=(generator_config, endpoint, hmac_key, ready, stop),
        name="trtllm-generator",
    )
    generator_process.start()
    frontends: list[multiprocessing.Process] = []
    frontend_sockets: list[socket.socket] = []
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def stop_supervisor(signum, _frame):
        raise KeyboardInterrupt(f"Received {signal.Signals(signum).name}")

    for signum in previous_handlers:
        signal.signal(signum, stop_supervisor)
    try:
        deadline = time.monotonic() + startup_timeout
        while not ready.wait(0.1):
            if not generator_process.is_alive():
                raise RuntimeError(
                    "Generator process exited during startup "
                    f"(exitcode={generator_process.exitcode})"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for generator IPC readiness")

        frontend_sockets = [
            bind_reuseport_http_socket(frontend_config.host, frontend_config.port)
            for _ in range(num_http_workers)
        ]
        reservation.close()
        address = (endpoint, hmac_key)
        for worker_id, sock in enumerate(frontend_sockets):
            process = context.Process(
                target=_run_frontend_process,
                args=(frontend_config, address, worker_id, sock),
                name=f"trtllm-http-{worker_id}",
            )
            process.start()
            frontends.append(process)
        fleet = HttpFleet(frontends)
        for sock in frontend_sockets:
            sock.close()
        frontend_sockets.clear()

        logger.info("Started one generator and %d FastAPI workers", num_http_workers)
        while generator_process.is_alive() and not fleet.exited():
            time.sleep(0.5)

        exited = [
            f"{process.name} (exitcode={process.exitcode})"
            for process in [generator_process, *fleet.processes]
            if not process.is_alive()
        ]
        if exited:
            raise RuntimeError(f"Required multi-process serve child exited: {exited}")
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        reservation.close()
        for sock in frontend_sockets:
            sock.close()
        stop.set()
        HttpFleet(frontends).cleanup()
        generator_process.join(timeout=10)
        if generator_process.is_alive():
            generator_process.kill()
            generator_process.join()
