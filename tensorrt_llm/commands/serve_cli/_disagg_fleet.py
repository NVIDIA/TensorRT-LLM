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
"""Fleet and coordinator processes of `trtllm-serve disaggregated`, and the
environment variables the launchers pass to the processes they start."""

import asyncio
import atexit
import gc
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

from strenum import StrEnum

from tensorrt_llm.commands import _telemetry as _command_telemetry
from tensorrt_llm.commands.serve_cli._net import (_bind_listener,
                                                  _listener_address_family)
from tensorrt_llm.commands.serve_cli._telemetry import \
    _report_observed_child_failure
from tensorrt_llm.llmapi.disagg_utils import (parse_disagg_config_file,
                                              parse_metadata_server_config_file)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve import OpenAIDisaggServer
from tensorrt_llm.usage import apply_usage_session_config
from tensorrt_llm.usage import config as _telemetry_config
from tensorrt_llm.usage import set_lifecycle_phase


def _launch_disagg_fleet(disagg_cfg, config_file, metadata_server_config_file,
                         request_timeout, server_start_timeout, num_workers,
                         coordinator_url):
    """Launch delegating disaggregated-server workers.

    Each worker has its own ``Popen`` running ``_run_fleet_worker`` on the shared
    public port and pointing at ``coordinator_url``. Separate processes allow an
    explicit per-worker ``TLLM_DISAGG_WORKER_PROCESS_ID`` and independent
    ``SO_REUSEPORT`` sockets. MPI/PMIX/SLURM environment variables are stripped.

    Returns the list of ``Popen`` handles.
    """
    from tensorrt_llm.llmapi.disagg_utils import disagg_process_id_space
    public_host, public_port = disagg_cfg.hostname, disagg_cfg.port
    base_env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("SLURM_", "PMIX_", "PMI_", "OMPI_", "UCX_",
                             "I_MPI_", "HYDRA_", "MPI_"))
    }
    # Fleet frontends delegate to the coordinator; do not inherit its role.
    base_env.pop(DisaggLauncherEnvs.TLLM_DISAGG_ROLE, None)
    # num_workers is explicit config now; ensure no stale WEB_CONCURRENCY leaks in
    # and re-forks each plain-HTTP worker into a nested fleet.
    base_env.pop("WEB_CONCURRENCY", None)
    base_env[DisaggWorkerEnvs.TLLM_DISAGG_COORDINATOR_URL] = coordinator_url
    base_env[DisaggWorkerEnvs.TLLM_DISAGG_CONFIG_FILE] = os.path.abspath(
        config_file)
    if metadata_server_config_file:
        base_env[DisaggWorkerEnvs.TLLM_DISAGG_METADATA_CONFIG_FILE] = \
            os.path.abspath(metadata_server_config_file)
    base_env[DisaggWorkerEnvs.TLLM_DISAGG_REQUEST_TIMEOUT] = str(
        request_timeout)
    base_env[DisaggWorkerEnvs.TLLM_DISAGG_SERVER_START_TIMEOUT] = str(
        server_start_timeout)
    base_env[DisaggWorkerEnvs.TLLM_DISAGG_SCHEDULE_STYLE] = \
        disagg_cfg.schedule_style
    # Propagate the parent's log level so fleet workers' INFO logs (e.g. the
    # per-request [ttft_split] / [coord_api] breakdowns) are not dropped.
    base_env[DisaggWorkerEnvs.TLLM_DISAGG_LOG_LEVEL] = logger.level

    cmd = [
        sys.executable, "-c",
        "from tensorrt_llm.commands.serve_cli._disagg_fleet import "
        "_run_fleet_worker; _run_fleet_worker()"
    ]
    logger.info(
        f"Launching disagg fleet: {num_workers} SO_REUSEPORT workers on "
        f"{public_host}:{public_port}, coordinator={coordinator_url}")

    procid_space = disagg_process_id_space()
    if not 1 <= num_workers <= procid_space:
        raise ValueError(
            f"num_workers must be between 1 and {procid_space}, got {num_workers}"
        )
    fleet = []
    for i in range(num_workers):
        worker_env = dict(base_env)
        # Explicit per-worker process index (no shared counter file).
        worker_env[DisaggWorkerEnvs.TLLM_DISAGG_WORKER_PROCESS_ID] = str(i)
        p = subprocess.Popen(cmd,
                             env=worker_env,
                             stdout=sys.stdout,
                             stderr=sys.stderr)
        logger.info(f"Disagg fleet worker {i} launched (pid={p.pid})")
        fleet.append(p)

    def _cleanup():
        for p in fleet:
            if p.poll() is None:
                p.terminate()
        deadline = time.monotonic() + 10
        for p in fleet:
            try:
                p.wait(timeout=max(0, deadline - time.monotonic()))
            except Exception:
                p.kill()
                p.wait()

    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, _command_telemetry.raise_signal_exit)
    signal.signal(signal.SIGINT, _command_telemetry.raise_signal_exit)
    return fleet


def _serve_disagg_fleet(disagg_cfg, config_file, metadata_server_config_file,
                        request_timeout, server_start_timeout, num_workers,
                        coordinator_url):
    """External coordinator: fork the delegating fleet and block on it.

    No coordinator is started in this process -- the fleet delegates to the
    already-running coordinator at ``coordinator_url``.
    """
    fleet = _launch_disagg_fleet(disagg_cfg, config_file,
                                 metadata_server_config_file, request_timeout,
                                 server_start_timeout, num_workers,
                                 coordinator_url)
    set_lifecycle_phase("serving")
    # Block until any worker exits; a nonzero exit from any worker is a failure.
    try:
        while True:
            for i, p in enumerate(fleet):
                rc = p.poll()
                if rc is not None:
                    if rc != 0:
                        _report_observed_child_failure(rc, "disagg_worker",
                                                       "serving")
                        raise RuntimeError(
                            f"Disagg fleet worker {i} (pid={p.pid}) exited with "
                            f"code {rc}")
                    # A clean exit of one worker ends the fleet.
                    return
            time.sleep(1)
    finally:
        for process in fleet:
            if process.poll() is None:
                process.terminate()


_COORDINATOR_UDS_NAME = "coordinator.sock"

# Linux's sun_path holds 108 bytes, including the terminating NUL.
_MAX_UDS_PATH_BYTES = 107


def _make_coordinator_uds_dir(public_port: int) -> tempfile.TemporaryDirectory:
    """Create a new directory for the coordinator's Unix socket.

    The directory is created with mode 0o700, so only this user can reach the
    socket inside it. It is made in the temporary directory, or in /tmp when
    the socket path there would not fit in a Unix socket address.
    """
    prefix = f"trtllm_disagg_coord_{public_port}_"
    uds_dir = tempfile.TemporaryDirectory(prefix=prefix)
    uds_path = os.path.join(uds_dir.name, _COORDINATOR_UDS_NAME)
    if len(os.fsencode(uds_path)) <= _MAX_UDS_PATH_BYTES:
        return uds_dir
    uds_dir.cleanup()
    logger.warning(f"Coordinator socket path {uds_path} is longer than "
                   f"{_MAX_UDS_PATH_BYTES} bytes; creating its directory in "
                   f"/tmp instead.")
    return tempfile.TemporaryDirectory(prefix=prefix, dir="/tmp")  # nosec B108


def _serve_coordinator_and_fleet(disagg_cfg, config_file,
                                 metadata_server_config_file,
                                 metadata_server_cfg, request_timeout,
                                 server_start_timeout, num_workers):
    """workers>1, no external URL: coordinator server here + a delegating fleet.

    This process runs the coordinator server (owns the ctx/gen routers, cluster
    state, and centralized ZMQ ingest) on ``port-1``; a uvicorn fleet on the
    public port delegates to it.
    """
    from tensorrt_llm.serve.coordinator_server import CoordinatorServer
    from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService

    public_host, public_port = disagg_cfg.hostname, disagg_cfg.port
    coord_port = int(
        os.environ.get(DisaggWorkerEnvs.TLLM_DISAGG_COORDINATOR_PORT,
                       public_port - 1))
    # The fleet is co-located with the implicit coordinator, so route its hot
    # /select,/finish over a Unix domain socket (avoids the TCP loopback stack
    # that dominated per-request latency). Opt-out via the UDS env = "0".
    use_uds = os.environ.get(DisaggWorkerEnvs.TLLM_DISAGG_COORDINATOR_UDS,
                             "1") == "1"
    # Set before the try so that its finally also cleans up after a failed
    # startup.
    coord_socket = None
    fleet = []
    # The socket lives in a new directory that only this user can enter; the
    # permissions uvicorn sets on the socket file itself do not widen that.
    coord_uds_dir = _make_coordinator_uds_dir(public_port) if use_uds else None
    try:
        coord_uds = (os.path.join(coord_uds_dir.name, _COORDINATOR_UDS_NAME)
                     if coord_uds_dir is not None else None)
        # Fleet points at the UDS when enabled; the TCP port stays up for health.
        coord_url = f"unix:{coord_uds}" if coord_uds else \
            f"http://{public_host}:{coord_port}"

        # Bind the coordinator's TCP socket here rather than letting uvicorn bind the
        # hostname, so this listener starts the same way as the standalone server and
        # the fleet workers. uvicorn would resolve the name itself, and a hostname
        # whose AAAA record is link-local resolves to fe80:: with scope id 0 -- which
        # the kernel always rejects with "invalid argument", because the scope id can
        # only be derived from an interface name. Binding AF_INET here also surfaces a
        # port conflict with the same diagnostics as the other two listeners.
        coord_socket = _bind_listener("coordinator",
                                      public_host,
                                      coord_port,
                                      family=_listener_address_family(
                                          "coordinator",
                                          public_host,
                                          coord_port,
                                          policy="ipv4"))

        # 1. Launch the delegating fleet pointed at the implicit coordinator we start
        #    below (port-1 for TCP; UDS for the hot path). Workers hold
        #    CoordinatorClients (no core), so they can't race the ZMQ ingest bind.
        fleet = _launch_disagg_fleet(disagg_cfg, config_file,
                                     metadata_server_config_file,
                                     request_timeout, server_start_timeout,
                                     num_workers, coord_url)

        # 2. Build + serve the coordinator in this process. It OWNS routing state and
        #    builds the owner routers itself (single shared namespace-aware core + ONE
        #    ZMQ ingest server, started once here).
        def _client_factory(router, role, max_retries=1):
            from tensorrt_llm.serve.openai_client import OpenAIHttpClient
            return OpenAIHttpClient(
                router,
                role,
                request_timeout,
                max_retries,
                internal_disagg_auth_key=disagg_cfg.internal_request_auth_key,
                frontend_affinity=disagg_cfg.ctx_frontend_affinity)

        coordinator = DisaggCoordinatorService(
            disagg_cfg,
            _client_factory,
            metadata_config=metadata_server_cfg,
            server_start_timeout_secs=server_start_timeout,
            request_timeout_secs=request_timeout)
        logger.info(f"Coordinator serving on {public_host}:{coord_port} "
                    f"(uds={coord_uds}) (fleet on public port {public_port})")
        set_lifecycle_phase("serving")

        async def _serve_and_monitor():
            server_task = asyncio.create_task(
                CoordinatorServer(coordinator)(
                    public_host,
                    coord_port,
                    uds=coord_uds,
                    keep_alive_timeout=disagg_cfg.server_keep_alive_timeout,
                    sockets=[coord_socket]))

            async def _monitor_fleet():
                while True:
                    for i, process in enumerate(fleet):
                        return_code = process.poll()
                        if return_code is not None:
                            if return_code != 0:
                                _report_observed_child_failure(
                                    return_code, "disagg_worker", "serving")
                                raise RuntimeError(
                                    f"Disagg fleet worker {i} (pid={process.pid}) "
                                    f"exited with code {return_code}")
                            return
                    await asyncio.sleep(1)

            monitor_task = asyncio.create_task(_monitor_fleet())
            done, pending = await asyncio.wait(
                (server_task, monitor_task),
                return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()

        asyncio.run(_serve_and_monitor())
    finally:
        if coord_socket is not None:
            coord_socket.close()
        for process in fleet:
            if process.poll() is None:
                process.terminate()
        if coord_uds_dir is not None:
            coord_uds_dir.cleanup()


def _init_fleet_worker_process():
    """Per-process setup shared by every fleet worker (one OS process each).

    Restores logging/GC state for the fresh ``python -c`` interpreter and tags
    every log line with this worker's PID so the shared-stdout fleet output stays
    attributable.
    """
    # A worker is a plain HTTP process, never an MPI rank; drop WEB_CONCURRENCY so
    # it is never itself re-forked into multiple uvicorn workers.
    os.environ.pop("WEB_CONCURRENCY", None)
    if os.getenv("TRTLLM_DISAGG_SERVER_DISABLE_GC", "1") == "1":
        gc.disable()

    # This is a fresh Python process, so the TRT-LLM logger defaults to WARNING
    # and would drop the workers' INFO logs (per-request [ttft_split] /
    # [coord_api]). Restore the parent's level.
    _worker_log_level = os.environ.get(DisaggWorkerEnvs.TLLM_DISAGG_LOG_LEVEL)
    if _worker_log_level:
        logger.set_level(_worker_log_level)

    # All N fleet workers share one stdout; tag every trtllm log line from this
    # worker with its PID so interleaved fleet output is attributable.
    import logging as _logging
    for _h in _logging.getLogger("TRT-LLM").handlers:
        _h.setFormatter(
            _logging.Formatter(
                fmt=
                f"[%(asctime)s] [fleet-worker pid={os.getpid()}] %(message)s",
                datefmt="%m/%d/%Y-%H:%M:%S"))


def _build_disagg_server_from_env() -> "OpenAIDisaggServer":
    """Build one delegating disagg server from the env the launcher exported.

    Fully stateless -- reads config + coordinator URL from ``DisaggWorkerEnvs``;
    the server holds a remote ``CoordinatorClient`` so routing/readiness are
    delegated to the coordinator. The worker's process index (for the snowflake
    disagg-id) is read from ``TLLM_DISAGG_WORKER_PROCESS_ID``, which the launcher
    set explicitly per worker (see ``worker_local_process_id``).
    """
    config_file = os.environ[DisaggWorkerEnvs.TLLM_DISAGG_CONFIG_FILE]
    coordinator_url = os.environ[DisaggWorkerEnvs.TLLM_DISAGG_COORDINATOR_URL]
    metadata_config_file = os.environ.get(
        DisaggWorkerEnvs.TLLM_DISAGG_METADATA_CONFIG_FILE)
    request_timeout = int(
        os.environ.get(DisaggWorkerEnvs.TLLM_DISAGG_REQUEST_TIMEOUT, "180"))
    server_start_timeout = int(
        os.environ.get(DisaggWorkerEnvs.TLLM_DISAGG_SERVER_START_TIMEOUT,
                       "180"))

    disagg_cfg = parse_disagg_config_file(config_file)
    schedule_style = os.environ.get(DisaggWorkerEnvs.TLLM_DISAGG_SCHEDULE_STYLE)
    if schedule_style:
        disagg_cfg.schedule_style = schedule_style
    metadata_server_cfg = parse_metadata_server_config_file(
        metadata_config_file)

    server = OpenAIDisaggServer(config=disagg_cfg,
                                req_timeout_secs=request_timeout,
                                server_start_timeout_secs=server_start_timeout,
                                metadata_server_cfg=metadata_server_cfg,
                                coordinator_url=coordinator_url)
    logger.info(f"Disagg server built, coordinator={coordinator_url}")
    return server


def _run_fleet_worker():
    """Run one fleet worker process.

    Each ``Popen`` receives an explicit ``TLLM_DISAGG_WORKER_PROCESS_ID`` and
    binds its own ``SO_REUSEPORT`` socket on the shared public port.
    """
    apply_usage_session_config(
        default_usage_context=_telemetry_config.UsageContext.DISAGGREGATED.
        value,
        component="server",
        lifecycle_phase="config_validation",
    )
    return _command_telemetry.run_with_terminal_reporting(
        _run_fleet_worker_impl,
        default_usage_context=_telemetry_config.UsageContext.DISAGGREGATED.
        value,
        infer_signal_from_exit_code=True,
    )


def _run_fleet_worker_impl():
    """Build and run one fleet worker inside its telemetry boundary."""
    _init_fleet_worker_process()
    server = _build_disagg_server_from_env()
    host, port = server._config.hostname, server._config.port

    s = socket.socket(
        _listener_address_family("fleet worker", host, port, policy="ipv4"),
        socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # SO_REUSEPORT: every worker binds the same (host, port); the kernel spreads
    # connections across the workers' accept queues by 4-tuple hash.
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    try:
        s.bind((host, port))
    except OSError as e:
        raise RuntimeError(
            f"Fleet worker failed to SO_REUSEPORT-bind {host}:{port}: {e}")
    pidx = os.environ.get(DisaggWorkerEnvs.TLLM_DISAGG_WORKER_PROCESS_ID, "0")
    logger.info(f"Fleet worker process_id={pidx} bound {host}:{port} "
                f"(SO_REUSEPORT)")
    set_lifecycle_phase("serving")
    asyncio.run(server(host, port, sockets=[s]))


class DisaggLauncherEnvs(StrEnum):
    TLLM_DISAGG_INSTANCE_IDX = "TLLM_DISAGG_INSTANCE_IDX"
    TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT = "TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT"
    TLLM_DISAGG_DEPLOYMENT_ID = "TRTLLM_DISAGG_DEPLOYMENT_ID"
    TLLM_DISAGG_ROLE = "TRTLLM_DISAGG_ROLE"


class DisaggWorkerEnvs(StrEnum):
    # Passed from the `disaggregated` coordinator to the fleet of worker processes
    # (one Popen per worker) via env, then read by _run_fleet_worker in each worker.
    TLLM_DISAGG_COORDINATOR_URL = "TRTLLM_DISAGG_COORDINATOR_URL"
    TLLM_DISAGG_CONFIG_FILE = "TRTLLM_DISAGG_CONFIG_FILE"
    TLLM_DISAGG_METADATA_CONFIG_FILE = "TRTLLM_DISAGG_METADATA_CONFIG_FILE"
    TLLM_DISAGG_REQUEST_TIMEOUT = "TRTLLM_DISAGG_REQUEST_TIMEOUT"
    TLLM_DISAGG_SERVER_START_TIMEOUT = "TRTLLM_DISAGG_SERVER_START_TIMEOUT"
    TLLM_DISAGG_SCHEDULE_STYLE = "TRTLLM_DISAGG_SCHEDULE_STYLE"
    # Parent's logger level; the forked uvicorn fleet is a fresh process that
    # otherwise defaults to WARNING and drops the workers' INFO logs.
    TLLM_DISAGG_LOG_LEVEL = "TRTLLM_DISAGG_LOG_LEVEL"
    # Coordinator transport (read in _serve_coordinator_and_fleet): PORT overrides
    # the coordinator TCP port (default public_port-1); UDS="1" (default) routes the
    # co-located fleet's hot /select,/finish over a Unix domain socket, "0" for TCP.
    TLLM_DISAGG_COORDINATOR_PORT = "TRTLLM_DISAGG_COORDINATOR_PORT"
    TLLM_DISAGG_COORDINATOR_UDS = "TRTLLM_DISAGG_COORDINATOR_UDS"
    # Per-fleet-worker process index (0..N-1) → the snowflake disagg-id process_id
    # field so co-located workers never collide; the launcher sets one distinct
    # value per worker Popen (see worker_local_process_id). Defaults to 0 if unset.
    TLLM_DISAGG_WORKER_PROCESS_ID = "TRTLLM_DISAGG_WORKER_PROCESS_ID"
