# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAML-configured two-node disaggregated stress tests.

Each test loads ``configs/<test function name without test_>.yaml``.

Run the same selected case on both nodes, with LLM_MODELS_ROOT set::

    srun --kill-on-bad-exit=1 --nodes=2 --ntasks-per-node=1 \
        python -m pytest stress_test/disaggregated/test_multinode.py \
        --output-dir=/path/to/results -sv

Hardware requirements and worker settings come from each case's YAML.
Node 0 owns the context worker, frontend and client; node 1 owns the generation
worker. The nodes need HTTP and NIXL/UCX connectivity and a shared working
filesystem for collecting all process logs.
"""

import os
import re
import socket
import time
from collections import deque
from collections.abc import Iterator
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml
from defs.conftest import skip_pre_blackwell
from defs.stress_test.gsm8k import Gsm8kConfig, check_acceptance_length, evaluate_gsm8k_server
from multinode_utils import (
    check_peer,
    finish_log_writers,
    peer_connection,
    shared_log_directory,
    slurm_topology,
    wait_for_signal,
)

if TYPE_CHECKING:
    from openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

__extra_import_path__ = [
    "~/tests/unittest/llmapi/apps",
    "~/tests/integration/defs",
    "~/tests/integration/defs/disaggregated",
]

pytestmark = [skip_pre_blackwell, pytest.mark.threadleak(enabled=False)]

CTX_PORT, GEN_PORT, FRONTEND_PORT, CONTROL_PORT = 8001, 8002, 8000, 8003


@dataclass(frozen=True)
class _Workload:
    request_count: int
    concurrency: int
    input_len_mean: int
    input_len_std: int
    output_len_mean: int
    output_len_std: int
    extra_args: list[str]
    max_error_rate: float

    def __post_init__(self) -> None:
        if self.request_count <= 0 or self.concurrency <= 0:
            raise ValueError("Stress request_count and concurrency must be positive")
        if not 0 <= self.max_error_rate <= 1:
            raise ValueError("Stress max_error_rate must be between 0 and 1")


@dataclass(frozen=True)
class _StressConfig:
    model: str
    hardware: dict[str, int]
    timeouts: dict[str, int]
    worker_config: dict[str, object]
    context_config: dict[str, object]
    generation_config: dict[str, object]
    env: dict[str, str]
    unset_env_prefixes: list[str]
    stress: _Workload
    accuracy: Gsm8kConfig
    min_acceptance_length: float | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "_StressConfig":
        with path.open() as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise ValueError(f"Stress config must be a YAML mapping: {path}")
        data["stress"] = _Workload(**data["stress"])
        data["accuracy"] = Gsm8kConfig(**data["accuracy"])
        return cls(**data)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "stress_config" not in metafunc.fixturenames:
        return
    name = metafunc.function.__name__.removeprefix("test_")
    config = _StressConfig.from_yaml(Path(__file__).parent / "configs" / f"{name}.yaml")
    metafunc.parametrize(
        "stress_config",
        [pytest.param(config, id="yaml", marks=pytest.mark.timeout(config.timeouts["test"]))],
    )


def _worker_config(config: _StressConfig, generation: bool) -> dict[str, object]:
    overrides = config.generation_config if generation else config.context_config
    return deepcopy({**config.worker_config, **overrides})


def _worker_env(config: _StressConfig) -> dict[str, str]:
    from test_common.cluster_env import apply_ucx_env

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(tuple(config.unset_env_prefixes))
    }
    apply_ucx_env(env)
    env = {
        key: value
        for key, value in env.items()
        if not key.startswith(("SLURM_", "OMPI_", "PMI_", "PMIX_"))
        and key not in {"MASTER_ADDR", "MASTER_PORT"}
    }
    env.update(config.env)
    return env


def _print_process_logs(artifacts: Path) -> None:
    for path in sorted(artifacts.glob("*.log")):
        with path.open(errors="replace") as log:
            tail = "".join(deque(log, maxlen=100))
        print(f"Last 100 lines of {path}:\n{tail}", flush=True)


@dataclass
class _Cluster:
    rank: int
    hosts: list[str]
    connection: socket.socket
    worker: "RemoteOpenAIServer"
    frontend: "RemoteDisaggOpenAIServer | None"
    model_path: str
    artifacts: Path
    config: _StressConfig

    def check_local(self) -> None:
        for server in (self.worker, self.frontend):
            if server is not None:
                assert server.proc is not None and server.proc.poll() is None, (
                    f"Server exited unexpectedly; inspect {server.log_path}"
                )

    def check(self) -> None:
        self.check_local()
        check_peer(self.connection)

    def check_logs(self) -> None:
        from defs.disaggregated.test_disaggregated import scan_logs_for_fatal_errors

        servers = [server for server in (self.worker, self.frontend) if server is not None]
        findings = scan_logs_for_fatal_errors(servers)
        assert not findings, f"Fatal worker/frontend log errors: {findings}"
        assert self.worker.log_path is not None
        with Path(self.worker.log_path).open(errors="replace") as log:
            assert any("Using KvCacheTransceiverV2" in line for line in log), self.worker.log_path


@pytest.fixture
def disagg_cluster(
    request: pytest.FixtureRequest, output_dir: str, stress_config: _StressConfig
) -> Iterator[_Cluster]:
    from defs.common import resolve_llm_model_path
    from defs.conftest import get_device_count, get_device_memory, get_sm_version
    from defs.disaggregated.test_disaggregated import archive_process_logs
    from openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer
    from test_common.http_utils import wait_for_endpoint_ready

    from tensorrt_llm.llmapi.disagg_utils import ServerRole

    rank, hosts = slurm_topology(os.environ)
    hardware = stress_config.hardware
    timeouts = stress_config.timeouts
    if (
        get_sm_version() < hardware["min_sm"]
        or get_device_count() < hardware["gpus_per_node"]
        or get_device_memory() < hardware["min_gpu_memory_mib"]
    ):
        pytest.skip(f"Each node must satisfy the stress config hardware requirements: {hardware}")
    model_path = resolve_llm_model_path(stress_config.model)
    test_name = re.sub(r"[^A-Za-z0-9._-]+", "_", request.node.name)[:200]
    artifacts = Path(output_dir) / "disagg-logs" / test_name
    artifacts.mkdir(parents=True, exist_ok=True)
    with (
        peer_connection(rank, hosts[0], CONTROL_PORT, request.node.nodeid) as connection,
        shared_log_directory(rank, connection, Path.cwd() / "disagg-logs") as work_dir,
    ):
        with ExitStack() as stack:
            stack.callback(_print_process_logs, work_dir)
            if rank == 0:
                stack.callback(archive_process_logs, str(work_dir), output_dir)
            stack.callback(
                finish_log_writers, rank, connection, work_dir, timeouts["peer_completion"]
            )
            worker = stack.enter_context(
                RemoteOpenAIServer(
                    model_path,
                    host="0.0.0.0",
                    port=GEN_PORT if rank else CTX_PORT,
                    role=ServerRole.GENERATION if rank else ServerRole.CONTEXT,
                    rank=0,
                    env=_worker_env(stress_config),
                    extra_config=_worker_config(stress_config, generation=bool(rank)),
                    log_path=str(work_dir / ("worker_gen_0.log" if rank else "worker_ctx_0.log")),
                    wait=False,
                )
            )
            cluster = _Cluster(
                rank, hosts, connection, worker, None, model_path, artifacts, stress_config
            )
            assert worker.log_path is not None
            wait_for_endpoint_ready(
                f"http://127.0.0.1:{worker.port}/health",
                timeout=timeouts["worker_startup"],
                server_proc=worker.proc,
                check_files=[worker.log_path],
            )
            if rank == 0:
                wait_for_signal(connection, b"R", timeouts["worker_startup"], cluster.check_local)
                cluster.frontend = stack.enter_context(
                    RemoteDisaggOpenAIServer(
                        ctx_servers=[f"{hosts[0]}:{CTX_PORT}"],
                        gen_servers=[f"{hosts[1]}:{GEN_PORT}"],
                        port=FRONTEND_PORT,
                        env=_worker_env(stress_config),
                        log_path=str(work_dir / "disagg_server.log"),
                        wait_ready=False,
                        server_start_timeout=timeouts["frontend_startup"],
                    )
                )
                assert cluster.frontend.log_path is not None
                wait_for_endpoint_ready(
                    f"http://127.0.0.1:{FRONTEND_PORT}/health",
                    timeout=timeouts["frontend_startup"],
                    server_proc=cluster.frontend.proc,
                    additional_server_procs=[(worker.proc, "Context worker")],
                    check_files=[cluster.frontend.log_path, worker.log_path],
                )
            else:
                connection.sendall(b"R")
            yield cluster


def _run_stress_test(cluster: _Cluster) -> None:
    """Drive the configured stress load and post-stress server accuracy checks."""
    timeouts = cluster.config.timeouts
    workload = cluster.config.stress
    if cluster.rank == 1:
        wait_for_signal(
            cluster.connection,
            b"D",
            timeouts["worker_startup"] + timeouts["stress"] + timeouts["accuracy"],
            cluster.check_local,
        )
        cluster.check_local()
        cluster.check_logs()
        cluster.connection.sendall(b"P")
        return

    from defs.disaggregated.test_disaggregated import enforce_aiperf_error_rate
    from defs.stress_test.stress_test import (
        ServerConfig,
        create_aiperf_command,
        run_aiperf_process,
        warmup_inference,
    )

    server = ServerConfig(host=cluster.hosts[0], port=FRONTEND_PORT)
    assert warmup_inference(server.url, cluster.model_path, timeout=timeouts["warmup"])
    artifacts = cluster.artifacts / "aiperf"
    cmd = create_aiperf_command(
        cluster.model_path,
        cluster.model_path,
        workload.request_count,
        workload.concurrency,
        server.url,
        input_len_mean=workload.input_len_mean,
        input_len_std=workload.input_len_std,
        output_len_mean=workload.output_len_mean,
        output_len_std=workload.output_len_std,
        artifact_dir=str(artifacts),
    )
    cmd.extend(workload.extra_args)
    assert run_aiperf_process(
        cmd, time.time(), timeouts["stress"], server, health_check=cluster.check
    )
    enforce_aiperf_error_rate(
        str(artifacts),
        max_error_rate=workload.max_error_rate,
        expected_records=workload.request_count,
    )
    cluster.check()
    result = evaluate_gsm8k_server(
        server.url,
        cluster.model_path,
        cluster.config.accuracy,
        cluster.artifacts / "gsm8k",
        timeout=timeouts["accuracy"],
        collect_acceptance_stats=cluster.config.min_acceptance_length is not None,
    )
    if cluster.config.min_acceptance_length is not None:
        check_acceptance_length(result, cluster.config.min_acceptance_length)
    assert warmup_inference(
        server.url,
        cluster.model_path,
        timeout=timeouts["post_stress_warmup"],
        num_warmup_requests=1,
    )
    cluster.check()
    cluster.check_logs()
    cluster.connection.sendall(b"D")
    wait_for_signal(cluster.connection, b"P", timeouts["peer_completion"], cluster.check_local)


def test_deepseek_v41_flash_1p1d_tp4(disagg_cluster: _Cluster) -> None:
    """Gate cross-node KV-transfer stability under 8k/1k load and post-stress accuracy."""
    _run_stress_test(disagg_cluster)
