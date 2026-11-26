# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-node DeepSeek-V4.1-Flash NVFP4 stress with a post-load GSM8K gate.

Run the same selected case on both nodes, with LLM_MODELS_ROOT set::

    srun --kill-on-bad-exit=1 --nodes=2 --ntasks-per-node=1 \
        python -m pytest stress_test/test_disaggregated_multinode_stress.py \
        --output-dir=/path/to/results -sv

Each node needs four Blackwell-or-newer GPUs with at least 140000 MiB each.
Node 0 owns the TP4 context worker, frontend and client; node 1 owns the TP4
generation worker with DSpark. The nodes need HTTP and NIXL/UCX connectivity.
"""

import os
import socket
import time
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from multinode_utils import check_peer, peer_connection, slurm_topology, wait_for_signal

if TYPE_CHECKING:
    from openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

__extra_import_path__ = [
    "~/tests/unittest/llmapi/apps",
    "~/tests/integration/defs",
    "~/tests/integration/defs/disaggregated",
]

pytestmark = [pytest.mark.threadleak(enabled=False), pytest.mark.timeout(28800)]

MODEL = "DeepSeek-V4.1-Flash-NVFP4"
CTX_PORT, GEN_PORT, FRONTEND_PORT, CONTROL_PORT = 8001, 8002, 8000, 8003
STARTUP_TIMEOUT = 7200
FRONTEND_TIMEOUT = 300
STRESS_TIMEOUT = 12600
ACCURACY_TIMEOUT = 7200
REQUEST_COUNT = 10000
CONCURRENCY = 512


def _worker_config(generation: bool) -> dict[str, object]:
    from deepseek_v41_serving import model_kwargs

    config = model_kwargs()
    config.update(
        max_batch_size=64,
        max_num_tokens=8192,
        max_seq_len=10240,
        enable_iter_perf_stats=True,
        print_iter_log=True,
        cache_transceiver_config={"backend": "NIXL", "transceiver_runtime": "PYTHON"},
        kv_cache_config={
            "dtype": "fp8_ds_mla",
            "tokens_per_block": 128,
            "max_tokens": 262144,
            "free_gpu_memory_fraction": 0.5,
            "host_cache_size": 2 * (1 << 30),
            "enable_block_reuse": False,
            "enable_swa_scratch_reuse": False,
        },
        cuda_graph_config={"enable_padding": True, "batch_sizes": [1, 2, 4, 8, 16, 32, 64]},
    )
    if generation:
        config["speculative_config"] = {"decoding_type": "DSpark", "max_draft_len": 5}
    else:
        config.update(cuda_graph_config=None, disable_overlap_scheduler=True)
    return config


def _worker_env() -> dict[str, str]:
    from defs.deepseek_v41_serving import baseline_env
    from test_common.cluster_env import apply_ucx_env

    env = {
        key: value
        for key, value in baseline_env(dict(os.environ)).items()
        if not key.startswith(("SLURM_", "OMPI_", "PMI_", "PMIX_"))
        and key not in {"MASTER_ADDR", "MASTER_PORT"}
    }
    apply_ucx_env(env, cluster_name=os.environ.get("SLURM_CLUSTER_NAME", ""))
    env.update(
        OMPI_ALLOW_RUN_AS_ROOT="1",
        OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
        TRTLLM_NIXL_KVCACHE_BACKEND="UCX",
        TLLM_KV_CACHE_MANAGER_V2_THP="0",
    )
    return env


@dataclass
class _Cluster:
    rank: int
    hosts: list[str]
    connection: socket.socket
    worker: "RemoteOpenAIServer"
    frontend: "RemoteDisaggOpenAIServer | None"
    model_path: str
    artifacts: Path

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
def disagg_cluster(request: pytest.FixtureRequest, output_dir: str) -> Iterator[_Cluster]:
    from defs.common import resolve_llm_model_path
    from defs.conftest import get_device_count, get_device_memory, get_sm_version
    from openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer
    from test_common.http_utils import wait_for_endpoint_ready

    from tensorrt_llm.llmapi.disagg_utils import ServerRole

    rank, hosts = slurm_topology(os.environ)
    if get_sm_version() < 100 or get_device_count() < 4 or get_device_memory() < 140000:
        pytest.skip("Requires four Blackwell-or-newer GPUs with 140000 MiB each on each node")
    model_path = resolve_llm_model_path(MODEL)
    artifacts = Path(output_dir) / "disagg-stress" / request.node.name / f"node-{rank}"
    artifacts.mkdir(parents=True, exist_ok=True)
    with peer_connection(rank, hosts[0], CONTROL_PORT, request.node.nodeid) as connection:
        with ExitStack() as stack:
            worker = stack.enter_context(
                RemoteOpenAIServer(
                    model_path,
                    host="0.0.0.0",
                    port=GEN_PORT if rank else CTX_PORT,
                    role=ServerRole.GENERATION if rank else ServerRole.CONTEXT,
                    rank=0,
                    env=_worker_env(),
                    extra_config=_worker_config(generation=bool(rank)),
                    log_path=str(artifacts / "worker.log"),
                    wait=False,
                )
            )
            cluster = _Cluster(rank, hosts, connection, worker, None, model_path, artifacts)
            assert worker.log_path is not None
            wait_for_endpoint_ready(
                f"http://127.0.0.1:{worker.port}/health",
                timeout=STARTUP_TIMEOUT,
                server_proc=worker.proc,
                check_files=[worker.log_path],
            )
            if rank == 0:
                wait_for_signal(connection, b"R", STARTUP_TIMEOUT, cluster.check_local)
                cluster.frontend = stack.enter_context(
                    RemoteDisaggOpenAIServer(
                        ctx_servers=[f"{hosts[0]}:{CTX_PORT}"],
                        gen_servers=[f"{hosts[1]}:{GEN_PORT}"],
                        port=FRONTEND_PORT,
                        env=_worker_env(),
                        log_path=str(artifacts / "frontend.log"),
                        wait_ready=False,
                        server_start_timeout=FRONTEND_TIMEOUT,
                    )
                )
                assert cluster.frontend.log_path is not None
                wait_for_endpoint_ready(
                    f"http://127.0.0.1:{FRONTEND_PORT}/health",
                    timeout=FRONTEND_TIMEOUT,
                    server_proc=cluster.frontend.proc,
                    additional_server_procs=[(worker.proc, "Context worker")],
                    check_files=[cluster.frontend.log_path, worker.log_path],
                )
            else:
                connection.sendall(b"R")
            yield cluster


def test_deepseek_v41_flash_1p1d_tp4(disagg_cluster: _Cluster) -> None:
    """Gate cross-node KV-transfer stability under 8k/1k load and post-stress accuracy."""
    cluster = disagg_cluster
    if cluster.rank == 1:
        wait_for_signal(
            cluster.connection,
            b"D",
            STARTUP_TIMEOUT + STRESS_TIMEOUT + ACCURACY_TIMEOUT,
            cluster.check_local,
        )
        cluster.check_local()
        cluster.check_logs()
        cluster.connection.sendall(b"P")
        return

    from defs.deepseek_v41_gsm8k import evaluate_gsm8k_server
    from defs.disaggregated.test_disaggregated import enforce_aiperf_error_rate
    from defs.stress_test.stress_test import (
        ServerConfig,
        create_aiperf_command,
        run_aiperf_process,
        warmup_inference,
    )

    from tensorrt_llm.tokenizer.deepseek_v41 import DeepseekV41Tokenizer

    server = ServerConfig(host=cluster.hosts[0], port=FRONTEND_PORT)
    assert warmup_inference(server.url, cluster.model_path, timeout=600)
    artifacts = cluster.artifacts / "aiperf"
    cmd = create_aiperf_command(
        cluster.model_path,
        cluster.model_path,
        REQUEST_COUNT,
        CONCURRENCY,
        server.url,
        input_len_mean=8192,
        input_len_std=0,
        output_len_mean=1024,
        output_len_std=0,
        artifact_dir=str(artifacts),
    )
    cmd.extend(
        [
            "--tokenizer-trust-remote-code",
            "--extra-inputs",
            "ignore_eos:true",
            "--request-cancellation-rate",
            "10",
            "--request-cancellation-delay",
            "0.5",
        ]
    )
    assert run_aiperf_process(cmd, time.time(), STRESS_TIMEOUT, server, health_check=cluster.check)
    enforce_aiperf_error_rate(str(artifacts), max_error_rate=0.01, expected_records=REQUEST_COUNT)
    cluster.check()
    tokenizer = DeepseekV41Tokenizer.from_pretrained(
        cluster.model_path, trust_remote_code=True, local_files_only=True
    )
    evaluate_gsm8k_server(server.url, cluster.model_path, tokenizer, cluster.artifacts / "gsm8k")
    assert warmup_inference(server.url, cluster.model_path, timeout=30, num_warmup_requests=1)
    cluster.check()
    cluster.check_logs()
    cluster.connection.sendall(b"D")
    wait_for_signal(cluster.connection, b"P", 300, cluster.check_local)
