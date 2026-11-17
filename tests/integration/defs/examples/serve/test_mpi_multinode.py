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

"""Serving checks inside the CI launcher's external multi-node MPI world."""

import json
import os
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import requests
from defs.common import get_free_port_in_ci
from defs.conftest import llm_models_root
from defs.examples.serve.test_serve import _wait_for_server_ready

__extra_import_path__ = ["~/tests/unittest/llmapi/apps"]
from openai_server import RemoteOpenAIServer  # noqa: E402

_MODEL = "Qwen3/Qwen3-30B-A3B-Instruct-2507"
_TOPOLOGY_PREFIX = "MPI_SERVE_TOPOLOGY="
_TOPOLOGY_PROBE = """
import json
import socket
from tensorrt_llm.executor.utils import create_mpi_comm_session
from tensorrt_llm.llmapi.mpi_session import get_mpi_world_size

session = create_mpi_comm_session(get_mpi_world_size())
hosts = session.submit_sync(socket.gethostname)
print("MPI_SERVE_TOPOLOGY=" + json.dumps(hosts), flush=True)
"""


def _probe_topology() -> list[str]:
    """Query actual worker hosts without keeping the server's PAIR socket open.

    Returns:
        One hostname per MPI worker, in arbitrary rank order.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", _TOPOLOGY_PROBE],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        print(error.stdout, flush=True)
        print(error.stderr, flush=True)
        raise
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    result.check_returncode()
    rows = [line for line in result.stdout.splitlines() if line.startswith(_TOPOLOGY_PREFIX)]
    assert len(rows) == 1, "MPI worker topology probe did not return exactly one result"
    return json.loads(rows[0].removeprefix(_TOPOLOGY_PREFIX))


@pytest.mark.timeout(4500)
def test_mpi_moe_serve_startup(output_dir: str | None, tmp_path: Path) -> None:
    """Reach readiness and generate across two MPI hosts on three fresh frontends.

    Args:
        output_dir: Persistent CI artifact directory, when configured.
        tmp_path: Local artifact directory for standalone pytest runs.
    """
    assert os.environ.get("TLLM_SPAWN_PROXY_PROCESS") == "1", (
        "Run under the multi-node CI trtllm-llmapi-launch wrapper"
    )
    artifacts = Path(output_dir) / "mpi-serve-startup" if output_dir else tmp_path
    artifacts.mkdir(parents=True, exist_ok=True)
    hosts: list[str] | None = None
    model = Path(llm_models_root()) / _MODEL
    assert (model / "config.json").is_file(), f"Missing CI model checkpoint: {model}"

    for attempt in range(1, 4):
        log_path = artifacts / f"attempt-{attempt}.log"
        try:
            # The outer CI launcher already owns the MPI world. A nested launcher
            # would create a different world and fail to exercise its workers.
            with RemoteOpenAIServer(
                str(model),
                port=get_free_port_in_ci(),
                rank=0,
                wait=False,
                log_path=str(log_path),
                cli_args=[
                    "--backend",
                    "pytorch",
                    "--tp_size",
                    "16",
                    "--ep_size",
                    "16",
                    "--pp_size",
                    "1",
                    "--max_batch_size",
                    "1",
                    "--max_num_tokens",
                    "256",
                    "--max_seq_len",
                    "512",
                ],
                extra_config={
                    "dtype": "bfloat16",
                    "orchestrator_type": None,
                    "enable_attention_dp": True,
                    "kv_cache_config": {"free_gpu_memory_fraction": 0.1},
                    "env_overrides": {
                        "ENABLE_CONFIGURABLE_MOE": "1",
                        "TRTLLM_FORCE_COMM_METHOD": "NCCL_EP",
                        "TRTLLM_WORKER_PRINT_STACKS_PERIOD": "120",
                    },
                },
            ) as server:
                proc = server.proc
                assert proc is not None
                _wait_for_server_ready(proc, server.port, timeout=900)
                response = requests.post(
                    server.url_for("v1", "completions"),
                    json={
                        "model": model.name,
                        "prompt": "The capital of France is",
                        "temperature": 0,
                        "max_tokens": 8,
                    },
                    timeout=(10, 120),
                )
                response.raise_for_status()
                result = response.json()
                (artifacts / f"attempt-{attempt}.json").write_text(json.dumps(result, indent=2))
                assert result["usage"]["completion_tokens"] > 0
                assert result["choices"][0]["text"].strip(), "Serving returned no generated text"
            assert proc.returncode in (0, -signal.SIGTERM), (
                f"Attempt {attempt} required forced shutdown or exited with {proc.returncode}"
            )
            # Probe only after serving so the first frontend submits the first
            # worker task. This collective also requires every serving task to
            # have returned; frontend exit alone does not establish that.
            current_hosts = _probe_topology()
            (artifacts / f"attempt-{attempt}-topology.json").write_text(
                json.dumps(current_hosts, indent=2)
            )
            assert sorted(Counter(current_hosts).values()) == [8, 8], (
                "Expected 16 MPI workers on two distinct hosts, eight per host; "
                f"got {current_hosts}"
            )
            if hosts is None:
                hosts = current_hosts
            else:
                assert Counter(current_hosts) == Counter(hosts)
        finally:
            if log_path.is_file():
                print(log_path.read_text(errors="replace")[-32000:], flush=True)
