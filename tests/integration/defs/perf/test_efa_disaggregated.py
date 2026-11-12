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

"""Two-node KV cache V2 inference through the NIXL LIBFABRIC/EFA backend."""

import json
import os
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests
import yaml
from test_common.efa_validation import validate_efa_worker_logs

from .test_perf_sanity import AccuracyConfig, PerfSanityTestConfig

REQUEST_COUNT = 8


class EfaOutputParity(AccuracyConfig):
    """Use the harness's live-server accuracy hook for greedy output parity."""

    def __init__(self, hostnames_dir: Path, concurrency: int) -> None:
        super().__init__({"enable_accuracy_test": True})
        self.hostnames_dir = hostnames_dir
        self.concurrency = concurrency
        self.requests_checked = 0

    def run(
        self,
        model_name: str,
        server_hostname: str,
        server_port: int,
        output_dir: str,
        server_idx: int,
    ) -> None:
        hosts = [
            json.loads((self.hostnames_dir / f"efa-{role}.json").read_text())["hostname"]
            for role in ("CTX_0", "GEN_0")
        ]
        assert all(hosts) and len(set(hosts)) == 2, (
            f"EFA test requires workers on two distinct nodes, got {hosts}"
        )
        server_config = yaml.safe_load(
            (Path(output_dir) / f"server_config.{server_idx}.yaml").read_text()
        )
        gen_urls = server_config["generation_servers"]["urls"]
        assert len(gen_urls) == 1
        gen_url = gen_urls[0]
        if not gen_url.startswith("http://"):
            gen_url = f"http://{gen_url}"
        router_url = f"http://{server_hostname}:{server_port}"
        response = requests.get(f"{gen_url}/v1/models", timeout=30)
        response.raise_for_status()
        model = response.json()["data"][0]["id"]
        passage = "The river passes a forest, a village, and a stone bridge. " * 160
        prompts = [
            passage + "\nList three places that the river passes:",
            passage + "\nDescribe the bridge in one short sentence:",
        ]

        def complete(url: str, prompt: str) -> dict:
            response = requests.post(
                f"{url}/v1/completions",
                json={"model": model, "prompt": prompt, "temperature": 0, "max_tokens": 32},
                timeout=(10, 120),
            )
            response.raise_for_status()
            data = response.json()
            assert data["usage"]["prompt_tokens"] >= 1024
            assert data["usage"]["completion_tokens"] > 0
            text = data["choices"][0]["text"]
            assert text.strip(), "Inference returned no text"
            return {"text": text, "usage": data["usage"]}

        # Ordinary requests to the generation worker compute their own prefill,
        # providing a reference without crossing the KV transfer boundary.
        references = [complete(gen_url, prompt) for prompt in prompts]
        requests_to_send = prompts * (REQUEST_COUNT // len(prompts))
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(complete, router_url, prompt) for prompt in requests_to_send]
            results = [future.result() for future in futures]
        (Path(output_dir) / "efa-output-parity.json").write_text(
            json.dumps({"hosts": hosts, "references": references, "results": results}, indent=2)
        )
        for index, result in enumerate(results):
            reference = references[index % len(prompts)]
            assert result["text"] == reference["text"], (
                f"Disaggregated result {index} differs from ordinary inference"
            )
        self.requests_checked = len(results)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("perf_sanity_test_case", ["disagg-e2e-b300_qwen3_0.6b_efa_functional"])
def test_kv_cache_v2_efa(output_dir: str, perf_sanity_test_case: str) -> None:
    """Run the existing multi-node harness with functional, rather than throughput, gates."""
    # The MPI launcher deliberately clears SLURM_* in pytest. The workers'
    # recorded hostnames below establish actual placement instead.
    assert "DISAGG_SERVING_TYPE" in os.environ, (
        "Launch through the multi-node disaggregated harness"
    )
    assert "NIXL_DISABLE_CUDA_ADDR_WA" not in os.environ, (
        "Test must run without the CUDA address WAR"
    )
    config = PerfSanityTestConfig(perf_sanity_test_case, output_dir)
    config.parse_config_file()
    commands = config.get_commands()
    hostnames_dir = Path(commands._hostnames_dir(0))
    role = commands.disagg_serving_type
    if role in ("CTX_0", "GEN_0"):
        probe = subprocess.run(
            ["/opt/amazon/efa/bin/fi_info", "-p", "efa", "-t", "FI_EP_RDM"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        assert "provider: efa" in probe.stdout, "No EFA RDM provider on this worker"
        hostnames_dir.mkdir(parents=True, exist_ok=True)
        (hostnames_dir / f"efa-{role}.json").write_text(
            json.dumps({"hostname": socket.gethostname(), "fi_info": probe.stdout}, indent=2)
        )

    parity = EfaOutputParity(hostnames_dir, commands.client_configs[0][0].concurrency)
    for clients in commands.client_configs.values():
        for client in clients:
            client.only_run_accuracy = True
            client.accuracy_config = parity
    config.run_ex(commands)
    if role == "BENCHMARK":
        assert parity.requests_checked == REQUEST_COUNT, "The functional client did not run"
        commands.wait_for_gen_log_sentinels()
        logs = {}
        for worker in ("CTX_0", "GEN_0"):
            # MPI executor output is inherited by the srun log; the serving
            # frontend writes its own log. Inspect both files for this role.
            paths = [
                Path(commands.test_output_dir) / f"trtllm-serve.{worker}.0.log",
                Path(commands.test_output_dir) / f"{worker.lower().split('_')[0]}_server_0.log",
            ]
            logs[worker] = "\n".join(
                path.read_text(errors="replace") for path in paths if path.is_file()
            )
        validate_efa_worker_logs(logs["CTX_0"], logs["GEN_0"], received_requests=REQUEST_COUNT)
