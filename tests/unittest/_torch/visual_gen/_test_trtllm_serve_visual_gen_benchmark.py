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
"""Test benchmark_visual_gen.py against a live trtllm-serve server.

Follows the same subprocess pattern as
``tests/unittest/llmapi/apps/_test_trtllm_serve_benchmark.py``.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import pytest
import requests
import yaml

from tensorrt_llm._utils import get_free_port

# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

_LLM_MODELS_ROOT = Path(
    os.environ.get("LLM_MODELS_ROOT", "/home/scratch.trt_llm_data_ci/llm-models/")
)
if not _LLM_MODELS_ROOT.exists():
    _LLM_MODELS_ROOT = Path("/scratch.trt_llm_data/llm-models/")

_WAN_T2V_MODEL = "Wan2.1-T2V-1.3B-Diffusers"
_WAN_T2V_PATH = _LLM_MODELS_ROOT / _WAN_T2V_MODEL


# ---------------------------------------------------------------------------
# Remote server (mirrors RemoteVisualGenServer from test_trtllm_serve_e2e.py)
# ---------------------------------------------------------------------------


class RemoteVisualGenServer:
    MAX_SERVER_START_WAIT_S = 1200

    def __init__(
        self,
        model: str,
        extra_visual_gen_options: Optional[dict] = None,
        cli_args: Optional[List[str]] = None,
        host: str = "localhost",
        port: Optional[int] = None,
    ) -> None:
        self.host = host
        self.port = port if port is not None else get_free_port()
        self._config_file: Optional[str] = None
        self.proc: Optional[subprocess.Popen] = None

        args = ["--host", self.host, "--port", str(self.port)]
        if cli_args:
            args += cli_args

        if extra_visual_gen_options:
            fd, self._config_file = tempfile.mkstemp(suffix=".yml", prefix="vg_bench_cfg_")
            with os.fdopen(fd, "w") as f:
                yaml.dump(extra_visual_gen_options, f)
            args += ["--extra_visual_gen_options", self._config_file]

        launch_cmd = ["trtllm-serve", model] + args
        self.proc = subprocess.Popen(
            launch_cmd,
            env=os.environ.copy(),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._wait_for_server(timeout=self.MAX_SERVER_START_WAIT_S)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

    def terminate(self):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=30)
        self.proc = None
        if self._config_file:
            try:
                os.remove(self._config_file)
            except OSError:
                pass
            self._config_file = None

    def _wait_for_server(self, timeout: float):
        url = f"http://{self.host}:{self.port}/health"
        start = time.time()
        while True:
            try:
                if requests.get(url, timeout=5).status_code == 200:
                    return
            except requests.RequestException as err:
                result = self.proc.poll()
                if result is not None and result != 0:
                    raise RuntimeError("Visual-gen server exited unexpectedly.") from err
            time.sleep(2)
            if time.time() - start > timeout:
                self.terminate()
                raise RuntimeError(f"Visual-gen server failed to start within {timeout}s.")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_visual_gen_options(**extra) -> dict:
    config = {
        "linear": {"type": "default"},
        "parallel": {"dit_cfg_size": 1, "dit_ulysses_size": 1},
    }
    config.update(extra)
    return config


@pytest.fixture(scope="module")
def server():
    with RemoteVisualGenServer(
        model=str(_WAN_T2V_PATH),
        extra_visual_gen_options=_make_visual_gen_options(),
    ) as srv:
        yield srv


@pytest.fixture(scope="module")
def benchmark_script():
    llm_root = os.getenv("LLM_ROOT")
    if llm_root is None:
        llm_root = str(Path(__file__).resolve().parents[4])
    return os.path.join(llm_root, "tensorrt_llm", "serve", "scripts", "benchmark_visual_gen.py")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _WAN_T2V_PATH.is_dir(), reason=f"Model not found at {_WAN_T2V_PATH}")
@pytest.mark.parametrize("backend", ["openai-videos"])
def test_visual_gen_benchmark_video(
    server: RemoteVisualGenServer, benchmark_script: str, backend: str
):
    """Run benchmark_visual_gen.py for video generation and validate output."""
    benchmark_cmd = [
        sys.executable,
        benchmark_script,
        "--backend",
        backend,
        "--model",
        _WAN_T2V_MODEL,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--prompt",
        "A cat walking in a garden",
        "--num-prompts",
        "2",
        "--size",
        "480x320",
        "--num-frames",
        "9",
        "--fps",
        "8",
        "--num-inference-steps",
        "4",
        "--seed",
        "42",
        "--max-concurrency",
        "1",
        "--disable-tqdm",
    ]

    result = subprocess.run(
        benchmark_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
    )

    assert result.returncode == 0
    assert "Serving Benchmark Result (VisualGen)" in result.stdout


@pytest.mark.skipif(not _WAN_T2V_PATH.is_dir(), reason=f"Model not found at {_WAN_T2V_PATH}")
@pytest.mark.parametrize("backend", ["openai-videos"])
def test_visual_gen_benchmark_save_result(
    server: RemoteVisualGenServer, benchmark_script: str, backend: str, tmp_path
):
    """Verify --save-result produces a valid JSON output file."""
    result_dir = str(tmp_path / "results")
    benchmark_cmd = [
        sys.executable,
        benchmark_script,
        "--backend",
        backend,
        "--model",
        _WAN_T2V_MODEL,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--prompt",
        "A bird flying over the ocean",
        "--num-prompts",
        "1",
        "--size",
        "480x320",
        "--num-frames",
        "9",
        "--fps",
        "8",
        "--num-inference-steps",
        "4",
        "--seed",
        "42",
        "--max-concurrency",
        "1",
        "--save-result",
        "--result-dir",
        result_dir,
        "--disable-tqdm",
    ]

    result = subprocess.run(
        benchmark_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
    )

    assert result.returncode == 0
    assert "Serving Benchmark Result (VisualGen)" in result.stdout

    import json

    result_files = list(Path(result_dir).glob("*.json"))
    assert len(result_files) >= 1, f"No JSON result file found in {result_dir}"

    with open(result_files[0]) as f:
        data = json.load(f)
    assert "completed" in data
    assert data["completed"] >= 1
    assert "mean_e2e_latency_ms" in data
