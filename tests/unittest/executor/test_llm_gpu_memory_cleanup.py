# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess  # nosec B404
import sys
import threading
from subprocess import PIPE, Popen

import psutil
import pytest

from tensorrt_llm import LLM
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams

# isort: off
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.llm_data import llm_models_root
from utils.util import get_current_process_gpu_memory, skip_single_gpu
# isort: on

TINYLLAMA_REL_PATH = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
MEMORY_TOLERANCE_BYTES = 1_500_000_000


def _gpu_mem_bytes(include_subprocess: bool = True) -> int:
    return get_current_process_gpu_memory(include_subprocess=include_subprocess)


def _mpi_server_children() -> list[psutil.Process]:
    current = psutil.Process()
    return [
        proc
        for proc in current.children(recursive=True)
        if any("mpi4py.futures.server" in arg for arg in proc.cmdline())
    ]


def _assert_memory_near_baseline(
    baseline: int, tolerance_bytes: int = MEMORY_TOLERANCE_BYTES
) -> None:
    current = _gpu_mem_bytes(include_subprocess=True)
    allowed = baseline + tolerance_bytes
    assert current <= allowed, (
        f"GPU memory {current} bytes exceeds baseline {baseline} + "
        f"tolerance {tolerance_bytes} bytes (allowed: {allowed} bytes)"
    )


def _tinyllama_model_path() -> str:
    root = llm_models_root()
    if root is None:
        pytest.skip("LLM_MODELS_ROOT is not set or not accessible")
    model_path = root / TINYLLAMA_REL_PATH
    if not model_path.is_dir():
        pytest.skip(f"TinyLlama model not found at {model_path}")
    return str(model_path)


def _llm_kwargs(model: str) -> dict:
    return dict(
        model=model,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, max_tokens=512),
        cuda_graph_config=None,
        max_seq_len=512,
    )


@pytest.mark.gpu1
def test_llm_shutdown_releases_gpu_memory_single_gpu(process_gpu_memory_info_available):
    model = _tinyllama_model_path()
    baseline = _gpu_mem_bytes(include_subprocess=True)
    sampling_params = SamplingParams(max_tokens=4)

    with LLM(**_llm_kwargs(model)) as llm:
        llm.generate(["hi"], sampling_params)

    if process_gpu_memory_info_available:
        _assert_memory_near_baseline(baseline)
    assert _mpi_server_children() == []

    with LLM(**_llm_kwargs(model)) as llm2:
        llm2.generate(["hi"], sampling_params)


@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")
@skip_single_gpu
def test_llm_shutdown_releases_gpu_memory_mpi_reuse():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    runner_script = os.path.join(cur_dir, "_run_llm_gpu_memory_cleanup_mpi.py")
    assert os.path.exists(runner_script), f"Runner script {runner_script} does not exist"

    command = [
        "mpirun",
        "-n",
        "2",
        "--allow-run-as-root",
        "trtllm-llmapi-launch",
        "python3",
        runner_script,
    ]
    print(" ".join(command))

    with Popen(
        command,
        env=os.environ,
        stdout=PIPE,
        stderr=PIPE,
        bufsize=1,
        start_new_session=True,
        universal_newlines=True,
        cwd=cur_dir,
    ) as process:

        def read_stream(stream, output_stream):
            for line in stream:
                output_stream.write(line)
                output_stream.flush()

        stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, sys.stdout))
        stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, sys.stderr))
        stdout_thread.start()
        stderr_thread.start()

        return_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
