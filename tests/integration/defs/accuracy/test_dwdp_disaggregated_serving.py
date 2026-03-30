"""DWDP disaggregated serving accuracy tests.

Separated from test_disaggregated_serving.py to isolate MPI-dependent test
infrastructure for easier maintenance.
"""

import contextlib
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional

import openai
import pytest
import requests
import yaml
from defs.common import get_free_port_in_ci as get_free_port

from tensorrt_llm.llmapi import CompletionOutput, RequestOutput, SamplingParams
from tensorrt_llm.llmapi.llm_args import LlmArgs
from tensorrt_llm.llmapi.tokenizer import load_hf_tokenizer

from ..conftest import llm_models_root, skip_pre_blackwell
from ..trt_test_alternative import popen
from .accuracy_core import LlmapiAccuracyTestHarness
from .test_disaggregated_serving import (
    DEFAULT_SERVER_WAITING_TIMEOUT,
    DEFAULT_TEST_TIMEOUT,
    DuckLLM,
    MyThreadPoolExecutor,
    Result,
    run_accuracy_test,
)


@contextlib.contextmanager
def launch_dwdp_disaggregated_llm(
    worker_config: Dict[str, Any],
    frontend_config: Dict[str, Any],
    model_path: str,
    total_gpus: int,
    server_waiting_timeout: int = DEFAULT_SERVER_WAITING_TIMEOUT,
    max_workers: int = 128,
):
    """Launch DWDP disaggregated serving via mpirun.

    DWDP requires all workers (CTX + GEN) in a single MPI world for
    IPC handle exchange and DWDP group formation.  This function starts
    all workers with ``mpirun`` and launches a separate disaggregated
    frontend server for the client-facing OpenAI API.
    """
    temp_dir = tempfile.TemporaryDirectory()
    worker_config_path = os.path.join(temp_dir.name, "worker_config.yaml")
    frontend_config_path = os.path.join(temp_dir.name, "frontend_config.yaml")

    with open(worker_config_path, "w") as f:
        yaml.dump(worker_config, f, default_flow_style=False, sort_keys=False)
    with open(frontend_config_path, "w") as f:
        yaml.dump(frontend_config, f, default_flow_style=False, sort_keys=False)

    serve_port = frontend_config["port"]

    child_env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(('OMPI_', 'PMIX_', 'PMI_'))
    }

    mpi_cmd = [
        "mpirun", "--allow-run-as-root", "-n",
        str(total_gpus), "trtllm-serve", "disaggregated_mpi_worker", "-c",
        worker_config_path
    ]

    frontend_cmd = [
        "trtllm-serve", "disaggregated", "-c", frontend_config_path,
        "--server_start_timeout",
        str(server_waiting_timeout), "-r", "360000"
    ]

    with (
            MyThreadPoolExecutor(max_workers=max_workers) as thread_pool,
            temp_dir,
            popen(mpi_cmd, env=child_env) as mpi_proc,
            popen(frontend_cmd, env=child_env) as frontend_proc,
    ):
        start_time = time.time()
        server_is_ready = False
        while time.time() - start_time < server_waiting_timeout:
            time.sleep(5)
            for proc, name in [
                (mpi_proc, "mpirun"),
                (frontend_proc, "frontend"),
            ]:
                if proc.poll() is not None:
                    raise Exception(
                        f"{name} process exited with code {proc.returncode}")
            try:
                response = requests.get(
                    f"http://localhost:{serve_port}/cluster_info")
                if response.status_code == 200:
                    cluster_info = response.json()
                    if cluster_info.get("is_ready"):
                        print(f"DWDP cluster ready: {cluster_info}")
                        server_is_ready = True
                        break
            except requests.exceptions.ConnectionError:
                continue
        if not server_is_ready:
            pytest.fail(
                f"DWDP server not ready after {server_waiting_timeout}s")

        model_name = worker_config.get("model", model_path)
        client = openai.OpenAI(api_key="1234567890",
                               base_url=f"http://localhost:{serve_port}/v1",
                               timeout=1800000)

        def send_request(prompt: str, sampling_params: SamplingParams,
                         streaming: bool):
            kwargs = {}
            if sampling_params is not None:
                kwargs.update(
                    max_tokens=sampling_params.max_tokens,
                    temperature=(sampling_params.temperature
                                 if sampling_params.top_p is not None else 0),
                    top_p=sampling_params.top_p,
                    stop=sampling_params.stop,
                    seed=sampling_params.seed)
            response = client.completions.create(model=model_name,
                                                 prompt=prompt,
                                                 stream=streaming,
                                                 **kwargs)
            result = Result(id=0,
                            sampling_params=sampling_params,
                            outputs=[
                                CompletionOutput(text=response.choices[0].text,
                                                 index=0)
                            ])
            requested_output = RequestOutput._from_generation_result(
                result, prompt=prompt)
            setattr(requested_output, "result", result.result)
            return requested_output

        def generate_async(prompt: str,
                           sampling_params: Optional[SamplingParams] = None,
                           streaming: bool = False):
            future = thread_pool.submit(send_request, prompt, sampling_params,
                                        streaming)
            thread_pool.futures.append(future)
            return future

        args = LlmArgs(model=model_path)
        tokenizer = load_hf_tokenizer(model_path)
        try:
            yield DuckLLM(args, tokenizer, generate_async)
        finally:
            all_procs = [frontend_proc, mpi_proc]
            for proc in all_procs:
                if proc.poll() is None:
                    proc.terminate()
            deadline = time.monotonic() + 5
            for proc in all_procs:
                remaining = max(0, deadline - time.monotonic())
                try:
                    proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                except OSError:
                    pass


@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
class TestDwdpDeepSeekV3Lite(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-V3-Lite"

    @pytest.mark.skip_less_device(4)
    @skip_pre_blackwell
    def test_dwdp_accuracy(self):
        model_path = f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only_mtp"

        ctx_port_0 = get_free_port()
        ctx_port_1 = get_free_port()
        gen_port = get_free_port()
        serve_port = get_free_port()

        ctx_server_config = {
            "num_instances": 2,
            "urls": [
                f"localhost:{ctx_port_0}",
                f"localhost:{ctx_port_1}",
            ],
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "disable_overlap_scheduler": True,
            "enable_autotuner": False,
            "enable_chunked_prefill": False,
            "cuda_graph_config": None,
            "max_batch_size": 16,
            "max_num_tokens": 8192,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.4,
                "enable_block_reuse": False,
                "enable_partial_reuse": False,
                "tokens_per_block": 32,
            },
            "cache_transceiver_config": {
                "backend": "UCX",
                "max_tokens_in_buffer": 8192,
            },
            "moe_config": {
                "backend": "CUTEDSL",
            },
            "dwdp_config": {
                "enabled": True,
                "dwdp_size": 2,
                "num_group": 1,
                "experts_per_worker": 36,
                "num_prefetch_experts": 36,
            },
        }

        gen_server_config = {
            "num_instances": 1,
            "urls": [f"localhost:{gen_port}"],
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "disable_overlap_scheduler": True,
            "enable_autotuner": False,
            "enable_chunked_prefill": False,
            "cuda_graph_config": None,
            "max_batch_size": 128,
            "max_num_tokens": 1024,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.5,
                "enable_block_reuse": False,
                "enable_partial_reuse": False,
                "tokens_per_block": 32,
            },
            "cache_transceiver_config": {
                "backend": "UCX",
                "max_tokens_in_buffer": 8192,
            },
            "moe_config": {
                "backend": "CUTEDSL",
            },
        }

        worker_config = {
            "model": model_path,
            "hostname": "localhost",
            "port": serve_port,
            "backend": "pytorch",
            "context_servers": ctx_server_config,
            "generation_servers": gen_server_config,
        }

        frontend_config = {
            "backend": "pytorch",
            "hostname": "localhost",
            "port": serve_port,
            "context_servers": {
                "num_instances": 2,
                "urls": [
                    f"localhost:{ctx_port_0}",
                    f"localhost:{ctx_port_1}",
                ],
            },
            "generation_servers": {
                "num_instances": 1,
                "urls": [f"localhost:{gen_port}"],
            },
        }

        with launch_dwdp_disaggregated_llm(worker_config,
                                           frontend_config,
                                           model_path,
                                           total_gpus=4,
                                           max_workers=128) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])
