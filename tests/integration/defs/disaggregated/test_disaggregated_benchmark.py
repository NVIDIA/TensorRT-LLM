# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import re
import subprocess
import tempfile

import pytest
import yaml
from defs.common import get_disagg_server_url_from_cfg, wait_for_server
from defs.conftest import get_sm_version, llm_models_root
from defs.disaggregated.test_disaggregated_parametrized import cleanup_output_files
from defs.trt_test_alternative import check_call, check_output, popen

from tensorrt_llm._utils import get_free_port
from tensorrt_llm.logger import logger


@pytest.fixture(scope="module")
def benchmark_root():
    llm_root = os.getenv("LLM_ROOT")
    return os.path.join(llm_root, "tensorrt_llm", "serve", "scripts")


@pytest.fixture(scope="module")
def shared_gpt_path():
    DEFAULT_LLM_MODEL_ROOT = os.path.join("/scratch.trt_llm_data", "llm-models")
    LLM_MODELS_ROOT = os.environ.get("LLM_MODELS_ROOT", DEFAULT_LLM_MODEL_ROOT)
    return os.path.join(LLM_MODELS_ROOT, "datasets", "ShareGPT_V3_unfiltered_cleaned_split.json")


@pytest.fixture(scope="function")
def benchmark_model_root(request):
    models_root = llm_models_root()
    if request.param == "DeepSeek-V3-Lite-fp8":
        model_path = os.path.join(models_root, "DeepSeek-V3-Lite", "fp8")
    elif request.param == "DeepSeek-V3-Lite-bf16":
        model_path = os.path.join(models_root, "DeepSeek-V3-Lite", "bf16")
    elif request.param == "llama-v3-8b-hf":
        model_path = os.path.join(models_root, "llama-models-v3", "8B")
    elif request.param == "llama-3.1-8b-instruct-hf-fp8":
        model_path = os.path.join(models_root, "llama-3.1-model", "Llama-3.1-8B-Instruct-FP8")
    else:
        raise ValueError(f"Failed to find the model: {request.param}")
    return model_path


def run_disaggregated_benchmark(
    example_dir,
    config_file,
    benchmark_root,
    benchmark_model_root,
    shared_gpt_path,
    env=None,
    cwd=None,
    num_ranks=2,
    random_input_len=16,
    random_output_len=64,
    num_prompts=100,
    max_concurrency=32,
    skip_warmup=False,
):
    """Run disaggregated benchmark with given configuration."""
    run_env = env.copy()
    run_env["UCX_TLS"] = "^ib"
    workers_cmd = [
        "mpirun",
        "--allow-run-as-root",
        "--oversubscribe",
        "-n",
        str(num_ranks),
        "trtllm-serve",
        "disaggregated_mpi_worker",
        "-c",
        config_file,
    ]

    server_start_timeout = 1200
    server_cmd = [
        "trtllm-serve",
        "disaggregated",
        "--server_start_timeout",
        str(server_start_timeout),
        "-c",
        config_file,
    ]
    server_host, server_port = get_disagg_server_url_from_cfg(config_file)
    try:
        with (  # Start workers
            open("output_workers.log", "w") as output_workers,
            popen(
                workers_cmd, stdout=output_workers, stderr=subprocess.STDOUT, env=run_env, cwd=cwd
            ) as workers_proc,
            # Start server
            open("output_disagg.log", "w") as output_disagg,
            popen(
                server_cmd, stdout=output_disagg, stderr=subprocess.STDOUT, env=run_env, cwd=cwd
            ) as server_proc,
        ):
            # Ensure the server has started
            client_dir = f"{example_dir}/clients"
            client_cmd = [
                "python3",
                f"{client_dir}/disagg_client.py",
                "-c",
                config_file,
                "-p",
                f"{client_dir}/prompts.json",
                "--ignore-eos",
                "--server-start-timeout",
                str(server_start_timeout),
            ]
            # Warm up
            check_call(client_cmd, env=env, poll_procs=[workers_proc, server_proc])
            # Start Benchmark
            benchmark_script = os.path.join(benchmark_root, "benchmark_serving.py")
            benchmark_cmd = [
                "python3",
                benchmark_script,
                "--model",
                benchmark_model_root,
                "--tokenizer",
                benchmark_model_root,
                "--dataset-name",
                "random",
                "--dataset-path",
                shared_gpt_path,
                "--random-input-len",
                str(random_input_len),
                "--random-output-len",
                str(random_output_len),
                "--random-prefix-len",
                "0",
                "--num-prompts",
                str(num_prompts),
                "--max-concurrency",
                str(max_concurrency),
                "--host",
                server_host,
                "--port",
                str(server_port),
                "--ignore-eos",
                "--no-test-input",
                "--percentile-metrics",
                "e2el,ttft",
            ]
            # warm up
            if not skip_warmup:
                check_call(benchmark_cmd, env=env)
            output = check_output(benchmark_cmd, env=env)
            e2el_pattern = r"Median E2EL \(ms\):\s*(\d+\.?\d*)"
            ttft_pattern = r"Median TTFT \(ms\):\s*(\d+\.?\d*)"
            e2el_match = re.search(e2el_pattern, output)
            ttft_match = re.search(ttft_pattern, output)
            if e2el_match and ttft_match:
                median_e2el = float(e2el_match.group(1))
                median_ttft = float(ttft_match.group(1))
                return median_e2el, median_ttft
            else:
                raise ValueError("No benchmark result found")

    except Exception:
        # Print outputs on error
        logger.error("-------- Workers output --------")
        with open("output_workers.log", "r") as f:
            logger.error(f.read())

        logger.error("-------- Disagg server output --------")
        with open("output_disagg.log", "r") as f:
            logger.error(f.read())
        raise
    finally:
        server_proc.terminate()
        workers_proc.terminate()
        server_proc.wait()
        workers_proc.wait()


def get_config_for_benchmark(model_root, backend):
    """Generate config for benchmark test."""
    serve_config = {
        "model": model_root,
        "hostname": "localhost",
        "port": get_free_port(),
        "backend": "pytorch",
        "context_servers": {
            "num_instances": 1,
            "max_batch_size": 2,
            "max_num_tokens": 384,
            "max_seq_len": 384,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": backend,
                "max_tokens_in_buffer": 512,
            },
            "urls": ["localhost:8001"],
        },
        "generation_servers": {
            "num_instances": 1,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "max_batch_size": 2,
            "max_num_tokens": 384,
            "max_seq_len": 384,
            "cache_transceiver_config": {
                "backend": backend,
                "max_tokens_in_buffer": 512,
            },
            "urls": ["localhost:8002"],
        },
    }
    return serve_config


def get_config_for_llama4_kv_cache_overflow(model_root):
    serve_config = {
        "model": model_root,
        "hostname": "localhost",
        "port": get_free_port(),
        "backend": "pytorch",
        "context_servers": {
            "num_instances": 1,
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "moe_expert_parallel_size": 1,
            "enable_attention_dp": False,
            "max_num_tokens": 8192,
            "max_seq_len": 257000,
            "max_input_len": 256000,
            "max_batch_size": 1,
            "trust_remote_code": True,
            "enable_chunked_prefill": True,
            "kv_cache_config": {"enable_block_reuse": False, "free_gpu_memory_fraction": 0.3},
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "cache_transceiver_config": {"backend": "UCX", "max_tokens_in_buffer": 2048},
            "urls": ["localhost:8001"],
        },
        "generation_servers": {
            "num_instances": 1,
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "moe_expert_parallel_size": 1,
            "enable_attention_dp": False,
            "max_num_tokens": 8192,
            "max_seq_len": 257000,
            "max_input_len": 256000,
            "max_batch_size": 1,
            "trust_remote_code": True,
            "enable_chunked_prefill": True,
            "kv_cache_config": {"enable_block_reuse": False, "free_gpu_memory_fraction": 0.3},
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "cache_transceiver_config": {"backend": "UCX", "max_tokens_in_buffer": 2048},
            "urls": ["localhost:8002"],
        },
    }
    return serve_config


def run_disaggregated_genai_perf(
    config_file,
    model_path,
    num_ranks,
    server_start_timeout=1200,
    input_tokens=128000,
    output_tokens=100,
    env=None,
    cwd=None,
):
    """Run disaggregated test with genai-perf for performance/stress testing."""
    cleanup_output_files()
    run_env = env.copy()
    run_env["UCX_TLS"] = "^ib"

    workers_cmd = [
        "mpirun",
        "--allow-run-as-root",
        "--oversubscribe",
        "-n",
        str(num_ranks),
        "trtllm-serve",
        "disaggregated_mpi_worker",
        "-c",
        config_file,
    ]

    server_cmd = [
        "trtllm-serve",
        "disaggregated",
        "--server_start_timeout",
        str(server_start_timeout),
        "-c",
        config_file,
    ]

    server_host, server_port = get_disagg_server_url_from_cfg(config_file)

    artifact_dir = os.path.join(cwd or ".", "benchmark-results")

    try:
        with (
            open("output_workers.log", "w") as output_workers,
            popen(
                workers_cmd, stdout=output_workers, stderr=subprocess.STDOUT, env=run_env, cwd=cwd
            ) as workers_proc,
            open("output_disagg.log", "w") as output_disagg,
            popen(
                server_cmd, stdout=output_disagg, stderr=subprocess.STDOUT, env=run_env, cwd=cwd
            ) as server_proc,
        ):
            # Wait for server to be ready
            if not wait_for_server(server_host, server_port, timeout_seconds=server_start_timeout):
                raise RuntimeError(
                    f"Disaggregated server did not become ready within {server_start_timeout} seconds"
                )

            # Run genai-perf
            genai_perf_cmd = [
                "genai-perf",
                "profile",
                "--model",
                model_path,
                "--tokenizer",
                model_path,
                "--endpoint-type",
                "chat",
                "--endpoint",
                "/v1/chat/completions",
                "--streaming",
                "--url",
                f"{server_host}:{server_port}",
                "--synthetic-input-tokens-mean",
                str(input_tokens),
                "--synthetic-input-tokens-stddev",
                "0",
                "--output-tokens-mean",
                str(output_tokens),
                "--output-tokens-stddev",
                "0",
                "--extra-inputs",
                f"max_tokens:{output_tokens}",
                "--extra-inputs",
                f"min_tokens:{output_tokens}",
                "--extra-inputs",
                "ignore_eos:true",
                "--concurrency",
                "1",
                "--warmup-request-count",
                "8",
                "--num-dataset-entries",
                "64",
                "--random-seed",
                "100",
                "--artifact-dir",
                artifact_dir,
                "--",
                "-v",
                "-H",
                "Authorization: Bearer NOT USED",
                "-H",
                "Accept: text/event-stream",
                "-p",
                "200000",
            ]

            check_call(genai_perf_cmd, env=env, poll_procs=[workers_proc, server_proc])

    except Exception:
        # Print outputs on error
        logger.error("-------- Workers output (last 30 lines) --------")
        try:
            with open("output_workers.log", "r") as f:
                lines = f.read().split("\n")
                for line in lines[-30:]:
                    if line.strip():
                        logger.error(line)
        except FileNotFoundError:
            pass

        logger.error("-------- Disagg server output (last 30 lines) --------")
        try:
            with open("output_disagg.log", "r") as f:
                lines = f.read().split("\n")
                for line in lines[-30:]:
                    if line.strip():
                        logger.error(line)
        except FileNotFoundError:
            pass
        raise
    finally:
        server_proc.terminate()
        workers_proc.terminate()
        server_proc.wait()
        workers_proc.wait()


@pytest.mark.parametrize(
    "benchmark_model_root",
    [
        "DeepSeek-V3-Lite-fp8",
        "DeepSeek-V3-Lite-bf16",
        "llama-v3-8b-hf",
        "llama-3.1-8b-instruct-hf-fp8",
    ],
    indirect=True,
)
def test_disaggregated_benchmark_on_diff_backends(
    disaggregated_test_root,
    disaggregated_example_root,
    llm_venv,
    benchmark_model_root,
    benchmark_root,
    shared_gpt_path,
):
    """Benchmark test comparing NIXL vs UCX cache transceiver backends."""
    if (
        "DeepSeek-V3-Lite" in benchmark_model_root
        and "fp8" in benchmark_model_root
        and get_sm_version() != 90
    ):
        pytest.skip("The test should only run on Hopper")
    nixl_config = get_config_for_benchmark(benchmark_model_root, "NIXL")
    ucx_config = get_config_for_benchmark(benchmark_model_root, "UCX")
    temp_dir = tempfile.TemporaryDirectory()
    nixl_config_path = os.path.join(temp_dir.name, "nixl_config.yaml")
    ucx_config_path = os.path.join(temp_dir.name, "ucx_config.yaml")
    with open(nixl_config_path, "w", encoding="utf-8") as f:
        yaml.dump(nixl_config, f)
    with open(ucx_config_path, "w", encoding="utf-8") as f:
        yaml.dump(ucx_config, f)

    env = llm_venv._new_env.copy()
    nixl_e2el, nixl_ttft = run_disaggregated_benchmark(
        disaggregated_example_root,
        nixl_config_path,
        benchmark_root,
        benchmark_model_root,
        shared_gpt_path,
        env=env,
        cwd=llm_venv.get_working_directory(),
    )
    ucx_e2el, ucx_ttft = run_disaggregated_benchmark(
        disaggregated_example_root,
        ucx_config_path,
        benchmark_root,
        benchmark_model_root,
        shared_gpt_path,
        env=env,
        cwd=llm_venv.get_working_directory(),
    )
    print(f"Nixl E2EL: {nixl_e2el} ms, UCX E2EL: {ucx_e2el} ms")
    print(f"Nixl TTFT: {nixl_ttft} ms, UCX TTFT: {ucx_ttft} ms")

    assert ucx_e2el > 0 and nixl_e2el > 0 and nixl_e2el < 1.05 * ucx_e2el
    assert ucx_ttft > 0 and nixl_ttft > 0 and nixl_ttft < 1.05 * ucx_ttft


def get_config_for_empty_batch_test(model_root):
    """Generate config for benchmark test."""
    serve_config = {
        "hostname": "localhost",
        "port": get_free_port(),
        "model": model_root,
        "backend": "pytorch",
        "context_servers": {
            "num_instances": 1,
            "build_config": {"max_batch_size": 10, "max_num_tokens": 512, "max_seq_len": 768},
            "max_batch_size": 10,
            "max_num_tokens": 512,
            "max_seq_len": 768,
            "tensor_parallel_size": 2,
            "moe_expert_parallel_size": 2,
            "enable_attention_dp": True,
            "pipeline_parallel_size": 1,
            "print_iter_log": True,
            "cuda_graph_config": None,
            "disable_overlap_scheduler": True,
            "kv_cache_config": {
                "enable_block_reuse": False,
                "free_gpu_memory_fraction": 0.05,
                "max_tokens": 512,
            },
            "cache_transceiver_config": {"max_tokens_in_buffer": 8448, "backend": "DEFAULT"},
            "urls": ["localhost:8001"],
        },
        "generation_servers": {
            "num_instances": 1,
            "build_config": {"max_batch_size": 1, "max_num_tokens": 2048, "max_seq_len": 2560},
            "tensor_parallel_size": 1,
            "moe_expert_parallel_size": 1,
            "enable_attention_dp": False,
            "enable_lm_head_tp_in_adp": False,
            "pipeline_parallel_size": 1,
            "max_batch_size": 1,
            "max_num_tokens": 2048,
            "max_seq_len": 2560,
            "cuda_graph_config": {"enable_padding": True, "batch_sizes": [1]},
            "print_iter_log": True,
            "kv_cache_config": {
                "enable_block_reuse": False,
                "free_gpu_memory_fraction": 0.7,
                "max_tokens": 2560,
            },
            "moe_config": {"backend": "CUTLASS"},
            "cache_transceiver_config": {"max_tokens_in_buffer": 8448, "backend": "DEFAULT"},
            "stream_interval": 1,
            "num_postprocess_workers": 1,
            "urls": ["localhost:8002"],
        },
    }
    return serve_config


@pytest.mark.parametrize("benchmark_model_root", ["DeepSeek-V3-Lite-bf16"], indirect=True)
def test_disaggregated_deepseek_v3_lite_bf16_empty_batch(
    disaggregated_example_root, llm_venv, benchmark_model_root, benchmark_root, shared_gpt_path
):
    src_dst_dict = {
        benchmark_model_root: f"{llm_venv.get_working_directory()}/DeepSeek-V3-Lite/bf16",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    num_ranks = 3
    config = get_config_for_empty_batch_test(benchmark_model_root)
    temp_dir = tempfile.TemporaryDirectory()
    config_path = os.path.join(temp_dir.name, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    env = llm_venv._new_env.copy()
    e2el, ttft = run_disaggregated_benchmark(
        disaggregated_example_root,
        config_path,
        benchmark_root,
        benchmark_model_root,
        shared_gpt_path,
        env=env,
        cwd=llm_venv.get_working_directory(),
        num_ranks=num_ranks,
        num_prompts=10,
        max_concurrency=10,
        random_input_len=384,
        random_output_len=1536,
        skip_warmup=True,
    )
    print(f"E2EL: {e2el} ms, TTFT: {ttft} ms")

    assert e2el > 0 and ttft > 0


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(140000)
@pytest.mark.parametrize(
    "model_path", ["llama4-models/nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8"]
)
def test_llama4_long_context_kv_cache_overflow(
    disaggregated_test_root, disaggregated_example_root, llm_venv, model_path
):
    """Test to reproduce KV cache buffer overflow bug with long context.

    RCCA: https://nvbugspro.nvidia.com/bug/5555681
    """
    models_root = llm_models_root()
    llama4_model_root = os.path.join(models_root, model_path)

    # Create symlink to match config file path
    src_dst_dict = {
        llama4_model_root: f"{llm_venv.get_working_directory()}/{model_path}",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    num_ranks = 8
    config = get_config_for_llama4_kv_cache_overflow(llama4_model_root)
    temp_dir = tempfile.TemporaryDirectory()
    config_path = os.path.join(temp_dir.name, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    run_disaggregated_genai_perf(
        config_file=config_path,
        model_path=llama4_model_root,
        num_ranks=num_ranks,
        server_start_timeout=1200,
        input_tokens=128000,
        output_tokens=100,
        env=llm_venv._new_env,
        cwd=llm_venv.get_working_directory(),
    )
