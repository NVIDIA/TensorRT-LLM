import concurrent
import contextlib
import functools
import itertools
import json
import os
import re
import subprocess
import tempfile
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import openai
import pytest
import requests
import yaml
from defs.common import get_free_port_in_ci as get_free_port

from tensorrt_llm.executor.result import GenerationResultBase
from tensorrt_llm.llmapi import CompletionOutput, RequestOutput, SamplingParams
from tensorrt_llm.llmapi.llm_args import LlmArgs, MTPDecodingConfig
from tensorrt_llm.llmapi.tokenizer import load_hf_tokenizer

from ..conftest import (get_device_count, llm_models_root, parametrize_with_ids,
                        skip_no_hopper, skip_pre_blackwell, skip_pre_hopper)
from ..trt_test_alternative import popen
from .accuracy_core import (GSM8K, MMLU, CnnDailymail,
                            LlmapiAccuracyTestHarness, get_accuracy_task)


class Result(GenerationResultBase):

    def __init__(self, id: int, sampling_params: SamplingParams,
                 outputs: List[CompletionOutput]):
        super().__init__(id, sampling_params)
        self._outputs = outputs
        self._streaming = False

    @property
    def outputs(self) -> List[CompletionOutput]:
        return self._outputs

    def result(self):
        return self


DuckLLM = namedtuple('DuckLLM', ['args', 'tokenizer', 'generate_async'])

# Timeout for the entire test
DEFAULT_TEST_TIMEOUT = 3600
# Timeout for the server waiting
DEFAULT_SERVER_WAITING_TIMEOUT = 2100
# Timeout for the accuracy evaluation
DEFAULT_ACC_EVALUATION_TIMEOUT = 1500
DEEPSEEKV4_TEST_MAX_BATCH_SIZE = 128


@functools.lru_cache(maxsize=1)
def has_nvlink():
    """
    Check if the system has NVLink connectivity between GPUs.

    Returns:
        bool: True if NVLink is detected, False otherwise.
    """
    try:
        # Execute nvidia-smi nvlink command to query NVLink status
        result = subprocess.run(['nvidia-smi', 'nvlink', '-s'],
                                capture_output=True,
                                text=True,
                                check=False)

        # Check if the command executed successfully
        if result.returncode != 0:
            return False

        # Look for bandwidth information (Link X: XX.XXX GB/s pattern)
        # which indicates active NVLink connections
        if re.search(r'Link \d+:\s+[\d.]+\s+GB/s', result.stdout):
            return True

        return False

    except (FileNotFoundError, subprocess.SubprocessError):
        # nvidia-smi not found or execution failed
        return False
    except Exception:
        # Any other unexpected error
        return False


class MyThreadPoolExecutor(ThreadPoolExecutor):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.futures: list[concurrent.futures.Future[RequestOutput]] = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            for future in self.futures:
                future.result()
            return super().__exit__(exc_type, exc_val, exc_tb)

        for future in self.futures:
            future.cancel()
        self.shutdown(wait=True, cancel_futures=True)
        return False


def run_accuracy_test(llm: "DuckLLM",
                      model_name: str,
                      test_sets: List[Union[str, type]] = ["MMLU", "GSM8K"],
                      extra_evaluator_kwargs: Optional[Dict[Union[str, type],
                                                            Dict[str,
                                                                 Any]]] = None,
                      extra_acc_spec: Optional[str] = None,
                      sampling_params: Optional[SamplingParams] = None,
                      timeout: int = DEFAULT_ACC_EVALUATION_TIMEOUT):
    start_time = time.time()
    for test_set in test_sets:
        if isinstance(test_set, str):
            test_set = get_accuracy_task(test_set)
        task = test_set(model_name)

        if extra_evaluator_kwargs is not None:
            kwargs = extra_evaluator_kwargs.get(test_set, {})
        else:
            kwargs = {}
        task.evaluate(llm,
                      extra_acc_spec=extra_acc_spec,
                      extra_evaluator_kwargs=kwargs,
                      sampling_params=sampling_params)
    elapsed_time = time.time() - start_time
    if elapsed_time > timeout:
        pytest.fail(
            f"The accuracy evaluation took too long to complete. Expected: {timeout}s, Actual: {elapsed_time:.2f}s"
        )


@contextlib.contextmanager
def launch_disaggregated_llm(
    disaggregated_server_config: Dict[str, Any],
    ctx_server_config: Dict[str, Any],
    gen_server_config: Dict[str, Any],
    model_name: str,
    tensor_parallel_size: int = 1,
    ctx_model: str = None,
    gen_model: str = None,
    server_waiting_timeout: int = DEFAULT_SERVER_WAITING_TIMEOUT,
    max_workers: int = 16,
    enable_perf=False,
    extra_env: Optional[Dict[str, str]] = None,
    gen_extra_env: Optional[Dict[str, str]] = None,
):
    temp_dir = tempfile.TemporaryDirectory()
    disaggregated_serving_config_path = os.path.join(
        temp_dir.name, "disaggregated_serving_config.yaml")

    if tensor_parallel_size > 1:
        print(
            f"Using unified tp parameter for testing is not recommended. Please use server configs instead."
        )
    perf_max_requests = 50

    def _apply_perf_flags(cfg: Optional[Dict[str, Any]]):
        if not isinstance(cfg, dict):
            return
        if enable_perf:
            # Only set these if the switch is enabled.
            # Use `setdefault` so explicit per-test overrides are preserved.
            cfg.setdefault("return_perf_metrics", True)
            cfg.setdefault("perf_metrics_max_requests", perf_max_requests)

    _apply_perf_flags(disaggregated_server_config)
    _apply_perf_flags(ctx_server_config)
    _apply_perf_flags(gen_server_config)

    # Always assign free port dynamically for service discovery
    serve_port = get_free_port()
    disaggregated_server_config["port"] = serve_port

    # Use HTTP service discovery
    cluster_uri = f"http://localhost:{serve_port}"
    print(f"Using HTTP service discovery at {cluster_uri}")

    # Create service discovery config
    disagg_cluster = {
        "cluster_uri": cluster_uri,
        "cluster_name": "test_cluster",
        "heartbeat_interval_sec": 5,
        "inactive_timeout_sec": 10,
    }

    # Auto-deduce minimal_instances from num_instances
    num_ctx_instances = disaggregated_server_config["context_servers"][
        "num_instances"]
    num_gen_instances = disaggregated_server_config["generation_servers"][
        "num_instances"]
    disagg_cluster["minimal_instances"] = {
        "context_servers": num_ctx_instances,
        "generation_servers": num_gen_instances
    }

    # Inject disagg_cluster into server config (for minimal_instances and is_ready check)
    disaggregated_server_config["disagg_cluster"] = disagg_cluster

    # Inject into worker configs
    ctx_server_config = {**ctx_server_config, "disagg_cluster": disagg_cluster}
    gen_server_config = {**gen_server_config, "disagg_cluster": disagg_cluster}

    with open(disaggregated_serving_config_path, "w") as f:
        yaml.dump(disaggregated_server_config, f)
    ctx_server_config_path = os.path.join(temp_dir.name,
                                          "ctx_server_config.yaml")
    with open(ctx_server_config_path, "w") as f:
        yaml.dump(ctx_server_config, f)
    gen_server_config_path = os.path.join(temp_dir.name,
                                          "gen_server_config.yaml")
    with open(gen_server_config_path, "w") as f:
        yaml.dump(gen_server_config, f)

    args = LlmArgs(model=model_name, tensor_parallel_size=tensor_parallel_size)

    if "FP4" in model_name:
        args.quant_config.quant_algo = "NVFP4"

    trtllm_serve_path = "trtllm-serve"
    # Common arguments for both servers
    ctx_model = ctx_model or model_name
    gen_model = gen_model or model_name
    ctx_args = [
        trtllm_serve_path,
        ctx_model,
        "--host",
        "localhost",
        "--backend",
        "pytorch",
    ]
    gen_args = [
        trtllm_serve_path,
        gen_model,
        "--host",
        "localhost",
        "--backend",
        "pytorch",
    ]
    gen_tp, gen_pp, gen_cp = gen_server_config.get(
        "tensor_parallel_size", tensor_parallel_size), gen_server_config.get(
            "pipeline_parallel_size",
            1), gen_server_config.get("context_parallel_size", 1)
    ctx_tp, ctx_pp, ctx_cp = ctx_server_config.get(
        "tensor_parallel_size", tensor_parallel_size), ctx_server_config.get(
            "pipeline_parallel_size",
            1), ctx_server_config.get("context_parallel_size", 1)

    ctx_total_gpus = ctx_tp * ctx_pp * ctx_cp
    gen_total_gpus = gen_tp * gen_pp * gen_cp

    # Auto-assign ports for workers (port=0 means dynamic assignment)
    ctx_ports = [0] * num_ctx_instances
    gen_ports = [0] * num_gen_instances

    ctx_servers = []
    current_gpu_offset = 0

    base_env = os.environ.copy()
    if extra_env:
        base_env.update(extra_env)

    kv_cache_perf_dir = os.path.join(temp_dir.name, "kv_cache_perf")

    for i, port in enumerate(ctx_ports):
        env = base_env.copy()
        cache_transceiver_config_backend = ctx_server_config.get(
            "cache_transceiver_config", {}).get("backend", "DEFAULT")
        # NIXL backend ignores this env-var fallback; skip it.
        if cache_transceiver_config_backend != "NIXL":
            env["TRTLLM_USE_UCX_KVCACHE"] = "1"
        # Need to set UCX_TLS to ^ib to avoid hangs on CI B200 cluster.
        env["UCX_TLS"] = "^ib"
        if enable_perf:
            env["TRTLLM_KVCACHE_TIME_OUTPUT_PATH"] = kv_cache_perf_dir

        if cache_transceiver_config_backend == "NIXL":
            env["UCX_MM_ERROR_HANDLING"] = "y"
        gpu_range = range(current_gpu_offset,
                          current_gpu_offset + ctx_total_gpus)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_range))
        if not has_nvlink():
            env["UCX_TLS"] = "^cuda_ipc"
        current_gpu_offset += ctx_total_gpus

        ctx_server_args = ctx_args + [
            "--port",
            str(port), "--config", ctx_server_config_path, "--server_role",
            "context", f"--tp_size={ctx_tp}", f"--pp_size={ctx_pp}",
            f"--cp_size={ctx_cp}"
        ]
        if "max_num_tokens" in ctx_server_config:
            ctx_server_args.append(
                f"--max_num_tokens={ctx_server_config['max_num_tokens']}")

        ctx_servers.append((env, ctx_server_args))

    gen_servers = []

    for i, port in enumerate(gen_ports):
        env = base_env.copy()
        if gen_extra_env:
            env.update(gen_extra_env)
        cache_transceiver_config_backend = gen_server_config.get(
            "cache_transceiver_config", {}).get("backend", "DEFAULT")
        # NIXL backend ignores this env-var fallback; skip it.
        if cache_transceiver_config_backend != "NIXL":
            env["TRTLLM_USE_UCX_KVCACHE"] = "1"
        # Need to set UCX_TLS to ^ib to avoid hangs on CI B200 cluster.
        env["UCX_TLS"] = "^ib"
        if enable_perf:
            env["TRTLLM_KVCACHE_TIME_OUTPUT_PATH"] = kv_cache_perf_dir
        if cache_transceiver_config_backend == "NIXL":
            env["UCX_MM_ERROR_HANDLING"] = "y"
        gpu_range = range(current_gpu_offset,
                          current_gpu_offset + gen_total_gpus)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_range))
        if not has_nvlink():
            env["UCX_TLS"] = "^cuda_ipc"
        current_gpu_offset += gen_total_gpus

        gen_server_args = gen_args + [
            "--port",
            str(port), "--config", gen_server_config_path, "--server_role",
            "generation", f"--tp_size={gen_tp}", f"--pp_size={gen_pp}",
            f"--cp_size={gen_cp}"
        ]
        if "max_num_tokens" in gen_server_config:
            gen_server_args.append(
                f"--max_num_tokens={gen_server_config['max_num_tokens']}")

        gen_servers.append((env, gen_server_args))

    @contextlib.contextmanager
    def multi_popen(server_configs, server_name="", enable_redirect_log=False):
        processes = []
        log_files = []
        try:
            for i, (env, args) in enumerate(server_configs):
                if enable_redirect_log:
                    f = open(f"output_{server_name}_{i}.log", "w+")
                    proc = popen(args, env=env, stdout=f, stderr=f)
                    log_files.append(f)
                else:
                    proc = popen(args, env=env)
                processes.append(proc)

            with contextlib.ExitStack() as stack:
                opened_processes = [
                    stack.enter_context(proc) for proc in processes
                ]
                yield opened_processes
            for f in log_files:
                f.close()
        except Exception as e:
            print(
                f"Failed to start disaggregated server processes in multi_popen: {e}"
            )
            raise

    server_cmd = [
        trtllm_serve_path, "disaggregated", "-c",
        disaggregated_serving_config_path, "--server_start_timeout",
        str(server_waiting_timeout), "-r", "360000"
    ]
    with (
            MyThreadPoolExecutor(max_workers=max_workers) as thread_pool,
            temp_dir,
            multi_popen(ctx_servers, "ctx") as ctx_processes,
            multi_popen(gen_servers, "gen") as gen_processes,
            multi_popen([(base_env, server_cmd)], "disagg") as server_processes,
    ):
        start_time = time.time()
        server_is_ready = False
        while time.time() - start_time < server_waiting_timeout:
            time.sleep(5)
            for process in itertools.chain(ctx_processes, gen_processes,
                                           server_processes):
                if process.poll() is not None:
                    raise Exception(
                        f"process {process.pid} exited with code {process.returncode}"
                    )
            try:
                print("Checking cluster_info endpoint for worker registration")
                response = requests.get(
                    f"http://localhost:{serve_port}/cluster_info")
                if response.status_code == 200:
                    cluster_info = response.json()
                    if cluster_info.get("is_ready"):
                        print(f"Cluster ready: {cluster_info}")
                        server_is_ready = True
                        break
            except requests.exceptions.ConnectionError:
                continue
        if not server_is_ready:
            pytest.fail(
                f"Server is not ready after {server_waiting_timeout} seconds. Please check the logs for more details."
            )

        client = openai.OpenAI(api_key="1234567890",
                               base_url=f"http://localhost:{serve_port}/v1",
                               timeout=1800000)

        def send_request(prompt: str, sampling_params: SamplingParams,
                         streaming: bool):
            kwargs = {}
            if sampling_params is not None:
                extra_body = {}
                kwargs.update(
                    max_tokens=sampling_params.max_tokens,
                    n=sampling_params.n,
                    # NB: 'LLM' (cf. SamplingParams) and OpenAI API
                    #     defaults differ (top_p=0 vs. top_p=1).
                    # FIXME: Because 'LLM' does not permit expressly setting
                    #     top_p=0, diverting to temperature=0.
                    temperature=(sampling_params.temperature
                                 if sampling_params.top_p is not None else 0),
                    top_p=sampling_params.top_p,
                    stop=sampling_params.stop,
                    seed=sampling_params.seed)
                if sampling_params.use_beam_search:
                    extra_body.update(use_beam_search=True)
                if (guided_decoding_params :=
                        sampling_params.guided_decoding) is not None:
                    if (schema := guided_decoding_params.json) is not None:
                        extra_body.update(response_format={
                            "type": "json",
                            "schema": json.loads(schema)
                        })
                    elif guided_decoding_params.json_object:
                        extra_body.update(
                            response_format={"type": "json_object"})
                    else:
                        # TODO: Support other guided decoding types
                        raise ValueError(
                            f"Unsupported guided decoding params: {guided_decoding_params}."
                        )
                if extra_body:
                    kwargs.update(extra_body=extra_body)

            response = client.completions.create(model=model_name,
                                                 prompt=prompt,
                                                 stream=streaming,
                                                 **kwargs)
            result = Result(id=0,
                            sampling_params=sampling_params,
                            outputs=[
                                CompletionOutput(text=choice.text, index=idx)
                                for idx, choice in enumerate(response.choices)
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

        def _get_perf_metrics():
            path = "/perf_metrics"
            perf_url = f"http://localhost:{serve_port}{path}"
            try:
                print(f"Fetching perf metrics from {perf_url}")
                resp = requests.get(perf_url, timeout=10)
                if resp.status_code == 200:
                    try:
                        metrics = resp.json()
                        print("perf_metrics JSON:")
                        print(json.dumps(metrics, indent=2, ensure_ascii=False))
                    except ValueError:
                        print("perf_metrics returned non-JSON response:",
                              resp.text)
                else:
                    print(
                        f"perf_metrics returned status {resp.status_code}: {resp.text}"
                    )
            except requests.exceptions.RequestException as e:
                print(f"Error fetching {perf_url}: {e}")

        def _show_kvcache_time(kv_cache_perf_dir, max_lines=100):
            print(f"kv_cache_perf_dir: {kv_cache_perf_dir}")
            for file in os.listdir(kv_cache_perf_dir):
                print(f"file: {file}")
                print(f"{'-'*25} {file}:{max_lines} {'-'*25}")
                with open(os.path.join(kv_cache_perf_dir, file), "r") as f:
                    for line in f.readlines()[-max_lines:]:
                        print(line.strip())

        tokenizer = load_hf_tokenizer(model_name)
        try:
            yield DuckLLM(args, tokenizer, generate_async)
        finally:
            if enable_perf:
                _show_kvcache_time(kv_cache_perf_dir)
                _get_perf_metrics()

            # Gracefully shut down all server processes
            all_processes = list(
                itertools.chain(ctx_processes, gen_processes, server_processes))

            # SIGTERM triggers llm.shutdown() inside each trtllm-serve, cleaning up the executor and MPI workers.
            for process in all_processes:
                if process.poll() is None:
                    process.terminate()

            # Wait up to 5s total, then SIGKILL any process that doesn't exit.
            # This is a safety net for when llm.shutdown() hangs.
            deadline = time.monotonic() + 5
            for process in all_processes:
                remaining = max(0, deadline - time.monotonic())
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass  # already exited between timeout and kill
                except OSError:
                    pass  # process already gone


def run_parallel_test(model_name: str,
                      model_path: str,
                      *,
                      ctx_pp: int,
                      ctx_tp: int,
                      gen_pp: int,
                      gen_tp: int,
                      ctx_instances: int,
                      gen_instances: int,
                      test_sets: List[LlmapiAccuracyTestHarness],
                      ctx_model: str = None,
                      gen_model: str = None):
    total_ctx_gpus = ctx_tp * ctx_pp * ctx_instances
    total_gen_gpus = gen_tp * gen_pp * gen_instances
    if total_ctx_gpus + total_gen_gpus > get_device_count():
        pytest.skip(
            f"Not enough devices for {ctx_instances} ctx instances (ctx_pp={ctx_pp}*ctx_tp={ctx_tp}) + {gen_instances} gen instances (gen_pp={gen_pp}*gen_tp={gen_tp}), total: {total_ctx_gpus + total_gen_gpus}"
        )

    kv_cache_config = {
        "free_gpu_memory_fraction": 0.5,
        "enable_block_reuse": True
    }
    ctx_server_config = {
        "pipeline_parallel_size": ctx_pp,
        "tensor_parallel_size": ctx_tp,
        "disable_overlap_scheduler": True,
        "kv_cache_config": kv_cache_config,
        "cache_transceiver_config": {
            "backend": "DEFAULT",
            "max_tokens_in_buffer": 4096
        }
    }
    gen_server_config = {
        "tensor_parallel_size": gen_tp,
        "pipeline_parallel_size": gen_pp,
        "disable_overlap_scheduler": True,
        "kv_cache_config": kv_cache_config,
        "cache_transceiver_config": {
            "backend": "DEFAULT",
            "max_tokens_in_buffer": 4096
        }
    }

    # No need to generate URLs - workers will register via service discovery
    disaggregated_server_config = {
        "hostname": "localhost",
        "backend": "pytorch",
        "context_servers": {
            "num_instances": ctx_instances,
        },
        "generation_servers": {
            "num_instances": gen_instances,
        }
    }
    with launch_disaggregated_llm(disaggregated_server_config,
                                  ctx_server_config,
                                  gen_server_config,
                                  model_path,
                                  ctx_model=ctx_model,
                                  gen_model=gen_model) as llm:
        run_accuracy_test(llm, model_name, test_sets)


@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
class TestLlama3_1_8BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"

    @skip_pre_hopper
    @pytest.mark.skip_less_device(2)
    @pytest.mark.parametrize("ctx_disable_overlap_scheduler", [False, True])
    @pytest.mark.parametrize("gen_disable_overlap_scheduler", [False, True])
    @pytest.mark.parametrize("ctx_enable_block_reuse", [True, False])
    @pytest.mark.parametrize("gen_enable_block_reuse", [True, False])
    def test_auto_dtype(self, ctx_disable_overlap_scheduler,
                        gen_disable_overlap_scheduler, ctx_enable_block_reuse,
                        gen_enable_block_reuse):
        ctx_server_config = {
            "disable_overlap_scheduler": ctx_disable_overlap_scheduler,
            "kv_cache_config": {
                "enable_block_reuse": ctx_enable_block_reuse
            }
        }
        ctx_server_config["cache_transceiver_config"] = {
            "backend": "DEFAULT",
            "max_tokens_in_buffer": 4096
        }
        gen_server_config = {
            "disable_overlap_scheduler": gen_disable_overlap_scheduler,
            "kv_cache_config": {
                "enable_block_reuse": gen_enable_block_reuse
            }
        }
        gen_server_config["cache_transceiver_config"] = {
            "backend": "DEFAULT",
            "max_tokens_in_buffer": 4096
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["MMLU", "GSM8K"])

    @skip_pre_hopper
    @pytest.mark.skip_less_device(2)
    def test_beam_search(self):
        max_beam_width = 2
        sampling_params = SamplingParams(n=max_beam_width,
                                         best_of=max_beam_width,
                                         use_beam_search=True)
        kv_cache_config = {
            "free_gpu_memory_fraction": 0.5,
            "enable_block_reuse": True,
            "enable_partial_reuse": True,
            "use_kv_cache_manager_v2": False,
        }
        cache_transceiver_config = {
            "backend": "NIXL",
            "transceiver_runtime": "PYTHON",
            "max_tokens_in_buffer": 4096,
        }
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "max_beam_width": max_beam_width,
            "kv_cache_config": kv_cache_config,
            "cache_transceiver_config": cache_transceiver_config,
        }
        gen_server_config = {
            "disable_overlap_scheduler": True,
            "max_beam_width": max_beam_width,
            "kv_cache_config": kv_cache_config,
            "cache_transceiver_config": cache_transceiver_config,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm,
                              self.MODEL_NAME, [CnnDailymail],
                              extra_acc_spec=f"beam_width={max_beam_width}",
                              sampling_params=sampling_params)

    @skip_pre_hopper
    @pytest.mark.skip_less_device(2)
    def test_kv_cache_v2_nixl_python(self):
        """Test with use_kv_cache_manager_v2=True, block_reuse=False, backend=NIXL, transceiver_runtime=PYTHON."""
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "kv_cache_config": {
                "enable_block_reuse": False,
                "use_kv_cache_manager_v2": True
            },
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON"
            }
        }
        gen_server_config = {
            "disable_overlap_scheduler": False,
            "kv_cache_config": {
                "enable_block_reuse": False,
                "use_kv_cache_manager_v2": True
            },
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON"
            }
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "port": 8000,
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1,
                "urls": ["localhost:8001"]
            },
            "generation_servers": {
                "num_instances": 1,
                "urls": ["localhost:8002"]
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])

    @pytest.mark.skip_less_device(2)
    def test_ngram(self):
        speculative_decoding_config = {
            "decoding_type": "NGram",
            "max_draft_len": 4,
            "max_matching_ngram_size": 4,
            "is_keep_all": True,
            "is_use_oldest": True,
            "is_public_pool": True
        }
        kv_cache_config = {
            "free_gpu_memory_fraction": 0.5,
            "enable_block_reuse": False
        }
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "kv_cache_config": kv_cache_config,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            }
        }
        gen_server_config = {
            "disable_overlap_scheduler": True,
            "speculative_config": speculative_decoding_config,
            "kv_cache_config": kv_cache_config,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            }
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])

    @pytest.mark.skip_less_device(2)
    @skip_pre_hopper
    @parametrize_with_ids("overlap_scheduler", [True, False])
    @parametrize_with_ids("eagle3_one_model", [True, False])
    def test_eagle3(self, overlap_scheduler, eagle3_one_model):
        speculative_decoding_config = {
            "decoding_type": "Eagle",
            "max_draft_len": 4,
            "speculative_model":
            f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B",
            "eagle3_one_model": eagle3_one_model
        }
        ctx_server_config = {
            "disable_overlap_scheduler":
            True,  # BS=1 does not need overlap scheduling
            "speculative_config": speculative_decoding_config,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.5,
                "enable_block_reuse": True  # reuse on context requests
            },
            "max_num_tokens": 13393 * 2,
            "max_batch_size": 1,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            },
            "cuda_graph_config": None,
        }
        gen_server_config = {
            "disable_overlap_scheduler": not overlap_scheduler,
            "speculative_config": speculative_decoding_config,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.5,
                "enable_block_reuse": False
            },
            "max_num_tokens": 13393 * 2,
            "max_batch_size": 16,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            },
            "cuda_graph_config": None,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])

    @pytest.mark.skip_less_device(2)
    @skip_pre_hopper
    def test_gen_only_spec_dec(self):
        speculative_decoding_config = {
            "decoding_type": "Eagle",
            "max_draft_len": 4,
            "speculative_model":
            f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B",
            "eagle3_one_model": True,
        }
        ctx_server_config = {
            "disable_overlap_scheduler":
            True,  # BS=1 does not need overlap scheduling
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.5,
                "enable_block_reuse": True  # reuse on context requests
            },
            "max_num_tokens": 13393 * 2,
            "max_batch_size": 1,
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON",
                "max_tokens_in_buffer": 4096,
            },
            "cuda_graph_config": None,
        }
        gen_server_config = {
            "disable_overlap_scheduler": False,
            "speculative_config": speculative_decoding_config,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.5,
                "enable_block_reuse": False
            },
            "max_num_tokens": 13393 * 2,
            "max_batch_size": 16,
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON",
                "max_tokens_in_buffer": 4096,
            },
            "cuda_graph_config": None,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])

    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
    def test_guided_decoding(self, backend: str, mocker):
        mocker.patch.dict(os.environ, {"TRTLLM_XGUIDANCE_LENIENT": "1"})
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "guided_decoding_backend": backend,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            }
        }
        gen_server_config = {
            "guided_decoding_backend": backend,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            }
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["JsonModeEval"])

    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_less_device_memory(48000)
    @parametrize_with_ids("eagle3_one_model", [True, False])
    @pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
    def test_guided_decoding_with_eagle3(self, backend: str,
                                         eagle3_one_model: bool, mocker):
        mocker.patch.dict(os.environ, {"TRTLLM_XGUIDANCE_LENIENT": "1"})
        speculative_decoding_config = {
            "decoding_type": "Eagle",
            "max_draft_len": 3,
            "speculative_model":
            f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B",
            "eagle3_one_model": eagle3_one_model
        }

        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "speculative_config": speculative_decoding_config,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.8,
            },
            "guided_decoding_backend": backend,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            }
        }
        gen_server_config = {
            # Two-model eagle3 does not support overlap scheduler
            "disable_overlap_scheduler": not eagle3_one_model,
            "speculative_config": speculative_decoding_config,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.8,
            },
            "guided_decoding_backend": backend,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            }
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["JsonModeEval"])

    @pytest.mark.parametrize("tp,pp", [(1, 2), (2, 1), (2, 2)],
                             ids=["tp1pp2", "tp2pp1", "tp2pp2"])
    @pytest.mark.parametrize("testset", ["GSM8K", "MMLU"])
    def test_tp_pp_symmetric(self, tp, pp, testset):
        if tp * pp * 2 > get_device_count():
            pytest.skip(f"Not enough devices for tp={tp}*pp={pp} test")
        return run_parallel_test(self.MODEL_NAME,
                                 self.MODEL_PATH,
                                 ctx_pp=pp,
                                 ctx_tp=tp,
                                 gen_pp=pp,
                                 gen_tp=tp,
                                 ctx_instances=1,
                                 gen_instances=1,
                                 test_sets=[get_accuracy_task(testset)])

    @parametrize_with_ids("ctx_pp", [2, 4])
    @parametrize_with_ids("gen_tp", [1, 2])
    @pytest.mark.parametrize("testset", ["GSM8K", "MMLU"])
    def test_ctx_pp_gen_tp_asymmetric(self, ctx_pp, gen_tp, testset):
        if ctx_pp + gen_tp > get_device_count():
            pytest.skip(
                f"Not enough devices for ctx_pp={ctx_pp}+gen_tp={gen_tp} test")
        return run_parallel_test(self.MODEL_NAME,
                                 self.MODEL_PATH,
                                 ctx_pp=ctx_pp,
                                 ctx_tp=1,
                                 gen_pp=1,
                                 gen_tp=gen_tp,
                                 ctx_instances=1,
                                 gen_instances=1,
                                 test_sets=[get_accuracy_task(testset)])

    @pytest.mark.parametrize("testset", ["GSM8K", "MMLU"])
    def test_multi_instance(self, testset):
        return run_parallel_test(self.MODEL_NAME,
                                 self.MODEL_PATH,
                                 ctx_pp=1,
                                 ctx_tp=1,
                                 gen_pp=1,
                                 gen_tp=1,
                                 ctx_instances=2,
                                 gen_instances=2,
                                 test_sets=[get_accuracy_task(testset)])


@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
@skip_pre_hopper
class TestDeepSeekV3Lite(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-V3-Lite"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"

    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_less_device_memory(60000)
    @skip_no_hopper
    def test_nixl_backend(self):
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": "NIXL",
                "max_tokens_in_buffer": 4096
            }
        }
        gen_server_config = {
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": "NIXL",
                "max_tokens_in_buffer": 4096
            }
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["MMLU", "GSM8K"])

    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_less_device_memory(60000)
    @skip_no_hopper
    def test_gen_only_sync(self):
        """Test gen-only synchronous KV transfer path with NIXL Python transceiver.

        Sets TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1 so the gen worker calls
        request_and_receive_sync instead of the async path. Accuracy must be
        identical to the standard async path.
        """
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON",
                "max_tokens_in_buffer": 4096,
            },
        }
        gen_server_config = {
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON",
                "max_tokens_in_buffer": 4096,
            },
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            },
        }
        with launch_disaggregated_llm(
                disaggregated_server_config,
                ctx_server_config,
                gen_server_config,
                self.MODEL_PATH,
                # Apply to both servers: gen worker uses sync receive path.
                extra_env={"TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP": "1"},
        ) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])

    @pytest.mark.skip_less_device(8)
    @skip_pre_hopper
    def test_gen_only_spec_dec(self):
        ctx_server_config = {"disable_overlap_scheduler": True}
        gen_server_config = {"disable_overlap_scheduler": False}
        cache_transceiver_config = {
            "backend": "NIXL",
            "max_tokens_in_buffer": 4096,
            "transceiver_runtime": "PYTHON",
        }
        ctx_server_config["cache_transceiver_config"] = cache_transceiver_config
        gen_server_config["cache_transceiver_config"] = cache_transceiver_config
        gen_server_config["speculative_config"] = {
            "decoding_type": "MTP",
            "max_draft_len": 2
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config,
                                      gen_server_config,
                                      self.MODEL_PATH,
                                      tensor_parallel_size=4) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["MMLU", "GSM8K"])

    @pytest.mark.skip_less_device(8)
    @parametrize_with_ids("overlap_scheduler", [True, False])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @pytest.mark.skip_less_device(8)
    def test_auto_dtype(self, overlap_scheduler, mtp_nextn):
        ctx_server_config = {"disable_overlap_scheduler": True}
        gen_server_config = {"disable_overlap_scheduler": not overlap_scheduler}
        ctx_server_config["cache_transceiver_config"] = {
            "backend": "DEFAULT",
            "max_tokens_in_buffer": 4096
        }
        gen_server_config["cache_transceiver_config"] = {
            "backend": "DEFAULT",
            "max_tokens_in_buffer": 4096
        }
        if mtp_nextn > 0:
            ctx_server_config["speculative_config"] = {
                "decoding_type": "MTP",
                "max_draft_len": mtp_nextn
            }
            gen_server_config["speculative_config"] = {
                "decoding_type": "MTP",
                "max_draft_len": mtp_nextn
            }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config,
                                      gen_server_config,
                                      self.MODEL_PATH,
                                      tensor_parallel_size=4) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["MMLU", "GSM8K"])

    @skip_pre_blackwell
    @pytest.mark.skip_less_device(8)
    @pytest.mark.parametrize(
        "gen_pp,gen_tp,gen_cp,enable_attention_dp", [
            (1, 1, 4, False),
            (1, 2, 2, False),
            (1, 2, 2, True),
            (2, 1, 2, False),
        ],
        ids=["pp1tp1cp4", "pp1tp2cp2", "pp1dp2cp2", "pp2tp1cp2"])
    @pytest.mark.parametrize("cuda_graph_config", [
        None,
        {
            "enable_padding": False,
            "batch_sizes": [1, 2, 4, 8, 16, 32, 64]
        },
        {
            "enable_padding": True,
            "batch_sizes": [1, 2, 4, 8, 16, 32, 64]
        },
    ],
                             ids=[
                                 "cudagraph:none", "cudagraph:without_padding",
                                 "cudagraph:with_padding"
                             ])
    @pytest.mark.parametrize("comms_medium", ["fifo_v1", "fifo_v2", "nccl"])
    def test_auto_dtype_with_helix(self, comms_medium, cuda_graph_config,
                                   gen_pp, gen_tp, gen_cp, enable_attention_dp):
        # Parse comms_medium to get use_nccl_for_alltoall and fifo_version.
        if comms_medium == "nccl":
            use_nccl_for_alltoall = True
            fifo_version = 2  # Not used when NCCL is enabled.
        elif comms_medium == "fifo_v1":
            use_nccl_for_alltoall = False
            fifo_version = 1
        elif comms_medium == "fifo_v2":
            use_nccl_for_alltoall = False
            fifo_version = 2
        else:
            raise ValueError(f"Unknown comms_medium: {comms_medium}")
        gen_ep = gen_tp * gen_cp
        kv_cache_config = {
            "free_gpu_memory_fraction": 0.5,
            "enable_block_reuse": False,
            "enable_partial_reuse": False,
            "tokens_per_block": 32,
        }
        ctx_server_config = {
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 4,
            "context_parallel_size": 1,
            "disable_overlap_scheduler": True,
            "kv_cache_config": kv_cache_config,
            "enable_chunked_prefill": False,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": "UCX",
                "max_tokens_in_buffer": 8192,
            },
        }
        gen_server_config = {
            "tensor_parallel_size": gen_tp,
            "pipeline_parallel_size": gen_pp,
            "context_parallel_size": gen_cp,
            "moe_expert_parallel_size": gen_ep,
            "cp_config": {
                "cp_type": "HELIX",
                "tokens_per_block": 32,
                "use_nccl_for_alltoall": use_nccl_for_alltoall,
                "fifo_version": fifo_version,
            },
            "disable_overlap_scheduler": True,
            "kv_cache_config": kv_cache_config,
            "enable_chunked_prefill": False,
            "cuda_graph_config": cuda_graph_config,
            "cache_transceiver_config": {
                "backend": "UCX",
                "max_tokens_in_buffer": 8192,
            },
            "enable_attention_dp": enable_attention_dp,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["MMLU", "GSM8K"])

    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_less_device_memory(60000)
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
    def test_guided_decoding(self, backend: str, mtp_nextn: int, mocker):
        mocker.patch.dict(os.environ, {"TRTLLM_XGUIDANCE_LENIENT": "1"})
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.8,
            },
            "guided_decoding_backend": backend,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            }
        }
        gen_server_config = {
            "disable_overlap_scheduler": False,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.8,
            },
            "guided_decoding_backend": backend,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            }
        }
        if mtp_nextn > 0:
            ctx_server_config["speculative_config"] = {
                "decoding_type": "MTP",
                "max_draft_len": mtp_nextn
            }
            gen_server_config["speculative_config"] = {
                "decoding_type": "MTP",
                "max_draft_len": mtp_nextn
            }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["JsonModeEval"])

    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_less_device_memory(60000)
    @skip_pre_hopper
    def test_kv_cache_v2_nixl_python(self):
        """Test with use_kv_cache_manager_v2=True, block_reuse=False, backend=NIXL, transceiver_runtime=PYTHON."""
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "kv_cache_config": {
                "enable_block_reuse": False,
                "use_kv_cache_manager_v2": True
            },
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON"
            }
        }
        gen_server_config = {
            "disable_overlap_scheduler": True,
            "kv_cache_config": {
                "enable_block_reuse": False,
                "use_kv_cache_manager_v2": True
            },
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON"
            }
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "port": 8000,
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1,
                "urls": ["localhost:8001"]
            },
            "generation_servers": {
                "num_instances": 1,
                "urls": ["localhost:8002"]
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])

    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(60000)
    @skip_pre_hopper
    @pytest.mark.parametrize(
        "enable_attention_dp,mtp_nextn",
        [(False, 0), (True, 2)],
        ids=["noadp-mtp0", "adp-mtp2"],
    )
    def test_gen_first(self, enable_attention_dp, mtp_nextn):
        """Gen-first MLA coverage on KVCacheManagerV2 + NIXL python; diagonal ADP/MTP combos."""
        kv_cache_config = {
            "enable_block_reuse": False,
            "use_kv_cache_manager_v2": True,
        }
        ctx_server_config = {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "enable_attention_dp": enable_attention_dp,
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON",
            },
            "kv_cache_config": kv_cache_config,
        }
        gen_server_config = {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "enable_attention_dp": enable_attention_dp,
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON",
            },
            "kv_cache_config": kv_cache_config,
        }
        if mtp_nextn > 0:
            spec_config = {
                "decoding_type": "MTP",
                "max_draft_len": mtp_nextn,
            }
            ctx_server_config["speculative_config"] = spec_config
            gen_server_config["speculative_config"] = spec_config
        disaggregated_server_config = {
            "hostname": "localhost",
            "port": 8000,
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1,
                "urls": ["localhost:8001"]
            },
            "generation_servers": {
                "num_instances": 1,
                "urls": ["localhost:8002"]
            },
            "schedule_style": "generation_first",
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])


@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
class TestGemma3_1BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "google/gemma-3-1b-it"
    MODEL_PATH = f"{llm_models_root()}/gemma/gemma-3-1b-it/"

    @pytest.mark.skip_less_device(2)
    @pytest.mark.parametrize("block_reuse", [False, True])
    @skip_pre_hopper
    def test_auto_dtype(self, block_reuse):

        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            }
        }
        gen_server_config = {
            "disable_overlap_scheduler": False,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            }
        }
        ctx_server_config["kv_cache_config"] = {
            "max_attention_window": [512, 512, 512, 512, 512, 32768],
            "enable_block_reuse": block_reuse,
            "enable_partial_reuse": block_reuse,
        }
        gen_server_config["kv_cache_config"] = {
            "max_attention_window": [512, 512, 512, 512, 512, 32768],
            "enable_block_reuse": block_reuse,
            "enable_partial_reuse": block_reuse,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["MMLU", "GSM8K"])

    @pytest.mark.skip_less_device(2)
    @pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True],
                             ids=["cache_mgr_v1", "cache_mgr_v2"])
    @skip_pre_hopper
    def test_kv_cache_v2_nixl_python(self, use_kv_cache_manager_v2):
        """Test with KV cache manager v1 and v2, block_reuse=False, backend=NIXL, transceiver_runtime=PYTHON."""
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "kv_cache_config": {
                "enable_block_reuse": False,
                "use_kv_cache_manager_v2": use_kv_cache_manager_v2
            },
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON"
            }
        }
        gen_server_config = {
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "kv_cache_config": {
                "enable_block_reuse": False,
                "use_kv_cache_manager_v2": use_kv_cache_manager_v2
            },
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON"
            }
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "port": 8000,
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1,
                "urls": ["localhost:8001"]
            },
            "generation_servers": {
                "num_instances": 1,
                "urls": ["localhost:8002"]
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["MMLU", "GSM8K"])


@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
class TestGPTOSS(LlmapiAccuracyTestHarness):
    extra_evaluator_kwargs = {
        "fewshot_as_multiturn": True,
        "apply_chat_template": True,
    }

    MODEL_PATH = f"{llm_models_root()}/gpt_oss/gpt-oss-120b"

    @pytest.mark.skip_less_device(8)
    @pytest.mark.parametrize("block_reuse", [False, True])
    def test_auto_dtype(self, block_reuse, mocker):
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", 8192)
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS,
                          {"scores_filter": "exact_match,flexible-extract"})
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON",
                "max_tokens_in_buffer": 4096
            },
            "tensor_parallel_size": 4
        }
        gen_server_config = {
            "disable_overlap_scheduler": False,
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON",
                "max_tokens_in_buffer": 4096
            },
            "tensor_parallel_size": 4
        }
        ctx_server_config["kv_cache_config"] = {
            "max_attention_window": [128, 32768],
            "enable_block_reuse": block_reuse,
            "enable_partial_reuse": block_reuse,
            "free_gpu_memory_fraction": 0.5,
        }
        gen_server_config["kv_cache_config"] = {
            "max_attention_window": [128, 32768],
            "enable_block_reuse": block_reuse,
            "enable_partial_reuse": block_reuse,
            "free_gpu_memory_fraction": 0.5,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            model_name = "GPT-OSS/120B-MXFP4"
            run_accuracy_test(
                llm,
                model_name,
                test_sets=["GSM8K"],
                extra_evaluator_kwargs={GSM8K: self.extra_evaluator_kwargs})

    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True],
                             ids=["cache_mgr_v1", "cache_mgr_v2"])
    def test_kv_cache_v2_nixl_python(self, use_kv_cache_manager_v2, mocker):
        """GPT-OSS disagg, NIXL Python transceiver (v2), KV cache manager v1 and v2 (ctx tp2 + gen tp2)."""
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", 8192)
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS,
                          {"scores_filter": "exact_match,flexible-extract"})
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON",
                "max_tokens_in_buffer": 4096
            },
            "tensor_parallel_size": 2,
            "kv_cache_config": {
                "max_attention_window": [128, 32768],
                "enable_block_reuse": False,
                "free_gpu_memory_fraction": 0.5,
                "use_kv_cache_manager_v2": use_kv_cache_manager_v2
            }
        }
        gen_server_config = {
            "disable_overlap_scheduler": False,
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON",
                "max_tokens_in_buffer": 4096
            },
            "tensor_parallel_size": 2,
            "kv_cache_config": {
                "max_attention_window": [128, 32768],
                "enable_block_reuse": False,
                "free_gpu_memory_fraction": 0.5,
                "use_kv_cache_manager_v2": use_kv_cache_manager_v2
            }
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            model_name = "GPT-OSS/120B-MXFP4"
            run_accuracy_test(
                llm,
                model_name,
                test_sets=["GSM8K"],
                extra_evaluator_kwargs={GSM8K: self.extra_evaluator_kwargs})


@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
@skip_pre_blackwell
class TestDeepSeekV32Exp(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-V3.2-Exp"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V3.2-Exp-FP4-v2"

    @pytest.mark.skip_less_device(8)
    @pytest.mark.parametrize("overlap_scheduler", [False])
    def test_auto_dtype(self, overlap_scheduler):
        cache_transceiver_config = {
            "backend": "DEFAULT",
            "max_tokens_in_buffer": 4096
        }
        max_num_tokens = 8192
        ctx_kv_cache_config = {
            "free_gpu_memory_fraction": 0.3,
            "tokens_per_block": 64,
            "dtype": "fp8",
        }
        moe_config = {"backend": "TRTLLM", "max_num_tokens": max_num_tokens}
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "cache_transceiver_config": cache_transceiver_config,
            "kv_cache_config": ctx_kv_cache_config,
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "max_batch_size": 16,
            "max_num_tokens": max_num_tokens,
            "enable_autotuner": False,
        }
        gen_kv_cache_config = {
            "free_gpu_memory_fraction": 0.5,
            "tokens_per_block": 64,
            "dtype": "fp8",
        }
        gen_server_config = {
            "disable_overlap_scheduler": overlap_scheduler,
            "cuda_graph_config": None,
            "cache_transceiver_config": cache_transceiver_config,
            "kv_cache_config": gen_kv_cache_config,
            "moe_config": moe_config,
            "max_batch_size": 128,
            "max_num_tokens": 1024,
            "cuda_graph_config": None,
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "moe_expert_parallel_size": 4,
            "enable_attention_dp": True,
            "enable_autotuner": False,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config=ctx_server_config,
                                      gen_server_config=gen_server_config,
                                      model_name=self.MODEL_PATH,
                                      max_workers=128) as llm:
            run_accuracy_test(llm,
                              model_name=self.MODEL_NAME,
                              test_sets=["MMLU", "GSM8K"])

    @skip_pre_blackwell
    @pytest.mark.skip_less_device(8)
    @pytest.mark.parametrize(
        "gen_pp,gen_tp,gen_cp,enable_attention_dp", [
            (1, 1, 4, False),
            (1, 2, 2, False),
            (1, 2, 2, True),
            (2, 1, 2, False),
        ],
        ids=["pp1tp1cp4", "pp1tp2cp2", "pp1dp2cp2", "pp2tp1cp2"])
    @pytest.mark.parametrize("cuda_graph_config", [
        None,
        {
            "enable_padding": True,
            "batch_sizes": [1, 2, 4, 8, 16, 32, 64],
        },
    ],
                             ids=[
                                 "cudagraph:none",
                                 "cudagraph:with_padding",
                             ])
    @pytest.mark.parametrize("comms_medium", ["fifo", "nccl"])
    def test_auto_dtype_with_helix(self, comms_medium, cuda_graph_config,
                                   gen_pp, gen_tp, gen_cp, enable_attention_dp):
        use_nccl_for_alltoall = comms_medium == "nccl"
        fifo_version = 2
        gen_ep = gen_tp * gen_cp
        kv_cache_config = {
            "free_gpu_memory_fraction": 0.5,
            "enable_block_reuse": False,
            "enable_partial_reuse": False,
            "tokens_per_block": 32,
            "dtype": "fp8",
        }
        ctx_server_config = {
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 4,
            "context_parallel_size": 1,
            "disable_overlap_scheduler": True,
            "kv_cache_config": kv_cache_config,
            "enable_chunked_prefill": False,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": "UCX",
                "max_tokens_in_buffer": 8192,
            },
            "moe_config": {
                "backend": "TRTLLM",
                "max_num_tokens": 16384,
            },
        }
        gen_server_config = {
            "tensor_parallel_size": gen_tp,
            "pipeline_parallel_size": gen_pp,
            "context_parallel_size": gen_cp,
            "moe_expert_parallel_size": gen_ep,
            "cp_config": {
                "cp_type": "HELIX",
                "tokens_per_block": 32,
                "use_nccl_for_alltoall": use_nccl_for_alltoall,
                "fifo_version": fifo_version,
            },
            "disable_overlap_scheduler": True,
            "kv_cache_config": kv_cache_config,
            "enable_chunked_prefill": False,
            "cuda_graph_config": cuda_graph_config,
            "cache_transceiver_config": {
                "backend": "UCX",
                "max_tokens_in_buffer": 8192,
            },
            "moe_config": {
                "backend": "TRTLLM",
                "max_num_tokens": 16384,
            },
            "enable_attention_dp": enable_attention_dp,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "port": 8000,
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1,
                "urls": ["localhost:8001"]
            },
            "generation_servers": {
                "num_instances": 1,
                "urls": ["localhost:8002"]
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config,
                                      gen_server_config,
                                      self.MODEL_PATH,
                                      max_workers=128) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])


@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
@skip_pre_hopper
class TestQwen3_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen3/Qwen3-8B"
    MODEL_PATH = f"{llm_models_root()}/Qwen3/Qwen3-8B-FP8"

    @pytest.mark.skip_less_device(2)
    def test_nixl_backend(self):
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": "NIXL",
                "max_tokens_in_buffer": 4096
            }
        }
        gen_server_config = {
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": "NIXL",
                "max_tokens_in_buffer": 4096
            }
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])

    @pytest.mark.skip_less_device(2)
    @pytest.mark.parametrize("overlap_scheduler", [False, True])
    @pytest.mark.parametrize("enable_partial_reuse", [True, False])
    def test_auto_dtype(self, overlap_scheduler, enable_partial_reuse):
        kv_cache_config = {
            "enable_block_reuse": True,
            "enable_partial_reuse": enable_partial_reuse,
        }
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            },
            "kv_cache_config": kv_cache_config,
        }
        gen_server_config = {
            "disable_overlap_scheduler": overlap_scheduler,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            },
            "kv_cache_config": kv_cache_config,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["MMLU", "GSM8K"])

    def _test_chunked_prefill_helper(self, *, ctx_pp: int):
        # bs=1 will stabilize the result, but the test will be much slower
        max_batch_size = 32

        kv_cache_config = {
            "enable_block_reuse": True,
        }

        ctx_server_config = {
            "pipeline_parallel_size": ctx_pp,
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": "UCX",
                "max_tokens_in_buffer": 4096
            },
            "enable_chunked_prefill": True,
            "max_num_tokens": 256,
            "max_batch_size": max_batch_size,
            "kv_cache_config": kv_cache_config,
        }
        gen_server_config = {
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": "UCX",
                "max_tokens_in_buffer": 4096
            },
            "max_batch_size": max_batch_size,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["MMLU", "GSM8K"])

    @pytest.mark.skip_less_device(2)
    def test_chunked_prefill(self):
        self._test_chunked_prefill_helper(ctx_pp=1)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device(8)
    @pytest.mark.parametrize(
        "gen_pp,gen_tp,gen_cp,enable_attention_dp", [
            (1, 1, 4, False),
            (1, 2, 2, False),
            (1, 2, 2, True),
            (2, 1, 2, False),
        ],
        ids=["pp1tp1cp4", "pp1tp2cp2", "pp1dp2cp2", "pp2tp1cp2"])
    @pytest.mark.parametrize("cuda_graph_config", [
        None,
        {
            "enable_padding": False,
            "batch_sizes": [1, 2, 4, 8, 16, 32, 64]
        },
        {
            "enable_padding": True,
            "batch_sizes": [1, 2, 4, 8, 16, 32, 64]
        },
    ],
                             ids=[
                                 "cudagraph:none", "cudagraph:without_padding",
                                 "cudagraph:with_padding"
                             ])
    @pytest.mark.parametrize("comms_medium", ["fifo_v1", "fifo_v2", "nccl"])
    def test_auto_dtype_with_helix(self, comms_medium, cuda_graph_config,
                                   gen_pp, gen_tp, gen_cp, enable_attention_dp):
        # Parse comms_medium to get use_nccl_for_alltoall and fifo_version.
        if comms_medium == "nccl":
            use_nccl_for_alltoall = True
            fifo_version = 2  # Not used when NCCL is enabled.
        elif comms_medium == "fifo_v1":
            use_nccl_for_alltoall = False
            fifo_version = 1
        elif comms_medium == "fifo_v2":
            use_nccl_for_alltoall = False
            fifo_version = 2
        else:
            raise ValueError(f"Unknown comms_medium: {comms_medium}")
        gen_ep = gen_tp * gen_cp
        kv_cache_config = {
            "free_gpu_memory_fraction": 0.5,
            "enable_block_reuse": False,
            "enable_partial_reuse": False,
            "tokens_per_block": 32,
        }
        cache_transceiver_config = {
            "backend": "UCX",
            "max_tokens_in_buffer": 8192,
        }
        ctx_server_config = {
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 4,
            "context_parallel_size": 1,
            "disable_overlap_scheduler": True,
            "kv_cache_config": kv_cache_config,
            "enable_chunked_prefill": False,
            "cuda_graph_config": None,
            "cache_transceiver_config": cache_transceiver_config.copy(),
        }
        gen_server_config = {
            "tensor_parallel_size": gen_tp,
            "pipeline_parallel_size": gen_pp,
            "context_parallel_size": gen_cp,
            "moe_expert_parallel_size": gen_ep,
            "cp_config": {
                "cp_type": "HELIX",
                "tokens_per_block": 32,
                "use_nccl_for_alltoall": use_nccl_for_alltoall,
                "fifo_version": fifo_version,
            },
            "disable_overlap_scheduler": True,
            "kv_cache_config": kv_cache_config,
            "enable_chunked_prefill": False,
            "cuda_graph_config": cuda_graph_config,
            "cache_transceiver_config": cache_transceiver_config.copy(),
            "enable_attention_dp": enable_attention_dp,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "port": 8000,
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1,
                "urls": ["localhost:8001"]
            },
            "generation_servers": {
                "num_instances": 1,
                "urls": ["localhost:8002"]
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])

    @pytest.mark.skip_less_device(2)
    def test_gen_first(self):
        """Gen-first dense-model smoke test on KVCacheManagerV2 + NIXL python."""
        kv_cache_config = {
            "enable_block_reuse": False,
            "use_kv_cache_manager_v2": True,
        }
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON",
            },
            "kv_cache_config": kv_cache_config,
        }
        gen_server_config = {
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON",
            },
            "kv_cache_config": kv_cache_config,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "port": 8000,
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1,
                "urls": ["localhost:8001"]
            },
            "generation_servers": {
                "num_instances": 1,
                "urls": ["localhost:8002"]
            },
            "schedule_style": "generation_first",
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["MMLU"])

    @pytest.mark.skip_less_device(2)
    def test_gen_first_kv_cache_v1(self):
        """Gen-first smoke test on the legacy V1 KV cache manager with block reuse."""
        transceiver_runtime = "PYTHON"
        transceiver_backend = "NIXL"
        kv_cache_config = {
            "enable_block_reuse": True,
            "enable_partial_reuse": False,
            "use_kv_cache_manager_v2": False,
        }
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": transceiver_backend,
                "transceiver_runtime": transceiver_runtime,
            },
            "kv_cache_config": kv_cache_config,
        }
        gen_server_config = {
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": transceiver_backend,
                "transceiver_runtime": transceiver_runtime,
            },
            "kv_cache_config": kv_cache_config,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "port": 8000,
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1,
                "urls": ["localhost:8001"]
            },
            "generation_servers": {
                "num_instances": 1,
                "urls": ["localhost:8002"]
            },
            "schedule_style": "generation_first",
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["MMLU"])


@skip_pre_blackwell
@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
class TestQwen3_30B_A3B(LlmapiAccuracyTestHarness):
    FP4_MODEL = f"{llm_models_root()}/Qwen3/saved_models_Qwen3-30B-A3B_nvfp4_hf"
    FP8_MODEL = f"{llm_models_root()}/Qwen3/saved_models_Qwen3-30B-A3B_fp8_hf"

    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize("ctx_pp,gen_tp", [(2, 2)], ids=["ctxpp2gentp2"])
    def test_mixed_ctx_gen_model(self, ctx_pp, gen_tp):
        ctx_model = self.FP4_MODEL
        gen_model = self.FP8_MODEL
        return run_parallel_test("Qwen3/Qwen3-30B-A3B",
                                 ctx_model,
                                 ctx_pp=ctx_pp,
                                 ctx_tp=1,
                                 gen_pp=1,
                                 gen_tp=gen_tp,
                                 test_sets=[GSM8K, MMLU],
                                 ctx_model=ctx_model,
                                 gen_model=gen_model,
                                 ctx_instances=1,
                                 gen_instances=1)


@pytest.mark.timeout(10800)
@skip_pre_blackwell
class TestKimiK2(LlmapiAccuracyTestHarness):
    MODEL_NAME = "moonshotai/Kimi-K2-Thinking"
    MODEL_PATH = f"{llm_models_root()}/Kimi-K2-Thinking-NVFP4"

    @pytest.mark.skip_less_device(8)
    @pytest.mark.skip_less_device_memory(200000)
    def test_nvfp4(self):
        ctx_server_config = {
            "max_batch_size": 16,
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            },
            "tensor_parallel_size": 4,
            "enable_attention_dp": True,
            "trust_remote_code": True,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.8,
            },
        }
        gen_server_config = {
            "max_batch_size": 16,
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            },
            "tensor_parallel_size": 4,
            "enable_attention_dp": True,
            "trust_remote_code": True,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.8,
            },
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])


@pytest.mark.timeout(10800)
@skip_pre_blackwell
class TestKimiK25(LlmapiAccuracyTestHarness):
    MODEL_NAME = "moonshotai/Kimi-K2.5"
    MODEL_PATH = f"{llm_models_root()}/Kimi-K2.5-NVFP4"

    @pytest.mark.skip_less_device(8)
    @pytest.mark.skip_less_device_memory(180000)
    def test_nvfp4(self):
        """Disaggregated GSM8K accuracy for Kimi-K2.5 (NVFP4).

        ctx and gen servers are each TP4 (8 GPUs total) over the default cache
        transceiver. GSM8K is text-only, so requests run through the DeepSeek-V3
        MLA backbone (no vision) and the ctx->gen KV transfer is the MLA latent.
        Kimi-K2.5 ships custom HF modeling code (auto_map in config.json), so
        trust_remote_code must be set on both servers or executor init fails at
        config parse time.

        Kimi-K2.5 has a ~256k default context. Without an explicit cap the
        context server tries to allocate a ~234k-token KV window and stalls in
        warmup before it registers as a disagg worker (the generation worker
        registers, the context worker never does, so the cluster never reports
        is_ready and the test hangs). GSM8K prompts are short, so cap the
        sequence/token budget to keep startup fast and the KV pool small.
        """
        ctx_server_config = {
            "max_batch_size": 16,
            "max_seq_len": 8192,
            "max_num_tokens": 8192,
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            },
            "tensor_parallel_size": 4,
            "enable_attention_dp": True,
            "trust_remote_code": True,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.6,
            },
        }
        gen_server_config = {
            "max_batch_size": 16,
            "max_seq_len": 8192,
            "max_num_tokens": 8192,
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": "DEFAULT",
                "max_tokens_in_buffer": 4096
            },
            "tensor_parallel_size": 4,
            "enable_attention_dp": True,
            "trust_remote_code": True,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.6,
            },
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])


@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
class TestNemotron3Super120B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Nemotron-Super-V3"
    MODEL_PATH = f"{llm_models_root()}/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"

    def _make_configs(self, use_py_transceiver: bool = False):
        cache_transceiver_config = {
            "backend": "NIXL",
            "max_tokens_in_buffer": 8192,
        }
        if use_py_transceiver:
            cache_transceiver_config["transceiver_runtime"] = "PYTHON"

        ctx_server_config = {
            "max_batch_size": 32,
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": cache_transceiver_config,
            "tensor_parallel_size": 4,
            "moe_expert_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "kv_cache_config": {
                "enable_block_reuse": False,
                "mamba_ssm_cache_dtype": "float16",
                "free_gpu_memory_fraction": 0.5,
            },
            "moe_config": {
                "backend": "CUTLASS"
            }
        }

        gen_server_config = {
            "max_batch_size": 32,
            "disable_overlap_scheduler": False,
            "cache_transceiver_config": cache_transceiver_config,
            "tensor_parallel_size": 4,
            "moe_expert_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "cuda_graph_config": {
                "max_batch_size": 32,
                "enable_padding": True,
            },
            "kv_cache_config": {
                "enable_block_reuse": False,
                "mamba_ssm_cache_dtype": "float16",
                "free_gpu_memory_fraction": 0.5,
            },
            "moe_config": {
                "backend": "CUTLASS"
            }
        }

        disaggregated_server_config = {
            "hostname": "localhost",
            "port": 8000,
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1,
                "urls": ["localhost:8001"]
            },
            "generation_servers": {
                "num_instances": 1,
                "urls": ["localhost:8002"]
            }
        }
        return ctx_server_config, gen_server_config, disaggregated_server_config

    @pytest.mark.skip_less_device(8)
    @parametrize_with_ids("use_py_transceiver", [True, False])
    @parametrize_with_ids("block_reuse", [True, False])
    @parametrize_with_ids("mtp_nextn", [0, 1, 3])
    def test_auto_dtype(self, use_py_transceiver, block_reuse, mtp_nextn):
        if use_py_transceiver and block_reuse:
            pytest.skip("Python transceiver does not support block reuse")

        ctx_cfg, gen_cfg, disagg_cfg = self._make_configs(use_py_transceiver)
        if mtp_nextn > 0:
            spec = {"decoding_type": "MTP", "max_draft_len": mtp_nextn}
            ctx_cfg["speculative_config"] = spec
            gen_cfg["speculative_config"] = spec
        if block_reuse:
            ctx_cfg["kv_cache_config"]["enable_block_reuse"] = True
            gen_cfg["kv_cache_config"]["enable_block_reuse"] = True
        with launch_disaggregated_llm(disagg_cfg, ctx_cfg, gen_cfg,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])

    @pytest.mark.skip_less_device(8)
    def test_ctx_dp2_gen_tp4(self):
        ctx_cfg, gen_cfg, disagg_cfg = self._make_configs(
            use_py_transceiver=False)
        # corner case: max_batch_size = 1 + dp for ctx to check if dp dummy requests are handled correctly
        ctx_cfg["max_batch_size"] = 1
        ctx_cfg["enable_attention_dp"] = True
        ctx_cfg["tensor_parallel_size"] = 2
        ctx_cfg["moe_expert_parallel_size"] = 2
        gen_cfg["tensor_parallel_size"] = 4
        gen_cfg["moe_expert_parallel_size"] = 4
        gen_cfg["pipeline_parallel_size"] = 1
        with launch_disaggregated_llm(disagg_cfg, ctx_cfg, gen_cfg,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])


@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
class TestQwen3NextInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen3/Qwen3-Next-80B-A3B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen3-Next/Qwen3-Next-80B-A3B-Instruct"

    def _make_configs(self, use_py_transceiver: bool):
        cache_transceiver_config = {
            "backend": "NIXL",
            "max_tokens_in_buffer": 8192,
        }
        if use_py_transceiver:
            cache_transceiver_config["transceiver_runtime"] = "PYTHON"

        ctx_server_config = {
            "max_batch_size": 32,
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": cache_transceiver_config,
            "tensor_parallel_size": 4,
            "moe_expert_parallel_size": 4,
            "kv_cache_config": {
                "enable_block_reuse": False,
                "mamba_ssm_cache_dtype": "float16",
                "free_gpu_memory_fraction": 0.5,
            },
            "moe_config": {
                "backend": "CUTLASS"
            }
        }

        gen_server_config = {
            "max_batch_size": 32,
            "disable_overlap_scheduler": False,
            "cache_transceiver_config": cache_transceiver_config,
            "tensor_parallel_size": 2,
            "moe_expert_parallel_size": 2,
            "pipeline_parallel_size": 2,
            "cuda_graph_config": {
                "max_batch_size": 32,
                "enable_padding": True,
            },
            "kv_cache_config": {
                "enable_block_reuse": False,
                "mamba_ssm_cache_dtype": "float16",
                "free_gpu_memory_fraction": 0.5,
            },
            "moe_config": {
                "backend": "CUTLASS"
            }
        }

        disaggregated_server_config = {
            "hostname": "localhost",
            "port": 8000,
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1,
                "urls": ["localhost:8001"]
            },
            "generation_servers": {
                "num_instances": 1,
                "urls": ["localhost:8002"]
            }
        }
        return ctx_server_config, gen_server_config, disaggregated_server_config

    @pytest.mark.skip_less_device(8)
    @parametrize_with_ids("use_py_transceiver", [True, False])
    def test_auto_dtype(self, use_py_transceiver, mocker):
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", 512)
        ctx_cfg, gen_cfg, disagg_cfg = self._make_configs(use_py_transceiver)
        with launch_disaggregated_llm(disagg_cfg, ctx_cfg, gen_cfg,
                                      self.MODEL_PATH) as llm:
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])


@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
class TestGLM52NVFP4(LlmapiAccuracyTestHarness):
    MODEL_NAME = "zai-org/GLM-5.2"
    MODEL_PATH = f"{llm_models_root()}/GLM-5.2-NVFP4"

    @pytest.mark.skip_less_device(8)
    @pytest.mark.parametrize("use_kv_cache_manager_v2", [False],
                             ids=["cache_mgr_v1"])
    def test_nvfp4_nixl_python(self, use_kv_cache_manager_v2):
        kv_cache_config = {
            "free_gpu_memory_fraction": 0.7,
            "enable_block_reuse": False,
            "use_kv_cache_manager_v2": use_kv_cache_manager_v2,
        }
        cache_transceiver_config = {
            "backend": "NIXL",
            "transceiver_runtime": "PYTHON",
        }
        moe_config = {"backend": "CUTEDSL"}
        speculative_config = {
            "decoding_type": "MTP",
            "max_draft_len": 1,
        }
        ctx_server_config = {
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "moe_expert_parallel_size": 4,
            "enable_attention_dp": True,
            "disable_overlap_scheduler": True,
            "enable_chunked_prefill": True,
            "cuda_graph_config": None,
            "trust_remote_code": True,
            "max_seq_len": 8192,
            "kv_cache_config": kv_cache_config,
            "moe_config": moe_config,
            "speculative_config": speculative_config,
            "cache_transceiver_config": cache_transceiver_config,
        }
        gen_server_config = {
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "moe_expert_parallel_size": 4,
            "enable_attention_dp": True,
            "disable_overlap_scheduler": False,
            "enable_chunked_prefill": True,
            "trust_remote_code": True,
            "max_seq_len": 8192,
            "kv_cache_config": kv_cache_config,
            "moe_config": moe_config,
            "speculative_config": speculative_config,
            "cache_transceiver_config": cache_transceiver_config,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config,
                                      gen_server_config,
                                      self.MODEL_PATH,
                                      max_workers=128) as llm:
            # launch_disaggregated_llm builds a bare LlmArgs for the DuckLLM,
            # so the specs used for the accuracy reference lookup must be
            # filled in to match the registered entry (NVFP4 + FP8 KV cache
            # + MTP).
            llm.args.quant_config.kv_cache_quant_algo = "FP8"
            llm.args.speculative_config = MTPDecodingConfig(
                max_draft_len=speculative_config["max_draft_len"])
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])


@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
@skip_pre_blackwell
class TestDeepSeekV4Flash(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V4-Flash"

    @pytest.mark.skip_less_device(4)
    def test_auto_dtype(self):
        # Disagg smoke test: CTX TP=2 + GEN TP=2 = 4 GPUs.
        # NVFP4 weights ~71 GB/rank at TP=2, leaving ~107 GB for KV on B200.
        # TRTLLM backend required (WIDEEP lacks MXFP4 support for V4-Flash).
        # V4 uses pure-Python KVCacheManagerV2; needs Python transceiver.
        # NIXL (not DEFAULT) skips the TRTLLM_USE_UCX_KVCACHE=1 fallback.
        cache_transceiver_config = {
            "backend": "NIXL",
            "transceiver_runtime": "PYTHON",
            "max_tokens_in_buffer": 4096,
        }
        ctx_server_config = {
            "tensor_parallel_size": 2,
            "moe_expert_parallel_size": 2,
            "disable_overlap_scheduler": True,
            "max_batch_size": DEEPSEEKV4_TEST_MAX_BATCH_SIZE,
            "max_seq_len": 4096,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.5,
            },
            "cache_transceiver_config": cache_transceiver_config,
        }
        gen_server_config = {
            "tensor_parallel_size": 2,
            "moe_expert_parallel_size": 2,
            "enable_attention_dp": True,
            "disable_overlap_scheduler": True,
            "max_batch_size": DEEPSEEKV4_TEST_MAX_BATCH_SIZE,
            "max_seq_len": 4096,
            "moe_config": {
                "backend": "TRTLLM",
            },
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.5,
            },
            "cache_transceiver_config": cache_transceiver_config,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            },
        }
        # V4-Flash 148GB weight prefetch + warmup needs >35 min, default wait timeout times out.
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config,
                                      gen_server_config,
                                      self.MODEL_PATH,
                                      server_waiting_timeout=3600) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, is_integration_test=True)

    @pytest.mark.skip_less_device(4)
    def test_gen_first(self):
        """Gen-first quick validation for DSv4-Flash on KVCacheManagerV2 + NIXL python."""
        cache_transceiver_config = {
            "backend": "NIXL",
            "transceiver_runtime": "PYTHON",
            "max_tokens_in_buffer": 4096,
        }
        ctx_server_config = {
            "tensor_parallel_size": 2,
            "moe_expert_parallel_size": 2,
            "disable_overlap_scheduler": True,
            "max_batch_size": DEEPSEEKV4_TEST_MAX_BATCH_SIZE,
            "max_seq_len": 4096,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.5,
            },
            "cache_transceiver_config": cache_transceiver_config,
        }
        gen_server_config = {
            "tensor_parallel_size": 2,
            "moe_expert_parallel_size": 2,
            "enable_attention_dp": True,
            "disable_overlap_scheduler": True,
            "max_batch_size": DEEPSEEKV4_TEST_MAX_BATCH_SIZE,
            "max_seq_len": 4096,
            "moe_config": {
                "backend": "TRTLLM",
            },
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.5,
            },
            "cache_transceiver_config": cache_transceiver_config,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            },
            "schedule_style": "generation_first",
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config,
                                      gen_server_config,
                                      self.MODEL_PATH,
                                      server_waiting_timeout=3600) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, is_integration_test=True)


@pytest.mark.timeout(14400)
@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(140000)
class TestDeepSeekV4FlashBase(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash-Base"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V4-Flash-Base"

    @pytest.mark.skip_less_device(4)
    def test_auto_dtype(self):
        # Disagg smoke test: CTX TP=2 + GEN TP=2 = 4 GPUs.
        # FP8 weights ~71 GB/rank at TP=4 → ~142 GB/rank at TP=2; requires
        # ≥140 GB per GPU (fits on B300 288 GB, tight on B200 178 GB).
        # TRTLLM backend: WIDEEP's FP8 block-scale path is Hopper-only.
        # Compact batching keeps KV cache ~1 GB/rank (default ~100 GB requires fully-clean GPU memory).
        # V4 uses pure-Python KVCacheManagerV2; needs Python transceiver.
        # NIXL (not DEFAULT) skips the TRTLLM_USE_UCX_KVCACHE=1 fallback.
        cache_transceiver_config = {
            "backend": "NIXL",
            "transceiver_runtime": "PYTHON",
            "max_tokens_in_buffer": 4096,
        }
        ctx_server_config = {
            "tensor_parallel_size": 2,
            "moe_expert_parallel_size": 2,
            "disable_overlap_scheduler": True,
            "max_batch_size": 16,
            "max_num_tokens": 4096,
            "max_seq_len": 4096,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.5,
            },
            "cache_transceiver_config": cache_transceiver_config,
        }
        gen_server_config = {
            "tensor_parallel_size": 2,
            "moe_expert_parallel_size": 2,
            "enable_attention_dp": True,
            "disable_overlap_scheduler": True,
            "max_batch_size": DEEPSEEKV4_TEST_MAX_BATCH_SIZE,
            "max_num_tokens": 4096,
            "max_seq_len": 4096,
            "moe_config": {
                "backend": "TRTLLM",
            },
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.5,
            },
            "cache_transceiver_config": cache_transceiver_config,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1
            },
            "generation_servers": {
                "num_instances": 1
            },
        }
        # Same long-init reason as TestDeepSeekV4Flash above.
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config,
                                      gen_server_config,
                                      self.MODEL_PATH,
                                      server_waiting_timeout=3600) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, is_integration_test=True)
