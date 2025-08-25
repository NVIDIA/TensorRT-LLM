# I want to create accuracy tests for disaggregated serving.
# I need to to this by creating a new class that mimics LLM class. Instead of implementing the
# actual methods it will send OAI requests to the disaggregated serving endpoint.
# Please take a look at the existing test_llm_api_pytorch.py file for reference.
import concurrent
import contextlib
import os
import tempfile
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import openai
import pytest
import requests
import yaml

from tensorrt_llm.executor.result import GenerationResultBase
from tensorrt_llm.llmapi import CompletionOutput, RequestOutput, SamplingParams
from tensorrt_llm.llmapi.llm_args import LlmArgs

from ..conftest import (get_device_count, llm_models_root, parametrize_with_ids,
                        skip_no_hopper, skip_pre_hopper)
from ..trt_test_alternative import popen
from .accuracy_core import (GSM8K, MMLU, LlmapiAccuracyTestHarness,
                            get_accuracy_task)


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


DuckLLM = namedtuple('DuckLLM', ['args', 'generate_async'])


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


@contextlib.contextmanager
def launch_disaggregated_llm(disaggregated_server_config: Dict[str, Any],
                             ctx_server_config: Dict[str, Any],
                             gen_server_config: Dict[str, Any],
                             model_name: str,
                             tensor_parallel_size: int = 1):
    temp_dir = tempfile.TemporaryDirectory()
    disaggregated_serving_config_path = os.path.join(
        temp_dir.name, "disaggregated_serving_config.yaml")

    if tensor_parallel_size > 1:
        print(
            f"Using unified tp parameter for testing is not recommended. Please use server configs instead."
        )

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

    args = LlmArgs.from_kwargs(model=model_name,
                               tensor_parallel_size=tensor_parallel_size)

    trtllm_serve_path = "trtllm-serve"
    # Common arguments for both servers
    common_args = [
        trtllm_serve_path,
        model_name,
        "--host",
        "localhost",
        "--backend",
        "pytorch",
    ]
    gen_tp, gen_pp = gen_server_config.get(
        "tensor_parallel_size",
        tensor_parallel_size), gen_server_config.get("pipeline_parallel_size",
                                                     1)
    ctx_tp, ctx_pp = ctx_server_config.get(
        "tensor_parallel_size",
        tensor_parallel_size), ctx_server_config.get("pipeline_parallel_size",
                                                     1)

    ctx_total_gpus = ctx_tp * ctx_pp
    gen_total_gpus = gen_tp * gen_pp

    ctx_urls = disaggregated_server_config["context_servers"]["urls"]
    gen_urls = disaggregated_server_config["generation_servers"]["urls"]

    ctx_ports = [int(url.split(":")[1]) for url in ctx_urls]
    gen_ports = [int(url.split(":")[1]) for url in gen_urls]

    ctx_servers = []
    current_gpu_offset = 0

    for i, port in enumerate(ctx_ports):
        env_ctx = os.environ.copy()
        env_ctx["TRTLLM_USE_UCX_KVCACHE"] = "1"
        gpu_range = range(current_gpu_offset,
                          current_gpu_offset + ctx_total_gpus)
        env_ctx["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_range))
        current_gpu_offset += ctx_total_gpus

        ctx_server_args = common_args + [
            "--port",
            str(port), "--extra_llm_api_options", ctx_server_config_path,
            f"--tp_size={ctx_tp}", f"--pp_size={ctx_pp}"
        ]
        if "max_num_tokens" in ctx_server_config:
            ctx_server_args.append(
                f"--max_num_tokens={ctx_server_config['max_num_tokens']}")

        ctx_servers.append((env_ctx, ctx_server_args))

    gen_servers = []

    for i, port in enumerate(gen_ports):
        env_gen = os.environ.copy()
        env_gen["TRTLLM_USE_UCX_KVCACHE"] = "1"
        gpu_range = range(current_gpu_offset,
                          current_gpu_offset + gen_total_gpus)
        env_gen["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_range))
        current_gpu_offset += gen_total_gpus

        gen_server_args = common_args + [
            "--port",
            str(port), "--extra_llm_api_options", gen_server_config_path,
            f"--tp_size={gen_tp}", f"--pp_size={gen_pp}"
        ]
        if "max_num_tokens" in gen_server_config:
            gen_server_args.append(
                f"--max_num_tokens={gen_server_config['max_num_tokens']}")

        gen_servers.append((env_gen, gen_server_args))

    @contextlib.contextmanager
    def multi_popen(server_configs):
        processes = []
        try:
            for env, args in server_configs:
                proc = popen(args, env=env)
                processes.append(proc)

            with contextlib.ExitStack() as stack:
                opened_processes = [
                    stack.enter_context(proc) for proc in processes
                ]
                yield opened_processes
        except Exception as e:
            print(
                f"Failed to start disaggregated server processes in multi_popen: {e}"
            )
            raise

    with (MyThreadPoolExecutor(max_workers=16) as thread_pool, temp_dir):
        with multi_popen(ctx_servers + gen_servers):
            with popen([
                    trtllm_serve_path, "disaggregated", "-c",
                    disaggregated_serving_config_path, "--server_start_timeout",
                    "3600"
            ]):
                while True:
                    time.sleep(1)
                    try:
                        print("Checking health endpoint")
                        response = requests.get("http://localhost:8000/health")
                        if response.status_code == 200:
                            break
                    except requests.exceptions.ConnectionError:
                        continue

                client = openai.OpenAI(api_key="1234567890",
                                       base_url=f"http://localhost:8000/v1")

                def send_request(prompt: str, sampling_params: SamplingParams,
                                 streaming: bool):
                    response = client.completions.create(
                        model=model_name,
                        prompt=prompt,
                        stream=streaming,
                        **({
                            "max_tokens": sampling_params.max_tokens,
                            "temperature": sampling_params.temperature,
                            "top_p": sampling_params.top_p,
                            "stop": sampling_params.stop,
                            "seed": sampling_params.seed
                        } if sampling_params else {}))
                    result = Result(id=0,
                                    sampling_params=sampling_params,
                                    outputs=[
                                        CompletionOutput(
                                            text=response.choices[0].text,
                                            index=0)
                                    ])
                    requested_output = RequestOutput._from_generation_result(
                        result, prompt=prompt)
                    setattr(requested_output, "result", result.result)
                    return requested_output

                def generate_async(
                        prompt: str,
                        sampling_params: Optional[SamplingParams] = None,
                        streaming: bool = False):
                    future = thread_pool.submit(send_request, prompt,
                                                sampling_params, streaming)
                    thread_pool.futures.append(future)
                    return future

                yield DuckLLM(args, generate_async)


def run_parallel_test(model_name: str, model_path: str, ctx_pp: int,
                      ctx_tp: int, gen_pp: int, gen_tp: int, ctx_instances: int,
                      gen_instances: int, test_set: LlmapiAccuracyTestHarness):
    total_ctx_gpus = ctx_tp * ctx_pp * ctx_instances
    total_gen_gpus = gen_tp * gen_pp * gen_instances
    if total_ctx_gpus + total_gen_gpus > get_device_count():
        pytest.skip(
            f"Not enough devices for {ctx_instances} ctx instances (ctx_pp={ctx_pp}*ctx_tp={ctx_tp}) + {gen_instances} gen instances (gen_pp={gen_pp}*gen_tp={gen_tp}), total: {total_ctx_gpus + total_gen_gpus}"
        )

    kv_cache_config = {
        "free_gpu_memory_fraction": 0.5,
        "enable_block_reuse": False
    }
    ctx_server_config = {
        "pipeline_parallel_size": ctx_pp,
        "tensor_parallel_size": ctx_tp,
        "disable_overlap_scheduler": True,
        "kv_cache_config": kv_cache_config,
        "cache_transceiver_config": {
            "backend": "DEFAULT"
        }
    }
    gen_server_config = {
        "tensor_parallel_size": gen_tp,
        "pipeline_parallel_size": gen_pp,
        "disable_overlap_scheduler": True,
        "kv_cache_config": kv_cache_config,
        "cache_transceiver_config": {
            "backend": "DEFAULT"
        }
    }

    ctx_urls = [f"localhost:{8001 + i * 2}" for i in range(ctx_instances)]
    gen_urls = [f"localhost:{8002 + i * 2}" for i in range(gen_instances)]

    disaggregated_server_config = {
        "hostname": "localhost",
        "port": 8000,
        "backend": "pytorch",
        "context_servers": {
            "num_instances": ctx_instances,
            "urls": ctx_urls
        },
        "generation_servers": {
            "num_instances": gen_instances,
            "urls": gen_urls
        }
    }
    with launch_disaggregated_llm(disaggregated_server_config,
                                  ctx_server_config, gen_server_config,
                                  model_path) as llm:
        task = test_set(model_name)
        task.evaluate(llm)


@pytest.mark.timeout(3600)
class TestLlama3_1_8BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"

    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.skip_less_device(2)
    @pytest.mark.parametrize("disable_overlap_scheduler", [False, True])
    def test_auto_dtype(self, disable_overlap_scheduler):
        ctx_server_config = {"disable_overlap_scheduler": True}
        gen_server_config = {
            "disable_overlap_scheduler": disable_overlap_scheduler
        }
        ctx_server_config["cache_transceiver_config"] = {"backend": "DEFAULT"}
        gen_server_config["cache_transceiver_config"] = {"backend": "DEFAULT"}
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
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(2)
    @skip_pre_hopper
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
                "backend": "DEFAULT"
            }
        }
        gen_server_config = {
            "disable_overlap_scheduler": True,
            "speculative_config": speculative_decoding_config,
            "kv_cache_config": kv_cache_config,
            "cache_transceiver_config": {
                "backend": "DEFAULT"
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
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @parametrize_with_ids("overlap_scheduler", [True, False])
    @parametrize_with_ids("eagle3_one_model", [True, False])
    @pytest.mark.skip_less_device(2)
    def test_eagle3(self, overlap_scheduler, eagle3_one_model):
        speculative_decoding_config = {
            "decoding_type": "Eagle",
            "max_draft_len": 4,
            "speculative_model_dir":
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
                "backend": "DEFAULT"
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
                "backend": "DEFAULT"
            },
            "cuda_graph_config": None,
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
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.parametrize("tp,pp", [(1, 2), (2, 1), (2, 2)],
                             ids=["tp1pp2", "tp2pp1", "tp2pp2"])
    @pytest.mark.parametrize("testset", ["GSM8K", "MMLU"])
    def test_tp_pp_symmetric(self, tp, pp, testset):
        return run_parallel_test(self.MODEL_NAME, self.MODEL_PATH, pp, tp, pp,
                                 tp, 1, 1, get_accuracy_task(testset))

    @parametrize_with_ids("ctx_pp", [2, 4])
    @parametrize_with_ids("gen_tp", [1, 2])
    @pytest.mark.parametrize("testset", ["GSM8K", "MMLU"])
    def test_ctx_pp_gen_tp_asymmetric(self, ctx_pp, gen_tp, testset):
        return run_parallel_test(self.MODEL_NAME, self.MODEL_PATH, ctx_pp, 1, 1,
                                 gen_tp, 1, 1, get_accuracy_task(testset))

    @pytest.mark.parametrize("testset", ["GSM8K", "MMLU"])
    def test_multi_instance(self, testset):
        return run_parallel_test(self.MODEL_NAME, self.MODEL_PATH, 1, 1, 1, 1,
                                 2, 2, get_accuracy_task(testset))


class TestLlama4ScoutInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama4-models/Llama-4-Scout-17B-16E-Instruct"

    @pytest.mark.skip_less_device_memory(140000)
    @pytest.mark.timeout(3600)
    @pytest.mark.skip_less_device(8)
    @pytest.mark.parametrize("overlap_scheduler", [False, True])
    def test_auto_dtype(self, overlap_scheduler):
        ctx_server_config = {"disable_overlap_scheduler": True}
        gen_server_config = {"disable_overlap_scheduler": overlap_scheduler}
        ctx_server_config["cache_transceiver_config"] = {"backend": "DEFAULT"}
        gen_server_config["cache_transceiver_config"] = {"backend": "DEFAULT"}
        # Keep this low to avoid warmup OOM in CI
        ctx_server_config["max_seq_len"] = 8192
        gen_server_config["max_seq_len"] = 8192
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
                                      tensor_parallel_size=4) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


@pytest.mark.timeout(3600)
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
                "backend": "NIXL"
            }
        }
        gen_server_config = {
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": "NIXL"
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
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @parametrize_with_ids("overlap_scheduler", [True, False])
    @parametrize_with_ids("mtp_nextn",
                          [0, pytest.param(2, marks=skip_pre_hopper)])
    @pytest.mark.skip_less_device(8)
    def test_auto_dtype(self, overlap_scheduler, mtp_nextn):
        ctx_server_config = {"disable_overlap_scheduler": True}
        gen_server_config = {"disable_overlap_scheduler": not overlap_scheduler}
        ctx_server_config["cache_transceiver_config"] = {"backend": "DEFAULT"}
        gen_server_config["cache_transceiver_config"] = {"backend": "DEFAULT"}
        if mtp_nextn > 0:
            ctx_server_config["speculative_config"] = {
                "decoding_type": "MTP",
                "num_nextn_predict_layers": mtp_nextn
            }
            gen_server_config["speculative_config"] = {
                "decoding_type": "MTP",
                "num_nextn_predict_layers": mtp_nextn
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
                                      tensor_parallel_size=4) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


@pytest.mark.timeout(3600)
class TestGemma3_1BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "google/gemma-3-1b-it"
    MODEL_PATH = f"{llm_models_root()}/gemma/gemma-3-1b-it/"

    @pytest.mark.skip_less_device(2)
    @pytest.mark.parametrize("overlap_scheduler", [False, True])
    def test_auto_dtype(self, overlap_scheduler):
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": "DEFAULT"
            }
        }
        gen_server_config = {
            "disable_overlap_scheduler": overlap_scheduler,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": "DEFAULT"
            }
        }
        ctx_server_config["kv_cache_config"] = {
            # "max_attention_window": [512, 512, 512, 512, 512, 32768],
            "enable_block_reuse": True
        }
        gen_server_config["kv_cache_config"] = {
            # "max_attention_window": [512, 512, 512, 512, 512, 32768],
            "enable_block_reuse": True
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
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


@pytest.mark.timeout(3600)
class TestQwen3_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen3/Qwen3-8B"
    MODEL_PATH = f"{llm_models_root()}/Qwen3/Qwen3-8B-FP8"

    @pytest.mark.skip_less_device(2)
    @skip_no_hopper
    def test_nixl_backend(self):
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": "NIXL"
            }
        }
        gen_server_config = {
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {
                "backend": "NIXL"
            }
        }
        ctx_server_config["cache_transceiver_config"]
        ctx_server_config["cache_transceiver_config"]
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
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @pytest.mark.skip_less_device(2)
    @pytest.mark.parametrize("overlap_scheduler", [False, True])
    def test_auto_dtype(self, overlap_scheduler):
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": "DEFAULT"
            }
        }
        gen_server_config = {
            "disable_overlap_scheduler": overlap_scheduler,
            "cuda_graph_config": None,
            "cache_transceiver_config": {
                "backend": "DEFAULT"
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
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
