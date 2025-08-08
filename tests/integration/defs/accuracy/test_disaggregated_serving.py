# I want to create accuracy tests for disaggregated serving.
# I need to to this by creating a new class that mimics LLM class. Instead of implementing the
# actual methods it will send OAI requests to the disaggregated serving endpoint.
# Please take a look at the existing test_llm_api_pytorch.py file for reference.
import concurrent
import contextlib
import json
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
from tensorrt_llm.llmapi.tokenizer import load_hf_tokenizer

from ..conftest import (get_device_count, llm_models_root, parametrize_with_ids,
                        skip_pre_blackwell, skip_pre_hopper)
from ..trt_test_alternative import popen
from .accuracy_core import (GSM8K, MMLU, JsonModeEval,
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
                             tensor_parallel_size: int = 1,
                             ctx_model: str = None,
                             gen_model: str = None):
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

    env_ctx = os.environ.copy()
    env_ctx["TRTLLM_USE_UCX_KVCACHE"] = "1"
    env_ctx["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(ctx_total_gpus)))

    env_gen = os.environ.copy()
    env_gen["TRTLLM_USE_UCX_KVCACHE"] = "1"
    env_gen["CUDA_VISIBLE_DEVICES"] = ",".join(
        map(str, range(ctx_total_gpus, ctx_total_gpus + gen_total_gpus)))
    ctx_server_args = ctx_args + [
        "--port", "8001", "--extra_llm_api_options", ctx_server_config_path,
        f"--tp_size={ctx_tp}", f"--pp_size={ctx_pp}"
    ]
    gen_server_args = gen_args + [
        "--port", "8002", "--extra_llm_api_options", gen_server_config_path,
        f"--tp_size={gen_tp}", f"--pp_size={gen_pp}"
    ]
    if "max_num_tokens" in ctx_server_config:
        ctx_server_args.append(
            f"--max_num_tokens={ctx_server_config['max_num_tokens']}")
    if "max_num_tokens" in gen_server_config:
        gen_server_args.append(
            f"--max_num_tokens={gen_server_config['max_num_tokens']}")

    with (MyThreadPoolExecutor(max_workers=16) as
          thread_pool, temp_dir, popen(ctx_server_args, env=env_ctx) as
          ctx_server, popen(gen_server_args, env=env_gen) as gen_server,
          popen([
              trtllm_serve_path, "disaggregated", "-c",
              disaggregated_serving_config_path, "--server_start_timeout",
              "3600"
          ]) as disaggregated_server):
        start_time = time.time()
        while time.time() - start_time < 3600:
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
            kwargs = {}
            if sampling_params is not None:
                kwargs.update(max_tokens=sampling_params.max_tokens,
                              temperature=sampling_params.temperature,
                              top_p=sampling_params.top_p,
                              stop=sampling_params.stop,
                              seed=sampling_params.seed)
                if (guided_decoding_params :=
                        sampling_params.guided_decoding) is not None:
                    extra_body = {}
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
                    kwargs.update(extra_body=extra_body)

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

        tokenizer = load_hf_tokenizer(model_name)

        try:
            yield DuckLLM(args, tokenizer, generate_async)
        finally:
            ctx_server.terminate()
            gen_server.terminate()
            disaggregated_server.terminate()

            ctx_server.wait()
            gen_server.wait()
            disaggregated_server.wait()


def run_parallel_test(model_name: str,
                      model_path: str,
                      ctx_pp: int,
                      ctx_tp: int,
                      gen_pp: int,
                      gen_tp: int,
                      test_sets: List[LlmapiAccuracyTestHarness],
                      ctx_model: str = None,
                      gen_model: str = None):
    if ctx_tp * ctx_pp + gen_tp * gen_pp > get_device_count():
        pytest.fail(
            f"Not enough devices for ctx_pp={ctx_pp}+ctx_tp={ctx_tp} and gen_pp={gen_pp}+gen_tp={gen_tp} test"
        )
    kv_cache_config = {
        "free_gpu_memory_fraction": 0.5,
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
                                  model_path,
                                  ctx_model=ctx_model,
                                  gen_model=gen_model) as llm:
        for test_set in test_sets:
            task = test_set(model_name)
            task.evaluate(llm)


@pytest.mark.timeout(3600)
class TestLlama3_1_8BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"

    @pytest.mark.skip_less_device_memory(32000)
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

    @parametrize_with_ids("overlap_scheduler", [True, False])
    @parametrize_with_ids("eagle3_one_model", [True, False])
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

    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
    def test_guided_decoding(self, backend: str, mocker):
        mocker.patch.dict(os.environ, {"TRTLLM_XGUIDANCE_LENIENT": "1"})
        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "guided_decoding_backend": backend,
            "cache_transceiver_config": {
                "backend": "DEFAULT"
            }
        }
        gen_server_config = {
            "guided_decoding_backend": backend,
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
            task = JsonModeEval(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
    def test_guided_decoding_with_eagle3(self, backend: str, mocker):
        mocker.patch.dict(os.environ, {"TRTLLM_XGUIDANCE_LENIENT": "1"})
        speculative_decoding_config = {
            "decoding_type": "Eagle",
            "max_draft_len": 3,
            "speculative_model_dir":
            f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B",
            "eagle3_one_model": False
        }

        ctx_server_config = {
            "disable_overlap_scheduler": True,
            "speculative_config": speculative_decoding_config,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.8,
            },
            "guided_decoding_backend": backend,
            "cache_transceiver_config": {
                "backend": "DEFAULT"
            }
        }
        gen_server_config = {
            "disable_overlap_scheduler": True,
            "speculative_config": speculative_decoding_config,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.8,
            },
            "guided_decoding_backend": backend,
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
            task = JsonModeEval(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.parametrize("tp,pp", [(1, 2), (2, 1), (2, 2)],
                             ids=["tp1pp2", "tp2pp1", "tp2pp2"])
    @pytest.mark.parametrize("testset", ["GSM8K", "MMLU"])
    def test_tp_pp_symmetric(self, tp, pp, testset):
        if tp * pp * 2 > get_device_count():
            pytest.skip(f"Not enough devices for tp={tp}*pp={pp} test")
        return run_parallel_test(self.MODEL_NAME, self.MODEL_PATH, pp, tp, pp,
                                 tp, [get_accuracy_task(testset)])

    @parametrize_with_ids("ctx_pp", [2, 4])
    @parametrize_with_ids("gen_tp", [1, 2])
    @pytest.mark.parametrize("testset", ["GSM8K", "MMLU"])
    def test_ctx_pp_gen_tp_asymmetric(self, ctx_pp, gen_tp, testset):
        if ctx_pp * gen_tp * 2 > get_device_count():
            pytest.skip(
                f"Not enough devices for ctx_pp={ctx_pp}*gen_tp={gen_tp} test")
        return run_parallel_test(self.MODEL_NAME, self.MODEL_PATH, ctx_pp, 1, 1,
                                 gen_tp, [get_accuracy_task(testset)])


@pytest.mark.skip_less_device_memory(140000)
@pytest.mark.timeout(3600)
class TestLlama4ScoutInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama4-models/Llama-4-Scout-17B-16E-Instruct"

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

    @pytest.mark.skip_less_device(8)
    @parametrize_with_ids("overlap_scheduler", [True, False])
    @parametrize_with_ids("mtp_nextn",
                          [0, pytest.param(2, marks=skip_pre_hopper)])
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

    @pytest.mark.parametrize("overlap_scheduler", [False, True])
    def test_auto_dtype(self, overlap_scheduler):
        pytest.skip(
            "Currently we require full kvcache for variable sliding window. "
            "This test only transfers the kvcache inside the sliding window.")

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
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


@skip_pre_blackwell
@pytest.mark.timeout(3600)
class TestQwen3_30B_A3B(LlmapiAccuracyTestHarness):
    fp4_model = f"{llm_models_root()}/Qwen3/saved_models_Qwen3-30B-A3B_nvfp4_hf"
    fp8_model = f"{llm_models_root()}/Qwen3/saved_models_Qwen3-30B-A3B_fp8_hf"

    @pytest.mark.parametrize("ctxpp,gentp", [(2, 2)], ids=["ctxpp2gentp2"])
    def test_mixed_ctx_gen_model(self, ctxpp, gentp):
        ctx_model = self.fp4_model
        gen_model = self.fp8_model
        return run_parallel_test("Qwen3/Qwen3-30B-A3B",
                                 ctx_model,
                                 ctx_pp=ctxpp,
                                 ctx_tp=1,
                                 gen_pp=1,
                                 gen_tp=gentp,
                                 test_sets=[GSM8K, MMLU],
                                 ctx_model=ctx_model,
                                 gen_model=gen_model)
