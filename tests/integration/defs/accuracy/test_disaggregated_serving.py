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

from ..conftest import llm_models_root, parametrize_with_ids, skip_pre_hopper
from ..trt_test_alternative import popen
from .accuracy_core import GSM8K, MMLU, LlmapiAccuracyTestHarness


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
        self.shutdown(wait=False, cancel_futures=True)
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
                               backend="pytorch",
                               tensor_parallel_size=tensor_parallel_size)

    trtllm_serve_path = "trtllm-serve"
    # Common arguments for both servers
    common_args = [
        trtllm_serve_path, model_name, "--host", "localhost", "--backend",
        "pytorch"
    ]
    if tensor_parallel_size > 1:
        common_args.append(f"--tp_size={tensor_parallel_size}")

    env_ctx = os.environ.copy()
    env_ctx["TRTLLM_USE_UCX_KVCACHE"] = "1"
    env_ctx["CUDA_VISIBLE_DEVICES"] = ",".join(
        map(str, range(tensor_parallel_size)))

    env_gen = os.environ.copy()
    env_gen["TRTLLM_USE_UCX_KVCACHE"] = "1"
    env_gen["CUDA_VISIBLE_DEVICES"] = ",".join(
        map(str, range(tensor_parallel_size, 2 * tensor_parallel_size)))

    with (MyThreadPoolExecutor(max_workers=16) as thread_pool, temp_dir,
          popen(common_args + [
              "--port", "8001", "--extra_llm_api_options",
              ctx_server_config_path
          ],
                env=env_ctx) as ctx_server,
          popen(common_args + [
              "--port", "8002", "--extra_llm_api_options",
              gen_server_config_path
          ],
                env=env_gen) as gen_server,
          popen([
              trtllm_serve_path, "disaggregated", "-c",
              disaggregated_serving_config_path, "--server_start_timeout",
              "3600"
          ]) as disaggregated_server):
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

        def send_request(prompt: str, sampling_params: SamplingParams):
            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                stream=False,
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
                                CompletionOutput(text=response.choices[0].text,
                                                 index=0)
                            ])
            requested_output = RequestOutput._from_generation_result(
                result, prompt=prompt)
            setattr(requested_output, "result", result.result)
            return requested_output

        def generate_async(prompt: str,
                           sampling_params: Optional[SamplingParams] = None):
            future = thread_pool.submit(send_request, prompt, sampling_params)
            thread_pool.futures.append(future)
            return future

        yield DuckLLM(args, generate_async)

        ctx_server.terminate()
        gen_server.terminate()
        disaggregated_server.terminate()

        ctx_server.wait()
        gen_server.wait()
        disaggregated_server.wait()


@pytest.mark.timeout(3600)
class TestLlama3_1_8BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"

    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.skip_device_not_contain(["H100", "H200"])
    @pytest.mark.parametrize("disable_overlap_scheduler", [False, True])
    def test_auto_dtype(self, disable_overlap_scheduler):
        ctx_server_config = {"disable_overlap_scheduler": True}
        gen_server_config = {
            "disable_overlap_scheduler": disable_overlap_scheduler
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


@pytest.mark.timeout(3600)
@pytest.mark.skip_less_device_memory(140000)
class TestLlama4ScoutInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama4-models/Llama-4-Scout-17B-16E-Instruct"

    @pytest.mark.parametrize("overlap_scheduler", [False, True])
    def test_auto_dtype(self, overlap_scheduler):
        ctx_server_config = {"disable_overlap_scheduler": True}
        gen_server_config = {"disable_overlap_scheduler": overlap_scheduler}
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

    @parametrize_with_ids("overlap_scheduler", [True, False])
    @parametrize_with_ids("mtp_nextn",
                          [0, pytest.param(2, marks=skip_pre_hopper)])
    def test_auto_dtype(self, overlap_scheduler, mtp_nextn):
        ctx_server_config = {
            "pytorch_backend_config": {
                "disable_overlap_scheduler": True
            }
        }
        gen_server_config = {
            "pytorch_backend_config": {
                "disable_overlap_scheduler": not overlap_scheduler
            }
        }
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
