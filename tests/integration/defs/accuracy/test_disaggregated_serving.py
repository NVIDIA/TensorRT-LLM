# I want to create accuracy tests for disaggregated serving.
# I need to to this by creating a new class that mimics LLM class. Instead of implementing the
# actual methods it will send OAI requests to the disaggregated serving endpoint.
# Please take a look at the existing test_llm_api_pytorch.py file for reference.
import contextlib
import os
import tempfile
import time
from collections import namedtuple
from typing import Any, Dict, List, Optional

import openai
import pytest
import requests
import yaml

from tensorrt_llm.executor.result import GenerationResultBase
from tensorrt_llm.llmapi import CompletionOutput, RequestOutput, SamplingParams
from tensorrt_llm.llmapi.llm_args import LlmArgs

from ..conftest import llm_models_root
from ..trt_test_alternative import popen
from .accuracy_core import MMLU, LlmapiAccuracyTestHarness


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


@contextlib.contextmanager
def launch_disaggregated_llm(disaggregated_server_config: Dict[str, Any],
                             ctx_server_config: Dict[str, Any],
                             gen_server_config: Dict[str,
                                                     Any], model_name: str):
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

    args = LlmArgs.from_kwargs(model=model_name)

    trtllm_serve_path = "trtllm-serve"
    # Common arguments for both servers
    common_args = [
        trtllm_serve_path, model_name, "--host", "localhost", "--backend",
        "pytorch"
    ]
    env_ctx = os.environ.copy()
    env_ctx["TRTLLM_USE_UCX_KVCACHE"] = "1"

    env_gen = os.environ.copy()
    env_gen["TRTLLM_USE_UCX_KVCACHE"] = "1"

    client = openai.OpenAI(api_key="1234567890",
                           base_url=f"http://localhost:8000/v1")

    with (temp_dir,
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
              disaggregated_serving_config_path
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

        def generate_async(prompt: str,
                           sampling_params: Optional[SamplingParams] = None):
            # TODO: Make this async
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

        yield DuckLLM(args, generate_async)

        ctx_server.terminate()
        gen_server.terminate()
        disaggregated_server.terminate()

        ctx_server.wait()
        gen_server.wait()
        disaggregated_server.wait()


class TestLlama3_1_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Meta-Llama-3.1-8B"

    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.skip_device_not_contain(["H100"])
    @pytest.mark.parametrize("overlap_scheduler", [False, True])
    def test_auto_dtype(self, overlap_scheduler):
        ctx_server_config = {
            "pytorch_backend_config": {
                "enable_overlap_scheduler": False
            }
        }
        gen_server_config = {
            "pytorch_backend_config": {
                "enable_overlap_scheduler": overlap_scheduler
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
        task = MMLU(self.MODEL_NAME)
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            task.evaluate(llm)
