# I want to create accuracy tests for disaggregated serving.
# I need to to this by creating a new class that mimics LLM class. Instead of implementing the
# actual methods it will send OAI requests to the disaggregated serving endpoint.
# Please take a look at the existing test_llm_api_pytorch.py file for reference.

import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

import openai
import pytest
import requests
import yaml

from tensorrt_llm._torch import LLM
from tensorrt_llm.executor.result import GenerationResultBase
from tensorrt_llm.llmapi import CompletionOutput, RequestOutput, SamplingParams

from ..conftest import llm_models_root
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


class OpenAIServerClient:

    def __init__(self, disaggregated_server_config: Dict[str, Any],
                 ctx_server_config: Dict[str, Any],
                 gen_server_config: Dict[str, Any], model_name: str):
        self.temp_dir = tempfile.mkdtemp()
        self.disaggregated_serving_config_path = os.path.join(
            self.temp_dir, "disaggregated_serving_config.yaml")
        with open(self.disaggregated_serving_config_path, "w") as f:
            yaml.dump(disaggregated_server_config, f)
        ctx_server_config_path = os.path.join(self.temp_dir,
                                              "ctx_server_config.yaml")
        with open(ctx_server_config_path, "w") as f:
            yaml.dump(ctx_server_config, f)
        gen_server_config_path = os.path.join(self.temp_dir,
                                              "gen_server_config.yaml")
        with open(gen_server_config_path, "w") as f:
            yaml.dump(gen_server_config, f)

        with LLM(model_name) as llm:
            self.args = llm.args

        trtllm_serve_path = "trtllm-serve"
        # Common arguments for both servers
        common_args = [
            trtllm_serve_path, model_name, "--host", "localhost", "--backend",
            "pytorch"
        ]
        env_ctx = os.environ.copy()
        env_ctx["TRTLLM_USE_UCX_KVCACHE"] = "1"

        # Start the context server
        self._ctx_server = subprocess.Popen(common_args + [
            "--port", "8001", "--extra_llm_api_options", ctx_server_config_path
        ],
                                            env=env_ctx)
        # Start the generation server
        env_gen = os.environ.copy()
        env_gen["TRTLLM_USE_UCX_KVCACHE"] = "1"
        self._gen_server = subprocess.Popen(common_args + [
            "--port", "8002", "--extra_llm_api_options", gen_server_config_path
        ],
                                            env=env_gen)

        # Start the disaggregated server
        self._disaggregated_server = subprocess.Popen([
            trtllm_serve_path, "disaggregated", "-c",
            self.disaggregated_serving_config_path
        ])
        self.model_name = model_name

        while True:
            time.sleep(1)
            try:
                print("Checking health endpoint")
                response = requests.get("http://localhost:8000/health")
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                continue

        self.client = openai.OpenAI(api_key="1234567890",
                                    base_url=f"http://localhost:8000/v1")

    def generate_async(self,
                       prompt: str,
                       sampling_params: Optional[SamplingParams] = None):
        # TODO: Make this async
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            stream=False,
            **({
                "max_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "stop": sampling_params.stop,
                "seed": sampling_params.seed
            } if sampling_params else {}))
        result = Result(
            id=0,
            sampling_params=sampling_params,
            outputs=[CompletionOutput(text=response.choices[0].text, index=0)])
        requested_output = RequestOutput._from_generation_result(result,
                                                                 prompt=prompt)
        setattr(requested_output, "result", result.result)
        return requested_output

    def __del__(self):
        shutil.rmtree(self.temp_dir)
        self._ctx_server.terminate()
        self._gen_server.terminate()
        self._disaggregated_server.terminate()

        self._ctx_server.wait()
        self._gen_server.wait()
        self._disaggregated_server.wait()


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
        client = OpenAIServerClient(disaggregated_server_config,
                                    ctx_server_config, gen_server_config,
                                    self.MODEL_PATH)
        task = MMLU(self.MODEL_NAME)
        task.evaluate(client)
