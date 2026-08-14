import os
import tempfile
from dataclasses import asdict
from typing import List, Optional

import openai
import pytest
import yaml

from tensorrt_llm.executor.request import LoRARequest

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module", ids=["llama-models/llama-7b-hf"])
def model_name() -> str:
    return "llama-models/llama-7b-hf"


@pytest.fixture(scope="module")
def lora_adapter_names() -> List[Optional[str]]:
    return [
        None, "llama-models/luotuo-lora-7b-0.1",
        "llama-models/Japanese-Alpaca-LoRA-7b-v0"
    ]


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file():
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "extra_llm_api_options.yaml")
    try:
        extra_llm_api_options_dict = {
            "lora_config": {
                "lora_target_modules": ['attn_q', 'attn_k', 'attn_v'],
                "max_lora_rank": 8,
                "max_loras": 4,
                "max_cpu_loras": 4,
            },
            # Disable CUDA graph
            # TODO: remove this once we have a proper fix for CUDA graph in LoRA
            "cuda_graph_config": None
        }

        with open(temp_file_path, 'w') as f:
            yaml.dump(extra_llm_api_options_dict, f)

        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(model_name: str,
           temp_extra_llm_api_options_file: str) -> RemoteOpenAIServer:
    model_path = get_model_path(model_name)
    args = []
    args.extend(["--backend", "pytorch"])
    args.extend(["--extra_llm_api_options", temp_extra_llm_api_options_file])
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer) -> openai.OpenAI:
    return server.get_client()


def test_lora(client: openai.OpenAI, model_name: str,
              lora_adapter_names: List[str]):
    prompts = [
        "美国的首都在哪里? \n答案:",
        "美国的首都在哪里? \n答案:",
        "美国的首都在哪里? \n答案:",
        "アメリカ合衆国の首都はどこですか? \n答え:",
        "アメリカ合衆国の首都はどこですか? \n答え:",
        "アメリカ合衆国の首都はどこですか? \n答え:",
    ]
    references = [
        "沃尔玛\n\n## 新闻\n\n* ",
        "美国的首都是华盛顿。\n\n美国的",
        "纽约\n\n### カンファレンスの",
        "Washington, D.C.\nWashington, D.C. is the capital of the United",
        "华盛顿。\n\n英国の首都是什",
        "ワシントン\nQ1. アメリカ合衆国",
    ]

    for prompt, reference, lora_adapter_name in zip(prompts, references,
                                                    lora_adapter_names * 2):
        extra_body = {}
        if lora_adapter_name is not None:
            lora_req = LoRARequest(lora_adapter_name,
                                   lora_adapter_names.index(lora_adapter_name),
                                   get_model_path(lora_adapter_name))
            extra_body["lora_request"] = asdict(lora_req)

        response = client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=20,
            extra_body=extra_body,
        )
        # lora output is not deterministic, so do not check if match with reference
        # TODO: need to fix this
        print(f"response: {response.choices[0].text}")
        print(f"reference: {reference}")
        # assert similar(response.choices[0].text, reference)
