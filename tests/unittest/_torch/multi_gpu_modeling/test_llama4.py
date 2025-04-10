# TODO: Enable this test when the models are checked into the llm repo!!
from difflib import SequenceMatcher

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
import torch

@pytest.mark.parametrize(
    "model_name",
    ["Llama-4-Maverick-17B-128E-Instruct", "Llama-4-Scout-17B-16E-Instruct"],
    ids=['maverick', 'scout'])
@pytest.mark.parametrize("backend", ["TRTLLM", "FLASHINFER"],
                         ids=["trtllm", "flashinfer"])
@pytest.mark.parametrize("tp_size", [8], ids=["tp8"])
def test_llama4(model_name, backend, tp_size):
    prompts = [
        {
            "prompt": "The president of the United States is"
        },
        {
            "prompt": "<|image|>This image is of color",
            "multi_modal_data": {
                "image": [torch.ones(3, 1024, 1024)]
            }
        }
    ]

    expected_outputs = [
        " the head of state and head of government of the",
        " solid white"
    ]

    pytorch_config = PyTorchConfig(attn_backend=backend, )
    model_dir = str(llm_models_root() / model_name)

    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tp_size,
        pytorch_backend_config=pytorch_config,
    )
    with llm:
        outputs = llm.generate(
            prompts,
            sampling_params=SamplingParams(max_tokens=10),
        )

    assert len(outputs) == len(expected_outputs), "Output length mismatch"

    def similar(a, b, threshold=0.9):
        return SequenceMatcher(None, a, b).ratio() >= threshold

    for output, expected in zip(outputs, expected_outputs):
        output_text = output.outputs[0].text
        assert similar(
            output_text,
            expected), f"Expected '{expected}' but get '{output_text}'"
