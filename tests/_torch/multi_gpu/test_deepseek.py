import os
import sys
from difflib import SequenceMatcher

import pytest
import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.llmapi.utils import get_total_gpu_memory

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pathlib import Path

from utils.llm_data import llm_models_root
from utils.util import getSMVersion

MAX_SEQ_LEN = 2048


def similar(a, b, threshold=0.9):
    "similar compare a and b "
    return SequenceMatcher(None, a, b).ratio() >= threshold


@pytest.mark.parametrize("model_name", ["DeepSeek-V3-Lite"],
                         ids=["deepseekv3_lite"])
@pytest.mark.parametrize("quant", ["bf16", "fp4"])
@pytest.mark.parametrize("tp_size", [1, 2], ids=["tp1", "tp2"])
@pytest.mark.parametrize("enable_dp", [True, False],
                         ids=["enable_dp", "disable_dp"])
def test_model(model_name, quant, tp_size, enable_dp):
    model_path = {
        "bf16": "bf16",
        "fp8": "fp8",
        "fp4": "nvfp4_moe_only",
    }
    assert quant in model_path.keys()

    is_fp8 = quant == "fp8"
    is_fp4 = quant == "fp4"

    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"Not enough GPUs available, need {tp_size} "
                    f"but only have {torch.cuda.device_count()}")

    if is_fp8 and getSMVersion() < 90:
        pytest.skip(f"FP8 is not supported in this SM version {getSMVersion()}")

    if is_fp4 and getSMVersion() < 100:
        pytest.skip(f"FP4 is not supported in this SM version {getSMVersion()}")

    if enable_dp and tp_size == 1:
        pytest.skip(
            f"Testing attention_dp is unnecessary when tp_size equals 1.")

    if get_total_gpu_memory(0) < 60 * 1024**3:
        pytest.skip(f"Not enough GPU memory to run. {get_total_gpu_memory(0)}")

    prompts = [
        "The president of the United States is",
    ] * 32

    expected_outputs = [
        " the head of state and head of government of the",
    ] * 32

    pytorch_config = PyTorchConfig(enable_overlap_scheduler=False,
                                   kv_cache_dtype="auto")

    model_dir = str(llm_models_root() / model_name / model_path[quant])

    assert Path(model_dir).exists()

    llm = LLM(model=model_dir,
              tensor_parallel_size=tp_size,
              enable_chunked_prefill=False,
              pytorch_backend_config=pytorch_config,
              moe_expert_parallel_size=-1,
              moe_tensor_parallel_size=-1,
              enable_attention_dp=enable_dp,
              kv_cache_config=KvCacheConfig(enable_block_reuse=False))

    with llm:
        outputs = llm.generate(
            prompts,
            sampling_params=SamplingParams(max_tokens=10),
        )

    assert len(outputs) == len(expected_outputs), "Output length mismatch"
    for output, expected in zip(outputs, expected_outputs):
        output_text = output.outputs[0].text
        # print(output_text)
        # print(output.outputs[0].token_ids)
        assert similar(
            output_text,
            expected), f"Expected '{expected}' but get '{output_text}'"


if __name__ == '__main__':
    test_model("DeepSeek-V3-Lite", "bf16", 1, enable_dp=False)
    test_model("DeepSeek-V3-Lite", "fp4", 1, enable_dp=True)
