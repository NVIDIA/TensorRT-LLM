import asyncio
from difflib import SequenceMatcher
from pathlib import Path

import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import getSMVersion

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig
from tensorrt_llm.llmapi.utils import get_total_gpu_memory


def similar(a, b, threshold=0.9):
    "similar compare a and b "
    return SequenceMatcher(None, a, b).ratio() >= threshold


@pytest.mark.parametrize("model_name", ["DeepSeek-V3-Lite"],
                         ids=["deepseekv3_lite"])
@pytest.mark.parametrize("backend", ["TRTLLM"], ids=["trtllm"])
@pytest.mark.parametrize("quant", ["bf16"])
@pytest.mark.parametrize("tp_size", [1, 4], ids=["tp1", "tp4"])
def test_deepseek_streaming(model_name, backend, quant, tp_size):
    model_path = {
        "bf16": "bf16",
        "fp8": "fp8",
        "fp4": "nvfp4_moe_only",
    }
    assert quant in model_path.keys()

    is_fp8 = quant == "fp8"
    is_fp4 = quant == "fp4"

    if tp_size == 4:
        pytest.skip(f"https://nvbugs/5515753")

    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"Not enough GPUs available, need {tp_size} "
                    f"but only have {torch.cuda.device_count()}")

    if is_fp8 and getSMVersion() != 90:
        pytest.skip(f"FP8 is not supported in this SM version {getSMVersion()}")

    if is_fp4 and getSMVersion() < 100:
        pytest.skip(f"FP4 is not supported in this SM version {getSMVersion()}")

    if get_total_gpu_memory(0) < 60 * 1024**3:
        pytest.skip(f"Not enough GPU memory to run. {get_total_gpu_memory(0)}")

    if tp_size == 1:
        enable_attention_dp = False
        moe_max_num_tokens = None
    else:
        enable_attention_dp = True
        moe_max_num_tokens = 64

    prompts = [
        "The president of the United States is",
    ] * 32

    expected_outputs = [
        " the head of state and head of government of the",
    ] * 32

    pytorch_config = dict(
        disable_overlap_scheduler=True,
        attn_backend=backend,
    )
    moe_config = MoeConfig(max_num_tokens=moe_max_num_tokens)
    model_dir = str(llm_models_root() / model_name / model_path[quant])

    assert Path(model_dir).exists()

    llm = LLM(model=model_dir,
              tensor_parallel_size=tp_size,
              enable_chunked_prefill=False,
              **pytorch_config,
              moe_config=moe_config,
              moe_expert_parallel_size=-1,
              moe_tensor_parallel_size=-1,
              enable_attention_dp=enable_attention_dp,
              kv_cache_config=KvCacheConfig(enable_block_reuse=False))

    sampling_params = SamplingParams(max_tokens=10)

    async def task(prompt: str):
        future = llm.generate_async(prompt,
                                    streaming=True,
                                    sampling_params=sampling_params)
        output = await future.aresult()
        return output.outputs[0].text

    async def test():
        tasks = [task(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(expected_outputs), "Output length mismatch"
        for result, expected in zip(results, expected_outputs):
            assert similar(result, expected,
                           1.0), f"Expected '{expected}' but get '{result}'"

    asyncio.run(test())
