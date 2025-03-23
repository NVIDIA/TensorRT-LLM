import asyncio
import os
import sys
from difflib import SequenceMatcher

import pytest
import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.llmapi import MTPDecodingConfig
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
@pytest.mark.parametrize("backend", ["TRTLLM"], ids=["trtllm"])
@pytest.mark.parametrize("quant", ["bf16", "fp4", "fp8"])
@pytest.mark.parametrize("tp_size", [1, 2, 4], ids=["tp1", "tp2", "tp4"])
@pytest.mark.parametrize("pp_size", [1, 2, 4], ids=["pp1", "pp2", "pp4"])
@pytest.mark.parametrize("ep_size", [1, 2, 4], ids=["ep1", "ep2", "ep4"])
@pytest.mark.parametrize("mtp_nextn", [0, 1, 2],
                         ids=["nextn0", "nextn1", "nextn2"])
@pytest.mark.parametrize("enable_dp", [True, False],
                         ids=["enable_dp", "disable_dp"])
@pytest.mark.parametrize("enable_cuda_graph", [True, False],
                         ids=["enable_cuda_graph", "disable_cuda_graph"])
@pytest.mark.parametrize(
    "enable_overlap_scheduler", [True, False],
    ids=["enable_overlap_scheduler", "disable_overlap_scheduler"])
def test_deepseek(model_name, backend, quant, tp_size, pp_size, ep_size,
                  mtp_nextn, enable_dp, enable_cuda_graph,
                  enable_overlap_scheduler):
    model_path = {
        "bf16": "bf16",
        "fp8": "fp8",
        "fp4": "nvfp4_moe_only",
    }
    assert quant in model_path.keys()

    is_fp8 = quant == "fp8"
    is_fp4 = quant == "fp4"

    if ep_size > tp_size:
        pytest.skip(
            f"Expert parallel size {ep_size} must be less than or equal to tensor parallel size {tp_size}"
        )

    if torch.cuda.device_count() < tp_size * pp_size:
        pytest.skip(f"Not enough GPUs available, need {tp_size * pp_size} "
                    f"but only have {torch.cuda.device_count()}")

    if is_fp8 and getSMVersion() != 90:
        pytest.skip(f"FP8 is not supported in this SM version {getSMVersion()}")

    if is_fp4 and getSMVersion() < 100:
        pytest.skip(f"FP4 is not supported in this SM version {getSMVersion()}")

    if is_fp4 and mtp_nextn > 0:
        pytest.skip(f"FP4 checkpoint has no MTP weights")

    if mtp_nextn > 0 and getSMVersion() < 100:
        pytest.skip(f"Only Blackwell MLA kernel can support MTP now")

    if pp_size > 1 and (enable_dp or mtp_nextn > 0):
        pytest.skip(
            "Hang issue with DP attention / MTP + PP: https://nvbugspro.nvidia.com/bug/5170160"
        )
    if pp_size > 2 and enable_cuda_graph and enable_overlap_scheduler:
        pytest.skip(
            "Race condition causes incorrect output for some requests: https://nvbugspro.nvidia.com/bug/5177565"
        )

    if get_total_gpu_memory(0) < 60 * 1024**3:
        pytest.skip(f"Not enough GPU memory to run. {get_total_gpu_memory(0)}")

    prompts = [
        "The president of the United States is",
    ] * 32

    expected_outputs = [
        " the head of state and head of government of the",
    ] * 32

    pytorch_config = PyTorchConfig(
        enable_overlap_scheduler=enable_overlap_scheduler,
        use_cuda_graph=enable_cuda_graph,
        kv_cache_dtype="auto",
        attn_backend=backend,
    )

    mtp_config = MTPDecodingConfig(
        num_nextn_predict_layers=mtp_nextn) if mtp_nextn > 0 else None

    model_dir = str(llm_models_root() / model_name / model_path[quant])

    assert Path(model_dir).exists()

    llm = LLM(model=model_dir,
              tensor_parallel_size=tp_size,
              pipeline_parallel_size=pp_size,
              enable_chunked_prefill=False,
              pytorch_backend_config=pytorch_config,
              moe_expert_parallel_size=ep_size,
              moe_tensor_parallel_size=-1,
              enable_attention_dp=enable_dp,
              kv_cache_config=KvCacheConfig(enable_block_reuse=False),
              speculative_config=mtp_config)

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
        # Limited by the kv cache length, the output length of MTP maybe
        # a little smaller than original model.
        expected = expected[0:len(output_text)] if mtp_nextn > 0 else expected
        assert similar(output_text, expected,
                       1.0), f"Expected '{expected}' but get '{output_text}'"


@pytest.mark.parametrize("model_name", ["DeepSeek-V3-Lite"],
                         ids=["deepseekv3_lite"])
@pytest.mark.parametrize("backend", ["TRTLLM"], ids=["trtllm"])
@pytest.mark.parametrize("quant", ["bf16"])
@pytest.mark.parametrize("tp_size", [1], ids=["tp1"])
def test_deepseek_streaming(model_name, backend, quant, tp_size):
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

    if is_fp8 and getSMVersion() != 90:
        pytest.skip(f"FP8 is not supported in this SM version {getSMVersion()}")

    if is_fp4 and getSMVersion() < 100:
        pytest.skip(f"FP4 is not supported in this SM version {getSMVersion()}")

    if get_total_gpu_memory(0) < 60 * 1024**3:
        pytest.skip(f"Not enough GPU memory to run. {get_total_gpu_memory(0)}")

    prompts = [
        "The president of the United States is",
    ] * 32

    expected_outputs = [
        " the head of state and head of government of the",
    ] * 32

    pytorch_config = PyTorchConfig(
        enable_overlap_scheduler=False,
        use_cuda_graph=False,
        kv_cache_dtype="auto",
        attn_backend=backend,
    )

    model_dir = str(llm_models_root() / model_name / model_path[quant])

    assert Path(model_dir).exists()

    llm = LLM(model=model_dir,
              tensor_parallel_size=tp_size,
              enable_chunked_prefill=False,
              pytorch_backend_config=pytorch_config,
              moe_expert_parallel_size=-1,
              moe_tensor_parallel_size=-1,
              enable_attention_dp=False,
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
