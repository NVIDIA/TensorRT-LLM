import os
import sys
from difflib import SequenceMatcher

import pytest
import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi.llm_utils import BuildConfig, CalibConfig
from tensorrt_llm.llmapi.utils import get_total_gpu_memory
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.llm_data import llm_models_root
from utils.util import getSMVersion

MAX_SEQ_LEN = 2048


def similar(a, b, threshold=0.9):
    "similar compare a and b "
    return SequenceMatcher(None, a, b).ratio() >= threshold


@pytest.mark.parametrize("model_name", ["llama-3.1-model/Meta-Llama-3.1-8B"],
                         ids=["llama-3.1-8b"])
@pytest.mark.parametrize("backend", ["TRTLLM", "FLASHINFER"],
                         ids=["trtllm", "flashinfer"])
@pytest.mark.parametrize("quant", ["bf16", "fp8", "fp8_kv_cache"])
@pytest.mark.parametrize("tp_size", [1, 4], ids=["tp1", "tp4"])
@pytest.mark.parametrize("pp_size", [1, 2], ids=["pp1", "pp2"])
@pytest.mark.parametrize("torch_compile", [True, False],
                         ids=["torch_compile", "eager"])
def test_llama(model_name, backend, quant, tp_size, pp_size, torch_compile):
    quant_configs = {
        "bf16":
        QuantConfig(),
        "fp8":
        QuantConfig(quant_algo=QuantAlgo.FP8),
        "fp8_kv_cache":
        QuantConfig(
            quant_algo=QuantAlgo.FP8,
            kv_cache_quant_algo=QuantAlgo.FP8,
        ),
    }
    quant_config = quant_configs[quant]
    is_fp8 = quant_config.quant_algo == QuantAlgo.FP8
    is_fp8_kv_cache = quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    if torch.cuda.device_count() < tp_size * pp_size:
        pytest.skip(f"Not enough GPUs available, need {tp_size * pp_size} "
                    f"but only have {torch.cuda.device_count()}")
    if is_fp8 and getSMVersion() < 90:
        pytest.skip(f"FP8 is not supported in this SM version {getSMVersion()}")
    # 8GB weight + 8GB KV cache + 8GB cache_indirection (TRT engine only) = 24GB
    if is_fp8 and get_total_gpu_memory(0) < 24 * 1024**3:
        pytest.skip("Not enough GPU memory to run FP8 model")
    # 16GB weight + 8GB KV cache + 8GB cache_indirection (TRT engine only) = 32GB
    if not is_fp8 and get_total_gpu_memory(0) < 32 * 1024**3:
        pytest.skip("Not enough GPU memory to run BF16 model")
    if pp_size > 1 and tp_size > 1:
        pytest.skip(
            "https://nvbugs/5164088 - Fails creating IPC memory when creating allreduce workspace"
        )
    if torch_compile and pp_size > 1:
        pytest.skip(
            "Pipeline parallel with torch.compile is not supported yet.\n"
            "Issue: Unfusing flashinfer_fused_add_rmsnorm causes outputs to be "
            "discarded at graph breaks.")

    prompts = [
        "The president of the United States is",
    ]

    expected_outputs = [
        " the head of state and head of government of the",
    ]

    pytorch_config = PyTorchConfig(
        torch_compile_enabled=torch_compile,
        cuda_graph_padding_enabled=torch_compile,
        cuda_graph_batch_sizes=[4],
        attn_backend=backend,
    )
    if is_fp8_kv_cache:
        pytorch_config.kv_cache_dtype = "fp8"

    model_dir = str(llm_models_root() / model_name)
    if is_fp8:
        fp8_model_names = {
            "llama-3.1-model/Meta-Llama-3.1-8B":
            "llama-3.1-model/Llama-3.1-8B-Instruct-FP8"
        }
        model_dir = str(llm_models_root() / fp8_model_names[model_name])

    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        quant_config=quant_config,
        pytorch_backend_config=pytorch_config,
        calib_config=CalibConfig(calib_dataset=str(llm_models_root() /
                                                   "datasets/cnn_dailymail")),
        build_config=BuildConfig(
            max_seq_len=2048,
        ),  # This verifies a bug in compile warmUp that does not generate valid warmup requests
    )
    with llm:
        outputs = llm.generate(
            prompts,
            sampling_params=SamplingParams(max_tokens=10),
        )

    assert len(outputs) == len(expected_outputs), "Output length mismatch"
    for output, expected in zip(outputs, expected_outputs):
        output_text = output.outputs[0].text
        print(output_text)
        print(output.outputs[0].token_ids)
        assert similar(
            output_text,
            expected), f"Expected '{expected}' but get '{output_text}'"
