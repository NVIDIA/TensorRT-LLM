import json
import os

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.utils import get_total_gpu_memory
from tensorrt_llm.mapping import CpType
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

MAX_SEQ_LEN = 4096 + 1024


@pytest.mark.post_merge
@pytest.mark.parametrize("backend", ["pytorch"])
@pytest.mark.parametrize("model_name",
                         ["llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k"],
                         ids=["llama-3-8b-1048k"])
@pytest.mark.parametrize("quant", ["bf16", "fp8"])
@pytest.mark.parametrize("sp_size", [1, 2, 4], ids=["sp1", "sp2", "sp4"])
@pytest.mark.parametrize("sa_block_size", [256, 1024],
                         ids=["block1024", "block4096"])
@pytest.mark.parametrize("sa_anchor_size", [256, 1024],
                         ids=["anchor1024", "anchor4096"])
def test_model(backend, model_name, quant, sp_size, sa_block_size,
               sa_anchor_size):
    pytest.skip("https://nvbugs/5391679")
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
    if sp_size != 1:
        pytest.skip(f"skip multi gpu tests due to flashinfer's jitting mode")
    if torch.cuda.device_count() < sp_size:
        pytest.skip(f"Not enough GPUs available, need {sp_size} "
                    f"but only have {torch.cuda.device_count()}")
    if sa_anchor_size > sa_block_size:
        pytest.skip(
            f"Unsupported sa_anchor_size {sa_anchor_size} > sa_block_size {sa_block_size}"
        )

    if get_total_gpu_memory(0) < 32 * 1024**3:
        pytest.skip("Not enough GPU memory to run BF16 model")

    model_dir = str(llm_models_root() / model_name)
    cp_config = {
        "cp_type": CpType.STAR,
        "cp_anchor_size": sa_anchor_size,
        "block_size": sa_block_size
    }
    max_batch_size = 20
    max_output_tokens = 128
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7)
    pytorch_backend_options = dict(attn_backend='FLASHINFER_STAR_ATTENTION',
                                   disable_overlap_scheduler=True)

    llm = LLM(model=model_dir,
              backend=backend,
              kv_cache_config=kv_cache_config,
              tensor_parallel_size=1,
              quant_config=quant_config,
              context_parallel_size=sp_size,
              cp_config=cp_config,
              **pytorch_backend_options,
              max_batch_size=max_batch_size,
              max_input_len=MAX_SEQ_LEN - max_output_tokens,
              max_seq_len=MAX_SEQ_LEN,
              max_num_tokens=(sa_block_size + sa_anchor_size) * max_batch_size)

    inputs, references = [], []
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    with open(f'{current_dir}/test_star_attention_input.jsonl', 'r') as f:
        for line in f:
            sample = json.loads(line)
            inputs.append({
                'prompt': sample['input_context'],
                'query': sample['input_query']
            })
            references.append(sample['outputs'][0])
    with llm:
        outputs = llm.generate(
            inputs,
            use_tqdm=True,
            sampling_params=SamplingParams(
                max_tokens=max_output_tokens,
                add_special_tokens=False,
            ),
        )

    count = 0
    for ref, ret in zip(references, outputs):
        #print(f'reference = {ref}')
        #print(f'prediction = {ret.outputs[0].text}')
        if ref not in ret.outputs[0].text:
            print(f'reference {ref} is not in the output {ret.outputs[0].text}')
        else:
            count = count + 1
    acc = count / len(outputs)
    if acc < 1.0:
        assert False, 'accuracy test of star attention failed'


if __name__ == '__main__':
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 1, 256, 256)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 1, 1024, 256)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 1, 1024, 1024)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "fp8", 1, 256, 256)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "fp8", 1, 1024, 256)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "fp8", 1, 1024, 1024)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 2, 1024, 256)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 2, 1024, 1024)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 2, 256, 256)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 4, 1024, 256)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 4, 1024, 1024)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 4, 256, 256)
