import json
import os
import shutil

import pytest

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm.llmapi import KvCacheConfig

configs = """
{
    "architectures": [
            "OranginaForCausalLM"
    ],
    "model_type": "mixtral",
    "torch_dtype": "bfloat16",
    "num_hidden_layers": 4,
    "num_experts": 128,
    "experts_per_token": 4,
    "vocab_size": 201088,
    "hidden_size": 2880,
    "intermediate_size": 2880,
    "head_dim": 64,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "sliding_window": 128,
    "initial_context_length": 131072,
    "rope_theta": 150000,
    "rope_scaling_factor": 32.0,
    "rope_ntk_alpha": 1,
    "rope_ntk_beta": 32
}
"""


def dump_config_json(dst_dir):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)

    dst_path = os.path.join(dst_dir, 'config.json')
    with open(dst_path, 'w', encoding='utf-8') as f:
        json_configs = json.loads(configs)
        json.dump(json_configs, f, indent=2, ensure_ascii=False)


@pytest.mark.parametrize("kv_cache", ["fp8", "auto"])
def test_orangina_trtllmgen(kv_cache):
    prompts = [
        "How are you?",
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    pytorch_config = dict(
        disable_overlap_scheduler=False,
        use_cuda_graph=True,
        kv_cache_dtype=kv_cache,
        attn_backend="TRTLLM",
        load_format="dummy",
    )

    tmp_model_dir = f"/tmp/test_model_trtllm"

    dump_config_json(tmp_model_dir)

    llm = LLM(model=tmp_model_dir,
              tensor_parallel_size=1,
              enable_chunked_prefill=False,
              **pytorch_config,
              moe_expert_parallel_size=-1,
              moe_tensor_parallel_size=-1,
              enable_attention_dp=False,
              kv_cache_config=KvCacheConfig(enable_block_reuse=False,
                                            free_gpu_memory_fraction=0.4))

    sampling_params = SamplingParams(max_tokens=20)
    llm.generate(prompts, sampling_params)
