import json
import os
import shutil

import pytest
from transformers import AutoTokenizer
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import \
    IS_TRITON_KERNELS_AVAILABLE
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

configs = """
{
    "architectures": [
        "GptOssForCausalLM"
    ],
    "model_type": "gpt_oss",
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
    "initial_context_length": 4096,
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


@pytest.mark.parametrize("moe_backend", ["CUTLASS", "TRITON"])
def test_gpt_oss_trtllmgen(moe_backend):
    if moe_backend == "TRITON" and not IS_TRITON_KERNELS_AVAILABLE:
        pytest.skip("Triton kernels are not available")

    prompts = [
        "How are you?",
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    pytorch_config = dict(
        disable_overlap_scheduler=False,
        cuda_graph_config=CudaGraphConfig(),
        attn_backend="TRTLLM",
        load_format="dummy",
        moe_config=MoeConfig(backend=moe_backend),
    )

    tmp_model_dir = f"/tmp/test_model_trtllm"

    dump_config_json(tmp_model_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        f"{llm_models_root()}/gpt_oss/gpt-oss-20b")

    llm = LLM(model=tmp_model_dir,
              tokenizer=tokenizer,
              tensor_parallel_size=1,
              enable_chunked_prefill=False,
              **pytorch_config,
              max_batch_size=16,
              max_seq_len=1024,
              moe_expert_parallel_size=-1,
              moe_tensor_parallel_size=-1,
              enable_attention_dp=False,
              kv_cache_config=KvCacheConfig(enable_block_reuse=False,
                                            free_gpu_memory_fraction=0.4))

    sampling_params = SamplingParams(max_tokens=20)
    llm.generate(prompts, sampling_params)
