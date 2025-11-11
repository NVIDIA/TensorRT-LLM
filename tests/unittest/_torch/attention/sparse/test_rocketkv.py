import json
import os

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, RocketSparseAttentionConfig


@pytest.mark.parametrize("backend", ["pytorch"])
@pytest.mark.parametrize("model_name",
                         ["llama-3.1-model/Llama-3.1-8B-Instruct"])
@pytest.mark.parametrize("attention_backend", ["VANILLA", "TRTLLM"])
def test_model(backend, model_name, attention_backend):
    model_dir = str(llm_models_root() / model_name)
    max_batch_size = 16
    max_output_tokens = 128
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7,
                                    enable_block_reuse=False)

    sparse_attention_config = RocketSparseAttentionConfig(
        window_size=32,
        kernel_size=63,
        prompt_budget=2048,
    )

    llm = LLM(
        model=model_dir,
        backend=backend,
        kv_cache_config=kv_cache_config,
        attn_backend=attention_backend,
        sparse_attention_config=sparse_attention_config,
        max_batch_size=max_batch_size,
        max_seq_len=8192,
        max_num_tokens=8192,
        cuda_graph_config=
        None,  # sparse attention does not support cuda graph now
    )

    inputs, references = [], []
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(os.path.dirname(
        os.path.dirname(current_file)))
    input_file = f'{current_dir}/multi_gpu/test_star_attention_input.jsonl'
    with open(input_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            inputs.append({
                'prompt':
                sample['input_context'] + sample['input_query'],
            })
            references.append(sample['outputs'][0])

    with llm:
        outputs = llm.generate(
            inputs,
            use_tqdm=True,
            sampling_params=SamplingParams(add_special_tokens=False,
                                           max_tokens=max_output_tokens,
                                           temperature=0.8,
                                           top_p=0.95),
        )

    count = 0
    for ref, ret in zip(references, outputs):
        print(f"ret: {ret.outputs[0].text}")
        print(f"ref: {ref}")
        if ref not in ret.outputs[0].text:
            print(f'reference {ref} is not in the output {ret.outputs[0].text}')
        else:
            count = count + 1
    acc = count / len(outputs)

    assert acc >= 0.9, 'accuracy test of rocketkv sparse attention failed'


if __name__ == '__main__':
    test_model("pytorch", "llama-3.1-model/Llama-3.1-8B-Instruct", "VANILLA")
    test_model("pytorch", "llama-3.1-model/Llama-3.1-8B-Instruct", "TRTLLM")
