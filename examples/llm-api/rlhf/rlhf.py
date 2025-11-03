import os

import torch
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoModelForCausalLM

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.utils import get_device_uuid
from tensorrt_llm.llmapi import KvCacheConfig


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        orchestrator_type="ray",
        ray_worker_extension_cls="rlhf_utils.WorkerExtension",
        kv_cache_config=kv_cache_config,
        tensor_parallel_size=1,
    )

    # Generate texts from the prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0, return_generation_logits=True)

    for output in llm.generate(prompts, sampling_params):
        print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")

    hf_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ipc_handles = {}
    device_uuid = get_device_uuid(0)
    ipc_handles[device_uuid] = []
    for name, param in hf_model.named_parameters():
        zero_tensor = torch.zeros_like(param).to("cuda:0")
        handle = reduce_tensor(zero_tensor)
        ipc_handles[device_uuid].append((name, handle))

    llm._collective_rpc("update_weights", args=(ipc_handles,))

    weights_updated = llm._collective_rpc("check_weights_updated")
    assert all(weights_updated), f"Not all weights updated: {weights_updated}"

    print("After updating weights to zero, the output is expected to be nonsense:")
    for output in llm.generate(prompts, sampling_params):
        print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")

    del os.environ["CUDA_VISIBLE_DEVICES"]


if __name__ == "__main__":
    main()
