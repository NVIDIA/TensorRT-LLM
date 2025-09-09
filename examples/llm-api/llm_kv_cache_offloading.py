import time

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig


def main():

    prompt_a = (
        "Given the following question and four candidate answers (A, B, C and D), choose the best answer."
        "The following excerpt is from a pamphlet. You will do me the justice to remember, "
    )

    prompt_b = (
        "Question: This question refers to the following information. Read the following excerpt."
        "The revolutionary seed had penetrated into every country and spread more or less. "
    )
    max_batch_size = 1
    max_seq_len = 512

    kv_cache_free_gpu_memory_fraction = 0.001
    kv_cache_page_size = 16
    kv_cache_host_size_in_bytes = 1024**3

    # Offloading Off
    print("\n ======  Offloading Off ======  \n")
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              max_batch_size=max_batch_size,
              max_seq_len=max_seq_len,
              kv_cache_config=KvCacheConfig(
                  enable_block_reuse=True,
                  free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
                  tokens_per_block=kv_cache_page_size))
    # prompt_a occupies kv cache pool
    output_a = llm.generate(prompt_a)
    print(
        f"Prompt: {output_a.prompt!r}, Generated text: {output_a.outputs[0].text!r}"
    )

    # since max_batch_size=1, prompt_b clears and update kv cache
    output_b = llm.generate(prompt_b)
    print(
        f"Prompt: {output_b.prompt!r}, Generated text: {output_b.outputs[0].text!r}"
    )

    # prompt_a clears and update kv cache again
    # no kv cache reuse happens
    output_a = llm.generate(prompt_a)
    print(
        f"Prompt: {output_a.prompt!r}, Generated text: {output_a.outputs[0].text!r}"
    )

    # prompt_b clears and update kv cache again
    # no kv cache reuse happens
    output_b = llm.generate(prompt_b)
    print(
        f"Prompt: {output_b.prompt!r}, Generated text: {output_b.outputs[0].text!r}"
    )

    llm.shutdown()
    time.sleep(5)

    # Offloading On
    print("\n ======  Offloading On ======  \n")
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              max_batch_size=max_batch_size,
              max_seq_len=max_seq_len,
              kv_cache_config=KvCacheConfig(
                  enable_block_reuse=True,
                  free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
                  tokens_per_block=kv_cache_page_size,
                  host_cache_size=kv_cache_host_size_in_bytes))
    # prompt_a occupies kv cache pool
    output_a = llm.generate(prompt_a)
    print(
        f"Prompt: {output_a.prompt!r}, Generated text: {output_a.outputs[0].text!r}"
    )

    # since max_batch_size=1, and offloading is enabled,
    # kv cache of prompt_a will be offloaded to host memory.
    # kv cache of prompt_b keeps in device memory.
    output_b = llm.generate(prompt_b)
    print(
        f"Prompt: {output_b.prompt!r}, Generated text: {output_b.outputs[0].text!r}"
    )

    # kv cache of prompt_a will be onboarded to device memory,
    # kv cache of prompt_b will be offloaded to host memory.
    # kv cache of prompt_a will be reused.
    output_a = llm.generate(prompt_a)
    print(
        f"Prompt: {output_a.prompt!r}, Generated text: {output_a.outputs[0].text!r}"
    )

    # kv cache of prompt_b will be onboarded to device memory,
    # kv cache of prompt_a will be offloaded to host memory.
    # kv cache of prompt_b will be reused.
    output_b = llm.generate(prompt_b)
    print(
        f"Prompt: {output_b.prompt!r}, Generated text: {output_b.outputs[0].text!r}"
    )

    llm.shutdown()


if __name__ == "__main__":
    main()
