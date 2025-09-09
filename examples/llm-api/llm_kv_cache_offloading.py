from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig


def main():

    prompt_a = (
        "Given the following question and four candidate answers (A, B, C and D), choose the best answer."
        "The following excerpt is from a pamphlet.\nYou will do me the justice to remember, "
        "that I have always strenuously supported the Right of every man to his own opinion"
    )

    prompt_b = (
        "Question: This question refers to the following information. Read the following excerpt."
        "The revolutionary seed had penetrated into every country and spread more or less. "
        "It was greatly developed under")

    kv_cache_max_tokens = 256
    kv_cache_page_size = 16
    kv_cache_host_size_in_bytes = 1024**3

    # Offloading Off
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              max_batch_size=1,
              max_seq_len=256,
              kv_cache_config=KvCacheConfig(
                  enable_block_reuse=True,
                  max_tokens=kv_cache_max_tokens,
                  tokens_per_block=kv_cache_page_size))
    # prompt_a occupies kv cache pool
    output_a = llm.generate(prompt_a)
    print(output_a.prompt)

    # since max_batch_size=1, prompt_b clears and update kv cache
    output_b = llm.generate(prompt_b)
    print(output_b.prompt)

    # prompt_a clears and update kv cache again
    # no kv cache reuse happens
    output_a = llm.generate(prompt_a)
    print(output_a.prompt)

    # prompt_b clears and update kv cache again
    # no kv cache reuse happens
    output_b = llm.generate(prompt_b)
    print(output_b.prompt)

    llm.shutdown()

    # Offloading On
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              max_batch_size=1,
              max_seq_len=256,
              kv_cache_config=KvCacheConfig(
                  enable_block_reuse=True,
                  max_tokens=kv_cache_max_tokens,
                  tokens_per_block=kv_cache_page_size,
                  host_cache_size=kv_cache_host_size_in_bytes))
    # prompt_a occupies kv cache pool
    output_a = llm.generate(prompt_a)
    print(output_a.prompt)

    # since max_batch_size=1, and offloading is enabled, kv cache of prompt_a will be offloaded to host memory.
    # kv cache of prompt_b keeps in device memory.
    output_b = llm.generate(prompt_b)
    print(output_b.prompt)

    # kv cache of prompt_a will be onboarded to device memory, kv cache of prompt_b will be offloaded to host memory.
    # kv cache of prompt_a will be reused.
    output_a = llm.generate(prompt_a)
    print(output_a.prompt)

    # kv cache of prompt_b will be onboarded to device memory, kv cache of prompt_a will be offloaded to host memory.
    # kv cache of prompt_b will be reused.
    output_b = llm.generate(prompt_b)
    print(output_b.prompt)

    llm.shutdown()


if __name__ == "__main__":
    main()
