from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig


def main():
    print("\n=== KV Cache Configuration Example ===")
    print("\n1. KV Cache Configuration:")

    llm_advanced = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       max_batch_size=1,
                       max_seq_len=1024,
                       kv_cache_config=KvCacheConfig(
                           free_gpu_memory_fraction=0.5,
                           enable_block_reuse=True,
                           max_tokens=1024,
                           host_cache_size=1 * 1024 * 1024 * 1024,
                           tokens_per_block=32))

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    outputs = llm_advanced.generate(prompts)
    for i, output in enumerate(outputs):
        print(f"Query {i+1}: {output.prompt}")
        print(f"Answer: {output.outputs[0].text[:100]}...")
        print()


if __name__ == "__main__":
    main()
