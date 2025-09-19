### :title KV Cache Offloading
### :order 6
### :section Customization
'''
This script demonstrates the effectiveness of KV cache host offloading in TensorRT-LLM.

**Scenario:**
The script simulates a scenario where the GPU's KV cache is severely limited,
while multiple requests with recurring prompts (like system prompts) are processed.

1.  **Constrained GPU Cache:** The GPU KV cache is configured to be very small,
    only large enough to hold the state for a single request.
2.  **Alternating Prompts:** Four requests are sent sequentially (batch size of 1)
    with two distinct prompts in an A, B, A, B pattern.
3.  **Cache Eviction:** Due to the small GPU cache, processing prompt B will
    force the eviction of the cache generated for prompt A.

**Demonstration:**

* **Without Offloading (Default):**
    - When the first prompt 'A' is processed, its KV cache is stored on the GPU.
    - When prompt 'B' arrives, the cache manager needs space and discards the cache for 'A'.
    - When prompt 'A' is sent again, its cache must be recomputed from scratch.
    - **Expected Outcome:** The log will show `reused blocks: 0` and `cache hit rate: 0`.

* **With Offloading (`--enable_offloading`):**
    - When prompt 'B' arrives, the cache for 'A' is not discarded but is instead
      *offloaded* from the fast GPU VRAM to the slower (but larger) host CPU RAM.
    - When prompt 'A' is sent again, its KV cache is loaded back from host RAM
      to the GPU, which is significantly faster than recomputing it.
    - **Expected Outcome:** The log will show positive values for `reused blocks`
      and a non-zero `cache hit rate`, confirming that the cache was successfully
      reused from the host.

**How to Run & Verify:**

1.  **Without Offloading:**
    ```bash
    TLLM_LOG_LEVEL=DEBUG python llm_kv_cache_offloading.py 2>&1 | tee offloading_disabled.log
    ```
    (Check the log for zero reuse)

2.  **With Offloading:**
    ```bash
    TLLM_LOG_LEVEL=DEBUG python llm_kv_cache_offloading.py --enable_offloading 2>&1 | tee offloading_enabled.log
    ```
    (Check the log for non-zero reuse)
'''

import argparse

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig


def main(args):
    # Define two distinct prompts to simulate different requests or system prompts.
    prompt_a = (
        "Returns the per-iterations statistics computed since last call to this method. "
        "Contains at most iter_stats_max_iterations iterations.")
    prompt_b = ("Use for skipping decoding step for non generation model, "
                "and return the batch_output (such as mm_embeddings)")

    # Use a batch size of 1 to process requests sequentially, making the cache
    # eviction and reuse cycle easy to observe.
    max_batch_size = 1
    max_seq_len = 256

    # --- KV Cache Configuration ---
    # Set a small GPU KV cache size (in number of tokens). This is crucial for the demo,
    # as it's only large enough to hold the KV cache for a single request.
    kv_cache_max_tokens = 256
    # Define the size of a single cache block.
    kv_cache_page_size = 16
    # Enable a 1 GB host cache if offloading is requested, otherwise disable it (size 0).
    # This is the key toggle for the experiment.
    kv_cache_host_size = 1024**3 if args.enable_offloading else 0

    sampling_params = SamplingParams(max_tokens=max_seq_len)

    llm = LLM(
        model="Qwen/Qwen3-8B",
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=True,  # Enable reuse of cached blocks
            max_tokens=kv_cache_max_tokens,  # Max tokens in GPU cache
            tokens_per_block=kv_cache_page_size,
            host_cache_size=kv_cache_host_size  # Host cache size for offloading
        ))

    # Process four requests sequentially using two distinct prompts (A, B, A, B).
    # This pattern is designed to showcase the cache eviction and reuse behavior.
    print("--- First Round ---")
    # 1. Process prompt A. Its cache is stored on the GPU.
    output_a = llm.generate(prompt_a, sampling_params)
    print(
        f"Prompt: {output_a.prompt!r}, Generated text: {output_a.outputs[0].text!r}"
    )
    # 2. Process prompt B. Its cache replaces/offloads A's cache.
    output_b = llm.generate(prompt_b, sampling_params)
    print(
        f"Prompt: {output_b.prompt!r}, Generated text: {output_b.outputs[0].text!r}"
    )

    print("\n--- Second Round ---")
    # 3. Process prompt A again.
    #    - Without offloading: Must recompute from scratch.
    #    - With offloading: Recovers cache from host RAM.
    output_a = llm.generate(prompt_a, sampling_params)
    print(
        f"Prompt: {output_a.prompt!r}, Generated text: {output_a.outputs[0].text!r}"
    )
    # 4. Process prompt B again.
    #    - Without offloading: Must recompute from scratch.
    #    - With offloading: Recovers cache from host RAM.
    output_b = llm.generate(prompt_b, sampling_params)
    print(
        f"Prompt: {output_b.prompt!r}, Generated text: {output_b.outputs[0].text!r}"
    )

    llm.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "A script to demonstrate the effectiveness of KV cache host offloading."
    )
    parser.add_argument('--enable_offloading',
                        action='store_true',
                        help='Enable host RAM for KV cache offloading.')
    args = parser.parse_args()
    main(args)
