import argparse

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig


def main(args):
    prompt_a = (
        "Returns the per-iterations statistics computed since last call to this method. "
        "Contains at most iter_stats_max_iterations iterations.")
    prompt_b = ("Use for skipping decoding step for non generation model, "
                "and return the batch_output (such as mm_embeddings)")
    max_batch_size = 1
    max_seq_len = 256

    kv_cache_max_tokens = 256
    kv_cache_page_size = 16
    kv_cache_host_size = 1024**3 if args.enable_offloading else 0

    sampling_params = SamplingParams(max_tokens=max_seq_len)

    llm = LLM(model="Qwen/Qwen3-8B",
              max_batch_size=max_batch_size,
              max_seq_len=max_seq_len,
              kv_cache_config=KvCacheConfig(enable_block_reuse=True,
                                            max_tokens=kv_cache_max_tokens,
                                            tokens_per_block=kv_cache_page_size,
                                            host_cache_size=kv_cache_host_size))
    '''
    prompt_a occupies kv cache pool
    '''
    output_a = llm.generate(prompt_a, sampling_params)
    print(
        f"Prompt: {output_a.prompt!r}, Generated text: {output_a.outputs[0].text!r}"
    )
    '''
    since max_batch_size=1,
    if not enable_offloading:
        prompt_b clears and updates kv cache
    else:
        kv cache of prompt_a will be offloaded to host memory.
        kv cache of prompt_b will be in device memory.
    '''
    output_b = llm.generate(prompt_b, sampling_params)
    print(
        f"Prompt: {output_b.prompt!r}, Generated text: {output_b.outputs[0].text!r}"
    )
    '''
    if not enable_offloading:
        prompt_a clears and updates kv cache again, no kv cache reuse happens
    else:
        kv cache of prompt_a will be onboarded to device memory, and be reused.
        kv cache of prompt_b will be offloaded to host memory.
    '''
    output_a = llm.generate(prompt_a, sampling_params)
    print(
        f"Prompt: {output_a.prompt!r}, Generated text: {output_a.outputs[0].text!r}"
    )
    '''
    if not enable_offloading:
        prompt_b clears and updates kv cache again, no kv cache reuse happens
    else:
        kv cache of prompt_b will be onboarded to device memory, and be reused.
        kv cache of prompt_a will be offloaded to host memory.
    '''
    output_b = llm.generate(prompt_b, sampling_params)
    print(
        f"Prompt: {output_b.prompt!r}, Generated text: {output_b.outputs[0].text!r}"
    )

    llm.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_offloading',
                        default=False,
                        action='store_true')
    args = parser.parse_args()
    main(args)
