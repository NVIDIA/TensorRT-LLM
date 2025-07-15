### Generate Text Using Eagle Decoding

from tensorrt_llm import SamplingParams
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import EagleDecodingConfig, KvCacheConfig


def main():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]
    # The end user can customize the sampling configuration with the SamplingParams class
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # The end user can customize the kv cache configuration with the KVCache class
    kv_cache_config = KvCacheConfig(enable_block_reuse=True)

    llm_kwargs = {}

    model = "lmsys/vicuna-7b-v1.3"

    # The end user can customize the eagle decoding configuration by specifying the
    # speculative_model_dir, max_draft_len, num_eagle_layers, max_non_leaves_per_layer, eagle_choices
    # greedy_sampling,posterior_threshold, use_dynamic_tree and dynamic_tree_max_topK
    # with the EagleDecodingConfig class

    speculative_config = EagleDecodingConfig(
        speculative_model_dir="yuhuili/EAGLE-Vicuna-7B-v1.3",
        max_draft_len=63,
        num_eagle_layers=4,
        max_non_leaves_per_layer=10,
                            eagle_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], \
                                            [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], \
                                            [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], \
                                            [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], \
                                            [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], \
                                            [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]
    )

    llm = LLM(model=model,
              kv_cache_config=kv_cache_config,
              speculative_config=speculative_config,
              max_batch_size=1,
              max_seq_len=1024,
              **llm_kwargs)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    main()
