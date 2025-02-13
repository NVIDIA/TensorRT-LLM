### Generate Text Using Eagle Decoding

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (LLM, BuildConfig, EagleDecodingConfig,
                                 KvCacheConfig, SamplingParams)
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode


def main():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # The end user can customize the sampling configuration with the SamplingParams class
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # The end user can customize the build configuration with the BuildConfig class
    build_config = BuildConfig(
        max_batch_size=1,
        max_seq_len=1024,
        max_draft_len=63,
        speculative_decoding_mode=SpeculativeDecodingMode.EAGLE)

    # The end user can customize the kv cache configuration with the KVCache class
    kv_cache_config = KvCacheConfig(enable_block_reuse=True)

    llm_kwargs = {}

    model = "lmsys/vicuna-7b-v1.3"
    speculative_model = "yuhuili/EAGLE-Vicuna-7B-v1.3"

    # The end user can customize the eagle decoding configuration by specifying the
    # num_eagle_layers, max_non_leaves_per_layer, eagle_choices
    # greedy_sampling,posterior_threshold, use_dynamic_tree and dynamic_tree_max_topK
    # with the EagleDecodingConfig class

    speculative_config = EagleDecodingConfig(
        num_eagle_layers=4,
        max_non_leaves_per_layer=10,
                            eagle_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], \
                                            [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], \
                                            [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], \
                                            [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], \
                                            [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], \
                                            [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]
    )
    llm_kwargs = {"speculative_model": speculative_model}

    llm = LLM(model=model,
              build_config=build_config,
              kv_cache_config=kv_cache_config,
              speculative_config=speculative_config,
              **llm_kwargs)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    main()
