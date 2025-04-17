### Generate Text Using Prompt-Lookup Decoding
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (LLM, BuildConfig, KvCacheConfig,
                                 PromptLookupConfig, SamplingParams)


def main():

    # The end user can customize the build configuration with the build_config class
    build_config = BuildConfig()
    build_config.max_batch_size = 32

    # The configuration for Prompt-Lookup decoding
    prompt_lookup_config = PromptLookupConfig(prompt_lookup_num_tokens=4,
                                              max_matching_ngram_size=2,
                                              candidate_set_size=2)

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              kv_cache_config=kv_cache_config,
              build_config=build_config,
              speculative_config=prompt_lookup_config)

    prompt = "NVIDIA is a great company because"
    print(f"Prompt: {prompt!r}")

    sampling_params = SamplingParams(prompt_lookup_config=prompt_lookup_config)

    output = llm.generate(prompt, sampling_params=sampling_params)
    print(output)

    #Output should be similar to:
    # Prompt: 'NVIDIA is a great company because'
    #RequestOutput(request_id=2, prompt='NVIDIA is a great company because', prompt_token_ids=[1, 405, 13044, 10764, 338, 263, 2107, 5001, 1363], outputs=[CompletionOutput(index=0, text='they are always pushing the envelope. They are always trying to make the best graphics cards and the best processors. They are always trying to make the best', token_ids=[896, 526, 2337, 27556, 278, 427, 21367, 29889, 2688, 526, 2337, 1811, 304, 1207, 278, 1900, 18533, 15889, 322, 278, 1900, 1889, 943, 29889, 2688, 526, 2337, 1811, 304, 1207, 278, 1900], cumulative_logprob=None, logprobs=[], finish_reason='length', stop_reason=None, generation_logits=None)], finished=True)


if __name__ == '__main__':
    main()
