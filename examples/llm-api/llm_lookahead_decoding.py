### Generate Text Using Lookahead Decoding
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import (BuildConfig, KvCacheConfig,
                                 LookaheadDecodingConfig, SamplingParams)


def main():

    # The end user can customize the build configuration with the build_config class
    build_config = BuildConfig()
    build_config.max_batch_size = 32

    # The configuration for lookahead decoding
    lookahead_config = LookaheadDecodingConfig(max_window_size=4,
                                               max_ngram_size=4,
                                               max_verification_set_size=4)

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              kv_cache_config=kv_cache_config,
              build_config=build_config,
              speculative_config=lookahead_config)

    prompt = "NVIDIA is a great company because"
    print(f"Prompt: {prompt!r}")

    sampling_params = SamplingParams(lookahead_config=lookahead_config)

    output = llm.generate(prompt, sampling_params=sampling_params)
    print(output)

    #Output should be similar to:
    # Prompt: 'NVIDIA is a great company because'
    #RequestOutput(request_id=2, prompt='NVIDIA is a great company because', prompt_token_ids=[1, 405, 13044, 10764, 338, 263, 2107, 5001, 1363], outputs=[CompletionOutput(index=0, text='they are always pushing the envelope. They are always trying to make the best graphics cards and the best processors. They are always trying to make the best', token_ids=[896, 526, 2337, 27556, 278, 427, 21367, 29889, 2688, 526, 2337, 1811, 304, 1207, 278, 1900, 18533, 15889, 322, 278, 1900, 1889, 943, 29889, 2688, 526, 2337, 1811, 304, 1207, 278, 1900], cumulative_logprob=None, logprobs=[], finish_reason='length', stop_reason=None, generation_logits=None)], finished=True)


if __name__ == '__main__':
    main()
