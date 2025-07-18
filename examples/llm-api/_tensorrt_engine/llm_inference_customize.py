### Generate text with customization
import tempfile

from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import BuildConfig, KvCacheConfig, SamplingParams


def main():
    # The end user can customize the build configuration with the build_config class and other arguments borrowed from the lower-level APIs
    build_config = BuildConfig()
    build_config.max_batch_size = 128
    build_config.max_num_tokens = 2048

    build_config.max_beam_width = 4

    # Model could accept HF model name or a path to local HF model.

    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        build_config=build_config,
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=0.8
        ),  # Similar to `build_config`, you can also customize the runtime configuration with the `kv_cache_config`, `runtime_config`, `peft_cache_config` or \
        # other arguments borrowed from the lower-level APIs.
    )

    # You can save the engine to disk and load it back later, the LLM class can accept either a HF model or a TRT-LLM engine.
    llm.save(tempfile.mkdtemp())

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    # With SamplingParams, you can customize the sampling strategy, such as beam search, temperature, and so on.
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     n=4,
                                     use_beam_search=True)

    for output in llm.generate(prompts, sampling_params):
        print(
            f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
        )

    # Got output like
    # Prompt: 'Hello, my name is', Generated text: '\n\nJane Smith. I am a student pursuing my degree in Computer Science at [university]. I enjoy learning new things, especially technology and programming'
    # Prompt: 'The capital of France is', Generated text: 'Paris.'
    # Prompt: 'The future of AI is', Generated text: 'an exciting time for us. We are constantly researching, developing, and improving our platform to create the most advanced and efficient model available. We are'


if __name__ == '__main__':
    main()
