# How To Change KV Cache Behavior

KV cache behavior is set by providing the optional argument ```kv_cache_config``` when LLM engine is created. Consider the quickstart example (found in examples/pytorch/quickstart.py):

```
from tensorrt_llm import LLM, SamplingParams


def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(max_tokens=32)

    llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    main()
```

This example runs with default KV cache properties. The default for ```free_gpu_memory_fraction``` is 0.9, which means TRTLLM will try to allocate 90% of free GPU memory for KV cache. Depending on your system, this may be too aggressive, so you decide to dial that back to 0.7. This is done by adding the following lines to the quickstart example:

```
kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7)
llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', kv_cache_config=kv_cache_config)
```

You can also set properties after you create KvCacheConfig instance:

```
kv_cache_config = KvCacheConfig()
kv_cache_config.enable_block_reuse = False
llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', kv_cache_config=kv_cache_config)
```

will disable block reuse for the quickstart example.


