# How to Change KV Cache Behavior

Set KV cache behavior by providing the optional ```kv_cache_config argument``` when you create the LLM engine. Consider the quickstart example found in ```examples/pytorch/quickstart.py```:

```python
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

This example runs with default KV cache properties. The default value for `free_gpu_memory_fraction` is 0.9, which means TensorRT-LLM tries to allocate 90% of free GPU memory (after loading weights) for KV cache. Depending on your use case, this allocation can be too aggressive. You can reduce this value to 0.7 by adding the following lines to the quickstart example:

```python
from tensorrt_llm.llmapi import KvCacheConfig
kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7)
llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', kv_cache_config=kv_cache_config)
```

You can also set properties after you create ```KvCacheConfig```. For example:

```python
kv_cache_config = KvCacheConfig()
kv_cache_config.enable_block_reuse = False
llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', kv_cache_config=kv_cache_config)
```

This code disables block reuse for the quick start example.
