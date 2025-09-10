# How to Change Block Priorities

You can change block priority by providing the optional ```kv_cache_retention_config``` argument when you submit a request to the LLM engine. Consider the quick start example found in ```examples/pytorch/quickstart.py```:

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

The blocks from the prompts are stored for reuse with the default priority of 35 on a scale from 1 to 100, where 100 is highest priority and 1 is lowest priority. Assume you know that the first four tokens of each prompt represent a system prompt that should be stored with high priority (100). You can achieve this by providing a KV cache retention config object when you submit the prompts for generation:

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheRetentionConfig


def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(max_tokens=32)

    llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0')

    # Set priority for first 4 prompt tokens to 100. All other tokens set to default (35) priority.
    # This policy never lapses.
    tokenRangeRetentionConfig = KvCacheRetentionConfig.TokenRangeRetentionConfig(0, 4, 100, None)
    kv_cache_retention_config = KvCacheRetentionConfig(
        token_range_retention_configs=[tokenRangeRetentionConfig],
        decode_retention_priority=35, # Set generated tokens to default priority
        decode_duration_ms=None)
    outputs = llm.generate(prompts, sampling_params, kv_cache_retention_config=kv_cache_retention_config)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    main()
```

This example uses a single ```kv_cache_retention_config``` object for all the prompts. You can also provide a list that must have the same length as the list of prompts.
