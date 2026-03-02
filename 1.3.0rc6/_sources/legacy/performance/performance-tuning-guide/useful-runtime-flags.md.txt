(useful-runtime-flags)=

# Useful Runtime Options

This part summarizes the runtime configuration knobs that can be tweaked to
enhance the performance of already built engines. As compared to previous examples where
 the LLM-API was used to build and save an engine but not to process any requests,
runtime knobs would be specified when you are using the LLM-API to actually run inference
like in the [LLM-API end-to-end example](./benchmarking-default-performance.md#before-you-begin-tensorrt-llm-llm-api)


## Capacity Scheduler Policy

TensorRT-LLM currently supports three batch scheduler policies: `GUARANTEED_NO_EVICT` (default),
`MAX_UTILIZATION` and `STATIC_BATCH`.

The scheduling policy can be set to `MAX_UTILIZATION` to pack as many
requests as possible at each iteration of the forward loop, when in-flight
sequence batching is enabled. It maximizes the utilization of the GPUs by
aggressively scheduling requests at the risk of having to pause requests if the
KV cache size limit is reached.

For a more conservative approach with respect to the KV cache limitations in
terms of memory allocation, `CapacitySchedulerPolicy` should be set to
`GUARANTEED_NO_EVICT` to guarantee that a started request is never paused.

If the goal is to maximizes the throughput, users should try `MAX_UTILIZATION`.
However, they need to keep in mind that it may have a negative impact on
latency if requests have to be paused.

`STATIC_BATCH` is a legacy mode and is not recommended for production usage.

To switch the capacity scheduler policy from the default of `GUARANTEED_NO_EVICT` to `MAX_UTILIZATION`
you would modify the [LLM-API end-to-end example](./benchmarking-default-performance.md#before-you-begin-tensorrt-llm-llm-api) to be:

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.bindings.executor import SchedulerConfig, CapacitySchedulerPolicy


def main():
    prompts = [
        "Hello, I am",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    scheduler_config = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION
    )

    llm  =  LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
    scheduler_config=scheduler_config
    )

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    main()
```

## Context Chunking Policy

As discussed [previously](tuning-max-batch-size-and-max-num-tokens.md#revisiting-paged-context-attention-and-context-chunking) context chunking will increase the chance of batch processing between
the context and the generation phase, thereby balancing the calculation amount
of each iteration and typically increasing throughput.

TensorRT-LLM currently supports two context chunking policies: `FIRST_COME_FIRST_SERVED` (default) which would prioritize scheduling all the context chunks of a request that comes in first,
 and `EQUAL_PROGRESS` which schedules context chunks from all requests before scheduling the next chunk of any request.

`FIRST_COME_FIRST_SERVED` should achieve overall better performance, while
`EQUAL_PROGRESS` can be helpful in theory to make sure time to first token (TTFT)
for most requests are relatively similar.

To switch the context chunking policy from the default of `FIRST_COME_FIRST_SERVED` to `EQUAL_PROGRESS`
you would modify the [LLM-API end-to-end example](./benchmarking-default-performance.md#before-you-begin-tensorrt-llm-llm-api) to be:

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.bindings.executor import SchedulerConfig, ContextChunkingPolicy


def main():
    prompts = [
        "Hello, I am",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    scheduler_config = SchedulerConfig(
        context_chunking_policy=ContextChunkingPolicy.EQUAL_PROGRESS
    )

    llm  =  LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
    scheduler_config=scheduler_config
    )

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    main()
```

## Max Tokens in Paged KV Cache and KV Cache Free GPU Memory Fraction

The `max_tokens_in_paged_kv_cache` and `kv_cache_free_gpu_mem_fraction`
parameters can be used to control the maximum number of tokens handled by the
KV cache manager. Setting them properly helps better control the amount of
available memory for the KV cache manager during inference. Keeping in mind
that increasing the amount of memory available to the KV cache manager tends to
translate to a higher achievable throughput.

The `max_tokens_in_paged_kv_cache` flag directly sets the maximum number of
tokens in the KV cache manager. When left unset, that value will be computed
based on the `kv_cache_free_gpu_mem_fraction` setting.

The `kv_cache_free_gpu_mem_fraction` is a floating-point number between `0.0`
and `1.0` that indicates the maximum fraction of GPU memory (after loading the
model) that will be used for the KV cache. The default value is `0.90` and
means that 90% of the free GPU memory will be used to save tokens in the KV
cache. Based on that value, TensorRT-LLM can determine the maximum number of
tokens in the KV cache manager.

When both parameters are set, the maximum number of tokens in the KV cache
manager will be set to the smaller value between `max_tokens_in_paged_kv_cache`
and the value computed from the amount of memory available for the KV cache.

Unless users clearly know the maximum number of tokens in the KV cache needed
by the model, it is recommended to leave `max_tokens_in_paged_kv_cache` unset.
For `kv_cache_free_gpu_mem_fraction`, if no other programs are executed on the
same GPU, it is recommended to test with a as high value as `0.95` to target a
high throughput. Note that the `kv_cache_free_gpu_mem_fraction` parameter
cannot be set to `1.0` because some amount of memory has to be reserved for
inputs and outputs.

To set `kv_cache_free_gpu_mem_fraction` you would modify the [LLM-API end-to-end example](./benchmarking-default-performance.md#before-you-begin-tensorrt-llm-llm-api) to be:

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.bindings.executor import KvCacheConfig


def main():
    prompts = [
        "Hello, I am",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.95)

    llm  =  LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=8,
    kv_cache_config=kv_cache_config
    )

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    main()
```
If you wanted to set `max_tokens_in_paged_kv_cache` instead, you would replace `free_gpu_memory_fraction` with `max_tokens` and specify the number.

```python
    kv_cache_config = KvCacheConfig(max_tokens=<number of tokens>)
```


## Maximum Attention Window Size

The `max_attention_window_size` flag sets the maximum number of tokens that are
attended to in order to generate one token when using techniques like sliding window
attention. See this
[Document](../../advanced/gpt-attention.md#sliding-window-attention-cyclic-rolling-buffer-kv-cache)
for more details. It defaults to the maximum sequence length
(`max_seq_len` when building the engine), which means
that the feature is disabled by default.

When set to a smaller value than `max_seq_len` (during
engine build), only the KV cache of the last `max_attention_window_size` tokens
will be stored. If the input sequence length at runtime exceeds the
`max_attention_window_size` value, the accuracy may start dropping, but the
runtime performance will be better (due to the reduction in terms of
computations and GPU memory allocation). Users can modify that value to
increase runtime performance at the expense of reduced accuracy.

Just like [`kv_cache_free_gpu_mem_fraction`](./useful-runtime-flags.md#max-tokens-in-paged-kv-cache-and-kv-cache-free-gpu-memory-fraction), `max_attention_window_size` can be specified in the LLM-API
via `KVCacheConfig`. To specify `max_attention_window_size` you would instantiate `KVCacheConfig` like so

```python
    kv_cache_config = KvCacheConfig(max_attention_window=<number of tokens>)
```
