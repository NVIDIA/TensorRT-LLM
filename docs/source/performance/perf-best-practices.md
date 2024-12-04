(perf-best-practice)=

# Best Practices for Tuning the Performance of TensorRT-LLM

This document provides some best practices for tuning the performance of TensorRT-LLM.

## How To Measure Performance?

TensorRT-LLM can be benchmarked using the
[C++](https://github.com/NVIDIA/TensorRT-LLM/blob/main/benchmarks/cpp/README.md) tools. We are actively developing `trtllm-bench` command, which is going to be the recommended way of benchmarking TensorRT-LLM.

For detailed performance data and
the steps to reproduce those results, see
this [Document](https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html).
The [TensorRT-LLM backend](https://github.com/triton-inference-server/tensorrtllm_backend)
can also be used to measure the performance of TensorRT-LLM for online serving.

## Build Options to Optimize the Performance of TensorRT-LLM Models

This part summarizes how to build engines to enhance the performance of the
runtime. The following options have reasonable default values but for some of them,
it's possible that tuning is needed to get the peak numbers.

***Note that some of those features and how to enable them may change in the future.***

### `max_batch_size`, `max_seq_len` and `max_num_tokens`

<p align="center">
    <img src="https://github.com/NVIDIA/TensorRT-LLM/blob/rel/docs/source/media/max_bs_toks_len.svg?raw=true" alt="Explain `max_batch_size`, `max_seq_len` and `max_num_tokens`" width="30%" height="auto">
</p>

Regarding the impacts of those three arguments to the GPU memory usage, please refer to [memory.md](https://nvidia.github.io/TensorRT-LLM/reference/memory.html)

#### `max_batch_size`

`max_batch_size` defines the maximum number of requests that the engine can handle.​

It controls the maximum number of requests that can be scheduled at runtime.

Set high enough `max_batch_size` when building the engine so that it does not become the bottleneck of the throughput, and use runtime `max_batch_size` to tune it without re-building the engine if you want to get better user throughput or lower latency.

#### `max_seq_len`

`max_seq_len` defines the maximum sequence length of single request​

Starting from TensorRT-LLM v0.11, when `--remove_input_padding` and `--context_fmha` are enabled, `max_seq_len` can replace `max_input_len` and `max_output_len`, and is set to `max_position_embeddings` by default.

Use default `max_seq_len` (which is `max_position_embeddings`), no need to tune it unless you are very sure what max sequence lengths would be on your workloads. If the GPU memory is so limited that it cannot make sure even one request to reach `max_seq_len`, you'll need to reduce it.

#### `max_num_tokens`

`max_num_tokens` defines the maximum number of batched input tokens after padding is removed in each batch.​

`max_num_tokens` is set to 8192 by default starting from v0.11, you can tune it using the runtime `max_num_tokens` without re-buliding the engine. It is recommended to tune `--max_num_tokens` for better performance.

The maximum number of tokens equals will not take effects when input padding is
not removed. When input padding is removed (see [Remove Input
Padding](#remove-input-padding)), the tokens from different sequences are
packed together and the maximum number of the tokens can be set to a different
(lower) value, which by default to be 8192.

There are two aspects that must be considered. Firstly, some input sequences
will be shorter than the maximum input length. Secondly, when in-flight
sequence batching is enabled, requests in context phase will be executed with
requests in generation phase. Those latter requests produce a lot fewer tokens
than `max_input_len` (at most, `beam_width` tokens).

Using a more realistic value for `max_num_tokens` allows TensorRT-LLM to
allocate more memory to store the KV cache and execute more requests together.
It leads to an increased efficiency.

Increasing `max_num_tokens` appropriately will be beneficial to performance.
When increasing `--max_num_tokens` to some point, GPU utilization will plateau,
going beyond that saturation point may hurt both first token latency as well as
total end-to-end latency.

See also [chunked context](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html#chunked-context).

### Multiple profiles

`--multiple_profiles` enables multiple TensorRT optimization profiles in the
built engines, it will benefits the performance especially when GEMM plugin is
disabled, because more optimization profiles help TensorRT have more chances to
select better kernels.

Note: This feature increases engine build time but no other adverse effects are expected.

#### FP8 Context Fused Multi-Head Attention

`--use_fp8_context_fmha` enables FP8 Context fused multi-head attention. We
recommend enabling this when fp8 quantization is used to improve the context phase
attention performance. Note that only NVIDIA Hopper architecture is supported.

### GPT Attention Plugin and Context Fused Multi-Head Attention

The GPT attention plugin and fused  multi-head attention kernel are enabled by
default. For the context phase, use the `--gpt_attention_plugin`
and `--context_fmha` arguments with `trtllm-build` to control.

The TensorRT-LLM GPT attention plugin uses efficient kernels and enables an
in-place update of the KV cache. It results in reduced memory consumption as
well as the removal of unneeded memory copy operations (compared with the
implementation that uses the `concat` operator to update the KV cache).

Enabling the fused multi-head attention, during the context phase, will trigger
a kernel that performs the MHA/MQA/GQA block using a single kernel, for more
details, see this [Document](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html#context-phase).

### Remove Input Padding

The remove input padding feature is enabled by default, the `--remove_input_padding`
argument in `trtllm-build` is used to control it.

When input padding is removed, the different tokens are packed together. It
reduces both the amount of computations and memory consumption. For more details, see
this [Document](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html#padded-and-packed-tensors).

### Paged KV Cache

Paged KV cache is enabled by default, the `--paged_kv_cache` argument in
`trtllm-build` is used to control it.

The paged KV cache helps manage memory for the KV cache more efficiently (see
this [Document](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html#paged-kv-cache)). It usually leads to an
increase in the batch size and an improved efficiency.

### Reduce Norm Fusion

There is an experimental feature called "Reduce Norm Fusion"
available to extend the custom AllReduce functionality. It can be enabled by
using the `--reduce_fusion enable` argument with `trtllm-build` when the
custom AllReduce is already enabled.

This feature aims to fuse the `ResidualAdd`
and `LayerNorm` kernels after `AllReduce` into a single kernel, resulting in
improved end-to-end performance.

Please note that currently, this feature is
only supported for the llama model. It is recommended to enable this feature when the batch size is small and the generation phase time is the dominant factor.

### User Buffer

An experimental feature called "User Buffer" is available to enhance communication performance. It can be enabled by using the `--user_buffer enable` argument with `trtllm-build`.
This feature aims to eliminate extra copies from the local buffer to the shared buffer in the communication kernel, leading to improved end-to-end performance.
This feature must be enabled with `--reduce_fusion enable` and is only supported for the FP8 LLAMA model.

### Embedding Parallelism, Embedding Sharing, and Look-Up Plugin

The embedding parallelism feature enables the sharding of the embedding table
across multiple GPUs, so that the memory usage could be reduced and the
throughput improved. The embedding sharing feature enables the sharing of the
embedding table between `look_up` and `lm_head` layers to reduced memory usage.

It is recommended to enable embedding parallelism to improve throughput with `--use_parallel_embedding` and `--embedding_sharding_dim` in `convert_checkpoint.py`.

Embedding sharing is by default enabled if following conditions are met:
1. `look_up` and `lm_head` layers have identical weights.
2. `--gemm_plugin` is not used when building the engine.
3. For tensor parallelism cases, `-embedding_sharding_dim 0` must be set. In other words, we must enable embedding parallelism along the vocab dimension,

See those [Examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt#embedding-parallelism) for details.

### Horizontal Fusion in Gated-MLP

Horizontal fusion in Gated-MLP combines two Matmul operations into a single one
followed by a separate SwiGLU kernel. It can effectively reduce latency.
This feature is enabled by default.

### GEMM Plugin

The GEMM plugin utilizes NVIDIA cuBLASLt to perform GEMM operations. On FP16 and
BF16, it's recommended to be enabled for better performance and smaller GPU
memory usage. On FP8, it's recommended to be disabled.

#### FP8 GEMM Plugin for Small Batch Size Performance Optimization

FP8 gemm plugin is an experimental feature aimed to improve performance in
small-batch-size cases(e.g. BS<=4) and can be enabled by `--gemm_plugin fp8`
when building FP8 models. Although inputs with larger batch size can be correctly
inferenced, the performance may decrease as batch size grows. Therefore, this
feature is only recommended for latency reduction in small-batch-size scenarios
currently.

#### GEMM + SwiGLU Fusion in Gated-MLP

The GEMM + SwiGLU fusion in Gated-MLP combines two Matmul operations and one SwiGLU operation into a single kernel. Currently this is only supported for FP8 precision on Hopper. While this fusion improves performance, it can slightly reduce accuracy in FP8 PTQ because one quantization scaling factor is discarded.

We recommend enabling this feature for large models running on Hopper with FP8 precision. Use the following `trtllm-build` arguments to enable it:

* For large models: `--use_fused_mlp=enable --gemm_swiglu_plugin=fp8`
* For small batch sizes: `--use_fused_mlp=enable --low_latency_gemm_swiglu_plugin=fp8` to improve latency.

We do not recommend enabling this feature for very small workloads or if the
accuracy loss is unacceptable.

### BERT Attention Plugin and Context Fused Multi-Head Attention

BERT attention plugin and context fused multi-head attention are both
recommended for the BERT model. They are enabled by default using the
`--bert_attention_plugin` and `--context_fmha` arguments with
`trtllm-build`.

## Runtime Options to Optimize the Performance of TensorRT-LLM Models

This part summarizes the runtime configuration knobs that can be tweaked to
enhance the performance of already built engines. Note that currently the
configurations can be modified using the
[Executor API](https://nvidia.github.io/TensorRT-LLM/advanced/executor.html#executor-api)
as well as the
[TensorRT-LLM backend](https://github.com/triton-inference-server/tensorrtllm_backend).

### Capacity Scheduler Policy

There currently are three batch scheduler policies: `GUARANTEED_NO_EVICT` (default),
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

### Context Chunking Policy

Context chunking will increase the chance of batch processing between
the context and the generation phase, thereby balancing the calculation amount
of each iteration and increasing throughput.

There currently are two context chunking policies: `FIRST_COME_FIRST_SERVED` (default)
and `EQUAL_PROGRESS`.

`FIRST_COME_FIRST_SERVED` should achieve overall better performance, while
`EQUAL_PROGRESS` can be helpful in theory to make sure time to first token (TTFT)
for most requests are relatively similar.

### Batching Type

The batching type can be set to `INFLIGHT` (default) and `STATIC`.
It is recommended to use `INFLIGHT` to increase throughput and reduce latency.

### Max Tokens in Paged KV Cache and KV Cache Free GPU Memory Fraction

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

### Maximum Attention Window Size

The `max_attention_window_size` flag sets the maximum number of tokens that are
attended to in order to generate one token when using techniques like sliding window
attention. See this
[Document](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.md#sliding-window-attention-cyclic-rolling-buffer-kv-cache)
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
