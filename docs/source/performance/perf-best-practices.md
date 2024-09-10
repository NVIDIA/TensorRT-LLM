(perf-best-practice)=

# Best Practices for Tuning the Performance of TensorRT-LLM

This document provides some best practices for tuning the performance of TensorRT-LLM.

## How To Measure Performance?

TensorRT-LLM can be benchmarked using the included
[C++](https://github.com/NVIDIA/TensorRT-LLM/blob/main/benchmarks/cpp/README.md)
and
[Python](https://github.com/NVIDIA/TensorRT-LLM/blob/main/benchmarks/python/README.md) tools. However, it is *strongly*
recommended to use the C++ benchmarking tool. For detailed performance data and
the steps to reproduce those results, see
this [Document](https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html).
The [TensorRT-LLM backend](https://github.com/triton-inference-server/tensorrtllm_backend)
can also be used to measure the performance of TensorRT-LLM for online serving.

## Build Options to Optimize the Performance of TensorRT-LLM Models

This part summarizes how to build engines to enhance the performance of the
runtime and, for some of them, decrease the engine build time.

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

However, this feature will increase the engine build time.

**Known issue**: We observed that enabling multiple profiles can lead to extra
unexpected GPU memory usage on some cases starting from v0.11. The issue will be
addressed in future releases.

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

#### FP8 Context Fused Multi-Head Attention

`--use_fp8_context_fmha` enables FP8 Context fused multi-head attention, which
is recommended to be enabled when fp8 quantization is used to improve the
performance. Note that only NVIDIA Hopper architecture is supported.

### Remove Input Padding

The remove input padding feature is enabled by default, the `--remove_input_padding`
argument in `trtllm-build` is used to control it.

When input padding is removed, the different tokens are packed together. It
reduces both the amount of computations and memory consumption. For more details, see
this [Document](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.md#padded-and-packed-tensors).

### Paged KV Cache

Paged KV cache is enabled by default, the `--paged_kv_cache` argument in
`trtllm-build` is used to control it.

The paged KV cache helps manage memory for the KV cache more efficiently (see
this [Document](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html#paged-kv-cache)). It usually leads to an
increase in the batch size and an improved efficiency.

### In-flight Sequence Batching

In-flight sequence batching is enabled by default with `trtllm-build`,
which requires that the GPT attention plugin, input padding removal and paged KV
cache are all enabled together.

In-flight sequence batching schedules sequences in context phase together with
sequences in generation phase to increase efficiency and reduce latency, see
this [Document](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html#in-flight-batching) for more details.

### Reduce Norm Fusion

There is an experimental feature called "Reduce Norm Fusion"
available to extend the custom AllReduce functionality. It can be enabled by
using the `--reduce_fusion enable` argument with `trtllm-build` when the
custom AllReduce is already enabled. This feature aims to fuse the ResidualAdd
and LayerNorm kernels after AllReduce into a single kernel, resulting in
improved end-to-end performance. Please note that currently, this feature is
only supported for the llama model. It is recommended to enable this feature when the batch size is small and the generation phase time is the dominant factor.

### Embedding Parallelism, Embedding Sharing, and Look-Up Plugin

The embedding parallelism feature enables the sharding of the embedding table
across multiple GPUs, so that the memory usage could be reduced and the
throughput improved. The embedding sharing feature enables the sharing of the
embedding table between `look_up` and `lm_head` layers.

The look-up plugin implements the embedding sharing feature and is required to
enable the aforementioned features for now (until TensorRT native layers
support embedding sharing).

It is recommended to enable the embedding parallelism and sharing features to
improve throughput. However, the following conditions have to be satisfied:

1. The model shares the embedding table between `look_up` and `lm_head` layers,
2. Both look_up plugin and gemm plugin are enabled,
3. The sharding dimension of the embedding lookup table is set correctly.

To enable the features, use the `--use_parallel_embedding`,
`--use_embedding_sharing`, `--use_lookup_plugin`, `--use_gemm_plugin`
arguments, and set correct dimension to `--embedding_sharding_dim` argument
with `trtllm-build`. See those
[Examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt#embedding-parallelism-and-sharing)
for details.

### Horizontal Fusion in Gated-MLP

Horizontal fusion in Gated-MLP combines two Matmul operations into a single one
followed by a separate SwiGLU kernel. However, for FP8 PTQ, the
downside is slight reduction of accuracy because one of the quantization scaling
factors are discarded.

If both model and batch sizes are large, it is recommended to enable the feature
by using the `--use_fused_mlp=enable` argument with `trtllm-build`. When the workload
is very small, or if you're using FP8 PTQ and the accuracy after enabling it
does not satisfy your requirement, it is not recommended to enable that feature.

### GEMM + SwiGLU Fusion in Gated-MLP

GEMM + SwiGLU fusion in Gated-MLP combines two Matmul operations and one SwiGLU
operation into a single kernel. It only supports FP8 on Hopper now. For FP8 PTQ,
the downside is slight reduction of accuracy because one of the quantization
scaling factors are discarded.

If model is large and you are running it on Hopper with FP8 precision, it is
recommended to enable the feature by using the `--use_fused_mlp=enable --gemm_swiglu_plugin fp8`
argument with `trtllm-build`. When the workload is very small, or the accuracy
after enabling it does not satisfy your requirement, it is not recommended to
enable that feature.

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

### BERT Attention Plugin and Context Fused Multi-Head Attention

BERT attention plugin and context fused multi-head attention are both
recommended for the BERT model. They are enabled by default using the
`--bert_attention_plugin` and `--context_fmha` arguments with
`trtllm-build`.

## Runtime Options to Optimize the Performance of TensorRT-LLM Models

This part summarizes the runtime configuration knobs that can be tweaked to
enhance the performance of already built engines. Note that currently the
configurations can be modified using the
[Batch Manager API](https://nvidia.github.io/TensorRT-LLM/advanced/batch-manager.html#the-batch-manager-api)
as well as the
[TensorRT-LLM backend](https://github.com/triton-inference-server/tensorrtllm_backend).

### GPT Model Type

The GPT model type can be set to `V1`, `inflight_batching` and
`inflight_fused_batching`. It is recommended to use `inflight_fused_batching`
to increase throughput and reduce latency.

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

### Batch Scheduler Policy

There currently are two batch scheduler policies: `MAX_UTILIZATION` and
`GUARANTEED_NO_EVICT`.

As explained in the [GPT Manager Design](https://nvidia.github.io/TensorRT-LLM/advanced/batch-manager.html#gptmanager-design)
section, the scheduling policy can be set to `MAX_UTILIZATION` to pack as many
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

### TensorRT Overlap

***Note that this option is now deprecated and only available with the GptManager API.***

This option allowed to partition available requests into 2
micro-batches that could be run concurrently and thereby allowed TensorRT-LLM to hide
some exposed CPU runtime. However, optimization work has been done to reduce this
exposed CPU runtime and it has been found that the concurrent execution
of micro-batches did not provide additional benefits in terms of throughput,
and in most cases, was hurting latency.

### Maximum Attention Window Size

The `max_attention_window_size` flag sets the maximum number of tokens that are
attended to in order to generate one token when using techniques like sliding window
attention. See this
[Document](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.md#sliding-window-attention-cyclic-rolling-buffer-kv-cache)
for more details. It defaults to the maximum sequence length
(`max_input_length + max_output_length` when building the engine), which means
that the feature is disabled by default.

When set to a smaller value than `max_input_length + max_output_length` (during
engine build), only the KV cache of the last `max_attention_window_size` tokens
will be stored. If the input sequence length at runtime exceeds the
`max_attention_window_size` value, the accuracy may start dropping, but the
runtime performance will be better (due to the reduction in terms of
computations and GPU memory allocation). Users can modify that value to
increase runtime performance at the expense of reduced accuracy.

### Chunked Context

Turning on context chunking by specifying `enable_chunked_context` in
`TrtGptModelOptionalParams` will increase the chance of batch processing between
the context and the generation phase, thereby balancing the calculation amount
of each iteration and increasing throughput. When this function is turned on,
different performance can be obtained by adjusting `max_num_tokens`. Usually
its recommended value is `N * tokens_per_block`, and `N` is an integer that is
recommended to start from `1` and increase until the best performance is achieved.
