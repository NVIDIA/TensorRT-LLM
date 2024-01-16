# Best Practices for Tuning the Performance of TensorRT-LLM

This document provides some best practices for tuning the performance of TensorRT-LLM.

## How To Measure Performance?

TensorRT-LLM can be benchmarked using the included
[C++](../../benchmarks/cpp/README.md)
and
[Python](../../benchmarks/python/README.md) tools. However, it is *strongly*
recommended to use the C++ benchmarking tool. For detailed performance data and
the steps to reproduce those results, see
this [Document](performance.md).
The [TensorRT-LLM backend](https://github.com/triton-inference-server/tensorrtllm_backend)
can also be used to measure the performance of TensorRT-LLM for online serving.

## Build Options to Optimize the Performance of TensorRT-LLM Models?

This part summarizes how to build engines to enhance the performance of the
runtime and, for some of them, decrease the engine build time.

***Note that some of those features and how to enable them may change in the future.***

### GPT Attention Plugin and Context Fused Multi-Head Attention

It is recommended to enable the GPT attention plugin and fused  multi-head
attention kernel, for the context phase, using the `--use_gpt_attention_plugin`
and `--enable_context_fmha` arguments with `build.py`.

The TensorRT-LLM GPT attention plugin uses efficient kernels and enables an
in-place update of the KV cache. It results in reduced memory consumption as
well as the removal of unneeded memory copy operations (compared with the
implementation that uses the `concat` operator to update the KV cache).

Enabling the fused multi-head attention, during the context phase, will trigger
a kernel that performs the MHA/MQA/GQA block using a single kernel, for more
details, see this [Document](gpt_attention.md#context-phase).

### Remove Input Padding

It is recommended to remove input padding using the `--remove_input_padding`
argument with `build.py`.

When input padding is removed, the different tokens are packed together. It
reduces both the amount of computations and memory consumption. For more details, see
this [Document](gpt_attention.md#padded-and-packed-tensors).

### Maximum Number of Tokens

It is recommended to tune `--max_num_tokens` for better performance. The
`--max_num_tokens` could be roughly estimated as:
```
max_batch_size * max_input_len * alpha + max_batch_size * max_beam_width * (1 - alpha)
```
where `alpha` is a floating-point value between `0.0` and `1.0`. It stands for
a rough estimation of the number of requests in their context phase at each
invocation of the forward function during inference. It is recommended to use a
value between `0.05` and `0.20` (between 5%-20%) but it may depend on the
actual scenario.

The maximum number of tokens equals will not take effects when input padding is
not removed. When input padding is removed (see [Remove Input
Padding](#remove-input-padding)), the tokens from different sequences are
packed together and the maximum number of the tokens can be set to a different
(lower) value, which by default to be `min(max_input_len * max_batch_size, 4096)`.
Note that it has to be higher than `max_input_len`.

There are two aspects that must be considered. Firstly, some input sequences
will be shorter than the maximum input length. Secondly, when in-flight
sequence batching is enabled, requests in context phase will be executed with
requests in generation phase. Those latter requests produce a lot fewer tokens
than `max_input_len` (at most, `beam_width` tokens).

Using a more realistic value for `max_num_tokens` allows TensorRT-LLM to
allocate more memory to store the KV cache and execute more requests together.
It leads to an increased efficiency.

Note that choosing a low value for `--max_num_tokens` will result in lower GPU
utilization. When increasing `--max_num_tokens` to some point, GPU utilization
will plateau, going beyond that saturation point may hurt both first token
latency as well as total end-to-end latency.

### Paged KV Cache

It is recommended to enable paged KV cache using the `--paged_kv_cache` argument
with `build.py`.

The paged KV cache helps manage memory for the KV cache more efficiently (see
this [Document](gpt_attention.md#paged-kv-cache)). It usually leads to an
increase in the batch size and an improved efficiency.

### In-flight Sequence Batching

It is recommended to enable in-flight sequence batching using the
`--use_inflight_batching` argument with `build.py`. Note that this flag enables
the GPT attention plugin, input padding removal and paged KV cache all
together.

In-flight sequence batching schedules sequences in context phase together with
sequences in generation phase to increase efficiency and reduce latency, see
this [Document](gpt_attention.md#inflight-batching) for more details.

### Multi-Block Mode

When the following conditions are met, it is recommended to try the
`--multi_block_mode` argument with `build.py` and evaluate the impact on
performance:

 1. `input_seq_len` > 1024 (An empirically derived value that indicates that the
    context length is long enough),
 2. `sequence_count` * `num_head` < `multiprocessor_count` / 2

Multi-block mode can be beneficial when `batch_size * num_heads` is not large
enough to fully utilize the GPU (the number of CUDA thread blocks is low
compared to the number of streaming multiprocessors). Hence, the multi-block
mode is expected to reduce the latency of the multi-head attention kernel in
the generation phase. However, it requires the context length to be long enough
for the work performed by each CUDA thread block to remain sufficient for
efficiency.

### Custom AllReduce Plugin

On NVLink-based nodes, it is recommended to enable the custom AllReduce plugin
by using the `--use_custom_all_reduce` argument with `build.py`. On PCIE-based
nodes, it is not recommended to enabled that plugin.

The custom AllReduce plugin activates a latency-optimized algorithm for
the AllReduce operation instead of the native NCCL operator. However, the
performance benefits may not be seen on PCIE-based systems.

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
with `build.py`. See those
[Examples](../../examples/gpt/README.md#tensor-parallelism-for-embedding-lookup-table)
for details.

### Horizontal Fusion in Gated-MLP

Horizontal fusion in Gated-MLP combines two Matmul operations into a single one
followed by a separate SwiGLU kernel. If both model and batch sizes are large,
it is recommended to enable the feature by using the `--use_fused_mlp` argument
with `build.py`. When the workload is very small, it is not recommended to
enable that feature.

### BERT Attention Plugin and Context Fused Multi-Head Attention

BERT attention plugin and context fused multi-head attention are both
recommended for the BERT model. They must be enabled using the
`--use_bert_attention_plugin` and `--enable_context_fmha` arguments with
`build.py`.

## Runtime Options to Optimize the Performance of TensorRT-LLM Models?

This part summarizes the runtime configuration knobs that can be tweaked to
enhance the performance of already built engines.  Note that currently the
configurations can be modified using the
[Batch Manager API](batch_manager.md#the-batch-manager-api)
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

As explained in the [GPT Manager Ddesign](batch_manager.md#gptmanager-design)
section, the scheduling policy can be set to `MAX_UTILIZATION` to pack as many
requests as possible at each iteration of the forward loop, when in-flight
sequence batching is enabled. It maximizes the utilization of the GPUs by
aggressively scheduling requests at the risk of having to pause requests if the
KV cache size limit is reached.

For a more conservative approach with respect to the KV cache limitations in
terms of memory allocation, `schedulerPolicy` should be set to
`GUARANTEED_NO_EVICT` to guarantee that a started request is never paused.

If the goal is to maximizes the throughput, users should try `MAX_UTILIZATION`.
However, they need to keep in mind that it may have a negative impact on
latency if requests have to be paused.

### TensorRT Overlap

When TensorRT overlap is enabled, available requests are partitioned into 2
micro-batches that can be run concurrently. It allows TensorRT-LLM to hide
exposed CPU runtime. However, it may not give performance benefits when the
size of the model is not big enough to overlap the host overhead, or when the
number of requests is too small.

If the goal is to increase throughput, it is recommended to try setting that
argument to `True`. However, it must be noted that it may actually hurt
latency.

### Maximum Attention Window Size

The `max_attention_window_size` flag sets the maximum number of tokens that are
attended to in order to generate one token when using techniques like sliding window
attention. See this
[Document](gpt_attention.md#sliding-window-attention-cyclic-rolling-buffer-kv-cache)
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
