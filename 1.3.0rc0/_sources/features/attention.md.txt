(attention)=


# Multi-Head, Multi-Query, and Group-Query Attention


This document details the implementation of multi-head attention (MHA),
multi-query attention (MQA), and group-query attention (GQA) for autoregressive
models in TensorRT LLM's PyTorch backend.

Multi-head attention involves a sequence of batched matrix multiplications, a softmax operation, and another batched matrix multiplication,
as described in the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper.
[Multi-query Attention (MQA)](https://arxiv.org/abs/1911.02150) and [Group-query Attention (GQA)](https://arxiv.org/abs/2307.09288) are
variants of MHA that use fewer KV heads than the number of query heads.
TensorRT LLM provides several implementations using different backends in `tensorrt_llm/_torch/attention_backend/`.
The following sections explain how to use these implementations and provide a brief guide on implementing new backends.


## Attention Backends


There are currently three available attention backends: the vanilla backend, the TRT-LLM backend, and the Flashinfer backend.
You can specify the desired attention backend using `PyTorchConfig.attn_backend`. For instance, to utilize the Flashinfer backend, you can pass `attn_backend="flashinfer"` to the `LLM` constructor as follows: `LLM(attn_backend="flashinfer")`. This will enable the use of the Flashinfer backend for your model.

The vanilla backend, `VanillaAttention`, is a reference implementation designed primarily for inflight batching and linear KV cache support. While it serves as a useful baseline, it is not recommended for production use due to its limited optimizations.

In contrast, the Flashinfer backend, `FlashInferAttention`, is performance-optimized and supports both inflight batching and paged KV cache. It also includes the following advanced features:

1. **FP8 Quantization**: This feature enables the quantization of inputs and KV cache into FP8 format, significantly reducing memory usage and improving computational throughput.
2. **RoPE Fusion**: By integrating rotary position embedding (RoPE) directly into the attention computation, this feature enhances efficiency and reduces overhead.

The TRT-LLM backend, `TrtllmAttention`, serves as the default backend and supports all the features available in the Flashinfer backend while being further optimized for enhanced performance. It is the recommended choice for production environments. Additionally, it offers the following advanced features:

1. **Fused QKV Input**: It can accept a single QKV tensor as input, which is more efficient compared to using separate Q, K, and V tensors.
2. **FP8 Output**: It supports outputting the attention result in FP8 format, fusing quantization into the attention computation process.

## Implement a New Attention Backend

You can implement a new attention backend to integrate other attention libraries.
An attention backend consists of an `AttentionBackend` class and an `AttentionMetadata` class.
There are three stages in the PyTorch that involve the attention backend:

1. Model construction: During the model's `__init__`, call `AttentionBackend.__init__` to create an attention backend for each layer.
2. Metadata preparation: Before each forward step of the model:
   1. If the metadata is uninitialized, call `AttentionMetadata.__init__` to create the attention metadata.
   2. If using CUDA graphs, call `AttentionMetadata.create_cuda_graph_metadata` to convert the metadata to CUDA graph metadata, which pre-allocates all tensors and can be used to capture CUDA graphs. Do not re-allocate any tensors stored inside `AttentionMetadata` after the initial warmup run when using CUDA graphs.
   3. To prepare parameters of the input and KV cache, call `AttentionMetadata.prepare` to convert from existing metadata and KV cache manager.
3. Single step forward: During the forward pass of each attention layer, call `AttentionBackend.forward` to perform the attention operation. The `AttentionMetadata` will be provided as a forward argument.

### Implement `AttentionMetadata`

The `AttentionMetadata` class stores metadata from the batched input and KV cache for the attention backend.
It contains the following predefined fields:

| Field | Type | Description |
| ----- | ---- | ----------- |
| max_num_requests | int | The max number of requests in a single batch. |
| num_contexts | int | The number of context-phase sequences in the batch. |
| num_generations | int | The number of generation-phase sequences in the batch. |
| max_num_tokens | int | The max number of tokens in all requests in a single batch. |
| num_tokens | int | Number of tokens in the batch. |
| num_ctx_tokens | int | Number of tokens in sequences in the context phase. |
| kv_cache_manager | KVCacheManager | The KV cache manager. |
| is_cuda_graph | bool | Whether CUDA graph is enabled. |
| seq_lens | Tensor | The length of each sequence in the batch. The shape is (batch_size), and located on CPU memory. |
| seq_lens_cuda | Tensor | A copy of `seq_lens` store on the GPU. |
| context_lens | Tensor | The length of each context-phase sequence in the batch. The shape is (`num_contexts`). |
| position_ids | Optional[Tensor] | The position of each token in each sequence. May be None if positional embedding is applied outside of the backend. |
| request_ids | List[int] | The request ID of each sequence in the batch. |
| prompt_lens | List[int] | The prompt length of each sequence in the batch. |
| kv_cache_params | KVCacheParams | The parameters for the KV cache. |

During `AttentionMetadata.__init__`, you can initialize additional fields for the new attention metadata.
For example, the Flashinfer metadata initializes `decode_wrapper` here.
During `AttentionMetadata.prepare`, the runtime will fill all predefined fields, and you can fill your customized fields according to these predefined fields.
For example, the Flashinfer metadata fills `qo_indptr` by combining `context_lens` and `num_generations` here.

### Implement `AttentionBackend`

The `AttentionBackend` delegates the attention operation to the backend implementation.

Its `__init__` accepts the following arguments:

| Field | Type | Description |
| ----- | ---- | ----------- |
| layer_idx | int | The index of the attention layer in the model. |
| num_heads | int | The number of query heads. |
| head_dim | int | The size of each attention head `(hidden_size // num_heads)`. |
| num_kv_heads | Optional[int] | The number of KV heads. Defaults to num_heads if None. |
| quant_config | QuantConfig | Optional quantization configuration. If None, no quantization is applied. |
| pos_embd_params | PositionalEmbeddingParams | Optional parameters defining how positional embedding should be applied. If None, positional embedding should be applied by the model before calling the backend. Otherwise, the backend is in-charge of applying positional embedding and may cache K without embedding it first. |

Its `forward` accepts the following arguments:

| Field | Type | Description |
| ----- | ---- | ----------- |
| q | Tensor | Query tensor with shape `(num_tokens, num_heads * head_dim)`. |
| k | Tensor | Key tensor with shape `(num_tokens, num_kv_heads * head_dim)`. |
| v | Tensor | Value tensor with shape `(num_tokens, num_kv_heads * head_dim)`. |
| metadata | AttentionMetadata | Metadata for the attention operation. |
| attention_mask | AttentionMask | Optional attention mask. If None, causal mask is applied. |

For example, the Flashinfer backend calls `append_paged_kv_cache` and then `wrapper.run` to perform the attention operation here.


## The Features of the `TrtllmAttention` Backend

The following sections introduce some features of the default `TrtllmAttention` backend.

### Packed Tensors

In the `TrtllmAttention` backend, the attention operator supports the packed (i.e. non padded) QKV inputs.
A naive layout for the QKV inputs is padding the sequences
that are shorter than the `max_sequence_length` to the maximum
length. It may result in excessive memory consumption as well as unneeded
computations on padding tokens (in the various matrix multiplications that
surround the MHA block).
To overcome that problem, TensorRT LLM supports a mode without padding where
the different tokens are packed together and the user provides the operator
with a 1D tensor containing the lengths of the different sequences.

### Context and Generation Phases

The `TrtllmAttention` backend encapsulates different implementations for both
context and generation phases into a single custom torch op.

#### Context Phase

A context-phase implementation without optimization maps to a sequence of GPU kernels that will store the
intermediate `Q*K^T` tensor in memory before calling the softmax operator. It
is the slowest method and the memory footprint is significant (grows quadratically in proportion to the sequence length).

The `TrtllmAttention` backend will trigger a kernel that performs the MHA/MQA block
using a single kernel instead. For short sequences, that kernel uses a vanilla
implementation of MHA/MQA. For larger sequences, this kernel uses the Flash
Attention algorithm as described in
[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
and
[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691).

Currently, the implementation triggers extra kernels that apply pre-processing
to the elements (like RoPE) and populate the KV cache (see below). In a future
release, the number of such kernels may be reduced to improve the overall performance.

#### FP8 Context FMHA

When FP8 quantization is activated, the attention can be further accelerated by
enabling FP8 Context FMHA.

FP8 Paged Context FMHA is also supported with the fp8 quantization workflow.
You need to specify `use_paged_context_fmha = True` for the attention operator.

Please be aware that this feature is only supported on Ada, Hopper and above.

#### Generation Phase

The generation phase is implemented using a single kernel called the masked
multi-head attention in TensorRT LLM. That kernel is able to apply
pre-processing on the Q, K, and V elements on-the-fly: it adds the QKV bias, applies
RoPE, and performs dequantization and quantization. TensorRT LLM will continue to add (or
enable) additional features in future releases, such as enabling support for IA3.

The masked MHA kernel has a special version that distributes the work across
multiple CUDA thread-blocks on the GPU for cases where the GPU occupancy is
low. That mode called multi-block is always enabled.
NVIDIA recommends users to test that mode in scenarios where both the batch
size and the number of heads in the model are relatively small.
The definition of 'small' in that context is hard to quantify because it depends on the model of the GPU.
However, NVIDIA currently recommends testing that mode when `batch_size * num_heads` is less than the number of multi-processors on the GPU.
This guidance may be subject to change in the future.

Note that even if the multi-block mode is enabled, the attention operator will
not immediately trigger the multi-block version of the GPU kernel. There is a
minimum number of tokens (input + generated) that are required for the
multi-block version to become more efficient than the "vanilla" implementation
that uses a single CUDA thread-block per head. It is controlled by an internal
heuristic.

Another note is that as the masked MHA kernels use shared memory size
proportional to sequence length, so there can be some cases that GPU's shared
memory is not enough when multi-block mode is not enabled. To get masked MHA kernel to work in those cases, multi-block mode is forced on and a warning message is printed in the log.

#### XQA Optimization

XQA optimization is another optimization for MQA/GQA in the generation phase.
It currently only supports a limited number of model configurations, such as the LLAMA2 70B model.

Support matrix of the XQA optimization:
 - FP16 / BF16 compute data type.
 - FP16 / BF16 / FP8 / INT8 KV cache data type.
 - Paged KV cache (8 / 16 / 32 / 64 / 128 tokens per block).

By default, this is enabled. Note that a heuristic algorithm
is also used to decide whether to use XQA kernel or masked MHA kernel to get
better performance.
If you want to use that kernel whenever possible, set `TRTLLM_FORCE_XQA=1` to force use of the XQA kernel when the model config is supported.
Supported configurations can be found using the `shouldUse` function of the `DecoderXQARunner` class in
`cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h`.


(inflight-batching)=

### In-flight Batching

TensorRT LLM supports in-flight batching of requests (also known as continuous
batching or iteration-level batching) for higher serving throughput. With this feature,
sequences in the context phase can be processed together with sequences in
the generation phase. The purpose of that technique is to better interleave
requests to reduce latency as well as make better use of the GPUs.
For efficiency reasons (1), the support for inflight batching ***requires the
input tensors to be packed (no padding)***.

***In the current implementation, the sequences that are going through the
context phase must be before the sequences in the generation phase in the input
tensor. For example, for sequences `S0`, `S1` and `S2`, if `S0` and `S2` are in
context phase (and `S1` in generation), tokens from `S0` and `S2` must appear
before the tokens of `S1` in the input tensor***.

_(1) Padding sequences in the generation phase, that contain a single token, to
the length of the maximum input sequence is inefficient use of resources_.



### Chunked Context

In the original state, the common behavior was to process all context tokens at
once. This feature splits the context into several chunks. In this way, the
context chunks can be batched with more tokens during the generation phase,
which is expected to increase the total throughput. Chunking contexts also removes
constraints on input length. Except for the last one, the size of the context chunk needs
to be an integer multiple of the kv-cache block size.

> To enable this feature, the FMHA paged kv-cache also needs to be enabled.

### KV Cache

In the generation phase, a common optimization is to provide the MHA kernel
with a cache containing the values of the past K and V elements that have
already been computed.  That cache is known as the KV cache. TensorRT LLM uses
that technique to accelerate its generation phase. In TensorRT LLM, there is
one KV cache per Transformer layer, which means that there are as many KV
caches as layers in a model. The current version of TensorRT LLM supports two
different types of KV caches: **contiguous** and **paged** KV caches.

#### Contiguous KV Cache

The contiguous KV cache is a monolithic tensor. Its shape is:
```
[max_batch_size * max_beam_width, 2, num_heads, max_seqlen, hidden_dim_per_head].
```

That implementation uses a lot more memory than needed when the sequences are
shorter than the maximum sequence length (even if they end up close to the
limit after the generation of many output tokens, it may take a lot of steps to
reach that point).

#### Paged KV Cache

The paged KV cache decomposes the KV cache into blocks that are distributed to
the different requests by a cache manager during processing. That cache manager
keeps track of the sequences, allocates new blocks from a pool and recycles those
blocks when required. See the implementation of
[`KVCacheManager`](source:tensorrt_llm/_torch/pyexecutor/resource_manager.py).

#### INT8/FP8 KV Caches

In its current implementation, even if the rest of the network runs in INT8 or
FP8, the attention operator works with FP32, FP16, and BFloat16 inputs and
outputs. However, TensorRT LLM supports INT8 and FP8
(`QuantMode.INT8_KV_CACHE` and
`QuantMode.FP8_KV_CACHE`) KV caches.

The attention operator populates the KV cache. When INT8 or FP8 KV caches
are enabled, the input values have to be quantized to 8 bits using a scaling
factor. For quantization, the scaling factor is stored in the
`kv_cache_scaling_factor` tensor. Its shape is `[1]` and only per-tensor
quantization is supported in the current version. Quantization uses inversed scale
since it does multiply as `fp_value * (1.0 / kv_cache_scaling_factor)` in plugin.

During generation, the values read from the cache are dequantized on-the-fly in
the MHA/MQA kernel. Dequantization is defined as
`quantized_value * kv_cache_scaling_factor`.


### Sliding Window Attention, Cyclic (Rolling Buffer) KV Cache

TensorRT LLM has a feature called `Cyclic KV Cache`, which treats the kv cache
as a circular buffer. This means that it only stores the kv cache for the last N
tokens, where N is determined by the `attention_window_size` parameter in
`TrtllmAttention.forward`. When the cache is full, new tokens’ kv cache will
overwrite the "least recently used" caches.

In the context phase, if the input length surpasses the `attention_window_size`,
`Sliding Window Attention` will be activated. This serves the same function as
the sliding window size.

This feature helps to reduce the memory footprint of the kv cache when
dealing with very long sequences.

_Note that the cyclic kv cache feature doesn't work with beam searching currently as
the context kv cache are shared across beams.

### StreamingLLM

The StreamingLLM feature uses a window attention to perform efficient and stable LLM
on long texts, which means that only `N` tokens need to be stored in the KV cache.
Similar to the cyclic KV cache feature in TensorRT LLM, `attention_window_size`
parameter is used to determine `N`. Different from the cyclic KV cache feature,
the first `S` tokens, called sink tokens, are always kept in the attention window,
where `S` is determined by `sink_token_length` parameter.
But in context phase, the self-attentions are dense in the official implementation of
StreamingLLM. It uses all of the tokens for computation and only saves `N` tokens
to the KV cache.

In addition, the relative position embedding is also changed in StreamingLLM.
When determining the relative distance and adding positional information to tokens,
StreamingLLM use the positions within the cache rather than those in the original text.

`sink_token_length` is also used to enable this feature.

### Beam-Search

The attention operator supports beam-search. In the context phase, a single
beam is computed per input sequence. In the generation phase, the MHA/MQA/GQA
kernel uses an additional tensor to reconstruct the correct path for each beam.
That tensor is called the `cache_indirection`. Its shape is `[batch_size,
beam_width, max_seqlen]`.

For a sequence `si`, a beam `bi` and a token `ti`, the element
`cache_indirection[si][bi][ti]` is an integer between `0` and `beam_width-1`
that indicates which path in the beam to read the K and V elements from in the
KV cache. This tensor is populated in the sampling stage.

### Input QKV tensor

The input QKV tensor packs the Q, K and V tensors (concatenated along the last
dimension) after the projection of the hidden states. It is a 3D tensor. RoPE
and quantization to INT8 or FP8 (when needed) are performed by the GPT
attention operator.

In packed mode, its shape is `[num_tokens, 3 * hidden_dim]` where
`num_tokens` is the total number of tokens in the batch. For the sequences in
context phase, the number of tokens of a sequence corresponds to its input
length (even if the beam width is greater than `1` for beam search). For the
sequences in generation phase, there are `beam_width` tokens per sequence. The
beam width can be different for each sequence.

The following pseudo code explains how the number of tokens is computed:

```python
num_tokens = 0

# Add the length of each sequence in context phase.
for seq in context_phase:
    num_tokens += seq.length

# Add the width of the beam for each sequence in generation phase.
for seq in generation_phase:
    num_tokens += seq.beam_width
```

### Rotary Positional Embedding (RoPE)

The attention operator can perform the computation of the Rotary
Positional Embedding (RoPE). When that operation is enabled,
`rotary_embedding_dim` is set to a value greater than 0, it is fused with other
operations. The GPT operator supports GPT-NeoX and GPT-J forms of RoPE by
setting `position_embedding_type` to `PositionEmbeddingType.rope_gpt_neox`
or `PositionEmbeddingType.rope_gptj`.

### ALiBi

The attention operator can apply ALiBi to the result of the `Q*K^T`
product. The bias is computed on-the-fly from the ALiBi slopes in the optimized
kernel.

### Scaling factor(s)

In MHA, the output of the `Q*K^T` product is scaled by a constant value that
is computed as:

```
norm_factor = 1.f / (q_scaling * sqrt(head_size)).
```

### Cross Attention

On top of the MHA as self attention needed by GPT-style decoder-only models, the attention operator also supports cross attention.

This enables the attention operator to be more broadly used as a generic decoder component. For example, the Encoder-Decoder model uses it to issue both the self attention and cross attention modules in its Decoder.
