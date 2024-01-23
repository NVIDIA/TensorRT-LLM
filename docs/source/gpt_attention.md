# Multi-head, Multi-query and Group-query Attention

This document details the implementation of multihead attention (MHA),
multiquery attention (MQA) and group-query attention (GQA) for auto-regressive
GPT-like models in TensorRT-LLM.  As a quick reminder, the multihead attention
is the sequence of a batched matmul, a softmax and another batched matmul
described in the
[Attention Is All You Need](https://arxiv.org/abs/1706.03762) article.
Multi-query Attention (MQA) [[https://arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)]
Group-query Attention (GQA) [[https://arxiv.org/abs/2307.09288](https://arxiv.org/abs/2307.09288)]
are variants of MHA that use fewer, so-called, K/V head than the number of
query heads.  TensorRT-LLM, MHA, MQA and GQA are implemented by the operator
[`tensorrt_llm.functional.gpt_attention`](source:tensorrt_llm/functional.py).

## Important Note

As discussed below, the current implementation supports two input modes: Padded
and packed (non-padded). As the packed mode is always more memory-efficient and
faster than the padded mode, ***support for padded mode may be removed in the
future***.

## Padded and Packed Tensors

In TensorRT-LLM, the GPT attention operator supports two different types
of QKV inputs: Padded and packed (i.e. non padded) inputs. The mode is
determined by the global configuration parameter `remove_input_padding` defined
in [`tensorrt_llm.plugin`](source:tensorrt_llm/plugin/plugin.py).

When padding is enabled (i.e. `remove_input_padding` is `False`), the sequences
that are shorter than the `max_sequence_length` are padded to that maximum
length. It may result in excessive memory consumption as well as unneeded
computations on padding tokens (in the various matrix multiplications that
surround the MHA block).

To overcome that problem, TensorRT-LLM supports a mode without padding where
the different tokens are packed together and the user provides the operator
with a 1D tensor containing the lengths of the different sequences.  It is
recommended that users to always use packed mode (and support for the padded
mode may be removed in the future).

## Context and Generation Phases

The GPT attention operator encapsulates different implementations for both
context and generation phases in auto-regressive models like GPT.

### Context Phase

If the `context_fmha_type` is set to `disabled` (see
[`tensorrt_llm.plugin`](source:tensorrt_llm/plugin/plugin.py)),
the implementation maps to a sequence of GPU kernels that will store the
intermediate `Q*K^T` tensor in memory before calling the softmax operator. It
is the slowest method and the memory footprint is significant (quadratically
depends on the sequence length).

Otherwise, if `context_fmha_type` is set to a `enabled` or
`enabled_with_fp32_acc` (accumulation in the first batched matmul is forced to
FP32), that function will trigger a kernel that performs the MHA/MQA block
using a single kernel. For short sequences, that kernel uses a vanilla
implementation of MHA/MQA. For larger sequences, this kernel uses the Flash
Attention algorithm as described in
[https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
and
[https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691).

Currently, the implementation triggers extra kernels that apply pre-processing
to the elements (like RoPE) and populate the KV cache (see below). In a future
release, the number of such kernels is planned on being reduced in order to
improve the overall performance.

### Generation Phase

The generation phase is implemented using a single kernel, called the masked
multihead attention in TensorRT-LLM. That kernel is able to apply
pre-processing on the Q, K and V elements on-the-fly: Add the QKV bias, apply
RoPE, do dequantization/quantization. TensorRT-LLM will continue to add (or
enable) additional features in future releases. For example, enable the support
for ALiBi or IA3.

_The masked MHA kernel has a special version that distributes the work across
multiple CUDA thread-blocks on the GPU for cases where the GPU occupancy is
low. That mode called multi-block can be enabled using the `multi_block_mode`
flag. Users are recommended to test that mode in scenarios where both the batch
size and the number of heads in the model are relatively small. The exact
definition of small in that context will depend on the model of the GPU and is
hard to predict but to provide with a rule of thumb, it is worth testing that
mode when `batch_size * num_heads` is less than the number of multi-processors
on the GPU (that suggestion may evolve in the future as more research is
conducted and the software improves)_.

_Note that even if the multi-block mode is enabled, the attention operator will
not immediately trigger the multi-block version of the GPU kernel. There is a
minimum number of tokens (input + generated) that are required for the
multi-block version to become more efficient than the "vanilla" implementation
that uses a single CUDA thread-block per head. It is controlled by an internal
heuristic._

Another note is that as the masked MHA kernels use shared memory size
proportional to sequence length, so there can be some cases that GPU's shared
memory is not enough when multi-block mode is not enabled. To get masked MHA
kernel work in these cases, multi-block mode is forced on and a warning log is
printed.

#### XQA Optimization

Another optimization for MQA/GQA in generation phase called XQA optimization.
It is still experimental feature and support limited configurations. LLAMA2 70B
is one model that it supports.

Support matrix of the XQA optimization:
 - FP16 / BF16 compute data type.
 - FP16 / BF16 / FP8 / INT8 KV cache data type.
 - Paged KV cache (64 / 128 tokens per block).

This is default enabled. To disable this, you need to use the
flag `--disable_xqa` when building the engines. Note that a heuristic algorithm
is also used to decide whether to use XQA kernel or masked MHA kernel to get
better performance. That means even `--disable_xqa` is not set, XQA kernels
may not also be used. If you want to always use that kernel when possible,
`TRTLLM_FORCE_XQA=1` can be set to force use XQA kernels when the model config
is supported. Detailed supported configuration can be found function `shouldUse`
of class `DecoderXQARunner` in
`cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h`.


## Inflight batching

TensorRT-LLM supports a feature called in-flight batching. With that feature,
sequences in context phase can be processed together with sequences in
generation phase. The purpose of that technique is to better interleave
requests to reduce latency as well as make better use the of the GPUs.
For efficiency reasons (1), the support for inflight batching ***requires the
input tensors to be packed (no padding)***.

***In the current implementation, the sequences that are going through the
context phase must be before the sequences in the generation phase in the input
tensor. For example, for sequences `S0`, `S1` and `S2`, if `S0` and `S2` are in
context phase (and `S1` in generation), tokens from `S0` and `S2` must appear
before the tokens of `S1` in the input tensor***. The constraint may or may not
be relaxed in a future version.

_(1) Padding sequences in the generation phase, that contain a single token, to
the length of the maximum input sequence is inefficient use of resources_.

## KV Cache(s)

In the generation phase, a common optimization is to provide the MHA kernel
with a cache containing the values of the past K and V elements that have
already been computed.  That cache is known as the KV cache. TensorRT-LLM uses
that technique to accelerate its generation phase. In TensorRT-LLM, there is
one KV cache per Transformer layer, which means that there are as many KV
caches as layers in a model. The current version of TensorRT-LLM supports two
different types of KV caches: **contiguous** and **paged** KV caches.

### Contiguous KV Cache

The contiguous KV cache is a monolithic tensor. Its shape is:
```
[max_batch_size * max_beam_width, 2, num_heads, max_seqlen, hidden_dim_per_head].
```

That implementation uses a lot more memory than needed when the sequences are
shorter than the maximum sequence length (even if they end up close to the
limit after the generation of many output tokens, it may take a lot of steps to
reach that point).

### Paged KV Cache

The paged KV cache decomposes the KV cache into blocks that are distributed to
the different requests by a cache manager during processing. That cache manager
keeps track of the sequences, allocate new blocks from a pool and recycle those
blocks when required. See the simplified implementation of
[`tensorrt_llm.runtime.KVCacheManager`](source:tensorrt_llm/runtime/kv_cache_manager.py).
A more efficient C++ implementation is included in the
[Batch Manager](source:cpp/include/tensorrt_llm/batch_manager).

## INT8/FP8 KV Caches

In its current implementation, even if the rest of the network runs in INT8 or
FP8, the GPT attention operator works with FP32, FP16, and BFloat16 inputs and
outputs. However, TensorRT-LLM supports INT8 and FP8
(`kv_cache_quant_mode=QuantMode.INT8_KV_CACHE` and
`kv_cache_quant_mode=QuantMode.FP8_KV_CACHE`) KV caches.

The GPT attention operator populates the KV cache. When INT8 or FP8 KV caches
are enabled, the input values have to be quantized to 8 bits using a scaling
factor. For quantization, the scaling factor is stored in the
`kv_cache_scaling_factor` tensor. Its shape is `[1]` and only per-tensor
quantization is supported in the current version. Quantization uses inversed scale
since it does multiply as `fp_value * (1.0 / kv_cache_scaling_factor)` in plugin.

During generation, the values read from the cache are dequantized on-the-fly in
the MHA/MQA kernel, dequantization can be described as
`quantized_value * kv_cache_scaling_factor`.


## Sliding Window Attention, Cyclic (Rolling Buffer) KV Cache

TensorRT-LLM has a feature called `Cyclic KV Cache`, which treats the kv cache
as a circular buffer. This means that it only stores the kv cache for the last N
tokens, where N is determined by the `max_attention_window_size` parameter in
`GenerationSession.setup`. You can see examples of this in the `run.py` or
`summarize.py` files. When the cache is full, new tokens’ kv cache will
overwrite the "least recently used" caches.

In the context phase, if the input length surpasses the `max_attention_window_size`,
`Sliding Window Attention` will be activated. This serves the same function as
the `sliding window_size`.

This feature helps to reduce the memory footprint of the kv cache when
dealing with very long sequences.

_Note that the cyclic kv cache feature doesn't work with beam searching currently as
the context kv cache are shared across beams.

_The experimental feature, which allows different `max_attention_window_size` values
for each layer, is also supported. To utilize this feature, simply provide an
`int32 torch.Tensor` with a shape of `[num_layers]` to the `GenerationSession.setup`.
This tensor will serve as the buffer for `max_attention_window_size`,
setting unique values for each layer. However, it’s important to note that the
memory allocation for the kv cache still relies on the buffer’s maximum value._

## StreamingLLM

The StreamingLLM feature uses a window attention to perform efficient and stable LLM
on long texts, which means that only `N` tokens need to be stored in the KV cache.
Similar to the cyclic KV cache feature in TensorRT-LLM, `max_attention_window_size`
parameter is used to determine `N`. Different from the cyclic KV cache feature,
the first `S` tokens, called sink tokens, are always kept in the attention window,
where `S` is determined by `sink_token_length` parameter in `GenerationSession.setup`.
In addition, the relative position embedding is also changed in StreamingLLM.
When determining the relative distance and adding positional information to tokens,
StreamingLLM use the positions within the cache rather than those in the original text.
`enable_pos_shift` flag is used to enable this feature.

In context phase, the self-attentions is dense in the official implementation of
StreamingLLM, and it uses all of the tokens for computation and only saves `N` tokens
to the KV cache. This mode is determined by the `dense_context_fmha` flag.

## Beam-Search

The GPT attention operator supports beam-search. In the context phase, a single
beam is computed per input sequence. In the generation phase, the MHA/MQA/GQA
kernel uses an additional tensor to reconstruct the correct path for each beam.
That tensor is called the `cache_indirection`. Its shape is `[batch_size,
beam_width, max_seqlen]`.

For a sequence `si`, a beam `bi` and a token `ti`, the element
`cache_indirection[si][bi][ti]` is an integer between `0` and `beam_width-1`
that indicates which path in the beam to read the K and V elements from in the
KV cache. This tensor is populated in the sampling stage.

## Input QKV tensor

The input QKV tensor packs the Q, K and V tensors (concatenated along the last
dimension) after the projection of the hidden states. It is a 3D tensor. RoPE
and quantization to INT8 or FP8 (when needed) are performed by the GPT
attention operator.

In padded mode, its shape is `[batch_beam_size, max_seqlen, 3 * hidden_dim]`
where `batch_beam_size` is the batch size (number of sequences) for the context
phase and the batch size multiplied by the beam width for the generation phase.
Having different beam widths per sequence in padded mode is not supported.

In packed mode, its shape is `[num_tokens, 3 * hidden_dim]` where
`num_tokens` is the total number of tokens in the batch. For the sequences in
context phase, the number of tokens of a sequence corresponds to its input
length (even if the beam width is greater than `1` for beam search).  For the
sequences in generation phase, there are `beam_width` tokens per sequence. The
beam width can be different for each sequence.

In other words, the pseudo-code to compute the number of tokens is:
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

The GPT attention operation can perform the computation of the Rotary
Positional Embedding (RoPE). When that operation is enabled,
`rotary_embedding_dim` is set to a value greater than 0, it is fused with other
operations. The GPT operator supports GPT-NeoX and GPT-J forms of RoPE by
setting `position_embedding_type` to `PositionEmbeddingType.rope_gpt_neox`
or `PositionEmbeddingType.rope_gptj`.

### ALiBi

The GPT attention operator can apply ALiBi to the result of the `Q*K^T`
product. The bias is computed on-the-fly from the ALiBi slopes in the optimized
kernel.

### Scaling factor(s)

In MHA, the output of the `Q*K^T` product is scaled by a constant value that
is computed as:

```
norm_factor = 1.f / (q_scaling * sqrt(head_size)).
```

### Cross Attention

On top of the MHA as self attention needed by GPT-style decoder-only models, `gpt_attention` also supports cross attention.

This enables using `gpt_attention` in a broader aspect as a generic decoder component. For example, the Encoder-Decoder model uses `gpt_attention` to issue both the self attention and cross attention modules in its Decoder.

### Relative Attention Bias (RAB)

Relative attention bias (RAB) is a kind of relative position modeling, adding an attention bias (`Q*K^T+bias`) according to relative positions. RAB is a lightweight method to include the information of relative positions, and is used in the popular Encoder-Decoder model [T5](https://huggingface.co/docs/transformers/model_doc/t5) and also other models in the T5 family.

RAB is supported in two modes: i) regular mode which user passes in relative attention bias computed ahead of MHA. ii) implicit mode which computes the relative attention bias on the fly in MHA. The implicit mode suits the case when the relative attention bias is too large to fit in memory and can be turned on by passing in `max_distance`.
