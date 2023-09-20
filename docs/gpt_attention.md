# Multi-head, Multi-query Attention and Group-query Attention

This document details the implementation of multihead attention (MHA) and
multiquery attention (MQA) for auto-regressive GPT-like models in TensorRT-LLM.
As a quick reminder, the multihead attention is the sequence of a batched
matmul, a softmax and another batched matmul described in the
[Attention Is All You Need](https://arxiv.org/abs/1706.03762) article.
Multi-query Attention (MQA) [[https://arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)]
Group-query Attention (GQA) [[https://arxiv.org/abs/2307.09288](https://arxiv.org/abs/2307.09288)]
are variants of MHA that use fewer, so-called, K/V head than the number of
query heads.  TensorRT-LLM, MHA, MQA and GQA are implemented by the operator
[`tensorrt_llm.functional.gpt_attention`](../tensorrt_llm/functional.py).

## Important Notes

***The signature of the Python function `gpt_attention` may change in the
future release - we are still in the process of adapting the API to more cases.
The current version is still work-in-progress! See at the end of the document
for hints regarding the arguments that are likely to be removed or merged with
others in the future release.***

As mentioned in the sequel, we plan to apply the following changes (and
possibility other ones) to this function:

 * Change the rank of the input QKV tensor from 3 to 2 for packed, i.e.
   non-padded, tensors. See below for an explanation.

 * Add the QKV bias tensor. It would enable an optimization for adding the bias
   on-the-fly in the MHA/MQA kernel for the generation phase.  It will also
   enable a reduction in the number of GPU kernel calls in the context phase.

As discussed below, the current implementation supports two input modes: Padded
and packed (non-padded). As the packed mode is always more memory-efficient and
faster than the padded mode, ***support for padded mode may be removed in the
future***.

## Padded and Packed Tensors

The current version of the GPT attention operator supports two different types
of QKV inputs: Padded and packed (i.e. non padded) inputs. The mode is
determined by the global configuration parameter `remove_input_padding` defined
in [`tensorrt_llm.plugin`](../tensorrt_llm/plugin/plugin.py).

When padding is enabled (i.e. `remove_input_padding` is `False`), the sequences
that are shorter than the `max_sequence_length` are padded to that maximum
length. It results in excessive memory consumption as well as unneeded
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
[`tensorrt_llm.plugin`](../tensorrt_llm/plugin/plugin.py)),
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

## Inflight batching

TensorRT-LLM supports a feature called inflight batching. With that feature,
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

_Advanced: In the previous early access release of TensorRT-LLM, only the
contiguous KV cache was supported and K and V had different layouts (K had 16B
vectors in its inner-most dimension). Starting with this release, K and V have
the same layout:_
```
max_batch_size * max_beam_width x num_heads x max_seqlen x hidden_dim_per_head.
```

### Paged KV Cache

The paged KV cache decomposes the KV cache into blocks that are distributed to
the different requests by a cache manager during processing. That cache manager
keeps track of the sequences, allocate new blocks from a pool and recycle those
blocks when required. See the implementation of
[`tensorrt_llm.runtime.KVCacheManager`](../tensorrt_llm/runtime/kv_cache_manager.py).

In its current implementation, TensorRT-LLM allocates a tensor of memory to
keep the KV cache data. Its shape is:
```
[max_blocks, 2, num_heads, num_tokens_per_block, hidden_dim_per_head],
```
where `max_blocks` is the maximum number of blocks in the KV cache and
`num_tokens_per_block` is the number of tokens that can be stored in each
block. For efficiency reasons, `num_tokens_per_block` must be a power-of-two
(faster divisions and modulus).

During execution, the blocks are dynamically allocated to the requests based on
their memory requirements. The "pointers" to the blocks are kept in the
`kv_cache_block_pointers` tensor. Its shape is:
```
[max_batch_size, max_beam_width, 2, max_blocks_per_sequence * 2],
```
where `max_blocks_per_sequence` indicates the largest number of blocks that can
be assigned to a sequence. It must be a power-of-two. In the inner-most
dimension, that tensor packs two 32-bit integers to represent 64-bit pointers.

_Advanced: The fact that the data cache is a single tensor (per layer) is not
ideal if we need to implement the possibility to move the cache from the GPU to
the host and vice-versa. In a future release of TensorRT-LLM this monolithic
tensor will be split into multiple tensors to support the offloading of the KV
cache_.

## INT8/FP8 KV Caches

In its current implementation, even if the rest of the network runs in INT8 or
FP8, the GPT attention operator works with FP32, FP16, and BFloat16 inputs and
outputs.

However, TensorRT-LLM supports INT8 (`use_int8_kv_cache=True`) and FP8
(`use_fp8_kv_cache=True`) KV caches. It is not possible to set both to `True`
(and the API might change in the future to enforce that point).

The GPT attention operator populates the KV cache. When INT8 or FP8 KV caches
are enabled, the input values have to be quantized to 8 bits using a scaling
factor. For quantization, the scaling factor is stored in the
`kv_orig_quant_scale` tensor. Its shape is `[1]` and only per-tensor
quantization is supported in the current version.

During generation, the values read from the cache are dequantized on-the-fly in
the MHA/MQA kernel. The scaling factor to dequantize those values is stored in
the `kv_quant_orig_scale` tensor. That tensor contains a single value (per
tensor scaling).

_Advanced: Future versions of TensorRT-LLM will likely implement more advanced
quantization/dequantization methods. An obvious first candidate would be
per-channel scaling._

## Beam-Search

The GPT attention operator supports beam-search. In the context phase, a single
beam is computed per input sequence. In the current implementation, the data is
then replicated in the KV cache. It is a known limitation that will be
addressed in a future version of TensorRT-LLM.

In the generation phase, the MHA/MQA kernel uses an additional tensor to
reconstruct the correct path for each beam. That tensor is called the
`cache_indirection`. Its shape is `[batch_size, beam_width, max_seqlen]`.

For a sequence `si`, a beam `bi` and a token `ti`, the element
`cache_indirection[si][bi][ti]` is an integer between `0` and `beam_width-1`
that indicates which path in the beam to read the K and V elements from in the
KV cache. This tensor is populated in the sampling stage.

## Input QKV tensor

The input QKV tensor packs the Q, K and V tensors (concatenated along the last
dimension) after the projection of the hidden states. It is a 3D tensor. In the
current implementation, the bias associated with QKV operator must be added
before calling the GPT attention operator. ***This requirement may be removed
in the future (for performance reasons)***. RoPE and quantization to INT8 or
FP8 (when needed) will be performed by the GPT attention operator.

In padded mode, its shape is `[batch_beam_size, max_seqlen, 3 * hidden_dim]`
where `batch_beam_size` is the batch size (number of sequences) for the context
phase and the batch size multiplied by the beam width for the generation phase.
Having different beam widths per sequence in padded mode is not supported.

In packed mode, its shape is `[1, num_tokens, 3 * hidden_dim]` where
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

In a future release of TensorRT-LLM, the rank of that packed input tensor
will be reduced from 3 to 2. The current rank is to maintain the homogeneity
between padded and packed modes. It is no longer justified if support for
padded mode is removed.

## Additional Features

### Rotary Positional Embedding (RoPE)

The GPT attention operation can perform the computation of the Rotary
Positional Embedding (RoPE). When that operation is enabled,
`rotary_embedding_dim` is set to a value greater than 0, it is fused with other
operations. The GPT operator supports GPT-NeoX and GPT-J forms of RoPE by
setting `position_embedding_type` to `PositionEmbeddingType.rope_gpt_neox`
or `PositionEmbeddingType.rope_gptj`.

### Scaling factor(s)

In MHA, the output of the `Q*K^T` product is scaled by a constant value that
is computed as:

```
scaling = 1.f / (q_scaling * sqrt(head_size)).
```
