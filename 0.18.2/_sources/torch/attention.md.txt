(attention)=

# Attention

This document details the implementation of multi-head attention (MHA),
multi-query attention (MQA), and group-query attention (GQA) for autoregressive
models in TensorRT-LLM's PyTorch backend. As a quick reminder, multi-head attention
involves a sequence of batched matrix multiplications, a softmax operation, and another batched matrix multiplication,
as described in the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper.
[Multi-query Attention (MQA)](https://arxiv.org/abs/1911.02150) and [Group-query Attention (GQA)](https://arxiv.org/abs/2307.09288) are
variants of MHA that use fewer KV heads than the number of query heads.
TensorRT-LLM provides several implementations using different backends in `tensorrt_llm/_torch/attention_backend/`.
The following sections explain how to use these implementations and provide a brief guide on implementing new backends.

## Attention Backends


There are currently three available attention backends: the vanilla backend, the TRT-LLM backend, and the Flashinfer backend.
You can specify the desired attention backend using `PyTorchConfig.attn_backend`. For instance, to utilize the Flashinfer backend, you can create a `PyTorchConfig` with `attn_backend = "flashinfer"` and then pass it to the `LLM` constructor as follows: `LLM(pytorch_backend_config=pytorch_config)`. This will enable the use of the Flashinfer backend for your model.

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

For example, the Flashinfer backend calls `append_paged_kv_cache` and then wrapper's `run` to perform the attention operation here.
