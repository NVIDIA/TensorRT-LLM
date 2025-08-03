# Long Sequences

In many real-world scenarios, such as long documents summarization or multi-turn conversations, LLMs are required to perform cognitive tasks across long sequences to get better results. This will present challenges to the LLM inference. TensorRT-LLM can support different methods to process long sequences efficiently. This document will introduce those optimization techniques.


## Chunked Context

Chunked context allows TensorRT-LLM to divide the input tokens into smaller chunks and batch those chunks with the decode requests.

With the chunked context feature, there are two benefits:
- This can prevent the context phase from becoming a bottleneck, enable more parallelization with tokens in the decode phase, and increase GPU utilization.
- Chunked context allows TensorRT-LLM to handle requests with longer contexts while achieving higher concurrency. Since memory usage depends on the number of tokens processed per iteration, chunked context decouples memory consumption from the input request's context length, changing it to the smaller chunk size. This enables TensorRT-LLM to process longer contexts without increasing memory requirements, which can also help increase the concurrency under the same memory consumption.

To enable chunked context, please set the `enable_chunked_prefill` in `LLM` API to `True`.
```bash
    llm = LLM(
        ...
        enable_chunked_prefill=True,
        ...
    )
```

Note that if chunked context is enabled, please set the `max_num_tokens` to be an integer multiple of the kv-cache block size `tokens_per_block`, which defaults to 64.

## Chunked attention

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/feat/1.0_doc_dev/docs/source/1.0/features/media/feat_long_seq_chunked_attention.png" alt="feat_long_seq_chunked_attention" width="320" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 1. Illustration of chunked attention </em></sub></p>

Instead of splitting the input tokens into smaller chunks for the whole model, chunked attention is another method that is only applied to the attention layers in models.

With chunked attention, the tokens in context requests are split into chunks of a specified size. Then tokens can only attend to other tokens in the same chunk. For example, if the chunk size is 3, we might have a mask illustrated in Figure 1. Each token only needs to attend to at most the past chunk-sized tokens. As a result, both the KV cache size and the attention computation can be significantly reduced.

Currently TensorRT-LLM can only support chunked attention in llama4 model with TRTLLM attention backend. TensorRT-LLM will read `attention_chunk_size` from the model config. If it is not None, the chunked attention will be enabled with chunk size `attention_chunk_size`. If you want to enable chunked attention to other models, you can set the `attention_chunk_size` in attention API to a valid value.

Note that chunked attention can only be applied to context requests.

## Sliding Window Attention

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/feat/1.0_doc_dev/docs/source/1.0/features/media/feat_long_seq_sliding_win_attn.png" alt="feat_long_seq_sliding_win_attn" width="320" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 2. Illustration of sliding window attention </em></sub></p>


Since attention layers are usually the performance bottleneck when processing requests with long sequences, sliding window attention is an effective method to limit the attention span of each token to a fixed size window around it, dramatically reducing the amount of computation and memory required.

Figure 2 shows the sliding window attention mask. Each token will only attend to the past `N` tokens. If the number of past tokens surpasses the max attention window size, `Sliding Window Attention` will be activated.

TensorRT-LLM treats the kv cache as a circular buffer to support this feature, which is also called `Cyclic KV Cache`. It only stores the kv cache for the last `N` tokens, where `N` is determined by the `KvCacheConfig.max_attention_window` parameter in `LLM` API. TensorRT-LLM allows different `N` values for each layer and users can simply provide a `list[int]` to the `KvCacheConfig.max_attention_window`. To enable this feature, users can set
```bash
    kv_cache_config = KvCacheConfig(
        ...
        max_attention_window = [...],
        ...
    )
    llm = LLM(
        ...
        kv_cache_config=kv_cache_config,
        ...
    )
```
If the number of the provided elements in `KvCacheConfig.max_attention_window` is less than the number of layers, the provided list will be repeated multiple times to the number of layers to set unique values for each layer. However, it's important to note that the memory allocation for the kv cache still relies on the buffer's maximum value.

Note that the `Sliding Window Attention` feature doesn't work with beam searching currently as the context kv cache is shared across beams.

## StreamingLLM

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/feat/1.0_doc_dev/docs/source/1.0/features/media/feat_long_seq_streamingllm.png" alt="feat_long_seq_streamingllm" width="320" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 3. Illustration of StreamingLLM </em></sub></p>

[StreamingLLM](https://arxiv.org/pdf/2309.17453) demonstrates that not only are the most recent `N` tokens important, but the initial `S` sink tokens also play a crucial role in restoring the performance of `Sliding Window Attention`.

Figure 3 shows the StreamingLLM attention mask. Different from the `Sliding Window Attention` feature, the first `S` tokens, called sink tokens, are always kept in the attention window.

Similar to the `Sliding Window Attention` feature, TensorRT-LLM uses `KvCacheConfig.max_attention_window` parameter in `LLM` API to determine `N` and uses `KvCacheConfig.sink_token_length` to determine `S`, which is an `int` value to set the number of sink tokens for all of the attention layers. In the generation phase, for each input token, the self-attention only attends to the keys/values of the past `N` tokens, including `S` sink tokens and `N-S` most recent tokens. But in the context phase, the self-attention is dense in the official implementation of [StreamingLLM](https://github.com/mit-han-lab/streaming-llm), and it uses all of the tokens for computation and only saves `N` tokens to the KV cache. TensorRT-LLM follows the same implementation and uses `dense_context_fmha` in attention API to enable it. In addition, the relative position embedding is also changed in StreamingLLM. When determining the relative distance and adding positional information to tokens, StreamingLLM uses the positions within the cache rather than those in the original text. TensorRT-LLM uses `enable_pos_shift` in attention API to enable it.

To enable `StreamingLLM` feature, users can set
```bash
    kv_cache_config = KvCacheConfig(
        ...
        max_attention_window = [...],
        sink_token_length = 4, # use 4 as an example
        ...
    )
    llm = LLM(
        ...
        kv_cache_config=kv_cache_config,
        enable_streamingllm=True,
        ...
    )
```

Note that currently this feature is only supported in llama models.
