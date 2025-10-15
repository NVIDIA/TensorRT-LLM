# KV Cache System

The KV cache stores previously computed key-value pairs for reuse during generation in order to avoid redundant calculations. The TensorRT LLM KV cache system also supports reuse across requests and uses a suite of tools like offloading and prioritized eviction to increase reuse. It supports variable attention window sizes and Multi-Head Attention (MHA) optimization techniques such as MQA and GQA.

## The Basics

The KV cache is a pool of blocks that can hold KV state for a fixed number of tokens. Multiple layers are packed within a single block, which requires all the layers to have the same number of heads and the same attention window size. A separate pool is created for each combination of attention window size and number of heads to support variable attention window size and optimization techniques like GQA.

The number of tokens that can be stored in a single block can be set by user when the model engine is created. It must be a power of two greater than 1. Blocks are assigned to requests as needed. Blocks are stored in a search structure as they are filled by requests, this allows later requests to reuse KV state if they have a matching prefix.

If more than one pool is created, available memory is divided among the pools. The fraction to assign to each pool is determined during initialization and is static. This is not optimal and we are working on providing a better solution.

## Reuse Across Requests

Blocks containing KV state computed for previous requests are stored in a radix search tree as soon as they are filled. A search is performed when a new request is added, and matched blocks are reused instead of calculated. Blocks that are reused can be shared among multiple requests, so reuse saves memory as well as computations.

Blocks remain reusable until they are evicted from the search tree. Eviction happens when a new (blank) block is needed. The core eviction scheme is prioritized LRU. All blocks are assigned a priority between 0 and 100 (100 being most important). All blocks of the lowest priority must be evicted before any blocks of the next priority can be evicted. If all blocks have the same priority, the least recently used block is evicted.

When a block is evicted from primary memory, its KV state is copied to a block in secondary memory. The secondary memory block remains in the search tree, so the block remains reusable until it is evicted from secondary memory. Eviction from secondary memory happens when a new block in secondary memory is needed to offload a primary block. The eviction scheme is the same for primary and secondary blocks.

One caveat in the current code is that only leaf blocks can be evicted (leaves are blocks with no descendants in the radix tree). This design works well for full attention layers, but not for limited attention layers. This will be fixed in a future version.

### Retention Policy

Blocks are assigned priority in line with the [retention policy](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.KvCacheRetentionConfig) of the request. Blocks with lower priority scores will be freed preferentially to blocks with higher priority. The retention policy is a list of [TokenRangeRetentionConfig](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.KvCacheRetentionConfig.TokenRangeRetentionConfig) objects, each specifying priority for a given range of tokens, such as "assign priority X to tokens 10 through 61". You can also assign a duration in milliseconds for this to remain in effect. Priority reverts to the default of 35 after a period of ```duration_ms``` has elapsed from the first time the block was made available for reuse. TokenRangeRetentionConfig only applies to input (prompt) tokens. The property ```decode_retention_policy``` specifies what priority to assign to blocks with generated (decoded) tokens and ```decode_duration_ms``` specifies how long this should remain in effect. Priority reverts to the default after expiration. Any property that expects a duration can be set to None. This indicates that particular part of the retention policy never expires.

Not in use: ```transfer_mode``` is a debug option and should not be used.

See [this example](../examples/kvcacheretentionconfig.md) for an example of how to change block priorities of specific requests by altering their retention policy.

### Speculative Decoding

Reuse across requests is supported by all speculative decoding models. Please see [speculative decoding](speculative-decoding.md) for more details.

## Limited Attention Window Size

TensorRT LLM takes advantage of layers with limited attention window size in order to reduce computations and memory usage. Blocks that leave the attention window are freed and placed on the radix search tree so they can be reused.

## MQA / GQA

TensorRT LLM takes advantage of grouped query attention in order to save memory. KV cache will create blocks with only enough space to store state for the discrete query head groups. For MHA, there is one group per head, for MQA there is a single group for all the heads. GQA strikes a balance between these two.

## Controlling KV Cache Behavior

Many of the features in the KV cache system are optional or have user defined properties that alter how they work. Users can control KV cache features through class [KVCacheConfig](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.KvCacheConfig). The remainder of this section describes how to change the most important behaviors of the KV cache system.

See [this example](../examples/kvcacheconfig.md) for an example of how to use KvCacheConfig to control KV cache behavior.

### Datatype

Perhaps the most important property is ```dtype``` which specifies what data type is held in KV cache. The default 'auto' specifies that data type should be inferred from model config.

### How Much Memory is Allocated to KV Cache

Property ```free_gpu_memory_fraction``` is a ratio > 0 and < 1 that specifies how much of free GPU memory should be allocated to KV cache. The default is 90% (ratio of 0.9). If ```max_tokens``` is also set, KV cache will determine how much memory is needed to hold ```max_tokens``` and will allocate the lesser of ```max_tokens``` and ```free_gpu_memory_fraction```.

### Enable/Disable Cross Request Reuse

Block reuse across requests is enabled by default, but can be disabled by setting ```enable_block_reuse``` to False.

### KV Cache Salting for Secure Reuse

KV cache salting provides a security mechanism to control which requests can reuse cached KV states. When a `cache_salt` parameter is provided with a request, the KV cache system will only allow reuse of cached blocks given the same cache salt value. This prevents potential security issues such as prompt theft attacks, where malicious users might try to infer information from cached states of other users' requests.

To use cache salting, specify the `cache_salt` parameter as a string when creating requests. Only requests with matching cache salt values can share cached KV blocks. The salt value can be any non-empty string, such as a user ID, tenant ID, or hash string.

### Enable Offloading to Host Memory

Before a block is evicted from GPU memory, it can optionally be offloaded to host (CPU) memory. The block remains reusable until it is evicted from host memory. When an offloaded block is reused, it is first copied back into GPU memory. Offloading is controlled with property ```host_cache_size``` which specifies how much host memory (in bytes) should be allocated for offloading. The default is 0.

When offloading is enabled, the client can prevent specific blocks from being offloaded by toggling block priority. Blocks with lower priority than a certain threshold are not offloaded; they are evicted directly from GPU memory to reduce traffic between GPU and host. This priority is set with ```secondary_offload_min_priority```. Default value is 35, meaning any block with lower priority than 35 will not be offloaded.

Here is an [example](../../../examples/llm-api/llm_kv_cache_offloading.py) to show how to enable host offloading.

### Partial Reuse

Partial reuse of a block can happen when some but not all tokens are matched. It is enabled by default, but can be disabled by setting ```enable_partial_reuse``` to False.

The property ```copy_on_partial_reuse``` specifies whether a block should be copied or not in order to allow partial reuse. If copying is disabled, a partially matched block can only be reused if no other request is using it. If copying is enabled, partially matched blocks are not reused directly, instead a new block is created and the matched tokens are copied into the new block. This allows multiple requests to partially reuse a block.

### Attention Window Size

Property ```max_attention_window``` specifies the maximum attention window size for each layer in the model as a list of integer values. If the length of this list is less than number of layers, the list is repeated as many times as necessary. For instance, if the model has only full attention layers and maximum sequence length is 4096, you can specify this as ```max_attention_window = [4096]```. If the first layer is full attention, the second layer is limited attention with window size 256 and then this repeats for the remaining layers, you specify this as ```max_attention_window = [4096,256]```. This means first layer is full attention, second layer is limited attention, third layer is full attention, fourth layer is limited attention and so on.

### Deprecated Properties

Properties ```use_uvm``` and ```sink_token_length``` have been deprecated and will be removed in a future release.
