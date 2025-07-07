# KV Cache System

The KV cache stores previously computed key-value pairs for reuse during generation in order to avoid redundant calculations. The TensorRT-LLM KV cache system also supports reuse across requests and uses a suite of tools like offloading and prioritized eviction to increase reuse. It has support for variable attention window sizes and MHA optimization techniques like MQA and GQA.

## The Basics

The KV cache is a pool of blocks that can hold KV state for a fixed number of tokens. Multiple layers are packed within a single block, which requires all the layers to have the same number of heads and the same attention window size. A separate pool is created for each combination of attention window size and number of heads in order to support variable attention window size and optimization techniques like GQA. Number of tokens that can be stored in a single block must be a power of two greater than 1. Blocks are assigned to requests as needed.

## Reuse Across Requests

Blocks containing KV state computed for previous requests are stored in a radix search tree as soon as they are filled. A search is performed when a new request is added, matched blocks are reused instead of calculated. Blocks that are reused can be shared among multiple requests, thus reuse saves memory as well as computations. Blocks remain reusable until they are evicted from the search tree. Eviction happens when a new (blank) block is needed. The core eviction scheme is prioritized LRU. All blocks are assigned a priority between 0 and 100 (100 being most important), all blocks of the lowest priority must be evicted before any blocks of the next priority can be evicted. If all blocks have the same priority, the least recently used block is evicted. When a block is evicted from primary memory, it's KV state is copied to a block in secondary memory. The secondary memory block remains in the search tree, hence the block remains reusable until it is evicted from secondary memory. Eviction from secondary memory happens when a new block in secondary memory is needed to offload a primary block. The eviction scheme is the same for primary and secondary blocks.

## Limited Attention Window Size

TensorRT-LLM takes advantage of layers with limited attention window size in order to reduce computations and memory usage. Blocks that leave the attention window are freed and placed on the radix search tree so they can be reused. 

## MQA / GQA

TensorRT-LLM takes advantage of grouped query attention in order to save memory. KV cache will create blocks with only enough space to store state for the discrete query head groups. For MHA, there is one group per head, for MQA there is a single group for all the heads. GQA strikes a balance between these two.

