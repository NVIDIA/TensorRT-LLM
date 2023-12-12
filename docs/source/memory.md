# Memory Usage of TensorRT-LLM


This document summarizes the memory usage of TensorRT-LLM, and addresses common issues and questions reported by users.


## Understand inference time GPU memory usage


At inference time, there are 3 major contributors to GPU memory usage for a given TRT engine generated from a TensorRT-LLM model: weights, internal activation tensors, and IO tensors. For IO tensors, the major memory footprint comes from the KV cache tensor.


### Weights size

Weights size is fixed depending on the model size, the chosen precision of the weights and the parallelization strategy.
Using lower precision like INT8 or FP8 can reduce the weights size.
When tensor parallelism or pipeline parallelism is used, each rank stores only some portion of the weights.
For example, each rank typically uses just 1/8 of the model weights when using 8-way tensor parallelism or 8-stages pipeline parallelism.


### Activation size


TensorRT can optimize the memory usage by reusing memory for different tensors based on live analysis and tensor size. To avoid out of memory errors at runtime and to reduce the runtime cost of switching optimization profiles and changing shapes, **TensorRT pre-computes the activation tensors memory requirement at build time**. The memory requirement is computed based on an optimized TensorRT graph, one profiles’ memory usage is computed by using the max tensor shape, and the memory requirement of one engine is computed by the maximum size between different profiles. There are external and internal factors that can affect the activation size returned by TensorRT, such as the network structure, kernel fusion, operation scheduling, etc.
Once the TensorRT engine is built, the activation memory size of that engine can be queried by the API `trt.ICudaEngine.device_memory_size`.


Practically, for a given model, specified precision and parallelization strategy, one can tune the activation memory usage by adjusting the max batch size, max input length, max beam width, max number of tokens, padding removal on/off flag, context FMHA on/off flag.
Here some explanations on how these values affect the memory:


1. Reduce build time max input tokens

   Most of the tensors inside a transformer network have a linear relationship with number of input tokens, so activation size will be close to `max number of input tokens * some constant factor`, the constant factor depends on the network structure and TRT internal optimization. The max number of input tokens is derived from build time arguments, one can change the parameters provided to the `prepare_inputs` function, like `GPTLMHeadModel.prepare_inputs` to affect the memory usage, or one can change the command line options of the `build.py` scripts used in the examples.

   When using the [padded tensors](./gpt_attention.md#padded-and-packed-tensors) format, the max number of input tokens equals to `max_batch_size*max_input_len`, so reducing `max_batch_size` and `max_input_len` can almost linearly reduce the activation memory size.
   When using the [packed tensors](./gpt_attention.md#padded-and-packed-tensors) format and `max_num_tokens` is specified, reducing its value will also reduce activation memory size. If `max_num_tokens` is not specified, the max number of input tokens will be derived as `max_batch_size*max_input_len`.

   The packed tensors format is recommended, because it saves both memory and compute.
   The beam width will be folded into the batch size dimension when passing the tensors range into TensorRT, so reducing `max_beam_width` can also reduce the memory usage.


2. Turn on context FMHA

	When the GPT attention plugin is used, turning on the `context_fmha_type` of the plugin will reduce the memory footprint significantly. See the [Context Phase](./gpt_attention.md#context-phase) for details. When the `context_fmha_type` is set to disabled, a workspace size of the plugin will quadratically depend on the sequence length.


3. Tensor parallelism and pipeline parallelism

   TensorRT will reuse memory between layers as much as possible, for a typical example, given *N* decoder blocks in one transformer network, TRT will not allocate *N* copies of the activation memory for each block, since the memory of tensors in the 1st block can be released after the execution, memory can be reused for later blocks, only 1 block’s memory is needed.


   When using tensor parallelism, some tensors are split into smaller chunks and each rank only holds one chunk of the tensor, the activation memory size of each rank will be smaller than when executing the network on a single GPU. When using pipeline parallelism, each rank executes several decoder blocks, and all the tensors are full-size tensors, so the activation memory size is equal to 1 block’s memory size. Thus tensor parallelism normally has higher memory efficiency than pipeline parallelism when all other parameters are the same.


## KV cache tensor

### Python runtime

The Python runtime allocates KV cache tensors based on the parameters of the `GenerationSession.setup` function, the KV cache size is linearly dependent on the `batch_size` and `max_context_length+max_new_tokens`. **Note: This may change in the future, as the Python bindings of the C++ runtime may replace the current python runtime in the future. The Python bindings of C++ runtime behave like C++ runtime.**

### C++ runtime

* When paged KV cache is enabled

   TensorRT-LLM runtime pre-allocates KV cache tensors during initialization for a configured number of blocks and distributes them at runtime.
   KV cache tensors are allocated based on the `KVCacheConfig` object when creating `GptSession`. If neither `maxTokens` nor `freeGpuMemoryFraction` is specified, KV cache will by default allocate 85% of the remaining free GPU memory. When either `maxTokens` or `freeGpuMemoryFraction` is specified, the specified value will be used to compute the KV cache memory size. And if both are specified, firstly the `freeGpuMemoryFraction` is used to compute the number of tokens in KV cache, and then the minimum between this computed number of tokens and `maxTokens` is used.

   In in-flight batching the scheduler can automatically schedule requests as long as enough KV cache space is available (exact behavior depends on the scheduler policy).
   If paged KV cache is used in `GptSession` without in-flight batching, TensorRT-LLM may report OOM errors with message "Can't allocate new blocks. No free blocks left", if the paged KV cache is not large enough for the whole batch.

* When paged KV cache is disabled

   C++ runtime allocates the KV cache tensors for each layer with shape `[batch size, 2, heads,  max seq length, hidden dimension per head]`, where `max seq length` is specified by `GptSession::Config::maxSequenceLength` when creating `GptSession`.

## Memory pool

TensorRT-LLM C++ runtime is using stream-ordered memory allocator to allocate and free buffers, see [BufferManager::initMemoryPool](source:cpp/tensorrt_llm/runtime/bufferManager.cpp), which uses the default memory pool managed by the CUDA driver. When a `GptSession` object is destroyed, memory is returned to the memory pool and can be reused by the next instance of a `GptSession` object. Memory will be released from the pool if it is required for other memory allocations.
However, `nvidia-smi` may still show high memory occupation after memory is returned to the CUDA driver's memory pool. This should not be a concern and is intended behavior. The amount of reserved and free memory in the pool can be inspected by [BufferManager::memoryPoolReserved())](source:cpp/tensorrt_llm/runtime/bufferManager.cpp) and [BufferManager::memoryPoolFree())](source:cpp/tensorrt_llm/runtime/bufferManager.cpp), respectively.

## Known Issues


1. When 2 optimization profiles are used, the weights memory size may be doubled in some cases if the underlying kernel choices require different weights layouts for different optimization profiles. This issue will be fixed in a future release.

2. When FP8 GEMM is used, the activation memory might be larger than the theoretical optimized memory size, this will be enhanced in a future release.

## FAQ

1. Why is the memory size large even though a small batch size and sequence length are used in the runtime?

   As explained above, the activation memory size is computed based on the max tensor shapes at TensorRT engine building time, try to reduce the engine building time parameters like `max_num_token`, `max_batch_size`, `max_input_len`, see [Activation size](#activation-size) for details.


2. Why can the engine be generated, but the inference will run out of memory (OOM) at runtime?

   At engine building time, TensorRT will tune the kernel selection layer by layer, it does not necessarily allocate all the memory required to run the entire engine. If the activation tensors required to run a single layer are small, while the I/O tensor (like KV cache) sizes required to run the engine are large, building will succeed since it may not need to allocate the large I/O tensors, runtime may fail with OOM errors on allocating large IO tensors.

   TensorRT-LLM has provided a `check_gpt_mem_usage` utility function to check the upper bound of the memory size given an engine, and the related batch size, I/O sequence length, etc., when the upper boundary check exceeded the GPU physical memory size, warning messages will be printed.

3. How to debug the memory usage of TensorRT-LLM?

   When the verbose logging level is used, TensorRT and TensorRT-LLM will print messages about memory usage details.
   The line showing "Total Weights Memory" indicates the weights memory size, and the line "Total Activation Memory" indicates the activation memory size.

   Normally the weights memory size is close to the TensorRT engine size, since most of the content in the engine is from weights for LLM networks.

4. For pipeline parallelism, is build time max batch size the limit of micro batch size?

   Yes, in pipeline parallel mode, TensorRT-LLM runtime will split the batch of requests into micro batches, and enqueue these micro batches into TRT engine sequentially.
   The `max_batch_size` at build time means that batch size of one engine enqueue call shall be smaller than it. The total batch size before splitting into micro batches can be larger than the build time `max_batch_size`.

   For example, if you have 4-stages pipeline parallelism, and intend to run the engine using micro batch size 2 and run 16 micro batches (total batch size 32) in one `generate` call.
   You could just set the `max_batch_size` at building time to 2, instead of 32. Setting build time `max_batch_size` 32 will occupy almost 16x more activation memory.
