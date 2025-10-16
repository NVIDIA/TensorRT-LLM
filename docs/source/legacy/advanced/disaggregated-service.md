(disaggregated-service)=

# Disaggregated-Service (Prototype)

```{note}
Note:
This feature is currently in prototype, and the related API is subjected to change in future versions.
```
Currently TRT-LLM supports `disaggregated-service`, where the context and generation phases of a request can run on different executors. TRT-LLM's disaggregated service relies on the executor API, please make sure to read the [executor page](executor.md) before reading the document.

For more information on disaggregated service in LLM inference, one can refer to papers such as [DistServe](https://arxiv.org/abs/2401.09670), [SplitWise](https://arxiv.org/abs/2311.18677).

An [architectural and performance overview](../../../docs/source/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.md), as well as [usage examples](../../../examples/disaggregated/README.md), are provided.

## Environment Variables

TRT-LLM uses some environment variables to control the behavior of disaggregated service.


* `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP`: If set to `1`, generationExecutor will not overlap KV cache transfer with model inference. The default value is `0`.

* `TRTLLM_ENABLE_KVCACHE_RECEIVE_PARALLEL`:  When the generation rank receives KV cache from multiple context ranks within a single context instance, it will receive KV cache from each rank sequentially. If set to `1`, the generation rank will receive KV cache from each rank within one context instance in parallel. The default value is `0`.

* `TRTLLM_REQUEST_KV_CACHE_CONCURRENT`: If set to `1`, generationExecutor prepares independent resources for each context executor to receive KV cache, requests whose KV cache are received from different context executors will be processed concurrently. If set to `0`, the generation executor will reuse the same resource to process KV cache transfer for each request sequentially, reducing the resources used by KV cache transmission and thereby lowering the risk of running out of memory. The default value is `0`.

* `TRTLLM_TRY_ZCOPY_FOR_KVCACHE_TRANSFER`: TRT-LLM typically copies non-contiguous data into a temporary buffer before sending KV cache. If set to `1`, TRT-LLM will attempt to directly transmit each KV cache block, eliminating extra copies. The default value is `0`.

* `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE`: By default, TRT-LLM uses a `stream-ordered memory allocator` to allocate temporary buffers. If this environment variable is set to #Size, TRT-LLM will use `cudaMalloc` to allocate buffer of size #Size for KV cache transmission. The default value is `512MB`. Users can set `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE=1GB` to allocate a 1 GB buffer with `cudaMalloc` for KV cache transmission.

* `TRTLLM_KVCACHE_TRANSFER_USE_ASYNC_BUFFER`: If set to `1`, TRT-LLM will use `cudaMallocAsync` to allocate buffers for KV cache transmission. The default value is `0`. This environment variable only takes effect when `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE` is greater than 0.

* `TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM`: The maximum number of concurrent KV cache sends. The default value is `1`. This environment variable only takes effect when `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE` is greater than 0.

There are some other useful environment variables that may help when encountering failures or performance issues.

* `NCCL_GRAPH_MIXING_SUPPORT`: With the default value `1`, the CUDA driver may create too many CUDA streams while working with one CUDA graph, leading to performance drop. Setting it to `0` will reduce the number of CUDA streams, but please make sure there are no other NCCL ops outside the one CUDA graph, otherwise it's unsafe.

* `UCX_MAX_RNDV_RAILS`: With the default value `2`, UCX attempts to use two InfiniBand (IB) NIC devices per GPU for Rendezvous (RNDV) transfers. When both the context and generation instances enable tensor- and expert-parallel (TEP), multiple TP ranks may transfer KV cache concurrently. Because each TP rank can use up to two NIC devices, some NIC devices can be shared across GPUs, causing contention and reduced throughput. Setting `UCX_MAX_RNDV_RAILS=1` can reduce contention in this case.

## Troubleshooting and FAQ

### General FAQs

*Q. What are the limitations of disaggregated-service in TRT-LLM?*

A. Currently, only `decoder-only engine` and `beamWidth=1` are supported, and the KV cache at each layer of the model is required to be homogeneous, with the same data type and the same number of attention headers.

*Q. Is the engine used by disaggregated-service different from other engines?*

A. No. There are no special requirements for the arguments to build engine.

*Q. Do the engines used by the context executor and generation executor need to be the same?*

A. No. The engines used by context executor and generation executor can be different, and their parallelism can be heterogeneous, i.e., TP,PP can be different, and TRT-LLM will handle the heterogeneity of KV cache.

*Q. Does TRT-LLM support running multiple context executor instances and generation executor instances?*

A. Yes. TRT-LLM supports running multiple context executors and generation executors at the same time, and each executor can use different engine, but it is the user's responsibility to route requests to different executors and  manage `requestId`.

*Q. Can an executor handle both context-only requests and generation-only requests?*

A. Yes, but it's not recommended, TRT-LLM does not implement proper scheduling for the case where the executor handles mixed context-only requests and generation-only requests, it's better to run context-only requests and generation-only requests on different executors.

*Q. Does disaggregated-service in TRT-LLM support multi-gpu and multi-node?*

A. Yes, it's recommended that different executor use different GPUs . We support context-only executor and genertion-only executor run on same node or different nodes. The `participantIds` and `deviceIds` used by each executor need to be explicitly set by the user, and the `participantIds` of each executor must not be intersecting.

### Debugging FAQs

*Q. Does TRT-LLM support using GPU direct RDMA for inter-node KV Cache transfer?*

A. Yes, TRT-LLM supports using GPU direct RDMA for inter-node KV cache transfer.

*Q. What causes the substantial bandwidth fluctuations in kvCache transfers, especially during the first few requests following service initialization?*

A. The communication for kvCache transfer between executors are established dynamically. The connection establishment process incurs significant overhead, which explains the apparently lower kvCache transfer bandwidth observed during the initial requests after service startup. This lower bandwidth reflects the inclusion of connection establishment overhead. When conducting benchmarks, it is recommended to perform a warm-up phase to ensure accurate performance measurements.

*Q. When my servers are running on different NVLink domains, some servers hang or have a lower performance. How to fix that?*

A. NVLink domain can be found with `nvidia-smi -q` in the `Fabric.ClusterUUID` field. A few UCX environment variables can be adjusted when your servers have different NVLink domains:

* `UCX_CUDA_IPC_ENABLE_MNNVL`: Set to `n`. This also can reduce UCX timeout error messages like `UCX  ERROR   cuMemImportFromShareableHandle failed: invalid resource handle`, although these errors don't necessarily cause your trtllm-serve to fail.

* `UCX_NET_DEVICES`: Check if this is set correctly, or unset this variable to allow UCX to use all possible devices.

* `UCX_RNDV_SCHEME`: Set to `get_zcopy` or `put_zcopy` on GB200 for better performance. The default value is `auto`.
