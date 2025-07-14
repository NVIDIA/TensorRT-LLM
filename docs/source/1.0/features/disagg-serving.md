# Disaggregated Serving

- [Motivation](#Motivation)
- [KV Cache Exchange](#KV-Cache-Exchange)
  - [Multi-backend Support](#Multi-backend-Support)
  - [Overlap Optimization](#Overlap-Optimization)
  - [Cache Layout Transformation](#Cache-Layout-Transformation)
- [Usage](#Usage)
  - [trtllm-serve](#trtllm-serve)
  - [Dynamo](#Dynamo)
- [Environment Variables](#Environment-Variables)
- [Troubleshooting and FAQ](#Troubleshooting-and-FAQ]

## Motivation

LLM inference has two stages: context (prefill) and generation (decode) phases. The context phase computes KV cache for prompt tokens whereas the generation phase generates tokens one by one using cached values. These phases have different compute characteristics.

There are two ways of serving LLM inference requests:

* Aggregated LLM serving (sometimes called in-flight batching or IFB in this tech blog), in which the context and generation phases are run on the same GPU.
* Disaggregated LLM serving, in which the context and generation phases are run on different GPUs.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture1.png" width="640" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 1. The execution timeline of aggregated LLM serving</em></sub></p>

In aggregated LLM serving, both the context and generation phases share the same GPU resources and parallelism strategy. This can lead to interference where context processing delays token generation, increasing token-to-token latency (TPOT) and reducing interactivity. This is illustrated in Figure 1 which shows the execution timeline for aggregated LLM serving. Aggregated LLM serving also forces a single GPU type and parallelism configuration for both phases, even though their compute needs differ. As a result, optimizing for one metric such as time-to-first-token (TTFT), often comes at the expense of another metric such as TPOT.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture2.png" width="580" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 2. The execution timeline of dis-aggregated LLM serving</em></sub></p>

Disaggregated serving resolves these challenges by decoupling the two phases, allowing each to run on separate GPU pools and using different parallelism strategies. This separation removes the interference between context and generation phases, as shown in Figure 2, and enables independent optimization of TTFT and TPOT. Although disaggregation incurs overhead for transferring the KV cache blocks from context to generation GPUs, the advantages can be substantial—particularly for workloads with long input sequences and moderate output lengths where interference is most severe.

You can also refer to [this paper](https://arxiv.org/pdf/2506.05508) for more details about the rational and design considerations of disaggregated serving.

## KV Cache Exchange

### Multi-backend Support

In TensorRT-LLM, the KV cache exchange is modularly decoupled from the KV cache manager and the underlying communication libraries, as shown in Figure 6. The KV cache exchange module is responsible for efficient transmission and reception of the cache, promptly releasing cache space, and performing cache layout conversions during the exchange process. Currently, mainstream communication protocols—MPI, UCX, and NIXL—are all supported by TensorRT-LLM, and the underlying communication protocols utilize RDMA / NVLink. Currently, we recommend using UCX and NIXL backends, as we are adding a dynamic scaling mechanism on top of them—specifically, dynamic node joining and leaving. This allows customers to adjust the load based on traffic demands or switch roles between context and generation dynamically.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture6.png" width="890" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 6. KV cache exchange architecture</em></sub></p>

### Overlap Optimization

To optimize the overall performance of disaggregated serving, TensorRT-LLM overlaps the KV cache transmission with computation for multiple independent requests. While one request is sending or receiving its KV cache blocks, other requests can proceed with computation, as illustrated in Figure 7. Furthermore, if context and generation instances are using multiple GPUs per instance, KV cache transmission between different sets of GPUs can occur in parallel.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture7.png" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 7. KV cache exchange timing diagram</em></sub></p>

### Cache Layout Transformation

To minimize KV cache transmission latency, TensorRT-LLM currently uses direct transmission between device memories for cache transfer. The KV cache transmission supports using different parallel strategies for the context and generation phases. In such cases, careful orchestration of KV cache block mapping is required. Figure 8 illustrates this using the example of context phase with TP2 and generation phase with PP2.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture8.png" width="680" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 8. KV cache layout conversion</em></sub></p>

The optimizations required for KV cache transmission vary depending on whether it's single-node multi-GPU, multi-node multi-GPU, or different GPU models. To accommodate this, TensorRT-LLM provides a set of environment variables for selection in different environments. Please refer to [this document](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/advanced/disaggregated-service.md) for details.

## Usage

### trtllm-serve

The first approach to do disaggregated LLM inference with TensorRT-LLM involves launching a separate OpenAI-compatible server per context and generation instance using `trtllm-serve`. An additional server, referred to as the "disaggregated" server, is also launched with `trtllm-serve` and acts as an orchestrator which receives client requests and dispatches them to the appropriate context and generation servers via OpenAI REST API. Figure 3 below illustrates the disaggregated serving workflow when using this approach. When a context instance is done generating the KV blocks associated with the prompt, it returns a response to the disaggregated server. This response includes the prompt tokens, the first generated token and metadata associated with the context request and context instance. This metadata is referred to as context parameters (`ctx_params` in Figure 3). These parameters are then used by the generation instances to establish communication with the context instance and retrieve the KV cache blocks associated with the request.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture3.png" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 3. `trtllm-serve` integration with disaggregated service</em></sub></p>


To run TRT-LLM in disaggregated mode, you must first launch context (prefill) and generation (decode) servers using `trtllm-serve`.

For example, you could launch two context servers and one generation servers as follows:

```
echo -e "disable_overlap_scheduler: True\ncache_transceiver_config:\n  max_num_tokens: 2048" > context_extra-llm-api-config.yml
echo -e "cache_transceiver_config:\n  max_num_tokens: 2048" > gen_extra-llm-api-config.yml

export TRTLLM_USE_UCX_KVCACHE=1
#Context servers
CUDA_VISIBLE_DEVICES=0 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8001 --backend pytorch --extra_llm_api_options ./context_extra-llm-api-config.yml &> log_ctx_0 &
CUDA_VISIBLE_DEVICES=1 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8002 --backend pytorch --extra_llm_api_options ./context_extra-llm-api-config.yml &> log_ctx_1 &
#Generation servers
CUDA_VISIBLE_DEVICES=2 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8003 --backend pytorch --extra_llm_api_options ./gen_extra-llm-api-config.yml &> log_gen_0 &
```
Once the context and generation servers are launched, you can launch the disaggregated
server, which will accept requests from clients and do the orchestration between context
and generation servers. The disaggregated server can be launched with:

```
trtllm-serve disaggregated -c disagg_config.yaml
```
where `disagg_config.yaml` contains information about the context and generation servers. For the current example,
it would look like:
```
hostname: localhost
port: 8000
backend: pytorch
context_servers:
  num_instances: 2
  urls:
      - "localhost:8001"
      - "localhost:8002"
generation_servers:
  num_instances: 1
  urls:
      - "localhost:8003"
```

When routing requests to the context servers, the disaggregated server will mark the requests as "context-only" to skip the generation phase. Similarly,
when routing requests to the generation serfvers, the disaggregated server will mark the requests as "generation-only" to skip the context phase.

Clients can then send requests to the disaggregated server at `localhost:8000`, which is an OpenAI compatible endpoint.

### Dynamo

The second approach involves the use of [Dynamo](https://github.com/ai-dynamo/dynamo), a data center-scale inference server developed specifically for LLM workloads. Dynamo introduces several advanced features not present in the other methods, including decoupled pre- and post-processing workers, which are particularly beneficial under high concurrency conditions. The disaggregated LLM inference workflow with Dynamo is illustrated in Figure 4.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture4.png" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 4. Dynamo integration with disaggregated service</em></sub></p>

In the Dynamo workflow, requests are initially processed by pre- and post-processing workers, which then query a smart router to determine the optimal decode worker to route the requests to. Depending on the availability of KV cache blocks, the decoder worker may bypass the prefill stage or forward the request to the prefill worker. Once the prefill worker is done processing the prompt, the KV cache blocks can be sent from the prefill worker to the decoder worker, using the metadata referred to as ctx_params in the figure above.

Dynamo also includes built-in support for Kubernetes deployment, monitoring, and metrics collection. The development team is actively working on enabling dynamic instance scaling, further enhancing its suitability for production environments.

For more information on how to use Dynamo with TensorRT-LLM, please refer to [this documentation](https://docs.nvidia.com/dynamo/latest/examples/trtllm.html).

## Environment Variables

TRT-LLM uses some environment variables to control the behavior of disaggregated serving.

* `TRTLLM_USE_MPI_KVCACHE`: Whether to use MPI to transfer KV cache. Currently, the default value is `0`.

* `TRTLLM_USE_UCX_KVCACHE`: Whether to use UCX to transfer KV cache. Currently, the default value is `0`. To use disaggregated serving, either `TRTLLM_USE_MPI_KVCACHE=1` or `TRTLLM_USE_UCX_KVCACHE=1` is required to be set.

* `TRTLLM_PARALLEL_CACHE_SEND`: If set to `1`, contextExecutor will attempt to send KV cache for multiple requests in parallel. The default value is `0`.

* `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP`: If set to `1`, generationExecutor will not overlap KV cache transfer with model inference. The default value is `0`.

* `TRTLLM_ENABLE_KVCACHE_RECEIVE_PARALLEL`:  When the generation rank receives KV cache from multiple context ranks within a single context instance, it will receive KV cache from each rank sequentially. If set to `1`, the generation rank will receive KV cache from each rank within one context instance in parallel. The default value is `0`.

* `TRTLLM_REQUEST_KV_CACHE_CONCURRENT`: If set to `1`, generationExecutor prepares independent resources for each context executor to receive KV cache, requests whose KV cache are received from different context executors will be processed concurrently. If set to `0`, the generation executor will reuse the same resource to process KV cache transfer for each request sequentially, reducing the resources used by KV cache transmission and thereby lowering the risk of running out of memory. The default value is `0`.

* `TRTLLM_TRY_ZCOPY_FOR_KVCACHE_TRANSFER`: TRT-LLM typically copies non-contiguous data into a temporary buffer before sending KV cache. If set to `1`, TRT-LLM will attempt to directly transmit each KV cache block, eliminating extra copies. The default value is `0`.

* `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE`: By default, TRT-LLM uses a `stream-ordered memory allocator` to allocate temporary buffers. If this environment variable is set to #Size, TRT-LLM will use `cudaMalloc` to allocate buffer of size #Size for KV cache transmission. The default value is `512MB`. Users can set `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE=1GB` to allocate a 1 GB buffer with `cudaMalloc` for KV cache transmission.

* `TRTLLM_KVCACHE_TRANSFER_USE_ASYNC_BUFFER`: If set to `1`, TRT-LLM will use `cudaMallocAsync` to allocate buffers for KV cache transmission. The default value is `0`. This environment variable only takes effect when `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE` is greater than 0.

* `TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM`: The maximum number of concurrent KV cache sends. The default value is `4`. This environment variable only takes effect when `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE` is greater than 0.

## Troubleshooting and FAQ

### General FAQs

*Q. What are the limitations of disaggregated serving in TRT-LLM?*

A. Currently, only decoder-only models and beam width of 1  are supported. Also the KV cache at each layer of the model is required to be homogeneous, with the same data type and the same number of attention headers.

*Q. When using the TRT backend, is the engine used for disaggregated serving different from other engines?*

A. No. There are no special requirements for the arguments to build engine.

*Q. When using the TRT backend, do the engines used by the context and generation instances need to be the same?*

A. No. The engines used by context and generation instances can be different, and their parallelism can be heterogeneous, i.e., TP,PP can be different, and TRT-LLM will handle the heterogeneity of KV cache.

*Q. Can a TRT-LLM server instance handle both context-only requests and generation-only requests?*

A. Yes, but it's not recommended. TRT-LLM does not implement optimal scheduling for the case where the instance handles mixed context-only requests and generation-only requests. It's better to run context-only requests and generation-only requests on sets of servers.

*Q. Does disaggregated serving in TRT-LLM support multi-gpu and multi-node?*

A. Yes, it's recommended that different server instances use different GPUs. We support running context and generation servers on the same node or different nodes. The `CUDA_VISIBLE_DEVICES` env variable can be used to control which GPUs are used by each instance.

*Q. What's the requirement for disaggregated serving in TRT-LLM?*

A. TRT-LLM requires `UCX`-backend `CUDA-aware MPI` currently, TRT-LLM implements KV cache transfer with [`CUDA-aware MPI`](https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/cuda.html#how-do-i-build-open-mpi-with-cuda-aware-support), and will support more communication components for KV cache transfer in future version.

### Debugging FAQs

*Q. How to handle error `Disaggregated serving is not enabled, please check the configuration?`*

A. please set the environment variables
```
export TRTLLM_USE_MPI_KVCACHE=1
```
or
```
export TRTLLM_USE_UCX_KVCACHE=1
```
When the environment variable `TRTLLM_USE_MPI_KVCACHE=1` is set, TRT-LLM will transfer the KV cache using `CUDA-aware MPI`. All executor processes involved must share the same MPI world communicator. Consequently, with `TRTLLM_USE_MPI_KVCACHE=1`, TRT-LLM only supports launching multiple executors via `MPI`. Additionally, the `CommunicationMode` for the executors must be set to `kLEADER` or `kORCHESTRATOR` with `SpawnProcesses=false` for the `disaggregated-service`. These restrictions do not apply when `TRTLLM_USE_UCX_KVCACHE=1` is set.


*Q. Why do some profiling tools show that TRT-LLM's KV cache transfer does not utilize NVLink even on devices equipped with NVLink?*

A. Ensure TRT-LLM is running with `UCX`-backend `CUDA-aware MPI` , and check version of `UCX` with `ucx_info -v`.
If the version of UCX <=1.17, set the environment variables `UCX_RNDV_FRAG_MEM_TYPE=cuda` and `UCX_MEMTYPE_CACHE=n` to enable NVLink. For BlackWell architecture GPUs, UCX version >=1.19 is required to enable NVLink.
If the version of UCX >=1.18, there are several ways to enable NVLink:
1. Set the environment variables `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE=0B`,`UCX_CUDA_COPY_ASYNC_MEM_TYPE=cuda`, `UCX_CUDA_COPY_DMABUF=no`, `UCX_MEMTYPE_CACHE=n` and `UCX_RNDV_PIPELINE_ERROR_HANDLING=y`.
2. Set the environment variables `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE=$Size`, `UCX_MEMTYPE_CACHE=n` and `UCX_RNDV_PIPELINE_ERROR_HANDLING=y`. $Size represents the size of the buffer for KV cache transfer, which is recommended to be larger than the size of the KV cache for the longest request.

*Q. Does TRT-LLM support using GPU direct RDMA for inter-node KV Cache transfer?*

A. Yes, TRT-LLM supports using GPU direct RDMA for inter-node KV cache transfer, but it is not enabled by default. There are several ways to enable GPU direct RDMA:
1. Set the environment variables `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE=0B`,`UCX_RNDV_FRAG_MEM_TYPE=cuda`, `UCX_MEMTYPE_CACHE=n` and `UCX_RNDV_PIPELINE_ERROR_HANDLING=y`.
2. Set the environment variables `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE=$Size`, `UCX_MEMTYPE_CACHE=n` and `UCX_RNDV_PIPELINE_ERROR_HANDLING=y`, $Size represents the size of the buffer for KV cache transfer, which is recommended to be larger than the size of the KV cache for the longest request.
To achieve the optimal performance when using GPU direct RDMA, it is advisable to create CUDA context before MPI initialization when TRTLLM_USE_MPI_KVCACHE=1 is set. One possible approach is to rely on MPI environment variables to set the correct device before MPI initialization.

*Q. Are there any guidelines for performance tuning of KV cache transfer?*

A. Depending on the user's use case, certain sets of environment variables can help avoid poor KV cache transfer performance.

Environment Variable Set A

```
export TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE=0B
export UCX_RNDV_FRAG_MEM_TYPES=cuda
export UCX_MEMTYPE_CACHE=n
export UCX_RNDV_PIPELINE_ERROR_HANDLING=y
```
This set allows KV cache transfers to utilize NVLink within nodes and GDRDMA between nodes.

Environment Variable Set B

```
export TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE=0B
export UCX_CUDA_COPY_ASYNC_MEM_TYPE=cuda
export UCX_CUDA_COPY_DMABUF=no
export UCX_MEMTYPE_CACHE=n
export UCX_RNDV_PIPELINE_ERROR_HANDLING=y
```
Set B may provide slightly better performance on a single node compared to Set A. However, when transferring KV cache across multiple nodes, it may cause program instability.

Environment Variable Set C

```
export TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE=$Size
export UCX_MEMTYPE_CACHE=n
export UCX_RNDV_PIPELINE_ERROR_HANDLING=y
```
Set C can achieve better performance than Sets A and B, both within and between nodes. However, if the KV cache size exceeds the specified $Size, performance may degrade.
