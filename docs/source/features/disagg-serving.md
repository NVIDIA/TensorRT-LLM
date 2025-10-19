# Disaggregated Serving (Beta)

```{note}
Note:
This feature is currently in beta, and the related APIs are subjected to change in future versions.
```

- [Motivation](#Motivation)
- [KV Cache Exchange](#KV-Cache-Exchange)
  - [Multi-backend Support](#Multi-backend-Support)
  - [Overlap Optimization](#Overlap-Optimization)
  - [Cache Layout Transformation](#Cache-Layout-Transformation)
- [Usage](#Usage)
  - [trtllm-serve](#trtllm-serve)
  - [Dynamo](#Dynamo)
- [Environment Variables](#Environment-Variables)
- [Troubleshooting and FAQ](#Troubleshooting-and-FAQ)

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

In TensorRT-LLM, the KV cache exchange is modularly decoupled from the KV cache manager and the underlying communication libraries, as shown in Figure 3. The KV cache exchange module is responsible for efficient transmission and reception of the cache, promptly releasing cache space, and performing cache layout conversions during the exchange process. Currently, mainstream communication protocols—MPI, UCX, and NIXL—are all supported by TensorRT-LLM, and the underlying communication protocols utilize RDMA / NVLink. Currently, we recommend using UCX and NIXL backends, as we are adding a dynamic scaling mechanism on top of them—specifically, dynamic node joining and leaving. This allows customers to adjust the load based on traffic demands or switch roles between context and generation dynamically.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture6.png" width="890" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 3. KV cache exchange architecture</em></sub></p>

### Overlap Optimization

To optimize the overall performance of disaggregated serving, TensorRT LLM overlaps the KV cache transmission with computation for multiple independent requests. While one request is sending or receiving its KV cache blocks, other requests can proceed with computation, as illustrated in Figure 4. Furthermore, if context and generation instances are using multiple GPUs per instance, KV cache transmission between different sets of GPUs can occur in parallel.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture7.png" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 4. KV cache exchange timing diagram</em></sub></p>

### Cache Layout Transformation

To minimize KV cache transmission latency, TensorRT LLM currently uses direct transmission between device memories for cache transfer. The KV cache transmission supports using different parallel strategies for the context and generation phases. In such cases, careful orchestration of KV cache block mapping is required. Figure 5 illustrates this using the example of context phase with TP2 and generation phase with PP2.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture8.png" width="680" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 5. KV cache layout conversion</em></sub></p>

The optimizations required for KV cache transmission vary depending on whether it's single-node multi-GPU, multi-node multi-GPU, or different GPU models. To accommodate this, TensorRT LLM provides a set of environment variables for selection in different environments. Please refer to the following section for details [Environment Variables](#Environment-Variables).

## Usage

### trtllm-serve

The first approach to do disaggregated LLM inference with TensorRT LLM involves launching a separate OpenAI-compatible server per context and generation instance using `trtllm-serve`. An additional server, referred to as the "disaggregated" server, is also launched with `trtllm-serve` and acts as an orchestrator which receives client requests and dispatches them to the appropriate context and generation servers via OpenAI REST API. Figure 6 below illustrates the disaggregated serving workflow when using this approach. When a context instance is done generating the KV blocks associated with the prompt, it returns a response to the disaggregated server. This response includes the prompt tokens, the first generated token and metadata associated with the context request and context instance. This metadata is referred to as context parameters (`ctx_params` in Figure 6). These parameters are then used by the generation instances to establish communication with the context instance and retrieve the KV cache blocks associated with the request.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture3.png" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 6. `trtllm-serve` integration with disaggregated service</em></sub></p>


To run TRT-LLM in disaggregated mode, you must first launch context (prefill) and generation (decode) servers using `trtllm-serve`.

We use the `cache_transceiver_config` configuration to set up disaggregated serving, which includes the following parameters:

```yaml
cache_transceiver_config:
  backend: <str>
  max_tokens_in_buffer: <int>
```

`backend` specifies the communication backend for transferring the kvCache, valid options include `DEFAULT`,`UCX`, `NIXL`, and `MPI`, the default backend is NIXL.

`max_tokens_in_buffer` defines the buffer size for kvCache transfers, it is recommended to set this value greater than or equal to the maximum ISL (Input Sequence Length) of all requests for optimal performance.

For example, you could launch two context servers and one generation servers as follows:

```

# Generate context_extra-llm-api-config.yml
# Overlap scheduler for context servers are disabled because it's not supported for disaggregated context servers yet
echo -e "disable_overlap_scheduler: True\ncache_transceiver_config:\n  backend: UCX\n  max_tokens_in_buffer: 2048" > context_extra-llm-api-config.yml

# Start Context servers
CUDA_VISIBLE_DEVICES=0 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8001 --backend pytorch --extra_llm_api_options ./context_extra-llm-api-config.yml &> log_ctx_0 &
CUDA_VISIBLE_DEVICES=1 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8002 --backend pytorch --extra_llm_api_options ./context_extra-llm-api-config.yml &> log_ctx_1 &

# Generate gen_extra-llm-api-config.yml
echo -e "cache_transceiver_config:\n  backend: UCX\n  max_tokens_in_buffer: 2048" > gen_extra-llm-api-config.yml

# Start Generation servers
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
when routing requests to the generation servers, the disaggregated server will mark the requests as "generation-only" to skip the context phase.

Clients can then send requests to the disaggregated server at `localhost:8000`, which is an OpenAI compatible endpoint. For example,  you can send requests to the disaggregated server using curl:
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "NVIDIA is a great company because",
        "max_tokens": 16,
        "temperature": 0
    }' -w "\n"
```

#### Launching disaggregated servers on SLURM clusters

Please refer to [Disaggregated Inference Benchmark Scripts](../../../examples/disaggregated/slurm).

### Dynamo

The second approach involves the use of [Dynamo](https://github.com/ai-dynamo/dynamo), a data center-scale inference server developed specifically for LLM workloads. Dynamo introduces several advanced features not present in the other methods, including decoupled pre- and post-processing workers, which are particularly beneficial under high concurrency conditions. The disaggregated LLM inference workflow with Dynamo is illustrated in Figure 7.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture4.png" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 7. Dynamo integration with disaggregated service</em></sub></p>

In the Dynamo workflow, requests are initially processed by pre- and post-processing workers, which then query a smart router to determine the optimal decode worker to route the requests to. Depending on the availability of KV cache blocks, the decoder worker may bypass the prefill stage or forward the request to the prefill worker. Once the prefill worker is done processing the prompt, the KV cache blocks can be sent from the prefill worker to the decoder worker, using the metadata referred to as ctx_params in the figure above.

Dynamo also includes built-in support for Kubernetes deployment, monitoring, and metrics collection. The development team is actively working on enabling dynamic instance scaling, further enhancing its suitability for production environments.

For more information on how to use Dynamo with TensorRT-LLM, please refer to [this documentation](https://docs.nvidia.com/dynamo/latest/examples/trtllm.html).

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

* ``UCX_MAX_RNDV_RAILS`: With the default value 2, UCX attempts to use two InfiniBand (IB) NIC devices per GPU for Rendezvous (RNDV) transfers. When both the context and generation instances enable tensor- and expert-parallel (TEP), multiple TP ranks may transfer KV cache concurrently. Because each TP rank can use up to two NIC devices, some NIC devices can be shared across GPUs, causing contention and reduced throughput. Setting UCX_MAX_RNDV_RAILS=1 can reduce contention in this case.

## Troubleshooting and FAQ

### General FAQs

*Q. What are the limitations of disaggregated serving in TRT-LLM?*

A. Currently, only decoder-only models and beam width of 1  are supported. Also the KV cache at each layer of the model is required to be homogeneous, with the same data type and the same number of attention heads.

*Q. When using the TRT backend, is the engine used for disaggregated serving different from other engines?*

A. No. There are no special requirements for the arguments to build engine.

*Q. When using the TRT backend, do the engines used by the context and generation instances need to be the same?*

A. No. The engines used by context and generation instances can be different, and their parallelism can be heterogeneous, i.e., TP,PP can be different, and TRT-LLM will handle the heterogeneity of KV cache.

*Q. Can a TRT-LLM server instance handle both context-only requests and generation-only requests?*

A. Yes, but it's not recommended. TRT-LLM does not implement optimal scheduling for the case where the instance handles mixed context-only requests and generation-only requests. It's better to run context-only requests and generation-only requests on sets of servers.

*Q. Does disaggregated serving in TRT-LLM support multi-gpu and multi-node?*

A. Yes, it's recommended that different server instances use different GPUs. We support running context and generation servers on the same node or different nodes. The `CUDA_VISIBLE_DEVICES` env variable can be used to control which GPUs are used by each instance.

### Debugging FAQs

*Q. How to handle error `Disaggregated serving is not enabled, please check the configuration?`*

A. please set `backendType` of `CacheTransceiverConfig`.
```cpp
ExecutorConfig executorConfig{...};

executorConfig.setCacheTransceiverConfig(texec::CacheTransceiverConfig(BackendType::DEFAULT));
```
*Q. Does TRT-LLM support using GPU direct RDMA for inter-node KV Cache transfer?*

A. Yes, TRT-LLM supports using GPU direct RDMA for inter-node KV cache transfer.

*Q. What causes the substantial bandwidth fluctuations in kvCache transfers, especially during the first few requests following service initialization?*

A. The communication for kvCache transfer between executors are established dynamically. The connection establishment process incurs significant overhead, which explains the apparently lower kvCache transfer bandwidth observed during the initial requests after service startup. This lower bandwidth reflects the inclusion of connection establishment overhead. When conducting benchmarks, it is recommended to perform a warm-up phase to ensure accurate performance measurements.

*Q. When my servers are running on different NVLink domains, some servers hang or have a lower performance. How to fix that?*

A. NVLink domain can be found with `nvidia-smi -q` in the `Fabric.ClusterUUID` field. A few UCX environment variables can be adjusted when your servers have different NVLink domains:

* `UCX_CUDA_IPC_ENABLE_MNNVL`: Set to `n`. This also can reduce UCX timeout error messages like `UCX  ERROR   cuMemImportFromShareableHandle failed: invalid resource handle`, although these errors don't necessarily cause your trtllm-serve to fail.

* `UCX_NET_DEVICES`: Check if this is set correctly, or unset this variable to allow UCX to use all possible devices.

* `UCX_RNDV_SCHEME`: Set to `get_zcopy` or `put_zcopy` on GB200 for better performance. The default value is `auto`.
