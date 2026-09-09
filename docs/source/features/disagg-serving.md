# Disaggregated Serving

- [Motivation](#motivation)
- [KV Cache Exchange](#kv-cache-exchange)
  - [Multi-backend Support](#multi-backend-support)
  - [Overlap Optimization](#overlap-optimization)
  - [Cache Layout Transformation](#cache-layout-transformation)
  - [Unique Global Request ID](#unique-global-request-id)
- [Usage](#usage)
  - [Dynamo](#dynamo)
  - [trtllm-serve](#trtllm-serve)
  - [Multiple Instances](#multiple-instances)
  - [Coordinator and Worker Fleet](#coordinator-and-worker-fleet)
- [Environment Variables](#environment-variables)
- [Troubleshooting and FAQ](#troubleshooting-and-faq)

For the internals of the component that actually moves the KV blocks, see
[Introduction to KV Cache Transmission](../developer-guide/kv-transfer.md).

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

You can also refer to [this paper](https://arxiv.org/pdf/2506.05508) for more details about the rationale and design considerations of disaggregated serving.

## KV Cache Exchange

### Multi-backend Support

In TensorRT-LLM, the KV cache exchange is modularly decoupled from the KV cache manager and the underlying communication libraries, as shown in Figure 3. The KV cache exchange module is responsible for efficient transmission and reception of the cache, promptly releasing cache space, and performing cache layout conversions during the exchange process. Use `backend: NIXL`. It transfers over RDMA / NVLink, and a dynamic scaling mechanism—specifically, dynamic node joining and leaving—is being built on top of it. This allows customers to adjust the load based on traffic demands or switch roles between context and generation dynamically.

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

The optimizations required for KV cache transmission vary depending on whether it's single-node multi-GPU, multi-node multi-GPU, or different GPU models. To accommodate this, TensorRT LLM provides a set of environment variables for selection in different environments. Please refer to the following section for details [Environment Variables](#environment-variables).

### Unique Global Request ID

The context and generation phases of one request must share a single request ID: the ctx↔gen KV-cache transfer is keyed by it, so a collision (two in-flight requests with the same ID) corrupts the transfer. This shared ID is carried on `DisaggregatedParams.disagg_request_id`.

The disaggregated server generates this ID itself as a **snowflake** — a self-contained 64-bit positive integer that is unique without any cross-process coordination. The bit layout is:

```
[ 0 (1 bit) | timestamp_ms (39 bits) | node_id (8 bits) | process_id (6 bits) | counter (10 bits) ]
```

- `node_id` (0–255) identifies the node (defaults to a hash of the MAC address; overridable via `node_id` in the disaggregated config).
- `process_id` (0–63) identifies the orchestrator process on that node. In a [coordinator + worker fleet](#coordinator-and-worker-fleet) each fleet worker receives a distinct value, so co-located workers never emit the same ID in the same millisecond. It is set from the `TRTLLM_DISAGG_WORKER_PROCESS_ID` environment variable (assigned automatically per worker by the launcher).
- The `(node_id, process_id)` pair therefore makes the ID unique across all orchestrator processes without a shared counter or an extra network round trip — each worker mints its own IDs locally.

Global disaggregated IDs occupy the range `[1 << 40, 2**63)`; worker-local and warm-up request IDs occupy the disjoint range `[0, 1 << 40)`, so the two never collide. If a client supplies its own positive `disagg_request_id`, that value is used verbatim and must be globally unique; when unset, the server mints a snowflake ID as above.

## Usage

### Dynamo

The first approach involves the use of [Dynamo](https://github.com/ai-dynamo/dynamo), a data center-scale inference server developed specifically for LLM workloads. Dynamo introduces several advanced features not present in the other methods, including decoupled pre- and post-processing workers, which are particularly beneficial under high concurrency conditions. The disaggregated LLM inference workflow with Dynamo is illustrated in Figure 7.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture4.png" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 7. Dynamo integration with disaggregated service</em></sub></p>

In the Dynamo workflow, requests are initially processed by pre- and post-processing workers, which then query a smart router to determine the optimal decode worker to route the requests to. Depending on the availability of KV cache blocks, the decoder worker may bypass the prefill stage or forward the request to the prefill worker. Once the prefill worker is done processing the prompt, the KV cache blocks can be sent from the prefill worker to the decoder worker, using the metadata referred to as ctx_params in the figure above.

Dynamo also includes built-in support for Kubernetes deployment, monitoring, and metrics collection. The development team is actively working on enabling dynamic instance scaling, further enhancing its suitability for production environments.

For more information on how to use Dynamo with TensorRT-LLM, please refer to [this documentation](https://docs.nvidia.com/dynamo/backends/tensor-rt-llm).

### trtllm-serve

The second approach to evaluate disaggregated LLM inference with TensorRT LLM involves launching a separate OpenAI-compatible server per context and generation instance using `trtllm-serve`. An additional server, referred to as the "disaggregated" server, is also launched with `trtllm-serve` and acts as an orchestrator which receives client requests and dispatches them to the appropriate context and generation servers via OpenAI REST API. Figure 6 below illustrates the disaggregated serving workflow when using this approach. When a context instance is done generating the KV blocks associated with the prompt, it returns a response to the disaggregated server. This response includes the prompt tokens, the first generated token and metadata associated with the context request and context instance. This metadata is referred to as context parameters (`ctx_params` in Figure 6). These parameters are then used by the generation instances to establish communication with the context instance and retrieve the KV cache blocks associated with the request.

```{eval-rst}
.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-config-flag-alias
   :end-before: .. end-note-config-flag-alias
```

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
  backend: NIXL
  max_tokens_in_buffer: <int>
  kv_transfer_timeout_ms: <int>
  kv_cache_bounce_size_mb: <int>
```

`backend` selects the communication library used to transfer the KV cache. Set it to `NIXL`, which transfers over RDMA / NVLink. The field has no default — if it is left unset, the worker still starts, but it brings up no cache transceiver and rejects the disaggregated requests it is then routed. Set the same value on the context and the generation worker.

`max_tokens_in_buffer` is best left unset. It bounds how many KV transfers a generation worker admits concurrently, and the built-in default is derived from the model's maximum sequence length, so a small hand-written value only throttles the transfer path.

`kv_transfer_timeout_ms` bounds how long a request may wait for its KV cache before it is cancelled and cleaned up. The default is `60000`.

`kv_cache_bounce_size_mb` is `0` by default, which sends each KV block separately. Setting it to a positive size coalesces a request's blocks into one contiguous buffer of that many MiB per direction and issues a single NIXL write, which helps when a request's blocks are scattered. It requires fabric (MNNVL) memory.

For example, you could launch two context servers and one generation server as follows:

```bash

# Generate context_config.yml
echo -e "disable_overlap_scheduler: True\ncache_transceiver_config:\n  backend: NIXL" > context_config.yml

# Start Context servers
CUDA_VISIBLE_DEVICES=0 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8001 --backend pytorch --config ./context_config.yml &> log_ctx_0 &
CUDA_VISIBLE_DEVICES=1 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8002 --backend pytorch --config ./context_config.yml &> log_ctx_1 &

# Generate gen_config.yml
echo -e "cache_transceiver_config:\n  backend: NIXL" > gen_config.yml

# Start Generation servers
CUDA_VISIBLE_DEVICES=2 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8003 --backend pytorch --config ./gen_config.yml &> log_gen_0 &
```

Both workers must carry the same `cache_transceiver_config`; confirm each one brought the transceiver up with `grep "Using KvCacheTransceiverV2" log_ctx_0 log_gen_0`.

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

The config also accepts an optional field that tunes the HTTP listeners:

- `server_keep_alive_timeout` (int, default `10`) — HTTP keep-alive timeout in seconds, applied to the client-facing listener and to the coordinator's listener when it runs in-process (see [Coordinator and Worker Fleet](#coordinator-and-worker-fleet)). Raise it (for example, `3600`) when clients hold large idle connection pools and hit `Connection reset by peer` on a reused connection: the server closing an idle connection first leaves the client with a half-closed socket that fails on the next request.

Clients can then send requests to the disaggregated server at `localhost:8000`, which is an OpenAI-compatible endpoint. For example, you can send requests to the disaggregated server using curl:
```
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

Please refer to [Disaggregated Inference Benchmark Scripts](source:examples/disaggregated/slurm).

### Multiple Instances

To increase maximum concurrency without more GPU nodes, you can deploy multiple disaggregated server instances across different nodes, while each instance manages the same context/generation servers. This is helpful when one disaggregated server becomes a performance bottleneck or runs out of ephemeral ports.

Example (two-node deployment):

- **Node A**
  - Context servers: `node-a:8001`
  - Generation servers: `node-b:8002`
  - Disaggregated orchestrator endpoint: `node-a:8000`
- **Node B**
  - Context servers: `node-a:8001`
  - Generation servers: `node-b:8002`
  - Disaggregated orchestrator endpoint: `node-b:8000`
- **Client entrypoint**
  - Send requests or use a load balancer forwarding to `node-a:8000` and `node-b:8000`

### Coordinator and Worker Fleet

A single disaggregated server process is itself a single-threaded orchestrator and can become a throughput bottleneck (it terminates every client connection, runs routing, and proxies the ctx→gen hop). To scale the orchestrator on one node without standing up multiple independent instances, `trtllm-serve disaggregated` can run a **fleet** of stateless disaggregated-server worker processes behind a shared **coordinator**.

The two roles split as follows:

- **Coordinator** — a single process that owns all cluster state: the ctx/gen routers, worker readiness, and (for the KV-cache-aware router) the single ZMQ event-ingest endpoint. It exposes an internal coordination API (`/select`, `/finish`, `/cluster_info`, `/health`).
- **Fleet workers** — `num_workers` stateless disaggregated servers that share the public port via `SO_REUSEPORT` (each worker is its own process binding the same port, so the kernel load-balances incoming connections across them by 4-tuple hash). Each holds a lightweight delegating client: it computes the routing key locally (e.g. block hashes) and delegates the placement decision to the coordinator over HTTP. Workers own no routing state, so routing stays globally consistent no matter which worker terminates a connection. Each worker also gets a distinct `process_id` for the [global request ID](#unique-global-request-id).

This is controlled by two fields in the disaggregated config:

- `num_workers` (int, default `1`) — number of disaggregated-server worker processes to run on the public port.
- `disagg_coordinator_url` (str, optional) — URL of an already-running coordinator. When set, this process starts **no** coordinator and its fleet delegates to that external one.

The three resulting topologies:

| `num_workers` | `disagg_coordinator_url` | Behavior |
|---------------|--------------------------|----------|
| `1` | unset | Single self-contained server with an in-process coordinator (the default; unchanged from earlier examples). |
| `> 1` | unset | An **implicit** coordinator starts in this process (on `port - 1`) and a fleet of `num_workers` delegating servers runs on the public port. |
| any | set | **No** coordinator starts here; a fleet of `num_workers` delegating servers points at the external `disagg_coordinator_url`. |

```{note}
The fleet is most useful with a *stateful* router (`kv_cache_aware`, `conversation`) where placement must be globally consistent — that decision is delegated to the coordinator. With a *stateless* router (`round_robin`, `load_balancing`) each worker simply places locally and no coordinator round-trip occurs.
```

#### Example: implicit coordinator + 4-worker fleet

Extend the `disagg_config.yaml` from the [trtllm-serve](#trtllm-serve) example with `num_workers` and a router type:

```yaml
hostname: localhost
port: 8000
backend: pytorch
# Run 4 stateless disaggregated-server workers on port 8000, with an implicit
# coordinator started in-process on port 7999 (port - 1).
num_workers: 4
context_servers:
  num_instances: 2
  urls:
      - "localhost:8001"
      - "localhost:8002"
  router:
    type: kv_cache_aware
generation_servers:
  num_instances: 1
  urls:
      - "localhost:8003"
  router:
    type: kv_cache_aware
```

Launch it exactly as before — the coordinator and fleet are started for you:

```bash
trtllm-serve disaggregated -c disagg_config.yaml
```

Clients still send requests to the public endpoint (`localhost:8000`); the fleet transparently delegates routing to the coordinator.

#### Example: external coordinator

To point a fleet at a coordinator already running elsewhere (for example, one shared across nodes), set `disagg_coordinator_url` and omit the coordinator from this process:

```yaml
hostname: localhost
port: 8000
backend: pytorch
num_workers: 4
disagg_coordinator_url: "http://coordinator-host:7999"
context_servers:
  num_instances: 2
  urls:
      - "localhost:8001"
      - "localhost:8002"
  router:
    type: kv_cache_aware
generation_servers:
  num_instances: 1
  urls:
      - "localhost:8003"
  router:
    type: kv_cache_aware
```

```{note}
A fleet worker fails fast if its coordinator is unreachable: on startup it probes the coordinator's `/cluster_info` with bounded retry (up to `--server_start_timeout` seconds) and exits with an error rather than coming up and returning `Cluster is not ready` for every request.
```

## Environment Variables

TRT-LLM uses some environment variables to control the behavior of disaggregated service.

* `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP`: If set to `1`, the generation worker will not overlap KV cache transfer with model inference. The default value is `0`.

* `TRTLLM_NIXL_KVCACHE_BACKEND`: Selects the transport NIXL itself uses. Valid values are `UCX` (default) and `LIBFABRIC`; an unsupported value logs a warning and falls back to `UCX`. `LIBFABRIC` additionally requires a NIXL build carrying the libfabric plugin — see the [disaggregated serving examples](source:examples/disaggregated/README.md).

* `TRTLLM_GPU_KEEPALIVE`: If set to `1`, a generation worker that is waiting at the benchmark fill gate (`TLLM_BENCHMARK_REQ_QUEUES_SIZE`) keeps a resident warp on every SM in ~100 ms chunks instead of idling through the wait, so GPU-activity metrics do not read idle while the context tier fills it. The work is drained when the gate opens and never overlaps a forward pass. The default value is `0`.

There are some other useful environment variables that may help when encountering failures or performance issues.

* `NCCL_GRAPH_MIXING_SUPPORT`: TensorRT-LLM now initializes common NCCL communicators with graph
  mixing support off by default to reduce launch overhead for CUDA graph-captured NCCL operations.
  This assumes the communicator is not used by parallel graph launches or by uncaptured NCCL calls
  while a graph launch is outstanding. Set `NCCL_GRAPH_MIXING_SUPPORT=1` to restore NCCL's default
  graph mixing behavior if your workload needs it. For more details, see the
  [NCCL_GRAPH_MIXING_SUPPORT documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-mixing-support).

* `UCX_MAX_RNDV_RAILS`: With the default value 2, UCX attempts to use two InfiniBand (IB) NIC devices per GPU for Rendezvous (RNDV) transfers. When both the context and generation instances enable tensor- and expert-parallel (TEP), multiple TP ranks may transfer KV cache concurrently. Because each TP rank can use up to two NIC devices, some NIC devices can be shared across GPUs, causing contention and reduced throughput. Setting UCX_MAX_RNDV_RAILS=1 can reduce contention in this case.

## Troubleshooting and FAQ

### General FAQs

*Q. What are the limitations of disaggregated serving in TRT-LLM?*

A. Currently, only decoder-only models are supported. Also the KV cache at each layer of the model is required to be homogeneous, with the same data type and the same number of attention heads. The context and generation instances may use different parallelism, that is, TP and PP can differ, and TRT-LLM will handle the heterogeneity of KV cache.

*Q. Can a TRT-LLM server instance handle both context-only requests and generation-only requests?*

A. Yes, but it's not recommended. TRT-LLM does not implement optimal scheduling for the case where the instance handles mixed context-only requests and generation-only requests. It's better to run context-only requests and generation-only requests on sets of servers.

*Q. Does disaggregated serving in TRT-LLM support multi-gpu and multi-node?*

A. Yes, it's recommended that different server instances use different GPUs. We support running context and generation servers on the same node or different nodes. The `CUDA_VISIBLE_DEVICES` env variable can be used to control which GPUs are used by each instance.

### Debugging FAQs

*Q. How to handle error `Disaggregated serving is not enabled, please check the configuration?`*

A. `cache_transceiver_config.backend` has no default, so leaving it unset disables the transceiver on that worker. Set it in the worker's `--config` file, on both the context and the generation worker:

```yaml
cache_transceiver_config:
  backend: NIXL
```

*Q. How do I confirm that a worker actually brought up the cache transceiver?*

A. Each worker logs the outcome at INFO level, which `trtllm-serve` enables by default:

```bash
grep -E "Using KvCacheTransceiverV2|cache_transceiver is disabled" log_ctx_0 log_gen_0
```

Check the context and the generation worker separately, and configure the two together — nothing negotiates the transfer settings across workers, so a mismatch is not reported as a configuration error.

*Q. Does TRT-LLM support using GPU direct RDMA for inter-node KV Cache transfer?*

A. Yes, TRT-LLM supports using GPU direct RDMA for inter-node KV cache transfer.

*Q. How do I debug a suspected hang from overlapping NCCL graph operations?*

A. TensorRT-LLM turns graph mixing support off by default for common NCCL communicators. To check if
a hang might be related to NCCL graph mixing support, set `NCCL_GRAPH_MIXING_SUPPORT=1` to restore
NCCL's default graph mixing behavior.

*Q. What causes the substantial bandwidth fluctuations in kvCache transfers, especially during the first few requests following service initialization?*

A. The communication for kvCache transfer between executors are established dynamically. The connection establishment process incurs significant overhead, which explains the apparently lower kvCache transfer bandwidth observed during the initial requests after service startup. This lower bandwidth reflects the inclusion of connection establishment overhead. When conducting benchmarks, it is recommended to perform a warm-up phase to ensure accurate performance measurements.

*Q. When my servers are running on different NVLink domains, some servers hang or have a lower performance. How to fix that?*

A. NVLink domain can be found with `nvidia-smi -q` in the `Fabric.ClusterUUID` field. A few UCX environment variables can be adjusted when your servers have different NVLink domains:

* `UCX_CUDA_IPC_ENABLE_MNNVL`: Set to `n`. This also can reduce UCX timeout error messages like `UCX  ERROR   cuMemImportFromShareableHandle failed: invalid resource handle`, although these errors don't necessarily cause your trtllm-serve to fail.

* `UCX_NET_DEVICES`: Check if this is set correctly, or unset this variable to allow UCX to use all possible devices.

* `UCX_RNDV_SCHEME`: Set to `get_zcopy` or `put_zcopy` on GB200 for better performance. The default value is `auto`.
