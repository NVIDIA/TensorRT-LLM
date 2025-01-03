(disaggregated-service)=

# Disaggregated-Service (experimental)


```{note}
Note:
This feature is currently experimental, and the related API is subjected to change in future versions.
```

Currently TRT-LLM supports `disaggregated-service`, where the context and generation phases of a request can run on different executors. TRT-LLM's disaggregated service relies on the executor API, please make sure to read the [executor page](executor.md) before reading the document.

For more information on disaggregated service in LLM inference, one can refer to papers such as [DistServe](https://arxiv.org/abs/2401.09670), [SplitWise](https://arxiv.org/abs/2311.18677).


## Usage

```cpp
enum class RequestType
{
    REQUEST_TYPE_CONTEXT_AND_GENERATION = 0,
    REQUEST_TYPE_CONTEXT_ONLY = 1,
    REQUEST_TYPE_GENERATION_ONLY = 2
};
```
The TRT-LLM executor can execute three types of requests: `REQUEST_TYPE_CONTEXT_AND_GENERATION`, `REQUEST_TYPE_CONTEXT_ONLY`, and `REQUEST_TYPE_GENERATION_ONLY`. Executor executes the context phase of the context-only request and the generation phase of the generation-only request. When the executor completes the context phase of a context-only request, it will maintain the corresponding kvCache, which will be requested by the executor handling the subsequent generation-only request.

Note that the environment variable `TRTLLM_USE_MPI_KVCACHE=1` should be set for executing context-only request or generation-only request.



Here are some key APIs to use disaggregated service:
```cpp

Request request{...};

request.setRequestType(tensorrt_llm::executor::RequestType::REQUEST_TYPE_CONTEXT_ONLY);

auto contextRequestId = contextExecutor.enqueueRequest(request);

auto contextResposnes = contextExecutor.awaitResponses(contextRequestId);

auto contextPhaseParams = contextResposnes.back().getResult().contextPhaseParams.value();

request.setContextPhaseParams(contextPhaseParams);
request.setRequestType(tensorrt_llm::executor::RequestType::REQUEST_TYPE_GENERATION_ONLY);

auto generationRequestId = generationExecutor.enqueueRequest(request);

auto genResposnes = generationExecutor.awaitResponses(generationRequestId);

```

The generationExecutor will require data such as kvCache from the corresponding contextExecutor based on the `contextPhaseParams` attached to the request, so please make sure that the corresponding contextExecutor is not shut down before getting the generationExecutor's response.

In the code above, the `requestId` assigned to a request by different executors may be different, it is the user's responsibility to manage the mapping of the `requestId` for context-only requests to the `requestId` for generation-only requests.


![disaggregated-service usage](images/disaggregated-service_usage.png)

An `orchestrator` is required in `disaggregated-service` to manage multiple executor instance and route requests to different executors, TRT-LLM provides class `DisaggExecutorOrchestrator` in `cpp/include/tensorrt_llm/executor/disaggServerUtil.h` to help user to launch multiple executor instances, however, `DisaggExecutorOrchestrator` only routes requests to executors in a simple round-robin policy, users need to implement their own orchestrator for disaggregated-service based on their business.

TRT-LLM currently implements kvCache transfer using `CUDA-aware MPI`, and all executor processes involved need to hold same MPI world communicator. Therefore, TRT-LLM only supports launching multiple executors using `MPI`, and the `CommunicationMode` of the executors must be set to `KLEADER` or `kORCHESTRATOR` with `SpawnProcesses=false` for `disaggregated-service`, TRT-LLM will relax this restriction in future version to manage executors with greater ease.

## Benchmarks

Please refer to `benchmarks/cpp/disaggServerBenchmark.cpp` and `benchmarks/cpp/README.md`


## Troubleshooting and FAQ

### General FAQs

*Q. What are the limitations of disaggregated service in TRT-LLM?*

A. Currently, only `decoder-only engine` and `beamWidth=1` are supported, and the kvCache at each layer of the model is required to be homogeneous, with the same data type and the same number of attention headers.

*Q. Is the engine used by disaggregated_service any different from other engines?*

A. No. There are no special requirements for the arguments of the engine build.

*Q. Do the engine used by the context executor and generation executor need to be the same?*

A. No. The engine used by context executor and generation executor can be different, and their parallelism can be heterogeneous, i.e., TP,PP can be different, and TRT-LLM will handle the heterogeneity of kvCache.

*Q. Does TRT-LLM support running multiple context executor instances and generation executor instances?*

A. Yes. TRT-LLM supports running multiple context executors and generation executors at the same time, and each executor can use different engine, but it is the user's responsibility to route requests to different executors and  manage `requestId`.

*Q. Can an executor run both context-only requests and generation-only requests?*

A. Yes, but it's not recommended, and it's better to run context-only requests and generation-only requests on different executors.


*Q. Does disaggregated-Service in TRT-LLM support multi-gpu and multi-node?*

A. Yes, it's recommended that different executor use different GPUs . We support context-only executor and genertion-only executor run on same node or different node. The `participantIds` and `deviceIds` used by each executor need to be explicitly set by the user, and the `participantIds` of each executor must not be intersecting.

*Q. What's the requirement for disaggregated-service in TRT-LLM?*

A. TRT-LLM requires `UCX`-backend `CUDA-aware-MPI` currently, TRT-LLM implement kvCache transfer with [`CUDA-aware MPI`](https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/cuda.html#how-do-i-build-open-mpi-with-cuda-aware-support), and will support more communication components for kvCache transfer in future version.

### Debugging FAQs

*Q. How to handle error `Disaggregated serving is not enabled, please check the configuration?`*

A. please set env
```
export TRTLLM_USE_MPI_KVCACHE=1
```

*Q. Why do some profiling tools show that TRT-LLM's kvCache transfer does not utilize NVLink even on devices equipped with NVLink?*

A. Ensure run TRT-LLM with `UCX`-backend `CUDA-aware-MPI` ,and check version of `UCX` with `ucx_info -v`.
If version of UCX <=1.17, set `UCX_RNDV_FRAG_MEM_TYPE=cuda` to enable NVLink.
If version of UCX =1.18, set `UCX_CUDA_COPY_ASYNC_MEM_TYPE=cuda` and `UCX_CUDA_COPY_DMABUF=no`


*Q. Why is the token-token latency for generation-only requests unstable?*

A. The current version of TRT-LLM does not implement the overlap between kvCache transfers and engine compute for generation-only requests, which will be fixed in a future release.


*Q. How to handle error `All available sequence slots are used`?*

A. If generation_engine's pp_size >1, the error "All available sequence slots are used" may occur, will be fixed in a future release.
