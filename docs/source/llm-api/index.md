# LLM API Introduction

The LLM API is a high-level Python API designed to streamline LLM inference workflows.

It supports a broad range of use cases, from single-GPU setups to multi-GPU and multi-node deployments, with built-in support for various parallelism strategies and advanced features. The LLM API integrates seamlessly with the broader inference ecosystem, including NVIDIA [Dynamo](https://github.com/ai-dynamo/dynamo).

While the LLM API simplifies inference workflows with a high-level interface, it is also designed with flexibility in mind. Under the hood, it uses a PyTorch-native and modular backend, making it easy to customize, extend, or experiment with the runtime.


## Quick Start Example
A simple inference example with TinyLlama using the LLM API:

```{literalinclude} ../../../examples/llm-api/quickstart_example.py
    :language: python
    :linenos:
```

For more advanced usage including distributed inference, multimodal, and speculative decoding, please refer to this [README](source:examples/llm-api/README.md).

## Model Input

The `LLM()` constructor accepts either a Hugging Face model ID or a local model path as input.

### 1. Using a Model from the Hugging Face Hub

To load a model directly from the [Hugging Face Model Hub](https://huggingface.co/), simply pass its model ID (i.e., repository name) to the LLM constructor. The model will be automatically downloaded:

```python
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

You can also use [quantized checkpoints](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4) (FP4, FP8, etc) of popular models provided by NVIDIA in the same way.

### 2. Using a Local Hugging Face Model

To use a model from local storage, first download it manually:

```console
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
```

Then, load the model by specifying a local directory path:

```python
llm = LLM(model=<local_path_to_model>)
```

> **Note:** Some models require accepting specific [license agreements](https://ai.meta.com/resources/models-and-libraries/llama-downloads/). Make sure you have agreed to the terms and authenticated with Hugging Face before downloading.

## Startup Metrics

For the PyTorch backend, the beta `LLM.startup_metrics` property reports executor construction,
model-engine warmup, and weight-loading timings from worker rank 0. Values are wall-clock seconds.
The property returns an empty dictionary when the backend does not provide startup metrics.

```python
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(llm.startup_metrics)
```

A typical result has the following structure (values are illustrative):

```json
{
  "initial_model_engine": {
    "sampling_warmup_seconds": 0.342,
    "attention_warmup_seconds": 2.911,
    "general_warmup_seconds": 0.387,
    "autotuner_warmup_seconds": 0.041,
    "mamba_hybrid_warmup_seconds": 0.000,
    "ctx_cuda_graph_warmup_seconds": 0.480,
    "ctx_cuda_graph_capture_seconds": 0.320,
    "post_ctx_cuda_graph_capture_warmup_seconds": 0.110,
    "gen_cuda_graph_warmup_seconds": 0.566,
    "gen_cuda_graph_capture_seconds": 0.950,
    "dg_paged_mqa_warmup_seconds": 0.000,
    "cute_dsl_radix_topk_warmup_seconds": 0.000,
    "memory_pool_prepopulation_seconds": 0.094,
    "kv_cache_cleanup_seconds": 0.194,
    "total_warmup_seconds": 5.485
  },
  "final_model_engine": {
    "sampling_warmup_seconds": 0.291,
    "attention_warmup_seconds": 0.057,
    "general_warmup_seconds": 0.492,
    "autotuner_warmup_seconds": 0.046,
    "mamba_hybrid_warmup_seconds": 0.000,
    "gen_cuda_graph_warmup_seconds": 0.599,
    "gen_cuda_graph_capture_seconds": 0.991,
    "dg_paged_mqa_warmup_seconds": 0.000,
    "cute_dsl_radix_topk_warmup_seconds": 0.000,
    "memory_pool_prepopulation_seconds": 0.097,
    "kv_cache_cleanup_seconds": 1.657,
    "total_warmup_seconds": 4.230
  },
  "py_executor": {
    "config_and_checkpoint_loader_initialization_seconds": 0.004,
    "model_engine_creation_seconds": 1.554,
    "sampler_creation_seconds": 1.195,
    "initial_kv_cache_creation_seconds": 0.025,
    "speculative_decoding_resource_manager_creation_seconds": 0.000,
    "speculative_drafter_creation_seconds": 0.000,
    "initial_py_executor_creation_seconds_for_kv_cache_estimation": 5.766,
    "kv_cache_capacity_configuration_seconds": 0.075,
    "final_kv_cache_creation_seconds": 0.057,
    "final_py_executor_creation_seconds": 4.459,
    "worker_start_seconds": 0.000,
    "total_py_executor_creation_seconds": 13.373
  },
  "model_loader": {
    "checkpoint_preparation_seconds": 0.719,
    "weight_population_seconds": 0.272,
    "checkpoint_finalization_seconds": 0.031,
    "post_load_processing_seconds": 0.004,
    "total_model_loading_seconds": 1.545
  }
}
```

Some of the timing metrics may not appear or may show as a very small value if the corresponding codepath is not executed, e.g.
if the prefill CUDA graphs are not enabled, its related metrics "...ctx_cuda_graph..." will not appear.

The `py_executor` property contains the timed scopes in `create_py_executor()`, which creates the [PyExecutor](../developer-guide/overview.md).

| PyExecutor metric | Scope |
|-----------------|-------|
| `config_and_checkpoint_loader_initialization_seconds` | Initialize the model config and checkpoint loader. |
| `model_engine_creation_seconds` | Construct the main model engine, which includes loading model weights. |
| `guided_decoder_creation_seconds` | Construct guided-decoding resources. |
| `sampler_creation_seconds` | Construct the sampler. |
| `initial_kv_cache_creation_seconds` | Construct the temporary KV cache used to estimate final capacity. |
| `speculative_decoding_resource_manager_creation_seconds` | Construct the speculative-decoding resource manager. |
| `speculative_drafter_creation_seconds` | Construct the speculative drafter. |
| `initial_py_executor_creation_seconds_for_kv_cache_estimation` | Construct the temporary PyExecutor for KV cache capacity estimation, which includes the initial model engine warmup. |
| `kv_cache_capacity_configuration_seconds` | Determine final KV cache capacity from the temporary executor and available memory. |
| `final_kv_cache_creation_seconds` | Construct the final KV cache retained for serving. |
| `final_py_executor_creation_seconds` | Using the final KV cache, construct the final PyExecutor retained for serving, which includes the final model engine warmup. |
| `worker_start_seconds` | Start the final PyExecutor worker. |
| `total_py_executor_creation_seconds` | Total time for the `create_py_executor()` call, including all applicable scopes above. |

The two model-engine properties contain times for various `ModelEngine` warmup stages.
`initial_model_engine` measures the timings for the initial model engine creation as part of the
`py_executor.initial_py_executor_creation_seconds_for_kv_cache_estimation` timing.
Likewise, `final_model_engine` measures the timings for the final model engine creation as part of
the `py_executor.final_py_executor_creation_seconds` timing.

Each model-engine property contains the warmup scopes below. Which scopes appear depends on the model
and configuration; for example, encoder-decoder and context-parallel configurations can skip some
stages.

| Model-engine metric | Scope |
|---------------------|-------|
| `sampling_warmup_seconds` | Warm up sampling kernels and run `torch.compile()` on sampling operators. |
| `attention_warmup_seconds` | Warm up the attention backend and kernels. |
| `general_warmup_seconds` | Prepare and warm up general input shapes and release temporary workspaces. |
| `autotuner_warmup_seconds` | Run kernel autotuning warmup. |
| `mamba_hybrid_warmup_seconds` | Warm up Mamba hybrid kernels, when applicable. |
| `gen_cuda_graph_warmup_seconds` | Run the warmup-only pass for generation and mixed encoder-decoder CUDA graphs, including LoRA autotuning and pipeline-parallel cache synchronization and cleanup. |
| `gen_cuda_graph_capture_seconds` | Capture generation and mixed encoder-decoder CUDA graphs for serving. |
| `ctx_cuda_graph_warmup_seconds` | Warm up prefill forwards before CUDA graph capture. |
| `ctx_cuda_graph_capture_seconds` | Capture prefill CUDA graphs. |
| `post_ctx_cuda_graph_capture_warmup_seconds` | Warm up prefill logits buffers with many requests after CUDA graph capture. |
| `dg_paged_mqa_warmup_seconds` | Warm up DeepGEMM paged-MQA metadata, when applicable. |
| `cute_dsl_radix_topk_warmup_seconds` | Warm up the CuTe DSL radix top-k kernel, when applicable. |
| `memory_pool_prepopulation_seconds` | Prepare maximum-shape requests and pre-populate the memory pool. |
| `kv_cache_cleanup_seconds` | Check and clear invalid KV cache values produced during warmup. |
| `total_warmup_seconds` | Complete model-engine warmup, including KV cache cleanup. |

The `model_loader` property contains timings for loading the main LLM weights, including
`checkpoint_finalization_seconds` for checkpoint-session synchronization and cleanup after weight population.
If a draft model is used, additional fields `draft_checkpoint_preparation_seconds`,
`draft_weight_population_seconds`, and `draft_checkpoint_finalization_seconds` will appear.
A `draft_model_loader` property appears when a user-provided drafter exposes a `draft_model_engine`.
The `py_executor.model_engine_creation_seconds` timing includes the total `model_loader` timing.

| Metric | Description |
|--------|-------------|
| `checkpoint_preparation_seconds` | Time spent warming up, parsing and preparing checkpoint tensors for the model. Some checkpoint formats can populate model storage directly during this phase. |
| `weight_population_seconds` | Time spent copying prepared checkpoint tensors into model parameters on GPUs. This metric can be absent for formats that populate weights directly during the above checkpoint preparation phase. |
| `checkpoint_finalization_seconds` | Time spent finalizing the checkpoint session after weight population. This includes loader-specific synchronization and cleanup. |
| `draft_checkpoint_preparation_seconds` | Checkpoint preparation time for draft weights loaded as part of the model loader. |
| `draft_weight_population_seconds` | Weight population time for draft weights loaded as part of the model loader. |
| `draft_checkpoint_finalization_seconds` | Checkpoint finalization time for draft weights loaded as part of the model loader. |
| `post_load_processing_seconds` | Time spent in format-specific hooks and model finalization, including post-load weight transformation, quantization and memory cleanup. |
| `total_model_loading_seconds` | Overall model construction and loading interval measured after checkpoint configuration validation. It includes the named phases below. |

`trtllm-serve` exposes the same rank-0 payload in the `startup_metrics` field of the
`GET /server_info` response:

```console
curl http://localhost:8000/server_info
```

```json
{
  "startup_metrics": {
    "model_loader": {
      "total_model_loading_seconds": 1.545,
      ...
    }
  }
}
```

## Unified Cache for Compiled Artifacts

Set `TRTLLM_CACHE_DIR` before importing TensorRT-LLM to place its compiled
artifact caches under a single root. TensorRT-LLM supplies the following
defaults. Except for the isolation behavior described below, an individually
configured environment variable takes precedence.

| Environment variable | Default cache path | Default under `TRTLLM_CACHE_DIR` |
|----------------------|--------------------|------------------------------------|
| `TLLM_AUTOTUNER_CACHE_PATH` | Disabled | `autotuner/cache.json` |
| `TORCHINDUCTOR_CACHE_DIR` | `/tmp/torchinductor_$USER` | `inductor` |
| `TRITON_CACHE_DIR` | `$HOME/.triton/cache` | `triton` |
| `TORCH_EXTENSIONS_DIR` | `$HOME/.cache/torch_extensions` | `torch_extensions` |
| `FLASHINFER_WORKSPACE_BASE` | `$HOME/.cache/flashinfer` | `flashinfer` |
| `CUTE_DSL_CACHE_DIR` | `/tmp/$USER/cutlass_python_cache` | `cute_dsl` |
| `DG_JIT_CACHE_DIR` | `$HOME/.deep_gemm` | `deep_gemm` |
| `TRTLLM_DG_CACHE_DIR` | Disabled | `trtllm_deep_gemm` |
| `CUDA_CACHE_PATH` | `$HOME/.nv/ComputeCache` | `cuda` |

**Important note**: while some of the caches controlled by `TRTLLM_CACHE_DIR`
are enabled and persisted by default (e.g., `FLASHINFER_WORKSPACE_BASE`),
others are disabled or with a path defaulting to `/tmp` (e.g., `CUTE_DSL_CACHE_DIR`
or `TLLM_AUTOTUNER_CACHE_PATH`). These caches are not validated for sharing
across model instances or long-term persistence, which may lead to errors 
and other unintended behavior. As such, we recommend using distinct 
`TRTLLM_CACHE_DIR` caches for different models, and invalidating them anytime 
a component in the software stack changes.

The directories are created by their respective consumers when needed.
TensorRT-LLM does not add cache locking or cleanup; per-rank isolation is
enabled only for the exceptions described below. The
configured values are forwarded to dynamically spawned MPI workers.

Configuring `TRTLLM_DG_CACHE_DIR`, either directly or through
`TRTLLM_CACHE_DIR`, also defaults `TRTLLM_DG_JIT_DUMP_CUBIN` to `1` so its
NVRTC-generated cubins are persisted. An explicitly configured dump setting
still takes precedence. See the
[DeepGEMM environment variables](source:examples/models/core/deepseek_v3/README.md#deepgemm)
for details.

Note that, when `TRTLLM_CACHE_DIR` is set, the FlashInfer workspace isolation
mechanism described below still applies. This will prevent race conditions
during cache read/writes, at the cost of limiting cache sharing. To disable
workspace isolation, either set `TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS=0`,
or set `FLASHINFER_WORKSPACE_BASE` to a different path from the one assigned
under `TRTLLM_CACHE_DIR`.

Ray workers treat `DG_JIT_CACHE_DIR` equal to
`<TRTLLM_CACHE_DIR>/deep_gemm` as unified-cache-managed and isolate it by rank
and GPU, even when that value was set explicitly. Other explicit values are
preserved. Set `TRTLLM_DEEP_GEMM_CACHE_PER_PROCESS=0` to preserve and share the
unified path.

### Launcher process ownership

The rank-0 `trtllm-llmapi-launch` shell starts a small Python guard for the MPI
communication server, task, and shutdown helper. Each guard owns a separate
process group and monitors the shell, so killing only the shell with `SIGKILL`
under plain `mpirun` still triggers cleanup of those groups. Startup registration
finishes before a command can run. Guards also clean remaining group members
when the command exits, then return its original exit status.

Cleanup sends `SIGTERM`, followed by `SIGKILL` after
`TLLM_LLMAPI_LAUNCH_TERM_GRACE_SECONDS` (default 5 seconds). This covers descendants
that remain in their inherited process group. A command that deliberately
creates a separate session or process group must manage those processes itself;
killing a guard independently also requires the job's scheduler/container to
reclaim its workload.

### Remote MPI shutdown timeouts

Configure these settings together when using `trtllm-llmapi-launch`:

| Setting | Default | Scope |
| --- | --- | --- |
| `TLLM_MGMN_SHUTDOWN_GRACE_SECONDS` | 60 seconds | Server budget for draining a failed batch or stopping the MPI world. Final session and shared-executor teardown use the remaining budget. |
| `TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT` | 120 seconds | Launcher wait for the stop helper and server after the task exits. Also used separately for guard registration and for the task to exit naturally after a server failure. |
| `TLLM_LLMAPI_LAUNCH_TERM_GRACE_SECONDS` | 5 seconds | Additional cleanup grace between TERM and KILL for owned process groups. |

The server shutdown grace starts when it receives the stop request; an earlier
worker-failure deadline can shorten the remaining grace. The launcher also waits
for the stop helper's Python startup and `pyzmq` import before that request
can be sent. If the standalone helper is unavailable, the fallback imports
`tensorrt_llm` instead. Set the launcher stop timeout above the server grace by enough to
cover measured cold startup/import time, connection/send and queue cleanup, and
scheduling margin. The defaults leave 60 seconds of allowance; raising only the
server grace does not raise the launcher timeout.

For example, a 120-second server grace with a 180-second launcher timeout leaves
60 seconds for this overhead. This is an illustrative allowance: increase it if
cold or loaded nodes need more time.

If task-first shutdown exhausts the launcher budget, the launcher terminates
owned groups and returns 124 when the task succeeded; an existing nonzero task
status takes precedence. After a server-first failure, the launcher instead
preserves a naturally completed task's nonzero status, otherwise the server's
failure status. TERM-to-KILL cleanup is a separate phase. Launcher waits count
100 ms sleeps, so polling and scheduling overhead add time; these settings are
not a strict end-to-end wall-clock limit.

The launcher returns exit code 2 for invalid
`TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT` or
`TLLM_LLMAPI_LAUNCH_TERM_GRACE_SECONDS` values; both require a positive integer
of at most six digits. A process-guard registration timeout also returns 124.
If registration of the stop helper fails after the task has exited, an existing
nonzero task status still takes precedence.

### Remote MPI batch ordering

The remote MPI server dispatches the next batch only after every future in the
previous batch has completed. Each batch runs exactly once on every rank, but
the task wrapper has no trailing barrier: an individual rank can finish or report
an error while its peers are still running. Cross-batch ordering is enforced by
the server; directly invoking the wrapper does not provide a collective
completion barrier.

### Disaggregated MPI leader failures

The disaggregated MPI leader uses the same remote MPI server. An asynchronous
worker failure is fatal to that server: it reports the worker error, stops
accepting new work, and keeps the error channel available until a stop request
or the original failure deadline. Final teardown can abort the MPI communicator
if its remaining grace expires. This does not restart the failed service.

The leader launches its proxy through an autonomous process guard. Normal leader
cleanup signals the guard; if the leader exits through `MPI_Abort` or is killed,
the guard detects parent death and cleans the proxy's inherited process group.
It allows 5 seconds between `SIGTERM` and `SIGKILL`. Descendants that create their
own session or process group still require separate ownership, as described
above. The Bash launcher's `TLLM_LLMAPI_LAUNCH_*` settings do not configure this
Python launcher.

### FlashInfer JIT workspaces for MPI workers

`trtllm-llmapi-launch` ranks and dynamically spawned `MpiPoolSession` workers
isolate FlashInfer JIT workspaces by default. Each process claims a locked,
persistent cache slot under `~/.cache/tensorrt_llm/flashinfer` that holds its
generated sources, compiled modules and downloaded FlashInfer artifacts, so
concurrent MPI processes never write to each other's compiler inputs while JIT
artifacts stay warm across launches. Set
`TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS=0` before invoking the launcher or
creating the LLM instance to disable this behavior.

Persistent slots are not pruned automatically, so their count can grow with
peak job concurrency. When no TensorRT-LLM processes are using the cache, the
slots may be deleted safely; subsequent launches rebuild the removed JIT
artifacts.

If persistent workspace setup is unavailable, each process falls back to a
process-unique temporary workspace that is removed, artifacts included, when
the process exits.

An explicitly configured `FLASHINFER_WORKSPACE_BASE` takes precedence in both
launch modes, while setting `TRTLLM_CACHE_DIR` does not. An explicitly configured `FLASHINFER_CUBIN_DIR` is propagated
unchanged to every rank and is the way to share downloaded artifacts between
ranks; populate it before the multi-rank launch, either by installing the
`flashinfer-cubin` package matching `flashinfer-python` or with a single-rank
warm-up run, and set `FLASHINFER_NO_DOWNLOAD=1` so a missing artifact fails
instead of being downloaded concurrently into the shared directory.

## Tips and Troubleshooting

The following tips typically assist new LLM API users who are familiar with other APIs that are part of TensorRT-LLM:

### RuntimeError: only rank 0 can start multi-node session, got 1

  There is no need to add an `mpirun` prefix for launching single node multi-GPU inference with the LLM API.

  For example, you can run `python llm_inference_distributed.py` to perform multi-GPU on a single node.

### Hang issue on Slurm Node

  If you experience a hang or other issue on a node managed with Slurm, add prefix `mpirun -n 1 --oversubscribe --allow-run-as-root` to your launch script.

  For example, try `mpirun -n 1 --oversubscribe --allow-run-as-root python llm_inference_distributed.py`.

### MPI_ABORT was invoked on rank 1 in communicator MPI_COMM_WORLD with errorcode 1.

  Because the LLM API relies on the `mpi4py` library, put the LLM class in a function and protect the main entrypoint to the program under the `__main__` namespace to avoid a [recursive spawn](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor) process in `mpi4py`.

  This limitation is applicable for multi-GPU inference only.

### Cannot quit after generation

  The LLM instance manages threads and processes, which may prevent its reference count from reaching zero. To address this issue, there are two common solutions:
  1. Wrap the LLM instance in a function, as demonstrated in the quickstart guide. This will reduce the reference count and trigger the shutdown process.
  2. Use LLM as a context manager, with the following code: `with LLM(...) as llm: ...`, the shutdown method will be invoked automatically once it goes out of the `with`-statement block.

### Remote MPI shutdown

After the task exits, `trtllm-llmapi-launch` sends the authenticated stop request
with a standalone helper. The helper imports `pyzmq`, but does not import
TensorRT-LLM or initialize MPI. It waits for an authenticated acknowledgment and
retries on a fresh connection if the server has not accepted the request. Each
of up to 300 attempts waits one second for the acknowledgment; socket close
discards unsent messages without waiting. The acknowledgment confirms receipt,
not completion of worker shutdown. The launcher supervises the helper and server
using the shutdown budgets described above, including helper startup. Its
120-second default stop timeout can end the helper before all 300 attempts run.

The launcher resolves symlinks before locating the helper beside the launcher
file. If resolution fails or the helper is missing or unreadable, it logs a
warning and uses
`python3 -m tensorrt_llm.llmapi.mgmn_leader_node --action stop`. This fallback
imports TensorRT-LLM and can take longer to start. A failure after the helper
starts is reported directly and does not trigger the fallback.

After a task-first exit, a nonzero task status takes precedence over stop or
server failures. The helper returns `1` if its attempts finish without an ACK;
the launcher reports that failure and cleans up its owned processes. It skips
the stop helper if the server has already exited. Server-first failures and
shutdown timeouts retain the supervisor behavior described above. The module
stop command uses the same acknowledged protocol and unacknowledged exit code.

### Single node hanging when using `docker run --net=host`

The root cause may be related to `mpi4py`. There is a [workaround](https://github.com/mpi4py/mpi4py/discussions/491#discussioncomment-12660609) suggesting a change from `--net=host` to `--ipc=host`, or setting the following environment variables:

```bash
export OMPI_MCA_btl_tcp_if_include=lo
export OMPI_MCA_oob_tcp_if_include=lo
```

Another option to improve compatibility with `mpi4py` is to launch the task using:

```bash
mpirun -n 1 --oversubscribe --allow-run-as-root python my_llm_task.py
```

This command can help avoid related runtime issues.
