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

For the PyTorch backend, the beta `LLM.startup_metrics` property reports weight-loading timings from
worker rank 0. Values are wall-clock seconds. The property returns an empty dictionary when the
backend does not provide startup metrics.

```python
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(llm.startup_metrics)
```

A typical result has the following structure:

```json
{
  "model_loader": {
    "total_model_loading_seconds":  1.971,
    "checkpoint_preparation_seconds": 1.177,
    "weight_population_seconds": 0.598,
    "checkpoint_finalization_seconds": 0.031,
    "post_load_processing_seconds": 0.005
  }
}
```

The `model_loader` object contains timings for the main LLM weights. If a draft
model is used, the additional fields `draft_checkpoint_preparation_seconds`,
`draft_weight_population_seconds`, and `draft_checkpoint_finalization_seconds`
appear.

| Metric | Description |
|--------|-------------|
| `total_model_loading_seconds` | Overall model construction and loading interval measured after checkpoint configuration validation. It includes the named phases below. |
| `checkpoint_preparation_seconds` | Time spent warming up, parsing and preparing checkpoint tensors for the model. Some checkpoint formats can populate model storage directly during this phase. |
| `weight_population_seconds` | Time spent copying prepared checkpoint tensors into model parameters on GPUs. This metric can be absent for formats that populate weights directly during the above checkpoint preparation phase. |
| `checkpoint_finalization_seconds` | Time spent finalizing the checkpoint session after weight population. This includes loader-specific synchronization and cleanup; rank-striped read-ahead includes waiting for peer ranks and stopping background readers. |
| `draft_checkpoint_preparation_seconds` | Checkpoint preparation time for draft weights loaded as part of the model loader. |
| `draft_weight_population_seconds` | Weight population time for draft weights loaded as part of the model loader. |
| `draft_checkpoint_finalization_seconds` | Checkpoint finalization time for draft weights loaded as part of the model loader. |
| `post_load_processing_seconds` | Time spent in format-specific hooks and model finalization, including post-load weight transformation, quantization and memory cleanup. |

`trtllm-serve` exposes the same rank-0 payload in the `startup_metrics` field of the
`GET /server_info` response:

```console
curl http://localhost:8000/server_info
```

```json
{
  "startup_metrics": {
    "model_loader": {
      "total_model_loading_seconds": 1.971,
      ...
    }
  }
}
```


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
for the stop helper's Python startup and `tensorrt_llm` imports before that request
can be sent. Set the launcher stop timeout above the server grace by enough to
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
previous batch has completed. The task wrapper retains its entry barrier to
place one task on each rank, but has no trailing barrier: an individual rank can
finish or report an error while its peers are still running. Cross-batch ordering
is enforced by the server; directly invoking the wrapper does not provide a
collective completion barrier.

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
launch modes. An explicitly configured `FLASHINFER_CUBIN_DIR` is propagated
unchanged to every rank and is the way to share downloaded artifacts between
ranks; populate it before the multi-rank launch, either by installing the
`flashinfer-cubin` package matching `flashinfer-python` or with a single-rank
warm-up run, and set `FLASHINFER_NO_DOWNLOAD=1` so a missing artifact fails
instead of being downloaded concurrently into the shared directory.

### Cannot quit after generation

  The LLM instance manages threads and processes, which may prevent its reference count from reaching zero. To address this issue, there are two common solutions:
  1. Wrap the LLM instance in a function, as demonstrated in the quickstart guide. This will reduce the reference count and trigger the shutdown process.
  2. Use LLM as a context manager, with the following code: `with LLM(...) as llm: ...`, the shutdown method will be invoked automatically once it goes out of the `with`-statement block.

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
