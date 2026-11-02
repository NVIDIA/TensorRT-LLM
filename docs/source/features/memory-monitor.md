<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPU Memory Monitor (Prototype)

When a run fails with an out-of-memory error, the error says how much could
not be allocated, not where the rest of the GPU went. The memory monitor
answers that question. It reports, for each worker process, how the GPU memory
the process holds splits by purpose -- KV cache, model weights, runtime
buffers, CUDA graphs, communication buffers, library overhead -- with the
parts adding up exactly to what the process holds, and how each part has
changed since warmup.

The report is meant to be read without knowing which allocator holds what: the
summary at the top names memory by what it is for. A section below it breaks
the same memory down by allocator for whoever needs to dig further.

```{warning}
The memory monitor is a prototype and its report format may change.
```

## Turning it on

The monitor is off by default and costs nothing while off. Set
`TLLM_MEM_MONITOR=1` for the run:

```bash
TLLM_MEM_MONITOR=1 trtllm-serve <model> ...
```

With the monitor on, each worker process reports:

* once startup has finished, including warmup and CUDA graph capture;
* when an allocation runs out of memory, whether it fails in PyTorch or in
  native code such as NCCL, cuBLAS or TensorRT-LLM's own C++ pools, and
  whether it happens on the executor thread or a background thread;
* every `TLLM_MEM_MONITOR_INTERVAL` seconds, if set.

Reports go to the log at INFO (startup, periodic) or WARNING (out of memory)
level.

| Variable | Default | Effect |
|---|---|---|
| `TLLM_MEM_MONITOR` | unset | `1` turns the monitor on. |
| `TLLM_MEM_MONITOR_RANKS` | all ranks | Comma-separated ranks that report, e.g. `0` or `0,4`. |
| `TLLM_MEM_MONITOR_OOM_INTERVAL` | `60` | Minimum seconds between two out-of-memory reports; a retry loop that fails repeatedly produces one report per interval. |
| `TLLM_MEM_MONITOR_INTERVAL` | unset | Report every this many seconds while serving, to watch memory that grows slowly. |
| `TLLM_MEM_MONITOR_DIR` | unset | Also write every report to its own file in this directory, named by rank, process and time. Logs from several ranks interleave and are often cut short when a job dies; the files survive. |
| `TLLM_MEM_MONITOR_STACKS` | unset | `1` records the allocating Python stack of every live PyTorch block, so that the "PyTorch, unattributed" row is broken down by the code that allocated it. Slows PyTorch allocations; turn it on to reproduce a problem, not in production. |

## Asking for a report

`LLM.get_memory_report()` (also on `AsyncLLM`) returns one report per rank,
rank 0 first, whether or not `TLLM_MEM_MONITOR` is set. Under the default MPI
executor the other ranks answer only while the monitor is on; with it off,
only rank 0 reports.

```python
from tensorrt_llm import LLM

with LLM(model="Qwen/Qwen2-0.5B-Instruct") as llm:
    llm.generate(["Hello"])
    for report in llm.get_memory_report():
        print(report)
```

Without `TLLM_MEM_MONITOR=1` for the run, the report has no startup-step
breakdown and no comparison with warmup, but every row is still there.

A `trtllm-serve` started with `TLLM_MEM_MONITOR=1` answers the same request
at `GET /memory_report`, with `{"reports": [...]}`. The route exists only
while the monitor is on.

## Reading a report

An out-of-memory report from a run whose logits processor kept allocating:

```text
=== TensorRT-LLM GPU memory monitor: out_of_memory ===
  pid=856 rank=0 device=0
  OUT OF MEMORY: PyTorch could not allocate 150.56 GiB on device 0 (the caller may still recover).
  This process uses 27.79 GiB of 178.34 GiB (other processes 0.72 GiB, free 149.83 GiB).

                                      size   share   since warmup
  KV cache                       17.56 GiB   63.2%      +0.00 GiB   allocated up front; free blocks stay resident
  PyTorch, unattributed           8.00 GiB   28.8%      +8.00 GiB   in use by PyTorch, owner not identified; typically activations
  CUDA and library overhead       1.07 GiB    3.8%      +0.00 GiB   CUDA context, cuBLAS, NCCL internals and other driver memory
  Model weights                   0.92 GiB    3.3%      +0.00 GiB   parameters and buffers of the loaded model(s)
  Runtime buffers                 0.14 GiB    0.5%      +0.00 GiB   buffers owned by named runtime components
  CUDA graphs                     0.05 GiB    0.2%      +0.00 GiB   held by captured CUDA graphs for replay; not reusable
  PyTorch cache (free)            0.05 GiB    0.2%      -0.26 GiB   held by the PyTorch allocator but not in use; reusable

  Startup plan: peak outside the KV cache 3.02 GiB + KV cache 17.56 GiB = 20.58 GiB; device now 28.47 GiB, ...
  Usage exceeds the startup plan by 7.89 GiB: the workload needs more than profiling measured, or another process grew.

  Conclusion: since warmup, PyTorch, unattributed grew the most (+8.00 GiB).
```

* **The total** is this process's own GPU memory, as the driver reports it.
  Memory held by other processes on the same GPU is shown separately; when
  they hold more than this process, the conclusion says so.
* **The rows** add up exactly to the total. Large rows list what they are made
  of (`of which`).
  * *Model weights* follows the loaded model, including weights that were
    replaced or moved after loading.
  * *Runtime buffers* are named by the startup step that allocated them.
  * *Communication buffers* name NCCL communicators, NCCL window buffers,
    multicast and symmetric-memory workspaces.
  * *CUDA and library overhead* is what no component accounts for: the CUDA
    context, cuBLAS, NCCL internals and code loaded on demand. It is split by
    when it appeared -- before startup, in each startup step, or since.
* **since warmup** compares with the report taken when startup finished. The
  row that grew is usually the one to look at; growth in *PyTorch,
  unattributed* while serving points at activations or at memory something
  keeps allocating, which `TLLM_MEM_MONITOR_STACKS=1` attributes to code.
* **Startup plan** compares the memory in use with what KV cache sizing
  planned for when the engine started.
* **Suggestion** appears for an out-of-memory error or a failed startup and
  names the setting that controls the largest consumer.

While parts of the runtime are released with `LLM.release()`, the rows show
only memory that is resident, and one line lists what is released and returns
on resume.

## Cost

With the monitor off, nothing runs while serving. With it on, allocation
history is recorded during startup only, and a report takes about a
millisecond once startup has finished; no work is added to the executor loop.
`TLLM_MEM_MONITOR_STACKS=1` is the exception and slows every PyTorch
allocation while it is on.
