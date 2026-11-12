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

## Modes

`memory_monitor_config.mode` in the LLM arguments, or the
`TLLM_MEM_MONITOR` environment variable, selects one of three modes:

| Value | Mode | What it does |
|---|---|---|
| unset or `oom` | out-of-memory reports (default) | A report when an allocation fails once the executor has started, and nothing else. |
| `1` (or `full`) | full | Also records startup and reports when it finishes, compares later reports with that point, and enables periodic reports and, with a runtime-control key, `GET /memory_report`. |
| `0` (or `off`) | off | No reports are produced on their own. |

```yaml
# trtllm-serve <model> --config config.yaml
memory_monitor_config:
  mode: full
  report_dir: /home/alice/memory-reports  # a directory of your own
```

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import MemoryMonitorConfig

llm = LLM(model=..., memory_monitor_config=MemoryMonitorConfig(mode="full"))
```

Each field of `memory_monitor_config` has an environment variable, listed
below. A variable set to a non-empty value overrides the field, so a run can
be diagnosed without changing its configuration, and logs one warning when
the two differ; an empty variable (`docker -e VAR=`, `export VAR=${VAR:-}`)
counts as unset:

```bash
TLLM_MEM_MONITOR=1 trtllm-serve <model> ...
```

Out-of-memory errors are reported whether they fail in PyTorch or in native
code such as NCCL, cuBLAS or TensorRT-LLM's own C++ pools, and whether they
happen on the executor thread or a background thread, as long as the error
reaches Python: raised as an exception, or logged at ERROR level through
TensorRT-LLM's Python logger. A failure that native code only logs itself is
not reported. Errors are reported only once the executor has been created;
see [Limitations](#limitations).

In the default mode a report has every row, but no breakdown by startup step
and no comparison with warmup: both need startup to be recorded, which only
the full mode does. With `TLLM_MEM_MONITOR=1` each worker process also
reports:

* once startup has finished, including warmup and CUDA graph capture;
* every `TLLM_MEM_MONITOR_INTERVAL` seconds, if set.

Out-of-memory reports go to the log at ERROR level, so they are visible at
TensorRT-LLM's default log level. Startup and periodic reports go at INFO
level, and a failed startup at WARNING: set `TLLM_LOG_LEVEL=info` to see
them, on every rank -- the LLM API raises the level for rank 0 only.

| `memory_monitor_config` field | Config values | Environment variable | Environment values |
|---|---|---|---|
| `mode` | `off`, `oom` (default), `full`; YAML's unquoted `off`/`on` and `0`/`1` are read as `off`/`full` | `TLLM_MEM_MONITOR` | `0`, `off`, `false`, `no`; `oom`; `1`, `full`, `on`, `true`, `yes`. An unrecognised value is reported once and the default is used. |
| `ranks` | list of ranks that report, each below the job's world size; default all ranks | `TLLM_MEM_MONITOR_RANKS` | ranks and ranges, e.g. `0,4` or `0-3,8`, from 0 to 65535; a filter that selects none of the job's ranks is reported and ignored |
| `oom_report_interval` | minimum seconds between two out-of-memory reports, 0 to 604800, default 60; a retry loop that fails repeatedly produces one report per interval | `TLLM_MEM_MONITOR_OOM_INTERVAL` | seconds, 0 to 604800 |
| `report_dir` | directory that also receives each startup, periodic and out-of-memory report as its own file; see [Report files](#report-files) | `TLLM_MEM_MONITOR_DIR` | directory path |
| `periodic_report_interval` | full mode only: report every this many seconds while serving, to watch memory that grows slowly; 1 to 604800 | `TLLM_MEM_MONITOR_INTERVAL` | seconds, 0 to 604800 (below 1 uses 1) |
| `record_stacks` | full mode only: `true` records the allocating Python stack of every live PyTorch block, so that the "PyTorch, unattributed" row is broken down by the code that allocated it. Needs PyTorch's native allocator. Slows PyTorch allocations; turn it on to reproduce a problem, not in production. Recording starts once startup has completed and stops when that executor shuts down. | `TLLM_MEM_MONITOR_STACKS` | `1`/`0` |

`periodic_report_interval` and `record_stacks` act only in full mode; setting
them in another mode, or `record_stacks` on an allocator that drops allocation
metadata, logs a warning.

The settings are process-wide. When several executors run in one process,
the first executor's configuration stays in force while it lives; a later
executor's differing configuration is reported once and not applied.

### Report files

With `report_dir` set, the directory is created readable by its owner only
(mode 0700) and each report file likewise (0600). Each report goes to its own
file, named by rank, process, time and tag; only the newest 100 periodic
reports of each rank and process are kept, older ones are deleted. Logs from
several ranks interleave and are often cut short when a job dies; the files
survive. A directory that cannot be written is reported once, and reports are
then only logged. Use a directory of your own, for example
`TLLM_MEM_MONITOR_DIR=$HOME/memory-reports`.

### Hooks installed in your process

In every mode except `off`, arming the monitor installs three hooks:

* a `threading.excepthook` that reports a thread dying of an out-of-memory
  error and then calls the previously installed hook;
* a filter on TensorRT-LLM's Python logger (`tensorrt_llm`) that passes ERROR
  messages to the monitor and filters nothing out. A report for a logged
  error is built on its own short-lived thread, so the thread that logged it
  continues at once;
* a torch out-of-memory observer
  (`torch._C._cuda_attach_out_of_memory_observer`) that records the failed
  allocation.

The first two are removed when the last executor shuts down; torch cannot
detach an observer, so it stays attached and is made inert.

## Asking for a report

`LLM.get_memory_report()` (also on `AsyncLLM`) returns one entry per rank,
rank 0 first, in every mode. Under the default MPI executor the other ranks
send their reports only in full mode; otherwise their entries say that they
were not gathered.

```python
from tensorrt_llm import LLM

with LLM(model="Qwen/Qwen2-0.5B-Instruct") as llm:
    llm.generate(["Hello"])
    for report in llm.get_memory_report():
        print(report)
```

Outside full mode, the report has no startup-step
breakdown and no comparison with warmup, but every row is still there.

A `trtllm-serve` in full mode with `TRTLLM_RUNTIME_CONTROL_API_KEY` set
also answers at `GET /memory_report`, with `{"reports": [...]}`. Sleep mode is
not needed:

```bash
export TRTLLM_RUNTIME_CONTROL_API_KEY='replace-with-a-long-random-secret'
TLLM_MEM_MONITOR=1 trtllm-serve <model>
```

Requests must be signed with that key like the runtime-control routes; see
[Control-plane HTTP endpoints](sleep-wakeup.md#control-plane-http-endpoints)
for the headers and the signature. A report shows process IDs, other
processes' GPU usage and, with `TLLM_MEM_MONITOR_STACKS=1`, source paths,
and each request makes every rank build a report. Without the key, or with
more than one serving frontend (`num_serve_frontends`), the route is not
served and the server logs why.

## Reading a report

An out-of-memory report with `TLLM_MEM_MONITOR=1`, from rank 1 of a
Llama-3.1-8B-Instruct run at TP=2, where a logits processor kept 4 GiB per
step:

```text
=== TensorRT-LLM GPU memory monitor: out_of_memory ===
  pid=858 rank=1 device=1
  OUT OF MEMORY: PyTorch could not allocate 73.41 GiB on device 1 (device free 72.13 GiB; PyTorch may use up to 91.93 GiB; PyTorch held 19.74 GiB live, 19.77 GiB reserved). Error: [TRT-LLM] [E] [_torch][RANK 1] Encountered an error in forward function: CUDA out of memory. Tried to allocate 73.41 GiB. GPU 1 has a total capacity of 178.39 GiB of which 72.13 GiB is free. Including
  This process uses 106.08 GiB of 178.39 GiB (other processes 0.18 GiB, free 72.13 GiB).

                                      size   share   since warmup
  KV cache                       83.03 GiB   78.3%      +0.00 GiB   allocated up front; free blocks stay resident
  PyTorch, unattributed          12.00 GiB   11.3%     +12.00 GiB   in use by PyTorch, owner not identified; typically activations
  Model weights                   7.48 GiB    7.1%      +0.00 GiB   parameters and buffers of the loaded model(s)
  Communication buffers           2.01 GiB    1.9%      +0.00 GiB   multi-GPU communication buffers
                                of which (GiB): NCCL communicators 0.94, NCCLWindowAllocator 0.56, custom allreduce workspace 0.50
  CUDA and library overhead       1.28 GiB    1.2%      +0.06 GiB   CUDA context, cuBLAS, NCCL internals and other driver memory
                                of which (GiB): before the first startup step 0.60, Additional executor resources (temporary for KV cache size estimation) 0.38, Model 0.10, Sampler 0.07, Model resources created during usage 0.06, +1 more
  Runtime buffers                 0.23 GiB    0.2%      -0.00 GiB   buffers owned by named runtime components
                                of which (GiB): Additional executor resources 0.10, Model 0.06, Additional executor resources (temporary for KV cache size estimation) 0.03, Model resources created during usage 0.03
  CUDA graphs                     0.05 GiB    0.1%      +0.00 GiB   held by captured CUDA graphs for replay; not reusable
                                of which (GiB): in use by graphs 0.03, free in graph pools 0.02
  PyTorch cache (free)            0.01 GiB    0.0%      -0.72 GiB   held by the PyTorch allocator but not in use; reusable

  Startup plan: peak outside the KV cache 11.68 GiB + KV cache 83.03 GiB = 94.71 GiB; device now 106.25 GiB, excluding 0.01 GiB of free PyTorch cache (+11.55).
    of which the profiled forward pass allocated 0.67 GiB in PyTorch.
  Usage exceeds the startup plan by 11.55 GiB: the workload needs more than profiling measured, or another process grew.

  Conclusion: since warmup, PyTorch, unattributed grew the most (+12.00 GiB).
  Suggestion: PyTorch memory with no known owner grew 12.00 GiB after warmup, 11.33 GiB more than the profiled forward pass allocated (0.67 GiB), so something may be holding on to memory (a cache, a callback, a leak); rerun with TLLM_MEM_MONITOR_STACKS=1 to name the code that allocated it; to reduce activation memory, reduce max_num_tokens.
```

Below the summary, a section breaks the same memory down by allocator and
pool, followed by consistency checks between independent readings.

* **The total** is this process's own GPU memory, as the driver reports it.
  Memory held by other processes on the same GPU is shown separately; when
  they hold more than this process, the conclusion says so.
* **The rows** add up exactly to the total. Large rows list what they are made
  of (`of which`).
  * *Model weights* follows the loaded model, including weights that were
    replaced or moved after loading.
  * *Runtime buffers* are named by the startup step that allocated them.
  * *Communication buffers* name NCCL communicators, NCCL window buffers,
    CUDA IPC buffers (such as the custom allreduce workspace), NVLS and
    multicast workspaces (including the fused GEMM+allreduce one),
    symmetric-memory allreduce workspaces, MNNVL buffers, DeepEP buffers and
    Ulysses all-to-all slots.
  * *CUDA and library overhead* is what no component accounts for: the CUDA
    context, cuBLAS, NCCL internals and code loaded on demand. It is split by
    when it appeared -- before startup, in each startup step, or since.
* **since warmup** compares with the report taken when startup finished. The
  row that grew is usually the one to look at. The warmup report is taken at
  rest, while an out-of-memory report is taken inside a forward pass, so
  growth in *PyTorch, unattributed* up to what the profiled forward pass
  allocated is that step's activations; only growth beyond it is called held
  memory, which `TLLM_MEM_MONITOR_STACKS=1` attributes to code.
* **Startup plan** compares the memory in use with what KV cache sizing
  planned for when the engine started: the peak it profiled outside the KV
  cache, workspace reserves included, plus the KV cache pools. It appears
  only when sizing profiled a forward pass.
* **Suggestion** appears for an out-of-memory error, or a startup that failed
  because it ran out of device memory, and names the setting that controls
  the largest consumer. A startup that fails for another reason, such as a
  missing checkpoint key, gets none.
* **The headline** names the failed PyTorch allocation only when the error is
  that failure, the same size on the same device; any other error is headed
  by its own first line.

While parts of the runtime are released with `LLM.release()`, the rows show
only memory that is resident, and one line lists what is released and returns
on resume.

## Limitations

* **Out-of-memory errors during startup are not reported in the default
  mode.** Loading weights, sizing the KV cache, warmup and CUDA graph capture
  deliberately run allocations until they fail and recover, so reporting is
  armed only once the executor has been created. With `TLLM_MEM_MONITOR=1` a
  startup that fails is reported, with the step that failed.
* **The process total needs NVML to see this process.** In a container
  without the host PID namespace (for example `docker run` without
  `--pid=host`) NVML lists host PIDs, and the report falls back to the whole
  device's usage, marked as an upper bound. Other processes' memory on the
  same GPU then shows up in *CUDA and library overhead*.
* **No comparison with warmup in the default mode.** The rows and the total
  are the same, but *since warmup*, the startup plan and the split of the
  overhead row by startup step need `TLLM_MEM_MONITOR=1`.
* **Allocation stacks need PyTorch's native allocator.** With
  `PYTORCH_ALLOC_CONF=backend:cudaMallocAsync` or a pluggable allocator,
  `TLLM_MEM_MONITOR_STACKS=1` records nothing.
* **Startup records belong to one executor per process.** Each executor's
  components are reported separately. When an executor starts while another
  one in the same process is still running, its startup is not recorded
  (a warning says so), so the running executor's startup breakdown, warmup
  comparison and buffer names stay correct; the records are dropped when the
  executor they describe shuts down.

## Cost

In the default `oom` mode no work is done while serving until an allocation
fails. When startup completes, the process's memory pools are classified
once, which walks the Python heap (about 100 ms per million objects), so
that a later report does not have to. With `TLLM_MEM_MONITOR=1`, allocation
history is recorded during startup only, and a report takes about a
millisecond once startup has finished. Neither mode adds work to the
executor loop. `TLLM_MEM_MONITOR_STACKS=1` is the exception and slows every
PyTorch allocation while it is on.
