(perf-analysis)=

# Performance Analysis

NVIDIA Nsight Systems reports at the application level are highly informative. Metric sampling capabilities have increased over generations and provide a clean middle-ground between timing analysis and kernel-level deep dives with NVIDIA Nsight Compute.

Given the potential long runtimes of Large Languages Models (LLMs) and the diversity of workloads a model may experience during a single inference pass or binary execution, NVIDIA has added features to TensorRT LLM to get the most out of Nsight Systems capabilities. This document outlines those features as well as provides examples of how to best utilize them to understand your application.


## Feature Descriptions

The main functionality:
  * Relies on toggling the CUDA profiler runtime API on and off.
  * (PyTorch workflow only) Toggling the PyTorch profiler on and off.
  * Provides a means to understand which regions a user may want to focus on.

Toggling the CUDA profiler runtime API on and off:
  * Allows users to know specifically what the profiled region corresponds to.
  * Results in smaller files to post-process (for metric extraction or similar).

(PyTorch workflow only) Toggling the PyTorch profiler on and off:
  * Help users to analyze the performance breakdown in the model.
  * Results in smaller files to post-process (for metric extraction or similar).

### Perf Time Events (per-rank live capture)

For scheduling / gen-bubble debugging on the PyTorch workflow, TensorRT LLM can
capture a per-request timeline directly from inside the executor and write it
**live, one line per event, flushed as each event happens** — no HTTP
`/perf_metrics` scrape and no server teardown race. Because every event is
appended the instant it fires, a request that **never completes** (a KV-transfer
timeout livelock at iter 0, a fill-gate livelock, a HangDetector `MPI_Abort` at
300 s) still leaves its *partial* timeline on disk, and the last line it wrote
localizes exactly where it wedged. Only the **gen-init request's lifecycle** is
recorded — per-decode-step events are deliberately not emitted, so the volume is
a handful of lines per request rather than a dense per-step series.

```bash
# Set on every worker (both ctx and gen servers for a disaggregated run).
export TRTLLM_PERF_TIME_EVENTS_PATH=/tmp/perf_events
# Optional but recommended for disaggregated serving — the KV-transfer CSVs
# written by the native transceiver perf logger:
export TRTLLM_KVCACHE_TIME_OUTPUT_PATH=/tmp/kv_csv
# Optional — extend the timeline to the two processes outside the executor.
# Set on the disaggregated router process (uvicorn) and on the benchmark client
# respectively; each writes its own per-pid JSONL, independent of the workers:
export TRTLLM_PERF_TIME_EVENTS_ROUTER_PATH=/tmp/perf_events_router
export TRTLLM_PERF_TIME_EVENTS_CLIENT_PATH=/tmp/perf_events_client
```

Setting `TRTLLM_PERF_TIME_EVENTS_PATH` (to an output **directory**, symmetric
with `TRTLLM_KVCACHE_TIME_OUTPUT_PATH`) does two things:

* **Force-enables capture** regardless of `return_perf_metrics` / `LLM` args, so
  you do not have to thread the flag through the benchmark client.
* Starts a **daemon writer thread per rank** that drains an in-process queue and
  appends flat event lines to `time_events_rank{N}_pid{P}.jsonl`. The executor
  loop only builds a small dict and does a non-blocking `queue.put_nowait`
  (drop-on-full), so capture stays off the critical path (no file I/O on the loop
  thread).

Every worker and router line is one flat envelope, identical across roles:

```json
{"role": "gen", "event": "gen_first_token", "request_id": 42,
 "ctx_request_id": "abc", "rank": 3, "t": 12345.678, "pid": 12345}
```

`role` ∈ `{router, ctx, gen, client}`; `event` is the lifecycle event name;
`request_id` joins the TP ranks of one worker request (worker) or is the
router-local sequence (router); `ctx_request_id` is the cross-process join key;
`rank` is the global worker rank (router/client use `0`); `t` is `steady_clock`
seconds (the same clock the C++ `/perf_metrics` scalars use); `pid` is the
emitting process.

The captured events are:

* **ctx worker** (`role=ctx`): `ctx_arrival`, `ctx_first_scheduled`,
  `ctx_first_token`, `ctx_ready_sent`.
* **gen worker** (`role=gen`, gen-init lifecycle only): `gen_arrival`,
  `gen_init_scheduled`, `gen_kv_transfer_start`, `gen_kv_transfer_end`,
  `gen_first_scheduled`, `gen_first_token`, `gen_last_token`.
* **router** (`role=router`): `arrival`, `ctx_dispatch`, `gen_dispatch`,
  `first_token`, `resp_done` (router lines also carry `disagg_request_id` /
  `ctx_server` / `gen_server` provenance).

**Known limitations:**

* **Two clock domains.** The steady clocks split into domain A = {router, gen
  worker} and domain B = {ctx worker, client}, which are not NTP-aligned to each
  other, so a cross-domain subtraction (e.g. ctx → gen relay) is invalid off the
  raw event lines. The offline compiler therefore derives only *within-domain*
  spans from the events; the 12 canonical cross-worker spans come only from an
  offset-corrected aggregated `/perf_metrics` dump (`--perf-json`).
* **ctx ⇄ gen correlation** across the two disaggregated servers relies on the
  request's disaggregated `ctx_request_id`. The gen-only benchmark path hardcodes
  `ctx_request_id=1`, so a `ctx_request_id` shared by more than one request is
  treated as **non-joinable** rather than false-attached.

**Extending the timeline beyond the workers (router + client).** The two env
vars above capture the two processes the executor never sees, so the timeline can
span client-send → router-dispatch → worker-execute:

* `TRTLLM_PERF_TIME_EVENTS_ROUTER_PATH` — set on the **disaggregated router**
  (the `trtllm-serve disaggregated` uvicorn process). Each router event writes one
  flat line to `disagg_router_pid{P}.jsonl` (`arrival` / `ctx_dispatch` /
  `gen_dispatch` / `first_token` / `resp_done`), stamped on the router's steady
  clock and keyed by `ctx_request_id` when known (gen-only requests have none and
  are collected separately).
* `TRTLLM_PERF_TIME_EVENTS_CLIENT_PATH` — set on the **benchmark client**
  (`benchmark_serving.py`). The client writes **one compound `client_pid{P}.jsonl`
  record per request** (send / first-token / completion times + `ttft` /
  `latency` / `output_tokens`), not per-event lines: a post-hoc batch write, immune
  to server-side hangs, preserving the vLLM `ttft` / `e2e` / `tpot` path. It shares
  no request id or clock epoch with the server, so the client timeline is
  **standalone** — surfaced verbatim, never joined onto a request record.

Both the router and the client stay torch-free (they must — see the GPU-free
import contract), so they use the same stdlib-only writer as the workers
(`tensorrt_llm/serve/perf_time_events_writer.py`); they already hold the steady
clock and pass `t` in, so the writer never imports it.

**Offline compiler (long → wide).** To stitch the per-rank event files (plus the
KV CSVs, the router file, and the client file) into **one combined record per
request** — and, optionally, a latency aggregate and the interactive HTML
timeline — run:

```bash
python -m tensorrt_llm.serve.scripts.perf_time_events \
    --event-dir /tmp/perf_events \
    --kv-csv-dir /tmp/kv_csv \
    --router-dir /tmp/perf_events_router \
    --client-dir /tmp/perf_events_client \
    --events-jsonl /tmp/perf_events/combined_time_events.jsonl \
    -o /tmp/perf_events/combined.json \
    -a /tmp/perf_events/agg.jsonl
```

The input flags default to their respective env vars
(`--event-dir` → `$TRTLLM_PERF_TIME_EVENTS_PATH`, `--kv-csv-dir` →
`$TRTLLM_KVCACHE_TIME_OUTPUT_PATH`, `--router-dir` →
`$TRTLLM_PERF_TIME_EVENTS_ROUTER_PATH`, `--client-dir` →
`$TRTLLM_PERF_TIME_EVENTS_CLIENT_PATH`); any one input is sufficient. The
compiler:

* groups events by `request_id`, dedups the TP ranks (first-seen wins, `rank0`
  canonical by filename sort), and joins ctx ⇄ gen ⇄ router on `ctx_request_id`;
* pivots long → wide into **`--events-jsonl combined_time_events.jsonl`** (the
  primary output): one line per request carrying every event timestamp that fired
  plus the derived **within-domain** spans. A hung request emits its **partial**
  record — events that never fired are absent, not zero;
* joins KV-transfer rows by request id, reporting unmatched rows under
  `unjoined_kv_events`; unmatched router rows (including gen-only records whose
  `ctx_request_id` is ambiguous) surface under `unjoined_router_events`;
* carries the client records through untouched (standalone, as noted above);
* emits an optional `-a/--agg-jsonl` latency aggregate — 24 rows of
  mean / P50 / P99 (the 12 canonical spans from `--perf-json`, 5 router spans, 4
  worker-event spans, and the 3 vLLM client metrics `ttft` / `e2e` / `tpot`).

It is a convenience over the per-rank files, not the load-bearing capture path;
only the `--html` path pulls in `plotly` (and it needs `--perf-json` for the
cross-worker lifecycle spans). See [Introduction to KV Cache
Transmission](./kv-transfer.md) for the KV cache transfer CSVs it consumes.


## Coordinating with NVIDIA Nsight Systems Launch

Consult the Nsight Systems User Guide for full overview of options.

On the PyTorch workflow, basic NVTX markers are by default provided. On the C++/TensorRT workflow, append `--nvtx` when calling `scripts/build_wheel.py` script to compile, and clean build the code.

### Only collect specific iterations

To reduce the Nsight Systems profile size, and ensure that only specific iterations are collected, set environment variable `TLLM_PROFILE_START_STOP=A-B`, and append `-c cudaProfilerApi` to `nsys profile` command.


### Enable more NVTX markers for debugging

Set environment variable `TLLM_NVTX_DEBUG=1`.

### Enable garbage collection (GC) NVTX markers

Set environment variable `TLLM_PROFILE_RECORD_GC=1`.


### Enable GIL information in NVTX markers

Append “python-gil” to Nsys “-t” option.


## Coordinating with PyTorch profiler (PyTorch workflow only)

### Collect PyTorch profiler results

1. Set environment variable `TLLM_PROFILE_START_STOP=A-B` to specify the range of the iterations to be collected.
2. Set environment variable `TLLM_TORCH_PROFILE_TRACE=<path>`, and the results will be saved to `<path>`.

### Visualize the PyTorch profiler results

Use [chrome://tracing/](chrome://tracing/) to inspect the saved profile.


## Examples

Consult the Nsight Systems User Guide for full overview of MPI-related options.

### Profiling specific iterations on a `trtllm-bench`/`trtllm-serve` run

Say we want to profile iterations 100 to 150 on a `trtllm-bench`/`trtllm-serve` run, we want to collect as much information as possible for debugging, such as GIL, debugging NVTX markers, etc:

```bash
#!/bin/bash

# Prepare dataset for the benchmark
trtllm-bench --model ${MODEL_PATH} \
    prepare-dataset \
    --output dataset.txt \
    token-norm-dist \
    --num-requests=${NUM_SAMPLES} \
    --input-mean=1000 --output-mean=1000 --input-stdev=0 --output-stdev=0

# Benchmark and profile
TLLM_PROFILE_START_STOP=100-150 nsys profile \
  -o trace -f true \
  -t 'cuda,nvtx,python-gil' -c cudaProfilerApi \
  --cuda-graph-trace node \
  -e TLLM_PROFILE_RECORD_GC=1,TLLM_LLMAPI_ENABLE_NVTX=1,TLLM_TORCH_PROFILE_TRACE=trace.json \
  --trace-fork-before-exec=true \
  trtllm-bench \ # or trtllm-serve command
    --model deepseek-ai/DeepSeek-V3 \
    --model_path ${MODEL_PATH} \
    throughput \
    --dataset /tmp/dataset.txt --warmup 0 \
    --backend pytorch \
    --streaming
```

The Nsight Systems reports will be saved to `trace.nsys-rep`. Use NVIDIA Nsight Systems application to open it.

The PyTorch profiler results will be saved to `trace.json`. Use [chrome://tracing/](chrome://tracing/) to inspect the saved profile.

## MoE Expert Load Balance Analysis (Perfect Router)

For Mixture-of-Experts (MoE) models, performance can vary significantly based on how tokens are routed to experts. Uneven expert load distribution can cause some GPUs to be overloaded while others are underutilized, leading to suboptimal throughput.

TensorRT-LLM provides the `ENABLE_PERFECT_ROUTER` environment variable to help analyze and isolate expert load balancing issues from kernel performance.

### What It Does

When enabled, this feature **bypasses the learned router** and replaces it with pre-computed, perfectly load-balanced routing logits. This creates an idealized scenario where tokens are distributed evenly across all experts and GPUs.

Key behaviors:
- The learned gate/router is still computed (to maintain realistic timing)
- The gate output is **discarded** and replaced with ideal balanced logits
- Logits are pre-computed and cached for common batch sizes to minimize overhead
- Works with all MoE backends (CUTLASS, TRTLLM, TRITON)

```{warning}
This feature is for **performance analysis only**. It produces **incorrect model outputs** because the learned router decisions are discarded. Never use this in production inference.
```

### When to Use It

Use `ENABLE_PERFECT_ROUTER` when you want to:

1. **Establish performance upper bounds**: Measure the theoretical best-case MoE throughput when expert loads are perfectly balanced.

2. **Isolate routing bottlenecks**: Compare performance with vs. without perfect routing to determine if the learned router is causing load imbalance issues.

3. **Test different load balancing strategies**: Validate that MoE kernels and communication patterns behave correctly with balanced loads before implementing custom routing logic.

4. **Benchmark kernel efficiency**: Remove routing variability to get consistent, reproducible kernel performance measurements.

### How to Enable

Set the environment variable before running your workload. This works with both `trtllm-bench` and `trtllm-serve`:

```bash
export ENABLE_PERFECT_ROUTER=1
```

### Example Workflow

```bash
# Step 1: Benchmark with normal (learned) routing
trtllm-bench ...
# or
trtllm-serve ...

# Step 2: Benchmark with perfect routing (upper bound)
ENABLE_PERFECT_ROUTER=1 trtllm-bench ...
# or
ENABLE_PERFECT_ROUTER=1 trtllm-serve ...

# Step 3: Compare the throughput numbers
# If perfect router shows >10% improvement, routing imbalance is significant
```

### Interpreting Results

| Scenario | Interpretation |
|----------|----------------|
| Similar performance with/without perfect router | Router load balancing is not a bottleneck; focus optimization efforts elsewhere |
| Significant improvement with perfect router | The learned router is causing load imbalance; consider router optimization or load balancing strategies |

### Supported Models

```{note}
This feature currently requires model-specific integration. The plumbing to support perfect routing must be added to each MoE model implementation. If you need this feature for a model that doesn't yet support it, you will need to add the integration following the pattern used in existing implementations.
```

```{note}
The perfect router logits are specifically designed for `RenormalizeMoeRoutingMethod` (TopK first, then Softmax). Models using other routing methods such as `DefaultMoeRoutingMethod` or `DeepSeekV3MoeRoutingMethod` would require adapting the logit generation logic to match their routing behavior.
```

Currently supported:
- GPT-OSS (uses `RenormalizeMoeRoutingMethod`)
- DeepSeek-V3 / DeepSeek-R1 (uses `DeepSeekV3MoeRoutingMethod`)
