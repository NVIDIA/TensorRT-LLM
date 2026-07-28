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
**live, one file per rank** — no HTTP `/perf_metrics` scrape and no server
teardown race. This is driven by a single environment variable that also
enriches the timeline with per-iteration batch context:

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
with `TRTLLM_KVCACHE_TIME_OUTPUT_PATH`) does three things:

* **Force-enables capture** regardless of `return_perf_metrics` / `LLM` args, so
  you do not have to thread the flag through the benchmark client.
* Records **extended per-iteration batch context** on each per-iteration metric
  entry: `iter_counter`, `iter_batch_size`, `num_ctx_requests`,
  `num_gen_requests`, `context_token_number`, `generation_token_number`, the
  per-request `req_context_token_number` / `req_generation_token_number`, and the
  per-iteration starvation counters `num_capacity_fitting` / `num_scheduled`.
* Starts a **daemon writer thread per rank** that drains an in-process queue and
  appends to `time_events_rank{N}_pid{P}.jsonl`. The executor loop only builds a
  dict and does a non-blocking `queue.put_nowait`, so capture stays off the
  critical path (no file I/O on the loop thread).

Each JSONL record is `{request_id, rank, ctx_request_id, time_breakdown_metrics}`
— the same `time_breakdown_metrics` shape the `time_breakdown` tool consumes.

On the serve / disaggregated path (where the entrypoint force-sets
`return_perf_metrics` under the env), the record additionally carries a
`request_timing_metrics` sub-dict with the C++ `RequestPerfMetrics` lifecycle
scalars keyed by their stable enum names: `arrival_time`, `first_scheduled_time`,
`first_token_time`, `last_token_time`, `kv_cache_transfer_start`,
`kv_cache_transfer_end`, `kv_cache_size`. These are steady-clock seconds and give
the request's arrival → first-schedule (admission) → first-token → last-token
lifecycle. Enrichment is best-effort: on a raw LLM-API run without perf metrics
the key is simply absent (never a crash).

**Known limitations (Python-only capture):**

* **Starvation is a per-iteration count**, not per-request attribution — the
  Python scheduler exposes only `num_fitting_reqs`.
* **Pipeline parallelism**: the PP executor loop does not record the extended
  fields, so batch-context keys are absent under PP.
* **ctx ⇄ gen correlation** across the two disaggregated servers relies on the
  request's disaggregated `ctx_request_id` when exposed; otherwise the ctx-side
  and gen-side records live in separate per-rank files.

**Extending the timeline beyond the workers (router + client).** The two env
vars above capture the two processes the executor never sees, so the timeline can
span client-send → router-dispatch → worker-execute:

* `TRTLLM_PERF_TIME_EVENTS_ROUTER_PATH` — set on the **disaggregated router**
  (the `trtllm-serve disaggregated` uvicorn process). Each request writes one
  `disagg_router_pid{P}.jsonl` record with the router's steady-clock stamps:
  `arrival_time`, `ctx_dispatch_time` (dispatch to the CTX worker),
  `gen_dispatch_time` (dispatch to the Gen worker), `first_token_time`,
  `resp_done_time`, plus `ctx_server` / `gen_server` and the join key
  `ctx_request_id` (known only after the CTX round-trip; gen-only requests have
  none and are collected separately). The router's clock shares the workers'
  steady-clock epoch, so router-vs-worker spans are directly comparable.
* `TRTLLM_PERF_TIME_EVENTS_CLIENT_PATH` — set on the **benchmark client**
  (`benchmark_serving.py`). Each request writes one `client_pid{P}.jsonl` record:
  `send_time` / `first_token_time` / `completion_time` (process-local
  `time.perf_counter`), `ttft`, `latency`, and `send_wall_time` (`time.time`).
  The client is torch-free and its `response_id` is a fresh UUID, **not** a
  worker join key — so the client timeline is **standalone**, aligned to the rest
  only through the wall-clock anchor. It is emitted verbatim in the combined
  JSON's `client_events`, never joined onto a request record.

Both the router and the client stay torch-free (they must — see the GPU-free
import contract), so they use a stdlib-only writer with an `atexit` flush rather
than the worker's daemon-thread writer.

**Optional offline aggregator.** To stitch the per-rank files (plus the KV CSVs,
the router file, and the client file) into one combined JSON — and, optionally,
the interactive HTML timeline — run:

```bash
python -m tensorrt_llm.serve.scripts.perf_time_events \
    --event-dir /tmp/perf_events \
    --kv-csv-dir /tmp/kv_csv \
    --router-dir /tmp/perf_events_router \
    --client-dir /tmp/perf_events_client \
    -o /tmp/perf_events/combined.json \
    --html /tmp/perf_events/timeline.html
```

The four input flags default to their respective env vars
(`--event-dir` → `$TRTLLM_PERF_TIME_EVENTS_PATH`, `--kv-csv-dir` →
`$TRTLLM_KVCACHE_TIME_OUTPUT_PATH`, `--router-dir` →
`$TRTLLM_PERF_TIME_EVENTS_ROUTER_PATH`, `--client-dir` →
`$TRTLLM_PERF_TIME_EVENTS_CLIENT_PATH`); any one input is sufficient. The
aggregator:

* joins KV-transfer rows by request id (`unique_rid` for the native transceiver;
  `RequestID` for C++), reporting unmatched rows under `unjoined_kv_events`;
* joins the router record onto each worker record by `ctx_request_id`, exposing
  it as `router_dispatch` and deriving the cross-process spans
  `router_arrival_to_ctx_dispatch`, `router_ctx_to_gen_dispatch`, and
  `router_to_worker_arrival`; unmatched router rows (including gen-only records
  with no `ctx_request_id`) surface under `unjoined_router_events`;
* derives the worker-side request-lifecycle spans `arrival_to_first_schedule`,
  `schedule_to_first_token`, and `decode_duration` from `request_timing_metrics`
  when present;
* carries the client records through untouched under `client_events` (standalone,
  as noted above);
* computes derived `inter_step_gaps` / `inter_chunk_gaps` and per-iteration
  `starved` counts, and emits a match-rate warning for each join.

It is a convenience over the per-rank files, not the load-bearing capture path;
only the `--html` path pulls in `plotly`. See [Introduction to KV Cache
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
