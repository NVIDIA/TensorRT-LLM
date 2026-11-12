<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared performance metrics collection

## Shared collection and exposure

Performance measurements are collected before selecting their output. The
backend-independent `PerfMetricsCollector` creates owned immutable
`PerfMetricsRecord` values with three categories:

| Category | Measurements |
| --- | --- |
| `per_req` | Candidate-completion latency/tokens/cache metrics, or explicitly enabled diagnostic request iteration records. Diagnostic request records are not a second completion stream. |
| `per_iter` | Completed-batch iteration data and separately identified loop-boundary observations: host/device timing, scheduled/active/unscheduled/dummy work, fetched requests, and model state. |
| `per_rank` | Rank-owned KV snapshots, memory and capacity limits. Actual snapshot capture iteration/time remain explicit; `batchIter` separately identifies the batch whose processing triggered collection. |

Request and iteration records also retain their source rank. A delayed batch
record and its attached `executorLoop` can describe different iterations under
overlap or pipeline parallelism. The loop has its own iteration, rank and
`loopSequence`; the sequence advances even while the executor iteration counter
is stalled. Free-form model state and iteration identifiers remain diagnostic
data, not Prometheus labels.

`LoggerExposure` formats collected loop records in the executor thread using
the existing iteration-line fields. It also writes structured `per_rank`
records when a new rank snapshot is collected. With `enable_iter_req_stats`
and `enable_iter_perf_stats`, request iteration diagnostics are available to
the logger as categorized records. Detailed rank snapshots are exposed when
collected, so the final snapshot need not wait for another executor loop.

`PrometheusExposure` is the long-lived serving-side adapter. HTTP and OpenEngine
normalize transported records through the same category model before exposing
them. `MetricsCollector`, `log_iteration_stats` and `log_request_metrics_dict`
remain compatibility interfaces. Scraping and idle periods do not trigger
source reads, reset counters or require an active request.

Neither exposure reads an executor/KV manager or drains native events. Source
collection alone owns those operations. When batch stats are enabled, the loop
logger reuses collected rank snapshots without an additional KV query. In
logger-only mode the collector retains the existing basic-KV refresh cadence
of 16 loop boundaries on selected logging ranks. Detailed KV/tier collection
still requires `enable_iter_perf_stats`; enabling text alone does not enable
all expensive request/tier diagnostics.

| Use case | Existing controls |
| --- | --- |
| Production Prometheus | `return_perf_metrics: true`, `enable_iter_perf_stats: true`, `print_iter_log: false` |
| Offline iteration text with basic KV utilization | `print_iter_log: true`; no HTTP server or Prometheus client instance is required |
| Offline text plus detailed rank observations | `print_iter_log: true`, `enable_iter_perf_stats: true`; optionally enable request iteration diagnostics |
| Both outputs | Enable iteration text alongside production collection; source reads and instrumentation are shared |

This separation adds record construction/serialization and optional logging
work. It introduces no CUDA events or synchronization beyond the existing
collection/timing settings. The existing device-step profiler timing policy is
preserved. CPU and
end-to-end overhead of this refactor have not been benchmarked.

## Create a Grafana dashboard

The [Grafana setup example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/serve/grafana)
provides a standard-library Python renderer and publisher for Kubernetes serving.
It discovers engine workers and has E2E, Prefill, Decode and All views. Configure
your workload pod/job patterns, datasource UIDs, endpoint/container labels and
Grafana's supported schema-v2 API version in a copy of `example.config.json`.
The default configuration generates metrics only; Loki, scrape-resource creation
and trace-PVC panels are optional.

```sh
cp examples/serve/grafana/example.config.json /tmp/trtllm-dashboard.json
# Edit the copied configuration for your deployment.
python3 examples/serve/grafana/dashboard.py render \
  --config /tmp/trtllm-dashboard.json --output /tmp/trtllm-dashboard-rendered
python3 examples/serve/grafana/dashboard.py deploy \
  --config /tmp/trtllm-dashboard.json --output /tmp/trtllm-dashboard-rendered --dry-run
```

Rendering is offline. The deployment dry-run validates the selected Grafana API
and enabled Kubernetes resources without publishing them. Run `deploy` without
`--dry-run` to publish. Use API persistence for API-owned dashboards and
`configmap-v2` for existing file-sidecar ownership; applying its generated ConfigMap
alone is incomplete. The tool guards resource versions and reads back saved
content. ConfigMap checkpoint plus native finalization is not a cross-system
transaction; the README documents `--finalize-only` recovery.

The 80 default metric templates include physical tier states/resident occupancy,
restore/eviction turnover, final drops, sampled restore latency/skips and
completed-request transfer duration. Native ratios use logical byte capacity and
remain separated by level/tier and reporting rank. Unsupported observations hide
optional panels. GPU active/available block charts retain their separate meaning:
available blocks can contain reclaimable prefixes. These panels cannot prove
which conversation prefix was evicted.

Select one application scrape job path to avoid duplicate engine observations.
Existing `irate` charts use the last two samples in `counter_lookback`; new tier
rate/turnover charts average the window, and sampled latency uses a 5-minute
histogram window. Missing metrics, sparse timing samples, rank coverage and native
build requirements still apply; dashboard creation does not enable instrumentation.

The template includes optional KV-cache tier and transfer panels whose added
instrumentation is provided separately by [PR #22083](https://github.com/NVIDIA/TensorRT-LLM/pull/22083).
This metrics refactor does not introduce those metric families.
