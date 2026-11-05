<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prometheus performance metrics

The HTTP server and OpenEngine gRPC server use the shared `MetricsCollector`
and expose engine metrics with the `trtllm_` prefix. Enable request metrics and
iteration statistics in your serving configuration:

```yaml
return_perf_metrics: true
enable_iter_perf_stats: true
iter_perf_stats_interval: 1
```

```bash
trtllm-serve /path/to/model --config metrics.yaml --port 8000
curl --fail http://localhost:8000/prometheus/metrics
```

For OpenEngine, use its separate HTTP metrics port (8001 by default), as
described in the [OpenEngine metrics guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/grpc/openengine/README.md#prometheus-metrics).
When using Dynamo with an OpenEngine worker, scrape this engine endpoint as a
separate Prometheus target.

## Existing coverage

These are the main performance metric families. All names below have the
`trtllm_` prefix.

| Area | Metrics |
| --- | --- |
| Request latency | `e2e_request_latency_seconds`, `time_to_first_token_seconds`, `time_per_output_token_seconds`, `request_queue_time_seconds`, `request_prefill_time_seconds`, `request_decode_time_seconds`, `request_inference_time_seconds` histograms |
| Request outcomes and tokens | `request_success_total`, `request_error_total`, `prompt_tokens_total`, `generation_tokens_total`, cached-prompt counters and histograms |
| Scheduler load | `num_requests_running`, `num_requests_waiting`, `num_requests_completed_total`, `max_num_active_requests`, and batch/token limits |
| Local batching | `num_context_requests`, `num_generation_requests`, `num_paused_requests`, `num_scheduled_requests`, `total_context_tokens`, `prefill_batch_occupancy`, `prefill_batch_tokens` |
| KV cache | Cache utilization, hit/reuse rates, free/used/max blocks, reused/missed blocks, and per-iteration allocation/transfer counters; additional lifecycle and storage-tier views when provided by the engine |
| Memory | `gpu_memory_usage_bytes`, `cpu_memory_usage_bytes`, `pinned_memory_usage_bytes` |
| Speculative decoding | Drafted/accepted tokens, acceptance length, draft overhead, and bounded per-position counters |
| Configuration | Model, parallelism, speculative-decoding, and KV-cache configuration info |

Queue, memory, legacy iteration latency, and speculative-decoding iteration
metrics are rank-0 measurements. Local KV and scheduling metrics carry
`dp_rank`. They should not all be interpreted as deployment-wide totals.

## Iteration-log timings

The following metrics expose timing fields already present in the engine's
structured iteration statistics. `print_iter_log` does not need to be enabled.
Values are converted from milliseconds to seconds.

| Metric | Type | Engine field and meaning |
| --- | --- | --- |
| `trtllm_host_step_time_seconds` | Gauge | `hostStepTimeMS`: latest executor-loop wall time, corresponding to the log's `host_step_time` |
| `trtllm_prev_device_step_time_seconds` | Gauge | `prevDeviceStepTimeMS`: latest device-step timing from the ping-pong CUDA events, corresponding to `prev_device_step_time` |
| `trtllm_gpu_forward_time_seconds` | Gauge | `gpuForwardTimeMS`: GPU-forward timing matched to the exported batch |
| `trtllm_host_step_duration_seconds` | Histogram | Distribution of exported `hostStepTimeMS` samples |
| `trtllm_gpu_forward_duration_seconds` | Histogram | Distribution of exported `gpuForwardTimeMS` samples |

These metrics use the model labels plus `scheduler_mode`, restricted to
`overlap`, `non_overlap`, or `unknown` for older or unrecognized records. They
are updated only from rank 0: attention-DP fanout currently copies rank-0
timings to other rank rows, so counting each row would duplicate observations.
They do not measure rank imbalance or aggregate GPU time across workers.

`trtllm_iteration_latency_seconds` remains available with its existing labels
and semantics. With overlap scheduling (including pipeline parallelism), its
`iterLatencyMS` source spans the batch lifecycle, roughly two executor loops.
Use `host_step_time_seconds` for single-loop wall time. The lagged
`prev_device_step_time_seconds` describes an older loop than the host timing
beside it; subtracting these gauges does not produce a valid CPU overhead
measurement. Even the batch-matched GPU timing does not have the same scope
as the host-loop timer.

Missing, null, negative, or non-finite timing fields do not create observations.
Zero is a valid measurement. Gauges retain their previous sample when no new
measurement is available, including while the engine is idle.

Histograms observe every valid timing record delivered to the collector,
rather than just the last value before a scrape. With
`iter_perf_stats_interval > 1`, they describe sampled iterations. Their counts
do not count skipped iterations, and their sums must not be treated as exact
engine busy time. Records lost through a bounded engine stats buffer also
cannot contribute to these distributions. Buckets span 0.1 ms through 10 s,
with the standard `+Inf` bucket for longer samples.

For example, plot the p99 host-loop time over five minutes:

```promql
histogram_quantile(
  0.99,
  sum by (le, model_name, scheduler_mode) (
    rate(trtllm_host_step_duration_seconds_bucket[5m])
  )
)
```

For diagnosis of one worker, retain the scrape target's `instance` label in
the aggregation as well.

## Fields that remain in iteration logs

The log also includes routing fetch counts, dummy and unscheduled request
counts, dummy context/generation tokens, and model state details. These are
not all present in the structured iteration-statistics records, so the shared
collector cannot expose them directly. Adding them requires capturing
iteration-matched rank-local snapshots in the executor and attention-DP
fanout. Iteration IDs, timestamps, and free-form model states should not be
used as Prometheus labels.
