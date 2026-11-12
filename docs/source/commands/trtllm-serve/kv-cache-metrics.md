<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tiered KV-cache observability

Use tier occupancy, movement/drop counters and optional migration timing to
investigate poor prefix reuse in conversation workloads. These metrics extend
the existing Prometheus collector and do not require the shared performance
collector refactor.

## Enable collection

```yaml
return_perf_metrics: true
enable_iter_perf_stats: true
enable_iter_req_stats: false
iter_perf_stats_interval: 1
```

Save this as `metrics.yaml`, then start serving and inspect the endpoint:

```bash
trtllm-serve /path/to/model --config metrics.yaml --port 8000
curl --fail http://localhost:8000/prometheus/metrics
```

Use KVCacheManagerV2 with the tier capacities appropriate to your deployment.
The metric families use the `trtllm_` prefix. Request iteration diagnostics are
not required. Timing is disabled by default; enable it explicitly as described
below. No serving configuration fields are added.

## Labels

The normal serving labels are `model_name` and `engine_type`; Prometheus adds
`instance` and `job`. Keep the worker instance when diagnosing cache pressure.

| Label | Meaning |
| --- | --- |
| `dp_rank` | Reporting attention-DP rank; absent rank measurements are not zero. |
| `cache_level`, `cache_tier` | Configured cache level index and backing tier. |
| `pool_group` | Pool-group index within its cache level; hot and cold numbering can differ. |
| `state` | Disjoint `free`, `evictable`, or `unavailable` storage slots. |
| `operation` | `restore`, `evict`, `drop_capacity`, or `drop_priority`; timing covers only restore and evict. |
| `source_cache_level`, `source_cache_tier`, `destination_cache_level`, `destination_cache_tier` | Actual migration path. |

## Capacity, movement, and latency

The following V2 families expose storage pool groups at every configured cache
level supplied by the producer. They distinguish two GPU or host levels rather
than merging equal tier names. They require `enable_iter_perf_stats`; request
iteration statistics are not required. Existing V1 rows do not fabricate these
measurements. Older native builds can supply occupancy without the new movement
fields, in which case movement/timing samples remain absent.

| Metric | Type | Meaning |
| --- | --- | --- |
| `kv_cache_tier_blocks` | Gauge | Physical storage slots by `dp_rank`, `cache_level`, `cache_tier`, `pool_group`, and disjoint `state`. |
| `kv_cache_tier_bytes` | Gauge | The same slot counts multiplied by the pool group's summed slot sizes. Logical payload capacity, not allocator padding or device-wide memory usage. |
| `kv_cache_tier_moved_blocks_total` | Counter | Cumulative slot movements/drops by level, pool group and `operation`. |
| `kv_cache_tier_moved_bytes_total` | Counter | Movement/drop counts multiplied by the attributed pool's slot size. With compressed cold pages, source and destination bytes can differ. |
| `kv_cache_tier_migration_duration_seconds` | Histogram | Sampled migration-batch CUDA-stream duration with source/destination labels; opt-in as described below. |
| `kv_cache_migration_timing_skipped_total` | Counter | Eligible timing samples skipped because the manager's pending-event budget was full, by reporting rank. |

`free` means unallocated slots. `evictable` means resident pages the policy
allows the engine to reclaim; these can still be valuable conversation prefixes.
`unavailable = capacity - free - evictable` includes held/locked slots and other
non-reclaimable storage. The three states sum to capacity. Unlike this new
`state="free"`, legacy `kv_cache_free_blocks` includes evictable slots.

Use resident occupancy `(unavailable + evictable) / capacity` to see whether
prefixes have filled the tier, and `unavailable / capacity` to see immediate
allocation pressure. Neither alone proves thrashing. Combine them with
movement/drop rates, recent reused/missed block rates, tiered prompt-cache hits,
and request TTFT. A fully resident cache with low churn may be working well.

Movement operations have explicit attribution:

- `restore`: accepted cross-level transfer toward a hotter level, counted on
  its destination level/pool. Includes cold-to-cold disk prefetch and partial
  slot copies; it is not limited to recovery of suspended requests.
- `evict`: accepted downward transfer, counted on its source level/pool. The
  prefix still exists on a colder level; this is not permanent cache loss.
- `drop_capacity`: pages discarded while making space or force-evicting, counted
  where they last resided. Includes pages rejected by the final tier; a page
  that bypasses intermediate tiers is attributed to its actual source.
- `drop_priority`: hot pages discarded because their priority is below the
  cold-tier offload threshold, counted on the hot source pool.

Ordinary request release, sliding-window trimming, explicit cache clearing and
shutdown are not pressure-drop events. Same-level copies/defragmentation are
not cross-tier movement. Transfer counters record accepted submissions, not
confirmed successful DMA; a later asynchronous failure does not roll them back.
They are lifetime snapshots, so skipped iteration records do not lose counts.
Byte counters express logical slot traffic on the attributed side, not physical
bus traffic, redundant staging transfers or compressed wire bytes.

Capacity and movement use the manager's existing API lock and already-read
storage snapshots; they add CPU bookkeeping and exporter work. Detailed storage
rows currently follow the existing attention-DP ownership: only rank 0 supplies
them, while non-ADP follows the engine's existing rank-0 reporting. Missing ranks
are absent, not measured zeros. These families cannot establish all-rank pressure
or attribute a specific conversation miss to a prior eviction. Use request traces
and KV eviction events for that correlation; routing, changed prefixes and cache
keys can also cause misses. Do not put conversation IDs or hashes in metric labels.

Interpret these as an attribution aid rather than a single pressure score:

| Observation | Supported interpretation and next check |
| --- | --- |
| High resident occupancy, low movement, good recent hits | Cache is retaining useful prefixes; high occupancy alone is healthy. |
| Frequent eviction/restore, few final drops | Working data is cycling through cold tiers; inspect sampled transfer latency and TTFT. |
| Capacity drops rising with misses | Evidence for retention-capacity pressure; check which tier exhausts and whether a pool-level bottleneck is hidden by the tier average. |
| Priority drops rising with misses | Check retention priorities and cold-tier offload threshold before increasing memory. |
| Poor hits without corresponding pressure drops | Investigate placement across workers/ranks, prefix/key changes, explicit invalidation and the true conversation working set. |

### Investigate low hit rate in a conversation workload

Start with timing disabled and use one application scrape path. Keep worker
`instance`, `dp_rank`, cache level and pool group visible; do not merge prefill
and decode targets or treat rank-0-only detail as an all-rank measurement.

1. Check recent reused/missed counter rates over the same interval as the
   workload. The lifetime `kv_cache_hit_rate` gauge can hide a recent change.
   Reused blocks and missed blocks measure different units from request-level
   cached-token counts; compare trends without adding their totals together.
2. Inspect resident occupancy and the truly free fraction for each tier and
   pool. Reclaimable prefixes still occupy storage and can be needed by the next
   conversation turn. A small exhausted pool can constrain allocation even when
   the tier's byte-weighted average has headroom.
3. Compare normalized `evict`, `restore`, `drop_capacity` and `drop_priority`
   rates. Downward migration preserves a prefix on a colder tier; a final drop
   loses that copy. Drop counts do not count unique conversations or prove that
   a later request would have reused the dropped prefix.
4. If eviction/restore traffic is high, enable sampled timing for a diagnostic
   run and compare host/GPU/disk paths, sample counts and skipped samples with
   TTFT and prefill latency. A high movement rate with fast transfers differs
   from a low movement rate with slow disk restoration.
5. Correlate individual failed reuse cases using request traces and KV events,
   where available. Check worker/rank placement and matching prefix/cache-key
   identity before attributing a miss to capacity. These aggregate metrics do
   not establish that the dropped page belongs to the later missed request.

See the dashboard queries below for [resident occupancy](#per-tier-resident-occupancy),
[turnover](#per-tier-restoreevictiondrop-turnover),
[recent block reuse](#recent-prefill-block-reuse-and-misses), and
[sampled restore latency](#sampled-restore-p95-by-migration-path).

### Optional sampled migration latency

Timing is disabled by default. Set this on the engine processes before startup:

```bash
export TRTLLM_KV_CACHE_TIMING_SAMPLE_INTERVAL=64
```

A positive value N selects the first and then every Nth eligible batch on each
source/destination/pool-group path. Zero or a negative value disables timing.
`enable_stats` must also be enabled on the native KV manager. All movement
counters remain unsampled regardless of this setting.

The setting is read when the native manager is constructed. Set it in the
engine processes' environment before starting the server; changing an
administrative shell after startup does not change an existing manager. To
start a subsequent run with timing disabled:

```bash
export TRTLLM_KV_CACHE_TIMING_SAMPLE_INTERVAL=0
```

This switch controls timing instrumentation only. It does not change cache
eviction policy, tier capacity, request routing, or the full movement counters.

A sample adds start/end timing-enabled CUDA events around the transfer commands.
The event pool is lazy and reuses events; at most 64 samples are pending per
manager, with at most 128 cached timing events. Collection queries completion
without synchronization or waiting; unfinished samples remain pending and ready
samples are drained once. The native `take_migration_timings()` API returns those
completed samples and a lifetime cumulative skipped count; it never resets the
skipped count. `get_and_reset_migration_timings()` remains a compatibility alias.
A full budget skips eligible samples and increments
`kv_cache_migration_timing_skipped_total`. No stats polling means no further
retirement, so an idle worker may expose results only when another record arrives.
Samples on a stream that never completes retain their budget slots until manager
teardown. There is no timeout retirement: elapsed time alone cannot distinguish
slow work from abandoned work. Retention is bounded by the pending-sample limit;
monitor the skipped counter to detect a saturated timing sampler.

The histogram measures a batch's stream interval, including codec and staging
work on that path and interference from concurrent work. It excludes dependencies
already ordered before the start event and is not a per-block latency, exclusive
GPU busy time, or a request's total restore stall. Timing a batch does not time a
final drop. The histogram count is sampled batches, not total operations or blocks;
do not multiply sampled counts or sums by N to claim exact totals. A systematic
sample and saturation can bias percentiles, and completed requests are not its
sampling unit. CPU/event/exporter overhead has not been benchmarked; compare
sampling disabled/enabled under the same conversation workload before production
use. Disabling timing creates no timing events, but CPU counters still have cost.

For an overhead comparison, use the same model, topology, cache capacities,
conversation dataset, arrival pattern and scrape settings in both runs. Keep
cache warmup and measurement phases comparable and report cache-hit/movement
rates alongside TTFT, ITL and output throughput. A changed cache state can
change performance independently of instrumentation. The implementation has
not yet been validated by this comparison; no zero-overhead or negligible-tail-
latency claim is established.

## Grafana setup

Use the [Grafana example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/serve/grafana)
for physical tier occupancy, turnover, final drop, sampled restore-latency and
skipped-sample panels in Prefill/Decode views. The [setup walkthrough](prometheus-metrics.md#create-a-grafana-dashboard)
covers offline rendering, datasource/workload selection and explicit publication.
Missing native observations hide these optional panels; the dashboard does not
change collection flags or migration timing's default-off setting.

## Dashboard queries

### Per-tier resident occupancy

Bytes weight pools with different slot sizes correctly. Keep pool-group series
when looking for an individual allocation bottleneck; a tier average can hide it.

```promql
sum by (instance, dp_rank, cache_level, cache_tier) (
  trtllm_kv_cache_tier_bytes{state=~"unavailable|evictable"}
)
/
sum by (instance, dp_rank, cache_level, cache_tier) (trtllm_kv_cache_tier_bytes)
```

Select only `state="unavailable"` in the numerator for immediate allocation
pressure, or `state="evictable"` for the retained reclaimable-prefix fraction.
An empty zero-capacity pool has an undefined ratio, not measured zero pressure.

### Per-tier restore/eviction/drop turnover

```promql
sum by (instance, dp_rank, cache_level, cache_tier, operation) (
  rate(trtllm_kv_cache_tier_moved_bytes_total[5m])
)
/ ignoring(operation) group_left
sum by (instance, dp_rank, cache_level, cache_tier) (trtllm_kv_cache_tier_bytes)
```

Units are tier-capacities per second. Replace `rate` with `increase` for the
fraction of current capacity moved during the window; it can exceed 100%.
Use the matching block families for per-pool block turnover. Changing capacity
during the window changes this normalization; it is not a fraction of unique
prefixes. Track `evict` separately from `drop_capacity` and `drop_priority`.
Sustained eviction/restore with high resident occupancy suggests churn;
capacity drops followed by increased misses support working-set overflow.
These aggregate correlations alone cannot prove why one conversation missed.

### Recent prefill block reuse and misses

Select the prefill scrape targets when diagnosing conversation-prefix reuse.
Use the same labels and time window for the reused and missed block streams:

```promql
sum by (instance, model_name, dp_rank) (
  rate(trtllm_kv_cache_reused_blocks_total[5m])
)
/
(
  sum by (instance, model_name, dp_rank) (
    rate(trtllm_kv_cache_reused_blocks_total[5m])
  )
  +
  sum by (instance, model_name, dp_rank) (
    rate(trtllm_kv_cache_missed_blocks_total[5m])
  )
)
```

Graph `rate(trtllm_kv_cache_missed_blocks_total[5m])` separately: a ratio alone
does not show the volume of recomputation demand. The ratio is undefined when
there are no new reused or missed blocks; do not turn that into a measured
zero hit rate. This block ratio is not the fraction of requests with a hit.

### Per-tier capacity-drop fraction over a window

```promql
sum by (instance, dp_rank, cache_level, cache_tier) (
  increase(trtllm_kv_cache_tier_moved_bytes_total{operation="drop_capacity"}[5m])
)
/
sum by (instance, dp_rank, cache_level, cache_tier) (trtllm_kv_cache_tier_bytes)
```

For example, a value of 2 means the dropped logical bytes equal twice the
currently reported tier capacity during five minutes. It does not mean that
200% of unique cache contents were lost: slots can be refilled and dropped
repeatedly. Use `operation="drop_priority"` separately to investigate policy-
driven loss, and retain `pool_group` when locating an individual pool bottleneck.

### Sampled restore p95 by migration path

```promql
histogram_quantile(
  0.95,
  sum by (le, instance, source_cache_level, destination_cache_level) (
    rate(trtllm_kv_cache_tier_migration_duration_seconds_bucket{operation="restore"}[5m])
  )
)
```

Compare source/destination paths instead of merging disk and host restoration.
Check sample counts and skipped timing samples alongside this percentile.

The following query counts the actual completed timing observations in the
window, with the same path grouping:

```promql
sum by (instance, source_cache_level, destination_cache_level) (
  increase(trtllm_kv_cache_tier_migration_duration_seconds_count{operation="restore"}[5m])
)
```

Also graph `rate(trtllm_kv_cache_migration_timing_skipped_total[5m])`.
No timing observations can mean timing is disabled, no eligible migration
occurred, samples are still pending, or source records have not been delivered;
it is not evidence of zero restore latency. Sparse samples and a saturated
pending-event budget limit the interpretation of a percentile.
