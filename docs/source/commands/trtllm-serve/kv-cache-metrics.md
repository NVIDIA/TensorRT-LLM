<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tiered KV-cache observability

Use the [Prometheus setup](prometheus-metrics.md) with KVCacheManagerV2 and
`enable_iter_perf_stats: true`. The metrics below use the `trtllm_` prefix.
V1 and older native builds omit unsupported measurements; absence is not zero.

## Capacity and movement

All families carry `dp_rank`, `cache_level`, `cache_tier` and `pool_group`, except
`kv_cache_migration_timing_skipped_total`, which carries only `dp_rank`.
Level indices distinguish two levels backed by the same tier. Pool-group
numbering is local to each level. Detailed tier snapshots currently follow the
existing rank-0 ownership under attention DP; they do not measure every rank.

| Metric | Type | Additional labels and meaning |
| --- | --- | --- |
| `kv_cache_tier_blocks` | Gauge | `state`: disjoint free, evictable and unavailable physical slots. |
| `kv_cache_tier_bytes` | Gauge | Same states multiplied by the pool's summed slot sizes; logical capacity, excluding allocator padding. |
| `kv_cache_tier_moved_blocks_total` | Counter | `operation`: restore, evict, drop_capacity or drop_priority. |
| `kv_cache_tier_moved_bytes_total` | Counter | Movement multiplied by the attributed pool's slot size, not wire traffic. |
| `kv_cache_tier_migration_duration_seconds` | Histogram | Sampled restore/evict batches, with source/destination level and tier labels. |
| `kv_cache_migration_timing_skipped_total` | Counter | Eligible timing samples skipped at the pending-event limit. |

`free` means unallocated. `evictable` means resident pages reclaimable by eviction
policy. `unavailable = total - free - evictable` includes held slots and retained
cold-tier copies, which are also reclaimable; it differs from native
`StorageStatistics::unavailable()`. Legacy `kv_cache_free_blocks` includes evictable
slots. The three new states sum to capacity, but do not isolate all reclaimable
storage. Resident occupancy is `(unavailable + evictable) / total`.

`restore` counts accepted transfers toward a hotter destination, on that
destination pool; it includes cold-to-cold prefetch and partial-slot copies.
`evict` counts downward transfers on the source pool and preserves a colder copy.
`drop_capacity` counts discarded pages where they last resided, including pages
rejected by the final tier. `drop_priority` counts hot pages below the offload
priority threshold. Request release, sliding-window trimming, clearing and
same-level copies are excluded from pressure drops/cross-tier movement.

Movement is cumulative across skipped records and counts accepted submissions;
asynchronous failure does not roll it back. Source/destination logical byte sizes
can differ under compression. Collection uses the existing manager API lock and
storage snapshots, adding bookkeeping and exporter work.

High occupancy alone is not pressure. Compare per-pool drops, eviction/restore
rates, recent reuse/miss rates and TTFT. Low hits without drops can reflect routing,
changed prefixes or cache keys. Aggregate metrics cannot identify which
conversation lost a reusable prefix; correlate request traces and KV events.

## Optional sampled migration latency

Set `TRTLLM_KV_CACHE_TIMING_SAMPLE_INTERVAL=64` in engine processes before startup.
A positive N selects the first and every Nth eligible batch per source/destination/
pool path. Zero or negative disables timing (the default). Native `enable_stats`
must also be enabled. Movement counters remain unsampled.

A selected batch records start/end CUDA events around transfer commands. At most
64 samples are pending per manager, with at most 128 cached events. Collection
queries completion without waiting; completed samples drain once. Without polling,
samples remain pending; a never-completing stream holds its slot until teardown.
Skipped-sample counts are lifetime cumulative and reveal budget saturation.

The duration includes path codec/staging work and concurrent-stream interference,
not dependencies preceding the start event. It is neither per-block latency nor
a request's restore stall. Sampling and saturation can bias percentiles; inspect
sample counts and skipped samples, and do not multiply histogram sums/counts by N
to claim exact totals. No samples can mean disabled timing, pending work or missing
records. Timing overhead has not been benchmarked; compare the same warmed-up
workload with sampling disabled/enabled before drawing latency conclusions.

## Dashboard queries

Keep worker, rank and level labels. Byte-weighted resident occupancy:

```promql
sum by (instance, dp_rank, cache_level, cache_tier) (
  trtllm_kv_cache_tier_bytes{state=~"unavailable|evictable"}
)
/
sum by (instance, dp_rank, cache_level, cache_tier) (trtllm_kv_cache_tier_bytes)
```

A tier average can hide an exhausted pool; retain `pool_group` when diagnosing
allocation. Zero-capacity pools have undefined ratios. Turnover/drop rates:

```promql
sum by (instance, dp_rank, cache_level, cache_tier, operation) (
  rate(trtllm_kv_cache_tier_moved_bytes_total[5m])
)
```

Sampled restore p95, grouped by actual path:

```promql
histogram_quantile(0.95,
  sum by (le, instance, source_cache_level, destination_cache_level) (
    rate(trtllm_kv_cache_tier_migration_duration_seconds_bucket{operation="restore"}[5m])
  )
)
```

Compare `_count` increases and
`rate(trtllm_kv_cache_migration_timing_skipped_total[5m])` alongside this percentile.

## Conversations

`trtllm_kv_cache_num_conversations{dp_rank="..."}` reads KVCM2's conversation-map
size in O(1). Keys are `(conversation_id, lora_task_id, cache_salt)`; IDs and salts
are never exported. Entries include in-flight context requests and retained drop
plans, and do not guarantee resident KV blocks. The gauge decreases on removal
or clearing, and is absent when retention tracking is disabled. `increase()` does
not turn it into a unique-session count.

With `enable_iter_perf_stats: true`, `ConversationManager` independently tracks
admissions over 10 minutes and 1 hour, even without retention. It observes execution
requests after waiting-queue admission, conversion and ADP placement. Output
children, dummy requests, and missing/blank IDs are excluded. Cancellation or
failure after admission does not undo it. Repeated turns refresh the last-seen
time in the same namespace; prefill and decode workers count independently.

| Metric | Type | Meaning |
| --- | --- | --- |
| `trtllm_conversation_unique_seen{window="10m/1h"}` | Gauge | Distinct tracked identities in `(now - window, now]`; lower bound while incomplete. |
| `trtllm_conversation_unique_seen_complete{window="10m/1h"}` | Gauge | Full window observed without capacity skips. |
| `trtllm_conversation_tracking_overflow_total{window="10m/1h",reason="entries/identity_bytes"}` | Counter | Skipped admission observations, not distinct missing identities. |

Actual window labels are `10m` and `1h`; worker target labels plus `dp_rank` identify
the source. Summing workers double-counts shared conversations. Dynamo's optional
log-based distinct-session panel uses trace coverage and different identity semantics.

Two ordered last-seen maps use monotonic time and share fixed limits of
100,000 window entries and 16 MiB of accounted identity storage. An identity in
both windows consumes two entries. Byte accounting charges key tuples and
referenced identity objects per entry, including shared objects; this is not a
limit on total Python allocator or worker RSS. Entry capacity also bounds
timestamp/container metadata.

At capacity, known entries still refresh and new identities are skipped.
Oversized identities are not retained.
A window remains incomplete until a full window after its latest skip; startup
also needs a full window. Cache clears preserve history; restart resets it.

Snapshots refresh at most once per second on the executor thread and ride the
existing ADP rank exchange without a new collective. Queue/batch-fill waits share
a one-second deadline while tracking is enabled; shorter transfer deadlines are
preserved. Busy execution, transport and scrape cadence add lag. Scrapes read
published snapshots. Expiry is amortized O(1) per entry, with O(entry limit) work
for a mass expiry; admission bookkeeping and idle wakeups also cost CPU.

Restrict queries to complete windows when exact counts are required:

```promql
trtllm_conversation_unique_seen{window="10m"}
  and on(instance, model_name, dp_rank, window)
    (trtllm_conversation_unique_seen_complete{window="10m"} == 1)
```
