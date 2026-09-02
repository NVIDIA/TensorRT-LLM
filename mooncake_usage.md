# Using the mooncake-store KV connector

The `mooncake-store` connector publishes KV cache pages into a shared,
content-addressed pool in host DRAM, so a prefix computed by one engine can be
replayed by another. It is a KV cache *connector*, unrelated to the Mooncake
*transfer engine* that the cache transceiver can use for prefill/decode handoff
— different component, different config, and they are usually not both in play.

Use it when local block reuse leaves reuse on the table: several context
instances that see the same prefixes, prefixes that should outlive a restart, or
a working set larger than one node's host memory. It replaces TensorRT-LLM's
native host offload tier rather than layering on top of it (`host_cache_size`
must be `0`).

This page is the entry point. Depth lives elsewhere:

| For | Read |
|---|---|
| API surface, gates, keying | `docs/source/features/kv-cache-connector.md` § *Mooncake distributed store* |
| Full SLURM runbook, install debugging, experiment matrix | `mooncake_disagg/README.md` |
| Working configs | `mooncake_disagg/m3_ctx_mooncake.yaml`, `m3_gen_mooncake.yaml` |
| Correctness tests (no GPU, no store) | `tests/unittest/_torch/executor/test_mooncake_store_connector.py` |

## 1. Install

The connector needs the Mooncake **Python** bindings
(`mooncake.store.MooncakeDistributedStore`). Containers have shipped the C++
library for months without them, and a CMake-installed `mooncake` package
shadows the working wheel, so a bare `pip install` reports success and the
import still fails. Inside the container:

```bash
bash mooncake_disagg/install_mooncake_runtime.sh   # idempotent, ~8s warm
python3 -c "from mooncake.store import MooncakeDistributedStore; print('ok')"
```

`disaggr_torch.slurm` runs this per node automatically when a worker config
mentions `mooncake-store`. Images built from this branch have it baked in.
`mooncake_disagg/README.md` §2 explains why it is this awkward.

## 2. Configure

A `mooncake_master` process must be reachable, and every worker needs
`MOONCAKE_CONFIG_PATH` pointing at a JSON client config naming it. The SLURM
harness starts the master and writes that JSON per job; outside SLURM, see
§4 of the runbook.

Put the connector on the **context** workers only:

```yaml
kv_connector_config:
  connector: mooncake-store
kv_cache_config:
  use_kv_cache_manager_v2: true   # required: only V2 describes its pools
  enable_block_reuse: true
  host_cache_size: 0              # required, and must be explicit, not omitted
  disk_cache_size: 0
  tokens_per_block: 128
scheduler_config:
  capacity_scheduler_policy: GUARANTEED_NO_EVICT   # required
enable_attention_dp: false        # required
```

Generation workers deliberately get no `kv_connector_config`: generated tokens
are rarely a reused prefix, and leaving it off is the only way to say "off"
(`StoreRole` has no off value). It also lets them keep their host cache tier and
`MAX_UTILIZATION` scheduler, both of which the connector forbids.

Per-process environment, on the workers that open a handle:

| Variable | Purpose |
|---|---|
| `TRTLLM_MOONCAKE_STORE_ROLE` | `producer` / `consumer` / `both` |
| `TRTLLM_MOONCAKE_STORE_PREFIX` | Cache namespace. Bump it after any change to page layout or contents. |
| `TRTLLM_MOONCAKE_STORE_MODEL_KEY` | Defaults to the checkpoint directory's basename — set it explicitly for anything long-lived. |

Pool capacity comes only from processes that open a store handle, so a
prefill-only connector gives a prefill-only pool. `mooncake_segment_donor.py`
contributes host memory from the generation nodes without any traffic;
`disaggr_torch.slurm` starts one per generation node
(`MOONCAKE_DONOR_SEGMENT_SIZE`, default `32GiB`).

## 3. Partial reuse must be off — now enforced

This is the one setting that decides whether the feature works at all.

The store is addressed by whole blocks. The connector is handed
`num_computed_tokens`, the device match, and offers only blocks beyond it — but
it can only continue from a block boundary, so when the device match lands
mid-block it declines the lookup entirely. `enable_partial_reuse=true` is
precisely what makes the device match land mid-block, so it trades part of one
block of device reuse for *every* stored block of the remaining prefix.

On MiniMax-M3 that guard declined **97.2% of lookups**. The pool was never
asked, and a 1.6 TB pool measured as if it were not there.

`py_executor_creator` now forces `enable_partial_reuse=false` whenever this
connector is configured, and says so:

```
Disabling partial reuse: it is not usable with the mooncake-store connector...
```

Nothing to set; the warning fires even from the default (`true`), and is the
confirmation that the coercion ran. The field is otherwise untouched, so
configs that already set `false` are unaffected.

## 4. Verify a run

Startup, at INFO, on every context worker:

```bash
grep -h "mooncake-store" <log_dir>/3_output_CTX_*.log | head -40
```

`registered layout: ... bytes/page=...` is the line to keep — pool sizing
depends on it, and `window=None` confirms no sliding-window group (one would
have aborted startup).

Then check that the pool spans the hosts you expect.
`disaggr_torch.slurm` writes the per-segment breakdown to
`<log_dir>/9_mooncake_summary.log`; a single host means a prefill-only pool.
Pool occupancy and eviction come from `<log_dir>/2_mooncake_master.log`.

**Which reuse number counts store hits:** per-request stats
(`reused_blocks_per_request`, `kv_cache_hit_rate_per_request`) **do**;
`/prometheus/metrics` iteration counters (`kv_cache_iter_reused_blocks`) **do
not** — those come from the local reuse tree. So store hits ≈ per-request reuse
− local-tree reuse. All of this needs `enable_iter_perf_stats`,
`enable_iter_req_stats` and `return_perf_metrics`, which all default to false.

## 5. What it measured

MiniMax-M3-NVFP4 on GB300, 1 context server (TP=2) + 2 generation servers
(TP=4), connector on context only, ~1.6 TB pool (160 GiB per context rank plus
640 GiB donated per generation node), real conversation trace.

"Baseline" is the same configuration with partial reuse left at its default.

| Run | Theoretical hit | Actual hit | Output tok/s |
|---|---|---|---|
| c50 baseline | 96.29% | 35.19% | 319.43 |
| **c50 with partial reuse off** | 96.64% | **93.53%** | **697.29** (2.18x) |
| c70 baseline | 96.01% | 38.44% | 397.43 |
| **c70 with partial reuse off** | 96.51% | **86.46%** | **643.57** (1.62x) |

A 61-point gap between the reuse the workload allowed and the reuse the system
achieved closed to 3 points. For comparison, native host offload on the same
workload reached 35.59% actual hit at 318.14 tok/s — it wrote 1.83 TB to host
and read 32.5 GB back, behaving as a write-only tier.

Where the reuse comes from, in steady state (attribution counters, c50):
**~95% of all reuse is served by the pool and ~5% by the device cache.** The
residual ~3–5% of misses are prefixes never written by anyone, which no store
can serve. Blocks stranded behind a contiguity gap measured exactly zero, as did
unattributed blocks.

**Peak is at c50, not higher.** By c70 the pool runs 85–90% full with active
eviction, hit rate falls to 86% and throughput with it. Concurrency headroom is
a function of pool size; size the pool for the working set rather than assuming
the c50 result scales.

## 6. Things that will bite

- **`host_cache_size: 0` must be written explicitly.** Left at its `None`
  default, V2 still provisions a host tier and startup is rejected. Falsy is not
  the same as absent.
- **The key namespace pins world size, rank, `tokens_per_block`, layer groups
  and `bytes_per_page`.** Change tensor parallelism and every stored page
  becomes unreachable — a miss, not an error.
- **No build hash in the key.** After changing page layout or contents, bump
  `TRTLLM_MOONCAKE_STORE_PREFIX` or restart the master.
- **Loads are synchronous** (`start_load_kv`, before the forward pass), so every
  loaded byte is exposed to TTFT. A store hit wins only when it displaces real
  prefill.
- **`mooncake-store failed to load N of M pages` is not flaky.** The runtime had
  already counted those tokens as computed; it is the tripwire against a wrong
  answer. Stop and investigate.
- **Rejected outright at startup:** pipeline or context parallelism,
  sliding-window attention, attention DP, beam search, host/disk cache tiers,
  `MAX_UTILIZATION`, and M3's index-V cache unless
  `sparse_disable_index_value=true`. `mooncake_disagg/README.md` §9 has the rest.

## 7. Diagnostics

The `user/brb/m3-mooncake-store-instrumentation` branch carries attribution
counters that sort every reusable block of every prompt into served-from-device,
served-from-pool, stranded behind a contiguity gap, never written, evicted, or
unattributed, with cumulative and per-window reporting. That is what produced
§5's split. They are kept off this branch because they cost a store probe on
lookups the connector would otherwise decline.

Enable with `MOONCAKE_DEBUG_COVERAGE=1`, which must be set via
`environment.ctx_worker_env_var` — the SLURM harness consumes `MOONCAKE_*`
itself to build the pool config and does not forward it to workers.
