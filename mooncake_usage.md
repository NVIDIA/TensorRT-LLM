# Using the mooncake-store KV connector

The `mooncake-store` connector publishes KV cache pages into a shared,
content-addressed pool in host DRAM, so a prefix computed by one engine can be
replayed by another. It is a KV cache *connector*, unrelated to the Mooncake
*transfer engine* that the cache transceiver can use for prefill/decode handoff:
a different component with a different config, and the two are rarely both in
play.

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
mentions `mooncake-store`, and images built from this repo have it baked in.
`mooncake_disagg/README.md` §2 explains why it is this awkward.

## 2. Configure

A `mooncake_master` process must be reachable, and every worker needs
`MOONCAKE_CONFIG_PATH` pointing at a JSON client config naming it.

Three ways to get there, in increasing order of how much you have to arrange:

| Deployment | Master |
|---|---|
| One `trtllm-serve`, own pool, **including the SLURM harness** | `mooncake_store: {launch_master: true}`, so the server starts it |
| Several engines, or a pool that outlives them | `trtllm-serve mooncake_master --address_file P`, then `mooncake_store: {master_server_address: file://P}` |
| An externally provisioned pool | Nothing in the config: an inherited `MOONCAKE_CONFIG_PATH` wins over `mooncake_store` and says so in the log |

The first two make `trtllm-serve` render the client config and export
`MOONCAKE_CONFIG_PATH` itself. Nothing outside `trtllm-serve` starts a master,
writes a JSON config or picks an HCA, the SLURM harness included: all
`disaggr_torch.slurm` does is install the bindings and tell the configs which
directory the run is in. `mooncake_disagg/README.md` §4 covers running a master
as its own SLURM job, for the second row.

One thing a scheduler-launched server does need: `TRTLLM_MOONCAKE_RUN_DIR`
pointing somewhere all its ranks can read. Provisioning happens in the server
process and reaches the ranks it spawns through the environment, but under
`trtllm-llmapi-launch` each rank is its own task and was already running, so
those ranks read the rendered config back from that directory instead. Without
it they fail during bringup naming `MOONCAKE_CONFIG_PATH`. `start_worker.sh`
sets it to the job's log directory, which is also where the master's log and
published address land.

A launched master dies with the server, so use it only for a single engine:
two context servers that each launch one get two disjoint pools, and the
survival-across-restart case is impossible by construction. Those cases want
row two, where the master is its own command and nothing else's lifetime
bounds it.

`master_server_address` takes a plain `host:port` or `file://<path>`. The file
is what a scheduler-placed master needs: its host is not known when the configs
are written, `--address_file` publishes it once the master answers, and a
server reading it waits for the master to exist. Nobody has to write an address
down, and a stale one cannot be dialed because the file is removed on exit.

Set `TRTLLM_MOONCAKE_RUN_DIR` to keep the generated JSON and the master's log,
which otherwise sit in a temporary directory that shutdown removes.
`TRTLLM_MOONCAKE_MASTER_TIMEOUT` (default 60s) bounds the wait for the port;
that wait is also what turns an unreachable external master from a failure in
every rank after the model loads into one line before it starts.

Put the connector on the **context** workers only:

```yaml
kv_connector_config:
  connector: mooncake-store
  mooncake_store:                 # omit when an orchestrator sets
    launch_master: true           # MOONCAKE_CONFIG_PATH for you
    protocol: tcp                 # rdma with a device_name for real numbers
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
| `TRTLLM_MOONCAKE_STORE_MODEL_KEY` | Defaults to the checkpoint directory's basename. Set it explicitly for anything long-lived. |

Pool capacity comes only from processes that open a store handle, so a
prefill-only connector gives a prefill-only pool, caching prefill's GPUs in
prefill's own DRAM and largely duplicating the native host offload. Ask the
generation servers to lend their memory and the pool spans both sides while
their engines stay connector-free:

```yaml
# generation worker: no connector, memory only
mooncake_donation:
  master_server_address: file:///$WORK_DIR/master.addr
  segment_size: 320GiB            # per server process, not per rank
  protocol: rdma
  device_name: mlx5_1
```

The context worker publishes the address this reads by adding
`master_address_file: $WORK_DIR/master.addr` next to its `launch_master: true`;
one is written to the run directory regardless. Startup order stops mattering,
because a server lending memory waits for the master rather than needing to
follow it.

Note the granularity: `global_segment_size` is charged per *rank*,
`segment_size` per *server process*. Two generation servers on one node lend
`segment_size` each. It is charged to the process, so it competes with that
node's own `kv_cache_config.host_cache_size`. Size the two together.

A node running no server can lend as its own command, which is also how the
pool gets memory from a machine with no GPUs at all:

```bash
trtllm-serve mooncake_donor --master_server_address file://$WORK_DIR/master.addr \
    --segment_size 160GiB --protocol rdma --device_name mlx5_0
```

## 3. Partial reuse is forced off

This is the one setting that decides whether the feature works at all.

The store is addressed by whole blocks. The connector is handed the device match
as `num_computed_tokens` and offers only blocks beyond it, but it can continue
only from a block boundary, so when the device match lands mid-block it declines
the lookup entirely. `enable_partial_reuse=true` is precisely what makes the
device match land mid-block, so it trades part of one block of device reuse for
*every* stored block of the remaining prefix. On MiniMax-M3 that declined 97.2%
of lookups, leaving a 1.6 TB pool measuring as if it were not there.

`py_executor_creator` therefore forces `enable_partial_reuse=false` whenever
this connector is configured, and says so:

```
Disabling partial reuse: it is not usable with the mooncake-store connector...
```

There is nothing to set. The warning fires even from the default of `true`, and
is the confirmation that the coercion ran. Configs that already set `false` are
unaffected.

## 4. Verify a run

Startup, at INFO, on every context worker:

```bash
grep -h "mooncake-store" <log_dir>/3_output_CTX_*.log | head -40
```

`registered layout: ... bytes/page=...` is the line to keep, since pool sizing
depends on it, and `window=None` confirms no sliding-window group (one would
have aborted startup).

Then check that the pool spans the hosts you expect.
`disaggr_torch.slurm` writes the per-segment breakdown to
`<log_dir>/9_mooncake_summary.log`; a single host means a prefill-only pool.
Pool occupancy and eviction come from the master's own log,
`$TRTLLM_MOONCAKE_RUN_DIR/mooncake_master.log`, which under the harness is
`<log_dir>/mooncake_master.log`. The startup line reports the path either way.

**Which reuse number counts store hits:** per-request stats
(`reused_blocks_per_request`, `kv_cache_hit_rate_per_request`) **do**;
`/prometheus/metrics` iteration counters (`kv_cache_iter_reused_blocks`) **do
not**, since those come from the local reuse tree. Store hits are therefore
roughly per-request reuse minus local-tree reuse. All of this needs
`enable_iter_perf_stats`, `enable_iter_req_stats` and `return_perf_metrics`,
which all default to false.

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
workload reached 35.59% actual hit at 318.14 tok/s: it wrote 1.83 TB to host and
read 32.5 GB back, behaving as a write-only tier.

Where the reuse comes from, in steady state (attribution counters, c50):
**~95% of all reuse is served by the pool and ~5% by the device cache.** The
residual 3-5% of misses are prefixes never written by anyone, which no store can
serve. Blocks stranded behind a contiguity gap measured exactly zero, as did
unattributed blocks.

**Peak is at c50, not higher.** By c70 the pool runs 85-90% full with active
eviction, hit rate falls to 86% and throughput with it. Concurrency headroom is
a function of pool size; size the pool for the working set rather than assuming
the c50 result scales.

## 6. Things that will bite

- **`host_cache_size: 0` must be written explicitly.** Left at its `None`
  default, V2 still provisions a host tier and startup is rejected. Falsy is not
  the same as absent.
- **The key namespace pins world size, rank, `tokens_per_block`, layer groups
  and `bytes_per_page`.** Change tensor parallelism and every stored page
  becomes unreachable, as a miss rather than an error.
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

Enable with `MOONCAKE_DEBUG_COVERAGE=1`, set via
`environment.ctx_worker_env_var`: `slurm.extra_args` reaches the harness rather
than the worker processes, so a flag the connector reads has to go where the
worker environment is built.
