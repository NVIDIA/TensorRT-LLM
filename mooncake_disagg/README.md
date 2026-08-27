# Validating the mooncake-store KV connector on MiniMax-M3

This is a runbook for testing the `mooncake-store` KV cache connector
(commits `f3c092187e`..`b8d3f43c43`) on MiniMax-M3 under load, using the SLURM
disaggregated benchmark harness in
`examples/disaggregated/slurm/benchmark/`.

The unit tests in `tests/unittest/_torch/executor/test_mooncake_store_connector.py`
cover the pieces that decide whether a cache hit is *correct* -- key namespacing,
hash chaining, page addressing, the startup gates. They deliberately do not run
a store, a model, or two engines. What is untested is everything that decides
whether the feature is *worth having*: whether a real prefix survives the round
trip, whether one M3 instance can replay a prefix another computed, and whether
the synchronous load path costs less than the prefill it avoids.

## 1. The claim under test

Local block reuse never leaves the instance that computed the prefix. The store
publishes KV pages into a shared, content-addressed CPU pool so any engine can
replay them. So there are exactly three things the store can do that local reuse
cannot, and each gets its own experiment in §7:

1. **Cross-instance reuse.** A request routed to context instance B replays a
   prefix computed on instance A.
2. **Survival across restarts.** Pages outlive the process that wrote them.
3. **Pool capacity beyond one host.** The pool is the sum of every worker's
   segment rather than one node's host memory.

### Why M3 makes this a hard test rather than an easy one

The residency measurements in `../m3-kv-residency-measurement-README.md` (taken
on this same model and workload) found that M3 production traffic already serves
**97.0% of prompt tokens from local cache**, with eviction responsible for under
0.7% of misses. On a *single* instance there is almost no headroom for the store
to recover -- over 99% of misses are prefixes never cached anywhere, which no
store can serve either.

That is not an argument against the feature; it is an argument about where to
look. Set expectations accordingly:

- Do not expect a single-instance hit-rate improvement. Expect roughly zero.
- The store's value on M3 is concentrated in the cross-instance and
  post-restart cases, where local reuse scores zero by construction.
- The connector's loads are **synchronous** (`start_load_kv`, before the forward
  pass), so every loaded byte is fully exposed to TTFT. The host-offload tier it
  replaces achieved 38-43% overlap at 45-51 GiB/s. At M3's context lengths a
  loaded prefix is gigabytes, so a store hit is a win only when it displaces
  real prefill, and a *needless* store hit is pure added latency.
- The design already rules out the worst version of that: the leader is handed
  `num_computed_tokens` (the local match) and offers only blocks *beyond* it, so
  the store cannot re-fetch something the GPU already holds. It cannot regress a
  local hit; it can only add latency on a genuine local miss that it then fails
  to make cheaper. That is what experiment 4 measures.

## 2. Prerequisites: is Mooncake actually installed?

**Short answer: probably not the part this connector needs.** Check before you
burn an allocation.

Two different Mooncake components exist, and TensorRT-LLM's history treats them
differently:

| Component | What uses it | How it gets installed | Since |
|---|---|---|---|
| C++ transfer engine (`/usr/local/Mooncake`) | the C++ cache transceiver's Mooncake backend | CMake source build in `docker/common/install_mooncake.sh` | PR #8447, Nov 2025 |
| Python bindings (`mooncake.store.MooncakeDistributedStore`) | **this connector** | pip wheel, added to the same script | commit `d36dae435e`, **this branch** |

So Mooncake has been in the container images for months, but only usefully as
the C++ library. Three consequences:

- Any image built before commit `d36dae435e` lacks a working set of bindings.
  The image pinned in `jenkins/current_image_tags.properties` is tagged
  `202607211045` (2026-07-21), which predates that commit, so **the currently
  pinned CI image does not have them**.
- The CMake install *does* drop a `mooncake` Python package into the image, but
  it is both broken and actively harmful: it shadows the working one. This is the
  single biggest time sink in this section; see "Fix" below.
- The wheel is also the only usable source of the `mooncake_master` and
  `mooncake_http_metadata_server` entry points. The CMake build's
  `/usr/local/Mooncake/bin/mooncake_master` does exist and does run, but it is
  the 0.3.7 build and it is not what ends up on `PATH` once the wheel is
  installed.

Also note `install_mooncake.sh` runs only in the `tritondevel` stage of
`docker/Dockerfile.multi`, and is skipped entirely on Rocky8. The CI image and
the internal `trtllm_build` release image descend from `tritondevel`, so they
get it; the NGC `release` image descends from the plain `devel` stage, so it
does not.

### Verify

Inside the container you will actually run:

```bash
python3 -c "from mooncake.store import MooncakeDistributedStore; print('store bindings OK')"
command -v mooncake_master || ls /usr/local/Mooncake/bin
```

### Fix

Run `mooncake_disagg/install_mooncake_runtime.sh` inside the container. It takes
under ten seconds on a warm pip cache, it is idempotent, and it verifies itself,
so it is safe in a SLURM prolog on every node.

```bash
bash mooncake_disagg/install_mooncake_runtime.sh
```

A bare `pip3 install mooncake-transfer-engine` is *not* enough, and its failure
mode is what makes this step so confusing: pip reports success and the import
still fails. Two independent problems, both of which the script handles.

**1. A broken `mooncake` package that pip cannot displace.** The CMake build in
`install_mooncake.sh` emits its own `mooncake` Python package, omitting
`libmooncake_store.so`, so it cannot load. `mooncake-integration/CMakeLists.txt`
chooses where to put it with:

```cmake
COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print([s for s in sys.path if 'packages' in s][0])"
```

-- the *first* `sys.path` entry whose name merely contains `"packages"`. That
gives two different failures depending on what else is installed, and both
produce the same confusing symptom: an `ImportError` **after a `pip install`
that reported success**.

- **With `nvidia-cutlass-dsl` present** (the normal case: the `devel` stage
  uninstalls it at `Dockerfile.multi:58`, then `constraints.txt` pulls it back
  in) the first match is `nvidia_cutlass_dsl/dsl_packages`, because
  `nvidia_cutlass_dsl_packages.pth` does `sys.path.insert(0, ...)` on it. It
  therefore outranks `dist-packages` on every interpreter start and shadows the
  wheel permanently. CUTLASS DSL never references `mooncake`, so deleting it
  breaks nothing.
- **Without it**, the match is `dist-packages` itself and the broken package
  *collides* with the wheel. This one is nastier: CMake writes
  `store.cpython-312-<plat>.so` while the wheel writes `store.so`, and
  `importlib.machinery.EXTENSION_SUFFIXES` puts the interpreter-tagged suffix
  first, so the broken extension still wins. pip also overwrites `__init__.py`,
  erasing the `# Auto-generated by CMake` marker, so afterwards there is no
  reliable way to tell leftover files from wheel files.

Because of that second case, `install_mooncake_runtime.sh` removes any
`mooncake` package directory outright and reinstalls, rather than trying to
identify individual bad files. Both cases are covered.

**2. The default wheel is built for CUDA 12.** `mooncake-transfer-engine` links
against `libcudart.so.12`; containers from `pytorch-26.05` on ship CUDA 13 only.
Every extension and the `mooncake_master` binary then fail to load:

```
ImportError: libcudart.so.12: cannot open shared object file
```

Use **`mooncake-transfer-engine-cuda13`**, the same project built for CUDA 13,
which needs no shim. Its releases start at 0.3.9, so it cannot match the
`MOONCAKE_VERSION` pin (`0.3.7.post2`) in `install_mooncake.sh`. That drift is
safe here: `/usr/local/Mooncake` backs the *cache transceiver's* Mooncake
backend, a different feature that these configs do not use
(`cache_transceiver_config.backend: "NIXL"`). The connector only ever talks to
the wheel, and the wheel also supplies the `mooncake_master` that lands on
`PATH` ahead of the CMake one, so client and master stay matched. Revisit this
only if you set the transceiver backend to `MOONCAKE`.

If you would rather match `install_mooncake.sh` exactly, the script keeps that
path working and applies the `libcudart.so.12` shim for you:

```bash
MOONCAKE_WHEEL="mooncake-transfer-engine==0.3.7.post2" \
  bash mooncake_disagg/install_mooncake_runtime.sh
```

Both wheel choices were validated against the full set of store methods the
connector calls -- see "Validating the install" below.

### How often does the script need to run?

It depends on whether the container filesystem persists, because the script
writes into `dist-packages` inside the container, not into your checkout.

| Situation | How often |
|---|---|
| Long-lived container you `docker exec` into | **Once.** It survives until the container is deleted; `docker restart` keeps it. |
| SLURM via `disaggr_torch.slurm` | **Once per job, per node** -- and the harness now does it for you, see below. |
| Image built from this branch | **Never.** `install_mooncake.sh` now does it at build time and fails the build if the import does not work. |

For the SLURM case this is already wired up: `disaggr_torch.slurm` now runs the
script on every node, right after its `pip install -e .[devel]` step, and gates
it on whether a worker config actually asks for the connector:

```bash
if grep -qs "mooncake-store" "${full_logdir}/ctx_config.yaml" "${full_logdir}/gen_config.yaml"; then
```

So arms A and B of the run matrix pay nothing, arm C installs automatically, and
there is no new config key to remember. It resolves the script as
`${trtllm_repo}/mooncake_disagg/install_mooncake_runtime.sh` and fails the job
with an explicit message if `environment.trtllm_repo` is unset or does not
contain it -- which is the case if you benchmark from
`environment.trtllm_wheel_path` instead, so use an image with the bindings baked
in for that path. Output lands in `<log_dir>/2_install_mooncake.log`. Set
`MOONCAKE_WHEEL` in the submitting environment to override the wheel; it is
forwarded to every node.

Note that `--container-name` gives each node a container that lives for the whole
job, so the install survives from that step through to the worker `srun`s. It
does not survive into the *next* job, which is why this runs per job rather than
once. Do not try to persist it via `~/.local` unless home is genuinely shared and
mounted (`disaggr_torch.slurm` passes `--no-container-mount-home` to most of its
`srun` calls).

Baking an image is the only option that removes the step entirely. Given the
script takes about eight seconds from cold, that is a convenience decision rather
than a necessity.

### Will the shadow package come back?

The root cause is upstream in Mooncake's `CMakeLists.txt` and is **not** fixed;
both scripts clean up after it. Practically:

- **Images built from this branch:** no. The cleanup runs in the same script,
  immediately after `make install` and before the wheel install, and the build
  now fails if `import mooncake.store` does not work.
- **Any pre-existing image**, including the one pinned in
  `jenkins/current_image_tags.properties` (`202607211045`): the broken package is
  baked in, so the runtime script is required.
- **Inside a running container:** only if something re-runs Mooncake's CMake
  install. Reinstalling `nvidia-cutlass-dsl` does not recreate it -- that package
  has never shipped a `mooncake` directory; it only supplies the `.pth` that made
  CMake choose the wrong destination.
- **If Mooncake is ever upgraded** to a version that fixes its install path, or
  the `.pth` ordering changes, the cleanup becomes a no-op rather than a hazard.

### Validating the install

Two scripts in this directory, in increasing order of strictness. Both need a
running master and `MOONCAKE_CONFIG_PATH`, exactly like a real worker:

```bash
mooncake_master --rpc_port=50051 --metrics_port=9004 &

export MOONCAKE_CONFIG_PATH=$PWD/mooncake.json   # TCP config; edit the master address
python3 mooncake_disagg/mooncake_smoke_test.py        # setup + put/get round trip
python3 mooncake_disagg/mooncake_api_surface_test.py  # needs a GPU
```

`mooncake_smoke_test.py` proves the bindings load and `store.setup()` succeeds
with the same argument list `worker.py` passes. `mooncake_api_surface_test.py` is
the one that matters when changing wheel versions: the connector's hot path never
uses `put`/`get`, it uses `register_buffer` plus the zero-copy
`batch_put_from_multi_buffers` / `batch_get_into_multi_buffers` / `batch_is_exist`
calls against registered GPU pages. Those take `list[list[int]]` -- one buffer
list per key, because `PageAddressing.page_buffers` returns one address per
layer-group region -- and that is the signature most likely to drift.

Then the unit tests, which need no store and no GPU:

```bash
pytest tests/unittest/_torch/executor/test_mooncake_store_connector.py
```

## 3. Topology

```
                       mooncake_master (1 CPU core, its own job)
                              ^         ^                    ^
              register/put/get|         |                    |mount segment only
        ┌─────────────────────┴──┐   ┌──┴──────────────────────┐   │
        │  CTX instance 0  TP=4  │   │  CTX instance 1  TP=4   │   │ store: role=both
        └────────────┬───────────┘   └───────────┬─────────────┘   │
                     │      NIXL KV handoff      │                 │
                     └──────────┬────────────────┘                 │
                                v                                  │
                     ┌──────────────────────────┐   ┌──────────────┴────────────┐
                     │   GEN instance   TP=4    │   │ segment donor (same node) │
                     └──────────────────────────┘   └───────────────────────────┘
                                ^                     no connector, no put/get,
                     round-robin│                     contributes host memory
                     ┌──────────┴───────────┐
                     │  trtllm-serve disagg │  <- benchmark_serving client
                     └──────────────────────┘
```

12 GPUs = 3 nodes at 4 GPUs/node. The generation worker deliberately has **no**
`kv_connector_config`: generated tokens are rarely a reused prefix, and that
absence is the only way to express "off" (`StoreRole` has no off value). It also
lets the generation worker keep its host cache tier and `MAX_UTILIZATION`
scheduler, both of which the connector would forbid.

### Why the generation node still needs a donor process

Pool capacity comes only from processes that open a store handle: `setup`
registers `global_segment_size` bytes of the calling process's host memory, and
the master then places blocks in it. Since only the context workers configure
the connector, only they contribute memory — so by default every byte of the
pool is prefill-node DRAM, and the store is a prefill-DRAM-caches-prefill-GPU
tier that largely duplicates TensorRT-LLM's native host offload. Confirm this on
any run by grouping the master's `allocation_succeeded ... segment=<ip>:<port>`
lines by host: a single host means a prefill-only pool.

`mooncake_segment_donor.py` closes that gap. One donor per generation node opens
a handle, contributes memory, and then idles forever without a single put or get,
so the pool spans both sides while the generation engine stays connector-free
and keeps its cache transceiver for the KV handoff. A donor is deliberately not
a `StoreRole`: the roles describe traffic (`producer` writes, `consumer` reads,
`both`), and none of them means "contribute memory only", so capacity and
traffic have to be separate processes.

`disaggr_torch.slurm` starts the donors automatically, reading the generation
nodes off the generated worker commands and waiting for each segment to mount
before any worker starts — so the first blocks prefill writes can already land
on a decode node. Tune with `MOONCAKE_DONOR_SEGMENT_SIZE` (default `32GiB`, set
`0` to keep the pool prefill-only) and `MOONCAKE_DONOR_NODES` to override node
selection.

The donated memory is charged to the donor process and competes with the
generation worker's own `kv_cache_config.host_cache_size` on that node, so size
the two together. The worker logs its own share as `KV cache manager v2 host
cache quota set to N GiB`, **per rank**, against the `available host memory` it
reports on the same line.

## 4. Step 1 -- run the Mooncake master

`master_server_address` is mandatory, so a master must exist and be reachable
from every worker.

**For a single-job experiment you can skip this section.**
`disaggr_torch.slurm` now starts a `mooncake_master` on the first node of the
allocation, waits for its port to accept connections, and writes
`<log_dir>/mooncake.json` naming it; `start_worker.sh` then resolves
`MOONCAKE_CONFIG_PATH` from the log directory, which is why the harness config
does not set it. Defaults are the bring-up ones (TCP, 16GiB per worker);
`MOONCAKE_PROTOCOL`, `MOONCAKE_DEVICE_NAME`, `MOONCAKE_GLOBAL_SEGMENT_SIZE` and
`MOONCAKE_LOCAL_BUFFER_SIZE` in the submitting environment override them, and
the master's own log lands in `<log_dir>/2_mooncake_master.log`.

That master dies with the job, so read on if you need a pool that outlives one
allocation -- which experiment 3 does, by construction. Run it as its own
long-lived job and export `MOONCAKE_MASTER_ADDRESS=<host>:50051` before
`submit.py`; the harness then skips launching one and only writes the client
config pointing at yours.

```bash
# mooncake_master.sbatch
#!/bin/bash
#SBATCH --job-name=mooncake-master
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

srun --container-image=$CONTAINER_IMAGE \
     --container-mounts=$WORK_DIR:$WORK_DIR \
     bash -lc '
       hostname -i | awk "{print \$1}" > '"$WORK_DIR"'/master.addr
       exec mooncake_master \
         --rpc_port=50051 \
         --metrics_port=9004 \
         --eviction_ratio=0.05
     '
```

Flag names above were read out of the shipped `mooncake_master` binary. Run
`mooncake_master --help` inside the container to confirm defaults and to see the
rest (`--rpc_address`, `--rpc_thread_num`, `--default_kv_lease_ttl`,
`--eviction_high_watermark_ratio`, `--enable_http_metadata_server`,
`--cluster_id`, `--root_fs_dir`).

Keeping the master in a separate job is what makes experiment 3 (§7) possible:
the pool outlives the engines, so a second benchmark job finds a warm store.

Then write the client config, substituting the address the master job just
recorded -- or let `disaggr_torch.slurm` generate it, as above. The schema is
vLLM's, so one pool can serve both engines:

```bash
MASTER_IP=$(cat $WORK_DIR/master.addr)
cat > $WORK_DIR/mooncake.json <<EOF
{
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "${MASTER_IP}:50051",
  "protocol": "rdma",
  "device_name": "mlx5_0",
  "global_segment_size": "64GiB",
  "local_buffer_size": "4GiB",
  "role": "both",
  "cache_prefix": "trtllm-m3",
  "transfer_batch_size": 64
}
EOF
```

- `metadata_server`: `P2PHANDSHAKE` keeps a separate metadata service out of the
  picture entirely, which is one less process to place and one less port to get
  right. Use `http://<host>:<port>/metadata` with
  `mooncake_http_metadata_server` (or the master's own
  `--enable_http_metadata_server`) only if you need the shared-metadata
  behaviour; confirm the port from `--help` rather than assuming.
- `device_name`: pick from `ibv_devinfo` on a compute node. For first bring-up
  only, `"protocol": "tcp"` with `"device_name": ""` removes RDMA from the
  variable list -- that is what `mooncake.json` in this directory currently
  does. Do not draw performance conclusions from a TCP run.
- `global_segment_size` is contributed **per worker process**, so the pool is
  `global_segment_size x (ctx instances x TP)` = 8 segments here.
- Sizing: after startup, each worker logs its page geometry (§8). Pool bytes for
  a corpus of `T` unique prefix tokens is
  `T / tokens_per_block x Σ_layer_groups bytes_per_page x world_size`.
  As an anchor, the residency work measured M3 at ~22 KiB/token aggregate across
  TP=4 (fp8 KV plus a per-rank-replicated index-K), so ~21 GiB per million
  unique prefix tokens. Confirm against your own log line rather than trusting
  that number.
- `role`/`cache_prefix` can also be overridden per process by
  `TRTLLM_MOONCAKE_STORE_ROLE` and `TRTLLM_MOONCAKE_STORE_PREFIX`. Bump the
  prefix whenever you change anything that should not be shared with an earlier
  run's pages.

## 5. Step 2 -- the harness config

Copy `examples/disaggregated/slurm/benchmark/config.yaml` and replace the
`worker_config` section with M3's. `submit.py` serializes `worker_config.ctx`
and `worker_config.gen` straight to `ctx_config.yaml`/`gen_config.yaml` with
`yaml.dump`, so any LLM-API key passes through untouched -- including
`kv_connector_config`.

The context worker below is `m3_ctx_mooncake.yaml` from this directory; the
generation worker is `m3_gen_mooncake.yaml`. Every deviation from the production
M3 config is marked and explained there, and those comments are the reason to
read those two files rather than treating this block as self-explanatory.

```yaml
# m3_store_2ctx.yaml
slurm:
  script_file: "disaggr_torch.slurm"
  partition: "<partition>"
  account: "<account>"
  job_time: "04:00:00"
  job_name: "m3-mooncake-store"
  extra_args: ""
  set_segment: true
  numa_bind: true          # GB200/GB300 NVL72

benchmark:
  mode: "e2e"
  use_nv_sa_benchmark: false
  multi_round: 8           # num_prompts = concurrency x multi_round
  streaming: true
  concurrency_list: "8"
  input_length: 131072     # log-dir naming only; the dataset is authoritative
  output_length: 1024
  dataset_file: "<work_dir>/m3_shared_prefix.jsonl"

hardware:
  gpus_per_node: 4
  num_ctx_servers: 2       # >= 2 is the whole point; see experiment 2
  num_gen_servers: 1

environment:
  container_mount: "<mounts>"
  container_image: "<image with the mooncake wheel -- see section 2>"
  model_path: "<path to MiniMax-M3-NVFP4>"
  trtllm_repo: "<this checkout>"
  build_wheel: false
  trtllm_wheel_path: ""
  work_dir: "<work_dir>"
  worker_env_var: "TLLM_LOG_LEVEL=INFO TRTLLM_SERVER_DISABLE_GC=1 TRTLLM_WORKER_DISABLE_GC=1 TRTLLM_ENABLE_PDL=1 ENROOT_ALLOW_DEV=yes NCCL_GRAPH_MIXING_SUPPORT=0"
  # Only the context workers open a store handle. MOONCAKE_CONFIG_PATH is
  # deliberately absent: the harness generates the file per job in the log
  # directory, whose path is not known when submit.py builds this environment.
  ctx_worker_env_var: "TRTLLM_MOONCAKE_STORE_ROLE=both TRTLLM_MOONCAKE_STORE_PREFIX=trtllm-m3-run1"
  server_env_var: "TRTLLM_SERVER_DISABLE_GC=1"

profiling:
  nsys_on: false
  ctx_profile_range: "10-30"
  gen_profile_range: "200-250"

accuracy:
  enable_accuracy_test: false
  tasks: {}

worker_config:
  ctx:
    # ---- contents of m3_ctx_mooncake.yaml, plus parallelism ----
    tensor_parallel_size: 4
    moe_expert_parallel_size: 4
    pipeline_parallel_size: 1     # gated: connector refuses PP > 1
    context_parallel_size: 1      # gated: connector refuses CP > 1
    enable_attention_dp: false    # required: dummy DP-balancing requests reach the hooks
    max_seq_len: 1048576
    max_num_tokens: 16384
    max_batch_size: 20
    sparse_attention_config:
      algorithm: minimax_m3
      implementation: msa
      indexer_kv_dtype: fp8
      sparse_disable_index_value: true   # gated: index-V lives outside the paged pools
      fuse_qkv_index_projection: true
    kv_cache_config:
      free_gpu_memory_fraction: 0.94
      enable_block_reuse: true
      tokens_per_block: 128
      use_kv_cache_manager_v2: true      # required: only V2 can describe its pools
      dtype: fp8
      event_buffer_max_size: 0
      host_cache_size: 0                 # gated: must be explicit 0, not omitted
      disk_cache_size: 0
    scheduler_config:
      capacity_scheduler_policy: GUARANTEED_NO_EVICT   # gated
    cache_transceiver_config:
      backend: "NIXL"
      transceiver_runtime: "PYTHON"      # M3 is always-V2; C++ transceiver is refused
    enable_chunked_prefill: true
    enable_autotuner: true
    trust_remote_code: true
    reasoning_parser: minimax_m3
    stream_interval: 20
    print_iter_log: true
    num_postprocess_workers: 8
    # Required to see any reuse number at all -- see section 8. All three
    # default to false, and without them /metrics returns an empty list.
    enable_iter_perf_stats: true
    enable_iter_req_stats: true
    return_perf_metrics: true
    kv_connector_config:
      connector: mooncake-store         # <-- the only line experiment 1 removes

  gen:
    # ---- contents of m3_gen_mooncake.yaml; no connector, so no gates ----
    tensor_parallel_size: 4
    moe_expert_parallel_size: 4
    enable_attention_dp: false
    max_seq_len: 1048576
    max_num_tokens: 16384
    max_batch_size: 20
    sparse_attention_config:
      algorithm: minimax_m3
      implementation: msa
      indexer_kv_dtype: fp8
      sparse_disable_index_value: true   # must match ctx: it changes the model
      fuse_qkv_index_projection: true
    kv_cache_config:
      free_gpu_memory_fraction: 0.94
      enable_block_reuse: true
      block_reuse_policy: per_conversation
      tokens_per_block: 128
      use_kv_cache_manager_v2: true
      dtype: fp8
      event_buffer_max_size: 0
      host_cache_size: 388554555392      # kept: no connector here
    scheduler_config:
      capacity_scheduler_policy: MAX_UTILIZATION   # kept: no connector here
    cache_transceiver_config:
      backend: "NIXL"
      transceiver_runtime: "PYTHON"      # must match ctx
    enable_chunked_prefill: true
    enable_autotuner: true
    trust_remote_code: true
    reasoning_parser: minimax_m3
    stream_interval: 20
    print_iter_log: true
    num_postprocess_workers: 8
    enable_iter_perf_stats: true
    enable_iter_req_stats: true
    return_perf_metrics: true
```

Eagle3 is left off. It is not gated, but `MiniMaxM3KVCacheManagerV2` sets
`supports_shared_draft_layers`, so draft layers join the unified V2 cache and
therefore the registered layout -- extra page geometry the store must key
correctly, on a path with no coverage. Turn it on only after a clean run
without it.

Submit with:

```bash
cd examples/disaggregated/slurm/benchmark
python3 submit.py -c <work_dir>/m3_store_2ctx.yaml --dry-run   # inspect first
python3 submit.py -c <work_dir>/m3_store_2ctx.yaml
```

## 6. Step 3 -- the workload

`run_benchmark.sh` invokes `benchmark_serving` with
`--dataset-name trtllm_custom --dataset-path <dataset_file>`, so you supply a
JSONL file. `CustomDataset` reads `input.messages[1].content` as the prompt,
`input.max_tokens` as the output length, and skips re-tokenization when
`input.num_tokens` is present. It shuffles the file on load, which is what
spreads repeated prefixes apart in time.

The workload must have **repeated prefixes across requests**, because that is
the only structure a content-addressed store can exploit. `P` distinct prefixes
each repeated `R` times, with a unique suffix per request so no two requests are
identical:

```python
# gen_shared_prefix_dataset.py
import json, random
from transformers import AutoTokenizer

MODEL = "<path to MiniMax-M3-NVFP4>"
NUM_PREFIXES = 8        # P distinct shared prefixes
REPEATS = 8             # R requests per prefix -> P*R = 64 total
PREFIX_TOKENS = 131072  # must be >> tokens_per_block (128) to be worth storing
SUFFIX_TOKENS = 512
OUTPUT_TOKENS = 1024
OUT = "m3_shared_prefix.jsonl"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
rng = random.Random(1234)
vocab = tok.vocab_size

def text_of(n_tokens, seed):
    r = random.Random(seed)
    ids = [r.randrange(1000, vocab - 1000) for _ in range(int(n_tokens * 1.3))]
    text = tok.decode(ids, skip_special_tokens=True)
    # Re-encode and trim: decode/encode is not a round trip, so measure.
    ids = tok.encode(text, add_special_tokens=False)[:n_tokens]
    return tok.decode(ids, skip_special_tokens=True), len(ids)

prefixes = [text_of(PREFIX_TOKENS, 100 + i) for i in range(NUM_PREFIXES)]

with open(OUT, "w") as f:
    for i, (prefix, plen) in enumerate(prefixes):
        for r in range(REPEATS):
            suffix, slen = text_of(SUFFIX_TOKENS, 900000 + i * 1000 + r)
            f.write(json.dumps({"input": {
                "messages": [{"role": "system", "content": ""},
                             {"role": "user", "content": prefix + suffix}],
                "max_tokens": OUTPUT_TOKENS,
                "num_tokens": plen + slen,
            }}) + "\n")
```

Size the file against what the client will actually request.
`run_benchmark.sh` computes
`num_prompts = (concurrency x num_gen_servers) x multi_round`, so the §5 config
(`concurrency_list: "8"`, `multi_round: 8`, one generation server) asks for 64
prompts -- which is why `P x R` above is 64. Ask for more than the file holds
and the extra is not sampled; write more than you ask for and the tail of your
repeat structure never runs.

Random token text is deliberate: it defeats any accidental prefix sharing
between "distinct" prefixes, so the hit rate you measure is the one you
designed. If you would rather test genuine production traffic, substitute a
real multi-turn trace -- but keep an eye on whether it actually contains
repeated prefixes, since without them the store has nothing to do and a flat
result means nothing.

## 7. Step 4 -- the run matrix

Three arms, and the middle one is the one people skip:

| Arm | Config | Purpose |
|---|---|---|
| **A** production reference | today's M3 config: host tier on, `MAX_UTILIZATION`, no connector | where you are today |
| **B** gated baseline | arm A's config edited to satisfy every connector gate (`host_cache_size: 0`, `GUARANTEED_NO_EVICT`, ...), still no connector | isolates what the gates cost |
| **C** store | arm B plus `kv_connector_config` | isolates what the store adds |

Comparing C against A alone conflates two independent changes: the store's
benefit and the loss of a 362 GiB host cache tier plus a scheduler policy
change. **The store's efficacy is C vs B.** A vs B tells you the entry price,
and A vs C tells you whether the whole package is deployable. All three are
worth knowing and they answer different questions.

Then, within that:

**Experiment 1 -- does it work at all (`num_ctx_servers: 1`).**
Arm C, one context instance, small `PREFIX_TOKENS` (say 4096) and a short run.
You are looking for a clean startup, the registration log line, non-zero store
traffic, no load failures, and coherent output text. Do this before spending an
allocation on anything larger. Expect no throughput change; local reuse already
serves this case.

**Experiment 2 -- cross-instance reuse (`num_ctx_servers: 2`).**
The router defaults to round-robin, so consecutive requests alternate between
context instances and roughly half of each prefix's repeats land on the instance
that did not compute it. Those are the requests local reuse must recompute from
scratch and the store can serve. Compare arm C against arm B on:
- TTFT p50/p99 (the store's whole thesis is prefill avoided)
- `reused_blocks_per_request` distribution (§8)
- output tokens/s/GPU

This is the primary result. A store that does not win here does not work.

**Experiment 3 -- survival across restarts.**
Run experiment 2's arm C twice, same `TRTLLM_MOONCAKE_STORE_PREFIX`, with the
master job left running between them. The second job starts with an empty local
cache but a warm pool. First-round TTFT should fall toward the warm steady-state
value. Local reuse scores zero here by construction, so any improvement is
attributable to the store alone -- which makes this the cleanest signal in the
whole matrix, and the cheapest to run.

**Experiment 4 -- the cost when there is nothing to gain.**
Arm C against arm B on a workload with *no* repeated prefixes (unique prompts).
This measures pure overhead: lookups, key hashing on the leader, background
saves competing for host bandwidth. Ideally indistinguishable from arm B. This
is the arm that catches a feature that helps its benchmark and hurts the fleet.

## 8. Step 5 -- reading the results

### Did the connector even load?

Every context worker logs at INFO on startup:

```
mooncake-store leader ready (role=both, tokens_per_block=128)
mooncake-store worker rank 0/4 ready (role=both, model_key=MiniMax-M3-NVFP4, master=10.0.0.5:50051)
mooncake-store worker rank 0 registered layout: tokens_per_block=128, lg0(layers=N, regions=..., bytes/page=..., slots=..., window=None)
```

```bash
grep -h "mooncake-store" <log_dir>/3_output_CTX_*.log | head -40
```

The registration line is the one to keep: `bytes/page` per layer group is what
your pool sizing in §4 depends on, and `window=None` is the confirmation that no
sliding-window group is present (one would have aborted startup).

### Is the store actually moving pages?

Hit and transfer counts are at DEBUG. The connector logs under module `_torch`,
so:

```
TLLM_LOG_LEVEL_BY_MODULE="debug:_torch"
```

added to `environment.ctx_worker_env_var`. This is verbose -- it enables DEBUG
for all of `_torch` -- so use it for experiment 1 and for diagnosis, not for the
runs you intend to quote numbers from. The lines worth counting:

```
mooncake-store matched N blocks (M tokens) for request R    # leader, a hit
mooncake-store rank K loaded P pages                        # worker, a load
```

The store's own counters are the alternative that costs nothing at runtime:
`mooncake_master --metrics_port=9004` exposes pool-level statistics over HTTP.
Scrape it before and after a run and diff.

### Where did the pages land?

Page counts alone do not say whether the pool is doing anything the native host
offload tier could not. For that, group the master's allocations by segment host
— a segment is one client process's donated memory, so the host tells you which
node the block physically lives on:

```bash
grep -o "allocation_succeeded size=[0-9]* segment=[0-9.]*:[0-9]*" 2_mooncake_master.log \
  | awk '{sub(/size=/,"",$2); sub(/segment=/,"",$3); split($3,p,":");
          n[p[1]]++; b[p[1]]+=$2}
         END {for (h in n) printf "%-16s pages=%-7d %.2f GiB\n", h, n[h], b[h]/1073741824}'
```

One host means a prefill-only pool (see §3). Two or more, with the generation
node among them, means blocks written by prefill are living on decode-side DRAM
and being read back from there. `disaggr_torch.slurm` writes this breakdown into
`9_mooncake_summary.log` at the end of every run, alongside the donor hosts, so
it needs running by hand only when diagnosing a partial run.

Requires `GLOG_v=1` on the master, which `disaggr_torch.slurm` sets; raise it
with `MOONCAKE_MASTER_GLOG_V`.

### Which reuse number means what

This distinction matters and is easy to get backwards:

| Signal | Where | Includes store hits? |
|---|---|---|
| `reused_blocks_per_request`, `kv_cache_hit_rate_per_request` | per-request iteration stats | **Yes.** `_reserve_connector_prefix` calls `set_prepopulated_prompt_len` with the connector-served position, and these derive from `mPrepopulatedPromptLen`. |
| `kv_cache_iter_reused_blocks`, `kv_cache_iter_reuse_rate` | `GET /prometheus/metrics` | **No.** These come from the local V2 reuse tree's committed stats. |

So **store hits ≈ per-request reuse − local-tree reuse**. Confirm that
relationship on experiment 3, where the local tree starts empty and the
difference is unambiguous, before relying on it elsewhere.

Getting at either one requires the three flags added to the worker configs in
§5, all of which default to false:

- `enable_iter_perf_stats: true` -- without it `get_latest_iteration_stats`
  short-circuits and `GET /metrics` returns `[]`.
- `enable_iter_req_stats: true` -- needed for the *per-request* half of the
  table above.
- `return_perf_metrics: true` -- mounts `/prometheus/metrics`. `GET /metrics`
  (plain JSON iteration stats) is routed unconditionally but still needs
  `enable_iter_perf_stats`.

`print_iter_log: true` is worth keeping on, but note it prints iteration timing
and KV *utilization* only -- no reuse counters. Do not go looking for hit rates
there.

Per-request client-side results land in `<log_dir>/concurrency_<N>/result.json`
with TTFT/TPOT/ITL/E2EL percentiles, which is where the headline numbers for
§7's arms come from.

### Failure signatures

| Log line | Meaning |
|---|---|
| `mooncake-store failed to load N of M pages` (raises) | **Stop.** The runtime had already counted those tokens as computed, so this is the tripwire against silently wrong answers. Do not treat as flaky. |
| `mooncake-store background save failed` | A save thread exception, re-raised on the executor thread. |
| `mooncake-store rank K failed to save N of M pages` (warning) | Dropped write. Costs a future miss, not correctness. A trickle is tolerable; a flood means the pool is full or the master is overloaded. |
| `mooncake-store lookup failed; treating as a miss` (warning) | Probe failed. Degrades to no-store behavior. |
| `could not reserve connector prefix up to N, falling back to the local match` (debug) | Out of GPU pages. The store offered more than the engine could hold -- expected under pressure, but frequent occurrences mean the offer is outrunning capacity. |

### Sanity check that is not a performance number

Run a handful of prompts through arms B and C with temperature 0 and compare the
text. A store that returns the wrong bytes shows up as degraded output long
before it shows up as an error. `accuracy.enable_accuracy_test: true` with gsm8k
gives a coarser version of the same check.

## 9. Things that will bite

- **`host_cache_size: 0` must be written explicitly.** Left at its default of
  `None`, V2 still provisions a host tier, and the gate rejects it. Falsy is not
  the same as absent here.
- **The gates change the config out from under you.** `GUARANTEED_NO_EVICT`
  instead of `MAX_UTILIZATION`, no host tier, no attention DP. That is why
  arm B exists.
- **`sparse_disable_index_value: true` changes the model, not just the cache.**
  Hold it fixed across every arm, generation workers included, or you are
  comparing two different models.
- **Key namespace pins world size and rank.** Change TP and every stored page
  becomes unreachable -- a miss, not an error. Same for `tokens_per_block`, the
  layer group set, and `bytes_per_page`.
- **`model_key` defaults to the checkpoint directory's basename.** Two hosts
  mounting the same checkpoint at different paths still share cache, which is
  intended; two *different* checkpoints in identically-named directories also
  share it, which is not. Set `TRTLLM_MOONCAKE_STORE_MODEL_KEY` explicitly for
  anything long-lived.
- **Stale pages across code changes.** The key namespace does not include a
  build hash. After changing anything about page layout or contents, bump
  `TRTLLM_MOONCAKE_STORE_PREFIX` or restart the master.
- **UCX warmup requests hit the store too.** `run_benchmark.sh` sends
  `2 x ctx_instances x gen_instances` 100-token requests before the real run.
  Harmless, but they are in the counters.
- **`enable_chunked_prefill` interacts with the offer.** The connector offers
  only whole blocks and only when the local match is block-aligned; a partial
  local match disables the store for that request entirely. With
  `tokens_per_block: 128` this is rare, but it explains occasional zero-offer
  requests.
- **`block_reuse_policy: per_conversation` is off on the context worker** in
  these configs. It is not gated, but the connector derives its own
  `cache_salt`-seeded hash chain and the interaction is untested. Restore it
  only after the store is proven, and treat it as its own experiment.

## 10. What this does not test

Worth stating so the results are not oversold: single-node only insofar as the
master is one process (no HA master, no `--root_fs_dir` persistence); no
pipeline or context parallelism (both refused); no VSWA or sliding-window model
(refused); no Eagle3; no shared pool between TensorRT-LLM and vLLM, though the
config schema is deliberately compatible with it. Load bandwidth under
contention from many simultaneous large prefixes is exercised only incidentally
by concurrency, not measured directly -- if experiment 2 shows a TTFT
regression at high concurrency despite hits, that is the first thing to profile.
