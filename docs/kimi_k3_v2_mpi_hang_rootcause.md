<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Root cause: Python V2 KV-cache-transceiver "MPI allgather hang" in multi-node SLURM launches

**Status: root-caused and validated.** The hang reported against
`KvCacheTransceiverV2._exchange_rank_info` → `mpi_allgather` (job 2655477,
blocking M1.1) is **not an MPI/pmix/communicator bug and not a V2
transceiver code bug**. It is a downstream symptom of rank-asymmetric
blocking/failure inside **native NIXL/UCX agent initialization** under
`UCX_TLS=all` — an environment that is fatally broken on these NVL72 nodes
(`ud_verbs` cannot initialize). The NIXL/PYTHON combination was **only ever
run multi-node under that broken env**; with the known-good
`UCX_TLS=tcp,self,sm,cuda_copy,cuda_ipc` it passes multi-node with no code
change (validation job below).

## 1. Symptom as reported

Job 2655477 (2 nodes × 4 ranks per srun step, `srun --mpi=pmix`, pytorch
26.05 container): the `{backend: NIXL, runtime: PYTHON}` cell of the
`cache_transceiver_test` harness never finished setup;
`KvCacheTransceiverV2.__init__` appeared to block forever in
`self._dist.allgather(...)` → `mpi_allgather`
(`tensorrt_llm/_torch/disaggregation/transceiver.py` `_exchange_rank_info`;
`tensorrt_llm/_torch/distributed/communicator.py` `MPIDist.allgather` →
`tensorrt_llm/_utils.py mpi_allgather`). A watchdog killed the sweep after
900 s. Single-node loopback V2 NIXL transfers worked
(`tests/unittest/disaggregated/test_kda_mamba_transfer.py`), so the hang was
suspected to be launch-mode-specific (pmix / mpi4py world mismatch).

## 2. Evidence chain

All artifacts under
`.claude/worktrees/kimi-k3-track-e-analysis/log/ctt_kda_payload/`
(status JSONL is append-only across jobs; entries attributed by job below).

1. **Job 2655477 ran NIXL/PYTHON only under `UCX_TLS=all`.** Its single
   sweep was `venv_all` (`logs/ctt-2655477.log`: `=== sweep 0 (venv_all)
   UCX_TLS=all ...`). No multi-node run of NIXL/PYTHON with the good
   transport list ever existed before the validation job below.

2. **`UCX_TLS=all` is fatally broken on these nodes.** Job 2655507 (CPP
   combos, same sweep env) failed with `mlx5dv_devx_alloc_uar ... Cannot
   allocate memory. Consider increasing PF_LOG_BAR_SIZE using mlxconfig` →
   `uct_iface_open (ud_verbs) failed`, and NIXL agent creation raised
   `RuntimeError('Failed to create NIXL backend: UCX ...')`
   (`status/sweep0_{ctx,gen}.jsonl`, TRANSFER_ERROR entries). So under
   `UCX_TLS=all`, native NIXL/UCX agent init on these nodes either fails
   with an exception or wedges inside the transport open — and which of the
   two happens varies per rank/run.

3. **MPI collectives under `srun --mpi=pmix` in this container demonstrably
   work.** Job 2655624 (same harness, same launch mode, good `UCX_TLS`) ran
   the CPP combos green at ~500 GB/s. Those runs exercise the exact same
   MPI machinery per srun step: the harness's own `COMM_WORLD`
   allreduce/bcast/Barrier calls plus the `KVCacheManager` constructor's
   collective. `mpi_comm()` returns
   `pkl5.Intracomm(MPI.COMM_WORLD)` (`tensorrt_llm/_utils.py`), i.e. the
   4-rank world of the step — the same communicator the harness itself uses
   successfully. Hypotheses (a) "world/communicator mismatch under pmix"
   and (d) "TLLM MPI init flags" are ruled out empirically.

4. **The V2 setup sequence interleaves MPI collectives with per-rank native
   NIXL/UCX init.** `KvCacheTransceiverV2.__init__`
   (`tensorrt_llm/_torch/disaggregation/transceiver.py`):

   1. `_broadcast_instance_name()` — MPI bcast (collective)
   2. `TransferWorker(...)` — **native, per-rank**: NIXL agent creation
      (`native/transfer.py _setup_transfer_engine` → `_create_nixl_agent`),
      VRAM/DRAM `register_memory`, ZMQ endpoints. None of these release the
      GIL around the UCX transport-open path.
   3. `_broadcast_context_endpoint()` — MPI bcast (collective)
   4. `_exchange_rank_info()` — **MPI allgather** (collective)

   If any rank blocks (or dies out-of-band) in step 2, every other rank of
   the instance parks forever in step 3/4. The observed stack —
   "blocked in `mpi_allgather`" — is where the *healthy* ranks wait, not
   where the fault is.

5. **The per-rank kill pattern of 2655477 matches exactly.**
   `logs/ctt-2655477.log`: on the ctx node tasks 0,3 were SIGKILLed (the
   harness `HangDetector` fired: those ranks were in GIL-*released* MPI
   waits, and ctx rank 0 recorded `TIMEOUT "hang detected during setup
   NIXL/PYTHON/V1"`); on the gen node tasks 1,3 were SIGKILLed, and **gen
   rank 0 recorded nothing at all** — the documented `HangDetector`
   limitation (`run_cache_transceiver_test.py`): its callback cannot fire
   on a rank stuck in a native call that *holds* the GIL. I.e. on each
   node some ranks were wedged inside native NIXL/UCX init (GIL held,
   silent) while the others waited in the MPI collectives (GIL released,
   watchdog fired there and blamed `mpi_allgather`).

6. **Hypothesis (c) — harness vs real worker divergence — ruled out by
   inspection.** The harness builds `Mapping(world_size=4, rank, tp=4)` and
   `Distributed.get(mapping)` → `MPIDist` over `mpi_comm()`; the real
   PyExecutor path (`tensorrt_llm/_torch/pyexecutor/_util.py` →
   `create_kv_cache_transceiver(mapping, dist, ...)`) passes the same
   `MPIDist` built the same way. No structural difference.

## 3. Root cause statement

Under `UCX_TLS=all` on NVL72 nodes with the `PF_LOG_BAR_SIZE` NIC
misconfiguration, NIXL/UCX agent initialization inside
`TransferWorker.__init__` blocks or fails **per-rank, asymmetrically**.
Because `KvCacheTransceiverV2.__init__` runs that native init between MPI
collectives, the surviving ranks hang in the next collective
(`_exchange_rank_info` → `mpi_allgather`). The hang was misattributed to
MPI/pmix; the actual faulting phase was invisible because it holds the GIL,
which also blinded the in-process watchdog on those ranks.

Consequences:

* **Not harness-specific in the important sense**: a real
  `trtllm-serve` disagg worker started with `transceiver_runtime: PYTHON`
  in the same broken UCX environment would hang identically (same init
  sequence, same collectives). The *trigger* is environmental, not the
  launch mode: `srun --mpi=pmix` is innocent.
* **No product-code deadlock bug to fix for M1.1**: with the environment
  pinned per the cluster prerequisites (`UCX_TLS=tcp,self,sm,cuda_copy,
  cuda_ipc`; never `all`; unset the container's `UCX_TLS=tcp`), V2
  NIXL/PYTHON works multi-node (validation below).

## 4. Fix

Two parts, matching where the fault actually lives:

1. **Product (diagnostic hardening),
   `tensorrt_llm/_torch/disaggregation/transceiver.py`**: per-rank,
   per-phase `logger.info` markers around each setup phase of
   `KvCacheTransceiverV2.__init__` (instance-name bcast → TransferWorker
   native init → context-endpoint bcast → rank-info allgather → complete).
   This turns the next occurrence of "hangs in mpi_allgather" into a
   one-glance diagnosis: the blocked rank is the one whose last line is
   `creating TransferWorker (native NIXL agent init + KV memory
   registration)`. Justification: the deadlock-by-asymmetric-native-init
   pattern is generic (any env/NIC fault reproduces it), and the incident
   showed the existing logs cannot localize it.

2. **Harness/config**: the NIXL/PYTHON combination is re-enabled in the
   micro-bench config with the pinned transport list
   (`log/ctt_v2_hang/config.yaml` in this worktree); the stale "hangs under
   srun --mpi=pmix" comment in the Track-E config is superseded by this
   document. No change to `run_cache_transceiver_test.py` launch logic —
   its MPI usage was correct.

No change is made to `mpi_allgather`/`MPIDist`: they behaved correctly.

## 5. Validation (job 2656113)

2 nodes × 4 GPUs (`nvl72d094-T05` ctx → `nvl72d094-T06` gen), same
container and `srun --mpi=pmix` launch as the failing job, worktree
`gpd-v2-mpi-hang` code via `PYTHONPATH`. Config:
`log/ctt_v2_hang/config.yaml` — one request length (2144 tokens =
454,459,392 B/rank, the exact Kimi K3 KDA payload), combos NIXL/PYTHON +
NIXL/CPP (control), two sweeps:

| Sweep | UCX_TLS | NIXL/PYTHON/V1 result | NIXL/CPP/V1 (control) |
|---|---|---|---|
| 0 `venv_no_verbs` | `tcp,self,sm,cuda_copy,cuda_ipc` | **setup completes, transfers run** (no hang; MISMATCH + slow transport, see below) | PASS, 507.81 GB/s/GPU (cuda_ipc) |
| 1 `venv_all` (repro of 2655477) | `all` | **hang reproduced**, step SIGKILLed (ctx_rc=143) | TRANSFER_ERROR (`Failed to create NIXL backend: UCX`, rc_mlx5) |

Measured bandwidth (per GPU, req_len 2144): NIXL/CPP 507.81 GB/s
(cuda_ipc); NIXL/PYTHON 0.35 GB/s over `tcp(sw-emul)` — see follow-ups.

Phase-log evidence from sweep 1 (which rank blocked where): with the new
per-phase markers, ctx ranks 0, 2, 3 each end at
`creating TransferWorker (native NIXL agent init + KV memory registration)`
— wedged in native init, GIL held — while ctx rank 1 reached
`TransferWorker ready; broadcast context endpoint (collective)` and parked
in the collective. This is the asymmetric-native-init deadlock pattern
directly observed, closing the loop on §2.4–2.5.

The sweep-0 result also **disproves the original framing**: NIXL/PYTHON
setup (including `_exchange_rank_info` → `mpi_allgather`) completes fine
multi-node under `srun --mpi=pmix` once the UCX environment is sane.

### 5.1 New issues exposed once past setup (not the hang)

Sweep 0's NIXL/PYTHON cell reported `MISMATCH` at 0.35 GB/s over
`tcp(sw-emul)`:

1. **MISMATCH — harness bug, fixed in this worktree.** The PYTHON
   runtime's `check_context_transfer_status(None)` is bounded per-session
   by `kv_transfer_sender_future_timeout_ms` (default 1000 ms); on a slow
   transport it returns with the send still in progress, the harness frees
   and refills the KV blocks mid-flight, and the receiver verifies against
   the next request's pattern (every request FAILs except the last). Fix:
   `_wait_ctx_complete` poll loop in `run_cache_transceiver_test.py`.
   Validated by job **2662383** (`log/ctt_v2_fix/`): NIXL/PYTHON now
   **PASS** (0.43 GB/s over tcp(sw-emul)); NIXL/CPP control PASS at
   563.01 GB/s (cuda_ipc).
2. **Slow transport selection — root-caused (fabric memory), validation
   pending.** V2 PYTHON selected `tcp(sw-emul)` where CPP selected
   `cuda_ipc` (563 vs 0.43 GB/s) under the identical `UCX_TLS` list.
   The UCX proto table in the PYTHON cell shows the fallback happening on
   the inter-node bulk-data path: `remote memory write by ucp_put*(multi)
   from cuda/GPU0 to cuda | software emulation | tcp/eth0`
   (`log/ctt_v2_fix/logs/sweep0_ctx_rank0.log`). Why the paths differ:

   * Both runtimes use the same C++ `NixlTransferAgent`/UCX backend, so
     agent config is not the difference.
   * The **CPP runtime never sends from the KV pool**: it stages through
     `CacheTransBufferManager` buffers allocated with **fabric memory**
     (`mUseFabricMemory:1` in the same log). On NVL72, UCX `cuda_ipc` can
     map peer memory **across nodes** (MNNVL) only for
     `CU_MEM_HANDLE_TYPE_FABRIC` allocations — hence cuda_ipc at 563 GB/s.
   * The **V2 PYTHON runtime (bounce off, the default) sends directly
     pool-to-pool**. The V1 KV pool is a plain device allocation (no
     fabric handle), so inter-node cuda_ipc is impossible; and the pinned
     `UCX_TLS` list contains no RDMA transport (verbs are broken on these
     nodes, §2.2), so the only inter-node transport left is `tcp` with
     host-staged software emulation. 0.43 GB/s is the expected ceiling.
   * V2's own **bounce path is the designed fix**: `kv_cache_bounce_size_mb
     > 0` coalesces each request into a fabric-VMM staging buffer
     (`native/bounce/buffer.py` → `PooledPhysMemAllocator` requests
     `CU_MEM_HANDLE_TYPE_FABRIC`), restoring cuda_ipc/MNNVL eligibility.
     First validation attempt, job **2662439** (`log/ctt_v2_bounce/`,
     bounce_size_mb=512, harness plumbed via `kv_cache.bounce_size_mb`):
     fabric send/recv regions allocated on every rank, but bandwidth stayed
     at 0.39 GB/s over tcp(sw-emul) — the KV data never took the bounce
     path, because `Config.min_blocks` defaults to **96** and the 2144-token
     payload is only **67 blocks**, so `reserve()` silently skipped every
     request (the `[kv-bounce] in-place:` skip line is `logger.debug`,
     invisible at INFO). The same run still proves the fabric mechanism:
     the proto table shows an inter-node put to a fabric-mapped destination
     (`from cuda/GPU0 to cuda/dev[0]`) selecting `zero-copy | cuda_ipc/cuda`,
     while puts to the plain pool (`to cuda`, no dev) get
     `software emulation | tcp/eth0`. Second attempt with the gate lowered
     (`TRTLLM_KV_CACHE_BOUNCE_MIN_BLOCKS=1` in the sweep env): job
     **2662467** (`log/ctt_v2_bounce2/`) — **CONFIRMED**. NIXL/PYTHON
     PASS at **454.81 GB/s/GPU over cuda_ipc** (p90 559.83), on par with
     the NIXL/CPP control (555.75 GB/s, cuda_ipc); the bounce log shows
     `coalesced 67 blocks / 433MiB into one region`. Transport issue
     closed: ~1160x over the pool-to-pool path (0.39 GB/s).

   **Deployment guidance (V2 on this cluster):** set
   `kv_cache_bounce_size_mb > 0` (512 validated); no gate env is needed
   anymore. The hardening candidates identified here have LANDED (branch
   `brnguyen/kv-bounce-gate-hardening`): (a) the gate is now expressed in
   **bytes** (`Config.min_bytes`, default 2 MiB, env
   `TRTLLM_KV_CACHE_BOUNCE_MIN_BYTES`), so it is independent of the
   model's tokens-per-block — the K3 433 MiB / 67-block payload clears it
   by construction; the legacy block-count gate
   (`TRTLLM_KV_CACHE_BOUNCE_MIN_BLOCKS`) is kept for back-compat but now
   defaults to 1 (vacuous); (b) every `[kv-bounce] in-place:` skip reason
   is logged at `warning_once` (per distinct reason), so a fall-back to
   the per-fragment path is visible at the default log level — a ~1000x
   bandwidth cliff is no longer silent.

## 6. Follow-ups / deployment guidance

* Any K3 disagg deployment on this cluster must pin
  `UCX_TLS=tcp,self,sm,cuda_copy,cuda_ipc` (and unset the container's
  `UCX_TLS=tcp` for V2 NIXL VRAM registration) — already captured in the
  payload memo's environment prerequisites; this doc removes the "V2
  can't run multi-node" blocker from M1.1.
* The §5.1 bounce-gate hardening (byte-based `min_bytes` gate + visible
  skip logs) has landed; deployments only need
  `kv_cache_bounce_size_mb > 0`.
* Optional future hardening (not required for M1.1): bound the native
  NIXL agent init with a timeout/heartbeat so a wedged transport surfaces
  as a per-rank error instead of an instance-wide collective stall.
