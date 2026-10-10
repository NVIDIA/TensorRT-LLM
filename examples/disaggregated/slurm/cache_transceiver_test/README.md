# KV Cache Transceiver Bandwidth / UCX-Tuning Harness

A standalone SLURM harness that validates the TensorRT-LLM KV cache transceiver
across two cluster nodes and helps you **find UCX environment variables that give
good cache-transceiver bandwidth**.

It does **not** run a model or full serving. It allocates a KV cache, fills it
with deterministic data on the context (ctx) side, transfers it to the
generation (gen) side via the transceiver, verifies the received data, and
reports the achieved transfer bandwidth — for both the **C++** and **Python**
transceivers, and for every UCX environment set you list.

The `host_transfer_benchmark` runs after each sweep on the same nodes and GPU
ranks. It sends the same byte payload through NIXL as GPU→GPU, host→host,
host→GPU, and GPU→host transfers. GPU→GPU is the comparison baseline. Every
mode begins with the payload on the context GPU and ends with it on the
generation GPU. Host modes time the necessary GPU→pinned-host and/or
pinned-host→GPU copies separately from NIXL, and report both transport-only
bandwidth and GPU-ready bandwidth from the sum of those stages. Every sample
verifies the GPU-resident destination against a deterministic pattern. Two untimed
cases retain registered buffers after submission: `delayed_completion` defers
completion reporting, and `logical_cancel` sends cancellation notice before a
separate physical-drain handshake. If physical
completion cannot be proven, the process terminates without deregistering the
possibly live buffer.

This measures the physical NIXL DRAM/VRAM payload path and local staging cost.
It does not allocate offloaded KVCM pages or exercise request claims. GPU-ready
bandwidth excludes allocation, registration, KVCM admission, scheduling, and
model execution; it is not end-to-end serving throughput.

The `kvcm_transfer_benchmark` adds the native-page measurement. It creates a
single sparse attention layer in KVCM V2, writes deterministic bytes to its GPU
history, and calls `enter_decode()` to offload that history when a mode uses a
host source. The destination reserves detached GPU or host pages. Both sides
pin and expose their pages through KVCM external-access claims; NIXL registers
the claimed pools and transfers pages paired by group, ordinal, and pool index.
After each transfer the receiver compares every byte. On completion it
finalizes the receive pages, then releases claims only after the backend status
and peer handshake prove physical settlement. `mixed` reserves alternating
host and GPU destination pages, exercising two NIXL memory-type batches.

The KVCM benchmark compares host modes against its own GPU→GPU baseline with
the same page geometry. It measures transport-only bandwidth through real KVCM
addresses. Source offload occurs once during setup, and destination promotion
to GPU is not timed, so its host→host ratio must not be read as a GPU-ready
speedup. Production PD transfer planning, layer remapping, and decode are
outside this benchmark.

## Topology

One SLURM job on **2 nodes**. Per UCX env set, two `srun` steps run concurrently:

```
node A  ── ctx instance (MPI world = N = gpus_per_node) ──┐
                                                          │  UCX / NIXL (KV transfer)
node B  ── gen instance (MPI world = N = gpus_per_node) ──┘
                          ▲
                          └── ZMQ (leader-to-leader): hands the ctx connection info to gen
```

Each `srun` step is its own MPI world of size `N`, matching how the C++
transceiver expects `MpiComm::session()` to be one instance. ZMQ carries only the
small control payload (the context connection info / endpoint); the KV data
itself flows over UCX/NIXL.

## Usage

1. Edit `config.yaml`: set `slurm.{partition,account,job_time,job_name}`,
   `environment.{container_image,container_mount,work_dir}`, and how TensorRT-LLM
   is installed (`trtllm_wheel_path`, or `trtllm_repo` + `build_wheel`, or leave
   both empty to use the container's pre-installed build).
2. List the UCX environment sets to sweep under `ucx_env_sweep`.
3. Submit:

   ```bash
   ./submit.py --dry-run  # validate + print sbatch only
   ./submit.py            # submit
   ```

The older `python3 submit.py -c config.yaml` command still works. Pass
`-c other_config.yaml` to either form to use another configuration file.
The host and KVCM benchmark modes use sections in the same config and write
their measurements into the same `results.json` as the transceiver sweep.

## What it tests

- **Transceiver combinations** (`test_matrix.combinations`): C++ UCX, C++ NIXL,
  Python NIXL (each entry is a `{backend, runtime}` pair).
- **Cache managers** (`test_matrix.cache_manager_versions`): `V1` and/or `V2`.
  The full matrix is combinations × versions; **V2 only supports the Python
  transceiver**, so `V2 + CPP` cases are skipped automatically. (In `results.json`
  a "combination" is the full `backend/runtime/version` tested unit.)
- **Request lengths** (`test_matrix.request_lengths`): e.g. 100 / 1k / 8k tokens.
  One transceiver + cache manager is built per case and reused across all lengths.
- Multiple requests per length for stable bandwidth (plus warmup).
- Every rank prints `UCX_TLS` and `UCX_NET_DEVICES`; `UCX_PROTO_INFO=used` is set
  so UCX >= 1.21 captures the selected GPU↔GPU transport in the stderr logs.
- Failing transfers are reported per cell as `TRANSFER_ERROR` / `MISMATCH` /
  `TIMEOUT`. A verification mismatch is safe to continue; a result that does
  not prove transfer quiescence hard-aborts only the current UCX sweep so its
  cache manager cannot recycle or deregister pages still owned by transport.

## Outputs (under `environment.work_dir`)

```
resolved_config.json            # the validated config the job ran with
logs/ctt-<jobid>.log            # batch-level log (stdout+stderr merged)
logs/install.log                # TensorRT-LLM install log
logs/sweep<i>_<role>_rank*.log  # per-rank logs: transfer START/DONE+verify, UCX_PROTO_INFO
csv/<i>/ctx|gen/<instanceId>_*_{send,recv}.csv # C++ transceiver timing (Bandwidth(Gbps)); renamed to <instanceId>_*_{send,recv}__c<i>.csv per combination
csv/<i>/ctx/py_*_*.csv                 # Python transceiver perf log (throughput_mbs)
status/sweep<i>_<role>.jsonl           # PASS / MISMATCH / TRANSFER_ERROR / TIMEOUT
results.json                    # full results, grouped per combination (longest req_len)
results.best.json               # best UCX env per combination (the deliverable)
host_transfer/sweep<i>_rank<r>_<mode>.json # verified physical transfer samples
kvcm_transfer/sweep<i>_rank<r>_<mode>.json # verified native-page transfer samples
```

`results.json` is organized **per combination** (`by_combination`); under each combination every
UCX sweep is listed with the GPU↔GPU transport UCX selected and the **per-GPU**
bandwidth at the **longest** request length (the peak/representative figure;
small requests are latency-bound). `results.best.json` (and the printed table)
ranks, for each combination, the UCX env set with the best per-GPU bandwidth — the
deliverable for tuning your cluster.

- **Per-GPU, not aggregate**: bandwidth is the median across ranks (GPUs), not a
  sum — summing per-rank rates with unequal durations overstates throughput.
  `aggregate_BW_GBps` (= per-GPU × #GPUs) is reported separately and labeled.
- **Transport is per (sweep, combination)** (constant across request lengths: one
  transceiver per case). It is read from the KV-data-path rows of the
  `UCX_PROTO_INFO=used` table (`ucp_put`/`rendezvous` *to cuda*), not from
  control traffic. A host-staged fallback shows as `tcp(sw-emul)` — a red flag
  that GPUDirect/cuda_ipc was unavailable (e.g. cross-NVLink-domain nodes).
- **Attribution is timestamp-based**: the driver logs each request's `transfer
  START/DONE` with a timestamp; `report.py` matches each UCX proto config (which
  carries its own UCX timestamp) to the case whose transfer window it falls in.
  This is robust to UCX writing its proto table to the log buffered / out of
  order relative to the `[CTT_CASE_BEGIN]` markers (line position is not).

## Bandwidth sources

| Transceiver | Env enabling timing | File | Column (native) |
|---|---|---|---|
| C++ (UCX/NIXL) | `TRTLLM_KVCACHE_TIME_OUTPUT_PATH` (set by the driver) | `<instanceId>_*_recv.csv` (renamed `<instanceId>_*_recv__c<i>.csv` per combination) | `Bandwidth(Gbps)` |
| Python (NIXL) | `TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO=1`, `TLLM_KV_TRANSFER_PERF_LOG_FILE` (set by the driver) | `py_*_*.csv` | `throughput_mbs` (MiB/s) |

`report.py` normalizes both to **per-GPU GB/s** (bytes, ÷1e9): C++
`Bandwidth(Gbps) ÷ 8`, Python `throughput_mbs × 1024² ÷ 1e9`, then takes the
median across ranks. The two runtimes time slightly different spans, so compare
within a `(combination)` across UCX sweeps.

For `host_transfer_benchmark`, `per_gpu_GBps` and `ratio_to_gpu` remain the
transport-only measurements. `gpu_ready_GBps` and `gpu_ready_ratio_to_gpu` use
`gpu_ready_seconds = source_stage_seconds + seconds + destination_stage_seconds`
for each timed request, then take the median per-GPU rate. The staging copies
are repeated for every request; warmup, delayed completion, and logical cancel
are excluded from bandwidth statistics. Existing result files without staging
timings retain transport-only values and report `null` for GPU-ready values.
`kvcm_transfer_benchmark` reports only transport-only rates.

## Notes & limitations

- Requires a **symmetric** layout (`ctx_tp == gen_tp`, `ctx_pp == gen_pp`, both
  equal to `gpus_per_node`); enforced by `submit.py`. Verification compares each
  gen rank's received blocks against a locally regenerated, rank-specific pattern,
  so no cross-node gather is needed.
- A bad `UCX_NET_DEVICES`/`UCX_TLS` can hang a transfer. Hangs are handled at
  three layers so one stuck sweep never blocks the others:
  1. **The Python sender deadline plus `signal.alarm`** bounds Python-level
     stalls. `timeout_per_cell_s` must be greater than five seconds. The Python
     sender deadline is five seconds lower, leaving time for CTX and GEN to
     exchange a final ownership decision. The alarm is the per-cell bound for
     the C++ path and for later requests after the shared cell budget is spent.
     A timeout or any other result without explicit completion retains the
     pages and hard-aborts the current sweep; the harness never rebuilds a
     cache manager over possibly active transfer memory.
  2. **Watchdog thread** (per cell) best-effort records `TIMEOUT` and `SIGKILL`s
     the process for hangs inside a *GIL-released* native call (e.g.
     `check_*_transfer_status`); `srun --kill-on-bad-exit` then tears down the
     sweep. This is **best-effort**: it cannot fire if the hang is in a native
     call that *holds* the GIL (e.g. the UCX connection handshake inside
     `respond_and_send_async`/`request_and_receive_async`), because the watchdog
     thread can't acquire the GIL.
  3. **`timeout -k 60 <max_sweep_s>` around each `srun`** is the *guaranteed*
     killer — it is external to the process, so it works regardless of the GIL.
     This caps one full sweep at `run.max_sweep_s` (default 180 s = 3 min): a
     hung sweep is killed and the loop advances to the next UCX env set within
     that bound. **Set `max_sweep_s` to comfortably exceed a healthy sweep's
     runtime** (all cases × request lengths), or healthy sweeps get cut off.
- `max_tokens_in_buffer` must be ≥ the largest request length.
