<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Wide-EP Fault Tolerance Developer Guide

## Scope

Wide-EP fault tolerance (`enable_wide_ep_fault_tolerance`) keeps a wide expert-parallel deployment serving when
ranks fail. The survivors remove the failed ranks, restore their experts from host weights into spare EPLB slots, and
serve at N-K ranks on the same CUDA graphs. A replacement process then starts on the failed rank's GPU, rejoins, and
the deployment serves at N ranks again.

Use this guide when changing anything under `tensorrt_llm/_torch/distributed/membership/`, the files in the
[File map](#file-map), or a shared path that a fault-tolerant deployment runs (`Distributed`, the attention-DP vote,
the `NVLinkOneSided` all-to-all, EPLB, KV cache manager V2's resize and reuse commit). User-facing configuration,
launch and monitoring are in the
[feature page](../../../../docs/source/features/wide-ep-fault-tolerance.md). The MoE side of the backend contract is
also summarized in the [MoE Developer Guide](../../moe/fused_moe/MOE_DEVELOPER_GUIDE.md#wide-ep-fault-tolerance).

| In scope | Out of scope (rejected at startup with the reason, or not covered) |
|---|---|
| A rank's process exits (crash, `kill -9`) or hangs | Hardware and network failures (a node that crashes, reboots, loses power or is cut off); the authority's keepalive (`DEAD_TIMEOUT_S`) still detects a silent connection |
| Every process of the deployment on one node exits, while the launcher's own processes there survive | Losing rank 0, node 0 or the authority (the deployment ends) |
| Serving at N-K on the same CUDA graphs, then back at N after a replacement rejoins on the same GPU | A replacement on another GPU or node; restarting a lost node automatically |
| DeepSeek-V3 / R1 (`DeepseekV3ForCausalLM`), attention DP, `NVLinkOneSided` fence path, standard EPLB, KV cache manager V2, the overlap scheduler, MTP, the MPI executor, one NVLink domain | Every configuration that `check_llm_args` (`tensorrt_llm/_wide_ep_fault_tolerance.py`), `check_supported` and the MoE backend checks reject, with the reason (the feature page's Requirements list the main ones). A MoE backend without spare-slot support is not rejected where a Cutlass leaf serves the layer: selection falls back to it with a warning; Kubernetes |

## Glossary

| Term | Meaning |
|---|---|
| Active rank mask | The ranks in the current membership (`MembershipRecord.active_mask`). The all-to-all kernels read it from device memory (`active_rank_mask`); `EPGroupHealth` mirrors it per rank |
| Authority | The CPU-only process, started by rank 0, that decides membership and hosts the rendezvous store |
| Membership record | What the authority publishes and every rank follows: generation, state, active rank mask, resume state, incarnations, joining ranks, given-up ranks, reason |
| Generation | Version of the membership, incremented by every PROPOSE |
| Epoch | Round counter of the host collectives (`ResumeState.epoch`). Communication backends derive their per-round counters from it, so every rank agrees after a change |
| Rearm | Clearing the incomplete-round word of the all-to-all for its next round |
| Rank report | What a rank sends the authority every 0.5 s: progress, the peers and rounds it waits on, its reply, where it paused, whether it is ready to join |
| Pause word | A 4-byte word per rank in `/dev/shm` that holds the rank's PID while its hang detector is paused and 0 while armed. The hang detector writes it; the node supervisor reads it |
| Node report | What a node supervisor sends the authority every 0.5 s: its given-up ranks and its paused ranks (pause word set, process not stopped) |
| Paused (`paused_mask`) | The rank's executor hang detector is paused (`HangDetector.pause`), unlike the pause of the forward for a membership change (`paused_at`, `ResumeState`) |
| Forward vote | The attention-DP vote before each forward, under fault tolerance `RankFailureRecovery.forward_vote` |
| Hello | The first line a process sends on its liveness connection: the token, its rank and whether it is a replacement |
| Survivor / joining rank | An active rank that was a member before the change / a replacement that the change adds |
| Incarnation | 0 for a rank's original process, n for its n-th replacement. The authority assigns it when the process connects |
| Liveness connection | A rank's (or node supervisor's) one TCP connection to the authority. The kernel closes it when the process ends |
| Hang timeout | `HANG_TIMEOUT_S` (60 s): how long the end of a wait-for chain may make no progress before the rank counts as failed |
| Abort state | Pinned host words that the all-to-all kernels poll inside their waits. Set at a removal, so waits on a dead peer end |
| Incomplete round | An all-to-all round that an abort cut short. Its outputs are never consumed |
| Fence path | The `NVLinkOneSided` synchronization mode that honors the active rank mask and the abort state |
| Late-write remap | Remapping peer-writable MNNVL memory among the active ranks at the same virtual addresses, with fresh physical pages, so a former member's late writes land in pages nobody reads |
| Spare slots | Extra EPLB slots per rank, unbacked while healthy, backed when a failed rank's experts are restored into them. Active slots are the slots the load balancer routes to |
| Host weights | Host copies of expert weights in `/dev/shm`, from which lost experts are restored |
| Startup record | Rank 0's record of the startup host collectives, which a replacement replays so it reaches the same results alone |
| Node supervisor | One process per node that starts the node's ranks as separate one-process MPI worlds and replaces a failed one |

## Design Decisions

These are fixed for this feature. A change that needs to revisit one is a design change, not an implementation detail.

| Decision | Reason |
|---|---|
| Fence path only | Only the fence path stops waiting on a failed rank: its waits read the active rank mask and poll the abort state. Under FT the all-to-all skips the sentinel transport, whose receivers poll the data itself |
| Unbacked spare slots | Healthy memory and KV capacity equal FT off. A spare slot maps one shared scratch chunk until a restore backs it, and is unbacked again once it is empty |
| Separate authority process, started by rank 0 | Membership decisions must not stall with any rank's executor (GIL, CUDA calls, a hang). Rank 0 is never replaced, so the authority adds no failure domain beyond node 0 |
| Same CUDA graphs | Recapturing costs time and memory at every change. The kernels read the mask and the abort state through fixed addresses, and the remap keeps every address |
| Process failures only | A hardware failure takes the launcher's processes on the node with it, and its recovery needs another GPU or node |
| A replacement restarts on the same GPU | The GPU and node are usable after a process failure, the replacement shares the job's IMEX channel, and no spare nodes are needed |
| Host collectives are `MnnvlDist` over MNNVL memory; no NCCL or MPI communicator | MPI cannot admit a process started outside its job, and torch/NCCL groups are fixed. NCCL can abort and rebuild, but the rebuild adds to every membership change's pause and its per-step host collectives are slower than `MnnvlDist`'s. The `HostExchangeGroup` seam keeps another transport possible |
| The node supervisor restarts single ranks automatically; restarting a lost node is manual | One rank's restart needs only its own node's supervisor. A lost node has no supervisor left, and starting processes there goes through the launcher, which differs per deployment. Its ranks are given up and the deployment serves at N-K until the operator starts the node's supervisor again with `trtllm-llmapi-launch --rejoin-node`. Its replacements rejoin in one membership change |
| One feature switch, `enable_wide_ep_fault_tolerance`, per instance | It is the only switch that turns the feature on. It flows from `TorchLlmArgs` to the model config, the worker and the server. A multi-node launch also takes two deployment settings, `TLLM_FAULT_TOLERANCE_SHARED_SECRET` and `TLLM_FAULT_TOLERANCE_RENDEZVOUS`; with the secret exported, `trtllm-llmapi-launch` runs each node's supervisor instead of the command. The rest is derived inside each rank process: `Distributed.set_implementation` installs `MnnvlDist`, and the startup sets internal variables such as `TLLM_RANK`, and `TRTLLM_SKIP_KV_CACHE_ESTIMATION` on a replacement |
| Narrow first target, extensible later | Each supported combination needs end-to-end coverage. The generic layer (membership, host collectives, executor failover and rejoin, serving) knows no model or backend, so a new target only has to meet the backend contract |

## File Map

| Component | Files (key symbols) |
|---|---|
| Requirement checks, rank identity | `tensorrt_llm/_wide_ep_fault_tolerance.py` (`check_llm_args`, `check_node_supervisor_launch`, `SUPPORTED_ARCHITECTURES`, `FAULT_TOLERANCE_RANK_ENV`, `RankIdentity`); `check_supported`, `prepare_model`, `moe_comms` and `create_moe_load_balancer` in `tensorrt_llm/_torch/pyexecutor/rank_failure_model_load.py` |
| Record, reports, decisions | `tensorrt_llm/_torch/distributed/membership/records.py` (`MembershipRecord`, `MembershipState`, `RankReport`, `NodeReport`, `ResumeState`, `TransactionSummary`); `tensorrt_llm/_torch/distributed/membership/authority.py` (`Authority`: no I/O, driven by simulated ranks in tests) |
| Authority process | `tensorrt_llm/_torch/distributed/membership/authority_process.py` (`AuthorityProcess`), `tensorrt_llm/_torch/distributed/membership/__main__.py` |
| Liveness and the rank side of the protocol | `tensorrt_llm/_torch/distributed/membership/liveness.py` (`MembershipClient`, `open_store`, `configure_connection`) |
| Pause words | `tensorrt_llm/_torch/distributed/membership/pause_word.py` (`pause_word_path`, `open_pause_word`, `read_pause_word`) |
| Rendezvous store | `tensorrt_llm/_torch/distributed/membership/signed_store.py` (`SignedStore`: HMAC-signed values); `tensorrt_llm/_torch/distributed/membership/rendezvous.py` (`StoreGroup`, `wait_for_keys`, store keys), on `StoreComm` (`tensorrt_llm/_torch/modules/dwdp/rendezvous.py`) |
| Rank startup and replay | `tensorrt_llm/_torch/distributed/membership/startup.py` (`worker_node_main`, the authority's start, the fingerprint, the tuned tactics); `tensorrt_llm/_torch/distributed/startup_record.py` (`StartupRecord`, `REPLACEMENT_CANNOT_START`); the hooks of `Distributed` in `tensorrt_llm/_torch/distributed/communicator.py` |
| Host collectives | `tensorrt_llm/_torch/distributed/host_exchange.py` (`MnnvlDist`, `HostExchange`, `HostExchangeGroup`); `tensorrt_llm/_torch/distributed/mnnvl_host_exchange.py` (`MnnvlHostExchangeGroup`, `HostExchangeMnnvlMemory`); native wait in `cpp/tensorrt_llm/nanobind/runtime/hostExchange.cpp` |
| Late-write remap | `tensorrt_llm/_torch/distributed/membership_mnnvl_memory.py` (`MembershipMnnvlMemory.rows_needing_fresh_pages`, `remap`); `tensorrt_llm/_torch/distributed/mnnvl_memory.py` |
| Masked, abortable all-to-all | `cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.{cu,h}`, `cpp/tensorrt_llm/thop/moe/communication/moeAlltoAllOp.cpp`; `tensorrt_llm/_torch/moe/fused_moe/communication/nvlink_one_sided.py`; `tensorrt_llm/_torch/moe/fused_moe/communication/execution_state.py` (`MoeExecutionState`, `MembershipWorkspace`); `tensorrt_llm/_torch/moe/fused_moe/ep_group_health.py` (`EPGroupHealth`) |
| Spare slots | `cpp/tensorrt_llm/runtime/moeLoadBalancer/unbackedSlotMemory.{h,cpp}`, `cpp/tensorrt_llm/runtime/moeLoadBalancer/hostAccessibleDeviceAllocator.{h,cpp}`, `cpp/tensorrt_llm/thop/moe/loadBalance/moeUnbackedSlotOp.cpp`; `create_spare_slot_storage` in `tensorrt_llm/_torch/moe/fused_moe/eplb_expert_restore.py`; `tensorrt_llm/_torch/moe/fused_moe/impl_blocks.py` (`uses_spare_slots`, `tuner_layout_kwargs`); `tensorrt_llm/_torch/moe/fused_moe/impl_contract.py` (`supports_spare_slots`, `spare_slots_rejection`) |
| Expert restore | `tensorrt_llm/_torch/moe/fused_moe/eplb_expert_restore.py` (`FaultTolerantMoeLoadBalancer`, `EplbExpertRestore`, `plan_layer_placement`, `DeploymentHostWeights`); `tensorrt_llm/_torch/distributed/membership/host_weights.py` (`host_weights_name`: the `/dev/shm` names derived from the shared secret and the node index); `MoeLoadBalancer::restorePlacement` and `abandonIterations` in `cpp/tensorrt_llm/runtime/moeLoadBalancer/moeLoadBalancer.cpp` |
| Executor failover and rejoin | `tensorrt_llm/_torch/pyexecutor/rank_failure_recovery.py` (`RankFailureRecovery`); `tensorrt_llm/_torch/pyexecutor/rank_failure_memory.py` (`StartupMemory`, `RecoveryMemory`); hooks in `tensorrt_llm/_torch/pyexecutor/py_executor.py`, `tensorrt_llm/_torch/pyexecutor/executor_request_queue.py`, `tensorrt_llm/_torch/pyexecutor/hang_detector.py`, `tensorrt_llm/_torch/pyexecutor/model_loader.py` and `tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py` |
| Node supervisor and launch | `tensorrt_llm/llmapi/_llmapi_node_supervisor.py` (`NodeSupervisor`); `tensorrt_llm/llmapi/_node_supervised_mpi_session.py` (one node, `trtllm-serve` and the LLM API); `tensorrt_llm/llmapi/_mpirun_job.py` (ends a failed job through `mpirun`); `tensorrt_llm/llmapi/trtllm-llmapi-launch` |
| Serving | `tensorrt_llm/executor/rank_failure_requests.py` (`RankFailureError`, `RANK_FAILURE_CODE`); `tensorrt_llm/serve/fault_tolerance_health.py` (`FaultToleranceHealth`, `FaultToleranceMetricsCollector`); `tensorrt_llm/serve/openai_server.py` |

## Process Model

```text
node 0                                            node i
┌───────────────────────────────────────┐        ┌──────────────────────────────┐
│ node supervisor                       │        │ node supervisor              │
│  ├─ rank 0 ── starts ─► authority ◄───┼─ TCP ──┼─ (liveness, records down,    │
│  │   (server, HTTP)    (store + decisions)     │    reports up)               │
│  ├─ rank 1 ... rank G-1               │        │  ├─ rank i*G ... (i+1)*G-1   │
└───────────────────────────────────────┘        └──────────────────────────────┘
      all ranks: MNNVL memory for the all-to-all and the host collectives
```

- Every rank is its own one-process MPI world (Open MPI's shared-memory transport off, `records.rank_env`), so a
  failed rank restarts alone. The launcher's MPI job only places the node supervisors.
- A rank's startup sets the hang detector's `TLLM_RANK_CRASH_HARD_KILL_GRACE` to -1
  (`startup._turn_off_rank_crash_kill`), so a crashed rank ends only itself and the membership removes it. A user
  value that keeps the kill on is replaced, with one warning that names the variable.
- The startup group runs over the rendezvous store (`StoreGroup` as `mpi_comm()`); the host collectives of
  `Distributed` run in `MnnvlDist` over MNNVL memory. No MPI or NCCL communicator spans ranks. Code that runs after
  the startup makes no collective on `mpi_comm()`: the startup group waits for every original rank, so a failed rank
  stalls it, and a replacement takes the original ranks' values only while it replays its startup.
- A rank ends with its node supervisor: every rank is started through `_exec_rank`, which arms the parent-death
  signal (SIGTERM) and refuses to start otherwise.
- When a node supervisor's connection closes, the authority counts the node's ranks as failed and gives them up. On
  one node, `NodeSupervisedMpiPoolSession` keeps rank 0 in the MPI pool and a node supervisor starts the other ranks.
- The authority runs without library or MPI initialization (`TRT_LLM_NO_LIB_INIT=1`, `MPI4PY_RC_INITIALIZE=false`)
  and hosts the rendezvous store on the rendezvous port. Its liveness listener binds a free port published under
  `LIVENESS_KEY`. Store values are signed with the shared secret (`TLLM_FAULT_TOLERANCE_SHARED_SECRET`), which also
  derives the hello token, the host-weight names and the pause-word names; the arrival counters are not signed.
  Values above 4 MiB, such as the startup record, are stored in chunks under `<key>/chunks/<i>`, below the 8 MiB
  value limit of torch's TCPStore server (`SignedStore.set`).
- A node supervisor unlinks its node's host weights and its ranks' pause words at the first `SHUTDOWN` record, while
  its ranks still map them, so the unlink only removes the names and the kernel frees the pages when the last rank
  ends. It also unlinks them on SIGTERM or SIGINT and when it stops the node's ranks (`_unlink_shared_memory`).

## Membership Protocol

### States

| State | Meaning | Ranks |
|---|---|---|
| `STARTING` | Ranks connect and start. Any failure aborts | Run startup collectives |
| `COMMITTED` | Membership is settled | Serve |
| `PROPOSE` | A change removes failed ranks and/or adds ready replacements | Stop, reply `READY` with where they paused |
| `INSTALL` | Every active rank applies the change from one resume state | Install, reply `INSTALLED`, or `FAILED` |
| `ABORTED` | The deployment cannot recover | Exit with status 1; `/health` fails; node 0's supervisor ends the `mpirun` job |
| `SHUTDOWN` | Rank 0 shuts down | Exit normally; a closed connection is no failure |

```text
COMMITTED ─failure or ready replacement─► PROPOSE(gen+1) ─all READY─► INSTALL(resume) ─all INSTALLED─► COMMITTED
                                             ▲                           │
                                             └── new failure ────────────┘   (restarts with the larger failed set)
any state ── rank 0 not in the new mask, or a survivor replies FAILED ──► ABORTED
```

### Who decides

- `Authority.tick` runs every report interval (`REPORT_INTERVAL_S`, 0.5 s) on the reports it holds. It is the single
  source of truth for membership. Ranks never decide membership locally.
- Disconnected: the rank's liveness connection closed (a crash closes it at once), or TCP keepalive gave up after
  `DEAD_TIMEOUT_S` (60 s).
- Hung: the rank is waited on (named by an active rank's fresh report, or has not replied in PROPOSE or INSTALL), is
  the end of a wait-for chain (`_chain_ends`), is not paused (below), and made no progress for `HANG_TIMEOUT_S`.
  Progress counts its iteration, install steps and the rounds its peers wait for it in (all-to-all round flags,
  host-collective epochs), so a long step whose rounds advance is not hung. A joining rank
  that waits on no peer gets `ADMISSION_TIMEOUT_S` (20 s). During the first startup the same rule runs with
  `RENDEZVOUS_TIMEOUT_S` (600 s) in place of the hang timeout: progress is the number of startup collectives
  completed, a rank that started waits on every rank that has not, host-collective rounds count as in serving, and a
  hung rank aborts the startup. A replacement's startup is timed apart from the rule: no progress (startup
  collectives, autotuner warmup steps) for 600 s and its node supervisor kills it (see [Rejoin](#rejoin)).
- Paused: outside a change, a rank in the paused mask of its node supervisor's latest report is not hung, even once
  its own reports stop, if that node report arrived within half the hang timeout. The executor pauses its hang
  detector (`HangDetector.pause`) around calls that may block longer than a hang timeout:
  rank 0's request fetch, the request broadcasts, a control action, and the profiler stop. A call that holds the GIL,
  as the profiler stop can while the trace flushes, keeps the rank's membership client thread from reporting, so the
  pause travels through shared memory instead: `HangDetector.disarm` and `checkpoint` write the rank's pause word
  (`RankFailureRecovery.start` shares it with `HangDetector.share_pause_word`), and the node supervisor reads every
  local rank's word and `/proc/<pid>/stat` every 0.5 s. A rank counts as paused while its word is set and its process
  is not stopped (state `T` or `t`), so a rank that left the pause, or that was stopped inside it, is hung as usual,
  with its stall starting no earlier than the latest node report that counted it as paused, also if its own reports
  stopped during the pause.
  A process that no longer exists does not count as paused; one whose state cannot be read for another reason (for
  example `hidepid`) counts as not stopped, and the supervisor warns once. Every rank pauses its detector during a
  change, so the pause does not count then.
  The cost is that of the paused hang detector: a rank that truly hangs inside a paused call is never removed as hung,
  and the ranks that wait on it stall until the hang detector of a waiting rank ends that rank. A process that dies is
  still removed when its connection closes or TCP keepalive gives up. A process frozen through its cgroup (for
  example `docker pause`) does not show as stopped, so it counts as paused while its word is set; only the main
  thread's state is read, so a single stopped thread other than the main one is not seen.
- `INSTALL` carries the resume state: the furthest paused iteration of the survivors, and the epoch after their
  latest paused epoch. If a newer PROPOSE replaced an INSTALL while some survivor had not reached its epoch, every rank
  resumes from that epoch again, so ranks that already installed it stay consistent (`_resume_epoch`).
- Rank 0 in the failed set ends in `ABORTED`. A survivor's `FAILED` ends in `ABORTED`. A joining rank's `FAILED`
  leaves it out through a new PROPOSE; there is no separate rejected outcome. The authority checks no placement: the
  ranks agree on it among themselves at INSTALL (see [Backend Contract](#backend-contract)).
- The authority sends each rank an empty line when it sent that rank nothing for `HEARTBEAT_INTERVAL_S` (1 s). A
  rank that hears nothing for `AUTHORITY_SILENCE_TIMEOUT_S` (30 s), or whose connection closes outside `SHUTDOWN`,
  exits, so the ranks outlast an authority stall of up to 28.5 s (the silence timeout less the heartbeat and report
  intervals). The authority's loss ends the deployment like rank 0's.

### What a rank does

`RankFailureRecovery` runs on every rank. Its membership client thread applies records; the executor loop does the
work at its resume point, where no forward and no EPLB iteration runs. Phases: `SERVING`, `STOPPED`, `INSTALLING`,
`RESUMING`, `JOINING` (`_PHASE_TRANSITIONS`).

| Event | Survivors | Joining rank |
|---|---|---|
| PROPOSE that removes ranks | Request the all-to-all abort and interrupt host-collective waits; drop the step; reply `READY` | - |
| PROPOSE that only adds ranks | No abort: the forward vote stops every rank at a step boundary; steps in flight complete | Its `joining_ready` report counts as `READY` |
| INSTALL | Host exchange over the active ranks at the resume epoch (fresh rows when a former member may still write); end EPLB iterations; retire the step in flight; queue interrupted requests for recompute; gather request ids and each rank's slots; plan the placement; shrink the KV cache by the restore bytes; remap the all-to-all workspaces; restore experts; rearm the all-to-all and install the mask; reply `INSTALLED` | Same path; enters the host exchange at rank 0's next epoch; receives rank 0's waiting queue |
| COMMITTED | Resume; fail requests that no active rank holds (`RankFailureError`); release the empty spare slots on a worker thread and give their memory back to the KV cache at voted step boundaries | Serve |
| Excluded or ABORTED | Exit (status 1); the node supervisor kills a process that does not | - |

The install runs inside the interrupted host collective, which keeps the executor loop free of rollback code:

1. A rank stops in host-collective round e. The interrupt handler takes part in the change from inside the wait
   (`_on_exchange_interrupt`, `_change_membership`).
2. At INSTALL the host exchange switches to the active ranks (`HostExchange.install`).
3. Round e is completed from the peers' rows, or run again over the new membership. Survivors are at most one round
   apart, so no rank undoes a result it already used.
4. The executor runs iterations without a forward (`pauses_forward`) up to `resume.iteration`.
5. At the first iteration past it, `before_iteration` calls `_finish_install`: gather, restore, `INSTALLED`, then
   the wait for `COMMITTED`.

Memory follows the same steps. At INSTALL, `RecoveryMemory.kv_cache_shrunk_by` suspends the KV cache pools and
resizes the GPU pool to the startup quota less the backed spare-slot bytes and the restore bytes. While the pages in
use do not fit, it preempts the newest recomputable requests (1, 1, 2, 4, ...); with none left, the rank replies
`FAILED`. After `COMMITTED`, a worker thread releases the empty spare slots while the rank serves; no token reaches
them. Once it is done, the pool grows back by at most 1 GiB per step boundary that the forward vote agrees on, because
resizing suspends requests and retiring the step in flight runs collectives. A restore and the teardown wait for the
worker first. Workspace rows remapped to fresh pages are owed
until the excluded process releases the old pages. Nothing announces that, so the pool grows into that memory when a
poll of free device memory, at most once a second, finds it free.

### Rejoin

1. The node supervisor starts a replacement on the same GPU, with the same command, once the predecessor ended and
   the membership no longer lists the rank. The replacement waits for a `COMMITTED` record without its rank before it
   loads, so the survivors have released the memory they imported from its predecessor. The authority assigns the
   next incarnation at the hello.
2. Isolated startup: the replacement starts with membership {itself}, replays the startup record (see below), checks
   the fingerprint (version and checkpoint `config.json`), adopts its predecessor's tuned tactics (each rank publishes
   its own under `tuned_tactics/<rank>`, because some MoE cache keys hold the rank's expert offset), and reattaches
   the host weights its rank wrote.
3. It reports `joining_ready`. The authority proposes adding it; admission uses the same INSTALL path as a failover.
4. A replacement that fails before it served 600 s, counted from its first `COMMITTED` listing, is restarted after a
   backoff (1 s, doubling). A replacement with no startup progress for 600 s, or not connected 600 s after it
   started, is killed and counts as such a failure. After 3 failures in a row (`_MAX_FAILED_REPLACEMENTS`) the rank
   is given up (`given_up_mask`). A rank is given up at once when its node supervisor left, when a replacement exits
   with `REPLACEMENT_CANNOT_START` (78), when a killed process is not reaped within 600 s, or when its GPU is still
   busy after 600 s (see step 5).
5. A lost node returns when the operator starts its node supervisor again in the same job and container, with the
   same shared secret, rendezvous address and settings (not node 0), through the launcher. On Slurm, for lost node
   `<i>`, the one-task step passes the supervisor's srun check (`SLURM_STEP_NUM_TASKS == 1`), and the launcher passes
   `--rejoin` to the supervisor:

   ```bash
   export TLLM_FAULT_TOLERANCE_SHARED_SECRET=<the deployment's secret>
   export TLLM_FAULT_TOLERANCE_RENDEZVOUS=<the deployment's host:port>  # and the deployment's other settings
   srun --jobid <job> --overlap -w <host of node i> -N1 -n1 --container-name=<the deployment's container> \
       trtllm-llmapi-launch --rejoin-node <i> --nodes <number of nodes> <the deployment command>
   ```

   The flags a step needs depend on the cluster: make sure the step sees the node's GPUs (`nvidia-smi -L`), all of
   its CPUs, and no memory limit lower than the node's, because the replacements write host weights to `/dev/shm`.

   The authority refuses a second supervisor for a node while the old one is still connected (`_node_refusal`), so
   the rejoining supervisor connects again for up to `DEAD_TIMEOUT_S` (60 s). It exits with status 1 when no
   authority admitted it within that time (the deployment ended, the old supervisor is still connected, or the
   node's ranks are not in the deployment), when the store's values are not signed with this node's shared secret
   (`UnsignedValueError`, at once), when the record is `SHUTDOWN` or `ABORTED`, when the deployment has not
   started, or when the record's rank count is not `--nodes` times the step's GPUs. Its report clears the node's
   given-up ranks, and their replacements rejoin in one membership change. A replacement of a rank that a previous
   supervisor started waits until no process runs on its GPU (NVML), polling every `_BACKOFF_S`, up to
   `RENDEZVOUS_TIMEOUT_S` (600 s), and the rank is given up after that. A GPU that NVML cannot check counts as a
   replacement that could not start (`_count_failed_replacement`), so no replacement starts unchecked.

### Startup record and replay

A replacement starts while the other ranks serve, so it cannot run the startup collectives with them, yet its CUDA
graphs and buffers must match theirs. Rank 0 records the result of each host collective of `MnnvlDist` in call order,
plus a few named values, and publishes the record with the fingerprint. A replacement replays the record instead of
running the collectives, checks at each call that its own payload equals its predecessor's, and exits with
`REPLACEMENT_CANNOT_START` (78) at the first divergence; it is not restarted then. The startup group (`mpi_comm()`)
is not recorded: a replacement's holds only itself. Five knobs follow from two facts, the replacement skips the KV
cache estimation and it runs a tuning pass of its own:

| Knob | Reason |
|---|---|
| `Distributed.rank_varying(name)` | Some payloads differ per process by nature: measurements (measured tactics, the autotuner warmup skip vote, the KV cache load sample) and process identities. The replay accepts a different own payload there, and still returns the recorded payloads, so every rank keeps the values its CUDA graphs and plans were built from |
| `Distributed.pause_startup_record` / `Distributed.resume_startup_record` | The KV cache estimation runs only on the original ranks, and the tuning pass only on a replacement, so their collectives cannot match call for call. They are neither recorded nor replayed. Pauses nest. The outermost resume on an original rank keeps the tactics tuned so far, which the rank publishes for its replacement |
| `Distributed.agree_per_process(name, gather, own=None)` | A value each process agrees once, at whichever startup step first needs it, such as the steady-clock offset, the KV cache budget vote, the FP8 context-MLA admission cap or a communicator split. The original ranks gather the steady-clock offset, the KV cache budget vote and the admission cap inside the estimation (paused); a replacement gathers them in its final executor. The replacement takes its predecessor's admission cap, as its own budget lacks the measured activation peak. The value is recorded and replayed even while paused. With `own=(index, value)`, this process's input to the gather, a replacement whose input differs from its predecessor's cannot start (`REPLACEMENT_CANNOT_START`): the communicator split's color and key, and the sentinel vote |
| `Distributed.agree_rank_capacity(name, rank, capacity, gather)` | The replacement keeps its predecessor's KV cache quota, because warmup shapes follow it. A replacement whose own capacity is lower fails, and a later restart may succeed once that memory is free. The host cache tier quota is the original ranks' minimum, replayed the same way: a replacement fails if the quota exceeds its RLIMIT_MEMLOCK (`REPLACEMENT_CANNOT_START`, not restarted) or its share of the node's available host memory (a later restart may succeed) |
| `reduce_op` of a min or max reduction | A replacement's value below the recorded minimum (a capacity) raises. A value above the recorded maximum (a failed status) takes part in the result, so the replacement fails with its own error. The record keeps each collective's reduction, so a replacement whose collective replays one with another reduction (min, max or none) cannot start (`REPLACEMENT_CANNOT_START`) |

## Backend Contract

`prepare_model` binds every MoE layer's `NVLinkOneSided` to one `MoeExecutionState`
(`bind_execution_state`) before weights load, and refuses a layer with another communication strategy, naming the
cause. A MoE backend declares `MoEStaticCapability.supports_spare_slots` (the Cutlass unquantized and
NVFP4 leaves and the CuteDSL NVFP4 grouped-GEMM leaf). While the load balancer adds spare slots
(`MoEDeployment.spare_slots_enabled`), MoE backend selection turns down every other backend with
`MoERejectReason.SPARE_SLOTS_UNSUPPORTED`, as it does for EPLB. A backend literal (`moe_config.backend`) then falls
back to a Cutlass leaf with the capability and logs one warning for all layers that names Wide-EP fault tolerance
and the reason; a pinned tactic, or a layer that no leaf with the capability serves, fails with the reason. A communication strategy
that lacks a property is rejected at model load with the reason.

| Property | Communication must | Implementation (`NVLinkOneSided`) |
|---|---|---|
| Membership by pointer | Read the active rank mask at execution time from a fixed device address, so captured CUDA graphs follow a change | `MoeExecutionState.active_rank_mask`, written in place by `install_mask` |
| Stop on abort | Poll the abort state inside every wait on a peer and stop waiting when it is set | `MoeExecutionState.abort` (pinned host memory) |
| Never consume an incomplete round | Mark a round that an abort cut short and never use its outputs | The kernels' incomplete-round word; `rearm` at INSTALL |
| No late writes after INSTALL | No write from a former member reaches memory a survivor reads | Late-write remap of the workspace (`MembershipWorkspace.remap`), specific to MNNVL fabric handles. An RDMA transport would revoke memory keys instead |
| Waited flags | Report the round flags it waits on, so a peer that still delivers rounds is not hung | Published in the abort state, reported by the rank through `waited_flags` |

| Property | MoE backend must | Implementation |
|---|---|---|
| Tune active slots only | Profile only the active slots (`tuner_layout_kwargs`) | Cutlass, CuteDSL NVFP4 grouped GEMM |
| Workspace independent of spare slots | Size workspaces by active slots, not every local slot | Cutlass, CuteDSL |
| Unbacked spare storage | Keep spare-slot weights in the storage that the shared weight creation allocates (`create_spare_slot_weights`, `create_spare_slot_storage`) | Shared code in `tensorrt_llm/_torch/moe/fused_moe/impl_blocks.py` |

| Backend | Lacks | Outcome |
|---|---|---|
| `NVLinkOneSided` + Cutlass, or + CuteDSL NVFP4 | nothing | Supported |
| TRTLLM-Gen | Tunes every local slot, spares included | Turned down in MoE backend selection (`SPARE_SLOTS_UNSUPPORTED`): a BF16 or NVFP4 layer falls back to Cutlass with a warning |
| DeepGEMM | Workspace rows for every local slot | Rejected in MoE backend selection (`SPARE_SLOTS_UNSUPPORTED`): no Cutlass leaf with spare slots serves FP8 block scales |
| CuteDSL with locality domains or SM107 kernels | Spare-slot tuning | Turned down in MoE backend selection (`EPLB_UNSUPPORTED`, `SPARE_SLOTS_UNSUPPORTED`): falls back to Cutlass with a warning |
| DeepEP and other communication without `bind_execution_state` | Mask, abort state, fence path | Rejected at model load (`moe_comms` in `tensorrt_llm/_torch/pyexecutor/rank_failure_model_load.py`) |
| MegaMoE (communication inside its own kernels, `comm` is None) | Kernels that honor mask, abort and remap | Rejected at model load |
| Any other quantization on Cutlass (FP8 block scales, W4A16 NVFP4, ...) | Spare-slot weight creation and tuning | Rejected in MoE backend selection (`SPARE_SLOTS_UNSUPPORTED`) |

Expert restore is the EPLB target's restore (`EplbExpertRestore`): `plan_expert_restore` then `restore_lost_experts`.
Every active rank plans from the same input, the slots each rank gathered after abandoning its EPLB iterations
(`agreed_placement`). `restore_lost_experts` compares the plan digest and workspace row size with the lowest active
rank's before it writes any slot. A rank that differs replies `FAILED`: a survivor's reply aborts the deployment, a
joining rank's leaves it out. Experts load from
the node's host weights, or are staged through the idle all-to-all workspace from a rank whose node holds them. With
static EPLB each node keeps the next node's experts (ring copies); with online EPLB every node keeps every expert.

## Invariants

A change must keep all of these. Each has a test listed under [Testing](#testing).

1. **With the flag off, the shared paths run the default path.** The fault-tolerance path is chosen once, when a
   module is built or bound, never per call. With the flag off, the per-step and per-layer paths add only:
   - One `is not None` check per site of an optional component (`rank_failure_recovery`, the deferred KV reuse
     commits), defaulted arguments that the default path ignores (`active_only=True` on the batch gathers,
     `tuner_layout_kwargs` is `{}`, a null abort state), and a slower attribute read per op call, since each op is
     read through an attribute that resolves it at first use, which CPython does not specialize. The KV reuse commit
     order and the attention-DP vote are those of the default path.
   - The NVLink one-sided all-to-all builds its default layout and kernels: rank-mask mode is on only with fault
     tolerance. Its kernels select `ENABLE_RANK_MASK=false`, and the prepare-combine and sanitize launches select
     the default kernels (`moeA2APrepareCombineKernel`, `moeA2ASanitizeExpertIdsKernel`). With an incomplete-round word in the layout
     (rank-mask mode), they launch `moeA2AIncompleteRoundPrepareCombineKernel` and
     `moeA2AIncompleteRoundSanitizeExpertIdsKernel`, which also take the rank and the word. In the kernel pointer
     structs the 4-word mask array becomes a mask pointer, an incomplete-round offset, an abort-state pointer and a
     reserved word, which keep the structs' size and the offset of `timeout_cycles` (static asserts); the dispatch
     and combine fences gain a `"memory"` clobber. The BF16 combine's reduce loop relies on all remote loads of an
     iteration issuing before the first use of their data; check the SASS of `moeA2ACombineKernel` after changing
     this loop.
     The ops take new optional arguments (`abort_state`, `barrier` on `moe_a2a_initialize`, `rank_mask` on the
     workspace layout), and the metainfo has one more field, 0 here.
   - `RankState` carries one more int, `num_in_transfer_requests`, in the attention-DP allgather: 0 except with
     `KVCacheAwareADPRouter` and `kv_cache_routing_account_for_in_transfer`. It only lets the fleet stop stepping
     while requests merely send KV cache.
   - One error: a rank that a node supervisor started (`TLLM_FAULT_TOLERANCE_RANK` set) with the flag off is
     rejected.
2. **Healthy FT on adds work to every all-to-all call and once per step.** Per call, the kernels load the rank mask
   and the incomplete-round word after the grid dependency wait. Once per step, `RankFailureRecovery.check_step` runs
   the incomplete-round check after the forward: one `fence.sc.sys` and one load of the word per workspace. A peer
   that may overwrite rows that a combine still reads first sets the reader's word and runs `fence.sc.sys`, so a
   reader-side `fence.sc.sys` that follows the step's last combine orders every combine of the step against that
   overwrite, and the combines themselves need no fence. The word stays set until `rearm`, so a round cut short
   outside a step's forward (warmup) is reported by the next step's check.
   Abort polls run only inside waits. The
   attention-DP forward vote is `RankFailureRecovery.forward_vote` in place of the default vote: one gather over the
   active ranks of the batch size plus two flags (a change not yet installed, a memory release). The host collectives
   run over MNNVL. Reports, heartbeats and status reads run on their own threads. A new per-step host step needs a
   measured budget.
3. **The authority is the only source of membership.** Ranks, node supervisors and the server follow its records; a
   rank never removes a peer on its own.
4. **Never consume an incomplete round.** A step whose all-to-all an abort cut short is not accepted (`accept_step`,
   `discard_failed_forward`): its stored outputs are rewound and its requests continue by recompute. Under FT the KV
   reuse commit of a context chunk waits until its step is accepted (`ContextCommits`), so a discarded step stores no
   reusable pages.
5. **No write from a former member reaches memory a survivor reads**, for the all-to-all workspaces and the host
   collectives alike. Safety never relies on an excluded process stopping: its writes land on remapped-away pages,
   its packets on segments nobody reads, its reports carry an old incarnation.
6. **Every rank resumes from the same state.** One resume iteration and epoch from INSTALL; one placement plan from
   one gathered input; restored placement agreed before any rank routes with it.
7. **CUDA graph addresses never change.** Masks, abort state and remapped memory are edited in place.
8. **Host collectives respect the active rank mask in one place.** Gathers stay indexed by rank (None or zero rows
   for an inactive rank), reductions and votes cover active ranks. Every host collective of `Distributed` used under
   FT is overridden in `MnnvlDist`; a direct `tp_comm` use fails under FT.
9. **A replacement reaches the original ranks' startup results.** Every startup host collective goes through
   `Distributed` and the startup record; a payload that differs by process is marked `Distributed.rank_varying`.
10. **Memory is borrowed, not ratcheted.** At every change the KV quota is derived afresh from the startup quota,
    less the backed spare-slot bytes and the bytes the change restores, so repeated failures do not lower it. It
    returns to the startup quota once the device memory is free again: the spare slots at the voted boundaries, the
    replaced workspace rows once the excluded process releases them. A startup check rejects a KV cache that cannot
    lend the largest recoverable failure's restore.
11. **Hang bound.** A hung rank is excluded about `HANG_TIMEOUT_S` plus one report interval after its stall starts;
    the all-to-all kernel timeout is at least twice the hang timeout (checked at startup), so no kernel times out
    first.
12. **Every new step in a shared path has a "nothing to do" fast path** and a feature-off test.

## Testing

| Layer | What it covers | Where | CI stage |
|---|---|---|---|
| Protocol, CPU | Authority decisions on simulated ranks, hang rule, store groups, startup group and record, host exchange, remap with a fake driver | `tests/unittest/_torch/distributed/` | `l0_cpu` |
| Launch and config, CPU | Node supervisor, rejected configurations, rank identity, model-load checks and binding | `tests/unittest/llmapi/test_node_supervisor.py`, `tests/unittest/llmapi/test_wide_ep_fault_tolerance_args.py`, `tests/unittest/_torch/pyexecutor/test_rank_failure_model_load.py` | `l0_cpu` |
| Serving | Status, `/health`, metrics, `rank_failure` responses, frontends | `tests/unittest/serve/test_fault_tolerance_health.py`, `tests/unittest/executor/test_base_worker.py` | `l0_cpu` |
| Recovery, CPU (mocked CUDA) | Spare slots and restore, ring host weights, executor recovery steps | `tests/unittest/_torch/moe/test_spare_slots.py`, `tests/unittest/_torch/moe/test_ring_host_weights.py`, `tests/unittest/_torch/pyexecutor/test_rank_failure_recovery.py` | `l0_cpu` |
| Single GPU | Abort and rearm, execution state, KV reuse under FT, KV quota resize | `tests/unittest/_torch/moe/test_moe_a2a_abort.py`, `tests/unittest/_torch/moe/test_moe_execution_state.py`, `tests/unittest/_torch/pyexecutor/test_rank_failure_kv_reuse.py`, `tests/unittest/_torch/pyexecutor/kv_cache/test_kv_cache_v2_gpu_quota.py` | `l0_b200` (MoE); the GPU stages that run `unittest/_torch/pyexecutor` |
| C++ | Unbacked slots, restore and abandon in the load balancer, host-accessible allocator | `cpp/tests/unit_tests/runtime/moeLoadBalancerTest.cpp`, `cpp/tests/unit_tests/kernels/moe/moeLoadBalanceKernelTest.cpp`, `cpp/tests/unit_tests/runtime/hostAccessibleDeviceAllocatorTest.cu`, `cpp/tests/unit_tests/runtime/virtualMemoryTest.cpp` | gtests |
| Multi-GPU, MNNVL | `MnnvlDist`, late writes after a remap, abort of a bound collective waiting on a stopped rank, staged restore, a rank starting alone | `tests/unittest/_torch/multi_gpu/test_mnnvl_dist.py`, `tests/unittest/_torch/moe/multi_gpu/test_moe_a2a_late_writes.py`, `tests/unittest/_torch/moe/multi_gpu/test_nvlink_one_sided.py`, `tests/unittest/_torch/moe/multi_gpu/test_ring_restore_staging.py`, `tests/unittest/_torch/multi_gpu/test_independent_rank_startup.py`, rank-mask cases of `tests/unittest/_torch/moe/test_moe_comm.py` | `l0_gb200_multi_gpus` |
| End to end | DeepSeek-V3-Lite on 4 GPUs under load and GSM8K clients; phases healthy, failover, degraded, admission, rejoined | `tests/integration/defs/test_wide_ep_rank_failure.py` | post-merge: every case |

Every end-to-end case runs on one node of 4 GPUs, in the post-merge CI stage. The cases cover a crash and the
rejoin of its replacement (`test_rank_failure_sequence[crash]`), a crash then a hang, a rejoin on each other target
(NVFP4 CuteDSL, NVFP4 Cutlass, BF16 with online EPLB), refused joins, a second failure during an admission or an
install, a second failure during a failover that the spare slots cannot cover (the deployment ends), rejoins when the
server spawns the ranks and through two frontends, the authority's loss, and shutdowns during an admission and after
a rank was given up. No CI test covers several nodes: the loss of a whole node under `mpirun --enable-recovery` and
its return were verified on DeepSeek-R1 over 16 GPUs (4 nodes) by running the node supervisor's `--rejoin`
directly. The `trtllm-llmapi-launch --rejoin-node` launcher option returned a lost node on DeepSeek-V3-Lite over 2
nodes (8 GPUs): every process of node 1 was killed, `trtllm-llmapi-launch --rejoin-node 1 --nodes 2` started its
supervisor again, and its 4 ranks were admitted in one membership change with every request complete.

```bash
pytest tests/unittest/_torch/distributed tests/unittest/llmapi/test_node_supervisor.py
pytest tests/unittest/_torch/moe/test_spare_slots.py tests/unittest/_torch/pyexecutor/test_rank_failure_recovery.py
# 4 GB200 GPUs in one NVLink domain, with an IMEX channel
cd tests/integration/defs
LLM_MODELS_ROOT=/path/to/models \
    pytest "test_wide_ep_rank_failure.py::TestRankFailures::test_rank_failure_sequence[crash]"
```

Rules for new tests:

- Every behavior change comes with a test that fails without it, and a GPU test for each behavior that touches the
  device, not mocks only.
- A new host collective in `Distributed` must be overridden in `MnnvlDist`, with a test in `test_mnnvl_dist.py`.
- A new startup collective whose payload differs by process must run inside `Distributed.rank_varying`. Otherwise
  every replacement exits with 78, which only a rejoin end-to-end test shows.
- A protocol change gets a case in `test_membership.py` driven through `Authority.tick` with simulated reports.
- Check that a test log says passed or failed, not that no tests ran: most multi-GPU tests skip without MNNVL fabric
  handles.

## Known Limits

- Node 0 is the whole control plane. Losing rank 0, node 0 or the authority, or an authority stall (a network or CPU
  stall on node 0) longer than 28.5 s, and always one of 30 s, ends the deployment.
- After a node loss the deployment serves at N-K, its ranks `given_up`, until the operator starts the node's
  supervisor again with `trtllm-llmapi-launch --rejoin-node` (same job, container, secret, rendezvous and settings).
- A second failure before the replacement rejoins is recovered only if the remaining spare slots hold every expert,
  the survivors' KV cache can lend the memory, and (static EPLB) the lost experts' ring copy survives.
- Repeated failures of a rank that served 600 s between them are replaced again and again; no failure budget over a
  window stops them. `trtllm_fault_tolerance_rank_restarts_total` shows the rate.
- Host work inside a step counts toward the hang timeout. A grammar compile for a new guide and user logits processors
  are not bounded by the batch; on rank 0 a false hang ends the deployment.
- KV cache manager V2 regions with a fixed slot count are not shrinkable. DeepSeek-V3 declares none; a model that
  does must take them out of the restore headroom.
- A node supervisor killed with SIGKILL leaves its node's host weights in `/dev/shm` until they are removed by hand.
  A startup that finds `/dev/shm` too small lists the same user's leftover segments (`_leftover_entries`).
- The control plane assumes a trusted cluster network. The rendezvous store (torch's TCPStore) and the liveness
  listener use plaintext TCP and accept any process that reaches their ports. The shared secret comes from
  `TLLM_FAULT_TOLERANCE_SHARED_SECRET`, so the launcher and the same user's processes can read it. It keys an
  HMAC-SHA256 tag over each store value and its key, so the ranks never accept, or unpickle, a value that a process
  without the secret made, nor a value from a deployment with another secret. The tags do not stop a process that
  reaches the port from stalling or ending the deployment: TCPStore has no authentication, and counters (`add`),
  deletes and waits are server-side operations with no value to sign. Replays fall within that reach. A process that
  captured the traffic can set a key again to an older value that this deployment signed for it (an older record or
  transaction, or a store collective's value from an earlier call), and the ranks act on stale membership. It can also
  reuse the hello token, which is the same for every liveness connection, to announce a rank or a node supervisor.
  The worst outcome is a failed recovery or the end of the deployment, as with a deleted key. Rejecting replays would
  take sequence numbers per sender that every reader checks. A replacement rank starts without them, so it would need
  a fresh, authenticated source (a handshake with the authority, or TLS). That is a protocol change, not a change of
  tags. The TCPStore that torch.distributed uses on the same network also takes `add`, deletes and waits without
  authentication. Run a deployment on a trusted network, with a secret of its own.
- One NVLink domain. More needs an RDMA all-to-all that revokes memory keys instead of the late-write remap, and a
  network host-collective transport behind `HostExchangeGroup`.

## Anti-Patterns

- **Do NOT check a backend's class in generic code.** Bind through `bind_execution_state` and declare
  `MoEStaticCapability.supports_spare_slots`.
- **Do NOT add a host collective that bypasses `Distributed`.** It hangs on a dead rank and a replacement cannot
  replay it.
- **Do NOT reallocate or recapture to follow a membership change.** Edit masks and state in place; remap at the same
  addresses.
- **Do NOT use a step's outputs before the step is accepted** (`accept_step`) when FT is on.
- **Do NOT add a second feature switch or an environment flag that changes behavior.**
  `enable_wide_ep_fault_tolerance` is the only feature switch; the shared secret and the rendezvous address are
  deployment settings.
- **Do NOT make the authority wait on a rank's executor.** It stays CPU-only and free of library initialization.
