<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Wide-EP Fault Tolerance (Prototype)

A wide expert-parallel (Wide-EP) deployment spreads the experts of every MoE layer over all its ranks, so one failed
rank ends the whole deployment. With `enable_wide_ep_fault_tolerance`, the surviving ranks remove a failed rank,
restore its experts into spare EPLB slots from host weights (copies of the experts in `/dev/shm`), and keep serving at
reduced capacity with the same CUDA graphs. A replacement of the failed rank then starts on the same GPU and rejoins.
A rank fails when its process exits, or when it makes no progress for 60 s while other ranks wait for it.

```{warning}
Wide-EP fault tolerance is a prototype and is subject to change. CI tests it on one GB200 NVL72 node (4 ranks).
Several nodes and the loss and rejoin of a whole node were verified on DeepSeek-R1 over 16 GPUs, with no CI test,
by running the node supervisor's `--rejoin` directly. The `--rejoin-node` launcher option returned a lost node on
DeepSeek-V3-Lite over 2 nodes (8 GPUs), with every request complete.
```

## Requirements

These are the main requirements. Startup checks most of them and others, and rejects a configuration with the
reason, but not all of them: see [Limitations](#limitations) for what it checks and what it does not.

- GB200 NVL72 nodes with every GPU of the deployment in one NVLink domain, and an IMEX channel in the container (see
  the [Wide-EP example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep)).
- DeepSeek-V3 or DeepSeek-R1 (`DeepseekV3ForCausalLM`) as a BF16 or NVFP4 checkpoint with the Cutlass MoE backend, or
  NVFP4 with `backend: CUTEDSL`. With another MoE backend, backend selection falls back to Cutlass and logs a warning
  that names fault tolerance and the reason. Other quantizations, FP8 among them, are rejected when the model loads.
- Attention DP with tensor and expert parallelism over all ranks, `NVLinkOneSided` MoE communication, EPLB, KV cache
  manager V2, the overlap scheduler, MTP with a fixed draft length or no speculative decoding, no `torch_compile_config`
  and no piecewise prefill CUDA graphs (`prefill_cuda_graph_backend: piecewise`), and `enable_chunked_prefill` or a
  `max_seq_len` of at most `max_num_tokens`. Online EPLB (`layer_updates_per_iter` above 0) also needs CDMM mode and
  GDRCopy.
- No pipeline parallelism, beam search, LM head TP in attention DP (`enable_lm_head_tp_in_adp`), KV-cache-aware
  attention-DP routing or admission, KV cache events, KV cache connector or compression, NUMA binding of the host cache
  tier (`host_cache_bind_to_gpu_numa_node`), disaggregated serving, sleep mode, routed-expert capture
  (`enable_return_routed_experts`) or `modeling_v2: require`.
- Each node's GPUs given to one deployment. A restarted node supervisor starts a rank's replacement only once NVML
  lists no compute process on that rank's GPU. A GPU that NVML cannot check counts as a replacement that could not
  start.
- Room in each rank's KV cache pool for its share of a lost node's experts, which startup checks: a refusal names the
  GiB a rank needs and, when a `free_gpu_memory_fraction` of at most 1 would give it, that fraction; otherwise it
  asks for more ranks. For DeepSeek-R1 NVFP4 with `num_slots: 256`
  on nodes of 4 GPUs, the share is about 43 GiB on 2 nodes, 8 GiB on 4 nodes and 2.7 GiB on 8 nodes, plus the
  all-to-all workspace, so a deployment of several nodes needs at least 4 of them.

## Single node

```yaml
enable_wide_ep_fault_tolerance: true
enable_attention_dp: true
max_batch_size: 32
max_seq_len: 4096
cuda_graph_config:
  enable_padding: true
  max_batch_size: 32
kv_cache_config:
  use_kv_cache_manager_v2: true
moe_config:
  load_balancer:
    num_slots: 256  # one slot per routed expert of DeepSeek-V3 and R1; fault tolerance adds the spare slots
```

```bash
trtllm-serve <model> --tp_size 4 --ep_size 4 --config config.yaml
```

The LLM API takes the same option, and on one node neither needs a launcher. After the loss of K of N ranks, each
survivor's batch grows by about N/(N-K) (2x on 2 nodes after a node loss, 4/3 on one node of 4 after a rank loss). If
any survivor's batch exceeds `cuda_graph_config.max_batch_size`, every surviving rank decodes that step without CUDA
graphs, which lowers decode throughput, so size both `max_batch_size` values for the degraded
load.

## Launching on several nodes

1. On node 0, export `TLLM_FAULT_TOLERANCE_SHARED_SECRET`, a new hex secret of at least 32 bytes for each deployment,
   and `TLLM_FAULT_TOLERANCE_RENDEZVOUS`, a `host:port` of node 0 that every node reaches, with a free port outside the
   ephemeral port range. Keep the secret out of the command line, where other users can read it.
2. Size `/dev/shm` on every node for the host weights. With static EPLB, a node keeps the experts of one other node,
   about 46 GB for DeepSeek-R1 NVFP4 on 8 nodes. With online EPLB, it keeps every routed expert, about 368 GB.
3. On node 0's host, start `trtllm-llmapi-launch` once per node with `mpirun --enable-recovery` of Open MPI 5, which
   the release container provides. A launch by `srun` of several tasks is rejected, unlike the one-task step that
   returns a lost node (see [Returning a lost node](#returning-a-lost-node)). On Slurm, run `mpirun` in a
   `--no-requeue` allocation, since Slurm would start a requeued job as a new deployment. With
   `TLLM_FAULT_TOLERANCE_SHARED_SECRET` exported, the launcher runs a node supervisor on each node, which starts one
   rank per GPU and a replacement for each rank that fails. Without it, the launcher starts the command as without fault
   tolerance, with no node supervisor, and rejects `--rejoin-node`. Rank 0 also runs the membership authority, the process that decides which
   ranks belong to the deployment.

```bash
export TLLM_FAULT_TOLERANCE_SHARED_SECRET=$(openssl rand -hex 32)
export TLLM_FAULT_TOLERANCE_RENDEZVOUS=node0:29600
mpirun --enable-recovery \
    --host node0,node1,node2,node3,node4,node5,node6,node7 -np 8 --map-by ppr:1:node --bind-to none \
    -x TLLM_FAULT_TOLERANCE_SHARED_SECRET -x TLLM_FAULT_TOLERANCE_RENDEZVOUS \
    trtllm-llmapi-launch trtllm-serve <model> --host 0.0.0.0 --tp_size 32 --ep_size 32 --gpus_per_node 4 \
    --config config.yaml
```

To stop with a drain, send SIGINT to `trtllm-serve` on node 0, the PID that ends the launcher's line
`Rank0 task guard PID: <guard PID>; workload PGID: <trtllm-serve PID>`. A SIGTERM to `mpirun` stops without a drain,
and one to another node's supervisor counts as the loss of that node. Keep the rendezvous port on a trusted network,
since the store accepts any process that reaches it.

### Returning a lost node

After the loss of every process of node `<i>` (not node 0), the deployment serves without its ranks, whose
`replacements` read `given_up` in `/fault_tolerance_status`. Once nothing uses the node's GPUs (`nvidia-smi`), start
its supervisor again in a one-task `srun` step of the same job and container, with the deployment's secret, rendezvous
address and other settings. Its ranks then rejoin in one membership change:

```bash
export TLLM_FAULT_TOLERANCE_SHARED_SECRET=<the deployment's secret>
export TLLM_FAULT_TOLERANCE_RENDEZVOUS=<the deployment's host:port>  # and the deployment's other settings
srun --jobid <job> --overlap -w <host of node i> -N1 -n1 --container-name=<the deployment's container> \
    trtllm-llmapi-launch --rejoin-node <i> --nodes <number of nodes> <the deployment command>
```

The flags a step needs depend on the cluster: make sure the step sees the node's GPUs (`nvidia-smi -L`), all of its
CPUs, and no memory limit lower than the node's, because the replacements write host weights to `/dev/shm`.
The replacements inherit the step's environment, so run it in the deployment's container image (the same
`--container-image` and `--container-name`) and do not reset its environment: a replacement without the deployment's
tools and settings fails its warmup until the rank is given up.

`--rejoin-node` makes the node's supervisor wait for the membership authority and start only replacements of the
node's ranks, which load at once. Until the authority admits it, the supervisor connects again for up to 60 s, which
covers a previous supervisor of the node that is still exiting. It exits with status 1 in five cases. The rendezvous
store's values are not signed with this node's shared secret, so the secret differs from the deployment's. The
deployment has not started yet. The deployment has ended (shut down or aborted). The deployment's number of ranks is not
`--nodes` times the step's GPUs (check `--nodes` and the step's GPUs). No authority admitted the node within 60 s,
because the deployment ended, the node's previous supervisor still runs (check with `ps` and `nvidia-smi`), or the
node's ranks are not in the deployment (check `--rejoin-node`, `--nodes` and the step's GPUs).

## Requests, health and monitoring

- Requests in flight on the surviving ranks continue by recompute and keep their streamed tokens. A request in flight
  on a failed rank is ended, not migrated: if it streamed nothing, it runs again on a surviving rank, at most once.
  Otherwise it fails with HTTP 503 and the error code `rank_failure`, or a started stream ends with an error event of
  that code, and the client discards the output and sends the request again. On `/v1/messages` the response is
  HTTP 503 with the error type `overloaded_error` (not the 529 that Anthropic's API uses for that type), and a started
  stream's error event has that type.
- Requests with `n > 1`, `best_of > 1` or `use_beam_search` are rejected with HTTP 400.
- `GET /health` answers 200 while the deployment serves, degraded or not. Once the deployment cannot recover, the
  server starts shutting down after its next status refresh (once a second) and store read (up to 1 s), so about 2 s
  later, and longer when several store reads time out. Probe `/health`, not
  `/health_generate`, which can block during a pause. A hung rank pauses serving for about 60 s, so set client
  timeouts above 70 s.
- `GET /fault_tolerance_status` returns the `state`, `active_ranks`, `failed_ranks`, the failed ranks' `replacements`
  (`pending` or `given_up`) and `last_transaction`, the last membership change. `/prometheus/metrics` reports them as
  the metrics below, also without `return_perf_metrics`.
- No replacement of a `given_up` rank starts automatically. A rank is given up when its replacements failed 3 times
  in a row, when a replacement cannot start at all (exit status 78, for example because its TensorRT-LLM version or
  checkpoint differs from the deployment's), when its GPU stays in use or a killed replacement does not end for
  10 min, or when its node supervisor is gone. A node's ranks return to `pending` once its supervisor is started
  again. Until then, requests pinned to such a rank (`attention_dp_rank` without `attention_dp_relax`) fail with HTTP
  400. Requests pinned to a `pending` rank fail with HTTP 503 `rank_failure`. A streamed request has already
  received HTTP 200, so it gets an error event instead, without an error code for a `given_up` rank.
- A node's ranks share their node supervisor's standard output and error. When every rank logs each step (for example
  `TLLM_PROFILE_LOG_RANKS=all`) into a file on a shared filesystem that another node reads while serving, the writing
  ranks can stall on the file lock and every rank waits for them. Keep such logs on node-local storage.

| State | Meaning |
| --- | --- |
| `healthy` | Every rank serves. |
| `recovering` | The deployment is starting or a membership change is in progress, and serving may pause. A change that hangs keeps this state for up to 60 s. |
| `degraded` | Some ranks failed and the others serve until replacements rejoin, or an operator acts on `given_up`. |
| `stalled` | The ranks have waited 30 s or longer on a rank that makes no progress, and serving pauses. |
| `unprotected` | The rendezvous store has not answered yet, or not for 30 s, so a further failure is not recovered. |
| `failed` | The deployment cannot recover. Start it again from outside, for example by submitting the job again. |
| `stopped` | The deployment shut down without a failure. |

Every metric has the labels `model_name` and `engine_type`. The rank and replacement metrics appear once a membership
record was read, and `last_change_pause_seconds` once a membership change ended.

| Metric | Type | Value |
| --- | --- | --- |
| `trtllm_fault_tolerance_state{state}` | gauge | 1 for the current state, 0 for the others |
| `trtllm_fault_tolerance_stalled_seconds` | gauge | How long the active ranks have waited on a rank that makes no progress |
| `trtllm_fault_tolerance_active_ranks` | gauge | Number of active ranks |
| `trtllm_fault_tolerance_failed_ranks` | gauge | Number of failed ranks |
| `trtllm_fault_tolerance_replacements{state}` | gauge | Failed ranks whose replacement is `pending` or `given_up` |
| `trtllm_fault_tolerance_rank_restarts_total` | counter | Replacements that the membership admitted, over all ranks |
| `trtllm_fault_tolerance_membership_changes_total{kind}` | counter | Ended membership changes by `kind`: `failover`, `admission` or `abort` |
| `trtllm_fault_tolerance_last_change_pause_seconds` | gauge | Time from the detection of the last change's cause to its end |

## Limitations

- Startup rejects, with the reason, a configuration outside the requirements that it checks: the LLM arguments
  above, also EPLB `mode` other than `standard`, `warmup_placement` and context parallelism; expert and tensor
  parallelism over every rank; the model, its quantization and MoE backend; MNNVL support and fabric handles of the
  GPUs; the experts' fit in the KV cache pool and `/dev/shm`; the fence path of the NVLink one-sided all-to-all
  (`TRTLLM_NVLINK_ONE_SIDED_A2A_SYNC_PROTOCOL` not `sentinel`) with a timeout of at least 120 s; `TLLM_DISABLE_MPI`
  unset; the shared secret; on several nodes, a rendezvous address that is not loopback; a launch by
  `mpirun --enable-recovery` on node 0's host rather than by `srun` of several tasks (a warning only when the
  supervisor cannot read `mpirun`'s command line); and one node supervisor per node.
- Startup does not check GB200 (any aarch64 host is taken to share memory through fabric handles), one NVLink domain,
  IMEX or its channel in the container, a `--no-requeue` allocation, Kubernetes, or that a node's GPUs belong to one
  deployment. A wrong NVLink domain or a missing IMEX channel fails later, when the ranks map each other's memory. A
  rendezvous port in the ephemeral port range only logs a warning.
- Startup ends if a node supervisor has not connected to the membership authority within 60 s.
- Only process failures are covered. A node that crashes, loses power or is cut off from the network ends the
  deployment, as does the loss of rank 0 or its node, a failure of the membership authority, or a network or CPU
  stall on node 0 longer than 28.5 s (always one of 30 s or longer). Kubernetes is not supported.
- A node supervisor that stops reporting for over 30 s while connected ends the exemption for its node's ranks, so a
  rank of that node in a paused call that others wait on counts as hung. An idle rank 0 on that node is then removed
  one hang timeout (60 s) later, which ends the deployment. The authority logs a warning that names the node when
  its reports stop, and an info line when they resume.
- Fault tolerance does not detect a rank that deadlocks inside a call during which the executor's hang detector is
  paused, such as rank 0's wait for requests or the request broadcasts. As without fault tolerance, the ranks that
  wait on it stall until the hang detector of a waiting rank that is not paused ends that rank after
  `hang_detection_timeout`. A rank stopped inside such a call (SIGSTOP) is removed as hung once other ranks wait on it
  and the membership's hang timeout (60 s, not `hang_detection_timeout`) has passed since it stopped. A process
  frozen through its cgroup, as by `docker pause`, does not count as stopped, so one frozen inside such a call is not
  removed; only the state of the process's main thread is read, so a single stopped thread other than the main one
  is not seen either.
- A rank or node supervisor that reads nothing from the membership authority while more than 4 MiB wait to be sent
  to it is disconnected. The rank counts as failed; for a node supervisor, every rank of its node does.
- While degraded, the survivors give up part of their KV cache memory to the failed ranks' experts, so fewer requests
  fit. A further failure before a replacement rejoins is recovered only if the spare slots and host weights still hold
  every expert. Each membership change maps the all-to-all workspace on new device memory, and the KV cache pool
  gives that memory back once the processes that held the old pages have released them.
- After a failure, serving pauses until the survivors have removed the failed ranks, and their output throughput is
  lower while degraded, because each survivor carries more requests. A rejoin pauses serving again for one membership
  change.
- A replacement takes minutes to rejoin, and one with no startup progress for 10 min is killed and retried. A rank
  whose replacements fail 3 times in a row is given up. A failure before the deployment first serves ends it.
- Fault tolerance lowers throughput while no rank fails.
- Fault tolerance runs only the fence path of the NVLink one-sided all-to-all. With a BF16 combine, the default
  (`auto`) takes the faster sentinel path for small steps, so there the all-to-all is slower with fault tolerance on.
  With `use_low_precision_moe_combine`, both use the fence path.
- Measured recovery times, throughput while degraded and the cost while no rank fails are reported separately.
- With more than 32 ranks, two hung ranks 32 apart are excluded one hang timeout apart.
- Online EPLB with DeepSeek-R1 NVFP4 needs about 368 GB of `/dev/shm` on every node (24.8 MB per expert), and
  startup refuses a node with less, with the reason.
- Each node supervisor removes its node's host weights when the deployment shuts down, or at once when SIGTERM or
  SIGINT stops it, before its ranks end within 10 s. After a SIGKILL of a node supervisor, its host weights stay in
  `/dev/shm/moe_shared_*` (`<TRTLLM_EPLB_SHM_NAME>_*` when set), and its ranks' 4-byte pause words in
  `/dev/shm/trtllm_pause_*`, until removed by hand. Remove them before the next
  job, or give each job its own `/dev/shm`. When `/dev/shm` is too small, the startup refusal lists such segments of
  the same user.
