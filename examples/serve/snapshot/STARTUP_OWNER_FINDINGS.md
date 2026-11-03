<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Additional Native Owners at the Capture Boundary

Source-audit date: 2026-11-02. These findings extend the native state inventory;
they do not establish process-restore support, passing tests or upstream approval.

Audited revisions:
- TensorRT-LLM base B: `4fd6d1cc4521502a90f08df491af17ebac8e8fb1`.
- Native V2 candidate C: `b8b0ef930808b8e02ba4c99912b8e8f44d0c1145`.
- Snapshot S: `4b1300a82c6676df020c201fa884cc602251e12e`.

The initial experiment is dense text-only Qwen3, TP1 or single-node TP2, PP1/CP1,
without live user requests or external KV exposure. Capture precedes the original
final `PyExecutor.start_worker`. Resident capture retains weights, graphs, runner
storage and KV; compact C changes only admitted KV resources. These owner paths
are shared by B/C; C's KV additions do not reconstruct UB or IPC. [Candidate hooks][candidate]

## Explicit Profile Guards

These are owner guards, not a complete serving configuration. Keep KV policy,
graph buckets and capacities explicit, with identical cold/restored settings.

```yaml
tensor_parallel_size: 1  # Use 2 for the separate TP2 experiment.
pipeline_parallel_size: 1
context_parallel_size: 1
enable_attention_dp: false
orchestrator_type: null
num_serve_frontends: 1
num_postprocess_workers: 0
enable_resource_governor: false
speculative_config: null
max_beam_width: 1
disable_overlap_scheduler: true
allreduce_strategy: NCCL
torch_compile_config: null
```

`torch_compile_config: null` skips torch.compile, not decode CUDA graphs. For compilation,
set `enable_userbuffers: false` (default true); no compile `enabled: false` field exists.
Verify resolved settings on every rank. [Compile selection][compile] [Schema][args]

Set these environment values before importing TRT-LLM in the parent and workers:

```text
TLLM_NCCL_SYMMETRIC_ZERO_COPY=0
TRTLLM_PROXY_BATCH_REQUESTS=0
TRTLLM_NO_USAGE_STATS=1
TLLM_WORKER_USE_SINGLE_PROCESS=0
```

The zero-copy flag is cached at import. NCCL alone still permits output windows;
AUTO can construct custom/MNNVL resources. Ordinary NCCL remains. [AllReduce][allreduce]

For direct serve with a local MPI pool, leave `TLLM_DISABLE_MPI`, `TLLM_SPAWN_PROXY_PROCESS`,
`TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR`, `TLLM_EXECUTOR_ATTACH_INFO` and `TLLM_EXECUTOR_FRONTEND_ID` unset.
Do not supply external `_mpi_session` or inherit launcher attachments. Exclude profile-control
calls, KV publication/discovery, RPC/Ray, OpenEngine/disaggregated serving and tracing endpoints.
Use `kv_cache_config.kv_events_config.enable_kv_cache_events: false` and publisher
`"null"`, or omit the event config. [Executor selection][selection] [Discovery ordering][creator]

Pass `--num_serve_frontends 1`: the CLI defaults to eight, unlike `LlmArgs`. [Serving defaults][serve-default]
TP1-only `TLLM_WORKER_USE_SINGLE_PROCESS=1` can avoid the proxy when no earlier spawn/reuse
branch applies; qualify it separately. It cannot bypass TP2 spawning. [Selection][selection]

## TP Userbuffers and Collective Storage

`ModelEngine._init_userbuffers` rejects TP1, PP > 1, MPI-disabled execution, unsupported
multicast hardware and NCCL_SYMMETRIC; otherwise compilation can initialize UB. [Guards][ub-init]
The process-wide `UserBuffersManager.release_buffer` only marks an entry reusable,
not unregistered. [Manager ownership][ub-manager]
`UserBufferAllocator.deallocate` is empty, and native `destroy_communicator` is a
WIP empty body. These are not capture teardown/reconnect APIs. [Allocator][ub-allocator] [Native communicator][ub-native]

Native initialization duplicates raw `MPI_COMM_WORLD`, creates GPU pointer/flag storage
and may establish multicast mappings. Registration imports and maps peer allocations.
POSIX-FD exchange sockets close after import; peer handles/mappings remain. Fabric-handle
exchange is a separate build path. [Native ownership][ub-native]

For both policies, exclude UB/windows on every rank and use fresh processes; settings
do not erase an existing singleton. A UB extension needs peer-mapping/topology/registration
ownership and all-rank work retirement. Reconstruct before collective graph replay.
Tensor deletion, stable VA or CUDA synchronization alone cannot prove restored peer ownership.

## Decoder Runner Storage

`DecoderRunner` owns `_buffers`, metadata, input preparation and graph runners;
metadata receives views of the same persistent storage. [Runner ownership][runner]
`DecoderInputBuffers` includes token/position storage, previous-batch indices,
ragged-layout buffers and host bookkeeping even without speculation. Disabling
beam search/speculation removes only their conditional allocations. [Input owners][inputs]

`DecoderInputPreparer` retains pinned staging, cached indices and an input-copy event.
Zero HOST KV capacity does not mean zero pinned host state. The input-copy event precedes
model/graph work; it is not a whole-iteration barrier. [Copy ordering][forward] [Input event][input-event]
Model-declared packing/H2D-skip hooks add owners; require the effective admitted
model hooks to be off or inventory their storage explicitly. [Runner binding][runner]

Capture after profiling-resource teardown/final rebind, before worker startup. Join copies,
retire GPU work and hold further preparation while collecting rank agreement. Retain storage,
aliases and pinned-source lifetimes; compact KV must refresh indices without replacing
runner buffers. [Final ordering][creator] Validate initialized content, not arbitrary
`torch.empty` bytes; require graph replay without recapture, not only pointer equality.

## Proxy IPC and Startup Control

The proxy creates request/result/status queues before engine construction. `IpcQueue`
is `ZeroMqQueue`, defaulting to TCP loopback. Server bind is eager; client connection
and poller setup are lazy. HMAC authenticates but does not encrypt pickled messages.
Keep keys private and retain peer/routing state. [Proxy][proxy] [Queue implementation][ipc]

| Resource | State before final `start_worker` | Capture consequence |
| --- | --- | --- |
| HTTP socket | Bound before LLM construction; server construction follows it | Include the frontend FD; a bound address is not ready |
| Request/result queues | Proxy binds; rank 0 holds peer queue objects | Include both endpoints; some client connections remain lazy |
| Init-status ROUTER/DEALER | Worker identities and ACK precede backend construction | Proxy is still waiting for startup `READY` |
| Process monitor | Local-pool identities registered; pidfds or procfs fallback active | Qualify restored pidfds, poller and cached process identity |
| Stats RPC | Address/key selected; worker server starts after engine setup returns | Verify post-restore construction, not assumed pre-capture presence |
| Profile/governor queues | Lazy or configuration-dependent | Prove exclusion, not merely empty queues |
| Executor threads | Event loop, async sampler and sleep listener not started | Release all ranks into the same startup phase |

Sources: [Serving][serving], [worker][worker], [READY][proxy-ready], [monitor][monitor], [startup][start-worker].
The proxy calls `torch.cuda.Stream()` before submission; include its CUDA state, not only model ranks. [Submission][proxy-ready]

READY waiting has no overall deadline; failed delivery can warn and continue. Enforce an
external deadline. Startup messages are not fresh host-validation authority. Cached process
identity must match restored workers. These paths have no restartable queue capture/rebind API.
[READY protocol][worker] [Waiting and monitoring][proxy-ready] [Identity checks][monitor]

The classic path uses ACKs/polling/process monitoring, not an application heartbeat protocol.
Usage heartbeats start after LLM construction; this profile disables telemetry. [Usage][usage] [Opt-out][optout]
MPI/PMIx/PRRTE/NCCL/ZeroMQ control or heartbeat state is not thereby excluded: pin the
libraries and inspect their processes, sockets and deadlines.

## Single-Container Containment

A local `MpiPoolSession` constructs `MPIPoolExecutor`; helper placement is not guaranteed.
External sessions/pre-spawn launchers require separate qualification. [Pool construction][pool]
Snapshot traverses descendants across all threads, not atomic cgroup membership. Dump targets
the container root and externalizes its network namespace. [Traversal][tree] [Dump options][dump]
Snapshot preserves listener ports and pairs supported reciprocal TCP endpoints;
established sockets without an in-image peer are restored disconnected. This is
not a generic application reconnect contract. [Socket reconstruction][sockets]

Require private PID/network namespaces, all ranks/proxy/MPI helpers under the capture
root/cgroup, and both retained control endpoints in-image. Inventory pidfds, sockets,
shared-memory files and mounts. Keep the host supervisor outside the MPI failure domain;
terminate all source helpers before validation. One Docker does not itself prove containment.

## Proposed Qualification Tests

All tests are **NOT RUN**. Lifecycle, KV, communication and qualification owners must record actual outcomes.

| Case | Required evidence |
| --- | --- |
| Guards | Reject unexpected UB/window/MNNVL, extra frontends, external sessions or publisher owners on any rank |
| Reused process | Show that a previous UB singleton cannot evade an exclusion by changing config |
| Copy retirement | Delay H2D and later graph work independently; capture must wait for both |
| Retained storage | Reject nested alias replacement; replay admitted shapes without recapture |
| Compact KV | Keep runner storage while remapping KV and refreshing all dependent indices |
| Startup IPC | Restore the pending identity/READY exchange and complete private generation |
| Missing READY/peer | Deadline expires without serving-ready; whole-cohort cleanup is verified |
| Liveness | Exercise pidfd and procfs fallback; detect rank death without stale-identity success |
| Containment | Verify descendant/cgroup/peer coverage and source-helper termination; reject incomplete trees |
| TP2 failure | Restore real NCCL communication, inject rank failure and verify external supervisor survival |

## Immutable References

[candidate]: https://github.com/chienchunhung/TensorRT-LLM/blob/b8b0ef930808b8e02ba4c99912b8e8f44d0c1145/tensorrt_llm/_torch/pyexecutor/py_executor.py#L2522-L2582
[compile]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/model_engine.py#L455-L553
[args]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/llmapi/llm_args.py#L6566-L6608
[allreduce]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/distributed/ops.py
[selection]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/executor/executor.py#L594-L714
[serve-default]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/commands/serve.py#L1179-L1195
[ub-init]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/model_engine.py#L1523-L1557
[ub-manager]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/cpp/tensorrt_llm/kernels/userbuffers/userbuffersManager.cpp#L27-L80
[ub-allocator]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/cpp/tensorrt_llm/kernels/userbuffers/ub_allocator.cpp#L29-L64
[ub-native]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/cpp/tensorrt_llm/kernels/userbuffers/userbuffers-host.cpp#L126-L439
[runner]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/runner.py#L202-L254
[inputs]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/inputs.py
[forward]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/runner.py#L699-L786
[input-event]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/inputs.py#L2736-L2748
[creator]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/py_executor_creator.py#L1271-L1389
[proxy]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/executor/proxy.py#L603-L672
[proxy-ready]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/executor/proxy.py#L1016-L1137
[ipc]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/executor/ipc.py
[worker]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/executor/worker.py
[monitor]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/executor/worker_process_monitor.py#L30-L180
[start-worker]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/_torch/pyexecutor/py_executor.py#L1861-L1948
[serving]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/commands/serve.py#L700-L779
[usage]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/llmapi/llm.py#L1880-L1905
[optout]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/usage/usage_lib.py#L265-L285
[pool]: https://github.com/NVIDIA/TensorRT-LLM/blob/4fd6d1cc4521502a90f08df491af17ebac8e8fb1/tensorrt_llm/llmapi/mpi_session.py#L675-L714
[tree]: https://github.com/ai-dynamo/snapshot/blob/4b1300a82c6676df020c201fa884cc602251e12e/agent/internal/runtime/process.go#L408-L452
[dump]: https://github.com/ai-dynamo/snapshot/blob/4b1300a82c6676df020c201fa884cc602251e12e/agent/internal/criu/dump.go#L48-L79
[sockets]: https://github.com/ai-dynamo/snapshot/blob/4b1300a82c6676df020c201fa884cc602251e12e/agent/internal/criu/restore_images.go#L130-L304
