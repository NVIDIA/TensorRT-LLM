<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native NCCL profiler metadata

Configure with `-DENABLE_NCCL_PROFILING=ON` to add `trtllm::nccl_metadata`
PyTorch USER_SCOPE events around raw NCCL launches. This option defaults to OFF;
`tensorrt_llm.bindings.BuildInfo.ENABLE_NCCL_PROFILING` is a Python bool describing
the build, not whether profiling is active. The binding module explicitly
initializes the native selector at startup and translates malformed configuration
into an import error. No collective or static shared-library initializer parses
the selector.

See [the shared metadata contract](profiler-metadata.md) for process-start
`TLLM_PROFILE_CUSTOM_REGIONS` selection and the `TRTLLM1` byte-length-prefixed
literal envelope. The native payload supports strings, signed int64 values and
int64 lists. With this build option OFF, the launch-site includes, probes and
metadata extraction are compiled out. With it ON, inactive/unselected scopes
do not extract fields, and active callbacks inspect inputs only when they request
them. Enabled profiling has overhead; it is not a zero-cost path.

## Coverage and ownership

| Owner/path | Observed primitive | Dimension |
| --- | --- | --- |
| Raw `allreduce`, NCCL strategy | `ncclAllReduce`, sum | `tp` |
| Raw `allreduce`, symmetric strategy | `ncclAllReduce`, sum; registered-buffer fallback is observed | `tp` |
| Raw `allgather` and list variant | `ncclAllGather` or grouped `ncclBroadcast` | `unknown` |
| Raw `alltoall_helix` | grouped `ncclSend`/`ncclRecv` | `unknown` |
| `NcclCommunicatorOp.send`/`recv` | runtime raw NCCL point-to-point | `pp` |

Generic all-gather serves both TP and CP; Helix accepts arbitrary groups. Their
unchanged schemas do not convey a model parallelism dimension, so `unknown` is
intentional. Point-to-point's current owner is pipeline parallelism, but its
communicator is world-sized: group records actual world membership, not a
fabricated PP subgroup.

ProcessGroup-backed all-gather and all-reduce do not get this custom event;
PyTorch's communication events remain authoritative. UserBuffer, alternate
all-reduce kernels, native Helix kernels, other transports and engine plugins
are not labeled NCCL by this feature. There are no additional collectives,
sequence counters, backend/strategy changes or synchronizations.

## Fields

- `nccl.operation`: `all_reduce`, `all_gather`, `all_to_all`, `send` or `recv`.
- `nccl.group`: ordered global ranks in the actual communicator. Collective
  owners use their sorted rank set; point-to-point uses `[0, ..., world_size-1]`.
- `nccl.rank`: the participating global rank, not its index in a subgroup.
- `nccl.parallelism_dimension`: `tp`, `pp` or `unknown` as described above.
- `nccl.kind`: `collective` or `p2p`.
- `nccl.logical_bytes`: local input payload (sum across input lists), or the
  receive buffer payload. This is not wire bytes, protocol overhead or total traffic.
- `nccl.strategy` / `nccl.selected_strategy`: observed primitive strategy,
  `NCCL`, `NCCL_SYMMETRIC`, `nccl_allgather`, `nccl_broadcast` or `nccl_send_recv`.
  Symmetric all-reduce reports `NCCL_SYMMETRIC` only when both buffers are
  registered; partial or complete fallback reports `NCCL`. P2P provides only
  `nccl.selected_strategy`.
- `nccl.requested_strategy`: logical `all_gather` or `all_to_all` for their
  wrappers. All-reduce's original request remains in the dispatcher arguments.
- `nccl.reduce_op`: `sum` for raw all-reduce.
- `nccl.input_split_sizes` / `nccl.output_split_sizes`: all-gather split sizes
  along native dimension zero when supplied; for Helix, input element counts
  in list-major, peer-major order (one entry per tensor, not per byte).
- `nccl.src_rank` / `nccl.dst_rank`: P2P communicator endpoints.
- `nccl.input_shape` or `nccl.output_shape`, and `nccl.dtype`: P2P tensor shape
  and scalar type.
- `nccl.cuda_graph_state`: `eager`, `capture`, or `unknown` when a stream is
  available. A failed observational stream query reports `unknown`; no
  capture-state field is added without a stream.

Algorithm, protocol, channel count, transport and profiler-only sequence state
are intentionally absent. Nothing at these observation points proves them.

## Capture and correlation limitations

Events describe host calls, including calls made during CUDA graph capture.
Graph replay does not re-enter those scopes and emits no new custom metadata.
Replay-only kernels cannot reliably be matched to a previous capture's helper.
The scope measures host enqueue lifetime, not asynchronous GPU completion or
later NCCL failures. Grouped scopes aggregate all operations in their input list.

The enclosing dispatcher is the logical operation; this event is a metadata
child, not a second collective. A helper's correlation ID need not match a
kernel's. Execution-trace parentage/containment may identify the dispatcher;
use its launch correlation only when present. Direct communicator launches can
lack usable kernel correlation. Aligned windows and identical launch order are
required for cross-rank interpretation; missing evidence must remain missing.

## Validation

The native `profilerTest` target verifies encoding and lazy input demand in OFF
and ON configurations. The local/QA test
`tests/unittest/_torch/distributed/test_nccl_profiler.py` uses the existing
two-worker MPI fixture with real CUDA devices. In fresh processes, run the OFF
build with selector `nccl`, then the ON build with selector `none`, `nccl` and
unset. It checks exact rank-aware numerical results, raw metadata, eager/capture
state, two replay outputs with no new events, P2P endpoints and ProcessGroup
exclusion. The existing CPU CI list includes its parent directory, so the
CPU-only BuildInfo test is collected and the two-GPU cases are deselected there.
The hardware cases are explicit local/QA coverage until included in a suitable
multi-GPU list.
Set the test-only `TLLM_NCCL_PROFILER_TEST_TRACE_DIR` to retain both ranks' eager,
capture and replay execution traces outside the source tree.

Full native linkage, supported Torch/CUDA/NCCL execution and hardware coverage
are separate requirements; CPU metadata tests do not establish those gates or
production-ready kernel correlation.
