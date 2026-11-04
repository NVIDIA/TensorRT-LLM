<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NCCL operation tracing

## Dispatcher-backed collectives

PyTorch records TensorRT-LLM's registered `allreduce`, `autotuned_allreduce`,
`allgather`, `allgather_list` and `alltoall_helix` operations in execution traces.
Their schemas and inputs provide communicator members, requested strategies,
split sizes where supplied, and tensor shape, dtype and size information.
These operations do not require `ENABLE_NCCL_PROFILING` and have no additional
`trtllm::nccl_metadata` child. ProcessGroup-backed operations likewise use
PyTorch's existing communication events.

A dispatcher operation is a logical collective, not proof of a particular
implementation. In particular, a requested `AUTO` all-reduce strategy does not
establish that NCCL was selected. Use the associated Kineto CUDA launch/kernel
evidence when available. Dispatcher metadata does not identify a selected native
strategy or symmetric-buffer fallback.

## Raw pipeline point-to-point operations

Configure with `-DENABLE_NCCL_PROFILING=ON` to add `trtllm::nccl_metadata`
PyTorch USER_SCOPE events around `NcclCommunicatorOp.send` and `recv`.
These raw NCCL calls have no equivalent dispatcher event. The option defaults
to OFF; `tensorrt_llm.bindings.BuildInfo.ENABLE_NCCL_PROFILING` describes the
build, not whether profiling is active.

See [the shared metadata contract](profiler-metadata.md) for process-start
`TLLM_PROFILE_CUSTOM_REGIONS` selection and the `TRTLLM1` envelope. With the
build option OFF, the P2P probes are compiled out. With it ON, inactive or
unselected scopes do not extract fields. Active metadata construction is lazy
and has profiling overhead. Native selector initialization remains at the
Python binding startup boundary.

The P2P metadata fields are:

- `nccl.operation`: `send` or `recv`.
- `nccl.group`: ordered global members of the actual world-sized communicator,
  `[0, ..., world_size-1]`, rather than a fabricated PP subgroup.
- `nccl.rank`: the participating global rank.
- `nccl.parallelism_dimension`: `pp`; `nccl.kind`: `p2p`.
- `nccl.src_rank` and `nccl.dst_rank`: actual communicator endpoints.
- `nccl.logical_bytes`: send or receive buffer payload, not wire traffic.
- `nccl.input_shape` or `nccl.output_shape`, and `nccl.dtype`.
- `nccl.selected_strategy`: `nccl_send_recv`.
- `nccl.cuda_graph_state`: `eager`, `capture`, or `unknown` when an observational
  stream query fails.

The instrumentation adds no communication, synchronization, sequence counters,
algorithm, protocol, channel or transport claims.

## Post-processing and capture limits

Keep MPI metadata enabled when reconstructing MPI communication: `mpi4py`
operations have no equivalent Torch dispatcher records. Its rank spaces,
roots, peers and tags follow [the MPI contract](profiler-metadata.md#mpi-operations).

For serial cross-rank collective matching, use aligned capture windows and
communicator membership. Matching requires complete coverage of the relevant
ordered operations; MPIDist annotations omit direct `_utils.mpi_*` calls and
background threads without an active profiler. Missing coverage must remain
unresolved. Where coverage is complete, validate
payload descriptors and one participant per member. Match P2P endpoints and
ordering, comparing payload descriptors when both endpoints record them. MPI
object receives do not record result payloads. These records do not identify
arbitrary duplicated
communicator contexts or resolve wildcard MPI receives. Missing or ambiguous
matches must remain unresolved.

Execution-trace `rf_id` and Kineto record-function IDs are process-local.
Associate a dispatcher with its local launch/kernel correlation only when
present; a metadata child's ID need not match the kernel's. Direct communicator
launches may lack usable kernel correlation. Host scopes describe enqueue
lifetime, not asynchronous completion or later NCCL errors.

CUDA graph capture records host operations once. Replay does not re-enter
these scopes; replay-only kernels cannot reliably be paired with a prior
capture's host records using these annotations alone.

The `TRTLLM1` fields appear as execution-trace inputs, not Chrome-export fields.
A PyTorch profiler session and execution-trace observer must cover the emitting
thread and the selected operations. See the [capture prerequisites](profiler-metadata.md#capture-prerequisites-and-output-formats).
