<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase 0: Native Startup Capture Boundary

The prototype targets a clean, resident PyTorch template. Capture each MPI rank
after model warmup and before `PyExecutor.start_worker()`. No inference endpoint
has accepted requests at this point. Preserve CUDA-graph dummy allocations;
an empty user-request set does not imply an empty KV allocation map.

The external CPU coordinator must remain outside the captured process tree.
Native MPI workers, their launcher and IPC resources must be inside that tree.
The first host adapter uses the same Linux host, PID/mount/network namespaces,
filesystem and GPU assignment. It is not a portable container-image format.
Run it only in a dedicated test environment with checkpoint privileges.

Snapshot source pin: [`bc9d2161d2a7`](https://github.com/ai-dynamo/snapshot/tree/bc9d2161d2a7f203551e5b7e237defd90b1e9aa1).
Its `snapshotctl` still needs Kubernetes. The prototype instead uses its real
`cuda-checkpoint-helper` lock/checkpoint/restore/unlock commands and CRIU.
Multi-GPU shared mappings additionally require the matched cuinterpose stack.
The Snapshot agent prepares interposed mappings before CUDA lock, restores CUDA
before interposed mappings, and terminates a partially modified source on error.

## Acceptance, Not Assumptions

- Terminate the complete source tree before restore. Verify that restored
  execution crosses the saved boundary without loading weights or rerunning
  warmup. Compare fresh deterministic generation with cold startup.
- Require memory-ready, runtime-ready and serving-ready separately. A private
  validation request must not make the candidate generally available.
- Record model/configuration identity, runtime binaries, host/GPU identity,
  process membership, rank evidence and artifact hashes. Reject mismatches.
- Keep failures closed. Timeouts do not authorize memory reuse or continuation
  of a partly checkpointed source. Do not count cold fallback as restore.
- Measure capture and restore separately. Publish measured speedup only after
  repeated successful trials of the exact model and topology.

## Phase 2 Investigation Boundaries

KV validity and communication validity are separate checks. Preserve the clean
KV manager's allocations and metadata; verify graph-required addresses and
reserved dummy ownership. This is not a general live-cache reset API.

Qualify full-cohort single-node MPI first. Multi-node process-tree packaging,
external transport reconstruction, P/D discovery, late-transfer retirement and
survivor membership require additional implementation and tests. Reject these
profiles until supported. They must not inherit a passing single-node result.
GMS, live-request migration, KV migration and interrupted-transfer resumption
remain deferred. Fault detection, partial-rank rejoin and elasticity belong to
the separate fault-tolerance effort.
