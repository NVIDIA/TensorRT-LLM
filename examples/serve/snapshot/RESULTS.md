<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qualification Status and Remaining Gates

Recorded 2026-11-02. **No native Snapshot profile is qualified.** Implemented
instrumentation, passing unit tests and a compiled adapter do not establish
process/CUDA restore or a startup improvement.

## Executed Checks

| Check | Observed result | What it establishes |
| --- | --- | --- |
| Pinned host-driver build | Go tests and executable build passed on Linux against Snapshot `4b1300a82c6676df020c201fa884cc602251e12e`. | The adapter uses compilable upstream interfaces; tested input, ownership, isolation and single-use checks work at unit level. |
| Real upstream-reference attempt | Failed before workload capture: the test cluster rejected privileged PageBroker creation. | This environment does not authorize the required host operations. No capture/restore or semantic output result exists. |
| CPU harness tests | See the PR Verification section for final-head counts across the probe and qualification suites. | Reporting, attempt/cohort validation, no-reload/replay checks, compact backing checks, endpoint validation, device-I/O sampling and cleanup contracts. Not GPU restoration. |
| Native TP1 and TP2 | Both returned `NATIVE_ONLY_PASS`, with `process_restore: NOT_RUN`. Each topology passed one matched cold/native-cycle trial per resident and compact policy on B200 GPUs. | Real private generation, output parity, per-rank graph replay and native compact release/materialization. These checks do not establish process restore. |

The successful reviewed adapter build produced image ID
`sha256:ba53d2a8761e4eab08b2058f5c3c7b45517475fbcdbe620383b030f0e2171bfe`.
Its build record binds the exact Go/Dockerfile hashes. The preceding reference
attempt used an earlier adapter build and failed at host setup, not in the
checkpoint executor. Do not attribute a runtime result to the reviewed build.

The native runtime candidate is TRTLLM
`b8b0ef930808b8e02ba4c99912b8e8f44d0c1145` with wheel SHA-256
`2d4a6a0cfc2c3b955e3700376a8aa884d7077ae73d95da743e93c4fd58553727`.
It is not the newer audit base. Use the complete matching dependencies and Python
package; installing its wheel without dependencies did not reach engine startup.

## Native-Only Evidence

The successful workload image is
`sha256:01123ba570c1bc575db7e4785f9309a46749316a0091f5681760eb5d6fad3ad7`.
It contains the matching dependency environment and offline FlashInfer assets.
The profile used Qwen3-0.6B, CUDA 13.4, NCCL 2.31.2 and PyTorch
`2.14.0a0+b2c75dd062.nv26.09`. The model weight SHA-256 is
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
Evidence records bind the executed runner, observer and private client to their
exact source hashes at implementation revision `4bfdc07061c6`. Subsequent changes
add host-operation I/O sampling and failed-attempt diagnostics. CPU tests cover
those changes; these historical GPU runs do not. The native observer is unchanged.
All owned workload containers were removed successfully.

| Topology | Policy | Cold launch to correct completion | Native-cycle launch to correct completion | Result |
| --- | --- | --- | --- | --- |
| MPI TP1 | Resident | 42.03 s | 43.25 s | PASS |
| MPI TP1 | Compact | 41.91 s | 45.77 s | PASS |
| MPI TP2 | Resident | 44.21 s | 44.49 s | PASS |
| MPI TP2 | Compact | 44.11 s | 44.51 s | PASS |

Each cell has one sample and includes initialization and instrumentation. A
native cycle starts a new server, prepares its KV state and materializes it
without process restore. These times are **not restored-startup latency** and
do not show a startup speedup or reliable tail latency.

Compact preparation reduced measured GPU KV backing from 492,830,720 bytes to
zero for TP1 and from 247,463,936 bytes to zero on each TP2 rank. Materialization
restored the original backing counts. It took 17.29 ms for TP1 and 16.88/16.86 ms
for TP2 ranks 0/1, excluding the subsequent state-observation cost. Weight state,
graph-visible addresses, graph inventory, runner buffers and KV state checks
passed. Fresh generation replayed graphs on every rank without repeating load,
post-load or graph capture. Artifact size reduction remains unmeasured.

Earlier failed attempts exposed incomplete runtime dependencies, missing offline
kernel assets and an observer that entered the profiling executor. The first
two were environment failures. The observer now follows the creator's explicit
profiling/final-executor designation across timer replacement; regression tests
cover that distinction. Failed attempts are retained, not counted as passes.

## Required Before Completion

| Gate | Current state and required evidence |
| --- | --- |
| Approved host | Blocked. Obtain an administrator-approved isolated Linux host or documented exception. Do not bypass the cluster restriction. |
| Upstream reference | Runner implemented; real capture/restore is unexecuted. Reproduce the pinned application's semantic and captured-token checks, preserving failed attempts. |
| Native TP1 and TP2 | Runner implemented; process restore is unexecuted. Require matched cold/resident/compact trials, source termination, output parity and real graph replay on every rank without reload or recapture. |
| State and communication closure | Source audit and initial guards exist. Runtime proof remains open for full process-tree/cgroup/socket-peer containment, pidfds, launcher state, graph-dependent indices and all retained aliases. |
| Negative qualification | CPU invalid-evidence cases and real-host ownership/stale-target cases are implemented. Real restored-rank abort, missing-peer, partial-restore and compatibility-mismatch fault experiments remain to implement and execute on the approved host. |
| Performance/footprint | Timing events, reconstruction/observer cost, file inventories, Linux device-I/O sampling and paired summaries are implemented. Native-only measurements are above; Snapshot artifact, device-I/O and restored-startup measurements are absent. Device counters are not artifact-exclusive; storage without a local block-device mapping needs a storage-specific counter binding. |
| Multi-node | Unsupported by the Snapshot pin. The README records the required upstream transaction, topology, communication and failure-test extensions; no simulated pass is allowed. |
| Host contract agreement | The adapter is our implementation, not an upstream supported SDK. Snapshot/runtime and security owners must review the pinned internal-interface and privilege contract. |

Keep this PR in draft until the applicable gates have evidence. The missing
experiments and storage-specific measurement gaps are part of qualification work, not silently
delegated to a later production-controller PR. Production request-drain/admission
APIs, external-transfer reconstruction and live-request migration remain separate
work as defined in the source audit.
