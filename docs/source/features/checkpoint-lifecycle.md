<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Experimental Resident Checkpoint Lifecycle

This prototype separates execution-state restoration from serving readiness.
A host checkpoint tool restores the process and device state. TensorRT-LLM must
still validate its resources and participant state before it accepts requests.
The engine operations do not implement a host checkpoint or restore driver.

## Scope

The first implementation targets one host, MPI TP1 or TP2, dense Qwen3, PP1/CP1,
ordinary NCCL and one HTTP frontend. It requires final warmed CUDA graphs,
KV Manager V2, GPU-only KV storage and disabled prefix reuse. Startup rejects
external KV transceivers, connectors and lenders, auxiliary resource owners,
resource governance, active or scheduled profiling and optional collective windows. An empty request queue
does not prove that external access is absent.
The first restore attempt must retain the recorded host and device identities.
Relocation to another node or device set requires a separately qualified profile.

This is a resident capture path. It does not call sleep, release weights or
release KV backing. It does not require GMS. Native memory release/resume cannot
clear the independent checkpoint admission gate.

## Enable

Set `TRTLLM_RUNTIME_CONTROL_API_KEY` through your existing secret mechanism. Use
`--enable_checkpoint_mode` with `trtllm-serve`. The prototype requires these
explicit environment settings on the frontend and workers:

```text
TLLM_NCCL_SYMMETRIC_ZERO_COPY=0
TRTLLM_PROXY_BATCH_REQUESTS=0
TRTLLM_GPU_KEEPALIVE=0
```

Select `allreduce_strategy: NCCL`, `num_serve_frontends: 1`,
`kv_cache_config.use_kv_cache_manager_v2: true`, and
`kv_cache_config.enable_block_reuse: false` in the serving configuration.
Set `kv_cache_config.host_cache_size: 0` explicitly: its automatic default can
create an unqualified host tier. Do not enable disk KV tiers. No `sleep_config`
is required.

## Operations

All routes require runtime-control HMAC authentication, including status reads.
Use `build_runtime_control_auth_headers` from
`tensorrt_llm.serve.runtime_control_auth`. Each transport retry needs a fresh
nonce, but retains the same operation ID and payload. The operation journal
returns the existing result and never repeats mutations or extends the deadline.

| Route | Required input | Result |
| --- | --- | --- |
| `GET /checkpoint/status` | Signed request | Source identity, profile/cohort, version, stage, readiness, last result |
| `POST /checkpoint/prepare_capture` | Source context; natural drain; resident policy | All-rank protected boundary and drained HTTP responses |
| `POST /checkpoint/validate_runtime` | Fresh restore context; source identity; prompt and reference token IDs | Memory checks, private generation, graph replay and retired validation work |
| `POST /checkpoint/activate` | Current readiness ID and authenticated final host-success attestation | All-rank activation, followed by public admission |
| `POST /checkpoint/abort_operation` | Current context and reason | Stops progression; admission stays closed; host cleanup remains unconfirmed |

Every mutation context has protocol version `1`, operation ID, session ID and
kind, immutable template ID, profile/cohort digests, expected state version and
a relative time budget. Read the current version and resolved digests from
status. Do not calculate them from requested configuration alone. Each worker
accepts one source capture and one fresh restore binding in this prototype.
Shared settings must agree across ranks. Resolved GPU capacity budgets can differ
by rank; the profile still binds every rank's budget, and resident validation
requires each budget to remain unchanged.

```text
Public serving
  -> prepare_capture: close ingress; drain all requests; park all ranks
  -> capture_safe: retain resident storage and scheduler protection
  -> external host capture and restore
  -> validate_runtime: bind a fresh destination session
     -> memory-ready: compare resident owners and graph-visible storage
     -> run private generation; require warmed graph replay; drain again
     -> runtime-ready: return a current readiness ID
  -> activate: verify readiness ID and matching host-success attestation
     -> collect every rank's prepare/commit acknowledgement
     -> serving-ready: open public ingress
```

The private probe uses the normal executor submit and result path. It receives
one internal, single-use permit. Public requests cannot obtain this permit.
Generation must match reference tokens and replay a warmed graph on each rank.
The engine then retires the probe and checks retained storage again.

The host-success attestation is a statement from the authenticated external
controller. The engine does not independently inspect CRIU or host-driver logs.
The controller must verify final host success and bind it to the destination,
artifact, profile and cohort. A restored marker file or `/memory_status` response
is not that evidence.
Before restore operations, the host integration must establish current controller
authority and deal with captured credentials and replay state. This prototype
does not implement credential rotation or prove that retained transport state is
safe to clone. These remain production host-integration gates.

## Failure And Ownership

After preparation starts, timeout, missing participant evidence or validation
failure keeps admission closed. Stale commands cannot advance the lifecycle.
Abort does not resume the source. It does not prove
host cleanup, DMA retirement or safe memory reuse. The external supervisor
must remain outside the worker job's effective failure/abort scope.

HTTP control operations remain reachable while inference is blocked. Public
health and generation endpoints return 503 during the protected lifecycle.
Already accepted HTTP responses, including streams, must finish before capture
safety is returned. Their preprocessing and generation drain before the native
queue closes. Keep the controller connection and capture boundary under
the host tool's application contract; this PR does not implement socket handling.

## Acceptance And Remaining Dependencies

| Requirement | Implementation or remaining gate |
| --- | --- |
| Persistent serving gate independent of memory readiness | HTTP ingress and native request queue |
| All affected ranks, bounded drain, exact retries and fresh restore identity | Fixed MPI cohort protocol and operation journal |
| Private generation with final graph storage retained | Native resident inventory, replay hook and normal executor submit |
| Unsupported external exposure | Explicit rejection, not an idle-queue inference |
| Compact capture and extended KV metadata reconstruction | Separate native memory/KV integration; not implemented here |
| Transfer-bearing and disaggregated profiles | Require lifecycle evidence for publication closure, transfer retirement and peer reconstruction; not implemented here |
| Production host contract and process restore qualification | External driver binding, fresh authority/secrets, socket/transport treatment and actual restore tests remain required |
| Multi-node, Ray, topology changes and repeated capture after activation | Not implemented by this first fixed-cohort prototype |

The native TP1/TP2 test sends a simulated host notification without destroying
or restoring the process. It tests engine lifecycle behavior, not Snapshot
compatibility, startup speedup, credential rotation or clone safety. Do not use
that result to claim end-to-end checkpoint/restore qualification.
