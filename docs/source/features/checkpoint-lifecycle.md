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
The prototype compares recorded hostname and device UUID observations, but these
are not fresh destination evidence: a process restore can retain the source UTS
namespace and cached device properties. The host integration must independently
validate the destination's compatibility. The current engine guard can also
reject a different GPU UUID; relocation is not supported or qualified by this
prototype. A production relocation contract must replace this fixed-cohort guard,
not simply treat a matching restored hostname as proof.

This is a resident capture path. It does not call sleep, release weights or
release KV backing. It does not require GMS. Native memory release/resume cannot
clear the independent checkpoint admission gate.

## Enable

Set `TRTLLM_RUNTIME_CONTROL_API_KEY` through your existing secret mechanism. Use
`--enable_checkpoint_mode` with `trtllm-serve`. This opt-in mode sets the following
environment overrides on the frontend and workers, including when the parent
environment enables these features. For pre-launched MPI ranks or an embedded
Python process, export these values **before importing TensorRT-LLM on every
rank**. Changing an environment variable after import does not reset a cached
NCCL setting; startup rejects an effective zero-copy setting that is still enabled:

```text
TLLM_NCCL_SYMMETRIC_ZERO_COPY=0
TRTLLM_PROXY_BATCH_REQUESTS=0
TRTLLM_GPU_KEEPALIVE=0
```

Native lifecycle opt-in is scoped to the engine's `env_overrides` mapping, not
an inherited `TRTLLM_ENABLE_CHECKPOINT_LIFECYCLE` value from an earlier LLM.
The CLI supplies this private opt-in. It does not enable sleep or a host driver.
Loaded standard fast tokenizers have a content-based profile identity. Custom
tokenizer wrappers, runtime checkpoint loaders and custom logits/postprocessing
hooks are outside this first profile and are rejected.

Select `allreduce_strategy: NCCL`, `num_serve_frontends: 1`,
`kv_cache_config.use_kv_cache_manager_v2: true`, and
`kv_cache_config.enable_block_reuse: false` in the serving configuration.
Set `kv_cache_config.host_cache_size: 0` explicitly: its automatic default can
create an unqualified host tier. Do not enable disk KV tiers. No `sleep_config`
is required.

## Operations

All `/checkpoint/*` routes require runtime-control HMAC authentication, including status reads.
Use `build_runtime_control_auth_headers` from
`tensorrt_llm.serve.runtime_control_auth`. Each transport retry needs a fresh
nonce, but retains the same operation ID and payload. The operation journal
returns the existing result and never repeats mutations or extends the deadline.
An exact retry can return HTTP 200 with `status: in_progress` while the original
operation still runs. HTTP 200 alone is not capture permission. Wait for the
same preparation operation to report `status: completed`, then read authenticated
`GET /checkpoint/status`. Before capture, verify the current session, template,
profile and cohort identities and require `readiness.capture_safe=true`.
A cached completed result does not replace this current readiness check.

Conflicting identity, version or transition order returns HTTP 409. An operation
that records `status: failed` returns HTTP 500 with its complete `CheckpointResult`,
including cleanup requirements. Exact retries return the same recorded failure
without repeating native work. An unrecorded runtime exception returns 500;
new progression requests after a failed or aborted attempt also return 500.
A budget rejected before operation acceptance returns 504. A timeout after
acceptance records a failed operation and returns 500, including an HTTP drain
timeout before native dispatch. Neither response authorizes admission or resource
reuse. Use authenticated status reads to inspect the current attempt.

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

`prepare_capture` starts the source budget. It covers HTTP/native drain and the
wait for host capture, not just the preparation RPC. After expiry, current
`capture_safe` is false even when the recorded preparation result is completed.
Status exposes `remaining_seconds` as a local observation, not a durable host
deadline. The external controller must budget capture and check current status.

`validate_runtime` starts the destination's shared deadline. That deadline also
limits `activate`; a later activation budget can shorten it, but cannot extend
it. Budget for validation, any wait for final host attestation, and activation
together. If this deadline expires before an otherwise valid new `activate`
request is accepted, the request returns HTTP 504 and admission stays closed.
Retries do not restart this deadline.

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
Before `prepare_capture`, select the probe's prompt token IDs and generate its
reference with the same model and resolved profile. Choose a reference length
`N` from 2 through 64 and use the same sampling settings as private validation:

```python
SamplingParams(temperature=0, max_tokens=N, ignore_eos=True, end_id=-1, pad_id=0)
```

Store the exact generated token IDs as `expected_token_ids`, and reuse the same
prompt as `prompt_token_ids`. Validation sets `max_tokens` to the reference
length and applies the normal LLM generation-config defaults, as `LLM.generate`
does for the reference. Prompt/reference token IDs and configured length limits
are checked before a restore attempt is bound. Different sampling or EOS settings can cause a mismatch and a terminal
validation failure; admission then stays closed.
Choose a probe whose decode graph shapes were already warmed before preparation.
Validation must not create a new graph or alter runner storage. The inventory
does not claim to validate warmup timing metrics, which startup can clear.
A lazy capture during validation fails the resident-inventory check; it does not
qualify warm reuse. Adding an architecture to the native allowlist requires this
profile's ownership, retained-storage, graph-replay and token-parity tests.

The host-success attestation is a statement from the authenticated external
controller. The engine does not independently inspect CRIU (Checkpoint/Restore
In Userspace) or host-driver logs.
The controller must verify final host success and bind it to the destination,
artifact, profile and cohort. A restored marker file or `/memory_status` response
is not that evidence.
Before restore operations, the host integration must establish current controller
authority and deal with captured credentials and replay state. This prototype
does not implement credential rotation or prove that retained transport state is
safe to clone. These remain production host-integration gates.

## Failure And Ownership

Checkpoint preparation failure keeps admission closed and `capture_safe=false`.
A proven pre-dispatch failure does not by itself invalidate the native runtime.
Returning to service requires an explicit, qualified controller action; it is
never an automatic timeout fallback.

The current implementation distinguishes two failure paths:

| Failure boundary | Current behavior |
| --- | --- |
| Preparation budget expires before the backend runs, or HTTP drain/budget expires before native RPC dispatch | Record `status: failed` and `native_prepare_not_dispatched: true`. Keep HTTP admission closed. Do not send native revocation solely for this failure. Without a concurrent abort, `cleanup_status` remains `not_requested`. |
| Native dispatch started, its outcome is uncertain, or later validation fails | Record failure, request native fail-closed handling and require host cleanup evidence. An RPC timeout does not prove that native work stopped. |

The pre-dispatch classification is made before native revocation. It requires
the synchronous prepare caller to have ended without dispatch; no queued native
PREPARE can execute later. The result describes this preparation only, not a
concurrent abort or shutdown. It is not capture permission, runtime-health
evidence or authority to resume serving. `FAILED` in lifecycle status describes
the attempt; it does not by itself mean the native runtime or KV state is invalid.
The default `native_prepare_not_dispatched: false` means no such proof is supplied;
it does not prove that dispatch occurred.

This prototype does not implement a new-attempt retry or an explicit resume API.
The same operation ID returns its original failure without repeating work or
extending its budget. New progression requests remain rejected. Avoiding native
revocation preserves a future recovery option; it does not make recovery usable
in this version. An operator can stop and replace the worker through the existing
deployment controls; there is no in-place retry/resume path yet.

Follow-up recovery must let the controller choose retry, explicit cancellation
and resume, or worker termination. It must prove no late dispatch, serialize with
abort and shutdown, preserve their stop decisions, and check health and every
other admission gate before returning to service.

Missing participant evidence or validation failure keeps admission closed.
Stale commands cannot advance the lifecycle.
Abort does not resume the source. It does not prove
host cleanup, DMA retirement or safe memory reuse. The external supervisor
must remain outside the worker job's effective failure/abort scope.
Rank zero requests peer park release and waits for acknowledgements before local
scheduler shutdown. If a native command or listener does not retire during shutdown, TensorRT-LLM
retains engine resources and reports a shutdown timeout. The external supervisor
must terminate the entire worker job before retrying startup. Do not continue
resource teardown while a checkpoint command can still access those resources.

HTTP control operations remain reachable while inference is blocked. `/health`
keeps the existing engine liveness check; a healthy parked engine returns 200.
In checkpoint mode, `/ready` returns 200 only when the engine is healthy and
checkpoint admission is open, including ordinary serving before preparation.
Use `/health` for liveness and `/ready` for traffic readiness. Do not use `/health`
as evidence that restore validation or activation completed. Generation paths,
including `/health_generate`, return 503 while admission is closed.
OpenAI and Anthropic generation routes keep their respective error envelopes.
Passive `/version`, `/v1/models`, `/metrics` and `/prometheus/metrics` reads remain
available. Other routes, including `/energy_metrics` and `/server_info`, remain
gated; this prototype does not classify all handlers as safe passive reads.
Runtime `/start_profile` and `/stop_profile` controls and Message Batch creation
are rejected for the entire checkpoint-mode lifetime, including before prepare.
Background batches require a separate lifetime/admission contract. Profiling
cannot start through an unauthenticated route and bypass the capture boundary.
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
