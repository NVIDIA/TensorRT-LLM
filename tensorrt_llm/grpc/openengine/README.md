<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TensorRT-LLM OpenEngine server

`trtllm-serve` can expose an experimental OpenEngine gRPC server instead of its normal OpenAI HTTP server. SMG remains the default gRPC protocol.

OpenEngine support ships in the base TensorRT-LLM package. `pip install "tensorrt_llm[openengine]"` remains available as a compatibility spelling but installs nothing extra.

Native protobuf (`upb`) is recommended for gRPC throughput. TensorRT-LLM images select it
by default. In older or custom environments, set
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb` before starting Python and use a
protobuf wheel that includes the native extension. Both gRPC launchers warn when
Python protobuf is active and continue startup. Native `cpp` is also accepted
without a warning.

Then select OpenEngine when starting the gRPC server:

```bash
trtllm-serve <model> \
  --grpc \
  --grpc-protocol openengine \
  --host 127.0.0.1 \
  --port 30001
```

Existing `--grpc` invocations continue to select SMG. OpenEngine and VisualGen cannot be enabled together.

## Engine shutdown

`Control.Shutdown` requests orderly shutdown of the entire engine serving group. Pass the intended lifetime's `GetServerInfo.instance_id` as `ShutdownRequest.expected_instance_id`. Missing IDs return `INVALID_ARGUMENT`; mismatches return `FAILED_PRECONDITION` without changing the running group. Identity is checked again by the launcher when an attached frontend forwards the request. The ID is a lifetime fence, not an authentication credential.

`ACCEPTED` acknowledges that the launcher has committed to shutdown and closed new inference admission. `ALREADY_STOPPING` acknowledges a repeat for that same lifetime. Neither response means that process exit or engine cleanup has completed. Accepted shutdown proceeds after RPC cancellation or a lost reply. After an uncertain outcome, check lifecycle and identity and retry only with the original expected ID. Older servers return `UNIMPLEMENTED`. An embedded `OpenEngineServer` without an injected engine lifecycle owner also returns `UNIMPLEMENTED`; stopping its listener alone would leave the engine running.

Shutdown uses the same stop event and cleanup path as SIGTERM/SIGINT. Inference health becomes `NOT_SERVING` before the shutdown acknowledgement, including when owner cleanup is deferred; `Control.Health` reports not-ready and suppresses new inference probes. Healthy liveness remains `SERVING` until listener closure, and known fatal failures retain their fatal status. Late health results cannot restore readiness. Work admitted before shutdown follows the existing grace policy: the gRPC listener allows up to five seconds for active RPCs, then cancels them; attached frontends receive SIGTERM and share a ten-second termination deadline before SIGKILL. The launcher then closes coordination and shuts down the engine. These grace periods are not a total cleanup deadline: media-pool drain and engine shutdown can take longer. An independent fatal failure remains a failure even when it races an intentional shutdown.

The listener's existing trusted-network/TLS-proxy requirements also apply to this destructive control operation. Engine shutdown does not authenticate a caller merely because the expected identity matches.

## Health checks

OpenEngine serves the standard `grpc.health.v1.Health` service on the same gRPC port as Inference and Control.

`Check` returns the current status. `Watch` sends the current status and later status changes. Rapid changes can merge into one update. The empty service has the same status as `inference`. Use these named services for health probes:

| Service | Meaning | Probe use |
|---------|---------|----------------|
| `inference` | The executor is healthy and, with multiple frontends, the frontend group is ready. | Readiness probe |
| `liveness` | The gRPC server is running and no known fatal engine failure requires a restart. | Startup and liveness probes |

The server refreshes inference status in a background task, so health RPCs read the published value without probing the executor on each request. A check that runs longer than five seconds makes inference `NOT_SERVING`; the server keeps at most one engine check in flight and discards a late result before polling again. `Control.Health` reads the same engine predicate directly, so its result can lead the published status by a poll interval or the check deadline. The multi-frontend coordinator reads the monitor's latest engine result without running another engine check. A known fatal executor or worker failure makes both services `NOT_SERVING` and stops the serving process. Other readiness failures leave liveness `SERVING` and can recover. During orderly shutdown, inference reports `NOT_SERVING` while liveness retains its previous status until the listener closes. Each `Watch` ends after its final status. A stopped server cannot answer the probe.

For a Pod with a Dynamo sidecar and native Kubernetes gRPC probes, bind one listener to all Pod interfaces:

```bash
trtllm-serve <model> --grpc --grpc-protocol openengine \
  --host 0.0.0.0 --port 30001
```

The sidecar can use `127.0.0.1:30001`. Kubernetes probes connect to the Pod IP on that same port:

```yaml
startupProbe:
  grpc:
    port: 30001
    service: liveness
  periodSeconds: 10
  failureThreshold: 60
readinessProbe:
  grpc:
    port: 30001
    service: inference
livenessProbe:
  grpc:
    port: 30001
    service: liveness
```

Set `failureThreshold * periodSeconds` to cover the longest expected model startup. The listener has no general client authentication or encryption, and Inference and Control are reachable through the Pod IP along with Health. Review that exposure for the deployment. If only same-Pod clients need access, bind to `127.0.0.1` and use exec probes instead.

OpenEngine gRPC does not yet expose sleep or weight updates. When those operations are added, they must publish `inference=NOT_SERVING` for their duration while keeping `liveness=SERVING`; the current executor health predicate alone does not detect those admission states. A sidecar that loses `Watch` or sees a new engine identity must validate health again.

The `Inference.Generate` RPC loads the selected model through TensorRT-LLM's PyTorch `LLM` API and streams incremental token and text events. It supports text and token-ID inputs, native sampling and stopping options, top-N prompt and output log probabilities, TensorRT-LLM guided-decoding modes, cache salt, trace-context propagation, multiple output sequences, finish reasons, and final usage.

Clients must continuously consume the response stream. If response delivery remains stalled for 30 seconds, the server aborts the engine request and terminates the stream with a retryable overload error.

Features without a faithful TensorRT-LLM mapping return `UNIMPLEMENTED`: prefix-cache bypass, LoRA lifecycle selection, multimodal media, explicit-token or all-vocabulary log-probability selection, nonzero prompt-logprob offsets, per-request grammar-backend selection, and priority metadata. Attention-DP rank hints are validated and used for first-turn placement. With conversation affinity on, the first hint establishes the conversation's rank, and later turns retain that rank even if their hints differ; the request must carry a stable conversation ID. `ServerInfo.extra.trtllm_supports_dp_rank_targeting` advertises this support. The AutoDeploy backend is rejected at startup until it supports request cancellation. `Control` implements `GetServerInfo`, `GetModelInfo`, `GetLoad`, `GetKvEventSources`, `Health`, `Abort`, and `Shutdown`; its LoRA lifecycle and `SubscribeKvEvents` RPCs return `UNIMPLEMENTED`.

### Disaggregated serving

A context worker returns its handoff as a `PrefillReady` event carrying a
`KvSessionRef`; a generation worker resumes it by echoing that `KvSessionRef`
back in `GenerateRequest.kv.session`. A request carrying a session is always
treated as `generation_only`.

The context phase has no native field in the protocol, so it is selected with
`extra["request_type"] = "context_only"` (`"context_and_generation"` is also
accepted, and is the default when the key is absent). `extra` is outside the
portable contract, so a client that omits it simply gets aggregated serving.
`"generation_only"` cannot be named this way: the phase needs the context
worker's address and request id, which only a `KvSessionRef` carries.

`Control.Abort` cannot yet release a prefill by `kv_session`
(`KvSessionRef.session_id` is the engine's context request id, not a `Generate`
`request_id`), so `GetServerInfo` reports
`kv_connector.supports_abort_cleanup = false`. A generation leg that never
arrives leaves its KV blocks held on the context worker until that process
exits.

OpenEngine and SMG are independent protocol integrations. This integration does not make a replacement or convergence decision between them.

## KV event discovery and routing load

KV events are captured by the native KVCM2 streaming sink and delivered directly over ZMQ. OpenEngine does not drain `LLM.get_kv_cache_events()` or forward event payloads. `GetKvEventSources` advertises the live and replay endpoints, topic, encoding, and attention-DP rank. `SubscribeKvEvents` returns `UNIMPLEMENTED`; its protocol definition is unchanged.

Enable streaming in the server configuration:

```yaml
kv_cache_config:
  use_kv_cache_manager_v2: true
  enable_block_reuse: true
  kv_events_config:
    enable_kv_cache_events: true
    endpoint: tcp://*:5557
    replay_endpoint: tcp://*:5657
```

Enable load reporting independently with `--grpc --grpc-protocol openengine --openengine-enable-load-metrics`. Without that flag, `GetLoad` omits KV load fields. Event streaming internally samples per-rank capacity so `GetServerInfo` remains accurate after pool rebalance, but does not expose occupancy through `GetLoad`. Enabling load reporting does not enable event publishing. Clients discover enabled capabilities from the running server rather than duplicating these switches.

For attention DP, each publisher binds `base_port + global_rank`. When event streaming is enabled, OpenEngine collects the live publisher descriptors at executor startup and advertises every rank, including remote nodes. Those hostnames and ports must be reachable from the subscriber. A missing publisher fails startup; an unavailable discovery snapshot fails the RPC rather than synthesizing socket addresses. PP and CP streaming remain unsupported. The extra placement collection is OpenEngine-only; native HTTP serving and OpenEngine with events disabled do not enable it.

### Node-local discovery

OpenEngine exposes optional, versioned `ServerInfo.extra.trtllm_node` metadata without changing the protocol schema. It describes a shared engine incarnation (`engine_id`), local hostname (`node_id`), whether this is the leader, node count, KV block size, local publishing DP ranks, and the global rank-to-node ownership map. The engine ID matches `GetServerInfo.instance_id` on the serving ingress and node metadata listeners. Consumers can use this information to partition telemetry collection without introducing routing-framework-specific server options.

The serving ingress retains engine-wide `GetKvEventSources` discovery for existing clients. With event streaming enabled, one executor on each other hostname starts a bounded metadata-only listener at the configured OpenEngine port. Its `GetKvEventSources` returns only actual local publishers. It exposes neither inference nor engine-wide load; other RPCs are unimplemented. Metadata listeners stop with their executor, and bind errors fail startup collectively.

Multinode deployments require a nonzero OpenEngine port and ingress placement on the engine-rank-zero host. Distinct nodes/pods must have distinct hostnames. Remote-ingress/Ray placement has not been validated. A node-local collector must not coexist with a leader-wide collector for the same sources. Engine incarnation changes require consumers to rediscover ownership; this metadata does not provide event replay or independent collector restart guarantees.

Placement exchange occurs only during OpenEngine executor startup. Metadata RPCs read snapshots without GPU synchronization or runtime collectives. Event publishing and load sampling remain independently controlled, and native HTTP serving does not start these listeners.

Events report full-block radix-cache residency; the streaming sink does not report live GPU-versus-cold-tier movement. Load snapshots independently report GPU block usage. The upstream sink emits signed 64-bit hashes from the first eight bytes of each V2 radix key, regardless of `kv_cache_event_hash_algo`. `ServerInfo.extra.trtllm_kv_event_hash_algo` advertises `v2_sha256_64` when streaming is enabled. These hashes identify blocks in event parent chains and removals. A router can derive its own request-compatible token hashes from the stored block `token_ids`, as Dynamo does.

Consumers must track rank-local sequence numbers and resynchronize on gaps. Publisher queue drops and failed sends leave sequence gaps, but native capture overflow is logged without a wire gap. Replay retains only the configured number of batches; a slow replay client can lose its replay response and must retry. Replay is sent in bounded, nonblocking chunks so it cannot stall live publication. The upstream publisher emits no idle heartbeat, so source health cannot be inferred from silence alone. A reset or restart does not reconstruct already-resident cache contents, and recovery can temporarily under-report cache affinity.

Direct ZMQ sockets are unauthenticated. Keep them on a trusted network and restrict access to intended subscribers.

## Transport security

The listener is plaintext h2c with no authentication. Any client that can reach
the port can run inference and call `Control.Abort`, including `all_requests`.
Bind it to loopback alongside its caller, or front it with a proxy that
terminates TLS and authenticates. The server logs a warning when it binds to a
non-loopback address.

## Multiple frontend processes

On Linux, the classic IPC PyTorch executor supports multiple OpenEngine frontend processes sharing one model instance:

```bash
trtllm-serve Qwen/Qwen3-0.6B \
  --grpc --grpc-protocol openengine \
  --num_serve_frontends 4 --port 50051
```

Every frontend binds the same fixed port with `SO_REUSEPORT`. The kernel distributes TCP connections, not individual RPCs within an HTTP/2 connection. Configure the client or router to open multiple independent connections; one multiplexed connection uses only one frontend, and connection distribution is not guaranteed to be even. Python `grpcio` channels with the same target and arguments share a connection; set the `grpc.use_local_subchannel_pool` channel argument to 1 on each channel to give them independent connections.

The launcher owns the executor and a private local coordinator. Other frontends attach to that executor; model weights and GPU execution are shared. Token streams go directly through each frontend's executor result lane. The coordinator reserves request IDs across all frontends and routes control operations to the owning process. `Abort(request_id)` works from any connection, `Abort(all_requests)` targets a snapshot of active requests, and `GetLoad` reports the aggregate active reservation count, including pending submissions and health probes. All frontends advertise the same engine instance ID.

Readiness requires the whole frontend group. Failure of a frontend, coordinator connection, or engine stops the group; individual frontends are not automatically restarted. Shutdown stops admission and drains or cancels streams before releasing the engine. A signal received during synchronous model initialization is handled at the next initialization cleanup boundary.

If the coordinator fails during final request cleanup, a client can receive a trailing `UNAVAILABLE` even after its stream delivered every response. Clients that retry this status may repeat completed work.

With multiple frontends, request IDs in `Generate` and `Abort(request_id)` are limited to 1024 UTF-8 bytes. Larger IDs return `INVALID_ARGUMENT` before reaching the coordinator, without affecting other requests or frontend readiness.

Abort snapshots are sent in batches of at most 128 requests, with at most one batch in flight per target frontend. Each batch is below 1 MiB even with maximum-length, JSON-escaped IDs. Private abort operations have a 30-second caller deadline. The coordinator limits snapshot dispatch to 25 seconds, leaving 5 seconds to return the outcome; individual batch RPCs retain the 5-second control deadline. When the snapshot budget expires, confirmed outcomes are preserved and unconfirmed or unsent requests are counted as failures. The public abort returns `INTERNAL` for incomplete cancellation without stopping the serving group.

Frontends send heartbeats once per second. A missing heartbeat for 10 seconds withdraws group readiness, stops admission, and shuts down the group; a late heartbeat cannot restore readiness. Coordinator RPCs must still complete within 5 seconds (except abort operations). The coordinator shares frontend 0's event loop, so a sufficiently long CPU stall there also stops the group. Provision CPU capacity for control traffic and synchronous input processing as well as token streaming.

Multiple frontends require the default classic IPC executor and a nonzero port. SMG and AutoDeploy do not support this mode. Frontends share host CPU resources, so provision sufficient CPU and benchmark the intended model, router, and connection pool before choosing a frontend count. More processes do not necessarily increase throughput once GPU execution becomes the bottleneck.

## Schema and binding provenance

The schema source is the Apache-2.0-licensed [`ai-dynamo/openengine`](https://github.com/ai-dynamo/openengine) repository. This shutdown implementation uses source commit `78ac56152781559b5e4e8ae9ef357aaee833f19f`, recorded in the vendor lock and schema manifest. It is an unpublished protocol revision: `bsr.release` is unset and `ServerInfo.schema_release` reports the source commit. Before upstream release, land the canonical protocol change and refresh both the immutable source pin and any published BSR release metadata. Do not reuse the earlier `v0.1.0` BSR release identifier for the extended schema.

TensorRT-LLM vendors that immutable schema under `tensorrt_llm/grpc/openengine/proto/`. `3rdparty/vendor_sources.lock.yaml` pins the upstream commit and verifies that the schema files are exact upstream copies. The adjacent `manifest.json` records protocol metadata, generator versions, and runtime floors. The deterministic private bindings under `tensorrt_llm.grpc.openengine._generated` are checked in so ordinary wheel and editable builds do not require a protocol compiler or network access.

When changing the schema, generator, or `requirements-build-openengine.txt`, regenerate and commit the bindings:

```bash
python scripts/generate_openengine_protos.py --tool-env-root build/openengine-proto-tools
```

Verify that a checkout is current without modifying it:

```bash
python scripts/generate_openengine_protos.py --check --tool-env-root build/openengine-proto-tools
```

The wheel ships these bindings only in TensorRT-LLM's private namespace; it does not provide or depend on a top-level `openengine` Python package. Its protobuf, `grpcio>=1.73.1,<2`, and `grpcio-health-checking>=1.73.1,<1.82` runtime constraints are base TensorRT-LLM requirements. The health-checking upper bound keeps this protobuf 6 installation on compatible releases. OpenEngine maintainers own Health API behavior and compatibility. Dependency updates require a check of gRPC and protobuf requirements, wheel installation, and the OpenEngine Check/Watch tests in CI. Route Health behavior failures to the OpenEngine maintainers and installation or dependency-resolution failures to the requirements maintainers.

The private Python namespace does not change the protobuf descriptor names: generated modules still register `openengine/v1/*.proto` in the process-wide default descriptor pool. Importing another OpenEngine binding package in the same process is safe only when it registers identical schema bytes; a different schema revision can raise a duplicate-file `TypeError` during import.

## Maintenance boundary

The OpenEngine contributor community owns this adapter, its tests, protocol version updates, and integration bugs. TensorRT-LLM internal APIs do not provide compatibility guarantees to protocol adapters. Adapter updates must follow core runtime changes and must not block normal TensorRT-LLM development or releases.

### Token-only output and conversation affinity

For token-only clients, send boolean `extra["detokenize"] = false`. Omitted detokenize retains normal text output. Forward the same stable `extra["conversation_id"]` on prefill and decode requests and across conversation turns.

Conversation-affinity placement requires `attention_dp_config.kv_cache_routing_conversation_affinity: true` when using attention DP; it is disabled by default. Per-conversation cache retention requires block reuse to be enabled and `kv_cache_config.block_reuse_config.policy: per_conversation` with KV cache manager v2; the default policy is `all_reusable`. Forwarding a conversation ID does not enable either setting. These extensions do not require KV-event discovery, publication, or routing-load snapshots.

For disaggregated subagent context requests, configure `internal_request_auth_key` in the `trtllm-serve` config and use the same key in the Dynamo sidecar's `TRTLLM_SUBAGENT_AFFINITY_AUTH_KEY`. The sidecar sends the parent session as signed `x-trtllm-subagent-affinity-id` and `x-trtllm-subagent-affinity-auth` gRPC metadata. OpenEngine accepts the hint only on `context_only` requests with a stable child `extra["conversation_id"]` and conversation-aware attention DP; it passes the parent to the ADP router without changing the child conversation ID. `ServerInfo.extra.trtllm_supports_subagent_affinity` reports whether the receiving server is configured for this extension. Keep the gRPC listener behind a trusted gateway: the signature authenticates the sidecar-to-engine hop, while the gateway must authorize the parent-child relationship. OpenEngine rejects `openengine-routing-key` and unknown `openengine-*` metadata rather than silently ignoring routing requests.
