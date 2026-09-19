<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TensorRT-LLM OpenEngine server

`trtllm-serve` can expose an experimental OpenEngine gRPC server instead of its normal OpenAI HTTP server. SMG remains the default gRPC protocol.

Install the optional Python bindings from the Buf Schema Registry:

```bash
python -m pip install \
  --extra-index-url https://buf.build/gen/python \
  "tensorrt_llm[openengine]"
```

Then select OpenEngine when starting the gRPC server:

```bash
trtllm-serve <model> \
  --grpc \
  --grpc-protocol openengine \
  --host 0.0.0.0 \
  --port 50051
```

Existing `--grpc` invocations continue to select SMG. OpenEngine and VisualGen cannot be enabled together.

## Independent KV telemetry

The OpenEngine server is the configuration authority. Clients discover event sources through `GetKvEventSources` and KV occupancy through optional `GetLoad` fields. The published OpenEngine protocol and Python bindings are unchanged; the server exposes engine telemetry, not router-specific policy.

- Enable routing-load sampling with `--openengine-enable-load-metrics`. This option requires `--grpc --grpc-protocol openengine` and does not enable iteration statistics.
- Enable KV events independently with `kv_cache_config.kv_events_config.enable_kv_cache_events: true`, a reachable ZMQ endpoint, and `kv_cache_event_hash_algo: v1_block_key` for Dynamo-compatible hashes.
- Either, both, or neither may be enabled. DP-rank targeting remains independently available. Full Dynamo KV-aware routing uses both, with the frontend in `--router-mode kv`.
- Native HTTP serving does not enable this load sampler. Its existing KV-event configuration and publisher behavior are unchanged.

An empty event-source list means no external publisher is configured. Absent KV occupancy fields mean load sampling is disabled; enabled-but-unavailable load returns `UNAVAILABLE` instead of pretending to be disabled. Older servers may return `UNIMPLEMENTED`; clients can fall back to generation-only operation.

Existing `ServerInfo.extra` carries optional engine metadata: `trtllm_supports_dp_rank_targeting` and, when streaming events are configured, `kv_event_heartbeat_interval_ms`. These are TensorRT-LLM extensions, not portable OpenEngine capabilities. Missing metadata disables the corresponding client optimization. Strict rank validation remains on the server, and clients without heartbeat metadata must not infer event-source failure merely from idle silence.

The `Inference.Generate` RPC loads the selected model through TensorRT-LLM's PyTorch `LLM` API and streams incremental token and text events. It supports text and token-ID inputs, native sampling and stopping options, top-N prompt and output log probabilities, TensorRT-LLM guided-decoding modes, cache salt, trace-context propagation, multiple output sequences, finish reasons, and final usage.

Clients must continuously consume the response stream. If response delivery remains stalled for 30 seconds, the server aborts the engine request and terminates the stream with a retryable overload error.

Features without a faithful TensorRT-LLM mapping return `UNIMPLEMENTED`: prefix-cache bypass, LoRA lifecycle selection, multimodal media, explicit-token or all-vocabulary log-probability selection, nonzero prompt-logprob offsets, per-request grammar-backend selection, and priority metadata. `openengine-target-dp-rank` is accepted and mapped to strict attention-DP scheduling. The AutoDeploy backend is rejected at startup until it supports request cancellation. `Control` implements `GetServerInfo`, `GetModelInfo`, `GetLoad`, `Health`, `Abort` and the KV-event RPCs; its LoRA lifecycle RPCs return `UNIMPLEMENTED`.

### KV cache events

`Control.GetKvEventSources` and `Control.SubscribeKvEvents` expose the native KV cache manager's direct per-rank publishers. Configure direct publishing and Dynamo-compatible block hashes as follows:

```yaml
kv_cache_config:
  use_kv_cache_manager_v2: true
  enable_block_reuse: true
  kv_cache_event_hash_algo: v1_block_key
  kv_events_config:
    enable_kv_cache_events: true
    endpoint: tcp://*:5557
    replay_endpoint: tcp://*:5657
```

Each KVCM2 rank writes routing events for one attention lifecycle to its own bounded native queue. A worker-local background publisher blocks on that queue, serializes each batch once as msgpack, and publishes it over ZMQ. `GetKvEventSources` advertises the connectable endpoint, topic, replay endpoint, buffer size, queue size, and HWM for every attention-DP rank. The direct data path does not use an attention-DP gather, frontend worker RPC, JSON, protobuf conversion, or polling.

Direct OpenEngine discovery currently supports a single host. It returns `FAILED_PRECONDITION` for multi-host attention-DP because one OpenEngine listener cannot infer a connectable host for each remote publisher.

`SubscribeKvEvents` is a demand-driven compatibility bridge for clients that cannot reach the worker-local ZMQ ports. It subscribes to the same normalized msgpack stream and converts batches to protobuf only while the RPC is active. It does not drain `LLM.get_kv_events()`, so it cannot race the public buffered event API.

Discovery returns no sources when direct publishing is disabled. Non-TCP endpoints and multi-host attention-DP return `FAILED_PRECONDITION` rather than advertising unreachable sources. OpenEngine checks the loaded executor's publisher capability before starting the listener and rejects enabled events without a publisher. Native HTTP retains its warning-only behavior for event configuration on V1. Discovery describes the startup-validated configuration; clients must still treat connection failures after startup as source failures because a publisher can fail after it has been advertised.

Internally detected publisher queue loss, native queue loss, and translation failures schedule a rank-local `AllBlocksCleared` batch, including at an idle tail. Pre-loss queued batches are discarded before that recovery fence. ZeroMQ can independently drop a PUB message for a slow subscriber at its HWM; the publisher cannot observe that per-subscriber loss, so a direct subscriber detects it from a later sequence gap and an idle-tail drop has no immediate recovery signal. Native reads are chunked, and encoded publisher/replay retention has a 64 MiB byte cap in addition to its configured count cap. Sequence replay remains bounded by `buffer_steps`; a replay that starts after the requested sequence still requires the client to resynchronize. Sequence numbers start from a wall-clock epoch so resume cursors remain ordered across publisher restarts. Direct replay clients must use a receive deadline because a disconnected or persistently unwritable client cannot be guaranteed the terminal sentinel. The in-tree bridge uses a two-second deadline. A slow protobuf subscriber receives a retryable terminal stream error instead of silently losing a state-changing batch.

Cold-tier, salted, LoRA-scoped, and multimodal blocks are omitted because Dynamo routing currently targets directly reusable GPU KV and the wire contract cannot faithfully carry the scoped cache namespace. LoRA, multimodal generation, and authoritative snapshots remain outside this initial routing contract.

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

## Transport security

The listener is plaintext h2c with no authentication. Any client that can reach
the port can run inference and call `Control.Abort`, including `all_requests`.
Bind it to loopback alongside its caller, or front it with a proxy that
terminates TLS and authenticates. The server logs a warning when it binds to a
non-loopback address.

## Dependency provenance

The schema source is the Apache-2.0-licensed [`ai-dynamo/openengine`](https://github.com/ai-dynamo/openengine) repository at signed Git tag [`v0.1.0`](https://github.com/ai-dynamo/openengine/releases/tag/v0.1.0). That release maps to the public [`buf.build/openengine/openengine`](https://buf.build/openengine/openengine) module at immutable BSR commit `768a93c7b44e40f28c692ad0b471a8f2`.

The BSR generated the pinned wheels from that module commit:

| Package | Generator | Version | SHA-256 |
| --- | --- | --- | --- |
| `openengine-openengine-grpc-python` | [`grpc/python`](https://buf.build/grpc/python) | `1.67.1.2.20260730172104+768a93c7b44e` | `1485aed9799c4eb9367d1a261ca5cc5319f1e9b8d950ac98a26f3cb3641b8cf6` |
| `openengine-openengine-protocolbuffers-python` | [`protocolbuffers/python`](https://buf.build/protocolbuffers/python) | `31.1.0.2.20260730172104+768a93c7b44e` | `6eae12c3d8d06147fccf608da9772d6391139031fabdafdb7cf4c71a19c1f25e` |
| `openengine-openengine-protocolbuffers-pyi` | [`protocolbuffers/pyi`](https://buf.build/protocolbuffers/pyi) | `31.1.0.2.20260730172104+768a93c7b44e` | `8b0a054dbdaaa67459b3fa4786f13d8f6f4d30cf30be325f5416dbd97aba46a6` |

Buf documents the package naming and version format in its [Python-generated SDK guide](https://buf.build/docs/bsr/generated-sdks/python/). The final version segment is the BSR commit prefix. The exact requirements are pinned in `requirements-openengine.txt`.

## Maintenance boundary

The OpenEngine contributor community owns this adapter, its tests, protocol version updates, and integration bugs. TensorRT-LLM internal APIs do not provide compatibility guarantees to protocol adapters. Adapter updates must follow core runtime changes and must not block normal TensorRT-LLM development or releases.
