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

The `Inference.Generate` RPC loads the selected model through TensorRT-LLM's PyTorch `LLM` API and streams incremental token and text events. It supports text and token-ID inputs, native sampling and stopping options, top-N prompt and output log probabilities, TensorRT-LLM guided-decoding modes, cache salt, trace-context propagation, multiple output sequences, finish reasons, and final usage.

Clients must continuously consume the response stream. If response delivery remains stalled for 30 seconds, the server aborts the engine request and terminates the stream with a retryable overload error.

Features without a faithful TensorRT-LLM mapping return `UNIMPLEMENTED`: prefix-cache bypass, LoRA lifecycle selection, multimodal media, explicit-token or all-vocabulary log-probability selection, nonzero prompt-logprob offsets, per-request grammar-backend selection, and priority metadata. Attention-DP rank hints are validated and routed strictly to the requested rank. The AutoDeploy backend is rejected at startup until it supports request cancellation. `Control` implements `GetServerInfo`, `GetModelInfo`, `GetLoad`, `GetKvEventSources`, `Health` and `Abort`; its LoRA lifecycle and `SubscribeKvEvents` RPCs return `UNIMPLEMENTED`.

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

KV events are captured by the native KVCM2 streaming sink and delivered directly over ZMQ. OpenEngine does not drain `LLM.get_kv_events()` or forward event payloads. `GetKvEventSources` advertises the live and replay endpoints, topic, encoding, and attention-DP rank. `SubscribeKvEvents` returns `UNIMPLEMENTED`; its protocol definition is unchanged.

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

Enable load reporting independently with `--grpc --grpc-protocol openengine --openengine-enable-load-metrics`. Without that flag, `GetLoad` omits KV load fields. Enabling events does not enable load sampling, and enabling load sampling does not enable event publishing. Clients discover enabled capabilities from the running server rather than duplicating these switches.

For attention DP, each publisher binds `base_port + global_rank`. OpenEngine collects rank hostnames once at executor startup and advertises every rank, including remote nodes. Those hostnames and ports must be reachable from the subscriber. Missing multi-node placement fails discovery rather than returning a partial set. PP and CP streaming remain unsupported. The extra placement collection is OpenEngine-only; native HTTP serving does not enable it.

### Node-local discovery

OpenEngine exposes optional, versioned `ServerInfo.extra.trtllm_node` metadata without changing the protocol schema. It describes a shared engine incarnation (`engine_id`), local hostname (`node_id`), whether this is the leader, node count, KV block size, local publishing DP ranks, and the global rank-to-node ownership map. Consumers can use this information to partition telemetry collection without introducing routing-framework-specific server options.

The serving ingress retains engine-wide `GetKvEventSources` discovery for existing clients. On each other hostname, one executor starts a bounded metadata-only listener at the configured OpenEngine port. Its `GetKvEventSources` returns only actual local publishers. It exposes neither inference nor engine-wide load; other RPCs are unimplemented. A node with no publishing rank returns an empty source list. Metadata listeners stop with their executor, and bind errors fail startup collectively.

Multinode deployments require a nonzero OpenEngine port and ingress placement on the engine-rank-zero host. Distinct nodes/pods must have distinct hostnames. Remote-ingress/Ray placement has not been validated. A node-local collector must not coexist with a leader-wide collector for the same sources. Engine incarnation changes require consumers to rediscover ownership; this metadata does not provide event replay or independent collector restart guarantees.

Placement exchange occurs only during OpenEngine executor startup. Metadata RPCs read snapshots without GPU synchronization or runtime collectives. Event publishing and load sampling remain independently controlled, and native HTTP serving does not start these listeners.

Events describe radix-cache residency, including reusable offloaded blocks; `medium` is not a live GPU-tier inventory. Load snapshots independently report GPU block usage. Consumers must track rank-local sequence numbers and replay or invalidate their state on gaps. Publisher startup, detected queue loss, and native capture overflow invalidate stale state with `AllBlocksCleared`; idle heartbeats expose otherwise silent gaps. A reset does not reconstruct already-resident cache contents, so recovery can temporarily under-report cache affinity.

Direct ZMQ sockets are unauthenticated. Keep them on a trusted network and restrict access to intended subscribers. No Dynamo-specific protocol fields or hash algorithm are required; consumers treat published block hashes as opaque identities and use token/parent data to build their routing index.

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
