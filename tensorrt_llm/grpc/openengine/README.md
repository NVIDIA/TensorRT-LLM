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

Features without a faithful TensorRT-LLM mapping return `UNIMPLEMENTED`: prefix-cache bypass, LoRA lifecycle selection, multimodal media, explicit-token or all-vocabulary log-probability selection, nonzero prompt-logprob offsets, per-request grammar-backend selection, and priority or data-parallel-rank metadata. The AutoDeploy backend is rejected at startup until it supports request cancellation. `Control` implements `GetServerInfo`, `GetModelInfo`, `GetLoad`, `Health`, `Abort` and the KV-event RPCs; its LoRA lifecycle RPCs return `UNIMPLEMENTED`.

### KV cache events

`Control.GetKvEventSources` and `Control.SubscribeKvEvents` expose TensorRT-LLM's
streaming KV cache event publisher, which is off until
`kv_cache_config.kv_events_config` enables it:

```yaml
kv_cache_config:
  enable_block_reuse: true
  kv_events_config:
    enable_kv_cache_events: true
    endpoint: "tcp://*:5557"
    replay_endpoint: "tcp://*:5657"
```

Events are produced inside the engine: every attention-DP rank binds its own
ZeroMQ `PUB` socket at `base_port + rank` and publishes msgpack batches from a
background thread. `GetKvEventSources` advertises those sockets -- resolving the
bind wildcard to a routable host, because a client cannot connect to one -- so a
router subscribes to the engine directly and events never touch this server's
event loop. `SubscribeKvEvents` is for clients that cannot reach the ZeroMQ
ports: it subscribes on the client's behalf and re-publishes each batch as
protobuf, at the cost of a decode and re-encode per batch in this process.

Both return nothing rather than an error when publishing is disabled, so a
client can tell "not configured" from "not supported". Two configurations are
rejected with `FAILED_PRECONDITION` instead of being advertised unusably:
`ipc://` and `inproc://` endpoints, which have no host and port for the protocol
to carry, and attention DP spanning more than one node, where ranks bind on
hosts this process cannot name.

Delivery is best effort in both modes. Every batch carries a sequence number, so
a batch dropped by a full queue leaves an observable gap; a subscriber must treat
any gap as lost KV-cache state and resynchronize. When a replay endpoint is
configured, `SubscribeKvEventsRequest.start_sequence_number` and
`include_snapshot` replay the retained batches first.

The ZeroMQ listeners are a second unauthenticated, unencrypted surface alongside
the gRPC one -- see [Transport security](#transport-security) -- so the same
colocation rule applies to them.

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
