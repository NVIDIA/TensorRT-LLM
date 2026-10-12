<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TensorRT-LLM OpenEngine server

`trtllm-serve` can expose an experimental OpenEngine gRPC server instead of its normal OpenAI HTTP server. SMG remains the default gRPC protocol.

OpenEngine support ships in the base TensorRT-LLM package. `pip install "tensorrt_llm[openengine]"` remains available as a compatibility spelling but installs nothing extra.

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

Features without a faithful TensorRT-LLM mapping return `UNIMPLEMENTED`: prefix-cache bypass, LoRA lifecycle selection, multimodal media, explicit-token or all-vocabulary log-probability selection, nonzero prompt-logprob offsets, per-request grammar-backend selection, and priority or data-parallel-rank metadata. `Control` implements `GetServerInfo`, `GetModelInfo`, `GetLoad`, `Health` and `Abort`; its LoRA lifecycle and KV-event RPCs return `UNIMPLEMENTED`.

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

The schema source is the Apache-2.0-licensed [`ai-dynamo/openengine`](https://github.com/ai-dynamo/openengine) repository at signed Git tag [`v0.1.0`](https://github.com/ai-dynamo/openengine/releases/tag/v0.1.0), Git commit `b5f2bd93721f7b888d3e2440679e0ae7012939d1`. That release maps to the public [`buf.build/openengine/openengine`](https://buf.build/openengine/openengine) module at immutable BSR release `768a93c7b44e40f28c692ad0b471a8f2`.

TensorRT-LLM vendors that immutable schema under `tensorrt_llm/grpc/openengine/proto/`. `3rdparty/vendor_sources.lock.yaml` pins the upstream commit and verifies that the schema files are exact upstream copies. The adjacent `manifest.json` records protocol metadata, generator versions, and runtime floors. The deterministic private bindings under `tensorrt_llm.grpc.openengine._generated` are checked in so ordinary wheel and editable builds do not require a protocol compiler or network access.

When changing the schema, generator, or `requirements-build-openengine.txt`, regenerate and commit the bindings:

```bash
python scripts/generate_openengine_protos.py --tool-env-root build/openengine-proto-tools
```

Verify that a checkout is current without modifying it:

```bash
python scripts/generate_openengine_protos.py --check --tool-env-root build/openengine-proto-tools
```

The wheel ships these bindings only in TensorRT-LLM's private namespace; it does not provide or depend on a top-level `openengine` Python package. Its protobuf and `grpcio>=1.67.1,<2` runtime constraints are base TensorRT-LLM requirements.

The private Python namespace does not change the protobuf descriptor names: generated modules still register `openengine/v1/*.proto` in the process-wide default descriptor pool. Importing another OpenEngine binding package in the same process is safe only when it registers identical schema bytes; a different schema revision can raise a duplicate-file `TypeError` during import.

## Maintenance boundary

The OpenEngine contributor community owns this adapter, its tests, protocol version updates, and integration bugs. TensorRT-LLM internal APIs do not provide compatibility guarantees to protocol adapters. Adapter updates must follow core runtime changes and must not block normal TensorRT-LLM development or releases.

### Token-only output and conversation affinity

For token-only clients, send boolean `extra["detokenize"] = false`. Omitted detokenize retains normal text output. Forward the same stable `extra["conversation_id"]` on prefill and decode requests and across conversation turns.

Conversation-affinity placement requires `attention_dp_config.kv_cache_routing_conversation_affinity: true` when using attention DP; it is disabled by default. Per-conversation cache retention requires block reuse to be enabled and `kv_cache_config.block_reuse_config.policy: per_conversation` with KV cache manager v2; the default policy is `all_reusable`. Forwarding a conversation ID does not enable either setting. These extensions do not require KV-event discovery, publication, or routing-load snapshots.
