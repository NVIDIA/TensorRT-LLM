<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TensorRT-LLM OpenEngine server

`trtllm-serve` can expose an experimental OpenEngine gRPC server instead of its normal OpenAI HTTP server. SMG remains the default gRPC protocol.

Install the optional gRPC runtime:

```bash
python -m pip install "tensorrt_llm[openengine]"
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

Features without a faithful TensorRT-LLM mapping return `UNIMPLEMENTED`: prefix-cache bypass, LoRA lifecycle selection, multimodal media, explicit-token or all-vocabulary log-probability selection, nonzero prompt-logprob offsets, per-request grammar-backend selection, and priority or data-parallel-rank metadata. The AutoDeploy backend is rejected at startup until it supports request cancellation. `Control` implements `GetServerInfo`, `GetModelInfo`, `GetLoad`, `Health` and `Abort`; its LoRA lifecycle and KV-event RPCs return `UNIMPLEMENTED`.

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

## Schema and binding provenance

The schema source is the Apache-2.0-licensed [`ai-dynamo/openengine`](https://github.com/ai-dynamo/openengine) repository at signed Git tag [`v0.1.0`](https://github.com/ai-dynamo/openengine/releases/tag/v0.1.0), Git commit `b5f2bd93721f7b888d3e2440679e0ae7012939d1`. That release maps to the public [`buf.build/openengine/openengine`](https://buf.build/openengine/openengine) module at immutable BSR release `768a93c7b44e40f28c692ad0b471a8f2`.

TensorRT-LLM vendors that immutable schema under `tensorrt_llm/grpc/openengine/proto/`. The adjacent `manifest.json` records the source mapping, generator versions, runtime floors, and per-file checksums. During a wheel or supported editable build, `scripts/generate_openengine_protos.py` uses the fully pinned compiler environment in `requirements-build-openengine.txt` to generate private bindings under `tensorrt_llm.grpc.openengine._generated`.

The generated modules are build outputs and are not checked in. They are shipped only in TensorRT-LLM's private namespace; the wheel does not provide or depend on a top-level `openengine` Python package. The runtime dependencies remain TensorRT-LLM's base protobuf constraint plus `grpcio>=1.67.1,<2` from the OpenEngine extra.

## Maintenance boundary

The OpenEngine contributor community owns this adapter, its tests, protocol version updates, and integration bugs. TensorRT-LLM internal APIs do not provide compatibility guarantees to protocol adapters. Adapter updates must follow core runtime changes and must not block normal TensorRT-LLM development or releases.
