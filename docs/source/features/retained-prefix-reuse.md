<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Experimental native retained-prefix reuse

The Python/NIXL disaggregated executor can opt in to retaining a completed decoder request's materialized prefix for a subsequent context request. A return consumes an opaque engine-issued reference and copies the admitted prefix through the existing native transfer sessions. The source allocation remains owned until every admitted native reader has drained and closed.

This is an experimental API. Omitted configuration and request fields keep the ordinary disaggregated behavior. It requires the Python transceiver, KV cache manager V2, block reuse enabled and GPU-only cache storage. Host/disk tiers, UVM, custom auxiliary caches, encoder/cross-attention state, and speculative or adapter state are rejected. The implementation does not provide a shared-store transport. Selecting NIXL does not establish an RDMA route.

## Administrative configuration

`NativeRetainedPrefixConfig` is an optional prototype field on `LlmArgs`. It contains this engine's opaque peer alias, a full immutable checkpoint/effective-model manifest and its SHA256, a local control endpoint, reciprocal trusted peer endpoints, and bounded export/claim leases. Endpoint and manifest fields are excluded from telemetry. They are administrator configuration; requests cannot supply transport addresses or physical buffer coordinates.

The manifest must be built from the actual final native executor and the complete immutable checkpoint/tokenizer inventory. The implementation verifies the real native loader, resolved model configuration, owned attention registry and all manifest members. A caller-provided model name or digest is insufficient. Unknown effective state fails closed. Capacity and scheduling differences do not authorize differences in model semantics. This change does not add a public manifest-generation HTTP endpoint.

## Request and lifetime contract

`DisaggregatedParams.retain_prefix=True` requests retention after decoder completion. A successful export is returned as a typed immutable `retained_prefix` containing only a version, peer alias, engine generation, opaque export identifier, materialized token boundary and identity. Supplying that reference on a later context request requests a reverse import. It does not prove that an import occurred.

The provider checks exact token, model, layout, nonce and generation bindings. It independently certifies layer/role/head/token/page coverage before native writes. A positive import advances the real request cursor only after verified completion, drained resources and successful native session closure. Partial or full local prefix hits fall back to the ordinary path; this implementation does not stitch local and remote prefixes. Expiry closes admission; it does not free buffers that still have native readers.

## Validation and limits

The checks below ran against initial public main `0fe3f26a9f6c2ba5ae360f2491d228aafa36dd06` with candidate010. The port onto `9484c5921027f2024e888bc129f3b73a582c1939` has not yet run native validation.

Those initial unit checks cover immutable wire validation, malformed/raw-coordinate rejection, stale/foreign bindings, complete role coverage, cursor/decoder packing, source lifetime and strict effective identity. Physical native component checks covered dense matching and asymmetric TP mappings and synthetic MiniMax cache layouts. These component identities are not checkpoint or inference qualification.

A full Qwen3-4B final-executor startup and sealed manifest were verified. A complete HTTP prefill-to-decode-to-next-prefill cycle has not passed. Full MiniMax-M3 checkpoint qualification, its block-reuse restriction, retained-HBM backing/route evidence, and performance/quality studies remain open. Forward or component success must not be reported as reverse model reuse. CUDA-graph compatibility and arbitrary asymmetric ordinary-forward ownership are not established by these checks.
