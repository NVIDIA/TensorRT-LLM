(kv-transfer)=

# Introduction to KV Cache Transmission

This article provides a general overview of the components used for device-to-device transmission of KV cache, which is relied upon by disaggregated serving. It is intended as a reference for users who wish to understand the internal implementation or develop extended functionalities. For configuring a deployment, see [Disaggregated Serving](../features/disagg-serving.md) instead.

The implementation described here is the Python transceiver, which lives under [`tensorrt_llm/_torch/disaggregation/`](source:tensorrt_llm/_torch/disaggregation). `cache_transceiver_config.transceiver_runtime` defaults to `auto`, which selects it on the supported NIXL path and otherwise falls back to the older C++ transceiver (`cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp` and friends). That C++ implementation is deprecated and will be removed in a future release; new work should target the Python path.

## Table of Contents

- [Workflow](#workflow)
- [Key Components](#key-components)
  - [Transceiver](#transceiver)
  - [Messenger](#messenger)
  - [Peer Registrar and Mappers](#peer-registrar-and-mappers)
  - [Page Table](#page-table)
  - [Transfer Agent](#transfer-agent)
- [Customization](#customization)
  - [Integrating a New Transport](#integrating-a-new-transport)
  - [Supporting a New Cache Layout](#supporting-a-new-cache-layout)
- [Current Limitations](#current-limitations)

## Workflow

The control plane — discovery, per-request metadata, and completion notices — is ZeroMQ. The data plane is one-sided NIXL writes issued by the context worker into memory the generation worker has registered. The two never share a channel.

1. Context phase completes computation, KV cache stays in device memory awaiting transmission. The context worker publishes only the endpoint where it can be reached, on `ContextPhaseParams.disagg_info_endpoint`.
2. On first contact with a context instance, the generation worker fetches that instance's `RankInfo` — topology, attention geometry, serialized page table, transfer agent descriptor — from the context rank 0, then registers its own `RankInfo` with every context rank. The result is cached per endpoint, so only the first request pays for the round trip.
3. Generation phase requests KV cache for specific tokens, sending its destination block IDs to each overlapping context rank.
4. Context computes both source and destination addresses locally and submits a one-sided NIXL transfer. On the default path that transfer carries one descriptor per cache fragment; only the optional bounce path coalesces a request into a single contiguous buffer and a single write. All address arithmetic happens on the context side; the generation worker's CPU is not involved.
5. Context reports completion over the control plane. The receive session completes once every expected writer has reported.
6. Generation phase resumes computation, context releases KV cache. Both sides run an allgather first, so every rank agrees on the outcome before any state transition.

## Key Components

### Transceiver

`KvCacheTransceiverV2` in `transceiver.py`. Coordinates the sending and receiving of cache among the ranks of one executor, and implements the same interface `PyExecutor` drives for both runtimes. Every one of its methods runs on the executor main loop.

### Messenger

`native/messenger.py` wraps ZeroMQ `ROUTER`/`DEALER` sockets and owns a listener thread per socket. `native/transfer.py` builds the actual endpoints on top: `Sender` on the context side, `Receiver` on the generation side, and a rendezvous server on rank 0. Listener handlers mutate session state and enqueue work; they never block on a transfer.

### Peer Registrar and Mappers

`native/peer.py` pairs each rank with the peer ranks whose share of every parallel dimension overlaps its own (`layout_mapping.fan`), checks at registration that the two ranks' page layouts correspond, and builds the mapper of each matched pool pair. Which bytes a rank sends a peer rank follows from the two ranks' page layouts alone (`kv_cache/layout_mapping.py`, `map_buffers`): every buffer is the local pieces of tensors whose axes say how they split over tensor and context ranks, and of the indices both ranks hold the source sends those it is elected for among the paired ranks holding them, so nothing is sent twice and no cache kind or model is special-cased. `native/runs.py`'s `RunsMapper` adds the resulting byte runs to each block's or slot's base pointers.

### Page Table

`KVCacheManagerV2.get_page_table()` describes how the manager lays out its pages (`kv_cache/page_layout.py`): each buffer's tensors, where it lies in its pool slot, and where the rank stands. `resource/kv_extractor.py` turns that table into the views of `resource/page.py`, which index the pools and layer groups for transfer, and turns block IDs into slot pointers; `resource/cache_reuse.py` reports the cached prefix length uniformly, so context-side reuse, generation-side reuse, and sliding-window stale prefixes all resolve without a special case.

### Transfer Agent

Unidirectional read/write protocol facility, defined by the `BaseTransferAgent` ABC in `base/agent.py`. It provides memory registration, remote agent loading, and transfer request submission. NIXL accesses the system through this facility, either through the `tensorrt_llm_transfer_agent_binding` nanobind module (preferred, releases the GIL while waiting) or through the Python `nixl` package when `TRTLLM_USE_PY_NIXL_KVCACHE=1`.

## Customization

### Integrating a New Transport

Inherit `BaseTransferAgent`. Any transport that can perform one-sided writes into registered remote memory fits; everything above the agent is transport-agnostic. Note that agent selection today happens in `nixl/agent.py` and `TransferWorker._setup_transfer_engine` constructs a NIXL agent directly, so a new backend needs a hook there as well — there is no configuration-driven agent registry yet.

Note also that `cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/` outlives the C++ transceiver: it builds the nanobind module that `base/agent.py` and `nixl/_agent_cpp.py` import.

### Supporting a New Cache Layout

A new cache's manager declares how its buffers spread over ranks by overriding a few `KVCacheManagerV2` hooks: `_page_buffer_tensors` (built from `page_layout`'s `head_major`, `token_major`, `sectioned` and `replicated`, or declaring axes of its own), `_page_untransferred_roles` and `_page_rebuilds_untransferred` for buffers that never move, and, when its layers or heads differ from the base manager's, `_page_layer_ids` and `_page_kv_heads`. The base manager declares only the roles it registers itself (K/V and their block scales split by head, `Role.INDEX_KEY` replicated); a role a manager adds without declaring it is refused when the page table is built, rather than guessed. What a transfer of one request moves is the manager's to say as well, through its `disagg_*` methods: how much of the prompt (`disagg_transfer_length`), which buffers the receiver rebuilds instead (`disagg_untransferred_buffers`), whether the first tokens travel (`disagg_sends_first_tokens`), which slot holds the recurrent state (`disagg_state_slot`), and whether the pages can move in pieces while the prompt is computed (`disagg_pipelined_transfer_blocker`). The transfer itself needs no change. Peers whose layouts do not correspond — a buffer one side transfers that the other lacks, tensors of different extents, or a pairing that leaves a destination short — are refused at registration rather than silently transferring a subset.

To test a new cache, add a `tests/unittest/disaggregated/test_<model>_kv_transfer.py` on `kv_transfer_harness.run_kv_transfer_test` with a manager factory and hooks that write and read the cache through the model's own tensor views, over the context/generation topologies the model is served with. The harness first checks the layouts the managers declare against each other byte for byte (`tests/unittest/utils/page_layout_oracle.py`, usable on its own for CPU checks of hand-written or manager-built layouts), then runs a real transfer and hands the received caches to the read-back hook.

Unit tests for these seams live in `tests/unittest/disaggregated/` — byte-level transfer checks, rank pairing, page-table construction, and multi-process end-to-end coverage; `tests/unittest/_torch/executor/kv_cache/test_layout_mapping.py` checks the mapping byte by byte across TP, attention DP, Helix, PP and uneven head splits.

## Current Limitations

- **NIXL only.** MPI, UCX, and Mooncake deployments stay on the C++ transceiver.
- **One KV slice per request.** The whole prompt moves in a single transfer after prefill completes. The session machinery is multi-slice capable and `KVSlice.layer_range` exists, but no producer sets it — there is no layer-wise or chunked transfer overlapping compute yet.
- **Both sides must run the same build and the same transceiver runtime.** `RankInfo` is serialized field for field and carries no version tag, so there is nothing to negotiate against. Pairing a C++ context worker with a Python generation worker fails on the first request, because the context worker publishes no `disagg_info_endpoint`.
- **Submitted NIXL operations cannot be aborted.** Cancellation is cooperative: an operation already handed to the transfer agent runs to completion. Lifecycle ownership is not complete either — safe retirement of a session after an ambiguous failure or a cancellation is not fully enforced, so do not assume the KV pages are reclaimed at a well-defined point on those paths.
