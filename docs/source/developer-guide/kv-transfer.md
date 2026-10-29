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
2. On first contact with a context instance, the generation worker fetches that instance's `RankInfo` — topology, the manager's page table, transfer agent descriptor — from the context rank 0, and from each context rank that may write its pages that rank's own `RankInfo`, whose page layout says what it sends (which sizes its share of a bounce region, and a pairing that would leave the generation rank short is refused here); it then registers its own `RankInfo` with every context rank. The result is cached per endpoint, so only the first request pays for the round trips.
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

`native/peer.py` pairs each rank with the peer ranks whose share of every parallel dimension overlaps its own (`layout_mapping.fan`), checks at registration that the two ranks' page layouts correspond, and groups the runs a rank sends a peer rank by the pool each run leaves and the peer's pool it lands in. Which bytes a rank sends a peer rank follows from the two ranks' page layouts alone (`kv_cache/layout_mapping.py`, `map_buffers`): every buffer is the local pieces of tensors whose axes say how they split over tensor and context ranks, and of the indices both ranks hold the source sends those it is elected for among the paired ranks holding them, so nothing is sent twice and no cache kind or model is special-cased. `native/runs.py`'s `RunsMapper` adds the resulting byte runs to each block's or slot's base pointers.

### Page Table

`KVCacheManagerV2.get_page_table()` describes how the manager lays out its pages (`kv_cache/page_layout.py`): each buffer's tensors, where it lies in its pool slot, and where the rank stands. Native uses that table as it is: it travels to peers in `RankInfo` (`page_table_to_dict`), and `resource/utils.py` turns block IDs into slot pointers from its device pools; `resource/cache_reuse.py` reports the cached prefix length uniformly, so context-side reuse, generation-side reuse, and sliding-window stale prefixes all resolve without a special case.

### Transfer Agent

Unidirectional read/write protocol facility, defined by the `BaseTransferAgent` ABC in `base/agent.py`. It provides memory registration, remote agent loading, and transfer request submission. NIXL accesses the system through this facility, either through the `tensorrt_llm_transfer_agent_binding` nanobind module (preferred, releases the GIL while waiting) or through the Python `nixl` package when `TRTLLM_USE_PY_NIXL_KVCACHE=1`.

## Customization

### Integrating a New Transport

Inherit `BaseTransferAgent`. Any transport that can perform one-sided writes into registered remote memory fits; everything above the agent is transport-agnostic. Note that agent selection today happens in `nixl/agent.py` and `TransferWorker._setup_transfer_engine` constructs a NIXL agent directly, so a new backend needs a hook there as well — there is no configuration-driven agent registry yet.

Note also that `cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/` outlives the C++ transceiver: it builds the nanobind module that `base/agent.py` and `nixl/_agent_cpp.py` import.

### Supporting a New Cache Layout

A new cache's manager declares how its buffers spread over ranks by overriding a few `KVCacheManagerV2` hooks: `_page_buffer_tensors` (built from `page_layout`'s `head_major`, `token_major`, `sectioned` and `replicated`, or declaring axes of its own), `_page_untransferred_roles` for buffers that never move because they hold nothing a peer needs and `_page_derived_roles` for those the receiver rebuilds from the transferred ones (a manager naming any overrides `rebuild_disagg_imported_cache`, or its page table is refused), and, when its layers or heads differ from the base manager's, `_page_layer_ids` and `_page_kv_heads`; recurrent state splits over the ranks `_page_state_split` names, as its pools are sized. The base manager declares only K/V and their block scales, split by head and head-major (HND) as the TRTLLM attention kernels write them (a manager whose kernels write them token-major, NHD, declares `token_major`); every other role, `Role.INDEX_KEY` included, is declared by the manager registering it, and a role a manager adds without declaring it, or tensors whose pieces on a rank do not add up to a buffer's size, are refused when the page table is built, rather than guessed. Peers pair buffers by global layer id, so each cache of a model layer goes by the same global ids on every pipeline stage of every peer, and peers naming a cache both transfer by other ids are refused: `_page_layer_ids` derives it from the model layer alone, or from the model layer and attention type for a manager keeping several caches per model layer (`_virtual_layers`, from `_layer_attn_to_layer_id`, which such a manager sets on every stage), never from what a stage happens to hold. Each layer group's shard, which names its blocks to the lender, follows from the indices its buffers' tensors hold, so ranks holding the same bytes share names. What a transfer of one request moves is the manager's to say as well, through its `disagg_*` methods (a transfer leaving part of the prompt, a buffer or the first tokens behind needs the receiver's executor to recompute them, which today only CSA2's remote-tail replay does): how much of the prompt (`disagg_transfer_length`; a manager moving part of it is refused under Helix), which buffers the receiver rebuilds instead (`disagg_untransferred_buffers`), whether the first tokens travel (`disagg_sends_first_tokens`), which slot holds each state layer group's recurrent state (`disagg_state_slot`, by default the one the request's cache holds; a state group without one raises rather than moving nothing), and whether the pages can move in pieces while the prompt is computed (`disagg_pipelined_transfer_blocker`, by default not where the model keeps recurrent state, `model_has_recurrent_state`); caches the generation worker derives from imported ones are rebuilt in `rebuild_disagg_imported_cache` before a request's first forward. The transfer itself needs no change. It moves only what lies in the manager's V2 pools: request state the generation worker reads from anywhere else never travels, so a manager keeping such state refuses disaggregated serving, as MiniMax M3 does for its plain-tensor index-V. Peers whose layouts do not correspond — a buffer one side transfers that the other lacks, a cache of a model layer named by other global layers or a global layer naming another model layer, tensors of different extents or element types (a tensor states its type with `dtype`), KV of another element type or quantization as the managers report them (`_page_dtype`, `_page_quant_mode`; a mixed-precision manager reports its high-precision type), other sub-pages (a buffer of several sub-pages per block is declarable only replicated) or a layer group of another kind, or paired source ranks that do not hold every index of a tensor a destination holds — are refused at registration rather than silently transferring a subset or bytes of another meaning. A layer, or a cache of a layer, that a destination holds and no source transfers is not refused by the layouts; it is the receiver's to rebuild, as CSA2's generation workers rebuild the sliding caches a context-only worker keeps none of. How many tokens each side keeps of a windowed layer is not compared: retention margins such as reuse backoff and draft tokens legitimately differ between the sides, and a transfer moves the blocks both sides still hold.

To test a new cache, add a `tests/unittest/disaggregated/test_<model>_kv_transfer.py` on `kv_transfer_harness.run_kv_transfer_test` with a manager factory and hooks that write and read the cache through the model's own tensor views, over the context/generation topologies the model is served with. The harness first checks the layouts the managers declare against each other byte for byte (`tests/unittest/utils/page_layout_oracle.py`, usable on its own for CPU checks of hand-written or manager-built layouts), then runs a real transfer and hands the received caches to the read-back hook.

Unit tests for these seams live in `tests/unittest/disaggregated/` — byte-level transfer checks, rank pairing, page-table construction, and multi-process end-to-end coverage; `tests/unittest/_torch/executor/kv_cache/test_layout_mapping.py` checks the mapping byte by byte across TP, attention DP, Helix, PP and uneven head splits.

## Current Limitations

- **NIXL only.** MPI, UCX, and Mooncake deployments stay on the C++ transceiver.
- **Chunked transfer only through pipelined transfer.** By default the whole prompt moves in a single transfer after prefill completes. `cache_transceiver_config.enable_pipelined_transfer` sends each completed prefill chunk while later chunks compute; the transceiver refuses it at start with the Python bounce buffer, context parallelism, block reuse other than `all_reusable`, or a manager whose `disagg_pipelined_transfer_blocker` names a reason (by default a model keeping recurrent state). `KVSlice.layer_range` exists but no producer sets it, so there is no layer-wise transfer.
- **Both sides must run the same build and the same transceiver runtime.** `RankInfo` is serialized field for field and carries no version tag, so there is nothing to negotiate against. Pairing a C++ context worker with a Python generation worker fails on the first request, because the context worker publishes no `disagg_info_endpoint`.
- **Submitted NIXL operations cannot be aborted.** Cancellation is cooperative: an operation already handed to the transfer agent runs to completion. Lifecycle ownership is not complete either — safe retirement of a session after an ambiguous failure or a cancellation is not fully enforced, so do not assume the KV pages are reclaimed at a well-defined point on those paths.
