# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-loop behaviour a model may opt into, read once by the engine.

A model declares its hooks as ``runtime_hooks`` (an instance attribute or a class
attribute). Every field defaults to off, and a model that declares nothing gets
``RuntimeHooks()``: the step loop, the samplers and the attention metadata then
run exactly as they do without this module.

The hooks change *where* and *how* the step loop moves data, or how the engine
sizes and manages its resources (device memory, cache pages, scheduling), never
what a forward computes. Each one is armed only by a model that has measured it
on its own deployment; none is a general default.

A modeling-v2 target declares them on the registered (outer) model class, the
class the loader resolves and the engine holds as ``model_engine.model``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Protocol, Sequence, Tuple

if TYPE_CHECKING:
    import torch


class SpecStepFold(Protocol):
    """Speculative step-input staging folded into the previous step's forward.

    A one-model speculative step ends by storing its accepted tokens, draft
    tokens and lengths into the sampler's per-slot store, and the next step
    begins by gathering those values back into the engine's flat input buffers.
    A model that implements this protocol does both inside its last draft pass:
    it writes the sampler store and stages the next step's inputs directly, and
    the engine and the sampler skip their copies on the steps where it did.

    The object is owned by the model; the engine and the sampler only call it.
    Every method is host-side bookkeeping or a single kernel launch on the
    current stream.
    """

    def bind_engine_buffers(
        self,
        input_ids: "torch.Tensor",
        draft_tokens: "torch.Tensor",
        position_id_offsets: "torch.Tensor",
        kv_lens_offsets: "torch.Tensor",
        max_batch_size: int,
    ) -> None:
        """Receive the engine's persistent speculative input buffers, once, at construction."""

    def bind_sampler_store(self, store: object) -> None:
        """Receive the speculative sampler's persistent per-slot store, once, at construction."""

    def begin_step(self, seq_slots: Optional[list[int]], num_rows: int) -> bool:
        """Start a step of ``num_rows`` generation rows.

        ``seq_slots`` are those rows' sequence slots on a decode-only step whose
        rows all have one, else None: the forward then stores nothing for them.
        Returns whether the previous step's forward staged its inputs for
        ``seq_slots``; the engine additionally checks that this step adds no
        inputs of its own before it skips its gathers.
        """

    def gather(
        self,
        new_tokens: "torch.Tensor",
        next_draft_tokens: "torch.Tensor",
        new_tokens_lens: "torch.Tensor",
        previous_slots: "torch.Tensor",
        previous_pos_indices: "torch.Tensor",
        previous_batch_len: int,
        tokens_per_gen_step: int,
        draft_token_buffer_width: int,
        num_tokens: int,
        num_draft_tokens: int,
        first_previous_row: int,
    ) -> bool:
        """Do the engine's previous-batch gathers in one launch.

        Writes the engine buffers exactly as the engine's own gathers would,
        including zeroing both offset buffers outside the gathered range.
        Returns False, having written nothing, when it cannot take the step.
        """

    def sampler_rows_stored(self, outputs: dict, num_skip: int, seq_slots: list[int]) -> bool:
        """Whether the forward already stored this step's outputs for ``seq_slots`` into the sampler store."""

    def scatter(
        self,
        seq_slots: "torch.Tensor",
        new_tokens: "torch.Tensor",
        next_new_tokens: "torch.Tensor",
        new_tokens_lens: "torch.Tensor",
        next_draft_tokens: "torch.Tensor",
    ) -> bool:
        """Store the step's outputs into the sampler store in one launch.

        Returns False, having written nothing, when it cannot take the step.
        """


class MsaKvUpdate(Protocol):
    """Advance the MSA decode kernels' per-request state after the KV lengths moved.

    Called by ``MiniMaxM3MsaSparseAttentionMetadata.on_update_kv_lens`` with the
    metadata's own buffers; recomputes the attended lengths, the new tokens'
    cache slots and their valid-block counts in one launch, exactly as the
    metadata's own update does. Returns False, having written nothing, when it
    cannot take the step.
    """

    def __call__(
        self,
        kv_lens: "torch.Tensor",
        kv_lens_staged: "torch.Tensor",
        seq_lens: "torch.Tensor",
        q_batch_row: "torch.Tensor",
        q_intra: "torch.Tensor",
        block_table: "torch.Tensor",
        tokens_per_block: int,
        batch: int,
        total_q: int,
        msa_seq_lens: "torch.Tensor",
        cu_kv_lens: Optional["torch.Tensor"],
        out_cache_loc: "torch.Tensor",
        n_valid_blocks: Optional["torch.Tensor"],
    ) -> bool: ...


class MsaKvUpdateTail(Protocol):
    """``MsaKvUpdate`` that also writes -1 over ``out_cache_loc[tail_start:]`` (the slots past the live ones, which
    the metadata otherwise invalidates with a fill of its own inside the forward) in the same launch.

    The metadata calls it only with ``total_q <= tail_start < len(out_cache_loc)``. Returns False, having written
    nothing, when it cannot take the step; the metadata then fills the tail and runs the plain update.
    """

    def __call__(
        self,
        kv_lens: "torch.Tensor",
        kv_lens_staged: "torch.Tensor",
        seq_lens: "torch.Tensor",
        q_batch_row: "torch.Tensor",
        q_intra: "torch.Tensor",
        block_table: "torch.Tensor",
        tokens_per_block: int,
        batch: int,
        total_q: int,
        msa_seq_lens: "torch.Tensor",
        cu_kv_lens: Optional["torch.Tensor"],
        out_cache_loc: "torch.Tensor",
        n_valid_blocks: Optional["torch.Tensor"],
        tail_start: int,
    ) -> bool: ...


@dataclass(frozen=True)
class MsaRuntimePolicy:
    """MiniMax-M3 MSA metadata preparation a model may opt into.

    Every option changes how a step's selection and cache-write tables reach
    the device, not which blocks are attended.
    """

    # A step whose every query token has at most top-k valid blocks attends all
    # of them, so its selection is the identity table: prepare stages that table
    # instead of a scoring plan, and decode CUDA graphs are keyed by short and
    # long contexts.
    short_sequence_identity: bool = False
    # The model's one-launch replacement for the update after the KV lengths move.
    kv_update: Optional[MsaKvUpdate] = None
    # Skip a block-table copy whose inputs equal the previous step's.
    skip_unchanged_block_offsets: bool = False
    # Build each KV pool view once per pool geometry instead of on every attention call.
    reuse_pool_views: bool = False
    # Decode steps whose summed attended length (over the step's requests) reaches this many tokens replay a
    # CUDA graph family of their own, so a model can take a different long-context path there. None: no such
    # family. Both paths must be exact for every length: the family only decides which one a captured graph holds.
    long_family_min_total_tokens: Optional[int] = None
    # kv_update that also invalidates the out-cache-loc tail in its launch, instead of the metadata's own fill
    # before it (one launch fewer per step).
    kv_update_tail: Optional[MsaKvUpdateTail] = None
    # Prefill indexer proxy plan (the max-score fmha_sm100 pass) of a backend that reads these (a model's own MSA
    # backend, RuntimeHooks.sparse_attention). Max-score plans need no split merge, so every option below gives the
    # same score table bits. prefill_proxy_qo_tile: maxscore_qo_tile for a step whose longest planned query span
    # exceeds 128 rows, when the loaded planner accepts it; prefill_proxy_kv_splits is then that plan's num_kv_splits.
    # prefill_proxy_kv_splits_by_rows: ((max_rows, top_split, min_kv_pages_per_split), ...) in ascending max_rows: the
    # first entry covering the step's longest planned query span gives S and F, and the plan takes S halved until
    # ceil(max kv_len / page) / S >= F (down to 1), over every planned request. A span of at most 128 rows takes it
    # under the planner's own tile; a longer span takes it only with the tile.
    prefill_proxy_qo_tile: Optional[int] = None
    prefill_proxy_kv_splits: Optional[int] = None
    prefill_proxy_kv_splits_by_rows: Optional[Tuple[Tuple[int, int, int], ...]] = None


@dataclass(frozen=True)
class SparseAttentionClasses:
    """A model's own classes for its sparse attention algorithm, used in place of those the registry resolves.

    Each class must derive from the one the sparse attention registry resolves for the model's configuration, so the
    engine code that checks for the registry's class still holds; the engine refuses a class that does not. A model
    whose attention layers construct ``backend`` declares it here so the step metadata the engine builds
    (``backend.Metadata``) is the one those layers read.
    """

    # The attention backend class whose ``Metadata`` the engine builds the step metadata from.
    backend: Optional[type] = None
    # The target model's KV cache manager class.
    kv_cache_manager: Optional[type] = None
    # Wraps the decoder warmup (``DecoderRunner.warmup``); called with the runner, returns a context manager.
    warmup_scope: Optional[Callable[[Any], Any]] = None
    # Replaces the sparse indexer prewarm the decoder warmup runs (called with the runner and its KV cache manager).
    indexer_prewarm: Optional[Callable[[Any, Any], None]] = None


class KvBalancedAdmissionPlanner(Protocol):
    """Chooses the requests attention-DP KV-balanced admission admits, and their ranks.

    Used by ``DefaultADPRouter`` in place of its own window and planner when
    ``attention_dp_config.enable_kv_balanced_admission`` is on. Every rank calls it
    with the same all-gathered inputs, so it must be deterministic.
    """

    def window(self, max_new_requests: int, free_slots: int) -> int:
        """Number of requests at the head of the queue to choose among."""

    def plan(
        self,
        *,
        candidate_tokens: list[int],
        candidate_forced: list[bool],
        rank_tokens: list[int],
        rank_num_requests: list[int],
        max_requests_per_rank: int,
        max_new_requests: int,
        candidate_groups: list[int],
    ) -> list[tuple[int, int]]:
        """``(candidate_index, rank)`` pairs in admission order.

        The arguments are those of ``adp_router.plan_kv_balanced_admission``.
        """


class DecodePrologue(Protocol):
    """Replaces the decode step's input prologue with one launch.

    Called first in the step's input preprocessing, inside CUDA graph capture
    as well. It decides on the host before any device work; returning False
    keeps the native prologue, and returning True means every engine-visible
    tensor the native prologue writes holds the value it would have written.
    """

    def __call__(
        self, preparer: Any, inputs: Any, *, enable_spec_decode: bool, runtime_draft_len: int
    ) -> bool:
        """``preparer`` is the ``DecoderInputPreparer`` running the step's preprocessing."""


class ExecutorEvents(Protocol):
    """Points in the executor's life a model may act at."""

    def after_warmup(self) -> None:
        """After warmup and CUDA graph capture, before the first request is served."""

    def before_overlap_loop(self) -> None:
        """Right before the overlap executor loop starts, on every rank; must not raise."""

    def at_shutdown(self) -> None:
        """When the executor loop has completed, during cleanup; must not raise."""


@dataclass(frozen=True)
class RuntimeHooks:
    """The step-loop hooks a model opts into. All off by default."""

    # Pack a decode step's small host-to-device input copies into one upload and
    # one scatter launch, issued just before the CUDA graph replay.
    pack_step_inputs: bool = False
    # Issue the step-input pack's host copies with one parallel scatter (a program per copy) instead of the
    # serial one-CTA scatter, when their destinations do not overlap (pack_step_inputs).
    pack_parallel_scatter: bool = False
    # Issue the step-input pack inside the replayed graph (a prologue copies a pinned window and scatters it), so no
    # operation is left between the decode graphs (needs pack_parallel_scatter).
    pack_in_graph: bool = False
    # With pack_in_graph: a side copy stream uploads the pinned window into the prologue's own device buffer while the
    # previous graph still runs, so the prologue only waits on that copy and scatters device to device (no host copy
    # at the head of the replay).
    pack_early_copy: bool = False
    # With pack_in_graph: the pack keeps a step epoch, one int64 device counter the graph's prologue increments once per
    # replay and never resets, so in-graph per-step flags can be tagged with it instead of cleared between steps.
    pack_step_epoch: bool = False
    # With pack_early_copy: the model's one-launch prologue, called inside the capture as
    # ``op(window, epoch_or_None, max_desc)``. It writes the window's host copies, then its device copies, zeroes the
    # window's header and increments the epoch, which the default prologue does with two scatter launches and a fill.
    pack_prologue_op: Optional[Callable[[torch.Tensor, Optional[torch.Tensor], int], None]] = None
    # The step-input pack launches no Triton kernel: eager flushes take the native serial scatter, and the in-graph
    # prologue runs only as pack_prologue_op (step_inputs.StepInputPack cpp_scatter).
    pack_cpp_scatter: bool = False
    decode_prologue: Optional[DecodePrologue] = None
    executor_events: Optional[ExecutorEvents] = None
    # Skip host-to-device metadata copies whose values the destination already holds.
    skip_unchanged_h2d: bool = False
    # With skip_unchanged_h2d: a changed packed 2-D upload stages only each row's changed column span (a page table that
    # gained a page re-sends a few bytes, not every padded row), so the pack keeps fitting the in-graph window.
    h2d_changed_span: bool = False
    # A speculative step under the overlap scheduler re-derives its KV-length metadata (on_update_kv_lens) once,
    # after the forward's KV-length correction, instead of also before it: the first pass's device writes are all
    # rewritten by the second.
    kv_lens_update_once: bool = False
    # The overlap scheduler's speculative step correction (position_ids += position offsets, kv_lens += KV-length
    # offsets) as the model's one launch, called as ``op(positions, position_offsets, kv_lens, kv_lens_offsets)`` on the
    # generation rows; it returns False to decline (the two in-tree adds run).
    spec_offsets_op: Optional[
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], bool]
    ] = None
    # Run the overlap executor loop with the execution stream current, so the
    # loop's own stream work and the forward are ordered on one stream.
    loop_on_execution_stream: bool = False
    # Under multi-rank MPI attention DP, rank 0 sends each admitted request's prompt only to the rank
    # that owns it instead of broadcasting every prompt to every rank (RequestBroadcaster).
    prompts_to_owners: bool = False
    # After autotuning, keep the fastest profiled tactic per shape on every
    # tensor-parallel rank, so the ranks capture the same kernels.
    merge_tuned_tactics_across_ranks: bool = False
    # With TLLM_AUTOTUNER_CACHE_PATH set, save the autotuner cache again after the
    # FlashInfer MXFP8 passes (warmup and the decode-graph pass) tune, so their
    # tactics persist across starts instead of being re-profiled on every start.
    persist_flashinfer_mxfp8_tactics: bool = False
    # Called once with this process's rank before the warmup tuning passes, to preload tuned tactics (the model's
    # bundled pin files) into the autotuner; shapes it covers are served instead of profiled.
    install_tactic_pins: Optional[Callable[[int], Any]] = None
    # Streaming: also emit a response on decoding iterations 2..N, so the first
    # counted text reaches the client one step after the first token instead of
    # one stream interval later (a reasoning parser holds a leading reasoning
    # tag back, so the iteration-1 chunk can carry no content). Emission only;
    # tokens are unchanged. 1 keeps the stock schedule.
    first_content_emit_iters: int = 1
    # On SM107, run FlashInfer's CuTe DSL MXFP8 GEMM above 32 rows at the pinned tuned tactics of
    # custom_ops/mxfp8_cute_pinned_tactics (bitwise the default tactic's output, faster). The GEMM runner does not
    # know the model, so the executor arms the table process-wide when the model opts in.
    mxfp8_pinned_tactics: bool = False
    # After warmup, log every CuTe DSL / Triton compile as a late compile (tensorrt_llm/_torch/late_compile.py).
    late_compile_detect: bool = False
    # Skip the per-step Philox seed and offset uploads on all-greedy
    # one-model speculative batches (the argmax path never reads them).
    skip_greedy_rng_upload: bool = False
    # Stage the one-model speculative sampler's per-step Philox seed and offset
    # uploads into the step-input pack (pack_step_inputs), or into the steady MTP
    # stager's slab on the steps it prepares, instead of four separate copies.
    pack_spec_rng_uploads: bool = False
    # The one-model speculative step's small per-step uploads: the accepted-draft
    # counts join the step-input pack, and the spec sampler keeps its slot ids in
    # a device buffer it re-uploads only when they change.
    spec_small_h2d: bool = False
    # Stage a decode step's inputs in fewer launches: attention metadata through a
    # private copy stream; a graph replay's ids, positions and speculative inputs in one launch each.
    stage_decode_inputs: bool = False
    # Capture the decode CUDA graphs (at most 256 tokens per step) on a high-priority stream.
    prioritize_decode_graphs: bool = False
    # The one-model speculative metadata re-uploads its per-request slot-id table (batch_slot_ids, read by rejection
    # sampling and the penalties) only when the batch's slot ids change (steady decode keeps them), instead of
    # building a pinned tensor and copying it every step.
    spec_slot_ids_changed_only: bool = False
    # Under attention DP, the graph-shape exchange also agrees on replaying one captured
    # generation-only graph, whose shape fixes every rank's input counts; a speculative step
    # then skips the all-gather of those counts.
    skip_adp_count_gather: bool = False
    # A generation-only one-model speculative step verifies its hidden states as they are: the
    # decoder inputs gather every row there, in order.
    skip_verify_row_gather: bool = False
    # Speculative step-input staging folded into the forward (see SpecStepFold).
    spec_step_fold: Optional[SpecStepFold] = None
    # Copy the one-model speculative sampler's per-slot store to the host on the sampler's D2H side
    # stream, so those copies overlap the next step's input staging instead of preceding it.
    sampler_d2h_side_stream: bool = False
    # Time the iteration log's device step with one timing event per iteration boundary, recorded on a side
    # stream that waits for the current stream, instead of an end and a start event back to back on the
    # current stream (the same instant, with no GPU work between them). The reported step times are unchanged.
    iteration_timing_one_event: bool = False
    # Record the per-request perf-metrics timing events (return_perf_metrics) on a side stream that waits for
    # the current stream, so a timing record does not cost the current stream. The elapsed times are unchanged.
    perf_metrics_timing_side_stream: bool = False
    # The one-model speculative sampler's tail bookkeeping without its no-op launches, bitwise the same outputs
    # (copied to spec_metadata.tail_bookkeep).
    spec_tail_bookkeep: bool = False
    # Write the context one-hot draft-probability rows with index_fill_ instead of an advanced-index put that
    # synchronizes the host with the GPU (SpecMetadata.context_onehot_index_fill, set by the decoder runner on this
    # model's spec metadata only).
    context_onehot_index_fill: bool = False
    # The DFlash / DSpark worker computes a request's drafter context allocation on the host where it is known instead
    # of reading it back from the device (DFlashWorker.enable_host_ctx_alloc, called by the decoder runner on this
    # model's spec worker only).
    dflash_host_ctx_alloc: bool = False
    # With KV cache manager V2 and block reuse: a waiting context request whose reusable prefix sits in the host tier
    # has it migrated to the GPU off the execution stream (KVCacheManagerV2.prefetch_to_gpu) and is held out of
    # scheduling for a few rank-agreed steps while the copy runs, so the swap does not stall the whole step. Read by
    # this model's executor at construction.
    kv_gpu_prefetch: bool = False
    # MiniMax-M3 MSA metadata preparation.
    msa: MsaRuntimePolicy = MsaRuntimePolicy()
    # Device memory, in bytes, the executor leaves outside the PyTorch cap for driver allocations made after
    # setup (streams, events, copy launches of the KV cache manager and transfer agents, CUDA IPC mappings of
    # peer buffers, kernel attribute and module loads), up to the memory free at that point. None keeps the
    # engine's default margin (1 GiB). The KV cache budget does not change.
    torch_mem_safety_margin: Optional[int] = None
    # With long-prefill demotion on (TLLM_V2_CTX_DEMOTE_UNCACHED_TOKENS > 0), the
    # KV cache manager V2 scheduler classifies context requests by the prefix cache rather than
    # the router's estimate alone: an estimate above the threshold is checked once against the
    # radix tree (read-only probe), a request that already holds its cache is judged by its exact
    # remainder, and a request ordered as short whose first preparation shows a long remainder is
    # retried after the rest of the iteration. Without demotion it does nothing.
    exact_context_demotion: bool = False
    # Replaces the attention-DP KV-balanced admission's window and planner
    # (attention_dp_config.enable_kv_balanced_admission); None keeps the engine's.
    kv_balanced_admission_planner: Optional[KvBalancedAdmissionPlanner] = None
    # A subclass of the KV cache manager class the engine resolves for the model's own KV cache; the
    # engine builds this subclass instead (KvCacheCreator rejects a class that is not a subclass). A
    # separate draft model's KV cache manager is resolved from the draft model and does not use it.
    # None keeps the resolved class.
    kv_cache_manager_cls: Optional[type] = None
    # The model's own sparse attention backend, KV cache manager and warmup steps (None: the registry's).
    sparse_attention: Optional[SparseAttentionClasses] = None
    # Standalone-drafter KV pool: ceil(heads / tp) per-rank cost and the history-first context resize
    # (kv_cache_manager_v2.py). Passed by the KV-cache creator to this model's draft manager and its sizing.
    draft_pool_fixes: bool = False
    # DFlash / DSpark draft KV manager: reserve the drafter's block width per generation step and keep page 0 as an
    # unowned guard page, so drafter writes through unallocated block-table entries cannot land in another request's
    # page (kv_cache_manager_v2.py). Passed by the KV-cache creator to this model's draft manager and its sizing.
    draft_kv_reserve_guard: bool = False
    # A separate draft KV cache manager (KVCacheManagerV2) skips a decode-only block-table copy whose destination,
    # copied rows and those rows' page indices equal the previous copy's (the standalone DSpark drafter's table is
    # unchanged on most decode steps).
    skip_unchanged_draft_block_offsets: bool = False
    # With skip_unchanged_draft_block_offsets: a decode-only draft block-table copy whose rows changed in a few page
    # indices (the drafter's sliding window) rewrites the rows with one host-to-device row copy instead of the copy
    # kernel.
    patch_unchanged_draft_block_offsets: bool = False
    # Under disaggregated serving, ship a standalone drafter's context window (its separate draft
    # KV pool) alongside the target's KV, so the generation worker drafts from the prompt instead
    # of an empty window. Takes effect only with a separate draft KV cache manager on both sides
    # (see disaggregation/draft_kv.py).
    disagg_draft_kv: bool = False
    # Commit a disaggregated-generation request's received prompt blocks for reuse right after the step's forward
    # is launched instead of before it: the commit is host work, and the request itself never reads the reuse
    # tree (DisaggTransferCoordinator.commit_deferred_gen_receives).
    defer_gen_reuse_commit: bool = False
    # A non-blocking request fetch skips batch_wait_timeout_ms while the executor already holds context work for its
    # next step (a waiting request that is not generation-only, or an active request still in CONTEXT_INIT): the
    # wait then only delays that step. Read by this model's executor at construction; kill switch
    # TRTLLM_DISABLE_CTX_BATCH_WAIT_SKIP=1.
    ctx_batch_wait_skip: bool = False
    # Post-launch admission on an attention-DP disaggregated generation server: build fetched generation-only
    # requests and prepare their gen-init resources after the step's forward is launched
    # (PyExecutor._init_postlaunch_admit). Kill switch TRTLLM_DISABLE_POSTLAUNCH_ADMISSION=1.
    postlaunch_admission: bool = False
    # Time the executor (per-request perf metrics, the iteration log's device step) with %globaltimer stamps
    # recorded on a follower stream instead of timed CUDA events, whose record waits for the stream to drain
    # (perf_metrics_manager.GpuTimestamp). Read by this model's executor at construction; kill switch
    # TRTLLM_DISABLE_GPU_TIMESTAMPS=1.
    gpu_timestamps: bool = False
    # Shortest-uncached-first context scheduling also counts the prompt prefix a KV connector can load when it ranks a
    # first chunk, and charges a store-served first chunk only the tokens its forward computes (context fill). Passed
    # to the KV cache manager V2 scheduler; TLLM_CTX_SHORTEST_FIRST_STORE_AWARE=1 / 0 turns it on / off either way.
    ctx_store_aware_ranking: bool = False
    # Re-probe the stored prefix of waiting first-chunk requests so a store that leases looked-up keys keeps it
    # (attention DP with a loading KV connector only). Passed to the KV cache manager V2 scheduler;
    # TLLM_STORE_LEASE_TOUCH=1 / 0 turns it on / off either way.
    store_lease_touch: bool = False
    # On a disaggregated context worker, a started context-only request stays a preemption victim, and the victim is
    # the eligible request with the least prefill progress. Passed to the KV cache manager V2 scheduler;
    # TLLM_V2_CTX_ONLY_PREEMPTION and TLLM_V2_CTX_PREEMPT_LEAST_PROGRESS (1 / 0) override either way.
    ctx_only_preemption: bool = False
    # Generation-only requests a disaggregated generation worker admits beyond max_batch_size so their KV lands while
    # the batch decodes (disagg_gen_prefetch.py; where it applies). Read by the engine before any pool is sized;
    # TRTLLM_DISAGG_GEN_PREFETCH_REQUESTS overrides it.
    disagg_gen_prefetch_requests: int = 0
    # A disaggregated generation worker's KV cache manager V2 with no tier below GPU reserves each admitted
    # generation-only request's completion headroom (and frees a failed admission's matched pages) without
    # kv_cache_config.enable_disagg_completion_reservation, which keeps working (it also arms the connector prefill
    # reservation). Read by the KV cache creator; TRTLLM_DISAGG_GEN_COMPLETION_RESERVATION=1 / 0 overrides it either
    # way.
    disagg_gen_completion_reservation: bool = False
    # Pin the executor thread to this many dedicated physical cores on its GPU's NUMA node (host_stall.py; 0: no pin).
    # Read at executor loop entry; TRTLLM_EXECUTOR_THREAD_PIN_CORES overrides it either way.
    executor_thread_pin_cores: int = 0
    # Write KV to peers whose pools export fabric handles with batched copy-engine copies instead of one transfer
    # agent operation per fragment (disaggregation/native/fabric_writer.py). Applied before the KV cache transceiver
    # is created, so the context and generation workers of the model agree; TRTLLM_KV_FABRIC_WRITE overrides it.
    kv_fabric_write: bool = False
    # The transfer agent's bounce buffer scatters received KV on the execution stream, between forwards, and its
    # private copy streams take the default priority (agent_bounce_params scatter_on_order_stream=1,
    # copy_stream_priority=default). Applied before the transfer agent is created; agent_bounce_params and the
    # TRTLLM_NIXL_BOUNCE_* variables override it.
    bounce_scatter_on_order_stream: bool = False
    # Prefill CUDA graph token buckets the model adds to the configured capture list (breakable backend only; each
    # one at most the engine's capturable ceiling). They extend a configured list and never start prefill graphs on
    # their own: an empty configured list stays empty. Applied once by the engine (apply_prefill_capture_hooks).
    prefill_capture_extra_num_tokens: Tuple[int, ...] = ()
    # The extra buckets join only a configured list whose top bucket is at least this. A shorter list would otherwise
    # pad every step between its top bucket and the first extra one up to that extra bucket; such a list keeps its
    # own buckets and the engine logs a warning.
    prefill_capture_extra_min_top: int = 0
    # Cap on the final breakable prefill capture list (None: no cap).
    prefill_capture_max_num_tokens: Optional[int] = None
    # Called with the model after the engine releases that model's breakable prefill graphs
    # (DecoderRunner._release_decoder_graphs), so state the model instance keeps per captured graph (the addresses a
    # captured producer baked) ends with the graphs. The engine releases them, among other times, between the
    # KV-estimation executor and the final one, which recaptures on a new KV pool and new metadata.
    prefill_graphs_released: Optional[Callable[[Any], None]] = None

    @property
    def skips_unchanged_metadata_uploads(self) -> bool:
        """Whether a hook skips uploads into graph metadata buffers that already hold the values.

        Such a skip is sound only while no other engine writes those buffers, so the engine
        then gives its CUDA graph metadata a buffer pool of its own.
        """
        return (
            self.skip_unchanged_h2d
            or self.msa.skip_unchanged_block_offsets
            or self.skip_unchanged_draft_block_offsets
        )


DEFAULT_RUNTIME_HOOKS = RuntimeHooks()


def override_sparse_attention_class(stock: type, override: Optional[type], what: str) -> type:
    """``override`` when it is set and derives from ``stock``, else ``stock``; raises for an unrelated class."""
    if override is None or override is stock:
        return stock
    if not (isinstance(override, type) and issubclass(override, stock)):
        raise TypeError(
            f"RuntimeHooks.sparse_attention.{what} must derive from {stock.__name__}, got {override!r}"
        )
    return override


def runtime_hooks_of(model: object) -> RuntimeHooks:
    """The hooks ``model`` declares, or the all-off default when it declares none.

    Declaring hooks is optional for every model class, so absence is the common
    case and means the default.
    """
    hooks = getattr(model, "runtime_hooks", None)
    if hooks is None:
        return DEFAULT_RUNTIME_HOOKS
    if not isinstance(hooks, RuntimeHooks):
        raise TypeError(
            f"{type(model).__name__}.runtime_hooks must be a RuntimeHooks, got {type(hooks).__name__}"
        )
    return hooks


def apply_prefill_capture_hooks(
    kept: Sequence[int], hooks: RuntimeHooks, *, max_num_tokens: int, breakable: bool
) -> Tuple[List[int], List[int], List[int]]:
    """The prefill capture list after a model's hooks: ``(final, added, dropped)``.

    ``kept`` is the engine's filtered list. On the breakable backend (any other backend keeps ``kept``),
    ``prefill_capture_extra_num_tokens`` entries in (0, ``max_num_tokens``] join a non-empty ``kept`` whose top bucket
    is at least ``prefill_capture_extra_min_top`` (``added`` lists the new ones; a shorter list gets none and a
    warning); then every entry above ``prefill_capture_max_num_tokens`` is removed (``dropped``). ``final`` is sorted
    ascending and deduplicated.
    """
    final = sorted({int(n) for n in kept})
    added: List[int] = []
    if not breakable:
        return final, added, []
    if (
        final
        and hooks.prefill_capture_extra_num_tokens
        and final[-1] < hooks.prefill_capture_extra_min_top
    ):
        from tensorrt_llm.logger import logger

        logger.warning(
            f"Prefill CUDA graph capture list tops out at {final[-1]} (< {hooks.prefill_capture_extra_min_top}): "
            f"the model's extra buckets {list(hooks.prefill_capture_extra_num_tokens)} are not added, since every "
            f"step between {final[-1]} and the first of them would pad up to it. List the buckets up to "
            f"{hooks.prefill_capture_extra_min_top} to take them."
        )
    elif final:
        extra = {
            int(n) for n in hooks.prefill_capture_extra_num_tokens if 0 < int(n) <= max_num_tokens
        }
        added = sorted(extra - set(final))
        final = sorted(set(final) | extra)
    cap = hooks.prefill_capture_max_num_tokens
    dropped: List[int] = []
    if cap is not None:
        dropped = [n for n in final if n > cap]
        final = [n for n in final if n <= cap]
        added = [n for n in added if n <= cap]
    return final, added, dropped
