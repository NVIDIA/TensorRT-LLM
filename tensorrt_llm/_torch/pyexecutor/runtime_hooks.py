# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-loop behaviour a model may opt into, read once by the engine.

A model declares its hooks as ``runtime_hooks`` (an instance attribute or a class
attribute). Every field defaults to off, and a model that declares nothing gets
``RuntimeHooks()``: the step loop, the samplers and the attention metadata then
run exactly as they do without this module.

The hooks change *where* and *how* the step loop moves data, never what a
forward computes. Each one is armed only by a model that has measured it on its
own deployment; none is a general default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Protocol

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
    # Fill the per-request block-id table by slice-copying each request's page-index view instead of
    # building a Python list of the whole padded row (max_seq_len / tokens_per_block entries) first.
    block_ids_from_views: bool = False


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
    # After autotuning, keep the fastest profiled tactic per shape on every
    # tensor-parallel rank, so the ranks capture the same kernels.
    merge_tuned_tactics_across_ranks: bool = False
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
    # uploads into the step-input pack (pack_step_inputs), so they ride its one
    # upload instead of each being a copy the next graph waits for.
    pack_spec_rng_uploads: bool = False
    # The one-model speculative step's small per-step uploads: the accepted-draft
    # counts join the step-input pack, and the spec sampler keeps its slot ids in
    # a device buffer it re-uploads only when they change.
    spec_small_h2d: bool = False
    # Speculative step-input staging folded into the forward (see SpecStepFold).
    spec_step_fold: Optional[SpecStepFold] = None
    # Copy the one-model speculative sampler's per-slot store to the host on the sampler's D2H side
    # stream, so those copies overlap the next step's input staging instead of preceding it.
    sampler_d2h_side_stream: bool = False
    # Time the iteration log's device step with one timing event per iteration boundary, recorded on a side
    # stream that waits for the current stream, instead of an end and a start event back to back on the
    # current stream (the same instant, with no GPU work between them). The reported step times are unchanged.
    iteration_timing_one_event: bool = False
    # MiniMax-M3 MSA metadata preparation.
    msa: MsaRuntimePolicy = MsaRuntimePolicy()

    @property
    def skips_unchanged_metadata_uploads(self) -> bool:
        """Whether a hook skips uploads into graph metadata buffers that already hold the values.

        Such a skip is sound only while no other engine writes those buffers, so the engine
        then gives its CUDA graph metadata a buffer pool of its own.
        """
        return self.skip_unchanged_h2d or self.msa.skip_unchanged_block_offsets


DEFAULT_RUNTIME_HOOKS = RuntimeHooks()


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
