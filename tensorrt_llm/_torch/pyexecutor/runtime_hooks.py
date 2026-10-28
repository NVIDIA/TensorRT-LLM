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
from typing import TYPE_CHECKING, Any, Optional, Protocol

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
    decode_prologue: Optional[DecodePrologue] = None
    executor_events: Optional[ExecutorEvents] = None
    # Skip host-to-device metadata copies whose values the destination already holds.
    skip_unchanged_h2d: bool = False
    # Run the overlap executor loop with the execution stream current, so the
    # loop's own stream work and the forward are ordered on one stream.
    loop_on_execution_stream: bool = False
    # After autotuning, keep the fastest profiled tactic per shape on every
    # tensor-parallel rank, so the ranks capture the same kernels.
    merge_tuned_tactics_across_ranks: bool = False
    # Skip the per-step Philox seed and offset uploads on all-greedy
    # one-model speculative batches (the argmax path never reads them).
    skip_greedy_rng_upload: bool = False
    # Speculative step-input staging folded into the forward (see SpecStepFold).
    spec_step_fold: Optional[SpecStepFold] = None
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
