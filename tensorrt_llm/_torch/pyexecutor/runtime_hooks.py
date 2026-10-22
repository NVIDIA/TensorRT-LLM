# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-loop behaviour a model may opt into, read through ``runtime_hooks_of``.

A model declares its hooks as ``runtime_hooks`` (an instance attribute, a class
attribute or a property). Every field defaults to off, and a model that declares
nothing gets ``RuntimeHooks()``: the step loop, the samplers and the attention
metadata then run exactly as they do without this module. Every engine call site
of a hook runs its original statements when the hook is off.

The hooks change *where* and *how* the step loop moves data, never what a
forward computes. The protocols below state only the methods an engine call
site needs; the objects that implement them are owned by the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Protocol, Sequence

if TYPE_CHECKING:
    import torch


class StepInputStager(Protocol):
    """Records a decode step's input preparation and issues it as one launch.

    ``begin_if_eligible`` opens a deferred-write window on a step it supports
    and returns the stager, or returns None and the engine prepares the step
    natively. While the window is open, the engine's call sites route their
    device writes to the staging methods instead of issuing them. ``end``
    issues every staged write in staging order, then the deferred native
    operations; ``discard`` drops them after a failed preparation.

    While its window is open the stager sets itself as ``step_input_stager`` on
    the decoder's input preparer, the attention metadata, the speculative
    metadata and the KV cache managers, and resets that attribute to None at
    ``end`` and ``discard``. Every call site reads the attribute and runs its
    original statement when it is None.
    """

    def begin_if_eligible(
        self,
        preparer: Any,
        requests: Any,
        attn_metadata: Any,
        spec_metadata: Any,
        new_tensors_device: Any,
        *,
        graph_replay: bool,
        static_inputs: Optional[Dict[str, "torch.Tensor"]],
        is_dummy: bool,
        enable_spec_decode: bool,
        promoted: bool,
    ) -> Optional["StepInputStager"]:
        """Open the window for this step, or return None to keep the native preparation.

        The engine calls this only for the replay of an already captured CUDA
        graph, so ``graph_replay`` is always True: the window closes right
        before the replay, and an eager step, an initial capture or a warmup
        would run the model before the staged writes are issued. Those steps
        always prepare natively.

        ``preparer`` is the ``DecoderInputPreparer`` whose writes the window
        records. ``static_inputs`` are the static input tensors of the CUDA
        graph the step replays.
        """

    def copy(
        self, dst: "torch.Tensor", src: "torch.Tensor", clamp_min: Optional[int] = None
    ) -> None:
        """Stage ``dst.copy_(src, non_blocking=True)`` for a host ``src``."""

    def zero(self, dst: "torch.Tensor") -> None:
        """Stage ``dst.zero_()``."""

    def clamp_min(self, dst: "torch.Tensor", lo: int) -> None:
        """Stage ``dst.clamp_(min=lo)``."""

    def block_offsets(
        self,
        host_table: "torch.Tensor",
        dst: "torch.Tensor",
        copy_index: Sequence[int],
        index_scales: Sequence[int],
        kv_offsets: Sequence[int],
    ) -> None:
        """Stage the KV cache manager's batch copy of block offsets to the device."""

    def position_ids(self, dst: "torch.Tensor", host_position_ids: "torch.Tensor") -> None:
        """Stage the position-id upload and the CUDA graph's static copy of it."""

    def overlap_inputs(
        self,
        buffers: Any,
        previous_batch_indices: Sequence[int],
        previous_pos_indices: Sequence[int],
        num_tokens: int,
        num_draft_tokens: int,
        tokens_per_gen_step: int,
        num_extend_requests: int,
        new_tokens: "torch.Tensor",
        next_draft_tokens: "torch.Tensor",
        new_tokens_lens: "torch.Tensor",
    ) -> None:
        """Stage the overlap scheduler's gathers of the previous step's outputs into ``buffers``."""

    def scheduler_metadata(self, metadata: Any) -> bool:
        """Stage the sparse-attention indexer's scheduler metadata.

        Returns False, having staged nothing, when the caller must defer the
        native computation instead.
        """

    def defer(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` (native device work that reads staged values) after the launch."""

    def flush(self) -> None:
        """Issue every write staged so far, before the window closes."""

    def end(self, record_event: bool = True) -> None:
        """Close the window and issue the staged preparation."""

    def discard(self) -> None:
        """Close the window after a failed preparation; the staged writes are dropped."""

    def set_completion_event(self, event: "torch.cuda.Event") -> None:
        """Event the caller recorded right after ``end(record_event=False)``."""

    def static_inputs_tokens(self) -> int:
        """Tokens of the CUDA graph's static inputs the last launch wrote (0: none)."""


class SamplerStoreWriter(Protocol):
    """Stores a speculative sampler step's outputs and copies the store to the host in one launch."""

    def supported(
        self,
        store: Any,
        new_tokens: "torch.Tensor",
        next_new_tokens: "torch.Tensor",
        new_tokens_lens: "torch.Tensor",
        next_draft_tokens: "torch.Tensor",
    ) -> bool:
        """Whether this step's output tensors fit the writer; False keeps the native stores."""

    def store_and_copy_out(
        self,
        store: Any,
        slots: Sequence[int],
        new_tokens: "torch.Tensor",
        next_new_tokens: "torch.Tensor",
        new_tokens_lens: "torch.Tensor",
        next_draft_tokens: "torch.Tensor",
        host_new_tokens: "torch.Tensor",
        host_new_tokens_lens: "torch.Tensor",
        host_next_draft_tokens: "torch.Tensor",
        record_event: bool = True,
    ) -> None:
        """Scatter the outputs into the store rows ``slots`` and copy the store to the host tensors."""

    def set_completion_event(self, event: "torch.cuda.Event") -> None:
        """Event the caller recorded right after ``store_and_copy_out(record_event=False)``."""


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

    # Run the overlap executor loop with the execution stream current, so the
    # loop's own stream work and the forward are ordered on one stream.
    loop_on_execution_stream: bool = False
    # Decode input preparation recorded and issued as one launch (see StepInputStager).
    step_input_stager: Optional[StepInputStager] = None
    # The speculative sampler's store update and host copy in one launch (see SamplerStoreWriter).
    sampler_store_writer: Optional[SamplerStoreWriter] = None
    # The decode step's input prologue in one launch (see DecodePrologue).
    decode_prologue: Optional[DecodePrologue] = None
    # Actions after warmup, before the overlap loop and at shutdown (see ExecutorEvents).
    executor_events: Optional[ExecutorEvents] = None


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
