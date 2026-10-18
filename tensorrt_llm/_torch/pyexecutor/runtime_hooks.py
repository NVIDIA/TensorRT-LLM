# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Step-loop behaviour a model may opt into, read through ``runtime_hooks_of``.

A model declares its hooks as ``runtime_hooks`` (an instance attribute, a class
attribute or a property). Every field defaults to off, and a model that declares
nothing gets ``DEFAULT_RUNTIME_HOOKS``: the step loop, the samplers and the
attention metadata then run exactly as they do without this module. Every
engine call site of a hook runs its original statement when the hook is off.

The hooks change *where* and *how* the step loop moves data, never what a
forward computes. The protocols below state only the methods an engine call
site needs; the objects that implement them are owned by the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence

if TYPE_CHECKING:
    import torch


class StepInputStagerSource(Protocol):
    """Opens a step-input staging window on the decode steps it supports."""

    def begin_if_eligible(
        self,
        runner: Any,
        requests: Any,
        attn_metadata: Any,
        spec_metadata: Any,
        new_tensors_device: Any,
        *,
        graph_replay: bool,
        is_dummy: bool,
        enable_spec_decode: bool,
        promoted: bool,
    ) -> Optional["StepInputStager"]:
        """Open the window for this step and return its stager, or None to keep the native
        preparation."""


class StepInputStager(Protocol):
    """One decode step's input preparation, recorded and issued as one launch.

    While its window is open, the stager sets itself as ``step_input_stager`` on
    the runner, the attention metadata and the speculative metadata it was
    opened on, and resets that attribute to None at ``end`` and ``discard``.
    Every call site reads the attribute and runs its original statement when it
    is None.

    ``copy``, ``overlap_inputs`` and ``block_offsets`` return True when the
    stager recorded the write, and False when it declines it; on False the call
    site runs its original statement. A stager that declines a write first
    issues every write it recorded, then declines every later write of the
    window, so the rest of the preparation runs natively and in its original
    order. ``end`` issues every recorded write; ``discard`` drops them after a
    failed preparation.
    """

    def copy(self, dst: "torch.Tensor", src: "torch.Tensor") -> bool:
        """Record ``dst.copy_(src, non_blocking=True)`` for a host ``src``."""

    def overlap_inputs(
        self,
        runner: Any,
        *,
        new_tokens_device: "torch.Tensor",
        new_tokens_lens_device: "torch.Tensor",
        next_draft_tokens_device: "torch.Tensor",
        previous_batch_indices: Sequence[int],
        previous_pos_indices: Sequence[int],
        num_tokens: int,
        num_draft_tokens: int,
        tokens_per_gen_step: int,
        num_extend_requests: int,
    ) -> bool:
        """Record the overlap scheduler's writes of the previous step's outputs into the inputs."""

    def block_offsets(
        self,
        manager: Any,
        dst: "torch.Tensor",
        request_ids: Sequence[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
    ) -> bool:
        """Record ``manager.copy_batch_block_offsets(dst, ...)``."""

    def end(self, inputs: dict) -> None:
        """Close the window and issue the recorded writes."""

    def discard(self) -> None:
        """Close the window after a failed preparation; the recorded writes are dropped."""

    def static_inputs(self) -> Optional[tuple]:
        """``(input_ids ptr, position_ids ptr, tokens)`` the last ``end`` also wrote into the
        CUDA graph's static inputs, or None when the replay must copy them."""


class SamplerStoreWriter(Protocol):
    """Stores a speculative sampler step's outputs and copies the store to the host in one launch."""

    def store_and_copy_out(
        self,
        sampler: Any,
        slots: Sequence[int],
        new_tokens: "torch.Tensor",
        next_new_tokens: "torch.Tensor",
        new_tokens_lens: "torch.Tensor",
        next_draft_tokens: "torch.Tensor",
        verify_lens: Optional["torch.Tensor"],
    ) -> Optional[tuple]:
        """Scatter the outputs into the store rows ``slots`` and copy the store to the host.

        Returns the host copies of the store's ``new_tokens``, ``new_tokens_lens``
        and ``next_draft_tokens``, or None, having written nothing, when the
        sampler must store and copy them itself.
        """


@dataclass(frozen=True)
class RuntimeHooks:
    """The step-loop hooks a model opts into. All off by default."""

    # Decode input preparation recorded and issued as one launch per step: opens the
    # step's StepInputStager window (see StepInputStagerSource).
    step_input_stager: Optional[StepInputStagerSource] = None
    # The speculative sampler's store update and host copy in one launch (see SamplerStoreWriter).
    sampler_store_writer: Optional[SamplerStoreWriter] = None


DEFAULT_RUNTIME_HOOKS = RuntimeHooks()


def runtime_hooks_of(model: object) -> RuntimeHooks:
    """The hooks ``model`` declares, or the all-off default when it declares none.

    Declaring hooks is optional for every model class, so absence is the common
    case and means the default.

    Raises:
        TypeError: if ``model.runtime_hooks`` is neither None nor a ``RuntimeHooks``.
    """
    hooks = getattr(model, "runtime_hooks", None)
    if hooks is None:
        return DEFAULT_RUNTIME_HOOKS
    if not isinstance(hooks, RuntimeHooks):
        raise TypeError(
            f"{type(model).__name__}.runtime_hooks must be a RuntimeHooks, got {type(hooks).__name__}"
        )
    return hooks
