# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts shared by model runners and the engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionBackend,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.sampler.sampler import SampleStateTensors


@dataclass(frozen=True)
class PreparedInputs:
    """Wrap, rather than replace, the keyword arguments passed to a model."""

    kwargs: dict[str, Any]
    gather_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class PackedInputs:
    """Packed encode-only inputs and per-call output options.

    The token lists are referenced, not copied, so callers must not mutate them
    while a forward pass is in flight. ``model_inputs`` carries model-specific
    inputs for the batch, not scheduling or lifecycle controls; supported keys
    depend on the runner.
    """

    input_ids: list[int]
    sequence_lengths: list[int]
    multi_item_part_lens: list[list[int]] | None = None
    model_inputs: dict[str, Any] = field(default_factory=dict)
    gather_context_logits: bool = False


@dataclass(frozen=True)
class ScheduledInputs:
    """Inputs for one scheduled forward; batch and tensors are borrowed.

    Keep borrowed data valid until GPU consumption completes. Spec enablement
    is independent of draft length; updates use outputs["runtime_draft_len"].
    Omitting that key preserves the input length without modifying this record.
    """

    batch: ScheduledRequests
    new_tensors_device: SampleStateTensors | None = None
    cache_indirection_buffer: torch.Tensor | None = None
    num_accepted_tokens_device: torch.Tensor | None = None
    previous_request_slots: dict[int, int] | None = None
    gather_context_logits: bool = False
    enable_spec_decode: bool = False
    runtime_draft_len: int = 0


@dataclass(frozen=True)
class RunnerConfig:
    """Immutable settings shared by encoder, decoder, and no-KV-cache runners."""

    max_batch_size: int
    max_num_tokens: int
    max_seq_len: int
    max_beam_width: int
    without_logits: bool
    attention_backend: type[AttentionBackend]
    attention_runtime_features: AttentionRuntimeFeatures


class ModelRunner(ABC):
    """Shared lifecycle for scheduled and packed model execution."""

    def release_graphs(self) -> None:
        """Release captured graphs before their referenced resources are replaced.

        The caller must stop execution and synchronize outstanding work first.
        This keeps the model and compiled/autotuned state available for another
        warmup with new resources. Runners without graphs need no action.
        """

    def wait_for_input_copy(self) -> None:
        """Wait before host input is reused, if the runner owns async copies."""


class ScheduledModelRunner(ModelRunner):
    """Execute scheduled requests with explicit runtime dependencies."""

    def warmup(self, resource_manager: ResourceManager) -> None:
        """Prepare execution, including optional graph capture, when needed."""

    @abstractmethod
    def forward(
        self,
        inputs: ScheduledInputs,
        *,
        resource_manager: ResourceManager,
        is_dummy: bool = False,
    ) -> dict[str, Any]:
        """Return model outputs with an optional ``runtime_draft_len`` update.

        ``is_dummy`` marks warmup and memory-profiling passes.
        """


class PackedModelRunner(ModelRunner):
    """Run a model family whose batch arrives already packed.

    A runner implements this contract or ``ScheduledModelRunner``; the engine
    resolves which one it holds at initialization. The packed batch carries its
    own request boundaries and model-specific inputs, so none of the scheduling
    collaborators apply.
    """

    def warmup(self) -> None:
        """Prepare execution, including optional graph capture, when needed."""

    @abstractmethod
    def forward(
        self,
        inputs: PackedInputs,
    ) -> dict[str, Any]:
        """Validate and execute the packed inputs without silently ignoring them."""
