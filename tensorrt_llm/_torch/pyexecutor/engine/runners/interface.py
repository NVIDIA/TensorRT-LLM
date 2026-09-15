# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts shared by model runners and the engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionBackend,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_manager import CudaGraphLoraManager
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.mapping import Mapping

from ..lora import LoraParamBuilder

if TYPE_CHECKING:
    from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancer


@dataclass(frozen=True)
class PreparedInputs:
    """Wrap, rather than replace, the keyword arguments passed to a model."""

    kwargs: dict[str, Any]
    gather_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class PackedEncoderBatch:
    """An encode-only batch whose tokens the caller already packed.

    The token lists are referenced, not copied, so callers must not mutate them
    while a forward pass is in flight. ``model_inputs`` carries model-specific
    inputs for the batch, not scheduling or lifecycle controls; supported keys
    depend on the runner.
    """

    input_ids: list[int]
    sequence_lengths: list[int]
    multi_item_part_lens: list[list[int]] | None = None
    model_inputs: dict[str, Any] = field(default_factory=dict)


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


@dataclass(frozen=True)
class RunnerDeps:
    """Engine-owned runtime collaborators shared by model runners."""

    dist: Distributed | None
    mapping: Mapping
    input_ids_cuda: torch.Tensor
    position_ids_cuda: torch.Tensor
    gather_ids_cuda: torch.Tensor | None
    draft_tokens_cuda: torch.Tensor | None
    cache_indirection: torch.Tensor | None
    lora: LoraParamBuilder
    moe_load_balancer: MoeLoadBalancer | None
    model_forward: Callable[..., Any]


class ModelRunner(Protocol):
    """Run a model family whose batch arrives as scheduled requests."""

    def warmup(self, resource_manager: ResourceManager) -> None: ...

    def capture_graphs(self, resource_manager: ResourceManager) -> None: ...

    def prepare_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        *,
        resource_manager: ResourceManager,
        cuda_graph_lora_manager: CudaGraphLoraManager | None,
        runtime_draft_len: int,
    ) -> PreparedInputs:
        """Prepare the scheduled batch for execution.

        Graph compatibility is decided here, before execution.
        """
        ...

    def forward(
        self,
        scheduled_requests: ScheduledRequests,
        *,
        resource_manager: ResourceManager,
        cuda_graph_lora_manager: CudaGraphLoraManager | None,
        runtime_draft_len: int,
        gather_context_logits: bool,
    ) -> dict[str, Any]:
        """Pass the scheduled batch to ``prepare_inputs`` and execute its result."""
        ...

    def cleanup(self) -> None:
        """Release everything the runner owns; the engine drops it afterwards."""
        ...


class PackedModelRunner(Protocol):
    """Run a model family whose batch arrives already packed.

    A runner implements this contract or ``ModelRunner``, never both; the engine
    resolves which one it holds at initialization. The packed batch carries its
    own request boundaries and model-specific inputs, so none of the scheduling
    collaborators apply.
    """

    def warmup(self) -> None: ...

    def capture_graphs(self) -> None: ...

    def prepare_inputs(self, batch: PackedEncoderBatch) -> PreparedInputs:
        """Prepare the packed batch for execution.

        Implementations must validate the batch and its model-specific inputs, and
        reject unsupported keys or overrides of runner-managed fields. Graph
        compatibility is decided here, before execution; inputs must not be
        silently ignored.
        """
        ...

    def forward(
        self,
        batch: PackedEncoderBatch,
        *,
        gather_context_logits: bool = False,
    ) -> dict[str, Any]:
        """Pass the packed batch to ``prepare_inputs`` and execute its result."""
        ...

    def cleanup(self) -> None:
        """Release everything the runner owns; the engine drops it afterwards."""
        ...
