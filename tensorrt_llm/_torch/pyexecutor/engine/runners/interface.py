# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts shared by model runners and the engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
    model_forward: Callable[..., Any]


class ModelRunner(Protocol):
    """Run a model family through its four lifecycle phases."""

    def prepare_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        *,
        resource_manager: ResourceManager,
        cuda_graph_lora_manager: CudaGraphLoraManager | None,
        runtime_draft_len: int,
    ) -> PreparedInputs: ...

    def warmup(self, resource_manager: ResourceManager) -> None: ...

    def capture_graphs(self, resource_manager: ResourceManager) -> None: ...

    def forward(
        self,
        scheduled_requests: ScheduledRequests,
        *,
        resource_manager: ResourceManager,
        cuda_graph_lora_manager: CudaGraphLoraManager | None,
        runtime_draft_len: int,
        moe_load_balancer: MoeLoadBalancer | None,
        gather_context_logits: bool,
    ) -> dict[str, Any]: ...
