# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts shared by model runners and the engine."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_manager import CudaGraphLoraManager
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend
from tensorrt_llm.mapping import Mapping

from ..lora import LoraParamBuilder


@dataclass(frozen=True)
class PreparedInputs:
    """Wrap, rather than replace, the keyword arguments passed to a model."""

    kwargs: dict[str, Any]
    gather_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class RunnerDeps:
    """Engine-owned collaborators and startup constants used by a runner."""

    dist: Distributed | None
    mapping: Mapping
    enable_attention_dp: bool
    max_num_tokens: int
    max_seq_len: int
    prefill_cuda_graph_backend: PrefillCudaGraphBackend
    prefill_cuda_graph_num_tokens: list[int]
    mm_encoder_cache_enabled: bool
    input_ids_cuda: torch.Tensor
    position_ids_cuda: torch.Tensor
    gather_ids_cuda: torch.Tensor | None
    draft_tokens_cuda: torch.Tensor | None
    lora: LoraParamBuilder
    forward_step: Callable[..., dict[str, Any]]


class ModelRunner(Protocol):
    """Run a model family through its four lifecycle phases."""

    def prepare_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        attn_metadata: AttentionMetadata,
        *,
        spec_metadata: SpecMetadata | None,
        resource_manager: ResourceManager,
        enable_spec_decode: bool,
        cuda_graph_lora_manager: CudaGraphLoraManager | None,
        runtime_draft_len: int,
    ) -> PreparedInputs: ...

    def warmup(self, resource_manager: ResourceManager) -> None: ...

    def capture_graphs(self, resource_manager: ResourceManager) -> None: ...

    def forward(
        self,
        scheduled_requests: ScheduledRequests,
        attn_metadata: AttentionMetadata,
        *,
        spec_metadata: SpecMetadata | None,
        resource_manager: ResourceManager,
        enable_spec_decode: bool,
        cuda_graph_lora_manager: CudaGraphLoraManager | None,
        runtime_draft_len: int,
        moe_load_balancer: Any,
        gather_context_logits: bool,
    ) -> dict[str, Any]: ...
