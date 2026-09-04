# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model runner for embedding, classification, and reward-scoring models."""

from typing import Any

from torch import nn

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancerIterContext
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_manager import CudaGraphLoraManager
from tensorrt_llm._torch.speculative import SpecMetadata

from ...resource_manager import ResourceManager
from ...scheduler import ScheduledRequests
from .interface import PreparedInputs, RunnerDeps
from .no_cache import prepare_no_cache_inputs


class PoolingRunner:
    """Run models whose outputs feed embedding, classification, or scoring pools.

    This is a model runner because the family diverges during input preparation. It is
    distinct from vLLM's output-stage ``PoolingRunner``, which corresponds to a sampler.
    """

    def __init__(self, model: nn.Module, deps: RunnerDeps) -> None:
        self._model = model
        self._deps = deps

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
    ) -> PreparedInputs:
        inputs, gather_ids = prepare_no_cache_inputs(
            scheduled_requests,
            attn_metadata,
            spec_metadata,
            resource_manager,
            model=self._model,
            dist=self._deps.dist,
            mapping=self._deps.mapping,
            enable_attention_dp=self._deps.enable_attention_dp,
            enable_spec_decode=enable_spec_decode,
            cuda_graph_lora_manager=cuda_graph_lora_manager,
            runtime_draft_len=runtime_draft_len,
            max_num_tokens=self._deps.max_num_tokens,
            max_seq_len=self._deps.max_seq_len,
            prefill_cuda_graph_backend=self._deps.prefill_cuda_graph_backend,
            prefill_cuda_graph_num_tokens=self._deps.prefill_cuda_graph_num_tokens,
            mm_encoder_cache_enabled=self._deps.mm_encoder_cache_enabled,
            input_ids_cuda=self._deps.input_ids_cuda,
            position_ids_cuda=self._deps.position_ids_cuda,
            gather_ids_cuda=self._deps.gather_ids_cuda,
            draft_tokens_cuda=self._deps.draft_tokens_cuda,
            lora=self._deps.lora,
        )
        return PreparedInputs(inputs, gather_ids)

    def warmup(self, resource_manager: ResourceManager) -> None:
        return

    def capture_graphs(self, resource_manager: ResourceManager) -> None:
        return

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
    ) -> dict[str, Any]:
        prepared = self.prepare_inputs(
            scheduled_requests,
            attn_metadata,
            spec_metadata=spec_metadata,
            resource_manager=resource_manager,
            enable_spec_decode=enable_spec_decode,
            cuda_graph_lora_manager=cuda_graph_lora_manager,
            runtime_draft_len=runtime_draft_len,
        )
        with MoeLoadBalancerIterContext(moe_load_balancer):
            return self._deps.forward_step(
                prepared.kwargs,
                gather_ids=prepared.gather_ids,
                gather_context_logits=gather_context_logits,
            )
