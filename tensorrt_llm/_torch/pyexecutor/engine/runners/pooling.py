# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model runner for embedding, classification, and reward-scoring models."""

from typing import Any

import torch
from torch import nn

from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancer
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.mapping import Mapping

from ..model_call import ModelCaller
from .no_kv_cache import NoKVCacheRunner, NoKVCacheRunnerConfig


class PoolingRunner(NoKVCacheRunner):
    """Run models whose outputs feed embedding, classification, or scoring pools.

    This is a model runner because the family diverges during input preparation. It is
    distinct from vLLM's output-stage ``PoolingRunner``, which corresponds to a sampler.
    """

    def __init__(
        self,
        model: nn.Module,
        config: NoKVCacheRunnerConfig,
        *,
        mapping: Mapping,
        dist: Distributed | None,
        moe_load_balancer: MoeLoadBalancer | None,
        model_caller: ModelCaller,
    ) -> None:
        super().__init__(
            model, config, mapping=mapping, dist=dist, moe_load_balancer=moe_load_balancer
        )
        self._model_caller = model_caller

    @nvtx_range("_forward_step")
    def _forward_step(
        self,
        inputs: dict[str, Any],
        scheduled_requests: ScheduledRequests,
        *,
        gather_ids: torch.Tensor | None = None,
        gather_context_logits: bool = False,
    ) -> dict[str, Any]:
        attn_metadata = inputs.get("attn_metadata")
        if attn_metadata is not None:
            attn_metadata.on_update_kv_lens()
        if inputs.get("spec_metadata") is not None:
            gather_ids = inputs["spec_metadata"].gather_ids

        outputs = self._model_caller(
            **inputs,
            return_context_logits=gather_ids is not None or gather_context_logits,
        )
        if self._config.without_logits:
            return outputs
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
            if logits is None:
                return outputs
        else:
            logits = outputs
            outputs = {"logits": logits}
        if gather_ids is not None:
            outputs["logits"] = logits[gather_ids]
        return outputs
