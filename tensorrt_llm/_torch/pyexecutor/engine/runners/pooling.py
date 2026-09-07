# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model runner for embedding, classification, and reward-scoring models."""

from typing import Any

import torch

from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import nvtx_range

from .no_cache import NoCacheRunner


class PoolingRunner(NoCacheRunner):
    """Run models whose outputs feed embedding, classification, or scoring pools.

    This is a model runner because the family diverges during input preparation. It is
    distinct from vLLM's output-stage ``PoolingRunner``, which corresponds to a sampler.
    """

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

        outputs = self._deps.model_forward(
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
