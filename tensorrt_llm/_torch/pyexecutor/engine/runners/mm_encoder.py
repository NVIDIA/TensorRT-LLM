# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model runner for multimodal-encoder-only serving."""

from typing import Any

import torch

from tensorrt_llm._torch.models.modeling_multimodal_mixin import _build_request_multimodal_input
from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    _join_embeddings,
    _own_encoder_auxiliary_features,
)
from tensorrt_llm._torch.nccl_window_tensor_scope import discard_nccl_window_tensor_outputs
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.inputs.multimodal import _has_mm_payload_keys
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor

from .no_kv_cache import NoKVCacheRunner


class MultimodalEncoderRunner(NoKVCacheRunner):
    """Run an engine whose only job is multimodal encoding.

    This family is distinct from ``MultimodalItemScheduler``, which schedules vision items
    inside a decoder engine. Its graph phase is a no-op because
    ``MultimodalEncoderGraphRunner`` belongs to, and is owned by, the vision model layer.
    """

    def __init__(
        self, *args: Any, input_processor: BaseMultimodalInputProcessor, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._input_processor = input_processor

    @nvtx_range("_forward_step_mm_encoder_only")
    def _forward_step(
        self,
        inputs: dict[str, Any],
        scheduled_requests: ScheduledRequests,
        *,
        gather_ids: torch.Tensor | None = None,
        gather_context_logits: bool = False,
    ) -> dict[str, Any]:
        """Return encoded Params views aligned with their scheduled requests."""
        params = inputs.get("multimodal_params", [])
        requests = [
            (index, request)
            for index, request in enumerate(scheduled_requests.context_requests)
            if request.py_multimodal_data is not None
        ]
        if len(requests) != len(params):
            raise ValueError("Encoder inputs must align with multimodal context requests")
        selected = [
            (index, request, param)
            for (index, request), param in zip(requests, params, strict=True)
            if _has_mm_payload_keys(param.multimodal_data)
        ]
        if not selected:
            return {"mm_encoder_outputs": [], "mm_embedding_request_indices": [], "logits": None}
        selected_params = [param for _, _, param in selected]
        with discard_nccl_window_tensor_outputs(inputs):
            results = self._model.encode_multimodal_inputs(selected_params)
            if len(results) != len(selected):
                raise ValueError("Encoder outputs must align with multimodal context requests")
            # One ownership copy before releasing temporary NCCL windows. Adjacent
            # result views recover their original physical batch without a concat.
            owned = _join_embeddings(
                [result.multimodal_data["multimodal_embedding"] for result in results]
            ).clone()
            lengths = [sum(result.get_item_layout()[1]) for result in results]
            for result, embedding in zip(results, torch.split(owned, lengths, dim=0), strict=True):
                result.multimodal_data["multimodal_embedding"] = embedding
                _own_encoder_auxiliary_features(result)
        for (_, request, _), result in zip(selected, results, strict=True):
            result.multimodal_input = _build_request_multimodal_input(request, True)
            if result.multimodal_input is not None:
                _, finalized = self._input_processor.finalize_multimodal_prompt(
                    {"prompt_token_ids": request.get_tokens(0)}, result
                )
                result.multimodal_data = finalized.multimodal_data
        return {
            "mm_encoder_outputs": results,
            "mm_embedding_request_indices": [index for index, _, _ in selected],
            "logits": None,
        }
