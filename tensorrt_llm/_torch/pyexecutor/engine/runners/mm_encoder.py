# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model runner for multimodal-encoder-only serving."""

from typing import Any

import torch

from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    _join_embeddings,
    _own_encoder_auxiliary_features,
)
from tensorrt_llm._torch.nccl_window_tensor_scope import discard_nccl_window_tensor_outputs
from tensorrt_llm._torch.pyexecutor.llm_request import get_multimodal_embedding_lengths
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.inputs.multimodal import _has_mm_payload_keys

from .no_kv_cache import NoKVCacheRunner


class MultimodalEncoderRunner(NoKVCacheRunner):
    """Run an engine whose only job is multimodal encoding.

    This family is distinct from ``MultimodalItemScheduler``, which schedules vision items
    inside a decoder engine. Its graph phase is a no-op because
    ``MultimodalEncoderGraphRunner`` belongs to, and is owned by, the vision model layer.
    """

    @nvtx_range("_forward_step_mm_encoder_only")
    def _forward_step(
        self,
        inputs: dict[str, Any],
        scheduled_requests: ScheduledRequests,
        *,
        gather_ids: torch.Tensor | None = None,
        gather_context_logits: bool = False,
    ) -> dict[str, Any]:
        """Batch encoder views and preserve the existing encoder-result contract."""
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
            if _has_mm_payload_keys(request.py_multimodal_data)
            and get_multimodal_embedding_lengths(request) is not None
        ]
        if not selected:
            return {
                "mm_embeddings": [],
                "mm_embedding_lengths": [],
                "mm_embedding_request_indices": [],
                "logits": None,
            }
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
        # Project common encoder views onto the existing sampler and E/P
        # interface. Batching does not change the external handoff format.
        output: dict[str, Any] = {
            "mm_embeddings": [result.multimodal_data["multimodal_embedding"] for result in results],
            "mm_embedding_lengths": [list(result.get_item_layout()[1]) for result in results],
            "mm_embedding_request_indices": [index for index, _, _ in selected],
            "logits": None,
        }
        metadata = [
            result.multimodal_data.get("multimodal_embedding_metadata") for result in results
        ]
        if any(item is not None for item in metadata):
            output["mm_embedding_metadata"] = metadata
        position_ids = []
        position_deltas = []
        for (_, _, param), result in zip(selected, results, strict=True):
            mrope = result.multimodal_data.get(
                "mrope_config", param.multimodal_data.get("mrope_config", {})
            )
            if mrope.get("mrope_position_ids") is not None:
                position_ids.append(mrope["mrope_position_ids"])
            if mrope.get("mrope_position_deltas") is not None:
                position_deltas.append(mrope["mrope_position_deltas"])
        if len(position_ids) != len(position_deltas) or len(position_ids) not in (0, len(results)):
            raise ValueError("M-RoPE tensors must align with encoder results")
        if position_ids:
            output["mrope_position_ids"] = position_ids
            output["mrope_position_deltas"] = position_deltas
        return output
