# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model runner for multimodal-encoder-only serving."""

from typing import Any

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import get_multimodal_embedding_lengths
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.inputs.multimodal import _has_mm_payload_keys

from .no_cache import NoCacheRunner


class MultimodalEncoderRunner(NoCacheRunner):
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
        """Forward step for multimodal encoder only mode - returns mm_embeddings instead of logits."""  # noqa: E501
        # Keep the old profiling label above so this movement is trace-compatible.
        multimodal_params = inputs.get("multimodal_params", [])
        if not multimodal_params or len(multimodal_params) == 0:
            return {
                "mm_embeddings": [],
                "mm_embedding_request_indices": [],
                "mm_embedding_lengths": [],
            }
        # Some ctx requests carry only mrope metadata (no actual vision
        # content). Skip them so the encoder only runs on real image payloads.
        mm_context_requests = [
            (request_idx, request)
            for request_idx, request in enumerate(scheduled_requests.context_requests)
            if request.py_multimodal_data is not None
        ]
        if len(mm_context_requests) != len(multimodal_params):
            raise ValueError(
                "mm_encoder_only expects one multimodal payload per context "
                "request carrying py_multimodal_data"
            )
        mm_request_indices_with_payload = []
        mm_params_with_payload = []
        mm_embedding_lengths = []
        for (request_idx, request), multimodal_param in zip(mm_context_requests, multimodal_params):
            if not _has_mm_payload_keys(request.py_multimodal_data):
                # mrope-only warmup request (no actual vision content) -> skip.
                continue
            multimodal_embedding_lengths = get_multimodal_embedding_lengths(request)
            if multimodal_embedding_lengths is None:
                # Vision payload keys present but no pre-computed embedding
                # lengths — skip to avoid a downstream sum(None) TypeError.
                continue
            mm_request_indices_with_payload.append(request_idx)
            mm_params_with_payload.append(multimodal_param)
            mm_embedding_lengths.append(multimodal_embedding_lengths)
        if not mm_params_with_payload:
            return {
                "mm_embeddings": [],
                "mm_embedding_request_indices": [],
                "mm_embedding_lengths": [],
            }
        # For mm_encoder_only mode, we only run the vision encoder part.
        # The model should be a vision encoder (e.g., Qwen2VisionModelBase).
        mm_embeddings = self._model.forward(mm_params_with_payload)
        assert len(mm_embeddings) == 1, (
            "mm_embeddings should be a 1-element list, mix modality (video+image) is not supported"
        )

        split_lengths = [sum(lengths) for lengths in mm_embedding_lengths]
        mm_embeddings = list(torch.split(mm_embeddings[0], split_lengths, dim=0))
        if len(mm_embeddings) != len(mm_embedding_lengths):
            raise ValueError(
                "mm_encoder_only produced an embedding batch that does not match "
                "mm_embedding_lengths"
            )

        # Extract mrope position data from multimodal_params if available.
        mrope_position_ids_list = []
        mrope_position_deltas_list = []
        for multimodal_param in mm_params_with_payload:
            mrope_config = multimodal_param.multimodal_data.get("mrope_config", {})
            mrope_position_ids = mrope_config.get("mrope_position_ids")
            mrope_position_deltas = mrope_config.get("mrope_position_deltas")
            if mrope_position_ids is not None:
                mrope_position_ids_list.append(mrope_position_ids)
            if mrope_position_deltas is not None:
                mrope_position_deltas_list.append(mrope_position_deltas)

        # mrope lists must align 1:1 with multimodal_params (or be empty);
        # the sampler indexes them by per-MM-result position into mm_embeddings.
        assert len(mrope_position_ids_list) == len(mrope_position_deltas_list) and len(
            mrope_position_ids_list
        ) in (0, len(mm_params_with_payload)), (
            f"mrope alignment: got {len(mrope_position_ids_list)} ids, "
            f"{len(mrope_position_deltas_list)} deltas, "
            f"{len(mm_params_with_payload)} mm params"
        )

        result = {
            "mm_embeddings": mm_embeddings,
            "logits": None,
            "mm_embedding_request_indices": mm_request_indices_with_payload,
            "mm_embedding_lengths": mm_embedding_lengths,
        }
        if mrope_position_ids_list:
            result["mrope_position_ids"] = mrope_position_ids_list
        if mrope_position_deltas_list:
            result["mrope_position_deltas"] = mrope_position_deltas_list

        return result
