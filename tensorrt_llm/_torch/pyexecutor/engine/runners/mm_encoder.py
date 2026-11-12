# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model runner for multimodal-encoder-only serving."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from typing_extensions import Self

from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    _build_request_multimodal_input,
    request_device_paths,
)
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    MoeLoadBalancerIterContext,
)
from tensorrt_llm._torch.nccl_window_tensor_scope import discard_nccl_window_tensor_outputs
from tensorrt_llm._torch.pyexecutor.llm_request import get_multimodal_embedding_lengths
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.utils import with_model_extra_attrs
from tensorrt_llm._utils import nvtx_range, prefer_pinned
from tensorrt_llm.inputs.multimodal import MultimodalParams, _has_mm_payload_keys

from ..multimodal import mm_encoder_cache_enabled
from .interface import ScheduledInputs, ScheduledModelRunner


@dataclass(frozen=True)
class _PayloadBatch:
    """Multimodal payloads and their positions in the scheduled context batch."""

    request_indices: list[int]
    params: list[MultimodalParams]
    embedding_lengths: list[list[int]]


class MultimodalEncoderRunner(ScheduledModelRunner):
    """Run an engine whose only job is multimodal encoding.

    This family is distinct from ``MultimodalItemScheduler``, which schedules vision items
    inside a decoder engine. Its graph phase is a no-op because
    ``MultimodalEncoderGraphRunner`` belongs to, and is owned by, the vision model layer.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        encoder_cache_enabled: bool,
        moe_load_balancer: MoeLoadBalancer | None,
    ) -> None:
        self._model = model
        self._encoder_cache_enabled = encoder_cache_enabled
        self._moe_load_balancer = moe_load_balancer
        # Raw media of default-builder encoders goes to the GPU in the encoder call; MRoPE
        # still moves per request, for the CUDA IPC handoff. None moves every tensor.
        self._device_paths = request_device_paths(model)

    @classmethod
    def create(cls, model: nn.Module, *, moe_load_balancer: MoeLoadBalancer | None = None) -> Self:
        return cls(
            model,
            encoder_cache_enabled=mm_encoder_cache_enabled(model),
            moe_load_balancer=moe_load_balancer,
        )

    def _prepare_inputs(self, scheduled_requests: ScheduledRequests) -> _PayloadBatch:
        request_indices = []
        params = []
        embedding_lengths = []
        start_offset = 0
        for request_idx, request in enumerate(scheduled_requests.context_requests):
            request_start = start_offset
            start_offset += request.get_num_tokens(0)
            multimodal_data = request.py_multimodal_data
            if multimodal_data is None or not _has_mm_payload_keys(multimodal_data):
                continue
            lengths = get_multimodal_embedding_lengths(request)
            if lengths is None:
                continue
            param = MultimodalParams(
                multimodal_input=_build_request_multimodal_input(
                    request, self._encoder_cache_enabled
                ),
                multimodal_data=request.py_multimodal_params.multimodal_data,
                input_ids_start_offset=request_start,
            )
            param.to_device(
                "multimodal_data",
                "cuda",
                pin_memory=prefer_pinned(),
                target_keywords=self._device_paths,
            )
            request_indices.append(request_idx)
            params.append(param)
            embedding_lengths.append(list(lengths))
        return _PayloadBatch(request_indices, params, embedding_lengths)

    def _validate_resources(self, resource_manager: ResourceManager) -> None:
        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        assert kv_cache_manager is None, (
            "a no-KV-cache runner was initialized, but a KV cache manager was allocated"
        )

    def warmup(self, resource_manager: ResourceManager) -> None:
        """Validate the resources used by this runner before execution."""
        self._validate_resources(resource_manager)

    @torch.inference_mode()
    @with_model_extra_attrs(lambda self: self._model.extra_attrs)
    def forward(
        self,
        inputs: ScheduledInputs,
        *,
        resource_manager: ResourceManager,
        is_dummy: bool = False,
    ) -> dict[str, Any]:
        del is_dummy
        self._validate_resources(resource_manager)
        batch = self._prepare_inputs(inputs.batch)
        with MoeLoadBalancerIterContext(self._moe_load_balancer):
            return self._forward_step(batch)

    @nvtx_range("_forward_step_mm_encoder_only")
    def _forward_step(self, batch: _PayloadBatch) -> dict[str, Any]:
        """Forward step for multimodal encoder only mode - returns mm_embeddings instead of logits."""  # noqa: E501
        if not batch.params:
            return {
                "mm_embeddings": [],
                "mm_embedding_request_indices": [],
                "mm_embedding_lengths": [],
            }
        # For mm_encoder_only mode, we only run the vision encoder part.
        # The model should be a vision encoder (e.g., Qwen2VisionModelBase).
        # This call bypasses ModelCaller, so scope the TP all-reduce windows here.
        # The embeddings outlive this step, so keep a clone rather than a window.
        with discard_nccl_window_tensor_outputs(batch.params):
            mm_embeddings = self._model.forward(batch.params)
            assert len(mm_embeddings) == 1, (
                "mm_embeddings should be a 1-element list, mix modality (video+image) is not "
                "supported"
            )
            mm_embedding_batch = mm_embeddings[0].clone()

        split_lengths = [sum(lengths) for lengths in batch.embedding_lengths]
        mm_embeddings = list(torch.split(mm_embedding_batch, split_lengths, dim=0))
        if len(mm_embeddings) != len(batch.embedding_lengths):
            raise ValueError(
                "mm_encoder_only produced an embedding batch that does not match "
                "mm_embedding_lengths"
            )

        # Extract mrope position data from multimodal_params if available.
        mrope_position_ids_list = []
        mrope_position_deltas_list = []
        for multimodal_param in batch.params:
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
        ) in (0, len(batch.params)), (
            f"mrope alignment: got {len(mrope_position_ids_list)} ids, "
            f"{len(mrope_position_deltas_list)} deltas, "
            f"{len(batch.params)} mm params"
        )

        result = {
            "mm_embeddings": mm_embeddings,
            "logits": None,
            "mm_embedding_request_indices": batch.request_indices,
            "mm_embedding_lengths": batch.embedding_lengths,
        }
        embedding_metadata = [
            param.multimodal_data.get("multimodal_embedding_metadata") for param in batch.params
        ]
        if any(metadata is not None for metadata in embedding_metadata):
            result["mm_embedding_metadata"] = embedding_metadata
        if mrope_position_ids_list:
            result["mrope_position_ids"] = mrope_position_ids_list
        if mrope_position_deltas_list:
            result["mrope_position_deltas"] = mrope_position_deltas_list

        return result
