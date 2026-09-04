# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the multimodal-encoder tests.

Used by both `test_multimodal_scheduler.py` (scheduler / executor / request state)
and `engine/test_multimodal.py` (the MultimodalItemScheduler component).
"""

from collections.abc import Sequence
from typing import Any

import torch

from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.pyexecutor.engine.multimodal import MultimodalItemScheduler
from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    MultimodalEncoderRequestState,
    initialize_multimodal_encoder_request,
)
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.inputs.multimodal import MULTIMODAL_ENCODER_ITEM_METADATA_KEY
from tensorrt_llm.inputs.registry import (
    BaseMultimodalDummyInputsBuilder,
    MultimodalEncoderItemMetadata,
)


def bare_mm_item_scheduler(
    model: MultimodalModelMixin,
    input_processor: BaseMultimodalDummyInputsBuilder | None = None,
) -> MultimodalItemScheduler:
    """A scheduler with no budget resolution -- the engine only builds one when item
    scheduling is engaged, so these tests skip `create()` and exercise the item path."""
    return MultimodalItemScheduler(model=model, input_processor=input_processor)


def make_llm_request(request_id: int, multimodal_data: dict[str, Any] | None = None) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=[1, 2, 3],
        sampling_config=SamplingConfig(),
        is_streaming=False,
        py_multimodal_data=multimodal_data,
    )


def record_output(
    state: MultimodalEncoderRequestState,
    item_idx: int,
    *,
    hidden: int = 1,
    fill: float = 0.0,
) -> None:
    """Write one item the way the encoder step does, sized from its declaration."""
    state.record(
        item_idx,
        torch.full((state.embedding_lengths[item_idx], hidden), fill),
    )


def make_mm_request(request_id: int, costs: list[int], *, ready: Sequence[int] = ()) -> LlmRequest:
    request = make_llm_request(
        request_id,
        multimodal_data={
            "image": {"pixel_values": torch.empty(len(costs), 1)},
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", item_idx) for item_idx in range(len(costs))],
                encoder_token_lengths=costs,
                output_embedding_lengths=[1] * len(costs),
            ),
            "multimodal_embedding_lengths": [1] * len(costs),
        },
    )
    initialize_multimodal_encoder_request(request, max_num_tokens=1 << 30)
    for item_idx in ready:
        record_output(request.py_mm_encoder_state, item_idx)
    return request
