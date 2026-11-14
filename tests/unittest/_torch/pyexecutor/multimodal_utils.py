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
from tensorrt_llm._torch.tensor_lru_cache import TensorLRUCache
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
)
from tensorrt_llm.inputs.registry import BaseMultimodalDummyInputsBuilder


def bare_mm_item_scheduler(
    model: MultimodalModelMixin,
    input_processor: BaseMultimodalDummyInputsBuilder | None = None,
) -> MultimodalItemScheduler:
    """A scheduler with no budget resolution -- the engine only builds one when item
    scheduling is engaged, so these tests skip `create()` and exercise the item path."""
    if model._multimodal_encoder_cache is None:
        model._multimodal_encoder_cache = TensorLRUCache(1 << 20)
    return MultimodalItemScheduler(model=model, input_processor=input_processor)


def make_llm_request(
    request_id: int,
    multimodal_data: dict[str, Any] | None = None,
    *,
    input_tokens: list[int] | None = None,
    prompt_len: int = 3,
) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=input_tokens if input_tokens is not None else list(range(1, prompt_len + 1)),
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
    """Mark one item ready without materializing an output tensor."""
    del hidden, fill
    state.item_ready[item_idx] = True


def make_mm_request(
    request_id: int,
    costs: list[int],
    *,
    ready: Sequence[int] = (),
    interleaved: bool = False,
    embedding_lengths: Sequence[int] | None = None,
    prompt_len: int = 3,
) -> LlmRequest:
    """Make a request with one single-row item per cost.

    With `interleaved`, the prompt places one text token before each item and
    after the last one, so item `i` sits at prompt position `2 * i + 1`.
    """
    if embedding_lengths is None:
        embedding_lengths = [1] * len(costs)
    multimodal_data = {
        "image": {"pixel_values": torch.empty(len(costs), 1)},
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
            item_refs=[("image", item_idx) for item_idx in range(len(costs))],
            encoder_token_lengths=costs,
            output_embedding_lengths=list(embedding_lengths),
        ),
    }
    input_tokens = None
    if interleaved:
        embed_mask = [0]
        for length in embedding_lengths:
            embed_mask.extend([1] * length)
            embed_mask.append(0)
        multimodal_data["multimodal_embed_mask_cumsum"] = torch.tensor(embed_mask).cumsum(0)
        input_tokens = list(range(len(embed_mask)))
    request = make_llm_request(
        request_id, multimodal_data, input_tokens=input_tokens, prompt_len=prompt_len
    )
    initialize_multimodal_encoder_request(request, max_num_tokens=1 << 30)
    for item_idx in ready:
        record_output(request.py_mm_encoder_state, item_idx)
    return request
