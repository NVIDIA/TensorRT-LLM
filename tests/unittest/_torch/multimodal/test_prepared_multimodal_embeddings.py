# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared encoder-row preparation independently of text embedding fusion."""

from collections.abc import Sequence

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_multimodal_mixin as legacy_model
from tensorrt_llm._torch.models import modeling_multimodal_utils as legacy_embeddings
from tensorrt_llm._torch.multimodal import embeddings, model
from tensorrt_llm.inputs.multimodal import MultimodalParams, MultimodalRuntimeData

pytestmark = pytest.mark.cpu_only


class RowModel(model.MultimodalModelMixin):
    def __init__(self) -> None:
        self.model_config = ModelConfig()
        self.encode_calls = 0
        self.hooks: list[str] = []

    @property
    def text_embedding_layer(self) -> torch.nn.Module:
        raise AssertionError("Row preparation must not access text embeddings")

    def encode_multimodal_inputs(self, params: Sequence[MultimodalParams]) -> torch.Tensor:
        self.encode_calls += 1
        return torch.cat([param.multimodal_data["image"]["rows"] for param in params])

    def after_full_multimodal_embeddings(
        self, *, input_ids: torch.Tensor, embeddings: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.hooks.append("full")
        return input_ids + 1, embeddings + 10

    def after_active_multimodal_embeddings(
        self, *, active_embeddings: list[torch.Tensor], **kwargs
    ) -> tuple[list[torch.Tensor], tuple[torch.Tensor, ...]]:
        self.hooks.append("active")
        return [rows * 2 for rows in active_embeddings], tuple(
            rows + 100 for rows in active_embeddings
        )


def runtime(start: int, end: int) -> MultimodalRuntimeData:
    return MultimodalRuntimeData(
        past_seen_token_num=start,
        chunk_end_pos=end,
        embed_mask_cumsum=torch.arange(1, 7, dtype=torch.int64),
    )


def test_row_preparation_reuses_full_embeddings_and_runs_hooks_after_chunk_selection():
    instance = RowModel()
    rows = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    params = MultimodalParams(
        multimodal_data={"image": {"rows": rows}}, multimodal_runtime=runtime(0, 2)
    )
    input_ids = torch.tensor([7, 7])
    first = instance.prepare_multimodal_embeddings(
        input_ids=input_ids, multimodal_params=[params], num_context_requests=1
    )
    torch.testing.assert_close(first.input_ids, input_ids + 1)
    torch.testing.assert_close(first.multimodal_embeddings[0], (rows[:2] + 10) * 2)
    torch.testing.assert_close(first.extra_embeds[0], rows[:2] + 110)
    torch.testing.assert_close(params.multimodal_data["multimodal_embedding"], rows)

    params.multimodal_runtime = runtime(2, 5)
    second = instance.prepare_multimodal_embeddings(
        input_ids=torch.tensor([7, 7, 7]), multimodal_params=[params], num_context_requests=1
    )
    torch.testing.assert_close(second.multimodal_embeddings[0], (rows[2:5] + 10) * 2)
    assert instance.encode_calls == 1
    assert instance.hooks == ["full", "active", "full", "active"]


@pytest.mark.parametrize("num_contexts,start,end", [(0, 0, 2), (1, 6, 6)])
def test_inactive_rows_do_not_encode_or_run_hooks(num_contexts, start, end):
    instance = RowModel()
    params = MultimodalParams(
        multimodal_data={"image": {"rows": torch.zeros(6, 2)}},
        multimodal_runtime=runtime(start, end),
    )
    input_ids = torch.tensor([7])
    prepared = instance.prepare_multimodal_embeddings(
        input_ids=input_ids, multimodal_params=[params], num_context_requests=num_contexts
    )
    assert prepared.input_ids is input_ids
    assert not prepared.multimodal_embeddings and not prepared.extra_embeds
    assert instance.encode_calls == 0 and not instance.hooks


def test_already_chunk_local_embeddings_are_not_sliced_twice():
    instance = RowModel()
    rows = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    params = MultimodalParams(
        multimodal_data={"multimodal_embedding": rows, "multimodal_embedding_is_chunk": True},
        multimodal_runtime=runtime(2, 4),
    )
    prepared = instance.prepare_multimodal_embeddings(
        input_ids=torch.tensor([7, 7]), multimodal_params=[params], num_context_requests=1
    )
    torch.testing.assert_close(prepared.multimodal_embeddings[0], (rows + 10) * 2)
    assert instance.encode_calls == 0


def test_legacy_imports_preserve_runtime_identity():
    assert legacy_model.MultimodalModelMixin is model.MultimodalModelMixin
    assert legacy_model.MultimodalRuntimeMixin is model.MultimodalRuntimeMixin
    assert legacy_model.PreparedLlmInputs is model.PreparedLlmInputs
    assert legacy_model.EncoderGroup is model.EncoderGroup


@pytest.mark.parametrize(
    "name",
    [
        "_get_uncached_multimodal_params",
        "_store_chunked_prefill_embeddings",
        "_normalize_encoder_embeddings",
        "_join_embeddings",
        "get_multimodal_embeddings",
        "get_attached_multimodal_embeddings",
        "find_input_mm_embeds",
        "filter_mm_token_from_input_ids",
        "_as_copy_index",
        "fuse_input_embeds",
    ],
)
def test_legacy_embedding_exports_preserve_runtime_identity(name):
    assert getattr(legacy_embeddings, name) is getattr(embeddings, name)
