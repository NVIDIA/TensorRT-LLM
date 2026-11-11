# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The multimodal runtime base, its composite subclass, and the contract checks."""

from collections.abc import Sequence

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.multimodal.model import (
    MultimodalModelContractError,
    MultimodalModelMixin,
    MultimodalRuntimeMixin,
)
from tensorrt_llm.inputs.multimodal import MultimodalParams, MultimodalRuntimeData

pytestmark = pytest.mark.cpu_only

_COMPOSITE_MEMBERS = (
    "language_model",
    "runtime_hooks",
    "vocab_size_padded",
    "infer_max_seq_len",
    "set_guided_decoder",
    "forward",
    "get_language_model_forward_kwargs",
    "text_embedding_layer",
    "prepare_multimodal_inputs",
)


class RowRuntime(torch.nn.Module, MultimodalRuntimeMixin):
    """A model that runs its own decoder: rows come from `multimodal_data["image"]["rows"]`."""

    def __init__(self, hidden_size: int = 2) -> None:
        super().__init__()
        self.model_config = ModelConfig(pretrained_config=PretrainedConfig(hidden_size=hidden_size))
        self.hook_kwargs: list[dict] = []

    def encode_multimodal_inputs(self, params: Sequence[MultimodalParams]) -> torch.Tensor:
        return torch.cat([param.multimodal_data["image"]["rows"] for param in params])

    @property
    def embedding_dim(self) -> int:
        return self.model_config.pretrained_config.hidden_size

    @property
    def embedding_dtype(self) -> torch.dtype:
        return torch.float32

    def after_active_multimodal_embeddings(self, *, active_embeddings, **kwargs):
        self.hook_kwargs.append(kwargs)
        return active_embeddings, ()


class Composite(MultimodalModelMixin):
    """A composite model with a real text embedding table and in-vocabulary placeholders."""

    def __init__(self) -> None:
        self.model_config = ModelConfig()
        self.embedding = torch.nn.Embedding(16, 2)

    def encode_multimodal_inputs(self, params: Sequence[MultimodalParams]) -> torch.Tensor:
        return torch.cat([param.multimodal_data["image"]["rows"] for param in params])

    @property
    def text_embedding_layer(self) -> torch.nn.Module:
        return self.embedding

    @property
    def multimodal_token_ids(self) -> Sequence[int]:
        return (7,)


def image_params(num_rows: int, start: int = 0, end: int | None = None) -> MultimodalParams:
    return MultimodalParams(
        multimodal_data={"image": {"rows": torch.arange(num_rows * 2.0).reshape(num_rows, 2)}},
        multimodal_runtime=MultimodalRuntimeData(
            past_seen_token_num=start,
            chunk_end_pos=num_rows if end is None else end,
            embed_mask_cumsum=torch.arange(1, num_rows + 1),
        ),
    )


def test_the_runtime_base_delegates_nothing_to_a_separate_language_model():
    assert issubclass(MultimodalModelMixin, MultimodalRuntimeMixin)
    for name in _COMPOSITE_MEMBERS:
        assert not hasattr(MultimodalRuntimeMixin, name), name
        assert hasattr(MultimodalModelMixin, name), name
    assert not MultimodalRuntimeMixin.supports_concurrent_encoder
    assert MultimodalModelMixin.supports_concurrent_encoder


def test_runtime_defaults_leave_processor_limits_and_refuse_breakable_graphs():
    model = RowRuntime()
    assert model.max_encoder_tokens_per_item("image") is None
    assert not model.allows_breakable_prefill_cuda_graph(None)


def test_rows_come_back_paired_with_their_placeholder_positions():
    model = RowRuntime()
    indices = torch.tensor([1, 2, 4])
    prepared = model.prepare_multimodal_embeddings(
        input_ids=torch.zeros(5, dtype=torch.int32),
        multimodal_params=[image_params(3)],
        num_context_requests=1,
        mm_token_indices=indices,
    )
    assert prepared.mm_token_indices is indices
    assert sum(rows.shape[0] for rows in prepared.multimodal_embeddings) == 3
    # The model hooks keep receiving the executor's indices.
    assert model.hook_kwargs[0]["mm_token_indices"] is indices


def test_placeholder_positions_must_count_the_active_rows():
    with pytest.raises(ValueError, match="2 multimodal placeholder positions but 3 active"):
        RowRuntime().prepare_multimodal_embeddings(
            input_ids=torch.zeros(5, dtype=torch.int32),
            multimodal_params=[image_params(3)],
            num_context_requests=1,
            mm_token_indices=torch.tensor([1, 2]),
        )


def test_a_forward_without_active_rows_carries_no_positions():
    prepared = RowRuntime().prepare_multimodal_embeddings(
        input_ids=torch.zeros(2, dtype=torch.int32),
        multimodal_params=[image_params(3, start=3, end=3)],
        num_context_requests=1,
        mm_token_indices=torch.tensor([], dtype=torch.int64),
    )
    assert not prepared.multimodal_embeddings
    assert prepared.mm_token_indices is None


def test_composite_fusion_keeps_deriving_positions_it_was_not_given():
    """Composite models may receive executor indices that do not describe their rows; the
    default fusion derives positions from `multimodal_token_ids` when one index set is
    missing, so the composite path does not apply the runtime's pairing check."""
    model = Composite()
    input_ids = torch.tensor([1, 7, 7, 3])
    prepared = model.prepare_multimodal_inputs(
        input_ids=input_ids,
        positions=None,
        multimodal_params=[image_params(2)],
        num_context_requests=1,
        mm_token_indices=torch.tensor([], dtype=torch.int64),
    )
    assert prepared.input_ids is None
    torch.testing.assert_close(prepared.inputs_embeds[1:3], torch.arange(4.0).reshape(2, 2))
    torch.testing.assert_close(prepared.inputs_embeds[0], model.embedding.weight[1])


def test_class_contract_names_every_missing_member():
    class Incomplete(MultimodalRuntimeMixin):
        def encode_multimodal_inputs(self, params):
            return torch.zeros(0, 1)

    RowRuntime.check_multimodal_class_contract()
    with pytest.raises(MultimodalModelContractError, match="embedding_dim, embedding_dtype"):
        Incomplete.check_multimodal_class_contract()


def test_instance_contract_reports_the_failing_property_with_its_cause():
    class ComposedConfig(RowRuntime):
        @property
        def embedding_dim(self) -> int:
            return self.model_config.pretrained_config.text_width

    model = ComposedConfig()
    # Through `torch.nn.Module`, the property's AttributeError loses its cause.
    with pytest.raises(AttributeError, match="has no attribute 'embedding_dim'"):
        _ = model.embedding_dim
    with pytest.raises(MultimodalModelContractError, match=r"embedding_dim failed.*text_width"):
        model.check_multimodal_instance_contract()
    RowRuntime().check_multimodal_instance_contract()


@pytest.mark.parametrize("hidden_size", [0, 2.0])
def test_instance_contract_rejects_invalid_row_widths(hidden_size):
    with pytest.raises(MultimodalModelContractError, match="positive int"):
        RowRuntime(hidden_size=hidden_size).check_multimodal_instance_contract()
