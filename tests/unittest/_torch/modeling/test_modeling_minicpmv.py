# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.minicpmv_weight_mapper import MiniCPMVHfWeightMapper
from tensorrt_llm._torch.models.modeling_minicpmv import (
    MiniCPMVForConditionalGeneration,
    MiniCPMVInputProcessor,
    MiniCPMVVisionModel,
)
from tensorrt_llm._torch.models.modeling_utils import (
    MODEL_CLASS_MAPPER_MAPPING,
    MODEL_CLASS_MAPPING,
)
from tensorrt_llm.inputs.multimodal import MultimodalParams, find_mm_token_lengths


def _make_vision_model(patch_size: int = 2) -> MiniCPMVVisionModel:
    model = MiniCPMVVisionModel.__new__(MiniCPMVVisionModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(patch_size=patch_size),
        vision_batch_size=16,
    )
    return model


def _make_multimodal_param(
    modality: str,
    pixel_values: list[torch.Tensor] | None,
    tgt_sizes: list[list[int]] | None,
    temporal_ids: list[list[int]] | None = None,
) -> MultimodalParams:
    return MultimodalParams(
        multimodal_data={
            modality: {
                "pixel_values": pixel_values,
                "tgt_sizes": tgt_sizes,
                "temporal_ids": temporal_ids,
            }
        }
    )


def test_minicpmv_model_and_mapper_are_registered():
    assert MODEL_CLASS_MAPPING["MiniCPMV"] is MiniCPMVForConditionalGeneration
    assert MODEL_CLASS_MAPPER_MAPPING["MiniCPMV_HF"] is MiniCPMVHfWeightMapper


def test_minicpmv_weight_mapper_normalizes_llm_prefix():
    llm_weight = torch.ones(1)
    vision_weight = torch.zeros(1)

    mapped = MiniCPMVHfWeightMapper().preprocess_weights(
        {
            "llm.model.embed_tokens.weight": llm_weight,
            "vpm.embeddings.patch_embedding.weight": vision_weight,
        }
    )

    assert mapped["model.embed_tokens.weight"] is llm_weight
    assert mapped["vpm.embeddings.patch_embedding.weight"] is vision_weight
    assert "llm.model.embed_tokens.weight" not in mapped


def test_minicpmv_tokenizer_loader_does_not_mask_unexpected_errors():
    processor = MiniCPMVInputProcessor.__new__(MiniCPMVInputProcessor)
    processor._processor = SimpleNamespace(tokenizer=object())
    processor._tokenizer = object()
    processor._use_fast = True

    with (
        patch(
            "tensorrt_llm._torch.models.modeling_minicpmv.AutoTokenizer.from_pretrained",
            side_effect=RuntimeError("unexpected tokenizer failure"),
        ),
        pytest.raises(RuntimeError, match="unexpected tokenizer failure"),
    ):
        processor._load_processor_tokenizer("unused", trust_remote_code=True)


def test_minicpmv_batching_accepts_missing_pixel_values():
    model = _make_vision_model()
    param = _make_multimodal_param("image", None, None)

    batched = model._parse_and_batch_multimodal_data([param])

    assert batched["pixel_values"] is None
    assert batched["tgt_sizes"] is None
    assert batched["request_slice_counts"] == [0]
    assert batched["request_embedding_counts"] == [0]


def test_minicpmv_batching_aligns_temporal_groups_and_builds_2d_mask():
    model = _make_vision_model()
    image_param = _make_multimodal_param(
        "image",
        [torch.ones(3, 4, 6)],
        [[2, 3]],
    )
    video_param = _make_multimodal_param(
        "video",
        [torch.ones(3, 4, 4), torch.ones(3, 4, 4)],
        [[2, 2], [2, 2]],
        temporal_ids=[[0, 1]],
    )

    batched = model._parse_and_batch_multimodal_data([image_param, video_param])

    assert batched["pixel_values"].shape == (3, 3, 4, 6)
    assert batched["patch_attention_mask"].shape == (3, 2, 3)
    assert torch.equal(
        batched["patch_attention_mask"][0],
        torch.tensor([[True, True, True], [True, True, True]]),
    )
    assert torch.equal(
        batched["patch_attention_mask"][1],
        torch.tensor([[True, True, False], [True, True, False]]),
    )
    assert batched["temporal_ids"] == [[-1], [0, 1]]
    assert batched["request_slice_counts"] == [1, 2]
    assert batched["request_embedding_counts"] == [1, 1]


def test_minicpmv_batching_builds_mask_for_packed_patch_sequence():
    model = _make_vision_model()
    param = _make_multimodal_param(
        "image",
        [torch.ones(3, 2, 12)],
        [[2, 3]],
    )

    batched = model._parse_and_batch_multimodal_data([param])

    assert batched["patch_attention_mask"].shape == (1, 1, 6)
    assert torch.all(batched["patch_attention_mask"])


def test_minicpmv_batching_rejects_misaligned_temporal_ids():
    model = _make_vision_model()
    param = _make_multimodal_param(
        "video",
        [torch.ones(3, 4, 4), torch.ones(3, 4, 4)],
        [[2, 2], [2, 2]],
        temporal_ids=[[0]],
    )

    with pytest.raises(ValueError, match="temporal_ids must cover every visual slice"):
        model._parse_and_batch_multimodal_data([param])


class _FakeVisionTower(nn.Module):
    dtype = torch.float32

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.Tensor,
        tgt_sizes: torch.Tensor,
    ) -> torch.Tensor:
        del patch_attention_mask, tgt_sizes
        return torch.zeros(pixel_values.shape[0], 1, 4)


class _FakeResampler(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("output", output)

    def forward(
        self,
        vision_hidden_states: torch.Tensor,
        tgt_sizes: torch.Tensor,
        temporal_ids: list[list[int]] | None,
    ) -> torch.Tensor:
        del vision_hidden_states, tgt_sizes, temporal_ids
        return self.output


def test_minicpmv_forward_preserves_request_embedding_order():
    model = _make_vision_model()
    expected = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    model.vision_tower = _FakeVisionTower()
    model.resampler = _FakeResampler(expected)
    image_param = _make_multimodal_param(
        "image",
        [torch.ones(3, 4, 4)],
        [[2, 2]],
    )
    video_param = _make_multimodal_param(
        "video",
        [torch.ones(3, 4, 4), torch.ones(3, 4, 4)],
        [[2, 2], [2, 2]],
        temporal_ids=[[0, 1]],
    )

    embeddings = model([image_param, video_param])

    assert len(embeddings) == 1
    assert torch.equal(embeddings[0], expected.reshape(-1, 4))


class _RecordingVideoProcessor:
    def __init__(self) -> None:
        self.calls = []

    def get_num_tokens_per_video(self, **kwargs) -> int:
        self.calls.append(kwargs)
        return len(kwargs["video"])


def test_find_mm_token_lengths_ignores_misaligned_video_temporal_ids():
    processor = _RecordingVideoProcessor()
    frames = [[object()], [object()]]

    with patch("tensorrt_llm.inputs.multimodal.logger.warning") as warning:
        token_lengths = find_mm_token_lengths(
            {"video": frames},
            processor,
            multimodal_data={"video": {"temporal_ids": [[0]]}},
        )

    assert token_lengths == {"video": [1, 1]}
    assert all("temporal_ids" not in call for call in processor.calls)
    warning.assert_called_once()
    assert "temporal_ids item count" in warning.call_args.args[0]
