# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_llava_next import \
    LlavaNextInputProcessor
from tensorrt_llm._torch.models.modeling_qwen3vl import \
    Qwen3VLInputProcessorBase
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


def _make_qwen3_vl_config(hidden_size: int):
    text_config = SimpleNamespace(hidden_size=hidden_size,
                                  vocab_size=100,
                                  dtype=torch.float32)
    vision_config = SimpleNamespace(
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=2,
        deepstack_visual_indexes=[0],
    )
    return SimpleNamespace(
        torch_dtype=torch.float32,
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=12,
        video_token_id=14,
        vision_start_token_id=11,
        vision_end_token_id=13,
    )


def _make_tokenizer(input_ids):
    tokenizer = Mock()
    tokenizer.return_value.input_ids = [torch.tensor(input_ids)]
    return tokenizer


def _make_qwen3_vl_processor(config, tokenizer):
    with patch("tensorrt_llm._torch.models.modeling_qwen2vl.AutoProcessor"
               ) as mock_auto_processor:
        mock_auto_processor.from_pretrained.return_value = Mock(
            mm_special_token_ids=None)
        return Qwen3VLInputProcessorBase(
            model_path="dummy_path",
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=True,
        )


def test_qwen3_vl_attach_multimodal_embeddings_builds_mrope_config():
    hidden_size = 8
    config = _make_qwen3_vl_config(hidden_size)
    expected_embedding_width = hidden_size * (
        1 + len(config.vision_config.deepstack_visual_indexes))
    tokenizer = _make_tokenizer([
        7,
        config.vision_start_token_id,
        config.image_token_id,
        config.vision_end_token_id,
        8,
    ])
    processor = _make_qwen3_vl_processor(config, tokenizer)

    num_image_tokens = 4
    image_embedding = torch.arange(
        num_image_tokens * expected_embedding_width,
        dtype=torch.float32,
    ).reshape(num_image_tokens, expected_embedding_width)

    prompt_token_ids, extra_processed_inputs = processor.attach_multimodal_embeddings(
        {"prompt": "Describe this image."},
        {
            "image": [{
                "mm_embeddings": image_embedding,
                "image_grid_thw": [1, 4, 4]
            }]
        },
        SamplingParams(),
    )

    # In-vocab contract: placeholders stay as the real image_token_id (no
    # legacy OOV remap); the model engine locates mm positions via torch.isin.
    placeholder_id = config.image_token_id
    assert prompt_token_ids == [
        7,
        config.vision_start_token_id,
        placeholder_id,
        placeholder_id,
        placeholder_id,
        placeholder_id,
        config.vision_end_token_id,
        8,
    ]
    tokenizer.assert_called_once_with("Describe this image.",
                                      return_tensors="pt")

    multimodal_data = extra_processed_inputs["multimodal_data"]
    assert multimodal_data["multimodal_embedding"] == [image_embedding]

    mrope_config = multimodal_data["mrope_config"]
    expected_position_ids = torch.tensor(
        [
            [[0, 1, 2, 2, 2, 2, 4, 5]],
            [[0, 1, 2, 2, 3, 3, 4, 5]],
            [[0, 1, 2, 3, 2, 3, 4, 5]],
        ],
        dtype=torch.long,
    )
    torch.testing.assert_close(mrope_config["mrope_position_ids"],
                               expected_position_ids)
    torch.testing.assert_close(
        mrope_config["mrope_position_deltas"],
        torch.tensor([[-2]], dtype=torch.int32),
    )


@pytest.mark.parametrize("with_grid", [False, True], ids=["tensor", "grid"])
def test_llava_next_embeddings_keep_tensor_list(with_grid):
    """Models without mRoPE take a plain list of tensors and ignore a grid."""
    processor = object.__new__(LlavaNextInputProcessor)
    processor._config = SimpleNamespace(
        image_token_index=9, text_config=SimpleNamespace(hidden_size=4))
    embeddings = [torch.ones(3, 4), torch.zeros(2, 4)]
    items = embeddings
    if with_grid:
        # trtllm-serve passes an image_embeds part with a grid in this form.
        items = [{
            "mm_embeddings": embedding,
            "image_grid_thw": [1, 2, 2]
        } for embedding in embeddings]

    ids, extra = processor.attach_multimodal_embeddings(
        {"prompt_token_ids": [1, 9, 2, 9, 3]}, {"image": items},
        SamplingParams())

    assert ids == [1, 9, 9, 9, 2, 9, 9, 3]
    torch.testing.assert_close(extra["multimodal_data"]["multimodal_embedding"],
                               torch.cat(embeddings))


def test_llava_prefill_uses_encoder_prompt_and_item_handles():
    from tensorrt_llm.disaggregated_params import DisaggregatedParams
    from tensorrt_llm.llmapi.llm import BaseLLM

    processor = object.__new__(LlavaNextInputProcessor)
    processor._config = SimpleNamespace(
        vocab_size=100, text_config=SimpleNamespace(hidden_size=4))
    processor.image_token_index = 9
    processor._processor = SimpleNamespace()
    llm = object.__new__(BaseLLM)
    llm.input_processor = processor
    llm._is_encoder_decoder_model = lambda: False
    ids = [1, 9, 9, 9, 9, 9, 2]
    handles = [{"tensor_size": [3, 4]}, {"tensor_size": [2, 4]}]
    handoff = DisaggregatedParams(
        request_type="context_only",
        multimodal_embedding_handles=handles,
        multimodal_input=dict(multimodal_hashes=[[1] * 8, [2] * 8],
                              multimodal_positions=[1, 4],
                              multimodal_lengths=[3, 2],
                              multimodal_item_run_cu_offsets=[0, 1, 2],
                              multimodal_run_positions=[1, 4],
                              multimodal_run_lengths=[3, 2]),
    )
    out, _, params, _ = llm._preprocess({"prompt_token_ids": ids},
                                        SamplingParams(), handoff)
    assert out == ids
    assert params.multimodal_data["multimodal_embedding"] == handles
    assert params.multimodal_input.multimodal_positions == [1, 4]
    assert params.multimodal_data["multimodal_embed_mask_cumsum"].tolist() == [
        0, 1, 2, 3, 4, 5, 5
    ]


if __name__ == "__main__":
    pytest.main([__file__])
