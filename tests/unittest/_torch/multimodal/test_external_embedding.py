# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_llama import Llama4InputProcessor
from tensorrt_llm._torch.models.modeling_llava_next import \
    LlavaNextInputProcessor
from tensorrt_llm._torch.models.modeling_qwen3vl import \
    Qwen3VLInputProcessorBase
from tensorrt_llm.inputs.multimodal import MultimodalInput
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor
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


def _make_qwen3_vl_processor(
    config: SimpleNamespace,
    tokenizer: Mock,
    processor_cls: type[Qwen3VLInputProcessorBase] = Qwen3VLInputProcessorBase
) -> Qwen3VLInputProcessorBase:
    with patch("tensorrt_llm._torch.models.modeling_qwen2vl.AutoProcessor"
               ) as mock_auto_processor:
        mock_auto_processor.from_pretrained.return_value = Mock(
            mm_special_token_ids=None)
        return processor_cls(
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


def _attach(processor: BaseMultimodalInputProcessor, inputs: dict[str, Any],
            image_embeddings: Any) -> tuple[list[int], dict[str, Any] | None]:
    return processor.attach_multimodal_embeddings(inputs,
                                                  {"image": image_embeddings},
                                                  SamplingParams())


def test_qwen3_vl_attached_embeddings_are_keyed_by_content() -> None:
    config = _make_qwen3_vl_config(hidden_size=4)
    num_levels = 1 + len(config.vision_config.deepstack_visual_indexes)
    width = num_levels * config.text_config.hidden_size
    vision = (config.vision_start_token_id, config.image_token_id,
              config.vision_end_token_id)
    # Two back-to-back images of different sizes.
    processor = _make_qwen3_vl_processor(
        config, _make_tokenizer([5, *vision, *vision, 6]))
    inputs = {"prompt": "Compare the two pictures."}
    first = torch.full((3, width), 0.5, dtype=torch.bfloat16)
    second = torch.full((2, width), -0.5, dtype=torch.bfloat16)

    _, extra_processed_inputs = _attach(processor, inputs, [first, second])
    multimodal_input = extra_processed_inputs["multimodal_input"]
    hashes = multimodal_input.multimodal_hashes

    # Prompt: 5 <vs> img*3 <ve> <vs> img*2 <ve> 6.
    assert multimodal_input.multimodal_positions == [2, 7]
    assert multimodal_input.multimodal_lengths == [3, 2]
    assert multimodal_input.multimodal_run_positions == [2, 7]
    assert multimodal_input.multimodal_run_lengths == [3, 2]
    assert hashes[0] != hashes[1]
    cumsum = extra_processed_inputs["multimodal_data"][
        "multimodal_embed_mask_cumsum"]
    assert cumsum.tolist() == [0, 0, 1, 2, 3, 3, 3, 4, 5, 5, 5]

    same = _attach(processor, inputs, [first.clone(), second.clone()])[1]
    assert same["multimodal_input"].multimodal_hashes == hashes
    swapped = _attach(processor, inputs, [second, first])[1]
    assert swapped["multimodal_input"].multimodal_hashes == hashes[::-1]
    changed = _attach(processor, inputs, [first, second + 1])[1]
    assert changed["multimodal_input"].multimodal_hashes[0] == hashes[0]
    assert changed["multimodal_input"].multimodal_hashes[1] != hashes[1]


def test_qwen3_vl_attached_embedding_uuid_joins_its_hash() -> None:
    hidden_size = 8
    config = _make_qwen3_vl_config(hidden_size)
    embedding_width = hidden_size * (
        1 + len(config.vision_config.deepstack_visual_indexes))
    processor = _make_qwen3_vl_processor(
        config,
        _make_tokenizer([
            7,
            config.vision_start_token_id,
            config.image_token_id,
            config.vision_end_token_id,
        ]))
    embedding = torch.zeros(4, embedding_width)

    uuids = {"image": ["user-a"]}

    plain = _attach(processor, {"prompt": "Describe."},
                    [embedding])[1]["multimodal_input"]
    with_uuid = _attach(processor, {
        "prompt": "Describe.",
        "multi_modal_uuids": uuids
    }, [embedding])[1]["multimodal_input"]
    other_uuid = _attach(processor, {
        "prompt": "Describe.",
        "multi_modal_uuids": {
            "image": ["user-b"]
        }
    }, [embedding])[1]["multimodal_input"]
    other_rows = _attach(processor, {
        "prompt": "Describe.",
        "multi_modal_uuids": uuids
    }, [embedding + 1])[1]["multimodal_input"]
    # Token IDs are decoded back to text before the hook runs.
    from_token_ids = _attach(
        processor, {
            "prompt_token_ids": [7, config.vision_start_token_id],
            "multi_modal_uuids": uuids
        }, [embedding])[1]["multimodal_input"]

    assert plain.multimodal_uuids is None
    assert with_uuid.multimodal_uuids == ["user-a"]
    assert with_uuid.multimodal_hashes != plain.multimodal_hashes
    assert other_uuid.multimodal_hashes != with_uuid.multimodal_hashes
    # The UUID joins the content; it does not replace it.
    assert other_rows.multimodal_hashes != with_uuid.multimodal_hashes
    assert from_token_ids.multimodal_uuids == ["user-a"]
    assert from_token_ids.multimodal_hashes == with_uuid.multimodal_hashes


class _NoItemMetadataProcessor(Qwen3VLInputProcessorBase):

    def _attach_multimodal_embeddings_impl(
            self, inputs: dict[str, Any], multimodal_embedding: dict[str, Any],
            sampling_params: SamplingParams
    ) -> tuple[list[int], dict[str, Any]]:
        return [7], {
            "multimodal_data": {
                "multimodal_embedding": multimodal_embedding["image"]
            }
        }


def test_attached_embeddings_require_item_metadata() -> None:
    processor = _make_qwen3_vl_processor(_make_qwen3_vl_config(8),
                                         _make_tokenizer([7]),
                                         processor_cls=_NoItemMetadataProcessor)

    with pytest.raises(ValueError, match="MultimodalItemMetadata"):
        _attach(processor, {"prompt": "Describe."}, [torch.zeros(1, 16)])


def _make_llava_next_processor(
        hidden_size: int, image_token_index: int) -> LlavaNextInputProcessor:
    config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=hidden_size,
                                    torch_dtype=torch.float32),
        torch_dtype=torch.float32,
        image_token_index=image_token_index,
        vocab_size=64,
    )
    with patch("tensorrt_llm._torch.models.modeling_llava_next.AutoProcessor"
               ) as mock_auto_processor:
        mock_auto_processor.from_pretrained.return_value = Mock(
            mm_special_token_ids=None)
        return LlavaNextInputProcessor(model_path="dummy_path",
                                       config=config,
                                       tokenizer=Mock(),
                                       trust_remote_code=True)


def test_llava_next_attached_embeddings_are_keyed_per_image() -> None:
    hidden_size = 8
    image_token_index = 32
    processor = _make_llava_next_processor(hidden_size, image_token_index)
    inputs = {"prompt_token_ids": [1, image_token_index, 5, image_token_index]}
    first = torch.zeros(3, hidden_size)
    second = torch.ones(3, hidden_size)

    prompt_token_ids, extra_processed_inputs = _attach(processor, inputs,
                                                       [first, second])
    multimodal_input = extra_processed_inputs["multimodal_input"]

    assert prompt_token_ids == [
        1, *[image_token_index] * 3, 5, *[image_token_index] * 3
    ]
    assert multimodal_input.multimodal_positions == [1, 5]
    assert multimodal_input.multimodal_lengths == [3, 3]
    # The processor also takes the images as one tensor split per placeholder.
    concatenated = _attach(processor, inputs, torch.cat([first, second]))[1]
    assert (concatenated["multimodal_input"].multimodal_hashes ==
            multimodal_input.multimodal_hashes)
    changed = _attach(processor, inputs, [first, second + 1])[1]
    assert (changed["multimodal_input"].multimodal_hashes[0] ==
            multimodal_input.multimodal_hashes[0])
    assert (changed["multimodal_input"].multimodal_hashes[1]
            != multimodal_input.multimodal_hashes[1])
    # Images of different sizes keep their own row counts.
    uneven = _attach(processor, inputs,
                     [first[:2], second])[1]["multimodal_input"]
    assert uneven.multimodal_positions == [1, 4]
    assert uneven.multimodal_lengths == [2, 3]


def test_llama4_attached_embeddings_are_keyed_per_image() -> None:
    hidden_size = 8
    patch_token, image_start, tile_separator, image_end = 40, 41, 43, 42
    config = SimpleNamespace(
        torch_dtype=torch.float32,
        text_config=SimpleNamespace(vocab_size=64, hidden_size=hidden_size),
        image_token_index=patch_token,
        boi_token_index=image_start,
        eoi_token_index=image_end,
    )
    # Each image: four patches in two tiles, framed by special tokens.
    image_tokens = [
        image_start, patch_token, patch_token, tile_separator, patch_token,
        patch_token, image_end
    ]
    tokenizer = Mock(vocab_size=64)
    tokenizer.return_value.input_ids = torch.tensor(
        [[1, *image_tokens, 5, *image_tokens, 2]])
    tokenizer.decode.return_value = "<|special|>"
    with patch("tensorrt_llm._torch.models.modeling_llama.AutoProcessor"
               ) as mock_auto_processor:
        mock_auto_processor.from_pretrained.return_value = Mock(
            fake_image_token="<|image|>",
            img_patch_token="<|patch|>",
            mm_special_token_ids=None)
        processor = Llama4InputProcessor(model_path="dummy_path",
                                         config=config,
                                         tokenizer=tokenizer)

    def attach(*embeddings: torch.Tensor) -> MultimodalInput:
        items = [{
            "mm_embeddings":
            embedding,
            "image_special_tokens": [image_start, tile_separator, image_end],
            "image_special_token_offsets": [0, 3, 6],
        } for embedding in embeddings]
        return _attach(processor, {"prompt": "<|image|>Compare.<|image|>"},
                       items)[1]["multimodal_input"]

    first = torch.zeros(4, hidden_size)
    second = torch.ones(4, hidden_size)
    multimodal_input = attach(first, second)
    hashes = multimodal_input.multimodal_hashes

    assert multimodal_input.multimodal_positions == [2, 10]
    assert multimodal_input.multimodal_lengths == [4, 4]
    assert multimodal_input.multimodal_run_positions == [2, 5, 10, 13]
    assert multimodal_input.multimodal_run_lengths == [2, 2, 2, 2]
    assert hashes[0] != hashes[1]
    assert attach(second, first).multimodal_hashes == hashes[::-1]
    changed = attach(first, second + 1).multimodal_hashes
    assert changed[0] == hashes[0]
    assert changed[1] != hashes[1]


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


if __name__ == "__main__":
    pytest.main([__file__])
