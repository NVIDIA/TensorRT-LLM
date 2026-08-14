from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen2vl import \
    Qwen2_5VLInputProcessorBase
from tensorrt_llm._torch.models.modeling_qwen3vl import \
    Qwen3VLInputProcessorBase
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


def _make_qwen_vl_config(hidden_size: int, deepstack_visual_indexes=None):
    text_config = SimpleNamespace(hidden_size=hidden_size,
                                  vocab_size=100,
                                  dtype=torch.float32)
    vision_config = SimpleNamespace(
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=2,
        deepstack_visual_indexes=deepstack_visual_indexes or [],
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


def _make_qwen_vl_processor(processor_cls, config, tokenizer):
    with patch("tensorrt_llm._torch.models.modeling_qwen2vl.AutoProcessor"
               ) as mock_auto_processor:
        mock_auto_processor.from_pretrained.return_value = Mock()
        return processor_cls(
            model_path="dummy_path",
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=True,
        )


@pytest.mark.parametrize(
    "processor_cls,deepstack_visual_indexes",
    [
        pytest.param(Qwen2_5VLInputProcessorBase, [], id="qwen2.5-vl"),
        pytest.param(Qwen3VLInputProcessorBase, [0], id="qwen3-vl"),
    ],
)
def test_qwen_vl_attach_multimodal_embeddings_builds_mrope_config(
    processor_cls,
    deepstack_visual_indexes,
):
    hidden_size = 8
    config = _make_qwen_vl_config(hidden_size, deepstack_visual_indexes)
    expected_embedding_width = hidden_size * (1 + len(deepstack_visual_indexes))
    tokenizer = _make_tokenizer([
        7,
        config.vision_start_token_id,
        config.image_token_id,
        config.vision_end_token_id,
        8,
    ])
    processor = _make_qwen_vl_processor(processor_cls, config, tokenizer)

    num_image_tokens = 4
    image_embedding = torch.arange(
        num_image_tokens * expected_embedding_width,
        dtype=torch.float32,
    ).reshape(num_image_tokens, expected_embedding_width)

    prompt_token_ids, extra_processed_inputs = processor.attach_multimodal_embeddings(
        {"prompt": "Describe this image."},
        {"image": [image_embedding]},
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


if __name__ == "__main__":
    pytest.main([__file__])
