from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_llava_next import \
    LlavaNextInputProcessor
from tensorrt_llm._torch.models.modeling_qwen2vl import \
    Qwen2_5VLInputProcessorBase
from tensorrt_llm._torch.models.modeling_qwen3vl import \
    Qwen3VLInputProcessorBase
from tensorrt_llm.inputs.data import TextPrompt
from tensorrt_llm.sampling_params import SamplingParams

# Model configurations for different processors
MODEL_CONFIGS = {
    "LlavaNextInputProcessor": {
        "processor_class":
        LlavaNextInputProcessor,
        "module_path":
        "tensorrt_llm._torch.models.modeling_llava_next",
        "config_setup":
        lambda mock_config: setattr_multiple(
            mock_config, {
                "image_token_index": 32001,
                "vocab_size": 32001,
                "text_config.hidden_size": 4096,
                "text_config.vocab_size": 32000,
                "vision_config": Mock(),
                "vision_feature_select_strategy": "default"
            })
    },
    # TODO: Add test for more VLM models
    # Future model configurations can be added here
    # "LlamaInputProcessor": {
    #     "processor_class": LlamaInputProcessor,
    #     "module_path": "tensorrt_llm._torch.models.modeling_llama",
    # },
    # "QwenInputProcessor": {
    #     "processor_class": QwenInputProcessor,
    #     "module_path": "tensorrt_llm._torch.models.modeling_qwen2vl",
    # },
}


def setattr_multiple(obj, attr_dict):
    """Helper function to set multiple nested attributes."""
    for attr_path, value in attr_dict.items():
        if '.' in attr_path:
            # Handle nested attributes like 'text_config.hidden_size'
            attrs = attr_path.split('.')
            current_obj = obj
            for attr in attrs[:-1]:
                if not hasattr(current_obj, attr):
                    setattr(current_obj, attr, Mock())
                current_obj = getattr(current_obj, attr)
            setattr(current_obj, attrs[-1], value)
        else:
            setattr(obj, attr_path, value)


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


@pytest.fixture(params=["LlavaNextInputProcessor"])
def processor_setup(request):
    """Fixture to set up different input processors based on the parameter."""
    processor_name = request.param
    config = MODEL_CONFIGS[processor_name]

    # Mock model configuration
    mock_config = Mock()
    config["config_setup"](mock_config)

    # Mock tokenizer and processor
    mock_tokenizer = Mock()
    mock_processor = Mock()

    # Create processor instance with mocks
    with patch(f'{config["module_path"]}.AutoTokenizer') as mock_auto_tokenizer, \
         patch(f'{config["module_path"]}.AutoProcessor') as mock_auto_processor:

        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_processor.from_pretrained.return_value = mock_processor

        processor = config["processor_class"](model_path="dummy_path",
                                              config=mock_config,
                                              tokenizer=mock_tokenizer,
                                              trust_remote_code=True)

        # Return processor along with mocks for test usage
        return {
            "processor": processor,
            "mock_tokenizer": mock_tokenizer,
            "mock_processor": mock_processor,
            "mock_config": mock_config,
            "processor_name": processor_name
        }


class TestExternalEmbedding:
    # TODO: Add test for more modalities (audio, video, etc.)
    def test_attach_multimodal_embeddings_image_basic(self, processor_setup):
        """Test basic functionality of attach_multimodal_embeddings."""
        import torch

        # Prepare test inputs
        text_prompt: TextPrompt = {"prompt": "What is in this image? <image>"}

        # Create mock embedding tensors (batch_size=1, seq_len=144, hidden_dim=768)
        image_embeddings = [
            torch.randn(144,
                        4096),  # Single image embedding so num_frames=1 always
        ]

        multimodal_embedding = {"image": image_embeddings}

        sampling_params = SamplingParams()

        # Mock tokenizer output - simulate tokenizing the text with image token
        mock_token_ids = torch.tensor([1, 2, 3, 32001, 4,
                                       5])  # image token at index 3
        processor_setup["mock_tokenizer"].return_value.input_ids = [
            mock_token_ids
        ]

        result_token_ids, extra_processed_inputs = processor_setup[
            "processor"].attach_multimodal_embeddings(text_prompt,
                                                      multimodal_embedding,
                                                      sampling_params)

        # Verify outputs
        assert isinstance(result_token_ids, list)
        assert len(result_token_ids) == len(mock_token_ids) + len(
            image_embeddings[0]) - 1
        assert isinstance(extra_processed_inputs, dict)
        assert "multimodal_data" in extra_processed_inputs
        assert "multimodal_embedding" in extra_processed_inputs[
            "multimodal_data"]

        # Check that multimodal embedding is properly formatted
        mm_embedding = extra_processed_inputs["multimodal_data"][
            "multimodal_embedding"]
        assert mm_embedding.shape[-1] == 4096  # Hidden dimension should match
        assert mm_embedding.shape[0] == 144

        # Verify tokenizer was called with correct prompt
        processor_setup["mock_tokenizer"].assert_called_once_with(
            text_prompt["prompt"], return_tensors="pt")

    def test_attach_multimodal_embeddings_multiple_images(
            self, processor_setup):
        """Test with multiple image embeddings."""
        import torch

        text_prompt: TextPrompt = {
            "prompt": "Compare these images: <image> and <image>"
        }

        # Create multiple image embeddings
        image_embeddings = [
            torch.randn(144, 4096),  # First image
            torch.randn(288, 4096),  # Second image
        ]

        # multiple images are concatenated along the first dimension
        multimodal_embedding = {"image": image_embeddings}

        sampling_params = SamplingParams()

        # Mock tokenizer output with two image tokens
        mock_token_ids = torch.tensor([1, 2, 32001, 3, 4, 32001, 5])
        processor_setup["mock_tokenizer"].return_value.input_ids = [
            mock_token_ids
        ]

        # Call the function
        result_token_ids, extra_processed_inputs = processor_setup[
            "processor"].attach_multimodal_embeddings(text_prompt,
                                                      multimodal_embedding,
                                                      sampling_params)

        # Verify outputs
        assert isinstance(result_token_ids, list)
        assert len(result_token_ids) == len(mock_token_ids) + torch.cat(
            image_embeddings, dim=0).shape[0] - 2
        assert isinstance(extra_processed_inputs, dict)
        assert "multimodal_data" in extra_processed_inputs
        assert "multimodal_embedding" in extra_processed_inputs[
            "multimodal_data"]

        # Check that multimodal embedding is properly formatted
        mm_embedding = extra_processed_inputs["multimodal_data"][
            "multimodal_embedding"]
        assert mm_embedding.shape[-1] == 4096  # Hidden dimension should match
        assert mm_embedding.shape[0] == 144 + 288

        # Verify tokenizer was called with correct prompt
        processor_setup["mock_tokenizer"].assert_called_once_with(
            text_prompt["prompt"], return_tensors="pt")


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
