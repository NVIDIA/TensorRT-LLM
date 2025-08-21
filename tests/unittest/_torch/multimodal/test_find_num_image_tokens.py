import io

import pytest
import requests
from PIL import Image
from transformers import AutoConfig, AutoTokenizer

from tensorrt_llm import MultimodalEncoder
from tensorrt_llm._torch.models.modeling_llava_next import \
    LlavaNextInputProcessor
from tensorrt_llm._torch.models.modeling_qwen2vl import \
    Qwen2VLInputProcessorBase
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.inputs import default_multimodal_input_loader

example_images = [
    "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
    "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg",
]


def download_image(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content))
    return img.convert("RGB")


@pytest.fixture(scope="function")
def multimodal_model_configs():
    """Get multimodal model configurations for testing."""
    model_configs = {
        'llava-v1.6-mistral-7b-hf': {
            'hf_model_dir': 'llava-hf/llava-v1.6-mistral-7b-hf',
            'model_type': 'llava_next',
        },
        'qwen2.5-vl': {
            'hf_model_dir': 'Qwen/Qwen2.5-VL-3B-Instruct',
            'model_type': 'qwen2_5_vl',
        }
    }
    return model_configs


@pytest.mark.parametrize("model_key", [
    "llava-v1.6-mistral-7b-hf",
    "qwen2.5-vl",
])
def test_get_num_tokens_per_image(model_key, multimodal_model_configs):
    """Test that get_num_tokens_per_image predicts the correct number of tokens.

    This test verifies that the get_num_tokens_per_image method correctly predicts
    the number of image tokens by comparing against actual encoder output shapes.
    Tests all example images in a single test run.
    """
    # Get model configuration
    if model_key not in multimodal_model_configs:
        pytest.skip(f"Skipping test for {model_key} - model not available")

    model_config = multimodal_model_configs[model_key]
    encoder_model_dir = model_config['hf_model_dir']
    model_type = model_config['model_type']

    # Test configuration
    max_batch_size = len(example_images)  # Batch size to handle all test images

    encoder = None

    try:
        encoder = MultimodalEncoder(model=encoder_model_dir,
                                    max_batch_size=max_batch_size)

        # Load model configuration and create input processor once
        model_config_dict = AutoConfig.from_pretrained(encoder_model_dir,
                                                       trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(encoder_model_dir,
                                                  trust_remote_code=True)

        # Create input processor once
        if model_type == 'llava_next':
            input_processor = LlavaNextInputProcessor(
                model_path=encoder_model_dir,
                model_config=model_config_dict,
                tokenizer=tokenizer,
                trust_remote_code=True)
        elif model_type == 'qwen2_5_vl':
            input_processor = Qwen2VLInputProcessorBase(
                model_path=encoder_model_dir,
                model_config=model_config_dict,
                tokenizer=tokenizer,
                trust_remote_code=True)
        else:
            pytest.fail(f"Unsupported model type: {model_type}")

        # Prepare batch inputs with all 3 images
        prompts = ["dummy"] * len(example_images)  # One prompt per image
        media = example_images  # All image URLs

        # Prepare inputs and get actual embeddings for all images
        inputs = default_multimodal_input_loader(tokenizer=tokenizer,
                                                 model_dir=encoder_model_dir,
                                                 model_type=model_type,
                                                 modality="image",
                                                 prompts=prompts,
                                                 media=media,
                                                 image_data_format="pt")

        # Get actual embeddings from encoder (batch processing)
        encoder_outputs = encoder.generate(inputs)
        assert len(encoder_outputs) == len(
            example_images
        ), f"Expected {len(example_images)} encoder outputs, got {len(encoder_outputs)}"

        for image_idx, test_image_url in enumerate(example_images):

            # Get test image dimensions
            test_image = download_image(test_image_url)
            image_width, image_height = test_image.size

            # Get actual embedding tensor for this image
            actual_embedding = SharedTensorContainer.from_dict(
                encoder_outputs[image_idx].mm_embedding_handle).get_local_view(
                )

            # The first dimension should be the number of image tokens
            actual_num_tokens = actual_embedding.shape[0]

            # Get predicted number of tokens using get_num_tokens_per_image
            if model_type == 'llava_next':
                predicted_num_tokens = input_processor.get_num_tokens_per_image(
                    image_width=image_width, image_height=image_height)
            elif model_type == 'qwen2_5_vl':
                predicted_num_tokens = input_processor.get_num_tokens_per_image(
                    image_width=image_width,
                    image_height=image_height,
                    num_frames=1,
                    do_resize=True)

            # The key assertion: predicted should match actual
            assert predicted_num_tokens == actual_num_tokens, \
                f"Token count mismatch for {model_key} with image {image_idx} ({image_width}x{image_height}): " \
                f"predicted={predicted_num_tokens}, actual={actual_num_tokens}"

            # Additional validation: ensure we got reasonable token counts
            assert actual_num_tokens > 0, f"Got zero image tokens for {model_key} image {image_idx}"
            assert actual_embedding.ndim >= 2, f"Expected at least 2D embedding, got {actual_embedding.ndim}D"

    finally:
        # Cleanup resources
        if encoder is not None:
            del encoder
