import os
from pathlib import Path

import pytest
from transformers import AutoConfig, AutoTokenizer
from utils.llm_data import llm_models_root

from tensorrt_llm import MultimodalEncoder
from tensorrt_llm._torch.models.modeling_llava_next import \
    LlavaNextInputProcessor
from tensorrt_llm._torch.models.modeling_qwen2vl import \
    Qwen2VLInputProcessorBase
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.inputs import default_multimodal_input_loader
from tensorrt_llm.inputs.utils import load_image, load_video

test_data_root = Path(
    os.path.join(llm_models_root(), "multimodals", "test_data"))
example_images = [
    str(test_data_root / "seashore.png"),
    str(test_data_root / "inpaint.png"),
    str(test_data_root / "61.jpg"),
]
example_videos = [
    str(test_data_root / "OAI-sora-tokyo-walk.mp4"),
    str(test_data_root / "world.mp4"),
]


@pytest.fixture(scope="function")
def multimodal_model_configs():
    """Get multimodal model configurations for testing."""
    model_configs = {
        'llava-v1.6-mistral-7b-hf': {
            'hf_model_dir': 'llava-hf/llava-v1.6-mistral-7b-hf',
            'model_dir':
            llm_models_root() / "multimodals" / "llava-v1.6-mistral-7b-hf",
            'model_type': 'llava_next',
        },
        'qwen2.5-vl': {
            'hf_model_dir': 'Qwen/Qwen2.5-VL-3B-Instruct',
            'model_dir': llm_models_root() / "Qwen2.5-VL-3B-Instruct",
            'model_type': 'qwen2_5_vl',
        },
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
    encoder_model_dir = model_config['model_dir']
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

        for image_idx, test_image in enumerate(example_images):

            # Get test image dimensions
            test_image = load_image(test_image, format="pil")
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
                    image=test_image)
            elif model_type == 'qwen2_5_vl':
                predicted_num_tokens = input_processor.get_num_tokens_per_image(
                    image=test_image)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

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


@pytest.mark.parametrize("model_key", [
    "qwen2.5-vl",
])
def test_get_num_tokens_per_video(model_key, multimodal_model_configs):
    """Test that get_num_tokens_per_video predicts the correct number of tokens.

    This test verifies that the get_num_tokens_per_video method correctly predicts
    the number of video tokens by comparing against actual encoder output shapes.
    Tests all example videos in a single test run.
    """
    # Get model configuration
    if model_key not in multimodal_model_configs:
        pytest.skip(f"Skipping test for {model_key} - model not available")

    model_config = multimodal_model_configs[model_key]
    encoder_model_dir = model_config['model_dir']
    model_type = model_config['model_type']

    # Test configuration
    max_batch_size = len(example_videos)  # Batch size to handle all test videos

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
        prompts = ["dummy"] * len(example_videos)  # One prompt per video
        media = example_videos  # All video URLs

        # Prepare inputs and get actual embeddings for all images
        inputs = default_multimodal_input_loader(tokenizer=tokenizer,
                                                 model_dir=encoder_model_dir,
                                                 model_type=model_type,
                                                 modality="video",
                                                 prompts=prompts,
                                                 media=media,
                                                 image_data_format="pt")

        # Get actual embeddings from encoder (batch processing)
        encoder_outputs = encoder.generate(inputs)
        assert len(encoder_outputs) == len(
            example_videos
        ), f"Expected {len(example_videos)} encoder outputs, got {len(encoder_outputs)}"

        for video_idx, test_video in enumerate(example_videos):

            # Get test video dimensions
            video_data = load_video(test_video, num_frames=8, format="pil")
            # load_video returns VideoData with frames and metadata
            video_width, video_height = video_data.frames[0].size

            # Get actual embedding tensor for this image
            actual_embedding = SharedTensorContainer.from_dict(
                encoder_outputs[video_idx].mm_embedding_handle).get_local_view(
                )

            # The first dimension should be the number of image tokens
            actual_num_tokens = actual_embedding.shape[0]

            # Get predicted number of tokens using get_num_tokens_per_video
            if model_type == 'llava_next':
                predicted_num_tokens = input_processor.get_num_tokens_per_video(
                    video=video_data.frames)
            elif model_type == 'qwen2_5_vl':
                predicted_num_tokens = input_processor.get_num_tokens_per_video(
                    video=video_data.frames)

            # The key assertion: predicted should match actual
            assert predicted_num_tokens == actual_num_tokens, \
                f"Token count mismatch for {model_key} with video {video_idx} ({video_width}x{video_height}): " \
                f"predicted={predicted_num_tokens}, actual={actual_num_tokens}"

            # Additional validation: ensure we got reasonable token counts
            assert actual_num_tokens > 0, f"Got zero video tokens for {model_key} video {video_idx}"
            assert actual_embedding.ndim >= 2, f"Expected at least 2D embedding, got {actual_embedding.ndim}D"

    finally:
        # Cleanup resources
        if encoder is not None:
            del encoder
