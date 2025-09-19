from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from accelerate.utils import modeling
from transformers.models.llama4.configuration_llama4 import Llama4Config

from tensorrt_llm._torch.auto_deploy.models.hf import (
    AutoModelForCausalLMFactory,
    hf_load_state_dict_with_device,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)


def test_hf_load_state_dict_with_device():
    """Test that hf_load_state_dict_with_device correctly patches modeling.load_state_dict."""
    # Create mock for original load_state_dict
    original_load_state_dict = MagicMock()

    # Test with CPU device
    with patch.object(modeling, "load_state_dict", original_load_state_dict):
        with hf_load_state_dict_with_device(device="cpu"):
            # Call the patched function
            modeling.load_state_dict("dummy_checkpoint")

            # Check that device was set correctly
            original_load_state_dict.assert_called_once_with(
                "dummy_checkpoint", device_map={"": "cpu"}
            )

            # Reset mock for next test
            original_load_state_dict.reset_mock()

        # Check that original behavior is restored
        modeling.load_state_dict("dummy_checkpoint", device_map="original_device_map")
        original_load_state_dict.assert_called_once_with(
            "dummy_checkpoint", device_map="original_device_map"
        )
        original_load_state_dict.reset_mock()

    # Test with CUDA device (if available)
    if torch.cuda.is_available():
        with patch.object(modeling, "load_state_dict", original_load_state_dict):
            with hf_load_state_dict_with_device(device="cuda"):
                # Call the patched function
                modeling.load_state_dict("dummy_checkpoint")

                # Check that device was set correctly
                original_load_state_dict.assert_called_once_with(
                    "dummy_checkpoint", device_map={"": "cuda"}
                )


@pytest.fixture
def mock_factory():
    with (
        patch.object(AutoModelForCausalLMFactory, "prefetch_checkpoint"),
        patch.object(AutoModelForCausalLMFactory, "_load_quantization_config"),
    ):
        # Create factory instance with mocked methods to avoid HTTP requests
        factory = AutoModelForCausalLMFactory(model="dummy_model")
        # Set model path directly to avoid prefetch
        factory._prefetched_model_path = "/dummy/path"
        yield factory


def test_recursive_update_config(mock_factory):
    """Test that _recursive_update_config correctly updates a config object recursively."""
    # Get the mocked factory instance
    factory = mock_factory

    # Create a Llama4Config instance
    config = Llama4Config()

    # Create an update dictionary with both simple and nested values
    update_dict = {
        "bos_token_id": 42,  # Simple value at root level
        "text_config": {  # Nested config update
            "hidden_size": 4096,
            "num_attention_heads": 32,
        },
        "vision_config": {  # Another nested config update
            "hidden_size": 1024,
            "image_size": 224,
        },
        "non_existent_key": "this should be ignored",  # This key doesn't exist in the config
    }

    # Apply the recursive update
    updated_config, nested_unused = factory._recursive_update_config(config, update_dict)

    # Check that it returns the same object
    assert updated_config is config

    # Check root level updates
    assert config.bos_token_id == 42

    # Check nested updates in text_config
    assert config.text_config.hidden_size == 4096
    assert config.text_config.num_attention_heads == 32

    # Check nested updates in vision_config
    assert config.vision_config.hidden_size == 1024
    assert config.vision_config.image_size == 224

    # Check that non-existent keys were ignored
    assert not hasattr(config, "non_existent_key")
    # Check that nested_unused contains the non-existent key
    assert nested_unused == {"non_existent_key": "this should be ignored"}

    # Create a more complex update with deeper nesting
    complex_update = {"text_config": {"rope_scaling": {"factor": 2.0, "type": "linear"}}}

    # Apply the recursive update again
    factory._recursive_update_config(config, complex_update)

    # Check that complex nested updates were applied correctly
    assert config.text_config.rope_scaling["factor"] == 2.0
    assert config.text_config.rope_scaling["type"] == "linear"
