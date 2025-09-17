import pathlib as _pl
from typing import Any

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints import HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.base_config_loader import \
    BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    BaseWeightLoader
from tensorrt_llm._torch.models.modeling_dummy import DummyConfig
from tensorrt_llm._torch.models.modeling_utils import (
    register_checkpoint_weight_loader, register_config_loader)
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.sampling_params import AdditionalModelOutput


@pytest.mark.part0
def test_additional_model_outputs_sampling_params():
    """Test that additional_model_outputs can be configured in SamplingParams."""
    # Create sampling params with additional outputs
    additional_outputs = [
        AdditionalModelOutput(name="context_output", gather_context=True),
        AdditionalModelOutput(name="generation_output", gather_context=False),
    ]

    sampling_params = SamplingParams(
        max_tokens=5,
        temperature=0.0,  # Deterministic for testing
        additional_model_outputs=additional_outputs)

    # Verify the sampling params are configured correctly
    assert sampling_params.additional_model_outputs is not None
    assert len(sampling_params.additional_model_outputs) == 2
    assert sampling_params.additional_model_outputs[0].name == "context_output"
    assert sampling_params.additional_model_outputs[0].gather_context
    assert sampling_params.additional_model_outputs[
        1].name == "generation_output"
    assert not sampling_params.additional_model_outputs[1].gather_context


@pytest.mark.part0
def test_additional_model_outputs_no_outputs():
    """Test that no additional outputs are returned when not requested."""
    # Create sampling params without additional outputs
    sampling_params = SamplingParams(
        max_tokens=5,
        temperature=0.0,  # Deterministic for testing
    )

    # Verify that no additional outputs are configured
    assert sampling_params.additional_model_outputs is None


@register_checkpoint_weight_loader("DUMMY_FORMAT")
class DummyWeightLoader(BaseWeightLoader):

    def load_weights(self, checkpoint_dir: str, **kwargs) -> dict[str, Any]:
        """Load weights from your dummy format.

        Args:
            checkpoint_dir: Directory containing checkpoint files
            **kwargs: Additional loading parameters
        Returns:
            Dictionary mapping parameter names to tensors
        """
        weights = {}

        return weights


@register_config_loader("DUMMY_FORMAT")
class DummyConfigLoader(BaseConfigLoader):

    def load(self, checkpoint_dir: str, **kwargs) -> ModelConfig:
        """Load and parse configuration from your dummy format.

        Args:
            checkpoint_dir: Directory containing configuration files
            **kwargs: Additional loading parameters
        Returns:
            ModelConfig object containing parsed configuration
        """
        return ModelConfig(pretrained_config=DummyConfig())


@pytest.mark.part0
def test_additional_model_outputs_integration():
    """Integration test for additional_model_outputs.

    This test uses a dummy model to test the additional_model_outputs feature.
    """
    # Create sampling params with additional outputs
    additional_outputs = [
        AdditionalModelOutput(name="context_output", gather_context=True),
        AdditionalModelOutput(name="generation_output", gather_context=False),
    ]

    num_generated_tokens = 5

    sampling_params = SamplingParams(
        max_tokens=num_generated_tokens,
        temperature=0.0,  # Deterministic for testing
        end_id=-1,  # Use -1 to indicate that the end token is not used
        additional_model_outputs=additional_outputs)

    # Create LLM with the provided model
    llm = LLM(model=_pl.Path("dummy_path"),
              backend='pytorch',
              max_batch_size=2,
              max_seq_len=128,
              max_num_tokens=5,
              kv_cache_config=KvCacheConfig(enable_block_reuse=False),
              checkpoint_loader=HfCheckpointLoader(
                  weight_loader=DummyWeightLoader(),
                  config_loader=DummyConfigLoader()))

    # Test prompts
    prompts = [[1, 2, 3], [4, 5, 6]]
    num_prompts = len(prompts)
    prompt_lens = [len(prompt) for prompt in prompts]

    config = DummyConfig()
    max_beam_width = 1

    # Generate outputs
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    # Verify outputs
    assert len(outputs) == num_prompts
    for i in range(num_prompts):
        output = outputs[i]
        assert len(output.outputs) == 1
        sequence = output.outputs[0]

        # Check that additional outputs are present
        assert sequence.additional_context_outputs is not None
        assert sequence.additional_generation_outputs is not None

        # Check that the requested outputs are present
        assert "context_output" in sequence.additional_context_outputs
        assert "generation_output" in sequence.additional_generation_outputs

        # Verify tensor shapes are reasonable
        context_output = sequence.additional_context_outputs["context_output"]
        generation_output = sequence.additional_generation_outputs[
            "generation_output"]

        # Verify that the outputs are tensors
        assert isinstance(context_output, torch.Tensor)
        assert isinstance(generation_output, torch.Tensor)

        # Verify context output shape
        assert context_output.dim() == 2
        assert context_output.shape[0] == prompt_lens[i]
        assert context_output.shape[1] == config.hidden_size

        # Verify generation output shape
        assert generation_output.dim() == 3
        assert generation_output.shape[0] == num_generated_tokens
        assert generation_output.shape[1] == max_beam_width
        assert generation_output.shape[2] == config.hidden_size


if __name__ == "__main__":
    pytest.main([__file__])
