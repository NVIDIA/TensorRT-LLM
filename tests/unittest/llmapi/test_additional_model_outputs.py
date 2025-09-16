import pathlib as _pl
from typing import Any, Optional

import pytest
import torch
from torch import nn

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import \
    BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.base_config_loader import \
    BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.models.modeling_dummy import DummyConfig
from tensorrt_llm._torch.models.modeling_utils import (
    register_checkpoint_loader, register_checkpoint_weight_loader,
    register_config_loader, register_mapper)
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

        # Implement your dummy weight loading logic here
        # Examples:
        # - Load from custom binary files
        # - Load from databases
        # - Load from compressed archives
        # - Apply custom preprocessing

        return weights


@register_mapper("DUMMY_FORMAT")
class DummyWeightMapper(BaseWeightMapper):

    def __init__(self):
        super().__init__()
        # Define any weight transformation callbacks
        self._callbacks = [
            # Add your dummy weight transformation functions
            # self._dummy_transform_function,
        ]

    def map_weights(self) -> None:
        """Define mappings between source and target weight names."""
        self.mapping.update({
            # Map source names to target names
            # 'target_module_name': ['source_param1', 'source_param2'],
            # Example: 'qkv_proj': ['q_proj', 'k_proj', 'v_proj']
        })

    def apply_callbacks(self, module: nn.Module, module_name: str,
                        module_names_breakdown: list[str],
                        weights: dict) -> list[dict]:
        """Apply weight transformations for modules that require special handling.

        Args:
            module: The target module
            module_name: The specific module name being processed
            module_names_breakdown: Module path components
            weights: Source weights dictionary
        Returns:
            List of transformed weight dictionaries
        """
        module_weights = []

        for new_name in self._mapping[module_name]:
            # Filter weights for this specific parameter
            fw = self.filter_weights(
                '.'.join(module_names_breakdown + [new_name]), weights)

            # Apply transformation callbacks
            for callback in self._callbacks:
                fw = callback(module, new_name, fw)

            module_weights.append(fw)

        return module_weights

    def should_skip_module(self, module_name: str) -> bool:
        """Define which modules should be skipped during loading."""
        # Add logic to skip specific modules based on your requirements
        # Examples:
        # - Skip LoRA-specific modules
        # - Skip temporary/auxiliary modules

        return super().should_skip_module(module_name)


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
        pretrained_config = self._load_pretrained_config(
            checkpoint_dir, **kwargs)

        return ModelConfig(pretrained_config=pretrained_config,
                           # Add other ModelConfig parameters as needed
                           )

    def _load_pretrained_config(self, checkpoint_dir: str, **kwargs):
        """Load the raw configuration from your dummy format."""
        return DummyConfig()


@register_checkpoint_loader("DUMMY_FORMAT")
class DummyCheckpointLoader(BaseCheckpointLoader):

    def __init__(self):
        self._weight_loader = DummyWeightLoader()
        self._config_loader = DummyConfigLoader()
        self._weight_mapper = DummyWeightMapper()
        self._checkpoint_format = "DUMMY_FORMAT"

    def get_default_weight_loader(self) -> BaseWeightLoader:
        return DummyWeightLoader()

    def get_default_config_loader(self) -> BaseConfigLoader:
        return DummyConfigLoader()

    def cleanup(self) -> None:
        pass

    @property
    def weight_loader(self) -> BaseWeightLoader:
        return self._weight_loader

    @property
    def weight_mapper(self) -> Optional[BaseWeightMapper]:
        return self._weight_mapper

    @property
    def config_loader(self) -> Optional[BaseConfigLoader]:
        return self._config_loader

    @property
    def checkpoint_format(self) -> str:
        return self._checkpoint_format


@pytest.mark.part0
def test_additional_model_outputs_integration():
    """Real integration test for additional_model_outputs.

    This test can be run when a model is available by setting the LLM_MODEL_PATH
    environment variable to point to a valid model directory.
    """
    # Create sampling params with additional outputs
    additional_outputs = [
        AdditionalModelOutput(name="context_output", gather_context=True),
        AdditionalModelOutput(name="generation_output", gather_context=False),
    ]

    sampling_params = SamplingParams(
        max_tokens=5,
        temperature=0.0,  # Deterministic for testing
        end_id=-1,  # Use -1 to indicate that the end token is not used
        additional_model_outputs=additional_outputs)

    # Create LLM with the provided model
    llm = LLM(model=_pl.Path("dummy_path"),
              backend='pytorch',
              max_batch_size=2,
              max_seq_len=128,
              kv_cache_config=KvCacheConfig(enable_block_reuse=False),
              checkpoint_loader=DummyCheckpointLoader())

    # Test prompts
    prompts = [[1, 2, 3]]

    # Generate outputs
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    # Verify outputs
    assert len(outputs) == 1
    output = outputs[0]
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

    assert context_output.dim(
    ) >= 2  # Should have at least [seq_len, hidden_size]
    assert generation_output.dim(
    ) >= 2  # Should have at least [batch, hidden_size]

    # Verify that the outputs are tensors
    assert isinstance(context_output, torch.Tensor)
    assert isinstance(generation_output, torch.Tensor)

    print(
        f"Integration test passed! context_output shape: {context_output.shape}"
    )
    print(
        f"Integration test passed! generation_output shape: {generation_output.shape}"
    )


if __name__ == "__main__":
    pytest.main([__file__])
