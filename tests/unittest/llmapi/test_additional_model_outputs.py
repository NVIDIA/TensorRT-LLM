import pathlib as _pl
from typing import Any, Dict, List, Optional

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints import HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.base_config_loader import \
    BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.models.modeling_utils import (
    ModelConfig, register_auto_model, register_checkpoint_weight_loader,
    register_config_loader)
from tensorrt_llm.llmapi import KvCacheConfig


class DummyConfig(PretrainedConfig):

    def __init__(self):
        self.architectures: list[str] = ["DummyModel"]
        self.torch_dtype: torch.dtype = torch.float16
        self.num_key_value_heads: int = 16
        self.num_attention_heads: int = 16
        self.hidden_size: int = 256
        self.vocab_size: int = 1000
        self.num_hidden_layers: int = 1

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@register_auto_model("DummyModel")
class DummyModel(torch.nn.Module):

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.dtype = model_config.pretrained_config.torch_dtype
        self.model_config = model_config
        self.recorded_position_ids = None

    def infer_max_seq_len(self):
        return 2048

    @property
    def config(self):
        return self.model_config.pretrained_config

    def forward(self,
                *args,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                attn_metadata: AttentionMetadata,
                return_context_logits: bool = False,
                **kwargs) -> torch.Tensor:
        num_batch_tokens = input_ids.size(0)

        vocab_size = self.config.vocab_size
        hidden_size = self.config.hidden_size

        last_tokens = torch.cumsum(
            attn_metadata.seq_lens_cuda,
            dim=0,
            dtype=torch.long,
        ) - 1

        # Logits: fixed values for testing
        logits = torch.ones((num_batch_tokens, vocab_size), device='cuda') * 0.1

        # Logits shape depends on return_context_logits flag
        if not return_context_logits:
            # For context logits, return logits for all positions
            logits = logits[last_tokens]

        # Context output: values depend on position for testing, one output per input token
        context_output = (
            position_ids.reshape(-1, 1).expand(num_batch_tokens, hidden_size) *
            0.2)

        # Generation output: values depend on position for testing, one output per sequence
        generation_output = (
            position_ids.reshape(-1, 1).expand(num_batch_tokens, hidden_size) *
            0.3)
        generation_output = generation_output[last_tokens]

        return {
            "logits": logits,
            "context_output": context_output,
            "generation_output": generation_output
        }

    def load_weights(self,
                     weights: Dict,
                     weight_mapper: Optional[BaseWeightMapper] = None,
                     skip_modules: List[str] = []):
        pass


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
def test_additional_model_outputs_sampling_params():
    """Test that additional_model_outputs can be configured in SamplingParams."""
    # Create sampling params with additional outputs
    additional_outputs = ["context_output", "generation_output"]

    sampling_params = SamplingParams(
        max_tokens=5,
        temperature=0.0,  # Deterministic for testing
        additional_model_outputs=additional_outputs)

    # Verify the sampling params are configured correctly
    assert sampling_params.additional_model_outputs is not None
    assert len(sampling_params.additional_model_outputs) == 2
    assert sampling_params.additional_model_outputs[0] == "context_output"
    assert sampling_params.additional_model_outputs[1] == "generation_output"


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


@pytest.mark.part0
def test_additional_model_outputs_integration():
    """Integration test for additional_model_outputs.

    This test uses a dummy model to test the additional_model_outputs feature.
    """
    # Create sampling params with additional outputs
    additional_outputs = ["context_output", "generation_output"]

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
              disable_overlap_scheduler=True,
              checkpoint_loader=HfCheckpointLoader(
                  weight_loader=DummyWeightLoader(),
                  config_loader=DummyConfigLoader()))

    # Test prompts
    prompts = [[1, 2, 3], [4, 5, 6]]
    num_prompts = len(prompts)
    prompt_lens = [len(prompt) for prompt in prompts]

    config = DummyConfig()
    max_beam_width = 1

    with llm:
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
            context_output = sequence.additional_context_outputs[
                "context_output"]
            context_generation_output = sequence.additional_generation_outputs[
                "context_output"]
            generation_output = sequence.additional_generation_outputs[
                "generation_output"]

            # Verify that the outputs are tensors
            assert isinstance(context_output, torch.Tensor)
            assert isinstance(context_generation_output, torch.Tensor)
            assert isinstance(generation_output, torch.Tensor)

            # Verify context output shape
            assert context_output.dim() == 2
            assert context_output.shape[0] == prompt_lens[i]
            assert context_output.shape[1] == config.hidden_size

            expected_context_output = torch.arange(
                prompt_lens[i], dtype=torch.float32).unsqueeze(1).expand(
                    prompt_lens[i], config.hidden_size) * 0.2

            assert torch.equal(context_output, expected_context_output)

            # Verify context generation output shape
            assert context_generation_output.dim() == 3
            assert context_generation_output.shape[0] == num_generated_tokens
            assert context_generation_output.shape[1] == max_beam_width
            assert context_generation_output.shape[2] == config.hidden_size

            gen_start_idx = prompt_lens[i] - 1
            gen_end_idx = gen_start_idx + num_generated_tokens

            expected_context_generation_output = torch.arange(
                gen_start_idx, gen_end_idx,
                dtype=torch.float32).unsqueeze(1).expand(
                    num_generated_tokens, config.hidden_size) * 0.2

            assert torch.equal(context_generation_output,
                               expected_context_generation_output.unsqueeze(1))

            # Verify generation output shape
            assert generation_output.dim() == 3
            assert generation_output.shape[0] == num_generated_tokens
            assert generation_output.shape[1] == max_beam_width
            assert generation_output.shape[2] == config.hidden_size

            expected_generation_output = torch.arange(
                gen_start_idx, gen_end_idx,
                dtype=torch.float32).unsqueeze(1).expand(
                    num_generated_tokens, config.hidden_size) * 0.3

            assert torch.equal(generation_output,
                               expected_generation_output.unsqueeze(1))


if __name__ == "__main__":
    pytest.main([__file__])
