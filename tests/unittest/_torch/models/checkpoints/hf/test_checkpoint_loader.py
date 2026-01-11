import pathlib as _pl
from typing import Any, Optional

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm import LLM, Mapping
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints import HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.base_config_loader import BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_auto_model


class DummyConfig(PretrainedConfig):
    def __init__(self):
        self.architectures: list[str] = ["DummyModel"]
        self.dtype: torch.dtype = torch.float16
        self.num_attention_heads: int = 16
        self.hidden_size: int = 256
        self.vocab_size: int = 1000
        self.num_hidden_layers: int = 1


@register_auto_model("DummyModel")
class DummyModel(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

    def infer_max_seq_len(self):
        return 2048

    @property
    def config(self):
        return self.model_config.pretrained_config

    def forward(self, *args, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        num_batch_tokens = input_ids.size(0)
        vocab_size = self.config.vocab_size

        # Logits: dummy values for testing
        logits = torch.ones((num_batch_tokens, vocab_size), device="cuda") * 0.1

        return {
            "logits": logits,
        }

    def load_weights(
        self,
        weights: dict,
        weight_mapper: Optional[BaseWeightMapper] = None,
        skip_modules: list[str] = [],
    ):
        pass


class DummyWeightLoader(BaseWeightLoader):
    def load_weights(self, checkpoint_dir: str, mapping: Mapping, **kwargs) -> dict[str, Any]:
        """Load weights from your dummy format.

        Args:
            checkpoint_dir: Directory containing checkpoint files
            mapping: Mapping object containing the distributed configuration
            **kwargs: Additional loading parameters
        Returns:
            Dictionary mapping parameter names to tensors
        """

        assert mapping is not None
        assert isinstance(mapping, Mapping)
        assert mapping.world_size == 1
        assert mapping.rank == 0

        weights = {}

        return weights


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


def test_weight_loader_mapping():
    """Test that the mapping in weight loader is correct."""

    # Create LLM with the provided model
    with LLM(
        model=_pl.Path("dummy_path"),
        backend="pytorch",
        cuda_graph_config=None,
        checkpoint_loader=HfCheckpointLoader(
            weight_loader=DummyWeightLoader(), config_loader=DummyConfigLoader()
        ),
    ):
        pass


if __name__ == "__main__":
    pytest.main([__file__])
