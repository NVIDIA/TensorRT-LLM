from abc import ABC, abstractmethod
from typing import Any

from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.auto_mapper import \
    AutoCheckpointMapper
from tensorrt_llm._torch.models.checkpoints.base_config_loader import \
    BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.models.modeling_utils import \
    CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING


class BaseCheckpointLoader(ABC):

    @abstractmethod
    def get_default_weight_loader(self) -> BaseWeightLoader:
        raise NotImplementedError

    @abstractmethod
    def get_default_config_loader(self) -> BaseConfigLoader:
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def weight_loader(self) -> BaseWeightLoader:
        ...

    @property
    @abstractmethod
    def weight_mapper(self) -> BaseWeightMapper:
        ...

    @property
    @abstractmethod
    def config_loader(self) -> BaseConfigLoader:
        ...

    @property
    @abstractmethod
    def checkpoint_format(self) -> str:
        ...

    def load_config(self, checkpoint_dir: str, **kwargs) -> ModelConfig:
        return self.config_loader.load(checkpoint_dir, **kwargs)

    def load_weights(self, checkpoint_dir: str, **kwargs) -> dict[str, Any]:
        return self.weight_loader.load_weights(checkpoint_dir, **kwargs)

    @classmethod
    def get(cls, checkpoint_format: str, **kwargs) -> "BaseCheckpointLoader":
        try:
            return CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING[checkpoint_format](
                **kwargs)
        except KeyError:
            raise ValueError(
                f"Checkpoint loader for format {checkpoint_format} not found, "
                f"available formats are: {CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING.keys()}"
            )

    def get_initialized_weight_mapper(self, model: nn.Module,
                                      config: ModelConfig) -> BaseWeightMapper:
        weight_mapper = None
        if self.weight_mapper is not None:
            self.weight_mapper.init_model_and_config(model, config)
            return self.weight_mapper
        else:
            # The name of the registered mapper should be the model architecture
            if config.pretrained_config and config.pretrained_config.architectures:
                model_arch = config.pretrained_config.architectures[0]
            else:
                raise ValueError(
                    "Cannot determine model architecture from config")
            weight_mapper = AutoCheckpointMapper.get(self.checkpoint_format,
                                                     model_arch)
            weight_mapper.init_model_and_config(model, config)
            self.weight_mapper = weight_mapper
            return weight_mapper
