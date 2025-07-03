from abc import ABC, abstractmethod
from typing import Any

from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.config_loader_interface import \
    ConfigLoaderInterface
from tensorrt_llm._torch.models.checkpoints.mapper_auto import \
    CheckpointMapperAuto
from tensorrt_llm._torch.models.checkpoints.weight_loader_interface import \
    WeightLoaderInterface
from tensorrt_llm._torch.models.checkpoints.weight_mapper_interface import \
    WeightMapperInterface
from tensorrt_llm._torch.models.modeling_utils import \
    CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING


class BaseCheckpointLoader(ABC):

    @abstractmethod
    def ensure_fully_initialized(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_default_weight_loader(self) -> WeightLoaderInterface:
        raise NotImplementedError

    @abstractmethod
    def get_default_config_loader(self) -> ConfigLoaderInterface:
        raise NotImplementedError

    @property
    @abstractmethod
    def weight_loader(self) -> WeightLoaderInterface:
        ...

    @weight_loader.setter
    @abstractmethod
    def weight_loader(self, value: WeightLoaderInterface) -> None:
        ...

    @property
    @abstractmethod
    def weight_mapper(self) -> WeightMapperInterface:
        ...

    @weight_mapper.setter
    @abstractmethod
    def weight_mapper(self, value: WeightMapperInterface) -> None:
        ...

    @property
    @abstractmethod
    def config_loader(self) -> ConfigLoaderInterface:
        ...

    @config_loader.setter
    @abstractmethod
    def config_loader(self, value: ConfigLoaderInterface) -> None:
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
                f"Checkpoint loader for format {checkpoint_format} not found")

    def get_initilized_weight_mapper(
            self, model: nn.Module,
            config: ModelConfig) -> WeightMapperInterface:
        weight_mapper = None

        if self.weight_mapper is not None:
            self.weight_mapper.init(model, config)
            return self.weight_mapper
        else:
            # The name of the registered mapper should be the model architecture
            if config.pretrained_config and config.pretrained_config.architectures:
                model_arch = config.pretrained_config.architectures[0]
            else:
                raise ValueError(
                    "Cannot determine model architecture from config")
            weight_mapper = CheckpointMapperAuto.get(self.checkpoint_format,
                                                     model_arch)
            weight_mapper.init(model, config)
            self.weight_mapper = weight_mapper
            return weight_mapper
