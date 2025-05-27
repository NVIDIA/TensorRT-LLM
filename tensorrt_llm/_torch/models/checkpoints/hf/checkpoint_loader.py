from typing import Optional

from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import \
    BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.base_config_loader import \
    BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.config_loader import \
    HfConfigLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import \
    HfWeightLoader
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_loader


@register_checkpoint_loader("HF")
class HfCheckpointLoader(BaseCheckpointLoader):

    def __init__(self,
                 weight_loader: Optional[BaseWeightLoader] = None,
                 weight_mapper: Optional[BaseWeightMapper] = None,
                 config_loader: Optional[BaseConfigLoader] = None):
        self._weight_loader = weight_loader
        self._weight_mapper = weight_mapper
        self._config_loader = config_loader
        self._checkpoint_format = "HF"

    def ensure_fully_initialized(self) -> None:
        if self.weight_loader is None:
            self.weight_loader = self.get_default_weight_loader()
        if self.config_loader is None:
            self.config_loader = self.get_default_config_loader()

    def cleanup(self) -> None:
        # Clean up weight mapper first as it may hold model references
        if self._weight_mapper is not None:
            if hasattr(self._weight_mapper, 'cleanup'):
                self._weight_mapper.cleanup()
            self._weight_mapper = None

        if self._weight_loader is not None:
            if hasattr(self._weight_loader, 'cleanup'):
                self._weight_loader.cleanup()
            self._weight_loader = None

        if self._config_loader is not None:
            if hasattr(self._config_loader, 'cleanup'):
                self._config_loader.cleanup()
            self._config_loader = None

    def get_default_weight_loader(self) -> BaseWeightLoader:
        return HfWeightLoader()

    def get_default_config_loader(self) -> BaseConfigLoader:
        return HfConfigLoader()

    @property
    def weight_loader(self) -> Optional[BaseWeightLoader]:
        return self._weight_loader

    @weight_loader.setter
    def weight_loader(self, value: BaseWeightLoader) -> None:
        self._weight_loader = value

    @property
    def weight_mapper(self) -> Optional[BaseWeightMapper]:
        return self._weight_mapper

    @weight_mapper.setter
    def weight_mapper(self, value: BaseWeightMapper) -> None:
        self._weight_mapper = value

    @property
    def config_loader(self) -> Optional[BaseConfigLoader]:
        return self._config_loader

    @config_loader.setter
    def config_loader(self, value: BaseConfigLoader) -> None:
        self._config_loader = value

    @property
    def checkpoint_format(self) -> str:
        return self._checkpoint_format
