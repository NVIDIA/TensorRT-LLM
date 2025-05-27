from typing import Optional

from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import \
    BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.config_loader_interface import \
    ConfigLoaderInterface
from tensorrt_llm._torch.models.checkpoints.hf.config_loader import \
    HfConfigLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import \
    HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.weight_loader_interface import \
    WeightLoaderInterface
from tensorrt_llm._torch.models.checkpoints.weight_mapper_interface import \
    WeightMapperInterface
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_loader


@register_checkpoint_loader("HF")
class HfCheckpointLoader(BaseCheckpointLoader):

    def __init__(self,
                 weight_loader: Optional[WeightLoaderInterface] = None,
                 weight_mapper: Optional[WeightMapperInterface] = None,
                 config_loader: Optional[ConfigLoaderInterface] = None):
        self._weight_loader = weight_loader
        self._weight_mapper = weight_mapper
        self._config_loader = config_loader
        self._checkpoint_format = "HF"

    def ensure_fully_initialized(self) -> None:
        if self.weight_loader is None:
            self.weight_loader = self.get_default_weight_loader()
        if self.config_loader is None:
            self.config_loader = self.get_default_config_loader()

    def get_default_weight_loader(self) -> WeightLoaderInterface:
        return HfWeightLoader()

    def get_default_config_loader(self) -> ConfigLoaderInterface:
        return HfConfigLoader()

    @property
    def weight_loader(self) -> Optional[WeightLoaderInterface]:
        return self._weight_loader

    @weight_loader.setter
    def weight_loader(self, value: WeightLoaderInterface) -> None:
        self._weight_loader = value

    @property
    def weight_mapper(self) -> Optional[WeightMapperInterface]:
        return self._weight_mapper

    @weight_mapper.setter
    def weight_mapper(self, value: WeightMapperInterface) -> None:
        self._weight_mapper = value

    @property
    def config_loader(self) -> Optional[ConfigLoaderInterface]:
        return self._config_loader

    @config_loader.setter
    def config_loader(self, value: ConfigLoaderInterface) -> None:
        self._config_loader = value

    @property
    def checkpoint_format(self) -> str:
        return self._checkpoint_format
