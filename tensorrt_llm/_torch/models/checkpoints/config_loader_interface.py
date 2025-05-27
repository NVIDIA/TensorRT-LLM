from abc import ABC, abstractmethod

from tensorrt_llm._torch.model_config import ModelConfig


class ConfigLoaderInterface(ABC):

    @abstractmethod
    def load(self, checkpoint_dir: str, **kwargs) -> ModelConfig:
        pass
