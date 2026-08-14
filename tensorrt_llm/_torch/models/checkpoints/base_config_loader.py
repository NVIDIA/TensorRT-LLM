from abc import ABC, abstractmethod

from tensorrt_llm._torch.model_config import ModelConfig


class BaseConfigLoader(ABC):

    @abstractmethod
    def load(self, checkpoint_dir: str, **kwargs) -> ModelConfig:
        pass

    def cleanup(self) -> None:
        pass
