from typing import Generic

from ..model_config import ModelConfig
from .modeling_utils import (MODEL_CLASS_MAPPING, DecoderModelForCausalLM,
                             TConfig, TModel)


class AutoModelForCausalLM(Generic[TModel, TConfig]):

    @staticmethod
    def from_config(
        config: ModelConfig[TConfig],
    ) -> DecoderModelForCausalLM[TModel, TConfig]:
        cls = MODEL_CLASS_MAPPING.get(config.pretrained_config.architectures[0])
        if cls is None:
            raise ValueError(
                f"Unknown architecture for AutoModelForCausalLM: {config.pretrained_config.architectures[0]}"
            )
        return cls(config)
