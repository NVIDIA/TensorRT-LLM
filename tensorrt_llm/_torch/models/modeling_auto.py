from typing import Generic

from ..model_config import ModelConfig
from ..utils import model_extra_attrs
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
        if issubclass(cls, DecoderModelForCausalLM):
            config.skip_create_weights = True
        extra_attrs = {}
        with model_extra_attrs(extra_attrs):
            model = cls(config)
        model.extra_attrs = extra_attrs
        return model
