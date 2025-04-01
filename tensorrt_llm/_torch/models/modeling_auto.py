from typing import Generic

from ..model_config import ModelConfig
from .modeling_utils import (MODEL_CLASS_MAPPING, DecoderModelForCausalLM,
                             TConfig, TModel)


class AutoModelForCausalLM(Generic[TModel, TConfig]):

    @staticmethod
    def from_config(
        config: ModelConfig[TConfig],
    ) -> DecoderModelForCausalLM[TModel, TConfig]:
        model_arch = config.pretrained_config.architectures[0]
        # Hack to detect eagle3 checkpoints. TODO: should we provide
        # our own checkpoints with the correct arch? It would let us
        # avoid nasty stuff like this.
        if hasattr(config.pretrained_config, "draft_vocab_size"):
            model_arch = "EAGLE3" + model_arch

        cls = MODEL_CLASS_MAPPING.get(model_arch)
        if cls is None:
            raise ValueError(
                f"Unknown architecture for AutoModelForCausalLM: {config.pretrained_config.architectures[0]}"
            )
        if issubclass(cls, DecoderModelForCausalLM):
            config.skip_create_weights = True
        return cls(config)
