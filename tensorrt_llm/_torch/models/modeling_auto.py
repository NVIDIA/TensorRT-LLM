from typing import Generic

from ..model_config import ModelConfig
from ..utils import model_extra_attrs
from .modeling_utils import (MODEL_CLASS_MAPPING,
                             MODEL_CLASS_VISION_ENCODER_MAPPING,
                             DecoderModelForCausalLM, TConfig, TModel)


class AutoModelForCausalLM(Generic[TModel, TConfig]):

    @staticmethod
    def from_config(
        config: ModelConfig[TConfig],
    ) -> DecoderModelForCausalLM[TModel, TConfig]:
        model_arch = config.pretrained_config.architectures[0]
        if config.mm_encoder_only:
            vision_encoder_info = MODEL_CLASS_VISION_ENCODER_MAPPING.get(
                model_arch)
            if vision_encoder_info is None:
                raise ValueError(
                    f"Unknown architecture for AutoModelForMultimodalEncoder: {model_arch}"
                )
            vision_encoder_cls, vlm_base_model = vision_encoder_info
            return vision_encoder_cls(config, vlm_base_model)

        # Hack to detect eagle3 checkpoints. TODO: should we provide
        # our own checkpoints with the correct arch? It would let us
        # avoid nasty stuff like this.
        model_arch = model_arch.replace("Eagle3",
                                        "")  # Strip the appended EAGLE3
        if hasattr(config.pretrained_config, "draft_vocab_size"):
            model_arch = "EAGLE3" + model_arch
        if model_arch == "DeepseekV3ForCausalLM" and config.spec_config is not None and config.spec_config.max_draft_len == 0:
            model_arch = "MTPDraftModelForCausalLM"

        cls = MODEL_CLASS_MAPPING.get(model_arch)
        if cls is None:
            raise ValueError(
                f"Unknown architecture for AutoModelForCausalLM: {config.pretrained_config.architectures[0]}"
            )
        if issubclass(cls, DecoderModelForCausalLM):
            config._frozen = False
            config.skip_create_weights_in_init = True
            config._frozen = True
        extra_attrs = {}
        with model_extra_attrs(extra_attrs):
            model = cls(config)
        model.extra_attrs = extra_attrs
        return model
