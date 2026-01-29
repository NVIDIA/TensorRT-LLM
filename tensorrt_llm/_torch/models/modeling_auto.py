from typing import Generic, Optional, Type

from ..model_config import ModelConfig
from ..utils import model_extra_attrs
from .modeling_utils import (MODEL_CLASS_MAPPING,
                             MODEL_CLASS_VISION_ENCODER_MAPPING,
                             DecoderModelForCausalLM, TConfig, TModel)


class AutoModelForCausalLM(Generic[TModel, TConfig]):

    @staticmethod
    def _resolve_class(config: ModelConfig) -> Optional[Type]:
        """Internal: Resolve the model class for a config without instantiating."""
        pretrained_config = config.pretrained_config
        if not pretrained_config.architectures:
            return None

        model_arch = pretrained_config.architectures[0]

        # Hack to detect eagle3 checkpoints. TODO: should we provide
        # our own checkpoints with the correct arch? It would let us
        # avoid nasty stuff like this.
        if hasattr(pretrained_config, "draft_vocab_size"):
            model_arch = model_arch.replace("Eagle3",
                                            "")  # Strip the appended EAGLE3
            model_arch = "EAGLE3" + model_arch

        if model_arch in (
                "DeepseekV3ForCausalLM", "Glm4MoeForCausalLM",
                "ExaoneMoEForCausalLM"
        ) and config.spec_config is not None and config.spec_config.max_draft_len == 0:
            model_arch = "MTPDraftModelForCausalLM"

        if config.mm_encoder_only:
            vision_encoder_info = MODEL_CLASS_VISION_ENCODER_MAPPING.get(
                model_arch)
            if vision_encoder_info is None:
                return None
            vision_encoder_cls, _ = vision_encoder_info
            return vision_encoder_cls

        return MODEL_CLASS_MAPPING.get(model_arch)

    @staticmethod
    def from_config(
        config: ModelConfig[TConfig],
    ) -> DecoderModelForCausalLM[TModel, TConfig]:
        if config.mm_encoder_only:
            model_arch = config.pretrained_config.architectures[0]
            vision_encoder_info = MODEL_CLASS_VISION_ENCODER_MAPPING.get(
                model_arch)
            if vision_encoder_info is None:
                raise ValueError(
                    f"Unknown architecture for AutoModelForMultimodalEncoder: {model_arch}"
                )
            vision_encoder_cls, vlm_base_model = vision_encoder_info
            return vision_encoder_cls(config, vlm_base_model)
        cls = AutoModelForCausalLM._resolve_class(config)
        if cls is None:
            raise ValueError(
                f"Unknown architecture for AutoModelForCausalLM: {config.pretrained_config.architectures[0]}"
            )
        if issubclass(cls, DecoderModelForCausalLM):
            config._frozen = False
            config.skip_create_weights_in_init = True
            config._frozen = True
        extra_attrs = config.extra_attrs
        with model_extra_attrs(extra_attrs):
            model = cls(config)
        model.extra_attrs = extra_attrs
        return model
