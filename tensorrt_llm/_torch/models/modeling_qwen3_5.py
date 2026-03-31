import re
from typing import Dict, List

import torch
from transformers import PretrainedConfig

from ...inputs import (
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    register_input_processor,
    support_multimodal_disaggregated,
)
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from .modeling_multimodal_utils import _is_disagg
from .modeling_qwen3_next import Qwen3NextForCausalLM
from .modeling_qwen3vl import (
    Qwen3VisionModel,
    Qwen3VisionModelBase,
    Qwen3VLInputProcessorBase,
    Qwen3VLModelBase,
)
from .modeling_utils import ModelConfig, register_auto_model, register_vision_encoder

_LANG_PREFIX = "model.language_model."


def _normalize_qwen35_exclude_modules(model_config):
    """Normalize NVFP4/FP8 exclude_modules from HF naming to TRT-LLM naming.

    hf_quant_config.json stores exclude patterns in HF checkpoint namespace
    (e.g. "model.language_model.layers.0.linear_attn*"), but TRT-LLM modules
    use "model.layers.0.linear_attn.in_proj_qkvz".  This function translates
    the patterns so that apply_quant_config_exclude_modules can match them.
    """
    qc = model_config.quant_config
    if qc is None or qc.exclude_modules is None:
        return

    normalized = set()
    for name in qc.exclude_modules:
        # Strip VLM prefix: model.language_model.X -> model.X
        if name.startswith(_LANG_PREFIX):
            name = "model." + name[len(_LANG_PREFIX) :]
        # Drop vision / MTP patterns
        if name.startswith("model.visual") or name.startswith("mtp."):
            continue
        # Map split projection names to packed TRT-LLM names
        name = re.sub(r"\.in_proj_[ab](\b|\*)", ".in_proj_ba*", name)
        name = re.sub(r"\.in_proj_(q|k|v|z|qkv)(\b|\*)", ".in_proj_qkvz*", name)
        normalized.add(name)

    qc.exclude_modules = sorted(normalized)


def _ensure_qwen35_mrope_compat(text_config: PretrainedConfig) -> None:
    """Normalize Qwen3.5 mRoPE fields for the shared Qwen3-VL wrapper.

    Qwen3.5 stores RoPE metadata in ``rope_parameters``.  Some config classes
    may also materialize default top-level ``rope_theta`` or
    ``partial_rotary_factor`` values, so prefer the checkpoint-provided nested
    values unconditionally here.
    """
    rope_parameters = getattr(text_config, "rope_parameters", None)
    if not rope_parameters:
        return

    rope_params = dict(rope_parameters)
    rope_theta = rope_params.pop("rope_theta", None)
    if rope_theta is not None:
        text_config.rope_theta = rope_theta

    partial_rotary_factor = rope_params.pop("partial_rotary_factor", None)
    if partial_rotary_factor is not None:
        text_config.partial_rotary_factor = partial_rotary_factor

    if not getattr(text_config, "rope_scaling", None):
        rope_params.pop("rope_type", None)
        text_config.rope_scaling = rope_params


@register_auto_model("Qwen3_5MoeForCausalLM")
class Qwen3_5MoeForCausalLM(Qwen3NextForCausalLM):
    """Thin wrapper that registers the Qwen3.5 MoE text architecture.

    Qwen3.5 text reuses the same model internals as Qwen3Next
    (Qwen3NextModel) -- the transformer, linear-attention layers, MoE blocks,
    and hybrid cache logic are all shared.  This separate class exists because:

    1. HF architecture routing: the HF checkpoint advertises
       Qwen3_5MoeForCausalLM (or top-level Qwen3_5MoeForConditionalGeneration
       with nested text_config), so TRT-LLM needs a matching
       @register_auto_model entry to route to the right class.
    2. Weight mapper dispatch: registering a distinct architecture name lets
       the checkpoint loader pick Qwen3_5MoeHfWeightMapper (which handles
       Qwen3.5-specific HF weight layout differences like split linear-attention
       projections and fused MoE expert tensors) instead of the base
       Qwen3NextHfWeightMapper.

    See Qwen3NextForCausalLM in modeling_qwen3_next.py for the equivalent
    class that serves the vanilla Qwen3NextForCausalLM architecture.
    """

    def __init__(self, model_config):
        _normalize_qwen35_exclude_modules(model_config)
        super().__init__(model_config)


@register_auto_model("Qwen3_5ForCausalLM")
class Qwen3_5ForCausalLM(Qwen3NextForCausalLM):
    """Thin wrapper for dense (non-MoE) Qwen3.5 text architecture.

    Same reuse pattern as Qwen3_5MoeForCausalLM, but for the dense 27B
    variant which uses GatedMLP instead of SparseMoeBlock.  The config
    normalizer (_Qwen35ConfigCompat) sets num_experts=0 so that
    Qwen3NextModel selects GatedMLP for the feed-forward layers.
    """

    def __init__(self, model_config):
        _normalize_qwen35_exclude_modules(model_config)
        super().__init__(model_config)


@support_multimodal_disaggregated
@register_vision_encoder(Qwen3VisionModelBase, vlm_base_model=Qwen3VisionModel)
@register_auto_model("Qwen3_5MoeForConditionalGeneration")
@register_input_processor(
    Qwen3VLInputProcessorBase,
    model_type="qwen3_5_moe",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|vision_start|><|image_pad|><|vision_end|>",
            "video": "<|vision_start|><|video_pad|><|vision_end|>",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        placeholders_separator="",
    ),
)
class Qwen3_5MoeVLModel(Qwen3VLModelBase):
    """VLM wrapper composing Qwen3 vision encoder with Qwen3.5 MoE text decoder."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        _ensure_qwen35_mrope_compat(model_config.pretrained_config.text_config)

        kwargs["vision_model_class"] = Qwen3VisionModel
        kwargs["disable_fuse_rope"] = kwargs.get("disable_fuse_rope", False)
        super().__init__(model_config, *args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values",
            "video.pixel_values_videos",
            "multimodal_embedding",
        ]

    def load_weights(self, weights: Dict[str, torch.Tensor], weight_mapper: BaseWeightMapper):
        if not _is_disagg():
            self.mm_encoder.load_weights(weights)

        weight_mapper = Qwen3_5MoeHfWeightMapper()
        weight_mapper.init_model_and_config(self.llm, self.model_config)
        filtered_weights = {k: v for k, v in weights.items() if not k.startswith("model.visual.")}
        params_map = {
            r"^model\.language_model\.(.*)$": r"model.\1",
        }
        self.llm.load_weights(filtered_weights, weight_mapper, params_map=params_map)
