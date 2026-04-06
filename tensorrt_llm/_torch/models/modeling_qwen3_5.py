import re

from .modeling_qwen3_next import Qwen3NextForCausalLM
from .modeling_utils import register_auto_model

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
