import re

from .modeling_qwen3_next import Qwen3NextForCausalLM
from .modeling_utils import register_auto_model

_LANG_PREFIX = "model.language_model."


_MTP_TOP_TO_TRTLLM = {
    "fc": "fc",
    "norm": "shared_head.norm",
    "pre_fc_norm_embedding": "pre_fc_norm_embedding",
    "pre_fc_norm_hidden": "pre_fc_norm_hidden",
}


def _translate_mtp_pattern(name, n_hidden_layers):
    """Translate an HF ``mtp.*`` exclude pattern to a TRT-LLM module path.

    HF checkpoints store the MTP layer under ``mtp.layers.<idx>.*`` (and a
    few top-level entries like ``mtp.fc``, ``mtp.norm``,
    ``mtp.pre_fc_norm_*``).  Qwen3NextForCausalLM extends ``self.model.layers``
    with the MTP layers, so at runtime they live at
    ``model.layers.<n_hidden_layers + idx>.*``.  Returns ``None`` if the
    pattern can't be translated (e.g. ``mtp.<unknown>``).
    """
    if not name.startswith("mtp."):
        return None
    if name.startswith("mtp.layers."):
        tail = name[len("mtp.layers.") :]
        idx_str, sep, rest = tail.partition(".")
        idx_clean = idx_str.rstrip("*")
        idx_wildcard = "*" if idx_str.endswith("*") and not sep else ""
        try:
            layer_idx = int(idx_clean) if idx_clean else 0
        except ValueError:
            return None
        new_layer = n_hidden_layers + layer_idx
        if sep:
            return f"model.layers.{new_layer}.{rest}"
        return f"model.layers.{new_layer}{idx_wildcard}"

    suffix = name[len("mtp.") :]
    for hf_top, trtllm_top in _MTP_TOP_TO_TRTLLM.items():
        if suffix == hf_top:
            return f"model.layers.{n_hidden_layers}.{trtllm_top}"
        if suffix == hf_top + "*":
            return f"model.layers.{n_hidden_layers}.{trtllm_top}*"
        if suffix.startswith(hf_top + "."):
            return f"model.layers.{n_hidden_layers}.{trtllm_top}.{suffix[len(hf_top) + 1 :]}"
    return None


def _normalize_qwen35_exclude_modules(model_config):
    """Normalize NVFP4/FP8 exclude_modules from HF naming to TRT-LLM naming.

    hf_quant_config.json stores exclude patterns in HF checkpoint namespace
    (e.g. ``model.language_model.layers.0.linear_attn*`` and ``mtp.layers.0*``),
    but TRT-LLM modules use ``model.layers.0.linear_attn.in_proj_qkvz`` and
    map the MTP layer to ``model.layers.<num_hidden_layers>.*``.  This
    function translates the patterns so that
    ``apply_quant_config_exclude_modules`` can match them.
    """
    qc = model_config.quant_config
    if qc is None or qc.exclude_modules is None:
        return

    n_hidden_layers = getattr(model_config.pretrained_config, "num_hidden_layers", None)

    normalized = set()
    for name in qc.exclude_modules:
        # Strip VLM prefix: model.language_model.X -> model.X
        if name.startswith(_LANG_PREFIX):
            name = "model." + name[len(_LANG_PREFIX) :]
        # Drop vision tensors (not part of the language model graph)
        if name.startswith("model.visual"):
            continue
        # Translate MTP-namespace patterns to TRT-LLM paths so the MTP
        # layer (which the checkpoint stores unquantized) gets correctly
        # excluded from the global NVFP4/FP8 quant_config.
        if name.startswith("mtp."):
            if n_hidden_layers is None:
                continue
            translated = _translate_mtp_pattern(name, n_hidden_layers)
            if translated is not None:
                normalized.add(translated)
            continue
        # Map split projection names to packed TRT-LLM names
        name = re.sub(r"\.in_proj_[ab](\b|\*)", ".in_proj_ba*", name)
        name = re.sub(r"\.in_proj_(q|k|v|z|qkv)(\b|\*)", ".in_proj_qkvz*", name)
        normalized.add(name)

    # gdn_mixer uses Linear module for weight management of depthwise conv1d
    # but conv1d is not a proper linear module and should be excluded from quant
    normalized.add("*linear_attn.conv1d")

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
