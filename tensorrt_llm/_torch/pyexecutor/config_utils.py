import re
from types import SimpleNamespace
from typing import Optional

import transformers


def is_nemotron_hybrid(config):
    if hasattr(config, "hybrid_override_pattern"
               ) and config.hybrid_override_pattern is not None and len(
                   config.hybrid_override_pattern) > 0:
        return True
    return False


def is_mla(config):
    if getattr(config, "kv_lora_rank", None) and getattr(
            config, "qk_rope_head_dim", None):
        return True
    return False


def is_qwen3_next(config):
    return hasattr(
        config, 'architectures'
    ) and config.architectures is not None and config.architectures[
        0] == 'Qwen3NextForCausalLM'


def is_qwen3_5(config):
    """True when config was loaded for a Qwen3.5 text checkpoint (MoE or dense)."""
    _QWEN3_5_ARCHS = {
        'Qwen3_5MoeForCausalLM',
        'Qwen3_5ForCausalLM',
    }
    return hasattr(
        config, 'architectures'
    ) and config.architectures is not None and config.architectures[
        0] in _QWEN3_5_ARCHS


def is_qwen3_hybrid(config):
    """True for any Qwen3 hybrid (linear-attention + full-attention) model.

    Both Qwen3Next and Qwen3.5 use the same hybrid layer layout and share
    cache-manager routing (MambaHybridCacheManager) and layer-type derivation.
    """
    return is_qwen3_next(config) or is_qwen3_5(config)


def get_qwen3_hybrid_layer_types(config):
    """Return per-layer type list for a Qwen3 hybrid model.

    Accepts either an explicit layer_types list from the config, or derives
    one from full_attention_interval (every Nth layer is full attention,
    all others are linear attention).
    """
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        if len(layer_types) != config.num_hidden_layers:
            raise ValueError(
                f"Expected {config.num_hidden_layers} layer types for "
                f"{config.architectures[0]}, got {len(layer_types)}")
        for layer_type in layer_types:
            if layer_type not in {"full_attention", "linear_attention"}:
                raise ValueError(
                    f"Unsupported Qwen3 hybrid layer type: {layer_type}")
        return layer_types

    full_attention_interval = getattr(config, "full_attention_interval", None)
    if not isinstance(full_attention_interval,
                      int) or full_attention_interval <= 0:
        raise ValueError(
            "Qwen3 hybrid configs must define either layer_types or a valid "
            "full_attention_interval")

    return [
        "linear_attention" if i %
        full_attention_interval != full_attention_interval -
        1 else "full_attention" for i in range(config.num_hidden_layers)
    ]


def get_qwen3_hybrid_layer_masks(config):
    layer_types = get_qwen3_hybrid_layer_types(config)
    layer_mask = [layer_type == "full_attention" for layer_type in layer_types]
    mamba_layer_mask = [
        layer_type == "linear_attention" for layer_type in layer_types
    ]
    return layer_mask, mamba_layer_mask


def get_qwen3_hybrid_num_attention_layers(config):
    layer_mask, _ = get_qwen3_hybrid_layer_masks(config)
    return sum(layer_mask)


class _Qwen35ConfigCompat:
    """Temporary shim that normalizes Qwen3.5 HF configs into Qwen3NextConfig.

    To remove: delete this class and the elif branch in
    load_pretrained_config that references it.
    """

    @staticmethod
    def normalize(config_dict: dict) -> dict:
        """Entry point: raw config.json dict -> flat Qwen3NextConfig-compatible dict."""
        text_config = _Qwen35ConfigCompat._extract_text_config(config_dict)
        text_config = _Qwen35ConfigCompat._inherit_quantization_config(
            config_dict, text_config)
        text_config = _Qwen35ConfigCompat._flatten_rope(text_config)

        # Detect dense vs MoE and set architecture + MoE defaults accordingly
        is_moe = "num_experts" in text_config and text_config["num_experts"] > 0
        if is_moe:
            text_config["architectures"] = ["Qwen3_5MoeForCausalLM"]
        else:
            text_config["architectures"] = ["Qwen3_5ForCausalLM"]
            # Ensure MoE fields are zeroed so Qwen3NextConfig defaults don't
            # accidentally enable MoE for the dense model.
            text_config.setdefault("num_experts", 0)
            text_config.setdefault("num_experts_per_tok", 0)
            text_config.setdefault("moe_intermediate_size", 0)
            text_config.setdefault("shared_expert_intermediate_size", 0)
        return text_config

    _VLM_ARCHITECTURES = {
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
    }

    @staticmethod
    def _extract_text_config(config_dict: dict) -> dict:
        """Pull nested text_config from VLM checkpoints, or use dict as-is."""
        architectures = config_dict.get("architectures") or []
        if architectures and architectures[
                0] in _Qwen35ConfigCompat._VLM_ARCHITECTURES:
            text_config = dict(config_dict.get("text_config") or {})
        else:
            text_config = dict(config_dict)
        if not text_config:
            raise ValueError("Qwen3.5 config is missing a usable text_config")
        return text_config

    @staticmethod
    def _inherit_quantization_config(config_dict: dict,
                                     text_config: dict) -> dict:
        """Copy top-level quantization_config into text_config with name normalization.

        Also adds a temporary workaround that keeps packed linear-attention
        in_proj_qkvz on the bf16 path until FP8 block-scale TP loading is
        fixed for that layout.
        """
        if "quantization_config" in text_config:
            return text_config
        if "quantization_config" not in config_dict:
            return text_config

        quantization_config = dict(config_dict["quantization_config"])
        if "modules_to_not_convert" in quantization_config:
            modules = _Qwen35ConfigCompat._normalize_exclude_modules(
                quantization_config["modules_to_not_convert"])
            modules = _Qwen35ConfigCompat._add_qkvz_bf16_workaround(
                text_config, modules)
            quantization_config["modules_to_not_convert"] = sorted(set(modules))
        text_config["quantization_config"] = quantization_config
        return text_config

    @staticmethod
    def _normalize_exclude_modules(modules: list[str]) -> list[str]:
        """Translate HF quantization exclude-module paths to TRT-LLM names.

        - Strip model.language_model. prefix -> model.
        - Drop model.visual.* and mtp.* entries
        - Map split projection names to packed TRT-LLM names
        """
        normalized = set()
        for name in modules:
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model."):]
            if name.startswith("model.visual.") or name.startswith("mtp."):
                continue
            name = re.sub(r"\.in_proj_[ab]$", ".in_proj_ba", name)
            name = re.sub(r"\.in_proj_(q|k|v|z|qkv)$", ".in_proj_qkvz", name)
            normalized.add(name)
        return sorted(normalized)

    @staticmethod
    def _add_qkvz_bf16_workaround(text_config: dict,
                                  modules: list[str]) -> list[str]:
        """Keep packed linear-attention qkvz on bf16 path for all linear-attention layers.

        Temporary until FP8 block-scale TP loading is fixed for this layout.
        """
        try:
            layer_types = get_qwen3_hybrid_layer_types(
                SimpleNamespace(**text_config))
        except (ValueError, AttributeError):
            return modules
        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type == "linear_attention":
                modules.append(
                    f"model.layers.{layer_idx}.linear_attn.in_proj_qkvz")
        return modules

    @staticmethod
    def _flatten_rope(text_config: dict) -> dict:
        """Flatten rope_parameters into top-level rope_theta / partial_rotary_factor / rope_scaling.

        Qwen3.5 nests these inside a rope_parameters dict and uses rope_type
        instead of type in rope_scaling.  Qwen3NextConfig expects them as
        top-level fields with rope_scaling.type.
        """
        rope_parameters = dict(text_config.pop("rope_parameters", {}) or {})
        rope_scaling = dict(text_config.get("rope_scaling") or {})
        if rope_parameters:
            rope_theta = rope_parameters.pop("rope_theta", None)
            if rope_theta is not None:
                text_config.setdefault("rope_theta", rope_theta)
            partial_rotary_factor = rope_parameters.pop("partial_rotary_factor",
                                                        None)
            if partial_rotary_factor is not None:
                text_config.setdefault("partial_rotary_factor",
                                       partial_rotary_factor)
            if rope_parameters:
                rope_scaling = rope_parameters | rope_scaling
        if rope_scaling:
            has_mrope = ("mrope_section" in rope_scaling
                         or rope_scaling.get("mrope_interleaved", False))
            if has_mrope:
                rope_scaling["type"] = "mrope"
                rope_scaling.pop("rope_type", None)
            elif "type" not in rope_scaling and "rope_type" in rope_scaling:
                rope_scaling["type"] = rope_scaling.pop("rope_type")
            text_config["rope_scaling"] = rope_scaling
        return text_config


# TODO: remove this once the transformers can support all of those models in _CONFIG_REGISTRY
class LazyConfigDict(dict):

    def __getitem__(self, key):
        import tensorrt_llm._torch.configs as configs
        return getattr(configs, super().__getitem__(key))


_CONFIG_REGISTRY: dict[str, type[transformers.PretrainedConfig]] = LazyConfigDict(
    deepseek_v32="DeepseekV3Config",
    kimi_k2="DeepseekV3Config",
    glm_moe_dsa="DeepseekV3Config",
    qwen3_5_moe="Qwen3_5MoeConfig",
)  # NOTE: HF config.json uses deepseek_v32 as model_type but with same DSV3 config class


def load_pretrained_config(model_name_or_path: str,
                           trust_remote_code: bool = False,
                           checkpoint_format: Optional[str] = None,
                           **kwargs) -> transformers.PretrainedConfig:
    config_dict, _ = transformers.PretrainedConfig.get_config_dict(
        model_name_or_path, **kwargs)
    model_type = config_dict.get("model_type")
    architectures = config_dict.get("architectures") or []

    if model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[model_type]
        model_config = config_class.from_pretrained(model_name_or_path,
                                                    **kwargs)
    elif model_type in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe",
                        "qwen3_5_moe_text") or (
                            architectures and architectures[0] in (
                                "Qwen3_5MoeForCausalLM",
                                "Qwen3_5MoeForConditionalGeneration",
                                "Qwen3_5ForCausalLM",
                                "Qwen3_5ForConditionalGeneration",
                            )):
        model_config = transformers.Qwen3NextConfig.from_dict(
            _Qwen35ConfigCompat.normalize(config_dict))
    elif checkpoint_format in ("mistral", "mistral_large_3"):
        from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import \
            MistralConfigLoader
        model_config = getattr(
            MistralConfigLoader().load(model_name_or_path).pretrained_config,
            "text_config")
    else:
        model_config = transformers.AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code)
    return model_config
