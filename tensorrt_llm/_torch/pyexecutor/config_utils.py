import dataclasses
from typing import List, Optional

import torch
import transformers

from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.logger import logger


def is_gemma4_hybrid(config):
    """True for Gemma4 models with hybrid attention (different head_dim per layer type)."""
    global_head_dim = getattr(config, 'global_head_dim', None)
    head_dim = getattr(config, 'head_dim', None)
    return (global_head_dim is not None and isinstance(head_dim, int)
            and global_head_dim != head_dim)


def is_hybrid_linear(config):
    return is_nemotron_hybrid(config) or is_qwen3_hybrid(config)


def _coerce_torch_dtype(dtype):
    """Normalize dtype values from HF configs into torch dtype objects.

    HF configs may store dtype fields as torch dtypes, strings, or the sentinel
    value "auto". Returning None for "auto" lets the caller keep its normal
    fallback path instead of treating "auto" as a concrete dtype.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        return None
    if isinstance(dtype, str):
        return str_dtype_to_torch(dtype)
    return dtype


def resolve_hf_torch_dtype(config):
    """Return the model's regular tensor dtype from common HF config fields.

    Transformers has used both dtype and torch_dtype across versions and model
    families. This helper checks both names and coerces whichever one is present
    into the form expected by TRT-LLM runtime code. An "auto" value in any
    field is treated the same as missing, so scanning continues to the next
    field instead of stopping with None.
    """
    for attr in ("dtype", "torch_dtype"):
        coerced = _coerce_torch_dtype(getattr(config, attr, None))
        if coerced is not None:
            return coerced
    return None


def resolve_mamba_ssm_cache_dtype(config):
    """Return the dtype to use for hybrid Mamba/SSM cache allocations.

    Qwen3.5-style configs may store this field on the top-level config or the
    nested text_config, and may call it either mamba_ssm_cache_dtype or
    mamba_ssm_dtype. This helper centralizes that lookup so cache creation does
    not fail later with a missing dtype. An "auto" value in any field is
    treated the same as missing.
    """
    configs = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        configs.append(text_config)

    for candidate_config in configs:
        for attr in ("mamba_ssm_cache_dtype", "mamba_ssm_dtype"):
            coerced = _coerce_torch_dtype(getattr(candidate_config, attr, None))
            if coerced is not None:
                return coerced
    return None


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


@dataclasses.dataclass
class MambaKVCacheParams:
    """Normalized mamba-related inputs for kv_cache_manager_cls.

    Field naming is consistent across the Nemotron-hybrid and Qwen3-hybrid
    families so call sites don't special-case per family.
    """
    # Mamba / linear-attention state dimensions
    state_size: int  # ssm_state_size      | linear_key_head_dim
    conv_kernel: int  # conv_kernel         | linear_conv_kernel_dim
    num_heads: int  # mamba_num_heads     | linear_num_value_heads
    n_groups: int  # n_groups            | linear_num_key_heads
    head_dim: int  # mamba_head_dim      | linear_value_head_dim

    # Per-layer masks and counts (trailing entries cover MTP/draft layers,
    # which are attention-only and carry no Mamba state).
    mamba_layer_mask: List[bool]
    full_attention_layer_mask: List[bool]
    num_mamba_layers: int
    num_full_attention_layers: int

    # Dtypes
    dtype: torch.dtype  # config.torch_dtype
    mamba_ssm_cache_dtype: Optional[
        torch.dtype]  # quant_config.mamba_ssm_cache_dtype

    def get_states_bytes_per_layer(self, mapping) -> int:
        """Return the total bytes of Mamba state per layer, used for budgeting."""
        tp_size = mapping.tp_size if not mapping.enable_attention_dp else 1
        d_inner = self.head_dim * self.num_heads
        conv_dim = (d_inner + 2 * self.n_groups * self.state_size) // tp_size
        nheads = self.num_heads // tp_size

        conv_dtype = self.dtype
        ssm_dtype = (self.mamba_ssm_cache_dtype
                     if self.mamba_ssm_cache_dtype is not None else self.dtype)
        conv_bytes = conv_dim * (self.conv_kernel - 1) * conv_dtype.itemsize
        ssm_bytes = (nheads * self.head_dim * self.state_size *
                     ssm_dtype.itemsize)
        state_bytes_per_layer = conv_bytes + ssm_bytes
        return state_bytes_per_layer


def _nemotron_hybrid_layer_masks(config, layer_mask):
    pattern = config.hybrid_override_pattern
    if layer_mask is None:
        return ([c == "*" for c in pattern], [c == "M" for c in pattern])

    # One-model speculative decoding: layer_mask may extend past the hybrid
    # pattern; treat trailing positions as attention-only draft layers.
    full_attn, mamba = [], []
    for i, include in enumerate(layer_mask):
        if i < len(pattern):
            is_attn = pattern[i] == "*"
            is_mamba = pattern[i] == "M"
        else:
            is_attn, is_mamba = True, False
        full_attn.append(is_attn and include)
        mamba.append(is_mamba and include)
    return full_attn, mamba


def _qwen3_hybrid_layer_masks(config, layer_mask):
    full_attn, mamba = get_qwen3_hybrid_layer_masks(config)
    if layer_mask is None:
        return full_attn, mamba

    if len(layer_mask) < len(full_attn):
        raise ValueError(
            "layer_mask is shorter than the Qwen3 hybrid layer pattern")
    base_len = len(full_attn)
    new_full_attn, new_mamba = [], []
    for i, include in enumerate(layer_mask):
        if i < base_len:
            new_full_attn.append(full_attn[i] and include)
            new_mamba.append(mamba[i] and include)
        else:
            new_full_attn.append(include)
            new_mamba.append(False)
    return new_full_attn, new_mamba


def extract_mamba_kv_cache_params(
    config,
    layer_mask: Optional[List[bool]] = None,
    spec_config=None,
    quant_config=None,
) -> MambaKVCacheParams:
    """Build the mamba-related inputs for kv_cache_manager_cls.

    Supports Nemotron-hybrid and Qwen3-hybrid (Qwen3-Next + Qwen3.5).

    Args:
        config: HuggingFace model config of a hybrid Mamba model.
        layer_mask: Optional per-layer keep mask used by one-model speculative
            decoding. Entries past the underlying hybrid pattern length are
            treated as attention-only draft layers. When provided, the caller
            is responsible for already including spec layers in the mask.
        spec_config: When `layer_mask` is None, used to extend the masks with
            MTP/draft attention layers (no Mamba state) so they receive KV
            cache entries.
        quant_config: Optional, used only to surface `mamba_ssm_cache_dtype`.

    Returns:
        MambaKVCacheParams with normalized field names.
    """
    if is_nemotron_hybrid(config):
        state_size = config.ssm_state_size
        conv_kernel = config.conv_kernel
        num_heads = config.mamba_num_heads
        n_groups = config.n_groups
        head_dim = config.mamba_head_dim
        full_attn_mask, mamba_mask = _nemotron_hybrid_layer_masks(
            config, layer_mask)
    elif is_qwen3_hybrid(config):
        state_size = config.linear_key_head_dim
        conv_kernel = config.linear_conv_kernel_dim
        num_heads = config.linear_num_value_heads
        n_groups = config.linear_num_key_heads
        head_dim = config.linear_value_head_dim
        full_attn_mask, mamba_mask = _qwen3_hybrid_layer_masks(
            config, layer_mask)
    else:
        raise ValueError(
            f"{type(config).__name__} is not a supported hybrid Mamba config")

    # When no explicit layer_mask is given, extend the masks here so MTP/draft
    # layers (attention-only, no Mamba state) get KV cache entries. With an
    # explicit layer_mask, the caller already encoded those entries.
    if layer_mask is None and spec_config is not None:
        # Imported lazily to avoid a circular dependency between
        # config_utils and tensorrt_llm._torch.speculative.
        from ..speculative.utils import get_num_spec_layers
        num_spec_layers = get_num_spec_layers(spec_config)
        if num_spec_layers > 0:
            full_attn_mask.extend([True] * num_spec_layers)
            mamba_mask.extend([False] * num_spec_layers)

    mamba_ssm_cache_dtype = None
    if quant_config is not None:
        mamba_ssm_cache_dtype = _coerce_torch_dtype(
            quant_config.mamba_ssm_cache_dtype)
    if mamba_ssm_cache_dtype is None:
        mamba_ssm_cache_dtype = (resolve_mamba_ssm_cache_dtype(config)
                                 or resolve_hf_torch_dtype(config)
                                 or torch.bfloat16)

    return MambaKVCacheParams(
        state_size=state_size,
        conv_kernel=conv_kernel,
        num_heads=num_heads,
        n_groups=n_groups,
        head_dim=head_dim,
        mamba_layer_mask=mamba_mask,
        full_attention_layer_mask=full_attn_mask,
        num_mamba_layers=sum(mamba_mask),
        num_full_attention_layers=sum(full_attn_mask),
        dtype=resolve_hf_torch_dtype(config) or torch.bfloat16,
        mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
    )


# TODO: remove this once the transformers can support all of those models in _CONFIG_REGISTRY
class LazyConfigDict(dict):

    def __getitem__(self, key):
        import tensorrt_llm._torch.configs as configs
        return getattr(configs, super().__getitem__(key))


_CONFIG_REGISTRY: dict[str, type[transformers.PretrainedConfig]] = LazyConfigDict(
    deepseek_v32="DeepseekV3Config",
    kimi_k2="DeepseekV3Config",
    glm_moe_dsa="DeepseekV3Config",
    laguna="LagunaConfig",
)  # NOTE: HF config.json uses deepseek_v32 as model_type but with same DSV3 config class


def load_pretrained_config(model_name_or_path: str,
                           trust_remote_code: bool = False,
                           checkpoint_format: Optional[str] = None,
                           **kwargs) -> transformers.PretrainedConfig:
    config_dict, _ = transformers.PretrainedConfig.get_config_dict(
        model_name_or_path, **kwargs)
    model_type = config_dict.get("model_type")
    architectures = config_dict.get("architectures") or []

    if checkpoint_format in ("mistral", "mistral_large_3"):
        from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import \
            MistralConfigLoader
        model_config = MistralConfigLoader().load(
            model_name_or_path).pretrained_config
    elif (model_type == "qwen3_5_moe" and
          (("text_config" in config_dict and "vision_config" in config_dict) or
           (architectures
            and architectures[0] == "Qwen3_5MoeForConditionalGeneration"))):
        # Qwen3.5-MoE VLM: HF native composite config + model-side normalizer.
        from tensorrt_llm._torch.models.modeling_qwen3_5 import \
            _normalize_qwen35_moe_vl_config
        model_config = transformers.Qwen3_5MoeConfig.from_pretrained(
            model_name_or_path, **kwargs)
        _normalize_qwen35_moe_vl_config(model_config)
    elif model_type in _CONFIG_REGISTRY:
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
        # Qwen3.5 text-only: flatten to Qwen3NextConfig via the model-side shim.
        from tensorrt_llm._torch.models.modeling_qwen3_5 import \
            Qwen35ConfigCompat
        model_config = transformers.Qwen3NextConfig.from_dict(
            Qwen35ConfigCompat.normalize(config_dict))
    elif (model_type == "exaone4" and config_dict.get("sliding_window") is None
          and config_dict.get("layer_types") is None):
        # transformers 5.5.x Exaone4Config.__post_init__ first forces
        # `sliding_window_pattern = 0` when `sliding_window is None`, then
        # synthesizes `layer_types` via `(i + 1) % sliding_window_pattern`,
        # which raises ZeroDivisionError. The released EXAONE-4.0 ckpts
        # publish both fields as `null`. Inject layer_types directly into
        # the config dict (all full attention, since sliding_window is null
        # means no sliding) and bypass AutoConfig.from_pretrained's kwargs
        # filtering by going through CONFIG_MAPPING + from_dict.
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        n_layers = config_dict.get("num_hidden_layers") or 32
        patched_dict = {
            **config_dict, "layer_types": ["full_attention"] * n_layers
        }
        model_config = CONFIG_MAPPING[model_type].from_dict(patched_dict)
    else:
        model_config = transformers.AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code)

    # Transformers 5.x sets rope_scaling to {"rope_type": "default"} instead
    # of None for models with standard RoPE (no scaling).  Clear it so that
    # downstream code (e.g. RopeParams.from_config) treats it the same as
    # rope_scaling=None, which is what transformers 4.x produced.
    rope_scaling = getattr(model_config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        has_real_scaling = any(k for k in rope_scaling
                               if k not in ("rope_type", "type", "rope_theta"))
        if rope_type == "default" and not has_real_scaling:
            # Preserve rope_theta before clearing, since rope_parameters
            # (which rope_scaling delegates to) will also become None and
            # rope_theta may only exist there in transformers 5.x.
            # When rope_theta is missing from rope_scaling, no preservation is
            # needed — model_config.rope_theta (if any) is already canonical.
            rope_theta = rope_scaling.get("rope_theta")
            if rope_theta is not None:
                # `rope_scaling.rope_theta` mirrors `rope_parameters.rope_theta`
                # and is the canonical value in transformers 5.x. Adopt it as
                # `model_config.rope_theta` so downstream code can read it.
                # Note: `model_config.rope_theta` may be absent entirely
                # (transformers 5.x dropped it from some configs) or may be a
                # config-class default that does NOT reflect the user's
                # config.json (e.g. DeepseekV3Config default 10000 vs. user's
                # 1000000 — the original bug).
                user_top_level = config_dict.get("rope_theta")
                if user_top_level is not None and user_top_level != rope_theta:
                    # User explicitly set both top-level rope_theta and
                    # rope_parameters.rope_theta to disagreeing values — keep
                    # the top-level value but warn loudly.
                    logger.warning(
                        f"rope_scaling.rope_theta ({rope_theta}) differs from "
                        f"config rope_theta ({user_top_level}); keeping the "
                        f"top-level value.")
                    model_config.rope_theta = user_top_level
                else:
                    model_config.rope_theta = rope_theta
            model_config.rope_scaling = None

    return model_config
