# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import List, Optional, Sequence

import torch
import transformers

from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.logger import logger


def uses_vswa_kv_cache_layout(
        max_attention_windows: Optional[Sequence[Optional[int]]]) -> bool:
    """Return whether windows require a variable-window KV cache layout.

    Recurrent-state cache sentinels are negative and do not represent
    attention windows, so hybrid linear-attention layouts are not VSWA.
    """
    return (max_attention_windows is not None
            and len(set(max_attention_windows)) > 1
            and all(window is None or window > 0
                    for window in max_attention_windows))


def _is_sliding_attention_layer(layer_type: object) -> bool:
    """Return whether a config layer type denotes sliding attention."""
    layer_type_name = getattr(layer_type, "name", str(layer_type)).lower()
    return "sliding" in layer_type_name


def get_layer_attention_window(
    config: object,
    layer_idx: int,
) -> Optional[int]:
    """Return the active sliding window for one layer, or ``None``.

    An explicit ``use_sliding_window=False`` disables sliding attention.
    Otherwise, infer it from ``sliding_window`` and ``layer_types`` so legacy
    configs that omit the flag continue to work. Qwen2-style configs that omit
    ``layer_types`` use ``max_window_layers`` as the first sliding-layer index.
    """
    use_sliding_window = getattr(config, "use_sliding_window", None)
    if use_sliding_window is False:
        return None

    sliding_window = getattr(config, "sliding_window", None)
    if isinstance(sliding_window, (list, tuple)):
        raise NotImplementedError(
            "Attention-window derivation assumes a single sliding-window "
            f"size, got multiple: {sliding_window}")
    if sliding_window is None:
        if use_sliding_window is True:
            raise ValueError(
                "use_sliding_window=True requires a positive integer "
                "sliding_window.")
        return None

    layer_types = getattr(config, "layer_types", None)
    if layer_types:
        layer_type = layer_types[layer_idx % len(layer_types)]
        if not _is_sliding_attention_layer(layer_type):
            return None
    else:
        max_window_layers = getattr(config, "max_window_layers", None)
        if (isinstance(max_window_layers, int)
                and not isinstance(max_window_layers, bool)
                and layer_idx < max_window_layers):
            return None

    if (not isinstance(sliding_window, int) or isinstance(sliding_window, bool)
            or sliding_window <= 0):
        raise ValueError(
            "Sliding attention requires a positive integer sliding_window.")
    return sliding_window


def is_gemma4_hybrid(config):
    """True for Gemma4 models with hybrid attention (different head_dim per layer type)."""
    global_head_dim = getattr(config, 'global_head_dim', None)
    head_dim = getattr(config, 'head_dim', None)
    return (global_head_dim is not None and isinstance(head_dim, int)
            and global_head_dim != head_dim)


def is_hybrid_linear(config):
    return is_nemotron_hybrid(config) or is_qwen3_hybrid(config) or \
        is_kimi_linear(config)


def is_kimi_linear(config):
    """True for Kimi K3 ("kimi_linear") hybrid KDA + MLA text models.

    Handles both the flattened text config (model_type "kimi_linear") and the
    composite VLM config (model_type "kimi_k3" with a nested text_config).
    """
    model_type = getattr(config, "model_type", None)
    if model_type == "kimi_linear":
        return getattr(config, "linear_attn_config", None) is not None
    if model_type == "kimi_k3":
        text_config = getattr(config, "text_config", None)
        return text_config is not None and is_kimi_linear(text_config)
    return False


def unwrap_kimi_text_config(config):
    """Return the flattened Kimi text config.

    ``is_kimi_linear`` accepts both the flattened text config and the
    composite "kimi_k3" config with a nested ``text_config``; consumers read
    text-level fields (``linear_attn_config``, ``num_hidden_layers``, ...),
    so they must unwrap the composite form first.
    """
    if getattr(config, "model_type", None) == "kimi_k3":
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            return text_config
    return config


def get_kimi_linear_layer_masks(config):
    """Return (full_attention_layer_mask, kda_layer_mask) for Kimi K3.

    The config's ``linear_attn_config`` carries 1-indexed ``kda_layers`` and
    ``full_attn_layers`` lists; every decoder layer must be exactly one of
    the two.
    """
    config = unwrap_kimi_text_config(config)
    lin = config.linear_attn_config
    kda_layers = set(lin["kda_layers"])
    full_attn_layers = set(lin["full_attn_layers"])
    full_mask, kda_mask = [], []
    for layer_idx in range(config.num_hidden_layers):
        is_kda = (layer_idx + 1) in kda_layers
        is_full = (layer_idx + 1) in full_attn_layers
        if is_kda == is_full:
            raise ValueError(
                f"Kimi K3 layer {layer_idx} (1-indexed {layer_idx + 1}) must "
                f"be exactly one of KDA / full attention; got is_kda={is_kda} "
                f"is_full={is_full}")
        kda_mask.append(is_kda)
        full_mask.append(is_full)
    return full_mask, kda_mask


def get_kimi_linear_num_attention_layers(config):
    full_mask, _ = get_kimi_linear_layer_masks(config)
    return sum(full_mask)


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


def resolve_ssm_cache_dtype(config):
    """Return the dtype to use for hybrid SSM-style state cache allocations.

    Covers all recurrent-state hybrids (Mamba SSM, Qwen3.5 GDN linear
    attention); the "mamba" in the config field name is historical.
    Qwen3.5-style configs may store the field on the top-level config or the
    nested text_config. An "auto" value is treated the same as missing.

    Only the explicit mamba_ssm_cache_dtype field is honored here. The
    checkpoint's mamba_ssm_dtype field expresses the SSM *compute* dtype
    intent, not the cache dtype: honoring it for cache allocation silently
    switches Qwen3.5 GDN state caches to fp32, which disables the FlashInfer
    bf16-state decode kernel and doubles state memory traffic (~20% serving
    throughput loss). Users can opt in explicitly via
    kv_cache_config.mamba_ssm_cache_dtype.
    """
    configs = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        configs.append(text_config)

    for candidate_config in configs:
        coerced = _coerce_torch_dtype(
            getattr(candidate_config, "mamba_ssm_cache_dtype", None))
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


def is_minimax_m3(sparse_attention_config):
    """True when the sparse attention config selects the MiniMax-M3 algorithm."""
    return sparse_attention_config is not None and sparse_attention_config.algorithm == "minimax_m3"


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

    # Target-layer masks and counts. Appended MTP/draft layers need only a
    # count because every draft layer is full attention.
    mamba_layer_mask: List[bool]
    target_full_attention_layer_mask: List[bool]
    num_mamba_layers: int
    num_draft_layers: int

    # Dtypes
    dtype: torch.dtype  # config.torch_dtype
    mamba_ssm_cache_dtype: Optional[
        torch.dtype]  # quant_config.mamba_ssm_cache_dtype

    def get_layer_masks(
        self,
        *,
        is_draft: bool = False,
        use_separate_draft_kv_cache: bool = False,
    ) -> tuple[List[bool], List[bool]]:
        """Return Mamba and attention masks for one cache manager.

        Target masks use target-model layer indices. Appended one-model draft
        layers use indices immediately following the target layers, so a
        draft-only manager receives target-sized false prefixes. A combined
        manager receives the concatenated target and draft layouts.
        """
        target_mamba_mask = list(self.mamba_layer_mask)
        target_attention_mask = list(self.target_full_attention_layer_mask)
        num_target_layers = len(target_mamba_mask)
        num_draft_layers = self.num_draft_layers
        draft_attention_mask = [True] * num_draft_layers

        if is_draft and num_draft_layers > 0:
            return (
                [False] * (num_target_layers + num_draft_layers),
                [False] * num_target_layers + draft_attention_mask,
            )
        if use_separate_draft_kv_cache:
            return target_mamba_mask, target_attention_mask
        return (
            target_mamba_mask + [False] * num_draft_layers,
            target_attention_mask + draft_attention_mask,
        )

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


def extract_mamba_kv_cache_params(
    config,
    spec_config=None,
    quant_config=None,
) -> MambaKVCacheParams:
    """Build the mamba-related inputs for kv_cache_manager_cls.

    Supports Nemotron-hybrid, Qwen3-hybrid (Qwen3-Next + Qwen3.5) and
    Kimi K3 (kimi_linear).

    Args:
        config: HuggingFace model config of a hybrid Mamba model.
        spec_config: Optional speculative-decoding config used to describe
            appended attention-only MTP/draft layers separately from target
            layers.
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
        pattern = config.hybrid_override_pattern
        target_full_attn_mask = [layer_type == "*" for layer_type in pattern]
        mamba_mask = [layer_type == "M" for layer_type in pattern]
    elif is_qwen3_hybrid(config):
        state_size = config.linear_key_head_dim
        conv_kernel = config.linear_conv_kernel_dim
        num_heads = config.linear_num_value_heads
        n_groups = config.linear_num_key_heads
        head_dim = config.linear_value_head_dim
        target_full_attn_mask, mamba_mask = get_qwen3_hybrid_layer_masks(config)
    elif is_kimi_linear(config):
        # Kimi K3 KDA (Kimi Delta Attention) state, mapped onto the Mamba
        # cache-manager parametrization (see PythonMambaCacheManager):
        #   conv_dim = head_dim*num_heads + 2*n_groups*state_size
        #            = 3 * num_heads * head_dim  -> [q | k | v] short-conv
        #   ssm state shape = [num_heads, head_dim, state_size]
        #            = [H, V, K] fp32 delta-rule recurrent state.
        # conv_kernel is set to short_conv_kernel_size + 1 so the pool's
        # (conv_kernel - 1) columns hold the FULL FLA ShortConvolution cache
        # window of `short_conv_kernel_size` columns.
        lin = unwrap_kimi_text_config(config).linear_attn_config
        state_size = lin["head_dim"]
        conv_kernel = lin["short_conv_kernel_size"] + 1
        num_heads = lin["num_heads"]
        n_groups = lin["num_heads"]
        head_dim = lin["head_dim"]
        target_full_attn_mask, mamba_mask = get_kimi_linear_layer_masks(config)
    else:
        raise ValueError(
            f"{type(config).__name__} is not a supported hybrid Mamba config")

    num_draft_layers = 0
    if spec_config is not None:
        # Imported lazily to avoid a circular dependency between
        # config_utils and tensorrt_llm._torch.speculative.
        from ..speculative.utils import get_num_spec_layers
        num_draft_layers = get_num_spec_layers(spec_config) or 0

    mamba_ssm_cache_dtype = None
    if quant_config is not None:
        mamba_ssm_cache_dtype = _coerce_torch_dtype(
            quant_config.mamba_ssm_cache_dtype)
    if mamba_ssm_cache_dtype is None:
        mamba_ssm_cache_dtype = (resolve_ssm_cache_dtype(config)
                                 or resolve_hf_torch_dtype(config)
                                 or torch.bfloat16)
    if is_kimi_linear(config) and mamba_ssm_cache_dtype != torch.float32:
        # The KDA delta-rule recurrent state must be kept in fp32 for
        # numerical parity with the HF reference (fla chunk/fused_recurrent
        # KDA kernels carry the state in fp32).
        logger.info(
            f"Kimi K3: overriding mamba_ssm_cache_dtype "
            f"{mamba_ssm_cache_dtype} -> torch.float32 (KDA recurrent state "
            "must be fp32)")
        mamba_ssm_cache_dtype = torch.float32

    return MambaKVCacheParams(
        state_size=state_size,
        conv_kernel=conv_kernel,
        num_heads=num_heads,
        n_groups=n_groups,
        head_dim=head_dim,
        mamba_layer_mask=mamba_mask,
        target_full_attention_layer_mask=target_full_attn_mask,
        num_mamba_layers=sum(mamba_mask),
        num_draft_layers=num_draft_layers,
        dtype=resolve_hf_torch_dtype(config) or torch.bfloat16,
        mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
    )


def is_qwen_image_bench_config(config_dict: dict) -> bool:
    """Detect Qwen-Image-Bench's composite VLM checkpoint config.

    The checkpoint advertises the generic Qwen3.5 VLM architecture, but
    TRT-LLM needs a dedicated architecture key to route it to
    QwenImageBenchModel. Generic Qwen3.5 text checkpoints can also publish
    composite text/vision metadata, so require the explicit
    `language_model_only: false` marker from Qwen-Image-Bench before
    rewriting the architecture.

    The text-config normalization itself lives on
    `modeling_qwen3_5.Qwen35ConfigCompat`; this stays here because it is pure
    dispatch detection over the raw config dict and keeps the model module out
    of the hot path until a Qwen3.5 checkpoint is actually loaded.
    """
    architectures = config_dict.get("architectures") or []
    text_config = config_dict.get("text_config")
    vision_config = config_dict.get("vision_config")
    required_multimodal_token_ids = {
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    }
    return (config_dict.get("language_model_only") is False
            and architectures[:1] == ["Qwen3_5ForConditionalGeneration"]
            and isinstance(text_config, dict) and bool(text_config)
            and isinstance(vision_config, dict) and bool(vision_config)
            and required_multimodal_token_ids.issubset(config_dict))


def _resolve_composite_torch_dtype(*config_dicts: dict) -> torch.dtype:
    """Resolve a concrete torch dtype from one or more raw config dicts.

    Respects an explicit ``torch_dtype``/``dtype`` declaration (string or
    ``torch.dtype``) from the first dict that provides one, and otherwise falls
    back to ``bfloat16`` (TensorRT-LLM's default, matching the checkpoint).
    """
    for config_dict in config_dicts:
        for key in ("torch_dtype", "dtype"):
            coerced = _coerce_torch_dtype(config_dict.get(key))
            if coerced is not None:
                return coerced
    return torch.bfloat16


def _build_minicpmv4_6_config(
        config_dict: dict) -> transformers.PretrainedConfig:
    """Build the composite MiniCPM-V 4.6 config from a raw config.json dict.

    The top-level ``minicpmv4_6`` model_type is only known to
    ``transformers>=5.7.0``; rebuild it locally so the PyTorch backend loads on
    older releases.  The inner text tower is normalized into a
    ``Qwen3NextConfig`` through the shared ``Qwen35ConfigCompat`` shim (the same
    path standalone Qwen3.5 dense uses), and the top-level ``model_type`` is kept
    as ``minicpmv4_6`` so ``MULTIMODAL_PLACEHOLDER_REGISTRY`` lookup succeeds.

    TODO: this local builder is a transition path for the repo's pinned
    transformers 5.5.4.  Once the pin is bumped to ``>=5.7.0``, drop the
    ``MiniCPMV4_6Config`` shim and build the config from the native
    ``transformers.MiniCPMV4_6Config`` instead.
    """
    from tensorrt_llm._torch.configs.minicpmv4_6 import MiniCPMV4_6Config
    from tensorrt_llm._torch.models.modeling_qwen3_5 import Qwen35ConfigCompat

    # Extract + normalize the Qwen3.5 dense text tower (rope flatten,
    # quantization inheritance, architectures -> Qwen3_5ForCausalLM). MiniCPM-V
    # 4.6 always nests its language model under ``text_config``, so force
    # extraction of that nested config (same path Qwen-Image-Bench uses).
    text_dict = Qwen35ConfigCompat.normalize(config_dict,
                                             require_text_config=True)
    text_config = transformers.Qwen3NextConfig.from_dict(text_dict)

    # MiniCPM-V 4.6's config.json declares no torch_dtype/dtype (neither at the
    # top level nor inside text_config).  Several engine-build paths read
    # ``pretrained_config.torch_dtype`` directly with no fallback -- the hybrid
    # (mamba) KV-cache byte sizing and ``validate_and_set_mamba_ssm_cache_dtype``
    # -- so a ``None`` dtype crashes with ``None.itemsize``.  Pin a concrete
    # dtype on both the text tower and the composite config so every downstream
    # consumer (config validation before model build, and the post_config swap
    # that replaces pretrained_config with the inner text_config) agrees.
    resolved_dtype = _resolve_composite_torch_dtype(config_dict, text_dict)
    text_config.torch_dtype = resolved_dtype

    composite_config = MiniCPMV4_6Config(
        text_config=text_config,
        vision_config=config_dict.get("vision_config"),
        insert_layer_id=config_dict.get("insert_layer_id", 6),
        image_size=config_dict.get("image_size", 448),
        drop_vision_last_layer=config_dict.get("drop_vision_last_layer", False),
        image_token_id=config_dict.get("image_token_id"),
        video_token_id=config_dict.get("video_token_id"),
        downsample_mode=config_dict.get("downsample_mode", "16x"),
        merge_kernel_size=config_dict.get("merge_kernel_size", (2, 2)),
        merger_times=config_dict.get("merger_times", 1),
        tie_word_embeddings=config_dict.get("tie_word_embeddings", False),
        architectures=config_dict.get("architectures"),
    )
    composite_config.torch_dtype = resolved_dtype
    return composite_config


def is_kimi_k3_multimodal_config(config_dict: dict) -> bool:
    """Detect Kimi K3's composite multimodal VLM checkpoint config.

    The released Kimi K3 VLM checkpoint advertises ``model_type: kimi_k3`` with
    nested ``text_config`` (the ``kimi_linear`` text core) and ``vision_config``
    (the MoonViT-3D tower + projector). When both sub-configs are present and
    multimodal is not explicitly disabled, TRT-LLM keeps the composite
    ``KimiK3Config`` and routes the checkpoint to
    ``KimiK3ForConditionalGeneration``. Text-only ``kimi_linear`` checkpoints (or
    a ``kimi_k3`` config with ``language_model_only: true`` / no vision_config)
    fall through to the text-only flatten path.
    """
    text_config = config_dict.get("text_config")
    vision_config = config_dict.get("vision_config")
    return (config_dict.get("model_type") == "kimi_k3"
            and config_dict.get("language_model_only") is not True
            and isinstance(text_config, dict) and bool(text_config)
            and isinstance(vision_config, dict) and bool(vision_config))


# TODO: remove this once the transformers can support all of those models in _CONFIG_REGISTRY
class LazyConfigDict(dict):

    def __getitem__(self, key):
        import tensorrt_llm._torch.configs as configs
        return getattr(configs, super().__getitem__(key))


_CONFIG_REGISTRY: dict[str, type[transformers.PretrainedConfig]] = LazyConfigDict(
    cosmos3="Cosmos3Config",
    cosmos3_omni=
    "Cosmos3Config",  # backward-compat alias for pre-rename checkpoints
    deepseek_v32="DeepseekV3Config",
    kimi_k2="DeepseekV3Config",
    glm_moe_dsa="DeepseekV3Config",
    laguna="LagunaConfig",
)  # NOTE: HF config.json uses deepseek_v32 as model_type but with same DSV3 config class


def load_pretrained_config(model_name_or_path: str,
                           trust_remote_code: bool = False,
                           checkpoint_format: Optional[str] = None,
                           **kwargs) -> transformers.PretrainedConfig:
    if checkpoint_format in ("mistral", "mistral_large_3"):
        from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import \
            MistralConfigLoader
        return MistralConfigLoader().load(model_name_or_path).pretrained_config

    config_dict, _ = transformers.PretrainedConfig.get_config_dict(
        model_name_or_path, **kwargs)
    model_type = config_dict.get("model_type")
    architectures = config_dict.get("architectures") or []

    if checkpoint_format in ("mistral", "mistral_large_3"):
        from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import \
            MistralConfigLoader
        model_config = MistralConfigLoader().load(
            model_name_or_path).pretrained_config
    elif is_qwen_image_bench_config(config_dict):
        from tensorrt_llm._torch.models.modeling_qwen3_5 import (
            Qwen35ConfigCompat, _normalize_qwen35_quantization_config)
        model_config = transformers.AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code)
        # Keep the composite VLM config so the vision encoder and multimodal
        # token IDs remain available, but normalize the text side to the
        # Qwen3Next-compatible shape used by TRT-LLM's Qwen3.5 decoder.
        model_config.architectures = ["QwenImageBenchForConditionalGeneration"]
        model_config.text_config = transformers.Qwen3NextConfig.from_dict(
            Qwen35ConfigCompat.normalize(config_dict, require_text_config=True))
        _normalize_qwen35_quantization_config(model_config)
    elif model_type == "glm_moe_dsa":
        # GLM-MoE-DSA configs tag every layer with
        # layer_types=['deepseek_sparse_attention', ...] for HF bookkeeping.
        # TRT-LLM never reads it (get_layer_types is Gemma3-only; DSA layer
        # routing is driven by index_topk_freq / index_skip_topk_offset), and
        # transformers' validate_layer_type rejects the unknown entry. Drop the
        # unused field and build from the dict via the registered config class,
        # mirroring the exaone4 handling below.
        config_dict.pop("layer_types", None)
        model_config = _CONFIG_REGISTRY[model_type].from_dict(
            config_dict, **kwargs)
    elif (model_type == "qwen3_5_moe" and
          (("text_config" in config_dict and "vision_config" in config_dict) or
           (architectures
            and architectures[0] == "Qwen3_5MoeForConditionalGeneration"))):
        # Qwen3.5-MoE VLM: HF native composite config + model-side normalizer.
        from tensorrt_llm._torch.models.modeling_qwen3_5 import \
            _normalize_qwen35_vl_config
        model_config = transformers.Qwen3_5MoeConfig.from_pretrained(
            model_name_or_path, **kwargs)
        _normalize_qwen35_vl_config(model_config,
                                    inner_arch="Qwen3_5MoeForCausalLM")
    elif (model_type == "qwen3_5" and
          (("text_config" in config_dict and "vision_config" in config_dict) or
           (architectures
            and architectures[0] == "Qwen3_5ForConditionalGeneration"))):
        # Qwen3.5 dense VLM: HF native composite config + model-side normalizer.
        # Must precede the text-only `qwen3_5` branch below so the composite
        # config isn't flattened and vision_config dropped.
        from tensorrt_llm._torch.models.modeling_qwen3_5 import \
            _normalize_qwen35_vl_config
        model_config = transformers.Qwen3_5Config.from_pretrained(
            model_name_or_path, **kwargs)
        _normalize_qwen35_vl_config(model_config,
                                    inner_arch="Qwen3_5ForCausalLM")
    elif is_kimi_k3_multimodal_config(config_dict):
        # Kimi K3 multimodal VLM: keep the composite KimiK3Config so the vision
        # tower, projector, and multimodal token id remain available, and route
        # to KimiK3ForConditionalGeneration. Must precede the text-only flatten
        # branch below so vision_config isn't dropped. Built from the in-tree
        # config classes (no trust_remote_code needed for the config).
        from tensorrt_llm._torch.configs import KimiK3Config
        model_config = KimiK3Config.from_dict(config_dict, **kwargs)
        model_config.architectures = ["KimiK3ForConditionalGeneration"]
    elif model_type in ("kimi_k3", "kimi_linear"):
        # Kimi K3 text-only (or multimodal explicitly disabled): flatten to the
        # in-tree KimiLinearConfig and run the text model only (this also avoids
        # trust_remote_code for the config).
        from tensorrt_llm._torch.configs import KimiLinearConfig
        text_dict = dict(config_dict.get("text_config") or config_dict)
        model_config = KimiLinearConfig.from_dict(text_dict)
        model_config.architectures = ["KimiLinearForCausalLM"]
    elif model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[model_type]
        model_config = config_class.from_pretrained(model_name_or_path,
                                                    **kwargs)
    elif model_type == "minicpmv4_6" or (
            architectures
            and architectures[0] == "MiniCPMV4_6ForConditionalGeneration"):
        model_config = _build_minicpmv4_6_config(config_dict)
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
