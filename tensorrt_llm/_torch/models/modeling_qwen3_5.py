# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from types import SimpleNamespace
from typing import Dict, List

import torch
from transformers import PretrainedConfig

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization import QuantAlgo

from ...inputs import (
    ContentFormat,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    register_input_processor,
    support_multimodal_disaggregated,
)
from ..pyexecutor.config_utils import get_qwen3_hybrid_layer_types
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from .modeling_multimodal_utils import _is_mm_disagg
from .modeling_qwen3_next import Qwen3NextForCausalLM
from .modeling_qwen3vl import (
    Qwen3VisionModel,
    Qwen3VisionModelBase,
    Qwen3VLInputProcessorBase,
    Qwen3VLModelBase,
)
from .modeling_utils import ModelConfig, register_auto_model, register_vision_encoder

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


# --- Config adapters --------------------------------------------------------
#
# These run from `load_pretrained_config` in
# `tensorrt_llm/_torch/pyexecutor/config_utils.py` via lazy import — the
# runtime layer asks the model module how to load its own config.
#
# There are two entry points:
#   - `Qwen35ConfigCompat.normalize(config_dict)` — for text-only
#     Qwen3.5 (MoE and dense). Returns a dict that
#     `transformers.Qwen3NextConfig.from_dict(...)` can consume, so the
#     existing Qwen3Next runtime is reused unchanged.
#   - `_normalize_qwen35_moe_vl_config(model_config)` — for the
#     Qwen3.5-MoE VLM. Mutates the HF-native `transformers.Qwen3_5MoeConfig`
#     in place, attaching the runtime aliases the Qwen3Next-based LM expects
#     while keeping `text_config` / `vision_config` composite.


class Qwen35ConfigCompat:
    """Temporary shim for flattening Qwen3.5 text configs into Qwen3NextConfig.

    We normalize to `Qwen3NextConfig` (rather than to a Qwen3.5-native
    schema) so the runtime can reuse the existing `Qwen3NextForCausalLM`
    model implementation unchanged — Qwen3.5 text is structurally identical
    to Qwen3Next, so matching the config schema lets the same code serve
    both.

    This is used for Qwen3.5 text-only configs and for shared helper logic such
    as RoPE and quantization exclude-module normalization. Qwen3.5-MoE VLM
    configs should stay composite and use transformers.Qwen3_5MoeConfig plus
    _normalize_qwen35_moe_vl_config instead.

    To remove: delete this class and the elif branch in
    load_pretrained_config that flattens Qwen3.5 text configs.
    """

    @staticmethod
    def normalize(config_dict: dict, require_text_config: bool = False) -> dict:
        """Entry point: raw config.json dict -> flat Qwen3NextConfig-compatible dict.

        `require_text_config=True` is used by the Qwen-Image-Bench composite
        checkpoint, which always nests its language model under `text_config`;
        it forces extraction of that nested config rather than falling back to
        treating the top-level dict as the text config.
        """
        text_config = Qwen35ConfigCompat._extract_text_config(
            config_dict, require_text_config=require_text_config
        )
        text_config = Qwen35ConfigCompat._inherit_quantization_config(config_dict, text_config)
        text_config = Qwen35ConfigCompat._flatten_rope(text_config)

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
    def _extract_text_config(config_dict: dict, require_text_config: bool = False) -> dict:
        """Pull nested text_config from VLM checkpoints, or use dict as-is.

        `require_text_config=True` forces extraction of the nested `text_config`
        (used by the Qwen-Image-Bench composite checkpoint) and raises if it is
        missing, instead of falling back to the top-level dict.
        """
        if require_text_config:
            text_config = config_dict.get("text_config")
            if not isinstance(text_config, dict) or not text_config:
                raise ValueError("Qwen3.5 composite config is missing a usable text_config")
            return dict(text_config)

        architectures = config_dict.get("architectures") or []
        if (
            architectures
            and architectures[0] in Qwen35ConfigCompat._VLM_ARCHITECTURES
            and isinstance(config_dict.get("vision_config"), dict)
        ):
            text_config = config_dict.get("text_config")
            if not isinstance(text_config, dict) or not text_config:
                raise ValueError("Qwen3.5 composite config is missing a usable text_config")
            text_config = dict(text_config)
        elif (
            architectures
            and architectures[0] in Qwen35ConfigCompat._VLM_ARCHITECTURES
            and isinstance(config_dict.get("text_config"), dict)
        ):
            text_config = dict(config_dict["text_config"])
        else:
            text_config = dict(config_dict)
        if not text_config:
            raise ValueError("Qwen3.5 config is missing a usable text_config")
        return text_config

    @staticmethod
    def _inherit_quantization_config(config_dict: dict, text_config: dict) -> dict:
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
            modules = Qwen35ConfigCompat._normalize_exclude_modules(
                quantization_config["modules_to_not_convert"]
            )
            modules = Qwen35ConfigCompat._add_qkvz_bf16_workaround(text_config, modules)
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
                name = "model." + name[len("model.language_model.") :]
            if name.startswith("model.visual.") or name.startswith("mtp."):
                continue
            name = re.sub(r"\.in_proj_[ab]$", ".in_proj_ba", name)
            name = re.sub(r"\.in_proj_(q|k|v|z|qkv)$", ".in_proj_qkvz", name)
            normalized.add(name)
        return sorted(normalized)

    @staticmethod
    def _add_qkvz_bf16_workaround(text_config: dict, modules: list[str]) -> list[str]:
        """Keep packed linear-attention qkvz on bf16 path for all linear-attention layers.

        Temporary until FP8 block-scale TP loading is fixed for this layout.
        """
        try:
            layer_types = get_qwen3_hybrid_layer_types(SimpleNamespace(**text_config))
        except (ValueError, AttributeError):
            return modules
        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type == "linear_attention":
                modules.append(f"model.layers.{layer_idx}.linear_attn.in_proj_qkvz")
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
            partial_rotary_factor = rope_parameters.pop("partial_rotary_factor", None)
            if partial_rotary_factor is not None:
                text_config.setdefault("partial_rotary_factor", partial_rotary_factor)
            if rope_parameters:
                rope_scaling = rope_parameters | rope_scaling
        if rope_scaling:
            has_mrope = "mrope_section" in rope_scaling or rope_scaling.get(
                "mrope_interleaved", False
            )
            if has_mrope:
                rope_scaling["type"] = "mrope"
                rope_scaling.pop("rope_type", None)
            elif "type" not in rope_scaling and "rope_type" in rope_scaling:
                rope_type = rope_scaling.pop("rope_type")
                # "default" means standard RoPE (no scaling) — don't set
                # rope_scaling to avoid triggering scaling code paths.
                if rope_type == "default":
                    rope_scaling = {}
                else:
                    rope_scaling["type"] = rope_type
            if rope_scaling:
                text_config["rope_scaling"] = rope_scaling
        return text_config


def _normalize_qwen35_mrope_config(text_config) -> None:
    """Materialize Qwen3.5 mRoPE aliases needed by the Qwen3-VL path.

    HF stores RoPE metadata under `rope_parameters`; the shared Qwen3-VL
    wrapper reads `rope_theta`, `partial_rotary_factor`, and
    `rope_scaling` directly on the text config.
    """
    rope_parameters = getattr(text_config, "rope_parameters", None)
    if not rope_parameters:
        return
    if hasattr(rope_parameters, "to_dict"):
        rope_parameters = rope_parameters.to_dict()
    flattened = Qwen35ConfigCompat._flatten_rope(
        {
            "rope_parameters": dict(rope_parameters),
            "rope_scaling": dict(getattr(text_config, "rope_scaling", None) or {}),
        }
    )
    for attr in ("rope_theta", "partial_rotary_factor", "rope_scaling"):
        value = flattened.get(attr)
        if value is not None:
            setattr(text_config, attr, value)


def _normalize_qwen35_qwen3next_text_aliases(text_config) -> None:
    """Materialize Qwen3Next-style text aliases used by the shared runtime."""
    if getattr(text_config, "intermediate_size", None) is None:
        moe_intermediate_size = getattr(text_config, "moe_intermediate_size", None)
        num_experts_per_tok = getattr(text_config, "num_experts_per_tok", None)
        shared_expert_intermediate_size = (
            getattr(text_config, "shared_expert_intermediate_size", 0) or 0
        )
        if moe_intermediate_size is not None and num_experts_per_tok is not None:
            text_config.intermediate_size = (
                num_experts_per_tok * moe_intermediate_size + shared_expert_intermediate_size
            )


def _normalize_qwen35_quantization_config(model_config) -> None:
    quantization_config = getattr(model_config, "quantization_config", None)
    if not isinstance(quantization_config, dict):
        return

    modules = quantization_config.get("modules_to_not_convert")
    if modules is None:
        return

    text_config = getattr(model_config, "text_config", None)
    normalized_modules = Qwen35ConfigCompat._normalize_exclude_modules(modules)
    if text_config is not None:
        normalized_modules = Qwen35ConfigCompat._add_qkvz_bf16_workaround(
            text_config.to_dict(), normalized_modules
        )
    quantization_config["modules_to_not_convert"] = sorted(set(normalized_modules))


# Map the inner (text) causal-LM arch to the outer VLM arch, used only for the
# defensive fallback when a config arrives without an `architectures` field.
_INNER_TO_OUTER_VL_ARCH = {
    "Qwen3_5MoeForCausalLM": "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5ForCausalLM": "Qwen3_5ForConditionalGeneration",
}


def _normalize_qwen35_vl_config(model_config, inner_arch: str) -> None:
    """Adapt an HF Qwen3.5 VLM config (MoE or dense) to TRT-LLM conventions.

    Shared by both the MoE (`Qwen3_5MoeForConditionalGeneration` ->
    `Qwen3_5MoeForCausalLM`) and dense (`Qwen3_5ForConditionalGeneration` ->
    `Qwen3_5ForCausalLM`) VLM paths. The only difference between the two is the
    inner causal-LM arch string written onto `text_config`; everything else
    (mRoPE flattening, Qwen3Next text aliases, quantization exclude-module
    rewrite) is identical. `_normalize_qwen35_qwen3next_text_aliases` is a no-op
    for dense (its native `intermediate_size` is already present).
    """
    if not getattr(model_config, "architectures", None):
        model_config.architectures = [_INNER_TO_OUTER_VL_ARCH.get(inner_arch, inner_arch)]

    text_config = getattr(model_config, "text_config", None)
    if text_config is None:
        raise ValueError("Qwen3.5 VLM config is missing text_config")

    text_config.architectures = [inner_arch]
    _normalize_qwen35_qwen3next_text_aliases(text_config)
    _normalize_qwen35_mrope_config(text_config)

    model_config.get_text_config = lambda decoder=False: text_config
    _normalize_qwen35_quantization_config(model_config)


def _normalize_qwen35_moe_vl_config(model_config) -> None:
    """Adapt HF Qwen3.5-MoE VLM config to TRT-LLM runtime conventions."""
    _normalize_qwen35_vl_config(model_config, inner_arch="Qwen3_5MoeForCausalLM")


def _lm_head_nvfp4_enabled(model_config):
    """Whether the checkpoint's quantized lm_head should stay quantized.

    ModelOpt MIXED_PRECISION exports for Qwen3.5/3.6 quantize lm_head to
    W4A16_NVFP4 (packed FP4 weight + per-group FP8 scales).  On SM100/103 the
    NVFP4 (W4A4) Linear path can consume it directly, cutting the lm_head
    GEMM's weight traffic 4x vs the bf16 dequant fallback -- the decode
    lm_head is purely weight-bandwidth-bound.  Conditions mirror what the
    quantized LMHead supports (see LMHead.__init__ guards) plus the paths
    that bypass the Linear machinery entirely:

    - tie_word_embeddings shares the weight with the embedding lookup, which
      needs a dense bf16 weight;
    - ADP builds a TP-less LMHead (or slices the raw weight for the
      spec-decoding head), assuming bf16;
    - COLUMN TP pads the vocab shard when it doesn't divide evenly, which the
      quantized path does not support (vocab 248320 divides all common tp).

    Must be evaluated BEFORE _normalize_qwen35_quant_config_dict runs (it
    promotes the entry to NVFP4 or drops it).
    """
    qcd = getattr(model_config, "quant_config_dict", None) or {}
    cfg = qcd.get("lm_head")
    pretrained = model_config.pretrained_config
    mapping = model_config.mapping
    return (
        cfg is not None
        and cfg.quant_algo == QuantAlgo.W4A16_NVFP4
        and get_sm_version() in (100, 103)
        and not getattr(pretrained, "tie_word_embeddings", False)
        and not mapping.enable_attention_dp
        and getattr(pretrained, "vocab_size", 0) % mapping.tp_size == 0
    )


def _normalize_qwen35_exclude_modules(model_config, keep_lm_head_quant=False):
    """Normalize NVFP4/FP8 exclude_modules from HF naming to TRT-LLM naming.

    hf_quant_config.json stores exclude patterns in HF checkpoint namespace
    (e.g. ``model.language_model.layers.0.linear_attn*`` and ``mtp.layers.0*``),
    but TRT-LLM modules use ``model.layers.0.linear_attn.in_proj_qkvz`` and
    map the MTP layer to ``model.layers.<num_hidden_layers>.*``.  This
    function translates the patterns so that
    ``apply_quant_config_exclude_modules`` can match them.

    ``keep_lm_head_quant`` (see _lm_head_nvfp4_enabled) skips the lm_head
    force-exclusion so the quantized LMHead path can engage.
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

    # By default LMHead allocates an unquantized (bf16) weight, so a quantized
    # lm_head (e.g. NVFP4 in some ModelOpt MIXED_PRECISION exports) must be
    # excluded from quant and the weight mapper dequantizes it to bf16.  When
    # the quantized LMHead path is enabled (_lm_head_nvfp4_enabled), lm_head
    # must NOT be excluded so DecoderModelForCausalLM builds it quantized.
    if not keep_lm_head_quant:
        normalized.add("lm_head")

    qc.exclude_modules = sorted(normalized)


def _normalize_qwen35_quant_config_dict(model_config, keep_lm_head_quant=False):
    """Normalize MIXED_PRECISION per-layer quant config keys from HF naming.

    ModelOpt MIXED_PRECISION checkpoints key ``quant_config_dict`` by HF names
    (``model.language_model.layers.N...``, ``mtp.layers.N...``), but
    ``apply_layerwise_quant_config`` matches them against TRT-LLM module names
    (``model.layers.N...``).  Without this translation the per-layer entries
    never match, so quantized modules (MoE experts, shared_expert, attention,
    linear_attn.out_proj) silently fall back to the MIXED_PRECISION global
    config -> unquantized, and their quantized checkpoint weights fail to load.

    On SM100/SM103, W4A16_NVFP4 routed experts AND dense MLP projections
    (gate_proj/up_proj/down_proj) are promoted to NVFP4 so the CuteDSL/TRTLLM
    GEMM path can consume the checkpoint's packed FP4 weights and static input
    scales. Dense MLP keys are additionally re-pathed to the doubled
    ``.mlp.mlp.`` form to match the ``_DenseMlpAdapter`` runtime module tree.
    Other W4A16_NVFP4 modules retain their original algorithm.

    Mutates ``quant_config_dict`` in place (model_config is frozen).

    Split linear-attn in_proj projections (in_proj_qkv/in_proj_z or fully
    split q/k/v/z) are fused into a single ``in_proj_qkvz`` Linear at runtime.
    When the checkpoint quantizes every one of them per-tensor FP8, a single
    FP8 entry is synthesized under the fused module name so the Linear is
    built FP8; the weight mapper then requantizes the split weights onto one
    shared scale (_requantize_linear_attn_fp8_qkvz).  Incomplete or non-FP8
    sets get no fused entry, and the mapper dequantizes them to bf16 instead.

    The ``lm_head`` entry is promoted W4A16_NVFP4 -> NVFP4 when
    ``keep_lm_head_quant`` (see _lm_head_nvfp4_enabled) and dropped otherwise:
    a leftover entry would make DecoderModelForCausalLM build a quantized
    LMHead whose weights the mapper had already dequantized to bf16.
    """
    qcd = getattr(model_config, "quant_config_dict", None)
    if not qcd:
        return

    n_hidden_layers = getattr(model_config.pretrained_config, "num_hidden_layers", None)
    convert_to_nvfp4 = get_sm_version() in (100, 103)

    normalized = {}
    in_proj_fp8 = {}
    for name, cfg in qcd.items():
        if name.startswith(_LANG_PREFIX):
            name = "model." + name[len(_LANG_PREFIX) :]
        if name.startswith("model.visual"):
            continue
        if name == "lm_head":
            if keep_lm_head_quant:
                normalized[name] = cfg.model_copy(update={"quant_algo": QuantAlgo.NVFP4})
            else:
                # Make the fallback visible: the checkpoint quantizes lm_head
                # but this configuration can't keep it quantized (see
                # _lm_head_nvfp4_enabled), so it dequantizes to bf16 at load.
                logger.info(
                    f"lm_head quant entry ({cfg.quant_algo}) dropped: "
                    "unsupported configuration for quantized LMHead "
                    "(requires SM100/103, untied embeddings, no attention-DP, "
                    "vocab divisible by tp_size); lm_head runs bf16"
                )
            continue
        from_mtp = name.startswith("mtp.")
        if from_mtp:
            if n_hidden_layers is None:
                continue
            translated = _translate_mtp_pattern(name, n_hidden_layers)
            if translated is None:
                continue
            name = translated
        # Collect split linear-attn in_proj FP8 entries for fusion below.
        # MTP-origin entries are excluded: the weight mapper requantizes by
        # checkpoint prefix (mtp.layers.N), which never matches the translated
        # model.layers.{offset+N} key, so a fused entry would build an FP8
        # module whose weights load as bf16.
        in_proj_match = re.search(r"^(.+\.linear_attn)\.in_proj_(qkv|q|k|v|z)$", name)
        if in_proj_match and not from_mtp and cfg.quant_algo == QuantAlgo.FP8:
            in_proj_fp8.setdefault(in_proj_match.group(1), {})[in_proj_match.group(2)] = cfg
            continue
        if (
            convert_to_nvfp4
            and name.endswith(".mlp.experts")
            and cfg.quant_algo == QuantAlgo.W4A16_NVFP4
        ):
            cfg = cfg.model_copy(update={"quant_algo": QuantAlgo.NVFP4})
        else:
            # Dense MLP (gate_proj/up_proj/down_proj directly under ``.mlp``) is
            # wrapped by ``_DenseMlpAdapter``, so the runtime module path has a
            # doubled ``.mlp.mlp.`` segment
            # (layer.mlp[adapter].mlp[GatedMLP].{gate_up_proj,down_proj}).
            # Translate the per-layer key to that path so
            # ``apply_layerwise_quant_config`` matches it; otherwise the dense
            # MLP silently falls back to the global MIXED_PRECISION config and
            # its quantized checkpoint weights fail to load. On SM100/SM103 also
            # promote W4A16_NVFP4 -> NVFP4 so the CuteDSL/TRTLLM GEMM path can
            # consume the checkpoint's packed FP4 weights and static input scales.
            dense_mlp_match = re.search(r"\.mlp\.(gate_proj|up_proj|down_proj)$", name)
            if dense_mlp_match and cfg.quant_algo == QuantAlgo.W4A16_NVFP4:
                if convert_to_nvfp4:
                    cfg = cfg.model_copy(update={"quant_algo": QuantAlgo.NVFP4})
                proj = dense_mlp_match.group(1)
                name = name[: -len(dense_mlp_match.group(0))] + f".mlp.mlp.{proj}"
        normalized[name] = cfg

    # Synthesize one FP8 entry per fused in_proj_qkvz whose split projections
    # are all FP8; the weight mapper requantizes them onto a shared scale.
    for prefix, projs in in_proj_fp8.items():
        if set(projs) in ({"qkv", "z"}, {"q", "k", "v", "z"}):
            normalized[f"{prefix}.in_proj_qkvz"] = next(iter(projs.values()))

    qcd.clear()
    qcd.update(normalized)


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
        keep_lm_head_quant = _lm_head_nvfp4_enabled(model_config)
        _normalize_qwen35_exclude_modules(model_config, keep_lm_head_quant=keep_lm_head_quant)
        _normalize_qwen35_quant_config_dict(model_config, keep_lm_head_quant=keep_lm_head_quant)
        super().__init__(model_config)


@register_auto_model("Qwen3_5ForCausalLM")
class Qwen3_5ForCausalLM(Qwen3NextForCausalLM):
    """Thin wrapper for dense (non-MoE) Qwen3.5 text architecture.

    Same reuse pattern as Qwen3_5MoeForCausalLM, but for the dense 27B
    variant which uses GatedMLP instead of SparseMoeBlock.  The config
    normalizer (Qwen35ConfigCompat) sets num_experts=0 so that
    Qwen3NextModel selects GatedMLP for the feed-forward layers.
    """

    def __init__(self, model_config):
        keep_lm_head_quant = _lm_head_nvfp4_enabled(model_config)
        _normalize_qwen35_exclude_modules(model_config, keep_lm_head_quant=keep_lm_head_quant)
        _normalize_qwen35_quant_config_dict(model_config, keep_lm_head_quant=keep_lm_head_quant)
        super().__init__(model_config)


# Shared placeholder metadata for both Qwen3.5 VLM variants. The image/video
# placeholder layout is identical for MoE and dense; only the registration
# `model_type` differs (set per concrete class below).
_QWEN3_5_VL_PLACEHOLDER_METADATA = MultimodalPlaceholderMetadata(
    placeholder_map={
        "image": "<|vision_start|><|image_pad|><|vision_end|>",
        "video": "<|vision_start|><|video_pad|><|vision_end|>",
    },
    placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    placeholders_separator="",
    content_format=ContentFormat.STRING,
)


class _Qwen3_5VLModel(Qwen3VLModelBase):
    """Shared VLM wrapper composing the Qwen3 vision encoder with a Qwen3.5
    (Qwen3Next-based) text decoder.

    MoE and dense differ only in the inner causal-LM the config normalizer
    selects (`Qwen3_5MoeForCausalLM` vs `Qwen3_5ForCausalLM`) — both reuse the
    same vision tower, weight mapper, and forward path, so the wrapper body is
    shared here. The concrete subclasses below carry only the registration
    decorators (outer arch string + input-processor `model_type`).
    """

    @classmethod
    def get_model_defaults(cls, llm_args):
        # `ModelLoader` applies `get_model_defaults()` on the resolved outer
        # model class (this VLM wrapper), not on the inner decoder. Both
        # inner LMs (`Qwen3_5MoeForCausalLM` / `Qwen3_5ForCausalLM`) inherit
        # `Qwen3NextForCausalLM`'s defaults unchanged, so delegate to it to
        # propagate `enable_block_reuse=False` — the hybrid Mamba/SSM path
        # doesn't support KV-cache block reuse. Without this the VLM path
        # would silently fall back to the global default (block reuse on).
        return Qwen3NextForCausalLM.get_model_defaults(llm_args)

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        kwargs["vision_model_class"] = Qwen3VisionModel
        kwargs["disable_fuse_rope"] = kwargs.get("disable_fuse_rope", False)
        super().__init__(model_config, *args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values",
            "video.pixel_values_videos",
            "multimodal_embedding",
            "mrope_config.mrope_position_ids",
            "mrope_config.mrope_position_deltas",
        ]

    def load_weights(self, weights: Dict[str, torch.Tensor], weight_mapper: BaseWeightMapper):
        if not _is_mm_disagg():
            self.mm_encoder.load_weights(weights)

        weight_mapper = Qwen3_5MoeHfWeightMapper()
        weight_mapper.init_model_and_config(self.llm, self.model_config)
        filtered_weights = {k: v for k, v in weights.items() if not k.startswith("model.visual.")}
        params_map = {
            r"^model\.language_model\.(.*)$": r"model.\1",
        }
        self.llm.load_weights(filtered_weights, weight_mapper, params_map=params_map)


# TODO(TRTLLM-13417): Add tests for disaggregated support.
@support_multimodal_disaggregated
@register_vision_encoder(Qwen3VisionModelBase, vlm_base_model=Qwen3VisionModel)
@register_auto_model("Qwen3_5MoeForConditionalGeneration")
@register_input_processor(
    Qwen3VLInputProcessorBase,
    model_type="qwen3_5_moe",
    placeholder_metadata=_QWEN3_5_VL_PLACEHOLDER_METADATA,
)
class Qwen3_5MoeVLModel(_Qwen3_5VLModel):
    """VLM wrapper composing Qwen3 vision encoder with Qwen3.5 MoE text decoder."""


# TODO(TRTLLM-13417): Add tests for disaggregated support.
@support_multimodal_disaggregated
@register_vision_encoder(Qwen3VisionModelBase, vlm_base_model=Qwen3VisionModel)
@register_auto_model("Qwen3_5ForConditionalGeneration")
@register_input_processor(
    Qwen3VLInputProcessorBase,
    model_type="qwen3_5",
    placeholder_metadata=_QWEN3_5_VL_PLACEHOLDER_METADATA,
)
class Qwen3_5VLModel(_Qwen3_5VLModel):
    """VLM wrapper composing Qwen3 vision encoder with dense Qwen3.5 text decoder.

    Dense sibling of `Qwen3_5MoeVLModel` (arch `Qwen3_5ForConditionalGeneration`,
    `model_type="qwen3_5"`). Same hybrid Qwen3Next runtime, with `GatedMLP`
    instead of `SparseMoeBlock` (the dense text config has a native `intermediate_size`
    and no `num_experts`).
    """
