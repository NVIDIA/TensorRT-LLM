# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BFL pre-quantized FLUX.3 transformers in a checkpoint's ``variants/<variant>/`` directory.

Each holds BFL-named weights and scales in model.safetensors, a BFL config.json and an
hf_quant_config.json, which is translated here to a ModelOpt-style quantization_config.
"""

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from safetensors import safe_open
from torch import Tensor

from tensorrt_llm._torch.visual_gen.checkpoints import WeightLoader
from tensorrt_llm.mapping import Mapping

from .bfl_keys import bfl_to_diffusers_module, bfl_to_diffusers_state_dict

# Quantization algorithm of each variant, as its hf_quant_config.json names it.
VARIANT_QUANT_ALGOS = {"nvfp4": "NVFP4", "fp8r": "FP8_PER_CHANNEL_PER_TOKEN"}
# Tensors each quantized Linear of a variant carries besides its weight.
VARIANT_SCALES = {
    "nvfp4": ("weight_scale", "weight_scale_2", "input_scale"),
    "fp8r": ("weight_scale",),
}
# NVFP4 activations use the calibrated input_scale; FP8 rowwise quantizes each token at runtime.
_DYNAMIC_ACTIVATIONS = {"nvfp4": False, "fp8r": True}
_SCALE_SUFFIXES = sorted({suffix for scales in VARIANT_SCALES.values() for suffix in scales})
NVFP4_GROUP_SIZE = 16


def check_variant(variant: Any) -> Optional[str]:
    """Return ``pipeline_config.variant`` if it is None or a known variant; else ValueError."""
    if variant is not None and (not isinstance(variant, str) or variant not in VARIANT_QUANT_ALGOS):
        raise ValueError(
            "FLUX.3 pipeline_config.variant must be null or one of "
            f"{sorted(VARIANT_QUANT_ALGOS)}, got {variant!r}."
        )
    return variant


def variant_dir(checkpoint_dir: str, variant: str) -> str:
    return os.path.join(checkpoint_dir, "variants", variant)


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise ValueError(f"FLUX.3 variant file not found: {path}")
    with open(path) as f:
        return json.load(f)


def _format_names(names: List[str], limit: int = 10) -> str:
    return f"{len(names)} [{', '.join(names[:limit])}{', ...' if len(names) > limit else ''}]"


def collapse_exclusions(excluded: Iterable[str], modules: Iterable[str]) -> List[str]:
    """Name each excluded module by its highest ancestor whose ``modules`` are all excluded.

    ``modules`` lists every Linear, so the returned names exclude exactly ``excluded``.
    """
    modules, excluded = set(modules), set(excluded)
    unknown = sorted(excluded - modules)
    if unknown:
        raise ValueError(f"FLUX.3 variant excludes modules it does not have: {unknown}")
    names = set()
    for name in excluded:
        parts = name.split(".")
        for depth in range(1, len(parts) + 1):
            ancestor = ".".join(parts[:depth])
            if all(
                module in excluded
                for module in modules
                if module == ancestor or module.startswith(ancestor + ".")
            ):
                names.add(ancestor)
                break
    return sorted(names)


def variant_quantization_config(
    variant: str, hf_quant_config: Dict[str, Any], tensor_names: Iterable[str]
) -> Dict[str, Any]:
    """ModelOpt-style quantization_config of a variant with static weights.

    BFL excludes become Diffusers module names via :func:`collapse_exclusions` over the Linears
    whose weights are among the BFL ``tensor_names``.
    """
    quantization = hf_quant_config.get("quantization") or {}
    quant_algo = VARIANT_QUANT_ALGOS[variant]
    if quantization.get("quant_algo") != quant_algo:
        raise ValueError(
            f"FLUX.3 {variant} variant needs quant_algo {quant_algo} in hf_quant_config.json, "
            f"got {quantization.get('quant_algo')!r}."
        )
    weights = {"dynamic": False}
    if variant == "nvfp4":
        group_size = quantization.get("group_size", NVFP4_GROUP_SIZE)
        if group_size != NVFP4_GROUP_SIZE:
            raise ValueError(
                f"FLUX.3 nvfp4 variant needs group_size {NVFP4_GROUP_SIZE}, got {group_size}."
            )
        weights["group_size"] = group_size
    modules = [
        bfl_to_diffusers_module(name[: -len(".weight")])
        for name in tensor_names
        if name.endswith(".weight")
    ]
    excluded = [bfl_to_diffusers_module(name) for name in quantization.get("exclude_modules", [])]
    return {
        "quant_algo": quant_algo,
        "config_groups": {
            "default": {
                "weights": weights,
                "input_activations": {"dynamic": _DYNAMIC_ACTIVATIONS[variant]},
            }
        },
        "ignore": collapse_exclusions(excluded, modules),
    }


def _check_variant_config(
    variant: str, bfl_config: Dict[str, Any], transformer_config: Dict[str, Any]
) -> None:
    """Reject a variant exported from another FLUX.3 model than the Diffusers transformer."""
    dit = bfl_config.get("dit") or {}
    heads = transformer_config.get("num_attention_heads")
    head_dim = transformer_config.get("attention_head_dim")
    expected = {
        "quantization": variant,
        "hidden": heads * head_dim if heads and head_dim else None,
        "heads": heads,
        "depth": transformer_config.get("num_layers"),
        "depth_single": transformer_config.get("num_single_layers"),
        "flow_map": bool(transformer_config.get("flow_map", False)),
    }
    actual = {
        "quantization": bfl_config.get("quantization"),
        **{key: dit.get(key) for key in ("hidden", "heads", "depth", "depth_single")},
        "flow_map": bool(dit.get("flow_map", False)),
    }
    mismatched = {
        key: (actual[key], value) for key, value in expected.items() if actual[key] != value
    }
    if mismatched:
        raise ValueError(
            f"FLUX.3 {variant} variant config.json does not match transformer/config.json, "
            f"as (variant, transformer): {mismatched}"
        )


def resolve_variant_quantization_config(
    checkpoint_dir: str, variant: str, transformer_config: Dict[str, Any]
) -> Dict[str, Any]:
    """quantization_config of ``variants/<variant>/`` for the Diffusers ``transformer_config``."""
    directory = variant_dir(checkpoint_dir, variant)
    _check_variant_config(
        variant, _read_json(os.path.join(directory, "config.json")), transformer_config
    )
    weights_path = os.path.join(directory, "model.safetensors")
    if not os.path.isfile(weights_path):
        raise ValueError(f"FLUX.3 variant file not found: {weights_path}")
    with safe_open(weights_path, framework="pt") as f:
        tensor_names = list(f.keys())
    hf_quant_config = _read_json(os.path.join(directory, "hf_quant_config.json"))
    return variant_quantization_config(variant, hf_quant_config, tensor_names)


def check_variant_scales(
    variant: str, weights: Dict[str, Tensor], quantized_modules: Dict[str, bool]
) -> None:
    """Raise unless each quantized module has every scale of ``variant`` and the others have none.

    ``quantized_modules`` maps Diffusers checkpoint module names to whether they load quantized.
    """
    missing, unexpected = [], []
    for module, quantized in sorted(quantized_modules.items()):
        if quantized:
            missing += [
                f"{module}.{suffix}"
                for suffix in VARIANT_SCALES[variant]
                if f"{module}.{suffix}" not in weights
            ]
        else:
            unexpected += [
                f"{module}.{suffix}"
                for suffix in _SCALE_SUFFIXES
                if f"{module}.{suffix}" in weights
            ]
    if missing or unexpected:
        raise ValueError(
            f"FLUX.3 {variant} weights do not match their quantization config: quantized layers "
            f"miss {_format_names(missing)}, BF16 layers carry {_format_names(unexpected)}"
        )


def load_variant_weights(
    checkpoint_dir: str, variant: str, quantized_modules: Dict[str, bool], mapping: Mapping
) -> Dict[str, Tensor]:
    """Weights of ``variants/<variant>/`` with Diffusers names, checked by check_variant_scales."""
    weights = WeightLoader().load_weights(variant_dir(checkpoint_dir, variant), mapping)
    weights = bfl_to_diffusers_state_dict(weights)
    check_variant_scales(variant, weights, quantized_modules)
    return weights
