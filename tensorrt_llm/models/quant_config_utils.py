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
from collections.abc import Iterable, Mapping
from typing import Any

from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# compressed-tensors marks a ``targets``/``ignore`` entry as a regex with this
# prefix; every other entry is an exact module-name (or class-name) match.
_TARGET_REGEX_PREFIX = "re:"


def _matches_target(module_name: str, target: str) -> bool:
    """Whether ``module_name`` matches a compressed-tensors target/ignore entry.

    Mirrors ``compressed_tensors.utils.match.match_name``: ``re:``-prefixed
    entries are matched with ``re.match`` (anchored at the start only, *not*
    ``re.fullmatch``), everything else must equal the module name. Class-name
    targets (e.g. ``"Linear"``) are not resolvable here because they need the
    instantiated module tree, which does not exist at config-parse time.

    Args:
        module_name: Module name in the checkpoint (HF) namespace.
        target: One ``targets`` or ``ignore`` entry.

    Returns:
        True if the entry matches the module name.
    """
    if target.startswith(_TARGET_REGEX_PREFIX):
        return re.match(target.removeprefix(_TARGET_REGEX_PREFIX), module_name) is not None
    return target == module_name


def _ordered_targets(targets: Iterable[str]) -> list[str]:
    r"""Order ``config_groups`` targets by compressed-tensors match precedence.

    Mirrors ``compressed_tensors.utils.match.match_targets``: exact-name
    targets sort before ``re:`` targets, each block in ascending lexicographic
    order. ``_scheme_from_targets`` then takes the *first* matching target, so
    this ordering -- not pattern specificity -- decides which ``config_group``
    owns a module that several groups target.

    Qwen3.8-27B-NVFP4 depends on exactly this: its FP8 group targets
    ``re:.*layers\.(56|...|63)\.mlp\.(gate|up|down)_proj$`` while its NVFP4
    group targets ``re:.*mlp\.(gate|up|down)_proj$``. Both match the last
    eight MLP blocks, and ``layers`` sorts before ``mlp``, so those blocks are
    FP8.

    Args:
        targets: Target entries collected across all ``config_groups``.

    Returns:
        The targets, most-specific-first per compressed-tensors' ordering.
    """
    return sorted(targets, key=lambda target: (_TARGET_REGEX_PREFIX in target, target))


def _exclude_modules_from_hf_config(hf_quant_config: Mapping[str, Any]) -> list[str]:
    """Collect the modules the producer left unquantized.

    Args:
        hf_quant_config: ``quantization_config`` from the checkpoint's config.

    Returns:
        ``modules_to_not_convert`` merged with ``ignore``, order-preserving and
        deduplicated.
    """
    hf_exclude_modules = hf_quant_config.get("modules_to_not_convert", None)
    ignore = list(hf_quant_config.get("ignore", []))
    if hf_exclude_modules is None:
        return ignore
    return list(dict.fromkeys(list(hf_exclude_modules) + ignore))


def _quant_config_from_config_group(
    group_config: Mapping[str, Any], group_format: str | None
) -> QuantConfig:
    """Resolve one compressed-tensors ``config_group`` to a TRT-LLM QuantConfig.

    Only ``quant_algo`` and ``group_size`` are set; the KV-cache algorithm and
    the exclude-module list are global properties of the checkpoint and stay
    with the caller.

    Args:
        group_config: One entry of the checkpoint's ``config_groups``.
        group_format: The group's ``format``, falling back to the checkpoint's
            top-level ``format`` when the group does not declare one.

    Returns:
        A QuantConfig carrying the group's ``quant_algo`` and ``group_size``.

    Raises:
        ValueError: The group uses a bit width, strategy or group size that
            TRT-LLM does not support.
    """
    weights_quant_config = group_config["weights"]
    weights_quant_strategy = weights_quant_config["strategy"]

    # MXFP4 pack-quantized (weight-only): FP4 E2M1 weights packed two per
    # uint8 with per-32-group uint8 E8M0 scales and no activation
    # quantization (e.g. Kimi K3 routed experts). Handled before reading the
    # input-activation strategy, which is null for weight-only recipes.
    if group_format == "mxfp4-pack-quantized" or (
        weights_quant_config["num_bits"] == 4
        and weights_quant_config.get("type") == "float"
        and weights_quant_strategy == "group"
        and group_config.get("input_activations") is None
    ):
        group_size = weights_quant_config["group_size"]
        if group_size != 32:
            raise ValueError(f"Unsupported group_size: {group_size}. Supported: 32 for MXFP4.")
        return QuantConfig(quant_algo=QuantAlgo.W4A16_MXFP4, group_size=group_size)

    inputs_quant_config = group_config["input_activations"]
    inputs_quant_strategy = inputs_quant_config["strategy"]

    if weights_quant_config["num_bits"] == 8:
        if weights_quant_strategy == "channel":
            if inputs_quant_strategy != "token":
                raise ValueError(f"Unsupported inputs_quant_strategy: {inputs_quant_strategy}.")
            return QuantConfig(quant_algo=QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN)
        if weights_quant_strategy == "block":
            if inputs_quant_strategy != "group":
                raise ValueError(f"Unsupported inputs_quant_strategy: {inputs_quant_strategy}.")
            group_size = inputs_quant_config["group_size"]

            # TRT-LLM only supports group_size=128 for FP8_BLOCK_SCALES.
            if group_size != 128:
                raise ValueError(f"Unsupported group_size: {group_size}. Supported: 128.")
            return QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=group_size)

        raise ValueError(
            f"Unsupported weights_quant_strategy: {weights_quant_strategy}. "
            "Supported strategies: 'channel', 'block'."
        )

    if (
        weights_quant_config["num_bits"] == 4
        and weights_quant_config.get("type") == "float"
        and weights_quant_strategy == "tensor_group"
    ):
        # llm-compressor NVFP4 quantizes both weights and activations. Model
        # adapters must normalize module names independently of the algorithm;
        # SM120/121 execute this W4A4 contract through the FP4 CUTLASS path.
        if inputs_quant_strategy != "tensor_group":
            raise ValueError(
                f"Unsupported inputs_quant_strategy for NVFP4: {inputs_quant_strategy}."
            )
        group_size = weights_quant_config["group_size"]
        if group_size != 16:
            raise ValueError(f"Unsupported group_size: {group_size}. Supported: 16 for NVFP4.")
        return QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=group_size)

    raise ValueError(
        f"Unsupported quant_bits: {weights_quant_config['num_bits']}. "
        "Supported: 8 (FP8) or 4 (NVFP4)."
    )


def _build_layer_quant_configs(
    hf_quant_config: Mapping[str, Any],
    module_names: Iterable[str],
    kv_cache_quant_algo: QuantAlgo | None,
) -> dict[str, QuantConfig]:
    """Resolve a multi-group checkpoint to one QuantConfig per quantized module.

    Args:
        hf_quant_config: ``quantization_config`` from the checkpoint's config.
        module_names: Module names (HF namespace) present in the checkpoint.
        kv_cache_quant_algo: Global KV-cache algorithm, copied onto every
            per-module config the way the modelopt MIXED_PRECISION path does.

    Returns:
        QuantConfigs keyed by module name, covering exactly the modules the
        producer quantized. Modules absent from the mapping inherit the global
        ``MIXED_PRECISION`` config, i.e. they stay unquantized.

    Raises:
        ValueError: A ``config_group`` has no ``targets``, uses a scheme
            TRT-LLM does not support, or a target matched no checkpoint module.
    """
    top_level_format = hf_quant_config.get("format")
    target_quant_configs: dict[str, QuantConfig] = {}
    for group_name, group_config in hf_quant_config["config_groups"].items():
        targets = group_config.get("targets")
        if not targets:
            raise ValueError(
                f"config_group '{group_name}' has no 'targets'. Targets are required to "
                "resolve a compressed-tensors checkpoint with multiple config groups."
            )
        group_quant_config = _quant_config_from_config_group(
            group_config, group_config.get("format", top_level_format)
        )
        group_quant_config.kv_cache_quant_algo = kv_cache_quant_algo
        for target in targets:
            if target in target_quant_configs:
                logger.warning(
                    f"compressed-tensors target '{target}' appears in more than one "
                    "config group; the last group takes precedence."
                )
            target_quant_configs[target] = group_quant_config

    ignore = hf_quant_config.get("ignore", [])
    ordered_targets = _ordered_targets(target_quant_configs)

    layer_quant_configs: dict[str, QuantConfig] = {}
    matched_targets: set[str] = set()
    for module_name in module_names:
        matching_targets = [
            target for target in ordered_targets if _matches_target(module_name, target)
        ]
        matched_targets.update(matching_targets)
        if any(_matches_target(module_name, entry) for entry in ignore):
            continue
        if matching_targets:
            layer_quant_configs[module_name] = target_quant_configs[
                matching_targets[0]
            ].model_copy()

    unmatched_targets = sorted(set(ordered_targets) - matched_targets)
    if unmatched_targets:
        raise ValueError(
            f"config_groups targets matched no checkpoint module: {unmatched_targets}. "
            "compressed-tensors targets that select modules by class name are not supported."
        )
    return layer_quant_configs


def update_quant_config_from_compressed_tensors(
    quant_config: QuantConfig,
    hf_quant_config: Mapping[str, Any],
    module_names: Iterable[str] | None = None,
) -> dict[str, QuantConfig] | None:
    """Mutate QuantConfig from an llm-compressor compressed-tensors config.

    A checkpoint with a single ``config_group`` maps onto one global
    ``quant_algo``. A checkpoint with several groups (``"format":
    "mixed-precision"``) quantizes different modules differently and maps onto
    ``QuantAlgo.MIXED_PRECISION`` plus one QuantConfig per quantized module.
    Qwen3.8-27B-NVFP4 is the motivating example: ``group_0`` is FP8
    (attention, GDN projections, ``lm_head`` and the last eight MLP blocks) and
    ``group_1`` is NVFP4 (every other MLP block).

    Args:
        quant_config: Mutated in place with the checkpoint's global settings.
        hf_quant_config: ``quantization_config`` from the checkpoint's config.
        module_names: Module names (HF namespace) present in the checkpoint.
            Multi-group checkpoints need them to resolve ``config_groups``
            targets; single-group checkpoints ignore them.

    Returns:
        Per-module QuantConfigs keyed by module name for a multi-group
        checkpoint, or ``None`` when the checkpoint has a single group or
        ``module_names`` was not supplied.

    Raises:
        ValueError: ``config_groups`` is missing or empty, the KV-cache scheme
            is unsupported or conflicts with an explicit request, or a group
            uses a scheme TRT-LLM does not support.
    """
    config_groups = hf_quant_config.get("config_groups")
    if config_groups is None:
        raise ValueError(f"config_groups is not set in {hf_quant_config}.")
    if not config_groups:
        raise ValueError(f"config_groups is empty in {hf_quant_config}.")

    # kv_cache_scheme (llm-compressor): FP8 per-tensor KV cache. Handled
    # before the weight-algo branches so recipes that bail out early (MXFP4
    # pack-quantized) still pick up the KV-cache quantization.
    kv_cache_scheme = hf_quant_config.get("kv_cache_scheme")
    if kv_cache_scheme is not None:
        if kv_cache_scheme.get("num_bits") == 8 and kv_cache_scheme.get("type") == "float":
            if quant_config.kv_cache_quant_algo in (None, QuantAlgo.FP8):
                quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            else:
                raise ValueError(
                    f"Specified kv_cache_quant_algo={quant_config.kv_cache_quant_algo}, "
                    "conflicting with FP8 KV cache from HF quant config."
                )
        else:
            raise ValueError(f"Unsupported kv_cache_scheme: {kv_cache_scheme}.")

    exclude_modules = _exclude_modules_from_hf_config(hf_quant_config)

    if len(config_groups) > 1:
        quant_config.quant_algo = QuantAlgo.MIXED_PRECISION
        # The per-layer map is authoritative for a mixed checkpoint. Producer
        # ``ignore`` entries are non-recursive and have already been applied
        # while building that map; copying them into TRT-LLM's recursive
        # exclude list could incorrectly shadow quantized children.
        quant_config.exclude_modules = list(
            hf_quant_config.get("modules_to_not_convert", None) or []
        )
        if module_names is None:
            # The global algo is fully determined without the module list, but
            # the per-module configs are not. Callers that only need
            # ``quant_config`` (e.g. reporting the checkpoint's quantization
            # back through LlmArgs) are fine; a caller that builds the model
            # from this config must pass ``module_names``.
            logger.debug(
                f"compressed-tensors checkpoint has multiple config groups "
                f"({sorted(config_groups)}) but no checkpoint module names were supplied; "
                "per-layer quantization configs were not resolved."
            )
            return None
        layer_quant_configs = _build_layer_quant_configs(
            hf_quant_config, module_names, quant_config.kv_cache_quant_algo
        )
        return layer_quant_configs

    # compressed-tensors keys config_groups by group name: custom recipes use
    # "group_0"; named preset schemes (e.g. FP8_DYNAMIC) use the scheme name.
    group_config = next(iter(config_groups.values()))
    group_quant_config = _quant_config_from_config_group(
        group_config, group_config.get("format", hf_quant_config.get("format"))
    )
    quant_config.quant_algo = group_quant_config.quant_algo
    quant_config.group_size = group_quant_config.group_size
    quant_config.exclude_modules = exclude_modules
    return None
