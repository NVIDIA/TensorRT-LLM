# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fnmatch
import math
import numbers
from dataclasses import dataclass

import torch

from tensorrt_llm._torch.visual_gen.cuda_graph_runner import resolved_extra_key
from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.sparse_attention import SkipSoftmaxAttentionConfig


def dense_prefix(timestep: object, cutoff: float | None) -> bool:
    if cutoff is None or timestep is None:
        return False
    if isinstance(timestep, torch.Tensor):
        if timestep.numel() == 0:
            return False
        timestep = timestep.amax().item()
    if isinstance(timestep, bool) or not isinstance(timestep, numbers.Real):
        raise TypeError("timestep must be a real scalar or tensor")
    if not math.isfinite(timestep):
        raise ValueError("timestep must be finite")
    return timestep >= cutoff


def _skip_group(config: dict) -> dict:
    groups = config.get("config_groups")
    if not isinstance(groups, dict):
        return config
    matches = [
        group
        for group in groups.values()
        if isinstance(group, dict)
        and group.get("algorithm", group.get("sparse_algo")) in ("skip_softmax", "softmax_skip")
    ]
    if len(matches) > 1:
        raise ValueError("Expected at most one skip-softmax group in the checkpoint")
    return matches[0] if matches else {}


def _calibrated_threshold(config: dict, group: dict, target: float) -> float:
    import numexpr

    calibration = group.get("threshold_scale_factor")
    if not isinstance(calibration, dict):
        calibration = config if "formula" in config else {}
    nested = calibration.get("coefficients")
    scalars = nested if isinstance(nested, dict) and nested else calibration
    expression = calibration.get("formula") or scalars.get("formula")
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Skip-softmax target_sparsity requires a checkpoint calibration formula")
    coefficients = {
        k: float(v) for k, v in scalars.items() if k not in ("target_sparsity", "formula")
    }
    if not coefficients:
        raise ValueError("Skip-softmax calibration requires coefficient values")
    try:
        names = set(numexpr.NumExpr(expression).input_names)
    except (TypeError, ValueError, SyntaxError, KeyError) as error:
        raise ValueError(f"Invalid skip-softmax formula {expression!r}") from error
    if "target_sparsity" not in names or not names.issubset(
        coefficients.keys() | {"target_sparsity"}
    ):
        raise ValueError(
            "Skip-softmax formula must reference target_sparsity and the supplied coefficients"
        )
    return float(
        numexpr.evaluate(expression, local_dict={**coefficients, "target_sparsity": target}).item()
    )


@dataclass(frozen=True)
class H3SkipSoftmax:
    threshold: float
    disabled_until_timestep: float | None

    @classmethod
    def from_config(cls, recipe: SkipSoftmaxAttentionConfig, name: str, pretrained_config):
        if isinstance(pretrained_config, dict):
            sparse = pretrained_config.get("sparse_attention_config")
            config = sparse if isinstance(sparse, dict) else pretrained_config
        else:
            config = getattr(pretrained_config, "sparse_attention_config", None)
            config = config if isinstance(config, dict) else {}
        group = _skip_group(config)
        names = {name, name.replace("._orig_mod.", ".")}
        for label in tuple(names):
            names.update(
                label.removeprefix(prefix) for prefix in ("transformer.", "transformer_2.")
            )
        for source in (config, group):
            ignored = source.get("ignore")
            if isinstance(ignored, list) and any(
                fnmatch.fnmatch(n, str(p)) for n in names for p in ignored
            ):
                return None

        threshold = recipe.threshold_scale_factor
        if threshold is None:
            target = recipe.target_sparsity
            if target is None:
                target = group.get("target_sparsity")
                if isinstance(target, dict):
                    raise ValueError(
                        "VisualGen skip-softmax checkpoint target_sparsity must be a scalar"
                    )
                if not isinstance(target, (float, int)):
                    return None
            threshold = _calibrated_threshold(config, group, float(target))
        if threshold <= 0:
            return None
        cutoff = recipe.disabled_until_timestep
        if cutoff is None:
            checkpoint_cutoff = group.get("disabled_until_timestep")
            if isinstance(checkpoint_cutoff, (int, float)):
                cutoff = float(checkpoint_cutoff)
        return cls(threshold, cutoff)


def resolve_skip_softmax_threshold_scale_factor(
    threshold_scale_factor: float | None, sparse_params: H3SkipSoftmax | None, timestep
) -> float | None:
    if sparse_params is not None:
        if sparse_params.disabled_until_timestep is None:
            threshold_scale_factor = sparse_params.threshold
        else:
            threshold_scale_factor = _scheduled_threshold(sparse_params, timestep)
    if threshold_scale_factor is None or threshold_scale_factor <= 0:
        return None
    return threshold_scale_factor


@torch.compiler.disable
def _scheduled_threshold(params: H3SkipSoftmax, timestep) -> float:
    if timestep is None:
        logger.warning_once(
            f"SkipSoftmax scheduler has disabled_until_timestep={params.disabled_until_timestep} "
            "configured, but no timestep was passed to CuTeDSL attention.",
            key="cute_dsl_skip_softmax_missing_timestep",
        )
    phase = resolved_extra_key("sparse_attn_phase")
    dense = (
        phase == 0 if phase is not None else dense_prefix(timestep, params.disabled_until_timestep)
    )
    return 0.0 if dense else params.threshold
