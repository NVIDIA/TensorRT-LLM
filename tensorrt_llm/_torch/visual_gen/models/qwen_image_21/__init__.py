# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-2.1 VisualGen model package.

The package intentionally exposes TRTLLM-owned classes plus small stateless
runtime helpers.  Exact Diffusers reuse is limited to the declared image-VAE
fallback inside the pipeline runtime and is recorded in the enablement run's
``design/external_component_reuse.yaml``.
"""

from importlib import import_module
from typing import Any

from .config import (
    DEFAULT_HF_MODEL_ID,
    DEFAULT_NUM_INFERENCE_STEPS,
    QwenImage21PipelineConfig,
    QwenImage21TransformerConfig,
    resolve_num_inference_steps,
)
from .ops import (
    QwenImage21FlowMatchEulerScheduler,
    append_target_slots,
    calculate_dimensions,
    calculate_shift,
    pack_latents,
    unpack_latents,
)

__all__ = [
    "DEFAULT_HF_MODEL_ID",
    "DEFAULT_NUM_INFERENCE_STEPS",
    "QwenImage21FlowMatchEulerScheduler",
    "QwenImage21Pipeline",
    "QwenImage21PipelineConfig",
    "QwenImage21SchedulerConfig",
    "QwenImage21TransformerConfig",
    "QwenImage21TransformerModel",
    "QwenImage21Transformer2DModel",
    "append_target_slots",
    "calculate_dimensions",
    "calculate_shift",
    "load_qwen_image_21_scheduler",
    "pack_latents",
    "resolve_num_inference_steps",
    "unpack_latents",
]

_LAZY_ATTRS = {
    "QwenImage21Pipeline": (".pipeline_qwen_image_21", "QwenImage21Pipeline"),
    # Backward-compatible alias retained for earlier tests/import probes.
    "QwenImage21TransformerModel": (
        ".transformer_qwen_image_21",
        "QwenImage21Transformer2DModel",
    ),
    "QwenImage21Transformer2DModel": (
        ".transformer_qwen_image_21",
        "QwenImage21Transformer2DModel",
    ),
    "QwenImage21SchedulerConfig": (".scheduler_qwen_image_21", "QwenImage21SchedulerConfig"),
    "load_qwen_image_21_scheduler": (
        ".scheduler_qwen_image_21",
        "load_qwen_image_21_scheduler",
    ),
}


def __getattr__(name: str) -> Any:
    """Lazily import heavy runtime modules on first access."""

    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
