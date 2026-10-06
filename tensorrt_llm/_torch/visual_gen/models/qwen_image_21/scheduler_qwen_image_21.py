# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-2.1 scheduler compatibility exports.

Qwen-Image-2.1 uses the FlowMatch Euler scheduler declared by the checkpoint
(``FlowMatchEulerDiscreteScheduler``).  The runtime implementation lives in
``ops.py`` as TRTLLM-owned code so candidate E2E does not import an upstream
Diffusers scheduler at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import DEFAULT_NUM_INFERENCE_STEPS, resolve_num_inference_steps
from .ops import QwenImage21FlowMatchEulerScheduler


@dataclass(frozen=True)
class QwenImage21SchedulerConfig:
    """Configuration for the native TRTLLM FlowMatch Euler scheduler."""

    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    subfolder: str = "scheduler"
    class_name: str = "QwenImage21FlowMatchEulerScheduler"

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object] | None = None) -> "QwenImage21SchedulerConfig":
        return cls(num_inference_steps=resolve_num_inference_steps(manifest))


def load_qwen_image_21_scheduler(
    pretrained_model_name_or_path: str | Path,
    *,
    config: QwenImage21SchedulerConfig | None = None,
    **kwargs: Any,
) -> QwenImage21FlowMatchEulerScheduler:
    """Load the native scheduler from a Diffusers-format checkpoint folder.

    Extra keyword arguments are accepted for API compatibility with earlier
    helper revisions, but are intentionally ignored because this implementation
    never reaches out to the HuggingFace Hub or Diffusers runtime.
    """

    del kwargs
    config = config or QwenImage21SchedulerConfig()
    scheduler = QwenImage21FlowMatchEulerScheduler.from_pretrained(
        str(pretrained_model_name_or_path), subfolder=config.subfolder
    )
    # Do not call set_timesteps here: the Qwen-Image-2.1 denoise loop computes
    # the dynamic shift ``mu`` from the runtime latent sequence length first.
    return scheduler


__all__ = [
    "QwenImage21FlowMatchEulerScheduler",
    "QwenImage21SchedulerConfig",
    "load_qwen_image_21_scheduler",
]
