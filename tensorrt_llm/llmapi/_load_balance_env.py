# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configure process-start CUDA settings for opt-in MoE load balancing.

Keep this module free of GPU imports: it runs before argument validation and
worker initialization are allowed to probe CUDA.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .llm_args import MoeConfig


def configure_moe_launch_queues(
    moe_config: MoeConfig | None,
    env_overrides: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Configure the per-iteration EPLB launch queue before CUDA setup.

    ``moe_config`` must already be validated. Standard EPLB and configurations
    without a load balancer leave the environment and overrides untouched.
    CUDA has no supported operation to resize existing launch queues here.
    The late-init guard detects PyTorch initialization; external CUDA users
    must still arrange for this setting before initializing their contexts.
    """
    if moe_config is not None:
        moe_config.resolve_load_balancer_compatibility()
    load_balancer = getattr(moe_config, "load_balancer", None)
    if getattr(load_balancer, "mode", None) != "per_iteration":
        return env_overrides

    name = "CUDA_SCALE_LAUNCH_QUEUES"
    torch = sys.modules.get("torch")
    if os.environ.get(name) != "4x" and torch is not None and torch.cuda.is_initialized():
        raise RuntimeError(
            "Per-iteration EPLB requires CUDA_SCALE_LAUNCH_QUEUES=4x "
            "before CUDA initialization. Restart the process with that "
            "environment setting or construct the LLM before using CUDA."
        )

    overrides = dict(env_overrides or {})
    overrides[name] = "4x"
    os.environ[name] = "4x"
    return overrides
