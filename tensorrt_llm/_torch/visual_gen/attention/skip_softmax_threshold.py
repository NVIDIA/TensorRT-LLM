# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from tensorrt_llm._torch.attention.backends.sparse.skip_softmax import (
    SkipSoftmaxParams,
    SkipSoftmaxScheduler,
)
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import resolved_extra_key
from tensorrt_llm.logger import logger


def resolve_skip_softmax_threshold_scale_factor(
    threshold_scale_factor: float | None,
    sparse_params: SkipSoftmaxParams | None,
    timestep: Any,
) -> float | None:
    """Resolve the active CuTeDSL threshold for the current denoising phase."""
    if sparse_params is not None:
        scheduler = sparse_params.scheduler
        if scheduler.disabled_until_timestep is None:
            threshold_scale_factor = scheduler.threshold_scale_factor_prefill
        else:
            threshold_scale_factor = _scheduled_threshold(scheduler, timestep)
    if threshold_scale_factor is None or threshold_scale_factor <= 0.0:
        return None
    return threshold_scale_factor


@torch.compiler.disable
def _scheduled_threshold(scheduler: SkipSoftmaxScheduler, timestep: Any) -> float:
    """Read the host-resolved graph phase or evaluate the live timestep in eager mode."""
    if timestep is None:
        logger.warning_once(
            "SkipSoftmax scheduler has disabled_until_timestep="
            f"{scheduler.disabled_until_timestep} configured, but no "
            "`timestep` was passed to the CuTeDSL attention forward call. Skip-softmax "
            "will run unthrottled (as if past the cutoff) until `timestep` is threaded "
            "through.",
            key="cute_dsl_skip_softmax_missing_timestep",
        )
    runtime_params = scheduler.get_runtime_params(
        timestep=timestep,
        graph_phase=resolved_extra_key("sparse_attn_phase"),
    )
    return runtime_params.threshold_scale_factor_prefill
