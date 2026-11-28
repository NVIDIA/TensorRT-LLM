# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routed-expert statistics: the engine's EPLB routing-count hook, called once per routed call.

The engine records per-expert routing counts for the iterations ``EXPERT_STATISTIC_ITER_RANGE`` names (the input of
``examples/wide_ep/ep_load_balancer/generate_eplb_config.py``, which writes the static expert placement the DEP16
target runs with); without that variable the recorder does not exist and the call does nothing. This is a runtime
hook the engine owns, not model code: the family calls it from this one function only.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.moe.expert_statistic import ExpertStatistic


def record_expert_statistic(
    layer_idx: int, num_slots: int, token_selected_slots: torch.Tensor
) -> None:
    """Record one routed call's selected slots ``[rows, top_k]`` int32 for layer ``layer_idx`` (inert unless
    ``EXPERT_STATISTIC_ITER_RANGE`` is set)."""
    ExpertStatistic.set_layer(layer_idx)
    ExpertStatistic.maybe_add_info(num_slots, token_selected_slots)
