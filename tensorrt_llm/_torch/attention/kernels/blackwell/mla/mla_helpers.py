# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Compatibility imports for the shared Blackwell MLA helpers."""

from tensorrt_llm._torch.kernels.blackwell.mla_helpers import (
    LOG2_E,
    MAX_SPLITS,
    MLAStaticTileScheduler,
    MLAStaticTileSchedulerParams,
    WorkTileInfo,
    ceil_div,
    compute_q_tile_layout,
    create_mla_static_tile_scheduler,
    create_mla_static_tile_scheduler_params,
    get_variable_query_tile_info,
)

__all__ = [
    "LOG2_E",
    "MAX_SPLITS",
    "MLAStaticTileScheduler",
    "MLAStaticTileSchedulerParams",
    "WorkTileInfo",
    "ceil_div",
    "compute_q_tile_layout",
    "create_mla_static_tile_scheduler",
    "create_mla_static_tile_scheduler_params",
    "get_variable_query_tile_info",
]
