# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared tuning, launch and profiling services for VisualGen catalog providers."""

from tensorrt_llm._torch.autotuner import AutoTuner, DynamicTensorSpec, TunableRunner, TuningConfig
from tensorrt_llm._torch.cute_dsl_utils import sm_reserve
from tensorrt_llm._torch.locality_domain_utils import (
    get_current_locality_domain,
    node_local_max_active_clusters,
)
from tensorrt_llm._torch.utils import (
    fp4_scale_infer_shape,
    get_last_power_of_2_num_tokens_buckets,
    last_positive_power_of_2,
)
from tensorrt_llm._utils import nvtx_range

__all__ = [
    "AutoTuner",
    "DynamicTensorSpec",
    "TunableRunner",
    "TuningConfig",
    "fp4_scale_infer_shape",
    "get_current_locality_domain",
    "get_last_power_of_2_num_tokens_buckets",
    "last_positive_power_of_2",
    "node_local_max_active_clusters",
    "nvtx_range",
    "sm_reserve",
]
