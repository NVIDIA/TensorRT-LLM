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

"""Routing-control package.

This package implements the planning + materialisation + injection pipeline for
advanced routing-control benchmarks.
Submodules split the pipeline into cohesive stages:

* :mod:`bench_moe.routing.parsing`       -- CLI-style pattern spec parsing
* :mod:`bench_moe.routing.builders`      -- plan types and construction
* :mod:`bench_moe.routing.materialize`   -- selected-experts realisation
* :mod:`bench_moe.routing.native_logits` -- native logits projection and
  supplied-topk forward patches

The package re-exports every public symbol; the historical
``bench_moe.<helper>`` import paths continue to work via the package
entrypoint re-export.
"""

from .builders import (
    RoutingPlan,
    RoutingProjectionResult,
    _build_dispatch_matrix,
    _build_expert_histogram,
    _build_per_rank_num_tokens,
    _build_routing_plan,
    _largest_remainder_split,
    _load_2d_matrix_json,
    _load_dispatch_matrix_file,
    _load_expert_histogram_file,
    _load_routing_pattern_file,
    _per_rank_tokens,
)
from .materialize import (
    _flatten_plan_slots_for_rank,
    _make_uniform_topk_scales,
    _materialize_selected_experts_for_rank,
    _observe_routing_metrics,
    _observe_summary,
    _pack_slots_column_major,
    _repair_duplicate_experts,
    _split_slot_count_to_experts,
)
from .native_logits import (
    _NATIVE_PROJECTION_CAPABILITIES,
    _align_topk_to_batch,
    _classify_native_projection,
    _make_supplied_topk_apply,
    _make_supplied_topk_run_moe,
    _maybe_install_routing_control_patch,
    _project_router_logits_for_plan,
)
from .parsing import (
    _COMM_PATTERN_NAMES,
    _EXPERT_PATTERN_NAMES,
    _parse_comm_pattern,
    _parse_expert_pattern,
    _parse_pattern_spec,
    _parse_typed_pattern,
    _pop_hotness_kwarg,
)

__all__ = [
    "RoutingPlan",
    "RoutingProjectionResult",
    "_COMM_PATTERN_NAMES",
    "_EXPERT_PATTERN_NAMES",
    "_NATIVE_PROJECTION_CAPABILITIES",
    "_align_topk_to_batch",
    "_build_dispatch_matrix",
    "_build_expert_histogram",
    "_build_per_rank_num_tokens",
    "_build_routing_plan",
    "_classify_native_projection",
    "_flatten_plan_slots_for_rank",
    "_largest_remainder_split",
    "_load_2d_matrix_json",
    "_load_dispatch_matrix_file",
    "_load_expert_histogram_file",
    "_load_routing_pattern_file",
    "_make_supplied_topk_apply",
    "_make_supplied_topk_run_moe",
    "_make_uniform_topk_scales",
    "_materialize_selected_experts_for_rank",
    "_maybe_install_routing_control_patch",
    "_observe_routing_metrics",
    "_observe_summary",
    "_pack_slots_column_major",
    "_parse_comm_pattern",
    "_parse_expert_pattern",
    "_parse_pattern_spec",
    "_parse_typed_pattern",
    "_per_rank_tokens",
    "_pop_hotness_kwarg",
    "_project_router_logits_for_plan",
    "_repair_duplicate_experts",
    "_split_slot_count_to_experts",
]
