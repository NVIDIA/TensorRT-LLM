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

"""Pure matching policy shared by perf-sanity code and unit tests."""

from collections.abc import Iterable, Mapping

_SCENARIO_SERVER_MATCH_KEYS = (
    "s_model_name",
    "l_gpus",
)

_CONFIG_SERVER_MATCH_KEYS = (
    "s_model_name",
    "l_tp",
    "l_ep",
    "l_pp",
    "l_cp",
    "l_gpus_per_node",
    "l_max_batch_size",
    "b_enable_attention_dp",
    "s_serving_backend",
    "s_kv_cache_dtype",
    "s_cache_transceiver_backend",
    "s_spec_decoding_type",
    "l_num_nextn_predict_layers",
    "l_force_num_accepted_tokens",
    "l_load_balancer_num_slots",
)

_CLIENT_MATCH_KEYS = (
    "l_concurrency",
    "l_iterations",
    "l_isl",
    "l_osl",
    "d_random_range_ratio",
    "s_backend",
    "b_use_chat_template",
    "b_streaming",
    "b_use_nv_sa_benchmark",
    "b_eos",
)


def get_server_match_keys(match_mode: str) -> list[str]:
    """Return database fields that identify an equivalent server run."""
    if match_mode == "scenario":
        return list(_SCENARIO_SERVER_MATCH_KEYS)
    return list(_CONFIG_SERVER_MATCH_KEYS)


def get_client_match_keys() -> list[str]:
    """Return database fields that identify an equivalent client workload."""
    return list(_CLIENT_MATCH_KEYS)


def benchmark_data_matches(
    history_data: Mapping[str, object],
    new_data: Mapping[str, object],
    match_keys: Iterable[str],
) -> bool:
    """Return whether historical and new benchmark data match on all requested fields."""

    def is_empty(value: object) -> bool:
        return value is None or value == ""

    for field in match_keys:
        history_value = history_data.get(field)
        new_value = new_data.get(field)
        # Missing boolean fields represent the historical default of False.
        if field.startswith("b_"):
            if history_value is None:
                history_value = False
            if new_value is None:
                new_value = False
        if is_empty(history_value) and is_empty(new_value):
            continue
        if history_value != new_value:
            return False
    return True
