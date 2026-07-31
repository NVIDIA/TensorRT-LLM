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

from test_common.perf_sanity_matching import (
    benchmark_data_matches,
    get_client_match_keys,
    get_server_match_keys,
)


def _benchmark_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "s_gpu_type": "b200",
        "s_runtime": "aggr_server",
        "s_model_name": "example_model",
        "l_gpus": 8,
        "l_tp": 8,
        "l_ep": 1,
        "l_max_batch_size": 32,
        "s_kv_cache_dtype": "fp8",
        "l_concurrency": 32,
        "l_iterations": 10,
        "l_isl": 1024,
        "l_osl": 1024,
    }
    data.update(overrides)
    return data


def _benchmark_match_keys(match_mode: str) -> list[str]:
    return [
        "s_gpu_type",
        "s_runtime",
        *get_server_match_keys(match_mode),
        *get_client_match_keys(),
    ]


def test_scenario_matching_ignores_recipe_tuning_changes() -> None:
    previous_data = _benchmark_data()
    updated_data = _benchmark_data(
        l_tp=4,
        l_ep=2,
        l_max_batch_size=64,
        s_kv_cache_dtype="auto",
    )
    match_keys = _benchmark_match_keys("scenario")

    assert get_server_match_keys("scenario") == ["s_model_name", "l_gpus"]
    assert {"l_concurrency", "l_isl", "l_osl"}.issubset(match_keys)
    assert benchmark_data_matches(previous_data, updated_data, match_keys)


def test_config_matching_keeps_recipe_tuning_keys() -> None:
    previous_data = _benchmark_data()
    updated_data = _benchmark_data(
        l_tp=4,
        l_ep=2,
        l_max_batch_size=64,
        s_kv_cache_dtype="auto",
    )

    assert not benchmark_data_matches(
        previous_data,
        updated_data,
        _benchmark_match_keys("config"),
    )


def test_scenario_matching_distinguishes_gpu_count() -> None:
    previous_data = _benchmark_data()
    updated_data = _benchmark_data(l_gpus=4)

    assert not benchmark_data_matches(
        previous_data,
        updated_data,
        _benchmark_match_keys("scenario"),
    )


def test_scenario_matching_distinguishes_client_workload() -> None:
    previous_data = _benchmark_data()
    updated_data = _benchmark_data(l_concurrency=64)

    assert not benchmark_data_matches(
        previous_data,
        updated_data,
        _benchmark_match_keys("scenario"),
    )
