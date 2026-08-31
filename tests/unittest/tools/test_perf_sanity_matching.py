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

from test_common.perf_sanity_matching import benchmark_data_matches, get_test_case_match_keys


def _benchmark_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "s_test_case_name": "example_model_fp8_tp8-con32_iter10_1k1k",
        "s_gpu_type": "b200",
        "s_runtime": "aggr_server",
        "s_branch": "main",
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


def test_match_keys_are_name_and_environment_only() -> None:
    assert get_test_case_match_keys() == [
        "s_test_case_name",
        "s_gpu_type",
        "s_runtime",
        "s_branch",
    ]


def test_matching_ignores_tuning_changes() -> None:
    previous_data = _benchmark_data()
    updated_data = _benchmark_data(
        l_max_batch_size=64,
        s_kv_cache_dtype="auto",
        l_iterations=20,
        l_force_num_accepted_tokens=3,
    )

    assert benchmark_data_matches(previous_data, updated_data, get_test_case_match_keys())


def test_matching_distinguishes_test_case_name() -> None:
    previous_data = _benchmark_data()
    updated_data = _benchmark_data(s_test_case_name="example_model_fp8_tp4-con32_iter10_1k1k")

    assert not benchmark_data_matches(previous_data, updated_data, get_test_case_match_keys())


def test_matching_distinguishes_gpu_type() -> None:
    previous_data = _benchmark_data()
    updated_data = _benchmark_data(s_gpu_type="gb200")

    assert not benchmark_data_matches(previous_data, updated_data, get_test_case_match_keys())


def test_matching_distinguishes_runtime() -> None:
    previous_data = _benchmark_data()
    updated_data = _benchmark_data(s_runtime="multi_node_aggr_server")

    assert not benchmark_data_matches(previous_data, updated_data, get_test_case_match_keys())


def test_matching_distinguishes_branch() -> None:
    previous_data = _benchmark_data()
    updated_data = _benchmark_data(s_branch="release/1.3.0")

    assert not benchmark_data_matches(previous_data, updated_data, get_test_case_match_keys())


def test_benchmark_mode_is_not_a_match_key() -> None:
    assert "s_benchmark_mode" not in get_test_case_match_keys()
    history = _benchmark_data(
        s_test_case_name="e2e-example_disagg-con32_iter10_1k1k",
        s_runtime="multi_node_disagg_server",
        s_benchmark_mode=None,
    )
    new = _benchmark_data(
        s_test_case_name="e2e-example_disagg-con32_iter10_1k1k",
        s_runtime="multi_node_disagg_server",
        s_benchmark_mode="e2e",
    )

    assert benchmark_data_matches(history, new, get_test_case_match_keys())


def test_a_pre_merge_branch_does_not_match_post_merge_history() -> None:
    """Branch is identity, so a PR run cannot match main's history unaided.

    This is the precondition that makes the baseline-branch substitution in
    process_and_upload_test_results necessary; the substitution itself is tested
    against that function in
    tests/unittest/others/test_perf_regression_branch.py.
    """
    history = _benchmark_data(s_branch="main")
    pre_merge_data = _benchmark_data(s_branch="github-pr-12345")

    assert not benchmark_data_matches(history, pre_merge_data, get_test_case_match_keys())
