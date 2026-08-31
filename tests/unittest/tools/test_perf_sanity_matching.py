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

import ast
import importlib.util
import pathlib
import sys
import types

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
    """Tunables do not fork a case *while the name is held constant*.

    That proviso is the whole story for l_iterations, which reaches the case name
    on the disagg path -- see test_iterations_fork_a_case_through_the_derived_name.
    """
    previous_data = _benchmark_data()
    updated_data = _benchmark_data(
        l_max_batch_size=64,
        s_kv_cache_dtype="auto",
        l_iterations=20,
        l_force_num_accepted_tokens=3,
    )

    assert benchmark_data_matches(previous_data, updated_data, get_test_case_match_keys())


def _load_client_config() -> type:
    """Return the real ClientConfig, without the integration-test packages.

    The derived-name rule is only worth testing against the class that owns it;
    re-implementing the f-string here would assert nothing. test_perf_sanity.py
    reaches torch and the OpenSearch client through its imports, so those are
    stubbed -- ClientConfig.__init__ touches none of them.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    module_path = repo_root / "tests" / "integration" / "defs" / "perf" / "test_perf_sanity.py"

    def stub(name: str, **attrs: object) -> types.ModuleType:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    def noop(*args: object, **kwargs: object) -> None:
        return None

    defs_pkg = stub("defs")
    defs_pkg.__path__ = []
    perf_pkg = stub("defs.perf")
    perf_pkg.__path__ = []
    stubs = {
        "defs": defs_pkg,
        "defs.perf": perf_pkg,
        "defs.common": stub("defs.common", wait_for_reported_addr=noop),
        "defs.trt_test_alternative": stub(
            "defs.trt_test_alternative", print_info=noop, print_warning=noop
        ),
        "defs.conftest": stub(
            "defs.conftest",
            get_llm_root=lambda *a, **k: "/repo",
            llm_models_root=lambda *a, **k: "/models",
        ),
        "defs.perf._model_paths": stub("defs.perf._model_paths", MODEL_PATH_DICT={}),
        "defs.perf.open_search_db_utils": stub(
            "defs.perf.open_search_db_utils",
            add_id=noop,
            get_history_data=noop,
            post_new_perf_data=noop,
        ),
        "defs.perf.perf_regression_utils": stub(
            "defs.perf.perf_regression_utils",
            process_and_upload_test_results=noop,
            get_job_info=noop,
            _percentile=noop,
        ),
    }

    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("defs.perf.test_perf_sanity", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module.ClientConfig


def _disagg_client_data(multi_round: int) -> dict[str, object]:
    """The client dict a disagg config builds (test_perf_sanity.py:2266-2277).

    Notably it carries no "name", so ClientConfig derives one.
    """
    return {
        "concurrency": 12,
        "iterations": multi_round,
        "isl": 50000,
        "osl": 2048,
    }


def test_disagg_iterations_come_from_multi_round() -> None:
    """Pin the yaml key that feeds "iterations" on the disagg path.

    _parse_disagg_config_file cannot be called here -- PerfSanityTestConfig's
    constructor shells out to nvidia-smi and raises without a GPU -- so the
    mapping is asserted against its source instead of re-stated in a docstring.
    Without this, renaming the yaml key would leave the tests below green while
    the documented behaviour silently changed.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    module_path = repo_root / "tests" / "integration" / "defs" / "perf" / "test_perf_sanity.py"
    tree = ast.parse(module_path.read_text())

    iterations_values = [
        value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "iterations"
    ]

    assert iterations_values, 'no dict literal builds an "iterations" entry'
    assert any("multi_round" in ast.dump(value) for value in iterations_values), (
        '"iterations" is no longer derived from benchmark.multi_round; '
        "update README_test_perf_sanity.md and the tests below"
    )


def test_iterations_fork_a_case_through_the_derived_name() -> None:
    """benchmark.multi_round renames a disagg case, so it still forks history.

    multi_round becomes the client's "iterations" (test_perf_sanity.py:2270), which
    lands in the derived name (:1119) and hence in s_test_case_name. Dropping
    l_iterations from the match key therefore does not make it fork-free on this
    path. Intended -- iterations sets the measurement length, so iter10 and iter12
    do not measure the same quantity -- but it costs the renamed case its history,
    so it is pinned here rather than left as an assumption.
    """
    client_config = _load_client_config()

    ten = client_config(_disagg_client_data(10), "example_model")
    twelve = client_config(_disagg_client_data(12), "example_model")

    assert ten.name == "con12_iter10_isl50000_osl2048"
    assert twelve.name == "con12_iter12_isl50000_osl2048"

    stem = "e2e-gb300_example_model_50k2k_con12_ctx1_dep4_gen6_tep4-"
    assert not benchmark_data_matches(
        _benchmark_data(s_test_case_name=stem + ten.name, l_iterations=10),
        _benchmark_data(s_test_case_name=stem + twelve.name, l_iterations=12),
        get_test_case_match_keys(),
    )


def test_an_explicit_client_name_keeps_the_case_across_an_iterations_change() -> None:
    """Aggregated configs all name their clients, so tuning keeps the history.

    This is the path where removing l_iterations from the key pays off: the name is
    pinned by the yaml, so a changed iterations count stays on one curve.
    """
    client_config = _load_client_config()

    ten = client_config({**_disagg_client_data(10), "name": "con1024_iter10_1k1k"}, "example_model")
    twenty = client_config(
        {**_disagg_client_data(20), "name": "con1024_iter10_1k1k"}, "example_model"
    )

    assert ten.name == twenty.name == "con1024_iter10_1k1k"
    assert benchmark_data_matches(
        _benchmark_data(s_test_case_name="example_model_fp8_tp8-" + ten.name, l_iterations=10),
        _benchmark_data(s_test_case_name="example_model_fp8_tp8-" + twenty.name, l_iterations=20),
        get_test_case_match_keys(),
    )


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
