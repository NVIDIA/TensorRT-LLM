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


def _load_module() -> types.ModuleType:
    """Import test_perf_sanity.py without the integration-test packages.

    Rules under test are only worth asserting against the code that owns them;
    re-implementing them here would assert nothing. test_perf_sanity.py reaches
    torch and the OpenSearch client through its imports, so those are stubbed --
    ClientConfig.__init__ and wants_warmup touch none of them.
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
    return module


def _load_client_config() -> type:
    """Return the real ClientConfig."""
    return _load_module().ClientConfig


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


def test_warmup_lets_the_initial_test_request_through() -> None:
    """A warmup lane drops --no-test-input, which is what creates the warmup.

    benchmark_serving's initial test request is excluded from the reported
    metrics, so it is the cheapest available warmup. It reuses
    input_requests[0], hence carries the lane's own ISL and OSL: on a disagg e2e
    lane it absorbs the KV cache transceiver's one-time lazy connection setup
    (ZMQ mesh + NIXL metadata registration) that otherwise slows the first
    measured ctx->gen handover, and on a ctx_only lane it is a full-ISL prefill
    that absorbs the first cold prefill out of the reported TTFT.
    """
    client_config = _load_client_config()

    cold = client_config(_disagg_client_data(10), "example_model")
    warm = client_config(_disagg_client_data(10), "example_model", warmup=True)

    assert "--no-test-input" in cold._to_default_benchmark_cmd()
    assert "--no-test-input" not in warm._to_default_benchmark_cmd()


def test_warmup_cannot_be_enabled_from_lane_config() -> None:
    """A "warmup" key in a lane yaml must not reach ClientConfig.warmup.

    b_warmup is deliberately not a match key (see
    test_match_keys_are_name_and_environment_only), so warmed results merge into
    the same baseline history as their cold predecessors. That is only sound
    while the value stays fully determined by benchmark_mode. Both config
    parsers hand the raw yaml client dict straight to ClientConfig, so if warmup
    were read from it, any lane -- including an aggregated one -- could enable
    warmup for itself and silently fork its own baseline history with no visible
    config difference. Hence the constructor argument.
    """
    client_config = _load_client_config()

    from_yaml = client_config({**_disagg_client_data(10), "warmup": True}, "example_model")

    assert from_yaml.warmup is False
    assert from_yaml.to_db_data()["b_warmup"] is False
    assert "--no-test-input" in from_yaml._to_default_benchmark_cmd()


def test_warmup_is_suppressed_for_the_non_default_benchmark_clients() -> None:
    """b_warmup records the EFFECTIVE value, not the requested one.

    to_cmd dispatches to three builders, and only the built-in
    benchmark_serving one has an initial test request to suppress. The agentx
    and nv_sa builders emit no equivalent flag, so a requested warmup would not
    happen there -- and a b_warmup=True row for a run that never warmed up is
    worse than no row at all: it invites a later investigator to rule warmup out
    as a cause it never had. Same convention as b_disable_overlap_scheduler,
    which also reports what the run actually did.
    """
    client_config = _load_client_config()

    nv_sa = client_config(
        {**_disagg_client_data(10), "use_nv_sa_benchmark": True}, "example_model", warmup=True
    )
    agentx = client_config(
        {**_disagg_client_data(10), "benchmark_client": "agentx"}, "example_model", warmup=True
    )
    default = client_config(_disagg_client_data(10), "example_model", warmup=True)

    assert nv_sa.warmup is False
    assert nv_sa.to_db_data()["b_warmup"] is False
    assert agentx.warmup is False
    assert agentx.to_db_data()["b_warmup"] is False
    assert default.warmup is True
    assert default.to_db_data()["b_warmup"] is True


def test_warmup_defaults_off_and_is_reported() -> None:
    """Every other lane keeps today's behaviour, and the DB records which warmed."""
    client_config = _load_client_config()

    cold = client_config(_disagg_client_data(10), "example_model")
    warm = client_config(_disagg_client_data(10), "example_model", warmup=True)

    assert cold.warmup is False
    assert cold.to_db_data()["b_warmup"] is False
    assert warm.to_db_data()["b_warmup"] is True


def test_warmup_is_derived_from_exactly_the_e2e_and_ctx_only_modes() -> None:
    """Pin warmup to benchmark_mode, the reason b_warmup can skip the match key.

    Asserted through wants_warmup rather than against the source text: the set
    of warmed lanes is the contract, the expression that computes it is not, so
    a behaviour-preserving refactor must not fail here. The tests above cover
    what ClientConfig does with the value; only this one covers which lanes get
    it.

    Every mode the disagg parser can see is checked explicitly, so adding a mode
    without deciding whether it warms up fails here rather than silently
    inheriting a default. gen_only in particular must stay excluded: #18011
    established that the extra handover leaves a stale mSenderFutures entry that
    the CTX worker's blocking idle KV-transfer poll then waits on.
    """
    module = _load_module()

    assert module.wants_warmup("e2e") is True
    assert module.wants_warmup("ctx_only") is True
    assert module.wants_warmup("gen_only") is False
    assert module.wants_warmup("gen_only_no_context") is False
    assert module.wants_warmup("") is False

    # Pinned as an exact set, not by substring: a membership test against a
    # tuple still "contains e2e" after gen_only has been added to it.
    assert set(module.WARMUP_BENCHMARK_MODES) == {"e2e", "ctx_only"}, (
        "the set of warmup lanes changed; e2e absorbs the KV transceiver's lazy "
        "connection setup and ctx_only absorbs the first cold prefill, while "
        "gen_only must stay excluded (#18011)"
    )


def test_warmup_reaches_client_config_only_from_the_disagg_parser() -> None:
    """No second producer may hand ClientConfig a warmup value.

    Locality, not expression shape: b_warmup is not a baseline match key, which
    is only sound while one code path decides warmup for every lane. A second
    ClientConfig(warmup=...) call site elsewhere -- notably in the aggregated
    parser, which forwards lane yaml keys through verbatim -- would reintroduce
    exactly the lane-settable warmup that test_warmup_cannot_be_enabled_from_
    lane_config forbids by value.

    _parse_disagg_config_file cannot be called directly here (PerfSanityTest-
    Config's constructor shells out to nvidia-smi and raises without a GPU), so
    this one property is checked against the source, scoped to that function.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    module_path = repo_root / "tests" / "integration" / "defs" / "perf" / "test_perf_sanity.py"
    tree = ast.parse(module_path.read_text())

    def warmup_call_sites(node: ast.AST) -> list[ast.AST]:
        return [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "ClientConfig"
            and any(keyword.arg == "warmup" for keyword in call.keywords)
        ]

    disagg_parsers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_disagg_config_file"
    ]
    assert len(disagg_parsers) == 1, "expected exactly one _parse_disagg_config_file"

    inside = warmup_call_sites(disagg_parsers[0])
    total = warmup_call_sites(tree)
    assert len(inside) == 1, (
        f"expected exactly one ClientConfig(warmup=...) in _parse_disagg_config_file, "
        f"found {len(inside)}"
    )
    assert len(total) == len(inside), (
        f"ClientConfig(warmup=...) appears {len(total) - len(inside)} time(s) outside "
        "_parse_disagg_config_file; warmup must be decided in exactly one place, or "
        "b_warmup stops being fully determined by benchmark_mode"
    )


def test_warmup_is_not_a_match_key() -> None:
    """Warmup is a measurement-quality knob, not part of case identity.

    Making b_warmup a match key would fork all ~26 warmed lanes into a second
    tracked series and make the improvement invisible in its own history -- a
    permanent cost to paper over a one-time step. The four match keys are
    identity, hardware, runtime and branch; none of them describes how well the
    run was set up. Same rationale as s_benchmark_client.
    """
    assert "b_warmup" not in get_test_case_match_keys()
