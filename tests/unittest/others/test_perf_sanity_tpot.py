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
"""Latency reporting and regression comparisons use latency semantics."""

import ast
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import redirect_stdout
from copy import deepcopy
from io import BytesIO, TextIOWrapper
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

pytestmark = pytest.mark.cpu_only
_PERF_DIR = Path(__file__).resolve().parents[2] / "integration" / "defs" / "perf"
_TPOT_METRICS = ("d_mean_tpot", "d_median_tpot", "d_p99_tpot")


@pytest.fixture
def regression_utils(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the real comparison code without GPU or OpenSearch dependencies."""
    alternative = ModuleType("defs.trt_test_alternative")
    alternative.print_info = lambda *args, **kwargs: None
    alternative.print_warning = lambda *args, **kwargs: None
    db_utils = ModuleType("defs.perf.open_search_db_utils")
    db_utils.add_id = lambda *args, **kwargs: None
    db_utils.get_history_data = lambda *args, **kwargs: None
    db_utils.post_new_perf_data = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "defs.trt_test_alternative", alternative)
    monkeypatch.setitem(sys.modules, "defs.perf.open_search_db_utils", db_utils)
    spec = importlib.util.spec_from_file_location(
        "defs.perf.perf_regression_utils", _PERF_DIR / "perf_regression_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def metric_lists() -> dict[str, list[str]]:
    """Read the harness lists without importing its serving dependencies."""
    # The unrelated time-breakdown metrics are not exercised by these tests.
    namespace = {"TIME_BREAKDOWN_METRICS": ()}
    names = {"MAXIMIZE_METRICS", "MINIMIZE_METRICS", "REGRESSION_METRICS"}
    tree = ast.parse((_PERF_DIR / "test_perf_sanity.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in names
        ):
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), "test_perf_sanity", "exec"),
                namespace,
            )
    return {name: namespace[name] for name in names}


@pytest.fixture
def perf_sanity_caller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[Callable[..., tuple[list[dict], RuntimeError | None]]]:
    """Run test_e2e with real parsing, history matching and verdicts, without servers."""
    repo_root = _PERF_DIR.parents[3]
    monkeypatch.syspath_prepend(str(repo_root))
    monkeypatch.delenv("PERF_BASELINE_BRANCH", raising=False)
    monkeypatch.delenv("BOLT_ITER_MULT", raising=False)
    monkeypatch.delenv("RAY_LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("AGG_CONFIG_FOLDER", str(tmp_path))
    monkeypatch.setenv("DISAGG_CONFIG_FOLDER", str(tmp_path))
    monkeypatch.setenv("OPEN_SEARCH_DB_BASE_URL", "https://example.test/unused")
    monkeypatch.setenv("DISAGG_SERVING_TYPE", "BENCHMARK")
    monkeypatch.setenv("stageName", "")

    def unused_boundary(*args, **kwargs) -> None:
        raise AssertionError("The CPU verdict test must not start native execution")

    definitions = ModuleType("defs")
    definitions.__path__ = [str(_PERF_DIR.parent)]
    performance = ModuleType("defs.perf")
    performance.__path__ = [str(_PERF_DIR)]
    configuration = ModuleType("defs.conftest")
    configuration.get_llm_root = lambda: str(repo_root)
    configuration.get_device_count = lambda: 4
    configuration.llm_models_root = unused_boundary
    common = ModuleType("defs.common")
    common.wait_for_reported_addr = unused_boundary
    alternative = ModuleType("defs.trt_test_alternative")
    alternative.print_info = lambda *args, **kwargs: None
    alternative.print_warning = lambda *args, **kwargs: None
    native_utils = ModuleType("tensorrt_llm._torch.disaggregation.native.utils")
    native_utils.get_local_ip = lambda: "127.0.0.1"
    opensearch = ModuleType("jenkins.scripts.open_search_db")
    opensearch.PERF_SANITY_PROJECT_NAME = "perf-tests"
    opensearch.OpenSearchDB = SimpleNamespace(
        add_id_of_json=lambda data: None,
        queryPerfDataFromOpenSearchDB=unused_boundary,
        postToOpenSearchDB=unused_boundary,
    )
    for module in (
        definitions,
        performance,
        configuration,
        common,
        alternative,
        native_utils,
        opensearch,
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    module_names = [
        f"defs.perf.{name}"
        for name in (
            "_model_paths",
            "open_search_db_utils",
            "perf_regression_utils",
            "time_breakdown_metrics",
            "warmup_pass",
            "test_perf_sanity",
        )
    ]
    for name in module_names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        harness = importlib.import_module("defs.perf.test_perf_sanity")
        database = importlib.import_module("defs.perf.open_search_db_utils")
        real_check_output = subprocess.check_output

        def replay(
            history: list[dict] | None,
            *,
            ttft: float | str = 125.0,
            throughput: float = 100.0,
            is_post_merge: bool = False,
            config: dict | None = None,
            test_case: str = "aggr_upload-latency_control",
            after_launch: Callable[[], None] | None = None,
            launches: list[tuple[Path, dict]] | None = None,
            worker_device_count: Callable[[dict[str, str]], int] | None = None,
            probe_timeouts: list[float | None] | None = None,
            probe_outputs: list[bytes | str] | None = None,
            worker_probe_stdout: bytes | None = None,
            warmup_only: bool = False,
            expect_upload: bool = True,
            real_worker_probe: bool = False,
        ) -> tuple[list[dict], RuntimeError | None]:
            def check_output(command, *args, **kwargs) -> bytes | str:
                if isinstance(command, str):
                    return "Product Name : B200\n"
                assert command[:2] == [sys.executable, "-c"]
                if probe_timeouts is not None:
                    probe_timeouts.append(kwargs.get("timeout"))
                if worker_probe_stdout is not None:
                    encoding = kwargs.get("encoding")
                    return worker_probe_stdout.decode(encoding) if encoding else worker_probe_stdout
                if real_worker_probe:
                    result = real_check_output(command, *args, **kwargs)
                    if probe_outputs is not None:
                        probe_outputs.append(result)
                    return result
                cuda_module = ModuleType("torch.cuda")
                cuda_module.device_count = lambda: (
                    worker_device_count(dict(os.environ)) if worker_device_count is not None else 4
                )
                torch_module = ModuleType("torch")
                torch_module.__path__ = []
                torch_module.cuda = cuda_module
                output_buffer = BytesIO()
                query_input = kwargs.get("input", b"")
                if isinstance(query_input, str):
                    query_input = query_input.encode("utf-8")
                with (
                    TextIOWrapper(output_buffer, encoding="utf-8") as output,
                    patch.dict(os.environ, kwargs["env"], clear=True),
                    monkeypatch.context() as query_context,
                    redirect_stdout(output),
                ):
                    query_context.setitem(sys.modules, "torch", torch_module)
                    query_context.setitem(sys.modules, "torch.cuda", cuda_module)
                    query_context.setattr(
                        sys,
                        "stdin",
                        TextIOWrapper(BytesIO(query_input), encoding="utf-8"),
                    )
                    exec(command[2], {})
                    output.flush()
                    result = output_buffer.getvalue()
                encoding = kwargs.get("encoding")
                return result.decode(encoding) if encoding else result

            monkeypatch.setattr(harness.subprocess, "check_output", check_output)
            configuration_data = config or {
                "metadata": {"model_name": "latency-control"},
                "hardware": {"gpus_per_node": 1},
                "server_configs": [
                    {"name": "plain", "client_configs": [{"name": "con1", "concurrency": 1}]}
                ],
            }
            (tmp_path / "latency_control.yaml").write_text(yaml.safe_dump(configuration_data))
            merge_type = "PostMerge" if is_post_merge else "MergeRequest_PR"
            monkeypatch.setenv(
                "globalVars",
                json.dumps(
                    {
                        "build_branch": "main" if is_post_merge else "latency-check",
                        "action_info": {
                            "parents": [{"url": f"https://example.test/L0_{merge_type}/1/"}]
                        },
                    }
                ),
            )
            output = "\n".join(
                [
                    "Successful requests: 1",
                    "Failed requests: 0",
                    "Request throughput (req/s): 1.0",
                    f"Output token throughput (tok/s): {throughput}",
                    "Total Token throughput (tok/s): 1000.0",
                    "User throughput (tok/s): 1.0",
                    f"Mean TTFT (ms): {ttft}",
                    "Median TTFT (ms): N/A",
                    "P99 TTFT (ms): N/A",
                    "Mean ITL (ms): N/A",
                    "Median ITL (ms): N/A",
                    "P99 ITL (ms): N/A",
                    "Mean TPOT (ms): N/A",
                    "Median TPOT (ms): N/A",
                    "P99 TPOT (ms): N/A",
                    "Mean E2EL (ms): 1.0",
                    "Median E2EL (ms): 1.0",
                    "P99 E2EL (ms): 1.0",
                    "Mean Avg Decoded Tokens per Iter: 2.0",
                ]
            )

            def run_cmd(commands, server_idx: int) -> list[str]:
                if launches is not None:
                    command = commands.server_cmds[server_idx]
                    config_path = Path(command[command.index("--config") + 1])
                    launches.append((config_path, yaml.safe_load(config_path.read_text())))
                if after_launch is not None:
                    after_launch()
                return [output for _ in commands.client_cmds[server_idx]]

            monkeypatch.setattr(harness.AggrTestCmds, "run_cmd", run_cmd)
            monkeypatch.setattr(harness.DisaggTestCmds, "run_cmd", run_cmd)

            def cold_warmup(*args, **kwargs) -> None:
                harness.record_warmup_time(args[4], args[5], args[6], 1.0)

            monkeypatch.setattr(harness, "run_cold_warmup", cold_warmup)
            uploaded = []
            monkeypatch.setattr(
                database.OpenSearchDB,
                "queryPerfDataFromOpenSearchDB",
                lambda *args, **kwargs: deepcopy(history),
            )

            def post_to_database(values: list[dict], project: str) -> bool:
                assert expect_upload, "This caller must not write results to the database"
                uploaded.extend(deepcopy(values))
                return True

            monkeypatch.setattr(database.OpenSearchDB, "postToOpenSearchDB", post_to_database)
            regression_error = None
            try:
                runner = harness.test_warmup if warmup_only else harness.test_e2e
                runner(str(tmp_path / "output"), test_case)
            except RuntimeError as error:
                regression_error = error
            if expect_upload:
                assert uploaded, f"Caller did not reach the upload/verdict path: {regression_error}"
            return uploaded, regression_error

        yield replay
    finally:
        for name in module_names:
            sys.modules.pop(name, None)


def _ttft_history(**overrides: object) -> dict:
    record = {
        "@timestamp": "2026-01-02T00:00:00.000Z",
        "ts_created": 1767312000000,
        "b_is_post_merge": True,
        "b_is_valid": True,
        "b_is_baseline": False,
        "s_test_case_name": "plain-con1",
        "s_gpu_type": "b200",
        "s_runtime": "aggr_server",
        "s_branch": "main",
        "d_mean_ttft": 100.0,
        "d_token_throughput": 100.0,
        "d_total_token_throughput": 1000.0,
        "d_baseline_mean_ttft": 100.0,
        "d_baseline_token_throughput": 100.0,
        "d_baseline_total_token_throughput": 1000.0,
    }
    record.update(overrides)
    return record


def _metric_comparison(record: dict, metric: str) -> tuple[float, float]:
    comparison = next(
        (
            line
            for line in record["s_regression_info"].splitlines()
            if line.strip().startswith(f"{metric}:")
        ),
        None,
    )
    assert comparison is not None, f"Missing matched comparison for {metric}"
    fields = dict(field.split("=", 1) for field in comparison.split()[1:])
    return float(fields["value"]), float(fields["baseline"])


@pytest.mark.parametrize("is_post_merge", [False, True], ids=["pre-merge", "post-merge"])
def test_ttft_caller_default_tolerances(perf_sanity_caller: Callable, is_post_merge: bool) -> None:
    """An 8% TTFT increase only crosses the default post-merge threshold."""
    uploaded, error = perf_sanity_caller([_ttft_history()], ttft=108.0, is_post_merge=is_post_merge)
    assert error is None
    assert uploaded[0]["b_is_regression"] is is_post_merge


@pytest.mark.parametrize("tolerance", [0.2, 0.3])
def test_ttft_caller_configured_tolerance(perf_sanity_caller: Callable, tolerance: float) -> None:
    """The selected latency gate honors the case's stored pre-merge tolerance."""
    uploaded, error = perf_sanity_caller([_ttft_history(d_threshold_pre_merge_mean_ttft=tolerance)])
    should_fail = tolerance < 0.25
    assert (error is not None) is should_fail
    assert uploaded[0]["b_is_regression"] is should_fail
    assert _metric_comparison(uploaded[0], "d_mean_ttft") == pytest.approx((125.0, 100.0))
    if should_fail:
        info = uploaded[0]["s_regression_info"]
        assert "new gated misses: (d_mean_ttft)" in info
        assert "unknown prior values: (none)" in info


@pytest.mark.parametrize("has_ttft_history", [True, False], ids=["healthy-history", "no-history"])
def test_ttft_caller_without_stored_override(
    perf_sanity_caller: Callable, has_ttft_history: bool
) -> None:
    """Only absent TTFT measurements, not an absent override, leave TTFT ungated."""
    history = _ttft_history()
    history.pop("d_baseline_mean_ttft")
    if not has_ttft_history:
        history.pop("d_mean_ttft")
    uploaded, error = perf_sanity_caller([history])
    assert uploaded[0]["b_is_regression"] is has_ttft_history
    assert (error is not None) is has_ttft_history
    if has_ttft_history:
        assert _metric_comparison(uploaded[0], "d_mean_ttft") == pytest.approx((125.0, 100.0))


def test_ttft_caller_selects_each_client(perf_sanity_caller: Callable) -> None:
    """Only plain single-node concurrency-one clients gain the latency gate."""
    config = {
        "metadata": {"model_name": "latency-control"},
        "hardware": {"gpus_per_node": 4},
        "server_configs": [
            {
                "name": "plain",
                "tensor_parallel_size": 4,
                "client_configs": [
                    {"name": "con1", "concurrency": 1},
                    {"name": "con2", "concurrency": 2},
                ],
            },
            {
                "name": "partial-node",
                "client_configs": [{"name": "con1", "concurrency": 1}],
            },
            {
                "name": "speculative",
                "tensor_parallel_size": 4,
                "speculative_config": {"decoding_type": "MTP"},
                "client_configs": [{"name": "con1", "concurrency": 1}],
            },
            {
                "name": "multi-node",
                "tensor_parallel_size": 8,
                "client_configs": [{"name": "con1", "concurrency": 1}],
            },
            {
                "name": "throughput-only",
                "tensor_parallel_size": 4,
                "server_env_var": "LATENCY_CONTROL_SERVER=throughput-only",
                "client_configs": [{"name": "con64", "concurrency": 64}],
            },
        ],
    }
    expected = {
        "plain-con1": True,
        "plain-con2": False,
        "partial-node-con1": True,
        "speculative-con1": False,
        "multi-node-con1": False,
        "throughput-only-con64": False,
    }
    history = [
        _ttft_history(
            s_test_case_name=name,
            s_runtime="multi_node_aggr_server"
            if name.startswith(("multi-node", "partial-node"))
            else "aggr_server",
        )
        for name in expected
    ]

    def device_count(environment: dict[str, str]) -> int:
        assert environment.get("LATENCY_CONTROL_SERVER") != "throughput-only"
        return 4

    uploaded, error = perf_sanity_caller(history, config=config, worker_device_count=device_count)
    for row in uploaded:
        assert _metric_comparison(row, "d_mean_ttft") == pytest.approx((125.0, 100.0))
    assert {row["s_test_case_name"]: row["b_is_regression"] for row in uploaded} == expected
    assert error is not None
    assert "Regression in plain-con1:" in str(error)


@pytest.mark.parametrize("speculative", [False, True], ids=["plain", "speculative"])
def test_ttft_caller_reads_external_server_config(
    perf_sanity_caller: Callable, tmp_path: Path, speculative: bool
) -> None:
    """Eligibility follows the effective worker config, including external files."""
    config_path = tmp_path / "worker.yaml"
    config_path.write_text(
        yaml.safe_dump({"speculative_config": {"decoding_type": "MTP"}} if speculative else {})
    )
    config = {
        "metadata": {"model_name": "latency-control"},
        "hardware": {"gpus_per_node": 1},
        "server_configs": [
            {
                "name": "plain",
                "extra_llm_api_config_path": str(config_path),
                "client_configs": [{"name": "con1", "concurrency": 1}],
            }
        ],
    }
    uploaded, error = perf_sanity_caller([_ttft_history()], config=config)
    assert _metric_comparison(uploaded[0], "d_mean_ttft") == pytest.approx((125.0, 100.0))
    assert uploaded[0]["b_eos"] is False
    assert uploaded[0]["b_is_regression"] is not speculative
    assert (error is None) is speculative


@pytest.mark.parametrize(
    "initial_speculation,replacement,concurrency",
    [
        pytest.param(True, False, 1, id="speculative-to-plain"),
        pytest.param(False, True, 1, id="plain-to-speculative"),
        pytest.param(True, None, 1, id="remove-speculative-source"),
        pytest.param(False, None, 2, id="remove-ineligible-source"),
    ],
)
def test_ttft_caller_preserves_rendered_configuration(
    perf_sanity_caller: Callable,
    tmp_path: Path,
    initial_speculation: bool,
    replacement: bool | None,
    concurrency: int,
) -> None:
    """Completed measurements use the rendered launch, not a later source file."""
    config_path = tmp_path / "worker.yaml"

    def write_config(speculative: bool) -> None:
        config_path.write_text(
            yaml.safe_dump({"speculative_config": {"decoding_type": "MTP"}} if speculative else {})
        )

    write_config(initial_speculation)

    def change_source() -> None:
        if replacement is None:
            config_path.unlink(missing_ok=True)
        else:
            write_config(replacement)

    client_name = f"con{concurrency}"
    config = {
        "metadata": {"model_name": "latency-control"},
        "hardware": {"gpus_per_node": 1},
        "server_configs": [
            {
                "name": "plain",
                "extra_llm_api_config_path": str(config_path),
                "client_configs": [{"name": client_name, "concurrency": concurrency}],
            }
        ],
    }
    uploaded, error = perf_sanity_caller(
        [_ttft_history(s_test_case_name=f"plain-{client_name}")],
        config=config,
        after_launch=change_source,
    )
    assert _metric_comparison(uploaded[0], "d_mean_ttft") == pytest.approx((125.0, 100.0))
    should_gate = concurrency == 1 and not initial_speculation
    assert uploaded[0]["b_is_regression"] is should_gate
    assert (error is not None) is should_gate


@pytest.mark.parametrize("speculative_first", [True, False])
def test_ttft_caller_isolates_duplicate_server_names(
    perf_sanity_caller: Callable, tmp_path: Path, speculative_first: bool
) -> None:
    """Unnamed servers keep their case identities and distinct launch configurations."""
    server_name = "latency-control_tp1_ep1_pp1_cp1_bs512_attnTRTLLM_moe"
    servers = []
    for name, speculative in (("first", speculative_first), ("second", not speculative_first)):
        worker_path = tmp_path / f"{name}.yaml"
        worker_path.write_text(
            yaml.safe_dump({"speculative_config": {"decoding_type": "MTP"}} if speculative else {})
        )
        servers.append(
            {
                "extra_llm_api_config_path": str(worker_path),
                "client_configs": [{"name": name, "concurrency": 1}],
            }
        )
    config = {
        "metadata": {"model_name": "latency-control"},
        "hardware": {"gpus_per_node": 1},
        "server_configs": servers,
    }
    launches = []
    uploaded, error = perf_sanity_caller(
        [_ttft_history(s_test_case_name=f"{server_name}-{name}") for name in ("first", "second")],
        config=config,
        launches=launches,
    )
    assert len(launches) == 2
    assert launches[0][0] != launches[1][0]
    assert [bool(worker.get("speculative_config")) for _, worker in launches] == [
        speculative_first,
        not speculative_first,
    ]
    assert [row["s_server_name"] for row in uploaded] == [server_name, server_name]
    assert all(row["b_eos"] is False for row in uploaded)
    for row, name, speculative in zip(
        uploaded, ("first", "second"), (speculative_first, not speculative_first)
    ):
        assert row["s_test_case_name"] == f"{server_name}-{name}"
        assert _metric_comparison(row, "d_mean_ttft") == pytest.approx((125.0, 100.0))
        assert row["b_is_regression"] is not speculative
        assert (f"Regression in {server_name}-{name}:" in str(error)) is not speculative
    assert error is not None


@pytest.mark.parametrize(
    ("mode", "collision"),
    [("aggr_upload", "servers"), ("aggr", "servers"), ("aggr_upload", "clients")],
)
def test_ttft_caller_rejects_ambiguous_history_keys(
    perf_sanity_caller: Callable, tmp_path: Path, mode: str, collision: str
) -> None:
    """Configured measurements reject colliding history identities before launch."""
    worker_path = tmp_path / "speculative.yaml"
    worker_path.write_text(yaml.safe_dump({"speculative_config": {"decoding_type": "MTP"}}))
    if collision == "servers":
        servers = [
            {
                "extra_llm_api_config_path": str(worker_path),
                "client_configs": [{"name": "con1", "concurrency": 1}],
            },
            {"client_configs": [{"name": "con1", "concurrency": 1}]},
        ]
    else:
        servers = [
            {
                "client_configs": [
                    {"name": "con1", "concurrency": 1},
                    {"name": "con1", "concurrency": 2},
                ],
            }
        ]
    config = {
        "metadata": {"model_name": "latency-control"},
        "hardware": {"gpus_per_node": 1},
        "server_configs": servers,
    }
    launches = []
    timeouts = []
    with pytest.raises(
        ValueError,
        match="Duplicate aggregated performance history key.*distinct server or client names",
    ):
        perf_sanity_caller(
            [
                _ttft_history(
                    s_test_case_name="latency-control_tp1_ep1_pp1_cp1_bs512_attnTRTLLM_moe-con1"
                )
            ],
            config=config,
            test_case=f"{mode}-latency_control",
            launches=launches,
            probe_timeouts=timeouts,
            expect_upload=False,
        )
    assert not launches and not timeouts


def test_ttft_caller_distinguishes_runtime_history_keys(perf_sanity_caller: Callable) -> None:
    """An equal case name with different reporting runtimes is not a key collision."""
    config = {
        "metadata": {"model_name": "latency-control"},
        "hardware": {"gpus_per_node": 4},
        "server_configs": [
            {
                "name": "plain",
                "speculative_config": {"decoding_type": "MTP"},
                "client_configs": [{"name": "con1", "concurrency": 1}],
            },
            {
                "name": "plain",
                "tensor_parallel_size": 4,
                "client_configs": [{"name": "con1", "concurrency": 1}],
            },
        ],
    }
    uploaded, error = perf_sanity_caller(
        [
            _ttft_history(s_runtime="multi_node_aggr_server", d_al=2.0, d_baseline_al=2.0),
            _ttft_history(),
        ],
        config=config,
    )
    assert error is not None and len(uploaded) == 2
    assert [row["s_test_case_name"] for row in uploaded] == ["plain-con1", "plain-con1"]
    assert [row["s_runtime"] for row in uploaded] == ["multi_node_aggr_server", "aggr_server"]
    assert [row["b_is_regression"] for row in uploaded] == [False, True]
    assert all(_metric_comparison(row, "d_mean_ttft") == (125.0, 100.0) for row in uploaded)


def test_ttft_caller_allows_offline_duplicate_names(
    perf_sanity_caller: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline measurements have no history identity to collide with or capacity query."""
    monkeypatch.delenv("OPEN_SEARCH_DB_BASE_URL", raising=False)
    config = {
        "metadata": {"model_name": "latency-control"},
        "hardware": {"gpus_per_node": 1},
        "server_configs": [
            {"name": "plain", "client_configs": [{"name": "con1", "concurrency": 1}]},
            {"name": "plain", "client_configs": [{"name": "con1", "concurrency": 1}]},
        ],
    }
    launches = []
    timeouts = []
    uploaded, error = perf_sanity_caller(
        None, config=config, launches=launches, probe_timeouts=timeouts, expect_upload=False
    )
    assert error is None and not uploaded and not timeouts
    assert len(launches) == 2 and launches[0][0] != launches[1][0]


def test_ttft_caller_ignores_accuracy_only_history_keys(perf_sanity_caller: Callable) -> None:
    """An accuracy-only client emits no performance row and cannot collide with one."""
    config = {
        "metadata": {"model_name": "latency-control"},
        "hardware": {"gpus_per_node": 1},
        "server_configs": [
            {
                "name": "plain",
                "client_configs": [
                    {
                        "name": "con1",
                        "concurrency": 1,
                        "only_run_accuracy": True,
                        "accuracy_config": {
                            "enable_accuracy_test": True,
                            "tasks": {"control-task": {}},
                        },
                    },
                    {"name": "con1", "concurrency": 1},
                ],
            }
        ],
    }
    uploaded, error = perf_sanity_caller([_ttft_history()], config=config)
    assert error is not None and len(uploaded) == 1
    assert uploaded[0]["s_test_case_name"] == "plain-con1"
    assert uploaded[0]["b_is_regression"] is True
    assert _metric_comparison(uploaded[0], "d_mean_ttft") == pytest.approx((125.0, 100.0))


@pytest.mark.parametrize(
    "worker,hardware_gpus_per_node,should_gate",
    [
        pytest.param({"tensor_parallel_size": 8}, 4, False, id="tensor-multi-node"),
        pytest.param(
            {"tensor_parallel_size": 2, "pipeline_parallel_size": 2, "context_parallel_size": 2},
            4,
            False,
            id="combined-parallelism-multi-node",
        ),
        pytest.param(
            {"tensor_parallel_size": 4, "moe_expert_parallel_size": 4},
            4,
            True,
            id="expert-parallelism-does-not-multiply-world-size",
        ),
        pytest.param({"tensor_parallel_size": 4}, 0, True, id="detected-node-size"),
    ],
)
def test_ttft_caller_resolves_launch_topology(
    perf_sanity_caller: Callable,
    tmp_path: Path,
    worker: dict,
    hardware_gpus_per_node: int,
    should_gate: bool,
) -> None:
    """Equivalent inline and external workers share eligibility, not legacy metadata."""
    worker = {"checkpoint_io_policy": "auto", **worker}
    worker_path = tmp_path / "worker.yaml"
    worker_path.write_text(yaml.safe_dump(worker))
    launches = []
    for inline in (True, False):
        server = worker if inline else {"extra_llm_api_config_path": str(worker_path)}
        config = {
            "metadata": {"model_name": "latency-control"},
            "hardware": {"gpus_per_node": hardware_gpus_per_node},
            "server_configs": [
                {**server, "name": "plain", "client_configs": [{"name": "con1", "concurrency": 1}]}
            ],
        }
        legacy_gpus = (
            server.get("tensor_parallel_size", 1)
            * server.get("pipeline_parallel_size", 1)
            * server.get("context_parallel_size", 1)
        )
        legacy_node_size = hardware_gpus_per_node or legacy_gpus
        runtime = "aggr_server" if legacy_gpus == legacy_node_size else "multi_node_aggr_server"
        uploaded, error = perf_sanity_caller(
            [_ttft_history(s_runtime=runtime)],
            config=config,
            launches=launches,
            worker_device_count=lambda environment: 8 if hardware_gpus_per_node == 0 else 4,
        )
        assert _metric_comparison(uploaded[0], "d_mean_ttft") == pytest.approx((125.0, 100.0))
        assert uploaded[0]["b_is_regression"] is should_gate
        assert (error is not None) is should_gate
        assert uploaded[0]["l_gpus"] == legacy_gpus
        assert uploaded[0]["l_gpus_per_node"] == legacy_node_size
        assert uploaded[0]["l_tp"] == server.get("tensor_parallel_size", 1)
        assert uploaded[0]["l_pp"] == server.get("pipeline_parallel_size", 1)
        assert uploaded[0]["l_cp"] == server.get("context_parallel_size", 1)
        assert uploaded[0]["s_runtime"] == runtime
    assert len(launches) == 2
    assert launches[0][1] == launches[1][1]


def test_ttft_caller_uses_worker_node_size(perf_sanity_caller: Callable, tmp_path: Path) -> None:
    """An explicit worker node size governs topology without rewriting matching fields."""
    worker_path = tmp_path / "worker.yaml"
    worker_path.write_text(yaml.safe_dump({"tensor_parallel_size": 4, "gpus_per_node": 2}))
    config = {
        "metadata": {"model_name": "latency-control"},
        "hardware": {"gpus_per_node": 4},
        "server_configs": [
            {
                "name": "plain",
                "extra_llm_api_config_path": str(worker_path),
                "client_configs": [{"name": "con1", "concurrency": 1}],
            }
        ],
    }

    def unexpected_query(environment: dict[str, str]) -> int:
        raise AssertionError("An explicit worker node size must not probe CUDA capacity")

    uploaded, error = perf_sanity_caller(
        [_ttft_history(s_runtime="multi_node_aggr_server")],
        config=config,
        worker_device_count=unexpected_query,
    )
    assert _metric_comparison(uploaded[0], "d_mean_ttft") == pytest.approx((125.0, 100.0))
    assert uploaded[0]["l_gpus_per_node"] == 4
    assert uploaded[0]["l_gpus"] == 1
    assert uploaded[0]["b_is_regression"] is False
    assert error is None


@pytest.mark.parametrize(
    "worker,should_gate",
    [
        pytest.param({"tensor_parallel_size": "4", "gpus_per_node": 4}, True, id="quoted-tp"),
        pytest.param({"tensor_parallel_size": 4, "gpus_per_node": "4"}, True, id="quoted-node"),
        pytest.param(
            {"tensor_parallel_size": "16", "gpus_per_node": "8"},
            False,
            id="lexical-order-is-not-node-order",
        ),
        pytest.param(
            {
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": "2",
                "context_parallel_size": "2",
                "gpus_per_node": "4",
            },
            False,
            id="quoted-pipeline-and-context",
        ),
        pytest.param(
            {"tensor_parallel_size": "4.0", "gpus_per_node": "4.0"},
            True,
            id="worker-integer-coercion",
        ),
    ],
)
@pytest.mark.parametrize("inline", [True, False], ids=["inline", "external"])
def test_ttft_caller_normalizes_rendered_topology(
    perf_sanity_caller: Callable, tmp_path: Path, worker: dict, should_gate: bool, inline: bool
) -> None:
    """Inline and external quoted sizes agree with numeric verdicts and metadata."""
    worker_path = tmp_path / "worker.yaml"
    numeric = {key: int(float(value)) for key, value in worker.items()}
    for configuration in (numeric, worker):
        worker_path.write_text(yaml.safe_dump(configuration))
        server = configuration if inline else {"extra_llm_api_config_path": str(worker_path)}
        node_size = numeric["gpus_per_node"]
        gpus = (
            (
                numeric.get("tensor_parallel_size", 1)
                * numeric.get("pipeline_parallel_size", 1)
                * numeric.get("context_parallel_size", 1)
            )
            if inline
            else 1
        )
        config = {
            "metadata": {"model_name": "latency-control"},
            "hardware": {"gpus_per_node": configuration["gpus_per_node"]},
            "server_configs": [
                {**server, "name": "plain", "client_configs": [{"name": "con1", "concurrency": 1}]}
            ],
        }
        runtime = "aggr_server" if gpus == node_size else "multi_node_aggr_server"
        uploaded, error = perf_sanity_caller(
            [_ttft_history(s_runtime=runtime)],
            config=config,
            worker_device_count=lambda environment: node_size,
        )
        assert _metric_comparison(uploaded[0], "d_mean_ttft") == pytest.approx((125.0, 100.0))
        assert uploaded[0]["b_is_regression"] is should_gate
        assert (error is not None) is should_gate
        for field, key in (
            ("l_tp", "tensor_parallel_size"),
            ("l_pp", "pipeline_parallel_size"),
            ("l_cp", "context_parallel_size"),
        ):
            assert uploaded[0][field] == (numeric.get(key, 1) if inline else 1)
        assert uploaded[0]["l_gpus"] == gpus
        assert uploaded[0]["l_gpus_per_node"] == node_size


def test_ttft_caller_normalizes_inline_gpu_hints(perf_sanity_caller: Callable) -> None:
    """Quoted harness GPU hints are normalized before topology comparisons."""
    config = {
        "metadata": {"model_name": "latency-control"},
        "hardware": {"gpus_per_node": "4"},
        "server_configs": [
            {
                "name": "plain",
                "tensor_parallel_size": "4",
                "gpus": "4",
                "client_configs": [{"name": "con1", "concurrency": 1}],
            }
        ],
    }
    uploaded, error = perf_sanity_caller([_ttft_history()], config=config)
    assert error is not None and uploaded[0]["b_is_regression"] is True
    assert _metric_comparison(uploaded[0], "d_mean_ttft") == pytest.approx((125.0, 100.0))
    assert uploaded[0]["l_tp"] == uploaded[0]["l_gpus"] == uploaded[0]["l_gpus_per_node"] == 4


@pytest.mark.parametrize(
    ("ttft", "throughput", "regression_metric"),
    [
        (125.0, 100.0, "d_mean_ttft"),
        (100.0, 75.0, "d_token_throughput"),
        (100.0, 100.0, None),
    ],
    ids=["ttft-loss", "throughput-loss", "healthy"],
)
def test_ttft_caller_checks_read_only_history(
    perf_sanity_caller: Callable,
    ttft: float,
    throughput: float,
    regression_metric: str | None,
) -> None:
    """Matched history still gates read-only runs without posting any results."""
    uploaded, error = perf_sanity_caller(
        [_ttft_history(d_threshold_pre_merge_mean_ttft=0.1)],
        ttft=ttft,
        throughput=throughput,
        test_case="aggr-latency_control",
        expect_upload=False,
    )
    assert not uploaded
    assert (error is not None) is (regression_metric is not None)
    if error is not None:
        record = {"s_regression_info": str(error)}
        assert _metric_comparison(record, "d_mean_ttft") == pytest.approx((ttft, 100.0))
        assert _metric_comparison(record, "d_token_throughput") == pytest.approx(
            (throughput, 100.0)
        )


@pytest.mark.parametrize("startup_banner", [False, True], ids=["quiet", "startup-banner"])
def test_ttft_caller_keeps_rendered_encoding_after_probe_startup(
    perf_sanity_caller: Callable,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    startup_banner: bool,
) -> None:
    """Startup encoding and banners do not break a real read-only capacity probe."""
    monkeypatch.delenv("RAY_LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-16")
    torch_path = tmp_path / "probe modules"
    torch_path.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(torch_path))
    if startup_banner:
        (torch_path / "sitecustomize.py").write_text("print('probe startup banner', flush=True)\n")
    (torch_path / "torch.py").write_text(
        "import os\n"
        "class cuda:\n"
        "    @staticmethod\n"
        "    def device_count():\n"
        "        return len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))\n"
    )
    worker_path = tmp_path / "worker.yaml"
    worker_path.write_text(
        yaml.safe_dump(
            {
                "gpus_per_node": None,
                "env_overrides": {
                    "CUDA_VISIBLE_DEVICES": "0,1",
                    "PYTHONIOENCODING": "utf-8",
                },
            }
        )
    )
    launches = []
    probe_timeouts = []
    probe_outputs = []
    uploaded, error = perf_sanity_caller(
        [_ttft_history()],
        config={
            "metadata": {"model_name": "latency-control"},
            "server_configs": [
                {
                    "name": "plain",
                    "tensor_parallel_size": 2,
                    "extra_llm_api_config_path": str(worker_path),
                    "client_configs": [{"name": "con1", "concurrency": 1}],
                }
            ],
        },
        test_case="aggr-latency_control",
        expect_upload=False,
        real_worker_probe=True,
        launches=launches,
        probe_timeouts=probe_timeouts,
        probe_outputs=probe_outputs,
    )
    assert probe_timeouts
    if startup_banner:
        assert probe_outputs and all(
            isinstance(output, bytes) and "probe startup banner".encode("utf-16")[2:] in output
            for output in probe_outputs
        )
    assert len(launches) == 1 and not uploaded
    assert error is not None
    assert _metric_comparison({"s_regression_info": str(error)}, "d_mean_ttft") == pytest.approx(
        (125.0, 100.0)
    )
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert os.environ["PYTHONIOENCODING"] == "utf-16"


@pytest.mark.parametrize(
    ("launch_visibility", "worker_visibility"),
    [
        ("0,1", "0,1,2,3"),
        ("0,1,2,3", "0,1"),
        ("0,1,2,3", "0,1,2,3"),
    ],
    ids=["expand", "shrink", "matching-counts"],
)
def test_ttft_caller_rejects_conflicting_capacity_records(
    perf_sanity_caller: Callable,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    launch_visibility: str,
    worker_visibility: str,
) -> None:
    """Conflicting adjacent count records must not choose TTFT eligibility."""
    monkeypatch.delenv("RAY_LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", launch_visibility)
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    torch_path = tmp_path / "probe modules"
    torch_path.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(torch_path))
    (torch_path / "torch.py").write_text(
        "import os\n"
        "class cuda:\n"
        "    @staticmethod\n"
        "    def device_count():\n"
        "        return len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))\n"
    )
    (torch_path / "sitecustomize.py").write_text(
        "import os\n"
        "capacity = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))\n"
        "os.write(1, b'launch topology\\nTRTLLM_CUDA_DEVICE_COUNT=' + "
        "str(capacity).encode('ascii'))\n"
    )
    worker_path = tmp_path / "worker.yaml"
    worker_path.write_text(
        yaml.safe_dump(
            {
                "gpus_per_node": None,
                "env_overrides": {"CUDA_VISIBLE_DEVICES": worker_visibility},
            }
        )
    )
    launches = []
    probe_timeouts = []
    probe_outputs = []
    uploaded, error = perf_sanity_caller(
        [_ttft_history()],
        config={
            "metadata": {"model_name": "latency-control"},
            "server_configs": [
                {
                    "name": "plain",
                    "tensor_parallel_size": 4,
                    "extra_llm_api_config_path": str(worker_path),
                    "client_configs": [{"name": "con1", "concurrency": 1}],
                }
            ],
        },
        test_case="aggr-latency_control",
        expect_upload=False,
        real_worker_probe=True,
        launches=launches,
        probe_timeouts=probe_timeouts,
        probe_outputs=probe_outputs,
    )
    launch_count = len(launch_visibility.split(","))
    worker_count = len(worker_visibility.split(","))
    adjacent_records = (
        f"TRTLLM_CUDA_DEVICE_COUNT={launch_count}\nTRTLLM_CUDA_DEVICE_COUNT={worker_count}\n"
    ).encode("ascii")
    assert probe_outputs and all(
        isinstance(output, bytes) and adjacent_records in output for output in probe_outputs
    )
    assert probe_timeouts and all(timeout == 30 for timeout in probe_timeouts)
    assert not uploaded and error is not None
    if launch_count != worker_count:
        assert not launches
        assert "CUDA capacity probe for server 'plain' failed" in str(error)
        assert "conflicting" in str(error)
        assert isinstance(error.__cause__, ValueError)
    else:
        assert len(launches) == 1
        assert _metric_comparison(
            {"s_regression_info": str(error)}, "d_mean_ttft"
        ) == pytest.approx((125.0, 100.0))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == launch_visibility


@pytest.mark.parametrize("mode", ["aggr", "aggr_upload"])
def test_ttft_caller_skips_offline_capacity_probes(
    perf_sanity_caller: Callable, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Offline measurements need no capacity probe when history cannot be read."""
    monkeypatch.delenv("OPEN_SEARCH_DB_BASE_URL", raising=False)
    launches = []
    timeouts = []

    def failing_query(environment: dict[str, str]) -> int:
        raise subprocess.TimeoutExpired("offline CUDA query", 30)

    uploaded, error = perf_sanity_caller(
        None,
        test_case=f"{mode}-latency_control",
        launches=launches,
        worker_device_count=failing_query,
        probe_timeouts=timeouts,
        expect_upload=False,
    )
    assert error is None and not uploaded
    assert len(launches) == 1
    assert not timeouts


def test_ttft_caller_skips_unused_capacity_probes(perf_sanity_caller: Callable) -> None:
    """Accuracy-only clients have no TTFT verdict and need no capacity query."""
    client = {
        "name": "con1",
        "concurrency": 1,
        "only_run_accuracy": True,
        "accuracy_config": {
            "enable_accuracy_test": True,
            "tasks": {"control-task": {}},
        },
    }
    config = {
        "metadata": {"model_name": "latency-control"},
        "hardware": {"gpus_per_node": 1},
        "server_configs": [{"name": "plain", "client_configs": [client]}],
    }

    def unexpected_query(environment: dict[str, str]) -> int:
        raise AssertionError("This path cannot use a TTFT capacity query")

    uploaded, error = perf_sanity_caller(
        [],
        config=config,
        worker_device_count=unexpected_query,
        expect_upload=False,
    )
    assert not uploaded and error is None


def test_ttft_caller_skips_warmup_capacity_probes(perf_sanity_caller: Callable) -> None:
    """The real warmup entrypoint uploads warmup data without a TTFT capacity query."""

    def unexpected_query(environment: dict[str, str]) -> int:
        raise AssertionError("Warmup-only runs cannot use a TTFT capacity query")

    uploaded, error = perf_sanity_caller([], warmup_only=True, worker_device_count=unexpected_query)
    assert error is None and len(uploaded) == 1
    assert uploaded[0]["b_warmup_only"] is True
    assert uploaded[0]["s_test_case_name"] == "plain-warmup"
    assert uploaded[0]["d_warmup_time"] == 1.0


@pytest.mark.parametrize("failure", ["timeout", "process-error", "missing-result"])
def test_ttft_caller_reports_capacity_probe_failure(
    perf_sanity_caller: Callable, failure: str
) -> None:
    """Necessary probes receive a finite bound and failures retain server context."""
    timeouts = []

    def failing_query(environment: dict[str, str]) -> int:
        if failure == "timeout":
            raise subprocess.TimeoutExpired("worker CUDA query", 30)
        raise subprocess.CalledProcessError(2, "worker CUDA query")

    uploaded, error = perf_sanity_caller(
        [],
        worker_device_count=failing_query,
        probe_timeouts=timeouts,
        worker_probe_stdout=b"probe startup banner\n" if failure == "missing-result" else None,
        expect_upload=False,
    )
    assert not uploaded and error is not None
    assert "plain" in str(error) and "CUDA capacity probe" in str(error)
    assert timeouts and all(
        timeout is not None and 0 < timeout < float("inf") for timeout in timeouts
    )
    expected = {
        "timeout": subprocess.TimeoutExpired,
        "process-error": subprocess.CalledProcessError,
        "missing-result": ValueError,
    }[failure]
    assert isinstance(error.__cause__, expected)


def test_ttft_caller_uses_worker_visible_devices(
    perf_sanity_caller: Callable, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Omitted capacity keeps the CLI count; explicit null recounts after worker overrides."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    monkeypatch.setenv("LATENCY_TEST_PARENT", "inherited")
    monkeypatch.delenv("RAY_LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("LATENCY_TEST_RENDERED", raising=False)
    queried_masks = []
    visible_counts = {"0,1,2": 3, "0,1,2,3": 4, "0,1": 2}

    def device_count(environment: dict[str, str]) -> int:
        assert environment["LATENCY_TEST_PARENT"] == "inherited"
        assert all(isinstance(value, str) for value in environment.values())
        mask = environment["CUDA_VISIBLE_DEVICES"]
        queried_masks.append(mask)
        return visible_counts[mask]

    server_configs = [
        {
            "name": name,
            "tensor_parallel_size": 4,
            "server_env_var": variables,
            "client_configs": [{"name": "con1", "concurrency": 1}],
        }
        for name, variables in (
            ("inherited", ""),
            ("full", "CUDA_VISIBLE_DEVICES=0,1,2,3"),
            ("masked", "CUDA_VISIBLE_DEVICES=0,1"),
        )
    ]
    for source in ("inline", "external"):
        for capacity in ("omitted", "null"):
            for name, variables, mask in (
                ("full", "CUDA_VISIBLE_DEVICES=0,1", "0,1,2,3"),
                ("masked", "CUDA_VISIBLE_DEVICES=0,1,2,3", "0,1"),
            ):
                worker_config = {
                    "env_overrides": {"CUDA_VISIBLE_DEVICES": mask, "LATENCY_TEST_RENDERED": 17}
                }
                server = {
                    "name": f"{source}-{capacity}-{name}",
                    "tensor_parallel_size": 4,
                    "server_env_var": variables,
                    "client_configs": [{"name": "con1", "concurrency": 1}],
                }
                file_config = {"gpus_per_node": None} if capacity == "null" else {}
                if source == "inline":
                    server.update(worker_config)
                else:
                    file_config.update(worker_config)
                if file_config:
                    worker_path = tmp_path / f"{source}-{capacity}-{name}.yaml"
                    worker_path.write_text(yaml.safe_dump(file_config))
                    server["extra_llm_api_config_path"] = str(worker_path)
                server_configs.append(server)
    config = {
        "metadata": {"model_name": "latency-control"},
        "server_configs": server_configs,
    }
    uploaded, error = perf_sanity_caller(
        [_ttft_history(s_test_case_name=f"{server['name']}-con1") for server in server_configs],
        config=config,
        worker_device_count=device_count,
    )
    assert set(queried_masks) == set(visible_counts)
    assert len(uploaded) == len(server_configs)
    eligible_names = {
        "full-con1",
        "inline-omitted-masked-con1",
        "external-omitted-masked-con1",
        "inline-null-full-con1",
        "external-null-full-con1",
    }
    for row in uploaded:
        name = row["s_test_case_name"]
        assert _metric_comparison(row, "d_mean_ttft") == pytest.approx((125.0, 100.0))
        assert row["l_gpus"] == row["l_gpus_per_node"] == 4
        assert row["b_is_regression"] is (name in eligible_names)
        assert (f"Regression in {name}:" in str(error)) is (name in eligible_names)
    assert error is not None
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2"
    assert "LATENCY_TEST_RENDERED" not in os.environ


def test_ttft_caller_respects_ray_node_capacity(
    perf_sanity_caller: Callable, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Server and rendered Ray environments use TP even with an explicit larger capacity."""
    monkeypatch.delenv("RAY_LOCAL_WORLD_SIZE", raising=False)
    worker_path = tmp_path / "worker.yaml"

    def unexpected_probe(environment: dict[str, str]) -> int:
        pytest.fail("Explicit or Ray worker capacity must not query CUDA")

    for source, pipeline_parallel_size in (
        ("server", 2),
        ("worker", 2),
        ("server", 1),
        ("worker", 1),
    ):
        worker_config = {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": pipeline_parallel_size,
            "gpus_per_node": 4,
        }
        server = {
            "name": "plain",
            "extra_llm_api_config_path": str(worker_path),
            "client_configs": [{"name": "con1", "concurrency": 1}],
        }
        if source == "server":
            server["server_env_var"] = "RAY_LOCAL_WORLD_SIZE=4"
        else:
            worker_config["env_overrides"] = {"RAY_LOCAL_WORLD_SIZE": 4}
        worker_path.write_text(yaml.safe_dump(worker_config))
        config = {
            "metadata": {"model_name": "latency-control"},
            "hardware": {"gpus_per_node": 4},
            "server_configs": [server],
        }
        probe_timeouts = []
        uploaded, error = perf_sanity_caller(
            [_ttft_history(s_runtime="multi_node_aggr_server")],
            config=config,
            worker_device_count=unexpected_probe,
            probe_timeouts=probe_timeouts,
        )
        assert not probe_timeouts
        assert _metric_comparison(uploaded[0], "d_mean_ttft") == pytest.approx((125.0, 100.0))
        assert uploaded[0]["b_is_regression"] is (pipeline_parallel_size == 1)
        assert (error is not None) is (pipeline_parallel_size == 1)
    assert "RAY_LOCAL_WORLD_SIZE" not in os.environ


@pytest.mark.parametrize(
    "prior_state", ["missing", "invalid", "nonpositive", "infinite", "negative-infinite", "nan"]
)
def test_ttft_caller_distinguishes_unknown_prior_with_complete_baselines(
    perf_sanity_caller: Callable, prior_state: str
) -> None:
    """After baseline resolution, unusable prior values do not establish a new loss."""
    latest = _ttft_history(
        d_baseline_seq_throughput=1.0,
        d_baseline_user_throughput=1.0,
        d_baseline_mean_e2el=1.0,
        d_baseline_median_e2el=1.0,
        d_baseline_p99_e2el=1.0,
    )
    if prior_state == "missing":
        latest.pop("d_mean_ttft")
    else:
        latest["d_mean_ttft"] = {
            "invalid": "not-a-number",
            "nonpositive": 0.0,
            "infinite": "Infinity",
            "negative-infinite": "-Infinity",
            "nan": "NaN",
        }[prior_state]
    older = _ttft_history(
        **{
            "@timestamp": "2026-01-01T00:00:00.000Z",
            "ts_created": 1767225600000,
            "d_mean_ttft": 125.0,
        }
    )
    uploaded, error = perf_sanity_caller([older, latest])
    assert _metric_comparison(uploaded[0], "d_mean_ttft") == pytest.approx((125.0, 100.0))
    assert uploaded[0]["b_is_regression"] is True
    assert error is not None
    info = uploaded[0]["s_regression_info"]
    assert "new gated misses: (none)" in info
    assert "unknown prior values: (d_mean_ttft)" in info
    assert "Not failing this stage" not in info


def test_ttft_caller_qualifies_unmeasured_prior_metrics(perf_sanity_caller: Callable) -> None:
    """The prior-miss diagnostic describes only metrics compared by the current run."""
    uploaded, error = perf_sanity_caller(
        [_ttft_history(d_mean_ttft=125.0)], ttft="N/A", throughput=75.0
    )
    assert "d_mean_ttft" not in uploaded[0]
    assert _metric_comparison(uploaded[0], "d_token_throughput") == pytest.approx((75.0, 100.0))
    assert uploaded[0]["b_is_regression"] is True
    assert error is not None
    info = uploaded[0]["s_regression_info"]
    assert "Latest matched post-merge gated misses among current comparisons: (none)" in info
    assert "new gated misses: (d_token_throughput)" in info


@pytest.mark.parametrize("mode", ["e2e", "ctx_only"])
def test_ttft_caller_preserves_other_modes(perf_sanity_caller: Callable, mode: str) -> None:
    """Disaggregated e2e and context-only TTFT remains diagnostic, not a gate."""
    config = {
        "metadata": {"model_name": "latency-control"},
        "hardware": {"gpus_per_node": 1, "num_ctx_servers": 1, "num_gen_servers": 1},
        "benchmark": {"concurrency_list": [1], "input_length": 16, "output_length": 8},
        "worker_config": {"ctx": {}, "gen": {}},
    }
    prefix = "disagg_upload" if mode == "e2e" else "aggr_upload"
    history = _ttft_history(
        s_test_case_name=f"{mode}-latency_control-con1_iter1_isl16_osl{8 if mode == 'e2e' else 1}",
        s_runtime="multi_node_disagg_server" if mode == "e2e" else "aggr_server",
    )
    uploaded, error = perf_sanity_caller(
        [history], config=config, test_case=f"{prefix}-{mode}-latency_control"
    )
    assert error is None
    assert uploaded[0]["b_is_regression"] is False
    assert _metric_comparison(uploaded[0], "d_mean_ttft") == pytest.approx((125.0, 100.0))


def test_ttft_caller_preserves_legacy_exemption(perf_sanity_caller: Callable) -> None:
    """Mixed batches choose independent TTFT and whole-case exemptions per client."""
    config = {
        "metadata": {"model_name": "latency-control"},
        "hardware": {"gpus_per_node": 1},
        "server_configs": [
            {"name": "plain", "client_configs": [{"name": "con1", "concurrency": 1}]},
            {
                "name": "speculative",
                "speculative_config": {"decoding_type": "MTP"},
                "client_configs": [{"name": "con1", "concurrency": 1}],
            },
        ],
    }
    history = [
        _ttft_history(d_token_throughput=75.0),
        _ttft_history(
            s_test_case_name="speculative-con1",
            d_al=1.5,
            d_baseline_al=2.0,
            b_is_regression=True,
        ),
    ]
    uploaded, error = perf_sanity_caller(history, throughput=75.0, config=config)
    plain, speculative = uploaded
    for row in uploaded:
        assert row["b_is_regression"] is True
        assert _metric_comparison(row, "d_mean_ttft") == pytest.approx((125.0, 100.0))
        assert _metric_comparison(row, "d_token_throughput") == pytest.approx((75.0, 100.0))
    assert _metric_comparison(speculative, "d_al") == pytest.approx((2.0, 2.0))
    assert error is not None
    assert "Regression in plain-con1:" in str(error)
    assert "Regression in speculative-con1:" not in str(error)
    assert "Not failing this stage" not in plain["s_regression_info"]
    assert "Not failing this stage" in speculative["s_regression_info"]
    assert "exempt as a whole" in speculative["s_regression_info"]
    assert "predates the change" not in speculative["s_regression_info"]


@pytest.mark.parametrize(
    "main_ttft,main_throughput,ttft,throughput,should_fail",
    [
        pytest.param(125.0, 100.0, 100.0, 75.0, True, id="recovered-ttft-new-throughput"),
        pytest.param(125.0, 100.0, 125.0, 75.0, True, id="inherited-ttft-new-throughput"),
        pytest.param(100.0, 75.0, 125.0, 75.0, True, id="inherited-throughput-new-ttft"),
        pytest.param(125.0, 100.0, 125.0, 100.0, False, id="inherited-ttft"),
        pytest.param(100.0, 75.0, 100.0, 75.0, False, id="inherited-throughput"),
        pytest.param(125.0, 75.0, 125.0, 100.0, False, id="both-inherited-only-ttft-misses"),
        pytest.param(125.0, 75.0, 100.0, 75.0, False, id="both-inherited-only-throughput-misses"),
    ],
)
def test_ttft_caller_keeps_independent_regressions(
    perf_sanity_caller: Callable,
    main_ttft: float,
    main_throughput: float,
    ttft: float,
    throughput: float,
    should_fail: bool,
) -> None:
    """TTFT and grouped throughput losses cannot excuse each other."""
    latest = _ttft_history(d_mean_ttft=main_ttft, d_token_throughput=main_throughput)
    older = _ttft_history(**{"@timestamp": "2026-01-01T00:00:00.000Z", "ts_created": 1767225600000})
    uploaded, error = perf_sanity_caller([older, latest], ttft=ttft, throughput=throughput)
    assert uploaded[0]["b_is_regression"] is True
    assert (error is not None) is should_fail
    info = uploaded[0]["s_regression_info"]
    if should_fail:
        assert "Not failing this stage" not in info
        assert "Latest matched post-merge gated misses among current comparisons:" in info
        assert "new gated misses:" in info
    else:
        warning = next(line for line in info.splitlines() if "Not failing this stage" in line)
        excusing_metric = "d_mean_ttft" if ttft > 110.0 else "d_token_throughput"
        assert f"({excusing_metric})" in warning


def test_ttft_caller_groups_throughput_exemptions(perf_sanity_caller: Callable) -> None:
    """Different stored throughput gates retain their shared mainline exemption."""
    latest = _ttft_history(
        d_token_throughput=75.0,
        d_total_token_throughput=1150.0,
        d_baseline_total_token_throughput=1200.0,
    )
    older = _ttft_history(
        **{
            "@timestamp": "2026-01-01T00:00:00.000Z",
            "ts_created": 1767225600000,
            "d_baseline_total_token_throughput": 1200.0,
        }
    )
    uploaded, error = perf_sanity_caller([older, latest], ttft=100.0, throughput=75.0)
    assert _metric_comparison(uploaded[0], "d_total_token_throughput") == pytest.approx(
        (1000.0, 1200.0)
    )
    assert uploaded[0]["b_is_regression"] is True
    assert error is None
    warning = next(
        line
        for line in uploaded[0]["s_regression_info"].splitlines()
        if "Not failing this stage" in line
    )
    assert "(d_token_throughput)" in warning


def test_metric_override_does_not_change_exemption_policy(
    regression_utils: ModuleType,
    metric_lists: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting the default gates explicitly must preserve whole-case exemptions."""
    latest = _ttft_history(d_token_throughput=75.0, d_total_token_throughput=1150.0)
    baseline = _ttft_history(d_baseline_total_token_throughput=1200.0)
    monkeypatch.setattr(regression_utils, "get_job_info", lambda: {"b_is_post_merge": False})
    monkeypatch.setattr(
        regression_utils,
        "get_history_data",
        lambda *args: ({0: latest}, {0: baseline}, {}),
    )
    new_data = {0: {"d_token_throughput": 75.0, "d_total_token_throughput": 1000.0}}
    regression_utils.process_and_upload_test_results(
        new_data_dict=new_data,
        match_keys=[],
        maximize_metrics=metric_lists["MAXIMIZE_METRICS"],
        minimize_metrics=metric_lists["MINIMIZE_METRICS"],
        regression_metrics=metric_lists["REGRESSION_METRICS"],
        regression_metrics_by_cmd_idx={0: metric_lists["REGRESSION_METRICS"]},
        upload_to_db=False,
    )
    assert new_data[0]["b_is_regression"] is True
    assert "Not failing this stage" in new_data[0]["s_regression_info"]


def test_branchless_exemption_describes_matched_record(
    regression_utils: ModuleType,
    metric_lists: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A branchless caller must not attribute a matched record to the baseline branch."""
    monkeypatch.setenv("PERF_BASELINE_BRANCH", "main")
    latest = _ttft_history(s_branch="release/example", d_token_throughput=75.0)
    monkeypatch.setattr(
        regression_utils, "get_job_info", lambda: {"b_is_post_merge": False, "s_branch": "feature"}
    )
    monkeypatch.setattr(
        regression_utils, "get_history_data", lambda *args: ({0: latest}, {0: latest}, {})
    )
    data = {0: {"s_test_case_name": "plain-con1", "d_token_throughput": 75.0}}
    regression_utils.process_and_upload_test_results(
        new_data_dict=data,
        match_keys=["s_test_case_name"],
        maximize_metrics=metric_lists["MAXIMIZE_METRICS"],
        minimize_metrics=metric_lists["MINIMIZE_METRICS"],
        regression_metrics=metric_lists["REGRESSION_METRICS"],
        upload_to_db=False,
    )
    assert data[0]["b_is_regression"] is True
    info = data[0]["s_regression_info"]
    assert "Not failing this stage" in info
    assert "latest matched post-merge record" in info
    assert "baseline branch" not in info


@pytest.mark.parametrize(
    "parameter", ["regression_metrics_by_cmd_idx", "independent_regression_metrics_by_cmd_idx"]
)
def test_per_case_regression_metrics_are_validated(
    regression_utils: ModuleType, metric_lists: dict[str, list[str]], parameter: str
) -> None:
    """Per-case overrides must use the same metric definitions as default gates."""
    with pytest.raises(ValueError, match=rf"{parameter}\[0\].*d_unknown_metric"):
        regression_utils.process_and_upload_test_results(
            new_data_dict={0: {}},
            match_keys=[],
            maximize_metrics=metric_lists["MAXIMIZE_METRICS"],
            minimize_metrics=metric_lists["MINIMIZE_METRICS"],
            regression_metrics=metric_lists["REGRESSION_METRICS"],
            **{parameter: {0: ["d_unknown_metric"]}},
        )


@pytest.mark.parametrize("metric", _TPOT_METRICS)
@pytest.mark.parametrize("gate_tpot", [False, True])
@pytest.mark.parametrize(
    "value, expected_diff, is_slowdown",
    [(15.24, -150.6578947368421, True), (3.04, 50.0, False)],
)
def test_tpot_report_and_gate(
    regression_utils: ModuleType,
    metric_lists: dict[str, list[str]],
    metric: str,
    gate_tpot: bool,
    value: float,
    expected_diff: float,
    is_slowdown: bool,
) -> None:
    """Honor a stored baseline while applying latency signs and the selected gate."""
    data = {0: {metric: value}}
    regression_utils.prepare_regressive_test_cases(
        latest_history_data_dict={},
        latest_baseline_threshold_dict={0: {f"d_baseline_{metric[2:]}": 6.08}},
        history_data_dict={},
        new_data_dict=data,
        maximize_metrics=metric_lists["MAXIMIZE_METRICS"],
        minimize_metrics=metric_lists["MINIMIZE_METRICS"],
        regression_metrics=[metric] if gate_tpot else metric_lists["REGRESSION_METRICS"],
    )
    assert f"diff={expected_diff:+.2f}%" in data[0]["s_regression_info"]
    assert data[0]["b_is_regression"] is (gate_tpot and is_slowdown)


@pytest.mark.parametrize("metric", _TPOT_METRICS)
def test_tpot_baseline_uses_low_latency_percentile(
    regression_utils: ModuleType, metric_lists: dict[str, list[str]], metric: str
) -> None:
    """Compute P5 from historical latency measurements, including older samples."""
    history = [
        {"ts_created": "2026-10-01", metric: 6.0},
        {"ts_created": "2026-10-02", metric: 12.0},
        {"ts_created": "2026-10-03", metric: 18.0},
    ]
    baseline = regression_utils.calculate_baseline_metrics(
        history, None, metric_lists["MAXIMIZE_METRICS"], metric_lists["MINIMIZE_METRICS"]
    )
    # The trailing means are 6, 9, 12; their P5 is 6.3 (P95 would be 11.7).
    assert baseline[metric] == pytest.approx(6.3)


@pytest.mark.parametrize("metric", _TPOT_METRICS)
def test_calculated_tpot_baseline_is_not_persisted_as_an_override(
    regression_utils: ModuleType,
    metric_lists: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
) -> None:
    """Reclassifying old measurements recomputes P5 instead of inheriting a calculated P95."""
    history = [
        {"ts_created": "2026-10-01", metric: 6.0},
        {"ts_created": "2026-10-02", metric: 12.0},
        {"ts_created": "2026-10-03", metric: 18.0},
    ]
    uploaded = []
    monkeypatch.setattr(regression_utils, "get_job_info", lambda: {"b_is_post_merge": True})
    monkeypatch.setattr(
        regression_utils, "post_new_perf_data", lambda payload: uploaded.append(deepcopy(payload))
    )
    stored_fields = {}
    for maximize, expected_baseline in ((True, 11.7), (False, 6.3)):
        monkeypatch.setattr(
            regression_utils,
            "get_history_data",
            lambda *args: ({}, {0: stored_fields}, {0: history}),
        )
        data = {0: {metric: 9.0}}
        regression_utils.process_and_upload_test_results(
            new_data_dict=data,
            match_keys=[],
            maximize_metrics=[metric] if maximize else metric_lists["MAXIMIZE_METRICS"],
            minimize_metrics=[] if maximize else metric_lists["MINIMIZE_METRICS"],
            regression_metrics=metric_lists["REGRESSION_METRICS"] if not maximize else [],
        )
        assert uploaded[-1] == data
        uploaded_record = uploaded[-1][0]
        assert f"baseline={expected_baseline:.4f}" in uploaded_record["s_regression_info"]
        assert uploaded_record["b_is_regression"] is False
        stored_fields = {
            key: value for key, value in uploaded_record.items() if key.startswith("d_baseline_")
        }
        assert not stored_fields
    assert "diff=-42.86%" in uploaded[-1][0]["s_regression_info"]
