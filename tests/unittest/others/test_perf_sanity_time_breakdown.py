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
"""Tests for the ``time_breakdown`` perf-sanity modifier's metric plumbing.

The mode has a three-hop contract, and each hop fails *silently* if it drifts:

1. ``benchmark_serving`` prints one ``Time Breakdown <span> <stat> (ms): <v>``
   line per span/statistic, ``test_perf_sanity`` scrapes those lines back out of
   the captured stdout with a regex, and a mismatch between the two formats
   yields zero parsed metrics -- an upload with no ``d_tb_*`` fields, which on a
   dashboard is indistinguishable from a case that simply has no breakdown.
2. The metric names come from ``time_breakdown_metrics``, not from
   ``TimingMetricsConfig``, because test collection must not import
   ``tensorrt_llm``. Nothing at runtime notices if the two go stale relative to
   each other: a span the tool emits but the harness does not know still
   uploads, just with no baseline, so the drift is invisible until someone looks
   for a missing history line.
3. Every uploaded name must be registered in ``MINIMIZE_METRICS`` (so it gets a
   baseline) and must *not* be in ``REGRESSION_METRICS`` (so it cannot fail a
   build). ``perf_regression_utils`` asserts the first relationship at import
   time; nothing asserts the second.
"""

import importlib.util
import json
import os
import pathlib
import sys
import types

import pytest

from tensorrt_llm.serve.scripts.time_breakdown import TimingMetricsConfig

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "tests" / "integration" / "defs" / "perf" / "test_perf_sanity.py"


def _load_test_perf_sanity():
    """Load the harness module without importing the integration-test packages.

    ``defs/__init__.py`` pre-imports ``torch._inductor`` and
    ``perf_regression_utils`` pulls in the OpenSearch client. None of the module
    level constants or the two pure functions under test reach either, and
    requiring them would turn this into a GPU-image test.
    """
    defs_pkg = types.ModuleType("defs")
    defs_pkg.__path__ = []
    perf_pkg = types.ModuleType("defs.perf")
    # Real path, so ``from .time_breakdown_metrics import ...`` resolves to the
    # actual module -- it is stdlib-only, so loading it costs nothing and the
    # metric names under test are the ones the harness really uses. The heavy
    # siblings stay stubbed via sys.modules below, which wins over this path.
    perf_pkg.__path__ = [str(_MODULE_PATH.parent)]

    conftest = types.ModuleType("defs.conftest")
    conftest.get_llm_root = lambda *a, **k: ""
    conftest.llm_models_root = lambda *a, **k: ""

    common = types.ModuleType("defs.common")
    common.wait_for_reported_addr = lambda *a, **k: None

    alternative = types.ModuleType("defs.trt_test_alternative")
    alternative.print_info = lambda *a, **k: None
    alternative.print_warning = lambda *a, **k: None

    model_paths = types.ModuleType("defs.perf._model_paths")
    model_paths.MODEL_PATH_DICT = {}

    regression = types.ModuleType("defs.perf.perf_regression_utils")
    regression._percentile = lambda *a, **k: 0.0
    regression.process_and_upload_test_results = lambda *a, **k: None

    test_common = types.ModuleType("test_common")
    test_common.__path__ = []
    error_utils = types.ModuleType("test_common.error_utils")
    error_utils.report_error = lambda *a, **k: None
    http_utils = types.ModuleType("test_common.http_utils")
    http_utils.fail_if_proc_died = lambda *a, **k: None
    http_utils.wait_for_endpoint_ready = lambda *a, **k: None
    matching = types.ModuleType("test_common.perf_sanity_matching")
    matching.get_test_case_match_keys = lambda *a, **k: {}

    stubs = {
        "defs": defs_pkg,
        "defs.conftest": conftest,
        "defs.common": common,
        "defs.trt_test_alternative": alternative,
        "defs.perf": perf_pkg,
        "defs.perf._model_paths": model_paths,
        "defs.perf.perf_regression_utils": regression,
        "test_common": test_common,
        "test_common.error_utils": error_utils,
        "test_common.http_utils": http_utils,
        "test_common.perf_sanity_matching": matching,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("defs.perf.test_perf_sanity", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules["defs.perf.test_perf_sanity"] = module
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("defs.perf.test_perf_sanity", None)
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


_sanity = _load_test_perf_sanity()


@pytest.fixture(autouse=True)
def _no_settle_wait(monkeypatch):
    """Collapse the perf_metrics settle window; these tests write their files up front.

    The window itself is covered directly in test_time_breakdown_metrics.py, with an
    injected clock. Paying it here would only add wall time per call.
    """
    monkeypatch.setattr(_sanity, "PERF_METRICS_SETTLE_SECONDS", 0.0)


def _format_line(span: str, stat: str, value: float) -> str:
    """Reproduce exactly what ``benchmark_serving.main`` prints.

    Kept as a helper, and deliberately duplicated from the producer rather than
    imported, so that a change on either side of the contract makes this test
    fail rather than making both sides agree on something new.
    """
    return f"Time Breakdown {span} {stat} (ms): {value:.4f}"


def test_every_tool_span_is_known_to_the_harness():
    """The tool's spans must all be metrics the harness has a baseline for.

    A subset rather than an equality: the harness also uploads the per-chunk and
    per-step breakdowns, which the tool has no span for. What must not happen is
    the tool emitting a span the harness cannot name -- that span would upload
    with no baseline and no history line.
    """
    tool_spans = tuple(m.name for m in TimingMetricsConfig().metrics)
    unknown = set(tool_spans) - set(_sanity.TIME_BREAKDOWN_METRIC_NAMES)
    assert not unknown, f"spans the tool emits but the harness does not know: {sorted(unknown)}"


def test_metric_names_cover_every_metric_and_statistic():
    assert len(_sanity.TIME_BREAKDOWN_METRICS) == (
        len(_sanity.TIME_BREAKDOWN_METRIC_NAMES) * len(_sanity.TIME_BREAKDOWN_STATS)
    )
    assert len(set(_sanity.TIME_BREAKDOWN_METRICS)) == len(_sanity.TIME_BREAKDOWN_METRICS)
    for name in _sanity.TIME_BREAKDOWN_METRICS:
        assert name.startswith("tb_")


def test_every_benchmark_mode_is_one_the_parser_supports():
    """The case type IS the benchmark mode now -- no map in between.

    ``_append_time_breakdown_metrics`` feeds ``record["benchmark_mode"]``
    straight to the parser, so the two vocabularies have to be the same set. A
    mode the parser did not know would silently skip aggregation instead of
    uploading, and the run would fail only on the "parsed no lines" check.
    """
    from defs.perf.time_breakdown_metrics import MODE_GROUPS

    assert set(MODE_GROUPS) == {"ctx_only", "gen_only", "e2e"}


@pytest.mark.parametrize("stat", ["mean", "median", "p75", "p99"])
def test_regex_round_trips_every_printed_line(stat):
    """Every metric/stat line the client prints must parse back to its metric."""
    for span in _sanity.TIME_BREAKDOWN_METRIC_NAMES:
        line = _format_line(span, stat, 12.5)
        match = _sanity.TIME_BREAKDOWN_METRIC_LOG_QUERY.search(line)
        assert match is not None, line
        parsed_span, parsed_stat, value = match.groups()
        assert parsed_span == span
        assert parsed_stat == stat
        assert float(value) == pytest.approx(12.5)
        assert (
            _sanity.time_breakdown_metric_name(parsed_span, parsed_stat)
            in _sanity.TIME_BREAKDOWN_METRICS
        )


def test_regex_round_trips_the_harness_aggregator_output():
    """The second producer's lines must parse back too, negatives included.

    ``_append_time_breakdown_metrics`` appends ``format_metric_log_lines`` output
    to the benchmark log and the harness scrapes it back with the same regex. The
    format differs from ``benchmark_serving``'s (6 decimals, not 4), and
    ``tb_step_preprocessing`` is negative whenever the overlap scheduler is on --
    a regex that only accepted unsigned values would drop exactly the metric that
    proves overlap is working, on every row, forever.
    """
    from defs.perf.time_breakdown_metrics import format_metric_log_lines

    values = {
        f"d_{name}": (-11.718660 if name == "tb_step_preprocessing_mean" else 12.5)
        for name in _sanity.TIME_BREAKDOWN_METRICS
    }
    # format_metric_log_lines keys off d_tb_<metric>_<stat>, the upload name.
    lines = format_metric_log_lines(values)
    assert len(lines) == len(_sanity.TIME_BREAKDOWN_METRICS)

    parsed = {}
    for line in lines:
        match = _sanity.TIME_BREAKDOWN_METRIC_LOG_QUERY.search(line)
        assert match is not None, line
        span, stat, value = match.groups()
        parsed[_sanity.time_breakdown_metric_name(span, stat)] = float(value)

    assert set(parsed) == set(_sanity.TIME_BREAKDOWN_METRICS)
    assert parsed["tb_step_preprocessing_mean"] == pytest.approx(-11.718660)
    assert parsed["tb_ctx_queue_median"] == pytest.approx(12.5)


def test_regex_tolerates_a_log_prefix_and_a_zero_value():
    """Real benchmark stdout is line-prefixed by the launcher and by srun."""
    line = "[2026-08-31 04:05:06] [Rank 0] " + _format_line("gen_kv_transfer", "p99", 0.0)
    match = _sanity.TIME_BREAKDOWN_METRIC_LOG_QUERY.search(line)
    assert match is not None
    assert match.group(1) == "gen_kv_transfer"
    assert float(match.group(3)) == 0.0


def test_regex_ignores_an_unknown_statistic():
    """An unlisted statistic must not be uploaded under a mangled name."""
    assert (
        _sanity.TIME_BREAKDOWN_METRIC_LOG_QUERY.search(_format_line("ctx_queue", "stddev", 1.0))
        is None
    )


def test_regex_captures_a_span_the_harness_does_not_know_about():
    """A span added to the tool still reaches OpenSearch (without a baseline).

    This is why the query uses capture groups instead of one literal pattern
    per metric/statistic pair:
    extending TimingMetricsConfig should never silently drop data.
    """
    match = _sanity.TIME_BREAKDOWN_METRIC_LOG_QUERY.search(
        _format_line("some_future_span", "median", 3.25)
    )
    assert match is not None
    assert _sanity.time_breakdown_metric_name(*match.groups()[:2]) == ("tb_some_future_span_median")


def test_every_metric_is_a_minimize_metric_and_gates_nothing():
    for name in _sanity.TIME_BREAKDOWN_METRICS:
        assert f"d_{name}" in _sanity.MINIMIZE_METRICS
        assert f"d_{name}" not in _sanity.MAXIMIZE_METRICS
        assert f"d_{name}" not in _sanity.REGRESSION_METRICS


def _parsed_metrics(**extra):
    """A fully populated parse result plus whatever the caller adds.

    ``add_perf_metric_value`` indexes every ``PERF_METRIC_LOG_QUERIES`` key
    unconditionally, so a partial dict would raise ``KeyError`` for reasons
    unrelated to what these cases are checking.
    """
    metrics = {name: 0.0 for name in _sanity.PERF_METRIC_LOG_QUERIES}
    metrics.update(extra)
    return metrics


def test_add_perf_metric_value_uploads_only_with_the_modifier():
    metrics = _parsed_metrics(tb_ctx_queue_median=4.5, tb_gen_kv_transfer_p99=41.25)

    new_data = {}
    _sanity.add_perf_metric_value(
        new_data, metrics, spec_decoding=False, benchmark_mode="e2e", time_breakdown=True
    )
    assert new_data["d_tb_ctx_queue_median"] == pytest.approx(4.5)
    assert new_data["d_tb_gen_kv_transfer_p99"] == pytest.approx(41.25)

    # The same parsed dict without the modifier must not grow any d_tb_* field:
    # the two cases share every other metric name, and a plain e2e case that
    # started uploading breakdown fields would fork its own history series.
    e2e_data = {}
    _sanity.add_perf_metric_value(e2e_data, metrics, spec_decoding=False, benchmark_mode="e2e")
    assert not [key for key in e2e_data if key.startswith("d_tb_")]


def test_add_perf_metric_value_skips_a_missing_span():
    """A span with no measured requests is omitted, never uploaded as 0.0."""
    new_data = {}
    _sanity.add_perf_metric_value(
        new_data,
        _parsed_metrics(tb_gen_kv_transfer_median=None),
        spec_decoding=False,
        benchmark_mode="e2e",
        time_breakdown=True,
    )
    assert "d_tb_gen_kv_transfer_median" not in new_data


# A disaggregated yaml reduced to what the parser actually reads. ctx_only runs
# only the `ctx:` worker, but `gen:` has to be present because the same file is
# what the e2e and gen_only ids read.
_DISAGG_YAML = """
metadata:
  model_name: deepseek-ai/DeepSeek-R1
hardware:
  gpus_per_node: 4
  num_ctx_servers: 1
  num_gen_servers: 1
  nodes_per_ctx_server: 1
  nodes_per_gen_server: 1
  gpus_per_node_per_ctx_server: 4
  gpus_per_node_per_gen_server: 4
benchmark:
  input_length: 8192
  output_length: 1024
  concurrency_list: "666"
worker_config:
  ctx:
    tensor_parallel_size: 4
    cache_transceiver_config:
      backend: NIXL
  gen:
    tensor_parallel_size: 4
"""


def _parsed_config(tmp_path, benchmark_mode: str, time_breakdown: bool):
    """Drive the real parser without constructing a whole pytest fixture graph.

    ``PerfSanityTestConfig.__init__`` derives everything under test from the test
    id; setting the four derived attributes directly keeps the case about the
    parser instead of about id parsing, which the round-trip tests already cover.
    """
    config_path = tmp_path / "gb300_stem-NIXL.yaml"
    config_path.write_text(_DISAGG_YAML, encoding="utf-8")
    config = object.__new__(_sanity.PerfSanityTestConfig)
    config.benchmark_mode = benchmark_mode
    config.time_breakdown = time_breakdown
    config._output_dir = str(tmp_path)
    config._test_param_labels = "case"
    config._parse_disagg_config_file(str(config_path), config_path.name)
    return config


def test_ctx_only_time_breakdown_instruments_its_one_server(tmp_path):
    """The regression this guards: ctx_only runs on the *aggregated* runtime.

    ctx_only is parsed by the disagg parser but executed by AggrTestCmds, so it
    takes the ``else`` branch's worker overrides nowhere -- it builds its own
    single ServerConfig. Without the splat in that branch the case runs perfectly
    green, the server is simply never asked to record any timings, and all 44
    ctx_only ``d_tb_*`` fields upload as 0.0. Nothing downstream can tell that
    apart from a lane that genuinely measured nothing.
    """
    config = _parsed_config(tmp_path, "ctx_only", time_breakdown=True)

    assert len(config.server_configs) == 1
    extra = config.server_configs[0].extra_llm_api_config_data
    assert extra["return_perf_metrics"] is True
    assert extra["num_postprocess_workers"] == 0
    assert extra["perf_metrics_output_dir"] == config.time_breakdown_dir()
    # The directory both halves have to agree on: the server writes here, and
    # append_time_breakdown_metrics scans here.
    assert config.time_breakdown_dir().endswith(os.path.join("case", "perf_metrics"))


def test_ctx_only_without_the_modifier_stays_untouched(tmp_path):
    """The plain ctx_only lane must not acquire any of the three keys.

    num_postprocess_workers: 0 measurably changes throughput, so leaking it into
    the unmodified lane would move that lane's baseline rather than fork a new
    series -- a regression that looks like a real one.
    """
    config = _parsed_config(tmp_path, "ctx_only", time_breakdown=False)

    extra = config.server_configs[0].extra_llm_api_config_data
    for key in ("return_perf_metrics", "perf_metrics_output_dir", "num_postprocess_workers"):
        assert key not in extra
    assert config.time_breakdown_dir() == ""


def test_ctx_only_client_records_the_breakdown(tmp_path):
    """The other half of the contract: the client has to be told to read it back.

    The server writing a JSONL is useless on its own -- benchmark_serving is what
    prints the ``Time Breakdown`` lines the harness scrapes, and it only does that
    when handed --save-request-time-breakdown.
    """
    config = _parsed_config(tmp_path, "ctx_only", time_breakdown=True)
    client_cmd = config.server_client_configs[0][0].to_cmd()

    assert "--save-request-time-breakdown" in client_cmd
    assert client_cmd[client_cmd.index("--save-request-time-breakdown") + 1] == (
        config.time_breakdown_dir()
    )


def test_the_no_lines_guard_is_not_scoped_to_the_disagg_runtime(tmp_path):
    """A runtime predicate on that guard would have exempted exactly ctx_only.

    check_test_failure's "parsed no Time Breakdown lines" check is the only thing
    that turns a silently uninstrumented modified run into a red build. It used to
    be gated on runtime == multi_node_disagg_server, which is false for ctx_only.
    """
    config = _parsed_config(tmp_path, "ctx_only", time_breakdown=True)
    config.runtime = "aggr_server"
    config.gpu_type = "gb300"
    config._perf_results = {0: [_parsed_metrics()]}

    with pytest.raises(RuntimeError, match="parsed no 'Time Breakdown"):
        config.check_test_failure()


def test_the_no_lines_guard_passes_once_a_span_is_present(tmp_path):
    """Same lane, one parsed span: the guard must not fire.

    Individual spans stay ungated on purpose -- a span whose endpoints were never
    populated is legitimately absent -- so presence of any one of them is the
    whole condition.
    """
    config = _parsed_config(tmp_path, "ctx_only", time_breakdown=True)
    config.runtime = "aggr_server"
    config.gpu_type = "gb300"
    config._perf_results = {0: [_parsed_metrics(tb_ctx_queue_mean=1.5)]}

    config.check_test_failure()


def test_get_commands_hands_the_breakdown_dir_to_the_aggregated_runner(tmp_path):
    """ctx_only dispatches to _get_aggr_commands, so AggrTestCmds must carry the dir.

    DisaggTestCmds reads perf_metrics_output_dir off its disagg config; the
    aggregated runner had no equivalent, so this is the hop that decides whether
    run_cmd aggregates at all. An empty string here means every ctx_only
    breakdown run silently skips the reduction.
    """
    config = _parsed_config(tmp_path, "ctx_only", time_breakdown=True)
    config.runtime = "aggr_server"

    cmds = config.get_commands()

    assert isinstance(cmds, _sanity.AggrTestCmds)
    assert cmds.perf_metrics_output_dir == config.time_breakdown_dir()
    assert cmds.benchmark_mode == "ctx_only"

    # And the plain ctx_only lane must leave it empty, which is what makes run_cmd
    # skip the aggregation rather than reduce an empty directory every run.
    plain = _parsed_config(tmp_path, "ctx_only", time_breakdown=False)
    plain.runtime = "aggr_server"
    assert plain.get_commands().perf_metrics_output_dir == ""


def _ctx_worker_jsonl(directory, count=3):
    """A context worker's perf_metrics JSONL, in the shape the server really writes.

    Only the fields the reduction reads: the request-level timing metrics and one
    prefill chunk per request.
    """
    os.makedirs(directory, exist_ok=True)
    lines = []
    for i in range(count):
        t0 = 1000.0 + i
        chunk_start = t0 + 0.010
        lines.append(
            json.dumps(
                {
                    "request_id": i,
                    "perf_metrics": {
                        "timing_metrics": {
                            "server_arrival_time": t0,
                            "arrival_time": t0 + 0.001,
                            "first_scheduled_time": t0 + 0.003,
                            "first_token_time": t0 + 0.130,
                            "server_first_token_time": t0 + 0.131,
                            "last_token_time": t0 + 0.130,
                        }
                    },
                    "time_breakdown_metrics": {
                        "ctx_chunk_metrics": [
                            {
                                "forward_start_time": chunk_start,
                                "forward_end_time": chunk_start + 0.100,
                                "sample_start_time": chunk_start + 0.102,
                                "sample_end_time": chunk_start + 0.103,
                                "token_time": t0 + 0.130,
                                "gpu_forward_time": 98.0,
                                "gpu_sample_time": 0.9,
                            }
                        ]
                    },
                }
            )
        )
    path = os.path.join(directory, "perf_metrics-server-hostA-0-run.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def test_append_time_breakdown_metrics_reduces_a_single_aggregated_worker(tmp_path):
    """The aggregated path has one server file, not a ctx/gen pair.

    The reduction classifies files by content, so it needs no changes for this --
    but the caller does, and this is what proves the shared entry point works
    with a lone ctx-role file and emits lines in the format the scraper matches.
    """
    breakdown_dir = str(tmp_path / "perf_metrics")
    _ctx_worker_jsonl(breakdown_dir)
    benchmark_file = tmp_path / "benchmark.log"
    benchmark_file.write_text("existing client output\n", encoding="utf-8")
    outputs = ["existing client output\n"]

    _sanity.append_time_breakdown_metrics(
        [
            {
                "output_index": 0,
                "benchmark_file_path": str(benchmark_file),
                "benchmark_mode": "ctx_only",
            }
        ],
        outputs,
        breakdown_dir,
    )

    # Both sinks must be written: the parser reads the captured stdout, and the
    # benchmark file is what a human reads afterwards.
    assert "Time Breakdown ctx_processing mean (ms):" in outputs[0]
    assert "Time Breakdown ctx_processing mean (ms):" in benchmark_file.read_text()

    # The whole point of the wiring: the real consumer regex now finds populated
    # ctx spans in what the run captured. Scraped with the harness's own pattern
    # rather than a copy, so a producer/consumer format drift fails here.
    scraped = {}
    for line in outputs[0].splitlines():
        match = _sanity.TIME_BREAKDOWN_METRIC_LOG_QUERY.search(line)
        if match:
            span, stat, value = match.groups()
            scraped[_sanity.time_breakdown_metric_name(span, stat)] = float(value)

    # Every field is emitted every run, so the OpenSearch doc schema never varies.
    assert set(scraped) == set(_sanity.TIME_BREAKDOWN_METRICS)
    assert scraped["tb_ctx_processing_mean"] > 0.0
    assert scraped["tb_chunk_forward_mean"] > 0.0
    # ctx_only populates 44 of the 108; the gen-side groups stay 0.0.
    assert scraped["tb_gen_kv_transfer_mean"] == 0.0
    assert len([v for v in scraped.values() if v != 0.0]) <= 44


def test_append_time_breakdown_metrics_is_a_no_op_without_files(tmp_path):
    """A missing directory must not raise: it is diagnosed by the "no lines" check.

    Aggregation runs inside the test's teardown path. Raising here would replace a
    clear "parsed no Time Breakdown lines" failure with a traceback from cleanup,
    and on the disagg path it would mask whatever actually killed the workers.
    """
    outputs = ["client output\n"]
    benchmark_file = tmp_path / "benchmark.log"
    benchmark_file.write_text("client output\n", encoding="utf-8")

    _sanity.append_time_breakdown_metrics(
        [
            {
                "output_index": 0,
                "benchmark_file_path": str(benchmark_file),
                "benchmark_mode": "ctx_only",
            }
        ],
        outputs,
        str(tmp_path / "absent"),
    )

    assert outputs == ["client output\n"]


def _listed_time_breakdown_ids():
    """Every ``time_breakdown`` perf-sanity id referenced by a CI lane list."""
    test_db = _REPO_ROOT / "tests" / "integration" / "test_lists" / "test-db"
    found = []
    for path in sorted(test_db.glob("*.yml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("- perf/test_perf_sanity.py::"):
                continue
            if f"-{_sanity.TIME_BREAKDOWN_MODIFIER}-" not in stripped:
                continue
            found.append((path.name, stripped.split("[", 1)[1].split("]", 1)[0]))
    return found


def test_every_listed_breakdown_lane_is_an_id_the_harness_generates():
    """A lane id the collector never produces is an error at collection, not a skip.

    The modified ids are allowlisted per config stem, so a lane list and an
    allowlist can disagree in either direction: an id in a list but not in the
    allowlist fails the whole stage, and an allowlisted stem no list references
    is a case that is generated but never run. Both are invisible until CI runs,
    which for a post_merge perf stage is a slow way to find out.
    """
    listed = _listed_time_breakdown_ids()
    assert listed, "no time_breakdown lane found; did the lists move?"

    allowlists = {
        "e2e": _sanity.E2E_TIME_BREAKDOWN_CONFIGS,
        "ctx_only": _sanity.CTX_ONLY_TIME_BREAKDOWN_CONFIGS,
    }
    seen = set()
    for list_name, test_id in listed:
        stem, _pattern, _runtime, mode, time_breakdown = _sanity.parse_test_string(test_id)
        assert time_breakdown, f"{list_name}: {test_id} did not parse as a modified id"
        assert mode in allowlists, f"{list_name}: {test_id} has unsupported mode {mode!r}"
        assert stem in allowlists[mode], (
            f"{list_name}: {test_id} is not generated -- {stem!r} is missing from the "
            f"{mode} allowlist"
        )
        seen.add((mode, stem))

    # And the reverse direction: nothing is allowlisted but unreferenced.
    for mode, stems in allowlists.items():
        for stem in stems:
            assert (mode, stem) in seen, f"{mode} breakdown case {stem!r} is in no lane list"


# Loading the harness imported this for real (test_perf_sanity does
# ``from .time_breakdown_metrics import ...``), and _load_test_perf_sanity only restores the
# names it stubbed -- which this is not -- so it is still registered. Taken from sys.modules
# rather than imported: ``from defs.perf import time_breakdown_metrics`` would need the
# ``defs.perf`` parent, and that stub *was* restored away.
_tbm = sys.modules["defs.perf.time_breakdown_metrics"]


def _burst_ctx_worker_jsonl(directory, count=4, warmup=False):
    """A ctx worker's JSONL whose measured requests overlap, as a real lane's do.

    The measured requests arrive within 50 ms of each other and each takes 500 ms, so
    every one of them overlaps its neighbour -- the property _drop_warmup_record relies
    on to tell them apart from a warmup request. With ``warmup=True`` an additional
    request is prepended that arrives 10 s earlier and completes 9.5 s before the burst
    opens, which is what benchmark_serving's "initial single prompt test run" looks like.

    Returns the path written.
    """
    os.makedirs(directory, exist_ok=True)

    def record(request_id, t0, forward_ms):
        chunk_start = t0 + 0.010
        return {
            "request_id": request_id,
            "perf_metrics": {
                "timing_metrics": {
                    "server_arrival_time": t0,
                    "arrival_time": t0 + 0.001,
                    "first_scheduled_time": t0 + 0.003,
                    "first_token_time": t0 + 0.500,
                    "server_first_token_time": t0 + 0.501,
                    "last_token_time": t0 + 0.500,
                }
            },
            "time_breakdown_metrics": {
                "ctx_chunk_metrics": [
                    {
                        "forward_start_time": chunk_start,
                        "forward_end_time": chunk_start + forward_ms / 1000.0,
                        "sample_start_time": chunk_start + 0.402,
                        "sample_end_time": chunk_start + 0.403,
                        "token_time": t0 + 0.500,
                        "gpu_forward_time": forward_ms,
                        "gpu_sample_time": 0.9,
                    }
                ]
            },
        }

    records = []
    if warmup:
        # Deliberately pathological, the way a cold first request is: ~4x the forward
        # time of a measured request. If it is not excluded it moves the mean.
        records.append(record("warmup", 1000.0 - 10.0, forward_ms=1600.0))
    records.extend(record(i, 1000.0 + i * 0.05, forward_ms=400.0) for i in range(count))

    path = os.path.join(directory, "perf_metrics-server-host-1-20260101T000000Z.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for raw in records:
            handle.write(json.dumps(raw) + "\n")
    return path


def test_the_warmup_request_is_excluded_and_the_measured_population_recovered(tmp_path):
    """Dropping the warmup record must reproduce the warmup-free run exactly."""
    clean = _burst_ctx_worker_jsonl(str(tmp_path / "clean"), warmup=False)
    dirty = _burst_ctx_worker_jsonl(str(tmp_path / "dirty"), warmup=True)

    baseline, _ = _tbm.compute_time_breakdown_metrics([clean], "ctx_only")
    left_in, _ = _tbm.compute_time_breakdown_metrics([dirty], "ctx_only")
    dropped, info = _tbm.compute_time_breakdown_metrics(
        [dirty], "ctx_only", drop_warmup_request=True
    )

    # The warmup request has to actually perturb the result, or this proves nothing.
    assert left_in["d_tb_chunk_gpu_forward_mean"] != baseline["d_tb_chunk_gpu_forward_mean"]
    assert dropped == baseline
    assert sum(info["warmup_dropped"].values()) == 1


def test_a_burst_of_measured_requests_is_never_mistaken_for_a_warmup_request(tmp_path):
    """The guard is isolation: overlapping requests must all survive."""
    clean = _burst_ctx_worker_jsonl(str(tmp_path / "clean"), warmup=False)

    kept, info = _tbm.compute_time_breakdown_metrics([clean], "ctx_only", drop_warmup_request=True)
    untouched, _ = _tbm.compute_time_breakdown_metrics([clean], "ctx_only")

    assert info["warmup_dropped"] == {}
    assert kept == untouched


def test_a_single_record_file_is_never_emptied(tmp_path):
    """One record cannot be shown to be isolated, and dropping it would erase the file."""
    path = _burst_ctx_worker_jsonl(str(tmp_path / "one"), count=1, warmup=False)

    metrics, info = _tbm.compute_time_breakdown_metrics(
        [path], "ctx_only", drop_warmup_request=True
    )

    assert info["warmup_dropped"] == {}
    assert metrics["d_tb_ctx_processing_mean"] > 0.0


def test_the_harness_forwards_the_clients_warmup_flag(tmp_path):
    """append_time_breakdown_metrics must honour the pending record's warmup flag."""
    breakdown_dir = str(tmp_path / "tb")
    _burst_ctx_worker_jsonl(breakdown_dir, warmup=True)

    def scraped(warmup):
        benchmark_file = tmp_path / f"benchmark-{warmup}.log"
        benchmark_file.write_text("client output\n", encoding="utf-8")
        outputs = ["client output\n"]
        _sanity.append_time_breakdown_metrics(
            [
                {
                    "output_index": 0,
                    "benchmark_file_path": str(benchmark_file),
                    "benchmark_mode": "ctx_only",
                    "warmup": warmup,
                }
            ],
            outputs,
            breakdown_dir,
        )
        return {
            match.group(1) + "_" + match.group(2): float(match.group(3))
            for match in _sanity.TIME_BREAKDOWN_METRIC_LOG_QUERY.finditer(outputs[0])
        }

    with_warmup = scraped(True)
    without = scraped(False)

    assert with_warmup and without
    # Same schema either way; only the values move.
    assert set(with_warmup) == set(without)
    assert with_warmup["chunk_gpu_forward_mean"] != without["chunk_gpu_forward_mean"]
    # Excluding the cold request must lower the mean forward time, not raise it.
    assert with_warmup["chunk_gpu_forward_mean"] < without["chunk_gpu_forward_mean"]
