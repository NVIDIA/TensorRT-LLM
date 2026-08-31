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
"""Tests for the ``e2e_time_breakdown`` perf-sanity mode's metric plumbing.

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


def test_every_case_type_the_harness_maps_is_one_the_parser_supports():
    """A mode mapped to an unsupported case type would upload 108 zeros."""
    from defs.perf.time_breakdown_metrics import MODE_GROUPS

    for mode, case_type in _sanity.TIME_BREAKDOWN_CASE_TYPE.items():
        assert case_type in MODE_GROUPS, f"{mode} -> {case_type}"


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


def test_add_perf_metric_value_uploads_only_in_the_new_mode():
    metrics = _parsed_metrics(tb_ctx_queue_median=4.5, tb_gen_kv_transfer_p99=41.25)

    new_data = {}
    _sanity.add_perf_metric_value(
        new_data, metrics, spec_decoding=False, benchmark_mode=_sanity.E2E_TIME_BREAKDOWN_MODE
    )
    assert new_data["d_tb_ctx_queue_median"] == pytest.approx(4.5)
    assert new_data["d_tb_gen_kv_transfer_p99"] == pytest.approx(41.25)

    # The same parsed dict in e2e mode must not grow any d_tb_* field: the two
    # modes share every other metric name, and an e2e case that started
    # uploading breakdown fields would fork its own history series.
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
        benchmark_mode=_sanity.E2E_TIME_BREAKDOWN_MODE,
    )
    assert "d_tb_gen_kv_transfer_median" not in new_data
