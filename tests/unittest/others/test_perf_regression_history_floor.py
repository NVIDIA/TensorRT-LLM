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
"""A metric's history floor keeps rows recorded under an older metric definition out of its baseline.

The gen_only gen-worker device step time switched from the modal num_generation_tokens
bucket to the expected steady-state bucket. On a run with two near-tied buckets the two
definitions publish values ~3x apart, so a single old row in the 90-day history pulls the
P5 baseline down to the short bucket and every current run is flagged.
"""

import ast
import importlib.util
import pathlib
import sys
import types
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.cpu_only

_PERF_DIR = pathlib.Path(__file__).resolve().parents[2] / "integration" / "defs" / "perf"

STEP = "d_mean_gen_worker_per_iter_device_step_time"
MEDIAN = "d_median_gen_worker_per_iter_device_step_time"
THROUGHPUT = "d_total_token_throughput"


def _load_perf_regression_utils():
    """Load the real module without importing the integration-test packages (torch, OpenSearch)."""
    defs_pkg = types.ModuleType("defs")
    defs_pkg.__path__ = []
    alternative = types.ModuleType("defs.trt_test_alternative")
    alternative.print_info = lambda *args, **kwargs: None
    alternative.print_warning = lambda *args, **kwargs: None
    perf_pkg = types.ModuleType("defs.perf")
    perf_pkg.__path__ = []
    db_utils = types.ModuleType("defs.perf.open_search_db_utils")
    for name in ("add_id", "get_history_data", "post_new_perf_data"):
        setattr(db_utils, name, lambda *args, **kwargs: None)

    stubs = {
        "defs": defs_pkg,
        "defs.trt_test_alternative": alternative,
        "defs.perf": perf_pkg,
        "defs.perf.open_search_db_utils": db_utils,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "defs.perf.perf_regression_utils", _PERF_DIR / "perf_regression_utils.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


pru = _load_perf_regression_utils()


def _harness_constants():
    """Evaluate the floor constants straight from test_perf_sanity.py's source.

    Importing the module pulls in torch and the integration conftest, so the three
    assignments the floor is built from are evaluated on their own instead.
    """
    tree = ast.parse((_PERF_DIR / "test_perf_sanity.py").read_text())
    wanted = (
        "DEVICE_STEP_TIME_METRICS",
        "GEN_ONLY_REGRESSION_METRICS",
        "GEN_ONLY_DEVICE_STEP_TIME_HISTORY_FLOOR_MS",
        "GEN_ONLY_METRIC_HISTORY_FLOORS_MS",
    )
    namespace = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in wanted
        ):
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), "test_perf_sanity", "exec"),
                namespace,
            )
    missing = set(wanted) - namespace.keys()
    assert not missing, f"test_perf_sanity.py no longer defines {sorted(missing)}"
    return namespace


def _ms(year, month, day, hour=12):
    return int(datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp() * 1000)


FLOOR = _ms(2026, 10, 1, 12)


def _row(ts_ms, step, median=None, throughput=None):
    row = {"ts_created": ts_ms, "b_is_baseline": False, STEP: step}
    if median is not None:
        row[MEDIAN] = median
    if throughput is not None:
        row[THROUGHPUT] = throughput
    return row


def _two_wave_history():
    """The nvbug 6925906 shape: a few old-definition rows on the short bucket, the rest ~98 ms."""
    old = [_row(_ms(2026, 9, d), 98.4, throughput=520000.0) for d in range(20, 30)]
    for d in (23, 25, 26, 27):
        old.append(_row(_ms(2026, 9, d, 18), 31.6, throughput=518000.0))
    new = [_row(_ms(2026, 10, d), 98.5, throughput=517000.0) for d in range(2, 14)]
    return old, new


def test_old_definition_rows_pull_the_baseline_down_without_a_floor():
    old, new = _two_wave_history()
    baseline = pru.calculate_baseline_metrics(old + new, None, [], [STEP])[STEP]

    # Every current run sits at the steady 98.5 ms, yet each one is flagged.
    assert pru._is_regressive(98.5, baseline, pru.POST_MERGE_THRESHOLD, maximize=False)


def test_floor_drops_rows_created_before_it():
    old, new = _two_wave_history()
    baseline = pru.calculate_baseline_metrics(
        old + new, None, [], [STEP], metric_history_floors_ms={STEP: FLOOR}
    )[STEP]

    assert baseline == pytest.approx(98.5)
    assert not pru._is_regressive(98.5, baseline, pru.POST_MERGE_THRESHOLD, maximize=False)


def test_a_real_slowdown_after_the_floor_is_still_caught():
    old, new = _two_wave_history()
    baseline = pru.calculate_baseline_metrics(
        old + new, None, [], [STEP], metric_history_floors_ms={STEP: FLOOR}
    )[STEP]

    assert pru._is_regressive(98.5 * 1.10, baseline, pru.POST_MERGE_THRESHOLD, maximize=False)


def test_floor_is_per_metric():
    """An unfloored metric keeps its whole history, including the old rows."""
    old, new = _two_wave_history()
    floored = pru.calculate_baseline_metrics(
        old + new, None, [THROUGHPUT], [STEP], metric_history_floors_ms={STEP: FLOOR}
    )
    unfloored = pru.calculate_baseline_metrics(old + new, None, [THROUGHPUT], [STEP])

    assert floored[THROUGHPUT] == unfloored[THROUGHPUT]
    assert floored[STEP] != unfloored[STEP]


def test_metric_with_no_history_above_its_floor_gets_no_baseline():
    """No baseline means prepare_regressive_test_cases skips the metric rather than gating it."""
    old, _ = _two_wave_history()
    baselines = pru.calculate_baseline_metrics(
        old, None, [], [STEP], metric_history_floors_ms={STEP: FLOOR}
    )

    assert STEP not in baselines


def test_new_record_without_ts_created_does_not_seed_a_floored_baseline():
    """The record under test carries no ts_created, so it cannot become its own baseline."""
    old, _ = _two_wave_history()
    baselines = pru.calculate_baseline_metrics(
        old, {STEP: 98.5}, [], [STEP], metric_history_floors_ms={STEP: FLOOR}
    )

    assert STEP not in baselines


@pytest.mark.parametrize(
    ("ts_created", "kept"),
    [
        pytest.param(FLOOR, True, id="at-floor-ms"),
        pytest.param(FLOOR + 1, True, id="after-floor-ms"),
        pytest.param(FLOOR - 1, False, id="before-floor-ms"),
        pytest.param(FLOOR // 1000 + 1, True, id="after-floor-seconds"),
        pytest.param(FLOOR // 1000 - 1, False, id="before-floor-seconds"),
        pytest.param(None, False, id="missing"),
        pytest.param("2026-10-02T00:00:00Z", False, id="string"),
        pytest.param(True, False, id="bool"),
    ],
)
def test_floor_boundary_and_malformed_timestamps(ts_created, kept):
    """A record whose age cannot be established is not trusted under a floor."""
    row = {"b_is_baseline": False, STEP: 98.5, "@timestamp": "2026-10-02T00:00:00.000Z"}
    if ts_created is not None:
        row["ts_created"] = ts_created
    baselines = pru.calculate_baseline_metrics(
        [row], None, [], [STEP], metric_history_floors_ms={STEP: FLOOR}
    )

    assert (STEP in baselines) is kept


def test_prepare_regressive_test_cases_applies_the_floor():
    old, new = _two_wave_history()
    history = [dict(r, **{MEDIAN: r[STEP]}) for r in old + new]

    def verdict(floors):
        data = {0: {"b_is_post_merge": True, STEP: 98.5, MEDIAN: 98.4, "s_test_case_name": "c"}}
        pru.prepare_regressive_test_cases(
            {0: history[-1]},
            {0: None},
            {0: history},
            data,
            [],
            [STEP, MEDIAN],
            [STEP, MEDIAN],
            metric_history_floors_ms=floors,
        )
        return data[0]["b_is_regression"]

    assert verdict(None) is True
    assert verdict({STEP: FLOOR, MEDIAN: FLOOR}) is False


def _gate_with_stored_baseline(stored, floors):
    """Run prepare_regressive_test_cases with a stored baseline; return each metric's baseline."""
    old, new = _two_wave_history()
    history = [dict(r, **{MEDIAN: r[STEP]}) for r in old + new]
    data = {
        0: {
            "b_is_post_merge": True,
            STEP: 98.5,
            MEDIAN: 98.4,
            THROUGHPUT: 517000.0,
            "s_test_case_name": "c",
        }
    }
    pru.prepare_regressive_test_cases(
        {0: history[-1]},
        {0: dict(stored)},
        {0: history},
        data,
        [THROUGHPUT],
        [STEP, MEDIAN],
        [STEP, MEDIAN, THROUGHPUT],
        metric_history_floors_ms=floors,
    )
    return {
        line.split(":")[0].strip(): float(line.split("baseline=")[1].split()[0])
        for line in data[0]["s_regression_info"].splitlines()
        if "baseline=" in line
    }


_STORED = {
    f"d_baseline_{STEP[2:]}": 31.6,
    f"d_baseline_{MEDIAN[2:]}": 31.5,
    f"d_baseline_{THROUGHPUT[2:]}": 600000.0,
}
_FLOORS = {STEP: FLOOR, MEDIAN: FLOOR}


@pytest.mark.parametrize(
    "stored_created",
    [
        pytest.param(FLOOR - 1, id="recorded-before-floor"),
        pytest.param(None, id="unknown-age"),
    ],
)
def test_stored_baseline_older_than_the_floor_is_recomputed(stored_created):
    """A d_baseline_* that may predate the metric's current definition cannot set its baseline."""
    stored = dict(_STORED)
    if stored_created is not None:
        stored["ts_created"] = stored_created
    baselines = _gate_with_stored_baseline(stored, _FLOORS)

    assert baselines[STEP] == pytest.approx(98.5)
    assert baselines[MEDIAN] == pytest.approx(98.5)
    # The unfloored metric keeps its stored baseline.
    assert baselines[THROUGHPUT] == pytest.approx(600000.0)


@pytest.mark.parametrize(
    "stored_created",
    [pytest.param(FLOOR, id="at-floor"), pytest.param(FLOOR + 1, id="after-floor")],
)
def test_stored_baseline_set_after_the_floor_is_honoured(stored_created):
    """A pin set under the current definition still applies to a floored metric."""
    baselines = _gate_with_stored_baseline(dict(_STORED, ts_created=stored_created), _FLOORS)

    assert baselines[STEP] == pytest.approx(31.6)
    assert baselines[MEDIAN] == pytest.approx(31.5)


def test_stored_baseline_is_honoured_without_a_floor():
    baselines = _gate_with_stored_baseline(dict(_STORED, ts_created=FLOOR - 1), None)

    assert baselines[STEP] == pytest.approx(31.6)
    assert baselines[THROUGHPUT] == pytest.approx(600000.0)


@pytest.mark.parametrize(
    ("stored_created", "carried"),
    [
        pytest.param(FLOOR - 1, False, id="recorded-before-floor"),
        pytest.param(None, False, id="unknown-age"),
        pytest.param(FLOOR, True, id="at-floor"),
    ],
)
def test_post_merge_record_does_not_inherit_a_pre_floor_baseline(stored_created, carried):
    """Step 7 must not copy a baseline the gate refuses onto a new record with a fresh timestamp."""
    stored = {
        f"d_baseline_{STEP[2:]}": 31.6,
        f"d_baseline_{THROUGHPUT[2:]}": 600000.0,
        f"d_threshold_post_merge_{STEP[2:]}": 0.07,
    }
    if stored_created is not None:
        stored["ts_created"] = stored_created
    data = {0: {STEP: 98.5, THROUGHPUT: 517000.0}}
    pru.add_baseline_fields_to_post_merge_data(
        {0: stored}, data, [THROUGHPUT], [STEP], metric_history_floors_ms={STEP: FLOOR}
    )

    assert (f"d_baseline_{STEP[2:]}" in data[0]) is carried
    # Thresholds and unfloored metrics are carried as before.
    assert data[0][f"d_threshold_post_merge_{STEP[2:]}"] == 0.07
    assert data[0][f"d_baseline_{THROUGHPUT[2:]}"] == 600000.0
    assert "ts_created" not in data[0]


def _load_open_search_db_utils(history_rows):
    """Load the real open_search_db_utils with its OpenSearch client stubbed to *history_rows*."""
    defs_pkg = types.ModuleType("defs")
    defs_pkg.__path__ = []
    alternative = types.ModuleType("defs.trt_test_alternative")
    alternative.print_info = lambda *args, **kwargs: None
    alternative.print_warning = lambda *args, **kwargs: None
    perf_pkg = types.ModuleType("defs.perf")
    perf_pkg.__path__ = []
    matching = types.ModuleType("test_common.perf_sanity_matching")
    matching.benchmark_data_matches = lambda history, new, keys: True
    test_common_pkg = types.ModuleType("test_common")
    test_common_pkg.__path__ = []
    jenkins_pkg = types.ModuleType("jenkins")
    jenkins_pkg.__path__ = []
    scripts_pkg = types.ModuleType("jenkins.scripts")
    scripts_pkg.__path__ = []
    osdb = types.ModuleType("jenkins.scripts.open_search_db")
    osdb.PERF_SANITY_PROJECT_NAME = "perf_sanity_info"

    class _OpenSearchDB:
        @staticmethod
        def queryPerfDataFromOpenSearchDB(project, must_clauses, size=0):
            return [dict(row) for row in history_rows]

    osdb.OpenSearchDB = _OpenSearchDB
    stubs = {
        "defs": defs_pkg,
        "defs.trt_test_alternative": alternative,
        "defs.perf": perf_pkg,
        "test_common": test_common_pkg,
        "test_common.perf_sanity_matching": matching,
        "jenkins": jenkins_pkg,
        "jenkins.scripts": scripts_pkg,
        "jenkins.scripts.open_search_db": osdb,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "defs.perf.open_search_db_utils", _PERF_DIR / "open_search_db_utils.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def test_stored_baseline_lookup_carries_its_source_records_age():
    """get_history_data must report when the stored baseline was recorded, or no pin can be aged."""
    rows = [
        {"@timestamp": "2026-10-03T00:00:00.000Z", "ts_created": FLOOR + 10, STEP: 98.4},
        {
            "@timestamp": "2026-10-02T00:00:00.000Z",
            "ts_created": FLOOR + 5,
            STEP: 98.5,
            f"d_baseline_{STEP[2:]}": 97.0,
        },
        {
            "@timestamp": "2026-09-20T00:00:00.000Z",
            "ts_created": FLOOR - 5,
            STEP: 31.6,
            f"d_baseline_{STEP[2:]}": 31.6,
        },
    ]
    osdbu = _load_open_search_db_utils(rows)
    _, latest_baseline_threshold, _ = osdbu.get_history_data({0: {}}, [], {})

    stored = latest_baseline_threshold[0]
    assert stored[f"d_baseline_{STEP[2:]}"] == 97.0
    assert stored["ts_created"] == FLOOR + 5


def test_process_and_upload_test_results_forwards_the_floor(monkeypatch):
    """The real pipeline, with only its OpenSearch seams replaced, honours the floor."""
    old, new = _two_wave_history()
    history = [dict(r, **{MEDIAN: r[STEP]}) for r in old + new]
    monkeypatch.setenv(
        "globalVars",
        '{"build_branch": "main", "action_info": {"parents": '
        '[{"url": "https://ci/job/LLM/job/main/job/L0_PostMerge/3018/", "build_number": "3018"}]}}',
    )
    monkeypatch.setattr(pru, "get_common_values", lambda data_dict, keys: {})
    monkeypatch.setattr(
        pru,
        "get_history_data",
        lambda data_dict, keys, common: ({0: history[-1]}, {0: None}, {0: history}),
    )
    monkeypatch.setattr(pru, "post_new_perf_data", lambda data_dict: None)

    def verdict(floors):
        data = {0: {"s_test_case_name": "c", STEP: 98.5, MEDIAN: 98.4}}
        pru.process_and_upload_test_results(
            data,
            ["s_test_case_name"],
            maximize_metrics=[],
            minimize_metrics=[STEP, MEDIAN],
            regression_metrics=[STEP, MEDIAN],
            metric_history_floors_ms=floors,
        )
        return data[0]["b_is_regression"]

    assert verdict(None) is True
    assert verdict({STEP: FLOOR, MEDIAN: FLOOR}) is False


def test_process_and_upload_test_results_drops_a_pre_floor_pin_from_the_upload(monkeypatch):
    """A pre-floor d_baseline_* neither gates the run nor rides onto the uploaded record."""
    old, new = _two_wave_history()
    history = [dict(r, **{MEDIAN: r[STEP]}) for r in old + new]
    pin = {
        f"d_baseline_{STEP[2:]}": 31.6,
        f"d_baseline_{MEDIAN[2:]}": 31.5,
        "ts_created": FLOOR - 1,
    }
    uploaded = {}
    monkeypatch.setenv(
        "globalVars",
        '{"build_branch": "main", "action_info": {"parents": '
        '[{"url": "https://ci/job/LLM/job/main/job/L0_PostMerge/3018/", "build_number": "3018"}]}}',
    )
    monkeypatch.setattr(pru, "get_common_values", lambda data_dict, keys: {})
    monkeypatch.setattr(
        pru,
        "get_history_data",
        lambda data_dict, keys, common: ({0: history[-1]}, {0: dict(pin)}, {0: history}),
    )
    monkeypatch.setattr(pru, "post_new_perf_data", lambda data_dict: uploaded.update(data_dict))

    data = {0: {"s_test_case_name": "c", STEP: 98.5, MEDIAN: 98.4}}
    pru.process_and_upload_test_results(
        data,
        ["s_test_case_name"],
        maximize_metrics=[],
        minimize_metrics=[STEP, MEDIAN],
        regression_metrics=[STEP, MEDIAN],
        metric_history_floors_ms={STEP: FLOOR, MEDIAN: FLOOR},
    )

    assert uploaded[0]["b_is_regression"] is False
    assert f"d_baseline_{STEP[2:]}" not in uploaded[0]
    assert f"d_baseline_{MEDIAN[2:]}" not in uploaded[0]


def test_harness_floors_every_gen_worker_statistic_and_nothing_else():
    constants = _harness_constants()
    floors = constants["GEN_ONLY_METRIC_HISTORY_FLOORS_MS"]
    stats = {f"d_{name}" for name in constants["DEVICE_STEP_TIME_METRICS"]}

    assert set(floors) == stats
    assert set(constants["GEN_ONLY_REGRESSION_METRICS"]) <= set(floors)
    assert set(floors.values()) == {constants["GEN_ONLY_DEVICE_STEP_TIME_HISTORY_FLOOR_MS"]}


def _upload_method():
    tree = ast.parse((_PERF_DIR / "test_perf_sanity.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "upload_test_results_to_database":
            return node
    raise AssertionError("test_perf_sanity.py no longer defines upload_test_results_to_database")


def _assigned_value(stmts, name):
    """The value assigned to *name* directly inside *stmts* (one branch of an if)."""
    values = [
        stmt.value
        for stmt in stmts
        if isinstance(stmt, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets)
    ]
    assert len(values) == 1, f"expected one assignment to {name}, found {len(values)}"
    return values[0]


def test_harness_upload_floors_gen_only_and_not_e2e():
    """The gen_only branch selects the floors, the e2e branch none, and the call forwards them.

    The floors are inert unless the call site hands them over, and the integration path that
    makes the call needs GPUs, so the wiring is checked on the source instead.
    """
    method = _upload_method()
    branches = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and any(
            isinstance(stmt, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "regression_metrics" for t in stmt.targets)
            for stmt in node.body
        )
    ]
    assert len(branches) == 1, "expected one if/else choosing regression_metrics"
    gen_only, other = branches[0].body, branches[0].orelse

    assert ast.unparse(_assigned_value(gen_only, "regression_metrics")) == (
        "list(GEN_ONLY_REGRESSION_METRICS)"
    )
    assert ast.unparse(_assigned_value(gen_only, "metric_history_floors_ms")) == (
        "GEN_ONLY_METRIC_HISTORY_FLOORS_MS"
    )
    assert ast.unparse(_assigned_value(other, "metric_history_floors_ms")) == "None"

    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "process_and_upload_test_results"
    ]
    assert len(calls) == 1
    forwarded = {kw.arg: kw.value for kw in calls[0].keywords}
    assert ast.unparse(forwarded.get("metric_history_floors_ms")) == "metric_history_floors_ms"


def test_harness_floor_is_the_expected_bucket_cutover():
    """The floor must sit between the last modal-bucket row and the first expected-bucket row.

    CI build 2995 (6bfc3ad499, modal rule) uploaded at 2026-10-01T05:35Z and build 2996
    (ee510fc85d, the first commit with the expected-bucket rule) at 2026-10-01T16:06Z.
    """
    floor = _harness_constants()["GEN_ONLY_DEVICE_STEP_TIME_HISTORY_FLOOR_MS"]

    assert _ms(2026, 10, 1, 5) < floor < _ms(2026, 10, 1, 16)
