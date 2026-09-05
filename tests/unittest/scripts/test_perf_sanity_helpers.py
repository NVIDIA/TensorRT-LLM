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

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch._inductor")

__extra_import_path__ = ["~/tests/integration"]
from defs.perf import test_perf_sanity as perf_sanity  # noqa: E402
from defs.perf.perf_regression_utils import (  # noqa: E402
    calculate_baseline_metrics,
    prepare_regressive_test_cases,
)


def test_run_benchmark_with_log_returns_successful_output(tmp_path: Path) -> None:
    benchmark_log = tmp_path / "trtllm-benchmark.0.0.log"
    command = [sys.executable, "-c", "print('benchmark succeeded')"]

    output = perf_sanity._run_benchmark_with_log(command, {}, str(benchmark_log))

    assert output == "benchmark succeeded\n"
    assert benchmark_log.read_text(encoding="utf-8") == output


def test_run_benchmark_with_log_preserves_failed_output(tmp_path: Path) -> None:
    benchmark_log = tmp_path / "trtllm-benchmark.0.0.log"
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "print('benchmark stdout'); "
            "print('benchmark stderr', file=sys.stderr); "
            "sys.exit(7)"
        ),
    ]

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        perf_sanity._run_benchmark_with_log(command, {}, str(benchmark_log))

    expected_output = "benchmark stdout\nbenchmark stderr\n"
    assert exc_info.value.returncode == 7
    assert exc_info.value.output.decode() == expected_output
    assert benchmark_log.read_text(encoding="utf-8") == expected_output


def test_benchmark_log_is_included_in_report_logs(tmp_path: Path) -> None:
    benchmark_log = tmp_path / "trtllm-benchmark.0.0.log"
    benchmark_log.touch()
    aggr_commands = perf_sanity.AggrTestCmds(
        server_cmds=[[]],
        client_cmds={0: [[]]},
        timeout=1,
        output_dir=str(tmp_path),
        test_output_dir=str(tmp_path),
    )
    disagg_commands = perf_sanity.DisaggTestCmds(
        server_cmds=[],
        client_cmds={},
        timeout=1,
        hostname="localhost",
        disagg_serving_type="BENCHMARK",
        num_ctx_servers=0,
        num_gen_servers=0,
        output_dir=str(tmp_path),
        test_output_dir=str(tmp_path),
    )

    assert str(benchmark_log) in aggr_commands.get_server_logs(0)
    assert str(benchmark_log) in disagg_commands.get_server_logs(0)


def test_sentinel_timeout_falls_back_to_current_gen_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark_log = tmp_path / "trtllm-benchmark.0.0.log"
    benchmark_log.write_text("benchmark output", encoding="utf-8")
    outputs = ["benchmark output"]
    pending = [
        {
            "output_index": 0,
            "benchmark_file_path": str(benchmark_log),
            "start_offsets": [10, 20],
        }
    ]
    commands = perf_sanity.DisaggTestCmds(
        server_cmds=[],
        client_cmds={},
        timeout=1,
        hostname="localhost",
        disagg_serving_type="BENCHMARK",
        num_ctx_servers=1,
        num_gen_servers=2,
        output_dir=str(tmp_path),
        test_output_dir=str(tmp_path),
    )

    monkeypatch.setattr(
        perf_sanity.DisaggTestCmds,
        "wait_for_gen_log_sentinels",
        lambda self: False,
    )
    parse_calls: list[tuple[str, int, list[int]]] = []

    def parse_device_step_time(
        output_dir: str,
        num_gen_servers: int,
        start_offsets: list[int],
    ) -> perf_sanity._DeviceStepTimeStats:
        parse_calls.append((output_dir, num_gen_servers, start_offsets))
        return perf_sanity._DeviceStepTimeStats(mean=7.25, median=7.2, std=0.115, p75=7.3, p99=7.42)

    monkeypatch.setattr(
        perf_sanity,
        "parse_gen_worker_device_step_time",
        parse_device_step_time,
    )

    commands._append_gen_worker_device_step_time(pending, outputs)

    assert parse_calls == [(str(tmp_path), 2, [10, 20])]
    expected = (
        "Average Per Iter Device Step Time (ms): 7.25\n"
        "Median Per Iter Device Step Time (ms): 7.2000\n"
        "Stdev Per Iter Device Step Time (ms): 0.1150\n"
        "P75 Per Iter Device Step Time (ms): 7.3000\n"
        "P99 Per Iter Device Step Time (ms): 7.4200\n"
    )
    assert outputs == [f"benchmark output\n{expected}"]
    assert benchmark_log.read_text(encoding="utf-8").endswith(f"\n{expected}")


def test_missing_device_step_time_appends_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A None parse must leave the log untouched.

    check_test_failure raises on the absent mean later; writing a partial or
    placeholder line here would upload a fabricated number instead.
    """
    benchmark_log = tmp_path / "trtllm-benchmark.0.0.log"
    benchmark_log.write_text("benchmark output", encoding="utf-8")
    outputs = ["benchmark output"]
    pending = [
        {
            "output_index": 0,
            "benchmark_file_path": str(benchmark_log),
            "start_offsets": [0, 0],
        }
    ]
    commands = perf_sanity.DisaggTestCmds(
        server_cmds=[],
        client_cmds={},
        timeout=1,
        hostname="localhost",
        disagg_serving_type="BENCHMARK",
        num_ctx_servers=1,
        num_gen_servers=2,
        output_dir=str(tmp_path),
        test_output_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        perf_sanity.DisaggTestCmds,
        "wait_for_gen_log_sentinels",
        lambda self: True,
    )
    monkeypatch.setattr(
        perf_sanity,
        "parse_gen_worker_device_step_time",
        lambda *args, **kwargs: None,
    )

    commands._append_gen_worker_device_step_time(pending, outputs)

    assert outputs == ["benchmark output"]
    assert benchmark_log.read_text(encoding="utf-8") == "benchmark output"


# ---------------------------------------------------------------------------
# Gen-worker per-iteration device step time
# ---------------------------------------------------------------------------


def _iter_line(
    iter_no: int,
    nsr: int,
    device_step_time: str,
    ngen: int = 256,
    host_step_time: float = 7.4,
    rank: int = 0,
) -> str:
    """One gen-worker iteration log line, in py_executor.py's real format.

    Keep the ' = ' spelling and the field order: the parsers depend on both.
    rank defaults to 0 because that is the only rank py_executor.py logs unless
    TLLM_PROFILE_LOG_RANKS is set; pass it to build a mixed-rank file.
    """
    return (
        f"[TRT-LLM] [I] [_torch][RANK {rank}] iter = {iter_no}, "
        f"global_rank = {rank}, "
        f"rank = {rank}, num_scheduled_requests = {nsr}, kv_cache_util = 0.1, "
        f"currank_total_requests = 0/1, host_step_time = {host_step_time}ms, "
        f"prev_device_step_time = {device_step_time}, "
        "timestamp = 08-23-2026 01:02:03, "
        "states = {'num_ctx_requests': 0, 'num_ctx_tokens': 0, "
        f"'num_generation_tokens': {ngen}}}"
    )


def _write_gen_log(tmp_path: Path, index: int, lines: list[str]) -> None:
    path = tmp_path / f"gen_server_{index}.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scan(tmp_path: Path, num_gen_servers: int = 1) -> list[list]:
    return perf_sanity._scan_gen_worker_device_step_time(str(tmp_path), num_gen_servers, None)


def test_empty_iteration_successor_is_excluded(tmp_path: Path) -> None:
    """Nvbugs 6627789, verbatim from the culprit run's gen_server_0.log.

    iter 259 scheduled zero requests and spent 1456.7 ms idle; the device
    reports that idle period as iter 260's prev_device_step_time. Averaging it
    in turned an unchanged ~7.3 ms workload into a +19% "regression".
    """
    _write_gen_log(
        tmp_path,
        0,
        [
            _iter_line(258, 1, "7.326848030090332ms", host_step_time=7.426261901855469),
            _iter_line(259, 0, "7.391776084899902ms", host_step_time=1456.6991329193115),
            _iter_line(260, 1, "1450.6143798828125ms", host_step_time=2.572298049926758),
            _iter_line(261, 1, "7.31ms"),
        ],
    )

    rows = _scan(tmp_path)

    assert [row.device_step_time for row in rows[0]] == [
        7.326848030090332,
        7.391776084899902,
        7.31,
    ], "the 1450.61 ms row must be dropped, and only that row"


def test_the_empty_iteration_row_itself_is_kept(tmp_path: Path) -> None:
    """Iter 259's own value describes iter 258, which did real work."""
    _write_gen_log(
        tmp_path,
        0,
        [
            _iter_line(258, 1, "7.32ms"),
            _iter_line(259, 0, "7.39ms"),
            _iter_line(260, 1, "1450.61ms"),
        ],
    )

    rows = _scan(tmp_path)

    assert 7.39 in [row.device_step_time for row in rows[0]]


def test_non_adjacent_predecessor_does_not_trigger_the_exclusion(
    tmp_path: Path,
) -> None:
    """The guard fails toward keeping the row.

    Under rank interleaving or a counter reset the previous *line* is not the
    previous *iteration*, so its num_scheduled_requests says nothing about
    what this row measured. Dropping real data is the worse error.
    """
    _write_gen_log(
        tmp_path,
        0,
        [
            _iter_line(100, 0, "7.30ms"),
            _iter_line(200, 1, "999.0ms"),
        ],
    )

    rows = _scan(tmp_path)

    assert 999.0 in [row.device_step_time for row in rows[0]]


def test_unparsable_predecessor_does_not_trigger_the_exclusion(
    tmp_path: Path,
) -> None:
    _write_gen_log(
        tmp_path,
        0,
        [
            "[TRT-LLM] [I] iter = ?, num_scheduled_requests = ?, prev_device_step_time = 1.0ms",
            _iter_line(201, 1, "999.0ms"),
        ],
    )

    rows = _scan(tmp_path)

    assert [row.device_step_time for row in rows[0]] == [999.0]


def test_interleaved_ranks_do_not_defeat_the_exclusion(tmp_path: Path) -> None:
    """Predecessor state is per rank, so a foreign line cannot mask the idle one.

    py_executor.py logs rank 0 only unless TLLM_PROFILE_LOG_RANKS says otherwise,
    but lane YAML can inject that variable. With one shared predecessor slot,
    rank 1's line between rank 0's iters 259 and 260 would supply a nonzero
    num_scheduled_requests and the 1450 ms row would survive -- the exclusion
    silently off while still looking armed.
    """
    _write_gen_log(
        tmp_path,
        0,
        [
            _iter_line(258, 1, "7.32ms", rank=0),
            _iter_line(259, 0, "7.39ms", rank=0),
            _iter_line(259, 4, "7.40ms", rank=1),
            _iter_line(260, 1, "1450.61ms", rank=0),
        ],
    )

    rows = _scan(tmp_path)

    assert 1450.61 not in [row.device_step_time for row in rows[0]]


def test_another_ranks_idle_iteration_does_not_drop_a_valid_row(
    tmp_path: Path,
) -> None:
    """The mirror image: rank 1 going idle must not cost rank 0 a good sample."""
    _write_gen_log(
        tmp_path,
        0,
        [
            _iter_line(259, 0, "7.39ms", rank=1),
            _iter_line(260, 1, "7.31ms", rank=0),
        ],
    )

    rows = _scan(tmp_path)

    assert [row.device_step_time for row in rows[0]] == [7.39, 7.31]


def test_warmup_iterations_are_excluded(tmp_path: Path) -> None:
    """Iter 0/1 include KV-cache transfer wait; 2-4 have not reached steady state."""
    _write_gen_log(
        tmp_path,
        0,
        [_iter_line(n, 1, f"{n}.0ms") for n in range(1, 8)],
    )

    rows = _scan(tmp_path)

    assert [row.device_step_time for row in rows[0]] == [5.0, 6.0, 7.0]


def test_na_device_step_time_is_skipped(tmp_path: Path) -> None:
    """Iter 1's 'N/A' does not match the value regex, and must not crash."""
    _write_gen_log(
        tmp_path,
        0,
        [_iter_line(5, 1, "N/A"), _iter_line(6, 1, "7.3ms")],
    )

    rows = _scan(tmp_path)

    assert [row.device_step_time for row in rows[0]] == [7.3]


def test_unparsable_num_generation_tokens_is_retained_as_none(
    tmp_path: Path,
) -> None:
    """Nvbugs 6487036 / 6487040: the field rendered as tensor(256).

    Such a row keeps its device step time so the all-rows fallback in
    _stats_at_mode_ngen has something to work with.
    """
    line = _iter_line(5, 1, "7.3ms").replace("'num_generation_tokens': 256", "")
    _write_gen_log(tmp_path, 0, [line])

    rows = _scan(tmp_path)

    assert rows[0][0].ngen is None
    assert rows[0][0].device_step_time == 7.3


def test_retention_is_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The percentiles need the sample, so a pathological log must be bounded."""
    monkeypatch.setattr(perf_sanity, "_MAX_RETAINED_ITER_ROWS", 3)
    _write_gen_log(
        tmp_path,
        0,
        [_iter_line(n, 1, f"{n}.0ms") for n in range(5, 25)],
    )

    rows = _scan(tmp_path)

    assert [row.device_step_time for row in rows[0]] == [5.0, 6.0, 7.0]


def test_missing_gen_log_is_skipped_not_fatal(tmp_path: Path) -> None:
    _write_gen_log(tmp_path, 1, [_iter_line(5, 1, "7.3ms")])

    rows = _scan(tmp_path, num_gen_servers=3)

    assert len(rows) == 1


def test_start_offsets_slice_each_clients_own_segment(tmp_path: Path) -> None:
    """Per-client byte offsets, not one global average over the whole run."""
    first = _iter_line(5, 1, "1.0ms")
    second = _iter_line(6, 1, "2.0ms")
    path = tmp_path / "gen_server_0.log"
    path.write_text(f"{first}\n{second}\n", encoding="utf-8")

    rows = perf_sanity._scan_gen_worker_device_step_time(str(tmp_path), 1, [len(first) + 1])

    assert [row.device_step_time for row in rows[0]] == [2.0]


def test_stdev_of_fewer_than_two_samples_is_zero() -> None:
    assert perf_sanity._stdev([]) == 0.0
    assert perf_sanity._stdev([7.3]) == 0.0


def test_stdev_uses_ddof_one() -> None:
    assert perf_sanity._stdev([1.0, 3.0]) == pytest.approx(1.4142135623730951)


def _rows(*specs: tuple[int | None, float]) -> list:
    return [perf_sanity._IterRow(ngen=ngen, device_step_time=v) for ngen, v in specs]


def test_all_five_statistics_on_a_known_sample() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    stats = perf_sanity._stats_at_mode_ngen([_rows(*((256, v) for v in values))])

    assert stats.mean == pytest.approx(22.0)
    assert stats.median == pytest.approx(3.0)
    assert stats.std == pytest.approx(43.617656975128774)
    assert stats.p75 == pytest.approx(4.0)
    assert stats.p99 == pytest.approx(96.16)


def test_only_the_mode_num_generation_tokens_bucket_is_used() -> None:
    """Concurrency ramps down at the tail; those iterations do less work."""
    stats = perf_sanity._stats_at_mode_ngen(
        [_rows((256, 7.0), (256, 7.0), (256, 7.0), (8, 1.0), (4, 1.0))]
    )

    assert stats.mean == pytest.approx(7.0)


def test_mode_bucket_tie_prefers_the_larger_token_count() -> None:
    stats = perf_sanity._stats_at_mode_ngen([_rows((256, 7.0), (8, 1.0))])

    assert stats.mean == pytest.approx(7.0)


def test_no_parseable_token_count_falls_back_to_every_row() -> None:
    stats = perf_sanity._stats_at_mode_ngen([_rows((None, 6.0), (None, 8.0))])

    assert stats.mean == pytest.approx(7.0)


def test_statistics_are_averaged_unweighted_across_workers() -> None:
    """The same per-file rule the mean has always used, for all five."""
    stats = perf_sanity._stats_at_mode_ngen(
        [
            _rows((256, 6.0)),
            _rows((256, 8.0), (256, 8.0), (256, 8.0)),
        ]
    )

    assert stats.mean == pytest.approx(7.0)
    assert stats.median == pytest.approx(7.0)


def test_no_usable_rows_reports_none() -> None:
    assert perf_sanity._stats_at_mode_ngen([]) is None


def test_parse_gen_worker_device_step_time_end_to_end(tmp_path: Path) -> None:
    _write_gen_log(
        tmp_path,
        0,
        [
            _iter_line(5, 1, "7.0ms"),
            _iter_line(6, 0, "7.0ms"),
            _iter_line(7, 1, "500.0ms"),
            _iter_line(8, 1, "9.0ms"),
        ],
    )

    stats = perf_sanity.parse_gen_worker_device_step_time(str(tmp_path), 1)

    assert stats.mean == pytest.approx(23.0 / 3)
    assert stats.median == pytest.approx(7.0)


def test_parse_gen_worker_device_step_time_reports_none_with_no_logs(
    tmp_path: Path,
) -> None:
    assert perf_sanity.parse_gen_worker_device_step_time(str(tmp_path), 2) is None


def test_every_written_line_parses_and_none_shadows_another(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Round trip: the five appended lines through the upload-side regexes.

    parse_metrics_from_output is a closure inside PerfSanityTestConfig, so its
    first-match-per-line loop is reproduced here. That loop is the reason the
    five lines must carry mutually exclusive leading words -- a shared prefix
    would silently shadow whichever pattern lost the ordering race.
    """
    benchmark_log = tmp_path / "trtllm-benchmark.0.0.log"
    benchmark_log.write_text("benchmark output", encoding="utf-8")
    outputs = [""]
    commands = perf_sanity.DisaggTestCmds(
        server_cmds=[],
        client_cmds={},
        timeout=1,
        hostname="localhost",
        disagg_serving_type="BENCHMARK",
        num_ctx_servers=1,
        num_gen_servers=1,
        output_dir=str(tmp_path),
        test_output_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        perf_sanity.DisaggTestCmds,
        "wait_for_gen_log_sentinels",
        lambda self: True,
    )
    _write_gen_log(
        tmp_path,
        0,
        [_iter_line(n, 1, f"{7.0 + n / 100:.4f}ms") for n in range(5, 30)],
    )

    commands._append_gen_worker_device_step_time(
        [
            {
                "output_index": 0,
                "benchmark_file_path": str(benchmark_log),
                "start_offsets": None,
            }
        ],
        outputs,
    )

    metrics: dict[str, float] = {}
    for line in outputs[0].split("\n"):
        for name, regex in perf_sanity.GEN_ONLY_PERF_METRIC_LOG_QUERIES.items():
            if name in metrics:
                continue
            match = regex.search(line)
            if match:
                metrics[name] = float(match.group(1))
                break

    assert set(metrics) == set(perf_sanity.GEN_ONLY_DEVICE_STEP_TIME_METRICS)
    assert metrics["mean_gen_worker_per_iter_device_step_time"] == pytest.approx(7.17, abs=0.01)
    assert metrics["std_gen_worker_per_iter_device_step_time"] > 0.0


def test_every_device_step_time_metric_is_a_minimize_metric() -> None:
    """A metric absent from both lists raises ValueError in check_regression."""
    for name in perf_sanity.GEN_ONLY_DEVICE_STEP_TIME_METRICS:
        assert f"d_{name}" in perf_sanity.MINIMIZE_METRICS


def test_add_perf_metric_value_skips_absent_statistics() -> None:
    """TypeCheckForOpenSearchDB rejects both None and int for a d_ key."""
    metrics = dict.fromkeys(perf_sanity.PERF_METRIC_LOG_QUERIES, 1.0)
    metrics["mean_gen_worker_per_iter_device_step_time"] = 7

    new_data: dict = {}
    perf_sanity.add_perf_metric_value(new_data, metrics, False, "gen_only")

    assert new_data["d_mean_gen_worker_per_iter_device_step_time"] == 7.0
    assert isinstance(new_data["d_mean_gen_worker_per_iter_device_step_time"], float)
    assert "d_p99_gen_worker_per_iter_device_step_time" not in new_data


def test_add_perf_metric_value_omits_the_family_outside_gen_only() -> None:
    """e2e and ctx_only never emit these lines, so they must not be uploaded."""
    metrics = dict.fromkeys(perf_sanity.PERF_METRIC_LOG_QUERIES, 1.0)
    metrics["mean_gen_worker_per_iter_device_step_time"] = 7.0

    new_data: dict = {}
    perf_sanity.add_perf_metric_value(new_data, metrics, False, "e2e")

    assert not [key for key in new_data if "gen_worker_per_iter" in key]


def test_every_gated_metric_is_checkable() -> None:
    """A gated metric absent from both lists would look armed and never be checked.

    check_regression only iterates maximize + minimize, so pin the pair: adding a
    third gated statistic cannot silently regress into a dead gate.
    """
    checkable = set(perf_sanity.MINIMIZE_METRICS) | set(perf_sanity.MAXIMIZE_METRICS)
    for name in perf_sanity.GEN_ONLY_REGRESSION_METRICS:
        assert name in checkable, f"{name} is gated but check_regression never sees it"


def test_every_gated_metric_is_actually_emitted() -> None:
    """A gated metric the log never carries is skipped by 'not in new_data'."""
    emitted = {f"d_{name}" for name in perf_sanity.GEN_ONLY_DEVICE_STEP_TIME_METRICS}
    assert set(perf_sanity.GEN_ONLY_REGRESSION_METRICS) <= emitted


def _regression_row(gate: list, baseline: dict, measured: dict) -> dict:
    """Row the real regression pipeline produces for *measured* against *baseline*.

    Driven through prepare_regressive_test_cases rather than asserting list
    membership: membership is the input, and every property below is about what
    the pipeline then decides with it. A single history day pins the derived
    fallback baseline on *baseline* whichever percentile the metric's direction
    selects, since every percentile of a one-element series is that element. The
    first argument is only tested for None (the network-failure bail-out), so its
    contents do not matter.
    """
    new_data_dict = {0: {"s_test_case_name": "unit", **measured}}
    prepare_regressive_test_cases(
        {},
        None,
        {0: [dict(baseline, ts_created="2026-09-01")]},
        new_data_dict,
        perf_sanity.MAXIMIZE_METRICS,
        perf_sanity.MINIMIZE_METRICS,
        gate,
    )
    return new_data_dict[0]


def test_tpot_is_classified_as_a_latency() -> None:
    """TPOT is "Time per Output Token (ms)", not a rate.

    benchmark_serving computes it as (e2e_latency - ttft) / (output_len - 1), so
    it is a latency like its ttft/itl/e2el siblings. d_al is pinned alongside
    because it is the one non-rate on the maximize side -- mean accepted draft
    tokens per iteration is a count, and more of them is better -- so a later
    substring-driven sweep of the tpot trio must not carry it along.
    """
    for statistic in ("d_mean_tpot", "d_median_tpot", "d_p99_tpot"):
        assert statistic in perf_sanity.MINIMIZE_METRICS
        assert statistic not in perf_sanity.MAXIMIZE_METRICS
    assert "d_al" in perf_sanity.MAXIMIZE_METRICS


def test_a_slower_tpot_can_actually_fail_a_build() -> None:
    """A gate on a maximize metric tests new < baseline*(1-threshold).

    A rising latency never satisfies that, so while TPOT sat on the maximize side
    a lane could name a TPOT statistic and still be gated on nothing. Graded on
    the verdict, and in both directions, so a gate that simply always fires would
    not pass either (https://nvbugs/6706765).
    """
    assert _regression_row(["d_p99_tpot"], {"d_p99_tpot": 6.08}, {"d_p99_tpot": 15.24})[
        "b_is_regression"
    ]
    assert not _regression_row(["d_p99_tpot"], {"d_p99_tpot": 6.08}, {"d_p99_tpot": 5.5})[
        "b_is_regression"
    ]


def test_a_slower_tpot_is_reported_as_a_regression() -> None:
    """The sign a human reads off the CI report, which no pass/fail check covers.

    prepare_regressive_test_cases emits a diff line for every metric it has a
    baseline for, before it consults the gate at all, so an empty gate isolates
    the reported sign from any verdict. The measured 2.5x p99 TPOT slowdown of
    nvbugs/6706765 printed diff=+150.66% -- a 2.5x degradation indistinguishable
    from a 150% improvement.
    """
    row = _regression_row([], {"d_p99_tpot": 6.08}, {"d_p99_tpot": 15.24})

    assert "d_p99_tpot: value=15.2400 baseline=6.0800" in row["s_regression_info"]
    assert "diff=-150.66%" in row["s_regression_info"]


def test_a_tpot_baseline_tracks_the_steady_state_not_the_worst_day() -> None:
    """calculate_baseline_metrics takes P95 for maximize and P5 for minimize.

    So the direction also decides which end of the smoothed history becomes the
    bar. On the maximize side a single stall day used to raise the bar that the
    next run is compared against; assert the baseline sits at the steady state,
    with the mutated-direction value alongside to show the difference is material
    rather than rounding.
    """
    history = [
        {"ts_created": f"2026-09-0{day}", "d_median_tpot": value}
        for day, value in enumerate([5.0, 5.1, 4.9, 12.0, 5.0], start=1)
    ]

    baseline = calculate_baseline_metrics(
        history, None, perf_sanity.MAXIMIZE_METRICS, perf_sanity.MINIMIZE_METRICS
    )["d_median_tpot"]
    as_maximize = calculate_baseline_metrics(history, None, ["d_median_tpot"], [])["d_median_tpot"]

    assert baseline == pytest.approx(5.0, abs=0.2)
    assert as_maximize > baseline * 1.4


# One rank identity as an MPI launcher actually exports it, measured under
# `mpirun -np 1` inside the CI container image. Spelled out rather than read from
# os.environ so the test says the same thing on a developer laptop.
_LAUNCHER_IDENTITY = {
    "OMPI_COMM_WORLD_RANK": "0",
    "OMPI_COMM_WORLD_SIZE": "1",
    "OMPI_COMM_WORLD_LOCAL_RANK": "0",
    "OMPI_MCA_ess": "^singleton",
    "OMPI_MCA_ess_base_jobid": "3826286593",
    "OMPI_MCA_pmix": "^s1,s2,cray,isolated",
    "OMPI_MCA_orte_hnp_uri": "3826286592.0;tcp://10.0.0.1:39633",
    "OMPI_MCA_orte_local_daemon_uri": "3826286592.0;tcp://10.0.0.1:39633",
    "OMPI_MCA_orte_precondition_transports": "0000000000000000-0000000000000000",
    "OMPI_UNIVERSE_SIZE": "1",
    "PMIX_RANK": "0",
    "PMIX_NAMESPACE": "3826286593",
    "PMIX_SERVER_URI21": "3826286592.0;tcp4://127.0.0.1:39633",
    "SLURM_PROCID": "0",
    "SLURM_LOCALID": "0",
    "SLURM_STEPID": "33",
    "SLURM_SRUN_COMM_HOST": "10.0.0.1",
}

# What the same environment says about the ALLOCATION rather than about this
# process's place in it. The child runs its own MPI_Comm_spawn, and OMPI's Slurm
# resource allocator reads SLURM_NODELIST / SLURM_TASKS_PER_NODE by name to size
# it, so stripping these converts the bug into MPI_ERR_SPAWN instead of fixing it.
_ALLOCATION_FACTS = {
    "SLURM_NODELIST": "node-gpu-2",
    "SLURM_TASKS_PER_NODE": "1",
    "SLURM_NTASKS": "1",
    "SLURM_NNODES": "1",
    "SLURM_JOB_ID": "1871020",
    "SLURM_JOB_NODELIST": "node-gpu-2",
    "SLURM_STEP_GPUS": "0",
    "SLURM_GPUS_ON_NODE": "1",
    "OMPI_MCA_rmaps_base_oversubscribe": "1",
    "OMPI_MCA_hwloc_base_binding_policy": "none",
    "OMPI_MCA_plm_slurm_args": "--external-launcher",
}


def test_trtllm_child_env_drops_every_launcher_identity_variable() -> None:
    """A child that imports tensorrt_llm must not inherit its launcher's identity.

    tensorrt_llm/_utils.py does `from mpi4py import MPI` at module scope, so
    MPI_Init_thread runs on import. With any of these present the child tries to
    (re-)join the launcher's job and aborts on a NULL communicator -- the server
    before launch_server can publish --report_addr (surfacing only as an exit
    code), the benchmark client with an empty results log
    (https://nvbugs/6706765).
    """
    cleaned = perf_sanity.trtllm_child_env({**_LAUNCHER_IDENTITY, **_ALLOCATION_FACTS})

    assert not [k for k in _LAUNCHER_IDENTITY if k in cleaned]


def test_trtllm_child_env_keeps_the_allocation_the_child_spawns_into() -> None:
    """Stripping the allocation too would just move the failure, not remove it.

    The child's own MPI_Comm_spawn sizes itself through OMPI's Slurm resource
    allocator, which requires SLURM_NODELIST and SLURM_TASKS_PER_NODE by name and
    FORCE-TERMINATEs without them. So this is the half of the environment that has
    to survive, and it is asserted separately from the half that must not.
    """
    cleaned = perf_sanity.trtllm_child_env({**_LAUNCHER_IDENTITY, **_ALLOCATION_FACTS})

    assert {k: cleaned.get(k) for k in _ALLOCATION_FACTS} == _ALLOCATION_FACTS
    # Non-vacuity: the two halves are disjoint, so neither assertion above is
    # trivially satisfied by the other's keys.
    assert not set(_LAUNCHER_IDENTITY) & set(_ALLOCATION_FACTS)


def test_trtllm_child_env_preserves_unrelated_variables() -> None:
    """Everything the lane actually needs must ride through untouched.

    The child is the real server: it needs the model cache, the venv, and any
    per-config env the lane set. A prefix list that swept these would break the
    run in a way no MPI assertion above would notice.
    """
    payload = {
        "LLM_MODELS_ROOT": "/code/llm-models",
        "PATH": "/usr/bin",
        "VIRTUAL_ENV": "/code/venv",
        "TRTLLM_MOE_BACKEND": "TRTLLM",
        "CUDA_VISIBLE_DEVICES": "0",
    }

    assert perf_sanity.trtllm_child_env(dict(payload)) == payload
