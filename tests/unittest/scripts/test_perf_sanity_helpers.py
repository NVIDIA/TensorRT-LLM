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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests" / "integration"))

from defs.perf import test_perf_sanity as perf_sanity  # noqa: E402


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
            "end_offsets": [110, 120],
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
    parse_calls: list[tuple[str, int, list[int], list[int]]] = []

    def parse_device_step_time(
        output_dir: str,
        num_gen_servers: int,
        start_offsets: list[int],
        end_offsets: list[int],
    ) -> perf_sanity._DeviceStepTimeStats:
        parse_calls.append((output_dir, num_gen_servers, start_offsets, end_offsets))
        return perf_sanity._DeviceStepTimeStats(mean=7.25, median=7.2, std=0.115, p75=7.3, p99=7.42)

    monkeypatch.setattr(
        perf_sanity,
        "parse_gen_worker_device_step_time",
        parse_device_step_time,
    )

    commands._append_gen_worker_device_step_time(pending, outputs)

    # Both window bounds must reach the parser: a record whose end_offsets were
    # dropped on the way through would read to EOF and, in a multi-client mode,
    # attribute every later client's iterations to this one.
    assert parse_calls == [(str(tmp_path), 2, [10, 20], [110, 120])]
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


def _two_segment_gen_log(tmp_path: Path) -> int:
    """One gen log holding two clients' iterations. Returns the split offset.

    Both segments use the same iter numbers and the same ngen, exactly as two
    clients against one long-lived gen worker would: the only thing telling them
    apart is the byte range, which is the whole point of the window.
    """
    first = [_iter_line(n, 1, "7.0ms") for n in range(5, 15)]
    second = [_iter_line(n, 1, "70.0ms") for n in range(5, 15)]
    path = tmp_path / "gen_server_0.log"
    first_text = "\n".join(first) + "\n"
    path.write_text(first_text + "\n".join(second) + "\n", encoding="utf-8")
    return len(first_text.encode())


def test_invalid_utf8_in_the_log_does_not_abort_the_scan(tmp_path: Path) -> None:
    """Tqdm progress bars write partial multibyte sequences during model load.

    The scan reads bytes and decodes per line (so end_offsets can be accounted
    exactly), which moves where errors="replace" applies. Without it a single
    malformed byte anywhere in a multi-hundred-MB worker log would raise
    UnicodeDecodeError and lose the whole metric.
    """
    path = tmp_path / "gen_server_0.log"
    lines = [_iter_line(n, 1, "7.0ms").encode() for n in range(5, 15)]
    # A truncated 3-byte UTF-8 sequence, mid-file, on its own line.
    path.write_bytes(b"\n".join(lines[:5] + [b"loading \xe2\x96"] + lines[5:]) + b"\n")

    stats = perf_sanity.parse_gen_worker_device_step_time(str(tmp_path), 1)

    assert stats.mean == pytest.approx(7.0)


def test_crlf_line_endings_still_parse(tmp_path: Path) -> None:
    r"""Binary reads keep the \r that text mode stripped.

    None of the iteration-line regexes are end-anchored, so this holds -- but it
    holds by a property of those patterns rather than by construction, so it is
    pinned here: adding a trailing anchor to any of them would silently zero the
    metric on a log that ever carries CRLF.
    """
    path = tmp_path / "gen_server_0.log"
    path.write_bytes(b"".join(_iter_line(n, 1, "7.0ms").encode() + b"\r\n" for n in range(5, 15)))

    stats = perf_sanity.parse_gen_worker_device_step_time(str(tmp_path), 1)

    assert stats.mean == pytest.approx(7.0)


def test_end_offsets_confines_a_client_to_its_own_segment(tmp_path: Path) -> None:
    """Without an end bound the first client would report the whole run.

    A multi-client mode appends every client's iterations to the same
    gen_server_{i}.log, and the parse is deferred until after teardown, so at
    parse time all segments are already on disk. An unbounded read would give
    client 0 a mean averaged over client 1's iterations too -- silently wrong
    rather than absent, which is why this is pinned.
    """
    split = _two_segment_gen_log(tmp_path)

    first = perf_sanity.parse_gen_worker_device_step_time(
        str(tmp_path), 1, start_offsets=[0], end_offsets=[split]
    )
    second = perf_sanity.parse_gen_worker_device_step_time(
        str(tmp_path), 1, start_offsets=[split], end_offsets=None
    )

    assert first.mean == pytest.approx(7.0)
    assert second.mean == pytest.approx(70.0)


def test_no_end_offsets_reads_to_eof(tmp_path: Path) -> None:
    """Backward compatibility: the single-client gen_only lane passes None.

    That lane must stay byte-identical to before the window existed, so this
    asserts the unbounded read still spans both segments (mean of 7 and 70).
    """
    _two_segment_gen_log(tmp_path)

    stats = perf_sanity.parse_gen_worker_device_step_time(str(tmp_path), 1)

    assert stats.mean == pytest.approx(38.5)


def test_an_end_offset_past_eof_is_harmless(tmp_path: Path) -> None:
    """The bound comes from a getsize() snapshot of a file still being written.

    It can therefore sit past what a later reader sees only if the log were
    truncated, but it must degrade to "read everything" rather than raise.
    """
    _two_segment_gen_log(tmp_path)

    stats = perf_sanity.parse_gen_worker_device_step_time(
        str(tmp_path), 1, start_offsets=[0], end_offsets=[10**9]
    )

    assert stats.mean == pytest.approx(38.5)


def test_a_line_straddling_the_end_offset_is_dropped(tmp_path: Path) -> None:
    """A bound mid-line means the worker was flushing; drop that one row.

    The dropped row belongs to the earlier client, so failing this direction
    costs one iteration out of hundreds. Reading on instead would pull in every
    later client's rows, which is unbounded error.
    """
    split = _two_segment_gen_log(tmp_path)

    stats = perf_sanity.parse_gen_worker_device_step_time(
        str(tmp_path), 1, start_offsets=[0], end_offsets=[split - 20]
    )

    # 9 of the first segment's 10 rows, none of the second's.
    assert stats.mean == pytest.approx(7.0)
    assert stats.std == pytest.approx(0.0)


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
                "end_offsets": None,
            }
        ],
        outputs,
    )

    metrics: dict[str, float] = {}
    for line in outputs[0].split("\n"):
        for name, regex in perf_sanity.DEVICE_STEP_TIME_LOG_QUERIES.items():
            if name in metrics:
                continue
            match = regex.search(line)
            if match:
                metrics[name] = float(match.group(1))
                break

    assert set(metrics) == set(perf_sanity.DEVICE_STEP_TIME_METRICS)
    assert metrics["mean_gen_worker_per_iter_device_step_time"] == pytest.approx(7.17, abs=0.01)
    assert metrics["std_gen_worker_per_iter_device_step_time"] > 0.0


def test_every_device_step_time_metric_is_a_minimize_metric() -> None:
    """A metric absent from both lists raises ValueError in check_regression."""
    for name in perf_sanity.DEVICE_STEP_TIME_METRICS:
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


@pytest.mark.parametrize("mode", ["gen_only", "e2e", perf_sanity.E2E_TIME_BREAKDOWN_MODE])
def test_add_perf_metric_value_uploads_the_family_for_every_gen_worker_mode(
    mode: str,
) -> None:
    """Every mode with a gen worker publishes all five statistics."""
    metrics = dict.fromkeys(perf_sanity.PERF_METRIC_LOG_QUERIES, 1.0)
    for name in perf_sanity.DEVICE_STEP_TIME_METRICS:
        metrics[name] = 7.0

    new_data: dict = {}
    perf_sanity.add_perf_metric_value(new_data, metrics, False, mode)

    for name in perf_sanity.DEVICE_STEP_TIME_METRICS:
        assert new_data[f"d_{name}"] == pytest.approx(7.0)


def test_add_perf_metric_value_omits_the_family_without_a_benchmark_mode() -> None:
    """ctx_only and the aggregated lanes call this with benchmark_mode=None.

    ctx_only runs the *aggregated* runtime from a disagg YAML, so it has no gen
    worker and no gen_server_*.log; its call site passes no benchmark_mode at
    all. Pinned because ``None in DEVICE_STEP_TIME_MODES`` being False is the
    only thing keeping the family off those rows.
    """
    metrics = dict.fromkeys(perf_sanity.PERF_METRIC_LOG_QUERIES, 1.0)
    metrics["mean_gen_worker_per_iter_device_step_time"] = 7.0

    new_data: dict = {}
    perf_sanity.add_perf_metric_value(new_data, metrics, False, None)

    assert not [key for key in new_data if "gen_worker_per_iter" in key]
    assert "ctx_only" not in perf_sanity.DEVICE_STEP_TIME_MODES


def test_the_family_gates_only_in_gen_only() -> None:
    """The executable form of "upload in e2e, but never gate on it there".

    Modes outside gen_only take the ``else`` branch in
    get_regression_check_config and get REGRESSION_METRICS, so the only way a
    device-step-time name could ever set b_is_regression for e2e is by leaking
    into that default list. MINIMIZE_METRICS membership still buys each name a
    baseline and an s_regression_info diff line, which is the diagnostic value --
    the two lists are independent.
    """
    for name in perf_sanity.DEVICE_STEP_TIME_METRICS:
        assert f"d_{name}" not in perf_sanity.REGRESSION_METRICS
        assert f"d_{name}" in perf_sanity.MINIMIZE_METRICS
    assert not set(perf_sanity.GEN_ONLY_REGRESSION_METRICS) & set(perf_sanity.REGRESSION_METRICS)


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
    emitted = {f"d_{name}" for name in perf_sanity.DEVICE_STEP_TIME_METRICS}
    assert set(perf_sanity.GEN_ONLY_REGRESSION_METRICS) <= emitted
