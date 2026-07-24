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
"""Unit tests for the gen-worker per-iter device-step-time parser.

These are pure-CPU tests over synthetic ``gen_server_{i}.log`` content — no GPU,
no model. They pin the behavior of ``parse_gen_worker_device_step_time`` and its
helpers ``_scan_gen_worker_device_step_time`` / ``_mean_at_mode_ngen``.

Regression coverage for nvbugs 6487036 / 6487040: a gen_only disagg run whose
gen worker logged perfectly good ``prev_device_step_time`` values still parsed
the ``mean_gen_worker_per_iter_device_step_time`` metric as ``None`` — because
PR #16298 began requiring a parseable ``num_generation_tokens`` on every line
and dropped a whole worker when none matched. A ``None`` metric then raised a
spurious ``check_test_failure``. The fix falls back to an un-bucketed all-iter
mean when ``num_generation_tokens`` is unparsable on every row of a worker, so
a present metric is never lost.

``test_perf_sanity`` pulls heavy integration imports (conftest ->
``tensorrt_llm.bindings``, the OpenSearch DB utils, ...) at module load, so this
test reads a fixed source slice out of that module and ``exec``-s just the
self-contained parser block with a stubbed ``print_info`` rather than importing
the whole module. The block is purely regex + file IO + arithmetic, so exec-ing
it in isolation is faithful.
"""

import os
import re
import time
from typing import Dict, List, Optional, Tuple

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TPS_PATH = os.path.join(_THIS_DIR, "test_perf_sanity.py")


def _load_parser_namespace() -> dict:
    """Exec the parser block out of test_perf_sanity.py into an isolated ns.

    We slice from the ``_DEVICE_STEP_TIME_RE`` module constant through the end
    of ``parse_gen_worker_device_step_time`` (the marker is the next top-level
    ``def add_perf_metric_value``). This avoids importing test_perf_sanity's
    GPU/DB dependency chain while running the exact shipped source.
    """
    src = open(_TPS_PATH).read()
    start = src.index("_DEVICE_STEP_TIME_RE = re.compile")
    end = src.index("def add_perf_metric_value(")
    block = src[start:end]
    ns = {
        "re": re,
        "os": os,
        "time": time,
        "Optional": Optional,
        "List": List,
        "Dict": Dict,
        "Tuple": Tuple,
        # print_info is imported at module scope in test_perf_sanity; stub it.
        "print_info": lambda *a, **k: None,
    }
    # _TPS_PATH is a fixed, repository-local source file (the sibling
    # test_perf_sanity.py), not external/untrusted input; exec-ing a slice of it
    # is how we run the shipped parser without its GPU/DB import chain.
    exec(compile(block, _TPS_PATH, "exec"), ns)  # noqa: S102
    return ns


_NS = _load_parser_namespace()
parse_gen_worker_device_step_time = _NS["parse_gen_worker_device_step_time"]
_scan_gen_worker_device_step_time = _NS["_scan_gen_worker_device_step_time"]
_mean_at_mode_ngen = _NS["_mean_at_mode_ngen"]


def _iter_line(iter_n: int, dt_ms, ngen_render: Optional[str]) -> str:
    """Build a gen-worker iter log line matching the real emitter format.

    ``ngen_render`` is the verbatim ``'num_generation_tokens': <...>`` fragment
    to embed in the states dict, or ``None`` to omit the key entirely.
    """
    states = "'num_ctx_requests': 0, 'num_ctx_tokens': 0"
    if ngen_render is not None:
        states += f", {ngen_render}"
    states += ", 'cached_kv_tokens': 0"
    return (
        f"[07/22/2026-00:00:00] [TRT-LLM] [I] [_torch][RANK 0] "
        f"iter = {iter_n}, global_rank = 0, rank = 0, "
        f"num_scheduled_requests = 1, kv_cache_util = 0.100, "
        f"currank_total_requests = 1/1, host_step_time = 5.0ms, "
        f"prev_device_step_time = {dt_ms}ms, timestamp = 2026-07-22 00:00:00, "
        f"states = {{{states}}}"
    )


def _write_gen_log(output_dir: str, idx: int, lines: List[str]) -> None:
    with open(os.path.join(output_dir, f"gen_server_{idx}.log"), "w") as f:
        f.write("\n".join(lines) + "\n")


# A short settle window so the polling loop returns promptly in unit tests
# (two scans of static files agree immediately).
_FAST = dict(settle_timeout=0.2, poll_interval=0.01)


def test_healthy_bare_int_uses_mode_ngen_bucket(tmp_path):
    """Bare-int num_generation_tokens: the steady-state mode-ngen mean is used.

    Two ngen buckets; the larger-count bucket (256, 20 rows @ 15.0) is the mode
    and wins over a 3-row 512 bucket @ 99.0.
    """
    lines = [_iter_line(i, 15.0, "'num_generation_tokens': 256") for i in range(5, 25)]
    lines += [_iter_line(i, 99.0, "'num_generation_tokens': 512") for i in range(25, 28)]
    _write_gen_log(str(tmp_path), 0, lines)
    assert parse_gen_worker_device_step_time(str(tmp_path), 1, **_FAST) == pytest.approx(15.0)


def test_iters_below_five_are_skipped(tmp_path):
    """Iters 0-4 (KV-transfer + warmup) never contribute to the mean."""
    lines = [_iter_line(i, 1000.0, "'num_generation_tokens': 4") for i in range(0, 5)]
    lines += [_iter_line(i, 12.0, "'num_generation_tokens': 4") for i in range(5, 15)]
    _write_gen_log(str(tmp_path), 0, lines)
    assert parse_gen_worker_device_step_time(str(tmp_path), 1, **_FAST) == pytest.approx(12.0)


def test_na_device_step_time_is_ignored(tmp_path):
    """A non-numeric prev_device_step_time = N/A line does not match / crash."""
    lines = [_iter_line(5, "N/A", "'num_generation_tokens': 4")]  # unmatched
    lines += [_iter_line(i, 14.0, "'num_generation_tokens': 4") for i in range(6, 16)]
    _write_gen_log(str(tmp_path), 0, lines)
    assert parse_gen_worker_device_step_time(str(tmp_path), 1, **_FAST) == pytest.approx(14.0)


def test_missing_num_generation_tokens_falls_back_to_all_iter_mean(tmp_path):
    """A worker whose lines omit num_generation_tokens must not parse to None.

    Regression for nvbugs 6487036 / 6487040: it falls back to the un-bucketed
    all-iter mean of prev_device_step_time instead of dropping the worker.
    """
    lines = [_iter_line(i, 13.5, ngen_render=None) for i in range(5, 25)]
    _write_gen_log(str(tmp_path), 0, lines)
    result = parse_gen_worker_device_step_time(str(tmp_path), 1, **_FAST)
    assert result is not None
    assert result == pytest.approx(13.5)


@pytest.mark.parametrize(
    "ngen_render",
    [
        "'num_generation_tokens': tensor(256)",
        "'num_generation_tokens': np.int64(256)",
    ],
)
def test_unparsable_num_generation_tokens_falls_back(tmp_path, ngen_render):
    """An unparsable num_generation_tokens render still yields a value.

    When num_generation_tokens renders in a form the bucket regex can't read
    (e.g. a tensor / numpy scalar repr) on every row, fall back to the all-iter
    mean rather than dropping the worker to None.
    """
    lines = [_iter_line(i, 12.9, ngen_render) for i in range(5, 25)]
    _write_gen_log(str(tmp_path), 0, lines)
    result = parse_gen_worker_device_step_time(str(tmp_path), 1, **_FAST)
    assert result is not None
    assert result == pytest.approx(12.9)


def test_mixed_workers_bucket_and_fallback_average(tmp_path):
    """Multi-worker average combines a bucketed worker with a fallback worker.

    worker0 (bucketed) -> 15.0, worker1 (ngen absent) -> 20.0, mean across
    workers -> 17.5.
    """
    _write_gen_log(
        str(tmp_path),
        0,
        [_iter_line(i, 15.0, "'num_generation_tokens': 256") for i in range(5, 25)],
    )
    _write_gen_log(
        str(tmp_path),
        1,
        [_iter_line(i, 20.0, ngen_render=None) for i in range(5, 25)],
    )
    assert parse_gen_worker_device_step_time(str(tmp_path), 2, **_FAST) == pytest.approx(17.5)


def test_no_usable_lines_returns_none(tmp_path):
    """No iter>=5 numeric prev_device_step_time line anywhere yields None.

    This is the only case that should still return None.
    """
    _write_gen_log(
        str(tmp_path),
        0,
        [
            "some banner line without a device step time",
            _iter_line(2, 99.0, "'num_generation_tokens': 4"),  # iter < 5
            _iter_line(3, 99.0, ngen_render=None),  # iter < 5
        ],
    )
    assert parse_gen_worker_device_step_time(str(tmp_path), 1, **_FAST) is None


def test_missing_log_file_returns_none(tmp_path):
    """No gen_server_*.log at all -> None (nothing to parse)."""
    assert parse_gen_worker_device_step_time(str(tmp_path), 1, **_FAST) is None


def test_scan_counts_all_usable_rows_regardless_of_ngen(tmp_path):
    """_scan total_count counts every iter>=5 numeric row (the settle signal).

    Rows whose num_generation_tokens did not parse are still counted, and the
    per-file record still carries the all-iter fallback aggregate for them.
    """
    # Distinct step times for the bucketed vs fallback-only rows so the asserts
    # can tell the mode-ngen bucket mean apart from the all-iter mean: a healthy
    # bucket must win over the fallback when any bucket exists.
    lines = [_iter_line(i, 10.0, "'num_generation_tokens': 8") for i in range(5, 15)]
    lines += [_iter_line(i, 99.0, ngen_render=None) for i in range(15, 20)]
    _write_gen_log(str(tmp_path), 0, lines)
    per_file_scans, total_count = _scan_gen_worker_device_step_time(str(tmp_path), 1)
    assert total_count == 15  # 10 bucketed + 5 fallback-only, all iter>=5
    assert len(per_file_scans) == 1
    by_ngen, all_count, all_mean = per_file_scans[0]
    assert by_ngen == {8: (10, pytest.approx(10.0))}
    assert all_count == 15
    assert all_mean == pytest.approx((10 * 10.0 + 5 * 99.0) / 15)
    # With a non-empty bucket present, selection uses the bucket mean (10.0),
    # NOT the all-iter mean — the fallback only fires when buckets are empty.
    assert _mean_at_mode_ngen(per_file_scans) == pytest.approx(10.0)
