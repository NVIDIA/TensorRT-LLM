# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""TensorRT LLM perf sanity tests."""

import copy
import fcntl
import glob
import http.client
import json
import math
import os
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, NamedTuple, Optional, Tuple

import pytest
import yaml
from test_common.error_utils import report_error
from test_common.http_utils import fail_if_proc_died, wait_for_endpoint_ready
from test_common.perf_sanity_matching import get_test_case_match_keys

from defs.common import wait_for_reported_addr
from defs.trt_test_alternative import print_info, print_warning

from ..conftest import get_llm_root, llm_models_root
from ._model_paths import MODEL_PATH_DICT
from .perf_regression_utils import _percentile, process_and_upload_test_results

SUPPORTED_GPU_MAPPING = {
    "GB200": "gb200",
    "GB300": "gb300",
    "B200": "b200",
    "B300": "b300",
    "H200": "h200",
}

# benchmark_client value selecting the AgentX trace-replay client
# (agentx_client.py). Any other non-empty value is rejected at parse time.
AGENTX_BENCHMARK_CLIENT = "agentx"

WARMUP_BENCHMARK_MODES = ("e2e", "ctx_only")


def wants_warmup(benchmark_mode: str) -> bool:
    return benchmark_mode in WARMUP_BENCHMARK_MODES


BENCH_SERVING_REPO = "https://github.com/kedarpotdar-nv/bench_serving.git"
BENCH_SERVING_COMMIT = "f3ea022a5780de5d0babc5fffa53634e2023d28f"
BENCH_SERVING_DIR = "/tmp/bench_serving"


def ensure_bench_serving_repo() -> str:
    """Clone bench_serving repo if not already present. Returns path to benchmark_serving.py.

    Uses a file lock to avoid race conditions when multiple ranks within the
    same container simultaneously attempt to clone the repository.
    """
    bench_script = os.path.join(BENCH_SERVING_DIR, "benchmark_serving.py")
    lock_file = BENCH_SERVING_DIR + ".lock"

    with open(lock_file, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if not os.path.exists(bench_script):
                if os.path.exists(BENCH_SERVING_DIR):
                    shutil.rmtree(BENCH_SERVING_DIR)
                subprocess.check_call(
                    ["git", "clone", "--depth", "1", BENCH_SERVING_REPO, BENCH_SERVING_DIR]
                )
                subprocess.check_call(
                    [
                        "git",
                        "-C",
                        BENCH_SERVING_DIR,
                        "fetch",
                        "--depth",
                        "1",
                        "origin",
                        BENCH_SERVING_COMMIT,
                    ]
                )
                subprocess.check_call(
                    ["git", "-C", BENCH_SERVING_DIR, "checkout", BENCH_SERVING_COMMIT]
                )
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

    return bench_script


DEFAULT_TIMEOUT = 10800
# Defaults for the server *ready* wait, separate from the whole-test timeout:
# a server that is not healthy after this long is not going to be, and failing
# here (with server-log tails, see wait_for_endpoint_ready) instead of at the
# per-test pytest kill both saves GPU-hours and leaves a classifiable failure
# in the CI log. The disagg bound is larger because its /health only answers
# once EVERY ctx/gen worker has finished model load + autotune + warmup.
# 1800 proved too tight for the largest agg cases: gb300 DeepSeek-V4-Pro
# ctx_only (con4301) needs ~2000s of model load + autotune before /health
# answers, so it failed readiness while the server was still coming up
# (nvbugs/6517846). Raised to match the disagg bound.
AGG_SERVER_READY_TIMEOUT = 3600
DISAGG_SERVER_READY_TIMEOUT = 3600
# GEN workers normally reap within seconds after benchmark_status is written.
# Keep this well below the whole-test timeout so a stuck multi-node srun cannot
# turn the optional log-flush synchronization into a pytest/Slurm cancellation.
GEN_LOG_SENTINEL_TIMEOUT = 120


def server_ready_timeout(default: int, mode: str) -> int:
    """Ready-wait bound for one serving mode ("AGG" or "DISAGG").

    Agg and disagg servers have very different init times (disagg's /health
    answers only after every ctx/gen worker is up), so each mode has its own
    override var, with the generic one as a shared fallback:
    TRTLLM_TEST_<mode>_SERVER_READY_TIMEOUT > TRTLLM_TEST_SERVER_READY_TIMEOUT
    > the built-in per-mode default.

    Read at call time (not import time) so the env vars can be adjusted per
    invocation, and parsed defensively so a malformed value cannot break
    pytest collection of this module.
    """
    for var in (
        f"TRTLLM_TEST_{mode.upper()}_SERVER_READY_TIMEOUT",
        "TRTLLM_TEST_SERVER_READY_TIMEOUT",
    ):
        raw = os.environ.get(var)
        if not raw:
            continue
        try:
            timeout = int(raw)
        except ValueError:
            timeout = 0
        if timeout > 0:
            return timeout
        print_info(f"Invalid {var}={raw!r}; ignoring it")
    return default


AGG_CONFIG_FOLDER = os.environ.get("AGG_CONFIG_FOLDER", "tests/scripts/perf-sanity/aggregated")
DISAGG_CONFIG_FOLDER = os.environ.get(
    "DISAGG_CONFIG_FOLDER", "tests/scripts/perf-sanity/disaggregated"
)

# Regex patterns for parsing benchmark output metrics
# Key is the metric name used in database (e.g., "mean_e2el", "seq_throughput")
PERF_METRIC_LOG_QUERIES = {
    "seq_throughput": re.compile(r"Request throughput \(req\/s\):\s+(-?[\d\.]+)"),
    "token_throughput": re.compile(r"Output token throughput \(tok\/s\):\s+(-?[\d\.]+)"),
    "total_token_throughput": re.compile(r"Total Token throughput \(tok\/s\):\s+(-?[\d\.]+)"),
    "user_throughput": re.compile(r"User throughput \(tok\/s\):\s+(-?[\d\.]+)"),
    "mean_ttft": re.compile(r"Mean TTFT \(ms\):\s+(-?[\d\.]+)"),
    "median_ttft": re.compile(r"Median TTFT \(ms\):\s+(-?[\d\.]+)"),
    "p99_ttft": re.compile(r"P99 TTFT \(ms\):\s+(-?[\d\.]+)"),
    "mean_itl": re.compile(r"Mean ITL \(ms\):\s+(-?[\d\.]+)"),
    "median_itl": re.compile(r"Median ITL \(ms\):\s+(-?[\d\.]+)"),
    "p99_itl": re.compile(r"P99 ITL \(ms\):\s+(-?[\d\.]+)"),
    "mean_tpot": re.compile(r"Mean TPOT \(ms\):\s+(-?[\d\.]+)"),
    "median_tpot": re.compile(r"Median TPOT \(ms\):\s+(-?[\d\.]+)"),
    "p99_tpot": re.compile(r"P99 TPOT \(ms\):\s+(-?[\d\.]+)"),
    "mean_e2el": re.compile(r"Mean E2EL \(ms\):\s+(-?[\d\.]+)"),
    "median_e2el": re.compile(r"Median E2EL \(ms\):\s+(-?[\d\.]+)"),
    "p99_e2el": re.compile(r"P99 E2EL \(ms\):\s+(-?[\d\.]+)"),
}

# Spec-decoding-only metrics: parsed from benchmark output but only stored
# (and regression-checked) when the test runs with speculative decoding.
SPEC_DECODING_PERF_METRIC_LOG_QUERIES = {
    "al": re.compile(r"Mean Avg Decoded Tokens per Iter:\s+(-?[\d\.]+)"),
}

# gen_only-only metrics: appended to each trtllm-benchmark log by
# DisaggTestCmds.run_cmd after parsing gen_server_*.log; only forwarded to
# the database for gen_only mode.
#
# The distribution is published, not just the mean, because the mean alone is
# not self-diagnosing: a single anomalous iteration can move it by >30% while
# the workload is unchanged (nvbugs 6627789), and the only way a reader can
# tell that from a real regression is to see the spread next to it. The mean
# and median are both regression-gated (see regression_metrics in the gen_only
# branch) because they fail on different shapes of slowdown; std/p75/p99 are
# uploaded for diagnosis only.
#
# One statistic per line, and the leading words must stay mutually exclusive:
# parse_metrics_from_output breaks out of the regex loop on the first match per
# line, so a shared prefix would silently shadow whichever pattern lost the
# ordering race.
GEN_ONLY_PERF_METRIC_LOG_QUERIES = {
    "mean_gen_worker_per_iter_device_step_time": re.compile(
        r"Average Per Iter Device Step Time \(ms\):\s+(-?[\d\.]+)"
    ),
    "median_gen_worker_per_iter_device_step_time": re.compile(
        r"Median Per Iter Device Step Time \(ms\):\s+(-?[\d\.]+)"
    ),
    "std_gen_worker_per_iter_device_step_time": re.compile(
        r"Stdev Per Iter Device Step Time \(ms\):\s+(-?[\d\.]+)"
    ),
    "p75_gen_worker_per_iter_device_step_time": re.compile(
        r"P75 Per Iter Device Step Time \(ms\):\s+(-?[\d\.]+)"
    ),
    "p99_gen_worker_per_iter_device_step_time": re.compile(
        r"P99 Per Iter Device Step Time \(ms\):\s+(-?[\d\.]+)"
    ),
}

# Every gen_only device-step-time metric, in log-line order. The mean is first
# because it is the one check_test_failure keys on; mean and median are both
# regression-gated, std/p75/p99 are diagnostic.
GEN_ONLY_DEVICE_STEP_TIME_METRICS = (
    "mean_gen_worker_per_iter_device_step_time",
    "median_gen_worker_per_iter_device_step_time",
    "std_gen_worker_per_iter_device_step_time",
    "p75_gen_worker_per_iter_device_step_time",
    "p99_gen_worker_per_iter_device_step_time",
)

# The regression gate for disagg gen_only lanes. Mean and median are both gated
# because they fail on different shapes of slowdown: the mean catches a cost
# spread thinly across many iterations, the median catches a shift in the typical
# iteration while ignoring outliers. A real slowdown moves both; a single
# anomalous iteration moves only the mean, so the pair is self-diagnosing on the
# CI report itself. std/p75/p99 are uploaded for diagnosis but not gated.
#
# Every name here must also appear in MINIMIZE_METRICS (or MAXIMIZE_METRICS):
# check_regression only iterates those two lists, so a gated name absent from
# both is silently never checked. test_perf_sanity_helpers.py pins that.
GEN_ONLY_REGRESSION_METRICS = (
    "d_mean_gen_worker_per_iter_device_step_time",
    "d_median_gen_worker_per_iter_device_step_time",
)

# Per-iter prev_device_step_time logged by each gen worker. Example line:
#   [TRT-LLM] [I] [_torch][RANK 0] iter = 5, global_rank = 0, ...,
#   host_step_time = 6.79ms, prev_device_step_time = 6.94ms, ...,
#   states = {..., 'num_generation_tokens': 512, ...}
# Only the gen worker (decode) emits this. The device value reported at iter N
# is the device step time of iter N-1 (device runs async). Iters 0-4 are
# skipped: iter 0/1 include KV-cache transfer wait time, and iters 2-4 are
# warmup that has not yet reached steady state. prev_device_step_time may be
# 'N/A' (e.g. iter 1); the regex requires a numeric value so those lines do
# not match. num_generation_tokens is captured by a separate regex applied
# to the same line (name is stable, order relative to prev_device_step_time
# is not) so the scanner can bucket rows by ngen without silently dropping
# any line whose states dict is printed before prev_device_step_time. See
# _scan_gen_worker_device_step_time.
_DEVICE_STEP_TIME_RE = re.compile(r"iter\s*=\s*(\d+),.*?prev_device_step_time\s*=\s*([\d.]+)\s*ms")
_NUM_GEN_TOKENS_RE = re.compile(r"'num_generation_tokens':\s*(\d+)")
# num_scheduled_requests from the same line. An iteration that scheduled zero
# requests did no GPU work, so its loop period is pure idle (waiting on KV-cache
# transfer) -- and because the device runs async that period is reported by the
# NEXT iteration's prev_device_step_time. Such a row is excluded; see
# _scan_gen_worker_device_step_time. The ' = ' spelling is what
# py_executor.py's iteration log actually emits (do not copy the ': ' form in
# examples/wide_ep/slurm_scripts/process_gen_iterlog.py, which is stale).
_ITER_NSR_RE = re.compile(r"iter\s*=\s*(\d+),.*?num_scheduled_requests\s*=\s*(\d+)")
# The emitting rank, used to key _scan_gen_worker_device_step_time's predecessor
# bookkeeping so that interleaved ranks in one file cannot be read as each
# other's predecessor.
# py_executor.py only logs rank 0 by default (TLLM_PROFILE_LOG_RANKS, default
# "0"), and no lane sets that variable today -- both artifact sets for nvbug
# 6627789 are 518/518 global_rank = 0. But the variable accepts "all" or a rank
# list and lane YAML can inject arbitrary server env vars, and in a mixed-rank
# file the failure would be silent in the WRONG direction: the contaminated
# row's immediate predecessor line would usually belong to a different rank,
# whose num_scheduled_requests is nonzero, so the exclusion would quietly stop
# excluding while still looking armed. Matching global_rank (not the trailing
# 'rank = ') keeps this unambiguous; a line with neither shares one bucket,
# which is exactly the pre-existing single-rank behaviour.
_ITER_RANK_RE = re.compile(r"global_rank\s*=\s*(\d+)")

# Hard cap on retained per-iteration samples per gen worker. The percentile and
# stdev statistics need the whole sample, so the scan cannot be O(1) memory the
# way a streaming mean could. A steady-state run holds ~512 rows per worker
# (~16 KB), so this bound is ~1000x headroom and exists only to keep a
# pathological log (a runaway worker, a concatenated log) from growing the
# scan without limit. Excess rows are dropped, not sampled: truncating the tail
# keeps the steady-state plateau these statistics describe.
_MAX_RETAINED_ITER_ROWS = 500_000


class _IterRow(NamedTuple):
    """One usable per-iteration sample from a gen worker log.

    ngen is the line's num_generation_tokens, or None when it did not parse
    (see _scan_gen_worker_device_step_time for why such rows are retained).
    """

    ngen: Optional[int]
    device_step_time: float


class _DeviceStepTimeStats(NamedTuple):
    """Distribution of gen-worker per-iter device step time, in ms."""

    mean: float
    median: float
    std: float
    p75: float
    p99: float


def _stdev(values: List[float]) -> float:
    """Sample standard deviation (ddof=1). Returns 0.0 for fewer than 2 values.

    ddof=1 because the iterations are a sample of the workload's steady state,
    not its entire population. A single-sample file reports 0.0 rather than
    raising: the metric is diagnostic, and a run that produced one usable
    iteration has bigger problems than its spread.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def gen_worker_log_sizes(output_dir: str, num_gen_servers: int) -> List[int]:
    """Current byte size of each gen_server_{i}.log (0 if missing).

    Used to delimit per-client segments in DisaggTestCmds.run_cmd: snapshot
    sizes before launching a client, then pass the snapshot as start_offsets
    to parse_gen_worker_device_step_time after the client exits.
    """
    sizes: List[int] = []
    for i in range(num_gen_servers):
        log_path = os.path.join(output_dir, f"gen_server_{i}.log")
        sizes.append(os.path.getsize(log_path) if os.path.isfile(log_path) else 0)
    return sizes


def _scan_gen_worker_device_step_time(
    output_dir: str,
    num_gen_servers: int,
    start_offsets: Optional[List[int]] = None,
) -> List[List[_IterRow]]:
    """Single-pass scan of the gen logs.

    Returns one list of _IterRow per file that produced at least one usable
    row. A row is usable when iter >= 5, prev_device_step_time is numeric, and
    the row is not the successor of an empty iteration (below). _IterRow.ngen
    is None when num_generation_tokens did not parse on that line; such rows
    are retained rather than dropped so a worker whose states dict renders it
    unparseably (e.g. tensor(256), nvbugs 6487036 / 6487040) still produces a
    metric instead of None -- PR #16298 began requiring num_generation_tokens
    on every line, and dropping those rows would silently lose the metric.

    Empty-iteration successors are excluded. An iteration with
    num_scheduled_requests == 0 did no GPU work, so its loop period is entirely
    idle -- typically waiting for KV-cache transfer in a disaggregated run --
    and because the device runs one step behind, that idle period is what the
    NEXT iteration reports as prev_device_step_time. Averaging it in credits
    the GPU with hundreds or thousands of milliseconds of "step time" that no
    kernel spent, which is how nvbug 6627789 read a +19% regression out of two
    runs whose steady-state iterations were both ~7.3 ms.

    The row is dropped only when that rank's immediately preceding parsed line
    is provably the predecessor iteration: pred_iter == cur_iter - 1.
    Predecessor state is kept per emitting rank (_ITER_RANK_RE), so ranks
    interleaved in one file are never read as each other's predecessor. If the
    predecessor is missing, unparsable, or non-adjacent (a restarted iteration
    counter, the first line after a seek), the row is KEPT.
    That is the safe direction to fail: the exclusion is an accuracy
    improvement on a metric that must keep reporting, so a scan that cannot
    prove a row is idle-contaminated should behave exactly as it did before.
    Note the num_scheduled_requests == 0 row itself is kept -- its own
    prev_device_step_time describes the previous iteration, which did work.

    Memory is O(retained rows), bounded by _MAX_RETAINED_ITER_ROWS per file:
    the percentile and stdev statistics need the whole sample, unlike the
    streaming mean this replaced.

    errors="replace" guards against invalid UTF-8: tqdm progress bars
    (model load) write partial multibyte sequences that would otherwise raise
    UnicodeDecodeError mid-scan.
    """
    per_file_rows: List[List[_IterRow]] = []
    for i in range(num_gen_servers):
        log_path = os.path.join(output_dir, f"gen_server_{i}.log")
        if not os.path.isfile(log_path):
            continue

        seek_to = (
            start_offsets[i]
            if start_offsets is not None and i < len(start_offsets) and start_offsets[i]
            else 0
        )

        rows: List[_IterRow] = []
        # rank -> (iter, num_scheduled_requests) of that rank's previous line.
        prev_by_rank: Dict[Optional[int], Tuple[Optional[int], Optional[int]]] = {}
        with open(log_path, errors="replace") as f:
            if seek_to:
                f.seek(seek_to)
            for line in f:
                # Every iteration line carries this literal, including the ones
                # whose value is 'N/A', so this fast-reject cannot skip a line
                # the num_scheduled_requests tracking below needs to see.
                if "prev_device_step_time" not in line:
                    continue
                # Snapshot this rank's predecessor before this line overwrites it.
                rank_m = _ITER_RANK_RE.search(line)
                rank = int(rank_m.group(1)) if rank_m is not None else None
                pred_iter, pred_nsr = prev_by_rank.get(rank, (None, None))
                nsr_m = _ITER_NSR_RE.search(line)
                if nsr_m is None:
                    prev_by_rank[rank] = (None, None)
                else:
                    prev_by_rank[rank] = (int(nsr_m.group(1)), int(nsr_m.group(2)))

                m = _DEVICE_STEP_TIME_RE.search(line)
                if m is None:
                    continue
                cur_iter = int(m.group(1))
                # iter 0/1 include KV-cache transfer wait; 2-4 are warmup.
                if cur_iter < 5:
                    continue
                if pred_nsr == 0 and pred_iter is not None and pred_iter == cur_iter - 1:
                    continue
                if len(rows) >= _MAX_RETAINED_ITER_ROWS:
                    continue
                ngen_m = _NUM_GEN_TOKENS_RE.search(line)
                rows.append(
                    _IterRow(
                        ngen=int(ngen_m.group(1)) if ngen_m is not None else None,
                        device_step_time=float(m.group(2)),
                    )
                )
        if rows:
            per_file_rows.append(rows)
    return per_file_rows


def _stats_at_mode_ngen(
    per_file_rows: List[List[_IterRow]],
) -> Optional[_DeviceStepTimeStats]:
    """Aggregate per-file rows into one set of distribution statistics.

    Within each file pick the num_generation_tokens value with the most
    iterations (the mode) and describe only that bucket; ties break to the
    largest ngen because the steady-state plateau is the upper of any tied
    clusters. Mode is more robust than strict == max -- a one-off spike where
    a single iter's ngen briefly exceeds the sustained batch would otherwise
    collapse the statistics to 1-2 samples. Iterations near the end of a run
    have a shrinking num_generation_tokens as sequences finish and land in
    smaller-ngen buckets, so they do not drag the mean below steady state.
    When a file produced usable rows but no parseable num_generation_tokens on
    any of them, fall back to that file's whole sample so a present metric is
    never lost (nvbugs 6487036 / 6487040).

    Then average each statistic across workers, unweighted -- one vote per
    worker, matching how the mean has always been combined. Averaging a median
    or a percentile across workers is not itself a median or a percentile of
    the pooled sample; these are per-worker statistics summarised across
    workers, which is the comparison the regression check makes.

    Returns None if no file had a usable row.
    """
    per_file_stats: List[_DeviceStepTimeStats] = []
    for rows in per_file_rows:
        by_ngen: Dict[int, List[float]] = {}
        for row in rows:
            if row.ngen is not None:
                by_ngen.setdefault(row.ngen, []).append(row.device_step_time)
        if by_ngen:
            _mode_ngen, values = max(by_ngen.items(), key=lambda kv: (len(kv[1]), kv[0]))
        else:
            # No parseable ngen anywhere in this worker; use every row.
            values = [row.device_step_time for row in rows]
        if not values:
            continue
        per_file_stats.append(
            _DeviceStepTimeStats(
                mean=sum(values) / len(values),
                median=_percentile(values, 50),
                std=_stdev(values),
                p75=_percentile(values, 75),
                p99=_percentile(values, 99),
            )
        )
    if not per_file_stats:
        return None
    num_files = len(per_file_stats)
    return _DeviceStepTimeStats(*(sum(column) / num_files for column in zip(*per_file_stats)))


def parse_gen_worker_device_step_time(
    output_dir: str,
    num_gen_servers: int,
    start_offsets: Optional[List[int]] = None,
) -> Optional[_DeviceStepTimeStats]:
    """Per-iter prev_device_step_time statistics (ms) across all gen workers.

    For each gen_server_{i}.log, take the iter >= 5 rows that are not the
    successor of an empty (num_scheduled_requests == 0) iteration, bucket them
    by num_generation_tokens, pick the bucket with the most rows (the mode;
    ties break to the largest ngen), and describe that bucket with mean,
    median, stdev, P75 and P99. Then average each statistic across the
    num_gen_servers workers. A worker whose num_generation_tokens never parses
    falls back to its whole sample rather than being dropped to None. Returns
    None only if no usable line is found in any file.

    The mean and the median are the regression-gated statistics
    (GEN_ONLY_REGRESSION_METRICS); the other three are uploaded for diagnosis,
    because a mean on its own cannot distinguish a slower workload from one
    anomalous iteration. See
    _scan_gen_worker_device_step_time for the empty-iteration exclusion and
    _stats_at_mode_ngen for the bucket selection.

    When start_offsets is provided, only the bytes from start_offsets[i] to
    end-of-file are considered for gen_server_{i}.log — used to slice out a
    single client's iteration segment.

    The log is read exactly once. The caller (DisaggTestCmds.run_cmd) normally
    waits for the gen_server_{i}.done sentinels first, so every gen srun has
    exited and its &> aggregate log is fully flushed. If the dedicated
    sentinel wait expires, the caller parses the current contents instead of
    consuming the whole-test timeout; a missing metric then hard-fails before
    upload. This replaces the earlier settle-poll heuristic, which could
    accept a truncated prefix while the log was still flushing across NFS
    (nvbugs 6487036 / 6487040 / 6487038).
    """
    per_file_rows = _scan_gen_worker_device_step_time(output_dir, num_gen_servers, start_offsets)
    return _stats_at_mode_ngen(per_file_rows)


def add_perf_metric_value(
    new_data: dict,
    metrics: dict,
    spec_decoding: bool,
    benchmark_mode: Optional[str] = None,
) -> None:
    """Populate `new_data` with per-test perf metrics from `metrics`.

    - Always copies every key in PERF_METRIC_LOG_QUERIES as `d_<name>`.
    - Adds `d_al` only when spec_decoding=True *and* the value was parsed;
      non-spec rows omit it so OpenSearch baselines don't blend the two
      populations, and spec rows exempted from reporting it (AgentX) omit it
      rather than failing the upload.
    - Adds the `d_*_gen_worker_per_iter_device_step_time` family only for the
      disagg gen_only mode (the only mode that emits them). Of these the mean
      and the median are regression-gated (GEN_ONLY_REGRESSION_METRICS); the
      rest are uploaded for diagnosis.

    A missing or non-numeric gen_only statistic is omitted rather than
    forwarded: typeCheckForOpenSearchDB rejects both None and int for a `d_`
    key, so uploading one would fail the whole row instead of just losing a
    diagnostic column. check_test_failure separately hard-fails a gen_only run
    whose mean is absent, before results are uploaded.
    """
    for metric_name in PERF_METRIC_LOG_QUERIES:
        new_data[f"d_{metric_name}"] = metrics[metric_name]
    if spec_decoding:
        # 'al' is legitimately absent for AgentX lanes: aiperf does not propagate
        # TRT-LLM's non-standard avg_decoded_tokens_per_iter field. Omit the
        # column instead of raising -- check_test_failure runs immediately before
        # upload and has already hard-failed any non-exempt spec-decoding run
        # whose 'al' is missing, so reaching here without it means the run is
        # exempt by design. Omitted rather than defaulted: typeCheckForOpenSearchDB
        # rejects None for a d_ key (losing the whole row), and a substituted 0.0
        # would corrupt the spec-decoding baseline population.
        al = metrics.get("al")
        if al is not None:
            new_data["d_al"] = al
    if benchmark_mode == "gen_only":
        for metric_name in GEN_ONLY_DEVICE_STEP_TIME_METRICS:
            value = metrics.get(metric_name)
            if value is None:
                continue
            new_data[f"d_{metric_name}"] = float(value)


# Metrics where larger is better
MAXIMIZE_METRICS = [
    "d_seq_throughput",
    "d_token_throughput",
    "d_total_token_throughput",
    "d_user_throughput",
    "d_mean_tpot",
    "d_median_tpot",
    "d_p99_tpot",
    "d_al",
]

# Metrics where smaller is better
MINIMIZE_METRICS = [
    "d_mean_ttft",
    "d_median_ttft",
    "d_p99_ttft",
    "d_mean_itl",
    "d_median_itl",
    "d_p99_itl",
    "d_mean_e2el",
    "d_median_e2el",
    "d_p99_e2el",
    # gen_only-only: per-iter device step time across gen workers. Lower is
    # better for all five, including the spread statistics -- a tighter
    # distribution is a more trustworthy measurement as well as a steadier
    # workload. Mean and median are listed in regression_metrics; std/p75/p99
    # get baselines but cannot fail a build (see check_regression).
    "d_mean_gen_worker_per_iter_device_step_time",
    "d_median_gen_worker_per_iter_device_step_time",
    "d_std_gen_worker_per_iter_device_step_time",
    "d_p75_gen_worker_per_iter_device_step_time",
    "d_p99_gen_worker_per_iter_device_step_time",
]

# Default key metrics that determine regression (throughput metrics only).
# d_al is appended at runtime when any client runs spec decoding.
REGRESSION_METRICS = [
    "d_token_throughput",
    "d_total_token_throughput",
]

STARTUP_METRIC_NAMES = (
    "total_model_loading_seconds",
    "checkpoint_preparation_seconds",
    "weight_population_seconds",
    "checkpoint_finalization_seconds",
    "draft_checkpoint_preparation_seconds",
    "draft_weight_population_seconds",
    "draft_checkpoint_finalization_seconds",
    "post_load_processing_seconds",
)
CHECKPOINT_PIPELINE_PHASES = (
    "checkpoint_preparation_seconds",
    "weight_population_seconds",
    "checkpoint_finalization_seconds",
)
DRAFT_CHECKPOINT_PIPELINE_PHASES = (
    "draft_checkpoint_preparation_seconds",
    "draft_weight_population_seconds",
    "draft_checkpoint_finalization_seconds",
)
CHECKPOINT_IO_POLICY_PATTERN = re.compile(
    r"Checkpoint I/O policy: requested=(?P<requested>[^,]+), "
    r"selected=(?P<selected>[^,]+), activated=(?P<activated>True|False), "
    r"effective=(?P<effective>[^,]+), fallback_reason=(?P<fallback_reason>.*)\."
)


def parse_checkpoint_io_policies(log_paths: List[str]) -> List[dict]:
    """Return every checkpoint I/O status emitted by the server."""
    statuses = []
    for log_path in log_paths:
        if not os.path.exists(log_path):
            continue
        with open(log_path, "r", encoding="utf-8", errors="replace") as log_file:
            for line in log_file:
                for match in CHECKPOINT_IO_POLICY_PATTERN.finditer(line):
                    status = match.groupdict()
                    status["activated"] = status["activated"] == "True"
                    statuses.append(status)
    return statuses


def make_startup_observation(server_info: dict, log_paths: List[str], role: str) -> dict:
    """Normalize startup data from ``/server_info`` and server logs."""
    startup_metrics = server_info.get("startup_metrics", {})
    if not isinstance(startup_metrics, dict):
        startup_metrics = {}
    metrics = {}
    for loader_name, metric_prefix in (
        ("model_loader", ""),
        ("draft_model_loader", "draft_model_"),
    ):
        loader_metrics = startup_metrics.get(loader_name, {})
        if not isinstance(loader_metrics, dict):
            continue
        for metric_name in STARTUP_METRIC_NAMES:
            value = loader_metrics.get(metric_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"{metric_prefix}{metric_name}"] = float(value)

        if all(f"{metric_prefix}{name}" in metrics for name in CHECKPOINT_PIPELINE_PHASES):
            metrics[f"{metric_prefix}checkpoint_pipeline_seconds"] = sum(
                metrics[f"{metric_prefix}{name}"] for name in CHECKPOINT_PIPELINE_PHASES
            )
        if all(f"{metric_prefix}{name}" in metrics for name in DRAFT_CHECKPOINT_PIPELINE_PHASES):
            metrics[f"{metric_prefix}draft_checkpoint_pipeline_seconds"] = sum(
                metrics[f"{metric_prefix}{name}"] for name in DRAFT_CHECKPOINT_PIPELINE_PHASES
            )

    return {
        "role": role,
        "metrics": metrics,
        "checkpoint_io_policies": parse_checkpoint_io_policies(log_paths),
    }


def fetch_startup_observation(server_address: str, log_paths: List[str], role: str) -> dict:
    """Fetch one server's startup metrics after it reports ready."""
    request = urllib.request.Request(
        f"http://{server_address}/server_info",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        server_info = json.load(response)
    if not isinstance(server_info, dict):
        raise ValueError("/server_info did not return a JSON object")
    return make_startup_observation(server_info, log_paths, role)


def collect_startup_observation(
    server_address: str,
    log_paths: List[str],
    role: str,
    server_name: str,
) -> dict:
    """Collect startup data without making optional telemetry fail the test."""
    try:
        observation = fetch_startup_observation(server_address, log_paths, role)
    except (OSError, ValueError, http.client.HTTPException) as error:
        print_warning(
            f"Failed to collect startup metrics from {server_name} ({server_address}): {error}"
        )
        return {
            "role": role,
            "server_name": server_name,
            "error": f"{type(error).__name__}: {error}",
        }
    observation["server_name"] = server_name
    return observation


def write_startup_observations(
    test_output_dir: str, server_idx: int, observations: List[dict]
) -> None:
    """Persist observations as a CI artifact for later result upload."""
    path = os.path.join(test_output_dir, f"startup_metrics.{server_idx}.json")
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(observations, output_file, indent=2, sort_keys=True)


def read_startup_observations(test_output_dir: str, server_idx: int) -> List[dict]:
    """Read observations captured while the server was alive."""
    path = os.path.join(test_output_dir, f"startup_metrics.{server_idx}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as input_file:
        observations = json.load(input_file)
    return observations if isinstance(observations, list) else []


def add_startup_metric_values(
    new_data: dict,
    observations: List[dict],
    role: Optional[str] = None,
    expected_server_count: Optional[int] = None,
) -> None:
    """Add informational startup fields to one OpenSearch result row.

    Disaggregated serving can have several servers per role. The maximum time
    is recorded because the slowest server determines fleet readiness.
    """
    selected = [entry for entry in observations if role is None or entry.get("role") == role]
    if expected_server_count is None:
        expected_server_count = len(selected)
    if expected_server_count == 0 and not selected:
        return

    field_prefix = f"{role}_" if role else ""
    failed = [entry for entry in selected if entry.get("error")]
    new_data[f"l_{field_prefix}startup_metrics_expected_server_count"] = expected_server_count
    new_data[f"l_{field_prefix}startup_metrics_discovered_server_count"] = len(selected)
    new_data[f"l_{field_prefix}startup_metrics_failed_server_count"] = len(failed)
    new_data[f"b_{field_prefix}startup_metrics_collection_complete"] = (
        len(selected) == expected_server_count and not failed
    )

    successful = [entry for entry in selected if not entry.get("error")]
    metric_names = {name for entry in successful for name in entry.get("metrics", {})}
    for metric_name in metric_names:
        values = [entry.get("metrics", {}).get(metric_name) for entry in successful]
        numeric_values = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if numeric_values:
            new_data[f"d_{field_prefix}{metric_name}"] = max(numeric_values)

    policies = [
        policy for entry in successful for policy in entry.get("checkpoint_io_policies", [])
    ]
    for policy_field in ("requested", "selected", "effective"):
        values = sorted(
            {str(policy[policy_field]) for policy in policies if policy.get(policy_field)}
        )
        if values:
            new_data[f"s_{field_prefix}checkpoint_io_policy_{policy_field}"] = ",".join(values)
    activated = [
        policy["activated"] for policy in policies if isinstance(policy.get("activated"), bool)
    ]
    if activated:
        new_data[f"l_{field_prefix}checkpoint_io_policy_status_count"] = len(activated)
        new_data[f"l_{field_prefix}checkpoint_io_policy_activated_status_count"] = sum(activated)


def get_model_dir(model_name: str) -> str:
    """Get model directory path from model name."""
    if model_name in MODEL_PATH_DICT:
        return os.path.join(llm_models_root(), MODEL_PATH_DICT[model_name])
    return ""


def get_dataset_dir(dataset_file: Optional[str]) -> str:
    """Get dataset directory path from dataset file."""
    if not dataset_file or dataset_file == "<dataset_file>":
        return ""

    # return os.path.join(llm_models_root(), "datasets", "ShareGPT_V3_unfiltered_cleaned_split.json")
    llm_models_path = os.path.join(llm_models_root(), dataset_file)
    if os.path.exists(llm_models_path):
        return llm_models_path
    elif os.path.exists(dataset_file):
        return dataset_file
    else:
        print_info(f"Dataset file not found in {llm_models_path} and {dataset_file}")
        return ""


def to_env_dict(env_vars: str) -> Dict[str, str]:
    """Convert env vars string to dict."""
    env = {}
    for env_var in env_vars.split():
        if "=" in env_var:
            key, value = env_var.split("=", 1)
            env[key] = value
    return env


def force_num_accepted_tokens_from_env_str(env_vars: str) -> int:
    """Extract TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS from a space-separated KEY=val env-var string.

    Returns 0 when not set.

    The runtime accepts a fractional value (see get_force_num_accepted_tokens_float
    in tensorrt_llm), so parse as float first and truncate. The return value is
    uploaded as l_force_num_accepted_tokens, which is a long, so a fractional
    setting is not preserved in the record. It is reported rather than matched on:
    case identity is keyed on the test case name, so two lanes differing solely in
    the fractional part are already separate cases by name.
    """
    val = to_env_dict(env_vars).get("TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS")
    return int(float(val)) if val is not None else 0


def add_host_port_to_cmd(cmd: List[str], host: str, port: int) -> List[str]:
    """Add host and port to command."""
    return cmd + ["--host", host, "--port", str(port)]


# Ports reserved for multi-frontend servers. Module-level so a reservation is
# never garbage-collected: closing the socket would release the port and reopen
# the very race the reservation exists to close.
_RESERVED_PORT_SOCKETS: List[socket.socket] = []


def reserve_multi_frontend_port(host: str) -> int:
    """Reserve a port for a server that runs several HTTP frontends.

    trtllm-serve rejects port 0 / --report_addr when num_serve_frontends > 1:
    the extra frontends re-exec the command line verbatim, so with port 0 each
    would bind its *own* kernel-assigned port instead of sharing one, and each
    would republish its address, leaving the reader with whichever wrote last.
    The port therefore has to be chosen on this side.

    Choosing it by binding and closing would reopen exactly the window the port-0
    scheme was introduced to remove -- anything on the node could take the port
    between the probe and the server's bind. So the socket stays bound instead.
    In multi-frontend mode every frontend binds with SO_REUSEPORT (see
    launch_server), and Linux lets same-uid SO_REUSEPORT sockets share a port
    provided the *first* binder set the flag, which this one does. So the
    reservation is transparent to the server while still refusing a plain bind()
    from any unrelated process on the node.

    The socket is deliberately never listen()ed: only *listening* SO_REUSEPORT
    sockets join the kernel's accept load-balancing group, so a bound-only socket
    holds the port without ever swallowing a request.
    """
    # Mirror launch_server's family choice; a reservation in a different address
    # family than the server's bind would not share the port.
    addr_info = socket.getaddrinfo(host, 0, socket.AF_UNSPEC, socket.SOCK_STREAM)
    family = (
        socket.AF_INET6
        if addr_info and all(info[0] == socket.AF_INET6 for info in addr_info)
        else socket.AF_INET
    )
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    _RESERVED_PORT_SOCKETS.append(sock)
    print_info(f"Reserved multi-frontend port {host}:{port} (holding SO_REUSEPORT socket)")
    return port


def publish_addr_file(path: str, host: str, port: int) -> None:
    """Write "host:port" to *path* the way trtllm-serve's --report_addr does.

    Used when this side picked the port (multi-frontend), so the disagg server's
    hostname-file reader needs no special case. The write is atomic
    (temp file in the same directory, then rename) with a ".tmp" suffix: the
    reader counts only ".txt" entries, and a partial read on the shared
    filesystem these tests coordinate through would be parsed as a URL.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    # Bracket IPv6 literals so the value is a usable URL authority: readers build
    # "http://<reported>/..." from it verbatim.
    reported_host = f"[{host}]" if ":" in host else host
    fd, tmp_path = tempfile.mkstemp(dir=parent, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as addr_file:
            addr_file.write(f"{reported_host}:{port}\n")
            addr_file.flush()
            os.fsync(addr_file.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _run_benchmark_with_log(cmd: List[str], env: Dict[str, str], log_path: str) -> str:
    """Run a benchmark while streaming its combined output to an artifact log."""
    benchmark_env = env.copy()
    benchmark_env.setdefault("PYTHONUNBUFFERED", "1")
    with open(log_path, "wb") as log_file:
        result = subprocess.run(
            cmd,
            env=benchmark_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    with open(log_path, "rb") as log_file:
        raw_output = log_file.read()

    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=raw_output)

    return raw_output.decode("utf-8", errors="replace")


class ServerConfig:
    """Configurations of trtllm-server."""

    def __init__(self, server_config_data: dict, env_vars: str = ""):
        # Extract required fields
        self.concurrency = server_config_data.get("concurrency", 1)
        self.model_name = server_config_data["model_name"]
        self.model_path = ""
        self.env_vars = env_vars
        self.force_num_accepted_tokens = force_num_accepted_tokens_from_env_str(env_vars)
        self.disagg_run_type = server_config_data.get("disagg_run_type", "aggr")

        # Extract optional fields with defaults
        self.tp = server_config_data.get("tensor_parallel_size", 1)
        self.ep = server_config_data.get("moe_expert_parallel_size", 1)
        self.pp = server_config_data.get("pipeline_parallel_size", 1)
        self.cp = server_config_data.get("context_parallel_size", 1)
        self.gpus = server_config_data.get("gpus", self.tp * self.cp * self.pp)
        self.gpus_per_node = server_config_data.get("gpus_per_node", 0) or self.gpus
        self.max_num_tokens = server_config_data.get("max_num_tokens", 2048)
        self.max_batch_size = server_config_data.get("max_batch_size", 512)
        self.max_seq_len = server_config_data.get("max_seq_len", 0)
        self.disable_overlap_scheduler = server_config_data.get("disable_overlap_scheduler", False)
        self.num_postprocess_workers = server_config_data.get("num_postprocess_workers", 0)
        self.stream_interval = server_config_data.get("stream_interval", 10)
        self.print_iter_log = server_config_data.get("print_iter_log", False)
        self.attn_backend = server_config_data.get("attn_backend", "TRTLLM")
        self.enable_chunked_prefill = server_config_data.get("enable_chunked_prefill", False)
        self.enable_attention_dp = server_config_data.get("enable_attention_dp", False)
        self.trust_remote_code = server_config_data.get("trust_remote_code", False)
        self.enable_lm_head_tp_in_adp = server_config_data.get("enable_lm_head_tp_in_adp", False)
        self.backend = server_config_data.get("backend", "pytorch")
        self.extra_llm_api_config_path = server_config_data.get("extra_llm_api_config_path", "")

        # attention_dp_config
        attention_dp_config = server_config_data.get("attention_dp_config", {})
        self.attention_dp_balance = attention_dp_config.get("enable_balance", False)
        self.batching_wait_iters = attention_dp_config.get("batching_wait_iters", 0)
        self.timeout_iters = attention_dp_config.get("timeout_iters", 60)

        # moe_config
        moe_config = server_config_data.get("moe_config", {})
        self.moe_backend = moe_config.get("backend", "")
        self.moe_max_num_tokens = moe_config.get("max_num_tokens", 0)
        self.use_low_precision_moe_combine = moe_config.get("use_low_precision_moe_combine", False)
        load_balancer_config = moe_config.get("load_balancer", {})
        # load_balancer may be either an inline dict (num_slots + layer_updates_per_iter)
        # or a path string to an offline-eplb YAML that the TRT-LLM engine loads at
        # runtime. When it is a string, skip the inline attribute extraction — those
        # metrics live inside the referenced YAML and aren't scraped by perf-sanity.
        if isinstance(load_balancer_config, str):
            self.load_balancer_num_slots = 0
            self.load_balancer_layer_updates_per_iter = 0
        else:
            self.load_balancer_num_slots = load_balancer_config.get("num_slots", 0)
            self.load_balancer_layer_updates_per_iter = load_balancer_config.get(
                "layer_updates_per_iter", 0
            )

        # cuda_graph_config
        cuda_graph_config = server_config_data.get("cuda_graph_config", {})
        self.enable_cuda_graph = False
        if cuda_graph_config:
            self.enable_cuda_graph = True
            self.enable_padding = cuda_graph_config.get("enable_padding", True)
            self.cuda_graph_batch_sizes = cuda_graph_config.get("batch_sizes", [])
            self.cuda_graph_max_batch_size = cuda_graph_config.get("max_batch_size", 0)
        else:
            self.enable_padding = True
            self.cuda_graph_batch_sizes = []
            self.cuda_graph_max_batch_size = 0

        # kv_cache_config
        kv_cache_config = server_config_data.get("kv_cache_config", {})
        self.kv_cache_dtype = kv_cache_config.get("dtype", "fp8")
        self.enable_block_reuse = kv_cache_config.get("enable_block_reuse", False)
        self.free_gpu_memory_fraction = kv_cache_config.get("free_gpu_memory_fraction", 0.8)

        # cache_transceiver_config
        cache_transceiver_config = server_config_data.get("cache_transceiver_config", {})
        self.cache_transceiver_backend = cache_transceiver_config.get("backend", "")
        self.cache_transceiver_max_tokens_in_buffer = cache_transceiver_config.get(
            "max_tokens_in_buffer", 0
        )

        # Generate default name if not provided
        self.name = server_config_data.get("name", "")
        if not self.name:
            self.name = (
                f"{self.model_name}_tp{self.tp}_ep{self.ep}_pp{self.pp}_cp{self.cp}"
                f"_bs{self.max_batch_size}_attn{self.attn_backend}_moe{self.moe_backend}"
            )
            if self.cache_transceiver_backend:
                self.name += f"_spec{self.cache_transceiver_backend}"

        # speculative_config
        speculative_config = server_config_data.get("speculative_config", {})
        self.spec_decoding_type = speculative_config.get("decoding_type", "")
        # MTP user config migrated from num_nextn_predict_layers to max_draft_len.
        # Keep both DB field names populated for perf/baseline compatibility while
        # the surrounding reporting pipeline transitions to l_max_draft_len.
        self.max_draft_len = speculative_config.get(
            "max_draft_len", speculative_config.get("num_nextn_predict_layers", 0)
        )
        eagle3_value = speculative_config.get("eagle3_layers_to_capture", [])
        if isinstance(eagle3_value, int):
            self.eagle3_layers_to_capture = [eagle3_value]
        elif isinstance(eagle3_value, list):
            self.eagle3_layers_to_capture = eagle3_value
        else:
            self.eagle3_layers_to_capture = []
        self.speculative_model = speculative_config.get("speculative_model", "")
        self.eagle3_one_model = speculative_config.get("eagle3_one_model", False)

        # match_mode: "config" (default) or "scenario"
        self.match_mode = server_config_data.get("match_mode", "config")

        # Store filtered config for extra_llm_api_config
        exclude_keys = [
            "mode",
            "concurrency",
            "name",
            "model_name",
            "disagg_run_type",
            "gpus",
            "gpus_per_node",
            "match_mode",
            "client_configs",
            "backend",
            "extra_llm_api_config_path",
            "server_env_var",
        ]
        self.extra_llm_api_config_data = {
            k: v for k, v in server_config_data.items() if k not in exclude_keys
        }

        # Not a recognized field, so it rides through in extra_llm_api_config_data
        # to the engine. Read it out here too: K > 1 HTTP frontends against one
        # executor is incompatible with the port-0 launch scheme, so the launcher
        # has to know the count before it builds the command line.
        self.num_serve_frontends = self.extra_llm_api_config_data.get("num_serve_frontends", 1)

    def to_cmd(
        self, output_dir: str, numa_bind: bool = False, disagg_serving_type: str = ""
    ) -> List[str]:
        """Generate server command."""
        model_dir = get_model_dir(self.model_name)
        self.model_path = model_dir if os.path.exists(model_dir) else self.model_name
        config_filename = f"extra-llm-api-config.{self.disagg_run_type}.{self.name}.yml"
        config_path = os.path.join(output_dir, config_filename)

        numa_bind_cmd = []
        if numa_bind:
            numa_bind_cmd = ["numactl", "-m 0,1"]

        cmd = numa_bind_cmd + [
            "trtllm-serve",
            self.model_path,
            "--backend",
            self.backend,
            "--config",
            config_path,
        ]
        return cmd

    def to_env(self) -> Dict[str, str]:
        return to_env_dict(self.env_vars)

    def to_db_data(self) -> dict:
        """Convert ServerConfig to database data."""
        db_data = {
            "s_server_name": self.name,
            "s_model_name": self.model_name.lower(),
            "l_gpus": self.gpus,
            "l_tp": self.tp,
            "l_ep": self.ep,
            "l_pp": self.pp,
            "l_cp": self.cp,
            "l_gpus_per_node": self.gpus_per_node,
            "l_max_num_tokens": self.max_num_tokens,
            "l_max_batch_size": self.max_batch_size,
            "l_max_seq_len": self.max_seq_len,
            "b_disable_overlap_scheduler": self.disable_overlap_scheduler,
            "l_num_postprocess_workers": self.num_postprocess_workers,
            "l_stream_interval": self.stream_interval,
            "s_attn_backend": self.attn_backend,
            "b_enable_chunked_prefill": self.enable_chunked_prefill,
            "b_enable_attention_dp": self.enable_attention_dp,
            "b_trust_remote_code": self.trust_remote_code,
            "b_enable_lm_head_tp_in_adp": self.enable_lm_head_tp_in_adp,
            "s_serving_backend": self.backend,
            # attention_dp_config
            "b_attention_dp_balance": self.attention_dp_balance,
            "l_batching_wait_iters": self.batching_wait_iters,
            "l_timeout_iters": self.timeout_iters,
            # moe_config
            "s_moe_backend": self.moe_backend,
            "l_moe_max_num_tokens": self.moe_max_num_tokens,
            "b_use_low_precision_moe_combine": self.use_low_precision_moe_combine,
            "l_load_balancer_num_slots": self.load_balancer_num_slots,
            "l_load_balancer_layer_updates_per_iter": self.load_balancer_layer_updates_per_iter,
            # cuda_graph_config
            "b_enable_cuda_graph": self.enable_cuda_graph,
            "b_enable_padding": self.enable_padding,
            "l_cuda_graph_max_batch_size": self.cuda_graph_max_batch_size,
            "s_cuda_graph_batch_sizes": ",".join(map(str, self.cuda_graph_batch_sizes)),
            # kv_cache_config
            "s_kv_cache_dtype": self.kv_cache_dtype,
            "b_enable_block_reuse": self.enable_block_reuse,
            "d_free_gpu_memory_fraction": self.free_gpu_memory_fraction,
            # cache_transceiver_config
            "s_cache_transceiver_backend": self.cache_transceiver_backend,
            "l_cache_transceiver_max_tokens_in_buffer": self.cache_transceiver_max_tokens_in_buffer,
            # speculative_config
            "s_spec_decoding_type": self.spec_decoding_type,
            "l_num_nextn_predict_layers": self.max_draft_len,
            "s_eagle3_layers_to_capture": ",".join(map(str, self.eagle3_layers_to_capture)),
            "l_max_draft_len": self.max_draft_len,
            "l_force_num_accepted_tokens": self.force_num_accepted_tokens,
            "s_speculative_model_dir": self.speculative_model,
            "b_eagle3_one_model": self.eagle3_one_model,
            "s_server_log_link": "",
            "s_server_env_var": self.env_vars,
        }
        return db_data

    def generate_extra_llm_api_config(self) -> str:
        """Generate extra-llm-api-config.yml content."""
        config_data = dict(self.extra_llm_api_config_data)

        # Merge external AutoDeploy config if specified
        if self.extra_llm_api_config_path:
            config_path = self.extra_llm_api_config_path
            if not os.path.isabs(config_path):
                config_path = os.path.join(get_llm_root(), config_path)
            with open(config_path, "r") as f:
                external_config = yaml.safe_load(f) or {}
            # Fields in extra_llm_api_config_data (from perf YAML) take precedence
            merged = {**external_config, **config_data}
            config_data = merged

        # Handle speculative_model path conversion
        if (
            "speculative_config" in config_data
            and "speculative_model" in config_data["speculative_config"]
        ):
            spec_model = config_data["speculative_config"]["speculative_model"]
            if spec_model:
                config_data["speculative_config"]["speculative_model"] = os.path.join(
                    llm_models_root(), spec_model
                )

        # Resolve `moe_config.load_balancer` when it is a repo-relative path
        # string. The TRT-LLM engine accepts either a dict (inline) or a path
        # to an offline-eplb YAML. Absolute paths and dicts are left alone.
        moe_cfg = config_data.get("moe_config")
        if isinstance(moe_cfg, dict):
            lb = moe_cfg.get("load_balancer")
            if isinstance(lb, str) and lb and not os.path.isabs(lb):
                moe_cfg["load_balancer"] = os.path.join(get_llm_root(), lb)

        return yaml.dump(config_data, default_flow_style=False, sort_keys=False)


class AccuracyConfig:
    """Accuracy test configuration (lm_eval against the running server).

    Shape mirrors the existing top-level `accuracy:` block in disagg yamls:
        enable_accuracy_test: bool
        env_var: dict[str, str]
        tasks:
          <task_name>:
            model: local-completions | local-chat-completions
            model_args_extra: str
            extra_kwargs: dict   # forwarded to lm_eval as --<k> <v>
    """

    _ENDPOINT_MAP = {
        "local-completions": "v1/completions",
        "local-chat-completions": "v1/chat/completions",
    }

    def __init__(self, accuracy_data: dict):
        self.enable_accuracy_test = bool(accuracy_data.get("enable_accuracy_test", False))
        self.env_var = dict(accuracy_data.get("env_var") or {})
        self.tasks = dict(accuracy_data.get("tasks") or {})

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["AccuracyConfig"]:
        if not data:
            return None
        # Short-circuit when accuracy is disabled so we don't try to parse
        # legacy-shape `tasks:` (a string instead of the expected dict).
        if not bool(data.get("enable_accuracy_test", False)):
            return None
        return cls(data)

    def build_lm_eval_invocations(
        self,
        model_name: str,
        server_hostname: str,
        server_port: int,
        output_dir: str,
        server_idx: int,
    ) -> List[Tuple[List[str], str, Dict[str, str], str]]:
        """Build (cmd, log_file, env, task_name) tuples for each configured task."""
        model_path = get_model_dir(model_name)
        invocations = []
        for task_name, task_cfg in self.tasks.items():
            model_type = task_cfg.get("model", "local-completions")
            model_args_extra = task_cfg.get("model_args_extra", "")
            extra_kwargs = dict(task_cfg.get("extra_kwargs") or {})
            base_url = (
                f"http://{server_hostname}:{server_port}/"
                f"{self._ENDPOINT_MAP.get(model_type, 'v1/completions')}"
            )
            model_args = f"model={model_path},base_url={base_url},{model_args_extra}"

            acc_output_dir = os.path.join(output_dir, f"accuracy_eval_{task_name}.{server_idx}")
            log_file = os.path.join(output_dir, f"accuracy_eval_{task_name}.{server_idx}.log")
            os.makedirs(acc_output_dir, exist_ok=True)

            cmd = [
                "lm_eval",
                "--model",
                model_type,
                "--tasks",
                task_name,
                "--model_args",
                model_args,
                "--log_samples",
                "--output_path",
                acc_output_dir,
            ]

            include_path = extra_kwargs.pop("include_path", None)
            custom_config = extra_kwargs.pop("custom_config", None)
            if custom_config and not include_path:
                # Substitute LLM_MODELS_ROOT (and other env vars) in the lm_eval
                # task yaml, write to <output_dir>/lm_eval_configs/, and pass the
                # directory to --include_path. lm_eval requires a directory.
                cfg_path = (
                    custom_config
                    if os.path.isabs(custom_config)
                    else os.path.join(get_llm_root(), custom_config)
                )
                lm_eval_dir = os.path.join(output_dir, "lm_eval_configs")
                os.makedirs(lm_eval_dir, exist_ok=True)
                with open(cfg_path, "r", encoding="utf-8") as f:
                    content = f.read()
                content = content.replace("LLM_MODELS_ROOT", llm_models_root())
                out_path = os.path.join(lm_eval_dir, os.path.basename(cfg_path))
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(content)
                # Copy sibling utils.py if present (some tasks like GPQA need it)
                sibling_utils = os.path.join(os.path.dirname(cfg_path), "utils.py")
                if os.path.exists(sibling_utils):
                    shutil.copy(sibling_utils, lm_eval_dir)
                include_path = lm_eval_dir
            if include_path:
                cmd += ["--include_path", include_path]

            for k, v in extra_kwargs.items():
                if isinstance(v, bool):
                    if v:
                        cmd += [f"--{k}"]
                else:
                    cmd += [f"--{k}", str(v)]

            run_env = copy.deepcopy(os.environ)
            run_env.update({k: str(v) for k, v in self.env_var.items()})
            invocations.append((cmd, log_file, run_env, task_name))
        return invocations

    def run(
        self,
        model_name: str,
        server_hostname: str,
        server_port: int,
        output_dir: str,
        server_idx: int,
    ) -> None:
        """Run all configured accuracy tasks against the live server."""
        for cmd, log_file, run_env, task_name in self.build_lm_eval_invocations(
            model_name, server_hostname, server_port, output_dir, server_idx
        ):
            print_info(f"[Accuracy] Running {task_name}, output: {log_file}")
            with open(log_file, "w") as lf:
                ret = subprocess.run(cmd, env=run_env, stdout=lf, stderr=subprocess.STDOUT)
            print_info(f"[Accuracy] {task_name} done, exit_code={ret.returncode}")


class ClientConfig:
    """Configurations of benchmark client."""

    def __init__(
        self,
        client_config_data: dict,
        model_name: str,
        env_vars: str = "",
        spec_decoding: bool = False,
        warmup: bool = False,
    ):
        self.model_name = model_name
        self.concurrency = client_config_data.get("concurrency", 1)
        self.iterations = client_config_data.get("iterations", 1)
        # BOLT knob: extend the measured serving window (num_requests =
        # concurrency * iterations) without touching the shared perf-sanity
        # config, so a longer run dilutes startup in the profile. Set by the
        # BOLT profile-gen job via EXTRA_CONTAINER_EXPORTS; unset/absent
        # (every normal build) is a no-op.
        try:
            _bolt_iter_mult = int(os.environ.get("BOLT_ITER_MULT", "1") or "1")
        except ValueError:
            _bolt_iter_mult = 1
        if _bolt_iter_mult > 1:
            self.iterations *= _bolt_iter_mult
        self.isl = client_config_data.get("isl", 1024)
        self.osl = client_config_data.get("osl", 1024)
        self.random_range_ratio = client_config_data.get("random_range_ratio", 0.0)
        self.backend = client_config_data.get("backend", "openai")
        self.use_chat_template = client_config_data.get("use_chat_template", False)
        self.streaming = client_config_data.get("streaming", True)
        self.trust_remote_code = client_config_data.get("trust_remote_code", True)
        self.model_path = ""
        self.dataset_file = client_config_data.get("dataset_file", "")
        self.use_nv_sa_benchmark = client_config_data.get("use_nv_sa_benchmark", False)
        # Which load generator drives the lane. "" selects the built-in
        # benchmark_serving client; "agentx" selects the trace-replay client in
        # agentx_client.py. Reported only -- see the s_benchmark_client note in
        # to_db_data for why it is not a match key.
        self.benchmark_client = client_config_data.get("benchmark_client", "")
        run_agentx_mode = self.benchmark_client == AGENTX_BENCHMARK_CLIENT
        self.warmup = warmup and not (run_agentx_mode or self.use_nv_sa_benchmark)
        self.env_vars = env_vars
        # spec_decoding flag is retained for DB matching (b_eos column). --ignore-eos
        # is now always passed; output-length stability with spec decoding comes from
        # TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS (set per-yaml).
        self.spec_decoding = spec_decoding

        # Accuracy testing (lm_eval after benchmark). only_run_accuracy is silently
        # ignored when no accuracy_config is present or enable_accuracy_test=False.
        self.accuracy_config = AccuracyConfig.from_dict(client_config_data.get("accuracy_config"))
        only_run_accuracy = bool(client_config_data.get("only_run_accuracy", False))
        self.only_run_accuracy = only_run_accuracy and bool(
            self.accuracy_config and self.accuracy_config.enable_accuracy_test
        )

        # Generate default name if not provided
        self.name = client_config_data.get("name", "")
        if not self.name:
            self.name = f"con{self.concurrency}_iter{self.iterations}_isl{self.isl}_osl{self.osl}"

    def to_cmd(self) -> List[str]:
        """Generate benchmark command."""
        model_dir = get_model_dir(self.model_name)
        self.model_path = model_dir if os.path.exists(model_dir) else self.model_name

        if self.benchmark_client == AGENTX_BENCHMARK_CLIENT:
            return self._to_agentx_cmd()
        elif self.use_nv_sa_benchmark:
            return self._to_sa_benchmark_cmd()
        else:
            return self._to_default_benchmark_cmd()

    def _to_agentx_cmd(self) -> List[str]:
        """Generate AgentX benchmark command (aiperf trace replay).

        AgentX replays a recorded conversation corpus for a fixed wall-clock
        duration, so it takes neither a prompt count nor ISL/OSL; every other
        knob comes from AGENTX_* env vars set in the lane's client_env_var. The
        dataset name is passed through verbatim rather than resolved to a path
        because it names an aiperf loader (which fetches from HF), not a file --
        so get_dataset_dir must not be applied to it.
        """
        if not self.dataset_file:
            raise ValueError(
                f"Client {self.name} uses benchmark_client={AGENTX_BENCHMARK_CLIENT} but sets no "
                "dataset_file; the agentx scenario has no default corpus."
            )
        return [
            "python",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentx_client.py"),
            "--model",
            self.model_path,
            "--concurrency",
            str(self.concurrency),
            "--dataset",
            self.dataset_file,
        ]

    def _to_sa_benchmark_cmd(self) -> List[str]:
        """Generate SA benchmark command (bench_serving repo)."""
        bench_script = ensure_bench_serving_repo()
        benchmark_cmd = [
            "python",
            bench_script,
            "--model",
            self.model_path,
            "--dataset-name",
            "random",
            "--num-prompts",
            str(self.concurrency * self.iterations),
            "--max-concurrency",
            str(self.concurrency),
            "--random-input-len",
            str(self.isl),
            "--random-output-len",
            str(self.osl),
            "--random-range-ratio",
            str(self.random_range_ratio),
            "--save-result",
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--ignore-eos",
        ]
        if self.backend:
            benchmark_cmd.extend(["--backend", self.backend])
        if self.trust_remote_code:
            benchmark_cmd.append("--trust-remote-code")
        if self.use_chat_template:
            benchmark_cmd.append("--use-chat-template")
        # Note: bench_serving has no --non-streaming flag; streaming is backend-determined
        return benchmark_cmd

    def _to_default_benchmark_cmd(self) -> List[str]:
        """Generate default benchmark command (tensorrt_llm benchmark_serving)."""
        dataset_path = get_dataset_dir(self.dataset_file)
        benchmark_cmd = [
            "python",
            "-m",
            "tensorrt_llm.serve.scripts.benchmark_serving",
            "--model",
            self.model_path,
            "--tokenizer",
            self.model_path,
            "--num-prompts",
            str(self.concurrency * self.iterations),
            "--max-concurrency",
            str(self.concurrency),
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--ignore-eos",
        ]
        if not self.warmup:
            benchmark_cmd.append("--no-test-input")
        if dataset_path:
            benchmark_cmd.append("--dataset-name")
            benchmark_cmd.append("trtllm_custom")
            benchmark_cmd.append("--dataset-path")
            benchmark_cmd.append(dataset_path)
            print_info(f"Dataset: {dataset_path} exists. Use trtllm_custom dataset for benchmark.")
        else:
            benchmark_cmd.append("--dataset-name")
            benchmark_cmd.append("random")
            benchmark_cmd.append("--random-ids")
            benchmark_cmd.append("--tokenize-on-client")
            benchmark_cmd.append("--random-input-len")
            benchmark_cmd.append(str(self.isl))
            benchmark_cmd.append("--random-output-len")
            benchmark_cmd.append(str(self.osl))
            benchmark_cmd.append("--random-range-ratio")
            benchmark_cmd.append(str(self.random_range_ratio))
            print_info(
                f"Dataset: {dataset_path} is not provided or does not exist. "
                f"Use random dataset (random_range_ratio={self.random_range_ratio}) for benchmark."
            )
        if self.backend:
            benchmark_cmd.append("--backend")
            benchmark_cmd.append(self.backend)
        if self.use_chat_template:
            benchmark_cmd.append("--use-chat-template")
        if not self.streaming:
            benchmark_cmd.append("--non-streaming")
        if self.trust_remote_code:
            benchmark_cmd.append("--trust-remote-code")
        return benchmark_cmd

    def to_env(self) -> Dict[str, str]:
        return to_env_dict(self.env_vars)

    def to_db_data(self) -> dict:
        """Convert ClientConfig to database data."""
        # b_eos retained for baseline-matching continuity. spec-decoding runs are now
        # further differentiated from non-spec-decoding runs via the
        # l_force_num_accepted_tokens match key on ServerConfig.
        db_data = {
            "s_client_name": self.name,
            "l_concurrency": self.concurrency,
            "l_iterations": self.iterations,
            "l_isl": self.isl,
            "l_osl": self.osl,
            "d_random_range_ratio": self.random_range_ratio,
            "s_dataset_file": self.dataset_file,
            "s_backend": self.backend,
            "b_use_chat_template": self.use_chat_template,
            "b_streaming": self.streaming,
            "b_trust_remote_code": self.trust_remote_code,
            "b_use_nv_sa_benchmark": self.use_nv_sa_benchmark,
            "b_warmup": self.warmup,
            # Reported, not matched. Case identity is keyed on s_test_case_name
            # (plus GPU type, runtime, branch), and a disagg case name embeds its
            # config stem, so an agentx lane already forms its own population by
            # name. Uploaded as "" (not "default") for the built-in client so the
            # column reads consistently against records written before this field.
            "s_benchmark_client": self.benchmark_client,
            "b_eos": self.spec_decoding,
            "s_client_log_link": "",
            "s_client_env_vars": self.env_vars,
        }
        if self.backend:
            db_data["s_backend"] = self.backend
        if self.use_chat_template:
            db_data["b_use_chat_template"] = self.use_chat_template
        return db_data


class DisaggConfig:
    """Configurations for disaggregated server."""

    def __init__(
        self,
        name: str,
        disagg_serving_type: str,
        hostname: str,
        numa_bind: bool,
        timeout: int,
        benchmark_mode: str,
        model_name: str,
        hardware: dict,
        server_env_var: str,
        internal_request_auth_key: str | None = None,
        router_config: dict | None = None,
        ctx_router_config: dict | None = None,
        gen_router_config: dict | None = None,
        server_config_extra: dict | None = None,
    ):
        self.name = name
        self.disagg_serving_type = disagg_serving_type
        self.hostname = hostname
        self.numa_bind = numa_bind
        self.timeout = timeout
        self.benchmark_mode = benchmark_mode
        self.model_name = model_name
        self.hardware = hardware
        self.server_env_var = server_env_var
        self.internal_request_auth_key = internal_request_auth_key
        self.router_config = router_config
        self.ctx_router_config = ctx_router_config
        self.gen_router_config = gen_router_config
        self.server_config_extra = server_config_extra
        self.num_ctx_servers = hardware.get("num_ctx_servers", 0)
        self.num_gen_servers = hardware.get("num_gen_servers", 0)


class AggrTestCmds(NamedTuple):
    """Commands for aggregated server perf sanity tests."""

    server_cmds: List[List[str]]
    client_cmds: Dict[int, List[List[str]]]
    timeout: int
    output_dir: str
    test_output_dir: str
    client_configs: Dict[int, List["ClientConfig"]] = {}
    model_name: str = ""
    server_configs: List["ServerConfig"] = []

    def get_server_logs(self, server_idx) -> List[str]:
        server_file_path = os.path.join(self.test_output_dir, f"trtllm-serve.{server_idx}.log")
        benchmark_logs = sorted(
            glob.glob(os.path.join(self.test_output_dir, f"trtllm-benchmark.{server_idx}.*.log"))
        )
        return [server_file_path, *benchmark_logs]

    def run_cmd(self, server_idx: int) -> List[str]:
        """Run all clients for a server and return outputs.

        For each client: starts benchmark unless only_run_accuracy=True; runs
        accuracy_config (lm_eval) afterward when configured. Empty string is
        appended to outputs for only_run_accuracy clients to keep client_idx
        aligned with self.client_cmds[server_idx].
        """
        outputs = []
        server_proc = None
        server_cmd = self.server_cmds[server_idx]
        client_configs = self.client_configs.get(server_idx, [])

        try:
            server_hostname = "localhost"
            # port 0 + --report_addr: let the server bind a kernel-assigned
            # port and tell us which one, instead of reserving one here and
            # racing whoever takes it before the server binds.
            server_addr_path = os.path.join(self.test_output_dir, f"trtllm-serve.{server_idx}.addr")
            if os.path.exists(server_addr_path):
                os.remove(server_addr_path)
            server_cmd_with_port = add_host_port_to_cmd(server_cmd, server_hostname, 0) + [
                "--report_addr",
                server_addr_path,
            ]

            print_info(f"Starting server. cmd is {server_cmd_with_port}")
            server_file_path = os.path.join(self.test_output_dir, f"trtllm-serve.{server_idx}.log")
            server_env = copy.deepcopy(os.environ)
            if server_idx < len(self.server_configs):
                server_env.update(self.server_configs[server_idx].to_env())
            with open(server_file_path, "w") as server_ctx:
                server_proc = subprocess.Popen(
                    server_cmd_with_port,
                    env=server_env,
                    stdout=server_ctx,
                    stderr=subprocess.STDOUT,
                )
                _, server_port = wait_for_reported_addr(server_addr_path, self.timeout, server_proc)

                wait_for_endpoint_ready(
                    f"http://{server_hostname}:{server_port}/health",
                    timeout=min(
                        self.timeout, server_ready_timeout(AGG_SERVER_READY_TIMEOUT, "AGG")
                    ),
                    check_files=[server_file_path],
                    server_proc=server_proc,
                )
                observation = collect_startup_observation(
                    f"{server_hostname}:{server_port}",
                    [server_file_path],
                    "aggregate",
                    f"aggregate_{server_idx}",
                )
                write_startup_observations(self.test_output_dir, server_idx, [observation])

            # Run all clients for this server
            for client_idx, client_cmd in enumerate(self.client_cmds[server_idx]):
                client_config = (
                    client_configs[client_idx] if client_idx < len(client_configs) else None
                )
                only_run_accuracy = bool(client_config and client_config.only_run_accuracy)

                if not only_run_accuracy:
                    client_file_path = os.path.join(
                        self.test_output_dir, f"trtllm-benchmark.{server_idx}.{client_idx}.log"
                    )
                    client_cmd_with_port = add_host_port_to_cmd(
                        client_cmd, server_hostname, server_port
                    )
                    print_info(f"Starting client. cmd is {client_cmd_with_port}")

                    client_env = copy.deepcopy(os.environ)
                    if client_config:
                        client_env.update(client_config.to_env())
                    output = _run_benchmark_with_log(
                        client_cmd_with_port,
                        client_env,
                        client_file_path,
                    )
                    outputs.append(output)
                else:
                    print_info(
                        f"Skipping perf benchmark for client {client_idx}: only_run_accuracy=True"
                    )
                    outputs.append("")

                if (
                    client_config
                    and client_config.accuracy_config
                    and client_config.accuracy_config.enable_accuracy_test
                ):
                    client_config.accuracy_config.run(
                        model_name=self.model_name or client_config.model_name,
                        server_hostname=server_hostname,
                        server_port=server_port,
                        output_dir=self.test_output_dir,
                        server_idx=server_idx,
                    )

        finally:
            if server_proc:
                server_proc.terminate()
                server_proc.wait()

        return outputs

    def get_cmd_str(self, server_idx: int) -> List[str]:
        return ["aggr_server tests, please check config files"]


class DisaggTestCmds(NamedTuple):
    """Commands for multi-node disaggregated server perf sanity tests."""

    server_cmds: List[Tuple[List[str], List[str], List[str]]]
    client_cmds: Dict[int, List[List[str]]]
    timeout: int
    hostname: str
    disagg_serving_type: str
    num_ctx_servers: int
    num_gen_servers: int
    output_dir: str
    test_output_dir: str
    model_name: str = ""
    internal_request_auth_key: str = ""
    client_configs: Dict[int, List["ClientConfig"]] = {}
    # Per-server-index ServerConfig triples (ctx_config, gen_config, disagg_config).
    # Used by run_cmd() to merge per-config env vars into the appropriate
    # subprocess env based on this rank's disagg_serving_type. For multi-node
    # disagg, only rank-0 pytest goes through this path; multi-rank workers
    # receive env via SLURM env propagation set up by submit.py.
    server_configs: List[Tuple["ServerConfig", "ServerConfig", "DisaggConfig"]] = []
    # Disagg-server-level keys, named as in bench-trtllm-disagg. A generic
    # router applies to both roles; a role-specific one overrides it.
    router_config: Optional[dict] = None
    ctx_router_config: Optional[dict] = None
    gen_router_config: Optional[dict] = None
    server_config_extra: Optional[dict] = None

    def _hostnames_dir(self, server_idx: int) -> str:
        """Directory the disagg tasks exchange bound addresses through.

        Scoped by SLURM job id so a rerun never reads the previous run's files:
        test_output_dir is derived from the test case name alone and is created
        with exist_ok=True, so it is reused across runs. The step id is
        deliberately excluded -- each role runs as a separate srun step within
        one job, and they must all agree on this path.
        """
        run_id = os.environ.get("SLURM_JOB_ID", "local")
        return os.path.join(self.test_output_dir, f"hostnames-{run_id}-{server_idx}")

    def _hostname_file(self, server_idx: int) -> str:
        """Path this task's server reports its bound address to."""
        hostnames_dir = self._hostnames_dir(server_idx)
        os.makedirs(hostnames_dir, exist_ok=True)
        return os.path.join(hostnames_dir, f"{self.disagg_serving_type}.txt")

    def _generate_disagg_server_config(self, server_idx: int) -> str:
        """Generate disagg server config from hostname files."""
        print_info(f"Generating disagg server config for server index {server_idx}")
        hostnames_folder = self._hostnames_dir(server_idx)
        expected_count = self.num_ctx_servers + self.num_gen_servers
        start_time = time.time()
        hostnames = []

        while True:
            elapsed_time = time.time() - start_time
            print_info(
                f"Waiting for hostnames in {hostnames_folder}, "
                f"elapsed time: {elapsed_time}s, current: {len(hostnames)}, "
                f"expected: {expected_count}"
            )
            if elapsed_time > self.timeout:
                raise RuntimeError(f"Time out. Hostnames files are not ready after {self.timeout}s")
            time.sleep(10)
            if not os.path.exists(hostnames_folder):
                continue
            # Only completed files: trtllm-serve publishes its address by
            # renaming a "<name>.<rand>.tmp" sibling into place, and counting
            # those transient entries would both inflate the count and get
            # parsed as a CTX/GEN url below.
            hostnames = [f for f in os.listdir(hostnames_folder) if f.endswith(".txt")]
            if len(hostnames) >= expected_count:
                break

        print_info(f"All hostnames found in {hostnames_folder} after elapsed time: {elapsed_time}s")

        # Read ctx and gen hostnames
        ctx_hostnames = []
        gen_hostnames = []
        for hostname_file in hostnames:
            hostname_file_path = os.path.join(hostnames_folder, hostname_file)
            with open(hostname_file_path, "r") as f:
                hostname_port = f.read().strip()
            if hostname_file.startswith("CTX"):
                ctx_hostnames.append(hostname_port)
            elif hostname_file.startswith("GEN"):
                gen_hostnames.append(hostname_port)

        # port 0: the disagg server binds a kernel-assigned port and reports it
        # back via --report_addr, so there is no window between choosing a port
        # here and the server binding it.
        server_config = {
            "hostname": self.hostname,
            "port": 0,
            "backend": "pytorch",
            "internal_request_auth_key": self.internal_request_auth_key,
            "context_servers": {
                "num_instances": self.num_ctx_servers,
                "urls": ctx_hostnames,
            },
            "generation_servers": {
                "num_instances": self.num_gen_servers,
                "urls": gen_hostnames,
            },
        }
        # Router selection, mirroring bench-trtllm-disagg's submit.py: a generic
        # router applies to both roles and a role-specific one overrides it for
        # that role, e.g. ctx_router_config={"type": "conversation"} puts a
        # conversation router only on the context servers. Deep-copied because
        # trtllm-serve pops keys out of this dict while parsing it.
        if self.router_config:
            server_config["context_servers"]["router"] = copy.deepcopy(self.router_config)
            server_config["generation_servers"]["router"] = copy.deepcopy(self.router_config)
        if self.ctx_router_config:
            server_config["context_servers"]["router"] = copy.deepcopy(self.ctx_router_config)
        if self.gen_router_config:
            server_config["generation_servers"]["router"] = copy.deepcopy(self.gen_router_config)

        if self.server_config_extra:
            # Merged last, as bench-trtllm-disagg does, so it wins over
            # everything above. The reserved keys are the exception: the harness
            # owns them, not the config file. port must stay 0 so the server
            # binds a kernel-assigned port and reports it via --report_addr, and
            # the url lists are discovered from the hostname files above.
            # Overriding either surfaces as a hang or a benchmark against the
            # wrong endpoint, a long way from the cause, so reject it here.
            reserved = {
                "port",
                "hostname",
                "internal_request_auth_key",
                "context_servers",
                "generation_servers",
            }
            clobbered = sorted(reserved & set(self.server_config_extra))
            if clobbered:
                raise RuntimeError(
                    f"server_config_extra may not override harness-owned keys {clobbered}: "
                    "the port is kernel-assigned and reported back via --report_addr, and "
                    "the server urls are discovered at runtime."
                )
            # Distinct from the above: these keys override nothing, but they put
            # trtllm-serve into fleet mode, which it refuses to combine with the
            # port-0 + --report_addr discovery this harness depends on (see
            # tensorrt_llm/commands/serve.py, "single self-contained
            # disaggregated server"). A fleet hands one port to N SO_REUSEPORT
            # workers; with port 0 each would get a *different* kernel-assigned
            # port, so the reported address would serve 1/N of requests.
            # Rejected here so the cause is named at config time instead of
            # surfacing ~30s later as "DISAGG_SERVER server exited unexpectedly
            # with code 2", which points nowhere near this key.
            fleet_keys = sorted(
                {"num_workers", "disagg_coordinator_url"} & set(self.server_config_extra)
            )
            if fleet_keys and (
                self.server_config_extra.get("num_workers", 1) > 1
                or self.server_config_extra.get("disagg_coordinator_url")
            ):
                raise RuntimeError(
                    f"server_config_extra sets {fleet_keys}, which selects a disaggregated "
                    "server fleet. perf-sanity binds port 0 and discovers the address via "
                    "--report_addr, and trtllm-serve rejects that combination. Remove the "
                    "key, or teach the harness to bind a fixed port first."
                )
            server_config.update(copy.deepcopy(self.server_config_extra))

        config_path = os.path.join(self.test_output_dir, f"server_config.{server_idx}.yaml")
        with open(config_path, "w") as f:
            yaml.dump(server_config, f)
        print_info(f"Server config file {config_path} generated")
        return config_path

    def _disagg_server_addr_file(self, server_idx: int) -> str:
        """Path the disagg server reports its bound address to."""
        return os.path.join(self._hostnames_dir(server_idx), f"DISAGG_SERVER.{server_idx}.addr")

    def _get_disagg_server_hostname_and_port(self, server_idx: int) -> Tuple[str, int]:
        """Wait for the disagg server to report the address it bound.

        The config carries port 0, so the address is only known once the server
        has bound; reading it from the config would yield 0.
        """
        addr_path = self._disagg_server_addr_file(server_idx)
        print_info(f"Waiting for disagg server address file {addr_path}")
        return wait_for_reported_addr(addr_path, self.timeout)

    def _collect_worker_startup_observations(self, server_idx: int) -> List[dict]:
        """Collect startup data from every ready context/generation server."""
        observations = []
        worker_requests = []
        hostnames_dir = self._hostnames_dir(server_idx)
        try:
            hostname_files = sorted(os.listdir(hostnames_dir))
        except OSError as error:
            print_warning(
                f"Failed to discover workers for startup metrics in {hostnames_dir}: {error}"
            )
            return observations

        for filename in hostname_files:
            if not filename.endswith(".txt"):
                continue
            worker_name = os.path.splitext(filename)[0]
            if worker_name.startswith("CTX"):
                role = "ctx"
            elif worker_name.startswith("GEN"):
                role = "gen"
            else:
                continue
            try:
                with open(
                    os.path.join(hostnames_dir, filename), "r", encoding="utf-8"
                ) as hostname_file:
                    server_address = hostname_file.read().strip()
            except OSError as error:
                print_warning(f"Failed to read startup metrics address for {worker_name}: {error}")
                observations.append(
                    {
                        "role": role,
                        "server_name": worker_name,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            log_paths = [
                os.path.join(
                    self.test_output_dir,
                    f"trtllm-serve.{worker_name}.{server_idx}.log",
                )
            ]
            worker_requests.append((server_address, log_paths, role, worker_name))

        # A missing or wedged optional endpoint must not serially add one full
        # request timeout per worker to an otherwise healthy benchmark.
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(collect_startup_observation, *request)
                for request in worker_requests
            ]
            observations.extend(future.result() for future in futures)
        return observations

    def wait_for_benchmark_ready(
        self,
        benchmark_status_file: str,
        server_proc: subprocess.Popen | None = None,
        server_log: str | None = None,
    ):
        """Wait for benchmark to complete, failing fast if our server dies.

        The liveness check is event-driven (process exit), not a timeout: a
        ctx/gen/disagg server that dies here raises within one loop iteration
        with its log tail in the CI log, and the rank exits nonzero. Teardown
        of the rest of the stage then follows from the launcher
        (``srun --kill-on-bad-exit=1`` kills this rank's step) plus the
        benchmark rank's bounded ready-wait failing fast on the dead endpoint
        -- instead of every rank sitting in this loop for the full timeout.

        The benchmark-done check runs FIRST so a server exiting just after a
        completed benchmark cannot fail an otherwise-passing test.
        """
        start_time = time.time()
        while True:
            if os.path.exists(benchmark_status_file):
                print_info(
                    f"Benchmark status file found, terminating server {self.disagg_serving_type}"
                )
                break
            fail_if_proc_died(
                server_proc,
                f"{self.disagg_serving_type} server",
                [server_log] if server_log else None,
            )
            elapsed_time = time.time() - start_time
            print_info(f"Waiting for benchmark status file, elapsed time: {elapsed_time}s")
            if elapsed_time > self.timeout:
                raise RuntimeError(
                    f"Timeout waiting for benchmark status file after {self.timeout}s"
                )
            time.sleep(10)

    def wait_for_gen_log_sentinels(
        self,
        timeout: float = GEN_LOG_SENTINEL_TIMEOUT,
        poll_interval: float = 2.0,
    ) -> bool:
        """Block until every gen worker signals that its log is fully written.

        Each gen worker's srun in slurm_launch_draft.sh redirects all of its
        ranks' stdout to gen_server_{i}.log via `&>` and touches
        gen_server_{i}.done only after that srun is reaped (fd closed, log
        flushed). The benchmark writes benchmark_status *before* calling this,
        which is what lets the gen srun exit — so this is not circular.

        Returns True once all sentinels exist, or False if the dedicated
        sentinel timeout is reached first. On False the caller falls back to
        parsing the current log contents. The bounded wait prevents a stuck
        multi-node srun from consuming the whole-test timeout and triggering
        Slurm's kill-on-bad-exit cascade (nvbugs 6487036 / 6487040 / 6487038).
        """
        sentinels = [
            os.path.join(self.test_output_dir, f"gen_server_{i}.done")
            for i in range(self.num_gen_servers)
        ]
        start_time = time.monotonic()
        while True:
            missing = [p for p in sentinels if not os.path.exists(p)]
            if not missing:
                print_info("All gen worker log sentinels present; log flush complete.")
                return True
            elapsed_time = time.monotonic() - start_time
            if elapsed_time > timeout:
                print_info(
                    f"Timeout ({timeout}s) waiting for gen worker log "
                    f"sentinels {missing}; parsing current log contents."
                )
                return False
            print_info(
                f"Waiting for gen worker log sentinels {missing}, elapsed time: {elapsed_time:.0f}s"
            )
            time.sleep(poll_interval)

    def _append_gen_worker_device_step_time(
        self,
        pending_device_step_time: List[dict],
        outputs: List[str],
    ) -> None:
        """Wait for GEN log flush, then append each pending client's metrics.

        A sentinel timeout is a bounded teardown fallback, not a reason to
        discard metrics that are already present in the GEN logs. If the
        fallback parse finds no usable metric, check_test_failure still fails
        the gen_only run before results are uploaded.

        Five lines are written, one statistic each -- see
        GEN_ONLY_PERF_METRIC_LOG_QUERIES for why they must not share a leading
        word. The mean keeps its original wording and 2 decimals so existing
        log readers and dashboards are unaffected; the four new lines use 4
        decimals because the stdev of a healthy run is O(0.1 ms) and would
        round to two significant figures away at 2.
        """
        if not pending_device_step_time:
            return

        self.wait_for_gen_log_sentinels()
        for record in pending_device_step_time:
            stats = parse_gen_worker_device_step_time(
                self.test_output_dir,
                self.num_gen_servers,
                start_offsets=record["start_offsets"],
            )
            if stats is None:
                continue
            summary_lines = "\n".join(
                [
                    f"Average Per Iter Device Step Time (ms): {stats.mean:.2f}",
                    f"Median Per Iter Device Step Time (ms): {stats.median:.4f}",
                    f"Stdev Per Iter Device Step Time (ms): {stats.std:.4f}",
                    f"P75 Per Iter Device Step Time (ms): {stats.p75:.4f}",
                    f"P99 Per Iter Device Step Time (ms): {stats.p99:.4f}",
                ]
            )
            with open(record["benchmark_file_path"], "a") as benchmark_ctx:
                benchmark_ctx.write(f"\n{summary_lines}\n")
            idx = record["output_index"]
            outputs[idx] = f"{outputs[idx]}\n{summary_lines}\n"

    def get_server_logs(self, server_idx: int) -> List[str]:
        server_logs = []
        for i in range(self.num_ctx_servers):
            server_logs.append(
                os.path.join(self.test_output_dir, f"trtllm-serve.CTX_{i}.{server_idx}.log")
            )
            server_logs.append(os.path.join(self.test_output_dir, f"ctx_server_{i}.log"))
        for i in range(self.num_gen_servers):
            server_logs.append(
                os.path.join(self.test_output_dir, f"trtllm-serve.GEN_{i}.{server_idx}.log")
            )
            server_logs.append(os.path.join(self.test_output_dir, f"gen_server_{i}.log"))
        server_logs.append(
            os.path.join(self.test_output_dir, f"trtllm-serve.DISAGG_SERVER.{server_idx}.log")
        )
        server_logs.append(os.path.join(self.test_output_dir, "disagg_server.log"))
        server_logs.extend(
            sorted(
                glob.glob(
                    os.path.join(self.test_output_dir, f"trtllm-benchmark.{server_idx}.*.log")
                )
            )
        )
        return server_logs

    @staticmethod
    def _wait_for_config_file(config_path: str, timeout: int = 600) -> None:
        """Wait for a config file to be written by the primary (_0) worker."""
        start_time = time.time()
        while not os.path.exists(config_path):
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise RuntimeError(
                    f"Timed out waiting for config file {config_path} after {timeout}s"
                )
            print_info(f"Waiting for config file {config_path}, elapsed: {elapsed:.0f}s")
            time.sleep(1)

    def run_cmd(self, server_idx: int) -> List[str]:
        """Run commands for a server and return outputs."""
        outputs = []
        benchmark_status_file = os.path.join(
            self.test_output_dir, f"benchmark_status.{server_idx}.txt"
        )
        ctx_cmd, gen_cmd, disagg_cmd = self.server_cmds[server_idx]
        configs_for_idx = (
            self.server_configs[server_idx] if server_idx < len(self.server_configs) else None
        )
        if "CTX" in self.disagg_serving_type or "GEN" in self.disagg_serving_type:
            hostname_file = self._hostname_file(server_idx)
            is_ctx = "CTX" in self.disagg_serving_type
            server_cmd = ctx_cmd if is_ctx else gen_cmd

            # Non-primary workers wait for _0 worker to write the config file
            if self.disagg_serving_type not in ("CTX_0", "GEN_0"):
                config_idx = server_cmd.index("--config") + 1
                self._wait_for_config_file(server_cmd[config_idx])

            worker_cfg = None
            if configs_for_idx is not None:
                ctx_cfg, gen_cfg, _ = configs_for_idx
                worker_cfg = ctx_cfg if is_ctx else gen_cfg
            num_frontends = getattr(worker_cfg, "num_serve_frontends", 1) or 1

            if num_frontends > 1:
                # trtllm-serve refuses port 0 / --report_addr with several
                # frontends (each would bind a different port), so reserve the
                # port here and publish it on the worker's behalf; the disagg
                # server's hostname-file reader is unchanged.
                #
                # Publishing before the server is up matches the semantics this
                # replaces rather than loosening them: launch_server publishes at
                # *bind* time, well before it constructs the engine, so a reader
                # has always been able to see the address of a worker that is
                # still loading weights. The disagg server's readiness wait is
                # what covers that, and it is untouched here.
                worker_port = reserve_multi_frontend_port(self.hostname)
                server_cmd = add_host_port_to_cmd(server_cmd, self.hostname, worker_port)
                publish_addr_file(hostname_file, self.hostname, worker_port)
            else:
                # port 0 + --report_addr: the worker binds a kernel-assigned port
                # and publishes host:port itself, so no port is reserved here and
                # left unbound while anything on the node could take it. The disagg
                # server reads these files to build its config, exactly as before.
                server_cmd = add_host_port_to_cmd(server_cmd, self.hostname, 0) + [
                    "--report_addr",
                    hostname_file,
                ]
            try:
                print_info(
                    f"Starting server. disagg_serving_type: {self.disagg_serving_type} cmd is {server_cmd}"
                )
                server_file_path = os.path.join(
                    self.test_output_dir,
                    f"trtllm-serve.{self.disagg_serving_type}.{server_idx}.log",
                )
                worker_env = copy.deepcopy(os.environ)
                if worker_cfg is not None:
                    worker_env.update(worker_cfg.to_env())
                with open(server_file_path, "w") as server_ctx:
                    server_proc = subprocess.Popen(
                        server_cmd,
                        env=worker_env,
                        stdout=server_ctx,
                        stderr=subprocess.STDOUT,
                    )
                    self.wait_for_benchmark_ready(
                        benchmark_status_file,
                        server_proc=server_proc,
                        server_log=server_file_path,
                    )
            finally:
                print_info(f"Server {self.disagg_serving_type} stopped")
                server_proc.terminate()
                server_proc.wait()

        elif self.disagg_serving_type == "DISAGG_SERVER":
            try:
                # _hostnames_dir is scoped by job, so a new job never sees an
                # older one's files, but a retry within the same job and the
                # same server_idx would. Drop the previous attempt's address
                # first, or the BENCHMARK task connects to a dead port. This
                # task owns the file exclusively, so removing it here is safe.
                disagg_addr_path = self._disagg_server_addr_file(server_idx)
                if os.path.exists(disagg_addr_path):
                    os.remove(disagg_addr_path)
                self._generate_disagg_server_config(server_idx)
                # The config carries port 0; publish the resolved address so
                # the BENCHMARK task can find the server.
                disagg_cmd = disagg_cmd + ["--report_addr", disagg_addr_path]
                print_info(f"Starting disagg server. cmd is {disagg_cmd}")
                disagg_server_file_path = os.path.join(
                    self.test_output_dir,
                    f"trtllm-serve.{self.disagg_serving_type}.{server_idx}.log",
                )
                disagg_env = copy.deepcopy(os.environ)
                if configs_for_idx is not None:
                    _, _, disagg_cfg = configs_for_idx
                    disagg_env.update(to_env_dict(disagg_cfg.server_env_var))
                with open(disagg_server_file_path, "w") as disagg_server_ctx:
                    disagg_server_proc = subprocess.Popen(
                        disagg_cmd,
                        env=disagg_env,
                        stdout=disagg_server_ctx,
                        stderr=subprocess.STDOUT,
                    )
                    self.wait_for_benchmark_ready(
                        benchmark_status_file,
                        server_proc=disagg_server_proc,
                        server_log=disagg_server_file_path,
                    )
            finally:
                print_info(f"Disagg server {self.disagg_serving_type} stopped")
                disagg_server_proc.terminate()
                disagg_server_proc.wait()

        elif self.disagg_serving_type == "BENCHMARK":
            # Perf-benchmark clients whose gen-worker device step time must be
            # parsed after the gen-log flush wait. The parse is deferred out of
            # the client loop because gen_server_*.log keeps being written until
            # the gen srun exits, and the gen srun only exits after
            # benchmark_status is written in the finally below. Parsing inside
            # the loop (as before) could read a truncated / not-yet-flushed log
            # and report a wrong mean (nvbugs 6487036 / 6487040).
            pending_device_step_time: List[dict] = []
            collect_device_step_time = (
                configs_for_idx is not None and configs_for_idx[2].benchmark_mode == "gen_only"
            )
            try:
                disagg_server_hostname, disagg_server_port = (
                    self._get_disagg_server_hostname_and_port(server_idx)
                )

                wait_for_endpoint_ready(
                    f"http://{disagg_server_hostname}:{disagg_server_port}/health",
                    timeout=min(
                        self.timeout, server_ready_timeout(DISAGG_SERVER_READY_TIMEOUT, "DISAGG")
                    ),
                    check_files=self.get_server_logs(server_idx),
                )
                observations = self._collect_worker_startup_observations(server_idx)
                write_startup_observations(self.test_output_dir, server_idx, observations)

                client_configs = self.client_configs.get(server_idx, [])

                # Run all clients for this server
                for client_idx, client_cmd in enumerate(self.client_cmds[server_idx]):
                    client_config = (
                        client_configs[client_idx] if client_idx < len(client_configs) else None
                    )
                    only_run_accuracy = bool(client_config and client_config.only_run_accuracy)

                    if not only_run_accuracy:
                        benchmark_file_path = os.path.join(
                            self.test_output_dir, f"trtllm-benchmark.{server_idx}.{client_idx}.log"
                        )
                        client_cmd_with_port = add_host_port_to_cmd(
                            client_cmd, disagg_server_hostname, disagg_server_port
                        )
                        print_info(f"Starting benchmark. cmd is {client_cmd_with_port}")

                        # Snapshot gen_server log sizes so the gen_only
                        # per-client average covers only iterations driven by
                        # this client. Other modes do not emit this metric and
                        # must not wait for the GEN teardown sentinel.
                        gen_log_start_offsets = None
                        if collect_device_step_time:
                            gen_log_start_offsets = gen_worker_log_sizes(
                                self.test_output_dir, self.num_gen_servers
                            )

                        bench_env = copy.deepcopy(os.environ)
                        if client_config:
                            bench_env.update(client_config.to_env())
                        # Keep aiperf's artifacts (its own logs included) with
                        # the rest of the lane's output; ignored by other
                        # clients.
                        bench_env["TRTLLM_AGENTX_ARTIFACT_DIR"] = os.path.join(
                            self.test_output_dir, f"agentx.{server_idx}.{client_idx}"
                        )
                        output = _run_benchmark_with_log(
                            client_cmd_with_port,
                            bench_env,
                            benchmark_file_path,
                        )

                        outputs.append(output)
                        if collect_device_step_time:
                            # Defer the gen-worker device-step-time parse until
                            # the gen logs are flushed (see below); remember
                            # where to write the summary back.
                            pending_device_step_time.append(
                                {
                                    "output_index": len(outputs) - 1,
                                    "benchmark_file_path": benchmark_file_path,
                                    "start_offsets": gen_log_start_offsets,
                                }
                            )
                    else:
                        print_info(
                            f"Skipping perf benchmark for client {client_idx}: "
                            "only_run_accuracy=True"
                        )
                        outputs.append("")

                # Prefer per-client AccuracyConfig (sourced from yaml). Fall back
                # to ACCURACY_CONFIG_JSON env-var injected by submit.py for older
                # workflows.
                accuracy_cfg = None
                if client_configs and client_configs[0].accuracy_config:
                    accuracy_cfg = client_configs[0].accuracy_config
                else:
                    acc_cfg_json = os.environ.get("ACCURACY_CONFIG_JSON")
                    if acc_cfg_json:
                        import json as _json

                        accuracy_cfg = AccuracyConfig.from_dict(_json.loads(acc_cfg_json))

                if accuracy_cfg and accuracy_cfg.enable_accuracy_test:
                    accuracy_cfg.run(
                        model_name=self.model_name,
                        server_hostname=disagg_server_hostname,
                        server_port=disagg_server_port,
                        output_dir=self.test_output_dir,
                        server_idx=server_idx,
                    )

            finally:
                with open(benchmark_status_file, "w") as status_file:
                    status_file.write("Done")

            # benchmark_status is written, so the gen workers can now stop and
            # their srun will exit and drop gen_server_{i}.done. Wait once for
            # those sentinels (bounded independently of the whole-test timeout),
            # then parse each benchmark client's gen-worker device step time a
            # single time. A timeout falls back to the current log contents.
            # Only gen_only runs populate this queue; other modes skip both the
            # sentinel wait and device-step-time parsing.
            self._append_gen_worker_device_step_time(pending_device_step_time, outputs)

        return outputs

    def get_cmd_str(self, server_idx: int) -> List[str]:
        return ["multi-node disaggregated server tests, please check config files"]


def parse_select_pattern(select_pattern: str) -> list:
    """Parse select pattern (server config names).

    Args:
        select_pattern: Server config names separated by comma
            (e.g., "r1_fp4_v2_dep4_mtp1_1k1k,r1_fp4_v2_tep4_mtp3_1k1k,r1_fp4_v2_tp4_mtp3_1k1k").

    Returns:
        List of server config name strings.
    """
    return [name.strip() for name in select_pattern.split(",")]


def parse_test_string(test_case_name: str):
    """Parse test case name to get config base name, select pattern, runtime, and benchmark_mode.

    Test name formats:
    - Disagg e2e: disagg_upload-e2e-{config_base}
    - Disagg gen_only: disagg_upload-gen_only-{config_base}
    - ctx_only: aggr_upload-ctx_only-{config_base} (runs aggr mode but reads disagg config)
    - Regular aggr: aggr_upload-{config}-{server_name}

    Returns:
        tuple: (config_base_name, select_pattern, runtime_mode, benchmark_mode)
            - runtime_mode: "aggregated" or "disaggregated"
            - benchmark_mode: "e2e", "gen_only", "ctx_only", or None (for normal aggr)
    """
    labels = test_case_name.split("-")

    assert len(labels) > 1, "perf_sanity test must have a config file!"

    prefix = labels[0]
    is_disagg_prefix = "disagg" in prefix
    is_aggr_prefix = "aggr" in prefix

    if is_disagg_prefix:
        # Disagg format: disagg_upload-{e2e|gen_only}-{config_base}
        assert len(labels) > 2, "Disagg test must have benchmark_mode and config!"
        benchmark_mode = labels[1]  # e2e or gen_only
        assert benchmark_mode in ("e2e", "gen_only"), (
            f"Invalid benchmark_mode for disagg: {benchmark_mode}"
        )
        runtime_mode = "disaggregated"
        config_base_name = "-".join(labels[2:])
        select_pattern = None
    elif is_aggr_prefix:
        # Check if this is ctx_only (aggr_upload-ctx_only-{config_base})
        if len(labels) > 2 and labels[1] == "ctx_only":
            # ctx_only: aggr_upload-ctx_only-{config_base}
            # Runs in aggregated mode but reads disagg config
            benchmark_mode = "ctx_only"
            runtime_mode = "aggregated"
            config_base_name = "-".join(labels[2:])
            select_pattern = None
        else:
            # Regular aggr: aggr_upload-config_yml or aggr_upload-config_yml-server_config_name
            benchmark_mode = None
            runtime_mode = "aggregated"
            config_base_name = labels[1]
            # select_pattern is server config name (e.g., "r1_fp8_dep8_mtp1_1k1k")
            select_pattern = "-".join(labels[2:]) if len(labels) > 2 else None
    else:
        raise ValueError(f"Invalid test name prefix: {prefix}")

    return config_base_name, select_pattern, runtime_mode, benchmark_mode


def get_config_dir(benchmark_mode: Optional[str]) -> str:
    """Get config directory based on benchmark_mode.

    Args:
        benchmark_mode: "e2e", "gen_only", "ctx_only", or None (for normal aggr)

    Returns:
        str: Absolute config directory path
    """
    if benchmark_mode in ("e2e", "gen_only", "ctx_only"):
        config_dir = DISAGG_CONFIG_FOLDER
    else:
        config_dir = AGG_CONFIG_FOLDER
    # If relative path, join with llm root
    if not os.path.isabs(config_dir):
        config_dir = os.path.join(get_llm_root(), config_dir)
    return config_dir


class PerfSanityTestConfig:
    """Configuration for perf sanity tests."""

    def __init__(self, test_case_name: str, output_dir: str):
        self._output_dir = output_dir
        self._perf_results: Dict[int, List[Dict[str, float]]] = {}

        # Initialize server configs
        self.server_configs: List = []
        self.server_client_configs: Dict[int, List[ClientConfig]] = {}

        # Parse test case name
        self.parse_test_case_name(test_case_name)

    def parse_test_case_name(self, test_case_name: str):
        """Parse test case name into components."""
        self._test_param_labels = test_case_name

        def get_gpu_type() -> str:
            try:
                output = subprocess.check_output(
                    "nvidia-smi -q | grep 'Product Name' | head -1",
                    shell=True,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                model = output.split()[-1]
                return SUPPORTED_GPU_MAPPING.get(model, "unsupported")
            except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
                raise RuntimeError("Failed to get GPU type")

        self.upload_to_db = "upload" in test_case_name.split("-")[0] and bool(
            os.environ.get("OPEN_SEARCH_DB_BASE_URL", "")
        )
        self.gpu_type = get_gpu_type()

        # Parse test case name to get config_base_name, select_pattern, runtime, benchmark_mode
        config_base_name, self.select_pattern, runtime, self.benchmark_mode = parse_test_string(
            test_case_name
        )

        # Set runtime based on parsed result
        if runtime == "disaggregated":
            self.runtime = "multi_node_disagg_server"
        else:
            self.runtime = "aggr_server"

        # Set config_file
        self.config_file = (
            f"{config_base_name}.yaml"
            if not config_base_name.endswith(".yaml")
            else config_base_name
        )

        # Get config_dir based on benchmark_mode
        self.config_dir = get_config_dir(self.benchmark_mode)

    def parse_config_file(self):
        """Parse config file based on runtime and benchmark_mode."""
        config_file_path = os.path.join(self.config_dir, self.config_file)

        # benchmark_mode determines which parser to use:
        # - e2e, gen_only, ctx_only: use _parse_disagg_config_file (reads disagg config)
        # - None (normal aggr): use _parse_aggr_config_file
        if self.benchmark_mode in ("e2e", "gen_only", "ctx_only"):
            self._parse_disagg_config_file(config_file_path, self.config_file)
        else:
            # Normal aggregated mode
            self._parse_aggr_config_file(config_file_path)

    def _parse_aggr_config_file(self, config_file_path: str):
        """Parse YAML config file for aggregated server."""
        # Parse selection pattern (server config names)
        if self.select_pattern:
            selected_server_names = parse_select_pattern(self.select_pattern)
        else:
            selected_server_names = None

        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)

        metadata = config.get("metadata", {})
        hardware = config.get("hardware", {})
        gpus_per_node = hardware.get("gpus_per_node", 0)

        model_name = metadata.get("model_name", "")

        server_configs = []
        server_client_configs = {}

        for server_idx, server_config_data in enumerate(config["server_configs"]):
            # Check if this server should be included based on selected_server_names
            if (
                selected_server_names is not None
                and server_config_data.get("name") not in selected_server_names
            ):
                continue

            server_config_data["model_name"] = (
                model_name
                if "model_name" not in server_config_data
                else server_config_data["model_name"]
            )
            server_config_data["concurrency"] = -1
            server_config_data["gpus_per_node"] = gpus_per_node

            # Per-config env vars: server_env_var lives on each server_config entry,
            # client_env_var lives on each client_config entry.
            server_env_var = server_config_data.get("server_env_var", "")
            server_config = ServerConfig(server_config_data, server_env_var)
            server_id = len(server_configs)
            server_configs.append(server_config)

            client_configs = []
            for client_config_data in server_config_data["client_configs"]:
                client_env_var = client_config_data.get("client_env_var", "")
                client_config = ClientConfig(
                    client_config_data,
                    server_config_data["model_name"],
                    env_vars=client_env_var,
                    spec_decoding=bool(server_config.spec_decoding_type),
                )
                client_configs.append(client_config)

            server_client_configs[server_id] = client_configs

        self.server_configs = server_configs
        self.server_client_configs = server_client_configs

    def _parse_disagg_config_file(self, config_file_path: str, config_file: str):
        """Parse YAML config file for disaggregated server.

        This method handles e2e, gen_only, and ctx_only modes.
        For ctx_only: output is on par with _parse_aggr_config_file (single ServerConfig),
                     OSL is set to 1, and cache_transceiver_config is ignored.
        """
        disagg_serving_type = os.environ.get("DISAGG_SERVING_TYPE", "BENCHMARK")

        # Get config file base name (without extension)
        config_file_base_name = os.path.splitext(config_file)[0]

        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)

        metadata = config.get("metadata", {})
        hardware = config.get("hardware", {})
        benchmark = config.get("benchmark", {})
        environment = config.get("environment", {})
        slurm_config = config.get("slurm", {})
        worker_config = config.get("worker_config", {})

        timeout = slurm_config.get("timeout", DEFAULT_TIMEOUT)
        numa_bind = slurm_config.get("numa_bind", False)
        gpus_per_node = hardware.get("gpus_per_node", 0)
        model_name = metadata.get("model_name", "")
        assert model_name, "model_name is required in metadata section"

        # Use self.benchmark_mode instead of reading from config file
        benchmark_mode = self.benchmark_mode
        if benchmark_mode == "gen_only":
            # Check if it's gen_only_no_context from config
            config_mode = benchmark.get("mode", "e2e")
            if "gen_only_no_context" in config_mode:
                hardware["num_ctx_servers"] = 0

        worker_env_var = environment.get("worker_env_var", "")
        # Optional per-role env vars appended to the shared worker_env_var so
        # ctx and gen workers can diverge (e.g. PYTORCH_CUDA_ALLOC_CONF on ctx
        # only). Absent keys leave the shared value untouched.
        ctx_worker_env_var_extra = environment.get("ctx_worker_env_var", "") or ""
        gen_worker_env_var_extra = environment.get("gen_worker_env_var", "") or ""
        ctx_worker_env_var = " ".join(
            part for part in (worker_env_var, ctx_worker_env_var_extra) if part
        )
        gen_worker_env_var = " ".join(
            part for part in (worker_env_var, gen_worker_env_var_extra) if part
        )
        server_env_var = environment.get("server_env_var", "")
        client_env_var = environment.get("client_env_var", "")
        internal_request_auth_key = self._resolve_internal_request_auth_key(config)
        # Optional disagg-server-level keys, same names as bench-trtllm-disagg's
        # sweep config so a recipe can be carried over unchanged.
        router_config = config.get("router_config", None)
        ctx_router_config = config.get("ctx_router_config", None)
        gen_router_config = config.get("gen_router_config", None)
        server_config_extra = config.get("server_config_extra", None)

        # Parse concurrency_list - can be string or list
        concurrency_str = benchmark.get("concurrency_list", "1")
        if isinstance(concurrency_str, str):
            concurrency_values = [int(x) for x in concurrency_str.split()]
        elif isinstance(concurrency_str, list):
            concurrency_values = [int(x) for x in concurrency_str]
        else:
            concurrency_values = [int(concurrency_str)]

        # Gen only mode only runs the first concurrency
        if benchmark_mode == "gen_only":
            concurrency_values = [concurrency_values[0]]

        # Handle ctx_only mode specially - output should be on par with _parse_aggr_config_file
        if benchmark_mode == "ctx_only":
            # Get ctx worker config and modify it
            ctx_config = dict(worker_config.get("ctx", {}))
            # Ignore cache_transceiver_config for ctx_only
            ctx_config.pop("cache_transceiver_config", None)

            # Create server config for ctx_only (single ServerConfig, not tuple)
            ctx_server_config_data = {
                "concurrency": -1,  # Same as aggr
                "name": f"{benchmark_mode}-{config_file_base_name}",
                "model_name": model_name,
                "gpus_per_node": gpus_per_node,
                "disagg_run_type": "aggr",  # Run as aggr
                **ctx_config,
            }

            # ctx_only runs the ctx worker in aggregated mode; use the merged
            # ctx-side env var so the aggregated run still gets any ctx-only
            # extras from the disagg yaml.
            ctx_server_config = ServerConfig(ctx_server_config_data, ctx_worker_env_var)
            self.server_configs = [ctx_server_config]
        else:
            # For e2e and gen_only modes - create ctx and gen server configs
            ctx_server_config_data = {
                "internal_request_auth_key": internal_request_auth_key,
                "concurrency": concurrency_values[0],
                "name": f"{benchmark_mode}-{config_file_base_name}",
                "model_name": model_name,
                "gpus_per_node": gpus_per_node,
                "disagg_run_type": "ctx",
                **worker_config.get("ctx", {}),
            }

            gen_server_config_data = {
                "internal_request_auth_key": internal_request_auth_key,
                "concurrency": concurrency_values[0],
                "name": f"{benchmark_mode}-{config_file_base_name}",
                "model_name": model_name,
                "gpus_per_node": gpus_per_node,
                "disagg_run_type": "gen",
                **worker_config.get("gen", {}),
            }

            ctx_server_config = ServerConfig(ctx_server_config_data, ctx_worker_env_var)
            gen_server_config = ServerConfig(gen_server_config_data, gen_worker_env_var)

            disagg_config = DisaggConfig(
                name=f"{benchmark_mode}-{config_file_base_name}",
                disagg_serving_type=disagg_serving_type,
                hostname=socket.gethostname(),
                numa_bind=numa_bind,
                timeout=timeout,
                benchmark_mode=benchmark_mode,
                model_name=model_name,
                hardware=hardware,
                server_env_var=server_env_var,
                internal_request_auth_key=internal_request_auth_key,
                router_config=router_config,
                ctx_router_config=ctx_router_config,
                gen_router_config=gen_router_config,
                server_config_extra=server_config_extra,
            )

            # server_configs is a list with one element (tuple of ctx, gen, disagg config)
            self.server_configs = [(ctx_server_config, gen_server_config, disagg_config)]

        # Create client configs for each concurrency value
        # For ctx_only: OSL is set to 1 and dataset_file is empty
        osl = 1 if benchmark_mode == "ctx_only" else benchmark.get("output_length", 1024)
        dataset_file = "" if benchmark_mode == "ctx_only" else benchmark.get("dataset_file", "")
        use_nv_sa_benchmark = benchmark.get("use_nv_sa_benchmark", False)
        benchmark_client = benchmark.get("benchmark_client", "")
        if benchmark_client not in ("", AGENTX_BENCHMARK_CLIENT):
            # There is no schema validation on these yamls, so an unrecognised
            # value would otherwise fall through to the default client and
            # quietly measure the wrong workload.
            raise ValueError(
                f"Unknown benchmark_client {benchmark_client!r}; "
                f"expected '' or {AGENTX_BENCHMARK_CLIENT!r}."
            )

        if benchmark_mode == "ctx_only":
            spec_decoding = bool(ctx_server_config.spec_decoding_type)
        else:
            spec_decoding = bool(ctx_server_config.spec_decoding_type) or bool(
                gen_server_config.spec_decoding_type
            )

        # Accuracy lives at the top of disagg yamls; only_run_accuracy lives inside
        # benchmark: (since `benchmark` is what becomes the disagg ClientConfig).
        accuracy_data = config.get("accuracy") or None
        only_run_accuracy = bool(benchmark.get("only_run_accuracy", False))

        client_configs = []
        for concurrency in concurrency_values:
            client_config_data = {
                "concurrency": concurrency,
                "iterations": 1
                if benchmark_mode == "gen_only"
                else benchmark.get("multi_round", 1),
                "isl": benchmark.get("input_length", 1024),
                "osl": osl,
                "random_range_ratio": benchmark.get("benchmark_ratio", 0.0),
                "backend": "openai",
                "use_chat_template": False,
                "streaming": benchmark.get("streaming", True),
                "dataset_file": dataset_file,
                "use_nv_sa_benchmark": use_nv_sa_benchmark,
                "benchmark_client": benchmark_client,
                "accuracy_config": accuracy_data,
                "only_run_accuracy": only_run_accuracy,
            }
            client_config = ClientConfig(
                client_config_data,
                model_name,
                env_vars=client_env_var,
                spec_decoding=spec_decoding,
                warmup=wants_warmup(benchmark_mode),
            )
            client_configs.append(client_config)

        self.server_client_configs = {0: client_configs}

    def _resolve_internal_request_auth_key(self, config: dict) -> str:
        explicit_key = config.get("internal_request_auth_key")
        if explicit_key:
            return explicit_key

        test_output_dir = os.path.join(self._output_dir, self._test_param_labels)
        os.makedirs(test_output_dir, exist_ok=True)
        key_path = os.path.join(test_output_dir, "internal_request_auth_key.txt")
        lock_path = f"{key_path}.lock"

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                if os.path.exists(key_path):
                    with open(key_path, "r") as key_file:
                        internal_request_auth_key = key_file.read().strip()
                    if internal_request_auth_key:
                        return internal_request_auth_key

                internal_request_auth_key = secrets.token_hex(32)
                with open(key_path, "w") as key_file:
                    key_file.write(f"{internal_request_auth_key}\n")
                return internal_request_auth_key
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def get_commands(self):
        """Get commands based on runtime and benchmark_mode."""
        self.test_output_dir = os.path.join(self._output_dir, self._test_param_labels)
        os.makedirs(self.test_output_dir, exist_ok=True)

        # ctx_only runs in aggregated mode (uses _get_aggr_commands)
        if self.runtime == "aggr_server":
            return self._get_aggr_commands(self._output_dir, self.test_output_dir)
        else:
            return self._get_disagg_commands(self._output_dir, self.test_output_dir)

    def _get_aggr_commands(self, output_dir: str, test_output_dir: str):
        """Get commands for aggregated server."""
        server_cmds = []
        client_cmds = {}

        for server_idx, client_configs in self.server_client_configs.items():
            server_config = self.server_configs[server_idx]
            server_cmd = server_config.to_cmd(test_output_dir)

            # Generate extra-llm-api-config.yml
            config_content = server_config.generate_extra_llm_api_config()
            config_filename = f"extra-llm-api-config.aggr.{server_config.name}.yml"
            config_path = os.path.join(test_output_dir, config_filename)
            with open(config_path, "w") as f:
                f.write(config_content)

            server_cmds.append(server_cmd)
            client_cmds[server_idx] = []

            for client_config in client_configs:
                client_cmd = client_config.to_cmd()
                client_cmds[server_idx].append(client_cmd)

        # AggrTestCmds needs the model name (for lm_eval --model_args). All
        # server_configs in an agg yaml share the same model_name.
        first_server = self.server_configs[0] if self.server_configs else None
        agg_model_name = first_server.model_name if first_server else ""

        return AggrTestCmds(
            server_cmds=server_cmds,
            client_cmds=client_cmds,
            timeout=DEFAULT_TIMEOUT,
            output_dir=output_dir,
            test_output_dir=test_output_dir,
            client_configs=self.server_client_configs,
            model_name=agg_model_name,
            server_configs=list(self.server_configs),
        )

    def _get_disagg_commands(self, output_dir: str, test_output_dir: str):
        """Get commands for disaggregated server."""
        server_cmds = []
        client_cmds = {}

        for server_idx, (ctx_config, gen_config, disagg_config) in enumerate(self.server_configs):
            numa_bind = disagg_config.numa_bind
            timeout = disagg_config.timeout
            disagg_serving_type = disagg_config.disagg_serving_type

            # Generate ctx server command
            ctx_cmd = ctx_config.to_cmd(test_output_dir, numa_bind, "CTX")
            if disagg_serving_type == "CTX_0":
                config_content = ctx_config.generate_extra_llm_api_config()
                config_path = os.path.join(
                    test_output_dir, f"extra-llm-api-config.ctx.{ctx_config.name}.yml"
                )
                with open(config_path, "w") as f:
                    f.write(config_content)

            # Generate gen server command
            gen_cmd = gen_config.to_cmd(test_output_dir, numa_bind, "GEN")
            if disagg_serving_type == "GEN_0":
                config_content = gen_config.generate_extra_llm_api_config()
                config_path = os.path.join(
                    test_output_dir, f"extra-llm-api-config.gen.{gen_config.name}.yml"
                )
                with open(config_path, "w") as f:
                    f.write(config_content)

            # Generate disagg server command
            disagg_cmd = [
                "trtllm-serve",
                "disaggregated",
                "-c",
                f"{test_output_dir}/server_config.{server_idx}.yaml",
                "-t",
                str(timeout),
                "-r",
                str(timeout),
            ]

            server_cmds.append((ctx_cmd, gen_cmd, disagg_cmd))

            # Add client commands
            client_cmds[server_idx] = []
            for client_config in self.server_client_configs[server_idx]:
                client_cmd = client_config.to_cmd()
                client_cmds[server_idx].append(client_cmd)

        disagg_config = self.server_configs[0][2]
        return DisaggTestCmds(
            server_cmds=server_cmds,
            client_cmds=client_cmds,
            timeout=disagg_config.timeout,
            hostname=disagg_config.hostname,
            disagg_serving_type=disagg_config.disagg_serving_type,
            num_ctx_servers=disagg_config.num_ctx_servers,
            num_gen_servers=disagg_config.num_gen_servers,
            output_dir=output_dir,
            test_output_dir=test_output_dir,
            model_name=disagg_config.model_name,
            internal_request_auth_key=disagg_config.internal_request_auth_key,
            router_config=disagg_config.router_config,
            ctx_router_config=disagg_config.ctx_router_config,
            gen_router_config=disagg_config.gen_router_config,
            server_config_extra=disagg_config.server_config_extra,
            client_configs=self.server_client_configs,
            server_configs=list(self.server_configs),
        )

    def _check_benchmark_errors(self, output: str) -> None:
        """Check whether the benchmark output contains error messages."""
        if not output:
            return

        # Check for non-zero failed requests (default benchmark)
        failed_requests_match = re.search(r"Failed requests:\s+(\d+)", output)
        if failed_requests_match:
            failed_count = int(failed_requests_match.group(1))
            if failed_count > 0:
                error_msg = f"Benchmark output contains {failed_count} failed requests."
                raise RuntimeError(error_msg)

        # Check for explicit failure markers (default benchmark)
        if "!FAILED REQUESTS!" in output or "!CHECK LOG FOR ERRORS!" in output:
            error_msg = "Benchmark output contains failure markers."
            raise RuntimeError(error_msg)

        # SA benchmark (bench_serving) only prints "Successful requests:"
        # without "Failed requests:". Detect failures by comparing successful
        # count against num_prompts from the Namespace output.
        if not failed_requests_match:
            successful_match = re.search(r"Successful requests:\s+(\d+)", output)
            num_prompts_match = re.search(r"num_prompts=(\d+)", output)
            if successful_match and num_prompts_match:
                successful_count = int(successful_match.group(1))
                num_prompts = int(num_prompts_match.group(1))
                failed_count = num_prompts - successful_count
                if failed_count > 0:
                    error_msg = (
                        f"SA benchmark: {failed_count} of {num_prompts} requests failed "
                        f"({successful_count} successful)."
                    )
                    raise RuntimeError(error_msg)

    def run_ex(self, commands) -> Dict[int, List[str]]:
        """Run commands and collect outputs."""
        outputs = {}
        for server_idx in range(len(commands.server_cmds)):
            try:
                server_outputs = commands.run_cmd(server_idx)
                for output in server_outputs:
                    self._check_benchmark_errors(output)
                outputs[server_idx] = server_outputs

            except Exception as e:
                outputs[server_idx] = []
                # Aggregated mode does not set DISAGG_SERVING_TYPE, so the
                # default "BENCHMARK" applies and report_error is always called.
                # Disagg mode sets DISAGG_SERVING_TYPE per srun; only the
                # BENCHMARK srun reports errors gathered from sibling logs.
                if os.environ.get("DISAGG_SERVING_TYPE", "BENCHMARK") == "BENCHMARK":
                    report_error(
                        error_msg=e,
                        log_files=commands.get_server_logs(server_idx),
                    )
                raise

        return outputs

    def get_perf_result(self, outputs: Dict[int, List[str]]):
        """Parse performance results from outputs."""

        def parse_metrics_from_output(output: str) -> Optional[Dict[str, float]]:
            """Parse all metrics from a single output string."""
            metrics = {}
            all_queries = {
                **PERF_METRIC_LOG_QUERIES,
                **SPEC_DECODING_PERF_METRIC_LOG_QUERIES,
                **GEN_ONLY_PERF_METRIC_LOG_QUERIES,
            }
            for line in output.split("\n"):
                for metric_type, regex in all_queries.items():
                    if metric_type in metrics:
                        continue
                    match = regex.search(line)
                    if match:
                        metrics[metric_type] = float(match.group(1))
                        break
            return metrics

        self._perf_results = {}
        for server_idx, client_configs in self.server_client_configs.items():
            self._perf_results[server_idx] = []
            server_outputs = outputs.get(server_idx, [])
            for client_idx, output in enumerate(server_outputs):
                # only_run_accuracy clients have no benchmark output to parse;
                # use None sentinel so check/upload paths can skip them.
                if (
                    client_idx < len(client_configs)
                    and client_configs[client_idx].only_run_accuracy
                ):
                    self._perf_results[server_idx].append(None)
                    continue
                metrics = parse_metrics_from_output(output)
                # SA benchmark (bench_serving) doesn't report user_throughput.
                # Use None as sentinel to distinguish "not available" from actual zero.
                if (
                    metrics
                    and "user_throughput" not in metrics
                    and client_idx < len(client_configs)
                    and client_configs[client_idx].use_nv_sa_benchmark
                ):
                    metrics["user_throughput"] = None
                self._perf_results[server_idx].append(metrics)

    def check_test_failure(self):
        """Check if any server failed based on perf results."""
        error_msg = ""
        for server_idx, client_configs in self.server_client_configs.items():
            server_perf_results = self._perf_results.get(server_idx, [])
            if len(server_perf_results) != len(client_configs):
                error_msg += (
                    f"Server {server_idx}'s perf results number: {len(server_perf_results)} "
                    f"is not equal to client number: {len(client_configs)}. "
                )
            for client_idx, metrics in enumerate(server_perf_results):
                # only_run_accuracy clients produce no perf metrics by design.
                if (
                    client_idx < len(client_configs)
                    and client_configs[client_idx].only_run_accuracy
                ):
                    continue
                missing = [k for k in PERF_METRIC_LOG_QUERIES if k not in (metrics or {})]
                if missing:
                    error_msg += (
                        f"Some metrics in Server {server_idx} Client {client_idx} are missing: "
                        f"{missing}. The parsed metrics is {metrics}. "
                    )
                # Spec-decoding tests must report 'Mean Avg Decoded Tokens per Iter'
                # (parsed as 'al'). If the field is missing the test fails here so the
                # data is never uploaded to OpenSearch.
                # AgentX is exempt: 'al' comes from TRT-LLM's non-standard
                # avg_decoded_tokens_per_iter response field, which aiperf does
                # not propagate. It is not a real loss of signal, because every
                # agentx lane pins the accepted length with
                # TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS -- recorded on the
                # case as l_force_num_accepted_tokens -- so 'al' would
                # be a restatement of a configured constant rather than a
                # measurement. The exemption is conditioned on that forcing
                # actually being in effect rather than merely documented, so an
                # agentx lane that ever runs spec decoding without pinning the
                # accepted count still hard-fails here.
                agentx_al_exempt = False
                if (
                    client_idx < len(client_configs)
                    and client_configs[client_idx].benchmark_client == AGENTX_BENCHMARK_CLIENT
                ):
                    server_entry = self.server_configs[server_idx]
                    # disagg stores (ctx, gen, disagg); aggregated stores one config.
                    candidates = (
                        server_entry if isinstance(server_entry, tuple) else (server_entry,)
                    )
                    agentx_al_exempt = any(
                        getattr(c, "force_num_accepted_tokens", 0) for c in candidates
                    )
                if (
                    client_idx < len(client_configs)
                    and client_configs[client_idx].spec_decoding
                    and not agentx_al_exempt
                    and "al" not in metrics
                ):
                    error_msg += (
                        f"Speculative decoding test Server {server_idx} Client {client_idx} "
                        f"is missing 'Mean Avg Decoded Tokens per Iter' in benchmark output. "
                    )
                # gen_only tests must report mean_gen_worker_per_iter_device_step_time
                # (parsed from gen_server_*.log). It is a regression metric for gen_only,
                # so a missing value must hard-fail rather than silently upload. Checking
                # the mean alone is sufficient: all five statistics come from the same
                # _DeviceStepTimeStats, so the mean is absent only if all of them are.
                if (
                    self.runtime == "multi_node_disagg_server"
                    and self.server_configs[server_idx][2].benchmark_mode == "gen_only"
                    and (
                        not metrics
                        or metrics.get("mean_gen_worker_per_iter_device_step_time") is None
                    )
                ):
                    error_msg += (
                        f"gen_only test Server {server_idx} Client {client_idx} is "
                        f"missing 'prev_device_step_time' in gen_server_*.log under "
                        f"{self._output_dir}. "
                    )
        if error_msg:
            raise RuntimeError(error_msg)

    def upload_test_results_to_database(self):
        """Upload test results and baseline to database."""

        def add_prefix(key: str, prefix_name: str) -> str:
            type_prefix = key[0:2]
            rest = key[2:]
            return f"{type_prefix}{prefix_name}_{rest}"

        def add_dict_prefix(config_dict: dict, prefix_name: str) -> dict:
            return {add_prefix(key, prefix_name): value for key, value in config_dict.items()}

        match_keys = get_test_case_match_keys()

        if self.runtime == "aggr_server":
            new_data_dict = {}
            cmd_idx = 0
            for server_idx, client_configs in self.server_client_configs.items():
                server_config = self.server_configs[server_idx]
                server_config_dict = server_config.to_db_data()
                server_perf_results = self._perf_results.get(server_idx, [])
                startup_observations = read_startup_observations(self.test_output_dir, server_idx)
                # Skip if server failed
                if len(server_perf_results) != len(client_configs):
                    cmd_idx += len(client_configs)
                    continue

                for client_idx, client_config in enumerate(client_configs):
                    client_config_dict = client_config.to_db_data()

                    # Skip if metrics missing
                    if server_perf_results[client_idx] is None:
                        print_info(
                            f"Skipped posting command {cmd_idx}'s test results since some metrics are missing."
                        )
                        cmd_idx += 1
                        continue

                    new_data = {
                        "s_gpu_type": self.gpu_type,
                        "s_runtime": "multi_node_aggr_server"
                        if server_config.gpus != server_config.gpus_per_node
                        else "aggr_server",
                    }
                    new_data.update(server_config_dict)
                    new_data.update(client_config_dict)
                    # Add test_case_name for convenient filtering on OpenSearch
                    new_data["s_test_case_name"] = f"{server_config.name}-{client_config.name}"

                    add_perf_metric_value(
                        new_data,
                        server_perf_results[client_idx],
                        spec_decoding=client_config.spec_decoding,
                    )
                    add_startup_metric_values(
                        new_data,
                        startup_observations,
                        expected_server_count=1,
                    )

                    new_data_dict[cmd_idx] = new_data
                    cmd_idx += 1

        elif self.runtime == "multi_node_disagg_server":
            # Only BENCHMARK node uploads
            if self.server_configs[0][2].disagg_serving_type != "BENCHMARK":
                return

            new_data_dict = {}
            cmd_idx = 0

            for server_idx, (ctx_config, gen_config, disagg_config) in enumerate(
                self.server_configs
            ):
                client_configs = self.server_client_configs[server_idx]
                server_perf_results = self._perf_results.get(server_idx, [])
                startup_observations = read_startup_observations(self.test_output_dir, server_idx)
                # Skip if server failed
                if len(server_perf_results) != len(client_configs):
                    cmd_idx += len(client_configs)
                    continue

                for client_idx, client_config in enumerate(client_configs):
                    # Skip if metrics missing
                    if server_perf_results[client_idx] is None:
                        print_info(
                            f"Skipped posting command {cmd_idx}'s test results since some metrics are missing."
                        )
                        cmd_idx += 1
                        continue

                    # Get server configs with prefixed keys
                    ctx_server_config_dict = add_dict_prefix(ctx_config.to_db_data(), "ctx")
                    gen_server_config_dict = add_dict_prefix(gen_config.to_db_data(), "gen")
                    client_config_dict = client_config.to_db_data()

                    num_ctx_servers = disagg_config.num_ctx_servers
                    num_gen_servers = disagg_config.num_gen_servers

                    new_data = {
                        "s_gpu_type": self.gpu_type,
                        "s_runtime": "multi_node_disagg_server",
                        "s_benchmark_mode": disagg_config.benchmark_mode,
                        "s_server_env_var": disagg_config.server_env_var,
                        "l_num_ctx_servers": num_ctx_servers,
                        "l_num_gen_servers": num_gen_servers,
                    }
                    if num_ctx_servers > 0:
                        new_data.update(ctx_server_config_dict)
                    if num_gen_servers > 0:
                        new_data.update(gen_server_config_dict)
                    new_data.update(client_config_dict)
                    # Add test_case_name for convenient filtering on OpenSearch
                    new_data["s_test_case_name"] = f"{disagg_config.name}-{client_config.name}"

                    add_perf_metric_value(
                        new_data,
                        server_perf_results[client_idx],
                        spec_decoding=client_config.spec_decoding,
                        benchmark_mode=disagg_config.benchmark_mode,
                    )
                    add_startup_metric_values(
                        new_data,
                        startup_observations,
                        role="ctx",
                        expected_server_count=num_ctx_servers,
                    )
                    add_startup_metric_values(
                        new_data,
                        startup_observations,
                        role="gen",
                        expected_server_count=num_gen_servers,
                    )

                    new_data_dict[cmd_idx] = new_data
                    cmd_idx += 1

        else:
            return

        stage_name = os.environ.get("stageName", "")
        extra_fields = {
            "s_stage_name": stage_name,
            "s_test_list": self._test_param_labels,
        }

        # Stages tagged "FUNCTIONAL-ONLY" run the full perf harness (numbers are
        # still uploaded to OpenSearch and dashboards) but do not fail CI on perf
        # regression -- same behavior as post-merge. Used for pre-merge disagg
        # coverage where the goal is functional-failure detection, not gating on
        # perf. Explicit False (not None) so the auto-detect in
        # process_and_upload_test_results does not flip it back on for pre-merge.
        fail_on_regression = False if "FUNCTIONAL-ONLY" in stage_name else None

        # gen_only tests are gated solely on per-iter prev_device_step_time, not
        # token throughput (token-based numbers are dominated by KV cache transfer
        # time in gen_only mode and are not a useful regression signal there).
        # For all other modes, d_al is added when any client runs spec decoding.
        if self.runtime == "multi_node_disagg_server" and any(
            sc[2].benchmark_mode == "gen_only" for sc in self.server_configs
        ):
            # See GEN_ONLY_REGRESSION_METRICS for why the median is gated too.
            # It has no baseline history yet, and check_regression skips any
            # metric whose baseline is absent or non-positive, so it stays inert
            # until enough runs accrue -- it cannot fail a build before then.
            regression_metrics = list(GEN_ONLY_REGRESSION_METRICS)
        else:
            regression_metrics = list(REGRESSION_METRICS)
            has_spec_decoding = any(
                cc.spec_decoding
                for clients in self.server_client_configs.values()
                for cc in clients
            )
            if has_spec_decoding:
                regression_metrics.append("d_al")

        process_and_upload_test_results(
            new_data_dict=new_data_dict,
            match_keys=match_keys,
            maximize_metrics=MAXIMIZE_METRICS,
            minimize_metrics=MINIMIZE_METRICS,
            regression_metrics=regression_metrics,
            extra_fields=extra_fields,
            upload_to_db=self.upload_to_db,
            fail_on_regression=fail_on_regression,
        )


# Perf sanity test case parameters
AGG_TEST_TYPES = ["aggr_upload", "aggr"]
DISAGG_TEST_TYPES = ["disagg_upload", "disagg"]


def get_server_config_names(yaml_path: str) -> List[str]:
    """Read a YAML file and return the list of server_config names."""
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        if data and "server_configs" in data:
            return [config.get("name", "") for config in data["server_configs"]]
    except Exception:
        pass
    return []


def get_yaml_files_with_server_names(directory: str) -> Dict[str, List[str]]:
    """Scan directory for YAML files and return dict of {basename: [server_config_names]}."""
    yaml_files = glob.glob(os.path.join(directory, "*.yaml"))
    result = {}
    for yaml_path in sorted(yaml_files):
        basename = os.path.splitext(os.path.basename(yaml_path))[0]
        server_names = get_server_config_names(yaml_path)
        result[basename] = server_names
    return result


def get_aggr_test_cases() -> List[str]:
    """Generate aggr test cases based on actual server_config names in YAML files."""
    aggr_config_dir = AGG_CONFIG_FOLDER
    # If relative path, join with llm root
    if not os.path.isabs(aggr_config_dir):
        aggr_config_dir = os.path.join(get_llm_root(), aggr_config_dir)
    yaml_server_names = get_yaml_files_with_server_names(aggr_config_dir)

    test_cases = []
    for config_yml, server_names in yaml_server_names.items():
        for test_type in AGG_TEST_TYPES:
            # Case without select_pattern (runs all server configs)
            test_cases.append(f"{test_type}-{config_yml}")

            # Cases with single server config name
            for server_name in server_names:
                test_cases.append(f"{test_type}-{config_yml}-{server_name}")

    return test_cases


def get_disagg_test_cases() -> List[str]:
    """Generate disagg test cases with benchmark modes."""
    disagg_config_dir = DISAGG_CONFIG_FOLDER
    # If relative path, join with llm root
    if not os.path.isabs(disagg_config_dir):
        disagg_config_dir = os.path.join(get_llm_root(), disagg_config_dir)
    yaml_files = glob.glob(os.path.join(disagg_config_dir, "*.yaml"))
    basenames = sorted([os.path.splitext(os.path.basename(f))[0] for f in yaml_files])

    test_cases = []
    for config_yml in basenames:
        # Disagg e2e and gen_only test cases
        for test_type in DISAGG_TEST_TYPES:
            test_cases.append(f"{test_type}-e2e-{config_yml}")
            test_cases.append(f"{test_type}-gen_only-{config_yml}")

        # ctx_only test cases (uses aggr prefix)
        for test_type in AGG_TEST_TYPES:
            test_cases.append(f"{test_type}-ctx_only-{config_yml}")

    return test_cases


# Hardcoded multi-test test cases from test db.
MULTI_TEST_TEST_CASES = []

# Generate all test case combinations
# For aggr: {test_type}-{config_yml}, {test_type}-{config_yml}-{server_config_name}
# For disagg: {test_type}-{config_yml}
PERF_SANITY_TEST_CASES = get_aggr_test_cases() + get_disagg_test_cases() + MULTI_TEST_TEST_CASES


@pytest.mark.parametrize("perf_sanity_test_case", PERF_SANITY_TEST_CASES)
def test_e2e(output_dir, perf_sanity_test_case):
    # Create config and parse test case name
    config = PerfSanityTestConfig(perf_sanity_test_case, output_dir)

    # Parse config file to get server_configs and server_client_configs
    config.parse_config_file()

    # Get commands
    commands = config.get_commands()

    # Run commands and collect outputs
    outputs = config.run_ex(commands)

    # For disagg mode, only BENCHMARK node parses results and uploads
    if config.runtime == "multi_node_disagg_server":
        disagg_config = config.server_configs[0][2]
        if disagg_config.disagg_serving_type != "BENCHMARK":
            print_info(
                f"Disagg serving type is {disagg_config.disagg_serving_type}, skipping perf result parsing and upload."
            )
            return

    # Parse performance results
    config.get_perf_result(outputs)

    # Check for test failures
    config.check_test_failure()

    # Upload results to database
    config.upload_test_results_to_database()
