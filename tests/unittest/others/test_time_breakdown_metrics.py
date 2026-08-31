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
"""Unit tests for perf-sanity time_breakdown metric aggregation."""

import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "integration", "defs", "perf")
)

from time_breakdown_metrics import (  # noqa: E402  isort:skip
    ALL_METRICS,
    GROUP_METRICS,
    MODE_GROUPS,
    STATS,
    compute_time_breakdown_metrics,
)


def _chunk(base, *, fwd=0.100, upd=0.002, smp=0.001, post=0.010, gpu_fwd=110.0):
    """One ctx_chunk_metrics entry starting at ``base`` (seconds)."""
    fwd_end = base + fwd
    smp_start = fwd_end + upd
    smp_end = smp_start + smp
    return {
        "forward_start_time": base,
        "forward_end_time": fwd_end,
        "sample_start_time": smp_start,
        "sample_end_time": smp_end,
        "token_time": smp_end + post,
        "gpu_forward_time": gpu_fwd,
        "gpu_sample_time": 0.02,
    }


def _step(base, idx, *, fwd=0.002, upd=0.009, smp=0.0002, post=0.012):
    entry = _chunk(base, fwd=fwd, upd=upd, smp=smp, post=post, gpu_fwd=12.0)
    entry["iter"] = idx
    return entry


def _worker_record(rid, t0, *, chunks=None, steps=None, gen=False):
    """A plain per-worker record, as ``perf_metrics-server-*.jsonl`` contains."""
    timing = {
        "server_arrival_time": t0,
        "arrival_time": t0 + 0.001,
        "first_scheduled_time": t0 + 0.003,
        "first_token_time": t0 + 0.500,
        "server_first_token_time": t0 + 0.501,
        "last_token_time": t0 + 1.500,
    }
    if gen:
        timing["kv_cache_transfer_start"] = t0 + 0.010
        timing["kv_cache_transfer_end"] = t0 + 0.030
    breakdown = {}
    if chunks is not None:
        breakdown["ctx_chunk_metrics"] = chunks
    if steps is not None:
        breakdown["step_metrics"] = steps
    return {
        "request_id": rid,
        "disagg_request_id": rid,
        "perf_metrics": {"timing_metrics": timing},
        "time_breakdown_metrics": breakdown,
    }


def _write(tmp_path, name, records):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return str(path)


def _ctx_file(tmp_path, name="perf_metrics-server-hostA-1-t.jsonl", n=5, offset=0.0):
    """A context worker whose chunk clock is ``offset`` seconds ahead of its request clock."""
    records = []
    for i in range(n):
        t0 = 1000.0 + i
        # The single chunk must end (token_time) exactly at first_token_time + offset.
        chunk = _chunk(t0 + 0.003 + offset + 0.005)
        shift = (t0 + 0.500 + offset) - chunk["token_time"]
        chunk = {k: (v + shift if k.endswith("_time") else v) for k, v in chunk.items()}
        records.append(_worker_record(i, t0, chunks=[chunk]))
    return _write(tmp_path, name, records)


def _gen_file(tmp_path, name, n=4, nsteps=6, offset=0.0):
    records = []
    for i in range(n):
        t0 = 2000.0 + i
        steps = []
        cursor = t0 + 0.100 + offset
        for s in range(nsteps):
            entry = _step(cursor, s + 1)
            steps.append(entry)
            cursor = entry["token_time"] + 0.0005
        shift = (t0 + 1.500 + offset) - steps[-1]["token_time"]
        steps = [
            {k: (v + shift if k.endswith("_time") else v) for k, v in s.items()} for s in steps
        ]
        records.append(_worker_record(i, t0, steps=steps, gen=True))
    return _write(tmp_path, name, records)


def _combined(ctx_records, gen_records, ctx_server, gen_servers):
    """Merge plain worker records into the disagg combined shape."""
    out = []
    for i, (c, g) in enumerate(zip(ctx_records, gen_records)):
        ctm = c["perf_metrics"]["timing_metrics"]
        gtm = g["perf_metrics"]["timing_metrics"]
        out.append(
            {
                "disagg_request_id": i,
                "disagg_server_arrival_time": ctm["server_arrival_time"] - 0.002,
                "disagg_server_first_token_time": gtm["server_first_token_time"] + 0.0005,
                "ctx_server": ctx_server,
                "gen_server": gen_servers[i % len(gen_servers)],
                "ctx_perf_metrics": c,
                "gen_perf_metrics": g,
            }
        )
    return out


def test_schema_is_stable_and_complete():
    metrics, _ = compute_time_breakdown_metrics([], "e2e")
    assert len(metrics) == len(ALL_METRICS) * len(STATS)
    assert len(ALL_METRICS) == len(set(ALL_METRICS)), "duplicate metric name"
    # Every mode yields the identical key set, so the OpenSearch doc schema never varies.
    for mode in list(MODE_GROUPS) + ["aggr"]:
        other, _ = compute_time_breakdown_metrics([], mode)
        assert set(other) == set(metrics)


def test_aggregated_mode_is_all_zero(tmp_path):
    paths = [_ctx_file(tmp_path), _gen_file(tmp_path, "perf_metrics-server-hostB-2-t.jsonl")]
    metrics, info = compute_time_breakdown_metrics(paths, "aggr")
    assert info["groups"] == []
    assert set(metrics.values()) == {0.0}


@pytest.mark.parametrize("mode", sorted(MODE_GROUPS))
def test_mode_gating_zeroes_unsupported_groups(tmp_path, mode):
    ctx = _ctx_file(tmp_path)
    gen = _gen_file(tmp_path, "perf_metrics-server-hostB-2-t.jsonl")
    metrics, info = compute_time_breakdown_metrics([ctx, gen], mode)
    for group, names in GROUP_METRICS.items():
        supported = group in MODE_GROUPS[mode]
        for name in names:
            values = [metrics[f"d_tb_{name}_{s}"] for s in STATS]
            if not supported:
                assert values == [0.0] * len(STATS), f"{mode}/{name} should be zeroed"
    assert info["groups"] == list(MODE_GROUPS[mode])


def test_ctx_only_needs_no_disagg_server(tmp_path):
    """ctx_only runs the ctx worker in aggregated mode, so there is no combined file."""
    metrics, info = compute_time_breakdown_metrics([_ctx_file(tmp_path)], "ctx_only")
    assert info["counts"]["stage_records"] == 5
    assert metrics["d_tb_ctx_processing_mean"] == pytest.approx(497.0, abs=1.0)
    # Non-chunked prefill is still reported as a single chunk (group 2 populated).
    assert info["sample_counts"]["chunk_forward"] == 5
    assert metrics["d_tb_chunk_forward_mean"] > 0.0


def test_chunk_spans_tile_ctx_processing(tmp_path):
    metrics, _ = compute_time_breakdown_metrics([_ctx_file(tmp_path)], "ctx_only")
    total = sum(
        metrics[f"d_tb_chunk_{n}_mean"]
        for n in ("preprocessing", "forward", "update", "sample", "postprocessing")
    )
    assert total == pytest.approx(metrics["d_tb_ctx_processing_mean"], abs=1e-6)


def test_gen_queue_sub_spans_sum_to_gen_queue(tmp_path):
    gen = _gen_file(tmp_path, "perf_metrics-server-hostB-2-t.jsonl")
    metrics, _ = compute_time_breakdown_metrics([gen], "gen_only")
    subs = sum(
        metrics[f"d_tb_{n}_mean"]
        for n in ("gen_queue_wait", "gen_kv_transfer", "gen_post_transfer")
    )
    assert subs == pytest.approx(metrics["d_tb_gen_queue_mean"], abs=1e-6)


def test_step_preprocessing_may_be_negative_under_overlap(tmp_path):
    """With the overlap scheduler on, step N forwards before step N-1's token_time.

    The value is a legitimate signed term of a contiguous decomposition, so it must be
    reported as-is rather than clamped.
    """
    records = []
    for i in range(4):
        t0 = 2000.0 + i
        steps, cursor = [], t0 + 0.100
        for s in range(6):
            entry = _step(cursor, s + 1)
            steps.append(entry)
            cursor = entry["token_time"] - 0.011  # next forward starts BEFORE this token
        shift = (t0 + 1.500) - steps[-1]["token_time"]
        steps = [
            {k: (v + shift if k.endswith("_time") else v) for k, v in s.items()} for s in steps
        ]
        records.append(_worker_record(i, t0, steps=steps, gen=True))
    path = _write(tmp_path, "perf_metrics-server-hostB-2-t.jsonl", records)
    metrics, _ = compute_time_breakdown_metrics([path], "gen_only")
    assert metrics["d_tb_step_preprocessing_median"] < 0.0
    total = sum(
        metrics[f"d_tb_step_{n}_mean"]
        for n in ("preprocessing", "forward", "update", "sample", "postprocessing")
    )
    assert total > 0.0, "the five step spans must still tile a positive step period"


def test_multi_worker_clock_offsets_are_corrected_per_worker(tmp_path):
    """Regression test for the bug this file exists to prevent.

    A merged/combined file mixes every worker together. Each worker process has its own
    instance-clock origin, so estimating ONE offset across N workers corrupts the
    first-instance preprocessing for N-1 of them. Here four gen workers sit at wildly
    different offsets (one of them ~377680 s, as measured on real hardware); the result must
    match the single-worker, zero-offset case.
    """
    offsets = [0.0003, 377680.565, 9.9706, 0.9927]
    per_worker = 4
    gen_records = []
    for widx, off in enumerate(offsets):
        path = _gen_file(
            tmp_path, f"perf_metrics-server-host{widx}-{widx}-t.jsonl", n=per_worker, offset=off
        )
        gen_records.extend(json.loads(line) for line in open(path))
    ctx_path = _ctx_file(tmp_path, n=len(gen_records))
    ctx_records = [json.loads(line) for line in open(ctx_path)]
    # Round-robin assignment, so consecutive records come from different workers -- exactly the
    # interleaving that makes a single pooled offset estimate look plausible but be wrong.
    gen_servers = [f"genhost{i}:1" for i in range(len(offsets))]
    combined = _combined(
        ctx_records,
        [
            gen_records[(i % len(offsets)) * per_worker + i // len(offsets)]
            for i in range(len(gen_records))
        ],
        "ctxhost:1",
        gen_servers,
    )
    merged_path = _write(tmp_path, "perf_metrics-disagg-hostZ-9-t.jsonl", combined)

    metrics, info = compute_time_breakdown_metrics([merged_path], "e2e")

    # Every worker contributed a first-step preprocessing value; none was discarded.
    assert info["sample_counts"]["step_preprocessing"] == info["sample_counts"]["step_forward"]
    assert any("clock-base offsets spanning" in w for w in info["warnings"])

    # The offsets are an artefact of the workers' clock bases. Removing them per worker must
    # reproduce, stat for stat, what one worker at offset 0 yields -- the four workers are
    # identical apart from their clock base, and each contributes the same share of samples.
    solo, _ = compute_time_breakdown_metrics(
        [_gen_file(tmp_path, "perf_metrics-server-solo-9-t.jsonl", n=per_worker, offset=0.0)],
        "gen_only",
    )
    for stat in STATS:
        assert metrics[f"d_tb_step_preprocessing_{stat}"] == pytest.approx(
            solo[f"d_tb_step_preprocessing_{stat}"], abs=1e-6
        ), stat
    # A single pooled offset would leave residuals of seconds to days on 3 of the 4 workers.
    assert metrics["d_tb_step_preprocessing_p99"] < 2000.0


def test_unverifiable_clock_offset_drops_the_crossing_span_not_guesses(tmp_path):
    """Too few requests to pin a worker's offset => omit that one value, never emit it raw.

    Only the FIRST instance's preprocessing crosses the clock-base boundary. If the offset
    cannot be estimated, emitting it uncorrected would inject a multi-second (here multi-day)
    outlier into an otherwise millisecond-scale distribution, which is far worse than a
    slightly smaller sample.
    """
    path = _gen_file(tmp_path, "perf_metrics-server-lonely-1-t.jsonl", n=1, offset=377680.565)
    metrics, info = compute_time_breakdown_metrics([path], "gen_only")
    # 6 steps: 5 intra-array preprocessing values survive, the boundary-crossing one is dropped.
    assert info["sample_counts"]["step_forward"] == 6
    assert info["sample_counts"]["step_preprocessing"] == 5
    assert abs(metrics["d_tb_step_preprocessing_mean"]) < 1000.0


def test_zero_and_nan_timestamps_are_excluded_not_counted_as_zero_spans(tmp_path):
    """0 / NaN mean 'endpoint not recorded'; they must not become zero-width spans."""
    records = [json.loads(line) for line in open(_ctx_file(tmp_path, n=4))]
    records[0]["perf_metrics"]["timing_metrics"]["server_arrival_time"] = 0
    records[1]["perf_metrics"]["timing_metrics"]["server_arrival_time"] = float("nan")
    path = _write(tmp_path, "perf_metrics-server-hostC-3-t.jsonl", records)
    metrics, info = compute_time_breakdown_metrics([path], "ctx_only")
    assert info["sample_counts"]["ctx_preprocessing"] == 2
    assert metrics["d_tb_ctx_preprocessing_mean"] > 0.0


def test_role_is_classified_by_content_not_filename(tmp_path):
    """Every worker writes kind 'server', so only content can tell ctx from gen."""
    ctx = _ctx_file(tmp_path, "perf_metrics-server-aaa-1-t.jsonl")
    gen = _gen_file(tmp_path, "perf_metrics-server-bbb-2-t.jsonl")
    _, info = compute_time_breakdown_metrics([ctx, gen], "e2e")
    kinds = {name.split("-")[2]: kind.split(" ")[0] for name, kind in info["files"].items()}
    assert kinds == {"aaa": "ctx_worker", "bbb": "gen_worker"}
