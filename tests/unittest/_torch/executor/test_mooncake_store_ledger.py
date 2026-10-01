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
"""Unit tests for reporting what a Mooncake pool was actually built from.

The report reads the records the ranks wrote rather than log wording, so these
tests are mostly about it staying right as things move around it. A report that
is merely wrong is the failure to guard against: nothing about a mis-attributed
byte looks like an error.
"""

import json
import os

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.ledger import (
    format_pool_report,
    read_segments,
    record_segment,
    segments_dir,
)

pytestmark = pytest.mark.cpu_only

GIB = 1 << 30


def write_segment(run_dir, name, *, host, rank, segment_size, role, model_key="m3", server=""):
    """A record as a rank would have left it, under a name we control.

    `server` is the run directory the writing server was given, relative to the
    pool's own. Empty means the two are the same, which is the single-server
    case; a job gives each of its servers one of its own.
    """
    directory = segments_dir(os.path.join(str(run_dir), server))
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, name), "w") as handle:
        json.dump(
            {
                "host": host,
                "rank": rank,
                "segment_size": segment_size,
                "role": role,
                "model_key": model_key,
            },
            handle,
        )


def build_2p5d(run_dir, *, prefill_segment=160 * GIB, decode_segment=160 * GIB):
    """2 context servers at DP2 and 5 generation servers at TP4.

    24 ranks, which at 160GiB each is the 3840GiB pool this topology is
    expected to produce. Each server writes into a run directory of its own,
    which is the layout the report has to read.
    """
    for server in range(2):
        for rank in range(2):
            write_segment(
                run_dir,
                f"segment-prefill{server}-{rank}.json",
                host=f"10.0.1.{server}",
                rank=rank,
                segment_size=prefill_segment,
                role="both",
                server=f"mooncake_CTX_{server}",
            )
    for server in range(5):
        for rank in range(4):
            write_segment(
                run_dir,
                f"segment-decode{server}-{rank}.json",
                host=f"10.0.2.{server}",
                rank=rank,
                segment_size=decode_segment,
                role="capacity",
                server=f"mooncake_GEN_{server}",
            )


# ---- recording ----


def test_a_rank_records_what_it_mounted(tmp_path):
    path = record_segment(
        str(tmp_path),
        host="10.0.0.1",
        rank=3,
        segment_size=160 * GIB,
        role="capacity",
        model_key="MiniMax-M3",
    )
    assert path is not None

    (record,) = read_segments(str(tmp_path))
    assert record.host == "10.0.0.1"
    assert record.rank == 3
    assert record.segment_size == 160 * GIB
    assert record.role == "capacity"
    assert record.model_key == "MiniMax-M3"


def test_records_are_keyed_by_process_not_by_rank(tmp_path):
    """Several servers share a run directory and each numbers its ranks from 0."""
    first = record_segment(
        str(tmp_path), host="10.0.0.1", rank=0, segment_size=GIB, role="both", model_key="m"
    )
    second = record_segment(
        str(tmp_path), host="10.0.0.2", rank=0, segment_size=GIB, role="both", model_key="m"
    )
    assert first != second
    assert len(read_segments(str(tmp_path))) == 2


def test_records_are_gathered_from_every_server_in_a_job(tmp_path):
    """A job's servers each write into a run directory of their own.

    They have to: every server joining the pool renders a client config, and
    two sharing a directory would leave whichever started second in charge of
    both. So the records are read from the whole tree beneath the pool's
    directory, not from one directory in it.
    """
    write_segment(
        tmp_path,
        "prefill.json",
        host="10.0.1.0",
        rank=0,
        segment_size=GIB,
        role="both",
        server="mooncake_CTX_0",
    )
    write_segment(
        tmp_path,
        "decode.json",
        host="10.0.2.0",
        rank=0,
        segment_size=GIB,
        role="capacity",
        server="mooncake_GEN_0",
    )

    assert [record.role for record in read_segments(str(tmp_path))] == ["both", "capacity"]


def test_nothing_is_recorded_without_a_run_directory(tmp_path):
    """A per-process temporary directory is nowhere anything else could read."""
    assert (
        record_segment(None, host="h", rank=0, segment_size=GIB, role="both", model_key="m") is None
    )


def test_a_record_is_never_read_half_written(tmp_path):
    record_segment(str(tmp_path), host="h", rank=0, segment_size=GIB, role="both", model_key="m")
    leftovers = [n for n in os.listdir(segments_dir(str(tmp_path))) if n.endswith(".partial")]
    assert leftovers == []


def test_an_unwritable_run_directory_does_not_fail_the_server(tmp_path, monkeypatch):
    """Losing a line of telemetry is not worth failing a healthy server."""

    def refuse(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(os, "makedirs", refuse)
    assert (
        record_segment(
            str(tmp_path), host="h", rank=0, segment_size=GIB, role="both", model_key="m"
        )
        is None
    )


def test_an_unreadable_record_is_skipped_rather_than_fatal(tmp_path):
    write_segment(tmp_path, "good.json", host="h", rank=0, segment_size=GIB, role="both")
    with open(os.path.join(segments_dir(str(tmp_path)), "bad.json"), "w") as handle:
        handle.write("{not json")

    assert len(read_segments(str(tmp_path))) == 1


def test_no_records_reads_as_no_records(tmp_path):
    assert read_segments(str(tmp_path)) == []


# ---- the report ----


def test_the_report_totals_the_pool(tmp_path):
    build_2p5d(tmp_path)
    report = format_pool_report(str(tmp_path))

    assert "ranks         : 24 across 7 host(s)" in report
    assert "capacity      : 3840.0 GiB" in report
    # The whole point of putting decode in the pool: it holds most of it.
    assert "role capacity :  20 rank(s),   3200.0 GiB   83.3%" in report
    assert "role both     :   4 rank(s),    640.0 GiB   16.7%" in report


def test_the_report_confirms_uniform_contribution(tmp_path):
    build_2p5d(tmp_path)
    assert "per rank      : 160.0 GiB, uniform" in format_pool_report(str(tmp_path))


def test_the_report_names_a_non_uniform_contribution(tmp_path):
    """No server can see another's segment_size, so this is the only place it shows."""
    build_2p5d(tmp_path, decode_segment=80 * GIB)
    report = format_pool_report(str(tmp_path))

    assert "NOT UNIFORM" in report
    assert "80.0 GiB" in report
    assert "160.0 GiB" in report
    assert "same segment_size" in report


def test_the_report_says_plainly_when_the_pool_was_empty(tmp_path):
    """A pool nothing joined serves no lookup, and otherwise just looks quiet."""
    report = format_pool_report(str(tmp_path))
    assert "nothing recorded" in report
    assert "mooncake_store" in report


def test_the_report_names_the_master_from_the_manifest(tmp_path):
    build_2p5d(tmp_path)
    (tmp_path / "pool.json").write_text(
        json.dumps(
            {
                "master_server_address": "10.66.5.9:50051",
                "protocol": "rdma",
                "metadata_server": "P2PHANDSHAKE",
            }
        )
    )
    report = format_pool_report(str(tmp_path))
    assert "master        : 10.66.5.9:50051" in report
    assert "transport     : rdma" in report


def test_the_report_survives_a_missing_manifest(tmp_path):
    """The capacity half is independent of it, so it is still worth printing."""
    build_2p5d(tmp_path)
    report = format_pool_report(str(tmp_path))
    assert "master        : unknown" in report
    assert "capacity      : 3840.0 GiB" in report


# ---- joining the master's own view against the records ----


def test_placement_is_grouped_by_host_and_labelled_by_role(tmp_path):
    """The question the pool exists for: did prefill's writes leave its nodes?"""
    build_2p5d(tmp_path)
    master_log = tmp_path / "mooncake_master.log"
    master_log.write_text(
        "\n".join(
            [
                "I0903 allocation_succeeded size=1048576 segment=10.0.2.0:14141",
                "I0903 allocation_succeeded size=1048576 segment=10.0.2.0:14141",
                "I0903 allocation_succeeded size=1048576 segment=10.0.2.1:14142",
                "I0903 allocation_succeeded size=1048576 segment=10.0.1.0:13001",
                "I0903 something_else_entirely",
            ]
        )
    )

    report = format_pool_report(str(tmp_path))
    assert "10.0.2.0" in report
    assert "capacity" in report
    # Three of four placements landed on capacity-only hosts.
    assert "held off the producing side: 75.0%" in report


def test_placement_reports_the_segment_ports_behind_each_host(tmp_path):
    """One host can hold several segments; the ports are how they are told apart."""
    build_2p5d(tmp_path)
    (tmp_path / "mooncake_master.log").write_text(
        "allocation_succeeded size=100 segment=10.0.2.0:14141\n"
        "allocation_succeeded size=100 segment=10.0.2.0:14142\n"
    )
    assert "segments: 14141,14142" in format_pool_report(str(tmp_path))


def test_a_master_log_with_no_placements_says_so(tmp_path):
    build_2p5d(tmp_path)
    (tmp_path / "mooncake_master.log").write_text("I0903 starting up\n")
    assert "(no allocations in the master log)" in format_pool_report(str(tmp_path))


def test_an_absent_master_log_is_not_an_error(tmp_path):
    build_2p5d(tmp_path)
    report = format_pool_report(str(tmp_path), master_log=str(tmp_path / "absent.log"))
    assert "could not be read" in report


def test_a_host_the_records_do_not_know_is_still_reported(tmp_path):
    """An engine outside this run may share the pool; do not drop its bytes."""
    build_2p5d(tmp_path)
    (tmp_path / "mooncake_master.log").write_text(
        "allocation_succeeded size=100 segment=10.9.9.9:15000\n"
    )
    report = format_pool_report(str(tmp_path))
    assert "10.9.9.9" in report
    assert "unknown role" in report
