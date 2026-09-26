# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the disaggregated request lifecycle JSONL events."""

import importlib.util
import json
import os
import pathlib
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.disaggregation.native.perf_logger import LIFECYCLE_EVENTS, PerfLogManager

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _fresh_manager(monkeypatch, output_dir):
    """Build a PerfLogManager as if the process started with the given env."""
    if output_dir is None:
        monkeypatch.delenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", str(output_dir))
    monkeypatch.delenv("TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO", raising=False)
    monkeypatch.setattr(PerfLogManager, "_instance", None)
    return PerfLogManager()


def _request(local_id, disagg_id=None, prompt_len=8):
    params = None if disagg_id is None else SimpleNamespace(disagg_request_id=disagg_id)
    return SimpleNamespace(
        py_request_id=local_id,
        py_disaggregated_params=params,
        prompt_len=prompt_len,
    )


def _read_events(output_dir):
    events = []
    for path in sorted(pathlib.Path(output_dir).glob("lifecycle_*.jsonl")):
        with open(path) as f:
            events.extend(json.loads(line) for line in f if line.strip())
    return events


def test_disabled_without_output_path_writes_nothing(monkeypatch, tmp_path):
    manager = _fresh_manager(monkeypatch, None)
    assert not manager.lifecycle_enabled
    manager.event("gen_ingress", _request(1, 100))
    assert list(tmp_path.iterdir()) == []


def test_enabled_writes_one_jsonl_record_per_event(monkeypatch, tmp_path):
    manager = _fresh_manager(monkeypatch, tmp_path)
    manager.configure_identity(rank=3, instance="gen0")
    manager.event("gen_ingress", _request(7, disagg_id=4242), prompt_len=8)
    manager.event("settled", 4242, side="gen", outcome="completed")

    events = _read_events(tmp_path)
    assert [e["event"] for e in events] == ["lifecycle_start", "gen_ingress", "settled"]
    ingress, settled = events[1], events[2]
    assert ingress["rid"] == 4242 and ingress["local_id"] == 7
    assert ingress["rank"] == 3 and ingress["instance"] == "gen0"
    assert ingress["pid"] == os.getpid() and ingress["prompt_len"] == 8
    # A bare rid is accepted where no LlmRequest is at hand (session retirement).
    assert settled["rid"] == 4242 and settled["local_id"] is None
    assert settled["outcome"] == "completed"
    assert settled["t_steady"] >= ingress["t_steady"]
    assert os.path.basename(list(tmp_path.glob("lifecycle_*.jsonl"))[0]) == (
        f"lifecycle_rank3_pid{os.getpid()}.jsonl"
    )


def test_request_without_disagg_params_falls_back_to_local_id(monkeypatch, tmp_path):
    manager = _fresh_manager(monkeypatch, tmp_path)
    manager.event("ctx_send_ready", _request(11))
    (_, rec) = _read_events(tmp_path)
    assert rec["rid"] == 11 and rec["local_id"] == 11


def test_event_never_raises_into_the_caller(monkeypatch, tmp_path):
    manager = _fresh_manager(monkeypatch, tmp_path)

    class Explosive:
        @property
        def py_request_id(self):
            raise RuntimeError("boom")

    manager.event("gen_ingress", Explosive())  # must not raise
    manager.event("gen_ingress", _request(1, 2), unserializable=object())  # default=str
    events = _read_events(tmp_path)
    assert [e["event"] for e in events][-1] == "gen_ingress"


def test_existing_csv_gating_is_unchanged(monkeypatch, tmp_path):
    manager = _fresh_manager(monkeypatch, tmp_path)
    assert manager.enabled and manager.use_file
    monkeypatch.setattr(PerfLogManager, "_instance", None)
    monkeypatch.delenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH")
    monkeypatch.setenv("TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO", "1")
    legacy = PerfLogManager()
    assert legacy.enabled and not legacy.lifecycle_enabled


def test_event_names_are_documented():
    assert len(set(LIFECYCLE_EVENTS)) == len(LIFECYCLE_EVENTS)
    assert {"gen_ingress", "gen_kv_admission", "gen_transfer_window", "settled"} <= set(
        LIFECYCLE_EVENTS
    )


def _load_timeline_script():
    path = _REPO_ROOT / "scripts" / "disagg_lifecycle_timeline.py"
    spec = importlib.util.spec_from_file_location("disagg_lifecycle_timeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timeline_script_orders_events_and_reports_gaps(monkeypatch, tmp_path, capfd):
    manager = _fresh_manager(monkeypatch, tmp_path)
    manager.configure_identity(rank=0, instance="gen0")
    req = _request(1, disagg_id=99)
    manager.event("gen_ingress", req)
    manager.event("gen_kv_admission", req, admitted=False)
    manager.event("gen_kv_admission", req, admitted=True)
    manager.event("gen_transfer_window", req, admitted=True, active_blocks=0, budget_blocks=8)
    manager.event("settled", 99, side="gen", outcome="completed")
    manager.event("gen_decode_ready", req)
    # A ctx-side record from another process for the same rid.
    with open(tmp_path / "lifecycle_rank0_pid1.jsonl", "w") as f:
        f.write(
            json.dumps(
                {
                    "event": "settled",
                    "rid": 99,
                    "side": "ctx",
                    "outcome": "completed",
                    "t_steady": 0.0,
                    "t_wall": 0.0,
                    "rank": 0,
                    "pid": 1,
                }
            )
            + "\n"
        )
        f.write("not json\n")

    script = _load_timeline_script()
    events, malformed = script.load_events(sorted(tmp_path.glob("lifecycle_*.jsonl")))
    assert malformed == 1
    timelines = script.build_timelines(events)
    labels = [row["event"] for row in timelines[99]]
    assert labels == [
        "ctx:settled[completed]",
        "gen_ingress",
        "gen_kv_admission[False]",
        "gen_kv_admission[True]",
        "gen_transfer_window[True]",
        "gen:settled[completed]",
        "gen_decode_ready",
    ]
    assert timelines[99][0]["gap_ms"] is None
    assert timelines[99][1]["cross_process"] is True
    assert all(row["gap_ms"] >= 0 for row in timelines[99][1:])

    stats = script.transition_stats(timelines)
    pairs = {(s["from"], s["to"]) for s in stats}
    assert ("gen_kv_admission[True]", "gen_transfer_window[True]") in pairs

    assert script.main([str(tmp_path), "--rid", "99", "--json", str(tmp_path / "out.json")]) == 0
    out = capfd.readouterr().out
    assert "rid=99 (7 events)" in out and "transitions" in out
    dumped = json.load(open(tmp_path / "out.json"))
    assert set(dumped) == {"requests", "transitions"}


def test_timeline_script_handles_missing_directory(tmp_path):
    script = _load_timeline_script()
    assert script.main([str(tmp_path / "empty")]) == 1
