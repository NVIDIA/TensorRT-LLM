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

import json
import os
import pstats
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch

from trtllm_profile import Profiler, active_session, component, enable_from_argv, trace_active
from trtllm_profile.resources import ResourceSampler, amd_gpu_snapshot

pytestmark = pytest.mark.cpu_only


def test_disabled_has_no_trace_or_sampler() -> None:
    before = {thread.ident for thread in threading.enumerate()}
    args = ["application", "--prompt", "hello"]
    assert enable_from_argv(args) is None
    assert active_session() is None
    assert args == ["application", "--prompt", "hello"]
    assert {thread.ident for thread in threading.enumerate()} == before


def test_cpu_components_files_floor_and_resource_reports(tmp_path) -> None:
    profiler = Profiler(tmp_path / "report", interval=0.01, calibration_ops=16)
    with profiler:
        with component("projections"):
            result = torch.ones(4, 4) @ torch.ones(4, 4)
        assert result.sum().item() == 64
    report = json.loads((tmp_path / "report.json").read_text())
    projections = next(row for row in report["components"] if row["component"] == "projections")
    assert projections["ops"] > 0 and projections["host_us"] > 0
    assert projections["dev_us"] is None and projections["percent_dev"] is None
    assert report["calibration"][0]["ops"] == 16
    assert report["calibration"][0]["dev_us_per_op"] is None
    assert any(Path(file["source"]).name == "test_profile.py" for file in report["files"])
    assert report["resources"]["summary"]["process_rss_bytes"]["peak"] > 0
    assert report["resources"]["summary"]["ram_total_bytes"]["peak"] > 0
    text = (tmp_path / "report.txt").read_text()
    assert "instrumentation floor" in text and "CPU system %" in text
    assert "VRAM" in text and "N/A" in text
    assert pstats.Stats(str(tmp_path / "report.prof")).total_calls > 0
    assert json.loads((tmp_path / "report.trace.json").read_text())["traceEvents"]
    assert profiler.finish() is profiler.report_data
    assert active_session() is None


def test_trace_cap_does_not_stop_application_or_resource_sampling(tmp_path) -> None:
    profiler = Profiler(tmp_path / "capped", max_ops=2, calibration_ops=2)
    with profiler:
        for _ in range(20):
            torch.ones(3) + 1
    assert profiler.report_data["recorded_ops"] == 2
    assert profiler.report_data["dropped_ops"] > 0
    assert "truncated" in (tmp_path / "capped.txt").read_text()


def test_worker_thread_is_enrolled_and_component_is_thread_local(tmp_path) -> None:
    profiler = Profiler(tmp_path / "workers", calibration_ops=2)

    def worker() -> None:
        with trace_active(), component("attention"):
            torch.ones(2) + 1

    with profiler:
        with component("projections"):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            torch.ones(2) + 2
    labels = {row["component"] for row in profiler.report_data["components"]}
    assert "attention" in labels and "projections" in labels
    functions = [
        function["name"] for file in profiler.report_data["files"] for function in file["functions"]
    ]
    # cProfile is enabled inside the worker, after its initial call, so its
    # nested PyTorch calls and source-operation attribution provide coverage.
    assert functions
    assert len(profiler._profiles) == 2


def test_failure_still_writes_report(tmp_path) -> None:
    with pytest.raises(ValueError, match="test failure"):
        with Profiler(tmp_path / "failure", tensors=False, calibration_ops=2) as profiler:
            with profiler.operation("attention"):
                raise ValueError("test failure")
    report = json.loads((tmp_path / "failure.json").read_text())
    assert report["status"] == "failed: ValueError"
    assert report["components"][0]["ops"] == 1
    assert active_session() is None


def test_argv_profile_options_are_consumed_and_separator_protected(tmp_path) -> None:
    args = [
        "program",
        "--profile",
        "--profile-output",
        str(tmp_path / "flags"),
        "--profile-no-tensors",
        "--profile-max-ops=3",
        "--model",
        "my-model",
    ]
    profiler = enable_from_argv(args)
    try:
        assert args == ["program", "--model", "my-model"]
        assert profiler.max_ops == 3
        assert not profiler.tensors
    finally:
        profiler.finish(print_report=False)
    protected = ["program", "--", "--profile"]
    assert enable_from_argv(protected) is None
    assert protected == ["program", "--", "--profile"]


def test_standalone_launcher_profiles_an_unmodified_script_and_preserves_exit(tmp_path) -> None:
    target = tmp_path / "plain_script.py"
    target.write_text(
        "import sys\nassert sys.argv[1:] == ['--value', 'hello']\nraise SystemExit(7)\n"
    )
    output = tmp_path / "standalone"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/profile.py",
            str(target),
            "--profile",
            "--profile-no-tensors",
            "--profile-output",
            str(output),
            "--",
            "--value",
            "hello",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 7, result.stderr
    report = json.loads(Path(f"{output}.json").read_text())
    assert report["status"] == "failed: exit 7"
    assert any(file["source"] == str(target) for file in report["files"])


def test_direct_import_bootstrap_accepts_profile_before_argparse(tmp_path) -> None:
    target = tmp_path / "library_script.py"
    target.write_text(
        "from tensorrt_llm import SamplingParams\nimport argparse\n"
        "parser=argparse.ArgumentParser()\nparser.add_argument('--value')\n"
        "assert parser.parse_args().value == 'hello'\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(target),
            "--profile",
            "--profile-output",
            str(tmp_path / "direct"),
            "--value",
            "hello",
        ],
        env={**os.environ, "PYTHONPATH": str(Path.cwd()), "TRTLLM_BACKEND": "rocm"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "direct.json").is_file()


def test_amd_sysfs_identity_and_missing_counters(tmp_path) -> None:
    card = tmp_path / "card10" / "device"
    card.mkdir(parents=True)
    for key, value in {
        "vendor": "0x1002",
        "gpu_busy_percent": "73",
        "mem_info_vram_used": "512",
        "mem_info_vram_total": "1024",
    }.items():
        (card / key).write_text(value)
    connector = tmp_path / "card10-DP-1" / "device"
    connector.mkdir(parents=True)
    (connector / "vendor").write_text("0x1002")
    devices = amd_gpu_snapshot(tmp_path)
    assert len(devices) == 1
    assert devices[0]["gpu_percent"] == 73
    assert devices[0]["vram_percent"] == 50
    assert devices[0]["memory_busy_percent"] is None
    assert devices[0]["id"].startswith("drm:card10:")
    (card / "gpu_busy_percent").unlink()
    assert amd_gpu_snapshot(tmp_path)[0]["gpu_percent"] is None


def test_bounded_raw_samples_keep_all_peak_and_mean_statistics(tmp_path, monkeypatch) -> None:
    sampler = ResourceSampler(max_samples=2, drm_root=tmp_path)
    values = iter([10, 90, 20, 30])
    monkeypatch.setattr(sampler, "_host_snapshot", lambda: {"cpu_system_percent": next(values)})
    for _ in range(4):
        sampler.sample()
    report = sampler.report()
    assert report["sample_count"] == 4 and report["discarded_raw_samples"] == 2
    assert report["summary"]["cpu_system_percent"]["peak"] == 90
    assert report["summary"]["cpu_system_percent"]["mean"] == 37.5


def test_default_floor_and_synthetic_device_trace_use_numeric_ids(tmp_path) -> None:
    from types import SimpleNamespace

    profiler = Profiler(tmp_path / "numeric", tensors=False)
    with profiler:
        with profiler.operation("attention"):
            pass
        record = profiler._records[-1]
        record.span = SimpleNamespace(resolve=lambda: (3.0, 1.0, record.start_ns / 1000, 0, 17))
    assert profiler.report_data["calibration"][0]["ops"] == 256
    trace = json.loads((tmp_path / "numeric.trace.json").read_text())["traceEvents"]
    assert all(isinstance(event["pid"], int) and isinstance(event["tid"], int) for event in trace)
    assert any(event["cat"] == "device event span" for event in trace)


def test_device_metadata_survives_property_failure(monkeypatch) -> None:
    from trtllm_profile.torch_trace import EventRecorder

    def unavailable(_device):
        raise RuntimeError("device unavailable")

    recorder = EventRecorder()
    recorder.devices = [0]
    monkeypatch.setattr(torch.cuda, "get_device_properties", unavailable)
    metadata = recorder.metadata()
    assert metadata[0]["logical_index"] == 0
    assert metadata[0]["telemetry_error"] == "device unavailable"
    assert "total_bytes" not in metadata[0]


def test_failed_device_synchronization_writes_reports_then_propagates(tmp_path) -> None:
    from types import SimpleNamespace

    def unavailable():
        raise RuntimeError("deferred GPU failure")

    profiler = Profiler(tmp_path / "device_failure", tensors=False, calibration_ops=2)
    profiler.start()
    profiler._recorder = SimpleNamespace(synchronize=unavailable, metadata=lambda: [])
    with pytest.raises(RuntimeError, match="deferred GPU failure"):
        profiler.finish(print_report=False)
    report = json.loads((tmp_path / "device_failure.json").read_text())
    assert report["status"] == "failed: device synchronization"
    assert active_session() is None
    assert not profiler.reserve()


def test_pid_guard_does_not_expose_an_inherited_session(tmp_path) -> None:
    with Profiler(tmp_path / "pid", tensors=False, calibration_ops=2) as profiler:
        pid = profiler.pid
        try:
            profiler.pid = -1
            assert active_session() is None
        finally:
            profiler.pid = pid


def test_async_worker_with_copied_context_is_still_enrolled(tmp_path) -> None:
    import asyncio

    profiler = Profiler(tmp_path / "async-worker", calibration_ops=2)

    def worker() -> None:
        with trace_active(), component("attention"):
            torch.ones(4) + 1

    async def run() -> None:
        await asyncio.to_thread(worker)

    with profiler:
        asyncio.run(run())
    assert len(profiler._profiles) == 2
    assert any(row["component"] == "attention" for row in profiler.report_data["components"])


def test_offline_validator_works_with_inference_and_dispatch_profiling(tmp_path) -> None:
    from tensorrt_llm.rocm.validation import validate

    with Profiler(tmp_path / "validation", calibration_ops=2) as profiler:
        report = validate("cpu", "torch", ("float32",))
    assert report["passed"] and not report["native_kernels_executed"]
    assert profiler.report_data["recorded_ops"] > 0
