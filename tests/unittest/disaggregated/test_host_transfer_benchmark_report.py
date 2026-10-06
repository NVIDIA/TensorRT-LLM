# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate host-transfer benchmark evidence and GPU baseline comparison."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only


def _report_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/disaggregated/slurm/cache_transceiver_test/report.py"
    )
    spec = importlib.util.spec_from_file_location("cache_transceiver_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _samples(seconds: float) -> list[dict]:
    return [
        {
            "scenario": "throughput",
            "sample": 0,
            "warmup": False,
            "seconds": seconds,
            "bytes": 1_000_000_000,
            "byte_correct": True,
            "logical_cancel": False,
        },
        {
            "scenario": "delayed_completion",
            "sample": 0,
            "warmup": False,
            "seconds": seconds,
            "bytes": 1_000_000_000,
            "byte_correct": True,
            "logical_cancel": False,
        },
        {
            "scenario": "logical_cancel",
            "sample": 0,
            "warmup": False,
            "seconds": seconds,
            "bytes": 1_000_000_000,
            "byte_correct": True,
            "logical_cancel": True,
        },
    ]


def _write(root: Path, directory: str, mode: str, samples: list[dict]) -> None:
    target = root / directory
    target.mkdir(exist_ok=True)
    (target / f"sweep0_rank0_{mode}.json").write_text(json.dumps({"samples": samples}))


@pytest.mark.parametrize(
    "section,directory",
    [
        ("host_transfer_benchmark", "host_transfer"),
        ("kvcm_transfer_benchmark", "kvcm_transfer"),
    ],
)
def test_host_bandwidth_compares_only_verified_samples_with_gpu(
    tmp_path: Path, section: str, directory: str
) -> None:
    config = {
        "environment": {"work_dir": str(tmp_path)},
        "hardware": {"gpus_per_node": 1},
        "ucx_env_sweep": [{"name": "default"}],
        section: {"enabled": True, "modes": ["gpu", "host"]},
    }
    _write(tmp_path, directory, "gpu", _samples(1))
    _write(tmp_path, directory, "host", _samples(2))
    modes = _report_module()._aggregate_transfer_evidence(config, section, directory)[0]["modes"]
    assert modes[0]["per_gpu_GBps"] == 1
    assert modes[1]["per_gpu_GBps"] == 0.5
    assert modes[1]["ratio_to_gpu"] == 0.5


@pytest.mark.parametrize("missing", ["byte_correct", "logical_cancel", "delayed_completion"])
@pytest.mark.parametrize(
    "section,directory",
    [
        ("host_transfer_benchmark", "host_transfer"),
        ("kvcm_transfer_benchmark", "kvcm_transfer"),
    ],
)
def test_host_bandwidth_rejects_unverified_cases(
    tmp_path: Path, missing: str, section: str, directory: str
) -> None:
    config = {
        "environment": {"work_dir": str(tmp_path)},
        "hardware": {"gpus_per_node": 1},
        "ucx_env_sweep": [{"name": "default"}],
        section: {"enabled": True, "modes": ["gpu", "host"]},
    }
    _write(tmp_path, directory, "gpu", _samples(1))
    samples = _samples(2)
    if missing == "byte_correct":
        samples[0]["byte_correct"] = False
    else:
        samples = [sample for sample in samples if sample["scenario"] != missing]
    _write(tmp_path, directory, "host", samples)
    host = _report_module()._aggregate_transfer_evidence(config, section, directory)[0]["modes"][1]
    assert host["status"] == "ERROR"
    assert host["per_gpu_GBps"] is None


def test_kvcm_report_rejects_a_previous_slurm_jobs_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {
        "environment": {"work_dir": str(tmp_path)},
        "hardware": {"gpus_per_node": 1},
        "ucx_env_sweep": [{"name": "default"}],
        "kvcm_transfer_benchmark": {"enabled": True, "modes": ["gpu", "host"]},
    }
    _write(tmp_path, "kvcm_transfer", "gpu", _samples(1))
    _write(tmp_path, "kvcm_transfer", "host", _samples(2))
    monkeypatch.setenv("SLURM_JOB_ID", "current-job")
    modes = _report_module()._aggregate_kvcm_transfer(config)[0]["modes"]
    assert all(mode["status"] == "ERROR" for mode in modes)
