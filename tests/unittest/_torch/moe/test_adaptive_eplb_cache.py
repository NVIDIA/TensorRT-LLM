# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the adaptive EPLB calibration cache: its key and its eviction."""

import json
import os
from pathlib import Path

import numpy as np
import pynvml
import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import calibrate
from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig

pytestmark = pytest.mark.cpu_only

IDENTITY = dict(device="GPU", ep_size=4, tactics={"8": "a"})


@pytest.fixture
def key(monkeypatch, tmp_path):
    """The cache file name of a run whose package keeps ``libs/libth_common.so`` under ``tmp_path``,
    on a stubbed NVML (driver 580.178.04, CUDA 13.0); ``key.lib`` is that library."""
    package = tmp_path / "tensorrt_llm"
    (package / "libs").mkdir(parents=True)
    monkeypatch.setattr(tensorrt_llm, "__file__", str(package / "__init__.py"))
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlSystemGetDriverVersion", lambda: b"580.178.04")
    monkeypatch.setattr(pynvml, "nvmlSystemGetCudaDriverVersion", lambda: 13000)
    config = MoeAdaptiveLoadBalancerConfig(num_slots=16, calibration_cache_dir=str(tmp_path))

    def name():
        return calibrate.cache_path(config, IDENTITY).name

    name.lib = package / "libs" / "libth_common.so"
    return name


def test_the_same_library_reinstalled_keeps_the_key_and_other_bytes_change_it(key):
    key.lib.write_bytes(b"kernels" * 1000)
    before = key()
    reinstalled = key.lib.with_name("new.so")  # a new file: another inode and write time
    reinstalled.write_bytes(b"kernels" * 1000)
    os.utime(reinstalled, ns=(1, 1))
    os.replace(reinstalled, key.lib)
    assert key() == before
    key.lib.write_bytes(b"kernelz" * 1000)  # one byte of another build, same size
    assert key() != before


@pytest.mark.parametrize(
    "module, name, value",
    [
        (pynvml, "nvmlSystemGetDriverVersion", lambda: b"580.200.01"),
        (pynvml, "nvmlSystemGetCudaDriverVersion", lambda: 13010),
        (torch.version, "cuda", "12.9"),
        (np, "__version__", "0.0.1"),
    ],
)
def test_a_driver_cuda_or_numpy_change_makes_a_new_key(key, monkeypatch, module, name, value):
    key.lib.write_bytes(b"kernels")
    before = key()
    monkeypatch.setattr(module, name, value)
    assert key() != before


def _entry(folder: Path, n: int, schema=calibrate.CALIBRATION[0]) -> Path:
    """A calibration file of write time ``n`` s with its times, named like the cache's."""
    path = folder / f"{calibrate.digest(n)}.json"
    path.write_text(json.dumps(dict(schema=schema, version=1)))
    path.with_suffix(".times.json").write_text("{}")
    for part in (path, path.with_suffix(".times.json")):
        os.utime(part, ns=(n * 10**9, n * 10**9))
    return path


def test_eviction_keeps_the_newest_and_never_touches_another_file(tmp_path):
    old = [_entry(tmp_path, n) for n in range(1, 6)]
    written = _entry(tmp_path, 0)  # the file just written, whatever its write time says
    foreign = [_entry(tmp_path, 9, schema="another.tool")]
    foreign += [tmp_path / f"{'0' * 64}.json", tmp_path / "notes.json", tmp_path / "a.txt"]
    foreign[1].write_text("not json")
    foreign[2].write_text(json.dumps(dict(schema=calibrate.CALIBRATION[0])))  # not a cache name
    foreign[3].write_text("")
    everything = set(tmp_path.iterdir())
    assert calibrate.evict(written, 0) == [] and set(tmp_path.iterdir()) == everything
    assert calibrate.evict(written, 3) == []
    kept = [written, *old[-2:]]
    gone = {p for e in old[:-2] for p in (e, e.with_suffix(".times.json"))}
    assert set(tmp_path.iterdir()) == everything - gone
    assert all(p.with_suffix(".times.json").exists() for p in kept + foreign[:1])


def test_eviction_counts_times_without_their_calibration_by_their_own_write_time(tmp_path):
    cal = {n: _entry(tmp_path, n) for n in range(1, 5)}
    written = _entry(tmp_path, 0)
    for n in (2, 4):  # no bucket held: the times stay alone
        cal[n].unlink()
    foreign = [tmp_path / f"{calibrate.digest(n)}.times.json" for n in ("list", "layout")]
    foreign[0].write_text("[]")
    foreign[1].write_text(json.dumps({"8": 1.0}))
    assert calibrate.evict(written, 3) == []
    left = [written, cal[3], *(p.with_suffix(".times.json") for p in (written, cal[3], cal[4]))]
    assert sorted(tmp_path.iterdir()) == sorted(left + foreign)


def test_eviction_tolerates_a_concurrent_delete_and_reports_a_failed_one(tmp_path, monkeypatch):
    old, written = [_entry(tmp_path, n) for n in range(1, 4)], _entry(tmp_path, 9)
    unlink = Path.unlink

    def racing(path, *args, **kwargs):
        if path == old[0]:  # another process evicted it after this one listed it
            unlink(path)
        if path == old[1]:
            raise PermissionError("read-only")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", racing)
    (failed,) = calibrate.evict(written, 1)
    assert str(old[1]) in failed and "PermissionError" in failed
    left = [written, written.with_suffix(".times.json"), old[1]]
    assert sorted(tmp_path.iterdir()) == sorted(left)
