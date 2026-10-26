# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

# This module is pure Python and can also be tested without a built TRT-LLM wheel.
_MODULE_PATH = (
    Path(__file__).resolve().parents[5] / "tensorrt_llm/_torch/pyexecutor/kv_cache/memory_utils.py"
)
_SPEC = importlib.util.spec_from_file_location("kv_memory_utils", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
memory_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(memory_utils)

pytestmark = pytest.mark.cpu_only
_GiB = 1 << 30


def _write_cgroup(
    path: Path, limit: int | str, current: int, file_bytes: int = 0, shmem: int = 0
) -> None:
    """Populate a synthetic cgroup v2 directory."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "memory.max").write_text(f"{limit}\n")
    (path / "memory.current").write_text(f"{current}\n")
    (path / "memory.stat").write_text(f"anon 123\nfile {file_bytes}\nshmem {shmem}\nkernel 0\n")


@pytest.mark.parametrize("value,expected", [("1000", 1024000), ("0", 0), ("-1", 0), ("bad", None)])
def test_meminfo_available(tmp_path: Path, value: str, expected: int | None) -> None:
    """Parse MemAvailable and reject malformed values."""
    path = tmp_path / "meminfo"
    path.write_text(f"MemFree: 1 kB\nMemAvailable: {value} kB\n")
    assert memory_utils._read_meminfo_available_bytes(str(path)) == expected


@pytest.mark.parametrize("content", [None, "MemTotal: 1 kB\n", "MemAvailable:\n"])
def test_meminfo_unavailable(tmp_path: Path, content: str | None) -> None:
    """Missing files or fields return no estimate."""
    path = tmp_path / "meminfo"
    if content is not None:
        path.write_text(content)
    assert memory_utils._read_meminfo_available_bytes(str(path)) is None


@pytest.mark.parametrize(
    "limit,current,file_bytes,shmem,expected",
    [
        (100, 90, 60, 0, 70),
        (100, 90, 80, 70, 20),
        (100, 90, 30, 40, 10),
        (100, 120, 5, 0, 0),
        (100, 90, 200, 0, 100),
        ("max", 90, 80, 0, None),
    ],
)
def test_cgroup_reclaimable_memory(
    tmp_path: Path,
    limit: int | str,
    current: int,
    file_bytes: int,
    shmem: int,
    expected: int | None,
) -> None:
    """Shared memory never increases the reclaimable-file estimate."""
    _write_cgroup(tmp_path, limit, current, file_bytes, shmem)
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/\n")
    assert memory_utils._read_cgroup_memory_headroom_bytes(str(tmp_path), str(proc)) == expected


@pytest.mark.parametrize("stat", [None, "file 80\n", "file 80\nshmem bad\n", "malformed\n"])
def test_cgroup_missing_stats_preserve_limit(tmp_path: Path, stat: str | None) -> None:
    """Unreadable reclaimable stats cannot discard a finite memory limit."""
    _write_cgroup(tmp_path, 100, 90)
    path = tmp_path / "memory.stat"
    path.unlink()
    if stat is not None:
        path.write_text(stat)
    assert (
        memory_utils._read_cgroup_memory_headroom_bytes(str(tmp_path), str(tmp_path / "absent"))
        == 10
    )


@pytest.mark.parametrize(
    "limits,expected", [(("max", "max", 40), 20), ((200, 50, 100), 30), ((30, 100, 200), 10)]
)
def test_cgroup_process_and_ancestors(
    tmp_path: Path, limits: tuple[int | str, ...], expected: int
) -> None:
    """The tightest headroom can be at the process, parent, or mount root."""
    root = tmp_path / "cgroup"
    for path, limit in zip([root, root / "user.slice", root / "user.slice/session.scope"], limits):
        _write_cgroup(path, limit, 30, 15, 5)
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/user.slice/session.scope\n")
    assert memory_utils._read_cgroup_memory_headroom_bytes(str(root), str(proc)) == expected


@pytest.mark.parametrize("relative", ["/", "/host/container", "/../../outside"])
def test_cgroup_namespaced_root(tmp_path: Path, relative: str) -> None:
    """A private namespace still enforces the limit exposed at its mount root."""
    root = tmp_path / "cgroup"
    _write_cgroup(root, 100, 90, 80, 70)
    proc = tmp_path / "proc_cgroup"
    proc.write_text(f"0::{relative}\n")
    assert memory_utils._read_cgroup_memory_headroom_bytes(str(root), str(proc)) == 20


def test_cgroup_parent_with_sibling_usage(tmp_path: Path) -> None:
    """Each ancestor uses its own usage and cache, including sibling charges."""
    root = tmp_path / "cgroup"
    _write_cgroup(root, 100, 95, 10, 5)
    _write_cgroup(root / "child", 50, 10, 0, 0)
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/child\n")
    assert memory_utils._read_cgroup_memory_headroom_bytes(str(root), str(proc)) == 10


def test_cgroup_unavailable(tmp_path: Path) -> None:
    """No readable cgroup limit leaves the host estimate uncapped."""
    assert (
        memory_utils._read_cgroup_memory_headroom_bytes(str(tmp_path), str(tmp_path / "absent"))
        is None
    )


@pytest.mark.parametrize(
    "cuda_free,cuda_total,available,cgroup,expected",
    [
        (4, 120, 60, None, 60),
        (4, 120, 200, None, 120),
        (4, 120, 60, 20, 20),
        (8, 120, 6, None, 8),
        (8, 120, 60, 2, 2),
        (8, 120, 60, 0, 0),
        (4, 120, None, None, 4),
        (4, 120, None, 2, 2),
    ],
)
def test_integrated_gpu_bounds(
    cuda_free: int, cuda_total: int, available: int | None, cgroup: int | None, expected: int
) -> None:
    """Device capacity and finite cgroup headroom cap even the CUDA fallback."""
    with (
        patch.object(
            memory_utils,
            "_read_meminfo_available_bytes",
            return_value=None if available is None else available * _GiB,
        ),
        patch.object(
            memory_utils,
            "_read_cgroup_memory_headroom_bytes",
            return_value=None if cgroup is None else cgroup * _GiB,
        ),
    ):
        assert (
            memory_utils.get_integrated_gpu_free_memory(cuda_free * _GiB, cuda_total * _GiB)
            == expected * _GiB
        )
