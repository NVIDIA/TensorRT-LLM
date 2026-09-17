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
"""CPU-only cache safety contracts; no real cache eviction or residency calls."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

__extra_import_path__ = ["~/jenkins/scripts"]
from startup_benchmark import cache

pytestmark = pytest.mark.cpu_only
_MINCORE_PROBE = cache._resident_pages


@pytest.fixture(autouse=True)
def fake_cache_operations(monkeypatch: pytest.MonkeyPatch) -> dict[str, str | Exception]:
    monkeypatch.setattr(cache.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cache.os, "posix_fadvise", Mock(), raising=False)
    monkeypatch.setattr(cache.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(cache, "_resident_pages", Mock(return_value=0))
    monkeypatch.setattr(
        cache.subprocess, "run", Mock(side_effect=AssertionError("Unexpected helper"))
    )
    proc: dict[str, str | Exception] = {
        "fdinfo": "pos:\t0\nflags:\t02100000\nmnt_id:\t42\n",
        "mountinfo": "42 1 0:40 / /models ro,relatime shared:3 - nfs4 server:/models rw,vers=4.2\n",
    }
    original_read = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        key = "fdinfo" if path.parent == Path("/proc/self/fdinfo") else "mountinfo"
        if key == "fdinfo" or path == Path("/proc/self/mountinfo"):
            value = proc[key]
            if isinstance(value, Exception):
                raise value
            return value
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    return proc


def _weight(directory: Path, name: str = "model.safetensors") -> Path:
    path = directory / name
    path.write_bytes(b"test checkpoint")
    return path


def test_discover_main_draft_auxiliary_and_deduplicate(tmp_path: Path) -> None:
    main = tmp_path / "main"
    draft = tmp_path / "draft"
    main.mkdir()
    draft.mkdir()
    shard = _weight(main, "model-00001-of-00001.safetensors")
    auxiliary = _weight(main, "auxiliary.pt")
    draft_file = _weight(draft)
    (main / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard.name}}), encoding="utf-8"
    )
    os.link(shard, main / "alias.safetensors")
    discovered = cache.discover_checkpoint_files([main, draft, main])
    assert len(discovered) == 3
    assert auxiliary in discovered and draft_file in discovered
    assert sum(entry["bytes"] for entry in cache.checkpoint_manifest(discovered)) == 45


@pytest.mark.parametrize("indexed", [True, False])
def test_missing_shard_is_rejected(tmp_path: Path, indexed: bool) -> None:
    shard = _weight(tmp_path, "model-00001-of-00002.safetensors")
    if indexed:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": shard.name, "b": "model-00002-of-00002.safetensors"}}),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="Missing .*shard"):
        cache.discover_checkpoint_files([tmp_path])


@pytest.mark.parametrize("layout", ["empty", "empty_weight", "invalid_index", "outside_root"])
def test_invalid_checkpoint_rejected(tmp_path: Path, layout: str) -> None:
    if layout == "empty_weight":
        (tmp_path / "model.safetensors").touch()
    elif layout in {"invalid_index", "outside_root"}:
        weight_map = {} if layout == "invalid_index" else {"a": "../outside.safetensors"}
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )
    with pytest.raises(ValueError):
        cache.discover_checkpoint_files([tmp_path])


def test_cold_success_is_client_only(tmp_path: Path) -> None:
    weight = _weight(tmp_path)
    evidence = cache.evict_and_verify([weight, weight], exclusive=True)
    assert evidence["verified"] is True
    assert evidence["backend_cache_verified"] is False
    assert evidence["resident_pages"] == 0
    assert evidence["checkpoint_bytes"] == 15
    assert evidence["total_pages"] == 1
    assert len(evidence["files"]) == 1
    cache.os.posix_fadvise.assert_called_once()
    cache._resident_pages.assert_called_once()


@pytest.mark.parametrize(
    "filesystem,options,expected",
    [
        ("nfs", "rw,vers=3", "disabled"),
        ("nfs4", "rw,vers=4.2,nofsc", "disabled"),
        ("ext4", "rw", "not_applicable"),
        ("xfs", "rw,attr2", "not_applicable"),
        ("overlay", "rw,lowerdir=/base", "not_applicable"),
    ],
)
def test_mount_cache_scope(
    tmp_path: Path,
    fake_cache_operations: dict,
    filesystem: str,
    options: str,
    expected: str,
) -> None:
    fake_cache_operations["mountinfo"] = f"42 1 0:40 / /models ro - {filesystem} source {options}\n"
    evidence = cache.evict_and_verify([_weight(tmp_path)], exclusive=True)
    assert evidence["verified"] is True
    assert evidence["files"][0]["mount"]["nfs_fscache"] == expected
    assert evidence["disk_cache_scope"] == "nfs_fscache_only"
    assert evidence["other_disk_caches_verified"] is False


@pytest.mark.parametrize("option", ["fsc", "fsc=checkpoint_cache"])
def test_nfs_disk_cache_blocks_reset_and_retains_evidence(
    tmp_path: Path, fake_cache_operations: dict, option: str
) -> None:
    fake_cache_operations["mountinfo"] = (
        f"42 1 0:40 / /models ro - nfs4 server:/models rw,vers=4.2,{option}\n"
    )
    with pytest.raises(cache.CacheVerificationError, match="NFS FS-Cache is enabled") as caught:
        cache.evict_and_verify([_weight(tmp_path)], exclusive=True, reset_command=["/helper"])
    assert caught.value.evidence["files"][0]["mount"]["nfs_fscache"] == "enabled"
    cache.subprocess.run.assert_not_called()
    cache.os.posix_fadvise.assert_not_called()
    cache._resident_pages.assert_not_called()


def test_open_file_mount_id_selects_container_bind_mount(
    tmp_path: Path, fake_cache_operations: dict
) -> None:
    fake_cache_operations["mountinfo"] = (
        "8 1 0:10 / /models rw - overlay overlay rw,lowerdir=/base\n"
        "35 1 0:39 / /other rw - nfs4 other:/export rw,fsc\n"
        r"42 8 0:40 /checkpoint\040root /models\040bind ro shared:3 - nfs4 server:/export rw,vers=4.2"
        "\n"
    )
    weight = _weight(tmp_path)
    alias = tmp_path / "linked.safetensors"
    alias.symlink_to(weight)
    evidence = cache.evict_and_verify([alias], exclusive=True)
    mount = evidence["files"][0]["mount"]
    assert mount["mount_id"] == 42
    assert mount["root"] == "/checkpoint root" and mount["mount_point"] == "/models bind"
    assert mount["filesystem"] == "nfs4" and mount["nfs_fscache"] == "disabled"


@pytest.mark.parametrize(
    "key,value",
    [
        ("fdinfo", PermissionError("fdinfo denied")),
        ("fdinfo", "pos:\t0\n"),
        ("fdinfo", "mnt_id:\t42\nmnt_id:\t43\n"),
        ("mountinfo", FileNotFoundError("mountinfo unavailable")),
        ("mountinfo", "41 1 0:40 / /models rw - nfs4 source rw\n"),
        ("mountinfo", "42 1 0:40 / /models rw - nfs4 source\n"),
        ("mountinfo", "42 1 0:40 / /models rw - nfs4 source vers=4.2\n"),
    ],
)
def test_unverifiable_mount_blocks_reset(
    tmp_path: Path, fake_cache_operations: dict, key: str, value: str | Exception
) -> None:
    fake_cache_operations[key] = value
    with pytest.raises(cache.CacheVerificationError) as caught:
        cache.evict_and_verify([_weight(tmp_path)], exclusive=True)
    assert caught.value.evidence["verified"] is False and caught.value.evidence["errors"]
    cache.os.posix_fadvise.assert_not_called()
    cache._resident_pages.assert_not_called()


def test_reset_cannot_change_mount_cache_policy(
    tmp_path: Path, fake_cache_operations: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def change_mount(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        fake_cache_operations["mountinfo"] = (
            "42 1 0:40 / /models ro - nfs4 server:/models rw,vers=4.2,fsc\n"
        )
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(cache.subprocess, "run", change_mount)
    with pytest.raises(cache.CacheVerificationError, match="mount changed during reset"):
        cache.evict_and_verify([_weight(tmp_path)], exclusive=True, reset_command=["/helper"])
    cache._resident_pages.assert_not_called()


def test_resident_or_permission_masked_pages_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache, "_resident_pages", Mock(return_value=1))
    with pytest.raises(cache.CacheVerificationError, match="permission-masked") as caught:
        cache.evict_and_verify([_weight(tmp_path)], exclusive=True)
    assert caught.value.evidence["verified"] is False
    assert caught.value.evidence["resident_pages"] == 1


@pytest.mark.parametrize(
    "error", [PermissionError("denied"), AttributeError("mincore unavailable")]
)
def test_unverifiable_cache_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(cache, "_resident_pages", Mock(side_effect=error))
    with pytest.raises(cache.CacheVerificationError) as caught:
        cache.evict_and_verify([_weight(tmp_path)], exclusive=True)
    assert caught.value.evidence["verified"] is False
    assert caught.value.evidence["errors"]


@pytest.mark.parametrize("exclusive,system", [(False, "Linux"), (True, "Darwin")])
def test_environment_guard_prevents_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exclusive: bool, system: str
) -> None:
    monkeypatch.setattr(cache.platform, "system", lambda: system)
    with pytest.raises(cache.CacheVerificationError):
        cache.evict_and_verify([_weight(tmp_path)], exclusive=exclusive)
    cache.os.posix_fadvise.assert_not_called()
    cache._resident_pages.assert_not_called()


@pytest.mark.parametrize("returncode", [0, 3])
def test_approved_reset_is_argv_only_and_must_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returncode: int
) -> None:
    command = ["/approved/helper", "checkpoint path"]
    helper = Mock(return_value=subprocess.CompletedProcess(command, returncode, "out", "err"))
    monkeypatch.setattr(cache.subprocess, "run", helper)
    weight = _weight(tmp_path)
    if returncode:
        with pytest.raises(cache.CacheVerificationError, match="exit code 3") as caught:
            cache.evict_and_verify([weight], exclusive=True, reset_command=command)
        assert caught.value.evidence["reset_returncode"] == 3
        cache._resident_pages.assert_not_called()
    else:
        assert cache.evict_and_verify([weight], exclusive=True, reset_command=command)["verified"]
    helper.assert_called_once_with(
        command, shell=False, check=False, capture_output=True, text=True, timeout=300
    )
    cache.os.posix_fadvise.assert_not_called()


def test_reset_cannot_change_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weight = _weight(tmp_path)

    def replace_weight(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        weight.write_bytes(b"different checkpoint")
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(cache.subprocess, "run", replace_weight)
    with pytest.raises(cache.CacheVerificationError, match="changed during reset"):
        cache.evict_and_verify([weight], exclusive=True, reset_command=["/approved/helper"])
    cache._resident_pages.assert_not_called()


def test_mincore_mapping_never_reads_payload_and_counts_partial_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libc = Mock()
    libc.mmap.return_value = 4096
    libc.munmap.return_value = 0

    def fill_vector(address: int, length: int, vector: object) -> int:
        assert address == 4096 and length == 4097
        assert len(vector) == 2
        vector[0] = 0xFE  # Only the low bit is defined by mincore.
        vector[1] = 1
        return 0

    libc.mincore.side_effect = fill_vector
    monkeypatch.setattr(cache.ctypes, "CDLL", Mock(return_value=libc))
    assert _MINCORE_PROBE(123, 4097, 4096) == 1
    libc.mmap.assert_called_once_with(None, 4097, 0, cache.mmap.MAP_SHARED, 123, 0)
    libc.munmap.assert_called_once_with(4096, 4097)


def test_mincore_failure_unmaps_and_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    libc = Mock()
    libc.mmap.return_value = 4096
    libc.mincore.return_value = -1
    libc.munmap.return_value = 0
    monkeypatch.setattr(cache.ctypes, "CDLL", Mock(return_value=libc))
    monkeypatch.setattr(cache.ctypes, "get_errno", lambda: 13)
    with pytest.raises(PermissionError, match="mincore failed"):
        _MINCORE_PROBE(123, 4096, 4096)
    libc.munmap.assert_called_once()


def test_mincore_checks_every_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    libc = Mock()
    libc.mmap.return_value = 4096
    libc.mincore.return_value = 0
    libc.munmap.return_value = 0
    monkeypatch.setattr(cache.ctypes, "CDLL", Mock(return_value=libc))
    assert _MINCORE_PROBE(123, (1 << 30) + 1, 4096) == 0
    assert [call.args[1] for call in libc.mincore.call_args_list] == [1 << 30, 1]
    assert [call.args[5] for call in libc.mmap.call_args_list] == [0, 1 << 30]
