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
"""Keep benchmark client installation independent of runtime GitHub access in built images."""

import hashlib
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = Path(__file__).resolve().parents[3]


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = _load(ROOT / "tests/integration/defs/perf/agentx_client.py")
    monkeypatch.setattr(module, "AIPERF_WHEEL_MANIFEST", tmp_path / "manifest.json", raising=False)
    monkeypatch.delenv("AGENTX_AIPERF_REF", raising=False)
    monkeypatch.delenv("AGENTX_FORCE_REINSTALL", raising=False)
    return module


def _cache(client: ModuleType) -> Path:
    wheel = client.AIPERF_WHEEL_MANIFEST.parent / "aiperf-0.12.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel content")
    client.AIPERF_WHEEL_MANIFEST.write_text(
        json.dumps(
            {
                "source_ref": client.DEFAULT_AIPERF_REF,
                "wheel": wheel.name,
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            }
        )
    )
    return wheel


@pytest.mark.parametrize("force", [False, True])
def test_ensure_uses_cached_wheel_without_git(
    client: ModuleType, monkeypatch: pytest.MonkeyPatch, force: bool
) -> None:
    wheel = _cache(client)
    probes = iter([False, True])
    monkeypatch.setattr(client, "aiperf_is_usable", lambda: next(probes))
    monkeypatch.setattr(client, "_Heartbeat", lambda *args: nullcontext())
    if force:
        monkeypatch.setenv("AGENTX_FORCE_REINSTALL", "1")
        monkeypatch.setattr(client, "aiperf_is_usable", lambda: True)
    calls = []

    def pip(*args: str) -> None:
        assert not any("git+https://" in arg for arg in args), "runtime GitHub clone"
        calls.append(args)

    monkeypatch.setattr(client, "_pip", pip)
    client.ensure_aiperf("http://unused", "model")
    assert calls[0] == ("--force-reinstall", "--no-deps", f"aiperf @ {wheel.as_uri()}")


def test_explicit_ref_overrides_cache(client: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    _cache(client)
    monkeypatch.setenv("AGENTX_AIPERF_REF", "file:///custom/aiperf.whl")
    assert client.aiperf_install_ref() == "file:///custom/aiperf.whl"


def test_old_images_keep_source_install(client: ModuleType) -> None:
    assert client.aiperf_install_ref() == client.DEFAULT_AIPERF_REF


def test_stale_cache_does_not_install_wrong_pin(client: ModuleType, capsys) -> None:
    _cache(client)
    record = json.loads(client.AIPERF_WHEEL_MANIFEST.read_text())
    record["source_ref"] += "-old"
    client.AIPERF_WHEEL_MANIFEST.write_text(json.dumps(record))
    assert client.aiperf_install_ref() == client.DEFAULT_AIPERF_REF
    assert "does not match" in capsys.readouterr().out


def test_corrupt_wheel_fails_instead_of_cloning(client: ModuleType) -> None:
    wheel = _cache(client)
    wheel.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        client.aiperf_install_ref()


def test_cache_cannot_reference_parent_directory(client: ModuleType) -> None:
    _cache(client)
    record = json.loads(client.AIPERF_WHEEL_MANIFEST.read_text())
    record["wheel"] = "../aiperf.whl"
    client.AIPERF_WHEEL_MANIFEST.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="filename"):
        client.aiperf_install_ref()


def test_builder_records_client_pin_and_wheel_digest(tmp_path: Path, monkeypatch) -> None:
    builder = _load(ROOT / "docker/common/cache_benchmark_client_wheel.py")
    source = ROOT / "tests/integration/defs/perf/agentx_client.py"
    client = _load(source)
    output = tmp_path / "cache"

    def build(argv: list[str], *, check: bool) -> None:
        assert check
        assert argv[1:5] == ["-m", "pip", "wheel", "--no-deps"]
        assert argv[-1] == f"aiperf @ {client.DEFAULT_AIPERF_REF}"
        (output / "aiperf-0.12.0-py3-none-any.whl").write_bytes(b"built wheel")

    monkeypatch.setattr(builder.subprocess, "run", build)
    builder.cache_wheel(source, output)
    record = json.loads((output / "manifest.json").read_text())
    assert record["source_ref"] == client.DEFAULT_AIPERF_REF
    assert record["sha256"] == hashlib.sha256(b"built wheel").hexdigest()


@pytest.mark.parametrize("version", [(3, 10), (3, 14)])
def test_unsupported_python_image_does_not_fetch_fork(tmp_path: Path, monkeypatch, version) -> None:
    builder = _load(ROOT / "docker/common/cache_benchmark_client_wheel.py")
    monkeypatch.setattr(builder, "sys", SimpleNamespace(version_info=version))
    output = tmp_path / "cache"
    builder.cache_wheel(tmp_path / "unused.py", output)
    assert not output.exists()


def test_usable_install_does_not_read_cache(client: ModuleType, monkeypatch) -> None:
    client.AIPERF_WHEEL_MANIFEST.write_text("invalid JSON")
    monkeypatch.setattr(client, "aiperf_is_usable", lambda: True)
    client.ensure_aiperf("http://unused", "model")


def test_builder_does_not_relabel_an_existing_wheel(tmp_path: Path) -> None:
    builder = _load(ROOT / "docker/common/cache_benchmark_client_wheel.py")
    (tmp_path / "aiperf-0.12.0-py3-none-any.whl").write_bytes(b"old source")
    with pytest.raises(RuntimeError, match="must be empty"):
        builder.cache_wheel(ROOT / "tests/integration/defs/perf/agentx_client.py", tmp_path)
    assert not (tmp_path / "manifest.json").exists()
