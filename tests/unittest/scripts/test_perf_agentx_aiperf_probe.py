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
"""Tests for the AgentX client's check that the installed aiperf is the agentx fork."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

# No GPU dependency. The l0_cpu stage runs pytest with ``-m cpu_only`` and conftest
# drops any file lacking this marker.
pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
AGENTX_CLIENT_PATH = REPO_ROOT / "tests" / "integration" / "defs" / "perf" / "agentx_client.py"

# Every module aiperf_is_usable imports besides the fork-only scenario registry.
_STUB_MODULES = ("cyclopts", "datasets", "transformers", "pydantic", "msgspec", "orjson")


@pytest.fixture
def client() -> ModuleType:
    spec = importlib.util.spec_from_file_location("perf_agentx_client", AGENTX_CLIENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_stub_site(root: Path, *, fork: bool, datasets_version: str = "4.7.0") -> None:
    """Stand-in packages: an upstream-like aiperf, or the fork with its scenario."""
    for name in _STUB_MODULES:
        version = datasets_version if name == "datasets" else "1.0.0"
        (root / name).mkdir()
        (root / name / "__init__.py").write_text(f"__version__ = {version!r}\n")
    (root / "aiperf").mkdir()
    (root / "aiperf" / "__init__.py").write_text("")
    if fork:
        registry = root / "aiperf" / "common" / "scenario"
        registry.mkdir(parents=True)
        (root / "aiperf" / "common" / "__init__.py").write_text("")
        (registry / "__init__.py").write_text("")
        (registry / "registry.py").write_text(
            "SCENARIOS = {'inferencex-agentx-mvp': object()}\n"
            "def get_scenario(name):\n"
            "    return SCENARIOS[name]\n"
        )


def _probe(client: ModuleType, monkeypatch, site: Path) -> tuple[bool, str]:
    """Run aiperf_is_usable with only the stub packages importable; return (usable, log)."""
    real_run = subprocess.run

    def run(argv, **kwargs):
        # -I keeps the host interpreter's own site-packages out of the probe.
        probe = argv[-1]
        return real_run(
            [sys.executable, "-I", "-c", f"import sys\nsys.path.insert(0, {str(site)!r})\n{probe}"],
            **kwargs,
        )

    logged = []
    monkeypatch.setattr(client.subprocess, "run", run)
    monkeypatch.setattr(client, "_log", logged.append)
    return client.aiperf_is_usable(), "\n".join(logged)


def test_upstream_aiperf_is_not_usable(client: ModuleType, monkeypatch, tmp_path: Path):
    # Stock AIPerf has no AgentX scenario and must fail the fork preflight.
    _write_stub_site(tmp_path, fork=False)
    usable, log = _probe(client, monkeypatch, tmp_path)
    assert not usable
    assert "aiperf.common.scenario.registry" in log


def test_agentx_fork_is_usable(client: ModuleType, monkeypatch, tmp_path: Path):
    _write_stub_site(tmp_path, fork=True)
    usable, log = _probe(client, monkeypatch, tmp_path)
    assert usable
    assert log == ""


def test_fork_without_the_scenario_is_reported(client: ModuleType, monkeypatch, tmp_path: Path):
    _write_stub_site(tmp_path, fork=True)
    registry = tmp_path / "aiperf" / "common" / "scenario" / "registry.py"
    registry.write_text("def get_scenario(name):\n    raise KeyError(name)\n")
    usable, log = _probe(client, monkeypatch, tmp_path)
    assert not usable
    assert client.AGENTX_SCENARIO in log


def test_fork_with_old_datasets_is_not_usable(client: ModuleType, monkeypatch, tmp_path: Path):
    _write_stub_site(tmp_path, fork=True, datasets_version="3.1.0")
    usable, log = _probe(client, monkeypatch, tmp_path)
    assert not usable
    assert "datasets 3.1.0" in log


def _default_cmd(client: ModuleType, monkeypatch) -> list[str]:
    monkeypatch.delenv("AGENTX_EXTRA_ARGS", raising=False)
    args = client.parse_args(
        ["--model", "/m", "--concurrency", "8", "--dataset", "d", "--host", "h", "--port", "1"]
    )
    return client.build_aiperf_cmd(args, "http://h:1", "/tmp/a")


def test_command_runs_the_scenario_the_probe_checks(client: ModuleType, monkeypatch):
    cmd = _default_cmd(client, monkeypatch)
    assert cmd[cmd.index("--scenario") + 1] == client.AGENTX_SCENARIO


def test_command_keeps_special_tokens_in_the_text(client: ModuleType, monkeypatch):
    # Under the scenario's forced ignore_eos, a one-token response can be the
    # end-of-sequence token; without this it decodes to no text at all.
    cmd = _default_cmd(client, monkeypatch)
    extra_inputs = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--extra-inputs"]
    assert "skip_special_tokens:false" in extra_inputs


def test_command_runs_aiperf_from_the_checked_interpreter(client: ModuleType, monkeypatch):
    # The probe and workload must use the same isolated fork environment.
    cmd = _default_cmd(client, monkeypatch)
    assert cmd[:5] == [client._agentx_python(), "-I", "-m", "aiperf", "profile"]


def test_missing_isolated_environment_does_not_use_ambient_python(
    client: ModuleType, monkeypatch, tmp_path: Path
):
    monkeypatch.delenv("AGENTX_FORCE_REINSTALL", raising=False)
    monkeypatch.setenv("AGENTX_PYTHON", str(tmp_path / "missing-python"))
    with pytest.raises(RuntimeError, match="during setup"):
        client.ensure_aiperf()


def test_benchmark_rejects_install_request(client: ModuleType, monkeypatch):
    monkeypatch.setenv("AGENTX_FORCE_REINSTALL", "1")
    with pytest.raises(RuntimeError, match="during setup"):
        client.ensure_aiperf()
