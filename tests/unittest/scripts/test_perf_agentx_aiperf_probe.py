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
"""``agentx_client.aiperf_is_usable`` tells the agentx-harness aiperf from PyPI's.

Each case builds a fake site directory holding stub modules and runs the real
probe in a subprocess against it, so no aiperf has to be installed.
"""

import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENTX_CLIENT_PATH = REPO_ROOT / "tests" / "integration" / "defs" / "perf" / "agentx_client.py"

# Every module the probe imports besides aiperf itself.
_STUB_MODULES = ("cyclopts", "transformers", "pydantic", "msgspec", "orjson")

_FORK_REGISTRY = textwrap.dedent(
    """
    SCENARIOS = {"inferencex-agentx-mvp": object()}

    def get_scenario(name):
        if name not in SCENARIOS:
            raise KeyError(name)
        return SCENARIOS[name]
    """
)


@pytest.fixture
def client() -> ModuleType:
    spec = importlib.util.spec_from_file_location("perf_agentx_client", AGENTX_CLIENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_site(
    root: Path, *, fork: bool, registry: str = _FORK_REGISTRY, datasets_version: str = "4.1.0"
) -> Path:
    for name in _STUB_MODULES:
        (root / f"{name}.py").write_text("")
    (root / "datasets.py").write_text(f"__version__ = {datasets_version!r}\n")
    pkg = root / "aiperf"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    if fork:
        scenario = pkg / "common" / "scenario"
        scenario.mkdir(parents=True)
        (pkg / "common" / "__init__.py").write_text("")
        (scenario / "__init__.py").write_text("")
        (scenario / "registry.py").write_text(registry)
    return root


@pytest.fixture
def run_probe(client, monkeypatch, capfd):
    def _run(site: Path) -> "tuple[bool, str]":
        monkeypatch.setenv("PYTHONPATH", str(site))
        monkeypatch.setenv("PYTHONNOUSERSITE", "1")
        # -S keeps the real site-packages (and any real aiperf) out of the probe.
        monkeypatch.setattr(client.sys, "executable", sys.executable)
        real_run = client.subprocess.run

        def _isolated(cmd, *args, **kwargs):
            return real_run([cmd[0], "-S", *cmd[1:]], *args, **kwargs)

        monkeypatch.setattr(client.subprocess, "run", _isolated)
        usable = client.aiperf_is_usable()
        return usable, capfd.readouterr().out

    return _run


def test_fork_build_is_usable(tmp_path, run_probe):
    usable, out = run_probe(_make_site(tmp_path, fork=True))
    assert usable, out
    assert out == ""


def test_pypi_aiperf_is_not_usable(tmp_path, run_probe):
    """The PyPI aiperf imports fine but has no scenario registry."""
    usable, out = run_probe(_make_site(tmp_path, fork=False))
    assert not usable
    assert "not the agentx-harness build" in out
    assert "inferencex-agentx-mvp" in out


def test_fork_without_agentx_scenario_is_not_usable(tmp_path, run_probe):
    registry = _FORK_REGISTRY.replace('"inferencex-agentx-mvp"', '"other-scenario"')
    usable, out = run_probe(_make_site(tmp_path, fork=True, registry=registry))
    assert not usable
    assert "not the agentx-harness build" in out


def test_old_datasets_is_not_usable(tmp_path, run_probe):
    usable, out = run_probe(_make_site(tmp_path, fork=True, datasets_version="3.1.0"))
    assert not usable
    assert "datasets 3.1.0 < 4" in out


def test_missing_module_is_not_usable(tmp_path, run_probe):
    site = _make_site(tmp_path, fork=True)
    (site / "orjson.py").unlink()
    usable, out = run_probe(site)
    assert not usable
    assert "missing module" in out


def test_command_uses_the_probed_scenario(client):
    args = client.parse_args(
        ["--model", "m", "--concurrency", "8", "--dataset", "d", "--host", "h", "--port", "1"]
    )
    cmd = client.build_aiperf_cmd(args, "http://h:1", "/tmp/out")
    assert cmd[cmd.index("--scenario") + 1] == client.AGENTX_SCENARIO
