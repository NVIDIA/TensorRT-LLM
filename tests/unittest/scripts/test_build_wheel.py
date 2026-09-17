#!/usr/bin/env python3
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

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BUILD_WHEEL_PATH = REPO_ROOT / "scripts" / "build_wheel.py"


@pytest.fixture(scope="module")
def build_wheel_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_wheel", BUILD_WHEEL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_linux_physical_cpu_count_counts_unique_cores(build_wheel_module: ModuleType) -> None:
    cpuinfo = """
processor   : 0
physical id : 0
core id     : 0

processor   : 1
physical id : 0
core id     : 0

processor   : 2
physical id : 0
core id     : 1

processor   : 3
physical id : 0
core id     : 1

processor   : 4
physical id : 1
core id     : 0

processor   : 5
physical id : 1
core id     : 0
"""

    assert (
        build_wheel_module._parse_linux_physical_cpu_count(
            cpuinfo, available_cpus={0, 1, 2, 3, 4, 5}
        )
        == 3
    )
    assert (
        build_wheel_module._parse_linux_physical_cpu_count(cpuinfo, available_cpus={0, 1, 2, 3})
        == 2
    )
    assert (
        build_wheel_module._parse_linux_physical_cpu_count(cpuinfo, available_cpus={0, 2, 4}) == 3
    )


def test_parse_linux_physical_cpu_count_returns_none_without_topology(
    build_wheel_module: ModuleType,
) -> None:
    cpuinfo = """
processor   : 0
model name  : Example CPU

processor   : 1
model name  : Example CPU
"""

    assert (
        build_wheel_module._parse_linux_physical_cpu_count(cpuinfo, available_cpus={0, 1}) is None
    )


def test_parse_linux_physical_cpu_count_rejects_malformed_topology(
    build_wheel_module: ModuleType,
) -> None:
    cpuinfo = """
processor   : 0
physical id : 0
core id     : 0

processor   : 1
physical id : 0
core id     : not-a-core

processor   : 2
physical id : not-a-socket
core id     : 1
"""

    assert (
        build_wheel_module._parse_linux_physical_cpu_count(cpuinfo, available_cpus={0, 1, 2})
        is None
    )


def test_parse_linux_physical_cpu_count_rejects_missing_topology_record(
    build_wheel_module: ModuleType,
) -> None:
    cpuinfo = """
processor   : 0
physical id : 0
core id     : 0

processor   : 1
model name  : Example CPU

processor   : 2
physical id : 0
core id     : 1
"""

    assert (
        build_wheel_module._parse_linux_physical_cpu_count(cpuinfo, available_cpus={0, 1, 2})
        is None
    )


def test_parse_linux_physical_cpu_count_rejects_too_low_physical_count(
    build_wheel_module: ModuleType,
) -> None:
    cpuinfo = "\n\n".join(
        f"""processor   : {processor}
physical id : 0
core id     : 0"""
        for processor in range(8)
    )

    assert (
        build_wheel_module._parse_linux_physical_cpu_count(cpuinfo, available_cpus=set(range(8)))
        is None
    )


def test_get_available_cpu_count_prefers_physical_count(
    build_wheel_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build_wheel_module, "_get_cpu_affinity", lambda: {0, 1, 2, 3})
    monkeypatch.setattr(
        build_wheel_module, "_get_linux_physical_cpu_count", lambda available_cpus: 2
    )

    assert build_wheel_module.get_available_cpu_count() == 2


def test_get_available_cpu_count_falls_back_to_affinity(
    build_wheel_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build_wheel_module, "_get_cpu_affinity", lambda: {0, 1, 2})
    monkeypatch.setattr(
        build_wheel_module, "_get_linux_physical_cpu_count", lambda available_cpus: None
    )

    assert build_wheel_module.get_available_cpu_count() == 3


def test_get_available_cpu_count_falls_back_to_logical_count(
    build_wheel_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build_wheel_module, "_get_cpu_affinity", lambda: None)
    monkeypatch.setattr(
        build_wheel_module, "_get_linux_physical_cpu_count", lambda available_cpus: None
    )
    monkeypatch.setattr(build_wheel_module, "cpu_count", lambda: 8)

    assert build_wheel_module.get_available_cpu_count() == 8


def test_get_available_cpu_count_handles_missing_logical_count(
    build_wheel_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build_wheel_module, "_get_cpu_affinity", lambda: None)
    monkeypatch.setattr(
        build_wheel_module, "_get_linux_physical_cpu_count", lambda available_cpus: None
    )
    monkeypatch.setattr(build_wheel_module, "cpu_count", lambda: None)

    assert build_wheel_module.get_available_cpu_count() == 1
