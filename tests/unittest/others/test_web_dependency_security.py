# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]


def _requirement(name: str, manifest: str = "requirements.txt") -> Requirement:
    for line in (REPO_ROOT / manifest).read_text().splitlines():
        # Trailing comments are common in these manifests and are not part of
        # the PEP 508 grammar, so Requirement() would reject the whole line.
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(f"{name}"):
            requirement = Requirement(line)
            if requirement.name == name:
                return requirement
    raise AssertionError(f"Missing requirement for {name} in {manifest}")


def _minimum_version(requirement: Requirement) -> Version:
    lower_bounds = [
        Version(specifier.version)
        for specifier in requirement.specifier
        if specifier.operator in {">", ">="}
    ]
    assert lower_bounds, f"Missing minimum version for {requirement.name}"
    return max(lower_bounds)


def test_web_framework_security_floors() -> None:
    fastapi = _requirement("fastapi")
    instrumentator = _requirement("prometheus_fastapi_instrumentator")
    starlette = _requirement("starlette")

    assert _minimum_version(fastapi) >= Version("0.136.3")
    assert Version("0.121.3") not in fastapi.specifier
    # Deliberately no upper bound on fastapi. The cap this PR removes is what
    # pinned Starlette below 1.x in the first place, and fastapi has carried no
    # starlette ceiling since 0.133.0 (0.134.0-0.141.1 all declare
    # starlette>=0.46.0 unbounded), so re-capping would recreate the same
    # deadlock a release later.
    assert not any(specifier.operator in {"<", "<="} for specifier in fastapi.specifier)
    assert _minimum_version(instrumentator) >= Version("8.1.0")
    assert _minimum_version(starlette) >= Version("1.3.1")
    assert Version("0.50.0") not in starlette.specifier


def test_starlette_constraint_floor() -> None:
    # constraints.txt carries the GHSA-82w8-qh3p-5jfq WAR. It must stay active
    # (uncommented) at the patched floor, which is what this PR turns back on.
    starlette = _requirement("starlette", "constraints.txt")

    assert _minimum_version(starlette) >= Version("1.3.1")
    assert Version("0.50.0") not in starlette.specifier
