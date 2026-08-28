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

import pathlib
import sys
import types

import pytest

_SYSINFO_DIR = pathlib.Path(__file__).resolve().parents[2] / "integration" / "defs" / "sysinfo"
sys.path.insert(0, str(_SYSINFO_DIR))

import get_sysinfo  # noqa: E402


def _disable_distro(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "distro", None)


def test_get_linux_distribution_from_distro(monkeypatch: pytest.MonkeyPatch) -> None:
    distro = types.ModuleType("distro")
    monkeypatch.setattr(distro, "id", lambda: "ubuntu", raising=False)
    monkeypatch.setattr(distro, "version", lambda: "24.04", raising=False)
    monkeypatch.setattr(distro, "codename", lambda: "noble", raising=False)
    monkeypatch.setitem(sys.modules, "distro", distro)

    assert get_sysinfo.get_linux_distribution() == ("ubuntu", "24.04", "noble")


def test_get_linux_distribution_from_os_release(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_distro(monkeypatch)
    monkeypatch.setattr(
        get_sysinfo.platform,
        "freedesktop_os_release",
        lambda: {
            "ID": "ubuntu",
            "VERSION_ID": "24.04",
            "VERSION_CODENAME": "noble",
        },
    )

    assert get_sysinfo.get_linux_distribution() == ("ubuntu", "24.04", "noble")


def test_get_linux_distribution_without_os_release(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_distro(monkeypatch)

    def raise_os_error() -> dict[str, str]:
        raise OSError("os-release is unavailable")

    monkeypatch.setattr(get_sysinfo.platform, "freedesktop_os_release", raise_os_error)

    assert get_sysinfo.get_linux_distribution() == ("na", "na", "na")


@pytest.mark.parametrize(
    "missing_field",
    ["ID", "VERSION_ID", "VERSION_CODENAME"],
)
def test_get_linux_distribution_with_missing_field(
    monkeypatch: pytest.MonkeyPatch,
    missing_field: str,
) -> None:
    _disable_distro(monkeypatch)
    os_release = {
        "ID": "ubuntu",
        "VERSION_ID": "24.04",
        "VERSION_CODENAME": "noble",
    }
    del os_release[missing_field]
    monkeypatch.setattr(
        get_sysinfo.platform,
        "freedesktop_os_release",
        lambda: os_release,
    )

    assert get_sysinfo.get_linux_distribution() == (
        os_release.get("ID", "na"),
        os_release.get("VERSION_ID", "na"),
        os_release.get("VERSION_CODENAME", "na"),
    )
