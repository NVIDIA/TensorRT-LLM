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
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FMHA_PATH = _REPO_ROOT / "tests/integration/defs/test_fmha.py"


def _load_fmha_module() -> types.ModuleType:
    """Load test_fmha without importing the integration-test conftest."""
    package = types.ModuleType("fmha_test_defs")
    package.__path__ = [str(_FMHA_PATH.parent)]
    conftest = types.ModuleType("fmha_test_defs.conftest")
    conftest.get_sm_version = lambda: 90

    previous_package = sys.modules.get(package.__name__)
    previous_conftest = sys.modules.get(conftest.__name__)
    sys.modules[package.__name__] = package
    sys.modules[conftest.__name__] = conftest
    try:
        spec = importlib.util.spec_from_file_location(f"{package.__name__}.test_fmha", _FMHA_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous_package is None:
            sys.modules.pop(package.__name__, None)
        else:
            sys.modules[package.__name__] = previous_package
        if previous_conftest is None:
            sys.modules.pop(conftest.__name__, None)
        else:
            sys.modules[conftest.__name__] = previous_conftest
    return module


_FMHA = _load_fmha_module()


@pytest.mark.parametrize("ccache_available", [False, True])
@pytest.mark.parametrize("inherited_value", [None, "0", "1"])
@pytest.mark.parametrize("sm_version", [80, 90])
def test_fmha_selects_ccache_without_overriding_user_setting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ccache_available: bool,
    inherited_value: str | None,
    sm_version: int,
) -> None:
    """Pass the selected ccache setting to every FMHA build invocation."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_FMHA, "get_sm_version", lambda: sm_version)
    monkeypatch.setattr(
        _FMHA,
        "which",
        lambda executable: "/usr/bin/ccache" if ccache_available else None,
    )
    monkeypatch.delenv("USE_CCACHE", raising=False)
    if inherited_value is not None:
        monkeypatch.setenv("USE_CCACHE", inherited_value)

    build_environments: list[dict[str, str]] = []

    def run(
        command: str,
        *,
        shell: bool,
        check: bool,
        env: dict[str, str] | None = None,
    ) -> None:
        assert shell
        assert check
        if env is not None:
            build_environments.append(env.copy())

    monkeypatch.setattr(_FMHA, "run", run)
    _FMHA.test_fmha()

    expected = inherited_value or ("1" if ccache_available else None)
    assert len(build_environments) == (7 if sm_version == 80 else 3)
    expected_architectures = {80, 100, 120} if sm_version == 80 else {90}
    assert {int(env["FMHA_FILTER_ARCH"]) for env in build_environments} == expected_architectures
    for env in build_environments:
        assert env.get("USE_CCACHE") == expected
