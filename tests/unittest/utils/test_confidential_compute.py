# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import sys
import types
from collections.abc import Generator
from typing import Protocol

import pytest

from tensorrt_llm._utils import confidential_compute_enabled, get_cc_and_nvle_status

pytestmark = pytest.mark.cpu_only


class _NvmlError(Exception):
    pass


class _NvmlErrorNotSupported(_NvmlError):
    pass


class _ConfComputeSettings(ctypes.Structure):
    _fields_ = [
        ("ccFeature", ctypes.c_uint),
        ("multiGpuMode", ctypes.c_uint),
    ]


class _ConfComputeSettingsPointer(Protocol):
    _obj: _ConfComputeSettings


def _make_pynvml(
    *,
    cc_feature: int,
    multi_gpu_mode: int,
    settings_supported: bool = True,
    legacy_cc_feature: int = 0,
) -> types.ModuleType:
    pynvml = types.ModuleType("pynvml")
    pynvml.NVMLError = _NvmlError
    pynvml.NVMLError_NotSupported = _NvmlErrorNotSupported
    pynvml.NVML_CC_SYSTEM_FEATURE_ENABLED = 1
    pynvml.NVML_CC_SYSTEM_MULTIGPU_PROTECTED_PCIE = 1
    pynvml.NVML_CC_SYSTEM_MULTIGPU_NVLE = 2
    pynvml.c_nvmlSystemConfComputeSettings_v1_t = _ConfComputeSettings
    pynvml.nvmlInit = lambda: None
    pynvml.nvmlShutdown = lambda: None
    pynvml._nvmlCheckReturn = lambda _: None
    pynvml.settings_queries = 0

    def get_settings(settings_ptr: _ConfComputeSettingsPointer) -> int:
        pynvml.settings_queries += 1
        if not settings_supported:
            raise _NvmlErrorNotSupported
        settings_ptr._obj.ccFeature = cc_feature
        settings_ptr._obj.multiGpuMode = multi_gpu_mode
        return 0

    pynvml.nvmlSystemGetConfComputeSettings = get_settings
    pynvml.nvmlSystemGetConfComputeState = lambda: types.SimpleNamespace(
        ccFeature=legacy_cc_feature
    )
    return pynvml


@pytest.fixture(autouse=True)
def clear_confidential_compute_status_cache() -> Generator[None, None, None]:
    get_cc_and_nvle_status.cache_clear()
    yield
    get_cc_and_nvle_status.cache_clear()


def test_get_cc_and_nvle_status_without_pynvml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)

    assert get_cc_and_nvle_status() == (False, False)


@pytest.mark.parametrize(
    "cc_feature,multi_gpu_mode,expected",
    [
        (0, 0, (False, False)),
        (0, 1, (True, False)),
        (0, 2, (False, True)),
        (1, 2, (True, True)),
    ],
)
def test_get_cc_and_nvle_status(
    monkeypatch: pytest.MonkeyPatch,
    cc_feature: int,
    multi_gpu_mode: int,
    expected: tuple[bool, bool],
) -> None:
    pynvml = _make_pynvml(
        cc_feature=cc_feature,
        multi_gpu_mode=multi_gpu_mode,
    )
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)

    assert get_cc_and_nvle_status() == expected


def test_get_cc_and_nvle_status_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    pynvml = _make_pynvml(cc_feature=0, multi_gpu_mode=2)
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)

    assert get_cc_and_nvle_status() == (False, True)
    assert get_cc_and_nvle_status() == (False, True)
    assert pynvml.settings_queries == 1


@pytest.mark.parametrize(
    "status",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_confidential_compute_enabled(
    monkeypatch: pytest.MonkeyPatch,
    status: tuple[bool, bool],
) -> None:
    def get_status() -> tuple[bool, bool]:
        return status

    monkeypatch.setattr("tensorrt_llm._utils.get_cc_and_nvle_status", get_status)

    assert confidential_compute_enabled() is status[0]


def test_get_cc_and_nvle_status_propagates_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pynvml = _make_pynvml(cc_feature=0, multi_gpu_mode=0)

    def raise_unexpected_error(_settings_ptr: _ConfComputeSettingsPointer) -> int:
        raise RuntimeError("unexpected NVML API mismatch")

    pynvml.nvmlSystemGetConfComputeSettings = raise_unexpected_error
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)

    with pytest.raises(RuntimeError, match="unexpected NVML API mismatch"):
        get_cc_and_nvle_status()


def test_get_cc_and_nvle_status_uses_legacy_cc_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    pynvml = _make_pynvml(
        cc_feature=0,
        multi_gpu_mode=0,
        settings_supported=False,
        legacy_cc_feature=1,
    )
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)

    assert get_cc_and_nvle_status() == (True, False)
