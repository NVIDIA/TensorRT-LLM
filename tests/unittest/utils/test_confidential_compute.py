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

import pytest

from tensorrt_llm._utils import confidential_compute_enabled, get_cc_and_nvle_status


class _NvmlError(Exception):
    pass


class _NvmlErrorNotSupported(_NvmlError):
    pass


class _ConfComputeSettings(ctypes.Structure):
    _fields_ = [
        ("ccFeature", ctypes.c_uint),
        ("multiGpuMode", ctypes.c_uint),
    ]


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

    def get_settings(settings_ptr):
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
def clear_confidential_compute_status_cache():
    get_cc_and_nvle_status.cache_clear()
    yield
    get_cc_and_nvle_status.cache_clear()


@pytest.mark.parametrize(
    "cc_feature,multi_gpu_mode,expected",
    [
        (0, 0, (False, False)),
        (0, 1, (True, False)),
        (0, 2, (False, True)),
        (1, 2, (True, True)),
    ],
)
def test_get_cc_and_nvle_status(monkeypatch, cc_feature, multi_gpu_mode, expected):
    pynvml = _make_pynvml(
        cc_feature=cc_feature,
        multi_gpu_mode=multi_gpu_mode,
    )
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)

    assert get_cc_and_nvle_status() == expected


def test_get_cc_and_nvle_status_is_cached(monkeypatch):
    pynvml = _make_pynvml(cc_feature=0, multi_gpu_mode=2)
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)

    assert get_cc_and_nvle_status() == (False, True)
    assert get_cc_and_nvle_status() == (False, True)
    assert pynvml.settings_queries == 1


@pytest.mark.parametrize(
    "cc_feature,multi_gpu_mode,expected",
    [
        (0, 0, False),
        (0, 1, True),
        (0, 2, False),
        (1, 2, True),
    ],
)
def test_confidential_compute_enabled(monkeypatch, cc_feature, multi_gpu_mode, expected):
    pynvml = _make_pynvml(
        cc_feature=cc_feature,
        multi_gpu_mode=multi_gpu_mode,
    )
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)

    assert confidential_compute_enabled() is expected


def test_get_cc_and_nvle_status_uses_legacy_cc_fallback(monkeypatch):
    pynvml = _make_pynvml(
        cc_feature=0,
        multi_gpu_mode=0,
        settings_supported=False,
        legacy_cc_feature=1,
    )
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)

    assert get_cc_and_nvle_status() == (True, False)
