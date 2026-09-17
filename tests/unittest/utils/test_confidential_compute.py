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

from tensorrt_llm._utils import confidential_compute_enabled


class _ConfComputeSettings(ctypes.Structure):
    _fields_ = [
        ("ccFeature", ctypes.c_uint),
        ("multiGpuMode", ctypes.c_uint),
    ]


def _make_pynvml(cc_feature: int, multi_gpu_mode: int) -> types.ModuleType:
    # The real NVML C API declares
    # nvmlSystemGetConfComputeSettings(nvmlSystemConfComputeSettings_v1_t *) and expects a
    # pointer argument. This fake query only writes through settings_arg._obj, which only
    # exists on a ctypes.byref(...) pointer -- passing the struct by value (the original bug)
    # makes the query raise AttributeError instead, reproducing the observed failure mode.
    pynvml = types.ModuleType("pynvml")
    pynvml.NVMLError_NotSupported = type("NVMLError_NotSupported", (Exception,), {})
    pynvml.NVML_SUCCESS = 0
    pynvml.NVML_CC_SYSTEM_FEATURE_ENABLED = 1
    pynvml.NVML_CC_SYSTEM_MULTIGPU_PROTECTED_PCIE = 1
    pynvml.NVML_CC_SYSTEM_MULTIGPU_NVLE = 2
    pynvml.c_nvmlSystemConfComputeSettings_v1_t = _ConfComputeSettings
    pynvml.nvmlInit = lambda: None
    pynvml.nvmlShutdown = lambda: None

    def get_settings(settings_arg) -> int:
        settings_arg._obj.ccFeature = cc_feature
        settings_arg._obj.multiGpuMode = multi_gpu_mode
        return pynvml.NVML_SUCCESS

    pynvml.nvmlSystemGetConfComputeSettings = get_settings
    return pynvml


@pytest.mark.parametrize(
    "cc_feature,multi_gpu_mode,expected",
    [
        (1, 0, True),  # ccFeature enabled
        (0, 1, True),  # PROTECTED_PCIE
        (0, 2, True),  # NVLE
        (1, 2, True),  # ccFeature enabled and NVLE
    ],
)
def test_confidential_compute_enabled(monkeypatch, cc_feature, multi_gpu_mode, expected):
    pynvml = _make_pynvml(cc_feature, multi_gpu_mode)
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)

    assert confidential_compute_enabled() is expected
