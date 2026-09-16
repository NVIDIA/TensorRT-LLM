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

import pytest

from tensorrt_llm._torch.moe.fused_moe.communication.moe_alltoall import (
    get_force_cft as get_force_cft_standalone,
)
from tensorrt_llm._torch.moe.fused_moe.communication.moe_alltoall import (
    should_use_cft as should_use_cft_standalone,
)
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import (
    FORCE_CFT_ENV,
    cft_driver_is_supported,
    get_force_cft,
    resolve_cft_counted_writes,
    should_use_cft,
)

# Environment-variable parsing only. The marker is also what makes the file
# reachable: the CPU stage collects only files that carry it.
pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("2", None),
        ("true", None),
        (" 1 ", None),
        ("0", False),
        ("1", True),
    ],
)
def test_get_force_cft(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool | None):
    if value is None:
        monkeypatch.delenv(FORCE_CFT_ENV, raising=False)
    else:
        monkeypatch.setenv(FORCE_CFT_ENV, value)

    assert get_force_cft() is expected
    assert get_force_cft_standalone() is expected


@pytest.mark.parametrize(
    ("can_use_cft", "force_cft", "runtime_max_tokens_per_rank", "expected"),
    [
        (True, None, 128, True),
        (True, None, 129, False),
        (True, False, 1, False),
        (True, True, 129, True),
        (False, True, 1, False),
        (False, None, 1, False),
    ],
)
def test_should_use_cft(
    can_use_cft: bool,
    force_cft: bool | None,
    runtime_max_tokens_per_rank: int,
    expected: bool,
):
    assert should_use_cft(can_use_cft, force_cft, 128, runtime_max_tokens_per_rank) is expected
    assert (
        should_use_cft_standalone(can_use_cft, force_cft, 128, runtime_max_tokens_per_rank)
        is expected
    )


@pytest.mark.parametrize(
    ("driver_version", "expected"),
    [
        ("610.47.04", False),
        (b"614.99", False),
        ("615.00", True),
        ("620.1", True),
        (None, False),
        ("unknown", False),
    ],
)
def test_cft_driver_is_supported(driver_version: str | bytes | None, expected: bool):
    assert cft_driver_is_supported(driver_version) is expected


@pytest.mark.parametrize(
    ("force_cft", "driver_version", "expected"),
    [
        (None, "610.47.04", False),
        (None, "615.00", True),
        (None, None, False),
        (False, "620.00", False),
        (True, "610.47.04", False),
        (True, "615.00", True),
        (True, "620.00", True),
        (True, None, False),
    ],
)
def test_resolve_cft_counted_writes(
    force_cft: bool | None,
    driver_version: str | bytes | None,
    expected: bool,
):
    assert resolve_cft_counted_writes(force_cft, driver_version) is expected


def test_get_nvidia_driver_version_reads_nvml(monkeypatch: pytest.MonkeyPatch):
    """A supported driver must be seen as supported, not just old ones rejected."""
    from tensorrt_llm._torch.moe.fused_moe.communication import nvlink_one_sided

    monkeypatch.setattr(nvlink_one_sided.pynvml, "nvmlDeviceGetCount", lambda: 1)
    monkeypatch.setattr(nvlink_one_sided.pynvml, "nvmlSystemGetDriverVersion", lambda: "615.00")

    version = nvlink_one_sided._get_nvidia_driver_version()
    assert version == "615.00"
    assert resolve_cft_counted_writes(True, version) is True


def test_get_nvidia_driver_version_returns_none_on_nvml_error(
    monkeypatch: pytest.MonkeyPatch,
):
    """An NVML failure must not be mistaken for a driver version."""
    from tensorrt_llm._torch.moe.fused_moe.communication import nvlink_one_sided

    def _raise():
        raise nvlink_one_sided.pynvml.NVMLError(nvlink_one_sided.pynvml.NVML_ERROR_UNKNOWN)

    monkeypatch.setattr(nvlink_one_sided.pynvml, "nvmlDeviceGetCount", lambda: 1)
    monkeypatch.setattr(nvlink_one_sided.pynvml, "nvmlSystemGetDriverVersion", _raise)

    assert nvlink_one_sided._get_nvidia_driver_version() is None


def test_cft_device_support_rejects_pre_blackwell(monkeypatch: pytest.MonkeyPatch):
    from tensorrt_llm._torch.moe.fused_moe.communication import nvlink_one_sided

    monkeypatch.setattr(nvlink_one_sided.torch.cuda, "get_device_capability", lambda: (9, 0))
    assert "SM90" in nvlink_one_sided._cft_device_support_reason()


@pytest.mark.parametrize("unsupported_index", [None, 0, 1, 2])
def test_cft_device_support_checks_required_capabilities(
    monkeypatch: pytest.MonkeyPatch, unsupported_index: int | None
):
    from tensorrt_llm._torch.moe.fused_moe.communication import nvlink_one_sided

    cuda = nvlink_one_sided.cuda
    attributes = (
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED,
    )
    unsupported = None if unsupported_index is None else attributes[unsupported_index]
    monkeypatch.setattr(nvlink_one_sided.torch.cuda, "get_device_capability", lambda: (10, 3))
    monkeypatch.setattr(nvlink_one_sided.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        cuda,
        "cuDeviceGetAttribute",
        lambda attribute, device: (cuda.CUresult.CUDA_SUCCESS, int(attribute != unsupported)),
    )
    reason = nvlink_one_sided._cft_device_support_reason()
    if unsupported is None:
        assert reason is None
    else:
        assert unsupported.name in reason
