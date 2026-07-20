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

from unittest.mock import patch

import pynvml

from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._torch.modules.fused_moe.communication.deep_ep_low_latency import DeepEPLowLatency


def setup_function() -> None:
    MnnvlMemory._is_pcie_nvl_sku.cache_clear()
    MnnvlMemory.support_nvlink.cache_clear()


def teardown_function() -> None:
    MnnvlMemory._is_pcie_nvl_sku.cache_clear()
    MnnvlMemory.support_nvlink.cache_clear()


@patch("tensorrt_llm._mnnvl_utils.torch.cuda.get_device_name", return_value="NVIDIA H200 NVL")
def test_pcie_nvl_sku_detected_by_name(mock_get_device_name) -> None:
    with patch.object(MnnvlMemory, "_ensure_nvml_initialized") as mock_initialize:
        assert MnnvlMemory._is_pcie_nvl_sku(0)

    mock_initialize.assert_not_called()


@patch("tensorrt_llm._mnnvl_utils.torch.cuda.get_device_name", return_value="NVIDIA H200")
@patch.object(MnnvlMemory, "_ensure_nvml_initialized")
@patch("tensorrt_llm._mnnvl_utils.pynvml.nvmlDeviceGetCount", return_value=8)
@patch(
    "tensorrt_llm._mnnvl_utils.pynvml.nvmlDeviceGetHandleByIndex", side_effect=lambda index: index
)
def test_split_nvlink_topology_detected(
    mock_get_handle, mock_get_count, mock_initialize, mock_get_device_name
) -> None:
    def common_ancestor(_self_handle, peer_handle):
        if peer_handle >= 4:
            return pynvml.NVML_TOPOLOGY_SYSTEM
        return pynvml.NVML_TOPOLOGY_NODE

    with patch(
        "tensorrt_llm._mnnvl_utils.pynvml.nvmlDeviceGetTopologyCommonAncestor",
        side_effect=common_ancestor,
    ):
        assert MnnvlMemory._is_pcie_nvl_sku(0)


@patch("tensorrt_llm._mnnvl_utils.torch.cuda.get_device_name", return_value="NVIDIA H200")
@patch.object(MnnvlMemory, "_ensure_nvml_initialized")
@patch("tensorrt_llm._mnnvl_utils.pynvml.nvmlDeviceGetCount", return_value=8)
@patch(
    "tensorrt_llm._mnnvl_utils.pynvml.nvmlDeviceGetHandleByIndex", side_effect=lambda index: index
)
@patch(
    "tensorrt_llm._mnnvl_utils.pynvml.nvmlDeviceGetTopologyCommonAncestor",
    return_value=pynvml.NVML_TOPOLOGY_NODE,
)
def test_nvswitch_topology_remains_supported(
    mock_common_ancestor,
    mock_get_handle,
    mock_get_count,
    mock_initialize,
    mock_get_device_name,
) -> None:
    assert not MnnvlMemory._is_pcie_nvl_sku(0)


def test_topology_probe_initializes_nvml() -> None:
    with (
        patch(
            "tensorrt_llm._mnnvl_utils.torch.cuda.get_device_name",
            return_value="NVIDIA H200",
        ),
        patch(
            "tensorrt_llm._mnnvl_utils.pynvml.nvmlDeviceGetCount",
            side_effect=[pynvml.NVMLError_Uninitialized(), 1],
        ),
        patch("tensorrt_llm._mnnvl_utils.pynvml.nvmlInit") as mock_nvml_init,
        patch("tensorrt_llm._mnnvl_utils.pynvml.nvmlDeviceGetHandleByIndex", return_value=0),
    ):
        assert not MnnvlMemory._is_pcie_nvl_sku(0)

    mock_nvml_init.assert_called_once_with()


@patch("tensorrt_llm._mnnvl_utils.get_sm_version", return_value=90)
@patch("tensorrt_llm._mnnvl_utils.torch.cuda.current_device", return_value=0)
@patch.object(MnnvlMemory, "_is_pcie_nvl_sku", return_value=True)
@patch.object(MnnvlMemory, "support_nvlink")
def test_supports_mnnvl_rejects_split_topology(
    mock_support_nvlink, mock_is_pcie_nvl_sku, mock_current_device, mock_get_sm_version
) -> None:
    assert not MnnvlMemory.supports_mnnvl()
    mock_support_nvlink.assert_not_called()


@patch("tensorrt_llm._mnnvl_utils.get_sm_version", return_value=90)
@patch("tensorrt_llm._mnnvl_utils.torch.cuda.current_device", return_value=0)
@patch.object(MnnvlMemory, "_is_pcie_nvl_sku", return_value=False)
@patch.object(MnnvlMemory, "support_nvlink", return_value=True)
def test_supports_mnnvl_accepts_full_fabric(
    mock_support_nvlink, mock_is_pcie_nvl_sku, mock_current_device, mock_get_sm_version
) -> None:
    assert MnnvlMemory.supports_mnnvl()
    mock_support_nvlink.assert_called_once_with(0, True)


@patch(
    "tensorrt_llm._torch.modules.fused_moe.communication.deep_ep_low_latency.deep_ep_installed",
    True,
)
@patch(
    "tensorrt_llm._torch.modules.fused_moe.communication.deep_ep_low_latency.get_sm_version",
    return_value=90,
)
@patch.object(MnnvlMemory, "supports_mnnvl", return_value=False)
def test_deep_ep_low_latency_rejects_split_topology(
    mock_supports_mnnvl, mock_get_sm_version
) -> None:
    assert not DeepEPLowLatency.is_platform_supported()


@patch(
    "tensorrt_llm._torch.modules.fused_moe.communication.deep_ep_low_latency.deep_ep_installed",
    True,
)
@patch(
    "tensorrt_llm._torch.modules.fused_moe.communication.deep_ep_low_latency.get_sm_version",
    return_value=90,
)
@patch.object(MnnvlMemory, "supports_mnnvl", return_value=True)
def test_deep_ep_low_latency_accepts_full_fabric(mock_supports_mnnvl, mock_get_sm_version) -> None:
    assert DeepEPLowLatency.is_platform_supported()
