# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""CPU-mocked tests for explicit DWDP VMM teardown."""

from unittest.mock import MagicMock, call, patch

import torch.nn as nn

from tensorrt_llm._torch.modules.dwdp.setup import teardown_dwdp
from tensorrt_llm._torch.modules.dwdp.weight_manager import DWDPWeightManager


def test_weight_manager_release_is_ordered_and_idempotent():
    manager = DWDPWeightManager.__new__(DWDPWeightManager)
    manager._released = False
    manager._weight_buffer = MagicMock(device_id=0)
    manager._peer_views = {}
    manager._transport = MagicMock()
    manager._batched_copy_plans = {}
    order = []
    manager._weight_buffer.release.side_effect = lambda: order.append("weight_buffer")
    manager._transport.release.side_effect = lambda: order.append("transport")

    with patch(
        "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.synchronize",
        side_effect=lambda _device: order.append("synchronize"),
    ) as synchronize:
        transport = manager._transport
        manager.release()
        manager.release()

    assert order == ["synchronize", "weight_buffer", "transport"]
    assert synchronize.call_args_list == [call(0)]
    manager._weight_buffer.release.assert_called_once_with()
    transport.release.assert_called_once_with()


def _model():
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList()
    return model


def test_teardown_detaches_both_model_references_and_releases_once():
    model = _model()
    manager = MagicMock()
    model.dwdp_weight_manager = manager
    model.model.dwdp_weight_manager = manager

    teardown_dwdp(model)
    teardown_dwdp(model)

    assert model.dwdp_weight_manager is None
    assert model.model.dwdp_weight_manager is None
    manager.release.assert_called_once_with()


def test_teardown_expected_manager_preserves_foreign_references():
    model = _model()
    expected = MagicMock()
    foreign = MagicMock()
    model.dwdp_weight_manager = foreign
    model.model.dwdp_weight_manager = expected

    teardown_dwdp(model, expected_manager=expected)

    assert model.dwdp_weight_manager is foreign
    assert model.model.dwdp_weight_manager is None
    expected.release.assert_called_once_with()
    foreign.release.assert_not_called()


def test_teardown_releases_missing_expected_manager_without_touching_foreign():
    model = nn.Module()
    model.layers = nn.ModuleList()
    expected = MagicMock()
    foreign = MagicMock()
    model.dwdp_weight_manager = foreign

    teardown_dwdp(model, expected_manager=expected)

    assert model.dwdp_weight_manager is foreign
    expected.release.assert_called_once_with()
    foreign.release.assert_not_called()
