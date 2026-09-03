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
"""Failure rollback tests for the DWDP setup ownership chain."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from tensorrt_llm._torch.modules.dwdp import setup as dwdp_setup_module


def _model():
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList()
    return model


def _mapping():
    return SimpleNamespace(dwdp_enabled=True, dwdp_rank=0, dwdp_size=2)


def _setup_mocks(stack: ExitStack):
    transport = MagicMock()
    transport.get_handle_set.return_value = MagicMock()
    transport.get_peer_ranges.return_value = [(0, 2), (2, 4)]
    transport.get_peer_views.return_value = {}
    weight_buffer = MagicMock()
    weight_manager = MagicMock()

    stack.enter_context(
        patch.object(dwdp_setup_module.torch.cuda, "mem_get_info", return_value=(1, 2))
    )
    stack.enter_context(
        patch.object(
            dwdp_setup_module,
            "collect_moe_params",
            return_value=({}, ["gate_up_proj", "down_proj"]),
        )
    )
    stack.enter_context(patch.object(dwdp_setup_module, "_get_model_num_experts", return_value=4))
    stack.enter_context(
        patch.object(dwdp_setup_module, "build_weight_specs", return_value=MagicMock())
    )
    stack.enter_context(
        patch.object(
            dwdp_setup_module,
            "_get_first_spec",
            return_value=SimpleNamespace(local_experts=2),
        )
    )
    stack.enter_context(patch.object(dwdp_setup_module, "_validate_partition_config"))
    stack.enter_context(
        patch.object(dwdp_setup_module.DWDPTransport, "create", return_value=transport)
    )
    stack.enter_context(
        patch.object(dwdp_setup_module.WeightBuffer, "create", return_value=weight_buffer)
    )
    stack.enter_context(patch.object(dwdp_setup_module, "fill_edge_bytes"))
    stack.enter_context(
        patch.object(dwdp_setup_module, "DWDPWeightManager", return_value=weight_manager)
    )
    stack.enter_context(patch.object(dwdp_setup_module, "fixup_moe_backends"))
    return transport, weight_buffer, weight_manager


def _run_setup():
    return dwdp_setup_module.setup_dwdp(
        model=_model(),
        mapping=_mapping(),
        device_id=0,
        comm=MagicMock(),
        layer_indices=[3],
        num_experts_per_worker=2,
        num_prefetch_experts=2,
    )


def test_weight_buffer_creation_failure_releases_transport():
    with ExitStack() as stack:
        transport, _weight_buffer, _weight_manager = _setup_mocks(stack)
        dwdp_setup_module.WeightBuffer.create.side_effect = RuntimeError("buffer failed")

        with pytest.raises(RuntimeError, match="buffer failed"):
            _run_setup()

    transport.release.assert_called_once_with()


def test_edge_fill_failure_releases_buffer_and_transport():
    with ExitStack() as stack:
        transport, weight_buffer, _weight_manager = _setup_mocks(stack)
        dwdp_setup_module.fill_edge_bytes.side_effect = RuntimeError("fill failed")

        with pytest.raises(RuntimeError, match="fill failed"):
            _run_setup()

    weight_buffer.release.assert_called_once_with()
    transport.release.assert_called_once_with()


def test_backend_fixup_failure_releases_owned_manager():
    with ExitStack() as stack:
        transport, _weight_buffer, weight_manager = _setup_mocks(stack)
        dwdp_setup_module.fixup_moe_backends.side_effect = RuntimeError("fixup failed")

        with pytest.raises(RuntimeError, match="fixup failed"):
            _run_setup()

    assert weight_manager._transport is transport
    weight_manager.release.assert_called_once_with()
