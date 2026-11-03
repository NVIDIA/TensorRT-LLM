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
"""Number of DWDP prefetch slots: config, slot assignment and slot sizing."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tensorrt_llm._torch.modules.dwdp.page_pool import compute_slot_sizes
from tensorrt_llm._torch.modules.dwdp.weight_buffer import WeightBuffer
from tensorrt_llm.llmapi.llm_args import DwdpConfig

pytestmark = pytest.mark.cpu_only

LAYERS = [3, 4, 6, 7, 9]


def _buffer(num_slots: int) -> WeightBuffer:
    # The constructor only records the layout inputs; ``create`` maps memory.
    return WeightBuffer(
        layer_weight_specs={layer: {} for layer in LAYERS},
        handles=None,
        local_start=0,
        local_end=4,
        dwdp_size=2,
        device_id=0,
        granularity=1 << 21,
        num_slots=num_slots,
    )


def test_config_defaults_to_two_slots_and_rejects_fewer():
    assert DwdpConfig().num_prefetch_slots == 2
    assert DwdpConfig(num_prefetch_slots=3).num_prefetch_slots == 3
    with pytest.raises(ValidationError):
        DwdpConfig(num_prefetch_slots=1)


@pytest.mark.parametrize("num_slots", [2, 3, 4])
def test_slot_is_moe_order_modulo_num_slots(num_slots):
    buffer = _buffer(num_slots)
    assert buffer.num_slots == num_slots
    assert [buffer.buffer_index_for_layer(layer) for layer in LAYERS] == [
        order % num_slots for order in range(len(LAYERS))
    ]


def test_weight_buffer_rejects_fewer_than_two_slots():
    with pytest.raises(ValueError):
        _buffer(1)


def test_slot_of_a_non_moe_layer_raises():
    with pytest.raises(KeyError):
        _buffer(2).buffer_index_for_layer(5)


def test_slot_size_is_the_largest_layer_of_the_slot():
    def layout(pre: int, post: int) -> SimpleNamespace:
        return SimpleNamespace(pre_size=pre, post_size=post)

    layouts = {
        3: {"w1": layout(10, 0), "w2": layout(5, 5)},
        4: {"w1": layout(30, 0)},
        6: {"w1": layout(0, 7)},
        7: {"w1": layout(1, 1)},
        9: {"w1": layout(0, 50)},
    }
    assignments = {layer: order % 3 for order, layer in enumerate(LAYERS)}
    assert compute_slot_sizes(layouts, assignments, 3) == [20, 50, 7]
    assignments = {layer: order % 2 for order, layer in enumerate(LAYERS)}
    assert compute_slot_sizes(layouts, assignments, 2) == [50, 30]
