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
"""DWDP over MoE layers of different sizes, shaped like Muse Spark Open.

Muse keeps the experts of its first and last MoE layers in BF16 inside an NVFP4 checkpoint and
nests its decoder under a multimodal wrapper (``model.llm.model``, ``layer.feed_forward``).
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.modules.dwdp.page_pool import assign_buffer_slots, compute_slot_sizes
from tensorrt_llm._torch.modules.dwdp.setup import (
    build_weight_specs,
    collect_moe_params,
    dense_layout,
    pack_dense_rows,
)

pytestmark = pytest.mark.cpu_only


def _layouts(remote_bytes):
    return {i: {"w": SimpleNamespace(pre_size=n, post_size=0)} for i, n in remote_bytes.items()}


def test_larger_layers_share_one_extra_slot():
    remote = {i: 5 for i in range(62)} | {0: 19, 61: 19}
    slots = assign_buffer_slots(_layouts(remote))

    assert slots == {0: 2, 61: 2} | {i: (i - 1) % 2 for i in range(1, 61)}
    assert compute_slot_sizes(_layouts(remote), slots) == [5, 5, 19]
    assert assign_buffer_slots(_layouts(remote), 3) == {0: 3, 61: 3} | {
        i: (i - 1) % 3 for i in range(1, 61)
    }
    assert assign_buffer_slots(_layouts({3: 5, 4: 5, 5: 5})) == {3: 0, 4: 1, 5: 0}


def test_collect_moe_params_reads_muse_layers():
    def layer(*weight_names):
        backend = nn.Module()
        for name in weight_names:
            backend.register_parameter(name, nn.Parameter(torch.zeros(64, 2), requires_grad=False))
        out = nn.Module()
        out.feed_forward = nn.Module()
        out.feed_forward.experts = nn.Module()
        out.feed_forward.experts.backend = backend
        return out

    bf16 = ("w3_w1_weight", "w2_weight")
    nvfp4 = bf16 + ("w3_w1_weight_scale", "w2_weight_scale")
    model = nn.Module()
    model.llm = nn.Module()
    model.llm.model = nn.Module()
    model.llm.model.layers = nn.ModuleList([layer(*bf16), layer(*nvfp4)])

    local_params, names = collect_moe_params(model, [0, 1])
    specs = build_weight_specs(local_params, names, mapping=None, num_experts_total=256)

    assert tuple(specs[0]) == bf16
    assert tuple(specs[1]) == nvfp4
    assert specs[1]["w2_weight_scale"].full_shape == (256, 2)


def test_dense_rows_compose_every_layer_blob():
    """The G ranks' rows of a layer, side by side in a slot, hold each weight at its offset."""
    torch.manual_seed(0)
    weights = [
        [torch.randn(64, 33).bfloat16(), torch.randn(100), torch.randn(3, 5).bfloat16()],
        [torch.randn(16, 16).bfloat16()],
    ]
    offsets, row = dense_layout(weights, num_ranks=4, granularity=512)
    rows = [torch.zeros(len(weights) * row, dtype=torch.uint8) for _ in range(4)]
    for rank, out in enumerate(rows):
        pack_dense_rows(weights, offsets, row, rank, out)

    assert row == 1536 and offsets == [[0, 4352, 4864], [0]]
    for i, (layer_weights, layer_offsets) in enumerate(zip(weights, offsets)):
        slot = torch.cat([out[i * row : (i + 1) * row] for out in rows])
        for w, o in zip(layer_weights, layer_offsets):
            assert torch.equal(slot[o : o + w.nbytes].view(w.dtype).view(w.shape), w)
