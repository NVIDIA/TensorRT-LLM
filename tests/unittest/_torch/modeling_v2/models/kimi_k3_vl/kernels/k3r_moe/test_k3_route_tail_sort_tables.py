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
"""The CuTe routing tail's ``moe_sort`` tables (``ops.moe.k3_route_tail``),
which the gate hands to the routed experts as ``SortTables`` in place of
their own ``moe_sort``: for the tail's top-k, the same tiles as ``trtllm::moe_sort`` (count,
expert, row limit) and every routed row of this rank's experts in a slot of its expert's rows
there, with the inverse map back to the row."""

import pytest
import torch

K3_RUBIN = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
NUM_EXPERTS = 896
TOP_K = 16

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 256])
@pytest.mark.parametrize(
    "local_expert_offset, local_num_experts",
    [(0, 896), (224, 224), (672, 224), (280, 56)],
)
def test_route_tail_tables_are_moe_sort_tables(num_tokens, local_expert_offset, local_num_experts):
    route_tail = pytest.importorskip(f"{K3_RUBIN}.kernels.k3r_moe.k3_route_tail")
    entry = pytest.importorskip(f"{K3_RUBIN}.ops.moe.k3_route_tail").k3_route_tail
    device = torch.device("cuda")
    if not route_tail.supports_device(device):
        pytest.skip("k3_route_tail runs on SM100 / SM103 / SM107")
    generator = torch.Generator(device=device).manual_seed(num_tokens)
    logits = torch.randn(
        num_tokens, NUM_EXPERTS, device=device, dtype=torch.float32, generator=generator
    )
    bias = 0.1 * torch.randn(NUM_EXPERTS, device=device, dtype=torch.float32, generator=generator)

    outs = entry(logits, bias, 2.5, local_expert_offset, local_num_experts, route_tail.TILE)
    topk_ids, topk_weights = outs[0], outs[1]
    ref = torch.ops.trtllm.moe_sort(
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        tile_tokens_dim=route_tail.TILE,
    )
    torch.cuda.synchronize()
    tile_expert, tile_limit, e2p, p2e, total, num_tiles = outs[2:8]
    ref_tile_expert, ref_tile_limit, ref_e2p, _, ref_total, ref_num_tiles = ref

    assert int(num_tiles.item()) == int(ref_num_tiles.item())
    assert int(total.item()) == int(ref_total.item())
    tiles = int(ref_num_tiles.item())
    assert torch.equal(tile_expert[:tiles], ref_tile_expert[:tiles])
    assert torch.equal(tile_limit[:tiles], ref_tile_limit[:tiles])

    experts = topk_ids.reshape(-1)
    local = (experts >= local_expert_offset) & (experts < local_expert_offset + local_num_experts)
    rows = local.nonzero().flatten()
    slots = e2p.reshape(-1)[rows].long()
    ref_slots = ref_e2p.reshape(-1)[rows].long()
    for expert in experts[rows].unique():
        mine = experts[rows] == expert
        assert torch.equal(slots[mine].sort().values, ref_slots[mine].sort().values)
    assert torch.equal(p2e[slots].long(), rows)
