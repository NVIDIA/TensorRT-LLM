# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""On the GPU, a batch with context requests adds only its generation tokens' routes to the
statistics: the layer records the generation rows of every chunk with the statistics op, the route
launch still maps every row, and a chunk without generation rows still counts the forward."""

import pytest
import torch
from _torch.moe import adaptive_eplb_reference as reference
from _torch.moe.test_adaptive_eplb_kernels import TOP_K, _dsv4_case, needs_gpu

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.balancer import AdaptiveMoeLoadBalancer
from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig

EXPERTS, SLOTS, EP_SIZE, EP_RANK = 384, 512, 32, 3


def _layer():
    config = MoeAdaptiveLoadBalancerConfig(num_slots=SLOTS)
    config.setup(ep_rank=EP_RANK, ep_size=EP_SIZE)
    balancer = AdaptiveMoeLoadBalancer(
        ep_rank=EP_RANK, ep_size=EP_SIZE, config=config, mapping=None
    )
    layer = balancer.add_layer(EXPERTS, TOP_K, SLOTS // EP_SIZE)
    layer.prepare_device_state(torch.int32, "cuda")
    return balancer, layer


@needs_gpu
@pytest.mark.parametrize("chunks", [(40,), (2, 38), (25, 15)])
def test_a_mixed_batch_counts_the_generation_rows_of_every_chunk(chunks):
    _, _, ids = _dsv4_case(5)
    ids = torch.tensor(ids[: 40 * TOP_K], dtype=torch.int32, device="cuda").view(40, TOP_K)
    context, stop = 24, 36  # 24 context tokens, 12 generation tokens, 4 padding rows
    balancer, layer = _layer()
    balancer.observe_scheduled_batch(
        3, False, False, lambda start, stop: (0, 0, 0), (context, stop)
    )
    routed = []
    for first, part in zip((True, False, False), ids.split(list(chunks))):
        layer.update_statistic_with_local_ids(part, first, False)
        routed.append(layer.route(part))
    torch.cuda.synchronize()
    want_rows = torch.zeros(EXPERTS)
    want_forwards = torch.zeros(1, dtype=torch.int64)
    reference.record(
        ids[context:stop].cpu(), want_rows, want_forwards, torch.ones(1, dtype=torch.int32)
    )
    assert torch.equal(layer.statistics.rows.cpu(), want_rows)
    assert int(layer.statistics.forwards.item()) == 1
    # The route still maps every row, as a generation-only batch's route does.
    _, plain = _layer()
    torch.cuda.synchronize()
    assert torch.equal(torch.cat(routed), plain.route(ids))
