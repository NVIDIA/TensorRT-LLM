# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which parallel layouts the MiniMax-M3.1-Flash routing gives the context targets, CPU only.

The tp2ctx target takes exactly the disaggregated context role's layout (tp 2, moe_ep 2, attention DP off) and the
dep2ctx target the same layout with attention DP; wider expert parallelism, pipeline parallelism and the aggregated tp2
layout never land on either. The rest of ``route`` (checkpoint, capability, KV cache, speculation) is
``test_modeling_v2_routing.py``'s.
"""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_CTX = dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2)


@pytest.mark.parametrize(
    "mapping_kwargs, topology",
    [
        (_CTX, "tp2ctx"),
        (dict(_CTX, enable_attention_dp=True), "dep2ctx"),
        (dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4), None),
        (dict(world_size=4, tp_size=2, pp_size=2, moe_tp_size=1, moe_ep_size=2), None),
        (dict(world_size=2, tp_size=2, moe_tp_size=2, moe_ep_size=1), "tp2"),
    ],
    ids=["context_role", "attention_dp", "tp4_ep4", "tp2_ep2_pp2", "aggregated_tp2"],
)
def test_the_context_layouts(mapping_kwargs, topology):
    assert routing._m31_topology(Mapping(**mapping_kwargs)) == topology
