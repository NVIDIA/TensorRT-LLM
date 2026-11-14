# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which parallel layouts the MiniMax-M3.1-Flash routing gives the DEP generation target, CPU only.

The DEP target takes one attention-DP group over 4, 8 or 16 ranks with the routed experts split over all of them by
expert parallelism; every other attention-DP layout and every layout without attention DP goes elsewhere or nowhere.
It is built only for a deployment that sets ``modeling_v2`` itself. The rest of ``route`` (checkpoint, capability,
KV cache, speculation) is ``test_modeling_v2_routing.py``'s.
"""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2 import _router_index
from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _dep(ep):
    return dict(world_size=ep, tp_size=ep, moe_tp_size=1, moe_ep_size=ep, enable_attention_dp=True)


@pytest.mark.parametrize("ep", [4, 8, 16])
def test_attention_dp_with_experts_over_every_rank_is_dep(ep):
    assert routing._m31_topology(Mapping(**_dep(ep))) == "dep"


@pytest.mark.parametrize(
    "mapping_kwargs, topology",
    [
        (
            dict(world_size=8, tp_size=8, moe_tp_size=2, moe_ep_size=4, enable_attention_dp=True),
            None,
        ),
        (
            dict(world_size=8, tp_size=8, moe_tp_size=8, moe_ep_size=1, enable_attention_dp=True),
            None,
        ),
        (
            dict(
                world_size=32, tp_size=32, moe_tp_size=1, moe_ep_size=32, enable_attention_dp=True
            ),
            None,
        ),
        (
            dict(
                world_size=16,
                tp_size=8,
                pp_size=2,
                moe_tp_size=1,
                moe_ep_size=8,
                enable_attention_dp=True,
            ),
            None,
        ),
        (dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8), None),
        (_dep(2), "dep2ctx"),
    ],
    ids=["adp8_ep4_tp2", "adp8_tp8_ep1", "adp32_ep32", "adp8_ep8_pp2", "ep8_no_adp", "adp2_ep2"],
)
def test_other_layouts_never_land_on_dep(mapping_kwargs, topology):
    assert routing._m31_topology(Mapping(**mapping_kwargs)) == topology


def test_the_dep_target_is_built_only_when_modeling_v2_is_set():
    try:
        from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep.modeling import (  # noqa: E501
            ModelingV2MiniMaxM31FlashNvfp4Sm107Dep,
        )
    except NoTarget as exc:
        pytest.skip(str(exc))
    assert ModelingV2MiniMaxM31FlashNvfp4Sm107Dep in _router_index._V2_DISABLED_BY_DEFAULT
