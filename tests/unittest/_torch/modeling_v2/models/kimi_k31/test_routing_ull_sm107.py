# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routing of the ULL generation layout (Helix CP16, MoE TP16 / EP1), on CPU.

``route`` sends that layout to ``_route_ull_generation`` (the
``kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1`` target) before the ModelOpt export's other
targets are considered, so the configurable default target (``kimi_k31_pro_nvfp4__sm_10x__default``)
never takes it, whatever recipe the deployment asks for and whatever that target admits. Other
layouts never reach the ULL route. The fixtures are ``test_routing_sm107.py``'s.
"""

import pytest
import test_routing_sm107 as _sm107

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31 import routing

pytestmark = pytest.mark.cpu_only

_ULL16 = dict(
    world_size=16,
    tp_size=1,
    cp_size=16,
    cp_config={"cp_type": "HELIX"},
    moe_tp_size=16,
    moe_ep_size=1,
)
_ULL_TARGET = type("ModelingV2KimiK31ProNvfp4Sm107Helix16Moetp16ep1", (), {})


@pytest.fixture(autouse=True)
def rubin(monkeypatch):
    """Route as on Rubin with stand-ins for the sm_107 targets and the configurable default target,
    a generation server, and the two routes recorded."""
    _sm107._on_gpu(monkeypatch, (10, 7))
    _sm107._stand_in_targets(monkeypatch)
    monkeypatch.setattr(routing, "disagg_role", lambda llm_args=None: "generation")
    calls = []

    def ull(model_config, llm_args, indexer_k_dtype):
        calls.append(("ull", indexer_k_dtype))
        return _ULL_TARGET

    real_modelopt = routing._route_modelopt

    def modelopt(*args, **kwargs):
        calls.append(("modelopt", None))
        return real_modelopt(*args, **kwargs)

    monkeypatch.setattr(routing, "_route_ull_generation", ull)
    monkeypatch.setattr(routing, "_route_modelopt", modelopt)
    return calls


@pytest.mark.parametrize(
    "kv_dtype, indexer_k_dtype, ssm_dtype",
    [
        ("fp8", "fp8", "float32"),  # the ULL recipe
        ("nvfp4", "fp8", "float32"),  # values only the default target serves elsewhere
        ("fp8", "fp4", "bfloat16"),
    ],
)
@pytest.mark.parametrize("capability", [(10, 7), (10, 3)])
def test_the_ull_layout_goes_to_the_ull_route_first(
    rubin, monkeypatch, tmp_path, kv_dtype, indexer_k_dtype, ssm_dtype, capability
):
    _sm107._on_gpu(monkeypatch, capability)
    config = _sm107._modelopt_config(tmp_path, _ULL16)
    args = _sm107._llm_args(
        backend="CUTEDSL",
        kv_dtype=kv_dtype,
        indexer_k_dtype=indexer_k_dtype,
        ssm_dtype=ssm_dtype,
    )
    assert routing.route(config, args, capability) is _ULL_TARGET
    assert rubin == [("ull", indexer_k_dtype)]


@pytest.mark.parametrize("layout", sorted(_sm107._LAYOUTS))
def test_other_layouts_never_reach_the_ull_route(rubin, tmp_path, layout):
    config = _sm107._modelopt_config(tmp_path, _sm107._LAYOUTS[layout])
    try:
        routing.route(config, _sm107._llm_args(backend="AUTO"), (10, 7))
    except routing.NoTarget:
        pass
    assert rubin == [("modelopt", None)]
