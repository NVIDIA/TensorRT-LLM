# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep target's routed MoE takes its expert placement from moe_config.load_balancer.

The runtime reads a layer's slot -> expert table from the engine's static EP load balancer configuration (identity
without one), refuses a placement where a routed call could leave the fused routed MoE, and builds the routed MoE
module without the engine's load balancer (contiguous identity partition; the target places the experts itself).
"""

import importlib
import random
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig
from tensorrt_llm.mapping import Mapping

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the dep target's MoE kernels are CuTe DSL kernels", allow_module_level=True)

_MODELING = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.modeling"
_NUM_EXPERTS = 896
_LAYER = 3


def _runtime_cls():
    return importlib.import_module(_MODELING).K31MoE


def _dep(world: int, rank: int, **kwargs) -> Mapping:
    return Mapping(
        world_size=world,
        rank=rank,
        tp_size=world,
        moe_ep_size=world,
        moe_tp_size=1,
        enable_attention_dp=True,
        **kwargs,
    )


def _placement(mapping: Mapping, load_balancer, max_num_tokens: int = 256) -> tuple:
    per_rank = _NUM_EXPERTS // mapping.moe_ep_size
    runtime = SimpleNamespace(
        routed_experts=SimpleNamespace(mapping=mapping),
        _fused_moe_max_tokens=None,
        num_experts=_NUM_EXPERTS,
        experts_per_rank=per_rank,
        expert_lo=mapping.moe_ep_rank * per_rank,
    )
    model_config = SimpleNamespace(moe_load_balancer=load_balancer, max_num_tokens=max_num_tokens)
    return _runtime_cls()._k31_expert_placement(runtime, model_config, _LAYER)


def _static(seed: int = 0) -> tuple[MoeLoadBalancerConfig, list[int]]:
    perm = list(range(_NUM_EXPERTS))
    random.Random(seed).shuffle(perm)
    return MoeLoadBalancerConfig(
        num_slots=_NUM_EXPERTS, initial_global_assignments={_LAYER: perm}
    ), perm


def test_no_load_balancer_is_the_identity_placement() -> None:
    assert _placement(_dep(4, 1), None) is None
    assert _placement(_dep(4, 1), MoeLoadBalancerConfig(num_slots=_NUM_EXPERTS)) is None


@pytest.mark.parametrize("world, rank", [(4, 2), (16, 9)])
def test_layer_placement_is_the_configured_assignment(world, rank) -> None:
    config, perm = _static(world)
    assert _placement(_dep(world, rank), config) == tuple(perm)


@pytest.mark.parametrize(
    "case, missed",
    [
        (dict(max_num_tokens=2048), "up to 1024 tokens per rank"),
        (dict(mapping=Mapping(world_size=4, rank=0, tp_size=4, moe_ep_size=4)), "attention DP"),
    ],
    ids=["max-num-tokens", "attention-tp"],
)
def test_placement_where_a_call_leaves_the_fused_path_raises(case, missed) -> None:
    case = dict(case)
    mapping = case.pop("mapping", _dep(4, 0))
    with pytest.raises(ValueError, match=missed):
        _placement(mapping, _static()[0], **case)


def test_redundant_slots_raise() -> None:
    with pytest.raises(ValueError, match="redundant expert slots"):
        _placement(_dep(4, 0), MoeLoadBalancerConfig(num_slots=_NUM_EXPERTS + 4))


def test_routed_experts_keep_the_identity_partition() -> None:
    """A static placement leaves the routed experts' mapping the contiguous EP partition (the target applies the
    placement itself) and does not touch the shared config."""
    from tensorrt_llm._torch.model_config import ModelConfig

    config, _ = _static()
    model_config = ModelConfig(mapping=_dep(4, 1), moe_backend="TRTLLM", moe_load_balancer=config)
    routed = _runtime_cls()._routed_mapping(model_config)
    assert model_config.moe_load_balancer is config
    assert routed.moe_ep_size == 4 and routed.moe_tp_size == 1 and routed.moe_ep_rank == 1
