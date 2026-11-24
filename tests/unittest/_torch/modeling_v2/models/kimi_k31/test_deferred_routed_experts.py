# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deferred routed-expert weights of the Kimi K3.1 modeling_v2 targets under standalone dense DWDP.

The routed experts keep their weights on the meta device through the model load
(``model_loader.DEFER_MATERIALIZATION_ATTR``); the loader neither materializes nor moves them, and the target's
``weights.load_deferred_weights`` allocates and loads them after the dense DWDP setup.
"""

import types

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default import (
    weights as general_weights,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep import (
    weights as dep_weights,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import routed_experts
from tensorrt_llm._torch.modules.dwdp.dense import manager as dense_manager
from tensorrt_llm._torch.modules.dwdp.dense.types import DenseMode
from tensorrt_llm._torch.pyexecutor.model_loader import (
    DEFER_MATERIALIZATION_ATTR,
    _model_to_cuda,
    has_deferred_tensors,
    is_deferred_tensor,
)
from tensorrt_llm.mapping import Mapping

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


class _Pair(nn.Module):
    """A module with one deferred and one ordinary parameter, built on the meta device."""

    def __init__(self) -> None:
        super().__init__()
        with torch.device("meta"):
            self.deferred = nn.Parameter(torch.empty(4, 8), requires_grad=False)
            self.plain = nn.Parameter(torch.empty(4, 8), requires_grad=False)
        setattr(self.deferred, DEFER_MATERIALIZATION_ATTR, True)


@pytest.mark.cpu_only
def test_deferred_tensor_predicates() -> None:
    module = _Pair()
    assert is_deferred_tensor(module.deferred)
    assert not is_deferred_tensor(module.plain)
    assert has_deferred_tensors(module)
    # The flag names a meta tensor only: once materialized, the tensor is an ordinary one.
    cpu = torch.empty(2)
    setattr(cpu, DEFER_MATERIALIZATION_ATTR, True)
    assert not is_deferred_tensor(cpu)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "mode,defers", [(None, False), (DenseMode.STANDALONE, True), (DenseMode.ATTACHED, False)]
)
def test_only_standalone_dense_dwdp_defers_the_experts(monkeypatch, mode, defers) -> None:
    registered = None if mode is None else types.SimpleNamespace(mode=mode)
    monkeypatch.setattr(dense_manager, "get_global_dense_dwdp_manager", lambda: registered)
    assert routed_experts.dense_dwdp_defers_routed_experts() is defers


@_cuda
def test_model_to_cuda_leaves_the_deferred_tensors() -> None:
    module = _Pair()
    module.plain = nn.Parameter(torch.ones(4, 8), requires_grad=False)
    _model_to_cuda(module)
    assert module.plain.is_cuda
    assert module.deferred.device.type == "meta" and is_deferred_tensor(module.deferred)


def _routed_experts() -> routed_experts.K31RoutedExperts:
    """One layer's routed experts of a single-rank layout (no token exchange), built on the meta device."""
    model_config = types.SimpleNamespace(
        max_num_tokens=256,
        moe_max_num_tokens=256,
        torch_dtype=torch.bfloat16,
        use_low_precision_moe_combine=False,
        allreduce_strategy=None,
    )
    with torch.device("meta"):
        return routed_experts.K31RoutedExperts(
            model_config,
            Mapping(world_size=1, rank=0, tp_size=1),
            num_experts=16,
            hidden_size=256,
            intermediate_size=512,
            gate_cap=4.0,
            linear_cap=25.0,
            layer_idx=1,
            e_score_correction_bias=lambda: None,
            routed_scaling_factor=1.0,
        )


@_cuda
@pytest.mark.parametrize("target_weights", [general_weights, dep_weights], ids=["general", "dep"])
def test_deferred_experts_materialize_after_the_load(target_weights) -> None:
    experts = _routed_experts()
    experts.defer_weight_materialization()
    assert experts.weights_deferred
    assert all(is_deferred_tensor(p) for p in experts.parameters(recurse=False))
    # The post-load walk runs before the deferred load: no derived state on meta weights.
    experts.post_load_weights()
    assert experts.act_alpha is None

    moe = types.SimpleNamespace(routed_experts=experts)
    target_weights._materialize_deferred_experts([(moe, "model.layers.1.block_sparse_moe.experts")])
    assert not experts.weights_deferred
    assert all(p.is_cuda and not is_deferred_tensor(p) for p in experts.parameters(recurse=False))
    assert not has_deferred_tensors(experts)
    experts.cache_derived_state()
    assert experts.act_alpha.is_cuda and experts.act_alpha.shape == (
        experts.expert_size_per_partition,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("target_weights", [general_weights, dep_weights], ids=["general", "dep"])
def test_no_deferred_load_is_a_no_op(target_weights) -> None:
    llm = types.SimpleNamespace(_k31_deferred_expert_load=None)
    target_weights.load_deferred_weights(llm)
    assert llm._k31_deferred_expert_load is None
