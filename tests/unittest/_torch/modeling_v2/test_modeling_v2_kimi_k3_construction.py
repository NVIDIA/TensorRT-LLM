# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 targets' construction checks, host-side: each target requires its own parallel layout, and
``tp16_moetp16ep1`` also an explicit expert split and no speculative decoding."""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    modeling as route_a,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1 import (  # noqa: E501
    modeling as route_b,
)
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.quantization.mode import QuantAlgo


def _config(moe_tp, moe_ep, split_set=True, spec_config=None, attention_dp=False):
    """The ModelConfig fields the construction checks read."""
    mapping = types.SimpleNamespace(
        world_size=16,
        tp_size=16,
        pp_size=1,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
        enable_attention_dp=attention_dp,
        moe_tp_ep_user_specified=split_set,
    )
    return types.SimpleNamespace(
        mapping=mapping,
        spec_config=spec_config,
        torch_dtype=torch.bfloat16,
        # The MXFP4 checkpoint's quantization, as the model config reads it.
        quant_config=types.SimpleNamespace(
            quant_algo=QuantAlgo.W4A16_MXFP4, kv_cache_quant_algo=None
        ),
        quant_config_dict=None,
        allreduce_strategy=AllReduceStrategy.AUTO,
    )


@pytest.fixture(autouse=True)
def sm_100(monkeypatch):
    """The targets' architecture, on any host."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))


@pytest.mark.parametrize(
    "target,parallel,split,other",
    [
        (route_a, "tp16_moetp4ep4", (4, 4), (16, 1)),
        (route_b, "tp16_moetp16ep1", (16, 1), (4, 4)),
    ],
    ids=["tp16_moetp4ep4", "tp16_moetp16ep1"],
)
def test_each_target_requires_its_layout(target, parallel, split, other):
    target._check_construction(_config(*split))
    with pytest.raises(AssertionError, match=f"the {parallel} target needs"):
        target._check_construction(_config(*other))
    with pytest.raises(AssertionError, match="enable_attention_dp false"):
        target._check_construction(_config(*split, attention_dp=True))


def test_tp16_moetp16ep1_requires_the_split_set_explicitly():
    with pytest.raises(AssertionError, match="set explicitly"):
        route_b._check_construction(_config(16, 1, split_set=False))


def test_tp16_moetp16ep1_decodes_without_speculation():
    with pytest.raises(
        AssertionError, match="moe_tensor_parallel_size 4 and moe_expert_parallel_size 4"
    ):
        route_b._check_construction(_config(16, 1, spec_config=object()))
