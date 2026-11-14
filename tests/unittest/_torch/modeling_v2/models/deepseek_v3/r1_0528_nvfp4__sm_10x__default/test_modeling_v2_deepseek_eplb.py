# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the default deepseek target registers with the engine's load balancer under EPLB.

The target is built under meta init, so no weight is allocated, against a real load balancer,
static or online, at one rank of dep4. The registrations are the target's half of the
contract: the loader finalizes the balancer and the executor drives it per forward, and both
rely on every routed MoE layer being registered once, in the order the layers run, with the
placement the target loads its weights by.
"""

from __future__ import annotations

import os
import random

import pytest
import torch
from transformers import DeepseekV3Config

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v3.r1_0528_nvfp4__sm_10x__default.modeling import (  # noqa: E501
    DeepseekV3DefaultModelingV2,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancer
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig, MTPDecodingConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the target is certified on sm_100 and sm_103",
)

# DeepSeek-R1-0528's config.json, the fields the target reads.
_R1 = dict(
    architectures=["DeepseekV3ForCausalLM"],
    num_hidden_layers=61,
    hidden_size=7168,
    num_attention_heads=128,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    kv_lora_rank=512,
    q_lora_rank=1536,
    vocab_size=129280,
    first_k_dense_replace=3,
    n_routed_experts=256,
    num_experts_per_tok=8,
    n_group=8,
    topk_group=4,
    routed_scaling_factor=2.5,
    moe_intermediate_size=2048,
    n_shared_experts=1,
    intermediate_size=18432,
    num_nextn_predict_layers=1,
    max_position_embeddings=163840,
    rms_norm_eps=1e-6,
    rope_theta=10000,
    rope_scaling=dict(
        type="yarn",
        factor=40,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
        mscale_all_dim=1.0,
    ),
    dtype=torch.bfloat16,
)
_EP, _RANK, _SLOTS = 4, 1, 288


def _placement(layer: int) -> list:
    """A per-layer placement of 288 slots: the experts shuffled, then 8 spare slots a rank."""
    order = list(range(256))
    random.Random(layer).shuffle(order)
    out = []
    for rank in range(_EP):
        own = order[rank * 64 : (rank + 1) * 64]
        out += own + [own[0]] + order[(rank + 1) * 64 % 256 :][:7]
    return out


@pytest.mark.parametrize("updates", [0, 2], ids=["static", "online"])
@pytest.mark.parametrize("mtp", [False, True], ids=["no_spec", "mtp3"])
def test_eplb_registers_every_routed_layer(mtp, updates):
    layers = range(3, 61 + int(mtp))
    assignments = {i: _placement(i) for i in layers}
    load_balancer = MoeLoadBalancerConfig(
        num_slots=_SLOTS, initial_global_assignments=assignments, layer_updates_per_iter=updates
    )
    load_balancer.setup(ep_rank=_RANK, ep_size=_EP)
    model_config = ModelConfig(
        pretrained_config=DeepseekV3Config(**_R1),
        mapping=Mapping(
            world_size=_EP,
            rank=_RANK,
            tp_size=_EP,
            moe_tp_size=1,
            moe_ep_size=_EP,
            enable_attention_dp=True,
        ),
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, kv_cache_quant_algo=QuantAlgo.FP8),
        moe_load_balancer=load_balancer,
        spec_config=MTPDecodingConfig(max_draft_len=3) if mtp else None,
    )

    balancer = MoeLoadBalancer(
        ep_rank=_RANK,
        ep_size=_EP,
        layer_updates_per_iter=updates,
        shared_memory_base_name=f"mv2_ds_eplb_{os.getpid()}",
    )
    try:
        with balancer, MetaInitMode():
            core = DeepseekV3DefaultModelingV2(model_config)

        registered = balancer.single_layer_load_balancers
        assert list(core.eplb_layers) == list(layers)
        assert [eplb.balancer_layer for eplb in core.eplb_layers.values()] == registered
        repeats = [layer.get_repeat_count() for layer in registered]
        assert repeats == [1] * 58 + ([3] if mtp else [])
        for i, layer in zip(layers, registered):
            placement = layer.single_layer_load_balancer_impl.get_old_rank_expert_ids()
            assert sum(placement, []) == assignments[i], i

        mode = "online" if updates else "static"
        assert [eplb.online for eplb in core.eplb_layers.values()] == [bool(updates)] * len(layers)
        d = core.dims
        assert (d.eplb, d.num_slots, d.local_slots, d.slot_offset) == (mode, 288, 72, 72)
        for i in layers:
            mine = tuple(assignments[i][72:144])
            assert d.slots[i].slot_experts == mine
            name = "fc1_w" if i < 61 else "fc1"
            assert core.w[f"l{i}_{name}"].shape[0] == 72
    finally:
        balancer.shutdown()
