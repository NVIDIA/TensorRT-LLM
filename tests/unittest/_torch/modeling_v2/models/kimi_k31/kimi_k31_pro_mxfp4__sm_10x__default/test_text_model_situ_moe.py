# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routed MoE: ``K31MoE`` with the family routed experts (``K31TargetRoutedExperts``) on the
TRTLLM-Gen SiTU runner, and its checkpoint helpers.

Covers the SiTU soft-cap validation, the routed-expert MoE split and backend selection and the runtime built with
the default W4A8_MXFP4_MXFP8 routed experts. The routed-expert checkpoint layout is tested with the rest of
``weights.py`` in ``test_weights.py``. The fused SiTU MoE kernels are certified by the catalog block-scale MoE runner
cells.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling as modeling_kimi_linear  # noqa: E501
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (
    K31MoE,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

situ_supported = pytest.mark.skipif(
    not torch.cuda.is_available() or not 100 <= get_sm_version() < 110,
    reason="native SiTU cubins require Blackwell SM10x",
)


@dataclasses.dataclass
class _K3Config:
    """Minimal config carrying the fields K31MoE reads."""

    hidden_size: int = 512
    num_experts: int = 8
    num_experts_per_token: int = 2
    moe_intermediate_size: int = 256
    moe_renormalize: bool = True
    moe_router_activation_func: str = "sigmoid"
    routed_scaling_factor: float = 1.0
    num_expert_group: int = 1
    topk_group: int = 1
    num_shared_experts: int = None
    routed_expert_hidden_size: int = None
    latent_moe_use_norm: bool = False
    rms_norm_eps: float = 1e-5
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0


def test_kimi_situ_betas_require_linear_beta():
    cfg = SimpleNamespace(activation_situ_beta=4.0, activation_situ_linear_beta=None)

    with pytest.raises(ValueError, match="require activation_situ_linear_beta"):
        modeling_kimi_linear._resolve_kimi_situ_betas(cfg)


@pytest.mark.parametrize(
    "situ_beta,situ_linear_beta",
    [(0.0, 25.0), (4.0, 0.0), (-1.0, 25.0), (4.0, -1.0)],
)
def test_kimi_situ_betas_must_be_positive(situ_beta, situ_linear_beta):
    cfg = SimpleNamespace(
        activation_situ_beta=situ_beta,
        activation_situ_linear_beta=situ_linear_beta,
    )

    with pytest.raises(ValueError, match="must be positive"):
        modeling_kimi_linear._resolve_kimi_situ_betas(cfg)


def test_kimi_k3_moe_split_selection() -> None:
    # Auto mapping default stays EP-only (the historical K3 layout), even
    # though the resolved mapping says moe_tp=8.
    auto = Mapping(world_size=8, tp_size=8)
    assert K31MoE._select_moe_tp_ep(auto) == (1, 8)

    # Explicit pure-TP and hybrid requests are honored. The MoE split is only
    # ever set through the config; there is no env override.
    tp = Mapping(world_size=8, tp_size=8, moe_tp_size=8, moe_ep_size=1)
    assert K31MoE._select_moe_tp_ep(tp) == (8, 1)
    tep = Mapping(world_size=8, tp_size=8, moe_tp_size=4, moe_ep_size=2)
    assert K31MoE._select_moe_tp_ep(tep) == (4, 2)


def test_kimi_k3_routed_mapping_keeps_the_shared_config():
    mapping = Mapping(world_size=16, rank=3, tp_size=16, moe_ep_size=16, enable_attention_dp=True)
    model_config = ModelConfig(mapping=mapping, moe_backend="TRTLLM")

    routed = K31MoE._routed_mapping(model_config)

    assert (routed.moe_tp_size, routed.moe_ep_size, routed.moe_ep_rank) == (1, 16, 3)
    assert model_config.mapping is mapping


@pytest.mark.parametrize("backend", ["TRITON", "CUTLASS", "CUTEDSL", "MEGAMOE_CUTEDSL"])
def test_kimi_k3_routed_config_rejects_backends_other_than_trtllm(backend):
    model_config = ModelConfig(
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        moe_backend=backend,
    )

    with pytest.raises(ValueError, match="TRTLLM"):
        K31MoE._routed_mapping(model_config)

    assert model_config.moe_backend == backend


@situ_supported
def test_kimi_k3_trtllm_builds_mxfp4_routed_experts():
    """With no per-layer routed-expert quantization (the K3.1 checkpoint), the runtime builds the TRTLLM-Gen
    W4A8_MXFP4_MXFP8 routed experts and reads the packed MXFP4 checkpoint layout."""
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.weights import (  # noqa: E501
        EXPERT_CKPT_SPECS,
    )

    pretrained_config = PretrainedConfig()
    pretrained_config.torch_dtype = torch.bfloat16
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        moe_backend="TRTLLM",
    )
    cfg = _K3Config(
        routed_expert_hidden_size=512,
        latent_moe_use_norm=True,
        num_shared_experts=1,
    )

    runtime = K31MoE(
        model_config,
        cfg,
        layer_idx=0,
        aux_stream_dict={stream_type: torch.cuda.Stream() for stream_type in AuxStreamType},
    )

    assert runtime.expert_ckpt_spec is EXPERT_CKPT_SPECS[QuantAlgo.W4A8_MXFP4_MXFP8]
    assert runtime.expert_ckpt_spec.kinds == ("weight_packed", "weight_scale")
    experts = runtime.routed_experts
    assert isinstance(experts, modeling_kimi_linear.K31TargetRoutedExperts)
    # The TRTLLM-Gen runner layout: [local, 2 x inter, hidden / 2] MXFP4 codes and group-32 UE8M0 scales.
    assert (
        experts.w3_w1_weight.shape == (8, 2 * 256, 256)
        and experts.w3_w1_weight.dtype == torch.uint8
    )
    assert experts.w2_weight.shape == (8, 512, 128) and experts.w2_weight_scale.shape == (8, 512, 8)
