# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.configs.nemotron_shortcut import NemotronShortcutTextConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.routing import route
from tensorrt_llm._torch.speculative.utils import update_spec_config_from_model_config
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only


def _route(
    quant,
    world,
    moe_tp,
    *,
    state="auto",
    stochastic=False,
    capability=(10, 3),
    spec=None,
    settings=None,
    **recipe,
):
    text = NemotronShortcutTextConfig(
        dtype=torch.bfloat16,
        num_attention_heads=8,
        index_topk=2048,
        residual_in_fp32=True,
        moe_router_bias_space="logit",
        gdp_g_clamp_value=20.0,
        num_nextn_predict_layers=2,
        mtp_use_repeated_layer=True,
        moe_shortcut_disable_mtp=True,
        **(settings or {}),
    )
    if spec is not None:
        update_spec_config_from_model_config(spec, text)
    model = ModelConfig(
        pretrained_config=text,
        spec_config=spec,
        mapping=Mapping(
            world_size=world, tp_size=world, moe_tp_size=moe_tp, moe_ep_size=world // moe_tp
        ),
        quant_config=QuantConfig(quant_algo=quant, **recipe),
    )
    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        modeling_v2="require",
        kv_cache_config=KvCacheConfig(
            mamba_ssm_cache_dtype=state,
            mamba_ssm_stochastic_rounding=stochastic,
        ),
    )
    return route(model, args, capability)


@pytest.mark.parametrize(
    "quant,kv,state,target",
    [
        (None, None, "auto", "bf16__sm_103__default"),
        (QuantAlgo.MIXED_PRECISION, QuantAlgo.NVFP4, "float32", "nvfp4__sm_103__default"),
        (QuantAlgo.MXFP8, None, "auto", "mxfp8__sm_103__default"),
    ],
)
def test_four_rank_mixed_expert_parallelism_uses_default(quant, kv, state, target):
    cls = _route(quant, 4, 2, state=state, kv_cache_quant_algo=kv)
    assert cls.__module__.split(".")[-2] == target


@pytest.mark.parametrize(
    "quant,kv", [(QuantAlgo.MIXED_PRECISION, QuantAlgo.FP8), (None, QuantAlgo.NVFP4)]
)
def test_rejects_unsupported_quantized_kv(quant, kv):
    with pytest.raises(NoTarget, match="Quantized KV"):
        _route(quant, 4, 2, kv_cache_quant_algo=kv)


@pytest.mark.parametrize("quant,world", [(None, 2), (QuantAlgo.MIXED_PRECISION, 8)])
def test_rejects_unqualified_mapping(quant, world):
    with pytest.raises(NoTarget, match="text decoder mapping"):
        _route(quant, world, 2)


def test_rejects_unsupported_attention_head_size():
    with pytest.raises(NoTarget, match="unsupported mixer, residual or expert settings"):
        _route(None, 4, 2, settings={"head_dim": 96})


def test_rejects_explicit_incompatible_mxfp8_scale_block():
    with pytest.raises(NoTarget, match="32-element scale blocks"):
        _route(QuantAlgo.MXFP8, 4, 2, group_size=64)


def test_rejects_mtp_sampler_with_negative_padding_ids():
    spec = MTPDecodingConfig(max_draft_len=2, use_rejection_sampling=True)
    with pytest.raises(NoTarget, match="token IDs in every draft row"):
        _route(None, 4, 2, spec=spec)


@pytest.mark.parametrize("state,stochastic", [("bfloat16", False), ("float32", True)])
def test_requires_fp32_recurrent_state(state, stochastic):
    with pytest.raises(NoTarget, match="FP32 recurrent state"):
        _route(None, 4, 2, state=state, stochastic=stochastic)


def test_requires_qualified_gpu():
    with pytest.raises(NoTarget, match="SM103"):
        _route(QuantAlgo.MIXED_PRECISION, 4, 2, capability=(10, 0))
