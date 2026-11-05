# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE selection of the GLM-5.2 sm107 low-latency target's embedded MTP draft follows the
built-in one's.

``build_mtp_draft`` must apply ``speculative_config.moe_tactic`` / ``moe_backend`` and
the target's pin exactly as ``_build_mtp_one_model_draft`` does. CPU only: the draft
class is patched out and the backend resolution is stubbed, so no module is built.
"""

from types import SimpleNamespace
from unittest.mock import patch, sentinel

import pytest
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__low_latency import (
    _mtp,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import MoePerLayerDynamicLoadBalancerConfig
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_ARCH = "GlmMoeDsaForCausalLM"
_MTP_EXPERTS = "model.layers.2.mlp.experts"


def _target_config(
    moe_backend="TRTLLM",
    moe_tactic=None,
    spec_moe_backend=None,
    spec_moe_tactic=None,
    uses_replacement_heads=False,
) -> ModelConfig:
    config = ModelConfig(
        pretrained_config=PretrainedConfig(
            architectures=[_ARCH], hidden_size=64, vocab_size=128, num_hidden_layers=2
        ),
        moe_backend=moe_backend,
    )
    config.moe_tactic = moe_tactic
    config.spec_config = SimpleNamespace(
        moe_backend=spec_moe_backend,
        moe_tactic=spec_moe_tactic,
        uses_replacement_heads=uses_replacement_heads,
    )
    config.quant_config_dict = {
        _MTP_EXPERTS: QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
    }
    return config


def _draft_config_of(target_config: ModelConfig, draft_config=None) -> ModelConfig:
    """The config ``build_mtp_draft`` hands the draft class."""
    target_model = SimpleNamespace(aux_stream_dict={}, preload_weight_modules=[])
    with patch.object(_mtp, "Glm52MTPForCausalLM", return_value=sentinel.draft) as draft_cls:
        assert (
            _mtp.build_mtp_draft(target_config, draft_config, sentinel.lm_head, target_model)
            is sentinel.draft
        )
    return draft_cls.call_args.args[0]


def test_without_a_request_the_draft_reuses_the_target_config():
    target_config = _target_config()
    assert _draft_config_of(target_config) is target_config


def test_a_draft_tactic_pins_the_mtp_layers():
    target_config = _target_config(spec_moe_tactic="cutlass")
    with patch.object(ModelConfig, "pinned_moe_backend", return_value="CUTLASS") as pinned:
        draft_config = _draft_config_of(target_config)

    assert draft_config is not target_config
    assert draft_config.moe_backend == "CUTLASS"
    assert draft_config.moe_tactic == "cutlass"
    assert target_config.moe_backend == "TRTLLM"
    assert target_config.moe_tactic is None
    # The family is read from the MTP layer's format alone.
    pinned.assert_called_once_with(
        "cutlass", target_config.quant_config_dict[_MTP_EXPERTS], None, architecture=_ARCH
    )


def test_a_draft_backend_selects_the_mtp_layers():
    target_config = _target_config(spec_moe_backend="CUTLASS")
    with patch.object(ModelConfig, "resolve_moe_backend", return_value="CUTLASS") as resolve:
        draft_config = _draft_config_of(target_config)

    assert draft_config.moe_backend == "CUTLASS"
    assert draft_config.moe_tactic is None
    resolve.assert_called_once_with(
        "CUTLASS", _ARCH, quant_config=target_config.quant_config_dict[_MTP_EXPERTS]
    )


def test_the_mtp_layers_of_a_pinned_target_resolve_auto():
    """The target's tactic names implementations of the target's experts, not the MTP layers'."""
    target_config = _target_config(moe_tactic="trtllm.trtllm_gen.fused_moe.nvfp4")
    with patch.object(ModelConfig, "resolve_moe_backend", return_value="TRTLLM") as resolve:
        draft_config = _draft_config_of(target_config)

    assert draft_config.moe_backend == "TRTLLM"
    assert draft_config.moe_tactic is None
    assert target_config.moe_tactic == "trtllm.trtllm_gen.fused_moe.nvfp4"
    resolve.assert_called_once_with(
        "AUTO", _ARCH, quant_config=target_config.quant_config_dict[_MTP_EXPERTS]
    )


def test_replacement_heads_of_a_pinned_target_resolve_auto():
    """Replacement heads load through their own weight mapper, so they select for their weights."""
    target_config = _target_config(
        moe_tactic="trtllm.trtllm_gen.fused_moe.nvfp4", uses_replacement_heads=True
    )
    replacement_config = _target_config(moe_tactic=target_config.moe_tactic)
    with patch.object(ModelConfig, "resolve_moe_backend", return_value="CUTLASS") as resolve:
        draft_config = _draft_config_of(target_config, replacement_config)

    assert draft_config is not replacement_config
    assert draft_config.moe_backend == "CUTLASS"
    assert draft_config.moe_tactic is None
    assert replacement_config.moe_tactic == "trtllm.trtllm_gen.fused_moe.nvfp4"
    resolve.assert_called_once_with(
        "AUTO", _ARCH, quant_config=replacement_config.quant_config_dict[_MTP_EXPERTS]
    )


@pytest.mark.parametrize(
    "moe_tactic, spec_moe_backend",
    [("trtllm.cutedsl.mega_moe.nvfp4", None), (None, "AUTO")],
    ids=["pinned_target", "explicit_auto"],
)
def test_the_mtp_layers_resolve_auto_under_per_layer_dynamic_eplb(moe_tactic, spec_moe_backend):
    """The MTP layers register into the target's balancer, which only MegaMoE CuteDSL serves."""
    target_config = _target_config(
        moe_backend="MEGAMOE_CUTEDSL", moe_tactic=moe_tactic, spec_moe_backend=spec_moe_backend
    )
    target_config.moe_load_balancer = MoePerLayerDynamicLoadBalancerConfig(num_slots=72)

    draft_config = _draft_config_of(target_config)

    assert draft_config.moe_backend == "MEGAMOE_CUTEDSL"
    assert draft_config.moe_tactic is None
