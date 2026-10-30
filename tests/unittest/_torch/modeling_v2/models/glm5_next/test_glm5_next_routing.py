# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash target routing, driven by synthetic configs on CPU."""

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig, NGramDecodingConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_SM100 = (10, 0)
_DUMMY_MODEL = "/tmp/dummy_model"
_TARGET = (
    "models.glm5_next.glm_5_3_flash_nvfp4__sm_100__default.modeling."
    "ModelingV2Glm53FlashNvfp4Sm100Default"
)
_TEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1)


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm100(monkeypatch):
    """Route as if this were a B200, wherever the test actually runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM100)


def _llm_args(mode: str) -> TorchLlmArgs:
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode)


def _flash_config(**text_overrides) -> PretrainedConfig:
    """The composite config GLM-5.3-Flash's config.json declares, its text_config cut to the routed fields."""
    text_config = dict(
        model_type="glm5_next_text",
        num_hidden_layers=45,
        hidden_size=4096,
        n_routed_experts=288,
        q_lora_rank=1536,
        linear_num_heads=64,
    )
    text_config.update(text_overrides)
    return PretrainedConfig(
        architectures=["Glm5NextForConditionalGeneration"],
        model_type="glm5_next",
        text_config=text_config,
    )


def _model_config(
    pretrained_config, quant_algo=QuantAlgo.NVFP4, spec_config=None, **mapping_kwargs
):
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(**(mapping_kwargs or _TEP4)),
        spec_config=spec_config,
        quant_config=QuantConfig(quant_algo=quant_algo),
    )


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize(
    "spec_config, mapping_kwargs",
    [
        (None, _TEP4),
        (MTPDecodingConfig(max_draft_len=3), _TEP4),
        (MTPDecodingConfig(max_draft_len=1), dict(world_size=8, tp_size=8, moe_tp_size=8)),
        (None, dict(world_size=4, tp_size=4, moe_ep_size=4, enable_attention_dp=True)),
    ],
    ids=["tep4", "tep4-mtp3", "tp8-mtp1", "dep4"],
)
def test_flash_nvfp4_matches(mode, spec_config, mapping_kwargs):
    """The default target takes every layout the built-in model computes, with or without MTP."""
    config = _model_config(_flash_config(), spec_config=spec_config, **mapping_kwargs)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _TARGET


@pytest.mark.parametrize(
    "text_overrides, quant_algo, spec_config, mapping_kwargs, missed",
    [
        (dict(num_hidden_layers=44), QuantAlgo.NVFP4, None, _TEP4, r"shape \(44, 4096"),
        (dict(), QuantAlgo.FP8_BLOCK_SCALES, None, _TEP4, "quantization FP8_BLOCK_SCALES"),
        (
            dict(),
            QuantAlgo.NVFP4,
            NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2),
            _TEP4,
            "NGRAM",
        ),
        (dict(), QuantAlgo.NVFP4, None, dict(world_size=8, tp_size=4, pp_size=2), "pp=2"),
        (
            dict(),
            QuantAlgo.NVFP4,
            MTPDecodingConfig(max_draft_len=1, use_mtp_vanilla=True),
            _TEP4,
            "vanilla MTP",
        ),
        (
            dict(),
            QuantAlgo.NVFP4,
            MTPDecodingConfig(max_draft_len=3, use_rejection_sampling=True),
            _TEP4,
            "rejection sampling",
        ),
        (
            dict(),
            QuantAlgo.NVFP4,
            MTPDecodingConfig(max_draft_len=3, use_relaxed_acceptance_for_thinking=True),
            _TEP4,
            "relaxed acceptance",
        ),
    ],
    ids=[
        "other-shape",
        "fp8-checkpoint",
        "ngram",
        "pipeline-parallel",
        "vanilla-mtp",
        "rejection-sampling",
        "relaxed-acceptance",
    ],
)
def test_flash_near_misses_do_not_match(
    text_overrides, quant_algo, spec_config, mapping_kwargs, missed
):
    config = _model_config(
        _flash_config(**text_overrides),
        quant_algo=quant_algo,
        spec_config=spec_config,
        **mapping_kwargs,
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_flash_on_sm103_does_not_match(monkeypatch):
    """The target has been run on B200 only; a GB300 falls back."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    config = _model_config(_flash_config())
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=r"compute capability \(10, 3\)"):
        modeling_v2_resolve(config, _llm_args("require"))
