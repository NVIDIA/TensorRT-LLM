# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 target routing, driven by synthetic configs on CPU."""

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig, NGramDecodingConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_SM103 = (10, 3)
_DUMMY_MODEL = "/tmp/dummy_model"
_NGRAM = NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2)


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm103(monkeypatch):
    """Route as if this were a GB300, wherever the test actually runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM103)


def _llm_args(mode="auto", **kwargs) -> TorchLlmArgs:
    """The deployment, as the LLM API would have been given it.

    The switch is an LLM API argument, so the tests set one too -- that is the
    surface under test.
    """
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode, **kwargs)


def _model_config(pretrained_config, spec_config=None, **mapping_kwargs):
    mapping = Mapping(**mapping_kwargs) if mapping_kwargs else Mapping()
    return ModelConfig(
        pretrained_config=pretrained_config, mapping=mapping, spec_config=spec_config
    )


_GLM52_TARGET = (
    "models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.modeling.ModelingV2Glm52Nvfp4Sm107Tp16"
)
_TP16 = dict(world_size=16, tp_size=16, moe_ep_size=1, moe_tp_size=16)


@pytest.fixture
def _on_sm107(monkeypatch):
    """Route as if this were a Rubin GPU."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 7))


def _glm52_config(**overrides):
    """The shape GLM-5.2-NVFP4's own config.json declares."""
    fields = dict(
        architectures=["GlmMoeDsaForCausalLM"],
        model_type="glm_moe_dsa",
        num_hidden_layers=78,
        hidden_size=6144,
        n_routed_experts=256,
        q_lora_rank=2048,
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize(
    "spec_config",
    [None, MTPDecodingConfig(max_draft_len=7), MTPDecodingConfig(max_draft_len=3)],
    ids=["no-spec", "mtp7", "mtp3"],
)
def test_glm52_tp16_matches(_on_sm107, spec_config, mode):
    """Any draft length: the draft loop replays the one MTP module per step."""
    config = _model_config(_glm52_config(), spec_config=spec_config, **_TP16)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _GLM52_TARGET


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, spec_config, missed",
    [
        (dict(num_hidden_layers=92), _TP16, None, r"shape \(92, 6144, 256, 2048\)"),
        (dict(), dict(_TP16, enable_attention_dp=True), None, "attention_dp=True"),
        (dict(), dict(_TP16, moe_ep_size=16, moe_tp_size=1), None, "moe_ep=16"),
        (dict(), dict(world_size=8, tp_size=8, moe_tp_size=8), None, "world_size=8"),
        (dict(), _TP16, MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True), "MTP"),
        (dict(), _TP16, _NGRAM, "NGRAM"),
    ],
    ids=["other-shape", "attention-dp", "expert-parallel", "tp8", "vanilla-mtp", "ngram"],
)
def test_glm52_near_misses_do_not_match(
    _on_sm107, config_kwargs, mapping_kwargs, spec_config, missed
):
    config = _model_config(
        _glm52_config(**config_kwargs), spec_config=spec_config, **mapping_kwargs
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_glm52_on_sm103_does_not_match():
    """The target's kernels are built for sm107; a GB300 falls back."""
    config = _model_config(_glm52_config(), **_TP16)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=r"compute capability \(10, 3\)"):
        modeling_v2_resolve(config, _llm_args("require"))
