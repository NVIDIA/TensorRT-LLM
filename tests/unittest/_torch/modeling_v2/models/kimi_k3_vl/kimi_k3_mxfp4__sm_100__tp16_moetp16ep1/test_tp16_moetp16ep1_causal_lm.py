# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp16_moetp16ep1 target class as a causal LM (``modeling.ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1``), host-side:
its real ``__init__`` and ``forward`` on a tiny text model (its construction checks, text config and text model
stubbed), meta-initialized parameters.

* The class inherits ``DecoderModelForCausalLM`` directly (README "Model lifecycle"), with the KDA Mamba metadata
  class and the one-engine fast paths off.
* It builds no drafter and no worker; its logits processor is this target's ``K3LogitsProcessor``; its forward runs
  the text model, drops the padded rows and calls the logits processor.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3Config

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1 import (  # noqa: E501
    modeling as target_modeling,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM, MetaInitMode
from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata

pytestmark = pytest.mark.cpu_only

HIDDEN, VOCAB = 64, 128
TARGET = target_modeling.ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1


class _TextModel(nn.Module):
    """A text model that records its inputs and returns 6 rows."""

    def __init__(self, rows=6):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.rows = rows
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return torch.arange(self.rows * HIDDEN, dtype=torch.float32).view(self.rows, HIDDEN)


class _Logits(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden_states, lm_head, attn_metadata, return_context_logits):
        self.calls.append((hidden_states, lm_head, return_context_logits))
        return "logits"


@pytest.fixture
def model(monkeypatch):
    """The target built by its own ``__init__`` around the tiny text model, its step contract already checked."""
    monkeypatch.setattr(target_modeling, "_check_construction", lambda model_config: None)
    monkeypatch.setattr(target_modeling, "_text_model_config", lambda model_config: model_config)
    monkeypatch.setattr(target_modeling, "KimiLinearModel", lambda text: _TextModel())
    cfg = Qwen3Config.from_dict(
        dict(
            architectures=["KimiLinearForCausalLM"],
            model_type="qwen3",
            hidden_size=HIDDEN,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=VOCAB,
            torch_dtype="bfloat16",
            tie_word_embeddings=False,
        )
    )
    cfg.text_config = cfg
    with MetaInitMode():
        built = TARGET(ModelConfig(pretrained_config=cfg))
    built._step_checked = True
    return built


def test_target_inherits_the_decoder_lm_interface_directly():
    assert TARGET.__bases__ == (DecoderModelForCausalLM,)
    assert (
        TARGET.mamba_metadata_cls is KimiK3MambaMetadata and TARGET.spec_decode_fast_paths is False
    )


def test_forward_without_speculation(model):
    assert (model.spec_config, model.draft_config, model.draft_model, model.spec_worker) == (
        None,
    ) * 4
    assert model.use_separate_draft_kv_cache is False and model.layer_idx == -1
    assert model.epilogue == [model.lm_head]
    assert isinstance(model.logits_processor, target_modeling._decode_gemv.K3LogitsProcessor)
    model.logits_processor = _Logits()
    attn = SimpleNamespace(padded_num_tokens=8, num_tokens=4)
    assert (
        model.forward(attn, input_ids="ids", position_ids="pos", return_context_logits=True)
        == "logits"
    )
    (kwargs,) = model.model.calls
    assert (
        kwargs["input_ids"] == "ids"
        and kwargs["position_ids"] == "pos"
        and kwargs["spec_metadata"] is None
    )
    ((hidden, head, ctx),) = model.logits_processor.calls
    assert hidden.shape[0] == 4 and head is model.lm_head and ctx is True
