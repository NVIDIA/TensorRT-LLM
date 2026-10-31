# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fusion flags and all-reduce construction of the Qwen3Next decoder layers.

The post-MoE all-reduce (finalized or with the experts' finalize folded in) runs on the layer's one
``AllReduce``; no second all-reduce workspace is opened, so attention DP, where the TP-group peers do not
exist, constructs nothing destructive. The heavyweight neighbours are mocked; the layer __init__ runs for
real. No GPU or multi-rank job is needed.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import tensorrt_llm._torch.models.modeling_qwen3_next as m


class _FakeMapping:
    """Just enough of Mapping for the decoder-layer fusion logic."""

    def __init__(
        self, tp_size: int = 4, pp_size: int = 1, enable_attention_dp: bool = False
    ) -> None:
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.enable_attention_dp = enable_attention_dp

    def has_tp(self) -> bool:
        return self.tp_size > 1

    def has_pp(self) -> bool:
        return self.pp_size > 1


def _fake_model_config(mapping: _FakeMapping) -> SimpleNamespace:
    pretrained = SimpleNamespace(
        hidden_size=8,
        rms_norm_eps=1e-6,
        torch_dtype=torch.float32,
        model_type="qwen3_next",
    )
    return SimpleNamespace(
        pretrained_config=pretrained,
        mapping=mapping,
        allreduce_strategy=None,
        mnnvl_allreduce_fp8_payload=False,
    )


@pytest.fixture(params=["linear", "full_attention"])
def layer_builder(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """(build, all_reduce_mock): ``build(mapping)`` runs the real layer __init__ against the fake mapping."""
    all_reduce = mock.MagicMock(name="AllReduce")
    monkeypatch.setattr(m, "RMSNorm", mock.MagicMock(name="RMSNorm"))
    monkeypatch.setattr(m, "AllReduce", all_reduce)
    monkeypatch.setattr(m, "_create_mlp", mock.MagicMock(name="_create_mlp"))

    if request.param == "linear":
        monkeypatch.setattr(
            m, "Qwen3NextGatedDeltaNet", mock.MagicMock(name="Qwen3NextGatedDeltaNet")
        )
        layer_cls = m.Qwen3NextLinearDecoderLayer
    else:
        monkeypatch.setattr(m, "Qwen3NextAttention", mock.MagicMock(name="Qwen3NextAttention"))
        layer_cls = m.Qwen3NextFullAttentionDecoderLayer

    def build(mapping: _FakeMapping):
        return layer_cls(
            _fake_model_config(mapping),
            layer_idx=0,
            aux_stream=mock.MagicMock(name="aux_stream"),
        )

    return build, all_reduce


@pytest.mark.parametrize(
    "mapping_kwargs,disabled,pre,post",
    [
        ({"tp_size": 4}, False, True, True),
        ({"tp_size": 4, "enable_attention_dp": True}, False, False, False),
        ({"tp_size": 4}, True, False, False),
        ({"tp_size": 1}, False, False, False),
        ({"tp_size": 4, "pp_size": 2}, False, True, False),
    ],
    ids=["tp", "attention_dp", "fusion_disabled", "no_tp", "pp"],
)
def test_fusion_flags_and_single_allreduce(
    layer_builder, monkeypatch: pytest.MonkeyPatch, mapping_kwargs, disabled, pre, post
) -> None:
    build, all_reduce = layer_builder
    if disabled:
        monkeypatch.setenv("TRTLLM_QWEN3_EAGER_FUSION_DISABLED", "1")
    else:
        monkeypatch.delenv("TRTLLM_QWEN3_EAGER_FUSION_DISABLED", raising=False)
    mapping = _FakeMapping(**mapping_kwargs)

    layer = build(mapping)

    assert layer.fusion_config.PRE_MOE_FUSION == pre
    assert layer.fusion_config.POST_MOE_FUSION == post
    # The mixer's own all-reduce runs only when no fused one takes its place.
    assert layer.disable_attn_allreduce == (
        pre or mapping.tp_size == 1 or mapping.enable_attention_dp
    )
    all_reduce.assert_called_once_with(mapping=mapping, strategy=None, dtype=torch.float32)
