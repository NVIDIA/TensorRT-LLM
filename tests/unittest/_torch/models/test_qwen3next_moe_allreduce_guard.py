# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the MoEAllReduce construction guard in Qwen3Next layers.

MoEAllReduce.__init__ eagerly opens a CUDA-IPC workspace across the TP group
(get_allreduce_workspace -> IpcMemory.open_ipc_memory). Under attention DP that
peer topology does not exist, so merely CONSTRUCTING it faults with a sticky
cudaErrorIllegalAddress that poisons the CUDA context and surfaces at an
unrelated later CUDA call. Both Qwen3Next decoder-layer variants therefore gate
the construction on ``fusion_config.POST_MOE_FUSION and tp_size > 1`` — the
only condition under which the instance is ever used. These tests pin that
guard so a refactor cannot silently reintroduce the destructive allocation on
the Ray / attention-DP path.

The heavyweight neighbours (attention, MLP, norms, AllReduce) are mocked; the
layer __init__ itself runs for real. No GPU or multi-rank job is needed.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import tensorrt_llm._torch.models.modeling_qwen3_next as m


class _FakeMapping:
    """Just enough of Mapping for the decoder-layer fusion/guard logic."""

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
    )
    return SimpleNamespace(
        pretrained_config=pretrained,
        mapping=mapping,
        allreduce_strategy=None,
    )


@pytest.fixture(params=["linear", "full_attention"])
def layer_builder(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Build either decoder-layer variant with mocked heavy submodules.

    Returns (build, moe_allreduce_mock): ``build(mapping)`` runs the real
    layer __init__ against the fake mapping; the mock records MoEAllReduce
    construction attempts.
    """
    moe_allreduce = mock.MagicMock(name="MoEAllReduce")
    # Neighbours whose real __init__ needs CUDA, weights, or a distributed
    # runtime. The guard under test never depends on them.
    for name in ("RMSNorm", "AllReduce"):
        monkeypatch.setattr(m, name, mock.MagicMock(name=name))
    monkeypatch.setattr(m, "_create_mlp", mock.MagicMock(name="_create_mlp"))
    monkeypatch.setattr(m, "MoEAllReduce", moe_allreduce)

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

    return build, moe_allreduce


def test_constructed_once_when_post_moe_fusion_and_tp(
    layer_builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    build, moe_allreduce = layer_builder
    monkeypatch.delenv("TRTLLM_QWEN3_EAGER_FUSION_DISABLED", raising=False)
    mapping = _FakeMapping(tp_size=4)

    layer = build(mapping)

    assert layer.fusion_config.POST_MOE_FUSION
    moe_allreduce.assert_called_once_with(mapping=mapping)
    assert layer.moe_allreduce is moe_allreduce.return_value


def test_not_constructed_under_attention_dp(layer_builder, monkeypatch: pytest.MonkeyPatch) -> None:
    """The original failure: attention DP has no TP-group CUDA-IPC peers, so
    construction itself is destructive and must not happen."""
    build, moe_allreduce = layer_builder
    monkeypatch.delenv("TRTLLM_QWEN3_EAGER_FUSION_DISABLED", raising=False)

    layer = build(_FakeMapping(tp_size=4, enable_attention_dp=True))

    moe_allreduce.assert_not_called()
    assert layer.moe_allreduce is None


def test_not_constructed_when_eager_fusion_disabled(
    layer_builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    build, moe_allreduce = layer_builder
    monkeypatch.setenv("TRTLLM_QWEN3_EAGER_FUSION_DISABLED", "1")

    layer = build(_FakeMapping(tp_size=4))

    assert not layer.fusion_config.POST_MOE_FUSION
    moe_allreduce.assert_not_called()
    assert layer.moe_allreduce is None


def test_not_constructed_without_tp(layer_builder, monkeypatch: pytest.MonkeyPatch) -> None:
    build, moe_allreduce = layer_builder
    monkeypatch.delenv("TRTLLM_QWEN3_EAGER_FUSION_DISABLED", raising=False)

    layer = build(_FakeMapping(tp_size=1))

    moe_allreduce.assert_not_called()
    assert layer.moe_allreduce is None


def test_not_constructed_under_pipeline_parallel(
    layer_builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PP disables POST_MOE_FUSION, so the workspace must not be allocated."""
    build, moe_allreduce = layer_builder
    monkeypatch.delenv("TRTLLM_QWEN3_EAGER_FUSION_DISABLED", raising=False)

    layer = build(_FakeMapping(tp_size=4, pp_size=2))

    assert not layer.fusion_config.POST_MOE_FUSION
    moe_allreduce.assert_not_called()
    assert layer.moe_allreduce is None
