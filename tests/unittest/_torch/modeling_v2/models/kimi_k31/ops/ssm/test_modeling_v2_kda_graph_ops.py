# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KDA layer registry and the breakable-CUDA-graph core break (``ops/ssm/kda_graph_ops``). CPU only."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm import kda_graph_ops

pytestmark = pytest.mark.cpu_only


class _Layer:
    """Writes 1 into the core rows of the step (``metadata.num_tokens``), as the KDA layer fills them."""

    def _forward_impl(self, hidden_states, metadata, output):
        output[: metadata.num_tokens].fill_(1.0)
        return output


@pytest.fixture
def extra_attrs(monkeypatch):
    """The model extra attributes the break reads, with a plain metadata type."""
    attrs = {}
    monkeypatch.setattr(kda_graph_ops, "get_model_extra_attrs", lambda: attrs)
    monkeypatch.setattr(kda_graph_ops, "AttentionMetadata", SimpleNamespace)
    return attrs


def test_register_suffixes_repeated_layer_indices() -> None:
    attrs = {}
    layers = [_Layer(), _Layer(), _Layer(), _Layer()]
    keys = [
        kda_graph_ops.register_kda_layer(attrs, layer, idx)
        for layer, idx in zip(layers, (3, 3, 3, 4))
    ]
    assert keys == ["3", "3_0", "3_1", "4"]
    registry = attrs[kda_graph_ops.KDA_LAYERS]
    assert [registry[key]() for key in keys] == layers


def test_core_break_zeroes_the_padding_rows(extra_attrs) -> None:
    layer = _Layer()
    key = kda_graph_ops.register_kda_layer(extra_attrs, layer, 0)
    metadata = SimpleNamespace(num_tokens=5)
    extra_attrs["attention_metadata"] = lambda: metadata
    # Graph-pool memory another capture may have left non-finite.
    output = torch.full((8, 2, 4), float("nan"))
    kda_graph_ops.kda_core_inplace(torch.zeros(8, 16), key, output)
    assert torch.equal(output[:5], torch.ones(5, 2, 4))
    assert torch.equal(output[5:], torch.zeros(3, 2, 4))
