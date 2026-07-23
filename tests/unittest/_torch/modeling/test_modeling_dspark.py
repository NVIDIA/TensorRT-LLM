# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for DSpark bucket-aware checkpoint loading."""

from unittest.mock import MagicMock

import pytest
from torch import nn

from tensorrt_llm._torch.models import modeling_dspark
from tensorrt_llm._torch.models.modeling_dspark import (
    DSparkDraftModel,
    DSparkForCausalLM,
    remap_dspark_draft_keys,
)


def _make_dspark_loader(num_stages: int = 2) -> DSparkForCausalLM:
    loader = DSparkForCausalLM.__new__(DSparkForCausalLM)
    nn.Module.__init__(loader)
    loader.num_stages = num_stages
    loader.dspark_model = MagicMock()
    loader.dspark_model._loaded_weight_stages = set()
    return loader


def test_dspark_layerwise_loading_defers_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_dspark_loader()
    weight_loader = MagicMock()
    monkeypatch.setattr(
        modeling_dspark,
        "DeepseekV4WeightLoader",
        MagicMock(return_value=weight_loader),
    )
    weights = {"mtp.0.main_proj.weight": MagicMock()}

    loader.load_weights(weights, initial_bucket_loading=True)

    weight_loader.load_weights.assert_called_once_with(
        {"mtp_layers.0.main_proj.weight": weights["mtp.0.main_proj.weight"]},
        initial_bucket_loading=True,
    )
    loader.dspark_model.post_load_weights.assert_not_called()
    loader.dspark_model.cache_attn_weights_from_state_dict.assert_called_once_with(weights)
    assert loader.dspark_model._loaded_weight_stages == {0}


def test_dspark_layerwise_loading_rejects_out_of_range_stage() -> None:
    loader = _make_dspark_loader()

    with pytest.raises(ValueError, match="out-of-range draft stages"):
        loader.load_weights(
            {"mtp.2.main_proj.weight": MagicMock()},
            initial_bucket_loading=True,
        )


def test_dspark_post_load_rejects_missing_stages() -> None:
    model = DSparkDraftModel.__new__(DSparkDraftModel)
    nn.Module.__init__(model)
    model.mtp_layers = nn.ModuleList([nn.Identity(), nn.Identity()])
    model._loaded_weight_stages = {0}
    model._cached_attn_stages = {0}

    with pytest.raises(RuntimeError, match="weight loading is incomplete"):
        model.post_load_weights()


def test_remap_dspark_draft_keys_ignores_target_weights() -> None:
    target_weight = MagicMock()

    assert remap_dspark_draft_keys({"layers.0.weight": target_weight}, num_stages=1) == {}
