# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.modeling_utils import (
    _get_load_weights_num_workers,
    _pageout_safetensors_after_moe_load,
    rename_weights_with_regex,
)


def test_filter_prefix_matches_only_subtree_and_strips_prefix():
    weights = ConsumableWeightsDict(
        {
            "model.layers.1.weight": 1,
            "model.layers.1.bias": 2,
            "model.layers.10.weight": 3,
        }
    )

    assert weights.filter_prefix("model.layers.1") == {
        "weight": 1,
        "bias": 2,
    }


def test_prefix_index_tracks_new_and_consumed_keys():
    weights = ConsumableWeightsDict({"a.weight": 1})

    assert weights.filter_prefix("a") == {"weight": 1}
    weights["a.bias"] = 2
    weights.update({"a.scale": 3, "b.weight": 4})
    assert weights.filter_prefix("a") == {
        "weight": 1,
        "bias": 2,
        "scale": 3,
    }
    assert weights.mark_consumed("a") == 3
    assert weights.filter_prefix("a") == {}
    assert weights.filter_prefix("b") == {"weight": 4}


def test_prefix_index_handles_replacement_deletion_and_clear():
    weights = ConsumableWeightsDict({"a.weight": 1, "b.weight": 2})

    assert weights.filter_prefix("a") == {"weight": 1}
    weights["a.weight"] = 3
    del weights["a.weight"]
    assert weights.filter_prefix("a") == {}
    assert weights.filter_prefix("") == {"b.weight": 2}
    weights.clear()
    assert weights.filter_prefix("") == {}
    assert weights.mark_consumed("b") == 0


def test_mark_consumed_keys_accepts_missing_and_duplicate_keys():
    weights = ConsumableWeightsDict({"a.weight": 1, "a.bias": 2})

    assert weights.mark_consumed_keys(["a.weight", "missing", "a.weight"]) == 1
    assert weights.filter_prefix("a") == {"bias": 2}


def test_regex_rename_transfers_consumable_weight_ownership():
    tensor = object()
    weights = ConsumableWeightsDict(
        {
            "model.language_model.layer.weight": tensor,
            "model.visual.weight": object(),
        }
    )

    renamed = rename_weights_with_regex({r"^model\.language_model\.(.*)$": r"model.\1"}, weights)

    assert isinstance(renamed, ConsumableWeightsDict)
    assert len(weights) == 0
    assert renamed["model.layer.weight"] is tensor
    assert "model.visual.weight" in renamed


def test_pageout_safetensors_after_moe_load():
    with (
        mock.patch("torch.cuda.synchronize") as synchronize,
        mock.patch(
            "tensorrt_llm._torch.mmap_utils.pageout_file_backed_regions"
        ) as pageout_file_backed_regions,
    ):
        _pageout_safetensors_after_moe_load()

    synchronize.assert_called_once_with()
    pageout_file_backed_regions.assert_called_once_with(".safetensors", mode="dontneed")


@pytest.mark.parametrize(
    "value, expected", [(None, None), ("", None), ("  ", None), ("1", 1), ("4", 4)]
)
def test_get_load_weights_num_workers(monkeypatch, value, expected):
    env_name = "TRT_LLM_LOAD_WEIGHTS_NUM_WORKERS"
    if value is None:
        monkeypatch.delenv(env_name, raising=False)
    else:
        monkeypatch.setenv(env_name, value)

    assert _get_load_weights_num_workers() == expected


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "workers"])
def test_get_load_weights_num_workers_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("TRT_LLM_LOAD_WEIGHTS_NUM_WORKERS", value)

    with pytest.raises(ValueError, match="must be a positive integer"):
        _get_load_weights_num_workers()
