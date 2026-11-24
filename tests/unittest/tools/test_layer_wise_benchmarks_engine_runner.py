# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the layer-wise benchmark engine backend's helpers."""

import json

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import registered_routes
from tensorrt_llm.tools.layer_wise_benchmarks.engine_runner import (
    fill_random_,
    merge_llm_args,
    prefix_num_layers,
    select_backend,
)


def test_merge_llm_args_nested():
    base = {"max_batch_size": 8, "kv_cache_config": {"dtype": "fp8", "tokens_per_block": 32}}
    override = {"kv_cache_config": {"tokens_per_block": 64}, "modeling_v2": "require"}
    assert merge_llm_args(base, override) == {
        "max_batch_size": 8,
        "kv_cache_config": {"dtype": "fp8", "tokens_per_block": 64},
        "modeling_v2": "require",
    }
    assert base["kv_cache_config"]["tokens_per_block"] == 32


def test_prefix_num_layers():
    assert prefix_num_layers([0, 1, 2, 3]) == 4
    with pytest.raises(ValueError, match="prefix"):
        prefix_num_layers([4, 5, 6, 7])


@pytest.mark.parametrize(
    "routed, llm_args, expected",
    [(True, {}, "engine"), (True, {"modeling_v2": "off"}, "layers"), (False, {}, "layers")],
)
def test_select_backend_auto(tmp_path, routed, llm_args, expected):
    arch = next(iter(registered_routes())) if routed else "NotARoutedForCausalLM"
    (tmp_path / "config.json").write_text(json.dumps({"architectures": [arch]}))
    assert select_backend("auto", str(tmp_path), llm_args) == expected
    assert select_backend("layers", str(tmp_path), llm_args) == "layers"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fill_random_is_finite_in_every_float_format():
    region = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
    fill_random_([region], seed=1)
    first = region.clone()
    fill_random_([region], seed=1)
    assert torch.equal(region, first)
    for dtype in (torch.float8_e4m3fn, torch.bfloat16, torch.float16, torch.float32):
        values = region.view(dtype).float()
        assert torch.isfinite(values).all() and values.abs().max() < 2
