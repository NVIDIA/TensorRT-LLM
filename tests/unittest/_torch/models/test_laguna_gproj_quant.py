# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""g_proj must not be built quantised when the checkpoint excludes it.

Only tp_size > 1 takes the sharding path, so this presents as a sharding
failure rather than a quantisation one. The existing Laguna accuracy tests are
all TP=1 and would not catch a regression.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.models.modeling_laguna import g_proj_quant_config
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

LAYER_IDX = 3
G_PROJ = f"model.layers.{LAYER_IDX}.self_attn.g_proj"


def _model_config(exclude_modules):
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
        kv_cache_quant_algo=QuantAlgo.FP8,
        exclude_modules=exclude_modules,
    )
    return SimpleNamespace(get_quant_config=lambda name=None: quant_config)


# Exact name, ancestor walk, ".*" subtree, glob and "re:" regex. The fleet uses
# literal names for g_proj but regexes for the fused-MoE exclusions.
@pytest.mark.parametrize(
    "pattern",
    [
        G_PROJ,
        f"model.layers.{LAYER_IDX}.self_attn",
        f"model.layers.{LAYER_IDX}",
        f"model.layers.{LAYER_IDX}.*",
        "*g_proj*",
        r"re:.*\.self_attn\.g_proj$",
    ],
)
def test_excluded_g_proj_is_not_quantised(pattern):
    resolved = g_proj_quant_config(_model_config([pattern, "lm_head"]), LAYER_IDX)

    assert resolved.quant_algo is None, f"{pattern!r} did not exclude g_proj"
    # KV cache mode is derived from this, so it must survive the override.
    assert resolved.kv_cache_quant_algo == QuantAlgo.FP8


def test_unexcluded_g_proj_keeps_the_model_quant_config():
    """Control: a checkpoint that does quantise g_proj is left alone."""
    model_config = _model_config(["lm_head", "model.layers.9.self_attn.g_proj"])

    resolved = g_proj_quant_config(model_config, LAYER_IDX)

    assert resolved is model_config.get_quant_config()
    assert resolved.quant_algo == QuantAlgo.FP8_BLOCK_SCALES


def test_unquantised_model_is_passed_through():
    model_config = SimpleNamespace(get_quant_config=lambda name=None: None)

    assert g_proj_quant_config(model_config, LAYER_IDX) is None


def test_missing_layer_idx_does_not_match_a_literal_none():
    """layer_idx is Optional; without the guard the name would render as
    "model.layers.None.self_attn.g_proj" and silently match nothing."""
    model_config = _model_config(["model.layers.None.self_attn.g_proj"])

    resolved = g_proj_quant_config(model_config, None)

    assert resolved.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
