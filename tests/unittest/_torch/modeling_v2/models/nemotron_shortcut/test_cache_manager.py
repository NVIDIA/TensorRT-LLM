# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.configs.nemotron_shortcut import NemotronShortcutTextConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.nemotron_shortcut.cache_manager import NemotronShortcutCacheManager
from tensorrt_llm._torch.pyexecutor.config_utils import extract_mamba_kv_cache_params
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


def _config(layer_types=None):
    return NemotronShortcutTextConfig(
        layer_types=layer_types
        or ["linear_attention", "mlp", "shortcut_moe_attn", "shortcut_moe_gdp"],
        architectures=[NemotronShortcutTextConfig.causal_lm_architecture],
        dtype=torch.bfloat16,
        head_dim=64,
        index_topk=32,
        index_head_dim=32,
        gdp_num_heads=8,
        gdp_num_groups=8,
        gdp_num_householder=5,
        gdp_head_dim=16,
        gdp_key_head_dim=32,
        gdp_conv_kernel=3,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "tp, attention_dp, local_heads, kv_algo, kv_bytes",
    [(4, False, 2, None, 256), (2, True, 8, None, 256), (4, False, 2, QuantAlgo.NVFP4, 72)],
)
def test_sparse_kv_replication_and_gdp_sharding(tp, attention_dp, local_heads, kv_algo, kv_bytes):
    mapping = Mapping(world_size=tp, tp_size=tp, enable_attention_dp=attention_dp)
    model = ModelConfig(
        pretrained_config=_config(),
        mapping=mapping,
        quant_config=QuantConfig(kv_cache_quant_algo=kv_algo),
    )
    params = extract_mamba_kv_cache_params(model.pretrained_config)
    assert model.get_num_attention_layers() == 1
    assert model.get_num_mamba_layers() == 2
    # Five (V, K) updates and a final query keep two BF16 convolution taps.
    conv_bytes = local_heads * (5 * (16 + 32) + 32) * 2 * torch.bfloat16.itemsize
    temporal_bytes = local_heads * 16 * 32 * torch.float32.itemsize
    assert params.get_states_bytes_per_layer(mapping) == conv_bytes + temporal_bytes
    assert NemotronShortcutCacheManager._estimate_attention_cache_cost(model, mapping) == (
        kv_bytes + 32 * torch.bfloat16.itemsize,
        0,
        0,
    )
