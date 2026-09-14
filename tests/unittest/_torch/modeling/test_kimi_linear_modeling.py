# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_kimi_linear
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = pytest.mark.cpu_only


def test_kimi_linear_model_builds_shared_aux_stream_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streams = [object() for _ in range(4)]
    stream_iter = iter(streams)
    layer_aux_stream_dicts = []

    class _FakeDecoderLayer(nn.Module):
        def __init__(self, _model_config, _config, _layer_idx, aux_stream_dict) -> None:
            super().__init__()
            layer_aux_stream_dicts.append(aux_stream_dict)

    monkeypatch.setattr(torch.cuda, "Stream", lambda: next(stream_iter))
    monkeypatch.setattr(modeling_kimi_linear, "KimiLinearDecoderLayer", _FakeDecoderLayer)

    config = KimiLinearConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        rms_norm_eps=1e-5,
        attn_res_block_size=1,
        linear_attn_config={"kda_layers": [1, 3], "full_attn_layers": [2]},
    )
    model = modeling_kimi_linear.KimiLinearModel(ModelConfig(pretrained_config=config))

    aux_stream_dict = model.aux_stream_dict
    assert set(aux_stream_dict) == {
        AuxStreamType.Attention,
        AuxStreamType.MoeShared,
        AuxStreamType.MoeChunkingOverlap,
        AuxStreamType.MoeBalancer,
        AuxStreamType.MoeOutputMemset,
    }
    assert aux_stream_dict[AuxStreamType.Attention] is streams[0]
    assert aux_stream_dict[AuxStreamType.MoeShared] is streams[0]
    assert aux_stream_dict[AuxStreamType.MoeChunkingOverlap] is streams[1]
    assert aux_stream_dict[AuxStreamType.MoeBalancer] is streams[2]
    assert aux_stream_dict[AuxStreamType.MoeOutputMemset] is streams[3]
    assert len(layer_aux_stream_dicts) == config.num_hidden_layers
    assert all(stream_dict is aux_stream_dict for stream_dict in layer_aux_stream_dicts)


def test_kimi_k3_routed_expert_exclusion_outranks_the_checkpoint_entry() -> None:
    """An excluded layer drops the checkpoint's per-layer format, and only it.

    Both layers carry the same per-layer NVFP4 entry, so the two resolutions
    differ only by the exclusion. Without the exclusion winning, layer 3 keeps
    NVFP4 and loads quantized weights the checkpoint left in bf16 -- wrong
    numerics rather than a failure. ``kv_cache_quant_algo`` is not part of what
    an expert exclusion turns off, so it has to survive.
    """
    per_layer = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    model_config = ModelConfig(
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.NVFP4,
            kv_cache_quant_algo=QuantAlgo.FP8,
            exclude_modules=["model.layers.3.mlp.experts"],
        ),
        quant_config_dict={
            "model.layers.3.mlp.experts": per_layer,
            "model.layers.4.mlp.experts": per_layer,
        },
    )
    resolve = modeling_kimi_linear.KimiK3MoERuntime._resolve_routed_quant_config

    excluded = resolve(model_config, 3)
    assert excluded.quant_algo is None
    assert excluded.kv_cache_quant_algo == QuantAlgo.FP8

    assert resolve(model_config, 4) is per_layer
