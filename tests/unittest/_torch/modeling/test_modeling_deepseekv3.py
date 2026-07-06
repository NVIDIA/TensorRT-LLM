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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_deepseekv3


class _RecordingGatedMLP(nn.Module):
    def __init__(self, *, small_m_fc1_pad_rows: int, **_: object) -> None:
        super().__init__()
        self.small_m_fc1_pad_rows = small_m_fc1_pad_rows


@pytest.mark.parametrize(
    ("value", "layer_idx", "expected"), [(None, 0, 0), ("0", 0, 0), ("16", 0, 16), ("16", 1, 0)]
)
def test_layer0_canonical_small_m_fc1_rows_environment(
    monkeypatch: pytest.MonkeyPatch, value: str | None, layer_idx: int, expected: int
) -> None:
    name = "TLLM_DSV3_LAYER0_CANONICAL_SMALL_M_FC1_ROWS"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)

    assert modeling_deepseekv3._get_layer0_canonical_small_m_fc1_rows(layer_idx) == expected


@pytest.mark.parametrize("value", ["true", "8", "17"])
def test_layer0_canonical_small_m_fc1_rows_rejects_invalid_environment(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("TLLM_DSV3_LAYER0_CANONICAL_SMALL_M_FC1_ROWS", value)

    with pytest.raises(
        ValueError, match="TLLM_DSV3_LAYER0_CANONICAL_SMALL_M_FC1_ROWS must be 0 or 16"
    ):
        modeling_deepseekv3._get_layer0_canonical_small_m_fc1_rows(0)


@pytest.mark.parametrize(("layer_idx", "expected"), [(0, 16), (1, 0)])
def test_decoder_layer_threads_layer0_canonical_small_m_fc1_rows(
    monkeypatch: pytest.MonkeyPatch, layer_idx: int, expected: int
) -> None:
    monkeypatch.setenv("TLLM_DSV3_LAYER0_CANONICAL_SMALL_M_FC1_ROWS", "16")
    monkeypatch.setattr(modeling_deepseekv3, "GatedMLP", _RecordingGatedMLP)
    monkeypatch.setattr(
        modeling_deepseekv3,
        "DeepseekV3Attention",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        modeling_deepseekv3,
        "RMSNorm",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(modeling_deepseekv3, "can_access_peer", lambda _mapping: False)

    quant_config = SimpleNamespace(
        group_size=None,
        layer_quant_mode=SimpleNamespace(has_nvfp4=lambda: False),
        quant_algo=None,
        is_module_excluded_from_quantization=lambda _name: False,
    )
    config = SimpleNamespace(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=64,
        n_routed_experts=None,
        n_shared_experts=0,
        num_experts_per_tok=0,
        model_type="deepseek_v3",
        torch_dtype=torch.bfloat16,
        rms_norm_eps=1e-5,
    )
    mapping = SimpleNamespace(
        enable_attention_dp=False,
        tp_size=1,
        gpus_per_node=1,
        has_tp=lambda: False,
    )
    model_config = SimpleNamespace(
        pretrained_config=config,
        mapping=mapping,
        quant_config=quant_config,
        allreduce_strategy=None,
        use_cute_dsl_blockscaling_mm=False,
    )
    aux_stream_dict = {modeling_deepseekv3.AuxStreamType.Attention: None}

    layer = modeling_deepseekv3.DeepseekV3DecoderLayer(model_config, layer_idx, aux_stream_dict)

    assert isinstance(layer.mlp, _RecordingGatedMLP)
    assert layer.mlp.small_m_fc1_pad_rows == expected
