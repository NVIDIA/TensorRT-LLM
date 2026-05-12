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
from unittest.mock import patch

import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_nemotron_h import NemotronHMOE
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _make_nemotron_h_moe_config(quant_config: QuantConfig) -> ModelConfig:
    return ModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=16,
            intermediate_size=32,
            mlp_bias=False,
            moe_intermediate_size=64,
            moe_latent_size=None,
            n_group=1,
            n_routed_experts=4,
            n_shared_experts=0,
            num_experts_per_tok=1,
            routed_scaling_factor=1.0,
            topk_group=1,
            torch_dtype=torch.float16,
        ),
        moe_backend="CUTLASS",
        quant_config=quant_config,
    )


def test_nemotron_h_moe_uses_w4a4_nvfp4_expert_config_for_w4a16_checkpoint():
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16, exclude_modules=["lm_head"]
    )
    quant_config.mamba_ssm_cache_dtype = torch.float32
    model_config = _make_nemotron_h_moe_config(quant_config)
    captured = {}

    def fake_create_moe(**kwargs):
        captured["model_config"] = kwargs["model_config"]
        return nn.Identity()

    with patch(
        "tensorrt_llm._torch.models.modeling_nemotron_h.create_moe", side_effect=fake_create_moe
    ):
        with patch("torch.cuda.Event", side_effect=lambda: object()):
            aux_stream_dict = {AuxStreamType.MoeShared: None}
            NemotronHMOE(model_config=model_config, layer_idx=1, aux_stream_dict=aux_stream_dict)

    moe_quant_config = captured["model_config"].quant_config
    assert moe_quant_config is not quant_config
    assert moe_quant_config.quant_algo == QuantAlgo.NVFP4
    assert moe_quant_config.group_size == 16
    assert moe_quant_config.exclude_modules == ["lm_head"]
    assert moe_quant_config.mamba_ssm_cache_dtype == torch.float32
    assert model_config.quant_config.quant_algo == QuantAlgo.W4A16_NVFP4
