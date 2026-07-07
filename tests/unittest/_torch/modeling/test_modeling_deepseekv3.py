# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.distributed import AllReduceFusionOp
from tensorrt_llm._torch.utils import Fp4QuantizedTensor


@pytest.fixture
def mock_decoder_layer():
    """Create a mock DeepseekV3DecoderLayer with the real forward_mlp method."""
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3DecoderLayer

    layer = MagicMock(spec=DeepseekV3DecoderLayer)

    # Bind the real forward_mlp method to our mock
    layer.forward_mlp = DeepseekV3DecoderLayer.forward_mlp.__get__(layer, DeepseekV3DecoderLayer)

    return layer


@pytest.mark.parametrize("has_nvfp4", [True, False], ids=["nvfp4_enabled", "nvfp4_disabled"])
def test_forward_mlp_pre_mlp_fusion_fallback_no_nvfp4(mock_decoder_layer, has_nvfp4):
    """Test that forward_mlp selects the correct AllReduceFusionOp based on
    mlp.gate_up_proj.has_nvfp4 when PRE_MLP_FUSION is enabled.

    When has_nvfp4 is True: uses RESIDUAL_RMS_NORM_QUANT_NVFP4
    When has_nvfp4 is False: falls back to RESIDUAL_RMS_NORM
    """
    layer = mock_decoder_layer

    # Setup fusion_config with PRE_MLP_FUSION enabled, POST_MLP_FUSION disabled
    layer.fusion_config = MagicMock()
    layer.fusion_config.PRE_MLP_FUSION = True
    layer.fusion_config.POST_MLP_FUSION = False

    # Setup mlp mock
    layer.mlp = MagicMock()
    layer.mlp.gate_up_proj = MagicMock()
    layer.mlp.gate_up_proj.has_nvfp4 = has_nvfp4
    layer.mlp.gate_up_proj.input_scale = torch.tensor(1.0)

    # Setup post_attention_layernorm mock
    layer.post_attention_layernorm = MagicMock()
    layer.post_attention_layernorm.weight = torch.ones(16)
    layer.post_attention_layernorm.variance_epsilon = 1e-6

    # Setup next_layer_layernorm mock
    layer.next_layer_layernorm = MagicMock()
    layer.next_layer_layernorm.weight = torch.ones(16)
    layer.next_layer_layernorm.variance_epsilon = 1e-6

    # Setup mlp_tp_size
    layer.mlp_tp_size = 1

    # Track allreduce calls
    captured_params = []

    def mock_allreduce(hidden_states, all_reduce_params=None):
        captured_params.append(all_reduce_params)
        if all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4:
            # Return (act_fp4, act_sf, residual) tuple for nvfp4 path
            act_fp4 = torch.zeros(4, dtype=torch.uint8)
            act_sf = torch.ones(1)
            residual = torch.ones(16)
            return act_fp4, act_sf, residual
        else:
            # Return (hidden_states, residual) tuple for non-nvfp4 path
            return torch.ones(16), torch.ones(16)

    layer.allreduce = mock_allreduce

    # Setup mlp to return a tensor
    mlp_output = torch.ones(16)
    layer.mlp.return_value = mlp_output

    # Setup next_layer_layernorm to return (hidden, residual)
    layer.next_layer_layernorm.return_value = (torch.ones(16), torch.ones(16))

    # Input tensors
    hidden_states = torch.randn(16)
    residual = torch.randn(16)

    # Call forward_mlp
    result_hidden, result_residual = layer.forward_mlp(
        hidden_states=hidden_states,
        residual=residual,
        spec_metadata=None,
    )

    # Verify the first allreduce call (PRE_MLP_FUSION path)
    assert len(captured_params) >= 1, "allreduce should have been called at least once"

    first_params = captured_params[0]

    if has_nvfp4:
        # Should use RESIDUAL_RMS_NORM_QUANT_NVFP4
        assert first_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4, (
            f"Expected RESIDUAL_RMS_NORM_QUANT_NVFP4 when has_nvfp4=True, got {first_params.fusion_op}"
        )
        # Verify scale is passed
        assert first_params.scale is not None, (
            "scale should be set when using RESIDUAL_RMS_NORM_QUANT_NVFP4"
        )
        # Verify mlp was called with Fp4QuantizedTensor
        mlp_call_args = layer.mlp.call_args
        assert mlp_call_args is not None, "mlp should have been called"
        first_arg = (
            mlp_call_args[0][0] if mlp_call_args[0] else mlp_call_args[1].get("hidden_states")
        )
        assert isinstance(first_arg, Fp4QuantizedTensor), (
            f"mlp input should be Fp4QuantizedTensor when has_nvfp4=True, got {type(first_arg)}"
        )
    else:
        # Should fall back to RESIDUAL_RMS_NORM
        assert first_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM, (
            f"Expected RESIDUAL_RMS_NORM when has_nvfp4=False, got {first_params.fusion_op}"
        )
        # Verify that RESIDUAL_RMS_NORM_QUANT_NVFP4 was NOT used
        assert first_params.fusion_op != AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4, (
            "RESIDUAL_RMS_NORM_QUANT_NVFP4 should NOT be used when has_nvfp4=False"
        )
        # Verify mlp was NOT called with Fp4QuantizedTensor
        mlp_call_args = layer.mlp.call_args
        assert mlp_call_args is not None, "mlp should have been called"
        first_arg = (
            mlp_call_args[0][0] if mlp_call_args[0] else mlp_call_args[1].get("hidden_states")
        )
        assert not isinstance(first_arg, Fp4QuantizedTensor), (
            f"mlp input should NOT be Fp4QuantizedTensor when has_nvfp4=False, got {type(first_arg)}"
        )
