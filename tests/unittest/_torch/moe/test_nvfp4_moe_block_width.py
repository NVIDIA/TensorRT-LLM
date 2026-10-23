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

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.quantization import (
    NVFP4CutlassFusedMoEMethod,
    W4A8NVFP4FP8TRTLLMGenFusedMoEMethod,
)
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = pytest.mark.cpu_only


def test_nvfp4_moe_rejects_32_element_scale_blocks() -> None:
    """Only the dense W4A16_NVFP4 Linear path dequantizes 32-element blocks; the
    MoE methods must fail before allocating weights for such a checkpoint."""
    module = SimpleNamespace(
        quant_config=QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=32)
    )

    with pytest.raises(NotImplementedError, match="group_size=32"):
        NVFP4CutlassFusedMoEMethod().create_weights(module)


@pytest.mark.parametrize("group_size", [None, 16, 32, 128])
def test_w4a8_nvfp4_fp8_moe_allocates_kernel_fixed_scale_blocks(group_size: int | None) -> None:
    """W4A8 allocates 32-wide scales even for a default or 16-wide QuantConfig."""
    module = torch.nn.Module()
    module.quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_NVFP4_FP8, group_size=group_size)
    module.expert_size_per_partition = 2
    module.hidden_size = 128
    module.intermediate_size_per_partition = 64
    module.expand_intermediate_size_per_partition = 128
    module.bias = False
    method = W4A8NVFP4FP8TRTLLMGenFusedMoEMethod()
    # Exercise actual allocation; EPLB and runtime scale setup are independent.
    with (
        patch.object(method, "_online_eplb_not_verified"),
        patch.object(method, "setup_quant_scales"),
    ):
        method.create_weights(module)

    assert module.scaling_vector_size == 32
    assert module.w3_w1_weight_scale.shape == (2, 128, 4)
    assert module.w2_weight_scale.shape == (2, 128, 2)
    assert module.w3_w1_weight.shape == (2, 128, 64)
    assert module.w2_weight.shape == (2, 128, 32)
