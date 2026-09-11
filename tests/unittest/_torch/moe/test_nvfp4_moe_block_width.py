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

from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4CutlassFusedMoEMethod
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = pytest.mark.cpu_only


def test_nvfp4_moe_rejects_32_element_scale_blocks():
    """Only the dense W4A16_NVFP4 Linear path dequantizes 32-element blocks; the
    MoE methods must fail before allocating weights for such a checkpoint."""
    module = SimpleNamespace(
        quant_config=QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=32)
    )

    with pytest.raises(NotImplementedError, match="group_size=32"):
        NVFP4CutlassFusedMoEMethod().create_weights(module)
