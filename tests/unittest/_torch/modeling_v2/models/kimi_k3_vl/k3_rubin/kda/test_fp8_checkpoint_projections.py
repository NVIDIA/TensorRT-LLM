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
"""Numerical coverage for shared KDA checkpoint projections on Rubin."""

import weakref

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.kda.kimi_kda_mixer import (
    KimiKDALinearAttention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.text_model import (
    _Fp8BlockScaleWeightReadLinear as K3Fp8Linear,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="needs SM107 with Rubin CuTe DSL support",
)
@pytest.mark.parametrize("owner_first", [False, True])
@pytest.mark.parametrize("num_rows", [1, 8, 64])
def test_fused_checkpoint_projections_preserve_outputs(owner_first: bool, num_rows: int) -> None:
    """Shared checkpoint storage must keep FP8 codes paired with their scales."""
    with torch.device("cuda"):
        sources = tuple(
            Linear(
                512,
                128,
                bias=False,
                dtype=torch.bfloat16,
                quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
                reduce_output=False,
            )
            for _ in range(4)
        )
    for source in sources:
        source.load_weights(
            [
                {
                    "weight": torch.full((128, 512), 448, dtype=torch.float8_e4m3fn, device="cuda"),
                    "weight_scale": torch.full((1, 4), 0.51, device="cuda"),
                }
            ]
        )

    fused_source = KimiKDALinearAttention._fuse_checkpoint_projections(sources)
    KimiKDALinearAttention._alias_projections_into_fused(sources, fused_source)
    fused = K3Fp8Linear.from_linear(fused_source)
    projections = tuple(K3Fp8Linear.from_linear(source) for source in sources)
    for index, projection in enumerate(projections):
        projection._fused_projection = (weakref.ref(fused), index * 128)
    modules = (fused, *projections) if owner_first else (*projections, fused)
    for module in modules:
        module.post_load_weights()

    # 448 * 0.51 rounds to FP8 224 at scale 1. Resmoothing that representation
    # again can produce 448 at scale 0.5, so stale scales would double outputs.
    x = torch.full((num_rows, 512), 1 / 512, dtype=torch.bfloat16, device="cuda")
    for module in (fused, *projections):
        actual = module(x)
        torch.testing.assert_close(actual, torch.full_like(actual, 224), rtol=0, atol=0)
