# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""The helix8 glue: the KDA q/k/v/g FP8 views and their fused qkvg projection share codes and
scale grids.

Resmoothing to UE8M0 is in place and not idempotent, so the shared storage must be
resmoothed exactly once (by the fused owner) whichever module is prepared first.
"""

import weakref

import pytest
import torch
from utils.util import getSMVersion

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__helix8_moeep8.modeling import (  # noqa: E501  # noqa: E501
    K3KDALinearAttention,
    _Fp8BlockScaleWeightReadLinear,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.gemm.sm107_linear import (
    Fp8BlockScaleLinear,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE


@pytest.mark.skipif(
    getSMVersion() not in (100, 103, 107)
    or (getSMVersion() == 107 and not IS_CUTLASS_DSL_RUBIN_AVAILABLE),
    reason="needs a supported block-scaled FP8 GPU and toolchain",
)
@pytest.mark.parametrize("owner_first", [False, True])
def test_fused_checkpoint_projections_preserve_outputs(owner_first: bool) -> None:
    """Shared checkpoint storage must keep FP8 codes paired with their scales."""
    sources = tuple(Fp8BlockScaleLinear(512, 128, device="cuda") for _ in range(4))
    for source in sources:
        source.load(
            torch.full((128, 512), 448, dtype=torch.float8_e4m3fn, device="cuda"),
            torch.full((1, 4), 0.51, device="cuda"),
        )

    fused_source = K3KDALinearAttention._fuse_checkpoint_projections(sources)
    K3KDALinearAttention._alias_projections_into_fused(sources, fused_source)
    fused = _Fp8BlockScaleWeightReadLinear.from_linear(fused_source)
    projections = tuple(_Fp8BlockScaleWeightReadLinear.from_linear(source) for source in sources)
    for index, projection in enumerate(projections):
        projection._fused_projection = (weakref.ref(fused), index * 128)
    modules = (fused, *projections) if owner_first else (*projections, fused)
    for module in modules:
        module.post_load_weights()

    # 448 * 0.51 rounds to FP8 224 at scale 1. Resmoothing that representation
    # again can produce 448 at scale 0.5, so stale scales would double outputs.
    x = torch.full((8, 512), 1 / 512, dtype=torch.bfloat16, device="cuda")
    for module in (fused, *projections):
        actual = module(x)
        torch.testing.assert_close(actual, torch.full_like(actual, 224), rtol=0, atol=0)
