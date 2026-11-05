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
"""CPU regression tests for the b12x NVFP4 block-scale arithmetic.

The b12x kernel dequantizes a weight block as ``fp4_code * stored_sf *
alpha / input_global_scale``. The original TRT-LLM conversion baked the
per-expert ``weight_scale_2`` into ``stored_sf`` before the E4M3 cast; with
``weight_scale_2`` around ``2e-5`` every block scale lands in the E4M3
subnormal range and some round to zero. These tests reproduce that loss with
torch's CPU ``float8_e4m3fn`` cast and pin the decoupled alternative: keep
the normalized block scale and carry ``weight_scale_2`` through the alpha.
"""

import math

import pytest
import torch

# The marker is what makes the file reachable from the CPU stage.
pytestmark = pytest.mark.cpu_only

E4M3_MAX = 448.0
E4M3_MIN_NORMAL = 2.0**-6
WEIGHT_SCALE_2 = 2e-5
FC1_INPUT_SCALE = 0.0020461310632526875
FC2_INPUT_SCALE = 0.0340401791036129
SF_VEC_SIZE = 16

# The issue's layer-1 shape and synthetic scale distribution.
NUM_EXPERTS = 8
INTERMEDIATE_SIZE = 2688
HIDDEN_SIZE = 1024
SEEDS = (0, 1, 2)


def _case(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized E4M3 block scales and the per-expert weight_scale_2."""
    generator = torch.Generator().manual_seed(seed)
    log_scales = torch.randn(
        (NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE // SF_VEC_SIZE),
        generator=generator,
    ) * 0.55 + math.log(180.0)
    normalized = log_scales.exp().clamp(E4M3_MIN_NORMAL, E4M3_MAX).to(torch.float8_e4m3fn)
    weight_scale_2 = (
        WEIGHT_SCALE_2 * (1.0 + 0.1 * torch.randn((NUM_EXPERTS,), generator=generator))
    ).clamp_min(1e-12)
    return normalized, weight_scale_2


def _mean_relative_error(pred: torch.Tensor, true: torch.Tensor) -> float:
    relative = (pred - true).abs() / true.abs().clamp_min(1e-30)
    return float(relative.mean())


def test_baking_weight_scale_2_underflows_e4m3_block_scales():
    """The removed conversion loses ~15.5% of the weight scale on average."""
    mean_errors = []
    zero_fractions = []
    for seed in SEEDS:
        normalized, weight_scale_2 = _case(seed)
        normalized = normalized.float()
        exact = normalized * weight_scale_2.view(-1, 1, 1)

        baked = exact.to(torch.float8_e4m3fn).float()

        assert baked.max() < E4M3_MIN_NORMAL
        mean_errors.append(_mean_relative_error(baked, exact))
        zero_fractions.append(float((baked == 0).float().mean()))

    # The report measured 15.50% mean error and 0.96% zeroed over these seeds.
    assert 0.14 <= sum(mean_errors) / len(mean_errors) <= 0.17
    assert 0.005 <= sum(zero_fractions) / len(zero_fractions) <= 0.015


def test_decoupled_fc1_scale_preserves_the_checkpoint_scale():
    """Keeping normalized SF and passing weight_scale_2 through alpha is exact."""
    for seed in SEEDS:
        normalized, weight_scale_2 = _case(seed)
        normalized = normalized.float()
        exact = normalized * weight_scale_2.view(-1, 1, 1)

        # What the b12x wrapper folds for FC1: alpha * input_global_scale.
        weight_alpha = weight_scale_2
        folded_alpha = weight_alpha * FC1_INPUT_SCALE
        effective = normalized * (folded_alpha / FC1_INPUT_SCALE).view(-1, 1, 1)

        assert normalized.min() >= E4M3_MIN_NORMAL
        assert normalized.max() <= E4M3_MAX
        assert _mean_relative_error(effective, exact) < 1e-7
        assert (effective - exact).abs().max() < exact.abs().max() * 1e-6


def test_decoupled_fc2_scale_preserves_the_checkpoint_scale():
    """FC2 folds fc2_input_scale into w2_alpha and divides it back out."""
    for seed in SEEDS:
        normalized, weight_scale_2 = _case(seed)
        normalized = normalized.float()
        exact = normalized * weight_scale_2.view(-1, 1, 1)

        weight_alpha = weight_scale_2 * FC2_INPUT_SCALE
        effective = normalized * (weight_alpha / FC2_INPUT_SCALE).view(-1, 1, 1)

        assert _mean_relative_error(effective, exact) < 1e-7
        assert (effective - exact).abs().max() < exact.abs().max() * 1e-6
