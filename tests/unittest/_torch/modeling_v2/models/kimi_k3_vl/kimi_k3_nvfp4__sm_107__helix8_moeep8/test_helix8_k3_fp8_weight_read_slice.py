# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output slices of the helix8 glue's FP8 weight-read projection (the Helix context heads of its
MLA)."""

import pytest
import torch
from utils.util import getSMVersion

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__helix8_moeep8.modeling import (  # noqa: E501
    _Fp8BlockScaleWeightReadLinear,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.gemm.sm107_linear import (
    Fp8BlockScaleLinear,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

IN_FEATURES, OUT_FEATURES = 512, 1152


def _projection() -> _Fp8BlockScaleWeightReadLinear:
    """A checkpoint FP8 block-scale projection swapped as K3 swaps attention projections."""
    torch.manual_seed(42)
    linear = Fp8BlockScaleLinear(IN_FEATURES, OUT_FEATURES, device="cuda")
    weight = torch.randn(OUT_FEATURES, IN_FEATURES, device="cuda") * 0.02
    blocks = weight.view(OUT_FEATURES // 128, 128, IN_FEATURES // 128, 128)
    scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-8) / 448.0
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    linear.load((weight / expanded).to(torch.float8_e4m3fn), scale)
    projection = _Fp8BlockScaleWeightReadLinear.from_linear(linear)
    projection.post_load_weights()
    return projection


@pytest.mark.skipif(
    getSMVersion() not in (100, 103, 107)
    or (getSMVersion() == 107 and not IS_CUTLASS_DSL_RUBIN_AVAILABLE),
    reason="needs a supported block-scaled FP8 GPU and toolchain",
)
@pytest.mark.parametrize("start,end", [(0, 576), (576, 1152), (513, 1107)])
@torch.inference_mode()
def test_forward_slice_matches_the_full_projection(start, end):
    """``forward_slice`` is ``forward`` cropped to the slice; the prepared scale is reused and the
    projection's own weight and scale are untouched."""
    projection = _projection()
    weight = projection.weight.clone()
    scale = projection.weight_scale.clone()
    inputs = torch.randn(2, 17, IN_FEATURES, dtype=torch.bfloat16, device="cuda")
    expected = projection(inputs)[..., start:end]

    projection.prepare_output_slice(start, end)
    cached = projection._output_slice_cache.get(projection.weight, start, end)[1]
    actual = projection.forward_slice(inputs, start, end)

    assert actual.shape == expected.shape and actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.03)
    assert projection._output_slice_cache.get(projection.weight, start, end)[1] is cached
    assert torch.equal(projection.weight.view(torch.uint8), weight.view(torch.uint8))
    assert torch.equal(projection.weight_scale, scale)


@pytest.mark.skipif(
    getSMVersion() not in (100, 103, 107)
    or (getSMVersion() == 107 and not IS_CUTLASS_DSL_RUBIN_AVAILABLE),
    reason="needs a supported block-scaled FP8 GPU and toolchain",
)
@torch.inference_mode()
def test_prepare_output_slice_rejects_an_out_of_range_slice():
    projection = _projection()
    with pytest.raises(ValueError, match="Invalid output feature range"):
        projection.prepare_output_slice(0, OUT_FEATURES + 1)
