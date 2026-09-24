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
"""Regression tests for LTX-2 spatiotemporal scale factor ordering.

Covers https://github.com/NVIDIA/TensorRT-LLM/issues/19495: SpatioTemporalScaleFactors
must declare fields in tensor-axis order (time, height, width) so that
VideoLatentShape.from_pixel_shape, VideoLatentShape.upscale, and
get_pixel_coords (which converts the tuple to a tensor positionally) all agree
on which value scales which axis. A swapped declaration order silently mixes up
height and width for any non-square (asymmetric) resolution.

No checkpoint or GPU required — pure Python/tensor-shape arithmetic.
"""

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.patchifier import get_pixel_coords
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.types import (
    SpatioTemporalScaleFactors,
    VideoLatentShape,
    VideoPixelShape,
)


def test_from_pixel_shape_uses_named_axes_for_asymmetric_factors():
    scale = SpatioTemporalScaleFactors(time=2, height=4, width=8)
    pixel = VideoPixelShape(batch=1, frames=9, height=16, width=32, fps=24.0)

    latent = VideoLatentShape.from_pixel_shape(pixel, scale_factors=scale)

    assert (latent.frames, latent.height, latent.width) == (5, 4, 4)


def test_upscale_round_trips_asymmetric_factors():
    scale = SpatioTemporalScaleFactors(time=2, height=4, width=8)
    pixel = VideoPixelShape(batch=1, frames=9, height=16, width=32, fps=24.0)

    latent = VideoLatentShape.from_pixel_shape(pixel, scale_factors=scale)
    upscaled = latent.upscale(scale)

    assert (upscaled.frames, upscaled.height, upscaled.width) == (9, 16, 32)


def test_get_pixel_coords_scales_height_and_width_independently():
    scale = SpatioTemporalScaleFactors(time=2, height=4, width=8)
    # One patch step along each of (time, height, width).
    latent_coords = torch.tensor([0, 1, 1], dtype=torch.float32).view(1, 3, 1, 1)

    pixel_coords = get_pixel_coords(latent_coords, scale)

    assert pixel_coords.view(-1).tolist() == [0, 4, 8]


def test_default_factors_are_symmetric():
    default = SpatioTemporalScaleFactors.default()

    assert default == SpatioTemporalScaleFactors(time=8, height=32, width=32)
