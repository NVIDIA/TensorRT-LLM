# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports from the pinned November 4 LTX core."""

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.convolution import (
    CausalConv3d as CausalConv3d,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.convolution import (
    DualConv3d as DualConv3d,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.convolution import (
    make_conv_nd as make_conv_nd,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.convolution import (
    make_linear_nd as make_linear_nd,
)
