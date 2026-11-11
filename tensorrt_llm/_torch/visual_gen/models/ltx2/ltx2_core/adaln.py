# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.transformer import adaln as upstream

from ._linear import use_linear_factory


class AdaLayerNormSingle(upstream.AdaLayerNormSingle):
    def __init__(self, *args, make_linear=None, **kwargs):
        super().__init__(*args, **kwargs)
        use_linear_factory(self, make_linear)
