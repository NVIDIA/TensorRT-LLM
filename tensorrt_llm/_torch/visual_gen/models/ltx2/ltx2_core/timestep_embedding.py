# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer import (
    timestep_embedding as upstream,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.timestep_embedding import (
    Timesteps as Timesteps,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.timestep_embedding import (
    get_timestep_embedding as get_timestep_embedding,
)

from ._linear import use_linear_factory


class TimestepEmbedding(upstream.TimestepEmbedding):
    def __init__(self, *args, make_linear=None, **kwargs):
        super().__init__(*args, **kwargs)
        use_linear_factory(self, make_linear)


class PixArtAlphaCombinedTimestepSizeEmbeddings(upstream.PixArtAlphaCombinedTimestepSizeEmbeddings):
    def __init__(self, *args, make_linear=None, **kwargs):
        super().__init__(*args, **kwargs)
        use_linear_factory(self, make_linear)
