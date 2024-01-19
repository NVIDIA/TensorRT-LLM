# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math

from ..._utils import fp32_array
from ...functional import concat, constant, cos, exp, silu, sin
from ...layers import Linear
from ...module import Module


def get_timestep_embedding(timesteps,
                           embedding_dim,
                           flip_sin_to_cos=False,
                           downscale_freq_shift=1.0,
                           scale=1.0,
                           max_period=10000):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert timesteps.rank() == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2

    exponent = [
        i * -math.log(max_period) / (half_dim - downscale_freq_shift)
        for i in range(half_dim)
    ]

    emb = exp(constant(fp32_array(exponent)))

    ts_shape = list(timesteps.size())
    ts_shape.append(1)
    emb_shape = list(emb.size())
    emb_shape.insert(0, 1)

    emb = timesteps.view(ts_shape) * emb.view(emb_shape)

    emb = scale * emb
    # concat sine and cosine embeddings

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = concat([cos(emb), sin(emb)], dim=1)
    else:
        emb = concat([sin(emb), cos(emb)], dim=1)

    #TODO Enable below logic when TensorRT-LLM supports pad feature.
    # zero pad
    # if embedding_dim % 2 == 1:
    #     emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(Module):

    def __init__(self, channel, time_embed_dim, act_fn="silu"):
        super().__init__()

        self.linear_1 = Linear(channel, time_embed_dim)
        self.act = None
        if act_fn == "silu":
            self.act = silu
        self.linear_2 = Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)
        return sample


class Timesteps(Module):

    def __init__(self, num_channels, flip_sin_to_cos, downscale_freq_shift):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb
