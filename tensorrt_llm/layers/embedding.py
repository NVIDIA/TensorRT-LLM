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
from typing import Optional, Sequence, Union

import numpy as np
import torch

from .._utils import set_obj_attrs, str_dtype_to_torch, trt_dtype_to_np
from ..functional import (ACT2FN, Tensor, arange, concat, constant, cos, div,
                          embedding, exp, expand, identity, meshgrid2d, outer,
                          pad, shape, sin, slice, unsqueeze, where)
from ..mapping import Mapping
from ..module import Module
from ..parameter import Parameter
from .linear import ColumnLinear, Linear, RowLinear


class Embedding(Module):
    """
    The embedding layer takes input indices (x) and the embedding lookup table (weight) as input.
    And output the corresponding embeddings according to input indices.
    The size of weight is [num_embeddings, embedding_dim]

    Four parameters (tp_size, tp_group, sharding_dim, tp_rank) are involved in tensor parallelism.
    Only when "tp_size > 1 and tp_group is not None", tensor parallelism is enabled.
        When "sharding_dim == 0", the weight is shared in the vocabulary dimension.
            tp_rank must be set when sharding_dim == 0.
        When "sharding_dim == 1",  the weight is shard in the hidden dimension.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 dtype: Optional[str] = None,
                 tp_size: int = 1,
                 tp_group: Optional[list] = None,
                 sharding_dim: int = 0,
                 tp_rank: Optional[int] = None):
        super().__init__()
        # num_embeddings records the total vocab size no matter using TP or not
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.sharding_dim = sharding_dim
        self.tp_rank = tp_rank
        self.dtype = dtype
        self.tp_dim = sharding_dim

        if sharding_dim == 1:
            shape = (self.num_embeddings, self.embedding_dim // self.tp_size)
        elif sharding_dim == 0:
            shape = (math.ceil(self.num_embeddings / self.tp_size),
                     self.embedding_dim)

        self.weight = Parameter(shape=shape, dtype=dtype)

        self.weight_padding_size = ((8 - shape[0] % 8) % 8, shape[1])

        set_obj_attrs(self.weight, {
            "weight_loader": self.weight_loader,
        })

    def forward(self, x):
        # The embedding weight is padded to the multiple of 8.
        # The reason is that when lm_head and vocab_embedding are using the same embedding weight,
        # previously weights can't be depulicated in the engine because gemm will pad the weight to the multiple of 8.
        # If we also pad the embedding weight to the multiple of 8, the weights can be successfully deduplicated.
        # This will not affect the input and output of the gather op and perf impact is negligible.
        if self.weight_padding_size[0] != 0:
            padding_values = np.zeros(self.weight_padding_size,
                                      dtype=trt_dtype_to_np(
                                          self.weight.value.dtype))
            padding = constant(padding_values)
        else:
            padding = None

        return embedding(x,
                         self.weight.value,
                         tp_size=self.tp_size,
                         tp_group=self.tp_group,
                         sharding_dim=self.sharding_dim,
                         tp_rank=self.tp_rank,
                         padding=padding)

    def weight_loader(self, mapping: Mapping, param: Parameter,
                      loaded_weight: torch.Tensor):
        # use_parallel_embedding
        tp_rank = mapping.tp_rank
        if self.tp_size > 1:
            sharding_dim = self.sharding_dim
            shard_size = param._shape[sharding_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(sharding_dim, start_idx,
                                                 shard_size)
        param.value = loaded_weight

    def postprocess(self, tllm_key, weights, **kwargs):
        if weights is None:
            return {}
        weights = weights.to(str_dtype_to_torch(self.dtype))
        return {tllm_key: weights}


class PromptTuningEmbedding(Embedding):
    """
    PromptTuningEmbedding handles fine-tuned prompts with virtual tokens. At runtime,
    a supplementary embedding dictionary is passed. Tokens whose ids are >= vocab_size are embedded
    with that additional dictionary.
    The prompt tuning dictionary holds multiple tasks, and each sequence is assigned a given task.
    Prompt-tuned tokens from a given sequence use the adequate task dictionary, as defined by the `tasks` input.
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 vocab_size=None,
                 dtype=None,
                 tp_size=1,
                 tp_group=None,
                 sharding_dim=0,
                 tp_rank=0):
        super().__init__(num_embeddings, embedding_dim, dtype, tp_size,
                         tp_group, sharding_dim, tp_rank)
        if vocab_size is None:
            vocab_size = num_embeddings
        self.vocab_size = vocab_size

    def forward(self, tokens, prompt_embedding_table, tasks, task_vocab_size):
        """
            Pass all tokens through both normal and prompt embedding tables.
            Tokens are masked so that "normal" embedding only see "normal" tokens. Same logic for "prompt" embedding.
            After those two embedding, combine results based on whether the token was "normal" or "prompt-tuned".

        Parameters:
            tokens : Tensor
                the ids to embed, size [batch_size, seq_len]

            prompt_embedding_table : Tensor
                the additional embedding table for prompt-tuned tokens, size [num_tasks * num_tokens_per_task, hidden_size]

            tasks: Tensor
                the task required by each token, size [batch_size, seq_len]

            task_vocab_size: Tensor
                the number of tokens used for each task, should be equal to prompt_embedding_table's num_tokens_per_task, size [1]

        Returns:
            Tokens' embedding
        """
        # do not use ">=" because internally the layer works with floating points
        prompt_tokens_mask = tokens > (self.vocab_size - 1)

        # clip tokens in the [0, vocab_size) range
        normal_tokens = where(prompt_tokens_mask, self.vocab_size - 1, tokens)
        normal_embeddings = embedding(normal_tokens, self.weight.value,
                                      self.tp_size, self.tp_group,
                                      self.sharding_dim, self.tp_rank)

        # put virtual tokens in the [0, max_prompt_vocab_size) range
        prompt_tokens = where(prompt_tokens_mask, tokens - self.vocab_size, 0)
        # add offsets to match the concatenated embedding tables
        tasks = tasks * task_vocab_size

        # tasks: [batch_size, seq_len]
        # prompt_tokens: [batch_size, seq_len]
        # if speculative decoding is enabled the shape of prompt_tokens is [batch_size, seq_len + max_draft_len],
        # so we need to expand tasks to [batch_size, seq_len + max_draft_len]
        tasks = expand(tasks, shape(prompt_tokens))
        prompt_tokens = prompt_tokens + tasks
        prompt_embeddings = embedding(prompt_tokens, prompt_embedding_table)

        # prompt_tokens_mask: [batch_size, seq_len] -> [batch_size, seq_len, 1]
        # combine the correct sources of embedding: normal/prompt
        return where(unsqueeze(prompt_tokens_mask, -1), prompt_embeddings,
                     normal_embeddings)


class LabelEmbedding(Module):

    def __init__(self,
                 num_classes: int,
                 hidden_size: int,
                 dropout_prob: float = 0.0,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = Embedding(num_classes + use_cfg_embedding,
                                         hidden_size,
                                         tp_size=mapping.tp_size,
                                         tp_group=mapping.tp_group,
                                         dtype=dtype)
        self.num_classes = num_classes

    def token_drop(self, labels: Tensor, force_drop_ids: Tensor):
        labels = where(force_drop_ids == 1, self.num_classes, labels)
        return labels

    def forward(self, labels: Tensor, force_drop_ids: Optional[Tensor] = None):
        if force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: Tensor):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    omega = constant(omega.numpy().astype(np.float32))

    pos = pos.view([-1])  # (M,)
    out = outer(pos, omega)  # (M, D/2), outer product

    emb_sin = sin(out)  # (M, D/2)
    emb_cos = cos(out)  # (M, D/2)

    emb = concat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: Sequence[Tensor]):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[1])  # (H*W, D/2)

    emb = concat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: Union[int, Sequence[int]],
    cls_token: bool = False,
    extra_tokens: int = 0,
    interpolation_scale: float = 1.0,
    base_size: int = 16,
):
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = div(
        div(arange(0, grid_size[0], 'float32'),
            float(grid_size[0] / base_size)), interpolation_scale)
    grid_w = div(
        div(arange(0, grid_size[1], 'float32'),
            float(grid_size[1] / base_size)), interpolation_scale)
    grid_h, grid_w = meshgrid2d(grid_w, grid_h)  # here w goes first
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, [
        grid_h.view([1, grid_size[1], grid_size[0]]),
        grid_w.view([1, grid_size[1], grid_size[0]])
    ])
    if cls_token and extra_tokens > 0:
        pos_embed = concat([
            constant(
                np.zeros(shape=(extra_tokens, embed_dim),
                         dtype=trt_dtype_to_np(pos_embed.dtype))), pos_embed
        ],
                           dim=0)
    return pos_embed


class SD3PatchEmbed(Module):
    """
    2D Image to Patch Embedding with support for SD3 cropping.
    """

    def __init__(
            self,
            height: int = 224,
            width: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            embed_dim: int = 768,
            layer_norm: bool = False,
            flatten: bool = True,
            bias: bool = True,
            interpolation_scale: int = 1,
            pos_embed_type: str = "sincos",
            pos_embed_max_size: Optional[int] = None,  # For SD3 cropping
            dtype=None):
        from diffusers.models.embeddings import \
            get_2d_sincos_pos_embed as get_2d_sincos_pos_embed_torch

        from .conv import Conv2d
        from .normalization import LayerNorm

        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = Conv2d(in_channels,
                           embed_dim,
                           kernel_size=(patch_size, patch_size),
                           stride=(patch_size, patch_size),
                           bias=bias,
                           dtype=dtype)
        if layer_norm:
            self.norm = LayerNorm(embed_dim,
                                  elementwise_affine=False,
                                  eps=1e-6,
                                  dtype=dtype)
        else:
            self.norm = None

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale

        # Calculate positional embeddings based on max size or default
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches**0.5)

        if pos_embed_type is None:
            self.pos_embed = None
        elif pos_embed_type == "sincos":
            pos_embed = get_2d_sincos_pos_embed_torch(
                embed_dim,
                grid_size,
                base_size=self.base_size,
                interpolation_scale=self.interpolation_scale,
                output_type="pt",
            )
            self.pos_embed = Parameter(
                pos_embed.detach().cpu().float().unsqueeze(0), dtype=dtype)
        else:
            raise ValueError(
                f"Unsupported pos_embed_type: {self.pos_embed_type}")

    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = identity(self.pos_embed.value).view(
            [1, self.pos_embed_max_size, self.pos_embed_max_size, -1])
        spatial_pos_embed = slice(spatial_pos_embed,
                                  starts=[0, top, left, 0],
                                  sizes=concat([
                                      shape(spatial_pos_embed, 0), height,
                                      width,
                                      shape(spatial_pos_embed, 3)
                                  ]))
        spatial_pos_embed = spatial_pos_embed.view(
            concat(
                [1, -1,
                 shape(spatial_pos_embed,
                       spatial_pos_embed.ndim() - 1)]))
        return spatial_pos_embed

    def forward(self, latent):
        # [TODO] to support height and width for runtime
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = latent.shape[-2] // self.patch_size, latent.shape[
                -1] // self.patch_size
        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)
        if self.pos_embed is None:
            return latent.cast(latent.dtype)
        # Interpolate or crop positional embeddings as needed
        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        else:
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.value.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = unsqueeze(pos_embed.cast('float32'), axis=0)
            else:
                pos_embed = self.pos_embed.value

        pos_embed = pos_embed.cast(latent.dtype)
        output = (latent + pos_embed).cast(latent.dtype)
        return output


def get_timestep_embedding(
    timesteps: Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * np.arange(
        start=0, stop=half_dim, dtype=np.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    exponent = constant(exponent)

    emb = exp(exponent)
    emb = unsqueeze(timesteps, -1).cast('float32') * unsqueeze(emb, 0)

    # scale embeddings
    emb = scale * emb

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = concat([cos(emb), sin(emb)], dim=-1)
    else:
        emb = concat([sin(emb), cos(emb)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(Module):

    def __init__(self,
                 in_channels: int,
                 time_embed_dim: int,
                 act_fn: str = "silu",
                 out_dim: int = None,
                 post_act_fn: Optional[str] = None,
                 cond_proj_dim=None,
                 sample_proj_bias=True,
                 mapping=None,
                 dtype=None):
        super().__init__()
        tp_group = mapping.tp_group
        tp_size = mapping.tp_size
        self.linear_1 = ColumnLinear(in_channels,
                                     time_embed_dim,
                                     sample_proj_bias,
                                     tp_group=tp_group,
                                     tp_size=tp_size,
                                     dtype=dtype,
                                     gather_output=False)

        if cond_proj_dim is not None:
            self.cond_proj = Linear(cond_proj_dim,
                                    in_channels,
                                    bias=False,
                                    dtype=dtype)
        else:
            self.cond_proj = None

        self.act = ACT2FN[act_fn]

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = RowLinear(time_embed_dim,
                                  time_embed_dim_out,
                                  sample_proj_bias,
                                  tp_group=tp_group,
                                  tp_size=tp_size,
                                  dtype=dtype)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = ACT2FN[post_act_fn]

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(Module):

    def __init__(self,
                 num_channels: int,
                 flip_sin_to_cos: bool,
                 downscale_freq_shift: float,
                 scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps) -> Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class PixArtAlphaTextProjection(Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self,
                 in_features,
                 hidden_size,
                 out_features=None,
                 act_fn="gelu_tanh",
                 mapping=None,
                 dtype=None):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        tp_group = mapping.tp_group
        tp_size = mapping.tp_size
        self.linear_1 = ColumnLinear(in_features=in_features,
                                     out_features=hidden_size,
                                     bias=True,
                                     tp_group=tp_group,
                                     tp_size=tp_size,
                                     dtype=dtype,
                                     gather_output=False)
        self.act_1 = ACT2FN[act_fn]
        self.linear_2 = RowLinear(in_features=hidden_size,
                                  out_features=out_features,
                                  bias=True,
                                  tp_group=tp_group,
                                  tp_size=tp_size,
                                  dtype=dtype)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class CombinedTimestepTextProjEmbeddings(Module):

    def __init__(self,
                 embedding_dim,
                 pooled_projection_dim,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256,
                                   flip_sin_to_cos=True,
                                   downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256,
                                                   time_embed_dim=embedding_dim,
                                                   mapping=mapping,
                                                   dtype=dtype)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim,
                                                       embedding_dim,
                                                       act_fn="silu",
                                                       mapping=mapping,
                                                       dtype=dtype)

    def forward(self, timestep: Tensor, pooled_projection: Tensor):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.cast(dtype=pooled_projection.dtype))  # (N, D)

        pooled_projections = self.text_embedder(pooled_projection)

        conditioning = timesteps_emb + pooled_projections
        self.register_network_output('output', conditioning)
        return conditioning


class CombinedTimestepLabelEmbeddings(Module):

    def __init__(self,
                 num_classes,
                 embedding_dim,
                 class_dropout_prob=0.0,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256,
                                   flip_sin_to_cos=True,
                                   downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256,
                                                   time_embed_dim=embedding_dim,
                                                   mapping=mapping,
                                                   dtype=dtype)
        self.class_embedder = LabelEmbedding(num_classes,
                                             embedding_dim,
                                             class_dropout_prob,
                                             mapping=mapping,
                                             dtype=dtype)

    def forward(self,
                timestep: Tensor,
                class_labels: Tensor,
                hidden_dtype: Optional[str] = 'float32'):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.cast(dtype=hidden_dtype))  # (N, D)
        class_labels = self.class_embedder(class_labels)  # (N, D)
        conditioning = timesteps_emb + class_labels  # (N, D)
        return conditioning
