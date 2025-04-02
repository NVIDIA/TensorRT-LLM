# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0
# This file is based on official VILA: https://github.com/NVlabs/VILA/
# and s2wrapper: https://github.com/bfshi/scaling_on_scales

import math

import torch
import torch.nn.functional as F
from einops import rearrange

#  ------------------------------------------------------------------------------------------
#  Original code by Baifeng Shi, licensed under the MIT License:
#  https://github.com/bfshi/scaling_on_scales/blob/master/LICENSE.md
#  ------------------------------------------------------------------------------------------


def s2_split_chessboard(x, num_split):
    """
        x: b * c * h * w
        Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
    """
    B, C, H, W = x.shape
    assert H % num_split == 0 and W % num_split == 0
    x_split = rearrange(x,
                        'b c (nh h) (nw w) -> (nh nw b) c h w',
                        nh=num_split,
                        nw=num_split)
    return x_split


def s2_merge_chessboard(x, num_split):
    """
        x: b * c * h * w
        Assuming x contains num_split**2 sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
        (inverse of split_chessboard)
    """
    B, C, H, W = x.shape
    assert B % (num_split**2) == 0
    x_merge = rearrange(x,
                        '(nh nw b) c h w -> b c (nh h) (nw w)',
                        nh=num_split,
                        nw=num_split)

    return x_merge


def s2_batched_forward(model, x, batch_size=-1):
    if batch_size == -1:
        return model(x)
    else:
        x_batched = x.split(batch_size)
        outs = [model(x) for x in x_batched]
        return torch.cat(outs, dim=0)


def multiscale_forward(model,
                       input,
                       scales=None,
                       img_sizes=None,
                       max_split_size=None,
                       resize_output_to_idx=0,
                       num_prefix_token=0,
                       output_shape='bnc',
                       split_forward=False):

    assert input.dim() == 4, "Input image must be in the shape of BxCxHxW."
    assert input.shape[2] == input.shape[
        3], "Currently only square images are supported."
    assert output_shape in [
        'bnc', 'bchw'
    ], "Output shape should be either BxNxC (e.g., ViT) or BxCxHxW (e.g., ConvNet)."
    assert output_shape == 'bnc' or num_prefix_token == 0, "For ConvNet there shouldn't be any prefix token."

    b, c, input_size, _ = input.shape

    # image size for each scale
    assert scales is not None or img_sizes is not None, "Please assign either scales or img_sizes."
    img_sizes = img_sizes or [int(input_size * scale) for scale in scales]

    # prepare multiscale inputs
    max_split_size = max_split_size or input_size  # The maximum size of each split of image. Set as the input size by default
    num_splits = [math.ceil(size / max_split_size)
                  for size in img_sizes]  # number of splits each scale
    input_multiscale = []
    for size, num_split in zip(img_sizes, num_splits):
        x = F.interpolate(input.to(torch.float32), size=size,
                          mode='bicubic').to(input.dtype)
        x = s2_split_chessboard(x, num_split=num_split)
        input_multiscale.append(x)

    # run feedforward on each scale
    outs_multiscale = [
        s2_batched_forward(model, x, b) if split_forward else model(x)
        for x in input_multiscale
    ]
    if num_prefix_token > 0:
        outs_prefix_multiscale = [
            out[:, :num_prefix_token] for out in outs_multiscale
        ]
        outs_multiscale = [out[:, num_prefix_token:] for out in outs_multiscale]
    if output_shape == 'bnc':
        outs_multiscale = [
            rearrange(out,
                      'b (h w) c -> b c h w',
                      h=int(out.shape[1]**0.5),
                      w=int(out.shape[1]**0.5)) for out in outs_multiscale
        ]

    # merge outputs of different splits for each scale separately
    outs_multiscale = [
        s2_merge_chessboard(out, num_split=num_split)
        for num_split, out in zip(num_splits, outs_multiscale)
    ]

    # interpolate outputs from different scales and concat together
    output_size = outs_multiscale[resize_output_to_idx].shape[-2]
    out = torch.cat([
        F.interpolate(outs_multiscale[i].to(torch.float32),
                      size=output_size,
                      mode='area').to(outs_multiscale[i].dtype)
        for i in range(len(outs_multiscale))
    ],
                    dim=1)
    if output_shape == 'bnc':
        out = rearrange(out, 'b c h w -> b (h w) c')
    if num_prefix_token > 0:
        # take the mean of prefix tokens from different splits for each scale
        outs_prefix_multiscale = [
            torch.stack(out.split(b, dim=0), dim=0).mean(dim=0)
            for out in outs_prefix_multiscale
        ]
        out_prefix_multiscale = torch.cat(outs_prefix_multiscale, dim=-1)
        out = torch.cat([out_prefix_multiscale, out], dim=1)

    return out
