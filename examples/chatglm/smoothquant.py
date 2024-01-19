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
'''
Utilities for SmoothQuant models
'''

import functools
from collections import defaultdict

import torch
from tqdm import tqdm


@torch.no_grad()
def apply_smoothing(
    scales,
    gemm_weights,
    norm_weights=None,
    norm_bias=None,
    dtype=torch.float32,
    norm_1p=False,
):
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]

    if norm_weights is not None:
        assert norm_weights.numel() == scales.numel()
        norm_weights.div_(scales).to(dtype)
    if norm_bias is not None:
        assert norm_bias.numel() == scales.numel()
        norm_bias.div_(scales).to(dtype)
    if norm_1p:
        norm_weights += (1 / scales) - 1

    for gemm in gemm_weights:
        gemm.mul_(scales.view(1, -1)).to(dtype)


@torch.no_grad()
def smooth_gemm(
    gemm_weights,
    act_scales,
    norm_weights=None,
    norm_bias=None,
    alpha=0.5,
    weight_scales=None,
):
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]
    orig_dtype = gemm_weights[0].dtype

    for gemm in gemm_weights:
        # gemm_weights are expected to be transposed
        assert gemm.shape[1] == act_scales.numel()

    if weight_scales is None:
        weight_scales = torch.cat(
            [gemm.abs().max(dim=0, keepdim=True)[0] for gemm in gemm_weights],
            dim=0)
        weight_scales = weight_scales.max(dim=0)[0]
    weight_scales.to(float).clamp(min=1e-5)
    scales = (act_scales.to(gemm_weights[0].device).to(float).pow(alpha) /
              weight_scales.pow(1 - alpha)).clamp(min=1e-5)

    apply_smoothing(scales, gemm_weights, norm_weights, norm_bias, orig_dtype)

    return scales


@torch.no_grad()
def capture_activation_range(
    model,
    tokenizer,
    dataset,
    num_samples=64,
    seq_len=512,
):

    model.eval()
    device = next(model.parameters()).device
    scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    def stat_tensor(name, tensor, key):
        tensor = tensor.view(-1, tensor.shape[-1]).detach()
        comming_max = tensor.abs().max(dim=0)[0].float()
        if scales[name][key] is None:
            scales[name][key] = comming_max
        else:
            scales[name][key] = torch.max(scales[name][key], comming_max)

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, "x")
        stat_tensor(name, y, "y")
        if scales[name]["w"] is None:
            scales[name]["w"] = m.weight.abs().clip(1e-8, None).max(dim=1)[0]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples), desc="Calibration"):
        input_ids = tokenizer(
            dataset[i]["article"],
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
        )
        model(input_ids.input_ids.to(device))

    for h in hooks:
        h.remove()

    return scales
