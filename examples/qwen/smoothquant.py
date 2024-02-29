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
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_dir)
from utils.utils import make_context


@torch.no_grad()
def apply_smoothing(scales,
                    gemm_weights,
                    rmsnorm_weights=None,
                    dtype=torch.float32,
                    rmsnorm_1p=False):
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]

    if rmsnorm_weights is not None:
        assert rmsnorm_weights.numel() == scales.numel()
        rmsnorm_weights.div_(scales).to(dtype)
    if rmsnorm_1p:
        rmsnorm_weights += (1 / scales) - 1

    for gemm in gemm_weights:
        gemm.mul_(scales.view(1, -1)).to(dtype)


@torch.no_grad()
def smooth_gemm(gemm_weights,
                act_scales,
                rmsnorm_weights=None,
                alpha=0.5,
                weight_scales=None):
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

    apply_smoothing(scales, gemm_weights, rmsnorm_weights, orig_dtype)

    return scales


@torch.no_grad()
def smooth_gemm_mlp(w1_weights,
                    w2_weights,
                    act_scales,
                    rmsnorm_weights=None,
                    alpha=0.5,
                    weight_scales=None):
    gemm_weights = []
    if not isinstance(w1_weights, list):
        w1_weights = [w1_weights]
    if not isinstance(w2_weights, list):
        w2_weights = [w2_weights]

    for i in range(len(w1_weights)):
        gemm_weight = torch.cat([w1_weights[i], w2_weights[i]], dim=0)
        gemm_weights.append(gemm_weight)

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

    apply_smoothing(scales, w1_weights + w2_weights, rmsnorm_weights,
                    orig_dtype)

    return scales


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (act_scales.pow(alpha) /
              weight_scales.pow(1 - alpha)).clamp(min=1e-5).to(device).to(dtype)

    if ln is not None:
        ln.weight.div_(scales)
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
    return scales


@torch.no_grad()
def capture_activation_range(
    model,
    tokenizer,
    dataset,
    system_prompt,
    chat_format,
    max_input_len,
    num_samples=512,
):
    model.eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    def stat_tensor(name, tensor, act_scales, key):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float()

        if act_scales[name][key] is None:
            act_scales[name][key] = comming_max
        else:
            act_scales[name][key] = torch.max(act_scales[name][key],
                                              comming_max)

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, act_scales, "x")
        stat_tensor(name, y, act_scales, "y")

        if act_scales[name]["w"] is None:
            act_scales[name]["w"] = m.weight.abs().clip(1e-8,
                                                        None).max(dim=1)[0]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, Conv1D):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))
    num_samples = min(num_samples, len(dataset))
    for i in tqdm(range(num_samples), desc="calibrating model"):
        line = dataset[i]["article"]
        line = line + ' TL;DR: '
        line = line.strip()
        line = line.replace(" n't", "n't")
        # use make_content to generate prompt
        _, input_id_list = make_context(tokenizer=tokenizer,
                                        query=line,
                                        history=[],
                                        system=system_prompt,
                                        chat_format=chat_format,
                                        max_input_length=max_input_len)
        line_encoded = torch.from_numpy(np.array(
            input_id_list, dtype=np.int32)).type(torch.int32).unsqueeze(0)
        line_encoded = line_encoded.to(device)
        model(line_encoded)

    for h in hooks:
        h.remove()

    return act_scales
