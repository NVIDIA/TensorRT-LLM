# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gc
import re
from typing import Optional

import torch
from torch import nn

from visual_gen.utils import get_logger

from .attention import ditAttnProcessor
from .linear import ditLinear
from .norm import ditLayerNorm, ditRMSNorm

__all__ = ["ditAttnProcessor", "ditLinear", "apply_visual_gen_linear"]
logger = get_logger(__name__)


def apply_visual_gen_linear(
    model: nn.Module,
    load_parameters: bool = True,
    quantize_weights: bool = True,
    exclude_pattern: Optional[str] = None,
):
    exclude_pattern_ = re.compile(exclude_pattern) if exclude_pattern else None

    # Collect all linear modules to replace first to avoid modifying during iteration
    linear_modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (
            exclude_pattern_ is None or exclude_pattern_.match(name) is None
        ):
            linear_modules_to_replace.append(name)

    logger.info(f"Number of layers converted using visual_gen: {len(linear_modules_to_replace)}")

    # Replace linear modules
    for name in linear_modules_to_replace:
        parent = model
        attrs = name.split(".")
        for attr in attrs[:-1]:
            parent = getattr(parent, attr)
        module = getattr(parent, attrs[-1])
        linear = ditLinear.from_linear(module, load_parameters)
        # must under no_grad to avoid memory leak
        if quantize_weights:
            with torch.no_grad():
                linear.select_linear_impl()
        linear.name = name
        setattr(parent, attrs[-1], linear)
        del module

    gc.collect()
    torch.cuda.empty_cache()


def apply_visual_gen_norm(
    model: nn.Module, rmsnorm: list = [], layernorm: list = [], load_parameters: bool = False
):
    rmsnorm_modules_to_replace = []
    layernorm_modules_to_replace = []

    for name, module in model.named_modules():
        is_rmsnorm, is_layernorm = False, False
        suffix_name = name.split(".")[-1]
        for n in rmsnorm:
            if n == suffix_name:
                is_rmsnorm = True
                break
        for n in layernorm:
            if n == suffix_name:
                is_layernorm = True
                break
        assert not (is_rmsnorm and is_layernorm), f"name: {name} is both rmsnorm and layernorm"
        if isinstance(module, nn.RMSNorm) or is_rmsnorm:
            rmsnorm_modules_to_replace.append((name, module))
        if isinstance(module, nn.LayerNorm) or is_layernorm:
            layernorm_modules_to_replace.append((name, module))

    for name, module in rmsnorm_modules_to_replace:
        rmsnorm = ditRMSNorm.from_torch(module, load_parameters)
        rmsnorm.name = name
        parent = model
        attrs = name.split(".")
        for attr in attrs[:-1]:
            parent = getattr(parent, attr)
        setattr(parent, attrs[-1], rmsnorm)
        del module

    for name, module in layernorm_modules_to_replace:
        layernorm = ditLayerNorm.from_torch(module, load_parameters)
        layernorm.name = name
        parent = model
        attrs = name.split(".")
        for attr in attrs[:-1]:
            parent = getattr(parent, attr)
        setattr(parent, attrs[-1], layernorm)
        del module

    gc.collect()
    torch.cuda.empty_cache()
