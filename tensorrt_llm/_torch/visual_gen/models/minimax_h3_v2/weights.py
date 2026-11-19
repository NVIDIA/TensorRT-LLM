# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

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

"""Checkpoint assembly for the packed H3 transformer."""

import torch
from tqdm import tqdm

from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel

from .components.linear import Linear


def remap_feed_forward_weights(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for key, value in weights.items():
        if ".ff.net.0.proj." in key:
            prefix, suffix = key.split("net.0.proj.", maxsplit=1)
            # The checkpoint packs up before gate.
            up, gate = value.chunk(2, dim=0)
            remapped[f"{prefix}gate.{suffix}"] = gate
            remapped[f"{prefix}up.{suffix}"] = up
        else:
            remapped[key.replace(".ff.net.2.", ".ff.down_proj.")] = value
    return remapped


@torch.no_grad()
def load_transformer_weights(
    model: BaseDiffusionModel, checkpoint: dict[str, torch.Tensor]
) -> None:
    weights = remap_feed_forward_weights(checkpoint)
    device = next(model.parameters()).device
    missing = []
    for name, module in tqdm(model.named_modules(), desc="Loading MiniMax-H3 weights"):
        if isinstance(module, Linear):
            module.create_weights()
            module.to(device)
            missing.extend(module.load_checkpoint(name, weights, model_config=model.model_config))
            continue
        for key, param in module.named_parameters(recurse=False):
            target = f"{name}.{key}"
            if target in weights:
                param.copy_(weights[target].to(param.dtype))
            else:
                missing.append(target)
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(
            f"MiniMax-H3 checkpoint is missing {len(missing)} required tensors: {preview}{suffix}"
        )
