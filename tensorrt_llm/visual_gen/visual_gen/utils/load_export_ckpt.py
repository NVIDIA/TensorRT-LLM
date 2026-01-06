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

import json
import os

import torch
from diffusers.models.modeling_utils import ModelMixin
from safetensors.torch import save_file

from visual_gen.layers.linear import ditLinear
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


def save_safetensors(model, export_path: str):
    """Export model checkpoint as safetensors with automatic splitting if > 5GB"""

    # Create export directory
    os.makedirs(export_path, exist_ok=True)

    # Get model state dict
    state_dict = model.state_dict()

    # Calculate tensor sizes in bytes
    MAX_SHARD_SIZE = 5 * 1024**3  # 5GB in bytes
    tensor_info = {}

    for name, tensor in state_dict.items():
        # Calculate size: num_elements * bytes_per_element
        size_bytes = tensor.numel() * tensor.element_size()
        tensor_info[name] = {"tensor": tensor, "size_bytes": size_bytes}

    # Group tensors into shards
    shards = []
    current_shard = {}
    current_shard_size = 0

    for name, info in tensor_info.items():
        tensor_size = info["size_bytes"]

        # If adding this tensor would exceed the limit, start a new shard
        if current_shard and (current_shard_size + tensor_size) > MAX_SHARD_SIZE:
            shards.append(current_shard)
            current_shard = {}
            current_shard_size = 0

        current_shard[name] = info["tensor"]
        current_shard_size += tensor_size

    # Add the last shard if it has any tensors
    if current_shard:
        shards.append(current_shard)

    # Save each shard as a separate safetensors file
    total_shards = len(shards)
    index_dict = {"metadata": {"total_size": sum(info["size_bytes"] for info in tensor_info.values())}}
    weight_map = {}

    for shard_idx, shard in enumerate(shards):
        if total_shards == 1:
            # Single file case
            filename = "diffusion_pytorch_model.safetensors"
        else:
            # Multiple files case - use same naming convention as Qwen-Image
            filename = f"diffusion_pytorch_model-{shard_idx+1:05d}-of-{total_shards:05d}.safetensors"

        filepath = os.path.join(export_path, filename)

        # Save shard
        save_file(shard, filepath)

        # Update weight map
        for tensor_name in shard.keys():
            weight_map[tensor_name] = filename

        # Calculate and log shard size
        shard_size_gb = sum(tensor_info[name]["size_bytes"] for name in shard.keys()) / (1024**3)
        logger.info(f"Saved shard {shard_idx+1}/{total_shards}: {filename} ({shard_size_gb:.2f} GB)")

    # Create index file
    index_dict["weight_map"] = weight_map
    index_filename = "diffusion_pytorch_model.safetensors.index.json"
    index_path = os.path.join(export_path, index_filename)

    with open(index_path, "w") as f:
        json.dump(index_dict, f, indent=2)

    logger.info(f"Saved index file: {index_path}")

    # Save model config if available
    if hasattr(model, "config"):
        config_path = os.path.join(export_path, "config.json")
        if hasattr(model.config, "to_dict"):
            config_dict = model.config.to_dict()
        else:
            config_dict = dict(model.config) if hasattr(model.config, "__dict__") else {}

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Saved model config: {config_path}")

    logger.info(f"Model checkpoint exported to {export_path} ({total_shards} shard{'s' if total_shards > 1 else ''})")


def export_quantized_checkpoint(model: torch.nn.Module, path: str):
    for module in model.modules():
        if isinstance(module, ditLinear):
            module.select_linear_impl()
    save_safetensors(model, path)


def load_quantized_checkpoint(model_cls, path: str, torch_dtype: torch.dtype = None):
    os.environ["LOADING_QUANT_CHECKPOINT"] = "True"
    if issubclass(model_cls, ModelMixin):
        # note: don't specify torch_dtype here, so that fp8 dtype can be inferred from the checkpoint
        model = model_cls.from_pretrained(path)
        keep_in_fp32_modules = []
        if hasattr(model, "_keep_in_fp32_modules"):
            keep_in_fp32_modules = model._keep_in_fp32_modules
        # convert parameters to the specified dtype if it is not fp8 or in the keep_in_fp32_modules
        for name, param in model.named_parameters():
            keep_in_fp32 = any(m in name for m in keep_in_fp32_modules)
            if param.dtype != torch.float8_e4m3fn and not keep_in_fp32:
                param.data = param.data.to(torch_dtype)
        return model
    else:
        raise NotImplementedError(f"Model {type(model)} is not supported")
    os.environ["LOADING_QUANT_CHECKPOINT"] = "False"
