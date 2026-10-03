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

"""Shared weight-name helpers for llm-compressor ("compressed-tensors") checkpoints.

llm-compressor and ModelOpt export the same NVFP4 tensors under different names
and store the two FP32 global scalars in reciprocal conventions. Every weight
mapper that has to consume both producers needs the same translation, so it
lives here instead of being re-derived per model.
"""

import torch

# compressed-tensors NVFP4 ("nvfp4-pack-quantized") tensor suffix -> the
# ModelOpt-style suffix TRT-LLM's NVFP4 Linear / fused-MoE loaders read:
#     weight_packed        (uint8, two packed FP4 E2M1 values per byte)
#     weight_scale         (FP8 E4M3, per group of 16) -- identical, not renamed
#     weight_global_scale  (FP32, (FP8_MAX*FP4_MAX)/amax_weight)
#     input_global_scale   (FP32, (FP8_MAX*FP4_MAX)/amax_input)
# The per-group ``weight_scale`` is byte-identical between the producers and
# keeps its name; renaming it would collide with the per-channel FP8
# ``weight_scale`` that a mixed FP8/NVFP4 checkpoint stores on its FP8 modules.
_NVFP4_SUFFIX_MAP = {
    ".weight_packed": ".weight",
    ".weight_global_scale": ".weight_scale_2",
    ".input_global_scale": ".input_scale",
}

# The subset of the above whose values are reciprocals, not just renames.
_GLOBAL_SCALE_SUFFIXES = frozenset({".weight_global_scale", ".input_global_scale"})


def invert_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a compressed-tensors global scale to TRT-LLM's divisor form.

    llm-compressor stores ``global_scale = (FP8_MAX*FP4_MAX)/amax``; TRT-LLM's
    ``weight_scale_2`` / ``input_scale`` are ``amax/(FP8_MAX*FP4_MAX)``, i.e. the
    reciprocal.

    Some published checkpoints store ``0`` for experts that received no
    calibration tokens (``amax`` unavailable). A naive ``1/x`` turns those into
    ``+Inf``, and the fused-MoE loader reduces ``input_scale`` with a max over
    all experts in a layer, so a single ``Inf`` poisons every expert's scale and
    ``alpha`` in that layer -- silently, with no load error. Mapping ``0 -> 0``
    instead makes a dead expert a no-op in that max reduction and leaves its own
    ``alpha`` at 0.

    Args:
        tensor: The stored global scale.

    Returns:
        The reciprocal as contiguous float32, with non-positive inputs and any
        residual non-finite result mapped to 0.
    """
    tensor = tensor.to(torch.float32)
    inverted = torch.where(
        tensor > 0,
        1.0 / torch.clamp(tensor, min=torch.finfo(torch.float32).tiny),
        torch.zeros_like(tensor),
    )
    # Belt-and-braces: any residual Inf/NaN becomes 0.
    return torch.nan_to_num(inverted, nan=0.0, posinf=0.0, neginf=0.0).contiguous()


def normalize_compressed_tensors_nvfp4_names(
    weights: dict, *, allow_zero_global_scales: bool = False
) -> dict:
    """Rename compressed-tensors NVFP4 tensors to the names the loaders expect.

    Selection is by tensor suffix, which is what makes this safe on a
    mixed-precision checkpoint: the three renamed suffixes only ever appear on
    NVFP4 modules, while the FP8 modules' ``weight`` / ``weight_scale`` and the
    NVFP4 modules' per-group ``weight_scale`` are left alone. ModelOpt exports
    never carry these suffixes, so they pass through untouched (the dict is
    returned unchanged when nothing matched).

    Values are only materialized for the two global scalars, so lazily-loaded
    ``weight_packed`` slices stay lazy.

    Args:
        weights: Checkpoint tensors keyed by name. Values may be
            ``torch.Tensor`` or lazy safetensors slices.
        allow_zero_global_scales: Preserve non-positive/non-finite scale
            sentinels as zero. This is valid for uncalibrated MoE experts that
            a fused loader can leave inactive, but not for a dense Linear.

    Returns:
        The tensors with NVFP4 names normalized and global scales inverted.

    Raises:
        ValueError: A rename would overwrite a tensor the checkpoint already
            stores under the target name, or a dense global scale is invalid.
    """
    renamed: dict = {}
    changed = False
    for name, value in weights.items():
        new_name = name
        for old_suffix, new_suffix in _NVFP4_SUFFIX_MAP.items():
            if not name.endswith(old_suffix):
                continue
            new_name = name[: -len(old_suffix)] + new_suffix
            changed = True
            if old_suffix in _GLOBAL_SCALE_SUFFIXES:
                tensor = value[...] if not isinstance(value, torch.Tensor) else value
                if not allow_zero_global_scales and (
                    not torch.isfinite(tensor).all() or not (tensor > 0).all()
                ):
                    raise ValueError(
                        f"compressed-tensors tensor '{name}' contains a non-positive "
                        "or non-finite global scale; dense NVFP4 layers require "
                        "calibrated positive scales."
                    )
                value = invert_global_scale(tensor)
            break
        if new_name in renamed:
            raise ValueError(
                f"compressed-tensors rename '{name}' -> '{new_name}' collides with a "
                "tensor already present in the checkpoint."
            )
        renamed[new_name] = value
    return renamed if changed else weights
