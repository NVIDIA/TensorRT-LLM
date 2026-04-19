# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Per-model default generation parameters for Wan pipelines.

Deduction cascade: model version (2.1/2.2) → model size → model name.
Shared by WanPipeline (T2V) and WanImageToVideoPipeline (I2V).
"""

from typing import Dict

from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

# ---------------------------------------------------------------------------
# Constant tables
# ---------------------------------------------------------------------------

_WAN21_480P_PARAMS = {
    "height": 480,
    "width": 832,
    "num_inference_steps": 50,
    "guidance_scale": 5.0,
    "max_sequence_length": 512,
    "num_frames": 81,
    "frame_rate": 16.0,
}

_WAN21_720P_PARAMS = {
    "height": 720,
    "width": 1280,
    "num_inference_steps": 50,
    "guidance_scale": 5.0,
    "max_sequence_length": 512,
    "num_frames": 81,
    "frame_rate": 16.0,
}

_WAN22_PARAMS = {
    "height": 720,
    "width": 1280,
    "num_inference_steps": 40,
    "guidance_scale": 4.0,
    "max_sequence_length": 512,
    "num_frames": 81,
    "frame_rate": 16.0,
}

_WAN22_EXTRA_SPECS: Dict[str, ExtraParamSchema] = {
    "guidance_scale_2": ExtraParamSchema(
        type="float",
        default=None,
        description="Second guidance scale for Wan 2.2 two-stage denoising.",
    ),
    "boundary_ratio": ExtraParamSchema(
        type="float",
        default=None,
        range=(0.0, 1.0),
        description="Timestep boundary ratio for switching guidance scales (Wan 2.2).",
    ),
}


# ---------------------------------------------------------------------------
# Deduction helpers
# ---------------------------------------------------------------------------


def _is_480p_model(name_or_path: str, num_heads: int, is_wan22: bool) -> bool:
    """Deduce whether the model's native resolution is 480p.

    Cascade:
    1. Wan 2.2 models are always 720p.
    2. Small models (≤12 heads, e.g. 1.3B) are 480p.
    3. Check model name for explicit "480p" (e.g. Wan2.1-I2V-14B-480P).
    4. Default to 720p (14B T2V, 14B-720P I2V).
    """
    if is_wan22:
        return False
    if num_heads <= 12:
        return True
    if "480p" in name_or_path.lower():
        return True
    return False


def get_wan_default_params(
    is_wan22: bool,
    name_or_path: str,
    num_heads: int,
    *,
    include_i2v: bool = False,
) -> dict:
    """Return the default generation params dict for a Wan model.

    Args:
        is_wan22: Whether this is a Wan 2.2 model (has boundary_ratio).
        name_or_path: Checkpoint path or HF model ID (_name_or_path).
        num_heads: Number of attention heads from transformer config.
        include_i2v: If True, add I2V-specific defaults (image_cond_strength).
    """
    if is_wan22:
        params = dict(_WAN22_PARAMS)
    elif _is_480p_model(name_or_path, num_heads, is_wan22):
        params = dict(_WAN21_480P_PARAMS)
    else:
        params = dict(_WAN21_720P_PARAMS)

    if include_i2v:
        params["image_cond_strength"] = 1.0

    return params


def get_wan_extra_param_specs(is_wan22: bool) -> Dict[str, ExtraParamSchema]:
    """Return extra_param_specs for a Wan model.

    Wan 2.2 exposes guidance_scale_2 and boundary_ratio.
    Wan 2.1 has no model-specific extra params.
    """
    if is_wan22:
        return dict(_WAN22_EXTRA_SPECS)
    return {}
