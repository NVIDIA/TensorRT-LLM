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

# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Shape contracts for automatic selective policy selection."""

MODEL_CONFIGS = {
    "DS-V3.2": (64, 128, 2048, 1),
    "DS-V4-Flash": (64, 128, 512, 4),
    "DS-V4-Pro": (64, 128, 1024, 4),
    "GLM-5.3-Flash": (32, 128, 512, 4),
    "GLM-5.3": (32, 128, 2048, 1),
    "Qwen3.8-Flash-Next": (4, 128, 512, 4),
}


def resolve_model_config(heads, head_dim, top_k, compression_ratio, model_config=None):
    """Infer a policy; an optional name asserts the same configuration.

    Unrecognized configurations retain the general numerical contract. In
    particular, H4 inputs do not imply Qwen's constant weights.
    Qwen's numerical constraints apply only to an explicitly named model.
    """
    config = (heads, head_dim, top_k, compression_ratio)
    inferred = next(
        (name for name, value in MODEL_CONFIGS.items() if value == config), None
    )
    if model_config is not None and (
        not isinstance(model_config, str)
        or model_config not in MODEL_CONFIGS
        or model_config != inferred
    ):
        raise ValueError(
            "cutedsl model_config must match num_heads, head_dim, top_k and compression_ratio"
        )
    if inferred == "Qwen3.8-Flash-Next" and model_config is None:
        return None
    return inferred
